"""
Open Dynamics Engine-based pool physics simulator
"""
import logging
import numpy as np
import ode


from .table import PoolTable
from .physics import _J, PoolPhysics
from .sound import play_ball_ball_collision_sound


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254

ZERO3 = np.zeros(3, dtype=np.float64)


def _create_ball(world, ball_mass, ball_radius, space=None):
    body = ode.Body(world)
    mass = ode.Mass()
    mass.setSphereTotal(ball_mass, ball_radius)
    body.setMass(mass)
    body.shape = "sphere"
    body.boxsize = (2*ball_radius, 2*ball_radius, 2*ball_radius)
    geom = ode.GeomSphere(space=space, radius=ball_radius)
    geom.setBody(body)
    return body, geom


def _create_cue(world, cue_mass, cue_radius, cue_length, space=None, kinematic=True):
        body = ode.Body(world)
        mass = ode.Mass()
        mass.setCylinderTotal(cue_mass, 3, cue_radius, cue_length)
        body.setMass(mass)
        body.shape = "cylinder"
        if kinematic:
            body.setKinematic()
        if space:
            geom = ode.GeomCylinder(space=space, radius=cue_radius, length=cue_length)
            geom.setBody(body)
            return body, geom
        return body


class ODEPoolPhysics(object):
    PhysicsEvent = PoolPhysics.PhysicsEvent
    StrikeBallEvent = PoolPhysics.StrikeBallEvent
    SlideToRollEvent = PoolPhysics.SlideToRollEvent
    RestEvent = PoolPhysics.RestEvent
    SlideToRestEvent = PoolPhysics.SlideToRestEvent
    RollToRestEvent = PoolPhysics.RollToRestEvent
    PositionBallEvent = PoolPhysics.PositionBallEvent
    SimpleBallCollisionEvent = PoolPhysics.SimpleBallCollisionEvent
    BallCollisionEvent = PoolPhysics.BallCollisionEvent

    STATIONARY = 0
    SLIDING    = 1
    ROLLING    = 2
    SPINNING   = 3

    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.2,
                 mu_b=0.06,
                 c_b=4000.0,
                 E_Y_b=2.4e9,
                 g=9.81,
                 linear_damping=0.008,
                 angular_damping=0.0125,
                 initial_positions=None,
                 table=None,
                 **kwargs):
        self.PhysicsEvent.physics = self
        if table is None:
            table = PoolTable(**kwargs)
        self.table = table
        self.num_balls = num_balls
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.mu_b = mu_b
        self.c_b = c_b
        self.E_Y_b = E_Y_b
        self.g = g
        self.all_balls = list(range(num_balls))
        self.on_table = np.array(num_balls * [True])
        self.events = []
        self.ball_events = {i: [] for i in self.all_balls}
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self.world = ode.World()
        self.world.setGravity((0.0, -g, 0.0))
        self.world.setLinearDamping(linear_damping)
        self.world.setAngularDamping(angular_damping)
        self.space = ode.SimpleSpace()
        self.t = 0.0
        self._t_last_strike = 0.0
        self.nsteps = 0
        self.ball_bodies = []
        self.ball_geoms = []
        for i in range(num_balls):
            body, geom = _create_ball(self.world, ball_mass, ball_radius, space=self.space)
            self.ball_bodies.append(body)
            self.ball_geoms.append(geom)
        # self.table_geom = ode.GeomBox(space=self.space, lengths=(self.table.width, self.table.height, self.table.length))
        self.table_geom = ode.GeomPlane(space=self.space, normal=(0.0, 1.0, 0.0),
                                        dist=self.table.height)
        def _setup_cushions():
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.headCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.headCushionGeom.indices.reshape(-1,3))
            self.head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.leftHeadCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.leftHeadCushionGeom.indices.reshape(-1,3))
            self.left_head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.rightHeadCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.rightHeadCushionGeom.indices.reshape(-1,3))
            self.right_head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.footCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.footCushionGeom.indices.reshape(-1,3)[:,::-1])
            self.foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.leftFootCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.leftFootCushionGeom.indices.reshape(-1,3)[:,::-1])
            self.left_foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
            tri_mesh_data = ode.TriMeshData()
            tri_mesh_data.build(self.table.rightFootCushionGeom.attributes['vertices'].reshape(-1,3).tolist(), self.table.rightFootCushionGeom.indices.reshape(-1,3)[:,::-1])
            self.right_foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        _setup_cushions()

        self._contactgroup = ode.JointGroup()
        self.events = []
        self.ball_events = {i: [] for i in self.all_balls}
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float64)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float64)
        if initial_positions is not None:
            for body, position in zip(self.ball_bodies, initial_positions):
                body.setPosition(position)
            for geom, position in zip(self.ball_geoms, initial_positions):
                geom.setPosition(position)
            self._a[:,0] = initial_positions
        self._on_cue_ball_collide = None

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def reset(self, ball_positions):
        self.on_table[:] = True
        for i, body in enumerate(self.ball_bodies):
            body.enable()
            body.setPosition(ball_positions[i])
            body.setLinearVel(ZERO3)
            body.setAngularVel(ZERO3)
        for i, geom in enumerate(self.ball_geoms):
            geom.setPosition(ball_positions[i])
        self._contactgroup.empty()
        self.events = []
        self.ball_events = {i: [] for i in self.all_balls}
        self._a[:] = 0
        self._a[:,0] = ball_positions

    def add_cue(self, cue):
        body, geom = _create_cue(self.world, cue.mass, cue.radius, cue.length,
                                 space=self.space, kinematic=True)
        self.cue_bodies = [body]
        self.cue_geoms = [geom]
        return body, geom

    def strike_ball(self, t, i, Q, V, cue_mass):
        if not self.on_table[i]:
            return
        body = self.ball_bodies[i]
        a, b, c = Q
        V = V.copy()
        V[1] = 0
        sin, cos = 0.0, 1.0
        M = cue_mass
        m, R = self.ball_mass, self.ball_radius
        norm_V = np.linalg.norm(V)
        F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
        v = np.zeros(3, dtype=np.float64) # <-- post-impact ball velocity
        v[::2] = F / m * V[::2] / np.linalg.norm(V[::2])
        omega = np.zeros(3, dtype=np.float64) # <-- post-impact ball angular velocity
        I = self._I
        omega_i = F * (-c * sin + b * cos) / I
        omega_j = F * a * sin / I
        omega_k = -F * a * cos / I
        _j = -V[:] / norm_V
        _k = _J
        _i = np.cross(_j, _k)
        omega[:] = omega_i * _i + omega_j * _j + omega_k * _k
        body.setLinearVel(v)
        body.setAngularVel(omega)
        self._t_last_strike = t
        self._add_event(self.StrikeBallEvent(t, i, Q, V, cue_mass))
        self._add_event(self.RollToRestEvent(t + 12, i, self._a[i,0]))
        return 1

    def step(self, dt):
        self.space.collide((self.world, self._contactgroup), self._near_callback)
        self.world.step(dt)
        self._contactgroup.empty()
        self.t += dt
        self.nsteps += 1

    def eval_positions(self, t, balls=None, out=None):
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getPosition()
        return out

    def eval_quaternions(self, t, balls=None, out=None):
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getQuaternion()
        return out

    def eval_velocities(self, t, balls=None, out=None):
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getLinearVel()
        return out

    def eval_angular_velocities(self, t, balls=None, out=None):
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getAngularVel()
        return out

    def next_turn_time(self):
        return self._t_last_strike + 0.04

    def _add_event(self, event):
        self.events.append(event)
        self.ball_events[event.i].append(event)

    def _near_callback(self, args, geom1, geom2):
        world, contactgroup = args
        try:
            i = self.ball_geoms.index(geom1)
            if not self.on_table[i]:
                return
        except:
            pass
        try:
            j = self.ball_geoms.index(geom2)
            if not self.on_table[j]:
                return
        except:
            pass
        contacts = ode.collide(geom1, geom2)
        if contacts:
            body1, body2 = geom1.getBody(), geom2.getBody()
        for c in contacts:
            if isinstance(geom1, ode.GeomPlane) or isinstance(geom2, ode.GeomPlane):
                c.setBounce(0.22)
                c.setMu(0.15)
                c.setBounceVel(0.01)
                c.setSoftERP(0.5)
                c.setSoftCFM(1e4)
                c.setSlip1(0.03)
            elif isinstance(geom1, ode.GeomTriMesh) or isinstance(geom2, ode.GeomTriMesh):
                c.setBounce(0.83)
                c.setMu(0.15)
                c.setBounceVel(0.07)
                c.setSoftERP(0.4)
                c.setSoftCFM(1e4)
                c.setSlip1(0.03)
            elif isinstance(geom1, ode.GeomCylinder) or isinstance(geom2, ode.GeomCylinder):
                c.setBounce(0.66)
                c.setMu(0.25)
                pos, normal, depth, g1, g2 = c.getContactGeomParams()
                v_n = np.array(normal).dot(np.array(body1.getPointVel(pos)) - np.array(body2.getPointVel(pos)))
                if self._on_cue_ball_collide:
                    self._on_cue_ball_collide(impact_speed=abs(v_n))
            else:
                c.setBounce(0.93)
                c.setMu(0.06)
                pos, normal, depth, g1, g2 = c.getContactGeomParams()
                v_n = np.array(normal).dot(np.array(body1.getLinearVel()) - np.array(body2.getLinearVel()))
                # if abs(v_n) > 0.07:
                #     #play_ball_ball_collision_sound(vol=(v_n**2 / 5))
                #     play_ball_ball_collision_sound(vol=max(0.07, min(0.85, v_n**2 / 7)))
            j = ode.ContactJoint(world, contactgroup, c)
            j.attach(body1, body2)
