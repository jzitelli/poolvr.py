"""
Open Dynamics Engine-based pool physics simulator
"""
import logging
import numpy as np
import ode


from .table import PoolTable
from .physics.events import BallRestEvent, CueStrikeEvent
try:
    from .sound import play_ball_ball_collision_sound
except Exception:
    play_ball_ball_collision_sound = None


INCH2METER = 0.0254
ZERO3 = np.zeros(3, dtype=np.float64)
_J = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_logger = logging.getLogger(__name__)


class ODEPoolPhysics(object):
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
                 balls_on_table=None,
                 initial_positions=None,
                 table=None,
                 **kwargs):
        if table is None:
            table = PoolTable(num_balls=num_balls, ball_radius=ball_radius)
        self.table = table
        self.num_balls = num_balls
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.ball_I = 2/5 * ball_mass * ball_radius**2 # moment of inertia
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.mu_b = mu_b
        self.c_b = c_b
        self.E_Y_b = E_Y_b
        self.g = g
        self.t = 0.0
        self._on_table = np.array(num_balls * [False])
        self._balls_on_table = balls_on_table
        self._on_table[np.array(balls_on_table)] = True
        self.world = ode.World()
        self.world.setGravity((0.0, -g, 0.0))
        self.world.setLinearDamping(linear_damping)
        self.world.setAngularDamping(angular_damping)
        self.space = ode.SimpleSpace()
        self.ball_bodies = []
        self.ball_geoms = []
        for i in range(num_balls):
            body, geom = self._create_ball(self.world, ball_mass, ball_radius, space=self.space)
            self.ball_bodies.append(body)
            self.ball_geoms.append(geom)
        self.table_geom = ode.GeomPlane(space=self.space, normal=(0.0, 1.0, 0.0), dist=self.table.height)
        # self.table_geom = ode.GeomBox(space=self.space, lengths=(self.table.width, self.table.height, self.table.length))
        self._contactgroup = ode.JointGroup()
        self._on_cue_ball_collide = None
        if initial_positions is None:
            initial_positions = self.table.calc_racked_positions()
        self.cues = []
        self.cue_bodies = []
        self.cue_geoms = []
        self.reset(ball_positions=initial_positions, balls_on_table=balls_on_table)

    def reset(self, ball_positions=None, balls_on_table=None):
        """
        Reset the state of the balls to at rest, at the specified positions.
        """
        from itertools import chain
        self.t = 0
        self._ball_motion_events = {}
        if ball_positions is None:
            ball_positions = self.table.calc_racked_positions()
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.balls_on_table = balls_on_table
        self.ball_events = {i: [BallRestEvent(self.t, i, r=ball_positions[i])]
                            for i in balls_on_table}
        self.events = list(chain.from_iterable(self.ball_events.values()))
        self._t_last_strike = 0.0
        self.nsteps = 0
        for body, geom, position in zip(self.ball_bodies, self.ball_geoms, ball_positions):
            body.enable()
            geom.setPosition(position)
            body.setPosition(position)
            body.setLinearVel(ZERO3)
            body.setAngularVel(ZERO3)
        self._contactgroup.empty()

    @property
    def balls_on_table(self):
        return set(self._balls_on_table)
    @balls_on_table.setter
    def balls_on_table(self, balls):
        self._balls_on_table = set(balls)
        self._on_table[:] = False
        self._on_table[np.array(balls)] = True

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def add_cue(self, cue):
        body, geom = self._create_cue(self.world, cue.mass, cue.radius, cue.length,
                                      space=self.space, kinematic=True)
        self.cues = [cue]
        self.cue_bodies = [body]
        self.cue_geoms = [geom]
        return body, geom

    def strike_ball(self, t, i, r_i, r_c, V, cue_mass):
        if not self._on_table[i]:
            return
        body = self.ball_bodies[i]
        Q = r_c - r_i
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
        I = self.ball_I
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
        self._add_event(CueStrikeEvent(t, i, r_i, Q + r_i, V, cue_mass))
        self._add_event(BallRestEvent(t + 12, i, r_0=r_i))
        return 1

    def step(self, dt):
        self.space.collide((self.world, self._contactgroup), self._near_callback)
        self.world.step(dt)
        self._contactgroup.empty()
        self.t += dt
        for cue, body, geom in zip(self.cues, self.cue_bodies, self.cue_geoms):
            body.setPosition(cue.world_position)
            w = cue.quaternion[3]; cue.quaternion[1:] = cue.quaternion[:3]; cue.quaternion[0] = w
            body.setQuaternion(cue.quaternion)
            geom.setQuaternion(cue.quaternion)
            body.setLinearVel(cue.velocity)
            body.setAngularVel(cue.angular_velocity)
        self.nsteps += 1

    def eval_positions(self, t, balls=None, out=None):
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getPosition()
        return out

    def eval_quaternions(self, t, balls=None, out=None):
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getQuaternion()
        return out

    def eval_velocities(self, t, balls=None, out=None):
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getLinearVel()
        return out

    def eval_angular_velocities(self, t, balls=None, out=None):
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            out[ii] = self.ball_bodies[i].getAngularVel()
        return out

    def next_turn_time(self):
        return self._t_last_strike + 0.04

    @property
    def cushion_meshes(self):
        if self._cushion_meshes is None:
            self._cushion_meshes = self._setup_cushion_meshes()

    def _setup_cushion_meshes(self):
        (headCushionGeom, leftHeadCushionGeom,
         rightHeadCushionGeom, footCushionGeom,
         leftFootCushionGeom, rightFootCushionGeom) = self.cushion_geoms
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.headCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.headCushionGeom.indices.reshape(-1,3))
        self.head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.leftHeadCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.leftHeadCushionGeom.indices.reshape(-1,3))
        self.left_head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.rightHeadCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.rightHeadCushionGeom.indices.reshape(-1,3))
        self.right_head_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.footCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.footCushionGeom.indices.reshape(-1,3)[:,::-1])
        self.foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.leftFootCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.leftFootCushionGeom.indices.reshape(-1,3)[:,::-1])
        self.left_foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)
        tri_mesh_data = ode.TriMeshData()
        tri_mesh_data.build(self.table.rightFootCushionGeom.attributes['vertices'].reshape(-1,3).tolist(),
                            self.table.rightFootCushionGeom.indices.reshape(-1,3)[:,::-1])
        self.right_foot_cushion_geom = ode.GeomTriMesh(tri_mesh_data, space=self.space)

    @staticmethod
    def _create_ball(world, ball_mass, ball_radius, space=None):
        mass = ode.Mass()
        mass.setSphereTotal(ball_mass, ball_radius)
        body = ode.Body(world)
        body.setMass(mass)
        body.shape = "sphere"
        body.boxsize = (2*ball_radius, 2*ball_radius, 2*ball_radius)
        geom = ode.GeomSphere(space=space, radius=ball_radius)
        geom.setBody(body)
        return body, geom

    @staticmethod
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

    def _add_event(self, event):
        self.events.append(event)
        self.ball_events[event.i].append(event)

    def _near_callback(self, args, geom1, geom2):
        world, contactgroup = args
        try:
            i = self.ball_geoms.index(geom1)
            if not self._on_table[i]:
                return
        except Exception:
            pass
        try:
            j = self.ball_geoms.index(geom2)
            if not self._on_table[j]:
                return
        except Exception:
            pass
        contacts = ode.collide(geom1, geom2)
        if contacts:
            body1, body2 = geom1.getBody(), geom2.getBody()
        for c in contacts:
            if isinstance(geom1, ode.GeomPlane) or isinstance(geom2, ode.GeomPlane):
                # ball-table contact
                c.setBounce(0.24)
                c.setMu(0.16)
                c.setBounceVel(0.033)
                c.setSoftERP(0.4)
                c.setSoftCFM(1e4)
                c.setSlip1(0.05)
            elif isinstance(geom1, ode.GeomTriMesh) or isinstance(geom2, ode.GeomTriMesh):
                # ball-cushion contact
                c.setBounce(0.86)
                c.setMu(0.16)
                c.setBounceVel(0.02)
                c.setSoftERP(0.4)
                c.setSoftCFM(1e2)
                c.setSlip1(0.04)
            elif isinstance(geom1, ode.GeomCylinder) or isinstance(geom2, ode.GeomCylinder):
                # cue-ball contact
                c.setBounce(0.69)
                c.setMu(0.25)
                c.setSoftERP(0.2)
                c.setSoftCFM(1e1)
                pos, normal, depth, g1, g2 = c.getContactGeomParams()
                v_n = np.array(normal).dot(np.array(body1.getPointVel(pos)) - np.array(body2.getPointVel(pos)))
                if self._on_cue_ball_collide:
                    self._on_cue_ball_collide(impact_speed=abs(v_n))
            else:
                # ball-ball contact
                c.setBounce(0.93)
                c.setMu(0.13)
                if play_ball_ball_collision_sound is not None:
                    pos, normal, depth, g1, g2 = c.getContactGeomParams()
                    v_n = abs(np.array(normal).dot(np.array(body1.getLinearVel()) - np.array(body2.getLinearVel())))
                    vol = max(0.02, min(0.8, 0.45*v_n + 0.55*v_n**2))
                    if vol > 0.02:
                        play_ball_ball_collision_sound(vol=vol)
            j = ode.ContactJoint(world, contactgroup, c)
            j.attach(body1, body2)
