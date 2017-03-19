"""
Open Dynamic Engine-based pool physics simulator
"""
import logging
import numpy as np
import ode

from .exceptions import TODO


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254

ZERO3 = np.zeros(3, dtype=np.float64)


def create_ball(mass, radius, world, space=None):
    body = ode.Body(world)
    mass = ode.Mass()
    mass.setSphereTotal(mass, radius)
    body.setMass(mass)
    body.shape = "sphere"
    body.boxsize = (2*radius, 2*radius, 2*radius)
    # Create a box geom for collision detection
    geom = ode.GeomSphere(space=space, radius=radius)
    geom.setBody(body)
    return body, geom


def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do.
    """
    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)
    # Create contact joints
    world, contactgroup = args
    for c in contacts:
        c.setBounce(0.93)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())


class ODEPoolPhysics(object):
    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 g=9.81,
                 linear_damping=0.06,
                 angular_damping=0.1,
                 initial_positions=None,
                 table_height=0.77,
                 **kwargs):
        self.num_balls = num_balls
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
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
        self.nsteps = 0
        self.ball_bodies = []
        self.ball_geoms = []
        for i in range(num_balls):
            body, geom = create_ball(self.world, self.space, ball_mass, ball_radius)
            self.ball_bodies.append(body)
            self.ball_geoms.append(geom)
        if initial_positions is not None:
            for body, position in zip(self.ball_bodies, initial_positions):
                body.setPosition(position)
        self.table_surface = ode.GeomPlane(self.space,
                                           (0.0, table_height, 0.0), 0)
        self._contactgroup = ode.JointGroup()

    def reset(self, ball_positions):
        self.nsteps = 0
        self.t = 0.0
        self.on_table[:] = True
        for i, body in enumerate(self.ball_bodies):
            body.enable()
            body.setPosition(ball_positions[i])
            body.setLinearVel(ZERO3)
            body.setAngularVel(ZERO3)

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
        body.setVelocity(*V)

    def step(self, dt):
        self.space.collide((self.world, self._contactgroup), near_callback)
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

