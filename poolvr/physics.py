import logging
from collections import deque
import numpy as np


_logger = logging.getLogger(__name__)


INCH2METER = 0.0254


class PoolPhysics(object):

    class StrikeBallEvent(object):
        def __init__(self, t, i, Q, V, cue_mass):
            self.t = t
            self.i = i
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass

    class SlideToRollEvent(object):
        def __init__(self, t, i):
            self.t = t
            self.i = i

    class RollToRestEvent(object):
        def __init__(self, t, i):
            self.t = t
            self.i = i

    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.2,
                 e=0.93,
                 g=9.81,
                 **kwargs):
        self.num_balls = num_balls
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.e = e
        self.g = g
        self.t = 0.0
        self.events = deque()
        self.nevent = 0
        # state of balls:
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float32)
        self._b = np.zeros((num_balls, 3, 2), dtype=np.float32)
        self.on_table = np.array(self.num_balls * [True])
        self.is_sliding = np.array(self.num_balls * [False])
        self.is_rolling = np.array(self.num_balls * [False])
    def step(self, dt):
        lt = self.t
        self.t += dt
    def strike_ball(self, t, i, cue_mass, q, Q, V, omega):
        a, c, b = Q
        V_xz = V[::2]
        norm_V = np.linalg.norm(V)
        norm_V_xz = np.linalg.norm(V_xz)
        # sin, cos = abs(V_xz) / norm_V_xz
        sin, cos = 0.0, 1.0
        M, m, R = cue_mass, self.ball_mass, self.ball_radius
        F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
        norm_v = -F / m * cos
        v = self._a[i,:,1] # post-impact ball velocity
        v[0] = norm_v * V[0] / norm_V_xz
        v[2] = norm_v * V[2] / norm_V_xz
        v[1] = 0 # TODO
        I = self._I
        omega_x = F * (-c * sin + b * cos) / I
        omega_z = F * a * sin / I
        omega = self._b[i,:,0] # post-impact ball angular velocity
        omega[0] = omega_x * V[0] / norm_V_xz
        omega[2] = omega_z * V[2] / norm_V_xz
        omega[1] = 0
        # TODO: omega[1] = -F * a * cos / I
        u = v + R * np.array((-omega[2], 0.0, omega[0]), dtype=np.float32) # relative velocity
        mu_s, mu_sp, g = self.mu_s, self.mu_sp, self.g
        self._a[i,::2,2] = -0.5 * mu_s * g * u[::2]
        self._b[i,::2,1] = -5 * mu_s * g / (2 * R) * np.array((-u[2], u[0]), dtype=np.float32)
        self._b[i,1,1] = -5 * mu_sp * g / (2 * R)
        tau_s = 2 * np.linalg.norm(u) / (7 * mu_s * g) # duration of sliding state
        event = self.StrikeBallEvent(t, i, Q, V, cue_mass)
        self.nevent += 1
    def eval_positions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        lt = self.events[-1].t
        tarray = np.array((1.0, t - lt, (t - lt)**2), dtype=np.float32)
        return self._a.dot(tarray, out=out)
    def eval_quaternions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 4), dtype=np.float32)
        raise Exception('TODO')
