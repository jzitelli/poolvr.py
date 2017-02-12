import logging
from collections import deque
import numpy as np


_logger = logging.getLogger(__name__)


from .exceptions import TODO


INCH2METER = 0.0254


class PoolPhysics(object):

    class Event(object):
        def __init__(self, t):
            self.t = t
        def project_state(self, t, dof,
                          positions=None, quaternions=None,
                          velocities=None, angular_velocities=None):
            pass

    class StrikeBallEvent(Event):
        def __init__(self, t, i, q, Q, V, cue_mass):
            super().__init__(t)
            self.i = i
            self.q = q
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass

    class SlideToRollEvent(Event):
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i

    class RollToRestEvent(Event):
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i

    class BallCollisionEvent(Event):
        def __init__(self, t, i, j):
            super().__init__(t)
            self.i = i
            self.j = j

    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.2,
                 e=0.93,
                 g=9.81,
                 initial_positions=None,
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
        self._t_E = np.zeros(num_balls, dtype=np.float32)
        self._positions = self._a[:,:,0]
        self._velocities = self._a[:,:,1]
        self.on_table = np.array(self.num_balls * [True])
        self.is_sliding = np.array(self.num_balls * [False])
        self.is_rolling = np.array(self.num_balls * [False])
        if initial_positions:
            self._a[:,:,0] = initial_positions
    @staticmethod
    def _quartic_solve(p):
        # TODO: use analytic solution method (e.g. Ferrari)
        return np.roots(p)
    def _in_global_t(self, balls, out=None):
        if isinstance(balls, int):
            balls = [balls]
        n = len(balls)
        if out is None:
            out = np.empty((n,3,3), dtype=np.float32)
        for ii, i in enumerate(balls):
            a_i = self._a[i]
            t_E = self._t_E[i]
            print(ii, t_E)
            out[ii,:,0] = a_i[:,0] - a_i[:,1] * t_E + a_i[:,2] * t_E**2
            out[ii,:,1] = a_i[:,1] - 2 * t_E * a_i[:,2]
            out[ii,:,2] = a_i[:,2]
        return out
    def strike_ball(self, t, i, q, Q, V, cue_mass):
        if not self.on_table[i]:
            return
        event = self.StrikeBallEvent(t, i, q, Q, V, cue_mass)
        self.events.append(event)
        self._t_E[i] = t
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
        self.is_sliding[i] = True
        # duration of sliding state:
        tau_s = 2 * np.linalg.norm(u) / (7 * mu_s * g)
        # duration until (potential) collision:
        tau_c = float('inf')
        a_i = self._in_global_t(i).reshape(3,3)
        p = np.empty(5, dtype=np.float32)
        for j, on in enumerate(self.on_table):
            if on:
                a_j = self._in_global_t(j).reshape(3,3)
                print(a_j)
                raise Exception()
                d = a_i - a_j
                a_x, a_y = d[::2, 2]
                b_x, b_y = d[::2, 1]
                c_x, c_y = d[::2, 0]
                p[0] = a_x**2 + a_y**2
                p[1] = 2 * (a_x * b_x + a_y * b_y)
                p[2] = b_x**2 + 2 * a_x * c_x + 2 * a_y * c_y + b_y**2
                p[3] = 2 * b_x * c_x + 2 * b_y * c_y
                p[4] = c_x**2 + c_y**2 - 4 * R**2
                print(p)
                roots = PoolPhysics._quartic_solve(p)
                print(roots)
        if tau_s < tau_c:
            leading_prediction = [self.SlideToRollEvent(t + tau_s, i)]
        else:
            leading_prediction = [self.BallCollisionEvent(t + tau_c, i, j)]
        predicted_events = self.predict_events(leading_prediciton=leading_prediction)
        self.events += predicted_events
        return predicted_events
    def predict_events(self, leading_prediction=None):
        if leading_prediction is None:
            leading_prediction = []
        for i, sliding in enumerate(self.is_sliding):
            if sliding:
                pass
    def eval_positions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        self._in_global_t(np.range(self.num_balls))
        # lt = self.events[-1].t
        # tarray = np.array((1.0, t - lt, (t - lt)**2), dtype=np.float32)
        # return self._a.dot(tarray, out=out)
        raise TODO()
    def eval_quaternions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 4), dtype=np.float32)
        raise TODO()
    def eval_velocities(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        raise TODO()
    def eval_angular_velocities(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        raise TODO()
