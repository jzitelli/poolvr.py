import logging
import numpy as np


_logger = logging.getLogger(__name__)


from .exceptions import TODO


INCH2METER = 0.0254


class PoolPhysics(object):

    class PhysicsEvent(object):

        physics = None

        def __init__(self, t, **kwargs):
            self.t = t
        def __lt__(self, other):
            return self.t < other.t
        def __gt__(self, other):
            return self.t > other.t
        def __eq__(self, other):
            return self.t == other.t

    class StrikeBallEvent(PhysicsEvent):
        def __init__(self, t, i, q, Q, V, cue_mass):
            super().__init__(t)
            self.i = i
            self.q = q
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass
            if self.physics:
                self._a = self.physics._a[i].copy()
                self._b = self.physics._b[i].copy()
            else:
                self._a = np.zeros((3,3), dtype=np.float32)
                self._b = np.zeros((2,3), dtype=np.float32)
            a, c, b = Q
            V_xz = V[::2]
            norm_V = np.linalg.norm(V)
            norm_V_xz = np.linalg.norm(V_xz)
            sin, cos = 0.0, 1.0
            M = cue_mass
            m, R = self.physics.ball_mass, self.physics.ball_radius
            F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
            norm_v = -F / m * cos
            v = self._a[1] # post-impact ball velocity
            v[0] = norm_v * V[0] / norm_V_xz
            v[2] = norm_v * V[2] / norm_V_xz
            v[1] = 0 # TODO
            I = self.physics._I
            omega_x = F * (-c * sin + b * cos) / I
            omega_z = F * a * sin / I
            omega = self._b[0] # post-impact ball angular velocity
            omega[0] = omega_x * V[0] / norm_V_xz
            omega[2] = omega_z * V[2] / norm_V_xz
            omega[1] = 0
            # TODO: omega[1] = -F * a * cos / I
            u = v + R * np.array((-omega[2], 0.0, omega[0]), dtype=np.float32) # relative velocity
            mu_s, mu_sp, g = self.physics.mu_s, self.physics.mu_sp, self.physics.g
            self._a[2,::2] = -0.5 * mu_s * g * u[::2]
            self._b[1,::2] = -5 * mu_s * g / (2 * R) * np.array((-u[2], u[0]), dtype=np.float32)
            self._b[1,1] = -5 * mu_sp * g / (2 * R)

    class SlideToRollEvent(PhysicsEvent):
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i

    class RollToRestEvent(PhysicsEvent):
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i

    class BallCollisionEvent(PhysicsEvent):
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
        self.PhysicsEvent.physics = self
        self.num_balls = num_balls
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.e = e
        self.g = g
        self.t = 0.0
        self.events = []
        self.nevent = 0
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float32)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float32)
        self._t_E = np.zeros(num_balls, dtype=np.float32)
        self.on_table = np.array(self.num_balls * [True])
        self.is_sliding = np.array(self.num_balls * [False])
        self.is_rolling = np.array(self.num_balls * [False])
        self.ball_events = self.num_balls * [None]
        if initial_positions is not None:
            self._a[:,0] = initial_positions
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
            out[ii,0] = a_i[0] - a_i[1] * t_E + a_i[2] * t_E**2
            out[ii,1] = a_i[1] - 2 * t_E * a_i[2]
            out[ii,2] = a_i[2]
        return out
    def _find_collision(self, a_i, a_j):
        d = a_i - a_j
        a_x, a_y = d[2, ::2]
        b_x, b_y = d[1, ::2]
        c_x, c_y = d[0, ::2]
        p = np.empty(5, dtype=np.float32)
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x * b_x + a_y * b_y)
        p[2] = b_x**2 + 2 * a_x * c_x + 2 * a_y * c_y + b_y**2
        p[3] = 2 * b_x * c_x + 2 * b_y * c_y
        p[4] = c_x**2 + c_y**2 - 4 * self.ball_radius**2
        roots = self._quartic_solve(p)
        roots = [t for t in roots if t.real > self.t and abs(t.imag / t.real) < 0.01]
        if not roots:
            return None
        roots = sorted(roots, key=lambda t: (abs(t.imag), t.real))
        return roots[0].real
    def strike_ball(self, t, i, q, Q, V, cue_mass):
        """
        Strike ball *i* at game time *t*.  The cue points in the direction specified
        by the unit vector *q*, and strikes the ball at point *Q*
        which is specified in coordinates local to the ball
        (i.e. :math:`||Q|| = \emph{ball_radius}`),
        and aligned with *q* along the second (:math:`y`-) axis.
        *V* is the velocity of the cue at the instant when it strikes the ball.
        """
        if not self.on_table[i]:
            return
        a, c, b = Q
        V_xz = V[::2]
        norm_V = np.linalg.norm(V)
        norm_V_xz = np.linalg.norm(V_xz)
        # sin, cos = abs(V_xz) / norm_V_xz
        sin, cos = 0.0, 1.0
        M, m, R = cue_mass, self.ball_mass, self.ball_radius
        F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
        norm_v = -F / m * cos
        v = self._a[i,1] # post-impact ball velocity
        v[0] = norm_v * V[0] / norm_V_xz
        v[2] = norm_v * V[2] / norm_V_xz
        v[1] = 0 # TODO
        I = self._I
        omega_x = F * (-c * sin + b * cos) / I
        omega_z = F * a * sin / I
        omega = self._b[i,0] # post-impact ball angular velocity
        omega[0] = omega_x * V[0] / norm_V_xz
        omega[2] = omega_z * V[2] / norm_V_xz
        omega[1] = 0
        # TODO: omega[1] = -F * a * cos / I
        u = v + R * np.array((-omega[2], 0.0, omega[0]), dtype=np.float32) # relative velocity
        mu_s, mu_sp, g = self.mu_s, self.mu_sp, self.g
        self._a[i,2,::2] = -0.5 * mu_s * g * u[::2]
        self._b[i,1,::2] = -5 * mu_s * g / (2 * R) * np.array((-u[2], u[0]), dtype=np.float32)
        self._b[i,1,1] = -5 * mu_sp * g / (2 * R)

        event = self.StrikeBallEvent(t, i, q, Q, V, cue_mass)

        events = [event]
        self.events.append(event)
        self.ball_events[i] = event
        self._t_E[i] = t

        # duration of sliding state:
        self.is_sliding[i] = True
        tau = 2 * np.linalg.norm(u) / (7 * mu_s * g)
        predicted_event = self.SlideToRollEvent(t + tau, i)
        # determine any collisions during sliding state:
        a_i = self._in_global_t(i).reshape(3,3)
        p = np.empty(5, dtype=np.float32)
        for j, on in enumerate(self.on_table):
            if on:
                a_j = self._in_global_t(j).reshape(3,3)
                t_E = self._find_collision(a_i, a_j)
                if t_E and t_E < predicted_event.t:
                    predicted_event = self.BallCollisionEvent(t_E, i, j)
        events.append(predicted_event)
        self.events.append(predicted_event)
        return events
    def eval_positions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        for i, e_i in enumerate(self.ball_events):
            if e_i is not None:
                tau = t - e_i.t
                out[:] = self._a[i,0] + tau * self._a[i,1] + tau**2 * self._a[i,2]
            else:
                out[i] = self._a[i,0]
        return out
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
