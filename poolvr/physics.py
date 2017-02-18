import logging
import bisect
import numpy as np


_logger = logging.getLogger(__name__)


from .exceptions import TODO


INCH2METER = 0.0254


class PoolPhysics(object):

    class PhysicsEvent(object):
        physics = None
        _num_balls = 0
        def __init__(self, t, **kwargs):
            self.t = t
            self.T = 0
            self.next_event = None
        def _calc_global_coeffs(self):
            if self._num_balls == 0:
                return
            a = self._a.copy().reshape(self._num_balls, 3, 3)
            b = self._b.copy().reshape(self._num_balls, 2, 3)
            t = self.t
            a[:,0] += -t * self._a[:,1] + t**2 * self._a[:,2]
            a[:,1] += -2 * t * self._a[:,2]
            return a, b
        def __lt__(self, other):
            return self.t < other.t
        def __gt__(self, other):
            return self.t > other.t
        def __eq__(self, other):
            return self.t == other.t

    class StrikeBallEvent(PhysicsEvent):
        _num_balls = 1
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
            tau_s = 2 * np.linalg.norm(u) / (7 * mu_s * g)
            self.T = tau_s
            self.next_event = self.physics.SlideToRollEvent(t + tau_s, i)

    class SlideToRollEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i
            if self.physics:
                self._a = self.physics._a[i].copy()
                self._b = self.physics._b[i].copy()
            else:
                self._a = np.zeros((3,3), dtype=np.float32)
                self._b = np.zeros((2,3), dtype=np.float32)
            tau_r = 2.0
            self.T = tau_r
            self.next_event = self.physics.RollToRestEvent(t + tau_r, i)

    class RollToRestEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i):
            super().__init__(t)
            self.i = i

    class BallCollisionEvent(PhysicsEvent):
        _num_balls = 2
        def __init__(self, t, i, j):
            super().__init__(t)
            self.i = i
            self.j = j

    _dummy_event = PhysicsEvent(0)

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
        self.ball_events = self.num_balls * [None]
        self.nevent = 0
        self.on_table = np.array(self.num_balls * [True])
        self.is_sliding = np.array(self.num_balls * [False])
        self.is_rolling = np.array(self.num_balls * [False])
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float32)
        if initial_positions is not None:
            self._a[:,0] = initial_positions
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float32)
        self._t_E = np.zeros(num_balls, dtype=np.float32)

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
        event = self.StrikeBallEvent(t, i, q, Q, V, cue_mass)
        _a, _b = event._calc_global_coeffs()
        self._a[i] = _a[0]
        self._b[i] = _b[0]
        events = [event]
        self.events.append(event)
        self.ball_events[i] = event
        self._t_E[i] = t
        self.is_sliding[i] = True
        # while events[-1].next_event is not None:
        #     predicted_event = event.next_event
        #     for j, on in enumerate(self.on_table):
        #         if j == i: continue
        #         if on:
        #             t_c = self._find_collision(self._a[i], self._a[j])
        #             if t_c and t_c < predicted_event.t:
        #                 predicted_event = self.BallCollisionEvent(t_c, i, j)
        #     events.append(predicted_event)
        predicted_event = event.next_event
        for j, on in enumerate(self.on_table):
            if j == i: continue
            if on:
                t_c = self._find_collision(self._a[i], self._a[j])
                if t_c and t_c < predicted_event.t:
                    predicted_event = self.BallCollisionEvent(t_c, i, j)
        events.append(predicted_event)
        self.events += events
        return events

    def eval_positions(self, t, out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        out[:] = self._a[:,0]
        for e in self._find_active_events(t):
            tau = t - e.t
            if e._num_balls == 1:
                out[e.i] = e._a[0] + tau * e._a[1] + tau**2 * e._a[2]
            elif e._num_balls == 2:
                out[e.i] = e._a[0,0] + tau * e._a[0,1] + tau**2 * e._a[0,2]
                out[e.j] = e._a[1,0] + tau * e._a[1,1] + tau**2 * e._a[1,2]
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

    @staticmethod
    def _quartic_solve(p):
        # TODO: use analytic solution method (e.g. Ferrari)
        return np.roots(p)

    def _find_active_events(self, t):
        self._dummy_event.t = t
        n = bisect.bisect_left(self.events, self._dummy_event)
        return [e for e in self.events[:n] if e.t + e.T >= t]

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
