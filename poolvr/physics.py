"""

Event-based pool physics simulation engine based on the paper
(available at http://web.stanford.edu/group/billiards/AnEventBasedPoolPhysicsSimulator.pdf):

  AN EVENT-BASED POOL PHYSICS SIMULATOR
  Will Leckie, Michael Greenspan
  DOI: 10.1007/11922155_19 · Source: DBLP
  Conference: Advances in Computer Games, 11th International Conference,
  Taipei, Taiwan, September 6-9, 2005.

"""
import logging
import bisect
import numpy as np


from .exceptions import TODO


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254

_I, _J, _K = np.eye(3, dtype=np.float32)


class PoolPhysics(object):
    """
    Pool physics simulator

    :param mu_r:  friction coefficient (rolling)
    :param mu_sp: friction coefficient (spining)
    :param mu_s:  friction coefficient (sliding)
    :param e:     coefficient of restitution for ball collisions
    :param g:     gravity
    """
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
        self._ball_events = {}
        self.on_table = np.array(self.num_balls * [True])
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float32)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float32)
        if initial_positions is not None:
            self._a[:,0] = initial_positions

    def strike_ball(self, t, i, Q, V, cue_mass):
        """
        Strike ball *i* at game time *t*.  The cue points in the direction specified
        by the unit vector *q* = :math:`\hat{q}`, and strikes the ball at point *Q* =
        :math:`Q_x \hat{i} + Q_y \hat{j} + Q_z \hat{k}` which is specified
        in coordinates relative to the ball's center with :math:`\hat{k}`
        aligned with the horizontal component of :math:`\hat{q}`.
        *V* is the velocity of the cue at the instant when it strikes the ball.
        """
        if not self.on_table[i]:
            return
        event = self.StrikeBallEvent(t, i, Q, V, cue_mass)
        events = [event]
        self.events.append(event)
        self._ball_events.clear()
        self._ball_events[i] = event
        collide_times = {}
        while self._ball_events:
            predicted_event = None
            for i, event in self._ball_events.items():
                if i != event.i: continue
                if predicted_event is None or (event.next_event and event.next_event.t < predicted_event.t):
                    predicted_event = event.next_event
                for j in [j for j, on in enumerate(self.on_table)
                          if on and j != i and self._ball_events.get(j) is not event]:
                    if i < j:
                        key = (event, self._ball_events.get(j, j))
                    else:
                        key = (self._ball_events.get(j, j), event)
                    if key not in collide_times:
                        _a, _b = event._calc_global_coeffs()
                        if j in self._ball_events:
                            _a_j, _b_j = self._ball_events[j]._calc_global_coeffs()
                            _a_j = _a_j[0] if j == self._ball_events[j].i else _a_j[1]
                            _b_j = _b_j[0] if j == self._ball_events[j].i else _b_j[1]
                        else:
                            _a_j, _b_j = self._a[j], self._b[j]
                        collide_times[key] = self._find_collision_time(_a[0], _a_j)
                    t_c = collide_times[key]
                    if t_c and j in self._ball_events \
                       and t_c - event.t <= event.T \
                       and t_c - self._ball_events[j].t <= self._ball_events[j].T \
                       and (predicted_event is None or t_c < predicted_event.t):
                        positions = self.eval_positions(t_c)[(i,j)]
                        velocities = self.eval_velocities(t_c)[(i,j)]
                        predicted_event = self.BallCollisionEvent(t_c, i, j, positions, velocities)
                        _logger.debug('i, j, t_c, t = %d, %d, %s, %s', i, j, t_c, predicted_event.t)
                if predicted_event is None:
                    self._ball_events.pop(i)
                    if hasattr(event, 'j'):
                        self._ball_events.pop(event.j)
            if predicted_event is None:
                break
            events.append(predicted_event)
            self.events.append(predicted_event)
            if isinstance(predicted_event, self.BallCollisionEvent):
                i, j = predicted_event.i, predicted_event.j
                if i in self._ball_events:
                    self._ball_events.pop(i)
                if j in self._ball_events:
                    self._ball_events.pop(j)
                # self._ball_events[predicted_event.i] = predicted_event
                # self._ball_events[predicted_event.j] = predicted_event
            elif isinstance(predicted_event, self.SlideToRollEvent):
                i = predicted_event.i
                self._ball_events[i] = predicted_event
            elif isinstance(predicted_event, self.RollToRestEvent):
                i = predicted_event.i
                self._ball_events.pop(i)
        return events

    def eval_positions(self, t, out=None):
        """
        Evaluate the positions of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
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
        """
        Evaluate the rotations of all balls (represented as quaternions) at game time *t*.

        :returns: shape (*N*, 4) array, where *N* is the number of balls
        """
        if out is None:
            out = np.empty((self.num_balls, 4), dtype=np.float32)
        raise TODO()

    def eval_velocities(self, t, out=None):
        """
        Evaluate the velocities of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        out[:] = self._a[:,1]
        for e in self._find_active_events(t):
            tau = t - e.t
            if e._num_balls == 1:
                out[e.i] = e._a[1] + 2 * tau * e._a[2]
            elif e._num_balls == 2:
                out[e.i] = e._a[0,1] + 2 * tau * e._a[0,2]
                out[e.j] = e._a[1,1] + 2 * tau * e._a[1,2]
        return out

    def eval_angular_velocities(self, t, out=None):
        """
        Evaluate the angular velocities of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        raise TODO()

    def reset(self, ball_positions):
        self._a[:] = 0
        self._a[:,0] = ball_positions
        self._b[:] = 0
        self.events = []
        self._ball_events.clear()
        self.on_table[:] = True

    @staticmethod
    def _quartic_solve(p):
        # TODO: use analytic solution method (e.g. Ferrari)
        return np.roots(p)

    def _find_active_events(self, t):
        self._dummy_event.t = t
        n = bisect.bisect_left(self.events, self._dummy_event)
        return [e for e in self.events[:n] if e.t + e.T >= t]

    def _find_collision_time(self, a_i, a_j):
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

    class PhysicsEvent(object):
        physics = None
        _num_balls = 0
        def __init__(self, t, **kwargs):
            self.t = t
            self.T = 0
            self.next_event = None
        def eval_positions(self, tau, out=None):
            if self._num_balls == 0:
                return None
            if out is None:
                out = np.empty((self._num_balls, 3), dtype=np.float32)
            _a = self._a.reshape(self._num_balls, 3, 3)
            if hasattr(self, 'i'):
                out[0] = _a[0,0] + tau * _a[0,1] + tau**2 * _a[0,2]
            if hasattr(self, 'j'):
                out[1] = _a[1,0] + tau * _a[1,1] + tau**2 * _a[1,2]
            return out
        def _calc_global_coeffs(self):
            if self._num_balls == 0:
                return
            a = self._a.copy().reshape(self._num_balls, 3, 3)
            b = self._b.copy().reshape(self._num_balls, 2, 3)
            t = self.t
            a[:,0] += -t * self._a[:,1] + t**2 * self._a[:,2]
            a[:,1] += -2 * t * self._a[:,2]
            return a, b
        def __str__(self):
            clsname = self.__class__.__name__.split('.')[-1]
            return "<%s: t=%f T=%f" % (clsname, self.t, self.T) + \
                (' i=%d'  % self.i if hasattr(self, 'i') else '') + \
                (' j=%d>' % self.j if hasattr(self, 'j') else '>')
        def __lt__(self, other):
            return self.t < other.t
        def __gt__(self, other):
            return self.t > other.t
        def __eq__(self, other):
            return str(self) == str(other)
        def __hash__(self):
            return hash(str(self))

    class StrikeBallEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i, Q, V, cue_mass):
            super().__init__(t)
            self.i = i
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass
            self._a = np.zeros((3,3), dtype=np.float32)
            self._b = np.zeros((2,3), dtype=np.float32)
            self._a[0] = self.physics._a[i,0]
            a, b, c = Q
            V[1] = 0
            sin, cos = 0.0, 1.0
            M = cue_mass
            m, R = self.physics.ball_mass, self.physics.ball_radius
            norm_V = np.linalg.norm(V)
            F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
            # post-impact ball velocity:
            v = self._a[1]
            v[::2] = F / m * V[::2] / np.linalg.norm(V[::2])
            # post-impact ball angular velocity:
            omega = self._b[0]
            I = self.physics._I
            omega_i = F * (-c * sin + b * cos) / I
            omega_j = F * a * sin / I
            omega_k = -F * a * cos / I
            _j = -V[:] / norm_V
            _k = _J
            _i = np.cross(_j, _k)
            omega[:] = omega_i * _i + omega_j * _j + omega_k * _k
            u = v + R * np.cross(_k, omega)
            u[1] = 0
            norm_u = np.linalg.norm(u)
            mu_s, mu_sp, g = self.physics.mu_s, self.physics.mu_sp, self.physics.g
            self._a[2,::2] = -0.5 * mu_s * g * u[::2] / norm_u
            self._b[1] = -5 * mu_s * g / (2 * R) * np.cross(_k, u) / norm_u
            self._b[1,1] = -5 * mu_sp * g / (2 * R)
            tau_s = 2 * norm_u / (7 * mu_s * g)
            self.T = tau_s
            end_position = self._a[0] + tau_s * self._a[1] + tau_s**2 * self._a[2]
            end_velocity = self._a[1] + 2 * tau_s * self._a[2]
            self.next_event = self.physics.SlideToRollEvent(t + tau_s, i,
                                                            end_position, end_velocity)
        def __str__(self):
            return super().__str__()[:-1] + ' Q=%s V=%s r=%s v=%s>' % (self.Q, self.V, self._a[0], self._a[1])

    class SlideToRollEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i, position, velocity):
            super().__init__(t)
            self.i = i
            self._a = np.zeros((3,3), dtype=np.float32)
            self._b = np.zeros((2,3), dtype=np.float32)
            self._a[0] = position
            self._a[1] = velocity
            self._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (velocity / np.linalg.norm(velocity))
            tau_r = np.linalg.norm(velocity) / (self.physics.mu_r * self.physics.g)
            self.T = tau_r
            end_position = self._a[0] + tau_r * self._a[1] + tau_r**2 * self._a[2]
            self.next_event = self.physics.RollToRestEvent(t + tau_r, i, end_position)
        def __str__(self):
            return super().__str__()[:-1] + ' r=%s v=%s>' % (self._a[0], self._a[1])

    class SlideToRestEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i, position):
            super().__init__(t)
            self.i = i
            self._a = np.zeros((3,3), dtype=np.float32)
            self._b = np.zeros((2,3), dtype=np.float32)
            self._a[0] = position
            self.T = 0

    class RollToRestEvent(PhysicsEvent):
        _num_balls = 1
        def __init__(self, t, i, position):
            super().__init__(t)
            self.i = i
            self._a = np.zeros((3,3), dtype=np.float32)
            self._b = np.zeros((2,3), dtype=np.float32)
            self._a[0] = position
            self.T = 0
        def __str__(self):
            return super().__str__()[:-1] + ' r=%s>' % self._a[0]

    class BallCollisionEvent(PhysicsEvent):
        _num_balls = 2
        def __init__(self, t, i, j, positions, velocities):
            super().__init__(t)
            self.i = i
            self.j = j
            self._a = np.zeros((2,3,3), dtype=np.float32)
            self._b = np.zeros((2,2,3), dtype=np.float32)
            self._a[:,0] = positions
            self._a[:,1] = velocities
            # TODO

    _dummy_event = PhysicsEvent(0)
