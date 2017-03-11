"""
This module implements an event-based pool physics simulator based on the paper
(available at http://web.stanford.edu/group/billiards/AnEventBasedPoolPhysicsSimulator.pdf): ::

  AN EVENT-BASED POOL PHYSICS SIMULATOR
  Will Leckie, Michael Greenspan
  DOI: 10.1007/11922155_19 · Source: DBLP
  Conference: Advances in Computer Games, 11th International Conference,
  Taipei, Taiwan, September 6-9, 2005.

"""
import logging
from bisect import bisect
import numpy as np
from copy import copy

from .exceptions import TODO


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254

_I, _J, _K = np.eye(3, dtype=np.float32)


class PoolPhysics(object):
    """
    Pool physics simulator

    :param mu_r:  :math:`\mu_r`, rolling friction coefficient
    :param mu_sp: :math:`\mu_{sp}`, spinning friction coefficient
    :param mu_s:  :math:`\mu_s`, sliding friction coefficient
    :param mu_b:  :math:`\mu_b`, ball-to-ball collision friction coefficient
    :param c_b:   :math:`c_b`, ball material's speed of sound
    :param E_Y_b: :math:`{E_Y}_b`, ball material's Young's modulus
    :param g:     downward acceleration due to gravity
    """
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
                 initial_positions=None,
                 **kwargs):
        self.PhysicsEvent.physics = self
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
        self.all_balls = list(range(self.num_balls))
        self.on_table = np.array(self.num_balls * [True])
        self.events = []
        self.ball_events = {i: [] for i in self.all_balls}
        self._ball_events = {}
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float32)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float32)
        if initial_positions is not None:
            self._a[:,0] = initial_positions

    def strike_ball(self, t, i, Q, V, cue_mass):
        r"""
        Strike ball *i* at game time *t*.  The cue strikes the ball at point *Q* =
        :math:`a \hat{i} + b \hat{j} + c \hat{k}`, which is specified
        in coordinates relative to the ball's center with the :math:`\hat{k}`-axis
        aligned with the horizontal component of the cue's impact velocity
        *V* = :math:`\vec{V}`, i.e.

        .. math::

          \hat{k} = - \frac{\vec{V} - (\vec{V} \cdot \hat{j}) \hat{j}}{ \| \vec{V} - (\vec{V} \cdot \hat{j}) \hat{j} \| }

        and :math:`c > 0`.
        """
        if not self.on_table[i]:
            return
        self._ball_events.clear()
        event = self.StrikeBallEvent(t, i, Q, V, cue_mass)
        self._add_event(event)
        events = [event]
        self._ball_events[i] = event
        collision_times = {}
        while self._ball_events:
            _logger.debug('\n'.join(['%d: %s' % (k, v) for k, v in sorted(self._ball_events.items())]))
            predicted_event = None
            for event in self._ball_events.values():
                if predicted_event is None or (event.next_event and event.next_event.t < predicted_event.t):
                    predicted_event = event.next_event
            for i, event in self._ball_events.items(): #sorted(list(self._ball_events.items())):
                t_i, T_i = event.t, event.T
                r_0i, v_0i = event._a[:2]
                _a, _b = event._calc_global_coeffs()
                for j in [j for j, on in enumerate(self.on_table)
                          if on and (j not in self._ball_events or j > i)]:
                    if j in self._ball_events:
                        e_j = self._ball_events[j]
                        if isinstance(e_j, self.BallCollisionEvent) and e_j.i == i:
                            continue
                        r_0j, v_0j = e_j._a[:2]
                        r_0ji = r_0i - r_0j
                        v_0ji = v_0i - v_0j
                        if ((abs(r_0ji) > 2*self.ball_radius) & (np.sign(r_0ji) == np.sign(v_0ji))).any():
                            continue
                        t_j, T_j = e_j.t, e_j.T
                        _a_j, _b_j = e_j._calc_global_coeffs()
                        t0 = max(t_i, t_j)
                        t1 = min(t_i + T_i, t_j + T_j)
                    else:
                        if isinstance(event, self.BallCollisionEvent) and event.j == j:
                            continue # <-- i think this is right
                        j_events = self.ball_events[j]
                        if j_events:
                            e_j = j_events[-1]
                            _a_j, _b_j = e_j._a, e_j._b
                            t0 = max(t_i, e_j.t)
                            t1 = min(t_i + T_i, e_j.t + e_j.T)
                        else:
                            _a_j, _b_j = self._a[j], self._b[j]
                            t0 = t_i
                            t1 = t_i + T_i
                        r_0j = _a_j[0]
                        r_0ji = r_0i - r_0j
                        v_0ji = v_0i
                        if ((abs(r_0ji) > 2*self.ball_radius) & (np.sign(r_0ji) == np.sign(v_0ji))).any():
                            continue
                    key = (event, j)
                    if key not in collision_times:
                        collision_times[key] = self._find_collision_time(_a, _a_j, t0, t1)
                    t_c = collision_times[key]
                    #t_c = self._find_collision_time(_a, _a_j, t0, t1)
                    if t_c and (predicted_event is None or t_c < predicted_event.t):
                        tau_i = t_c - t_i
                        r_i = event.eval_position(tau_i)
                        v_i = event.eval_velocity(tau_i)
                        if j in self._ball_events:
                            e_j = self._ball_events[j]
                            tau_j = t_c - e_j.t
                            r_j = e_j.eval_position(tau_j)
                            v_j = e_j.eval_velocity(tau_j)
                        else:
                            r_j = self.ball_events[j][-1]._a[0] if self.ball_events[j] else self._a[j,0]
                            v_j = np.zeros(3, dtype=np.float32)
                        predicted_event = self.BallCollisionEvent(t_c, i, j, r_i, r_j, v_i, v_j)
            if predicted_event is None:
                break
            determined_event = predicted_event
            events.append(determined_event)
            self._add_event(determined_event)
            i = determined_event.i
            if isinstance(determined_event, self.BallCollisionEvent):
                self._add_event(determined_event.paired_event)
                j = determined_event.j
                self._ball_events[i] = determined_event
                self._ball_events[j] = determined_event.paired_event
                events.append(determined_event.paired_event)
            elif isinstance(determined_event, self.SlideToRollEvent):
                self._ball_events[i] = determined_event
            elif isinstance(determined_event, self.RollToRestEvent) or isinstance(determined_event, self.SlideToRestEvent):
                self._ball_events.pop(i)
        return events

    STATIONARY = 0
    SLIDING    = 1
    ROLLING    = 2
    SPINNING   = 3

    def find_ball_states(self, t, balls=None):
        """
        Determine the states (stationary, sliding, rolling, or spinning) of a set of balls at game time *t*.

        :returns: dict mapping ball number to state
        """
        if balls is None:
            balls = self.all_balls
        states = {}
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        states[i] = e.state
            else:
                states[i] = self.STATIONARY
        return states

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float32)
        out[:] = self._a[balls,0] # TODO: use PositionBallEvent instead
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_position(t - e.t)
                        break
        return out

    def eval_quaternions(self, t, out=None):
        """
        Evaluate the rotations of a set of balls (represented as quaternions) at game time *t*.

        :returns: shape (*N*, 4) array, where *N* is the number of balls
        """
        if out is None:
            out = np.empty((self.num_balls, 4), dtype=np.float32)
        raise TODO()

    def eval_velocities(self, t, balls=None, out=None):
        """
        Evaluate the velocities of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float32)
        out[:] = 0
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_velocity(t - e.t)
                        break
        return out

    def eval_angular_velocities(self, t, balls=None, out=None):
        """
        Evaluate the angular velocities of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float32)
        out[:] = 0
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_angular_velocity(t - e.t)
                        break
        return out

    def reset(self, ball_positions):
        """
        Reset the state of the balls to at rest at the specified positions
        """
        self.events = []
        self.on_table[:] = True
        self._a[:] = 0
        self._a[:,0] = ball_positions
        self._b[:] = 0
        self._ball_events.clear()
        self.ball_events = {i: [] for i in self.all_balls}

    def _add_event(self, event):
        self.events.append(event)
        i_events = self.ball_events[event.i]
        if i_events:
            prev = i_events[-1]
            if event.t < prev.t + prev.T:
                prev.T = event.t - prev.t
        i_events.append(event)

    @staticmethod
    def _quartic_solve(p):
        # TODO: use analytic solution method (e.g. Ferrari)
        return np.roots(p)

    def _find_active_events(self, t):
        n = bisect(self.events, t)
        return [e for e in self.events[:n] if t >= e.t and t <= e.t + e.T]

    def _find_collision_time(self, a_i, a_j, t0, t1):
        _logger.debug('a_i = %s\na_j = %s', a_i, a_j)
        d = a_i - a_j
        if np.linalg.norm(d[1,::2]) * (t1-t0) < np.linalg.norm(d[0,::2]) - 2*self.ball_radius:
            return None
        a_x, a_y = d[2, ::2]
        b_x, b_y = d[1, ::2]
        c_x, c_y = d[0, ::2]
        p = np.empty(5, dtype=np.float32)
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x*b_x + a_y*b_y)
        p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
        p[3] = 2 * b_x * c_x + 2 * b_y * c_y
        p[4] = c_x**2 + c_y**2 - 4 * self.ball_radius**2
        roots = self._quartic_solve(p)
        roots = [t.real for t in roots if t.real > t0 and t.real < t1 and abs(t.imag / t.real) < 0.01]
        if not roots:
            return None
        else:
            return min(roots)

    def _calc_energy(self, t, balls=None):
        if balls is None:
            balls = self.all_balls
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return self.ball_mass * (velocities**2).sum() / 2 + self._I * (omegas**2).sum() / 2

    class PhysicsEvent(object):
        physics = None
        def __init__(self, t, i, **kwargs):
            self.t = t
            self.i = i
            self.T = 0
            self.next_event = None
            self._a = np.zeros((3,3), dtype=np.float32)
            self._b = np.zeros((2,3), dtype=np.float32)
        def eval_position(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float32)
            _a = self._a
            out[:] = _a[0] + tau * _a[1] + tau**2 * _a[2]
            return out
        def eval_velocity(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float32)
            _a = self._a
            out[:] = _a[1] + 2 * tau * _a[2]
            return out
        def eval_angular_velocity(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float32)
            _b = self._b
            out[:] = _b[0] + tau * _b[1]
            return out
        def _calc_global_coeffs(self):
            _a, _b, t = self._a, self._b, self.t
            a = _a.copy()
            b = _b.copy()
            a[0] += -t * _a[1] + t**2 * _a[2]
            a[1] += -2 * t * _a[2]
            return a, b
        def __str__(self):
            clsname = self.__class__.__name__.split('.')[-1]
            return "<%17s: t=%8.3f T=%8.3f i=%2d>" % (clsname, self.t, self.T, self.i)
        def __lt__(self, other):
            if isinstance(other, self.physics.PhysicsEvent):
                return self.t < other.t
            else:
                return self.t < other
        def __gt__(self, other):
            if isinstance(other, self.physics.PhysicsEvent):
                return self.t > other.t
            else:
                return self.t > other
        def __eq__(self, other):
            return self.i == other.i and self.t == other.t
        def __hash__(self):
            return hash((self.i, self.t))

    class PositionBallEvent(PhysicsEvent):
        def __init__(self, t, i, r):
            super().__init__(t, i)
            self._a[0] = r
            self.T = float('inf')
            self.state = self.physics.STATIONARY

    class StrikeBallEvent(PhysicsEvent):
        def __init__(self, t, i, Q, V, cue_mass):
            super().__init__(t, i)
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass
            self._a[0] = self.physics._a[i,0]
            a, b, c = Q
            V[1] = 0
            sin, cos = 0.0, 1.0
            M = cue_mass
            m, R = self.physics.ball_mass, self.physics.ball_radius
            norm_V = np.linalg.norm(V)
            F = 2.0 * m * norm_V / (1 + m/M + 5.0/(2*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
            v = self._a[1] # <-- post-impact ball velocity
            v[::2] = F / m * V[::2] / np.linalg.norm(V[::2])
            omega = self._b[0] # <-- post-impact ball angular velocity
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
            end_position = self.eval_position(tau_s)
            end_velocity = self.eval_velocity(tau_s)
            self.state = self.physics.SLIDING
            self.next_event = self.physics.SlideToRollEvent(t + tau_s, i, end_position, end_velocity)
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s v=%40s Q=%40s V=%40s>' % (self._a[0], self._a[1], self.Q, self.V)

    class SlideToRollEvent(PhysicsEvent):
        def __init__(self, t, i, position, velocity):
            super().__init__(t, i)
            self._a[0] = position
            self._a[1] = velocity
            self._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (velocity / np.linalg.norm(velocity))
            tau_r = np.linalg.norm(velocity) / (self.physics.mu_r * self.physics.g)
            self.T = tau_r
            end_position = self.eval_position(tau_r)
            self.state = self.physics.ROLLING
            self.next_event = self.physics.RollToRestEvent(t + tau_r, i, end_position)
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s v=%40s>' % (self._a[0], self._a[1])

    class SlideToRestEvent(PhysicsEvent):
        def __init__(self, t, i, position):
            super().__init__(t, i)
            self._a[:] = 0
            self._a[0] = position
            self.T = float('inf')
            self.state = self.physics.STATIONARY
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s>' % self._a[0]

    class RollToRestEvent(PhysicsEvent):
        def __init__(self, t, i, position):
            super().__init__(t, i)
            self._a[:] = 0
            self._a[0] = position
            self.T = float('inf')
            self.state = self.physics.STATIONARY
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s>' % self._a[0]

    class BallCollisionEvent(PhysicsEvent):
        def __init__(self, t, i, j, r_i, r_j, v_i, v_j):
            super().__init__(t, i)
            self.j = j
            self._a[0] = r_i
            v_ij = v_j - v_i
            norm_v_ij = np.linalg.norm(v_ij)
            delta_t = 284e-6 / norm_v_ij**0.294
            s_max = 1.65765 * (norm_v_ij / self.physics.c_b)**0.8
            F_max = 1.48001 * self.physics.ball_radius**2 * self.physics.E_Y_b * s_max**1.5
            J = 0.5 * F_max * delta_t
            r_ij = r_j - r_i
            _i = r_ij / np.linalg.norm(r_ij)
            post_v_i = self._a[1]
            post_v_i[:] = v_i - (J / self.physics.ball_mass) * _i
            self._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (post_v_i / np.linalg.norm(post_v_i))
            tau_i = np.linalg.norm(post_v_i) / (self.physics.mu_r * self.physics.g)
            self.T = tau_i
            self.next_event = self.physics.RollToRestEvent(t + tau_i, i, self.eval_position(tau_i))
            self.state = self.physics.ROLLING
            paired_event = copy(self)
            self.paired_event = paired_event
            paired_event.paired_event = self
            paired_event.i, paired_event.j = j, i
            paired_event._a = np.empty((3,3), dtype=np.float32)
            paired_event._b = np.zeros((2,3), dtype=np.float32)
            paired_event._a[0] = r_j
            post_v_j = paired_event._a[1]
            post_v_j[:] = v_j + (J / self.physics.ball_mass) * _i
            paired_event._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (post_v_j / np.linalg.norm(post_v_j))
            tau_j = np.linalg.norm(post_v_j) / (self.physics.mu_r * self.physics.g)
            paired_event.T = tau_j
            paired_event.next_event = self.physics.RollToRestEvent(t + tau_j, j, paired_event.eval_position(tau_j))
        def __str__(self):
            return super().__str__()[:-1] + ' j=%2d>' % self.j
