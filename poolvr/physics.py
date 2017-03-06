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
import bisect
import numpy as np


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
        self.t = 0.0
        self.events = []
        self._ball_events = {}
        self.all_balls = np.array(range(self.num_balls))
        self.on_table = np.array(self.num_balls * [True])
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
        event = self.StrikeBallEvent(t, i, Q, V, cue_mass)
        events = [event]
        self._ball_events.clear()
        self._ball_events[i] = event
        while self._ball_events:
            predicted_event = None
            for event in self._ball_events.values():
                if predicted_event is None or (event.next_event and event.next_event.t < predicted_event.t):
                    predicted_event = event.next_event
            for i, event in sorted(list(self._ball_events.items())):
                if i != event.i: continue
                t_i = event.t
                _a, _b = event._calc_global_coeffs()
                for j in [j for j, on in enumerate(self.on_table)
                          if on and (j not in self._ball_events or j > i)]:
                    if j in self._ball_events:
                        j_event = self._ball_events[j]
                        if j_event == event:
                            if predicted_event is None or event.next_events[j].t < predicted_event.t:
                                predicted_event = event.next_events[j]
                            continue
                        _a_j, _b_j = j_event._calc_global_coeffs()
                        _a_j = _a_j[0] if j == j_event.i else _a_j[1]
                        _b_j = _b_j[0] if j == j_event.i else _b_j[1]
                        t0 = max(t_i, j_event.t)
                    else:
                        _a_j, _b_j = self._a[j], self._b[j]
                        t0 = t_i
                    t_c = self._find_collision_time(_a[0], _a_j, t0)
                    if t_c and t_c <= t_i + event.T and (predicted_event is None or t_c < predicted_event.t) \
                        and (j not in self._ball_events \
                             or (j in self._ball_events and t_c <= self._ball_events[j].t + self._ball_events[j].T)):
                        positions = np.empty((2,3), dtype=np.float32)
                        velocities = np.empty((2,3), dtype=np.float32)
                        tau_i = t_c - event.t
                        event.eval_positions(tau_i, out=positions[0].reshape(1,3))
                        event.eval_velocities(tau_i, out=velocities[0].reshape(1,3))
                        if j in self._ball_events:
                            e_j = self._ball_events[j]
                            tau_j = t_c - e_j.t
                            e_j.eval_positions(tau_j, out=positions[1].reshape(1,3))
                            e_j.eval_velocities(tau_j, out=velocities[1].reshape(1,3))
                        else:
                            positions[1] = self._a[j,0]
                            velocities[1] = self._a[j,1]
                        predicted_event = self.BallCollisionEvent(t_c, i, j, positions, velocities)
            if predicted_event is None:
                break
            determined_event = predicted_event
            events.append(determined_event)
            i = determined_event.i
            if determined_event == self._ball_events[i].next_event and isinstance(self._ball_events[i], self.BallCollisionEvent):
                self._ball_events[i].next_event = self._ball_events[i].next_events[1]
            if isinstance(determined_event, self.BallCollisionEvent):
                j = determined_event.j
                self._ball_events[i] = determined_event
                self._ball_events[j] = determined_event
            elif isinstance(determined_event, self.SlideToRollEvent):
                self._ball_events[i] = determined_event
            elif isinstance(determined_event, self.RollToRestEvent):
                if i in self._ball_events:
                    e_i = self._ball_events.pop(i)
                    _a, _b = e_i._calc_global_coeffs()
                    if isinstance(e_i, self.BallCollisionEvent):
                        ii = 0 if e_i.i == i else 1
                        self._a[i], self._b[i] = _a[ii], _b[ii]
                    else:
                        self._a[i], self._b[i] = _a, _b
        self.events += events
        return events

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls.tolist()
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float32)
        out[:] = self._a[balls,0]
        for e in self._find_active_events(t):
            tau = t - e.t
            if e._num_balls == 1:
                out[balls.index(e.i)] = e._a[0] + tau * e._a[1] + tau**2 * e._a[2]
            elif e._num_balls == 2:
                out[balls.index(e.i)] = e._a[0,0] + tau * e._a[0,1] + tau**2 * e._a[0,2]
                out[balls.index(e.j)] = e._a[1,0] + tau * e._a[1,1] + tau**2 * e._a[1,2]
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
            balls = self.all_balls.tolist()
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float32)
        out[:] = self._a[balls,1]
        for e in self._find_active_events(t):
            tau = t - e.t
            if e._num_balls == 1:
                out[balls.index(e.i)] = e._a[1] + 2 * tau * e._a[2]
            elif e._num_balls == 2:
                out[balls.index(e.i)] = e._a[0,1] + 2 * tau * e._a[0,2]
                out[balls.index(e.j)] = e._a[1,1] + 2 * tau * e._a[1,2]
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
        """
        Reset the state of the balls to at rest at the specified positions
        """
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
        n = bisect.bisect(self.events, t)
        return [e for e in self.events[:n] if e.t + e.T >= t]

    def _find_collision_time(self, a_i, a_j, t0):
        d = a_i - a_j
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
        _logger.info('roots = %s', roots)
        roots = [t.real for t in roots if t.real >= t0 and abs(t.imag / t.real) < 0.01]
        _logger.info('roots (filtered) = %s', roots)
        if not roots:
            return None
        else:
            return min(roots)

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
        def eval_velocities(self, tau, out=None):
            if self._num_balls == 0:
                return None
            if out is None:
                out = np.empty((self._num_balls, 3), dtype=np.float32)
            _a = self._a.reshape(self._num_balls, 3, 3)
            if hasattr(self, 'i'):
                out[0] = _a[0,1] + 2 * tau * _a[0,2]
            if hasattr(self, 'j'):
                out[1] = _a[1,1] + 2 * tau * _a[1,2]
            return out
        def predict_events(self):
            return [self.next_event]
        def _calc_global_coeffs(self):
            if self._num_balls == 0:
                return
            _a = self._a.reshape(self._num_balls, 3, 3)
            _b = self._b.reshape(self._num_balls, 2, 3)
            a = _a.copy()
            b = _b.copy()
            t = self.t
            a[:,0] += -t * _a[:,1] + t**2 * _a[:,2]
            a[:,1] += -2 * t * _a[:,2]
            return a, b
        def __str__(self):
            clsname = self.__class__.__name__.split('.')[-1]
            return "<%17s: t=%7.3f T=%7.3f" % (clsname, self.t, self.T) + \
                (' i=%2d'  % self.i if hasattr(self, 'i') else '') + \
                (' j=%2d>' % self.j if hasattr(self, 'j') else '>')
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
            # end_position = self._a[0] + tau_s * self._a[1] + tau_s**2 * self._a[2]
            # end_velocity = self._a[1] + 2 * tau_s * self._a[2]
            end_position = self.eval_positions(tau_s)[0]
            end_velocity = self.eval_velocities(tau_s)[0]
            self.next_event = self.physics.SlideToRollEvent(t + tau_s, i,
                                                            end_position, end_velocity)
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s v=%40s Q=%40s V=%40s>' % (self._a[0], self._a[1], self.Q, self.V)

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
            end_position = self.eval_positions(tau_r)[0]
            self.next_event = self.physics.RollToRestEvent(t + tau_r, i, end_position)
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s v=%40s>' % (self._a[0], self._a[1])

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
            self.T = float('inf')
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s>' % self._a[0]

    class BallCollisionEvent(PhysicsEvent):
        _num_balls = 2
        def __init__(self, t, i, j, positions, velocities):
            super().__init__(t)
            self.i = i
            self.j = j
            self._a = np.zeros((2,3,3), dtype=np.float32)
            self._b = np.zeros((2,2,3), dtype=np.float32)
            self._a[:,0] = positions
            v_i, v_j = velocities[0], velocities[1]
            v_ij = v_j - v_i
            norm_v_ij = np.linalg.norm(v_ij)
            delta_t = 284e-6 / norm_v_ij**0.294
            s_max = 1.65765 * (norm_v_ij / self.physics.c_b)**0.8
            F_max = 1.48001 * self.physics.ball_radius**2 * self.physics.E_Y_b * s_max**1.5
            J = 0.5 * F_max * delta_t
            r_i, r_j = positions
            r_ij = r_j - r_i
            _i = r_ij / np.linalg.norm(r_ij)
            post_velocities = self._a[:,1]
            post_velocities[0] = v_i - (J / self.physics.ball_mass) * _i
            post_velocities[1] = v_j + (J / self.physics.ball_mass) * _i
            self._a[0,2] = -0.5 * self.physics.mu_r * self.physics.g * (post_velocities[0] / np.linalg.norm(post_velocities[0]))
            self._a[1,2] = -0.5 * self.physics.mu_r * self.physics.g * (post_velocities[1] / np.linalg.norm(post_velocities[1]))
            tau_i = np.linalg.norm(post_velocities[0]) / (self.physics.mu_r * self.physics.g)
            tau_j = np.linalg.norm(post_velocities[1]) / (self.physics.mu_r * self.physics.g)
            tau_min = min(tau_i, tau_j)
            tau_max = max(tau_i, tau_j)
            self.T = tau_max
            argmin = np.argmin([tau_i, tau_j])
            self.next_event = self.physics.RollToRestEvent(t + tau_min, (i,j)[argmin], self.eval_positions(tau_min)[argmin])
            other_event = self.physics.RollToRestEvent(t + tau_max, (i,j)[1-argmin], self.eval_positions(tau_max)[1-argmin])
            self.next_events = [self.next_event, other_event]
