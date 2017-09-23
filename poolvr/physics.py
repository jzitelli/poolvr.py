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
from copy import copy
import numpy as np


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254

_I, _J, _K = np.eye(3, dtype=np.float64)


def _create_cue(cue_mass, cue_radius, cue_length):
    try:
        import ode
    except:
        from . import fake_ode as ode
    body = ode.Body(ode.World())
    mass = ode.Mass()
    mass.setCylinderTotal(cue_mass, 3, cue_radius, cue_length)
    body.setMass(mass)
    return body


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
                 use_simple_ball_collisions=False,
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
        self._I = 2.0/5 * ball_mass * ball_radius**2
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float64)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float64)
        if initial_positions is not None:
            self._a[:,0] = initial_positions
        if use_simple_ball_collisions:
            self.BallCollisionEvent = self.SimpleBallCollisionEvent
        self.t = 0.0

    def add_cue(self, cue):
        body = _create_cue(cue.mass, cue.radius, cue.length)
        self.cue_bodies = [body]
        self.cue_geoms = [body]
        return body, body

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
        self._add_event(event)
        ball_events = {i: event}
        collision_times = {}
        a_i = np.zeros((3,3), dtype=np.float64)
        b_i = np.zeros((2,3), dtype=np.float64)
        a_j = np.zeros((3,3), dtype=np.float64)
        b_j = np.zeros((2,3), dtype=np.float64)
        n_events = 1
        while ball_events:
            predicted_event = None
            for event in ball_events.values():
                if predicted_event is None or (event.next_event and event.next_event.t < predicted_event.t):
                    predicted_event = event.next_event
            balls_in_motion = sorted(ball_events.keys())
            # i is in motion due to the sequence of events initiated by the strike:
            for ii, i in enumerate(balls_in_motion):
                e_i = ball_events[i]
                t_i, T_i = e_i.t, e_i.T
                if predicted_event and t_i >= predicted_event.t:
                    continue
                # examine balls j that are in motion due to the sequence of events initiated by the strike:
                for j in balls_in_motion[ii+1:]:
                    e_j = ball_events[j]
                    t_j, T_j = e_j.t, e_j.T
                    if predicted_event and t_j >= predicted_event.t:
                        continue
                    if t_i > t_j + T_j:
                        continue
                    if t_j > t_i + T_i:
                        continue
                    ######################
                    # here we ignore j if both i and j's last determined event was colliding with each other
                    # ||
                    # V
                    if t_i == t_j and isinstance(e_i, self.BallCollisionEvent) and isinstance(e_j, self.BallCollisionEvent) and e_i.j == j and e_j.j == i:
                        continue
                    key = (e_i, e_j)
                    if key not in collision_times:
                        t0 = max(t_i, t_j)
                        t1 = min(t_i + T_i, t_j + T_j)
                        if t_i > t_j:
                            a_j[0] = e_j.eval_position(t_i - t_j) # j's position at t0
                            a_j[1] = e_j.eval_velocity(t_i - t_j) # j's velocity at t0
                            a_j[2] = e_j._a[2] # j's acceleration is constant over the course of the event
                            a_i[:] = e_i._a # i's event local kinematic equations of motion
                        else:
                            a_i[0] = e_i.eval_position(t_j - t_i) # i's position at t0
                            a_i[1] = e_i.eval_velocity(t_j - t_i) # i's velocity at t0
                            a_i[2] = e_i._a[2] # i's acceleration is constant over the course of the event
                            a_j[:] = e_j._a # j's event local kinematic equations of motion
                        # r_0ji = a_i[0] - a_j[0]
                        # v_0ji = a_i[1] - a_j[1]
                        # if ((abs(r_0ji) > 2*self.ball_radius) & (np.sign(r_0ji) == np.sign(v_0ji))).any() \
                        #   or np.linalg.norm(v_0ji) * (t1 - t0) < np.linalg.norm(r_0ji) - 2*self.ball_radius:
                        #     collision_times[key] = None
                        # else:
                        #     t_c = self._find_collision_time(a_i, a_j, 0.0, t1 - t0)
                        #     if t_c is not None:
                        #         t_c += t0
                        #     collision_times[key] = t_c
                        t_c = self._find_collision_time(a_i, a_j, 0.0, t1 - t0)
                        if t_c is not None:
                            t_c += t0
                        collision_times[key] = t_c
                    t_c = collision_times[key]
                    if t_c is not None:
                        if predicted_event is None or t_c < predicted_event.t:
                            r_i = e_i.eval_position(t_c - t_i)
                            v_i = e_i.eval_velocity(t_c - t_i)
                            r_j = e_j.eval_position(t_c - t_j)
                            v_j = e_j.eval_velocity(t_c - t_j)
                            # update predicted next event to a collision between i and j at game time t_c:
                            predicted_event = self.BallCollisionEvent(t_c, i, j, r_i, r_j, v_i, v_j)
                # examine stationary j that have moved but are now at rest, still on the table:
                for j, j_events in [(j, j_events) for j, j_events in self.ball_events.items()
                                    if j_events and j not in ball_events]:
                    if not self.on_table[j]:
                        continue
                    e_j = j_events[-1]
                    assert isinstance(e_j, self.RestEvent)
                    t_j, T_j = e_j.t, e_j.T
                    if predicted_event and t_j >= predicted_event.t:
                        continue
                    #######################
                    # if t_i > t_j + T_j: #
                    #     continue         <===  T_j on a RestEvent is +inf, so this will never be true
                    if t_j > t_i + T_i:
                        continue
                    key = (e_i, e_j)
                    if key not in collision_times:
                        t0 = max(t_i, t_j) # lower bound on the interval of valid collision times
                        t1 = min(t_i + T_i, t_j + T_j) # upper bound on the interval of valid collision times
                        a_j[0] = e_j._a[0] # j's position at t0
                        a_j[1] = 0 # j's velocity at t0
                        a_j[2] = 0 # j's acceleration is constant over the course of the event
                        if t_i > t_j:
                            a_i[:] = e_i._a
                        else:
                            a_i[0] = e_i.eval_position(t_j - t_i) # i's position at t0
                            a_i[1] = e_i.eval_velocity(t_j - t_i) # i's velocity at t0
                            a_i[2] = e_i._a[2] # blahb blabhab
                        # r_0ji = a_i[0] - a_j[0]
                        # v_0ji = a_i[1] - a_j[1]
                        # if ((abs(r_0ji) > 2*self.ball_radius) & (np.sign(r_0ji) == np.sign(v_0ji))).any() \
                        #   or np.linalg.norm(v_0ji) * (t1 - t0) < np.linalg.norm(r_0ji) - 2*self.ball_radius:
                        #     collision_times[key] = None
                        # else:
                        #     t_c = self._find_collision_time(a_i, a_j, 0.0, t1 - t0)
                        #     if t_c is not None:
                        #         t_c += t0
                        #     collision_times[key] = t_c
                        t_c = self._find_collision_time(a_i, a_j, 0.0, t1 - t0)
                        if t_c is not None:
                            t_c += t0
                        collision_times[key] = t_c
                    t_c = collision_times[key]
                    if t_c is not None:
                        if predicted_event is None or t_c < predicted_event.t:
                            r_i = e_i.eval_position(t_c - t_i)
                            v_i = e_i.eval_velocity(t_c - t_i)
                            r_j = e_j.eval_position(t_c - t_j)
                            v_j = e_j.eval_velocity(t_c - t_j)
                            predicted_event = self.BallCollisionEvent(t_c, i, j, r_i, r_j, v_i, v_j)
                # examine j that are still in initial stationary state:
                for j in [j for j, j_events in self.ball_events.items()
                          if not j_events]:
                    if not self.on_table[j]:
                        continue
                    t0 = t_i
                    t1 = t_i + T_i
                    a_j[:] = 0
                    a_j[0] = self._a[j,0] # j's initial stationary kinematic equations
                    a_i[:] = e_i._a
                    key = (e_i, j)
                    if key not in collision_times:
                        t_c = self._find_collision_time(a_i, a_j, 0.0, t1 - t0)
                        if t_c is not None:
                            t_c += t0
                        collision_times[key] = t_c
                    t_c = collision_times[key]
                    if t_c is not None:
                        if predicted_event is None or t_c < predicted_event.t:
                            r_i = e_i.eval_position(t_c - t_i)
                            v_i = e_i.eval_velocity(t_c - t_i)
                            r_j = a_j[0]
                            v_j = a_j[1]
                            predicted_event = self.BallCollisionEvent(t_c, i, j, r_i, r_j, v_i, v_j)
            if predicted_event is None:
                raise Exception('no event could be predicted, por que???')
            determined_event = predicted_event
            n_events += 1
            i = determined_event.i
            ball_events[i] = determined_event
            self._add_event(determined_event)
            if isinstance(determined_event, self.BallCollisionEvent):
                n_events += 1
                paired_event = determined_event.paired_event
                ball_events[paired_event.i] = paired_event
                self._add_event(paired_event)
            elif isinstance(determined_event, self.RestEvent):
                ball_events.pop(i)
        return n_events

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
        states = {i: self.STATIONARY for i in balls}
        for i in balls:
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        states[i] = e.state
                        break
        return states

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
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
            out = np.empty((self.num_balls, 4), dtype=np.float64)
        # TODO
        out[:] = 0
        out[:,3] = 1
        return out

    def eval_velocities(self, t, balls=None, out=None):
        """
        Evaluate the velocities of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
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
            out = np.empty((len(balls), 3), dtype=np.float64)
        out[:] = 0
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_angular_velocity(t - e.t)
                        break
        return out

    def eval_accelerations(self, t, balls=None, out=None):
        """
        Evaluate the linear accelerations of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = self.all_balls
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
        out[:] = 0
        for ii, i in enumerate(balls):
            events = self.ball_events[i]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e._a[2]
                        break
        return out

    def next_turn_time(self):
        """
        Return the time at which all balls have come to rest.
        """
        return self.events[-1].t if self.events else None

    def reset(self, ball_positions):
        """
        Reset the state of the balls to at rest at the specified positions
        """
        self.events = []
        self.on_table[:] = True
        self._a[:] = 0
        self._a[:,0] = ball_positions
        self._b[:] = 0
        self.ball_events = {i: [] for i in self.all_balls}

    def step(self, dt):
        self.t += dt

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def _add_event(self, event):
        self.events.append(event)
        i_events = self.ball_events[event.i]
        if i_events:
            prev = i_events[-1]
            if prev.next_event and prev.next_event != event:
                assert event.t < prev.t + prev.T
            prev.T = event.t - prev.t
            prev.next_event = None
        i_events.append(event)

    @staticmethod
    def _quartic_solve(p):
        # TODO: use analytic solution method (e.g. Ferrari)
        return np.roots(p)

    def _find_active_events(self, t):
        n = bisect(self.events, t)
        return [e for e in self.events[:n] if t >= e.t and t <= e.t + e.T]

    def _find_collision_time(self, a_i, a_j, t0, t1):
        d = a_i - a_j
        a_x, a_y = d[2, ::2]
        b_x, b_y = d[1, ::2]
        c_x, c_y = d[0, ::2]
        p = np.empty(5, dtype=np.float64)
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x*b_x + a_y*b_y)
        p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
        p[3] = 2 * b_x * c_x + 2 * b_y * c_y
        p[4] = c_x**2 + c_y**2 - 4 * self.ball_radius**2
        roots = self._quartic_solve(p)
        roots = [t.real for t in roots if t0 < t.real and t.real < t1 and abs(t.imag) / np.sqrt(t.real**2+t.imag**2) < 0.01]
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
            self.T = float('inf')
            self.next_event = None
            self._a = np.zeros((3,3), dtype=np.float64)
            self._b = np.zeros((2,3), dtype=np.float64)
        def eval_position(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float64)
            _a = self._a
            out[:] = _a[0] + tau * _a[1] + tau**2 * _a[2]
            return out
        def eval_velocity(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float64)
            _a = self._a
            out[:] = _a[1] + 2 * tau * _a[2]
            return out
        def eval_angular_velocity(self, tau, out=None):
            if out is None:
                out = np.empty(3, dtype=np.float64)
            _b = self._b
            out[:] = _b[0] + tau * _b[1]
            return out
        def _calc_global_coeffs(self):
            _a, _b, t = self._a, self._b, self.t
            a = _a.copy()
            b = _b.copy()
            a[0] += -t * _a[1] + t**2 * _a[2]
            a[1] += -2 * t * _a[2]
            b[0] += -t * _b[1]
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
            if isinstance(other, self.physics.PhysicsEvent):
                return self.t == other.t and self.T == other.T and self.i == other.i
        def __hash__(self):
            return hash((self.i, self.t, self.T))

    class StrikeBallEvent(PhysicsEvent):
        def __init__(self, t, i, Q, V, cue_mass):
            super().__init__(t, i)
            self.Q = Q
            self.V = V
            self.cue_mass = cue_mass
            i_events = self.physics.ball_events[i]
            if i_events:
                self._a[0] = i_events[-1]._a[0]
            else:
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

    class RestEvent(PhysicsEvent):
        def __init__(self, t, i, position):
            super().__init__(t, i)
            self._a[0] = position
            self.T = float('inf')
            self.state = self.physics.STATIONARY
        def eval_position(self, tau, out=None):
            if out is None:
                out = self._a[0].copy()
            else:
                out[:] = self._a[0]
            return out
        def eval_velocity(self, tau, out=None):
            if out is None:
                out = np.zeros(3, dtype=np.float64)
            else:
                out[:] = 0
            return out
        def __str__(self):
            return super().__str__()[:-1] + ' r=%40s>' % self._a[0]

    class SlideToRestEvent(RestEvent):
        pass

    class RollToRestEvent(RestEvent):
        pass

    class PositionBallEvent(RestEvent):
        pass

    class SimpleBallCollisionEvent(PhysicsEvent):
        def __init__(self, t, i, j, r_i, r_j, v_i, v_j,
                     e=0.93):
            super().__init__(t, i)
            self.j = j
            self.r_i = r_i
            self.r_j = r_j
            self.v_i = v_i
            self.v_j = v_j
            self._a[0] = r_i
            v_ij = v_j - v_i
            norm_v_ij = np.linalg.norm(v_ij)
            delta_t = 284e-6 / norm_v_ij**0.294
            s_max = 1.65765 * (norm_v_ij / self.physics.c_b)**0.8
            F_max = 1.48001 * self.physics.ball_radius**2 * self.physics.E_Y_b * s_max**1.5
            J = 0.5 * F_max * delta_t
            r_ij = r_j - r_i
            _i = r_ij / np.linalg.norm(r_ij)
            norm_v_i_i = v_i.dot(_i)
            v_i_i = norm_v_i_i * _i
            norm_v_j_i = v_j.dot(_i)
            v_j_i = norm_v_j_i * _i
            v_ij_i = v_j_i - v_i_i
            norm_v_ij_i = abs(norm_v_j_i - norm_v_i_i)
            post_v_i = self._a[1]
            post_v_i[:] = (v_i - v_i_i) + e * norm_v_j_i / norm_v_ij_i * v_ij_i
            norm_post_v_i = np.linalg.norm(post_v_i)
            self._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (post_v_i / norm_post_v_i)
            self.state = self.physics.ROLLING
            self.T = norm_post_v_i / (self.physics.mu_r * self.physics.g)
            self.next_event = self.physics.RollToRestEvent(t + self.T, i, self.eval_position(self.T))
            paired_event = copy(self)
            self.paired_event = paired_event
            paired_event.paired_event = self
            paired_event.i, paired_event.j = j, i
            paired_event._a = np.zeros((3,3), dtype=np.float64)
            paired_event._b = np.zeros((2,3), dtype=np.float64)
            paired_event._a[0] = r_j
            post_v_j = paired_event._a[1]
            post_v_j[:] = (v_j - v_j_i) - e * norm_v_i_i / norm_v_ij_i * v_ij_i
            norm_post_v_j = np.linalg.norm(post_v_j)
            paired_event._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (post_v_j / norm_post_v_j)
            paired_event.T = norm_post_v_j / (self.physics.mu_r * self.physics.g)
            paired_event.next_event = self.physics.RollToRestEvent(t + paired_event.T, j,
                                                                   paired_event.eval_position(paired_event.T))
        def __str__(self):
            return super().__str__()[:-1] + ' j=%2d>' % self.j

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
            paired_event._a = np.zeros((3,3), dtype=np.float64)
            paired_event._b = np.zeros((2,3), dtype=np.float64)
            paired_event._a[0] = r_j
            post_v_j = paired_event._a[1]
            post_v_j[:] = v_j + (J / self.physics.ball_mass) * _i
            paired_event._a[2] = -0.5 * self.physics.mu_r * self.physics.g * (post_v_j / np.linalg.norm(post_v_j))
            tau_j = np.linalg.norm(post_v_j) / (self.physics.mu_r * self.physics.g)
            paired_event.T = tau_j
            paired_event.next_event = self.physics.RollToRestEvent(t + tau_j, j, paired_event.eval_position(tau_j))
        def __str__(self):
            return super().__str__()[:-1] + ' j=%2d>' % self.j
