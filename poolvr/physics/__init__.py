"""
This package implements an event-based pool physics simulator based on the paper
(available at http://web.stanford.edu/group/billiards/AnEventBasedPoolPhysicsSimulator.pdf): ::

  AN EVENT-BASED POOL PHYSICS SIMULATOR
  Will Leckie, Michael Greenspan
  DOI: 10.1007/11922155_19 · Source: DBLP
  Conference: Advances in Computer Games, 11th International Conference,
  Taipei, Taiwan, September 6-9, 2005.

"""
from itertools import chain
from bisect import bisect
from time import perf_counter
import numpy as np


from ..table import PoolTable
from .events import (CueStrikeEvent,
                     BallEvent,
                     BallStationaryEvent,
                     BallRestEvent,
                     BallMotionEvent,
                     BallCollisionEvent,
                     MarlowBallCollisionEvent,
                     SimpleBallCollisionEvent,
                     RailCollisionEvent)


PIx2 = np.pi*2
RAD2DEG = 180/np.pi
INCH2METER = 0.0254
SQRT2 = np.sqrt(2.0)
CUBE_ROOTS_OF_1_ANGLES = PIx2/3 * np.arange(3)
CUBE_ROOTS_OF_1 = np.exp(1j*CUBE_ROOTS_OF_1_ANGLES)


class PoolPhysics(object):
    _ZERO_TOLERANCE = 1e-8
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
    _IMAG_TOLERANCE = 1e-7
    _IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2
    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.2,
                 mu_b=0.06,
                 g=9.81,
                 balls_on_table=None,
                 ball_positions=None,
                 ball_collision_model="simple",
                 ball_collision_model_kwargs=None,
                 table=None,
                 enable_sanity_check=True,
                 enable_occlusion=True,
                 realtime=False,
                 collision_search_time_limit=0.2/90,
                 collision_search_time_forward=0.2,
                 **kwargs):
        r"""
        Pool physics simulator

        :param mu_r:  :math:`\mu_r`,    rolling friction coefficient
        :param mu_sp: :math:`\mu_{sp}`, spinning friction coefficient
        :param mu_s:  :math:`\mu_s`,    sliding friction coefficient
        :param mu_b:  :math:`\mu_b`,    ball-to-ball collision friction coefficient
        :param c_b:   :math:`c_b`,      ball material's speed of sound
        :param E_Y_b: :math:`{E_Y}_b`,  ball material's Young's modulus
        :param g:     :math:`g`,        downward acceleration due to gravity
        """
        if ball_collision_model == 'simple':
            self._ball_collision_event_class = SimpleBallCollisionEvent
        elif ball_collision_model == 'marlow':
            self._ball_collision_event_class = MarlowBallCollisionEvent
        else:
            raise Exception('dont know that collision model!')
        self.num_balls = num_balls
        # allocate for lower-level memory management:
        self._BALL_REST_EVENTS = [BallRestEvent(0.0, i, r=np.zeros(3, dtype=np.float64))
                                  for i in range(self.num_balls)]
        if table is None:
            table = PoolTable(num_balls=num_balls, ball_radius=ball_radius)
        self.table = table
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.ball_diameter = 2*ball_radius
        self.ball_I = 2/5 * ball_mass * ball_radius**2 # moment of inertia
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.mu_b = mu_b
        self.g = g
        self.t = 0.0
        self._balls_on_table = balls_on_table
        self._balls_at_rest = set(balls_on_table)
        self._on_table = np.array(self.num_balls * [False])
        self._on_table[np.array(balls_on_table, dtype=np.int32)] = True
        self._realtime = realtime
        self._collision_search_time_limit = collision_search_time_limit
        self._collision_search_time_forward = collision_search_time_forward
        self._enable_occlusion = enable_occlusion
        self._enable_sanity_check = enable_sanity_check
        self._p = np.empty(5, dtype=np.float64)
        self._mask = np.array(4*[True])
        self._sx = 0.5*table.W
        self._sz = 0.5*table.L
        self._rhsx = self._sx - ball_radius
        self._rhsz = self._sz - ball_radius
        self._bndx = self._sx - 0.999*ball_radius
        self._bndz = self._sz - 0.999*ball_radius
        self._sxcp = self._sx - table.M_cp/SQRT2
        self._szcp = self._sz - table.M_cp/SQRT2
        self._rail_tuples = ((2, -self._rhsz, self._bndx, self._sxcp),
                             (0,  self._rhsx, self._bndz, self._szcp),
                             (2,  self._rhsz, self._bndx, self._sxcp),
                             (0, -self._rhsx, self._bndz, self._szcp))
        self._a_ij = np.zeros((self.num_balls, 3), dtype=np.float64)
        self._a_ij_mag = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._r_ij = np.zeros((self.num_balls, self.num_balls, 3), dtype=np.float64)
        self._r_ij_mag = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._theta_ij = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._psi_ij = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._occ_ij = np.array(self.num_balls*[self.num_balls*[False]])
        self._velocity_meshes = None
        self._angular_velocity_meshes = None
        if ball_collision_model_kwargs:
            self._ball_collision_model_kwargs = ball_collision_model_kwargs
        else:
            self._ball_collision_model_kwargs = {}
        self.reset(ball_positions=ball_positions, balls_on_table=balls_on_table)

    def reset(self, ball_positions=None, balls_on_table=None):
        """
        Reset the state of the balls to at rest, at the specified positions.
        """
        self._ball_motion_events = {}
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.balls_on_table = balls_on_table
        if ball_positions is None:
            ball_positions = self.table.calc_racked_positions()[self.balls_on_table]
        else:
            ball_positions = ball_positions[self.balls_on_table]
        self.t = 0.0
        for ii, i in enumerate(self.balls_on_table):
            e = self._BALL_REST_EVENTS[i]
            e._r_0[:] = ball_positions[ii]
            e.t = self.t
            e.T = float('inf')
        self.ball_events = {i: [self._BALL_REST_EVENTS[i]]
                            for i in self.balls_on_table}
        self.events = list(chain.from_iterable(self.ball_events.values()))
        self._a_ij[:] = 0
        self._a_ij_mag[:] = 0
        # update occlusion buffers:
        self._occ_ij[:] = False
        self._occ_ij[range(self.num_balls), range(self.num_balls)] = True
        self._r_ij[:] = 0
        self._r_ij_mag[:] = 0
        self._psi_ij[:] = 0
        self._theta_ij[:] = 0
        self._balls_at_rest = set(self.balls_on_table)
        self._collisions = {}
        if self._realtime:
            self._find_collisions = False
        if self._enable_occlusion:
            self._update_occlusion({e.i: e._r_0 for e in self._BALL_REST_EVENTS})

    @property
    def ball_collision_model(self):
        return 'marlow' if self._ball_collision_event_class is MarlowBallCollisionEvent else 'simple'
    @ball_collision_model.setter
    def ball_collision_model(self, model='simple'):
        if model == 'marlow':
            self._ball_collision_event_class = MarlowBallCollisionEvent
        else:
            self._ball_collision_event_class = SimpleBallCollisionEvent

    @property
    def balls_on_table(self):
        return self._balls_on_table
    @balls_on_table.setter
    def balls_on_table(self, balls):
        self._balls_on_table = np.array(balls, dtype=np.int32)
        self._balls_on_table.sort()
        self._on_table[:] = False
        self._on_table[self._balls_on_table] = True

    @property
    def balls_in_motion(self):
        return list(self._ball_motion_events.keys())

    @property
    def balls_at_rest(self):
        return self._balls_at_rest

    def add_cue(self, cue):
        self.cues = [cue]

    def strike_ball(self, t, i, r_i, r_c, V, M):
        r"""
        Strike ball *i* at game time *t*.

        :param r_i: position of ball *i*
        :param r_c: point of contact
        :param V: impact velocity
        :param M: impact mass
        """
        if not self._on_table[i]:
            return
        #assert abs(np.linalg.norm(r_c - r_i) - self.ball_radius) < self._ZERO_TOLERANCE, 'abs(np.linalg.norm(r_c - r_i) - self.ball_radius) = %s' % abs(np.linalg.norm(r_c - r_i) - self.ball_radius)
        event = CueStrikeEvent(t, i, r_i, r_c, V, M)
        if self._realtime:
            self._find_collisions = True
            return self.add_event_sequence_realtime(event)
        else:
            return self.add_event_sequence(event)

    def add_event_sequence(self, event):
        num_events = len(self.events)
        self._add_event(event)
        while self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    def add_event_sequence_realtime(self, event):
        num_events = len(self.events)
        self._add_event(event)
        T, T_f = self._collision_search_time_limit, self._collision_search_time_forward
        lt = perf_counter()
        while T > 0 and self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
            if event.t - self.t > T_f:
                break
            t = perf_counter()
            T -= t - lt; lt = t
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    @property
    def next_turn_time(self):
        """The time at which all balls have come to rest."""
        return self.events[-1].t \
            if self.events and isinstance(self.events[-1], BallRestEvent) else 0.0

    def step(self, dt):
        if self._realtime:
            self._find_collisions = self.step_realtime(dt, find_collisions=self._find_collisions)
        else:
            self.t += dt

    def step_realtime(self, dt,
                      find_collisions=True):
        self.t += dt
        if not find_collisions:
            return False
        T = self._collision_search_time_limit
        t_max = self.t + self._collision_search_time_forward
        lt = perf_counter()
        while T > 0 and self.balls_in_motion:
            event = self._determine_next_event()
            if event:
                self._add_event(event)
                if event.t >= t_max:
                    return bool(self.balls_in_motion)
            t = perf_counter()
            dt = t - lt; lt = t
            T -= dt
        if T <= 0:
            return bool(self.balls_in_motion)

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_position(t - e.t)
                        break
        return out

    def eval_velocities(self, t, balls=None, out=None):
        """
        Evaluate the velocities of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
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
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_angular_velocity(t - e.t)
                        break
        return out

    def eval_energy(self, t, balls=None, out=None):
        if balls is None:
            balls = self.balls_on_table
        if out is None:
            out = np.zeros(len(balls), dtype=np.float64)
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return 0.5 * self.ball_mass * (velocities**2).sum() + 0.5 * self.ball_I * (omegas**2).sum()

    def find_active_events(self, t):
        active_events = []
        for i, events in self.ball_events.items():
            for e in events[:bisect(events, t)][::-1]:
                if t <= e.t + e.T:
                    active_events.append(e)
                    break
        return active_events

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def _add_event(self, event):
        self.events.append(event)
        if isinstance(event, RailCollisionEvent):
            event.e_i.T = event.t - event.e_i.t
        elif isinstance(event, BallCollisionEvent):
            event.e_i.T = event.t - event.e_i.t
            event.e_j.T = event.t - event.e_j.t
        if isinstance(event, BallEvent):
            i = event.i
            self._collisions.pop(i, None)
            for v in self._collisions.values():
                v.pop(i, None)
            if self.ball_events[i]:
                last_ball_event = self.ball_events[i][-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T = event.t - last_ball_event.t
            self.ball_events[i].append(event)
            if isinstance(event, BallStationaryEvent):
                if i in self._ball_motion_events:
                    self._ball_motion_events.pop(i)
                self._a_ij[i] = 0
                self._a_ij_mag[i,i] = 0
                self._a_ij_mag[i,:] = self._a_ij_mag[:,i] = self._a_ij_mag.diagonal()
                self._balls_at_rest.add(event.i)
                if self._enable_occlusion and isinstance(event, BallStationaryEvent):
                    self._update_occlusion({i: event._r_0})
            elif isinstance(event, BallMotionEvent):
                self._ball_motion_events[i] = event
                self._a_ij[i] = event.acceleration
                self._a_ij_mag[i,:] = self._a_ij_mag[:,i] = np.linalg.norm(self._a_ij - event.acceleration, axis=1)
                self._a_ij_mag[i,i] = np.sqrt(np.dot(event.acceleration, event.acceleration))
                if event.i in self._balls_at_rest:
                    self._balls_at_rest.remove(event.i)
        for child_event in event.child_events:
            self._add_event(child_event)
        if self._enable_sanity_check and isinstance(event, BallCollisionEvent):
            self._sanity_check(event)

    def _determine_next_event(self):
        next_motion_event = min(e.next_motion_event
                                for e in self._ball_motion_events.values()
                                if e.next_motion_event is not None)
        if next_motion_event:
            t_min = next_motion_event.t
        else:
            t_min = float('inf')
        next_collision = None
        next_rail_collision = None
        for i in sorted(self.balls_in_motion):
            if i not in self._collisions:
                self._collisions[i] = {}
            collisions = self._collisions[i]
            e_i = self.ball_events[i][-1]
            rail_collision = self._find_rail_collision(e_i, t_min)
            if rail_collision and rail_collision[0] < t_min:
                t_min = rail_collision[0]
                next_rail_collision = rail_collision
            for j in self.balls_on_table:
                if j <= i and j in self.balls_in_motion:
                    continue
                e_j = self.ball_events[j][-1]
                if j not in collisions:
                    if self._enable_occlusion and isinstance(e_j, BallStationaryEvent) and self._occ_ij[i,j]:
                        t_c = None
                    else:
                        t_c = self._find_collision(e_i, e_j, t_min)
                    collisions[j] = t_c
                t_c = collisions[j]
                if t_c is not None and t_c < t_min:
                    t_min = t_c
                    next_collision = (t_c, e_i, e_j)
                    next_rail_collision = None
        if next_rail_collision:
            t, i, side = next_rail_collision
            return RailCollisionEvent(t=t, e_i=self.ball_events[i][-1], side=side)
        if next_collision is not None:
            t_c, e_i, e_j = next_collision
            return self._ball_collision_event_class(t_c, e_i, e_j,
                                                    **self._ball_collision_model_kwargs)
        else:
            return next_motion_event

    def _find_rail_collision(self, e_i, t_min=None):
        a = e_i._a
        if e_i.parent_event and isinstance(e_i.parent_event, RailCollisionEvent):
            prev_side = e_i.parent_event.side
        else:
            prev_side = None
        if t_min is None:
            tau_min = e_i.T
        elif e_i.t >= t_min:
            return None
        else:
            tau_min = t_min - e_i.t
        side_min = None
        for side, (j, rhs, bnd, bnd_cp) in enumerate(self._rail_tuples):
            if side == prev_side:
                continue
            k = 2 - j
            if abs(a[2,j]) < 1e-15:
                if abs(a[1,j]) > 1e-15:
                    tau = (rhs - a[0,j]) / a[1,j]
                    if 0 < tau < tau_min:
                        r = e_i.eval_position(tau)
                        if abs(r[k]) < bnd:
                            tau_min = tau
                            side_min = side
            else:
                d = a[1,j]**2 - 4*a[2,j]*(a[0,j] - rhs)
                if d > 1e-15:
                    pn = np.sqrt(d)
                    tau_p = (-a[1,j] + pn) / (2*a[2,j])
                    tau_n = (-a[1,j] - pn) / (2*a[2,j])
                    if 0 < tau_n < tau_min and 0 < tau_p < tau_min:
                        tau_a, tau_b = min(tau_n, tau_p), max(tau_n, tau_p)
                        r = e_i.eval_position(tau_a)
                        if abs(r[k]) < bnd:
                            tau_min = tau_a
                            side_min = side
                        else:
                            r = e_i.eval_position(tau_b)
                            if abs(r[k]) < bnd:
                                tau_min = tau_b
                                side_min = side
                    elif 0 < tau_n < tau_min:
                        r = e_i.eval_position(tau_n)
                        if abs(r[k]) < bnd:
                            tau_min = tau_n
                            side_min = side
                    elif 0 < tau_p < tau_min:
                        r = e_i.eval_position(tau_p)
                        if abs(r[k]) < bnd:
                            tau_min = tau_p
                            side_min = side
        if side_min is not None:
            return (e_i.t + tau_min, e_i.i, side_min)

    def _find_collision(self, e_i, e_j, t_min):
        if e_j.parent_event and e_i.parent_event and e_j.parent_event == e_i.parent_event:
            return None
        t0 = max(e_i.t, e_j.t)
        t1 = min(e_i.t + e_i.T, e_j.t + e_j.T, t_min)
        if t0 >= t1:
            return None
        tau_i_0, tau_j_0 = t0 - e_i.t, t0 - e_j.t
        a_ij_mag = self._a_ij_mag[e_i.i, e_j.i]
        v_ij_0 = e_i.eval_velocity(tau_i_0) - e_j.eval_velocity(tau_j_0)
        r_ij_0 = e_i.eval_position(tau_i_0) - e_j.eval_position(tau_j_0)
        if   np.sqrt(v_ij_0.dot(v_ij_0))*(t1-t0) + 0.5*a_ij_mag*(t1-t0)**2 \
           < np.sqrt(r_ij_0.dot(r_ij_0)) - self.ball_diameter:
            return None
        a_i, b_i = e_i.global_motion_coeffs
        a_j, b_j = e_j.global_motion_coeffs
        return self._find_collision_time(a_i, a_j, t0, t1)

    def _find_collision_time(self, a_i, a_j, t0, t1):
        d = a_i - a_j
        a_x, a_y = d[2, ::2]
        b_x, b_y = d[1, ::2]
        c_x, c_y = d[0, ::2]
        p = self._p
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x*b_x + a_y*b_y)
        p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
        p[3] = 2 * b_x*c_x + 2 * b_y*c_y
        p[4] = c_x**2 + c_y**2 - 4 * self.ball_radius**2
        # try:
        #     return self._filter_roots(np.roots(p), t0, t1)
        # except np.linalg.linalg.LinAlgError as err:
        #     # _logger.warning('LinAlgError occurred during solve for collision time:\np = %s\nerror:\n%s', p, err)
        #     pass
        return self._filter_roots(self.quartic_solve(p[::-1]), t0, t1)

    @classmethod
    def quartic_solve(cls, p):
        if abs(p[-1]) / max(abs(p[:-1])) < cls._ZERO_TOLERANCE:
            return cls.cubic_solve(p[:-1])
        e, d, c, b, a = p
        Delta = 256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 144*a**2*c*d**2*e - 27*a**2*d**4 \
              + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 16*a*c**4*e \
              - 4*a*c**3*d**2 - 27*b**4*e**2 + 18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2
        P = 8*a*c - 3*b**2
        D = 64*a**3*e - 16*a**2*c**2 + 16*a*b**2*c - 16*a**2*b*d - 3*b**4
        if Delta > 0 and (P > 0 or D > 0):
            # _logger.debug('all roots are complex and distinct')
            return np.empty(0)
        R = (b**3 - 4*a*b*c + 8*a**2*d)
        Delta_0 = c**2 - 3*b*d + 12*a*e
        if Delta == 0 and D == 0:
            if P > 0 and R == 0:
                # _logger.debug('two complex-conjugate double roots')
                return np.empty(0)
            elif Delta_0 == 0:
                # _logger.debug('all roots are equal to -b / 4a')
                return np.array([-0.25 * b / a])
        Delta_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
        p = P / (8*a**2)
        q = R / (8*a**3)
        if Delta > 0:
            QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta + 0j)))
        else:
            QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta)))
        if Delta > 0:
            # _logger.debug('all roots are real and distinct')
            Q = QQQ**(1.0/3)
        elif Delta < 0:
            # _logger.debug('two distinct real roots and a complex-conjugate pair of roots')
            angle = np.angle(QQQ) / 3
            if abs(angle) < 1e-8:
                Q_mag = abs(QQQ)**(1.0/3)
                Q = Q_mag * np.exp(1j*(angle + CUBE_ROOTS_OF_1_ANGLES[1]))
            else:
                Q = QQQ**(1.0/3)
        elif Delta == 0:
            if P < 0 and D < 0 and Delta_0 != 0:
                # _logger.debug('one real double root and two other real roots')
                Q = QQQ**(1.0/3)
            elif D > 0 or (P > 0 and (D != 0 or R != 0)):
                # _logger.debug('one real double root and a complex-conjugate pair of roots')
                angle = np.angle(QQQ) / 3
                if abs(angle) < 1e-8:
                    Q_mag = abs(QQQ)**(1.0/3)
                    Q = Q_mag * np.exp(1j*(angle + CUBE_ROOTS_OF_1_ANGLES[1]))
                else:
                    Q = QQQ**(1.0/3)
            elif Delta_0 == 0 and D != 0:
                # _logger.debug('one real triple root and one other real root')
                Q = QQQ**(1.0/3)
            elif D == 0 and P < 0:
                # _logger.debug('two real double roots')
                Q = QQQ**(1.0/3)
        SSx4 = -2.0*p/3 + (Q + Delta_0/Q) / (3.0*a)
        S = 0.5*np.sqrt(SSx4 if SSx4 >= 0 else SSx4 + 0j)
        sqrp = -SSx4 - 2*p + q/S
        sqrm = -SSx4 - 2*p - q/S
        sqrtp = np.sqrt(sqrp if sqrp >= 0 else sqrp + 0j)
        sqrtm = np.sqrt(sqrm if sqrm >= 0 else sqrm + 0j)
        return np.array((
            -b/(4*a) - S + 0.5*sqrtp,
            -b/(4*a) - S - 0.5*sqrtp,
            -b/(4*a) + S + 0.5*sqrtm,
            -b/(4*a) + S - 0.5*sqrtm,
        ))

    @classmethod
    def cubic_solve(cls, p):
        if abs(p[-1]) / max(abs(p[:-1])) < cls._ZERO_TOLERANCE:
            return cls.quadratic_solve(p[:-1])
        a, b, c, d = p[::-1]
        Delta = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        Delta_0 = b**2 - 3*a*c
        if Delta == 0:
            if Delta_0 == 0:
                return np.array((-b/(3*a)))
            else:
                return np.array((              (9*a*d - b*c) / (2*Delta_0),
                                 (4*a*b*c - 9*a**2*d - b**3) / (a*Delta_0)))
        Delta_1 = 2*b**3 - 9*a*b*c + 27*a**2*d
        DD = -27*a**2*Delta
        CCC = 0.5 * (Delta_1 + np.sign(Delta_1)*np.sqrt(DD if DD >= 0 else DD + 0j))
        if CCC == 0:
            return np.array((-b/(3*a)))
        C = CCC**(1.0/3)
        return -(b + CUBE_ROOTS_OF_1*C + Delta_0 / (CUBE_ROOTS_OF_1*C)) / (3*a)

    @classmethod
    def quadratic_solve(cls, p):
        if abs(p[2]) / max(abs(p[:2])) < cls._ZERO_TOLERANCE:
            return np.array([-p[0] / p[1]])
        else:
            c, b, a = p
            sqrtd = np.sqrt(b**2 - 4*a*c)
            return np.array([(-b + sqrtd)/(2*a),
                             (-b - sqrtd)/(2*a)])

    def _filter_roots(self, roots, t0, t1):
        # filter out possible complex-conjugate pairs of roots:
        if len(roots) == 0:
            return None
        mask = self._mask; mask[:] = True
        for n in range(2):
            i, z = next(((i, z) for i, z in enumerate(roots)
                         if abs(z.imag) > PoolPhysics._IMAG_TOLERANCE),
                        (None, None))
            if z is not None:
                j = next((j for j, z_conj in enumerate(roots[i+1:])
                          if  abs(z.real - z_conj.real) < PoolPhysics._ZERO_TOLERANCE
                          and abs(z.imag + z_conj.imag) < PoolPhysics._ZERO_TOLERANCE),
                         None)
                if j is not None:
                    mask[i] = False; mask[i+j+1] = False
                    roots = roots[mask[:len(roots)]]
                else:
                    break
            else:
                break
        return min((t.real for t in roots
                    if t0 <= t.real <= t1
                    and t.imag**2 / (t.real**2 + t.imag**2) < self._IMAG_TOLERANCE_SQRD),
                   default=None)

    def _update_occlusion(self, ball_positions=None):
        r_ij = self._r_ij
        r_ij_mag = self._r_ij_mag
        thetas_ij = self._theta_ij
        psi_ij = self._psi_ij
        occ_ij = self._occ_ij
        if ball_positions is None:
            ball_positions = {}
        balls, positions = zip(*ball_positions.items())
        balls = np.array(balls, dtype=np.int32)
        argsort = balls.argsort()
        positions = np.array(positions)[argsort]
        U = balls[argsort]
        r_ij[U,U] = positions
        R = np.array([i for i in self.balls_at_rest
                      if i not in U], dtype=np.int32); R.sort()
        M = np.array(self.balls_in_motion, dtype=np.int32)
        occ_ij[M,:] = occ_ij[:,M] = False
        occ_ij[M,M] = True
        if len(R) > 0:
            F = np.hstack((U, R))
        else:
            F = U
        for ii, i in enumerate(U):
            F_i = F[ii+1:]
            if len(F_i) == 0:
                continue
            r_ij[i,F_i] = r_ij[F_i,F_i] - r_ij[i,i]
            r_ij_mag[i,F_i] = np.linalg.norm(r_ij[i,F_i], axis=1)
            thetas_ij[i,F_i] = np.arctan2(r_ij[i,F_i,2], r_ij[i,F_i,0])
            psi_ij[i,F_i] = np.arcsin(self.ball_diameter / r_ij_mag[i,F_i])
            jj_sorted = r_ij_mag[i,F_i].argsort()
            j_sorted = F_i[jj_sorted]
            theta_i = thetas_ij[i,j_sorted]
            psi_i = psi_ij[i,j_sorted]
            theta_i_a = theta_i - psi_i
            theta_i_b = theta_i + psi_i
            theta_i_occ_bnds = []
            for j, theta_ij_a, theta_ij, theta_ij_b in zip(j_sorted,
                                                           theta_i_a, theta_i, theta_i_b):
                if j == i:
                    continue
                if theta_ij_a < -np.pi:
                    thetas  = [(theta_ij_a,      theta_ij,      theta_ij_b),
                               (theta_ij_a+PIx2, theta_ij+PIx2, theta_ij_b+PIx2)]
                elif theta_ij_b >= np.pi:
                    thetas = [(theta_ij_a,      theta_ij,      theta_ij_b),
                              (theta_ij_a-PIx2, theta_ij-PIx2, theta_ij_b-PIx2)]
                else:
                    thetas = [(theta_ij_a, theta_ij, theta_ij_b)]
                for theta_a, theta, theta_b in thetas:
                    jj_a, jj, jj_b = (bisect(theta_i_occ_bnds, theta_a),
                                      bisect(theta_i_occ_bnds, theta),
                                      bisect(theta_i_occ_bnds, theta_b))
                    # jj = bisect(theta_i_occ_bnds, theta)
                    # jja = jj
                    # while theta_a < theta_i_occ_bnds[max(0, min(len(theta_i_occ_bnds)-1,jja))]:
                    #     jja -= 1
                    # jj_a = jja
                    # jjb = jj
                    # while theta_b >= theta_i_occ_bnds[max(0, min(len(theta_i_occ_bnds)-1,jjb))]:
                    #     jjb += 1
                    # jj_b = jjb
                    center_occluded, a_occluded, b_occluded = jj % 2 == 1, jj_a % 2 == 1, jj_b % 2 == 1
                    if center_occluded and jj_a == jj == jj_b:
                        if not occ_ij[i,j]:
                            pass
                        occ_ij[i,j] = occ_ij[j,i] = True
                        break
                    if a_occluded and b_occluded:
                        theta_i_occ_bnds = theta_i_occ_bnds[:jj_a] + theta_i_occ_bnds[jj_b:]
                    elif a_occluded:
                        theta_i_occ_bnds = theta_i_occ_bnds[:jj_a] + [theta_b] + theta_i_occ_bnds[jj_b:]
                    elif b_occluded:
                        theta_i_occ_bnds = theta_i_occ_bnds[:jj_a] + [theta_a] + theta_i_occ_bnds[jj_b:]
                    else:
                        theta_i_occ_bnds = theta_i_occ_bnds[:jj_a] + [theta_a, theta_b] + theta_i_occ_bnds[jj_b:]

    def _sanity_check(self, event):
        import pickle
        class Insanity(Exception):
            def __init__(self, physics, *args, **kwargs):
                with open('%s.pickle.dump' % self.__class__.__name__, 'wb') as f:
                    pickle.dump(physics, f)
                super().__init__(*args, **kwargs)

        if isinstance(event, BallCollisionEvent):
            e_i, e_j = event.child_events
            for t in np.linspace(event.t, event.t + min(e_i.T, e_j.T), 20):
                if e_i.t + e_i.T >= t or e_j.t + e_j.T >= t:
                    r_i, r_j = self.eval_positions(t, balls=[event.i, event.j])
                    d_ij = np.linalg.norm(r_i - r_j)
                    if d_ij - 2*self.ball_radius < -1e-6:
                        class BallsPenetratedInsanity(Insanity):
                            pass
                        raise BallsPenetratedInsanity(self, '''
ball_diameter = %s
         d_ij = %s
          r_i = %s
          r_j = %s
            t = %s
event: %s
  e_i: %s
  e_j: %s
''' % (2*self.ball_radius, d_ij, r_i, r_j, self.t, event, e_i, e_j))

    def glyph_meshes(self, t):
        if self._velocity_meshes is None:
            from ..gl_rendering import Material, Mesh
            from ..gl_primitives import ArrowMesh
            from ..gl_techniques import LAMBERT_TECHNIQUE
            self._velocity_material = Material(LAMBERT_TECHNIQUE, values={"u_color": [1.0, 0.0, 0.0, 0.0]})
            self._angular_velocity_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0.0, 1.0, 0.0]})
            self._velocity_meshes = {i: ArrowMesh(material=self._velocity_material,
                                                  head_radius=0.2*self.ball_radius,
                                                  head_length=0.5*self.ball_radius,
                                                  tail_radius=0.075*self.ball_radius,
                                                  tail_length=2*self.ball_radius)
                                     for i in range(self.num_balls)}
            self._angular_velocity_meshes = {
                i: Mesh({self._angular_velocity_material: self._velocity_meshes[i].primitives[self._velocity_material]})
                for i in range(self.num_balls)
            }
            for mesh in chain(self._velocity_meshes.values(), self._angular_velocity_meshes.values()):
                for prim in chain.from_iterable(mesh.primitives.values()):
                    prim.alias('vertices', 'a_position')
                mesh.init_gl()
        glyph_events = self.find_active_events(t)
        meshes = []
        for event in glyph_events:
            if isinstance(event, BallMotionEvent):
                tau = t - event.t
                r = event.eval_position(tau)
                v = event.eval_velocity(tau)
                v_mag = np.sqrt(v.dot(v))
                if v_mag > self._ZERO_TOLERANCE:
                    y = v / v_mag
                    mesh = self._velocity_meshes[event.i]
                    mesh.world_matrix[:] = 0
                    mesh.world_matrix[0,0] = mesh.world_matrix[1,1] = mesh.world_matrix[2,2] = mesh.world_matrix[3,3] = 1
                    mesh.world_matrix[1,:3] = y
                    x, z = mesh.world_matrix[0,:3], mesh.world_matrix[2,:3]
                    ydotx, ydotz = y.dot(x), y.dot(z)
                    if abs(ydotx) >= abs(ydotz):
                        mesh.world_matrix[2,:3] -= ydotz * y
                        mesh.world_matrix[2,:3] /= np.sqrt(mesh.world_matrix[2,:3].dot(mesh.world_matrix[2,:3]))
                        mesh.world_matrix[0,:3] = np.cross(mesh.world_matrix[1,:3], mesh.world_matrix[2,:3])
                    else:
                        mesh.world_matrix[0,:3] -= ydotx * y
                        mesh.world_matrix[0,:3] /= np.sqrt(mesh.world_matrix[0,:3].dot(mesh.world_matrix[0,:3]))
                        mesh.world_matrix[2,:3] = np.cross(mesh.world_matrix[0,:3], mesh.world_matrix[1,:3])
                    mesh.world_matrix[3,:3] = r + (2*self.ball_radius)*y
                    meshes.append(mesh)
                omega = event.eval_angular_velocity(tau)
                omega_mag = np.sqrt(omega.dot(omega))
                if omega_mag > self._ZERO_TOLERANCE:
                    y = omega / omega_mag
                    mesh = self._angular_velocity_meshes[event.i]
                    mesh.world_matrix[:] = 0
                    mesh.world_matrix[0,0] = mesh.world_matrix[1,1] = mesh.world_matrix[2,2] = mesh.world_matrix[3,3] = 1
                    mesh.world_matrix[1,:3] = y
                    x, z = mesh.world_matrix[0,:3], mesh.world_matrix[2,:3]
                    ydotx, ydotz = y.dot(x), y.dot(z)
                    if abs(ydotx) >= abs(ydotz):
                        mesh.world_matrix[2,:3] -= ydotz * y
                        mesh.world_matrix[2,:3] /= np.sqrt(mesh.world_matrix[2,:3].dot(mesh.world_matrix[2,:3]))
                        mesh.world_matrix[0,:3] = np.cross(mesh.world_matrix[1,:3], mesh.world_matrix[2,:3])
                    else:
                        mesh.world_matrix[0,:3] -= ydotx * y
                        mesh.world_matrix[0,:3] /= np.sqrt(mesh.world_matrix[0,:3].dot(mesh.world_matrix[0,:3]))
                        mesh.world_matrix[2,:3] = np.cross(mesh.world_matrix[0,:3], mesh.world_matrix[1,:3])
                    mesh.world_matrix[3,:3] = r + (2*self.ball_radius)*y
                    meshes.append(mesh)
        return meshes
