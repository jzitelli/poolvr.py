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
import logging
_logger = logging.getLogger(__name__)


from ..table import PoolTable
from .events import (CueStrikeEvent,
                     BallEvent,
                     BallStationaryEvent,
                     BallSpinningEvent,
                     BallRestEvent,
                     BallMotionEvent,
                     BallCollisionEvent,
                     MarlowBallCollisionEvent,
                     SimpleBallCollisionEvent,
                     RailCollisionEvent)
from .poly_solvers import quartic_solve


PIx2 = np.pi*2
SQRT2 = np.sqrt(2.0)
RAD2DEG = 180/np.pi
INCH2METER = 0.0254
INF = float('inf')


class PoolPhysics(object):
    _ZERO_TOLERANCE = 1e-9
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
    _IMAG_TOLERANCE = 1e-8
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
                 collision_search_time_limit=None,
                 collision_search_time_forward=None,
                 use_quartic_solver=True,
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
        self._collision_search_time_limit = collision_search_time_limit
        self._collision_search_time_forward = collision_search_time_forward
        if (collision_search_time_limit is not None and not np.isinf(collision_search_time_limit)) \
           or (collision_search_time_forward is not None and not np.isinf(collision_search_time_forward)):
            self._realtime = True
        else:
            self._realtime = False
        self._enable_occlusion = False #enable_occlusion
        self._enable_sanity_check = enable_sanity_check
        self._use_quartic_solver = use_quartic_solver
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
        self._rail_tuples = ((2, -1, -self._rhsz, self._bndx, self._sxcp),
                             (0,  1,  self._rhsx, self._bndz, self._szcp),
                             (2,  1,  self._rhsz, self._bndx, self._sxcp),
                             (0, -1, -self._rhsx, self._bndz, self._szcp))
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
        self._taus = np.zeros((num_balls, 3), dtype=np.float64)
        self._taus[:,0] = 1
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float64)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float64)
        self._F = np.zeros(num_balls, dtype=np.int)
        self.reset(ball_positions=ball_positions, balls_on_table=balls_on_table)

    def reset(self, ball_positions=None, balls_on_table=None):
        """
        Reset the state of the balls to at rest, at the specified positions.
        """
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
        self._balls_at_rest = set(self.balls_on_table)
        self._ball_motion_events = {}
        self._collisions = {}
        self._rail_collisions = {}
        self._a_ij[:] = 0
        self._a_ij_mag[:] = 0
        F, r_ij, r_ij_mag = self._F, self._r_ij, self._r_ij_mag
        F[:] = self.balls_on_table
        r_ij[F,F] = ball_positions
        for ii, i in enumerate(self.balls_on_table):
            r_ij[i,F] = r_ij[F,F] - ball_positions[ii]
            r_ij[F,i] = -r_ij[i,F]
            r_ij[i,i] = ball_positions[ii]
            r_ij_mag[i,F] = r_ij_mag[F,i] = np.linalg.norm(r_ij[i,F], axis=1)
        # update occlusion buffers:
        self._occ_ij[:] = False
        self._occ_ij[F, F] = True
        self._psi_ij[:] = 0
        self._theta_ij[:] = 0
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
            return self.add_event_sequence_realtime(event)
        else:
            return self.add_event_sequence(event)

    def add_event_sequence(self, event):
        num_events = len(self.events)
        self._add_event(event)
        #_logger.debug('added event %s', event)
        while self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
            #_logger.debug('added event %s', event)
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    def add_event_sequence_realtime(self, event):
        num_events = len(self.events)
        self._add_event(event)
        T, T_f = self._collision_search_time_limit, self._collision_search_time_forward
        lt = perf_counter()
        if T is None or np.isinf(T):
            while self.balls_in_motion:
                event = self._determine_next_event()
                self._add_event(event)
                if event.t - self.t > T_f:
                    break
        elif T_f is not None and not np.isinf(T_f):
            while T > 0 and self.balls_in_motion:
                event = self._determine_next_event()
                self._add_event(event)
                if event.t - self.t > T_f:
                    break
                t = perf_counter()
                T -= t - lt; lt = t
        else:
            while T > 0 and self.balls_in_motion:
                event = self._determine_next_event()
                self._add_event(event)
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
            self._step_realtime(dt)
        else:
            self.t += dt

    def _step_realtime(self, dt):
        self.t += dt
        if not self.balls_in_motion:
            return
        T, T_f = self._collision_search_time_limit, self._collision_search_time_forward
        if T_f is None:
            t_max = INF
        else:
            t_max = self.t + T_f
        if T is None or np.isinf(T):
            while self.balls_in_motion:
                event = self._determine_next_event()
                if event:
                    self._add_event(event)
                    if event.t >= t_max:
                        return
            return
        else:
            lt = perf_counter()
            while T > 0 and self.balls_in_motion:
                event = self._determine_next_event()
                if event:
                    self._add_event(event)
                    if event.t >= t_max:
                        return
                t = perf_counter()
                dt = t - lt; lt = t
                T -= dt
            return

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            num_balls = self.num_balls
            balls = range(num_balls)
        else:
            num_balls = len(balls)
        if out is None:
            out = np.zeros((num_balls, 3), dtype=np.float64)
        a = self._a
        taus = self._taus
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            events = events[:bisect(events, t)]
            if events:
                e = events[-1]
                taus[ii,1] = t - e.t
                if isinstance(e, BallMotionEvent):
                    a[ii] = e._a
                elif isinstance(e, BallStationaryEvent):
                    a[ii,0] = e._r_0
                    a[ii,1:] = 0
        taus[:num_balls,2] = taus[:num_balls,1]**2
        np.einsum('ijk,ij->ik', a[:num_balls], taus[:num_balls], out=out[:num_balls])
        return out

    def eval_velocities(self, t, balls=None, out=None):
        """
        Evaluate the velocities of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            num_balls = self.num_balls
            balls = range(num_balls)
        else:
            num_balls = len(balls)
        if out is None:
            out = np.zeros((num_balls, 3), dtype=np.float64)
        a = self._a
        taus = self._taus
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            events = events[:bisect(events, t)]
            if events:
                e = events[-1]
                taus[ii,1] = t - e.t
                if isinstance(e, BallMotionEvent):
                    a[ii] = e._a
                elif isinstance(e, BallStationaryEvent):
                    a[ii] = 0
        taus[:num_balls,1] *= 2
        np.einsum('ijk,ij->ik', a[:num_balls,1:], taus[:num_balls,:2], out=out[:num_balls])
        return out

    def eval_angular_velocities(self, t, balls=None, out=None):
        """
        Evaluate the angular velocities of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            num_balls = self.num_balls
            balls = range(num_balls)
        else:
            num_balls = len(balls)
        if out is None:
            out = np.zeros((num_balls, 3), dtype=np.float64)
        b = self._b
        taus = self._taus
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            events = events[:bisect(events, t)]
            if events:
                e = events[-1]
                taus[ii,1] = t - e.t
                if isinstance(e, BallMotionEvent):
                    b[ii] = e._b
                elif isinstance(e, BallSpinningEvent):
                    b[ii,:,::2] = 0
                    b[ii,:,1] = (e._omega_0_y, e._b)
                elif isinstance(e, BallStationaryEvent):
                    b[ii] = 0
        np.einsum('ijk,ij->ik', b[:num_balls], taus[:num_balls,:2], out=out[:num_balls])
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
            event.e_i.T_orig = event.e_i.T
            event.e_j.T_orig = event.e_j.T
            event.e_i.T = event.t - event.e_i.t
            event.e_j.T = event.t - event.e_j.t
        if isinstance(event, BallEvent):
            i = event.i
            self._rail_collisions.pop(i, None)
            self._collisions.pop(i, None)
            for k, v in self._collisions.items():
                v.pop(i, None)
            if self.ball_events[i]:
                last_ball_event = self.ball_events[i][-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T = event.t - last_ball_event.t
            self.ball_events[i].append(event)
            F = self._F
            F[:] = self.balls_on_table
            a_ij, a_ij_mag = self._a_ij, self._a_ij_mag
            if isinstance(event, BallStationaryEvent):
                if i in self._ball_motion_events:
                    self._ball_motion_events.pop(i)
                self._balls_at_rest.add(i)
                a_ij[i] = 0
                a_ij_mag[i,F] = a_ij_mag[F,i] = a_ij_mag[F,F]
                a_ij_mag[i,i] = 0
                if self._enable_occlusion and not self.balls_in_motion:
                    self._update_occlusion({i: event._r_0})
            elif isinstance(event, BallMotionEvent):
                accel = event.acceleration
                self._ball_motion_events[i] = event
                a_ij[i] = accel
                a_ij_mag[i,F] = a_ij_mag[F,i] = np.linalg.norm(a_ij[F] - accel, axis=1)
                a_ij_mag[i,i] = np.sqrt(np.dot(accel, accel))
                if i in self._balls_at_rest:
                    self._balls_at_rest.remove(i)
            if not isinstance(event, (CueStrikeEvent, RailCollisionEvent)):
                r_ij, r_ij_mag = self._r_ij, self._r_ij_mag
                r_ij[i,F] = r_ij[F,F] - event._r_0
                r_ij[F,i] = -r_ij[i,F]
                r_ij[i,i] = event._r_0
                r_ij_mag[i,F] = r_ij_mag[F,i] = np.linalg.norm(r_ij[i,F], axis=1)
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
            t_min = INF
        next_collision = None
        next_rail_collision = None
        r_ij_mag = self._r_ij_mag
        for i in self.balls_in_motion:
            e_i = self.ball_events[i][-1]
            if e_i.t >= t_min:
                continue
            nballs_in_motion = len(self.balls_in_motion)
            if i not in self._rail_collisions:
                self._rail_collisions[i] = self._find_rail_collision(e_i)
            rail_collision = self._rail_collisions[i]
            if rail_collision and rail_collision[0] < t_min:
                t_min = rail_collision[0]
                next_rail_collision = rail_collision
            if i not in self._collisions:
                self._collisions[i] = {}
            collisions = self._collisions[i]
            for j in sorted((j for j in self.balls_on_table if not (j <= i and j in self.balls_in_motion)),
                            key=lambda j: r_ij_mag[i,j]):
                e_j = self.ball_events[j][-1]
                t0 = max(e_i.t, e_j.t)
                if t_min <= t0:
                    continue
                if j not in collisions:
                    t1 = min(e_i.t + e_i.T, e_j.t + e_j.T)
                    if t1 <= t0:
                        collisions[j] = None
                        continue
                    # if self._enable_occlusion and isinstance(e_j, BallStationaryEvent) and self._occ_ij[i,j]:
                    #     collisions[j] = None
                    #     continue
                    if self._enable_occlusion and nballs_in_motion == 1 and self._occ_ij[i,j]:
                        collisions[j] = None
                        continue
                    if e_j.parent_event and e_i.parent_event and e_j.parent_event == e_i.parent_event:
                        collisions[j] = None
                        continue
                    if self._too_far_for_collision(e_i, e_j, t0, t1):
                        collisions[j] = None
                        continue
                    t_c = self._find_collision_time(e_i, e_j)
                    collisions[j] = t_c
                t_c = collisions[j]
                if t_c is not None and t0 < t_c < t_min:
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

    def _too_far_for_collision(self, e_i, e_j, t0, t1):
        a_ij_mag = self._a_ij_mag[e_i.i, e_j.i]
        if isinstance(e_j, BallStationaryEvent):
            tau_i_0 = t0 - e_i.t
            v_ij_0 = e_i.eval_velocity(tau_i_0)
            r_ij_0 = e_i.eval_position(tau_i_0) - e_j._r_0
        else:
            tau_i_0, tau_j_0 = t0 - e_i.t, t0 - e_j.t
            v_ij_0 = e_i.eval_velocity(tau_i_0) - e_j.eval_velocity(tau_j_0)
            r_ij_0 = e_i.eval_position(tau_i_0) - e_j.eval_position(tau_j_0)
        return np.sqrt(np.dot(v_ij_0, v_ij_0))*(t1-t0) + 0.5*a_ij_mag*(t1-t0)**2 \
             < np.sqrt(np.dot(r_ij_0, r_ij_0)) - self.ball_diameter

    def _find_collision_time(self, e_i, e_j):
        a_i, _ = e_i.global_motion_coeffs
        a_j, _ = e_j.global_motion_coeffs
        a_ji = a_i - a_j
        a_x, a_y = a_ji[2, ::2]
        b_x, b_y = a_ji[1, ::2]
        c_x, c_y = a_ji[0, ::2]
        p = self._p
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x*b_x + a_y*b_y)
        p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
        p[3] = 2 * b_x*c_x + 2 * b_y*c_y
        p[4] = c_x**2 + c_y**2 - 4 * self.ball_radius**2
        t0, t1 = max(e_i.t, e_j.t), min(e_i.t + e_i.T, e_j.t + e_j.T)
        return min((t.real for t in self._filter_roots(quartic_solve(p[::-1], only_real=True)
                                                       if self._use_quartic_solver else
                                                       np.roots(p))
                    if t0 <= t.real <= t1
                    and t.imag**2 / (t.real**2 + t.imag**2) < self._IMAG_TOLERANCE_SQRD),
                    default=None)

    def _filter_roots(self, roots):
        # filter out possible complex-conjugate pairs of roots:
        mask = self._mask; mask[:] = True
        while len(roots) > 0:
            i, z = next(((i, z) for i, z in enumerate(roots)
                         if abs(z.imag) > PoolPhysics._IMAG_TOLERANCE),
                        (None, None))
            if z is not None:
                j = next((j for j, z_conj in enumerate(roots[i+1:])
                          if  abs(z.real - z_conj.real) < PoolPhysics._ZERO_TOLERANCE
                          and abs(z.imag + z_conj.imag) < PoolPhysics._ZERO_TOLERANCE),
                         None)
                if j is not None:
                    mask[i] = mask[i+j+1] = False
                    roots = roots[mask[:len(roots)]]
                else:
                    break
            else:
                break
        return roots

    def _find_rail_collision(self, e_i):
        a = e_i._a
        # if e_i.parent_event and isinstance(e_i.parent_event, RailCollisionEvent):
        #     prev_side = e_i.parent_event.side
        # else:
        #     prev_side = None
        tau_min = e_i.T
        side_min = None
        for side, (j, sgn, rhs, bnd, bnd_cp) in enumerate(self._rail_tuples):
            # if side == prev_side:
            #     continue
            if sgn * a[1,j] <= 0:
                continue
            elif abs(a[1,j]) * tau_min < rhs - a[0,j]:
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

    def _update_occlusion(self, ball_positions=None):
        if ball_positions is None:
            ball_positions = {}
        balls, ball_positions = zip(*ball_positions.items())
        balls = list(balls)
        nballs = len(balls)
        r_ij = self._r_ij
        r_ij_mag = self._r_ij_mag
        thetas_ij = self._theta_ij
        psi_ij = self._psi_ij
        occ_ij = self._occ_ij
        F = self._F
        U = F[:nballs] = balls # U: ball no.s corresponding to the input ball_positions, respectively - these balls must be at rest.
        R = [i                 # R: ball no.s of all other balls at rest.
             for i in self.balls_at_rest if i not in U]
        R = F[nballs:nballs+len(R)] = R
        F = F[:nballs+len(R)]
        # r_ij[U,U] = ball_positions # diagonal contains the global rest position for balls at rest,
        #                            # and the motion start position for balls in motion
        U.sort()
        R.sort()
        M = np.array(self.balls_in_motion, dtype=np.int)
        occ_ij[M,:] = occ_ij[:,M] = False
        occ_ij[M,M] = True
        for ii, i in enumerate(U):
            F_i = F[ii+1:]
            if len(F_i) == 0:
                continue
            thetas_ij[i,F_i] = np.arctan2(r_ij[i,F_i,2], r_ij[i,F_i,0])
            thetas_ij[F_i,i] = thetas_ij[i,F_i] + np.pi
            psi_ij[i,F_i] = psi_ij[F_i,i] = np.arcsin(self.ball_diameter / r_ij_mag[i,F_i])
        for ii, i in enumerate(U):
            F_i = F[:-1].copy()
            if len(F_i) == 0:
                continue
            F_i[ii:] = F[ii+1:]
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
                    center_occluded, a_occluded, b_occluded = jj % 2 == 1, jj_a % 2 == 1, jj_b % 2 == 1
                    if center_occluded and jj_a == jj == jj_b:
                        occ_ij[i,j] = True
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
            for t in np.linspace(event.t, event.t + min(e_i.T, e_j.T), 100):
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
event %d: %s
  e_i: %s
  e_j: %s
''' % (2*self.ball_radius, d_ij, r_i, r_j, t, len(self.events)-1, event, e_i, e_j))

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
                v_mag = np.sqrt(np.dot(v, v))
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
