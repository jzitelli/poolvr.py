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
from math import sqrt
import logging
_logger = logging.getLogger(__name__)
import numpy as np
from numpy import dot, array, float64, cross


from ..table import PoolTable
from .events import (CueStrikeEvent,
                     BallEvent,
                     BallStationaryEvent,
                     BallSpinningEvent,
                     BallRestEvent,
                     BallMotionEvent,
                     BallSlidingEvent,
                     BallCollisionEvent,
                     SimpleBallCollisionEvent,
                     SimulatedBallCollisionEvent,
                     FSimulatedBallCollisionEvent,
                     CornerCollisionEvent,
                     SegmentCollisionEvent)
from .poly_solvers import f_find_collision_time as find_collision_time, f_quartic_solve
from . import collisions


PIx2 = np.pi*2
SQRT2 = sqrt(2.0)
RAD2DEG = 180/np.pi
INCH2METER = 0.0254
INF = float('inf')
BALL_COLLISION_MODELS = {
    'simple': SimpleBallCollisionEvent,
    'simulated': SimulatedBallCollisionEvent,
    'fsimulated': FSimulatedBallCollisionEvent
}
_k = array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
           dtype=float64)    # of any ball-centered frame, following the convention of Marlow


def sorrted(roots):
    from poolvr.physics.poly_solvers import sort_complex_conjugate_pairs
    roots.sort()
    sort_complex_conjugate_pairs(roots)
    return roots


def printit(roots):
    return ',  '.join('%5.17f + %5.17fj' % (r.real, r.imag) if r.imag else '%5.17f' % r for r in roots)


class PoolPhysics(object):
    _ZERO_TOLERANCE = 1e-9
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
    _IMAG_TOLERANCE = 1e-8
    _IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2
    def __init__(self,
                 num_balls=16,
                 ball_mass=0.1406,
                 ball_radius=0.02625,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.21,
                 mu_b=0.05,
                 e=0.89,
                 g=9.81,
                 balls_on_table=None,
                 ball_positions=None,
                 ball_collision_model="simple",
                 ball_collision_model_kwargs=None,
                 table=None,
                 enable_occlusion=False,
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
        :param g:     :math:`g`,        downward acceleration due to gravity
        """
        if ball_collision_model not in BALL_COLLISION_MODELS:
            raise Exception('%s: dont know that collision model!' % ball_collision_model)
        self._ball_collision_model = ball_collision_model
        self._ball_collision_event_class = BALL_COLLISION_MODELS[ball_collision_model]
        self.num_balls = num_balls
        # allocate for lower-level memory management:
        self._BALL_REST_EVENTS = [BallRestEvent(0.0, i, r=np.zeros(3, dtype=float64))
                                  for i in range(self.num_balls)]
        if table is None:
            table = PoolTable(num_balls=num_balls, ball_radius=ball_radius)
        self.table = table
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.set_params(M=ball_mass, R=ball_radius,
                        mu_s=mu_s, mu_b=mu_b, e=e)
        # collisions.print_params()
        self.ball_diameter = 2*ball_radius
        self.ball_I = 2/5 * ball_mass * ball_radius**2 # moment of inertia
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.g = g
        self.t = 0.0
        self._balls_on_table = balls_on_table
        self._on_table = array(self.num_balls * [False])
        self._on_table[array(balls_on_table, dtype=np.int32)] = True
        self._collision_search_time_limit = collision_search_time_limit
        self._collision_search_time_forward = collision_search_time_forward
        if (collision_search_time_limit is not None and not np.isinf(collision_search_time_limit)) \
           or (collision_search_time_forward is not None and not np.isinf(collision_search_time_forward)):
            self._realtime = True
        else:
            self._realtime = False
        self._enable_occlusion = enable_occlusion
        self._use_quartic_solver = use_quartic_solver
        self._p = np.empty(5, dtype=float64)
        self._mask = array(4*[True])
        corners = np.empty((24,3))
        corners[...,1] = table.H + self.ball_radius
        corners[...,::2] = table._corners
        self._corners = corners
        tangents = [(c1 - c0) / np.linalg.norm(c1 - c0) for c0, c1 in zip(corners[:-1], corners[1:])]
        self._segments = tuple(
            (corners[i], corners[i+1], cross(tangents[i], _k), tangents[i])
            for i in (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22)
        )
        self._velocity_meshes = None
        self._angular_velocity_meshes = None
        if ball_collision_model_kwargs:
            self._ball_collision_model_kwargs = ball_collision_model_kwargs
        else:
            self._ball_collision_model_kwargs = {}
        self._taus = np.zeros((num_balls, 3), dtype=float64)
        self._taus[:,0] = 1
        self._a = np.zeros((num_balls, 3, 3), dtype=float64)
        self._b = np.zeros((num_balls, 2, 3), dtype=float64)
        self._F = np.zeros(num_balls, dtype=np.int32)
        self.reset(ball_positions=ball_positions, balls_on_table=balls_on_table)

    @classmethod
    def set_params(cls,
                   M=0.1406,
                   R=0.02625,
                   mu_s=0.21,
                   mu_b=0.05,
                   e=0.89,
                   **params):
        cls.ball_mass = M
        cls.ball_radius = R
        cls.mu_s = mu_s
        cls.mu_b = mu_b
        cls.e = e
        collisions.set_params(M=M, R=R, mu_s=mu_s, mu_b=mu_b, e=e)

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
        self._ball_motion_events = {}
        self._ball_spinning_events = {}
        self._collisions = {}
        self._rail_collisions = {}
        self._collision_events = {i: [] for i in self.balls_on_table}
        self._bounce_cnt = {(i,j): 0 for i in self.balls_on_table
                            for j in self.balls_on_table if j > i}

    @property
    def ball_collision_model(self):
        return self._ball_collision_model

    @property
    def balls_on_table(self):
        return self._balls_on_table
    @balls_on_table.setter
    def balls_on_table(self, balls):
        self._balls_on_table = array(balls, dtype=np.int32)
        self._balls_on_table.sort()
        self._on_table[:] = False
        self._on_table[self._balls_on_table] = True

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
        while self._ball_motion_events or self._ball_spinning_events:
            event = self._determine_next_event()
            self._add_event(event)
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    def add_event_sequence_realtime(self, event):
        num_events = len(self.events)
        self._add_event(event)
        T, T_f = self._collision_search_time_limit, self._collision_search_time_forward
        lt = perf_counter()
        if T is None or np.isinf(T):
            while self._ball_motion_events or self._ball_spinning_events:
                event = self._determine_next_event()
                self._add_event(event)
                if event.t - self.t > T_f:
                    break
        elif T_f is not None and not np.isinf(T_f):
            while T > 0 and (self._ball_motion_events or self._ball_spinning_events):
                event = self._determine_next_event()
                self._add_event(event)
                if event.t - self.t > T_f:
                    break
                t = perf_counter()
                T -= t - lt; lt = t
        else:
            while T > 0 and (self._ball_motion_events or self._ball_spinning_events):
                event = self._determine_next_event()
                self._add_event(event)
                t = perf_counter()
                T -= t - lt; lt = t
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    @property
    def balls_at_rest_time(self):
        """The time at which all balls have come to rest."""
        return self.events[-1].t \
            if self.events and isinstance(self.events[-1], BallRestEvent) and not (self._ball_motion_events
                                                                                   or self._ball_spinning_events) \
            else None

    def step(self, dt):
        if self._realtime:
            self._step_realtime(dt)
        else:
            self.t += dt

    def _step_realtime(self, dt):
        self.t += dt
        if not (self._ball_motion_events or self._ball_spinning_events):
            return
        T, T_f = self._collision_search_time_limit, self._collision_search_time_forward
        if T_f is None:
            t_max = INF
        else:
            t_max = self.t + T_f
        if T is None or np.isinf(T):
            while self._ball_motion_events or self._ball_spinning_events:
                event = self._determine_next_event()
                if event:
                    self._add_event(event)
                    if event.t >= t_max:
                        return
            return
        else:
            lt = perf_counter()
            while T > 0 and (self._ball_motion_events or self._ball_spinning_events):
                event = self._determine_next_event()
                if event:
                    self._add_event(event)
                    if event.t >= t_max:
                        return
                t = perf_counter()
                dt = t - lt; lt = t
                T -= dt
            if self.t >= self.events[-1].t:
                _logger.warning('STALLING TO CATCH UP SIMULATION!')
                lt = perf_counter()
                while self._ball_motion_events or self._ball_spinning_events:
                    event = self._determine_next_event()
                    if event:
                        self._add_event(event)
                        if event.t >= self.t:
                            _logger.warning('WAITED %s SECONDS FOR SIMULATION TO CATCH UP!', perf_counter() - lt)
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
            out = np.zeros((num_balls, 3), dtype=float64)
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
            out = np.zeros((num_balls, 3), dtype=float64)
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
            out = np.zeros((num_balls, 3), dtype=float64)
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
            out = np.zeros(len(balls), dtype=float64)
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return 0.5 * self.ball_mass * (velocities**2).sum() + 0.5 * self.ball_I * (omegas**2).sum()

    def find_active_events(self, t, balls=None):
        active_events = []
        if balls is None:
            balls = range(self.num_balls)
        for i in balls:
            events = self.ball_events.get(i, [])
            for e in events[:bisect(events, t)][::-1]:
                if t <= e.t + e.T:
                    active_events.append(e)
                    break
        return active_events

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def _add_event(self, event):
        self.events.append(event)
        if isinstance(event, BallCollisionEvent):
            i, j = event.i, event.j
            ii, jj = (min(i,j),max(i,j))
            self._collision_events[ii].append(event)
            self._collision_events[jj].append(event)
        elif isinstance(event, BallEvent):
            i = event.i
            ball_events = self.ball_events[i]
            if ball_events:
                last_ball_event = ball_events[-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T_orig = last_ball_event.T
                    last_ball_event.T = event.t - last_ball_event.t
            ball_events.append(event)
            self._rail_collisions.pop(i, None)
            self._collisions.pop(i, None)
            for k, v in self._collisions.items():
                v.pop(i, None)
            if isinstance(event, BallStationaryEvent):
                self._ball_motion_events.pop(i, None)
                if isinstance(event, BallSpinningEvent):
                    self._ball_spinning_events[i] = event
                elif i in self._ball_spinning_events:
                    self._ball_spinning_events.pop(i)
            elif isinstance(event, BallMotionEvent):
                self._ball_motion_events[i] = event
                self._collisions[i] = {}
        for child_event in event.child_events:
            self._add_event(child_event)

    def _determine_next_event(self):
        next_motion_event = min(e.next_motion_event
                                for e in chain(self._ball_motion_events.values(),
                                               self._ball_spinning_events.values())
                                if e.next_motion_event is not None)
        if next_motion_event:
            t_min = next_motion_event.t
        else:
            t_min = INF
        next_collision = None
        next_rail_collision = None
        ball_events = self.ball_events
        for i, e_i in self._ball_motion_events.items():
            if e_i.t >= t_min:
                continue
            if i not in self._rail_collisions:
                self._rail_collisions[i] = self._find_rail_collision(e_i)
            rail_collision = self._rail_collisions[i]
            if rail_collision and rail_collision[0] < t_min:
                t_min = rail_collision[0]
                next_rail_collision = rail_collision
            collisions = self._collisions[i]
            for j in self.balls_on_table:
                if j in self._ball_motion_events and j <= i:
                    continue
                e_j = ball_events[j][-1]
                t0 = max(e_i.t, e_j.t)
                if t_min <= t0:
                    continue
                if j not in collisions:
                    collisions[j] = self._find_collision_time(e_i, e_j)
                t_c = collisions[j]
                if t_c is not None and t0 < t_c < t_min:
                    t_min = t_c
                    next_collision = (t_c, e_i, e_j)
                    next_rail_collision = None
        if next_rail_collision is not None:
            t, e_i, seg = next_rail_collision
            if seg >= 18:
                cor = seg - 18
                return CornerCollisionEvent(t, e_i, cor, self._corners[cor])
            return SegmentCollisionEvent(t, e_i, seg, self._segments[seg][-2], self._segments[seg][-1])
        elif next_collision is not None:
            t_c, e_i, e_j = next_collision
            next_collision_event = self._ball_collision_event_class(t_c, e_i, e_j,
                                                                    **self._ball_collision_model_kwargs)
            return next_collision_event
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
        return sqrt(dot(v_ij_0, v_ij_0))*(t1-t0) + 0.5*a_ij_mag*(t1-t0)**2 \
             < sqrt(dot(r_ij_0, r_ij_0)) - self.ball_diameter

    def _find_collision_time(self, e_i, e_j):
        t0, t1 = max(e_i.t, e_j.t), min(e_i.t + e_i.T, e_j.t + e_j.T)
        if t1 <= t0:
            return None
        a_i = e_i.global_linear_motion_coeffs
        a_j = e_j.global_linear_motion_coeffs
        t_c = find_collision_time(a_i, a_j, self.ball_radius, t0, t1)
        # if t_c is not None:
        #     tau_i = t_c - e_i.t
        #     tau_j = t_c - e_j.t
        #     r_i = e_i.eval_position(tau_i)
        #     r_j = e_j.eval_position(tau_j)
        #     r_ij = r_j - r_i
        #     v_i = e_i.eval_velocity(tau_i)
        #     v_j = e_j.eval_velocity(tau_j)
        #     v_ij = v_j - v_i
        #     if np.dot(v_ij, r_ij) > 0:
        #         return None
        return t_c

    def _find_rail_collision(self, e_i):
        """
        Determines minimum collision time, if any, of the ball
        with any side cushion of the pool table or other
        boundary features in the vicinity of the pockets.

        To determine collision times with the side cushions, we solve
        the quadratic equation expressing the distance of space
        (along the normal axis) between the ball and the cushion.
        """
        T = e_i.T
        tau_min = T
        seg_min = None
        cor_min = None
        for i_seg, (r_0, r_1, nor, tan) in enumerate(self._segments):
            if e_i.parent_event and isinstance(e_i.parent_event, SegmentCollisionEvent) and e_i.parent_event.seg == i_seg:
                continue
            tau_n, tau_p = self._find_segment_collision_time(e_i, r_0, r_1, nor, tan)
            tau_n, tau_p = min(tau_n, tau_p), max(tau_n, tau_p)
            if 0 < tau_n < tau_min:
                r = e_i.eval_position(tau_n)
                if 0 < dot(r - r_0, tan) < dot(r_1 - r_0, tan) and 0 < dot(r - r_0, nor):
                    tau_min = tau_n
                    seg_min = i_seg
                    continue
            if 0 < tau_p < tau_min:
                r = e_i.eval_position(tau_p)
                if 0 < dot(r - r_0, tan) < dot(r_1 - r_0, tan) and 0 < dot(r - r_0, nor):
                    tau_min = tau_p
                    seg_min = i_seg
        for i_c, r_c in enumerate(self._corners):
            tau = self._find_corner_collision_time(r_c, e_i, tau_min)
            if 0 < tau < tau_min:
                seg_min = None
                cor_min = i_c
                tau_min = tau
        if seg_min is not None:
            return e_i.t + tau_min, e_i, seg_min
        if cor_min is not None:
            return e_i.t + tau_min, e_i, 18 + cor_min

    def _find_segment_collision_time(self, e_i, r_0, r_1, nor, tan):
        a0, a1, a2 = e_i._a
        A = dot(a2, nor)
        B = dot(a1, nor)
        C = dot((a0 - r_0), nor) - self.ball_radius
        DD = B**2 - 4*A*C
        if DD < 0:
            return -1.0, -1.0
        D = sqrt(DD)
        tau_p = 0.5 * (-B + D) / A
        tau_n = 0.5 * (-B - D) / A
        return tau_n, tau_p

    def _find_corner_collision_time(self, r_c, e_i, tau_min):
        tau_min = min(tau_min, e_i.T)
        if tau_min <= 0:
            return None, None
        a0, a1, a2 = e_i._a
        a0a0 = dot(a0, a0)
        a1a1 = dot(a1, a1)
        a0a1 = dot(a0, a1)
        a0a2 = dot(a0, a2)
        R_sqrd = self.ball_radius**2
        p = self._p
        p[4] = dot(a2, a2)
        p[3] = 2*dot(a1, a2)
        r_c_mag_sqrd = dot(r_c,r_c)
        p[0] = r_c_mag_sqrd \
             - 2*dot(a0, r_c) \
             + a0a0 - R_sqrd
        p[1] = 2*(a0a1 - dot(a1, r_c))
        p[2] = 2*(a0a2 - dot(a2, r_c)) + a1a1
        tau_cp = min((t.real for t in self._filter_roots(f_quartic_solve(p, only_real=True)
                                                         if self._use_quartic_solver else
                                                         np.roots(p[::-1]))
                      if 0.0 < t.real < tau_min
                      and t.imag**2 / (t.real**2 + t.imag**2) < self._IMAG_TOLERANCE_SQRD),
                     default=None)
        if tau_cp is not None:
            return tau_cp
        return -1.0

    def _filter_roots(self, roots):
        "filter out any complex-conjugate pairs of roots"
        npairs = 0
        i = 0
        while i < len(roots) - 1:
            r = roots[i]
            if abs(r.imag) > self._IMAG_TOLERANCE:
                for j, r_conj in enumerate(roots[i+1:]):
                    if abs(r.real - r_conj.real) < self._ZERO_TOLERANCE \
                       and abs(r.imag + r_conj.imag) < self._ZERO_TOLERANCE:
                        roots[i] = roots[2*npairs]
                        roots[i+j+1] = roots[2*npairs+1]
                        roots[2*npairs] = r
                        roots[2*npairs+1] = r_conj
                        npairs += 1
                        i += 1
                        break
            i += 1
        return roots[2*npairs:]


    def glyph_meshes(self, t):
        from ..gl_primitives import ProjectedMesh
        R = self.ball_radius
        if self._velocity_meshes is None:
            from ..gl_rendering import Material, Mesh
            from ..gl_primitives import ArrowMesh
            from ..gl_techniques import LAMBERT_TECHNIQUE, EGA_TECHNIQUE
            self._velocity_material = Material(LAMBERT_TECHNIQUE, values={"u_color": [1.0, 0.0, 0.0, 0.0]})
            self._angular_velocity_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0.0, 1.0, 0.0]})
            self._slip_velocity_material = Material(EGA_TECHNIQUE, values={"u_color": [1.0, 0.75, 0.0, 0.0]})
            self._velocity_meshes = {
                i: ArrowMesh(material=self._velocity_material,
                             head_radius=0.2*R,
                             head_length=0.5*R,
                             tail_radius=0.075*R,
                             tail_length=2*R)
                for i in range(self.num_balls)
            }
            self._angular_velocity_meshes = {
                i: Mesh({self._angular_velocity_material: self._velocity_meshes[i].primitives[self._velocity_material]})
                for i in range(self.num_balls)
            }
            self._slip_velocity_meshes = {
                i: ProjectedMesh(ArrowMesh(material=self._velocity_material,
                                           head_radius=0.2*R,
                                           head_length=0.5*R,
                                           tail_radius=0.075*R,
                                           tail_length=2*R), self._slip_velocity_material)
                for i in range(self.num_balls)
            }
            for mesh in chain(self._velocity_meshes.values(), self._angular_velocity_meshes.values(),
                              self._slip_velocity_meshes.values()):
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
                omega = event.eval_angular_velocity(tau)
                glyphs = [(v, sqrt(dot(v, v)), self._velocity_meshes),
                          (omega, sqrt(dot(omega, omega)), self._angular_velocity_meshes)]
                if isinstance(event, BallSlidingEvent):
                    u = v + R * array((omega[2], 0.0, -omega[0]), dtype=float64)
                    glyphs.append((u, sqrt(dot(u, u)), self._slip_velocity_meshes))
                for (vec, vec_mag, vec_meshes) in glyphs:
                    if vec_mag > self._ZERO_TOLERANCE:
                        mesh = vec_meshes[event.i]
                        _mesh = mesh
                        if isinstance(mesh, ProjectedMesh):
                            mesh = mesh.mesh
                        M = mesh.world_matrix
                        y = vec / vec_mag
                        M[:] = 0
                        M[0,0] = M[1,1] = M[2,2] = M[3,3] = 1
                        M[1,:3] = y
                        x, z = mesh.world_matrix[0,:3], mesh.world_matrix[2,:3]
                        ydotx, ydotz = y.dot(x), y.dot(z)
                        if abs(ydotx) >= abs(ydotz):
                            M[2,:3] -= ydotz * y
                            M[2,:3] /= sqrt(M[2,:3].dot(M[2,:3]))
                            M[0,:3] = np.cross(M[1,:3], M[2,:3])
                        else:
                            M[0,:3] -= ydotx * y
                            M[0,:3] /= sqrt(M[0,:3].dot(M[0,:3]))
                            M[2,:3] = np.cross(M[0,:3], M[1,:3])
                        M[3,:3] = r + (2*self.ball_radius)*y
                        if isinstance(_mesh, ProjectedMesh):
                            mesh = _mesh
                            mesh.update(c=self.table.H+0.0012)
                        meshes.append(mesh)
        return meshes
