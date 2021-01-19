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
                     RailCollisionEvent,
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
        :param c_b:   :math:`c_b`,      ball material's speed of sound
        :param E_Y_b: :math:`{E_Y}_b`,  ball material's Young's modulus
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
        self._rhsx = 0.5*table.W - ball_radius
        self._rhsz = 0.5*table.L - ball_radius
        self._bndx = min(0.5*table.W - 0.999*ball_radius,
                         0.5*table.W - table.M_cp/SQRT2)
        self._bndz = min(0.5*table.L - 0.999*ball_radius,
                         0.5*table.L - table.M_cp/SQRT2)
        self._r_cp = np.empty((4,2,3), dtype=float64)
        self._r_cp[...,1] = table.H + self.ball_radius
        self._r_cp[0,0,::2] = ( -(0.5*table.W - table.M_cp/SQRT2),  -0.5*table.L                     )
        self._r_cp[0,1,::2] = (  (0.5*table.W - table.M_cp/SQRT2),  -0.5*table.L                     )
        self._r_cp[1,0,::2] = (   0.5*table.W,                     -(0.5*table.L - table.M_cp/SQRT2) )
        self._r_cp[1,1,::2] = (   0.5*table.W,                      (0.5*table.L - table.M_cp/SQRT2) )
        self._r_cp[2,0,::2] = (  (0.5*table.W - table.M_cp/SQRT2),   0.5*table.L                     )
        self._r_cp[2,1,::2] = ( -(0.5*table.W - table.M_cp/SQRT2),   0.5*table.L                     )
        self._r_cp[3,0,::2] = (  -0.5*table.W,                      (0.5*table.L - table.M_cp/SQRT2) )
        self._r_cp[3,1,::2] = (  -0.5*table.W,                     -(0.5*table.L - table.M_cp/SQRT2) )
        self._r_cp_len_sqrd = np.einsum('ijk,ijk->ij', self._r_cp, self._r_cp)
        self._rail_tuples = (
            # 0: collision eqn. var;
            #    1: normal sign;
            #        2: collision eqn. RHS;
            #                     3: bound of validity along perpendicular axis;
            #                                 4: positions of corner pocket corners;
            #                                                5: squared euclidean
            #                                                   lengths of the
            #                                                   corner pocket corner positions.
            ( 2, -1, -self._rhsz, self._bndx, self._r_cp[0], self._r_cp_len_sqrd[0] ),
            ( 0,  1,  self._rhsx, self._bndz, self._r_cp[1], self._r_cp_len_sqrd[1] ),
            ( 2,  1,  self._rhsz, self._bndx, self._r_cp[2], self._r_cp_len_sqrd[2] ),
            ( 0, -1, -self._rhsx, self._bndz, self._r_cp[3], self._r_cp_len_sqrd[3] )
        )
        corners = np.empty((24,3))
        corners[...,1] = table.H + self.ball_radius
        corners[0,::2] = -(0.5*table.W + 2*INCH2METER - table.T_cp/SQRT2), 0.5*table.L + 2*INCH2METER
        corners[1,::2] = -(0.5*table.W - table.M_cp/SQRT2),                0.5*table.L
        corners[2,::2] =  (0.5*table.W - table.M_cp/SQRT2),                0.5*table.L
        corners[3,::2] =  (0.5*table.W + 2*INCH2METER - table.T_cp/SQRT2), 0.5*table.L + 2*INCH2METER
        corners[4,::2] =  (0.5*table.W + 2*INCH2METER),                    0.5*table.L + 2*INCH2METER - table.T_cp/SQRT2
        corners[5,::2] =  (0.5*table.W),                                   0.5*table.L - table.M_cp/SQRT2
        corners[6,::2] =  (0.5*table.W),                0.5*table.M_sp
        corners[7,::2] =  (0.5*table.W + 2*INCH2METER), 0.5*table.T_sp
        corners[8,::2] =  (0.5*table.W + 2*INCH2METER), -0.5*table.T_sp
        corners[9,::2] =  (0.5*table.W), -0.5*table.M_sp
        corners[10,::2] = corners[5,0], -corners[5,2]
        corners[11,::2] = corners[4,0], -corners[4,2]
        corners[12,::2] = corners[3,0], -corners[3,2]
        corners[13,::2] = corners[2,0], -corners[2,2]
        corners[14,::2] = corners[1,0], -corners[1,2]
        corners[15,::2] = corners[0,0], -corners[0,2]
        corners[16,::2] = -corners[11,0], corners[11,2]
        corners[17,::2] = -corners[10,0], corners[10,2]
        corners[18,::2] = -corners[9,0], corners[9,2]
        corners[19,::2] = -corners[8,0], corners[8,2]
        corners[20,::2] = -corners[7,0], corners[7,2]
        corners[21,::2] = -corners[6,0], corners[6,2]
        corners[22,::2] = -corners[5,0], corners[5,2]
        corners[23,::2] = -corners[4,0], corners[4,2]
        self._corners = corners
        tangents = [c1 - c0 / np.linalg.norm(c1 - c0) for c0, c1 in zip(corners[:-1], corners[1:])]
        self._segments = (
            (corners[0], corners[1], -cross(tangents[0], _k), tangents[0]),
            (corners[1], corners[2], -cross(tangents[1], _k), tangents[1]),
            (corners[2], corners[3], -cross(tangents[2], _k), tangents[2]),
            (corners[4], corners[5], -cross(tangents[4], _k), tangents[4]),
            (corners[5], corners[6], -cross(tangents[5], _k), tangents[5]),
            (corners[6], corners[7], -cross(tangents[6], _k), tangents[6]),
            (corners[8], corners[9], -cross(tangents[8], _k), tangents[8]),
            (corners[9], corners[10], -cross(tangents[9], _k), tangents[9]),
            (corners[10], corners[11], -cross(tangents[10], _k), tangents[10]),
            (corners[12], corners[13], -cross(tangents[12], _k), tangents[12]),
            (corners[13], corners[14], -cross(tangents[13], _k), tangents[13]),
            (corners[14], corners[15], -cross(tangents[14], _k), tangents[14]),
            (corners[16], corners[17], -cross(tangents[16], _k), tangents[16]),
            (corners[17], corners[18], -cross(tangents[17], _k), tangents[17]),
            (corners[18], corners[19], -cross(tangents[18], _k), tangents[18]),
            (corners[20], corners[21], -cross(tangents[20], _k), tangents[20]),
            (corners[21], corners[22], -cross(tangents[21], _k), tangents[21]),
            (corners[22], corners[23], -cross(tangents[22], _k), tangents[22]),
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
        self._F = np.zeros(num_balls, dtype=np.int)
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
            _logger.info('t = %s, i = %s, seg = %s', t, e_i.i, seg)
            return SegmentCollisionEvent(t, e_i, seg, self._segments[seg][-2], self._segments[seg][-1])
            # if type(next_rail_collision[-1]) is tuple:
            #     t, i, (side, i_c) = next_rail_collision
            #     return CornerCollisionEvent(t=t, e_i=ball_events[i][-1],
            #                                 side=side, i_c=i_c,
            #                                 r_c=self._r_cp[side,i_c])
            # else:
            #     t, i, side = next_rail_collision
            #     return RailCollisionEvent(t=t, e_i=ball_events[i][-1],
            #                               side=side)
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
        # a = e_i._a
        T = e_i.T
        tau_min = T
        # side_min = None
        # cp_min = None
        seg_min = None
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
        if seg_min is not None:
            _logger.info('seg_min = %s, tau_min = %s', seg_min, tau_min)
            return e_i.t + tau_min, e_i, seg_min
        # for side, (j, sgn, rhs, bnd, r_cs, r_cs_mag_sqrd) in enumerate(self._rail_tuples):
        #     if e_i.parent_event and isinstance(e_i.parent_event, (RailCollisionEvent, CornerCollisionEvent)) \
        #        and e_i.parent_event.side == side:
        #         continue
        #     if sgn * a[1,j] <= 0 \
        #        or abs(a[1,j]) * tau_min < rhs - a[0,j]:
        #         continue
        #     check_corners = False
        #     k = 2 - j
        #     if abs(a[2,j]) < 1e-15:
        #         if abs(a[1,j]) > 1e-15:
        #             tau = (rhs - a[0,j]) / a[1,j]
        #             if 0 < tau < tau_min:
        #                 check_corners = True
        #                 r = e_i.eval_position(tau)
        #                 if abs(r[k]) < bnd:
        #                     tau_min = tau
        #                     side_min = side
        #                     cp_min = None
        #     else:
        #         d = a[1,j]**2 - 4*a[2,j]*(a[0,j] - rhs)
        #         if d > 1e-15:
        #             pn = sqrt(d)
        #             tau_p = (-a[1,j] + pn) / (2*a[2,j])
        #             tau_n = (-a[1,j] - pn) / (2*a[2,j])
        #             if 0 < tau_n < tau_min and 0 < tau_p < tau_min:
        #                 tau_a = min(tau_n, tau_p)
        #                 r = e_i.eval_position(tau_a)
        #                 if abs(r[k]) < bnd:
        #                     tau_min = tau_a
        #                     side_min = side
        #                     cp_min = None
        #                 else:
        #                     check_corners = True
        #             elif 0 < tau_n < tau_min:
        #                 r = e_i.eval_position(tau_n)
        #                 if abs(r[k]) < bnd:
        #                     tau_min = tau_n
        #                     side_min = side
        #                     cp_min = None
        #                 else:
        #                     check_corners = True
        #             elif 0 < tau_p < tau_min:
        #                 r = e_i.eval_position(tau_p)
        #                 if abs(r[k]) < bnd:
        #                     tau_min = tau_p
        #                     side_min = side
        #                     cp_min = None
        #                 else:
        #                     check_corners = True
        #     if check_corners:
        #         check_corners = False
        #         tau_cp, i_c_min = self._find_corner_collision_time(e_i, side, tau_min)
        #         if tau_cp is not None:
        #             tau_min = tau_cp
        #             side_min = side
        #             cp_min = i_c_min
        # if cp_min is not None:
        #     return (e_i.t + tau_min, e_i.i, (side_min, cp_min))
        # elif side_min is not None:
        #     return (e_i.t + tau_min, e_i.i, side_min)

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
        # if tau_n > 0:
        #     r = e_i.eval_position(tau_n)
        #     if 0 < dot(r - r_0, tan) < dot(r_1 - r_0, tan):
        #         return tau_n
        # if tau_p > 0:
        #     r = e_i.eval_position(tau_p)
        #     if 0 < dot(r - r_0, tan) < dot(r_1 - r_0, tan):
        #         return tau_p

    def _find_corner_collision_time(self, e_i, side, tau_min):
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
        i_c_min = None
        r_cs, r_cs_mag_sqrd = self._r_cp[side], self._r_cp_len_sqrd[side]
        for i_c, (r_c, r_c_mag_sqrd) in enumerate([(r_cs[0], r_cs_mag_sqrd[0]),
                                                   (r_cs[1], r_cs_mag_sqrd[1])]):
            p[0] = r_c_mag_sqrd \
                - 2*dot(a0, r_c) \
                + a0a0 \
                - R_sqrd
            p[1] = 2*(a0a1 - dot(a1, r_c))
            p[2] = 2*(a0a2 - dot(a2, r_c)) + a1a1
            tau_cp = min((t.real for t in self._filter_roots(f_quartic_solve(p, only_real=True)
                                                             if self._use_quartic_solver else
                                                             np.roots(p[::-1]))
                          if 0.0 < t.real < tau_min
                          and t.imag**2 / (t.real**2 + t.imag**2) < self._IMAG_TOLERANCE_SQRD),
                       default=None)
            if tau_cp is not None:
                # r = e_i.eval_position(tau_cp)
                # if dot(r, r) <= r_c_mag_sqrd:
                #     tau_min = tau_cp
                #     i_c_min = i_c
                tau_min = tau_cp
                i_c_min = i_c
        if i_c_min is None:
            return None, None
        return tau_min, i_c_min

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
