"""
This package implements an event-based pool physics simulator based on the paper
(available at http://web.stanford.edu/group/billiards/AnEventBasedPoolPhysicsSimulator.pdf): ::

  AN EVENT-BASED POOL PHYSICS SIMULATOR
  Will Leckie, Michael Greenspan
  DOI: 10.1007/11922155_19 · Source: DBLP
  Conference: Advances in Computer Games, 11th International Conference,
  Taipei, Taiwan, September 6-9, 2005.

"""
import logging
_logger = logging.getLogger(__name__)
from bisect import bisect
from itertools import chain
from time import perf_counter

import numpy as np


from .events import (CueStrikeEvent, BallEvent, BallRestEvent,
                     BallMotionEvent, BallCollisionEvent,
                     MarlowBallCollisionEvent, SimpleBallCollisionEvent)
from ..table import PoolTable


INCH2METER = 0.0254


class PoolPhysics(object):
    _ZERO_TOLERANCE = 1e-7
    _IMAG_TOLERANCE = 1e-5
    _IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2
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
                 balls_on_table=None,
                 ball_positions=None,
                 ball_collision_model="simple",
                 table=None,
                 enable_sanity_check=True,
                 collision_search_time_limit=0.2/90,
                 collision_search_time_forward=0.25,
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
        if table is None:
            table = PoolTable(num_balls=num_balls, ball_radius=ball_radius)
        self.table = table
        self.num_balls = num_balls
        # allocate for lower-level memory management:
        self._BALL_MOTION_EVENTS = [BallMotionEvent(0.0, i, float('inf'),
                                                    a=np.zeros((3,3), dtype=np.float64),
                                                    b=np.zeros((2,3), dtype=np.float64))
                                    for i in range(self.num_balls)]
        self._BALL_REST_EVENTS = [BallRestEvent(0.0, i, r=np.zeros(3, dtype=np.float64))
                                  for i in range(self.num_balls)]
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        if ball_collision_model == 'simple':
            self._ball_collision_event_class = SimpleBallCollisionEvent
        elif ball_collision_model == 'marlow':
            self._ball_collision_event_class = MarlowBallCollisionEvent
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.ball_I = 2/5 * ball_mass * ball_radius**2 # moment of inertia
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.mu_b = mu_b
        self.c_b = c_b
        self.E_Y_b = E_Y_b
        self.g = g
        self.t = 0.0
        self._on_table = np.array(self.num_balls * [False])
        self._balls_on_table = balls_on_table
        self._on_table[np.array(balls_on_table)] = True
        self._collision_search_time_limit = collision_search_time_limit
        self._collision_search_time_forward = collision_search_time_forward
        self._enable_sanity_check = enable_sanity_check
        self._p = np.empty(5, dtype=np.float64)
        self._mask = np.array(4*[True])
        self.reset(ball_positions=ball_positions, balls_on_table=balls_on_table)

    def reset(self, ball_positions=None, balls_on_table=None):
        """
        Reset the state of the balls to at rest, at the specified positions.
        """
        self._ball_motion_events = {}
        if ball_positions is None:
            ball_positions = self.table.calc_racked_positions()
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
        self.balls_on_table = balls_on_table
        self.t = 0.0
        for e in self._BALL_MOTION_EVENTS:
            e._a[:] = 0
            e._b[:] = 0
            e.t = self.t
            e.T = 0.0
        for e, r in zip(self._BALL_REST_EVENTS, ball_positions):
            e._r[:] = r
            e.t = self.t
            e.T = float('inf')
        self.ball_events = {i: [self._BALL_REST_EVENTS[i]]
                            for i in balls_on_table}
        self.events = list(chain.from_iterable(self.ball_events.values()))

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
        return set(self._balls_on_table)
    @balls_on_table.setter
    def balls_on_table(self, balls):
        self._balls_on_table = set(balls)
        self._on_table[:] = False
        self._on_table[np.array(balls)] = True

    @property
    def balls_in_motion(self):
        return self._ball_motion_events.keys()

    def add_cue(self, cue):
        self.cues = [cue]

    def strike_ball(self, t, i, r_i, r_c, V, cue_mass):
        r"""
        Strike ball *i* at game time *t*.

        :param r_i: position of ball *i*
        :param r_c: point of contact
        :param V: impact velocity
        """
        if not self._on_table[i]:
            return
        event = CueStrikeEvent(t, i, r_i, r_c, V, cue_mass)
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
        self._add_event(event)
        T = self._collision_search_time_limit
        lt = perf_counter()
        while T > 0 and self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
            if event.t - self.t > self._collision_search_time_forward:
                break
            t = perf_counter()
            T -= t - lt; lt = t
        return event.t - self.t

    @property
    def next_turn_time(self):
        """The time at which all balls have come to rest."""
        return self.events[-1].t if self.events and isinstance(self.events[-1], BallRestEvent) else 0.0

    def step(self, dt):
        self.t += dt

    def step_realtime(self, dt,
                      find_collisions=True):
        self.t += dt
        if not find_collisions:
            return
        T = self._collision_search_time_forward
        lt = perf_counter()
        while T > 0 and self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
            t = perf_counter()
            T -= t - lt; lt = t

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        ts = t
        if out is None:
            try: # if passed vector t
                out = np.zeros((len(balls), len(ts), 3), dtype=np.float64)
                t = ts[0]
            except: # passed a scalar t
                out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_position(ts - e.t)
                        break
        return out

    def eval_quaternions(self, t, balls=None, out=None):
        """
        Evaluate the rotations of a set of balls (represented as quaternions) at game time *t*.

        :returns: shape (*N*, 4) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 4), dtype=np.float64)
        out[:] = 0
        out[:,3] = 1
        # for ii, i in enumerate(balls):
        #     events = self.ball_events.get(i, ())
        #     if events:
        #         for e in events[:bisect(events, t)][::-1]:
        #             if t <= e.t + e.T:
        #                 out[ii] = e.eval_quaternion(t - e.t)
        #                 break
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
        out[:] = 0
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
        out[:] = 0
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_angular_velocity(t - e.t)
                        break
        return out

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def sanity_check(self, event):
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
                    if d_ij - 2*self.ball_radius < -1e-5:
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

    def _add_event(self, event):
        self.events.append(event)
        if isinstance(event, BallEvent):
            if self.ball_events[event.i]:
                last_ball_event = self.ball_events[event.i][-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T = event.t - last_ball_event.t
            self.ball_events[event.i].append(event)
            if isinstance(event, BallRestEvent):
                if event.i in self._ball_motion_events:
                    self._ball_motion_events.pop(event.i)
            elif isinstance(event, BallMotionEvent):
                self._ball_motion_events[event.i] = event
        for child_event in event.child_events:
            self._add_event(child_event)
        if self._enable_sanity_check and isinstance(event, BallCollisionEvent):
            self.sanity_check(event)

    def _determine_next_event(self):
        next_motion_event = min(e.next_motion_event
                                for e in self._ball_motion_events.values()
                                if e.next_motion_event is not None)
        if next_motion_event:
            t_min = next_motion_event.t
        else:
            t_min = float('inf')
        collision_times = {}
        next_collision = None
        for i in self.balls_in_motion:
            e_i = self.ball_events[i][-1]
            for j in (j for j in self.balls_on_table if j != i):
                key = min(i,j), max(i,j)
                if key in collision_times:
                    continue
                e_j = self.ball_events[j][-1]
                t_c = self._find_collision(e_i, e_j, t_min)
                collision_times[key] = t_c
                if t_c is not None and t_c < t_min:
                    t_min = t_c
                    next_collision = (t_c, e_i, e_j)
        if next_collision is not None:
            t_c, e_i, e_j = next_collision
            return self._ball_collision_event_class(t_c, e_i, e_j)
        else:
            return next_motion_event

    def _find_collision(self, e_i, e_j, t_min):
        if e_j.parent_event and e_i.parent_event and e_j.parent_event == e_i.parent_event:
            return None
        t0 = max(e_i.t, e_j.t)
        t1 = min(e_i.t + e_i.T, e_j.t + e_j.T, t_min)
        if t0 >= t1:
            return None
        tau_i_0, tau_j_0 = t0 - e_i.t, t0 - e_j.t
        a_ij_mag = np.linalg.norm(e_i.acceleration - e_j.acceleration)
        v_ij_0_mag = np.linalg.norm(e_i.eval_velocity(tau_i_0) - e_j.eval_velocity(tau_j_0))
        r_ij_0_mag = np.linalg.norm(e_i.eval_position(tau_i_0) - e_j.eval_position(tau_j_0))
        if v_ij_0_mag * (t1-t0) + 0.5 * a_ij_mag * (t1-t0)**2 < r_ij_0_mag - 2*self.ball_radius:
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
        try:
            roots = np.roots(p)
        except np.linalg.linalg.LinAlgError as err:
            _logger.warning('LinAlgError occurred during solve for collision time:\np = %s\nerror:\n%s', p, err)
            return None
        # filter out possible complex-conjugate pairs of roots:
        def find_i(roots):
            return next(((i, r) for i, r in enumerate(roots) if r.imag != 0), (None, None))
        def find_j(roots, i, r):
            return next((j for j, q in enumerate(roots[i+1:])
                         if  abs(q.real - r.real) < self._ZERO_TOLERANCE
                         and abs(q.imag + r.imag) < self._ZERO_TOLERANCE), None)
        mask = self._mask; mask[:] = True
        for n in range(2):
            i, r = find_i(roots)
            if r is not None:
                j = find_j(roots, i, r)
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

    def _calc_energy(self, t, balls=None):
        if balls is None:
            balls = self.balls_on_table
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return self.ball_mass * (velocities**2).sum() / 2 + self.ball_I * (omegas**2).sum() / 2
