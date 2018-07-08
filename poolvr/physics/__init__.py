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
from bisect import bisect
from itertools import chain
import numpy as np


from .events import CueStrikeEvent, BallEvent, BallMotionEvent, BallRestEvent, DefaultBallCollisionEvent, SimpleBallCollisionEvent
from ..table import PoolTable


INCH2METER = 0.0254
_logger = logging.getLogger(__name__)


def _create_cue(cue_mass, cue_radius, cue_length):
    try:
        import ode
    except ImportError as err:
        from .. import fake_ode as ode
    body = ode.Body(ode.World())
    mass = ode.Mass()
    mass.setCylinderTotal(cue_mass, 3, cue_radius, cue_length)
    body.setMass(mass)
    return body


class PoolPhysics(object):
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
                 on_table=None,
                 initial_positions=None,
                 use_simple_ball_collisions=False,
                 **kwargs):
        if on_table is None:
            on_table = num_balls * [True]
        if use_simple_ball_collisions:
            self._ball_collision_event_class = SimpleBallCollisionEvent
        else:
            self._ball_collision_event_class = DefaultBallCollisionEvent
        self.num_balls = num_balls
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
        self.balls_in_motion = set()
        self.on_table = np.array(on_table)
        self.balls_on_table = set(i for i in range(self.num_balls) if self.on_table[i])
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float64)
        self._b = np.zeros((num_balls, 2, 3), dtype=np.float64)
        self.ball_positions = self._a[:,0]
        self.ball_velocities = self._a[:,1]
        if initial_positions is None:
            initial_positions = PoolTable().calc_racked_positions(num_balls=num_balls)
        self.ball_positions[:] = initial_positions
        self.ball_events = {i: [BallRestEvent(self.t, i, r=self.ball_positions[i])]
                            for i in self.balls_on_table}
        self.events = [self.ball_events[i][0] for i in self.balls_on_table]
        self._ball_motion_events = {}

    def reset(self, ball_positions=None, on_table=None):
        """
        Reset the state of the balls to at rest, at the specified positions.
        """
        self.t = 0
        self._a[:] = 0
        self._b[:] = 0
        self.ball_positions[:] = ball_positions
        self.balls_in_motion = set()
        if on_table is None:
            self.on_table[:] = True
        else:
            self.on_table[:] = on_table
        self.balls_on_table = set(i for i in range(self.num_balls) if self.on_table[i])
        self.ball_events = {i: [BallRestEvent(self.t, i, r=self.ball_positions[i])]
                            for i in self.balls_on_table}
        self.events = list(chain.from_iterable(self.ball_events.values()))
        self._ball_motion_events.clear()

    def add_cue(self, cue):
        body = _create_cue(cue.mass, cue.radius, cue.length)
        self.cue_bodies = [body]
        self.cue_geoms = [body]
        return body, body

    def strike_ball(self, t, i, r_c, V, cue_mass):
        r"""
        Strike ball *i* at game time *t*.

        :param r_c: point of contact
        :param V: impact velocity
        """
        if not self.on_table[i]:
            return
        event = CueStrikeEvent(t, i, self.ball_positions[i], r_c, V, cue_mass)
        return self.add_event_sequence(event)

    def add_event_sequence(self, event):
        num_events = len(self.events)
        self._add_event(event)
        while self.balls_in_motion:
            event = self._determine_next_event()
            self._add_event(event)
        num_added_events = len(self.events) - num_events
        return self.events[-num_added_events:]

    def events_str(self, events=None, sep='\n\n'):
        if events is None:
            events = self.events
        return sep.join('%3d (%5.5f, %5.5f): %s' % (i_e, e.t, e.t+e.T, e) for i_e, e in enumerate(events))

    def _add_event(self, event):
        _logger.debug('adding event: %s', event)
        self.events.append(event)
        if isinstance(event, BallEvent):
            if self.ball_events[event.i]:
                last_ball_event = self.ball_events[event.i][-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T = event.t - last_ball_event.t
            self.ball_events[event.i].append(event)
            if isinstance(event, BallRestEvent):
                self.balls_in_motion.remove(event.i)
                if event.i in self._ball_motion_events:
                    self._ball_motion_events.pop(event.i)
            elif isinstance(event, BallMotionEvent):
                self.balls_in_motion.add(event.i)
                self._ball_motion_events[event.i] = event
        for child_event in event.child_events:
            self._add_event(child_event)

    def _determine_next_event(self):
        ball_motion_events = list(sorted(self._ball_motion_events.values()))
        #_logger.debug('ball_motion_events:\n%s', '\n'.join(str(e) for e in ball_motion_events))
        next_motion_events = [e.next_motion_event
                              for e in ball_motion_events if e.next_motion_event is not None]
        next_motion_events.sort()
        if next_motion_events:
            next_motion_event = next_motion_events[0]
        else:
            next_motion_event = None
        #_logger.debug('next_motion_event:\n%s', next_motion_event)
        collision_times = {}
        next_collision = None
        for i in self.balls_in_motion:
            e_i = self.ball_events[i][-1]
            for j in (j for j in self.balls_on_table if j != i):
                key = min(i,j), max(i,j)
                if key in collision_times:
                    continue
                e_j = self.ball_events[j][-1]
                t_c = self._find_collision(e_i, e_j)
                collision_times[key] = t_c
                if t_c is not None and (next_collision is None or t_c < next_collision[0]):
                    next_collision = (t_c, i, j)
        if next_collision is not None and (next_motion_event is None
                                           or next_collision[0] < next_motion_event.t):
            t_c, i, j = next_collision
            tau_i, tau_j = t_c - e_i.t, t_c - e_j.t
            return self._ball_collision_event_class(
                t_c, i, j,
                e_i.eval_position(tau_i),         e_j.eval_position(tau_j),
                e_i.eval_velocity(tau_i),         e_j.eval_velocity(tau_j),
                e_i.eval_angular_velocity(tau_i), e_j.eval_angular_velocity(tau_j)
            )
        else:
            return next_motion_event

    def _find_collision(self, e_i, e_j):
        # if e_i.parent_event:
        #     _logger.debug('e_i.parent_event = %s', e_i.parent_event)
        # if e_j.parent_event:
        #     _logger.debug('e_j.parent_event = %s', e_j.parent_event)
        if e_j.parent_event and e_i.parent_event and e_j.parent_event == e_i.parent_event:
            return None
        t0 = max(e_i.t, e_j.t)
        t1 = min(e_i.t + e_i.T, e_j.t + e_j.T)
        _logger.debug('t0, t1 = %s', (t0, t1))
        if t0 < self.t:
            _logger.debug('skipping because in the past')
            return None
        if e_i.t + e_i.T < e_j.t or e_j.t + e_j.T < e_i.t:
            _logger.debug('skipping because non-overlapping events')
            return None
        a_i, b_i = e_i.global_motion_coeffs
        a_j, b_j = e_j.global_motion_coeffs
        return self._find_collision_time(a_i, a_j, t0, t1)

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
        roots = np.roots(p)
        roots = [t.real for t in roots if t0 < t.real < t1 and abs(t.imag) / np.sqrt(t.real**2+t.imag**2) < 0.01]
        if not roots:
            return None
        else:
            return min(roots)

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        out[:] = self._a[balls,0] # TODO: use PositionBallEvent instead
        for ii, i in enumerate(balls):
            events = self.ball_events.get(i, ())
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
            balls = range(self.num_balls)
        if out is None:
            out = np.empty((len(balls), 3), dtype=np.float64)
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

    def next_turn_time(self):
        """
        Return the time at which all balls have come to rest.
        """
        return self.events[-1].t if self.events and isinstance(self.events[-1], BallRestEvent) else None

    def step(self, dt):
        self.t += dt

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def _calc_energy(self, t, balls=None):
        if balls is None:
            balls = self.balls_on_table #(i for i in range(self.num_balls) if self.on_table[i])
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return self.ball_mass * (velocities**2).sum() / 2 + self.ball_I * (omegas**2).sum() / 2

    def _find_active_events(self, t):
        n = bisect(self.events, t)
        return [e for e in self.events[:n] if e.t <= t <= e.t + e.T]
