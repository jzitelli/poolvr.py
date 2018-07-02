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
import numpy as np


from .events import CueStrikeEvent, BallEvent, BallMotionEvent, BallRestEvent


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
        self.events = []
        self.ball_events = {i: [] for i in range(num_balls)}
        self.balls_in_motion = set()
        self.on_table = np.array(self.num_balls * [True])
        self.balls_on_table = set(range(self.num_balls))
        self._a = np.zeros((num_balls, 3, 3), dtype=np.float64)
        self.ball_positions = self._a[:,0]
        self.ball_velocities = self._a[:,1]
        if initial_positions is not None:
            self.ball_positions[:] = initial_positions

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
        events = [event]
        self._add_event(event)
        while self.balls_in_motion:
            event = self._determine_next_event()
            events.append(event)
            self._add_event(event)
        return events

    def _add_event(self, event):
        self.events.append(event)
        if isinstance(event, BallEvent):
            if isinstance(event, BallRestEvent):
                self.balls_in_motion.remove(event.i)
            elif isinstance(event, BallMotionEvent):
                self.balls_in_motion.add(event.i)
            self.ball_events[event.i].append(event)
        _logger.debug('added event: %s', event)

    def _determine_next_event(self):
        next_event = None
        for i in self.balls_in_motion:
            e_i = self.ball_events[i][-1]
            if next_event is None or next_event.t > e_i.next_motion_event.t:
                next_event = e_i.next_motion_event
        for i in self.balls_in_motion:
            e_i = self.ball_events[i][-1]
            for j in self.balls_on_table:
                e_j = self.ball_events[j][-1]
                _logger.debug('e_i = %s, e_j = %s', e_i, e_j)
        _logger.debug('next_event = %s', next_event)
        return next_event

    def eval_positions(self, t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
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
            balls = range(self.num_balls)
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
            balls = range(self.num_balls)
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
        self.ball_events = {i: [] for i in range(self.num_balls)}

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
            balls = range(self.num_balls)
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return self.ball_mass * (velocities**2).sum() / 2 + self._I * (omegas**2).sum() / 2
