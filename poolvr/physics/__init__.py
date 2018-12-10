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


from ..table import PoolTable
from .events import (CueStrikeEvent,
                     BallEvent,
                     BallStationaryEvent,
                     BallRestEvent,
                     BallMotionEvent,
                     BallCollisionEvent,
                     MarlowBallCollisionEvent,
                     SimpleBallCollisionEvent)


PIx2 = np.pi*2
RAD2DEG = 180/np.pi
INCH2METER = 0.0254


class PoolPhysics(object):
    _ZERO_TOLERANCE = 1e-8
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
    _IMAG_TOLERANCE = 1e-8
    _IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2
    _BALL_MASS = 0.17
    _BALL_RADIUS = 1.125*INCH2METER
    _FR_COEFF_ROLLING = 0.016
    _FR_COEFF_SLIDING = 0.2
    _FR_COEFF_SPINING = 0.044
    _FR_COEFF_BALL2BALL = 0.06
    _GRAV_ACCEL = 9.81
    def __init__(self,
                 num_balls=16,
                 ball_mass=_BALL_MASS,
                 ball_radius=_BALL_RADIUS,
                 mu_r=_FR_COEFF_ROLLING,
                 mu_sp=_FR_COEFF_SPINING,
                 mu_s=_FR_COEFF_SLIDING,
                 mu_b=_FR_COEFF_BALL2BALL,
                 g=_GRAV_ACCEL,
                 balls_on_table=None,
                 ball_positions=None,
                 ball_collision_model="simple",
                 table=None,
                 enable_sanity_check=True,
                 enable_occlusion=True,
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
        if ball_collision_model == 'simple':
            self._ball_collision_event_class = SimpleBallCollisionEvent
        elif ball_collision_model == 'marlow':
            self._ball_collision_event_class = MarlowBallCollisionEvent
        else:
            raise Exception('dont know that collision model!')
        self.num_balls = num_balls
        # allocate for lower-level memory management:
        self._BALL_MOTION_EVENTS = [BallMotionEvent(0.0, i, float('inf'),
                                                    a=np.zeros((3,3), dtype=np.float64),
                                                    b=np.zeros((2,3), dtype=np.float64))
                                    for i in range(self.num_balls)]
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
        self._on_table = np.array(self.num_balls * [False])
        self._balls_on_table = balls_on_table
        self._balls_at_rest = set(balls_on_table)
        self._on_table[np.array(balls_on_table, dtype=np.int64)] = True
        self._collision_search_time_limit = collision_search_time_limit
        self._collision_search_time_forward = collision_search_time_forward
        self._enable_occlusion = enable_occlusion
        self._enable_sanity_check = enable_sanity_check
        self._p = np.empty(5, dtype=np.float64)
        self._mask = np.array(4*[True])
        self._a_ij_mag = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._a_ij = np.zeros((self.num_balls, 3), dtype=np.float64)
        self._r_ij = np.zeros((self.num_balls, self.num_balls, 3), dtype=np.float64)
        self._r_ij_mag = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._theta_ij = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._psi_ij = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
        self._occ_ij = np.array(self.num_balls*[self.num_balls*[False]])
        self._velocity_meshes = None
        self._angular_velocity_meshes = None
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
        self._update_positions(update_set=self.balls_on_table, rest_set=np.empty((0,3), dtype=np.float64), ball_positions=ball_positions)
        self._update_occlusion(update_set=self.balls_on_table, rest_set=np.empty((0,3), dtype=np.float64))

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
        self._balls_on_table = np.array(balls, dtype=np.int64)
        self._balls_on_table.sort()
        self._on_table[:] = False
        self._on_table[self._balls_on_table] = True

    @property
    def balls_in_motion(self):
        return self._ball_motion_events.keys()

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

    def set_cue_ball_collision_callback(self, cb):
        self._on_cue_ball_collide = cb

    def _add_event(self, event):
        self.events.append(event)
        if isinstance(event, BallEvent):
            i = event.i
            if self.ball_events[i]:
                last_ball_event = self.ball_events[i][-1]
                if event.t < last_ball_event.t + last_ball_event.T:
                    last_ball_event.T = event.t - last_ball_event.t
            if self._enable_occlusion and isinstance(event, BallStationaryEvent):
                update_set = [i]
                rest_set = sorted(self.balls_at_rest)
                self._update_positions(update_set=update_set, rest_set=rest_set, ball_positions=[event._r_0])
                self._update_occlusion(update_set=update_set, rest_set=rest_set)
            self.ball_events[i].append(event)
            if isinstance(event, BallStationaryEvent):
                if i in self._ball_motion_events:
                    self._ball_motion_events.pop(i)
                self._a_ij[i] = 0
                self._a_ij_mag[i,i] = 0
                self._a_ij_mag[i,:] = self._a_ij_mag[:,i] = self._a_ij_mag.diagonal()
                self._balls_at_rest.add(event.i)
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
        for i in sorted(self.balls_in_motion):
            e_i = self.ball_events[i][-1]
            for j in self.balls_on_table:
                if j <= i and j in self.balls_in_motion:
                    continue
                e_j = self.ball_events[j][-1]
                if self._enable_occlusion and isinstance(e_j, BallStationaryEvent) and self._occ_ij[i,j]:
                    continue
                t_c = self._find_collision(e_i, e_j, t_min)
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
        a_ij_mag = self._a_ij_mag[e_i.i, e_j.i]
        v_ij_0 = e_i.eval_velocity(tau_i_0) - e_j.eval_velocity(tau_j_0)
        r_ij_0 = e_i.eval_position(tau_i_0) - e_j.eval_position(tau_j_0)
        if np.sqrt(np.dot(v_ij_0, v_ij_0)) * (t1-t0) + 0.5 * a_ij_mag * (t1-t0)**2 < np.sqrt(np.dot(r_ij_0, r_ij_0)) - self.ball_diameter:
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
        def find_z(roots):
            return next(((i, z) for i, z in enumerate(roots) if z.imag != 0), (None, None))
        def find_z_conj(roots, i, z):
            return next((j for j, z_conj in enumerate(roots[i+1:])
                         if  abs(z.real - z_conj.real) < self._ZERO_TOLERANCE
                         and abs(z.imag + z_conj.imag) < self._ZERO_TOLERANCE), None)
        mask = self._mask; mask[:] = True
        for n in range(2):
            i, z = find_z(roots)
            if z is not None:
                j = find_z_conj(roots, i, z)
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

    def _update_positions(self, update_set, rest_set, ball_positions):
        r_ij = self._r_ij
        r_ij_mag = self._r_ij_mag
        theta_ij = self._theta_ij
        psi_ij = self._psi_ij
        U = np.array(update_set, dtype=np.int64)
        R = np.array(rest_set, dtype=np.int64)
        r_ij[U,U] = ball_positions
        # self._r_ij = r_ij = np.empty((ball_positions.shape[0],
        #                               ball_positions.shape[0],
        #                               3), dtype=np.float64)
        # r_ij[:] = [[ball_positions[i] - ball_positions[j]
        #             for j in range(ball_positions.shape[0])]
        #            for i in range(ball_positions.shape[0])]
        # iss = np.array(range(ball_positions.shape[0]), dtype=np.int64)
        # r_ij[iss,iss] = ball_positions
        U.sort(); R.sort()
        if len(R) > 0:
            F = np.hstack((U, R))
        else:
            F = U
        for ii, i in enumerate(U):
            F_i = F[ii+1:]
            r_ij[i,F_i] = r_ij[F_i,F_i] - ball_positions[ii]
            r_ij[F_i,i] = -r_ij[i,F_i]
            theta_ij[i,F_i] = np.arctan2(r_ij[i,F_i,2], r_ij[i,F_i,0])
            theta_ij[F_i,i] = PIx2 - theta_ij[i,F_i]
            r_ij_mag[i,F_i] = np.linalg.norm(r_ij[i,F_i], axis=1)
            r_ij_mag[F_i,i] = r_ij_mag[i,F_i]
#             if (psi_ij[i,F_i] == 0).any():
#                 _logger.debug('''
#         ball_positions = %s
#
#         i = %s,
#
#         F_i = %s
#
#         psi_ij[i,F_i] * RAD2DEG = %s
# ''',
#                               printit(ball_positions), i, printit(F_i), printit(psi_ij[i,F_i] * RAD2DEG))
#                 from sys import stdout
#                 stdout.flush()
            psi_ij[i,F_i] = np.arcsin(self.ball_diameter / r_ij_mag[i,F_i])
            psi_ij[F_i,i] = psi_ij[i,F_i]

    def _update_occlusion(self, update_set, rest_set, ball_positions=None):
        U = np.array(update_set, dtype=np.int64)
        R = np.array(rest_set, dtype=np.int64)
        U.sort(); R.sort()
        if len(R) > 0:
            F = np.hstack((U, R))
        else:
            F = U
        occ_ij = self._occ_ij
        r_ij_mag = self._r_ij_mag
        thetas_ij = self._theta_ij
        psi_ij = self._psi_ij
        for ii, i in enumerate(U):
            F_i = F[ii+1:]
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
                    jj_a, jj, jj_b = bisect(theta_i_occ_bnds, theta_a), bisect(theta_i_occ_bnds, theta), bisect(theta_i_occ_bnds, theta_b)
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

    def _calc_energy(self, t, balls=None):
        if balls is None:
            balls = self.balls_on_table
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return 0.5 * self.ball_mass * (velocities**2).sum() + 0.5 * self.ball_I * (omegas**2).sum()

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
            _logger.debug('initializing arrow meshes')
            from ..gl_rendering import Material, Mesh
            from ..primitives import ArrowMesh
            from ..techniques import LAMBERT_TECHNIQUE
            self._velocity_material = Material(LAMBERT_TECHNIQUE, values={"u_color": [1.0, 0.0, 0.0, 0.0]})
            self._velocity_meshes = {i: ArrowMesh(material=self._velocity_material,
                                                  head_radius=0.2*self.ball_radius,
                                                  head_length=0.5*self.ball_radius,
                                                  tail_radius=0.075*self.ball_radius,
                                                  tail_length=2*self.ball_radius)
                                     for i in range(self.num_balls)}
            self._angular_velocity_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0.0, 1.0, 0.0]})
            self._angular_velocity_meshes = {i: Mesh({self._angular_velocity_material: self._velocity_meshes[i].primitives[self._velocity_material]})
                                             for i in range(self.num_balls)}
            for mesh in chain(self._velocity_meshes.values(), self._angular_velocity_meshes.values()):
                for prim in chain.from_iterable(mesh.primitives.values()):
                    prim.attributes['a_position'] = prim.attributes['vertices']
                mesh.init_gl()
        glyph_events = []
        for i, events in self.ball_events.items():
            for e in events[:bisect(events, t)][::-1]:
                if t <= e.t + e.T:
                    glyph_events.append(e)
                    break
        meshes = []
        for event in glyph_events:
            if isinstance(event, BallMotionEvent):
                mesh = self._velocity_meshes[event.i]
                tau = t - event.t
                r = event.eval_position(tau)
                v = event.eval_velocity(tau)
                v_mag = np.sqrt(v.dot(v))
                y = v / v_mag
                mesh.world_matrix[:] = 0
                mesh.world_matrix[0,0] = mesh.world_matrix[1,1] = mesh.world_matrix[2,2] = mesh.world_matrix[3,3] = 1
                mesh.world_matrix[3,:3] = r + (2*self.ball_radius)*y
                mesh.world_matrix[1,:3] = y
                x, z = mesh.world_matrix[0,:3], mesh.world_matrix[2,:3]
                ydotx, ydotz = y.dot(x), y.dot(z)
                if ydotx >= ydotz:
                    mesh.world_matrix[2,:3] -= ydotz * y
                    mesh.world_matrix[2,:3] /= np.sqrt(mesh.world_matrix[2,:3].dot(mesh.world_matrix[2,:3]))
                    mesh.world_matrix[0,:3] = np.cross(mesh.world_matrix[1,:3], mesh.world_matrix[2,:3])
                else:
                    mesh.world_matrix[0,:3] -= ydotx * y
                    mesh.world_matrix[0,:3] /= np.sqrt(mesh.world_matrix[0,:3].dot(mesh.world_matrix[0,:3]))
                    mesh.world_matrix[2,:3] = np.cross(mesh.world_matrix[0,:3], mesh.world_matrix[1,:3])
                # mesh.world_matrix[1,1] *= v_mag
                meshes.append(mesh)

                mesh = self._angular_velocity_meshes[event.i]
                omega = event.eval_angular_velocity(tau)
                omega_mag = np.sqrt(omega.dot(omega))
                y = omega / omega_mag
                mesh.world_matrix[:] = 0
                mesh.world_matrix[0,0] = mesh.world_matrix[1,1] = mesh.world_matrix[2,2] = mesh.world_matrix[3,3] = 1
                mesh.world_matrix[3,:3] = r + (2*self.ball_radius)*y
                mesh.world_matrix[1,:3] = y
                x, z = mesh.world_matrix[0,:3], mesh.world_matrix[2,:3]
                ydotx, ydotz = y.dot(x), y.dot(z)
                if ydotx >= ydotz:
                    mesh.world_matrix[2,:3] -= ydotz * y
                    mesh.world_matrix[2,:3] /= np.sqrt(mesh.world_matrix[2,:3].dot(mesh.world_matrix[2,:3]))
                    mesh.world_matrix[0,:3] = np.cross(mesh.world_matrix[1,:3], mesh.world_matrix[2,:3])
                else:
                    mesh.world_matrix[0,:3] -= ydotx * y
                    mesh.world_matrix[0,:3] /= np.sqrt(mesh.world_matrix[0,:3].dot(mesh.world_matrix[0,:3]))
                    mesh.world_matrix[2,:3] = np.cross(mesh.world_matrix[0,:3], mesh.world_matrix[1,:3])
                #mesh.world_matrix[1,1] *= omega_mag
                meshes.append(mesh)

        return meshes
