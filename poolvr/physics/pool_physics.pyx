# import logging
# _logger = logging.getLogger(__name__)
from bisect import bisect
from itertools import chain
from time import perf_counter
import numpy as np
cimport numpy as np


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


cdef double INCH2METER = 0.0254
cdef double ball_radius = 1.125 * INCH2METER
cdef double _almost_ball_radius = 0.999 * ball_radius
cdef double ball_diameter = 2*ball_radius
cdef double ball_mass = 0.17
cdef double ball_I = 2.0/5 * ball_mass * ball_radius**2
cdef double g = 9.81 # magnitude of acceleration due to gravity
cdef double c_b = 4000.0 # ball material's speed of sound
cdef double E_Y_b = 2.4e9 # ball material's Young's modulus of elasticity
cdef double _ZERO_TOLERANCE = 1e-8
cdef double _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
cdef double _IMAG_TOLERANCE = 1e-7
cdef double _IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2
cdef double PIx2 = 2*np.pi
cdef double SQRT2 = np.sqrt(2)


cdef class PoolPhysics:
    cdef public int num_balls
    cdef public object table
    cdef public list cues
    cdef public double t
    cdef public list events
    cdef public dict ball_events
    cdef public object _ball_collision_event_class
    cdef public set _balls_at_rest
    cdef public object _balls_on_table
    cdef public np.ndarray _on_table
    cdef public np.ndarray _p
    cdef public np.ndarray _mask
    cdef public np.ndarray _a_ij
    cdef public np.ndarray _a_ij_mag
    cdef public np.ndarray _r_ij
    cdef public np.ndarray _r_ij_mag
    cdef public np.ndarray _theta_ij
    cdef public np.ndarray _psi_ij
    cdef public np.ndarray _occ_ij
    cdef public dict _velocity_meshes
    cdef public dict _angular_velocity_meshes
    cdef public object _velocity_material
    cdef public object _angular_velocity_material
    cdef public dict _ball_motion_events
    # cdef public list _BALL_MOTION_EVENTS
    cdef public list _BALL_REST_EVENTS
    cdef public dict _collisions
    cdef public double _sx
    cdef public double _sz
    cdef public double _rhsx
    cdef public double _rhsz
    cdef public double _sxcp
    cdef public double _szcp
    cdef public object _enable_occlusion
    cdef public object _realtime
    cdef public double _collision_search_time_limit
    cdef public double _collision_search_time_forward
    cdef public object _enable_sanity_check
    def __init__(self,
                 num_balls=16,
                 balls_on_table=None,
                 ball_positions=None,
                 ball_collision_model="simple",
                 table=None,
                 enable_sanity_check=True,
                 enable_occlusion=True,
                 realtime=False,
                 double collision_search_time_limit=0.2/90,
                 double collision_search_time_forward=0.2,
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
        # self._BALL_MOTION_EVENTS = [BallMotionEvent(0.0, int(i), T=float('inf'),
        #                                             a=np.zeros((3,3), dtype=np.float64),
        #                                             b=np.zeros((2,3), dtype=np.float64))
        #                             for i in range(self.num_balls)]
        self._BALL_REST_EVENTS = [BallRestEvent(0.0, int(i), r_0=np.zeros(3, dtype=np.float64))
                                  for i in range(self.num_balls)]
        if table is None:
            table = PoolTable(num_balls=num_balls, ball_radius=ball_radius)
        self.table = table
        if balls_on_table is None:
            balls_on_table = range(self.num_balls)
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
        self._sx   = 0.5*table.W
        self._sz   = 0.5*table.L
        self._sxcp = self._sx - table.M_cp/SQRT2
        self._szcp = self._sz - table.M_cp/SQRT2
        self._rhsx = 0.5*table.W - ball_radius
        self._rhsz = 0.5*table.L - ball_radius
        self._a_ij = np.zeros((self.num_balls, 3), dtype=np.float64)
        self._a_ij_mag = np.zeros((self.num_balls, self.num_balls), dtype=np.float64)
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
        # for e in self._BALL_MOTION_EVENTS:
        #     e._a[:] = 0
        #     e._b[:] = 0
        #     e.t = self.t
        #     e.T = 0.0
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
        if self._enable_occlusion:
            self._update_occlusion({e.i: e._r_0 for e in self._BALL_REST_EVENTS})

    @property
    def ball_collision_model(self):
        return 'marlow' if self._ball_collision_event_class is MarlowBallCollisionEvent else 'simple'

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

    def strike_ball(self, double t, int i, r_i, r_c, V, double M):
        r"""
        Strike ball *i* at game time *t*.

        :param r_i: position of ball *i*
        :param r_c: point of contact
        :param V: impact velocity
        :param M: impact mass
        """
        if not self._on_table[i]:
            return
        #assert abs(np.linalg.norm(r_c - r_i) - ball_radius) < _ZERO_TOLERANCE, 'abs(np.linalg.norm(r_c - r_i) - ball_radius) = %s' % abs(np.linalg.norm(r_c - r_i) - ball_radius)
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
        return self.events[-1].t \
            if self.events and isinstance(self.events[-1], BallRestEvent) else 0.0

    def step(self, double dt, **kwargs):
        if self._realtime:
            self.step_realtime(dt, **kwargs)
        else:
            self.t += dt

    def step_realtime(self, double dt,
                      find_collisions=True):
        self.t += dt
        if not find_collisions:
            return
        T = self._collision_search_time_limit
        t_max = self.t + self._collision_search_time_forward
        lt = perf_counter()
        while T > 0 and self.balls_in_motion:
            event = self._determine_next_event()
            if event:
                self._add_event(event)
                if event.t >= t_max:
                    return self.balls_in_motion
            t = perf_counter()
            dt = t - lt; lt = t
            T -= dt
        if T <= 0:
            return self.balls_in_motion

    def eval_positions(self, double t, balls=None, out=None):
        """
        Evaluate the positions of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = [e for e in self.ball_events.get(i, ()) if e.T > 0]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_position(t - e.t)
                        break
        return out

    def eval_velocities(self, double t, balls=None, out=None):
        """
        Evaluate the velocities of a set of balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = [e for e in self.ball_events.get(i, ()) if e.T > 0]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_velocity(t - e.t)
                        break
        return out

    def eval_angular_velocities(self, double t, balls=None, out=None):
        """
        Evaluate the angular velocities of all balls at game time *t*.

        :returns: shape (*N*, 3) array, where *N* is the number of balls
        """
        if balls is None:
            balls = range(self.num_balls)
        if out is None:
            out = np.zeros((len(balls), 3), dtype=np.float64)
        for ii, i in enumerate(balls):
            events = [e for e in self.ball_events.get(i, ()) if e.T > 0]
            if events:
                for e in events[:bisect(events, t)][::-1]:
                    if t <= e.t + e.T:
                        out[ii] = e.eval_angular_velocity(t - e.t)
                        break
        return out

    def find_active_events(self, double t):
        active_events = []
        for i, events in self.ball_events.items():
            events = [e for e in events if e.T > 0]
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
        rail_collisions = {}
        for i in sorted(self.balls_in_motion):
            if i not in self._collisions:
                self._collisions[i] = {}
            collisions = self._collisions[i]
            e_i = self.ball_events[i][-1]
            rail_collision = self._find_rail_collision(e_i)
            if rail_collision and rail_collision[0] < t_min:
                rail_collisions[i] = rail_collision
                t_min = rail_collision[0]
            for j in self.balls_on_table:
                if j <= i and j in self.balls_in_motion:
                    continue
                e_j = self.ball_events[j][-1]
                # if self._enable_occlusion and isinstance(e_j, BallStationaryEvent) and self._occ_ij[i,j]:
                #     t_c = None
                # else:
                #     t_c = self._find_collision(e_i, e_j, t_min)
                if j not in collisions:
                    if self._enable_occlusion and isinstance(e_j, BallStationaryEvent) and self._occ_ij[i,j]:
                        t_c = None
                    else:
                        t_c = self._find_collision(e_i, e_j, float('inf'))
                    collisions[j] = t_c
                t_c = collisions[j]
                if t_c is not None and t_c < t_min:
                    t_min = t_c
                    next_collision = (t_c, e_i, e_j)
        for i, rail_collision in rail_collisions.items():
            if rail_collision[0] == t_min:
                return RailCollisionEvent(t=rail_collision[0], e_i=self.ball_events[i][-1], side=rail_collision[1])
        if next_collision is not None:
            t_c, e_i, e_j = next_collision
            return self._ball_collision_event_class(t_c, e_i, e_j)
        else:
            return next_motion_event

    def _find_rail_collision(self, e_i):
        R = ball_radius
        rhsx, rhsz = self._rhsx, self._rhsz
        sxcp, szcp = self._sxcp, self._szcp
        a = e_i._a
        if e_i.parent_event and isinstance(e_i.parent_event, RailCollisionEvent):
            prev_side = e_i.parent_event.side
        else:
            prev_side = None
        tau_min = e_i.T
        side_min = None
        for side, (j, rhs, rhsp) in enumerate([(2,  rhsz, sxcp),
                                               (0,  rhsx, szcp),
                                               (2, -rhsz, sxcp),
                                               (0, -rhsx, szcp)]):
            if side == prev_side:
                continue
            k = 2 - j
            pa, pb = side, (side-1) % 4
            if side > 1:
                pa, pb = pb, pa
            if abs(a[2,j]) < 1e-15:
                if abs(a[1,j]) > 1e-15:
                    tau = (rhs - a[0,j]) / a[1,j]
                    if 0 < tau < tau_min:
                        r = e_i.eval_position(tau)
                        if self.is_position_in_bounds(r):
                            tau_min = tau
                            side_min = side
                            # if r[k] > rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pa, r)
                            # elif r[k] < -rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pb, r)
            else:
                d = a[1,j]**2 - 4*a[2,j]*(a[0,j] - rhs)
                if d > 1e-15:
                    pn = np.sqrt(d)
                    tau_p = (-a[1,j] + pn) / (2*a[2,j])
                    tau_n = (-a[1,j] - pn) / (2*a[2,j])
                    if 0 < tau_n < tau_min and 0 < tau_p < tau_min:
                        tau_a, tau_b = min(tau_n, tau_p), max(tau_n, tau_p)
                        r = e_i.eval_position(tau_a)
                        if self.is_position_in_bounds(r):
                            tau_min = tau_a
                            side_min = side
                            # if r[k] > rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pa, r)
                            # elif r[k] < -rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pb, r)
                        else:
                            r = e_i.eval_position(tau_b)
                            if self.is_position_in_bounds(r):
                                tau_min = tau_b
                                side_min = side
                                # if r[k] > rhsp:
                                #     _logger.debug('side %d pocket %d\nr = %s', side, pa, r)
                                # elif r[k] < -rhsp:
                                #     _logger.debug('side %d pocket %d\nr = %s', side, pb, r)
                    elif 0 < tau_n < tau_min:
                        r = e_i.eval_position(tau_n)
                        if self.is_position_in_bounds(r):
                            tau_min = tau_n
                            side_min = side
                            # if r[k] > rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pa, r)
                            # elif r[k] < -rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pb, r)
                    elif 0 < tau_p < tau_min:
                        r = e_i.eval_position(tau_p)
                        if self.is_position_in_bounds(r):
                            tau_min = tau_p
                            side_min = side
                            # if r[k] > rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pa, r)
                            # elif r[k] < -rhsp:
                            #     _logger.debug('side %d pocket %d\nr = %s', side, pb, r)
        if side_min is not None:
            return (e_i.t + tau_min, side_min)

    def _find_collision(self, e_i, e_j, double t_min):
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
           < np.sqrt(r_ij_0.dot(r_ij_0)) - ball_diameter:
            return None
        a_i, b_i = e_i.global_motion_coeffs
        a_j, b_j = e_j.global_motion_coeffs
        return self._find_collision_time(a_i, a_j, t0, t1)

    def _find_collision_time(self, a_i, a_j, double t0, double t1):
        d = a_i - a_j
        a_x, a_y = d[2, ::2]
        b_x, b_y = d[1, ::2]
        c_x, c_y = d[0, ::2]
        p = self._p
        p[0] = a_x**2 + a_y**2
        p[1] = 2 * (a_x*b_x + a_y*b_y)
        p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
        p[3] = 2 * b_x*c_x + 2 * b_y*c_y
        p[4] = c_x**2 + c_y**2 - 4 * ball_radius**2
        try:
            #roots = np.roots(p)
            roots = self.quartic_solve(p[::-1])
        except np.linalg.linalg.LinAlgError as err:
            #_logger.warning('LinAlgError occurred during solve for collision time:\np = %s\nerror:\n%s', p, err)
            return None
        #_logger.debug('roots: %s', roots)
        # # filter out possible complex-conjugate pairs of roots:
        # find_z, find_z_conj = self._find_z, self._find_z_conj
        # mask = self._mask; mask[:] = True
        # for n in range(2):
        #     i, z = find_z(roots)
        #     if z is not None:
        #         j = find_z_conj(roots, i, z)
        #         if j is not None:
        #             mask[i] = False; mask[i+j+1] = False
        #             roots = roots[mask[:len(roots)]]
        #         else:
        #             break
        #     else:
        #         break
        return min((t.real for t in roots
                    if t0 <= t.real <= t1
                    and t.imag**2 / (t.real**2 + t.imag**2) < _IMAG_TOLERANCE_SQRD),
                   default=None)

    @staticmethod
    def cubic_solve(p):
        a2, a1, a0 = p[2]/p[3], p[1]/p[3], p[0]/p[3]
        p = (3*a1 - a2**2) / 3.0
        q = (9*a1*a2 - 27*a0 - 2*a2**3) / 27.0
        d = q**2 + 4*p**3/27.0
        if d < 0:
            wc0 = 0.5 * (q + np.sqrt(d + 0j))
            wc1 = 0.5 * (q - np.sqrt(d + 0j))
        else:
            wc0 = 0.5 * (q + np.sqrt(d))
            wc1 = 0.5 * (q - np.sqrt(d))
        w0_mag = abs(wc0)**(1.0/3)
        w1_mag = abs(wc1)**(1.0/3)
        angle0 = np.angle(wc0) / 3
        angle1 = np.angle(wc1) / 3
        angles = 2*np.pi/3 * np.arange(3)
        w0 = w0_mag * np.exp(1j*(angles + angle0))
        w1 = w1_mag * np.exp(1j*(angles + angle1))
        z0 = w0 - p / (3*w0) - a2/3
        z1 = w1 - p / (3*w1) - a2/3
        # zs = remove_a_double_root(zs)
        # zs = remove_a_double_root(zs)
        # zs = remove_a_double_root(zs)
        # _logger.debug('zs:\n%s', '\n'.join(str(z) for z in zs))
        zs = np.hstack((z0, z1))
        return zs

    @staticmethod
    def quartic_solve(p):
        e, d, c, b, a = p
        if abs(p[-1]) / max(abs(p[:-1])) < 1e-10:
            return PoolPhysics.cubic_solve(p[:-1])
        Delta = 256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 144*a**2*c*d**2*e - 27*a**2*d**4 \
              + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 16*a*c**4*e \
              - 4*a*c**3*d**2 - 27*b**4*e**2 + 18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2
        # _logger.debug('Delta = %s', Delta)
        P = 8*a*c - 3*b**2
        R = (b**3 - 4*a*b*c + 8*a**2*d)
        D = 64*a**3*e - 16*a**2*c**2 + 16*a*b**2*c - 16*a**2*b*d - 3*b**4
        Delta_0 = c**2 - 3*b*d + 12*a*e
        Delta_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
        p = P / (8*a**2)
        q = R / (8*a**3)
        QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta + 0j)))
        Q_mag = abs(QQQ)**(1.0/3)
        Q = Q_mag * np.exp(
            1j * ( np.angle(QQQ) + 2*np.pi*np.arange(3) ) / 3.0
        )
        find_z, find_z_conj = PoolPhysics._find_z, PoolPhysics._find_z_conj
        # _logger.debug('Q:\n%s', '\n'.join(str(x) for x in Q))
        if Delta > 0:
            # if P < 0 and D < 0:
            #     _logger.debug('all roots are real and distinct')
            # elif P > 0 or D > 0:
            #     _logger.debug('all roots are complex and distinct')
            Q = Q[0]
        elif Delta < 0:
            # _logger.debug('two distinct real roots and a complex-conjugate pair of roots')
            i, z = find_z(Q)
            if z:
                j = find_z_conj(Q, i, z)
                if j:
                    Q = Q[i]
                else:
                    Q = Q[1]
            else:
                Q = Q[1]
        elif Delta == 0:
            if P < 0 and D < 0 and Delta_0 != 0:
                # _logger.debug('one real double root and two other real roots')
                Q = Q[0]
            elif D > 0 or (P > 0 and (D != 0 or R != 0)):
                # _logger.debug('one real double root and a complex-conjugate pair of roots')
                i, z = find_z(Q)
                if z:
                    j = find_z_conj(Q, i, z)
                    if j:
                        Q = Q[i]
                    else:
                        Q = Q[1]
                else:
                    Q = Q[1]
            elif Delta_0 == 0 and D != 0:
                # _logger.debug('one real triple root and one other real root')
                Q = Q[0]
            elif D == 0:
                if P < 0:
                    # _logger.debug('two real double roots')
                    Q = Q[0]
                elif P > 0 and R == 0:
                    # _logger.debug('two complex-conjugate double roots')
                    i, z = find_z(Q)
                    if z:
                        j = find_z_conj(Q, i, z)
                        if j:
                            Q = Q[i]
                        else:
                            Q = Q[1]
                    else:
                        Q = Q[1]
                elif Delta_0 == 0:
                    # _logger.debug('all roots are equal to -b / 4a')
                    return np.array(4*[-0.25 * b / a])
        S = 0.5*np.sqrt(-2.0*p/3 + (Q + Delta_0/Q) / (3.0*a) + 0j)
        return np.array([
            -b/(4*a) - S + 0.5*np.sqrt(-4*S**2 - 2*p + q/S + 0j),
            -b/(4*a) - S - 0.5*np.sqrt(-4*S**2 - 2*p + q/S + 0j),
            -b/(4*a) + S + 0.5*np.sqrt(-4*S**2 - 2*p - q/S + 0j),
            -b/(4*a) + S - 0.5*np.sqrt(-4*S**2 - 2*p - q/S + 0j),
        ])

    @staticmethod
    def _find_z(roots):
        return next(((i, z) for i, z in enumerate(roots) if abs(z.imag) > _IMAG_TOLERANCE), (None, None))

    @staticmethod
    def _find_z_conj(roots, i, z):
        return next((j for j, z_conj in enumerate(roots[i+1:])
                     if  abs(z.real - z_conj.real) < _ZERO_TOLERANCE
                     and abs(z.imag + z_conj.imag) < _ZERO_TOLERANCE), None)

    def is_position_in_bounds(self, np.ndarray r):
        sx, sz, R = self._sx, self._sz, _almost_ball_radius
        return  -sx <= r[0] - R \
            and  r[0] + R <= sx \
            and -sz <= r[2] - R \
            and  r[2] + R <= sz

    def is_position_near_pocket(self, np.ndarray r):
        sxcp, szcp = self._sxcp, self._szcp
        if r[0] < -sxcp:
            if r[2] < -szcp:
                return 1
            elif r[2] > szcp:
                return 2
        elif r[0] > sxcp:
            if r[2] < -szcp:
                return 3
            elif r[2] > szcp:
                return 4

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
            psi_ij[i,F_i] = np.arcsin(ball_diameter / r_ij_mag[i,F_i])
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

    def eval_energy(self, double t, balls=None):
        if balls is None:
            balls = self.balls_on_table
        velocities = self.eval_velocities(t, balls=balls)
        omegas = self.eval_angular_velocities(t, balls=balls)
        return 0.5 * ball_mass * (velocities**2).sum() + 0.5 * ball_I * (omegas**2).sum()

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
                    if d_ij - 2*ball_radius < -1e-6:
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
''' % (2*ball_radius, d_ij, r_i, r_j, self.t, event, e_i, e_j))

    def glyph_meshes(self, double t):
        if not self._velocity_meshes:
            from ..gl_rendering import Material, Mesh
            from ..gl_primitives import ArrowMesh
            from ..gl_techniques import EGA_TECHNIQUE #LAMBERT_TECHNIQUE
            self._velocity_material = Material(EGA_TECHNIQUE, #LAMBERT_TECHNIQUE,
                                               values={"u_color": [1.0, 0.0, 0.0, 0.0]})
            self._angular_velocity_material = Material(EGA_TECHNIQUE, #LAMBERT_TECHNIQUE,
                                                       values={'u_color': [0.0, 0.0, 1.0, 0.0]})
            self._velocity_meshes = {i: ArrowMesh(material=self._velocity_material,
                                                  head_radius=0.2*ball_radius,
                                                  head_length=0.5*ball_radius,
                                                  tail_radius=0.075*ball_radius,
                                                  tail_length=2*ball_radius)
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
                if v_mag > _ZERO_TOLERANCE:
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
                    mesh.world_matrix[3,:3] = r + (2*ball_radius)*y
                    meshes.append(mesh)
                omega = event.eval_angular_velocity(tau)
                omega_mag = np.sqrt(omega.dot(omega))
                if omega_mag > _ZERO_TOLERANCE:
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
                    mesh.world_matrix[3,:3] = r + (2*ball_radius)*y
                    meshes.append(mesh)
        return meshes
