import pickle
import sys
import logging
_DEBUG_LOGGING_FORMAT = '### %(asctime).19s.%(msecs).3s [%(levelname)s] %(name)s.%(funcName)s (%(filename)s:%(lineno)d) ###\n%(message)s\n'
logging.basicConfig(format=_DEBUG_LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
from poolvr.physics.events import PhysicsEvent, BallCollisionEvent, BallMotionEvent, BallSpinningEvent, BallStationaryEvent
from poolvr.physics.poly_solvers import f_find_collision_time, quartic_solve, f_quartic_solve, c_quartic_solve
from utils import plot_distance, sorrted, printit


fname = sys.argv[1]
logger.info('loading "%s"...', fname)
with open(fname, 'rb') as f:
    physics = pickle.load(f)


R = physics.ball_radius = 0.02625
i, j = physics.e_i.i, physics.e_j.i
ball_events = physics.ball_events
i_events = ball_events[i]
j_events = ball_events[j]
collisions = [e for e in physics.events if isinstance(e, BallCollisionEvent)]
i_collisions = [c for c in collisions if c.i == i or c.j == i]
j_collisions = [c for c in collisions if c.i == j or c.j == j]


# logger.info('''
# ball i events:

# %s
# ''', PhysicsEvent.events_str(sorted(i_events + i_collisions)))
# logger.info('''
# ball j events:

# %s
# ''', PhysicsEvent.events_str(sorted(j_events + j_collisions)))


plot_distance(physics, physics.i, physics.j)


def do_ij(e_i, e_j):
    t0, t1 = max(e_i.t, e_j.t), min(e_i.t + e_i.T, e_j.t + e_j.T)
    if t1 <= t0:
        return
    a_i = e_i.global_linear_motion_coeffs
    a_j = e_j.global_linear_motion_coeffs
    t_c = f_find_collision_time(a_i, a_j, R, t0, t1)
    a_ji = a_i - a_j
    a_x, a_y = a_ji[2, ::2]
    b_x, b_y = a_ji[1, ::2]
    c_x, c_y = a_ji[0, ::2]
    p = np.empty(5, dtype=np.float64)
    p[4] = a_x**2 + a_y**2
    p[3] = 2 * (a_x*b_x + a_y*b_y)
    p[2] = b_x**2 + b_y**2 + 2*(a_x*c_x + a_y*c_y)
    p[1] = 2 * (b_x*c_x + b_y*c_y)
    p[0] = c_x**2 + c_y**2 - 4 * R**2
    nproots = sorrted(np.roots(p[::-1]))
    froots = sorrted(f_quartic_solve(p))
    croots = sorrted(c_quartic_solve(p))
    roots = sorrted(quartic_solve(p))
    logger.info('''i, j = %d, %d
t0, t1 = %s, %s

      e_i: %s

      e_j: %s

      p = %s

      np.roots(p[::-1])  = %s
      f_quartic_solve(p) = %s
      c_quartic_solve(p) = %s
      quartic_solve(p)   = %s

      t_c = %s
''',
                e_i.i, e_j.i,
                t0, t1,
                e_i, e_j, p,
                printit(nproots),
                printit(froots),
                printit(croots),
                printit(roots),
                t_c)
    return np.hstack((nproots, froots, croots, roots))


for event in physics.events:
    if hasattr(event, 'T_orig'):
        event.T = event.T_orig



t = physics.t_penetrated
events = physics.events = [e for e in physics.events if e.t < t]


for i, events in list(physics.ball_events.items()):
    physics.ball_events[i] = [e for e in events if e.t < t]
for e in physics.events:
    if isinstance(e, BallMotionEvent):
        physics._ball_motion_events[e.i] = e
        physics._ball_spinning_events.pop(e.i, None)
        if e.i in physics._balls_at_rest:
            physics._balls_at_rest.remove(e.i)
    elif isinstance(e, BallStationaryEvent):
        physics._ball_motion_events.pop(e.i, None)
        if isinstance(e, BallSpinningEvent):
            physics._ball_spinning_events[e.i] = e
        else:
            physics._ball_spinning_events.pop(e.i, None)
            physics._balls_at_rest.add(e.i)
for i in physics._ball_motion_events.keys():
    physics._collisions[i] = {}
physics._rail_collisions = {}


# logger.info('physics._ball_motion_events:\n%s',
#             '\n'.join('%s: %s' % it for it in sorted(physics._ball_motion_events.items())))
n = 6
logger.info('last %d events before penetration by %d,%d at t = %s:\n%s',
            n, physics.i, physics.j,
            t, PhysicsEvent.events_str(physics.events[-n:]))

def do_next_event():
    next_event = physics._determine_next_event()
    logger.info('next event: %s', next_event)
    physics._add_event(next_event)
