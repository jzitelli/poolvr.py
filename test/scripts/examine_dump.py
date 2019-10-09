import pickle
import sys
import logging
_DEBUG_LOGGING_FORMAT = '### %(asctime).19s.%(msecs).3s [%(levelname)s] %(name)s.%(funcName)s (%(filename)s:%(lineno)d) ###\n%(message)s\n'
logging.basicConfig(format=_DEBUG_LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
from poolvr.physics.events import PhysicsEvent, BallCollisionEvent
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


logger.info('''
ball i events:

%s
''', PhysicsEvent.events_str(sorted(i_events + i_collisions)))
logger.info('''
ball j events:

%s
''', PhysicsEvent.events_str(sorted(j_events + j_collisions)))


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
    p[2] = b_x**2 + 2*a_x*c_x + 2*a_y*c_y + b_y**2
    p[1] = 2 * b_x*c_x + 2 * b_y*c_y
    p[0] = c_x**2 + c_y**2 - 4 * R**2
    nproots = sorrted(np.roots(p[::-1]))
    froots = sorrted(f_quartic_solve(p))
    croots = sorrted(c_quartic_solve(p))
    roots = sorrted(quartic_solve(p))
    logger.info('''i,j = %d,%d

e_i: %s

e_j: %s

p = %s

np.roots(p[::-1])  = %s
f_quartic_solve(p) = %s
c_quartic_solve(p) = %s
quartic_solve(p)   = %s

t0, t1 = %s, %s

t_c = %s
''',
                e_i.i, e_j.i,
                e_i, e_j, p,
                printit(nproots),
                printit(froots),
                printit(croots),
                printit(roots),
                t0, t1, t_c)
    return np.hstack((nproots, froots, croots, roots))


# logger.info(PhysicsEvent.events_str(sorted(set(i_collisions + j_collisions))))
# logger.info(PhysicsEvent.events_str(sorted(i_collisions + i_events)[-10:]))
# logger.info(PhysicsEvent.events_str(sorted(j_collisions + j_events)[-10:]))

t = 0.1120
a_events = physics.find_active_events(t, balls=[physics.i, physics.j])

logger.info('''active events at
  t = %s:
%s
''', t, '\n'.join(str(e) for e in a_events))


for event in a_events:
    if hasattr(event, 'T_orig'):
        event.T = event.T_orig
do_ij(*a_events)
