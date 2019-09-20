import sys
import pickle
import numpy as np


from poolvr.physics.events import PhysicsEvent
import poolvr.physics.poly_solvers as poly_solvers


import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'BallsPenetratedInsanity.fsimulated.pickle.dump'
with open(fname, 'rb') as f:
    physics = pickle.load(f)
print(PhysicsEvent.events_str(physics.events[-6:]))

collision_event = physics.events[-3]
e_i, e_j = collision_event.e_i, collision_event.e_j
e_i.T = e_i.T_orig
e_j.T = e_j.T_orig
print('e_i =', e_i)
print('e_j =', e_j)

a_i = e_i.global_linear_motion_coeffs
a_j = e_j.global_linear_motion_coeffs
t0, t1 = max(e_i.t, e_j.t), min(e_i.t + e_i.T, e_j.t + e_j.T)
print('t0 =', t0, ', t1 =', t1)


R = e_i.ball_radius
t = poly_solvers.find_collision_time(a_i, a_j, R, t0, t1)
print('collision time:', t)


a_ji = a_i - a_j
a_x, a_y = a_ji[2, ::2]
b_x, b_y = a_ji[1, ::2]
c_x, c_y = a_ji[0, ::2]
p = np.empty(5, dtype=np.float64)
p[4] = a_x**2 + a_y**2
p[3] = 2 * (a_x*b_x + a_y*b_y)
p[2] = b_x**2 + 2 * (a_x*c_x + a_y*c_y) + b_y**2
p[1] = 2 * (b_x*c_x + b_y*c_y)
p[0] = c_x**2 + c_y**2 - 4 * R**2
roots = poly_solvers.c_quartic_solve(p)
print('roots:\n', roots)


if e_i > e_j:
    t0 = e_i.t - e_j.t
    t1 = min(e_j.T, t0 + e_i.T)
    a_j = e_j._a.copy()
    a_i = e_i.calc_global_linear_motion_coeffs(t0, e_i._a)
else:
    t0 = e_j.t - e_i.t
    t1 = min(e_i.T, t0 + e_j.T)
    a_i = e_i._a.copy()
    a_j = e_j.calc_global_linear_motion_coeffs(t0, e_j._a)
print('t0 =', t0, ', t1 =', t1)
t = poly_solvers.find_collision_time(a_i, a_j, R, t0, t1)
print('relative collision time:', t)
t += min(e_i.t, e_j.t)
print('global collision time:', t)


a_ji = a_i - a_j
a_x, a_y = a_ji[2, ::2]
b_x, b_y = a_ji[1, ::2]
c_x, c_y = a_ji[0, ::2]
p[4] = a_x**2 + a_y**2
p[3] = 2 * (a_x*b_x + a_y*b_y)
p[2] = b_x**2 + 2 * (a_x*c_x + a_y*c_y) + b_y**2
p[1] = 2 * (b_x*c_x + b_y*c_y)
p[0] = c_x**2 + c_y**2 - 4 * R**2
print('p = [ ' + ',  '.join('%5.5f' % m for m in p) + ' ]')
roots = poly_solvers.c_quartic_solve(p)
print('roots:\n', roots)

roots = poly_solvers.quartic_solve(p, only_real=True)
print('roots:\n', roots)
