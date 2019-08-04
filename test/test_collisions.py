import logging
_logger = logging.getLogger(__name__)
import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])


def test_collide_balls():
    e = 0.9
    mu_s = 0.2
    mu_b = 0.06
    M = 0.17
    R = 1.125 * 0.0254
    r_i = np.array([0.0, 0.0, 0.0])
    rd = np.array([0.0, 1.0, 0.0])
    r_j = r_i + 2 * R * rd
    r_ij = r_j - r_i
    r_ij_mag = np.sqrt(np.dot(r_ij, r_ij))
    y_loc = r_ij / r_ij_mag
    z_loc = _k
    x_loc = np.cross(y_loc, z_loc)
    r_c = r_i + R * y_loc
    # v_i = 3 * rd
    v_i = 0.5 * (0.5*np.sqrt(3)*rd + 0.5*x_loc)
    v_j = np.zeros(3, dtype=np.float64)
    omega_i = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    v_i1, omega_i1, v_j1, omega_j1 = collide_balls(r_c,
                                                   r_i, v_i, omega_i,
                                                   r_j, v_j, omega_j,
                                                   e, mu_s, mu_b,
                                                   M, R,
                                                   9.81,
                                                   4000)
    v_ijy = np.dot(v_j, y_loc) - np.dot(v_i, y_loc)
    v_ijy1 = np.dot(v_j1, y_loc) - np.dot(v_i1, y_loc)
    v_il = np.array([np.dot(v_i, x_loc), np.dot(v_i, y_loc), 0])
    v_jl = np.array([np.dot(v_j, x_loc), np.dot(v_j, y_loc), 0])
    v_i1l = np.array([np.dot(v_i1, x_loc), np.dot(v_i1, y_loc), 0])
    v_j1l = np.array([np.dot(v_j1, x_loc), np.dot(v_j1, y_loc), 0])
    _logger.debug('''
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

(local frame):
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

v_ijy  = %s
v_ijy1 = %s

     |v_ijy1 / v_ijy|  = %s
sqrt(|v_ijy1 / v_ijy|) = %s
''',
                  v_i, v_i1, v_j, v_j1,
                  v_il, v_i1l, v_jl, v_j1l,
                  v_ijy, v_ijy1,
                  v_ijy1 / v_ijy,
                  np.sqrt(-v_ijy1 / v_ijy))
