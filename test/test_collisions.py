import logging
_logger = logging.getLogger(__name__)
import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])


def test_collide_balls():
    e = 0.9
    mu_s = 0 #0.2
    mu_b = 0 #0.06
    M = 0.17
    R = 1.125 * 0.0254
    r_i = np.array([0.0, 0.0, 0.0])
    # rd = 1 - 2 * np.random.rand(3); rd[2] = 0
    rd = np.array([0.0, 1.0, 0.0])
    rd_mag_sqrd = np.dot(rd, rd)
    # while rd_mag_sqrd < 0.05:
    #     rd = 1 - 2 * np.random.rand(3); rd[2] = 0
    #     rd_mag_sqrd = np.dot(rd, rd)
    r_j = r_i + 2 * R * rd / rd_mag_sqrd
    r_ij = r_j - r_i
    r_ij_mag = np.sqrt(np.dot(r_ij, r_ij))
    y_loc = r_ij / r_ij_mag
    z_loc = _k
    x_loc = np.cross(y_loc, z_loc)
    r_c = r_i + R * y_loc
    # v_i = 1 * rd - 0.1 * x_loc
    v_i = 3 * rd
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
    _logger.debug('''
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
                  v_ijy, v_ijy1,
                  v_ijy1 / v_ijy,
                  np.sqrt(-v_ijy1 / v_ijy))
