import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])


def test_collide_balls():
    r_i = np.array([0.0, 0.0, 0.0])
    rd = 1 - 2 * np.random.rand(3); rd[2] = 0
    rd_mag_sqrd = np.dot(rd, rd)
    R = 1.125 * 0.0254
    while rd_mag_sqrd < 0.05:
        rd = 1 - 2 * np.random.rand(3); rd[2] = 0
        rd_mag_sqrd = np.dot(rd, rd)
    rd_mag = np.sqrt(rd_mag_sqrd)
    y_loc = rd / rd_mag
    z_loc = _k
    x_loc = np.cross(y_loc, z_loc)
    r_c = r_i + R * y_loc
    r_j = r_i + 2 * R * y_loc
    v_i = 3 * rd - 0.1 * x_loc
    v_j = np.zeros(3, dtype=np.float64)
    omega_i = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    collide_balls(r_c,
                  r_i, v_i, omega_i,
                  r_j, v_j, omega_j,
                  0.9,
                  0.2,
                  0.06,
                  0.17,
                  1.125*0.0254,
                  9.81,
                  10000)
