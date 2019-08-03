"""
This module implements the ball-ball collision model described in: ::

  NUMERICAL SIMULATIONS OF THE FRICTIONAL COLLISIONS
  OF SOLID BALLS ON A ROUGH SURFACE
  S. Mathavan,  M. R. Jackson,  R. M. Parkin
  DOI: 10.1007/s12283-014-0158-y
  International Sports Engineering Association
  2014

"""
import logging
_logger = logging.getLogger(__name__)
import numpy as np


INCH2METER = 0.0254
_k = np.array([0, 0, 1], dtype=np.float64)


def collide_balls(r_c,
                  r_i, v_i, omega_i,
                  r_j, v_j, omega_j,
                  e,
                  mu_s,
                  mu_b,
                  M,
                  R,
                  g,
                  nP):
    r_ij = r_j - r_i
    r_ij_mag = np.sqrt(np.dot(r_ij, r_ij))
    r_ic = r_c - r_i
    r_jc = r_c - r_j
    z_loc = _k
    y_loc = r_ij / r_ij_mag
    x_loc = np.cross(y_loc, z_loc)
    v_ij = v_j - v_i
    u_iR = v_i + R * np.cross(_k, omega_i)
    u_iR_mag = np.sqrt(np.dot(u_iR, u_iR))
    u_jR = v_j + R * np.cross(_k, omega_j)
    u_jR_mag = np.sqrt(np.dot(u_jR, u_jR))
    u_iR_x = np.dot(u_iR, x_loc)
    u_iR_y = np.dot(u_iR, y_loc)
    u_jR_x = np.dot(u_jR, x_loc)
    u_jR_y = np.dot(u_jR, y_loc)
    u_iC = v_i - np.cross(r_ic, omega_i)
    u_jC = v_j - np.cross(r_jc, omega_j)
    u_ijC = u_jC - u_iC
    u_ijC_x = np.dot(u_ijC, x_loc)
    u_ijC_z = np.dot(u_ijC, z_loc)
    u_ijC_xz_mag = np.sqrt(u_ijC_x**2 + u_ijC_z**2)
    v_ix, v_iy = v_i[:2]
    v_jx, v_jy = v_j[:2]
    omega_ix, omega_iy, omega_iz = omega_i
    omega_jx, omega_jy, omega_jz = omega_j
    deltaP = (1 + e) * M * np.abs(np.dot(v_ij, y_loc)) / nP
    v_ijy = np.dot(v_ij, y_loc)
    W_f = float('inf')
    W_c = None
    W = 0
    niters = 0

    while v_ijy < 0 or W < W_f:
        _logger.debug('''

v_ijy = %s
W = %s
W_f = %s

        ''', v_ijy, W, W_f)
        if u_ijC_xz_mag < 1e-9:
            _logger.debug('no slip at ball-ball contact')
            deltaP_1 = deltaP_2 = 0
        else:
            _logger.debug('slip at ball-ball contact: %s', u_ijC_xz_mag)
            deltaP_1 = -mu_b * deltaP * u_ijC_x / u_ijC_xz_mag
            deltaP_2 = -mu_b * deltaP * u_ijC_z / u_ijC_xz_mag

        if u_iR_mag < 1e-9:
            _logger.debug('no slip at i-table contact')
            deltaP_ix = deltaP_iy = 0
        else:
            _logger.debug('slip at i-table contact: %s', u_iR_mag)
            deltaP_ix = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_iR_x / u_iR_mag)
            deltaP_iy = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_iR_y / u_iR_mag)
        if u_jR_mag < 1e-9:
            _logger.debug('no slip at j-table contact')
            deltaP_jx = deltaP_jy = 0
        else:
            _logger.debug('slip at j-table contact: %s', u_jR_mag)
            deltaP_jx = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_jR_x / u_jR_mag)
            deltaP_jy = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_jR_y / u_jR_mag)

        deltaV_ix = (deltaP_1 + deltaP_ix) / M
        deltaV_iy = (-deltaP + deltaP_iy) / M
        v_ix0, v_iy0 = v_ix, v_iy
        v_ix = v_ix0 + deltaV_ix
        v_iy = v_iy0 + deltaV_iy

        deltaV_jx = (-deltaP_1 + deltaP_jx) / M
        deltaV_jy = (deltaP + deltaP_jy) / M
        v_jx0, v_jy0 = v_jx, v_jy
        v_jx = v_jx0 + deltaV_jx
        v_jy = v_jy0 + deltaV_jy

        deltaV_ijy = (v_jy - v_iy) - (v_jy0 - v_iy0)
        v_ijy0 = v_ijy
        v_ijy = v_ijy0 + deltaV_ijy


        deltaW = 0.5 * deltaP * deltaV_ijy
        W += deltaW

        if W_c is None and v_ijy > 0:
            W_c = W
            W_f = (1 - e**2) * W_c
            W = 0
            _logger.debug('''

end of compression phase

W_c = %s
W_f = %s
niters = %s

''',
                          W_c, W_f, niters)

        deltaOm_ix = 5/(2*M*R) * (deltaP_2 + deltaP_iy)
        deltaOm_iy = 5/(2*M*R) * (-deltaP_ix)
        deltaOm_iz = 5/(2*M*R) * (-deltaP_1)
        omega_ix0 = omega_ix
        omega_ix = omega_ix0 + deltaOm_ix
        omega_iy0 = omega_iy
        omega_iy = omega_iy0 + deltaOm_iy
        omega_iz0 = omega_iz
        omega_iz = omega_iz0 + deltaOm_iz

        deltaOm_jx = 5/(2*M*R) * (-deltaP_2 + deltaP_jy)
        deltaOm_jy = 5/(2*M*R) * (-deltaP_jx)
        deltaOm_jz = 5/(2*M*R) * (deltaP_1)
        omega_jx0 = omega_jx
        omega_jx = omega_jx0 + deltaOm_jx
        omega_jy0 = omega_jy
        omega_jy = omega_jy0 + deltaOm_jy
        omega_jz0 = omega_jz
        omega_jz = omega_jz0 + deltaOm_jz

        v_i = np.array([v_ix, v_iy, 0])
        omega_i = np.array([omega_ix, omega_iy, omega_iz])

        v_j = np.array([v_jx, v_jy, 0])
        omega_j = np.array([omega_jx, omega_jy, omega_jz])

        u_iC = v_i - np.cross(r_ic, omega_i)
        u_jC = v_j - np.cross(r_jc, omega_j)
        u_ijC = u_jC - u_iC
        u_ijC_x = np.dot(u_ijC, x_loc)
        u_ijC_z = np.dot(u_ijC, z_loc)
        u_ijC_xz_mag = np.sqrt(u_ijC_x**2 + u_ijC_z**2)

        u_iR = v_i + R * np.cross(_k, omega_i)
        u_iR_mag = np.sqrt(np.dot(u_iR, u_iR))
        u_jR = v_j + R * np.cross(_k, omega_j)
        u_jR_mag = np.sqrt(np.dot(u_jR, u_jR))
        u_iR_x = np.dot(u_iR, x_loc)
        u_iR_y = np.dot(u_iR, y_loc)
        u_jR_x = np.dot(u_jR, x_loc)
        u_jR_y = np.dot(u_jR, y_loc)
        niters += 1
#         _logger.debug('''
# deltaP = %s
# deltaP_1 = %s
# deltaP_2 = %s
# deltaP_ix = %s
# deltaP_iy = %s
# deltaP_jx = %s
# deltaP_jy = %s
# deltaV_ijy = %s
# v_ijy = %s
# ''',
#                       deltaP, deltaP_1, deltaP_2,
#                       deltaP_ix, deltaP_iy, deltaP_jx, deltaP_jy,
#                       deltaV_ijy, v_ijy)

    _logger.debug('''
end of restitution phase

niters = %s
''',
                  niters)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
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
