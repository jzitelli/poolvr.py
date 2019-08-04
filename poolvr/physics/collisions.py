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
from numpy import dot, sqrt, cross

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
    r_ij_mag = sqrt(dot(r_ij, r_ij))
    r_ic = r_c - r_i
    r_jc = r_c - r_j
    z_loc = _k
    y_loc = r_ij / r_ij_mag
    x_loc = cross(y_loc, z_loc)
    u_iR = v_i + R * cross(_k, omega_i)
    u_iR_mag = sqrt(dot(u_iR, u_iR))
    u_jR = v_j + R * cross(_k, omega_j)
    u_jR_mag = sqrt(dot(u_jR, u_jR))
    u_iR_x = dot(u_iR, x_loc)
    u_iR_y = dot(u_iR, y_loc)
    u_jR_x = dot(u_jR, x_loc)
    u_jR_y = dot(u_jR, y_loc)
    u_iC = v_i - cross(r_ic, omega_i)
    u_jC = v_j - cross(r_jc, omega_j)
    u_ijC = u_jC - u_iC
    u_ijC_xz = u_ijC - dot(u_ijC, y_loc) * y_loc
    u_ijC_x = dot(u_ijC_xz, x_loc)
    u_ijC_z = dot(u_ijC_xz, z_loc)
    u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
    v_ij = v_j - v_i
    v_ijy = dot(v_ij, y_loc)
    v_ix, v_iy = dot(v_i, x_loc), dot(v_i, y_loc)
    v_jx, v_jy = dot(v_j, x_loc), dot(v_j, y_loc)
    omega_ix, omega_iy, omega_iz = dot(omega_i, x_loc), dot(omega_i, y_loc), dot(omega_i, z_loc)
    omega_jx, omega_jy, omega_jz = dot(omega_j, x_loc), dot(omega_j, y_loc), dot(omega_j, z_loc)
    deltaP = 0.5 * (1 + e) * M * np.abs(dot(v_ij, y_loc)) / nP
    W_f = float('inf')
    W_c = None
    W = 0
    niters = 0

    while v_ijy < 0 or W < W_f:
        if u_ijC_xz_mag < 1e-8:
            _logger.debug('no slip at ball-ball contact')
            deltaP_1 = deltaP_2 = 0
            deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
        else:
            _logger.debug('slip at ball-ball contact: %s', u_ijC_xz_mag)
            deltaP_1 = -mu_b * deltaP * u_ijC_x / u_ijC_xz_mag
            deltaP_2 = -mu_b * deltaP * u_ijC_z / u_ijC_xz_mag
            if u_iR_mag < 1e-8:
                _logger.debug('no slip at i-table contact')
                deltaP_ix = deltaP_iy = 0
            else:
                _logger.debug('slip at i-table contact: %s', u_iR_mag)
                deltaP_ix = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_iR_x / u_iR_mag)
                deltaP_iy = -mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_iR_y / u_iR_mag)
            if u_jR_mag < 1e-8:
                _logger.debug('no slip at j-table contact')
                deltaP_jx = deltaP_jy = 0
            else:
                _logger.debug('slip at j-table contact: %s', u_jR_mag)
                deltaP_jx = mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_jR_x / u_jR_mag)
                deltaP_jy = mu_b * mu_s * deltaP * (u_ijC_z / u_ijC_xz_mag) * (u_jR_y / u_jR_mag)
        # velocity changes within plane of impact:
        deltaV_ix = (deltaP_1 + deltaP_ix) / M
        deltaV_iy = (-deltaP + deltaP_iy) / M
        deltaV_jx = (-deltaP_1 + deltaP_jx) / M
        deltaV_jy = (deltaP + deltaP_jy) / M
        deltaV_ijy = deltaV_jy - deltaV_iy
        # update velocities:
        v_ix0, v_iy0 = v_ix, v_iy
        v_ix         = v_ix0 + deltaV_ix
        v_iy         = v_iy0 + deltaV_iy
        v_jx0, v_jy0 = v_jx, v_jy
        v_jx         = v_jx0 + deltaV_jx
        v_jy         = v_jy0 + deltaV_jy
        v_ijy0       = v_ijy
        v_ijy        = v_ijy0 + deltaV_ijy
        v_i = v_ix*x_loc + v_iy*y_loc
        v_j = v_jx*x_loc + v_jy*y_loc
        # angular velocity changes (due to frictional forces):
        deltaOm_ix = 5/(2*M*R) * (deltaP_2 + deltaP_iy)
        deltaOm_iy = 5/(2*M*R) * (-deltaP_ix)
        deltaOm_iz = 5/(2*M*R) * (-deltaP_1)
        deltaOm_jx = 5/(2*M*R) * (-deltaP_2 + deltaP_jy)
        deltaOm_jy = 5/(2*M*R) * (-deltaP_jx)
        deltaOm_jz = 5/(2*M*R) * (deltaP_1)
        # update angular velocities:
        omega_ix0 = omega_ix
        omega_ix  = omega_ix0 + deltaOm_ix
        omega_iy0 = omega_iy
        omega_iy  = omega_iy0 + deltaOm_iy
        omega_iz0 = omega_iz
        omega_iz  = omega_iz0 + deltaOm_iz
        omega_jx0 = omega_jx
        omega_jx  = omega_jx0 + deltaOm_jx
        omega_jy0 = omega_jy
        omega_jy  = omega_jy0 + deltaOm_jy
        omega_jz0 = omega_jz
        omega_jz  = omega_jz0 + deltaOm_jz
        omega_i = omega_ix*x_loc + omega_iy*y_loc + omega_iz*z_loc
        omega_j = omega_jx*x_loc + omega_jy*y_loc + omega_jz*z_loc
        # update ball-table slips:
        u_iR = v_i + R * cross(_k, omega_i)
        u_jR = v_j + R * cross(_k, omega_j)
        u_iR_mag = sqrt(dot(u_iR, u_iR))
        u_jR_mag = sqrt(dot(u_jR, u_jR))
        u_iR_x = dot(u_iR, x_loc)
        u_iR_y = dot(u_iR, y_loc)
        u_jR_x = dot(u_jR, x_loc)
        u_jR_y = dot(u_jR, y_loc)
        # update ball-ball slip:
        u_iC = v_i - cross(r_ic, omega_i)
        u_jC = v_j - cross(r_jc, omega_j)
        u_ijC = u_jC - u_iC
        u_ijC_xz = u_ijC - dot(u_ijC, y_loc) * y_loc
        u_ijC_x = dot(u_ijC_xz, x_loc)
        u_ijC_z = dot(u_ijC_xz, z_loc)
        u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
        v_ij = v_j - v_i
        v_ijy = dot(v_ij, y_loc)
        v_ix, v_iy = dot(v_i, x_loc), dot(v_i, y_loc)
        v_jx, v_jy = dot(v_j, x_loc), dot(v_j, y_loc)
        # increment work:
        deltaW = 0.5 * deltaP * deltaV_ijy
        W += deltaW
        niters += 1
        if W_c is None and v_ijy > 0:
            W_c = W
            W_f = (1 + e**2) * W_c
            _logger.debug('''
            end of compression phase
            W_c = %s
            W_f = %s
            niters = %s
            ''', W_c, W_f, niters)
    _logger.debug('''
    end of restitution phase
    niters = %s
    ''', niters)

    return v_i, omega_i, v_j, omega_j
