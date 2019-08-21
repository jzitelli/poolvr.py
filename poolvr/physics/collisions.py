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
                  g=9.81,
                  nP=4000):
    r_ij = r_j - r_i
    r_ij_mag_sqrd = dot(r_ij, r_ij)
    assert  abs(r_ij_mag_sqrd - 4 * R**2)  <  1e-11 * R**2, "r_ij_mag_sqrd - 4 * R**2 = %s" % (r_ij_mag_sqrd - 4 * R**2)
    r_ij_mag = sqrt(r_ij_mag_sqrd)
    z_loc = _k
    y_loc = r_ij / r_ij_mag
    x_loc = cross(y_loc, z_loc)
    G = np.vstack((x_loc, y_loc, z_loc))
    v_i = dot(G, v_i)
    v_j = dot(G, v_j)
    omega_i = dot(G, omega_i)
    omega_j = dot(G, omega_j)
    r_ic = np.array([0.0,  R, 0.0])
    r_jc = np.array([0.0, -R, 0.0])
    v_ij = v_j - v_i
    u_iR = v_i + R * cross(_k, omega_i)
    u_iR_mag = sqrt(dot(u_iR, u_iR))
    u_jR = v_j + R * cross(_k, omega_j)
    u_jR_mag = sqrt(dot(u_jR, u_jR))
    u_iC = v_i - cross(r_ic, omega_i)
    u_jC = v_j - cross(r_jc, omega_j)
    u_ijC = u_jC - u_iC
    # u_ijC = u_iC - u_jC
    u_ijC_xz = u_ijC[::2]
    u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
    deltaP = 0.5 * (1 + e) * M * abs(v_ij[1]) / nP
    W_f = float('inf')
    W_c = None
    W = 0
    niters = 0
    v_is = [v_i]
    v_js = [v_j]
    omega_is = [omega_i]
    omega_js = [omega_j]
    while v_ij[1] < 0 or W < W_f:
        if u_ijC_xz_mag < 1e-12:
            _logger.debug('no slip at ball-ball contact')
            deltaP_1 = deltaP_2 = 0
            deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
        else:
            _logger.debug('slip at ball-ball contact: %s', u_ijC_xz_mag)
            deltaP_1 = -mu_b * deltaP * u_ijC[0] / u_ijC_xz_mag
            deltaP_2 = -mu_b * deltaP * u_ijC[2] / u_ijC_xz_mag
            if deltaP_2 > 0:
                # i.e. u_ijC_z < 0
                deltaP_ix = deltaP_iy = 0
                if u_jR_mag < 1e-12:
                    _logger.debug('no slip at j-table contact')
                    deltaP_jx = deltaP_jy = 0
                else:
                    deltaP_jx = -mu_s * (u_jR[0] / u_jR_mag) * deltaP_2
                    deltaP_jy = -mu_s * (u_jR[1] / u_jR_mag) * deltaP_2
            else:
                deltaP_jx = deltaP_jy = 0
                if u_iR_mag < 1e-12:
                    _logger.debug('no slip at i-table contact')
                    deltaP_ix = deltaP_iy = 0
                else:
                    deltaP_ix = mu_s * (u_iR[0] / u_iR_mag) * deltaP_2
                    deltaP_iy = mu_s * (u_iR[1] / u_iR_mag) * deltaP_2
        # velocity changes:
        deltaV_ix = (deltaP_1 + deltaP_ix) / M
        deltaV_iy = (-deltaP + deltaP_iy) / M
        deltaV_jx = (-deltaP_1 + deltaP_jx) / M
        deltaV_jy = (deltaP + deltaP_jy) / M
        # angular velocity changes (due to frictional forces):
        deltaOm_i = 5/(2*M*R) * np.array([(deltaP_2 + deltaP_iy),
                                          (-deltaP_ix),
                                          (-deltaP_1)])
        deltaOm_j = 5/(2*M*R) * np.array([(-deltaP_2 + deltaP_jy),
                                          (-deltaP_jx),
                                          (deltaP_1)])
        # update velocities:
        v_i0 = v_i
        v_i = v_i0 + np.array([deltaV_ix, deltaV_iy, 0])
        v_j0 = v_j
        v_j = v_j0 + np.array([deltaV_jx, deltaV_jy, 0])
        v_ijy0 = v_ij[1]
        v_ij = v_j - v_i
        # update angular velocities:
        omega_i0 = omega_i
        omega_i = omega_i0 + deltaOm_i
        omega_j0 = omega_j
        omega_j = omega_j0 + deltaOm_j
        # update ball-table slips:
        u_iR = v_i + R * cross(_k, omega_i)
        u_jR = v_j + R * cross(_k, omega_j)
        u_iR_mag = sqrt(dot(u_iR, u_iR))
        u_jR_mag = sqrt(dot(u_jR, u_jR))
        # update ball-ball slip:
        u_iC = v_i - cross(r_ic, omega_i)
        u_jC = v_j - cross(r_jc, omega_j)
        u_ijC = u_jC - u_iC
        # u_ijC = u_iC - u_jC
        u_ijC_xz = u_ijC[::2]
        u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
        # increment work:
        deltaW = 0.5 * deltaP * abs(v_ijy0 + v_ij[1])
        W += deltaW
        v_is.append(v_i)
        v_js.append(v_j)
        omega_is.append(omega_i)
        omega_js.append(omega_j)
        niters += 1
        if W_c is None and v_ij[1] > 0:
            W_c = W
            W_f = (1 + e**2) * W_c
            niters_c = niters
            _logger.debug('''
            END OF COMPRESSION PHASE
            W_c = %s
            W_f = %s
            niters_c = %s
            ''', W_c, W_f, niters_c)
    _logger.debug('''
    END OF RESTITUTION PHASE
    niters_r = %s
    ''', niters - niters_c)

    return dot(G.T, v_i), dot(G.T, omega_i), dot(G.T, v_j), dot(G.T, omega_j)
    # import matplotlib.pyplot as plt
    # deltaPs = deltaP*np.arange(niters+1)

    # plt.figure()
    # plt.plot(deltaPs, np.array(v_is)[:,1], label='ball i')
    # plt.plot(deltaPs, np.array(v_js)[:,1], label='ball j')
    # plt.xlabel('cumulative impulse along y-axis')
    # plt.ylabel('velocity along y-axis')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(deltaPs, np.array(omega_is)[:,0], label='ball i')
    # plt.plot(deltaPs, np.array(omega_js)[:,0], label='ball j')
    # plt.xlabel('cumulative impulse along y-axis')
    # plt.ylabel('angular velocity along x-axis')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(deltaPs, np.array(omega_is)[:,2], label='ball i')
    # plt.plot(deltaPs, np.array(omega_js)[:,2], label='ball j')
    # plt.xlabel('cumulative impulse along y-axis')
    # plt.ylabel('angular velocity along z-axis')
    # plt.legend()
    # plt.show()
