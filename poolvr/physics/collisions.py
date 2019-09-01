"""
This module implements the ball-ball collision model described in: ::

  NUMERICAL SIMULATIONS OF THE FRICTIONAL COLLISIONS
  OF SOLID BALLS ON A ROUGH SURFACE
  S. Mathavan,  M. R. Jackson,  R. M. Parkin
  DOI: 10.1007/s12283-014-0158-y
  International Sports Engineering Association
  2014

"""
from logging import getLogger
_logger = getLogger(__name__)
from math import sqrt
import numpy as np
from numpy import dot, cross


INCH2METER = 0.0254
INF = float('inf')
_k = np.array([0, 1, 0], dtype=np.float64)


def collide_balls(r_c,
                  r_i, v_i, omega_i,
                  r_j, v_j, omega_j,
                  e=0.89,
                  mu_s=0.21,
                  mu_b=0.05,
                  M=0.1406,
                  R=0.02625,
                  g=9.81,
                  deltaP=None,
                  return_all=False):
    r_ij = r_j - r_i
    r_ij_mag_sqrd = dot(r_ij, r_ij)
    # D = 2*R
    #assert  abs(r_ij_mag_sqrd - D**2) / D**2  <  1e-4, "abs(r_ij_mag_sqrd - D**2) / D**2 = %s" % (abs(r_ij_mag_sqrd - D**2) / D**2)
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
    u_iR = v_i + R * cross(_k, omega_i)
    u_iR_mag = sqrt(dot(u_iR, u_iR))
    u_jR = v_j + R * cross(_k, omega_j)
    u_jR_mag = sqrt(dot(u_jR, u_jR))
    u_iC = v_i - cross(r_ic, omega_i)
    u_jC = v_j - cross(r_jc, omega_j)
    u_ijC = u_iC - u_jC
    u_ijC_xz = u_ijC[::2]
    u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
    v_ij = v_j - v_i
    if deltaP is None:
        deltaP = 0.5 * (1 + e) * M * abs(v_ij[1]) / 1000
    W_f = INF
    W_c = None
    W = 0
    niters = 0
    if return_all:
        v_is = [v_i]
        v_js = [v_j]
        omega_is = [omega_i]
        omega_js = [omega_j]
    while v_ij[1] < 0 or W < W_f:
        if u_ijC_xz_mag < 1e-16:
            deltaP_1 = deltaP_2 = 0
            deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
        else:
            deltaP_1 = -mu_b * deltaP * u_ijC[0] / u_ijC_xz_mag
            if abs(u_ijC[2]) < 1e-16:
                deltaP_2 = 0
                deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
            else:
                deltaP_2 = -mu_b * deltaP * u_ijC[2] / u_ijC_xz_mag
                if deltaP_2 > 0:
                    deltaP_ix = deltaP_iy = 0
                    if u_jR_mag == 0:
                        deltaP_jx = deltaP_jy = 0
                    else:
                        deltaP_jx = -mu_s * (u_jR[0] / u_jR_mag) * deltaP_2
                        deltaP_jy = -mu_s * (u_jR[1] / u_jR_mag) * deltaP_2
                else:
                    deltaP_jx = deltaP_jy = 0
                    if u_iR_mag == 0:
                        deltaP_ix = deltaP_iy = 0
                    else:
                        deltaP_ix = mu_s * (u_iR[0] / u_iR_mag) * deltaP_2
                        deltaP_iy = mu_s * (u_iR[1] / u_iR_mag) * deltaP_2
        # velocity changes:
        deltaV_ix = (deltaP_1  + deltaP_ix) / M
        deltaV_iy = (-deltaP   + deltaP_iy) / M
        deltaV_jx = (-deltaP_1 + deltaP_jx) / M
        deltaV_jy = (deltaP    + deltaP_jy) / M
        # angular velocity changes (due to frictional forces):
        deltaOm_i = 5/(2*M*R) * np.array([( deltaP_2 + deltaP_iy),
                                          (-deltaP_ix),
                                          (-deltaP_1)])
        deltaOm_j = 5/(2*M*R) * np.array([( deltaP_2 + deltaP_jy),
                                          (-deltaP_jx),
                                          (-deltaP_1)])
        # update velocities:
        v_i0 = v_i
        v_i = v_i0 + np.array([deltaV_ix, deltaV_iy, 0])
        v_j0 = v_j
        v_j = v_j0 + np.array([deltaV_jx, deltaV_jy, 0])
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
        u_ijC = u_iC - u_jC
        u_ijC_xz = u_ijC[::2]
        u_ijC_xz_mag = sqrt(dot(u_ijC_xz, u_ijC_xz))
        # increment work:
        v_ijy0 = v_ij[1]
        v_ij = v_j - v_i
        deltaW = 0.5 * deltaP * abs(v_ijy0 + v_ij[1])
        W += deltaW
        niters += 1
        if return_all:
            v_is.append(v_i)
            v_js.append(v_j)
            omega_is.append(omega_i)
            omega_js.append(omega_j)
        if W_c is None and v_ij[1] > 0:
            W_c = W
            W_f = (1 + e**2) * W_c
            # niters_c = niters
            # _logger.debug('''
            # END OF COMPRESSION PHASE
            # W_c = %s
            # W_f = %s
            # niters_c = %s
            # ''', W_c, W_f, niters_c)
    # _logger.debug('''
    # END OF RESTITUTION PHASE
    # niters_r = %s
    # ''', niters - niters_c)
    if return_all:
        v_is = np.array(v_is)
        v_js = np.array(v_js)
        omega_is = np.array(omega_is)
        omega_js = np.array(omega_js)
        for i in range(len(v_is)):
            dot(G.T, v_is[i], out=v_is[i])
            dot(G.T, v_js[i], out=v_js[i])
            dot(G.T, omega_is[i], out=omega_is[i])
            dot(G.T, omega_js[i], out=omega_js[i])
        return v_is, omega_is, v_js, omega_js
    return dot(G.T, v_i), dot(G.T, omega_i), dot(G.T, v_j), dot(G.T, omega_j)
