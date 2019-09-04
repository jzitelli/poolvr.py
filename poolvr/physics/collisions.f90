MODULE collisions
  USE iso_c_binding
  IMPLICIT NONE
  double precision, parameter :: INF = atan(-1.d0)
  double precision, parameter :: R = 0.02625
  double precision, parameter :: M = 0.1406
  double precision, parameter :: mu_s = 0.21
  double precision, parameter :: mu_b = 0.05
  double precision, parameter :: e = 0.89
CONTAINS

  SUBROUTINE collide_balls (deltaP, R, r_i, v_i, omega_i, r_j, v_j, omega_j, v_i1, omega_i1, v_j1, omega_j1) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none
    double precision, VALUE, intent(in) :: deltaP, R
    double precision, dimension(3), intent(in) :: r_i, r_j
    double precision, dimension(3), intent(in) :: v_i, v_j
    double precision, dimension(3), intent(in) :: omega_i, omega_j
    double precision, dimension(3), intent(out) :: v_i1, v_j1
    double precision, dimension(3), intent(out) :: omega_i1, omega_j1
    double precision, dimension(3), parameter :: z_loc = (/ 0.d0, 1.d0, 0.d0 /)
    double precision, dimension(3) :: r_c, r_ij
    double precision :: r_ij_mag_sqrd, r_ij_mag, v_ix, v_iy, v_jx, v_jy
    double precision :: omega_ix, omega_iy, omega_iz, omega_jx, omega_jy, omega_jz
    double precision :: u_iR_x, u_iR_y, u_jR_x, u_jR_y, u_iR_xy_mag, u_jR_xy_mag
    double precision :: u_ijC_x, u_ijC_z, u_ijC_xz_mag, v_ijy, v_ijy0
    double precision :: deltaP_ix, deltaP_iy, deltaP_jx, deltaP_jy, deltaP_1, deltaP_2
    double precision :: deltaV_ix, deltaV_iy, deltaV_jx, deltaV_jy
    double precision, dimension(3) :: deltaOm_i, deltaOm_j
    double precision, dimension(3) :: y_loc, x_loc
    double precision :: W = 0, deltaW
    double precision :: W_f = INF
    double precision :: W_c = INF
    double precision, dimension(3,3) :: G, G_T
    integer :: niters = 0
    r_ij = r_j - r_i
    r_ij_mag_sqrd = sum(r_ij**2)
    r_ij_mag = sqrt(r_ij_mag_sqrd)
    y_loc = r_ij / r_ij_mag
    x_loc(1) = -y_loc(3)
    x_loc(2) = 0
    x_loc(3) = y_loc(1)
    G(1,1:3) = x_loc
    G(2,1:3) = y_loc
    G(3,1:3) = z_loc
    G_T = transpose(G)
    v_ix = sum(v_i * x_loc)
    v_iy = sum(v_i * y_loc)
    v_jx = sum(v_j * x_loc)
    v_jy = sum(v_j * y_loc)
    omega_ix = sum(omega_i * x_loc)
    omega_iy = sum(omega_i * y_loc)
    omega_iz = sum(omega_i * z_loc)
    omega_jx = sum(omega_j * x_loc)
    omega_jy = sum(omega_j * y_loc)
    omega_jz = sum(omega_j * z_loc)
    u_iR_x = v_ix + R*omega_iy
    u_iR_y = v_iy - R*omega_ix
    u_jR_x = v_jx + R*omega_jy
    u_jR_y = v_jy - R*omega_jx
    u_iR_xy_mag = sqrt(u_iR_x**2 + u_iR_y**2)
    u_jR_xy_mag = sqrt(u_jR_x**2 + u_jR_y**2)
    u_ijC_x = v_ix - v_jx - R*(omega_iz + omega_jz)
    u_ijC_z =               R*(omega_ix + omega_jx)
    u_ijC_xz_mag = sqrt(u_ijC_x**2 + u_ijC_z**2)
    v_ijy = v_jy - v_iy
    v_i1(1) = v_ix
    v_i1(2) = v_iy
    v_i1(3) = 0
    v_j1(1) = v_jx
    v_j1(2) = v_jy
    v_j1(3) = 0
    omega_i1 = (/ omega_ix, omega_iy, omega_iz /)
    omega_j1 = (/ omega_jx, omega_jy, omega_jz /)
    do while (v_ijy < 0 .or. W < W_f)
       if (u_ijC_xz_mag < 1.d-16) then
          deltaP_1 = 0
          deltaP_2 = 0
          deltaP_ix = 0
          deltaP_iy = 0
          deltaP_jx = 0
          deltaP_jy = 0
       else
          deltaP_1 = -mu_b * deltaP * u_ijC_x / u_ijC_xz_mag
          if (abs(u_ijC_z) < 1.d-16) then
             deltaP_2 = 0
             deltaP_ix = 0
             deltaP_iy = 0
             deltaP_jx = 0
             deltaP_jy = 0
          else
             deltaP_2 = -mu_b * deltaP * u_ijC_z / u_ijC_xz_mag
             if (deltaP_2 > 0) then
                deltaP_ix = 0
                deltaP_iy = 0
                if (u_jR_xy_mag == 0) then
                   deltaP_jx = 0
                   deltaP_jy = 0
                else
                   deltaP_jx = -mu_s * (u_jR_x / u_jR_xy_mag) * deltaP_2
                   deltaP_jy = -mu_s * (u_jR_y / u_jR_xy_mag) * deltaP_2
                endif
             else
                deltaP_jx = 0
                deltaP_jy = 0
                if (u_iR_xy_mag == 0) then
                   deltaP_ix = 0
                   deltaP_iy = 0
                else
                   deltaP_ix = mu_s * (u_iR_x / u_iR_xy_mag) * deltaP_2
                   deltaP_iy = mu_s * (u_iR_y / u_iR_xy_mag) * deltaP_2
                end if
             end if
          end if
       endif
       deltaV_ix = ( deltaP_1 + deltaP_ix) / M
       deltaV_iy = (-deltaP   + deltaP_iy) / M
       deltaV_jx = (-deltaP_1 + deltaP_jx) / M
       deltaV_jy = ( deltaP   + deltaP_jy) / M
       !
       deltaOm_i = 5.d0/(2*M*R) * (/ (deltaP_2 + deltaP_iy), &
                                     (-deltaP_ix), &
                                     (-deltaP_1) /)
       deltaOm_j = 5.d0/(2*M*R) * (/ (deltaP_2 + deltaP_jy), &
                                     (-deltaP_jx), &
                                     (-deltaP_1) /)
       !
       v_ix = v_ix + deltaV_ix
       v_jx = v_jx + deltaV_jx
       v_iy = v_iy + deltaV_iy
       v_jy = v_jy + deltaV_jy
       !
       omega_i1 = omega_i1 + deltaOm_i
       omega_j1 = omega_j1 + deltaOm_j
       !
       u_iR_x = v_ix + R*omega_iy
       u_iR_y = v_iy - R*omega_ix
       u_jR_x = v_jx + R*omega_jy
       u_jR_y = v_jy - R*omega_jx
       u_iR_xy_mag = sqrt(u_iR_x**2 + u_iR_y**2)
       u_jR_xy_mag = sqrt(u_jR_x**2 + u_jR_y**2)
       !
       u_ijC_x = v_ix - v_jx - R*(omega_iz + omega_jz)
       u_ijC_z = R*(omega_ix + omega_jx)
       u_ijC_xz_mag = sqrt(u_ijC_x**2 + u_ijC_z**2)
       ! increment work:
       v_ijy0 = v_ijy
       v_ijy = v_jy - v_iy
       deltaW = 0.5 * deltaP * abs(v_ijy0 + v_ijy)
       W = W + deltaW
       niters = niters + 1
       if (W_c == INF .and. v_ijy > 0) then
          W_c = W
          W_f = (1 + e**2) * W_c
       end if
    end do
    v_i1 = (/ v_ix, v_iy, 0.d0 /)
    v_j1 = (/ v_jx, v_jy, 0.d0 /)
    v_i1 = MATMUL(G_T, v_i1)
    v_j1 = MATMUL(G_T, v_j1)
    omega_i1 = MATMUL(G_T, omega_i1)
    omega_j1 = MATMUL(G_T, omega_j1)
  END SUBROUTINE collide_balls
END MODULE COLLISIONS
