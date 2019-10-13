MODULE poly_solvers
  USE iso_c_binding, only: c_double, c_double_complex, c_int
  IMPLICIT NONE

  real(c_double), parameter :: ZERO_TOLERANCE = 1.d-14
  real(c_double), parameter :: IMAG_TOLERANCE = 5.d-8
  real(c_double), parameter :: IMAG_TOLERANCE_SQRD = IMAG_TOLERANCE**2
  real(c_double), parameter :: PI = acos(-1.d0)
  real(c_double), parameter :: PIx2 = 2*PI
  complex(c_double_complex), dimension(3), parameter :: CUBE_ROOTS_OF_1 = (/ (1.d0, 0.d0), &
                                                                             exp(complex(0.d0, PIx2/3)), &
                                                                             exp(complex(0.d0, 2*PIx2/3)) /)

CONTAINS

  SUBROUTINE quartic_solve (Poly, out) BIND(C)
    implicit none
    real(c_double), dimension(5), intent(in) :: Poly
    double complex, dimension(4), intent(out) :: out
    real(c_double) :: e, d, c, b, bb, p, q, r, cc, dd, ee, ccc, Delta, Dee, Delta_0, Delta_1
    complex(c_double_complex) :: S, phi, zQ, SSx4, sqrtp, sqrtm
    complex(c_double_complex), dimension(3) :: S_v, SSx4_v
    integer(c_int) :: ir
    e = Poly(1) / Poly(5)
    d = Poly(2) / Poly(5)
    c = Poly(3) / Poly(5)
    b = Poly(4) / Poly(5)
    bb = b*b
    p = 0.125 * (8*c - 3*bb)
    q = 0.125 * (bb*b - 4*b*c + 8*d)
    r = (-3*bb*bb + 256*e - 64*b*d + 16*bb*c) / 256
    c = p
    d = q
    e = r
    cc = c*c
    dd = d*d
    ee = e*e
    ccc = cc*c
    Delta = -27*dd*dd - 128*cc*ee + 256*ee*e + 16*cc*cc*e + 144*c*dd*e - 4*ccc*dd
    Dee = 64*e - 16*cc
    Delta_0 = cc + 12*e
    Delta_1 = 2*ccc + 27*dd - 72*c*e
    if (Delta > 0 .and. p < 0 .and. Dee < 0) then
       phi = acos(Delta_1 / (2*sqrt(complex(Delta_0**3,0.d0))))
       S = 0.5 * sqrt((-2*p + 2*sqrt(complex(Delta_0,0.d0))*cos(phi/3))/3)
       SSx4 = 4*S*S
    else
       zQ = (0.5*(Delta_1 + sqrt(complex(-27*Delta,0.d0))))**(1.d0/3)
       SSx4_v = (-2*p + (zQ*CUBE_ROOTS_OF_1 + Delta_0/(zQ*CUBE_ROOTS_OF_1))) / 3.d0
       S_v = 0.5*sqrt(SSx4_v)
       if (Delta .ne. 0) then
          ir = maxloc(abs(S_v), 1)
          SSx4 = SSx4_v(ir)
          S = S_v(ir)
       else
          SSx4 = SSx4_v(1)
          S = S_v(1)
       endif
    endif
    sqrtp = sqrt(-SSx4 - 2*p + q/S)
    sqrtm = sqrt(-SSx4 - 2*p - q/S)
    out(1) = -0.25*b - S + 0.5*sqrtp;
    out(2) = -0.25*b - S - 0.5*sqrtp;
    out(3) = -0.25*b + S + 0.5*sqrtm;
    out(4) = -0.25*b + S - 0.5*sqrtm;
  END SUBROUTINE quartic_solve

  integer(c_int) function sort_complex_conjugate_pairs(roots) BIND(C)
    implicit none
    complex(c_double_complex), dimension(4), intent(inout) :: roots
    integer(c_int) :: npairs, i, j
    complex(c_double_complex) :: r, r_conj
    npairs = 0
    i = 1
    do while (i <= 3)
       r = roots(i)
       if (abs(DIMAG(r)) > IMAG_TOLERANCE) then
          do j = 1, 4-i
             r_conj = roots(i+j)
             if (       abs(DBLE(r) - DBLE(r_conj)) < ZERO_TOLERANCE &
                  .and. abs(DIMAG(r) + DIMAG(r_conj)) < ZERO_TOLERANCE) then
                roots(i) = roots(2*npairs+1)
                roots(i+j) = roots(2*npairs+2)
                roots(2*npairs+1) = r
                roots(2*npairs+2) = r_conj
                npairs = npairs + 1
                i = i + 1
                exit
             endif
          enddo
       endif
       i = i + 1
    enddo
    sort_complex_conjugate_pairs = npairs
  end function sort_complex_conjugate_pairs

  real(c_double) function find_min_quartic_root_in_real_interval(P, t0, t1) BIND(C)
    implicit none
    real(c_double), dimension(5), intent(in) :: P
    real(c_double), value, intent(in) :: t0, t1
    complex(c_double_complex), dimension(4) :: roots
    integer(c_int) :: npairs, i
    real(c_double) :: min_root
    complex(c_double_complex) :: r
    call quartic_solve(P, roots)
    min_root = huge(1.d0)
    npairs = sort_complex_conjugate_pairs(roots)
    do i = 2*npairs+1, 4
       r = roots(i)
       if (t0 < DBLE(r) .and. DBLE(r) < t1 &
            .and. DBLE(r) < min_root &
            .and. DIMAG(r)**2 / (DBLE(r)**2 + DIMAG(r)**2) < IMAG_TOLERANCE_SQRD) then
          min_root = DBLE(r)
       endif
    enddo
    find_min_quartic_root_in_real_interval = min_root
  end function find_min_quartic_root_in_real_interval

  real(c_double) function find_collision_time(a_i, a_j, R, t0, t1) BIND(C)
    implicit none
    real(c_double), dimension(3,3), intent(in) :: a_i, a_j
    real(c_double), value, intent(in) :: R, t0, t1
    real(c_double), dimension(3,3) :: a_ji
    real(c_double), dimension(0:4) :: P
    real(c_double), dimension(3) :: r_i, r_j, v_i, v_j, omega_i, omega_j
    real(c_double) :: t, v_ijy
    a_ji = a_i - a_j
    p(4) = dot_product(a_ji(:,3), a_ji(:,3))
    p(3) = 2 * dot_product(a_ji(:,3), a_ji(:,2))
    p(2) = dot_product(a_ji(:,2), a_ji(:,2)) + 2 * dot_product(a_ji(:,3), a_ji(:,1))
    p(1) = 2 * dot_product(a_ji(:,2), a_ji(:,1))
    p(0) = dot_product(a_ji(:,1), a_ji(:,1)) - 4*R*R
    ! PRINT *, "P =", P
    t = find_min_quartic_root_in_real_interval(P, t0, t1)
    if (t .ne. huge(1.d0)) then
       r_i = a_i(:,1) + t*a_i(:,2) + t**2*a_i(:,3)
       r_j = a_j(:,1) + t*a_j(:,2) + t**2*a_j(:,3)
       v_i = a_i(:,2) + 2*t*a_i(:,3)
       v_j = a_j(:,2) + 2*t*a_j(:,3)
       v_ijy = dot_product(v_j-v_i, r_j-r_i) / (2*R)
       if (v_ijy > 0) then
          ! PRINT *, "t =", t, " v_ijy =", v_ijy
          find_collision_time = huge(1.d0)
          return
       endif
    endif
    find_collision_time = t
  end function find_collision_time

END MODULE poly_solvers
