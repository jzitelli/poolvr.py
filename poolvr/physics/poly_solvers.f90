MODULE poly_solvers
  USE iso_c_binding, only: c_double, c_double_complex, c_int
  IMPLICIT NONE

  real(c_double), parameter :: ZERO_TOLERANCE = 1.d-15
  real(c_double), parameter :: IMAG_TOLERANCE = 1.d-14
  real(c_double), parameter :: IMAG_TOLERANCE_SQRD = IMAG_TOLERANCE**2
  real(c_double), parameter :: PI = acos(-1.d0)
  real(c_double), parameter :: PIx2 = 2*PI
  complex(c_double_complex), dimension(3), parameter :: CUBE_ROOTS_OF_1 = (/ cmplx(1.d0,0.d0, kind=c_double),        &
                                                                             exp(cmplx(0.d0,PIx2/3, kind=c_double)), &
                                                                             exp(cmplx(0.d0,2*PIx2/3, kind=c_double)) /)

CONTAINS

  SUBROUTINE quartic_solve (Poly, out) BIND(C)
    implicit none
    real(c_double), dimension(5), intent(in) :: Poly
    complex(c_double_complex), dimension(4), intent(out) :: out
    real(c_double) :: e, d, c, b, bb, p, q, r, cc, dd, ee, ccc, Delta, Dee, Delta_0, Delta_1
    complex(c_double_complex) :: S, SSx4, phi, zQ, SSx4_max, sqrtp, sqrtm
    real(c_double) :: abs_SSx4_max, abs_SSx4
    integer(c_int) :: ir
    e = Poly(1) / Poly(5)
    d = Poly(2) / Poly(5)
    c = Poly(3) / Poly(5)
    b = Poly(4) / Poly(5)
    bb = b*b
    p = 0.125d0 * (8*c - 3*bb)
    q = 0.125d0 * (bb*b - 4*b*c + 8*d)
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
    if (Delta_1 > 0 .and. p < 0 .and. Dee < 0) then
       phi = acos(Delta_1 / (2*sqrt(Delta_0**3)))
       S = 0.5 * sqrt((-2*p + 2*sqrt(Delta_0)*cos(phi/3))/3)
       SSx4 = 4*S*S
    else
       zQ = (0.5*(Delta_1 + sqrt(-27*Delta)))**(1.d0/3)
       if (Delta .ne. 0) then
          abs_SSx4_max = 0.d0
          do ir = 1, 3
             SSx4 = (-2*p + (zQ*CUBE_ROOTS_OF_1(ir) + Delta_0/(zQ*CUBE_ROOTS_OF_1(ir)))) / 3
             abs_SSx4 = abs(SSx4)
             if (abs_SSx4 > abs_SSx4_max) then
                abs_SSx4_max = abs_SSx4
                SSx4_max = SSx4
             endif
          enddo
          SSx4 = SSx4_max
       else
          SSx4 = (-2*p + (zQ + Delta_0/zQ)) / 3
       endif
       S = 0.5*sqrt(SSx4)
    endif
    sqrtp = sqrt(-SSx4 - 2*p + q/S)
    sqrtm = sqrt(-SSx4 - 2*p - q/S)
    out(1) = -0.25*b - S + 0.5*sqrtp;
    out(2) = -0.25*b - S - 0.5*sqrtp;
    out(3) = -0.25*b + S + 0.5*sqrtm;
    out(4) = -0.25*b + S - 0.5*sqrtm;
  END SUBROUTINE quartic_solve

  function sort_complex_conjugate_pairs(roots) BIND(C)
    implicit none
    integer(c_int) :: sort_complex_conjugate_pairs
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
             if (       abs(DREAL(r) - DREAL(r_conj)) < ZERO_TOLERANCE &
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

  function find_min_quartic_root_in_real_interval(P, t0, t1) BIND(C)
    implicit none
    real(c_double) :: find_min_quartic_root_in_real_interval
    real(c_double), dimension(5), intent(in) :: P
    real(c_double), value :: t0, t1
    complex(c_double_complex), dimension(4) :: roots
    integer(c_int) :: npairs, i
    real(c_double) :: min_root
    complex(c_double_complex) :: r
    call quartic_solve(P, roots)
    min_root = huge(1.d0)
    npairs = sort_complex_conjugate_pairs(roots)
    do i = 2*npairs+1, 4
       r = roots(i)
       if (t0 < DREAL(r) .and. DREAL(r) < t1 &
            .and. DIMAG(r)**2 / (DREAL(r)**2 + DIMAG(r)**2) < IMAG_TOLERANCE_SQRD) then
          min_root = DREAL(r)
          t1 = min_root
       endif
    enddo
    find_min_quartic_root_in_real_interval = min_root
  end function find_min_quartic_root_in_real_interval

  function find_collision_time(a_i, a_j, R, t0, t1) BIND(C)
    implicit none
    real(c_double) :: find_collision_time
    real(c_double), dimension(3,3), intent(in) :: a_i, a_j
    real(c_double), value, intent(in) :: R, t0, t1
    real(c_double), dimension(3,3) :: a_ji
    real(c_double), dimension(5) :: P
    real(c_double) :: a_x, a_y, b_x, b_y, c_x, c_y
    a_ji = a_i - a_j
    a_x = a_ji(1,3)
    a_y = a_ji(3,3)
    b_x = a_ji(1,2)
    b_y = a_ji(3,2)
    c_x = a_ji(1,1)
    c_y = a_ji(3,1)
    p(5) = a_x*a_x + a_y*a_y
    p(4) = 2 * (a_x*b_x + a_y*b_y)
    p(3) = b_x*b_x + b_y*b_y + 2 * (a_x*c_x + a_y*c_y)
    p(2) = 2 * (b_x*c_x + b_y*c_y)
    p(1) = c_x*c_x + c_y*c_y - 4 * R*R
    find_collision_time = find_min_quartic_root_in_real_interval(P, t0, t1)
  end function find_collision_time

END MODULE poly_solvers
