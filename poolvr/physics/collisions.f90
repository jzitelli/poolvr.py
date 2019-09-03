MODULE collisions
  USE iso_c_binding
  IMPLICIT NONE
CONTAINS
  SUBROUTINE collide_balls (v_i, omega_i, v_j, omega_j, v_i1, omega_i1, v_j1, omega_j1) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none
    double precision, dimension(3), intent(in) :: v_i, v_j
    double precision, dimension(3), intent(in) :: omega_i, omega_j
    double precision, dimension(3), intent(out) :: v_i1, v_j1
    double precision, dimension(3), intent(out) :: omega_i1, omega_j1
    v_i1(:) = 0.0
    v_j1(:) = 0.0
    omega_i1(:) = 0.0
    omega_j1(:) = 0.0
  END SUBROUTINE collide_balls
END MODULE COLLISIONS
