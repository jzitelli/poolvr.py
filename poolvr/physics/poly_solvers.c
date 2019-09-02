#include <math.h>
#include <complex.h>


#define PIx2 (M_PI * 2.0)
#define ZERO_TOLERANCE 1e-9
#define IMAG_TOLERANCE 1e-8
#define IMAG_TOLERANCE_SQRD (IMAG_TOLERANCE * IMAG_TOLERANCE)


double complex CUBE_ROOTS_OF_1[] = {1.0, cexp(I*PIx2/3), cexp(I*2*PIx2/3)};


void quartic_solve(double P[5], double complex out[4]) {
  double e = P[0] / P[4];
  double d = P[1] / P[4];
  double c = P[2] / P[4];
  double b = P[3] / P[4];
  double bb = b*b;
  double p = 0.125 * (8*c - 3*bb);
  double q = 0.125 * (bb*b - 4*b*c + 8*d);
  double r = (-3*bb*bb + 256*e - 64*b*d + 16*bb*c) / 256;
  c = p;
  d = q;
  e = r;
  double cc = c*c;
  double dd = d*d;
  double ee = e*e;
  double ccc = cc*c;
  double Delta = -27*dd*dd - 128*cc*ee + 256*ee*e + 16*cc*cc*e + 144*c*dd*e - 4*ccc*dd;
  double D = 64*e - 16*cc;
  double Delta_0 = cc + 12*e;
  double Delta_1 = 2*ccc + 27*dd - 72*c*e;
  double complex S;
  double complex SSx4;
  if (Delta_1 > 0 && p < 0 && D < 0) {
    double complex phi = cacos(Delta_1 / (2*csqrt(Delta_0*Delta_0*Delta_0)));
    S = 0.5 * csqrt((-2*p + 2*csqrt(Delta_0)*ccos(phi/3))/3);
    SSx4 = 4*S*S;
  } else {
    double complex Q = cpow(0.5*(Delta_1 + csqrt(-27*Delta)), 1.0/3);
    if (Delta != 0) {
      double abs_SSx4_max = 0;
      double abs_SSx4;
      double complex SSx4_max;
      for (int ir = 0; ir < 3; ir++) {
	SSx4 = (-2.0*p + (Q*CUBE_ROOTS_OF_1[ir] + Delta_0/(Q*CUBE_ROOTS_OF_1[ir]))) / 3;
	abs_SSx4 = cabs(SSx4);
	if (abs_SSx4 > abs_SSx4_max) {
	  abs_SSx4_max = abs_SSx4;
	  SSx4_max = SSx4;
	}
      }
      SSx4 = SSx4_max;
    } else {
      SSx4 = (-2.0*p + (Q + Delta_0/Q)) / 3;
    }
    S = 0.5*csqrt(SSx4);
  }
  double complex sqrtp = csqrt(-SSx4 - 2*p + q/S);
  double complex sqrtm = csqrt(-SSx4 - 2*p - q/S);
  out[0] = -0.25*b - S + 0.5*sqrtp;
  out[1] = -0.25*b - S - 0.5*sqrtp;
  out[2] = -0.25*b + S + 0.5*sqrtm;
  out[3] = -0.25*b + S - 0.5*sqrtm;
}


int sort_complex_conjugate_pairs(double complex roots[4]) {
  int npairs = 0;
  double complex r;
  double complex r_conj;
  for (int i = 0; i < 3; i++) {
    r = roots[i];
    if (fabs(cimag(r)) > IMAG_TOLERANCE) {
      for (int j = 1; j + i < 4; j++) {
	r_conj = roots[i+j];
	if (fabs(creal(r) - creal(r_conj)) < ZERO_TOLERANCE
	    && fabs(cimag(r) + cimag(r_conj)) < ZERO_TOLERANCE) {
	  roots[i] = roots[2*npairs];
	  roots[i+j] = roots[2*npairs+1];
	  roots[2*npairs] = r;
	  roots[2*npairs+1] = r_conj;
	  npairs++;
	  i++;
	  break;
	}
      }
    }
  }
  return npairs;
}


double find_min_quartic_root_in_real_interval(double P[5], double t0, double t1) {
  double complex roots[4];
  quartic_solve(P, roots);
  int npairs = sort_complex_conjugate_pairs(roots);
  double min_root = NAN;
  for (int i = 2*npairs; i < 4; i++) {
    double complex r = roots[i];
    if (t0 < creal(r) && creal(r) < t1 &&
	cimag(r)*cimag(r) / (creal(r)*creal(r) + cimag(r)*cimag(r))
	< IMAG_TOLERANCE_SQRD) {
      min_root = t1 = creal(r);
    }
  }
  return min_root;
}


double find_collision_time(double a_i[3][3], double a_j[3][3],
			   double R, double t0, double t1) {
  double a_ji[3][3] = {{a_i[0][0] - a_j[0][0],
			a_i[0][1] - a_j[0][1],
			a_i[0][2] - a_j[0][2]},
		       {a_i[1][0] - a_j[1][0],
			a_i[1][1] - a_j[1][1],
			a_i[1][2] - a_j[1][2]},
		       {a_i[2][0] - a_j[2][0],
			a_i[2][1] - a_j[2][1],
			a_i[2][2] - a_j[2][2]}};
  double a_x = a_ji[2][0];
  double a_y = a_ji[2][2];
  double b_x = a_ji[1][0];
  double b_y = a_ji[1][2];
  double c_x = a_ji[0][0];
  double c_y = a_ji[0][2];
  double p[5];
  p[4] = a_x*a_x + a_y*a_y;
  p[3] = 2 * (a_x*b_x + a_y*b_y);
  p[2] = b_x*b_x + b_y*b_y + 2 * (a_x*c_x + a_y*c_y);
  p[1] = 2 * (b_x*c_x + b_y*c_y);
  p[0] = c_x*c_x + c_y*c_y - 4 * R*R;
  return find_min_quartic_root_in_real_interval(p, t0, t1);
}
