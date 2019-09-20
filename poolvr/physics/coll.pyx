import cython
import numpy as np
cimport numpy as np


cdef extern void collide_balls(double deltaP,
                               double[3] r_i, double[3] v_i, double[3] omega_i,
                               double[3] r_j, double[3] v_j, double[3] omega_j,
                               double[3] v_i1, double[3] omega_i1,
                               double[3] v_j1, double[3] omega_j1)


def collide_balls_f90(double deltaP,
                      double[:] r_i, double[:] v_i, double[:] omega_i,
                      double[:] r_j, double[:] v_j, double[:] omega_j):
    v_i1 = np.zeros(3, dtype=np.double)
    omega_i1 = np.zeros(3, dtype=np.double)
    v_j1 = np.zeros(3, dtype=np.double)
    omega_j1 = np.zeros(3, dtype=np.double)
    cdef double[3] v_i1_v #= v_i1
    cdef double[3] omega_i1_v #= omega_i1
    cdef double[3] v_j1_v #= v_j1
    cdef double[3] omega_j1_v #= omega_j1
    collide_balls(deltaP,
                  r_i, v_i, omega_i,
                  r_j, v_j, omega_j,
                  v_i1_v, omega_i1_v,
                  v_j1_v, omega_j1_v)
    return v_i1, omega_i1, v_j1, omega_j1


# from cpython cimport PyCapsule_GetPointer
# cimport scipy.linalg.cython_blas
# cimport scipy.linalg.cython_lapack
# import scipy.linalg as LA

# REAL = np.float64
# ctypedef np.float64_t REAL_t
# ctypedef np.uint64_t  INT_t

# cdef int ONE = 1
# cdef REAL_t ONEF = <REAL_t>1.0

# ctypedef void (*dger_ptr) (const int *M, const int *N, const double *alpha, const double *X, const int *incX, double *Y, const int *incY, double *A, const int * LDA) nogil
# cdef dger_ptr dger=<dger_ptr>PyCapsule_GetPointer(LA.blas.dger._cpointer, NULL)  # A := alpha*x*y.T + A

# #cpdef outer_prod(_x, _y): #comment above line & use this to use the reset output matrix to zeros
# cpdef outer_prod(_x, _y, _output):
#     cdef REAL_t *x = <REAL_t *>(np.PyArray_DATA(_x))
#     cdef int M = _y.shape[0]
#     cdef int N = _x.shape[0]
#     #cdef np.ndarray[np.float64_t, ndim=2, order='c'] _output = np.zeros((M,N)) #slow fix to uncomment to reset output matrix to zeros
#     cdef REAL_t *y = <REAL_t *>(np.PyArray_DATA(_y))
#     cdef REAL_t *output = <REAL_t *>(np.PyArray_DATA(_output))
#     with nogil:
#         dger(&M, &N, &ONEF, y, &ONE, x, &ONE, output, &M)
