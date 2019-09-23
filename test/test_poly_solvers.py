from itertools import chain
import logging
_logger = logging.getLogger(__name__)
from math import fsum
import numpy as np
import pytest


from poolvr.physics.poly_solvers import quadratic_solve, sort_complex_conjugate_pairs


def find_conjugate_pairs(zs):
    roots = zs.ravel()
    npairs = sort_complex_conjugate_pairs(roots)
    pairs, rest = roots[:2*npairs], roots[2*npairs:]
    return pairs, rest


def test_quartic_solve_quadratic_factorable_a(poly_solver):
    ###############
    # x^4 + 9 = 0 #
    ###############
    p = np.array([9, 0, 0, 0, 1], dtype=np.float64)
    q0 = np.array([3, np.sqrt(6), 1], dtype=np.float64)
    q1 = q0.copy(); q1[1] *= -1
    zs = np.hstack((quadratic_solve(q0), quadratic_solve(q1)))
    _logger.debug(r'''solving  x^4 + 9 = 0
    which factors into quadratics
    (x^2 + 6^0.5*x + 3)
    (x^2 - 6^0.5*x + 3)
    expected roots:
    %s
    %s
    %s
    %s
    ''', *(str(x) for x in chain.from_iterable(find_conjugate_pairs(zs))))
    _logger.debug(r'''roots: (solver = %s):
    %s
    %s
    %s
    %s
    ''',
                  poly_solver.__name__,
                  *(str(x) for x in chain.from_iterable(find_conjugate_pairs(poly_solver(p)))))
    # assert((abs(xs-zs) < 1e-7).all())


def test_quartic_solve_quadratic_factorable_b(poly_solver):
    #################################
    # x^4 + 2x^3 - x^2 - 2x - 3 = 0 #
    #################################
    p = np.array([-3, -2, -1, 2, 1], dtype=np.float64)
    q0 = np.array([1, 1, 1], dtype=np.float64)
    q1 = np.array([-3, 1, 1], dtype=np.float64)
    zs = np.hstack((quadratic_solve(q0), quadratic_solve(q1)))
    _logger.debug(r'''solving  x^4 + 2x^3 - x^2 - 2x - 3 = 0
    which factors into quadratics
    (x^2 + x + 1)
    (x^2 + x - 3)
    expected roots:
    %s
    %s
    %s
    %s
    ''', *(str(x) for x in chain.from_iterable(find_conjugate_pairs(zs))))
    _logger.debug(r'''roots (solver = %s):
    %s
    %s
    %s
    %s
    ''',
                  poly_solver.__name__,
                  *(str(x) for x in chain.from_iterable(find_conjugate_pairs(poly_solver(p)))))
    # assert((abs(xpairs-zpairs) < 1e-7).all())
    # assert((abs(xs-zs) < 1e-7).all())


@pytest.mark.parametrize('roots', [np.random.rand(4) for _ in range(2)])
def test_quartic_solve_manufactored_quartic(poly_solver, roots):
    a, b, c, d = roots
    p = np.array([
        1.0,
        fsum((-a, -b, -c, -d)),
        a*fsum((b, c, d)) + c*fsum((b, d)) + b*d,
        a*fsum((-b*c, -b*d, -c*d)) - b*c*d,
        a*b*c*d
    ][::-1], dtype=np.float64)
    _logger.debug(r'''solving  x^4 + (%s)x^3 + (%s)x^2 + (%s)x + %s = 0
    expected roots:
    %s
    %s
    %s
    %s
    ''', p[3], p[2], p[1], p[0], *sorted(roots))
    _logger.debug(r'''roots: (solver = %s):
    %s
    %s
    %s
    %s
    ''',
                  poly_solver.__name__,
                  *(str(x) for x in sorted(chain.from_iterable(find_conjugate_pairs(poly_solver(p))),
                                           key=lambda z: z.real)))
