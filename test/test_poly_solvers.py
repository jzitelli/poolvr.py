import logging
_logger = logging.getLogger(__name__)
from math import fsum
import numpy as np


def find_conjugate_pairs(zs):
    mask = np.array(len(zs)*[True])
    pairs = []
    for i, z in enumerate(zs):
        if z.imag == 0:
            continue
        j, z_conj = next(((j, z_conj) for j, z_conj in enumerate(zs[i+1:])
                          if z.imag + z_conj.imag < 1e-10
                          and z.real - z_conj.real < 1e-10),
                         (None, None))
        if z_conj is not None:
            pairs.append((z, z_conj))
            mask[i] = mask[i+j+1] = False
            zs = zs[mask]
    return np.array(pairs), zs


def test_quartic_solve():
    from poolvr.physics.poly_solvers import quartic_solve, quadratic_solve

    # x^4 + 9 = 0
    p = np.array([9, 0, 0, 0, 1], dtype=np.float64)
    q0 = np.array([3, np.sqrt(6), 1], dtype=np.float64)
    q1 = q0.copy(); q1[1] *= -1
    zs = np.hstack((quadratic_solve(q0), quadratic_solve(q1)))
    _logger.debug('''solving x^4 + 9 = 0
which factors into quadratics
  (x^2 + 6^0.5*x + 3): %s
  (x^2 - 6^0.5*x + 3): %s''', ',  '.join(str(x) for x in zs[:2]), ',  '.join(str(x) for x in zs[2:]))
    xs = quartic_solve(p)
    _logger.debug('roots:\n%s', '  '.join(str(x) for x in xs))
    xs.sort()
    zs.sort()
    assert((abs(xs-zs) < 1e-7).all())

    # x^4 + 2x^3 - x^2 - 2x - 3 = 0
    p = np.array([-3, -2, -1, 2, 1], dtype=np.float64)
    q0 = np.array([1, 1, 1], dtype=np.float64)
    q1 = np.array([-3, 1, 1], dtype=np.float64)
    zs = np.hstack((quadratic_solve(q0), quadratic_solve(q1)))
    _logger.debug('''solving  x^4 + 2x^3 - x^2 - 2x - 3 = 0
which factors into quadratics
  (x^2 + x + 1): %s
  (x^2 + x - 3): %s''', ',  '.join(str(x) for x in zs[:2]), ',  '.join(str(x) for x in zs[2:]))
    xs = quartic_solve(p)
    _logger.debug('roots:\n%s', '  '.join(str(x) for x in xs))
    xpairs, xs = find_conjugate_pairs(xs)
    zpairs, zs = find_conjugate_pairs(zs)
    xs.sort()
    zs.sort()
    assert((abs(xpairs-zpairs) < 1e-7).all())
    assert((abs(xs-zs) < 1e-7).all())

    a, b, c, d = xs = np.array((0.5761249714334903, 0.7040881051124456, 0.8687574666374065, 0.868724005945493), dtype=np.float64)
    p = np.array([
        1.0,
        fsum((-a, -b, -c, -d)),
        a*fsum((b, c, d)) + c*fsum((b, d)) + b*d,
        a*fsum((-b*c, -b*d, -c*d)) - b*c*d,
        a*b*c*d
    ][::-1], dtype=np.float64)
    _logger.debug('''solving
  x^4 + (%s)x^3 + (%s)x^2 + (%s)x + %s  =  0
which factors into
  (x - %s)
  (x - %s)
  (x - %s)
  (x - %s)''', p[3], p[2], p[1], p[0], a, b, c, d)
    Delta = np.prod((xs[1:] - a)**2) * (xs[0] - b)**2 * np.prod((xs[2:] - b)**2) * np.prod((xs[:2] - c)**2) * (xs[3] - c)**2 * np.prod((xs[:-1] - d)**2)
    _logger.debug('Delta = %s', Delta)
    zs = quartic_solve(p)
    xs.sort()
    zs.sort()
    _logger.debug('roots:\n%s', '  '.join(str(x) for x in zs))
    assert((abs(xs-zs)/abs(xs) < 1e-7).all())

    for i_poly in range(1000):
        a, b, c, d = xs = np.random.rand(4)
        p = np.array([
            1,
            (-a - b - c - d),
            (a*b + a*c + a*d + b*c + b*d + c*d),
            (-a*b*c - a*b*d - a*c*d - b*c*d),
            a*b*c*d
        ][::-1])
        _logger.debug('''solving random quartic #%d:
  x^4 + (%s)x^3 + (%s)x^2 + (%s)x + %s  =  0
which factors into
  (x - %s)
  (x - %s)
  (x - %s)
  (x - %s)''', i_poly+1, p[3], p[2], p[1], p[0], a, b, c, d)
        zs = quartic_solve(p)
        xs.sort()
        zs.sort()
        _logger.debug('roots:\n%s', '  '.join(str(x) for x in zs))
        assert((abs(xs-zs)/abs(xs) < 1e-7).all())
