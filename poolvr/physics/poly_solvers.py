import logging
_logger = logging.getLogger(__name__)
from math import fsum
import numpy as np


PIx2 = np.pi*2
CUBE_ROOTS_OF_1_ANGLES = PIx2/3 * np.arange(3)
CUBE_ROOTS_OF_1 = np.exp(1j*CUBE_ROOTS_OF_1_ANGLES)

_ZERO_TOLERANCE = 1e-15
_ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
_IMAG_TOLERANCE = 1e-8
_IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2


def quartic_solve(p, only_real=False):
    if abs(p[-1]) / max(abs(p[:-1])) < _ZERO_TOLERANCE:
        # _logger.debug('using cubic solver...')
        return cubic_solve(p[:-1])
    e, d, c, b = p[:-1] / p[-1]
    p = 0.125 * (8*c - 3*b**2)
    q = 0.125 * (b**3 - 4*b*c + 8*d)
    r = (-3*b**4 + 256*e - 64*b*d + 16*b**2*c) / 256
    c, d, e = p, q, r
    cc, dd, ee = c*c, d*d, e*e
    ccc, eee = cc*c, ee*e
    cccc, dddd = cc*cc, dd*dd
    Delta = fsum((-27*dddd, -128*cc*ee, 256*eee, 16*cccc*e, 144*c*dd*e, -4*ccc*dd))
    # _logger.debug('Delta = %s', Delta)
    D = 64*e - 16*cc
    if only_real and Delta > 0 and (p > 0 or D > 0):
        # _logger.debug('all roots are complex and distinct')
        return np.empty(0)
    # R = 8*d
    Delta_0 = cc + 12*e
    if Delta == 0 and D == 0:
        if only_real and p > 0 and r == 0:
            # _logger.debug('two complex-conjugate double roots')
            return np.empty(0)
        elif Delta_0 == 0:
            # _logger.debug('all roots are equal to -b / 4a')
            return np.array([-0.25*b])
    Delta_1 = 2*ccc + 27*dd - 72*c*e
    if Delta > 0 and p < 0 and D < 0:
        # _logger.debug('all roots are real and distinct')
        phi = np.arccos(Delta_1 / (2*np.sqrt(Delta_0**3)))
        S = 0.5 * (np.sqrt(-2*p/3 + 2*np.sqrt(Delta_0)*np.cos(phi/3)/3))
        SSx4 = 4*S**2
    else:
        QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta if Delta <= 0 else -27.0*Delta + 0j)))
        Q = QQQ**(1.0/3)
        SSx4 = -2.0*p/3 + (Q*CUBE_ROOTS_OF_1 + Delta_0/(Q*CUBE_ROOTS_OF_1)) / 3.0
        if Delta > 0:
            # _logger.debug('all roots are complex and distinct')
            S = 0.5*np.sqrt(SSx4 + 0j)
            argsort = np.argsort(abs(S))
            S = S[argsort[-1]]
            SSx4 = SSx4[argsort[-1]]
        elif Delta < 0:
            # _logger.debug('two distinct real roots and a complex-conjugate pair of roots')
            S = 0.5*np.sqrt(SSx4 + 0j)
            argsort = np.argsort(abs(S))
            S = S[argsort[-1]]
            SSx4 = SSx4[argsort[-1]]
        else:
            S = 0.5*np.sqrt(SSx4[0] + 0j)
            # if D > 0 or (p > 0 and (D != 0 or r != 0)):
            #     _logger.debug('one real double root and a complex-conjugate pair of roots')
            # elif p < 0 and D < 0 and Delta_0 != 0:
            #     _logger.debug('one real double root and two other real roots')
            # elif Delta_0 == 0 and D != 0:
            #     _logger.debug('one real triple root and one other real root')
            # elif D == 0 and p < 0:
            #     _logger.debug('two real double roots')
    sqrp = -SSx4 - 2*p + q/S
    sqrm = -SSx4 - 2*p - q/S
    sqrtp = np.sqrt(sqrp if sqrp >= 0 else sqrp + 0j)
    sqrtm = np.sqrt(sqrm if sqrm >= 0 else sqrm + 0j)
    return np.array((
        -0.25*b - S + 0.5*sqrtp,
        -0.25*b - S - 0.5*sqrtp,
        -0.25*b + S + 0.5*sqrtm,
        -0.25*b + S - 0.5*sqrtm,
    ))


def printit(zs):
    return ',  '.join(str(z) for z in zs)


def cubic_solve(p):
    if abs(p[-1]) / max(abs(p[:-1])) < _ZERO_TOLERANCE:
        return quadratic_solve(p[:-1])
    a, b, c, d = p[::-1]
    Delta = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
    Delta_0 = b**2 - 3*a*c
    if Delta == 0:
        if Delta_0 == 0:
            return np.array((-b/(3*a)))
        else:
            return np.array((              (9*a*d - b*c) / (2*Delta_0),
                             (4*a*b*c - 9*a**2*d - b**3) / (a*Delta_0)))
    Delta_1 = 2*b**3 - 9*a*b*c + 27*a**2*d
    DD = -27*a**2*Delta
    CCC = 0.5 * (Delta_1 + np.sign(Delta_1)*np.sqrt(DD if DD >= 0 else DD + 0j))
    if CCC == 0:
        return np.array((-b/(3*a)))
    C = CCC**(1.0/3)
    return -(b + CUBE_ROOTS_OF_1*C + Delta_0 / (CUBE_ROOTS_OF_1*C)) / (3*a)


def quadratic_solve(p):
    if abs(p[2]) / max(abs(p[:2])) < _ZERO_TOLERANCE:
        return np.array([-p[0] / p[1]])
    a, b, c = p[::-1]
    d = b**2 - 4*a*c
    sqrtd = np.sqrt(d if d >= 0 else d + 0j)
    return np.array(((-b + sqrtd) / (2*a),
                     (-b - sqrtd) / (2*a)))
