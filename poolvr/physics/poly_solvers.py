import logging
_logger = logging.getLogger(__name__)
import numpy as np


PIx2 = np.pi*2
CUBE_ROOTS_OF_1_ANGLES = PIx2/3 * np.arange(3)
CUBE_ROOTS_OF_1 = np.exp(1j*CUBE_ROOTS_OF_1_ANGLES)

_ZERO_TOLERANCE = 1e-9
_ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
_IMAG_TOLERANCE = 1e-7
_IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2


def quartic_solve(p):
    if abs(p[-1]) / max(abs(p[:-1])) < _ZERO_TOLERANCE:
        _logger.debug('using cubic solver...')
        return cubic_solve(p[:-1])
    e, d, c, b, a = p
    Delta = 256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 144*a**2*c*d**2*e - 27*a**2*d**4 \
          + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 16*a*c**4*e \
          - 4*a*c**3*d**2 - 27*b**4*e**2 + 18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2
    P = 8*a*c - 3*b**2
    D = 64*a**3*e - 16*a**2*c**2 + 16*a*b**2*c - 16*a**2*b*d - 3*b**4
    # if Delta > 0 and (P > 0 or D > 0):
    #     _logger.debug('all roots are complex and distinct')
    #     return np.empty(0)
    R = (b**3 - 4*a*b*c + 8*a**2*d)
    Delta_0 = c**2 - 3*b*d + 12*a*e
    if Delta == 0 and D == 0:
        if P > 0 and R == 0:
            _logger.debug('two complex-conjugate double roots')
            return np.empty(0)
        elif Delta_0 == 0:
            _logger.debug('all roots are equal to -b / 4a')
            return np.array([-0.25 * b / a])
    Delta_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
    p = P / (8*a**2)
    q = R / (8*a**3)
    _logger.debug('P = %s, D = %s, R = %s\nDelta_0 = %s, Delta_1 = %s',
                  P, D, R, Delta_0, Delta_1)
    if Delta > 0:
        QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta + 0j)))
    else:
        QQQ = (0.5*(Delta_1 + np.sqrt(-27.0*Delta)))
    if Delta > _ZERO_TOLERANCE:
        _logger.debug('all roots are real and distinct')
        Q = QQQ**(1.0/3)
    elif Delta < -_ZERO_TOLERANCE:
        _logger.debug('two distinct real roots and a complex-conjugate pair of roots')
        angle = np.angle(QQQ) / 3
        if abs(angle) < 1e-8:
            Q_mag = abs(QQQ)**(1.0/3)
            Q = Q_mag * np.exp(1j*angle) * CUBE_ROOTS_OF_1[1]
        else:
            Q = QQQ**(1.0/3)
    else: # abs(Delta) <= _ZERO_TOLERANCE:
        _logger.debug('P = %s, D = %s, Delta_0 = %s, R = %s',
                      P, D, Delta_0, R)
        if P < 0 and D < 0 and Delta_0 != 0:
            _logger.debug('one real double root and two other real roots')
            Q = QQQ**(1.0/3)
        elif D > 0 or (P > 0 and (D != 0 or R != 0)):
            _logger.debug('one real double root and a complex-conjugate pair of roots')
            angle = np.angle(QQQ) / 3
            if abs(angle) < 1e-8:
                Q_mag = abs(QQQ)**(1.0/3)
                Q = Q_mag * np.exp(1j*angle) * CUBE_ROOTS_OF_1[1]
            else:
                Q = QQQ**(1.0/3)
        elif Delta_0 == 0 and D != 0:
            _logger.debug('one real triple root and one other real root')
            Q = QQQ**(1.0/3)
        elif D == 0 and P < 0:
            _logger.debug('two real double roots')
            Q = QQQ**(1.0/3)
    _logger.debug('Delta = %s', Delta)
    SSx4 = -2.0*p/3 + (Q + Delta_0/Q) / (3.0*a)
    S = 0.5*np.sqrt(SSx4 if SSx4 >= 0 else SSx4 + 0j)
    sqrp = -SSx4 - 2*p + q/S
    sqrm = -SSx4 - 2*p - q/S
    sqrtp = np.sqrt(sqrp if sqrp >= 0 else sqrp + 0j)
    sqrtm = np.sqrt(sqrm if sqrm >= 0 else sqrm + 0j)
    return np.array((
        -b/(4*a) - S + 0.5*sqrtp,
        -b/(4*a) - S - 0.5*sqrtp,
        -b/(4*a) + S + 0.5*sqrtm,
        -b/(4*a) + S - 0.5*sqrtm,
    ))


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
