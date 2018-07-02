import logging
import numpy as np


_logger = logging.getLogger(__name__)
INCH2METER = 0.0254
_J = np.array((0, 1, 0), dtype=np.float64)


from ..decorators import allocs_out, allocs_out_vec4


class BallStates(object):
    STATIONARY = 0
    SLIDING    = 1
    ROLLING    = 2
    SPINNING   = 3
    POCKETED   = 4
    FREEFALL   = 5


class PhysicsEvent(object):
    I_VEC, J_VEC, K_VEC = np.eye(3, dtype=np.float64)
    physics = None
    ball_mass = 0.17
    ball_radius = 1.125*INCH2METER
    mu_r = 0.016
    mu_sp = 0.044
    mu_s = 0.2
    mu_b = 0.06
    c_b = 4000.0
    E_Y_b = 2.4e9
    g = 9.81
    _k = np.array((0,1,0), dtype=np.float64)
    def __init__(self, t, **kwargs):
        self.t = t
        self.T = 0.0
        self.child_events = ()
        self.next_motion_event = None

    @staticmethod
    def set_quaternion_from_euler_angles(psi=0.0, theta=0.0, phi=0.0, out=None):
        if out is None:
            out = np.empty(4, dtype=np.float64)
        angles = np.array((psi, theta, phi))
        c1, c2, c3 = np.cos(0.5 * angles)
        s1, s2, s3 = np.sin(0.5 * angles)
        out[0] = s1*c2*c3 + c1*s2*s3
        out[1] = c1*s2*c3 - s1*c2*s3
        out[2] = c1*c2*s3 + s1*s2*c3
        out[3] = c1*c2*c3 - s1*s2*s3
        return out

    def __lt__(self, other):
        return self.t < other.t

    def __gt__(self, other):
        return self.t > other.t


class BallEvent(PhysicsEvent):
    ball_radius = 1.125 * INCH2METER
    ball_mass = 0.17
    ball_I = 2/5 * ball_mass * ball_radius**2
    def __init__(self, t, i):
        super().__init__(t)
        self.i = i

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T and self.i == other.i

    def __hash__(self):
        return hash((self.__class__.__name__, self.i, self.t, self.T))

    def __str__(self):
        return "<%s t=%f T=%f i=%d>" % (self.__class__.__name__.split('.')[-1], self.t, self.T, self.i)


class BallRestEvent(BallEvent):
    def __init__(self, t, i, r=None, q=None,
                 psi=0.0, theta=0.0, phi=0.0):
        super().__init__(t, i)
        if r is None:
            self._r = self._r_0 = np.zeros(3, dtype=np.float64)
        else:
            self._r = self._r_0 = r.copy()
        self._psi, self._theta, self._phi = psi, theta, phi
        if q is None:
            self._q = self._q_0 = self.set_quaternion_from_euler_angles(psi=psi, theta=theta, phi=phi)
        else:
            self._q = self._q_0 = q.copy()

    @allocs_out
    def eval_position(self, tau, out=None):
        out[:] = self._r
        return out

    @allocs_out
    def eval_velocity(self, tau, out=None):
        out[:] = 0
        return out

    @allocs_out_vec4
    def eval_quaternion(self, tau, out=None):
        out[:] = self._q
        return out

    @allocs_out
    def eval_angular_velocity(self, tau, out=None):
        out[:] = 0
        return out


class BallMotionEvent(BallEvent):
    def __init__(self, t, i, a=None, b=None,
                 r_0=None, v_0=None, a_0=None,
                 psi_0=0.0, theta_0=0.0, phi_0=0.0,
                 omega_0=None):
        super().__init__(t, i)
        if a is None:
            a = np.zeros((3,3), dtype=np.float64)
        if b is None:
            b = np.zeros((2,3), dtype=np.float64)
        self._a = a
        self._b = b
        self._psi_0, self._theta_0, self._phi_0 = psi_0, theta_0, phi_0
        self._q_0 = self.set_quaternion_from_euler_angles(psi=psi_0, theta=theta_0, phi=phi_0)
        if r_0 is not None:
            a[0] = r_0
        if v_0 is not None:
            a[1] = v_0
        if a_0 is not None:
            a[2] = 0.5 * a_0
        if omega_0 is None:
            b[0] = 0
        else:
            b[0] = omega_0
        self._a_global, self._b_global = self.calc_global_coeffs(t, a, b)
        self._r_0 = a[0]
        self._v_0 = a[1]
        self._half_a = a[2]
        self._omega_0 = b[0]

    @staticmethod
    def calc_global_coeffs(t, a, b):
        "Calculates the global-time coefficients of the equations of motion."
        a_global, b_global = a.copy(), b.copy()
        a_global[0] += -t * a[1] + t**2 * a[2]
        a_global[1] += -2 * t * a[2]
        b_global[0] += -t * b[1]
        return a_global, b_global

    @allocs_out
    def eval_position(self, tau, out=None):
        a = self._a
        out[:] = a[0] + tau * a[1] + tau**2 * a[2]
        return out

    @allocs_out
    def eval_velocity(self, tau, out=None):
        _a = self._a
        out[:] = _a[1] + 2 * tau * _a[2]
        return out

    @allocs_out_vec4
    def eval_quaternion(self, tau, out=None):
        out[:-1] = 0
        out[-1] = 1
        return out

    @allocs_out
    def eval_angular_velocity(self, tau, out=None):
        b = self._b
        out[:] = b[0] + tau * b[1]
        return out


class BallSlidingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0,
                 **kwargs):
        super().__init__(t, i, r_0=r_0, v_0=v_0, omega_0=omega_0)
        u_0 = v_0 + self.ball_radius * np.cross(self._k, omega_0)
        u_0_mag = np.linalg.norm(u_0)
        self._a[2] = -0.5 * self.mu_s * self.g * u_0 / u_0_mag
        self.T = T = 2 * u_0_mag / (7 * self.mu_s * self.g)
        self.next_motion_event = BallRollingEvent(t + self.T, i,
                                                  r_0=self.eval_position(T),
                                                  v_0=self.eval_velocity(T))


class BallRollingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0):
        v_0_mag = np.linalg.norm(v_0)
        omega_0 = v_0 / self.ball_radius; omega_0[::2] = -omega_0[::-2]
        super().__init__(t, i, r_0=r_0, v_0=v_0, omega_0=omega_0)
        self._a[2] = -0.5 * self.mu_r * self.g * v_0 / v_0_mag
        self._b[1]
        self.T = v_0_mag / (self.mu_r * self.g)
        self.next_motion_event = BallRestEvent(t + self.T, i, r=self.eval_position(self.T))


class CueStrikeEvent(BallEvent):
    def __init__(self, t, i, r_i, r_c, V, M):
        """
        :param r_i: position of ball at moment of impact
        :param r_c: global coordinates of the point of contact
        :param V: cue velocity at moment of impact; the cue's velocity is assumed to be aligned with its axis
        :param M: cue mass
        """
        super().__init__(t, i)
        V = V.copy()
        V[1] = 0 # temporary: set vertical to 0
        self.V = V
        self.M = M
        Q = r_c - r_i
        self.Q = Q
        _j = V.copy(); _j[1] = 0; _j /= np.linalg.norm(_j)
        _i = np.cross(_j, self._k)
        a, c, b = (Q.dot(_i), #-_j[2] * Q[0] + _j[0] * Q[2],
                   Q.dot(_j),
                   Q[1])
        m, R, I = self.ball_mass, self.ball_radius, self.ball_I
        sin, cos = b/R, np.sqrt(a**2 + c**2)/R
        V_mag = np.linalg.norm(V)
        v_0_mag = 2 * V_mag / (1 + m / M + 5 / (2 * R**2) * (a**2 + b**2*cos**2 + c**2 * sin**2 - 2 * b * c * cos * sin))
        F = v_0_mag * m
        omega_0 = ((-c * F * sin + b * F * cos) * _i +
                   (a * F * sin)                * _j +
                   (-a * F * cos)          * self._k) / I
        self.next_motion_event = BallSlidingEvent(t, i, r_0=r_i, v_0=v_0_mag*_j, omega_0=omega_0)
    def __str__(self):
        return super().__str__()[:-1] + ' Q=%s V=%s M=%s>' % (self.Q, self.V, self.M)


class BallCollisionEvent(PhysicsEvent):
    def __init__(self, t, i, j, r_i, r_j, v_i, v_j):
        super().__init__(t)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T \
            and self.i == other.i and self.j == other.j

    def __hash__(self):
        return hash((self.__class__.__name__, self.i, self.j, self.t, self.T))
