import logging
import numpy as np


_logger = logging.getLogger(__name__)
INCH2METER = 0.0254


from ..decorators import allocs_out, allocs_out_vec4


class PhysicsEvent(object):
    ball_radius = 1.125 * INCH2METER
    ball_mass = 0.17
    ball_I = 2/5 * ball_mass * ball_radius**2
    mu_r = 0.016 # coefficient of rolling friction between ball and table
    mu_sp = 0.044 # coefficient of spinning friction between ball and table
    mu_s = 0.2 # coefficient of sliding friction between ball and table
    mu_b = 0.06 # coefficient of friction between ball and cushions
    c_b = 4000.0 # ball material's speed of sound
    E_Y_b = 2.4e9 # ball material's Young's modulus of elasticity
    g = 9.81 # magnitude of acceleration due to gravity
    _ZERO_TOLERANCE = 1e-7
    _k = np.array((0,1,0), dtype=np.float64) # basis vector :math`\hat{k}` of any ball-centered frame, following the
    def __init__(self, t, T=0.0, parent_event=None, **kwargs):
        """
        Base class of pool physics events.

        :param t: time of event start
        :param T: time duration of the event (default is 0, i.e. instantaneous)
        """
        self.t = t
        self.T = T
        self._parent_event = parent_event
    @property
    def child_events(self):
        return ()
    @property
    def parent_event(self):
        return self._parent_event
    @property
    def next_motion_event(self):
        return None
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
    @staticmethod
    def events_str(events=()):
        return '\n'.join('%4d: %s' % (i, e) for i, e in enumerate(events))
    def __lt__(self, other):
        if isinstance(other, PhysicsEvent):
            return self.t < other.t
        else:
            return self.t < other
    def __gt__(self, other):
        if isinstance(other, PhysicsEvent):
            return self.t > other.t
        else:
            return self.t > other
    def __str__(self):
        return '<%16s (%5.5f, %5.5f)>' % (self.__class__.__name__, self.t, self.t+self.T)


class BallEvent(PhysicsEvent):
    def __init__(self, t, i, **kwargs):
        super().__init__(t, **kwargs)
        self.i = i
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T and self.i == other.i
    def __hash__(self):
        return hash((self.__class__.__name__, self.i, self.t, self.T))
    def __str__(self):
        return super().__str__()[:-1] + " i=%d>" % self.i


class BallRestEvent(BallEvent):
    def __init__(self, t, i, r=None, q=None,
                 psi=0.0, theta=0.0, phi=0.0, **kwargs):
        # kwargs['T'] = float('inf')
        super().__init__(t, i, T=float('inf'), **kwargs)
        if r is None:
            self._r = self._r_0 = np.zeros(3, dtype=np.float64)
        else:
            self._r = self._r_0 = r.copy()
        self._psi, self._theta, self._phi = psi, theta, phi
        if q is None:
            self._q = self._q_0 = self.set_quaternion_from_euler_angles(psi=psi, theta=theta, phi=phi)
        else:
            self._q = self._q_0 = q.copy()
        self._a_global = None
    @property
    def global_motion_coeffs(self):
        if self._a_global is None:
            self._a_global = a = np.zeros((3,3), dtype=np.float64)
            a[0] = self._r
        return self._a_global, None
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
    def __str__(self):
        return super().__str__()[:-1] + ' r=%s>' % self._r


class BallMotionEvent(BallEvent):
    def __init__(self, t, i, T=None, a=None, b=None,
                 r_0=None, v_0=None, a_0=None, omega_0=None,
                 psi_0=0.0, theta_0=0.0, phi_0=0.0, **kwargs):
        """
        :param a: positional equation of motion coefficients (event-local time)
        :param b: angular velocity equation of motion coefficients (event-local time)
        :param r_0: ball position at start of event
        :param v_0: ball velocity at start of event
        :param omega_0: ball angular velocity at start of event
        """
        super().__init__(t, i, T=T, **kwargs)
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
        if omega_0 is not None:
            b[0] = omega_0
        self._r_0 = a[0]
        self._v_0 = a[1]
        self._half_a = a[2]
        self._omega_0 = b[0]
        self._half_alpha = b[1]
        self._ab_global = None
    @property
    def next_motion_event(self):
        return self._next_motion_event
    @property
    def global_motion_coeffs(self):
        if self._ab_global is None:
            self._ab_global = self.calc_global_motion_coeffs(self.t, self._a, self._b)
        return self._ab_global[:3], self._ab_global[3:]
    @staticmethod
    def calc_global_motion_coeffs(t, a, b):
        "Calculates the coefficients of the global-time equations of motion."
        ab_global = np.vstack((a, b))
        a_global, b_global = ab_global[:3], ab_global[3:]
        a_global[0] += -t * a[1] + t**2 * a[2]
        a_global[1] += -2 * t * a[2]
        b_global[0] += -t * b[1]
        return ab_global
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
    def __str__(self):
        return super().__str__()[:-1] + ' r_0=%s v_0=%s omega_0=%s>' % (self._r_0, self._v_0, self._omega_0)


class BallRollingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, **kwargs):
        v_0_mag = np.linalg.norm(v_0)
        T = v_0_mag / (self.mu_r * self.g)
        omega_0 = v_0 / self.ball_radius; omega_0[::2] = -omega_0[::-2]
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._a[2] = -0.5 * self.mu_r * self.g * v_0 / v_0_mag
        self._b[1] = -omega_0 / T
        self._next_motion_event = BallRestEvent(t + T, i, r=self.eval_position(T))


class BallSlidingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0, **kwargs):
        u_0 = v_0 + self.ball_radius * np.cross(self._k, omega_0)
        u_0_mag = np.linalg.norm(u_0)
        T = 2 * u_0_mag / (7 * self.mu_s * self.g)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._a[2] = -0.5 * self.mu_s * self.g * u_0 / u_0_mag
        self._b[1] = -5 * self.mu_s * self.g / (2 * self.ball_radius) * np.cross(self._k, u_0 / u_0_mag)
        self._next_motion_event = BallRollingEvent(t + T, i,
                                                   r_0=self.eval_position(T),
                                                   v_0=self.eval_velocity(T))


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
        a, c, b = (Q.dot(_i),
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
        self._child_events = (BallSlidingEvent(t, i, r_0=r_i, v_0=v_0_mag*_j, omega_0=omega_0, parent_event=self),)
    @property
    def child_events(self):
        return self._child_events
    @property
    def next_motion_event(self):
        return self._child_events[0]
    def __str__(self):
        return super().__str__()[:-1] + ' Q=%s V=%s M=%s>' % (self.Q, self.V, self.M)


class BallCollisionEvent(PhysicsEvent):
    def __init__(self, t, e_i, e_j):
        """Marlow collision model"""
        super().__init__(t)
        self.e_i, self.e_j = e_i, e_j
        self.i, self.j = e_i.i, e_j.i
        tau_i, tau_j = t - e_i.t, t - e_j.t
        self._r_i, self._r_j = e_i.eval_position(tau_i), e_j.eval_position(tau_j)
        self._v_i, self._v_j = e_i.eval_velocity(tau_i), e_j.eval_velocity(tau_j)
        self._omega_i, self._omega_j = e_i.eval_angular_velocity(tau_i), e_j.eval_angular_velocity(tau_j)
    @property
    def child_events(self):
        return self._child_events
    def __str__(self):
        return super().__str__()[:-1] + ' i=%s j=%s v_i_1=%s v_j_1=%s>' % (
            self.i, self.j, self._v_i_1, self._v_j_1)


class SimpleBallCollisionEvent(BallCollisionEvent):
    def __init__(self, t, e_i, e_j):
        """Perfectly elastic collision with no friction between balls."""
        super().__init__(t, e_i, e_j)
        i, j = self.i, self.j
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_i - r_j
        _i = r_ij / np.linalg.norm(r_ij)
        v_i, v_j = self._v_i, self._v_j
        v_i_1 = v_i + (v_j - v_i).dot(_i) * _i
        v_j_1 = v_j + (v_i - v_j).dot(_i) * _i
        v_j_1 = v_i.dot(_i) * _i
        v_i_1 = v_i - v_j_1
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        v_i_1_mag, v_j_1_mag = np.linalg.norm(v_i_1), np.linalg.norm(v_j_1)
        e_i_1, e_j_1 = None, None
        if v_i_1_mag < self._ZERO_TOLERANCE:
            e_i_1 = BallRestEvent(t, i, r_i, parent_event=self)
        if v_j_1_mag < self._ZERO_TOLERANCE:
            e_j_1 = BallRestEvent(t, j, r_j, parent_event=self)
        if isinstance(e_i, BallSlidingEvent) or isinstance(e_j, BallSlidingEvent):
            if e_i_1 is None:
                u_i_1 = v_i_1 + self.ball_radius * np.cross(self._k, self._omega_i)
                u_i_1_mag = np.linalg.norm(u_i_1)
                if u_i_1_mag >= self._ZERO_TOLERANCE:
                    e_i_1 = BallSlidingEvent(t, i, r_i, v_i_1, self._omega_i, parent_event=self)
            if e_j_1 is None:
                u_j_1 = v_j_1 + self.ball_radius * np.cross(self._k, self._omega_j)
                u_j_1_mag = np.linalg.norm(u_j_1)
                if u_j_1_mag >= self._ZERO_TOLERANCE:
                    e_j_1 = BallSlidingEvent(t, j, r_j, v_j_1, self._omega_j, parent_event=self)
        if e_i_1 is None:
            e_i_1 = BallRollingEvent(t, i, r_i, v_i_1, parent_event=self)
        if e_j_1 is None:
            e_j_1 = BallRollingEvent(t, j, r_j, v_j_1, parent_event=self)
        self._child_events = (e_i_1, e_j_1)


class MarlowBallCollisionEvent(BallCollisionEvent):
    def __init__(self, t, e_i, e_j):
        super().__init__(t, e_i, e_j)
        i, j = self.i, self.j
        v_i, v_j = self._v_i, self._v_j
        v_ij = v_j - v_i
        v_ij_mag = np.linalg.norm(v_ij)
        delta_t = 284e-6 / v_ij_mag**0.294
        s_max = 1.65765 * (v_ij_mag / self.c_b)**0.8
        F_max = 1.48001 * self.ball_radius**2 * self.E_Y_b * s_max**1.5
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_j - r_i
        _i = r_ij / np.linalg.norm(r_ij)
        J = max(0.5 * F_max * delta_t,
                abs(self.ball_mass * v_ij.dot(_i))) # missing 2 factor?
        v_i_1 = v_i - (J / self.ball_mass) * _i
        v_j_1 = v_j + (J / self.ball_mass) * _i
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        self._child_events = (
            # ball i event:
            BallRestEvent(t, i, r_i, parent_event=self) if np.linalg.norm(v_i_1) < self._ZERO_TOLERANCE
            else BallRollingEvent(t, i, r_i, v_i_1, parent_event=self),
            # ball j event:
            BallRestEvent(t, j, r_j, parent_event=self) if np.linalg.norm(v_j_1) < self._ZERO_TOLERANCE
            else BallRollingEvent(t, j, r_j, v_j_1, parent_event=self)
        )
