import logging
_logger = logging.getLogger(__name__)
import numpy as np
cimport numpy as np


_k = np.array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
              dtype=np.float64) # of any ball-centered frame, following the convention of Marlow


cdef double INCH2METER = 0.0254
cdef double ball_radius = 1.125 * INCH2METER
cdef double ball_mass = 0.17
cdef double ball_I = 2.0/5 * ball_mass * ball_radius**2
cdef double mu_r = 0.016 # coefficient of rolling friction between ball and table
cdef double mu_sp = 0.044 # coefficient of spinning friction between ball and table
cdef double mu_s = 0.2 # coefficient of sliding friction between ball and table
cdef double mu_b = 0.06 # coefficient of friction between ball and cushions
cdef double g = 9.81 # magnitude of acceleration due to gravity
cdef double c_b = 4000.0 # ball material's speed of sound
cdef double E_Y_b = 2.4e9 # ball material's Young's modulus of elasticity
cdef double _ZERO_TOLERANCE = 1e-8
cdef double _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2


cdef class PhysicsEvent:
    cdef public double t
    cdef public double T
    cdef public object _parent_event
    def __init__(self, double t, double T=0.0, parent_event=None, **kwargs):
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
    def set_quaternion_from_euler_angles(double psi=0.0, double theta=0.0, double phi=0.0, out=None):
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
    def events_str(events, sep='\n\n' + 48*'-' + '\n\n'):
        return sep.join('%3d:\n%s' % (i_e, e) for i_e, e in enumerate(events))
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


cdef class BallEvent(PhysicsEvent):
    cdef public int i
    def __init__(self, double t, int i, **kwargs):
        self.i = i
        super().__init__(t, **kwargs)
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T and self.i == other.i
    def __str__(self):
        return super().__str__()[:-1] + " i=%d>" % self.i


cdef class BallStationaryEvent(BallEvent):
    # cdef public double[3] _r_0
    # cdef public double[3] _q_0
    cdef public object _r_0
    cdef public object _q_0
    cdef public object _a_global
    def __init__(self, double t, int i, r_0=None, q_0=None,
                 double psi=0.0, double theta=0.0, double phi=0.0, **kwargs):
        super().__init__(t, i, **kwargs)
        if q_0 is None:
            q_0 = self.set_quaternion_from_euler_angles(psi=psi, theta=theta, phi=phi)
        self._q_0 = q_0
        if r_0 is None:
            r_0 = np.zeros(3, dtype=np.float64)
        self._r_0 = r_0
        self._a_global = None
    @property
    def acceleration(self):
        return np.zeros(3, dtype=np.float64)
    @property
    def global_motion_coeffs(self):
        if self._a_global is None:
            self._a_global = a = np.zeros((3,3), dtype=np.float64)
            a[0] = self._r_0
        return self._a_global, None
    def eval_position(self, double tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        return out
    def eval_velocity(self, double tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out
    def eval_slip_velocity(self, double tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out
    def __str__(self):
        return super().__str__()[:-1] + '\n r=%s>' % self._r_0


cdef class BallRestEvent(BallStationaryEvent):
    def __init__(self, double t, int i, **kwargs):
        super().__init__(t, i, T=float('inf'), **kwargs)
    def eval_angular_velocity(self, double tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out


cdef class BallSpinningEvent(BallStationaryEvent):
    cdef public double _omega_0_y
    cdef public double _b
    cdef public object _next_motion_event
    def __init__(self, double t, int i, r_0, double omega_0_y, **kwargs):
        R = ball_radius
        self._omega_0_y = omega_0_y
        self._b = -5 * np.sign(omega_0_y) * mu_sp * g / (2 * R)
        T = abs(omega_0_y / self._b)
        super().__init__(t, i, T=T, r_0=r_0, **kwargs)
        self._next_motion_event = BallRestEvent(t + T, i, r_0=r_0)
    @property
    def next_motion_event(self):
        return self._next_motion_event
    def eval_angular_velocity(self, double tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        if 0 <= tau <= self.T:
            out[1] = self._omega_0_y + self._b * tau
        return out


cdef class BallMotionEvent(BallEvent):
    cdef public object _a
    cdef public object _b
    # cdef public double[:,:] _a
    # cdef public double[:,:] _b
    cdef public object _r_0
    cdef public object _v_0
    cdef public object _q_0
    cdef public object _omega_0
    # cdef public double[3] _r_0
    # cdef public double[3] _v_0
    # cdef public double[3] _q_0
    # cdef public double[3] _omega_0
    cdef public object _ab_global
    cdef public object _next_motion_event
    def __init__(self, double t, int i, double T=0.0,
                 a=None, b=None,
                 r_0=None, v_0=None, a_0=None,
                 q_0=None, omega_0=None,
                 double psi_0=0.0, double theta_0=0.0, double phi_0=0.0, **kwargs):
        """
        :param t: start time of event
        :param T: duration of event
        :param a: coefficients of the positional quadratic equation of motion (event-local time)
        :param b: coefficients of the angular velocity linear equation of motion (event-local time)
        :param r_0: ball position at start of event
        :param v_0: ball velocity at start of event
        :param omega_0: ball angular velocity at start of event
        """
        super().__init__(t, i, T=T, **kwargs)
        if a is None:
            a = np.zeros((3,3), dtype=np.float64)
        if b is None:
            b = np.zeros((2,3), dtype=np.float64)
        if r_0 is not None:
            a[0] = r_0
        if q_0 is None:
            q_0 = self.set_quaternion_from_euler_angles(psi=psi_0, theta=theta_0, phi=phi_0)
        if v_0 is not None:
            a[1] = v_0
        if a_0 is not None:
            a[2] = 0.5 * a_0
        if omega_0 is not None:
            b[0] = omega_0
        self._a = a
        self._b = b
        self._r_0 = a[0]
        self._v_0 = a[1]
        self._q_0 = q_0
        self._omega_0 = b[0]
        self._ab_global = None
    @property
    def acceleration(self):
        return 2 * self._a[2]
    @property
    def next_motion_event(self):
        return self._next_motion_event
    @property
    def global_motion_coeffs(self):
        if self._ab_global is None:
            self._ab_global = self.calc_global_motion_coeffs(self.t, self._a, self._b)
        return self._ab_global[:3], self._ab_global[3:]
    @staticmethod
    def calc_global_motion_coeffs(double t, a, b, out=None):
        "Calculates the coefficients of the global-time equations of motion."
        if out is None:
            out = np.zeros((5,3), dtype=np.float64)
        out[:3] = a
        out[3:] = b
        a_global, b_global = out[:3], out[3:]
        a_global[0] += -t * a[1] + t**2 * a[2]
        a_global[1] += -2 * t * a[2]
        b_global[0] += -t * b[1]
        return out
    def eval_position(self, double tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        if tau != 0.0:
            a = self._a
            out += tau * a[1] + tau**2 * a[2]
        return out
    def eval_velocity(self, double tau, out=None):
        if out is None:
            out = self._v_0.copy()
        else:
            out[:] = self._v_0
        if tau != 0:
            out += 2 * tau * self._a[2]
        return out
    def eval_angular_velocity(self, double tau, out=None):
        if out is None:
            out = np.empty(3, dtype=np.float64)
        out[:] = self._b[0] + tau * self._b[1]
        if self._b[0,1] >= 0:
            out[1] = max(0, out[1])
        else:
            out[1] = min(0, out[1])
        return out
    def eval_slip_velocity(self, double tau, v=None, omega=None, out=None):
        if v is None:
            v = self.eval_velocity(tau)
        if omega is None:
            omega = self.eval_angular_velocity(tau)
        if out is None:
            out = np.empty(3, dtype=np.float64)
        out[:] = v + ball_radius * np.cross(_k, omega)
        return out
    def __str__(self):
        return super().__str__()[:-1] + '\n r_0=%s\n v_0=%s\n a=%s\n omega_0=%s>' % (self._r_0, self._v_0, self.acceleration, self._omega_0)


cdef class BallRollingEvent(BallMotionEvent):
    def __init__(self, double t, int i, r_0, v_0, double omega_0_y=0.0, **kwargs):
        R = ball_radius
        v_0_mag = np.linalg.norm(v_0)
        T = v_0_mag / (mu_r * g)
        omega_0 = np.array((v_0[2]/R, omega_0_y, -v_0[0]/R), dtype=np.float64)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._a[2] = -0.5 * mu_r * g * v_0 / v_0_mag
        self._b[1,::2] = -omega_0[::2] / T
        self._b[1,1] = -np.sign(omega_0_y) * 5 / 7 * mu_r * g / R
        #self._b[1,1] = -np.sign(omega_0_y) * 5 / 2 * mu_sp * g / R
        omega_1 = self.eval_angular_velocity(T)
        if abs(omega_1[1]) < _ZERO_TOLERANCE:
            self._next_motion_event = BallRestEvent(t + T, i, r_0=self.eval_position(T))
        else:
            self._next_motion_event = BallSpinningEvent(t + T, i, r_0=self.eval_position(T),
                                                        omega_0_y=omega_1[1])
    def eval_slip_velocity(self, double tau, out=None, **kwargs):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out


cdef class BallSlidingEvent(BallMotionEvent):
    cdef public object _u_0
    # cdef public double[3] _u_0
    # cdef public object _u_0_mag
    cdef public double _u_0_mag
    def __init__(self, double t, int i, r_0, v_0, omega_0, **kwargs):
        R = ball_radius
        u_0 = v_0 + R * np.cross(_k, omega_0)
        u_0_mag = np.sqrt(u_0.dot(u_0))
        T = 2 * u_0_mag / (7 * mu_s * g)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._u_0 = u_0
        self._u_0_mag = u_0_mag
        self._a[2] = -0.5 * mu_s * g * u_0 / u_0_mag
        self._b[0] = omega_0
        self._b[1,::2] = 5 * mu_s * g / (2 * R) * np.cross(_k, u_0 / u_0_mag)[::2]
        self._b[1,1] = -np.sign(omega_0[1]) * 5 * mu_sp * g / (2 * R)
        omega_1 = self.eval_angular_velocity(T)
        self._next_motion_event = BallRollingEvent(t + T, i,
                                                   r_0=self.eval_position(T),
                                                   v_0=self.eval_velocity(T),
                                                   omega_0_y=omega_1[1])


cdef class CueStrikeEvent(BallEvent):
    cdef public double[3] V
    cdef public double[3] Q
    cdef public double M
    cdef public object _child_events
    def __init__(self, double t, int i, r_i, r_c, V, double M, q_i=None):
        """
        :param r_i: position of ball at moment of impact
        :param r_c: global coordinates of the point of contact
        :param V: cue velocity at moment of impact; the cue's velocity is assumed to be aligned with its axis
        :param M: cue mass
        :param q_i: rotation quaternion of ball at moment of impact
        """
        super().__init__(t, i, T=0.0)
        m, R, I = ball_mass, ball_radius, ball_I
        V = V.copy()
        V[1] = 0 # temporary: set vertical to 0
        self.V[:] = V
        self.M = M
        self.Q[:] = Q = r_c - r_i
        _j = -V; _j[1] = 0; _j /= np.sqrt(_j.dot(_j))
        _i = np.cross(_j, _k)
        a, b = Q.dot(_i), Q[1]
        c = np.sqrt(R**2 - a**2 - b**2)
        sin, cos = b/R, np.sqrt(R**2 - b**2)/R
        V_mag = np.sqrt(V.dot(V))
        F_mag = 2*m*V_mag / (
            1 + m/M + 5/(2*R**2)*(a**2 + b**2*cos**2 + c**2*sin**2 - 2*b*c*cos*sin)
        )
        omega_0 = ( (-c*F_mag*sin + b*F_mag*cos) * _i +
                                   (a*F_mag*sin) * _j +
                                  (-a*F_mag*cos) * _k ) / I
        self._child_events = (BallSlidingEvent(t, i,
                                               r_0=r_i,
                                               v_0=-F_mag/m*_j,
                                               omega_0=omega_0,
                                               q_0=q_i,
                                               parent_event=self),)
    @property
    def child_events(self):
        return self._child_events
    @property
    def next_motion_event(self):
        return self._child_events[0]
    def __str__(self):
        return super().__str__()[:-1] + '\n Q=%s\n V=%s\n M=%s>' % (self.Q, self.V, self.M)


cdef class RailCollisionEvent(BallEvent):
    cdef public double kappa # coefficient of restitution
    cdef public int side
    cdef public object e_i
    cdef public BallSlidingEvent _child_event
    def __init__(self, double t, e_i, int side):
        super().__init__(t, e_i.i)
        self.kappa = 0.6
        self.e_i = e_i
        self.side = side
        tau = t - e_i.t
        r_1 = e_i.eval_position(tau)
        v_1 = e_i.eval_velocity(tau)
        omega_1 = e_i.eval_angular_velocity(tau)
        omega_1[::2] = 0
        self._child_event = BallSlidingEvent(self.t, self.e_i.i,
                                             r_0=r_1,
                                             v_0=v_1,
                                             omega_0=omega_1,
                                             parent_event=self)
    @property
    def child_events(self):
        return (self._child_event,)
    def __str__(self):
        return super().__str__()[:-1] + " side=%d>" % self.side


cdef class BallCollisionEvent(PhysicsEvent):
    cdef public int i
    cdef public int j
    cdef public object e_i
    cdef public object e_j
    cdef public np.ndarray _r_i
    cdef public np.ndarray _r_j
    cdef public np.ndarray _v_i
    cdef public np.ndarray _v_j
    cdef public np.ndarray _omega_i
    cdef public np.ndarray _omega_j
    cdef public np.ndarray _v_i_1
    cdef public np.ndarray _v_j_1
    cdef public np.ndarray _omega_i_1
    cdef public np.ndarray _omega_j_1
    cdef public object _child_events
    def __init__(self, double t, e_i, e_j):
        super().__init__(t)
        self.e_i, self.e_j = e_i, e_j
        self.i, self.j = e_i.i, e_j.i
        tau_i, tau_j = t - e_i.t, t - e_j.t
        self._r_i, self._r_j = e_i.eval_position(tau_i), e_j.eval_position(tau_j)
        self._v_i, self._v_j = e_i.eval_velocity(tau_i), e_j.eval_velocity(tau_j)
        self._omega_i, self._omega_j = e_i.eval_angular_velocity(tau_i), e_j.eval_angular_velocity(tau_j)
        self._child_events = None
    @property
    def child_events(self):
        if self._child_events is None:
            e_i, e_j = self.e_i, self.e_j
            v_i_1, v_j_1 = self._v_i_1, self._v_j_1
            omega_i_1, omega_j_1 = self._omega_i_1, self._omega_j_1
            # e_i_1 = BallSlidingEvent(self.t, e_i.i,
            #                          r_0=self._r_i,
            #                          v_0=v_i_1,
            #                          omega_0=omega_i_1,
            #                          parent_event=self)
            # e_j_1 = BallSlidingEvent(self.t, e_j.i,
            #                          r_0=self._r_j,
            #                          v_0=v_j_1,
            #                          omega_0=omega_j_1,
            #                          parent_event=self)
            if v_i_1.dot(v_i_1) < _ZERO_TOLERANCE_SQRD:
                if omega_i_1[1] < _ZERO_TOLERANCE:
                    e_i_1 = BallRestEvent(self.t, e_i.i,
                                          r_0=self._r_i,
                                          parent_event=self)
                else:
                    e_i_1 = BallSpinningEvent(self.t, e_i.i,
                                              r_0=self._r_i,
                                              omega_0_y=omega_i_1[1],
                                              parent_event=self)
                if isinstance(e_j, BallRestEvent):
                    if isinstance(e_i, BallRollingEvent):
                        e_j_1 = BallRollingEvent(self.t, e_j.i,
                                                 r_0=self._r_j,
                                                 v_0=v_j_1,
                                                 parent_event=self)
                    else:
                        e_j_1 = BallSlidingEvent(self.t, e_j.i,
                                                 r_0=self._r_j,
                                                 v_0=v_j_1,
                                                 omega_0=omega_j_1,
                                                 parent_event=self)
                elif omega_j_1[1] < _ZERO_TOLERANCE:
                    e_j_1 = BallRestEvent(self.t, e_j.i,
                                          r_0=self._r_j,
                                          parent_event=self)
                else:
                    e_j_1 = BallSpinningEvent(self.t, e_j.i,
                                              r_0=self._r_j,
                                              omega_0_y=omega_j_1[1],
                                              parent_event=self)
            else:
                u_i_1 = e_i.eval_slip_velocity(e_i.T)
                if u_i_1.dot(u_i_1) < _ZERO_TOLERANCE_SQRD:
                    e_i_1 = BallRollingEvent(self.t, e_i.i,
                                             r_0=self._r_i,
                                             v_0=v_i_1,
                                             parent_event=self)
                else:
                    e_i_1 = BallSlidingEvent(self.t, e_i.i,
                                             r_0=self._r_i,
                                             v_0=v_i_1,
                                             omega_0=omega_i_1,
                                             parent_event=self)
                if v_j_1.dot(v_j_1) < _ZERO_TOLERANCE_SQRD:
                    if omega_j_1[1] < _ZERO_TOLERANCE:
                        e_j_1 = BallRestEvent(self.t, e_j.i,
                                              r_0=self._r_j,
                                              parent_event=self)
                    else:
                        e_j_1 = BallSpinningEvent(self.t, e_j.i,
                                                  r_0=self._r_j,
                                                  omega_0_y=omega_j_1[1],
                                                  parent_event=self)
                else:
                    u_j_1 = e_j.eval_slip_velocity(e_j.T)
                    if u_j_1.dot(u_j_1) < _ZERO_TOLERANCE_SQRD:
                        e_j_1 = BallRollingEvent(self.t, e_j.i,
                                                 r_0=self._r_j,
                                                 v_0=v_j_1,
                                                 parent_event=self)
                    else:
                        e_j_1 = BallSlidingEvent(self.t, e_j.i,
                                                 r_0=self._r_j,
                                                 v_0=v_j_1,
                                                 omega_0=omega_j_1,
                                                 parent_event=self)
            self._child_events = (e_i_1, e_j_1)
        return self._child_events
    def __str__(self):
        return super().__str__()[:-1] + ' i=%s j=%s\n v_i_0=%s\n v_j_0=%s\n v_i_1=%s\n v_j_1=%s\n v_ij_0=%s\n v_ij_1=%s>' % (
            self.i, self.j, self._v_i, self._v_j, self._v_i_1, self._v_j_1, self._v_i - self._v_j, self._v_i_1 - self._v_j_1)


cdef class SimpleBallCollisionEvent(BallCollisionEvent):
    def __init__(self, double t, e_i, e_j, double v_factor=0.98):
        """Simple one-parameter elastic collision model with no friction between balls or any other surface."""
        super().__init__(t, e_i, e_j)
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_i - r_j
        _i = r_ij / np.linalg.norm(r_ij)
        v_i, v_j = self._v_i, self._v_j
        vp_i = v_i.dot(_i) * _i
        vp_j = v_j.dot(_i) * _i
        vo_i = v_i - vp_i
        vo_j = v_j - vp_j
        vp_i_1 = 0.5 * ((1 - v_factor) * vp_i + (1 + v_factor) * vp_j)
        vp_j_1 = 0.5 * ((1 - v_factor) * vp_j + (1 + v_factor) * vp_i)
        v_i_1 = vo_i + vp_i_1
        v_j_1 = vo_j + vp_j_1
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        self._omega_i_1, self._omega_j_1 = self._omega_i, self._omega_j


cdef class MarlowBallCollisionEvent(BallCollisionEvent):
    def __init__(self, double t, e_i, e_j):
        """Marlow collision model"""
        super().__init__(t, e_i, e_j)
        v_i, v_j = self._v_i, self._v_j
        v_ij = v_j - v_i
        v_ij_mag = np.linalg.norm(v_ij)
        delta_t = 284e-6 / v_ij_mag**0.294
        s_max = 1.65765 * (v_ij_mag / c_b)**0.8
        F_max = 1.48001 * ball_radius**2 * E_Y_b * s_max**1.5
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_j - r_i
        _i = r_ij / np.linalg.norm(r_ij)
        J = max(0.5 * F_max * delta_t,
                abs(ball_mass * v_ij.dot(_i))) # missing 2 factor?
        v_i_1 = v_i - (J / ball_mass) * _i
        v_j_1 = v_j + (J / ball_mass) * _i
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        self._omega_i_1, self._omega_j_1 = self._omega_i, self._omega_j
