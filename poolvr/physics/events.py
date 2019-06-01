import logging
_logger = logging.getLogger(__name__)
import numpy as np


INCH2METER = 0.0254
_k = np.array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
              dtype=np.float64) # of any ball-centered frame, following the convention of Marlow


class PhysicsEvent(object):
    ball_radius = 1.125 * INCH2METER
    ball_mass = 0.17
    ball_I = 2/5 * ball_mass * ball_radius**2
    mu_r = 0.016 # coefficient of rolling friction between ball and table
    mu_sp = 0.044 # coefficient of spinning friction between ball and table
    mu_s = 0.2 # coefficient of sliding friction between ball and table
    mu_b = 0.06 # coefficient of friction between ball and cushions
    g = 9.81 # magnitude of acceleration due to gravity
    _ZERO_TOLERANCE = 1e-8
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
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
    @staticmethod
    def set_quaternion_from_euler_angles(psi=0.0, theta=0.0, phi=0.0, out=None):
        if out is None: out = np.empty(4, dtype=np.float64)
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


class BallEvent(PhysicsEvent):
    def __init__(self, t, i, **kwargs):
        super().__init__(t, **kwargs)
        self.i = i
    @property
    def next_motion_event(self):
        return None
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T and self.i == other.i
    def __str__(self):
        return super().__str__()[:-1] + " i=%d>" % self.i


class BallStationaryEvent(BallEvent):
    def __init__(self, t, i, r_0=None, q_0=None,
                 psi=0.0, theta=0.0, phi=0.0, **kwargs):
        super().__init__(t, i, **kwargs)
        if r_0 is None:
            r_0 = np.zeros(3, dtype=np.float64)
        self._r_0 = self._r = r_0
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
    def calc_shifted_motion_coeffs(self, t0):
        return self.global_motion_coeffs
    def eval_position(self, tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        return out
    def eval_velocity(self, tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out
    def eval_slip_velocity(self, tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out
    def __str__(self):
        return super().__str__()[:-1] + '\n r=%s>' % self._r


class BallRestEvent(BallStationaryEvent):
    def __init__(self, t, i, **kwargs):
        super().__init__(t, i, T=float('inf'), **kwargs)
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out


class BallSpinningEvent(BallStationaryEvent):
    def __init__(self, t, i, r_0, omega_0_y, **kwargs):
        R = self.ball_radius
        self._omega_0_y = omega_0_y
        self._b = -5 * np.sign(omega_0_y) * self.mu_sp * self.g / (2 * R)
        T = abs(omega_0_y / self._b)
        super().__init__(t, i, r_0=r_0, T=T, **kwargs)
        self._next_motion_event = None
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            self._next_motion_event = BallRestEvent(self.t + self.T, self.i, r_0=self._r_0)
        return self._next_motion_event
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        if 0 <= tau <= self.T:
            out[1] = self._omega_0_y + self._b * tau
        return out


class BallMotionEvent(BallEvent):
    def __init__(self, t, i, T=None, a=None, b=None,
                 r_0=None, v_0=None, a_0=None,
                 q_0=None, omega_0=None,
                 psi_0=0.0, theta_0=0.0, phi_0=0.0, **kwargs):
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
        self._a = a
        self._b = b
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
        self._omega_0 = b[0]
        self._ab_global = None
        self._next_motion_event = None
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
    def calc_shifted_motion_coeffs(self, t0):
        ab_global = self.calc_global_motion_coeffs(self.t - t0, self._a, self._b)
        return ab_global[:3], ab_global[3:]
    @staticmethod
    def calc_global_motion_coeffs(t, a, b, out=None):
        """
        Calculates the coefficients of the global-time equations of motion.

        :param t: the global time of the start of the motion
        :param a: the local-time (0 at the start of the motion) linear motion coefficients
        :param b: the local-time angular motion coefficients
        """
        if out is None:
            out = np.zeros((5,3), dtype=np.float64)
        out[:3] = a
        out[3:] = b
        a_global, b_global = out[:3], out[3:]
        a_global[0] += -t * a[1] + t**2 * a[2]
        a_global[1] += -2 * t * a[2]
        b_global[0] += -t * b[1]
        return out
    def eval_position(self, tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        if tau != 0:
            a = self._a
            out += tau * a[1] + tau**2 * a[2]
        return out
    def eval_velocity(self, tau, out=None):
        if out is None:
            out = self._v_0.copy()
        else:
            out[:] = self._v_0
        if tau != 0:
            out += 2 * tau * self._a[2]
        return out
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = np.empty(3, dtype=np.float64)
        out[:] = self._b[0] + tau * self._b[1]
        if self._b[0,1] >= 0:
            out[1] = max(0, out[1])
        else:
            out[1] = min(0, out[1])
        return out
    def eval_slip_velocity(self, tau, v=None, omega=None, out=None):
        if v is None:
            v = self.eval_velocity(tau)
        if omega is None:
            omega = self.eval_angular_velocity(tau)
        if out is None:
            out = np.empty(3, dtype=np.float64)
        out[:] = v + self.ball_radius * np.cross(_k, omega)
        return out
    def eval_position_and_velocity(self, tau, out=None):
        if out is None:
            out = np.empty((2,3), dtype=np.float64)
        taus = np.array((1.0, tau, tau**2))
        a = self._a
        np.dot(taus, a, out=out[0])
        out[1] = a[1] + 2*tau*a[2]
        return out
    def __str__(self):
        return super().__str__()[:-1] + '\n r_0=%s\n v_0=%s\n a=%s\n omega_0=%s>' % (self._r_0, self._v_0, self.acceleration, self._omega_0)


class BallRollingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0_y=0.0, **kwargs):
        R = self.ball_radius
        v_0_mag = np.sqrt(np.dot(v_0, v_0))
        T = v_0_mag / (self.mu_r * self.g)
        omega_0 = np.array((v_0[2]/R, omega_0_y, -v_0[0]/R), dtype=np.float64)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._a[2] = -0.5 * self.mu_r * self.g * v_0 / v_0_mag
        self._b[1,::2] = -omega_0[::2] / T
        self._b[1,1] = -np.sign(omega_0_y) * 5 / 7 * self.mu_r * self.g / R
        #self._b[1,1] = -np.sign(omega_0_y) * 5 / 2 * self.mu_sp * self.g / R
        self._next_motion_event = None
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            i, t, T = self.i, self.t, self.T
            omega_1 = self.eval_angular_velocity(T)
            if abs(omega_1[1]) < self._ZERO_TOLERANCE:
                self._next_motion_event = BallRestEvent(t + T, i, r_0=self.eval_position(T))
            else:
                self._next_motion_event = BallSpinningEvent(t + T, i, r_0=self.eval_position(T),
                                                            omega_0_y=omega_1[1])
        return self._next_motion_event
    def eval_slip_velocity(self, tau, out=None, **kwargs):
        if out is None:
            out = np.zeros(3, dtype=np.float64)
        else:
            out[:] = 0
        return out


class BallSlidingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0, **kwargs):
        R = self.ball_radius
        u_0 = v_0 + R * np.cross(_k, omega_0)
        u_0_mag = np.sqrt(np.dot(u_0, u_0))
        T = 2 * u_0_mag / (7 * self.mu_s * self.g)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._u_0 = u_0
        self._u_0_mag = u_0_mag
        self._a[2] = -0.5 * self.mu_s * self.g * u_0 / u_0_mag
        self._b[0] = omega_0
        self._b[1,::2] = 5 * self.mu_s * self.g / (2 * R) * np.cross(_k, u_0 / u_0_mag)[::2]
        self._b[1,1] = -np.sign(omega_0[1]) * 5 * self.mu_sp * self.g / (2 * R)
        self._next_motion_event = None
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            i, t, T = self.i, self.t, self.T
            omega_1 = self.eval_angular_velocity(T)
            self._next_motion_event = BallRollingEvent(t + T, i,
                                                       r_0=self.eval_position(T),
                                                       v_0=self.eval_velocity(T),
                                                       omega_0_y=omega_1[1])
        return self._next_motion_event


class CueStrikeEvent(BallEvent):
    def __init__(self, t, i, r_i, r_c, V, M, q_i=None):
        """
        :param r_i: position of ball at moment of impact
        :param r_c: global coordinates of the point of contact
        :param V: cue velocity at moment of impact; the cue's velocity is assumed to be aligned with its axis
        :param M: cue mass
        :param q_i: rotation quaternion of ball at moment of impact
        """
        super().__init__(t, i)
        m, R, I = self.ball_mass, self.ball_radius, self.ball_I
        V = V.copy()
        V[1] = 0 # temporary: set vertical to 0
        self.V = V
        self.M = M
        self.Q = Q = r_c - r_i
        _j = -V; _j[1] = 0; _j /= np.sqrt(np.dot(_j, _j))
        _i = np.cross(_j, _k)
        a, b = np.dot(Q, _i), Q[1]
        c = np.sqrt(R**2 - a**2 - b**2)
        sin, cos = b/R, np.sqrt(R**2 - b**2)/R
        V_mag = np.sqrt(np.dot(V, V))
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
    def __str__(self):
        return super().__str__()[:-1] + '\n Q=%s\n V=%s\n M=%s>' % (self.Q, self.V, self.M)


class RailCollisionEvent(BallEvent):
    kappa = 0.6 # coefficient of restitution
    def __init__(self, t, e_i, side):
        super().__init__(t, e_i.i)
        self.e_i = e_i
        self.side = side
        tau = t - e_i.t
        self._r_1 = e_i.eval_position(tau)
        self._v_1 = e_i.eval_velocity(tau)
        omega_1 = e_i.eval_angular_velocity(tau)
        self._omega_1 = np.zeros(3, dtype=np.float64) #e_i.eval_angular_velocity(tau)
        self._omega_1[1] = 0.9*omega_1[1]
        self._child_events = None
    @property
    def child_events(self):
        if self._child_events is None:
            v_1 = self._v_1.copy()
            v_1[2*(1-(self.side % 2))] *= -self.kappa
            self._child_events = (BallSlidingEvent(self.t, self.e_i.i,
                                                   r_0=self._r_1,
                                                   v_0=v_1,
                                                   omega_0=self._omega_1,
                                                   parent_event=self),)
        return self._child_events
    def __str__(self):
        return super().__str__()[:-1] + " side=%d>" % self.side


class BallCollisionEvent(PhysicsEvent):
    _ZERO_VELOCITY_CLIP = 0.001
    _ZERO_VELOCITY_CLIP_SQRD = _ZERO_VELOCITY_CLIP**2
    _ZERO_ANGULAR_VELOCITY_CLIP = 0.001
    def __init__(self, t, e_i, e_j):
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
            child_events = []
            x_loc, y_loc = self._i, self._j
            for (r, v_1, omega_1, e) in (
                    (self._r_i, self._v_i_1, self._omega_i_1, self.e_i),
                    (self._r_j, self._v_j_1, self._omega_j_1, self.e_j)
            ):
                if np.dot(v_1, v_1) < self._ZERO_VELOCITY_CLIP_SQRD:
                    if abs(omega_1[1]) < self._ZERO_ANGULAR_VELOCITY_CLIP:
                        e_1 = BallRestEvent(self.t, e.i,
                                            r_0=r,
                                            parent_event=self)
                    else:
                        e_1 = BallSpinningEvent(self.t, e.i,
                                                r_0=r,
                                                omega_0_y=omega_1[1],
                                                parent_event=self)
                else:
                    if abs(np.dot(v_1, y_loc) / self.ball_radius) <= abs(np.dot(omega_1, x_loc)):
                        e_1 = BallRollingEvent(self.t, e.i,
                                               r_0=r,
                                               v_0=v_1,
                                               parent_event=self)
                    else:
                        e_1 = BallSlidingEvent(self.t, e.i,
                                               r_0=r,
                                               v_0=v_1,
                                               omega_0=omega_1,
                                               parent_event=self)
                child_events.append(e_1)
            self._child_events = tuple(child_events)
        return self._child_events
    def __str__(self):
        return super().__str__()[:-1] + ' i=%s j=%s\n v_i_0=%s\n v_j_0=%s\n v_i_1=%s\n v_j_1=%s\n v_ij_0=%s\n v_ij_1=%s>' % (
            self.i, self.j, self._v_i, self._v_j, self._v_i_1, self._v_j_1, self._v_i - self._v_j, self._v_i_1 - self._v_j_1)


class SimpleBallCollisionEvent(BallCollisionEvent):
    def __init__(self, t, e_i, e_j, v_factor=0.98):
        """Simple one-parameter elastic collision model with no friction between balls or any other surface."""
        super().__init__(t, e_i, e_j)
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_i - r_j
        self._i = _i = r_ij / np.sqrt(np.dot(r_ij, r_ij))
        self._j = _j = np.array((_i[2], 0.0, -_i[0]))
        v_i, v_j = self._v_i, self._v_j
        vp_i = np.dot(v_i, _i) * _i
        vp_j = np.dot(v_j, _i) * _i
        vo_i = v_i - vp_i
        vo_j = v_j - vp_j
        vp_i_1 = 0.5 * ((1 - v_factor) * vp_i + (1 + v_factor) * vp_j)
        vp_j_1 = 0.5 * ((1 - v_factor) * vp_j + (1 + v_factor) * vp_i)
        v_i_1 = vo_i + vp_i_1
        v_j_1 = vo_j + vp_j_1
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        omega_i, omega_j = self._omega_i, self._omega_j
        self._omega_i_1 = omega_i - np.dot(omega_i, _j) * _j
        self._omega_i_1 += np.dot(v_i_1 / self.ball_radius, _j) * _j
        self._omega_j_1 = omega_j - np.dot(omega_j, _j) * _j
        self._omega_j_1 += np.dot(v_j_1 / self.ball_radius, _j) * _j


class MarlowBallCollisionEvent(BallCollisionEvent):
    c_b = 4000.0 # ball material's speed of sound
    E_Y_b = 2.4e9 # ball material's Young's modulus of elasticity
    def __init__(self, t, e_i, e_j):
        """Marlow collision model"""
        super().__init__(t, e_i, e_j)
        v_i, v_j = self._v_i, self._v_j
        v_ij = v_j - v_i
        v_ij_mag = np.linalg.norm(v_ij)
        delta_t = 284e-6 / v_ij_mag**0.294
        s_max = 1.65765 * (v_ij_mag / self.c_b)**0.8
        F_max = 1.48001 * self.ball_radius**2 * self.E_Y_b * s_max**1.5
        r_i, r_j = self._r_i, self._r_j
        r_ij = r_j - r_i
        self._i = _i = r_ij / np.linalg.norm(r_ij)
        J = max(0.5 * F_max * delta_t,
                abs(self.ball_mass * np.dot(v_ij, _i))) # missing 2 factor?
        v_i_1 = v_i - (J / self.ball_mass) * _i
        v_j_1 = v_j + (J / self.ball_mass) * _i
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        self._omega_i_1, self._omega_j_1 = self._omega_i, self._omega_j
