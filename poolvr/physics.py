from collections import deque
import numpy as np


INCH2METER = 0.0254


class PoolPhysics(object):

    class Event(object):
        # event type enumerations:
        CUE_STRIKE = 999
        SLIDE2ROLL = 1000
        ROLL2SPIN = 1001
        ROLL2REST  = 1002
        SPIN2REST = 1003
        BALLCOLLISION = 1004
        RAILCOLLISION = 1005
        POCKETCOLLISION = 1006
        def __init__(self, t):
            self.t = t
        def predict_events(self, events=None,
                           **kwargs):
            if events is None:
                events = [self]
            elif events and self.t <= events[-1].t:
                events.push(self)
            return events

    class BallSlideEvent(Event):
        def __init__(self, t, i, v, omega):
            Event.__init__(self, t)
            self.i = i
            self.q = q
            self.v = v
            self.omega = omega
        def predict_events(self, ball_states, events=None,
                           ball_radius=1.125*INCH2METER,
                           mu_s=0.2, g=0.81,
                           **kwargs):
            events = Event.predict_events(self, events=events, **kwargs)
            # duration of slide:
            R = ball_radius
            u0_x = -R * self.omega[2]
            u0_x =  R * self.omega[0]
            tau_s = 2.0 * np.sqrt(u0_x**2 + u0_z**2) / (7.0 * mu_s * g)
            # slide end time:
            t_s = self.t + tau_s
            if t_s < events[-1].t:
                ball_roll_event = BallRollEvent(self.t + tau_s, self.i, v)
                events.push(ball_roll_event)
            # update trajectory coeffecients:
            self.ball_traject_sin[i] = V[2] / V_xz
            self.ball_traject_cos[i] = V[0] / V_xz
            # self.ball_traject_coeffs[i,0,0] = self.ball_positions[i,0]
            # self.ball_traject_coeffs[i,0,1] = self.ball_positions[i,2]
            self.ball_traject_coeffs[i,1,0] = v_xz
            self.ball_traject_coeffs[i,2,0] = -0.5 * self.mu_s * self.g * u_0x
            self.ball_traject_coeffs[i,2,1] = -0.5 * self.mu_s * self.g * u_0z
            # event prediction loop:
            sliders = [i for i, sliding in enumerate(self.is_sliding) if sliding]
            rollers = [i for i, rolling in enumerate(self.is_rolling) if rolling]
            while sliders or rollers:
                t_last = self.event_times[-1]
                t_next = float('inf')
                predicted = None
                for i in sliders:
                    omega = self.ball_angular_velocities[i]
                    u0_x, u0_z = -R * omega[2], R * omega[0]
                    tau_s = 2.0 * np.sqrt(u0_x**2 + u0_z**2) / (7.0 * self.mu_s * self.g)
                    if t_last + tau_s < t_next:
                        t_next = t_last + tau_s
                        predicted = (self.SLIDE2ROLL, i)
                for i in rollers:
                    v = self.ball_velocities[i]
                    tau_r = np.linalg.norm(v[::2]) / (self.mu_r * self.g)
                    if t_last + tau_r < t_next:
                        t_next = t_last + tau_r
                        predicted = (self.ROLL2REST, i)
            # record the determined event:
            self.events.append(predicted)
            self.event_times.append(t_next)
            # advance all balls forward in time to the determined event:
            t = t_next - t_last
            rx = self.ball_traject_coeffs[:,2,0] * t**2 + self.ball_traject_coeffs[:,1,0] * t
            rz = self.ball_traject_coeffs[:,2,1] * t**2
            self.ball_positions[:,0] += self.ball_traject_cos * rx - self.ball_traject_sin * rz
            self.ball_positions[:,2] += self.ball_traject_sin * rx + self.ball_traject_cos * rz
            vx = 2.0 * self.ball_traject_coeffs[:,2,0] * t
            vz = 2.0 * self.ball_traject_coeffs[:,2,1] * t
            self.ball_velocities[:,0] += self.ball_traject_cos * vx - self.ball_traject_sin * vz
            self.ball_velocities[:,2] += self.ball_traject_sin * vx + self.ball_traject_cos * vz
            event_type, event_info = predicted
            if event_type == self.SLIDE2ROLL:
                i = event_info
                self.is_sliding[i] = False
                self.is_rolling[i] = True
                self.ball_traject_coeffs[i,2] = -0.5 * self.mu_r * self.g * self.ball_velocities[i,::2]
                self.ball_traject_coeffs[i,1] = self.ball_velocities[i,::2]
                self.ball_traject_sin[i] = self.ball_velocities[i,2] / np.sqrt(self.ball_velocities[i,0]**2 + self.ball_velocities[i,2]**2)
                self.ball_traject_cos[i] = self.ball_velocities[i,0] / np.sqrt(self.ball_velocities[i,0]**2 + self.ball_velocities[i,2]**2)
                # angular velocity?
            elif event_type == self.ROLL2REST:
                i = event_info
                self.is_rolling[i] = False
                self.ball_traject_coeffs[i] = 0
                self.ball_velocities[i] = 0
                self.ball_angular_velocities[i] = 0
                self.ball_traject_sin[i] = 0
                self.ball_traject_cos[i] = 0
            return events

    class StrikeBallEvent(BallSlideEvent):
        def __init__(self, t, i, q, v, omega,
                     cue_mass=0.54,
                     ball_mass=0.17,
                     ball_radius=1.125*INCH2METER):
            self.q = q
            a, c, b = q
            M, m, R = cue_mass, ball_mass, ball_radius
            v_xz = v[::2]
            norm_v = np.linalg.norm(v)
            norm_v_xz = np.linalg.norm(v_xz)
            sin, cos = abs(v_xz) / norm_v_xz
            F = 2.0*m*norm_v / (1.0 + m/M + 5.0/(2.0*R**2) * (a**2 + (b*cos)**2 + (c*sin)**2 - 2*b*c*cos*sin))
            # linear velocity:
            norm_v = -F / m * cos
            v[0] *= norm_v / norm_v_xz
            v[2] *= norm_v / norm_v_xz
            #v[1] = -F / m * sin
            v[1] = 0.0
            # angular velocity:
            I = 2.0/5.0 * m * R**2
            omega[0] = F * (-c * sin + b * cos) / I
            omega[2] = F * a * sin / I
            omega[1] = -F * a * cos / I
            BallSlideEvent.__init__(self, t, i, v, omega)

    class BallRollEvent(Event):
        def __init__(self, t, i, v):
            Event.__init__(self, t)
            self.i = i
            self.v = v
        def predict_events(self, table_state, events=None, **kwargs):
            events = Event.predict_events(self, events=events, **kwargs)
            return events

    def __init__(self, num_balls=16):
        self.t = 0.0

    def __init__(self,
                 num_balls=16,
                 ball_mass=0.17,
                 ball_radius=1.125*INCH2METER,
                 mu_r=0.016,
                 mu_sp=0.044,
                 mu_s=0.2,
                 e=0.93,
                 g=9.81,
                 **kwargs):
        self.num_balls = num_balls
        self._positions = np.empty((num_balls, 3), dtype=np.float32)
        self._quaternions = np.zeros((self.num_balls, 4))
        self._velocities = np.empty((num_balls, 3), dtype=np.float32)
        self._angular_velocities = np.empty((num_balls, 3), dtype=np.float32)
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        ball_diameter = 2 * ball_radius
        self.mu_r = mu_r
        self.mu_sp = mu_sp
        self.mu_s = mu_s
        self.e = e
        self.g = g
        self.events = deque()
        self.t = 0.0
        self.nevent = 0
        # state of balls:
        self.on_table = np.array(self.num_balls * [True])
        self.is_sliding = np.array(self.num_balls * [False])
        self.is_rolling = np.array(self.num_balls * [False])
        # equations of motion coefficients:
        self.ball_traject_coeffs = np.zeros((self.num_balls, 3, 2))
        self.ball_omega_coeffs = np.zeros(self.num_balls)
        self.ball_traject_sin = np.zeros(self.num_balls)
        self.ball_traject_cos = np.zeros(self.num_balls)
        self.nevent = 0
    def step(self, dt):
        pass
    def strike_ball(i, M, q, v, omega):
        pass
    def eval_positions(self, t):
        pass
    def eval_velocities(self, t):
        pass
    def eval_angular_velocities(self, t):
        pass
