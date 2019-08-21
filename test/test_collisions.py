import logging
_logger = logging.getLogger(__name__)
import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi


def plot_collision(deltaPs, v_is, v_js, omega_is, omega_js):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(deltaPs, np.array(v_is)[:,1], label='ball i')
    plt.plot(deltaPs, np.array(v_js)[:,1], label='ball j')
    plt.xlabel('$P_I$: cumulative impulse along y-axis')
    plt.ylabel('velocity along y-axis')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(deltaPs, np.array(omega_is)[:,0], label='ball i')
    plt.plot(deltaPs, np.array(omega_js)[:,0], label='ball j')
    plt.xlabel('$P_I$: cumulative impulse along y-axis')
    plt.ylabel('angular velocity along x-axis')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(deltaPs, np.array(omega_is)[:,2], label='ball i')
    plt.plot(deltaPs, np.array(omega_js)[:,2], label='ball j')
    plt.xlabel('$P_I$: cumulative impulse along y-axis')
    plt.ylabel('angular velocity along z-axis')
    plt.legend()
    plt.show()


def test_collide_balls():
    """Reproduce results of Mathavan et al, 2014 - Table 1"""
    e = 0.89
    mu_s = 0.21
    mu_b = 0.05
    M = 0.1406
    R = 0.02625
    r_i = np.zeros(3)
    rd = np.array([1.0, 0.0, 0.0])
    r_c = np.array([R, 0.0, 0.0])
    r_j = r_i + 2 * R * rd
    v_j = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    expected = [
        (0.914, 0.831),
        (0.520, 0.599),
        (0.917, 0.676),
        (0.128, 0.780), # error in published table?
        (0.383, 0.579)
    ]
    for i_cond, (cue_ball_velocity, topspin, cut_angle) in enumerate([
        (1.539, 58.63, 33.83),
        (1.032, 39.31, 26.36),
        (1.364, 51.96, 40.52),
        (1.731, 65.94, 46.5),
        (0.942, 35.89, 18.05)
    ]):
        c, s = np.cos(cut_angle*DEG2RAD), np.sin(cut_angle*DEG2RAD)
        v_i = np.array([cue_ball_velocity*c, 0.0, cue_ball_velocity*s])
        omega_i = np.array([topspin*s, 0.0, -topspin*c])
        v_is, omega_is, v_js, omega_js, deltaPs = collide_balls(r_c,
                                                                r_i, v_i, omega_i,
                                                                r_j, v_j, omega_j,
                                                                e, mu_s, mu_b,
                                                                M, R,
                                                                9.81,
                                                                4000, return_all=True)
        v_i1 = v_is[-1]; v_j1 = v_js[-1]; omega_i1 = omega_is[-1]; omega_j1 = omega_js[-1]
        # plot_collision(deltaPs, v_is, v_js, omega_is, omega_js)
        from poolvr.physics.events import PhysicsEvent, BallSlidingEvent
        PhysicsEvent.ball_radius = R
        PhysicsEvent.ball_diameter = 2*R
        PhysicsEvent.ball_mass = M
        PhysicsEvent.ball_I = 2.0/5 * PhysicsEvent.ball_mass * PhysicsEvent.ball_radius**2
        PhysicsEvent.mu_s = mu_s
        PhysicsEvent.mu_b = mu_b
        e_i, e_j = BallSlidingEvent(0.0, 0, r_i, v_i1, omega_i1), BallSlidingEvent(0.0, 1, r_j, v_j1, omega_j1)
        v_is = e_i.next_motion_event.eval_velocity(0.0)
        v_js = e_j.next_motion_event.eval_velocity(0.0)
        _logger.info('''
        |v_i| = %s
        topspin = %s
        cut_angle = %s
        v_is = %s
        v_js = %s
        |v_is| = %s
        |v_js| = %s
        abs(|v_is| - predicted) = %s
        abs(|v_js| - predicted) = %s
        arctan2(*v_is[::-2]) = %s
        arctan2(*v_js[::-2]) = %s
        ''',
                     np.sqrt(np.dot(v_i, v_i)), topspin, cut_angle,
                     v_is, v_js,
                     np.sqrt(np.dot(v_is, v_is)), np.sqrt(np.dot(v_js, v_js)),
                     abs(np.sqrt(np.dot(v_is, v_is)) - expected[i_cond][0]),
                     abs(np.sqrt(np.dot(v_js, v_js)) - expected[i_cond][1]),
                     np.arctan2(*v_is[::-2])*RAD2DEG, np.arctan2(*v_js[::-2])*RAD2DEG)
