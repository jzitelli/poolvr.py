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
    plt.ylabel('$v_y$: velocity along y-axis')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(deltaPs, np.array(omega_is)[:,0], label='ball i')
    plt.plot(deltaPs, np.array(omega_js)[:,0], label='ball j')
    plt.xlabel('$P_I$: cumulative impulse along y-axis')
    plt.ylabel('$\omega_x$: angular velocity along x-axis')
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
        # Table 1       # Table 2
        (0.914, 0.831,  31.93, 32.20),
        (0.520, 0.599,  32.45, 25.07),
        (0.917, 0.676,  29.91, 38.62),
        (1.28,  0.780,  27.32, 44.38),
        (0.383, 0.579,  29.47, 17.15)
    ]
    for i_cond, (cue_ball_velocity, topspin, cut_angle) in enumerate([
        (1.539, 58.63, 33.83),
        (1.032, 39.31, 26.36),
        (1.364, 51.96, 40.52),
        (1.731, 65.94, 46.5),
        (0.942, 35.89, 18.05)
    ]):
        c, s = np.cos(cut_angle*DEG2RAD), np.sin(cut_angle*DEG2RAD)
        v_i, omega_i = np.array(((cue_ball_velocity*c, 0.0, cue_ball_velocity*s),
                                 (          topspin*s, 0.0,          -topspin*c)))
        v_is, omega_is, v_js, omega_js, deltaPs = collide_balls(r_c,
                                                                r_i, v_i, omega_i,
                                                                r_j, v_j, omega_j,
                                                                e, mu_s, mu_b,
                                                                M, R,
                                                                9.81,
                                                                4000, return_all=True)
        v_i1 = v_is[-1]; omega_i1 = omega_is[-1]
        v_j1 = v_js[-1]; omega_j1 = omega_js[-1]
        from poolvr.physics.events import PhysicsEvent, BallSlidingEvent
        PhysicsEvent.ball_radius = R
        PhysicsEvent.ball_diameter = 2*R
        PhysicsEvent.ball_mass = M
        PhysicsEvent.ball_I = 2.0/5 * M * R**2
        PhysicsEvent.mu_s = mu_s
        PhysicsEvent.mu_b = mu_b
        e_i, e_j = BallSlidingEvent(0.0, 0, r_i, v_i1, omega_i1), \
                   BallSlidingEvent(0.0, 1, r_j, v_j1, omega_j1)
        v_iS = e_i.next_motion_event.eval_velocity(0.0)
        v_jS = e_j.next_motion_event.eval_velocity(0.0)
        v_iS_mag = np.sqrt(np.dot(v_iS, v_iS))
        v_jS_mag = np.sqrt(np.dot(v_jS, v_jS))
        theta_i  = np.arctan2(*v_i1[::-2])*RAD2DEG
        theta_j  = np.arctan2(*v_j1[::-2])*RAD2DEG
        lambda_i = np.arctan2(*v_iS[::-2])*RAD2DEG
        lambda_j = np.arctan2(*v_jS[::-2])*RAD2DEG
        psi_i = theta_i - lambda_i
        psi_j = theta_j - lambda_j + cut_angle
        v_0 = np.sqrt(np.dot(v_i, v_i))
        _logger.info('''
    |v_0|: %s
    topspin: %s
    cut_angle: %s

        v_iS = %s
        v_jS = %s
        |v_iS| = %s
        |v_jS| = %s
        abs(|v_iS| - expected) / |expected| = %s
        abs(|v_jS| - expected) / |expected| = %s

        exit angle i = %s
        exit angle j = %s
        abs(exit angle i - expected) / |expected| = %s
        abs(exit angle j - expected) / |expected| = %s
        ''',
                     v_0, topspin, cut_angle,
                     v_iS, v_jS, v_iS_mag, v_jS_mag,
                     abs(v_iS_mag - expected[i_cond][0])/abs(expected[i_cond][0]),
                     abs(v_jS_mag - expected[i_cond][1])/abs(expected[i_cond][1]),
                     psi_i, psi_j,
                     abs(expected[i_cond][2] - psi_i)/abs(expected[i_cond][2]),
                     abs(expected[i_cond][3] - psi_j)/abs(expected[i_cond][3]))
