import os.path as path
from os import makedirs
import logging
_logger = logging.getLogger(__name__)
import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
PLOTS_DIR = path.join(path.dirname(__file__), 'plots')


def plot_collision_velocities(deltaPs, v_is, v_js,
                              title='velocities along axis of impulse',
                              show=True, filename=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(deltaPs, np.array(v_is)[:,0], label='ball i')
    plt.plot(deltaPs, np.array(v_js)[:,0], label='ball j')
    plt.xlabel(r'$P_I$: cumulative impulse along axis of impulse')
    plt.ylabel(r'$v_y$: velocity along axis of impulse')
    plt.title(title)
    plt.legend()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


def plot_collision_angular_velocities(deltaPs, omega_is, omega_js,
                                      title='angular velocities within horizontal plane',
                                      show=True, filename=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(deltaPs, np.array(omega_is)[:,0], label='ball i (axis of impulse)')
    plt.plot(deltaPs, np.array(omega_js)[:,0], label='ball j (axis of impulse)')
    plt.plot(deltaPs, np.array(omega_is)[:,2], '--', label='ball i (perpendicular axis)')
    plt.plot(deltaPs, np.array(omega_js)[:,2], '--', label='ball j (perpendicular axis)')
    plt.xlabel('$P_I$: cumulative impulse')
    plt.ylabel(r'$\omega_{xy}$: angular velocity within horizontal plane')
    plt.title(title)
    plt.legend()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


def test_collide_balls(request):
    """Reproduce results of Mathavan et al, 2014 - Table 1"""
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    e = 0.89
    mu_s = 0.21
    mu_b = 0.05
    M = 0.1406
    R = 0.02625
    r_i = np.zeros(3)
    rd = np.array([1.0, 0.0, 0.0])
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
        deltaP = (1 + e) * M * cue_ball_velocity / 8000
        v_is, omega_is, v_js, omega_js = collide_balls(r_i, v_i, omega_i,
                                                       r_j, v_j, omega_j,
                                                       e, mu_s, mu_b,
                                                       M, R,
                                                       deltaP=deltaP,
                                                       return_all=show_plots or save_plots)
        if show_plots or save_plots:
            test_name = request.function.__name__
            deltaPs = deltaP*np.arange(len(v_is))
            plot_collision_velocities(deltaPs, v_is, v_js, show=show_plots,
                                      filename=path.join(PLOTS_DIR, test_name + '-velocities.png') if save_plots else None)
            plot_collision_angular_velocities(deltaPs, omega_is, omega_js, show=show_plots,
                                              filename=path.join(PLOTS_DIR, test_name + '-angular-velocities.png') if save_plots else None)
            v_i1, omega_i1, v_j1, omega_j1 = v_is[-1], omega_is[-1], v_js[-1], omega_js[-1]
        else:
            v_i1, omega_i1, v_j1, omega_j1 = v_is, omega_is, v_js, omega_js
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
        lambda_i = np.arctan(v_iS[2]/v_iS[0])*RAD2DEG
        lambda_j = np.arctan(v_jS[2]/v_jS[0])*RAD2DEG
        theta_i = abs(lambda_i - cut_angle)
        theta_j = abs(lambda_j - cut_angle)
        v_iS_mag_ex, v_jS_mag_ex, theta_i_ex, theta_j_ex = expected[i_cond]
        v_0 = np.sqrt(np.dot(v_i, v_i))
        _logger.info('''
    |v_0|: %s
    topspin: %s
    cut_angle: %s

                 |v_iS| = %s
        expected |v_iS| = %s
                 |v_jS| = %s
        expected |v_jS| = %s
        ----------------------------------------
        abs(|v_iS| - expected) / |expected| = %s
        abs(|v_jS| - expected) / |expected| = %s

                 theta_i = %s
        expected theta_i = %s
                 theta_j = %s
        expected theta_j = %s
        -----------------------------------------
        abs(theta_i - expected) / |expected| = %s
        abs(theta_j - expected) / |expected| = %s
        ''',
                     v_0, topspin, cut_angle,
                     v_iS_mag, v_iS_mag_ex,
                     v_jS_mag, v_jS_mag_ex,
                     abs(v_iS_mag - v_iS_mag_ex)/abs(v_iS_mag_ex),
                     abs(v_jS_mag - v_jS_mag_ex)/abs(v_jS_mag_ex),
                     theta_i, theta_i_ex,
                     theta_j, theta_j_ex,
                     abs(theta_i - theta_i_ex)/abs(theta_i_ex),
                     abs(theta_j - theta_j_ex)/abs(theta_j_ex))
        assert abs(v_iS_mag - v_iS_mag_ex)/abs(v_iS_mag_ex) < 1e-2
        assert abs(v_jS_mag - v_jS_mag_ex)/abs(v_jS_mag_ex) < 1e-2
        assert abs(theta_i - theta_i_ex)/abs(theta_i_ex) < 1e-2
        assert abs(theta_j - theta_j_ex)/abs(theta_j_ex) < 1e-2


def test_collide_balls_f90(request):
    from poolvr.physics.collisions import collide_balls_f90
    e = 0.89
    mu_s = 0.21
    mu_b = 0.05
    M = 0.1406
    R = 0.02625
    r_i = np.zeros(3)
    rd = np.array([1.0, 0.0, 0.0])
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
        deltaP = (1 + e) * M * cue_ball_velocity / 8000
        v_is, omega_is, v_js, omega_js = collide_balls_f90(r_i, v_i, omega_i,
                                                           r_j, v_j, omega_j,
                                                           e=e, mu_s=mu_s, mu_b=mu_b,
                                                           M=M, R=R,
                                                           deltaP=deltaP)
        v_i1, omega_i1, v_j1, omega_j1 = v_is, omega_is, v_js, omega_js
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
        lambda_i = np.arctan(v_iS[2]/v_iS[0])*RAD2DEG
        lambda_j = np.arctan(v_jS[2]/v_jS[0])*RAD2DEG
        theta_i = abs(lambda_i - cut_angle)
        theta_j = abs(lambda_j - cut_angle)
        v_iS_mag_ex, v_jS_mag_ex, theta_i_ex, theta_j_ex = expected[i_cond]
        v_0 = np.sqrt(np.dot(v_i, v_i))
        _logger.info('''
    |v_0|: %s
    topspin: %s
    cut_angle: %s

                 |v_iS| = %s
        expected |v_iS| = %s
                 |v_jS| = %s
        expected |v_jS| = %s
        ----------------------------------------
        abs(|v_iS| - expected) / |expected| = %s
        abs(|v_jS| - expected) / |expected| = %s

                 theta_i = %s
        expected theta_i = %s
                 theta_j = %s
        expected theta_j = %s
        -----------------------------------------
        abs(theta_i - expected) / |expected| = %s
        abs(theta_j - expected) / |expected| = %s
        ''',
                     v_0, topspin, cut_angle,
                     v_iS_mag, v_iS_mag_ex,
                     v_jS_mag, v_jS_mag_ex,
                     abs(v_iS_mag - v_iS_mag_ex)/abs(v_iS_mag_ex),
                     abs(v_jS_mag - v_jS_mag_ex)/abs(v_jS_mag_ex),
                     theta_i, theta_i_ex,
                     theta_j, theta_j_ex,
                     abs(theta_i - theta_i_ex)/abs(theta_i_ex),
                     abs(theta_j - theta_j_ex)/abs(theta_j_ex))
        assert abs(v_iS_mag - v_iS_mag_ex)/abs(v_iS_mag_ex) < 1e-2
        assert abs(v_jS_mag - v_jS_mag_ex)/abs(v_jS_mag_ex) < 1e-2
        assert abs(theta_i - theta_i_ex)/abs(theta_i_ex) < 1e-2
        assert abs(theta_j - theta_j_ex)/abs(theta_j_ex) < 1e-2
