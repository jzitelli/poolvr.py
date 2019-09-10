import os.path as path
from cProfile import Profile
from time import perf_counter
import logging
_logger = logging.getLogger(__name__)
import numpy as np
import pytest


from utils import (plot_collision_velocities, plot_collision_angular_velocities,
                   plot_collision_velocity_maps, plot_collision_angular_velocity_maps,
                   gen_filename, git_head_hash)
from poolvr.physics.collisions import collide_balls, collide_balls_f90


_here = path.dirname(__file__)
_k = np.array([0.0, 0.0, 1.0])
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
PLOTS_DIR = path.join(path.dirname(__file__), 'plots')


e = 0.89
mu_s = 0.21
mu_b = 0.05
M = 0.1406
R = 0.02625
MATHAVAN_INITIAL_CONDITIONS = [
    (1.539, 58.63, 33.83),
    (1.032, 39.31, 26.36),
    (1.364, 51.96, 40.52),
    (1.731, 65.94, 46.5),
    (0.942, 35.89, 18.05)
]
MATHAVAN_EXPECTED_VALUES = [
    # Table 1       # Table 2
    (0.914, 0.831,  31.93, 32.20),
    (0.520, 0.599,  32.45, 25.07),
    (0.917, 0.676,  29.91, 38.62),
    (1.28,  0.780,  27.32, 44.38),
    (0.383, 0.579,  29.47, 17.15)
]


@pytest.mark.parametrize("collide_func", [collide_balls, collide_balls_f90])
@pytest.mark.parametrize("initial_conditions,expected", zip(MATHAVAN_INITIAL_CONDITIONS, MATHAVAN_EXPECTED_VALUES))
def test_collide_balls(request, initial_conditions, expected, collide_func):
    """Reproduce results of Mathavan et al, 2014 - Table 1"""
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    r_i = np.zeros(3)
    rd = np.array([1.0, 0.0, 0.0])
    r_j = r_i + 2 * R * rd
    v_j = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    cue_ball_velocity, topspin, cut_angle = initial_conditions
    c, s = np.cos(cut_angle*DEG2RAD), np.sin(cut_angle*DEG2RAD)
    v_i, omega_i = np.array(((cue_ball_velocity*c, 0.0, cue_ball_velocity*s),
                             (          topspin*s, 0.0,          -topspin*c)))
    deltaP = (1 + e) * M * cue_ball_velocity / 8000
    outname = gen_filename('test_collide_balls.%s.%s' % (collide_func.__name__, git_head_hash()), 'pstats',
                           directory=path.join(_here, 'pstats'))
    pr = Profile(timer=perf_counter)
    pr.enable()
    t0 = perf_counter()
    v_is, omega_is, v_js, omega_js = collide_func(r_i, v_i, omega_i,
                                                  r_j, v_j, omega_j,
                                                  e, mu_s, mu_b,
                                                  M, R,
                                                  deltaP=deltaP,
                                                  return_all=show_plots or save_plots)
    t1 = perf_counter()
    pr.dump_stats(outname)
    _logger.info('...dumped stats to "%s"', outname)
    _logger.info('evaluation time: %s', t1-t0)
    if collide_func is not collide_balls_f90 and (show_plots or save_plots):
        deltaPs = deltaP*np.arange(len(v_is))
        plot_collision_velocities(deltaPs, v_is, v_js, show=show_plots,
                                  filename=path.join(PLOTS_DIR,
                                                     '%s.%s.velocities.png' % (request.function.__name__, collide_func.__name__))
                                  if save_plots else None)
        plot_collision_angular_velocities(deltaPs, omega_is, omega_js, show=show_plots,
                                          filename=path.join(PLOTS_DIR,
                                                             '%s.%s.angular-velocities.png' % (request.function.__name__, collide_func.__name__))
                                          if save_plots else None)
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
    _calc_errors((v_iS_mag, v_jS_mag, theta_i, theta_j), expected,
                 header='''
|v_0|: %s
topspin: %s
cut_angle: %s''' % initial_conditions)


def test_collision_map(request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    r_i = np.zeros(3)
    rd = np.array([1.0, 0.0, 0.0])
    r_j = r_i + 2 * R * rd
    v_j = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    velocities = np.linspace(1e-5, 50.0, 64)
    angles = np.linspace(0.01, 89.99, 64) * DEG2RAD
    cs = np.cos(angles)
    ss = np.sin(angles)
    v_i = np.zeros(3, dtype=np.float64)
    omega_i = np.zeros(3, dtype=np.float64)
    v_i1s = np.zeros((len(velocities), len(angles), 3), dtype=np.float64)
    v_j1s = np.zeros((len(velocities), len(angles), 3), dtype=np.float64)
    omega_i1s = np.zeros((len(velocities), len(angles), 3), dtype=np.float64)
    omega_j1s = np.zeros((len(velocities), len(angles), 3), dtype=np.float64)
    t0 = perf_counter()
    for ii, velocity in enumerate(velocities):
        for jj, (c, s) in enumerate(zip(cs, ss)):
            v_i[0] = velocity * c
            v_i[2] = velocity * s
            omega_i[0] =  velocity / R * s
            omega_i[2] = -velocity / R * c
            deltaP = (1 + e) * M * velocity * c / 8000
            v_i1s[ii,jj], omega_i1s[ii,jj], \
                v_j1s[ii,jj], omega_j1s[ii,jj] = collide_balls_f90(r_i, v_i, omega_i,
                                                                   r_j, v_j, omega_j,
                                                                   e=e, mu_s=mu_s, mu_b=mu_b,
                                                                   M=M, R=R,
                                                                   deltaP=deltaP)
    t1 = perf_counter()
    _logger.info('evaluation time: %s', t1-t0)
    if show_plots or save_plots:
        test_name = request.function.__name__
        plot_collision_velocity_maps(v_i1s, v_j1s,
                                     filename=path.join(PLOTS_DIR, test_name + '.velocities.png')
                                     if save_plots else None,
                                     show=show_plots)
        plot_collision_angular_velocity_maps(omega_i1s, omega_j1s,
                                             filename=path.join(PLOTS_DIR, test_name + '.angular_velocities.png')
                                             if save_plots else None,
                                             show=show_plots)


def _calc_errors(actual, expected, header=''):
    v_iS_mag, theta_i, v_jS_mag, theta_j = actual
    v_iS_mag_ex, theta_i_ex, v_jS_mag_ex, theta_j_ex = expected
    _logger.info('''%s

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
                 header,
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
