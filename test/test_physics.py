import os
import logging
_logger = logging.getLogger(__name__)
import numpy as np


from utils import gen_filename, git_head_hash
from poolvr.physics.events import PhysicsEvent, CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent#, BallCollisionEvent


_here = os.path.dirname(__file__)


def test_occlusion(pool_physics, request):
    import matplotlib.pyplot as plt
    assert (pool_physics._occ_ij == ~np.array([[0,1,1,1,1,1,1,0,0,0,1,0,0,1,0,1],
                                               [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                               [1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0],
                                               [1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0],
                                               [1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0],
                                               [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
                                               [1,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0],
                                               [0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0],
                                               [0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0],
                                               [0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0],
                                               [1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0],
                                               [0,0,0,0,0,0,0,1,1,0,1,0,1,1,1,0],
                                               [0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0],
                                               [1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1],
                                               [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
                                               [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0]], dtype=np.bool)).all()
    plt.imshow(pool_physics._occ_ij)
    if request.config.getoption('--show_plots'):
        plt.show()
    filename = os.path.join(os.path.dirname(__file__), 'plots', 'test_occlusion.png')
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(filename)
    _logger.info('saved plot to "%s"', filename)
    plt.close()


def test_strike_ball(pool_physics,
                     plot_motion_z_position,
                     plot_motion_timelapse,
                     plot_energy):
    physics = pool_physics
    physics.reset(balls_on_table=[0])
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.zeros(3, dtype=np.float64)
    V[2] = -0.6
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))
    assert 4 == len(events)
    assert isinstance(events[0], CueStrikeEvent)
    assert isinstance(events[1], BallSlidingEvent)
    assert isinstance(events[2], BallRollingEvent)
    assert isinstance(events[3], BallRestEvent)


def test_ball_collision(pool_physics,
                        plot_motion_z_position,
                        plot_motion_timelapse,
                        plot_energy,
                        gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    start_event = BallSlidingEvent(0, 0, r_0=ball_positions[0],
                                   v_0=np.array((0.0, 0.0, -0.6)),
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallRollingEvent)
    # assert isinstance(events[2], BallCollisionEvent)
    # assert isinstance(events[3], BallRestEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


def test_angled_ball_collision(pool_physics,
                               plot_motion_z_position,
                               plot_motion_x_position,
                               plot_motion_timelapse,
                               plot_energy,
                               gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[1] = ball_positions[0]
    ball_positions[1,0] -= 8 / np.sqrt(2) * physics.ball_radius
    ball_positions[1,2] -= 8 / np.sqrt(2) * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    r_ij = ball_positions[1] - ball_positions[0]
    r_ij[0] += pool_physics.ball_radius
    e_ij = r_ij / np.linalg.norm(r_ij)
    v_0 = 0.9 * e_ij
    start_event = BallSlidingEvent(0, 0, r_0=ball_positions[0],
                                   v_0=v_0,
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallRollingEvent)
    # assert isinstance(events[2], BallCollisionEvent)
    # assert isinstance(events[3], BallRestEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


def test_sliding_ball_collision(pool_physics,
                                plot_motion_z_position,
                                plot_motion_timelapse,
                                plot_energy,
                                gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    start_event = BallSlidingEvent(0, 0, r_0=ball_positions[0],
                                   v_0=np.array((0.0, 0.0, -2.0)),
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallCollisionEvent)
    # assert isinstance(events[2], BallRestEvent)
    # assert isinstance(events[3], BallSlidingEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


def test_break(pool_physics,
               plot_motion_timelapse,
               plot_energy,
               gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0.0, -1.6), dtype=np.float64)
    M = 0.54
    import cProfile
    import time
    outname = gen_filename('test_break.%s' % git_head_hash(), 'pstats', directory=_here)
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    t1 = time.time()
    pr.dump_stats(outname)
    _logger.info('...dumped stats to "%s"', outname)
    _logger.info('elapsed time: %s', t1-t0)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))


def test_break_and_following_shot(pool_physics, gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0, -1.6), dtype=np.float64)
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike #1 on %d resulted in %d events:\n\n%s\n',
                  0, len(events), PhysicsEvent.events_str(events))
    ntt = physics.next_turn_time
    ball_positions = physics.eval_positions(ntt)
    r_02 = ball_positions[2] - ball_positions[0]
    r_02_mag = np.sqrt(np.dot(r_02, r_02))
    n_02 = r_02 / r_02_mag
    r_c = ball_positions[0] - physics.ball_radius * n_02
    V = 0.99 * n_02
    events = physics.strike_ball(ntt, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike #2 on %d resulted in %d events:\n\n%s\n',
                  0, len(events), PhysicsEvent.events_str(events))


def test_strike_ball_english(pool_physics, gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    sy = np.sin(45*np.pi/180)
    cy = np.cos(45*np.pi/180)
    sxz = np.sin(80*np.pi/180)
    cxz = np.cos(80*np.pi/180)
    r_c[1] += physics.ball_radius * sy
    r_c[0] += physics.ball_radius * cy * sxz
    r_c[2] += physics.ball_radius * cy * cxz
    V = np.zeros(3, dtype=np.float64)
    V[2] = -1.5
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))


def test_strike_ball_less_english(pool_physics, gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    sy = np.sin(40*np.pi/180)
    cy = np.cos(40*np.pi/180)
    sxz = np.sin(30*np.pi/180)
    cxz = np.cos(30*np.pi/180)
    r_c[1] += physics.ball_radius * sy
    r_c[0] += physics.ball_radius * cy * sxz
    r_c[2] += physics.ball_radius * cy * cxz
    V = np.zeros(3, dtype=np.float64)
    V[2] = -1.5
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))
