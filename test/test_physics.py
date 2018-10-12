import logging
_logger = logging.getLogger(__name__)
import numpy as np
import pytest


from poolvr.physics.events import PhysicsEvent, CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent


def test_occlusion(pool_physics):
    import matplotlib.pyplot as plt
    plt.matshow(pool_physics._occ_ij)
    plt.savefig('occ.png')
    plt.show()


@pytest.mark.parametrize("ball_collision_model", ['simple', 'marlow'])
def test_strike_ball(pool_physics, ball_collision_model,
                     plot_motion_z_position, plot_motion_timelapse):
    physics = pool_physics
    physics.reset(balls_on_table=[0])
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    _logger.info('r_c = %s', r_c)
    V = np.zeros(3, dtype=np.float64)
    V[2] = -0.6
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))
    assert 4 == len(events)
    assert isinstance(events[0], CueStrikeEvent)
    assert isinstance(events[1], BallSlidingEvent)
    assert isinstance(events[2], BallRollingEvent)
    assert isinstance(events[3], BallRestEvent)


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_ball_collision(pool_physics, ball_collision_model,
                        plot_motion_z_position, plot_motion_timelapse,
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
    _logger.info('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    assert 6 == len(events)
    assert isinstance(events[0], BallSlidingEvent)
    assert isinstance(events[1], BallRollingEvent)
    assert isinstance(events[2], BallCollisionEvent)
    assert isinstance(events[3], BallRestEvent)
    assert isinstance(events[4], BallRollingEvent)
    assert isinstance(events[5], BallRestEvent)


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_angled_ball_collision(pool_physics, ball_collision_model,
                               plot_motion_z_position, plot_motion_timelapse,
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
    _logger.info('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallRollingEvent)
    # assert isinstance(events[2], BallCollisionEvent)
    # assert isinstance(events[3], BallRestEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_sliding_ball_collision(pool_physics, ball_collision_model,
                                plot_motion_z_position, plot_motion_timelapse,
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
    _logger.info('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallCollisionEvent)
    # assert isinstance(events[2], BallRestEvent)
    # assert isinstance(events[3], BallSlidingEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


def test_break(pool_physics,
               plot_motion_z_position, plot_motion_timelapse,
               gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue = PoolCue()
    cue.velocity[2] = -1.6
    cue.velocity[0] = -0.01
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, cue.velocity, cue.mass)
    pr.dump_stats('test_break.pstats')
    pr.print_stats()
    _logger.info('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))


def test_break_and_following_shot(pool_physics, gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0, -1.6), dtype=np.float64)
    M = 0.54
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike #1 on %d resulted in %d events:\n\n%s\n',
                  0, len(events), PhysicsEvent.events_str(events))
    ntt = physics.next_turn_time
    ball_positions = physics.eval_positions(ntt)
    r_02 = ball_positions[2] - ball_positions[0]
    r_02_mag = np.sqrt(np.dot(r_02, r_02))
    n_02 = r_02 / r_02_mag
    r_c = ball_positions[0] - physics.ball_radius * n_02
    V = 0.99 * n_02
    events = physics.strike_ball(ntt, 0, ball_positions[0], r_c, V, M)
    pr.dump_stats('test_break_and_following_shot.pstats')
    pr.print_stats()
    _logger.info('strike #2 on %d resulted in %d events:\n\n%s\n',
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
    _logger.info('r_c = %s', r_c)
    V = np.zeros(3, dtype=np.float64)
    V[2] = -1.5
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
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
    _logger.info('r_c = %s', r_c)
    V = np.zeros(3, dtype=np.float64)
    V[2] = -1.5
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))
