import pytest
import os
from cProfile import Profile
import logging
_logger = logging.getLogger(__name__)
import numpy as np


from utils import gen_filename, git_head_hash, check_ball_distances
from poolvr.physics.events import (PhysicsEvent,
                                   CueStrikeEvent,
                                   BallSlidingEvent,
                                   BallRollingEvent,
                                   BallRestEvent,
                                   CornerCollisionEvent,
                                   BallSpinningEvent,
                                   BallCollisionEvent,
                                   SegmentCollisionEvent,
                                   BallPocketedEvent)


_here = os.path.dirname(__file__)
DEG2RAD = np.pi/180


def test_strike_ball(pool_physics,
                     plot_motion_timelapse,
                     plot_energy,
                     gl_rendering):
    physics = pool_physics
    physics.reset(balls_on_table=[0])
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0]
    r_c[2] += physics.ball_radius
    V = np.array((0, 0, -0.6), dtype=np.float64)
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  PhysicsEvent.events_str(events))
    assert 4 == len(events)
    assert isinstance(events[0], CueStrikeEvent)
    assert isinstance(events[1], BallSlidingEvent)
    assert isinstance(events[2], BallRollingEvent)
    assert isinstance(events[3], BallRestEvent)


def test_initially_stationary_sliding_ball(pool_physics,
                                           gl_rendering,
                                           plot_energy):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    R = physics.ball_radius
    physics.reset(balls_on_table=[0])
    omega_0 = np.zeros(3, dtype=np.float64)
    omega_0[0] = -1.0/R
    start_event = BallSlidingEvent(0, 0,
                                   r_0=ball_positions[0],
                                   v_0=np.zeros(3, dtype=np.float64),
                                   omega_0=omega_0)
    events = physics.add_event_sequence(start_event)
    _logger.info('%d events added:\n\n%s\n', len(events),
                 PhysicsEvent.events_str(events=events))
    assert 3 == len(events)
    assert isinstance(events[0], BallSlidingEvent)
    assert isinstance(events[1], BallRollingEvent)
    assert isinstance(events[2], BallRestEvent)


def test_ball_collision(pool_physics,
                        plot_motion_timelapse,
                        plot_energy,
                        gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[1] = ball_positions[0]
    ball_positions[1,2] -= 8 * physics.ball_radius
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
    r_ij[0] += physics.ball_radius
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
                                plot_motion_timelapse,
                                plot_energy,
                                gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[1] = ball_positions[0]
    ball_positions[1,2] -= 8 * physics.ball_radius
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
               gl_rendering,
               request):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0.0, -1.6), dtype=np.float64)
    M = 0.54
    outname = gen_filename('test_break.%s.%s' % (physics.ball_collision_model, git_head_hash()),
                           'pstats',
                           directory=os.path.join(_here, 'pstats'))
    from time import perf_counter
    pr = Profile()
    pr.enable()
    t0 = perf_counter()
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    t1 = perf_counter()
    pr.dump_stats(outname)
    _logger.info('...dumped stats to "%s"', outname)
    _logger.info('evaluation time: %s', t1-t0)
    _logger.info('\n'.join(['strike on %d resulted in %d events:',
                            '  %d BallSlidingEvents',
                            '  %d BallRollingEvents',
                            '  %d BallSpinningEvents',
                            '  %d BallRestEvents',
                            '  %d SegmentCollisionEvents',
                            '  %d CornerCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, SegmentCollisionEvent)]),
                 len([e for e in events if isinstance(e, CornerCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


def test_break_hard(pool_physics,
                    plot_motion_timelapse,
                    plot_energy,
                    gl_rendering,
                    request):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    R = physics.ball_radius
    r_c = ball_positions[0].copy()
    r_c[1] += 2/5 * R
    r_c[2] += np.sqrt(R**2 - (2/5*R)**2)
    V = np.array((-0.02, 0.0, -4.2), dtype=np.float64)
    M = 0.54
    outname = gen_filename('test_break_hard.%s.%s' % (physics.ball_collision_model, git_head_hash()),
                           'pstats',
                           directory=os.path.join(_here, 'pstats'))
    from time import perf_counter
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    t0 = perf_counter()
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    t1 = perf_counter()
    pr.dump_stats(outname)
    _logger.info('evaluation time: %s', t1-t0)
    _logger.info('...dumped stats to "%s"', outname)
    _logger.info('\n'.join(['strike on %d resulted in %d events:',
                            '  %d BallSlidingEvents',
                            '  %d BallRollingEvents',
                            '  %d BallSpinningEvents',
                            '  %d BallRestEvents',
                            '  %d SegmentCollisionEvents',
                            '  %d CornerCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, SegmentCollisionEvent)]),
                 len([e for e in events if isinstance(e, CornerCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


@pytest.mark.skip
def test_break_hard_realtime(pool_physics_realtime,
                             plot_motion_timelapse,
                             plot_energy,
                             gl_rendering,
                             request):
    physics = pool_physics_realtime
    nevents = len(physics.events)
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += 0.5 * np.sqrt(2.0) * physics.ball_radius
    r_c[1] += 0.5 * np.sqrt(2.0) * physics.ball_radius
    V = np.array((-0.006, 0.0, -3.4), dtype=np.float64)
    M = 0.54
    outname = gen_filename('test_break_hard_realtime.%s.%s' % (physics.ball_collision_model, git_head_hash()),
                           'pstats',
                           directory=os.path.join(_here, 'pstats'))
    from time import perf_counter
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    t0 = perf_counter()
    physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    lt = perf_counter()
    while physics._ball_motion_events or physics._ball_spinning_events:
        t = perf_counter()
        dt = t - lt
        lt = t
        physics.step(dt)
    t1 = perf_counter()
    pr.dump_stats(outname)
    _logger.info('evaluation time: %s', t1-t0)
    _logger.info('...dumped stats to "%s"', outname)
    events = physics.events[nevents:]
    _logger.info('\n'.join(['strike on %d resulted in %d events:',
                            '  %d BallSlidingEvents',
                            '  %d BallRollingEvents',
                            '  %d BallSpinningEvents',
                            '  %d BallRestEvents',
                            '  %d SegmentCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, SegmentCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)


def test_break_and_following_shot(pool_physics,
                                  gl_rendering,
                                  request):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_c = ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    V = np.array((-0.01, 0, -1.6), dtype=np.float64)
    M = 0.54
    events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike #1 on %d resulted in %d events', 0, len(events))
    # _logger.debug('strike #1 on %d resulted in %d events:\n\n%s\n',
    #               0, len(events), PhysicsEvent.events_str(events))
    ntt = physics.balls_at_rest_time
    ball_positions = physics.eval_positions(ntt)
    r_02 = ball_positions[2] - ball_positions[0]
    r_02_mag = np.sqrt(np.dot(r_02, r_02))
    n_02 = r_02 / r_02_mag
    r_c = ball_positions[0] - physics.ball_radius * n_02
    V = 0.99 * n_02
    events = physics.strike_ball(ntt, 0, ball_positions[0], r_c, V, M)
    _logger.info('strike #2 on %d resulted in %d events', 0, len(events))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike #2 on %d resulted in %d events:\n\n%s\n',
    #               0, len(events), PhysicsEvent.events_str(events))


def test_strike_ball_english(pool_physics,
                             gl_rendering,
                             plot_motion_timelapse,
                             request):
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
    _logger.info('strike on %d resulted in %d events', 0, len(events))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


def test_strike_ball_less_english(pool_physics,
                                  gl_rendering,
                                  plot_motion_timelapse,
                                  request):
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
    _logger.info('strike on %d resulted in %d events', 0, len(events))
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


inner_corners = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]


@pytest.mark.parametrize("i_c", inner_corners)
def test_corner_collision(pool_physics,
                          gl_rendering,
                          plot_energy,
                          i_c):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    if i_c not in inner_corners:
        return
    i_a, i_b = physics._corner_to_segments[i_c]
    seg_a = physics._segments[i_a]
    seg_b = physics._segments[i_b]
    n_a, n_b = seg_a[2], seg_b[2]
    n = n_a + n_b
    n /= np.linalg.norm(n)
    r_c = physics._corners[i_c]
    R = physics.ball_radius
    ball_positions[0] = r_c + 4*R*n
    physics.reset(balls_on_table=[0],
                  ball_positions=ball_positions)
    v_0 = -n * 1
    start_event = BallSlidingEvent(0, 0,
                                   r_0=ball_positions[0],
                                   v_0=v_0,
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.info('%d events added:\n\n%s\n', len(events),
                 PhysicsEvent.events_str(events=events))
    assert any(isinstance(e, CornerCollisionEvent) for e in events)
    cce = next(e for e in events if isinstance(e, CornerCollisionEvent))
    v_1 = cce.child_events[0].eval_velocity(0.0)
    # expect to rebound heading in the exact opposite direction:
    assert 0.9999 < abs(np.dot(v_1, n)) / np.linalg.norm(v_1)


@pytest.mark.parametrize("i_p", list(range(6)))
def test_pocket(pool_physics, i_p,
                gl_rendering):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    r_p = physics._pocket_positions[i_p]
    # start from table center, aimed directly at the pocket:
    ball_positions[0] = np.array([0.0, r_p[1], 0.0])
    direction = r_p - ball_positions[0]
    direction[1] = 0
    direction /= np.linalg.norm(direction)
    physics.reset(balls_on_table=[0],
                  ball_positions=ball_positions)
    v_0 = direction * 2.0
    start_event = BallSlidingEvent(0, 0,
                                   r_0=ball_positions[0],
                                   v_0=v_0,
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.info('%d events added:\n\n%s\n', len(events),
                 PhysicsEvent.events_str(events=events))
    assert any(isinstance(e, BallPocketedEvent) for e in events), \
        "expected BallPocketedEvent for pocket %d" % i_p
    bpe = next(e for e in events if isinstance(e, BallPocketedEvent))
    assert bpe.i_p == i_p
    assert bpe.i == 0
    assert not physics._on_table[0]


@pytest.mark.parametrize("i_seg", list(range(18)))
def test_segment_collision(pool_physics, gl_rendering, request, i_seg):
    physics = pool_physics
    R = physics.ball_radius
    ball_positions = physics.eval_positions(0.0)
    physics.reset(balls_on_table=[0],
                  ball_positions=ball_positions)
    segment = physics._segments[i_seg]
    r_a, r_b, nor, tan = segment
    ball_positions[0] = 0.5*(r_a + r_b) + 1.5*R*nor
    physics.reset(ball_positions=ball_positions, balls_on_table=[0])
    v_0 = -0.4 * nor
    start_event = BallSlidingEvent(0, 0,
                                   r_0=ball_positions[0],
                                   v_0=v_0,
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events),
                  PhysicsEvent.events_str(events=events))
    assert any(isinstance(e, SegmentCollisionEvent) for e in events)
    sce = next(e for e in events if isinstance(e, SegmentCollisionEvent))
    v_1 = sce.child_events[0].eval_velocity(0.0)
    # expect to rebound heading in the exact opposite direction:
    assert 0.9999 < abs(np.dot(v_1, nor)) / np.linalg.norm(v_1)


def test_degenerate_collision(pool_physics, gl_rendering, request):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_velocities = physics.eval_velocities(0.0)
    ball_positions[0] = (5.18317963e-05,  7.69200000e-01, -5.03094033e-01)
    ball_positions[1] = (-0.02813103,  0.7692,     -0.45880025)
    ball_velocities[0] = (0.00064499,  0,         -0.00113027)
    ball_velocities[1] = (-0.00423094,  0,         -0.00423761)
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    t_j = 0.570942120832911 - 0.570232363842049
    e_i = BallRollingEvent(0.0, 0, ball_positions[0], ball_velocities[0])
    e_j = BallRollingEvent(t_j, 1, ball_positions[1], ball_velocities[1])
    physics._add_event(e_i)
    physics.add_event_sequence(e_j)
    if not request.config.getoption('--no-distance-check'):
        check_ball_distances(physics, filename=request.node.originalname, t0=t_j, nt=64*4000)
