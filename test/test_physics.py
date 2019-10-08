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
                                   RailCollisionEvent)


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
                            '  %d RailCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, RailCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
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
    r_c = ball_positions[0].copy()
    r_c[2] += 0.5 * np.sqrt(2.0) * physics.ball_radius
    r_c[1] += 0.5 * np.sqrt(2.0) * physics.ball_radius
    V = np.array((-0.006, 0.0, -3.4), dtype=np.float64)
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
                            '  %d RailCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, RailCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
    check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


def test_break_hard_realtime(pool_physics_realtime,
                             plot_motion_timelapse,
                             plot_energy,
                             gl_rendering,
                             request):
    physics = pool_physics_realtime
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
    events = physics.events
    _logger.info('\n'.join(['strike on %d resulted in %d events:',
                            '  %d BallSlidingEvents',
                            '  %d BallRollingEvents',
                            '  %d BallSpinningEvents',
                            '  %d BallRestEvents',
                            '  %d RailCollisionEvents',
                            '  %d BallCollisionEvents']),
                 0, len(events),
                 len([e for e in events if isinstance(e, BallSlidingEvent)]),
                 len([e for e in events if isinstance(e, BallRollingEvent)]),
                 len([e for e in events if isinstance(e, BallSpinningEvent)]),
                 len([e for e in events if isinstance(e, BallRestEvent)]),
                 len([e for e in events if isinstance(e, RailCollisionEvent)]),
                 len([e for e in events if isinstance(e, BallCollisionEvent)]))
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
    check_ball_distances(physics, filename=request.node.originalname)
    # _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
    #               PhysicsEvent.events_str(events))


@pytest.mark.parametrize("side,i_c", [(s, i) for s in range(4) for i in range(2)])
def test_corner_collision(pool_physics,
                          gl_rendering,
                          plot_motion_timelapse,
                          plot_energy,
                          side, i_c):
    physics = pool_physics
    ball_positions = physics.eval_positions(0.0)
    ball_positions[0,::2] = 0
    physics.reset(balls_on_table=[0],
                  ball_positions=ball_positions)
    R = physics.ball_radius
    r_c = physics._r_cp[side,i_c]
    r_i = r_c + R * np.array([np.sign(r_c[0])*np.cos(10*DEG2RAD),
                              0.0,
                              np.sign(r_c[2])*np.sin(10*DEG2RAD)])
    r_0i = r_i - ball_positions[0]
    v_0 = 3.0 * r_0i / np.sqrt(np.dot(r_0i, r_0i))
    start_event = BallSlidingEvent(0, 0,
                                   r_0=ball_positions[0],
                                   v_0=v_0,
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    assert any(isinstance(e, CornerCollisionEvent) for e in events)
    _logger.debug('%d events added:\n\n%s\n', len(events),
                  PhysicsEvent.events_str(events=events))


# def test_pocket_scratch(pool_physics,
#                         gl_rendering,
#                         plot_motion_timelapse,
#                         plot_energy):
#     physics = pool_physics
#     ball_positions = physics.eval_positions(0.0)
#     physics.reset(balls_on_table=[0],
#                   ball_positions=ball_positions)
#     r_p = np.array(physics.table.pocket_positions[0])
#     r_0p = r_p - ball_positions[0]
#     v_0 = 1.9 * r_0p / np.sqrt(np.dot(r_0p, r_0p))
#     start_event = BallSlidingEvent(0, 0, r_0=ball_positions[0],
#                                    v_0=v_0,
#                                    omega_0=np.zeros(3, dtype=np.float64))
#     events = physics.add_event_sequence(start_event)
#     _logger.debug('%d events added:\n\n%s\n', len(events), PhysicsEvent.events_str(events=events))


# def test_occlusion(pool_physics, plot_occlusion, request):
#     physics = pool_physics
#     assert (physics._occ_ij == ~np.array([[0,1,1,1,1,1,1,0,0,0,1,0,0,1,0,1],
#                                           [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
#                                           [1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0],
#                                           [1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0],
#                                           [1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0],
#                                           [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
#                                           [1,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0],
#                                           [0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0],
#                                           [0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0],
#                                           [0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0],
#                                           [1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0],
#                                           [0,0,0,0,0,0,0,1,1,0,1,0,1,1,1,0],
#                                           [0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0],
#                                           [1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1],
#                                           [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
#                                           [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0]], dtype=np.bool)).all()


# def test_updating_occlusion(pool_physics,
#                             plot_motion_timelapse,
#                             plot_initial_positions,
#                             plot_final_positions,
#                             request):
#     show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
#     if not (show_plots or save_plots):
#         return
#     import matplotlib.pyplot as plt
#     physics = pool_physics
#     plt.imshow(physics._occ_ij)
#     if show_plots:
#         plt.show()
#     if save_plots:
#         try:
#             filename = os.path.join(os.path.dirname(__file__), 'plots', 'test_updating_occlusion_0.png')
#             dirname = os.path.dirname(filename)
#             if not os.path.exists(dirname):
#                 os.makedirs(dirname, exist_ok=True)
#             plt.savefig(filename)
#             _logger.info('saved plot to "%s"', filename)
#         except Exception as err:
#             _logger.error(err)
#     plt.close()
#     ball_positions = physics.eval_positions(0.0)
#     r_c = ball_positions[0].copy()
#     r_c[2] += physics.ball_radius
#     V = np.array((-0.01, 0.0, -1.8), dtype=np.float64)
#     M = 0.54
#     from time import perf_counter
#     t0 = perf_counter()
#     events = physics.strike_ball(0.0, 0, ball_positions[0], r_c, V, M)
#     t1 = perf_counter()
#     _logger.info('evaluation time: %s', t1-t0)
#     _logger.debug('strike on %d resulted in %d events', 0, len(events))
#     plt.imshow(physics._occ_ij)
#     if show_plots:
#         plt.show()
#     if save_plots:
#         try:
#             filename = os.path.join(os.path.dirname(__file__), 'plots', 'test_updating_occlusion_1.png')
#             dirname = os.path.dirname(filename)
#             if not os.path.exists(dirname):
#                 os.makedirs(dirname, exist_ok=True)
#             plt.savefig(filename)
#             _logger.info('saved plot to "%s"', filename)
#         except Exception as err:
#             _logger.error(err)
#     plt.close()
