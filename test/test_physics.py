import logging
_logger = logging.getLogger(__name__)
import numpy as np
import pytest


from poolvr.cue import PoolCue
from poolvr.physics.events import PhysicsEvent, CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent


#@pytest.mark.skip
@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_strike_ball(pool_physics, ball_collision_model,
                     gl_rendering):#,
                     #plot_motion_timelapse,
                     #plot_motion_z_position):
    physics = pool_physics
    physics.reset(balls_on_table=[0])
    r_c = physics.ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue = PoolCue()
    cue.velocity[2] = -0.6
    events = physics.strike_ball(0.0, 0, r_c, cue.velocity, cue.mass)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  physics.events_str(events))
    # assert 4 == len(events)
    # assert isinstance(events[0], CueStrikeEvent)
    # assert isinstance(events[1], BallSlidingEvent)
    # assert isinstance(events[2], BallRollingEvent)
    # assert isinstance(events[3], BallRestEvent)


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_ball_collision(pool_physics, ball_collision_model,
                        gl_rendering):
    physics = pool_physics
    ball_positions = physics.ball_positions.copy()
    ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    start_event = BallSlidingEvent(0, 0, r_0=physics.ball_positions[0],
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


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_sliding_ball_collision(pool_physics, ball_collision_model,
                                gl_rendering):
    physics = pool_physics
    ball_positions = physics.ball_positions.copy()
    ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    start_event = BallSlidingEvent(0, 0, r_0=physics.ball_positions[0],
                                   v_0=np.array((0.0, 0.0, -2.0)),
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events), physics.events_str(events=events))
    # assert 6 == len(events)
    # assert isinstance(events[0], BallSlidingEvent)
    # assert isinstance(events[1], BallCollisionEvent)
    # assert isinstance(events[2], BallRestEvent)
    # assert isinstance(events[3], BallSlidingEvent)
    # assert isinstance(events[4], BallRollingEvent)
    # assert isinstance(events[5], BallRestEvent)


@pytest.mark.parametrize("ball_collision_model", ['simple'])
def test_break(pool_physics, ball_collision_model, gl_rendering):
    physics = pool_physics
    r_c = physics.ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue = PoolCue()
    cue.velocity[2] = -1.6
    cue.velocity[0] = -0.01
    events = physics.strike_ball(0.0, 0, r_c, cue.velocity, cue.mass)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  physics.events_str(events))
