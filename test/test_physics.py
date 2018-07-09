import os.path
import logging
import numpy as np
import pytest


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.physics import PoolPhysics
from poolvr.physics.events import CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


@pytest.fixture
def pool_table():
    return PoolTable()


@pytest.fixture()
def pool_physics(request, pool_table):
    return PoolPhysics(initial_positions=np.array(pool_table.calc_racked_positions(), dtype=np.float64),
                       use_simple_ball_collisions=True)


@pytest.fixture
def plot_motion(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_ball_motion as plot
    yield
    test_name = str(request.function.__name__)
    plot(0, pool_physics,
         title=test_name + ' (position)',
         coords=(0,2),
         filename=os.path.join(PLOTS_DIR, test_name + '.png'))


@pytest.fixture
def plot_energy(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_energy as plot
    yield
    test_name = str(request.function.__name__)
    plot(pool_physics, title=test_name + ' (energy)',
         filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'))


@pytest.fixture
def plot_motion_timelapse(pool_physics, pool_table, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_motion_timelapse as plot
    yield
    test_name = str(request.function.__name__)
    plot(pool_physics, table=pool_table,
         title=test_name + ' (timelapse)',
         filename=os.path.join(PLOTS_DIR, test_name + '_timelapse.png'),
         show=True)


def test_strike_ball(pool_physics, plot_motion_timelapse):
    physics = pool_physics
    physics.reset(balls_on_table=[0])
    r_c = physics.ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue = PoolCue()
    cue.velocity[2] = -0.6
    events = physics.strike_ball(0.0, 0, r_c, cue.velocity, cue.mass)
    _logger.debug('strike on %d resulted in %d events:\n\n%s\n', 0, len(events),
                  physics.events_str(events))
    assert 4 == len(events)
    assert isinstance(events[0], CueStrikeEvent)
    assert isinstance(events[1], BallSlidingEvent)
    assert isinstance(events[2], BallRollingEvent)
    assert isinstance(events[3], BallRestEvent)


def test_ball_collision(pool_physics, plot_motion, plot_energy):
    physics = pool_physics
    ball_positions = physics.ball_positions.copy()
    ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * physics.ball_radius
    physics.reset(balls_on_table=[0, 1],
                  ball_positions=ball_positions)
    start_event = BallSlidingEvent(0, 0, r_0=physics.ball_positions[0],
                                   v_0=np.array((0.0, 0.0, -0.6)),
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n\n%s\n', len(events), physics.events_str(events=events))
    assert 6 == len(events)
    assert isinstance(events[0], BallSlidingEvent)
    assert isinstance(events[1], BallRollingEvent)
    assert isinstance(events[2], BallCollisionEvent)
    assert isinstance(events[3], BallRestEvent)
    assert isinstance(events[4], BallRollingEvent)
    assert isinstance(events[5], BallRestEvent)
    # plot_ball_motion(0, physics, title=test_name, coords=(0,2),
    #                  collision_depth=1,
    #                  filename=os.path.join(PLOTS_DIR, test_name + '.png'),
    #                  t_0=0.0, t_1=2.0)


# def test_break(self):
#     self.game.reset()
#     self.physics.on_table[:] = True
#     self.cue.velocity[2] = -1.8
#     self.cue.velocity[0] = -0.01
#     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     i = 0
#     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     test_name = traceback.extract_stack(None, 1)[0][2]
#     plot_ball_motion(i, self.game, title=test_name, coords=0)
#     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
#     plot_ball_motion(i, self.game, title=test_name, coords=2)
#     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
#     plot_energy(self.game, title=test_name + ' - energy')
#     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     if self.show:
#         show(self.game, title=test_name,
#              screenshots_dir=SCREENSHOTS_DIR)
