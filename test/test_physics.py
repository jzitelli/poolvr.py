import os.path
import logging
import traceback
import numpy as np
import pytest


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.physics import PoolPhysics
from poolvr.physics.events import CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent


from .utils import plot_ball_motion, plot_energy


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
    yield
    test_name = str(request.function.__name__)
    plot_ball_motion(0, pool_physics, title=test_name, coords=(0,2),
                     filename=os.path.join(PLOTS_DIR, test_name + '.png'))


def test_strike_ball(pool_physics, plot_motion):
    # test_name = 'test_strike_ball'
    physics = pool_physics
    physics.balls_on_table = [0]
    cue = PoolCue()
    r_c = physics.ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue.velocity[2] = -0.6
    events = physics.strike_ball(0.0, 0, r_c, cue.velocity, cue.mass)
    _logger.debug('strike on %d resulted in %d events: %s', 0, len(events),
                  '\n'.join(str(e) for e in events))
    assert(4 == len(events))
    assert(isinstance(events[0], CueStrikeEvent))
    assert(isinstance(events[1], BallSlidingEvent))
    assert(isinstance(events[2], BallRollingEvent))
    assert(isinstance(events[3], BallRestEvent))
    # plot_ball_motion(0, physics, title=test_name, coords=(0,2),
    #                  filename=os.path.join(PLOTS_DIR, test_name + '.png'))
    # plot_energy(physics, title=test_name + ' - energy', t_1=8.0, filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'))


@pytest.mark.skip
def test_ball_collision(pool_physics):
    test_name = 'test_ball_collision'
    physics = pool_physics
    physics.ball_positions[1] = physics.ball_positions[0]
    physics.ball_positions[1,2] -= 8 * physics.ball_radius
    physics.balls_on_table = [0, 1]
    start_event = BallSlidingEvent(0, 0, r_0=physics.ball_positions[0],
                                   v_0=np.array((0.0, 0.0, -0.6)),
                                   omega_0=np.zeros(3, dtype=np.float64))
    events = physics.add_event_sequence(start_event)
    _logger.debug('%d events added:\n%s', len(events), physics.events_str(events=events))
    plot_ball_motion(0, physics, title=test_name, coords=(0,2),
                     collision_depth=1,
                     filename=os.path.join(PLOTS_DIR, test_name + '.png'),
                     t_0=0.0, t_1=2.0)
    plot_energy(physics, title=test_name + ' - energy',
                filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'))







#     # def test_simple_ball_collision(self):
#     #     self.physics = PoolPhysics(num_balls=self.game.num_balls,
#     #                                ball_radius=self.game.ball_radius,
#     #                                initial_positions=self.game.ball_positions,
#     #                                use_simple_ball_collision=True)
#     #     self.game.physics = self.physics
#     #     self.game.reset()
#     #     self.physics.on_table[:] = False
#     #     self.physics.on_table[0] = True
#     #     self.physics.on_table[8] = True
#     #     self.cue.velocity[2] = -1.6
#     #     self.cue.velocity[0] = -0.02
#     #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     #     i = 0
#     #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     #     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     #     test_name = traceback.extract_stack(None, 1)[0][2]
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
#     #     plot_ball_motion(i, self.game, title=test_name, coords=2)
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
#     #     plot_energy(self.game, title=test_name + ' - energy')
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     #     if self.show:
#     #         show(self.game, title=test_name,
#     #              screenshots_dir=SCREENSHOTS_DIR)

#     # def test_ball_collision(self):
#     #     self.physics.on_table[:] = False
#     #     self.physics.on_table[0] = True
#     #     self.physics.on_table[8] = True
#     #     self.cue.velocity[2] = -1.6
#     #     self.cue.velocity[0] = -0.02
#     #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     #     i = 0
#     #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     #     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     #     test_name = traceback.extract_stack(None, 1)[0][2]
#     #     plot_ball_motion(i, self.game, title=test_name, coords=0)
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
#     #     plot_ball_motion(i, self.game, title=test_name, coords=2)
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
#     #     plot_energy(self.game, title=test_name + ' - energy')
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     #     if self.show:
#     #         show(self.game, title=test_name,
#     #              screenshots_dir=SCREENSHOTS_DIR)

#     # def test_break(self):
#     #     self.game.reset()
#     #     self.physics.on_table[:] = True
#     #     self.cue.velocity[2] = -1.8
#     #     self.cue.velocity[0] = -0.01
#     #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     #     i = 0
#     #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     #     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     #     test_name = traceback.extract_stack(None, 1)[0][2]
#     #     plot_ball_motion(i, self.game, title=test_name, coords=0)
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
#     #     plot_ball_motion(i, self.game, title=test_name, coords=2)
#     #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
#     #     plot_energy(self.game, title=test_name + ' - energy')
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     #     if self.show:
#     #         show(self.game, title=test_name,
#     #              screenshots_dir=SCREENSHOTS_DIR)

#     # def test_ball_collision_2(self):
#     #     self.game.reset()
#     #     self.physics.on_table[2:8:2] = False
#     #     self.physics.on_table[8:] = False
#     #     self.cue.velocity[2] = -1.6
#     #     self.cue.velocity[0] = -0.02
#     #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     #     i = 0
#     #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     #     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     #     test_name = traceback.extract_stack(None, 1)[0][2]
#     #     plot_ball_motion(i, self.game, title=test_name)
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
#     #     plot_energy(self.game, title=test_name + ' - energy')
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     #     if self.show:
#     #         show(self.game, title=test_name,
#     #              screenshots_dir=SCREENSHOTS_DIR)


#     # def test_simple_ball_collision_2(self):
#     #     self.physics = PoolPhysics(num_balls=self.game.num_balls,
#     #                                ball_radius=self.game.ball_radius,
#     #                                initial_positions=self.game.ball_positions,
#     #                                use_simple_ball_collision=True)
#     #     self.game.physics = self.physics
#     #     self.game.reset()
#     #     self.physics.on_table[2:8:2] = False
#     #     self.physics.on_table[8:] = False
#     #     self.cue.velocity[2] = -1.6
#     #     self.cue.velocity[0] = -0.02
#     #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
#     #     i = 0
#     #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#     #     _logger.debug('strike on %d resulted in %d events', i, n_events)
#     #     test_name = traceback.extract_stack(None, 1)[0][2]
#     #     plot_ball_motion(i, self.game, title=test_name)
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
#     #     plot_energy(self.game, title=test_name + ' - energy')
#     #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#     #     if self.show:
#     #         show(self.game, title=test_name,
#     #              screenshots_dir=SCREENSHOTS_DIR)
