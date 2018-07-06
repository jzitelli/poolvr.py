import os.path
import logging
from unittest import TestCase
import traceback
import numpy as np


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.game import PoolGame
from poolvr.physics.events import CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent


from .utils import plot_ball_motion, plot_energy


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


class PhysicsTests(TestCase):
    show = False

    def setUp(self):
        self.game = PoolGame()
        self.physics = self.game.physics
        self.table = self.game.table
        self.cue = PoolCue()
        self.playback_rate = 1


    def test_strike_ball(self):
        test_name = traceback.extract_stack(None, 1)[0][2]
        on_table = np.array(self.physics.num_balls*[False])
        on_table[:1] = True
        self.physics.reset(on_table=on_table,
                           ball_positions=self.physics.ball_positions)
        r_c = self.physics.ball_positions[0].copy()
        r_c[2] += self.physics.ball_radius
        self.cue.velocity[2] = -0.8
        events = self.physics.strike_ball(0.0, 0, r_c, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events: %s', 0, len(events),
                      '\n'.join(str(e) for e in events))
        self.assertEqual(4, len(events))
        self.assertIsInstance(events[0], CueStrikeEvent)
        self.assertIsInstance(events[1], BallSlidingEvent)
        self.assertIsInstance(events[2], BallRollingEvent)
        self.assertIsInstance(events[3], BallRestEvent)
        plot_ball_motion(0, self.game, title=test_name, coords=(0,2),
                         filename=os.path.join(PLOTS_DIR, test_name + '.png'))
        plot_energy(self.game, title=test_name + ' - energy', t_1=8.0, filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'))


    def test_ball_collision(self):
        test_name = traceback.extract_stack(None, 1)[0][2]
        on_table = np.array(self.physics.num_balls*[False])
        on_table[:2] = True
        ball_positions = self.physics.ball_positions.copy()
        ball_positions[1] = ball_positions[0]; ball_positions[1,2] -= 8 * self.physics.ball_radius
        self.physics.reset(on_table=on_table,
                           ball_positions=ball_positions)
        start_event = BallRollingEvent(0, 0, self.physics.ball_positions[0], np.array((0.0, 0.0, -0.5)))
        events = self.physics.add_event_sequence(start_event)
        _logger.debug('%d events added:\n%s', len(events), self.physics.events_str(events=events))
        plot_ball_motion(0, self.game, title=test_name, coords=(0,2),
                         collision_depth=1,
                         filename=os.path.join(PLOTS_DIR, test_name + '.png'),
                         t_0=0.0, t_1=2.0)
        #plot_ball_motion(i, self.game, title=test_name, coords=0)
        #plot_energy(self.game, title=test_name + ' - energy', t_1=8.0, filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'))


    # def test_simple_ball_collision(self):
    #     self.physics = PoolPhysics(num_balls=self.game.num_balls,
    #                                ball_radius=self.game.ball_radius,
    #                                initial_positions=self.game.ball_positions,
    #                                use_simple_ball_collision=True)
    #     self.game.physics = self.physics
    #     self.game.reset()
    #     self.physics.on_table[:] = False
    #     self.physics.on_table[0] = True
    #     self.physics.on_table[8] = True
    #     self.cue.velocity[2] = -1.6
    #     self.cue.velocity[0] = -0.02
    #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
    #     i = 0
    #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
    #     _logger.debug('strike on %d resulted in %d events', i, n_events)
    #     test_name = traceback.extract_stack(None, 1)[0][2]
    #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
    #     plot_ball_motion(i, self.game, title=test_name, coords=2)
    #     savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
    #     plot_energy(self.game, title=test_name + ' - energy')
    #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
    #     if self.show:
    #         show(self.game, title=test_name,
    #              screenshots_dir=SCREENSHOTS_DIR)

    # def test_ball_collision(self):
    #     self.physics.on_table[:] = False
    #     self.physics.on_table[0] = True
    #     self.physics.on_table[8] = True
    #     self.cue.velocity[2] = -1.6
    #     self.cue.velocity[0] = -0.02
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

    # def test_ball_collision_2(self):
    #     self.game.reset()
    #     self.physics.on_table[2:8:2] = False
    #     self.physics.on_table[8:] = False
    #     self.cue.velocity[2] = -1.6
    #     self.cue.velocity[0] = -0.02
    #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
    #     i = 0
    #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
    #     _logger.debug('strike on %d resulted in %d events', i, n_events)
    #     test_name = traceback.extract_stack(None, 1)[0][2]
    #     plot_ball_motion(i, self.game, title=test_name)
    #     savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
    #     plot_energy(self.game, title=test_name + ' - energy')
    #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
    #     if self.show:
    #         show(self.game, title=test_name,
    #              screenshots_dir=SCREENSHOTS_DIR)


    # def test_simple_ball_collision_2(self):
    #     self.physics = PoolPhysics(num_balls=self.game.num_balls,
    #                                ball_radius=self.game.ball_radius,
    #                                initial_positions=self.game.ball_positions,
    #                                use_simple_ball_collision=True)
    #     self.game.physics = self.physics
    #     self.game.reset()
    #     self.physics.on_table[2:8:2] = False
    #     self.physics.on_table[8:] = False
    #     self.cue.velocity[2] = -1.6
    #     self.cue.velocity[0] = -0.02
    #     Q = np.array((0.0, 0.0, self.physics.ball_radius))
    #     i = 0
    #     n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
    #     _logger.debug('strike on %d resulted in %d events', i, n_events)
    #     test_name = traceback.extract_stack(None, 1)[0][2]
    #     plot_ball_motion(i, self.game, title=test_name)
    #     savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
    #     plot_energy(self.game, title=test_name + ' - energy')
    #     savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
    #     if self.show:
    #         show(self.game, title=test_name,
    #              screenshots_dir=SCREENSHOTS_DIR)
