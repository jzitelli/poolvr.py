from sys import stdout
import os.path
import logging
from unittest import TestCase, skip
import traceback
import numpy as np
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics
from .utils import plot_ball_motion, savefig, plot_energy
from .utils.gl_viewer import show


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
        self.physics.on_table[1:] = False
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        self.cue.velocity[2] = -0.8
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        events = self.physics.events
        self.assertEqual(3, len(events))
        self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        self.assertIsInstance(events[2], PoolPhysics.RollToRestEvent)
        test_name = traceback.extract_stack(None, 1)[0][2]
        plot_ball_motion(i, self.game, title=test_name, coords=(0,2))
        savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
        plot_energy(self.game, title=test_name + ' - energy', t_1=8.0)
        savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
        if self.show:
            show(self.game, title=test_name,
                 screenshots_dir=SCREENSHOTS_DIR)


    def test_ball_collision(self):
        self.game.reset()
        #self.physics.on_table[8:] = False
        self.physics.on_table[:] = False
        self.physics.on_table[0] = True
        self.physics.on_table[8] = True
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        test_name = traceback.extract_stack(None, 1)[0][2]
        plot_ball_motion(i, self.game, title=test_name, coords=0)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
        plot_ball_motion(i, self.game, title=test_name, coords=2)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
        plot_energy(self.game, title=test_name + ' - energy')
        savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
        if self.show:
            show(self.game, title=test_name,
                 screenshots_dir=SCREENSHOTS_DIR)

    def test_simple_ball_collision(self):
        self.physics = PoolPhysics(num_balls=self.game.num_balls,
                                   ball_radius=self.game.ball_radius,
                                   initial_positions=self.game.ball_positions,
                                   use_simple_ball_collision=True)
        self.game.physics = self.physics
        self.game.reset()
        #self.physics.on_table[8:] = False
        self.physics.on_table[:] = False
        self.physics.on_table[0] = True
        self.physics.on_table[8] = True
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        test_name = traceback.extract_stack(None, 1)[0][2]
        plot_ball_motion(i, self.game, title=test_name, coords=0)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
        plot_ball_motion(i, self.game, title=test_name, coords=2)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
        plot_energy(self.game, title=test_name + ' - energy')
        savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
        if self.show:
            show(self.game, title=test_name,
                 screenshots_dir=SCREENSHOTS_DIR)

    def test_break(self):
        self.game.reset()
        self.physics.on_table[:] = True
        self.cue.velocity[2] = -1.8
        self.cue.velocity[0] = -0.01
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        test_name = traceback.extract_stack(None, 1)[0][2]
        plot_ball_motion(i, self.game, title=test_name, coords=0)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'x')))
        plot_ball_motion(i, self.game, title=test_name, coords=2)
        savefig(os.path.join(PLOTS_DIR, '%s-%s.png' % (test_name, 'z')))
        plot_energy(self.game, title=test_name + ' - energy')
        savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
        if self.show:
            show(self.game, title=test_name,
                 screenshots_dir=SCREENSHOTS_DIR)

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
