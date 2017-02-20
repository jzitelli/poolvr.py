import os.path
import logging
from unittest import TestCase, skip
import traceback
import numpy as np
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')


class PhysicsTests(TestCase):


    def setUp(self):
        self.table = PoolTable()
        self.game = PoolGame()
        self.cue = PoolCue()
        self.physics = PoolPhysics(initial_positions=self.game.ball_positions)


    def test_reset(self):
        self.physics.reset(self.game.initial_positions())
        self.assertLessEqual(np.linalg.norm(self.game.initial_positions() -
                                            self.physics.eval_positions(0.0)),
                             0.001 * self.physics.ball_radius)
        self.assertTrue((self.physics.eval_velocities(0.0) == 0).all())


    def test_strike_ball(self):
        self.physics.reset(self.game.initial_positions())
        self.physics.on_table[1:] = False
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        self.cue.velocity[2] = -2.0
        events = self.physics.strike_ball(0.0, 0, Q, self.cue.velocity, self.cue.mass)
        _logger.info('\n'.join(['  %f: %s' % (e.t, e) for e in events]))
        self.assertEqual(3, len(events))
        self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        self.assertIsInstance(events[2], PoolPhysics.RollToRestEvent)

        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        ts = np.concatenate([np.linspace(a.t, b.t, int((b.t - a.t) * 20 + 2))
                             for a, b in zip(events[:-1], events[1:])])
        for b in events[1:]:
            plt.axvline(b.t)
        plt.plot(ts, [self.physics.eval_positions(t)[0,0] for t in ts], '-o', label='$x$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-d', label='$z$')
        plt.legend()
        self._savefig()


    def test_ball_collision_event(self):
        self.physics.reset(self.game.initial_positions())
        self.physics.on_table[2:] = False
        self.cue.velocity[2] = -4.0
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        events = self.physics.strike_ball(0.0, 0, Q,
                                          self.cue.velocity, self.cue.mass)
        _logger.info('\n'.join(['  %f: %s' % (e.t, e) for e in events]))
        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        ts = np.concatenate([np.linspace(a.t, b.t, int((b.t - a.t) * 23))
                             for a, b in zip(events[:-1], events[1:])])
        plt.plot(ts, [self.physics.eval_positions(t)[0,0] for t in ts], '-o', label='$x$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-d', label='$z$')
        plt.legend()
        self._savefig()


    def _savefig(self):
        title = traceback.extract_stack(None, 2)[0][2]
        pth = os.path.join(PLOTS_DIR, '%s.png' % title)
        plt.title(title)
        try:
            plt.savefig(pth)
            _logger.info("...saved figure to %s", pth)
        except:
            _logger.warning("could not save the plot to %s. i'll just show it to you:", pth)
            plt.show()
