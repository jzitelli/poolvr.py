import os.path
import logging
from unittest import TestCase
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
        self.physics.PhysicsEvent.physics = self.physics


    def test_strike_ball(self):
        self.cue.position[:] = self.game.ball_positions[0]
        self.cue.position[2] += 0.5 * self.cue.length + self.physics.ball_radius
        self.cue.velocity[2] = -6.0
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        events = self.physics.strike_ball(0.0, 0, self.cue.world_matrix[1,:3], Q,
                                          self.cue.velocity,
                                          self.cue.mass)
        _logger.info('\n'.join(['  %f: %s' % (e.t, e) for e in events]))
        self.assertEquals(3, len(events))
        self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        self.assertIsInstance(events[2], PoolPhysics.RollToRestEvent)

        fig = plt.figure()
        for a, b in zip(events[:-1], events[1:]):
            ts = np.linspace(a.t, b.t, 50)
            plt.plot(ts, [self.physics.eval_positions(t)[0,0] for t in ts], '-o', label='$x$')
            plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
            plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-d', label='$z$')
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        plt.legend()

        pth = os.path.join(PLOTS_DIR, 'test_strike_ball.png')
        try:
            plt.savefig(pth)
        except:
            _logger.warning("could not save the plot to {}. i'll just show it to you:", pth)
            plt.show()
