import logging
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


from poolvr.cue import Cue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics


from .viewer import OpenGLViewer


class PhysicsTests(TestCase):
    def setUp(self):
        self.table = PoolTable()
        self.game = PoolGame()
        self.cue = Cue()
        self.physics = PoolPhysics(initial_positions=self.game.ball_positions)
    def test_strike_ball(self):
        self.cue.position[:] = self.game.ball_positions[0]
        self.cue.position[2] += 0.5 * self.cue.length + self.physics.ball_radius
        self.cue.velocity[2] = -6.0
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        events = self.physics.strike_ball(0.0, 0, self.cue.world_matrix[1,:3], Q,
                                          self.cue.velocity,
                                          self.cue.mass)
        print({e.t: e for e in events})
        T = events[-1].t
        fig = plt.figure()
        ts = np.linspace(0.0, T, 50)
        plt.plot(ts, [self.physics.eval_positions(t)[0,0] for t in ts], '-o', label='$x$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-x', label='$z$')
        plt.xlabel('$t$')
        plt.legend()
        plt.show()
