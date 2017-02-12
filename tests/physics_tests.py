import logging
from unittest import TestCase


_logger = logging.getLogger(__name__)


from poolvr.cue import Cue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics


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
        events = self.physics.strike_ball(0.0, 0, self.cue.world_matrix[1,:3],
                                          self.cue.tip_position - self.game.ball_positions[0],
                                          self.cue.velocity,
                                          self.cue.mass)
        print({e.t: e for e in events})
