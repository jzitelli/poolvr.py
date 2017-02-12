from unittest import TestCase


from poolvr.cue import Cue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics


class PhysicsTests(TestCase):
    def setUp(self):
        self.table = PoolTable()
        self.physics = PoolPhysics()
        self.cue = Cue()
        self.game = PoolGame()
    # def tearDown(self):
    #     pass
    def test_strike_ball(self):
        self.cue.position[:] = self.game.ball_positions[0]
        self.cue.position[2] += 0.5 * self.cue.length
        self.cue.velocity[2] = -2.0
        events = self.physics.strike_ball(0.0, 0, self.cue.world_matrix[1,:3],
                                          self.cue.tip_position, self.cue.velocity,
                                          self.cue.mass)
