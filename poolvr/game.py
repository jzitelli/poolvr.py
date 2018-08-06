import logging
import numpy as np


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254


from .table import PoolTable
from .physics import PoolPhysics


class PoolGame(object):
    """
    Game state for a pool "game"

    :param ball_colors: array defining a base color for each ball
    """
    BALL_COLORS = [0xddddde,
                   0xeeee00,
                   0x0000ee,
                   0xee0000,
                   0xee00ee,
                   0xee7700,
                   0x00ee00,
                   0xbb2244,
                   0x111111]
    BALL_COLORS = BALL_COLORS + BALL_COLORS[1:-1]
    def __init__(self, ball_colors=BALL_COLORS, ball_radius=1.125*INCH2METER,
                 table=None,
                 physics=None,
                 **kwargs):
        if table is None:
            table = PoolTable(**kwargs)
        self.table = table
        self.ball_colors = ball_colors
        self.num_balls = len(ball_colors)
        self.ball_radius = ball_radius
        if physics is None:
            physics = PoolPhysics(num_balls=self.num_balls,
                                  ball_radius=ball_radius,
                                  **kwargs)
        self.physics = physics
        self.ball_positions = self.physics.ball_positions.copy()
        self.ball_quaternions = np.zeros((self.num_balls, 4), dtype=np.float32)
        self.ball_quaternions[:,3] = 1
        self.t = 0.0
        self.ntt = 0.0

    def reset(self):
        """
        Resets the game state, which means: set balls in their initial stationary
        positions; reset physics engine.
        """
        self.physics.reset()
        self.ball_positions[:] = self.physics.ball_positions
        self.ball_quaternions[:] = 0; self.ball_quaternions[:,3] = 1
        self.t = 0.0
        self.ntt = 0.0
