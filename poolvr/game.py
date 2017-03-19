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
    :param stripe_colors: *optional* map defining a stripe color
                          for each ball that is striped
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
                 **kwargs):
        table = PoolTable(**kwargs)
        self.table = table
        self.ball_colors = ball_colors
        self.num_balls = len(ball_colors)
        self.ball_positions = np.empty((self.num_balls, 3), dtype=np.float32)
        self.ball_radius = ball_radius
        self.initial_positions(out=self.ball_positions)
        self.physics = PoolPhysics(num_balls=self.num_balls,
                                   ball_radius=ball_radius,
                                   initial_positions=self.ball_positions,
                                   **kwargs)
        self.t = 0.0
        self.ntt = 0.0
    def initial_positions(self, d=None, out=None):
        """Set balls to initial (racked) positions"""
        if d is None:
            d = 0.04 * self.ball_radius
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float32)
        height = self.table.height
        length = self.table.length
        ball_diameter = 2 * self.ball_radius
        # triangle racked:
        out[:,1] = height + self.ball_radius + 0.0001
        side_length = 4 * (ball_diameter + d)
        x_positions = np.concatenate([np.linspace(0,                        0.5 * side_length,                         5),
                                      np.linspace(-0.5*(ball_diameter + d), 0.5 * side_length - (ball_diameter + d),   4),
                                      np.linspace(-(ball_diameter + d),     0.5 * side_length - 2*(ball_diameter + d), 3),
                                      np.linspace(-1.5*(ball_diameter + d), 0.5 * side_length - 3*(ball_diameter + d), 2),
                                      np.array([-2*(ball_diameter + d)])])
        z_positions = np.concatenate([np.linspace(0,                                    np.sqrt(3)/2 * side_length, 5),
                                      np.linspace(0.5*np.sqrt(3) * (ball_diameter + d), np.sqrt(3)/2 * side_length, 4),
                                      np.linspace(np.sqrt(3) * (ball_diameter + d),     np.sqrt(3)/2 * side_length, 3),
                                      np.linspace(1.5*np.sqrt(3) * (ball_diameter + d), np.sqrt(3)/2 * side_length, 2),
                                      np.array([np.sqrt(3)/2 * side_length])])
        z_positions *= -1
        z_positions -= length / 8
        out[1:,0] = x_positions
        out[1:,2] = z_positions
        # cue ball at head spot:
        out[0,0] = 0.0
        out[0,2] = 0.25 * length
        return out
    def reset(self):
        self.initial_positions(out=self.ball_positions)
        self.physics.reset(self.ball_positions)
        self.t = 0.0
        self.ntt = 0.0
    def advance_time(self):
        ntt = self.physics.next_turn_time()
        if ntt:
            self.ntt = ntt
            self.t = ntt
