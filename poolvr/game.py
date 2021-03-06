import logging
import numpy as np


_logger = logging.getLogger(__name__)

INCH2METER = 0.0254


from .table import PoolTable
from .physics import PoolPhysics


class PoolGame(object):
    """
    Game state for a pool "game".

    :param ball_colors: array defining a base color for each ball
    :
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
    def __init__(self, ball_colors=BALL_COLORS,
                 ball_radius=0.02625,
                 table=None,
                 physics=None,
                 **kwargs):
        self.ball_colors = ball_colors
        self.ball_radius = ball_radius
        if table is None:
            table = PoolTable(ball_radius=ball_radius, **kwargs)
        self.table = table
        if physics is None:
            physics = PoolPhysics(ball_radius=ball_radius, **kwargs)
        self.physics = physics
        self.ball_positions = self.table.calc_racked_positions()
        self.ball_velocities = np.zeros((self.num_balls, 3), dtype=np.float64)
        self.ball_angular_velocities = np.zeros((self.num_balls, 3), dtype=np.float64)
        self.ball_quaternions = np.zeros((self.num_balls, 4), dtype=np.float64)
        self.ball_quaternions[:,3] = 1
        self.t = 0.0
        self.ntt = 0.0

    def reset(self, **kwargs):
        """
        Resets the game state, which means: set balls in their initial stationary
        positions; reset physics engine.
        """
        self.physics.reset(**kwargs)
        self.ball_positions[:] = self.table.calc_racked_positions()
        self.ball_velocities[:] = 0
        self.ball_angular_velocities[:] = 0
        self.ball_quaternions[:] = 0
        self.ball_quaternions[:,3] = 1
        self.t = 0.0
        self.ntt = 0.0
        _logger.debug('game reset')

    @property
    def num_balls(self):
        return self.physics.num_balls

    def step(self, dt, **kwargs):
        self.t += dt
        self.physics.step(dt, **kwargs)
        self.physics.eval_positions(self.t, out=self.ball_positions)
        self.physics.eval_velocities(self.t, out=self.ball_velocities)
        self.physics.eval_angular_velocities(self.t, out=self.ball_angular_velocities)
        for q, omega in zip(self.ball_quaternions, self.ball_angular_velocities):
            q_w = q[3]
            q[3] -= 0.5 * dt * omega.dot(q[:3])
            q[:3] += 0.5 * dt * (q_w * omega + np.cross(omega, q[:3]))
            q /= np.sqrt(np.dot(q, q))
