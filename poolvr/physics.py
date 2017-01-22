import numpy as np


class PoolPhysics(object):
    def __init__(self, num_balls=16):
        self.t = 0.0
        self._positions = np.empty((16, 3), dtype=np.float32)
    def strike_ball(i, q, v, omega):
        pass
    def get_poses(t):
        pass
