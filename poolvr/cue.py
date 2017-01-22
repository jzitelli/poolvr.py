from .gl_rendering import *
from .primitives import *
from .techniques import *


INCH2METER = 0.0254


class Cue(Mesh):
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]], dtype=np.float32)
    def __init__(self, radius=0.007, length=1.15, mass=0.54):
        self.radius = radius
        self.length = length
        self.mass = mass
        cylinder = CylinderPrimitive(radius=radius, height=length)
        cylinder.attributes['a_position'] = cylinder.attributes['vertices']
        Mesh.__init__(self, {Material(EGA_TECHNIQUE): [cylinder]})
        self._positions = None
        self.position = self.world_matrix[3,:3]
        self.velocity = np.zeros(3, dtype=np.float32)
        self.bb = np.array([[-radius, -0.5*length, -radius],
                            [radius, 0.5*length, radius]], dtype=np.float32)
    def aabb_check(self, positions, ball_radius):
        positions = (positions - self.world_matrix[3,:3]).dot(self.world_matrix[:3,:3].T)
        aabb = self.bb
        separate = ((aabb[0] > positions + ball_radius) | (aabb[1] < positions - ball_radius)).any(axis=-1)
        return ~separate
