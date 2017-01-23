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
    def aabb_check(self, positions, radius):
        positions = (positions - self.world_matrix[3,:3]).dot(self.world_matrix[:3,:3].T)
        aabb = self.bb
        separate = ((aabb[0] > positions + radius) | (aabb[1] < positions - radius)).any(axis=-1)
        return ~separate
    def contact(self, position, radius):
        x, y, z = position
        r_sqrd = x**2 + z**2
        if abs(y) <= 0.5*self.length:
            # potential contact on the side of the cue:
            if r_sqrd > self.radius**2 and r_sqrd <= (self.radius + radius)**2:
                # find point of contact on ball:
                n = position.copy()
                n[1] = 0.0
                n /= np.sqrt(r_sqrd)
                poc = position - radius * n
                return i, poc
        elif abs(y) <= 0.5*self.length + radius:
            # potential contact on flat end of the cue:
            if r_sqrd <= radius**2:
                # contact on the flat end:
                poc = position.copy()
                if y >= 0.0:
                    poc[1] -= radius
                else:
                    poc[1] += radius
                return i, poc
            else:
                r = np.sqrt(r_sqrd)
                if (r - radius)**2 + (abs(y) - 0.5*self.length)**2 <= radius**2:
                    # contact on the ring edge of the flat end:
                    if y >= 0.0:
                        n = np.array([0.0, -(y - 0.5*self.length), 0.0])
                    else:
                        n = np.array([0.0, -y - 0.5*self.length, 0.0])
                    n[::2] += -(r - radius) / r * position[::2]
                    n /= np.linalg.norm(n)
                    poc = position + radius * n
                    return i, poc
        return None
