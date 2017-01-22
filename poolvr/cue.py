from .gl_rendering import *
from .primitives import *
from .techniques import *


INCH2METER = 0.0254


class Cue(Mesh):
    def __init__(self, radius=0.007, length=1.15, mass=0.54):
        self.radius = radius
        self.length = length
        self.mass = mass
        cylinder = CylinderPrimitive(radius=radius, height=length)
        cylinder.attributes['a_position'] = cylinder.attributes['vertices']
        Mesh.__init__(self, {Material(EGA_TECHNIQUE): [cylinder]})
        self._positions = None
    def aabb_check(self, positions, radius):
        if self._positions is None:
            self._positions = positions.copy()
        R, r = self.radius, radius
