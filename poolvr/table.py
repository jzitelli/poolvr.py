from .gl_rendering import Mesh, Material, Technique, Program
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive
from .techniques import EGA_TECHNIQUE


INCH2METER = 0.0254


class PoolTable(object):
    def __init__(self,
                 length=2.34,
                 height=0.77,
                 width=None,
                 width_rail=2*INCH2METER,
                 **kwargs):
        self.length = length
        self.height = height
        self.length = length
        self.height = height
        if width is None:
            width = 0.5 * length
        self.width = width
        self.width_rail = width_rail
        surface_material = Material(EGA_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface = PlanePrimitive(width=width, depth=length)
        surface.attributes['vertices'][:,1] = height
        surface.attributes['a_position'] = surface.attributes['vertices']
        self.mesh = Mesh({surface_material: [surface]})
