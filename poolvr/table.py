from .gl_rendering import Mesh, Material
from .techniques import EGA_TECHNIQUE
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive


INCH2METER = 0.0254


class PoolTable(object):
    def __init__(self,
                 L_table=2.34,
                 H_table=0.77,
                 W_table=None,
                 W_rail=2*INCH2METER,
                 ball_radius=INCH2METER*1.125):
        self.L_table = L_table
        self.H_table = H_table
        if W_table is None:
            W_table = 0.5 * L_table
        self.W_table = W_table
        self.ball_radius = ball_radius
        surface_material = Material(EGA_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface = PlanePrimitive(width=W_table, depth=L_table)
        surface.attributes['vertices'][:,1] = H_table
        surface.attributes['a_position'] = surface.attributes['vertices']
        self.mesh = Mesh({surface_material: [surface]})
