from .gl_rendering import Mesh, Material
from .techniques import EGA_TECHNIQUE
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive


INCH2METER = 0.0254


class Table(Mesh):
    def __init__(self, L_table=2.34, H_table=0.77, W_table=None, W_rail=2*INCH2METER):
        self.L_table = L_table
        self.H_table = H_table
        if W_table is None:
            W_table = 0.5 * L_table
        self.W_table = W_table
        surface_material = Material(EGA_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface = PlanePrimitive(length=W_table, depth=L_table)
        Mesh.__init__(self, {surface_material: [surface]})
