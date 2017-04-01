import numpy as np


from .gl_rendering import Mesh, Material, Technique, Program
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive
from .techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE


INCH2METER = 0.0254


class PoolTable(object):
    def __init__(self,
                 length=2.34,
                 height=0.77,
                 width=None,
                 width_rail=2*INCH2METER,
                 W_cushion=2*INCH2METER,
                 H_cushion=0.635*2.25*INCH2METER,
                 **kwargs):
        self.length = length
        self.height = height
        self.length = length
        self.height = height
        if width is None:
            width = 0.5 * length
        self.width = width
        self.width_rail = width_rail
        # surface_material = Material(EGA_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface = PlanePrimitive(width=width, depth=length)
        surface.attributes['vertices'][:,1] = height
        surface.attributes['a_position'] = surface.attributes['vertices']
        W_playable = width
        H_rail = 1.25 * H_cushion
        H_nose = 0.5 * H_cushion
        W_nose = 0.05 * W_cushion
        sqrt2 = np.sqrt(2)
        self.headCushionGeom = HexaPrimitive(vertices=np.array([
            # bottom quad:
            [[-0.5*W_playable + 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 1.2*sqrt2*W_cushion, 0.0, -0.5*W_cushion + W_nose],
             [-0.5*W_playable + 1.2*sqrt2*W_cushion, 0.0, -0.5*W_cushion + W_nose]],
            # top quad:
            [[-0.5*W_playable + 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 1.2*sqrt2*W_cushion, H_cushion, -0.5*W_cushion],
             [-0.5*W_playable + 1.2*sqrt2*W_cushion, H_cushion, -0.5*W_cushion]]], dtype=np.float32))
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,1] += self.height
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,2] += 0.5 * self.length - 0.5*W_cushion
        self.headCushionGeom.attributes['a_position'] = self.headCushionGeom.attributes['vertices']
        vertices = self.headCushionGeom.attributes['vertices'].copy()
        vertices.reshape(-1,3)[:,2] *= -1
        self.footCushionGeom = HexaPrimitive(vertices=vertices)
        self.footCushionGeom.attributes['a_position'] = self.footCushionGeom.attributes['vertices']
        self.mesh = Mesh({surface_material: [surface, self.headCushionGeom, self.footCushionGeom]})
