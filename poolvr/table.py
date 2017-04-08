import os.path
import numpy as np
import itertools

from .gl_rendering import Mesh, Material, Texture
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive, SpherePrimitive
from .techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE
from .billboards import BillboardParticles


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


INCH2METER = 0.0254
SQRT2 = np.sqrt(2)


class PoolTable(object):
    def __init__(self,
                 length=2.34,
                 height=0.77,
                 width=None,
                 width_rail=2*INCH2METER,
                 W_cushion=1.6*INCH2METER,
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
        #surface_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0.3, 0.0, 0.0]})
        surface_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0xaa/0xff, 0.0, 0.0]})
        cushion_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0x02/0xff, 0x88/0xff, 0x44/0xff, 0.0]})
        surface = PlanePrimitive(width=width, depth=length)
        surface.attributes['vertices'][:,1] = height
        surface.attributes['a_position'] = surface.attributes['vertices']
        W_playable = width - 2*W_cushion
        H_rail = 1.4 * H_cushion
        W_nose = 0.045 * W_cushion
        ball_diameter = 2.25*INCH2METER
        H_cushion = 0.82*ball_diameter
        self.headCushionGeom = HexaPrimitive(vertices=np.array([
            # bottom quad:
            [[-0.5*W_playable + 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 1.2*SQRT2*W_cushion, 0.71*ball_diameter, -0.5*W_cushion + W_nose],
             [-0.5*W_playable + 1.2*SQRT2*W_cushion, 0.71*ball_diameter, -0.5*W_cushion + W_nose]],
            # top quad:
            [[-0.5*W_playable + 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 1.2*SQRT2*W_cushion, H_cushion, -0.5*W_cushion],
             [-0.5*W_playable + 1.2*SQRT2*W_cushion, H_cushion, -0.5*W_cushion]]], dtype=np.float32))
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,1] += self.height
        _vertices = self.headCushionGeom.attributes['vertices'].copy()
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,2] += 0.5 * self.length - 0.5*W_cushion
        self.headCushionGeom.attributes['a_position'] = self.headCushionGeom.attributes['vertices']
        vertices = _vertices.copy()
        vertices.reshape(-1,3)[:,2] *= -1
        vertices.reshape(-1,3)[:,2] -= 0.5 * self.length - 0.5*W_cushion
        self.footCushionGeom = HexaPrimitive(vertices=vertices)
        self.footCushionGeom.attributes['a_position'] = self.footCushionGeom.attributes['vertices']
        rotation = np.array([[0.0, 0.0, -1.0],
                             [0.0, 1.0,  0.0],
                             [1.0, 0.0,  0.0]], dtype=np.float32).T
        vertices = _vertices.copy()
        vertices[0, 2, 0] = 0.5*W_playable - 0.6*SQRT2*W_cushion
        vertices[1, 2, 0] = vertices[0, 2, 0]
        vertices.reshape(-1,3)[:] = rotation.dot(vertices.reshape(-1,3).T).T
        vertices.reshape(-1,3)[:,2] += 0.25 * self.length
        vertices.reshape(-1,3)[:,0] += 0.5 * self.width - 0.5*W_cushion
        self.rightHeadCushionGeom = HexaPrimitive(vertices=vertices)
        self.rightHeadCushionGeom.attributes['a_position'] = self.rightHeadCushionGeom.attributes['vertices']
        rotation = np.array([[ 0.0, 0.0,  1.0],
                             [ 0.0, 1.0,  0.0],
                             [-1.0, 0.0,  0.0]], dtype=np.float32).T
        vertices = _vertices.copy()
        vertices[0, 3, 0] = -(0.5*W_playable - 0.6*SQRT2*W_cushion)
        vertices[1, 3, 0] = vertices[0, 3, 0]
        vertices.reshape(-1,3)[:] = rotation.dot(vertices.reshape(-1,3).T).T
        vertices.reshape(-1,3)[:,2] += 0.25 * self.length
        vertices.reshape(-1,3)[:,0] -= 0.5 * self.width - 0.5*W_cushion
        self.leftHeadCushionGeom = HexaPrimitive(vertices=vertices)
        self.leftHeadCushionGeom.attributes['a_position'] = self.leftHeadCushionGeom.attributes['vertices']
        vertices = self.rightHeadCushionGeom.attributes['vertices'].copy()
        vertices.reshape(-1,3)[:,2] *= -1
        self.rightFootCushionGeom = HexaPrimitive(vertices=vertices)
        self.rightFootCushionGeom.attributes['a_position'] = self.rightFootCushionGeom.attributes['vertices']
        vertices = self.leftHeadCushionGeom.attributes['vertices'].copy()
        vertices.reshape(-1,3)[:,2] *= -1
        self.leftFootCushionGeom = HexaPrimitive(vertices=vertices)
        self.leftFootCushionGeom.attributes['a_position'] = self.leftFootCushionGeom.attributes['vertices']
        self.cushionGeoms = [self.headCushionGeom, self.footCushionGeom,
                             self.leftHeadCushionGeom, self.rightHeadCushionGeom,
                             self.leftFootCushionGeom, self.rightFootCushionGeom]
        self.mesh = Mesh({surface_material: [surface],
                          cushion_material: self.cushionGeoms})
    def setup_balls(self, ball_radius, ball_colors, ball_positions, striped_balls=None, use_billboards=False,
                    technique=LAMBERT_TECHNIQUE):
        ball_materials = [Material(technique, values={'u_color': [(c&0xff0000) / 0xff0000,
                                                                  (c&0x00ff00) / 0x00ff00,
                                                                  (c&0x0000ff) / 0x0000ff,
                                                                  0.0]})
                          for c in ball_colors]
        ball_materials += ball_materials[1:-1]
        num_balls = len(ball_materials)
        sphere_prim = SpherePrimitive(radius=ball_radius)
        sphere_prim.attributes['a_position'] = sphere_prim.attributes['vertices']
        if striped_balls is None:
            striped_balls = set()
        else:
            stripe_prim = SpherePrimitive(radius=1.012*ball_radius, phiStart=0.0, phiLength=2*np.pi,
                                          thetaStart=np.pi/3, thetaLength=np.pi/3)
            stripe_prim.attributes['a_position'] = stripe_prim.attributes['vertices']
        ball_quaternions = np.zeros((num_balls, 4), dtype=np.float32)
        ball_quaternions[:,3] = 1
        if use_billboards:
            ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'ball.png')),
                                                 num_particles=num_balls,
                                                 scale=2*ball_radius,
                                                 color=np.array([[(c&0xff0000) / 0xff0000, (c&0x00ff00) / 0x00ff00, (c&0x0000ff) / 0x0000ff]
                                                                 for c in ball_colors], dtype=np.float32),
                                                 translate=ball_positions)
            ball_meshes = [ball_billboards]
        else:
            ball_meshes = [Mesh({material : [sphere_prim]})
                           if i not in striped_balls else
                           Mesh({ball_materials[0]: [sphere_prim],
                                 material: [stripe_prim]})
                           for i, material in enumerate(ball_materials)]
            for i, mesh in enumerate(ball_meshes):
                mesh.world_position[:] = ball_positions[i]
        return ball_meshes
