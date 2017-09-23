import os.path
import numpy as np

from .gl_rendering import Mesh, Material, Texture
from .primitives import PlanePrimitive, HexaPrimitive, SpherePrimitive, CirclePrimitive, BoxPrimitive
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
                 W_cushion=2*INCH2METER,
                 H_cushion=0.635*2.25*INCH2METER,
                 width_rail=None,
                 H_rail=None,
                 ball_diameter=2.25*INCH2METER,
                 **kwargs):
        self.length = length
        self.height = height
        if width is None:
            width = 0.5 * length
        self.width = width
        if width_rail is None:
            width_rail = 1.5 * W_cushion
        self.width_rail = width_rail
        if H_rail is None:
            H_rail = 1.25 * H_cushion
        self.H_rail = H_rail
        W_nose = 0.05 * W_cushion
        self.W_nose = W_nose
        H_nose = 0.5 * H_cushion
        self.H_nose = H_nose
        surface_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 0xaa/0xff, 0.0, 0.0]})
        cushion_material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0x02/0xff, 0x88/0xff, 0x44/0xff, 0.0]})
        surface = PlanePrimitive(width=width, depth=length)
        surface.attributes['vertices'][:,1] = height
        surface.attributes['a_position'] = surface.attributes['vertices']
        W_playable = width - 2*W_cushion
        H_cushion = 0.82*ball_diameter
        self.headCushionGeom = HexaPrimitive(vertices=np.array([
            # bottom quad:
            [[-0.5*W_playable + 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 1.2*SQRT2*W_cushion, 0.57*ball_diameter, -0.5*W_cushion + W_nose],
             [-0.5*W_playable + 1.2*SQRT2*W_cushion, 0.57*ball_diameter, -0.5*W_cushion + W_nose]],
            # top quad:
            [[-0.5*W_playable + 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       H_rail,     0.5*W_cushion],
             [ 0.5*W_playable - 1.2*SQRT2*W_cushion, H_cushion, -0.5*W_cushion],
             [-0.5*W_playable + 1.2*SQRT2*W_cushion, H_cushion, -0.5*W_cushion]]], dtype=np.float32))
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,1] += self.height
        _vertices = self.headCushionGeom.attributes['vertices'].copy()
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,2] += 0.5 * self.length - 0.5*W_cushion
        vertices = _vertices.copy()
        vertices.reshape(-1,3)[:,2] *= -1
        vertices.reshape(-1,3)[:,2] -= 0.5 * self.length - 0.5*W_cushion
        self.footCushionGeom = HexaPrimitive(vertices=vertices)
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
        vertices = self.rightHeadCushionGeom.attributes['vertices'].copy()
        vertices.reshape(-1,3)[:,2] *= -1
        self.rightFootCushionGeom = HexaPrimitive(vertices=vertices)
        vertices = self.leftHeadCushionGeom.attributes['vertices'].copy()
        vertices.reshape(-1,3)[:,2] *= -1
        self.leftFootCushionGeom = HexaPrimitive(vertices=vertices)
        self.cushionGeoms = [self.headCushionGeom, self.footCushionGeom,
                             self.leftHeadCushionGeom, self.rightHeadCushionGeom,
                             self.leftFootCushionGeom, self.rightFootCushionGeom]
        self.headRailGeom = BoxPrimitive(W_playable - 2 * 0.4 * W_cushion,
                                         H_rail,
                                         width_rail)
        self.railGeoms = [self.headRailGeom]
        rail_material = Material(EGA_TECHNIQUE, values={'u_color': [0xdd/0xff, 0xa4/0xff, 0.0, 0.0]})
        self.headRailMesh = Mesh({rail_material: [self.headRailGeom]})
        for geom in self.cushionGeoms + self.railGeoms:
            geom.alias('vertices', 'a_position')
        self.mesh = Mesh({surface_material: [surface],
                          cushion_material: self.cushionGeoms,
                          rail_material   : self.railGeoms})


    def setup_balls(self, ball_radius, ball_colors, ball_positions, striped_balls=None,
                    use_bb_particles=False,
                    technique=LAMBERT_TECHNIQUE):
        ball_colors = list(ball_colors) + ball_colors[1:-1]
        num_balls = len(ball_colors)
        ball_quaternions = np.zeros((num_balls, 4), dtype=np.float32)
        ball_quaternions[:,3] = 1
        if use_bb_particles:
            ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'sphere_bb_alpha.png')),
                                                 Texture(os.path.join(TEXTURES_DIR, 'sphere_bb_normal.png')),
                                                 num_particles=num_balls,
                                                 scale=2*ball_radius / 0.975,
                                                 color=np.array([[(c&0xff0000) / 0xff0000, (c&0x00ff00) / 0x00ff00, (c&0x0000ff) / 0x0000ff]
                                                                 for c in ball_colors + ball_colors[1:-1]], dtype=np.float32),
                                                 translate=ball_positions)
            ball_meshes = [ball_billboards]
        else:
            ball_materials = [Material(technique, values={'u_color': [(c&0xff0000) / 0xff0000,
                                                                      (c&0x00ff00) / 0x00ff00,
                                                                      (c&0x0000ff) / 0x0000ff,
                                                                      0.0]})
                              for c in ball_colors]
            sphere_prim = SpherePrimitive(radius=ball_radius)
            sphere_prim.attributes['a_position'] = sphere_prim.attributes['vertices']
            if striped_balls is None:
                striped_balls = set()
            else:
                stripe_prim = SpherePrimitive(radius=1.012*ball_radius, phiStart=0.0, phiLength=2*np.pi,
                                              thetaStart=np.pi/3, thetaLength=np.pi/3)
                stripe_prim.attributes['a_position'] = stripe_prim.attributes['vertices']
            circle_prim = CirclePrimitive(radius=ball_radius, num_radial=16)
            circle_prim.attributes['a_position'] = circle_prim.attributes['vertices']
            circle_prim.attributes['a_position'][:,1] -= 0.981*ball_radius
            shadow_material = Material(EGA_TECHNIQUE, values={'u_color': [0.01, 0.03, 0.001, 1.0]})
            ball_meshes = [Mesh({material        : [sphere_prim]})
                           if i not in striped_balls else
                           Mesh({ball_materials[0] : [sphere_prim],
                                 material          : [stripe_prim]})
                           for i, material in enumerate(ball_materials)]
            ball_shadow_meshes = [Mesh({shadow_material : [circle_prim]})
                                  for i in range(num_balls)]
            for i, mesh in enumerate(ball_meshes):
                mesh.world_position[:] = ball_positions[i]
                mesh.shadow_mesh = ball_shadow_meshes[i]
            for i, mesh in enumerate(ball_shadow_meshes):
                mesh.world_position[:] = ball_positions[i]
        return ball_meshes
