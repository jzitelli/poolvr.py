import os.path
import logging
_logger = logging.getLogger(__name__)
import numpy as np
cimport numpy as np

from .gl_rendering import Mesh, Material, Texture
from .gl_primitives import PlanePrimitive, HexaPrimitive, SpherePrimitive, CirclePrimitive, BoxPrimitive
from .gl_techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE
from .billboards import BillboardParticles


cdef double INCH2METER = 0.0254
cdef double SQRT2 = np.sqrt(2)
cdef list BALL_COLORS = [0xddddde,
                         0xeeee00,
                         0x0000ee,
                         0xee0000,
                         0xee00ee,
                         0xee7700,
                         0x00ee00,
                         0xbb2244,
                         0x111111] + \
                         [0xddddde,
                          0xeeee00,
                          0x0000ee,
                          0xee0000,
                          0xee00ee,
                          0xee7700,
                          0x00ee00,
                          0xbb2244,
                          0x111111][1:-1]


cdef class PoolTable(object):
    cdef public double length
    cdef public double height
    cdef public double width
    cdef public double W_playable
    cdef public double L_playable
    cdef public double W_cushion
    cdef public double mouth_size
    cdef public double throat_size
    cdef public double H_cushion
    cdef public double W_nose
    cdef public double H_nose
    cdef public double width_rail
    cdef public double H_rail
    cdef public double ball_radius
    cdef public double ball_diameter
    cdef public int num_balls
    cdef public list ball_colors
    cdef public double _almost_ball_radius
    def __init__(self,
                 double length=2.34,
                 double height=0.77,
                 double width=0.5*2.34,
                 double W_cushion=2*INCH2METER,
                 double mouth_size=5*INCH2METER,
                 double throat_size=4.125*INCH2METER,
                 double H_cushion=0.635*2.25*INCH2METER,
                 double W_nose=0.05*2*INCH2METER,
                 double H_nose=0.5*0.635*2.25*INCH2METER,
                 double width_rail=1.52*INCH2METER,
                 double H_rail=1.25*0.635*2.25*INCH2METER,
                 double ball_radius=1.125*INCH2METER,
                 int num_balls=len(BALL_COLORS),
                 list ball_colors=BALL_COLORS,
                 **kwargs):
        self.length = length
        self.height = height
        self.width = width
        self.width_rail = width_rail
        self.H_rail = H_rail
        self.ball_radius = ball_radius
        self.ball_diameter = 2*ball_radius
        self.W_cushion = W_cushion
        self.W_nose = W_nose
        self.H_nose = H_nose
        self.W_playable = width - 2*W_cushion
        self.L_playable = length - 2*W_cushion
        self.num_balls = num_balls
        self.ball_colors = ball_colors
        self.throat_size = throat_size
        self.mouth_size = mouth_size
        self._almost_ball_radius = 0.999*ball_radius

    def is_position_in_bounds(self, np.ndarray r):
        """ r: position vector; R: ball radius """
        R = self._almost_ball_radius
        return  -0.5*self.W_playable <= r[0] - R            \
            and             r[0] + R <= 0.5*self.W_playable \
            and -0.5*self.L_playable <= r[2] - R            \
            and             r[2] + R <= 0.5*self.L_playable

    def is_position_near_pocket(self, np.ndarray r):
        if r[0] < -0.5*self.W_playable + self.mouth_size/SQRT2:
            if r[2] < -0.5*self.L_playable + self.mouth_size/SQRT2:
                _logger.info('corner pocket 0')
                return 0
            elif r[2] > 0.5*self.L_playable - self.mouth_size/SQRT2:
                _logger.info('corner pocket 1')
                return 1
        elif r[0] > 0.5*self.W_playable - self.mouth_size/SQRT2:
            if r[2] < -0.5*self.L_playable + self.mouth_size/SQRT2:
                _logger.info('corner pocket 2')
                return 2
            elif r[2] > 0.5*self.L_playable - self.mouth_size/SQRT2:
                _logger.info('corner pocket 3')
                return 3

    def export_mesh(self,
                    surface_material=None,
                    surface_technique=LAMBERT_TECHNIQUE,
                    cushion_material=None,
                    cushion_technique=LAMBERT_TECHNIQUE,
                    rail_material=None,
                    rail_technique=EGA_TECHNIQUE):
        surface_material = surface_material or \
            Material(surface_technique,
                     values={'u_color': [0.0, 0xaa/0xff, 0.0, 0.0]})
        cushion_material = cushion_material or \
            Material(cushion_technique,
                     values={'u_color': [0x02/0xff, 0x88/0xff, 0x44/0xff, 0.0]})
        rail_material = rail_material or \
            Material(rail_technique,
                     values={'u_color': [0xdd/0xff, 0xa4/0xff, 0.0, 0.0]})
        length, width, W_cushion = self.length, self.width, self.W_cushion
        surface = PlanePrimitive(width=width, depth=length)
        surface.attributes['vertices'][:,1] = self.height
        surface.alias('vertices', 'a_position')
        H_cushion = 0.82*2*self.ball_radius
        W_playable = self.W_playable
        L_playable = self.L_playable
        T = self.throat_size
        M = self.mouth_size
        self.headCushionGeom = HexaPrimitive(vertices=np.array([
            # bottom quad:
            [[-0.5*W_playable + 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 0.4*W_cushion,       0.0,           0.5*W_cushion],
             [ 0.5*W_playable - 1.2*SQRT2*W_cushion, 0.57*2*self.ball_radius, -0.5*W_cushion + self.W_nose],
             [-0.5*W_playable + 1.2*SQRT2*W_cushion, 0.57*2*self.ball_radius, -0.5*W_cushion + self.W_nose]],
            # top quad:
            [[-0.5*W_playable + (T/SQRT2 - W_cushion), self.H_rail,  0.5*W_cushion],
             [ 0.5*W_playable - (T/SQRT2 - W_cushion), self.H_rail,  0.5*W_cushion],
             [ 0.5*W_playable - M/SQRT2,               H_cushion,   -0.5*W_cushion],
             [-0.5*W_playable + M/SQRT2,               H_cushion,   -0.5*W_cushion]]], dtype=np.float32))
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,1] += self.height
        _vertices = self.headCushionGeom.attributes['vertices'].copy()
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[:,2] += 0.5 * self.length - 0.5*W_cushion
        vertices = _vertices.copy()
        vertices.reshape(-1,3)[:,2] *= -1
        vertices.reshape(-1,3)[:,2] -= 0.5 * L_playable
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
                                         self.H_rail,
                                         self.width_rail)
        self.railGeoms = [self.headRailGeom]
        self.headRailMesh = Mesh({rail_material: [self.headRailGeom]})
        for geom in self.cushionGeoms + self.railGeoms:
            geom.alias('vertices', 'a_position')
        return Mesh({surface_material: [surface],
                     cushion_material: self.cushionGeoms,
                     rail_material   : self.railGeoms})

    def export_ball_meshes(self,
                           striped_balls=tuple(range(9,16)),
                           use_bb_particles=False,
                           technique=LAMBERT_TECHNIQUE):
        num_balls = self.num_balls
        ball_quaternions = np.zeros((num_balls, 4), dtype=np.float32)
        ball_quaternions[:,3] = 1
        if use_bb_particles:
            ball_billboards = BillboardParticles(num_particles=num_balls,
                                                 scale=2*self.ball_radius / 0.975,
                                                 color=np.array([[(c & 0xff0000) / 0xff0000,
                                                                  (c & 0x00ff00) / 0x00ff00,
                                                                  (c & 0x0000ff) / 0x0000ff]
                                                                 for c in self.ball_colors],
                                                                dtype=np.float32))
            return [ball_billboards]
        else:
            ball_materials = [Material(technique, values={'u_color': [(c & 0xff0000) / 0xff0000,
                                                                      (c & 0x00ff00) / 0x00ff00,
                                                                      (c & 0x0000ff) / 0x0000ff, 0.0]})
                              for c in self.ball_colors]
            sphere_prim = SpherePrimitive(radius=self.ball_radius)
            sphere_prim.attributes['a_position'] = sphere_prim.attributes['vertices']
            if striped_balls is None:
                striped_balls = set()
            else:
                stripe_prim = SpherePrimitive(radius=1.001*self.ball_radius,
                                              heightSegments=8,
                                              thetaStart=np.pi/3, thetaLength=np.pi/3)
                stripe_prim.attributes['a_position'] = stripe_prim.attributes['vertices']
            circle_prim = CirclePrimitive(radius=self.ball_radius, num_radial=16)
            circle_prim.attributes['a_position'] = circle_prim.attributes['vertices']
            shadow_material = Material(EGA_TECHNIQUE, values={'u_color': [0.01, 0.03, 0.001, 0.0]})
            ball_meshes = [Mesh({material        : [sphere_prim]})
                           if i not in striped_balls else
                           Mesh({ball_materials[0] : [sphere_prim],
                                 material          : [stripe_prim]})
                           for i, material in enumerate(ball_materials)]
            ball_shadow_meshes = [Mesh({shadow_material : [circle_prim]})
                                  for i in range(num_balls)]
            for i, mesh in enumerate(ball_meshes):
                mesh.shadow_mesh = ball_shadow_meshes[i]
                mesh.shadow_mesh.world_position[:] = self.height + 0.001
            return ball_meshes

    def calc_racked_positions(self, d=None,
                              out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float64)
        ball_radius = self.ball_radius
        if d is None:
            d = 0.04 * ball_radius
        length = self.length
        ball_diameter = 2*ball_radius
        # triangle racked:
        out[:,1] = self.height + ball_radius
        side_length = 4 * (self.ball_diameter + d)
        x_positions = np.concatenate([np.linspace(0,                        0.5 * side_length,                         5),
                                      np.linspace(-0.5*(ball_diameter + d), 0.5 * side_length - (ball_diameter + d),   4),
                                      np.linspace(-(ball_diameter + d),     0.5 * side_length - 2*(ball_diameter + d), 3),
                                      np.linspace(-1.5*(ball_diameter + d), 0.5 * side_length - 3*(ball_diameter + d), 2),
                                      np.array([-2*(ball_diameter + d)])])
        z_positions = np.concatenate([np.linspace(0,                                    np.sqrt(3)/2 * side_length, 5),
                                      np.linspace(0.5*np.sqrt(3) * (ball_diameter + d), np.sqrt(3)/2 * side_length, 4),
                                      np.linspace(np.sqrt(3) * (ball_diameter + d),     np.sqrt(3)/2 * side_length, 3),
                                      np.linspace(1.5*np.sqrt(3) * (ball_diameter + d), np.sqrt(3)/2 * side_length, 2),
                                      np.array([np.sqrt(3)/2 * side_length])])
        z_positions *= -1
        z_positions -= length / 8
        out[1:,0] = x_positions
        out[1:,2] = z_positions
        # cue ball at head spot:
        out[0,0] = 0.0
        out[0,2] = 0.25 * length
        return out