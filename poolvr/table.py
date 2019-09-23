import os.path
import logging
_logger = logging.getLogger(__name__)
import numpy as np


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


INCH2METER = 0.0254
SQRT2 = np.sqrt(2)
DEG2RAD = np.pi/180


class PoolTable(object):
    BALL_COLORS = [0xddddde,
                   0xeeee00,
                   0x0000ee,
                   0xee0000,
                   0xee00ee,
                   0xee7700,
                   0x00ee00,
                   0xbb2244,
                   0x111111]
    BALL_COLORS = BALL_COLORS + BALL_COLORS[1:-1]
    def __init__(self,
                 L=100*INCH2METER,
                 H=29.25*INCH2METER,
                 W=None,
                 ell_1=0.5*INCH2METER,
                 ell_2=1.5*INCH2METER,
                 h=1.575*INCH2METER,
                 h_Q=1.625*INCH2METER,
                 r_P=0.1*INCH2METER,
                 delta_QP=0.25*INCH2METER,
                 delta_PT=0.25*INCH2METER,
                 a=1.75*INCH2METER,
                 A=60*DEG2RAD,
                 C=60*DEG2RAD,
                 # corner pocket params:
                 M_cp=5*INCH2METER,
                 T_cp=4.125*INCH2METER,
                 S_cp=1.75*INCH2METER,
                 D_cp=2.5*INCH2METER,
                 r_cpc=2.625*INCH2METER,
                 r_cpd=0.1875*INCH2METER,
                 # side pocket params:
                 M_sp=5.5*INCH2METER,
                 T_sp=4.625*INCH2METER,
                 S_sp=0,
                 D_sp=2*INCH2METER,
                 r_spc=2*INCH2METER,
                 r_spd=0.1875*INCH2METER,
                 width_rail=None,
                 ball_radius=1.125*INCH2METER,
                 num_balls=len(BALL_COLORS),
                 ball_colors=BALL_COLORS,
                 **kwargs):
        self.L = L
        self.H = H
        if W is None:
            W = 0.5 * L
        self.W = W
        self.ell_1 = ell_1
        self.ell_2 = ell_2
        self.w = ell_1 + ell_2
        self.h = h
        self.M_cp = M_cp
        self.T_cp = T_cp
        self.S_cp = S_cp
        self.D_cp = D_cp
        self.r_cpc = r_cpc
        self.r_cpd = r_cpd
        self.M_sp = M_sp
        self.T_sp = T_sp
        self.S_sp = S_sp
        self.D_sp = D_sp
        self.r_spc = r_spc
        self.r_spd = r_spd
        if width_rail is None:
            width_rail = 1.5 * self.w
        self.width_rail = width_rail
        self.ball_radius = ball_radius
        self.ball_diameter = 2*ball_radius
        self.num_balls = num_balls
        self.ball_colors = ball_colors
        self.pocket_positions = np.zeros((6, 3), dtype=np.float64)
        self.pocket_positions[:,1] = H
        self._almost_ball_radius = 0.999*ball_radius

    def is_position_in_bounds(self, r):
        """ r: position vector; R: ball radius """
        R = self._almost_ball_radius
        return  -0.5*self.W + R <= r[0] <= 0.5*self.W - R \
            and -0.5*self.L + R <= r[2] <= 0.5*self.L - R

    def is_position_near_pocket(self, r):
        """ r: position vector; R: ball radius """
        if r[0] < -0.5*self.W + self.M_cp/np.sqrt(2):
            if r[2] < -0.5*self.L + self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 0')
                return 0
            elif r[2] > 0.5*self.L - self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 1')
                return 1
        elif r[0] > 0.5*self.W - self.M_cp/np.sqrt(2):
            if r[2] < -0.5*self.L + self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 2')
                return 2
            elif r[2] > 0.5*self.L - self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 3')
                return 3

    def export_mesh(self,
                    surface_material=None,
                    surface_technique=None,
                    cushion_material=None,
                    cushion_technique=None,
                    rail_material=None,
                    rail_technique=None):
        from .gl_rendering import Mesh, Material
        from .gl_primitives import PlanePrimitive, HexaPrimitive, BoxPrimitive
        from .gl_techniques import EGA_TECHNIQUE
        if surface_technique is None:
            surface_technique = EGA_TECHNIQUE
        if cushion_technique is None:
            cushion_technique = EGA_TECHNIQUE
        if rail_technique is None:
            rail_technique = EGA_TECHNIQUE
        surface_material = surface_material or \
            Material(surface_technique,
                     values={'u_color': [0.0, 0xaa/0xff, 0.0, 0.0]})
        cushion_material = cushion_material or \
            Material(cushion_technique,
                     values={'u_color': [0x02/0xff, 0x88/0xff, 0x44/0xff, 0.0]})
        rail_material = rail_material or \
            Material(rail_technique,
                     values={'u_color': [0xdd/0xff, 0xa4/0xff, 0.0, 0.0]})
        length, width, w = self.L, self.W, self.w
        surface = PlanePrimitive(width=width+2*w, depth=length+2*w)
        surface.attributes['vertices'][:,1] = self.H
        surface.alias('vertices', 'a_position')
        L = self.L
        W = self.W
        H = self.H
        h = self.h
        w = self.w
        T_cp = self.T_cp
        M_cp = self.M_cp
        T_sp = self.T_sp
        M_sp = self.M_sp
        self.headCushionGeom = HexaPrimitive(vertices=np.array(
            [# bottom quad:
             [[-(0.5*W + w - T_cp/SQRT2),      0, -0.5*L - w],
              [    -(0.5*W - M_cp/SQRT2), 0.95*h, -0.5*L    ],
              [     (0.5*W - M_cp/SQRT2), 0.95*h, -0.5*L    ],
              [ (0.5*W + w - T_cp/SQRT2),      0, -0.5*L - w]],
             # top quad:
             [[-(0.5*W + w - T_cp/SQRT2), 1.3*h, -0.5*L - w],
              [    -(0.5*W - M_cp/SQRT2),     h, -0.5*L    ],
              [     (0.5*W - M_cp/SQRT2),     h, -0.5*L    ],
              [ (0.5*W + w - T_cp/SQRT2), 1.3*h, -0.5*L - w]]
            ], dtype=np.float32))
        self.headCushionGeom.attributes['vertices'].reshape(-1,3)[...,1] += H
        R = np.array([[-1, 0,  0],
                      [ 0, 1,  0],
                      [ 0, 0, -1]], dtype=np.float32)
        self.footCushionGeom = HexaPrimitive(vertices=np.dot(self.headCushionGeom.attributes['vertices'].reshape(-1,3), R.T).reshape(2,4,3))
        vs = self.headCushionGeom.attributes['vertices'].copy()
        vs[:,2] += 0.5*L
        R = np.array([[0, 0, -1],
                      [0, 1,  0],
                      [1, 0,  0]], dtype=np.float32)
        sideCushionGeom = HexaPrimitive(vertices=np.array(
            [#bottom quad:
             [[0.5*W + w,      0, -(0.5*L + w - T_cp/SQRT2)],
              [    0.5*W, 0.95*h,     -(0.5*L - M_cp/SQRT2)],
              [    0.5*W, 0.95*h,                 -0.5*M_sp],
              [0.5*W + w,      0,                 -0.5*T_sp]],
             # top quad:
             [[0.5*W + w, 1.3*h, -(0.5*L + w - T_cp/SQRT2)],
              [    0.5*W,     h,     -(0.5*L - M_cp/SQRT2)],
              [    0.5*W,     h,                 -0.5*M_sp],
              [0.5*W + w, 1.3*h,                 -0.5*T_sp]]
            ], dtype=np.float32))
        sideCushionGeom.attributes['vertices'][...,1] += H
        sideCushionGeoms = [sideCushionGeom]
        sideCushionGeoms.append(HexaPrimitive(vertices=sideCushionGeoms[-1].attributes['vertices'].copy()))
        sideCushionGeoms[-1].attributes['vertices'][...,2] *= -1
        sideCushionGeoms[-1].attributes['vertices'][0,:] = sideCushionGeoms[-1].attributes['vertices'][0,::-1]
        sideCushionGeoms[-1].attributes['vertices'][1,:] = sideCushionGeoms[-1].attributes['vertices'][1,::-1]
        sideCushionGeoms.append(HexaPrimitive(vertices=sideCushionGeoms[-1].attributes['vertices'].copy()))
        sideCushionGeoms[-1].attributes['vertices'][...,0] *= -1
        sideCushionGeoms[-1].attributes['vertices'][0,:] = sideCushionGeoms[-1].attributes['vertices'][0,::-1]
        sideCushionGeoms[-1].attributes['vertices'][1,:] = sideCushionGeoms[-1].attributes['vertices'][1,::-1]
        sideCushionGeoms.append(HexaPrimitive(vertices=sideCushionGeoms[-1].attributes['vertices'].copy()))
        sideCushionGeoms[-1].attributes['vertices'][...,2] *= -1
        sideCushionGeoms[-1].attributes['vertices'][0,:] = sideCushionGeoms[-1].attributes['vertices'][0,::-1]
        sideCushionGeoms[-1].attributes['vertices'][1,:] = sideCushionGeoms[-1].attributes['vertices'][1,::-1]
        self.cushionGeoms = [self.headCushionGeom,
                             self.footCushionGeom] + sideCushionGeoms
        headRailGeom = BoxPrimitive(W + 2*(w + self.width_rail), 1.3*h, self.width_rail)
        headRailGeom.attributes['vertices'][...,1] += H + 0.5*1.3*h
        headRailGeom.attributes['vertices'][...,2] -= 0.5*L + w + 0.5*self.width_rail
        footRailGeom = HexaPrimitive(vertices=headRailGeom.attributes['vertices'].copy().reshape(2,4,3))
        footRailGeom.attributes['vertices'][...,2] *= -1
        footRailGeom.attributes['vertices'][0,:] = footRailGeom.attributes['vertices'][0,::-1]
        footRailGeom.attributes['vertices'][1,:] = footRailGeom.attributes['vertices'][1,::-1]
        leftSideRailGeom = BoxPrimitive(self.width_rail, 1.3*h, L + 2*w)
        leftSideRailGeom.attributes['vertices'][...,0] -= 0.5*W + w + 0.5*self.width_rail
        leftSideRailGeom.attributes['vertices'][...,1] += H + 0.5*1.3*h
        rightSideRailGeom = BoxPrimitive(self.width_rail, 1.3*h, L + 2*w)
        rightSideRailGeom.attributes['vertices'][...,0] += 0.5*W + w + 0.5*self.width_rail
        rightSideRailGeom.attributes['vertices'][...,1] += H + 0.5*1.3*h
        self.railGeoms = [headRailGeom, footRailGeom, leftSideRailGeom, rightSideRailGeom]
        for geom in self.cushionGeoms + self.railGeoms:
            geom.alias('vertices', 'a_position')
        return Mesh({surface_material: [surface],
                     cushion_material: self.cushionGeoms,
                     rail_material   : self.railGeoms})

    def export_ball_meshes(self,
                           striped_balls=tuple(range(9,16)),
                           use_bb_particles=False,
                           technique=None):
        from .gl_rendering import Mesh, Material, Texture
        from .gl_primitives import SpherePrimitive, CirclePrimitive
        from .gl_techniques import EGA_TECHNIQUE
        from .billboards import BillboardParticles
        if technique is None:
            technique = EGA_TECHNIQUE
        num_balls = self.num_balls
        ball_quaternions = np.zeros((num_balls, 4), dtype=np.float32)
        ball_quaternions[:,3] = 1
        if use_bb_particles:
            ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'sphere_bb_alpha.png')),
                                                 Texture(os.path.join(TEXTURES_DIR, 'sphere_bb_normal.png')),
                                                 num_particles=num_balls,
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
                                              heightSegments=4,
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
                mesh.shadow_mesh.world_position[:] = self.H + 0.001
            return ball_meshes

    def calc_racked_positions(self, d=None,
                              out=None):
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float64)
        ball_radius = self.ball_radius
        if d is None:
            d = 0.04 * ball_radius
        length = self.L
        ball_diameter = 2*ball_radius
        # triangle racked:
        out[:,1] = self.H + ball_radius
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
