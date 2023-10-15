import os.path
import logging
_logger = logging.getLogger(__name__)
import numpy as np


INCH2METER = 0.0254
SQRT2 = np.sqrt(2)
DEG2RAD = np.pi/180


class PoolTable(object):
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
                 # M_cp=5*INCH2METER,
                 M_cp=7.5*INCH2METER,
                 # T_cp=4.125*INCH2METER,
                 T_cp=4*INCH2METER,
                 S_cp=1.75*INCH2METER,
                 D_cp=2.5*INCH2METER,
                 r_cpc=2.625*INCH2METER,
                 r_cpd=0.1875*INCH2METER,
                 # side pocket params:
                 M_sp=6.5*INCH2METER,
                 T_sp=4.25*INCH2METER,
                 S_sp=0,
                 D_sp=1.25*INCH2METER,
                 r_spc=2*INCH2METER,
                 r_spd=0.1875*INCH2METER,
                 width_rail=None,
                 ball_radius=1.125*INCH2METER,
                 num_balls=16,
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
        self._almost_ball_radius = 0.999*ball_radius
        corners = np.empty((24,2))
        w = 0.5 * W
        l = 0.5 * L
        b = self.w
        T_s, M_s, T_c, M_c = self.T_sp, self.M_sp, self.T_cp, self.M_cp
        corners[0] = -(w + b) + T_c/SQRT2, -(l + b)
        corners[1] = -w + M_c/SQRT2, -l
        corners[2] = w - M_c/SQRT2, -l
        corners[3] = w + b - T_c/SQRT2, -(l + b)
        corners[4] = w + b, -(l + b) + T_c/SQRT2,
        corners[5] = w, -l + M_c/SQRT2
        corners[6] = w, -M_s/2
        corners[7] = w + b, -T_s/2
        corners[8] = corners[7,0], -corners[7,1]
        corners[9] = corners[6,0], -corners[6,1]
        corners[10] = corners[5,0], -corners[5,1]
        corners[11] = corners[4,0], -corners[4,1]
        corners[12] = corners[3,0], -corners[3,1]
        corners[13] = corners[2,0], -corners[2,1]
        corners[14] = corners[1,0], -corners[1,1]
        corners[15] = corners[0,0], -corners[0,1]
        corners[16] = -corners[11,0], corners[11,1]
        corners[17] = -corners[10,0], corners[10,1]
        corners[18] = -corners[9,0], corners[9,1]
        corners[19] = -corners[8,0], corners[8,1]
        corners[20] = -corners[7,0], corners[7,1]
        corners[21] = -corners[6,0], corners[6,1]
        corners[22] = -corners[5,0], corners[5,1]
        corners[23] = -corners[4,0], corners[4,1]
        self._corners = corners
        self.pocket_positions = np.zeros((6, 3), dtype=np.float64)
        self.pocket_positions[:,1] = H
        D_s, D_c = self.D_sp, self.D_cp
        R_c = 2*self.ball_radius
        R_s = 2.75*self.ball_radius
        self.R_c, self.R_s = R_c, R_s
        self.pocket_positions[0,::2] = 0.5 * (corners[22] + corners[1])  + (D_c + R_c) * np.array([-np.cos(np.pi/4), -np.sin(np.pi/4)])
        self.pocket_positions[1,::2] = 0.5 * (corners[2] + corners[5])   + (D_c + R_c) * np.array([ np.cos(np.pi/4), -np.sin(np.pi/4)])
        self.pocket_positions[2,::2] = 0.5 * (corners[6] + corners[9])   + (D_s + R_s) * np.array([1.0, 0.0])
        self.pocket_positions[3,::2] = 0.5 * (corners[10] + corners[13]) + (D_c + R_c) * np.array([ np.cos(np.pi/4), np.sin(np.pi/4)])
        self.pocket_positions[4,::2] = 0.5 * (corners[14] + corners[17]) + (D_c + R_c) * np.array([-np.cos(np.pi/4), np.sin(np.pi/4)])
        self.pocket_positions[5,::2] = 0.5 * (corners[18] + corners[21]) + (D_s + R_s) * np.array([-1.0, 0.0])

    def corner_to_pocket(self, i_c):
        return (i_c + 2) % 24 // 4

    def pocket_to_corner(self, i_p):
        return i_p * 4 - 2

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
        from .gl_primitives import PlanePrimitive, HexaPrimitive, BoxPrimitive, CylinderPrimitive
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
        H = self.H
        h = self.h
        w = self.w
        corners = self._corners
        cushionGeoms = [HexaPrimitive(vertices=np.array(
                            [# bottom quad:
                             [[corners[i+0,0],      H, corners[i+0,1]],
                              [corners[i+1,0], 0.95*h+H, corners[i+1,1]],
                              [corners[i+2,0], 0.95*h+H, corners[i+2,1]],
                              [corners[i+3,0],      H, corners[i+3,1]]],
                             # top quad:
                             [[corners[i+0,0], 1.3*h+H, corners[i+0,1]],
                              [corners[i+1,0],     h+H, corners[i+1,1]],
                              [corners[i+2,0],     h+H, corners[i+2,1]],
                              [corners[i+3,0], 1.3*h+H, corners[i+3,1]]]
                            ], dtype=np.float32)) for i in (0, 4, 8, 12, 16, 20)]
        self.cushionGeoms = cushionGeoms
        w_r = self.width_rail
        railGeoms = [
            HexaPrimitive(vertices=np.array(
                [# bottom quad:
                 [[corners[0,0],      H, corners[0,1]],
                  [corners[3,0],      H, corners[3,1]],
                  [corners[3,0],      H, corners[3,1] - w_r],
                  [corners[0,0],      H, corners[0,1] - w_r]],
                 # top quad:
                 [[corners[0,0],     1.3*h+H, corners[0,1]],
                  [corners[3,0],     1.3*h+H, corners[3,1]],
                  [corners[3,0],     1.3*h+H, corners[3,1] - w_r],
                  [corners[0,0],     1.3*h+H, corners[0,1] - w_r]]
                ], dtype=np.float32)),
            HexaPrimitive(vertices=np.array(
                [# bottom quad:
                 [[corners[4,0],      H, corners[4,1]],
                  [corners[7,0],      H, corners[7,1]],
                  [corners[7,0] + w_r,      H, corners[7,1]],
                  [corners[4,0] + w_r,      H, corners[4,1]]],
                 # top quad:
                 [[corners[4,0],     1.3*h+H, corners[4,1]],
                  [corners[7,0],     1.3*h+H, corners[7,1]],
                  [corners[7,0] + w_r,     1.3*h+H, corners[7,1]],
                  [corners[4,0] + w_r,     1.3*h+H, corners[4,1]]]
                ], dtype=np.float32)),
            HexaPrimitive(vertices=np.array(
                [# bottom quad:
                 [[corners[8,0],      H, corners[8,1]],
                  [corners[11,0],      H, corners[11,1]],
                  [corners[11,0] + w_r,      H, corners[11,1]],
                  [corners[8,0] + w_r,      H, corners[8,1]]],
                 # top quad:
                 [[corners[8,0],     1.3*h+H, corners[8,1]],
                  [corners[11,0],     1.3*h+H, corners[11,1]],
                  [corners[11,0] + w_r,     1.3*h+H, corners[11,1]],
                  [corners[8,0] + w_r,     1.3*h+H, corners[8,1]]]
                ], dtype=np.float32)),
        ]
        railGeoms2 = [HexaPrimitive(vertices=g.attributes['vertices'].copy().reshape(2,4,3)) for g in railGeoms]
        for g in railGeoms2:
            g.attributes['vertices'][...,::2] *= -1
        self.railGeoms = railGeoms + railGeoms2
        self.pocketGeoms = []
        for i_p in range(6):
            pocket_prim = CylinderPrimitive(self.R_s if i_p in (2,5) else self.R_c, 0.01)
            pocket_prim.attributes['vertices'][...,1] += H
            pocket_prim.attributes['vertices'][...,::2] += self.pocket_positions[i_p,::2]
            self.pocketGeoms.append(pocket_prim)
        for geom in self.cushionGeoms + self.railGeoms + self.pocketGeoms:
            geom.alias('vertices', 'a_position')
        return Mesh({surface_material: [surface],
                     cushion_material: self.cushionGeoms,
                     rail_material   : self.railGeoms,
                     Material(EGA_TECHNIQUE, values={'u_color': [0.0,0.0,0.0,0.0]}) : self.pocketGeoms})

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
