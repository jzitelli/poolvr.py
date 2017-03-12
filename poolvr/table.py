import pkgutil
import OpenGL.GL as gl


from .gl_rendering import Mesh, Material, Technique, Program
from .primitives import BoxPrimitive, PlanePrimitive, HexaPrimitive


INCH2METER = 0.0254


EGA_TECHNIQUE = Technique(Program(pkgutil.get_data('poolvr', 'shaders/ega_vs.glsl').decode(),
                                  pkgutil.get_data('poolvr', 'shaders/ega_fs.glsl').decode()),
                          attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                          uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                    'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                    'u_color': {'type': gl.GL_FLOAT_VEC4}})


LAMBERT_TECHNIQUE = Technique(Program(pkgutil.get_data('poolvr', 'shaders/lambert_vs.glsl').decode(),
                                      pkgutil.get_data('poolvr', 'shaders/lambert_fs.glsl').decode()),
                              attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                              uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                        'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                        'u_color': {'type': gl.GL_FLOAT_VEC4},
                                        'u_lightpos': {'type': gl.GL_FLOAT_VEC3,
                                                       'value': [0.0, 10.0, -2.0]}})


class PoolTable(object):
    def __init__(self,
                 length=2.34,
                 height=0.77,
                 width=None,
                 width_rail=2*INCH2METER):
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
