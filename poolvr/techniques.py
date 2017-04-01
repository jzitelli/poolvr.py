import pkgutil
import OpenGL.GL as gl
from poolvr.gl_rendering import Program, Technique


EGA_TECHNIQUE = Technique(Program(pkgutil.get_data('poolvr', 'shaders/ega_vs.glsl').decode(),
                                  pkgutil.get_data('poolvr', 'shaders/ega_fs.glsl').decode()),
                          attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                          uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                    'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                    'u_color': {'type': gl.GL_FLOAT_VEC4,
                                                'value': [1.0, 0.0, 0.0, 0.0]}})


LAMBERT_TECHNIQUE = Technique(Program(pkgutil.get_data('poolvr', 'shaders/lambert_vs.glsl').decode(),
                                      pkgutil.get_data('poolvr', 'shaders/lambert_fs.glsl').decode()),
                              attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                              uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                        'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                        'u_color': {'type': gl.GL_FLOAT_VEC4,
                                                    'value': [0.0, 1.0, 1.0, 0.0]},
                                        'u_lightpos': {'type': gl.GL_FLOAT_VEC3,
                                                       'value': [1.0, 15.0, 1.5]}})
