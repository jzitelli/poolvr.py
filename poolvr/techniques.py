import OpenGL.GL as gl
from .gl_rendering import Program, Technique


EGA_TECHNIQUE = Technique(Program("""precision highp float;
uniform mat4 u_modelview;
uniform mat4 u_projection;
attribute vec3 a_position;
void main(void) {
  gl_Position = u_projection * (u_modelview * vec4(a_position, 1.0));
}""",
                                """precision highp float;
uniform vec4 u_color;
void main(void) {
  gl_FragColor = u_color;
}"""),
                          attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                          uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                    'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                    'u_color': {'type': gl.GL_FLOAT_VEC4,
                                                'value': [0.5, 0.5, 0.0, 0.0]}})


LAMBERT_TECHNIQUE = Technique(Program("""precision highp float;
uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform vec3 u_lightpos = vec3(0.0, 10.0, -2.0);
attribute vec3 a_position;
varying vec3 v_position;
varying vec3 v_lightpos;
void main(void) {
  v_lightpos = (u_modelview * vec4(u_lightpos, 1.0)).xyz;
  vec4 view_pos = u_modelview * vec4(a_position, 1.0);
  v_position = view_pos.xyz;
  gl_Position = u_projection * view_pos;
}""",
                              """precision highp float;
uniform vec4 u_color;
varying vec3 v_position;
varying vec3 v_lightpos;
void main(void) {
  vec3 dpdx = dFdx(v_position);
  vec3 dpdy = dFdy(v_position);
  float incFactor = clamp(dot(normalize(v_lightpos - v_position), normalize(cross(dpdx, dpdy))), 0.0, 0.9);
  gl_FragColor = (0.1 + incFactor) * u_color;
}"""),
                              attributes={'a_position': {'type': gl.GL_FLOAT_VEC3}},
                              uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                        'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                        'u_color': {'type': gl.GL_FLOAT_VEC4,
                                                    'value': [0.5, 0.0, 0.5, 0.0]},
                                        'u_lightpos': {'type': gl.GL_FLOAT_VEC3,
                                                       'value': [0.0, 10.0, -2.0]}})
