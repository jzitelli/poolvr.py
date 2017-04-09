precision highp float;
uniform mat4 u_modelview;
uniform mat4 u_projection;
attribute vec3 a_position;
varying vec3 v_texcoord;
void main(void) {
  v_texcoord = a_position;
  vec4 view_pos = u_modelview * vec4(a_position, 1.0);
  gl_Position = u_projection * view_pos;
}
