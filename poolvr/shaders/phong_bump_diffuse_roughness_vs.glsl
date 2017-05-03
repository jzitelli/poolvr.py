#version 120
precision highp float;

uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform mat3 u_modelview_inverse_transpose;

attribute vec2 a_texcoord;
attribute vec3 a_position;
attribute vec3 a_normal;

varying vec3 v_position;
varying vec3 v_normal;
varying vec2 v_texcoord;

void main() {
  vec4 position = u_modelview * vec4(a_position, 1.0);
  v_position = position.xyz;
  gl_Position = u_projection * position;
  v_normal = u_modelview_inverse_transpose * a_normal;
  v_texcoord = a_texcoord;
}
