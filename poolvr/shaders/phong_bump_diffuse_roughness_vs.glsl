#version 120
precision highp float;

uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform mat3 u_modelview_inverse_transpose;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_tangent;
attribute vec2 a_texcoord;

varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_texcoord;

void main() {
  vec4 position = u_modelview * vec4(a_position, 1.0);
  v_position = position.xyz;
  gl_Position = u_projection * position;
  v_normal = normalize(u_modelview_inverse_transpose * a_normal);
  v_tangent = normalize(u_modelview_inverse_transpose * a_tangent);
  v_bitangent = normalize(u_modelview_inverse_transpose * cross(a_normal, a_tangent));
  v_texcoord = a_texcoord;
}
