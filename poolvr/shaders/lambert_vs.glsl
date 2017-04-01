precision highp float;

uniform mat4 u_view;
uniform mat4 u_modelview;
// uniform mat4 u_modelview_inverse;
uniform mat4 u_projection;
uniform vec3 u_lightpos = vec3(3.0, 10.0, -2.0);

attribute vec3 a_position;

varying vec3 v_position;
varying vec3 v_lightpos;

void main(void) {
  v_lightpos = (u_view * vec4(u_lightpos, 1.0)).xyz;
  vec4 view_pos = u_modelview * vec4(a_position, 1.0);
  v_position = view_pos.xyz;
  gl_Position = u_projection * view_pos;
}
