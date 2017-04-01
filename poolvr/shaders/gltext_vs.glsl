precision highp float;
uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform float advance;
attribute vec2 a_position;
attribute vec2 a_texcoord0;
varying vec2 v_texcoord0;
void main(void) {
  vec4 pos = u_modelviewMatrix * vec4(a_position.x + advance, a_position.y, 0.0, 1.0);
  v_texcoord0 = a_texcoord0;
  gl_Position = u_projectionMatrix * pos;
}
