precision highp float;

uniform vec4 u_color;

varying vec3 v_position;
varying vec3 v_lightpos;

void main(void) {
  vec3 dpdx = dFdx(v_position);
  vec3 dpdy = dFdy(v_position);
  float incFactor = clamp(dot(normalize(v_lightpos - v_position), normalize(cross(dpdx, dpdy))), 0.0, 0.9);
  gl_FragColor = (0.1 + incFactor) * u_color;
}
