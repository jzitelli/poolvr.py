precision highp float;

uniform mat4 u_modelview;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec2 uv;
attribute vec3 translate;
attribute vec3 color;

varying vec2 vUv;
varying vec3 v_color;

void main() {
  vec4 mvPosition = u_modelview * vec4( translate, 1.0 );
  vec3 z = normalize(-mvPosition.xyz);
  vec3 x = normalize(vec3(z.z, 0.0, -z.x));
  vec3 y = cross(z, x);
  mvPosition.xyz += position.x * x + position.y * y + position.z * z;
  vUv = uv;
  v_color = color;
  gl_Position = u_projection * mvPosition;
}
