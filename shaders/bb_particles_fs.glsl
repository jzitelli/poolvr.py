precision highp float;

uniform sampler2D map;

varying vec2 vUv;
varying vec3 v_color;

void main() {
  vec4 diffuseColor = texture2D(map, vUv);
  gl_FragColor = vec4(v_color, 1.0) * diffuseColor;
  if (diffuseColor.w < 0.25) discard;
}
