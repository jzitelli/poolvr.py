precision highp float;

uniform sampler2D map;
uniform sampler2D u_normal;
uniform vec3 u_light = vec3(1.0, -2.0, -0.5);
varying vec2 vUv;
varying vec3 v_color;

void main() {
  vec4 diffuseColor = texture2D(map, vUv);
  if (diffuseColor.w < 0.25) discard;
  gl_FragColor = vec4(v_color, 1.0) * (0.2 + clamp(dot(u_light, (2.0*texture2D(u_normal, vUv).xyz - 1.0)), 0.0, 1.0)) * diffuseColor;

  // vec4 clipPos = cameraToClipMatrix * vec4(cameraPos, 1.0);
  // float ndcDepth = clipPos.z / clipPos.w;
  // gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
}
