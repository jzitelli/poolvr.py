#version 120
precision highp float;

uniform vec4 u_light_position = vec4(0.1, 1.0, 0.9, 1.0);
uniform vec3 u_light_intensity = vec3(0.8, 0.7, 0.6);

uniform sampler2D u_diffuse_map;
uniform sampler2D u_bump_map;
uniform sampler2D u_roughness_map;

uniform vec3 u_Ka = vec3(0.02, 0.03, 0.07); // ambient reflectivity
uniform vec3 u_Ks = vec3(0.2, 0.3, 0.32); // specular reflectivity

varying vec2 v_texcoord;
varying vec3 v_position;
varying vec3 v_normal;

vec3 ads() {
  vec3 n = normalize(v_normal);
  vec3 s = normalize(u_light_position.xyz - v_position.xyz);
  vec3 v = normalize(-v_position.xyz);
  vec3 r = reflect(-s, n);
  float r_dot_v = max(dot(r, v), 0.0);
  //vec4 roughness = texture2D(u_roughness_map, v_texcoord);
  //return u_light_intensity * (u_Ka + texture2D(u_diffuse_map, v_texcoord).rgb * max(dot(s, n), 0.0) + u_Ks.r * pow(r_dot_v, roughness.r) + u_Ks.g * pow(r_dot_v, roughness.g) + u_Ks.b * pow(r_dot_v, roughness.b));
  float roughness = 30;//0.05; //texture2D(u_roughness_map, v_texcoord);
  return u_Ka + u_light_intensity * (texture2D(u_diffuse_map, v_texcoord).rgb * max(dot(s, n), 0.0) + u_Ks * pow(r_dot_v, texture2D(u_roughness_map, v_texcoord).r));
}

void main() {
  gl_FragColor = vec4(ads(), 1.0);
}
