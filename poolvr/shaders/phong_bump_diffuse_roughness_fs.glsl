#version 120
precision highp float;

uniform vec4 u_light_position = vec4(0.1, 1.0, 0.9, 1.0);
uniform vec3 u_light_intensity = vec3(0.7, 0.7, 0.7);

uniform sampler2D u_diffuse_map;
uniform sampler2D u_normal_map;
uniform sampler2D u_roughness_map;

uniform vec3 u_Ka = vec3(0.01, 0.01, 0.01); // ambient reflectivity
uniform vec3 u_Ks = vec3(0.5, 0.5, 0.5); // specular reflectivity

varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_texcoord;

vec3 ads() {
  vec3 normal_tex = 2.0 * texture2D(u_normal_map, v_texcoord).rgb - 1.0;
  vec3 n = sqrt(1.0 - normal_tex.r*normal_tex.r - normal_tex.g*normal_tex.g) * v_normal \
    + normal_tex.r * v_tangent + normal_tex.g * v_bitangent;
  vec3 s = normalize(u_light_position.xyz - v_position.xyz);
  vec3 v = normalize(-v_position.xyz);
  vec3 r = reflect(-s, n);
  float r_dot_v = max(dot(r, v), 0.0);
  return u_Ka + u_light_intensity * (texture2D(u_diffuse_map, v_texcoord).rgb * max(dot(s, n), 0.0) + u_Ks * pow(r_dot_v, texture2D(u_roughness_map, v_texcoord).r));
}

void main() {
  gl_FragColor = vec4(ads(), 1.0);
}
