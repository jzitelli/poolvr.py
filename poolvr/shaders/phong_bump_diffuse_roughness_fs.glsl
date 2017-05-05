#version 120
precision highp float;

uniform mat3 u_modelview_inverse_transpose;

uniform vec4 u_light_position = vec4(0.1, 1.0, 0.9, 1.0);
uniform vec3 u_light_intensity = vec3(0.8, 0.7, 0.6);

uniform sampler2D u_diffuse_map;
uniform sampler2D u_normal_map;
uniform sampler2D u_roughness_map;

uniform vec3 u_Ka = vec3(0.02, 0.03, 0.07); // ambient reflectivity
uniform vec3 u_Ks = vec3(0.2, 0.3, 0.32); // specular reflectivity

varying vec2 v_texcoord;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;

vec3 ads() {
  vec3 normal_tex = u_modelview_inverse_transpose * normalize(texture2D(u_normal_map, v_texcoord).rgb);
  vec3 normal = normal_tex.z * v_normal + normal_tex.x * v_tangent + normal_tex.y * cross(v_normal, v_tangent);
  vec3 n = u_modelview_inverse_transpose * normal;
  vec3 s = normalize(u_light_position.xyz - v_position.xyz);
  vec3 v = normalize(-v_position.xyz);
  vec3 r = reflect(-s, n);
  float r_dot_v = max(dot(r, v), 0.0);
  return u_Ka + u_light_intensity * (texture2D(u_diffuse_map, v_texcoord).rgb * max(dot(s, n), 0.0) + u_Ks * pow(r_dot_v, texture2D(u_roughness_map, v_texcoord).r));
}

void main() {
  gl_FragColor = vec4(ads(), 1.0);
}
