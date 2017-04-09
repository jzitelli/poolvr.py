precision highp float;
uniform samplerCube u_map;
varying vec3 v_texcoord;
void main(void) {
  gl_FragColor = textureCube(u_map, v_texcoord);
}
