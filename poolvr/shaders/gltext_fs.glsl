precision highp float;
uniform sampler2D u_fonttex;
uniform vec4 u_color;
varying vec2 v_texcoord0;
void main(void) {
  vec4 tex = vec4(u_color.rgb, texture2D(u_fonttex, v_texcoord0).r);
  gl_FragColor = tex;
}
