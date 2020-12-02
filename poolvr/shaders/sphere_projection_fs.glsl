// The MIT License
// Copyright © 2014 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// Analytic projection of a sphere to screen pixels.

// Spheres in world space become ellipses when projected to the camera view plane. In fact, these
// ellipses can be analytically determined from the camera parameters and the sphere geometry,
// such that their exact position, orientation and surface area can be compunted. This means that,
// given a sphere and a camera and buffer resolution, there is an analytical formula that
// provides the amount of pixels covered by a sphere in the image. This can be very useful for
// implementing LOD for objects based on their size in screen (think of trees, vegetation, characters
// or any other such complex object).

// This shaders implements this formula, and provides too the center and axes of the ellipse

// More info, here: http://www.iquilezles.org/www/articles/sphereproj/sphereproj.htm

// ---------------------------------------------------------------------------------------------


#version 410 core
uniform mat4 u_camera = mat4(1.0);
uniform vec4 u_projection_lrbt = vec4(1.0);
uniform vec2 iResolution = vec2(1024.0, 768.0);
uniform float u_znear;
uniform vec3[16] ball_positions;
uniform vec4[16] ball_quaternions;
uniform float ball_radius = 1.125*0.0254;
uniform mat4 cue_world_matrix = mat4(1.0);
uniform float cue_radius;
uniform float cue_length;
const vec3 ball_colors[16] = vec3[16](
  vec3(0.8666667,0.8666667,0.87058824),
  vec3(0.93333334,0.93333334,0.0),
  vec3(0.0,0.0,0.93333334),
  vec3(0.93333334,0.0,0.0),
  vec3(0.93333334,0.0,0.93333334),
  vec3(0.93333334,0.46666667,0.0),
  vec3(0.0,0.93333334,0.0),
  vec3(0.73333335,0.13333334,0.26666668),
  vec3(0.06666667,0.06666667,0.06666667),
  vec3(0.93333334,0.93333334,0.0),
  vec3(0.0,0.0,0.93333334),
  vec3(0.93333334,0.0,0.0),
  vec3(0.93333334,0.0,0.93333334),
  vec3(0.93333334,0.46666667,0.0),
  vec3(0.0,0.93333334,0.0),
  vec3(0.73333335,0.13333334,0.26666668)
);
const float L_2 = 50*0.0254;
const float W_2 = 25*0.0254;
const float table_height = 29.25*0.0254;
const vec3 table_color = vec3(0.0, float(0xaa)/0xff, 0.0);
const vec3 lig = normalize( vec3(0.0,8.0,0.0) );

vec3 rotateByQuaternion(inout vec3 v, in vec4 q) {
  v += 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
  return v;
}

float iSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
  vec3 oc = ro - sph.xyz;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - sph.w*sph.w;
  float h = b*b - c;
  if( h<0.0 ) return -1.0;
  return -b - sqrt( h );
}

float iCylinder( in vec3 ro, in vec3 rd, in mat4 cue_world_matrix, in float h, in float rad ) {
  // transform to cylinder local-coordinates:
  mat3 cue_rot_inv = transpose(mat3(cue_world_matrix));
  vec3 ro_loc = cue_rot_inv * (ro - cue_world_matrix[3].xyz);
  vec3 rd_loc = cue_rot_inv * rd;
  float A = dot(rd_loc.xz, rd_loc.xz);
  float B = dot(ro_loc.xz, rd_loc.xz);
  float C = dot(ro_loc.xz, ro_loc.xz) - rad*rad;
  float DD = B*B - A*C;
  if (DD < 0.0) {
    return -1.0;
  } else if (DD == 0.0) {
    return -B/A;
  }
  float D = sqrt(DD);
  float tm = (-B-D)/A;
  float tp = (-B+D)/A;
  if ( tm < tp && tm > 0.0 && abs(ro_loc.y + rd_loc.y*tm) < 0.5*h ) {
    return tm;
  } else if ( tp > 0.0 && abs(ro_loc.y + rd_loc.y*tp) < 0.5*h ) {
    return tp;
  }
  return -1.0;
}

float oSphere( in vec3 pos, in vec3 nor, in vec4 sph ) {
  vec3 di = sph.xyz - pos;
  float l = length(di);
  return 1.0 - max(0.0,dot(nor,di/l))*sph.w*sph.w/(l*l);
}

float ssSphere( in vec3 ro, in vec3 rd, in vec4 sph ) {
  vec3 oc = sph.xyz - ro;
  float b = dot( oc, rd );
  float res = 1.0;
  if( b>0.0 ) {
    float h = dot(oc,oc) - b*b - sph.w*sph.w;
    res = smoothstep( 0.0, 1.0, 12.0*h/b );
  }
  return res;
}

float sdCylinder( vec3 p, vec3 c ) {
  return length(p.xz-c.xy)-c.z;
}

float sdCappedCylinder( vec3 p, vec2 h ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 p = fragCoord.xy / iResolution.xy;
  vec3 uu = normalize(u_camera[0].xyz);
  vec3 vv = normalize(u_camera[1].xyz);
  vec3 ww = normalize(-u_camera[2].xyz);
  vec3 ro = u_camera[3].xyz;

  vec3 rd = normalize(
		      ((1.0-p.x) * u_projection_lrbt.x + p.x * u_projection_lrbt.y) * uu
		    + ((1.0-p.y) * u_projection_lrbt.z + p.y * u_projection_lrbt.w) * vv
		    + ww
		      );

  float tmin = 10000.0;
  vec3 nor = vec3(0.0);
  float occ = 1.0;
  vec3 pos = vec3(0.0);
  vec3 sur = vec3(1.0);
  vec4 sph = vec4(0.0, 0.0, 0.0, ball_radius);
  int imin = -1;
  float h, ss;

  for (int i = 0; i < 16; i++) {
    sph.xyz = ball_positions[i];
    h = iSphere( ro, rd, sph );
    if( h>0.0 && h<tmin ) {
      tmin = h;
      imin = i;
      pos = ro + h*rd;
      nor = normalize(pos-sph.xyz);
      sur = ball_colors[i];
    }
  }

  h = (table_height-ro.y)/rd.y;
  if ( h>0.0 && h<tmin && abs(ro.x+h*rd.x) < W_2 && abs(ro.z+h*rd.z) < L_2) {
    pos = ro + h*rd;
    tmin = h;
    imin = -1;
    nor = vec3(0.0, 1.0, 0.0);
    occ = 1.0;
    for (int j = 0; j < 16; j++) {
      sph.xyz = ball_positions[j];
      occ *= oSphere( pos, nor, sph );
    }
    float h2 = iCylinder( pos, nor, cue_world_matrix, cue_length, cue_radius );
    if (h2 > 0.0) {
      occ *= 0.5;
      sur = 0.4 * table_color;
    } else {
      sur = table_color;
    }
  }

  h = iCylinder( ro, rd, cue_world_matrix, cue_length, cue_radius );
  if ( h > 0.0 && h < tmin ) {
    pos = ro + h*rd;
    tmin = h;
    imin = -1;
    nor = pos - cue_world_matrix[3].xyz;
    nor -= dot(nor,cue_world_matrix[1].xyz) * cue_world_matrix[1].xyz;
    nor = normalize(nor);
    sur = vec3(0.7, 0.3, 0.05);
  }

  if (tmin == 10000.0) {
    discard;
  }

  if (imin > -1) {
    if (imin > 8) {
      vec3 sv = vec3(1.0, 0.0, 0.0);
      rotateByQuaternion(sv, ball_quaternions[imin]);
      sph.xyz = ball_positions[imin];
      ss = smoothstep(-0.1,0.1,cos(0.4*321.0*(dot(pos.xyz-sph.xyz, sv))));
      sur += vec3(ss*ss);
    }
    occ = 1.0;
    for (int j = 0; j < 16; j++) {
      if (imin == j) continue;
      sph.xyz = ball_positions[j];
      occ *= oSphere( pos, nor, sph );
    }
  }

  vec3 col = vec3(0.0);

  if( tmin < 400.0 ) {
    pos = ro + tmin*rd;
    col = vec3(1.0);
    float sha = 1.0;
    for (int i = 0; i < 16; i++) {
      if (imin == i) continue;
      sph.xyz = ball_positions[i];
      sha *= ssSphere( pos, lig, sph );
    }
    float ndl = clamp( dot(nor,lig), 0.0, 1.0 );
    col = occ*(0.5+0.5*nor.y)*vec3(0.04, 0.06, 0.08)
      + sha*vec3(1.0,0.9,0.8)*ndl
      + sha*vec3(1.5)*ndl*pow( clamp(dot(normalize(-rd+lig),nor),0.0,1.0), 16.0 );
    col *= sur;
    col *= exp( -0.25*(max(0.0,tmin-3.0)) );
  }

  col = pow( col, vec3(0.45) );
  fragColor = vec4( col, 1.0 );

}


void main() { mainImage(gl_FragColor, gl_FragCoord.xy); } // added by jz
