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
const vec3 ball_colors[16] = vec3[16](vec3(1.0, 1.0, 1.0),
				      vec3(1.0, 0.0, 0.0),
				      vec3(0.0, 1.0, 0.0),
				      vec3(0.0, 0.0, 1.0),
				      vec3(1.0, 1.0, 0.0),
				      vec3(0.0, 1.0, 1.0),
				      vec3(1.0, 0.0, 1.0),
				      vec3(0.0, 0.0, 0.0),
				      vec3(1.0, 1.0, 1.0),
				      vec3(1.0, 0.0, 0.0),
				      vec3(0.0, 1.0, 0.0),
				      vec3(0.0, 0.0, 1.0),
				      vec3(1.0, 1.0, 0.0),
				      vec3(0.0, 1.0, 1.0),
				      vec3(1.0, 0.0, 1.0),
				      vec3(0.0, 0.0, 0.0));

vec3 rotateByQuaternion(inout vec3 v, in vec4 q) {
  float ix =  q.w * v.x + q.y * v.z - q.z * v.y;
  float iy =  q.w * v.y + q.z * v.x - q.x * v.z;
  float iz =  q.w * v.z + q.x * v.y - q.y * v.x;
  float iw = - q.x * v.x - q.y * v.y - q.z * v.z;
  v.x = ix * q.w + iw * - q.x + iy * - q.z - iz * -q.y;
  v.y = iy * q.w + iw * - q.y + iz * - q.x - ix * -q.z;
  v.z = iz * q.w + iw * - q.z + ix * - q.y - iy * -q.x;
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

float oSphere( in vec3 pos, in vec3 nor, in vec4 sph )
{
  vec3 di = sph.xyz - pos;
  float l = length(di);
  return 1.0 - max(0.0,dot(nor,di/l))*sph.w*sph.w/(l*l);
}

float ssSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
  vec3 oc = sph.xyz - ro;
  float b = dot( oc, rd );

  float res = 1.0;
  if( b>0.0 )
    {
      float h = dot(oc,oc) - b*b - sph.w*sph.w;
      res = smoothstep( 0.0, 1.0, 12.0*h/b );
    }
  return res;
}

float sdSegment( vec2 a, vec2 b, vec2 p )
{
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );

  return length( pa - ba*h );
}

float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
  const float N = 10.0;
  vec2 w = max(abs(ddx), abs(ddy)) + 0.01;
  vec2 a = p + 0.5*w;
  vec2 b = p - 0.5*w;
  vec2 i = (floor(a)+min(fract(a)*N,1.0)-
            floor(b)-min(fract(b)*N,1.0))/(N*w);
  return (1.0-i.x)*(1.0-i.y);
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

  vec4 sph1 = vec4(-2.0, 1.0,0.0,1.1);
  vec4 sph2 = vec4( 3.0, 1.5,1.0,1.2);
  vec4 sph3 = vec4( 1.0,-1.0,1.0,1.3);

  float tmin = 10000.0;
  vec3  nor = vec3(0.0);
  float occ = 1.0;
  vec3  pos = vec3(0.0);

  vec3 sur = vec3(1.0);

  vec4 sph = vec4(0.0, 0.0, 0.0, ball_radius);

  float h, ss;
  for (int i = 0; i < 16; i++) {
    sph.xyz = ball_positions[i];
    h = iSphere( ro, rd, sph );
    if( h>0.0 && h<tmin ) {
      tmin = h;
      pos = ro + h*rd;
      nor = normalize(pos-sph.xyz);
      // sur = vec3(1.0,0.7,0.2)*smoothstep(-0.6,-0.2,sin(20.0*(pos.x-sph.x)));
      vec3 sv = vec3(1.0, 0.0, 0.0);
      rotateByQuaternion(sv, ball_quaternions[i]);
      ss = i > 8 ? smoothstep(-0.6,-0.2,sin(200.0*(dot(pos.xyz-sph.xyz, sv)))) : 1.0;
      sur = ball_colors[i]*ss;
      // occ = oSphere( pos, nor, sph2 ) * oSphere( pos, nor, sph3 );
      occ = 1.0;
      for (int j = 0; j < 16; j++) {
        if (i == j) {
          continue;
        }
        sph.xyz = ball_positions[j];
        occ *= oSphere( pos, nor, sph );
      }
    }
  }

  // float h = iSphere( ro, rd, sph1 );
  // if( h>0.0 && h<tmin )
  //   {
  //     tmin = h;
  //     pos = ro + h*rd;
  //     nor = normalize(pos-sph1.xyz);
  //     occ = oSphere( pos, nor, sph2 ) * oSphere( pos, nor, sph3 );
  //     sur = vec3(1.0,0.7,0.2)*smoothstep(-0.6,-0.2,sin(20.0*(pos.x-sph1.x)));
  //   }
  // h = iSphere( ro, rd, sph2 );
  // if( h>0.0 && h<tmin )
  //   {
  //     tmin = h;
  //     pos = ro + h*rd;
  //     nor = normalize(pos-sph2.xyz);
  //     occ = oSphere( pos, nor, sph1 ) * oSphere( pos, nor, sph3 );
  //     sur = vec3(0.7,1.0,0.2)*smoothstep(-0.6,-0.2,sin(20.0*(pos.z-sph2.z)));
  //   }
  // h = iSphere( ro, rd, sph3 );
  // if( h>0.0 && h<tmin )
  //   {
  //     tmin = h;
  //     pos = ro + h*rd;
  //     nor = normalize(pos-sph3.xyz);
  //     occ = oSphere( pos, nor, sph1 ) * oSphere( pos, nor, sph2 );
  //     sur = vec3(1.0,0.2,0.2)*smoothstep(-0.6,-0.2,sin(20.0*(pos.y-sph3.y)));
  //   }
  h = (-2.0-ro.y)/rd.y;
  if( h>0.0 && h<tmin )
    {
      tmin = h;
      pos = ro + h*rd;
      nor = vec3(0.0,1.0,0.0);
      occ = 1.0; //oSphere( pos, nor, sph1 ) * oSphere( pos, nor, sph2 ) * oSphere( pos, nor, sph3 );
      sur = vec3(1.0)*gridTextureGradBox( pos.xz, dFdx(pos.xz), dFdy(pos.xz) );
    }

  vec3 col = vec3(0.0);

  if( tmin<100.0 )
    {
      pos = ro + tmin*rd;
      col = vec3(1.0);

      vec3 lig = normalize( vec3(2.0,1.4,-1.0) );
      float sha = 1.0;

      // sha *= ssSphere( pos, lig, sph1 );
      // sha *= ssSphere( pos, lig, sph2 );
      // sha *= ssSphere( pos, lig, sph3 );

      float ndl = clamp( dot(nor,lig), 0.0, 1.0 );
      col = occ*(0.5+0.5*nor.y)*vec3(0.2,0.3,0.4) + sha*vec3(1.0,0.9,0.8)*ndl + sha*vec3(1.5)*ndl*pow( clamp(dot(normalize(-rd+lig),nor),0.0,1.0), 16.0 );
      col *= sur;

      col *= exp( -0.25*(max(0.0,tmin-3.0)) );

    }

  col = pow( col, vec3(0.45) );

  fragColor = vec4( col, 1.0 );
}


void main() { mainImage(gl_FragColor, gl_FragCoord.xy); } // added by jz
