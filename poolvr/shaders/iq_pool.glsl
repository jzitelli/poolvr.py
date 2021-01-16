// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

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
const vec2 bounds = vec2(W_2, L_2);
const float table_height = 29.25*0.0254;
const vec3 table_color = vec3(0.0, float(0xaa)/0xff, 0.0);
const vec3 lig = normalize( vec3(0.0,8.0,0.0) );
uniform int iGlobalTime = 0;
uniform vec3 iMouse = vec3(0.0,0.0,0.0);

#define eps 0.001

float hash1( float n )
{
    return fract(sin(n)*43758.5453123);
}

vec2 hash2( float n )
{
    return fract(sin(vec2(n,n+1.0))*vec2(43758.5453123,22578.1459123));
}

vec3 hash3( float n )
{
    return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(43758.5453123,22578.1459123,19642.3490423));
}

float distanceToSegment( vec2 a, vec2 b, vec2 p )
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );

    return length( pa - ba*h );
}

vec3 nSphere( in vec3 pos, in vec4 sph )
{
    return (pos-sph.xyz)/sph.w;
}

float iSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h<0.0 ) return -1.0;

    h = sqrt(h);

    float t1 = -b - h;
    float t2 = -b + h;

    if( t1<eps && t2<eps )
        return -1.0;

    if( t1<eps )
        return t2;

    return t1;
}


float sSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    float res = 1.0;

    vec3 oc = sph.xyz - ro;
    float b = dot( oc, rd );

    if( b<0.0 )
    {
        res = 1.0;
    }
    else
    {
        float h = sqrt( dot(oc,oc) - b*b ) - sph.w;

        res = clamp( 16.0 * h / b, 0.0, 1.0 );
    }
    return res;
}


vec3 nPlane( in vec3 ro, in vec4 obj )
{
    return obj.xyz;
}


float iPlane( in vec3 ro, in vec3 rd, in vec4 pla, in vec2 bounds )
{
    float t = (-pla.w - dot(pla.xyz,ro)) / dot( pla.xyz, rd );
    if (t > 0) {
      vec3 x = ro + t*rd;
      if (abs(x.x) < bounds.x && abs(x.z) < bounds.y) {
        return t;
      }
    }
    return -1.0;
}

float sPlane( in vec3 ro, in vec3 rd, in vec4 pla )
{
    float t = (-pla.w - dot(pla.xyz,ro)) / dot( pla.xyz, rd );

    if( t<0.0 ) return 1.0;
    return 0.0;
}

vec4 sphere[16];

float iCylinder( in vec3 ro, in vec3 rd, in mat4 cue_world_matrix, in float h, in float rad ) {
  // transform to cylinder local-coordinates:
  mat3 cue_rot_inv = transpose(mat3(cue_world_matrix));
  vec3 ro_loc = cue_rot_inv * (ro - cue_world_matrix[3].xyz);
  vec3 rd_loc = cue_rot_inv * rd;
  float A = dot(rd_loc.xz, rd_loc.xz);
  float B = dot(ro_loc.xz, rd_loc.xz);
  float C = dot(ro_loc.xz, ro_loc.xz) - rad*rad;
  float DD = B*B - A*C;
  if (DD <= 0.0) {
    return -1.0;
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

vec2 intersect( in vec3 ro, in vec3 rd, out vec3 uvw )
{
    vec2 res = vec2( 1e20, -1.0 );
    float t;

    t = iCylinder( ro, rd, cue_world_matrix, cue_length, cue_radius ); if( t>eps && t<res.x ) { res = vec2( t, 20.0 ); uvw = ro+rd*t; }
    t = iPlane( ro, rd, vec4(0.0,1.0,0.0,-table_height), bounds ); if( t>eps && t<res.x ) { res = vec2( t, 0.0 ); uvw = ro+rd*t; }
    // t = iSphere( ro, rd, sphere[0] ); if( t>eps && t<res.x ) { res = vec2( t, 1.0 ); uvw = ro+rd*t - vec3(sphere[0].xyz); }
    // t = iSphere( ro, rd, sphere[0] ); if( t>eps && t<res.x ) { res = vec2( t, 1.0 ); uvw = ro+rd*t - vec3(sphere[0].xyz); }
    // t = iSphere( ro, rd, sphere[1] ); if( t>eps && t<res.x ) { res = vec2( t, 2.0 ); uvw = ro+rd*t - vec3(sphere[1].xyz); }
    // t = iSphere( ro, rd, sphere[2] ); if( t>eps && t<res.x ) { res = vec2( t, 3.0 ); uvw = ro+rd*t - vec3(sphere[2].xyz); }
    // t = iSphere( ro, rd, sphere[3] ); if( t>eps && t<res.x ) { res = vec2( t, 4.0 ); uvw = ro+rd*t - vec3(sphere[3].xyz); }

    for (int is = 0; is < 16; is++) {
      t = iSphere( ro, rd, sphere[is] ); if( t>eps && t<res.x ) { res = vec2( t, float(is)+1.0 ); uvw = (ro+rd*t - sphere[is].xyz)/ball_radius; }
    }
    return res;
}

vec3 calcNormal( in vec3 pos, in float mat )
{
    vec3 nor = vec3(0.0);

    // if( mat<4.5 ) nor = nSphere( pos, sphere[3] );
    // if( mat<3.5 ) nor = nSphere( pos, sphere[2] );
    // if( mat<2.5 ) nor = nSphere( pos, sphere[1] );
    // if( mat<1.5 ) nor = nSphere( pos, sphere[0] );
    if( mat<=16.0 && mat>=1.0 ) { nor = nSphere( pos, sphere[int(mat)-1] ); }
    else if( mat<0.5 ) nor = nPlane(  pos, vec4(0.0,1.0,0.0,1.0) );
    return nor;
}

float shadow( in vec3 ro, in vec3 rd )
{
#if 1
    vec2 res = vec2( 1e20, 1.0 );

    float t = 0.0;
    t = iCylinder( ro, rd, cue_world_matrix, cue_length, cue_radius ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    // t = iSphere( ro, rd, sphere[0] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    // t = iSphere( ro, rd, sphere[1] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    // t = iSphere( ro, rd, sphere[2] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    // t = iSphere( ro, rd, sphere[3] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    for (int is = 0; is < 16; is++) {
      t = iSphere( ro, rd, sphere[is] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    }
    t = iPlane(  ro, rd, vec4(0.0,1.0,0.0,-table_height), bounds  ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );

    return res.y;
#else

    float res = 1.0;

    float t = 0.0;
    t = iCylinder( ro, rd, cue_world_matrix, cue_length, cue_radius ); res = min( t, res );
    t = sSphere( ro, rd, sphere[0] ); res = min( t, res );
    t = sSphere( ro, rd, sphere[1] ); res = min( t, res );
    t = sSphere( ro, rd, sphere[2] ); res = min( t, res );
    t = sSphere( ro, rd, sphere[3] ); res = min( t, res );
    t = sPlane(  ro, rd, vec4(0.0,1.0,0.0,1.0) ); res = min( t, res );

    return res;


#endif
}

vec3 doEnvironment( in vec3 rd )
{
    //return 24.0*pow( texture( iChannel1, rd ).xyz, vec3(2.2) );
    return 24.0*pow( vec3(0.5), vec3(2.2) );
}

vec3 doEnvironmentBlurred( in vec3 rd )
{
    //return 9.0*pow( texture( iChannel2, rd ).xyz, vec3(2.2) );
    return 9.0*pow( vec3(0.5), vec3(2.2) );
}


vec4 doMaterial( in vec3 pos, in vec3 nor, float ma )
{
    vec4 mate = vec4(0.0);
    //if( ma >= 1.0) { mate = vec4(ball_colors[int(ma)-1], 1.0); }
    if( ma>=4.5) { mate = vec4(ma/32.0,ma/32.0,0.20,1.0)*1.2; }
    if( ma == 12.0 ) { mate = vec4(0.20,0.20,0.00,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.x) ) );
                   float d1 = distanceToSegment( vec2(0.22,0.12), vec2(-0.22,0.12), pos.yz );
                   float d2 = distanceToSegment( vec2(0.22,-0.12), vec2(-0.22,-0.12), pos.yz );
                   float d = min( d1, d2 );
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }
    if( ma == 5.0 ) { mate = vec4(0.20,0.00,0.00,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.x) ) + smoothstep( 0.55, 0.56, abs(pos.y) ) );
                   float d1 = distanceToSegment( vec2(0.22,0.0), vec2(-0.22,0.0), pos.yz );
                   float d2 = distanceToSegment( vec2(0.22,0.0), vec2( -0.07,-0.2), pos.yz*vec2(1.0,-sign(pos.x)) );
                   float d3 = distanceToSegment( vec2(-0.07,-0.2), vec2(-0.07,0.04), pos.yz*vec2(1.0,-sign(pos.x)) );
                   float d = min(d1,min(d2,d3));
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }
    if( ma == 2.0 ) { mate = vec4(0.00,0.10,0.20,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.z) ) );
                   float d = distanceToSegment( vec2(0.22,0.0), vec2(-0.22,0.0), pos.yx );
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }

    if( ma<4.5 ) { mate = vec4(0.30,0.25,0.20,1.0)*1.25; }
    if( ma<0.5 ) { mate = vec4(0.01,0.20,0.03,0.0)*1.25;
                   //mate.xyz *= 0.78 + 0.22*texture( iChannel3, 0.1*pos.xz ).x;
    		   mate.xyz *= 0.78 + 0.22*0.5;
                 }

    return mate;
}

vec3 rotateByQuaternion(inout vec3 v, in vec4 q) {
  vec3 t = 2*cross(q.xyz, v);
  v += q.w * t + cross(q.xyz, t);
  return v;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 q = fragCoord.xy / iResolution.xy;
    vec2 m = vec2(0.5);
    if( iMouse.z>0.0 ) m = iMouse.xy/iResolution.xy;

    //-----------------------------------------------------
    // animate
    //-----------------------------------------------------
    float an = 1.3 + 0.1*iGlobalTime - 7.0*m.x;
    // sphere[0] = vec4( 1.0,1.0, 1.0,1.0);
    // sphere[1] = vec4(-4.0,1.0, 0.0,1.0);
    // sphere[2] = vec4( 0.0,1.0, 3.0,1.0);
    // sphere[3] = vec4( 5.0,1.0, -2.0,1.0);
    // sphere[0] = vec4(ball_positions[0], ball_radius);
    // sphere[1] = vec4(ball_positions[1], ball_radius);
    // sphere[2] = vec4(ball_positions[2], ball_radius);
    // sphere[3] = vec4(ball_positions[3], ball_radius);
    for (int is = 0; is < 16; is++) {
      sphere[is] = vec4(ball_positions[is], ball_radius);
    }
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


    // montecarlo
    vec3 tot = vec3(0.0);


    for( int a=0; a<1; a++ )
    {
        vec3 col = vec3(0.0);
        vec3 uvw = vec3(0.0);

        // raymarch
        vec2 tmat = intersect( ro, rd, uvw );

	if( tmat.y == 20.0) {
	  col = vec3(0.7, 0.3, 0.05);
	}
	else if( tmat.y>-0.5 )
        {
            vec3 pos = ro + tmat.x*rd;
            vec3 nor = calcNormal( pos, tmat.y );
            vec3 ref = reflect( rd, nor );

            // material
	    // vec4 q = vec4(ball_quaternions[int(tmat.y)-1]);
	    // q.xyz *= -1.0;
	    // uvw = rotateByQuaternion(uvw, q);
            vec4 mate = doMaterial( uvw, nor, tmat.y );

            // lighting
            vec3 lin = vec3(0.0);

            // diffuse
            #if 0
            vec3  ru  = normalize( cross( nor, vec3(0.0,1.0,1.0) ) );
            vec3  rv  = normalize( cross( ru, nor ) );
            #else
            // see http://orbit.dtu.dk/fedora/objects/orbit:113874/datastreams/file_75b66578-222e-4c7d-abdf-f7e255100209/content
            // (link provided by nimitz)
            vec3 tc = vec3( 1.0+nor.z-nor.xy*nor.xy, -nor.x*nor.y)/(1.0+nor.z);
            vec3 ru = vec3( tc.xz, -nor.x );
            vec3 rv = vec3( tc.zy, -nor.y );
            #endif

            for( int j=1; j<2; j++ )
            {
                //vec2  aa = hash2( rrr.x + float(j)*203.1 + float(a)*13.713 );
                vec2  aa = hash2( float(j)*203.1 + float(a)*13.713 );
                float ra = sqrt(aa.y);
                float rx = ra*cos(6.2831*aa.x);
                float ry = ra*sin(6.2831*aa.x);
                float rz = sqrt( 1.0-aa.y );
                vec3  rr = vec3( rx*ru + ry*rv + rz*nor );
                lin += shadow( pos, rr ) * doEnvironmentBlurred( rr );
            }
            lin /= 1.5;
            // bounce
            lin += 1.5*clamp(0.3-0.7*nor.y,0.0,1.0)*vec3(0.0,0.2,00);
            // rim
            lin *= 1.0 + 6.0*mate.xyz*mate.w*pow( clamp( 1.0 + dot(nor,rd), 0.0, 1.0 ), 2.0 );;
            // specular
            //float fre = 0.04 + 4.0*pow( clamp( 1.0 + dot(nor,rd), 0.0, 1.0 ), 5.0 );
            //lin += 1.0*doEnvironment(ref ).xyz * mate.w * fre * shadow( pos, ref );
            //lin += 1.0*doEnvironment(ref ).xyz * mate.w * fre * step( 0.0, ref.y );

            // light-material interaction
            col = mate.xyz * lin;
        }

        tot += col;
    }
    //tot /= 2.0;

    // gamma
    tot = pow( clamp(tot,0.0,1.0), vec3(0.45) );

    //tot = tot*0.5 + 0.5*tot*tot*(3.0-2.0*tot);
    //tot *= 0.5 + 0.5*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.15 );

    fragColor = vec4( tot, 1.0 );
}


void main() { mainImage(gl_FragColor, gl_FragCoord.xy); } // added by jz
