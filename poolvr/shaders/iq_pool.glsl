// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

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



    return res;
}


vec3 nPlane( in vec3 ro, in vec4 obj )
{
    return obj.xyz;
}


float iPlane( in vec3 ro, in vec3 rd, in vec4 pla )
{
    return (-pla.w - dot(pla.xyz,ro)) / dot( pla.xyz, rd );
}

float sPlane( in vec3 ro, in vec3 rd, in vec4 pla )
{
    float t = (-pla.w - dot(pla.xyz,ro)) / dot( pla.xyz, rd );

    if( t<0.0 ) return 1.0;
    return 0.0;
}

vec4 sphere[4];

vec2 intersect( in vec3 ro, in vec3 rd, out vec3 uvw )
{
    vec2 res = vec2( 1e20, -1.0 );
    float t;

    t = iPlane( ro, rd, vec4(0.0,1.0,0.0,0.0) ); if( t>eps && t<res.x ) { res = vec2( t, 0.0 ); uvw = ro+rd*t; }

    t = iSphere( ro, rd, sphere[0] ); if( t>eps && t<res.x ) { res = vec2( t, 1.0 ); uvw = ro+rd*t - vec3(sphere[0].xyz); }
    t = iSphere( ro, rd, sphere[1] ); if( t>eps && t<res.x ) { res = vec2( t, 2.0 ); uvw = ro+rd*t - vec3(sphere[1].xyz); }
    t = iSphere( ro, rd, sphere[2] ); if( t>eps && t<res.x ) { res = vec2( t, 3.0 ); uvw = ro+rd*t - vec3(sphere[2].xyz); }
    t = iSphere( ro, rd, sphere[3] ); if( t>eps && t<res.x ) { res = vec2( t, 4.0 ); uvw = ro+rd*t - vec3(sphere[3].xyz); }

    return res;
}

vec3 calcNormal( in vec3 pos, in float mat )
{
    vec3 nor = vec3(0.0);

    if( mat<4.5 ) nor = nSphere( pos, sphere[3] );
    if( mat<3.5 ) nor = nSphere( pos, sphere[2] );
    if( mat<2.5 ) nor = nSphere( pos, sphere[1] );
    if( mat<1.5 ) nor = nSphere( pos, sphere[0] );
    if( mat<0.5 ) nor = nPlane(  pos, vec4(0.0,1.0,0.0,1.0) );

    return nor;
}

float shadow( in vec3 ro, in vec3 rd )
{
#if 0
    vec2 res = vec2( 1e20, 1.0 );

    float t = 0.0;
    t = iSphere( ro, rd, sphere[0] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    t = iSphere( ro, rd, sphere[1] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    t = iSphere( ro, rd, sphere[2] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    t = iSphere( ro, rd, sphere[3] ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );
    t = iPlane(  ro, rd, vec4(0.0,1.0,0.0,1.0)  ); if( t>eps && t<res.x ) res = vec2( t, 0.0 );

    return res.y;
#else

    float res = 1.0;

    float t = 0.0;
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
    return 24.0*pow( texture( iChannel1, rd ).xyz, vec3(2.2) );
}

vec3 doEnvironmentBlurred( in vec3 rd )
{
    return 9.0*pow( texture( iChannel2, rd ).xyz, vec3(2.2) );
}


vec4 doMaterial( in vec3 pos, in vec3 nor, float ma )
{
    vec4 mate = vec4(0.0);
    if( ma<4.5 ) { mate = vec4(0.30,0.25,0.20,1.0)*1.25; }
    if( ma<3.5 ) { mate = vec4(0.00,0.10,0.20,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.z) ) );
                   float d = distanceToSegment( vec2(0.22,0.0), vec2(-0.22,0.0), pos.yx );
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }
    if( ma<2.5 ) { mate = vec4(0.20,0.20,0.00,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.x) ) );
                   float d1 = distanceToSegment( vec2(0.22,0.12), vec2(-0.22,0.12), pos.yz );
                   float d2 = distanceToSegment( vec2(0.22,-0.12), vec2(-0.22,-0.12), pos.yz );
                   float d = min( d1, d2 );
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }
    if( ma<1.5 ) { mate = vec4(0.20,0.00,0.00,1.0)*1.25;
                   mate.xyz = mix( mate.xyz, vec3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.x) ) + smoothstep( 0.55, 0.56, abs(pos.y) ) );
                   float d1 = distanceToSegment( vec2(0.22,0.0), vec2(-0.22,0.0), pos.yz );
                   float d2 = distanceToSegment( vec2(0.22,0.0), vec2( -0.07,-0.2), pos.yz*vec2(1.0,-sign(pos.x)) );
                   float d3 = distanceToSegment( vec2(-0.07,-0.2), vec2(-0.07,0.04), pos.yz*vec2(1.0,-sign(pos.x)) );
                   float d = min(d1,min(d2,d3));
                   mate.xyz *= smoothstep( 0.04, 0.05, d );
                 }
    if( ma<0.5 ) { mate = vec4(0.01,0.20,0.03,0.0)*1.25;
                   mate.xyz *= 0.78 + 0.22*texture( iChannel3, 0.1*pos.xz ).x;
                 }
    return mate;
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
    sphere[0] = vec4( 1.0,1.0, 1.0,1.0);
    sphere[1] = vec4(-4.0,1.0, 0.0,1.0);
    sphere[2] = vec4( 0.0,1.0, 3.0,1.0);
    sphere[3] = vec4( 5.0,1.0, -2.0,1.0);

    // montecarlo
    vec3 tot = vec3(0.0);
    for( int a=0; a<40; a++ )
    {
        vec4 rrr = textureLod( iChannel0, (fragCoord.xy +0.5+3.3137*float(a))/iChannelResolution[0].xy, 0.0  ).xzyw;

        //-----------------------------------------------------
        // camera
        //-----------------------------------------------------

        vec2 p = -1.0 + 2.0 * (fragCoord.xy + rrr.xy) / iResolution.xy;
        p.x *= iResolution.x/iResolution.y;

        vec3 ro = vec3(8.0*sin(an),4.0,8.0*cos(an));
        vec3 ta = vec3(0.0,0.0,0.0);
        vec3 ww = normalize( ta - ro );
        vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0) ) );
        vec3 vv = normalize( cross(uu,ww));
        vec3 rd = normalize( p.x*uu + p.y*vv + 3.0*ww );

        // dof
        vec3 fp = ro + rd * 8.0;
        ro += (uu*rrr.z + vv*rrr.w)*0.5;
        rd = normalize( fp - ro );


        //-----------------------------------------------------
        // render
        //-----------------------------------------------------

        vec3 col = vec3(0.0);
        vec3 uvw = vec3(0.0);

        // raymarch
        vec2 tmat = intersect( ro, rd, uvw );
    //  if( tmat.y>-0.5 )
        {
            vec3 pos = ro + tmat.x*rd;
            vec3 nor = calcNormal( pos, tmat.y );
            vec3 ref = reflect( rd, nor );

            // material
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

            for( int j=0; j<3; j++ )
            {
                vec2  aa = hash2( rrr.x + float(j)*203.1 + float(a)*13.713 );
                float ra = sqrt(aa.y);
                float rx = ra*cos(6.2831*aa.x);
                float ry = ra*sin(6.2831*aa.x);
                float rz = sqrt( 1.0-aa.y );
                vec3  rr = vec3( rx*ru + ry*rv + rz*nor );
                lin += shadow( pos, rr ) * doEnvironmentBlurred( rr );
            }
            lin /= 3.0;
            // bounce
            lin += 1.5*clamp(0.3-0.7*nor.y,0.0,1.0)*vec3(0.0,0.2,00);
            // rim
            lin *= 1.0 + 6.0*mate.xyz*mate.w*pow( clamp( 1.0 + dot(nor,rd), 0.0, 1.0 ), 2.0 );;
            // specular
            float fre = 0.04 + 4.0*pow( clamp( 1.0 + dot(nor,rd), 0.0, 1.0 ), 5.0 );
            //lin += 1.0*doEnvironment(ref ).xyz * mate.w * fre * shadow( pos, ref );
            lin += 1.0*doEnvironment(ref ).xyz * mate.w * fre * step( 0.0, ref.y );

            // light-material interaction
            col = mate.xyz * lin;
        }

        tot += col;
    }
    tot /= 40.0;

    // gamma
    tot = pow( clamp(tot,0.0,1.0), vec3(0.45) );

    //tot = tot*0.5 + 0.5*tot*tot*(3.0-2.0*tot);
    tot *= 0.5 + 0.5*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.15 );

    fragColor = vec4( tot, 1.0 );
}
