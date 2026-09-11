(() => {
  'use strict';

  // Shared by the surface and persistent-foam pass so breakers, wet sand and
  // foam occupy the same shoreline. Depth is signed: dry land is negative.
  const shoreSource = `
    vec2 shoreDistances(vec2 p) {
      float headland = exp(-pow((p.y - 245.0) / 138.0,2.0));
      float left = -25.0 - p.y * .18 + headland * 79.0
        + sin(p.y * .025) * 4.5 + sin(p.y * .083) * 1.25;
      float right = 215.0 + p.y * .17 + sin(p.y * .012 + 1.8) * 28.0;
      return vec2(left - p.x,p.x - right);
    }

    vec3 shoreGeometry(vec2 p) {
      vec2 coast = shoreDistances(p);
      float headland = exp(-pow((p.y - 245.0) / 138.0,2.0));
      float leftSlope = -.18 - headland * 158.0 * (p.y - 245.0) / 19044.0
        + cos(p.y * .025) * .1125 + cos(p.y * .083) * .10375;
      float rightSlope = .17 + cos(p.y * .012 + 1.8) * .336;
      return coast.x > coast.y ? vec3(-coast.x * .085,.085,-leftSlope * .085)
        : vec3(-coast.y * .085,-.085,rightSlope * .085);
    }

    vec3 shoreWashGeometry(vec2 p, vec3 shore, float seconds, float height) {
      float depth = max(shore.x,-.35);
      // Integrating k proportional to 1/sqrt(depth) compresses incoming crests
      // toward the beach; phase moves landward, then the same front recedes.
      float root = sqrt(depth + .4);
      float phase = root * 2.8 + seconds * .85 + p.y * .021;
      float s = sin(phase), c = cos(phase);
      float profile = s + .22 * (s * s - .5);
      float profileSlope = c * (1.0 + .44 * s);
      float beach = smoothstep(-.35,-.08,shore.x);
      float beachT = clamp((shore.x + .35) / .27,0.0,1.0);
      float beachSlope = 6.0 * beachT * (1.0 - beachT) / .27;
      float decay = exp(-max(0.0,depth) * .26);
      float strength = (.015 + min(height,3.0) * .14) * smoothstep(0.0,.12,height);
      float amplitude = strength * decay * beach;
      float amplitudeSlope = strength * decay * (beachSlope - .26 * beach * step(0.0,depth));
      float slope = amplitudeSlope * profile + amplitude * profileSlope * 1.4 / root;
      return vec3(amplitude * profile,shore.yz * slope + vec2(0.0,amplitude * profileSlope * .021));
    }

    vec3 shoreWash(vec2 p, float seconds, float height) {
      return shoreWashGeometry(p,shoreGeometry(p),seconds,height);
    }

    vec3 nearshoreSurfaceGeometry(vec2 p, vec3 shore, vec3 wave, float seconds, float height) {
      float depth = max(0.0,shore.x);
      float t = clamp(depth / 3.0,0.0,1.0);
      float shelter = mix(.10,.72,t * t * (3.0 - 2.0 * t));
      float shelterSlope = .62 * 6.0 * t * (1.0 - t) / 3.0;
      float shoal = exp(-pow((depth - 1.4) / .85,2.0));
      float scale = shelter * (1.0 + .18 * shoal);
      float scaleSlope = shelterSlope * (1.0 + .18 * shoal)
        - shelter * .36 * shoal * (depth - 1.4) / (.85 * .85);
      wave.yz = wave.yz * scale + shore.yz * wave.x * scaleSlope * step(0.0,shore.x);
      wave.x *= scale;
      return wave + shoreWashGeometry(p,shore,seconds,height);
    }

    vec3 nearshoreSurface(vec2 p, vec3 wave, float seconds, float height) {
      return nearshoreSurfaceGeometry(p,shoreGeometry(p),wave,seconds,height);
    }
  `;

  // Linear radiance throughout; tone mapping happens once at the display edge.
  // Water IOR 1.333 gives F0 = ((1 - 1.333) / (1 + 1.333))^2 = 0.0204.
  const fragment = `
    precision highp float;
    uniform vec2 resolution;
    uniform float time;
    uniform float wind;
    uniform float waveHeight;
    uniform float swellStyle;
    uniform float elevation;
    uniform vec2 cameraAngle;
    uniform vec3 cameraPosition;
    uniform vec4 mood;
    uniform float brightness;
    uniform sampler2D swellField;
    uniform sampler2D rippleField;
    uniform float spectralReady;
    uniform float spectralMipmaps;
    uniform vec2 fieldLengths;
    uniform vec2 fieldSizes;
    uniform float spectralLinear;
    uniform sampler2D foamField;
    uniform float foamReady;
    uniform vec3 foamMapping;
    uniform float foamSize;
    uniform sampler2D environmentA;
    uniform sampler2D environmentB;
    uniform vec2 environmentScale;
    uniform vec2 environmentEncoding;
    uniform vec2 environmentRotation;
    uniform float environmentMix;
    uniform float environmentReady;
    uniform vec3 sunDirection;
    uniform float solarStrength;
    uniform vec3 solarRadiance;
    uniform float sceneKind;
    uniform float renderQuality;
    const float PI = 3.14159265359;
    const float TAU = 6.28318530718;

    vec2 rotate(vec2 p, float angle) {
      float c = cos(angle), s = sin(angle);
      return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
    }

    vec3 fallbackWave(vec2 p, vec2 direction, float k, float amp, float phase) {
      if (spectralReady < .5) k *= swellStyle < .5 ? 1.0 : swellStyle < 1.5 ? .65 : 1.55;
      float angle = dot(p, direction) * k - time * sqrt(9.81 * k) * 0.6 + phase;
      return vec3(sin(angle), cos(angle) * direction * k) * amp;
    }

    vec3 sampleField(sampler2D field, vec2 uv, float level, float size) {
      #ifndef OCEAN_FLOAT_LINEAR
      if (spectralLinear < .5) {
        // Float-linear filtering is optional on WebGL 1, including tablets.
        // Reconstruct the field explicitly instead of displaying square cells.
        vec2 pixel = uv * size - .5;
        vec2 base = (floor(pixel) + .5) / size;
        vec2 f = fract(pixel);
        vec3 a = texture2D(field,base).rgb;
        vec3 b = texture2D(field,base + vec2(1.0 / size,0.0)).rgb;
        vec3 c = texture2D(field,base + vec2(0.0,1.0 / size)).rgb;
        vec3 d = texture2D(field,base + vec2(1.0 / size)).rgb;
        return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
      }
      #endif
      #ifdef OCEAN_EXPLICIT_LOD
        return texture2DLodEXT(field, uv, level).rgb;
      #else
        return texture2D(field, uv, level).rgb;
      #endif
    }

    float sampleFoam(vec2 uv) {
      #ifdef OCEAN_FLOAT_LINEAR
      return texture2D(foamField,uv).r;
      #else
      if (spectralLinear > .5) return texture2D(foamField,uv).r;
      vec2 pixel = uv * foamSize - .5;
      vec2 base = (floor(pixel) + .5) / foamSize;
      vec2 f = fract(pixel);
      return mix(mix(texture2D(foamField,base).r,texture2D(foamField,base + vec2(1.0 / foamSize,0.0)).r,f.x),
        mix(texture2D(foamField,base + vec2(0.0,1.0 / foamSize)).r,texture2D(foamField,base + vec2(1.0 / foamSize)).r,f.x),f.y);
      #endif
    }

    float noise(vec2 p) {
      vec2 cell = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      vec3 q = fract(vec3(cell.xyx) * .1031);
      q += dot(q, q.yzx + 33.33);
      float a = fract((q.x + q.y) * q.z);
      q = fract(vec3((cell + vec2(1.0,0.0)).xyx) * .1031);
      q += dot(q, q.yzx + 33.33);
      float b = fract((q.x + q.y) * q.z);
      q = fract(vec3((cell + vec2(0.0,1.0)).xyx) * .1031);
      q += dot(q, q.yzx + 33.33);
      float c = fract((q.x + q.y) * q.z);
      q = fract(vec3((cell + vec2(1.0)).xyx) * .1031);
      q += dot(q, q.yzx + 33.33);
      float d = fract((q.x + q.y) * q.z);
      return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
    }

    ${shoreSource}

    float terrain(vec2 p) {
      vec2 coast = shoreDistances(p);
      float d = max(coast.x,coast.y);
      if (d < -8.0) return max(-24.0,d * .09);
      float promontory = exp(-pow((p.y - 245.0) / 170.0,2.0));
      float headlandMass = promontory;
      promontory = mix(promontory,.65,step(coast.x,coast.y));
      if (renderQuality > 2.5) {
        // Ultra resolves the cliff as geometry: broken faces, cut gullies
        // and separate talus rocks, rather than a smooth hill with noisy color.
        // The signed coast and intertidal slope remain shared with the water.
        if (d < 0.0) return d * .085;
        vec2 erosionPoint = p + vec2(noise(p * .028),noise(p * .031 + vec2(17.0,9.0))) * 11.0;
        float broadRidge = 1.0 - abs(noise(erosionPoint * .034) * 2.0 - 1.0);
        float brokenRidge = 1.0 - abs(noise(rotate(erosionPoint,.49) * .092) * 2.0 - 1.0);
        float ridge = broadRidge * broadRidge * .69 + brokenRidge * brokenRidge * .31;
        float edgeWarp = noise(vec2(p.y * .057,p.x * .018)) * 4.0
          + noise(erosionPoint * .19) * 1.7;
        // The opposite shore begins as a lower bluff and develops its own
        // changing relief, instead of mirroring an equally high cliff wall.
        if (coast.y > coast.x) promontory = .16 + headlandMass * .47;
        float cliffStart = 6.0 + edgeWarp + (1.0 - promontory) * 4.0;
        float cliffDepth = d - cliffStart;
        float faceWidth = 4.8 + brokenRidge * 6.0;
        float face = smoothstep(.0,faceWidth,cliffDepth);
        // Occasional broken ledges fade into the main face. There are no
        // continuous, equally spaced terraces wrapping the entire headland.
        float ledge = smoothstep(.0,1.2 + brokenRidge,cliffDepth) * .24
          + smoothstep(2.5 + broadRidge * 2.0,faceWidth + 1.0,cliffDepth) * .76;
        float ledgePresence = smoothstep(.68,.91,broadRidge) * (1.0 - smoothstep(.55,.9,brokenRidge));
        face = mix(face,ledge,ledgePresence * .48);
        float endpoint = .48 + .52 * smoothstep(.05,.72,headlandMass);
        float cliffHeight = (9.0 + promontory * 25.0) * (.60 + ridge * .60) * endpoint;
        float gully = pow(1.0 - abs(noise(vec2(p.y * .11,p.x * .032)) * 2.0 - 1.0),6.0);
        float erodedFace = face * cliffHeight * (1.0 - gully * .25);
        float upperLand = (1.0 - exp(-max(0.0,cliffDepth - faceWidth * .8) * .024))
          * (7.0 + promontory * 15.0) * (.3 + ridge * .7);
        float blocks = (noise(erosionPoint * .46) - .5) * .85
          + (noise(rotate(erosionPoint,-.36) * 1.13) - .5) * .24;
        erodedFace += blocks * smoothstep(.0,5.0,cliffDepth);
        // Jittered, sparsely occupied cells make individual boulder silhouettes
        // on the beach. Every cap stays inside its cell, so there are no seams.
        vec2 cell = floor(p / 3.4);
        vec2 local = fract(p / 3.4) - .5;
        float seed = fract(sin(dot(cell,vec2(127.1,311.7))) * 43758.5453);
        float seedB = fract(sin(dot(cell,vec2(269.5,183.3))) * 43758.5453);
        local -= (vec2(seed,seedB) - .5) * .16;
        local = rotate(local,seed * 6.28318530718);
        vec2 radius = vec2(.23 + seed * .13,.23 + seedB * .12);
        vec2 stonePoint = local / radius;
        float stoneShape = max(0.0,1.0 - dot(stonePoint,stonePoint));
        stoneShape = min(stoneShape,max(0.0,1.0 - abs(stonePoint.x + stonePoint.y * .36) * .84));
        float talus = pow(stoneShape,.55) * (.45 + seedB * 1.75)
          * smoothstep(.42,.65,seed) * smoothstep(1.0,3.0,d)
          * (1.0 - smoothstep(cliffStart - 1.0,cliffStart + 2.0,d));
        float beach = 1.6 * (1.0 - exp(-d * .055));
        return beach + erodedFace + upperLand + talus;
      }
      vec2 warped = p + vec2(noise(p * .017),noise(p * .019 + vec2(13.0,7.0))) * 15.0;
      float texture = noise(warped * .023) * .54 + noise(rotate(warped,.67) * .071) * .29
        + noise(rotate(warped,-.49) * .19) * .17;
      float beach = d < 0.0 ? d * .085 : 2.5 * (1.0 - exp(-d * .034));
      float rise = 1.0 - exp(-pow(max(0.0,d - 17.0) * .013,1.25));
      float hill = rise * (9.0 + 27.0 * promontory) * (.35 + texture * 1.12);
      float cliff = smoothstep(5.0,28.0,d) * promontory * (2.0 + texture * 9.0);
      return beach + hill + cliff + smoothstep(4.0,17.0,d) * (texture - .5) * 2.5;
    }

    vec3 surface(vec2 p, float travel, float detail) {
      if (spectralReady > 0.5) {
        float footprint = max(0.01, travel * travel / max(0.5, cameraPosition.y) / resolution.y);
        float shading = step(1.5,detail) * step(2.5,renderQuality);
        detail = min(1.0,detail);
        // A grazing pixel is long along the viewing direction but narrow
        // across it. Ultra retains those cross-wave slopes with three taps.
        float filteredFootprint = mix(footprint,max(travel / resolution.y,footprint / 3.0),shading);
        float filter = detail * clamp(log2(max(1.0, filteredFootprint * fieldSizes.y / fieldLengths.y)) - 0.7, 0.0, 9.0);
        float largeFilter = detail * clamp(log2(max(1.0, footprint * fieldSizes.x / fieldLengths.x)) - 0.7, 0.0, 9.0);
        vec3 large = sampleField(swellField, p / fieldLengths.x, largeFilter * spectralMipmaps,fieldSizes.x);
        vec2 smallUv = rotate(p, 0.42) / fieldLengths.y + vec2(0.173, 0.387);
        vec3 small = sampleField(rippleField, smallUv, filter * spectralMipmaps,fieldSizes.y);
        if (shading > .5) {
          vec2 alongView = normalize(p - cameraPosition.xz + vec2(.0001));
          vec2 tap = rotate(alongView,.42) * footprint * .33 / fieldLengths.y;
          small = (small + sampleField(rippleField,smallUv + tap,filter * spectralMipmaps,fieldSizes.y)
            + sampleField(rippleField,smallUv - tap,filter * spectralMipmaps,fieldSizes.y)) / 3.0;
        }
        small.yz = rotate(small.yz, -0.42);
        // The unresolved slopes become roughness in the reflection model below.
        small.yz *= mix(1.0, exp(-footprint * 0.65), detail * (1.0 - spectralMipmaps));
        vec3 wave = large + small;
        // A second-order Stokes term narrows crests and opens the troughs.
        // Its derivative is applied to the normals as well as the geometry.
        float sigma = max(.08,waveHeight * .25);
        float shape = clamp(wave.x / sigma,-2.5,2.5);
        wave.x += sigma * .12 * (shape * shape - 1.0) * smoothstep(0.0,.15,waveHeight);
        wave.yz *= max(.4,1.0 + .24 * shape);
        float capillary = detail * exp(-filteredFootprint * 38.0) * smoothstep(.2,2.5,wind);
        wave += (fallbackWave(p,vec2(.83,.558),22.0,.0011,2.3)
          + fallbackWave(p,vec2(-.38,.925),31.0,.00065,.7)
          + fallbackWave(p,vec2(.15,.989),47.0,.00032,4.1)) * capillary;
        if (sceneKind > .5) {
          wave = nearshoreSurface(p,wave,time,waveHeight);
        }
        return wave;
      }
      vec3 wave = fallbackWave(p, vec2(.24,.971), .3,.24,.4);
      wave += fallbackWave(p, vec2(-.32,.947), .53,.15,2.0);
      wave += fallbackWave(p, vec2(.77,.638), .90,.08,4.2);
      wave += fallbackWave(p, vec2(-.91,.415), 1.47,.047,1.2);
      wave += fallbackWave(p, vec2(.13,.991), 2.41,.027,3.6);
      wave += fallbackWave(p, vec2(.94,.341), 3.86,.018,5.2);
      wave += fallbackWave(p, vec2(-.48,.877), 5.96,.008,.8);
      wave *= waveHeight * mix(.7,1.2,wind / 20.0);
      return sceneKind > .5 ? nearshoreSurface(p,wave,time,waveHeight) : wave;
    }

    vec3 analyticSky(vec3 ray) {
      float altitude = clamp(ray.y,0.0,1.0);
      float sunward = pow(max(0.0,dot(normalize(vec3(ray.x,.04,ray.z)),normalize(vec3(sunDirection.x,.04,sunDirection.z)))),6.0);
      vec3 zenith = vec3(.16,.34,.64) * mood.x + vec3(.12,.34,.75) * mood.y
        + vec3(.15,.28,.49) * mood.z + vec3(.055,.085,.18) * mood.w;
      vec3 horizon = vec3(.58,.48,.44) * mood.x + vec3(.62,.77,.93) * mood.y
        + vec3(.61,.39,.26) * mood.z + vec3(.27,.21,.33) * mood.w;
      float haze = exp(-altitude * 5.5);
      vec3 color = mix(zenith,horizon,haze);
      vec3 warm = vec3(1.0,.48,.19) * (mood.x * .33 + mood.z * .43)
        + vec3(.5,.48,.4) * mood.y * .17;
      color += warm * sunward * exp(-altitude * 4.0);
      return color;
    }

    vec2 environmentUv(vec3 ray, float rotation) {
      ray.xz = rotate(ray.xz, rotation);
      // Avoid sampling the pole row beyond the panorama's vertical extent.
      return vec2(atan(ray.x,ray.z) / TAU + .5, clamp(acos(clamp(ray.y,-1.0,1.0)) / PI,.00025,.99975));
    }

    vec3 sky(vec3 ray, float blur) {
      ray = normalize(ray);
      vec3 photographedA = texture2D(environmentA, environmentUv(ray, environmentRotation.x), blur).rgb * environmentScale.x;
      vec3 photographedB = texture2D(environmentB, environmentUv(ray, environmentRotation.y), blur).rgb * environmentScale.y;
      photographedA = mix(photographedA,pow(max(photographedA / environmentScale.x,vec3(0.0)),vec3(2.2)) * environmentScale.x,environmentEncoding.x);
      photographedB = mix(photographedB,pow(max(photographedB / environmentScale.y,vec3(0.0)),vec3(2.2)) * environmentScale.y,environmentEncoding.y);
      vec3 photograph = mix(photographedA, photographedB, environmentMix);
      photograph *= mix(vec3(1.0),vec3(1.025,1.0,.95),mood.z);
      vec3 color = mix(analyticSky(ray), photograph, environmentReady);
      if (blur < .01) {
        // The photographic core was extracted into solarRadiance. Draw one
        // physical half-degree disk here; reflected light uses the water BRDF.
        float radius = .00465;
        float angularPixel = 1.2 / max(1.0,resolution.y);
        float separation = 1.0 - clamp(dot(ray,sunDirection),-1.0,1.0);
        float edge = max(.000001,radius * angularPixel);
        float disk = 1.0 - smoothstep(radius * radius * .5 - edge,radius * radius * .5 + edge,separation);
        float horizon = smoothstep(-angularPixel,angularPixel,ray.y);
        color += solarRadiance * disk * horizon;
      }
      return color;
    }

    vec3 reflectedSky(vec3 direction, float roughness) {
      // Sample filtered photographic radiance; the solar core is accounted
      // for once by direct GGX lighting, rather than four shifted sun disks.
      vec3 axis = abs(direction.y) < .98 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0);
      vec3 tangent = normalize(cross(direction,axis));
      vec3 bitangent = cross(direction,tangent);
      float spread = roughness * roughness * .9;
      float blur = max(.25,log2(max(1.0,spread * 320.0)));
      return (sky(direction + tangent * spread, blur) + sky(direction - tangent * spread, blur)
        + sky(direction + bitangent * spread, blur) + sky(direction - bitangent * spread, blur)) * .25;
    }

    float traceTerrain(vec3 origin, vec3 direction, float limit) {
      float travel = .3;
      float previous = 0.0;
      bool ultraCoast = renderQuality > 2.5;
      for (int i=0; i<384; i++) {
        if (float(i) >= (ultraCoast ? 384.0 : 80.0 + renderQuality * 36.0)) break;
        vec3 p = origin + direction * travel;
        if (travel >= limit || (p.y > (ultraCoast ? 96.0 : 58.0) && direction.y > 0.0)) break;
        vec2 coasts = shoreDistances(p.xz);
        float waterClearance = -max(coasts.x,coasts.y);
        if (p.y > 0.0 && waterClearance > 1.0) {
          // The seabed is below zero here. Bound the shoreline's horizontal
          // derivative to skip clear water without stepping across a headland.
          previous = travel;
          travel += max(.35,waterClearance / (abs(direction.x) + 1.1 * abs(direction.z) + .001));
          continue;
        }
        float clearance = p.y - terrain(p.xz);
        if (clearance < 0.0) {
          float lo = previous, hi = travel;
          for (int j=0; j<16; j++) {
            if (float(j) >= (ultraCoast ? 16.0 : min(12.0,7.0 + renderQuality * 2.0))) break;
            float mid = (lo + hi) * .5;
            vec3 q = origin + direction * mid;
            if (q.y > terrain(q.xz)) lo = mid; else hi = mid;
          }
          return (lo + hi) * .5;
        }
        previous = travel;
        float minimumStep = ultraCoast ? .035 : mix(.48,.10,clamp(renderQuality / 3.0,0.0,1.0));
        float angularStep = ultraCoast ? .00012 : mix(.004,.0008,clamp(renderQuality / 3.0,0.0,1.0));
        // A steep shelf cannot use the gentle-hill clearance bound: shorter
        // advances prevent rays stepping over narrow rocks and cliff faces.
        float clearanceScale = ultraCoast ? max(1.5,4.0 - direction.y) : max(.35,1.2 - direction.y);
        travel += clamp(max(clearance / clearanceScale,minimumStep + travel * angularStep),minimumStep,50.0);
      }
      return limit;
    }

    float stoneTexture(vec3 p, vec3 weights) {
      float base = noise(p.yz) * weights.x + noise(p.zx) * weights.y + noise(p.xy) * weights.z;
      if (renderQuality > 2.5) {
        vec3 q = p * vec3(1.73,.83,1.31) + vec3(7.3,13.1,3.8);
        float fracture = noise(q.yz) * weights.x + noise(q.zx) * weights.y + noise(q.xy) * weights.z;
        float ridge = 1.0 - abs(fracture * 2.0 - 1.0);
        return clamp(base * .7 + ridge * ridge * .3,0.0,1.0);
      }
      return base;
    }

    vec3 landRadiance(vec3 p, vec3 direction, float travel, float reflectionBlur) {
      bool ultraCoast = renderQuality > 2.5;
      float epsilon = max(max(ultraCoast ? .012 : mix(.12,.025,clamp(renderQuality / 3.0,0.0,1.0)),
        travel / resolution.y * (ultraCoast ? .16 : .3)),reflectionBlur * .18);
      float h = terrain(p.xz);
      vec3 n = normalize(vec3(terrain(p.xz - vec2(epsilon,0.0)) - terrain(p.xz + vec2(epsilon,0.0)),
        2.0 * epsilon,terrain(p.xz - vec2(0.0,epsilon)) - terrain(p.xz + vec2(0.0,epsilon))));
      vec3 geometricNormal = n;
      vec3 weights = pow(abs(n),vec3(4.0));
      weights /= max(.001,weights.x + weights.y + weights.z);
      // A beach seen at a grazing angle covers far more world-space ground
      // per pixel than distance alone suggests. Filter that actual footprint.
      float geometricFootprint = max(.008,travel / resolution.y / max(.08,abs(dot(n,-direction))));
      float footprint = geometricFootprint;
      #ifdef OCEAN_DERIVATIVES
        footprint = clamp(max(length(dFdx(p)),length(dFdy(p))),geometricFootprint * .35,geometricFootprint * 2.0);
      #endif
      // Reflected terrain covers the water's roughness lobe, not a pin-sharp
      // mirror texel. Filter soil and bump detail over that wider footprint.
      footprint = max(footprint,reflectionBlur);
      float patch = noise(rotate(p.xz,.67) * .058 + vec2(7.1,3.4)) * .63 + noise(p.xz * .17) * .37;
      float stone = stoneTexture(p * .72,weights);
      float baseStone = stone;
      float fine = stoneTexture(p * 3.1,weights);
      float filteredFine = mix(.5,fine,1.0 - smoothstep(.25,1.0,footprint * 3.1));
      stone = mix(.5,stone,1.0 - smoothstep(.35,1.3,footprint * .72));
      float grain = stone * .56 + filteredFine * .44;
      if (renderQuality > 1.5) {
        float mineral = stoneTexture(p * 11.3,weights);
        mineral = mix(.5,mineral,1.0 - smoothstep(.2,.9,footprint * 11.3));
        grain = grain * .88 + mineral * .12;
      }
      float sandGrain = mix(.5,noise(p.xz * 8.0),exp(-footprint * 9.0));
      vec3 sand = mix(vec3(.27,.219,.147),vec3(.48,.411,.30),patch * .7 + sandGrain * .3);
      float wash = shoreWash(p.xz,time,waveHeight).x;
      float wet = 1.0 - smoothstep(.03 + wash,.42 + min(waveHeight,3.0) * .10,h);
      sand *= mix(1.0,.52,wet);
      vec3 rock = mix(vec3(.072,.079,.065),vec3(.34,.298,.226),smoothstep(.12,.86,grain));
      rock = mix(rock,vec3(.39,.371,.309),smoothstep(.57,.79,stone) * (.22 + patch * .38));
      float steep = 1.0 - smoothstep(.55,.95,n.y);
      float rocky = smoothstep(.75,3.9,h + (patch - .5) * 4.0) * (.55 + steep * .45);
      vec3 vegetation = mix(vec3(.026,.059,.019),vec3(.146,.184,.063),smoothstep(.14,.85,patch * .6 + grain * .4));
      vegetation = mix(vegetation,vec3(.16,.132,.066),smoothstep(.63,.85,patch) * .6);
      float grass = smoothstep(1.8,5.5,h + (patch - .5) * 8.0) * smoothstep(.57,.89,n.y);
      grass *= smoothstep(.15,.49,patch + grain * .24);
      float joint = 0.0;
      float mineralSeam = 0.0;
      if (ultraCoast) {
        // Geological bedding lives in world space, crossing differently
        // oriented cliff faces consistently instead of repeating a flat decal.
        float tiltedHeight = p.y + p.x * .09 + sin(p.z * .047) * .72;
        float bedding = sin(tiltedHeight * 3.7 + stoneTexture(p * .11,weights) * 2.4);
        float beddingDetail = exp(-footprint * 3.0);
        mineralSeam = (1.0 - smoothstep(.035,.15,abs(bedding))) * beddingDetail;
        float layer = (.5 + .5 * bedding) * beddingDetail;
        float fractureField = stoneTexture(p * .43 + vec3(11.2,0.0,7.4),weights);
        joint = (1.0 - smoothstep(.018,.065,abs(fractureField - .51))) * exp(-footprint * 1.8);
        vec3 sandstone = mix(vec3(.145,.133,.108),vec3(.42,.377,.298),grain * .68 + layer * .32);
        vec3 slate = mix(vec3(.067,.077,.075),vec3(.245,.252,.224),grain);
        rock = mix(sandstone,slate,smoothstep(.42,.72,patch) * .66);
        rock = mix(rock,vec3(.47,.444,.355),mineralSeam * .38);
        rock *= 1.0 - joint * .38;
        float lichen = smoothstep(.55,.72,stoneTexture(p * .16,weights))
          * smoothstep(.22,.84,geometricNormal.y) * (1.0 - wet) * .36;
        rock = mix(rock,vec3(.207,.231,.133),lichen);
        // Steep talus and ledges remain rock; vegetation occupies flatter,
        // sheltered patches and retains exposed stony gaps between clumps.
        rocky = max(rocky,smoothstep(.12,.62,steep) * smoothstep(.3,1.15,h));
        grass *= .82 + .18 * smoothstep(.35,.68,stoneTexture(p * .52,weights));
        grass *= 1.0 - joint * .35;
        vegetation = mix(vec3(.035,.064,.024),vec3(.139,.172,.062),patch * .55 + grain * .45);
        float ripple = sin(dot(p.xz,vec2(2.8,.72)) + noise(p.xz * .65) * 1.8);
        sand *= 1.0 + ripple * .055 * exp(-footprint * 3.0) * (1.0 - wet);
        rock *= mix(1.0,.57,wet);
      }
      vec3 albedo = mix(mix(sand,rock,rocky),vegetation,grass * .96);
      // The geometry supplies the silhouette; centimetre-scale stone and soil
      // detail affects only the normal and fades before becoming subpixel.
      float bumpEpsilon = max(mix(.12,.035,clamp(renderQuality / 3.0,0.0,1.0)),footprint * .6);
      vec3 tangent = normalize(cross(n,vec3(.001,1.0,0.0)));
      vec3 bitangent = cross(n,tangent);
      vec3 materialPoint = p * .72;
      float bumpT = stoneTexture(materialPoint + tangent * bumpEpsilon,weights) - baseStone;
      float bumpB = stoneTexture(materialPoint + bitangent * bumpEpsilon,weights) - baseStone;
      float bump = mix(.22,.48,rocky) * mix(1.0,.65,grass) * exp(-footprint * .85);
      n = normalize(n - (tangent * bumpT + bitangent * bumpB) * bump / bumpEpsilon);
      vec3 ambient = clamp(sky(vec3(n.x * .45,.55 + n.y * .45,n.z * .45),3.0),vec3(.12),vec3(1.3));
      vec3 sunColor = mix(vec3(1.0,.73,.49),vec3(1.0,.97,.9),mood.y);
      float diffuse = max(0.0,dot(n,sunDirection));
      float terrainLight = 1.0;
      if (ultraCoast && diffuse > .0) {
        // Trace terrain toward the sun at growing distances. This gives
        // shelves and boulders real cast shadows, including low sunset light.
        float shadowDistance = .45;
        for (int i=0; i<9; i++) {
          if (float(i) >= renderQuality * 3.0) break;
          float blocker = terrain(p.xz + sunDirection.xz * shadowDistance) - h
            - sunDirection.y * shadowDistance;
          float penumbra = .06 + shadowDistance * .035;
          terrainLight = min(terrainLight,1.0 - smoothstep(.025,penumbra,blocker) * .9);
          shadowDistance = shadowDistance * 1.68 + .23;
          if (terrainLight < .12) break;
        }
      } else if (renderQuality > .5 && diffuse > .0) {
        float obstruction = terrain(p.xz + sunDirection.xz * 6.0) - h - sunDirection.y * 6.0;
        if (renderQuality > 1.5) obstruction = max(obstruction,
          (terrain(p.xz + sunDirection.xz * 18.0) - h - sunDirection.y * 18.0) * .5);
        terrainLight = 1.0 - smoothstep(.1,2.4,obstruction) * .7;
      }
      float occlusion = mix(.78,1.0,smoothstep(.15,.67,grain));
      if (ultraCoast) {
        float relief = 0.0;
        for (int i=0; i<4; i++) {
          if (float(i) >= renderQuality + 1.0) break;
          float angle = float(i) * 1.57079632679 + .39;
          vec2 offset = vec2(cos(angle),sin(angle)) * 2.3;
          float plane = -dot(geometricNormal.xz,offset) / max(.18,geometricNormal.y);
          relief += max(0.0,terrain(p.xz + offset) - h - plane - .12);
        }
        occlusion *= max(.4,exp(-relief * .17)) * (1.0 - joint * rocky * .2);
      }
      vec3 color = albedo * (ambient * .68 * occlusion + sunColor * diffuse * terrainLight * (.32 + solarStrength * .18));
      if (ultraCoast) {
        vec3 halfVector = normalize(sunDirection - direction);
        float wetGlint = pow(max(0.0,dot(n,halfVector)),72.0) * wet * (1.0 - grass);
        color += sunColor * wetGlint * solarStrength * terrainLight * .055;
      }
      float foamEdge = (1.0 - smoothstep(.015,.095,abs(h - wash)))
        * smoothstep(.28,.68,noise(p.xz * 2.8 + vec2(time * .027,0.0))) * .5;
      color = mix(color,ambient * .66,foamEdge);
      // Aerial perspective integrates a broad patch of sky radiance. Sampling
      // one horizon texel would project tiny HDR stitching marks onto the land.
      vec3 haze = (clamp(sky(vec3(.82,.15,.55),3.0),vec3(.03),vec3(2.4))
        + clamp(sky(vec3(-.73,.15,.66),3.0),vec3(.03),vec3(2.4))
        + clamp(sky(vec3(0.0,.15,-1.0),3.0),vec3(.03),vec3(2.4))) / 3.0;
      float sunward = pow(max(0.0,dot(normalize(direction.xz),normalize(sunDirection.xz))),8.0);
      haze = mix(haze,sunColor * .95,sunward * .12 * (1.0 - mood.w));
      color = mix(color,haze,1.0-exp(-travel * .0007));
      return mix(color,sky(direction,0.0),smoothstep(850.0,1500.0,travel));
    }

    vec3 filmic(vec3 x) {
      return clamp((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0.0, 1.0);
    }

    vec3 cameraRay(vec2 pixel) {
      vec2 uv = pixel / resolution;
      vec2 point = (uv * 2.0 - 1.0) * vec2(resolution.x / resolution.y, 1.0);
      vec3 ray = normalize(vec3(point * .531709, 1.0));
      ray.yz = mat2(cos(cameraAngle.y),-sin(cameraAngle.y),sin(cameraAngle.y),cos(cameraAngle.y)) * ray.yz;
      ray.xz = mat2(cos(cameraAngle.x),-sin(cameraAngle.x),sin(cameraAngle.x),cos(cameraAngle.x)) * ray.xz;
      return ray;
    }

    void main() {
      vec3 ray = cameraRay(gl_FragCoord.xy);
      vec3 color = sky(ray, 0.0);
      vec3 landPoint = vec3(0.0);
      vec3 landDirection = ray;
      float landDistance = 0.0;
      float landBlur = 0.0;
      float landWeight = 0.0;
      float terrainTravel = 1600.0;
      if (sceneKind > .5 && ray.y < .75) {
        float landLimit = ray.y < -.001 ? min(1600.0,(cameraPosition.y + max(.4,waveHeight)) / -ray.y) : 1600.0;
        float hit = traceTerrain(cameraPosition,ray,landLimit);
        if (hit < landLimit) terrainTravel = hit;
      }
      float waterTravel = 12000.0;
      float waterNearBound = max(0.0,(cameraPosition.y - max(.4,waveHeight * 1.6)) / max(.0002,-ray.y));
      if (ray.y < -.0002 && terrainTravel > max(1.0,waterNearBound)) {
        float travel = min(12000.0, cameraPosition.y / -ray.y);
        // Bracket the first surface crossing, then refine it. Translation moves
        // through a fixed world-space field rather than sliding the image.
        float extent = max(.4, waveHeight * 1.6);
        float nearT = max(.0, (cameraPosition.y - extent) / -ray.y);
        float farT = min(12000.0, (cameraPosition.y + extent) / -ray.y);
        if (renderQuality > 2.5) {
          // Find the first crest crossing before refining. A single bisection
          // across several waves can select a hidden trough and draw seams.
          float start = nearT;
          float span = farT - start;
          float previous = start;
          for (int i=1; i<=24; i++) {
            if (float(i) > renderQuality * 8.0) break;
            float probe = start + span * float(i) / 24.0;
            float waterY = surface(cameraPosition.xz + ray.xz * probe,probe,1.0).x;
            if (cameraPosition.y + ray.y * probe <= waterY) {
              nearT = previous;
              farT = probe;
              break;
            }
            previous = probe;
          }
        }
        for (int i=0; i<13; i++) {
          if (float(i) >= 8.0 + max(0.0,renderQuality - 2.0) * 5.0) break;
          travel = (nearT + farT) * .5;
          float waterY = surface(cameraPosition.xz + ray.xz * travel,travel,1.0).x;
          float distanceY = cameraPosition.y + ray.y * travel - waterY;
          if (distanceY > 0.0) nearT = travel; else farT = travel;
        }
        travel = (nearT + farT) * .5;
        vec3 intersection = surface(cameraPosition.xz + ray.xz * travel,travel,1.0);
        float derivative = ray.y - dot(intersection.yz,ray.xz);
        if (derivative < -.0001) {
          travel = clamp(travel - (cameraPosition.y + ray.y * travel - intersection.x) / derivative,nearT,farT);
        }
        vec2 position = cameraPosition.xz + ray.xz * travel;
        waterTravel = travel;
        vec3 wave = surface(position,travel,renderQuality > 2.5 ? 2.0 : 1.0);
        // Anisotropic shading filters the slopes, while every depth/reflection
        // calculation stays attached to the surface intersection we solved.
        wave.x = cameraPosition.y + ray.y * travel;
        vec3 normal = normalize(vec3(-wave.y,1.0,-wave.z));
        vec3 view = -ray;
        normal = normalize(normal + view * max(0.0,.025 - dot(normal,view)));
        float ndotv = max(.001,dot(normal,view));
        vec3 reflected = reflect(ray,normal);
        reflected.y = max(.003,reflected.y);
        float fresnel = .0204 + .9796 * pow(1.0-clamp(ndotv,0.0,1.0),5.0);
        float distanceRoughness = smoothstep(30.0,500.0,travel) * .13;
        float roughness = .09 + wind / 20.0 * .10 + distanceRoughness;
        vec3 reflectedLight = reflectedSky(reflected,roughness);
        if (sceneKind > .5 && renderQuality > .5 && reflected.y < .42) {
          vec3 reflectionOrigin = vec3(position.x,wave.x + .045,position.y);
          float landReflection = traceTerrain(reflectionOrigin,reflected,1200.0);
          if (landReflection < 1200.0) {
            landPoint = reflectionOrigin + reflected * landReflection;
            landDirection = reflected;
            landDistance = landReflection;
            landBlur = max(.03,landReflection * roughness * roughness * .55);
            float scatter = clamp(roughness * 2.4 + smoothstep(80.0,600.0,landReflection) * .12,.3,.75);
            reflectedLight *= scatter;
            landWeight = 1.0 - scatter;
          }
        }
        // Diffuse sky irradiance must not accidentally sample the solar disk.
        vec3 ambient = clamp(sky(vec3(0.0,1.0,0.0),2.0),vec3(.05),vec3(1.5));
        vec3 deep = mix(vec3(.012,.060,.073), vec3(.008,.024,.044), mood.w);
        vec3 scatter = mix(vec3(.035,.16,.155), vec3(.021,.055,.094), mood.w);
        float crest = clamp(.45 + wave.x * 1.1,0.0,1.0);
        float backLight = pow(max(0.0,dot(view,-sunDirection)),3.0) * crest;
        vec3 transmitted = (deep + scatter * (.12 + backLight * .22)) * (ambient * .6 + .3);
        if (sceneKind > .5) {
          float depth = max(.025,wave.x - terrain(position));
          vec3 sand = vec3(.28,.246,.17) * (ambient * .6 + .3);
          if (renderQuality > 2.5 && depth < 12.0) {
            vec3 refracted = refract(ray,normal,.7502);
            vec2 bed = position + refracted.xz * depth / max(.25,-refracted.y);
            float grain = noise(bed * 3.8) * .55 + noise(bed * 15.0) * .45;
            float ridges = sin(bed.x * 5.6 + noise(bed * .28) * 4.0);
            sand *= .88 + grain * .20 + ridges * .075 * exp(-travel / resolution.y * 20.0);
            // Refracted shallow light picks up soft, moving ripple caustics.
            float caustic = pow(max(0.0,sin(bed.x * 2.7 + time * .31)
              * sin(bed.y * 3.1 - time * .24 + wave.x * 2.0)),8.0);
            sand *= 1.0 + caustic * .18 * exp(-depth * .32) * solarStrength;
          }
          vec3 attenuation = exp(-depth * vec3(.36,.14,.075) * (1.0 + .65 / max(.3,ndotv)));
          vec3 shallowScatter = vec3(.018,.105,.095) * (ambient * .55 + .4);
          transmitted = mix(transmitted,sand * attenuation + shallowScatter * (1.0 - attenuation),exp(-depth * .045));
          float rippleShade = .92 + .08 * sin(position.x * 6.0 + sin(position.y * 3.0 + time * .25));
          transmitted *= mix(1.0,rippleShade,exp(-depth * .6));
        }
        color = mix(transmitted,reflectedLight,fresnel);
        landWeight *= fresnel;
        // The sky's solar core is integrated separately, so highlights retain
        // energy as ripples narrow them instead of clipping into flat patches.
        vec3 halfVector = normalize(sunDirection + view);
        float ndotl = max(.0,dot(normal,sunDirection));
        float ndoth = max(.0,dot(normal,halfVector));
        float vdoth = max(.0,dot(view,halfVector));
        float a = max(.018,roughness * roughness);
        float a2 = a * a;
        float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
        float distribution = a2 / max(.0000001,PI * denom * denom);
        float k = roughness * roughness * .5;
        float masking = ndotv / (ndotv * (1.0-k)+k) * ndotl / max(.001,ndotl*(1.0-k)+k);
        float solarFresnel = .0204 + .9796 * pow(1.0-vdoth,5.0);
        float specular = distribution * masking * solarFresnel / max(.01,4.0*ndotv);
        color += solarRadiance * .000068 * specular;
        // Calm seas have little foam; only steep, energetic crests catch it.
        float slope = length(wave.yz);
        float foam = smoothstep(.55,.95,slope) * smoothstep(5.0,15.0,wind) * .16;
        if (foamReady > .5) {
          vec2 foamUv = (position - foamMapping.xy) / max(1.0,foamMapping.z);
          vec2 edge = smoothstep(vec2(0.0),vec2(.06),foamUv)
            * (1.0 - smoothstep(vec2(.94),vec2(1.0),foamUv));
          float history = sampleFoam(clamp(foamUv,vec2(0.0),vec2(1.0)));
          vec2 drift = vec2(.436,.900) * time * (.035 + wind * .013);
          float lace = smoothstep(.18,.78,noise((position - drift) * 2.4));
          if (renderQuality > 2.5) {
            float bubbles = noise((position - drift) * 13.0);
            float resolved = exp(-travel * travel / max(.5,cameraPosition.y) / resolution.y * 13.0);
            lace *= mix(1.0,smoothstep(.19,.69,bubbles) * .7 + .3,resolved);
          }
          foam = max(foam,history * edge.x * edge.y * (.4 + lace * .6) * .72);
        }
        if (sceneKind > .5) {
          float depth = wave.x - terrain(position);
          float front = exp(-pow((depth - .035) / .14,2.0));
          float lace = smoothstep(.22,.67,noise(position * 2.8 + vec2(time * .027,0.0)));
          float wash = shoreWash(position,time,waveHeight).x;
          float advancing = smoothstep(-.025,.07,wash);
          foam = max(foam,front * (.18 + advancing * .4) * lace * smoothstep(-.08,.0,depth));
        }
        color = mix(color,ambient * .75,foam);
        landWeight *= 1.0 - foam;
        vec3 haze = sky(normalize(vec3(ray.x,.007,ray.z)),1.5);
        color = mix(color,haze,1.0-exp(-travel*.000075));
        color = mix(color,haze,smoothstep(-.0008,-.0002,ray.y));
        landWeight *= exp(-travel*.000075) * (1.0 - smoothstep(-.0008,-.0002,ray.y));
      }
      if (sceneKind > .5 && terrainTravel < min(waterTravel,1600.0)) {
        landPoint = cameraPosition + ray * terrainTravel;
        landDirection = ray;
        landDistance = terrainTravel;
        landBlur = 0.0;
        landWeight = 1.0;
        color = vec3(0.0);
      }
      // Both visible and reflected terrain share one shader call. Carry its
      // linear contribution through the water mix to avoid duplicating the
      // large material/shadow program in driver-generated shader code.
      if (landWeight > .0) color += landRadiance(landPoint,landDirection,landDistance,landBlur) * landWeight;
      float sceneExposure = dot(mood,vec4(.95,1.0,.85,.38));
      color = pow(filmic(max(color,vec3(0.0)) * brightness * sceneExposure * .85),vec3(1.0/2.2));
      float dither = fract(dot(gl_FragCoord.xy,vec2(.75487766,.56984029))) - .5;
      gl_FragColor = vec4(clamp(color + dither/255.0,0.0,1.0),1.0);
    }
  `;
  const constrainCoveCamera = (camera) => {
    camera.z = Math.max(-180, Math.min(1250, camera.z));
    const z = camera.z;
    const headland = Math.exp(-(((z - 245) / 138) ** 2));
    const left = -25 - z * 0.18 + headland * 79
      + Math.sin(z * 0.025) * 4.5 + Math.sin(z * 0.083) * 1.25;
    const right = 215 + z * 0.17 + Math.sin(z * 0.012 + 1.8) * 28;
    camera.x = Math.max(left + 2.5, Math.min(right - 2.5, camera.x));
  };
  const getWaterDepth = (x, z) => {
    const headland = Math.exp(-(((z - 245) / 138) ** 2));
    const left = -25 - z * .18 + headland * 79 + Math.sin(z * .025) * 4.5 + Math.sin(z * .083) * 1.25;
    const right = 215 + z * .17 + Math.sin(z * .012 + 1.8) * 28;
    return Math.min(x - left,right - x) * .085;
  };
  const getShoreProximity = (x, z) => Math.exp(-Math.max(0,getWaterDepth(x,z)) / 2.5);
  window.OceanWaveShaders = Object.freeze({ fragment, shoreSource, constrainCoveCamera, getWaterDepth, getShoreProximity });
})();
