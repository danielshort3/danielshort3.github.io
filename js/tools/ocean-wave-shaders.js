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
    uniform sampler2D foamField;
    uniform float foamReady;
    uniform vec3 foamMapping;
    uniform sampler2D environmentA;
    uniform sampler2D environmentB;
    uniform vec2 environmentScale;
    uniform vec2 environmentRotation;
    uniform float environmentMix;
    uniform float environmentReady;
    uniform vec3 sunDirection;
    uniform float solarStrength;
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

    vec3 sampleField(sampler2D field, vec2 uv, float level) {
      #ifdef OCEAN_EXPLICIT_LOD
        return texture2DLodEXT(field, uv, level).rgb;
      #else
        return texture2D(field, uv, level).rgb;
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
      promontory = mix(promontory,.65,step(coast.x,coast.y));
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
        float filter = detail * clamp(log2(max(1.0, footprint * 128.0 / fieldLengths.y)) - 0.7, 0.0, 6.0);
        float largeFilter = detail * clamp(log2(max(1.0, footprint * 128.0 / fieldLengths.x)) - 0.7, 0.0, 6.0);
        vec3 large = sampleField(swellField, p / fieldLengths.x, largeFilter * spectralMipmaps);
        vec2 smallUv = rotate(p, 0.42) / fieldLengths.y + vec2(0.173, 0.387);
        vec3 small = sampleField(rippleField, smallUv, filter * spectralMipmaps);
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
        float capillary = detail * exp(-footprint * 38.0) * smoothstep(.2,2.5,wind);
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
      vec3 blue = vec3(.26,.48,.72) * mood.x + vec3(.18,.43,.76) * mood.y
        + vec3(.32,.40,.58) * mood.z + vec3(.12,.16,.28) * mood.w;
      vec3 horizon = vec3(.86,.69,.52) * mood.x + vec3(.67,.78,.89) * mood.y
        + vec3(.95,.52,.27) * mood.z + vec3(.43,.38,.51) * mood.w;
      return mix(horizon, blue, pow(clamp(ray.y,0.0,1.0),.45));
    }

    vec2 environmentUv(vec3 ray, float rotation) {
      ray.xz = rotate(ray.xz, rotation);
      return vec2(atan(ray.x,ray.z) / TAU + .5, acos(clamp(ray.y,-1.0,1.0)) / PI);
    }

    vec3 sky(vec3 ray, float blur) {
      ray = normalize(ray);
      vec3 photographedA = texture2D(environmentA, environmentUv(ray, environmentRotation.x), blur).rgb * environmentScale.x;
      vec3 photographedB = texture2D(environmentB, environmentUv(ray, environmentRotation.y), blur).rgb * environmentScale.y;
      vec3 photograph = mix(photographedA, photographedB, environmentMix);
      photograph *= mix(vec3(1.0),vec3(1.08,1.0,.87),mood.z);
      return mix(analyticSky(ray), photograph, environmentReady);
    }

    vec3 reflectedSky(vec3 direction, float roughness) {
      // Four radiance samples approximate a broad lobe after ripples become
      // subpixel. This retains texture energy without a sparkling horizon.
      vec3 tangent = normalize(cross(direction, vec3(0.0,1.0,0.001)));
      vec3 bitangent = cross(direction,tangent);
      float spread = roughness * roughness * .9;
      float blur = max(0.0,log2(max(1.0,spread * 180.0)));
      return (sky(direction + tangent * spread, blur) + sky(direction - tangent * spread, blur)
        + sky(direction + bitangent * spread, blur) + sky(direction - bitangent * spread, blur)) * .25;
    }

    float traceTerrain(vec3 origin, vec3 direction, float limit) {
      float travel = .3;
      float previous = 0.0;
      for (int i=0; i<192; i++) {
        if (float(i) >= 80.0 + renderQuality * 36.0) break;
        vec3 p = origin + direction * travel;
        if (travel >= limit || (p.y > 58.0 && direction.y > 0.0)) break;
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
          for (int j=0; j<12; j++) {
            if (float(j) >= 7.0 + renderQuality * 2.0) break;
            float mid = (lo + hi) * .5;
            vec3 q = origin + direction * mid;
            if (q.y > terrain(q.xz)) lo = mid; else hi = mid;
          }
          return (lo + hi) * .5;
        }
        previous = travel;
        float minimumStep = mix(.48,.10,clamp(renderQuality / 3.0,0.0,1.0));
        float angularStep = mix(.004,.0008,clamp(renderQuality / 3.0,0.0,1.0));
        travel += clamp(max(clearance / max(.35,1.2 - direction.y),minimumStep + travel * angularStep),minimumStep,50.0);
      }
      return limit;
    }

    float stoneTexture(vec3 p, vec3 weights) {
      return noise(p.yz) * weights.x + noise(p.zx) * weights.y + noise(p.xy) * weights.z;
    }

    vec3 landRadiance(vec3 p, vec3 direction, float travel, float reflectionBlur) {
      float epsilon = max(max(mix(.12,.025,clamp(renderQuality / 3.0,0.0,1.0)),travel / resolution.y * .3),reflectionBlur * .18);
      float h = terrain(p.xz);
      vec3 n = normalize(vec3(terrain(p.xz - vec2(epsilon,0.0)) - terrain(p.xz + vec2(epsilon,0.0)),
        2.0 * epsilon,terrain(p.xz - vec2(0.0,epsilon)) - terrain(p.xz + vec2(0.0,epsilon))));
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
      if (renderQuality > .5 && diffuse > .0) {
        float obstruction = terrain(p.xz + sunDirection.xz * 6.0) - h - sunDirection.y * 6.0;
        if (renderQuality > 1.5) obstruction = max(obstruction,
          (terrain(p.xz + sunDirection.xz * 18.0) - h - sunDirection.y * 18.0) * .5);
        terrainLight = 1.0 - smoothstep(.1,2.4,obstruction) * .7;
      }
      float occlusion = mix(.78,1.0,smoothstep(.15,.67,grain));
      vec3 color = albedo * (ambient * .68 * occlusion + sunColor * diffuse * terrainLight * (.32 + solarStrength * .18));
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
      float terrainTravel = 1600.0;
      if (sceneKind > .5 && ray.y < .75) {
        float landLimit = ray.y < -.001 ? min(1600.0,(cameraPosition.y + max(.4,waveHeight)) / -ray.y) : 1600.0;
        float hit = traceTerrain(cameraPosition,ray,landLimit);
        if (hit < landLimit) terrainTravel = hit;
      }
      float waterTravel = 12000.0;
      if (ray.y < -.0002 && terrainTravel > 1.0) {
        float travel = min(12000.0, cameraPosition.y / -ray.y);
        // Bracket the first surface crossing, then refine it. Translation moves
        // through a fixed world-space field rather than sliding the image.
        float extent = max(.4, waveHeight * 1.6);
        float nearT = max(.0, (cameraPosition.y - extent) / -ray.y);
        float farT = min(12000.0, (cameraPosition.y + extent) / -ray.y);
        for (int i=0; i<8; i++) {
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
        vec3 wave = surface(position,travel,1.0);
        vec3 normal = normalize(vec3(-wave.y,1.0,-wave.z));
        vec3 view = -ray;
        float ndotv = max(.001,dot(normal,view));
        vec3 reflected = reflect(ray,normal);
        reflected.y = max(.003,reflected.y);
        float fresnel = .0204 + .9796 * pow(1.0-clamp(ndotv,0.0,1.0),5.0);
        float distanceRoughness = smoothstep(20.0,300.0,travel) * .13;
        float roughness = .11 + wind / 20.0 * .085 + distanceRoughness;
        vec3 reflectedLight = reflectedSky(reflected,roughness);
        if (sceneKind > .5 && renderQuality > .5 && reflected.y < .42) {
          vec3 reflectionOrigin = vec3(position.x,wave.x + .045,position.y);
          float landReflection = traceTerrain(reflectionOrigin,reflected,1200.0);
          if (landReflection < 1200.0) {
            float reflectionBlur = max(.03,landReflection * roughness * roughness * .55);
            vec3 reflectedLand = landRadiance(reflectionOrigin + reflected * landReflection,reflected,landReflection,reflectionBlur);
            float scatter = clamp(roughness * 2.4 + smoothstep(80.0,600.0,landReflection) * .12,.3,.75);
            reflectedLight = mix(reflectedLand,reflectedLight,scatter);
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
          vec3 attenuation = exp(-depth * vec3(.36,.14,.075) * (1.0 + .65 / max(.3,ndotv)));
          vec3 shallowScatter = vec3(.018,.105,.095) * (ambient * .55 + .4);
          transmitted = mix(transmitted,sand * attenuation + shallowScatter * (1.0 - attenuation),exp(-depth * .045));
          float rippleShade = .92 + .08 * sin(position.x * 6.0 + sin(position.y * 3.0 + time * .25));
          transmitted *= mix(1.0,rippleShade,exp(-depth * .6));
        }
        color = mix(transmitted,reflectedLight,fresnel);
        // A restrained microfacet solar lobe supplements the photographed disk.
        vec3 halfVector = normalize(sunDirection + view);
        float ndotl = max(.0,dot(normal,sunDirection));
        float ndoth = max(.0,dot(normal,halfVector));
        float vdoth = max(.0,dot(view,halfVector));
        float a = max(.035,roughness * roughness);
        float a2 = a * a;
        float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
        float distribution = a2 / max(.00001,PI * denom * denom);
        float k = roughness * roughness * .5;
        float masking = ndotv / (ndotv * (1.0-k)+k) * ndotl / max(.001,ndotl*(1.0-k)+k);
        float solarFresnel = .0204 + .9796 * pow(1.0-vdoth,5.0);
        float specular = distribution * masking * solarFresnel / max(.01,4.0*ndotv);
        color += mix(vec3(1.0,.73,.44),vec3(1.0,.97,.90),mood.y) * min(specular,.5) * solarStrength * .16;
        // Calm seas have little foam; only steep, energetic crests catch it.
        float slope = length(wave.yz);
        float foam = smoothstep(.55,.95,slope) * smoothstep(5.0,15.0,wind) * .16;
        if (foamReady > .5) {
          vec2 foamUv = (position - foamMapping.xy) / max(1.0,foamMapping.z);
          vec2 edge = smoothstep(vec2(0.0),vec2(.06),foamUv)
            * (1.0 - smoothstep(vec2(.94),vec2(1.0),foamUv));
          float history = texture2D(foamField,clamp(foamUv,vec2(0.0),vec2(1.0))).r;
          vec2 drift = vec2(.436,.900) * time * (.035 + wind * .013);
          float lace = smoothstep(.18,.78,noise((position - drift) * 2.4));
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
        vec3 haze = sky(normalize(vec3(ray.x,.007,ray.z)),1.5);
        color = mix(color,haze,1.0-exp(-travel*.000075));
        color = mix(color,haze,smoothstep(-.0008,-.0002,ray.y));
      }
      if (sceneKind > .5 && terrainTravel < min(waterTravel,1600.0)) {
        color = landRadiance(cameraPosition + ray * terrainTravel,ray,terrainTravel,0.0);
      }
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
