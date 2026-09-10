(() => {
  'use strict';

  // Tessendorf's deep-water spectrum, evaluated entirely on the GPU after a
  // wind change. Two independent length scales avoid a single repeating swell.
  // https://jtessen.people.clemson.edu/reports/papers_files/waterslides2001.pdf
  // https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models
  const TAU = Math.PI * 2;
  const GRAVITY = 9.81;
  const WIND_ANGLE = 1.12;
  const SWELL_PROFILES = Object.freeze({
    balanced: Object.freeze({ lengthScale: 1, cutoff: .76, ripple: 1, spread: 1 }),
    long: Object.freeze({ lengthScale: 1.65, cutoff: .56, ripple: .7, spread: 1.7 }),
    chop: Object.freeze({ lengthScale: .62, cutoff: 1.05, ripple: 1.6, spread: .7 }),
  });
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  const VERTEX_SOURCE = `
    attribute vec2 aPosition;
    varying vec2 vUv;
    void main() {
      vUv = aPosition * 0.5 + 0.5;
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }
  `;

  const EVOLUTION_SOURCE = `
    precision highp float;
    uniform sampler2D uSpectrum;
    uniform float uSize;
    uniform float uLength;
    uniform float uTime;
    varying vec2 vUv;
    vec2 multiplyComplex(vec2 a, vec2 b) {
      return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    void main() {
      vec2 index = floor(vUv * uSize);
      vec2 signedIndex = index - step(vec2(uSize * 0.5), index) * uSize;
      vec2 waveNumber = signedIndex * (6.28318530718 / uLength);
      float omega = sqrt(9.81 * length(waveNumber));
      float phase = mod(omega * uTime, 6.28318530718);
      vec2 rotation = vec2(cos(phase), sin(phase));
      vec4 initial = texture2D(uSpectrum, vUv);
      vec2 height = multiplyComplex(initial.xy, rotation)
        + multiplyComplex(initial.zw, vec2(rotation.x, -rotation.y));
      // Nyquist bins are self-conjugate and have zero sampled derivative.
      vec2 derivativeK = waveNumber * (1.0 - step(vec2(0.25),
        vec2(0.5) - abs(index - vec2(uSize * 0.5))));
      // Pack slope X + i*slope Z as a second complex transform. Its inverse
      // yields both real spatial derivatives without two additional FFTs.
      vec2 slopes = multiplyComplex(height, vec2(-derivativeK.y, derivativeK.x));
      gl_FragColor = vec4(height, slopes);
    }
  `;

  const FFT_SOURCE = `
    precision highp float;
    uniform sampler2D uInput;
    uniform float uSize;
    uniform float uSubSize;
    uniform vec2 uAxis;
    varying vec2 vUv;
    vec2 multiplyComplex(vec2 a, vec2 b) {
      return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    void main() {
      float index = floor(dot(vUv, uAxis) * uSize);
      float halfSize = uSubSize * 0.5;
      float sourceIndex = floor(index / uSubSize) * halfSize + mod(index, halfSize);
      float original = dot(vUv, uAxis);
      vec2 evenUv = vUv + uAxis * ((sourceIndex + 0.5) / uSize - original);
      vec2 oddUv = evenUv + uAxis * 0.5;
      vec4 evenValue = texture2D(uInput, evenUv);
      vec4 oddValue = texture2D(uInput, oddUv);
      float angle = 6.28318530718 * mod(index, uSubSize) / uSubSize;
      vec2 twiddle = vec2(cos(angle), sin(angle));
      gl_FragColor = evenValue + vec4(
        multiplyComplex(twiddle, oddValue.xy),
        multiplyComplex(twiddle, oddValue.zw));
    }
  `;

  const PACK_SOURCE = `
    precision highp float;
    uniform sampler2D uInput;
    uniform float uScale;
    varying vec2 vUv;
    void main() {
      vec4 spatial = texture2D(uInput, vUv);
      gl_FragColor = vec4(spatial.r, spatial.b, spatial.a, 0.0) * uScale;
    }
  `;

  const FOAM_SOURCE = `
    precision highp float;
    uniform sampler2D uPrevious;
    uniform sampler2D uSwell;
    uniform sampler2D uRipple;
    uniform vec2 uLengths;
    uniform vec2 uOrigin;
    uniform vec2 uPreviousOrigin;
    uniform float uLength;
    uniform float uSize;
    uniform float uDelta;
    uniform float uTime;
    uniform float uWind;
    uniform float uHeight;
    uniform float uScene;
    uniform float uReset;
    varying vec2 vUv;
    /* OCEAN_SHORE */
    vec2 rotate(vec2 p, float angle) {
      float c = cos(angle), s = sin(angle);
      return vec2(c * p.x - s * p.y,s * p.x + c * p.y);
    }
    float previousDensity(vec2 uv) {
      // Explicit bilinear advection also works on devices without float-linear
      // textures. Otherwise sub-texel drift would stick or disappear in steps.
      vec2 pixel = uv * uSize - .5;
      vec2 base = (floor(pixel) + .5) / uSize;
      vec2 f = fract(pixel);
      float a = texture2D(uPrevious,base).r;
      float b = texture2D(uPrevious,base + vec2(1.0 / uSize,0.0)).r;
      float c = texture2D(uPrevious,base + vec2(0.0,1.0 / uSize)).r;
      float d = texture2D(uPrevious,base + vec2(1.0 / uSize)).r;
      return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
    }
    void main() {
      vec2 world = uOrigin + vUv * uLength;
      vec3 wave = texture2D(uSwell,world / uLengths.x).rgb;
      vec3 ripple = texture2D(uRipple,rotate(world,.42) / uLengths.y + vec2(.173,.387)).rgb;
      ripple.yz = rotate(ripple.yz,-.42);
      wave += ripple;
      float sigma = max(.08,uHeight * .25);
      float shape = clamp(wave.x / sigma,-2.5,2.5);
      wave.x += sigma * .12 * (shape * shape - 1.0) * smoothstep(0.0,.15,uHeight);
      wave.yz *= max(.4,1.0 + .24 * shape);
      vec3 shore = shoreGeometry(world);
      if (uScene > .5) wave = nearshoreSurfaceGeometry(world,shore,wave,uTime,uHeight);
      float crest = smoothstep(.12,.95,wave.x / sigma);
      float source = smoothstep(.38,.9,length(wave.yz)) * smoothstep(4.0,14.0,uWind) * crest;
      vec2 drift = vec2(.436,.900) * (.035 + uWind * .013);
      float wet = 1.0;
      if (uScene > .5) {
        float width = max(.24,uHeight * .3);
        float breaker = exp(-pow((shore.x - max(.1,uHeight * .35)) / width,2.0));
        source = max(source,breaker * smoothstep(.015,.13,wave.x) * .65 * smoothstep(.03,.2,uHeight));
        drift -= normalize(shore.yz) * .11 * exp(-max(0.0,shore.x));
        wet = smoothstep(-.22,.02,shore.x + wave.x);
      }
      vec2 previousUv = (world - drift * uDelta - uPreviousOrigin) / uLength;
      float inside = step(0.0,previousUv.x) * step(0.0,previousUv.y)
        * step(previousUv.x,1.0) * step(previousUv.y,1.0);
      float history = previousDensity(clamp(previousUv,vec2(0.0),vec2(1.0)))
        * inside * (1.0 - uReset) * exp(-uDelta * .24);
      // A rate-based source and exponential decay give the same lifetime at
      // 30/60 Hz; previous-frame density is never replaced by procedural noise.
      float density = history + (1.0 - history) * (1.0 - exp(-source * uDelta * 2.4));
      density *= mix(exp(-uDelta * 2.5),1.0,wet);
      gl_FragColor = vec4(clamp(density,0.0,1.0),0.0,0.0,1.0);
    }
  `;

  const makeGaussian = (size, seed) => {
    const values = new Float32Array(size * size * 2);
    let state = seed >>> 0;
    const random = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state + 0.5) / 4294967296;
    };
    for (let i = 0; i < values.length; i += 2) {
      const radius = Math.sqrt(-2 * Math.log(random()));
      const angle = TAU * random();
      values[i] = radius * Math.cos(angle);
      values[i + 1] = radius * Math.sin(angle);
    }
    return values;
  };

  const create = (gl, options = {}) => {
    if (!gl || gl.isContextLost()) return null;
    const precision = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    if (!precision || precision.precision < 16 || !gl.getExtension('OES_texture_float')) return null;
    gl.getExtension('WEBGL_color_buffer_float');
    const linear = Boolean(gl.getExtension('OES_texture_float_linear'));
    let mipmapped = linear;
    const size = options.size === 256 ? 256 : 128;
    const resources = { textures: [], programs: [], shaders: [], buffers: [], framebuffers: [] };
    let disposed = false;
    const previous = {
      framebuffer: gl.getParameter(gl.FRAMEBUFFER_BINDING),
      arrayBuffer: gl.getParameter(gl.ARRAY_BUFFER_BINDING),
      activeTexture: gl.getParameter(gl.ACTIVE_TEXTURE),
      texture: gl.getParameter(gl.TEXTURE_BINDING_2D),
    };

    const dispose = () => {
      if (disposed) return;
      disposed = true;
      resources.textures.forEach((texture) => gl.deleteTexture(texture));
      resources.programs.forEach((program) => gl.deleteProgram(program));
      resources.shaders.forEach((shader) => gl.deleteShader(shader));
      resources.buffers.forEach((buffer) => gl.deleteBuffer(buffer));
      resources.framebuffers.forEach((framebuffer) => gl.deleteFramebuffer(framebuffer));
    };

    const makeProgram = (fragmentSource, names) => {
      const compile = (type, source) => {
        const shader = gl.createShader(type);
        if (!shader) throw new Error('Unable to allocate ocean spectrum shader.');
        resources.shaders.push(shader);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(gl.getShaderInfoLog(shader) || 'Ocean spectrum shader compilation failed.');
        }
        return shader;
      };
      const program = gl.createProgram();
      if (!program) throw new Error('Unable to allocate ocean spectrum program.');
      resources.programs.push(program);
      gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX_SOURCE));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
      gl.bindAttribLocation(program, 0, 'aPosition');
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || 'Ocean spectrum program linking failed.');
      }
      const uniforms = {};
      names.forEach((name) => { uniforms[name] = gl.getUniformLocation(program, name); });
      return { program, uniforms };
    };

    const makeTexture = (filtered, withMipmaps = true, repeat = true) => {
      const texture = gl.createTexture();
      if (!texture) throw new Error('Unable to allocate ocean spectrum texture.');
      resources.textures.push(texture);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const filter = filtered && linear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.FLOAT, null);
      if (filtered && withMipmaps && mipmapped) {
        gl.generateMipmap(gl.TEXTURE_2D);
        mipmapped = gl.getError() === gl.NO_ERROR;
        if (mipmapped) gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      }
      return texture;
    };

    try {
      const framebuffer = gl.createFramebuffer();
      if (!framebuffer) throw new Error('Unable to allocate ocean spectrum framebuffer.');
      resources.framebuffers.push(framebuffer);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const ping = makeTexture(false);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ping, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        dispose();
        return null;
      }
      const pong = makeTexture(false);
      const cascades = [
        { length: 180, seed: 78197, swell: true },
        { length: 18, seed: 249433, swell: false },
      ].map((config) => ({
        ...config,
        texture: makeTexture(true),
        initial: makeTexture(false),
        gaussian: makeGaussian(size, config.seed),
        coefficients: new Float32Array(size * size * 2),
        upload: new Float32Array(size * size * 4),
      }));
      const fields = cascades.map((cascade) => ({ texture: cascade.texture, size, length: cascade.length }));
      const evolution = makeProgram(EVOLUTION_SOURCE, ['uSpectrum', 'uSize', 'uLength', 'uTime']);
      const fft = makeProgram(FFT_SOURCE, ['uInput', 'uSize', 'uSubSize', 'uAxis']);
      const pack = makeProgram(PACK_SOURCE, ['uInput', 'uScale']);
      const shoreSource = window.OceanWaveShaders?.shoreSource || `
        vec3 shoreGeometry(vec2 p) { return vec3(100.0,.085,0.0); }
        vec3 nearshoreSurfaceGeometry(vec2 p, vec3 shore, vec3 wave, float seconds, float height) { return wave; }
      `;
      const foamProgram = makeProgram(FOAM_SOURCE.replace('/* OCEAN_SHORE */', shoreSource), [
        'uPrevious', 'uSwell', 'uRipple', 'uLengths', 'uOrigin', 'uPreviousOrigin', 'uLength',
        'uSize', 'uDelta', 'uTime', 'uWind', 'uHeight', 'uScene', 'uReset',
      ]);
      let foamPrevious = makeTexture(true, false, false);
      let foamTarget = makeTexture(true, false, false);
      // Stable descriptor; texture swaps after update. World mapping is
      // uv = (worldXZ - vec2(originX, originZ)) / length, with no rotation.
      const foam = { texture: foamPrevious, size, length: 180, originX: -90, originZ: -90 };
      let viewX = -90;
      let viewZ = -90;
      let scene = 'ocean';
      let viewVersion = 0;
      let previousViewVersion = -1;
      let previousScene = scene;
      const triangle = gl.createBuffer();
      if (!triangle) throw new Error('Unable to allocate ocean spectrum triangle.');
      resources.buffers.push(triangle);
      gl.bindBuffer(gl.ARRAY_BUFFER, triangle);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      let previousWind = NaN;
      let previousTime = NaN;
      let previousHeight = NaN;
      let previousSwell = '';

      const initialize = (cascade, wind, profile) => {
        const effectiveWind = cascade.swell ? 7.6 + wind * 0.25 : Math.max(0.8, wind);
        const largestWave = effectiveWind * effectiveWind / GRAVITY * (cascade.swell ? profile.lengthScale : 1);
        const damping = cascade.swell ? 0.7 : 0.028;
        const direction = WIND_ANGLE + (cascade.swell ? -0.3 : 0.16);
        const windX = Math.cos(direction);
        const windZ = Math.sin(direction);
        let energy = 0;
        for (let y = 0; y < size; y++) {
          const kz = (y < size / 2 ? y : y - size) * TAU / cascade.length;
          for (let x = 0; x < size; x++) {
            const kx = (x < size / 2 ? x : x - size) * TAU / cascade.length;
            const k2 = kx * kx + kz * kz;
            const index = (y * size + x) * 2;
            let power = 0;
            if (k2 > 0) {
              const k = Math.sqrt(k2);
              const alignment = (kx * windX + kz * windZ) / k;
              // Complementary broad bands retain a continuous range of scales.
              const longBand = Math.exp(-Math.pow(k / profile.cutoff, 4));
              const band = cascade.swell ? longBand : 1 - longBand;
              const directionality = 0.07 + 0.93 * Math.pow(alignment * alignment, cascade.swell ? profile.spread : 1);
              const travel = alignment > 0 ? 0.12 : 1;
              power = Math.exp(-1 / (k2 * largestWave * largestWave))
                * Math.exp(-k2 * damping * damping)
                * directionality * travel * band / (k2 * k2);
            }
            const scale = Math.sqrt(power * 0.5);
            cascade.coefficients[index] = cascade.gaussian[index] * scale;
            cascade.coefficients[index + 1] = cascade.gaussian[index + 1] * scale;
            energy += power;
          }
        }
        // E[h^2] = 2*sum(P). An unnormalized inverse FFT retains this variance.
        const normalize = energy > 0 ? 1 / Math.sqrt(2 * energy) : 0;
        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            const index = (y * size + x) * 2;
            const mirror = (((size - y) % size) * size + (size - x) % size) * 2;
            const output = index * 2;
            cascade.upload[output] = cascade.coefficients[index] * normalize;
            cascade.upload[output + 1] = cascade.coefficients[index + 1] * normalize;
            cascade.upload[output + 2] = cascade.coefficients[mirror] * normalize;
            cascade.upload[output + 3] = -cascade.coefficients[mirror + 1] * normalize;
          }
        }
        gl.bindTexture(gl.TEXTURE_2D, cascade.initial);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, size, size, gl.RGBA, gl.FLOAT, cascade.upload);
      };

      const drawTo = (texture) => {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      };

      // Call before the scene draw. For efficiency there are no synchronous
      // state queries per frame: this binds its framebuffer, program, ARRAY_BUFFER,
      // attribute 0 and texture units 0-2, and disables blend/depth/scissor/cull.
      // The caller must bind its framebuffer, viewport, program and vertex data
      // after update(), then bind fields[].texture and foam.texture for its
      // own rendering pass. Call setView(scene, cameraX, cameraZ) beforehand.
      const setView = (value = 'ocean', x = 0, z = 0) => {
        const nextScene = value === 'cove' ? 'cove' : 'ocean';
        const texel = foam.length / size;
        const nextX = Math.floor((Number.isFinite(x) ? x : 0) / texel) * texel - foam.length * .5;
        const nextZ = Math.floor((Number.isFinite(z) ? z : 0) / texel) * texel - foam.length * .5;
        if (nextScene !== scene || nextX !== viewX || nextZ !== viewZ) viewVersion += 1;
        scene = nextScene;
        viewX = nextX;
        viewZ = nextZ;
      };
      const update = (timeSeconds, windMetresPerSecond, significantWaveHeightMetres, swell = 'balanced') => {
        if (disposed || gl.isContextLost()) return false;
        const time = Number.isFinite(timeSeconds) ? timeSeconds : 0;
        const wind = clamp(Number.isFinite(windMetresPerSecond) ? windMetresPerSecond : 2.4, 0, 25);
        const waveHeight = clamp(Number.isFinite(significantWaveHeightMetres) ? significantWaveHeightMetres : 0.65, 0, 8);
        const swellName = Object.prototype.hasOwnProperty.call(SWELL_PROFILES, swell) ? swell : 'balanced';
        const profile = SWELL_PROFILES[swellName];
        if (time === previousTime && wind === previousWind && waveHeight === previousHeight
          && swellName === previousSwell && viewVersion === previousViewVersion) return true;
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.SCISSOR_TEST);
        gl.disable(gl.CULL_FACE);
        gl.colorMask(true, true, true, true);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.viewport(0, 0, size, size);
        gl.bindBuffer(gl.ARRAY_BUFFER, triangle);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        if (wind !== previousWind || swellName !== previousSwell) {
          for (let index = 0; index < cascades.length; index++) initialize(cascades[index], wind, profile);
        }

        const rippleShare = clamp(clamp(0.12 + wind * 0.025, 0.12, 0.38) * profile.ripple, .07, .62);
        for (let index = 0; index < cascades.length; index++) {
          const cascade = cascades[index];
          gl.useProgram(evolution.program);
          gl.uniform1i(evolution.uniforms.uSpectrum, 0);
          gl.uniform1f(evolution.uniforms.uSize, size);
          gl.uniform1f(evolution.uniforms.uLength, cascade.length);
          gl.uniform1f(evolution.uniforms.uTime, time);
          gl.bindTexture(gl.TEXTURE_2D, cascade.initial);
          drawTo(ping);
          let source = ping;
          let target = pong;
          gl.useProgram(fft.program);
          gl.uniform1i(fft.uniforms.uInput, 0);
          gl.uniform1f(fft.uniforms.uSize, size);
          for (let axis = 0; axis < 2; axis++) {
            gl.uniform2f(fft.uniforms.uAxis, axis === 0 ? 1 : 0, axis === 1 ? 1 : 0);
            for (let subSize = 2; subSize <= size; subSize *= 2) {
              gl.uniform1f(fft.uniforms.uSubSize, subSize);
              gl.bindTexture(gl.TEXTURE_2D, source);
              drawTo(target);
              const next = source;
              source = target;
              target = next;
            }
          }
          gl.useProgram(pack.program);
          gl.uniform1i(pack.uniforms.uInput, 0);
          const share = cascade.swell ? Math.sqrt(1 - rippleShare * rippleShare) : rippleShare;
          gl.uniform1f(pack.uniforms.uScale, waveHeight * 0.25 * share);
          gl.bindTexture(gl.TEXTURE_2D, source);
          drawTo(cascade.texture);
          if (mipmapped) {
            gl.bindTexture(gl.TEXTURE_2D, cascade.texture);
            gl.generateMipmap(gl.TEXTURE_2D);
          }
        }
        const resetFoam = !Number.isFinite(previousTime) || time < previousTime || scene !== previousScene;
        // Advection is a backtrace and decay is exponential, so a slower frame
        // can consume its full elapsed time without Euler-integration drift.
        const delta = resetFoam ? 1 / 30 : Math.max(0, time - previousTime);
        const uniforms = foamProgram.uniforms;
        gl.useProgram(foamProgram.program);
        gl.uniform1i(uniforms.uPrevious, 0);
        gl.uniform1i(uniforms.uSwell, 1);
        gl.uniform1i(uniforms.uRipple, 2);
        gl.uniform2f(uniforms.uLengths, fields[0].length, fields[1].length);
        gl.uniform2f(uniforms.uOrigin, viewX, viewZ);
        gl.uniform2f(uniforms.uPreviousOrigin, foam.originX, foam.originZ);
        gl.uniform1f(uniforms.uLength, foam.length);
        gl.uniform1f(uniforms.uSize, size);
        gl.uniform1f(uniforms.uDelta, delta);
        gl.uniform1f(uniforms.uTime, time);
        gl.uniform1f(uniforms.uWind, wind);
        gl.uniform1f(uniforms.uHeight, waveHeight);
        gl.uniform1f(uniforms.uScene, scene === 'cove' ? 1 : 0);
        gl.uniform1f(uniforms.uReset, resetFoam ? 1 : 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, foamPrevious);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, fields[0].texture);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, fields[1].texture);
        drawTo(foamTarget);
        const completedFoam = foamTarget;
        foamTarget = foamPrevious;
        foamPrevious = completedFoam;
        foam.texture = completedFoam;
        foam.originX = viewX;
        foam.originZ = viewZ;
        previousWind = wind;
        previousTime = time;
        previousHeight = waveHeight;
        previousSwell = swellName;
        previousViewVersion = viewVersion;
        previousScene = scene;
        return true;
      };

      return { fields, foam, linear, mipmapped, setView, update, dispose };
    } catch {
      dispose();
      return null;
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, previous.framebuffer);
      gl.bindBuffer(gl.ARRAY_BUFFER, previous.arrayBuffer);
      gl.activeTexture(previous.activeTexture);
      gl.bindTexture(gl.TEXTURE_2D, previous.texture);
    }
  };

  window.OceanWaveSpectrum = { create };
})();
