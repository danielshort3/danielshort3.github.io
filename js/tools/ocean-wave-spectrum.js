(() => {
  'use strict';

  // Tessendorf's deep-water spectrum, evaluated entirely on the GPU after a
  // wind change. Two independent length scales avoid a single repeating swell.
  // https://jtessen.people.clemson.edu/reports/papers_files/waterslides2001.pdf
  // https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models
  const TAU = Math.PI * 2;
  const GRAVITY = 9.81;
  const WIND_ANGLE = 1.12;
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

    const makeTexture = (filtered) => {
      const texture = gl.createTexture();
      if (!texture) throw new Error('Unable to allocate ocean spectrum texture.');
      resources.textures.push(texture);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const filter = filtered && linear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.FLOAT, null);
      if (filtered && mipmapped) {
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
      const triangle = gl.createBuffer();
      if (!triangle) throw new Error('Unable to allocate ocean spectrum triangle.');
      resources.buffers.push(triangle);
      gl.bindBuffer(gl.ARRAY_BUFFER, triangle);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      let previousWind = NaN;
      let previousTime = NaN;
      let previousHeight = NaN;

      const initialize = (cascade, wind) => {
        const effectiveWind = cascade.swell ? 7.6 + wind * 0.25 : Math.max(0.8, wind);
        const largestWave = effectiveWind * effectiveWind / GRAVITY;
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
              const longBand = Math.exp(-Math.pow(k / 0.76, 4));
              const band = cascade.swell ? longBand : 1 - longBand;
              const directionality = 0.07 + 0.93 * alignment * alignment;
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
      // attribute 0 and texture unit 0, and disables blend/depth/scissor/cull.
      // The caller must bind its framebuffer, viewport, program and vertex data
      // after update(), then bind fields[].texture for its own rendering pass.
      const update = (timeSeconds, windMetresPerSecond, significantWaveHeightMetres) => {
        if (disposed || gl.isContextLost()) return false;
        const time = Number.isFinite(timeSeconds) ? timeSeconds : 0;
        const wind = clamp(Number.isFinite(windMetresPerSecond) ? windMetresPerSecond : 2.4, 0, 25);
        const waveHeight = clamp(Number.isFinite(significantWaveHeightMetres) ? significantWaveHeightMetres : 0.65, 0, 8);
        if (time === previousTime && wind === previousWind && waveHeight === previousHeight) return true;
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
        if (wind !== previousWind) cascades.forEach((cascade) => initialize(cascade, wind));

        const rippleShare = clamp(0.12 + wind * 0.025, 0.12, 0.38);
        cascades.forEach((cascade) => {
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
        });
        previousWind = wind;
        previousTime = time;
        previousHeight = waveHeight;
        return true;
      };

      return { fields, linear, mipmapped, update, dispose };
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
