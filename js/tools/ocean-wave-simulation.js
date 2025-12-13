(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const canvas = $('#ocean-canvas');
  const fallback = $('#ocean-fallback');
  const windInput = $('#ocean-wind');
  const heightInput = $('#ocean-height');
  const sunInput = $('#ocean-sun');
  const windValue = $('#ocean-wind-value');
  const heightValue = $('#ocean-height-value');
  const sunValue = $('#ocean-sun-value');
  const toggleBtn = $('#ocean-toggle');
  const resetBtn = $('#ocean-reset');

  if (!canvas || !windInput || !heightInput || !sunInput) return;

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const saturate = (value) => clamp(value, 0, 1);
  const mix = (a, b, t) => a * (1 - t) + b * t;
  const smoothstep = (edge0, edge1, x) => {
    const t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3 - 2 * t);
  };

  const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3v14H7zM14 5h3v14h-3z"></path></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l12-7-12-7Z"></path></svg>';

  const defaults = {
    windSpeed: parseFloat(windInput.value),
    waveHeight: parseFloat(heightInput.value),
    sunElevation: parseFloat(sunInput.value),
    yaw: 0,
    pitch: 0.22
  };

  const state = { ...defaults };

  let prefersReducedMotion = false;
  try {
    prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (err) {
    prefersReducedMotion = false;
  }

  let paused = prefersReducedMotion;
  let pausedByVisibility = false;
  let needsRender = true;

  const showFallback = (message) => {
    if (!fallback) return;
    fallback.hidden = false;
    if (message) fallback.textContent = message;
  };

  const hideFallback = () => {
    if (!fallback) return;
    fallback.hidden = true;
  };

  const computeSunDir = (elevationDeg) => {
    const elev = clamp(elevationDeg, 0, 89.9) * (Math.PI / 180);
    const az = 0.72;
    const x = Math.cos(elev) * Math.cos(az);
    const y = Math.sin(elev);
    const z = Math.cos(elev) * Math.sin(az);
    const len = Math.hypot(x, y, z) || 1;
    return [x / len, y / len, z / len];
  };

  let sunDir = computeSunDir(state.sunElevation);

  const setToggleUI = () => {
    if (!toggleBtn) return;
    toggleBtn.setAttribute('aria-pressed', paused ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', paused ? 'Resume animation' : 'Pause animation');
    toggleBtn.innerHTML = paused ? ICON_PLAY : ICON_PAUSE;
  };

  const updateReadouts = () => {
    windValue && (windValue.textContent = `${state.windSpeed.toFixed(1)} m/s`);
    heightValue && (heightValue.textContent = `${state.waveHeight.toFixed(1)} m`);
    sunValue && (sunValue.textContent = `${Math.round(state.sunElevation)}° sun`);
  };

  const initWebGLRenderer = () => {
    const attributes = {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance'
    };

    const gl = canvas.getContext('webgl2', attributes)
      || canvas.getContext('webgl2')
      || canvas.getContext('webgl', attributes)
      || canvas.getContext('webgl')
      || canvas.getContext('experimental-webgl', attributes)
      || canvas.getContext('experimental-webgl');
    if (!gl) return null;

    let lost = false;
    canvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      lost = true;
      paused = true;
      needsRender = false;
      showFallback('WebGL context was lost. Reload to try again.');
    });

    const VERT = `
      attribute vec2 aPos;
      varying vec2 vUv;
      void main() {
        vUv = aPos * 0.5 + 0.5;
        gl_Position = vec4(aPos, 0.0, 1.0);
      }
    `;

    const FRAG = `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif

      uniform vec2 uResolution;
      uniform float uTime;
      uniform float uWindSpeed;
      uniform float uWaveHeight;
      uniform vec3 uSunDir;
      uniform vec2 uView;

      varying vec2 vUv;

      float saturate(float x) {
        return clamp(x, 0.0, 1.0);
      }

      vec2 rot(vec2 v, float a) {
        float c = cos(a);
        float s = sin(a);
        return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
      }

      vec3 skyColor(vec3 rd, vec3 sunDir) {
        float t = saturate(rd.y * 0.5 + 0.5);
        float sunLow = saturate(1.0 - sunDir.y);
        vec3 horizon = mix(vec3(0.70, 0.82, 0.92), vec3(0.92, 0.70, 0.55), sunLow * 0.55);
        vec3 zenith = mix(vec3(0.05, 0.13, 0.22), vec3(0.04, 0.10, 0.18), sunLow * 0.35);
        vec3 sky = mix(horizon, zenith, pow(t, 1.25));

        float sunAmt = max(dot(rd, sunDir), 0.0);
        vec3 sunCol = mix(vec3(1.00, 0.58, 0.36), vec3(1.00, 0.96, 0.90), saturate(sunDir.y * 1.4));
        sky += sunCol * pow(sunAmt, 256.0) * 1.35;
        sky += sunCol * pow(sunAmt, 18.0) * 0.10;
        return sky;
      }

      vec3 waveTerm(vec2 p, vec2 dir, float k, float amp, float w, float time) {
        float ph = dot(dir, p) * k + time * w;
        float s = sin(ph);
        float c = cos(ph);
        return vec3(amp * s, amp * k * dir.x * c, amp * k * dir.y * c);
      }

      vec3 seaHeightDeriv(vec2 p, float time, float windSpeed, float waveHeight) {
        float windNorm = saturate(windSpeed / 18.0);
        float speedMul = mix(0.70, 1.70, windNorm);

        float ampBase = clamp(waveHeight * 0.5, 0.04, 1.8);
        vec2 windDir = normalize(vec2(0.82, 0.57));

        vec3 acc = vec3(0.0);
        vec3 t0 = waveTerm(p, rot(windDir, 0.0),   0.35, ampBase * 0.55, sqrt(9.81 * 0.35) * speedMul, time);
        vec3 t1 = waveTerm(p, rot(windDir, 0.35),  0.65, ampBase * 0.25, sqrt(9.81 * 0.65) * speedMul, time);
        vec3 t2 = waveTerm(p, rot(windDir,-0.35),  1.05, ampBase * 0.15, sqrt(9.81 * 1.05) * speedMul, time);
        vec3 t3 = waveTerm(p, rot(windDir, 0.70),  1.60, ampBase * 0.08, sqrt(9.81 * 1.60) * speedMul, time);
        vec3 t4 = waveTerm(p, rot(windDir,-0.70),  2.50, ampBase * 0.05, sqrt(9.81 * 2.50) * speedMul, time);
        vec3 t5 = waveTerm(p, rot(windDir, 1.28),  3.60, ampBase * 0.03, sqrt(9.81 * 3.60) * speedMul, time);
        acc += t0 + t1 + t2 + t3 + t4 + t5;

        float rippleAmp = mix(0.006, 0.026, windNorm) * clamp(waveHeight, 0.2, 3.0);
        vec2 rp = p * 4.2;
        float rph = sin(dot(rp, vec2(1.1, 0.9)) + time * (1.8 + windNorm * 1.6));
        float rph2 = sin(dot(rp, vec2(-0.8, 1.2)) - time * (2.1 + windNorm * 1.9));
        float rip = rph * rph2;
        acc.x += rippleAmp * rip;
        acc.yz += rippleAmp * vec2(0.18, 0.16) * (rph + rph2);

        return acc;
      }

      void main() {
        vec2 frag = gl_FragCoord.xy;
        vec2 uv = (frag / uResolution) * 2.0 - 1.0;
        uv.x *= uResolution.x / uResolution.y;

        float yaw = uView.x;
        float pitch = uView.y;

        vec3 ro = vec3(0.0, 1.75, 4.2);
        vec3 f0 = normalize(vec3(sin(yaw), 0.0, -cos(yaw)));
        vec3 r = normalize(vec3(cos(yaw), 0.0, sin(yaw)));
        vec3 u0 = vec3(0.0, 1.0, 0.0);
        vec3 f = normalize(f0 * cos(pitch) - u0 * sin(pitch));
        vec3 u = normalize(cross(r, f));

        float fov = 1.10;
        vec3 rd = normalize(f + uv.x * r * fov + uv.y * u * fov);

        vec3 sky = skyColor(rd, uSunDir);

        if (rd.y >= -0.02) {
          vec3 outSky = sky;
          float vignette = smoothstep(1.45, 0.25, length(uv));
          outSky *= 0.86 + vignette * 0.14;
          outSky = pow(outSky, vec3(1.0 / 2.2));
          gl_FragColor = vec4(outSky, 1.0);
          return;
        }

        float tPlane = (-ro.y) / rd.y;
        float t = min(tPlane, 520.0);
        vec3 hit = ro + rd * t;

        float time = uTime;
        vec2 p = hit.xz;

        if (t < 90.0 && abs(rd.y) > 0.06) {
          for (int i = 0; i < 4; i++) {
            vec3 hdIt = seaHeightDeriv(p, time, uWindSpeed, uWaveHeight);
            float diff = hit.y - hdIt.x;
            t -= diff / rd.y;
            t = clamp(t, 0.0, 520.0);
            hit = ro + rd * t;
            p = hit.xz;
          }
        }

        vec3 hd = seaHeightDeriv(p, time, uWindSpeed, uWaveHeight);
        vec3 pos = vec3(hit.x, hd.x, hit.z);
        vec3 n = normalize(vec3(-hd.y, 1.0, -hd.z));

        vec3 v = normalize(ro - pos);
        vec3 sunDir = normalize(uSunDir);
        float diff = saturate(dot(n, sunDir));

        float ndv = saturate(dot(n, v));
        float fres = pow(1.0 - ndv, 5.0);
        float reflectance = 0.03 + 0.97 * fres;

        float sunLerp = saturate(sunDir.y * 1.4);
        vec3 sunCol = mix(vec3(1.00, 0.62, 0.40), vec3(1.00, 0.96, 0.92), sunLerp);

        vec3 reflDir = reflect(-v, n);
        vec3 refl = skyColor(reflDir, sunDir);

        vec3 deepWater = vec3(0.02, 0.09, 0.12);
        vec3 shallowWater = vec3(0.03, 0.14, 0.15);
        float crest = saturate((pos.y / max(0.06, uWaveHeight * 0.45)) * 0.55 + 0.5);
        vec3 water = mix(deepWater, shallowWater, crest);
        water += sunCol * diff * 0.055;

        vec3 hVec = normalize(v + sunDir);
        float spec = pow(saturate(dot(n, hVec)), 180.0);
        vec3 color = mix(water, refl, reflectance);
        color += sunCol * spec * (0.35 + 0.65 * fres);

        float slope = saturate(1.0 - n.y);
        float foam = smoothstep(0.62, 0.98, crest) * smoothstep(0.10, 0.45, slope);
        float noiseA = sin(dot(p, vec2(0.18, 0.14)) * 26.0 + time * 0.8);
        float noiseB = sin(dot(p, vec2(-0.12, 0.20)) * 31.0 - time * 0.9);
        foam *= 0.70 + 0.30 * (noiseA * noiseB);
        foam = saturate(foam);
        color = mix(color, vec3(0.92, 0.96, 0.98), foam * 0.85);

        float fog = 1.0 - exp(-t * 0.018);
        vec3 haze = mix(sky, vec3(0.65, 0.76, 0.86), 0.25);
        color = mix(color, haze, fog);

        float vignette = smoothstep(1.45, 0.25, length(uv));
        color *= 0.86 + vignette * 0.14;

        color = pow(color, vec3(1.0 / 2.2));
        gl_FragColor = vec4(color, 1.0);
      }
    `;

    const compileShader = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      const ok = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
      if (ok) return shader;
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      return { error: log || 'Shader compile failed.' };
    };

    const linkProgram = (vertexShader, fragmentShader) => {
      const program = gl.createProgram();
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      const ok = gl.getProgramParameter(program, gl.LINK_STATUS);
      if (ok) return program;
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      return { error: log || 'Program link failed.' };
    };

    const vertexShader = compileShader(gl.VERTEX_SHADER, VERT);
    if (vertexShader.error) {
      showFallback(`Vertex shader error: ${vertexShader.error}`);
      return null;
    }

    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, FRAG);
    if (fragmentShader.error) {
      showFallback(`Fragment shader error: ${fragmentShader.error}`);
      return null;
    }

    const program = linkProgram(vertexShader, fragmentShader);
    if (program.error) {
      showFallback(`WebGL program error: ${program.error}`);
      return null;
    }

    const attribPos = gl.getAttribLocation(program, 'aPos');
    const uniforms = {
      resolution: gl.getUniformLocation(program, 'uResolution'),
      time: gl.getUniformLocation(program, 'uTime'),
      windSpeed: gl.getUniformLocation(program, 'uWindSpeed'),
      waveHeight: gl.getUniformLocation(program, 'uWaveHeight'),
      sunDir: gl.getUniformLocation(program, 'uSunDir'),
      view: gl.getUniformLocation(program, 'uView')
    };

    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      1, 1
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(attribPos);
    gl.vertexAttribPointer(attribPos, 2, gl.FLOAT, false, 0, 0);

    const maxDpr = 2;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = clamp(window.devicePixelRatio || 1, 1, maxDpr);
      const width = Math.max(2, Math.floor(rect.width * dpr));
      const height = Math.max(2, Math.floor(rect.height * dpr));
      if (canvas.width === width && canvas.height === height) return;
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
      gl.uniform2f(uniforms.resolution, width, height);
      needsRender = true;
    };

    const render = (timeSec) => {
      if (lost) return;
      gl.uniform1f(uniforms.time, timeSec);
      gl.uniform1f(uniforms.windSpeed, state.windSpeed);
      gl.uniform1f(uniforms.waveHeight, state.waveHeight);
      gl.uniform3f(uniforms.sunDir, sunDir[0], sunDir[1], sunDir[2]);
      gl.uniform2f(uniforms.view, state.yaw, state.pitch);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    return { resize, render, frameIntervalMs: 0 };
  };

  const initCanvasRenderer = () => {
    let ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const invGamma = 1 / 2.2;
    const roX = 0;
    const roY = 1.75;
    const roZ = 4.2;
    const fov = 1.10;

    const deepWater = [0.02, 0.09, 0.12];
    const shallowWater = [0.03, 0.14, 0.15];
    const foamColor = [0.92, 0.96, 0.98];

    const windBaseX = 0.82;
    const windBaseY = 0.57;
    const windLen = Math.hypot(windBaseX, windBaseY) || 1;
    const windDirX = windBaseX / windLen;
    const windDirY = windBaseY / windLen;

    const gravity = 9.81;
    const waveAngles = [0.0, 0.35, -0.35, 0.70, -0.70, 1.28];
    const waveKs = [0.35, 0.65, 1.05, 1.60, 2.50, 3.60];
    const waveAmpMuls = [0.55, 0.25, 0.15, 0.08, 0.05, 0.03];
    const waveBaseWs = waveKs.map((k) => Math.sqrt(gravity * k));
    const waveDirX = waveAngles.map((a) => {
      const c = Math.cos(a);
      const s = Math.sin(a);
      return c * windDirX - s * windDirY;
    });
    const waveDirY = waveAngles.map((a) => {
      const c = Math.cos(a);
      const s = Math.sin(a);
      return s * windDirX + c * windDirY;
    });

    const skyOut = [0, 0, 0];
    const hdOut = [0, 0, 0];

    let width = 0;
    let height = 0;
    let imageData = null;
    let data = null;
    let uvX = null;
    let uvY = null;
    let vignetteMul = null;

    let cachedSpeedMul = 1;
    let cachedAmpBase = 1;
    let cachedRippleAmp = 0.01;
    let cachedWindNorm = 0;
    let cachedWaveHeight = 1;

    const invLen3 = (x, y, z) => {
      const len = Math.sqrt(x * x + y * y + z * z);
      return len ? 1 / len : 1;
    };

    const skyColor = (rdX, rdY, rdZ, sunX, sunY, sunZ) => {
      const t = saturate(rdY * 0.5 + 0.5);
      const sunLow = saturate(1 - sunY);
      const horizonT = sunLow * 0.55;
      const zenithT = sunLow * 0.35;

      const horizonR = mix(0.70, 0.92, horizonT);
      const horizonG = mix(0.82, 0.70, horizonT);
      const horizonB = mix(0.92, 0.55, horizonT);

      const zenithR = mix(0.05, 0.04, zenithT);
      const zenithG = mix(0.13, 0.10, zenithT);
      const zenithB = mix(0.22, 0.18, zenithT);

      const tt = Math.pow(t, 1.25);
      let skyR = mix(horizonR, zenithR, tt);
      let skyG = mix(horizonG, zenithG, tt);
      let skyB = mix(horizonB, zenithB, tt);

      const sunAmt = Math.max(rdX * sunX + rdY * sunY + rdZ * sunZ, 0);
      const sunColT = saturate(sunY * 1.4);
      const sunColG = mix(0.58, 0.96, sunColT);
      const sunColB = mix(0.36, 0.90, sunColT);
      const sun256 = Math.pow(sunAmt, 256) * 1.35;
      const sun18 = Math.pow(sunAmt, 18) * 0.10;

      skyR += (sun256 + sun18);
      skyG += sunColG * (sun256 + sun18);
      skyB += sunColB * (sun256 + sun18);

      skyOut[0] = skyR;
      skyOut[1] = skyG;
      skyOut[2] = skyB;
    };

    const seaHeightDeriv = (pX, pY, timeSec) => {
      let h = 0;
      let dX = 0;
      let dY = 0;

      for (let i = 0; i < 6; i++) {
        const k = waveKs[i];
        const amp = cachedAmpBase * waveAmpMuls[i];
        const w = waveBaseWs[i] * cachedSpeedMul;
        const dirX = waveDirX[i];
        const dirY = waveDirY[i];
        const ph = (dirX * pX + dirY * pY) * k + timeSec * w;
        const s = Math.sin(ph);
        const c = Math.cos(ph);
        h += amp * s;
        const common = amp * k * c;
        dX += common * dirX;
        dY += common * dirY;
      }

      const rpX = pX * 4.2;
      const rpY = pY * 4.2;
      const rph = Math.sin(rpX * 1.1 + rpY * 0.9 + timeSec * (1.8 + cachedWindNorm * 1.6));
      const rph2 = Math.sin(rpX * -0.8 + rpY * 1.2 - timeSec * (2.1 + cachedWindNorm * 1.9));
      const rip = rph * rph2;
      h += cachedRippleAmp * rip;
      const rSum = (rph + rph2);
      dX += cachedRippleAmp * 0.18 * rSum;
      dY += cachedRippleAmp * 0.16 * rSum;

      hdOut[0] = h;
      hdOut[1] = dX;
      hdOut[2] = dY;
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
      const scale = 0.55;
      const minW = 220;
      const maxW = 360;
      const targetW = clamp(Math.floor(rect.width * dpr * scale), minW, maxW);
      const aspect = rect.width ? rect.height / rect.width : (9 / 16);
      const targetH = Math.max(2, Math.floor(targetW * aspect));
      if (targetW === width && targetH === height) return;

      width = targetW;
      height = targetH;
      canvas.width = width;
      canvas.height = height;
      imageData = ctx.createImageData(width, height);
      data = imageData.data;

      const aspectFix = width / height;
      uvX = new Float32Array(width);
      for (let x = 0; x < width; x++) {
        uvX[x] = (((x + 0.5) / width) * 2 - 1) * aspectFix;
      }
      uvY = new Float32Array(height);
      for (let y = 0; y < height; y++) {
        uvY[y] = 1 - ((y + 0.5) / height) * 2;
      }

      vignetteMul = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        const v = uvY[y];
        const row = y * width;
        for (let x = 0; x < width; x++) {
          const u = uvX[x];
          const len = Math.sqrt(u * u + v * v);
          const vig = smoothstep(1.45, 0.25, len);
          vignetteMul[row + x] = 0.86 + vig * 0.14;
        }
      }

      needsRender = true;
    };

    const render = (timeSec) => {
      if (!imageData) return;

      cachedWindNorm = saturate(state.windSpeed / 18);
      cachedSpeedMul = 0.70 + cachedWindNorm;
      cachedWaveHeight = state.waveHeight;
      cachedAmpBase = clamp(cachedWaveHeight * 0.5, 0.04, 1.8);
      cachedRippleAmp = (0.006 + cachedWindNorm * 0.020) * clamp(cachedWaveHeight, 0.2, 3.0);

      const yaw = state.yaw;
      const pitch = state.pitch;
      const sinYaw = Math.sin(yaw);
      const cosYaw = Math.cos(yaw);
      const cosPitch = Math.cos(pitch);
      const sinPitch = Math.sin(pitch);

      const fX = sinYaw * cosPitch;
      const fY = -sinPitch;
      const fZ = -cosYaw * cosPitch;

      const rX = cosYaw;
      const rY = 0;
      const rZ = sinYaw;

      const uX = sinYaw * sinPitch;
      const uY = cosPitch;
      const uZ = -cosYaw * sinPitch;

      const sunX = sunDir[0];
      const sunY = sunDir[1];
      const sunZ = sunDir[2];

      const crestDen = Math.max(0.06, cachedWaveHeight * 0.45);
      const sunColT = saturate(sunY * 1.4);
      const sunCol = [1.0, mix(0.62, 0.96, sunColT), mix(0.40, 0.92, sunColT)];
      const outMul = 255;

      for (let y = 0; y < height; y++) {
        const v = uvY[y];
        const row = y * width;
        let idx = row * 4;
        for (let x = 0; x < width; x++) {
          const u = uvX[x];
          const mulVig = vignetteMul[row + x];

          const rayX = fX + (u * rX + v * uX) * fov;
          const rayY = fY + (u * rY + v * uY) * fov;
          const rayZ = fZ + (u * rZ + v * uZ) * fov;
          const invRay = invLen3(rayX, rayY, rayZ);
          let rdX = rayX * invRay;
          let rdY = rayY * invRay;
          let rdZ = rayZ * invRay;

          skyColor(rdX, rdY, rdZ, sunX, sunY, sunZ);
          const skyR = skyOut[0];
          const skyG = skyOut[1];
          const skyB = skyOut[2];

          if (rdY >= -0.02) {
            const outR = Math.pow(Math.max(0, skyR * mulVig), invGamma);
            const outG = Math.pow(Math.max(0, skyG * mulVig), invGamma);
            const outB = Math.pow(Math.max(0, skyB * mulVig), invGamma);
            data[idx] = clamp(Math.round(outR * outMul), 0, 255);
            data[idx + 1] = clamp(Math.round(outG * outMul), 0, 255);
            data[idx + 2] = clamp(Math.round(outB * outMul), 0, 255);
            data[idx + 3] = 255;
            idx += 4;
            continue;
          }

          const tPlane = (-roY) / rdY;
          let t = Math.min(tPlane, 520);
          let hitX = roX + rdX * t;
          let hitY = roY + rdY * t;
          let hitZ = roZ + rdZ * t;
          let pX = hitX;
          let pY = hitZ;

          if (t < 90 && Math.abs(rdY) > 0.06) {
            for (let i = 0; i < 2; i++) {
              seaHeightDeriv(pX, pY, timeSec);
              const diff = hitY - hdOut[0];
              t -= diff / rdY;
              t = clamp(t, 0, 520);
              hitX = roX + rdX * t;
              hitY = roY + rdY * t;
              hitZ = roZ + rdZ * t;
              pX = hitX;
              pY = hitZ;
            }
          }

          seaHeightDeriv(pX, pY, timeSec);
          const posX = hitX;
          const posY = hdOut[0];
          const posZ = hitZ;

          const nX0 = -hdOut[1];
          const nY0 = 1;
          const nZ0 = -hdOut[2];
          const invN = invLen3(nX0, nY0, nZ0);
          const nX = nX0 * invN;
          const nY = nY0 * invN;
          const nZ = nZ0 * invN;

          const vX0 = roX - posX;
          const vY0 = roY - posY;
          const vZ0 = roZ - posZ;
          const invV = invLen3(vX0, vY0, vZ0);
          const vX = vX0 * invV;
          const vY = vY0 * invV;
          const vZ = vZ0 * invV;

          const diff = saturate(nX * sunX + nY * sunY + nZ * sunZ);
          const ndv = saturate(nX * vX + nY * vY + nZ * vZ);
          const fres = Math.pow(1 - ndv, 5);
          const reflectance = 0.03 + 0.97 * fres;

          const iX = -vX;
          const iY = -vY;
          const iZ = -vZ;
          const dotNI = nX * iX + nY * iY + nZ * iZ;
          const reflX = iX - 2 * dotNI * nX;
          const reflY = iY - 2 * dotNI * nY;
          const reflZ = iZ - 2 * dotNI * nZ;
          const invRefl = invLen3(reflX, reflY, reflZ);
          rdX = reflX * invRefl;
          rdY = reflY * invRefl;
          rdZ = reflZ * invRefl;

          skyColor(rdX, rdY, rdZ, sunX, sunY, sunZ);
          const reflR = skyOut[0];
          const reflG = skyOut[1];
          const reflB = skyOut[2];

          const crest = saturate((posY / crestDen) * 0.55 + 0.5);
          let waterR = mix(deepWater[0], shallowWater[0], crest) + sunCol[0] * diff * 0.055;
          let waterG = mix(deepWater[1], shallowWater[1], crest) + sunCol[1] * diff * 0.055;
          let waterB = mix(deepWater[2], shallowWater[2], crest) + sunCol[2] * diff * 0.055;

          const hX0 = vX + sunX;
          const hY0 = vY + sunY;
          const hZ0 = vZ + sunZ;
          const invH = invLen3(hX0, hY0, hZ0);
          const hX = hX0 * invH;
          const hY = hY0 * invH;
          const hZ = hZ0 * invH;
          const spec = Math.pow(saturate(nX * hX + nY * hY + nZ * hZ), 180);

          let colR = mix(waterR, reflR, reflectance);
          let colG = mix(waterG, reflG, reflectance);
          let colB = mix(waterB, reflB, reflectance);
          const specBoost = spec * (0.35 + 0.65 * fres);
          colR += sunCol[0] * specBoost;
          colG += sunCol[1] * specBoost;
          colB += sunCol[2] * specBoost;

          const slope = saturate(1 - nY);
          let foam = smoothstep(0.62, 0.98, crest) * smoothstep(0.10, 0.45, slope);
          const noiseA = Math.sin((pX * 0.18 + pY * 0.14) * 26 + timeSec * 0.8);
          const noiseB = Math.sin((pX * -0.12 + pY * 0.20) * 31 - timeSec * 0.9);
          foam *= 0.70 + 0.30 * (noiseA * noiseB);
          foam = saturate(foam);
          colR = mix(colR, foamColor[0], foam * 0.85);
          colG = mix(colG, foamColor[1], foam * 0.85);
          colB = mix(colB, foamColor[2], foam * 0.85);

          const fog = 1 - Math.exp(-t * 0.018);
          const hazeR = mix(skyR, 0.65, 0.25);
          const hazeG = mix(skyG, 0.76, 0.25);
          const hazeB = mix(skyB, 0.86, 0.25);
          colR = mix(colR, hazeR, fog);
          colG = mix(colG, hazeG, fog);
          colB = mix(colB, hazeB, fog);

          colR = Math.pow(Math.max(0, colR * mulVig), invGamma);
          colG = Math.pow(Math.max(0, colG * mulVig), invGamma);
          colB = Math.pow(Math.max(0, colB * mulVig), invGamma);

          data[idx] = clamp(Math.round(colR * outMul), 0, 255);
          data[idx + 1] = clamp(Math.round(colG * outMul), 0, 255);
          data[idx + 2] = clamp(Math.round(colB * outMul), 0, 255);
          data[idx + 3] = 255;
          idx += 4;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    };

    return { resize, render, frameIntervalMs: 33 };
  };

  const renderer = initWebGLRenderer() || initCanvasRenderer();
  if (!renderer) {
    if (fallback && !fallback.hidden) return;
    showFallback('This browser cannot render the simulation.');
    return;
  }

  hideFallback();

  const resize = () => renderer.resize();

  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(() => resize());
    ro.observe(canvas);
  } else {
    window.addEventListener('resize', () => resize(), { passive: true });
  }

  windInput.addEventListener('input', () => {
    state.windSpeed = parseFloat(windInput.value);
    updateReadouts();
    needsRender = true;
  });
  heightInput.addEventListener('input', () => {
    state.waveHeight = parseFloat(heightInput.value);
    updateReadouts();
    needsRender = true;
  });
  sunInput.addEventListener('input', () => {
    state.sunElevation = parseFloat(sunInput.value);
    sunDir = computeSunDir(state.sunElevation);
    updateReadouts();
    needsRender = true;
  });

  toggleBtn?.addEventListener('click', () => {
    paused = !paused;
    setToggleUI();
    needsRender = true;
  });

  resetBtn?.addEventListener('click', () => {
    state.windSpeed = defaults.windSpeed;
    state.waveHeight = defaults.waveHeight;
    state.sunElevation = defaults.sunElevation;
    state.yaw = defaults.yaw;
    state.pitch = defaults.pitch;
    windInput.value = String(defaults.windSpeed);
    heightInput.value = String(defaults.waveHeight);
    sunInput.value = String(defaults.sunElevation);
    sunDir = computeSunDir(state.sunElevation);
    updateReadouts();
    needsRender = true;
  });

  document.addEventListener('visibilitychange', () => {
    pausedByVisibility = document.hidden;
    needsRender = true;
  });

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const onPointerUp = () => {
    dragging = false;
  };

  canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture?.(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    state.yaw += dx * 0.004;
    state.pitch = clamp(state.pitch + dy * 0.003, 0.05, 0.62);
    needsRender = true;
  });
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerUp);

  setToggleUI();
  updateReadouts();
  resize();

  let simTimeSec = 0;
  let lastNow = window.performance ? window.performance.now() : Date.now();
  let lastRenderMs = lastNow;
  const renderIntervalMs = renderer.frameIntervalMs || 0;

  const tick = (now) => {
    const nowMs = now || (window.performance ? window.performance.now() : Date.now());
    const dt = Math.min(0.05, Math.max(0, (nowMs - lastNow) / 1000));
    lastNow = nowMs;

    const animating = !paused && !pausedByVisibility && !document.hidden;
    if (animating) simTimeSec += dt;

    const shouldRender = needsRender
      || (animating && (!renderIntervalMs || (nowMs - lastRenderMs) >= renderIntervalMs));
    if (shouldRender) {
      renderer.render(simTimeSec);
      lastRenderMs = nowMs;
      needsRender = false;
    }

    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
})();
