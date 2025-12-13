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

  const gl = canvas.getContext('webgl', {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance'
  });

  const showFallback = (message) => {
    if (!fallback) return;
    fallback.hidden = false;
    if (message) fallback.textContent = message;
  };

  if (!gl) {
    showFallback('WebGL is unavailable in this browser, so the simulation can’t run here.');
    return;
  }

  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
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

  const vs = compileShader(gl.VERTEX_SHADER, VERT);
  if (vs.error) {
    showFallback(`Vertex shader error: ${vs.error}`);
    return;
  }
  const fs = compileShader(gl.FRAGMENT_SHADER, FRAG);
  if (fs.error) {
    showFallback(`Fragment shader error: ${fs.error}`);
    return;
  }
  const program = linkProgram(vs, fs);
  if (program.error) {
    showFallback(`WebGL program error: ${program.error}`);
    return;
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

  const setUniforms = (timeSec) => {
    gl.uniform1f(uniforms.time, timeSec);
    gl.uniform1f(uniforms.windSpeed, state.windSpeed);
    gl.uniform1f(uniforms.waveHeight, state.waveHeight);
    gl.uniform3f(uniforms.sunDir, sunDir[0], sunDir[1], sunDir[2]);
    gl.uniform2f(uniforms.view, state.yaw, state.pitch);
  };

  const render = (timeSec) => {
    setUniforms(timeSec);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

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

  const tick = (now) => {
    const nowMs = now || (window.performance ? window.performance.now() : Date.now());
    const dt = Math.min(0.05, Math.max(0, (nowMs - lastNow) / 1000));
    lastNow = nowMs;

    if (!paused && !pausedByVisibility && !document.hidden) {
      simTimeSec += dt;
      render(simTimeSec);
      needsRender = false;
    } else if (needsRender) {
      render(simTimeSec);
      needsRender = false;
    }

    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
})();

