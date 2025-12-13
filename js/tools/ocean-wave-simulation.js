(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const stage = $('#ocean-wave-stage');
  const canvas = $('#ocean-wave-canvas');
  const windInput = $('#ocean-wave-wind');
  const windValue = $('#ocean-wave-wind-value');
  const heightInput = $('#ocean-wave-height');
  const heightValue = $('#ocean-wave-height-value');
  const lightInput = $('#ocean-wave-light');
  const lightValue = $('#ocean-wave-light-value');
  const toggleBtn = $('#ocean-wave-toggle');
  const resetBtn = $('#ocean-wave-reset');
  const statusEl = $('#ocean-wave-status');

  if (!stage || !canvas || !windInput || !heightInput || !lightInput || !toggleBtn || !resetBtn || !statusEl) return;

  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  if (!ctx) {
    statusEl.textContent = 'Canvas unavailable in this browser.';
    return;
  }

  const TAU = Math.PI * 2;
  const G = 9.81;
  const DEG = Math.PI / 180;
  const MAX_WAVE_HEIGHT = 5.5;

  const DEFAULTS = {
    wind: 7.5,
    waveHeight: 1,
    sunElevationDeg: 38,
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const lerp = (a, b, t) => a + (b - a) * t;

  const smoothstep = (edge0, edge1, x) => {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  };

  const frac = (x) => x - Math.floor(x);

  const pseudo = (seed) => frac(Math.sin(seed * 127.1 + 311.7) * 43758.5453123);

  const BAYER4 = new Float32Array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]);

  const state = {
    wind: DEFAULTS.wind,
    waveHeight: DEFAULTS.waveHeight,
    sunElevationDeg: DEFAULTS.sunElevationDeg,
    paused: false,
  };

  const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) state.paused = true;

  let imageData = null;
  let pixels = null;
  let width = 0;
  let height = 0;
  let waterMask = null;
  let worldX = null;
  let worldZ = null;
  let viewX = null;
  let viewY = null;
  let viewZ = null;
  let distance = null;
  let rowNorm = null;
  let rowSq = null;
  let colNorm = null;
  let colSq = null;
  let skyRowR = null;
  let skyRowG = null;
  let skyRowB = null;

  const camera = {
    height: 6.2,
    yaw: 0,
    pitch: -11 * DEG,
    minPitch: -32 * DEG,
    maxPitch: 10 * DEG,
    maxDistance: 260,
    fovY: 56 * DEG,
    tanHalfFovX: 1,
    tanHalfFovY: 1,
    cosYaw: 1,
    sinYaw: 0,
    cosPitch: 1,
    sinPitch: 0,
  };

  const light = {
    azimuth: 62 * DEG,
    dirX: 0,
    dirY: 1,
    dirZ: 0,
    sunU: 0.5,
    sunV: 0.22,
    sunVisible: false,
  };

  let waves = [];
  let rafId = 0;
  let lastFrame = null;
  let simTimeSec = 0;

  const setToggleIcon = (paused) => {
    const svg = toggleBtn.querySelector('svg');
    if (!svg) return;
    svg.innerHTML = paused
      ? '<path d="M9 6v12l10-6-10-6Z"></path>'
      : '<path d="M8 6v12M16 6v12"></path>';
  };

  const setStatus = (text) => {
    statusEl.textContent = text;
  };

  const fmt = (n, digits = 1) => Number(n).toFixed(digits);

  const syncUI = () => {
    windValue.textContent = `${fmt(state.wind, 1)} m/s`;
    heightValue.textContent = `${fmt(state.waveHeight, 2)}×`;
    lightValue.textContent = `${Math.round(state.sunElevationDeg)}°`;
    toggleBtn.setAttribute('aria-pressed', state.paused ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', state.paused ? 'Play animation' : 'Pause animation');
    setToggleIcon(state.paused);
    setStatus(state.paused ? 'Paused' : 'Running');
  };

  const updateLight = () => {
    const elev = clamp(state.sunElevationDeg, 5, 85) * DEG;
    const cosElev = Math.cos(elev);
    light.dirX = Math.cos(light.azimuth) * cosElev;
    light.dirY = Math.sin(elev);
    light.dirZ = Math.sin(light.azimuth) * cosElev;

    const xYaw = light.dirX * camera.cosYaw - light.dirZ * camera.sinYaw;
    const zYaw = light.dirX * camera.sinYaw + light.dirZ * camera.cosYaw;

    const camY = light.dirY * camera.cosPitch + zYaw * camera.sinPitch;
    const camZ = -light.dirY * camera.sinPitch + zYaw * camera.cosPitch;
    const camX = xYaw;

    if (!Number.isFinite(camX) || !Number.isFinite(camY) || !Number.isFinite(camZ) || camZ <= 0.02) {
      light.sunVisible = false;
      return;
    }

    const ndcX = (camX / camZ) / camera.tanHalfFovX;
    const ndcY = (camY / camZ) / camera.tanHalfFovY;
    light.sunU = ndcX * 0.5 + 0.5;
    light.sunV = -ndcY * 0.5 + 0.5;
    light.sunVisible = light.sunU > -0.2 && light.sunU < 1.2 && light.sunV > -0.2 && light.sunV < 1.2;
  };

  const wrapAngle = (rad) => {
    const wrapped = (rad + Math.PI) % TAU;
    return (wrapped < 0 ? wrapped + TAU : wrapped) - Math.PI;
  };

  const rebuildCameraRays = () => {
    if (!width || !height || !waterMask) return;

    camera.tanHalfFovY = Math.tan(camera.fovY / 2);
    camera.tanHalfFovX = camera.tanHalfFovY * (width / height);
    camera.cosPitch = Math.cos(camera.pitch);
    camera.sinPitch = Math.sin(camera.pitch);
    camera.cosYaw = Math.cos(camera.yaw);
    camera.sinYaw = Math.sin(camera.yaw);

    const horizonCut = camera.maxDistance;
    const camY = camera.height;
    const eps = 1e-4;

    let p = 0;
    for (let y = 0; y < height; y++) {
      const ndcY = 1 - (2 * (y + 0.5)) / height;
      const ry = ndcY * camera.tanHalfFovY;
      for (let x = 0; x < width; x++, p++) {
        const ndcX = (2 * (x + 0.5)) / width - 1;
        const rx = ndcX * camera.tanHalfFovX;
        const rz = 1;
        const len = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
        const dx = rx / len;
        const dy = ry / len;
        const dz = rz / len;

        const dyP = dy * camera.cosPitch - dz * camera.sinPitch;
        const dzP = dy * camera.sinPitch + dz * camera.cosPitch;
        const dxP = dx;

        const dxW = dxP * camera.cosYaw + dzP * camera.sinYaw;
        const dzW = -dxP * camera.sinYaw + dzP * camera.cosYaw;
        const dyW = dyP;

        if (dyW >= -eps) {
          waterMask[p] = 0;
          continue;
        }

        const t = camY / -dyW;
        if (t >= horizonCut) {
          waterMask[p] = 0;
          continue;
        }

        const wx = dxW * t;
        const wz = dzW * t;
        worldX[p] = wx;
        worldZ[p] = wz;
        distance[p] = t;
        waterMask[p] = 1;

        let vx = -wx;
        let vy = camY;
        let vz = -wz;
        const vLen = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
        vx /= vLen;
        vy /= vLen;
        vz /= vLen;
        viewX[p] = vx;
        viewY[p] = vy;
        viewZ[p] = vz;
      }
    }

    updateLight();
  };

  const buildWaves = () => {
    const sea = clamp(state.wind / 20, 0, 1);
    const lengthScale = lerp(0.85, 1.55, sea);
    const speedScale = lerp(0.72, 1.65, sea);
    const ampScale = lerp(0.06, 0.92, Math.pow(sea, 0.9));
    const spread = lerp(1.55, 0.65, sea);
    const windDir = 86 * DEG;
    const baseChop = lerp(0.05, 0.62, Math.pow(sea, 0.95));

    const components = [
      { length: 92, weight: 1.0, offset: 0.0 },
      { length: 58, weight: 0.72, offset: 0.52 },
      { length: 36, weight: 0.52, offset: -0.66 },
      { length: 22, weight: 0.36, offset: 1.18 },
      { length: 14, weight: 0.26, offset: -1.28 },
      { length: 9.2, weight: 0.19, offset: 0.28 },
    ];

    waves = components.map((c, idx) => {
      const shortness = components.length === 1 ? 0 : idx / (components.length - 1);
      const jitter = (pseudo(idx + 11.2) - 0.5) * 0.18;
      const theta = windDir + c.offset * spread;
      const dirX = Math.cos(theta + jitter);
      const dirZ = Math.sin(theta + jitter);
      const lambda = c.length * lengthScale;
      const k = TAU / lambda;
      const omega = Math.sqrt(G * k) * speedScale;
      const phase = pseudo(idx + 1) * TAU;
      const amp = ampScale * c.weight;
      const chop = baseChop * lerp(0.18, 1.0, shortness);
      return {
        kx: k * dirX,
        kz: k * dirZ,
        omega,
        phase,
        amp,
        chop,
        shortness,
      };
    });
  };

  const resize = () => {
    const rect = stage.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const cores = typeof navigator !== 'undefined' && Number.isFinite(navigator.hardwareConcurrency)
      ? navigator.hardwareConcurrency
      : 4;
    const quality = cores >= 8
      ? { dprMax: 2.25, maxPixels: 210000, scale: 0.9, maxW: 920, maxH: 700 }
      : cores >= 6
        ? { dprMax: 2.1, maxPixels: 185000, scale: 0.84, maxW: 860, maxH: 660 }
        : { dprMax: 2, maxPixels: 160000, scale: 0.78, maxW: 820, maxH: 620 };

    const dpr = Math.min(quality.dprMax, window.devicePixelRatio || 1);
    const targetW = rect.width * dpr;
    const targetH = rect.height * dpr;

    let nextW = Math.round(clamp(targetW * quality.scale, 320, quality.maxW));
    let nextH = Math.round(nextW * (targetH / targetW));
    nextH = Math.round(clamp(nextH, 220, quality.maxH));

    const pix = nextW * nextH;
    if (pix > quality.maxPixels) {
      const scale = Math.sqrt(quality.maxPixels / pix);
      nextW = Math.max(240, Math.round(nextW * scale));
      nextH = Math.max(180, Math.round(nextH * scale));
    }

    if (nextW === width && nextH === height) return;

    width = nextW;
    height = nextH;
    canvas.width = width;
    canvas.height = height;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    imageData = ctx.createImageData(width, height);
    pixels = imageData.data;

    const count = width * height;
    waterMask = new Uint8Array(count);
    worldX = new Float32Array(count);
    worldZ = new Float32Array(count);
    viewX = new Float32Array(count);
    viewY = new Float32Array(count);
    viewZ = new Float32Array(count);
    distance = new Float32Array(count);

    rowNorm = new Float32Array(height);
    rowSq = new Float32Array(height);
    for (let y = 0; y < height; y++) {
      const t = height === 1 ? 0 : y / (height - 1);
      rowNorm[y] = t;
      const v = (t - 0.5) * 2;
      rowSq[y] = v * v;
    }

    colNorm = new Float32Array(width);
    colSq = new Float32Array(width);
    for (let x = 0; x < width; x++) {
      const t = width === 1 ? 0 : x / (width - 1);
      colNorm[x] = t;
      const u = (t - 0.5) * 2;
      colSq[x] = u * u;
    }

    skyRowR = new Float32Array(height);
    skyRowG = new Float32Array(height);
    skyRowB = new Float32Array(height);
    rebuildCameraRays();
    renderFrame(simTimeSec);
  };

  let cameraDirty = false;
  let draggingPointerId = null;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartYaw = 0;
  let dragStartPitch = 0;

  const markCameraDirty = () => {
    cameraDirty = true;
    if (state.paused) {
      rebuildCameraRays();
      renderFrame(simTimeSec);
    }
  };

  stage.addEventListener('pointerdown', (event) => {
    if (event.target && event.target.closest && event.target.closest('button')) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    draggingPointerId = event.pointerId;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragStartYaw = camera.yaw;
    dragStartPitch = camera.pitch;
    stage.classList.add('ocean-wave-dragging');
    stage.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  stage.addEventListener('pointermove', (event) => {
    if (draggingPointerId === null || event.pointerId !== draggingPointerId) return;
    const dx = event.clientX - dragStartX;
    const dy = event.clientY - dragStartY;
    const sensitivity = 0.0055;
    camera.yaw = wrapAngle(dragStartYaw + dx * sensitivity);
    camera.pitch = clamp(dragStartPitch + dy * sensitivity, camera.minPitch, camera.maxPitch);
    markCameraDirty();
    event.preventDefault();
  });

  const endDrag = (event) => {
    if (draggingPointerId === null || event.pointerId !== draggingPointerId) return;
    draggingPointerId = null;
    stage.classList.remove('ocean-wave-dragging');
    if (stage.hasPointerCapture && stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  };

  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);
  stage.addEventListener('lostpointercapture', () => {
    draggingPointerId = null;
    stage.classList.remove('ocean-wave-dragging');
  });

  const renderFrame = (t) => {
    if (!imageData || !pixels) return;

    const elevNorm = clamp((state.sunElevationDeg - 5) / 70, 0, 1);
    const warmth = Math.pow(1 - elevNorm, 1.35);

    const skyTopR = lerp(10, 22, elevNorm);
    const skyTopG = lerp(22, 74, elevNorm);
    const skyTopB = lerp(56, 124, elevNorm);

    let skyHorizonR = lerp(64, 138, elevNorm);
    let skyHorizonG = lerp(84, 182, elevNorm);
    let skyHorizonB = lerp(104, 206, elevNorm);

    skyHorizonR = lerp(skyHorizonR, 196, warmth);
    skyHorizonG = lerp(skyHorizonG, 132, warmth);
    skyHorizonB = lerp(skyHorizonB, 102, warmth);

    const sunR = lerp(252, 255, 0.4) * (1 - warmth) + lerp(255, 255, 0.4) * warmth;
    const sunG = lerp(252, 220, warmth);
    const sunB = lerp(255, 186, warmth);

    for (let y = 0; y < height; y++) {
      const s = smoothstep(0.0, 0.78, rowNorm[y]);
      skyRowR[y] = lerp(skyTopR, skyHorizonR, s);
      skyRowG[y] = lerp(skyTopG, skyHorizonG, s);
      skyRowB[y] = lerp(skyTopB, skyHorizonB, s);
    }

    const maxD = camera.maxDistance;
    const lx = light.dirX;
    const ly = light.dirY;
    const lz = light.dirZ;

    const baseWaterNearR = 6;
    const baseWaterNearG = 30;
    const baseWaterNearB = 42;
    const baseWaterFarR = 14;
    const baseWaterFarG = 86;
    const baseWaterFarB = 108;

    const foamR = 204;
    const foamG = 232;
    const foamB = 236;

    const sunSpecStrength = lerp(0.66, 1.02, 1 - elevNorm);
    const shininess = lerp(128, 186, 1 - elevNorm);
    const heightScale = clamp(state.waveHeight, 0, MAX_WAVE_HEIGHT);
    const heightNorm = MAX_WAVE_HEIGHT ? heightScale / MAX_WAVE_HEIGHT : 0;
    const ampScale = heightScale;
    const chopBoost = lerp(0.8, 1.25, heightNorm);

    const vignetteStrength = 0.14;
    const glowStrength = 0.9;
    const discR2 = 0.0007;
    const glowR2 = 0.035;

    let o = 0;
    let p = 0;
    for (let y = 0; y < height; y++) {
      const skyR = skyRowR[y];
      const skyG = skyRowG[y];
      const skyB = skyRowB[y];
      const v2 = rowSq[y];
      const yv = rowNorm[y];
      for (let x = 0; x < width; x++, p++, o += 4) {
        const u = colNorm[x];
        const u2 = colSq[x];
        const vignette = clamp(1 - (u2 + v2) * vignetteStrength, 0.72, 1);
        const dither = (BAYER4[(x & 3) + ((y & 3) << 2)] / 16 - 0.5) * 1.4;

        if (!waterMask[p]) {
          let r = skyR;
          let g = skyG;
          let b = skyB;

          if (light.sunVisible) {
            const dx = u - light.sunU;
            const dy = yv - light.sunV;
            const r2 = dx * dx + dy * dy;
            const disc = Math.max(0, 1 - r2 / discR2);
            const glow = Math.max(0, 1 - r2 / glowR2);
            const sun = disc * disc * disc * 1.15 + glow * glow * 0.32;
            r += sun * sunR * glowStrength;
            g += sun * sunG * glowStrength;
            b += sun * sunB * glowStrength;
          }

          pixels[o] = clamp(r * vignette + dither, 0, 255);
          pixels[o + 1] = clamp(g * vignette + dither, 0, 255);
          pixels[o + 2] = clamp(b * vignette + dither, 0, 255);
          pixels[o + 3] = 255;
          continue;
        }

        const wx = worldX[p];
        const wz = worldZ[p];
        let dhdx = 0;
        let dhdz = 0;
        let h = 0;

        for (let i = 0; i < waves.length; i++) {
          const wave = waves[i];
          const shortTaper = 1 - wave.shortness * heightNorm * 0.42;
          const amp = wave.amp * ampScale * Math.max(0.35, shortTaper);
          const phase = wx * wave.kx + wz * wave.kz + t * wave.omega + wave.phase;
          const s1 = Math.sin(phase);
          const c1 = Math.cos(phase);
          const chop = wave.chop * chopBoost;
          const s2 = 2 * s1 * c1;
          const c2 = c1 * c1 - s1 * s1;
          const s = s1 + chop * 0.35 * s2;
          const dSdP = c1 + chop * 0.70 * c2;
          h += amp * s;
          dhdx += amp * dSdP * wave.kx;
          dhdz += amp * dSdP * wave.kz;
        }

        let nx = -dhdx;
        let ny = 1;
        let nz = -dhdz;
        const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;

        const vx = viewX[p];
        const vy = viewY[p];
        const vz = viewZ[p];
        const ndv = nx * vx + ny * vy + nz * vz;
        const ndotv = clamp(ndv, 0, 1);
        const ndotl = clamp(nx * lx + ny * ly + nz * lz, 0, 1);

        const f0 = 0.02;
        const fresnel = f0 + (1 - f0) * Math.pow(1 - ndotv, 5);
        const ambient = lerp(0.12, 0.20, elevNorm);
        const diffuse = ambient + ndotl * lerp(0.55, 0.82, elevNorm);

        const reflY = -vy + 2 * ndv * ny;
        const reflUp = clamp(reflY, 0, 1);
        const reflMix = smoothstep(0.0, 0.78, 1 - reflUp);
        const reflSkyR = lerp(skyTopR, skyHorizonR, reflMix);
        const reflSkyG = lerp(skyTopG, skyHorizonG, reflMix);
        const reflSkyB = lerp(skyTopB, skyHorizonB, reflMix);

        const inX = -lx;
        const inY = -ly;
        const inZ = -lz;
        const inDotN = inX * nx + inY * ny + inZ * nz;
        const rx = inX - 2 * inDotN * nx;
        const ry = inY - 2 * inDotN * ny;
        const rz = inZ - 2 * inDotN * nz;
        const rdotv = clamp(rx * vx + ry * vy + rz * vz, 0, 1);
        const rough = clamp((dhdx * dhdx + dhdz * dhdz) * 0.12, 0, 1);
        const spec = Math.pow(rdotv, shininess * lerp(1, 0.55, rough)) * sunSpecStrength;

        const d = clamp(distance[p] / maxD, 0, 1);
        const depthMix = smoothstep(0.05, 0.92, d);
        let baseR = lerp(baseWaterNearR, baseWaterFarR, depthMix);
        let baseG = lerp(baseWaterNearG, baseWaterFarG, depthMix);
        let baseB = lerp(baseWaterNearB, baseWaterFarB, depthMix);

        const tint = clamp(h * 0.06 + 0.5, 0, 1);
        baseR = lerp(baseR * 0.92, baseR * 1.06, tint);
        baseG = lerp(baseG * 0.92, baseG * 1.06, tint);
        baseB = lerp(baseB * 0.92, baseB * 1.06, tint);

        let r = lerp(baseR, reflSkyR, fresnel);
        let g = lerp(baseG, reflSkyG, fresnel);
        let b = lerp(baseB, reflSkyB, fresnel);

        r = r * diffuse + spec * sunR;
        g = g * diffuse + spec * sunG;
        b = b * diffuse + spec * sunB;

        const haze = smoothstep(0.28, 1.0, d);
        r = lerp(r, skyR, haze * 0.82);
        g = lerp(g, skyG, haze * 0.82);
        b = lerp(b, skyB, haze * 0.86);

        const slope2 = dhdx * dhdx + dhdz * dhdz;
        const foamStrength = lerp(0.16, 0.62, clamp(heightNorm, 0, 1));
        const foam = smoothstep(0.22, lerp(1.55, 1.02, elevNorm), slope2) * foamStrength;
        if (foam > 0.001) {
          r = lerp(r, foamR, foam);
          g = lerp(g, foamG, foam);
          b = lerp(b, foamB, foam);
        }

        pixels[o] = clamp(r * vignette + dither, 0, 255);
        pixels[o + 1] = clamp(g * vignette + dither, 0, 255);
        pixels[o + 2] = clamp(b * vignette + dither, 0, 255);
        pixels[o + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const start = () => {
    if (rafId) return;
    lastFrame = null;
    rafId = window.requestAnimationFrame(tick);
  };

  const stop = () => {
    if (!rafId) return;
    window.cancelAnimationFrame(rafId);
    rafId = 0;
    lastFrame = null;
  };

  const tick = (ts) => {
    if (!lastFrame) lastFrame = ts;
    const dt = clamp((ts - lastFrame) / 1000, 0, 0.05);
    lastFrame = ts;
    simTimeSec += dt;
    if (cameraDirty) {
      cameraDirty = false;
      rebuildCameraRays();
    }
    renderFrame(simTimeSec);
    rafId = window.requestAnimationFrame(tick);
  };

  const setPaused = (paused) => {
    state.paused = !!paused;
    syncUI();
    if (state.paused) stop();
    else start();
    renderFrame(simTimeSec);
  };

  const syncFromInputs = () => {
    state.wind = clamp(parseFloat(windInput.value || DEFAULTS.wind), 0, 20);
    state.waveHeight = clamp(parseFloat(heightInput.value || DEFAULTS.waveHeight), 0, MAX_WAVE_HEIGHT);
    state.sunElevationDeg = clamp(parseFloat(lightInput.value || DEFAULTS.sunElevationDeg), 5, 75);
    buildWaves();
    updateLight();
    syncUI();
  };

  const debounce = (fn, ms) => {
    let t = 0;
    return (...args) => {
      window.clearTimeout(t);
      t = window.setTimeout(() => fn(...args), ms);
    };
  };

  windInput.addEventListener('input', () => {
    state.wind = clamp(parseFloat(windInput.value), 0, 20);
    buildWaves();
    syncUI();
    renderFrame(simTimeSec);
  });

  heightInput.addEventListener('input', () => {
    state.waveHeight = clamp(parseFloat(heightInput.value), 0, MAX_WAVE_HEIGHT);
    syncUI();
    renderFrame(simTimeSec);
  });

  lightInput.addEventListener('input', () => {
    state.sunElevationDeg = clamp(parseFloat(lightInput.value), 5, 75);
    updateLight();
    syncUI();
    renderFrame(simTimeSec);
  });

  toggleBtn.addEventListener('click', () => {
    setPaused(!state.paused);
  });

  resetBtn.addEventListener('click', () => {
    windInput.value = DEFAULTS.wind;
    heightInput.value = DEFAULTS.waveHeight;
    lightInput.value = DEFAULTS.sunElevationDeg;
    camera.yaw = 0;
    camera.pitch = -11 * DEG;
    camera.pitch = clamp(camera.pitch, camera.minPitch, camera.maxPitch);
    syncFromInputs();
    cameraDirty = false;
    rebuildCameraRays();
    renderFrame(simTimeSec);
  });

  const onResize = debounce(() => {
    resize();
  }, 160);
  window.addEventListener('resize', onResize);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else if (!state.paused) start();
  });

  syncFromInputs();
  resize();

  if (prefersReducedMotion) {
    setStatus('Paused (reduce motion)');
    setToggleIcon(true);
    toggleBtn.setAttribute('aria-pressed', 'true');
  }

  if (!state.paused) start();
})();
