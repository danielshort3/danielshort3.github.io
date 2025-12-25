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
  const WIND_DIR = 86 * DEG;

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
  const NOISE_SIZE = 128;
  const NOISE_MASK = NOISE_SIZE - 1;
  const noiseTile = new Float32Array(NOISE_SIZE * NOISE_SIZE);

  const buildNoise = () => {
    let i = 0;
    for (let y = 0; y < NOISE_SIZE; y++) {
      for (let x = 0; x < NOISE_SIZE; x++, i++) {
        const v1 = pseudo(x * 12.9898 + y * 78.233);
        const v2 = pseudo(x * 39.3468 + y * 11.135 + 3.1);
        noiseTile[i] = (v1 + v2 * 0.5) / 1.5;
      }
    }
  };

  const sampleNoise = (x, y) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const tx = x - xi;
    const ty = y - yi;
    const x0 = xi & NOISE_MASK;
    const y0 = yi & NOISE_MASK;
    const x1 = (x0 + 1) & NOISE_MASK;
    const y1 = (y0 + 1) & NOISE_MASK;
    const i00 = x0 + y0 * NOISE_SIZE;
    const i10 = x1 + y0 * NOISE_SIZE;
    const i01 = x0 + y1 * NOISE_SIZE;
    const i11 = x1 + y1 * NOISE_SIZE;
    const v00 = noiseTile[i00];
    const v10 = noiseTile[i10];
    const v01 = noiseTile[i01];
    const v11 = noiseTile[i11];
    const sx = tx * tx * (3 - 2 * tx);
    const sy = ty * ty * (3 - 2 * ty);
    const ax = lerp(v00, v10, sx);
    const bx = lerp(v01, v11, sx);
    return lerp(ax, bx, sy);
  };

  const fbmNoise = (x, y) => {
    let sum = 0;
    let amp = 0.65;
    let freq = 1;
    for (let i = 0; i < 3; i++) {
      sum += amp * sampleNoise(x * freq, y * freq);
      amp *= 0.5;
      freq *= 2.07;
    }
    return sum;
  };

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
    const energy = lerp(0.2, 0.95, Math.pow(sea, 0.9));
    const speedScale = lerp(0.76, 1.75, sea);
    const windSpread = lerp(0.18, 0.85, sea);
    const swellSpread = lerp(0.05, 0.14, sea);
    const baseChop = lerp(0.06, 1.05, Math.pow(sea, 0.92));
    const peakLength = lerp(70, 180, Math.pow(sea, 1.1));
    const maxLength = peakLength * 2.6;
    const minLength = peakLength * 0.12;
    const swellDir = WIND_DIR - lerp(0.55, 0.25, sea);

    const shortnessOf = (length) => {
      const span = Math.log(maxLength / minLength) || 1;
      return clamp(Math.log(maxLength / length) / span, 0, 1);
    };

    const spectralWeight = (length) => {
      const ratio = length / peakLength;
      const spread = lerp(0.95, 0.65, sea);
      return Math.exp(-Math.pow(Math.log(ratio) / spread, 2));
    };

    waves = [];

    const pushWave = (length, weight, direction, speedMul, chop, seed) => {
      const k = TAU / length;
      const omega = Math.sqrt(G * k) * speedScale * speedMul;
      const dirX = Math.cos(direction);
      const dirZ = Math.sin(direction);
      waves.push({
        kx: k * dirX,
        kz: k * dirZ,
        omega,
        phase: pseudo(seed) * TAU,
        amp: weight,
        chop,
        shortness: shortnessOf(length)
      });
    };

    const swellCount = 3;
    for (let i = 0; i < swellCount; i++) {
      const t = swellCount === 1 ? 0 : i / (swellCount - 1);
      const length = peakLength * lerp(2.6, 1.5, t) * lerp(1.05, 0.95, sea);
      const dirJitter = (pseudo(i + 2.1) - 0.5) * swellSpread;
      const weight = energy * lerp(0.34, 0.2, t) * lerp(0.95, 0.7, sea);
      pushWave(length, weight, swellDir + dirJitter, 0.82, baseChop * 0.2, i + 10.7);
    }

    const windCount = 6;
    for (let i = 0; i < windCount; i++) {
      const t = windCount === 1 ? 0 : i / (windCount - 1);
      const length = peakLength * lerp(1.2, 0.28, t);
      const dirJitter = (pseudo(i + 19.8) - 0.5) * windSpread * lerp(0.6, 1.3, t);
      const spectral = spectralWeight(length);
      const weight = energy * (0.12 + 0.32 * spectral) * lerp(0.85, 1.1, sea);
      const chop = baseChop * lerp(0.35, 1.05, t);
      const speedMul = lerp(0.9, 1.2, t);
      pushWave(length, weight, WIND_DIR + dirJitter, speedMul, chop, i + 29.3);
    }

    const chopCount = 3;
    for (let i = 0; i < chopCount; i++) {
      const t = chopCount === 1 ? 0 : i / (chopCount - 1);
      const length = peakLength * lerp(0.26, 0.12, t);
      const dirJitter = (pseudo(i + 38.6) - 0.5) * windSpread * 1.9;
      const spectral = spectralWeight(length);
      const weight = energy * (0.08 + 0.12 * spectral) * lerp(0.9, 1.1, sea);
      const chop = baseChop * lerp(1.05, 1.35, t);
      const speedMul = lerp(1.05, 1.28, t);
      pushWave(length, weight, WIND_DIR + dirJitter, speedMul, chop, i + 40.2);
    }
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

    const sea = clamp(state.wind / 20, 0, 1);
    const heightScale = clamp(state.waveHeight, 0, MAX_WAVE_HEIGHT);
    const heightNorm = MAX_WAVE_HEIGHT ? heightScale / MAX_WAVE_HEIGHT : 0;
    const elevNorm = clamp((state.sunElevationDeg - 5) / 70, 0, 1);
    const warmth = Math.pow(1 - elevNorm, 1.35);
    const sunIntensity = lerp(0.88, 1.3, 1 - elevNorm) * lerp(0.95, 1.08, sea);

    const skyTopR = lerp(6, 22, elevNorm);
    const skyTopG = lerp(14, 70, elevNorm);
    const skyTopB = lerp(48, 140, elevNorm);

    let skyHorizonR = lerp(70, 174, elevNorm);
    let skyHorizonG = lerp(94, 208, elevNorm);
    let skyHorizonB = lerp(120, 232, elevNorm);
    const hazeBlend = clamp(warmth * (0.85 + sea * 0.2), 0, 1);

    skyHorizonR = lerp(skyHorizonR, 222, hazeBlend);
    skyHorizonG = lerp(skyHorizonG, 168, hazeBlend * 0.9);
    skyHorizonB = lerp(skyHorizonB, 124, hazeBlend * 0.7);

    const sunR = lerp(250, 255, 0.5) * (1 - warmth * 0.35);
    const sunG = lerp(244, 214, warmth);
    const sunB = lerp(232, 182, warmth);

    const cloudScale = lerp(2.0, 3.8, elevNorm) * lerp(0.95, 1.1, sea);
    const cloudStrength = lerp(0.14, 0.06, elevNorm) * lerp(0.95, 1.2, sea);
    const cloudDrift = t * lerp(0.006, 0.022, sea);

    for (let y = 0; y < height; y++) {
      const s = smoothstep(0.0, 0.8, rowNorm[y]);
      skyRowR[y] = lerp(skyTopR, skyHorizonR, s);
      skyRowG[y] = lerp(skyTopG, skyHorizonG, s);
      skyRowB[y] = lerp(skyTopB, skyHorizonB, s);
    }

    const maxD = camera.maxDistance;
    const lx = light.dirX;
    const ly = light.dirY;
    const lz = light.dirZ;
    const windX = Math.cos(WIND_DIR);
    const windZ = Math.sin(WIND_DIR);

    const baseWaterNearR = lerp(4, 8, sea);
    const baseWaterNearG = lerp(30, 38, sea);
    const baseWaterNearB = lerp(58, 72, sea);
    const baseWaterFarR = lerp(6, 12, sea);
    const baseWaterFarG = lerp(64, 94, sea);
    const baseWaterFarB = lerp(96, 126, sea);
    const scatterR = lerp(22, 34, elevNorm);
    const scatterG = lerp(80, 116, elevNorm);
    const scatterB = lerp(110, 146, elevNorm);

    const foamR = 218;
    const foamG = 244;
    const foamB = 246;

    const ampScale = heightScale;
    const chopBoost = lerp(0.78, 1.35, heightNorm);
    const rippleScale = lerp(0.55, 1.05, sea);
    const rippleAmp = lerp(0.04, 0.16, sea) * (0.35 + 0.65 * heightNorm);
    const rippleDrift = t * lerp(0.06, 0.18, sea);
    const foamScale = lerp(0.03, 0.075, sea) * lerp(0.9, 1.08, heightNorm);
    const foamDrift = t * lerp(0.04, 0.12, sea);

    const vignetteStrength = 0.12;
    const glowStrength = 1.05;
    const discR2 = 0.00055;
    const glowR2 = 0.028;

    let o = 0;
    let p = 0;
    for (let y = 0; y < height; y++) {
      const skyR = skyRowR[y];
      const skyG = skyRowG[y];
      const skyB = skyRowB[y];
      const v2 = rowSq[y];
      const yv = rowNorm[y];
      const cloudFade = smoothstep(0.05, 0.75, 1 - yv);
      for (let x = 0; x < width; x++, p++, o += 4) {
        const u = colNorm[x];
        const u2 = colSq[x];
        const vignette = clamp(1 - (u2 + v2) * vignetteStrength, 0.76, 1);
        const dither = (BAYER4[(x & 3) + ((y & 3) << 2)] / 16 - 0.5) * 1.2;

        if (!waterMask[p]) {
          let r = skyR;
          let g = skyG;
          let b = skyB;

          if (cloudStrength > 0.001) {
            const cloudU = u * cloudScale + cloudDrift;
            const cloudV = yv * cloudScale * 0.7 + cloudDrift * 0.6;
            const cloudNoise = fbmNoise(cloudU, cloudV);
            const cloudMask = smoothstep(0.52, 0.82, cloudNoise) * cloudFade;
            const cloud = cloudMask * cloudStrength;
            r = lerp(r, r * 0.86 + 16, cloud);
            g = lerp(g, g * 0.88 + 18, cloud);
            b = lerp(b, b * 0.9 + 22, cloud);
          }

          if (light.sunVisible) {
            const dx = u - light.sunU;
            const dy = yv - light.sunV;
            const r2 = dx * dx + dy * dy;
            const disc = Math.max(0, 1 - r2 / discR2);
            const glow = Math.max(0, 1 - r2 / glowR2);
            const sun = disc * disc * disc * 1.2 + glow * glow * 0.34;
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
        let crest = -1;

        for (let i = 0; i < waves.length; i++) {
          const wave = waves[i];
          const shortTaper = 1 - wave.shortness * heightNorm * 0.45;
          const amp = wave.amp * ampScale * Math.max(0.3, shortTaper);
          const phase = wx * wave.kx + wz * wave.kz + t * wave.omega + wave.phase;
          const s1 = Math.sin(phase);
          const c1 = Math.cos(phase);
          const chop = wave.chop * chopBoost;
          const s2 = 2 * s1 * c1;
          const c2 = c1 * c1 - s1 * s1;
          const s3 = s2 * c1 + c2 * s1;
          const c3 = c2 * c1 - s2 * s1;
          const s4 = 2 * s2 * c2;
          const c4 = c2 * c2 - s2 * s2;
          const s = s1 + chop * (0.32 * s2 + 0.15 * s3 + 0.07 * s4);
          const dSdP = c1 + chop * (0.64 * c2 + 0.45 * c3 + 0.28 * c4);
          h += amp * s;
          dhdx += amp * dSdP * wave.kx;
          dhdz += amp * dSdP * wave.kz;
          const crestCandidate = s1 + chop * (0.28 * s2 + 0.1 * s3);
          if (crestCandidate > crest) crest = crestCandidate;
        }

        const flowX = wx * windX + wz * windZ;
        const flowZ = -wx * windZ + wz * windX;
        const rippleU = flowX * rippleScale + rippleDrift;
        const rippleV = flowZ * rippleScale - rippleDrift * 0.6;
        const rippleBase = sampleNoise(rippleU, rippleV);
        const rippleDx = sampleNoise(rippleU + 0.8, rippleV);
        const rippleDz = sampleNoise(rippleU, rippleV + 0.8);
        const rippleHigh = sampleNoise(rippleU * 2.2 + 1.7, rippleV * 2.2 - 0.6);
        const rippleMix = rippleBase * 0.72 + rippleHigh * 0.28;
        const microSlopeX = (rippleDx - rippleBase) * 1.45 + (rippleHigh - rippleBase) * 0.6;
        const microSlopeZ = (rippleDz - rippleBase) * 1.45 + (rippleHigh - rippleBase) * 0.6;
        dhdx += microSlopeX * rippleAmp;
        dhdz += microSlopeZ * rippleAmp;

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
        const ndotv = clamp(ndv, 0.001, 1);
        const ndotl = clamp(nx * lx + ny * ly + nz * lz, 0, 1);

        const fresnel = 0.02 + (1 - 0.02) * Math.pow(1 - ndotv, 5);

        const reflX = 2 * ndv * nx - vx;
        const reflY = 2 * ndv * ny - vy;
        const reflZ = 2 * ndv * nz - vz;
        const reflUp = clamp(reflY, 0, 1);
        const reflMix = smoothstep(0.0, 0.85, 1 - reflUp);
        let reflSkyR = lerp(skyTopR, skyHorizonR, reflMix);
        let reflSkyG = lerp(skyTopG, skyHorizonG, reflMix);
        let reflSkyB = lerp(skyTopB, skyHorizonB, reflMix);

        const sunAlign = clamp(reflX * lx + reflY * ly + reflZ * lz, 0, 1);
        const slope2 = dhdx * dhdx + dhdz * dhdz;
        const foamHint = smoothstep(0.2, 0.9, slope2 * (0.6 + sea))
          * smoothstep(0.0, 0.75, h / (ampScale + 0.6));
        const roughBase = lerp(0.055, 0.24, sea);
        const rough = clamp(
          roughBase + slope2 * 0.14 + rippleAmp * (0.55 + 0.45 * rippleMix) + foamHint * 0.32,
          0.035,
          0.7
        );

        const sunFocus = Math.pow(sunAlign, lerp(50, 230, 1 - rough));
        const sunBoost = sunFocus * sunIntensity;
        reflSkyR += sunBoost * sunR;
        reflSkyG += sunBoost * sunG;
        reflSkyB += sunBoost * sunB;

        const hx = lx + vx;
        const hy = ly + vy;
        const hz = lz + vz;
        const hLen = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
        const hxN = hx / hLen;
        const hyN = hy / hLen;
        const hzN = hz / hLen;
        const ndoth = clamp(nx * hxN + ny * hyN + nz * hzN, 0, 1);
        const vdoth = clamp(vx * hxN + vy * hyN + vz * hzN, 0, 1);

        const alpha = rough * rough;
        const alpha2 = alpha * alpha;
        const denom = ndoth * ndoth * (alpha2 - 1) + 1;
        const D = alpha2 / (Math.PI * denom * denom);
        const k = rough + 1;
        const k2 = (k * k) / 8;
        const Gv = ndotv / (ndotv * (1 - k2) + k2);
        const Gl = ndotl / (ndotl * (1 - k2) + k2);
        const G = Gv * Gl;
        const F = 0.02 + (1 - 0.02) * Math.pow(1 - vdoth, 5);
        let spec = (D * G * F) / Math.max(0.001, 4 * ndotv * ndotl);
        const sparkle = smoothstep(0.58, 0.94, rippleMix) * (1 - rough);
        const glint = Math.pow(sunAlign, lerp(70, 240, 1 - rough))
          * smoothstep(0.2, 1, ndotl)
          * (0.4 + 0.6 * heightNorm);
        spec *= (1 + sparkle * 1.6 + glint * 0.9) * (1 - foamHint * 0.5);

        const d = clamp(distance[p] / maxD, 0, 1);
        const depthMix = smoothstep(0.04, 0.98, d);
        let baseR = lerp(baseWaterNearR, baseWaterFarR, depthMix);
        let baseG = lerp(baseWaterNearG, baseWaterFarG, depthMix);
        let baseB = lerp(baseWaterNearB, baseWaterFarB, depthMix);

        const depth = lerp(0.8, 12.5, depthMix * depthMix);
        const attenR = 1 / (1 + depth * 0.28);
        const attenG = 1 / (1 + depth * 0.12);
        const attenB = 1 / (1 + depth * 0.05);
        baseR *= attenR;
        baseG *= attenG;
        baseB *= attenB;

        const scatterMix = (1 - depthMix) * lerp(0.18, 0.34, 1 - elevNorm) * lerp(0.9, 1.15, heightNorm);
        baseR = lerp(baseR, scatterR, scatterMix);
        baseG = lerp(baseG, scatterG, scatterMix);
        baseB = lerp(baseB, scatterB, scatterMix);

        const tint = clamp(h * 0.08 + 0.5, 0, 1);
        baseR = lerp(baseR * 0.92, baseR * 1.06, tint);
        baseG = lerp(baseG * 0.92, baseG * 1.06, tint);
        baseB = lerp(baseB * 0.92, baseB * 1.06, tint);

        const ambient = lerp(0.08, 0.2, elevNorm) + sea * 0.02;
        const diffuse = ambient + ndotl * lerp(0.32, 0.7, elevNorm);
        const refractR = baseR * diffuse;
        const refractG = baseG * diffuse;
        const refractB = baseB * diffuse;

        let r = lerp(refractR, reflSkyR, fresnel);
        let g = lerp(refractG, reflSkyG, fresnel);
        let b = lerp(refractB, reflSkyB, fresnel);

        r += spec * sunR * sunIntensity;
        g += spec * sunG * sunIntensity;
        b += spec * sunB * sunIntensity;

        const haze = clamp(smoothstep(0.22, 1.0, d) * lerp(0.85, 1.1, sea), 0, 1);
        r = lerp(r, skyR, haze * 0.78);
        g = lerp(g, skyG, haze * 0.78);
        b = lerp(b, skyB, haze * 0.82);

        const foamNoiseA = sampleNoise(flowX * foamScale + foamDrift, flowZ * foamScale * 0.7 - foamDrift * 0.5);
        const foamNoiseB = sampleNoise(flowX * foamScale * 2.3 + foamDrift * 1.3, flowZ * foamScale * 1.8 - foamDrift * 1.1);
        const foamField = foamNoiseA * 0.6 + foamNoiseB * 0.4;
        const crestBoost = smoothstep(0.1, 0.9, crest) * smoothstep(0.0, 0.85, h / (ampScale + 0.6));
        const foamBase = smoothstep(0.18, lerp(1.0, 0.7, elevNorm), slope2 * (0.7 + 1.4 * sea));
        const foam = clamp(
          foamBase * smoothstep(0.35, 0.82, foamField) * (0.25 + 0.75 * crestBoost)
            * lerp(0.2, 0.75, sea) * lerp(0.75, 1.05, rippleMix),
          0,
          1
        );
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

  buildNoise();
  syncFromInputs();
  resize();

  if (prefersReducedMotion) {
    setStatus('Paused (reduce motion)');
    setToggleIcon(true);
    toggleBtn.setAttribute('aria-pressed', 'true');
  }

  if (!state.paused) start();
})();
