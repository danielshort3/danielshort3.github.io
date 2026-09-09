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
  const qualityInput = $('#ocean-wave-quality');
  const sceneInput = $('#ocean-wave-scene');
  const toggleBtn = $('#ocean-wave-toggle');
  const resetBtn = $('#ocean-wave-reset');
  const resetCameraBtn = $('#ocean-wave-reset-camera');
  const randomizeBtn = $('#ocean-wave-randomize');
  const copyLinkBtn = $('#ocean-wave-copy-link');
  const statusEl = $('#ocean-wave-status');
  const summaryEl = $('#ocean-wave-summary');
  const presetButtons = Array.from(document.querySelectorAll('[data-ocean-preset]'));

  if (
    !stage || !canvas || !windInput || !heightInput || !lightInput || !qualityInput
    || !toggleBtn || !resetBtn || !resetCameraBtn || !randomizeBtn || !copyLinkBtn
    || !statusEl || !summaryEl
  ) return;

  let gl = null;
  let ctx = null;
  try {
    gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  } catch {}
  if (!gl && !ctx) {
    statusEl.textContent = 'Canvas unavailable in this browser.';
    stage.classList.add('has-render-error');
    toggleBtn.disabled = true;
    return;
  }
  let rendererAvailable = true;
  let bufferUnavailable = false;
  if (!gl) {
    const coveOption = sceneInput?.querySelector('option[value="cove"]');
    if (coveOption) coveOption.disabled = true;
  }

  const TAU = Math.PI * 2;
  const G = 9.81;
  const DEG = Math.PI / 180;
  const MAX_WAVE_HEIGHT = 5.5;
  const WIND_DIR = 86 * DEG;

  const DEFAULTS = {
    wind: 2.4,
    waveHeight: 0.65,
    sunElevationDeg: 6,
    qualityMode: 'auto',
    mood: 'dawn',
    sceneKind: 'ocean',
    cameraYawDeg: 0,
    cameraPitchDeg: -6,
  };

  const PRESETS = {
    'calm-dawn': { wind: 2.4, waveHeight: 0.65, sunElevationDeg: 6, yawDeg: 0, pitchDeg: -6, mood: 'dawn' },
    'open-ocean-swell': { wind: 4.8, waveHeight: 0.95, sunElevationDeg: 42, yawDeg: -8, pitchDeg: -8, mood: 'daylight' },
    'golden-hour': { wind: 3.2, waveHeight: 0.8, sunElevationDeg: 7, yawDeg: 8, pitchDeg: -5, mood: 'golden' },
    dusk: { wind: 1.8, waveHeight: 0.55, sunElevationDeg: 5, yawDeg: -30, pitchDeg: -5, mood: 'dusk' },
  };

  const QUALITY_PROFILES = {
    low: { label: 'Low', tier: 0, fps: 30, dprMax: 1, maxPixels: 720000, scale: 0.8, maxW: 1600, maxH: 1200 },
    medium: { label: 'Medium', tier: 1, fps: 30, dprMax: 1, maxPixels: 1500000, scale: 1, maxW: 2200, maxH: 1800 },
    high: { label: 'High', tier: 2, fps: 60, dprMax: 2, maxPixels: 4000000, scale: 1, maxW: 3200, maxH: 2560 },
    ultra: { label: 'Ultra', tier: 3, fps: 30, dprMax: 2, maxPixels: 8300000, scale: 1.25, maxW: 4096, maxH: 4096 },
  };
  const AUTO_QUALITY_PROFILE = { ...QUALITY_PROFILES.high, fps: 30 };
  const QUALITY_MODES = new Set(['auto', ...Object.keys(QUALITY_PROFILES)]);
  const normalizeQuality = value => {
    const migrated = value === 'battery' ? 'low' : value === 'quality' ? 'high' : value;
    return QUALITY_MODES.has(migrated) ? migrated : null;
  };
  const MOODS = ['dawn', 'daylight', 'golden', 'dusk'];
  const SCENES = new Set(['ocean', 'cove']);
  const PREFERENCES_KEY = 'ocean-wave-preferences-v1';
  const LIGHT_TRANSITION_SECONDS = 6;

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
    qualityMode: DEFAULTS.qualityMode,
    mood: DEFAULTS.mood,
    sceneKind: DEFAULTS.sceneKind,
    brightness: 1,
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
    x: 0,
    z: 0,
    height: 2.2,
    yaw: DEFAULTS.cameraYawDeg * DEG,
    pitch: DEFAULTS.cameraPitchDeg * DEG,
    minPitch: -65 * DEG,
    maxPitch: 65 * DEG,
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
    azimuth: 78 * DEG,
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
  let lastRenderedAt = null;
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

  const getWindDescription = () => {
    if (state.wind < 1.5) return 'calm air';
    if (state.wind < 6) return 'light breeze';
    if (state.wind < 12) return 'fresh breeze';
    if (state.wind < 17) return 'strong wind';
    return 'gale-force wind';
  };

  const getWaveDescription = () => {
    if (state.waveHeight < 0.7) return 'gentle ripples';
    if (state.waveHeight < 1.8) return 'rolling swell';
    if (state.waveHeight < 3.3) return 'rough sea';
    return 'steep storm waves';
  };

  const getLightDescription = () => {
    if (state.sunElevationDeg < 15) return 'golden low-angle light';
    if (state.sunElevationDeg < 35) return 'low daylight';
    if (state.sunElevationDeg < 58) return 'clear daylight';
    return 'high overhead light';
  };

  const getQualityLabel = () => {
    return QUALITY_PROFILES[state.qualityMode]?.label || 'Auto';
  };

  const updateConditionSummary = () => {
    const yawDeg = Math.round(camera.yaw / DEG);
    const pitchDeg = Math.round(camera.pitch / DEG);
    summaryEl.textContent = `${state.sceneKind === 'cove' ? 'Quiet cove' : 'Open ocean'}, ${state.mood}. `
      + `${getWindDescription()} at ${fmt(state.wind, 1)} meters per second, `
      + `${getWaveDescription()} at ${fmt(state.waveHeight, 2)} meters, ${Math.round(state.brightness * 100)} percent brightness. `
      + `Camera height ${fmt(camera.height, 1)} meters, heading ${yawDeg} degrees, pitch ${pitchDeg} degrees. `
      + `Animation ${state.paused ? 'paused' : 'running'} in ${getQualityLabel()} mode.`;
  };

  const buildSceneUrl = () => {
    const url = new URL(window.location.href);
    url.searchParams.set('wind', fmt(state.wind, 1));
    url.searchParams.set('waves', fmt(state.waveHeight, 2));
    url.searchParams.set('sun', String(Math.round(state.sunElevationDeg)));
    url.searchParams.set('yaw', String(Math.round(camera.yaw / DEG)));
    url.searchParams.set('pitch', String(Math.round(camera.pitch / DEG)));
    url.searchParams.set('quality', state.qualityMode);
    url.searchParams.set('mood', state.mood);
    url.searchParams.set('scene', state.sceneKind);
    url.searchParams.set('brightness', fmt(state.brightness, 2));
    url.searchParams.set('cx', fmt(camera.x, 2));
    url.searchParams.set('cz', fmt(camera.z, 2));
    url.searchParams.set('alt', fmt(camera.height, 2));
    return url;
  };

  let savedPreferences = '';
  const savePreferences = () => {
    const serialized = JSON.stringify({
      mood: state.mood, scene: state.sceneKind, wind: state.wind, waves: state.waveHeight,
      brightness: state.brightness, quality: state.qualityMode, sun: state.sunElevationDeg,
    });
    if (serialized === savedPreferences) return;
    try {
      window.localStorage.setItem(PREFERENCES_KEY, serialized);
      savedPreferences = serialized;
    } catch {}
  };

  const replaceSceneUrl = () => {
    if (disposed) return;
    savePreferences();
    if (!window.history || typeof window.history.replaceState !== 'function') return;
    const url = buildSceneUrl();
    try {
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    } catch {}
  };

  let sceneUrlTimeout = 0;
  const scheduleSceneUrlUpdate = () => {
    if (disposed) return;
    window.clearTimeout(sceneUrlTimeout);
    sceneUrlTimeout = window.setTimeout(replaceSceneUrl, 180);
  };

  const syncUI = () => {
    windValue.textContent = `${fmt(state.wind, 1)} m/s`;
    heightValue.textContent = `${fmt(state.waveHeight, 2)} m`;
    lightValue.textContent = `${Math.round(state.brightness * 100)}%`;
    windInput.setAttribute('aria-valuetext', `${fmt(state.wind, 1)} meters per second, ${getWindDescription()}`);
    heightInput.setAttribute('aria-valuetext', `${fmt(state.waveHeight, 2)} meters, ${getWaveDescription()}`);
    lightInput.setAttribute('aria-valuetext', `${Math.round(state.brightness * 100)} percent brightness`);
    qualityInput.value = state.qualityMode;
    if (sceneInput) sceneInput.value = state.sceneKind;
    toggleBtn.setAttribute('aria-pressed', state.paused ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', state.paused ? 'Play animation' : 'Pause animation');
    toggleBtn.title = state.paused ? 'Play animation' : 'Pause animation';
    setToggleIcon(state.paused);
    setStatus(rendererAvailable ? `${state.paused ? 'Paused' : 'Running'} - ${getQualityLabel()}`
      : bufferUnavailable ? 'Graphics memory is unavailable. Choose a lower quality.' : 'Live waves are unavailable on this device.');
    stage.dataset.oceanPaused = String(state.paused);
    updateConditionSummary();
  };

  const updateLight = () => {
    const elev = clamp(state.sunElevationDeg, 5, 85) * DEG;
    const cosElev = Math.cos(elev);
    light.dirX = Math.cos(light.azimuth) * cosElev;
    light.dirY = Math.sin(elev);
    light.dirZ = Math.sin(light.azimuth) * cosElev;

    const xYaw = light.dirX * camera.cosYaw - light.dirZ * camera.sinYaw;
    const zYaw = light.dirX * camera.sinYaw + light.dirZ * camera.cosYaw;

    const camY = light.dirY * camera.cosPitch - zYaw * camera.sinPitch;
    const camZ = light.dirY * camera.sinPitch + zYaw * camera.cosPitch;
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
    if (!width || !height) return;

    camera.tanHalfFovY = Math.tan(camera.fovY / 2);
    camera.tanHalfFovX = camera.tanHalfFovY * (width / height);
    camera.cosPitch = Math.cos(camera.pitch);
    camera.sinPitch = Math.sin(camera.pitch);
    camera.cosYaw = Math.cos(camera.yaw);
    camera.sinYaw = Math.sin(camera.yaw);

    if (gl) {
      updateLight();
      return;
    }
    if (!waterMask) return;

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

        const dyP = dy * camera.cosPitch + dz * camera.sinPitch;
        const dzP = -dy * camera.sinPitch + dz * camera.cosPitch;
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

        const wx = camera.x + dxW * t;
        const wz = camera.z + dzW * t;
        worldX[p] = wx;
        worldZ[p] = wz;
        distance[p] = t;
        waterMask[p] = 1;

        let vx = camera.x - wx;
        let vy = camY;
        let vz = camera.z - wz;
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
    const speedScale = lerp(0.38, 0.85, sea);
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

  let gpuProgram = null;
  let gpuBuffer = null;
  let gpuUniforms = null;
  let contextLost = false;
  let adaptiveScale = 1;
  let qualityProfileMode = state.qualityMode;
  let slowFrameCount = 0;
  let fastFrameCount = 0;
  let lastQualityAdjustment = 0;
  let disposed = false;
  let displayedScene = null;
  let lastGpuTime = null;
  let waveTimeSec = 0;
  let spectrum = null;
  let environment = null;
  let emptyTexture = null;
  let previousSky = null;
  let currentSky = null;
  let pendingSky = null;
  let skyMix = 0;
  let skyReady = 0;
  let lastSpectrumTime = null;
  let lastSpectrumWind = null;
  let lastSpectrumHeight = null;

  const skyRotation = (sky) => sky?.hasSun
    ? Math.PI / 2 - light.azimuth - Math.atan2(sky.sunDirection[0], sky.sunDirection[2]) : 0;

  const releaseOceanResources = () => {
    environment?.dispose();
    spectrum?.dispose();
    if (emptyTexture) gl.deleteTexture(emptyTexture);
    spectrum = null;
    environment = null;
    emptyTexture = null;
    previousSky = null;
    currentSky = null;
    pendingSky = null;
    skyMix = 0;
    skyReady = 0;
    lastSpectrumTime = null;
  };

  const vertexSource = `
    attribute vec2 position;
    void main() {
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentSource = window.OceanWaveShaders.fragment;

  const setupGpu = () => {
    if (!gl || contextLost || disposed) return false;
    const shaders = [];
    let program = null;
    try {
      const precision = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
      let shaderFragment = precision && precision.precision > 0
        ? fragmentSource : fragmentSource.replace('precision highp float;', 'precision mediump float;');
      if (gl.getExtension('EXT_shader_texture_lod')) {
        shaderFragment = '#extension GL_EXT_shader_texture_lod : enable\n#define OCEAN_EXPLICIT_LOD\n' + shaderFragment;
      }
      if (gl.getExtension('OES_standard_derivatives')) {
        shaderFragment = '#extension GL_OES_standard_derivatives : enable\n#define OCEAN_DERIVATIVES\n' + shaderFragment;
      }
      for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, shaderFragment]]) {
        const shader = gl.createShader(type);
        if (!shader) throw new Error('Shader unavailable.');
        shaders.push(shader);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Shader unavailable.');
      }
      program = gl.createProgram();
      if (!program) throw new Error('Renderer unavailable.');
      shaders.forEach((shader) => gl.attachShader(program, shader));
      gl.bindAttribLocation(program, 0, 'position');
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Renderer unavailable.');
      gpuProgram = program;
      gpuBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, gpuBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.useProgram(gpuProgram);
      const position = gl.getAttribLocation(gpuProgram, 'position');
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gpuUniforms = Object.fromEntries(
        ['resolution', 'time', 'wind', 'waveHeight', 'elevation', 'cameraAngle', 'cameraPosition', 'mood', 'brightness', 'sceneKind', 'renderQuality',
          'swellField', 'rippleField', 'spectralReady', 'spectralMipmaps', 'fieldLengths', 'environmentA', 'environmentB',
          'environmentScale', 'environmentRotation', 'environmentMix', 'environmentReady', 'sunDirection', 'solarStrength']
          .map((name) => [name, gl.getUniformLocation(gpuProgram, name)])
      );
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      emptyTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, emptyTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      spectrum = window.OceanWaveSpectrum?.create(gl, { size: 128 });
      environment = window.OceanWaveEnvironment?.create(gl, { onChange: (sky) => {
        if (disposed || contextLost) return;
        if (sky === currentSky) { pendingSky = null; return; }
        if (currentSky && skyMix < 1 && !state.paused && !prefersReducedMotion) {
          // Finish the visible fade before changing its destination. Rapid
          // choices therefore never replace half of the sky in one frame.
          pendingSky = sky;
          return;
        }
        previousSky = currentSky || sky;
        currentSky = sky;
        pendingSky = null;
        skyMix = state.paused || prefersReducedMotion ? 1 : 0;
        if (state.paused || prefersReducedMotion) skyReady = 1;
        stage.dataset.oceanSky = 'photographic';
        renderGpu(simTimeSec);
      } });
      environment?.select(state.mood);
      canvas.style.opacity = '';
      rendererAvailable = true;
      toggleBtn.disabled = false;
      stage.classList.remove('has-render-error');
      stage.dataset.oceanRenderer = 'webgl';
      stage.dispatchEvent(new CustomEvent('ocean:renderer', { detail: { renderer: 'webgl', available: true } }));
      return true;
    } catch {
      releaseOceanResources();
      if (program) gl.deleteProgram(program);
      gpuProgram = null;
      rendererAvailable = false;
      toggleBtn.disabled = true;
      stage.classList.add('has-render-error');
      stop();
      // The CSS horizon remains available if an older driver cannot compile.
      canvas.style.opacity = '0';
      setStatus('Live waves are unavailable on this device.');
      stage.dispatchEvent(new CustomEvent('ocean:renderer', { detail: { renderer: 'webgl', available: false } }));
      return false;
    } finally {
      shaders.forEach((shader) => gl.deleteShader(shader));
    }
  };

  const renderGpu = (t) => {
    if (!gpuProgram || !gpuUniforms || contextLost || !rendererAvailable || !width || !height) return;
    if (!displayedScene) {
      displayedScene = {
        wind: state.wind,
        waveHeight: state.waveHeight,
        elevation: state.sunElevationDeg,
        brightness: state.brightness,
        mood: MOODS.map((mood) => mood === state.mood ? 1 : 0),
      };
    }
    const elapsed = lastGpuTime === null ? 0 : clamp(t - lastGpuTime, 0, 0.1);
    lastGpuTime = t;
    const immediate = state.paused || prefersReducedMotion;
    const blend = immediate ? 1 : 1 - Math.exp(-elapsed / 1.1);
    const lightBlend = immediate ? 1 : 1 - Math.exp(-elapsed / (LIGHT_TRANSITION_SECONDS / 3));
    displayedScene.wind = lerp(displayedScene.wind, state.wind, blend);
    displayedScene.waveHeight = lerp(displayedScene.waveHeight, state.waveHeight, blend);
    displayedScene.elevation = lerp(displayedScene.elevation, state.sunElevationDeg, lightBlend);
    displayedScene.brightness = lerp(displayedScene.brightness, state.brightness, blend);
    for (let index = 0; index < MOODS.length; index++) {
      displayedScene.mood[index] = lerp(displayedScene.mood[index], MOODS[index] === state.mood ? 1 : 0, lightBlend);
    }
    waveTimeSec += elapsed;
    // Rebuild the seeded spectrum only at meaningful wind changes; its height
    // and time still evolve continuously on the GPU between these steps.
    const spectralWind = Math.round(displayedScene.wind * 20) / 20;
    if (spectrum && (lastSpectrumTime !== waveTimeSec || lastSpectrumWind !== spectralWind
      || lastSpectrumHeight !== displayedScene.waveHeight)) {
      spectrum.update(waveTimeSec, spectralWind, displayedScene.waveHeight);
      lastSpectrumTime = waveTimeSec;
      lastSpectrumWind = spectralWind;
      lastSpectrumHeight = displayedScene.waveHeight;
    }
    if (currentSky) {
      if (pendingSky && (immediate || skyMix >= 1)) {
        previousSky = currentSky;
        currentSky = pendingSky;
        pendingSky = null;
        skyMix = immediate ? 1 : 0;
      }
      skyMix = immediate ? 1 : Math.min(1, skyMix + elapsed / LIGHT_TRANSITION_SECONDS);
      skyReady = immediate ? 1 : Math.min(1, skyReady + elapsed / LIGHT_TRANSITION_SECONDS);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(gpuProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, gpuBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const textures = [spectrum?.fields[0].texture, spectrum?.fields[1].texture, previousSky?.texture, currentSky?.texture];
    textures.forEach((texture, unit) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture || emptyTexture);
    });
    gl.uniform1i(gpuUniforms.swellField, 0);
    gl.uniform1i(gpuUniforms.rippleField, 1);
    gl.uniform1i(gpuUniforms.environmentA, 2);
    gl.uniform1i(gpuUniforms.environmentB, 3);
    gl.uniform1f(gpuUniforms.spectralReady, spectrum ? 1 : 0);
    gl.uniform1f(gpuUniforms.spectralMipmaps, spectrum?.mipmapped ? 1 : 0);
    gl.uniform2f(gpuUniforms.fieldLengths, spectrum?.fields[0].length || 180, spectrum?.fields[1].length || 18);
    gl.uniform2f(gpuUniforms.environmentScale, previousSky?.textureScale || 1, currentSky?.textureScale || 1);
    gl.uniform2f(gpuUniforms.environmentRotation, skyRotation(previousSky), skyRotation(currentSky));
    const skyBlend = smoothstep(0, 1, skyMix);
    gl.uniform1f(gpuUniforms.environmentMix, skyBlend);
    gl.uniform1f(gpuUniforms.environmentReady, skyReady);
    const fallbackSolarHeight = Math.sin(displayedScene.elevation * DEG);
    const solarHeight = lerp(previousSky?.hasSun ? previousSky.sunDirection[1] : fallbackSolarHeight,
      currentSky?.hasSun ? currentSky.sunDirection[1] : fallbackSolarHeight, skyBlend);
    const solarHorizontal = Math.sqrt(1 - solarHeight * solarHeight);
    gl.uniform3f(gpuUniforms.sunDirection, Math.cos(light.azimuth) * solarHorizontal, solarHeight, Math.sin(light.azimuth) * solarHorizontal);
    gl.uniform1f(gpuUniforms.solarStrength, lerp(previousSky ? Number(previousSky.hasSun) : 1,
      currentSky ? Number(currentSky.hasSun) : 1, skyBlend));
    gl.uniform1f(gpuUniforms.sceneKind, state.sceneKind === 'cove' ? 1 : 0);
    gl.uniform1f(gpuUniforms.renderQuality, getQualityProfile().tier);
    gl.uniform1f(gpuUniforms.brightness, displayedScene.brightness);
    gl.uniform2f(gpuUniforms.resolution, width, height);
    gl.uniform1f(gpuUniforms.time, waveTimeSec);
    gl.uniform1f(gpuUniforms.wind, displayedScene.wind);
    gl.uniform1f(gpuUniforms.waveHeight, displayedScene.waveHeight);
    gl.uniform1f(gpuUniforms.elevation, displayedScene.elevation);
    gl.uniform2f(gpuUniforms.cameraAngle, camera.yaw, camera.pitch);
    gl.uniform3f(gpuUniforms.cameraPosition, camera.x, camera.height, camera.z);
    gl.uniform4f(gpuUniforms.mood, displayedScene.mood[0], displayedScene.mood[1], displayedScene.mood[2], displayedScene.mood[3]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    contextLost = true;
    releaseOceanResources();
    rendererAvailable = false;
    toggleBtn.disabled = true;
    stage.classList.add('has-render-error');
    stop();
    canvas.style.opacity = '0';
    setStatus('Live waves are resting.');
    stage.dispatchEvent(new CustomEvent('ocean:renderer', { detail: { renderer: 'webgl', available: false } }));
  });

  canvas.addEventListener('webglcontextrestored', () => {
    contextLost = false;
    gpuProgram = null;
    gpuBuffer = null;
    gpuUniforms = null;
    if (setupGpu()) {
      renderGpu(simTimeSec);
      syncUI();
      start();
    }
  });

  const getQualityProfile = () => {
    if (qualityProfileMode !== state.qualityMode) {
      qualityProfileMode = state.qualityMode;
      adaptiveScale = 1;
      slowFrameCount = 0;
      fastFrameCount = 0;
    }
    if (gl) {
      // Manual choices remain fixed. Auto begins with crisp, high-DPI detail
      // and alone may trade resolution for smooth animation.
      return state.qualityMode === 'auto'
        ? AUTO_QUALITY_PROFILE : QUALITY_PROFILES[state.qualityMode];
    }
    // Keep the compatibility renderer usable on devices without WebGL.
    return { tier: 0, fps: 30, dprMax: 1, maxPixels: ['high', 'ultra'].includes(state.qualityMode) ? 75000 : 48000, scale: 0.5, maxW: 500, maxH: 500 };
  };

  const resize = () => {
    const rect = stage.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const quality = getQualityProfile();

    const dpr = Math.min(quality.dprMax, window.devicePixelRatio || 1);
    const targetW = rect.width * dpr;
    const targetH = rect.height * dpr;

    const renderScale = quality.scale;
    const viewportLimit = gl?.getParameter(gl.MAX_VIEWPORT_DIMS);
    const bufferLimit = gl?.getParameter(gl.MAX_RENDERBUFFER_SIZE);
    const maxBuffer = Number.isFinite(bufferLimit) && bufferLimit > 0 ? bufferLimit : Infinity;
    const maxW = Math.min(maxBuffer, viewportLimit?.[0] > 0 ? Math.min(quality.maxW, viewportLimit[0]) : quality.maxW);
    const maxH = Math.min(maxBuffer, viewportLimit?.[1] > 0 ? Math.min(quality.maxH, viewportLimit[1]) : quality.maxH);
    const aspectScale = Math.min(
      1,
      maxW / (targetW * renderScale),
      maxH / (targetH * renderScale),
      Math.sqrt(quality.maxPixels / (targetW * targetH * renderScale * renderScale))
    );
    const autoScale = state.qualityMode === 'auto' ? adaptiveScale : 1;
    const nextW = Math.max(1, Math.round(targetW * renderScale * aspectScale * autoScale));
    const nextH = Math.max(1, Math.round(targetH * renderScale * aspectScale * autoScale));
    stage.dataset.oceanQuality = state.qualityMode;
    stage.dataset.oceanResolution = `${nextW}x${nextH}`;

    if (nextW === width && nextH === height) return;

    width = nextW;
    height = nextH;
    canvas.width = width;
    canvas.height = height;

    if (gl) {
      // Drivers may allocate a smaller buffer than requested under memory
      // pressure. Shader coordinates must always match the actual GPU target.
      if (Number.isFinite(gl.drawingBufferWidth)) width = gl.drawingBufferWidth;
      if (Number.isFinite(gl.drawingBufferHeight)) height = gl.drawingBufferHeight;
      stage.dataset.oceanResolution = `${width}x${height}`;
      if (!width || !height) {
        rendererAvailable = false;
        bufferUnavailable = true;
        stage.classList.add('has-buffer-error');
        toggleBtn.disabled = true;
        canvas.style.opacity = '0';
        stop();
        setStatus('Graphics memory is unavailable. Choose a lower quality.');
        return;
      }
      const wasUnavailable = !rendererAvailable;
      if (!gpuProgram && !setupGpu()) return;
      bufferUnavailable = false;
      stage.classList.remove('has-buffer-error');
      rendererAvailable = true;
      toggleBtn.disabled = false;
      canvas.style.opacity = '';
      rebuildCameraRays();
      renderGpu(simTimeSec);
      if (wasUnavailable) start();
      return;
    }
    stage.dataset.oceanRenderer = 'canvas';

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
  let cameraController = null;
  const markCameraDirty = () => {
    cameraDirty = true;
    if (state.paused) {
      cameraDirty = false;
      rebuildCameraRays();
      renderFrame(simTimeSec);
    }
  };
  const renderFrame = (t) => {
    if (gl) {
      renderGpu(t);
      return;
    }
    if (!imageData || !pixels) return;

    const sea = clamp(state.wind / 20, 0, 1);
    const heightScale = clamp(state.waveHeight, 0, MAX_WAVE_HEIGHT);
    const heightNorm = MAX_WAVE_HEIGHT ? heightScale / MAX_WAVE_HEIGHT : 0;
    const elevNorm = clamp((state.sunElevationDeg - 5) / 70, 0, 1);
    const warmth = Math.pow(1 - elevNorm, 1.35);
    const sunIntensity = lerp(0.88, 1.3, 1 - elevNorm) * lerp(0.95, 1.08, sea);

    const dusk = state.mood === 'dusk';
    const golden = state.mood === 'golden';
    const skyTopR = dusk ? 69 : lerp(97, 87, elevNorm);
    const skyTopG = dusk ? 92 : lerp(166, 158, elevNorm);
    const skyTopB = dusk ? 135 : lerp(204, 199, elevNorm);

    let skyHorizonR = lerp(70, 174, elevNorm);
    let skyHorizonG = lerp(94, 208, elevNorm);
    let skyHorizonB = lerp(120, 232, elevNorm);
    const hazeBlend = clamp(warmth * (0.85 + sea * 0.2), 0, 1);

    skyHorizonR = lerp(skyHorizonR, 222, hazeBlend);
    skyHorizonG = lerp(skyHorizonG, 168, hazeBlend * 0.9);
    skyHorizonB = lerp(skyHorizonB, 124, hazeBlend * 0.7);
    if (dusk) {
      skyHorizonR = 150;
      skyHorizonG = 145;
      skyHorizonB = 184;
    } else if (!golden) {
      skyHorizonR = lerp(skyHorizonR, 190, 0.55);
      skyHorizonG = lerp(skyHorizonG, 212, 0.55);
      skyHorizonB = lerp(skyHorizonB, 220, 0.55);
    }

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

    const baseWaterNearR = lerp(14, 18, sea);
    const baseWaterNearG = lerp(68, 76, sea);
    const baseWaterNearB = lerp(83, 91, sea);
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
    const rippleAmp = lerp(0.015, 0.07, sea) * (0.35 + 0.65 * heightNorm);
    const rippleDrift = t * lerp(0.06, 0.18, sea);
    const foamScale = lerp(0.03, 0.075, sea) * lerp(0.9, 1.08, heightNorm);
    const foamDrift = t * lerp(0.04, 0.12, sea);

    const vignetteStrength = 0.035;
    const glowStrength = 0.55;
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

          pixels[o] = clamp(r * vignette * state.brightness + dither, 0, 255);
          pixels[o + 1] = clamp(g * vignette * state.brightness + dither, 0, 255);
          pixels[o + 2] = clamp(b * vignette * state.brightness + dither, 0, 255);
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

        const fresnel = 0.14 + 0.86 * Math.pow(1 - ndotv, 4);

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
        const sunBoost = sunFocus * sunIntensity * 0.22;
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
        spec = Math.min(0.15, spec * (1 + sparkle * 0.2 + glint * 0.1)) * (1 - foamHint * 0.5);

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

        const ambient = lerp(0.48, 0.55, elevNorm) + sea * 0.02;
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

        pixels[o] = clamp(r * vignette * state.brightness + dither, 0, 255);
        pixels[o + 1] = clamp(g * vignette * state.brightness + dither, 0, 255);
        pixels[o + 2] = clamp(b * vignette * state.brightness + dither, 0, 255);
        pixels[o + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const start = () => {
    if (rafId || disposed || !rendererAvailable || state.paused || document.hidden || stage.dataset.oceanVisible === 'false'
      || contextLost || (gl && !gpuProgram)) return;
    lastFrame = null;
    lastRenderedAt = null;
    rafId = window.requestAnimationFrame(tick);
  };

  const stop = () => {
    if (!rafId) return;
    window.cancelAnimationFrame(rafId);
    rafId = 0;
    lastFrame = null;
    lastRenderedAt = null;
  };

  const tick = (ts) => {
    if (state.paused || document.hidden || stage.dataset.oceanVisible === 'false' || contextLost) {
      stop();
      return;
    }
    const minimumFrameInterval = 1000 / getQualityProfile().fps;
    if (lastRenderedAt !== null && ts - lastRenderedAt < minimumFrameInterval - 1) {
      rafId = window.requestAnimationFrame(tick);
      return;
    }
    if (lastFrame === null) lastFrame = ts;
    const elapsed = ts - lastFrame;
    const dt = clamp(elapsed / 1000, 0, 0.05);
    lastFrame = ts;
    lastRenderedAt = ts;
    simTimeSec += dt;
    if (cameraDirty) {
      cameraDirty = false;
      rebuildCameraRays();
    }
    if (gl && state.qualityMode === 'auto') {
      const slow = elapsed > minimumFrameInterval * 1.35;
      slowFrameCount = slow ? slowFrameCount + 1 : Math.max(0, slowFrameCount - 1);
      fastFrameCount = elapsed > 0 && elapsed <= minimumFrameInterval * 1.14 ? fastFrameCount + 1 : 0;
      if (slowFrameCount >= 24 && adaptiveScale > 0.7 && ts - lastQualityAdjustment > 4000) {
        adaptiveScale = Math.max(0.7, adaptiveScale * 0.85);
        slowFrameCount = 0;
        fastFrameCount = 0;
        lastQualityAdjustment = ts;
        resize();
      } else if (fastFrameCount >= 180 && adaptiveScale < 1 && ts - lastQualityAdjustment > 8000) {
        adaptiveScale = Math.min(1, adaptiveScale / 0.85);
        fastFrameCount = 0;
        lastQualityAdjustment = ts;
        resize();
      }
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
    state.brightness = clamp(parseFloat(lightInput.value || 100) / 100, 0.5, 1.5);
    state.qualityMode = normalizeQuality(qualityInput.value) || DEFAULTS.qualityMode;
    buildWaves();
    updateLight();
    syncUI();
  };

  let activePreset = '';

  const setActivePreset = (presetName = Object.keys(PRESETS).find((name) => PRESETS[name].mood === state.mood) || '') => {
    activePreset = presetName;
    const sceneName = $('#ocean-wave-scene-name');
    if (sceneName) {
      sceneName.textContent = state.sceneKind === 'cove' ? 'Quiet cove' : 'Open ocean';
    }
    presetButtons.forEach((button) => {
      const active = button.dataset.oceanPreset === activePreset;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  };

  const syncInputsFromState = () => {
    windInput.value = fmt(state.wind, 1);
    heightInput.value = fmt(state.waveHeight, 2);
    lightInput.value = String(Math.round(state.brightness * 100));
    qualityInput.value = state.qualityMode;
    if (sceneInput) sceneInput.value = state.sceneKind;
  };

  const homePose = () => ({
    x: 0, z: 0, height: Math.max(2.2, state.waveHeight * 1.3),
    yaw: DEFAULTS.cameraYawDeg * DEG, pitch: DEFAULTS.cameraPitchDeg * DEG,
  });

  const constrainCameraPose = (pose) => {
    if (state.sceneKind === 'cove') window.OceanWaveShaders.constrainCoveCamera?.(pose);
  };

  const applyLighting = (preset, presetName) => {
    state.mood = preset.mood;
    state.sunElevationDeg = preset.sunElevationDeg;
    environment?.select(state.mood);
    updateLight();
    syncUI();
    setActivePreset(presetName);
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  };

  const applyScene = (scene, resetView = true) => {
    state.sceneKind = gl && SCENES.has(scene) ? scene : DEFAULTS.sceneKind;
    stage.dataset.oceanScene = state.sceneKind;
    if (resetView) {
      Object.assign(camera, homePose());
      cameraController?.setHomePose(homePose());
      cameraController?.sync();
      rebuildCameraRays();
    }
    stage.dispatchEvent(new CustomEvent('ocean:scene-change', { detail: { scene: state.sceneKind } }));
    syncUI();
    setActivePreset();
  };

  const applyConditions = (conditions, presetName = '') => {
    state.wind = clamp(Number(conditions.wind), 0, 20);
    state.waveHeight = clamp(Number(conditions.waveHeight), 0, MAX_WAVE_HEIGHT);
    state.sunElevationDeg = clamp(Number(conditions.sunElevationDeg), 5, 75);
    state.brightness = 1;
    if (MOODS.includes(conditions.mood)) state.mood = conditions.mood;
    environment?.select(state.mood);
    camera.height = Math.max(camera.height, state.waveHeight * 1.3);
    cameraController?.sync();
    cameraController?.setHomePose(homePose());
    syncInputsFromState();
    buildWaves();
    rebuildCameraRays();
    syncUI();
    setActivePreset(presetName || undefined);
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  };

  const restoreSceneFromUrl = () => {
    const params = new URLSearchParams(window.location.search);
    let preferences = {};
    try {
      const saved = JSON.parse(window.localStorage.getItem(PREFERENCES_KEY));
      if (saved && typeof saved === 'object' && !Array.isArray(saved)) preferences = saved;
    } catch {}
    const readNumber = (key, fallback) => {
      const saved = preferences[key];
      const baseline = typeof saved === 'number' && Number.isFinite(saved) ? saved : fallback;
      const value = params.has(key) && params.get(key).trim() !== '' ? Number(params.get(key)) : NaN;
      return Number.isFinite(value) ? value : baseline;
    };
    const readChoice = (key, allowed, fallback) => allowed.has(params.get(key)) ? params.get(key)
      : allowed.has(preferences[key]) ? preferences[key] : fallback;
    const readCamera = (key, fallback) => {
      const value = params.has(key) && params.get(key).trim() !== '' ? Number(params.get(key)) : NaN;
      return Number.isFinite(value) ? value : fallback;
    };

    state.wind = clamp(readNumber('wind', DEFAULTS.wind), 0, 20);
    state.waveHeight = clamp(readNumber('waves', DEFAULTS.waveHeight), 0, MAX_WAVE_HEIGHT);
    state.sunElevationDeg = clamp(readNumber('sun', DEFAULTS.sunElevationDeg), 5, 75);
    state.brightness = clamp(readNumber('brightness', 1), 0.5, 1.5);
    camera.x = clamp(readCamera('cx', 0), -100000, 100000);
    camera.z = clamp(readCamera('cz', 0), -100000, 100000);
    camera.height = clamp(readCamera('alt', 2.2), Math.max(1.4, state.waveHeight * 1.3), 24);
    state.mood = readChoice('mood', new Set(MOODS), DEFAULTS.mood);
    state.sceneKind = gl ? readChoice('scene', SCENES, DEFAULTS.sceneKind) : DEFAULTS.sceneKind;
    state.qualityMode = normalizeQuality(params.get('quality')) || normalizeQuality(preferences.quality) || DEFAULTS.qualityMode;
    camera.yaw = wrapAngle(readCamera('yaw', DEFAULTS.cameraYawDeg) * DEG);
    camera.pitch = clamp(
      readCamera('pitch', DEFAULTS.cameraPitchDeg) * DEG,
      camera.minPitch,
      camera.maxPitch
    );
    constrainCameraPose(camera);
    syncInputsFromState();
  };

  const copyText = async (text) => {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (!copied) throw new Error('Copy command was unavailable.');
  };

  windInput.addEventListener('input', () => {
    state.wind = clamp(parseFloat(windInput.value), 0, 20);
    buildWaves();
    syncUI();
    setActivePreset();
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  });

  heightInput.addEventListener('input', () => {
    state.waveHeight = clamp(parseFloat(heightInput.value), 0, MAX_WAVE_HEIGHT);
    camera.height = Math.max(camera.height, state.waveHeight * 1.3);
    cameraController?.sync();
    cameraController?.setHomePose(homePose());
    rebuildCameraRays();
    syncUI();
    setActivePreset();
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  });

  lightInput.addEventListener('input', () => {
    state.brightness = clamp(parseFloat(lightInput.value) / 100, 0.5, 1.5);
    syncUI();
    setActivePreset();
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  });

  qualityInput.addEventListener('change', () => {
    state.qualityMode = normalizeQuality(qualityInput.value) || DEFAULTS.qualityMode;
    lastFrame = null;
    lastRenderedAt = null;
    resize();
    renderFrame(simTimeSec);
    syncUI();
    scheduleSceneUrlUpdate();
  });

  sceneInput?.addEventListener('change', () => {
    if (sceneInput.value === state.sceneKind) return;
    applyScene(sceneInput.value);
    renderFrame(simTimeSec);
    scheduleSceneUrlUpdate();
  });

  const onRest = () => {
    cameraController?.setEnabled(false);
    setPaused(true);
  };
  const onResume = () => { setPaused(false); };
  stage.addEventListener('ocean:rest', onRest);
  stage.addEventListener('ocean:resume', onResume);

  toggleBtn.addEventListener('click', () => {
    setPaused(!state.paused);
  });

  resetBtn.addEventListener('click', () => {
    applyConditions(PRESETS['calm-dawn'], 'calm-dawn');
  });

  resetCameraBtn.addEventListener('click', () => {
    cameraController?.reset();
    stage.focus({ preventScroll: true });
  });

  randomizeBtn.addEventListener('click', () => {
    applyConditions({
      wind: 1 + Math.random() * 6,
      waveHeight: 0.25 + Math.random() * 1.15,
      sunElevationDeg: 6 + Math.random() * 49,
      yawDeg: -20 + Math.random() * 40,
      pitchDeg: -5 - Math.random() * 4,
      mood: MOODS[Math.floor(Math.random() * MOODS.length)],
    });
  });

  presetButtons.forEach((button) => {
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => {
      const presetName = button.dataset.oceanPreset;
      const preset = PRESETS[presetName];
      if (preset) applyLighting(preset, presetName);
    });
  });

  let copyLinkTimeout = 0;
  copyLinkBtn.addEventListener('click', async () => {
    const url = buildSceneUrl();
    replaceSceneUrl();
    try {
      await copyText(url.toString());
      if (disposed) return;
      setStatus('Scene link copied');
      copyLinkBtn.textContent = 'Link copied';
    } catch {
      if (disposed) return;
      setStatus('Copy unavailable - use the URL in the address bar');
      copyLinkBtn.textContent = 'Copy unavailable';
    }
    window.clearTimeout(copyLinkTimeout);
    copyLinkTimeout = window.setTimeout(() => { if (!disposed) copyLinkBtn.textContent = 'Copy link'; }, 2600);
  });

  let resizeTimeout = 0;
  const onResize = () => {
    window.clearTimeout(resizeTimeout);
    resizeTimeout = window.setTimeout(() => {
      if (!disposed) resize();
    }, 100);
  };
  window.addEventListener('resize', onResize);

  const onVisibilityChange = () => {
    if (document.hidden) stop();
    else if (!state.paused) start();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);
  window.addEventListener('pagehide', savePreferences);

  const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(onResize) : null;
  resizeObserver?.observe(stage);
  const intersectionObserver = typeof IntersectionObserver === 'function' ? new IntersectionObserver((entries) => {
    const visible = entries.some((entry) => entry.isIntersecting);
    const changed = stage.dataset.oceanVisible !== String(visible);
    stage.dataset.oceanVisible = String(visible);
    if (changed) stage.dispatchEvent(new CustomEvent('ocean:visibility', { detail: { visible } }));
    if (visible) start();
    else stop();
  }, { threshold: 0.01 }) : null;
  intersectionObserver?.observe(stage);

  window.SiteRoutes?.addCleanup?.(() => {
    savePreferences();
    disposed = true;
    stop();
    window.clearTimeout(resizeTimeout);
    window.clearTimeout(sceneUrlTimeout);
    window.clearTimeout(copyLinkTimeout);
    resizeObserver?.disconnect();
    intersectionObserver?.disconnect();
    window.removeEventListener('resize', onResize);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    window.removeEventListener('pagehide', savePreferences);
    stage.removeEventListener('ocean:rest', onRest);
    stage.removeEventListener('ocean:resume', onResume);
    cameraController?.dispose();
    if (gl) releaseOceanResources();
    if (gl && !contextLost) {
      if (gpuBuffer) gl.deleteBuffer(gpuBuffer);
      if (gpuProgram) gl.deleteProgram(gpuProgram);
    }
    gpuBuffer = null;
    gpuProgram = null;
    gpuUniforms = null;
  });

  restoreSceneFromUrl();
  cameraController = window.OceanWaveCamera.create({
    stage, camera,
    toggleButton: $('#ocean-wave-camera-toggle'),
    relaxButton: $('#ocean-wave-relax'),
    resetButton: resetCameraBtn,
    pad: $('#ocean-wave-camera-pad'),
    hint: $('#ocean-wave-camera-hint'),
    constrainPose: constrainCameraPose,
    minHeight: () => Math.max(1.4, state.waveHeight * 1.3),
    maxHeight: 24,
    speed: 2.4,
    onChange: markCameraDirty,
    onCommit: () => { updateConditionSummary(); scheduleSceneUrlUpdate(); },
    onReset: () => { syncUI(); scheduleSceneUrlUpdate(); },
  });
  cameraController.setHomePose(homePose());
  buildNoise();
  syncFromInputs();
  applyScene(state.sceneKind, false);
  resize();

  if (prefersReducedMotion) {
    setStatus('Paused (reduce motion)');
    setToggleIcon(true);
    toggleBtn.setAttribute('aria-pressed', 'true');
  }

  if (!state.paused) start();
})();
