'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = name => fs.readFileSync(path.join(__dirname, '../../js/tools', name), 'utf8');
const PREFERENCES_KEY = 'ocean-wave-preferences-v1';

class Element {
  constructor() {
    this.dataset = {};
    this.style = {};
    this.attributes = new Map();
    this.listeners = new Map();
    const classes = new Set();
    this.classList = {
      add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name),
      toggle: (name, force) => { if (force) classes.add(name); else classes.delete(name); },
    };
  }
  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(callback);
  }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  dispatchEvent(event) {
    event.target ??= this;
    event.preventDefault ??= () => { event.defaultPrevented = true; };
    for (const listener of [...this.listeners.get(event.type) || []]) listener(event);
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  getBoundingClientRect() { return { width: 600, height: 400 }; }
  focus() {}
  closest() { return null; }
}

const harness = ({ stored = {}, search = '', reduced = false, blockedStorage = false, webgl = true,
  viewport = { width: 600, height: 400 }, dpr = 1, maxViewport = null, maxRenderbuffer = null,
  drawingBuffer = null } = {}) => {
  const elements = new Map();
  const get = name => {
    if (!elements.has(name)) elements.set(name, new Element());
    return elements.get(name);
  };
  const presets = ['calm-dawn', 'open-ocean-swell', 'golden-hour', 'dusk'].map(name => {
    const button = new Element();
    button.dataset.oceanPreset = name;
    return button;
  });
  const document = new Element();
  document.hidden = false;
  document.querySelector = selector => get(selector.replace('#ocean-wave-', ''));
  document.querySelectorAll = () => presets;
  const window = new Element();
  const queue = new Map();
  const timers = new Map();
  let sequence = 0;
  let timestamp = 0;
  const cleanups = [];
  const uniforms = {};
  const contextRequests = [];
  const gpuViewports = [];
  let bufferAllocation = drawingBuffer;
  let draws = 0;
  let camera;
  let skyChange;
  let visibilityChange;
  const conditions = [];
  const spectrumUpdates = [];
  const spectrumViews = [];
  const motionPreference = new Element();
  motionPreference.matches = reduced;
  const storage = new Map([[PREFERENCES_KEY, typeof stored === 'string' ? stored : JSON.stringify(stored)]]);
  window.localStorage = {
    getItem: key => { if (blockedStorage) throw Error('Storage blocked'); return storage.get(key) || null; },
    setItem: (key, value) => { if (blockedStorage) throw Error('Storage blocked'); storage.set(key, value); },
  };
  window.location = { href: `https://example.test/games/ocean-wave-simulation${search}`, search };
  window.history = { replaceState: (_state, _title, url) => { window.location.href = new URL(url, window.location.href).href; } };
  window.matchMedia = () => motionPreference;
  window.devicePixelRatio = dpr;
  window.performance = { now: () => timestamp };
  window.requestAnimationFrame = callback => { const id = ++sequence; queue.set(id, callback); return id; };
  window.cancelAnimationFrame = id => queue.delete(id);
  window.setTimeout = callback => { const id = ++sequence; timers.set(id, callback); return id; };
  window.clearTimeout = id => timers.delete(id);
  window.SiteRoutes = { addCleanup: callback => cleanups.push(callback) };
  window.OceanWaveSpectrum = {
    create: () => ({
      fields: [{ texture: {}, length: 180 }, { texture: {}, length: 18 }],
      update: (...values) => spectrumUpdates.push(values),
      setView: (...values) => spectrumViews.push(values),
      dispose() {},
    }),
  };
  window.OceanWaveEnvironment = {
    create: (_gl, options) => {
      skyChange = options.onChange;
      return { select() {}, dispose() {} };
    },
  };
  const gl = new Proxy({
    MAX_VIEWPORT_DIMS: 0x0D3A,
    MAX_RENDERBUFFER_SIZE: 0x84E8,
    getParameter: name => name === 0x0D3A && maxViewport ? new Int32Array(maxViewport)
      : name === 0x84E8 ? maxRenderbuffer : null,
    get drawingBufferWidth() { return bufferAllocation?.[0] ?? get('canvas').width; },
    get drawingBufferHeight() { return bufferAllocation?.[1] ?? get('canvas').height; },
    viewport: (...values) => { gpuViewports.push(values); },
    getShaderPrecisionFormat: () => ({ precision: 23 }),
    getShaderParameter: () => true, getProgramParameter: () => true,
    getUniformLocation: (_program, name) => name,
    getAttribLocation: () => 0, getExtension: () => null,
    uniform1f: (name, value) => { uniforms[name] = value; },
    uniform2f: (name, ...value) => { uniforms[name] = value; },
    uniform3f: (name, ...value) => { uniforms[name] = value; },
    uniform4f: (name, ...value) => { uniforms[name] = value; },
    drawArrays: () => { draws++; },
  }, { get: (target, name) => name in target ? target[name] : (() => ({})) });
  get('scene').querySelector = selector => selector === 'option[value="cove"]' ? get('cove-option') : null;
  const canvas2d = { createImageData: (width, height) => ({ data: new Uint8ClampedArray(width * height * 4) }), putImageData() {} };
  get('canvas').getContext = (type, options) => {
    contextRequests.push({ type, options });
    return type === 'webgl' ? (webgl ? gl : null) : canvas2d;
  };
  get('stage').getBoundingClientRect = () => webgl ? viewport : { width: 24, height: 24 };
  get('stage').addEventListener('ocean:conditions', event => conditions.push({ ...event.detail }));
  const context = vm.createContext({
    window, document, navigator: {}, URL, URLSearchParams, performance: window.performance,
    CustomEvent: class { constructor(type, options) { this.type = type; Object.assign(this, options); } },
    IntersectionObserver: class {
      constructor(callback) { visibilityChange = callback; }
      observe() {}
      disconnect() {}
    },
  });
  vm.runInContext(script('ocean-wave-shaders.js'), context);
  vm.runInContext(script('ocean-wave-camera.js'), context);
  const createCamera = window.OceanWaveCamera.create;
  window.OceanWaveCamera.create = options => { camera = options.camera; return createCamera(options); };
  vm.runInContext(script('ocean-wave-simulation.js'), context);
  const frame = (milliseconds = 1000 / 60) => {
    timestamp += milliseconds;
    const callbacks = [...queue.values()];
    queue.clear();
    callbacks.forEach(callback => callback(timestamp));
  };
  return {
    window, document, get, camera, uniforms, queue, contextRequests, gpuViewports, frame,
    conditions, spectrumUpdates, spectrumViews, motionPreference,
    setDrawingBuffer: value => { bufferAllocation = value; },
    get drawCount() { return draws; },
    dimensions: () => [get('canvas').width, get('canvas').height],
    preferences: () => JSON.parse(storage.get(PREFERENCES_KEY)),
    click: name => get(name).dispatchEvent({ type: 'click' }),
    preset: name => presets.find(button => button.dataset.oceanPreset === name).dispatchEvent({ type: 'click' }),
    selectedPreset: () => presets.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.oceanPreset,
    input: (name, value, type = 'input') => { get(name).value = String(value); get(name).dispatchEvent({ type }); },
    check: (name, checked) => { get(name).checked = checked; get(name).dispatchEvent({ type: 'change' }); },
    reduceMotion: value => { motionPreference.matches = value; motionPreference.dispatchEvent({ type: 'change' }); },
    visible: value => visibilityChange([{ isIntersecting: value }]),
    flush: () => { const pending = [...timers.values()]; timers.clear(); pending.forEach(callback => callback()); },
    sky: (name, height = .3) => skyChange({ texture: { name }, textureScale: 1, hasSun: true, sunDirection: [0, height, .8] }),
    step: (seconds, milliseconds = 1000 / 60) => {
      for (let i = 0; i < seconds * 1000 / milliseconds; i++) frame(milliseconds);
    },
    dispose: () => cleanups.forEach(cleanup => cleanup()),
  };
};

{
  const app = harness({ stored: { mood: 'dusk', scene: 'cove', wind: 3.7, waves: .85, brightness: .8,
    quality: 'battery', cx: 800, cz: 900, alt: 18, yaw: 120, pitch: 25 } });
  assert.equal(app.selectedPreset(), 'dusk');
  assert.equal(app.get('stage').dataset.oceanScene, 'cove');
  assert.equal(app.get('wind').value, '3.7');
  assert.equal(app.get('height').value, '0.85');
  assert.equal(app.get('light').value, '80');
  assert.equal(app.get('quality').value, 'low');
  assert.equal(app.camera.x, 0, 'Returning visitors must receive the composed view rather than a saved exploration position.');
  assert.equal(app.camera.z, 0);
  assert.equal(app.camera.height, 2.2);
  assert.equal(app.camera.yaw, 0);
  assert.equal(app.get('stage').dataset.oceanCameraMode, 'relax');
  app.dispose();
}

{
  const app = harness({ stored: { mood: 'dusk', scene: 'cove', wind: 3.7, waves: .85, brightness: .8 },
    search: '?mood=daylight&wind=4.2&cx=12&cz=-5&alt=7&yaw=24&pitch=-12' });
  assert.equal(app.selectedPreset(), 'open-ocean-swell');
  assert.equal(app.get('wind').value, '4.2');
  assert.equal(app.get('height').value, '0.85', 'A URL only overrides preferences for parameters it actually contains.');
  assert.equal(app.get('light').value, '80');
  assert.equal(app.get('stage').dataset.oceanScene, 'cove');
  assert.equal(app.camera.x, 12);
  assert.equal(app.camera.z, -5);
  assert.equal(app.camera.height, 7);
  const pose = JSON.stringify(app.camera);
  app.preset('golden-hour');
  assert.equal(JSON.stringify(app.camera), pose, 'Choosing lighting must not teleport or tilt the camera.');
  assert.equal(app.get('wind').value, '4.2');
  assert.equal(app.get('height').value, '0.85');
  assert.equal(app.get('light').value, '80', 'Choosing lighting must preserve brightness.');
  app.input('wind', 2.6);
  assert.equal(app.selectedPreset(), 'golden-hour', 'Adjusting sea conditions must not clear the selected lighting.');
  app.flush();
  assert.equal(app.preferences().mood, 'golden');
  assert.equal(app.preferences().wind, 2.6);
  assert.equal(app.preferences().cx, undefined, 'Persistent preferences must exclude exploration positions.');
  assert.equal(new URL(app.window.location.href).searchParams.get('cx'), '12.00', 'Shared scene links retain explicitly positioned cameras.');
  app.click('camera-toggle');
  assert.equal(app.get('reset-camera').hidden, false);
  app.click('reset-camera');
  assert.equal(app.camera.x, 0, 'Recenter must return to the composed scene rather than the initial URL viewpoint.');
  assert.equal(app.camera.height, 2.2);
  app.dispose();
}

{
  const app = harness();
  app.sky('dawn', .1);
  app.step(7);
  app.preset('open-ocean-swell');
  app.sky('daylight', .8);
  assert.equal(app.uniforms.environmentMix, 0);
  app.step(3);
  assert.ok(app.uniforms.environmentMix > .45 && app.uniforms.environmentMix < .55, 'HDR skies must ease over six seconds.');
  assert.ok(app.uniforms.sunDirection[1] > .4 && app.uniforms.sunDirection[1] < .5, 'Sunlight must move with the sky transition.');
  const midFade = app.uniforms.environmentMix;
  app.preset('dusk');
  app.sky('dusk', .15);
  assert.equal(app.uniforms.environmentMix, midFade, 'A rapid lighting selection must not replace a partially faded sky abruptly.');
  app.step(3.2);
  assert.ok(app.uniforms.environmentMix < .05, 'The latest requested lighting begins once the visible transition finishes.');
  app.step(6);
  assert.equal(app.uniforms.environmentMix, 1);
  assert.ok(Math.abs(app.uniforms.sunDirection[1] - .15) < .0001);
  app.click('toggle');
  app.preset('calm-dawn');
  app.sky('dawn', .1);
  assert.equal(app.uniforms.environmentMix, 1, 'Paused scenes must show newly selected lighting immediately.');
  assert.equal(app.queue.size, 0, 'Changing paused lighting must not start an animation loop.');
  app.dispose();
}

{
  const app = harness({ reduced: true });
  app.sky('dawn', .1);
  app.preset('dusk');
  app.sky('dusk', .15);
  assert.equal(app.uniforms.environmentMix, 1);
  assert.equal(app.get('stage').dataset.oceanPaused, 'true');
  app.camera.x = 40;
  app.input('scene', 'cove', 'change');
  assert.equal(app.get('stage').dataset.oceanScene, 'cove');
  assert.equal(app.uniforms.sceneKind, 1);
  assert.equal(app.camera.x, 0, 'Choosing a different location returns to its composed view.');
  app.click('camera-toggle');
  app.get('stage').dispatchEvent({ type: 'ocean:rest' });
  assert.equal(app.get('stage').dataset.oceanCameraMode, 'relax');
  assert.equal(app.get('stage').dataset.oceanPaused, 'true');
  app.get('stage').dispatchEvent({ type: 'ocean:resume' });
  assert.equal(app.get('stage').dataset.oceanPaused, 'false');
  assert.equal(app.get('stage').dataset.oceanCameraMode, 'relax');
  app.dispose();
  assert.equal(app.queue.size, 0);
  assert.equal([...app.window.listeners.values()].every(listeners => listeners.size === 0), true);
  assert.equal(app.get('stage').listeners.get('ocean:rest').size, 0);
}

{
  const app = harness({ webgl: false, reduced: true, stored: { scene: 'cove', mood: 'dusk', wind: 3.2 }, search: '?scene=cove' });
  assert.equal(app.get('stage').dataset.oceanRenderer, 'canvas');
  assert.equal(app.get('stage').dataset.oceanScene, 'ocean', 'The compatibility renderer must not claim to render an unsupported cove.');
  assert.equal(app.get('scene').value, 'ocean');
  assert.equal(app.get('cove-option').disabled, true);
  assert.equal(app.get('wind').value, '3.2', 'Disabling the cove must preserve unrelated preferences.');
  assert.equal(app.selectedPreset(), 'dusk');
  app.input('scene', 'cove', 'change');
  assert.equal(app.get('stage').dataset.oceanScene, 'ocean');
  app.dispose();
}

for (const options of [{ stored: '{broken' }, { blockedStorage: true }, { stored: { wind: 'oops', waves: Infinity, mood: 'invalid' }, search: '?wind=NaN&brightness=&scene=unknown' }]) {
  const app = harness(options);
  assert.equal(app.get('wind').value, '2.4');
  assert.equal(app.get('light').value, '100');
  assert.equal(app.get('stage').dataset.oceanScene, 'ocean');
  app.input('wind', 3);
  app.flush();
  app.dispose();
}

const qualityTiers = [
  { name: 'low', shader: 0, budget: 720000, smallDimensions: [640, 480] },
  { name: 'medium', shader: 1, budget: 1500000, smallDimensions: [800, 600] },
  { name: 'high', shader: 2, budget: 4000000, smallDimensions: [1600, 1200] },
  { name: 'ultra', shader: 3, budget: 8300000, smallDimensions: [2000, 1500] },
];
const pixelCount = dimensions => dimensions[0] * dimensions[1];
const cameraPose = app => ['x', 'z', 'height', 'yaw', 'pitch'].map(axis => app.camera[axis]);

for (const { stored, search, expected } of [
  { stored: 'battery', search: '', expected: 'low' },
  { stored: 'quality', search: '', expected: 'high' },
  { stored: 'ultra', search: '?quality=battery', expected: 'low' },
  { stored: 'medium', search: '?quality=quality', expected: 'high' },
  { stored: 'battery', search: '?quality=ultra', expected: 'ultra' },
  { stored: 'medium', search: '?quality=invalid', expected: 'medium' },
]) {
  const app = harness({ stored: { quality: stored, mood: 'dusk', scene: 'cove', wind: 3.7, brightness: .8 }, search });
  assert.equal(app.get('quality').value, expected, `Quality must migrate legacy values and honor a valid explicit URL (${stored}, ${search}).`);
  assert.equal(app.get('wind').value, '3.7');
  assert.equal(app.get('stage').dataset.oceanScene, 'cove');
  assert.equal(app.selectedPreset(), 'dusk');
  app.dispose();
  assert.equal(app.preferences().quality, expected, 'Persisted quality must use the canonical tier after migration.');
  assert.equal(app.preferences().brightness, .8, 'Quality migration must preserve unrelated saved preferences.');
}

{
  const app = harness({ stored: { quality: 'auto' }, search: '?cx=14&cz=-6&alt=4&yaw=20&pitch=-8', dpr: 2,
    viewport: { width: 800, height: 600 } });
  const originalPose = cameraPose(app);
  assert.equal(app.contextRequests.find(request => request.type === 'webgl').options.powerPreference, 'high-performance',
    'WebGL must be allowed to use the graphics processor appropriate for the higher quality tiers.');
  assert.equal(app.uniforms.renderQuality, 2, 'Auto must begin at the high-detail shader tier.');
  for (const tier of qualityTiers) {
    app.input('quality', tier.name, 'change');
    assert.equal(app.get('quality').value, tier.name);
    assert.equal(app.uniforms.renderQuality, tier.shader, `${tier.name} must select the corresponding shader detail tier immediately.`);
    assert.deepEqual(app.dimensions(), tier.smallDimensions, `${tier.name} must honor its device-pixel ratio and supersampling settings.`);
    assert.deepEqual(cameraPose(app), originalPose, 'Quality changes must preserve the exploration viewpoint.');
    app.flush();
    assert.equal(app.preferences().quality, tier.name);
    assert.equal(new URL(app.window.location.href).searchParams.get('quality'), tier.name);
  }
  app.dispose();
}

{
  const sizes = [];
  for (const tier of qualityTiers) {
    const app = harness({ search: `?quality=${tier.name}`, viewport: { width: 3840, height: 2160 }, dpr: 2 });
    const dimensions = app.dimensions();
    const pixels = pixelCount(dimensions);
    sizes.push(pixels);
    assert.ok(pixels <= tier.budget * 1.002, `${tier.name} must respect its pixel budget, allowing only integer dimension rounding.`);
    assert.ok(pixels >= tier.budget * .97, `${tier.name} should use its available pixel budget at a large viewport.`);
    app.step(30, 100);
    assert.deepEqual(app.dimensions(), dimensions, `Manual ${tier.name} quality must not silently lower resolution during slow frames.`);
    assert.equal(app.uniforms.renderQuality, tier.shader, `Manual ${tier.name} must retain shader detail during slow frames.`);
    app.dispose();
  }
  assert.ok(sizes.every((pixels, index) => index === 0 || pixels > sizes[index - 1] * 1.5),
    'The four manual tiers must provide materially distinct render resolutions.');
}

{
  const app = harness({ search: '?quality=auto', viewport: { width: 1920, height: 1080 }, dpr: 2 });
  const initialPixels = pixelCount(app.dimensions());
  const originalPose = cameraPose(app);
  app.step(30, 100);
  const reducedPixels = pixelCount(app.dimensions());
  assert.ok(reducedPixels < initialPixels * .9, 'Auto must reduce resolution after sustained slow frames.');
  app.step(120);
  const recoveredPixels = pixelCount(app.dimensions());
  assert.ok(recoveredPixels > reducedPixels * 1.1, 'Auto must recover resolution after sustained smooth frames.');
  assert.ok(recoveredPixels <= initialPixels * 1.002, 'Auto recovery must stay within its original rendering budget.');
  assert.deepEqual(cameraPose(app), originalPose, 'Automatic quality adjustment must preserve the camera.');
  assert.equal(app.get('quality').value, 'auto');
  app.step(30, 100);
  app.input('quality', 'high', 'change');
  const high = harness({ search: '?quality=high', viewport: { width: 1920, height: 1080 }, dpr: 2 });
  assert.deepEqual(app.dimensions(), high.dimensions(), 'Selecting a manual tier must clear any reduced scale left by Auto.');
  assert.equal(app.uniforms.renderQuality, 2);
  high.dispose();
  app.dispose();
}

for (const maxViewport of [[1024, 512], [640, 4096]]) {
  const app = harness({ search: '?quality=ultra', viewport: { width: 3840, height: 2160 }, dpr: 2, maxViewport });
  const [width, height] = app.dimensions();
  assert.ok(width <= maxViewport[0] && height <= maxViewport[1], 'Ultra must respect both reported hardware viewport limits.');
  assert.ok(Math.abs(width / height - 3840 / 2160) < .005, 'Applying a hardware limit must preserve the view aspect ratio within pixel rounding.');
  assert.equal(app.uniforms.renderQuality, 3, 'A hardware resolution cap must preserve the selected shader detail tier.');
  app.dispose();
}

{
  const app = harness({ search: '?quality=ultra', viewport: { width: 3840, height: 2160 }, dpr: 2,
    maxViewport: [8192, 8192], maxRenderbuffer: 1024 });
  const [width, height] = app.dimensions();
  assert.ok(width <= 1024 && height <= 1024, 'The renderbuffer limit must constrain dimensions even when viewport limits allow larger buffers.');
  assert.equal(width, 1024);
  assert.equal(app.uniforms.renderQuality, 3);
  app.dispose();
}

{
  const app = harness({ reduced: true, search: '?quality=high', viewport: { width: 1200, height: 800 }, dpr: 2,
    drawingBuffer: [960, 640] });
  assert.deepEqual(app.dimensions(), [2400, 1600], 'The renderer should still request the selected quality dimensions.');
  assert.deepEqual(app.uniforms.resolution, [960, 640], 'Shader coordinates must use the drawing buffer actually allocated by the driver.');
  assert.deepEqual(app.gpuViewports.at(-1), [0, 0, 960, 640]);
  assert.equal(app.get('stage').dataset.oceanResolution, '960x640');
  app.dispose();
}

for (const failAtStartup of [true, false]) {
  const app = harness({ search: '?quality=high', drawingBuffer: failAtStartup ? [0, 0] : null });
  if (!failAtStartup) {
    app.setDrawingBuffer([0, 0]);
    app.input('quality', 'ultra', 'change');
  }
  const drawsBefore = app.drawCount;
  app.step(2);
  assert.equal(app.drawCount, drawsBefore, 'A zero-size drawing buffer must not issue render calls.');
  assert.equal(app.queue.size, 0, 'A zero-size drawing buffer must stop the animation loop.');
  assert.equal(app.get('toggle').disabled, true);
  assert.match(app.get('status').textContent, /lower quality/i);
  app.setDrawingBuffer(null);
  app.input('quality', 'low', 'change');
  assert.equal(app.get('toggle').disabled, false, 'Choosing a supported lower quality must recover the renderer.');
  assert.equal(app.get('stage').classList.contains('has-buffer-error'), false);
  assert.ok(app.drawCount > drawsBefore, 'Recovered buffers must render an updated frame.');
  assert.ok(app.queue.size > 0, `Recovered running scenes must restart animation, including initial allocation failure (${failAtStartup}).`);
  app.dispose();
}

{
  const app = harness({ search: '?quality=auto', viewport: { width: 1920, height: 1080 }, dpr: 2 });
  const initialPixels = pixelCount(app.dimensions());
  app.step(20, 50);
  assert.ok(pixelCount(app.dimensions()) < initialPixels * .9, 'Auto must react to sustained 20 FPS rather than waiting for a severe slowdown.');
  app.dispose();
}

{
  const app = harness({ reduced: true, search: '?quality=medium', viewport: { width: 800, height: 600 }, dpr: 1 });
  const dimensions = app.dimensions();
  const initialDraws = app.drawCount;
  assert.equal(app.uniforms.renderQuality, 1);
  app.input('quality', 'high', 'change');
  assert.deepEqual(app.dimensions(), dimensions, 'Medium and High have identical canvas dimensions at this viewport and DPR.');
  assert.equal(app.uniforms.renderQuality, 2, 'A paused quality selection must update shader detail even when canvas dimensions are unchanged.');
  assert.ok(app.drawCount > initialDraws, 'A paused quality selection must draw the updated still immediately.');
  assert.equal(app.queue.size, 0, 'Updating paused quality must not resume the animation.');
  assert.equal(app.get('stage').dataset.oceanPaused, 'true');
  app.dispose();
}

{
  const app = harness({ stored: { swell: 'long', mood: 'dusk', wind: 3.7, waves: .85 },
    search: '?cx=14&cz=-6&alt=4&yaw=20&pitch=-8' });
  const originalPose = cameraPose(app);
  assert.equal(app.get('swell').value, 'long', 'Swell style must restore independently from saved lighting and wind.');
  assert.equal(app.spectrumUpdates.at(-1)[3], 'long', 'The restored style must reach the wave spectrum.');
  for (const style of ['chop', 'balanced', 'long']) {
    app.input('swell', style, 'change');
    assert.equal(app.get('stage').dataset.oceanSwell, style);
    assert.equal(app.spectrumUpdates.at(-1)[3], style, 'Each selection must update the spectrum immediately.');
    assert.deepEqual(cameraPose(app), originalPose, 'Changing swell style must preserve the exploration viewpoint.');
    assert.equal(app.get('wind').value, '3.7');
    assert.equal(app.get('height').value, '0.85');
    assert.equal(app.selectedPreset(), 'dusk');
    app.flush();
    assert.equal(app.preferences().swell, style);
    assert.equal(new URL(app.window.location.href).searchParams.get('swell'), style);
  }
  const shared = harness({ stored: { swell: 'balanced', mood: 'dawn', wind: 8 }, search: new URL(app.window.location.href).search });
  assert.equal(shared.get('swell').value, 'long', 'Explicit shared swell must override the local preference.');
  assert.deepEqual(cameraPose(shared), originalPose, 'A complete scene link must reproduce the canonical viewpoint.');
  assert.equal(shared.get('wind').value, '3.7');
  assert.equal(shared.selectedPreset(), 'dusk');
  app.input('swell', 'invalid', 'change');
  assert.equal(app.get('swell').value, 'balanced', 'An invalid selection must recover to natural swell.');
  shared.dispose();
  app.dispose();
}

{
  const app = harness({ search: '?alt=4&yaw=20&pitch=-8' });
  const canonical = cameraPose(app);
  assert.equal(app.get('float').checked, false);
  assert.equal(app.get('stage').dataset.oceanFloating, 'false', 'A new scene must keep its viewpoint stationary by default.');
  app.step(2);
  assert.equal(app.uniforms.cameraPosition[1], app.camera.height);
  assert.equal(app.uniforms.cameraAngle[1], app.camera.pitch);
  app.check('float', true);
  app.flush();
  assert.equal(app.get('stage').dataset.oceanFloating, 'true');
  assert.equal(app.preferences().floating, true);
  assert.equal(new URL(app.window.location.href).searchParams.get('float'), '1');
  let maximumRise = 0;
  let maximumPitch = 0;
  for (let frame = 0; frame < 600; frame++) {
    app.frame(1000 / 30);
    maximumRise = Math.max(maximumRise, Math.abs(app.uniforms.cameraPosition[1] - app.camera.height));
    maximumPitch = Math.max(maximumPitch, Math.abs(app.uniforms.cameraAngle[1] - app.camera.pitch));
  }
  assert.ok(maximumRise > .015 && maximumRise <= .065, 'The checked control must produce gentle rise and fall in actual render uniforms.');
  assert.ok(maximumPitch > .0003 && maximumPitch <= .001625, 'The checked control must produce a subtle rendered pitch.');
  assert.deepEqual(cameraPose(app), canonical, 'Rendered floating must never alter the canonical camera.');
  app.input('swell', 'long', 'change');
  app.flush();
  const sharedUrl = new URL(app.window.location.href);
  assert.equal(sharedUrl.searchParams.get('alt'), '4.00', 'Sharing during floating must preserve the user-selected height.');
  assert.equal(sharedUrl.searchParams.get('yaw'), '20');
  assert.equal(sharedUrl.searchParams.get('pitch'), '-8');
  const shared = harness({ search: sharedUrl.search });
  assert.equal(shared.get('float').checked, true, 'A shared scene may explicitly opt into floating.');
  assert.deepEqual(cameraPose(shared), canonical);
  const stored = harness({ stored: { floating: true } });
  assert.equal(stored.get('float').checked, true, 'A saved opt in must restore on a motion-capable device.');
  app.click('reset');
  app.flush();
  assert.equal(app.get('float').checked, false, 'Reset scene must disable floating.');
  assert.equal(app.get('swell').value, 'balanced');
  assert.equal(app.preferences().floating, false);
  assert.equal(new URL(app.window.location.href).searchParams.get('float'), '0');
  stored.dispose();
  shared.dispose();
  app.dispose();
}

{
  const app = harness({ stored: { floating: true }, search: '?float=1', reduced: true });
  assert.equal(app.get('float').checked, false, 'Reduced motion must override both saved and shared floating settings.');
  assert.equal(app.get('float').disabled, true);
  assert.match(app.get('float').title, /reduced motion/i);
  assert.equal(app.get('stage').dataset.oceanFloating, 'false');
  assert.equal(app.queue.size, 0);
  assert.equal(app.uniforms.cameraPosition[1], app.camera.height);
  app.click('toggle');
  app.step(5);
  assert.equal(app.uniforms.cameraPosition[1], app.camera.height, 'Explicitly playing waves must still honor reduced camera motion.');
  app.dispose();
}

{
  const app = harness({ stored: { floating: true } });
  app.step(3);
  assert.ok(Math.abs(app.uniforms.cameraPosition[1] - app.camera.height) > .01);
  app.click('toggle');
  const pausedTime = app.uniforms.time;
  const pausedPosition = [...app.uniforms.cameraPosition];
  app.step(10);
  assert.equal(app.uniforms.time, pausedTime);
  assert.deepEqual(app.uniforms.cameraPosition, pausedPosition, 'Pause must also freeze the rendered floating pose.');
  app.reduceMotion(true);
  assert.equal(app.get('float').disabled, true);
  assert.equal(app.get('float').checked, false);
  assert.equal(app.get('stage').dataset.oceanFloating, 'false');
  assert.equal(app.uniforms.cameraPosition[1], app.camera.height, 'Live reduced motion must redraw the canonical view even while paused.');
  assert.equal(app.uniforms.cameraAngle[1], app.camera.pitch);
  assert.equal(app.queue.size, 0);
  app.flush();
  assert.equal(app.preferences().floating, false);
  app.reduceMotion(false);
  assert.equal(app.get('float').disabled, false);
  assert.equal(app.get('float').checked, false, 'Ending reduced motion must leave camera movement opted out.');
  app.dispose();
  assert.equal([...app.motionPreference.listeners.values()].every(listeners => listeners.size === 0), true,
    'Scene cleanup must remove both floating and UI media-query listeners.');
}

for (const interruption of ['hidden', 'paused', 'offscreen']) {
  const app = harness({ search: '?quality=medium' });
  app.frame(100);
  const start = app.uniforms.time;
  app.step(10, 100);
  assert.ok(Math.abs(app.uniforms.time - start - 10) < .000001, 'Sustained 10 FPS must advance ten simulation seconds in ten real seconds.');
  assert.ok(Math.abs(app.spectrumUpdates.at(-1)[0] - app.uniforms.time) < .000001, 'The spectrum and shading must receive the same elapsed sea time.');
  if (interruption === 'hidden') { app.document.hidden = true; app.document.dispatchEvent({ type: 'visibilitychange' }); }
  if (interruption === 'paused') app.click('toggle');
  if (interruption === 'offscreen') app.visible(false);
  const suspendedAt = app.uniforms.time;
  const draws = app.drawCount;
  app.frame(60000);
  assert.equal(app.uniforms.time, suspendedAt, `${interruption} must not advance sea time.`);
  assert.equal(app.drawCount, draws, `${interruption} must release the renderer animation loop.`);
  if (interruption === 'hidden') { app.document.hidden = false; app.document.dispatchEvent({ type: 'visibilitychange' }); }
  if (interruption === 'paused') app.click('toggle');
  if (interruption === 'offscreen') app.visible(true);
  app.frame(100);
  assert.equal(app.uniforms.time, suspendedAt, `Resuming from ${interruption} must not catch up the suspended minute.`);
  app.frame(100);
  assert.ok(Math.abs(app.uniforms.time - suspendedAt - .1) < .000001, 'Normal sea time must resume after establishing the new frame clock.');
  app.dispose();
}

{
  const app = harness({ search: '?quality=auto', viewport: { width: 1920, height: 1080 }, dpr: 2 });
  const initialDimensions = app.dimensions();
  const originalPose = cameraPose(app);
  app.step(14, 100);
  const floorDimensions = app.dimensions();
  assert.equal(app.uniforms.renderQuality, 2, 'Auto must exhaust its resolution adjustment before lowering reflection detail.');
  assert.ok(pixelCount(floorDimensions) < pixelCount(initialDimensions) * .51);
  app.step(12, 100);
  assert.deepEqual(app.dimensions(), floorDimensions, 'Auto must keep its established resolution floor when lowering shader detail.');
  assert.equal(app.uniforms.renderQuality, 0, 'Sustained slow frames at the resolution floor must reach the lowest reflection tier.');
  assert.equal(app.get('stage').dataset.oceanDetail, '0');
  app.step(18);
  assert.equal(app.uniforms.renderQuality, 2, 'Smooth frames must restore reflection detail before increasing resolution.');
  assert.deepEqual(app.dimensions(), floorDimensions);
  app.step(30);
  assert.deepEqual(app.dimensions(), initialDimensions, 'Continued smooth rendering must recover the original Auto pixel budget.');
  assert.deepEqual(cameraPose(app), originalPose);
  assert.equal(app.get('quality').value, 'auto', 'Internal adaptation must not rewrite the selected quality preference.');
  app.dispose();
}

{
  const app = harness();
  assert.deepEqual(app.conditions.at(-1), { wind: 2.4, waveHeight: .65, shore: 0 }, 'Initial sound conditions must describe the visible open ocean.');
  app.input('wind', 7.1);
  assert.equal(app.conditions.at(-1).wind, 7.1);
  app.input('height', 1.25);
  assert.equal(app.conditions.at(-1).waveHeight, 1.25);
  app.input('scene', 'cove', 'change');
  const cove = app.conditions.at(-1);
  assert.equal(cove.wind, 7.1);
  assert.equal(cove.waveHeight, 1.25);
  assert.ok(cove.shore > 0 && cove.shore < 1, 'Moving into the cove must expose its shoreline proximity to audio.');
  assert.equal(Number(app.get('stage').dataset.oceanShore), cove.shore);
  const eventCount = app.conditions.length;
  app.click('camera-toggle');
  app.get('stage').dispatchEvent({ type: 'keydown', code: 'KeyA', key: 'a' });
  app.step(3);
  app.window.dispatchEvent({ type: 'keyup', code: 'KeyA' });
  app.step(3);
  const approaching = app.conditions.at(-1);
  assert.ok(app.conditions.length > eventCount, 'Exploration must update sound conditions as the camera approaches shore.');
  assert.ok(approaching.shore > cove.shore + .04);
  assert.ok(approaching.shore <= 1 && approaching.shore >= 0);
  assert.ok(Math.abs(Number(app.get('stage').dataset.oceanShore) - approaching.shore) < .02,
    'Throttled condition events must stay close to the current shoreline proximity.');
  const stationaryEvents = app.conditions.length;
  app.step(10);
  assert.equal(app.conditions.length, stationaryEvents, 'A stationary scene must not emit redundant condition events each frame.');
  app.input('scene', 'ocean', 'change');
  assert.equal(app.conditions.at(-1).shore, 0);
  assert.equal(app.conditions.at(-1).wind, 7.1);
  app.dispose();
}

console.log('Ocean preferences, swell, floating, real-time clocks, quality adaptation, condition events, and scene lifecycle checks passed.');
