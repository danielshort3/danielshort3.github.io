'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { renderProjectPage } = require('../../build/generate-project-pages');
const project = require('../../content/projects/sheetMusicUpscale.json');
const source = fs.readFileSync(path.join(__dirname, '../../js/portfolio/project-image-comparison.js'), 'utf8');

class Element {
  constructor(dataset = {}) {
    this.dataset = dataset;
    this.listeners = new Map();
    this.attributes = new Map();
    this.queries = new Map();
    this.classes = new Set();
    this.classList = { add: (value) => this.classes.add(value), remove: (value) => this.classes.delete(value), contains: (value) => this.classes.has(value) };
    this.properties = new Map();
    this.style = { setProperty: (key, value) => this.properties.set(key, value) };
    this.rect = { left: 0, top: 0, width: 612, height: 792 };
    this.hidden = false;
    this.disabled = false;
    this.capture = null;
  }
  querySelector(selector) { return this.queries.get(selector) || null; }
  querySelectorAll(selector) { return this.queries.get(selector) || []; }
  addEventListener(type, handler, options) {
    const listeners = this.listeners.get(type) || [];
    listeners.push({ handler, signal: options?.signal });
    this.listeners.set(type, listeners);
  }
  emit(type, fields = {}) {
    const event = { type, button: 0, target: this, pointerId: 1, pointerType: 'mouse', clientX: 0, clientY: 0, preventDefault() { this.defaultPrevented = true; }, ...fields };
    for (const { handler, signal } of this.listeners.get(type) || []) if (!signal?.aborted) handler(event);
    return event;
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getBoundingClientRect() { return this.rect; }
  focus() { this.focused = true; }
  setPointerCapture(id) { this.capture = id; }
  hasPointerCapture(id) { return this.capture === id; }
  releasePointerCapture() { this.capture = null; }
  closest(selector) { return (selector === '[data-comparison-selection]' && this.isSelection) || (selector === '[data-comparison-divider]' && this.dataset.comparisonDivider) ? this : null; }
  decode() { return Promise.resolve(); }
}

function harness({ failing = false, delayed = false, width = 600, gap = 10, initialLeft = 33, initialRight = 67, aspect = 4 / 3 } = {}) {
  const comparison = new Element({ comparisonId: 'sheetMusicUpscale', comparisonLeft: String(initialLeft), comparisonRight: String(initialRight), comparisonDefaultLeft: String(initialLeft), comparisonDefaultRight: String(initialRight), comparisonMinimumGap: String(gap), comparisonSelectionAspect: String(aspect), comparisonPageRatio: String(612 / 792), comparisonCrop: JSON.stringify({ left: 240 / 612 * 100, top: 552 / 792 * 100, width: 320 / 612 * 100, height: 180 / 792 * 100 }) });
  const nodes = {};
  for (const name of ['overview', 'selection', 'controls', 'viewport']) nodes[name] = new Element();
  for (const name of ['controls', 'zoom', 'zoom-value', 'reset', 'status', 'retry']) nodes[`selection-${name}`] = new Element();
  nodes.selection.isSelection = true;
  nodes.viewport.rect = { left: 0, top: 0, width, height: width / aspect };
  for (const [name, node] of Object.entries(nodes)) comparison.queries.set(`[data-${name.startsWith('selection-') ? name : `comparison-${name}`}]`, node);
  const images = project.previewComparison.stages.map((stage) => Object.assign(new Element({ comparisonSource: stage.fullImage }), { src: stage.image }));
  const slides = images.map((image) => { const slide = new Element(); slide.queries.set('img', image); return slide; });
  const dividers = ['left', 'right'].map((side) => new Element({ comparisonDivider: side }));
  comparison.queries.set('[data-stage-slide]', slides);
  comparison.queries.set('[data-comparison-divider]', dividers);
  const root = new Element();
  root.queries.set('[data-project-image-comparison]', [comparison]);
  const frames = new Map();
  const decoders = [];
  const observers = [];
  let nextFrame = 0;
  const window = {
    addEventListener() {}, cancelAnimationFrame: (id) => frames.delete(id),
    requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe() {}
      disconnect() { this.disconnected = true; }
    }
  };
  const document = { readyState: 'loading', addEventListener() {} };
  class Image {
    set src(value) { this.source = value; this.naturalWidth = value.includes('upscaled') ? 1700 : 612; this.naturalHeight = value.includes('upscaled') ? 2200 : 792; }
    get src() { return this.source; }
    decode() { return delayed ? new Promise((resolve, reject) => decoders.push({ resolve, reject })) : failing ? Promise.reject(new Error('offline')) : Promise.resolve(); }
  }
  const execute = () => vm.runInNewContext(source, { window, document, Image, AbortController, console });
  execute();
  return { comparison, nodes, dividers, images, root, execute, mount: () => window.ProjectImageComparisons.mount(root), flush() { for (const callback of frames.values()) callback(); frames.clear(); }, pendingFrames: () => frames.size, resize(nextWidth) { nodes.viewport.rect.width = nextWidth; observers.filter((observer) => !observer.disconnected).forEach((observer) => observer.callback()); }, resolve() { delayed = false; decoders.forEach(({ resolve }) => resolve()); }, recover() { failing = false; } };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));
const percent = (h, name) => parseFloat(h.comparison.properties.get(`--comparison-${name}`));
const close = (actual, expected) => assert(Math.abs(actual - expected) < 0.00001, `${actual} should equal ${expected}`);

async function main() {
  const html = renderProjectPage(project);
  assert(html.includes('data-comparison-selection') && html.includes('data-selection-zoom'));
  assert.strictEqual((html.match(/data-full-stage/g) || []).length, 1, 'one original sheet is the region selector');
  assert.strictEqual((html.match(/data-comparison-source=/g) || []).length, 3, 'all native stages participate');
  assert(!html.includes('class="project-comparison-source"') && !html.includes('class="project-comparison-credit"'), 'the sheet preview should omit the source link and credit text');
  assert.deepStrictEqual(project.previewComparison.stages.map((stage) => [stage.fullWidth, stage.fullHeight]), [[612, 792], [612, 792], [1700, 2200]]);

  const h = harness();
  const dispose = h.mount();
  await settle();
  assert(h.comparison.classList.contains('is-selectable'));
  assert.strictEqual(h.nodes.selection.disabled, false);
  close(percent(h, 'crop-left') / 100 * 612, 240);
  close(percent(h, 'crop-top') / 100 * 792, 522);
  close(percent(h, 'crop-width') / 100 * 612, 320);
  close(percent(h, 'crop-height') / 100 * 792, 240);
  h.nodes['selection-zoom'].value = '6';
  h.nodes['selection-zoom'].emit('input');
  h.nodes.overview.emit('click', { clientX: 900, clientY: -50 });
  close(percent(h, 'crop-left') + percent(h, 'crop-width'), 100);
  close(percent(h, 'crop-top'), 0);
  close(percent(h, 'crop-width') * 612 / (percent(h, 'crop-height') * 792), 4 / 3);
  const prior = percent(h, 'crop-left');
  h.nodes.selection.emit('keydown', { key: 'ArrowLeft' });
  close(percent(h, 'crop-left'), prior - 1);
  h.nodes.selection.emit('pointerdown', { clientX: 500, clientY: 60 });
  h.nodes.selection.emit('pointermove', { clientX: 350, clientY: 180 });
  h.flush();
  h.nodes.selection.emit('pointercancel');
  h.nodes.overview.emit('click', { clientX: 200, clientY: 400 });
  close(percent(h, 'crop-center-x'), 200 / 612 * 100);
  close(percent(h, 'crop-center-y'), 400 / 792 * 100);
  h.dividers[0].emit('keydown', { key: 'ArrowRight' });
  const rememberedLeft = h.comparison.dataset.comparisonLeft;
  const rememberedRegion = percent(h, 'crop-left');
  dispose();
  assert(h.comparison.inert, 'detached or outgoing controls become inert');
  assert(h.comparison.classList.contains('is-enhanced') && h.comparison.classList.contains('is-selectable'), 'cleanup preserves painted dimensions');
  h.nodes.overview.emit('click', { clientX: 0, clientY: 0 });
  close(percent(h, 'crop-left'), rememberedRegion);
  h.mount();
  await settle();
  assert.strictEqual(h.comparison.inert, false);
  assert.strictEqual(h.nodes.selection.disabled, false);
  close(percent(h, 'crop-left'), rememberedRegion);
  assert.strictEqual(h.comparison.dataset.comparisonLeft, rememberedLeft);
  dispose();
  assert.strictEqual(h.comparison.inert, false, 'old disposal cannot disable a newer mount');
  h.nodes['selection-reset'].emit('click');
  close(percent(h, 'crop-left') / 100 * 612, 240);
  close(percent(h, 'crop-top') / 100 * 792, 522);
  close(percent(h, 'crop-height') / 100 * 792, 240);
  assert.strictEqual(h.comparison.dataset.comparisonLeft, '33.00', 'reset restores canonical left divider after saved-state remount');
  assert.strictEqual(h.comparison.dataset.comparisonRight, '67.00', 'reset restores canonical right divider after saved-state remount');

  // Mouse drags begin anywhere in the preview and retain the initially nearest handle.
  h.nodes.viewport.emit('pointerdown', { clientX: 120 });
  close(percent(h, 'left'), 20);
  assert.strictEqual(h.nodes.viewport.capture, 1);
  h.nodes.viewport.emit('pointermove', { clientX: 330 });
  h.flush();
  close(percent(h, 'left'), 55);
  h.nodes.viewport.emit('pointermove', { clientX: 600 });
  h.nodes.viewport.emit('pointerup', { clientX: 600 });
  close(percent(h, 'left'), 80);
  close(percent(h, 'right'), 90);
  h.nodes.viewport.emit('click', { clientX: 600 });
  close(percent(h, 'left'), 80);
  close(percent(h, 'right'), 90);
  assert.strictEqual(h.nodes.viewport.capture, null);
  h.nodes['selection-reset'].emit('click');
  h.nodes.viewport.emit('pointerdown', { clientX: 450 });
  close(percent(h, 'right'), 75);
  h.nodes.viewport.emit('pointermove', { clientX: 0 });
  h.flush();
  close(percent(h, 'left'), 10);
  close(percent(h, 'right'), 20);
  h.nodes.viewport.emit('pointerup');

  // Reset discards both queued render frames and captures; late gesture events cannot undo it.
  h.nodes['selection-zoom'].value = '6';
  h.nodes['selection-zoom'].emit('input');
  h.nodes.selection.emit('pointerdown', { pointerId: 2, clientX: 100, clientY: 100 });
  h.nodes.selection.emit('pointermove', { pointerId: 2, clientX: 200, clientY: 300 });
  h.nodes.viewport.emit('pointerdown', { pointerId: 3, clientX: 350 });
  h.nodes.viewport.emit('pointermove', { pointerId: 3, clientX: 500 });
  assert(h.pendingFrames() > 0);
  h.nodes['selection-reset'].emit('click');
  assert.strictEqual(h.pendingFrames(), 0);
  assert.strictEqual(h.nodes.selection.capture, null);
  assert.strictEqual(h.nodes.viewport.capture, null);
  h.nodes.selection.emit('pointerup', { pointerId: 2 });
  h.nodes.viewport.emit('pointermove', { pointerId: 3, clientX: 600 });
  h.nodes.viewport.emit('pointerup', { pointerId: 3, clientX: 600 });
  h.nodes.viewport.emit('click', { pointerId: 3, clientX: 600 });
  h.flush();
  close(percent(h, 'left'), 33);
  close(percent(h, 'right'), 67);
  close(percent(h, 'crop-left') / 100 * 612, 240);
  close(percent(h, 'crop-top') / 100 * 792, 522);
  close(Number(h.nodes['selection-zoom'].value), 612 / 320);

  // Touching the image permits scrolling, while a tap still selects a divider.
  const touchDown = h.nodes.viewport.emit('pointerdown', { pointerType: 'touch', clientX: 240 });
  assert(!touchDown.defaultPrevented);
  h.nodes.viewport.emit('pointermove', { pointerType: 'touch', clientX: 300 });
  h.nodes.viewport.emit('pointercancel', { pointerType: 'touch' });
  close(percent(h, 'left'), 33);
  assert.strictEqual(h.nodes.viewport.capture, null);
  h.nodes.viewport.emit('pointerdown', { pointerType: 'touch', clientX: 240 });
  h.nodes.viewport.emit('pointerup', { pointerType: 'touch', clientX: 240 });
  h.nodes.viewport.emit('click', { pointerType: 'touch', clientX: 240 });
  close(percent(h, 'left'), 40);
  h.dividers[1].emit('pointerdown', { pointerType: 'touch', clientX: 420 });
  close(percent(h, 'right'), 70);
  h.dividers[1].emit('pointermove', { pointerType: 'touch', clientX: 480 });
  h.dividers[1].emit('pointerup', { pointerType: 'touch', clientX: 480 });
  close(percent(h, 'right'), 80);
  assert.strictEqual(h.dividers[1].capture, null);

  for (const type of ['pointercancel', 'lostpointercapture']) {
    h.nodes['selection-reset'].emit('click');
    h.nodes.viewport.emit('pointerdown', { clientX: 240 });
    h.nodes.viewport.emit('pointermove', { clientX: 500 });
    assert(h.pendingFrames() > 0);
    h.nodes.viewport.emit(type);
    assert.strictEqual(h.pendingFrames(), 0, `${type} discards obsolete mouse movement`);
    assert.strictEqual(h.nodes.viewport.capture, null);
    h.flush();
    close(percent(h, 'left'), 40);
    h.nodes.viewport.emit('pointerdown', { pointerType: 'touch', clientX: 120 });
    h.nodes.viewport.emit('pointerup', { pointerType: 'touch', clientX: 120 });
    h.nodes.viewport.emit('click', { pointerType: 'touch', clientX: 120 });
    close(percent(h, 'left'), 20);

    h.dividers[1].emit('pointerdown', { pointerType: 'touch', clientX: 420 });
    h.dividers[1].emit('pointermove', { pointerType: 'touch', clientX: 540 });
    h.dividers[1].emit(type, { pointerType: 'touch' });
    assert.strictEqual(h.pendingFrames(), 0, `${type} also cancels direct-handle movement`);
    assert.strictEqual(h.dividers[1].capture, null);
    h.flush();
    close(percent(h, 'right'), 70);
  }
  h.nodes['selection-reset'].emit('click');
  h.dividers[0].emit('pointerdown', { clientX: 120 });
  h.dividers[0].emit('pointermove', { clientX: 450 });
  h.nodes['selection-reset'].emit('click');
  h.dividers[0].emit('pointerup', { clientX: 450 });
  h.flush();
  close(percent(h, 'left'), 33);
  assert.strictEqual(h.dividers[0].capture, null);
  const secondaryMouse = h.nodes.viewport.emit('pointerdown', { button: 2, clientX: 300 });
  assert(!secondaryMouse.defaultPrevented);
  h.nodes.viewport.emit('pointerdown', { isPrimary: false, clientX: 300 });
  close(percent(h, 'left'), 33);
  assert.strictEqual(h.nodes.viewport.capture, null);

  // Equal minimums apply to all three panes, including responsive pointer spacing.
  for (const [width, gap] of [[600, 10], [246, 10], [100, 10], [300, 30]]) {
    const sized = harness({ width, gap });
    sized.mount();
    await settle();
    const effectiveGap = Math.max(gap, Math.min(24, 44 / width * 100));
    const checkMinimums = () => {
      const left = Number(sized.comparison.dataset.comparisonLeft);
      const right = Number(sized.comparison.dataset.comparisonRight);
      assert(left >= effectiveGap - .011 && right - left >= effectiveGap - .011 && 100 - right >= effectiveGap - .011);
    };
    sized.dividers[0].emit('keydown', { key: 'End' });
    checkMinimums();
    assert.strictEqual(sized.dividers[0].attributes.get('aria-valuemax'), String(Math.round(100 - 2 * effectiveGap)));
    assert.strictEqual(sized.dividers[1].attributes.get('aria-valuemax'), String(Math.round(100 - effectiveGap)));
    sized.dividers[1].emit('keydown', { key: 'Home' });
    checkMinimums();
    assert.strictEqual(sized.dividers[0].attributes.get('aria-valuemin'), String(Math.round(effectiveGap)));
    assert.strictEqual(sized.dividers[1].attributes.get('aria-valuemin'), String(Math.round(2 * effectiveGap)));
    sized.resize(100);
    const resizedGap = Math.max(gap, 24);
    assert(Number(sized.comparison.dataset.comparisonLeft) >= resizedGap - .011);
    assert(100 - Number(sized.comparison.dataset.comparisonRight) >= resizedGap - .011);
  }

  const custom = harness({ initialLeft: 28, initialRight: 72, aspect: 16 / 9 });
  const disposeCustom = custom.mount();
  await settle();
  close(percent(custom, 'crop-top') / 100 * 792, 552);
  close(percent(custom, 'crop-height') / 100 * 792, 180);
  custom.dividers[0].emit('keydown', { key: 'End' });
  disposeCustom();
  custom.mount();
  await settle();
  custom.nodes['selection-reset'].emit('click');
  close(percent(custom, 'left'), 28);
  close(percent(custom, 'right'), 72);

  // The route loader executes this script anew against fresh HTML on each return.
  const rerun = harness();
  let disposeRerun = rerun.mount();
  await settle();
  rerun.dividers[0].emit('keydown', { key: 'ArrowRight' });
  rerun.nodes['selection-zoom'].value = '3';
  rerun.nodes['selection-zoom'].emit('input');
  rerun.nodes.selection.emit('keydown', { key: 'ArrowLeft' });
  const restoredState = {
    left: percent(rerun, 'left'), right: percent(rerun, 'right'),
    cropLeft: percent(rerun, 'crop-left'), cropTop: percent(rerun, 'crop-top'),
    zoom: Number(rerun.nodes['selection-zoom'].value)
  };
  for (let visit = 0; visit < 3; visit += 1) {
    disposeRerun();
    // Simulate a fresh route fragment rather than reusing mutated position attributes.
    rerun.comparison.dataset.comparisonLeft = '33';
    rerun.comparison.dataset.comparisonRight = '67';
    rerun.comparison.properties.clear();
    rerun.execute();
    disposeRerun = rerun.mount();
    await settle();
    close(percent(rerun, 'left'), restoredState.left);
    close(percent(rerun, 'right'), restoredState.right);
    close(percent(rerun, 'crop-left'), restoredState.cropLeft);
    close(percent(rerun, 'crop-top'), restoredState.cropTop);
    close(Number(rerun.nodes['selection-zoom'].value), restoredState.zoom);
    // Re-evaluation while already mounted must not double-bind any interaction.
    rerun.execute();
    rerun.mount();
    rerun.dividers[0].emit('keydown', { key: 'ArrowRight' });
    close(percent(rerun, 'left'), restoredState.left + 1);
    rerun.dividers[0].emit('keydown', { key: 'ArrowLeft' });
    close(percent(rerun, 'left'), restoredState.left);
  }
  rerun.nodes['selection-reset'].emit('click');
  close(percent(rerun, 'left'), 33);
  close(percent(rerun, 'right'), 67);
  close(percent(rerun, 'crop-left') / 100 * 612, 240);
  close(percent(rerun, 'crop-top') / 100 * 792, 522);
  close(Number(rerun.nodes['selection-zoom'].value), 612 / 320);

  const offline = harness({ failing: true });
  offline.mount();
  await settle();
  assert(!offline.comparison.classList.contains('is-selectable'));
  assert(offline.images.every((image) => image.src.includes('-comparison.webp')), 'failed preparation preserves the static sample');
  assert.strictEqual(offline.nodes['selection-retry'].hidden, false);
  offline.recover();
  offline.nodes['selection-retry'].emit('click');
  await settle();
  assert(offline.comparison.classList.contains('is-selectable'));

  const delayed = harness({ delayed: true });
  const cancel = delayed.mount();
  cancel();
  delayed.resolve();
  await settle();
  assert(!delayed.comparison.classList.contains('is-selectable'), 'late decoding cannot mutate an obsolete mount');
  assert(delayed.images.every((image) => image.src.includes('-comparison.webp')));
  process.stdout.write('Project image selection: geometry, interaction, lifecycle, and failure checks passed.\n');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
