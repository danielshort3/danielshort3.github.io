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
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getBoundingClientRect() { return this.rect; }
  focus() { this.focused = true; }
  setPointerCapture(id) { this.capture = id; }
  hasPointerCapture(id) { return this.capture === id; }
  releasePointerCapture() { this.capture = null; }
  closest(selector) { return selector === '[data-comparison-selection]' && this.isSelection ? this : null; }
  decode() { return Promise.resolve(); }
}

function harness({ failing = false, delayed = false } = {}) {
  const comparison = new Element({ comparisonId: 'sheetMusicUpscale', comparisonLeft: '33', comparisonRight: '67', comparisonMinimumGap: '10', comparisonPageRatio: String(612 / 792), comparisonCrop: JSON.stringify({ left: 240 / 612 * 100, top: 552 / 792 * 100, width: 320 / 612 * 100, height: 180 / 792 * 100 }) });
  const nodes = {};
  for (const name of ['overview', 'selection', 'controls', 'viewport']) nodes[name] = new Element();
  for (const name of ['controls', 'zoom', 'zoom-value', 'reset', 'status', 'retry']) nodes[`selection-${name}`] = new Element();
  nodes.selection.isSelection = true;
  nodes.viewport.rect = { left: 0, top: 0, width: 600, height: 337.5 };
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
  let nextFrame = 0;
  const window = {
    addEventListener() {}, cancelAnimationFrame: (id) => frames.delete(id),
    requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; }
  };
  const document = { readyState: 'loading', addEventListener() {} };
  class Image {
    set src(value) { this.source = value; this.naturalWidth = value.includes('upscaled') ? 1700 : 612; this.naturalHeight = value.includes('upscaled') ? 2200 : 792; }
    get src() { return this.source; }
    decode() { return delayed ? new Promise((resolve, reject) => decoders.push({ resolve, reject })) : failing ? Promise.reject(new Error('offline')) : Promise.resolve(); }
  }
  vm.runInNewContext(source, { window, document, Image, AbortController, console });
  return { comparison, nodes, dividers, images, root, mount: () => window.ProjectImageComparisons.mount(root), flush() { for (const callback of frames.values()) callback(); frames.clear(); }, resolve() { delayed = false; decoders.forEach(({ resolve }) => resolve()); }, recover() { failing = false; } };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));
const percent = (h, name) => parseFloat(h.comparison.properties.get(`--comparison-${name}`));
const close = (actual, expected) => assert(Math.abs(actual - expected) < 0.00001, `${actual} should equal ${expected}`);

async function main() {
  const html = renderProjectPage(project);
  assert(html.includes('data-comparison-selection') && html.includes('data-selection-zoom'));
  assert.strictEqual((html.match(/data-full-stage/g) || []).length, 1, 'one original sheet is the region selector');
  assert.strictEqual((html.match(/data-comparison-source=/g) || []).length, 3, 'all native stages participate');
  assert(html.includes('https://www.praisecharts.com/23870'), 'source attribution remains visible');
  assert.deepStrictEqual(project.previewComparison.stages.map((stage) => [stage.fullWidth, stage.fullHeight]), [[612, 792], [612, 792], [1700, 2200]]);

  const h = harness();
  const dispose = h.mount();
  await settle();
  assert(h.comparison.classList.contains('is-selectable'));
  assert.strictEqual(h.nodes.selection.disabled, false);
  close(percent(h, 'crop-left') / 100 * 612, 240);
  close(percent(h, 'crop-top') / 100 * 792, 552);
  close(percent(h, 'crop-width') / 100 * 612, 320);
  close(percent(h, 'crop-height') / 100 * 792, 180);
  h.nodes['selection-zoom'].value = '6';
  h.nodes['selection-zoom'].emit('input');
  h.nodes.overview.emit('click', { clientX: 900, clientY: -50 });
  close(percent(h, 'crop-left') + percent(h, 'crop-width'), 100);
  close(percent(h, 'crop-top'), 0);
  close(percent(h, 'crop-width') * 612 / (percent(h, 'crop-height') * 792), 16 / 9);
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
  close(percent(h, 'crop-top') / 100 * 792, 552);
  assert.strictEqual(h.comparison.dataset.comparisonLeft, rememberedLeft, 'selection reset preserves comparison dividers');

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
