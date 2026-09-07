'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/background-remover.js'), 'utf8');
const markup = fs.readFileSync(path.join(__dirname, '../../pages/background-remover.html'), 'utf8');
const closing = source.lastIndexOf('})();');
assert(closing > 0, 'Background Remover must retain its enclosing controller.');
const instrumented = `${source.slice(0, closing)}
  globalThis.api = { state, active, addFiles, processQueue, reprocessJobs, selectJob, readProcessingSettings, updateActionButtons };
${source.slice(closing)}`;
const tick = () => new Promise((resolve) => setImmediate(resolve));
async function until(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await tick();
  }
  assert.fail(message);
}

function createHarness() {
  const nodes = [];
  const ids = new Map();
  const calls = { events: [], decoded: [], encoded: [], revoked: [], warnings: [] };
  let document;
  let sequence = 0;
  let decodeGate = null;
  let encodeGate = null;
  class Element {
    constructor(tagName = 'div', attributes = {}) {
      this.tagName = tagName.toUpperCase();
      this.attributes = attributes;
      this.id = attributes.id || '';
      this.type = attributes.type || '';
      this.name = attributes.name || '';
      this.value = attributes.value || '';
      this.checked = Object.hasOwn(attributes, 'checked');
      this.hidden = Object.hasOwn(attributes, 'hidden');
      this.disabled = false;
      this.dataset = {};
      this.style = {};
      this.listeners = new Map();
      this.classList = { add() {}, remove() {}, toggle() {} };
      for (const [name, value] of Object.entries(attributes)) {
        if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = value;
      }
    }
    addEventListener(type, handler) {
      const handlers = this.listeners.get(type) || [];
      handlers.push(handler);
      this.listeners.set(type, handlers);
    }
    dispatchEvent(event) {
      event.target ||= this;
      for (const handler of this.listeners.get(event.type) || []) handler(event);
      if (event.bubbles && this !== document && this !== ids.get('bgtool-form')) ids.get('bgtool-form')?.dispatchEvent(event);
      return true;
    }
    getAttribute(name) { return this.attributes[name] ?? null; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    querySelectorAll() { return []; }
    querySelector() { return null; }
    closest() { return null; }
    focus() {}
    click() { this.dispatchEvent({ type: 'click', preventDefault() {}, stopPropagation() {} }); }
    appendChild() {}
    remove() {}
  }
  class Canvas extends Element {
    constructor(attributes) {
      super('canvas', attributes);
      this._width = 2;
      this._height = 1;
      this.pixels = new Uint8ClampedArray(8);
      this.lastDrawnBlob = null;
      this.context = {
        canvas: this,
        globalCompositeOperation: 'source-over',
        clearRect: () => { this.pixels.fill(0); this.lastDrawnBlob = null; },
        drawImage: (image) => {
          this.lastDrawnBlob = image.blob || image.lastDrawnBlob || null;
          const input = image.pixels;
          assert(input, 'Canvas draws must come from a decoded image or another canvas.');
          for (let offset = 0; offset < this.pixels.length; offset += 4) {
            const from = offset % input.length;
            if (this.context.globalCompositeOperation === 'destination-in') {
              this.pixels[offset + 3] = Math.round(this.pixels[offset + 3] * input[from + 3] / 255);
            } else {
              this.pixels.set(input.subarray(from, from + 4), offset);
            }
          }
        },
        createImageData: (width, height) => ({ data: new Uint8ClampedArray(width * height * 4) }),
        getImageData: () => ({ data: this.pixels.slice() }),
        putImageData: (image) => { this.pixels = image.data.slice(); },
        fillRect() {}, save() {}, restore() {}, beginPath() {}, arc() {}, fill() {},
      };
    }
    get width() { return this._width; }
    set width(value) { this._width = value; this.pixels = new Uint8ClampedArray(this.width * this.height * 4); }
    get height() { return this._height; }
    set height(value) { this._height = value; this.pixels = new Uint8ClampedArray(this.width * this.height * 4); }
    getContext() { return this.context; }
    toBlob(callback, type) {
      const blob = Object.assign(new Blob([this.pixels], { type }), {
        width: this.width, height: this.height, pixels: this.pixels.slice(), serial: ++sequence,
      });
      calls.encoded.push({ canvas: this, blob });
      const deliver = () => queueMicrotask(() => callback(blob));
      if (encodeGate?.predicate(this, blob)) {
        const gate = encodeGate;
        encodeGate = null;
        gate.release = deliver;
      } else deliver();
    }
    toDataURL() { return 'data:image/png;base64,dGVzdA=='; }
  }
  const attributesFrom = (tag) => Object.fromEntries([...tag.matchAll(/([\w-]+)(?:="([^"]*)")?/g)].slice(1).map(([, name, value]) => [name, value || '']));
  for (const match of markup.matchAll(/<(input|select|button|canvas|form|div|span|p|fieldset|details)\b[^>]*>/g)) {
    const attributes = attributesFrom(match[0]);
    const node = match[1] === 'canvas' ? new Canvas(attributes) : new Element(match[1], attributes);
    if (match[1] === 'select') {
      const options = markup.slice(match.index + match[0].length).split('</select>')[0];
      const selected = options.match(/<option\b(?=[^>]*\bselected)[^>]*value="([^"]+)"/) || options.match(/<option\b[^>]*value="([^"]+)"/);
      node.value = selected?.[1] || '';
    }
    nodes.push(node);
    if (node.id) ids.set(node.id, node);
  }
  for (const id of ['bgtool-reprocess', 'bgtool-reprocess-all', 'bgtool-pending']) {
    assert(ids.has(id), `Missing processing action markup: ${id}`);
  }
  const queryAll = (selector) => {
    if (selector.startsWith('#')) return ids.has(selector.slice(1)) ? [ids.get(selector.slice(1))] : [];
    const attribute = selector.match(/\[([^=\]]+)(?:="([^"]*)")?\]/);
    return nodes.filter((node) => attribute && Object.hasOwn(node.attributes, attribute[1])
      && (attribute[2] === undefined || node.attributes[attribute[1]] === attribute[2])
      && (!selector.endsWith(':checked') || node.checked));
  };
  document = new Element('document');
  document.querySelectorAll = queryAll;
  document.querySelector = (selector) => queryAll(selector)[0] || null;
  document.createElement = (tag) => tag === 'canvas' ? new Canvas() : new Element(tag);
  document.body = new Element('body');
  document.head = new Element('head');
  document.addEventListener('tools:run-start', (event) => calls.events.push(event.type));
  document.addEventListener('tools:run-complete', (event) => calls.events.push(event.type));
  document.addEventListener('tools:run-error', (event) => calls.events.push(event.type));
  ids.get('bgtool-method').value = 'colorkey';
  ids.get('bgtool-silhouette-enabled').checked = false;
  const createImageBitmap = (blob) => new Promise((resolve) => {
    assert(blob.pixels, 'Image decoder must receive the actual file/mask blob.');
    const bitmap = { width: blob.width, height: blob.height, pixels: blob.pixels.slice(), blob, closed: false, close() { this.closed = true; } };
    calls.decoded.push(bitmap);
    const deliver = () => resolve(bitmap);
    if (decodeGate?.predicate(blob)) {
      const gate = decodeGate;
      decodeGate = null;
      gate.release = deliver;
    } else deliver();
  });
  const window = { requestAnimationFrame: (callback) => queueMicrotask(callback), setTimeout, addEventListener() {} };
  const context = vm.createContext({
    document, window, Blob, Uint8ClampedArray, createImageBitmap, setTimeout,
    CustomEvent: class { constructor(type, options = {}) { this.type = type; Object.assign(this, options); } },
    URL: { createObjectURL: () => `blob:test-${++sequence}`, revokeObjectURL: (url) => calls.revoked.push(url) },
    console: { warn: (...args) => calls.warnings.push(args), log() {} },
  });
  // Only expose controller internals; every processing function and event listener runs unchanged.
  vm.runInContext(instrumented, context, { filename: 'background-remover.js' });
  const change = (id, value, type = 'change') => {
    const element = ids.get(id);
    assert(element, `Missing control ${id}`);
    if (typeof value === 'boolean') element.checked = value;
    else element.value = String(value);
    element.dispatchEvent({ type, bubbles: true });
  };
  const hold = (kind, predicate) => {
    const gate = { predicate, release: null };
    if (kind === 'decode') decodeGate = gate;
    else encodeGate = gate;
    return gate;
  };
  const file = (name = 'sample.png') => Object.assign(new Blob(['fixture'], { type: 'image/png' }), {
    name, width: 2, height: 1, pixels: new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]),
  });
  const settle = async () => {
    await until(() => !context.api.state.working && context.api.state.jobs.every((job) => !['queued', 'processing'].includes(job.status)), 'Processing did not settle.');
    await tick();
    assert.deepEqual(calls.warnings, [], 'The controller must not swallow async processing failures.');
    for (const job of context.api.state.jobs) assert.equal(job.status, 'ready', job.message);
  };
  return { api: context.api, ids, nodes, document, calls, change, hold, file, settle };
}

async function run() {
  const h = createHarness();
  const { api } = h;
  api.addFiles([h.file()]);
  await h.settle();
  const job = api.state.jobs[0];
  const firstMask = job.maskBlob;
  assert.equal(api.active.job, job);
  assert.equal(api.active.maskCanvas.lastDrawnBlob, firstMask, 'First upload must load its generated mask.');
  assert.deepEqual(Array.from(api.active.maskCanvas.pixels).filter((_, index) => index % 4 === 3), [0, 255]);
  assert.equal(h.calls.events.filter((event) => event === 'tools:run-start').length, 1, 'Upload still starts processing automatically.');

  for (const [id, value, type] of [
    ['bgtool-method', 'ai-fast', 'change'], ['bgtool-method', 'colorkey', 'change'],
    ['bgtool-processing', '2048', 'change'], ['bgtool-device', 'cpu', 'change'],
    ['bgtool-color', '#000000', 'input'], ['bgtool-tolerance', '32', 'input'],
  ]) h.change(id, value, type);
  await tick();
  assert.equal(job.maskBlob, firstMask, 'Editing removal settings must preserve the current result until applied.');
  assert.equal(h.calls.events.filter((event) => event === 'tools:run-start').length, 1, 'Settings changes must not rerun automatically.');
  assert.equal(h.ids.get('bgtool-reprocess').disabled, false);
  assert.equal(h.ids.get('bgtool-pending').hidden, false);
  h.ids.get('bgtool-reprocess').click();
  await until(() => job.maskBlob && job.maskBlob !== firstMask, 'The selected Reprocess button did not generate a new mask.');
  await h.settle();
  assert.notEqual(job.maskBlob, firstMask);
  assert.equal(api.active.maskCanvas.lastDrawnBlob, job.maskBlob, 'Rerun must decode the fresh mask into the selected editor.');
  assert.deepEqual(Array.from(api.active.maskCanvas.pixels).filter((_, index) => index % 4 === 3), [255, 0], 'The editor must show the new color removal, not the stale mask.');
  assert(job.appliedSettingsKey, 'Successful results need an applied-settings identity.');

  const beforeRefine = job.maskBlob;
  for (const [id, value, type] of [
    ['bgtool-threshold', '60', 'input'], ['bgtool-feather', '1', 'input'],
    ['bgtool-silhouette-width', '8', 'input'], ['bgtool-format', 'image/jpeg', 'change'],
    ['bgtool-bg', '#123456', 'input'],
  ]) h.change(id, value, type);
  await tick();
  assert.equal(job.maskBlob, beforeRefine, 'Refinement and export controls must not generate a new base mask.');
  assert.equal(api.state.thresholdPct, 60);
  assert.equal(api.state.outputFormat, 'image/jpeg');
  const radios = h.nodes.filter((node) => node.name === 'bgtool-mask-type');
  for (const radio of radios) {
    radio.checked = radio.value === 'alpha';
    radio.dispatchEvent({ type: 'input', bubbles: true });
    radio.dispatchEvent({ type: 'change', bubbles: true });
  }
  h.document.dispatchEvent({ type: 'tools:session-applied', detail: { toolId: 'background-remover' } });
  assert.equal(api.state.maskType, 'alpha', 'Session restore dispatches change to unchecked radios too.');
  const outline = h.ids.get('bgtool-silhouette-enabled');
  assert.equal(api.state.silhouetteEnabled, false, 'Normal cutouts should start without an outline.');
  outline.checked = true;
  outline.dispatchEvent({ type: 'input', bubbles: true });
  outline.dispatchEvent({ type: 'change', bubbles: true });
  assert.equal(api.state.silhouetteEnabled, true, 'Input updates must not overwrite a checkbox before its change event.');
  outline.checked = false;
  outline.dispatchEvent({ type: 'input', bubbles: true });
  outline.dispatchEvent({ type: 'change', bubbles: true });
  assert.equal(api.state.silhouetteEnabled, false);

  const batch = createHarness();
  batch.api.addFiles([batch.file('one.png'), batch.file('two.png')]);
  await batch.settle();
  const jobs = batch.api.state.jobs;
  batch.change('bgtool-color', '#000000', 'input');
  batch.change('bgtool-processing', '2048');
  const initialRunCount = batch.calls.events.filter((event) => event === 'tools:run-start').length;
  const held = batch.hold('decode', (blob) => blob === jobs[0].file);
  const rerun = batch.api.reprocessJobs(jobs.map((entry) => entry.id));
  const duplicate = batch.api.reprocessJobs(jobs.map((entry) => entry.id));
  await until(() => held.release, 'Batch rerun did not reach its first asynchronous decode.');
  const selectedBeforeRun = batch.api.state.activeJobId;
  await batch.api.selectJob(jobs[1].id);
  assert.equal(batch.api.state.activeJobId, selectedBeforeRun, 'Selection must stay fixed while processing is busy.');
  batch.change('bgtool-color', '#ffffff', 'input');
  batch.change('bgtool-processing', '512');
  held.release();
  await Promise.all([rerun, duplicate]);
  await batch.settle();
  assert.equal(batch.calls.events.filter((event) => event === 'tools:run-start').length, initialRunCount + 1, 'Duplicate Apply requests must produce one batch run.');
  assert.equal(jobs[0].appliedSettingsKey, jobs[1].appliedSettingsKey, 'Every job in the batch must use one settings snapshot.');
  for (const entry of jobs) {
    assert.equal(entry.processing.maxDim, 2048, 'Changing controls mid-batch must not change queued job settings.');
    assert.deepEqual(Array.from(entry.maskBlob.pixels).filter((_, index) => index % 4 === 3), [255, 0]);
  }
  assert.equal(batch.ids.get('bgtool-pending').hidden, false, 'A mid-run settings edit must remain visibly pending.');

  await batch.api.selectJob(jobs[0].id);
  const selectionGate = batch.hold('decode', (blob) => blob === jobs[0].file);
  const oldSelection = batch.api.selectJob(jobs[0].id);
  await until(() => selectionGate.release, 'Selection did not reach asynchronous decode.');
  await batch.api.selectJob(jobs[1].id);
  assert.equal(batch.api.state.activeJobId, jobs[0].id, 'A second selection must be ignored while selection decode is busy.');
  selectionGate.release();
  await oldSelection;
  await tick();
  await batch.api.selectJob(jobs[1].id);
  assert.equal(batch.api.state.activeJobId, jobs[1].id);
  assert.equal(batch.api.active.job, jobs[1]);
  assert.equal(batch.api.active.sourceCanvas.lastDrawnBlob, jobs[1].file, 'An older selection decode must not overwrite the current source.');
  assert.equal(batch.api.active.maskCanvas.lastDrawnBlob, jobs[1].maskBlob, 'An older selection decode must not overwrite the current mask.');
  batch.api.active.dirtyMask = true;
  const commitGate = batch.hold('encode', (canvas) => canvas === batch.api.active.maskCanvas);
  const selectionAfterEdit = batch.api.selectJob(jobs[0].id);
  await until(() => commitGate.release, 'Changing selection did not commit the edited mask.');
  await batch.api.selectJob(jobs[1].id);
  commitGate.release();
  await selectionAfterEdit;
  assert.equal(batch.api.state.activeJobId, jobs[0].id, 'The accepted selection must survive its asynchronous mask commit.');
  assert.equal(batch.api.active.job, jobs[0]);
  assert.equal(batch.api.active.maskCanvas.lastDrawnBlob, jobs[0].maskBlob);
  assert(batch.calls.decoded.every((bitmap) => bitmap.closed), 'Decoded image resources must be closed, including stale selections.');
}

run().then(() => {
  console.log('Background Remover processing tests passed: explicit apply, fresh masks, batch snapshots, duplicate requests, and selection races.');
}).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
