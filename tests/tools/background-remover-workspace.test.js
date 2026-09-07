'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

class Element {
  constructor(value = '') {
    this.value = value;
    this.checked = false;
    this.hidden = false;
    this.open = false;
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.classList = { toggle() {}, add() {}, remove() {} };
    this.textContent = '';
    this.innerHTML = '';
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  dispatchEvent(event) {
    (this.listeners.get(event.type) || []).forEach((listener) => listener(event));
    return true;
  }

  setAttribute(name, value) { this.attributes[name] = String(value); }
  getContext() { return { clearRect() {} }; }
  querySelectorAll() { return []; }
}

const elements = new Map();
const element = (selector) => {
  if (!elements.has(selector)) elements.set(selector, new Element());
  return elements.get(selector);
};
element('#bgtool-method').value = 'ai-best';
element('#bgtool-format').value = 'image/png';
element('input[name="bgtool-mask-type"]:checked').value = 'alpha';
const viewButtons = ['cutout', 'original', 'mask'].map((view) => {
  const button = new Element();
  button.dataset.bgtoolView = view;
  return button;
});
const document = new Element();
document.body = new Element();
document.querySelector = element;
document.querySelectorAll = (selector) => selector === '[data-bgtool-view]' ? viewButtons : [];
document.createElement = () => new Element();

const context = vm.createContext({
  document,
  console,
  CustomEvent: class { constructor(type, options = {}) { this.type = type; Object.assign(this, options); } },
});
const source = fs.readFileSync(path.join(__dirname, '../../js/tools/background-remover.js'), 'utf8');
// Expose existing functions from the evaluated tool; processing and rendering logic are not copied.
const instrumented = source.replace(/\}\)\(\);\s*$/, `
  globalThis.testApi = { state, active, updateSummary, renderResults, updateActionButtons, readProcessingSettings, processingSettingsKey };
})();`);
vm.runInContext(instrumented, context, { filename: 'background-remover.js' });
const api = context.testApi;
assert(api, 'Background Remover must initialize against its required UI elements.');

assert.equal(element('[data-bgtool="workspace-summary"]').textContent, 'AI best quality · Smooth edges');
assert.equal(element('[data-bgtool="output-summary"]').textContent, 'PNG · Transparent');
assert.equal(element('[data-bgtool="selection-details"]').hidden, true);
assert.equal(element('#bgtool-download-selected').disabled, true);

element('#bgtool-method').value = 'colorkey';
element('#bgtool-format').value = 'image/webp-solid';
element('input[name="bgtool-mask-type"]:checked').value = 'binary';
element('#bgtool-form').dispatchEvent({ type: 'change' });
assert.equal(element('[data-bgtool="workspace-summary"]').textContent, 'Solid color · Hard edges');
assert.equal(element('[data-bgtool="output-summary"]').textContent, 'WebP · Solid background');

element('#bgtool-format').value = 'image/tiff';
document.dispatchEvent({ type: 'tools:session-applied', detail: { toolId: 'background-remover' } });
assert.equal(element('[data-bgtool="output-summary"]').textContent, 'TIFF · Transparent');

const ready = {
  id: 'ready-1', name: '<photo "one">.png', status: 'ready', bytes: 2048,
  original: { width: 1200, height: 800 }, cutoutUrl: 'blob:preview',
  approved: true, message: 'Completed <locally>.',
  appliedSettingsKey: api.processingSettingsKey(api.readProcessingSettings()),
};
const error = {
  id: 'error-1', name: 'failed.png', status: 'error', bytes: 1024,
  original: {}, approved: false, message: 'Could not decode <photo>.',
};
api.state.jobs.push(ready, error);
api.state.activeJobId = ready.id;
api.active.job = ready;
api.updateSummary();
api.renderResults();
api.updateActionButtons();
assert.equal(element('[data-bgtool="selection-details"]').hidden, false);
assert.equal(element('[data-bgtool="selection-details"]').open, true, 'File errors must reveal the selected-photo details.');
assert.equal(element('#bgtool-download-selected').disabled, false);
assert.equal(element('#bgtool-download-all').disabled, false);
assert.equal(element('#bgtool-selected').textContent, ready.name);
assert.equal(viewButtons[0].attributes['aria-pressed'], 'true');
assert.match(element('#bgtool-results').innerHTML, /&lt;photo &quot;one&quot;&gt;\.png/);
assert.doesNotMatch(element('#bgtool-results').innerHTML, /<photo/);
assert.match(element('#bgtool-results').innerHTML, /data-bgtool-approve="ready-1"[^>]*checked/);
assert.match(element('#bgtool-results').innerHTML, /aria-label="Include &lt;photo &quot;one&quot;&gt;\.png in batch download"/);
assert.match(element('#bgtool-results').innerHTML, /data-bgtool-rerun="ready-1"/);
assert.match(element('#bgtool-results').innerHTML, /<details class="bgtool-result-details">/);
assert.match(element('#bgtool-results').innerHTML, /role="status">Could not decode &lt;photo&gt;\./);

api.state.view = 'mask';
ready.approved = false;
api.updateActionButtons();
assert.equal(viewButtons[2].attributes['aria-pressed'], 'true');
assert.equal(element('#bgtool-download-all').disabled, true);
assert.equal(element('#bgtool-download-selected').disabled, false);

element('#bgtool-results').dispatchEvent({
  type: 'click',
  target: { closest: (selector) => selector === 'details, .bgtool-approve' ? {} : null },
});
assert.equal(api.state.activeJobId, ready.id, 'Opening processing details must preserve selection.');

console.log('Background Remover workspace checks passed.');
