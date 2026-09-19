'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const core = require('../../js/tools/text-compare-core.js');
const source = fs.readFileSync(path.resolve(__dirname, '../../js/tools/text-compare.js'), 'utf8');

class Element {
  constructor(id = '') {
    Object.assign(this, { id, value: '', textContent: '', innerHTML: '', hidden: false, disabled: false, dataset: {}, attributes: new Map(), listeners: new Map(), style: { setProperty() {} } });
  }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter(entry => entry !== listener)); }
  dispatchEvent(event) { (this.listeners.get(event.type) || []).forEach(listener => listener(event)); return true; }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  toggleAttribute(name, force) { if (force) this.setAttribute(name, ''); else this.removeAttribute(name); }
  closest() { return null; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  focus() { this.ownerDocument.activeElement = this; }
}

function createHarness() {
  const document = new Element();
  const elements = new Map();
  const create = id => {
    const element = new Element(id);
    element.ownerDocument = document;
    elements.set(id, element);
    return element;
  };
  [
    'textcompare-form', 'textcompare-original', 'textcompare-revised', 'textcompare-output', 'textcompare-summary',
    'textcompare-clear', 'textcompare-swap', 'textcompare-copy', 'textcompare-copy-status',
    'textcompare-warning', 'textcompare-view-drafts', 'textcompare-view-comparison', 'ready', 'mode-summary'
  ].forEach(create);
  const modes = ['auto', 'document', 'structured'].map(mode => Object.assign(create(`textcompare-mode-${mode}`), { value: mode, checked: mode === 'auto' }));
  document.querySelectorAll = selector => selector === 'input[name="textcompare-mode"]' ? modes : [];
  document.getElementById = id => elements.get(id) || null;
  document.querySelector = selector => selector.startsWith('#') ? elements.get(selector.slice(1)) || null
    : selector === '[data-textcompare-ready]' ? elements.get('ready')
      : selector === '[data-textcompare-mode-summary]' ? elements.get('mode-summary') : null;
  document.body = create('body');
  document.activeElement = document.body;
  document.readyState = 'complete';
  const output = elements.get('textcompare-output');
  output.innerHTML = '<p class="textcompare-empty">Waiting for input.</p>';
  const requests = [];
  let worker;
  class CompareWorker extends Element {
    constructor() { super(); worker = this; }
    postMessage(payload) { requests.push(payload); }
    terminate() {}
  }
  let now = 0;
  let nextTimerId = 0;
  const timers = new Map();
  const setTimer = (callback, delay = 0) => { const id = ++nextTimerId; timers.set(id, { callback, at: now + delay }); return id; };
  const clearTimer = id => timers.delete(id);
  const window = { TextCompareCore: core, setTimeout: setTimer, clearTimeout: clearTimer };
  const context = vm.createContext({
    document, window, Worker: CompareWorker, requestAnimationFrame: callback => callback(),
    setTimeout: setTimer, clearTimeout: clearTimer, console,
    Event: class { constructor(type, options = {}) { this.type = type; Object.assign(this, options); } preventDefault() {} },
    CustomEvent: class { constructor(type, options = {}) { this.type = type; Object.assign(this, options); } },
    URL, URLSearchParams, Blob, navigator: {}
  });
  vm.runInContext(source, context, { filename: 'text-compare.js' });
  const flush = async () => { for (let index = 0; index < 5; index += 1) await Promise.resolve(); };
  return {
    document, output, requests, flush,
    get: id => elements.get(id),
    fill(id, value) { const element = elements.get(id); element.value = value; element.dispatchEvent({ type: 'input' }); },
    click(id) { elements.get(id).dispatchEvent({ type: 'click' }); },
    submit() { elements.get('textcompare-form').dispatchEvent({ type: 'submit', preventDefault() {} }); },
    async tick(milliseconds) {
      const until = now + milliseconds;
      while (true) {
        const next = [...timers.entries()].filter(([, timer]) => timer.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        timers.delete(next[0]); now = next[1].at; next[1].callback(); await flush();
      }
      now = until; await flush();
    },
    async respond(request = requests.at(-1)) {
      const result = core.compareText(request);
      worker.dispatchEvent({ type: 'message', data: { ...result, requestId: request.requestId, ok: true } });
      await flush();
    },
    capture() {
      const payload = { inputs: { existing: 'preserved' } };
      document.dispatchEvent({ type: 'tools:session-capture', detail: { toolId: 'text-compare', payload } });
      return JSON.parse(JSON.stringify(payload));
    },
    apply(snapshot, toolId = 'text-compare') { document.dispatchEvent({ type: 'tools:session-applied', detail: { toolId, snapshot } }); }
  };
}

async function run() {
  const h = createHarness();
  const copy = h.get('textcompare-copy');
  h.fill('textcompare-original', 'Publish the ORIGINALMARKER draft.');
  h.fill('textcompare-revised', 'Publish the FIRSTREVISION draft.');
  await h.tick(1000);
  assert.equal(h.requests.length, 0, 'Typing an unfinished first draft does not start comparison.');
  assert(copy.disabled, 'Copy cannot export an unprocessed first draft.');
  assert.equal(h.capture().inputs.existing, 'preserved', 'Capturing results preserves other captured inputs.');
  h.get('textcompare-revised').focus();
  h.submit();
  assert.equal(h.requests.length, 1, 'Explicit Compare starts the first calculation.');
  const obsolete = h.requests[0];
  h.fill('textcompare-revised', 'Publish the CURRENTREVISION draft.');
  await h.respond(obsolete);
  assert(!h.output.innerHTML.includes('FIRSTREVISION') && copy.disabled, 'An old in-flight result cannot replace newer edits or enable Copy.');
  await h.tick(449);
  assert.equal(h.requests.length, 1, 'Editing waits for the debounce before comparing again.');
  await h.tick(1);
  assert.equal(h.requests.length, 2, 'The latest edit automatically compares after 450ms.');
  await h.respond();
  assert(h.output.innerHTML.includes('CURRENTREVISION') && !copy.disabled, 'The current result becomes available to review and copy.');
  assert.equal(h.document.activeElement, h.get('textcompare-revised'), 'Automatic comparison keeps focus in the edited draft.');
  assert(!h.get('textcompare-view-drafts').hidden && !h.get('textcompare-view-comparison').hidden, 'Editors and result remain visible together.');

  h.fill('textcompare-revised', 'Publish the PENDINGREVISION draft.');
  assert(copy.disabled && !h.output.innerHTML.includes('CURRENTREVISION'), 'Editing immediately removes the stale result and disables Copy.');
  h.click('textcompare-clear');
  await h.tick(1000);
  assert.equal(h.requests.length, 2, 'Clear cancels the pending automatic comparison.');
  assert.equal(h.get('textcompare-original').value, '');
  assert.equal(h.get('textcompare-revised').value, '');
  assert(copy.disabled && !h.output.innerHTML.includes('PENDINGREVISION'), 'Clear leaves no old result available for copying.');
  h.fill('textcompare-original', 'Original to clear.');
  h.fill('textcompare-revised', 'Revised to clear.');
  h.submit();
  h.click('textcompare-clear');
  await h.respond();
  assert(copy.disabled && !h.output.innerHTML.includes('Revised to clear'), 'Clear also invalidates an already-running comparison.');

  h.get('textcompare-original').value = 'Restored ORIGINALMARKER draft.';
  h.get('textcompare-revised').value = 'Restored CURRENTREVISION draft.';
  h.get('textcompare-revised').focus();
  const beforeDraft = h.requests.length;
  h.apply({ inputs: { view: 'comparison' }, output: { kind: 'html', html: '<p class="textcompare-empty">Waiting for input.</p>' } });
  await h.tick(1000);
  assert.equal(h.requests.length, beforeDraft, 'A legacy unprocessed draft remains unprocessed regardless of its saved view.');
  assert(copy.disabled && !h.get('textcompare-view-drafts').hidden && !h.get('textcompare-view-comparison').hidden, 'Legacy view metadata does not hide the drafts or invent a copyable comparison.');
  for (const view of ['drafts', 'comparison', 'unknown']) {
    h.apply({ inputs: { view }, output: { kind: 'html', html: '<p>LEGACYPREVIEW</p>', summary: 'Saved comparison' } });
    assert(copy.disabled, 'A saved preview cannot be copied with runs from a different comparison.');
    await h.respond();
    assert(h.output.innerHTML.includes('CURRENTREVISION') && !copy.disabled, 'Restored comparison regenerates its copyable runs from the restored drafts.');
    assert(!h.get('textcompare-view-drafts').hidden && !h.get('textcompare-view-comparison').hidden, `Legacy ${view} view keeps the single continuous workspace.`);
    assert.equal(h.document.activeElement, h.get('textcompare-revised'), 'Session restoration does not steal focus.');
  }
  const beforeOtherTool = h.requests.length;
  const beforeOtherOutput = h.output.innerHTML;
  h.apply({ output: { kind: 'text', text: 'Another tool result' } }, 'word-frequency');
  assert.equal(h.requests.length, beforeOtherTool, 'Another tool session cannot trigger Text Compare.');
  assert.equal(h.output.innerHTML, beforeOtherOutput, 'Another tool session cannot replace the comparison.');

  const defaults = createHarness();
  const before = defaults.get('textcompare-original');
  const after = defaults.get('textcompare-revised');
  assert.equal(before.value, '', 'The default example never becomes saved Before input.');
  assert.equal(after.value, '', 'The default example never becomes saved After input.');
  assert(before.placeholder.includes('Monday') && after.placeholder.includes('Friday'), 'Both empty editors visibly preview the example.');
  await defaults.tick(1000);
  assert.equal(defaults.requests.length, 0, 'The initial example waits for Compare.');
  defaults.submit();
  const exampleRequest = defaults.requests.at(-1);
  assert.equal(exampleRequest.leftText, before.placeholder, 'Compare uses the visible Before example.');
  assert.equal(exampleRequest.rightText, after.placeholder, 'Compare uses the visible After example.');
  await defaults.respond();
  assert(defaults.output.innerHTML.includes('diff-ins') && defaults.output.innerHTML.includes('diff-del'), 'The default example produces a real comparison.');
  assert.equal(before.value + after.value, '', 'Comparing the example leaves actual input values empty.');
  const savedExample = defaults.capture();
  defaults.apply({ output: savedExample.output });
  assert(defaults.get('textcompare-copy').disabled, 'A restored example waits for its own copyable result.');
  await defaults.respond();
  assert(!defaults.get('textcompare-copy').disabled, 'Restoring an example regenerates a copyable comparison from blank inputs.');

  for (const field of ['original', 'revised']) {
    const oneSided = createHarness();
    oneSided.submit();
    const obsoleteExample = oneSided.requests.at(-1);
    oneSided.fill('textcompare-' + field, 'ONLY_USER_TEXT');
    assert(!oneSided.get('textcompare-original').placeholder.includes('Product analytics') &&
      !oneSided.get('textcompare-revised').placeholder.includes('Product analytics'), 'Typing in either editor immediately removes both example previews.');
    await oneSided.respond(obsoleteExample);
    assert(!oneSided.output.innerHTML.includes('Product analytics'), 'An example finishing after a user edit cannot reappear.');
    await oneSided.tick(450);
    const request = oneSided.requests.at(-1);
    assert.equal(request.leftText, field === 'original' ? 'ONLY_USER_TEXT' : '', 'Before uses only actual user input once either side is edited.');
    assert.equal(request.rightText, field === 'revised' ? 'ONLY_USER_TEXT' : '', 'After never fills its blank side with example text.');
    await oneSided.respond();
    assert(oneSided.output.innerHTML.includes(field === 'original' ? 'diff-del' : 'diff-ins'), 'One-sided comparisons show complete deletion or insertion.');
    assert(!oneSided.output.innerHTML.includes('Product analytics'), 'The completed one-sided result contains no example text.');
    const saved = oneSided.capture();
    oneSided.apply({ output: saved.output });
    await oneSided.respond();
    assert(!oneSided.get('textcompare-copy').disabled && oneSided.output.innerHTML.includes('ONLY_USER_TEXT'), 'One-sided saved comparisons restore copyable runs.');
    oneSided.click('textcompare-clear');
    assert(oneSided.get('textcompare-original').placeholder.includes('Product analytics') &&
      oneSided.get('textcompare-revised').placeholder.includes('Product analytics'), 'Clear restores the shared default example preview.');
  }
  const whitespace = createHarness();
  whitespace.fill('textcompare-original', ' ');
  whitespace.submit();
  assert.equal(whitespace.requests.at(-1).leftText, ' ', 'Even whitespace input is treated as user text.');
  assert.equal(whitespace.requests.at(-1).rightText, '', 'Whitespace input does not reintroduce the default example.');
  console.log('Text Compare default example, one-sided comparisons, continuation, debounced edits, stale worker cancellation, Clear, and legacy restoration passed.');
}

module.exports = run;
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
