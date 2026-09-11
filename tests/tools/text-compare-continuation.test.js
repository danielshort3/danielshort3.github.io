'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const read = (file) => fs.readFileSync(path.resolve(__dirname, '../../js/tools', file), 'utf8');
const toolSource = read('text-compare.js');
const workspaceSource = read('tool-workspace.js');

class Element {
  constructor(id = '') {
    this.id = id;
    this.value = '';
    this.textContent = '';
    this.innerHTML = '';
    this.hidden = false;
    this.attributes = new Map();
    this.listeners = new Map();
    this.style = { setProperty() {} };
    this.focusCount = 0;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  dispatchEvent(event) {
    (this.listeners.get(event.type) || []).forEach((listener) => listener(event));
    if (event.bubbles) this.parent?.dispatchEvent(event);
    return true;
  }

  getAttribute(name) { return this.attributes.get(name) ?? null; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  focus() { this.focusCount += 1; }
  closest(selector) {
    if (selector === '[data-workspace-tabset]') return this.group || null;
    if (selector === '[data-workspace-tab]') return this.isTab ? this : null;
    return null;
  }
}

function createHarness() {
  const document = new Element();
  const elements = new Map();
  const create = (id) => {
    const element = new Element(id);
    elements.set(id, element);
    return element;
  };
  const group = create('textcompare-workspace');
  group.parent = document;
  const tabs = ['drafts', 'comparison'].map((view) => {
    const tab = create(`textcompare-view-tab-${view}`);
    tab.group = group;
    tab.isTab = true;
    tab.setAttribute('aria-controls', `textcompare-view-${view}`);
    tab.setAttribute('aria-selected', String(view === 'drafts'));
    return tab;
  });
  const panels = ['drafts', 'comparison'].map((view) => {
    const panel = create(`textcompare-view-${view}`);
    panel.group = group;
    return panel;
  });
  group.querySelectorAll = (selector) => selector === '[data-workspace-tab]' ? tabs : panels;
  document.querySelectorAll = (selector) => selector === '[data-workspace-tabset]' ? [group] : [];
  document.getElementById = (id) => elements.get(id) || null;
  document.querySelector = (selector) => selector.startsWith('#')
    ? elements.get(selector.slice(1)) || null
    : selector === '[data-textcompare-ready]' ? elements.get('ready') : null;
  document.body = new Element();
  document.readyState = 'complete';
  ['textcompare-form', 'textcompare-original', 'textcompare-revised', 'textcompare-output', 'textcompare-summary', 'ready'].forEach(create);
  const output = elements.get('textcompare-output');
  output.innerHTML = '<p class="textcompare-empty">Waiting for input.</p>';
  output.textContent = 'Waiting for input.';
  const window = { TextCompareCore: { compareText() { throw new Error('Restoring a draft must not run a comparison.'); } } };
  const context = vm.createContext({
    document,
    window,
    CustomEvent: class {
      constructor(type, options = {}) { this.type = type; Object.assign(this, options); }
    }
  });
  vm.runInContext(workspaceSource, context, { filename: 'tool-workspace.js' });
  vm.runInContext(toolSource, context, { filename: 'text-compare.js' });
  let dirtyCount = 0;
  document.addEventListener('tools:session-dirty', () => { dirtyCount += 1; });
  const selectedView = () => tabs.find((tab) => tab.getAttribute('aria-selected') === 'true')?.id;
  return {
    document, elements, output, window,
    dirtyCount: () => dirtyCount,
    selectedView,
    capture() {
      const payload = { inputs: { existing: 'preserved' } };
      document.dispatchEvent({ type: 'tools:session-capture', detail: { toolId: 'text-compare', payload } });
      return JSON.parse(JSON.stringify(payload));
    },
    apply(snapshot, toolId = 'text-compare') {
      document.dispatchEvent({ type: 'tools:session-applied', detail: { toolId, snapshot } });
    }
  };
}

function run() {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };
  const h = createHarness();
  h.elements.get('textcompare-original').value = 'Original unfinished draft';
  h.elements.get('textcompare-revised').value = 'Revised unfinished draft';
  const draft = h.capture();
  check(draft.inputs.view === 'drafts', 'An unprocessed autosave must capture Drafts as the selected view.');
  check(draft.inputs.existing === 'preserved', 'View capture must preserve other captured inputs.');

  h.window.ToolWorkspace.selectTab('textcompare-view-comparison');
  check(h.dirtyCount() === 1, 'Selecting Comparison must mark the account session dirty.');
  check(h.capture().inputs.view === 'comparison', 'Capture must follow the selected workspace tab.');
  h.apply(draft);
  check(h.selectedView() === 'textcompare-view-tab-drafts' && !h.elements.get('textcompare-view-drafts').hidden,
    'Restoring an unfinished draft must reveal the restored draft fields.');
  check(h.elements.get('textcompare-view-comparison').hidden, 'An unfinished saved Drafts view must hide placeholder comparison output.');
  check(h.elements.get('ready').textContent === 'Ready to compare', 'Restored draft fields must refresh the workspace readiness summary.');
  check(h.dirtyCount() === 1, 'Restoring the selected tab must not create an autosave loop.');
  check(h.elements.get('textcompare-view-tab-drafts').focusCount === 0, 'Restoring a view must not steal focus.');

  h.apply({ inputs: { view: 'comparison' }, output: draft.output });
  check(h.selectedView() === 'textcompare-view-tab-comparison', 'An explicitly saved Comparison view must remain selected even before processing.');
  const comparison = { kind: 'html', html: '<span class="textcompare-ins">Added text</span>', summary: '1 insertion' };
  h.apply({ inputs: { view: 'drafts' }, output: comparison });
  check(h.selectedView() === 'textcompare-view-tab-drafts', 'An explicitly saved Drafts view must take precedence over an existing comparison.');
  check(h.output.innerHTML === comparison.html, 'Restoring Drafts must retain the saved comparison output for later viewing.');
  check(h.elements.get('textcompare-summary').textContent === comparison.summary, 'The comparison summary must remain restored.');
  h.apply({ output: comparison });
  check(h.selectedView() === 'textcompare-view-tab-comparison', 'Legacy snapshots with real HTML comparison output must reopen Comparison.');
  h.apply({ output: { kind: 'text', text: 'A real comparison result' } });
  check(h.selectedView() === 'textcompare-view-tab-comparison', 'Legacy text comparison output must reopen Comparison.');

  for (const output of [
    draft.output,
    { kind: 'html', html: '<p class="textcompare-empty">Ready to compare.</p>' },
    { kind: 'html', html: '' },
    { kind: 'text', text: 'Waiting for input.' },
    { kind: 'text', text: '' },
    undefined
  ]) {
    h.window.ToolWorkspace.selectTab('textcompare-view-comparison', { notify: false });
    h.apply({ output });
    check(h.selectedView() === 'textcompare-view-tab-drafts', 'Legacy empty or placeholder output must reopen Drafts.');
  }
  h.apply({ inputs: { view: 'unknown' }, output: comparison });
  check(h.selectedView() === 'textcompare-view-tab-comparison', 'An invalid view value must fall back to the legacy output rule.');
  h.apply({ inputs: { view: 'drafts' } }, 'word-frequency');
  check(h.selectedView() === 'textcompare-view-tab-comparison', 'Restoring another tool must not change Text Compare.');
  h.document.dispatchEvent({ type: 'tool:tab-change', detail: { panelId: 'another-tool-panel' } });
  check(h.dirtyCount() === 1, 'Unrelated workspace tabs and session restores must not mark Text Compare dirty.');
  h.window.ToolWorkspace.selectTab('textcompare-view-drafts');
  check(h.dirtyCount() === 2 && h.capture().inputs.view === 'drafts', 'Returning to Drafts must automatically save that view selection.');
  return checks;
}

module.exports = run;
if (require.main === module) console.log(`Text Compare continuation: ${run()} checks passed.`);
