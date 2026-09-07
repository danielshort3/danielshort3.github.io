'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '../../js/tools/tool-workspace.js'), 'utf8');

const listeners = new Map();
const cleanup = [];
const panels = new Map();
const groups = [];
const notifications = [];
const makeGroup = (name) => {
  const group = {
    tabs: [], panels: [], nested: [],
    querySelectorAll(selector) {
      const key = selector === '[data-workspace-tab]' ? 'tabs' : 'panels';
      return [...this[key], ...this.nested.flatMap((child) => child[key])];
    },
    dispatchEvent(event) { notifications.push(event.detail.panelId); }
  };
  for (const suffix of ['first', 'second', 'third']) {
    const id = `${name}-${suffix}`;
    const panel = { id, hidden: suffix !== 'first', value: 'Keep the draft', scrollTop: 47, closest: () => group };
    const attrs = { 'aria-controls': id, 'aria-selected': String(suffix === 'first') };
    const tab = {
      disabled: false, hidden: false, tabIndex: suffix === 'first' ? 0 : -1,
      closest: (selector) => selector === '[data-workspace-tab]' ? tab : selector === '[role="tablist"]' ? { getAttribute: () => 'horizontal' } : group,
      getAttribute: (key) => attrs[key],
      setAttribute: (key, value) => { attrs[key] = value; },
      focus: () => { document.activeElement = tab; }
    };
    group.tabs.push(tab);
    group.panels.push(panel);
    panels.set(id, panel);
  }
  groups.push(group);
  return group;
};
const document = {
  readyState: 'complete',
  getElementById: (id) => panels.get(id),
  querySelectorAll: () => groups,
  addEventListener(type, listener) {
    const entries = listeners.get(type) || [];
    entries.push(listener);
    listeners.set(type, entries);
    cleanup.push(() => listeners.set(type, (listeners.get(type) || []).filter((item) => item !== listener)));
  }
};
const window = { SiteRoutes: { addCleanup: (callback) => cleanup.push(callback) } };
class TestEvent {
  constructor(type, options) { this.type = type; Object.assign(this, options); }
  preventDefault() { this.prevented = true; }
}
const context = vm.createContext({ document, window, CustomEvent: TestEvent });
const run = () => vm.runInContext(source, context);
const fire = (type, target, key) => {
  const event = new TestEvent(type, { target, key });
  for (const callback of [...(listeners.get(type) || [])]) callback(event);
  return event;
};

const outer = makeGroup('setup');
const inner = makeGroup('results');
outer.nested.push(inner);
run();
run();
assert.strictEqual(listeners.get('click').length, 1, 'replayed scripts must not duplicate actions');
assert.strictEqual(notifications.length, 0, 'initialization must not trigger tool actions');
fire('click', outer.tabs[1]);
assert.strictEqual(outer.panels[0].hidden, true);
assert.strictEqual(outer.panels[1].hidden, false);
assert.strictEqual(inner.panels[0].hidden, false, 'parent navigation must preserve nested view selection');
assert.strictEqual(outer.tabs[1].tabIndex, 0);
assert.strictEqual(outer.tabs[0].tabIndex, -1);
assert.strictEqual(outer.panels[0].value, 'Keep the draft');
assert.strictEqual(outer.panels[0].scrollTop, 47, 'navigation must not replace content nodes');

outer.tabs[2].disabled = true;
assert(fire('keydown', outer.tabs[1], 'ArrowRight').prevented);
assert.strictEqual(document.activeElement, outer.tabs[0], 'arrows wrap and skip unavailable tabs');
fire('keydown', outer.tabs[0], 'End');
assert.strictEqual(document.activeElement, outer.tabs[1]);
fire('keydown', outer.tabs[1], 'Home');
assert.strictEqual(document.activeElement, outer.tabs[0]);
assert.strictEqual(fire('keydown', outer.tabs[0], 'Tab').prevented, undefined, 'ordinary Tab must keep native behavior');
outer.tabs[1].hidden = true;
assert.strictEqual(window.ToolWorkspace.selectTab('setup-second'), false, 'hidden tabs cannot be selected');
assert.strictEqual(window.ToolWorkspace.selectTab('missing-panel'), false);
window.ToolWorkspace.selectTab('results-second');
assert.strictEqual(inner.panels[1].hidden, false);
assert.strictEqual(outer.panels[0].hidden, false);

cleanup.splice(0).forEach((callback) => callback());
assert.strictEqual(window.ToolWorkspace, undefined, 'route teardown must release the scoped API');
run();
fire('click', inner.tabs[0]);
assert.strictEqual(inner.panels[0].hidden, false, 'tabs work after returning through soft navigation');
assert.strictEqual(listeners.get('click').length, 1);
console.log('Shared tool workspace tests passed.');
