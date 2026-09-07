'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/qr-code-generator.js'), 'utf8');
const markup = fs.readFileSync(path.join(__dirname, '../../pages/qr-code-generator.html'), 'utf8');
const sourceBlock = (start, end) => {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert(from >= 0 && to > from, `Missing QR controller source: ${start}`);
  return source.slice(from, to);
};
const tabMarkup = [...markup.matchAll(/<button\b[^>]*data-qrtool-tab="([^"]+)"[^>]*>([^<]+)<\/button>/g)];
assert.deepEqual(tabMarkup.map((match) => match[2].trim()), ['Content', 'Style', 'Export']);

const defaults = vm.runInNewContext(`${sourceBlock('  const DEFAULTS =', '  const TEMPLATES =')}\nDEFAULTS;`);
assert(defaults.marginModules >= 4, 'New QR codes need at least four blank modules on each side.');
assert.equal(defaults.bg, '#FFFFFF', 'Default QR codes need a solid light background.');
assert.equal(defaults.transparent, false);
assert.equal(defaults.centerMode, 'none', 'Logo cutouts must be an explicit customization.');
assert.equal(defaults.dotStyle, 'square');
assert.equal(defaults.cornerStyle, 'square');
assert.equal(defaults.ecc, 'H');
const fgChannels = defaults.fg.slice(1).match(/../g).map((channel) => parseInt(channel, 16));
assert(Math.max(...fgChannels) < 128, 'Default modules must be dark on the light background.');
assert.match(markup, new RegExp(`id="qrtool-margin"[^>]*value="${defaults.marginModules}"`), 'Markup and renderer quiet-zone defaults must agree.');

function createHarness(hash = '') {
  const calls = { dirty: 0, persist: 0, share: 0 };
  let activeElement = null;
  class Element {
    constructor() {
      this.attributes = {};
      this.dataset = {};
      this.listeners = new Map();
      this.hidden = false;
      this.open = false;
      this.tabIndex = -1;
    }
    getAttribute(name) { return this.attributes[name] ?? null; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    addEventListener(type, handler) { this.listeners.set(type, handler); }
    dispatch(type, values = {}) {
      const event = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...values };
      this.listeners.get(type)?.(event);
      return event;
    }
    focus() { activeElement = this; }
  }
  const buttons = tabMarkup.map(([tag, name]) => {
    const button = new Element();
    button.dataset.qrtoolTab = name;
    for (const [, key, value] of tag.matchAll(/([\w-]+)="([^"]*)"/g)) button.setAttribute(key, value);
    return button;
  });
  const panels = buttons.map((button) => {
    const panel = new Element();
    panel.dataset.qrtoolPanel = button.dataset.qrtoolTab;
    const id = button.getAttribute('aria-controls');
    assert(markup.includes(`id="${id}"`), `Missing controlled panel ${id}`);
    return panel;
  });
  const disclosures = Object.fromEntries(['templates', 'logo'].map((name) => [`#qrtool-panel-${name}`, new Element()]));
  const advancedOptions = new Element();
  const document = { body: new Element() };
  const state = { uiMode: 'basic' };
  const tabs = { buttons, panels, panelWrap: { scrollTop: 80 } };
  const context = vm.createContext({
    tabs, advancedOptions, document, state,
    window: { location: { hash } },
    $: (selector) => disclosures[selector],
    markSessionDirty: () => { calls.dirty += 1; },
    schedulePersistLastConfig: () => { calls.persist += 1; },
    scheduleSyncShareUrl: () => { calls.share += 1; },
  });
  // Execute the production controller and disclosure listener, not copies of their logic.
  const controller = sourceBlock('  const getVisibleTabButtons =', '  const setVerification =');
  const toggleListener = sourceBlock("  advancedOptions?.addEventListener('toggle'", "  document.querySelector('[data-qrtool-presets-open]')");
  vm.runInContext(`${controller}\n${toggleListener}\ninitTabs();\nglobalThis.api = { setUiMode };`, context, { filename: 'qr-code-generator.js' });
  const selected = (name, { focused = false } = {}) => {
    assert.equal(buttons.filter((button) => button.getAttribute('aria-selected') === 'true').length, 1);
    assert.equal(buttons.filter((button) => button.tabIndex === 0).length, 1);
    assert.equal(panels.filter((panel) => !panel.hidden).length, 1);
    assert.equal(panels.find((panel) => !panel.hidden).dataset.qrtoolPanel, name);
    const button = buttons.find((item) => item.dataset.qrtoolTab === name);
    assert.equal(button.getAttribute('aria-selected'), 'true');
    assert.equal(button.tabIndex, 0);
    if (focused) assert.equal(activeElement, button);
  };
  return { buttons, panels, tabs, disclosures, advancedOptions, state, document, calls, api: context.api, selected };
}

const keyboard = createHarness();
keyboard.selected('generate');
assert.equal(keyboard.tabs.panelWrap.scrollTop, 0);
for (const [index, key, expected] of [
  [0, 'ArrowRight', 'customize'],
  [1, 'ArrowRight', 'export'],
  [2, 'ArrowRight', 'generate'],
  [0, 'ArrowLeft', 'export'],
  [2, 'Home', 'generate'],
  [0, 'End', 'export'],
]) {
  assert.equal(keyboard.buttons[index].dispatch('keydown', { key }).defaultPrevented, true);
  keyboard.selected(expected, { focused: true });
}
assert.equal(keyboard.buttons[2].dispatch('keydown', { key: 'Tab' }).defaultPrevented, false);
keyboard.selected('export');
keyboard.buttons[1].dispatch('click');
keyboard.selected('customize', { focused: true });
keyboard.buttons[1].hidden = true;
keyboard.buttons[0].dispatch('keydown', { key: 'ArrowRight' });
keyboard.selected('export', { focused: true });

for (const [hash, expected] of Object.entries({
  content: 'generate', style: 'customize', templates: 'customize', logo: 'customize',
  generate: 'generate', customize: 'customize', export: 'export', unknown: 'generate',
})) {
  const harness = createHarness(`#${hash}`);
  harness.selected(expected);
  for (const name of ['templates', 'logo']) {
    assert.equal(harness.disclosures[`#qrtool-panel-${name}`].open, name === hash);
  }
}

const mode = createHarness('#style');
mode.api.setUiMode('advanced');
assert.equal(mode.advancedOptions.open, true);
assert.equal(mode.document.body.dataset.qrtoolUiMode, 'advanced');
assert.equal(mode.state.uiMode, 'advanced');
mode.advancedOptions.dispatch('toggle');
assert.deepEqual(mode.calls, { dirty: 0, persist: 0, share: 0 }, 'Restoring mode must not be treated as an edit.');
mode.selected('customize');
mode.advancedOptions.open = false;
mode.advancedOptions.dispatch('toggle');
assert.equal(mode.state.uiMode, 'basic');
assert.equal(mode.document.body.dataset.qrtoolUiMode, 'basic');
assert.deepEqual(mode.calls, { dirty: 1, persist: 1, share: 1 });
mode.advancedOptions.open = true;
mode.advancedOptions.dispatch('toggle');
assert.equal(mode.state.uiMode, 'advanced');
assert.deepEqual(mode.calls, { dirty: 2, persist: 2, share: 2 });
mode.api.setUiMode('invalid', { markDirty: true });
assert.equal(mode.state.uiMode, 'basic');
assert.equal(mode.advancedOptions.open, false);
assert.equal(mode.calls.dirty, 3);
assert(mode.buttons.every((button) => !button.hidden), 'Advanced mode must not hide workspace tabs.');

console.log('QR workspace tests passed: tab navigation, legacy aliases, disclosures, and persisted advanced mode.');
