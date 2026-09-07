'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function createHarness({ microphoneSupported = true } = {}) {
  let activeElement = null;
  let captureRequests = 0;

  class Element {
    constructor(tagName = 'div') {
      this.tagName = tagName.toUpperCase();
      this.dataset = {};
      this.attributes = {};
      this.children = [];
      this.listeners = new Map();
      this.hidden = false;
      this.checked = false;
      this.value = '';
      this.style = {};
      this.textContent = '';
    }

    appendChild(child) {
      child.parentNode = this;
      this.children.push(child);
      return child;
    }

    set innerHTML(value) {
      this.children = [];
    }

    setAttribute(name, value) {
      this.attributes[name] = String(value);
    }

    getAttribute(name) {
      return this.attributes[name] ?? null;
    }

    addEventListener(type, listener) {
      const listeners = this.listeners.get(type) || [];
      listeners.push(listener);
      this.listeners.set(type, listeners);
    }

    dispatchEvent(event) {
      if (!event.target) event.target = this;
      (this.listeners.get(event.type) || []).forEach((listener) => listener(event));
      if (event.bubbles && this.parentNode) this.parentNode.dispatchEvent(event);
      return !event.defaultPrevented;
    }

    querySelectorAll(selector) {
      const name = /\[name="([^"]+)"\]/.exec(selector)?.[1];
      const descendants = this.children.flatMap((child) => [child, ...child.querySelectorAll('*')]);
      if (selector === '*') return descendants;
      return descendants.filter((child) => child.tagName === 'INPUT' && (!name || child.name === name));
    }

    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }

    focus() {
      activeElement = this;
    }
  }

  const document = new Element('document');
  const body = new Element('body');
  const elements = new Map();
  document.body = body;
  document.appendChild(body);
  const make = (name, tag = 'div', value = '') => {
    const element = new Element(tag);
    element.value = value;
    element.dataset.screenrec = name;
    elements.set(`[data-screenrec="${name}"]`, element);
    body.appendChild(element);
    return element;
  };
  const controls = make('controls-body');
  const tabs = ['audio', 'video', 'output'].map((name) => {
    const tab = new Element('button');
    tab.dataset.screenrecTab = name;
    body.appendChild(tab);
    return tab;
  });
  const panels = tabs.map((tab) => {
    const panel = new Element();
    panel.dataset.screenrecPanel = tab.dataset.screenrecTab;
    controls.appendChild(panel);
    return panel;
  });
  const setting = (name, tag, value, panel = 0) => {
    const element = make(name, tag, value);
    panels[panel].appendChild(element);
    return element;
  };
  make('start-capture', 'button');
  make('start-record', 'button');
  make('video', 'video');
  make('settings-summary');
  make('output-summary');
  setting('audio-toggle', 'input');
  setting('mic-toggle', 'input');
  setting('system-audio-details');
  setting('microphone-details');
  setting('audio-status');
  setting('audio-meter');
  setting('audio-meter-fill');
  setting('fps-select', 'select', '15', 1);
  setting('quality-select', 'select', 'low', 1);
  setting('scale-select', 'select', '0.75', 1);
  setting('format-options', 'div', '', 2);
  setting('image-format-options', 'div', '', 2);
  document.querySelector = (selector) => elements.get(selector) || null;
  document.querySelectorAll = (selector) => {
    if (selector === '[data-screenrec-tab]') return tabs;
    if (selector === '[data-screenrec-panel]') return panels;
    return [];
  };
  document.createElement = (tag) => {
    const element = new Element(tag);
    if (tag === 'canvas') element.toDataURL = () => 'data:image/webp;base64,';
    return element;
  };
  const window = new Element('window');
  const MediaRecorder = { isTypeSupported: () => true };
  window.MediaRecorder = MediaRecorder;
  const navigator = {
    mediaDevices: {
      addEventListener() {},
      getDisplayMedia: async () => {
        captureRequests += 1;
        throw new Error('Capture should not start during settings changes.');
      },
      getUserMedia: async () => {
        throw new Error('Microphone is unavailable in this test.');
      }
    }
  };
  if (!microphoneSupported) delete navigator.mediaDevices.getUserMedia;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/tools/screen-recorder.js'), 'utf8'), {
    document, window, navigator, MediaRecorder, TextEncoder, Blob, URL, setTimeout, clearTimeout
  }, { filename: 'screen-recorder.js' });
  const get = (name) => elements.get(`[data-screenrec="${name}"]`);
  const fire = (element, type, extra = {}) => {
    const event = {
      type,
      bubbles: true,
      defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; },
      ...extra
    };
    element.dispatchEvent(event);
    return event;
  };
  return { document, tabs, panels, get, fire, activeElement: () => activeElement, captureRequests: () => captureRequests };
}

function run() {
  const h = createHarness();
  const selected = (index) => {
    h.tabs.forEach((tab, current) => {
      assert.strictEqual(tab.getAttribute('aria-selected'), String(current === index));
      assert.strictEqual(tab.getAttribute('tabindex'), current === index ? '0' : '-1');
      assert.strictEqual(h.panels[current].hidden, current !== index);
    });
  };
  selected(0);
  assert.strictEqual(h.get('settings-summary').textContent, '15 fps · Small quality · 75% scale');
  assert.strictEqual(h.get('output-summary').textContent, 'Output: Auto');
  assert.strictEqual(h.get('system-audio-details').hidden, true);
  assert.strictEqual(h.get('microphone-details').hidden, true);
  assert.strictEqual(h.get('audio-status').hidden, true);
  assert.strictEqual(h.get('audio-meter').hidden, true);

  h.fire(h.tabs[0], 'keydown', { key: 'ArrowLeft' });
  selected(2);
  assert.strictEqual(h.activeElement(), h.tabs[2]);
  h.fire(h.tabs[2], 'keydown', { key: 'ArrowRight' });
  selected(0);
  h.fire(h.tabs[0], 'keydown', { key: 'End' });
  selected(2);
  h.fire(h.tabs[2], 'keydown', { key: 'Home' });
  selected(0);
  const unrelatedKey = h.fire(h.tabs[0], 'keydown', { key: 'Tab' });
  assert.strictEqual(unrelatedKey.defaultPrevented, false);

  h.get('fps-select').value = '30';
  h.get('quality-select').value = 'medium';
  h.get('scale-select').value = '0.5';
  h.fire(h.get('quality-select'), 'change');
  assert.strictEqual(h.get('settings-summary').textContent, '30 fps · Balanced quality · 50% scale');
  const formats = h.get('format-options').querySelectorAll('input');
  const webm = formats.find((input) => input.value === 'video/webm;codecs=vp9');
  webm.checked = true;
  h.fire(webm, 'change');
  assert.strictEqual(h.get('output-summary').textContent, 'Output: WebM (VP9)');
  const png = h.get('image-format-options').querySelectorAll('input')[0];
  png.checked = true;
  h.fire(png, 'change');
  assert.strictEqual(h.get('output-summary').textContent, 'Output: WebM (VP9) · PNG first frame');
  h.get('audio-toggle').checked = true;
  h.fire(h.get('audio-toggle'), 'change');
  assert.strictEqual(h.get('system-audio-details').hidden, false);
  assert.strictEqual(h.get('audio-status').hidden, false);
  assert.match(h.get('audio-status').textContent, /System audio appears after you start capture/);
  assert.strictEqual(h.get('audio-meter').hidden, true);
  h.fire(h.tabs[1], 'click');
  selected(1);
  h.fire(h.tabs[2], 'click');
  selected(2);
  assert.strictEqual(h.get('fps-select').value, '30');
  assert.strictEqual(webm.checked, true);
  assert.strictEqual(png.checked, true);
  assert.strictEqual(h.get('audio-toggle').checked, true);

  // Account restore writes the existing controls before dispatching session-applied.
  h.get('fps-select').value = '60';
  h.get('audio-toggle').checked = false;
  h.get('mic-toggle').checked = true;
  h.fire(h.document, 'tools:session-applied', { detail: { toolId: 'screen-recorder' } });
  selected(2);
  assert.strictEqual(h.get('settings-summary').textContent, '60 fps · Balanced quality · 50% scale');
  assert.strictEqual(h.get('system-audio-details').hidden, true);
  assert.strictEqual(h.get('microphone-details').hidden, false);
  assert.strictEqual(webm.checked, true);
  assert.strictEqual(png.checked, true);
  assert.strictEqual(h.captureRequests(), 0);

  h.get('mic-toggle').checked = false;
  h.fire(h.document, 'tools:session-applied', { detail: { toolId: 'screen-recorder' } });
  assert.strictEqual(h.get('audio-status').hidden, true);
  assert.strictEqual(h.get('audio-meter').hidden, true);
  webm.checked = false;
  h.fire(webm, 'change');
  assert.strictEqual(h.get('output-summary').textContent, 'Output: Auto · PNG first frame');

  const unsupported = createHarness({ microphoneSupported: false });
  assert.strictEqual(unsupported.get('audio-status').hidden, false);
  assert.strictEqual(unsupported.get('audio-status').textContent, 'Microphone capture is not supported in this browser.');
  assert.strictEqual(unsupported.get('audio-meter').hidden, true);
}

module.exports = run;
if (require.main === module) {
  run();
  console.log('Screen Recorder settings tests passed.');
}
