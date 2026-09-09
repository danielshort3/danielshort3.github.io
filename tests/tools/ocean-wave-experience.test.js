'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/ocean-wave-experience.js'), 'utf8');

class Element {
  constructor(tagName = 'div') {
    this.tagName = tagName.toUpperCase();
    this.attributes = new Map();
    this.listeners = new Map();
    this.children = [];
    this.dataset = {};
    this.style = { overflow: '', setProperty(name, value) { this[name] = value; } };
    this.hidden = false;
    this.isConnected = true;
    this.value = '';
    this.textContent = '';
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      contains: name => classes.has(name),
      toggle: (name, force) => {
        const active = force === undefined ? !classes.has(name) : Boolean(force);
        if (active) classes.add(name);
        else classes.delete(name);
        return active;
      },
      toString: () => [...classes].join(' ')
    };
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
  appendChild(node) {
    node.remove();
    this.children.push(node);
    node.parentNode = this;
    return node;
  }
  append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
  insertBefore(node, reference) {
    if (!reference) return this.appendChild(node);
    node.remove();
    this.children.splice(this.children.indexOf(reference), 0, node);
    node.parentNode = this;
    return node;
  }
  remove() {
    if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
    this.parentNode = null;
  }
  before(node) { this.parentNode.insertBefore(node, this); }
  replaceWith(node) { this.before(node); this.remove(); }
  getBoundingClientRect() { return { width: 1200, height: 720 }; }
  get nextSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }
  removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); }
  click() { this.dispatchEvent({ type: 'click' }); }
  dispatchEvent(event) {
    event.target ??= this;
    event.currentTarget = this;
    event.preventDefault ??= () => { event.defaultPrevented = true; };
    event.stopPropagation ??= () => {};
    for (const listener of [...this.listeners.get(event.type) || []]) listener(event);
    return !event.defaultPrevented;
  }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  closest(selector) {
    return selector.split(',').some(name => this.tagName === name.trim().toUpperCase()) ? this : null;
  }
}

const flush = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};

const makeHarness = ({ audioAvailable = true, nativeFullscreen = false, rejectFullscreen = false, delayFullscreen = false } = {}) => {
  const document = new Element('document');
  document.body = new Element('body');
  document.documentElement = new Element('html');
  document.fullscreenElement = null;
  document.hidden = false;
  document.activeElement = document.body;
  const ids = [
    'stage', 'canvas', 'fullscreen', 'sound', 'settings-toggle', 'settings-close', 'settings',
    'volume', 'volume-value', 'scene-name', 'status', 'toggle', 'reset', 'wind', 'height', 'light', 'quality', 'controls',
    'timer', 'timer-status', 'rest-screen', 'resume'
  ];
  const elements = Object.fromEntries(ids.map(id => [id, new Element(id.includes('toggle') || ['fullscreen', 'sound', 'reset'].includes(id) ? 'button' : 'div')]));
  for (const [id, element] of Object.entries(elements)) {
    element.id = `ocean-wave-${id}`;
    element.focus = () => { document.activeElement = element; };
    element.querySelector = selector => ['svg', 'path', '[data-ocean-sound-waves]'].includes(selector)
      ? (element.icon ||= new Element('path')) : null;
    element.querySelectorAll = () => [];
  }
  elements.settings.hidden = true;
  elements['rest-screen'].hidden = true;
  elements['timer-status'].hidden = true;
  elements.timer.value = '0';
  elements['settings-toggle'].setAttribute('aria-expanded', 'false');
  elements.fullscreen.setAttribute('aria-pressed', 'false');
  elements.sound.setAttribute('aria-pressed', 'false');
  elements.volume.value = '35';
  elements.stage.append(...Object.values(elements).filter(element => element !== elements.stage));
  const hud = new Element();
  hud.append(elements.fullscreen, elements.sound, elements['settings-toggle'], elements.toggle);
  elements.stage.append(hud);
  elements.stage.querySelector = selector => selector === '.ocean-wave-hud'
    ? hud : elements[selector.replace('#ocean-wave-', '')] || null;
  const wrapper = new Element();
  wrapper.append(elements.stage);
  document.body.append(wrapper);
  document.createComment = () => new Element();
  document.createElement = tagName => new Element(tagName);
  document.querySelector = selector => elements[selector.replace('#ocean-wave-', '')] || null;
  document.getElementById = id => elements[id.replace('ocean-wave-', '')] || null;
  document.querySelectorAll = () => [];
  const timers = new Map();
  let timerSequence = 0;
  const cleanups = [];
  const contexts = [];
  const window = new Element('window');
  let nowMs = 1000;
  const saved = new Map();
  window.localStorage = { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) };
  window.setTimeout = (callback, delay) => {
    const id = ++timerSequence;
    timers.set(id, { callback, delay });
    return id;
  };
  window.clearTimeout = id => timers.delete(id);
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.SiteRoutes = { addCleanup: callback => cleanups.push(callback) };
  const param = () => ({ value: 0, setValueAtTime(value) { this.value = value; }, setTargetAtTime(value) { this.value = value; }, linearRampToValueAtTime(value) { this.value = value; }, cancelScheduledValues() {} });
  const node = () => ({
    gain: param(), frequency: param(), Q: param(),
    connect(target) { return target; }, disconnect() {}, start() {}, stop() {},
    playbackRate: param(), pan: param()
  });
  class AudioContext {
    constructor() {
      this.state = 'suspended';
      this.sampleRate = 32;
      this.currentTime = 0;
      this.destination = {};
      this.resumes = 0;
      this.suspends = 0;
      this.closes = 0;
      contexts.push(this);
    }
    createGain() { return node(); }
    createBiquadFilter() { return node(); }
    createBufferSource() { return node(); }
    createOscillator() { return node(); }
    createStereoPanner() { return node(); }
    createBuffer(channels, length) { return { getChannelData: () => new Float32Array(length) }; }
    async resume() { this.resumes++; this.state = 'running'; }
    async suspend() { this.suspends++; this.state = 'suspended'; }
    async close() { this.closes++; this.state = 'closed'; }
  }
  if (audioAvailable) window.AudioContext = AudioContext;
  window.OceanWaveAudio = { create: ({ onStatus, onError }) => {
    let audio = null;
    let enabled = false;
    let visible = true;
    return {
      async setEnabled(value) {
        enabled = value;
        if (enabled && !audioAvailable) { enabled = false; onStatus({ state: 'error', enabled }); onError(new Error('Unavailable')); return false; }
        if (enabled && !audio) audio = new AudioContext();
        if (audio) { if (enabled && visible) await audio.resume(); else await audio.suspend(); }
        onStatus({ state: enabled ? 'playing' : 'off', enabled });
        return enabled;
      },
      setVisible(value) { visible = value; if (audio) { if (enabled && visible) audio.resume(); else audio.suspend(); } },
      setVolume() {}, setScene() {}, setFade() {}, dispose() { audio?.close(); },
    };
  } };
  let finishFullscreen;
  if (nativeFullscreen || rejectFullscreen) {
    elements.stage.requestFullscreen = async () => {
      if (delayFullscreen) await new Promise(resolve => { finishFullscreen = resolve; });
      if (rejectFullscreen) throw new Error('Fullscreen denied');
      document.fullscreenElement = elements.stage;
      document.dispatchEvent({ type: 'fullscreenchange' });
    };
    document.exitFullscreen = async () => {
      document.fullscreenElement = null;
      document.dispatchEvent({ type: 'fullscreenchange' });
    };
  }
  const context = { window, document, console, Date: { now: () => nowMs },
    Event: class { constructor(type) { this.type = type; } },
    CustomEvent: class { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } },
    setTimeout: window.setTimeout, clearTimeout: window.clearTimeout };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/tools/ocean-wave-timer.js'), 'utf8'), context);
  vm.runInNewContext(source, context, { filename: 'ocean-wave-experience.js' });
  const fire = async (target, type, event = {}) => {
    target.dispatchEvent({ type, ...event });
    await flush();
  };
  return {
    document, window, elements, contexts, timers, cleanups, fire, saved,
    advance: milliseconds => { nowMs += milliseconds; },
    finishFullscreen: async () => { finishFullscreen?.(); await flush(); },
    click: id => fire(elements[id], 'click'),
    cleanup: async () => {
      for (const callback of cleanups) await callback();
      await flush();
    },
    runTimers: async () => {
      const pending = [...timers.values()];
      timers.clear();
      for (const { callback } of pending) callback();
      await flush();
    }
  };
};

const run = async () => {
  {
    const app = makeHarness();
    assert.equal(app.contexts.length, 0, 'Opening the ocean must not create an audio context or autoplay sound.');
    await app.click('settings-toggle');
    assert.equal(app.elements.settings.hidden, false, 'Settings must open on request.');
    assert.equal(app.elements['settings-toggle'].getAttribute('aria-expanded'), 'true');
    await app.click('settings-toggle');
    assert.equal(app.elements.settings.hidden, true, 'Settings must collapse without leaving the experience.');
    assert.equal(app.contexts.length, 0, 'Changing settings must not implicitly start audio.');
    await app.cleanup();
  }

  for (const options of [{}, { nativeFullscreen: true }, { rejectFullscreen: true }]) {
    const app = makeHarness(options);
    await app.click('fullscreen');
    assert.equal(app.elements.fullscreen.getAttribute('aria-pressed'), 'true', 'Fullscreen must work with native API, absent API, and denied native requests.');
    if (options.nativeFullscreen) assert.equal(app.document.fullscreenElement, app.elements.stage);
    await app.click('fullscreen');
    assert.equal(app.elements.fullscreen.getAttribute('aria-pressed'), 'false', 'The fullscreen control must exit immersion.');
    assert.equal(app.document.fullscreenElement, null);
    assert.equal(app.document.body.style.overflow, '', 'Exiting full screen must restore page scrolling.');
    await app.cleanup();
  }

  {
    const app = makeHarness({ nativeFullscreen: true });
    await app.click('fullscreen');
    await app.click('settings-toggle');
    await app.fire(app.elements.stage, 'keydown', { key: 'Escape' });
    assert.equal(app.elements.settings.hidden, true, 'Escape must close settings before leaving native fullscreen.');
    assert.equal(app.document.fullscreenElement, app.elements.stage);
    await app.fire(app.elements.stage, 'keydown', { key: 'Escape' });
    assert.equal(app.document.fullscreenElement, null, 'Escape must explicitly exit native fullscreen without relying on browser chrome shortcuts.');
    assert.equal(app.elements.fullscreen.getAttribute('aria-pressed'), 'false');
    assert.equal(app.document.activeElement, app.elements.stage, 'Native fullscreen exit must return focus to the ocean.');
    await app.cleanup();
  }

  {
    const app = makeHarness({ nativeFullscreen: true });
    await app.click('fullscreen');
    app.document.exitFullscreen = async () => { throw new Error('Fullscreen exit denied'); };
    await app.fire(app.elements.stage, 'keydown', { key: 'Escape' });
    assert.equal(app.document.fullscreenElement, app.elements.stage, 'A denied exit must preserve the actual fullscreen state.');
    assert.equal(app.elements.status.textContent, 'Fullscreen could not be changed. Try again.');
    await app.cleanup();
  }

  {
    const app = makeHarness();
    const initialParent = app.elements.stage.parentNode;
    app.document.body.style.overflow = 'clip';
    await app.click('fullscreen');
    assert.equal(app.elements.stage.parentNode, app.document.body, 'Fallback must escape clipping from shared site panels.');
    await app.runTimers();
    assert.equal(app.elements.stage.classList.contains('is-idle'), true, 'Fullscreen controls should fade after inactivity.');
    await app.fire(app.elements.stage, 'pointermove');
    assert.equal(app.elements.stage.classList.contains('is-idle'), false, 'Pointer movement must reveal the exit control.');
    await app.click('settings-toggle');
    await app.runTimers();
    assert.equal(app.elements.stage.classList.contains('is-idle'), false, 'Controls must remain visible while settings are open.');
    await app.fire(app.elements.stage, 'keydown', { key: 'Escape' });
    assert.equal(app.elements.settings.hidden, true, 'Escape must close settings before leaving fullscreen.');
    assert.equal(app.elements.fullscreen.getAttribute('aria-pressed'), 'true');
    await app.fire(app.elements.stage, 'keydown', { key: 'Escape' });
    assert.equal(app.elements.fullscreen.getAttribute('aria-pressed'), 'false', 'Escape must leave fallback fullscreen.');
    assert.equal(app.elements.stage.parentNode, initialParent, 'Leaving fullscreen must restore the stage in its original site panel.');
    assert.equal(app.document.body.style.overflow, 'clip', 'Fullscreen must restore the existing scroll policy.');
    await app.cleanup();
  }

  {
    const app = makeHarness();
    await app.click('sound');
    assert.equal(app.contexts.length, 1, 'The sound gesture should lazily create one audio context.');
    assert.equal(app.contexts[0].state, 'running');
    assert.equal(app.elements.sound.getAttribute('aria-pressed'), 'true');
    app.document.hidden = true;
    await app.fire(app.document, 'visibilitychange');
    await app.runTimers();
    assert.equal(app.contexts[0].state, 'suspended', 'A hidden ocean must stop consuming audio resources.');
    app.document.hidden = false;
    await app.fire(app.document, 'visibilitychange');
    assert.equal(app.contexts[0].state, 'running', 'Returning should resume sound only when it was enabled.');
    await app.click('sound');
    await app.runTimers();
    assert.equal(app.elements.sound.getAttribute('aria-pressed'), 'false');
    assert.equal(app.contexts[0].state, 'suspended', 'Turning sound off must suspend the audio context.');
    app.document.hidden = true;
    await app.fire(app.document, 'visibilitychange');
    app.document.hidden = false;
    await app.fire(app.document, 'visibilitychange');
    assert.equal(app.contexts[0].state, 'suspended', 'Tab visibility must not re-enable sound after the user switches it off.');
    await app.click('sound');
    assert.equal(app.contexts.length, 1, 'Toggling sound should reuse its context.');
    await app.click('fullscreen');
    assert(app.cleanups.length > 0, 'The experience must register cleanup for soft navigation.');
    await app.cleanup();
    assert.equal(app.contexts[0].state, 'closed', 'Leaving the ocean must release its audio context.');
    assert.equal(app.document.body.style.overflow, '', 'Soft navigation must release fullscreen scroll locking.');
    assert.equal(app.timers.size, 0, 'Soft navigation must cancel pending UI and sound timers.');
  }

  {
    const app = makeHarness({ audioAvailable: false });
    await app.click('sound');
    assert.equal(app.contexts.length, 0);
    assert.equal(app.elements.sound.getAttribute('aria-pressed'), 'false', 'Unsupported audio must never appear enabled.');
    await app.cleanup();
  }

  {
    const app = makeHarness({ rejectFullscreen: true, delayFullscreen: true });
    await app.click('fullscreen');
    await app.cleanup();
    await app.finishFullscreen();
    assert.equal(app.elements.stage.classList.contains('is-fullscreen'), false, 'A delayed fullscreen rejection must not enter fallback after navigating away.');
    assert.equal(app.document.body.style.overflow, '');
    assert.equal(app.timers.size, 0, 'A finished fullscreen request must not revive disposed timers.');
  }

  {
    const app = makeHarness();
    await app.click('sound');
    app.document.hidden = true;
    await app.fire(app.window, 'pagehide', { persisted: true });
    assert.equal(app.contexts[0].state, 'suspended', 'A cached page must suspend sound while preserving its controls for browser Back.');
    app.document.hidden = false;
    await app.fire(app.window, 'pageshow', { persisted: true });
    assert.equal(app.contexts[0].state, 'running', 'Browser Back should restore the enabled sound after cache suspension.');
    await app.click('settings-toggle');
    assert.equal(app.elements.settings.hidden, false, 'The restored page must retain working controls.');
    await app.cleanup();
  }

  {
    const app = makeHarness();
    let rests = 0;
    let resumes = 0;
    app.elements.stage.addEventListener('ocean:rest', () => rests++);
    app.elements.stage.addEventListener('ocean:resume', () => resumes++);
    app.elements.volume.value = '24';
    await app.fire(app.elements.volume, 'input');
    assert.equal(app.saved.get('ds-ocean-volume-v1'), '24', 'Volume preference must survive future visits.');
    await app.click('sound');
    app.elements.timer.value = '15';
    await app.fire(app.elements.timer, 'change');
    assert.equal(app.elements['timer-status'].hidden, false);
    app.advance(870000);
    await app.runTimers();
    assert.equal(app.elements.stage.style['--ocean-rest-fade'], '0.5');
    app.advance(30000);
    await app.runTimers();
    assert.equal(rests, 1, 'The timer must stop the animation through its public event.');
    assert.equal(app.elements['rest-screen'].hidden, false);
    assert.equal(app.contexts[0].state, 'suspended', 'Finishing must suspend audio, not just make it inaudible.');
    assert.equal(app.document.activeElement, app.elements.resume);
    await app.click('resume');
    assert.equal(resumes, 1);
    assert.equal(app.elements['rest-screen'].hidden, true);
    assert.equal(app.elements.stage.style['--ocean-rest-fade'], '0');
    assert.equal(app.contexts[0].state, 'running', 'A Resume gesture may restore sound that was previously enabled.');
    assert.equal(app.elements.timer.value, '0', 'Resume must not silently restart the sleep timer.');
    await app.cleanup();
  }

  console.log('Ocean experience tests passed: sound, preferences, timer fade/resume, visibility, fullscreen, route cleanup, and browser Back restoration.');
};

run().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
