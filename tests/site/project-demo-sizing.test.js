'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, callback) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(callback);
    },
    removeEventListener(type, callback) { listeners.get(type)?.delete(callback); },
    emit(type) { [...(listeners.get(type) || [])].forEach((callback) => callback()); },
    listenerCount() { return [...listeners.values()].reduce((total, set) => total + set.size, 0); }
  };
}

function createHarness({ compact = true, fit = 'content', inaccessible = false } = {}) {
  let height = 800;
  let nextId = 0;
  const tasks = new Map();
  const cleanups = [];
  const observers = [];
  const attributes = new Map();
  const element = () => ({ style: { removeProperty(name) { delete this[name]; } } });
  const main = element();
  const container = element();
  const frame = Object.assign(element(), eventTarget(), {
    dataset: { projectDemoSrc: '/demos/example.html', projectDemoFit: fit },
    closest: (selector) => selector === '.project-demo-wrapper-main' ? main : container,
    getAttribute: () => '/demos/example.html'
  });
  const document = Object.assign(eventTarget(), {
    readyState: 'loading',
    body: {
      setAttribute: (key, value) => attributes.set(key, value),
      removeAttribute: (key) => attributes.delete(key)
    }
  });
  const media = Object.assign(eventTarget(), { matches: compact });
  const schedule = (callback) => { const id = ++nextId; tasks.set(id, callback); return id; };
  const window = {
    location: { href: 'https://example.test/example?model=small#draw' },
    matchMedia: () => media,
    requestAnimationFrame: schedule,
    cancelAnimationFrame: (id) => tasks.delete(id),
    setTimeout: schedule,
    clearTimeout: (id) => tasks.delete(id)
  };
  const content = { getBoundingClientRect: () => ({ bottom: height }) };
  frame.contentDocument = inaccessible ? null : {
    body: { scrollHeight: 1800 },
    documentElement: { scrollHeight: 1800 },
    querySelector: () => content
  };
  frame.contentWindow = {
    scrollY: 0,
    getComputedStyle: () => ({ paddingBottom: '0px' }),
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe() {}
      disconnect() { this.disconnected = true; }
    }
  };
  const source = fs.readFileSync(path.join(__dirname, '../../js/navigation/project-demo-wrapper.js'), 'utf8');
  vm.runInNewContext(source, { window, document, URL, console });
  const controller = new AbortController();
  window.ProjectDemoWrapper.mount({
    root: { querySelector: () => frame },
    url: window.location.href,
    signal: controller.signal,
    cleanup: (callback) => cleanups.push(callback)
  });
  const flush = () => {
    for (let i = 0; tasks.size && i < 10; i += 1) {
      const pending = [...tasks.values()];
      tasks.clear();
      pending.forEach((callback) => callback());
    }
    if (tasks.size) throw new Error('Demo sizing did not settle');
  };
  const resize = (nextHeight) => {
    height = nextHeight;
    observers.filter((observer) => !observer.disconnected).forEach((observer) => observer.callback());
    flush();
  };
  return { frame, main, container, media, attributes, observers, controller, flush, resize,
    cleanup: () => cleanups.splice(0).reverse().forEach((callback) => callback()) };
}

module.exports = function runProjectDemoSizingTests({ assert }) {
  const mobile = createHarness();
  mobile.flush();
  assert(mobile.frame.src === '/demos/example.html?model=small#draw', 'Demo sizing should preserve route query and fragment');
  assert([mobile.frame, mobile.main, mobile.container].every((element) => element.style.height === '800px'),
    'A compact demo should size its entire iframe chain to the workspace');
  mobile.resize(1400);
  assert(mobile.frame.style.height === '1400px', 'Opening details or receiving results should grow the mobile frame');
  mobile.resize(700);
  assert(mobile.frame.style.height === '700px', 'Closing details should shrink the frame even when the old document scrollHeight is larger');
  mobile.resize(300);
  assert(mobile.frame.style.height === '560px', 'Short content should retain a usable minimum demo height');
  mobile.media.matches = false;
  mobile.media.emit('change');
  assert(!mobile.frame.style.height && !mobile.attributes.has('data-project-demo-autosize'),
    'Returning to desktop should release inline sizing to the fixed frame layout');
  mobile.media.matches = true;
  mobile.media.emit('change');
  mobile.flush();
  mobile.controller.abort();
  mobile.resize(1800);
  assert(mobile.frame.style.height === '560px', 'An aborted route must not resize the next page');
  mobile.cleanup();
  assert(!mobile.frame.style.height && mobile.frame.listenerCount() === 0 && mobile.media.listenerCount() === 0 &&
    mobile.observers.every((observer) => observer.disconnected), 'Route cleanup should remove heights, listeners, and observers');

  const chat = createHarness({ fit: 'viewport' });
  chat.flush();
  assert(!chat.frame.style.height && chat.observers.length === 0,
    'A bounded chat viewport must not enter a content-height feedback loop');
  chat.cleanup();
  const desktop = createHarness({ compact: false });
  desktop.flush();
  assert(!desktop.frame.style.height && desktop.observers.length === 0, 'Desktop uses CSS sizing without content-height observers');
  desktop.cleanup();
  const unavailable = createHarness({ inaccessible: true });
  unavailable.flush();
  assert(!unavailable.frame.style.height, 'An unavailable iframe document should preserve CSS fallback sizing');
  unavailable.cleanup();
};
