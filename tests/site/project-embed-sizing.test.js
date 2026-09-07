'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function createHarness({ fit = 'content', maxHeight, compact = false, inaccessible = false } = {}) {
  let bottom = 1400;
  let screenHeight = 844;
  let hidden = false;
  let emptyHeight = 0;
  let nextTask = 0;
  const tasks = new Map();
  const observers = [];
  const attributes = new Map();
  const listeners = new Map();
  const style = () => ({
    setProperty(name, value) { this[name] = value; },
    getPropertyValue(name) { return this[name]; },
    removeProperty(name) { delete this[name]; }
  });
  const block = (height) => ({ getBoundingClientRect: () => ({ height }) });
  const header = block(80);
  const shell = { parentElement: null, querySelector: () => header, querySelectorAll: () => [frame] };
  const embed = {
    dataset: { embedFit: fit, ...(maxHeight ? { embedMaxHeight: String(maxHeight) } : {}) },
    classList: { contains: () => false }, style: style(), parentElement: shell,
    getBoundingClientRect: () => ({ width: hidden ? 0 : 1280 })
  };
  const chrome = { querySelector: (selector) => block(selector.includes('site-tab') ? 82 : 60) };
  const viewport = { clientHeight: 790, closest: () => chrome };
  const chatRoot = { matches: () => true };
  const chatMessages = { parentElement: chatRoot, matches: () => false };
  const childDocument = {
    body: { scrollHeight: 2400, offsetHeight: 2400, getBoundingClientRect: () => ({ height: 2400 }) },
    documentElement: { scrollHeight: 2400 },
    querySelector: (selector) => {
      if (selector === '.chat-shell--regular .empty-state') return emptyHeight ? { parentElement: chatMessages, getBoundingClientRect: () => ({ height: emptyHeight }) } : null;
      if (selector === '.demo-toolbar') return block(60);
      if (selector === '.chat-shell--regular .chat-composer') return block(73);
      return { getBoundingClientRect: () => ({ bottom }) };
    },
    defaultView: { scrollY: 0, getComputedStyle: (element) => element === chatMessages
      ? { paddingTop: '16px', paddingBottom: '16px' }
      : element === chatRoot ? { rowGap: '8px' } : { paddingBottom: '12px' } }
  };
  chatRoot.parentElement = childDocument.body;
  const frame = {
    dataset: {}, style: style(), parentElement: embed, contentWindow: {},
    closest: (selector) => ({ '.project-embed': embed, '.project-demo-shell': shell, '.site-frame__viewport': viewport })[selector],
    setAttribute: (name, value) => attributes.set(name, value),
    getAttribute: (name) => attributes.get(name),
    removeAttribute: (name) => attributes.delete(name),
    addEventListener: (name, callback) => listeners.set(name, callback),
    removeEventListener: (name) => listeners.delete(name)
  };
  Object.defineProperty(frame, 'contentDocument', { get() {
    if (inaccessible) throw new Error('Cross-origin');
    return childDocument;
  } });
  const schedule = (callback) => { const id = ++nextTask; tasks.set(id, callback); return id; };
  const window = {
    innerHeight: 1000,
    getComputedStyle: (element) => element === embed ? { display: hidden ? 'none' : 'block' } : {},
    matchMedia: (query) => ({ matches: query.includes('959px') && compact }),
    visualViewport: { get height() { return screenHeight; } }
  };
  const source = fs.readFileSync(path.join(__dirname, '../../js/common/common.js'), 'utf8');
  const start = source.indexOf('  const PROJECT_EMBED_MIN_HEIGHT_PX');
  const end = source.indexOf('  const resetPersonalProjectDetailScroll', start);
  assert(start >= 0 && end > start, 'Shared embed sizing must remain independently testable');
  const sandbox = {
    window, document: { querySelector: () => block(62) }, console,
    requestAnimationFrame: schedule, cancelAnimationFrame: (id) => tasks.delete(id),
    setTimeout: schedule, clearTimeout: (id) => tasks.delete(id),
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe() {}
      disconnect() { this.disconnected = true; }
    }
  };
  vm.runInNewContext(`${source.slice(start, end)}\nthis.api = { resizeProjectEmbedIframe, bindProjectEmbedResize, bindProjectEmbedLoading, syncProjectEmbedLoading, cleanupProjectEmbeds };`, sandbox);
  const root = { querySelectorAll: () => [frame] };
  const flush = () => {
    for (let i = 0; tasks.size && i < 8; i += 1) {
      const current = [...tasks.values()]; tasks.clear(); current.forEach((callback) => callback());
    }
    assert.strictEqual(tasks.size, 0, 'Sizing must settle without a feedback loop');
  };
  return { frame, embed, viewport, attributes, listeners, tasks, observers, flush,
    resize: () => sandbox.api.resizeProjectEmbedIframe(frame),
    bind: () => sandbox.api.bindProjectEmbedResize(root),
    bindLoading: () => sandbox.api.bindProjectEmbedLoading(root),
    syncLoading: () => sandbox.api.syncProjectEmbedLoading(root),
    cleanup: () => sandbox.api.cleanupProjectEmbeds(root),
    setBottom: (value) => { bottom = value; }, setScreenHeight: (value) => { screenHeight = value; },
    setHidden: (value) => { hidden = value; }, setEmptyHeight: (value) => { emptyHeight = value; }
  };
}

function runProjectEmbedSizingTests() {
  const content = createHarness();
  content.bind(); content.flush();
  assert.strictEqual(content.frame.style.height, '1412px', 'A content dashboard must not be capped at the previous 960px maximum');
  assert.strictEqual(content.attributes.get('scrolling'), 'no', 'The page viewport should own dashboard scrolling');
  content.setBottom(680); content.resize();
  assert.strictEqual(content.frame.style.height, '692px', 'Closing a disclosure should shrink despite the old root scrollHeight');
  content.setBottom(2100); content.listeners.get('load')(); content.listeners.get('load')();
  assert(content.tasks.size <= 4, 'Reloading a child must replace its old resize timers');
  content.cleanup(); content.flush();
  assert.strictEqual(content.listeners.size, 0, 'Unmounting must remove iframe listeners');
  assert(content.observers.every((observer) => observer.disconnected), 'Unmounting must disconnect all observers');
  assert.strictEqual(content.frame.style.height, '2112px', 'Cleanup should preserve outgoing paint without later resize writes');

  const limited = createHarness({ maxHeight: 720 });
  limited.resize();
  assert.strictEqual(limited.frame.style.height, '720px', 'An explicit per-project height limit remains supported');
  assert.strictEqual(limited.attributes.get('scrolling'), 'auto');

  const chat = createHarness({ fit: 'viewport' });
  chat.bind(); chat.flush();
  assert.strictEqual(chat.frame.style.height, '710px', 'Chat should subtract its demo toolbar from the persistent viewport');
  chat.setBottom(5000); chat.resize();
  assert.strictEqual(chat.frame.style.height, '710px', 'Messages must never grow a bounded chat viewport');
  chat.setEmptyHeight(140); chat.resize();
  assert.strictEqual(chat.frame.style.height, '313px', 'An empty chat should fit its starter prompts and composer without a tall blank panel');
  chat.setEmptyHeight(0); chat.resize();
  assert.strictEqual(chat.frame.style.height, '710px', 'Starting a conversation should restore its bounded message viewport');
  chat.viewport.clientHeight = 580; chat.resize();
  assert.strictEqual(chat.frame.style.height, '500px', 'Chat should follow persistent viewport resizing');
  chat.cleanup();

  const phone = createHarness({ fit: 'viewport', compact: true });
  phone.resize();
  assert.strictEqual(phone.frame.style.height, '560px', 'Compact chat should account for the masthead, horizontal tab, toolbar and demo header');
  phone.setScreenHeight(520); phone.resize();
  assert.strictEqual(phone.frame.style.height, '236px', 'Opening the keyboard should reduce the available chat viewport');

  const unavailable = createHarness({ inaccessible: true });
  unavailable.frame.style.height = '600px'; unavailable.resize();
  assert.strictEqual(unavailable.frame.style.height, '600px', 'An unavailable child must preserve its current fallback');
  const tableau = createHarness({ fit: 'dashboard' });
  tableau.bind(); tableau.flush();
  assert.strictEqual(tableau.frame.style.height, undefined, 'Tableau sizing must remain independent of content measurements');
  tableau.attributes.set('data-src', 'https://public.tableau.com/views/example');
  tableau.setHidden(true); tableau.bindLoading(); tableau.syncLoading();
  assert.strictEqual(tableau.attributes.get('src'), undefined, 'A dashboard launch card must defer the external embed');
  tableau.setHidden(false); tableau.observers.at(-1).callback(); tableau.flush();
  assert.strictEqual(tableau.attributes.get('src'), 'https://public.tableau.com/views/example', 'Making the shell wide enough must load the visible dashboard');
  tableau.setHidden(true); tableau.observers.at(-1).callback(); tableau.flush();
  assert.strictEqual(tableau.attributes.get('src'), undefined, 'Shrinking back to a launch card must unload the hidden dashboard');
  tableau.setHidden(false); tableau.observers.at(-1).callback(); tableau.cleanup(); tableau.flush();
  assert.strictEqual(tableau.attributes.get('src'), undefined, 'An unmounted route must not load a pending dashboard');
  assert(tableau.observers.every((observer) => observer.disconnected), 'Dashboard visibility observers must be cleaned up');
}

if (require.main === module) {
  runProjectEmbedSizingTests();
  process.stdout.write('Project embed sizing tests passed.\n');
}
module.exports = runProjectEmbedSizingTests;
