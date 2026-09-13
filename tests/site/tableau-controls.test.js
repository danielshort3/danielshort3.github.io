'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const read = (file) => fs.readFileSync(path.join(__dirname, '../..', file), 'utf8');
const controlsSource = read('js/portfolio/tableau-controls.js');

function dashboard(device = 'desktop') {
  const listeners = new Set();
  const attributes = new Map([
    ['data-dashboard-default-src', 'https://public.tableau.com/views/Book/Overview?:embed=y&:device=desktop&:iid=8&City=The%20Colony'],
    ['data-src', 'https://public.tableau.com/views/Book/Overview'],
    ['data-dashboard-device', device]
  ]);
  const loads = [];
  const frame = {
    dataset: {},
    getAttribute: (name) => attributes.get(name),
    removeAttribute: (name) => attributes.delete(name),
    setAttribute(name, value) { attributes.set(name, value); if (name === 'src') loads.push(value); }
  };
  Object.defineProperty(frame.dataset, 'dashboardDevice', {
    get: () => attributes.get('data-dashboard-device'),
    set: value => attributes.set('data-dashboard-device', value)
  });
  const button = {
    dataset: {}, hidden: true,
    closest: () => ({ querySelector: () => frame }),
    addEventListener: (_, callback) => listeners.add(callback),
    removeEventListener: (_, callback) => listeners.delete(callback)
  };
  const root = {
    querySelector: (selector) => selector === '[data-dashboard-reset]' ? button : null,
    querySelectorAll: (selector) => selector === '[data-dashboard-reset]' ? [button] : []
  };
  return { root, button, frame, attributes, loads, listeners, click: () => [...listeners].forEach((callback) => callback()) };
}

function harness(readyState = 'complete') {
  let root = { querySelector: () => null, querySelectorAll: () => [] };
  let cleanups = [];
  const events = new Map();
  const registrations = new Map();
  const loads = [];
  const document = {
    readyState, baseURI: 'https://example.test/',
    querySelector: () => root,
    querySelectorAll: (selector) => root.querySelectorAll(selector),
    addEventListener(name, callback) {
      if (!events.has(name)) events.set(name, new Set());
      events.get(name).add(callback);
    }
  };
  const window = { SiteRoutes: {
    addCleanup: (callback) => cleanups.push(callback),
    register: (name, lifecycle) => registrations.set(name, lifecycle)
  } };
  const sandbox = vm.createContext({
    window, document, URL,
    mountSharedContent() {},
    async loadScriptOnce(file) { loads.push(file); vm.runInContext(controlsSource, sandbox); }
  });
  return {
    window, document, loads, registrations,
    setRoot(value) { root = value; },
    evaluate: (source) => vm.runInContext(source, sandbox),
    emit: (name) => [...(events.get(name) || [])].forEach((callback) => callback({ detail: { root } })),
    context: (value, aborted = false) => ({ root: value, signal: { aborted }, cleanup: (callback) => cleanups.push(callback) }),
    cleanup() { const pending = cleanups; cleanups = []; pending.reverse().forEach((callback) => callback()); }
  };
}

async function runTableauControlsTests() {
  const direct = harness('loading');
  const first = dashboard();
  direct.setRoot(first.root);
  direct.evaluate(controlsSource);
  assert.equal(first.button.hidden, true, 'direct loading must wait for DOM readiness');
  direct.emit('DOMContentLoaded');
  direct.emit('site:content-updated');
  assert.equal(first.button.hidden, false);
  assert.equal(first.listeners.size, 1, 'duplicate readiness events must not duplicate the click handler');
  first.frame.setAttribute('src', 'https://public.tableau.com/views/Book/Other?City=Changed');
  first.click();
  const resetUrl = new URL(first.loads.at(-1));
  assert.equal(resetUrl.pathname, '/views/Book/Overview');
  assert.equal(resetUrl.searchParams.get(':revert'), 'all');
  assert.equal(resetUrl.searchParams.has(':iid'), false);
  assert.equal(resetUrl.searchParams.get(':device'), 'desktop');
  assert.equal(resetUrl.searchParams.get('City'), 'The Colony');
  assert.equal(first.attributes.has('data-src'), false, 'reset must clear the deferred source');
  direct.cleanup();
  assert.equal(first.listeners.size, 0);
  assert.equal(first.button.hidden, true);

  const mobile = harness('loading');
  const phone = dashboard('phone');
  mobile.setRoot(phone.root);
  mobile.evaluate(controlsSource);
  mobile.emit('DOMContentLoaded');
  phone.frame.setAttribute('src', 'https://public.tableau.com/views/Book/Other?:device=phone&City=Changed');
  const phoneInitialLoads = phone.loads.length;
  phone.click();
  const phoneResetUrl = new URL(phone.loads.at(-1));
  assert.equal(phoneResetUrl.pathname, '/views/Book/Overview', 'Phone reset restores the canonical Overview.');
  assert.equal(phoneResetUrl.searchParams.get(':device'), 'phone', 'Phone reset must not force the original desktop source.');
  assert.equal(phoneResetUrl.searchParams.get(':revert'), 'all');
  assert.equal(phoneResetUrl.searchParams.get('City'), 'The Colony', 'Reset restores published filters in the active device.');
  assert.equal(phoneResetUrl.searchParams.has(':iid'), false);
  phone.frame.setAttribute('data-dashboard-device', 'desktop');
  phone.click();
  assert.equal(new URL(phone.loads.at(-1)).searchParams.get(':device'), 'desktop', 'A later breakpoint change is read at click time, not captured at mount.');
  assert.equal(phone.loads.length, phoneInitialLoads + 2, 'Each reset writes one source.');
  mobile.cleanup();
  assert.equal(phone.listeners.size, 0);
  assert.equal(phone.button.hidden, true);

  const registered = harness();
  const common = read('js/common/common.js');
  const start = common.indexOf('  const preloadSharedContent =');
  const end = common.indexOf("    window.SiteRoutes.register('portfolio:workbench'", start);
  assert(start >= 0 && end > start);
  registered.evaluate(common.slice(start, end) + '\n}');
  const lifecycle = registered.registrations.get('page:content');
  const second = dashboard();
  await lifecycle.preload({ document: second.root });
  assert.deepEqual(registered.loads, ['js/portfolio/tableau-controls.js'], 'soft-entry preload must load the dashboard helper');
  assert.equal(second.button.hidden, true, 'preloading must not bind detached destination controls');
  registered.setRoot(second.root);
  await lifecycle.mount(registered.context(second.root));
  registered.emit('site:content-updated');
  assert.equal(second.listeners.size, 1);
  second.click();
  assert.equal(second.loads.length, 1);
  registered.cleanup();
  assert.equal(second.listeners.size, 0, 'leaving the route must release the handler');
  assert.equal(second.button.hidden, true);
  await lifecycle.preload({ document: second.root });
  await lifecycle.mount(registered.context(second.root));
  registered.emit('site:content-updated');
  second.click();
  assert.equal(second.loads.length, 2, 'a click after route re-entry must issue exactly one reload');
  assert.equal(registered.loads.length, 1, 'route re-entry must reuse the loaded helper API');
  registered.cleanup();
  await lifecycle.mount(registered.context(second.root, true));
  assert.equal(second.listeners.size, 0, 'a cancelled mount must not bind controls');
}

module.exports = runTableauControlsTests;
if (require.main === module) {
  runTableauControlsTests()
    .then(() => console.log('Tableau reset lifecycle tests passed.'))
    .catch((error) => { console.error(error); process.exitCode = 1; });
}
