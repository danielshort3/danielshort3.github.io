'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

class Element {
  constructor() {
    this.dataset = {};
    this.value = '';
    this.textContent = '';
    this.innerHTML = '';
    this.listeners = new Map();
    this.nodes = new Map();
    this.attributes = new Map();
    this.nodeType = 1;
    const classes = new Set();
    this.classList = {
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
      contains: (name) => classes.has(name)
    };
  }
  querySelector(selector) { return this.nodes.get(selector) || null; }
  querySelectorAll() { return []; }
  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(callback);
  }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  emit(type) {
    for (const callback of this.listeners.get(type) || []) callback({ target: this, preventDefault() {} });
  }
  blur() {}
  matches(selector) { return selector === 'a[href]' && this.attributes.has('href'); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  setAttribute(name, value) { this.attributes.set(name, value); }
}

function clock() {
  let sequence = 0;
  const callbacks = new Map();
  return {
    callbacks,
    schedule(callback) { const id = ++sequence; callbacks.set(id, callback); return id; },
    cancel(id) { callbacks.delete(id); },
    flush() { const pending = [...callbacks.values()]; callbacks.clear(); pending.forEach((callback) => callback()); }
  };
}

function searchHarness() {
  const document = new Element();
  const headerForm = new Element();
  const headerInput = new Element();
  document.nodes.set('.nav-search', headerForm);
  document.nodes.set('.nav-search-input', headerInput);
  const timer = clock();
  const requests = [];
  const registrations = new Map();
  const window = {
    location: new URL('https://example.test/search?audience=analytics'),
    history: { state: {}, replaceState(state, title, url) { window.location = new URL(url, window.location); } },
    setTimeout: (callback) => timer.schedule(callback),
    clearTimeout: (id) => timer.cancel(id),
    SiteRoutes: { register: (id, lifecycle) => registrations.set(id, lifecycle) }
  };
  const source = fs.readFileSync(path.join(__dirname, '../../js/search/site-search.js'), 'utf8');
  vm.runInNewContext(source, {
    window, document, URL, AbortController, console,
    fetch(url, options) {
      let resolve;
      const promise = new Promise((done) => { resolve = done; });
      requests.push({ options, resolve: (pages) => resolve({ ok: true, json: async () => ({ pages }) }) });
      return promise;
    }
  });
  const createRoot = () => {
    const root = new Element();
    const form = new Element();
    const input = new Element();
    const results = new Element();
    const status = new Element();
    root.nodes.set('#search-page-form', form);
    root.nodes.set('#search-page-q', input);
    root.nodes.set('#search-results', results);
    root.nodes.set('#search-status', status);
    return { root, form, input, results, status };
  };
  return { window, headerForm, headerInput, timer, requests, registrations, createRoot };
}

module.exports = async function runProfessionalRouteLifecycleTests({ assert }) {
  const source = fs.readFileSync(path.join(__dirname, '../../js/portfolio/portfolio.js'), 'utf8');
  const bindingSource = source.slice(source.indexOf('function createPortfolioBindings()'), source.indexOf('const getSrStatus'));
  const frames = clock();
  const timers = clock();
  const context = vm.createContext({
    window: {
      requestAnimationFrame: (callback) => frames.schedule(callback),
      cancelAnimationFrame: (id) => frames.cancel(id),
      setTimeout: (callback) => timers.schedule(callback),
      clearTimeout: (id) => timers.cancel(id)
    }
  });
  vm.runInContext(bindingSource + '\nglobalThis.createBindings = createPortfolioBindings;', context);
  const bindings = context.createBindings();
  const target = new Element();
  let invocations = 0;
  bindings.listen(target, 'change', () => { invocations += 1; });
  target.emit('change');
  assert(invocations === 1, 'workbench listeners should run while their route is mounted');
  const staleListener = [...target.listeners.get('change')][0];
  const media = {
    listener: null,
    addListener(callback) { this.listener = callback; },
    removeListener(callback) { if (this.listener === callback) this.listener = null; }
  };
  bindings.media(media, () => { invocations += 1; });
  bindings.frame(() => { invocations += 1; });
  bindings.timer(() => { invocations += 1; }, 20);
  bindings.destroy();
  bindings.destroy();
  assert(!target.listeners.get('change').size && !media.listener, 'workbench disposal must release document and legacy media-query subscriptions exactly once');
  assert(!frames.callbacks.size && !timers.callbacks.size, 'workbench disposal must cancel delayed layout and focus work');
  staleListener({});
  frames.flush();
  timers.flush();
  assert(invocations === 1, 'a stale callback must not change content after workbench disposal');

  const h = searchHarness();
  assert(h.registrations.has('search:search'), 'all audience search routes must share a registered lifecycle');
  const first = h.createRoot();
  const cleanups = [];
  const firstContext = { root: first.root, url: 'https://example.test/search?audience=analytics&q=alpha', cleanup: (fn) => cleanups.push(fn) };
  const controller = h.window.SiteSearch.mount(firstContext);
  assert(h.window.SiteSearch.mount(firstContext) === controller && cleanups.length === 1,
    'repeated mounting of one search route must reuse its controller and subscriptions');
  first.input.value = 'beta';
  first.input.emit('input');
  h.timer.flush();
  h.requests[0].resolve([
    { url: '/portfolio/alpha', title: 'Alpha project', content: 'alpha' },
    { url: '/portfolio/beta', title: 'Beta project', content: 'beta' }
  ]);
  await controller.ready;
  await Promise.resolve();
  await Promise.resolve();
  assert(first.results.innerHTML.includes('/portfolio/beta') && !first.results.innerHTML.includes('/portfolio/alpha'),
    'a slower earlier search must not replace the most recent query results');
  assert(h.window.location.search === '?audience=analytics', 'search initialization must retain its audience while removing the raw query from the address');
  cleanups[0]();
  assert(h.requests[0].options.signal.aborted && !h.headerInput.listeners.get('input').size && !h.headerForm.listeners.get('submit').size,
    'search route disposal must abort its request and release persistent header bindings');
  assert(!h.headerInput.dataset.searchBound && !h.headerForm.dataset.searchBound,
    'the next route must be able to bind the persistent search controls');

  const second = h.createRoot();
  const restored = h.window.SiteSearch.mount({ root: second.root, url: 'https://example.test/search?audience=analytics' });
  assert(second.input.value === 'beta', 'returning to search must restore its in-memory audience-specific query');
  const oldMarkup = second.results.innerHTML;
  restored.dispose();
  h.requests[1].resolve([{ url: '/old-route', title: 'Beta old route' }]);
  await restored.ready;
  assert(second.results.innerHTML === oldMarkup, 'results arriving after disposal must not update detached route content');
  const tourism = h.createRoot();
  const otherAudience = h.window.SiteSearch.mount({ root: tourism.root, url: 'https://example.test/search?audience=tourism' });
  await otherAudience.ready;
  assert(tourism.input.value === '', 'search state must not leak from one professional audience to another');
  otherAudience.dispose();

  const warm = searchHarness();
  warm.window.location = new URL('https://example.test/analytics');
  const preparation = warm.registrations.get('search:search').preload({});
  assert(warm.window.location.pathname === '/analytics' && !warm.headerInput.listeners.size,
    'preparing search must leave the outgoing address and controls untouched');
  warm.requests[0].resolve([{ url: '/alpha', title: 'Alpha prepared result' }]);
  await preparation;
  const warmedRoot = warm.createRoot();
  const warmedController = warm.window.SiteSearch.mount({ root: warmedRoot.root, url: 'https://example.test/search?audience=analytics&q=alpha' });
  await warmedController.ready;
  assert(warm.requests.length === 1 && warmedRoot.results.innerHTML.includes('href="/alpha"'),
    'a prepared search must render from its warmed index without a second request');
  warmedController.dispose();

  const realmDocument = new Element();
  realmDocument.body = new Element();
  realmDocument.body.dataset = { audience: 'analytics', page: 'analytics' };
  realmDocument.documentElement = new Element();
  realmDocument.head = new Element();
  realmDocument.getElementById = () => null;
  const robots = new Element();
  robots.setAttribute('content', 'noindex, nofollow');
  realmDocument.head.querySelectorAll = () => [robots];
  let links = [];
  realmDocument.querySelectorAll = (selector) => selector === 'a[href]' ? links : [];
  const observers = [];
  const realmFrames = clock();
  const realmWindow = {
    location: new URL('https://example.test/analytics'),
    localStorage: { removeItem() {} },
    SITE_AUDIENCE_CONFIG: require('../../js/common/audience-config'),
    requestAnimationFrame: (callback) => realmFrames.schedule(callback),
    MutationObserver: function(callback) {
      this.callback = callback;
      this.disconnected = false;
      this.observe = () => {};
      this.disconnect = () => { this.disconnected = true; };
      observers.push(this);
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/common/site-realm.js'), 'utf8'), {
    window: realmWindow, document: realmDocument, URL, URLSearchParams, MutationObserver: realmWindow.MutationObserver
  });
  realmWindow.SiteRealm.sync({ url: realmWindow.location.href });
  const oldObserver = observers.at(-1);
  const addedLink = new Element();
  addedLink.setAttribute('href', '/portfolio/example');
  oldObserver.callback([{ addedNodes: [addedLink] }]);
  links = [addedLink];
  realmDocument.body.dataset = { audience: 'personal', page: 'home' };
  realmWindow.location = new URL('https://example.test/');
  realmWindow.SiteRealm.sync({ url: realmWindow.location.href });
  realmFrames.flush();
  assert(oldObserver.disconnected && addedLink.getAttribute('href') === '/portfolio/example',
    'queued professional link mutations must be cancelled before entering the personal audience');
  assert(realmWindow.SITE_AUDIENCE === 'personal' && realmDocument.documentElement.classList.contains('site-realm-personal'),
    'same-document audience changes must refresh both the public audience service and document styling');
  realmDocument.body.dataset = { audience: 'tourism', page: 'tourism' };
  realmWindow.location = new URL('https://example.test/tourism');
  realmWindow.SiteRealm.sync({ url: realmWindow.location.href });
  assert(addedLink.getAttribute('href') === '/portfolio/example?audience=tourism',
    'new professional content must use the destination audience, not the first loaded audience');
  assert(observers.length === 2 && !observers.at(-1).disconnected,
    'professional remounts must replace the observer rather than accumulate audience writers');
};

if (require.main === module) {
  let assertions = 0;
  module.exports({ assert(condition, message) { require('assert').ok(condition, message); assertions += 1; } })
    .then(() => console.log(`Professional route lifecycle tests passed (${assertions} assertions).`))
    .catch((error) => { console.error(error); process.exitCode = 1; });
}
