'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../js/navigation/page-transitions.js'), 'utf8');
const functions = (...names) => names.map((name) => {
  const start = source.search(new RegExp(`  (?:async )?function ${name}\\(`));
  assert(start >= 0, `Missing ${name}`);
  const end = source.slice(start + 1).search(/\n  (?:async )?function /);
  return end < 0 ? source.slice(start) : source.slice(start, start + 1 + end);
}).join('\n');
const origin = 'https://example.test';
const current = { styles: [`${origin}/dist/styles.1234abcd.css`], scripts: [`${origin}/dist/site-shell.5678abcd.js`] };
const old = { styles: [`${origin}/dist/styles.b81be451.css`], scripts: [`${origin}/dist/site-shell.ecbf0338.js`] };
const route = (manifest) => ({ manifest, document: {}, frame: {}, styles: [] });
const turn = () => new Promise((resolve) => setImmediate(resolve));

function setupPreparation(responses) {
  const calls = [];
  const key = `${origin}/tools/text-compare`;
  const context = {
    URL, Promise, Error,
    shellGeneration: { style: current.styles[0], script: current.scripts[0] },
    resolveUrl: (value) => new URL(value, origin),
    normalizeRouteUrl: (url) => url.href,
    preparedRoutes: new Map(), preparedStyleRefs: new Map(), routeCache: new Map(),
    activeNavigation: null, ROUTE_CACHE_LIMIT: 12,
    raceWithAbort: (promise) => promise,
    pinShellStyles: () => calls.push(['pin']),
    fetchRouteDocument: async (_, options) => {
      calls.push(['fetch', options.cache || 'default']);
      const response = responses.shift();
      if (response instanceof Error) throw response;
      assert(response, 'fixture must not issue an unexpected document request');
      return response;
    },
    prepareRouteStyles: async (value) => { calls.push(['styles', value.manifest.styles[0]]); return []; },
    prepareRouteScripts: async (value) => calls.push(['scripts', value.manifest.scripts[0]]),
    prunePreparedStyles() {},
    isAbortError: (error) => error.name === 'AbortError',
    window: { SiteFrame: { describe: () => ({}) } }
  };
  vm.createContext(context);
  vm.runInContext(functions('getShellGeneration', 'isCompatibleGeneration', 'generationMismatchError',
    'requireCompatibleGeneration', 'prepareRoute', 'prepareRouteResources'), context);
  return { context, calls, key, navigate: () => context.prepareRoute(new URL(key)) };
}

async function runNavigationGenerationTests() {
  const stale = setupPreparation([route(old), route(current)]);
  const refreshed = await stale.navigate();
  assert.equal(refreshed.manifest, current);
  assert.deepEqual(stale.calls.filter(([name]) => name === 'fetch'), [['fetch', 'default'], ['fetch', 'reload']]);
  assert(stale.calls.filter(([name]) => ['styles', 'scripts'].includes(name)).every(([, asset]) => !asset.includes('b81be451') && !asset.includes('ecbf0338')),
    'successfully cached old assets must be rejected before activation or script preparation');
  await stale.navigate();
  assert.equal(stale.calls.filter(([name]) => name === 'fetch').length, 2, 'a compatible warm revisit must reuse its refreshed preparation');

  const cached = setupPreparation([route(current)]);
  cached.context.preparedRoutes.set(cached.key, Promise.resolve(route(old)));
  cached.context.routeCache.set(cached.key, 'old HTML');
  cached.context.preparedStyleRefs.set(cached.key, new Set(old.styles));
  await cached.navigate();
  assert.deepEqual(cached.calls.filter(([name]) => name === 'fetch'), [['fetch', 'reload']], 'prepared cache hits must also revalidate their exact generation');
  assert(!cached.context.routeCache.has(cached.key), 'stale HTML must be evicted before fresh revalidation');
  assert(!cached.context.preparedStyleRefs.get(cached.key).has(old.styles[0]), 'stale stylesheet references must be retired from preparation ownership');

  for (const manifest of [old, { ...current, scripts: old.scripts }, { ...current, styles: old.styles }]) {
    const mismatch = setupPreparation([route(manifest), route(manifest)]);
    await assert.rejects(mismatch.navigate(), { code: 'SITE_ROUTE_GENERATION_MISMATCH' });
    assert.equal(mismatch.calls.filter(([name]) => name === 'fetch').length, 2, 'a persistent generation mismatch must stop after one revalidation');
    assert(!mismatch.calls.some(([name]) => name === 'styles' || name === 'scripts'), 'incompatible resources must never be mounted');
    await turn();
    assert.equal(mismatch.context.preparedRoutes.size, 0, 'a rejected generation must not poison future attempts');
  }
  const offline = setupPreparation([route(old), new Error('Network unavailable')]);
  await assert.rejects(offline.navigate(), /Network unavailable/);
  assert(!offline.calls.some(([name]) => name === 'styles' || name === 'scripts'), 'failed revalidation must leave the active page resources untouched');

  const links = [];
  const sheets = [];
  const makeLink = (href, connected = true) => {
    const link = {
      href, isConnected: connected, sheet: {}, disabled: false, media: '', dataset: {},
      getAttribute: (name) => name === 'href' ? href : link[name] || null,
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { if (name === 'media') this.media = ''; },
      remove() { this.isConnected = false; },
      addEventListener(name, callback) { this[name] = callback; }
    };
    links.push(link);
    sheets.push(link.sheet);
    return link;
  };
  const base = makeLink(current.styles[0]);
  const detail = makeLink(`${origin}/detail.css`);
  const oldBase = makeLink(old.styles[0]);
  const context = {
    URL, Promise, Error, Set, Array,
    shellGeneration: { style: base.href, script: current.scripts[0] }, shellStyleLink: base, shellStyleMedia: '',
    styleLoads: new Map(), preparedRoutes: new Map(), preparedStyleRefs: new Map(), activeNavigation: null,
    resolveUrl: (value) => new URL(value, origin), normalizeAssetUrl: (value) => new URL(value, origin).href,
    raceWithAbort: (promise) => promise, throwIfAborted: (signal) => signal?.throwIfAborted(),
    readRouteManifest: () => current,
    getDocumentAssetBase: () => `${origin}/`,
    document: {
      styleSheets: sheets,
      querySelectorAll: () => links.filter((link) => link.isConnected),
      createElement: () => makeLink('', false),
      head: { appendChild(link) { link.isConnected = true; queueMicrotask(() => link.load?.()); } }
    },
    window: { location: { href: `${origin}/` }, setTimeout, clearTimeout, requestAnimationFrame: () => assert.fail('ready styles should not need another animation frame'), cancelAnimationFrame() {} }
  };
  vm.createContext(context);
  vm.runInContext(functions('pinShellStyles', 'findStylesheet', 'loadStylesheet', 'activatePreparedStyles', 'waitForStylesheetActivation', 'retireOldStyles', 'prunePreparedStyles'), context);
  context.retireOldStyles({ styles: [base.href, detail.href] }, { styles: [oldBase.href] });
  assert.equal(base.media, '', 'the boot stylesheet must stay active even when an older manifest names another hash');
  assert.equal(detail.media, 'not all', 'ordinary route styles must still retire');
  base.disabled = true;
  base.sheet.disabled = true;
  base.media = 'not all';
  base.remove();
  context.pinShellStyles();
  assert(base.isConnected && !base.disabled && !base.sheet.disabled && !base.media, 'persistent shell styles must recover disabled or detached state');
  context.styleLoads.set(base.href, Promise.resolve({ link: base }));
  context.styleLoads.set(oldBase.href, Promise.resolve({ link: oldBase }));
  context.prunePreparedStyles();
  await turn();
  assert(base.isConnected && !oldBase.isConnected, 'pruning must preserve the exact pinned base while discarding unrelated old assets');
  detail.disabled = true;
  detail.sheet.disabled = true;
  context.activatePreparedStyles([detail]);
  await context.waitForStylesheetActivation([detail]);
  assert(!detail.disabled && !detail.sheet.disabled && !detail.media, 'activation must restore both link and sheet enabled state and intended media');
  const detached = makeLink(`${origin}/detached.css`, false);
  context.styleLoads.set(detached.href, Promise.resolve({ link: detached }));
  const repaired = await context.loadStylesheet(detached.href, { querySelectorAll: () => [] });
  assert.notEqual(repaired.link, detached, 'a cached promise to a disconnected route stylesheet must be replaced');
  assert(repaired.link.isConnected, 'the replacement stylesheet must be connected before preparation completes');

  let reload;
  let allowed = false;
  let resolveSave;
  let hardNavigations = 0;
  let guardCalls = 0;
  const promptNodes = [];
  const reloadContext = {
    navigationSequence: 4,
    clearRouteError() {},
    getRouteOutlet: () => ({ parentElement: { insertBefore() {} } }),
    document: { createElement: () => {
      const node = { isConnected: true, dataset: {}, setAttribute() {}, append() {}, focus() {}, addEventListener: (_, action) => { reload = action; } };
      promptNodes.push(node);
      return node;
    } },
    navigator: { onLine: true },
    window: { SiteRoutes: { beforeLeave: async (options) => {
      guardCalls += 1;
      assert.equal(options.navigationType, 'reload');
      if (allowed === 'saving') return new Promise((resolve) => { resolveSave = resolve; });
      return allowed;
    } } },
    hardNavigate: () => { hardNavigations += 1; },
    navigate: () => assert.fail('Reload must not retry an incompatible soft route')
  };
  vm.createContext(reloadContext);
  vm.runInContext(functions('showOfflineError'), reloadContext);
  reloadContext.showOfflineError(new URL(`${origin}/tools/text-compare`), { reloadRequired: true });
  assert.equal(promptNodes.at(-1).textContent, 'Reload');
  await reload();
  assert.equal(hardNavigations, 0, 'a leave veto must keep current unsaved work and the prompt');
  allowed = 'saving';
  const saving = reload();
  assert.equal(hardNavigations, 0, 'Reload must await asynchronous save guards');
  reloadContext.navigationSequence += 1;
  resolveSave(true);
  await saving;
  assert.equal(hardNavigations, 0, 'a stale Reload prompt must not navigate after another route wins');
  allowed = true;
  await reload();
  assert.equal(hardNavigations, 1, 'an explicit Reload may navigate once the current leave guards succeed');
  assert.equal(guardCalls, 3);

  const windowScrolls = [];
  let anchorScrolls = 0;
  const anchor = { getBoundingClientRect: () => ({ top: 320 }), scrollIntoView: () => { anchorScrolls += 1; } };
  const owner = {
    scrollTop: 20, scrollLeft: 0, scrollHeight: 1000, clientHeight: 400,
    contains: (node) => node === anchor, getBoundingClientRect: () => ({ top: 120 })
  };
  const scrollContext = {
    document: {
      getElementById: (id) => id === 'details section' ? anchor : null,
      querySelector: () => ({ getBoundingClientRect: () => ({ bottom: 62 }) })
    },
    window: {
      requestAnimationFrame: (callback) => callback(),
      getComputedStyle: () => ({ scrollMarginTop: '62px' }),
      scrollTo: (...args) => windowScrolls.push(args)
    },
    getScrollOwner: () => owner
  };
  vm.createContext(scrollContext);
  vm.runInContext(functions('getRouteHashTarget', 'getFrameScrollIntent', 'restoreScroll'), scrollContext);
  const anchorUrl = new URL(`${origin}/portfolio/retailStore#details%20section`);
  const intent = scrollContext.getFrameScrollIntent({ windowY: 425 }, anchorUrl);
  assert.equal(intent.target, anchor, 'hash targets must be passed as nodes for measurement in destination layout');
  assert.equal(intent.top, 425);
  assert.equal(intent.offset, 62);
  scrollContext.restoreScroll(null, anchorUrl, { window: false });
  assert.equal(owner.scrollTop, 158, 'compact completion may align a hash inside its nested scroll owner');
  assert.equal(windowScrolls.length, 0, 'compact completion must not scroll the window a second time');
  scrollContext.restoreScroll(null, anchorUrl);
  assert.equal(anchorScrolls, 1, 'desktop and direct-load anchor behavior must stay intact');
}

module.exports = runNavigationGenerationTests;
if (require.main === module) runNavigationGenerationTests().then(() => {
  process.stdout.write('Navigation generation, stylesheet ownership, and guarded Reload tests passed.\n');
}).catch((error) => { console.error(error); process.exitCode = 1; });
