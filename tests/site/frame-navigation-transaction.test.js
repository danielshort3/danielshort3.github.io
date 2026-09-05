'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const router = fs.readFileSync(path.join(__dirname, '../../js/navigation/page-transitions.js'), 'utf8');
const navigateSource = router.slice(router.indexOf('  async function navigate('), router.indexOf('  function schedulePrefetch('));
const turn = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

function setup() {
  const events = [];
  const prepared = deferred();
  const mounted = deferred();
  const stylesReady = deferred();
  const oldBody = { id: 'home' };
  const newBody = { id: 'project' };
  const oldManifest = { id: 'home', category: 'projects', view: 'overview', styles: [] };
  const manifest = { id: 'project:babynames', category: 'projects', view: 'detail', styles: [] };
  const route = { manifest, styles: [], document: {}, frame: { fit: 'viewport' } };
  const oldDocument = { body: {}, title: 'Home' };
  const geometry = { width: 1500, height: 763 };
  let body = oldBody;
  let held = false;
  let active = { root: oldBody };
  const frame = {
    snapshot: () => ({ body: oldBody }),
    current: () => ({ fit: 'viewport' }),
    transition: (description) => events.push(['unexpected-transition', description.fit]),
    wipe: async (open) => events.push([open ? 'reveal' : 'wipe']),
    setLoading: () => {},
    hold: () => { held = true; events.push(['hold', body.id]); return geometry; },
    release: () => { assert(held); held = false; events.push(['release', body.id]); },
    whenSettled: () => Promise.resolve(),
    restore: (saved, options) => {
      assert(held, 'rollback must reserve flow before restoring route styles and content');
      assert.equal(options.defer, true);
      body = saved.body;
      events.push(['restore', body.id]);
      return body;
    }
  };
  const runtime = {
    beforeLeave: async () => true,
    current: () => active,
    unmount: async () => {
      assert(held, 'outgoing geometry must be captured before lifecycle cleanup changes layout');
      events.push(['unmount']);
      active = null;
    },
    mount: async (key, options) => {
      events.push(['mount', options.root.id]);
      if (options.root === newBody) await mounted.promise;
      active = { root: options.root };
      events.push(['mounted', options.root.id]);
    }
  };
  const window = {
    setTimeout, clearTimeout,
    location: { href: 'https://example.test/#projects' },
    history: { state: {} },
    requestAnimationFrame: (callback) => callback(),
    SiteFrame: frame,
    SiteRoutes: runtime
  };
  const context = {
    URL, AbortController, window, document: { title: 'Home' },
    activeNavigation: null, navigationSequence: 0,
    preparedRoutes: new Map(), preparedStyleRefs: new Map(), routeCache: new Map(),
    committedSnapshot: { frame: { body: oldBody }, document: oldDocument, manifest: oldManifest },
    committedUrl: window.location.href,
    resolveUrl: (value) => new URL(value, window.location.href),
    isDocumentLikeUrl: () => true,
    getPersonalRouteIntent: () => ({ category: 'projects', view: 'detail' }),
    isProfessionalAudienceUrl: () => false,
    isHardBoundary: () => false,
    isCurrentRouteSoft: () => true,
    normalizeRouteUrl: (url) => url.href,
    makeAbortError: () => new DOMException('Aborted', 'AbortError'),
    isAbortError: (error) => error.name === 'AbortError',
    throwIfAborted: (signal) => { if (signal.aborted) throw signal.reason; },
    hardNavigate: () => assert.fail('this viewport route must stay in the frame'),
    beginNavigationUi: () => () => {},
    readRouteManifest: () => body === oldBody ? oldManifest : manifest,
    getReturnFocusId: () => '',
    getHomepageHistoryIntent: () => null,
    clearRouteError: () => {},
    dispatch: (name, detail) => events.push(['event', name, detail]),
    NAVIGATION_EVENT: 'navigation', CONTENT_EVENT: 'content', ROUTE_EVENT: 'route',
    prepareRoute: () => prepared.promise,
    getRouteOutlet: () => body,
    saveCurrentHistory: () => {},
    captureDocumentMetadata: () => oldDocument,
    activatePreparedStyles: () => {},
    waitForStylesheetActivation: () => stylesReady.promise,
    retireOldStyles: () => {},
    registeredRouteKey: (value) => value.id,
    replaceRouteContent: (value, metadata, options) => {
      assert(held, 'route-owned body attributes must change inside the held transaction');
      assert.equal(options.from, geometry);
      assert.equal(options.defer, true, 'geometry must wait for the mounted content dimensions');
      events.push(['commit', value.frame.fit]);
      body = newBody;
      return body;
    },
    beginProvisionalHistory: (url) => { window.location.href = url.href; return {}; },
    restoreProvisionalHistory: () => {},
    pushRouteHistory: () => events.push(['history']),
    acceptPoppedHistory: () => {},
    rememberCommittedRoute: () => {},
    restoreScroll: () => {},
    focusRouteHeading: () => {},
    announce: () => {},
    sendVirtualPageview: () => events.push(['pageview']),
    syncBody: () => assert(held),
    syncHead: () => {},
    syncSkipLink: () => {},
    findStylesheet: () => null,
    restoreVetoedPop: () => {},
    showOfflineError: (url, options) => events.push(['retry', options])
  };
  vm.createContext(context);
  vm.runInContext(navigateSource, context);
  return { context, events, prepared, mounted, stylesReady, route, isHeld: () => held };
}

(async () => {
  const activation = { document: { styleSheets: [] }, throwIfAborted: (signal) => signal?.throwIfAborted() };
  const animationFrames = new Map();
  let animationId = 0;
  activation.window = {
    setTimeout, clearTimeout,
    requestAnimationFrame: (callback) => { animationFrames.set(++animationId, callback); return animationId; },
    cancelAnimationFrame: (id) => animationFrames.delete(id)
  };
  vm.createContext(activation);
  vm.runInContext(router.slice(router.indexOf('  function waitForStylesheetActivation('),
    router.indexOf('  function retireOldStyles(')), activation);
  const render = () => {
    const callbacks = [...animationFrames.values()];
    animationFrames.clear();
    callbacks.forEach((callback) => callback());
  };
  const style = { sheet: null };
  let attached = false;
  const stylePending = activation.waitForStylesheetActivation([style]).then(() => { attached = true; });
  style.sheet = {};
  render();
  await turn();
  assert(!attached, 'a fetched stylesheet must belong to the active document before layout can use it');
  activation.document.styleSheets.push(style.sheet);
  render();
  await stylePending;
  assert.equal(animationFrames.size, 0);
  await activation.waitForStylesheetActivation([style]);
  assert.equal(animationFrames.size, 0, 'already attached styles must not add a rendering delay');
  const abortController = new AbortController();
  const abortedStyle = activation.waitForStylesheetActivation([{ sheet: null }], abortController.signal);
  abortController.abort(new DOMException('Superseded', 'AbortError'));
  await assert.rejects(abortedStyle, { name: 'AbortError' });
  assert.equal(animationFrames.size, 0, 'superseded stylesheet waits must clean up their callbacks');

  const successful = setup();
  const pending = successful.context.navigate(new URL('https://example.test/portfolio/babynames'));
  await turn();
  assert(successful.events.some(([event]) => event === 'wipe'), 'loading receives immediate content feedback');
  assert(!successful.events.some(([event]) => event === 'unexpected-transition'), 'unknown destinations must not create a guessed document-sized frame');
  successful.prepared.resolve(successful.route);
  await turn();
  assert(successful.isHeld(), 'the outgoing flow space stays reserved while the destination mounts');
  assert(!successful.events.some(([event]) => event === 'commit'), 'fetched CSS must attach before measuring the destination');
  successful.stylesReady.resolve();
  await turn();
  assert(!successful.events.some(([event]) => event === 'release'), 'mounting cannot expose provisional geometry');
  successful.mounted.resolve();
  assert.equal(await pending, true);
  assert.deepEqual(successful.events.filter(([event]) => ['hold', 'commit', 'mounted', 'release'].includes(event)),
    [['hold', 'home'], ['commit', 'viewport'], ['mounted', 'project'], ['release', 'project']]);
  assert.equal(successful.events.filter(([event]) => event === 'pageview').length, 1);
  assert(!successful.isHeld(), 'successful navigation must release its layout hold');

  const failed = setup();
  failed.stylesReady.resolve();
  const rejected = failed.context.navigate(new URL('https://example.test/portfolio/babynames'));
  failed.prepared.resolve(failed.route);
  await turn();
  failed.mounted.reject(new Error('Synthetic mount failure'));
  assert.equal(await rejected, false);
  assert.deepEqual(failed.events.filter(([event]) => ['restore', 'mounted', 'release'].includes(event)),
    [['restore', 'home'], ['mounted', 'home'], ['release', 'home']]);
  assert(failed.events.some(([event]) => event === 'retry'));
  assert(!failed.events.some(([event]) => event === 'pageview'), 'failed destinations must not publish a pageview');
  assert(!failed.isHeld(), 'recovery must release layout after restoring the previous controller');

  for (const phase of ['cleanup', 'styles', 'mount']) {
    const recovery = setup();
    const initialError = new Error('Synthetic destination mount failure');
    const rollbackError = new Error(`Synthetic rollback ${phase} failure`);
    const runtime = recovery.context.window.SiteRoutes;
    const mount = runtime.mount;
    const unmount = runtime.unmount;
    runtime.mount = (key, options) => options.navigationType === 'restore' && phase === 'mount'
      ? Promise.reject(rollbackError) : mount(key, options);
    runtime.unmount = (options) => options.reason === 'rollback' && phase === 'cleanup'
      ? Promise.reject(rollbackError) : unmount(options);
    let activations = 0;
    recovery.context.waitForStylesheetActivation = () => {
      activations += 1;
      return phase === 'styles' && activations > 1 ? Promise.reject(rollbackError) : Promise.resolve();
    };
    const navigation = recovery.context.navigate(new URL('https://example.test/portfolio/babynames'));
    recovery.prepared.resolve(recovery.route);
    await turn();
    recovery.mounted.reject(initialError);
    // A click handler does not await navigation. Give a discarded rejection a
    // complete turn to surface before observing its successful false result.
    await turn();
    assert.equal(await navigation, false, `failed rollback ${phase} must not reject navigation`);
    assert(!recovery.isHeld(), `failed rollback ${phase} must release the layout hold`);
    assert(recovery.events.some(([event]) => event === 'reveal'), `failed rollback ${phase} must reveal its retry`);
    assert(recovery.events.some(([event, options]) => event === 'retry' && options.hardRetry),
      `failed rollback ${phase} must offer a full-page retry`);
    assert.equal(recovery.events.find(([event, name]) => event === 'event' && name === 'site:route-navigation-error')[2].error, initialError,
      'recovery must preserve the original navigation diagnostic');
    const diagnostic = recovery.events.find(([event, name]) => event === 'event' && name === 'site:route-rollback-error')[2];
    assert.equal(diagnostic.error, rollbackError);
    assert.equal(diagnostic.originalError, initialError);
    assert(!recovery.events.some(([event]) => event === 'pageview'), 'failed recovery must not publish a pageview');
  }

  let retryAction;
  let hardNavigations = 0;
  let softNavigations = 0;
  const retryContext = {
    clearRouteError() {},
    getRouteOutlet: () => ({ parentElement: { insertBefore() {} } }),
    document: { createElement: () => ({ dataset: {}, setAttribute() {}, append() {}, focus() {},
      addEventListener(type, callback) { if (type === 'click') retryAction = callback; } }) },
    navigator: { onLine: true },
    hardNavigate: () => { hardNavigations += 1; },
    navigate: () => { softNavigations += 1; }
  };
  vm.createContext(retryContext);
  vm.runInContext(router.slice(router.indexOf('  function showOfflineError('), router.indexOf('  function navigationDirection(')), retryContext);
  retryContext.showOfflineError(new URL('https://example.test/portfolio/babynames'), { hardRetry: true });
  retryAction();
  assert.equal(hardNavigations, 1, 'unrecoverable route retry must perform a full document navigation');
  assert.equal(softNavigations, 0);
  retryContext.showOfflineError(new URL('https://example.test/portfolio/babynames'));
  retryAction();
  assert.equal(softNavigations, 1, 'normal recovery must keep its existing soft retry');
  console.log('Frame preparation, mounted geometry, and rollback transactions passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
