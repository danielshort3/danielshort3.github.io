'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const read = (file) => fs.readFileSync(path.join(__dirname, '../..', file), 'utf8');
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};
const turn = () => new Promise((resolve) => setImmediate(resolve));

async function runtimeTests() {
  const document = new EventTarget();
  Object.assign(document, { readyState: 'loading', body: { dataset: {} }, querySelector: () => null });
  const window = { location: { href: 'https://example.test/' } };
  vm.runInNewContext(read('js/navigation/site-route-runtime.js'), {
    window, document, URL, AbortController, CustomEvent, Event, console
  });
  const runtime = window.SiteRoutes;
  const firstRoot = {};
  const secondRoot = {};
  const mounted = [];
  let cleaned = 0;
  runtime.register('contact:contact', {
    mount(context) {
      mounted.push(context.id);
      context.cleanup(() => { cleaned += 1; });
    }
  });
  await runtime.mount('contact:contact', { manifest: { id: 'contact:analytics' }, root: firstRoot });
  assert.equal(runtime.current().id, 'contact:analytics');
  assert.equal(runtime.current().module, 'contact:contact');
  let sharedCleanup = 0;
  runtime.addCleanup(() => { sharedCleanup += 1; }, 'contact:analytics');
  const save = deferred();
  const guard = (event) => event.detail.waitUntil(save.promise);
  document.addEventListener('site:route-before-leave', guard);
  let finished = false;
  const leaving = runtime.beforeLeave().then((result) => { finished = true; return result; });
  await turn();
  assert.equal(finished, false, 'navigation must wait for the shared save guard');
  assert.equal(runtime.current().root, firstRoot, 'waiting must retain the mounted page');
  save.resolve(false);
  assert.equal(await leaving, false, 'a failed save must veto leaving');
  document.removeEventListener('site:route-before-leave', guard);
  assert.equal(await runtime.beforeLeave(), true);
  await runtime.mount('contact:contact', { manifest: { id: 'contact:tourism' }, root: secondRoot });
  assert.deepEqual(mounted, ['contact:analytics', 'contact:tourism']);
  assert.equal(cleaned, 1);
  assert.equal(sharedCleanup, 1, 'cleanup registered by route identity must run');
  assert.equal(runtime.current().root, secondRoot);

  const slow = deferred();
  let canceledCleanup = 0;
  runtime.register('slow', { mount(context) { context.cleanup(() => { canceledCleanup += 1; }); return slow.promise; } });
  const obsolete = runtime.mount('slow', { root: {} });
  await turn();
  await runtime.mount('contact:contact', { manifest: { id: 'contact:personal' }, root: firstRoot });
  slow.resolve();
  await obsolete;
  assert.equal(runtime.current().id, 'contact:personal', 'late mount completion cannot replace the newest route');
  assert.equal(canceledCleanup, 1, 'an interrupted mount must clean up exactly once');
  await runtime.unmount();
}

async function legacyManifestTests() {
  class RuntimeEventTarget extends EventTarget {}
  const document = new RuntimeEventTarget();
  const window = new RuntimeEventTarget();
  const requested = [];
  const removed = [];
  const cleaned = [];
  let clicks = 0;
  Object.assign(window, {
    EventTarget: RuntimeEventTarget,
    location: { href: 'https://example.test/' }
  });
  Object.assign(document, {
    readyState: 'loading',
    body: { dataset: {} },
    querySelector: () => null,
    createElement() {
      return Object.assign(new RuntimeEventTarget(), {
        dataset: {},
        remove() { removed.push(this.src); }
      });
    },
    head: {
      appendChild(script) {
        requested.push(script.src);
        queueMicrotask(() => {
          if (script.src.endsWith('/legacy.old.js')) {
            script.dispatchEvent(new Event('error'));
            return;
          }
          document.currentScript = script;
          window.SiteRoutes.addCleanup(() => { cleaned.push(script.src); });
          if (script.src.endsWith('/legacy.new.js')) {
            window.addEventListener('legacy-click', () => { clicks += 1; });
            window.addEventListener('beforeunload', (event) => event.preventDefault());
          }
          document.currentScript = null;
          script.dispatchEvent(new Event('load'));
        });
      }
    }
  });
  vm.runInNewContext(read('js/navigation/site-route-runtime.js'), {
    window, document, URL, AbortController, CustomEvent, Event, console
  });
  const runtime = window.SiteRoutes;
  const id = 'tools:legacy';
  const old = 'https://example.test/dist/legacy.old.js';
  const fresh = 'https://example.test/dist/legacy.new.js';
  const latest = 'https://example.test/dist/legacy.latest.js';
  const lifecycle = runtime.ensureLegacyRoute(id, { scripts: [old] });
  await assert.rejects(runtime.mount(id, {
    navigationType: 'push', manifest: { id, scripts: [old] }
  }), /Failed to load route script/);
  assert.equal(runtime.current(), null, 'an obsolete bundle failure must cleanly end its mount');

  await runtime.mount(id, {
    navigationType: 'push',
    manifest: { id, scripts: ['/dist/site-shell.current.js', '/js/common/common.js', '/dist/legacy.new.js', fresh] }
  });
  assert.deepEqual(requested, [old, fresh], 'Retry must use the rebuilt manifest, exclude persistent scripts, and normalize duplicates');
  window.dispatchEvent(new Event('legacy-click'));
  assert.equal(clicks, 1, 'the recovered script must attach live scoped hooks');
  assert.equal(await runtime.beforeLeave(), false, 'recovered legacy beforeunload hooks must still guard navigation');
  assert.equal(runtime.ensureLegacyRoute(id, { scripts: [latest] }), lifecycle, 'refreshing an existing legacy registration must preserve its lifecycle');
  await runtime.mount(id, { navigationType: 'push', manifest: { id, scripts: [latest] } });
  assert.deepEqual(requested, [old, fresh, latest], 'a later manifest must replace both registered and previously mounted script lists');
  assert.deepEqual(cleaned, [fresh], 'replacing the manifest must clean the departed script exactly once');
  window.dispatchEvent(new Event('legacy-click'));
  assert.equal(clicks, 1, 'the departed script must not leave duplicate event hooks');
  assert.equal(await runtime.beforeLeave(), true, 'departed beforeunload hooks must not veto the next route');
  await runtime.mount(id, { navigationType: 'push', manifest: { id, scripts: [] } });
  assert.deepEqual(requested, [old, fresh, latest], 'an explicit empty manifest must not replay obsolete fallback scripts');
  assert.deepEqual(cleaned, [fresh, latest]);
  await runtime.unmount();
  assert.deepEqual(removed, requested, 'failed and successful route-owned script nodes must be removed');

  const fallback = 'https://example.test/js/fallback.js';
  runtime.ensureLegacyRoute('fallback', { scripts: [fallback] });
  await runtime.mount('fallback', { navigationType: 'push' });
  assert.equal(requested.at(-1), fallback, 'callers without a manifest must retain their registered script fallback');
  await runtime.unmount();

  const count = requested.length;
  let directCleanup = 0;
  runtime.ensureLegacyRoute('direct', { scripts: ['/dist/direct.js'] });
  runtime.runInScope('direct', () => {
    window.addEventListener('beforeunload', (event) => event.preventDefault());
    runtime.addCleanup(() => { directCleanup += 1; });
  });
  await runtime.mount('direct', { navigationType: 'load', manifest: { id: 'direct', scripts: ['/dist/direct.js'] } });
  assert.equal(requested.length, count, 'direct loads must adopt already executed scripts without loading them again');
  assert.equal(await runtime.beforeLeave(), false, 'direct-load hooks must remain attached to the adopted cleanup scope');
  await runtime.unmount();
  assert.equal(directCleanup, 1);
}

async function draftTests() {
  const source = read('js/accounts/tools-account-ui.js');
  const start = source.indexOf('  const routeDrafts = new Map();');
  const end = source.indexOf('  let sharedServicesPromise', start);
  assert.ok(start > 0 && end > start);
  const document = new EventTarget();
  const window = new EventTarget();
  let authenticated = false;
  let owner = 'guest';
  let sessionParam = '';
  let saver = async () => ({ session: { sessionId: 'saved', version: 1 } });
  let calls = 0;
  Object.assign(window, {
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 2, clearInterval() {},
    ToolsAuth: {
      getAuth: () => ({}), getUser: () => ({ sub: owner }), authIsValid: () => authenticated
    },
    ToolsState: {
      saveSession: (...args) => { calls += 1; return saver(...args); },
      logActivity: async () => {}, getSession: async () => ({})
    }
  });
  const context = vm.createContext({
    window, document, CustomEvent, console, AUTO_SAVE_MS: 10000,
    getSessionParam: () => sessionParam, getActiveSessionId: () => '', setActiveSessionId() {}, setSessionParam() {},
    buildSnapshot: ({ root }) => ({ fields: { text: root.value } }),
    captureToolPayload: () => ({}),
    applyToolFields: (root, fields) => { root.value = fields.text; },
    notifySessionApplied() {}, logAsyncError: (label, error) => { throw error; }
  });
  vm.runInContext(`${source.slice(start, end)}\nglobalThis.mountSave = initToolAutoSave;`, context);
  const root = () => Object.assign(new EventTarget(), { value: '' });
  const options = (node, mode = 'manual') => ({ toolId: 'text-compare', root: node, persistenceMode: mode });

  const first = root();
  const leaveFirst = context.mountSave(options(first));
  first.value = 'Unsaved comparison draft';
  first.dispatchEvent(new Event('input'));
  assert.equal(await leaveFirst.beforeLeave(), true);
  leaveFirst();
  const second = root();
  const leaveSecond = context.mountSave(options(second));
  assert.equal(second.value, first.value, 'returning to an ordinary tool must restore its unsaved text');
  assert.equal(calls, 0, 'manual or signed-out drafts must not silently become cloud saves');
  leaveSecond();

  authenticated = true;
  owner = 'person-a';
  const third = root();
  const leaveThird = context.mountSave(options(third, 'autosave'));
  assert.equal(third.value, '', 'drafts belonging to another session owner must not restore');
  third.value = 'Save this before navigating';
  third.dispatchEvent(new Event('input'));
  const save = deferred();
  saver = () => save.promise;
  document.dispatchEvent(new CustomEvent('tools:save-session', { detail: { toolId: 'text-compare' } }));
  let left = false;
  const leave = leaveThird.beforeLeave().then((result) => { left = true; return result; });
  await turn();
  assert.equal(left, false, 'the leave guard must await an already-running save');
  assert.equal(calls, 1, 'waiting must not issue a duplicate save');
  save.resolve({ session: { sessionId: 'saved', version: 1 } });
  assert.equal(await leave, true);
  third.value = 'Keep this when saving fails';
  third.dispatchEvent(new Event('input'));
  saver = async () => { throw new Error('Offline'); };
  assert.equal(await leaveThird.beforeLeave(), false, 'save failures must retain the active page and draft');
  leaveThird();

  sessionParam = 'pending-load';
  const loading = deferred();
  window.ToolsState.getSession = () => loading.promise;
  const fourth = root();
  const leaveFourth = context.mountSave(options(fourth));
  fourth.value = 'Newer local edits';
  fourth.dispatchEvent(new Event('input'));
  await leaveFourth.beforeLeave();
  leaveFourth();
  const fifth = root();
  const leaveFifth = context.mountSave(options(fifth, 'autosave'));
  assert.equal(fifth.value, fourth.value);
  loading.resolve({ session: { version: 9, snapshot: { fields: { text: 'Older cloud value' } } } });
  await turn();
  assert.equal(fifth.value, 'Newer local edits', 'resumed session loading must not overwrite a dirty local draft');
  saver = async (request) => {
    assert.equal(request.expectedVersion, 9, 'resuming a pending session must resolve its version before saving');
    return { session: { sessionId: 'pending-load', version: 10 } };
  };
  assert.equal(await leaveFifth.beforeLeave(), true, 'a draft restored during session loading must remain saveable');
  leaveFifth();
}

(async () => {
  await runtimeTests();
  await legacyManifestTests();
  await draftTests();
  console.log('Route identity, legacy manifest recovery, guarded saves, interruption cleanup, and draft restoration passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
