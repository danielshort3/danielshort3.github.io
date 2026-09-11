'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../js/accounts/tools-account-ui.js'), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const turn = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
};

function fakeTimers() {
  let now = 0;
  let nextId = 0;
  const pending = new Map();
  const schedule = (callback, delay, interval = false) => {
    const id = ++nextId;
    pending.set(id, { callback, delay: Math.max(1, Number(delay) || 1), due: now + Math.max(1, Number(delay) || 1), interval });
    return id;
  };
  return {
    setTimeout: (callback, delay) => schedule(callback, delay),
    clearTimeout: (id) => pending.delete(id),
    setInterval: (callback, delay) => schedule(callback, delay, true),
    clearInterval: (id) => pending.delete(id),
    async advance(ms) {
      const target = now + ms;
      let count = 0;
      while (true) {
        const entry = [...pending.entries()].filter(([, timer]) => timer.due <= target).sort((a, b) => a[1].due - b[1].due)[0];
        if (!entry) break;
        assert(++count < 1000, 'autosave must not create an unbounded timer loop');
        const [id, timer] = entry;
        now = timer.due;
        if (timer.interval) timer.due += timer.delay;
        else pending.delete(id);
        timer.callback();
        await turn();
      }
      now = target;
      await turn();
    }
  };
}

function accountKeys() {
  let owner = 'person-a';
  const values = new Map([['toolsActiveSession:text-compare', 'legacy-other-account']]);
  const localStorage = {
    get length() { return values.size; },
    key: (index) => [...values.keys()][index],
    getItem: (key) => values.get(key),
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key)
  };
  const context = vm.createContext({
    localStorage, ACTIVE_SESSION_PREFIX: 'toolsActiveSession:',
    window: { ToolsAuth: { getAuth: () => ({}), authIsValid: () => Boolean(owner), getUser: () => ({ sub: owner }) } }
  });
  vm.runInContext(`${section('  const activeSessionKey = ', '  const dispatchValueEvents = ')}
    globalThis.api = { getActiveSessionId, setActiveSessionId, clearActiveSessionIds };`, context);
  assert.equal(context.api.getActiveSessionId('text-compare'), '', 'unowned legacy session IDs must never associate with a signed-in account');
  context.api.setActiveSessionId('text-compare', 'a-work');
  owner = 'person-b';
  assert.equal(context.api.getActiveSessionId('text-compare'), '', 'switching account must not restore the previous account session');
  context.api.setActiveSessionId('text-compare', 'b-work');
  values.set('toolsPendingDraft:person-a:text-compare', 'a-pending');
  values.set('toolsPendingDraft:person-b:text-compare', 'b-pending');
  context.api.clearActiveSessionIds();
  assert.equal(context.api.getActiveSessionId('text-compare'), '');
  assert.equal(values.has('toolsPendingDraft:person-b:text-compare'), false, 'deleting account data must also remove that account pending recovery records');
  assert.equal(values.get('toolsPendingDraft:person-a:text-compare'), 'a-pending', 'deleting one account data must preserve another account pending work');
  owner = 'person-a';
  assert.equal(context.api.getActiveSessionId('text-compare'), 'a-work', 'deleting one account cache must preserve another account cache');
  owner = '';
  context.api.setActiveSessionId('text-compare', 'guest');
  assert.equal(context.api.getActiveSessionId('text-compare'), '', 'guests must not create an account session association');
}

function continuationRows() {
  const context = vm.createContext({
    TOOL_CATALOG: { 'text-compare': {}, 'word-frequency': {} },
    getToolInfo: (toolId) => ({ name: toolId, href: `/tools/${toolId}` }),
    escapeHtml: (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'),
    formatTime: (time) => String(time)
  });
  vm.runInContext(`${section('  const getLatestSavedWork = ', '  const initAccountModal = ')}
    globalThis.api = { getLatestSavedWork, renderContinueWork };`, context);
  const sessions = [
    { toolId: 'text-compare', sessionId: 'older-pinned', updatedAt: 10, pinned: true },
    { toolId: 'word-frequency', sessionId: 'word-work', updatedAt: 15 },
    { toolId: 'text-compare', sessionId: 'newest&work', updatedAt: 20 },
    { toolId: 'unknown', sessionId: 'unsupported', updatedAt: 30 },
    { toolId: 'text-compare', updatedAt: 40 }
  ];
  const latest = JSON.parse(JSON.stringify(context.api.getLatestSavedWork(sessions)));
  assert.deepEqual(latest.map((session) => session.sessionId), ['newest&work', 'word-work'], 'Continue should show one newest record per supported tool, ignoring pin state');
  const markup = context.api.renderContinueWork(sessions);
  assert(markup.includes('/tools/text-compare?session=newest%26work'), 'continuation must retain the exact encoded session selection');
  assert.equal((markup.match(/>Continue<\/a>/g) || []).length, 2);
  assert(!/older-pinned|Search sessions|Pin session|User ID/.test(markup), 'continuation must not expose history management controls');
  assert(!/use Save|click Save/i.test(context.api.renderContinueWork([])), 'empty state must not instruct users to find a manual Save control');
}

function harness({ value = '', output, session = '', active = '', extraFields = [], list, get, save, confirm, mode, initialOwner = 'person-a', previousOwner = '', storage = new Map() } = {}) {
  let owner = initialOwner;
  let toolOutput = output;
  let sessionParam = session;
  let activeSession = active;
  const field = { tagName: 'TEXTAREA', value, defaultValue: '' };
  const root = new EventTarget();
  root.dataset = previousOwner ? { toolsDraftOwner: previousOwner } : {};
  let rootCleared = false;
  Object.defineProperty(root, 'value', { get: () => field.value, set: (next) => { field.value = next; } });
  Object.defineProperty(root, 'output', { get: () => toolOutput });
  root.querySelectorAll = () => [field, ...extraFields];
  root.replaceChildren = () => { rootCleared = true; };
  const calls = { list: [], get: [], save: [], applied: 0, capture: 0, reload: 0, states: [], status: [], confirm: [], signOut: 0 };
  let disposed = false;
  const document = new EventTarget();
  document.visibilityState = 'visible';
  const window = new EventTarget();
  const clock = fakeTimers();
  Object.assign(window, {
    location: { reload: () => { calls.reload += 1; } },
    setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval, clearInterval: clock.clearInterval,
    confirm: (message) => { calls.confirm.push(message); return confirm ? confirm(message) : false; },
    ToolsAuth: {
      getAuth: () => ({}), authIsValid: () => Boolean(owner), getUser: () => ({ sub: owner }),
      signOut: async () => { calls.signOut += 1; owner = ''; }
    },
    ToolsState: {
      listSessions: (request) => { calls.list.push(request); return list ? list(request) : Promise.resolve({ sessions: [{ toolId: 'text-compare', sessionId: 'latest' }] }); },
      getSession: (request) => { calls.get.push(request); return get ? get(request) : Promise.resolve({ session: { version: 4, snapshot: { fields: { text: 'Saved work' } } } }); },
      saveSession: async (request) => { calls.save.push(request); return save ? save(request) : { session: { sessionId: request.sessionId || 'new-work', version: 5 } }; },
      logActivity: async () => ({})
    }
  });
  const context = vm.createContext({
    window, document, CustomEvent, console, AUTO_SAVE_MS: 20000, AUTO_SAVE_DEBOUNCE_MS: 1000,
    localStorage: {
      get length() { return storage.size; },
      key: (index) => [...storage.keys()][index],
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key)
    },
    getSessionParam: () => sessionParam,
    setSessionParam: (next) => { sessionParam = next; },
    getActiveSessionId: () => activeSession,
    setActiveSessionId: (toolId, next) => { activeSession = next; },
    buildSnapshot: () => ({ fields: { text: root.value, ...Object.fromEntries(extraFields.filter(item => item.id).map(item => [item.id, item.value])) } }),
    captureToolPayload: () => { calls.capture += 1; return toolOutput ? { output: toolOutput } : {}; },
    applyToolFields: (target, fields) => {
      target.value = fields.text || '';
      extraFields.filter(item => item.id).forEach(item => { item.value = fields[item.id] || ''; });
    },
    notifySessionApplied: ({ snapshot }) => { calls.applied += 1; toolOutput = snapshot.output; },
    logAsyncError: (label, error) => { throw error; }
  });
  vm.runInContext(`${section('  const routeDrafts = new Map();', '  let sharedServicesPromise')}
    ${section('  const createSignOutHandler = ', '  const initAccountBar = ')}
    globalThis.mountSave = initToolAutoSave;
    globalThis.makeSignOut = createSignOutHandler;`, context);
  const cleanup = context.mountSave({
    toolId: 'text-compare', root, persistenceMode: mode,
    setPersistenceState: (state) => calls.states.push(state),
    setStatus: (status) => calls.status.push(status)
  });
  const signOut = context.makeSignOut({ beforeSignOut: cleanup.beforeSignOut, isActive: () => !disposed, setStatus: (status) => calls.status.push(status) });
  return {
    root, calls, signOut, clock, storage,
    beforeLeave: cleanup.beforeLeave,
    online: () => window.dispatchEvent(new Event('online')),
    pagehide: () => window.dispatchEvent(new Event('pagehide')),
    hide: () => { document.visibilityState = 'hidden'; document.dispatchEvent(new Event('visibilitychange')); },
    accountDataDeleted: () => document.dispatchEvent(new CustomEvent('tools:account-data-deleted')),
    cleanup: () => { disposed = true; cleanup(); },
    setOwner: (next) => { owner = next; },
    edit: (next) => { root.value = next; root.dispatchEvent(new Event('input')); },
    tabChanged: () => document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: 'text-compare' } })),
    save: () => document.dispatchEvent(new CustomEvent('tools:save-session', { detail: { toolId: 'text-compare' } })),
    get active() { return activeSession; },
    get rootCleared() { return rootCleared; },
    get session() { return sessionParam; }
  };
}

async function restoreBoundaries() {
  const fresh = harness();
  await turn();
  assert.equal(fresh.root.value, 'Saved work', 'a pristine tool on a new device should continue its latest saved work');
  assert.equal(fresh.active, 'latest');
  assert.equal(fresh.calls.applied, 1);
  assert.equal(fresh.calls.save.length, 0, 'restoring must not trigger an automatic cloud save');
  fresh.cleanup();

  const existing = harness({ value: 'Typed while authentication loaded' });
  await turn();
  assert.equal(existing.calls.list.length, 0, 'input present before auth/bootstrap completes must prevent automatic discovery');
  assert.equal(existing.root.value, 'Typed while authentication loaded');
  assert.equal(existing.calls.states.at(-1), 'dirty', 'preserved guest work must remain eligible for account saving');
  existing.cleanup();

  const explicit = harness({ session: 'chosen-session' });
  await turn();
  assert.equal(explicit.calls.list.length, 0, 'an explicit continuation link must take priority over discovery');
  assert.equal(explicit.calls.get[0].sessionId, 'chosen-session');
  explicit.cleanup();

  const listing = deferred();
  const duringLookup = harness({ list: () => listing.promise });
  duringLookup.edit('Newer work');
  duringLookup.save();
  await turn();
  assert.equal(duringLookup.calls.save.length, 0, 'saving must wait until latest-session lookup resolves');
  listing.resolve({ sessions: [{ toolId: 'text-compare', sessionId: 'old-cloud-work' }] });
  await turn();
  assert.equal(duringLookup.calls.get.length, 0, 'edits during discovery must cancel automatic cloud restoration');
  assert.equal(duringLookup.active, '', 'canceled discovery must not associate fresh work with an older session');
  duringLookup.save();
  await turn();
  assert.equal(duringLookup.calls.save[0].sessionId, undefined);
  assert.equal(duringLookup.calls.save[0].snapshot.fields.text, 'Newer work');
  duringLookup.cleanup();

  const loading = deferred();
  const duringLoad = harness({ get: () => loading.promise });
  await turn();
  duringLoad.edit('Edits while saved work loads');
  loading.resolve({ session: { version: 9, snapshot: { fields: { text: 'Old cloud data' } } } });
  await turn();
  assert.equal(duringLoad.root.value, 'Edits while saved work loads');
  duringLoad.save();
  await turn();
  assert.equal(duringLoad.calls.save[0].expectedVersion, 9, 'preserving edits must still resolve the saved record version safely');
  duringLoad.cleanup();

  const foreign = deferred();
  const switched = harness({ get: () => foreign.promise });
  await turn();
  switched.setOwner('person-b');
  foreign.resolve({ session: { version: 3, snapshot: { fields: { text: 'Person A private work' } } } });
  await turn();
  assert.equal(switched.root.value, '', 'late responses from a previous account must never populate the next account');
  assert.equal(switched.calls.applied, 0);
  switched.cleanup();

  const departedList = deferred();
  const departed = harness({ list: () => departedList.promise });
  departed.cleanup();
  departedList.resolve({ sessions: [{ toolId: 'text-compare', sessionId: 'late' }] });
  await turn();
  assert.equal(departed.calls.get.length, 0, 'discovery must stop when its route has unmounted');

  const missing = deferred();
  const editedMissing = harness({ session: 'deleted', get: () => missing.promise });
  editedMissing.edit('Keep this input');
  missing.reject(Object.assign(new Error('Gone'), { status: 404 }));
  await turn();
  assert.equal(editedMissing.root.value, 'Keep this input');
  assert.equal(editedMissing.calls.states.at(-1), 'dirty', 'a missing cloud record must not disable saving newer local input');
  editedMissing.save();
  await turn();
  assert.equal(editedMissing.calls.save[0].sessionId, undefined);
  editedMissing.cleanup();

  const colors = harness({ extraFields: [{ tagName: 'INPUT', type: 'color', value: '#2cb67d', defaultValue: '#2CB67D' }] });
  await turn();
  assert.equal(colors.calls.list.length, 1, 'browser normalization of an unchanged color must not count as user input');
  colors.cleanup();
}

async function deletionBoundaries() {
  async function run({ switchOwner }) {
    let owner = 'person-a';
    let opened = false;
    let click;
    let clearCalls = 0;
    let resetCalls = 0;
    const events = [];
    const deletion = deferred();
    const elements = new Map();
    const modalEl = {
      setAttribute() {},
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, { textContent: '', innerHTML: '', hidden: false });
        return elements.get(selector);
      },
      addEventListener(name, handler) { if (name === 'click') click = handler; }
    };
    const context = vm.createContext({
      CustomEvent, console,
      document: {
        createElement: () => modalEl, body: { appendChild() {} }, addEventListener() {},
        dispatchEvent: (event) => events.push(event.type)
      },
      window: {
        prompt: () => 'DELETE',
        ToolsAuth: { getAuth: () => ({}), authIsValid: () => true, getUser: () => ({ sub: owner, email: `${owner}@example.test` }) },
        ToolsState: { getDashboard: async () => ({}), deleteAllAccountData: () => deletion.promise }
      },
      createModalController: () => ({
        open() { opened = true; }, close(options) { opened = false; options?.onFinish?.(); },
        isOpen: () => opened, trapFocus() {}
      }),
      TOOL_CATALOG: {}, renderContinueWork: () => '', escapeHtml: (value) => String(value),
      getToolAccountCapabilities: () => ({ persistence: 'autosave', savePrivacyNote: 'Work saves automatically.' }),
      clearActiveSessionIds: () => { clearCalls += 1; }, setSessionParam: () => { resetCalls += 1; },
      logAsyncError: (label, error) => { throw error; }
    });
    vm.runInContext(`${section('  const initAccountModal = ', '  const initSessionModal = ')}
      globalThis.account = initAccountModal();`, context);
    context.account.open();
    await turn();
    const action = {
      dataset: { toolsAccountAction: 'delete-all-data' },
      closest: (selector) => selector === '[data-tools-account-action]' ? action : null
    };
    const completing = click({ target: action });
    context.account.close();
    if (switchOwner) owner = 'person-b';
    deletion.resolve({ ok: true });
    await completing;
    return { clearCalls, resetCalls, events, status: elements.get('[data-tools-account="modal-status"]').textContent };
  }
  const closed = await run({ switchOwner: false });
  assert.equal(closed.clearCalls, 1, 'closing Account during deletion must still clear the same account session association');
  assert.equal(closed.resetCalls, 1);
  assert.deepEqual(closed.events, ['tools:account-data-deleted']);
  assert.equal(closed.status, '', 'a completed deletion must not rewrite a closed dialog');
  const switched = await run({ switchOwner: true });
  assert.equal(switched.clearCalls, 0, 'a prior account deletion must not clear the new account cache');
  assert.equal(switched.resetCalls, 0);
  assert.deepEqual(switched.events, [], 'a prior account deletion must not reset the new account work');
}

async function signOutBoundaries() {
  const cancel = harness({ value: 'Unsaved work', confirm: () => false, mode: 'manual' });
  assert.equal(await cancel.signOut(), false);
  assert.deepEqual(cancel.calls.confirm, ['Sign out and discard your unsaved changes?']);
  assert.equal(cancel.calls.signOut, 0, 'canceling discard must preserve authentication');
  assert.equal(cancel.calls.save.length, 0, 'manual sign-out must never automatically upload inputs');
  assert.equal(cancel.root.value, 'Unsaved work');
  assert.equal(cancel.calls.states.at(-1), 'dirty');
  cancel.cleanup();

  const discard = harness({ value: 'Unsaved work', confirm: () => true, mode: 'manual' });
  assert.equal(await discard.signOut(), true);
  assert.equal(discard.calls.signOut, 1, 'an accepted discard should continue to the real logout handler');
  assert.equal(discard.calls.save.length, 0);
  discard.cleanup();

  const clean = harness({ list: async () => ({ sessions: [] }) });
  await turn();
  assert.equal(await clean.signOut(), true);
  assert.equal(clean.calls.confirm.length, 0, 'clean sign-out should have no confirmation');
  clean.cleanup();

  const saving = deferred();
  const pending = harness({ value: 'Saving this work', save: () => saving.promise });
  pending.save();
  const firstSignOut = pending.signOut();
  assert.equal(pending.signOut(), firstSignOut, 'duplicate account controls must share one pending sign-out');
  await turn();
  assert.equal(pending.calls.signOut, 0, 'logout must not clear auth while a Save is in flight');
  saving.resolve({ session: { sessionId: 'saved-before-logout', version: 1 } });
  assert.equal(await firstSignOut, true);
  assert.equal(pending.calls.signOut, 1);
  assert.equal(pending.calls.confirm.length, 0, 'successful Save must clear dirty state rather than asking to discard saved work');
  assert.equal(pending.calls.states.at(-1), 'clean');
  pending.cleanup();

  const failedSave = deferred();
  const failed = harness({ value: 'Keep failed work', save: () => failedSave.promise });
  failed.save();
  const failedSignOut = failed.signOut();
  failedSave.reject(new Error('Save unavailable'));
  assert.equal(await failedSignOut, false);
  assert.equal(failed.calls.signOut, 0, 'a pending Save failure must stop logout and retain the page');
  assert.equal(failed.calls.states.at(-1), 'error');
  failed.cleanup();

  const firstSave = deferred();
  const newer = harness({ value: 'Original save', save: () => firstSave.promise, confirm: () => false });
  newer.save();
  const newerSignOut = newer.signOut();
  newer.edit('New edits after Save started');
  firstSave.resolve({ session: { sessionId: 'first-revision', version: 1 } });
  assert.equal(await newerSignOut, true);
  assert.equal(newer.calls.save.length, 2, 'sign-out must flush edits made while the first save was in flight');
  assert.equal(newer.calls.save[1].snapshot.fields.text, 'New edits after Save started');
  assert.equal(newer.calls.signOut, 1);
  assert.equal(newer.calls.confirm.length, 0, 'ordinary autosave must not ask to discard unsaved work');
  newer.cleanup();

  const departedSave = deferred();
  const departed = harness({ value: 'Pending route', save: () => departedSave.promise });
  departed.save();
  const departedSignOut = departed.signOut();
  departed.cleanup();
  departedSave.resolve({ session: { sessionId: 'late-save', version: 1 } });
  assert.equal(await departedSignOut, false);
  assert.equal(departed.calls.signOut, 0, 'a departed route must never complete an old sign-out intent');

  const automatic = harness({ value: 'Opt-in automatic work', mode: 'autosave' });
  assert.equal(await automatic.signOut(), true);
  assert.equal(automatic.calls.save.length, 1, 'an existing autosave tool should flush its dirty work before logout');
  assert.equal(automatic.calls.confirm.length, 0);
  automatic.cleanup();
}

function automaticAccountPolicy() {
  const context = vm.createContext({});
  vm.runInContext(`${section('  const TOOL_ACCOUNT_CAPABILITIES = ', '  const $ = ')}
    globalThis.capabilities = getToolAccountCapabilities;`, context);
  assert.equal(context.capabilities({ page: 'text-compare', toolId: 'text-compare' }).persistence, 'autosave', 'ordinary account work must save automatically without a page opt-in');
  assert.equal(context.capabilities({ page: 'ga4-utm-performance', toolId: 'ga4-utm-performance' }).persistence, 'autosave', 'GA4 uses its existing safe capture hooks for automatic saving');
  assert.equal(context.capabilities({ page: 'job-application-tracker', toolId: 'job-application-tracker' }).persistence, 'custom');
  assert.equal(context.capabilities({ page: 'transcribe', toolId: 'transcribe' }).persistence, 'custom');
  assert.equal(context.capabilities({ page: 'short-links', toolId: 'short-links' }).persistence, 'none');
  assert(!/<button[^>]*data-tools-action="save-session"/.test(source), 'no shared account state may render a Save or Retry save button');
  assert(!source.includes('data-tools-account="save-privacy"'), 'the removed Save control must not leave a disconnected tooltip');
}

async function autosaveBoundaries() {
  const empty = harness({ list: async () => ({ sessions: [] }) });
  await turn();
  await empty.clock.advance(40000);
  assert.equal(empty.calls.save.length, 0, 'idle signed-in tools must not create empty sessions');
  empty.edit('First input');
  await empty.clock.advance(600);
  empty.edit('Last input in the same burst');
  await empty.clock.advance(999);
  assert.equal(empty.calls.save.length, 0, 'typing bursts should coalesce until the debounce expires');
  await empty.clock.advance(1);
  assert.equal(empty.calls.save.length, 1, 'omitting persistenceMode must still automatically save');
  assert.equal(empty.calls.save[0].snapshot.fields.text, 'Last input in the same burst');
  await empty.clock.advance(40000);
  assert.equal(empty.calls.save.length, 1, 'periodic fallback must not rewrite clean work');
  empty.cleanup();

  const first = deferred();
  const second = deferred();
  let saveNumber = 0;
  const serial = harness({ list: async () => ({ sessions: [] }), save: () => (++saveNumber === 1 ? first.promise : second.promise) });
  await turn();
  serial.edit('Revision one');
  await serial.clock.advance(1000);
  serial.edit('Revision two');
  await serial.clock.advance(1000);
  assert.equal(serial.calls.save.length, 1, 'autosave must never start a concurrent write against the same version');
  first.resolve({ session: { sessionId: 'one-session', version: 1 } });
  await turn();
  await serial.clock.advance(1000);
  assert.equal(serial.calls.save.length, 2);
  assert.equal(serial.calls.save[1].sessionId, 'one-session');
  assert.equal(serial.calls.save[1].expectedVersion, 1);
  assert.equal(serial.calls.save[1].snapshot.fields.text, 'Revision two', 'the automatic follow-up must capture the newest fields');
  second.resolve({ session: { sessionId: 'one-session', version: 2 } });
  await turn();
  assert.equal(serial.calls.states.at(-1), 'clean');
  serial.cleanup();

  let onlineAttempt = 0;
  const online = harness({ list: async () => ({ sessions: [] }), save: async () => {
    if (++onlineAttempt === 1) throw new Error('Offline');
    return { session: { sessionId: 'retry-session', version: 1 } };
  } });
  await turn();
  online.edit('Retry this work');
  await online.clock.advance(1000);
  assert.equal(online.calls.states.at(-1), 'error');
  await online.clock.advance(1000);
  assert.equal(online.calls.save.length, 1, 'failures must not cause a rapid retry loop');
  online.online();
  await online.clock.advance(1);
  assert.equal(online.calls.save.length, 2, 'restored connectivity should retry dirty work immediately');
  assert.equal(online.calls.states.at(-1), 'clean');
  online.cleanup();

  let fallbackAttempt = 0;
  const fallback = harness({ list: async () => ({ sessions: [] }), save: async () => {
    if (++fallbackAttempt === 1) throw new Error('Temporary failure');
    return { session: { sessionId: 'fallback-retry', version: 1 } };
  } });
  await turn();
  fallback.edit('Wait for automatic retry');
  await fallback.clock.advance(21000);
  assert.equal(fallback.calls.save.length, 2, 'a bounded timer must retry even without an online event');
  fallback.cleanup();

  const leavingFirst = deferred();
  const leavingSecond = deferred();
  let leaveSaveNumber = 0;
  const leaving = harness({ value: 'Before navigation', save: () => (++leaveSaveNumber === 1 ? leavingFirst.promise : leavingSecond.promise) });
  let left = false;
  const leave = leaving.beforeLeave().then((result) => { left = true; return result; });
  await turn();
  leaving.edit('Newest work before navigation');
  leavingFirst.resolve({ session: { sessionId: 'leave-session', version: 1 } });
  await turn();
  assert.equal(left, false);
  assert.equal(leaving.calls.save.length, 2, 'route leave must drain edits made during its first save');
  assert.equal(leaving.calls.save[1].snapshot.fields.text, 'Newest work before navigation');
  leavingSecond.resolve({ session: { sessionId: 'leave-session', version: 2 } });
  assert.equal(await leave, true);
  assert.equal(leaving.calls.states.at(-1), 'clean');
  leaving.cleanup();

  const hiding = harness({ list: async () => ({ sessions: [] }) });
  await turn();
  hiding.edit('Page is closing');
  hiding.pagehide();
  await turn();
  assert.equal(hiding.calls.save.length, 1, 'pagehide must flush before a debounce would fire');
  assert.equal(hiding.calls.save[0].keepalive, true);
  hiding.cleanup();

  const guest = harness({ value: 'Guest work stays local', initialOwner: '' });
  await guest.clock.advance(40000);
  assert.equal(guest.calls.save.length, 0, 'automatic saving must never upload unsigned guest input');
  guest.cleanup();

  const switchedTimer = harness({ list: async () => ({ sessions: [] }) });
  await turn();
  switchedTimer.edit('Person A pending autosave');
  switchedTimer.setOwner('person-b');
  await switchedTimer.clock.advance(40000);
  assert.equal(switchedTimer.calls.save.length, 0, 'an old account timer must not send its captured fields using a new account identity');
  switchedTimer.cleanup();

  const canceledTimer = harness({ list: async () => ({ sessions: [] }) });
  await turn();
  canceledTimer.edit('Leaving this mounted instance');
  canceledTimer.cleanup();
  await canceledTimer.clock.advance(40000);
  assert.equal(canceledTimer.calls.save.length, 0, 'disposing a route must cancel its automatic save timers');

  for (const nextOwner of ['person-b', '']) {
    const second = { id: 'revised', tagName: 'TEXTAREA', value: 'Person A private revised text', defaultValue: '' };
    const privateOutput = { kind: 'html', html: '<p>Person A private comparison result</p>' };
    const previousAccount = harness({
      value: 'Person A private original text', output: privateOutput, extraFields: [second],
      session: 'person-a-session', previousOwner: 'person-a', initialOwner: nextOwner,
      list: async () => ({ sessions: [] })
    });
    assert.equal(previousAccount.calls.reload, nextOwner ? 1 : 0, 'a different signed-in account needs a fresh document; hosted sign-out must retain its own redirect');
    assert.equal(previousAccount.rootCleared, Boolean(nextOwner), 'remove inherited form and result nodes before the browser can restore form state across reload');
    assert.equal(previousAccount.session, '', 'an inherited session URL must not be requested by the next account');
    assert.equal(previousAccount.root.dataset.toolsDraftOwner, 'person-a', 'inherited state must retain its owner until the document resets');
    previousAccount.tabChanged();
    previousAccount.edit('Only the first draft was edited');
    previousAccount.save();
    previousAccount.pagehide();
    await previousAccount.clock.advance(40000);
    previousAccount.cleanup();
    assert.equal(second.value, 'Person A private revised text', 'the scenario retains untouched second-field data while navigation is pending');
    assert.equal(previousAccount.root.output, privateOutput, 'the scenario includes private output retained outside the form');
    assert.equal(previousAccount.calls.list.length + previousAccount.calls.get.length, 0, 'inherited tool state must not be merged with the next account session');
    assert.equal(previousAccount.calls.capture, 0, 'input, tab, save, pagehide, and cleanup events must never capture either inherited field or output');
    assert.equal(previousAccount.calls.save.length, 0, 'inherited state must never upload while the document reset or hosted logout is pending');
    assert.equal(previousAccount.storage.size, 0, 'inherited state must not become a guest or new-account pending draft');
    assert.equal(previousAccount.root.dataset.toolsDraftOwner, 'person-a', 'guest input must not relabel inherited work and bypass the next sign-in reset');
  }

  const ownSecond = { id: 'revised', tagName: 'TEXTAREA', value: '', defaultValue: '' };
  const ownOutput = { kind: 'html', html: '<p>Person B comparison</p>' };
  const freshAccount = harness({
    initialOwner: 'person-b', extraFields: [ownSecond],
    get: async () => ({ session: { version: 2, snapshot: { fields: { text: 'Person B original', revised: 'Person B revised' }, output: ownOutput } } })
  });
  await turn();
  assert.equal(freshAccount.calls.reload, 0, 'the fresh document must not enter a reset loop');
  assert.equal(freshAccount.root.value, 'Person B original');
  assert.equal(ownSecond.value, 'Person B revised');
  assert.equal(freshAccount.root.output, ownOutput, 'normal continuation restores only the new account output');
  freshAccount.edit('Person B newer original');
  await freshAccount.clock.advance(1000);
  assert.deepEqual(JSON.parse(JSON.stringify(freshAccount.calls.save[0].snapshot)), {
    fields: { text: 'Person B newer original', revised: 'Person B revised' }, output: ownOutput
  }, 'normal autosaving must resume with both new-account fields and output');
  freshAccount.cleanup();
}

async function pendingDraftRecovery() {
  const key = 'toolsPendingDraft:person-a:text-compare';
  const storage = new Map();
  const interruptedSave = deferred();
  const interrupted = harness({ value: 'Request already in flight', storage, save: () => interruptedSave.promise });
  await interrupted.clock.advance(1000);
  interrupted.edit('Newest text before tab close');
  interrupted.pagehide();
  assert.equal(JSON.parse(storage.get(key)).snapshot.fields.text, 'Newest text before tab close', 'pagehide must capture newer edits synchronously even while an older request is pending');
  interrupted.cleanup();
  const reloaded = harness({ storage });
  assert.equal(reloaded.root.value, 'Newest text before tab close', 'the owning account must recover its unsynced draft after reload');
  assert.equal(reloaded.calls.list.length, 0, 'recovered local work must take precedence over an older remote continuation');
  await reloaded.clock.advance(1000);
  assert.equal(reloaded.calls.save[0].snapshot.fields.text, 'Newest text before tab close');
  assert.equal(storage.has(key), false, 'a successful matching save should remove its pending recovery record');
  interruptedSave.resolve({ session: { sessionId: 'interrupted-copy', version: 1 } });
  await turn();
  assert.equal(storage.has(key), false, 'a disposed old request must not recreate a recovered draft');
  reloaded.cleanup();

  const pending = { owner: 'person-a', dirty: true, sessionId: 'pending-session', sessionVersion: 3, snapshot: { fields: { text: 'Private pending work' } }, outputSummary: '' };
  const isolatedStore = new Map([[key, JSON.stringify(pending)]]);
  const other = harness({ storage: isolatedStore, initialOwner: 'person-b', list: async () => ({ sessions: [] }) });
  await turn();
  await other.clock.advance(40000);
  assert.equal(other.root.value, '', 'another account must not see a pending recovery snapshot');
  assert.equal(other.calls.save.length, 0, 'another account must not upload a pending recovery snapshot');
  assert.equal(isolatedStore.has(key), true, 'another account must not delete the owning account recovery record');
  other.cleanup();

  const fresh = harness({ storage: new Map([[key, JSON.stringify(pending)]]), value: 'New input during authentication' });
  assert.equal(fresh.root.value, 'New input during authentication', 'pending recovery must not overwrite fresh form input');
  fresh.cleanup();
  const selected = harness({ storage: new Map([[key, JSON.stringify(pending)]]), session: 'explicit-other-session' });
  await turn();
  assert.equal(selected.calls.get[0].sessionId, 'explicit-other-session', 'an explicitly selected session must not be replaced by an unrelated recovery draft');
  assert.equal(selected.root.value, 'Saved work');
  selected.cleanup();

  const matchingSave = deferred();
  const multipleTabsStore = new Map();
  const earlierTab = harness({ value: 'Earlier tab request', storage: multipleTabsStore, save: () => matchingSave.promise });
  await earlierTab.clock.advance(1000);
  const newerTabDraft = JSON.stringify({ ...pending, snapshot: { fields: { text: 'A newer tab still needs saving' } } });
  multipleTabsStore.set(key, newerTabDraft);
  matchingSave.resolve({ session: { sessionId: 'earlier-tab', version: 1 } });
  await turn();
  assert.equal(multipleTabsStore.get(key), newerTabDraft, 'a successful earlier request must not clear a newer tab recovery record');
  earlierTab.cleanup();

  for (const code of ['VERSION_CONFLICT', 'SESSION_EXPIRED']) {
    let attempt = 0;
    const conflicting = harness({ storage: new Map([[key, JSON.stringify(pending)]]), save: async () => {
      if (++attempt === 1) throw Object.assign(new Error(code), { data: { code } });
      return { session: { sessionId: 'preserved-as-new', version: 1 } };
    } });
    assert.equal(conflicting.root.value, 'Private pending work');
    await conflicting.clock.advance(1000);
    assert.equal(conflicting.calls.save[0].sessionId, 'pending-session');
    assert.equal(conflicting.calls.save[0].expectedVersion, 3);
    assert.equal(JSON.parse(conflicting.storage.get(key)).snapshot.fields.text, 'Private pending work', `${code} must retain the complete local draft`);
    conflicting.online();
    await conflicting.clock.advance(1);
    assert.equal(conflicting.calls.save[1].sessionId, undefined, `${code} should preserve both copies by retrying as a new continuation`);
    assert.equal(conflicting.calls.save[1].snapshot.fields.text, 'Private pending work');
    assert.equal(conflicting.storage.has(key), false);
    conflicting.cleanup();
  }
}

async function deletionDuringAutosave() {
  for (const outcome of ['resolved', 'rejected']) {
    const pending = deferred();
    let attempt = 0;
    const running = harness({ value: 'Account data being deleted', save: async () => {
      if (++attempt === 1) return pending.promise;
      return { session: { sessionId: 'fresh-after-deletion', version: 1 } };
    } });
    await running.clock.advance(1000);
    assert.equal(running.calls.save.length, 1);
    assert.equal(running.storage.has('toolsPendingDraft:person-a:text-compare'), true);
    running.accountDataDeleted();
    assert.equal(running.active, '');
    assert.equal(running.storage.has('toolsPendingDraft:person-a:text-compare'), false);
    if (outcome === 'resolved') pending.resolve({ session: { sessionId: 'deleted-session', version: 9 } });
    else pending.reject(new Error('Old save failed after deletion'));
    await turn();
    await running.clock.advance(40000);
    assert.equal(running.active, '', `a late ${outcome} save must not reattach deleted account data`);
    assert.equal(running.session, '', `a late ${outcome} save must not restore the deleted session URL`);
    assert.equal(running.storage.has('toolsPendingDraft:person-a:text-compare'), false, `a late ${outcome} save must not recreate the pending draft`);
    assert.equal(running.calls.save.length, 1, `a late ${outcome} save must not schedule account data for automatic recreation`);
    assert.equal(running.calls.states.at(-1), 'clean');
    running.edit('Fresh work after deletion');
    await running.clock.advance(1000);
    assert.equal(running.calls.save.length, 2, 'fresh input after deletion should resume ordinary saving');
    assert.equal(running.calls.save[1].sessionId, undefined);
    assert.equal(running.calls.save[1].expectedVersion, 0);
    assert.equal(running.calls.save[1].snapshot.fields.text, 'Fresh work after deletion');
    assert.equal(running.active, 'fresh-after-deletion');
    running.cleanup();
  }
}

async function deletionDuringReads() {
  const sessionRead = deferred();
  const loading = harness({ session: 'deleted-work', get: () => sessionRead.promise });
  loading.accountDataDeleted();
  sessionRead.resolve({ session: { version: 8, snapshot: { fields: { text: 'Deleted snapshot must stay deleted' } } } });
  await turn();
  await loading.clock.advance(40000);
  assert.equal(loading.root.value, '', 'a session read completing after account deletion must not restore its old snapshot');
  assert.equal(loading.calls.applied, 0, 'stale reads must not notify tool output hooks after deletion');
  assert.equal(loading.active, '');
  assert.equal(loading.calls.save.length, 0);
  assert.equal(loading.storage.size, 0, 'a stale session read must not recreate pending work');
  loading.edit('Fresh input after deleting loaded work');
  await loading.clock.advance(1000);
  assert.equal(loading.calls.save[0].sessionId, undefined);
  assert.equal(loading.calls.save[0].expectedVersion, 0, 'the discarded read must not leak its prior version into fresh work');
  loading.cleanup();

  const listing = deferred();
  const discovering = harness({ list: () => listing.promise });
  discovering.accountDataDeleted();
  listing.resolve({ sessions: [{ toolId: 'text-compare', sessionId: 'deleted-latest' }] });
  await turn();
  await discovering.clock.advance(40000);
  assert.equal(discovering.calls.get.length, 0, 'latest-session discovery completing after deletion must not start a new read of deleted work');
  assert.equal(discovering.active, '');
  assert.equal(discovering.session, '');
  assert.equal(discovering.calls.save.length, 0);
  assert.equal(discovering.storage.size, 0);
  discovering.cleanup();
}

(async () => {
  accountKeys();
  continuationRows();
  automaticAccountPolicy();
  await restoreBoundaries();
  await deletionBoundaries();
  await signOutBoundaries();
  await autosaveBoundaries();
  await pendingDraftRecovery();
  await deletionDuringAutosave();
  await deletionDuringReads();
  console.log('Account continuation passed: automatic saving, coalesced and serial writes, retries, guarded exit, latest work, and account isolation.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
