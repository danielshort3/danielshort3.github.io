'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'js/tools/job-application-tracker.js'), 'utf8');
function section(startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert(start >= 0 && end > start, `Missing tracker section ${startMarker}`);
  return source.slice(start, end);
}
const authHelpers = section('  const syncAuthState =', '  const getSignedInLabel =');
const requestSource = section('const requestJson =', '  const requestAllItems =');
const uiSource = section('const updateAuthUI =', 'const requestJson =');

function harness(initialAuth, response) {
  let auth = initialAuth;
  const calls = [];
  const messages = [];
  const logs = [];
  const state = { auth, trackerReconnectRequired: false, trackerRejectedToken: '' };
  const updates = [];
  let signOuts = 0;
  const context = vm.createContext({
    state,
    config: { apiBase: '/api/job-tracker' },
    joinUrl: (base, route) => `${base}${route}`,
    ensureFreshAuth: async () => auth,
    authIsValid: (value) => Boolean(value?.valid),
    setAuthMessage: (...args) => messages.push(args),
    updateAuthUI: () => updates.push({ blocked: state.trackerReconnectRequired }),
    console: { error: (...args) => logs.push(args) },
    document: { dispatchEvent() { throw new Error('Request errors must not broadcast sign-out'); } },
    window: {
      ToolsAuth: {
        getAuth: () => auth,
        signOut: () => { signOuts += 1; },
        fetchWithAuth: async (url, options) => {
          calls.push({ url, options });
          return response({ url, options, auth });
        }
      }
    }
  });
  const api = vm.runInContext(`${authHelpers}\n${requestSource}\n({ requestJson, syncAuthState, reportRequestError });`, context);
  return { ...api, state, calls, messages, updates, logs, setAuth: (value) => { auth = value; }, signOuts: () => signOuts };
}
const jsonResponse = (status, body) => ({ ok: status >= 200 && status < 300, status, statusText: 'Fixture', text: async () => JSON.stringify(body) });

test('cookie-only recovery pauses tracker requests and preserves the shared account', async () => {
  const cookie = { valid: true, sessionMode: 'cookie', user: { email: 'fixture@example.test' } };
  const h = harness(cookie, ({ options, auth }) => {
    assert.equal(options.requireIdToken, true, 'The tracker must request bearer-only API access');
    if (!auth.idToken) throw Object.assign(new Error('Reconnect needed'), { code: 'TOOLS_ID_TOKEN_REQUIRED', status: 401 });
    return jsonResponse(200, { items: [] });
  });
  await assert.rejects(h.requestJson('/api/applications'), { code: 'TRACKER_RECONNECT_REQUIRED', status: 401 });
  await assert.rejects(h.requestJson('/api/analytics/dashboard'), { code: 'TRACKER_RECONNECT_REQUIRED' });
  assert.equal(h.calls.length, 1, 'Repeated loaders must stop at the tracker gate after reconnect becomes necessary');
  assert.equal(h.state.auth, cookie, 'The valid shared cookie session must remain signed in');
  assert.equal(h.signOuts(), 0);
  assert.equal(h.updates.length, 1, 'Concurrent expected failures must not repeatedly reset auth UI');
  assert.match(h.messages[0][0], /Sign in again.*tracker/);

  h.setAuth({ valid: true, idToken: 'fresh-fixture-token' });
  const recovered = await h.requestJson('/api/applications');
  assert(Array.isArray(recovered.items) && recovered.items.length === 0);
  assert.equal(h.calls.length, 2, 'A fresh bearer must reopen tracker access');
  assert.equal(h.state.trackerReconnectRequired, false);
  assert.equal(h.updates.at(-1).blocked, false, 'Successful reconnection must remove the notice');
});

test('a rejected bearer stops further requests without logging out other tools', async () => {
  const auth = { valid: true, idToken: 'rejected-fixture-token' };
  const h = harness(auth, () => jsonResponse(401, { error: 'Unauthorized' }));
  await assert.rejects(h.requestJson('/api/applications'), { code: 'TRACKER_RECONNECT_REQUIRED' });
  await assert.rejects(h.requestJson('/api/views'), { code: 'TRACKER_RECONNECT_REQUIRED' });
  assert.equal(h.calls.length, 1);
  assert.equal(h.state.auth, auth);
  assert.equal(h.signOuts(), 0);
  h.setAuth({ valid: true, idToken: 'replacement-fixture-token' });
  await h.syncAuthState();
  assert.equal(h.state.trackerReconnectRequired, false, 'The shared auth-changed refresh must unlock a new credential');
});

test('403 authorization failures remain visible without being treated as expired login', async () => {
  const auth = { valid: true, idToken: 'valid-fixture-token' };
  const h = harness(auth, () => jsonResponse(403, { error: 'You do not have access to this export.' }));
  let failure;
  try { await h.requestJson('/api/exports', { method: 'POST', body: { format: 'csv' } }); } catch (error) { failure = error; }
  assert.equal(failure.status, 403);
  assert.match(failure.message, /do not have access/);
  assert.equal(h.state.trackerReconnectRequired, false);
  assert.equal(h.state.auth, auth);
  assert.equal(h.signOuts(), 0);
  assert.equal(h.messages.length, 0);
  h.reportRequestError('Export failed', failure);
  assert.equal(h.logs.length, 1, 'Real permission failures must remain diagnosable');
  h.reportRequestError('Dashboard load failed', { code: 'TRACKER_RECONNECT_REQUIRED' });
  h.reportRequestError('Dashboard load failed', { code: 'TOOLS_ID_TOKEN_REQUIRED' });
  assert.equal(h.logs.length, 1, 'Expected reconnect states must not create repeated console errors');
});

test('reconnect UI distinguishes tracker access from shared account sign-in', () => {
  const button = (tab) => ({ dataset: { jobtrackTab: tab }, textContent: tab, setAttribute() {}, removeAttribute() {} });
  const state = { auth: { valid: true }, trackerReconnectRequired: true, entryType: 'application' };
  const els = {
    authReconnect: { hidden: true }, authReconnectMessage: { textContent: '' },
    signIn: button('sign-in'), signOut: button('sign-out'), jumpEntryButtons: [], jumpTabButtons: []
  };
  const tabs = { buttons: [button('account'), button('dashboard')] };
  const update = vm.runInNewContext(`${uiSource}\nupdateAuthUI;`, {
    state, els, tabs, authIsValid: (auth) => auth.valid, startAuthWatcher() {}, closeAuthModal() {}
  });
  update();
  assert.equal(els.authReconnect.hidden, false);
  assert.match(els.authReconnectMessage.textContent, /account is still signed in/);
  assert.equal(els.signIn.disabled, false);
  assert.equal(els.signOut.disabled, false);
  assert.equal(tabs.buttons[0].disabled, false);
  assert.equal(tabs.buttons[1].disabled, true);
  state.trackerReconnectRequired = false;
  update();
  assert.equal(els.authReconnect.hidden, true);
  assert.equal(tabs.buttons[1].disabled, false);

  const html = fs.readFileSync(path.join(root, 'pages/job-application-tracker.html'), 'utf8');
  assert.match(html, /data-jobtrack="auth-reconnect"[^>]*hidden/);
  assert.match(html, /data-jobtrack="auth-reconnect-sign-in">Sign in again<\/button>/);
  assert.match(source, /ToolsAuth\.signIn\(\{\s*returnTo:/, 'The reconnect action must preserve the current tracker route');
});
