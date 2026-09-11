'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.resolve(__dirname, '../../js/accounts/tools-auth.js'), 'utf8');
const now = Math.floor(Date.now() / 1000);
const token = (sub = 'account', exp = now + 3600) => `header.${Buffer.from(JSON.stringify({ sub, exp, email_verified: true })).toString('base64url')}.signature`;
const expired = () => ({ idToken: token('account', now - 60), refreshToken: 'test-refresh-token', accessToken: 'old-access', expiresAt: (now - 60) * 1000 });
const session = () => ({ expiresAt: now + 3600, user: { sub: 'account', emailVerified: true } });
const response = (data, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};

function createClient({ initial, key = 'toolsAuth', mode = 'dual', fetchImpl } = {}) {
  const stored = new Map(initial ? [[key, JSON.stringify(initial)]] : []);
  const storage = {
    getItem: (name) => stored.get(name) || null,
    setItem: (name, value) => stored.set(name, String(value)),
    removeItem: (name) => stored.delete(name)
  };
  const listeners = new Map();
  const calls = [];
  const redirects = [];
  const window = {
    location: {
      origin: 'http://127.0.0.1:4181', pathname: '/tools/job-application-tracker', search: '', hash: '',
      assign: value => redirects.push(value)
    },
    TOOLS_AUTH_CONFIG: { sessionMode: mode, cognitoDomain: 'auth.example.com', cognitoClientId: 'public-test-client' },
    addEventListener: (name, callback) => listeners.set(name, callback)
  };
  const context = {
    window, document: { body: { dataset: {} }, dispatchEvent() {} }, localStorage: storage, sessionStorage: storage,
    Headers, URL, URLSearchParams, AbortController, atob, console,
    fetch: async (url, options) => {
      calls.push({ url, options });
      if (fetchImpl) return fetchImpl(url, options, calls);
      if (url.endsWith('/oauth2/token')) return response({ id_token: token(), access_token: 'new-access' });
      if (url === '/api/tools/auth/exchange' || url === '/api/tools/auth/session') return response(session());
      return response({ ok: true });
    }
  };
  vm.runInNewContext(source, context, { filename: 'js/accounts/tools-auth.js' });
  return { auth: window.ToolsAuth, calls, stored, listeners, context, redirects };
}

async function run() {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };

  const valid = createClient({ initial: { idToken: token(), expiresAt: (now + 3600) * 1000 } });
  const result = await valid.auth.fetchWithAuth('/api/job-tracker/api/applications', { requireIdToken: true });
  check(result.status === 200 && valid.calls.length === 1, 'A valid bearer must bypass refresh and return the API response.');
  check(valid.calls[0].options.headers.get('Authorization') === `Bearer ${token()}`, 'The current ID token must be forwarded.');
  check(valid.calls[0].options.credentials === 'same-origin' && !('requireIdToken' in valid.calls[0].options), 'The bearer requirement must not leak into native fetch options.');
  await valid.auth.fetchWithAuth('/api/tools/me', { headers: { Authorization: 'Bearer explicit' } });
  check(valid.calls[1].options.headers.get('Authorization') === 'Bearer explicit', 'An explicit authorization header must remain unchanged.');

  const refreshGate = deferred();
  const renewed = createClient({ initial: expired(), fetchImpl: async (url) => {
    if (url.endsWith('/oauth2/token')) return refreshGate.promise;
    return response(session());
  } });
  check(renewed.auth.getAuth() === null, 'Expired tokens must remain unavailable to synchronous UI access checks.');
  const concurrent = Array.from({ length: 6 }, () => renewed.auth.fetchWithAuth('/api/job-tracker/api/applications', { requireIdToken: true }));
  check(renewed.calls.length === 1 && renewed.calls[0].url.endsWith('/oauth2/token'), 'Concurrent requests must renew the expired bearer once before restoring cookies or calling the API.');
  const grant = new URLSearchParams(renewed.calls[0].options.body);
  check(grant.get('grant_type') === 'refresh_token' && grant.get('refresh_token') === 'test-refresh-token', 'Renewal must use the retained refresh token.');
  refreshGate.resolve(response({ id_token: token('renewed-account'), refresh_token: 'rotated-refresh-token', access_token: 'new-access' }));
  await Promise.all(concurrent);
  const storedRenewed = JSON.parse(renewed.stored.get('toolsAuth'));
  check(renewed.calls.filter((call) => call.url === '/api/tools/auth/exchange').length === 1, 'The shared renewal must establish only one server session.');
  check(renewed.calls.filter((call) => call.url.endsWith('/api/applications')).every((call) => call.options.headers.get('Authorization') === `Bearer ${token('renewed-account')}`), 'All waiting API calls must receive the renewed bearer.');
  check(storedRenewed.refreshToken === 'rotated-refresh-token' && storedRenewed.claims.sub === 'renewed-account' && storedRenewed.serverSession, 'Renewal must persist rotated refresh tokens, fresh claims, and the dual session.');
  check(!renewed.calls.some((call) => call.url === '/api/tools/auth/session'), 'A renewable bearer must not be replaced by a cookie-only restoration.');

  const legacy = createClient({ initial: expired(), key: 'jobTrackerAuth', mode: 'legacy' });
  const legacyAuth = await legacy.auth.ensureFreshAuth();
  check(legacyAuth.refreshToken === 'test-refresh-token' && legacy.stored.has('toolsAuth'), 'Expired legacy credentials must renew and migrate, retaining an unrotated refresh token.');
  check(legacy.calls.length === 1, 'Legacy mode must not establish or restore server sessions.');

  const nearExpiry = createClient({ initial: { ...expired(), idToken: token('account', now + 30), expiresAt: (now + 30) * 1000 } });
  await nearExpiry.auth.ensureFreshAuth();
  check(nearExpiry.calls[0].url.endsWith('/oauth2/token'), 'A token within the expiry safety window must renew before use.');

  const cookie = createClient({ initial: expired(), mode: 'cookie' });
  const cookieAuth = await cookie.auth.ensureFreshAuth();
  check(cookieAuth.sessionOnly && !cookieAuth.idToken && !cookieAuth.refreshToken, 'Cookie mode must continue to persist only its server session after renewal.');

  const restoreGate = deferred();
  const restored = createClient({ fetchImpl: (url) => url === '/api/tools/auth/session' ? restoreGate.promise : response({ ok: true }) });
  const restorations = Array.from({ length: 6 }, () => restored.auth.ensureFreshAuth());
  check(restored.calls.length === 1, 'Concurrent missing credentials must perform one cookie lookup.');
  restoreGate.resolve(response(session()));
  await Promise.all(restorations);
  const callCount = restored.calls.length;
  await assert.rejects(restored.auth.fetchWithAuth('/api/job-tracker/api/applications', { requireIdToken: true }), (error) => error.code === 'TOOLS_ID_TOKEN_REQUIRED' && error.status === 401);
  check(restored.calls.length === callCount && restored.auth.getAuth()?.sessionOnly, 'A bearer-only tool must fail before its request while preserving a valid cookie account.');
  await restored.auth.fetchWithAuth('/api/tools/me');
  check(!restored.calls.at(-1).options.headers.has('Authorization') && restored.calls.at(-1).options.credentials === 'same-origin', 'Ordinary same-origin Tools APIs must continue accepting the cookie session.');
  await assert.rejects(restored.auth.fetchWithAuth('https://external.example.com/api'), /Cookie-only tools sessions require a same-origin API proxy/);
  checks += 1;

  const rejected = createClient({ initial: expired(), fetchImpl: (url) => url.endsWith('/oauth2/token') ? response({ error: 'invalid_grant' }, 400) : response(session()) });
  check((await rejected.auth.ensureFreshAuth()).sessionOnly && !rejected.calls.some((call) => call.url.endsWith('/logout')), 'A rejected refresh token must fall back to a still-valid server session without logging it out.');
  check(!JSON.parse(rejected.stored.get('toolsAuth')).refreshToken, 'Rejected refresh credentials must not remain in the restored account.');

  let temporaryFailure = true;
  const retryable = createClient({ initial: expired(), fetchImpl: (url) => {
    if (url.endsWith('/oauth2/token')) return temporaryFailure ? response({}, 503) : response({ id_token: token() });
    return response(session());
  } });
  check(await retryable.auth.ensureFreshAuth() === null && JSON.parse(retryable.stored.get('toolsAuth')).refreshToken === 'test-refresh-token', 'A temporary refresh failure must preserve credentials for a later retry.');
  temporaryFailure = false;
  check((await retryable.auth.ensureFreshAuth()).idToken === token(), 'A later request must be able to retry a temporary refresh failure.');

  for (const stage of ['token', 'exchange', 'restore']) {
    const gate = deferred();
    const reached = deferred();
    const client = createClient({ initial: stage === 'restore' ? undefined : expired(), fetchImpl: (url) => {
      const waiting = stage === 'token' ? url.endsWith('/oauth2/token') : url === `/api/tools/auth/${stage}`;
      if (waiting) { reached.resolve(); return gate.promise; }
      if (url.endsWith('/oauth2/token')) return response({ id_token: token() });
      return response(session());
    } });
    // The restoration API is called 'session', while the test stage is 'restore'.
    if (stage === 'restore') client.context.fetch = async (url, options) => {
      client.calls.push({ url, options });
      if (url === '/api/tools/auth/session') { reached.resolve(); return gate.promise; }
      return response({ ok: true });
    };
    const pending = client.auth.ensureFreshAuth();
    await reached.promise;
    const waitingCall = client.calls.at(-1);
    const firstLogout = client.auth.signOut();
    const secondLogout = client.auth.signOut();
    gate.resolve(response(stage === 'token' ? { id_token: token() } : session()));
    check(await pending === null, `A late ${stage} response must not return a signed-in session after sign-out.`);
    check(!client.stored.has('toolsAuth') && waitingCall.options.signal.aborted, `Sign-out must abort ${stage} and prevent stale credential persistence even if fetch ignores cancellation.`);
    await Promise.all([firstLogout, secondLogout]);
    check(client.calls.filter((call) => call.url === '/api/tools/auth/logout').length === 1, 'Concurrent sign-outs must share a single logout request.');
    check(client.redirects.length === 1 && new URL(client.redirects[0]).pathname === '/logout', 'Concurrent sign-outs must redirect once to clear the Cognito browser session.');
    const countAfterLogout = client.calls.length;
    check(await client.auth.ensureFreshAuth() === null && client.calls.length === countAfterLogout, 'Auth change listeners must not restore a cookie while sign-out is in progress or complete.');
  }

  const logoutGate = deferred();
  const interruptedLogout = createClient({ initial: expired(), fetchImpl: () => logoutGate.promise });
  const loggingOut = interruptedLogout.auth.signOut();
  interruptedLogout.listeners.get('message')({
    origin: interruptedLogout.context.window.location.origin,
    data: { type: 'tools-auth:complete' }
  });
  check(await interruptedLogout.auth.ensureFreshAuth() === null && interruptedLogout.calls.length === 1, 'A late popup completion message must not reopen session restoration during logout.');
  interruptedLogout.stored.set('toolsAuth', JSON.stringify({ idToken: token('other-tab'), expiresAt: (now + 3600) * 1000 }));
  interruptedLogout.listeners.get('storage')({ key: 'toolsAuth', newValue: interruptedLogout.stored.get('toolsAuth') });
  check(await interruptedLogout.auth.ensureFreshAuth() === null && interruptedLogout.calls.length === 1, 'A credential storage event must not cancel the current tab logout or unblock its API requests.');
  logoutGate.resolve(response({ ok: true }));
  await loggingOut;
  check(interruptedLogout.redirects.length === 1 && new URL(interruptedLogout.redirects[0]).pathname === '/logout', 'Late auth notifications must not prevent the active hosted logout redirect.');

  const crossTabGate = deferred();
  const crossTab = createClient({ initial: expired(), mode: 'legacy', fetchImpl: () => crossTabGate.promise });
  const staleRenewal = crossTab.auth.ensureFreshAuth();
  const otherAccount = { idToken: token('other-account'), expiresAt: (now + 3600) * 1000 };
  crossTab.stored.set('toolsAuth', JSON.stringify(otherAccount));
  crossTab.listeners.get('storage')({ key: 'toolsAuth', newValue: JSON.stringify(otherAccount) });
  crossTabGate.resolve(response({ id_token: token('old-account') }));
  check(await staleRenewal === null && crossTab.auth.getAuth().idToken === otherAccount.idToken, 'A login in another tab must invalidate a pending renewal from the previous account.');

  const anonymousGate = deferred();
  const anonymous = createClient({ fetchImpl: () => anonymousGate.promise });
  const anonymousRequests = Array.from({ length: 4 }, () => anonymous.auth.ensureFreshAuth());
  anonymousGate.resolve(response({ authenticated: false }));
  check((await Promise.all(anonymousRequests)).every((auth) => auth === null) && anonymous.calls.length === 1 && !anonymous.stored.has('toolsAuth'), 'Concurrent anonymous restoration must stay anonymous without duplicate requests.');

  return checks;
}

module.exports = run;
if (require.main === module) run().then((checks) => console.log(`Tools auth renewal: ${checks} checks passed.`)).catch((error) => { console.error(error); process.exitCode = 1; });
