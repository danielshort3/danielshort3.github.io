'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');

const source = fs.readFileSync(path.resolve(__dirname, '../../js/accounts/tools-auth.js'), 'utf8');
const turn = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
};
const response = (data = {}, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });

function createClient({ pathname = '/tools/dashboard', search = '?code=test-code&state=test-state', hash = '', returnTo, logoutReturnTo, popup = false, fetchImpl, digestImpl } = {}) {
  const values = new Map([
    ['toolsAuthState', 'test-state'], ['toolsAuthCodeVerifier', 'test-verifier']
  ]);
  const localValues = new Map();
  if (returnTo !== undefined) values.set('toolsAuthReturnTo', returnTo);
  if (logoutReturnTo !== undefined) values.set('toolsAuthLogoutReturnTo', logoutReturnTo);
  if (popup) localValues.set('toolsAuthPopupState:test-state', JSON.stringify({
    verifier: 'popup-verifier', createdAt: Date.now(), returnTo: '/tools/text-compare'
  }));
  const storage = entries => ({
    getItem: key => entries.get(key) || null,
    setItem: (key, value) => entries.set(key, String(value)),
    removeItem: key => entries.delete(key)
  });
  const redirects = [];
  const events = [];
  const calls = [];
  const openedPopups = [];
  const digests = [];
  const location = {
    origin: 'https://www.danielshort.me', pathname,
    search, hash,
    replace: value => redirects.push(value), assign: value => redirects.push(value)
  };
  const window = {
    location,
    TOOLS_AUTH_CONFIG: { sessionMode: 'legacy', cognitoDomain: 'auth.example.com', cognitoClientId: 'test-client' },
    addEventListener() {}, close() { events.push('close'); },
    open() {
      const child = {
        location: { href: '' }, document: { title: '', body: { innerHTML: '' } },
        closed: false, focus() {}, close() { this.closed = true; }
      };
      openedPopups.push(child);
      return child;
    },
    opener: popup ? { postMessage: value => events.push(value.type) } : null,
    history: { replaceState(state, title, value) {
      const url = new URL(value, location.origin);
      Object.assign(location, { pathname: url.pathname, search: url.search, hash: url.hash });
    } }
  };
  const token = `header.${Buffer.from(JSON.stringify({ sub: 'test-user', exp: Math.floor(Date.now() / 1000) + 3600 })).toString('base64url')}.signature`;
  vm.runInNewContext(source, {
    window, document: { body: { dataset: {} }, title: 'Account', dispatchEvent() {} },
    localStorage: storage(localValues), sessionStorage: storage(values), URL, URLSearchParams, atob, btoa,
    crypto: {
      getRandomValues: value => webcrypto.getRandomValues(value),
      subtle: { digest: (...args) => {
        digests.push(args);
        return digestImpl ? digestImpl(...args) : webcrypto.subtle.digest(...args);
      } }
    },
    TextEncoder, AbortController, console,
    fetch: async (url, options) => {
      calls.push({ url, options });
      return fetchImpl ? fetchImpl(url, options) : response({ id_token: token });
    }
  });
  return { auth: window.ToolsAuth, values, localValues, location, redirects, events, calls, openedPopups, digests, token };
}

async function run() {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };
  for (const destination of ['/tools/text-compare?session=saved-work#output', '/#games', '/games/stormbreak', '/portfolio/website']) {
    const client = createClient({ returnTo: destination });
    const result = await client.auth.handleRedirect();
    assert.equal(client.redirects[0], destination);
    assert.equal(result.redirected, true);
    assert.equal(client.values.has('toolsAuthReturnTo'), false);
    checks += 3;
  }
  for (const destination of [undefined, '', '/tools/dashboard', '/tools/dashboard.html?tab=sessions', '/pages/tools-dashboard.html', 'https://untrusted.example/collect', '//untrusted.example/collect']) {
    const client = createClient({ returnTo: destination });
    await client.auth.handleRedirect();
    assert.deepEqual(client.redirects, ['/#tools']);
    checks += 1;
  }
  const directTool = createClient({ pathname: '/tools/job-application-tracker' });
  assert.equal((await directTool.auth.handleRedirect()).redirected, false);
  assert.deepEqual(directTool.redirects, []);
  checks += 2;

  const popup = createClient({ popup: true });
  assert.equal((await popup.auth.handleRedirect()).popup, true);
  assert.deepEqual(popup.redirects, []);
  assert.deepEqual(popup.events, ['tools-auth:complete', 'close']);
  checks += 3;

  const signedInFromLegacyPage = createClient();
  signedInFromLegacyPage.location.search = '';
  await signedInFromLegacyPage.auth.signIn();
  assert.equal(signedInFromLegacyPage.values.get('toolsAuthReturnTo'), '/#tools');
  assert.equal(new URL(signedInFromLegacyPage.redirects[0]).searchParams.get('redirect_uri'), 'https://www.danielshort.me/tools/dashboard');
  checks += 2;

  const mismatched = createClient({ returnTo: '/tools/text-compare' });
  mismatched.location.search = '?code=test-code&state=wrong-state';
  await assert.rejects(mismatched.auth.handleRedirect(), /Auth state mismatch/);
  assert.deepEqual(mismatched.redirects, []);
  checks += 2;

  const popupStart = createClient({ pathname: '/tools/transcribe', search: '?source=upload', hash: '#result' });
  const popupSignIn = popupStart.auth.signIn({ mode: 'popup' });
  check(popupStart.openedPopups.length === 1, 'Normal popup sign-in must open its window synchronously during the user gesture.');
  check((await popupSignIn).mode === 'popup' && popupStart.redirects.length === 0, 'Popup sign-in must leave the original page in place.');
  const popupAuthorize = new URL(popupStart.openedPopups[0].location.href);
  const popupRecord = JSON.parse(popupStart.localValues.get(`toolsAuthPopupState:${popupAuthorize.searchParams.get('state')}`));
  check(popupAuthorize.pathname === '/oauth2/authorize' && popupAuthorize.searchParams.get('code_challenge_method') === 'S256' && Boolean(popupAuthorize.searchParams.get('code_challenge')), 'Normal popup sign-in must retain a PKCE authorization request.');
  check(popupRecord.verifier && popupRecord.returnTo === '/tools/transcribe?source=upload#result', 'Popup state must retain its verifier and exact tool destination.');

  for (const route of ['/tools/text-compare?session=saved-work&view=compact#output', '/games/stormbreak', '/tools/dashboard']) {
    const destination = new URL(route, 'https://www.danielshort.me');
    const gate = deferred();
    const client = createClient({
      pathname: destination.pathname, search: destination.search, hash: destination.hash,
      fetchImpl: () => gate.promise
    });
    client.localValues.set('toolsAuth', JSON.stringify({ idToken: client.token }));
    client.localValues.set('jobTrackerAuth', JSON.stringify({ idToken: client.token }));
    const signOut = client.auth.signOut();
    const sameSignOut = client.auth.signOut();
    const pendingSignIn = client.auth.signIn({ mode: 'popup', returnTo: '/tools/word-frequency' });
    const expectedReturn = route === '/tools/dashboard' ? '/#tools'
      : route.startsWith('/tools/text-compare') ? '/tools/text-compare?view=compact#output' : route;
    check(!client.localValues.has('toolsAuth') && !client.localValues.has('jobTrackerAuth'), 'Sign-out must synchronously clear current and legacy credentials.');
    check(client.values.get('toolsAuthLogoutReturnTo') === expectedReturn && !client.localValues.has('toolsAuthLogoutReturnTo'), 'Sign-out must store a tab-local return route without the prior account session, preserving other query parameters and the fragment.');
    await turn();
    check(client.calls.length === 1 && client.calls[0].url === '/api/tools/auth/logout' && client.calls[0].options.method === 'POST' && client.calls[0].options.credentials === 'same-origin', 'Repeated sign-out and immediate sign-in must share one same-origin server logout.');
    check(client.redirects.length === 0 && client.openedPopups.length === 0 && client.digests.length === 0, 'Pending logout must prevent provider navigation, popup creation, and new PKCE work.');
    check(await client.auth.ensureFreshAuth() === null && client.calls.length === 1, 'Pending logout must block cookie restoration.');
    gate.resolve(response({ ok: true }));
    await Promise.all([signOut, sameSignOut]);
    check((await pendingSignIn).mode === 'redirect', 'Sign-in during logout must finish the existing logout redirect instead of starting authorization.');
    check(client.redirects.length === 1, 'Concurrent logout callers must trigger only one Cognito redirect.');
    const logoutUrl = new URL(client.redirects[0]);
    check(logoutUrl.origin === 'https://auth.example.com' && logoutUrl.pathname === '/logout', 'Sign-out must clear the Cognito browser session through its HTTPS logout endpoint.');
    check(logoutUrl.searchParams.get('client_id') === 'test-client' && logoutUrl.searchParams.get('logout_uri') === 'https://www.danielshort.me/tools/dashboard', 'Cognito logout must use the configured client and registered callback as its sign-out destination.');
    check([...logoutUrl.searchParams.keys()].sort().join(',') === 'client_id,logout_uri', 'Logout must not forward authorization, PKCE, provider, or arbitrary return-route parameters.');
    check(!client.values.has('toolsAuthState') && !client.values.has('toolsAuthCodeVerifier') && client.calls.length === 1, 'Logout must not persist replacement PKCE state or exchange tokens.');
  }

  for (const destination of ['/tools/text-compare?session=saved-work#output', '/games/stormbreak', '/tools/dashboard', '/pages/tools-dashboard.html', 'https://untrusted.example/collect', '//untrusted.example/collect']) {
    const client = createClient({ search: '', logoutReturnTo: destination });
    client.localValues.set('toolsAuth', JSON.stringify({ idToken: client.token }));
    const result = await client.auth.handleRedirect();
    const expectedReturn = destination.startsWith('/tools/text-compare') || destination === '/games/stormbreak' ? destination : '/#tools';
    check(result.handled === true && result.redirected === true && client.redirects[0] === expectedReturn, 'No-code logout callbacks must return to a safe original route, with legacy dashboard and external targets falling back to Tools.');
    check(!client.values.has('toolsAuthLogoutReturnTo') && client.auth.getAuth() === null, 'Logout callbacks must consume the marker and remain signed out.');
    check(await client.auth.ensureFreshAuth() === null && client.calls.length === 0, 'Logout callbacks must not restore cookies, authorize, or exchange tokens.');
  }
  const ordinaryVisit = createClient({ search: '' });
  check((await ordinaryVisit.auth.handleRedirect()).handled === false && ordinaryVisit.redirects.length === 0, 'An ordinary no-code visit without a logout marker must not act as a logout callback.');

  for (const retryWith of ['signOut', 'signIn']) {
    let attempts = 0;
    const client = createClient({ pathname: '/games/stormbreak', search: '', fetchImpl: async () => {
      attempts += 1;
      if (attempts === 1) {
        if (retryWith === 'signOut') return response({}, 503);
        throw new Error('Network unavailable');
      }
      return response({ ok: true });
    } });
    await assert.rejects(client.auth.signOut());
    checks += 1;
    check(client.redirects.length === 0 && client.values.get('toolsAuthLogoutReturnTo') === '/games/stormbreak', 'Failed server logout must preserve its return marker and avoid provider navigation.');
    check(await client.auth.ensureFreshAuth() === null && attempts === 1, 'Failed logout must stay signed out locally and prevent cookie restoration.');
    if (retryWith === 'signOut') await client.auth.signOut();
    else check((await client.auth.signIn({ mode: 'popup' })).mode === 'redirect', 'Sign-in after failed logout must retry logout before attempting authorization.');
    check(attempts === 2 && client.redirects.length === 1 && new URL(client.redirects[0]).pathname === '/logout', 'A later explicit action must retry failed logout and complete the Cognito redirect.');
    check(client.openedPopups.length === 0 && client.digests.length === 0, 'Retrying a failed logout must not create a new sign-in popup or PKCE state.');
  }

  for (const mode of ['redirect', 'popup']) {
    const digestGate = deferred();
    const digestStarted = deferred();
    const client = createClient({ pathname: '/tools/text-compare', search: '', digestImpl: async (...args) => {
      digestStarted.resolve();
      await digestGate.promise;
      return webcrypto.subtle.digest(...args);
    } });
    const oldSignIn = client.auth.signIn({ mode });
    const canceled = assert.rejects(oldSignIn, error => error.name === 'AbortError');
    await digestStarted.promise;
    await client.auth.signOut();
    digestGate.resolve();
    await canceled;
    checks += 1;
    check(client.redirects.length === 1 && new URL(client.redirects[0]).pathname === '/logout', 'Sign-out during asynchronous PKCE generation must cancel the older authorization navigation.');
    check(!client.values.has('toolsAuthState') && !client.values.has('toolsAuthCodeVerifier') && ![...client.localValues.keys()].some(key => key.startsWith('toolsAuthPopupState:')), 'Canceled PKCE work must not recreate redirect or popup state after sign-out clears it.');
    if (mode === 'popup') check(client.openedPopups[0].closed && !client.openedPopups[0].location.href, 'Canceled popup sign-in must close its waiting window without navigating it.');
  }
  return checks;
}

module.exports = run;
if (require.main === module) run().then(checks => console.log(`Tools auth return: ${checks} checks passed.`)).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
