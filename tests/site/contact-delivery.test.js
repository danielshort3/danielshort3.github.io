'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { createHarness } = require('./mobile-contact.test');
const { verifySignedRequest } = require('../../api/_lib/contact-protection');
const CONTACT_TEST_SECRET = 'offline-contact-test-secret-32-bytes-minimum';
const apiSource = fs.readFileSync(path.join(__dirname, '../../api/contact.js'), 'utf8');
const UNKNOWN = 'We couldn’t confirm delivery. Your message may have been sent. Your draft is still here.';
const settle = async () => { for (let index = 0; index < 12; index += 1) await Promise.resolve(); };
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };

async function apiCase(fetcher, {
  timeout = false,
  body = { name: 'Test', email: 'test@example.com', message: 'Never actually sent' },
  headers = { origin: 'https://www.danielshort.me', host: 'www.danielshort.me', 'x-forwarded-proto': 'https', 'x-forwarded-for': '203.0.113.5' },
  env = { CONTACT_PROXY_SECRET: CONTACT_TEST_SECRET }
} = {}) {
  const timers = new Map();
  const requests = [];
  const module = { exports: {} };
  vm.runInNewContext(apiSource, {
    module, process: { env }, Buffer, AbortController,
    require: (name) => name === './_lib/contact-protection' ? require('../../api/_lib/contact-protection') : require(name),
    fetch: (...args) => { requests.push(args); return fetcher(...args); },
    setTimeout: (callback, delay) => { timers.set(1, { callback, delay }); return 1; },
    clearTimeout: (id) => timers.delete(id)
  });
  const result = { headers: {} };
  const response = { setHeader: (name, value) => { result.headers[name] = value; }, end: (body) => { result.body = JSON.parse(body); result.status = response.statusCode; } };
  const pending = module.exports({ method: 'POST', body, headers }, response);
  await settle();
  if (timeout) {
    assert.equal(timers.get(1)?.delay, 20000);
    timers.get(1).callback();
  }
  await pending;
  assert.equal(result.headers['Cache-Control'], 'no-store');
  assert.equal(timers.size, 0, 'upstream deadline is always released');
  return { result, requests };
}

function browserHarness() {
  const h = createHarness();
  const scene = h.createScene();
  h.evaluate();
  const controller = h.window.initializeContactModal(scene.root);
  scene.fields.name.value = 'Test User';
  scene.fields.email.value = 'test@example.com';
  scene.fields.message.value = 'Never actually sent';
  scene.opener.dispatch('click');
  return { ...h, scene, controller };
}
const response = (status, data) => ({ ok: status >= 200 && status < 300, status, json: async () => data });

async function main() {
  const valid = await apiCase(async () => ({ ok: true, status: 200, text: async () => '{"ok":true}' }));
  assert.equal(valid.result.status, 200);
  assert.deepEqual(valid.result.body, { ok: true });
  const signed = verifySignedRequest({ headers: valid.requests[0][1].headers, body: valid.requests[0][1].body }, CONTACT_TEST_SECRET);
  assert(signed, 'the actual proxy signs the exact validated upstream body');
  assert.equal(JSON.parse(signed.body).email, 'test@example.com');
  assert(!JSON.stringify(valid.requests[0][1].headers).includes('203.0.113.5'), 'upstream headers contain a hashed actor instead of a raw client IP');
  const unreachable = async () => { throw new Error('rejected submissions must not reach the email sender'); };
  for (const origin of ['https://foreign.example', '']) {
    const denied = await apiCase(unreachable, { headers: { origin, host: 'www.danielshort.me', 'x-forwarded-proto': 'https' } });
    assert.equal(denied.result.status, 403);
    assert.equal(denied.requests.length, 0);
  }
  const malformedFields = await apiCase(unreachable, { body: { name: [], email: 'test@example.com', message: 'Invalid name type' } });
  assert.equal(malformedFields.result.status, 400);
  assert.equal(malformedFields.requests.length, 0);
  const unconfigured = await apiCase(unreachable, { env: {} });
  assert.equal(unconfigured.result.status, 503);
  assert.equal(unconfigured.requests.length, 0);
  const honeypot = await apiCase(unreachable, { body: { name: 'Test', email: 'test@example.com', message: 'Honeypot', company: 'spam' } });
  assert.equal(honeypot.result.status, 200);
  assert.equal(honeypot.requests.length, 0);
  const limited = await apiCase(async () => ({ ok: false, status: 429, headers: { get: () => '3599' }, text: async () => '{"error":"private quota details"}' }));
  assert.equal(limited.result.status, 429);
  assert.equal(limited.result.headers['Retry-After'], '3599');
  assert(!JSON.stringify(limited.result).includes('private'));
  const oversized = await apiCase(async () => { throw new Error('oversized bodies must not reach the upstream'); }, {
    body: { name: 'Test', email: 'test@example.com', message: 'x'.repeat(40 * 1024) }
  });
  assert.equal(oversized.result.status, 413);
  assert.equal(oversized.requests.length, 0);
  for (const body of ['', '{}', 'not json', '{"ok":false}', '{"ok":true,"error":"secret upstream detail"}']) {
    const malformed = await apiCase(async () => ({ ok: true, status: 200, text: async () => body }));
    assert.equal(malformed.result.status, 502);
    assert.equal(malformed.result.body.code, 'CONTACT_DELIVERY_UNKNOWN');
  }
  const rejected = await apiCase(async () => ({ ok: false, status: 400, text: async () => '{"error":"secret internal detail"}' }));
  assert.equal(rejected.result.status, 400);
  assert(!JSON.stringify(rejected.result).includes('secret'));
  for (const stalledBody of [false, true]) {
    const never = new Promise(() => {});
    const timed = await apiCase(async () => stalledBody ? { ok: true, text: () => never } : never, { timeout: true });
    assert.equal(timed.result.status, 504);
    assert.equal(timed.result.body.code, 'CONTACT_DELIVERY_UNKNOWN');
    assert.equal(timed.requests[0][1].signal.aborted, true);
  }
  const failure = await apiCase(async () => { throw new Error('sensitive network details'); });
  assert.equal(failure.result.body.error, UNKNOWN);

  // Pending submission remains one request across close/reopen and double-click.
  const pending = browserHarness();
  const first = deferred();
  let calls = 0;
  pending.window.fetch = () => { calls += 1; return first.promise; };
  pending.scene.form.dispatch('submit');
  pending.scene.close.dispatch('click');
  pending.records.get(pending.scene.modal).pending();
  pending.scene.opener.dispatch('click');
  pending.scene.form.dispatch('submit');
  assert.equal(calls, 1);
  assert(pending.controller.sending);
  assert.equal(pending.scene.form.getAttribute('aria-busy'), 'true');
  assert.equal(pending.scene.fields.message.readOnly, true);
  first.resolve(response(200, { ok: true }));
  await settle();
  assert(!pending.controller.sending);
  assert.equal(pending.scene.fields.message.value, '');
  assert.equal(pending.storedDrafts.size, 0, 'confirmed delivery clears stored contact draft');

  // Header and body stalls use the same 25-second deadline. Late responses
  // cannot clear a failed draft or overwrite the result of an explicit retry.
  for (const stalledBody of [false, true]) {
    const h = browserHarness();
    const late = deferred();
    const retry = deferred();
    let requestCount = 0;
    h.window.fetch = () => {
      requestCount += 1;
      if (requestCount > 1) return retry.promise;
      return stalledBody ? Promise.resolve({ ok: true, status: 200, json: () => late.promise }) : late.promise;
    };
    h.scene.form.dispatch('submit');
    await settle();
    const deadline = [...h.timers.values()].find((timer) => timer.delay === 25000);
    assert(deadline);
    deadline.callback();
    await settle();
    assert.equal(h.document.getElementById('contact-status').textContent, UNKNOWN);
    assert.equal(h.scene.form.getAttribute('aria-busy'), 'false');
    assert.equal(h.scene.fields.message.readOnly, false);
    assert(h.window.SiteContact.canLeave());
    assert.equal(h.storedDrafts.get('contact:personal').message, 'Never actually sent');
    assert.equal(requestCount, 1, 'failure never automatically resends');
    h.scene.form.dispatch('submit');
    late.resolve(stalledBody ? { ok: true } : response(200, { ok: true }));
    await settle();
    assert(h.controller.sending, 'stale response cannot finish the retry');
    assert.equal(h.scene.fields.message.value, 'Never actually sent');
    retry.resolve(response(200, { ok: true }));
    await settle();
    assert.equal(requestCount, 2);
    assert.equal(h.scene.fields.message.value, '');
  }
  for (const [status, data] of [[400, { error: 'Invalid' }], [200, {}], [200, null], [504, { code: 'CONTACT_DELIVERY_UNKNOWN' }]]) {
    const h = browserHarness();
    h.window.fetch = async () => response(status, data);
    h.scene.form.dispatch('submit');
    await settle();
    assert.equal(h.scene.fields.message.value, 'Never actually sent');
    assert.equal(h.storedDrafts.get('contact:personal').message, 'Never actually sent');
    assert(h.window.SiteContact.canLeave());
  }
  const draft = browserHarness();
  draft.scene.form.dispatch('input');
  assert([...draft.timers.values()].some((timer) => timer.delay === 500));
  draft.window.dispatch('pagehide');
  assert.equal(draft.storedDrafts.get('contact:personal').name, 'Test User');
  assert.deepEqual(Object.keys(draft.storedDrafts.get('contact:personal')), ['name', 'email', 'message']);
  draft.scene.fields.name.value = '';
  draft.scene.fields.email.value = '';
  draft.scene.fields.message.value = '';
  draft.window.dispatch('pagehide');
  assert.equal(draft.storedDrafts.size, 0);
  console.log('Contact delivery tests passed: bounded fetch/body, valid confirmation, draft retention, modal lifecycle, retry and late responses. No messages sent.');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
