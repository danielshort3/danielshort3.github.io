'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const proxy = require('../../api/_lib/demo-proxy');

function client(response) {
  const timers = new Set();
  const requests = [];
  const env = {
    window: {}, URLSearchParams, AbortController,
    setTimeout(callback, milliseconds) { const timer = { callback, milliseconds }; timers.add(timer); return timer; },
    clearTimeout(timer) { timers.delete(timer); },
    fetch: async (url, options) => { requests.push({ url, options }); return response; }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve('../../js/demos/aws-client'), 'utf8'), env);
  return { api: env.window.DemoAws, timers, requests };
}

async function browserDeadlines() {
  // A response may arrive while its body never finishes; the whole operation is bounded.
  const stalled = client({ ok: true, text: () => new Promise(() => {}) });
  const request = stalled.api.postJson('/rank', { query: 'offline' }, { timeoutMs: 130000 });
  await new Promise(setImmediate);
  assert.equal(stalled.requests.length, 1);
  assert.equal([...stalled.timers][0].milliseconds, 130000);
  [...stalled.timers][0].callback();
  await assert.rejects(request, { code: 'DEMO_REQUEST_TIMEOUT' });
  assert(stalled.requests[0].options.signal.aborted);
  assert.equal(stalled.timers.size, 0);

  const cancelled = client({ ok: true, text: () => new Promise(() => {}) });
  const controller = new AbortController();
  const pending = cancelled.api.postJson('/rank', { query: 'offline' }, { timeoutMs: 130000, signal: controller.signal });
  await new Promise(setImmediate);
  controller.abort();
  await assert.rejects(pending, { code: 'DEMO_REQUEST_CANCELLED' });
  assert.equal(cancelled.timers.size, 0);
  assert(cancelled.requests[0].options.signal.aborted);

  const alreadyCancelled = new AbortController();
  alreadyCancelled.abort();
  await assert.rejects(cancelled.api.postJson('/rank', {}, { signal: alreadyCancelled.signal }), { code: 'DEMO_REQUEST_CANCELLED' });
  assert.equal(cancelled.requests.length, 1, 'An already cancelled operation never sends');

  const terminal = client({ ok: false, status: 502, statusText: '', headers: { get() {} }, text: async () => '{"error":"Unavailable","code":"DEMO_FUNCTION_ERROR"}' });
  await assert.rejects(terminal.api.retryRequest(() => terminal.api.postJson('/rank', {}), { retries: 2, baseDelayMs: 0 }), { code: 'DEMO_FUNCTION_ERROR' });
  assert.equal(terminal.requests.length, 1, 'Known exhausted function/runtime deadlines do not repeat compute');

  const health = client({ ok: true, text: () => new Promise(() => {}) });
  const connection = health.api.retryRequest((attempt, { signal }) => health.api.getJson('/health', { signal }), { retries: 2, timeoutMs: 20000 });
  await new Promise(setImmediate);
  [...health.timers][0].callback();
  await assert.rejects(connection, { code: 'DEMO_REQUEST_TIMEOUT' });
  assert.equal(health.requests.length, 1, 'A stalled first health attempt cannot exceed the overall connection budget');
  assert.equal(health.timers.size, 0);
}

async function inputAndCorrelation() {
  const saved = { ...process.env };
  let calls = 0;
  let clientCreations = 0;
  let event;
  let functionError = false;
  Object.assign(process.env, {
    DEMO_REQUIRE_DDB_RATE_LIMIT: 'false', DEMO_RATE_LIMIT_TABLE: '', DEMO_PROXY_MODE: 'iam',
    DEMO_SMART_SENTENCE_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:offline-sentence',
    DEMO_INVOKE_AWS_ROLE_ARN: '', AWS_AUTH_MODE: 'auto', VERCEL_ENV: 'development'
  });
  proxy._internal.setClientFactoryForTests(() => {
    clientCreations += 1;
    return { lambda: { async send(command) {
      calls += 1;
      event = JSON.parse(command.input.Payload.toString());
      return functionError ? { FunctionError: 'Unhandled', Payload: Buffer.from('{"private":"must never leak"}') }
        : { Payload: Buffer.from('{"statusCode":200,"body":"{\\"top\\":[]}"}') };
    } } };
  });
  const invoke = async (body, requestId = 'offline-request-123') => {
    const result = { headers: {}, setHeader(key, value) { this.headers[key] = value; }, end(text) { this.body = JSON.parse(text); } };
    await proxy.handleDemoRequest({ method: 'POST', url: '/api/demos/smart-sentence/rank', headers: { host: 'localhost', 'content-type': 'application/json', 'x-request-id': requestId }, body }, result, ['smart-sentence', 'rank']);
    return result;
  };
  try {
    for (const body of [{}, { query: ' ' }, { query: 'a'.repeat(513) }, { query: '😀'.repeat(513) }, { query: 7 }, { query: 'a', top: 0 }, { query: 'a', top: 21 }, { query: 'a', top: true }, { query: 'a', top: '5' }, { query: 'a', extra: true }]) {
      assert.equal((await invoke(body)).statusCode, 400);
    }
    assert.equal(clientCreations, 0, 'Invalid requests are rejected before AWS client setup');
    assert.equal(calls, 0, 'Invalid requests are rejected before paid invocation');
    const boundary = 'a'.repeat(512);
    assert.equal((await invoke({ query: boundary })).statusCode, 200);
    assert.equal(calls, 1, 'The exact 512-character boundary makes one invocation');
    assert.deepEqual(JSON.parse(event.body), { query: boundary, top: 5 });
    const accepted = await invoke({ query: ' a ', top: 20 });
    assert.equal(accepted.statusCode, 200);
    assert.deepEqual(JSON.parse(event.body), { query: 'a', top: 20 });
    assert.equal(event.headers['x-request-id'], 'offline-request-123');
    assert.equal(accepted.headers['X-Request-Id'], 'offline-request-123');
    assert.equal((await invoke({ query: '😀'.repeat(512) })).statusCode, 200, 'Proxy character limits count Unicode code points');
    assert.equal(JSON.parse(event.body).top, 5);
    const sanitized = await invoke({ query: 'a' }, 'unsafe\nidentifier');
    assert.match(sanitized.headers['X-Request-Id'], /^[a-f0-9-]{36}$/);
    functionError = true;
    const failed = await invoke({ query: 'a' });
    assert.equal(failed.statusCode, 502);
    assert.equal(failed.body.code, 'DEMO_FUNCTION_ERROR');
    assert(!JSON.stringify(failed.body).includes('private'));
  } finally {
    proxy._internal.setClientFactoryForTests(null);
    for (const key of Object.keys(process.env)) if (!(key in saved)) delete process.env[key];
    Object.assign(process.env, saved);
  }
}

(async () => {
  const source = fs.readFileSync(require.resolve('../../demos/sentence-demo.html'), 'utf8');
  assert.match(source, /const QUERY_MAX_CHARS = 512;/);
  assert.match(source, /id="query"[^>]*maxlength="512"/);
  assert.match(source, /Use up to 512 characters\. Ideas longer than 128 model tokens/);
  assert.match(source, /Use 512 characters or fewer\./);
  assert.match(source, /const SEARCH_TIMEOUT_MS = 130000;/);
  await browserDeadlines();
  await inputAndCorrelation();
  console.log('Sentence reliability passed: full-body deadlines, cancellation, bounded health, terminal retry suppression, pre-invoke input limits and safe correlation. No AWS calls.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
