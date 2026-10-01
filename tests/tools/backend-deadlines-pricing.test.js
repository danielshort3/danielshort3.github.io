'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const proxy = require('../../api/_lib/demo-proxy');
const pricing = require('../../api/_lib/tools-endpoints/transcribe')._internal;

async function run() {
  assert.equal(pricing.calculateBillableSeconds(1), 1);
  assert.equal(pricing.calculateBillableSeconds(4.2), 5);
  assert.equal(pricing.calculateBillableSeconds(14.9), 15);
  assert.equal(pricing.calculateCostUsd(1, 0.0001), 0.0001);
  assert.equal(pricing.calculateCostUsd(4.2, 0.0001), 0.0005);
  const monitor = fs.readFileSync(require.resolve('../../js/tools/whisper-transcribe-monitor'), 'utf8');
  assert.match(monitor, /minDurationSeconds: 1\b/);
  assert.match(monitor, /minBillableSeconds: 1\b/);
  assert.doesNotMatch(monitor, /minBillableSeconds\) \|\| 15/);
  const page = fs.readFileSync(require.resolve('../../pages/transcribe.html'), 'utf8');
  assert.match(page, /id="transcribe-stat-minimum">1 sec<\/dd>/, 'Initial AWS minimum display should agree with the server and client config');
  const saved = { ...process.env };
  const nativeSetTimeout = global.setTimeout;
  const nativeClearTimeout = global.clearTimeout;
  const deadlines = [];
  const timers = new Set();
  let elapsed = 0;
  let invocationDuration = 70000;
  Object.assign(process.env, {
    DEMO_REQUIRE_DDB_RATE_LIMIT: 'false', DEMO_RATE_LIMIT_TABLE: '', DEMO_PROXY_MODE: 'iam',
    DEMO_SMART_SENTENCE_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:offline-sentence',
    DEMO_INVOKE_AWS_ROLE_ARN: '', AWS_AUTH_MODE: 'auto', VERCEL_ENV: 'development'
  });
  global.setTimeout = (callback, milliseconds) => {
    deadlines.push(milliseconds);
    const timer = { callback, due: elapsed + milliseconds };
    timers.add(timer);
    return timer;
  };
  global.clearTimeout = (timer) => timers.delete(timer);
  proxy._internal.setClientFactoryForTests(() => ({ lambda: { async send(command, options) {
    assert.equal(command.input.InvocationType, 'RequestResponse');
    elapsed += invocationDuration;
    for (const timer of timers) if (timer.due <= elapsed) timer.callback();
    if (options.abortSignal.aborted) {
      const error = new Error('Offline simulated deadline');
      error.name = 'AbortError';
      throw error;
    }
    return { Payload: Buffer.from(JSON.stringify({ statusCode: 200, body: '{"results":[]}' })) };
  } } }));
  try {
    const response = { setHeader() {}, end(body) { this.body = JSON.parse(body); } };
    await proxy.handleDemoRequest({ method: 'POST', url: '/api/demos/smart-sentence/rank', headers: { host: 'localhost', 'content-type': 'application/json' }, body: { query: 'Offline deadline test' } }, response, ['smart-sentence', 'rank']);
    assert.equal(response.statusCode, 200);
    assert.equal(deadlines[0], 125000, 'Proxy outlasts the live120sLambda limit while fitting Vercel150s');
    assert.equal(timers.size, 0, 'A response after the former 60-second cutoff releases its timer');
    invocationDuration = 126000;
    await proxy.handleDemoRequest({ method: 'POST', url: '/api/demos/smart-sentence/rank', headers: { host: 'localhost', 'content-type': 'application/json' }, body: { query: 'Offline bounded timeout' } }, response, ['smart-sentence', 'rank']);
    assert.equal(response.statusCode, 504, 'The extended deadline still bounds a stalled backend');
    assert.equal(timers.size, 0);
    assert.equal(proxy._internal.resolveDemoRoute(['smart-sentence', 'health'], 'GET', []).route.timeoutMs, 25000, 'Cheap health deadlines remain bounded');
    console.log('Backend deadlines and pricing passed: short clips use 1-second billing; Sentence Retriever accepts a simulated 70-second result and bounds a 126-second stall. No AWS calls.');
  } finally {
    global.setTimeout = nativeSetTimeout;
    global.clearTimeout = nativeClearTimeout;
    proxy._internal.setClientFactoryForTests(null);
    for (const name of Object.keys(process.env)) if (!(name in saved)) delete process.env[name];
    Object.assign(process.env, saved);
  }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
