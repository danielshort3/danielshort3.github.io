'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Module = require('module');
const { once } = require('events');
const { createLocalServer } = require('../../build/dev');
const proxy = require('../../api/_lib/demo-proxy');

let checks = 0;
const check = (value, message) => { assert(value, message); checks += 1; };
const unavailable = { ok: false, error: 'Demo proxy is unavailable.', code: 'DEMO_PROXY_CONFIGURATION_UNAVAILABLE' };
const fastRetry = { retries: 12, baseDelayMs: 0, maxDelayMs: 0 };

function createClient(respond) {
  const requests = [];
  const env = {
    window: { location: { search: '' } },
    localStorage: { getItem() { return null; }, setItem() {} },
    URLSearchParams,
    setTimeout: (callback) => queueMicrotask(callback),
    fetch: async (url, options) => {
      requests.push({ url, options });
      const response = respond(url, options, requests.length);
      return {
        ok: response.status < 400,
        status: response.status,
        statusText: '',
        headers: { get() { return null; } },
        text: async () => JSON.stringify(response.body)
      };
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/demos/aws-client.js'), 'utf8'), env);
  return { api: env.window.DemoAws, requests };
}

async function testConfigurationRetries() {
  let configured = false;
  const { api, requests } = createClient((url) => ({
    status: configured || url.includes('/other/') ? 200 : 503,
    body: configured || url.includes('/other/') ? { ok: true } : unavailable
  }));
  await assert.rejects(api.retryRequest(() => api.healthJson('/api/demos/shape/'), fastRetry), { code: unavailable.code });
  check(requests.length === 1, 'definitive configuration failures should make only one health request');
  await assert.rejects(api.warmupJson('/api/demos/shape/', {}), { code: unavailable.code });
  await assert.rejects(api.retryRequest(() => api.postWithFallback('/api/demos/shape/', ['predict'], {}), fastRetry), { code: unavailable.code });
  check(requests.length === 1, 'configuration failures should block warmup and inference fallback requests');
  await api.healthJson('/api/demos/other/');
  check(requests.length === 2, 'a failed demo configuration must not disable other demos');
  configured = true;
  await api.healthJson('/api/demos/shape/');
  await api.postWithFallback('/api/demos/shape/', ['predict'], {});
  check(requests.length === 4, 'a fresh health check should allow reconnect and inference after configuration is repaired');
  configured = false;
  await assert.rejects(api.healthJson('/api/demos/shape/'));
  await assert.rejects(api.postWithFallback('/api/demos/shape/', ['predict'], {}));
  configured = true;
  await api.postWithFallback('/api/demos/shape/', ['predict'], {});
  check(requests.length === 6, 'suppression should end after the initialization fallback so a later user retry can recover');

  const transient = createClient((url, options, attempt) => ({
    status: attempt < 3 ? 503 : 200,
    body: attempt < 3 ? { error: 'Cold start in progress.' } : { ok: true }
  }));
  await transient.api.retryRequest(() => transient.api.healthJson('/api/demos/shape/'), fastRetry);
  check(transient.requests.length === 3, 'transient cold-start failures should continue to retry');
  check(transient.api.isRetryableError({ status: 503, code: 'UNKNOWN_PROVIDER_FAILURE' }), 'unknown provider failures should remain retryable');
  check(transient.api.isRetryableError({ message: 'Failed to fetch' }), 'transient network failures should remain retryable');
  check(!transient.api.isRetryableError({ status: 429 }), 'deliberate rate limits should remain nonretryable');

  const warmupFailure = createClient((url) => ({
    status: url.endsWith('/health') ? 200 : 503,
    body: url.endsWith('/health') ? { ok: true } : unavailable
  }));
  await warmupFailure.api.healthJson('/api/demos/shape/');
  await assert.rejects(warmupFailure.api.retryRequest(() => warmupFailure.api.warmupJson('/api/demos/shape/'), fastRetry));
  await assert.rejects(warmupFailure.api.postWithFallback('/api/demos/shape/', ['predict'], {}));
  check(warmupFailure.requests.length === 2, 'configuration failure discovered at warmup should also prevent inference fallback');
}

function captureResponse() {
  return {
    statusCode: 200,
    setHeader() {},
    end(body) { this.body = JSON.parse(body); }
  };
}

async function testProxyConfigurationCodes() {
  const values = {
    DEMO_PROXY_MODE: 'off', DEMO_SHAPE_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:demo:live',
    DEMO_REQUIRE_DDB_RATE_LIMIT: 'false', AWS_AUTH_MODE: 'auto', VERCEL_ENV: 'development',
    DEMO_INVOKE_AWS_ROLE_ARN: '', DEMO_AWS_ACCESS_KEY_ID: '', DEMO_AWS_SECRET_ACCESS_KEY: '',
    AWS_ACCESS_KEY_ID: '', AWS_SECRET_ACCESS_KEY: ''
  };
  const previous = Object.fromEntries(Object.keys(values).map(key => [key, process.env[key]]));
  Object.assign(process.env, values);
  const req = { method: 'GET', url: '/api/demos/shape/health', headers: { host: 'localhost' } };
  try {
    const missing = captureResponse();
    await proxy.handleDemoRequest(req, missing, ['shape', 'health']);
    check(missing.statusCode === 503 && missing.body.code === unavailable.code,
      'known configuration failures should expose a stable nonsecret code');
    check(JSON.stringify(missing.body) === JSON.stringify(unavailable), 'configuration response should not expose settings or credential details');
    process.env.DEMO_PROXY_MODE = 'iam';
    proxy._internal.setClientFactoryForTests(() => { throw Object.assign(new Error('private provider details'), { code: 'UNKNOWN_PROVIDER_FAILURE' }); });
    const unknown = captureResponse();
    await proxy.handleDemoRequest(req, unknown, ['shape', 'health']);
    check(unknown.statusCode === 503 && !unknown.body.code && !JSON.stringify(unknown.body).includes('private'),
      'unknown client failures should stay retryable and retain sanitized diagnostics');
  } finally {
    proxy._internal.setClientFactoryForTests(null);
    for (const [key, value] of Object.entries(previous)) {
      if (typeof value === 'undefined') delete process.env[key];
      else process.env[key] = value;
    }
  }
}

async function testShapeExportRecovery() {
  const html = fs.readFileSync(path.join(__dirname, '../../demos/shape-demo.html'), 'utf8');
  const source = html.slice(html.indexOf('async function classify()'), html.indexOf("classifyBtn.addEventListener('click', classify);"));
  for (const exportImage of [
    (callback) => callback(null),
    () => { throw new Error('Canvas export failed'); },
    (callback) => callback({ arrayBuffer: async () => { throw new Error('Image buffer failed'); } })
  ]) {
    const predictions = [];
    let restored = false;
    let requested = false;
    const env = {
      serverReady: true, hasDrawn: true, classifying: false,
      canvas: { toBlob: exportImage },
      setStep() {}, setPredictionUI: (message) => predictions.push(message),
      renderShapeScores() {}, updateClassifyState() {},
      resultBadge: { classList: { add() {}, remove() { restored = true; } } },
      postToEndpoint() { requested = true; }
    };
    vm.runInNewContext(source, env);
    await env.classify();
    check(!env.classifying && restored && !requested && predictions.at(-1).startsWith('Error:'),
      'canvas export failures should be handled and restore classification controls without requesting inference');
  }
}

async function testLocalStreamRoute() {
  const previousArn = process.env.CHATBOT_STREAM_FUNCTION_ARN;
  delete process.env.CHATBOT_STREAM_FUNCTION_ARN;
  const server = createLocalServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const url = 'http://127.0.0.1:' + server.address().port + '/api/chatbot-stream';
  const options = { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: new URL(url).origin, 'Sec-Fetch-Site': 'same-origin' }, body: JSON.stringify({ prompt: 'Plan a trip.' }) };
  const originalLoad = Module._load;
  const originalError = console.error;
  try {
    const unavailableResponse = await fetch(url, options);
    check(unavailableResponse.status === 503 && unavailableResponse.headers.get('content-type').includes('application/json'),
      'unconfigured local streaming should return structured unavailable status instead of a static HTML 404');
    check((await unavailableResponse.json()).error === 'Chatbot stream is not configured.', 'local streaming should preserve the real handler configuration checks');
    const methodResponse = await fetch(url, { headers: options.headers });
    check(methodResponse.status === 405 && methodResponse.headers.get('allow') === 'POST', 'local streaming should delegate method validation to the real handler');
    await methodResponse.text();

    const streamPath = path.join(__dirname, '../../api/chatbot-stream.js');
    let delegatedMethod = '';
    let delegatedBody = '';
    Module._load = function(request, parent, isMain) {
      if (request === streamPath) {
        return async (req, res) => {
          delegatedMethod = req.method;
          for await (const chunk of req) delegatedBody += chunk;
          res.setHeader('Content-Type', 'application/x-ndjson');
          res.write(JSON.stringify({ type: 'token', text: 'Hello' }) + '\n');
          res.end(JSON.stringify({ type: 'done' }) + '\n');
        };
      }
      return originalLoad.call(this, request, parent, isMain);
    };
    const streamed = await fetch(url, options);
    const streamedText = await streamed.text();
    check(streamed.status === 200 && streamed.headers.get('content-type') === 'application/x-ndjson' &&
      streamedText.includes('"type":"token"') && streamedText.includes('"type":"done"') &&
      delegatedMethod === 'POST' && JSON.parse(delegatedBody).prompt === 'Plan a trip.',
      'local route should delegate incoming requests and preserve streamed response chunks');
    let errorsLogged = 0;
    console.error = () => { errorsLogged += 1; };
    Module._load = function(request, parent, isMain) {
      if (request === streamPath) return async () => { throw new Error('private handler failure'); };
      return originalLoad.call(this, request, parent, isMain);
    };
    const rejected = await fetch(url, options);
    const rejection = await rejected.json();
    check(rejected.status === 500 && rejection.error === 'Local chatbot stream handler failed.' && errorsLogged === 1,
      'local route should handle unexpected promise rejections, log them, and return safe JSON');
  } finally {
    Module._load = originalLoad;
    console.error = originalError;
    if (typeof previousArn === 'undefined') delete process.env.CHATBOT_STREAM_FUNCTION_ARN;
    else process.env.CHATBOT_STREAM_FUNCTION_ARN = previousArn;
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}

(async () => {
  await testConfigurationRetries();
  await testProxyConfigurationCodes();
  await testShapeExportRecovery();
  await testLocalStreamRoute();
  console.log('demo-runtime-resilience: ' + checks + ' checks passed');
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
