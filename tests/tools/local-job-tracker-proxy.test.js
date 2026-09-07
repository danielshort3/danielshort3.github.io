'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const path = require('path');
const vm = require('vm');
const { once } = require('events');
const { handleLocalJobTrackerRequest, MAX_BODY_BYTES, MAX_RESPONSE_BYTES } = require('../../build/lib/local-job-tracker-proxy');

const UPSTREAM = 'https://fhp2is6v8h.execute-api.us-east-2.amazonaws.com/prod';
const PREFIX = '/api/job-tracker';
const AUTHORIZATION = 'Bearer test.id.signature';

async function withServer(handler, run) {
  const server = http.createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try { await run(`http://127.0.0.1:${server.address().port}`); } finally {
    server.closeAllConnections();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

function rawRequest(origin, target, options = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(origin);
    const request = http.request({
      host: url.hostname, port: url.port, path: target,
      method: options.method || 'GET', headers: options.headers || {}
    }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve({ status: response.statusCode, headers: response.headers, text: Buffer.concat(chunks).toString('utf8') }));
    });
    request.on('error', reject);
    request.end(options.body);
  });
}

async function testProxy() {
  const calls = [];
  let respond = () => new Response('{"items":[],"nextCursor":""}', { headers: { 'content-type': 'application/json' } });
  const fetchImpl = async (url, options) => { calls.push({ url, options }); return respond(url, options); };
  await withServer((req, res) => {
    handleLocalJobTrackerRequest(req, res, { fetchImpl, timeoutMs: 100 }).catch(() => { res.statusCode = 500; res.end('Unexpected test failure'); });
  }, async (origin) => {
    const request = (target, options = {}) => rawRequest(origin, PREFIX + target, {
      ...options, headers: { authorization: AUTHORIZATION, ...options.headers }
    });
    const json = (method = 'POST', body = '{ "company": "Example", "expectedVersion": 3 }') => ({
      method, headers: { 'content-type': 'application/json; charset=utf-8' }, body
    });
    const allowed = [
      ['/api/applications?limit=500&cursor=abc_-123', {}],
      ['/api/prospects?start=2026-01-01&end=2026-12-31', {}],
      ['/api/views', {}],
      ...['dashboard', 'summary', 'applications-over-time', 'status-breakdown', 'calendar', 'funnel', 'time-in-stage'].map((name) => [`/api/analytics/${name}?start=2026-01-01&end=2026-12-31`, {}]),
      ['/api/analytics/followups?includeOverdue=true', {}],
      ...['applications', 'applications/capture', 'prospects', 'views', 'exports', 'attachments/presign', 'attachments/download', 'attachments/zip'].map((name) => [`/api/${name}`, json()]),
      ['/api/applications/APP%231234%23test-id', json('PATCH')],
      ['/api/prospects/PROSPECT%231234%23test-id', json('PATCH')],
      ...['applications/APP%231234%23test-id', 'prospects/PROSPECT%231234%23test-id', 'views/VIEW%231234%23test-id'].map((name) => [`/api/${name}`, { method: 'DELETE' }])
    ];
    for (const [target, options] of allowed) {
      const result = await request(target, options);
      assert.strictEqual(result.status, 200, target);
      const call = calls.at(-1);
      assert.strictEqual(call.url, UPSTREAM + target, 'Requests must use the fixed tracker upstream with unchanged path/query.');
      assert.strictEqual(call.options.method, options.method || 'GET');
      assert.strictEqual(new Headers(call.options.headers).get('authorization'), AUTHORIZATION, 'The original bearer token must reach API Gateway.');
      assert.strictEqual(call.options.redirect, 'error', 'Redirects cannot escape the fixed upstream.');
      if (options.body) assert.strictEqual(Buffer.from(call.options.body).toString(), options.body, 'JSON request bytes must remain unchanged.');
      assert.strictEqual(result.headers['cache-control'], 'no-store');
    }

    await request('/api/applications', {
      ...json(), headers: { ...json().headers, origin, cookie: 'private-cookie', 'x-api-key': 'private-key', 'x-forwarded-host': 'evil.example', 'x-private': 'secret' }
    });
    const forwarded = new Headers(calls.at(-1).options.headers);
    assert.strictEqual(forwarded.get('content-type'), 'application/json; charset=utf-8');
    for (const header of ['origin', 'cookie', 'x-api-key', 'x-forwarded-host', 'x-private']) {
      assert.strictEqual(forwarded.get(header), null, `${header} must never be sent to the tracker.`);
    }

    for (const [target, options, status] of [
      ['/api/unknown', {}, 404],
      ['/api/analytics/unknown', {}, 404],
      ['/api/applications/APP%2Fescape', json('PATCH'), 404],
      ['/api/applications/APP%2523double', json('PATCH'), 404],
      ['/api/applications/APP%3Fquery', json('PATCH'), 404],
      ['/api/applications/a/extra', json('PATCH'), 404],
      ['/api/applications/../prospects', {}, 400],
      ['/api/applications/%2e%2e/prospects', {}, 400],
      ['/api/views', { method: 'PUT' }, 405],
      ['/api/analytics/dashboard', json(), 405],
      ['/api/applications', { method: 'OPTIONS' }, 405],
      ['/api/applications?url=https://evil.example', {}, 400],
      ['/api/applications?limit=2&limit=3', {}, 400],
      ['/api/applications?cursor=' + 'a'.repeat(2049), {}, 400],
      ['/api/views?start=2026-01-01', {}, 400],
      ['/api/applications?limit=2', json(), 400],
      ['/api/analytics/followups?includeOverdue=maybe', {}, 400],
      ['/api/applications', { headers: { authorization: '' } }, 401],
      ['/api/applications', { headers: { authorization: 'Basic wrong' } }, 401],
      ['/api/applications', { headers: { cookie: 'tools-session-cookie', authorization: '' } }, 401],
      ['/api/applications', { headers: { origin: 'https://evil.example' } }, 403],
      ['/api/applications', { headers: { origin: origin.replace('127.0.0.1', 'localhost') } }, 403],
      ['/api/applications', { headers: { 'sec-fetch-site': 'cross-site' } }, 403],
      ['/api/applications', { headers: { 'sec-fetch-site': 'same-site' } }, 403],
      ['/api/applications', { headers: { host: 'evil.example' } }, 400],
      ['/api/applications', { headers: { host: '127.0.0.1.evil.example' } }, 400],
      ['/api/applications', { method: 'POST', headers: { 'content-type': 'text/plain' }, body: '{}' }, 415],
      ['/api/applications', { ...json(), headers: { ...json().headers, 'content-encoding': 'gzip' } }, 415],
      ['/api/applications', json('POST', '{broken'), 400],
      ['/api/applications', json('POST', '[]'), 400],
      ['/api/applications', json('POST', 'null'), 400],
      ['/api/applications', json('POST', JSON.stringify({ value: 'a'.repeat(MAX_BODY_BYTES) })), 413],
      ['/api/views/id', { method: 'DELETE', headers: { 'content-length': '2' }, body: '{}' }, 400]
    ]) {
      const before = calls.length;
      const result = await request(target, options);
      assert.strictEqual(result.status, status, target + ' ' + JSON.stringify(options.headers || {}));
      assert.strictEqual(calls.length, before, 'Rejected requests must not reach AWS.');
    }

    for (const status of [401, 403, 409, 429]) {
      respond = () => new Response(JSON.stringify({ error: 'Upstream validation', version: 4 }), {
        status, headers: { 'content-type': 'application/json', 'retry-after': '5', 'set-cookie': 'never-forward', 'access-control-allow-origin': '*' }
      });
      const response = await request('/api/applications');
      assert.strictEqual(response.status, status, 'Real authentication, conflict, and throttle status must survive the proxy.');
      assert.deepStrictEqual(JSON.parse(response.text), { error: 'Upstream validation', version: 4 });
      assert.strictEqual(response.headers['retry-after'], '5');
      assert.strictEqual(response.headers['set-cookie'], undefined);
      assert.strictEqual(response.headers['access-control-allow-origin'], undefined);
    }
    respond = () => new Response(null, { status: 204 });
    assert.strictEqual((await request('/api/views/id', { method: 'DELETE' })).status, 204);

    for (const failure of [
      () => { throw new Error('private-secret diagnostics'); },
      () => new Response('{"error":"private-secret"}', { status: 502, headers: { 'content-type': 'application/json' } }),
      () => new Response('<html>private-secret</html>', { headers: { 'content-type': 'text/html' } }),
      () => new Response('private-secret', { status: 302, headers: { location: 'https://evil.example' } }),
      () => new Response('{bad JSON', { headers: { 'content-type': 'application/json' } }),
      () => new Response('{}', { headers: { 'content-type': 'application/json', 'content-length': String(MAX_RESPONSE_BYTES + 1) } }),
      () => new Response(JSON.stringify({ value: 'a'.repeat(MAX_RESPONSE_BYTES) }), { headers: { 'content-type': 'application/json' } })
    ]) {
      respond = failure;
      const response = await request('/api/applications');
      assert.strictEqual(response.status, 502);
      assert(!response.text.includes('private-secret'));
      assert(response.text.length < 1000);
    }
    respond = (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('Request aborted')), { once: true });
    });
    assert.strictEqual((await request('/api/applications')).status, 504, 'Upstream timeouts must abort the pending fetch.');
  });
}

function testFrontendSelection() {
  const source = fs.readFileSync(path.join(__dirname, '../../js/tools/job-application-tracker.js'), 'utf8');
  const start = source.indexOf('  const resolveTrackerApiBase =');
  const end = source.indexOf('  const config =', start);
  assert(start >= 0 && end > start);
  for (const [hostname, protocol, expected] of [
    ['localhost', 'http:', PREFIX], ['127.0.0.1', 'http:', PREFIX], ['[::1]', 'http:', PREFIX],
    ['www.danielshort.me', 'https:', UPSTREAM], ['localhost.evil.example', 'https:', UPSTREAM], ['localhost', 'file:', UPSTREAM]
  ]) {
    const context = { window: { location: { hostname, protocol } } };
    vm.runInNewContext(source.slice(start, end) + '\nthis.resolve = resolveTrackerApiBase;', context);
    assert.strictEqual(context.resolve(UPSTREAM), expected);
  }
}

async function run() {
  await testProxy();
  testFrontendSelection();
  console.log('Local job tracker proxy: route/method and query limits, bearer forwarding, origin checks, bounded JSON, upstream errors/timeouts, and local-only frontend selection passed.');
}

module.exports = run;
if (require.main === module) run().catch((error) => { console.error(error); process.exitCode = 1; });
