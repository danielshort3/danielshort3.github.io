'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { once } = require('events');
const { createLocalServer } = require('../../build/dev');

const nativeFetch = global.fetch;
const root = path.resolve(__dirname, '../..');
const handlerFiles = new Map([
  ['index', 'api/short-links/index.js'],
  ['router', 'api/short-links/[...slug].js'],
  ['redirect', 'api/go/[...slug].js']
].map(([name, filename]) => [path.join(root, filename), name]));

async function withLocalServer(run) {
  const tempRoot = path.resolve(os.tmpdir());
  const envDir = fs.mkdtempSync(path.join(tempRoot, 'website-short-links-api-test-'));
  let server;
  try {
    server = createLocalServer({ envDir });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    await run(`http://127.0.0.1:${server.address().port}`);
  } finally {
    if (server?.listening) {
      server.closeAllConnections();
      await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    }
    assert.strictEqual(path.dirname(path.resolve(envDir)), tempRoot, 'Cleanup must stay inside the system temporary directory');
    assert(path.basename(envDir).startsWith('website-short-links-api-test-'));
    fs.rmdirSync(envDir);
  }
}

async function readJsonResponse(response, label) {
  const text = await response.text();
  assert((response.headers.get('content-type') || '').includes('application/json'), `${label} must return JSON, not a static page`);
  return { response, text, body: JSON.parse(text) };
}

async function testHandlerRouting() {
  const originalLoad = Module._load;
  const calls = [];
  const failureDetail = 'routing-test-private-storage-diagnostic';
  const rejectionEvents = [];
  const onUnhandledRejection = (error) => rejectionEvents.push(error);
  process.on('unhandledRejection', onUnhandledRejection);
  Module._load = function(request, parent, isMain) {
    const name = typeof request === 'string' && path.isAbsolute(request)
      ? handlerFiles.get(path.normalize(request))
      : null;
    if (!name) return originalLoad.call(this, request, parent, isMain);
    return async (req, res) => {
      let body = '';
      for await (const chunk of req) body += chunk;
      const call = {
        handler: name,
        method: req.method,
        url: req.url,
        query: req.query,
        authorization: req.headers.authorization || '',
        contentType: req.headers['content-type'] || '',
        marker: req.headers['x-routing-test'] || '',
        body
      };
      calls.push(call);
      if (req.headers['x-routing-test-failure'] === 'async') {
        await Promise.resolve();
        throw new Error(failureDetail);
      }
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('X-Delegated-Handler', name);
      res.end(JSON.stringify({ ok: true, ...call }));
    };
  };

  try {
    await withLocalServer(async (origin) => {
      const authorization = 'Bearer routing-test-workspace-access';
      const payload = ' { "destination": "https://example.test/new?keep=yes", "label": "Caf\u00e9 link" }\n';
      const routes = [
        { route: '/api/short-links', handler: 'index', query: {} },
        { route: '/api/short-links/?view=analytics&days=30', handler: 'index', query: { view: 'analytics', days: '30' } },
        { route: '/api/short-links?intent=create', handler: 'index', query: { intent: 'create' }, method: 'POST', body: payload },
        { route: '/api/short-links/health?check=retention', handler: 'router', query: { slug: ['health'], check: 'retention' } },
        { route: '/api/short-links/sets?setId=hostile&search=company+share', handler: 'router', query: { slug: ['sets'], setId: 'hostile', search: 'company share' } },
        { route: '/api/short-links/sets/', handler: 'router', query: { slug: ['sets'] } },
        { route: '/api/short-links/sets/campaign%2D2026?setId=hostile&keep=yes', handler: 'router', query: { slug: ['sets', 'campaign-2026'], setId: 'hostile', keep: 'yes' } },
        { route: '/api/short-links/sets/campaign%2D2026/generate?setId=hostile', handler: 'router', query: { slug: ['sets', 'campaign-2026', 'generate'], setId: 'hostile' }, method: 'POST', body: payload },
        { route: '/api/short-links/clicks/campaign%2Dlaunch/print%5Fqr?slug=hostile&limit=25', handler: 'router', query: { slug: ['clicks', 'campaign-launch', 'print_qr'], limit: '25' } },
        { route: '/api/short-links/test/campaign%2Dlaunch/print%5Fqr?slug=hostile&probe=1', handler: 'router', query: { slug: ['test', 'campaign-launch', 'print_qr'], probe: '1' } },
        { route: '/api/short-links/campaign%2Dlaunch/print%5Fqr?slug=hostile&keep=yes', handler: 'router', query: { slug: ['campaign-launch', 'print_qr'], keep: 'yes' }, method: 'PATCH', body: payload },
        { route: '/api/short-links/retired?slug=hostile', handler: 'router', query: { slug: ['retired'] }, method: 'DELETE' },
        { route: '/api/go/campaign%2Dlaunch/print%5Fqr?slug=hostile&__qr=1', handler: 'redirect', query: { slug: ['campaign-launch', 'print_qr'], __qr: '1' } },
        { route: '/go/portfolio?slug=hostile&utm_campaign=print', handler: 'redirect', query: { slug: ['portfolio'], utm_campaign: 'print' } }
      ];

      for (const test of routes) {
        const options = {
          method: test.method || 'GET',
          redirect: 'manual',
          headers: {
            Authorization: authorization,
            'Content-Type': 'application/json; charset=utf-8',
            'X-Routing-Test': 'forward-unchanged'
          }
        };
        if (test.body) options.body = test.body;
        const before = calls.length;
        const result = await readJsonResponse(await nativeFetch(origin + test.route, options), test.route);
        assert.strictEqual(result.response.status, 200, `${test.route} must reach its API handler`);
        assert.strictEqual(calls.length, before + 1, 'Each request must reach exactly one handler');
        assert.strictEqual(result.response.headers.get('x-delegated-handler'), test.handler);
        assert.strictEqual(result.body.handler, test.handler, `${test.route} selected the wrong handler`);
        assert.deepStrictEqual(result.body.query, test.query, 'Decoded path parameters must override conflicting query parameters without dropping unrelated fields');
        assert.strictEqual(result.body.method, options.method, 'The original request method must reach the handler');
        assert.strictEqual(result.body.authorization, authorization, 'Workspace authorization must reach the real handler');
        assert.strictEqual(result.body.contentType, options.headers['Content-Type']);
        assert.strictEqual(result.body.marker, options.headers['X-Routing-Test']);
        assert.strictEqual(result.body.body, test.body || '', 'The router must forward the original body bytes without parsing or reserializing them');
        assert.strictEqual(new URL(result.body.url, origin).search, new URL(test.route, origin).search, 'The original query string must remain available to handlers that parse req.url');
      }

      const beforeInvalid = calls.length;
      const invalid = await readJsonResponse(await nativeFetch(`${origin}/api/short-links/invalid%ZZ`), 'malformed encoded route');
      assert.strictEqual(invalid.response.status, 400, 'Malformed URL escapes must fail before invoking a handler');
      assert.strictEqual(calls.length, beforeInvalid);

      const failed = await readJsonResponse(await nativeFetch(`${origin}/api/short-links?view=analytics`, {
        headers: { 'X-Routing-Test-Failure': 'async' }
      }), 'rejected API promise');
      assert.strictEqual(failed.response.status, 500, 'A rejected handler promise must return a structured server error');
      assert.strictEqual(failed.body.ok, false);
      assert.strictEqual(typeof failed.body.error, 'string');
      assert(failed.body.error.length > 0);
      assert(!failed.text.includes(failureDetail), 'Internal handler diagnostics must not be exposed in JSON responses');
      assert((failed.response.headers.get('cache-control') || '').includes('no-store'));
      await new Promise((resolve) => setImmediate(resolve));
      assert.strictEqual(rejectionEvents.length, 0, 'Rejected API promises must not become unhandled process rejections');
      const afterFailure = await readJsonResponse(await nativeFetch(`${origin}/api/short-links/health`), 'request after rejection');
      assert.strictEqual(afterFailure.response.status, 200, 'The server must continue serving requests after a rejected handler promise');

      const publicNested = await readJsonResponse(await nativeFetch(`${origin}/go/campaign%2Dlaunch/print%5Fqr?slug=hostile&__qr=1`), 'nested public redirect');
      assert.strictEqual(publicNested.response.status, 200);
      assert.strictEqual(publicNested.body.handler, 'redirect');
      assert(Array.isArray(publicNested.body.query.slug), 'Public redirects must supply catch-all slug parameters');
      assert.strictEqual(publicNested.body.query.slug.join('/'), 'campaign-launch/print_qr', 'Public clean-URL rewrites must retain the decoded nested short-link identity');
      assert.strictEqual(publicNested.body.query.__qr, '1', 'Public rewrites must retain QR attribution');
    });
  } finally {
    Module._load = originalLoad;
    process.removeListener('unhandledRejection', onUnhandledRejection);
  }
}

async function testUnauthenticatedRealHandlers() {
  const originalLoad = Module._load;
  const previousFetch = global.fetch;
  const previousToken = process.env.SHORTLINKS_ADMIN_TOKEN;
  const storageCalls = [];
  const outboundCalls = [];
  const blockedStorage = new Proxy({}, {
    get(target, key) {
      return () => {
        storageCalls.push(String(key));
        throw new Error('Unauthenticated requests must never reach storage.');
      };
    }
  });
  process.env.SHORTLINKS_ADMIN_TOKEN = 'routing-test-unused-admin-token';
  global.fetch = async (...args) => {
    outboundCalls.push(args);
    throw new Error('Outbound requests are blocked in local routing tests.');
  };
  Module._load = function(request, parent, isMain) {
    if (typeof request === 'string' && /(?:^|[\\/])short-links-store(?:\.js)?$/.test(request)) return blockedStorage;
    return originalLoad.call(this, request, parent, isMain);
  };

  try {
    await withLocalServer(async (origin) => {
      for (const [route, method] of [
        ['/api/short-links', 'GET'],
        ['/api/short-links?view=analytics&days=30', 'GET'],
        ['/api/short-links/sets', 'GET'],
        ['/api/short-links/health', 'GET'],
        ['/api/short-links/clicks/example', 'GET'],
        ['/api/short-links/test/example', 'POST'],
        ['/api/short-links/example', 'PATCH'],
        ['/api/short-links', 'POST'],
        ['/api/short-links/sets/example/generate', 'POST']
      ]) {
        const options = { method, headers: { 'Content-Type': 'application/json' } };
        if (method === 'POST' || method === 'PATCH') options.body = JSON.stringify({ destination: 'https://example.test/never-written' });
        const result = await readJsonResponse(await nativeFetch(origin + route, options), `${method} ${route}`);
        assert.strictEqual(result.response.status, 401, `${method} ${route} must run the real authorization guard`);
        assert.deepStrictEqual(result.body, { ok: false, error: 'Unauthorized' });
      }
      assert.deepStrictEqual(storageCalls, [], 'Real list, template and analytics handlers must reject unauthorized requests before touching DynamoDB');
      assert.deepStrictEqual(outboundCalls, [], 'Requests without authentication must not perform network calls');
    });
  } finally {
    Module._load = originalLoad;
    global.fetch = previousFetch;
    if (typeof previousToken === 'undefined') delete process.env.SHORTLINKS_ADMIN_TOKEN;
    else process.env.SHORTLINKS_ADMIN_TOKEN = previousToken;
  }
}

async function testAuthenticatedMethodGuards() {
  const previousToken = process.env.SHORTLINKS_ADMIN_TOKEN;
  process.env.SHORTLINKS_ADMIN_TOKEN = 'routing-test-method-guard';
  try {
    await withLocalServer(async (origin) => {
      for (const [route, method, allow] of [
        ['/api/short-links/health', 'POST', 'GET'],
        ['/api/short-links/clicks/example', 'POST', 'GET'],
        ['/api/short-links/sets', 'PATCH', 'GET, POST'],
        ['/api/short-links/example', 'POST', 'GET, PATCH, DELETE']
      ]) {
        const result = await readJsonResponse(await nativeFetch(origin + route, {
          method,
          headers: { 'X-Admin-Token': process.env.SHORTLINKS_ADMIN_TOKEN, 'Content-Type': 'application/json' },
          body: '{}'
        }), `${method} ${route}`);
        assert.strictEqual(result.response.status, 405, 'A delegated handler must retain its own method guard');
        assert.strictEqual(result.response.headers.get('allow'), allow);
        assert.deepStrictEqual(result.body, { ok: false, error: 'Method Not Allowed' });
      }
    });
  } finally {
    if (typeof previousToken === 'undefined') delete process.env.SHORTLINKS_ADMIN_TOKEN;
    else process.env.SHORTLINKS_ADMIN_TOKEN = previousToken;
  }
}

(async () => {
  await require('./short-links-router.test')();
  await testUnauthenticatedRealHandlers();
  await testAuthenticatedMethodGuards();
  await testHandlerRouting();
  console.log('Local short-links API: handler routing, decoded path/query parameters, public redirects, request forwarding, safe async failures and authorization-before-storage passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
