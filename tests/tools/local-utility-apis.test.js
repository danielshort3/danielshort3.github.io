'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { once } = require('events');
const { createLocalServer } = require('../../build/dev');

async function run() {
  const root = path.resolve(__dirname, '../..');
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-utility-apis-'));
  const originalLoad = Module._load;
  const entries = new Map([
    [path.join(root, 'api/ga4/report.js'), 'ga4'],
    [path.join(root, 'api/contact.js'), 'contact']
  ]);
  const calls = [];
  let server;
  Module._load = function(request, parent, isMain) {
    if (!entries.has(request)) return originalLoad.call(this, request, parent, isMain);
    return async (req, res) => {
      if (req.headers['x-test-reject']) throw new Error('Private provider diagnostic');
      let body = '';
      for await (const chunk of req) body += chunk;
      const entry = {
        handler: entries.get(request), method: req.method,
        authorization: req.headers.authorization, query: req.query, body
      };
      calls.push(entry);
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(entry));
    };
  };
  try {
    server = createLocalServer({ envDir });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const origin = 'http://127.0.0.1:' + server.address().port;
    for (const [route, handler] of [['/api/ga4/report', 'ga4'], ['/api/contact', 'contact']]) {
      const body = '{"test":"local routing only"}';
      const response = await fetch(origin + route + '?mode=test', {
        method: 'POST', headers: { Authorization: 'Bearer test-only', 'Content-Type': 'application/json' }, body
      });
      assert.strictEqual(response.status, 200);
      assert.deepStrictEqual(await response.json(), {
        handler, method: 'POST', authorization: 'Bearer test-only', query: { mode: 'test' }, body
      });
      const rejected = await fetch(origin + route, { headers: { 'X-Test-Reject': '1' } });
      assert.strictEqual(rejected.status, 500);
      assert.strictEqual(rejected.headers.get('cache-control'), 'no-store');
      const failure = await rejected.json();
      assert.deepStrictEqual(failure, { ok: false, error: 'Local API request failed.' });
    }
    assert.strictEqual(calls.length, 2, 'Only supported requests should reach the API implementations.');
    const unknown = await fetch(origin + '/api/ga4/not-a-route');
    assert.strictEqual(unknown.status, 404);
    await unknown.text();
    console.log('Local utility APIs: GA4/contact routing, original request forwarding, safe errors and unknown routes passed.');
  } finally {
    Module._load = originalLoad;
    if (server?.listening) {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    }
    assert.strictEqual(path.dirname(envDir), path.resolve(os.tmpdir()));
    assert(path.basename(envDir).startsWith('local-utility-apis-'));
    fs.rmdirSync(envDir);
  }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
