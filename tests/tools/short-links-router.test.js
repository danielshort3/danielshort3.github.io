'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { Readable } = require('node:stream');
const helpers = require('../../api/_lib/short-links');

const root = path.resolve(__dirname, '../..');

function response() {
  return {
    headers: {},
    setHeader(name, value) { this.headers[name] = value; },
    end(raw) { this.body = raw ? JSON.parse(raw) : null; }
  };
}

async function run() {
  const calls = [];
  const writes = [];
  const delegated = (name) => async (req, res, options) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    calls.push({ name, req, options, body });
    helpers.sendJson(res, 200, { name, query: req.query, body });
  };
  const overrides = {
    '../_lib/short-links': {
      ...helpers,
      authorizeAdminRequest: async (req, res) => {
        if (req.headers.authorization === 'Bearer router-test-admin') return true;
        helpers.sendJson(res, 401, { ok: false, error: 'Unauthorized' });
        return false;
      }
    },
    '../_lib/short-links-management': {
      serializeLink: (link) => link,
      buildLinkPatch: (body) => body
    },
    '../_lib/short-links-store': {
      getLinkWithLegacyFallback: async (slug) => ({ slug, destination: '/original', updatedAt: 'before' }),
      updateLink: async (input) => { writes.push(input); return { slug: input.slug, ...input.patch }; },
      deleteLink: async (slug) => { writes.push({ delete: slug }); }
    },
    '../_lib/short-links-endpoints/health': delegated('health'),
    '../_lib/short-links-endpoints/clicks': delegated('clicks'),
    '../_lib/short-links-endpoints/sets': delegated('sets'),
    '../_lib/short-links-test': delegated('test')
  };
  const filename = path.join(root, 'api/short-links/[...slug].js');
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
    module, URL,
    require(name) {
      assert(Object.hasOwn(overrides, name), `Unexpected dependency: ${name}`);
      return overrides[name];
    }
  }, { filename });
  const router = module.exports;
  async function call(url, { method = 'GET', query = {}, body = '', authorized = true } = {}) {
    const req = Readable.from(body ? [Buffer.from(body)] : []);
    Object.assign(req, {
      url, method, query,
      headers: {
        host: 'example.test',
        'content-type': 'application/json',
        authorization: authorized ? 'Bearer router-test-admin' : ''
      }
    });
    const res = response();
    await router(req, res);
    return { req, res };
  }

  const payload = ' { "label": "Café", "destination": "/new?keep=yes" }\n';
  const routes = [
    ['/api/short-links/health?slug=sets/evil&check=all', 'health', { slug: 'sets/evil', check: 'all' }, { check: 'all' }],
    ['/api/short-links/sets?setId=evil&search=hello', 'sets', { setId: 'evil', search: 'hello' }, { setId: ['__collection__'], search: 'hello' }],
    ['/api/short-links/sets/', 'sets', {}, { setId: ['__collection__'] }],
    ['/api/short-links/sets%2F__collection__', 'sets', { slug: 'sets/__collection__' }, { setId: ['__collection__'] }],
    ['/api/short-links/sets/campaign%2D2026?setId=evil', 'sets', { setId: 'evil' }, { setId: ['campaign-2026'] }],
    ['/api/short-links/sets%2Fcampaign%2Fgenerate?setId=evil&keep=yes', 'sets', { setId: 'evil', keep: 'yes' }, { setId: ['campaign', 'generate'], keep: 'yes' }],
    ['/api/short-links/clicks/campaign%2Dlaunch/print%5Fqr?slug=evil&limit=25', 'clicks', { slug: 'evil', limit: '25' }, { slug: ['campaign-launch', 'print_qr'], limit: '25' }],
    ['/api/short-links/clicks%2Fcampaign%2Fprint_qr', 'clicks', { slug: 'clicks/campaign/print_qr' }, { slug: ['campaign', 'print_qr'] }],
    ['/api/short-links/test%2Fcampaign%2Fprint_qr?slug=evil', 'test', { slug: 'evil' }, { slug: ['campaign', 'print_qr'] }]
  ];
  for (const [url, name, query, expected] of routes) {
    const before = calls.length;
    const { req, res } = await call(url, { method: 'POST', query, body: payload });
    assert.equal(res.statusCode, 200);
    assert.equal(calls.length, before + 1);
    const last = calls.at(-1);
    assert.equal(last.name, name);
    assert.equal(last.req, req, 'Delegation must retain the original request stream');
    assert.equal(last.req.url, url, 'The original URL and query string must reach the handler');
    assert.equal(last.req.method, 'POST');
    assert.equal(last.req.headers.authorization, 'Bearer router-test-admin');
    assert.equal(last.body, payload, 'Delegation must preserve the original body bytes');
    assert.deepEqual(res.body.query, expected, 'URL path parameters must override spoofed query parameters');
    if (name === 'test') assert.equal(last.options.slug, 'campaign/print_qr');
  }

  const malformed = await call('/api/short-links/sets/invalid%ZZ');
  assert.equal(malformed.res.statusCode, 400);
  assert.equal(calls.length, routes.length);
  const forbidden = await call('/api/short-links/example', { method: 'PATCH', body: payload, authorized: false });
  assert.equal(forbidden.res.statusCode, 401);
  assert.equal(writes.length, 0);
  const updated = await call('/api/short-links/campaign%2Fprint_qr?slug=evil', { method: 'PATCH', query: { slug: 'evil' }, body: payload });
  assert.equal(updated.res.statusCode, 200);
  assert.equal(updated.res.body.link.slug, 'campaign/print_qr');
  assert.equal(writes[0].slug, 'campaign/print_qr');
  assert.equal(writes[0].patch.destination, '/new?keep=yes');
  assert.equal(writes[0].expectedUpdatedAt, 'before');
  const invalidBody = await call('/api/short-links/example', { method: 'PATCH', body: '{broken' });
  assert.equal(invalidBody.res.statusCode, 400);
  const unsupported = await call('/api/short-links/example', { method: 'POST', body: payload });
  assert.equal(unsupported.res.statusCode, 405);
  assert.equal(unsupported.res.headers.Allow, 'GET, PATCH, DELETE');
  const removed = await call('/api/short-links/retired?slug=evil', { method: 'DELETE', query: { slug: 'evil' } });
  assert.equal(removed.res.statusCode, 200);
  assert.equal(writes.at(-1).delete, 'retired');

  const config = JSON.parse(fs.readFileSync(path.join(root, 'vercel.json'), 'utf8'));
  for (const [source, destination] of [
    ['/api/short-links/sets', '/api/short-links/sets%2F__collection__'],
    ['/api/short-links/sets/:setId/generate', '/api/short-links/sets%2F:setId%2Fgenerate'],
    ['/api/short-links/sets/:setId', '/api/short-links/sets%2F:setId'],
    ['/api/short-links/clicks/:slug*', '/api/short-links/clicks%2F:slug*']
  ]) {
    assert(config.rewrites.some((rule) => rule.source === source && rule.destination === destination), `${source} must target the shared catchall`);
  }
  const deployed = [];
  const ignored = fs.readFileSync(path.join(root, '.vercelignore'), 'utf8').split(/\r?\n/).filter((line) => line.startsWith('/api/'));
  function countFunctions(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
      const absolute = path.join(directory, entry.name);
      const relative = '/' + path.relative(root, absolute).replaceAll(path.sep, '/');
      if (ignored.some((pattern) => pattern.endsWith('/') ? (relative + '/').startsWith(pattern) : relative === pattern)) continue;
      if (entry.isDirectory()) countFunctions(absolute);
      else if (/\.(?:[cm]?js|ts|py|go|rb)$/.test(entry.name)) deployed.push(relative);
    }
  }
  countFunctions(path.join(root, 'api'));
  assert(deployed.length <= 12, `Hobby supports at most 12 standalone functions; found ${deployed.length}: ${deployed.join(', ')}`);
  for (const removedFile of ['api/short-links/health.js', 'api/short-links/clicks/[...slug].js', 'api/short-links/sets/[...setId].js']) {
    assert(!fs.existsSync(path.join(root, removedFile)), `${removedFile} must not create a duplicate function`);
  }
  console.log(`Short-links shared router: encoded routes, query isolation, body/auth forwarding, mutations, errors and ${deployed.length}/12 functions passed.`);
}

module.exports = run;
if (require.main === module) {
  run().catch((error) => { console.error(error); process.exitCode = 1; });
}
