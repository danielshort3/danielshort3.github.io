'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../sw.js'), 'utf8');
const origin = 'https://cache.example';
const request = (pathname, options = {}) => {
  const req = new Request(new URL(pathname, origin), options);
  if (options.document) Object.defineProperty(req, 'mode', { value: 'navigate' });
  return req;
};

function harness() {
  const listeners = {};
  const stores = new Map();
  const calls = [];
  let now = 1_800_000_000_000;
  let failure = null;
  let claimed = 0;
  let handler = async () => new Response('network');
  const key = value => new URL(typeof value === 'string' ? value : value.url, origin).href;
  const caches = {
    async keys() { return [...stores.keys()]; },
    async delete(name) { return stores.delete(name); },
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const entries = stores.get(name);
      return {
        async keys() { return [...entries.keys()].map(url => new Request(url)); },
        async delete(value) { return entries.delete(key(value)); },
        async match(value) { return entries.get(key(value))?.clone(); },
        async put(value, response) {
          if (failure?.(name, key(value))) {
            const error = new Error('Storage full');
            error.name = 'QuotaExceededError';
            throw error;
          }
          const body = await response.arrayBuffer();
          entries.set(key(value), new Response(body, { status: response.status, headers: response.headers }));
        }
      };
    }
  };
  const sandbox = {
    URL, Request, Response, Date: { now: () => now }, location: { origin }, caches,
    fetch: async req => { calls.push(req.url); return handler(req); },
    self: {
      addEventListener: (type, callback) => { listeners[type] = callback; },
      clients: { claim: async () => { claimed += 1; } },
      skipWaiting: () => { throw new Error('Updates must wait for active pages'); }
    }
  };
  vm.runInNewContext(source + '\nglobalThis.api = { VERSION, POLICIES, shouldHandle, bucketFor, canStore, isVersioned, storeResponse, readCached };', sandbox);
  return {
    api: sandbox.api, stores, caches, calls,
    advance: milliseconds => { now += milliseconds; },
    network: callback => { handler = callback; },
    failWrites: callback => { failure = callback; },
    claimed: () => claimed,
    async lifecycle(type) {
      const work = [];
      listeners[type]({ waitUntil: promise => work.push(promise) });
      await Promise.all(work);
    },
    dispatch(req) {
      const work = [];
      let response;
      let synchronous = true;
      listeners.fetch({ request: req,
        waitUntil(promise) { assert(synchronous, 'waitUntil must be registered during event dispatch'); work.push(promise); },
        respondWith(promise) { response = Promise.resolve(promise); }
      });
      synchronous = false;
      return { response, work, settled: () => Promise.all(work) };
    }
  };
}

async function classification() {
  const h = harness();
  assert.deepEqual(JSON.parse(JSON.stringify(h.api.POLICIES)), {
    documents: { entries: 32, bytes: 8388608, entryBytes: 1048576, age: 604800000 },
    core: { entries: 192, bytes: 16777216, entryBytes: 8388608, age: 2592000000 },
    media: { entries: 96, bytes: 50331648, entryBytes: 8388608, age: 1209600000 }
  });
  for (const req of [request('/api/private.json'), request('/admin/config.json'), request('/api'),
    request('/tools/job-application-tracker'), request('/pages/transcribe.html'),
    request('/tools/dashboard?code=fixture&state=fixture', { document: true }),
    request('/tools/dashboard.html', { document: true }), request('/pages/tools-dashboard.html', { document: true }),
    request('/tools-dashboard', { document: true }), request('/?code=fixture', { document: true }),
    request('/tools/text-compare?state=private-fixture', { document: true }),
    request('/file.js?ACCESS_TOKEN=fixture'), request('/file.js?id_token=fixture'),
    request('/file.js?refresh_token=fixture'), request('/file.js?token=fixture'),
    request('/file.js', { headers: { Authorization: 'Bearer fixture' } }),
    request('/file.wasm', { headers: { Range: 'bytes=0-2' } }),
    request('/file.js', { method: 'POST' }), request('https://third.example/file.js')]) {
    assert.equal(h.api.shouldHandle(req), false, req.url);
    assert.equal(h.dispatch(req).response, undefined, 'excluded requests remain entirely browser-owned');
  }
  assert.equal(h.api.bucketFor(request('/tools', { document: true })), 'documents');
  assert.equal(h.api.bucketFor(request('/tools', { headers: { 'X-Site-Route': '1' } })), 'documents');
  assert.equal(h.api.bucketFor(request('/test.js')), 'core');
  assert.equal(h.api.bucketFor(request('/image.webp')), 'media');
  for (const extension of ['avif', 'gif', 'bmp']) assert.equal(h.api.bucketFor(request(`/image.${extension}`)), 'media');
  assert.equal(h.api.bucketFor(request('/model.bin')), null);
  assert(h.api.isVersioned(request('/dist/shell.012345abcdef.js')));
  assert(h.api.isVersioned(request('/component.css?v=012345abcdef')));
  assert(!h.api.isVersioned(request('/component.css?v=latest')));
}

async function limits() {
  const h = harness();
  Object.assign(h.api.POLICIES.core, { entries: 3, bytes: 12, entryBytes: 8, age: 1000 });
  for (const name of ['a', 'b', 'c']) {
    await h.api.storeResponse('core', request(`/${name}.js`), new Response('1234'));
    h.advance(5);
  }
  await h.api.readCached('core', request('/a.js'));
  await h.api.storeResponse('core', request('/d.js'), new Response('1234'));
  assert.equal(await h.api.readCached('core', request('/b.js')), undefined, 'least recently used entry is evicted');
  assert(await h.api.readCached('core', request('/a.js')), 'a cache hit protects recently used content');
  await h.api.storeResponse('core', request('/large.js'), new Response('12345678'));
  const cache = await h.caches.open(`${h.api.VERSION}-core`);
  assert.equal((await cache.keys()).length, 2, 'byte limit can evict before count limit');
  assert.equal(await h.api.storeResponse('core', request('/too-big.js'), new Response('123456789')), false);
  assert.equal(await cache.match(request('/too-big.js')), undefined);
  assert.equal(await h.api.storeResponse('core', request('/declared.js'), new Response('a', { headers: { 'Content-Length': '9' } })), false);
  h.advance(1001);
  assert.equal(await h.api.readCached('core', request('/large.js')), undefined, 'age expires even a recently accessed entry');
  await Promise.all(Array.from({ length: 12 }, (_, i) => h.api.storeResponse('core', request(`/concurrent-${i}.js`), new Response('1234'))));
  assert.equal((await cache.keys()).length, 3, 'concurrent writes cannot exceed limits');
  const metadata = await (await h.caches.open(`${h.api.VERSION}-metadata`)).match(`${origin}/__site-cache-metadata__/core`);
  const entries = await metadata.json();
  assert.equal(entries.length, 3);
  assert.equal(entries.reduce((sum, entry) => sum + entry.bytes, 0), 12);
  await cache.put(request('/orphan.js'), new Response('unrecorded'));
  assert.equal(await h.api.readCached('core', request('/orphan.js')), undefined, 'interrupted/untracked writes are not trusted');
  assert.equal(await cache.match(request('/orphan.js')), undefined);
}

async function privacyAndFailure() {
  const h = harness();
  const req = request('/privacy.js');
  for (const headers of [{ 'Cache-Control': 'private' }, { 'Cache-Control': 'public, no-store' },
    { 'Cache-Control': 'private="Set-Cookie"' }, { Vary: 'Accept-Encoding, *' }]) {
    await h.api.storeResponse('core', req, new Response('old public'));
    assert.equal(await h.api.storeResponse('core', req, new Response('private', { headers })), false);
    assert.equal(await h.api.readCached('core', req), undefined, 'a private replacement evicts old public content');
  }
  for (const response of [new Response('partial', { status: 206 }), new Response('error', { status: 500 }), Response.redirect(`${origin}/login`)]) {
    assert.equal(h.api.canStore(response), false);
  }
  let failures = 0;
  h.failWrites(name => name.endsWith('-core') && ++failures === 1);
  assert.equal(await h.api.storeResponse('core', req, new Response('retry once')), true, 'quota eviction permits one retry');
  assert.equal(failures, 2);
  h.failWrites(name => name.endsWith('-core'));
  const attempt = h.dispatch(request('/network-survives.js'));
  assert.equal(await (await attempt.response).text(), 'network', 'cache failures do not delay or reject a valid network response');
  await attempt.settled();
  assert.equal(await h.api.readCached('core', request('/network-survives.js')), undefined);
  h.failWrites(name => name.endsWith('-metadata'));
  assert.equal(await h.api.storeResponse('core', request('/orphan-failure.js'), new Response('body')), false);
  assert.equal(await (await h.caches.open(`${h.api.VERSION}-core`)).match(request('/orphan-failure.js')), undefined);
}

async function fetchBehavior() {
  const h = harness();
  const stable = request('/asset.js?v=012345abcdef');
  await h.api.storeResponse('core', stable, new Response('immutable'));
  let event = h.dispatch(stable);
  assert.equal(await (await event.response).text(), 'immutable');
  await event.settled();
  assert.equal(h.calls.length, 0, 'versioned cache hits avoid background requests');
  const bare = request('/asset.js');
  await h.api.storeResponse('core', bare, new Response('old'));
  let finish;
  h.network(() => new Promise(resolve => { finish = resolve; }));
  event = h.dispatch(bare);
  assert.equal(event.work.length, 1);
  assert.equal(await (await event.response).text(), 'old', 'unversioned cached response does not wait for refresh');
  finish(new Response('new'));
  await event.settled();
  assert.equal(await (await h.api.readCached('core', bare)).text(), 'new', 'waitUntil includes the cache write');
  const doc = request('/portfolio/example', { document: true });
  await h.api.storeResponse('documents', doc, new Response('exact project'));
  await h.api.storeResponse('documents', request('/index.html'), new Response('root only'));
  h.network(async () => { throw new Error('Offline'); });
  event = h.dispatch(doc);
  assert.equal(await (await event.response).text(), 'exact project');
  await event.settled();
  event = h.dispatch(request('/missing-project', { document: true }));
  await assert.rejects(event.response, /Offline/, 'another project never receives homepage HTML');
  await event.settled();
  event = h.dispatch(request('/?welcome', { document: true }));
  assert.equal(await (await event.response).text(), 'root only');
  await event.settled();
  h.network(async () => new Response('server failed', { status: 503 }));
  event = h.dispatch(doc);
  assert.equal((await event.response).status, 503, 'server errors are not silently replaced with stale success');
  await event.settled();
}

async function lifecycle() {
  const h = harness();
  await h.caches.open('ds-v2');
  await h.caches.open('ds-site-v2-core');
  await h.caches.open('another-app-cache');
  await h.lifecycle('install');
  assert(await h.api.readCached('documents', request('/')));
  await h.lifecycle('activate');
  const names = await h.caches.keys();
  assert(!names.includes('ds-v2') && !names.includes('ds-site-v2-core'));
  assert(names.includes('another-app-cache'), 'activation preserves unrelated origin caches');
  assert.equal(h.claimed(), 1);
  const failed = harness();
  failed.network(async () => new Response('Unavailable', { status: 503 }));
  await assert.rejects(failed.lifecycle('install'), /Offline shell unavailable/);
}

(async () => {
  await classification();
  await limits();
  await privacyAndFailure();
  await fetchBehavior();
  await lifecycle();
  console.log('Service worker cache: privacy, expiry/LRU/byte/count limits, concurrency, quota recovery, background lifetime, exact offline routes, and safe activation passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
