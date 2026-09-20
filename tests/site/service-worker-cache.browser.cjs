'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const source = fs.readFileSync(path.join(__dirname, '../../sw.js'), 'utf8');
const version = source.match(/const VERSION = '([^']+)'/)[1];
const legacy = `self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));`;

async function eventually(check, message) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (await check()) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(message);
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'worker-cache-env-'));
  const server = createLocalServer({ envDir });
  const original = server.listeners('request')[0];
  server.removeListener('request', original);
  let worker = legacy;
  let value = 'one';
  let privateAsset = false;
  const requests = new Map();
  server.on('request', (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/sw.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript', 'Cache-Control': 'no-store' });
      return res.end(worker);
    }
    if (pathname === '/__cache-fixture__') {
      res.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
      return res.end('<!doctype html><html lang="en"><title>Cache fixture</title><body>Offline cache test</body></html>');
    }
    // The development server intentionally marks HTML no-store. Supply a
    // cacheable public shell fixture without weakening that production rule.
    if (['/', '/index.html', '/portfolio', '/tools', '/games'].includes(pathname)) {
      res.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'public, max-age=0' });
      return res.end('<!doctype html><title>Public shell fixture</title>Public shell');
    }
    if (pathname.startsWith('/__cache_test__/')) {
      requests.set(pathname, (requests.get(pathname) || 0) + 1);
      res.writeHead(200, {
        'Content-Type': pathname.endsWith('.js') ? 'text/javascript' : 'text/html',
        'Cache-Control': pathname.endsWith('/private.js') && privateAsset ? 'private, no-store' : 'public, max-age=0',
        ...(pathname.endsWith('/vary.js') ? { Vary: 'X-Fixture-Mode' } : {})
      });
      return res.end(pathname.endsWith('/vary.js') ? req.headers['x-fixture-mode'] || 'default' : value);
    }
    return original(req, res);
  });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const context = await browser.newContext({ serviceWorkers: 'allow' });
    let page = await context.newPage();
    await page.goto(`${base}/__cache-fixture__`);
    await page.evaluate(async () => {
      await navigator.serviceWorker.register('/sw.js');
      await navigator.serviceWorker.ready;
    });
    await page.waitForFunction(() => navigator.serviceWorker.controller);
    await page.evaluate(async () => {
      await (await caches.open('ds-v2')).put('/old.css', new Response('legacy'));
      await (await caches.open('other-app-cache')).put('/keep', new Response('unrelated'));
    });
    worker = source;
    await page.evaluate(async () => (await navigator.serviceWorker.getRegistration()).update());
    await eventually(() => page.evaluate(async () => (await navigator.serviceWorker.getRegistration()).waiting?.state === 'installed'), 'Updated worker must wait while an old page remains open').catch(async error => {
      console.error(await page.evaluate(async () => {
        const r = await navigator.serviceWorker.getRegistration();
        const names = await caches.keys();
        return { installing: r.installing?.state, waiting: r.waiting?.state, active: r.active?.state,
          caches: await Promise.all(names.map(async name => ({ name, urls: (await (await caches.open(name)).keys()).map(key => key.url) }))) };
      }));
      throw error;
    });
    assert((await page.evaluate(() => caches.keys())).includes('ds-v2'), 'waiting update cannot delete caches underneath the active old page');
    await page.close();
    page = await context.newPage();
    await page.goto(`${base}/__cache-fixture__`);
    await eventually(() => page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration();
      return registration.active?.state === 'activated' && !registration.waiting && !(await caches.keys()).includes('ds-v2');
    }), 'New worker must activate and retire the legacy cache after the old page closes');
    assert((await page.evaluate(() => caches.keys())).includes('other-app-cache'), 'unrelated cache survives activation');

    const read = (pathname, options) => page.evaluate(async ({ pathname, options }) => (await fetch(pathname, options)).text(), { pathname, options });
    const cachedBody = async (pathname, expected, bucket = 'core') => {
      await eventually(() => page.evaluate(async ({ pathname, expected, bucket, version }) => {
        const response = await (await caches.open(`${version}-${bucket}`)).match(pathname);
        const record = await (await caches.open(`${version}-metadata`)).match(`${location.origin}/__site-cache-metadata__/${bucket}`);
        const entries = record ? await record.json() : [];
        return response && await response.text() === expected && entries.some(entry =>
          entry.url === new URL(pathname, location.origin).href && entry.bytes === new TextEncoder().encode(expected).byteLength);
      }, { pathname, expected, bucket, version }), `Cached body and metadata must settle for ${pathname}`);
    };
    const stable = '/__cache_test__/stable.0123456789abcdef.js';
    assert.equal(await read(stable), 'one');
    await cachedBody(stable, 'one');
    value = 'two';
    assert.equal(await read(stable), 'one');
    assert.equal(requests.get(stable), 1, 'immutable cache hit does not refetch');

    const bare = '/__cache_test__/bare.js';
    assert.equal(await read(bare), 'two');
    await cachedBody(bare, 'two');
    value = 'three';
    assert.equal(await read(bare), 'two', 'unversioned response remains immediate');
    await cachedBody(bare, 'three');
    assert.equal(requests.get(bare), 2, 'background refresh reaches the real server');

    await page.evaluate(async ({ version, bare }) => {
      const cache = await caches.open(`${version}-metadata`);
      const key = `${location.origin}/__site-cache-metadata__/core`;
      const entries = await (await cache.match(key)).json();
      entries.find(entry => entry.url === new URL(bare, location.origin).href).storedAt = 0;
      await cache.put(key, new Response(JSON.stringify(entries)));
    }, { version, bare });
    value = 'four';
    assert.equal(await read(bare), 'four', 'expired entries are fetched rather than served');
    await cachedBody(bare, 'four');

    const varied = '/__cache_test__/vary.js';
    assert.equal(await read(varied, { headers: { 'X-Fixture-Mode': 'first' } }), 'first');
    await eventually(() => page.evaluate(async ({ version, varied }) => {
      const response = await (await caches.open(`${version}-core`)).match(new Request(new URL(varied, location.origin), { headers: { 'X-Fixture-Mode': 'first' } }));
      return response && await response.text() === 'first';
    }, { version, varied }), 'First Vary response must be cached');
    assert.equal(await read(varied, { headers: { 'X-Fixture-Mode': 'second variant' } }), 'second variant');
    await eventually(() => page.evaluate(async ({ version, varied }) => {
      const cache = await caches.open(`${version}-core`);
      const matching = (await cache.keys()).filter(request => new URL(request.url).pathname === varied);
      const response = await cache.match(new Request(new URL(varied, location.origin), { headers: { 'X-Fixture-Mode': 'second variant' } }));
      return matching.length === 1 && response && await response.text() === 'second variant';
    }, { version, varied }), 'A Vary replacement must not leave uncounted response variants');

    const restricted = '/__cache_test__/private.js';
    assert.equal(await read(restricted), 'four');
    await cachedBody(restricted, 'four');
    privateAsset = true;
    value = 'private';
    await read(restricted);
    await eventually(() => page.evaluate(async ({ version, restricted }) => !await (await caches.open(`${version}-core`)).match(restricted), { version, restricted }), 'Private response must remove its prior cache entry');
    assert.equal(await read(restricted), 'private');
    const authenticated = '/__cache_test__/auth.js';
    await read(authenticated, { headers: { Authorization: 'Bearer test-fixture' } });
    assert.equal(await page.evaluate(async ({ version, authenticated }) => Boolean(await (await caches.open(`${version}-core`)).match(authenticated)), { version, authenticated }), false);

    value = 'exact document';
    const documentPath = '/__cache_test__/project';
    await read(documentPath, { headers: { 'X-Site-Route': '1' } });
    await cachedBody(documentPath, value, 'documents');
    await context.setOffline(true);
    assert.equal(await read(documentPath, { headers: { 'X-Site-Route': '1' } }), 'exact document');
    assert.equal(await read(stable), 'one');
    const missingFailed = await page.evaluate(async () => {
      try { await fetch('/__cache_test__/missing', { headers: { 'X-Site-Route': '1' } }); return false; } catch { return true; }
    });
    assert(missingFailed, 'offline unknown routes never receive the homepage');
    await context.setOffline(false);
    await context.close();
    console.log('Real service worker passed: natural upgrade, owned-cache migration, immutable hits, background refresh, expiry, Vary replacement, privacy, Authorization bypass, and exact offline routes.');
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
