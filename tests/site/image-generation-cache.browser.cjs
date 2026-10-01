'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { imageFixture } = require('./helpers/image-generation-fixture.cjs');
const icons = require('../../js/common/catalog-icons');

const worker = fs.readFileSync(path.resolve(__dirname, '../../sw.js'), 'utf8');
const version = worker.match(/const VERSION = '([^']+)'/)[1];

async function eventually(check, message) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(message);
}

(async () => {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'image-cache-upgrade-'));
  const fixture = await imageFixture(fixtureRoot);
  let deployed = false;
  const imageRequests = [];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/sw.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript', 'Cache-Control': 'no-store' });
      return res.end(worker);
    }
    if (url.pathname === fixture.newUrl.split('?')[0]) {
      imageRequests.push(url.pathname + url.search);
      res.writeHead(200, { 'Content-Type': 'image/webp', 'Cache-Control': 'public, max-age=2592000' });
      return res.end(deployed ? fixture.newWebp : fixture.oldWebp);
    }
    if (url.pathname === `/${fixture.relative}`) {
      res.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=2592000' });
      return res.end(fixture.png);
    }
    res.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'public, max-age=0, must-revalidate' });
    return res.end('<!doctype html><title>Image upgrade fixture</title><main>Cache upgrade fixture</main>');
  });
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ serviceWorkers: 'allow' });
    const page = await context.newPage();
    await page.goto(`${origin}/cache-fixture`);
    await page.evaluate(async () => {
      await navigator.serviceWorker.register('/sw.js');
      await navigator.serviceWorker.ready;
    });
    await page.waitForFunction(() => navigator.serviceWorker.controller);
    const read = (url) => page.evaluate(async (pathname) => {
      const bytes = await (await fetch(pathname)).arrayBuffer();
      const hash = await crypto.subtle.digest('SHA-256', bytes);
      return [...new Uint8Array(hash)].map((value) => value.toString(16).padStart(2, '0')).join('');
    }, url);
    const settled = (url, bytes) => eventually(() => page.evaluate(async ({ url, bytes, version }) => {
      const cache = await caches.open(`${version}-media`);
      const response = await cache.match(url);
      const metadata = await (await caches.open(`${version}-metadata`)).match(`${location.origin}/__site-cache-metadata__/media`);
      const entries = metadata ? await metadata.json() : [];
      return response && (await response.arrayBuffer()).byteLength === bytes && entries.some((entry) => entry.url === new URL(url, location.origin).href && entry.bytes === bytes);
    }, { url, bytes, version }), `Actual CacheStorage and size metadata must settle for ${url}`);

    assert.equal(await read(fixture.legacyUrl), fixture.oldHash);
    await settled(fixture.legacyUrl, fixture.oldWebp.length);
    deployed = true;
    assert.equal(await read(fixture.legacyUrl), fixture.oldHash, 'An older versioned CacheStorage hit still returns the old variant after deployment.');
    assert.equal(await read(fixture.newUrl), fixture.newHash, 'The recipe-aware URL must fetch the replacement WebP despite the old HTTP/SW caches.');
    await settled(fixture.newUrl, fixture.newWebp.length);
    const markup = icons.render(`<img src="${fixture.newPngUrl}" alt="Fixture">`);
    await page.evaluate((html) => { document.querySelector('main').innerHTML = html; }, markup);
    await page.waitForFunction(() => document.querySelector('img')?.complete && document.querySelector('img').naturalWidth === 128);
    assert.equal(await page.locator('img').evaluate((image) => image.currentSrc), `${origin}${fixture.newUrl}`,
      'The real picture element must display the new optimized variant.');
    assert.notEqual(fixture.legacyUrl, fixture.newUrl);
    assert(imageRequests.includes(fixture.newUrl), 'The new key must cause a real network request.');
    await context.setOffline(true);
    assert.equal(await read(fixture.newUrl), fixture.newHash, 'The new image remains usable from the actual cache while offline.');
    assert.equal(await read(fixture.legacyUrl), fixture.oldHash, 'The two cached generations remain independent.');
    assert.equal(await page.evaluate(() => navigator.serviceWorker.controller.scriptURL), `${origin}/sw.js`);
    console.log(JSON.stringify({ unchangedPngHash: fixture.pngHash, oldUrl: fixture.legacyUrl, newUrl: fixture.newUrl,
      oldBytes: fixture.oldWebp.length, newBytes: fixture.newWebp.length, imageRequests,
      cachedUpgrade: true, renderedWidth: 128, independentOfflineGenerations: true }, null, 2));
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(fixtureRoot, { recursive: true, force: true });
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
