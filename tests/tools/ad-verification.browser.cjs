'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const { chromium } = require('playwright');
const core = require('../../js/demos/ad-verification-core');
const repository = path.resolve(__dirname, '../..');
const root = fs.existsSync(path.join(repository, 'public/demos/ad-verification.html')) ? path.join(repository, 'public') : repository;
const output = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'ad-verification-browser');
const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.woff2': 'font/woff2', '.webp': 'image/webp' };
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    if (!path.extname(file)) file += '.html';
    const bytes = fs.readFileSync(file);
    response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    response.end(bytes);
  } catch (_) { response.writeHead(404).end('Not found'); }
});

(async () => {
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    const url = origin + '/demos/ad-verification';
    fs.mkdirSync(output, { recursive: true });
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1448, height: 1086 }, reducedMotion: 'reduce' });
    // Production is never requested, including by an accidental new integration.
    await context.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    const page = await context.newPage();
    const errors = [];
    const failedResources = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => { if (response.status() >= 400) failedResources.push(response.url()); });
    const verdict = async (text) => page.waitForFunction((expected) => document.querySelector('[data-av-verdict]').textContent === expected && document.querySelector('[data-av-demo]').getAttribute('aria-busy') === 'false', text);
    await page.goto(url);
    await verdict('Verified');
    assert.equal(await page.title(), 'Ad Verification Demo | Daniel Short');
    assert.equal(await page.evaluate(() => isSecureContext && !!crypto.subtle), true);
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    assert.equal(await page.locator('[data-av-step][data-state="verified"]').count(), 4);
    await page.screenshot({ path: path.join(output, 'ad-verification-desktop.png'), fullPage: true });

    await page.getByRole('button', { name: 'Simulate change', exact: true }).click();
    await verdict('Change detected');
    assert.equal(await page.locator('[data-av-step="2"]').getAttribute('data-state'), 'changed');
    assert.equal(await page.locator('[data-av-step="3"]').getAttribute('data-state'), 'dependent');
    assert.match(await page.locator('[data-av-value="2"]').textContent(), /12,500/);
    await page.screenshot({ path: path.join(output, 'ad-verification-tampered.png'), fullPage: true });
    await page.locator('[data-av-step="2"]').focus();
    await page.keyboard.press('Enter');
    await page.locator('dialog[open]').waitFor();
    assert.match(await page.locator('dialog').innerText(), /Event signature: fails/);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('dialog[open]').count(), 0);
    assert.equal(await page.locator('[data-av-step="2"]').evaluate((element) => document.activeElement === element), true);
    await page.getByRole('button', { name: 'Restore original', exact: true }).click();
    await verdict('Verified');
    await page.getByRole('button', { name: 'Launch Demo', exact: true }).click();
    await verdict('Verified');
    await page.locator('.av-details > summary').click();
    await page.getByRole('button', { name: 'Check chain again', exact: true }).click();
    await verdict('Verified');
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export public proof', exact: true }).click();
    const download = await downloadPromise;
    const proof = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
    assert.equal(Object.values(proof.trust.publicKeys).some((key) => 'd' in key), false);
    await page.locator('.av-details > summary').click();

    for (const width of [768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => scrollTo(0, 0));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `Overflow at ${width}px`);
      await page.screenshot({ path: path.join(output, `ad-verification-${width}.png`), fullPage: true });
      await page.getByRole('button', { name: 'Simulate change', exact: true }).click();
      await verdict('Change detected');
      await page.getByRole('button', { name: 'Restore original', exact: true }).click();
      await verdict('Verified');
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(failedResources, []);

    const noJs = await browser.newContext({ javaScriptEnabled: false });
    const noJsPage = await noJs.newPage();
    await noJsPage.goto(url);
    assert.match(await noJsPage.locator('noscript').innerText(), /JavaScript/);
    assert.equal(await noJsPage.getByRole('button', { name: 'Simulate change', exact: true }).isDisabled(), true);
    await noJs.close();

    const unavailable = await browser.newContext();
    const unavailablePage = await unavailable.newPage();
    await unavailablePage.route('**/js/demos/ad-verification-core.js*', (route) => route.abort());
    await unavailablePage.goto(url);
    await unavailablePage.waitForFunction(() => document.querySelector('[data-av-verdict]').textContent === 'Unable to verify');
    assert.equal(await unavailablePage.locator('[data-av-step][data-state="verified"]').count(), 0);
    await unavailable.close();
    console.log('Ad verification browser checks passed: real Web Crypto, tamper/restore, inspection, keyboard, export, error/no-JS states, 1448/768/390/320px.');
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
