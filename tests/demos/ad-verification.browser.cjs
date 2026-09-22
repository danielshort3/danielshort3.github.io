/** Run after npm run build. Uses the site's real routing, headers and public output. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const { createLocalServer } = require('../../build/dev');
const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'ad-verification-browser');
const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ad-verification-env-'));
fs.mkdirSync(artifactDir, { recursive: true });
const server = createLocalServer({ envDir });
const cases = [];
let browser;
const settle = page => page.waitForFunction(() => document.querySelector('[data-av-verdict-title]')?.textContent === 'Verified' && document.querySelector('#verification-demo')?.getAttribute('aria-busy') === 'false');
async function states(page) { return page.locator('[data-av-status]').allTextContents(); }
async function run() {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true, executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined });
  for (const width of [1448, 1024, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: width === 1448 ? 1086 : 900 } });
    const errors = [];
    const external = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    page.on('request', request => { if (!request.url().startsWith(base)) external.push(request.url()); });
    try {
      const response = await page.goto(`${base}/demos/ad-verification-demo`);
      assert.equal(response.status(), 200);
      await settle(page);
      assert.equal(await page.title(), 'Ad Verification Demo | Daniel Short');
      assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
      assert.deepEqual(await states(page), ['Verified', 'Verified', 'Verified', 'Verified']);
      assert(await page.locator('.av-landscape').evaluate(image => image.complete && image.naturalWidth > 0));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await page.screenshot({ path: path.join(artifactDir, `verified-${width}.png`), fullPage: true });
      const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      assert.deepEqual(accessibility.violations, [], `Accessibility violations at ${width}: ${JSON.stringify(accessibility.violations)}`);
      await page.getByRole('button', { name: 'Simulate change', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('[data-av-verdict-title]').textContent === 'Mismatch');
      assert.deepEqual(await states(page), ['Verified', 'Verified', 'Changed', 'Untrusted']);
      assert.match(await page.locator('[data-av-change-note]').innerText(), /12,500 → 13,000/);
      await page.screenshot({ path: path.join(artifactDir, `tampered-${width}.png`), fullPage: true });
      const publisher = page.getByRole('button', { name: 'Inspect publisher record', exact: true });
      await publisher.click();
      assert(await page.locator('dialog').isVisible());
      assert.match(await page.locator('[data-av-record-description]').innerText(), /13,000/);
      assert.notEqual(await page.locator('[data-av-record-hash]').innerText(), await page.locator('[data-av-computed-hash]').innerText());
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('dialog').isVisible(), false);
      assert(await publisher.evaluate(element => document.activeElement === element));
      await page.getByRole('button', { name: 'Reset demo', exact: true }).click();
      await settle(page);
      assert.deepEqual(await states(page), ['Verified', 'Verified', 'Verified', 'Verified']);
      await page.getByRole('button', { name: 'Launch Demo', exact: true }).click();
      await settle(page);
      assert.deepEqual(await states(page), ['Verified', 'Verified', 'Verified', 'Verified']);
      assert.equal(await page.locator('[data-av-change-note]').isVisible(), false);
      assert.deepEqual(external, [], 'The demo must not contact external services.');
      assert.deepEqual(errors, [], 'No browser runtime, CSP, or asset errors.');
      cases.push({ width, status: 'pass', checks: ['page identity', 'noindex', 'real verification', 'tamper', 'inspect', 'Escape and focus return', 'reset', 'replay', 'no overflow', 'axe', 'no external requests', 'no console errors'] });
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `failure-${width}.png`), fullPage: true }).catch(() => {});
      cases.push({ width, status: 'fail', error: String(error), errors });
      throw error;
    } finally { await page.close(); }
  }
  const reduced = await browser.newPage({ reducedMotion: 'reduce', viewport: { width: 390, height: 844 } });
  await reduced.goto(`${base}/demos/ad-verification-demo`);
  await settle(reduced);
  await reduced.getByRole('button', { name: 'Launch Demo', exact: true }).click();
  await settle(reduced);
  cases.push({ reducedMotion: 'pass' });
  await reduced.close();
  const unavailable = await browser.newPage();
  await unavailable.addInitScript(() => { Object.defineProperty(window, 'crypto', { value: {}, configurable: true }); });
  await unavailable.goto(`${base}/demos/ad-verification-demo`);
  await unavailable.waitForFunction(() => document.querySelector('[data-av-verdict-title]').textContent === 'Unavailable');
  assert.deepEqual(await states(unavailable), ['Unavailable', 'Unavailable', 'Unavailable', 'Unavailable']);
  assert(await unavailable.getByRole('button', { name: 'Simulate change', exact: true }).isDisabled());
  assert(await unavailable.getByRole('button', { name: 'Try again', exact: true }).isEnabled());
  cases.push({ cryptoUnavailable: 'fails closed' });
  await unavailable.close();
  // Confirm the direct .html alias retains noindex as well.
  const alias = await browser.newPage();
  await alias.goto(`${base}/demos/ad-verification-demo.html`);
  await settle(alias);
  assert.match(await alias.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
  await alias.close();
}
run().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  fs.writeFileSync(path.join(artifactDir, 'results.json'), JSON.stringify(cases, null, 2));
  await browser?.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  fs.rmSync(envDir, { recursive: true, force: true });
  console.log(JSON.stringify(cases, null, 2));
});
