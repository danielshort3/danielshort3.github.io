'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');

async function main() {
  const envDir = fs.mkdtempSync(os.tmpdir() + '/ds-recovery-env-');
  const server = createLocalServer({ envDir });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const context = await browser.newContext({ reducedMotion: 'reduce', serviceWorkers: 'block' });
  await context.route('**/api/**', route => route.fulfill({ status: 200, json: { ok: true, sessions: [], activity: [] } }));
  await context.route('https://**', route => route.abort());
  const page = await context.newPage();
  const key = 'ds:session-draft:v1:tools:text-compare';
  const read = () => page.evaluate(key => sessionStorage.getItem(key), key);
  const ready = async () => {
    await page.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
  };
  try {
    // The form is usable before the shared account bootstrap finishes loading.
    // That already-present text must be recovered without requiring another edit.
    const earlyContext = await browser.newContext({ reducedMotion: 'reduce', serviceWorkers: 'block' });
    let releaseBootstrap;
    const bootstrapGate = new Promise(resolve => { releaseBootstrap = resolve; });
    await earlyContext.route('**/api/**', route => route.fulfill({ status: 200, json: { ok: true, sessions: [], activity: [] } }));
    await earlyContext.route('https://**', route => route.abort());
    await earlyContext.route('**/dist/site-tools-account.*.js', async route => {
      await bootstrapGate;
      await route.continue();
    });
    const earlyPage = await earlyContext.newPage();
    try {
      await earlyPage.goto(base + '/tools/text-compare', { waitUntil: 'commit' });
      await earlyPage.locator('#textcompare-original').fill('Typed before account bootstrap');
      assert.notEqual(await earlyPage.locator('#main').getAttribute('data-tools-draft-owner'), 'guest');
      releaseBootstrap();
      await earlyPage.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
      await earlyPage.goto(base + '/tools/word-frequency');
      assert.match(await earlyPage.evaluate(key => sessionStorage.getItem(key), key), /Typed before account bootstrap/);
      await earlyPage.goto(base + '/tools/text-compare');
      await earlyPage.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
      await expect(earlyPage.locator('#textcompare-original')).toHaveValue('Typed before account bootstrap');
      console.log('Guest input entered before account bootstrap survives leaving and returning without another edit.');
    } finally {
      releaseBootstrap();
      await earlyContext.close();
    }
    await page.goto(base + '/tools/text-compare');
    await ready();
    await page.locator('#textcompare-original').fill('A private draft');
    await expect.poll(read).toContain('A private draft');
    await page.reload();
    await ready();
    await expect(page.locator('#textcompare-original')).toHaveValue('A private draft');
    await expect(page.locator('.draft-recovery-notice')).toBeVisible();
    await page.locator('.draft-recovery-notice button').click();
    await expect(page.locator('#textcompare-original')).toHaveValue('');
    assert.equal(await read(), null);
    await page.locator('#textcompare-original').fill('Navigation draft');
    await page.goto(base + '/tools/word-frequency');
    await page.goto(base + '/tools/text-compare');
    await ready();
    await expect(page.locator('#textcompare-original')).toHaveValue('Navigation draft');
    await page.locator('details').filter({ has: page.locator('#textcompare-clear') }).locator('summary').first().click();
    await page.locator('#textcompare-clear').click();
    await page.reload(); await ready();
    await expect(page.locator('#textcompare-original')).toHaveValue('');
    assert.equal(await read(), null, 'Clear must not resurrect');
    await page.evaluate(key => sessionStorage.setItem(key, JSON.stringify({ updated: Date.now() - 7200001, data: { 'textcompare-original': { kind: 'value', value: 'expired' } } })), key);
    await page.reload(); await ready();
    await expect(page.locator('#textcompare-original')).toHaveValue('');
    await page.locator('#textcompare-original').fill('guest-only');
    await expect.poll(read).toContain('guest-only');
    await page.evaluate(() => {
      window.ToolsAuth = { ...ToolsAuth, getAuth: () => ({}), authIsValid: () => true, getUser: () => ({ sub: 'mock-account' }) };
      document.dispatchEvent(new CustomEvent('tools:auth-changed'));
    });
    await page.waitForLoadState('domcontentloaded'); await ready();
    await expect(page.locator('#textcompare-original')).toHaveValue('');
    assert.equal(await read(), null, 'Guest data cannot enter an account');
    await page.evaluate(() => {
      document.querySelector('#textcompare-original').value = 'x'.repeat(1100000);
      document.querySelector('#textcompare-original').dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForTimeout(650);
    assert.equal(await read(), null, 'Oversized draft is omitted');
    await context.addInitScript(() => Object.defineProperty(window, 'sessionStorage', { get() { throw new DOMException('Denied', 'SecurityError'); } }));
    await page.reload(); await ready();
    await page.locator('#textcompare-original').fill('Still usable');
    await page.locator('#textcompare-revised').fill('Still usable with blocked storage');
    await page.getByRole('button', { name: 'Compare', exact: true }).click();
    await expect(page.locator('#textcompare-output')).toContainText('blocked storage');
    console.log('Guest refresh/navigation/discard/clear/expiry/denied/oversize/account isolation passed.');

    await require('./media-tool-draft-clear.browser.cjs')(browser, base);

    const analyticsContext = await browser.newContext({ serviceWorkers: 'block', reducedMotion: 'reduce' });
    const analyticsPage = await analyticsContext.newPage();
    const hits = [];
    analyticsPage.on('console', message => { if (message.type() === 'error') console.log('Analytics fixture:', message.text()); });
    await analyticsContext.route(/^https:\/\//, async route => {
      const url = route.request().url();
      if (url.includes('googletagmanager.com/gtm.js')) {
        // Deterministic GTM transport fixture; real export routing is tested separately.
        await route.fulfill({ contentType: 'application/javascript', body: `(()=>{const push=dataLayer.push.bind(dataLayer);dataLayer.push=function(item){if(item.event==='web_vital')fetch('https://www.google-analytics.com/g/collect',{method:'POST',body:JSON.stringify(item)}).catch(()=>{});return push(item)};window.__gtmFixtureReady=true})()` });
      } else if (url.includes('google-analytics.com/g/collect')) {
        hits.push(JSON.parse(route.request().postData()));
        await route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*' }, body: '' });
      } else await route.abort();
    });
    await analyticsPage.goto(base + '/tools/text-compare?analytics_debug=1&secret=PRIVATE');
    await analyticsPage.waitForFunction(() => typeof window.sendWebVital === 'function' && window.consentAPI);
    const report = () => analyticsPage.evaluate(() => sendWebVital({ name: 'LCP', value: 1234.123456, rating: 'good', account: 'PRIVATE', entries: [{ text: 'PRIVATE' }] }));
    await report(); assert.equal(hits.length, 0);
    await analyticsPage.evaluate(() => consentAPI.set({ analytics: true }));
    await analyticsPage.waitForFunction(() => document.querySelector('script[src*="googletagmanager.com/gtm.js"]'));
    await analyticsPage.waitForFunction(() => window.__gtmFixtureReady);
    await analyticsPage.evaluate(() => history.replaceState({}, '', '/#contact?PRIVATE'));
    await report();
    await expect.poll(() => hits.length).toBeGreaterThan(0);
    const hit = hits.at(-1);
    assert.equal(hit.page_path, '/tools/text-compare');
    assert.equal(hit.page_id, 'text-compare');
    assert.equal(hit.metric_value, 1234.1235);
    assert(!JSON.stringify(hit).includes('PRIVATE'));
    await analyticsPage.evaluate(() => consentAPI.set({ analytics: false }));
    const count = hits.length; await report();
    await analyticsPage.waitForTimeout(100);
    assert.equal(hits.length, count, 'Withdrawn consent must block transport');
    await analyticsContext.close();
    console.log('Intercepted Web Vital transport consent, sanitization and entry-route attribution passed.');
  } finally {
    await context.close(); await browser.close();
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); fs.rmdirSync(envDir);
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
