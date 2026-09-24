'use strict';
// Exercise the configured response headers in a real browser, without disabling CSP.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { createLocalServer } = require('../../build/dev');
const { inlineScripts } = require('../../build/csp-hashes.cjs');
async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'csp-test-'));
  const server = createLocalServer({ envDir });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined,
      args: ['--no-sandbox', '--disable-dev-shm-usage'] });
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block' });
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.origin !== base) return route.abort();
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
      return route.continue();
    });
    await context.routeWebSocket('**/*', route => route.close());
    await context.addInitScript(() => {
      window.__cspViolations = [];
      addEventListener('securitypolicyviolation', event => {
        if (event.disposition === 'enforce') window.__cspViolations.push({ directive: event.effectiveDirective, blocked: event.blockedURI });
      });
    });
    const page = await context.newPage();
    for (const route of ['/', '/tools/text-compare', '/portfolio/website', '/minesweeper-demo']) {
      const response = await page.goto(base + route, { waitUntil: 'load' });
      assert.equal(response.status(), 200, route);
      assert.ok(response.headers()['content-security-policy'], `${route}: real CSP response header is required`);
      assert.equal(await page.locator('html').evaluate(node => node.classList.contains('js')), true, route);
      const bad = await page.evaluate(() => window.__cspViolations.filter(v => v.directive.startsWith('script-src')));
      assert.deepEqual(bad, [], `${route}: legitimate scripts blocked`);
    }
    const demos = fs.readdirSync(path.resolve('public/demos')).filter(file => file.endsWith('.html')
      && inlineScripts(fs.readFileSync(path.resolve('public/demos', file), 'utf8')).length);
    assert.ok(demos.length >= 12);
    for (const file of demos) {
      const loaded = page.waitForEvent('framenavigated', { predicate: frame => frame.url().includes(`/demos/${file}`) });
      await page.evaluate(src => {
        document.querySelector('#csp-test-frame')?.remove();
        const frame = document.createElement('iframe'); frame.id = 'csp-test-frame'; frame.src = src;
        frame.title = 'CSP regression fixture'; document.body.appendChild(frame);
      }, `${base}/demos/${file}`);
      const frame = await loaded;
      await frame.waitForLoadState('load');
      const bad = await frame.evaluate(() => window.__cspViolations.filter(v => v.directive.startsWith('script-src')));
      assert.deepEqual(bad, [], `${file}: legitimate inline demo script blocked`);
      assert.equal(await frame.evaluate(() => document.documentElement.dataset.embedded), 'true', `${file}: approved inline bootstrap ran`);
    }
    // DevTools evaluation itself bypasses CSP. A DOM-inserted script does not.
    await page.evaluate(() => {
      const script = document.createElement('script');
      script.textContent = 'window.__arbitraryInlineRan = true'; document.head.appendChild(script);
    });
    await page.waitForFunction(() => window.__cspViolations.some(v => v.blocked === 'inline'));
    assert.equal(await page.evaluate(() => window.__arbitraryInlineRan), undefined);
    console.log(`CSP browser checks passed: 4 visitor routes, ${demos.length} raw demos, arbitrary inline script blocked.`);
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(envDir, { recursive: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
