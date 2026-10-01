/** Checks initial HTML and real consent transitions without contacting analytics APIs. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests } = require('../release/fixtures.cjs');

async function getHints(page) {
  return page.evaluate(() => [...document.querySelectorAll('link[rel="preconnect"], link[rel="dns-prefetch"]')]
    .filter(link => /^(?:www\.)?(?:googletagmanager|google-analytics)\.com$/.test(new URL(link.href).hostname))
    .map(link => ({ id: link.id, rel: link.rel, origin: new URL(link.href).origin })));
}

async function runAnalyticsConnectionHintChecks({ browser, base, artifactDir, browserName = 'browser' }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  for (const mode of ['new-choice', 'saved-allow', 'dnt', 'preview-disabled']) {
    const context = await browser.newContext({ serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(context, base);
    let gtmRequests = 0;
    await context.route('https://www.googletagmanager.com/gtm.js**', async route => {
      gtmRequests += 1;
      await route.fulfill({ contentType: 'application/javascript', body: 'window.__hintGtmFixture = true;' });
    });
    if (mode === 'saved-allow' || mode === 'dnt') await context.addInitScript(mode => {
      localStorage.setItem('pcz_consent_v1', JSON.stringify({ version: 1, timestamp: Date.now(), categories: {
        necessary: true, analytics: true, functional: false, advertising: false
      } }));
      if (mode === 'dnt') Object.defineProperty(navigator, 'doNotTrack', { configurable: true, value: '1' });
    }, mode);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
      const route = base + '/tools/text-compare' + (mode === 'preview-disabled' ? '' : '?analytics_debug=1');
      const response = await page.goto(route, { waitUntil: 'domcontentloaded' });
      const html = await response.text();
      const initialLinks = html.match(/<link\b[^>]*>/gi) || [];
      assert(!initialLinks.some(tag => /\brel\s*=\s*["'](?:preconnect|dns-prefetch)["']/i.test(tag) &&
        /\bhref\s*=\s*["'](?:https?:)?\/\/(?:www\.)?(?:googletagmanager|google-analytics)\.com\/?["']/i.test(tag)),
      `${mode}: document bytes contain no Google connection hints before the CMP runs.`);
      await page.waitForFunction(() => window.consentAPI && window.SiteAnalyticsEnvironment && document.readyState !== 'loading');
      const stages = [];
      const assertAbsent = async stage => {
        const hints = await getHints(page);
        assert.deepEqual(hints, [], `${mode}/${stage}: no analytics connection hints.`);
        stages.push({ stage, hints });
      };
      const assertPresent = async stage => {
        await page.waitForFunction(() => document.querySelectorAll('link[id^="pcz-analytics-"]').length === 4 && window.__hintGtmFixture);
        const hints = await getHints(page);
        assert.equal(hints.length, 4, `${mode}/${stage}: consent allows exactly four connection hints.`);
        assert.equal(new Set(hints.map(hint => hint.origin + '/' + hint.rel)).size, 4, `${mode}/${stage}: hints have no duplicate origin/rel pairs.`);
        stages.push({ stage, hints });
      };
      if (mode === 'saved-allow') {
        await assertPresent('saved opt-in');
        assert.equal(gtmRequests, 1, 'Saved opt-in starts GTM once.');
      } else {
        await assertAbsent('initial');
        assert.equal(gtmRequests, 0, `${mode}: GTM is not requested on initial denied/disabled state.`);
      }
      if (mode === 'new-choice') {
        await page.locator('#pcz-reject').click();
        await assertAbsent('reject');
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => window.consentAPI && window.SiteAnalyticsEnvironment);
        await assertAbsent('saved reject');
        assert.equal(gtmRequests, 0, 'Rejecting and reloading never requests GTM.');
        await page.evaluate(() => consentAPI.reset());
        await page.locator('#pcz-accept').click();
        await assertPresent('allow');
        assert.equal(gtmRequests, 1, 'Allowing requests GTM once.');
        await page.evaluate(() => consentAPI.set({ analytics: true }));
        await assertPresent('repeated allow');
        assert.equal(gtmRequests, 1, 'Repeated consent does not reload GTM.');
        await page.evaluate(() => consentAPI.set({ analytics: false }));
        await assertAbsent('revoke');
        await page.evaluate(() => consentAPI.set({ analytics: true }));
        await assertPresent('allow again');
        assert.equal(gtmRequests, 1, 'Allowing again restores hints without duplicating the loaded vendor.');
      } else if (mode !== 'saved-allow') {
        await page.evaluate(() => consentAPI.set({ analytics: true }));
        await assertAbsent('allow blocked by environment/privacy signal');
        assert.equal(gtmRequests, 0, `${mode}: environment/DNT gates also suppress hints after attempted opt-in.`);
      }
      assert.deepEqual(errors, [], `${mode}: no uncaught browser errors.`);
      results.push({ mode, stages, gtmRequests });
      console.log(`Analytics hint consent passed: ${browserName}/${mode}`);
    } finally { await context.close(); }
  }
  fs.writeFileSync(path.join(artifactDir, `analytics-hints-${browserName}.json`), JSON.stringify(results, null, 2));
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'analytics-hints-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runAnalyticsConnectionHintChecks({ browser, browserName, base,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-analytics-hints') });
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runAnalyticsConnectionHintChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
