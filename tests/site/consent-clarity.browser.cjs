/** Consent choices, readable purposes, modal reachability and real browser gating.
 * Google transport is a deterministic fixture; no live tracking is contacted.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const { createLocalServer } = require('../../build/dev');
const { isolateRequests, settle } = require('../release/fixtures.cjs');

const none = { necessary: true, analytics: false, functional: false, advertising: false };
const cases = [
  { mode: 'reject', width: 1440, height: 1000 },
  { mode: 'reject', width: 390, height: 844 },
  { mode: 'accept', width: 1440, height: 1000 },
  { mode: 'accept', width: 390, height: 844 },
  { mode: 'selective', width: 1440, height: 1000 },
  { mode: 'selective', width: 390, height: 844 },
  { mode: 'optional-only', width: 320, height: 568 },
  { mode: 'enlarged', width: 320, height: 844, textScale: 200 },
  { mode: 'short-modal', width: 390, height: 260 },
  { mode: 'privacy-page', width: 390, height: 844 },
  { mode: 'signals', width: 390, height: 844 }
];

async function readState(page) {
  return page.evaluate(() => ({
    consent: consentAPI.get(),
    record: JSON.parse(localStorage.getItem('pcz_consent_v1')),
    hints: document.querySelectorAll('link[id^="pcz-analytics-"]').length,
    googleMode: [...dataLayer].filter(item => item[0] === 'consent' && item[1] === 'update').at(-1)?.[2],
    projectStorage: sessionStorage.getItem('ds_analytics_project_views_v1')
  }));
}

async function checkModal(page, label) {
  await page.locator('#pcz-modal.pcz-visible').waitFor();
  await page.evaluate(() => document.fonts.ready);
  const modal = page.locator('#pcz-modal');
  assert.equal(await modal.getAttribute('aria-modal'), 'true', `${label}: modal semantics`);
  assert.equal(await modal.getAttribute('aria-labelledby'), 'pcz-modal-title', `${label}: visible modal title labels it`);
  assert.equal(await page.locator('#pcz-modal .pref-info').count(), 0, `${label}: purposes need no help-button action`);
  for (const category of ['necessary', 'analytics', 'functional', 'advertising']) {
    const description = page.locator(`#pcz-pref-desc-${category}`);
    assert.equal(await description.evaluate(el => !el.hidden && getComputedStyle(el).display !== 'none'), true,
      `${label}: ${category} purpose is displayed by default`);
    assert((await description.textContent()).trim().length > 35, `${label}: ${category} has useful inline copy`);
  }
  for (const category of ['analytics', 'functional', 'advertising']) {
    assert.equal(await modal.locator(`.pref-toggle[data-pref="${category}"]`).getAttribute('aria-describedby'),
      `pcz-pref-desc-${category}`, `${label}: purpose is associated with ${category}`);
  }
  assert.equal(await modal.locator('.pref-status-row .pref-state').textContent(), 'Always on');
  assert.equal(await modal.locator('.pref-toggle[data-pref="necessary"]').count(), 0, `${label}: necessary is not a changeable control`);
  await page.waitForFunction(() => document.activeElement?.id === 'pcz-close-modal');
  await page.keyboard.press('Shift+Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'pcz-save', `${label}: backwards Tab remains inside the modal`);
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'pcz-close-modal', `${label}: forward Tab wraps back to Close`);
  const geometry = await modal.evaluate(el => {
    const panel = el.querySelector('.pcz-panel');
    const content = el.querySelector('.pcz-panel-body');
    const close = el.querySelector('#pcz-close-modal');
    const save = el.querySelector('#pcz-save');
    const rect = node => node.getBoundingClientRect().toJSON();
    const hit = node => { const r = node.getBoundingClientRect(); return node.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)); };
    return { panel: rect(panel), close: rect(close), save: rect(save), content: rect(content),
      closeHit: hit(close), saveHit: hit(save), panelOverflow: getComputedStyle(panel).overflowY,
      contentOverflow: getComputedStyle(content).overflowY, width: innerWidth, height: innerHeight,
      documentWidth: document.documentElement.scrollWidth };
  });
  assert(geometry.panel.top >= 15 && geometry.panel.bottom <= geometry.height - 15,
    `${label}: comfortable viewport clearance ${JSON.stringify(geometry)}`);
  assert(geometry.panel.left >= 15 && geometry.panel.right <= geometry.width - 15,
    `${label}: panel fits narrow viewport ${JSON.stringify(geometry)}`);
  assert(geometry.close.width >= 44 && geometry.close.height >= 44 && geometry.closeHit && geometry.saveHit,
    `${label}: Close and Save remain reachable ${JSON.stringify(geometry)}`);
  assert(geometry.content.height > 0 && geometry.content.top >= geometry.close.bottom - 1 &&
    geometry.content.bottom <= geometry.save.top && geometry.panelOverflow === 'hidden' && geometry.contentOverflow === 'auto',
  `${label}: only the category content scrolls ${JSON.stringify(geometry)}`);
  assert(geometry.documentWidth <= geometry.width + 1, `${label}: no horizontal overflow`);
  const necessaryWordLines = await modal.locator('.pref-status-row .pref-label').evaluate(el => {
    const text = el.firstChild;
    const start = text.textContent.indexOf('necessary');
    const word = document.createRange();
    word.setStart(text, start);
    word.setEnd(text, start + 'necessary'.length);
    return word.getClientRects().length;
  });
  assert.equal(necessaryWordLines, 1, `${label}: Necessary remains readable without breaking a word into tiny fragments`);
  const axe = await new AxeBuilder({ page }).include('#pcz-modal').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  assert.deepEqual(axe.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), [], `${label}: modal accessibility`);
  return geometry;
}

async function runConsentClarityChecks({ browser, base, artifactDir, browserName = 'browser' }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  for (const entry of cases) {
    const label = `consent-${browserName}-${entry.mode}-${entry.width}x${entry.height}`;
    const context = await browser.newContext({ viewport: { width: entry.width, height: entry.height },
      serviceWorkers: 'block', reducedMotion: 'reduce', locale: 'en-US' });
    await isolateRequests(context, base);
    const fixtureTransport = base + '/api/_consent-fixture-analytics';
    const hits = [];
    let gtmRequests = 0;
    // Web Vitals also reports during pagehide. Keep the synthetic transport
    // alive for those events, as a beacon-based analytics transport would.
    await context.route('https://www.googletagmanager.com/gtm.js**', async route => {
      gtmRequests += 1;
      await route.fulfill({ contentType: 'application/javascript', body:
        `(()=>{const pending=window.__consentTransportPending=[];const prior=dataLayer.push.bind(dataLayer);dataLayer.push=function(item){if(item.event==='web_vital')pending.push(fetch(${JSON.stringify(fixtureTransport)},{method:'POST',keepalive:true,body:JSON.stringify(item)}).then(response=>response.text()));return prior(item)};window.__consentGtmFixture=true})()` });
    });
    await context.route(fixtureTransport, async route => {
      hits.push(JSON.parse(route.request().postData()));
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"accepted":true}' });
    });
    if (entry.mode === 'signals') await context.addInitScript(() => {
      Object.defineProperty(navigator, 'doNotTrack', { configurable: true, value: '1' });
      Object.defineProperty(navigator, 'globalPrivacyControl', { configurable: true, value: true });
    });
    if (entry.textScale) await context.addInitScript(scale => {
      document.addEventListener('DOMContentLoaded', () => document.documentElement.style.setProperty('font-size', scale + '%', 'important'));
    }, entry.textScale);
    const page = await context.newPage();
    const errors = [];
    const diagnostics = [];
    page.on('pageerror', error => {
      errors.push(error.message);
      diagnostics.push({ type: 'pageerror', message: error.message, stack: error.stack, url: page.url() });
    });
    page.on('requestfailed', request => diagnostics.push({ type: 'requestfailed', url: request.url(), failure: request.failure() }));
    const finishFixtureTransport = () => page.evaluate(() => Promise.allSettled(window.__consentTransportPending || []));
    const stages = [];
    try {
      // Home normalizes its URL before analytics initializes. Use the same
      // local QA entry route as the existing analytics-hints/transport checks.
      const response = await page.goto(`${base}/tools/text-compare?analytics_debug=1`, { waitUntil: 'domcontentloaded' });
      assert.equal(response.status(), 200);
      await settle(page);
      await page.locator('#pcz-banner.pcz-visible').waitFor();
      await page.waitForFunction(() => window.consentAPI && window.sendWebVital && window.trackProjectView);
      assert((await page.title()).includes('Daniel Short'), `${label}: page identity`);
      assert.equal(new URL(page.url()).pathname, '/tools/text-compare', `${label}: correct QA entry route`);
      assert((await page.locator('main:visible').textContent()).trim().length > 40, `${label}: meaningful content`);
      assert.equal(await page.locator('#pcz-reject').textContent(), 'Reject optional');
      assert.equal(await page.locator('#pcz-accept').textContent(), 'Accept optional');
      assert.equal(await page.locator('#pcz-manage').textContent(), 'Manage preferences');
      const choices = await page.locator('#pcz-banner .pcz-btn').evaluateAll(nodes => nodes.map(el => {
        const c = getComputedStyle(el);
        return { color: c.color, background: c.backgroundColor, border: c.borderColor, weight: c.fontWeight,
          height: el.getBoundingClientRect().height };
      }));
      assert.deepEqual(choices[0], choices[1], `${label}: Accept and Reject have equal visual weight`);
      assert(choices[0].height >= 44, `${label}: both choices have comfortable targets`);
      assert.equal(gtmRequests, 0, `${label}: no GTM before choice`);
      assert.equal((await readState(page)).hints, 0, `${label}: no Google connection hints before choice`);
      assert.equal(await page.evaluate(() => trackProjectView('website')), false);
      assert.equal((await readState(page)).projectStorage, null, `${label}: no analytics session storage before consent`);
      if (entry.height > 500) await page.screenshot({ path: path.join(artifactDir, `${label}-banner.png`) });
      let expected;
      if (entry.mode === 'accept' || entry.mode === 'signals') {
        await page.locator('#pcz-accept').click();
        expected = entry.mode === 'signals' ? { ...none, functional: true } :
          { necessary: true, analytics: true, functional: true, advertising: true };
      } else if (entry.mode === 'reject' || entry.mode === 'privacy-page') {
        await page.locator('#pcz-reject').click();
        expected = { ...none };
      } else {
        await page.locator('#pcz-manage').click();
        stages.push({ stage: 'fresh preferences', geometry: await checkModal(page, label) });
        assert.equal(await page.locator('#pcz-modal .pref-toggle[aria-pressed="true"]').count(), 0,
          `${label}: optional categories start off`);
        await page.screenshot({ path: path.join(artifactDir, `${label}-preferences.png`) });
        if (entry.mode === 'optional-only') {
          for (const category of ['functional', 'advertising']) await page.locator(`#pcz-modal .pref-toggle[data-pref="${category}"]`).click();
          expected = { ...none, functional: true, advertising: true };
        } else {
          await page.locator('#pcz-modal .pref-toggle[data-pref="analytics"]').click();
          expected = { ...none, analytics: true };
        }
        await page.locator('#pcz-save').click();
      }
      await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
      await page.locator('#pcz-modal').waitFor({ state: 'hidden' });
      let state = await readState(page);
      assert.deepEqual(state.consent, expected, `${label}: choice persisted accurately`);
      assert.deepEqual(state.record.categories, expected, `${label}: stored categories match effective choice`);
      assert.equal(state.googleMode.analytics_storage, expected.analytics ? 'granted' : 'denied');
      assert.equal(state.googleMode.ad_storage, expected.advertising ? 'granted' : 'denied');
      assert.equal(state.googleMode.ad_personalization, expected.advertising ? 'granted' : 'denied');
      if (expected.analytics) await page.waitForFunction(() => window.__consentGtmFixture);
      assert.equal(gtmRequests, expected.analytics ? 1 : 0, `${label}: only Analytics loads GTM`);
      assert.equal(await page.evaluate(() => trackProjectView('website')), expected.analytics);
      assert.equal(Boolean((await readState(page)).projectStorage), expected.analytics,
        `${label}: analytics session storage follows its consent`);
      const report = () => page.evaluate(() => sendWebVital({ name: 'LCP', value: 1000, rating: 'good' }));
      if (expected.analytics) {
        await Promise.all([page.waitForResponse(r => r.url() === fixtureTransport), report()]);
      } else await report();
      await finishFixtureTransport();
      assert.equal(hits.length, expected.analytics ? 1 : 0, `${label}: first-party event transport follows Analytics`);
      stages.push({ stage: 'saved choice', state: await readState(page), gtmRequests, hits: hits.length });
      await page.reload({ waitUntil: 'domcontentloaded' });
      await settle(page);
      await page.waitForFunction(() => window.consentAPI && window.trackProjectView);
      assert.equal(await page.locator('#pcz-banner').count(), 0, `${label}: reload remembers choice`);
      assert.deepEqual((await readState(page)).consent, expected, `${label}: reload restores effective categories`);
      await page.locator('#privacy-settings-link-footer').click();
      stages.push({ stage: 'reopened preferences', geometry: await checkModal(page, label) });
      for (const category of ['analytics', 'functional', 'advertising']) {
        assert.equal(await page.locator(`#pcz-modal .pref-toggle[data-pref="${category}"]`).getAttribute('aria-pressed'),
          String(expected[category]), `${label}: reopened ${category} matches saved choice`);
      }
      await page.keyboard.press('Escape');
      await page.locator('#pcz-modal').waitFor({ state: 'hidden' });
      assert.equal(await page.evaluate(() => document.activeElement.id), 'privacy-settings-link-footer', `${label}: close restores footer focus`);
      if (entry.mode === 'privacy-page') {
        await page.goto(`${base}/privacy?analytics_debug=1`, { waitUntil: 'domcontentloaded' });
        await settle(page);
        for (const category of ['necessary', 'analytics', 'functional', 'advertising']) {
          assert.equal(await page.locator(`#pref-desc-${category}`).evaluate(el => !el.hidden), true, `${label}: page purpose is inline`);
        }
        await page.locator('#privacy-preferences-form [data-pref="functional"].pref-toggle').click();
        await page.locator('#save-privacy-preferences').click();
        assert.deepEqual((await readState(page)).consent, { ...none, functional: true });
        assert.equal(gtmRequests, 0, `${label}: functional-only policy form does not load a vendor`);
        await page.screenshot({ path: path.join(artifactDir, `${label}-page.png`) });
      }
      if (expected.analytics) {
        await page.locator('#privacy-settings-link-footer').click();
        await page.locator('#pcz-modal .pref-toggle[data-pref="analytics"]').click();
        await page.locator('#pcz-save').click();
        await page.locator('#pcz-modal').waitFor({ state: 'hidden' });
        state = await readState(page);
        assert.equal(state.hints, 0, `${label}: withdrawal removes connection hints`);
        assert.equal(state.projectStorage, null, `${label}: withdrawal removes first-party analytics session storage`);
        assert.equal(await page.evaluate(() => trackProjectView('website')), false);
        const count = hits.length;
        await report();
        await finishFixtureTransport();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.equal(hits.length, count, `${label}: withdrawal suppresses transport even with the fixture script still loaded`);
      }
      assert.deepEqual(errors, [], `${label}: no uncaught first-party errors`);
      results.push({ ...entry, label, stages, gtmRequests, interceptedAnalyticsHits: hits.length });
      console.log(`Consent clarity passed: ${label}`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
      fs.writeFileSync(path.join(artifactDir, `${label}-failure.json`), JSON.stringify({ label, stages, errors, diagnostics, error: error.message }, null, 2));
      throw error;
    } finally { await context.close(); }
  }
  fs.writeFileSync(path.join(artifactDir, `consent-clarity-${browserName}.json`), JSON.stringify(results, null, 2));
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'consent-clarity-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runConsentClarityChecks({ browser, browserName, base,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-consent-clarity') });
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runConsentClarityChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
