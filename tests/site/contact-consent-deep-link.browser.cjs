/** First-visit Contact deep links must leave consent usable, then open one dialog. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests } = require('../release/fixtures.cjs');

async function runContactConsentDeepLinkChecks({ browser, base, artifactDir, browserName = 'browser' }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  const cases = [1440, 390].flatMap(width => ['accept', 'reject', ...(width === 390 ? ['preferences'] : [])]
    .map(choice => ({ width, choice, route: '/contact' })));
  cases.push({ width: 390, choice: 'reject', route: '/tools/text-compare' });
  for (const { width, choice, route } of cases) {
    const label = `contact-consent-${browserName}-${width}-${choice}${route === '/contact' ? '' : '-tool-route'}`;
    const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(context, base);
    const page = await context.newPage();
    await page.clock.install();
    const errors = [];
    let submissions = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (new URL(request.url()).pathname === '/api/contact') submissions += 1; });
    try {
      await page.goto(base + route + '#contact-modal', { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.consentAPI);
      await page.locator('#pcz-banner').waitFor({ state: 'visible' });
      // Advance beyond the old 120ms opener to make the collision deterministic.
      await page.clock.runFor(600);
      const assertConsentUsable = async () => {
        assert.equal(await page.locator('#contact-modal.active').count(), 0, `${label}: automatic Contact waits for first-visit consent UI.`);
        const choices = await page.locator('#pcz-accept, #pcz-reject').evaluateAll(buttons => buttons.map(button => {
          const rect = button.getBoundingClientRect();
          const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
          return { inert: Boolean(button.closest('[inert]')), hit: hit === button || button.contains(hit) };
        }));
        assert(choices.length === 2 && choices.every(button => !button.inert && button.hit), `${label}: both real consent choices remain unobscured and interactive.`);
      };
      await assertConsentUsable();
      if (choice === 'reject') {
        await page.locator('#pcz-manage').click();
        await page.locator('#pcz-modal.pcz-visible').waitFor();
        await page.clock.runFor(500);
        assert.equal(await page.locator('#contact-modal.active').count(), 0, `${label}: preferences keep exclusive dialog focus.`);
        assert(await page.locator('#pcz-modal').evaluate(modal => modal.contains(document.activeElement)), `${label}: focus remains in preferences.`);
        await page.keyboard.press('Escape');
        await page.locator('#pcz-modal').waitFor({ state: 'detached' });
        await page.clock.runFor(500);
        await assertConsentUsable();
      }
      if (choice === 'preferences') {
        await page.locator('#pcz-manage').click();
        await page.locator('#pcz-modal.pcz-visible').waitFor();
        await page.locator('#pcz-save').click();
      } else {
        await page.locator('#pcz-' + choice).click();
      }
      await page.locator('#pcz-banner').waitFor({ state: 'detached' });
      await page.locator('#pcz-modal').waitFor({ state: 'detached' });
      await page.locator('#contact-modal.active').waitFor();
      await page.waitForFunction(() => document.querySelector('#contact-modal .modal-content').contains(document.activeElement));
      const saved = await page.evaluate(() => consentAPI.get());
      assert.equal(saved.analytics, choice === 'accept', `${label}: the actual choice persists accurately.`);
      for (let step = 0; step < 12; step += 1) {
        await page.keyboard.press('Tab');
        if (await page.locator('#contact-name').evaluate(input => document.activeElement === input)) break;
      }
      await expect(page.locator('#contact-name')).toBeFocused();
      await page.keyboard.type('Deep link usable');
      await expect(page.locator('#contact-name')).toHaveValue('Deep link usable');
      await page.screenshot({ path: path.join(artifactDir, `${label}-form.png`) });
      await page.keyboard.press('Escape');
      await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
      if (route === '/contact') await expect(page.locator('#contact-form-toggle')).toBeFocused();
      // Closing preferences later must not reopen a Contact dialog already
      // dismissed by the user, even though the old fragment remains in URL.
      await page.evaluate(() => consentAPI.open());
      await page.locator('#pcz-modal.pcz-visible').waitFor();
      await page.keyboard.press('Escape');
      await page.locator('#pcz-modal').waitFor({ state: 'detached' });
      await page.clock.runFor(500);
      assert.equal(await page.locator('#contact-modal.active').count(), 0, `${label}: later privacy changes do not reopen dismissed Contact.`);
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.locator('#contact-modal.active').waitFor();
      assert.equal(await page.locator('#pcz-banner').count(), 0, `${label}: saved consent opens the deep link without another banner.`);
      await expect(page.locator('#contact-name')).toHaveValue('Deep link usable');
      assert(await page.locator('#contact-modal .modal-content').evaluate(dialog => !dialog.closest('[inert]')), `${label}: restored Contact remains interactive.`);
      assert.equal(submissions, 0, `${label}: no message is sent.`);
      assert.deepEqual(errors, [], `${label}: no uncaught errors.`);
      results.push({ label, saved });
      console.log(`Contact deep-link consent passed: ${label}`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
      throw error;
    } finally { await context.close(); }
  }
  fs.writeFileSync(path.join(artifactDir, `contact-consent-${browserName}.json`), JSON.stringify(results, null, 2));
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'contact-consent-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runContactConsentDeepLinkChecks({ browser, browserName, base,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-contact-consent') });
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runContactConsentDeepLinkChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
