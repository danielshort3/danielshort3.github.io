'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests, settle } = require('../release/fixtures.cjs');

const categories = ['about', 'projects', 'tools', 'games', 'contact'];
const navSelector = '[data-mobile-section-nav]';
const headerSelector = '[data-mobile-site-masthead]';

async function chromeState(page, hidden) {
  await page.waitForFunction(hidden => document.body.classList.contains('is-mobile-chrome-hidden') === hidden, hidden);
  await page.waitForFunction(({ hidden, navSelector, headerSelector }) => [navSelector, headerSelector].every(selector => {
    const node = document.querySelector(selector);
    const rect = node.getBoundingClientRect();
    return hidden
      ? node.inert && (selector === headerSelector ? rect.bottom <= 0 : rect.top >= innerHeight)
      : !node.inert && rect.top >= -1 && rect.bottom <= innerHeight + 1;
  }), { hidden, navSelector, headerSelector });
}

async function scrollPage(page, y) {
  const current = await page.evaluate(() => window.scrollY);
  await page.mouse.move(2, Math.floor(page.viewportSize().height / 2));
  await page.mouse.wheel(0, y - current);
  await page.waitForFunction(y => Math.abs(scrollY - Math.min(y, Math.max(0, document.documentElement.scrollHeight - innerHeight))) <= 2, y);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function runMobileScrollChromeChecks({ browser, base, artifactDir, prepareContext }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const viewport of [{ width: 390, height: 844 }, { width: 320, height: 740 }, { width: 844, height: 390 }, { width: 1440, height: 900 }]) {
    const context = await browser.newContext({ viewport, reducedMotion: viewport.width === 320 ? 'reduce' : 'no-preference', serviceWorkers: 'block' });
    await isolateRequests(context, base);
    await prepareContext?.(context);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.setDefaultTimeout(12000);
    try {
      await page.goto(`${base}/#about`);
      await settle(page);
      if (viewport.width >= 960) {
        assert(await page.locator(navSelector).isHidden(), 'Desktop does not show a second category navigation.');
        assert.equal(await page.locator('[data-site-tab]:visible').count(), 5, 'Desktop retains all five vertical tabs.');
        await page.screenshot({ path: path.join(artifactDir, 'desktop-unchanged.png') });
        continue;
      }
      const cookie = page.locator('#pcz-banner');
      const cookieBounds = await cookie.boundingBox();
      const navBounds = await page.locator(navSelector).boundingBox();
      if (await cookie.isVisible()) {
        assert(cookieBounds.y + cookieBounds.height <= navBounds.y + 1, 'First-visit consent does not cover the category panel.');
        await page.locator('#pcz-reject').click();
        await cookie.waitFor({ state: 'hidden' });
      }
      await chromeState(page, false);
      assert.deepEqual(await page.locator(`${navSelector} a`).evaluateAll(nodes => nodes.map(node => node.dataset.mobileSection)), categories);
      assert.equal(await page.locator('[data-site-tab]:visible').count(), 5, 'Open mobile home keeps every colored tab row available around the active content.');
      for (const link of await page.locator(`${navSelector} a`).all()) {
        const box = await link.boundingBox();
        assert(box.width >= 44 && box.height >= 44, 'Each category has a usable touch target.');
      }
      await page.screenshot({ path: path.join(artifactDir, `mobile-${viewport.width}-shown.png`) });
      await scrollPage(page, 240);
      await chromeState(page, true);
      assert.equal(await page.locator(`${navSelector}[aria-hidden="true"]`).count(), 1);
      await page.locator(`${navSelector} a`).first().evaluate(node => node.focus());
      assert(!await page.locator(navSelector).evaluate(node => node.contains(document.activeElement)), 'Hidden links cannot steal focus.');
      await page.screenshot({ path: path.join(artifactDir, `mobile-${viewport.width}-hidden.png`) });
      await scrollPage(page, 215);
      await chromeState(page, false);
      await scrollPage(page, 300);
      await chromeState(page, true);
      await page.keyboard.press('Tab');
      await chromeState(page, false);
      await page.locator('[data-site-tab="about"]').focus();
      await scrollPage(page, 0);
      await chromeState(page, false);
      await scrollPage(page, 6);
      await scrollPage(page, 0);
      await chromeState(page, false);

      await page.locator(`${navSelector} [data-mobile-section="tools"]`).click();
      await page.waitForFunction(() => SiteFrame.current()?.category === 'tools' && SiteFrame.current()?.view === 'overview');
      await settle(page);
      await page.locator(`${navSelector} [data-mobile-section="tools"][aria-current="page"]`).waitFor();
      assert.equal(await page.locator(`${navSelector} [aria-current="page"]`).getAttribute('data-mobile-section'), 'tools');
      await page.locator('[data-site-tab="tools"]').click();
      await page.waitForFunction(() => SiteFrame.current()?.view === 'closed');
      await settle(page);
      await page.waitForFunction(() => !document.querySelector('[data-mobile-section-nav] [aria-current]'));
      assert.equal(await page.locator('[data-site-tab]:visible').count(), 5, 'The closed homepage retains its flush colored rows.');
      assert.equal(await page.locator(`${navSelector} [aria-current]`).count(), 0, 'Closed homepage has no active bottom category.');

      await page.goto(`${base}/tools/text-compare`);
      await settle(page);
      await chromeState(page, false);
      const before = page.locator('#textcompare-original');
      await before.fill(Array(70).fill('A longer local draft for internal scrolling.').join('\n'));
      await before.evaluate(node => { node.scrollTop = node.scrollHeight; });
      await scrollPage(page, 180);
      await chromeState(page, false);
      await page.evaluate(() => document.activeElement.blur());
      await scrollPage(page, 280);
      await chromeState(page, true);
      await page.keyboard.press('Tab');
      await chromeState(page, false);
      await page.locator('.mobile-site-masthead__explore-button').click();
      await scrollPage(page, 380);
      await chromeState(page, false);
      await page.keyboard.press('Escape');

      await page.goto(`${base}/contact`);
      await settle(page);
      await page.locator('#contact-form-toggle').click();
      await page.locator('#contact-name').fill('Local reviewer');
      assert(!await page.locator('body').evaluate(node => node.classList.contains('is-mobile-chrome-hidden')), 'Modal interactions retain visible chrome.');
      assert(await page.locator(navSelector).evaluate(node => node.inert), 'Visible navigation respects the modal focus boundary.');
      await page.locator('#contact-modal .modal-close').click();
      await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
      await chromeState(page, false);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No horizontal overflow.');
      await page.goto(`${base}/portfolio/handwritingRating`);
      await settle(page);
      await scrollPage(page, 240);
      await chromeState(page, true);
      const questionDock = page.locator('.project-question-dock');
      const question = page.locator('.project-question-link');
      await page.waitForFunction(() => {
        const dock = document.querySelector('.project-question-dock');
        return dock.inert && getComputedStyle(dock).visibility === 'hidden' && dock.getBoundingClientRect().top >= innerHeight;
      });
      await question.evaluate(node => node.focus());
      assert(!await questionDock.evaluate(node => node.contains(document.activeElement)), 'The hidden project question cannot steal focus.');
      await page.screenshot({ path: path.join(artifactDir, `project-${viewport.width}-hidden.png`) });
      await scrollPage(page, 215);
      await chromeState(page, false);
      await question.waitFor({ state: 'visible' });
      assert(!await questionDock.evaluate(node => node.inert), 'Scrolling upward restores the project question action.');
      await question.click();
      await page.locator('#contact-modal.active').waitFor();
      await page.locator('#contact-message').fill('A local project question that is not sent.');
      assert(!await page.locator('body').evaluate(node => node.classList.contains('is-mobile-chrome-hidden')), 'The project question dialog preserves visible chrome.');
      await page.locator('#contact-modal .modal-close').click();
      await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
      await chromeState(page, false);
      await question.waitFor({ state: 'visible' });
      await page.screenshot({ path: path.join(artifactDir, `project-${viewport.width}-shown.png`) });
      assert.deepEqual(errors, [], 'Mobile navigation has no runtime exceptions.');
      console.log(`Mobile scroll chrome passed: ${viewport.width}x${viewport.height}, full hide/reveal, focus, nested inputs, navigation, consent and dialogs.`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `failure-${viewport.width}.png`) }).catch(() => {});
      throw error;
    } finally { await context.close(); }
  }
}

module.exports = runMobileScrollChromeChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mobile-scroll-chrome-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch();
    await runMobileScrollChromeChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-mobile-scroll-chrome') });
  } finally {
    await browser?.close(); server.closeAllConnections();
    await new Promise(resolve => server.close(resolve)); fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
