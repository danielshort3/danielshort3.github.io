'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests, settle } = require('../release/fixtures.cjs');

const categories = ['about', 'projects', 'tools', 'games', 'contact'];
const mastheadSelector = '[data-mobile-site-masthead]';
const sectionSelector = '.site-frame[data-frame-navigation="section"] .site-frame__tab.is-active';

async function scrollPage(page, y) {
  const current = await page.evaluate(() => scrollY);
  await page.mouse.move(2, Math.floor(page.viewportSize().height / 2));
  await page.mouse.wheel(0, y - current);
  await page.waitForFunction(y => Math.abs(scrollY - Math.min(y, Math.max(0, document.documentElement.scrollHeight - innerHeight))) <= 2, y);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function checkChrome(page, hidden) {
  await page.waitForFunction(hidden => document.body.classList.contains('is-mobile-chrome-hidden') === hidden, hidden);
  await page.waitForFunction(({ selector, hidden }) => {
    const header = document.querySelector(selector);
    const rect = header.getBoundingClientRect();
    return hidden ? header.inert && rect.bottom <= 0 : !header.inert && rect.top >= -1 && rect.bottom <= innerHeight + 1;
  }, { selector: mastheadSelector, hidden });
}

async function checkSectionPosition(page, category) {
  const section = page.locator(`${sectionSelector}[data-site-tab="${category}"]`);
  await section.waitFor({ state: 'visible' });
  assert.equal(await page.locator('.site-frame [data-site-tab]:visible').count(), 1,
    'Library and detail pages show one active horizontal section rail.');
  const geometry = await page.evaluate(() => {
    const header = document.querySelector('[data-mobile-site-masthead]').getBoundingClientRect();
    const section = document.querySelector('.site-frame[data-frame-navigation="section"] .site-frame__tab.is-active').getBoundingClientRect();
    const slot = document.querySelector('[data-site-frame-slot]').getBoundingClientRect();
    return {
      headerBottom: header.bottom,
      sectionTop: section.top,
      sectionBottom: section.bottom,
      contentTop: slot.top,
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: innerWidth
    };
  });
  assert(geometry.sectionTop >= geometry.headerBottom - 1 && geometry.sectionTop <= geometry.headerBottom + 8,
    `The active section begins below the measured masthead: ${JSON.stringify(geometry)}.`);
  assert(geometry.contentTop >= geometry.sectionBottom - 1 && geometry.contentTop <= geometry.sectionBottom + 8,
    `Content follows the active section without overlap: ${JSON.stringify(geometry)}.`);
  assert(geometry.documentWidth <= geometry.viewportWidth + 1, 'The route has no horizontal overflow.');
}

async function runMobileScrollChromeChecks({ browser, base, artifactDir, prepareContext }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 320, height: 740 },
    { width: 844, height: 390 },
    { width: 834, height: 1112 },
    { width: 1440, height: 900 }
  ]) {
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
      assert.equal(await page.locator('[data-mobile-section-nav], .mobile-site-dock').count(), 0,
        'The old five-button bottom navigation is absent from the DOM.');
      if (viewport.width === 834 || viewport.width >= 960) {
        assert.equal(await page.locator('[data-site-tab]:visible').count(), 5,
          'Tablet and desktop retain the five-tab site frame.');
        assert(await page.locator(mastheadSelector).isHidden(), 'The mobile masthead is not duplicated on wider screens.');
        continue;
      }

      if (await page.locator('#pcz-reject').isVisible()) {
        await page.locator('#pcz-reject').click();
        await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
      }
      await checkChrome(page, false);
      assert.deepEqual(await page.locator('[data-mobile-explore-category]').evaluateAll(nodes => nodes.map(node => node.dataset.mobileExploreCategory)), categories,
        'The top Explore menu retains all five sections.');
      assert.equal(await page.locator('[data-site-tab]:visible').count(), 5,
        'The expanded mobile homepage keeps its colored section rows.');
      await page.screenshot({ path: path.join(artifactDir, `mobile-${viewport.width}-header.png`) });
      if (viewport.width === 844) continue;

      await scrollPage(page, 240);
      await checkChrome(page, true);
      assert(!(await page.locator(mastheadSelector).evaluate(node => {
        node.querySelector('a')?.focus();
        return node.contains(document.activeElement);
      })), 'Hidden masthead controls cannot steal focus.');
      await scrollPage(page, 190);
      await checkChrome(page, false);
      await page.keyboard.press('Tab');
      await checkChrome(page, false);

      await page.locator('.mobile-site-masthead__explore-button').click();
      await page.locator('[data-mobile-explore-category="tools"]').click();
      await page.waitForFunction(() => SiteFrame.current()?.view === 'overview' && SiteFrame.current()?.category === 'tools');
      await settle(page);
      assert.equal(await page.locator('[data-site-tab]:visible').count(), 5);

      for (const [category, route] of [
        ['projects', '/portfolio'],
        ['tools', '/tools'],
        ['games', '/games']
      ]) {
        await page.goto(`${base}${route}`);
        await settle(page);
        await checkChrome(page, false);
        await checkSectionPosition(page, category);
        await page.screenshot({ path: path.join(artifactDir, `mobile-${viewport.width}-${category}-library.png`) });
      }

      for (const [category, route] of [
        ['projects', '/portfolio/handwritingRating'],
        ['tools', '/tools/text-compare'],
        ['games', '/games/project-starfall'],
        ['contact', '/contact']
      ]) {
        await page.goto(`${base}${route}`);
        await settle(page);
        await checkChrome(page, false);
        await checkSectionPosition(page, category);
      }

      await page.locator('#contact-name').fill('Inline reviewer');
      assert(!await page.locator(mastheadSelector).evaluate(node => node.inert),
        'Inline contact leaves the shared masthead interactive.');
      await page.goto(`${base}/portfolio/website`);
      await settle(page);
      await page.locator('.project-question-link').click();
      await page.locator('#contact-name').fill('Local reviewer');
      assert(!await page.locator('body').evaluate(node => node.classList.contains('is-mobile-chrome-hidden')),
        'A contact dialog retains the top masthead.');
      assert(await page.locator(mastheadSelector).evaluate(node => node.inert),
        'The visible masthead respects the dialog focus boundary.');
      await page.locator('#contact-modal .modal-close').click();
      await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });

      await page.goto(`${base}/tools`);
      await settle(page);
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
      const footer = await page.locator('[data-site-shell-footer]').evaluate(node => {
        const rect = node.getBoundingClientRect();
        return { bottom: rect.bottom, viewportHeight: innerHeight, height: rect.height };
      });
      assert(Math.abs(footer.bottom - footer.viewportHeight) <= 2 && footer.height < 140,
        `The compact footer reaches the viewport bottom without a dock-sized gap: ${JSON.stringify(footer)}.`);
      assert.deepEqual(errors, [], 'Mobile navigation has no runtime exceptions.');
      console.log(`Mobile chrome passed: ${viewport.width}x${viewport.height}, masthead hide/reveal, Explore, section labels, dialogs, footer.`);
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
