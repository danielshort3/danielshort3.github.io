/** Mobile navigation order and Explore behavior. Maps are stubbed, never billed. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const categories = ['about', 'projects', 'tools', 'games', 'contact'];
const menuSelector = '#mobile-explore-links';
const buttonSelector = '[data-mobile-explore] > button';

async function settle(page, category, view = 'overview') {
  await page.waitForFunction(({ category, view }) => {
    const frame = window.SiteFrame?.root();
    const viewport = window.SiteFrame?.viewport();
    return frame?.dataset.frameCategory === category && frame.dataset.frameView === view
      && !frame.matches('.site-frame--held, .site-frame--moving')
      && !viewport?.getAnimations().some(animation => animation.pending || animation.playState === 'running')
      && !window.SiteNavigation?.isNavigating?.()
      && (view !== 'overview' || location.pathname === '/' && (location.hash === `#${category}` || category === 'about' && !location.hash));
  }, { category, view });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function select(page, category) {
  await page.locator(buttonSelector).click();
  await page.locator(`[data-mobile-explore-category="${category}"]`).click();
  await settle(page, category);
}

async function checkViewport({ browser, base, artifactDir }, viewport) {
  const context = await browser.newContext({ viewport, reducedMotion: viewport.width === 320 ? 'reduce' : 'no-preference', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  let stage = 'initial';
  let mapRequests = 0;
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route(/^https:\/\/(?:[a-z0-9-]+\.)*(?:google\.com|googleapis\.com)\/maps(?:\/|\?)/, async route => {
    mapRequests += 1;
    await route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>Test map</title><p>Preserved map contents</p>' });
  });
  try {
    await page.goto(`${base}/`, { waitUntil: 'domcontentloaded' });
    await settle(page, 'about');
    await page.locator('#pcz-reject').click();
    await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
    await page.evaluate(() => document.fonts.ready);
    const button = page.locator(buttonSelector);
    const menu = page.locator(menuSelector);

    stage = 'reading-order';
    const order = await page.evaluate(() => [...document.querySelectorAll('a[href], button, input, [tabindex]')]
      .filter(node => node.tabIndex >= 0 && !node.closest('[hidden], [inert]') && node.getClientRects().length
        && getComputedStyle(node).visibility !== 'hidden')
      .map(node => node.dataset.siteTab || node.getAttribute('aria-label') || node.textContent.trim()));
    assert.deepEqual(order.slice(0, 5), ['Skip to main content', 'Daniel Short home', 'Explore', 'Open search', 'about'],
      'The visible mobile masthead follows the skip link and precedes page content in keyboard order.');
    await page.locator('[data-site-tab="about"]').focus();
    await page.keyboard.press('Tab');
    assert(await page.evaluate(() => Boolean(document.activeElement.closest('[data-home-accordion-item="about"]'))),
      'Tab from About enters its visible content instead of jumping to Projects below it.');
    assert(await page.evaluate(() => scrollY < 700), 'Entering About content avoids the former multi-screen jump.');
    await page.locator('[data-home-accordion-item="about"] a').last().focus();
    await page.keyboard.press('Tab');
    assert(await page.locator('[data-site-tab="projects"]').evaluate(node => node === document.activeElement),
      'The next section follows the active content in reading order.');

    stage = 'disclosure';
    await button.click();
    assert.deepEqual(await menu.locator('a').evaluateAll(links => links.map(link => link.dataset.mobileExploreCategory)), categories);
    assert.equal(await menu.locator('[aria-current="page"]').getAttribute('data-mobile-explore-category'), 'about');
    assert(await menu.evaluate(node => { const rect = node.getBoundingClientRect(); return rect.left >= 0 && rect.right <= innerWidth && rect.bottom <= innerHeight; }),
      'The menu fits the narrow viewport.');
    await page.keyboard.press('Tab');
    assert(await menu.locator('a').first().evaluate(node => node === document.activeElement), 'Tab enters the plain link list.');
    await page.keyboard.press('Escape');
    assert(await menu.isHidden() && await button.evaluate(node => node === document.activeElement), 'Escape closes the menu and restores its trigger.');
    await button.click();
    await menu.locator('a').last().focus();
    await page.keyboard.press('Tab');
    assert(await page.locator('.mobile-site-masthead__search-button').evaluate(node => node === document.activeElement),
      'Tab exits the last Explore link without a focus trap.');
    assert(await menu.isHidden(), 'The disclosure closes after focus leaves it.');
    await button.click();
    await page.locator('.mobile-site-masthead__search-button').click();
    assert(await menu.isHidden(), 'Opening search closes Explore.');
    await page.locator('.mobile-site-masthead__search-input').waitFor({ state: 'visible' });
    await page.keyboard.press('Escape');
    assert(await button.isVisible(), 'Closing search restores Explore and the brand.');

    stage = 'navigation-history';
    await select(page, 'tools');
    assert.equal(await menu.locator('[aria-current="page"]').getAttribute('data-mobile-explore-category'), 'tools');
    const historyLength = await page.evaluate(() => history.length);
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await select(page, 'tools');
    assert.equal(await page.evaluate(() => history.length), historyLength, 'Selecting the current section does not add duplicate history or close it.');
    assert(await page.locator('[data-site-tab="tools"]').evaluate(node => {
      const offset = document.querySelector('[data-mobile-site-masthead]').getBoundingClientRect().bottom;
      const top = node.getBoundingClientRect().top;
      const maximum = Math.max(0, document.documentElement.scrollHeight - innerHeight);
      return Math.abs(scrollY - Math.min(maximum, top + scrollY - offset)) <= 2 && SiteFrame.viewport().scrollTop === 0;
    }),
      'Explore returns the current section to its top.');
    await select(page, 'games');
    await page.goBack();
    await settle(page, 'tools');
    assert.equal(await menu.locator('[aria-current="page"]').getAttribute('data-mobile-explore-category'), 'tools');
    await page.goForward();
    await settle(page, 'games');

    stage = 'rapid-selection';
    await page.evaluate(() => {
      for (const category of ['projects', 'tools', 'about']) {
        document.querySelector('[data-mobile-explore] > button').click();
        document.querySelector(`[data-mobile-explore-category="${category}"]`).click();
      }
    });
    await settle(page, 'about');
    assert(await page.locator('[data-home-accordion-item="about"]').isVisible(), 'Rapid selections settle at the last requested section.');
    await page.locator('[data-site-tab="about"]').click();
    await settle(page, '', 'closed');
    assert.equal(await menu.locator('[aria-current="page"]').count(), 0, 'The closed homepage has no selected Explore destination.');
    await select(page, 'about');

    stage = 'persistent-map-resize';
    assert.equal(mapRequests, 0, 'Explore does not preload Contact.');
    await select(page, 'contact');
    const map = page.locator('[data-persistent-contact-map] iframe');
    await map.scrollIntoViewIfNeeded();
    await page.frameLocator('[data-persistent-contact-map] iframe').getByText('Preserved map contents').waitFor();
    await map.evaluate(node => {
      window.exploreTestMap = node;
      window.exploreTestMapWindow = node.contentWindow;
      window.exploreTestMapLoads = 0;
      node.addEventListener('load', () => window.exploreTestMapLoads += 1);
    });
    await select(page, 'about');
    await button.click();
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForFunction(() => SiteFrame.root().dataset.frameCompact === 'false');
    await settle(page, 'about');
    assert(await menu.isHidden() && await button.isHidden(), 'Changing to desktop closes and hides Explore.');
    const desktopRails = await page.locator('[data-site-tab]:visible').evaluateAll(links => links.map(link => {
      const rect = link.getBoundingClientRect(); return { width: rect.width, height: rect.height };
    }));
    assert(desktopRails.every(rect => rect.height > rect.width * 2), 'Desktop keeps its vertical rails.');
    await page.setViewportSize(viewport);
    await page.waitForFunction(() => SiteFrame.root().dataset.frameCompact === 'true');
    await settle(page, 'about');
    await page.locator('[data-site-tab="about"]').focus();
    await page.keyboard.press('Tab');
    assert(await page.evaluate(() => Boolean(document.activeElement.closest('[data-home-accordion-item="about"]'))),
      'Returning to mobile restores the visual reading order.');
    await select(page, 'contact');
    await map.scrollIntoViewIfNeeded();
    await page.frameLocator('[data-persistent-contact-map] iframe').getByText('Preserved map contents').waitFor();
    assert.equal(mapRequests, 1, 'Reordering rails and resizing never loads another map.');
    assert(await map.evaluate(node => node === exploreTestMap && node.contentWindow === exploreTestMapWindow && exploreTestMapLoads === 0),
      'The map node, browsing context and rendered contents survive rail reordering.');

    stage = 'detail-return';
    await select(page, 'tools');
    await page.locator('[data-home-accordion-item="tools"] a[href*="text-compare"]').first().click();
    await page.waitForURL('**/tools/text-compare');
    await settle(page, 'tools', 'detail');
    await select(page, 'projects');
    assert.equal(await menu.locator('[aria-current="page"]').getAttribute('data-mobile-explore-category'), 'projects');
    assert.equal(mapRequests, 1, 'Returning from a detail page does not reload the inactive map.');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Navigation produces no horizontal overflow.');
    assert.deepEqual(errors, [], 'Explore produces no browser runtime errors.');
    await button.click();
    await page.screenshot({ path: path.join(artifactDir, `mobile-explore-${viewport.width}.png`) });
    console.log(`Mobile Explore passed: ${viewport.width}px, focus order, history, rapid selection, resize and one preserved mocked map.`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `mobile-explore-${viewport.width}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${viewport.width}px ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runMobileExploreChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const viewport of [{ width: 390, height: 844 }, { width: 320, height: 740 }]) await checkViewport(options, viewport);
}

module.exports = runMobileExploreChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mobile-explore-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runMobileExploreChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-mobile-explore') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
