'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests, settle } = require('../release/fixtures.cjs');

async function checkMobilePlacement(page, label, { atTop = true } = {}) {
  await page.locator('#pcz-banner.pcz-visible').waitFor();
  await page.waitForFunction(() => {
    const banner = document.querySelector('#pcz-banner');
    return banner && getComputedStyle(banner).position !== 'fixed';
  });
  const geometry = await page.evaluate(() => {
    const banner = document.querySelector('#pcz-banner');
    const masthead = document.querySelector('[data-mobile-site-masthead]');
    const main = document.querySelector('main');
    const dock = document.querySelector('[data-mobile-section-nav]');
    let content = main;
    while (content && content.parentElement !== document.body) content = content.parentElement;
    const rect = element => element?.getBoundingClientRect();
    return {
      position: getComputedStyle(banner).position,
      banner: rect(banner),
      masthead: rect(masthead),
      content: rect(content),
      dock: dock && !dock.hidden ? rect(dock) : null,
      beforeContent: !!content && !!(banner.compareDocumentPosition(content) & Node.DOCUMENT_POSITION_FOLLOWING),
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: innerWidth
    };
  });
  assert(geometry.masthead && geometry.content && geometry.dock, `${label}: mobile shell exists`);
  assert(['relative', 'static'].includes(geometry.position), `${label}: banner participates in layout`);
  assert(geometry.beforeContent, `${label}: banner precedes main content in focus order`);
  if (atTop) {
    assert(geometry.banner.top >= geometry.masthead.bottom - 2,
      `${label}: banner starts below masthead: ${JSON.stringify(geometry)}`);
  }
  assert(geometry.content.top >= geometry.banner.bottom - 2,
    `${label}: content starts after the banner: ${JSON.stringify(geometry)}`);
  assert(geometry.banner.bottom <= geometry.dock.top - 2,
    `${label}: banner does not cover bottom navigation: ${JSON.stringify(geometry)}`);
  assert(geometry.banner.left >= -1 && geometry.banner.right <= geometry.viewportWidth + 1,
    `${label}: banner is fully inside the narrow viewport: ${JSON.stringify(geometry)}`);
  assert(geometry.documentWidth <= geometry.viewportWidth + 1,
    `${label}: no horizontal overflow: ${JSON.stringify(geometry)}`);
  return geometry;
}

async function run() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mobile-consent-env-'));
  const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-mobile-consent-in-flow');
  fs.mkdirSync(artifactDir, { recursive: true });
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch();

    for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, serviceWorkers: 'block', reducedMotion: 'reduce' });
      await isolateRequests(context, base);
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      try {
        await page.goto(base);
        await settle(page);
        await checkMobilePlacement(page, `Home ${viewport.width}`);
        await page.screenshot({ path: path.join(artifactDir, `home-first-visit-${viewport.width}.png`) });

        await page.locator('#pcz-manage').click();
        await page.locator('#pcz-modal.pcz-visible').waitFor();
        assert.equal(await page.locator('#pcz-modal').getAttribute('aria-modal'), 'true');
        await page.keyboard.press('Escape');
        await page.locator('#pcz-modal').waitFor({ state: 'hidden' });
        await checkMobilePlacement(page, `Home after settings ${viewport.width}`);

        await page.locator('[data-mobile-section-nav] [data-mobile-section="tools"]').click();
        await page.waitForFunction(() => document.body.dataset.siteRouteCategory === 'tools');
        await settle(page);
        await checkMobilePlacement(page, `Tools route ${viewport.width}`, { atTop: false });
        await page.locator('[data-mobile-section-nav] [data-mobile-section="projects"]').click();
        await page.waitForFunction(() => document.body.dataset.siteRouteCategory === 'projects');
        await settle(page);
        await checkMobilePlacement(page, `Projects route ${viewport.width}`, { atTop: false });

        const consentChoice = viewport.width === 320 ? '#pcz-reject' : '#pcz-accept';
        await page.locator(consentChoice).click();
        await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
        const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('pcz_consent_v1'))?.categories);
        assert(saved && saved.necessary === true && saved.analytics === (viewport.width === 390),
          `The ${consentChoice} choice persists accurately.`);
        assert.deepEqual(errors, [], `Mobile ${viewport.width} has no page errors`);
        console.log(`Mobile consent passed: ${viewport.width}x${viewport.height}, settings, route changes, ${consentChoice}.`);
      } finally {
        await context.close();
      }
    }

    const starfallContext = await browser.newContext({ viewport: { width: 320, height: 568 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(starfallContext, base);
    const starfall = await starfallContext.newPage();
    try {
      await starfall.goto(`${base}/games/project-starfall`);
      await settle(starfall);
      await checkMobilePlacement(starfall, 'Starfall first visit');
      await starfall.screenshot({ path: path.join(artifactDir, 'starfall-first-visit-320.png') });
      await starfall.locator('#pcz-reject').click();
      await starfall.locator('#pcz-banner').waitFor({ state: 'hidden' });
    } finally {
      await starfallContext.close();
    }

    const landscapeContext = await browser.newContext({ viewport: { width: 844, height: 390 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(landscapeContext, base);
    const landscape = await landscapeContext.newPage();
    try {
      await landscape.goto(base);
      await settle(landscape);
      await landscape.locator('#pcz-banner.pcz-visible').waitFor();
      const placement = await landscape.evaluate(() => {
        const banner = document.querySelector('#pcz-banner');
        const dock = document.querySelector('[data-mobile-section-nav]');
        return {
          position: getComputedStyle(banner).position,
          bannerBottom: banner.getBoundingClientRect().bottom,
          dockTop: dock.getBoundingClientRect().top,
          dockHidden: dock.hidden
        };
      });
      assert.equal(placement.position, 'fixed', 'Short landscape keeps the existing floating banner.');
      assert(!placement.dockHidden && placement.bannerBottom <= placement.dockTop - 1,
        `Short landscape banner stays above its bottom navigation: ${JSON.stringify(placement)}`);
      await landscape.screenshot({ path: path.join(artifactDir, 'landscape-first-visit-844.png') });
    } finally {
      await landscapeContext.close();
    }

    const desktopContext = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(desktopContext, base);
    const desktop = await desktopContext.newPage();
    try {
      await desktop.goto(base);
      await settle(desktop);
      await desktop.locator('#pcz-banner.pcz-visible').waitFor();
      assert.equal(await desktop.locator('#pcz-banner').evaluate(node => getComputedStyle(node).position), 'fixed',
        'Desktop keeps its existing fixed banner layout.');
      await desktop.screenshot({ path: path.join(artifactDir, 'desktop-first-visit-1440.png') });
    } finally {
      await desktopContext.close();
    }
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(envDir, { recursive: true, force: true });
  }
}

if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
module.exports = run;
