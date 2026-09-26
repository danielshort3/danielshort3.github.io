'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');

(async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'starfall-start-env-'));
  const server = createLocalServer({ envDir });
  const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'starfall-start-data');
  fs.mkdirSync(artifactDir, { recursive: true });
  let browser;
  let releaseDownload = () => {};
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ reducedMotion: 'reduce', serviceWorkers: 'block', viewport: { width: 360, height: 640 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let downloads = 0;
    const downloadGate = new Promise(resolve => { releaseDownload = resolve; });
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.origin !== base) return route.abort();
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 503, json: {} });
      if (/\/dist\/project-starfall-hurtboxes\.[a-f0-9]{8}\.js$/.test(url.pathname)) {
        downloads += 1;
        if (downloads === 1) return route.abort();
        return downloadGate.then(() => route.continue());
      }
      return route.continue();
    });
    await page.goto(`${base}/games/project-starfall`, { waitUntil: 'domcontentloaded' });
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').evaluate(button => button.click());
    const start = page.locator('[data-starfall-start-screen] [data-starfall-action="load"]');
    await expect(start).toBeVisible({ timeout: 60000 });
    const assertStartReachable = async (width, height, scroll = true) => {
      await page.setViewportSize({ width, height });
      if (scroll) await start.scrollIntoViewIfNeeded();
      const geometry = await page.evaluate(() => {
        const button = document.querySelector('[data-starfall-start-screen] [data-starfall-action="load"]');
        const bounds = button.getBoundingClientRect();
        const stage = document.querySelector('.project-starfall-canvas-wrap').getBoundingClientRect();
        return {
          insideStage: bounds.top >= stage.top && bounds.bottom <= stage.bottom,
          hit: document.elementFromPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2) === button
        };
      });
      assert(geometry.insideStage && geometry.hit, `Start must fit in the stage and receive input at ${width}×${height}`);
    };
    await assertStartReachable(360, 640, false);
    await page.screenshot({ path: path.join(artifactDir, 'starfall-start-360.png') });
    await assertStartReachable(360, 800);
    await assertStartReachable(320, 800);
    await assertStartReachable(360, 460);
    await page.setViewportSize({ width: 360, height: 640 });
    assert.equal(downloads, 0, 'loading the start screen must not request collision data');
    assert.equal(await page.evaluate(() => window.ProjectStarfallEnemyHurtboxesData != null), false);
    assert.equal(await page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.engine.start()), false,
      'the engine cannot run combat before the exact table is loaded');

    await start.click();
    await expect(page.locator('[data-starfall-toast]')).toContainText('Select Start to retry');
    await expect(start).toBeEnabled();
    assert.equal(downloads, 1);
    await start.click();
    await expect(page.locator('[data-starfall-start-screen]')).toHaveAttribute('aria-busy', 'true');
    await expect(start).toBeDisabled();
    await page.evaluate(() => { void document.querySelector('[data-starfall-root]').ProjectStarfall.ui.openCharacterSelect(); });
    assert.equal(downloads, 2, 'another activation cannot issue a concurrent download');
    assert.equal(await page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.engine.running), false);
    releaseDownload();
    await expect.poll(() => page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.ui.isCharacterSelectOpen)).toBe(true);
    assert.equal(await page.evaluate(() => Object.keys(window.ProjectStarfallEnemyHurtboxesData.sheets).length), 48);
    assert.equal(await page.evaluate(() => window.ProjectStarfallEngineModules.enemyHurtboxes.isReady()), true);

    await page.locator('[data-starfall-character-slot]').first().click();
    await page.locator('[data-starfall-character-create-open]').click();
    await page.locator('[data-starfall-character-name]').fill('Release Test');
    await page.locator('[data-starfall-character-create-confirm]').click();
    await page.locator('[data-starfall-character-class="fighter"]').click();
    await page.locator('[data-starfall-character-create-confirm]').click();
    await expect.poll(() => page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.engine.running)).toBe(true);
    assert.equal(downloads, 2);
    await page.screenshot({ path: path.join(artifactDir, 'starfall-started.png') });

    await page.setViewportSize({ width: 390, height: 844 });
    const moveRight = page.locator('[data-starfall-touch-controls] [data-starfall-touch-action="moveRight"]');
    await expect(moveRight).toBeVisible();
    await moveRight.scrollIntoViewIfNeeded();
    const bounds = await moveRight.boundingBox();
    assert(bounds, 'the phone-width movement control must have a hit target');
    const touch = await page.context().newCDPSession(page);
    await touch.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
    await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{
      x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2, id: 1
    }] });
    await expect(moveRight).toHaveAttribute('aria-pressed', 'true');
    assert.equal(await page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.engine.input.right), true);
    await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await expect(moveRight).toHaveAttribute('aria-pressed', 'false');
    assert.equal(await page.evaluate(() => document.querySelector('[data-starfall-root]').ProjectStarfall.engine.input.right), false);
    await page.screenshot({ path: path.join(artifactDir, 'starfall-started-mobile.png') });
    assert.deepEqual(errors, []);
    console.log('Starfall browser Start gate and phone-width controls passed: exact data, new-character gameplay, and held movement input.');
  } finally {
    releaseDownload();
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
