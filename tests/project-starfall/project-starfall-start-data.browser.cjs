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
    const page = await browser.newPage({ reducedMotion: 'reduce', serviceWorkers: 'block', viewport: { width: 1280, height: 900 } });
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
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    const start = page.locator('[data-starfall-start-screen] [data-starfall-action="load"]');
    await expect(start).toBeVisible({ timeout: 60000 });
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
    assert.deepEqual(errors, []);
    await page.screenshot({ path: path.join(artifactDir, 'starfall-started.png') });
    console.log('Starfall browser Start gate passed: no eager table, retry, single flight, exact data and new-character gameplay.');
  } finally {
    releaseDownload();
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
