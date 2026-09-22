'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');

// Exercise the real page/engine and the same Canvas item helper used by the
// inventory and tooltip. The isolated probe does not edit character/save data.
(async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'starfall-icons-env-'));
  const server = createLocalServer({ envDir });
  const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'starfall-icons');
  fs.mkdirSync(artifactDir, { recursive: true });
  let browser;
  let releaseIcon = () => {};
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    const errors = [];
    let iconRequests = 0;
    const gate = new Promise(resolve => { releaseIcon = resolve; });
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.origin !== base) return route.abort();
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 503, json: {} });
      if (url.pathname.endsWith('/items/icons/admin-worldwright-console.png')) {
        iconRequests += 1;
        return gate.then(() => route.continue());
      }
      return route.continue();
    });
    await page.goto(`${base}/games/project-starfall`, { waitUntil: 'domcontentloaded' });
    await expect(page).toHaveTitle(/Project Starfall/);
    await page.waitForFunction(() => document.querySelector('[data-starfall-root]')?.ProjectStarfall?.engine, null, { timeout: 60000 });
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    const cold = await page.evaluate(() => {
      const engine = document.querySelector('[data-starfall-root]').ProjectStarfall.engine;
      const data = window.ProjectStarfallData;
      const itemUi = window.ProjectStarfallUiModules.itemAssets;
      const canvas = document.createElement('canvas');
      canvas.id = 'starfall-icon-probe';
      canvas.style.cssText = 'position:fixed;left:16px;top:16px;z-index:999999;background:#f2dfbf';
      canvas.width = 64;
      canvas.height = 64;
      document.body.append(canvas);
      const settings = {
        data, frame: false, showAura: false,
        getAsset: assetPath => engine.getAsset(assetPath),
        drawAssetFrame: window.ProjectStarfallCore.drawAssetFrame
      };
      const item = { id: 'admin_worldwright_console', kind: 'consumable' };
      const first = itemUi.drawCanvasItemIcon(canvas.getContext('2d'), item, 0, 0, 64, settings);
      const second = itemUi.drawCanvasItemIcon(canvas.getContext('2d'), item, 0, 0, 64, settings);
      const assetPath = itemUi.getItemAsset(item, { data });
      return { first, second, assetPath, requested: !!engine.assets[assetPath] };
    });
    assert.equal(cold.assetPath, 'img/project-starfall/items/icons/admin-worldwright-console.png');
    assert.equal(cold.first, false);
    assert.equal(cold.second, false);
    assert.equal(cold.requested, true, 'drawing a cold occupied slot must initiate its image request');
    await expect.poll(() => iconRequests).toBe(1);
    releaseIcon();
    await expect.poll(() => page.evaluate(() => {
      const engine = document.querySelector('[data-starfall-root]').ProjectStarfall.engine;
      return !!engine.getAsset(window.ProjectStarfallData.ITEM_ASSETS.admin_worldwright_console);
    }), { timeout: 60000 }).toBe(true);
    const visible = await page.evaluate(() => {
      const engine = document.querySelector('[data-starfall-root]').ProjectStarfall.engine;
      const canvas = document.getElementById('starfall-icon-probe');
      const ctx = canvas.getContext('2d');
      const drawn = window.ProjectStarfallUiModules.itemAssets.drawCanvasItemIcon(ctx,
        { id: 'admin_worldwright_console', kind: 'consumable' }, 0, 0, 64,
        { data: window.ProjectStarfallData, frame: false, showAura: false,
          getAsset: assetPath => engine.getAsset(assetPath), drawAssetFrame: window.ProjectStarfallCore.drawAssetFrame });
      const pixels = ctx.getImageData(0, 0, 64, 64).data;
      let count = 0;
      for (let index = 3; index < pixels.length; index += 4) if (pixels[index] >= 128) count += 1;
      return { drawn, count };
    });
    assert.equal(visible.drawn, true);
    assert(visible.count >= 32, 'the decoded Worldwright icon must actually paint visible pixels');
    assert.equal(iconRequests, 1);
    await page.locator('#starfall-icon-probe').screenshot({ path: path.join(artifactDir, 'starfall-worldwright-icon.png') });

    await expect.poll(() => page.evaluate(() => {
      const engine = document.querySelector('[data-starfall-root]').ProjectStarfall.engine;
      return Object.values(window.ProjectStarfallData.ITEM_ASSETS).filter(assetPath => !engine.getAsset(assetPath)).length;
    }), { timeout: 60000 }).toBe(0);
    const count = await page.evaluate(() => {
      const engine = document.querySelector('[data-starfall-root]').ProjectStarfall.engine;
      const data = window.ProjectStarfallData;
      const entries = Object.entries(data.ITEM_ASSETS);
      const canvas = document.createElement('canvas');
      canvas.id = 'starfall-icon-catalog';
      canvas.style.cssText = 'position:absolute;left:0;top:0;z-index:999999;max-width:none';
      canvas.width = 12 * 96;
      canvas.height = Math.ceil(entries.length / 12) * 96;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#f2dfbf';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      entries.forEach(([id], index) => {
        const x = index % 12 * 96;
        const y = Math.floor(index / 12) * 96;
        const drawn = window.ProjectStarfallUiModules.itemAssets.drawCanvasItemIcon(ctx,
          { id }, x + 16, y + 4, 64, { data, frame: false, showAura: false,
            getAsset: assetPath => engine.getAsset(assetPath), drawAssetFrame: window.ProjectStarfallCore.drawAssetFrame });
        if (!drawn) throw new Error(`Missing catalog image: ${id}`);
        ctx.fillStyle = '#1c2631';
        ctx.font = '8px sans-serif';
        ctx.fillText(id, x + 3, y + 84, 90);
      });
      document.body.append(canvas);
      return entries.length;
    });
    await page.locator('#starfall-icon-catalog').screenshot({ path: path.join(artifactDir, 'starfall-assigned-item-icons.png') });
    assert.deepEqual(errors, []);
    console.log(`Starfall browser icon checks passed: one cold Worldwright request, visible inventory-helper pixels and ${count} loaded/drawn item assignments.`);
  } finally {
    releaseIcon();
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
