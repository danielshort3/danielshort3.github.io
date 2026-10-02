'use strict';

// Exercise the exact offline bundle shipped in the APK, using isolated test saves.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-mobile-')));
const files = path.join(output, 'offline-bundle');
const anchors = ['.wg-topline', '[data-goal]', '.wg-scene-column', '.wg-controls-column', '[data-nav]'];
const geometry = page => page.evaluate(selectors => Object.fromEntries(selectors.map(selector => {
  const rect = document.querySelector(selector).getBoundingClientRect();
  return [selector, { x: rect.x, y: rect.y, width: rect.width, height: rect.height }];
})), anchors);

async function checkLayout(page, name) {
  const result = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth - innerWidth,
    height: document.body.scrollHeight - innerHeight,
    stage: document.querySelector('[data-stage]').scrollHeight - document.querySelector('[data-stage]').clientHeight
  }));
  assert(result.width <= 1 && result.height <= 1, name + ': outer viewport overflow ' + JSON.stringify(result));
  if (await page.locator('[data-panel="play"]').isVisible()) assert(result.stage <= 1, name + ': working screen scrolls');
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  bundle(files);
  const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/assets\//, '/');
    const file = path.join(files, pathname);
    if (!file.startsWith(files + path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file, (error, bytes) => {
      if (error) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.webp': 'image/webp' }[path.extname(file)] || 'application/json');
      response.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  const browser = await chromium.launch({ headless: true });
  const evidence = [];
  async function open(width, height, state) {
    const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce' });
    if (state) {
      const saved = JSON.parse(JSON.stringify(state));
      saved.lastUpdate = 1000;
      Core.act(saved, { type: 'introduction-seen', ids: Core.getPresentation(saved).introductions.map(item => item.id) });
      Core.act(saved, { type: 'discovery-seen', seq: saved.luck.ledger.seq });
      const envelope = Storage.createStore({ storage: null, now: () => 1000 }).export(saved).text;
      await context.addInitScript(({ key, value }) => localStorage.setItem(key, value), { key: Storage.SAVE_KEY, value: envelope });
    }
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) errors.push(response.status() + ' ' + response.url()); });
    await page.clock.install({ time: new Date(1000) });
    await page.goto(base + '/assets/wayfarers/index.html');
    await page.locator('[data-scene][data-scene-status="ready"]').waitFor();
    assert.equal(await page.title(), 'Wayfarers’ Guild');
    return { context, page, errors };
  }
  try {
    for (const [width, height] of [[320, 740], [360, 800], [390, 844], [430, 932], [800, 480]]) {
      const { context, page, errors } = await open(width, height);
      const before = await geometry(page);
      const buy = page.locator('[data-main-actions] [data-perform="main:boots"]');
      const originalBuy = await buy.boundingBox();
      await page.screenshot({ path: path.join(output, 'mobile-new-' + width + '.png') });
      await page.clock.fastForward(10000);
      assert(await buy.isEnabled(), 'first upgrade is affordable by 10s');
      await buy.click();
      const after = await geometry(page);
      assert.deepEqual(after, before, 'purchase does not move layout at ' + width);
      assert.deepEqual(await buy.boundingBox(), originalBuy, 'purchase target stays in place');
      await page.locator('[data-inspect="main:boots"]').click();
      assert(await page.locator('[data-dialog][data-kind="inspect"]').isVisible());
      assert.match(await page.locator('[data-inspect-effect]').innerText(), /travel.*coins/);
      await page.locator('.wg-dialog-heading [data-close-dialog]').click();
      for (let second = 0; second < 120; second += 10) {
        await page.clock.fastForward(10000);
        if (await buy.isEnabled()) await buy.click();
      }
      assert(await page.locator('[data-tab="guild"]').isVisible(), 'Mine becomes accessible');
      assert.deepEqual(await geometry(page), before, 'earned system does not move layout');
      await checkLayout(page, 'new ' + width);
      await page.screenshot({ path: path.join(output, 'mobile-unlocked-' + width + '.png') });
      await page.locator('[data-tab="guild"]').click();
      await page.locator('[data-overview-rooms] canvas[data-scene-status="ready"]').first().waitFor();
      await page.screenshot({ path: path.join(output, 'mobile-mine-' + width + '.png') });
      await checkLayout(page, 'Mine ' + width);
      const button = await page.locator('[data-main-actions] [data-perform="main:miners"]').boundingBox();
      assert(button.width >= 48 && button.height >= 48 && button.height <= 64, 'compact usable purchase');
      assert.deepEqual(errors, []);
      evidence.push({ width, height, before, after, mineButton: button, errors });
      await context.close();
    }
    const fixture = Core.migrateState(require('./fixtures/wayfarers-v3-state.json'));
    assert(fixture && Core.validateState(fixture).valid, 'retained advanced save migrates');
    for (const [width, height] of [[320, 740], [390, 844], [800, 480]]) {
      const { context, page, errors } = await open(width, height, fixture);
      await page.locator('[data-tab="guild"]').click();
      await page.locator('[data-overview-rooms] canvas[data-scene-status="ready"]').first().waitFor();
      await checkLayout(page, 'advanced guild ' + width);
      for (const dock of await page.locator('[data-discovery-dock] button:visible').all()) {
        const rect = await dock.boundingBox();
        assert(rect.width >= 48 && rect.width <= 64 && rect.height >= 48 && rect.height <= 64, 'discovery utilities keep square touch targets');
      }
      const wallet = await page.locator('[data-wallet] .wg-resource strong').allTextContents();
      assert(wallet.every(text => !text.includes(',') && text.length <= 9), 'wallet values retain readable magnitudes');
      await page.locator('[data-wallet-resource]').first().click();
      assert.match(await page.locator('[data-dialog]').innerText(), /Resources/);
      await page.locator('.wg-dialog-heading [data-close-dialog]').click();
      await page.screenshot({ path: path.join(output, 'mobile-advanced-' + width + '.png') });
      await page.locator('[data-overview-rooms] [data-room="forge"]').click();
      await page.locator('[data-more-upgrades]').click();
      await page.locator('[data-all-upgrades] [data-inspect]').first().click();
      await page.locator('.wg-dialog-heading [data-sheet-back]').click();
      assert.equal(await page.locator('[data-dialog]').getAttribute('data-kind'), 'manage');
      await page.screenshot({ path: path.join(output, 'mobile-catalog-' + width + '.png') });
      await page.locator('.wg-dialog-heading [data-close-dialog]').click();
      await page.locator('[data-tab="trail"]').click();
      const before = await geometry(page);
      // Adversarial localization and larger text must not resize the game surface.
      await page.locator('[data-goal-title]').evaluate(node => { node.textContent = 'A much longer translated expedition objective that used to wrap the whole screen'; node.style.fontSize = '22px'; });
      await page.locator('[data-main-actions] .wg-action-heading strong').first().evaluate(node => { node.textContent = 'Exceptionally long translated boots upgrade'; node.style.fontSize = '22px'; });
      assert.deepEqual(await geometry(page), before);
      await checkLayout(page, 'large text ' + width);
      await page.screenshot({ path: path.join(output, 'mobile-text-stress-' + width + '.png') });
      assert.deepEqual(errors, []);
      await context.close();
    }
    fs.writeFileSync(path.join(output, 'mobile-geometry.json'), JSON.stringify(evidence, null, 2));
    console.log('Offline mobile UI: 5 viewport transition checks and 3 advanced/large-text flows passed. ' + output);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
