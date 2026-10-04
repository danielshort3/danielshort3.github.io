'use strict';

// Verify the published website wrapper, including its real privacy controls.
// Standalone Android content and canonical late-game saves have separate gates.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const Numbers = require('../../js/games/wayfarers-guild/numbers.js');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-website-')));
const base = process.env.WAYFARERS_BASE_URL || 'http://127.0.0.1:4224';
const url = new URL('/games/wayfarers-guild', base).href;
const report = { url, browser: 'Browser plugin not available; repository Playwright workflow used.', cases: [], errors: [], consoleErrors: [], missing: [], requests: [], textScale: '130% text-only CSS emulation; installed WebView textZoom has a separate native gate' };
const scaleText = '.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}.wx-guide .wx-guide-quote{font-size:15.6px!important}.wx-guide-meta{font-size:14.3px!important;line-height:20.8px!important}.wx-game .wx-skill-tile .wx-upgrade-info strong{font-size:16.9px!important;line-height:22.1px!important}.wx-game .wx-skill-tile small{font-size:14.3px!important}.wx-game .wx-price{font-size:15.6px!important}.wx-game .wx-inline-info>strong{font-size:14.3px!important;line-height:18.2px!important}.wx-game .wx-inline-rank{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-buy{font-size:15.6px!important;line-height:20.8px!important}.wx-game .wx-inline-price>span{font-size:15.6px!important;line-height:19.5px!important}.wx-game .wx-inline-buy>small{font-size:13px!important;line-height:16.9px!important}';

async function saved(page) {
  return page.evaluate(() => {
    document.dispatchEvent(new Event('freeze'));
    return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;
  });
}

async function geometry(page) {
  return page.evaluate(() => {
    const box = selector => {
      const rect = document.querySelector(selector)?.getBoundingClientRect();
      return rect && { x: rect.x, y: rect.y, width: rect.width, height: rect.height, right: rect.right, bottom: rect.bottom };
    };
    return { width: innerWidth, height: innerHeight, game: box('.wx-game'), hud: box('.wx-header'), dock: box('.wx-nav'), world: box('.wx-station-world'), art: box('.wx-station-illustration'), controls: box('.wx-station-controls'), canvas: box('.wx-station-segment canvas'), scroll: document.querySelector('.wx-station-world')?.scrollTop, horizontal: document.documentElement.scrollWidth - innerWidth, vertical: document.documentElement.scrollHeight - innerHeight };
  });
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    for (const { width, height, largeText } of [
      { width: 320, height: 740 },
      { width: 390, height: 844 },
      { width: 430, height: 932 },
      { width: 640, height: 256, largeText: true }
    ]) {
      const label = width + (largeText ? '-text130' : '');
      const context = await browser.newContext({ viewport: { width, height }, hasTouch: true, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.setDefaultTimeout(12000);
      page.on('pageerror', error => report.errors.push({ label, message: error.message }));
      page.on('console', message => { if (message.type() === 'error') report.consoleErrors.push({ label, message: message.text() }); });
      page.on('response', response => { if (response.status() === 404) report.missing.push({ label, url: response.url() }); });
      page.on('requestfailed', request => { if (new URL(request.url()).origin === new URL(base).origin) report.requests.push({ label, url: request.url(), failure: request.failure() }); });
      try {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !!window.SiteFrame?.whenSettled);
        await page.evaluate(() => window.SiteFrame.whenSettled());
        await page.waitForFunction(() => !document.querySelector('.site-frame--held,.site-frame--moving'));
        await page.locator('.wx-game').waitFor();
        await page.locator('#pcz-banner').waitFor();
        await page.waitForFunction(() => document.querySelector('.wx-station-segment canvas')?.dataset.sceneStatus === 'ready');
        await page.waitForTimeout(500);
        assert.equal(await page.locator('.wx-guide[open]').count(), 0, 'Website consent remains operable before mandatory teaching');
        assert.equal(await page.locator('.wx-sheet[open]').count(), 0, 'Game discovery overlays wait for privacy choice');
        const coinsBefore = (await saved(page)).resources.coins;
        await page.waitForTimeout(1100);
        assert(Numbers.cmp((await saved(page)).resources.coins, coinsBefore) > 0, 'Privacy choice does not stop normal production');
        await page.screenshot({ path: path.join(output, 'consent-' + label + '.png') });
        await page.locator('#pcz-reject').click();
        await page.locator('#pcz-banner').waitFor({ state: 'detached' });
        await page.locator('.wx-guide[open]').waitFor();
        if (largeText) await page.addStyleTag({ content: scaleText });
        const trace = [];
        for (let n = 0; n < 20 && (await saved(page)).onboarding.practice.progress.greenway < 3; n++) {
          const coach = page.locator('.wx-guide[open]');
          await coach.waitFor();
          while (/^currency:/.test(await coach.getAttribute('data-step'))) {
            await page.locator('[data-guide-next]').click();
            await page.waitForTimeout(100);
          }
          await page.waitForFunction(() => {
            const guide = document.querySelector('.wx-guide[open]'), target = document.querySelector('[data-guide-target]');
            if (!guide || guide.dataset.missing !== 'false' || !target) return false;
            const rect = target.getBoundingClientRect(), hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
            return hit && (hit === target || target.contains(hit));
          });
          assert.equal(await page.locator('[data-guide-leave]:visible').count(), 0, 'The mandatory lesson cannot be skipped');
          const target = page.locator('[data-guide-target]');
          const rect = await target.boundingBox();
          assert(rect.width >= 48 && rect.height >= 48, 'Actual highlighted control has a full touch target');
          trace.push({ step: await coach.getAttribute('data-step'), target: await target.getAttribute('data-wx-do') || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close') });
          if (n === 0 || await coach.getAttribute('data-step') === 'upgrade') await page.screenshot({ path: path.join(output, 'lesson-' + label + '-' + n + '.png') });
          await target.click();
          await page.waitForTimeout(200);
        }
        const taught = await saved(page);
        assert.equal(taught.onboarding.practice.progress.greenway, 3, JSON.stringify(trace));
        assert.equal(taught.stations.ranks['station:greenway:path:pathfinding'], 1, 'The real supplied upgrade was purchased');
        assert(trace.some(step => step.target === 'inline-buy:station:greenway:path:pathfinding'), 'Mandatory teaching uses the actual inline starter purchase');
        if (await page.locator('.wx-sheet[open]').count()) await page.locator('[data-wx-close]').click();
        const first = page.locator('[data-wx-inline-skill="station:greenway:path:pathfinding"]');
        const buy = first.locator('.wx-inline-buy');
        await buy.scrollIntoViewIfNeeded();
        const main = await geometry(page);
        assert(main.horizontal <= 1 && main.vertical <= 1, 'Website wrapper has no page overflow');
        assert(main.game.x >= -1 && main.game.y >= -1 && main.game.right <= width + 1 && main.game.bottom <= height + 1);
        assert(main.game.width >= width - 1 && main.game.height >= height - 1, 'Game fills its native-style website viewport');
        assert.equal(await page.locator('.wx-inline-upgrade').count(), 3, 'First station owns exactly three inline starters');
        assert.equal(await page.locator('.wx-inline-upgrade[data-state="locked"]').count(), 2);
        assert(Math.abs(main.controls.height - 104) < .1, 'Website reserves a fixed 104px strip');
        assert(Math.abs(main.art.height - main.art.width * 320 / 384) < 1, 'The first illustration preserves its original canvas aspect');
        assert(Math.abs(main.controls.y - main.art.bottom) < .1, 'Website controls sit directly beneath the illustration');
        await page.evaluate(() => { const cell = document.querySelector('[data-wx-inline-skill="station:greenway:path:pathfinding"]'); window.__websiteInline = { cell, info: cell.querySelector('.wx-inline-info'), buy: cell.querySelector('.wx-inline-buy'), canvas: cell.closest('.wx-station-segment').querySelector('canvas') }; });
        await page.locator('[data-wx-nav="upgrades"]').click();
        assert.equal(await page.locator('.wx-station-row').count(), 0, 'Drawer excludes duplicate starter purchase controls');
        assert.equal(await page.locator('[data-wx-do="station-core:greenway:path"]').count(), 1, 'Drawer identifies where the core controls live');
        await page.locator('[data-wx-drawer-close]').click();
        // Wait for actual production to fund the ordinary rank. No diagnostic
        // wallet funding or tutorial subsidy is used for this purchase.
        await page.waitForFunction(() => !document.querySelector('[data-wx-inline-skill="station:greenway:path:pathfinding"] .wx-inline-buy')?.disabled, null, { timeout: 45000 });
        await buy.click();
        const purchased = await saved(page);
        assert.equal(purchased.stations.ranks['station:greenway:path:pathfinding'], 2, 'A subsequent ordinary purchase changes the real rank');
        assert.deepEqual(await geometry(page), main, 'Opening the drawer and buying inline leave the world camera, scene and strip fixed');
        assert(await page.evaluate(() => { const prior = window.__websiteInline; return prior.cell.isConnected && prior.cell.querySelector('.wx-inline-info') === prior.info && prior.cell.querySelector('.wx-inline-buy') === prior.buy && prior.cell.closest('.wx-station-segment').querySelector('canvas') === prior.canvas; }), 'An ordinary purchase retains actual inline buttons and station canvas');
        await page.screenshot({ path: path.join(output, 'inline-purchase-' + label + '.png') });
        if (largeText) {
          await page.setViewportSize({ width: 320, height: 740 });
          await page.waitForTimeout(250);
          const portrait = await geometry(page);
          assert(portrait.horizontal <= 1 && portrait.vertical <= 1, 'Rotation and enlarged text preserve the website viewport');
          assert(Math.abs(portrait.controls.height - 104) < .1 && Math.abs(portrait.art.height - portrait.art.width * 320 / 384) < 1, '320px with 130% text keeps the strip and art contract');
          await page.screenshot({ path: path.join(output, 'rotated-text130-320.png') });
        }
        report.cases.push({ label, ...main, trace, lessonRank: 1, purchasedRank: purchased.stations.ranks['station:greenway:path:pathfinding'], productionContinuedDuringConsent: true });
      } catch (error) {
        await page.screenshot({ path: path.join(output, 'FAILED-' + label + '.png') });
        report.cases.push({ label, failed: error.message, geometry: await geometry(page), body: await page.locator('body').innerText() });
        throw error;
      } finally {
        await context.close();
      }
    }
    assert.deepEqual(report.errors, []);
    assert.deepEqual(report.consoleErrors, []);
    assert.deepEqual(report.missing, []);
    assert.deepEqual(report.requests, []);
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify({ ok: true, output, cases: report.cases.length, errors: report.errors, consoleErrors: report.consoleErrors, missing: report.missing, requests: report.requests }, null, 2));
}
run().catch(error => { console.error(error); process.exitCode = 1; });
