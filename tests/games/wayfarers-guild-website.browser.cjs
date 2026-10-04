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
const report = { url, cases: [], errors: [], consoleErrors: [], missing: [], requests: [], textScale: '130% text-only CSS emulation; installed WebView textZoom has a separate native gate' };
const scaleText = '.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}.wx-guide .wx-guide-quote{font-size:15.6px!important}.wx-guide-meta{font-size:14.3px!important;line-height:20.8px!important}.wx-game .wx-skill-tile .wx-upgrade-info strong{font-size:16.9px!important;line-height:22.1px!important}.wx-game .wx-skill-tile small{font-size:14.3px!important}.wx-game .wx-price{font-size:15.6px!important}';

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
    return { width: innerWidth, height: innerHeight, game: box('.wx-game'), hud: box('.wx-header'), dock: box('.wx-nav'), world: box('.wx-station-world'), horizontal: document.documentElement.scrollWidth - innerWidth, vertical: document.documentElement.scrollHeight - innerHeight };
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
        if (await page.locator('.wx-sheet[open]').count()) await page.locator('[data-wx-close]').click();
        const main = await geometry(page);
        assert(main.horizontal <= 1 && main.vertical <= 1, 'Website wrapper has no page overflow');
        assert(main.game.x >= -1 && main.game.y >= -1 && main.game.right <= width + 1 && main.game.bottom <= height + 1);
        assert(main.game.width >= width - 1 && main.game.height >= height - 1, 'Game fills its native-style website viewport');
        await page.locator('[data-wx-nav="upgrades"]').click();
        assert.equal(await page.locator('.wx-station-row:visible').count(), 3);
        assert.equal(await page.locator('.wx-station-row[data-state="locked"]:visible').count(), 2);
        const buy = page.locator('.wx-station-row').first().locator('.wx-station-buy');
        // Wait for actual production to fund the ordinary rank. No diagnostic
        // wallet funding or tutorial subsidy is used for this purchase.
        await page.waitForFunction(() => !document.querySelector('.wx-station-row .wx-station-buy')?.disabled, null, { timeout: 45000 });
        await buy.click();
        const purchased = await saved(page);
        assert.equal(purchased.stations.ranks['station:greenway:path:pathfinding'], 2, 'A subsequent ordinary purchase changes the real rank');
        assert.deepEqual(await geometry(page), main, 'Opening and buying in the drawer leaves the world camera fixed');
        await page.screenshot({ path: path.join(output, 'drawer-' + label + '.png') });
        if (largeText) {
          await page.setViewportSize({ width: 320, height: 740 });
          await page.waitForTimeout(250);
          const portrait = await geometry(page);
          assert(portrait.horizontal <= 1 && portrait.vertical <= 1, 'Rotation and enlarged text preserve the website viewport');
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
