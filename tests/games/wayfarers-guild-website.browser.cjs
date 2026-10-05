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
const scaleText = '.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}.wx-guide .wx-guide-quote{font-size:15.6px!important}.wx-guide-meta{font-size:14.3px!important;line-height:20.8px!important}.wx-game .wx-skill-tile .wx-upgrade-info strong{font-size:16.9px!important;line-height:22.1px!important}.wx-game .wx-skill-tile small{font-size:14.3px!important}.wx-game .wx-price{font-size:15.6px!important}.wx-game .wx-inline-rank{font-size:13px!important;line-height:18.2px!important}.wx-game .wx-inline-buy{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-price>span{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-price>span[data-wide="true"]{font-size:13px!important}.wx-game .wx-inline-action>small{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-requirement[data-lifetime="true"]{font-size:11.7px!important;line-height:16.9px!important}.wx-game .wx-inline-lock>b{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-count{font-size:11.7px!important;line-height:15.6px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero strong{font-size:22.1px!important;line-height:28.6px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero span{font-size:14.3px!important;line-height:19.5px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero small{font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-help-selector>span:not(.wg-icon){font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-preview strong{font-size:16.9px!important;line-height:23.4px!important}.wx-sheet[data-kind="station-help"] .wx-help-cost{font-size:15.6px!important;line-height:22.1px!important}.wx-sheet[data-kind="station-help"] .wx-help-gate strong{font-size:14.3px!important;line-height:19.5px!important}.wx-sheet[data-kind="station-help"] .wx-help-gate-progress,.wx-sheet[data-kind="station-help"] .wx-help-missing{font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-confirm{font-size:16.9px!important;line-height:23.4px!important}.wx-game .wx-inline-upgrade[data-ready="true"] .wx-inline-action>b{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-lock>b[data-stacked="true"]{line-height:14.3px!important}';

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
    return { width: innerWidth, height: innerHeight, game: box('.wx-game'), hud: box('.wx-header'), dock: box('.wx-nav'), world: box('.wx-station-world'), segment: box('.wx-station-segment'), art: box('.wx-station-illustration'), controls: box('.wx-station-controls'), canvas: box('.wx-station-segment canvas'), scroll: document.querySelector('.wx-station-world')?.scrollTop, horizontal: document.documentElement.scrollWidth - innerWidth, vertical: document.documentElement.scrollHeight - innerHeight };
  });
}

async function stationHelp(page, main, label) {
  const before = await saved(page);
  const skillId = 'station:greenway:path:pathfinding';
  await page.locator('[data-wx-station-help="greenway:path"]').click();
  const sheet = page.locator('.wx-sheet[open][data-kind="station-help"]');
  assert.equal(await sheet.count(), 1, 'The station question mark opens temporary help');
  assert.equal(await sheet.locator('[data-wx-help-skill]').count(), 3);
  await sheet.locator('[data-wx-help-skill="' + skillId + '"]').click();
  const state = await saved(page);
  const model = await page.evaluate(state => WayfarersCore.getView(state).stations.currentArea.stations[0].skills[0], state);
  assert.equal(await sheet.locator('.wx-station-help-hero strong').innerText(), model.name);
  assert.equal(await sheet.locator('.wx-station-help-preview strong').innerText(), model.comparison, 'Public help shows the engine-owned effect preview');
  assert.equal(await sheet.locator('[data-wx-help-cost="coins"] .wx-help-need').innerText(), await page.evaluate(amount => WayfarersCore.format(amount), model.cost[0].amount), 'Public Need uses the actual ordinary quote');
  const have = Number((await sheet.locator('[data-wx-help-cost="coins"] .wx-help-have').innerText()).replace(/,/g, ''));
  assert(Number.isFinite(have) && have >= Math.floor(Numbers.toNumber(before.resources.coins)) - 1 && have <= Math.ceil(Numbers.toNumber(state.resources.coins)) + 1, 'Public Have represents the live wallet during this inspection');
  const bounds = await sheet.boundingBox(), nav = await page.locator('.wx-nav').boundingBox();
  assert(bounds.x >= -1 && bounds.x + bounds.width <= main.width + 1 && bounds.y >= -1 && bounds.y + bounds.height <= nav.y + 1, 'Temporary public help stays above navigation');
  assert(await sheet.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
  await page.screenshot({ path: path.join(output, 'station-help-' + label + '.png') });
  const locked = 'station:greenway:path:trailcraft';
  await sheet.locator('[data-wx-help-skill="' + locked + '"]').click();
  assert.equal(await sheet.locator('[data-wx-help-skill="' + locked + '"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await sheet.locator('[data-wx-help-gate][data-met="false"]').count() > 0, true, 'Locked help names its actual missing gates');
  assert.equal(await sheet.locator('[data-wx-do="station-help-buy:' + locked + '"]').isDisabled(), true);
  assert.deepEqual((await saved(page)).stations.ranks, before.stations.ranks, 'Help selection never buys a rank');
  assert.deepEqual((await saved(page)).stations.unlocked, before.stations.unlocked, 'Help selection never claims an unlock');
  assert(await page.evaluate(() => WayfarersUI.handleBack()));
  assert.equal(await sheet.count(), 0, 'Native Back closes help');
  assert.deepEqual(await geometry(page), main, 'Inspection and native Back leave the public camera and art fixed');
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
          const question = await target.getAttribute('data-wx-station-help');
          const command = await target.getAttribute('data-wx-do');
          if(await target.getAttribute('data-wx-close')!==null && await coach.getAttribute('data-step')==='upgrade')assert((await coach.innerText()).includes('Close these details, then press the highlighted upgrade control in the scene.'),'The supplied purchase lesson first explains its highlighted help close');
          trace.push({ step: await coach.getAttribute('data-step'), target: command || question || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close') });
          if (n === 0 || await coach.getAttribute('data-step') === 'upgrade') await page.screenshot({ path: path.join(output, 'lesson-' + label + '-' + n + '.png') });
          await target.click();
          if (question) assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(), 1, 'One intended question-mark press opens the website lesson help');
          if (command?.startsWith('inline-buy:')) assert.equal((await saved(page)).stations.ranks['station:greenway:path:pathfinding'], 1, 'One intended highlighted press purchases its first supplied rank');
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
        assert(Math.abs(main.segment.height - main.art.height) < .1 && main.controls.height <= 80, 'Website has compact scene-contained controls without a permanent strip');
        assert(Math.abs(main.art.height - main.art.width * 320 / 384) < 1, 'The first illustration preserves its original canvas aspect');
        assert(main.controls.x >= main.art.x && main.controls.right <= main.art.right + .1 && main.controls.y >= main.art.y && main.controls.bottom <= main.art.bottom + .1, 'Website controls sit within their owner illustration');
        await stationHelp(page, main, label);
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
        assert(await page.evaluate(() => { const prior = window.__websiteInline; return prior.cell.isConnected && prior.cell.querySelector('.wx-inline-buy') === prior.buy && prior.cell.closest('.wx-station-segment').querySelector('canvas') === prior.canvas; }), 'An ordinary purchase retains actual inline buttons and station canvas');
        await page.screenshot({ path: path.join(output, 'inline-purchase-' + label + '.png') });
        if (largeText) {
          await page.setViewportSize({ width: 320, height: 740 });
          await page.waitForTimeout(250);
          const portrait = await geometry(page);
          assert(portrait.horizontal <= 1 && portrait.vertical <= 1, 'Rotation and enlarged text preserve the website viewport');
          assert(Math.abs(portrait.segment.height - portrait.art.height) < .1 && portrait.controls.height <= 80 && Math.abs(portrait.art.height - portrait.art.width * 320 / 384) < 1, '320px with 130% text keeps compact controls inside the original art');
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
