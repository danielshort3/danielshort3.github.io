'use strict';

// Real controls and the exact offline APK bundle, with isolated disposable saves.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-expedition-')));
const files = path.join(output, 'expedition-bundle');
const evidence = { opening: [], sessions: [], retained: [], errors: [] };
const anchors = ['.wx-header', '.wx-objective', '.wx-world', '.wx-tray', '.wx-nav'];

function command(type, id) { return '[data-wx-do=' + JSON.stringify(JSON.stringify(id === undefined ? { type } : { type, id })) + ']'; }
async function geometry(page) {
  return page.evaluate(selectors => Object.fromEntries(selectors.map(selector => {
    const box = document.querySelector(selector).getBoundingClientRect();
    return [selector, { x: box.x, y: box.y, width: box.width, height: box.height }];
  })), anchors);
}
async function state(page) {
  return page.evaluate(() => {
    document.dispatchEvent(new Event('freeze'));
    return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;
  });
}
async function layout(page, label) {
  const result = await page.evaluate(() => {
    const sheet = document.querySelector('.wx-sheet[open]');
    const targets = [...document.querySelectorAll('.wx-game button, .wx-sheet[open] button')].filter(button => button.getClientRects().length && (!sheet || sheet.contains(button))).map(button => {
      const box = button.getBoundingClientRect();
      return { name: button.getAttribute('aria-label') || button.textContent.trim().slice(0, 50), width: box.width, height: box.height };
    });
    return { width: document.documentElement.scrollWidth - innerWidth, height: document.body.scrollHeight - innerHeight, sheetWidth: sheet ? sheet.scrollWidth - sheet.clientWidth : 0, targets };
  });
  assert(result.width <= 1 && result.height <= 1 && result.sheetWidth <= 1, label + ': viewport overflow ' + JSON.stringify(result));
  assert.deepEqual(result.targets.filter(target => target.width < 47.5 || target.height < 47.5), [], label + ': touch target below 48 px');
}
async function dismissFind(page) {
  const keep = page.locator('[data-dismiss-find]');
  if (await keep.isVisible()) await keep.click();
}
async function closeSheet(page) {
  if (await page.locator('.wx-sheet[open]').count()) await page.locator('[data-wx-close]').click();
}
async function choose(page, id) {
  await page.locator('[data-wx-do="world-choice"]').click();
  await page.locator(command('expedition-choice', id)).click();
  await closeSheet(page);
}
async function screen(page, filename) { await page.screenshot({ path: path.join(output, filename + '.png') }); }
async function clickAction(page, type, id) {
  const key = await page.locator('.wx-sheet[open] [data-wx-do]').evaluateAll((buttons, wanted) => {
    for (const button of buttons) {
      try { const action = JSON.parse(button.dataset.wxDo); if (action.type === wanted.type && action.id === wanted.id && !button.disabled) return button.dataset.wxDo; } catch (error) {}
    }
    return null;
  }, { type, id });
  assert(key, 'rendered action available: ' + type + ':' + id);
  await page.locator('.wx-sheet[open] [data-wx-do=' + JSON.stringify(key) + ']').click();
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  bundle(files);
  const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/assets\//, '/');
    const file = path.resolve(files, '.' + pathname);
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
  async function open(width, height, savedState) {
    const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce' });
    if (savedState) {
      const saved = JSON.parse(JSON.stringify(savedState));
      saved.lastUpdate = 1000;
      Core.act(saved, { type: 'introduction-seen', ids: Core.getPresentation(saved).introductions.map(item => item.id) });
      Core.act(saved, { type: 'discovery-seen', seq: saved.luck.ledger.seq });
      const envelope = Storage.createStore({ storage: null, now: () => 1000 }).export(saved);
      assert(envelope.ok, 'fixture can be exported');
      await context.addInitScript(({ key, value }) => { if (!sessionStorage.getItem('guild-fixture-loaded')) { localStorage.setItem(key, value); sessionStorage.setItem('guild-fixture-loaded', 'true'); } }, { key: Storage.SAVE_KEY, value: envelope.text });
    }
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400 && !response.url().endsWith('favicon.ico')) errors.push(response.status() + ' ' + response.url()); });
    await page.clock.install({ time: new Date(1000) });
    await page.goto(base + '/assets/wayfarers/index.html');
    await page.locator('.wx-game').waitFor();
    await page.clock.runFor(50);
    await page.locator('[data-wx-canvas][data-scene-status="ready"]').waitFor();
    assert.equal(await page.title(), 'Wayfarers’ Guild');
    return { context, page, errors };
  }
  try {
    for (const [width, height] of [[320, 740], [390, 844], [800, 480]]) {
      const { context, page, errors } = await open(width, height);
      assert.equal(await page.locator('[data-wx-buy]').count(), 1, 'only Boots initially');
      assert(!(await page.locator('[data-wx-nav="guild"]').isVisible()), 'guild earned later');
      assert(!(await page.locator('[data-wx-nav="atlas"]').isVisible()), 'atlas earned later');
      const before = await geometry(page);
      await layout(page, 'opening ' + width);
      await screen(page, 'expedition-opening-' + width);
      await page.clock.runFor(8000);
      const buy = page.locator('[data-wx-buy="boots"]');
      assert(await buy.isEnabled(), 'first purchase ready by 8 seconds');
      const target = await buy.boundingBox();
      await buy.click();
      assert.deepEqual(await geometry(page), before, 'first upgrade cannot move the playfield');
      assert.deepEqual(await buy.boundingBox(), target, 'first price remains in place');
      await page.locator('[data-wx-do="local:boots"]').click();
      assert.equal(await page.locator('.wx-sheet').getAttribute('data-kind'), 'local');
      assert.match(await page.locator('.wx-sheet').innerText(), /travel|rank|milestone/i);
      await layout(page, 'upgrade sheet ' + width);
      await screen(page, 'expedition-upgrade-sheet-' + width);
      await closeSheet(page);
      for (let i = 0; i < 5; i += 1) {
        await page.clock.runFor(5000);
        const enabled = page.locator('[data-wx-buy]:enabled').first();
        if (await enabled.count()) await enabled.click();
      }
      assert((await page.locator('[data-wx-buy]').count()) >= 2, 'next track appears through play');
      assert.deepEqual(await geometry(page), before, 'earned track cannot move the playfield');
      await layout(page, 'unlocked ' + width);
      await screen(page, 'expedition-unlocked-' + width);
      const saved = await state(page);
      await page.reload();
      await page.locator('.wx-game').waitFor();
      await page.clock.runFor(50);
      const restored = await state(page);
      assert.equal(restored.createdAt, saved.createdAt, 'same guild after reload');
      assert.deepEqual(restored.expedition.ranks, saved.expedition.ranks, 'local upgrades survive reload');
      assert.equal(restored.expedition.purchases, saved.expedition.purchases);
      assert.deepEqual(errors, []);
      evidence.opening.push({ width, height, geometry: before, savedPurchases: saved.expedition.purchases });
      await context.close();
    }

    const { context, page, errors } = await open(390, 844);
    let stage = 0;
    let selected = new Set();
    let purchases = 0;
    let lastPurchase = 0;
    let longestGap = 0;
    const stages = [];
    const purchaseEvents = [];
    const choices = [];
    for (let seconds = 2; seconds <= 720 && stage < 3; seconds += 2) {
      await page.clock.runFor(2000);
      await dismissFind(page);
      const finale = page.locator('.wx-sheet[open][data-kind="finale"]');
      if (await finale.count()) {
        const saved = await state(page);
        await layout(page, 'finale ' + stage);
        await screen(page, 'expedition-finale-' + stage);
        stages.push({ stage, seconds, purchases: saved.expedition.purchases, ranks: saved.expedition.ranks });
        assert(saved.expedition.completed, 'finale is real completed engine state');
        stage += 1;
        if (stage < 3) {
          await finale.locator(command('expedition-next')).click();
          await page.clock.runFor(50);
          await screen(page, 'expedition-arrival-' + stage);
        }
        continue;
      }
      const choice = page.locator('[data-wx-do="world-choice"]');
      if (await choice.count()) {
        let id = !selected.has(stage + ':initial') ? ['short', 'throughput', 'repair'][stage] : null;
        if (stage === 2 && (await choice.innerText()).includes('Protect') && !selected.has('tower-guard')) { id = 'protect'; selected.add('tower-guard'); }
        if (id) { await choose(page, id); selected.add(stage + ':initial'); choices.push({ stage, id, seconds }); }
      }
      const available = await page.locator('[data-wx-buy]:enabled').evaluateAll(buttons => buttons.map(button => ({ id: button.dataset.wxBuy, level: Number(button.closest('article').querySelector('[data-wx-rank]').textContent.replace(/\D/g, '')), price: Number(button.getAttribute('aria-label').split(', ').pop().replace(/[^\d.]/g, '')) })).sort((a, b) => a.level - b.level || a.price - b.price));
      if (available.length) {
        await page.locator('[data-wx-buy="' + available[0].id + '"]').click();
        purchases += 1;
        longestGap = Math.max(longestGap, seconds - lastPurchase);
        lastPurchase = seconds;
        purchaseEvents.push({ stage, seconds, id: available[0].id });
      }
      if (seconds % 60 === 0) {
        await layout(page, 'natural session ' + seconds);
        await screen(page, 'expedition-session-' + seconds);
      }
    }
    assert.equal(stage, 3, 'three distinct expeditions finish within twelve minutes through visible controls');
    assert(purchases >= 20, 'a full opening contains frequent useful upgrade opportunities');
    assert(longestGap < 90, 'opening has no unexplained 90 second purchase gap');
    assert(choices.some(choice => choice.id === 'protect'), 'tower protection is a playable decision');
    await closeSheet(page);
    await page.locator('[data-wx-nav="guild"]').click();
    await page.getByRole('button', { name: /Automation.*priorities/i }).click();
    assert.equal(await page.locator('.wx-sheet').getAttribute('data-kind'), 'planning');
    await layout(page, 'earned automation');
    await screen(page, 'expedition-earned-automation');
    await closeSheet(page);
    const completed = await state(page);
    assert(completed.expedition.cleared >= 2, 'third zero-based stage is cleared');
    await page.reload();
    await page.locator('.wx-game').waitFor();
    await page.clock.runFor(50);
    const retained = await state(page);
    assert.equal(retained.createdAt, completed.createdAt);
    assert.equal(retained.expedition.cleared, completed.expedition.cleared);
    assert.deepEqual(retained.expedition.choices, completed.expedition.choices);
    assert.deepEqual(errors, []);
    evidence.sessions.push({ stages, purchases, longestGap, purchaseEvents, choices });
    await context.close();

    const legacy = Core.migrateState(require('./fixtures/wayfarers-v3-state.json'));
    assert(legacy && Core.validateState(legacy).valid, 'legacy guild migrates');
    // Fund one expensive mature equipment purchase in this isolated UI fixture.
    // The preceding first-three-stage playthrough uses only naturally earned resources.
    legacy.resources.ore = Core.Numbers.from('2e10');
    for (const [width, height] of [[320, 740], [390, 844], [800, 480]]) {
      const { context: oldContext, page: oldPage, errors: oldErrors } = await open(width, height, legacy);
      await closeSheet(oldPage);
      await oldPage.locator('[data-wx-nav="guild"]').click();
      await layout(oldPage, 'retained guild ' + width);
      await screen(oldPage, 'expedition-retained-guild-' + width);
      await oldPage.getByRole('button', { name: /Crew.*Explorers/i }).click();
      assert.equal(await oldPage.locator('.wx-sheet').getAttribute('data-kind'), 'crew');
      assert.equal(await oldPage.locator('.wx-sheet select').count(), 0, 'crew uses portrait choices');
      await layout(oldPage, 'retained crew ' + width);
      await screen(oldPage, 'expedition-retained-crew-' + width);
      if (width === 390) {
        await clickAction(oldPage, 'recruit', 'scout');
        assert((await state(oldPage)).crew.owned.includes('scout'), 'recruit through rendered price');
      }
      await oldPage.locator('[data-wx-do="crew-slot:0"]').click();
      assert.equal(await oldPage.locator('.wx-sheet').getAttribute('data-kind'), 'roster');
      await layout(oldPage, 'retained roster ' + width);
      if (width === 390) {
        await clickAction(oldPage, 'specialist', 'scout');
        assert.equal((await state(oldPage)).crew.specialists[0], 'scout', 'portrait assignment persists');
      }
      await oldPage.locator('[data-wx-back]').click();
      assert.equal(await oldPage.locator('.wx-sheet').getAttribute('data-kind'), 'crew', 'back preserves sheet hierarchy');
      if (width === 390) {
        await oldPage.getByRole('button', { name:/Companion.*Choose a travelling partner/ }).click();
        await clickAction(oldPage, 'recruit', 'fox');
        await clickAction(oldPage, 'companion', 'fox');
        assert.equal((await state(oldPage)).crew.companion, 'fox', 'companion choice persists');
        await closeSheet(oldPage);
        await oldPage.getByRole('button', { name:/Forge.*Working|Forge.*Rank/ }).click();
        const toolsBefore = (await state(oldPage)).upgrades['gear-tools'];
        await clickAction(oldPage, 'buy', 'gear-tools');
        assert.equal((await state(oldPage)).upgrades['gear-tools'], toolsBefore + 1, 'equipment bought from compact Forge');
        await closeSheet(oldPage);
        await oldPage.getByRole('button', { name:/Research.*Lasting guild improvements/ }).click();
        await clickAction(oldPage, 'research', 'smart-reserve');
        assert((await state(oldPage)).research.includes('smart-reserve'), 'research purchased through the sheet');
      }
      await closeSheet(oldPage);
      await oldPage.getByRole('button', { name: /Automation/i }).click();
      await layout(oldPage, 'retained automation ' + width);
      await screen(oldPage, 'expedition-retained-automation-' + width);
      if (width === 390) {
        await oldPage.getByRole('button', { name:'Turn helper on', exact:true }).click();
        assert((await state(oldPage)).expedition.automation.enabled, 'local helper enabled from rendered controls');
        await oldPage.getByRole('button', { name:/Purchase priorities/ }).click();
        await clickAction(oldPage, 'plan-priority', 'production');
        assert.equal((await state(oldPage)).guild.plan.priorities.operations, 'production', 'valid guild priority chosen');
      }
      await closeSheet(oldPage);
      await oldPage.locator('[data-wx-nav="atlas"]').click();
      await layout(oldPage, 'retained atlas ' + width);
      await screen(oldPage, 'expedition-retained-atlas-' + width);
      const save = await state(oldPage);
      assert.equal(save.createdAt, legacy.createdAt, 'legacy guild identity retained');
      assert.deepEqual(oldErrors, []);
      evidence.retained.push({ width, height, expedition: save.expedition.index, createdAt: save.createdAt });
      await oldContext.close();
    }
    const crowdedQuarry = Core.normalizeState(JSON.parse(JSON.stringify(legacy)), 1000);
    assert(Core.act(crowdedQuarry, { type:'route', id:'route-1' }).ok);
    crowdedQuarry.resources.coins = Core.Numbers.from(987654321);
    const crowded = await open(320, 740, crowdedQuarry);
    await closeSheet(crowded.page);
    await layout(crowded.page, 'large shared wallet with ingots, caravan and settings');
    assert.match(await crowded.page.locator('[data-wx-wallet] strong').innerText(), /M|B/);
    assert(await crowded.page.locator('[data-wx-local-count]').isVisible());
    assert(await crowded.page.locator('[data-wx-reward]').isVisible());
    assert(await crowded.page.locator('[data-wx-wallet] strong').evaluate(node => node.scrollWidth <= node.clientWidth), 'large balance keeps its magnitude visible');
    await screen(crowded.page, 'expedition-large-wallet-320');
    assert.deepEqual(crowded.errors, []);
    await crowded.context.close();
    fs.writeFileSync(path.join(output, 'expedition-browser-evidence.json'), JSON.stringify(evidence, null, 2));
    console.log('Expedition browser QA passed: 3 viewport openings, 3 natural stage completions, saved reloads, and 3 legacy guild flows. ' + output);
  } finally {
    fs.writeFileSync(path.join(output, 'expedition-browser-partial.json'), JSON.stringify(evidence, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
