'use strict';

// Retained shipped economy uses its real mapped tracks inside each scene. Funded
// fixtures isolate renderer and real transaction continuity, never pacing.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const F = require('./helpers/wayfarers-onboarding.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-station-legacy-')));
const report = { output, browser: 'Browser plugin not available; repository Playwright workflow used.', flows: [], errors: [] };
function legacyLesson(base, id) {
  const state = H.clone(base), practice = state.onboarding.practice;
  practice.progress[id] = 0;
  practice.active = null;
  delete practice.bindings[id];
  delete practice.intentions[id];
  for (const key of ['proofs', 'supplies']) practice[key] = practice[key].filter(entry => !entry.startsWith(id + ':'));
  for (const key of ['rewards', 'helpRewards']) practice[key] = practice[key].filter(entry => entry !== id);
  state.onboarding.progress[id] = 0;
  state.onboarding.rewardClaims = state.onboarding.rewardClaims.filter(entry => entry !== id);
  const guide = H.Core.getView(state).onboarding.guides.find(row => row.id === id);
  assert(guide?.available, 'The retained save actually earns its area lesson');
  assert(H.Core.act(state, { type: 'expedition-select', areaId: guide.areaId }).ok);
  assert(H.Core.act(state, guide.visitAction).ok);
  F.announceDiscoveries(state);
  assert(H.Core.validateState(state).valid);
  return state;
}
async function run() {
  const files = path.join(output, 'bundle');
  bundle(files);
  const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/assets\//, '/');
    const file = path.resolve(files, '.' + pathname);
    if (!file.startsWith(files + path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file, (error, bytes) => {
      if (error) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.webp': 'image/webp' })[path.extname(file)] || 'application/json');
      response.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({ headless: true });
  try {
    const seed = F.completeAreaGuides(H.mature());
    assert(H.Core.act(seed, { type: 'expedition-batch', count: 1 }).ok);
    F.announceDiscoveries(seed);
    seed.lastUpdate = 1000;
    const record = Storage.createStore({ storage: null, now: () => 1000 }).export(seed);
    assert(record.ok, record.message);
    for (const width of [320, 390]) {
      const context = await browser.newContext({ viewport: { width, height: width === 320 ? 740 : 844 }, hasTouch: true, reducedMotion: 'reduce' });
      await context.addInitScript(({ key, text }) => { if (!sessionStorage.seeded) { localStorage.setItem(key, text); sessionStorage.seeded = '1'; } }, { key: Storage.SAVE_KEY, text: record.text });
      const page = await context.newPage();
      page.setDefaultTimeout(10000);
      page.on('pageerror', error => report.errors.push(error.message));
      await page.clock.install({ time: new Date(1000) });
      await page.goto('http://127.0.0.1:' + server.address().port + '/assets/wayfarers/index.html');
      await page.clock.runFor(2000);
      async function saved() { return page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; }); }
      for (const area of H.Core.getView(seed).stations.areas.filter(area => width === 390 || area.id === 'quarry')) {
        await page.locator('[data-wx-objective]').click();
        await page.locator('[data-wx-area="' + area.id + '"]').click();
        await page.clock.runFor(600);
        assert.equal(await page.locator('.wx-station-segment').count(), area.stations.length);
        if (area.id === 'quarry') {
          assert.equal(await page.locator('[data-wx-station="quarry:crystal-lab"]').count(), 0, 'Retained save never depicts an unowned Crystal Lab');
          assert.match(await page.locator('[data-wx-station-open="quarry:tool-forge"]').innerText(), /Refinery/);
        }
        for (const station of area.stations) {
          const realRows = station.skills.filter(row => row.visible !== false && row.action?.type === 'expedition-buy');
          const segment = page.locator('[data-wx-station="' + station.id + '"]');
          assert.deepEqual(await segment.locator('.wx-inline-upgrade').evaluateAll(nodes => nodes.map(node => node.dataset.wxInlineSkill)), realRows.map(row => row.id), 'Only the real retained production tracks belong inline to this station');
          assert.equal(await segment.locator('.wx-inline-upgrade[data-state="locked"]').count(), 0, 'A developed retained save receives no artificial canonical locks');
          const sceneBox = await segment.locator('.wx-station-illustration').boundingBox(), controlsBox = await segment.locator('.wx-station-controls').boundingBox(), segmentBox = await segment.boundingBox();
          assert(Math.abs(segmentBox.height - sceneBox.height) < .1, 'Retained stations have no added permanent strip');
          assert(controlsBox.height <= 80 && controlsBox.x >= sceneBox.x && controlsBox.x + controlsBox.width <= sceneBox.x + sceneBox.width + .1 && controlsBox.y >= sceneBox.y && controlsBox.y + controlsBox.height <= sceneBox.y + sceneBox.height + .1, 'Actual retained controls stay compact and inside their owner scene');
          await page.locator('[data-wx-station-open="' + station.id + '"]').click();
          assert.equal(await page.locator('[data-wx-do="upgrade-scope:area"]').count(), 0, 'New Area infrastructure is absent until reset');
          assert.deepEqual(await page.locator('.wx-station-row').evaluateAll(nodes => nodes.map(node => node.dataset.wxStationSkill)), station.skills.filter(row => row.visible !== false && row.action?.type !== 'expedition-buy').map(row => row.id), 'The drawer retains advanced techniques without duplicating real inline production');
          assert.equal(await page.locator('[data-wx-do="station-core:' + station.id + '"]').count(), 1);
          await page.locator('[data-wx-drawer-close]').click();
          const inspectionBefore = await saved();
          const scrollBefore = await page.locator('.wx-station-world').evaluate(node => node.scrollTop);
          await page.locator('[data-wx-station-help="' + station.id + '"]').click();
          assert.deepEqual(await page.locator('[data-wx-help-skill]').evaluateAll(nodes => nodes.map(node => node.dataset.wxHelpSkill)), realRows.map(row => row.id), 'Retained help selects only the actual mapped tracks');
          for (const row of realRows) {
            await page.locator('[data-wx-help-skill="' + row.id + '"]').click();
            await page.clock.pauseAt(new Date(await page.evaluate(() => Date.now()) + 1000));
            try {
              await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
              const state = await saved();
              const actual = H.Core.getView(state).stations.currentArea.stations.find(item => item.id === station.id).skills.find(item => item.id === row.id);
              assert.equal(await page.locator('.wx-station-help-hero strong').innerText(), actual.name || actual.label || actual.id);
              assert.deepEqual(await page.locator('[data-wx-help-cost]').evaluateAll(nodes => nodes.map(node => node.dataset.wxHelpCost)), actual.cost.map(cost => cost.resource));
              for (const cost of actual.cost) {
                const line = page.locator('[data-wx-help-cost="' + cost.resource + '"]');
                assert.equal(await line.locator('.wx-help-have').innerText(), H.Core.format(state.resources[cost.resource]));
                assert.equal(await line.locator('.wx-help-need').innerText(), H.Core.format(cost.amount), 'Help retains the exact old-economy cost');
              }
              const bounds = await page.locator('.wx-sheet[open][data-kind="station-help"]').boundingBox(), nav = await page.locator('.wx-nav').boundingBox();
              assert(bounds.x >= -1 && bounds.x + bounds.width <= width + 1 && bounds.y >= -1 && bounds.y + bounds.height <= nav.y + 1, 'Retained help stays above navigation');
              assert.deepEqual(state.expedition.areas[area.id].ranks, inspectionBefore.expedition.areas[area.id].ranks, 'Selecting retained help never purchases');
            } finally { await page.clock.resume(); }
          }
          assert(await page.evaluate(() => WayfarersUI.handleBack()));
          assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(), 0);
          assert.equal(await page.locator('.wx-station-world').evaluate(node => node.scrollTop), scrollBefore, 'Help and native Back retain the legacy camera');
        }
        const purchase = area.stations.flatMap(station => station.skills.map(row => ({ station, row }))).find(({ row }) => row.action?.type === 'expedition-buy' && !row.disabled);
        assert(purchase, 'A real legacy purchase remains available');
        const cell = page.locator('[data-wx-inline-skill="' + purchase.row.id + '"]');
        await cell.scrollIntoViewIfNeeded();
        await page.evaluate(id => { const cell = document.querySelector('[data-wx-inline-skill="' + id + '"]'); window.__legacyInline = { cell, buy: cell.querySelector('.wx-inline-buy'), canvas: cell.closest('.wx-station-segment').querySelector('canvas'), scroll: document.querySelector('.wx-station-world').scrollTop }; }, purchase.row.id);
        const before = await saved();
        const expected = H.clone(before);
        const actualRow = H.Core.getView(before).stations.currentArea.stations.find(station => station.id === purchase.station.id).skills.find(row => row.id === purchase.row.id);
        const bought = H.Core.act(expected, actualRow.action);
        assert(bought.ok, bought.message);
        await cell.locator('.wx-inline-buy').click();
        const after = await saved();
        assert.equal(after.expedition.version, 3, 'Presentation does not adopt a new economy');
        assert.equal(after.expedition.areas[area.id].ranks[purchase.row.action.id], expected.expedition.areas[area.id].ranks[purchase.row.action.id], 'The visible purchase applies the actual legacy quote');
        assert(await page.evaluate(() => { const prior = window.__legacyInline; return prior.cell.isConnected && prior.cell.querySelector('.wx-inline-buy') === prior.buy && prior.cell.closest('.wx-station-segment').querySelector('canvas') === prior.canvas && document.querySelector('.wx-station-world').scrollTop === prior.scroll; }), 'Retained transaction keeps the art, button and camera intact');
        assert(H.Core.validateState(after).valid);
        await page.screenshot({ path: path.join(output, 'legacy-' + area.id + '-' + width + '.png') });
      }
      await page.reload();
      await page.clock.runFor(1500);
      assert.equal((await saved()).expedition.version, 3, 'Retained economy survives app-content reload');
      assert((await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 1);
      await context.close();
    }
    report.flows.push('320/390 retained economy: every real mapped expedition-buy track appears inside its owner scene, without artificial slots/locks or drawer duplicates; help selects real tracks and displays exact retained Have/Need without purchasing; native Back preserves the camera; advanced techniques, Refinery semantics and exact purchase effects survive; no invented Crystal Lab or new Area scope; actual inline purchases and reload preserve version3');
    for (const label of ['fresh-greenway', 'greenway', 'quarry', 'watchtower', 'workshop', 'ruins', 'harbor']) {
      const fresh = label === 'fresh-greenway', id = fresh ? 'greenway' : label;
      const lesson = fresh ? H.Core.createState(1000) : legacyLesson(seed, id);
      if (fresh) {
        lesson.expedition.version = 3;
        lesson.stations = H.Core.Stations.initial(lesson);
        const guide = H.Core.getView(lesson).onboarding.guides.find(row => row.id === id);
        assert(H.Core.act(lesson, guide.visitAction).ok);
        F.announceDiscoveries(lesson);
        assert(H.Core.validateState(lesson).valid);
      }
      lesson.lastUpdate = 1000;
      const exported = Storage.createStore({ storage: null, now: () => 1000 }).export(lesson);
      assert(exported.ok, exported.message);
      const context = await browser.newContext({ viewport: { width: fresh ? 320 : 390, height: fresh ? 740 : 844 }, hasTouch: true, reducedMotion: 'reduce' });
      await context.addInitScript(({ key, text }) => localStorage.setItem(key, text), { key: Storage.SAVE_KEY, text: exported.text });
      const page = await context.newPage(), trace = [];
      page.setDefaultTimeout(12000);
      page.on('pageerror', error => report.errors.push(error.message));
      await page.clock.install({ time: new Date(1000) });
      await page.goto('http://127.0.0.1:' + server.address().port + '/assets/wayfarers/index.html');
      await page.clock.runFor(2000);
      const saved = () => page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; });
      const expected = H.Core.getView(lesson).onboarding.guides.find(row => row.id === id).steps.length;
      try {
        for (let count = 0; count < 25 && (await saved()).onboarding.practice.progress[id] < expected; count += 1) {
          await page.clock.runFor(300);
          const coach = page.locator('.wx-guide[open]');
          if (!await coach.count()) await page.clock.runFor(1500);
          assert.equal(await coach.count(), 1, 'Retained ' + id + ' lesson remains active');
          while (/^currency:/.test(await coach.getAttribute('data-step'))) { await page.locator('[data-guide-next]').click(); await page.clock.runFor(100); }
          assert.equal(await coach.getAttribute('data-missing'), 'false', 'Retained ' + id + ' teaches a real visible control');
          const target = page.locator('[data-guide-target]');
          assert.equal(await target.count(), 1);
          const question = await target.getAttribute('data-wx-station-help');
          const command = await target.getAttribute('data-wx-do') || question || await target.getAttribute('data-wx-close');
          trace.push({ step: await coach.getAttribute('data-step'), target: command });
          await page.screenshot({ path: path.join(output, 'legacy-lesson-' + label + '-' + count + '.png') });
          await target.click();
          await page.clock.runFor(50);
          if (question) assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(), 1, 'One retained lesson question-mark press opens its actual help');
          await page.clock.runFor(250);
        }
        const result = await saved();
        assert.equal(result.onboarding.practice.progress[id], expected, JSON.stringify({ id, trace }));
        assert.equal(result.expedition.version, 3);
        assert(H.Core.validateState(result).valid);
        assert(trace.some(step => step.target === id + ':path' || step.target?.startsWith(id + ':')), 'The retained lesson opens its actual mapped station inspector');
        if (fresh) {
          assert.equal(result.expedition.areas.greenway.ranks.boots, 1, 'A fresh retained first lesson buys the actual old-economy supplied rank');
          assert(trace.some(step => step.target === 'inline-buy:area:greenway:boots'));
        } else assert.deepEqual(result.expedition.areas[id].ranks, lesson.expedition.areas[id].ranks, 'An existing-investment review never buys an extra legacy rank');
        report.flows.push('Retained ' + label + ' lesson completes through actual mapped inspector, ' + (fresh ? 'supplied old-economy purchase' : 'existing-investment review without extra purchases') + ' and objective controls: ' + JSON.stringify(trace));
      } catch (error) {
        await page.screenshot({ path: path.join(output, 'legacy-lesson-' + label + '-FAILED.png') });
        report.flows.push('FAILED retained ' + id + ': ' + error.message + '; trace: ' + JSON.stringify(trace) + '; visible screen: ' + await page.locator('body').innerText());
        throw error;
      } finally { await context.close(); }
    }
    assert.deepEqual(report.errors, []);
    process.stdout.write(JSON.stringify({ ok: true, ...report }, null, 2) + '\n');
  } finally {
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
