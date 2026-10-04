'use strict';

// Retained shipped economy uses the new world/drawer presentation. Funded
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
const output = fs.mkdtempSync(path.join(os.tmpdir(), 'guild-station-legacy-'));
const report = { output, flows: [], errors: [] };
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
          await page.locator('[data-wx-station-open="' + station.id + '"]').click();
          assert.equal(await page.locator('[data-wx-do="upgrade-scope:area"]').count(), 0, 'New Area infrastructure is absent until reset');
          assert.deepEqual(await page.locator('.wx-station-row').evaluateAll(nodes => nodes.map(node => node.dataset.wxStationSkill)), station.skills.map(row => row.id), 'Exactly the real retained purchases belong to this station');
          await page.locator('[data-wx-drawer-close]').click();
        }
        const purchase = area.stations.flatMap(station => station.skills.map(row => ({ station, row }))).find(({ row }) => row.action?.type === 'expedition-buy' && !row.disabled);
        assert(purchase, 'A real legacy purchase remains available');
        await page.locator('[data-wx-station-open="' + purchase.station.id + '"]').click();
        const before = await saved();
        const expected = H.clone(before);
        const actualRow = H.Core.getView(before).stations.currentStation.skills.find(row => row.id === purchase.row.id);
        const bought = H.Core.act(expected, actualRow.action);
        assert(bought.ok, bought.message);
        await page.locator('[data-wx-station-skill="' + purchase.row.id + '"] .wx-station-buy').click();
        const after = await saved();
        assert.equal(after.expedition.version, 3, 'Presentation does not adopt a new economy');
        assert.equal(after.expedition.areas[area.id].ranks[purchase.row.action.id], expected.expedition.areas[area.id].ranks[purchase.row.action.id], 'The visible purchase applies the actual legacy quote');
        assert(H.Core.validateState(after).valid);
        await page.screenshot({ path: path.join(output, 'legacy-' + area.id + '-' + width + '.png') });
        await page.locator('[data-wx-drawer-close]').click();
      }
      await page.reload();
      await page.clock.runFor(1500);
      assert.equal((await saved()).expedition.version, 3, 'Retained economy survives app-content reload');
      assert((await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 1);
      await context.close();
    }
    report.flows.push('320/390 retained economy: real owner-grouped tracks and techniques, Refinery semantics, no invented Crystal Lab or new Area scope; actual purchases and reload preserve version3');
    assert.deepEqual(report.errors, []);
    process.stdout.write(JSON.stringify({ ok: true, ...report }, null, 2) + '\n');
  } finally {
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
