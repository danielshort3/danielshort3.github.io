'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Onboarding = require('./helpers/wayfarers-onboarding.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-reset-')));
(async () => {
  fs.mkdirSync(output, { recursive: true });
  const files = path.join(output, 'browser-reset-bundle'); bundle(files);
  const html = path.join(files, 'wayfarers/index.html');
  // This exercises the browser storage path. Native acknowledgment/fencing has
  // separate VM/JVM tests and a real installed-app flow.
  fs.writeFileSync(html, fs.readFileSync(html, 'utf8').replace(/<script defer src="wayfarers\/(?:native-checkpoint|checkpoint|android)\.js"><\/script>/g, ''));
  const server = http.createServer((req, res) => {
    const filename = path.resolve(files, '.' + new URL(req.url, 'http://localhost').pathname.replace(/^\/assets\//, '/'));
    if (!filename.startsWith(files + path.sep)) { res.writeHead(403).end(); return; }
    fs.readFile(filename, (err, bytes) => {
      if (err) { res.writeHead(404).end(); return; }
      res.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' })[path.extname(filename)] || 'application/json'); res.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch(); const results = [], errors = [];
  try {
    for (const [width, height] of [[320, 740], [390, 844], [915, 390]]) {
      const state = H.mature(); state.lastUpdate = 2000000;
      for (const kind of ['cards', 'equipment']) H.Core.act(state, { type: 'collection-unlock', kind });
      Onboarding.announceDiscoveries(Onboarding.completeAreaGuides(state));
      const value = Storage.createStore({ storage: null, now: () => 2000000 }).export(state).text;
      const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce' });
      await context.addInitScript(({ key, value }) => {
        if (!sessionStorage.getItem('seeded')) {
          localStorage.setItem(key, value); localStorage.setItem(key + '-backup', value);
          localStorage.setItem('unrelated-site-data', 'keep'); localStorage.setItem('wayfarers-guild-quiet', 'true');
          sessionStorage.setItem('seeded', '1');
        }
      }, { key: Storage.SAVE_KEY, value });
      const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
      await page.clock.install({ time: new Date(2000000) });
      await page.goto('http://127.0.0.1:' + server.address().port + '/wayfarers/index.html');
      await page.locator('[data-wx-options]').click();
      await page.getByRole('button', { name: 'Settings & saves', exact: true }).click();
      await page.locator('[data-testing] summary').click();
      await page.getByRole('button', { name: 'Reset all game progress', exact: true }).click();
      const dialog = page.locator('[data-dialog][open]');
      await page.locator('#wg-reset-confirm').fill('reset');
      assert.equal(await page.locator('[data-confirm-testing-reset]').isDisabled(), true);
      await page.getByRole('button', { name: 'Keep my guild', exact: true }).click();
      const retained = await page.evaluate(() => JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state.createdAt);
      assert.equal(retained, state.createdAt);
      await page.locator('[data-wx-options]').click();
      await page.getByRole('button', { name: 'Settings & saves', exact: true }).click();
      await page.locator('[data-testing] summary').click();
      await page.getByRole('button', { name: 'Reset all game progress', exact: true }).click();
      await page.locator('#wg-reset-confirm').fill('RESET');
      const geometry = await page.evaluate(() => {
        const footer = document.querySelector('[data-confirm-testing-reset]').getBoundingClientRect();
        return { overflow: document.documentElement.scrollWidth - innerWidth, top: footer.top, bottom: footer.bottom, height: footer.height, viewport: innerHeight };
      });
      assert(geometry.overflow <= 1 && geometry.top >= 0 && geometry.bottom <= geometry.viewport + 1 && geometry.height >= 48);
      await page.screenshot({ path: path.join(output, 'reset-confirm-' + width + '.png') });
      if (width === 390) {
        await page.evaluate(() => {
          const set = Storage.prototype.setItem; let failed = false;
          Storage.prototype.setItem = function (key, text) {
            if (!failed && key === WayfarersStorage.BACKUP_KEY) { failed = true; throw new Error('Injected disk full'); }
            return set.call(this, key, text);
          };
        });
        await page.locator('[data-confirm-testing-reset]').click();
        await page.getByRole('button', { name: 'Retry reset', exact: true }).waitFor();
        await page.screenshot({ path: path.join(output, 'reset-retry.png') });
      }
      await Promise.all([page.waitForEvent('load'), page.locator('[data-confirm-testing-reset]').click()]);
      await page.locator('[data-wx-buy="boots"]').waitFor();
      const fresh = await page.evaluate(() => {
        const state = JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;
        const backup = JSON.parse(localStorage.getItem(WayfarersStorage.BACKUP_KEY)).state;
        return { createdAt: state.createdAt, backupCreatedAt: backup.createdAt, ranks: state.expedition.areas.greenway.ranks,
          areas: Object.keys(state.expedition.areas), refits: state.lifetime.refits, cards: state.collection.cards, gear: state.collection.gear,
          marker: JSON.parse(localStorage.getItem(WayfarersStorage.RESET_KEY)), unrelated: localStorage.getItem('unrelated-site-data'), quiet: localStorage.getItem('wayfarers-guild-quiet') };
      });
      assert.notEqual(fresh.createdAt, state.createdAt);
      assert.equal(fresh.createdAt, fresh.backupCreatedAt);
      assert.deepEqual(fresh.areas, ['greenway']); assert.equal(fresh.refits, 0);
      assert.deepEqual(fresh.cards, {}); assert.deepEqual(fresh.gear, {});
      assert(Object.values(fresh.ranks).every(value => value === 0));
      assert.equal(fresh.marker.text, null); assert.equal(fresh.unrelated, 'keep'); assert.equal(fresh.quiet, 'true');
      assert.equal(await page.locator('[data-wx-buy]:visible').count(), 1);
      await page.screenshot({ path: path.join(output, 'reset-fresh-' + width + '.png') });
      await page.reload();
      assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state.createdAt), fresh.createdAt);
      results.push({ width, height, geometry, fresh }); await context.close();
    }
    // A real earned tier has a scheduled safe-point timer when Testing is opened.
    // Reset must dispose that callback and must not revive the old notice on reload.
    const pendingState = H.Core.createState(2000000); H.fund(pendingState);
    for (let i = 0; i < 2; i += 1) assert(H.Core.act(pendingState, H.Core.getView(pendingState).expedition.cards[0].action).ok);
    const oldNotice = H.Core.getView(pendingState).expedition.tiers.notice;
    Onboarding.completeAreaGuides(pendingState);
    assert(oldNotice?.items.length, 'An actual earned notice exists before reset');
    const pendingContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' });
    const pendingValue = Storage.createStore({ storage: null, now: () => 2000000 }).export(pendingState).text;
    await pendingContext.addInitScript(({ key, value }) => { if (!sessionStorage.getItem('seeded')) { localStorage.setItem(key, value); sessionStorage.setItem('seeded', '1'); } }, { key: Storage.SAVE_KEY, value: pendingValue });
    const pendingPage = await pendingContext.newPage(); pendingPage.on('pageerror', error => errors.push(error.message));
    await pendingPage.clock.install({ time: new Date(2000000) });
    await pendingPage.goto('http://127.0.0.1:' + server.address().port + '/wayfarers/index.html');
    await pendingPage.locator('[data-wx-tier-ready]:visible').waitFor();
    await pendingPage.locator('[data-wx-options]').click();
    await pendingPage.getByRole('button', { name: 'Settings & saves', exact: true }).click();
    await pendingPage.locator('[data-testing] summary').click();
    await pendingPage.getByRole('button', { name: 'Reset all game progress', exact: true }).click();
    await pendingPage.clock.runFor(800);
    assert.equal(await pendingPage.locator('.wx-sheet[open]').count(), 0, 'A pending notice cannot interrupt Testing');
    await pendingPage.locator('#wg-reset-confirm').fill('RESET');
    await Promise.all([pendingPage.waitForEvent('load'), pendingPage.locator('[data-confirm-testing-reset]').click()]);
    await pendingPage.locator('[data-wx-buy="boots"]').waitFor(); await pendingPage.clock.runFor(4000);
    assert.equal(await pendingPage.locator('.wx-sheet[open]').count(), 0, 'Old scheduled tier cannot appear after reset');
    assert.equal(await pendingPage.locator('[data-wx-tier-ready]:visible').count(), 0);
    const resetTiers = await pendingPage.evaluate(() => JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state.upgradeTiers);
    assert.deepEqual(resetTiers.pending, []); assert.deepEqual(resetTiers.prompted, []); assert.deepEqual(resetTiers.claimed, ['area:greenway:boots']);
    await pendingPage.reload(); await pendingPage.locator('[data-wx-buy="boots"]').waitFor(); await pendingPage.clock.runFor(4000);
    assert.equal(await pendingPage.locator('.wx-sheet[open]').count(), 0, 'Reload keeps the old tier dismissed with the erased guild');
    await pendingPage.screenshot({ path: path.join(output, 'reset-pending-tier-fresh.png') });
    const pendingTimerReset = { passed: true, oldNotice: oldNotice.id, resetTiers, afterReloadPopup: false };
    await pendingContext.close();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'reset-browser.json'), JSON.stringify({ results, pendingTimerReset, errors }, null, 2));
    console.log(JSON.stringify({ viewports: results.length, errors, output }));
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
