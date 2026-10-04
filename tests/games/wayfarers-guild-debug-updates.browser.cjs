'use strict';

// Exercise the real app listener, Settings controls, storage transaction and
// checkpoint bridge. The isolated transport replaces Android, never the policy.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const Policy = require('../../js/games/wayfarers-guild/debug-updates');
const Stations = require('./helpers/wayfarers-stations.cjs');
const Guides = require('./helpers/wayfarers-onboarding.cjs');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-debug-updates-')));
const nativeSource = fs.readFileSync(path.join(__dirname, '../../mobile/android/wayfarers/src/main/java/me/danielshort/wayfarers/MainActivity.kt'), 'utf8');
const metadataGlobal = nativeSource.match(/return "window\.(\w+)=/)?.[1];
assert(metadataGlobal, 'Use the actual native bootstrap metadata namespace');
const report = { browser: 'Browser plugin not available; repository Playwright workflow used.', output, metadataGlobal, flows: [], errors: [] };

function seed() {
  const state = Stations.mature();
  state.lastUpdate = Date.now();
  state.resources.starshards = Core.Numbers.from(5);
  assert(Core.act(state, { type: 'premium-buy', id: 'banner-amber' }).ok);
  assert(Core.act(state, { type: 'premium-equip', id: 'banner-amber' }).ok);
  for (let index = 0; index < 2; index += 1) {
    state.caravan.sequence += 1;
    state.caravan.remainingMs = 0;
    state.caravan.offer = { id: 'wg-update-delivery-' + index, golden: false, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
    assert(Core.act(state, { type: 'caravan-select', kind: 'shipment', material: 'ore' }).ok);
    const quote = Core.getCaravanQuote(state);
    assert(Core.beginCaravanReward(state, quote).ok);
    assert(Core.grantCaravanReward(state, { receiptId: 'wg-update-receipt-' + index, offerId: quote.offerId, quote, completedAt: state.lastUpdate }).ok);
  }
  // A third real offer lets the browser reserve and verify a pending reward.
  state.caravan.sequence += 1;
  state.caravan.remainingMs = 0;
  state.caravan.offer = { id: 'wg-update-pending-delivery', golden: false, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
  Guides.completeAreaGuides(state);
  // Isolate verification/reset ordering from the optional first-visit review.
  // The fixture completes the real review actions without requesting an ad.
  const caravanGuide = Core.getView(state).onboarding.guides.find(item => item.id === 'caravan');
  if (caravanGuide && !caravanGuide.complete) {
    assert(Core.act(state, caravanGuide.visitAction).ok);
    for (let index = 0; index < 40 && Core.getView(state).onboarding.active; index += 1) {
      const active = Core.getView(state).onboarding.active;
      assert(Core.act(state, active.practiceAction || active.inspectAction || active.action).ok);
    }
    assert(Core.getView(state).onboarding.guides.find(item => item.id === 'caravan').complete);
  }
  Guides.announceDiscoveries(state);
  assert(Core.validateState(state).valid, JSON.stringify(Core.validateState(state).errors));
  return state;
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
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
  const url = 'http://127.0.0.1:' + server.address().port + '/assets/wayfarers/index.html';
  const browser = await chromium.launch({ headless: true });
  async function open(width, options = {}) {
    const state = options.state || seed();
    const exported = Storage.createStore({ storage: null, now: () => state.lastUpdate }).export(state);
    assert(exported.ok, exported.message);
    const context = await browser.newContext({ viewport: { width, height: width === 320 ? 740 : 844 }, reducedMotion: 'reduce' });
    await context.addInitScript(({ key, backupKey, text, metadataGlobal, options }) => {
      const transportKey = 'test-native-update-transport';
      let transport = JSON.parse(sessionStorage.getItem(transportKey) || 'null');
      if (!transport) {
        localStorage.setItem(key, text);
        localStorage.setItem(backupKey, text);
        localStorage.setItem('unrelated-site-data', 'keep');
        localStorage.setItem('wayfarers-guild-quiet', 'true');
        localStorage.setItem('test-paid-wallet', 'account-owned-balance');
        for (const [name, value] of options.storage || []) localStorage.setItem(name, value);
        transport = { identity: options.identity || { version: 5, label: 'Bundled game', apkVersion: 18, documentToken: 'qa-native-document-5', recoveryToken: '' }, blockCommit: !!options.blockCommit, billingNeverHydrates: !!options.billingNeverHydrates, documents: 0, ready: [], resets: [], exports: [], acknowledgements: [], reward: { phase: 'ready', quote: null, receipt: null }, requests: [] };
      }
      transport.documents += 1;
      const documentNumber = transport.documents;
      const persist = () => sessionStorage.setItem(transportKey, JSON.stringify(transport));
      const clone = value => JSON.parse(JSON.stringify(value));
      const committed = () => ({ apkVersion: transport.identity.apkVersion, contentVersion: transport.identity.version, recovered: !!transport.identity.recoveryToken, documentToken: transport.identity.documentToken });
      // MainActivity supplies this metadata through native-checkpoint.js after
      // the catalog and persistence scripts, before checkpoint.js and app.js.
      if (transport.billingNeverHydrates) window.WayfarersPlayBilling = { postMessage() {} };
      const rewardSnapshot = () => ({ available: transport.reward.phase === 'ready', configured: true, pending: ['showing', 'verification_pending'].includes(transport.reward.phase), state: transport.reward.phase, receipt: transport.reward.receipt, receipts: transport.reward.receipt ? [transport.reward.receipt] : [], message: 'Isolated verified transport' });
      const emitReward = id => window.dispatchEvent(new CustomEvent('wayfarers:rewarded', { detail: { id: id || 'event', ok: true, data: rewardSnapshot() } }));
      window.WayfarersAndroid = { postMessage(text) {
        const message = JSON.parse(text);
        transport.requests.push({ type: message.type, token: message.documentToken, version: message.version });
        if (message.type === 'checkpoint' || message.type === 'reset-guild') {
          if (message.documentToken !== transport.identity.documentToken) { persist(); return; }
          if (message.type === 'reset-guild') transport.resets.push(clone(message));
          persist();
          queueMicrotask(() => window.WayfarersAndroid.onmessage?.({ data: JSON.stringify({ type: message.type, requestId: message.requestId, ok: true }) }));
        } else if (message.type === 'content-ready') {
          const canvas = document.querySelector('.wx-station-segment canvas');
          if (message.documentToken !== transport.identity.documentToken || message.version !== transport.identity.version || !message.text || canvas?.dataset.sceneStatus !== 'ready') { persist(); return; }
          transport.ready.push({ document: documentNumber, identity: committed() });
          persist();
          if (!transport.blockCommit) window.dispatchEvent(new CustomEvent('wayfarers-update-committed', { detail: committed() }));
        } else if (message.type === 'export') { transport.exports.push(message.text); persist(); }
      } };
      window.WayfarersRewardedAds = { postMessage(text) {
        const request = JSON.parse(text);
        if (request.method === 'watch') { transport.reward.quote = request.args.quote; transport.reward.phase = 'verification_pending'; }
        if (request.method === 'acknowledge') {
          const state = JSON.parse(localStorage.getItem(key)).state;
          transport.acknowledgements.push({ receiptId: request.args.receiptId, saved: state.caravan.receipts.includes(request.args.receiptId) });
          transport.reward.receipt = null; transport.reward.phase = 'ready';
        }
        persist(); queueMicrotask(() => emitReward(request.id));
      } };
      window.__nativeUpdateTest = {
        read: () => clone(transport),
        stage(version, recovered) {
          transport.identity = { version, label: 'Selected updated game', apkVersion: 18, documentToken: 'qa-native-document-' + version, recoveryToken: recovered ? 'qa-restored-content' : '' };
          persist();
        },
        event(detail) { window.dispatchEvent(new CustomEvent('wayfarers-update-committed', { detail })); },
        duplicate() { window.dispatchEvent(new CustomEvent('wayfarers-update-committed', { detail: committed() })); },
        verify() {
          const quote = transport.reward.quote;
          transport.reward.receipt = { receiptId: 'wg-update-browser-verified', offerId: quote.offerId, quote, completedAt: Date.now() };
          transport.reward.phase = 'rewarded'; persist(); emitReward();
        }
      };
      persist();
    }, { key: Storage.SAVE_KEY, backupKey: Storage.BACKUP_KEY, text: exported.text, metadataGlobal, options });
    const page = await context.newPage();
    page.setDefaultTimeout(12000);
    page.on('pageerror', error => report.errors.push(error.message));
    let navigations = 0;
    await page.route(url, async route => { navigations += 1; await route.continue(); });
    await page.route(new URL('/assets/wayfarers/native-checkpoint.js', url).href, route => route.fulfill({ contentType: 'text/javascript', body: 'window.' + metadataGlobal + '=Object.assign(window.' + metadataGlobal + '||{},window.__nativeUpdateTest.read().identity);window.WayfarersNativeCheckpoint=null;' }));
    await page.goto(url);
    if (options.blockCommit) await page.waitForFunction(() => !!window.WayfarersUI);
    else await page.waitForFunction(key => JSON.parse(localStorage.getItem(key) || 'null')?.seen?.contentVersion === 5, Policy.KEY);
    return { page, context, state, navigations: () => navigations };
  }
  async function stored(page) { return page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; }); }
  async function readTransport(page) { return page.evaluate(() => window.__nativeUpdateTest.read()); }
  async function sceneReady(page) {
    await page.waitForFunction(() => document.querySelector('.wx-station-segment canvas')?.dataset.sceneStatus === 'ready');
    await page.waitForTimeout(100);
  }
  async function settings(page) {
    if (await page.locator('[data-dialog][open]').count()) await page.locator('[data-dialog][open] [data-close-dialog]').first().click();
    for (let index = 0; index < 40 && await page.locator('.wx-sheet[open][data-kind="onboarding-notice"]').count(); index += 1) await page.locator('[data-wx-close]').click();
    await page.locator('[data-wx-options]').click();
    await page.getByRole('button', { name: 'Settings & saves', exact: true }).click();
    await page.locator('[data-testing] summary').click();
  }
  async function enable(page) {
    await settings(page);
    const toggle = page.locator('[data-reset-after-update]');
    assert.equal(await toggle.isChecked(), false, 'Update resets start off');
    assert.equal(await toggle.isDisabled(), false, 'Actual native commit enables the setting');
    await toggle.click();
    assert.equal(await page.locator('[data-confirm-update-reset]').count(), 1, 'Enabling requires the real Settings confirmation');
    assert.equal(await page.evaluate(key => JSON.parse(localStorage.getItem(key)).enabled, Policy.KEY), false);
    await page.locator('[data-confirm-update-reset]').click();
    assert.equal(await page.locator('[data-reset-after-update]').isChecked(), true);
  }
  async function installed(page, version, recovered) {
    await page.evaluate(({ version, recovered }) => window.__nativeUpdateTest.stage(version, recovered), { version, recovered: !!recovered });
    await page.reload();
    await page.waitForFunction(({ key, version }) => JSON.parse(localStorage.getItem(key)).seen?.contentVersion === version, { key: Policy.KEY, version });
  }
  try {
    for (const width of [320, 390]) {
      const { page, context, state, navigations } = await open(width);
      await settings(page);
      assert.equal(await page.locator('[data-reset-after-update]').isChecked(), false);
      await page.locator('[data-reset-after-update]').click();
      await page.getByRole('button', { name: 'Keep resets off', exact: true }).click();
      assert.equal(await page.locator('[data-reset-after-update]').isChecked(), false, 'Cancelling the confirmation keeps resets off');
      assert.equal((await stored(page)).createdAt, state.createdAt);
      await installed(page, 6);
      assert.equal((await stored(page)).createdAt, state.createdAt, 'A committed update while off keeps the guild');
      await enable(page);
      const before = await stored(page);
      assert.equal(before.createdAt, state.createdAt, 'Enabling after an update is never retroactive');
      const valid = (await readTransport(page)).identity;
      for (const invalid of [
        { apkVersion: 18, contentVersion: 7, documentToken: 'retired-document' },
        { apkVersion: 19, contentVersion: valid.version, documentToken: valid.documentToken },
        { apkVersion: 18, contentVersion: 999, documentToken: valid.documentToken },
        { apkVersion: 18, contentVersion: valid.version }
      ]) await page.evaluate(detail => window.__nativeUpdateTest.event(detail), invalid);
      await page.waitForTimeout(250);
      assert.equal((await readTransport(page)).resets.length, 0, 'Retired or mismatched committed events cannot reset');
      await installed(page, 7);
      await page.locator('.wx-inline-upgrade').first().waitFor();
      const fresh = await stored(page), transport = await readTransport(page);
      assert.notEqual(fresh.createdAt, before.createdAt);
      assert.deepEqual(Object.keys(fresh.expedition.areas), ['greenway']);
      assert.equal(fresh.stations.ranks['station:greenway:path:pathfinding'], 0, 'Successful update restarts actual canonical progression');
      assert.deepEqual(fresh.premium.owned, before.premium.owned);
      assert.equal(fresh.premium.equipped, before.premium.equipped);
      assert.deepEqual(fresh.caravan.receipts, before.caravan.receipts);
      assert.deepEqual(fresh.caravan.completed, before.caravan.completed, 'Reward allowance timestamps survive the reset');
      assert.equal(transport.resets.length, 1, 'Native reset transaction runs once');
      assert.equal(transport.resets[0].updateId, 'debug-update-apk18-content7');
      assert.equal(transport.resets[0].documentToken, 'qa-native-document-7');
      assert(navigations() >= 4, 'The app performs a real same-document reload after its reset');
      const retention = await page.evaluate(({ backup, reset, policy }) => ({ backup: JSON.parse(localStorage.getItem(backup)).state, reset: JSON.parse(localStorage.getItem(reset)), policy: JSON.parse(localStorage.getItem(policy)), quiet: localStorage.getItem('wayfarers-guild-quiet'), unrelated: localStorage.getItem('unrelated-site-data'), wallet: localStorage.getItem('test-paid-wallet') }), { backup: Storage.UPDATE_RESET_BACKUP_KEY, reset: Storage.RESET_KEY, policy: Policy.KEY });
      assert.equal(retention.backup.createdAt, before.createdAt);
      assert.deepEqual(retention.backup.stations.ranks, before.stations.ranks, 'Pre-update backup retains the actual developed guild');
      assert.equal(retention.reset.text, null);
      assert.equal(retention.policy.pending, null);
      assert.equal(retention.policy.enabled, true);
      assert.equal(retention.quiet, 'true');
      assert.equal(retention.unrelated, 'keep');
      assert.equal(retention.wallet, 'account-owned-balance');
      if (width === 390) {
        await page.locator('[data-wx-options]').click();
        await page.getByRole('button', { name: 'Settings & saves', exact: true }).click();
        await page.locator('[data-testing] summary').click();
        const backupText = await page.evaluate(key => localStorage.getItem(key), Storage.UPDATE_RESET_BACKUP_KEY);
        await page.locator('[data-update-reset-backup]').click();
        assert.equal((await readTransport(page)).exports.at(-1), backupText, 'The actual Settings backup control exports the stored pre-update document verbatim');
      }
      await page.evaluate(() => window.__nativeUpdateTest.duplicate());
      await page.reload();
      await page.waitForFunction(() => !!window.WayfarersUI);
      assert.equal((await stored(page)).createdAt, fresh.createdAt, 'Duplicate commit and routine reload preserve the same fresh seed');
      assert.equal((await readTransport(page)).resets.length, 1);
      await sceneReady(page);
      await page.screenshot({ path: path.join(output, 'reset-fresh-' + width + '.png') });
      report.flows.push({ width, updateReset: true, resets: transport.resets.length, createdAt: fresh.createdAt, retainedShop: fresh.premium.owned, retainedReceipts: fresh.caravan.receipts.length, backupCreatedAt: retention.backup.createdAt });
      await context.close();
    }
    const { page, context, state } = await open(390);
    await enable(page);
    await page.locator('[data-dialog][open] [data-close-dialog]').first().click();
    const keepFind = page.locator('[data-dismiss-find]');
    if (await keepFind.isVisible()) await keepFind.click();
    await page.locator('[data-wx-options]').click();
    await page.locator('[data-wx-do="caravan-options"]').click();
    await page.locator('[data-perform="caravan:shipment"]').click();
    await page.locator('[data-watch-caravan]').click();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state.caravan.pendingQuote !== null);
    const pending = await stored(page);
    await page.evaluate(() => window.__nativeUpdateTest.stage(6, false));
    await page.reload();
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).pending?.contentVersion === 6, Policy.KEY);
    await page.waitForTimeout(1200);
    assert.equal((await stored(page)).createdAt, state.createdAt, 'Reserved verification defers the real app update reset');
    assert.deepEqual((await stored(page)).caravan.pendingQuote, pending.caravan.pendingQuote);
    assert.equal((await readTransport(page)).resets.length, 0);
    assert.equal(await page.evaluate(key => localStorage.getItem(key), Storage.UPDATE_RESET_BACKUP_KEY), null, 'Deferred reward prevents even the backup/reset transaction from starting');
    for (let index = 0; index < 40 && await page.locator('.wx-sheet[open][data-kind="onboarding-notice"]').count(); index += 1) await page.locator('[data-wx-close]').click();
    await page.locator('[data-wx-options]').click();
    await page.locator('[data-wx-do="caravan-options"]').click();
    assert.equal(await page.locator('[data-watch-caravan]').isDisabled(), true, 'The saved quote remains visibly pending verification');
    await sceneReady(page);
    await page.screenshot({ path: path.join(output, 'pending-reward-preserves-guild-390.png') });
    await page.evaluate(() => window.__nativeUpdateTest.verify());
    await page.waitForFunction(() => window.__nativeUpdateTest.read().resets.length === 1);
    await page.waitForFunction(() => !!window.WayfarersUI && JSON.parse(localStorage.getItem(WayfarersStorage.RESET_KEY) || 'null')?.text === null);
    const delivered = await stored(page), transport = await readTransport(page);
    assert.notEqual(delivered.createdAt, state.createdAt);
    assert(delivered.caravan.receipts.includes('wg-update-browser-verified'));
    assert.equal(delivered.caravan.completed.length, 3, 'Verified reward counts toward the retained daily allowance before reset');
    assert.deepEqual(transport.acknowledgements, [{ receiptId: 'wg-update-browser-verified', saved: true }], 'Verified reward saves before acknowledgment and automatic reset');
    assert.equal(transport.resets.length, 1);
    assert(Core.validateState(delivered).valid);
    report.flows.push({ deferredReward: true, retainedVerifiedReceipt: true, retainedDailyUses: delivered.caravan.completed.length, resetOnce: true });
    await context.close();
    const interruptedState = seed(), interruptedText = Storage.createStore({ storage: null, now: () => interruptedState.lastUpdate }).export(interruptedState).text;
    const data = new Map([[Storage.SAVE_KEY, interruptedText]]);
    const storage = { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, String(value)), removeItem: key => data.delete(key) };
    const interruptedStore = Storage.createStore({ storage, crypto, now: () => interruptedState.lastUpdate + 1000 });
    interruptedStore.load();
    const transaction = interruptedStore.resetForTesting({ updateId: 'debug-update-apk18-content6', preserveCommerce: true, backupBeforeReset: true, deferCommit: true });
    assert(transaction.ok, transaction.message);
    const expectedSeed = JSON.parse(transaction.journal.seedText).state;
    data.set(Policy.KEY, JSON.stringify({ version: 1, enabled: true, seen: { apkVersion: 18, contentVersion: 5 }, pending: { id: 'debug-update-apk18-content6', apkVersion: 18, contentVersion: 6 } }));
    const resumed = await open(390, { state: interruptedState, storage: [...data], identity: { version: 6, label: 'Selected updated game', apkVersion: 18, documentToken: 'qa-resume-document-6', recoveryToken: '' }, blockCommit: true, billingNeverHydrates: true });
    await resumed.page.waitForFunction(() => window.__nativeUpdateTest.read().resets.length === 1);
    await resumed.page.waitForFunction(key => JSON.parse(localStorage.getItem(key) || 'null')?.text === null, Storage.RESET_KEY);
    const recovery = await resumed.page.evaluate(({ reset, policy }) => ({ journal: JSON.parse(localStorage.getItem(reset)), policy: JSON.parse(localStorage.getItem(policy)), transport: window.__nativeUpdateTest.read() }), { reset: Storage.RESET_KEY, policy: Policy.KEY });
    assert.equal(recovery.journal.seedText, transaction.journal.seedText, 'A resumed automatic reset commits the exact existing seed');
    assert.equal((await stored(resumed.page)).createdAt, expectedSeed.createdAt);
    assert.equal(recovery.policy.pending, null);
    assert.equal(recovery.policy.seen.contentVersion, 6);
    assert.equal(recovery.transport.resets.length, 1);
    assert.equal(recovery.transport.billingNeverHydrates, true);
    assert.equal(recovery.transport.blockCommit, true, 'Recovery does not require another native committed event');
    await sceneReady(resumed.page);
    await resumed.page.screenshot({ path: path.join(output, 'pending-reset-wallet-unavailable-390.png') });
    report.flows.push({ pendingJournalRecovery: true, exactSeedPreserved: true, unhydratedPlayBridge: true, newCommittedEventRequired: false, resetOnce: true });
    await resumed.context.close();
    assert.deepEqual(report.errors, []);
  } catch (error) {
    for (const context of browser.contexts()) for (const page of context.pages()) if (!page.isClosed()) {
      await page.screenshot({ path: path.join(output, 'FAILED-' + browser.contexts().indexOf(context) + '.png') });
      report.flows.push({ failure: error.message, body: await page.locator('body').innerText(), transport: await page.evaluate(() => window.__nativeUpdateTest?.read()) });
    }
    throw error;
  } finally {
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
  process.stdout.write(JSON.stringify({ ok: true, output, flows: report.flows, errors: report.errors }, null, 2) + '\n');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
