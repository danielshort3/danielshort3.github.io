'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const vm = require('node:vm');
const Policy = require('../../js/games/wayfarers-guild/debug-updates.js');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');

function memory() {
  const data = new Map();
  return { data, getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, String(value)), removeItem: key => data.delete(key) };
}
function setup() {
  const storage = memory(), policy = Policy.createPolicy({ storage });
  const old = { apkVersion: 18, contentVersion: 3 }, next = { apkVersion: 18, contentVersion: 4 };
  assert.equal(policy.observeCommitted(old).action, 'none');
  assert.equal(policy.setEnabled(true).ok, true);
  return { storage, policy, old, next };
}

test('option defaults off and first observed successful installation never resets', () => {
  const storage = memory(), policy = Policy.createPolicy({ storage });
  assert.equal(policy.read().enabled, false);
  assert.equal(policy.setEnabled(true).ok, true);
  assert.equal(policy.observeCommitted({ apkVersion: 18, contentVersion: 3 }).action, 'none');
  assert.equal(policy.observeCommitted({ apkVersion: 18, contentVersion: 3 }).action, 'none');
});

test('disabled updates move the baseline without a retroactive reset when enabled', () => {
  const policy = Policy.createPolicy({ storage: memory() });
  policy.observeCommitted({ apkVersion: 18, contentVersion: 3 });
  policy.observeCommitted({ apkVersion: 19, contentVersion: 5 });
  policy.setEnabled(true);
  assert.equal(policy.observeCommitted({ apkVersion: 19, contentVersion: 5 }).action, 'none');
});

test('only a newer committed identity requests a reset with protected commerce and backup', () => {
  const { policy, old, next } = setup();
  assert.equal(policy.observeCommitted(old).action, 'none');
  assert.equal(policy.observeCommitted({ apkVersion: 18, contentVersion: 2 }).action, 'none');
  const request = policy.observeCommitted(next);
  assert.equal(request.action, 'reset');
  assert.deepEqual(request.options, { updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true });
  assert.equal(policy.completeReset(request.updateId, { updateId: request.updateId, text: 'pending' }).ok, false);
  assert.equal(policy.completeReset(request.updateId, { updateId: request.updateId, text: null }).ok, true);
  assert.equal(policy.observeCommitted(next).action, 'none');
});

test('an APK upgrade is detected even if its compatible bundled content has a lower version', () => {
  const { policy } = setup();
  assert.equal(policy.observeCommitted({ apkVersion: 19, contentVersion: 1 }).action, 'reset');
});

test('actual selected content identity handles an APK upgrade once despite stale bundled metadata', () => {
  const { policy } = setup();
  const actual = { apkVersion: 19, contentVersion: 5 }, request = policy.observeCommitted(actual);
  assert.equal(request.action, 'reset');
  assert.equal(policy.completeReset(request.updateId, { updateId: request.updateId, text: null }).ok, true);
  assert.equal(policy.observeCommitted({ apkVersion: 19, contentVersion: 3 }).action, 'none');
  assert.equal(policy.observeCommitted(actual).action, 'none');
});

test('rollback consumes its restored identity without wiping on this or later launches', () => {
  const { policy, next } = setup();
  assert.equal(policy.observeCommitted(Object.assign({}, next, { recovered: true })).action, 'none');
  assert.equal(policy.observeCommitted(next).action, 'none');
});

test('a completed reset journal recovers a lost policy acknowledgment without a second reset', () => {
  const { storage, policy, next } = setup();
  const request = policy.observeCommitted(next);
  const reopened = Policy.createPolicy({ storage });
  assert.equal(reopened.observeCommitted(next, { updateId: request.updateId, text: null }).action, 'complete');
  assert.equal(reopened.observeCommitted(next).action, 'none');
});

test('pending update and enabled preference survive a store reset and policy recreation', () => {
  const { storage, policy, next } = setup();
  let time = 1000;
  const store = Storage.createStore({ storage, crypto, now: () => time });
  const prior = store.load().state;
  assert.equal(store.save(prior).ok, true);
  time = 2000;
  const request = policy.observeCommitted(next);
  const reset = store.resetForTesting(Object.assign({}, request.options, { deferCommit: true }));
  assert.equal(reset.ok, true);
  const seed = reset.journal.seedText;
  const reopenedStore = Storage.createStore({ storage, crypto, now: () => 3000 });
  reopenedStore.load();
  const retry = reopenedStore.resetForTesting(request.options);
  assert.equal(retry.ok, true);
  assert.equal(reopenedStore.pendingReset().seedText, seed, 'retry resumes the exact existing seed');
  const reopenedPolicy = Policy.createPolicy({ storage });
  assert.equal(reopenedPolicy.read().enabled, true);
  assert.equal(reopenedPolicy.observeCommitted(next, reopenedStore.pendingReset()).action, 'complete');
  assert.equal(reopenedPolicy.observeCommitted(next).action, 'none');
});

test('unreadable or corrupt policy storage cannot request a destructive reset', () => {
  const { storage, next } = setup();
  storage.setItem(Policy.KEY, '{broken');
  assert.equal(Policy.createPolicy({ storage }).observeCommitted(next).ok, false);
  assert.equal(Policy.createPolicy({ storage: { getItem() { throw new Error('unavailable'); } } }).observeCommitted(next).action, 'none');
});

test('pre-reset backup failure leaves the old guild and reset generation untouched', () => {
  const storage = memory(), store = Storage.createStore({ storage, crypto, now: () => 1000 });
  assert.equal(store.save(store.load().state).ok, true);
  const before = storage.getItem(Storage.SAVE_KEY), write = storage.setItem;
  storage.setItem = (key, value) => { if (key === Storage.UPDATE_RESET_BACKUP_KEY) throw new Error('disk full'); write(key, value); };
  assert.equal(store.resetForTesting({ updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true }).ok, false);
  assert.equal(storage.getItem(Storage.SAVE_KEY), before);
  assert.equal(storage.getItem(Storage.RESET_KEY), null);
});

test('automatic update reset retains ownership and reward receipts but clears earned progression', () => {
  const storage = memory();
  let time = 1000;
  const store = Storage.createStore({ storage, crypto, now: () => time });
  const old = store.load().state;
  old.premium.owned.push('banner-amber');
  old.premium.equipped = 'banner-amber';
  old.caravan.receipts.push('verified-reward-receipt-001');
  old.caravan.completed.push(900);
  Core.advance(old, 100);
  assert.equal(Core.validateState(old).valid, true);
  assert.equal(store.save(old).ok, true);
  const before = storage.getItem(Storage.SAVE_KEY);
  time = 500000;
  const reset = store.resetForTesting({ updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true });
  assert.equal(reset.ok, true, reset.message);
  assert.equal(Core.validateState(reset.state).valid, true);
  assert.deepEqual(reset.state.premium.owned, ['banner-amber']);
  assert.equal(reset.state.premium.equipped, 'banner-amber');
  assert.deepEqual(reset.state.caravan.receipts, ['verified-reward-receipt-001']);
  assert.deepEqual(reset.state.caravan.completed, [900]);
  assert.equal(reset.state.expedition.areas.greenway.ranks.boots, 0);
  assert.equal(storage.getItem(Storage.UPDATE_RESET_BACKUP_KEY), before);
  assert.equal(store.pendingReset().updateId, 'debug-update-apk18-content4');
});

test('native reset bridge forwards the update receipt and preserves its exact pre-reset backup', async () => {
  const storage = memory();
  const initial = Storage.createStore({ storage, crypto, now: () => 1000 });
  assert.equal(initial.save(initial.load().state).ok, true);
  const before = storage.getItem(Storage.SAVE_KEY), api = Object.assign({}, Storage);
  let message;
  const bridge = { postMessage(text) {
    message = JSON.parse(text);
    if (message.type === 'reset-guild') queueMicrotask(() => bridge.onmessage({ data: JSON.stringify({ type: 'reset-guild', requestId: message.requestId, ok: true }) }));
  } };
  const context = { WayfarersStorage: api, WayfarersAndroid: bridge, localStorage: storage,
    document: { querySelector: () => null }, setTimeout, clearTimeout };
  context.window = context;
  vm.runInNewContext(fs.readFileSync(require.resolve('../../mobile/android/wayfarers/web/checkpoint.js'), 'utf8'), context);
  const store = api.createStore({ storage, crypto, now: () => 2000 });
  store.load();
  const reset = await store.resetForTesting({ updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true });
  assert.equal(reset.ok, true);
  assert.equal(message.updateId, 'debug-update-apk18-content4');
  assert.equal(storage.getItem(Storage.UPDATE_RESET_BACKUP_KEY), before);
  assert.equal(store.pendingReset().updateId, message.updateId);
});

test('native-first reset recovery retains update receipt so the policy cannot wipe a second guild', () => {
  const { storage, policy, next } = setup();
  const initial = Storage.createStore({ storage, crypto, now: () => 1000 });
  assert.equal(initial.save(initial.load().state).ok, true);
  const prior = storage.getItem(Storage.SAVE_KEY), request = policy.observeCommitted(next);
  const reset = initial.resetForTesting(Object.assign({}, request.options, { deferCommit: true }));
  assert.equal(reset.ok, true);
  const journal = initial.pendingReset();
  // Durable native generation wins when WebView dies before its journal flush.
  storage.removeItem(Storage.RESET_KEY);
  storage.setItem(Storage.SAVE_KEY, prior);
  const api = Object.assign({}, Storage), context = { WayfarersStorage: api, localStorage: storage,
    WayfarersNativeCheckpoint: { text: journal.text, generation: journal.id, previousGeneration: '', updateId: journal.updateId },
    document: { querySelector: () => null }, setTimeout, clearTimeout };
  context.window = context;
  vm.runInNewContext(fs.readFileSync(require.resolve('../../mobile/android/wayfarers/web/checkpoint.js'), 'utf8'), context);
  const recovered = api.createStore({ storage, crypto, now: () => 3000 });
  assert.equal(recovered.load().status, 'reset-pending');
  assert.equal(recovered.pendingReset().updateId, request.updateId);
  assert.equal(recovered.finishReset(true).ok, true);
  assert.equal(policy.observeCommitted(next, recovered.pendingReset()).action, 'complete');
  assert.equal(policy.observeCommitted(next).action, 'none');
});

function reservedReward() {
  const state = Core.migrateState(JSON.parse(JSON.stringify(require('./fixtures/wayfarers-v4-state.json'))));
  state.caravan.sequence += 1;
  state.caravan.remainingMs = 0;
  state.caravan.offer = { id: 'wg-debug-update-reserved-ad', golden: true, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
  assert.equal(Core.act(state, { type: 'caravan-select', kind: 'shipment', material: 'ore' }).ok, true);
  const quote = Core.getCaravanQuote(state);
  assert.equal(Core.beginCaravanReward(state, quote).ok, true);
  assert.equal(Core.validateState(state).valid, true);
  return { state, quote };
}

test('reserved verified reward blocks automatic reset before backup or journal writes', () => {
  const { state } = reservedReward(), storage = memory();
  const store = Storage.createStore({ storage, crypto, now: () => state.lastUpdate + 1000 });
  assert.equal(store.save(state).ok, true);
  const before = Array.from(storage.data);
  const result = store.resetForTesting({ updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true });
  assert.equal(result.status, 'reward-pending');
  assert.deepEqual(Array.from(storage.data), before, 'neither save nor transaction changes before the reserved ad finishes');
});

test('reset retains actual delivered-reward idempotency and the three-per-day allowance', () => {
  const { state, quote } = reservedReward();
  const receipt = { receiptId: 'wg-debug-update-verified-reward', offerId: quote.offerId, quote, completedAt: state.lastUpdate };
  assert.equal(Core.grantCaravanReward(state, receipt).ok, true);
  state.caravan.completed = [state.lastUpdate - 2000, state.lastUpdate - 1000, state.lastUpdate];
  const storage = memory(), store = Storage.createStore({ storage, crypto, now: () => state.lastUpdate + 1000 });
  storage.setItem('paid-account-wallet', 'account-owned-external-balance');
  assert.equal(store.save(state).ok, true);
  const reset = store.resetForTesting({ updateId: 'debug-update-apk18-content4', preserveCommerce: true, backupBeforeReset: true });
  assert.equal(reset.ok, true, reset.message);
  const resources = JSON.stringify(reset.state.resources);
  assert.equal(Core.grantCaravanReward(reset.state, receipt).duplicate, true);
  assert.equal(JSON.stringify(reset.state.resources), resources, 'replayed verified receipt cannot grant twice');
  const allowance = Core.beginCaravanReward(reset.state, quote);
  assert.equal(allowance.ok, false);
  assert.match(allowance.message, /Three caravan rewards/);
  assert.equal(storage.getItem('paid-account-wallet'), 'account-owned-external-balance');
});
