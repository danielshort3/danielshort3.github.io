'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));

// This is an integration boundary fixture, not pacing evidence. The released
// fixture supplies an actual completed run; gear and an arrival are placed at
// their valid pre-reset boundary so payment/receipt behavior is isolated.
function pendingReleasedReward() {
  const state = Core.migrateState(clone(require('./fixtures/wayfarers-v4-state.json')));
  state.upgrades['gear-tools'] = 1;
  state.upgrades['gear-boots'] = 1;
  state.resources.starshards = N.from(19);
  state.premium.owned = ['artisan'];
  state.caravan.sequence += 1;
  state.caravan.remainingMs = 0;
  state.caravan.offer = { id: 'wg-test-adoption-arrival', golden: true, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
  state.caravan.receipts = ['wg-test-previous-verified-receipt'];
  Core.setPremiumEntitlements(state, ['compass']);
  assert.ok(Core.act(state, { type: 'caravan-select', kind: 'shipment', material: 'ore' }).ok);
  const quote = Core.getCaravanQuote(state);
  assert.ok(Core.beginCaravanReward(state, quote).ok);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  assert.equal(Core.getRefitPreview(state).available, true);
  return { state, quote };
}

test('explicit adoption keeps locked rewards, receipt history, earned ownership and every RNG clock', () => {
  const { state, quote } = pendingReleasedReward();
  const before = clone(state), preview = Core.getRefitPreview(state);
  assert.equal(Core.act(state, { type: 'refit' }).ok, true);
  assert.equal(state.expedition.version, 4);
  assert.equal(state.schemaVersion, Core.VERSION);
  assert.equal(state.run.id, before.run.id + 1);
  assert.deepEqual(state.luck, before.luck);
  assert.deepEqual(state.caravan, before.caravan);
  assert.deepEqual(Core.getCaravanQuote(state), quote);
  for (const key of ['rng', 'eligibleSeconds', 'untilDrop', 'drops', 'owned', 'equipped']) assert.deepEqual(state.premium[key], before.premium[key]);
  assert.deepEqual(state.resources.starshards, N.add(before.resources.starshards, preview.premiumGift));
  assert.deepEqual(state.resources.notes, N.add(before.resources.notes, preview.reward));
  assert.deepEqual(state.premium.claimedMilestones, ['first-refit']);
  assert.equal(state.premium.owned.includes('compass'), false, 'verified account access is never exported as earned ownership');
  assert.deepEqual(Core.getView(state).premium.accountOwned, ['compass']);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  assert.equal(Core.getRefitPreview(state).premiumGift, 0);
  assert.equal(state.expedition.focus.charges, 3, 'only the initial Focus unlock grants its first charges');
});

test('a frozen pre-adoption reward pays exactly its original amount after save reload and only once', () => {
  const { state, quote } = pendingReleasedReward();
  assert.ok(Core.act(state, { type: 'refit' }).ok);
  const text = JSON.stringify({ format: Storage.FORMAT, version: Storage.VERSION, savedAt: state.lastUpdate, state });
  const values = new Map([[Storage.SAVE_KEY, text]]);
  const store = Storage.createStore({ storage: { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) }, now: () => state.lastUpdate });
  const loaded = store.load({ deferOffline: true });
  assert.equal(loaded.status, 'loaded');
  const next = loaded.state, before = clone(next.resources);
  const receipt = { receiptId: 'wg-test-new-verified-receipt', offerId: quote.offerId, quote, completedAt: next.lastUpdate };
  assert.equal(Core.grantCaravanReward(next, receipt).ok, true);
  for (const [id, amount] of Object.entries(quote.reward.resources)) assert.deepEqual(next.resources[id], N.add(before[id], amount));
  assert.equal(next.caravan.pendingQuote, null);
  assert.deepEqual(next.caravan.receipts, ['wg-test-previous-verified-receipt', receipt.receiptId]);
  const paid = clone(next);
  assert.equal(Core.grantCaravanReward(next, receipt).duplicate, true);
  assert.deepEqual(next, paid);
  assert.deepEqual(Core.validateState(next), { valid: true, errors: [] });
});

test('Charter keeps a reserved reward and partially spent Focus without duplicating a premium gift', () => {
  const { mature, advance } = require('./helpers/wayfarers-progression.cjs');
  const state = mature();
  for (let seconds = 0; !Core.getCharterPreview(state).available && seconds < 86400; seconds += 60) {
    if (state.expedition.completed) assert.ok(Core.act(state, { type: 'expedition-next' }).ok);
    advance(state, 60);
  }
  assert.equal(Core.getCharterPreview(state).available, true);
  if (!state.caravan.offer) {
    state.caravan.sequence += 1;
    state.caravan.remainingMs = 0;
    state.caravan.offer = { id: 'wg-test-charter-arrival', golden: false, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
  }
  assert.ok(Core.act(state, { type: 'caravan-select', kind: 'shipment', material: 'ore' }).ok);
  const quote = Core.getCaravanQuote(state);
  assert.ok(Core.beginCaravanReward(state, quote).ok);
  state.expedition.focus = { charges: 1, recharge: 4321, active: null, remaining: 0, unlocked: true };
  state.refitUpgrades.pace = 3;
  state.legacy.curriculum = 2;
  state.resources.notes = N.from(11);
  const before = clone(state);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  const result = Core.act(state, { type: 'charter' });
  assert.equal(result.ok, true);
  assert.deepEqual(state.caravan, before.caravan);
  assert.deepEqual(state.luck, before.luck);
  assert.deepEqual(state.premium, before.premium);
  assert.deepEqual(state.resources.starshards, before.resources.starshards);
  assert.deepEqual(state.expedition.focus, before.expedition.focus);
  assert.deepEqual(state.legacy, before.legacy);
  assert.deepEqual(Core.getCaravanQuote(state), quote);
  assert.equal(state.refitUpgrades.pace, 0);
  assert.equal(N.cmp(state.resources.notes, 0), 0);
  assert.equal(Core.getCharterPreview(state).premiumGift, 0);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
});
