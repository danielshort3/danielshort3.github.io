'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const P = require('../../js/games/wayfarers-guild/progression.js');
const M = require('../../js/games/wayfarers-guild/progression-modifiers.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
const near = (a, b) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(b)) * 1e-9, `${a} != ${b}`);
const ratio = (a, b) => N.toNumber(N.div(a, b));
function established() {
  const state = Core.migrateState(clone(require('./fixtures/wayfarers-v4-state.json')));
  P.reset(state, 'refit');
  state.crew.specialists = [null, null];
  state.crew.companion = null;
  state.doctrine = 'balanced';
  state.route.mode = 'frontier';
  state.guild.prepared = { index: -1, kinds: [] };
  return state;
}

test('paid and earned ownership improve actual Trail work once without inventing earned ownership', () => {
  const state = Core.createState(0);
  const before = P.rawRates(state).areas.greenway.work;
  const coins = Core.getRates(state).gain.coins;
  Core.setPremiumEntitlements(state, ['compass']);
  near(P.rawRates(state).areas.greenway.work / before, 1.1);
  assert.deepEqual(state.premium.owned, []);
  state.premium.owned.push('compass');
  near(P.rawRates(state).areas.greenway.work / before, 1.1);
  near(ratio(Core.getRates(state).gain.coins, coins), 1);
  const start = state.expedition.work;
  Core.advance(state, 10);
  near(state.expedition.work - start, before * 1.1 * 10);
});

test('the physical work helper preserves wallet and buffer calculations without mutating its input', () => {
  const state = established(), raw = P.rawRates(state, { baseOnly: true });
  const snapshot = clone(raw), rates = Core.getRates(state);
  const next = M.apply(state, raw, new Set(), { rates });
  assert.deepEqual(raw, snapshot);
  assert.deepEqual(next.gain, raw.gain);
  assert.deepEqual(next.drain, raw.drain);
  for (const id of ['quarry', 'ruins']) if (raw.areas[id]) {
    near(next.areas[id].input, raw.areas[id].input);
    near(next.areas[id].output, raw.areas[id].output);
  }
});

test('artisan ownership increases canonical ore and Quarry work without double wallet multiplication', () => {
  const state = established(), before = P.rawRates(state), rates = Core.getRates(state);
  Core.setPremiumEntitlements(state, ['artisan']);
  const after = P.rawRates(state);
  near(ratio(Core.getRates(state).gain.ore, rates.gain.ore), 1.1);
  near(after.areas.quarry.work / before.areas.quarry.work, 1.1);
  state.premium.owned.push('artisan');
  near(ratio(Core.getRates(state).gain.ore, rates.gain.ore), 1.1);
});

test('travel kits change real work and are excluded from unboosted reward quotes', () => {
  const state = established(), base = P.rawRates(state).areas.greenway.work;
  state.luck.owned = ['living-crucible']; state.luck.active = 'living-crucible';
  state.luck.kit = { prepared: null, active: 'travel', remainingSeconds: 90 };
  near(P.rawRates(state).areas.greenway.work / base, 2);
  const physical = Core.getRates(state), unboosted = Core.getRates(state, true);
  near(ratio(physical.travel, unboosted.travel), 2);
  state.luck.kit.remainingSeconds = 0; state.luck.kit.active = null;
  near(P.rawRates(state).areas.greenway.work / base, 1);
});

test('meal and route preparation affect actual work and respect the Light Pack constraint', () => {
  const state = established();
  if (!state.rooms.includes('kitchen')) state.rooms.push('kitchen');
  state.resources.provisions = N.from(100);
  state.guild.supply = 'steady'; state.meal = 'meal-none';
  const base = P.rawRates(state).areas.greenway.work;
  state.meal = 'meal-travel';
  near(P.rawRates(state).areas.greenway.work / base, 1.5);
  state.challenges.active = 'light-pack';
  near(P.rawRates(state).areas.greenway.work / base, 1);
  state.challenges.active = null; state.meal = 'meal-none';
  state.guild.prepared = { index: state.route.index, kinds: ['scout'] };
  near(P.rawRates(state).areas.greenway.work / base, 1.35);
  state.luck.owned = ['frost-compass']; state.luck.active = 'frost-compass';
  near(P.rawRates(state).areas.greenway.work / base, 1.35 * 1.4);
});

test('Focus cannot inflate unboosted random or reserved caravan reward rates', () => {
  const state = established(), before = Core.getRates(state, true);
  state.expedition.focus = { charges: 2, recharge: 0, active: 'greenway', remaining: 90, unlocked: true };
  const during = Core.getRates(state, true);
  assert.deepEqual(during.gain, before.gain);
  assert.deepEqual(during.drain, before.drain);
  assert.deepEqual(during.travel, before.travel);
  assert.ok(N.cmp(Core.getRates(state).gain.coins, before.gain.coins) > 0);
});

test('a paid compass accelerates a real saved Harbor manifest once and preserves its disclosed payout', () => {
  const { mature, advance } = require('./helpers/wayfarers-progression.cjs');
  const state = mature();
  advance(state, 1);
  assert.ok(state.expedition.areas.harbor.voyages.length);
  const faster = clone(state);
  Core.setPremiumEntitlements(faster, ['compass']);
  const base = P.rawRates(state).areas.harbor, improved = P.rawRates(faster).areas.harbor;
  near(improved.travel / base.travel, 1.1);
  near(base.duration / improved.duration, 1.1);
  const before = clone(state.expedition.areas.harbor.voyages[0]);
  advance(state, .01);
  advance(faster, .01);
  const ordinary = state.expedition.areas.harbor.voyages[0], enhanced = faster.expedition.areas.harbor.voyages[0];
  near((enhanced.work - before.work) / (ordinary.work - before.work), 1.1);
  assert.deepEqual(enhanced.payout, before.payout);
  assert.deepEqual(ordinary.payout, before.payout);
});

test('Workshop provisions receive retained crew, paid Artisan and verified surge effects once', () => {
  const { mature } = require('./helpers/wayfarers-progression.cjs');
  const state = mature(), before = Core.getRates(state);
  const baseWork = P.rawRates(state).areas.workshop.work;
  assert.ok(N.cmp(before.gain.provisions, 0) > 0);
  state.crew.owned.push('naturalist'); state.crew.specialists[0] = 'naturalist';
  near(ratio(Core.getRates(state).gain.provisions, before.gain.provisions), 1.25);
  Core.setPremiumEntitlements(state, ['artisan']);
  near(ratio(Core.getRates(state).gain.provisions, before.gain.provisions), 1.25 * 1.1);
  state.premium.owned.push('artisan');
  near(ratio(Core.getRates(state).gain.provisions, before.gain.provisions), 1.25 * 1.1);
  state.caravan.surge = { resource: 'provisions', remainingSeconds: 2700 };
  near(ratio(Core.getRates(state).gain.provisions, before.gain.provisions), 1.25 * 1.1 * 3);
  near(P.rawRates(state).areas.workshop.work / baseWork, 1.25 * 1.1 * 3);
  near(ratio(Core.getRates(state, true).gain.provisions, before.gain.provisions), 1.25 * 1.1);
  assert.deepEqual(Core.getRates(state).drain, before.drain, 'output bonuses do not silently spend extra ore');
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
});
