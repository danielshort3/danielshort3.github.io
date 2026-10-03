/** Opening economy acceptance checks; only the engine and Node built-ins are required. */
'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const CurrentCore = require('../../js/games/wayfarers-guild/core.js');
// These witnesses cover a released v4 run before explicit economy adoption.
const Core = Object.assign({}, CurrentCore, { createState(now = 0) { const state = CurrentCore.migrateState(require('./fixtures/wayfarers-v4-fresh.json')); state.createdAt = now; state.lastUpdate = now; return state; } });
const N = Core.Numbers;
const E = require('../../js/games/wayfarers-guild/expeditions.js');
const previousState = () => { const state = Core.createState(0); delete state.expedition; return state; };
const clone = value => JSON.parse(JSON.stringify(value));
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
const upgrade = id => Core.Content.UPGRADES.find(item => item.id === id);
const ratio = (a, b) => N.toNumber(N.div(a, b));
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, a + ' differs from ' + b);

function legacyState(version) {
  const state = Core.createState(0);
  state.schemaVersion = version;
  delete state.expedition;
  delete state.guild;
  delete state.introductions;
  delete state.collection;
  if (version < 3) { delete state.luck; delete state.caravan; }
  else Core.Content.RELICS.filter(item => item.chapter > 0).forEach(item => delete state.luck.duplicateProgress[item.id]);
  if (version === 1) { delete state.premium; delete state.resources.starshards; }
  return state;
}

function guidedOpening(seconds) {
  const state = Core.createState(0), purchases = [], milestones = {};
  for (let second = 1; second <= seconds; second += 1) {
    Core.advance(state, 1);
    let view = E.view(state);
    if (view.stage.completed) {
      milestones['stage-' + view.stage.index] = second;
      Core.act(state, view.next.action);
      view = E.view(state);
    }
    const card = view.cards.filter(item => item.visible && !item.disabled).sort((a, b) => N.cmp(a.cost[0].amount, b.cost[0].amount))[0];
    if (card && Core.act(state, card.action).ok) purchases.push({ second, id: card.id });
    state.rooms.forEach(id => { if (milestones[id] === undefined) milestones[id] = second; });
  }
  valid(state);
  return { state, purchases, milestones };
}

test('one initial local control becomes affordable in five to ten seconds', () => {
  const state = Core.createState(0);
  assert.deepEqual(E.view(state).cards.filter(item => item.visible).map(item => item.id), ['boots']);
  Core.advance(state, 5);
  assert.equal(Core.act(state, { type: 'expedition-buy', id: 'boots' }).ok, false);
  Core.advance(state, 2);
  assert.equal(Core.act(state, { type: 'expedition-buy', id: 'boots' }).ok, true);
  assert.deepEqual(state.rooms, ['trail']);
  valid(state);
});

test('the first two minutes deliver several upgrades and a complete learned stage', () => {
  const result = guidedOpening(120);
  assert.ok(result.purchases.length >= 8);
  assert.ok(result.milestones['stage-0'] >= 75 && result.milestones['stage-0'] <= 105);
  assert.equal(result.state.expedition.index, 1);
  assert.deepEqual(result.state.rooms, ['trail', 'mine']);
  assert.equal(result.state.resources.starshards.m, 0);
  assert.equal(result.state.lifetime.refits, 0);
  valid(result.state);
});

test('per-purchase benefits decline smoothly to baseline, with bounded mature uplift', () => {
  const state = previousState();
  let before = Core.getRates(state), lastTravel = Infinity, lastCoins = Infinity;
  for (let rank = 1; rank <= 40; rank += 1) {
    state.upgrades.boots = rank;
    const after = Core.getRates(state);
    const travel = ratio(after.travel, before.travel), coins = ratio(after.gain.coins, before.gain.coins);
    assert.ok(travel >= 1.28 - 1e-10 && coins >= 1.18 - 1e-10);
    assert.ok(travel <= lastTravel + 1e-10 && coins <= lastCoins + 1e-10);
    if (rank > 8) {
      near(travel, 1.28); near(coins, 1.18);
      assert.ok(ratio(after.travel, N.pow(1.28, rank)) < 1.3);
      assert.ok(ratio(after.gain.coins, N.mul(0.3, N.pow(1.18, rank))) < 1.32);
    }
    lastTravel = travel; lastCoins = coins; before = after;
  }
  state.upgrades.boots = 2;
  assert.match(Core.getView(state).actions.find(item => item.id === 'boots').description, /\+32.3% travel and \+22.3% coins/);
});

test('starter discounts rise smoothly and meet unmodified mature prices', () => {
  for (const [id, opening] of Object.entries(Core.Content.OPENING.discounts)) {
    const state = previousState(), definition = upgrade(id);
    let before = Core.upgradeCost(state, definition)[definition.resource];
    near(N.toNumber(before), opening.firstCost);
    for (let rank = 1; rank <= opening.untilRank + 3; rank += 1) {
      state.upgrades[id] = rank;
      const cost = Core.upgradeCost(state, definition)[definition.resource];
      assert.ok(N.cmp(cost, before) > 0, id + ' cost rises at rank ' + rank);
      if (rank >= opening.untilRank) {
        const extra = Math.max(0, rank - 8);
        const ordinary = N.mul(N.mul(definition.base, N.pow(definition.scale, rank)), N.pow(1.025, extra * extra / (1 + extra / 40)));
        near(ratio(cost, ordinary), 1);
      }
      before = cost;
    }
  }
});

test('ten minutes deliver three capstones, equipment, crew and persistent regional progression', () => {
  const result = guidedOpening(600), m = result.milestones;
  assert.ok(m.forge >= 180 && m.forge <= 260);
  assert.ok(m.hall >= 300 && m.hall <= 440);
  assert.equal(result.state.upgrades['gear-tools'], Math.floor((result.state.expedition.cleared + 2) / 3));
  assert.ok(result.purchases.length >= 50 && result.purchases.length <= 80);
  assert.ok(Math.max(...result.purchases.map((item, i, all) => item.second - (all[i - 1]?.second || 0))) <= 45);
  assert.equal(E.view(result.state).outposts.length, 3);
  assert.ok(result.state.expedition.index >= 3);
  assert.equal(result.state.lifetime.refits, 0);
});

test('all historical schemas preserve rates, prices, route distances and Forge cost', () => {
  for (const version of [1, 2, 3]) {
    const old = legacyState(version), original = clone(old), state = Core.migrateState(old);
    assert.ok(state); valid(state); assert.deepEqual(old, original);
    assert.equal(state.guild.grandfathered, true);
    assert.equal(Core.getView(state).expedition.local, true);
    delete state.expedition; // Compare retained historical guild formulas independently of the additive stage income.
    for (const rank of [0, 1, 8, 16, 30]) {
      state.upgrades.boots = rank;
      near(ratio(Core.getRates(state).travel, N.pow(1.28, rank)), 1);
      near(ratio(Core.getRates(state).gain.coins, N.mul(0.3, N.pow(1.18, rank))), 1);
      const def = upgrade('boots');
      const expected = N.mul(N.mul(def.base, N.pow(def.scale, rank)), N.pow(1.045, Math.pow(Math.max(0, rank - 8), 2)));
      near(ratio(Core.upgradeCost(state, def).coins, expected), 1);
    }
    for (let index = 0; index < 3; index += 1) assert.deepEqual(Core.getRoute(index, state).distance, N.from(Core.Content.ROUTES[index].distance));
    state.rooms.push('mine'); state.lifetime.highestRoute = 2;
    state.resources.ore = N.from(19);
    assert.equal(Core.act(state, { type: 'build-room', id: 'forge' }).ok, false);
    state.resources.ore = N.from(20);
    assert.equal(Core.act(state, { type: 'build-room', id: 'forge' }).ok, true);
    assert.equal(state.resources.ore.m, 0);
  }
});

test('new-save normalization preserves upgrades, resources, RNG and premium schedule', () => {
  const state = guidedOpening(120).state, snapshot = clone(state);
  const restored = Core.normalizeState(clone(state), state.lastUpdate);
  const retained = clone(restored); delete retained.upgradeTiers; delete retained.onboarding; delete retained.trailDeliveries;
  assert.deepEqual(retained, snapshot, 'Only additive knowledge and empty delivery tracking change the historical save');
  assert.equal(restored.trailDeliveries.sequence, 0);
  assert.equal(restored.trailDeliveries.lifetimeCoins.m, 0);
  assert.equal(Core.getView(restored).upgradeTiers.notice, null);
  assert.equal(Core.getView(restored).onboarding.notice, null);
  assert.deepEqual(Core.getRates(restored), Core.getRates(state));
  assert.deepEqual(Core.upgradeCost(restored, upgrade('boots')), Core.upgradeCost(state, upgrade('boots')));
  assert.deepEqual(state, snapshot);
  valid(restored);
});
