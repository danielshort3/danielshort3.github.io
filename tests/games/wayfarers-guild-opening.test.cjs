/** Opening economy acceptance checks; only the engine and Node built-ins are required. */
'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
const upgrade = id => Core.Content.UPGRADES.find(item => item.id === id);
const ratio = (a, b) => N.toNumber(N.div(a, b));
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, a + ' differs from ' + b);

function legacyState(version) {
  const state = Core.createState(0);
  state.schemaVersion = version;
  delete state.guild;
  delete state.introductions;
  if (version < 3) { delete state.luck; delete state.caravan; }
  else Core.Content.RELICS.filter(item => item.chapter > 0).forEach(item => delete state.luck.duplicateProgress[item.id]);
  if (version === 1) { delete state.premium; delete state.resources.starshards; }
  return state;
}

function guidedOpening(seconds) {
  const state = Core.createState(0), purchases = [], milestones = {};
  let room = 'trail';
  for (let second = 1; second <= seconds; second += 1) {
    Core.advance(state, 1);
    const goal = Core.getGoal(state);
    if (goal.action && goal.action.type === 'buy') room = upgrade(goal.action.id).room;
    if (goal.action && goal.action.type === 'build-room') room = 'mine';
    const buy = action => {
      if (!Core.act(state, action).ok) return;
      purchases.push({ second, id: action.id });
      if (milestones[action.id] === undefined) milestones[action.id] = second;
    };
    if (goal.ready && goal.action && ['buy', 'build-room'].includes(goal.action.type)) buy(goal.action);
    // Follow the highlighted goal and save for it. Otherwise buy from the two
    // cards in the selected room, without knowledge of any hidden upgrade.
    if (!goal.action || goal.action.type !== 'buy') {
      Core.getView(state).actions.filter(item => item.visible && item.room === room).slice(0, 2)
        .filter(item => !item.disabled && item.action.type === 'buy').forEach(item => buy(item.action));
    }
    state.rooms.forEach(id => { if (milestones[id] === undefined) milestones[id] = second; });
  }
  valid(state);
  return { state, purchases, milestones };
}

test('one opening control becomes affordable in five to ten seconds', () => {
  const state = Core.createState(0);
  assert.deepEqual(Core.getView(state).actions.filter(item => item.visible).map(item => item.id), ['boots']);
  Core.advance(state, 7);
  assert.equal(Core.act(state, { type: 'buy', id: 'boots' }).ok, false);
  Core.advance(state, 2);
  assert.equal(Core.act(state, { type: 'buy', id: 'boots' }).ok, true);
  assert.deepEqual(state.rooms, ['trail']);
  valid(state);
});

test('six useful purchases in two minutes precede more complex systems', () => {
  const state = Core.createState(0), purchases = [], base = Core.getRates(state);
  let mineAt = 0, bootsAtMine = 0;
  for (let second = 1; second <= 120; second += 1) {
    Core.advance(state, 1);
    for (const id of ['boots', 'miners']) {
      const before = Core.getRates(state);
      if (!Core.act(state, { type: 'buy', id }).ok) continue;
      const after = Core.getRates(state);
      purchases.push({ second, id });
      assert.ok(N.cmp(id === 'boots' ? after.travel : after.gain.ore, id === 'boots' ? before.travel : before.gain.ore) > 0);
    }
    if (state.rooms.includes('mine') && !mineAt) { mineAt = second; bootsAtMine = state.upgrades.boots; }
  }
  assert.ok(purchases.length >= 6);
  assert.ok(mineAt >= 30 && mineAt <= 90);
  assert.ok(bootsAtMine >= 3);
  assert.deepEqual(state.rooms, ['trail', 'mine']);
  assert.deepEqual(Core.getPresentation(state).resourceIds, ['coins', 'ore']);
  assert.ok(ratio(Core.getRates(state).gain.coins, base.gain.coins) >= 2.5);
  assert.ok(ratio(Core.getRates(state).travel, base.travel) >= 3.4);
  assert.equal(state.resources.starshards.m, 0);
  assert.equal(state.lifetime.refits, 0);
  valid(state);
});

test('per-purchase benefits decline smoothly to baseline, with bounded mature uplift', () => {
  const state = Core.createState(0);
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
    const state = Core.createState(0), definition = upgrade(id);
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

test('visible goal path teaches Mine, Forge and equipment within six minutes', () => {
  const result = guidedOpening(600), m = result.milestones;
  assert.ok(m.boots >= 5 && m.boots <= 10);
  assert.ok(result.purchases.filter(item => item.second <= 120).length >= 5);
  assert.ok(m.mine >= 30 && m.mine <= 90 && m.miners >= m.mine);
  assert.ok(m.forge >= 240 && m.forge <= 360);
  assert.ok(m['gear-tools'] > m.forge && m['gear-tools'] <= 360);
  assert.ok(m['gear-boots'] > m['gear-tools'] && m['gear-boots'] <= 360);
  assert.deepEqual(result.state.rooms, ['trail', 'mine', 'forge']);
  assert.equal(result.state.lifetime.refits, 0);
});

test('all historical schemas preserve rates, prices, route distances and Forge cost', () => {
  for (const version of [1, 2, 3]) {
    const old = legacyState(version), original = clone(old), state = Core.migrateState(old);
    assert.ok(state); valid(state); assert.deepEqual(old, original);
    assert.equal(state.guild.grandfathered, true);
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
  assert.deepEqual(restored, snapshot);
  assert.deepEqual(Core.getRates(restored), Core.getRates(state));
  assert.deepEqual(Core.upgradeCost(restored, upgrade('boots')), Core.upgradeCost(state, upgrade('boots')));
  assert.deepEqual(state, snapshot);
  valid(restored);
});
