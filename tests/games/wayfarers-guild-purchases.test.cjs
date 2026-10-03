'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const P = require('../../js/games/wayfarers-guild/progression.js');
const Purchases = require('../../js/games/wayfarers-guild/progression-purchases.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
const dependencies = { Content: Core.Content, upgradeCost: Core.upgradeCost, batchModes: P.batchModes, isOpen: () => true };
function guild() {
  const state = Core.createState(0);
  state.lifetime.refits = 10; state.lifetime.charters = 1; state.chapter.refits = 10;
  state.premium.claimedMilestones = ['first-refit', 'first-charter'];
  for (const id of Object.keys(state.resources)) state.resources[id] = N.from('1e80');
  return state;
}
function buy(state, action, count) {
  const quote = Purchases.quote(state, action, count, dependencies);
  return Purchases.commit(state, { ...action, count, quote: quote.token }, dependencies);
}

for (const action of [{ type: 'buy', id: 'boots' }, { type: 'buy', id: 'gear-instruments' }, { type: 'refit-upgrade', id: 'pace' }, { type: 'legacy-upgrade', id: 'foundations' }]) {
  test('exact guild batch equals individual prices for ' + action.type + ':' + action.id, () => {
    const bulk = guild();
    const quote = Purchases.quote(bulk, action, 10, dependencies), total = {};
    for (const [id, amount] of Object.entries(quote.costs)) bulk.resources[id] = N.mul(amount, 2);
    const singles = clone(bulk);
    for (let index = 0; index < 10; index += 1) {
      const next = Purchases.quote(singles, action, 1, dependencies);
      for (const [id, amount] of Object.entries(next.costs)) total[id] = N.add(total[id] || 0, amount);
      assert.ok(buy(singles, action, 1).ok);
    }
    assert.deepEqual(quote.costs, total);
    const result = buy(bulk, action, 10);
    assert.equal(result.ok, true);
    assert.equal(result.quantity, 10);
    for (const map of ['upgrades', 'refitUpgrades', 'legacy']) assert.deepEqual(bulk[map], singles[map]);
    for (const [id, amount] of Object.entries(quote.costs)) {
      assert.ok(Math.abs(N.toNumber(N.div(bulk.resources[id], amount)) - 1) < 1e-10, 'exact aggregate price was spent');
      assert.ok(Math.abs(N.toNumber(N.div(bulk.resources[id], singles.resources[id])) - 1) < 1e-10, 'same spend as ten ordinary purchases');
    }
    assert.equal(bulk.expedition.revision, 1, 'one transaction, not ten intermediate commits');
  });
}

test('multi-resource shortage, stale price or disabled entitlement leave every economic field unchanged', () => {
  const state = guild(), action = { type: 'buy', id: 'gear-instruments' };
  state.resources.herbs = N.zero();
  const before = clone(state);
  assert.equal(buy(state, action, 5).ok, false);
  assert.deepEqual(state, before);
  state.resources.herbs = N.from('1e80');
  const quote = Purchases.quote(state, action, 5, dependencies);
  state.upgrades.forge += 1; // A price-changing purchase must invalidate the old quote even before a revision update.
  const stale = clone(state);
  assert.equal(Purchases.commit(state, { ...action, count: 5, quote: quote.token }, dependencies).ok, false);
  assert.deepEqual(state, stale);
  state.lifetime.refits = 0; state.lifetime.charters = 0;
  const locked = clone(state);
  assert.equal(buy(state, action, 5).ok, false);
  assert.deepEqual(state, locked);
});

test('whole batch affordability has an exact boundary and never makes a partial purchase', () => {
  const state = guild(), action = { type: 'refit-upgrade', id: 'pace' };
  const quote = Purchases.quote(state, action, 5, dependencies);
  state.resources.notes = N.sub(quote.costs.notes, .01);
  const before = clone(state);
  assert.equal(buy(state, action, 5).ok, false);
  assert.deepEqual(state, before);
  state.resources.notes = quote.costs.notes;
  assert.ok(buy(state, action, 5).ok);
  assert.equal(state.refitUpgrades.pace, 5);
  assert.equal(N.cmp(state.resources.notes, 0), 0);
});

test('invalid counts, one-time actions and batches crossing a rank limit are rejected atomically', () => {
  for (const count of [0, -1, 7, 1000, 5.5, NaN, Infinity, '5']) {
    const state = guild(), before = clone(state);
    assert.equal(buy(state, { type: 'buy', id: 'boots' }, count).ok, false);
    assert.deepEqual(state, before);
  }
  for (const type of ['research', 'premium-buy', 'refit', 'charter', 'expedition-development']) {
    const state = guild(), before = clone(state);
    assert.equal(buy(state, { type, id: 'pace' }, 5).ok, false);
    assert.deepEqual(state, before);
  }
  const state = guild(); state.upgrades.boots = 999998;
  const before = clone(state);
  assert.equal(buy(state, { type: 'buy', id: 'boots' }, 5).ok, false);
  assert.deepEqual(state, before);
});

test('the opening and each lifetime milestone expose only earned batch modes', () => {
  const state = Core.createState(0);
  const modes = () => P.batchModes(state).filter(item => item.unlocked).map(item => item.count);
  assert.deepEqual(modes(), [1]);
  state.lifetime.refits = 1; assert.deepEqual(modes(), [1, 5]);
  state.lifetime.refits = 3; assert.deepEqual(modes(), [1, 5, 10]);
  state.lifetime.charters = 1; assert.deepEqual(modes(), [1, 5, 10, 25]);
  state.lifetime.refits = 10; assert.deepEqual(modes(), [1, 5, 10, 25, 100]);
  state.chapter.refits = 0; assert.deepEqual(modes(), [1, 5, 10, 25, 100]);
});

test('the canonical catalog quotes and executes five guild ranks with the actual resulting output', () => {
  const state = guild();
  assert.ok(Core.act(state, { type: 'expedition-batch', count: 5 }).ok);
  const before = Core.getRates(state);
  const offer = Core.getView(state).globalUpgrades.find(item => item.action.type === 'buy' && item.action.id === 'boots');
  assert.ok(offer);
  assert.equal(offer.quantity, 5);
  assert.equal(offer.rankAfter, 5);
  assert.equal(offer.action.count, 5);
  const preview = offer.impact.find(item => item.metric === 'guild:coins');
  assert.ok(preview, 'the preview must describe actual wallet output');
  const result = Core.act(state, offer.action);
  assert.equal(result.quantity, 5);
  assert.equal(state.upgrades.boots, 5);
  const after = Core.getRates(state);
  assert.ok(N.cmp(after.gain.coins, before.gain.coins) > 0);
  assert.deepEqual(preview.nextValue, N.sub(after.gain.coins, after.drain.coins));
  const snapshot = clone(state);
  assert.equal(Core.act(state, offer.action).ok, false, 'a used quote cannot buy again');
  assert.deepEqual(state, snapshot);
});

test('five canonical equipment ranks award five ranks of Forge experience in one action', () => {
  const state = guild();
  state.rooms = Core.Content.ROOMS.map(room => room.id);
  state.lifetime.highestRoute = 10;
  assert.ok(Core.act(state, { type: 'expedition-batch', count: 5 }).ok);
  const before = state.mastery.forge;
  const offer = Core.getView(state).globalUpgrades.find(item => item.action.type === 'buy' && item.action.id === 'gear-instruments');
  assert.ok(offer);
  const result = Core.act(state, offer.action);
  assert.equal(result.ok, true);
  assert.equal(state.upgrades['gear-instruments'], 5);
  assert.deepEqual(state.mastery.forge, N.add(before, 150));
});
