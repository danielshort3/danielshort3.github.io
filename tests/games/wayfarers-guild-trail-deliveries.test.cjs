'use strict';
const {createReleasedState}=require('./helpers/wayfarers-released.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const D = require('../../js/games/wayfarers-guild/trail-deliveries.js');
const N = require('../../js/games/wayfarers-guild/numbers.js');
const clone = value => JSON.parse(JSON.stringify(value));
const rates = { coins: N.from(10), travel: 1 };
function state() {
  const result = { run: { id: 1 }, resources: { coins: N.zero() }, lifetime: { coins: N.zero() }, expedition: { version: 3, cleared: 0 } };
  result.trailDeliveries = D.initial(result);
  return result;
}
const near = (a, b) => assert.ok(Math.abs(N.toNumber(a) - N.toNumber(b)) < 1e-6, `${N.format(a)} != ${N.format(b)}`);

test('a delivery pays at the real arrival boundary, dwells there, then begins a new trip', () => {
  const s = state();
  D.tick(s, 29, rates);
  assert.equal(s.resources.coins.m, 0);
  assert.equal(D.view(s, rates).progress, 29 / 30);
  assert.equal(D.nextEvent(s, rates), 1);
  D.tick(s, 1, rates);
  near(s.resources.coins, 120);
  near(s.lifetime.coins, 120);
  assert.equal(s.trailDeliveries.deliveries, 1);
  assert.equal(D.view(s, rates).phase, 'arrived');
  D.tick(s, 1.2, rates);
  assert.equal(D.view(s, rates).progress, 0);
  near(s.resources.coins, 120);
  assert.ok(D.validate(s.trailDeliveries, s));
});

test('offline batching and many small frames pay the same cargo without replaying arrivals after reload', () => {
  const a = state(), b = state();
  D.tick(a, 3600, rates);
  for (let i = 0; i < 36000; i += 1) D.tick(b, 0.1, rates);
  near(a.resources.coins, b.resources.coins);
  assert.equal(a.trailDeliveries.deliveries, b.trailDeliveries.deliveries);
  assert.ok(Math.abs(a.trailDeliveries.work - b.trailDeliveries.work) < 1e-5);
  const loaded = clone(a), before = clone(a.resources.coins);
  D.view(loaded, rates); D.tick(loaded, 0, rates);
  near(loaded.resources.coins, before);
  assert.ok(D.validate(loaded.trailDeliveries, loaded));
});

test('whole-trip batching preserves partial cargo, final rest and each minimum award', () => {
  for (const start of [13, 30.5]) {
    const a = state();
    D.tick(a, start, rates);
    const b = clone(a), low = { coins: N.zero(), travel: 1 };
    D.tick(a, 3600.25, low);
    for (let i = 0; i < 14401; i += 1) D.tick(b, 0.25, low);
    near(a.resources.coins, b.resources.coins);
    near(a.trailDeliveries.cargo, b.trailDeliveries.cargo);
    near(a.trailDeliveries.lastReward, 6);
    assert.equal(a.trailDeliveries.deliveries, b.trailDeliveries.deliveries);
    assert.equal(a.trailDeliveries.phase, b.trailDeliveries.phase);
    assert.ok(Math.abs(a.trailDeliveries.work - b.trailDeliveries.work) < 1e-5);
    assert.ok(Math.abs(a.trailDeliveries.hold - b.trailDeliveries.hold) < 1e-5);
  }
  const s = state(), seconds = 400 * 86400;
  D.tick(s, seconds, { coins: N.zero(), travel: 1 });
  const count = Math.floor((seconds + 1.2) / 31.2);
  assert.equal(s.trailDeliveries.deliveries, count);
  assert.equal(s.trailDeliveries.sequence, count);
  near(s.resources.coins, count * 6);
  near(s.trailDeliveries.lastReward, 6);
  assert.ok(D.validate(s.trailDeliveries, s));
});

test('a 400-day saved absence settles fully without a new delivery-induced event cap or repeated reward', () => {
  const Core = require('../../js/games/wayfarers-guild/core.js');
  const Storage = require('../../js/games/wayfarers-guild/persistence.js');
  const start = 1700000000000, elapsed = 400 * 86400000 + 1250;
  const original = createReleasedState(start);
  original.resources.coins = N.from(1);
  const expected = clone(original), result = Core.advanceTo(expected, start + elapsed);
  assert.equal(result.seconds, elapsed / 1000);
  assert.equal(result.pendingSeconds || 0, 0);
  assert.ok(expected.trailDeliveries.deliveries > 100000);
  const values = new Map([[Storage.SAVE_KEY, JSON.stringify({ format: Storage.FORMAT, version: Storage.VERSION, savedAt: start, state: original })]]);
  const store = Storage.createStore({ core: Core, now: () => start + elapsed,
    storage: { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) } });
  const loaded = store.load();
  assert.equal(loaded.status, 'loaded');
  assert.equal(loaded.offline.seconds, elapsed / 1000);
  assert.deepEqual(loaded.state, expected);
  assert.ok(store.save(loaded.state).ok);
  const again = store.load();
  assert.equal(again.offline.seconds, 0);
  assert.deepEqual(again.state.trailDeliveries, loaded.state.trailDeliveries);
  assert.deepEqual(Core.validateState(again.state), { valid: true, errors: [] });
});

test('cargo accrues at the income earned during the trip, not a rate chosen at the endpoint', () => {
  const s = state();
  D.tick(s, 29, { coins: N.from(1), travel: 1 });
  D.tick(s, 1, { coins: N.from(100), travel: 1 });
  near(s.resources.coins, 51.6);
});

test('fractional event tails still contribute work and cargo', () => {
  const s = state();
  for (let i = 0; i < 100; i += 1) D.tick(s, 1e-10, rates);
  assert.ok(Math.abs(s.trailDeliveries.work - 1e-8) < 1e-20);
  assert.ok(s.trailDeliveries.cargo.m > 0);
});

test('Trail must be established; changing away or returning cannot itself award coins', () => {
  const s = state(); s.expedition.cleared = -1;
  D.tick(s, 1000, rates);
  assert.equal(s.resources.coins.m, 0); assert.equal(D.nextEvent(s, rates), Infinity);
  s.expedition.cleared = 0;
  D.tick(s, 30, rates);
  assert.equal(s.trailDeliveries.deliveries, 1);
  const old = clone(s.resources.coins);
  for (let i = 0; i < 10; i += 1) D.view(s, rates);
  assert.deepEqual(s.resources.coins, old);
  s.expedition = { version: 2, areas: { greenway: { established: true } } };
  assert.equal(D.view(s, rates).active, true);
});

test('each completed Trail landmark pays once, historical migration pays nothing, real new runs can earn again', () => {
  const s = state();
  near(D.landmark(s, 0, 'greenway', rates), 300);
  near(D.landmark(s, 0, 'greenway', rates), 0);
  near(D.landmark(s, 1, 'quarry', rates), 0);
  near(s.resources.coins, 300);
  s.run.id += 1;
  near(D.landmark(s, 0, 'greenway', rates), 300);
  const migrated = state(); migrated.expedition.cleared = 6;
  migrated.trailDeliveries = D.initial(migrated, true);
  near(D.landmark(migrated, 6, 'greenway', rates), 0);
  near(migrated.resources.coins, 0);
  near(D.landmark(migrated, 9, 'greenway', rates), 300);
});

test('arrival gives at least six coins, faster travel is bounded, invalid saved ledgers fail closed', () => {
  const s = state();
  D.tick(s, 3.75, { coins: N.zero(), travel: 1e100 });
  near(s.resources.coins, 6);
  assert.equal(s.trailDeliveries.phase, 'arrived');
  assert.ok(D.validate(s.trailDeliveries, s));
  for (const mutate of [x => { x.work = NaN; }, x => { x.extra = 0; }, x => { x.hold = 0; }, x => { x.cargo = N.from(1); }, x => { x.landmarkRunId = 2; }, x => { x.lastReward = N.from(1000); }]) {
    const bad = clone(s.trailDeliveries); mutate(bad);
    assert.equal(D.validate(bad, s), false);
  }
});

test('prestige clears unearned cargo but retains the delivery history', () => {
  const s = state();
  D.tick(s, 45, rates);
  const count = s.trailDeliveries.deliveries, lifetime = clone(s.trailDeliveries.lifetimeCoins);
  assert.ok(s.trailDeliveries.cargo.m > 0);
  D.resetTrip(s);
  assert.equal(s.trailDeliveries.work, 0); assert.equal(s.trailDeliveries.cargo.m, 0);
  assert.equal(s.trailDeliveries.deliveries, count); assert.deepEqual(s.trailDeliveries.lifetimeCoins, lifetime);
  assert.ok(D.validate(s.trailDeliveries, s));
});

test('Core uses numeric Trail capacity and pays precisely at its next delivery boundary', () => {
  const { Core, mature, advance } = require('./helpers/wayfarers-progression.cjs');
  const s = mature();
  D.resetTrip(s);
  const r = Core.getTrailDeliveryRates(s);
  assert.equal(typeof r.travel, 'number'); assert.ok(r.travel > 0);
  const eta = D.nextEvent(s, r), count = s.trailDeliveries.deliveries;
  advance(s, eta - 1e-5);
  assert.equal(s.trailDeliveries.deliveries, count);
  advance(s, 2e-5);
  assert.equal(s.trailDeliveries.deliveries, count + 1);
  const v = Core.getView(s).expedition.delivery;
  assert.equal(v.phase, 'arrived'); assert.equal(v.progress, 1);
  assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
});

test('Core foreground and offline partitions agree, independently of the viewed area', () => {
  const { Core, mature, advance } = require('./helpers/wayfarers-progression.cjs');
  const a = mature(), b = clone(a), c = clone(a);
  assert.ok(Core.act(c, { type: 'expedition-select', areaId: 'quarry' }).ok);
  advance(a, 300); advance(c, 300);
  for (let i = 0; i < 300; i += 1) advance(b, 1);
  for (const s of [b, c]) {
    assert.equal(s.trailDeliveries.deliveries, a.trailDeliveries.deliveries);
    const av = N.toNumber(a.trailDeliveries.lifetimeCoins), bv = N.toNumber(s.trailDeliveries.lifetimeCoins);
    assert.ok(Math.abs(av - bv) <= Math.max(1, av) * 1e-7, `${av} != ${bv}`);
    assert.ok(Math.abs(s.trailDeliveries.work - a.trailDeliveries.work) < 1e-5);
    assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
  }
});

test('first real Trail completion adds a landmark bonus before repeat deliveries begin', () => {
  const { Core, fund, advance } = require('./helpers/wayfarers-progression.cjs');
  const s = createReleasedState(0); fund(s);
  assert.ok(Core.act(s, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' }).ok);
  for (let t = 0; !s.expedition.completed && t < 2000; t += 1) advance(s, 1);
  assert.equal(s.expedition.completed, true);
  assert.equal(s.trailDeliveries.deliveries, 0);
  assert.equal(s.trailDeliveries.sequence, 1);
  assert.equal(s.trailDeliveries.lastKind, 'landmark');
  assert.ok(N.cmp(s.trailDeliveries.lastReward, 24) >= 0);
  const before = clone(s.trailDeliveries);
  Core.getView(s); Core.getView(s); advance(s, 0.01);
  assert.equal(s.trailDeliveries.sequence, before.sequence);
  assert.deepEqual(s.trailDeliveries.lastReward, before.lastReward);
});

test('released E2 migration adds empty delivery tracking without changing its economy or paying historical milestones', () => {
  const Core = require('../../js/games/wayfarers-guild/core.js');
  const old = clone(require('./fixtures/wayfarers-v4-state.json'));
  const s = Core.normalizeState(old, old.lastUpdate);
  assert.equal(s.expedition.version, 2);
  assert.deepEqual(s.resources, old.resources);
  assert.equal(s.trailDeliveries.deliveries, 0); assert.equal(s.trailDeliveries.sequence, 0);
  const resources = clone(s.resources);
  Core.getView(s); Core.getView(s);
  assert.deepEqual(s.resources, resources);
  assert.equal(typeof Core.getTrailDeliveryRates(s).travel, 'number');
  assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
});

test('actual Refit and Charter keep delivery receipts but discard unfinished ordinary cargo', () => {
  const { Core, mature, advance } = require('./helpers/wayfarers-progression.cjs');
  for (const kind of ['refit', 'charter']) {
    const s = mature();
    if (kind === 'charter') {
      for (let elapsed = 0; !Core.getCharterPreview(s).available && elapsed < 86400; elapsed += 60) {
        if (s.expedition.completed) assert.ok(Core.act(s, { type: 'expedition-next' }).ok);
        advance(s, 60);
      }
      assert.ok(Core.getCharterPreview(s).available);
    }
    D.resetTrip(s); advance(s, 0.1);
    assert.ok(s.trailDeliveries.cargo.m > 0);
    const before = clone(s.trailDeliveries);
    assert.ok(Core.act(s, { type: kind }).ok);
    assert.equal(s.trailDeliveries.cargo.m, 0); assert.equal(s.trailDeliveries.work, 0);
    for (const key of ['deliveries', 'lifetimeCoins', 'sequence', 'landmarkRunId', 'landmarkIndex']) assert.deepEqual(s.trailDeliveries[key], before[key]);
    assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
  }
});
