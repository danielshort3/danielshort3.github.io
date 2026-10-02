/** Expedition acceptance: real core actions and storage, no browser or fixture dependency. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const E = require('../../js/games/wayfarers-guild/expeditions.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
const near = (a, b, tolerance = 1e-6) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(a), Math.abs(b)) * tolerance, `${a} != ${b}`);
function advance(state, seconds) {
  let remaining = seconds;
  while (remaining > 1e-7) { const result = Core.advance(state, remaining); assert.ok(result.seconds > 0); remaining = result.pendingSeconds || 0; }
}
function play(seconds, onTick) {
  const state = Core.createState(0), purchases = [], stages = [], snapshots = {};
  for (let second = 1; second <= seconds; second += 1) {
    advance(state, 1);
    let view = E.view(state);
    if (view.stage.completed) {
      stages.push({ second, index: state.expedition.index, purchases: state.expedition.stagePurchases });
      snapshots[state.expedition.index] = clone(state);
      assert.ok(Core.act(state, view.next.action).ok);
      view = E.view(state);
    }
    if (onTick) onTick(state, second);
    view.cards.forEach(card => { if (N.cmp(card.cost[0].amount, 1e6) < 0) near(N.toNumber(card.cost[0].amount), Math.round(N.toNumber(card.cost[0].amount)), 1e-10); });
    const offer = view.cards.filter(card => card.visible && !card.disabled).sort((a, b) => N.cmp(a.cost[0].amount, b.cost[0].amount))[0];
    if (offer && Core.act(state, offer.action).ok) purchases.push({ second, id: offer.id });
  }
  valid(state);
  return { state, purchases, stages, snapshots };
}
const opening = play(600);
function stage(index) {
  if (!index) return Core.createState(0);
  const state = clone(opening.snapshots[index - 1]);
  assert.ok(Core.act(state, { type: 'expedition-next' }).ok);
  return state;
}
function same(a, b) {
  valid(a); valid(b);
  assert.equal(a.expedition.index, b.expedition.index);
  assert.equal(a.expedition.completed, b.expedition.completed);
  assert.deepEqual(a.expedition.ranks, b.expedition.ranks);
  assert.deepEqual(a.expedition.mastery, b.expedition.mastery);
  assert.equal(a.expedition.purchases, b.expedition.purchases);
  for (const key of ['work', 'finaleWork', 'elapsed']) near(a.expedition[key], b.expedition[key]);
  for (const key of ['ore', 'smelt']) near(a.expedition.buffers[key], b.expedition.buffers[key]);
  Object.keys(a.resources).forEach(key => near(N.toNumber(N.div(N.max(a.resources[key], 1), N.max(b.resources[key], 1))), 1));
  assert.equal(a.luck.rng, b.luck.rng);
  assert.equal(a.premium.rng, b.premium.rng);
  near(a.premium.eligibleSeconds, b.premium.eligibleSeconds);
}

test('one opening control becomes three tracks only after learning; buys charge the shared wallet', () => {
  const state = stage(0);
  assert.deepEqual(E.view(state).cards.filter(card => card.visible).map(card => card.id), ['boots']);
  assert.equal(Core.act(state, { type: 'expedition-buy', id: 'porters' }).ok, false);
  advance(state, 7);
  const before = N.toNumber(state.resources.coins), speed = E.localRates(state).travel;
  assert.ok(Core.act(state, { type: 'expedition-buy', id: 'boots' }).ok);
  near(N.toNumber(state.resources.coins), before - 6);
  assert.ok(E.localRates(state).travel > speed);
  assert.equal(Core.act(state, { type: 'expedition-next' }).ok, false);
  assert.equal(state.upgrades.boots, 0, 'local and guild upgrades have separate retained lifecycles');
  valid(state);
});

test('visible-only purchases reach three different capstones in eight minutes with no long opening gap', () => {
  assert.equal(opening.purchases[0].second, 7);
  assert.deepEqual(opening.stages.map(item => item.index), [0, 1, 2]);
  const [first, quarry, tower] = opening.stages;
  assert.ok(first.second >= 75 && first.second <= 105);
  assert.ok(quarry.second >= 210 && quarry.second <= 285);
  assert.ok(tower.second >= 420 && tower.second <= 510);
  assert.ok(opening.purchases.length >= 25 && opening.purchases.length <= 35);
  assert.ok(Math.max(...opening.purchases.map((item, i, all) => item.second - (all[i - 1]?.second || 0))) <= 45);
  assert.ok(opening.stages.every(item => item.purchases >= 8));
  assert.ok(opening.state.rooms.includes('hall'));
  assert.deepEqual(opening.state.crew.owned.slice(0, 3), ['scout', 'prospector', 'quartermaster']);
  assert.ok(opening.state.expedition.index >= 3, 'new regional mechanics continue after the tutorial');
});

test('each capstone settles the canonical route once and waits safely for manual next', () => {
  const state = clone(opening.snapshots[1]), before = clone(state);
  assert.equal(state.route.index, 2);
  assert.equal(state.lifetime.highestRoute, 1);
  assert.equal(state.upgrades['gear-tools'], 1);
  assert.equal(E.view(state).outposts.length, 2);
  assert.equal(Core.act(state, { type: 'expedition-complete' }).ok, false);
  advance(state, 3600);
  assert.equal(state.expedition.index, 1);
  assert.equal(state.route.index, before.route.index);
  assert.deepEqual(state.expedition.mastery, before.expedition.mastery);
  assert.equal(state.upgrades['gear-tools'], 1);
  assert.ok(N.cmp(state.resources.coins, before.resources.coins) > 0, 'productive outposts keep earning while parked');
  assert.ok(Core.act(state, { type: 'expedition-next' }).ok);
  assert.equal(state.expedition.index, state.route.index);
  assert.equal(state.expedition.stagePurchases, 0);
  valid(state);
});

test('short and supply paths trade speed for investment; the next region changes the best route', () => {
  const state = stage(0); advance(state, 30);
  const short = E.localRates(state);
  assert.ok(Core.act(state, { type: 'expedition-choice', id: 'supply' }).ok);
  const supply = E.localRates(state);
  assert.ok(short.travel > supply.travel);
  assert.ok(N.cmp(supply.income, short.income) > 0);
  const later = stage(3), rough = E.localRates(later).travel;
  assert.ok(Core.act(later, { type: 'expedition-choice', id: 'supply' }).ok);
  assert.ok(E.localRates(later).travel > rough, 'supply avoids the new rough direct path');
  assert.match(E.view(later).condition, /Rough direct path/);
});

test('Quarry upgrades expose actual bottlenecks, buffers and a throughput-versus-value choice', () => {
  const state = stage(1); state.resources.coins = N.from(10000);
  const before = E.view(state), pick = before.cards.find(card => card.id === 'picks');
  assert.equal(before.scene.bottleneck, 'furnace');
  assert.equal(pick.chainOutput.current, pick.chainOutput.next, 'extra extraction alone cannot speed the constrained furnace');
  assert.ok(Core.act(state, pick.action).ok);
  advance(state, 50);
  const full = E.view(state);
  assert.ok(full.scene.oreBuffer <= full.scene.capacity + 1e-8);
  assert.ok(full.scene.flows.picks < full.scene.rates.picks, 'full ore buffer throttles actual mining');
  const fast = E.localRates(state);
  assert.ok(Core.act(state, { type: 'expedition-choice', id: 'quality' }).ok);
  const valuable = E.localRates(state);
  assert.ok(valuable.furnace < fast.furnace);
  assert.ok(N.cmp(valuable.income, fast.income) > 0);
  assert.ok(Core.act(state, { type: 'expedition-choice', id: 'equipment-boots' }).ok);
  advance(state, 10000);
  assert.equal(state.upgrades['gear-boots'], 1);
  assert.equal(state.upgrades['gear-tools'], 0);
  valid(state);
});

test('Watchtower allocations change repair versus beacon output; unattended pressure never removes work', () => {
  const state = stage(2); state.expedition.ranks.crew = 1; state.expedition.stagePurchases = 1; state.expedition.purchases += 1;
  state.expedition.elapsed = 70;
  const balanced = E.localRates(state);
  assert.ok(Core.act(state, { type: 'expedition-choice', id: 'protect' }).ok);
  const protection = E.localRates(state);
  assert.ok(balanced.repair > protection.repair);
  assert.ok(protection.beacon > balanced.beacon, 'protection has a real finale role under pressure');
  const work = state.expedition.work;
  advance(state, 40);
  assert.ok(state.expedition.work > work);
  valid(state);
});

test('persistent choices support actual local work, retain paid effects and disclose their causal rates', () => {
  const crossing = stage(3), base = E.localRates(crossing).travel;
  crossing.upgrades['gear-boots'] += 1;
  assert.ok(E.localRates(crossing).travel > base);
  const unpaid = E.localRates(crossing).travel, serialized = JSON.stringify(crossing);
  assert.ok(Core.setPremiumEntitlements(crossing, ['compass']).ok);
  near(E.localRates(crossing).travel / unpaid, 1.1);
  assert.equal(JSON.stringify(crossing), serialized);
  const descriptor = Core.getView(crossing).actions.find(item => item.id === 'gear-boots');
  assert.match(descriptor.effectText, /Crossing speed/);
  assert.doesNotMatch(descriptor.description, /35% travel/);
  const quarry = stage(1), before = E.localRates(quarry);
  quarry.crew.specialists[0] = 'prospector';
  assert.ok(E.localRates(quarry).picks > before.picks);
  quarry.research.push('efficient-smelting');
  assert.ok(E.localRates(quarry).furnace > before.furnace);
});

test('blueprints, mastery, outposts and automation survive both existing prestige resets', () => {
  const state = stage(3);
  assert.ok(Core.act(state, { type: 'expedition-blueprint', id: 'engineering' }).ok);
  assert.equal(Core.act(state, { type: 'expedition-blueprint', id: 'caravan' }).ok, false);
  assert.ok(Core.act(state, { type: 'expedition-automation', enabled: true, priority: 'progress', dispatch: true }).ok);
  // This first Refit is reachable using exactly the earned Quarry ore.
  assert.ok(Core.act(state, { type: 'buy', id: 'gear-boots' }).ok);
  assert.ok(Core.getRefitPreview(state).available);
  const retained = clone(state.expedition), premium = clone(state.premium);
  assert.ok(Core.act(state, { type: 'refit' }).ok);
  for (const key of ['cleared', 'blueprints', 'mastery', 'automation', 'sequence', 'seen', 'recent', 'purchases']) assert.deepEqual(state.expedition[key], retained[key]);
  assert.equal(state.expedition.stagePurchases, 0);
  assert.equal(state.expedition.work, 0);
  assert.equal(state.premium.rng, premium.rng);
  assert.ok(E.contribution(state).ore.m > 0);
  valid(state);
  // A self-contained, valid funded chapter checkpoint exercises the deeper reset.
  // Its timeline is deliberately not a pacing assertion.
  state.lifetime.highestRoute = 8; state.run.completed = 8; state.route.index = 9;
  state.expedition.cleared = 8; E.restart(state, 9);
  state.run.work = N.from(2e16); state.chapter.work = N.from(2e16); state.lifetime.work = N.from(2e16);
  state.guild.chapterProject.choice = 'industry';
  valid(state);
  const beforeCharter = clone(state.expedition);
  assert.ok(Core.getCharterPreview(state).available);
  assert.ok(Core.act(state, { type: 'charter' }).ok);
  for (const key of ['cleared', 'blueprints', 'mastery', 'automation']) assert.deepEqual(state.expedition[key], beforeCharter[key]);
  assert.equal(state.expedition.index, 0);
  assert.equal(state.upgrades['gear-boots'], 0);
  valid(state);
});

test('field-missing v4 and historical migrations join the current stage without losing stocks or RNG', () => {
  const old = clone(opening.state); delete old.expedition;
  old.rooms = old.rooms.filter(id => id !== 'hall'); // Earlier v4 did not grant this room before route 6.
  old.crew.owned = []; old.crew.specialists = [null, null];
  const before = clone(old), upgraded = Core.normalizeState(old, old.lastUpdate);
  assert.deepEqual(old, before);
  assert.equal(upgraded.expedition.index, old.route.index);
  near(E.progress(upgraded.expedition), N.toNumber(N.div(old.route.progress, Core.getRoute(old.route.index, old).distance)));
  for (const key of Object.keys(old)) assert.deepEqual(upgraded[key], old[key]);
  valid(upgraded);
  for (const version of [1, 2, 3]) {
    const historical = Core.createState(0); delete historical.expedition; delete historical.guild; delete historical.introductions;
    historical.schemaVersion = version;
    if (version < 3) { delete historical.luck; delete historical.caravan; }
    else Core.Content.RELICS.filter(item => item.chapter > 0).forEach(item => delete historical.luck.duplicateProgress[item.id]);
    if (version === 1) { delete historical.premium; delete historical.resources.starshards; }
    const migrated = Core.migrateState(historical);
    assert.ok(migrated); assert.equal(Core.getView(migrated).expedition.local, true); valid(migrated);
  }
});

test('malformed supplied expedition data is rejected instead of quietly repaired', () => {
  const mutations = [s => { s.expedition = null; }, s => { s.expedition.ranks.boots = -1; }, s => { s.expedition.buffers.ore = 999; }, s => { s.expedition.index += 1; }, s => { s.expedition.extra = 1; }, s => { s.expedition.cleared += 1; }, s => { s.expedition.automation.enabled = true; }, s => { s.expedition.stagePurchases = 2; }, s => { s.expedition.finaleWork = 1; }];
  for (const mutate of mutations) { const state = stage(0); mutate(state); assert.equal(Core.validateState(state).valid, false); }
});

test('offline boundaries match partitioned foreground across finite buffers, hazards and manual stage waiting', () => {
  for (const index of [0, 1, 2]) {
    const a = stage(index), b = clone(a);
    advance(a, 4200);
    for (let t = 0; t < 4200; t += 13.7) advance(b, Math.min(13.7, 4200 - t));
    same(a, b);
    assert.equal(a.expedition.index, index, 'absence never presses manual next');
  }
});

test('a catch-up that opens the Forge cannot retroactively earn premium eligibility', () => {
  const state = stage(1), before = state.premium.eligibleSeconds;
  assert.ok(!state.rooms.includes('forge'));
  advance(state, 1000);
  assert.ok(state.rooms.includes('forge'));
  assert.ok(state.premium.eligibleSeconds > before);
  assert.ok(state.premium.eligibleSeconds - before < 800, 'only the post-capstone portion can count');
  near(state.premium.eligibleSeconds - before, 1000 - state.expedition.elapsed, 1e-5);
});

test('offline automation and dispatch match live partitions, including a save/reload halfway', () => {
  const a = stage(3);
  assert.ok(Core.act(a, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: true }).ok);
  const b = clone(a);
  advance(a, 3600);
  for (let t = 0; t < 3600; t += 31.93) advance(b, Math.min(31.93, 3600 - t));
  same(a, b);
  const restored = Core.normalizeState(b, b.lastUpdate);
  advance(a, 3600); advance(restored, 3600); same(a, restored);
  assert.ok(a.expedition.index > 3);
});

test('save/export/backup preserve local progress and rare-reward schedules without rerolls', () => {
  const values = new Map(), state = clone(opening.state);
  const store = Storage.createStore({ core: Core, now: () => state.lastUpdate, storage: { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) } });
  assert.ok(store.save(state).ok);
  const text = store.export(state).text;
  const restored = store.replaceImport(text).state;
  assert.deepEqual(restored, state);
  assert.deepEqual(store.load({ deferOffline: true }).state, state);
  const previous = clone(state); advance(state, 10); assert.ok(store.save(state).ok);
  assert.deepEqual(JSON.parse(values.get(Storage.BACKUP_KEY)).state, previous);
});
