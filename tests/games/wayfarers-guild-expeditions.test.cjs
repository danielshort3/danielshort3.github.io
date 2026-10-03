/** Expedition acceptance: real core actions and storage, no browser or fixture dependency. */
'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const CurrentCore = require('../../js/games/wayfarers-guild/core.js');
// These witnesses cover a released v4 run before explicit economy adoption.
const Core = Object.assign({}, CurrentCore, { createState(now = 0) { const state = CurrentCore.migrateState(require('./fixtures/wayfarers-v4-fresh.json')); state.createdAt = now; state.lastUpdate = now; return state; } });
const E = require('../../js/games/wayfarers-guild/expeditions.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
const area = (state, id = state.expedition.selectedArea) => state.expedition.areas[id];
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
    if (state.expedition.completed) {
      stages.push({ second, index: state.expedition.index, purchases: area(state).purchases });
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
  for (const id of Object.keys(a.expedition.areas)) {
    assert.deepEqual(area(a, id).ranks, area(b, id).ranks);
    assert.deepEqual(area(a, id).choices, area(b, id).choices);
    for (const key of ['work', 'finaleWork', 'elapsed']) near(area(a, id)[key], area(b, id)[key]);
    for (const key of ['ore', 'smelt']) near(area(a, id).buffers[key], area(b, id).buffers[key]);
  }
  assert.deepEqual(a.expedition.mastery, b.expedition.mastery);
  assert.equal(a.expedition.purchases, b.expedition.purchases);
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
  assert.ok(opening.purchases[0].second >= 5 && opening.purchases[0].second <= 8);
  assert.deepEqual(opening.stages.slice(0, 3).map(item => item.index), [0, 1, 2]);
  const [first, quarry, tower] = opening.stages;
  assert.ok(first.second >= 75 && first.second <= 105);
  assert.ok(quarry.second >= 180 && quarry.second <= 260);
  assert.ok(tower.second >= 300 && tower.second <= 440);
  assert.ok(opening.purchases.length >= 50 && opening.purchases.length <= 80);
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
  assert.deepEqual(area(state, 'quarry').ranks, area(before, 'quarry').ranks);
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
  assert.ok(!pick.impact.some(item => item.metric === 'quarry:capacity'), 'extra extraction alone cannot speed the constrained furnace');
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
  const state = stage(2); area(state).ranks.crew = 1; area(state).purchases = 1; state.expedition.purchases += 1;
  area(state).elapsed = 70;
  const balanced = E.localRates(state);
  assert.ok(Core.act(state, { type: 'expedition-choice', id: 'protect' }).ok);
  const protection = E.localRates(state);
  assert.ok(balanced.repair > protection.repair);
  assert.ok(protection.beacon > balanced.beacon, 'protection has a real finale role under pressure');
  const work = area(state).work;
  advance(state, 40);
  assert.ok(area(state).work > work);
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

test('an explicit released-run Refit adopts rebuilding ranks while preserving permanent knowledge and plans', () => {
  const state = stage(3);
  assert.ok(Core.act(state, { type: 'expedition-blueprint', id: 'engineering' }).ok);
  assert.equal(Core.act(state, { type: 'expedition-blueprint', id: 'caravan' }).ok, false);
  assert.ok(Core.act(state, { type: 'expedition-automation', enabled: true, priority: 'progress', dispatch: true }).ok);
  assert.ok(Core.act(state, { type: 'buy', id: 'gear-boots' }).ok);
  assert.ok(Core.getRefitPreview(state).available);
  const retained = clone(state.expedition), premium = clone(state.premium);
  assert.ok(Core.act(state, { type: 'refit' }).ok);
  assert.equal(state.expedition.version, 3);
  for (const key of ['cleared', 'blueprints', 'mastery', 'purchases']) assert.deepEqual(state.expedition[key], retained[key]);
  for (const key of ['enabled', 'priority', 'dispatch']) assert.equal(state.expedition.automation[key], retained.automation[key]);
  for (const id of Object.keys(retained.areas)) {
    assert.ok(Object.values(area(state, id).ranks).every(rank => rank === 0));
    assert.deepEqual(area(state, id).buffers, { input: 0, output: 0 });
    assert.ok(area(state, id).learned.length >= 3);
  }
  assert.equal(state.expedition.work, 0);
  assert.equal(state.premium.rng, premium.rng);
  assert.ok(Core.getRates(state).gain.ore.m > 0);
  assert.ok(Core.getRefitPreview(state).resets.some(text => /ranks/.test(text)));
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
    const historical = Core.createState(0); delete historical.expedition; delete historical.guild; delete historical.introductions; delete historical.collection; delete historical.areaSkills;
    historical.schemaVersion = version;
    if (version < 3) { delete historical.luck; delete historical.caravan; }
    else Core.Content.RELICS.filter(item => item.chapter > 0).forEach(item => delete historical.luck.duplicateProgress[item.id]);
    if (version === 1) { delete historical.premium; delete historical.resources.starshards; }
    const migrated = Core.migrateState(historical);
    assert.ok(migrated); assert.equal(Core.getView(migrated).expedition.local, true); valid(migrated);
  }
});

test('malformed supplied expedition data is rejected instead of quietly repaired', () => {
  const mutations = [s => { s.expedition = null; }, s => { area(s).ranks.boots = -1; }, s => { area(s).buffers.ore = 999; }, s => { s.expedition.index += 1; }, s => { s.expedition.extra = 1; }, s => { s.expedition.cleared += 1; }, s => { s.expedition.automation.enabled = true; }, s => { area(s).purchases = 2; }, s => { area(s).finaleWork = 1; }];
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
  const partitioned = stage(1);
  for (let second = 0; second < 1000; second += 1) advance(partitioned, 1);
  near(state.premium.eligibleSeconds, partitioned.premium.eligibleSeconds, 1e-8);
});

test('offline automation and dispatch match live partitions, including a save/reload halfway', () => {
  const historical = stage(3);
  const a = Core.normalizeState(historical, historical.lastUpdate);
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
  const retained = clone(restored); delete retained.upgradeTiers; delete retained.onboarding; delete retained.trailDeliveries;
  assert.deepEqual(retained, state, 'Only additive knowledge and empty delivery tracking change the historical save');
  assert.equal(restored.trailDeliveries.sequence, 0);
  assert.equal(restored.trailDeliveries.lifetimeCoins.m, 0);
  assert.deepEqual(store.load({ deferOffline: true }).state, restored);
  const previous = clone(state); advance(state, 10); assert.ok(store.save(state).ok);
  assert.deepEqual(JSON.parse(values.get(Storage.BACKUP_KEY)).state, Core.normalizeState(previous, previous.lastUpdate));
});

test('visiting an earlier area retains every investment while hidden queues and production continue', () => {
  const state = stage(1);
  assert.ok(Core.act(state, { type: 'expedition-choice', areaId: 'quarry', id: 'quality' }).ok);
  advance(state, 5);
  const before = clone(area(state, 'quarry')), coins = N.from(state.resources.coins), ore = N.from(state.resources.ore);
  assert.ok(before.buffers.ore > 0);
  assert.ok(Core.act(state, { type: 'expedition-select', areaId: 'greenway' }).ok);
  assert.deepEqual(area(state, 'quarry'), before, 'selecting is not a reset');
  advance(state, 30);
  assert.deepEqual(area(state, 'quarry').ranks, before.ranks);
  assert.deepEqual(area(state, 'quarry').choices, before.choices);
  assert.ok(area(state, 'quarry').work > before.work);
  assert.notDeepEqual(area(state, 'quarry').buffers, before.buffers);
  assert.ok(N.cmp(state.resources.coins, coins) > 0 && N.cmp(state.resources.ore, ore) > 0);
  const old = area(state, 'greenway').ranks.boots;
  assert.ok(Core.act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' }).ok);
  assert.equal(area(state, 'greenway').ranks.boots, old + 1, 'established infrastructure remains upgradeable');
  assert.ok(Core.act(state, { type: 'expedition-select', areaId: 'quarry' }).ok);
  valid(state);
});

test('regional expansion retains ranks, queues, policies and the exact permanent rank price', () => {
  const state = clone(opening.snapshots[2]), before = clone(state.expedition.areas);
  const price = E.catalog(state).find(item => item.id === 'area:greenway:boots').cost;
  assert.ok(Core.act(state, { type: 'expedition-next' }).ok);
  for (const id of Object.keys(before)) {
    assert.deepEqual(area(state, id).ranks, before[id].ranks);
    assert.deepEqual(area(state, id).choices, before[id].choices);
    assert.deepEqual(area(state, id).buffers, before[id].buffers);
  }
  assert.deepEqual(E.catalog(state).find(item => item.id === 'area:greenway:boots').cost, price);
  assert.equal(area(state, 'greenway').work, 0);
  assert.ok(E.catalog(state).find(item => item.id === 'area:greenway:boots').maxRank > 12);
  valid(state);
});

function earnedNetwork() {
  const state = stage(5);
  advance(state, 6000);
  for (const item of E.catalog(state).filter(item => item.action.type === 'expedition-development')) {
    if (item.dependencies.every(dep => dep.met)) Core.act(state, item.action);
  }
  // Dependency chains become available as the previous entry is built.
  for (const item of E.catalog(state).filter(item => item.action.type === 'expedition-development')) {
    if (!item.owned && item.dependencies.every(dep => dep.met)) Core.act(state, item.action);
  }
  valid(state); return state;
}
const network = earnedNetwork();

test('Tower bootstraps maps and knowledge and buys real return links from naturally earned resources', () => {
  assert.ok(network.expedition.developments.includes('tower-survey'));
  assert.ok(network.expedition.developments.includes('quarry-precision'));
  assert.ok(network.expedition.developments.includes('relay-network'));
  assert.ok(network.expedition.developments.includes('shared-workshops'));
  const state = clone(network), before = E.rates(state);
  assert.ok(Core.act(state, { type: 'expedition-choice', areaId: 'greenway', id: 'survey' }).ok);
  const after = E.rates(state);
  assert.ok(N.cmp(after.greenway.income, before.greenway.income) < 0);
  assert.ok(N.cmp(after.greenway.maps, before.greenway.maps) > 0);
  assert.ok(Core.act(state, { type: 'expedition-choice', areaId: 'quarry', id: 'precision' }).ok);
  assert.ok(E.rates(state).quarry.furnace < before.quarry.furnace);
  valid(state);
});

function ranks(state, id, values) {
  const a = area(state, id), old = a.purchases;
  Object.assign(a.ranks, values);
  a.purchases = Object.values(a.ranks).reduce((sum, value) => sum + value, 0);
  state.expedition.purchases += a.purchases - old;
  a.buffers = { ore: 0, smelt: 0 };
}

test('equal-budget dispatch and processing choices reverse their value at real bottlenecks', () => {
  const state = clone(network);
  ranks(state, 'quarry', { picks: 7, carts: 1, furnace: 7 });
  Core.act(state, { type: 'expedition-choice', areaId: 'greenway', id: 'trade' });
  const trade = E.rates(state);
  Core.act(state, { type: 'expedition-choice', areaId: 'greenway', id: 'freight' });
  const freight = E.rates(state);
  assert.ok(N.cmp(freight.quarry.materials, trade.quarry.materials) > 0);
  assert.ok(N.cmp(freight.greenway.income, trade.greenway.income) < 0);
  const coinGoal = clone(state), oreGoal = clone(state);
  Core.act(coinGoal, { type: 'expedition-choice', areaId: 'greenway', id: 'trade' });
  advance(coinGoal, 120); advance(oreGoal, 120);
  assert.ok(N.cmp(coinGoal.resources.coins, oreGoal.resources.coins) > 0);
  assert.ok(N.cmp(oreGoal.resources.ore, coinGoal.resources.ore) > 0, 'same wallet and time, different winning goals');
  ranks(state, 'quarry', { picks: 7, carts: 7, furnace: 0 });
  const constrained = E.rates(state);
  Core.act(state, { type: 'expedition-choice', areaId: 'greenway', id: 'trade' });
  near(N.toNumber(E.rates(state).quarry.materials), N.toNumber(constrained.quarry.materials));
  ranks(state, 'quarry', { picks: 0, carts: 7, furnace: 7 });
  const volume = E.rates(state).quarry;
  Core.act(state, { type: 'expedition-choice', areaId: 'quarry', id: 'precision' });
  assert.ok(N.cmp(E.rates(state).quarry.materials, volume.materials) > 0);
  ranks(state, 'quarry', { picks: 7, carts: 7, furnace: 0 });
  const precision = E.rates(state).quarry;
  Core.act(state, { type: 'expedition-choice', areaId: 'quarry', id: 'throughput' });
  assert.ok(N.cmp(E.rates(state).quarry.materials, precision.materials) > 0);
  valid(state);
});

test('all-area automation is independent of selected tab and keeps investments through a day away', () => {
  const a = clone(network), b = clone(a);
  Core.act(a, { type: 'expedition-automation', enabled: true, priority: 'materials', dispatch: true });
  Core.act(b, { type: 'expedition-automation', enabled: true, priority: 'materials', dispatch: true });
  const invested = clone(a.expedition.areas);
  advance(a, 86400);
  for (let tick = 0; tick < 48; tick += 1) {
    Core.act(b, { type: 'expedition-select', areaId: ['greenway', 'quarry', 'watchtower'][tick % 3] });
    advance(b, 1800);
  }
  same(a, b);
  for (const id of Object.keys(invested)) for (const key of Object.keys(invested[id].ranks)) assert.ok(area(a, id).ranks[key] >= invested[id].ranks[key]);
  assert.ok(a.expedition.index > network.expedition.index);
  assert.deepEqual(a.expedition.developments, network.expedition.developments);
});

test('shared reserve and explicit development objective protect savings from both automated spenders', () => {
  const state = stage(3);
  assert.ok(Core.act(state, { type: 'buy', id: 'gear-boots' }).ok);
  assert.ok(Core.act(state, { type: 'refit' }).ok);
  assert.ok(Core.act(state, { type: 'plan-reserve', id: 'coins', amount: '100' }).ok);
  assert.ok(Core.act(state, { type: 'plan-goal', action: { type: 'expedition-development', id: 'wheelworks' } }).ok);
  assert.ok(Core.act(state, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: true }).ok);
  advance(state, 600);
  assert.ok(state.expedition.projects.includes('wheelworks'));
  assert.ok(N.cmp(state.resources.coins, 100) >= 0);
  assert.equal(state.guild.plan.goal, null);
  // A deliberately short cash interval cannot buy a rank just below its exact
  // cost. The next interval does; the wallet never supplies a free whole coin.
  Core.act(state, { type: 'expedition-automation', enabled: false, priority: 'balanced', dispatch: false });
  const offer = require('../../js/games/wayfarers-guild/progression.js').catalog(state).find(x => x.id === 'area:greenway:boots');
  state.resources.coins = N.sub(offer.cost[0].amount, .001);
  assert.equal(Core.act(state, offer.action).ok, false);
  advance(state, .01);
  assert.ok(Core.act(state, offer.action).ok);
  valid(state);
});

test('canonical catalog has authored late transformations, explicit causal capacities and pure previews', () => {
  const state = clone(network), before = JSON.stringify(state), view = Core.getView(state);
  assert.equal(JSON.stringify(state), before);
  assert.equal(new Set(view.globalUpgrades.map(item => item.id)).size, view.globalUpgrades.length);
  assert(view.globalUpgrades.every(item => item.visible !== false), 'Hidden future rows are absent from the canonical menu');
  const developments = view.globalUpgrades.filter(item => item.action.type === 'expedition-development');
  assert(developments.length > 0 && developments.length < 18);
  for (const id of ['optical-foundry', 'trail-prospectors', 'tower-control-room', 'survey-exchange']) {
    assert(!developments.some(d => d.action.id === id), 'Future transformation remains hidden: ' + id);
  }
  for (const item of developments) assert(item.sourceAreas.length && item.targetAreas.length);
  for (const key of ['picks', 'carts', 'furnace']) assert.ok(view.globalUpgrades.find(item => item.id === 'area:quarry:' + key).impact.some(item => item.metric === 'quarry:' + key));
  valid(state);
});

test('late optical and adaptive plans change earlier processing without partition or conversion exploits', () => {
  // A valid controlled late-landmark fixture isolates recipes from progression
  // timing. Natural acquisition is measured separately by the policy simulation.
  const base = clone(network);
  base.lifetime.highestRoute = 17; base.run.completed = 17; base.route.index = 18;
  base.lifetime.charters = 3; base.guild.chapterProject.number = 3;
  base.premium.claimedMilestones.push('first-charter');
  base.expedition.cleared = 17; E.restart(base, 18);
  for (const key of ['coins', 'ore', 'knowledge', 'maps']) base.resources[key] = N.from(1e12);
  for (const item of E.catalog(base).filter(item => item.action.type === 'expedition-development')) if (!item.owned) assert.ok(Core.act(base, item.action).ok);
  for (const key of ['coins', 'ore', 'knowledge', 'maps']) base.resources[key] = N.zero();
  ranks(base, 'quarry', { picks: 0, carts: 5, furnace: 7 });
  area(base, 'quarry').buffers = { ore: 20, smelt: 20 };
  const volume = E.rates(base);
  Core.act(base, { type: 'expedition-choice', areaId: 'quarry', id: 'optics' });
  const optical = E.rates(base);
  assert.ok(N.cmp(optical.quarry.materials, volume.quarry.materials) < 0);
  assert.ok(N.cmp(optical.watchtower.knowledge, volume.watchtower.knowledge) > 0);
  near(optical.quarry.yield, .3 * 1.15);
  assert.ok(optical.quarry.furnace * .35 <= .08 + area(base, 'watchtower').ranks.beacon * .02 + 1e-8);
  for (const plan of ['optics', 'adaptive']) {
    const a = clone(base), b = clone(base);
    Core.act(a, { type: 'expedition-choice', areaId: 'quarry', id: plan });
    Core.act(b, { type: 'expedition-choice', areaId: 'quarry', id: plan });
    advance(a, 1800);
    for (let t = 0; t < 1800; t += 11.37) advance(b, Math.min(11.37, 1800 - t));
    same(a, b);
    if (plan === 'adaptive') assert.equal(E.rates(a).quarry.processing, 'precision', 'cleared queues switch to the higher yield scarce-input recipe');
  }
});

test('a reachable Refit preserves infrastructure and credits only newly performed run work', () => {
  const state = clone(network);
  Core.act(state, { type: 'buy', id: 'gear-boots' });
  assert.ok(Core.getRefitPreview(state).available);
  const retained = clone(state.expedition.areas), developments = clone(state.expedition.developments);
  assert.ok(Core.act(state, { type: 'refit' }).ok);
  assert.equal(N.toNumber(state.run.work), 0);
  const target = N.toNumber(Core.getRoute(0, state).distance);
  advance(state, 1);
  assert.ok(N.toNumber(state.run.work) > 0 && N.toNumber(state.run.work) < target, 'past area work is not re-credited after a reset');
  for (const id of Object.keys(retained)) assert.ok(Object.values(area(state, id).ranks).every(rank => rank === 0));
  assert.deepEqual(state.expedition.legacyDevelopments, developments);
  valid(state);
});

test('late transformations require earned Charters and are advertised before renewal', () => {
  const state = clone(network);
  state.lifetime.highestRoute = 17; state.run.completed = 17; state.route.index = 18;
  state.expedition.cleared = 17; E.restart(state, 18);
  for (const key of ['coins', 'ore', 'knowledge', 'maps']) state.resources[key] = N.from(1e12);
  assert.ok(Core.getCharterPreview(state).resets.some(text => text.includes('adopts the new six-area')));
  const ids = ['trail-prospectors', 'tower-control-room', 'survey-exchange'];
  for (const id of ids) assert.equal(Core.act(state, { type: 'expedition-development', id }).ok, false);
  for (let count = 1; count <= 3; count += 1) {
    state.lifetime.charters = count; state.guild.chapterProject.number = count;
    if (count === 1) state.premium.claimedMilestones.push('first-charter');
    assert.ok(Core.act(state, { type: 'expedition-development', id: ids[count - 1] }).ok);
    valid(state);
  }
  const malformed = clone(state);
  malformed.lifetime.charters = 1; malformed.guild.chapterProject.number = 1;
  assert.equal(Core.validateState(malformed).valid, false, 'future owned transformations cannot bypass the reset prerequisite');
});

test('actual paid guild outputs match detached previews without serializing account ownership', () => {
  const state = clone(network);
  Core.setPremiumEntitlements(state, ['artisan', 'scholar', 'compass']);
  ranks(state, 'quarry', { picks: 7, carts: 7, furnace: 0 });
  const before = JSON.stringify(state), card = E.catalog(state).find(item => item.id === 'area:quarry:furnace');
  const ore = card.impact.find(item => item.metric === 'guild:ore');
  assert.ok(ore);
  near(N.toNumber(N.div(ore.currentValue, Core.getRates(state).gain.ore)), 1, 1e-12);
  assert.equal(JSON.stringify(state), before);
  assert.ok(Core.act(state, card.action).ok);
  near(N.toNumber(N.div(ore.nextValue, Core.getRates(state).gain.ore)), 1, 1e-12);
  assert.deepEqual(state.premium.owned, network.premium.owned);
});

test('published schema-four expedition-one saves load through strict storage migration', () => {
  const old = stage(1); advance(old, 9);
  const current = old.expedition, active = area(old), choices = clone(active.choices); delete choices.dispatch;
  old.expedition = {
    version: 1, index: current.index, cleared: current.cleared, completed: current.completed,
    elapsed: active.elapsed, work: active.work, finaleWork: active.finaleWork,
    buffers: clone(active.buffers), ranks: clone(active.ranks), choices,
    blueprints: clone(current.blueprints), mastery: clone(current.mastery), automation: clone(current.automation),
    sequence: current.sequence, seen: current.seen,
    recent: current.recent.map(({ sequence, kind, title, text, stage }) => ({ sequence, kind, title, text, stage })),
    purchases: current.purchases, stagePurchases: active.purchases
  };
  valid(old);
  const values = new Map(), store = Storage.createStore({ core: Core, now: () => old.lastUpdate, storage: { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) } });
  assert.ok(store.save(old).ok);
  const loaded = store.load({ deferOffline: true });
  assert.equal(loaded.status, 'loaded');
  assert.equal(loaded.state.expedition.version, 2);
  for (const key of Object.keys(old).filter(key => key !== 'expedition')) assert.deepEqual(loaded.state[key], old[key]);
  for (const key of ['ranks', 'buffers', 'work', 'finaleWork', 'elapsed']) assert.deepEqual(area(loaded.state, 'quarry')[key], old.expedition[key]);
  assert.deepEqual(Core.normalizeState(loaded.state, loaded.state.lastUpdate), loaded.state);
  valid(loaded.state);
});
