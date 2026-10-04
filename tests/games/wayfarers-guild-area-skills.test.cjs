'use strict';
const {createReleasedState}=require('./helpers/wayfarers-released.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, P, N, clone, advance, fund, mature, claimTiers } = require('./helpers/wayfarers-progression.cjs');
const S = require('../../js/games/wayfarers-guild/area-skills.js');
const T = require('../../js/games/wayfarers-guild/trail-deliveries.js');
const near = (a, b, tolerance = 1e-7) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(a), Math.abs(b)) * tolerance, `${a} != ${b}`);
const act = (s, a) => { const r = Core.act(s, a); assert.ok(r.ok, r.message); return r; };
const valid = s => assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
let earnedCache;
function earned() {
  if (!earnedCache) {
    const s = mature(); advance(s, 20000); fund(s);
    for (const d of S.Content.SKILLS) {
      act(s, { type: 'area-skill-unlock', id: d.id });
      for (let n = 0; n < 2; n += 1) act(s, { type: 'area-skill-buy', id: d.id, count: 1 });
    }
    valid(s); earnedCache = clone(s);
  }
  return clone(earnedCache);
}
function isolated(id, setup = () => {}) {
  const s = earned();
  for (const key of Object.keys(s.areaSkills.ranks)) s.areaSkills.ranks[key] = 0;
  for (const key of Object.keys(s.areaSkills.configs)) s.areaSkills.configs[key] = 'off';
  s.areaSkills.runtime = S.initial(s).runtime;
  s.expedition.areas.harbor.voyages = [];
  setup(s);
  const boosted = clone(s); boosted.areaSkills.ranks[id] = 2;
  const option = S.Content.SKILLS.find(d => d.id === id).options[1];
  if (option) boosted.areaSkills.configs[id] = option.value;
  return [s, boosted];
}
function effect(id, setup, metric, direction = 1) {
  test(id + ' changes its actual production rule', () => {
    const [a, b] = isolated(id, setup);
    const before = metric(a, P.rawRates(a)), after = metric(b, P.rawRates(b));
    assert.ok(direction * (after - before) > 1e-9, `${id}: ${before} → ${after}`);
  });
}

test('each area retains six tracks and adds nine unique techniques in five three-entry modules', () => {
  assert.equal(S.Content.SKILLS.length, 54);
  assert.equal(new Set(S.Content.SKILLS.map(d => d.icon)).size, 54);
  for (const d of P.Content.AREAS) {
    const counts = [0, 0, 0, 0, 0];
    assert.equal(d.tracks.length, 6);
    for (const t of d.tracks) counts[S.Content.TRACK_LAYOUT[d.id][t.id][0]] += 1;
    for (const t of S.Content.SKILLS.filter(t => t.areaId === d.id)) counts[t.module] += 1;
    assert.deepEqual(counts, [3, 3, 3, 3, 3]);
  }
});
test('wealth and elapsed time cannot replace local foundation work or explicit claims', () => {
  const s = createReleasedState(0); fund(s);
  for (let i = 0; i < 4; i += 1) act(s, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' });
  assert.equal(S.foundationRequirement(s, 'greenway', 'porters').met, false);
  assert.equal(Core.act(s, { type: 'upgrade-tier-unlock', id: 'area:greenway:porters' }).ok, false);
  advance(s, 150);
  assert.ok(s.trailDeliveries.deliveries >= 3);
  assert.equal(S.foundationRequirement(s, 'greenway', 'porters').met, true);
  assert.equal(s.expedition.areas.quarry, undefined);
  assert.equal(S.eligible(s, 'express-routes'), false);
  valid(s);
});
test('all techniques earn, claim, and purchase through canonical actions', () => {
  const s = earned();
  assert.equal(s.areaSkills.unlocked.length, 54);
  assert.ok(Object.values(s.areaSkills.ranks).every(v => v === 2));
  assert.ok(S.view(s).items.every(v => v.owned && v.maxRank === 10)); valid(s);
});
test('readiness of the third foundation cannot substitute for claiming its lesson', () => {
  const s = mature(); s.upgradeTiers.claimed = s.upgradeTiers.claimed.filter(id => id !== 'area:quarry:furnace');
  assert.ok(s.expedition.areas.quarry.learned.includes('furnace'));
  assert.equal(S.eligible(s, 'stockpiles'), false);
  const before = clone(s); assert.equal(Core.act(s, { type: 'area-skill-unlock', id: 'stockpiles' }).ok, false); assert.deepEqual(s, before);
});
test('malformed high-rank, configuration and unbacked receipt histories fail closed', () => {
  const fresh = createReleasedState(0);
  for (const mutate of [s => s.areaSkills.highRanks.stockpiles = 1, s => s.areaSkills.configs['ore-sorting'] = 'graded', s => s.areaSkills.ranks.stockpiles = 11]) {
    const s = clone(fresh); mutate(s); assert.equal(S.validate(s.areaSkills, s), false);
  }
  const s = earned();
  s.areaSkills.runtime.voyageReceipts.push({ key: 'no-funded-voyage', supplies: 100, refund: .1, domestic: .2, bonded: true });
  assert.equal(S.validate(s.areaSkills, s), false);
});
test('exact quantities reject stale, underfunded and overflowing purchases atomically', () => {
  const s = earned(), id = 'stockpiles';
  const q = S.quote(s, id, 5), before = clone(s.resources);
  act(s, { type: 'area-skill-buy', id, count: 5, quote: q.token });
  assert.equal(s.areaSkills.ranks[id], 7);
  for (const [key, price] of Object.entries(q.costs)) assert.deepEqual(s.resources[key], N.sub(before[key], price));
  let saved = clone(s);
  assert.equal(Core.act(s, { type: 'area-skill-buy', id, count: 5, quote: q.token }).ok, false);
  assert.deepEqual(s, saved);
  s.expedition.batch = 5;
  const row = S.view(s).items.find(v => v.skillId === id);
  assert.equal(row.disabled, true); assert.deepEqual(row.fittingQuantityAction, { type: 'expedition-batch', count: 1 });
  s.expedition.batch = 1; const fresh = S.quote(s, id, 1);
  act(s, { type: 'area-skill-config', id: 'ore-sorting', value: 'graded' }); saved = clone(s);
  assert.equal(Core.act(s, { type: 'area-skill-buy', id, count: 1, quote: fresh.token }).ok, false); assert.deepEqual(s, saved);
  s.resources.coins = N.zero(); saved = clone(s);
  assert.equal(Core.act(s, { type: 'area-skill-buy', id, count: 1 }).ok, false); assert.deepEqual(s, saved);
});
test('ten ranks are the bound, with one capped throughput bucket and bounded actual-input refunds', () => {
  const s = earned(), id = 'stockpiles';
  for (let i = 2; i < 10; i += 1) act(s, { type: 'area-skill-buy', id, count: 1 });
  assert.equal(S.value(s, id), 1.5);
  assert.equal(S.throughput(.5, .55), 1.75);
  assert.equal(S.refund(.2, .2), .25);
  const saved = clone(s); assert.equal(Core.act(s, { type: 'area-skill-buy', id, count: 1 }).ok, false); assert.deepEqual(s, saved);
});
test('reset retains knowledge and chosen operations, but clears ranks, prepaid buffers and active benefits', () => {
  const s = earned(); act(s, { type: 'area-skill-config', id: 'ore-sorting', value: 'graded' });
  const before = clone(s.areaSkills); P.reset(s);
  assert.deepEqual(s.areaSkills.unlocked, before.unlocked); assert.deepEqual(s.areaSkills.output, before.output);
  assert.deepEqual(s.areaSkills.highRanks, before.highRanks); assert.equal(s.areaSkills.configs['ore-sorting'], 'graded');
  assert.equal(S.mode(s, 'ore-sorting'), 'off'); assert.ok(Object.values(s.areaSkills.ranks).every(v => v === 0));
  assert.deepEqual(s.areaSkills.runtime, S.initial(s).runtime);
});

effect('express-routes', nullSetup, s => S.trailSettings(s).speed);
effect('cargo-lashing', nullSetup, s => S.trailSettings(s).cargo);
effect('caravan-escorts', s => s.expedition.areas.greenway.choice = 'freight', (_, r) => r.gain.coins);
effect('supply-depots', nullSetup, (_, r) => r.areas.quarry.capacity);
effect('bonded-routes', nullSetup, (_, r) => r.areas.harbor.payout.coins);
effect('continental-logistics', s => s.expedition.areas.greenway.choice = 'continental', (_, r) => r.gain.coins);
effect('stockpiles', nullSetup, (_, r) => r.areas.quarry.capacity);
effect('ore-sorting', nullSetup, (_, r) => r.areas.quarry.carts, -1);
effect('reinforced-shafts', s => s.expedition.areas.quarry.choice = 'rich', (_, r) => r.areas.quarry.alloy);
effect('rail-transfer', nullSetup, (_, r) => r.areas.quarry.furnace);
effect('slag-processing', nullSetup, (_, r) => r.gain.provisions);
effect('parallel-furnaces', s => s.expedition.areas.quarry.choice = 'rich', (_, r) => r.areas.quarry.furnace);
effect('triangulation', nullSetup, (_, r) => r.areas.watchtower.research);
effect('dispatch-codes', s => s.expedition.areas.watchtower.plans.assignments = ['survey'], (_, r) => r.gain.coins);
effect('mineral-cartography', s => s.expedition.areas.quarry.choice = 'rich', (_, r) => r.areas.quarry.picks);
effect('logistics-charts', s => s.expedition.areas.watchtower.plans.assignments = ['industry'], (_, r) => r.areas.quarry.capacity);
effect('weather-stations', s => { s.expedition.areas.harbor.elapsed = 21600; s.expedition.areas.harbor.plans.port = 'ocean'; }, (_, r) => r.areas.harbor.payout.coins / r.areas.harbor.duration);
effect('research-exchanges', s => s.expedition.areas.watchtower.plans.target = 'deep', (_, r) => r.gain.maps);
effect('long-signals', s => s.expedition.areas.watchtower.plans.assignments = ['trade'], (_, r) => r.gain.coins);
effect('celestial-calendar', s => s.expedition.areas.watchtower.plans.target = 'deep', (_, r) => r.areas.watchtower.research);
effect('material-hoppers', nullSetup, (_, r) => r.areas.workshop.hopperTarget);
effect('offcut-recovery', nullSetup, (_, r) => r.gain.ore);
effect('standard-tools', nullSetup, (_, r) => r.areas.quarry.picks);
effect('spare-parts', nullSetup, (_, r) => r.areas.quarry.carts);
effect('instrument-cases', s => s.expedition.areas.workshop.plans.templates = ['instruments'], (_, r) => r.gain.provisions);
effect('precision-fixtures', s => s.expedition.areas.workshop.choice = 'precision', (_, r) => r.areas.workshop.flow);
effect('modular-frames', s => s.expedition.areas.workshop.plans.templates = ['supplies', 'instruments'], (_, r) => r.areas.workshop.templates[0].flow);
effect('export-crates', nullSetup, (_, r) => r.areas.harbor.supply, -1);
effect('field-camps', nullSetup, (_, r) => r.areas.ruins.capacity);
effect('careful-recovery', nullSetup, (_, r) => r.areas.ruins.recovery, -1);
effect('survey-tablets', s => s.expedition.areas.ruins.plans.discovery = 'inscribed', (_, r) => r.gain.maps);
effect('botanical-remedies', s => s.expedition.areas.ruins.plans.discovery = 'botanical', (_, r) => r.gain.provisions);
effect('reclaimed-alloys', s => { s.expedition.areas.ruins.plans.discovery = 'metallic'; s.expedition.areas.ruins.plans.loadouts = ['industry']; s.expedition.areas.ruins.discoveries.metallic = 1; }, (_, r) => r.areas.workshop.templates[0].output);
effect('expedition-rigs', s => { const r = s.expedition.areas.ruins; r.buffers.input = 0; r.ranks.delving = 0; }, (_, r) => r.gain.herbs);
effect('resonant-pairings', s => s.expedition.areas.ruins.plans.loadouts = ['industry', 'survey'], (_, r) => r.areas.ruins.artifacts);
effect('archive-network', s => { const d = P.Content.PROJECTS.at(-1); s.expedition.projects = s.expedition.projects.filter(id => id !== d.id); s.expedition.commission = { id: d.id, work: 0 }; }, (_, r) => r.areas.ruins.research);
effect('provision-packing', nullSetup, (_, r) => r.areas.harbor.supply, -1);
effect('coastal-tenders', s => s.expedition.areas.harbor.plans.port = 'coast', (_, r) => r.areas.harbor.voyageTarget, -1);
effect('mixed-holds', s => s.expedition.areas.harbor.choice = 'trade', (_, r) => r.areas.harbor.payout.ore || 0);
effect('scheduled-convoys', s => s.areaSkills.runtime.convoyReady = true, (_, r) => r.areas.harbor.payout.coins);
effect('salvage-nets', s => s.expedition.areas.harbor.choice = 'discovery', (_, r) => r.areas.harbor.payout.ore || 0);
effect('weather-routing', s => { s.expedition.areas.harbor.elapsed = 21600; s.expedition.areas.harbor.plans.port = 'ocean'; }, (_, r) => r.areas.harbor.payout.coins / r.areas.harbor.duration);
effect('deepwater-holds', s => s.expedition.areas.harbor.choice = 'materials', (_, r) => r.areas.harbor.payout.maps);
function nullSetup() {}

test('arrival journals and relay runners pay only on a completed trip and only into a funded commission', () => {
  const s = earned(), d = P.Content.PROJECTS.at(-1); s.expedition.projects = s.expedition.projects.filter(id => id !== d.id); s.expedition.commission = { id: d.id, work: 0 };
  const before = N.toNumber(s.resources.maps), rates = { coins: N.from(5), travel: 1, skillMaps: 2, skillResearch: 3 };
  T.resetTrip(s); T.tick(s, 29, rates); near(N.toNumber(s.resources.maps), before); assert.equal(s.expedition.commission.work, 0);
  T.tick(s, 1, rates); near(N.toNumber(s.resources.maps) - before, 2 * S.value(s, 'field-journals'), 1e-3);
  near(s.expedition.commission.work, 3 * S.value(s, 'relay-runners'));
});
test('notebooks cannot finance a project, and contribute no more than their stored and per-project cap', () => {
  const [a, s] = isolated('field-notebooks'); const d = P.Content.PROJECTS.at(-1);
  P.tick(s, 1000); assert.ok(s.areaSkills.runtime.notebook > 0);
  const saved = s.areaSkills.runtime.notebook; S.fundCommission(s, d); assert.equal(s.areaSkills.runtime.notebook, saved);
  s.expedition.projects = s.expedition.projects.filter(id => id !== d.id); s.expedition.commission = { id: d.id, work: 0 }; S.fundCommission(s, d);
  near(s.expedition.commission.work, Math.min(saved, d.work * S.value(s, 'field-notebooks', .1, .25)));
});
test('paid manifest cargo, provisions, refunds and domestic support stay frozen across configuration changes', () => {
  const s = earned(); s.expedition.areas.harbor.voyages = []; s.areaSkills.runtime.voyageReceipts = [];
  act(s, { type: 'area-skill-config', id: 'exchange-houses', value: 'domestic' });
  act(s, { type: 'area-skill-config', id: 'bonded-routes', value: 'bonded' });
  act(s, { type: 'area-skill-config', id: 'twin-manifests', value: 'materials' });
  P.autoBuy(s); const manifests = clone(s.expedition.areas.harbor.voyages), receipts = clone(s.areaSkills.runtime.voyageReceipts);
  assert.equal(manifests.length, 2); assert.ok(manifests[1].payout.ore); assert.ok(S.domesticSupport(s) > 0);
  act(s, { type: 'area-skill-config', id: 'exchange-houses', value: 'off' });
  act(s, { type: 'area-skill-config', id: 'bonded-routes', value: 'off' });
  assert.deepEqual(s.expedition.areas.harbor.voyages, manifests); assert.deepEqual(s.areaSkills.runtime.voyageReceipts, receipts);
  const before = N.toNumber(s.resources.provisions), expected = receipts.reduce((n, r) => n + r.supplies * r.refund, 0);
  s.expedition.areas.harbor.voyages.forEach(v => v.work = v.target); P.tick(s, 1e-12);
  near(N.toNumber(s.resources.provisions) - before, expected, 1e-3); assert.equal(S.domesticSupport(s), 0); assert.equal(s.areaSkills.runtime.voyageReceipts.length, 0);
});
test('support allocations are mutually exclusive, and prepaid hoppers respect ore reserves', () => {
  const s = earned(); act(s, { type: 'area-skill-config', id: 'standard-tools', value: 'extraction' });
  act(s, { type: 'area-skill-config', id: 'spare-parts', value: 'refining' }); assert.equal(S.mode(s, 'standard-tools'), 'off');
  s.guild.plan.reserves.ore = N.from(100); s.resources.ore = N.from(112);
  act(s, { type: 'area-skill-config', id: 'material-hoppers', value: 'buffer' }); P.autoBuy(s);
  near(N.toNumber(s.resources.ore) + s.areaSkills.runtime.hopper, 112); assert.ok(N.cmp(s.resources.ore, 100) >= 0);
});
for (const family of [['batch-kilns', 'batch'], ['resonant-drills'], ['shift-planning', 'automatic'], ['template-queue', 'alternate'], ['site-catalogues', 'rotate'], ['material-hoppers', 'buffer']]) test(family[0] + ' agrees across offline and foreground boundaries, including save/reload', () => {
  const [, a] = isolated(family[0]); fund(a, 1e6);
  if (family[1]) a.areaSkills.configs[family[0]] = family[1];
  const b = clone(a); advance(a, 60);
  for (let i = 0; i < 120; i += 1) advance(b, .5);
  for (const key of Object.keys(a.resources)) near(N.toNumber(a.resources[key]), N.toNumber(b.resources[key]), 1e-6);
  for (const key of ['hopper', 'batchClock', 'drillWork', 'drillRemaining', 'shiftClock', 'templateWork', 'siteWork']) near(a.areaSkills.runtime[key], b.areaSkills.runtime[key], 1e-6);
  assert.deepEqual(a.expedition.areas.workshop.plans.templates, b.expedition.areas.workshop.plans.templates);
  assert.equal(a.expedition.areas.ruins.plans.discovery, b.expedition.areas.ruins.plans.discovery);
  const reload = clone(a); advance(a, 5); advance(reload, 5); assert.deepEqual(reload, a); valid(a);
});
