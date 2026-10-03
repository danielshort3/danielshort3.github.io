'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, P, N, clone, advance, fund, mature, claimTiers } = require('./helpers/wayfarers-progression.cjs');
const near = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(a), Math.abs(b)) * tolerance, `${a} != ${b}`);
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
const amount = value => N.toNumber(value);
const act = (state, action) => { const result = Core.act(state, action); assert.equal(result.ok, true, result.message); return result; };

test('the opening is one track, buys at six seconds and unlocks tracks through learning', () => {
  const state = Core.createState(0);
  assert.equal(P.view(state).cards.filter(row => row.visible).length, 1);
  assert.equal(P.view(state).scene.established, false);
  advance(state, 5);
  assert.equal(P.quote(state, 'greenway', 'boots', 1).affordable, false);
  advance(state, 1);
  act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' });
  fund(state); act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' });
  assert.deepEqual(state.expedition.areas.greenway.learned, ['boots', 'porters']);
  valid(state);
});

for (const count of [1, 5, 10, 25, 100]) test(`local exact ×${count} equals sequential prices across a material tier`, () => {
  const state = mature(), separate = clone(state);
  const q = P.quote(state, 'quarry', 'picks', count);
  assert.equal(q.affordable, true);
  act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count, quote: q.token });
  for (let i = 0; i < count; i += 1) act(separate, { type: 'expedition-buy', areaId: 'quarry', id: 'picks' });
  assert.equal(state.expedition.areas.quarry.ranks.picks, separate.expedition.areas.quarry.ranks.picks);
  for (const id of Object.keys(state.resources)) near(amount(state.resources[id]), amount(separate.resources[id]), 1e-12);
  assert.deepEqual(state.expedition.areas.quarry.highRanks, separate.expedition.areas.quarry.highRanks);
  assert.deepEqual(state.expedition.recent, separate.expedition.recent);
  valid(state);
});

test('whole-batch affordability, stale quotes, invalid counts and cap remainder are atomic', () => {
  const state = mature(), q = P.quote(state, 'quarry', 'picks', 5);
  state.resources.coins = N.from(q.costs.coins);
  state.resources.maps = N.sub(q.costs.maps, .0001);
  let before = clone(state);
  assert.equal(Core.act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count: 5, quote: q.token }).ok, false);
  assert.deepEqual(state, before);
  fund(state);
  const stale = P.quote(state, 'quarry', 'picks', 5);
  act(state, { type: 'expedition-choice', areaId: 'quarry', id: 'rich' });
  before = clone(state);
  assert.equal(Core.act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count: 5, quote: stale.token }).ok, false);
  assert.deepEqual(state, before);
  for (const count of [0, 2, 101, 1000, NaN]) {
    assert.equal(Core.act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count }).ok, false);
    assert.deepEqual(state, before);
  }
  state.expedition.areas.quarry.ranks.picks = 998; state.expedition.areas.quarry.highRanks.picks = 998;
  before = clone(state);
  assert.equal(Core.act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count: 5 }).ok, false);
  assert.deepEqual(state, before);
  assert.match(P.quote(state, 'quarry', 'picks', 5).reason, /2 ranks remain/);
});

test('saved areas, working choices and funded permanent research survive a real Refit while ranks rebuild', () => {
  const state = mature();
  state.expedition.projects = state.expedition.projects.filter(id => id !== 'guild-commerce');
  for (const key of ['entries', 'announced', 'read']) state.onboarding[key] = state.onboarding[key].filter(id => id !== 'project:guild-commerce');
  act(state, { type: 'expedition-development', id: 'guild-commerce' });
  state.expedition.commission.work = 123;
  act(state, { type: 'expedition-config', areaId: 'watchtower', kind: 'assignments', slot: 1, id: 'industry' });
  act(state, { type: 'expedition-config', areaId: 'workshop', kind: 'templates', slot: 1, id: 'instruments' });
  const before = clone(state), quote = P.quote(state, 'greenway', 'boots', 5);
  act(state, { type: 'refit' });
  assert.equal(state.expedition.commission.work, 123);
  for (const [id, area] of Object.entries(state.expedition.areas)) {
    assert.ok(Object.values(area.ranks).every(rank => rank === 0));
    assert.deepEqual(area.plans, before.expedition.areas[id].plans);
    assert.deepEqual(area.learned, before.expedition.areas[id].learned);
    assert.equal(area.cap, before.expedition.areas[id].cap);
    assert.deepEqual(area.buffers, { input: 0, output: 0 });
  }
  assert.equal(Core.act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots', count: 5, quote: quote.token }).ok, false);
  assert.equal(amount(state.resources.coins), amount(P.quote(state, 'greenway', 'boots', 100).costs.coins));
  valid(state);
});

test('navigation changes no production, stocks, queues, ranks or working choices', () => {
  const state = mature(), other = clone(state);
  act(other, { type: 'expedition-select', areaId: 'quarry' });
  advance(state, 180); advance(other, 180);
  for (const key of Object.keys(state.resources)) near(amount(state.resources[key]), amount(other.resources[key]));
  for (const id of Object.keys(state.expedition.areas)) {
    assert.deepEqual(state.expedition.areas[id].ranks, other.expedition.areas[id].ranks);
    assert.deepEqual(state.expedition.areas[id].buffers, other.expedition.areas[id].buffers);
    assert.deepEqual(state.expedition.areas[id].plans, other.expedition.areas[id].plans);
  }
});

test('six-area offline advancement preserves queues, voyages, arrivals and random decisions across partitions', () => {
  const state = mature(), other = clone(state);
  fund(state, 1000000); fund(other, 1000000);
  advance(state, 3600);
  for (let i = 0; i < 360; i += 1) advance(other, 10);
  for (const key of Object.keys(state.resources)) near(amount(state.resources[key]), amount(other.resources[key]), 1e-7);
  assert.deepEqual(state.premium, other.premium);
  const randomA = clone(state.luck), randomB = clone(other.luck);
  for (const key of ['commonRemainingMs', 'relicRemainingMs', 'pitySeconds']) { near(randomA[key], randomB[key], 1e-10); delete randomA[key]; delete randomB[key]; }
  randomA.ledger.recent.forEach((entry, i) => { near(entry.at, randomB.ledger.recent[i].at, 1e-10); delete entry.at; delete randomB.ledger.recent[i].at; });
  assert.deepEqual(randomA, randomB);
  assert.equal(state.expedition.sequence, other.expedition.sequence);
  assert.equal(state.expedition.areas.harbor.voyages.length, other.expedition.areas.harbor.voyages.length);
  state.expedition.areas.harbor.voyages.forEach((voyage, i) => { near(voyage.work, other.expedition.areas.harbor.voyages[i].work); assert.deepEqual(voyage.payout, other.expedition.areas.harbor.voyages[i].payout); });
  valid(state); valid(other);
});

test('Focus preserves full input constraints, previews are pure and repeated resets never refill it', () => {
  const state = mature();
  state.resources.ore = N.from(100); state.guild.plan.reserves.ore = N.from(100);
  const before = clone(state); P.view(state); Core.getView(state); assert.deepEqual(state, before);
  act(state, { type: 'expedition-focus', areaId: 'workshop', id: 'priority' });
  assert.equal(state.expedition.focus.charges, 2);
  const rates = Core.getRates(state);
  assert.ok(N.cmp(rates.drain.ore, rates.gain.ore) <= 0);
  const remaining = clone(state.expedition.focus);
  act(state, { type: 'refit' });
  assert.deepEqual(state.expedition.focus, remaining);
  advance(state, 14400);
  assert.equal(state.expedition.focus.charges, 3);
  assert.equal(state.expedition.focus.recharge, 0);
  advance(state, 28800);
  assert.equal(state.expedition.focus.recharge, 0);
  valid(state);
});

test('Harbor pays supply before departure, freezes its manifest and Focus advances only paid voyages', () => {
  const state = mature(), harbor = state.expedition.areas.harbor;
  harbor.voyages = []; state.resources.provisions = N.zero();
  const before = clone(state);
  assert.equal(Core.act(state, { type: 'expedition-focus', areaId: 'harbor', id: 'priority' }).ok, false);
  assert.deepEqual(state, before);
  fund(state); P.autoBuy(state);
  const manifest = clone(harbor.voyages), stocks = amount(state.resources.provisions);
  assert.equal(manifest.length, 2); assert.ok(stocks < 1e14);
  act(state, { type: 'expedition-config', areaId: 'harbor', kind: 'port', slot: 0, id: 'ocean' });
  assert.deepEqual(harbor.voyages, manifest);
  const normal = clone(state); act(state, { type: 'expedition-focus', areaId: 'harbor', id: 'priority' });
  P.tick(normal, .001); P.tick(state, .001);
  near(harbor.voyages[0].work / normal.expedition.areas.harbor.voyages[0].work, 25);
  assert.deepEqual(harbor.voyages[0].payout, manifest[0].payout);
});

test('new template, assignment and discovery slots support actual different production plans', () => {
  const state = mature(), base = Core.getRates(state);
  act(state, { type: 'expedition-config', areaId: 'workshop', kind: 'templates', slot: 1, id: 'instruments' });
  const mixed = Core.getRates(state);
  assert.ok(N.cmp(mixed.gain.knowledge, base.gain.knowledge) > 0);
  assert.ok(N.cmp(mixed.gain.provisions, base.gain.provisions) < 0);
  const before = P.rawRates(state).areas.quarry.carts;
  act(state, { type: 'expedition-config', areaId: 'watchtower', kind: 'assignments', slot: 1, id: 'industry' });
  assert.ok(P.rawRates(state).areas.quarry.carts > before);
  act(state, { type: 'expedition-config', areaId: 'ruins', kind: 'discovery', slot: 0, id: 'metallic' });
  advance(state, 60);
  assert.ok(state.expedition.areas.ruins.discoveries.metallic > 0);
  act(state, { type: 'expedition-config', areaId: 'ruins', kind: 'loadouts', slot: 1, id: 'industry' });
  valid(state);
});

test('all 36 repeatable tracks have an executable marginal effect in an appropriate earned configuration', () => {
  const base = mature();
  base.expedition.areas.ruins.discoveries = { botanical: 2, metallic: 2, inscribed: 2 };
  base.expedition.areas.ruins.plans.loadouts = ['industry', 'trade'];
  base.expedition.areas.workshop.plans.templates = ['supplies', 'instruments'];
  base.expedition.areas.workshop.choice = 'precision';
  for (const area of P.Content.AREAS) for (const track of area.tracks) {
    const before = clone(base);
    if (track.id === 'mechanisms') before.expedition.areas.workshop.choice = 'manufacture';
    const next = clone(before);
    act(next, { type: 'expedition-buy', areaId: area.id, id: track.id });
    assert.ok(P.impact(before, next).length, area.id + ':' + track.id + ' has no mechanical effect');
  }
});

test('Notes and Crest categories support different goals rather than universal duplicate multipliers', () => {
  const base = mature(), pace = clone(base), supply = clone(base), insight = clone(base);
  pace.refitUpgrades.pace += 1; supply.refitUpgrades.supply += 1; insight.refitUpgrades.insight += 1;
  const a = P.rawRates(base), b = P.rawRates(pace), c = P.rawRates(supply), d = P.rawRates(insight);
  assert.ok(b.areas.greenway.work > a.areas.greenway.work); near(b.gain.ore, a.gain.ore);
  assert.ok(c.gain.ore > a.gain.ore); near(c.gain.knowledge, a.gain.knowledge);
  assert.ok(d.researchRate > a.researchRate); near(d.gain.coins, a.gain.coins);
});

test('strict save validation refuses stranded commissions, locked modes/caps/slots and malformed chronology', () => {
  const fresh = Core.createState(0);
  const cases = [
    s => { s.expedition.batch = 100; },
    s => { s.expedition.focus.unlocked = true; },
    s => { s.expedition.areas.greenway.cap = 1000; },
    s => { s.expedition.areas.greenway.learned.push('railways'); },
    s => { s.expedition.commission = { id: 'guild-industry', work: 1 }; },
    s => { s.expedition.projects.push('navigation-artifact'); }
  ];
  for (const change of cases) { const state = clone(fresh); change(state); assert.equal(Core.validateState(state).valid, false); }
  const full = mature(); full.expedition.recent[1].sequence = full.expedition.recent[0].sequence;
  assert.equal(Core.validateState(full).valid, false);
  const before = clone(fresh); for (const action of [null, undefined, 5]) assert.equal(Core.act(fresh, action).ok, false);
  assert.deepEqual(fresh, before);
});

test('a depleted material reserve rebuilds before Workshop conversion resumes', () => {
  const state = mature();
  state.resources.ore = N.zero(); state.guild.plan.reserves.ore = N.from(1000);
  const rates = Core.getRates(state);
  assert.equal(amount(rates.drain.ore), 0);
  advance(state, 1);
  near(amount(state.resources.ore), amount(rates.gain.ore));
  assert.equal(P.rawRates(state).areas.workshop.flow, 0);
  valid(state);
});

test('retained regional mastery, preparation and Field journals have their actual advertised effects', () => {
  const state = mature(), before = P.rawRates(state).areas.greenway.work;
  state.expedition.mastery.greenway += 1;
  assert.ok(P.rawRates(state).areas.greenway.work > before);
  const withMastery = P.rawRates(state).areas.greenway.work;
  state.upgrades.preparation += 1;
  assert.ok(P.rawRates(state).areas.greenway.work > withMastery);
  state.run.completed = 7;
  const base = Core.getRefitPreview(state).reward;
  state.research.push('field-notes');
  assert.deepEqual(Core.getRefitPreview(state).reward, N.floor(N.mul(base, 1.25)));
});

test('milestone capacity bonuses apply on crossing but the discovery announcement is not farmed after a reset', () => {
  const state = Core.createState(0); fund(state);
  for (let i = 0; i < 3; i += 1) act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' });
  assert.ok(P.power(3) / P.power(2) > 1 + .38 / Math.pow(1 + 2 / 3, 1.1));
  assert.equal(state.expedition.recent.filter(e => e.title === 'Pathfinding 3').length, 1);
  state.expedition.areas.greenway.ranks.boots = 0;
  for (let i = 0; i < 3; i += 1) act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' });
  assert.equal(state.expedition.recent.filter(e => e.title === 'Pathfinding 3').length, 1);
});

test('standing plans rebuild invested tracks without adopting a new unbought branch', () => {
  const state = mature(), area = state.expedition.areas.greenway;
  area.ranks.railways = 0; area.highRanks.railways = 0;
  act(state, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: false });
  advance(state, 180);
  assert.equal(area.ranks.railways, 0);
  act(state, { type: 'expedition-buy', areaId: 'greenway', id: 'railways' });
  advance(state, 60);
  assert.ok(area.ranks.railways > 1);
});

test('a saved permanent-project goal protects its material bill from conversion and automatic ranks', () => {
  const state = mature();
  state.expedition.projects = state.expedition.projects.filter(id => id !== 'guild-industry');
  for (const key of ['entries', 'announced', 'read']) state.onboarding[key] = state.onboarding[key].filter(id => id !== 'project:guild-industry');
  act(state, { type: 'plan-goal', action: { type: 'expedition-development', id: 'guild-industry' } });
  state.resources.ore = N.zero();
  const project = P.Content.PROJECTS.find(p => p.id === 'guild-industry');
  assert.equal(amount(P.reserve(state, 'ore')), project.costs.ore);
  assert.equal(amount(Core.getRates(state).drain.ore), 0);
  act(state, { type: 'expedition-automation', enabled: true, priority: 'materials', dispatch: false });
  advance(state, 60);
  assert.ok(amount(state.resources.ore) > 0);
  assert.ok(amount(state.resources.ore) < project.costs.ore);
  assert.equal(amount(Core.getRates(state).drain.ore), 0);
  assert.equal(state.guild.plan.goal.id, 'guild-industry');
  valid(state);
});

test('the first Refit takes 30–45 minutes and sustained prior production returns in 20–40% of that time', () => {
  const state = Core.createState(0), refits = [];
  let firstBuy = null, originalRates = null, heldSince = null, sustainedRecovery = null;
  for (let time = 0; time < 5400; time += 1) {
    claimTiers(state);
    if (state.expedition.completed) {
      if (Core.getRefitPreview(state).available) {
        if (!refits.length) {
          refits.push(time);
          originalRates = Core.getRates(state).gain;
          const preview = Core.getRefitPreview(state);
          act(state, { type: 'refit' });
          claimTiers(state);
          assert.equal(N.cmp(state.resources.coins, preview.starter), 0);
          act(state, { type: 'refit-upgrade', id: 'pace' });
          act(state, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: false });
        } else if (refits.length === 1) refits.push(time);
      } else act(state, { type: 'expedition-next' });
    }
    const project = P.Content.PROJECTS.find(definition => {
      const task = P.developmentTask(state, definition.id);
      return task.open && !task.done && Object.entries(task.costs).every(([id, cost]) => N.cmp(state.resources[id], cost) >= 0);
    });
    if (project) act(state, { type: 'expedition-development', id: project.id });
    const offers = [];
    for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned) {
      const quote = P.quote(state, areaId, id, 1);
      if (quote.valid && quote.affordable) offers.push({ areaId, id, quote });
    }
    offers.sort((a, b) => N.cmp(a.quote.costs.coins, b.quote.costs.coins));
    for (const offer of offers.slice(0, 3)) {
      if (Core.act(state, { type: 'expedition-buy', areaId: offer.areaId, id: offer.id }).ok && firstBuy === null) firstBuy = time;
    }
    if (originalRates) {
      const current = Core.getRates(state).gain;
      const restored = Object.entries(originalRates).every(([id, value]) => N.cmp(current[id], value) >= 0);
      if (!restored) heldSince = null;
      else if (heldSince === null) heldSince = time;
      if (heldSince !== null && time - heldSince >= 60) { sustainedRecovery = heldSince - refits[0]; break; }
    }
    advance(state, 1);
  }
  assert.ok(firstBuy >= 5 && firstBuy <= 10);
  assert.equal(refits.length, 2);
  assert.ok(refits[0] >= 1800 && refits[0] <= 2700, String(refits));
  assert.ok(sustainedRecovery !== null);
  const recovery = sustainedRecovery / refits[0];
  assert.ok(recovery >= .2 && recovery <= .4, String(recovery));
  valid(state);
});

test('familiar-rank rebates apply only after reset and a mixed exact batch charges new best ranks normally', () => {
  const state = mature(), area = state.expedition.areas.quarry;
  area.ranks.picks = 23; area.highRanks.picks = 25;
  const normal = [23, 24, 25, 26, 27].map(rank => P.cost(state, 'quarry', 'picks', rank));
  act(state, { type: 'refit' }); fund(state);
  area.ranks.picks = 23;
  for (let index = 0; index < normal.length; index += 1) {
    const price = P.cost(state, 'quarry', 'picks', 23 + index);
    for (const [currency, value] of Object.entries(normal[index])) {
      assert.equal(amount(price[currency]), Math.ceil(amount(value) * (index < 2 ? .5 : 1)));
    }
  }
  const separate = clone(state), quote = P.quote(state, 'quarry', 'picks', 5);
  act(state, { type: 'expedition-buy', areaId: 'quarry', id: 'picks', count: 5, quote: quote.token });
  for (let index = 0; index < 5; index += 1) act(separate, { type: 'expedition-buy', areaId: 'quarry', id: 'picks' });
  for (const id of Object.keys(state.resources)) near(amount(state.resources[id]), amount(separate.resources[id]), 1e-12);
  assert.deepEqual(P.cost(state, 'quarry', 'picks', 28), P.cost(Object.assign({}, state, { expedition: Object.assign({}, state.expedition, { renewed: false }) }), 'quarry', 'picks', 28));
  valid(state);
});

test('material voyages exchange coin and map cargo for ore while funded manifests stay frozen', () => {
  const state = mature();
  const trade = P.rawRates(state).areas.harbor;
  advance(state, 1);
  const funded = clone(state.expedition.areas.harbor.voyages);
  act(state, { type: 'expedition-choice', areaId: 'harbor', id: 'materials' });
  const materials = P.rawRates(state).areas.harbor;
  near(materials.payout.coins, trade.payout.coins * .55);
  near(materials.payout.maps, trade.payout.maps * .75);
  assert.ok(materials.payout.ore > 0);
  near(materials.supply, trade.supply);
  near(materials.duration, trade.duration);
  assert.deepEqual(state.expedition.areas.harbor.voyages, funded);
  valid(state);
});

test('area picker describes each operation and Harbor uses real manifests rather than shared wallet rates', () => {
  const state = mature();
  state.expedition.areas.harbor.voyages = [];
  const before = clone(state), rates = Core.getRates(state);
  const rows = () => Object.fromEntries(P.view(state).areas.map(row => [row.id, row]));
  let areas = rows();
  assert.equal(areas.greenway.rateText, 'Delivering cargo');
  assert.equal(areas.quarry.rateText, 'Refining ore');
  assert.equal(areas.watchtower.rateText, 'Surveying');
  assert.equal(areas.workshop.rateText, 'Manufacturing');
  assert.equal(areas.ruins.rateText, 'Recovering finds');
  assert.equal(areas.harbor.rateText, 'Preparing voyage');
  for (const area of Object.values(areas)) assert.doesNotMatch(area.rateText, /\+|\/s|maps|coins/);
  assert.deepEqual(state, before);
  assert.deepEqual(Core.getRates(state), rates);
  advance(state, 1);
  const count = state.expedition.areas.harbor.voyages.length;
  assert.ok(count > 0);
  areas = rows();
  assert.equal(areas.harbor.rateText, count + ' voyage' + (count === 1 ? '' : 's') + ' at sea');
  state.expedition.areas.harbor.voyages = [];
  state.resources.provisions = N.zero();
  state.resources.ore = N.zero();
  state.guild.plan.reserves.ore = N.from(1000);
  areas = rows();
  assert.equal(areas.harbor.rateText, 'Waiting for provisions');
  assert.equal(areas.workshop.rateText, 'Waiting for ore');
  valid(state);
});
