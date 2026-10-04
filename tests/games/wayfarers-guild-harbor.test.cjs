'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, P, N, clone, advance, mature } = require('./helpers/wayfarers-progression.cjs');
const near = (a, b) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(b)) * 1e-8, `${a} != ${b}`);
const act = (state, action) => assert.equal(Core.act(state, action).ok, true);
const port = (state, id) => act(state, { type: 'expedition-config', areaId: 'harbor', kind: 'port', slot: 0, id });
function departed(destination = 'ocean') {
  const state = mature(), harbor = state.expedition.areas.harbor;
  harbor.voyages = [];
  harbor.elapsed = 0;
  port(state, destination);
  P.autoBuy(state);
  assert.equal(harbor.voyages.length, 2);
  assert.ok(harbor.voyages.every(voyage => voyage.convoy >= 1 && voyage.port === destination));
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  return state;
}

test('selecting the next port changes neither saved cargo nor actual speed of ships already at sea', () => {
  const state = departed(), changed = clone(state), manifest = clone(state.expedition.areas.harbor.voyages);
  port(changed, 'coast');
  assert.deepEqual(changed.expedition.areas.harbor.voyages, manifest);
  advance(state, .01); advance(changed, .01);
  for (let i = 0; i < manifest.length; i += 1) {
    const original = state.expedition.areas.harbor.voyages[i], nextPort = changed.expedition.areas.harbor.voyages[i];
    near(nextPort.work, original.work);
    assert.deepEqual(nextPort.payout, manifest[i].payout);
    assert.equal(nextPort.port, 'ocean');
    assert.equal(nextPort.convoy, manifest[i].convoy);
  }
});

test('arrival snaps accumulated sub-micro-work roundoff without delivering a visibly unfinished voyage', () => {
  const rounded = departed(), unfinished = clone(rounded);
  for (const voyage of rounded.expedition.areas.harbor.voyages) voyage.work = voyage.target - 2e-7;
  for (const voyage of unfinished.expedition.areas.harbor.voyages) voyage.work = voyage.target - 0.01;
  const before = clone(rounded.resources.coins), manifests = clone(rounded.expedition.areas.harbor.voyages);
  P.tick(rounded, 1e-12); P.tick(unfinished, 1e-12);
  assert.equal(rounded.expedition.areas.harbor.voyages.length, 0);
  assert.equal(unfinished.expedition.areas.harbor.voyages.length, manifests.length);
  const expected = manifests.reduce((total, voyage) => N.add(total, voyage.payout.coins), before);
  assert.deepEqual(rounded.resources.coins, expected);
  P.tick(rounded, 1e-12);
  assert.deepEqual(rounded.resources.coins, expected, 'a snapped arrival cannot pay twice');
});

test('concurrent coastal and ocean manifests experience weather on their own saved routes', () => {
  const state = mature(), harbor = state.expedition.areas.harbor;
  harbor.voyages = []; harbor.elapsed = 0;
  harbor.ranks['fleet-command'] = 0;
  port(state, 'ocean'); P.autoBuy(state);
  assert.equal(harbor.voyages.length, 1);
  act(state, { type: 'expedition-buy', areaId: 'harbor', id: 'fleet-command' });
  port(state, 'coast'); P.autoBuy(state);
  assert.deepEqual(harbor.voyages.map(voyage => voyage.port), ['ocean', 'coast']);
  const storm = clone(state);
  storm.expedition.areas.harbor.elapsed = 21600;
  P.tick(state, .01); P.tick(storm, .01);
  near(storm.expedition.areas.harbor.voyages[0].work / harbor.voyages[0].work, .65);
  near(storm.expedition.areas.harbor.voyages[1].work / harbor.voyages[1].work, 1);
  assert.deepEqual(Core.validateState(storm), { valid: true, errors: [] });
});

test('crossing a weather boundary in one offline step agrees with foreground partitions', () => {
  const state = departed();
  state.expedition.areas.harbor.elapsed = 21599.99;
  const partitioned = clone(state);
  advance(state, .02);
  advance(partitioned, .01); advance(partitioned, .01);
  const a = state.expedition.areas.harbor, b = partitioned.expedition.areas.harbor;
  assert.equal(a.voyages.length, b.voyages.length);
  a.voyages.forEach((voyage, i) => {
    near(voyage.work, b.voyages[i].work);
    assert.deepEqual(voyage.payout, b.voyages[i].payout);
  });
  for (const id of Object.keys(state.resources)) near(N.toNumber(state.resources[id]), N.toNumber(partitioned.resources[id]));
});

test('Fox affects new manifests exactly once and cannot reprice a paid manifest', () => {
  const state = mature(), fox = clone(state);
  state.expedition.areas.harbor.voyages = [];
  fox.expedition.areas.harbor.voyages = [];
  fox.crew.companions.push('fox'); fox.crew.companion = 'fox';
  P.autoBuy(state); P.autoBuy(fox);
  const base = state.expedition.areas.harbor.voyages, boosted = fox.expedition.areas.harbor.voyages;
  near(N.toNumber(N.div(boosted[0].payout.coins, base[0].payout.coins)), 1.3);
  near(N.toNumber(N.div(boosted[0].payout.maps, base[0].payout.maps)), 1);
  const paid = clone(boosted);
  fox.crew.companion = null;
  advance(fox, .01);
  boosted.forEach((voyage, i) => assert.deepEqual(voyage.payout, paid[i].payout));
});

test('Focus accelerates the paid manifest without multiplying its cargo or the next departure bill', () => {
  const state = departed(), boosted = clone(state);
  const quote = P.rawRates(state).areas.harbor, paid = clone(state.expedition.areas.harbor.voyages);
  act(boosted, { type: 'expedition-focus', areaId: 'harbor', id: 'priority' });
  const focus = P.rawRates(boosted).areas.harbor;
  near(focus.supply, quote.supply);
  assert.deepEqual(focus.payout, quote.payout);
  P.tick(state, .01); P.tick(boosted, .01);
  state.expedition.areas.harbor.voyages.forEach((voyage, i) => {
    const current = boosted.expedition.areas.harbor.voyages[i];
    near(current.work / voyage.work, P.Content.FOCUS.multiplier);
    assert.deepEqual(current.payout, paid[i].payout);
    assert.equal(current.supplies, paid[i].supplies);
  });
});

for (const resource of ['ore', 'provisions']) test(`a ${resource} delivery at the exact protected-reserve boundary is conserved`, () => {
  const state = mature(), areas = state.expedition.areas;
  if (resource === 'provisions') {
    for (let i = 0; !state.rooms.includes('kitchen') && i < 60; i += 1) {
      if (state.expedition.completed) act(state, { type: 'expedition-next' });
      advance(state, 60);
    }
    assert.ok(state.rooms.includes('kitchen'));
    state.meal = 'meal-study'; state.guild.supply = 'push';
  }
  areas.harbor.voyages = [];
  areas.harbor.choice = resource === 'ore' ? 'materials' : 'commerce';
  P.autoBuy(state);
  assert.equal(areas.harbor.voyages.length, 2);
  areas.workshop.choice = 'manufacture';
  areas.workshop.plans.templates = ['instruments'];
  areas.workshop.ranks.assembly = 1000; areas.workshop.highRanks.assembly = 1000;
  if (resource === 'ore') {
    for (const id of Object.keys(areas.quarry.ranks)) areas.quarry.ranks[id] = 0;
    areas.quarry.buffers = { input: 0, output: 0 };
  }
  state.guild.plan.reserves[resource] = N.from(1);
  const rates = Core.getRates(state), raw = P.rawRates(state), seconds = .01;
  const netDrain = N.toNumber(N.sub(rates.drain[resource], rates.gain[resource]));
  assert.ok(netDrain > 0, 'the real selected production chain reaches its reserve');
  for (const voyage of areas.harbor.voyages) {
    const condition = raw.areas.harbor.weather === 2 && voyage.port === 'ocean' ? .65 : raw.areas.harbor.weather === 1 && voyage.port === 'coast' ? .85 : 1;
    voyage.work = voyage.target - raw.areas.harbor.voyagePace * condition / voyage.convoy * seconds;
  }
  state.resources[resource] = N.from(1 + netDrain * seconds);
  const expected = areas.harbor.voyages.reduce((total, voyage) => total + N.toNumber(voyage.payout[resource] || 0), 1);
  assert.ok(expected > 1);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  advance(state, seconds);
  const reinvested = resource === 'provisions' ? areas.harbor.voyages.reduce((total, voyage) => total + voyage.supplies, 0) : 0;
  near(N.toNumber(state.resources[resource]) + reinvested, expected);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
});
