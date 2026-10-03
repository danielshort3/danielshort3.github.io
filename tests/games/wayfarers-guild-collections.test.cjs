'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, P, N, clone, advance, fund, mature, claimTiers } = require('./helpers/wayfarers-progression.cjs');
const K = require('../../js/games/wayfarers-guild/collections.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const near = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= Math.max(1, Math.abs(a), Math.abs(b)) * tolerance, `${a} != ${b}`);
const act = (state, action) => { const result = Core.act(state, action); assert.ok(result.ok, result.message); return result; };
const buy = (state, action) => act(state, { ...action, quote: K.quote(state, action).token });
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
function collected() {
  const state = mature();
  act(state, { type: 'collection-unlock', kind: 'cards' });
  act(state, { type: 'collection-unlock', kind: 'equipment' });
  return state;
}
function ownedCard(state, id, copies = 0) { state.collection.cards[id] = { rank: 1, copies }; }
function stripCollection(state) { const copy = clone(state); copy.schemaVersion = 5; delete copy.collection; delete copy.areaSkills; return copy; }

for (const kind of ['network', 'retained']) {
  const fixture = require('./fixtures/wayfarers-v5-' + kind + '.json');
  test('released v5 ' + kind + ' migrates additively with no default equipment or production change', () => {
    const before = clone(fixture.state), state = Core.migrateState(fixture.state);
    assert.deepEqual(fixture.state, before);
    assert.deepEqual(stripCollection(state), fixture.state);
    assert.deepEqual(state.collection, K.create(state.createdAt));
    assert.deepEqual(Core.getRates(state), fixture.expected.rates);
    valid(state);
  });
  for (const seconds of [120, 28800]) test('unclaimed collection preserves exact v5 ' + kind + ' production and RNG for ' + seconds + ' seconds', () => {
    const state = Core.migrateState(fixture.state);
    advance(state, seconds);
    assert.deepEqual(stripCollection(state), fixture.expected['after' + seconds]);
    assert.deepEqual(state.collection, K.create(state.createdAt));
    valid(state);
  });
}

test('starter introductions require learned areas, grant once and never equip a bonus automatically', () => {
  const fresh = Core.createState(0), before = clone(fresh);
  assert.equal(Core.act(fresh, { type: 'collection-unlock', kind: 'cards' }).ok, false);
  assert.deepEqual(fresh, before);
  const state = mature(), rates = Core.getRates(state);
  act(state, { type: 'collection-unlock', kind: 'cards' });
  act(state, { type: 'collection-unlock', kind: 'equipment' });
  assert.deepEqual(Core.getRates(state), rates);
  assert.ok(state.collection.decks.every(deck => deck.slots.every(id => id === null)));
  assert.ok(Object.values(state.collection.equipped).every(id => id === null));
  const claimed = clone(state);
  assert.equal(Core.act(state, { type: 'collection-unlock', kind: 'equipment' }).ok, false);
  assert.deepEqual(state, claimed);
  valid(state);
});

test('named saved decks allow shared ownership but prohibit a repeated card inside one deck', () => {
  const state = collected();
  act(state, { type: 'deck-name', id: 'deck-2', name: 'Ore orders' });
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'trail-courier' });
  act(state, { type: 'card-equip', deckId: 'deck-2', slot: 0, id: 'trail-courier' });
  const before = clone(state);
  assert.equal(Core.act(state, { type: 'card-equip', deckId: 'deck-1', slot: 1, id: 'trail-courier' }).ok, false);
  assert.deepEqual(state, before);
  act(state, { type: 'card-equip', deckId: 'deck-2', slot: 1, id: 'quarry-mole' });
  const picks = P.rawRates(state).areas.quarry.picks;
  act(state, { type: 'deck-select', id: 'deck-2' });
  near(P.rawRates(state).areas.quarry.picks, picks * 1.1);
  valid(state);
});

test('fusion consumes exact duplicates without rarity rolls or breaking any saved deck reference', () => {
  const state = collected(); state.collection.cards['trail-courier'].copies = 18;
  for (const deck of state.collection.decks) act(state, { type: 'card-equip', deckId: deck.id, slot: 0, id: 'trail-courier' });
  const random = { loot: state.collection.lootRng, scroll: state.collection.scrollRng };
  for (const cost of [2, 3, 5, 8]) { assert.equal(K.quote(state, { type: 'card-fuse', id: 'trail-courier' }).costCopies, cost); buy(state, { type: 'card-fuse', id: 'trail-courier' }); }
  assert.deepEqual(state.collection.cards['trail-courier'], { rank: 5, copies: 0 });
  assert.ok(state.collection.decks.every(deck => deck.slots[0] === 'trail-courier'));
  assert.deepEqual({ loot: state.collection.lootRng, scroll: state.collection.scrollRng }, random);
  const before = clone(state); assert.equal(Core.act(state, { type: 'card-fuse', id: 'trail-courier', quote: 'old' }).ok, false); assert.deepEqual(state, before);
  valid(state);
});

test('Archive Ink spends only loose duplicates and crafts only previously discovered cards', () => {
  const state = collected(); state.collection.cards['trail-courier'].copies = 10;
  for (let n = 0; n < 10; n += 1) buy(state, { type: 'card-recycle', id: 'trail-courier' });
  assert.equal(state.collection.ink, 10); assert.equal(state.collection.cards['trail-courier'].rank, 1);
  assert.deepEqual(state.collection.recent.at(-1).consumed, { copies: 1 });
  assert.deepEqual(state.collection.recent.at(-1).delta, { copies: -1, ink: 1 });
  assert.equal(act(state, { type: 'deck-select', id: 'deck-2' }).collectionEvent, null);
  const before = clone(state);
  assert.equal(K.quote(state, { type: 'card-recycle', id: 'trail-courier' }).disabled, true);
  assert.equal(Core.act(state, { type: 'card-craft', id: 'harbor-leviathan', quote: K.quote(state, { type: 'card-craft', id: 'harbor-leviathan' }).token }).ok, false);
  assert.deepEqual(state, before);
  buy(state, { type: 'card-craft', id: 'trail-courier' });
  assert.equal(state.collection.ink, 0); assert.equal(state.collection.cards['trail-courier'].copies, 1);
  valid(state);
});

test('gear crafting is a guaranteed complete ordinary-resource transaction and slot assignments are checked', () => {
  const state = collected(), action = { type: 'gear-forge', id: 'quarry-pick' }, q = K.quote(state, action);
  state.resources.ore = N.sub(q.cost.ore, 1);
  const before = clone(state); assert.equal(Core.act(state, { ...action, quote: q.token }).ok, false); assert.deepEqual(state, before);
  fund(state); buy(state, action);
  assert.equal(Core.act(state, { type: 'gear-equip', slot: 'head', id: 'quarry-pick' }).ok, false);
  const base = P.rawRates(state).areas.quarry.picks;
  act(state, { type: 'gear-equip', slot: 'tool', id: 'quarry-pick' });
  near(P.rawRates(state).areas.quarry.picks, base * 1.08);
  valid(state);
});

test('scroll success, failure and restoration conserve the six slots and cannot reuse a stale quote', () => {
  const state = collected(), id = 'trail-boots';
  act(state, { type: 'gear-equip', slot: 'boots', id });
  const action = { type: 'gear-scroll', id, scrollId: 'steady' }, old = K.quote(state, action).token;
  act(state, { ...action, quote: old });
  assert.equal(K.points(state.collection.gear[id]), 1);
  const after = clone(state); assert.equal(Core.act(state, { ...action, quote: old }).ok, false); assert.deepEqual(state, after);
  state.collection.scrollRng = 123456789;
  buy(state, { type: 'gear-scroll', id, scrollId: 'bold' });
  assert.equal(state.collection.gear[id].failed, 1);
  assert.equal(K.points(state.collection.gear[id]), 1);
  const rng = state.collection.scrollRng;
  buy(state, { type: 'gear-scroll', id, scrollId: 'restoration' });
  assert.equal(state.collection.gear[id].failed, 0);
  assert.equal(K.used(state.collection.gear[id]), 1);
  assert.equal(state.collection.scrollRng, rng);
  assert.equal(state.collection.scrolls.restoration, 0);
  valid(state);
});

test('all six successful slots remain permanent and restoration never removes a successful enhancement', () => {
  const state = collected(), id = 'trail-boots'; state.collection.scrolls.steady = 6;
  for (let n = 0; n < 6; n += 1) buy(state, { type: 'gear-scroll', id, scrollId: 'steady' });
  assert.equal(K.points(state.collection.gear[id]), 6);
  assert.equal(K.quote(state, { type: 'gear-scroll', id, scrollId: 'restoration' }).disabled, true);
  assert.equal(K.quote(state, { type: 'gear-scroll', id, scrollId: 'bold' }).disabled, true);
  valid(state);
});

test('Workshop Reforge atomically pays earned scrolls and materials and explicitly replaces enhancements', () => {
  const state = collected(), id = 'trail-boots';
  advance(state, 1);
  assert.ok(state.expedition.areas.harbor.voyages.length > 0);
  assert.equal(K.quote(state, { type: 'gear-reforge', id }).disabled, true, 'unused equipment needs no reforge');
  act(state, { type: 'gear-equip', slot: 'boots', id });
  buy(state, { type: 'gear-scroll', id, scrollId: 'steady' });
  state.collection.scrollRng = 123456789; buy(state, { type: 'gear-scroll', id, scrollId: 'bold' });
  state.collection.scrolls.restoration = 3;
  const action = { type: 'gear-reforge', id }, q = K.quote(state, action), before = clone(state);
  assert.equal(Core.act(state, { ...action, quote: q.token }).ok, false);
  assert.deepEqual(state, before);
  state.resources.ore = N.sub(q.cost.ore, 1);
  const short = clone(state);
  assert.equal(Core.act(state, { ...action, quote: q.token, confirm: true }).ok, false);
  assert.deepEqual(state, short); fund(state);
  state.collection.scrolls.restoration = 2;
  assert.equal(K.quote(state, action).disabled, true);
  state.collection.scrolls.restoration = 3;
  const itemView = K.view(state).equipment.items.find(item => item.id === id).reforge;
  assert.equal(itemView.pointsLost, 1); assert.equal(itemView.slotsAfter, 6); assert.equal(itemView.scrollCost, 3);
  assert.match(itemView.currentEffectText, /9.2%/); assert.match(itemView.nextEffectText, /8%/);
  assert.ok(itemView.impact.some(row => row.metric === 'guild:trailTravel' && N.cmp(row.nextValue, row.currentValue) < 0));
  const paid = clone(state), resources = clone(state.resources);
  act(state, itemView.action);
  assert.deepEqual(state.collection.gear[id], { successes: { steady: 0, bold: 0, brilliant: 0 }, failed: 0 });
  assert.equal(state.collection.equipped.boots, id);
  assert.equal(state.collection.scrolls.restoration, 0);
  assert.equal(state.collection.scrollRng, paid.collection.scrollRng);
  assert.equal(state.collection.lootRng, paid.collection.lootRng);
  assert.deepEqual(state.expedition.areas.harbor.voyages, paid.expedition.areas.harbor.voyages);
  for (const [key, amount] of Object.entries(q.cost)) assert.deepEqual(state.resources[key], N.sub(resources[key], amount));
  assert.equal(state.collection.recent.at(-1).outcome, 'reforged');
  const reforged = clone(state); assert.equal(Core.act(state, itemView.action).ok, false); assert.deepEqual(state, reforged);
  const collection = clone(state.collection); act(state, { type: 'refit' }); assert.deepEqual(state.collection, collection);
  valid(state);
  const legacy = Core.migrateState(require('./fixtures/wayfarers-v5-retained.json').state);
  act(legacy, { type: 'collection-unlock', kind: 'equipment' }); buy(legacy, { type: 'gear-scroll', id, scrollId: 'steady' });
  legacy.collection.scrolls.restoration = 3;
  assert.equal(K.quote(legacy, action).disabled, true, 'a route number never substitutes for discovering Workshop');
});

test('all collection clocks, inventories, decks and scroll RNG survive real Refit and Charter without regranting starters', () => {
  const state = collected(); act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'quarry-mole' });
  buy(state, { type: 'gear-scroll', id: 'trail-boots', scrollId: 'bold' }); advance(state, 321);
  const collection = clone(state.collection);
  act(state, { type: 'refit' }); assert.deepEqual(state.collection, collection);
  const target = 5 + state.lifetime.charters * 2;
  while (state.run.completed < target) { fund(state); for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned) while (area.ranks[id] < 25) act(state, { type: 'expedition-buy', areaId, id }); if (state.expedition.completed) act(state, { type: 'expedition-next' }); advance(state, 60); }
  const before = clone(state.collection); act(state, { type: 'charter' }); assert.deepEqual(state.collection, before);
  valid(state);
});

test('collection loot and scroll supply are deterministic through offline partitions and reload', () => {
  const batch = collected(), ticks = clone(batch); advance(batch, 28800);
  for (let index = 0; index < 480; index += 1) advance(ticks, 60);
  const a = clone(batch.collection), b = clone(ticks.collection);
  for (const key of ['cardRemainingMs', 'scrollRemainingMs', 'cardEligibleMs', 'scrollEligibleMs']) { near(a[key], b[key]); delete a[key]; delete b[key]; }
  assert.deepEqual(a, b);
  const restored = Core.normalizeState(JSON.parse(JSON.stringify(ticks)));
  advance(restored, 3600); advance(ticks, 3600); assert.deepEqual(restored.collection, ticks.collection);
  valid(restored);
});

test('pity awards an eligible Legendary without touching premium or relic RNG streams', () => {
  const state = collected(); state.collection.sinceLegendary = 79; state.collection.cardRemainingMs = 1;
  const legacyRng = [state.premium.rng, state.luck.rng, state.caravan.rng];
  advance(state, .001);
  assert.equal(state.collection.recent.at(-1).rarity, 'legendary');
  assert.equal(state.collection.sinceLegendary, 0);
  assert.deepEqual([state.premium.rng, state.luck.rng, state.caravan.rng], legacyRng);
  valid(state);
});

test('hauling card helps a haul bottleneck but cannot bypass a furnace bottleneck', () => {
  const state = collected(); ownedCard(state, 'quarry-hauler');
  const a = state.expedition.areas.quarry; Object.assign(a.ranks, { picks: 100, carts: 0, furnace: 400 }); Object.assign(a.highRanks, { picks: 100, furnace: 400 }); a.buffers = { input: 0, output: 0 };
  const before = Core.getRates(state).gain.ore;
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'quarry-hauler' });
  assert.ok(N.cmp(Core.getRates(state).gain.ore, before) > 0);
  a.ranks.furnace = 0;
  const constrained = Core.getRates(state).gain.ore;
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: null });
  assert.equal(N.cmp(Core.getRates(state).gain.ore, constrained), 0);
  valid(state);
});

test('Clockwork card changes actual Workshop input demand, not a cosmetic final-wallet multiplier', () => {
  const state = collected(); ownedCard(state, 'workshop-clockwork');
  const before = P.rawRates(state).areas.workshop;
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'workshop-clockwork' });
  const after = P.rawRates(state).areas.workshop;
  near(after.assembly / before.assembly, 1.08);
  near((after.demand / after.flow) / (before.demand / before.flow), .88);
  valid(state);
});

test('Tower Focus keeps card and hood knowledge bonuses exact and additive, without double stacking', () => {
  const state = collected(); ownedCard(state, 'tower-scribe');
  buy(state, { type: 'gear-forge', id: 'survey-hood' });
  for (const focused of [false, true]) {
    state.expedition.focus.active = focused ? 'watchtower' : null;
    state.expedition.focus.remaining = focused ? 90 : 0;
    act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: null });
    act(state, { type: 'gear-equip', slot: 'head', id: null });
    const base = N.toNumber(Core.getRates(state).gain.knowledge);
    act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'tower-scribe' });
    near(N.toNumber(Core.getRates(state).gain.knowledge), base * 1.1);
    act(state, { type: 'gear-equip', slot: 'head', id: 'survey-hood' });
    near(N.toNumber(Core.getRates(state).gain.knowledge), base * 1.16);
    act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: null });
    near(N.toNumber(Core.getRates(state).gain.knowledge), base * 1.06);
  }
  state.expedition.focus.active = 'greenway'; state.expedition.focus.remaining = 90;
  const coins = N.toNumber(N.sub(Core.getRates(state).gain.coins, Core.getRates(state, true).gain.coins));
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'trail-courier' });
  near(N.toNumber(N.sub(Core.getRates(state).gain.coins, Core.getRates(state, true).gain.coins)), coins * 1.08);
  valid(state);
});

test('equipping cargo cards changes future voyage quotes but preserves all previously funded cargo', () => {
  const state = collected(); ownedCard(state, 'harbor-deckhand'); advance(state, 1);
  const paid = clone(state.expedition.areas.harbor.voyages), before = P.rawRates(state).areas.harbor.payout.coins;
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'harbor-deckhand' });
  assert.deepEqual(state.expedition.areas.harbor.voyages, paid);
  assert.ok(P.rawRates(state).areas.harbor.payout.coins > before);
  valid(state);
});

test('cargo previews expose the complete future provisioning bill, canonical payout and readiness loss', () => {
  const state = collected(); ownedCard(state, 'harbor-deckhand');
  Core.setPremiumEntitlements(state, ['compass']);
  const raw = P.rawRates(state);
  state.resources.provisions = N.from(raw.areas.harbor.supply * 1.001);
  const card = K.view(state).cards.find(card => card.id === 'harbor-deckhand');
  const action = card.equipActions.find(option => option.deckId === 'deck-1' && option.slot === 0);
  const bill = action.impact.find(row => row.metric === 'guild:manifestSupply');
  const ready = action.impact.find(row => row.metric === 'guild:manifestReady');
  const cargo = action.impact.find(row => row.metric === 'guild:cargo');
  assert.ok(N.cmp(bill.nextValue, bill.currentValue) > 0); assert.equal(bill.unit, ''); assert.equal(bill.direction, 'lower');
  assert.equal(ready.current, 'Ready'); assert.equal(ready.next, 'Needs supplies');
  act(state, action.action);
  const after = P.rawRates(state);
  near(N.toNumber(cargo.nextValue), N.toNumber(P.manifestPayout(state, after).coins));
  near(N.toNumber(bill.nextValue), after.areas.harbor.supply);
  assert.equal(after.areas.harbor.canLaunch, false);
  valid(state);
});

test('collection view and previews preserve state and paid ownership while showing actual capacity changes', () => {
  const state = collected(); Core.setPremiumEntitlements(state, ['compass', 'artisan']);
  const before = clone(state), rng = state.collection.scrollRng;
  const view = Core.getView(state).collection;
  assert.deepEqual(state, before); assert.equal(state.collection.scrollRng, rng);
  const boots = view.equipment.items.find(item => item.id === 'trail-boots');
  const effect = boots.impact.find(row => row.metric === 'guild:trailTravel');
  act(state, boots.equipAction);
  near(P.rawRates(state).areas.greenway.travel, N.toNumber(effect.nextValue));
  valid(state);
});

test('strict validation rejects impossible collection histories, references, ownership and slot counts', () => {
  const seed = collected();
  const mutations = [s => { s.collection.extra = true; }, s => { s.collection.cards['fake'] = { rank: 1, copies: 0 }; }, s => { s.collection.cards['trail-courier'].rank = 6; }, s => { s.collection.decks[0].slots[0] = 'harbor-leviathan'; }, s => { s.collection.decks[0].slots = ['trail-courier', 'trail-courier', null, null]; }, s => { s.collection.equipped.head = 'trail-boots'; }, s => { s.collection.gear['trail-boots'].successes.brilliant = 7; }, s => { s.collection.gear['trail-boots'].points = 999; }, s => { s.collection.ink = -1; }, s => { s.collection.scrollRng = 0; }, s => { s.collection.recent[0].delta.points = Infinity; }];
  for (const mutate of mutations) { const state = clone(seed); mutate(state); assert.equal(Core.validateState(state).valid, false); }
  for (const id of ['constructor', '__proto__', 'toString', 'unknown']) {
    const state = clone(seed), before = clone(state);
    assert.equal(Core.act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id }).ok, false);
    for (const type of ['card-fuse', 'card-craft', 'card-recycle', 'gear-forge', 'gear-reforge']) {
      assert.equal(K.quote(state, { type, id }).disabled, true);
      assert.equal(Core.act(state, { type, id, quote: K.quote(state, { type, id }).token }).ok, false);
    }
    assert.deepEqual(state, before);
  }
});

test('malformed saved deck and equipment references reject without throwing and preserve rejected bytes', () => {
  const seed = collected();
  for (const id of ['constructor', '__proto__', 'toString', {}, [], 17, false]) for (const target of ['card', 'gear']) {
    const state = clone(seed);
    if (target === 'card') state.collection.decks[0].slots[0] = id;
    else state.collection.equipped.tool = id;
    assert.doesNotThrow(() => Core.validateState(state));
    assert.equal(Core.validateState(state).valid, false);
    const text = JSON.stringify({ format: Storage.FORMAT, version: Storage.VERSION, savedAt: state.lastUpdate, state });
    const values = new Map([[Storage.SAVE_KEY, text]]), storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
    const store = Storage.createStore({ storage, now: () => state.lastUpdate });
    assert.equal(store.load({ deferOffline: true }).canSave, false);
    assert.equal(store.inspectImport(text).ok, false);
    assert.equal(store.save(seed).ok, false);
    assert.equal(values.get(Storage.SAVE_KEY), text);
  }
});

test('drop odds reflect the available pool and next pity draw; unequipped scrolls expose real item stats', () => {
  const state = collected();
  state.collection.sinceLegendary = 79;
  assert.deepEqual(K.view(state).acquisition.odds, { common: 0, rare: 0, epic: 0, legendary: 100 });
  state.collection.sinceLegendary = 0; state.collection.sinceEpic = 11;
  assert.deepEqual(K.view(state).acquisition.odds, { common: 0, rare: 0, epic: 100, legendary: 0 });
  state.collection.sinceEpic = 0;
  const boots = K.view(state).equipment.items.find(item => item.id === 'trail-boots');
  const steady = boots.scrolls.find(scroll => scroll.id === 'steady');
  assert.deepEqual(steady.impact, []);
  assert.match(steady.currentEffectText, /8% Trail travel/);
  assert.match(steady.nextEffectText, /9.2% Trail travel/);
  const fresh = Core.migrateState(require('./fixtures/wayfarers-v4-fresh.json'));
  const odds = K.view(fresh).acquisition.odds;
  assert.equal(odds.legendary, 0);
  near(odds.common + odds.rare + odds.epic, 100);
});

test('current saves retain card inventories and reject future envelopes without sacrificing the old bytes', () => {
  const state = collected(), values = new Map(), storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const store = Storage.createStore({ storage, now: () => state.lastUpdate });
  assert.ok(store.save(state).ok);
  assert.equal(JSON.parse(values.get(Storage.SAVE_KEY)).version, Core.VERSION);
  assert.deepEqual(store.load({ deferOffline: true }).state, state);
  const unsupported = JSON.stringify({ format: Storage.FORMAT, version: Core.VERSION + 1, savedAt: state.lastUpdate, state });
  values.set(Storage.SAVE_KEY, unsupported);
  const protectedStore = Storage.createStore({ storage, now: () => state.lastUpdate });
  assert.equal(protectedStore.load({ deferOffline: true }).canSave, false);
  assert.equal(protectedStore.save(Core.createState(0)).ok, false);
  assert.equal(values.get(Storage.SAVE_KEY), unsupported);
  valid(state);
});

test('old fresh saves unlock introductions from actual persistent areas rather than the next route index', () => {
  const state = Core.migrateState(require('./fixtures/wayfarers-v4-fresh.json'));
  assert.deepEqual(Object.keys(state.expedition.areas), ['greenway']);
  assert.equal(K.view(state).cardsAvailable, false);
  assert.equal(K.view(state).equipmentAvailable, false);
  const before = clone(state);
  assert.equal(Core.act(state, { type: 'collection-unlock', kind: 'cards' }).ok, false);
  assert.deepEqual(state, before);
});

test('earned deck slots are gated and announce once without filling a new slot', () => {
  const state = Core.createState(0); fund(state);
  while (!state.expedition.completed || !P.canAdvance(state)) {
    claimTiers(state);
    for (const id of state.expedition.areas.greenway.learned) Core.act(state, { type: 'expedition-buy', areaId: 'greenway', id });
    advance(state, 30);
  }
  act(state, { type: 'expedition-next' });
  act(state, { type: 'collection-unlock', kind: 'cards' });
  assert.equal(K.slots(state), 2);
  assert.equal(Core.act(state, { type: 'card-equip', deckId: 'deck-1', slot: 2, id: 'trail-courier' }).ok, false);
  while (!state.expedition.completed || !P.canAdvance(state)) {
    claimTiers(state);
    for (const id of state.expedition.areas.quarry.learned) Core.act(state, { type: 'expedition-buy', areaId: 'quarry', id });
    advance(state, 30);
  }
  act(state, { type: 'expedition-next' });
  assert.equal(K.slots(state), 3);
  assert.equal(state.collection.recent.at(-1).delta.deckSlots, 3);
  const sequence = state.collection.sequence; K.syncUnlocks(state);
  assert.equal(state.collection.sequence, sequence);
  assert.ok(state.collection.decks.every(deck => deck.slots[2] === null));
  const late = collected(); assert.equal(K.slots(late), 4); assert.ok(late.collection.recent.some(event => event.delta.deckSlots === 4));
  valid(state); valid(late);
});

test('per-slot previews include displaced cards, matching tags and unequipped fusion strength', () => {
  const state = collected(); ownedCard(state, 'quarry-hauler', 2);
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'trail-courier' });
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 1, id: 'quarry-mole' });
  const view = K.view(state), item = view.cards.find(card => card.id === 'quarry-hauler');
  assert.match(item.fusion.nextEffectText, /18% hauling/);
  assert.deepEqual(item.fusion.impact, []);
  const replacement = item.equipActions.find(action => action.deckId === 'deck-1' && action.slot === 0);
  assert.ok(replacement.impact.some(row => row.metric === 'guild:coins' && N.cmp(row.nextValue, row.currentValue) < 0));
  assert.ok(replacement.impact.every(row => N.cmp(row.nextValue, row.currentValue) !== 0));
  act(state, replacement.action);
  assert.equal(K.view(state).synergies.find(row => row.id === 'industry').count, 2);
  near(K.modifiers(state).oreSaving, .04);
  valid(state);
});

test('explicit legacy travel gear increases real retained Trail work and explains the compatible rules', () => {
  const E = require('../../js/games/wayfarers-guild/expeditions.js');
  const state = Core.migrateState(require('./fixtures/wayfarers-v5-retained.json').state);
  act(state, { type: 'collection-unlock', kind: 'equipment' });
  const before = E.rates(state).greenway.travel;
  const item = K.view(state).equipment.items.find(item => item.id === 'trail-boots');
  assert.match(item.effectText, /Retained run: \+8% travel/);
  act(state, item.equipAction);
  near(E.rates(state).greenway.travel, before * 1.08);
  valid(state);
});

test('retained-run collection discoveries survive explicit adoption without inventing later area unlocks', () => {
  const state = Core.migrateState(require('./fixtures/wayfarers-v5-retained.json').state);
  act(state, { type: 'collection-unlock', kind: 'cards' });
  act(state, { type: 'collection-unlock', kind: 'equipment' });
  assert.equal(K.slots(state), 3);
  assert.ok(K.view(state).cards.every(card => ['greenway', 'quarry', 'watchtower'].includes(card.area)));
  advance(state, 86400);
  act(state, { type: 'buy', id: 'gear-tools' });
  act(state, { type: 'buy', id: 'gear-boots' });
  const collection = clone(state.collection);
  act(state, { type: 'refit' });
  assert.deepEqual(state.collection, collection);
  assert.equal(K.slots(state), 3);
  valid(state);
});

test('equipped cards and enhancements keep six-area wallet, cargo and loot partition invariants', () => {
  const batch = collected();
  // The funded factory completes century-scale commission work in single ticks.
  // Keep its weather phase, using a normal elapsed clock for this play witness.
  for (const area of Object.values(batch.expedition.areas)) area.elapsed = area.elapsed % 32400 + 32400;
  for (const id of ['quarry-hauler', 'workshop-clockwork', 'harbor-navigator', 'tower-astronomer']) ownedCard(batch, id);
  ['quarry-hauler', 'workshop-clockwork', 'harbor-navigator', 'tower-astronomer'].forEach((id, slot) => act(batch, { type: 'card-equip', deckId: 'deck-1', slot, id }));
  act(batch, { type: 'gear-equip', slot: 'boots', id: 'trail-boots' }); buy(batch, { type: 'gear-scroll', id: 'trail-boots', scrollId: 'bold' });
  const ticks = clone(batch); advance(batch, 28800);
  for (let n = 0; n < 480; n += 1) advance(ticks, 60);
  for (const key of Object.keys(batch.resources)) near(N.toNumber(batch.resources[key]), N.toNumber(ticks.resources[key]), 1e-8);
  for (const key of ['lootRng', 'scrollRng', 'cards', 'scrolls', 'gear', 'cardFinds', 'scrollFinds']) assert.deepEqual(batch.collection[key], ticks.collection[key]);
  assert.equal(batch.expedition.areas.harbor.deliveries, ticks.expedition.areas.harbor.deliveries);
  valid(batch); valid(ticks);
});

test('a reserved caravan quote stays frozen through equipment, fusion and deck changes', () => {
  const state = collected(); state.collection.cards['trail-courier'].copies = 2;
  state.caravan.sequence += 1; state.caravan.remainingMs = 0;
  state.caravan.offer = { id: 'wg-collection-reserved-arrival', golden: true, minutes: 100, arrivedAt: state.lastUpdate, quote: null, locked: false };
  act(state, { type: 'caravan-select', kind: 'shipment', material: 'ore' });
  const quote = clone(Core.getCaravanQuote(state)); assert.ok(Core.beginCaravanReward(state, quote).ok);
  act(state, { type: 'card-equip', deckId: 'deck-1', slot: 0, id: 'trail-courier' });
  buy(state, { type: 'card-fuse', id: 'trail-courier' });
  act(state, { type: 'gear-equip', slot: 'boots', id: 'trail-boots' });
  buy(state, { type: 'gear-scroll', id: 'trail-boots', scrollId: 'steady' });
  assert.deepEqual(Core.getCaravanQuote(state), quote);
  const before = clone(state.resources);
  const receipt = { receiptId: 'wg-collection-verified-receipt', offerId: quote.offerId, quote, completedAt: state.lastUpdate };
  assert.ok(Core.grantCaravanReward(state, receipt).ok);
  for (const [id, amount] of Object.entries(quote.reward.resources)) assert.deepEqual(state.resources[id], N.add(before[id], amount));
  valid(state);
});
