'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const H = require('./helpers/wayfarers-progression.cjs');
const { Core, N, P, clone, mature, advance, fund, claimTiers } = H;
const O = require('../../js/games/wayfarers-guild/onboarding.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const Collection = require('../../js/games/wayfarers-guild/collections.js');
const valid = s => assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
const act = (s, action) => { const result = Core.act(s, action); assert.ok(result.ok, JSON.stringify(action) + ': ' + result.message); valid(s); return result; };
const view = s => Core.getView(s).onboarding;
const active = s => view(s).active;
const explain = s => { const ids = []; while (active(s)?.mode === 'currency') { ids.push(active(s).currencyInfo.id); act(s, active(s).ackAction); } return ids; };
function memory() { const map = new Map(); return { getItem: k => map.get(k) || null, setItem: (k, v) => map.set(k, String(v)), removeItem: k => map.delete(k) }; }

test('currency Next is receipt-only, precedes the real price inspection, and never buys or funds a rank', () => {
  const s = Core.createState(123); act(s, { type: 'onboarding-visit', id: 'greenway' });
  const info = active(s), before = clone(s); assert.equal(info.mode, 'currency'); assert.equal(info.currencyInfo.id, 'coins');
  assert.match(info.currencyInfo.coverageText, /guild supplies/i); assert.equal(info.canLeave, false); assert.equal(info.practiceAction, undefined);
  assert.equal(Core.act(s, { type: 'expedition-buy', areaId: 'greenway', id: 'boots', count: 1 }).ok, false); assert.deepEqual(s, before);
  act(s, info.ackAction); before.onboarding.practice.currencyRead.push('coins'); assert.deepEqual(s, before);
  assert.equal(Core.act(s, info.ackAction).ok, false); assert.equal(active(s).mode, 'inspect');
  act(s, active(s).inspectAction); const purchase = active(s); assert.equal(purchase.costCoverage, 'guild'); assert.deepEqual(purchase.walletSpend, {});
  assert.equal(N.toNumber(purchase.practicePreview.cost[0].amount), 6);
  act(s, purchase.practiceAction); assert.equal(s.expedition.areas.greenway.ranks.boots, 1); assert.deepEqual(s.resources.coins, N.zero());
});

test('all quoted currencies are introduced before a multi-resource action; unrelated currencies stay hidden', () => {
  const s = mature(); s.onboarding.practice.currencyRead = [];
  act(s, { type: 'onboarding-visit', id: 'crew', intendedAction: { type: 'recruit', kind: 'specialist' } });
  const resources = clone(s.resources), supplies = clone(s.onboarding.practice.supplies);
  assert.deepEqual(explain(s), ['coins', 'knowledge']); assert.deepEqual(s.resources, resources); assert.deepEqual(s.onboarding.practice.supplies, supplies);
  assert.equal(active(s).mode, 'inspect'); assert.deepEqual(s.onboarding.practice.currencyRead, ['coins', 'knowledge']);
});

test('retained E2 guild and area prices both require the same real currency explanation', () => {
  const s = Core.normalizeState(require('./fixtures/wayfarers-v4-state.json'), 600001);
  const row = Core.getView(s).globalUpgrades.find(r => r.action?.type === 'buy' && !r.maxed);
  act(s, { type: 'onboarding-visit', id: 'guild-upgrades', intendedAction: row.action });
  assert.equal(active(s).currencyInfo.id, row.cost[0].resource || row.cost[0].id);
  const snapshot = clone(s.resources); explain(s); act(s, active(s).inspectAction); act(s, active(s).practiceAction); assert.deepEqual(s.resources, { ...snapshot, coins: N.add(snapshot.coins, 8) });
});

test('currency receipts survive reload and same-guild import before unlock without importing the wallet', () => {
  const s = mature(), old = clone(s); act(s, { type: 'onboarding-visit', id: 'crew', intendedAction: { type: 'recruit', kind: 'specialist' } }); explain(s);
  const loaded = Core.normalizeState(clone(s), s.lastUpdate); assert.equal(active(loaded).mode, 'inspect');
  const store = Storage.createStore({storage:memory(), now:() => old.lastUpdate}); store.load(); assert.equal(store.save(s).ok, true);
  const result = store.replaceImport(store.export(old).text, { preservePracticeFrom: s }); assert.equal(result.ok, true); const imported = result.state || store.load(old.lastUpdate).state;
  assert.deepEqual(imported.resources, old.resources); assert.deepEqual(imported.onboarding.practice.currencyRead, ['coins', 'knowledge']); valid(imported);
});

test('first-use interception stays mandatory even when the same lesson is reopened from Help', () => {
  const s = mature(); act(s, { type: 'onboarding-visit', id: 'reserves', intendedAction: { type: 'plan-reserve', id: 'ore', amount: '33' } });
  assert.equal(active(s).mandatory, true); assert.equal(active(s).canLeave, false);
  const before = clone(s); assert.equal(Core.act(s, { type: 'onboarding-leave', id: 'reserves' }).ok, false); assert.deepEqual(s, before);
  act(s, { type: 'onboarding-visit', id: 'reserves' }); assert.equal(active(s).canLeave, false); assert.ok(s.onboarding.practice.intentions.reserves);
  const other = mature(); act(other, { type: 'onboarding-visit', id: 'reserves' }); assert.equal(active(other).canLeave, true); act(other, active(other).leaveAction);
});

test('unseen state changes only after deliberate earned-target inspection, never rendering or time', () => {
  const s = Core.createState(10), before = clone(s), first = view(s);
  assert.deepEqual(s, before); assert.equal(first.attention.items.find(r => r.id === 'upgrade:area:greenway:boots').unseen, true);
  const item = first.attention.items.find(r => r.id === 'upgrade:area:greenway:boots'); const bad = { ...item.inspectAction, id: 'upgrade:area:harbor:shipbuilding' };
  assert.equal(Core.act(s, bad).ok, false); assert.deepEqual(s, before);
  act(s, item.inspectAction); assert.equal(view(s).attention.items.find(r => r.id === item.id).unseen, false); assert.deepEqual(s.resources, before.resources);
  assert.equal(Core.act(s, item.inspectAction).ok, false); advance(s, 1); const loaded = Core.normalizeState(clone(s), s.lastUpdate);
  assert.equal(view(loaded).attention.items.find(r => r.id === item.id).unseen, false);
});

test('currency rail exposes only earned wallets and remembers zero balances with honest Starshard ownership', () => {
  const fresh = Core.createState(11); assert.deepEqual(view(fresh).currencies.map(r => r.id), ['coins']); advance(fresh, 1); assert.deepEqual(view(fresh).currencies.map(r => r.id), ['coins']);
  const s = mature(); O.sync(s); const ids = view(s).currencies.map(r => r.id); Object.keys(s.resources).forEach(id => { s.resources[id] = N.zero(); }); O.sync(s);
  assert.deepEqual(view(s).currencies.map(r => r.id), ids); const shards = view(s).currencies.find(r => r.id === 'starshards'); assert.equal(shards.balanceSource, 'earned-plus-verified-account'); assert.equal(shards.paidValue, null);
  assert.ok(!ids.includes('steady') && !ids.includes('copies') && !ids.includes('focus'));
});

test('opening an actual currency sheet records learning while a tutorial overlay needs only its Next', () => {
  const s = Core.createState(12), item = view(s).attention.items.find(r => r.id === 'currency:coins'); act(s, item.inspectAction);
  assert.deepEqual(s.onboarding.practice.currencyRead, ['coins']); act(s, { type: 'onboarding-visit', id: 'greenway' }); assert.equal(active(s).mode, 'inspect');
});

test('new upgrade claims and earlier-area options get badges without leaking locked future tracks', () => {
  const s = Core.createState(13); fund(s); act(s, { type: 'expedition-buy', id: 'boots' }); act(s, { type: 'expedition-buy', id: 'boots' });
  assert.ok(!view(s).attention.items.some(r => r.id === 'upgrade:area:greenway:porters'));
  act(s, { type: 'upgrade-tier-unlock', id: 'area:greenway:porters' }); assert.ok(view(s).attention.items.some(r => r.id === 'upgrade:area:greenway:porters' && r.unseen));
  const m = mature(); const options = view(m).attention.items.filter(r => r.kind === 'option');
  assert.ok(options.some(r => r.id === 'option:greenway:plan:continental')); assert.ok(options.some(r => r.id === 'option:quarry:plan:rich'));
  const target = options.find(r => r.id === 'option:greenway:plan:continental'); assert.equal(target.destination.optionId, 'continental');
  const snapshot = clone(m.expedition); act(m, target.inspectAction); assert.deepEqual(m.expedition, snapshot); assert.equal(view(m).attention.items.find(r => r.id === target.id).unseen, false);
});

test('migration baselines old visible content but preserves pending discoveries and unseen new items', () => {
  const s = mature(); act(s, { type: 'collection-unlock', kind: 'cards' });
  s.onboarding.read = s.onboarding.entries.slice(); s.onboarding.announced = s.onboarding.entries.slice(); delete s.onboarding.attention;
  const loaded = Core.normalizeState(clone(s), s.lastUpdate), rows = view(loaded).attention.items;
  assert.ok(rows.filter(r => r.kind === 'upgrade').every(r => !r.unseen));
  const eventCards = s.collection.recent.filter(r => r.id > s.collection.seen && r.cardId).map(r => r.cardId);
  assert.ok(eventCards.length); for (const id of eventCards) assert.ok(rows.find(r => r.id === 'card:' + id).unseen);
  const pending = Core.createState(14); fund(pending); act(pending, { type: 'expedition-buy', id: 'boots' }); act(pending, { type: 'expedition-buy', id: 'boots' }); delete pending.onboarding.attention;
  const migrated = Core.normalizeState(pending, pending.lastUpdate); assert.ok(view(migrated).attention.items.find(r => r.id === 'discovery:ready:area:greenway:porters').unseen);
});

test('attention and currency metadata reject unknown/prototype/duplicate records without throwing', () => {
  const base = Core.createState(15);
  const edits = [s => s.onboarding.practice.currencyRead = ['constructor'], s => s.onboarding.practice.currencyRead = ['coins', 'coins'], s => s.onboarding.attention.seen = ['card:__proto__'], s => s.onboarding.attention.seen = ['upgrade:made-up'], s => s.onboarding.attention.currencies = ['focus'], s => s.onboarding.attention.extra = true];
  for (const edit of edits) { const s = clone(base); edit(s); assert.doesNotThrow(() => Core.validateState(s)); assert.equal(Core.validateState(s).valid, false); }
});

test('Refit keeps learned receipts and rejects old inspection tokens; Testing reset starts fresh', () => {
  const s = mature(); const coin = view(s).attention.items.find(r => r.id === 'currency:coins'); act(s, coin.inspectAction);
  const old = view(s).attention.items.find(r => r.kind === 'upgrade').inspectAction;
  assert.ok(Core.getView(s).refit.available); act(s, { type: 'refit' }); assert.ok(s.onboarding.practice.currencyRead.includes('coins')); assert.ok(s.onboarding.attention.seen.includes('currency:coins'));
  assert.equal(Core.act(s, old).ok, false); const fresh = Core.createState(999); assert.deepEqual(fresh.onboarding.practice.currencyRead, []); assert.deepEqual(fresh.onboarding.attention.seen, []);
});

test('new duplicate copies re-mark only their item; capped result history cannot erase its pending inspection', () => {
  const s = mature(); act(s, { type: 'collection-unlock', kind: 'cards' }); s.collection.ink = 1000;
  const item = () => view(s).attention.items.find(r => r.id === 'card:trail-courier'); act(s, item().inspectAction); assert.equal(item().unseen, false);
  const oldToken = item().inspectAction; const craft = { type: 'card-craft', id: 'trail-courier' }; act(s, { ...craft, quote: Collection.quote(s, craft).token });
  assert.equal(item().unseen, true); assert.equal(Core.act(s, oldToken).ok, false);
  s.collection.recent = []; const loaded = Core.normalizeState(clone(s), s.lastUpdate); assert.equal(view(loaded).attention.items.find(r => r.id === item().id).unseen, true);
  act(s, item().inspectAction); assert.equal(item().unseen, false);
  const recycle = { type: 'card-recycle', id: 'trail-courier' }; act(s, { ...recycle, quote: Collection.quote(s, recycle).token }); assert.equal(item().unseen, false, 'using a duplicate is not a new find');
});

test('older same-guild import keeps its acquisition cursor so a lower-sequence new find is still unseen', () => {
  const s = mature(); act(s, { type: 'collection-unlock', kind: 'cards' }); s.collection.ink = 1000;
  const id = 'card:trail-courier', item = state => view(state).attention.items.find(r => r.id === id); act(s, item(s).inspectAction); const old = clone(s);
  const craft = { type: 'card-craft', id: 'trail-courier' };
  for (let i = 0; i < 3; i += 1) { act(s, { ...craft, quote: Collection.quote(s, craft).token }); act(s, item(s).inspectAction); }
  const futureSeen = s.onboarding.attention.finds[id].seen;
  const store = Storage.createStore({ storage: memory(), now: () => old.lastUpdate }); store.load();
  const result = store.replaceImport(store.export(old).text, { preservePracticeFrom: s }); assert.ok(result.ok, result.message); const restored = result.state; valid(restored);
  assert.equal(restored.onboarding.attention.findSequence, old.collection.sequence); assert.equal(item(restored).unseen, false);
  act(restored, { ...craft, quote: Collection.quote(restored, craft).token }); assert.ok(restored.collection.sequence < futureSeen); assert.equal(item(restored).unseen, true);
  const same = clone(restored); Core.mergePracticeReceipts(same, restored); assert.equal(item(same).unseen, true, 'merging the same unread find cannot mark it read');
});
