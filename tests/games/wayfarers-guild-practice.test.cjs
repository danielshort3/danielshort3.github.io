'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const H = require('./helpers/wayfarers-progression.cjs');
const { Core, N, clone, mature, advance } = H;
const Collection = require('../../js/games/wayfarers-guild/collections.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const valid = s => assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
const act = (s, a) => { const r = Core.act(s, a); assert.ok(r.ok, JSON.stringify(a) + ': ' + r.message); valid(s); return r; };
const current = s => Core.getView(s).onboarding.active;
function step(s) { const a = current(s); assert.ok(a); return act(s, a.practiceAction || a.inspectAction); }
function finish(s, id, intendedAction) {
  act(s, { type: 'onboarding-visit', id, ...(intendedAction ? { intendedAction } : {}) });
  let result;
  for (let i = 0; current(s) && i < 10; i += 1) result = step(s);
  assert.equal(current(s), null); return result;
}
function collected() { const s = mature(); act(s, { type: 'collection-unlock', kind: 'cards' }); act(s, { type: 'collection-unlock', kind: 'equipment' }); return s; }
function memory() { const map = new Map(); return { getItem: key => map.get(key) || null, setItem: (key, v) => map.set(key, String(v)), removeItem: key => map.delete(key) }; }

test('fresh Trail requires actual information opening and a real supplied rank, never Next', () => {
  const s = Core.createState(1), before = clone(s);
  Core.getView(s); assert.deepEqual(s, before);
  act(s, { type: 'onboarding-visit', id: 'greenway' });
  assert.equal(current(s).mode, 'inspect');
  const unchanged = clone(s); assert.equal(Core.act(s, { type: 'onboarding-next', id: 'greenway', stepId: 'inspect' }).ok, false); assert.deepEqual(s, unchanged);
  step(s); assert.equal(current(s).requiredAction.type, 'expedition-buy');
  step(s); assert.equal(s.expedition.areas.greenway.ranks.boots, 1); assert.deepEqual(s.resources.coins, N.zero());
  const last = step(s); assert.deepEqual(last.reward.coins, N.from(12));
  assert.deepEqual(s.premium, before.premium); assert.deepEqual(s.luck, before.luck);
  assert.deepEqual(s.onboarding.practice.supplies, ['greenway:upgrade']);
});

test('stale tokens, substitutions, replay and duplicate transactions cannot spend or grant', () => {
  const s = Core.createState(2); act(s, { type: 'onboarding-visit', id: 'greenway' }); step(s);
  const request = clone(current(s).practiceAction), wrong = clone(request); wrong.action.id = 'porters';
  const before = clone(s); assert.equal(Core.act(s, wrong).ok, false); assert.deepEqual(s, before);
  act(s, request); const after = clone(s); assert.equal(Core.act(s, request).ok, false); assert.deepEqual(s, after);
  step(s); assert.equal(Core.act(s, { type: 'onboarding-visit', id: 'greenway' }).ok, false);
});

test('leaving before a supplied action grants nothing and reload resumes exactly', () => {
  let s = Core.createState(3); act(s, { type: 'onboarding-visit', id: 'greenway' }); step(s);
  act(s, current(s).leaveAction); assert.deepEqual(s.resources.coins, N.zero());
  s = Core.normalizeState(clone(s), s.lastUpdate); act(s, { type: 'onboarding-visit', id: 'greenway' });
  assert.equal(current(s).stepId, 'upgrade'); step(s); step(s);
  assert.equal(s.expedition.areas.greenway.ranks.boots, 1);
});

test('card lesson makes actual saved decks and fusion while preserving existing duplicates', () => {
  const s = collected(), wallet = clone(s.resources), inventory = clone(s.collection.cards);
  const active = s.collection.activeDeck; finish(s, 'cards');
  assert.equal(s.collection.activeDeck, active);
  assert.equal(s.collection.cards['trail-courier'].rank, 2);
  assert.equal(s.collection.cards['trail-courier'].copies, inventory['trail-courier'].copies);
  assert.equal(s.collection.decks.filter(d => d.slots.includes('trail-courier')).length, 2);
  assert.deepEqual(s.resources, wallet); assert.ok(s.onboarding.practice.supplies.includes('cards:fuse'));
});

test('equipment lesson really equips and adds one guaranteed point without using owned scrolls', () => {
  const s = collected(), scrolls = clone(s.collection.scrolls), expected = clone(s); const scroll = { type: 'gear-scroll', id: 'trail-boots', scrollId: 'steady' }; scroll.quote = Collection.quote(expected, scroll).token; assert.ok(Core.act(expected, scroll).ok);
  finish(s, 'equipment'); assert.equal(s.collection.equipped.boots, 'trail-boots');
  assert.equal(Collection.points(s.collection.gear['trail-boots']), 1);
  assert.deepEqual(s.collection.scrolls, scrolls); assert.equal(s.collection.scrollRng, expected.collection.scrollRng);
});

test('full enhanced equipment is reviewed without reforge, slot clearing or more scrolls', () => {
  const s = collected(); s.collection.gear['trail-boots'].successes.steady = 6; s.collection.equipped.boots = 'trail-boots';
  const item = clone(s.collection.gear), scrolls = clone(s.collection.scrolls); finish(s, 'equipment');
  assert.deepEqual(s.collection.gear, item); assert.deepEqual(s.collection.scrolls, scrolls);
  assert.ok(!s.onboarding.practice.supplies.includes('equipment:scroll'));
});

test('optional Help opening is a separate once-only reward, never automatic interception', () => {
  const s = mature(); s.resources.coins = N.from(100); const before = N.from(s.resources.coins);
  act(s, { type: 'onboarding-visit', id: 'reserves', intendedAction: { type: 'plan-reserve', id: 'ore', amount: '33' } });
  assert.equal(current(s).helpOpenAction, undefined); act(s, current(s).leaveAction);
  act(s, { type: 'onboarding-visit', id: 'reserves' }); const help = current(s).helpOpenAction;
  act(s, help); assert.deepEqual(s.resources.coins, N.add(before, 4));
  const saved = clone(s); assert.equal(Core.act(s, help).ok, false); assert.deepEqual(s, saved);
  step(s); step(s); assert.deepEqual(s.resources.coins, N.add(before, 12));
});

test('first-use interception binds the reviewed action and observes canonical supply and relic fields', () => {
  const s = mature();
  finish(s, 'reserves', { type: 'plan-reserve', id: 'ore', amount: '37' }); assert.deepEqual(s.guild.plan.reserves.ore, N.from(37));
  while (s.lifetime.highestRoute < 3) { if(s.expedition.completed) act(s,{type:'expedition-next'}); advance(s,60); } assert.ok(s.rooms.includes('kitchen')); s.luck.owned.push('living-crucible'); Core.act(s, { type: 'relic-equip', id: 'living-crucible' });
  finish(s, 'supply', { type: 'supply-plan', id: 'push' }); assert.equal(s.guild.supply, 'push');
  // Owned eligibility remains durable even when the currently required relic is unequipped.
  Core.act(s, { type: 'relic-equip', id: null }); valid(s); Core.getView(s);
});

test('exact batch practice and guild purchases preserve preexisting wallet inputs', () => {
  const s = mature(); for (const id of Object.keys(s.resources)) s.resources[id] = N.zero();
  const rank = s.expedition.areas[s.expedition.selectedArea].ranks.beacon;
  finish(s, 'bulk', { type: 'expedition-batch', count: 5 });
  assert.equal(s.expedition.areas[s.expedition.selectedArea].ranks.beacon, rank + 5);
  assert.deepEqual(s.resources.coins, N.from(8));
  const before = clone(s.resources); finish(s, 'guild-upgrades', { type: 'buy', id: 'boots', count: 1 });
  assert.equal(s.upgrades.boots, 1); assert.deepEqual(s.resources.coins, N.add(before.coins, 8));
});

test('recruitment, card archive/crafting, equipment craft and restoration receive exact practice inputs', () => {
  const s = collected(); for (const id of Object.keys(s.resources)) s.resources[id] = N.zero();
  finish(s, 'companions'); assert.ok(s.crew.companions.includes('fox')); assert.deepEqual(s.resources.knowledge, N.zero());
  finish(s, 'card-archive'); assert.equal(s.collection.cards['trail-courier'].copies, 0); const ink = s.collection.ink;
  finish(s, 'card-craft'); assert.equal(s.collection.cards['trail-courier'].copies, 1); assert.equal(s.collection.ink, ink);
  finish(s, 'gear-craft'); assert.equal(Object.keys(s.collection.gear).length, 2); assert.deepEqual(s.resources.ore, N.zero());
  s.collection.gear['trail-boots'].failed = 1; const scrolls = clone(s.collection.scrolls); Core.act(s, { type: 'collection-ack', sequence: s.collection.sequence });
  finish(s, 'gear-repair'); assert.equal(s.collection.gear['trail-boots'].failed, 0); assert.deepEqual(s.collection.scrolls, scrolls);
});

test('same-guild import before collection unlock retains consumed receipts without copying inventory', () => {
  const old = mature(), currentState = clone(old); act(currentState, { type: 'collection-unlock', kind: 'cards' }); finish(currentState, 'cards');
  act(currentState, { type: 'collection-unlock', kind: 'equipment' }); finish(currentState, 'equipment');
  finish(currentState, 'reserves');
  const store = Storage.createStore({ storage: memory(), now: () => old.lastUpdate });
  store.load(); const exported = store.export(old); assert.ok(exported.ok);
  const imported = store.replaceImport(exported.text, { preservePracticeFrom: currentState }); assert.ok(imported.ok, imported.message);
  const s = imported.state; valid(s); assert.deepEqual(s.collection.cards, {}); assert.deepEqual(s.collection.gear, {});
  assert.ok(s.onboarding.practice.supplies.includes('cards:fuse')); assert.ok(s.onboarding.practice.supplies.includes('equipment:scroll'));
  act(s, { type: 'collection-unlock', kind: 'cards' }); assert.equal(Core.getView(s).onboarding.guides.find(g => g.id === 'cards').complete, true);
  assert.equal(Core.act(s, { type: 'onboarding-visit', id: 'cards' }).ok, false);
});

test('missing imported card binding reconciles safely while preserving already proved steps', () => {
  const old = collected(), s = clone(old); s.collection.cards['trail-courier'].rank = 2; s.collection.cards['quarry-mole'].rank = 2;
  s.collection.cards['tower-astronomer'] = { rank: 1, copies: 0 }; valid(s);
  act(s, { type: 'onboarding-visit', id: 'cards' }); step(s);
  assert.equal(s.onboarding.practice.bindings.cards.cardId, 'tower-astronomer');
  Core.mergePracticeReceipts(old, s); valid(old); assert.doesNotThrow(() => Core.getView(old));
  act(old, { type: 'onboarding-visit', id: 'cards' }); assert.equal(current(old).stepId, 'fuse'); step(old); valid(old);
});

test('malformed ledger and prototype references reject without throwing or mutating', () => {
  const original = collected(); act(original, { type: 'onboarding-visit', id: 'cards' });
  const changes = [s => s.onboarding.practice.progress.cards = 1, s => s.onboarding.practice.active = 'constructor', s => s.onboarding.practice.bindings.cards.cardId = '__proto__', s => s.onboarding.practice.supplies.push('cards:fuse'), s => s.onboarding.practice.intentions.reserves = { type: 'refit' }, s => s.onboarding.practice.helpRewards.push('greenway')];
  for (const change of changes) { const s = clone(original); change(s); const copy = clone(s); assert.equal(Core.validateState(s).valid, false); assert.deepEqual(s, copy); }
});

test('Refit retains proofs and consumes no second practice supply after ordinary ranks rebuild', () => {
  const s = mature(); finish(s, 'greenway'); const ledger = clone(s.onboarding.practice);
  assert.ok(Core.getRefitPreview(s).available); act(s, { type: 'refit' });
  assert.deepEqual(s.onboarding.practice.proofs, ledger.proofs); assert.deepEqual(s.onboarding.practice.supplies, ledger.supplies);
  assert.equal(Core.act(s, { type: 'onboarding-visit', id: 'greenway' }).ok, false); valid(s);
});

test('offline time never completes lessons, opens Help, or spends practice receipts', () => {
  const s = Core.createState(11); act(s, { type: 'onboarding-visit', id: 'greenway' });
  const before = clone(s.onboarding.practice); advance(s, 7200);
  for (const key of ['active', 'bindings', 'proofs', 'supplies', 'rewards', 'helpRewards']) assert.deepEqual(s.onboarding.practice[key], before[key]); assert.ok(Object.values(s.onboarding.practice.progress).every(p => p === 0)); valid(s);
});

test('first-use Unlock and go both teaches the real tier claim and consumes its duplicate notice', () => {
  const s = Core.createState(44); H.fund(s); act(s, { type: 'expedition-buy', id: 'boots' }); act(s, { type: 'expedition-buy', id: 'boots' });
  const entry = Core.getView(s).onboarding.inbox.entries.find(r => r.id === 'ready:area:greenway:porters');
  act(s, { type: 'onboarding-visit', id: 'tiers', intendedAction: entry.openAction }); step(s);
  const result = step(s); assert.equal(result.destination.upgradeId, 'porters');
  assert.ok(s.upgradeTiers.claimed.includes('area:greenway:porters'));
  assert.ok(s.onboarding.read.includes('tier:area:greenway:porters')); assert.equal(Core.getView(s).onboarding.notice, null);
});

test('a supplied permanent project starts with zero stock and preserves its funded work across Refit', () => {
  const s = mature({ untilProject: 'rail-network' }); for (const id of Object.keys(s.resources)) s.resources[id] = N.zero();
  finish(s, 'projects', { type: 'expedition-development', id: 'industrial-supports' });
  assert.equal(s.expedition.commission.id, 'industrial-supports'); assert.deepEqual(s.resources.ore, N.zero());
  assert.ok(Core.getRefitPreview(s).available); const commission = clone(s.expedition.commission); act(s, { type: 'refit' });
  assert.deepEqual(s.expedition.commission, commission); assert.ok(s.onboarding.practice.supplies.includes('projects:practice'));
});

test('Focus practice consumes its supplied charge without refilling or borrowing a regular charge', () => {
  const s = mature(), before = clone(s.expedition.focus); finish(s, 'focus');
  assert.equal(s.expedition.focus.charges, before.charges); assert.equal(s.expedition.focus.recharge, before.recharge);
  assert.ok(s.expedition.focus.remaining > 0); assert.ok(s.onboarding.practice.supplies.includes('focus:practice'));
});

test('no-op intended actions cannot fabricate successful practice proof or collect completion rewards', () => {
  const s = mature(); act(s, { type: 'onboarding-visit', id: 'reserves', intendedAction: { type: 'plan-reserve', id: 'ore', amount: '0' } }); step(s);
  const before = clone(s); assert.equal(Core.act(s, current(s).practiceAction).ok, false); assert.deepEqual(s, before);
  assert.doesNotThrow(() => Core.act(s, { type: 'onboarding-visit', id: 'constructor', intendedAction: { type: 'refit' } }));
});

test('an enormous practice price preserves a small existing wallet exactly', () => {
  const s = mature(); s.upgrades.boots = 500; s.resources.coins = N.from(100); valid(s);
  assert.ok(N.cmp(Core.upgradeCost(s, Core.Content.UPGRADES.find(d => d.id === 'boots')).coins, '1e20') > 0);
  finish(s, 'guild-upgrades', { type: 'buy', id: 'boots', count: 1 });
  assert.equal(s.upgrades.boots, 501); assert.deepEqual(s.resources.coins, N.from(108));
});

test('retained expedition2 uses its actual resource-keyed first-rank quote without spending saved coins', () => {
  const old = clone(require('./fixtures/wayfarers-v4-fresh.json')), s = Core.normalizeState(old, old.lastUpdate);
  assert.equal(s.expedition.version, 2); s.resources.coins = N.from(100);
  const before = clone(s.resources); act(s, { type: 'onboarding-visit', id: 'greenway' }); step(s); step(s);
  assert.equal(s.expedition.areas.greenway.ranks.boots, 1); assert.deepEqual(s.resources, before); valid(s);
});

test('configuration practice selects a different value from its enclosing saved slot', () => {
  const s = mature(), previous = clone(s.expedition.areas.watchtower.plans.assignments);
  finish(s, 'configuration'); assert.notDeepEqual(s.expedition.areas.watchtower.plans.assignments, previous);
});

test('all currently earned optional lessons execute their real canonical control or safe review', () => {
  const original = require('./helpers/wayfarers-practice.cjs').advanced();
  const guides = Core.getView(original).onboarding.guides.filter(g => g.optional && g.available);
  assert.ok(guides.length >= 25);
  for (const guide of guides) {
    const s = clone(original), before = clone(s.premium); finish(s, guide.id);
    assert.equal(s.onboarding.practice.progress[guide.id], guide.steps.length, guide.id);
    assert.deepEqual(s.premium, before, guide.id + ' cannot spend premium currency or alter its RNG');
  }
});

test('the reviewed nondefault project and working plan keep matching inspector target metadata', () => {
  const s = mature({ untilProject: 'rail-network' });
  act(s, { type: 'onboarding-visit', id: 'projects', intendedAction: { type: 'expedition-development', id: 'ruins-expedition' } });
  assert.equal(current(s).targetData.catalogId, 'development:ruins-expedition'); step(s);
  assert.equal(current(s).requiredAction.id, 'ruins-expedition'); step(s);
  const option = Core.getView(s).expedition.choices.find(row => row.id === 'trade');
  assert.ok(option); act(s, { type: 'onboarding-visit', id: 'plans', intendedAction: option.action });
  assert.equal(current(s).targetData.choiceId, option.id);
});

test('first deliberate Help opening still pays once after action mastery and survives old-backup import', () => {
  const s = mature(); s.resources.coins = N.from(100);
  finish(s, 'reserves', { type: 'plan-reserve', id: 'ore', amount: '23' });
  const beforeHelp = clone(s), guide = Core.getView(s).onboarding.guides.find(g => g.id === 'reserves');
  assert.ok(guide.complete); assert.equal(guide.helpReward.amount, 4); assert.ok(guide.helpOpenAction);
  act(s, guide.helpOpenAction); assert.deepEqual(s.resources.coins, N.from(112));
  const reloaded = Core.normalizeState(clone(s), s.lastUpdate), snapshot = clone(reloaded);
  assert.equal(Core.act(reloaded, guide.helpOpenAction).ok, false); assert.deepEqual(reloaded, snapshot);
  const store = Storage.createStore({ storage: memory(), now: () => s.lastUpdate }); store.load();
  const restored = store.replaceImport(store.export(beforeHelp).text, { preservePracticeFrom: reloaded });
  assert.ok(restored.ok, restored.message); valid(restored.state);
  assert.equal(Core.getView(restored.state).onboarding.guides.find(g => g.id === 'reserves').helpOpenAction, undefined);
  assert.equal(Core.act(restored.state, guide.helpOpenAction).ok, false);
  assert.deepEqual(restored.state.resources.coins, beforeHelp.resources.coins);
});

test('a first-rank lesson previews exactly one rank while the retained global selector stays at100', () => {
  const s = mature(); s.expedition.areas.quarry.ranks.picks = 0;
  act(s, { type: 'expedition-select', areaId: 'quarry' }); act(s, { type: 'expedition-batch', count: 100 });
  Core.setPremiumEntitlements(s, ['compass']);
  act(s, { type: 'onboarding-visit', id: 'quarry' });
  const original = clone(s), shown = current(s).practicePreview;
  assert.equal(shown.quantity, 1); assert.equal(shown.rankAfter, 1); assert.equal(shown.rank, 0);
  assert.deepEqual(s, original, 'preview cannot change live batch, wallet, ranks or proof');
  const expected = clone(s); expected.expedition.batch = 1; Core.setPremiumEntitlements(expected, ['compass']);
  const exact = Core.getView(expected, { skipOnboarding: true }).globalUpgrades.find(row => row.id === shown.id);
  assert.deepEqual(shown.cost, exact.cost); assert.deepEqual(shown.impact, exact.impact);
  assert.equal(Core.getView(s).globalUpgrades.find(row => row.id === shown.id).quantity, 100);
  step(s); assert.equal(current(s).practicePreview.quantity, 1); const wallet = clone(s.resources); step(s);
  assert.equal(s.expedition.areas.quarry.ranks.picks, 1); assert.equal(s.expedition.batch, 100); assert.deepEqual(s.resources, wallet);
});
