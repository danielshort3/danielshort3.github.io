'use strict';
const {createReleasedState}=require('./helpers/wayfarers-released.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const H = require('./helpers/wayfarers-progression.cjs');
const { Core, N, clone, mature, advance } = H;
const Collection = require('../../js/games/wayfarers-guild/collections.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const valid = s => assert.deepEqual(Core.validateState(s), { valid: true, errors: [] });
const act = (s, a) => { const r = Core.act(s, a); assert.ok(r.ok, JSON.stringify(a) + ': ' + r.message); valid(s); return r; };
const current = s => Core.getView(s).onboarding.active;
function explain(s) { while (current(s)?.mode === 'currency') act(s, current(s).ackAction); }
function step(s) { explain(s); const a = current(s); assert.ok(a); return act(s, a.practiceAction || a.inspectAction); }
function finish(s, id, intendedAction) {
  act(s, { type: 'onboarding-visit', id, ...(intendedAction ? { intendedAction } : {}) });
  let result;
  for (let i = 0; current(s) && i < 10; i += 1) result = step(s);
  assert.equal(current(s), null); return result;
}
function collected() { const s = mature(); act(s, { type: 'collection-unlock', kind: 'cards' }); act(s, { type: 'collection-unlock', kind: 'equipment' }); return s; }
function memory() { const map = new Map(); return { getItem: key => map.get(key) || null, setItem: (key, v) => map.set(key, String(v)), removeItem: key => map.delete(key) }; }

test('fresh Trail requires actual information opening and a real supplied rank, never Next', () => {
  const s = createReleasedState(1), before = clone(s);
  Core.getView(s); assert.deepEqual(s, before);
  act(s, { type: 'onboarding-visit', id: 'greenway' });
  assert.equal(current(s).mode, 'currency'); explain(s); assert.equal(current(s).mode, 'inspect');
  const unchanged = clone(s); assert.equal(Core.act(s, { type: 'onboarding-next', id: 'greenway', stepId: 'inspect' }).ok, false); assert.deepEqual(s, unchanged);
  step(s); assert.equal(current(s).requiredAction.type, 'expedition-buy');
  step(s); assert.equal(s.expedition.areas.greenway.ranks.boots, 1); assert.deepEqual(s.resources.coins, N.zero());
  const last = step(s); assert.deepEqual(last.reward.coins, N.from(12));
  assert.deepEqual(s.premium, before.premium); assert.deepEqual(s.luck, before.luck);
  assert.deepEqual(s.onboarding.practice.supplies, ['greenway:upgrade']);
});

test('first Quarry visit exposes Processing inspection before any alternate plan exists', () => {
  const s = createReleasedState(12); finish(s, 'greenway');
  let seconds = 0;
  while (!s.expedition.areas.quarry && seconds < 3600) {
    for (const tier of Core.getView(s).upgradeTiers.ready.filter(row => row.id.startsWith('area:greenway:'))) act(s, tier.unlockAction);
    const card = Core.getView(s).expedition.cards.find(row => row.rank < ({ boots: 4, porters: 3, scouts: 1 }[row.trackId] || 0) && !row.disabled);
    if (card) act(s, card.action);
    advance(s, 10); seconds += 10;
  }
  assert.ok(s.expedition.areas.quarry, 'Quarry reached through ordinary opening upgrades and production');
  act(s, { type: 'expedition-next' }); act(s, { type: 'expedition-select', areaId: 'quarry' });
  act(s, { type: 'onboarding-visit', id: 'quarry' }); step(s); step(s);
  assert.equal(s.expedition.areas.quarry.ranks.picks, 1);
  assert.equal(current(s).stepId, 'operate'); assert.equal(current(s).mode, 'inspect');
  assert.equal(current(s).target, 'area-plans'); assert.equal(current(s).heading, 'Open Processing');
  assert.equal(current(s).canLeave, false);
  assert.deepEqual(Core.getView(s).expedition.choices.filter(row => row.visible !== false).map(row => row.id), ['balanced']);
  const saved = clone(s), resumed = Core.normalizeState(saved, s.lastUpdate);
  assert.deepEqual(resumed.onboarding.practice.proofs, s.onboarding.practice.proofs);
  assert.deepEqual(resumed.onboarding.practice.supplies, ['greenway:upgrade', 'quarry:upgrade']);
  const before = clone(resumed.resources), result = step(resumed);
  assert.equal(resumed.onboarding.practice.progress.quarry, 3);
  assert.deepEqual(resumed.resources.coins, N.add(before.coins, result.reward.coins));
  const settled = clone(resumed);
  assert.equal(Core.act(resumed, current(s).inspectAction).ok, false); assert.deepEqual(resumed, settled);
  const planGuide = Core.getView(resumed).onboarding.guides.find(guide => guide.id === 'plans');
  assert.equal(planGuide.available, false);
  assert.ok(!Core.getView(resumed).onboarding.triggers.some(trigger => trigger.id === 'plans'));
  assert.equal(Core.act(resumed, { type: 'onboarding-visit', id: 'plans' }).ok, false);
});

test('every area teaches its first operation at its minimum earned skill state', () => {
  const s = createReleasedState(13), visited = [];
  function visitNewAreas() {
    for (const id of Object.keys(s.expedition.areas)) {
      if (visited.includes(id)) continue;
      act(s, { type: 'expedition-select', areaId: id });
      assert.equal(Object.values(s.expedition.areas[id].ranks).reduce((a, b) => a + b, 0), 0, id + ' is inspected before ordinary investment');
      act(s, { type: 'onboarding-visit', id }); step(s); step(s);
      assert.equal(current(s).mode, 'inspect');
      assert.equal(current(s).target, id === 'greenway' ? 'area-goal' : 'area-plans');
      step(s); assert.equal(s.onboarding.practice.progress[id], 3); visited.push(id);
    }
  }
  H.fund(s); visitNewAreas();
  for (let index = 0; index < 3; index += 1) {
    let attempts = 0;
    while ((!s.expedition.completed || index < 2 && !H.P.canAdvance(s)) && attempts++ < 100) {
      H.claimTiers(s);
      for (const [areaId, area] of Object.entries(s.expedition.areas)) for (const id of area.learned.slice()) if (area.ranks[id] < 20) { H.claimTiers(s); act(s, { type: 'expedition-buy', areaId, id }); }
      advance(s, 60); visitNewAreas();
    }
    assert.ok(s.expedition.completed, 'First three landmarks completed');
    if (index < 2) act(s, { type: 'expedition-next' });
  }
  for (const definition of H.P.Content.PROJECTS) {
    H.fund(s); H.claimTiers(s); act(s, { type: 'expedition-development', id: definition.id });
    H.P.tick(s, definition.work / H.P.rawRates(s).researchRate + 1e-6); H.claimTiers(s); visitNewAreas();
    if (visited.length === 6) break;
  }
  assert.deepEqual(visited, ['greenway', 'quarry', 'watchtower', 'workshop', 'ruins', 'harbor']);
  valid(s);
});

test('three foundation descriptors keep readiness separate from inspection acknowledgement', () => {
  const s = createReleasedState(14), initial = clone(s);
  const foundations = () => Core.getView(s).upgradeTiers.foundations.filter(row => row.areaId === 'greenway');
  assert.deepEqual(foundations().map(row => row.status), ['learned', 'locked', 'locked']);
  assert.ok(foundations().slice(1).every(row => row.requirements.length > 0 && !row.unlockAction));
  assert.deepEqual(s, initial, 'Foundation descriptors are read-only');
  H.fund(s);
  for (let count = 0; count < 20 && foundations()[1].status !== 'ready'; count += 1) {
    act(s, { type: 'expedition-buy', areaId: 'greenway', id: 'boots' }); advance(s, 30);
  }
  const ready = foundations()[1]; assert.equal(ready.status, 'ready');
  const attention = Core.getView(s).onboarding.attention.items.find(row => row.id === 'discovery:ready:' + ready.id);
  assert.ok(attention?.inspectAction); act(s, attention.inspectAction);
  act(s, { type: 'upgrade-tier-defer', ids: [ready.id] });
  assert.equal(foundations()[1].status, 'ready', 'Looking at or deferring an unlock does not remove its ready state');
  assert.equal(s.expedition.areas.greenway.ranks.porters, 0);
  act(s, ready.unlockAction); assert.equal(foundations()[1].status, 'learned');
  assert.equal(s.expedition.areas.greenway.ranks.porters, 0, 'Claiming does not grant a paid rank');
  valid(s);
});

test('technique help supplies one real rank and the first mode lesson requires its actual selection', () => {
  const Skills = require('../../js/games/wayfarers-guild/area-skills.js');
  const s = mature();
  for (let n = 0; !Skills.eligible(s, 'express-routes') && n < 100; n += 1) advance(s, 60);
  assert.ok(Skills.eligible(s, 'express-routes'));
  act(s, { type: 'area-skill-unlock', id: 'express-routes' });
  act(s, { type: 'expedition-batch', count: 100 });
  s.resources.coins = N.from(100);
  assert.ok(!Core.getView(s).onboarding.triggers.some(trigger => trigger.actionTypes.includes('area-skill-buy') || trigger.actionTypes.includes('area-skill-unlock')), 'Familiar unlock and purchase controls do not repeat mandatory lessons');
  const wallet = clone(s.resources), guide = Core.getView(s).onboarding.guides.find(row => row.id === 'techniques');
  assert.ok(guide.available); act(s, guide.helpOpenAction);
  finish(s, 'techniques');
  assert.equal(s.areaSkills.ranks['express-routes'], 1); assert.equal(s.expedition.batch, 100);
  assert.deepEqual(s.resources.coins, N.add(wallet.coins, 12));
  for (const resource of Object.keys(wallet).filter(id => id !== 'coins')) assert.deepEqual(s.resources[resource], wallet[resource]);
  assert.equal(s.onboarding.practice.supplies.filter(id => id === 'techniques:practice').length, 1);
  const beforeNoOp = clone(s);
  assert.equal(Core.act(s, { type: 'onboarding-visit', id: 'technique-config', intendedAction: { type: 'area-skill-config', id: 'express-routes', value: 'off' } }).ok, false);
  assert.deepEqual(s, beforeNoOp, 'An already-selected mode cannot start a mandatory no-op lesson');
  const option = Skills.view(s).items.find(row => row.skillId === 'express-routes').options.find(row => row.value === 'express');
  act(s, { type: 'onboarding-visit', id: 'technique-config', intendedAction: option.action });
  assert.equal(current(s).canLeave, false); assert.equal(current(s).targetData.catalogId, 'skill:express-routes');
  step(s); assert.equal(current(s).mode, 'action');
  assert.equal(Core.act(s, { type: 'onboarding-next', id: 'technique-config', stepId: 'practice' }).ok, false);
  assert.equal(s.areaSkills.configs['express-routes'], 'off'); step(s);
  assert.equal(s.areaSkills.configs['express-routes'], 'express');
  assert.equal(s.onboarding.practice.progress['technique-config'], 2);
  const saved = clone(s.onboarding.practice); act(s, { type: 'refit' });
  assert.deepEqual(s.onboarding.practice.proofs, saved.proofs);
  assert.deepEqual(s.onboarding.practice.supplies, saved.supplies);
  valid(s);
});

test('released E2 zero-rank practice quotes one real rank with unchanged canonical costs and effects', () => {
  const exported = require('./fixtures/wayfarers-v5-retained.json');
  const s = Core.normalizeState(clone(exported.state || exported));
  // Diagnostic purchase-state variant of the released fixture, not player data.
  s.expedition.areas.greenway.ranks.boots = 0;
  valid(s); act(s, { type: 'expedition-select', areaId: 'greenway' });
  act(s, { type: 'onboarding-visit', id: 'greenway' }); explain(s);
  const before = clone(s), view = Core.getView(s), preview = view.onboarding.active.practicePreview;
  const canonical = view.globalUpgrades.find(row => row.id === 'area:greenway:boots');
  assert.equal(preview.quantity, 1); assert.equal(preview.rank, 0); assert.equal(preview.rankAfter, 1);
  assert.deepEqual(preview.cost, canonical.cost); assert.deepEqual(preview.impact, canonical.impact);
  assert.deepEqual(preview.action, canonical.action); assert.deepEqual(s, before, 'view is pure');
  step(s); assert.equal(current(s).practicePreview.quantity, 1);
  const wallet = clone(s.resources); step(s);
  assert.equal(s.expedition.areas.greenway.ranks.boots, 1); assert.deepEqual(s.resources, wallet);
  assert.equal(s.onboarding.practice.supplies.filter(id => id === 'greenway:upgrade').length, 1);
});

test('retained Quarry Processing reports canonical actual rates and queues without changing its economy', () => {
  const exported = require('./fixtures/wayfarers-v5-retained.json');
  const s = Core.normalizeState(clone(exported.state || exported));
  assert.equal(s.expedition.version, 2);
  act(s, { type: 'expedition-select', areaId: 'quarry' });
  const before = clone(s), view = Core.getView(s).expedition;
  assert.equal(view.operation.title, 'Processing');
  assert.deepEqual(view.operation.stages.map(row => [row.actualRate, row.maxRate, row.buffer, row.capacity, row.status]), view.stations.map(row => [row.rate, row.maxRate, row.buffer, row.capacity, row.status]));
  assert.ok(view.operation.stages.some(row => row.buffer > 0), 'Released fixture contains a real queued buffer');
  assert.deepEqual(s, before, 'Inspection never repairs or advances retained production');
});

test('stale tokens, substitutions, replay and duplicate transactions cannot spend or grant', () => {
  const s = createReleasedState(2); act(s, { type: 'onboarding-visit', id: 'greenway' }); step(s);
  const request = clone(current(s).practiceAction), wrong = clone(request); wrong.action.id = 'porters';
  const before = clone(s); assert.equal(Core.act(s, wrong).ok, false); assert.deepEqual(s, before);
  act(s, request); const after = clone(s); assert.equal(Core.act(s, request).ok, false); assert.deepEqual(s, after);
  step(s); assert.equal(Core.act(s, { type: 'onboarding-visit', id: 'greenway' }).ok, false);
});

test('mandatory lesson rejects leaving and reload resumes without granting supplies', () => {
  let s = createReleasedState(3); act(s, { type: 'onboarding-visit', id: 'greenway' }); step(s);
  assert.equal(current(s).canLeave, false); assert.equal(Core.act(s, {type:'onboarding-leave',id:'greenway'}).ok, false); assert.deepEqual(s.resources.coins, N.zero());
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
  assert.equal(current(s).helpOpenAction, undefined); assert.equal(Core.act(s, {type:'onboarding-leave',id:'reserves'}).ok, false);
  const optional = mature(); optional.resources.coins = N.from(100); Object.assign(s, optional);
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
  act(old, { type: 'onboarding-visit', id: 'cards' }); explain(old); assert.equal(current(old).stepId, 'fuse'); step(old); valid(old);
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
  const s = createReleasedState(11); act(s, { type: 'onboarding-visit', id: 'greenway' });
  const before = clone(s.onboarding.practice); advance(s, 7200);
  for (const key of ['active', 'bindings', 'proofs', 'supplies', 'rewards', 'helpRewards']) assert.deepEqual(s.onboarding.practice[key], before[key]); assert.ok(Object.values(s.onboarding.practice.progress).every(p => p === 0)); valid(s);
});

test('first-use Unlock and go both teaches the real tier claim and consumes its duplicate notice', () => {
  const s = createReleasedState(44); H.fund(s);
  for (let n = 0; n < 20 && !Core.getView(s).upgradeTiers.ready.some(row => row.id === 'area:greenway:porters'); n += 1) { act(s, { type: 'expedition-buy', id: 'boots' }); advance(s, 30); }
  const entry = Core.getView(s).onboarding.inbox.entries.find(r => r.id === 'ready:area:greenway:porters');
  assert.ok(entry);
  act(s, { type: 'onboarding-visit', id: 'tiers', intendedAction: entry.openAction }); step(s);
  const result = step(s); assert.equal(result.destination.upgradeId, 'porters');
  assert.ok(s.upgradeTiers.claimed.includes('area:greenway:porters'));
  assert.ok(s.onboarding.read.includes('tier:area:greenway:porters')); assert.ok(!(Core.getView(s).onboarding.notice?.items || []).some(item => /^(ready|tier):area:greenway:porters$/.test(item.id)), 'This tier has no duplicate notice; unrelated discoveries remain available');
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
