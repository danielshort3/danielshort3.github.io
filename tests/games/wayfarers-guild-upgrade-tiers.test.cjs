'use strict';
const {createReleasedState}=require('./helpers/wayfarers-released.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, P, N, clone, advance, fund, mature, claimTiers } = require('./helpers/wayfarers-progression.cjs');
const E = require('../../js/games/wayfarers-guild/expeditions.js');
const T = require('../../js/games/wayfarers-guild/upgrade-tiers.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const old = require('./fixtures/wayfarers-v4-state.json');
const valid = state => assert.deepEqual(Core.validateState(state), {valid:true,errors:[]});
const act = (state, action) => { const result = Core.act(state,action); assert(result.ok,result.message); return result; };
function ready() {
  const state = createReleasedState(0); advance(state,6);
  act(state,{type:'expedition-buy',id:'boots'}); fund(state);
  while (state.expedition.areas.greenway.ranks.boots < 4) act(state,{type:'expedition-buy',id:'boots'});
  for(let seconds=0;!state.upgradeTiers.pending.includes('area:greenway:porters') && seconds<600;seconds+=1) advance(state,1);
  assert(state.upgradeTiers.pending.includes('area:greenway:porters'),'Ranks and actual deliveries earn Porters');
  return state;
}
function discoverQuarry(state) {
  for(let seconds=0;!state.expedition.areas.quarry && seconds<3600;seconds+=10) {
    for(const row of T.view(state).ready.filter(row=>row.areaId==='greenway' && row.id.startsWith('area:'))) act(state,row.unlockAction);
    for(const [id,target] of Object.entries({boots:4,porters:3,scouts:1})) if(T.allows(state,{type:'expedition-buy',areaId:'greenway',id}) && state.expedition.areas.greenway.ranks[id]<target) act(state,{type:'expedition-buy',areaId:'greenway',id});
    advance(state,10);
  }
  assert(state.expedition.areas.quarry,'All three paid foundation skills establish the next area');
  act(state,{type:'expedition-next'});
}
const cardIds = state => Core.getView(state).expedition.cards.filter(card=>card.visible!==false).map(card=>card.trackId || card.id);

test('fresh games have one purchase and readiness alone does not expose or buy Porters', () => {
  const state = ready(), before = clone(state);
  assert.deepEqual(cardIds(state),['boots']);
  assert(T.view(state).ready.some(row=>row.id==='area:greenway:porters'));
  assert(!Core.getView(state).globalUpgrades.some(row=>row.trackId==='porters'));
  assert.equal(P.quote(state,'greenway','porters',1).valid,false);
  for(const purchase of [Core.act,P.act]) assert.equal(purchase(state,{type:'expedition-buy',areaId:'greenway',id:'porters'}).ok,false);
  assert.equal(Core.act(state,{type:'buy',id:'boots'}).ok,false,'Hidden global path cannot bypass opening');
  assert.deepEqual(state,before); valid(state);
});

test('claiming spends nothing, buys no rank, and duplicate or future claims are atomic', () => {
  const state = ready(), before = clone(state);
  act(state,{type:'upgrade-tier-unlock',id:'area:greenway:porters'});
  assert.deepEqual(state.resources,before.resources); assert.deepEqual(state.premium,before.premium); assert.deepEqual(state.luck,before.luck); assert.deepEqual(state.collection,before.collection);
  assert.deepEqual(state.expedition.areas.greenway.ranks,before.expedition.areas.greenway.ranks);
  assert.deepEqual(cardIds(state),['boots','porters']);
  const claimed = clone(state);
  for(const id of ['area:greenway:porters','area:harbor:stowage','constructor','__proto__',{},null]) {
    assert.equal(Core.act(state,{type:'upgrade-tier-unlock',id}).ok,false); assert.deepEqual(state,claimed);
  }
  act(state,{type:'expedition-buy',id:'porters'}); valid(state);
});

test('ready tiers persist outside the capped event log and Later survives save/reload', () => {
  const state = ready(); advance(state,7200);
  const notice = T.view(state).notice; assert(notice.items.length>1);
  assert(!T.view(state).ready.some(row=>row.id==='area:greenway:scouts'),'Later track waits for prior explicit claim');
  act(state,notice.deferAction); state.expedition.recent=[];
  const record = Storage.createStore({storage:null,now:()=>state.lastUpdate}).export(state); assert(record.ok);
  const restored = Core.normalizeState(JSON.parse(record.text).state,state.lastUpdate);
  assert.deepEqual(restored.upgradeTiers,state.upgradeTiers); assert.equal(T.view(restored).notice,null); assert(T.view(restored).ready.length>0);
  advance(restored,7200); assert.equal(T.view(restored).notice,null,'No repeated notice from elapsed time');
  act(restored,{type:'upgrade-tier-unlock',id:'area:greenway:porters'});
  for(let rank=0;rank<3;rank+=1) act(restored,{type:'expedition-buy',id:'porters'});
  assert(T.view(restored).ready.some(row=>row.id==='area:greenway:scouts')); valid(restored);
});

test('views are pure and offline partitions produce identical tier entitlement and backlog state', () => {
  const batch=ready(), ticks=clone(batch); advance(batch,7200);
  for(let i=0;i<120;i++) advance(ticks,60);
  assert.deepEqual(batch.upgradeTiers,ticks.upgradeTiers);
  const before=clone(batch); Core.getView(batch); T.view(batch); assert.deepEqual(batch,before); valid(batch);
});

test('new areas stage their tracks while a fresh local baseline remains immediately usable', () => {
  const state=ready(); discoverQuarry(state);
  assert.deepEqual(cardIds(state),['picks']);
  fund(state); for(let rank=0;rank<4;rank+=1) act(state,{type:'expedition-buy',areaId:'quarry',id:'picks'});
  for(let seconds=0;!T.view(state).ready.some(row=>row.id==='area:quarry:carts') && seconds<600;seconds+=1) advance(state,1);
  assert(T.view(state).ready.some(row=>row.id==='area:quarry:carts'));
  assert.equal(Core.act(state,{type:'expedition-buy',areaId:'quarry',id:'carts'}).ok,false);
  act(state,{type:'upgrade-tier-unlock',id:'area:quarry:carts'}); assert.deepEqual(cardIds(state),['picks','carts']); valid(state);
});

test('global project tiers enforce direct, catalog and planned purchase gates without hiding funded research', () => {
  const state=ready(); discoverQuarry(state); fund(state);
  const id='development:p:wheelworks'; assert(T.view(state).ready.some(row=>row.id===id));
  assert(!Core.getView(state).globalUpgrades.some(row=>row.id==='development:wheelworks'));
  const before=clone(state); assert.equal(Core.act(state,{type:'expedition-development',id:'wheelworks'}).ok,false); assert.equal(P.act(state,{type:'expedition-development',id:'wheelworks'}).ok,false); assert.deepEqual(state,before);
  act(state,{type:'upgrade-tier-unlock',id});
  assert(Core.getView(state).globalUpgrades.some(row=>row.id==='development:wheelworks'));
  act(state,{type:'expedition-development',id:'wheelworks'});
  const running=Core.getView(state).globalUpgrades.find(row=>row.id==='development:wheelworks'); assert(running && running.visible);
  assert(!Core.getView(state).globalUpgrades.some(row=>row.id==='development:harbor-foundation'));
  valid(state);
});

test('released retained E2 migrates silently with old available purchases, choices and economy intact', () => {
  const original=Core.migrateState(clone(old)), before=clone(original);
  const migrated=Core.normalizeState(original,original.lastUpdate);
  assert.equal(migrated.expedition.version,2); assert.deepEqual(migrated.expedition,before.expedition);
  assert.deepEqual(Core.getRates(migrated),Core.getRates(before)); assert.deepEqual(migrated.resources,before.resources);
  assert.equal(T.view(migrated).notice,null);
  for(const row of E.catalog(before).filter(row=>row.visible && row.group==='area')) assert(T.allows(migrated,row.action),row.id);
  valid(migrated);
});

test('retained E2 future area tiers also reject direct and automatic bypasses', () => {
  const historical=Core.migrateState(clone(require('./fixtures/wayfarers-v4-fresh.json')));
  const state=Core.normalizeState(historical,historical.lastUpdate);
  advance(state,7200); act(state,{type:'expedition-next'}); fund(state);
  assert(T.allows(state,{type:'expedition-buy',areaId:'quarry',id:'picks'}));
  assert(!T.allows(state,{type:'expedition-buy',areaId:'quarry',id:'carts'}));
  assert.equal(E.act(state,{type:'expedition-buy',areaId:'quarry',id:'carts'}).ok,false);
  advance(state,400); assert(T.view(state).ready.some(row=>row.id==='area:quarry:carts')); valid(state);
});

test('claimed tiers and deferred notices survive an actual Refit without reannouncement or a free rank', () => {
  const state=mature(), claimed=state.upgradeTiers.claimed.slice();
  act(state,{type:'refit'});
  assert.deepEqual(state.upgradeTiers.claimed,claimed);
  assert.equal(T.view(state).notice,null); assert(Object.values(state.expedition.areas.greenway.ranks).every(rank=>rank===0));
  assert(P.quote(state,'quarry','carts',1).valid); valid(state);
});

test('legacy adoption preserves the learned local purchase tiers despite renamed Tower tracks', () => {
  const state=Core.normalizeState(Core.migrateState(clone(old)),old.lastUpdate); fund(state);
  act(state,{type:'buy',id:'gear-tools'}); act(state,{type:'buy',id:'gear-boots'});
  assert(Core.getRefitPreview(state).available);
  act(state,{type:'refit'});
  assert.equal(state.expedition.version,4);
  assert(T.allows(state,{type:'expedition-buy',areaId:'watchtower',id:'signals'}));
  assert(!T.view(state).ready.some(row=>row.id.startsWith('area:')),'Old local knowledge is not announced again'); valid(state);
});

test('automation and saved plans cannot buy an unclaimed guild group', () => {
  const state=mature();
  state.upgradeTiers.claimed=state.upgradeTiers.claimed.filter(id=>id!=='guild:mine:0'); state.upgradeTiers.pending.push('guild:mine:0');
  state.automations.operations=true;
  act(state,{type:'plan-goal',action:{type:'buy',id:'miners'}});
  const rank=state.upgrades.miners; advance(state,120);
  assert.equal(state.upgrades.miners,rank);
  assert.match(Core.getView(state).planning.blockedReason,/upgrade tier/);
  assert(!Core.getView(state).globalUpgrades.some(row=>row.action.type==='buy' && row.action.id==='miners'));
  act(state,{type:'upgrade-tier-unlock',id:'guild:mine:0'}); advance(state,120);
  assert(state.upgrades.miners>rank); valid(state);
});

test('new pending tiers and their acknowledgements survive an actual Charter', () => {
  const state=mature(), id='guild:mine:0';
  state.upgradeTiers.claimed=state.upgradeTiers.claimed.filter(value=>value!==id); state.upgradeTiers.pending.push(id);
  for (const key of ['entries','announced','read']) state.onboarding[key]=state.onboarding[key].filter(value=>value!=='tier:'+id);
  act(state,{type:'upgrade-tier-defer',ids:[id]});
  while(!Core.getCharterPreview(state).available) {
    if (state.expedition.completed) act(state,{type:'expedition-next'});
    advance(state,600);
  }
  assert(Core.getCharterPreview(state).available);
  if (T.view(state).notice) act(state,T.view(state).notice.deferAction);
  const before=clone(state.upgradeTiers); act(state,{type:'charter'});
  assert.deepEqual(state.upgradeTiers.claimed,before.claimed); assert.deepEqual(state.upgradeTiers.prompted,before.prompted);
  assert(before.pending.every(id=>state.upgradeTiers.pending.includes(id)));
  assert.deepEqual(T.view(state).notice.items.map(row=>row.id),['planning:crests:2'],'Only the newly earned Charter capability may announce'); valid(state);
});

test('strict optional metadata rejects malformed arrays, future tiers, duplicate claims and bogus notices', () => {
  const state=ready();
  const changes=[s=>s.upgradeTiers=null,s=>s.upgradeTiers.claimed.push('area:harbor:stowage'),s=>s.upgradeTiers.pending.push('area:harbor:stowage'),s=>s.upgradeTiers.prompted.push('not-ready'),s=>s.upgradeTiers.pending.push('constructor'),s=>s.upgradeTiers.claimed.push('area:greenway:boots'),s=>s.upgradeTiers.pending.push('area:greenway:boots'),s=>s.upgradeTiers.pending={},s=>s.upgradeTiers.extra=1];
  for(const change of changes) {const bad=clone(state);change(bad);assert.equal(Core.validateState(bad).valid,false);const text=JSON.stringify({format:Storage.FORMAT,version:Core.VERSION,savedAt:bad.lastUpdate,state:bad});assert.equal(Storage.createStore({storage:null,now:()=>bad.lastUpdate}).inspectImport(text).ok,false);}
  const historical=clone(state);delete historical.upgradeTiers;delete historical.onboarding;valid(historical);const restored=Core.normalizeState(historical,historical.lastUpdate);assert(T.allows(restored,{type:'expedition-buy',areaId:'greenway',id:'porters'}),'Previously learned knowledge grandfathered');valid(restored);
});
