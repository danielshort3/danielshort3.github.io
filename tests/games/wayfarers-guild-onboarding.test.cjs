'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Core, N, P, clone, advance, fund, mature, claimTiers } = require('./helpers/wayfarers-progression.cjs');
const O = require('../../js/games/wayfarers-guild/onboarding.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const valid = state => assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
const act = (state, action) => { const result = Core.act(state, action); assert.ok(result.ok, result.message); return result; };
function finish(state, id) {
  act(state, { type: 'onboarding-visit', id });
  let result;
  for(let n=0;n<16;n+=1) { const active=Core.getView(state).onboarding.active; if(!active) break; result=act(state,active.ackAction||active.practiceAction||active.inspectAction); }
  return result;
}
function withoutGuidance(state) { const copy = clone(state); delete copy.onboarding; return copy; }
function porters() {
  const state = Core.createState(0); fund(state);
  for (let rank = 0; rank < 4; rank += 1) act(state, { type: 'expedition-buy', id: 'boots' });
  for (let seconds = 0; !state.upgradeTiers.pending.includes('area:greenway:porters') && seconds < 600; seconds += 1) advance(state, 1);
  assert.ok(state.upgradeTiers.pending.includes('area:greenway:porters'), 'Real Pathfinding ranks and three deliveries earn Porters');
  return state;
}

test('Unlock and go claims the real tier and acknowledges its resulting notice atomically', () => {
  const state=porters(), entry=Core.getView(state).onboarding.notice.items.find(row=>row.id==='ready:area:greenway:porters');
  assert.equal(entry.id,'ready:area:greenway:porters'); assert.equal(entry.openLabel,'Unlock & go');
  const before=clone(state), result=act(state,entry.openAction);
  assert.deepEqual(result.destination,{type:'ui',screen:'expedition',areaId:'greenway',upgradeId:'porters'});
  assert.ok(state.upgradeTiers.claimed.includes('area:greenway:porters'));
  assert.ok(state.onboarding.read.includes('ready:area:greenway:porters'));
  assert.ok(state.onboarding.read.includes('tier:area:greenway:porters'));
  assert.deepEqual(state.resources,before.resources); assert.deepEqual(state.expedition.areas.greenway.ranks,before.expedition.areas.greenway.ranks);
  assert.ok(!Core.getView(state).onboarding.notice?.items.some(row=>row.id.endsWith('area:greenway:porters')));
  const completed=clone(state); act(state,entry.openAction); assert.deepEqual(state,completed,'reopen does not claim or pay twice'); valid(state);
});

test('claiming outside a notice creates one truthful unlocked row and supersedes the ready row', () => {
  const state=porters(); act(state,{type:'upgrade-tier-unlock',id:'area:greenway:porters'});
  const notice=Core.getView(state).onboarding.notice;
  const relevant = notice.items.filter(row=>row.id.endsWith('area:greenway:porters'));
  assert.deepEqual(relevant.map(row=>row.id),['tier:area:greenway:porters']);
  assert.match(relevant[0].label,/unlocked$/);
  assert.ok(state.onboarding.read.includes('ready:area:greenway:porters'));
  act(state,notice.deferAction);
  assert.equal(Core.getView(state).onboarding.notice,null);
  assert.equal(Core.getView(state).onboarding.inbox.entries.filter(row=>row.id==='tier:area:greenway:porters'&&!row.read).length,1,'announced and read are different');
  const loaded=Core.normalizeState(clone(state),state.lastUpdate);
  assert.equal(Core.getView(loaded).onboarding.notice,null); assert.equal(Core.getView(loaded).onboarding.inbox.entries.filter(row=>row.id==='tier:area:greenway:porters'&&!row.read).length,1); valid(loaded);
});

test('Later consumes only automatic prompting, preserves a ready tier and survives capped logs', () => {
  const state=porters(), notice=Core.getView(state).onboarding.notice;
  act(state,notice.deferAction);
  assert.ok(state.upgradeTiers.prompted.includes('area:greenway:porters'));
  assert.ok(state.upgradeTiers.pending.includes('area:greenway:porters'));
  state.expedition.recent=[]; state.collection.recent=[]; advance(state,7200);
  const loaded=Core.normalizeState(clone(state),state.lastUpdate);
  assert.ok(Core.getView(loaded).onboarding.inbox.entries.some(row=>row.id==='ready:area:greenway:porters'&&!row.read));
  assert.ok(!Core.getView(loaded).onboarding.notice?.items.some(row=>row.id==='ready:area:greenway:porters')); valid(loaded);
});

test('all six earned areas teach actual operations and only the visited area becomes active', () => {
  const state=mature(), rewards={greenway:12,quarry:60,watchtower:120,workshop:200,ruins:360,harbor:600};
  for (const [id, amount] of Object.entries(rewards)) {
    act(state,{type:'expedition-select',areaId:id});
    assert.equal(Core.getView(state).onboarding.active,null);
    const before=N.from(state.resources.coins), result=finish(state,id);
    assert.deepEqual(result.reward.coins,N.from(amount)); assert.deepEqual(state.resources.coins,N.add(before,amount));
    assert.equal(state.onboarding.progress[id],3); valid(state);
  }
  const harbor=Core.getView(state).onboarding.guides.find(guide=>guide.id==='harbor');
  assert.ok(harbor.steps.every(step=>step.mode==='inspect'||step.requiredAction));
  assert.equal(harbor.rewardPreview.available,false);
});

test('returning released saves retain every economic field and receive guides without historical notices or rewards', () => {
  const prior=clone(require('./fixtures/wayfarers-v4-state.json'));
  const migrated=Core.migrateState(prior), retained=Core.normalizeState(migrated,migrated.lastUpdate);
  for (const [key,value] of Object.entries(migrated)) assert.deepEqual(retained[key],value,key+' retained exactly');
  assert.deepEqual(withoutGuidance(Core.normalizeState(withoutGuidance(retained),retained.lastUpdate)),withoutGuidance(retained));
  assert.deepEqual(retained.resources,prior.resources); assert.deepEqual(retained.expedition,migrated.expedition);
  assert.equal(Core.getView(retained).onboarding.notice,null);
  const id=retained.expedition.selectedArea, before=clone(retained.resources);
  assert.equal(retained.onboarding.progress[id],0); finish(retained,id);
  assert.deepEqual(retained.resources,before,'no retroactive payout for historical discoveries'); valid(retained);
});

test('Refit and Charter retain guide completion, in-progress steps and once-only reward claims', () => {
  const state=mature(); act(state,{type:'expedition-select',areaId:'greenway'}); finish(state,'greenway');
  act(state,{type:'expedition-select',areaId:'quarry'}); act(state,{type:'onboarding-visit',id:'quarry'}); while(Core.getView(state).onboarding.active.mode==='currency') act(state,Core.getView(state).onboarding.active.ackAction); act(state,Core.getView(state).onboarding.active.inspectAction); assert.equal(Core.act(state,{type:'onboarding-leave',id:'quarry'}).ok,false);
  const learned=clone(state.onboarding.progress), claimed=clone(state.onboarding.rewardClaims);
  assert.ok(Core.getRefitPreview(state).available); act(state,{type:'refit'});
  assert.deepEqual(state.onboarding.progress,learned); assert.deepEqual(state.onboarding.rewardClaims,claimed); valid(state);
  const charter=mature(); act(charter,{type:'expedition-select',areaId:'greenway'}); finish(charter,'greenway');
  const retained=clone(charter.onboarding.progress), paid=clone(charter.onboarding.rewardClaims);
  for(let elapsed=0;!Core.getCharterPreview(charter).available&&elapsed<86400;elapsed+=60){if(charter.expedition.completed)act(charter,{type:'expedition-next'});advance(charter,60);}
  assert.ok(Core.getCharterPreview(charter).available); act(charter,{type:'charter'});
  assert.deepEqual(charter.onboarding.progress,retained); assert.deepEqual(charter.onboarding.rewardClaims,paid); valid(charter);
});

test('retained expedition adoption keeps learned guides and does not announce mapped tracks as new', () => {
  const old=clone(require('./fixtures/wayfarers-v4-state.json'));
  const state=Core.normalizeState(old,old.lastUpdate);
  state.upgrades['gear-tools']=1; state.upgrades['gear-boots']=1;
  const before=clone(state.onboarding);
  assert.ok(Core.getRefitPreview(state).available); act(state,{type:'refit'});
  assert.equal(state.expedition.version,3);
  assert.deepEqual(state.onboarding.progress,before.progress); assert.deepEqual(state.onboarding.rewardClaims,before.rewardClaims);
  assert.ok(!Core.getView(state).onboarding.notice.items.some(row=>row.id.startsWith('tier:area:')));
  assert.ok(Core.getView(state).onboarding.notice.items.some(row=>row.id==='feature:focus'),'genuinely new earned feature still announces'); valid(state);
});

test('a pending legacy Tower Lift becomes non-actionable history when its run adopts renamed Signals', () => {
  const old=clone(require('./fixtures/wayfarers-v4-state.json'));
  let state=Core.normalizeState(old,old.lastUpdate);
  state.upgradeTiers.claimed=state.upgradeTiers.claimed.filter(id=>!['area:watchtower:lift','area:watchtower:beacon'].includes(id));
  state.upgradeTiers.pending.push('area:watchtower:lift'); delete state.onboarding;
  state=Core.normalizeState(state,state.lastUpdate); valid(state);
  const pending=Core.getView(state).onboarding.inbox.entries.find(row=>row.id==='ready:area:watchtower:lift'); assert.equal(pending.pending,true);
  state.resources.ore=N.from(1e6); state.resources.coins=N.from(1e6);
  act(state,{type:'buy',id:'gear-tools'}); act(state,{type:'buy',id:'gear-boots'});
  assert.ok(Core.getRefitPreview(state).available); act(state,{type:'refit'});
  const retained=Core.getView(state).onboarding.inbox.entries.find(row=>row.id===pending.id);
  assert.equal(retained.retired,true); assert.equal(retained.pending,false); assert.equal(retained.read,true); assert.equal(retained.openLabel,'View area');
  const result=act(state,retained.openAction); assert.equal(result.destination.upgradeId,'signals'); valid(state);
});

test('offline partitions preserve durable discovery order without paying or completing guides', () => {
  const state=porters(), bulk=clone(state), ticks=clone(state);
  advance(bulk,7200); for(let i=0;i<120;i+=1)advance(ticks,60);
  assert.deepEqual(bulk.onboarding,ticks.onboarding);
  assert.deepEqual(bulk.onboarding.rewardClaims,[]); assert.equal(bulk.onboarding.progress.greenway,0);
  assert.deepEqual(bulk.premium,ticks.premium); assert.deepEqual(bulk.collection,ticks.collection); valid(bulk);
});

test('strict validation and import reject forged discoveries, rewards, prototype keys and skipped completion', () => {
  const original=Core.createState(0);
  const mutations=[x=>{x.onboarding.progress.harbor=3;},x=>{x.onboarding.progress.greenway=3;},x=>{x.onboarding.active='constructor';},x=>{x.onboarding.entries.push('area:harbor');},x=>{x.onboarding.entries.push('tier:area:greenway:scouts');},x=>{x.onboarding.read.push('feature:cards');},x=>{x.onboarding.rewardClaims.push('harbor');},x=>{x.onboarding.announced.push('area:greenway');},x=>{x.onboarding.progress=JSON.parse('{"__proto__":1}');},x=>{x.onboarding.progress=[];},x=>{x.onboarding.version=2;}];
  for(const change of mutations){const state=clone(original);change(state);const before=clone(state);assert.equal(Core.validateState(state).valid,false);assert.deepEqual(state,before);const store=Storage.createStore({storage:null,now:()=>0});assert.equal(store.inspectImport(JSON.stringify({format:Storage.FORMAT,version:Storage.VERSION,savedAt:0,state})).ok,false);}
});

test('fresh testing-reset state starts guides again without carrying old rewards or purchased inventory', () => {
  const previous=mature(); act(previous,{type:'expedition-select',areaId:'greenway'}); finish(previous,'greenway');
  const fresh=Core.createState(previous.createdAt+1000);
  assert.deepEqual(fresh.onboarding,O.initial()); assert.deepEqual(fresh.onboarding.rewardClaims,[]);
  assert.notEqual(Core.getView(fresh).onboarding.identity,Core.getView(previous).onboarding.identity);
  assert.deepEqual(fresh.collection.cards,{}); valid(fresh);
});
