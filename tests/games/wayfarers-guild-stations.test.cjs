'use strict';
const assert=require('node:assert/strict');
const {test}=require('node:test');
const H=require('./helpers/wayfarers-stations.cjs');
const Storage=require('../../js/games/wayfarers-guild/persistence.js');
const F=require('./helpers/wayfarers-onboarding.cjs');
const {Core,P,N,St}=H, clone=x=>JSON.parse(JSON.stringify(x));
let mature;
const established=()=>clone(mature||(mature=H.mature()));
function advance(state,seconds) {while(seconds>1e-8){const r=Core.advance(state,seconds);assert(r.seconds>0);seconds=r.pendingSeconds||0;}}
const near=(a,b,label)=>assert(Math.abs(N.toNumber(a)-N.toNumber(b))<=Math.max(1e-6,Math.abs(N.toNumber(a))*1e-8),label);

test('canonical catalog has thirty stations, 180 unique subjects and eighteen area improvements',()=>{
  assert.equal(St.Content.STATIONS.length,30);assert.equal(St.Content.SKILLS.length,180);assert.equal(St.Content.AREA_UPGRADES.length,18);
  assert.equal(new Set(St.Content.SKILLS.map(d=>d.id)).size,180);assert.equal(new Set(St.Content.SKILLS.map(d=>d.icon)).size,180);
  for(const st of St.Content.STATIONS)assert.equal(st.skillIds.length,6);
  assert.deepEqual(St.Content.STATIONS.find(st=>st.id==='quarry:mine').skillIds.slice(3).map(id=>St.Content.SKILLS.find(d=>d.id===id).name),['Reinforced Shafts','Power Drill','Vein Mapping']);
});

test('opening earns a first purchase in 24 seconds and reveals only three starter icons',()=>{
  const state=Core.createState(1000),v=Core.getView(state).stations;
  assert.equal(state.schemaVersion,8);assert.equal(state.expedition.version,4);assert.equal(v.currentArea.stations.length,2);
  assert.equal(v.currentStation.skills.filter(s=>s.visible).length,3);assert.equal(v.currentStation.skills.filter(s=>s.owned).length,1);
  assert(v.currentStation.skills.slice(3).every(s=>!s.visible));
  const id=v.currentStation.skills[0].id;
  advance(state,23);assert.equal(St.quote(state,id,1).affordable,false);
  advance(state,1);assert.equal(St.quote(state,id,1).affordable,true);
  H.act(state,{type:'station-skill-buy',id,count:1});
  assert.equal(St.eligible(state,St.Content.SKILLS.find(d=>d.id===v.currentStation.skills[1].id)),false);
  assert.equal(Core.validateState(state).valid,true);
});

test('funded first lesson teaches currency, spends no wallet and purchases the canonical station skill',()=>{
  const state=F.completeAreaGuides(Core.createState(1000));
  assert.equal(state.stations.ranks[St.Content.SKILLS[0].id],1);assert.equal(state.stations.mastery['greenway:path'],1);
  assert.equal(N.toNumber(state.resources.coins),12);assert(state.onboarding.practice.currencyRead.includes('coins'));
  assert(state.onboarding.practice.supplies.includes('greenway:upgrade'));assert.equal(Core.validateState(state).valid,true);
});

test('station expansion uses lifetime output and flexible area rank totals, never wallet stock',()=>{
  const state=F.completeAreaGuides(Core.createState(1000));H.fund(state);
  const first=St.Content.STATIONS[0],second=St.Content.STATIONS[1],third=St.Content.STATIONS[2];
  while(state.stations.ranks[first.skillIds[0]]<15)H.act(state,{type:'station-skill-buy',id:first.skillIds[0],count:1});
  assert(!St.eligible(state,second),'huge funded wallet never substitutes for production');
  state.stations.output.greenway=N.from(1800);assert(St.eligible(state,second));H.act(state,{type:'station-build',id:second.id});
  while(state.stations.ranks[first.skillIds[0]]<45)H.act(state,{type:'station-skill-buy',id:first.skillIds[0],count:1});
  state.stations.output.greenway=N.from(14400);assert(St.eligible(state,third),'all45ranks can be invested in first station');
  state.resources.coins=N.zero();assert(St.eligible(state,third),'spending changes no lifetime milestone');
  for(const [order,threshold]of [[3,25],[4,50],[5,75]])assert.equal(St.requirements(state,St.Content.SKILLS.find(d=>d.id===first.skillIds[order]))[1].required,threshold);
});

test('organic balanced and utility builds reach station2 in8–12minutes and station3 in25–40minutes',()=>{
  for(const policy of ['balanced','utility']) {
    const state=F.completeAreaGuides(Core.createState(1000)),times={};
    for(let elapsed=1;elapsed<=2400;elapsed+=1) {
      advance(state,1);
      for(const d of St.Content.SKILLS.filter(d=>d.areaId==='greenway'))if(!state.stations.unlocked.includes(d.id)&&St.eligible(state,d))H.act(state,{type:'station-skill-unlock',id:d.id});
      const options=St.Content.SKILLS.filter(d=>d.areaId==='greenway'&&state.stations.unlocked.includes(d.id)).map(d=>({d,q:St.quote(state,d.id,1)})).filter(o=>o.q.valid&&o.q.affordable);
      options.sort((a,b)=>(policy==='utility'?(a.d.order===2?-.1:0)-(b.d.order===2?-.1:0):0)||N.cmp(a.q.costs.coins,b.q.costs.coins));
      if(options.length)H.act(state,{type:'station-skill-buy',id:options[0].d.id,count:1});
      for(const st of St.Content.STATIONS.filter(st=>st.areaId==='greenway'&&st.index>0))if(!times[st.index]&&St.eligible(state,st)){times[st.index]=elapsed;H.act(state,{type:'station-build',id:st.id});}
      if(times[2])break;
    }
    assert(times[1]>=480&&times[1]<=720,policy+' station2: '+times[1]);assert(times[2]>=1500&&times[2]<=2400,policy+' station3: '+times[2]);
    assert(Core.validateState(state).valid);
  }
});

test('area upgrades apply once, exclude paid prices and new stations leave earlier production intact',()=>{
  const state=established(),area='quarry',before=clone(state),training='area:quarry:training',tools='area:quarry:tools';
  state.stations.areaRanks[training]=0;const plain=P.rawRates(state).stationRates['quarry:mine'].primary;
  state.stations.areaRanks[training]=10;near(P.rawRates(state).stationRates['quarry:mine'].primary,plain*1.3,'all ranks give exactly1.3×once');
  state.stations.areaRanks[tools]=0;const def=St.Content.SKILLS.find(d=>d.id===St.Content.STATIONS.find(st=>st.id==='quarry:mine').skillIds[0]),p=St.cost(state,def,0).coins;
  state.stations.areaRanks[tools]=10;near(St.cost(state,def,0).coins,N.toNumber(p)*.8,'Shared Tools is20% ordinary station price saving before integer rounding');
  assert.deepEqual(state.resources.starshards,before.resources.starshards);assert.deepEqual(state.premium,before.premium);assert.deepEqual(state.collection,before.collection);assert.deepEqual(state.expedition.areas.harbor.voyages,before.expedition.areas.harbor.voyages);
  const fresh=F.completeAreaGuides(Core.createState(1000));H.fund(fresh);fresh.stations.output.greenway=N.from(1800);
  const first=St.Content.STATIONS[0],second=St.Content.STATIONS[1];while(fresh.stations.ranks[first.skillIds[0]]<15)H.act(fresh,{type:'station-skill-buy',id:first.skillIds[0],count:1});
  const rate=St.stationEconomy(fresh,first).primary;H.act(fresh,{type:'station-build',id:second.id});assert.equal(St.stationEconomy(fresh,first).primary,rate);
});

test('fresh station economies purchase real global upgrades and retain actual Trail arrivals',()=>{
  const state=established(),first=St.Content.STATIONS[0],before=Core.getView(state).stations.areas.find(a=>a.id==='greenway').stations[0].rate;
  const tier=Core.getView(state).upgradeTiers.ready.find(t=>t.tracks.some(row=>row.id==='boots'));if(tier)H.act(state,tier.unlockAction);
  H.act(state,{type:'buy',id:'boots',count:1});
  assert(Core.getView(state).stations.areas.find(a=>a.id==='greenway').stations[0].rate>before);
  const deliveries=state.trailDeliveries.deliveries;advance(state,60);assert(state.trailDeliveries.deliveries>deliveries,'real delivery arrivals stay enabled in expedition4');assert(Core.validateState(state).valid);
});

test('display rates and lifetime earnings use canonical companions and Focus through its expiry',()=>{
  const state=F.completeAreaGuides(Core.createState(1000));state.crew.owned.push('scout');state.crew.companions.push('fox');state.crew.companion='fox';
  const view=Core.getView(state).stations;near(view.currentStation.rate,Core.getRates(state).gain.coins,'display is actual credited primary');
  state.expedition.focus={charges:2,recharge:0,active:'greenway',remaining:2,unlocked:true};
  const before=clone(state.stations.output.greenway),rate=Core.getView(state).stations.currentStation.rate;
  advance(state,2);near(N.sub(state.stations.output.greenway,before),rate*2,'last Focus slice receives full quoted production');
  assert.equal(state.expedition.focus.active,null);
});

test('every starter and earned technique changes its real operation or scoped quote',()=>{
  const base=established();
  const signature=state=>JSON.stringify({rates:P.rawRates(state),prices:St.Content.SKILLS.filter(d=>d.core).map(d=>St.cost(state,d,state.stations.ranks[d.id]))});
  for(const def of St.Content.SKILLS) {
    const before=clone(base),after=clone(base),r=before.stations.ranks[def.id];
    after.stations.ranks[def.id]=r+1;after.stations.highRanks[def.id]=Math.max(after.stations.highRanks[def.id],r+1);after.stations.mastery[def.stationId]+=1;
    assert.notEqual(signature(before),signature(after),def.name+' affects its actual production, research or upgrade price');
    const row=St.row(before,def);assert(row.effectText&&row.comparison,def.name+' has truthful preview');
  }
  const target=St.Content.STATIONS.find(s=>s.id==='quarry:mine'),link=St.Content.SKILLS.find(d=>d.name==='Standard Tools');
  const before=St.stationEconomy(base,target).primary;const copy=clone(base);copy.stations.ranks[link.id]+=1;
  assert(St.stationEconomy(copy,target).primary>before,'later Workshop tools strengthen earlier Quarry');
});

test('bulk requires earned exact quantities and rejects stale quotes or cap leftovers atomically',()=>{
  const state=established(),id=St.Content.SKILLS[0].id;
  assert.equal(St.quote(state,id,5).valid,false);
  state.lifetime.refits=1;assert.equal(St.quote(state,id,5).valid,true);
  state.lifetime.refits=10;state.lifetime.charters=1;
  const q=St.quote(state,id,100);assert(q.valid&&q.affordable);const wallet=clone(state.resources.coins);
  H.act(state,{type:'station-skill-buy',id,count:1});const after=clone(state);
  assert.equal(Core.act(state,{type:'station-skill-buy',id,count:100,quote:q.token}).ok,false);assert.deepEqual(state,after);
  state.stations.ranks[id]=998;state.stations.highRanks[id]=998;state.stations.mastery[St.Content.SKILLS[0].stationId]=1200;
  assert.equal(St.quote(state,id,5).valid,false);assert.equal(St.quote(state,id,1).valid,true);assert(N.cmp(wallet,state.resources.coins)>0);
});

test('pending drill payout keeps pre-upgrade credit rather than repricing elapsed work',()=>{
  const state=established(),station=St.Content.STATIONS.find(d=>d.id==='quarry:mine'),drill=St.Content.SKILLS.find(d=>d.name==='Power Drill');
  state.resources.ore=N.from(1e6);state.stations.batchWork[station.id]=0;state.stations.batchCredit[station.id]=N.zero();
  const raw=P.rawRates(state,{baseOnly:true});
  St.tick(state,3,{...raw,stationRates:{[station.id]:raw.stationRates[station.id]}});const pending=clone(state.stations.batchCredit[station.id]);
  const oldBonus=raw.stationRates[station.id].bonus;
  H.act(state,{type:'station-skill-buy',id:drill.id,count:1});assert.deepEqual(state.stations.batchCredit[station.id],pending);
  const next=P.rawRates(state,{baseOnly:true}),before=clone(state.resources.ore),canonical=Core.getRates(state),ratio=N.toNumber(N.div(canonical.gain.ore,next.gain.ore));
  St.tick(state,5,{...next,stationRates:{[station.id]:next.stationRates[station.id]}});
  const expected=N.add(pending,next.stationRates[station.id].bonus*5*ratio);
  near(N.sub(state.resources.ore,before),expected,'one exact drill batch includes3oldseconds+5newseconds');
  assert(next.stationRates[station.id].bonus>oldBonus);assert.equal(state.stations.batchWork[station.id],0);
});

test('one offline interval equals foreground partitions with six areas, boosts, drills and paid voyages',()=>{
  const a=established(),b=clone(a);a.stations.encounter.remaining=0;b.stations.encounter.remaining=0;
  H.act(a,{type:'station-encounter',areaId:'quarry'});H.act(b,{type:'station-encounter',areaId:'quarry'});
  advance(a,300);for(let i=0;i<300;i+=1)advance(b,1);
  for(const id of ['coins','ore','herbs','provisions','knowledge','maps'])near(a.resources[id],b.resources[id],id+' foreground/offline');
  for(const area of St.Content.AREAS)near(a.stations.output[area.id],b.stations.output[area.id],area.id+' lifetime output');
  assert.equal(a.areaSkills.output.harbor,b.areaSkills.output.harbor);assert.equal(Core.validateState(a).valid,true);assert.equal(Core.validateState(b).valid,true);
});

test('Harbor unlocks use actually completed frozen convoys rather than maps stock',()=>{
  const state=established(),second=St.Content.STATIONS.find(s=>s.id==='harbor:sailing-pier');
  state.areaSkills.output.harbor=3;assert.equal(St.requirements(state,second)[2].met,false);
  St.voyageArrived(state,{convoy:1});assert.equal(St.requirements(state,second)[2].met,true);
});

test('released6/7 imports retain all former economy until a confirmed reset adopts stations',()=>{
  for(const fixture of [require('./fixtures/wayfarers-v6-network.json'),require('./fixtures/wayfarers-v5-retained.json').state]) {
    const state=Core.migrateState(clone(fixture)),oldExp=clone(state.expedition);
    assert(!St.active(state));assert.deepEqual(state.expedition,fixture.expedition);assert.equal(state.stations.built.length,0);
    const version7=clone(state);delete version7.stations;version7.schemaVersion=7;
    const upgraded=Core.migrateState(version7);assert.deepEqual(upgraded.expedition,oldExp);assert.equal(Core.validateState(upgraded).valid,true);
    const text=JSON.stringify({format:Storage.FORMAT,version:8,savedAt:upgraded.lastUpdate,state:upgraded});
    assert(Storage.createStore({storage:{getItem:()=>null,setItem:()=>{}}}).inspectImport(text).ok,'top8 preserves nested2/3 imports');
  }
  const state=Core.migrateState(clone(require('./fixtures/wayfarers-v4-state.json'))),before=clone(state);
  state.upgrades['gear-tools']=1;state.upgrades['gear-boots']=1;
  assert(Core.getRefitPreview(state).available);H.act(state,{type:'refit'});
  assert(St.active(state));assert.equal(state.run.id,before.run.id+1);assert(state.stations.built.length>0);assert.equal(Core.validateState(state).valid,true);
  assert.deepEqual(state.caravan,before.caravan);assert.deepEqual(state.premium.owned,before.premium.owned);
});

test('malformed station ledgers are rejected and saves round trip exact outstanding credits',()=>{
  const state=established(),bad=clone(state);delete bad.stations.batchCredit;assert.equal(Core.validateState(bad).valid,false);
  const missing=clone(state);missing.stations.built=[];assert.equal(Core.validateState(missing).valid,false);
  const over=clone(state);over.stations.ranks[St.Content.SKILLS[0].id]=1001;assert.equal(Core.validateState(over).valid,false);
  const store=Storage.createStore({storage:{getItem:()=>null,setItem:()=>{}},now:()=>state.lastUpdate});const output=store.export(state);assert(output.ok);assert.deepEqual(store.inspectImport(output.text).state,state);
});

test('working controls wait for ranked techniques and sorting priority changes actual ore and knowledge',()=>{
  const fresh=F.completeAreaGuides(Core.createState(1000));
  assert(!Core.getView(fresh).onboarding.guides.some(g=>['plans','configuration','technique-config'].includes(g.id)));
  assert.equal(P.choices(fresh,'greenway').filter(row=>row.visible!==false).length,0);
  const state=established(),sorting=St.Content.STATIONS.find(st=>st.id==='quarry:sorting');
  H.act(state,{type:'expedition-choice',areaId:'quarry',id:'rich'});
  const ore=St.stationEconomy(state,sorting);
  H.act(state,{type:'expedition-choice',areaId:'quarry',id:'optics'});
  const knowledge=St.stationEconomy(state,sorting);
  assert(ore.primary>knowledge.primary);
  assert(knowledge.byproducts.filter(row=>row.resource==='knowledge').reduce((n,row)=>n+row.rate,0)>ore.byproducts.filter(row=>row.resource==='knowledge').reduce((n,row)=>n+row.rate,0));
  assert(Core.validateState(state).valid);
  const restored=Storage.createStore({storage:{getItem:()=>null,setItem:()=>{}}}).inspectImport(JSON.stringify({format:Storage.FORMAT,version:8,savedAt:state.lastUpdate,state}));
  assert(restored.ok);assert.equal(restored.state.expedition.areas.quarry.choice,'optics');
});

test('Adaptive Scheduling trades cycles for real commission research or accrued eight-second awards',()=>{
  const state=established(),lab=St.Content.STATIONS.find(st=>st.id==='quarry:crystal-lab');
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'target',slot:0,id:'near'});const steady=St.stationEconomy(state,lab);
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'target',slot:0,id:'deep'});const study=St.stationEconomy(state,lab);
  assert(study.frequency<steady.frequency);assert(study.research>steady.research);
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'target',slot:0,id:'ocean'});const batch=St.stationEconomy(state,lab);
  assert(batch.frequency<steady.frequency);assert(batch.bonus>0);assert.equal(batch.interval,8);
  assert.match(St.row(state,St.Content.SKILLS.find(d=>d.name==='Adaptive Scheduling')).comparison,/bonus ore\/batch/);
  const a=clone(state),b=clone(state);advance(a,17);for(let i=0;i<17;i+=1)advance(b,1);
  for(const resource of ['ore','knowledge','coins','provisions'])near(a.resources[resource],b.resources[resource],resource+' configured batch/offline');
  near(a.stations.batchCredit[lab.id],b.stations.batchCredit[lab.id],'pending sample awards');assert(Core.validateState(a).valid);
});

test('Parallel Furnaces really run a second recipe while the original furnace keeps working',()=>{
  const state=established(),forge=St.Content.STATIONS.find(st=>st.id==='quarry:tool-forge');
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'templates',slot:0,id:'supplies'});const ore=St.stationEconomy(state,forge);
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'templates',slot:0,id:'tools'});const supplies=St.stationEconomy(state,forge);
  assert(ore.primary>supplies.primary);assert(supplies.primary>0);assert(supplies.byproducts.some(row=>row.resource==='provisions'&&row.rate>0));
  H.act(state,{type:'expedition-config',areaId:'quarry',kind:'templates',slot:0,id:'instruments'});const lenses=St.stationEconomy(state,forge);
  assert.equal(lenses.primary,supplies.primary);assert(lenses.byproducts.some(row=>row.resource==='knowledge'&&row.rate>0));assert(Core.validateState(state).valid);
  assert.match(St.row(state,St.Content.SKILLS.find(d=>d.name==='Parallel Furnaces')).comparison,/knowledge\/s from second furnace/);
});

test('before and after comparisons distinguish small high-rank improvements and early discoveries are real',()=>{
  const state=established();
  for(const name of ['Quick Swing','Rich Veins']) {
    const definition=St.Content.SKILLS.find(d=>d.name===name),comparison=St.row(state,definition).comparison;
    const [before,after]=comparison.split(' → ');assert.notEqual(before.trim(),after.split(' ')[0].trim(),name+' '+comparison);
  }
  const ready=H.discovery();assert(St.view(ready).ready.some(row=>row.name==='Waymarks'));
  assert(ready.stations.mastery['greenway:path']>=3);assert(N.cmp(ready.stations.output.greenway,180)>=0);assert(Core.validateState(ready).valid);
});

test('shared area improvements wait for the actual second-station destination action',()=>{
  const state=H.buildReady(),second=St.Content.STATIONS[1],training=St.Content.AREA_UPGRADES.find(d=>d.areaId==='greenway'&&d.kind==='training');
  while(state.stations.ranks[St.Content.SKILLS[0].id]<20)H.act(state,{type:'station-skill-buy',id:St.Content.SKILLS[0].id,count:1});
  H.act(state,{type:'station-build',id:second.id});
  assert.equal(St.view(state).currentArea.areaScopeIntroduced,false);assert.equal(St.eligible(state,training),false);
  assert.equal(St.row(state,training).visible,false);
  H.act(state,{type:'station-select',id:second.id});
  assert.equal(St.view(state).currentArea.areaScopeIntroduced,true);assert.equal(St.eligible(state,training),true);
  assert.equal(St.row(state,training).ready,true);assert(Core.validateState(state).valid);
});

test('reset review quotes the exact actual adoption starter without mutating the released save',()=>{
  const released=require('./helpers/wayfarers-progression.cjs').mature(),before=clone(released),preview=Core.getRefitPreview(released);
  assert.deepEqual(released,before);H.act(released,{type:'refit'});
  assert.equal(N.cmp(released.resources.coins,preview.starter),0);
  const path=St.Content.STATIONS[0].skillIds[0];assert.equal(N.cmp(St.quote(released,path,preview.batch).costs.coins,preview.starter),0);
  assert(Core.validateState(released).valid);
});
