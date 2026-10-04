'use strict';
const Core=require('../../../js/games/wayfarers-guild/core.js');
const P=require('../../../js/games/wayfarers-guild/progression.js');
const F=require('./wayfarers-onboarding.cjs');
const N=Core.Numbers, St=Core.Stations;
const fund=state=>{for(const id of ['coins','ore','herbs','provisions','knowledge','maps'])state.resources[id]=N.from(1e18);};
const act=(state,action)=>{const r=Core.act(state,action);if(!r.ok)throw new Error(action.type+': '+r.message);return r;};
let cached;
function establish(state,ranks=60) {
  fund(state);
  // Explicit diagnostic lifetime output makes this a renderer/behavior fixture,
  // never evidence of organic pacing. All construction/purchases are real.
  for(const areaId of Object.keys(state.expedition.areas))state.stations.output[areaId]=N.from(1e18);
  state.areaSkills.output.harbor=1e9;
  for(let pass=0;pass<12;pass+=1) {
    for(const st of St.Content.STATIONS)if(state.expedition.areas[st.areaId]) {
      if(!state.stations.built.includes(st.id)&&St.eligible(state,st))act(state,{type:'station-build',id:st.id});
      if(!state.stations.built.includes(st.id))continue;
      act(state,{type:'station-select',id:st.id});
      for(const id of st.skillIds) {
        const def=St.Content.SKILLS.find(d=>d.id===id);
        if(!state.stations.unlocked.includes(id)&&St.eligible(state,def))act(state,{type:'station-skill-unlock',id});
        if(state.stations.unlocked.includes(id))while(state.stations.ranks[id]<(def.core?ranks:1))act(state,{type:'station-skill-buy',id,count:1});
      }
    }
  }
  fund(state);return state;
}
function mature(options={}) {
  if(cached&&!options.beforeProject)return JSON.parse(JSON.stringify(cached));
  const state=F.completeAreaGuides(Core.createState(1000));
  for(let i=0;i<2;i+=1) {
    establish(state);
    while(!state.expedition.completed)Core.advance(state,60);
    act(state,{type:'expedition-next'});
    F.completeAreaGuides(state);
  }
  for(const project of P.Content.PROJECTS) {
    establish(state);fund(state);
    const tiers=Core.getView(state).upgradeTiers.ready;
    for(const tier of tiers)act(state,tier.unlockAction);
    if(options.beforeProject===project.id)return state;
    act(state,{type:'expedition-development',id:project.id});
    P.tick(state,project.work/P.rawRates(state).researchRate+1e-5);
    P.sync(state);St.sync(state);
    F.completeAreaGuides(state);
  }
  establish(state);
  for(const def of St.Content.AREA_UPGRADES) {
    if(St.eligible(state,def))act(state,{type:'station-area-unlock',id:def.id});
    if(state.stations.areaUnlocked.includes(def.id))act(state,{type:'station-area-buy',id:def.id,count:1});
  }
  const result=Core.validateState(state);if(!result.valid)throw new Error(result.errors.join('; '));
  cached=JSON.parse(JSON.stringify(state));return state;
}
function lesson(id) {
  const state=id==='tiers'?discovery():mature(id==='projects'?{beforeProject:'wheelworks'}:{});
  // Bulk/Focus renderer fixtures explicitly own one historical Refit, never
  // claim natural pacing or award any spendable prestige currency.
  if(['bulk','focus','automation','reserves','planner'].includes(id)) {
    state.lifetime.refits=1;if(!state.premium.claimedMilestones.includes('first-refit'))state.premium.claimedMilestones.push('first-refit');
    state.expedition.focus={charges:3,recharge:0,active:null,remaining:0,unlocked:true};
  }
  if(['cards','card-archive','card-craft'].includes(id)&&!state.collection.cardsUnlocked)act(state,{type:'collection-unlock',kind:'cards'});
  if(['equipment','gear-craft','gear-repair','gear-reforge'].includes(id)&&!state.collection.equipmentUnlocked)act(state,{type:'collection-unlock',kind:'equipment'});
  const x=state.onboarding.practice;
  x.progress[id]=0;x.active=null;delete x.bindings[id];delete x.intentions[id];
  for(const key of ['proofs','supplies'])x[key]=x[key].filter(entry=>!entry.startsWith(id+':'));
  for(const key of ['rewards','helpRewards'])x[key]=x[key].filter(entry=>entry!==id);
  if(Object.hasOwn(state.onboarding.progress,id)){state.onboarding.progress[id]=0;state.onboarding.rewardClaims=state.onboarding.rewardClaims.filter(entry=>entry!==id);}
  const guide=Core.getView(state).onboarding.guides.find(g=>g.id===id);
  if(!guide||!guide.available)throw new Error('No earned canonical lesson: '+id);
  if(guide.areaId)act(state,{type:'expedition-select',areaId:guide.areaId});
  act(state,guide.visitAction);
  const validation=Core.validateState(state);if(!validation.valid)throw new Error(validation.errors.join('; '));
  return state;
}
function discovery() {
  const state=F.completeAreaGuides(Core.createState(1000)),skill=St.Content.SKILLS[0];
  fund(state);
  while(state.stations.ranks[skill.id]<3)act(state,{type:'station-skill-buy',id:skill.id,count:1});
  Core.advance(state,200);
  if(!St.view(state).ready.some(row=>row.kind==='skill'))throw new Error('No actually earned station discovery');
  return state;
}
function buildReady() {
  const state=F.completeAreaGuides(Core.createState(1000)),first=St.Content.STATIONS[0];fund(state);
  while(state.stations.ranks[first.skillIds[0]]<15)act(state,{type:'station-skill-buy',id:first.skillIds[0],count:1});
  Core.advance(state,600);
  if(!St.eligible(state,St.Content.STATIONS[1]))throw new Error('Station build milestone was not actually earned');
  return state;
}
function expansionReady() {
  const state=buildReady(),first=St.Content.STATIONS[0],second=St.Content.STATIONS[1],third=St.Content.STATIONS[2];
  act(state,{type:'station-build',id:second.id});act(state,{type:'station-select',id:second.id});
  for(const id of first.skillIds.slice(1,3)) {
    act(state,{type:'station-skill-unlock',id});
    while(state.stations.ranks[id]<15)act(state,{type:'station-skill-buy',id,count:1});
  }
  Core.advance(state,1000);act(state,{type:'station-build',id:third.id});act(state,{type:'station-skill-buy',id:third.skillIds[0],count:1});
  Core.advance(state,60);if(!P.canAdvance(state))throw new Error('Expansion requirements were not actually earned');
  return state;
}
module.exports={Core,P,N,St,fund,act,establish,mature,lesson,discovery,buildReady,expansionReady};
