(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./station-content.js') : root.WayfarersStationContent, common ? require('./progression.js') : root.WayfarersProgression, common ? require('./collections.js') : root.WayfarersCollections);
  if (common) module.exports = api;
  if (root) root.WayfarersStations = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, C, P, Collection) {
  'use strict';
  const EPS = 1e-8, clone = x => JSON.parse(JSON.stringify(x));
  const areaDef = id => C.AREAS.find(a => a.id === id), stationDef = id => C.STATIONS.find(a => a.id === id), skillDef = id => C.SKILLS.find(a => a.id === id), areaUpgrade = id => C.AREA_UPGRADES.find(a => a.id === id);
  const map = (rows, value) => Object.fromEntries(rows.map(a => [a.id, typeof value === 'function' ? value(a) : value]));
  const active = s => s.expedition?.version === 4 && !!s.stations;
  const num = (v, cap = 1e15, int = false) => Number.isFinite(v) && v >= 0 && v <= cap && (!int || Number.isSafeInteger(v));
  const exact = (o, keys) => o && typeof o === 'object' && !Array.isArray(o) && Object.keys(o).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(o, k));
  let rateProvider = null, legacyCatalog = null, legacyStages = null;
  const configure = providers => { rateProvider = providers.rateProvider; legacyCatalog = providers.legacyCatalog; legacyStages = providers.legacyStages; };
  function initial(state) {
    return { version: 1, revision: 0, built: [], unlocked: [], ranks: map(C.SKILLS, 0), highRanks: map(C.SKILLS, 0), areaRanks: map(C.AREA_UPGRADES, 0), areaHighRanks: map(C.AREA_UPGRADES, 0), areaUnlocked: [],
      selected: map(C.AREAS, a => C.STATIONS.find(st => st.areaId === a.id).id), output: map(C.AREAS, () => N.zero()), mastery: map(C.STATIONS, 0), phase: map(C.STATIONS, 0), batchWork: map(C.STATIONS, 0), batchCredit: map(C.STATIONS, () => N.zero()), introduced: [],
      boost: { areaId: null, remaining: 0 }, encounter: { remaining: 180, sequence: 0, rng: ((Math.floor(state.createdAt) ^ 0x712f934b) >>> 0) || 1, lastReward: null } };
  }
  function validate(x, state) {
    const keys = ['version','revision','built','unlocked','ranks','highRanks','areaRanks','areaHighRanks','areaUnlocked','selected','output','mastery','phase','batchWork','batchCredit','introduced','boost','encounter'];
    if (!exact(x, keys) || x.version !== 1 || !num(x.revision, 1e12, true)) return false;
    const lists = [['built',C.STATIONS],['unlocked',C.SKILLS],['areaUnlocked',C.AREA_UPGRADES],['introduced',C.STATIONS.concat(C.SKILLS,C.AREA_UPGRADES)]];
    if (lists.some(([key, defs]) => !Array.isArray(x[key]) || new Set(x[key]).size !== x[key].length || x[key].some(id => !defs.some(d => d.id === id)))) return false;
    for (const [key, defs, limit] of [['ranks',C.SKILLS,1000],['highRanks',C.SKILLS,1000],['areaRanks',C.AREA_UPGRADES,15],['areaHighRanks',C.AREA_UPGRADES,15],['mastery',C.STATIONS,1e12],['phase',C.STATIONS,1],['batchWork',C.STATIONS,30]]) {
      if (!exact(x[key], defs.map(d => d.id)) || defs.some(d => !num(x[key][d.id], limit, key !== 'phase' && key !== 'batchWork'))) return false;
    }
    if (!exact(x.selected,C.AREAS.map(a=>a.id)) || C.AREAS.some(a=>stationDef(x.selected[a.id])?.areaId !== a.id)) return false;
    if (!exact(x.output,C.AREAS.map(a=>a.id)) || Object.values(x.output).some(v=>!N.valid(v))) return false;
    if (!exact(x.batchCredit,C.STATIONS.map(a=>a.id)) || Object.values(x.batchCredit).some(v=>!N.valid(v))) return false;
    if (C.SKILLS.some(d=>x.highRanks[d.id] < x.ranks[d.id] || !d.core && x.highRanks[d.id] > 10 || x.ranks[d.id] && !x.unlocked.includes(d.id) || x.unlocked.includes(d.id) && !x.built.includes(d.stationId))) return false;
    if (C.AREA_UPGRADES.some(d=>x.areaRanks[d.id] > d.maxRank || x.areaHighRanks[d.id] < x.areaRanks[d.id] || x.areaHighRanks[d.id] > d.maxRank || x.areaRanks[d.id] && !x.areaUnlocked.includes(d.id))) return false;
    if (!exact(x.boost,['areaId','remaining']) || !num(x.boost.remaining,60) || x.boost.areaId !== null && !areaDef(x.boost.areaId) || !exact(x.encounter,['remaining','sequence','rng','lastReward']) || !num(x.encounter.remaining,1200) || !num(x.encounter.sequence,1e12,true) || !num(x.encounter.rng,4294967295,true) || !x.encounter.rng) return false;
    const reward = x.encounter.lastReward;
    if (reward !== null && (!exact(reward,['areaId','resource','amount','sequence']) || !areaDef(reward.areaId) || !['coins','ore','herbs','provisions','knowledge','maps'].includes(reward.resource) || !N.valid(reward.amount) || !num(reward.sequence,x.encounter.sequence,true))) return false;
    if (active(state)) {
      if (x.built.some(id=>!state.expedition.areas[stationDef(id).areaId]) || C.SKILLS.some(d=>x.ranks[d.id]>maxRank(state,d)||x.highRanks[d.id]>maxRank(state,d))) return false;
      if(x.introduced.some(id=>stationDef(id)?!x.built.includes(id):skillDef(id)?!x.unlocked.includes(id):!x.areaUnlocked.includes(id)))return false;
      for (const areaId of Object.keys(state.expedition.areas)) {
        const first=C.STATIONS.find(s=>s.areaId===areaId);
        if (!x.built.includes(first.id)||!x.unlocked.includes(first.skillIds[0])||!x.built.includes(x.selected[areaId])) return false;
      }
      if (C.STATIONS.some(st=>x.mastery[st.id]<st.skillIds.reduce((sum,id)=>sum+x.ranks[id],0))) return false;
    }
    return true;
  }
  const rank = (state,id) => state.stations?.ranks[id] || 0;
  const areaRank = (state,id,kind) => state.stations?.areaRanks['area:' + id + ':' + kind] || 0;
  const percent = r => r ? .2 + .4 * Math.log(r) / Math.log(10) : 0;
  const strength = (state,def) => { const r=rank(state,def.id); return r ? def.from+(def.to-def.from)*Math.log(r)/Math.log(10) : 0; };
  const techniques = st => st.skillIds.map(skillDef).filter(d=>!d.core);
  const areaScopeIntroduced=(state,id)=>{const second=C.STATIONS.find(st=>st.areaId===id&&st.index===1);return !!second&&state.stations.built.includes(second.id)&&state.stations.introduced.includes(second.id);};
  const namedTechnique = (state,name) => {const d=C.SKILLS.find(d=>d.name===name);return d&&rank(state,d.id)?strength(state,d):0;};
  const PLAN_VALUES={greenway:['trade','freight','survey','mixed','continental','trade-survey'],quarry:['balanced','rich','alloy','precision','mixed','optics','adaptive'],watchtower:['survey','industry','trade'],workshop:['tools','extraction','manufacture','precision','integrated'],ruins:['industry','trade','survey'],harbor:['trade','materials','discovery','commerce']};
  const validChoice=(state,id,value)=>!!PLAN_VALUES[id]?.includes(value);
  function choices(state,id) {
    const a=state.expedition.areas[id];if(!a)return [];
    const open=id==='quarry'&&namedTechnique(state,'Ore Sorting')>0;
    const entries=open?[['balanced','Balanced sorting','Normal ore with a small knowledge recovery'],['rich','Ore priority','25% more Sorting ore; smaller knowledge recovery'],['optics','Sample recovery','20% less Sorting ore; four times the balanced knowledge recovery']]:[[PLAN_VALUES[id][0],'Standard work','Earn a ranked working technique to choose another production plan']];
    return entries.map(([key,label,effect])=>({id:key,label,effect,description:effect,selected:a.choice===key,visible:open,disabled:false,action:{type:'expedition-choice',areaId:id,id:key}}));
  }
  function configurations(state,id) {
    const a=state.expedition.areas[id];if(id!=='quarry'||!a)return [];
    const group=(kind,label,options)=>({id:kind,label,slots:1,selected:clone(a.plans[kind]),options:options.map(([key,name,description])=>({id:key,label:name,description,disabled:false,action:{type:'expedition-config',areaId:id,kind,slot:0,id:key}}))});
    const result=[];
    if(namedTechnique(state,'Adaptive Scheduling'))result.push(group('target','Crystal Lab shifts',[
      ['near','Steady shift','Apply the full Adaptive Scheduling bonus to cycle speed'],
      ['deep','Study shift','Half the cycle bonus; half the scheduling bonus also advances the funded commission'],
      ['ocean','Sample batches','Convert the cycle bonus into 150% extra samples, paid in an eight-second batch']
    ]));
    if(namedTechnique(state,'Parallel Furnaces'))result.push(group('templates','Second furnace recipe',[
      ['supplies','Ore furnace','The second furnace adds ore alongside the original station'],
      ['tools','Provision molds','The second furnace makes provisions instead of extra ore'],
      ['instruments','Precision lenses','The second furnace makes knowledge instead of extra ore']
    ]));
    return result;
  }
  const COLLECTION_KEYS = { greenway:['coins','cargo','travel'], quarry:['picks','haul','smelt','oreYield'], watchtower:['knowledge','maps','research'], workshop:['assembly','workshopYield','oreSaving'], ruins:['delving','interpretation','recovery','artifacts'], harbor:['voyage','cargo','maps'] };
  function sync(state) {
    if (!active(state)) return;
    const x = state.stations;
    for (const areaId of Object.keys(state.expedition.areas)) {
      const first = C.STATIONS.find(s=>s.areaId===areaId);
      if (!x.built.includes(first.id)) { x.built.push(first.id); x.unlocked.push(first.skillIds[0]); x.revision += 1; }
    }
  }
  function requirements(state, def) {
    const x = state.stations, station = stationDef(def.stationId || def.id), area = areaDef(def.areaId), built = station && x.built.includes(station.id);
    const req = (label, icon, current, required, big = false) => ({label,icon,current,required,met:big ? N.cmp(current,required)>=0 : current+EPS>=required});
    const lifetime=area.id==='harbor'?N.from(state.areaSkills.output.harbor):x.output[area.id], unit=area.id==='harbor'?'completed voyages':area.resource;
    if (def.stationId) {
      const needed = [0,3,10,25,50,75][def.order];
      const out = [0,.1,.4,2,8,32][def.order] * area.target;
      return [req('Build ' + station.name,area.icon,built?1:0,1),req(needed + ' ' + station.name + ' ranks',area.icon,x.mastery[station.id],needed),req(N.format(out) + ' lifetime ' + unit,area.resource,lifetime,N.from(out),true)]
        .concat(def.project ? [req(P.Content.PROJECTS.find(p=>p.id===def.project).name,'research',state.expedition.projects.includes(def.project)?1:0,1)] : []);
    }
    if (station) {
      const siblings = C.STATIONS.filter(s=>s.areaId===area.id), prev = siblings[station.index-1], total = siblings.reduce((n,s)=>n+x.mastery[s.id],0), purchased = [0,15,45,90,150][station.index];
      return station.index ? [req('Build ' + prev.name,area.icon,x.built.includes(prev.id)?1:0,1),req(purchased + ' ' + area.name + ' purchased ranks',area.icon,total,purchased),req(N.format(area.target * [0,1,8,40,200][station.index]) + ' lifetime ' + unit,area.resource,lifetime,N.from(area.target * [0,1,8,40,200][station.index]),true)]
        .concat(station.project ? [req(P.Content.PROJECTS.find(p=>p.id===station.project).name,'research',state.expedition.projects.includes(station.project)?1:0,1)] : []) : [req('Discover ' + area.name,area.icon,state.expedition.areas[area.id]?1:0,1)];
    }
    const siblings = C.STATIONS.filter(s=>s.areaId===area.id), count = siblings.filter(s=>x.built.includes(s.id)).length, total = siblings.reduce((n,s)=>n+x.mastery[s.id],0), order=C.AREA_UPGRADES.filter(d=>d.areaId===area.id).indexOf(def);
    return [req('Operate two stations',area.icon,count,2),req('Open '+siblings[1].name,'guild',areaScopeIntroduced(state,area.id)?1:0,1),req([20,35,55][order] + ' area ranks','guild',total,[20,35,55][order])];
  }
  function eligible(state, def) { return active(state) && !!def && requirements(state,def).every(r=>r.met); }
  function maxRank(state,def) { return def.stationId && def.core ? state.expedition.areas[def.areaId]?.cap || 100 : def.maxRank; }
  function cost(state,def,r) {
    const area=areaDef(def.areaId), station=stationDef(def.stationId), x=state.stations;
    const high=station ? x.highRanks[def.id] : x.areaHighRanks[def.id];
    const rebuilding=state.expedition.renewed && r<high ? .5 : 1;
    const saving=station?techniques(station).filter(d=>d.kind==='saving').reduce((sum,d)=>sum+strength(state,d),0):0;
    const localDiscount=station ? Math.max(.4,1-.02*areaRank(state,area.id,'tools')-saving-(station.specialty==='efficiency' ? .25*rank(state,station.skillIds[2])/(25+rank(state,station.skillIds[2])) : 0)) : 1;
    const base=station ? area.base * Math.pow(4,station.index) * (def.core ? [1,1.35,1.8][def.order] : 4) : area.base*30;
    const price=Math.ceil(base * (def.core || !station ? Math.pow(1+r/(station?4:2),2.6) : Math.pow(2.4,r)) * localDiscount * rebuilding);
    const costs={coins:N.from(price)};
    if (station && area.resource!=='coins' && (def.order>=3 || r>=25)) costs[area.resource]=N.from(Math.ceil(price*.025));
    return costs;
  }
  function quote(state,id,count=1) {
    const def=skillDef(id)||areaUpgrade(id), x=state.stations, station=stationDef(def?.stationId), r=def ? (station?x?.ranks[id]:x?.areaRanks[id])||0 : 0;
    const q={id,count,rank:r,rankAfter:r+count,costs:{},valid:false,affordable:false,reason:''};
    if (!active(state)||!def||!(station?x.unlocked:x.areaUnlocked).includes(id)) {q.reason='Unlock this upgrade first.';return q;}
    if (!Number.isSafeInteger(count)||!P.batchModes(state).some(m=>m.count===count&&m.unlocked)) {q.reason='Earn this exact purchase quantity first.';return q;}
    if (r+count>maxRank(state,def)) {q.reason='Only '+(maxRank(state,def)-r)+' ranks remain. Choose a smaller quantity.';return q;}
    for(let i=0;i<count;i+=1) for(const [resource,value] of Object.entries(cost(state,def,r+i))) q.costs[resource]=N.add(q.costs[resource]||0,value);
    q.valid=true;q.affordable=Object.entries(q.costs).every(([resource,v])=>N.cmp(state.resources[resource],v)>=0);
    if(!q.affordable)q.reason='Save for the entire ×'+count+' purchase.';
    q.token=[state.run.id,state.expedition.revision,x.revision,id,r,count,maxRank(state,def)].join(':');return q;
  }
  function stationEconomy(state,station,options={}) {
    const x=state.stations, area=areaDef(station.areaId), rs=station.skillIds.map(id=>rank(state,id));
    const activeBoost=options.unboosted?1:(x.boost.areaId===area.id&&x.boost.remaining>EPS?2:1)*(state.expedition.focus.active===area.id&&state.expedition.focus.remaining>EPS?P.Content.FOCUS.multiplier:1);
    const all=1+.03*areaRank(state,area.id,'training');
    const tech=techniques(station), amount=kind=>tech.filter(d=>d.kind===kind).reduce((sum,d)=>sum+strength(state,d),0);
    const backward=C.SKILLS.filter(d=>!d.core&&x.built.includes(d.stationId)&&(d.kind==='link'&&d.target===station.id||d.kind==='backlink'&&d.areaId===area.id&&stationDef(d.stationId).index>station.index)).reduce((sum,d)=>sum+strength(state,d),0);
    const meta=(1+.28*Math.sqrt(state.lifetime.refits))*(1+.65*state.lifetime.charters);
    const yieldPerCycle=area.rate*station.factor*station.baseCycle*(1+.25*rs[0]);
    const scheduling=station.id==='quarry:crystal-lab'?namedTechnique(state,'Adaptive Scheduling'):0,shift=state.expedition.areas[area.id].plans.target;
    const cadence=amount('cadence-tech')-scheduling+scheduling*(shift==='ocean'?0:shift==='deep'?.5:1);
    const frequency=(1+.15*rs[1])*(1+cadence)/station.baseCycle;
    const value=station.specialty==='value'?1+.12*rs[2]:1;
    const support=station.specialty==='support'?.75*rs[2]/(30+rs[2]):0;
    const equipment=Collection.modifiers(state), equipmentFactor=1+COLLECTION_KEYS[area.id].reduce((sum,key)=>sum+(equipment[key]||0),0);
    const upgraded=(id,p)=>1+p*(state.upgrades[id]||0);
    const guildFactor={greenway:upgraded('boots',.18)*upgraded('gear-boots',.2)*upgraded('preparation',.15),quarry:upgraded('miners',.32)*upgraded('gear-tools',.45),watchtower:upgraded('scholars',.35)*upgraded('gear-instruments',.4),workshop:upgraded('cooks',.3),ruins:upgraded('foragers',.35),harbor:upgraded('surveyors',.35)*upgraded('gear-instruments',.4)}[area.id];
    const supply=area.id==='quarry'||area.id==='workshop'||area.id==='ruins'?1+.25*Math.sqrt(state.refitUpgrades.supply):area.id==='watchtower'?1+.3*Math.sqrt(state.refitUpgrades.insight):1;
    const basePrimary=yieldPerCycle*frequency*all*value*(1+backward+amount('local')+.2*(amount('link')+amount('backlink')))*activeBoost*meta*equipmentFactor*supply*guildFactor;
    const supportPrimary=station.index===0?C.STATIONS.filter(st=>st.areaId===area.id&&st.specialty==='support'&&st.index&&x.built.includes(st.id)).reduce((sum,st)=>sum+.75*rank(state,st.skillIds[2])/(30+rank(state,st.skillIds[2])),0):0;
    let primary=basePrimary*(1+supportPrimary);
    const sorting=station.id==='quarry:sorting'?namedTechnique(state,'Ore Sorting'):0,sortingPlan=state.expedition.areas[area.id].choice;
    if(sorting)primary*=sortingPlan==='rich'?1.25:sortingPlan==='optics'?.8:1;
    const parallel=amount('parallel'),recipe=state.expedition.areas[area.id].plans.templates[0],parallelResource=recipe==='tools'?'provisions':recipe==='instruments'?'knowledge':area.resource,parallelRate=primary*parallel;
    if(parallel&&parallelResource===area.resource)primary+=parallelRate;
    const secondary=primary*(station.specialty==='discovery'?Math.min(.2,.2*rs[2]/(10+rs[2])):support*.15);
    const byproducts=tech.filter(d=>d.kind==='byproduct').map(d=>({resource:d.target,rate:primary*strength(state,d)}));
    if(sorting)byproducts.push({resource:'knowledge',rate:primary*sorting*(sortingPlan==='rich'?.05:sortingPlan==='optics'?.6:.15)});
    if(parallel&&parallelResource!==area.resource)byproducts.push({resource:parallelResource,rate:parallelRate});
    const batch=tech.find(d=>d.kind==='batch'), interval=batch?.interval||8;
    return {primary,secondary,byproducts,frequency,cycle:1/frequency,yieldPerCycle,support,backward,bonus:primary*(amount('batch')+(shift==='ocean'?scheduling*1.5:0)),interval,research:primary*(amount('research')+(shift==='deep'?scheduling*.5:0)),areaId:area.id,resource:area.resource,secondaryResource:station.secondary,activeBoost};
  }
  function enrichRates(state,raw,options={}) {
    if(!active(state))return raw;
    const gain=Object.fromEntries(Object.keys(raw.gain).map(id=>[id,0])), stationRates={};
    for(const st of C.STATIONS) if(state.stations.built.includes(st.id)) {
      const econ=stationEconomy(state,st,options);stationRates[st.id]=econ;
      gain[econ.resource]+=econ.primary;gain[econ.secondaryResource]+=econ.secondary;
      econ.byproducts.forEach(row=>{gain[row.resource]+=row.rate;});
    }
    const areas=Object.fromEntries(Object.entries(raw.areas).map(([id,r])=>{
      const throughput=Object.values(stationRates).filter(st=>st.areaId===id).reduce((n,st)=>n+st.primary,0), base=areaDef(id).rate;
      const research=Object.values(stationRates).filter(st=>st.areaId===id).reduce((sum,st)=>sum+st.research,0);
      return [id,{...r,work:r.work*Math.max(1,throughput/base),finale:r.finale*Math.max(1,throughput/base),research:(r.research||0)+throughput*.02+research,stationOutput:throughput}];
    }));
    // Base stations earn without compulsory downstream consumption. Paid voyage
    // manifests retain the existing launch/arrival conservation protocol.
    return {...raw,gain,drain:Object.fromEntries(Object.keys(raw.drain).map(id=>[id,0])),areas,stationRates,researchRate:Object.values(areas).reduce((n,a)=>n+(a.research||0),0)*.6};
  }
  function nextEvent(state,raw) {
    if(!active(state))return Infinity;
    let next=state.stations.boost.remaining>EPS?state.stations.boost.remaining:Infinity;
    for(const [id,econ] of Object.entries(raw.stationRates||{}))if(econ.bonus>EPS)next=Math.min(next,Math.max(EPS,econ.interval-state.stations.batchWork[id]));
    return next;
  }
  function tick(state,seconds,raw) {
    if(!active(state))return;
    const x=state.stations, canonical=rateProvider?rateProvider(state):null;
    for(const [id,econ] of Object.entries(raw.stationRates||{})) {
      const st=stationDef(id), ratio=canonical&&raw.gain[econ.resource]>EPS?N.toNumber(N.div(canonical.gain[econ.resource],raw.gain[econ.resource])):1;
      const earned=N.from(econ.primary*seconds*ratio);
      x.output[st.areaId]=N.add(x.output[st.areaId],earned);
      x.phase[id]=(x.phase[id]+econ.frequency*seconds)%1;
      if(econ.bonus>EPS) {
        x.batchWork[id]+=seconds;
        x.batchCredit[id]=N.add(x.batchCredit[id],econ.bonus*seconds*ratio);
        if(x.batchWork[id]>=econ.interval-EPS) {
          const remaining=Math.max(0,x.batchWork[id]-Math.floor((x.batchWork[id]+EPS)/econ.interval)*econ.interval);
          const pending=N.from(econ.bonus*remaining*ratio), payout=N.sub(x.batchCredit[id],pending);
          state.resources[econ.resource]=N.add(state.resources[econ.resource],payout);
          if(econ.resource==='coins')state.lifetime.coins=N.add(state.lifetime.coins,payout);
          x.output[st.areaId]=N.add(x.output[st.areaId],payout);x.batchWork[id]=remaining;x.batchCredit[id]=pending;
        }
      }
    }
    x.boost.remaining=Math.max(0,x.boost.remaining-seconds);if(x.boost.remaining<=EPS)x.boost.areaId=null;
    x.encounter.remaining=Math.max(0,x.encounter.remaining-seconds);sync(state);
  }
  function describe(state,def) {
    const r=skillDef(def.id)?rank(state,def.id):state.stations.areaRanks[def.id], st=stationDef(def.stationId);
    if(!st)return def.effect;
    if(!def.core) {
      const range=Math.round(def.from*100)+'–'+Math.round(def.to*100)+'%';
      if(def.kind==='batch')return range+' extra '+def.resource+' accrued as '+def.name+' batches every '+def.interval+' seconds.';
      if(def.kind==='cadence-tech')return range+' faster '+st.name+' cycles.';
      if(def.kind==='local')return range+' greater '+st.name+' production.';
      if(def.kind==='byproduct')return 'Recover '+def.target+' equal to '+range+' of '+st.name+' production, alongside its normal output.';
      if(def.kind==='saving')return range+' lower '+st.name+' upgrade prices, combined with Shared Tools.';
      if(def.kind==='research')return range+' of '+st.name+' output also advances the funded commission.';
      if(def.kind==='parallel')return 'A second furnace produces '+range+' extra output. Choose ore, provisions or knowledge in Processing.';
      if(def.kind==='link')return range+' stronger '+stationDef(def.target).name+' in '+areaDef(stationDef(def.target).areaId).name+' when built; one fifth also strengthens '+st.name+'.';
      if(def.kind==='backlink')return range+' stronger earlier '+areaDef(def.areaId).name+' stations; one fifth also strengthens '+st.name+'.';
    }
    const effect={yield:'+25% base '+st.unit+' per cycle per rank',cadence:'+15% base cycle frequency per rank',discovery:'More '+def.secondary+' from '+st.unit+' discoveries; bounded at 20% of station output',value:'+12% '+areaDef(def.areaId).resource+' value per rank',efficiency:'1% less '+st.name+' upgrade cost per rank; capped at 25%',support:'Strengthen the original '+areaDef(def.areaId).name+' station; bounded at 75%',batch:'Earn an extra '+Math.round(percent(Math.max(1,r))*.3*100)+'% bonus batch every 8 seconds',backlink:'Strengthen every earlier '+areaDef(def.areaId).name+' station',branch:'Add '+def.secondary+' alongside normal '+areaDef(def.areaId).resource+' production'};
    if(def.name==='Power Drill')return 'Automatically breaks ore pockets every 8 seconds for an extra ore batch.';
    return effect[def.kind];
  }
  function creditedRatios(state) {
    const raw=P.rawRates(state,{baseOnly:true}),canonical=rateProvider?rateProvider(state):null;
    return Object.fromEntries(Object.entries(raw.gain).map(([id,n])=>[id,canonical&&n>EPS?N.toNumber(N.div(canonical.gain[id],n)):1]));
  }
  function creditedEconomy(state,st,ratios) {
    const econ=stationEconomy(state,st), factors=ratios||creditedRatios(state);
    return {...econ,primary:econ.primary*factors[econ.resource],secondary:econ.secondary*factors[econ.secondaryResource],bonus:econ.bonus*factors[econ.resource],byproducts:econ.byproducts.map(row=>({...row,rate:row.rate*factors[row.resource]}))};
  }
  function preciseChange(current,next,unit) {
    const scale=Math.max(Math.abs(current),Math.abs(next)),power=scale>=1000?Math.min(4,Math.floor(Math.log10(scale)/3)):0,divisor=Math.pow(1000,power),suffix=['','K','M','B','T'][power];
    let from,to;
    for(let digits=2;digits<=8;digits+=1) {
      from=(current/divisor).toFixed(digits).replace(/\.?0+$/,'');to=(next/divisor).toFixed(digits).replace(/\.?0+$/,'');
      if(from!==to||current===next)break;
    }
    return from+suffix+' → '+to+suffix+(unit?' '+unit:'');
  }
  function row(state,def,ratios) {
    const x=state.stations, local=!!def.stationId, owned=(local?x.unlocked:x.areaUnlocked).includes(def.id), req=requirements(state,def), ready=!owned&&req.every(r=>r.met), r=local?x.ranks[def.id]:x.areaRanks[def.id], q=quote(state,def.id,state.expedition.batch), cap=maxRank(state,def);
    const visible=local?def.core||owned||ready:owned||ready;
    let comparison='';
    if(local) {
      const factors=ratios||creditedRatios(state),current=creditedEconomy(state,stationDef(def.stationId),factors), copy={...state,stations:{...x,ranks:{...x.ranks,[def.id]:Math.min(cap,r+(q.valid?q.count:1))}}}, next=creditedEconomy(copy,stationDef(def.stationId),factors);
      if(def.name==='Adaptive Scheduling'&&state.expedition.areas[def.areaId].plans.target==='ocean')comparison=preciseChange(current.bonus*current.interval,next.bonus*next.interval,'bonus '+def.resource+'/batch');
      else if(def.kind==='cadence'||def.kind==='cadence-tech')comparison=preciseChange(current.cycle,next.cycle,'seconds/cycle');
      else if(def.kind==='efficiency')comparison=(25*r/(25+r)).toFixed(1)+'% → '+(25*Math.min(cap,r+(q.valid?q.count:1))/(25+Math.min(cap,r+(q.valid?q.count:1)))).toFixed(1)+'% cost saving';
      else if(def.kind==='batch')comparison=N.format(current.bonus*def.interval)+' → '+N.format(next.bonus*def.interval)+' bonus '+def.resource+'/batch';
      else if(['backlink','link','saving','research'].includes(def.kind))comparison=Math.round(strength(state,def)*100)+'% → '+Math.round(strength(copy,def)*100)+'% '+({backlink:'earlier output',link:'linked support',saving:'price saving',research:'research share'}[def.kind]);
      else if(def.kind==='byproduct')comparison=preciseChange(current.byproducts.filter(d=>d.resource===def.target).reduce((sum,d)=>sum+d.rate,0),next.byproducts.filter(d=>d.resource===def.target).reduce((sum,d)=>sum+d.rate,0),def.target+'/s');
      else if(def.kind==='parallel'&&state.expedition.areas[def.areaId].plans.templates[0]!=='supplies') {
        const resource=state.expedition.areas[def.areaId].plans.templates[0]==='tools'?'provisions':'knowledge';
        comparison=preciseChange(current.byproducts.filter(d=>d.resource===resource).reduce((sum,d)=>sum+d.rate,0),next.byproducts.filter(d=>d.resource===resource).reduce((sum,d)=>sum+d.rate,0),resource+'/s from second furnace');
      }
      else if(def.kind==='support')comparison=(100*current.support).toFixed(1)+'% → '+(100*next.support).toFixed(1)+'% original station support';
      else comparison=preciseChange(def.kind==='discovery'||def.kind==='branch'?current.secondary:current.primary,def.kind==='discovery'||def.kind==='branch'?next.secondary:next.primary,'/s');
    } else comparison=def.kind==='training'?r*3+'% → '+Math.min(cap,r+(q.valid?q.count:1))*3+'% base output':def.kind==='discount'?r*2+'% → '+Math.min(cap,r+(q.valid?q.count:1))*2+'% discount':r*2+'s → '+Math.min(cap,r+(q.valid?q.count:1))*2+'s bonus duration';
    const type=local?'station-skill-':'station-area-';
    return {...def,catalogId:def.id,skillId:local?def.id:null,group:'area',scope:local?'station':'area',rank:r,level:r,maxRank:cap,maxLevel:cap,maxed:r===cap,status:owned?'learned':ready?'ready':'locked',state:owned?'learned':ready?'ready':'locked',visible,owned,ready,locked:!owned&&!ready,disabled:owned?!q.valid||!q.affordable:!ready,requirements:req,cost:Object.entries(q.costs).map(([resource,amount])=>({resource,amount,text:N.format(amount)+' '+resource})),costs:q.costs,quantity:q.count,rankAfter:q.rankAfter,comparison,effectText:describe(state,def),description:describe(state,def),reason:q.reason,action:owned?{type:type+'buy',id:def.id,areaId:def.areaId,count:q.count,quote:q.token}:{type:type+'unlock',id:def.id,areaId:def.areaId},unlockAction:{type:type+'unlock',id:def.id,areaId:def.areaId},fittingQuantityAction:owned&&q.count>cap-r?{type:'expedition-batch',count:P.batchModes(state).filter(m=>m.unlocked&&m.count<=cap-r).pop()?.count||1}:null};
  }
  function legacyView(state) {
    const exp=state.expedition, catalog=legacyCatalog?legacyCatalog(state):exp?.version===3?P.catalog(state):[], skills=state.areaSkills?requireSkillsView(state):[];
    const areas=C.AREAS.filter(a=>exp?.areas[a.id]).map(a=>{
      const operationStages=legacyStages?legacyStages(state,a.id):[];
      let stations=C.STATIONS.filter(st=>st.areaId===a.id&&!(a.id==='quarry'&&st.localId==='crystal-lab')).map(st=>{
        const aliases=st.skillIds.map(id=>skillDef(id).alias).filter(Boolean);
        if(a.id==='quarry'&&st.localId==='tool-forge')aliases.push('shift-planning');
        const rows=catalog.filter(r=>r.areaId===a.id&&r.visible!==false&&aliases.includes(r.action?.id)).concat(skills.filter(r=>r.areaId===a.id&&aliases.includes(r.skillId)&&r.owned));
        const stage=operationStages.find(stage=>aliases.includes(stage.id)), rate=stage?.rate||0;
        return {...st,name:a.id==='quarry'&&st.localId==='tool-forge'?'Refinery':st.name,status:'built',legacy:true,working:rate>EPS,rate,output:rate,outputText:stage?.rateText||N.format(rate)+'/s',cycleProgress:(state.lastUpdate/1000/st.baseCycle)%1,visualTier:Math.min(4,1+Math.floor(rows.reduce((n,r)=>n+(r.rank||0),0)/15)),boost:false,mastery:rows.reduce((n,r)=>n+(r.rank||0),0),skills:rows,requirements:[],selectAction:{type:'station-select',areaId:a.id,id:st.id}};
      }).filter(st=>st.skills.length);
      return {...a,stations,areaUpgrades:[],nextStation:null,areaScopeIntroduced:true};
    });
    const currentArea=areas.find(a=>a.id===exp?.selectedArea)||areas[0], selected=state.stations?.selected[currentArea?.id], currentStation=currentArea?.stations.find(s=>s.id===selected)||currentArea?.stations[0];
    return {active:false,legacy:true,selectedArea:currentArea?.id,selectedStation:currentStation?.id,areas,currentArea,currentStation,ready:[],batchModes:exp?.version===3?P.batchModes(state):[],batch:exp?.batch||1,encounter:null};
  }
  let skillsView=()=>[];
  const setSkillsView=fn=>{skillsView=fn;};
  const requireSkillsView=state=>skillsView(state);
  function view(state) {
    if(!active(state))return legacyView(state);
    const x=state.stations, ratios=creditedRatios(state), areas=C.AREAS.filter(a=>state.expedition.areas[a.id]).map(a=>{
      const stations=C.STATIONS.filter(st=>st.areaId===a.id).map(st=>{
        const built=x.built.includes(st.id), req=requirements(state,st), ready=!built&&req.every(r=>r.met), econ=creditedEconomy(state,st,ratios), skills=built?st.skillIds.map(id=>row(state,skillDef(id),ratios)):[];
        return {...st,status:built?'built':ready?'ready':'locked',legacy:false,output:econ.primary,outputText:N.format(econ.primary)+'/s',mastery:x.mastery[st.id],requirements:req,buildAction:{type:'station-build',id:st.id,areaId:a.id},selectAction:{type:'station-select',id:st.id,areaId:a.id},skills,working:built&&econ.primary>0,rate:econ.primary,cycleProgress:x.phase[st.id],visualTier:Math.min(4,1+Math.floor(x.mastery[st.id]/15)),boost:econ.activeBoost>1};
      });
      const nextStation=stations.find(st=>st.status!=='built');
      return {...a,stations:stations.filter(st=>st.status==='built'||st===nextStation),areaUpgrades:C.AREA_UPGRADES.filter(d=>d.areaId===a.id).map(d=>row(state,d)),areaScopeIntroduced:areaScopeIntroduced(state,a.id),nextStation,lifetimeOutput:x.output[a.id],lifetimeOutputText:N.format(x.output[a.id])};
    });
    const currentArea=areas.find(a=>a.id===state.expedition.selectedArea)||areas[0], currentStation=currentArea.stations.find(st=>st.id===x.selected[currentArea.id]&&st.status==='built')||currentArea.stations[0];
    const ready=areas.flatMap(a=>a.stations.filter(st=>st.status==='ready').map(st=>({...st,kind:'station'})).concat(a.stations.flatMap(st=>st.skills.filter(sk=>sk.ready).map(sk=>({...sk,kind:'skill'}))),a.areaUpgrades.filter(r=>r.ready).map(r=>({...r,kind:'area'}))));
    return {active:true,legacy:false,selectedArea:currentArea.id,selectedStation:currentStation.id,areas,currentArea,currentStation,ready,batchModes:P.batchModes(state),batch:state.expedition.batch,encounter:{ready:x.encounter.remaining<=EPS,remaining:x.encounter.remaining,sequence:x.encounter.sequence,lastReward:clone(x.encounter.lastReward),boost:clone(x.boost),action:{type:'station-encounter',areaId:currentArea.id}}};
  }
  function act(state,action) {
    if(action.type==='station-select') {
      const st=stationDef(action.id);
      if(!st||!state.expedition.areas[st.areaId]||active(state)&&!state.stations.built.includes(st.id))return {ok:false,message:'Build this station first.'};
      state.expedition.selectedArea=st.areaId;state.stations.selected[st.areaId]=st.id;
      if(!state.stations.introduced.includes(st.id))state.stations.introduced.push(st.id);
      return {ok:true,message:st.name+' selected.'};
    }
    if(!active(state))return {ok:false,message:'This run keeps its existing economy. Station progression starts at your next confirmed reset.'};
    const x=state.stations;
    if(action.type==='station-build') {
      const st=stationDef(action.id);if(!st||x.built.includes(st.id)||!eligible(state,st))return {ok:false,message:'Complete this station’s lifetime production and rank milestones.'};
      x.built.push(st.id);x.unlocked.push(st.skillIds[0]);x.revision+=1;state.expedition.revision+=1;
      return {ok:true,message:st.name+' unlocked. Earlier stations keep working.',stationUnlocked:st.id};
    }
    if(action.type==='station-encounter') {
      const area=areaDef(action.areaId||state.expedition.selectedArea);
      if(!area||!state.expedition.areas[area.id]||x.encounter.remaining>EPS)return {ok:false,message:'The next cache is still on its way.'};
      const current=rateProvider?rateProvider(state).gain[area.resource]:N.from(area.rate), amount=N.max(N.mul(current,60),area.rate*60);
      state.resources[area.resource]=N.add(state.resources[area.resource],amount);if(area.resource==='coins')state.lifetime.coins=N.add(state.lifetime.coins,amount);
      // Active cache rewards are not lifetime production achievements.
      let seed=x.encounter.rng;seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;x.encounter.rng=seed>>>0;
      x.encounter.remaining=600+(x.encounter.rng%301);x.encounter.sequence+=1;x.encounter.lastReward={areaId:area.id,resource:area.resource,amount,sequence:x.encounter.sequence};
      x.boost={areaId:area.id,remaining:30+2*areaRank(state,area.id,'shifts')};x.revision+=1;
      return {ok:true,message:'+'+N.format(amount)+' '+area.resource+' · ×2 '+area.name+' output',reward:{[area.resource]:amount},stationBoost:clone(x.boost)};
    }
    const def=skillDef(action.id)||areaUpgrade(action.id), local=!!def?.stationId;
    if(!def)return {ok:false,message:'Choose a station upgrade.'};
    if((action.type.startsWith('station-skill-')&&!local)||(action.type.startsWith('station-area-')&&local))return {ok:false,message:'Use this upgrade’s own menu control.'};
    const owned=local?x.unlocked:x.areaUnlocked;
    if(action.type==='station-skill-unlock'||action.type==='station-area-unlock') {
      if(owned.includes(def.id)||!eligible(state,def))return {ok:false,message:'Complete the listed upgrade requirements.'};
      owned.push(def.id);x.revision+=1;return {ok:true,message:def.name+' unlocked.',stationSkillUnlocked:def.id};
    }
    if(action.type==='station-skill-buy'||action.type==='station-area-buy') {
      const q=quote(state,def.id,action.count===undefined?state.expedition.batch:action.count);
      if(!q.valid||!q.affordable||action.quote!==undefined&&action.quote!==q.token)return {ok:false,message:!q.valid||!q.affordable?q.reason:'This quote changed. Review its current price.'};
      for(const [resource,value] of Object.entries(q.costs))state.resources[resource]=N.sub(state.resources[resource],value);
      (local?x.ranks:x.areaRanks)[def.id]=q.rankAfter;
      const highs=local?x.highRanks:x.areaHighRanks;highs[def.id]=Math.max(highs[def.id],q.rankAfter);
      if(local)x.mastery[def.stationId]+=q.count;
      x.revision+=1;state.expedition.revision+=1;
      // Historical track rank mirrors retain existing commission/area gates.
      const oldTrack=local&&def.alias&&P.Content.AREAS.find(a=>a.id===def.areaId).tracks.find(t=>t.id===def.alias);
      if(oldTrack&&(oldTrack.index<3||state.expedition.areas[def.areaId].learned.includes(def.alias))) {
        const a=state.expedition.areas[def.areaId];if(!a.learned.includes(def.alias))a.learned.push(def.alias);a.ranks[def.alias]=q.rankAfter;a.highRanks[def.alias]=Math.max(a.highRanks[def.alias],q.rankAfter);
      }
      if(!x.introduced.includes(def.id))x.introduced.push(def.id);
      P.sync(state);sync(state);return {ok:true,quantity:q.count,message:def.name+' +'+q.count+' · rank '+q.rankAfter};
    }
    return {ok:false,message:'Choose an available station action.'};
  }
  function reset(state,adopting=false) {
    const x=state.stations||initial(state);state.stations=x;
    if(adopting) {
      for(const st of C.STATIONS) if(state.expedition.areas[st.areaId]) {
        const known=st.index===0||st.skillIds.some(id=>{const d=skillDef(id),a=state.expedition.areas[st.areaId];return d.alias&&(a.learned.includes(d.alias)||state.areaSkills.unlocked.includes(d.alias));});
        if(known&&!x.built.includes(st.id))x.built.push(st.id);
        if(known)for(const id of st.skillIds) {
          const d=skillDef(id),a=state.expedition.areas[st.areaId];
          if(d.order===0||d.alias&&(a.learned.includes(d.alias)||state.areaSkills.unlocked.includes(d.alias))) {
            if(!x.unlocked.includes(id))x.unlocked.push(id);
            x.highRanks[id]=Math.min(d.core?a.cap:10,d.alias?a.highRanks[d.alias]||state.areaSkills.highRanks[d.alias]||0:0);
          }
        }
      }
    }
    x.ranks=map(C.SKILLS,0);x.areaRanks=map(C.AREA_UPGRADES,0);x.phase=map(C.STATIONS,0);x.batchWork=map(C.STATIONS,0);x.batchCredit=map(C.STATIONS,()=>N.zero());x.boost={areaId:null,remaining:0};x.revision+=1;sync(state);
    const first=C.STATIONS[0].skillIds[0], count=Math.max(...P.batchModes(state).filter(m=>m.unlocked).map(m=>m.count));
    state.resources.coins=quote(state,first,count).costs.coins||N.from(24);
    return {starter:state.resources.coins,quantity:count};
  }
  const canRelease=(state,areaId)=>{const third=C.STATIONS.filter(st=>st.areaId===areaId)[2];return state.stations.built.includes(third.id)&&rank(state,third.skillIds[0])>0;};
  const voyageArrived=(state,voyage)=>{if(active(state))state.areaSkills.output.harbor+=voyage.convoy||1;};
  function autoBuy(state,reserve) {
    if(!active(state))return;
    for(let pass=0;pass<100;pass+=1) {
      const options=C.SKILLS.filter(d=>state.stations.unlocked.includes(d.id)&&state.stations.highRanks[d.id]>0).map(d=>({d,q:quote(state,d.id,1)})).filter(o=>o.q.valid&&Object.entries(o.q.costs).every(([id,n])=>N.cmp(state.resources[id],N.add(n,reserve(state,id)))>=0));
      options.sort((a,b)=>rank(state,a.d.id)-rank(state,b.d.id)||N.cmp(a.q.costs.coins,b.q.costs.coins));
      if(!options.length)break;
      act(state,{type:'station-skill-buy',id:options[0].d.id,count:1,quote:options[0].q.token});
    }
  }
  return {Content:C,initial,validate,active,configure,setSkillsView,sync,view,legacyView,row,requirements,eligible,quote,cost,stationEconomy,enrichRates,nextEvent,tick,act,reset,canRelease,voyageArrived,autoBuy,choices,configurations,validChoice};
});
