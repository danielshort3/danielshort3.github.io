(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./progression-content.js') : root.WayfarersProgressionContent, common ? require('./progression-modifiers.js') : root.WayfarersProgressionModifiers);
  if (common) module.exports = api;
  if (root) root.WayfarersProgression = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, D, M) {
  'use strict';
  const EPS = 1e-8, RESOURCES = ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps'];
  const ids = D.AREAS.map(a => a.id), clone = x => JSON.parse(JSON.stringify(x));
  const object = x => !!x && typeof x === 'object' && !Array.isArray(x);
  const exact = (x, keys) => object(x) && Object.keys(x).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(x, k));
  const finite = (x, max = 1e15, integer = false) => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= max && (!integer || Number.isSafeInteger(x));
  const areaDef = id => D.AREAS.find(a => a.id === id);
  const trackDef = (area, id) => areaDef(area)?.tracks.find(t => t.id === id);
  const projectDef = id => D.PROJECTS.find(p => p.id === id);
  const active = state => state.expedition?.version === 3;
  const learned = (x, id, track) => x.areas[id]?.learned.includes(track);
  const built = (x, id) => x.projects.includes(id);
  const rank = (x, id, track) => x.areas[id]?.ranks[track] || 0;
  const powers = [1];
  const milestoneBonus = { 3: 1.08, 10: 1.06, 50: 1.04, 100: 1.04, 250: 1.03, 1000: 1.02 };
  function power(n) { while (powers.length <= n) { const r = powers.length - 1; powers.push(powers[r] * (1 + .38 / Math.pow(1 + r / 3, 1.1)) * (milestoneBonus[r + 1] || 1)); } return powers[n]; }
  let reserveProvider = (state, resource = 'coins') => state.guild.plan.reserves[resource];
  const reserve = (state, resource = 'coins') => reserveProvider(state, resource);
  let rateProvider = null;
  let workRateProvider = null;
  const ownership = new WeakMap();
  const setEntitlements = (state, values) => ownership.set(state, new Set(values));
  const setReserveProvider = fn => { reserveProvider = fn; };
  const setRateProvider = fn => { rateProvider = fn; };
  const setWorkRateProvider = fn => { workRateProvider = fn; };
  function detached(state) { const copy = Object.assign({}, state, { expedition: clone(state.expedition) }); if (ownership.has(state)) ownership.set(copy, ownership.get(state)); return copy; }
  const empty = () => Object.fromEntries(RESOURCES.map(id => [id, 0]));
  function makeArea(id) {
    const tracks = areaDef(id).tracks;
    return { ranks: Object.fromEntries(tracks.map(t => [t.id, 0])), highRanks: Object.fromEntries(tracks.map(t => [t.id, 0])), learned: [tracks[0].id], cap: 100,
      choice: id === 'greenway' ? 'trade' : id === 'quarry' ? 'balanced' : id === 'watchtower' ? 'survey' : id === 'workshop' ? 'tools' : id === 'ruins' ? 'industry' : 'trade',
      specialization: null, plans: { assignments: ['survey'], templates: ['supplies'], target: 'near', discovery: 'botanical', loadouts: ['industry'], port: 'coast' }, voyages: [], discoveryWork: 0, discoveries: { botanical: 0, metallic: 0, inscribed: 0 }, buffers: { input: 0, output: 0 }, elapsed: 0, purchases: 0, seen: 0 };
  }
  function create() {
    return { version: 3, index: 0, cleared: -1, completed: false, renewed: false, selectedArea: 'greenway', projectArea: 'greenway', work: 0, finaleWork: 0,
      areas: { greenway: makeArea('greenway') }, projects: [], commission: null, batch: 1, revision: 0,
      focus: { charges: 0, recharge: 0, active: null, remaining: 0, unlocked: false },
      automation: { enabled: false, priority: 'balanced', dispatch: false, clock: 0 },
      sequence: 0, seen: 0, recent: [], purchases: 0, blueprints: [], mastery: { greenway: 0, quarry: 0, watchtower: 0 }, legacyDevelopments: [] };
  }
  function event(x, kind, title, text, areaId) { x.sequence += 1; x.recent.push({ sequence: x.sequence, kind, title, text, stage: x.index, areaId }); x.recent = x.recent.slice(-12); }
  function batchModes(state) {
    let max = 1;
    for (const b of D.BATCHES) if (state.lifetime.refits >= b.refits && state.lifetime.charters >= b.charters) max = Math.max(max, b.count);
    return D.BATCHES.map(b => ({ count: b.count, unlocked: b.count <= max, requirement: b.count === 1 ? 'Available from the beginning' : (b.refits ? b.refits + ' lifetime Refit' + (b.refits === 1 ? '' : 's') : '') + (b.refits && b.charters ? ' and ' : '') + (b.charters ? '1 Guild Charter' : ''), action: { type: 'expedition-batch', count: b.count } }));
  }
  function cost(state, areaId, trackId, atRank) {
    const d = trackDef(areaId, trackId), r = atRank ?? rank(state.expedition, areaId, trackId);
    const price = Math.ceil(d.base * Math.pow(1 + r / 4, 2.6));
    const rebuilding = state.expedition.renewed && r < state.expedition.areas[areaId].highRanks[trackId];
    const discount = rebuilding ? .5 : 1;
    const result = { coins: N.from(Math.ceil(price * discount)) };
    if (r >= 25 && areaId !== 'greenway') result[areaId === 'workshop' ? 'ore' : areaId === 'ruins' ? 'knowledge' : areaId === 'harbor' ? 'provisions' : 'maps'] = N.from(Math.ceil(Math.ceil(price * .015) * discount));
    return result;
  }
  function quote(state, areaId, trackId, count) {
    const x = state.expedition, a = x.areas[areaId], d = trackDef(areaId, trackId);
    const result = { count, rank: a?.ranks[trackId] || 0, rankAfter: (a?.ranks[trackId] || 0) + count, costs: {}, valid: false, affordable: false, reason: '' };
    if (!d || !a || !learned(x, areaId, trackId)) { result.reason = 'Learn this track first.'; return result; }
    if (!batchModes(state).some(b => b.count === count && b.unlocked)) { result.reason = 'Earn this purchase quantity first.'; return result; }
    if (result.rankAfter > a.cap) { result.reason = 'Only ' + (a.cap - result.rank) + ' ranks remain. Choose a smaller batch.'; return result; }
    for (let i = 0; i < count; i += 1) Object.entries(cost(state, areaId, trackId, result.rank + i)).forEach(([key, value]) => { result.costs[key] = N.add(result.costs[key] || 0, value); });
    result.valid = true;
    result.affordable = Object.entries(result.costs).every(([key, value]) => N.cmp(state.resources[key], value) >= 0);
    if (!result.affordable) result.reason = 'Save for the entire ×' + count + ' batch.';
    result.token = [state.run.id, x.revision, areaId, trackId, result.rank, count, a.cap].join(':');
    return result;
  }
  function targets(x) {
    const first = [180, 650, 10000];
    const scale = x.index < 3 ? first[x.index] * (x.renewed ? .32 : 1) : 16000 * Math.pow(1.35, Math.min(60, x.index - 2));
    return { work: scale * .85, finale: scale * .15 };
  }
  function progress(x) { const t = targets(x); return x.completed ? 1 : Math.min(1, (x.work + x.finaleWork) / (t.work + t.finale)); }
  function stageName(x) { return areaDef(x.projectArea).name + (x.index < 3 ? ' establishment' : ' expansion ' + Math.max(1, x.index - 2)); }
  function localPower(x, area, track) {
    const a = x.areas[area], boost = a?.specialization ? a.specialization === track ? 1.35 : .9 : 1;
    return 1 + (power(rank(x, area, track)) - 1) * boost;
  }
  function rawRates(state, options = {}) {
    const x = state.expedition, gain = empty(), drain = empty(), areas = {};
    const p = (id, track) => localPower(x, id, track), has = id => !!x.areas[id];
    const meta = (1 + Math.sqrt(state.lifetime.refits) * .28) * (1 + state.lifetime.charters * .65);
    const travelBonus = (1 + .2 * Math.sqrt(state.refitUpgrades.pace)) * (1 + .35 * Math.sqrt(state.legacy.waystones));
    const materialBonus = 1 + .25 * Math.sqrt(state.refitUpgrades.supply);
    const researchBonus = (1 + .25 * Math.sqrt(state.refitUpgrades.insight)) * (1 + .35 * Math.sqrt(state.legacy.curriculum));
    const foundation = 1 + .3 * Math.sqrt(state.legacy.foundations);
    const global = meta;
    const inherited = id => x.legacyDevelopments.includes(id);
    const focus = id => !options.unboosted && x.focus.active === id && x.focus.remaining > EPS ? D.FOCUS.multiplier : 1;
    const tools = state.challenges.active === 'old-tools' ? 1 : 1 + .12 * Math.sqrt(state.upgrades['gear-tools']);
    const boots = state.challenges.active === 'old-tools' ? 1 : 1 + .12 * Math.sqrt(state.upgrades['gear-boots']);
    const coord = has('watchtower') ? .1 * Math.max(0, p('watchtower', 'signals') - 1) * p('watchtower', 'crew') * (1 + .15 * Math.max(0, p('watchtower', 'relay-grid') - 1)) : 0;
    const machinery = has('workshop') ? .12 * (p('workshop', 'mechanisms') - 1) : 0;
    const resonance = has('ruins') && new Set(x.areas.ruins.plans.loadouts).size > 1 ? 1 + .12 * (p('ruins', 'resonance') - 1) : 1;
    const restoration = has('ruins') ? .1 * (p('ruins', 'restoration') - 1) * p('ruins', 'attunement') * resonance : 0;
    const assignmentSlots = has('watchtower') ? 1 + (rank(x, 'watchtower', 'crew') >= 10 ? 1 : 0) + (learned(x, 'watchtower', 'relay-grid') && rank(x, 'watchtower', 'relay-grid') > 0 ? 1 : 0) : 0;
    const assignments = has('watchtower') ? x.areas.watchtower.plans.assignments.slice(0, assignmentSlots) : [];
    const allocation = name => assignments.filter(v => v === name).length / Math.max(1, assignments.length);
    const relicSlots = learned(x, 'ruins', 'resonance') && rank(x, 'ruins', 'resonance') > 0 ? 2 : 1;
    const relicRoles = has('ruins') ? x.areas.ruins.plans.loadouts.slice(0, relicSlots) : [];
    const ruinTrade = relicRoles.includes('trade') && x.areas.ruins.discoveries.botanical > 0;
    const ruinIndustry = relicRoles.includes('industry') && x.areas.ruins.discoveries.metallic > 0;
    const g = x.areas.greenway, dispatch = g.choice;
    const speed = p('greenway', 'boots') * boots * (1 + .08 * Math.sqrt(state.upgrades.preparation)) * (1 + .08 * Math.sqrt(state.ranks.trail)) * (inherited('paved-roads') ? 1.25 : 1) * (x.blueprints.includes('pathfinding') ? 1.15 : 1);
    const parallel = learned(x, 'greenway', 'caravans') ? Math.max(0, p('greenway', 'caravans') - 1) * .35 : 0;
    const share = dispatch === 'freight' ? .55 : dispatch === 'survey' ? .4 : ['mixed', 'trade-survey'].includes(dispatch) ? .75 : dispatch === 'continental' ? .7 : 1;
    const freight = (dispatch === 'freight' ? .6 : dispatch === 'mixed' ? .35 : 0) * p('greenway', 'porters') + parallel + (learned(x, 'greenway', 'railways') ? .3 * p('greenway', 'railways') : 0);
    gain.coins = Math.sqrt(speed) * p('greenway', 'porters') * (1 + parallel) * share * (1 + coord * (.25 + allocation('trade'))) * (1 + (ruinTrade ? restoration : 0));
    gain.maps = .008 * p('greenway', 'scouts') * (dispatch === 'survey' ? 3 : dispatch === 'trade-survey' ? 1.8 : 1);
    if (inherited('survey-charters') && ['survey', 'trade-survey'].includes(dispatch)) gain.knowledge += gain.maps * .75;
    gain.herbs = learned(x, 'greenway', 'waystations') ? .03 * p('greenway', 'waystations') : 0;
    areas.greenway = { work: speed, finale: speed * .8 + p('greenway', 'scouts') * .3, income: gain.coins, travel: speed, freight, research: .015 * p('greenway', 'scouts'), capacity: 10 * p('greenway', 'porters'), flow: speed };
    if (has('quarry')) {
      const a = x.areas.quarry;
      const plan = a.choice === 'adaptive' ? a.buffers.input > EPS || a.buffers.output > EPS ? 'balanced' : 'precision' : a.choice;
      const extraction = .12 * p('quarry', 'picks') * tools * (inherited('deep-veins') ? 1.2 : 1) * (inherited('trail-prospectors') && ['survey', 'trade-survey'].includes(dispatch) ? 1 + .08 * Math.sqrt(rank(x, 'greenway', 'scouts')) : 1) * (1 + (has('workshop') ? .1 * Math.max(0, p('workshop', 'toolmaking') - 1) : 0)) * (1 + (x.areas.workshop?.choice === 'extraction' ? machinery : 0)) * focus('quarry');
      const haul = .1 * p('quarry', 'carts') * (1 + freight) * (1 + coord * (.2 + allocation('industry'))) * focus('quarry');
      const refining = .09 * p('quarry', 'furnace') * (state.research.includes('efficient-smelting') ? 1.25 : 1) * focus('quarry');
      const capacity = 10 + rank(x, 'quarry', 'carts') * .5 + (inherited('trail-depot') ? 24 : 0) + (learned(x, 'greenway', 'waystations') ? p('greenway', 'waystations') * 3 : 0);
      const rich = plan === 'rich', alloys = plan === 'alloy';
      const geologic = 1 + .12 * (p('quarry', 'geology') - 1), deep = learned(x, 'quarry', 'deepworks') ? .15 * p('quarry', 'deepworks') : 0;
      const mineCap = extraction * (rich ? .7 : 1) + extraction * deep;
      const cartCap = haul, furnaceCap = refining * (alloys ? .7 : plan === 'precision' ? .6 : plan === 'mixed' ? .8 : 1);
      let cartFlow = a.buffers.input > EPS ? cartCap : Math.min(cartCap, mineCap);
      const furnaceFlow = a.buffers.output > EPS ? furnaceCap : Math.min(furnaceCap, cartFlow);
      if (a.buffers.output >= capacity - EPS) cartFlow = Math.min(cartFlow, furnaceFlow);
      const mineFlow = a.buffers.input >= capacity - EPS ? Math.min(mineCap, cartFlow) : mineCap;
      const yieldRate = (rich ? 1.65 : alloys ? .8 : plan === 'precision' ? 1.4 : plan === 'mixed' ? 1.15 : plan === 'optics' ? .3 : 1) * geologic * (inherited('efficient-crucibles') ? 1.15 : 1) * (1 + .2 * (p('quarry', 'recovery') - 1)) * (1 + (ruinIndustry ? restoration : 0));
      gain.ore = furnaceFlow * yieldRate;
      if (built(x, 'guild-industry') && x.areas.workshop?.choice === 'integrated') gain.ore += Math.min(mineFlow - Math.min(mineFlow, furnaceFlow), furnaceFlow) * .4;
      gain.coins += furnaceFlow * .5;
      if (plan === 'optics') gain.knowledge += furnaceFlow * .35;
      if (inherited('recovery-chutes') && a.buffers.input >= capacity - EPS) gain.coins += Math.max(0, mineCap - cartFlow) * .15;
      areas.quarry = { work: furnaceFlow * 10, finale: furnaceFlow * 9, picks: mineCap, carts: cartCap, furnace: furnaceCap, actualPicks: mineFlow, actualCarts: cartFlow, actualFurnace: furnaceFlow, capacity, input: mineFlow - cartFlow, output: cartFlow - furnaceFlow, materials: gain.ore, alloy: alloys ? .8 * geologic : built(x, 'industrial-supports') ? .3 * geologic : .1, bottleneck: mineCap <= cartCap && mineCap <= furnaceCap ? 'picks' : cartCap <= furnaceCap ? 'carts' : 'furnace', flow: furnaceFlow };
    }
    if (has('watchtower')) {
      const surveys = p('watchtower', 'beacon'), optics = 1 + .25 * (p('watchtower', 'optics') - 1), relay = 1 + .15 * (p('watchtower', 'relay-grid') - 1);
      const target = x.areas.watchtower.plans.target;
      const research = surveys * optics * (.65 + allocation('survey') * .35) * (target === 'deep' ? 1.5 : target === 'ocean' ? .75 : 1);
      gain.knowledge += .025 * research * relay;
      gain.maps += .016 * surveys * p('watchtower', 'signals') * (1 + .1 * (p('watchtower', 'crew') - 1)) * (target === 'deep' ? .55 : target === 'ocean' ? 1.8 : 1);
      if (built(x, 'guild-discovery') && assignments.includes('discovery')) { gain.maps *= 1.2; gain.knowledge *= 1.2; }
      areas.watchtower = { work: research * 1.6, finale: research, research, knowledge: .025 * research * relay, capacity: assignmentSlots, assignments: assignments.slice(), target, coordination: coord * relay, workers: { repair: 2 + rank(x, 'watchtower', 'crew'), protection: 1, total: 3 + rank(x, 'watchtower', 'crew') }, repair: research * 1.6, beacon: research, flow: research };
    }
    if (has('workshop')) {
      const a = x.areas.workshop, efficient = a.choice === 'precision', slots = learned(x, 'workshop', 'replication') && rank(x, 'workshop', 'replication') > 0 ? 2 : 1;
      const cap = .09 * p('workshop', 'assembly') * (1 + (a.choice === 'manufacture' || a.choice === 'integrated' ? machinery : 0)) * focus('workshop');
      const standardization = slots > 1 ? 1 + .12 * (p('workshop', 'replication') - 1) : 1;
      const inputPerUnit = (efficient ? .45 / Math.sqrt(p('workshop', 'precision')) : .8) / standardization;
      const relation = N.cmp(state.resources.ore, reserve(state, 'ore'));
      const available = relation > 0 ? Infinity : relation === 0 ? gain.ore : 0;
      const flow = Math.min(cap * (efficient ? .65 : 1), available / inputPerUnit);
      drain.ore = flow * inputPerUnit;
      const templates = a.plans.templates.slice(0, slots), lanes = [];
      for (const recipe of templates) {
        const lane = flow / Math.max(1, templates.length), output = lane * (1 + (areas.quarry?.alloy || 0)) * p('workshop', 'metallurgy');
        if (recipe === 'supplies') gain.provisions += output;
        if (recipe === 'tools') { gain.ore += output * .55; gain.provisions += output * .25; }
        if (recipe === 'instruments') gain.knowledge += output * .75 * Math.sqrt(p('workshop', 'toolmaking'));
        lanes.push({ recipe, flow: lane, input: lane * inputPerUnit, output });
      }
      if (built(x, 'guild-industry') && a.choice === 'integrated') gain.provisions += flow * .35;
      areas.workshop = { work: flow * 14, finale: flow * 12, assembly: cap, flow, demand: drain.ore, materials: gain.provisions, capacity: slots, templates: lanes, input: 0, output: 0, research: .05 * p('workshop', 'toolmaking') * (1 + .2 * p('workshop', 'precision')) };
    }
    if (has('ruins')) {
      const a = x.areas.ruins, cap = 10 + rank(x, 'ruins', 'recovery-teams') * .4;
      const delving = .07 * p('ruins', 'delving') * focus('ruins'), interpretation = .06 * p('ruins', 'archaeology') * focus('ruins'), recovery = .05 * p('ruins', 'recovery-teams') * (a.choice === 'survey' ? .75 : 1) * focus('ruins');
      let read = a.buffers.input > EPS ? interpretation : Math.min(interpretation, delving);
      const recover = a.buffers.output > EPS ? recovery : Math.min(recovery, read);
      if (a.buffers.output >= cap - EPS) read = Math.min(read, recover);
      const delve = a.buffers.input >= cap - EPS ? Math.min(delving, read) : delving;
      const discovery = a.plans.discovery;
      if (discovery === 'botanical') gain.herbs += recover * 2 * (a.choice === 'trade' ? 1.35 : 1);
      if (discovery === 'metallic') gain.ore += recover * 1.5 * (a.choice === 'industry' ? 1.35 : 1);
      if (discovery === 'inscribed') gain.knowledge += recover * 1.2;
      if (built(x, 'deepwater-equipment')) gain.herbs += Math.min(delving * .3, recovery * .5);
      gain.knowledge += recover * .5 * p('ruins', 'restoration');
      const insight = (a.choice === 'survey' ? 2 : 1) * (relicRoles.includes('survey') && a.discoveries.inscribed ? 1 + restoration : 1) * (built(x, 'guild-discovery') && assignments.includes('discovery') ? 1.4 : 1);
      areas.ruins = { work: recover * 20, finale: read * 18, delving, interpretation, recovery, flow: recover, capacity: cap, input: delve - read, output: read - recover, research: read * insight, artifacts: restoration, discovery, discoveryWork: a.discoveryWork, loadouts: relicRoles.slice(), discoveries: clone(a.discoveries) };
    }
    if (has('harbor')) {
      const a = x.areas.harbor, forecast = 1 + .15 * (p('watchtower', 'forecasting') - 1), distant = a.choice === 'discovery';
      const ship = p('harbor', 'shipbuilding'), sailing = p('harbor', 'seamanship') * forecast, cargo = p('harbor', 'stowage');
      const fleet = learned(x, 'harbor', 'fleet-command') ? 1 + .12 * (p('harbor', 'fleet-command') - 1) : 1;
      const supply = 3 * ship * (1 + .25 * (cargo - 1)), available = N.cmp(state.resources.provisions, N.add(supply, reserve(state, 'provisions'))) >= 0;
      const fraction = a.voyages.length ? 1 : 0;
      const navigation = 1 + .2 * (p('harbor', 'navigation') - 1);
      const contracts = 1 + .2 * (p('harbor', 'contracts') - 1), trailSupply = 1 + .1 * freight + (dispatch === 'continental' ? .5 * p('greenway', 'porters') : 0);
      const flow = ship * Math.sqrt(sailing) * cargo * fraction * trailSupply * fleet;
      const weather = Math.floor(a.elapsed / 10800) % 3;
      const port = a.plans.port, condition = weather === 2 && port === 'ocean' ? .65 : weather === 1 && port === 'coast' ? .85 : 1;
      const travel = sailing * navigation * condition;
      const far = port === 'ocean' ? 2.5 : port === 'ruins' ? 1.7 : 1;
      const payout = { coins: 30 * ship * cargo * trailSupply * contracts * (distant ? .3 : 1) * far * fleet, maps: ship * cargo * navigation * (distant ? 3 : .5) * far * fleet };
      if (a.choice === 'materials') { payout.coins *= .55; payout.maps *= .75; payout.ore = 8 * ship * cargo * contracts; }
      if (a.choice === 'commerce') { payout.provisions = supply * .5; payout.maps *= 1.5; }
      const slots = learned(x, 'harbor', 'fleet-command') && rank(x, 'harbor', 'fleet-command') > 0 ? 2 : 1;
      areas.harbor = { work: Math.max(.2, flow), finale: Math.max(.2, flow), flow, capacity: slots, cargo, travel, voyagePace: sailing * navigation, convoy: 1, duration: 900 * far / travel, voyageTarget: 900 * far, supply, canLaunch: available, payout, demand: 0, materials: 0, weather, port, research: distant ? flow * .1 : flow * .02 };
    }
    // Retained guild equipment/professions remain auxiliary investments in the
    // actual network, never an independent exponential stream bypassing it.
    gain.coins *= 1 + .08 * Math.sqrt(state.upgrades.boots);
    gain.ore *= 1 + .08 * Math.sqrt(state.upgrades.miners);
    gain.herbs *= 1 + .08 * Math.sqrt(state.upgrades.foragers);
    gain.knowledge *= 1 + .08 * Math.sqrt(state.upgrades.scholars + state.upgrades['gear-instruments']);
    gain.maps *= 1 + .08 * Math.sqrt(state.upgrades.surveyors + state.upgrades['gear-instruments']);
    gain.provisions *= 1 + .08 * Math.sqrt(state.upgrades.cooks);
    gain.coins *= foundation;
    gain.ore *= materialBonus * foundation;
    gain.herbs *= materialBonus * foundation;
    gain.provisions *= foundation;
    gain.knowledge *= researchBonus;
    gain.maps *= researchBonus;
    const collections = Math.pow(1.08, state.collections.length);
    RESOURCES.forEach(key => { gain[key] *= collections; });
    if (x.blueprints.includes('caravan')) gain.coins *= 1.2;
    if (x.blueprints.includes('engineering')) { gain.ore *= 1.15; gain.provisions *= 1.15; }
    if (areas.harbor) {
      const payout = areas.harbor.payout;
      payout.coins *= foundation * collections * (1 + .08 * Math.sqrt(state.upgrades.boots)) * (x.blueprints.includes('caravan') ? 1.2 : 1);
      payout.maps *= researchBonus * collections * (1 + .08 * Math.sqrt(state.upgrades.surveyors + state.upgrades['gear-instruments']));
      if (payout.ore) payout.ore *= materialBonus * foundation * collections * (1 + .08 * Math.sqrt(state.upgrades.miners));
      if (payout.provisions) payout.provisions *= foundation * collections * (1 + .08 * Math.sqrt(state.upgrades.cooks));
    }
    for (const id of Object.keys(areas)) {
      const mult = global * (['greenway', 'watchtower'].includes(id) ? focus(id) : 1), r = areas[id];
      for (const key of ['work', 'finale', 'research']) if (r[key]) r[key] *= mult;
      if (id in x.mastery) { const retained = 1 + .02 * Math.min(25, x.mastery[id]); r.work *= retained; r.finale *= retained; }
      if (id === 'greenway' || id === 'harbor') { r.work *= travelBonus; r.finale *= travelBonus; r.travel *= global * travelBonus; if (id === 'harbor') { r.voyagePace *= global * travelBonus; r.duration = r.voyageTarget / r.travel; } }
      if (r.research) r.research *= researchBonus;
      if (id === 'quarry') { r.work *= materialBonus * foundation; r.finale *= materialBonus * foundation; }
      // Fast fleets consolidate departures into a convoy. This preserves the
      // earned throughput and full supply bill without subsecond offline events.
      if (id === 'harbor' && r.duration < 60) {
        const convoy = 60 / r.duration;
        r.supply *= convoy;
        Object.keys(r.payout).forEach(key => { r.payout[key] *= convoy; });
        r.travel = r.voyageTarget / 60; r.duration = 60; r.convoy = convoy;
        r.canLaunch = N.cmp(state.resources.provisions, N.add(r.supply, reserve(state, 'provisions'))) >= 0;
      }
      if (id === 'harbor') { r.travel *= focus(id); r.voyagePace *= focus(id); r.duration /= focus(id); }
    }
    RESOURCES.forEach(id => { gain[id] *= global; drain[id] *= global; });
    // Focus is shared, finite and affects the selected area's output only.
    if (!options.unboosted && x.focus.remaining > EPS && x.focus.active === 'greenway') gain.coins += areas.greenway.income * global * foundation * collections * (1 + .08 * Math.sqrt(state.upgrades.boots)) * (x.blueprints.includes('caravan') ? 1.2 : 1) * (D.FOCUS.multiplier - 1);
    if (!options.unboosted && x.focus.remaining > EPS && x.focus.active === 'watchtower') gain.knowledge += areas.watchtower.knowledge * global * researchBonus * collections * (1 + .08 * Math.sqrt(state.upgrades.scholars + state.upgrades['gear-instruments'])) * (D.FOCUS.multiplier - 1);
    const researchRate = Object.values(areas).reduce((sum, a) => sum + (a.research || 0), 0) * .6;
    const raw = { gain, drain, areas, researchRate, global };
    return !options.baseOnly && workRateProvider ? M.apply(state, raw, ownership.get(state), { rates: workRateProvider(state, ownership.get(state)) }) : raw;
  }
  function contribution() { return { coins: N.zero(), ore: N.zero(), knowledge: N.zero(), maps: N.zero(), production: 1, travel: 1 }; }
  function localRates(state, id) { return rawRates(state).areas[id || state.expedition.selectedArea]; }
  function nextEvent(state, gains) {
    const x = state.expedition, r = rawRates(state), t = targets(x), a = r.areas[x.projectArea];
    let next = x.completed ? Infinity : x.work < t.work - EPS ? (t.work - x.work) / a.work : (t.finale - x.finaleWork) / a.finale;
    if (x.commission) next = Math.min(next, Math.max(EPS, (projectDef(x.commission.id).work - x.commission.work) / r.researchRate));
    if (x.automation.enabled) next = Math.min(next, Math.max(EPS, 60 - x.automation.clock));
    if (x.focus.unlocked && x.focus.charges < D.FOCUS.capacity) next = Math.min(next, Math.max(EPS, D.FOCUS.recharge - x.focus.recharge));
    if (x.focus.remaining > EPS) next = Math.min(next, x.focus.remaining);
    for (const [id, area] of Object.entries(x.areas)) for (let i = 1; i < 3; i += 1) if (!area.learned.includes(areaDef(id).tracks[i].id)) next = Math.min(next, Math.max(EPS, i * 180 - area.elapsed));
    for (const id of ['quarry', 'ruins']) if (x.areas[id]) for (const key of ['input', 'output']) {
      const flow = r.areas[id][key], amount = x.areas[id].buffers[key];
      if (flow > EPS) next = Math.min(next, Math.max(EPS, (r.areas[id].capacity - amount) / flow));
      if (flow < -EPS) next = Math.min(next, Math.max(EPS, -amount / flow));
    }
    if (x.areas.ruins && !x.areas.ruins.discoveries[x.areas.ruins.plans.discovery] && r.areas.ruins.flow > EPS) next = Math.min(next, Math.max(EPS, (5 - x.areas.ruins.discoveryWork) / r.areas.ruins.flow));
    if (x.areas.harbor) {
      const area = x.areas.harbor, harbor = r.areas.harbor;
      for (const voyage of area.voyages) next = Math.min(next, Math.max(EPS, (voyage.target - voyage.work) / voyageRate(harbor, voyage)));
      next = Math.min(next, Math.max(EPS, 10800 - area.elapsed % 10800));
      if (area.voyages.length < harbor.capacity && !harbor.canLaunch && gains?.provisions && N.cmp(gains.provisions, 0) > 0) {
        const needed = N.sub(N.add(harbor.supply, reserve(state, 'provisions')), state.resources.provisions);
        next = Math.min(next, Math.max(EPS, N.toNumber(N.div(needed, gains.provisions))));
      }
    }
    return next;
  }
  function learnIntro(x, id) {
    const a = x.areas[id], defs = areaDef(id).tracks;
    for (let i = 1; i < 3; i += 1) if (!a.learned.includes(defs[i].id) && (a.ranks[defs[i - 1].id] >= 2 || a.elapsed >= i * 180)) {
      a.learned.push(defs[i].id); event(x, 'development', defs[i].name + ' learned', defs[i].effect + '. This knowledge survives every reset.', id); x.revision += 1;
    }
  }
  function completeProject(x) {
    if (!x.commission) return;
    const d = projectDef(x.commission.id);
    if (x.commission.work < d.work - EPS) return;
    x.projects.push(d.id); x.commission = null; x.revision += 1;
    if (d.unlock.area && !x.areas[d.unlock.area]) x.areas[d.unlock.area] = makeArea(d.unlock.area);
    const a = x.areas[d.target];
    if (d.unlock.track && !a.learned.includes(d.unlock.track)) a.learned.push(d.unlock.track);
    if (d.unlock.cap) a.cap = Math.max(a.cap, d.unlock.cap);
    event(x, 'development', d.name + ' completed', d.effect, d.target);
  }
  function tick(state, seconds) {
    const x = state.expedition, r = rawRates(state), t = targets(x);
    for (const [id, a] of Object.entries(x.areas)) {
      a.elapsed += seconds;
      for (const key of ['input', 'output']) {
        const flow = r.areas[id][key] || 0;
        a.buffers[key] = Math.max(0, Math.min(r.areas[id].capacity, a.buffers[key] + flow * seconds));
        if (a.buffers[key] < EPS) a.buffers[key] = 0;
        if (r.areas[id].capacity - a.buffers[key] < EPS) a.buffers[key] = r.areas[id].capacity;
      }
      learnIntro(x, id);
    }
    if (x.areas.ruins) {
      const area = x.areas.ruins; area.discoveryWork += r.areas.ruins.flow * seconds;
      const completed = Math.floor((area.discoveryWork + EPS) / 5);
      if (completed) { area.discoveryWork = Math.max(0, area.discoveryWork - completed * 5); area.discoveries[area.plans.discovery] += completed; }
    }
    if (x.areas.harbor) {
      const area = x.areas.harbor, remaining = [];
      for (const voyage of area.voyages) {
        voyage.work = Math.min(voyage.target, voyage.work + voyageRate(r.areas.harbor, voyage) * seconds);
        if (voyage.work >= voyage.target - EPS) {
          Object.entries(voyage.payout).forEach(([key, value]) => { state.resources[key] = N.add(state.resources[key], value); if (key === 'coins') state.lifetime.coins = N.add(state.lifetime.coins, value); });
          event(x, 'stage', 'Voyage delivered', voyage.port + ' cargo arrived automatically. Supplies were paid when the ship departed.', 'harbor');
        } else remaining.push(voyage);
      }
      area.voyages = remaining;
    }
    if (!x.completed) {
      const rates = r.areas[x.projectArea];
      if (x.work < t.work - EPS) x.work = Math.min(t.work, x.work + rates.work * seconds);
      else x.finaleWork = Math.min(t.finale, x.finaleWork + rates.finale * seconds);
    }
    if (x.commission) { x.commission.work = Math.min(projectDef(x.commission.id).work, x.commission.work + r.researchRate * seconds); completeProject(x); }
    x.automation.clock = Math.min(60, x.automation.clock + seconds);
    if (x.focus.unlocked && x.focus.charges < D.FOCUS.capacity) {
      x.focus.recharge += seconds;
      while (x.focus.recharge >= D.FOCUS.recharge - EPS && x.focus.charges < D.FOCUS.capacity) { x.focus.recharge = Math.max(0, x.focus.recharge - D.FOCUS.recharge); x.focus.charges += 1; }
      if (x.focus.charges === D.FOCUS.capacity) x.focus.recharge = 0;
    }
    x.focus.remaining = Math.max(0, x.focus.remaining - seconds);
    if (x.focus.remaining <= EPS) { if (x.focus.active !== null) x.revision += 1; x.focus.remaining = 0; x.focus.active = null; }
  }
  function finish(state) {
    const x = state.expedition, t = targets(x);
    if (x.completed || x.work < t.work - EPS || x.finaleWork < t.finale - EPS) return false;
    x.work = t.work; x.finaleWork = t.finale; x.completed = true; x.cleared = Math.max(x.cleared, x.index);
    if (x.index === 0 && !x.areas.quarry) x.areas.quarry = makeArea('quarry');
    if (x.index === 1 && !x.areas.watchtower) x.areas.watchtower = makeArea('watchtower');
    x.revision += 1; event(x, 'stage', stageName(x) + ' completed', 'Every discovered area keeps producing.', x.projectArea); return true;
  }
  function restart(state, index, select) {
    const x = state.expedition, opened = ids.filter(id => x.areas[id]);
    x.index = index; x.completed = false; x.work = 0; x.finaleWork = 0;
    x.projectArea = index < 3 ? ids[Math.min(index, opened.length - 1)] : opened[(index - 3) % opened.length];
    if (select !== false) x.selectedArea = x.projectArea;
    x.revision += 1;
  }
  function projectDependencies(state, d) {
    const x = state.expedition;
    return [{ label: 'Discover ' + areaDef(d.source).name, met: !!x.areas[d.source] }]
      .concat(d.unlock.area ? [] : [{ label: 'Discover ' + areaDef(d.target).name, met: !!x.areas[d.target] }])
      .concat(d.requires.map(id => ({ label: projectDef(id).name, met: built(x, id) })));
  }
  function developmentTask(state, id) {
    const d = projectDef(id); if (!d) return null;
    return { name: d.name, costs: Object.fromEntries(Object.entries(d.costs).map(([k, v]) => [k, N.from(v)])), open: !!state && projectDependencies(state, d).every(v => v.met) && !state.expedition.commission, done: !!state && (built(state.expedition, id) || state.expedition.commission?.id === id) };
  }
  function act(state, action) {
    const x = state.expedition, id = action.areaId || x.selectedArea, a = x.areas[id];
    if (action.type === 'expedition-select') { if (!a) return { ok: false, message: 'Discover that area first.' }; x.selectedArea = id; a.seen = x.sequence; return { ok: true, message: areaDef(id).name + ' selected.' }; }
    if (action.type === 'expedition-batch') { if (!batchModes(state).some(b => b.count === action.count && b.unlocked)) return { ok: false, message: 'Earn this quantity first.' }; x.batch = action.count; return { ok: true, message: 'Exact ×' + action.count + ' purchases selected.' }; }
    if (action.type === 'expedition-buy') {
      const q = quote(state, id, action.id, action.count === undefined ? 1 : action.count);
      if (!q.valid || !q.affordable) return { ok: false, message: q.reason };
      if (action.quote !== undefined && action.quote !== q.token) return { ok: false, message: 'This quote changed. Review the refreshed cost and effect.' };
      Object.entries(q.costs).forEach(([key, value]) => { state.resources[key] = N.sub(state.resources[key], value); });
      const before = a.ranks[action.id]; a.ranks[action.id] = q.rankAfter; a.purchases += q.count; x.purchases += q.count; x.revision += 1;
      for (const milestone of [3, 10, 25, 50, 100, 250, 1000]) if (before < milestone && q.rankAfter >= milestone && a.highRanks[action.id] < milestone) event(x, 'milestone', trackDef(id, action.id).name + ' ' + milestone, milestone === 25 ? 'A specialist plan is now learned permanently.' : '+' + Math.round(((milestoneBonus[milestone] || 1) - 1) * 100) + '% extra rank capacity at this milestone.', id);
      a.highRanks[action.id] = Math.max(a.highRanks[action.id], q.rankAfter); learnIntro(x, id);
      return { ok: true, message: trackDef(id, action.id).name + ' +' + q.count + ' · rank ' + q.rankAfter, quantity: q.count };
    }
    if (action.type === 'expedition-development') {
      const d = projectDef(action.id), task = developmentTask(state, action.id);
      if (!d || !task.open || task.done || !Object.entries(task.costs).every(([k, v]) => N.cmp(state.resources[k], v) >= 0)) return { ok: false, message: 'Finish the listed prerequisites and fund this project first.' };
      Object.entries(task.costs).forEach(([k, v]) => { state.resources[k] = N.sub(state.resources[k], v); });
      x.commission = { id: d.id, work: 0 }; x.revision += 1; return { ok: true, message: d.name + ' started. Research continues while you are away.' };
    }
    if (action.type === 'expedition-choice') {
      if (!a || !choices(state, id).some(c => c.id === action.id && !c.disabled)) return { ok: false, message: 'Learn that working plan first.' };
      a.choice = action.id;
      if (id === 'watchtower') a.plans.assignments[0] = action.id;
      if (id === 'ruins' && a.discoveries[{ industry: 'metallic', trade: 'botanical', survey: 'inscribed' }[action.id]]) a.plans.loadouts[0] = action.id;
      x.revision += 1; return { ok: true, message: areaDef(id).name + ' plan saved.' };
    }
    if (action.type === 'expedition-config') {
      const groups = configurations(state, id), group = groups.find(g => g.id === action.kind), option = group?.options.find(o => o.id === action.id && !o.disabled);
      if (!a || !option || !Number.isSafeInteger(action.slot) || action.slot < 0 || action.slot >= group.slots) return { ok: false, message: 'This configuration has not been learned.' };
      if (['assignments', 'templates', 'loadouts'].includes(action.kind)) { const values = a.plans[action.kind].slice(); values[action.slot] = action.id; a.plans[action.kind] = values; }
      else a.plans[action.kind] = action.id;
      x.revision += 1; return { ok: true, message: group.label + ' updated. Existing paid voyages retain their original manifest.' };
    }
    if (action.type === 'expedition-specialize') {
      if (!a || action.id !== null && (!trackDef(id, action.id) || a.highRanks[action.id] < 25)) return { ok: false, message: 'Reach rank 25 once to learn this specialist assignment.' };
      a.specialization = action.id; x.revision += 1; return { ok: true, message: 'Specialist assignment saved: +35% to its capacity, −10% to other tracks in this area.' };
    }
    if (action.type === 'expedition-automation') {
      if (!state.lifetime.refits || typeof action.enabled !== 'boolean' || !['balanced', 'progress', 'income', 'materials'].includes(action.priority) || action.dispatch !== undefined && typeof action.dispatch !== 'boolean') return { ok: false, message: 'Complete a Refit to learn standing plans.' };
      x.automation.enabled = action.enabled; x.automation.priority = action.priority; if (action.dispatch !== undefined) x.automation.dispatch = action.dispatch; x.revision += 1;
      return { ok: true, message: 'Standing plan saved. New branches and projects remain your decision.' };
    }
    if (action.type === 'expedition-focus') {
      if (!a || !x.focus.unlocked || !x.focus.charges || x.focus.remaining || action.id !== 'priority') return { ok: false, message: 'A shared Focus charge must be available.' };
      if (id === 'harbor' && !a.voyages.length || id === 'workshop' && !(rawRates(state).areas.workshop.flow > EPS)) return { ok: false, message: id === 'harbor' ? 'Fund a voyage before advancing its paid manifest.' : 'Supply the Workshop before concentrating its assembly work.' };
      x.focus.charges -= 1; x.focus.active = id; x.focus.remaining = D.FOCUS.duration; x.revision += 1;
      return { ok: true, message: areaDef(id).name + ' Focus started. Inputs and reserves still apply.' };
    }
    if (action.type === 'expedition-seen') { x.seen = x.sequence; return { ok: true, message: 'Discoveries acknowledged.' }; }
    if (action.type === 'expedition-blueprint') return { ok: false, message: 'Permanent development is now learned through the project catalog.' };
    return { ok: false, message: 'Unknown expedition action.' };
  }
  function autoBuy(state) {
    const x = state.expedition;
    launchVoyages(state);
    if (!x.automation.enabled || !state.lifetime.refits || x.automation.clock < 60 - EPS) return;
    x.automation.clock = 0;
    for (let pass = 0; pass < 100; pass += 1) {
      const r = rawRates(state), options = [];
      for (const [id, a] of Object.entries(x.areas)) for (const track of a.learned) if (a.highRanks[track] > 0 && a.ranks[track] < a.cap) {
        const q = quote(state, id, track, 1);
        if (!q.valid || !Object.entries(q.costs).every(([key, value]) => N.cmp(state.resources[key], N.add(value, reserve(state, key))) >= 0)) continue;
        let score = a.ranks[track];
        if (x.automation.priority === 'income') score += id === 'greenway' && track === 'porters' ? -5 : 0;
        if (x.automation.priority === 'materials') score += id === 'quarry' && track === r.areas.quarry?.bottleneck ? -5 : 0;
        if (x.automation.priority === 'progress') score += id === x.projectArea ? -4 : 0;
        options.push({ id, track, q, score });
      }
      options.sort((a, b) => a.score - b.score || N.cmp(a.q.costs.coins, b.q.costs.coins));
      if (!options.length) break;
      const o = options[0]; act(state, { type: 'expedition-buy', areaId: o.id, id: o.track, count: 1, quote: o.q.token });
    }
  }
  function launchVoyages(state) {
    const x = state.expedition, area = x.areas.harbor; if (!area) return;
    const raw = rawRates(state), r = raw.areas.harbor;
    for (let i = area.voyages.length; i < r.capacity; i += 1) {
      if (N.cmp(state.resources.provisions, N.add(r.supply, reserve(state, 'provisions'))) < 0) break;
      state.resources.provisions = N.sub(state.resources.provisions, r.supply);
      const payout = manifestPayout(state, raw);
      // The second fleet slot may carry discovery alongside a trade voyage.
      if (i === 1) { payout.coins = N.mul(payout.coins, .5); payout.maps = N.mul(payout.maps, 2); }
      area.voyages.push({ work: 0, target: r.voyageTarget, port: area.plans.port, supplies: r.supply, convoy: r.convoy, payout });
    }
  }
  function voyageRate(harbor, voyage) {
    const condition = harbor.weather === 2 && voyage.port === 'ocean' ? .65 : harbor.weather === 1 && voyage.port === 'coast' ? .85 : 1;
    return harbor.voyagePace * condition / voyage.convoy;
  }
  function manifestPayout(state, raw) {
    const canonical = workRateProvider ? workRateProvider(state, ownership.get(state)) : null;
    return Object.fromEntries(Object.entries(raw.areas.harbor.payout).map(([key, value]) => {
      const modifier = canonical && raw.gain[key] > EPS ? N.toNumber(N.div(canonical.gain[key], raw.gain[key])) : 1;
      return [key, N.from(value * raw.global * modifier)];
    }));
  }
  function configurations(state, id) {
    const x = state.expedition, a = x.areas[id]; if (!a) return [];
    const group = (kind, label, slots, options) => ({ id: kind, label, slots, selected: clone(a.plans[kind]), options: options.map(([key, name, description, enabled = true]) => ({ id: key, label: name, description, disabled: !enabled, action: { type: 'expedition-config', areaId: id, kind, slot: 0, id: key } })) });
    if (id === 'watchtower') return [group('assignments', 'Coordination assignments', rawRates(state).areas.watchtower.capacity, [['survey', 'Survey', 'Research capacity'], ['industry', 'Industry', 'Quarry hauling'], ['trade', 'Trade', 'Trail coin deliveries'], ['discovery', 'Shared discovery', 'Survey and interpretation together', built(x, 'guild-discovery')]]), group('target', 'Survey target', 1, [['near', 'Nearlands', 'Balanced maps and research'], ['deep', 'Deep records', '+50% research, −45% maps', learned(x, id, 'optics')], ['ocean', 'Ocean routes', '+80% maps, −25% research', learned(x, id, 'forecasting')]])];
    if (id === 'workshop') return [group('templates', 'Manufacturing templates', learned(x, id, 'replication') && rank(x, id, 'replication') > 0 ? 2 : 1, [['supplies', 'Supplies', 'Turn ore into provisions'], ['tools', 'Tool parts', 'Recover useful ore and fewer provisions'], ['instruments', 'Instruments', 'Turn ore into knowledge', learned(x, 'watchtower', 'optics')]])];
    if (id === 'ruins') return [group('discovery', 'Selected discovery', 1, [['botanical', 'Botanical', 'Recover herbs and trade artifacts'], ['metallic', 'Metallic', 'Recover ore and industry artifacts'], ['inscribed', 'Inscribed', 'Recover knowledge and survey artifacts']]), group('loadouts', 'Artifact assignments', learned(x, id, 'resonance') && rank(x, id, 'resonance') > 0 ? 2 : 1, [['industry', 'Industry', 'Ore yield', a.discoveries.metallic > 0 && learned(x, id, 'attunement')], ['trade', 'Trade', 'Trail coin cargo', a.discoveries.botanical > 0 && learned(x, id, 'attunement')], ['survey', 'Survey', 'Interpretation research', a.discoveries.inscribed > 0 && learned(x, id, 'attunement')]])];
    if (id === 'harbor') return [group('port', 'Next voyage port', 1, [['coast', 'Coastal ports', 'Short automatic trade route'], ['ruins', 'Ancient coast', 'Longer journeys with larger cargo returns', learned(x, id, 'navigation')], ['ocean', 'Deep ocean', 'Large manifest; storms can slow travel', learned(x, 'watchtower', 'forecasting')]])];
    return [];
  }
  function choices(state, id) {
    const x = state.expedition, a = x.areas[id]; if (!a) return [];
    const entries = id === 'greenway' ? [['trade', 'Trade', 'Full coin cargo'], ['freight', 'Freight', 'Trade fewer coins for quarry hauling'], ['survey', 'Survey', 'Trade coins for more maps'], ['mixed', 'Parallel routes', 'Trade and freight together'], ['continental', 'Continental', 'Supply Harbor trade while domestic routes continue'], ['trade-survey', 'Trade + survey', 'Retained Survey exchange: trade and maps together']]
      : id === 'quarry' ? [['balanced', 'Balanced', 'Balanced ore and alloys'], ['rich', 'Rich ore', 'Less raw capacity, greater yield'], ['alloy', 'Alloys', 'Less ore, stronger Workshop conversion'], ['precision', 'Precision batches', 'Retained recipe: more yield with lower furnace capacity'], ['mixed', 'Split batches', 'Retained recipe: intermediate capacity and yield'], ['optics', 'Optical batches', 'Retained recipe: ore becomes research'], ['adaptive', 'Adaptive batches', 'Retained control room switches recipes with the queues']]
      : id === 'watchtower' ? [['survey', 'Survey', 'Full research'], ['industry', 'Industry', 'Coordinate quarry hauling'], ['trade', 'Trade', 'Coordinate Trail deliveries']]
      : id === 'workshop' ? [['tools', 'Tools', 'Balanced manufacturing support'], ['extraction', 'Extraction', 'Assign mechanisms to quarry extraction'], ['manufacture', 'Manufacture', 'Assign mechanisms to assembly'], ['precision', 'Precision', 'Use less ore at lower manufacturing speed'], ['integrated', 'Integrated industry', 'Recover a constrained extraction surplus while manufacturing supplies']]
      : id === 'ruins' ? [['industry', 'Industry', 'Recovered artifacts strengthen ore yield'], ['trade', 'Trade', 'Recovered artifacts strengthen coin cargo'], ['survey', 'Survey', 'Interpret discoveries for research']]
      : [['trade', 'Trade', 'Full trade earnings'], ['materials', 'Materials', 'Carry ore in exchange for 45% of coin cargo and 25% of map cargo'], ['discovery', 'Discovery', 'Less trade, more maps and research'], ['commerce', 'Continental manifest', 'Carry returning provisions and discovery cargo alongside trade']];
    return entries.map(([key, label, effect]) => {
      let open = true;
      if (id === 'greenway') open = key === 'trade' || key === 'freight' && learned(x, id, 'caravans') || key === 'survey' && learned(x, id, 'scouts') || key === 'mixed' && learned(x, id, 'caravans') || key === 'continental' && built(x, 'continental-exchange') || key === 'trade-survey' && x.legacyDevelopments.includes('survey-exchange');
      if (id === 'quarry' && key !== 'balanced') open = ['rich', 'alloy'].includes(key) ? learned(x, id, 'geology') : x.legacyDevelopments.includes({ precision: 'quarry-precision', mixed: 'shared-workshops', optics: 'optical-foundry', adaptive: 'tower-control-room' }[key]);
      if (id === 'watchtower' && key !== 'survey') open = learned(x, id, 'crew');
      if (id === 'workshop' && key !== 'tools') open = key === 'integrated' ? built(x, 'guild-industry') : learned(x, id, key === 'precision' ? 'precision' : 'mechanisms');
      if (id === 'harbor' && key === 'commerce') open = built(x, 'guild-commerce');
      return { id: key, label, effect, description: effect, selected: a.choice === key, visible: open, disabled: !open, action: { type: 'expedition-choice', areaId: id, id: key } };
    });
  }
  function metrics(state) {
    const r = rawRates(state), result = {};
    for (const [id, rates] of Object.entries(r.areas)) for (const key of ['work', 'capacity', 'picks', 'carts', 'furnace', 'research', 'flow', 'demand', 'freight', 'cargo', 'duration', 'travel', 'coordination', 'delving', 'interpretation', 'recovery', 'assembly', 'supply', 'voyagePace']) if (rates[key] !== undefined) result[id + ':' + key] = { areaId: id, label: areaDef(id).name + ' · ' + ({ picks: 'Extraction capacity', carts: 'Hauling capacity', furnace: 'Smelting capacity', flow: 'Actual throughput', work: 'Expansion work', capacity: 'Storage / assignment capacity', research: 'Research', demand: 'Input demand', freight: 'Freight support', cargo: 'Cargo', duration: 'Voyage duration', travel: 'Travel capacity', coordination: 'Coordination strength', delving: 'Delving capacity', interpretation: 'Interpretation capacity', recovery: 'Recovery capacity', assembly: 'Assembly capacity', supply: 'Next manifest provisions', voyagePace: 'Fleet travel capacity' }[key]), value: N.from(rates[key]), unit: key === 'duration' ? 's' : ['capacity', 'coordination', 'cargo', 'supply'].includes(key) ? '' : '/s' };
    if (r.areas.harbor) Object.entries(manifestPayout(state, r)).forEach(([key, value]) => { result['harbor:payout:' + key] = { areaId: 'harbor', label: 'Harbor · Next manifest ' + key, value, unit: ' per voyage' }; });
    const guild = rateProvider ? rateProvider(state, ownership.get(state)) : Object.fromEntries(RESOURCES.map(id => [id, N.from(Math.max(0, r.gain[id] - r.drain[id]))]));
    RESOURCES.forEach(id => { result['guild:' + id] = { label: 'Guild total · ' + id, value: guild[id], unit: '/s' }; }); return result;
  }
  function impact(state, next) {
    const before = metrics(state), after = metrics(next);
    return Object.keys(after).filter(key => before[key] && N.cmp(before[key].value, after[key].value) && Math.abs(N.toNumber(N.div(after[key].value, N.max(before[key].value, 1e-12))) - 1) > 1e-9).map(key => ({ metric: key, areaId: after[key].areaId, label: after[key].label, current: N.format(before[key].value), next: N.format(after[key].value), currentValue: before[key].value, nextValue: after[key].value, unit: after[key].unit }));
  }
  function catalog(state) {
    const x = state.expedition, rows = [];
    for (const [id, a] of Object.entries(x.areas)) for (const d of areaDef(id).tracks) {
      if (!a.learned.includes(d.id)) continue;
      const q = quote(state, id, d.id, x.batch), copy = detached(state);
      if (q.valid) copy.expedition.areas[id].ranks[d.id] = q.rankAfter;
      const effects = q.valid ? impact(state, copy) : [];
      const nextMilestone = [3, 10, 25, 50, 100, 250, 1000].find(r => r > a.ranks[d.id] && r <= a.cap);
      rows.push({ id: 'area:' + id + ':' + d.id, catalogId: 'area:' + id + ':' + d.id, trackId: d.id, areaId: id, name: d.name, label: d.name, icon: d.icon, group: 'area', rank: a.ranks[d.id], level: a.ranks[d.id], maxRank: a.cap, maxLevel: a.cap, maxed: a.ranks[d.id] === a.cap, visible: true, disabled: !q.valid || !q.affordable, reason: q.reason, quantity: x.batch, rankAfter: q.rankAfter,
        description: d.effect + '. Ranks rebuild after a reset; learned branches and rank ceilings stay.', effectText: d.effect, cost: Object.entries(q.costs).map(([resource, amount]) => ({ resource, amount, text: N.format(amount) + ' ' + resource })), impact: effects, comparison: effects.slice(0, 2).map(e => e.label + ' ' + e.current + ' → ' + e.next + e.unit).join(' · '), sourceAreas: [id], targetAreas: [...new Set([id].concat(effects.map(e => e.areaId).filter(Boolean)))], dependencies: [],
        nextMilestone: nextMilestone ? { rank: nextMilestone, remaining: nextMilestone - a.ranks[d.id], label: nextMilestone === 25 ? 'Specialist plan' : d.name + ' mastery' } : null,
        action: { type: 'expedition-buy', areaId: id, id: d.id, count: x.batch, quote: q.token } });
    }
    const nextLocked = D.PROJECTS.find(d => !built(x, d.id) && !projectDependencies(state, d).every(p => p.met));
    D.PROJECTS.forEach(d => {
      const dependencies = projectDependencies(state, d), owned = built(x, d.id), running = x.commission?.id === d.id;
      const costs = Object.entries(d.costs).map(([resource, value]) => ({ resource, amount: N.from(value), text: N.format(value) + ' ' + resource }));
      rows.push({ id: 'development:' + d.id, catalogId: 'development:' + d.id, name: d.name, label: d.name, icon: areaDef(d.target).icon, areaId: d.target, group: d.unlock.cap ? 'expansion' : 'research', effectKind: 'unlock', sourceAreas: [d.source], targetAreas: [d.target], description: d.effect, effectText: d.effect, owned, maxed: owned, level: owned ? 1 : 0, maxLevel: 1,
        visible: owned || running || dependencies.every(p => p.met) || d === nextLocked, disabled: owned || !!x.commission || dependencies.some(p => !p.met) || costs.some(c => N.cmp(state.resources[c.resource], c.amount) < 0), dependencies, cost: costs, impact: [{ metric: 'behavior:' + d.id, label: d.name, current: owned ? 'Learned' : running ? 'Researching' : 'Locked', next: d.effect, unit: '' }],
        progress: running ? x.commission.work / d.work : owned ? 1 : 0, action: { type: 'expedition-development', id: d.id } });
    });
    return rows;
  }
  function view(state) {
    const x = state.expedition, id = x.selectedArea, a = x.areas[id], raw = rawRates(state), r = raw.areas[id], t = targets(x), list = catalog(state), p = progress(x), current = id === x.projectArea;
    const operationStatus = key => {
      const operation = raw.areas[key], area = x.areas[key];
      if (key === 'harbor') return area.voyages.length ? area.voyages.length + ' voyage' + (area.voyages.length === 1 ? '' : 's') + ' at sea' : operation.canLaunch ? 'Preparing voyage' : 'Waiting for provisions';
      if (key === 'greenway') return operation.income > EPS ? 'Delivering cargo' : 'Preparing deliveries';
      if (key === 'quarry') return operation.actualFurnace > EPS ? 'Refining ore' : operation.actualPicks > EPS ? 'Extracting ore' : 'Waiting for ore';
      if (key === 'watchtower') return operation.research > EPS ? 'Surveying' : 'Preparing survey';
      if (key === 'workshop') return operation.flow > EPS ? 'Manufacturing' : 'Waiting for ore';
      return operation.flow > EPS ? 'Recovering finds' : 'Studying discoveries';
    };
    const areaRows = ids.filter(key => x.areas[key]).map(key => ({ id: key, label: areaDef(key).name, icon: areaDef(key).icon, unlocked: true, selected: key === id, established: ids.indexOf(key) > 2 || x.cleared >= ids.indexOf(key), objective: key === x.projectArea && !x.completed ? 'Expanding' : 'Producing', progress: key === x.projectArea ? p : 1, rateText: operationStatus(key), status: operationStatus(key), attention: x.recent.some(e => e.areaId === key && e.sequence > x.areas[key].seen), action: { type: 'expedition-select', areaId: key } }));
    const comparison = action => { const copy = detached(state); const result = act(copy, action); return result.ok ? impact(state, copy) : []; };
    const workingChoices = choices(state, id).map(c => Object.assign({}, c, { impact: c.disabled ? [] : comparison(c.action) }));
    const configurationsView = configurations(state, id).map(group => Object.assign({}, group, { options: group.options.map(o => Object.assign({}, o, { impact: o.disabled ? [] : comparison(o.action) })), slotOptions: Array.from({ length: group.slots }, (_, slot) => ({ slot, selected: Array.isArray(group.selected) ? group.selected[slot] : group.selected, options: group.options.map(o => { const action = Object.assign({}, o.action, { slot }); return Object.assign({}, o, { action, impact: o.disabled ? [] : comparison(action) }); }) })) }));
    const focusPreview = detached(state); focusPreview.expedition.focus.active = id; focusPreview.expedition.focus.remaining = D.FOCUS.duration;
    const focusImpact = impact(state, focusPreview);
    const focusedRates = rawRates(focusPreview), focusInputs = RESOURCES.filter(key => focusedRates.drain[key] > EPS).map(key => ({ resource: key, current: raw.drain[key], next: focusedRates.drain[key], unit: '/s' }));
    const focusContext = id === 'quarry' ? 'Current deposit: ' + a.choice + '. Mining, hauling and refining accelerate together; finite queues still apply.' : id === 'workshop' ? 'Current templates: ' + a.plans.templates.join(' + ') + '. Ore demand rises with actual assembly; reserves remain protected.' : id === 'ruins' ? 'Selected discovery: ' + a.plans.discovery + '. Delving, interpretation and recovery accelerate together.' : id === 'harbor' ? 'Advance ' + a.voyages.length + ' already funded voyage(s); frozen cargo and normal launch supply bills stay unchanged.' : id === 'watchtower' ? 'Current survey target: ' + a.plans.target + '. Concentrate its knowledge and research work.' : 'Concentrate Trail deliveries and expansion work; other area income is unchanged.';
    return { local: true, networkVersion: 3, legacy: false, sequence: x.sequence, stage: { id: id + ':' + x.index, name: areaDef(id).name, kind: id, areaId: id, index: x.index, region: 'Guild chapter ' + Math.max(1, Math.min(7, Math.floor(x.projects.length / 4) + 1)), completed: current ? x.completed : true, progress: current ? p : 1, work: current ? x.work : t.work, target: t.work, finaleWork: current ? x.finaleWork : t.finale, finaleTarget: t.finale, phase: current && !x.completed ? x.work < t.work ? 'Build' : 'Capstone' : 'Producing', progressText: current && !x.completed ? Math.floor(p * 100) + '%' : 'Producing', goal: current && !x.completed ? stageName(x) : 'Develop the network', objective: current && !x.completed ? stageName(x) : 'Develop the network', checkpoint: { label: current && !x.completed ? x.work < t.work ? 'Expand infrastructure' : 'Complete the landmark' : 'All areas keep working', progress: p } },
      cards: list.filter(row => row.group === 'area' && row.areaId === id), catalog: list, areas: areaRows, choices: workingChoices, configurations: configurationsView, blueprints: [], milestones: x.recent.slice(), recent: x.recent.slice(), events: x.recent.slice(), unseen: x.recent.filter(e => e.sequence > x.seen),
      batch: { selected: x.batch, options: batchModes(state) },
      specializations: [{ id: null, label: 'Balanced tracks', selected: a.specialization === null, action: { type: 'expedition-specialize', areaId: id, id: null } }].concat(areaDef(id).tracks.filter(d => a.highRanks[d.id] >= 25).map(d => ({ id: d.id, label: d.name, effect: '+35% of the rank bonus; other tracks −10% of their rank bonus', selected: a.specialization === d.id, action: { type: 'expedition-specialize', areaId: id, id: d.id } }))).map(o => Object.assign({}, o, { visible: true, disabled: false, impact: comparison(o.action) })),
      focus: { unlocked: x.focus.unlocked, charges: x.focus.charges, max: D.FOCUS.capacity, nextChargeSeconds: x.focus.charges < D.FOCUS.capacity ? D.FOCUS.recharge - x.focus.recharge : 0, active: x.focus.active, remaining: x.focus.remaining, actions: [{ id: 'priority', label: { greenway: 'Priority delivery', quarry: 'Target deposit', watchtower: 'Focus survey', workshop: 'Rush order', ruins: 'Study discovery', harbor: 'Advance voyage' }[id], description: 'Spend one shared Focus for 90 seconds. ' + focusContext, inputs: focusInputs, impact: focusImpact, disabled: !x.focus.unlocked || !x.focus.charges || x.focus.remaining > 0 || !focusImpact.length || id === 'harbor' && !a.voyages.length || id === 'workshop' && !(r.flow > EPS), action: { type: 'expedition-focus', areaId: id, id: 'priority' } }] },
      next: { label: 'Expand ' + areaDef(x.index + 1 < 3 ? ids[x.index + 1] : ids.filter(key => x.areas[key])[(x.index - 2) % Object.keys(x.areas).length]).name, description: 'All discovered areas remain available and keep producing.', disabled: !x.completed, action: { type: 'expedition-next' } },
      automation: Object.assign({ unlocked: state.lifetime.refits > 0, choices: ['balanced', 'progress', 'income', 'materials'].map(key => ({ id: key, label: key, action: { type: 'expedition-automation', enabled: true, priority: key } })) }, x.automation),
      commission: x.commission ? { id: x.commission.id, name: projectDef(x.commission.id).name, progress: x.commission.work / projectDef(x.commission.id).work, seconds: (projectDef(x.commission.id).work - x.commission.work) / raw.researchRate } : null,
      stations: id === 'quarry' ? [['picks', 'Mine', r.actualPicks, r.picks, a.buffers.input], ['carts', 'Cart', r.actualCarts, r.carts, a.buffers.output], ['furnace', 'Furnace', r.actualFurnace, r.furnace, 0]].map(([key, label, rate, maxRate, buffer]) => ({ id: key, label, rate, maxRate, rateText: rate.toFixed(2) + '/s', buffer, capacity: r.capacity, bottleneck: key === r.bottleneck, status: rate < maxRate - EPS ? 'Waiting for supply' : 'Working' })) : [],
      scene: { ruleset: 'progression', kind: id, index: x.index, region: 'greenway', progress: current ? p : 1, completed: current ? x.completed : true, established: ids.indexOf(id) > 2 || x.cleared >= ids.indexOf(id), ranks: clone(a.ranks), unlocked: a.learned.slice(), developments: x.projects.slice(), route: a.choice, dispatch: a.choice, allocation: a.choice, quality: a.choice, oreBuffer: a.buffers.input, smeltBuffer: a.buffers.output, buffers: clone(a.buffers), capacity: r.capacity, bottleneck: r.bottleneck, workers: r.workers || { total: 3, repair: 2, protection: 1 }, beacon: p, rates: Object.assign({}, r), flows: { picks: r.actualPicks ?? r.flow, carts: r.actualCarts ?? r.flow, furnace: r.actualFurnace ?? r.flow, production: r.flow }, templates: r.templates, discoveries: r.discoveries, assignments: r.assignments, voyage: id === 'harbor' ? { progress: a.voyages[0] ? a.voyages[0].work / a.voyages[0].target : 0, duration: r.duration, ships: a.voyages.length, capacity: r.capacity, cargo: r.cargo, manifests: clone(a.voyages), weather: r.weather } : null, focus: x.focus.active === id && x.focus.remaining > 0 } };
  }
  function reset(state, type) {
    const old = state.expedition;
    if (!active(state)) {
      const x = create(); x.cleared = state.lifetime.highestRoute;
      for (const id of ids.slice(0, 3)) if (old?.areas?.[id] || ids.indexOf(id) <= state.lifetime.highestRoute + 1) {
        x.areas[id] = makeArea(id); x.areas[id].learned = areaDef(id).tracks.slice(0, 3).map(t => t.id);
        const oldCap = Math.min(200, 20 + Math.max(0, Math.floor(((old?.cleared ?? state.lifetime.highestRoute) + 1) / 3)) * 4 + (id === 'quarry' && old?.developments?.includes('deep-veins') ? 8 : 0));
        if (oldCap > 100) x.areas[id].cap = 250;
        if (old?.areas?.[id]) {
          const prior = old.areas[id], target = x.areas[id];
          for (const track of areaDef(id).tracks) target.highRanks[track.id] = Math.min(1000, prior.ranks[track.id] || (id === 'watchtower' && track.id === 'signals' ? prior.ranks.lift : 0) || 0);
          if (id === 'greenway') target.choice = prior.choices.dispatch === 'relay' ? 'mixed' : prior.choices.dispatch;
          if (id === 'quarry') target.choice = ['throughput', 'quality'].includes(prior.choices.smelting) ? 'balanced' : prior.choices.smelting;
          if (id === 'watchtower') { target.choice = prior.choices.allocation === 'protect' ? 'industry' : prior.choices.allocation === 'repair' ? 'trade' : 'survey'; target.plans.assignments = [target.choice]; }
        }
      }
      x.legacyDevelopments = old?.developments?.slice() || [];
      for (const [prior, next] of Object.entries({ 'trail-caravans': 'wheelworks', 'tower-survey': 'tower-surveys', 'quarry-precision': 'deposit-maps' })) if (x.legacyDevelopments.includes(prior)) { x.projects.push(next); const d = projectDef(next); if (x.areas[d.target] && !x.areas[d.target].learned.includes(d.unlock.track)) x.areas[d.target].learned.push(d.unlock.track); }
      if (old?.automation) x.automation = Object.assign({}, x.automation, old.automation);
      x.blueprints = old?.blueprints?.slice() || [];
      if (old?.mastery) x.mastery = clone(old.mastery);
      x.purchases = old?.purchases || 0; state.expedition = x;
      const successors = { 'trail-caravans': 'wheelworks', 'paved-roads': 'rail-network', 'trail-depot': 'tower-surveys', 'quarry-precision': 'deposit-maps', 'efficient-crucibles': 'alloy-machinery', 'signal-network': 'tower-surveys', 'protective-escorts': 'industrial-supports', 'tower-survey': 'tower-surveys', 'relay-network': 'rail-network', 'survey-charters': 'deposit-maps', 'deep-veins': 'deepworks-commission', 'shared-workshops': 'precision-patterns', 'recovery-chutes': 'sorting-lines', 'dispatch-ledgers': 'sorting-lines', 'optical-foundry': 'workshop-lenses', 'trail-prospectors': 'deposit-maps', 'tower-control-room': 'precision-patterns', 'survey-exchange': 'continental-exchange' };
      const successor = action => action?.type === 'expedition-development' && successors[action.id] ? { type: action.type, id: successors[action.id] } : action;
      state.guild.plan.goal = successor(state.guild.plan.goal);
      state.guild.plan.queue = state.guild.plan.queue.map(successor);
      for (const loadout of state.guild.loadouts) { loadout.plan.goal = successor(loadout.plan.goal); loadout.plan.queue = loadout.plan.queue.map(successor); }
    }
    const x = state.expedition;
    x.renewed = true;
    for (const a of Object.values(x.areas)) { for (const key of Object.keys(a.ranks)) a.ranks[key] = 0; a.buffers = { input: 0, output: 0 }; a.voyages = []; a.discoveryWork = 0; }
    // A funded research commission is permanent learning, not repeatable run
    // expansion work. Keeping its partial work makes multi-day projects usable
    // while the player continues to Refit the production network.
    if (!x.focus.unlocked && state.lifetime.refits > 0) { x.focus.unlocked = true; x.focus.charges = D.FOCUS.capacity; x.focus.recharge = 0; }
    const max = Math.max(...batchModes(state).filter(b => b.unlocked).map(b => b.count));
    if (!batchModes(state).some(b => b.count === x.batch && b.unlocked)) x.batch = 1;
    RESOURCES.forEach(id => { state.resources[id] = N.zero(); });
    Object.keys(state.upgrades).forEach(id => { state.upgrades[id] = 0; });
    restart(state, 0);
    const starter = quote(state, 'greenway', 'boots', max);
    state.resources.coins = N.from(starter.costs.coins || 6);
    x.automation.clock = 0;
    return { starter: state.resources.coins, quantity: max };
  }
  function validate(x, state) {
    const keys = ['version', 'index', 'cleared', 'completed', 'renewed', 'selectedArea', 'projectArea', 'work', 'finaleWork', 'areas', 'projects', 'commission', 'batch', 'revision', 'focus', 'automation', 'sequence', 'seen', 'recent', 'purchases', 'blueprints', 'mastery', 'legacyDevelopments'];
    if (!exact(x, keys) || x.version !== 3 || !finite(x.index, 1e9, true) || !Number.isSafeInteger(x.cleared) || x.cleared < -1 || x.cleared > 1e9 || x.index > x.cleared + 1 || typeof x.completed !== 'boolean' || !object(x.areas) || !x.areas.greenway || !x.areas[x.selectedArea] || !x.areas[x.projectArea] || !finite(x.work) || !finite(x.finaleWork)) return false;
    if (typeof x.renewed !== 'boolean' || x.renewed && state && !(state.lifetime.refits || state.lifetime.charters) || !Array.isArray(x.projects) || new Set(x.projects).size !== x.projects.length || x.projects.some(id => !projectDef(id)) || !D.BATCHES.some(b => b.count === x.batch) || !finite(x.revision, 1e12, true)) return false;
    if (state && (!batchModes(state).some(b => b.count === x.batch && b.unlocked) || x.focus?.unlocked !== (state.lifetime.refits > 0))) return false;
    if (x.projects.some(id => !x.areas[projectDef(id).source] || !x.areas[projectDef(id).target] || projectDef(id).requires.some(required => !x.projects.includes(required)))) return false;
    if (!exact(x.focus, ['charges', 'recharge', 'active', 'remaining', 'unlocked']) || !finite(x.focus.charges, 3, true) || !finite(x.focus.recharge, D.FOCUS.recharge) || !finite(x.focus.remaining, D.FOCUS.duration) || typeof x.focus.unlocked !== 'boolean' || x.focus.active !== null && !x.areas[x.focus.active] || (x.focus.active === null) !== (x.focus.remaining === 0) || !x.focus.unlocked && (x.focus.charges || x.focus.remaining || x.focus.recharge) || x.focus.charges === 3 && x.focus.recharge !== 0) return false;
    if (!exact(x.automation, ['enabled', 'priority', 'dispatch', 'clock']) || typeof x.automation.enabled !== 'boolean' || typeof x.automation.dispatch !== 'boolean' || !['balanced', 'income', 'progress', 'materials'].includes(x.automation.priority) || !finite(x.automation.clock, 60)) return false;
    if (!finite(x.sequence, 1e12, true) || !finite(x.seen, x.sequence, true) || !finite(x.purchases, 1e12, true) || !Array.isArray(x.recent) || x.recent.length > 12 || x.recent.some(e => !exact(e, ['sequence', 'kind', 'title', 'text', 'stage', 'areaId']) || !finite(e.sequence, x.sequence, true) || !['milestone', 'development', 'stage'].includes(e.kind) || typeof e.title !== 'string' || e.title.length > 120 || typeof e.text !== 'string' || e.text.length > 400 || !finite(e.stage, 1e9, true) || !ids.includes(e.areaId))) return false;
    if (x.recent.some((e, i) => !e.sequence || i > 0 && e.sequence <= x.recent[i - 1].sequence)) return false;
    if (!Array.isArray(x.blueprints) || x.blueprints.some(v => !['caravan', 'engineering', 'pathfinding'].includes(v)) || !exact(x.mastery, ids.slice(0, 3)) || Object.values(x.mastery).some(v => !finite(v, 1e6, true)) || !Array.isArray(x.legacyDevelopments) || x.legacyDevelopments.length > 30 || x.legacyDevelopments.some(v => typeof v !== 'string' || v.length > 80)) return false;
    for (const [id, a] of Object.entries(x.areas)) {
      const d = areaDef(id); if (!d || !exact(a, ['ranks', 'highRanks', 'learned', 'cap', 'choice', 'specialization', 'plans', 'voyages', 'discoveryWork', 'discoveries', 'buffers', 'elapsed', 'purchases', 'seen']) || ![100, 250, 1000].includes(a.cap) || !exact(a.ranks, d.tracks.map(t => t.id)) || !exact(a.highRanks, d.tracks.map(t => t.id)) || Object.keys(a.ranks).some(k => !finite(a.ranks[k], a.cap, true) || !finite(a.highRanks[k], 1000, true) || a.highRanks[k] < a.ranks[k])) return false;
      if (ids.indexOf(id) > 2 && !x.projects.some(key => projectDef(key).unlock.area === id)) return false;
      if (a.cap > 100 && !x.projects.some(key => projectDef(key).target === id && (projectDef(key).unlock.cap || 0) >= a.cap) && !(a.cap === 250 && ids.indexOf(id) < 3 && Math.min(200, 20 + Math.floor((x.cleared + 1) / 3) * 4 + (id === 'quarry' && x.legacyDevelopments.includes('deep-veins') ? 8 : 0)) > 100)) return false;
      if (a.specialization !== null && (!trackDef(id, a.specialization) || a.highRanks[a.specialization] < 25)) return false;
      if (!exact(a.plans, ['assignments', 'templates', 'target', 'discovery', 'loadouts', 'port']) || !Array.isArray(a.plans.assignments) || a.plans.assignments.length < 1 || a.plans.assignments.length > 3 || a.plans.assignments.some(v => !['survey', 'industry', 'trade', 'discovery'].includes(v)) || !Array.isArray(a.plans.templates) || a.plans.templates.length < 1 || a.plans.templates.length > 2 || a.plans.templates.some(v => !['supplies', 'tools', 'instruments'].includes(v)) || !['near', 'deep', 'ocean'].includes(a.plans.target) || !['botanical', 'metallic', 'inscribed'].includes(a.plans.discovery) || !Array.isArray(a.plans.loadouts) || a.plans.loadouts.length < 1 || a.plans.loadouts.length > 2 || a.plans.loadouts.some(v => !['industry', 'trade', 'survey'].includes(v)) || !['coast', 'ruins', 'ocean'].includes(a.plans.port)) return false;
      if (!finite(a.discoveryWork, 5) || !exact(a.discoveries, ['botanical', 'metallic', 'inscribed']) || Object.values(a.discoveries).some(v => !finite(v, 1e12, true)) || !Array.isArray(a.voyages) || a.voyages.length > 2 || id !== 'harbor' && a.voyages.length || a.voyages.some(v => !exact(v, ['work', 'target', 'port', 'supplies', 'convoy', 'payout']) || !finite(v.target, 1e9) || v.target < 1 || !finite(v.work, v.target) || !finite(v.supplies, 1e15) || !finite(v.convoy, 1e12) || v.convoy < 1 || !['coast', 'ruins', 'ocean'].includes(v.port) || !object(v.payout) || Object.keys(v.payout).some(k => !RESOURCES.includes(k) || !N.valid(v.payout[k])))) return false;
      if (!Array.isArray(a.learned) || !a.learned.includes(d.tracks[0].id) || new Set(a.learned).size !== a.learned.length || a.learned.some(k => !trackDef(id, k)) || Object.keys(a.ranks).some(k => !a.learned.includes(k) && a.ranks[k]) || !exact(a.buffers, ['input', 'output']) || !finite(a.buffers.input, 1e9) || !finite(a.buffers.output, 1e9) || !finite(a.elapsed, 8.64e12) || !finite(a.purchases, 1e12, true) || !finite(a.seen, x.sequence, true)) return false;
      if (a.learned.some(key => trackDef(id, key).index >= 3 && !x.projects.some(project => projectDef(project).target === id && projectDef(project).unlock.track === key))) return false;
      if (id === 'watchtower' && (a.plans.assignments.length > 1 + (a.highRanks.crew >= 10 ? 1 : 0) + (a.learned.includes('relay-grid') ? 1 : 0) || a.plans.assignments.includes('discovery') && !built(x, 'guild-discovery') || a.plans.target === 'deep' && !a.learned.includes('optics') || a.plans.target === 'ocean' && !a.learned.includes('forecasting'))) return false;
      if (id === 'workshop' && (a.plans.templates.length > 1 && !a.learned.includes('replication') || a.plans.templates.includes('instruments') && !learned(x, 'watchtower', 'optics'))) return false;
      if (id === 'ruins' && a.plans.loadouts.length > 1 && !a.learned.includes('resonance')) return false;
      if (id === 'harbor' && (a.voyages.length > (a.learned.includes('fleet-command') && a.ranks['fleet-command'] > 0 ? 2 : 1) || a.plans.port === 'ruins' && !a.learned.includes('navigation') || a.plans.port === 'ocean' && !learned(x, 'watchtower', 'forecasting'))) return false;
      if (state && !choices(state, id).some(c => c.id === a.choice && !c.disabled)) return false;
    }
    if (x.commission !== null && (!exact(x.commission, ['id', 'work']) || !projectDef(x.commission.id) || x.projects.includes(x.commission.id) || !finite(x.commission.work, projectDef(x.commission.id).work))) return false;
    if (x.commission && (!x.areas[projectDef(x.commission.id).source] || !projectDef(x.commission.id).unlock.area && !x.areas[projectDef(x.commission.id).target] || projectDef(x.commission.id).requires.some(id => !x.projects.includes(id)))) return false;
    const t = targets(x);
    return x.work <= t.work + EPS && x.finaleWork <= t.finale + EPS && (x.work >= t.work - EPS || x.finaleWork === 0) && (!x.completed || x.work >= t.work - EPS && x.finaleWork >= t.finale - EPS && x.index <= x.cleared);
  }
  return { Content: D, active, create, validate, quote, cost, power, batchModes, targets, progress, stageName, rawRates, contribution, localRates, rates: state => rawRates(state).areas, nextEvent, tick, finish, restart, reset, autoBuy, act, view, catalog, impact, developmentTask, reserve, setEntitlements, setReserveProvider, setRateProvider, setWorkRateProvider };
});
