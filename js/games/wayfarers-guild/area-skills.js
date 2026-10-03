(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./progression-content.js') : root.WayfarersProgressionContent, common ? require('./area-skills-content.js') : root.WayfarersAreaSkillsContent);
  if (common) module.exports = api;
  if (root) root.WayfarersAreaSkills = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, D, C) {
  'use strict';
  const EPS = 1e-8, IDS = C.SKILLS.map(d => d.id), AREAS = Object.keys(C.AREA), RESOURCES = ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps'];
  const definition = id => C.SKILLS.find(d => d.id === id), areaDef = id => D.AREAS.find(d => d.id === id);
  const own = (o, key) => Object.prototype.hasOwnProperty.call(o, key), clone = value => JSON.parse(JSON.stringify(value));
  const object = o => o !== null && typeof o === 'object' && !Array.isArray(o);
  const exact = (o, keys) => object(o) && Object.keys(o).length === keys.length && keys.every(k => own(o, k));
  const finite = (v, max = 1e15) => Number.isFinite(v) && v >= 0 && v <= max;
  const active = state => state.expedition?.version === 3 && !!state.areaSkills;
  const emptyPayout = () => Object.fromEntries(RESOURCES.map(id => [id, N.zero()]));
  const runtimeInitial = () => ({ hopper: 0, notebook: 0, batchClock: 0, batch: emptyPayout(), drillWork: 0, drillRemaining: 0, shiftClock: 0, templateWork: 0, siteWork: 0, convoyReady: false, voyageSequence: 0, voyageReceipts: [], lastCommission: '', trip: null });
  function initial(state, historical = false) {
    return { version: 1, revision: 0, unlocked: [], ranks: Object.fromEntries(IDS.map(id => [id, 0])), highRanks: Object.fromEntries(IDS.map(id => [id, 0])),
      output: Object.fromEntries(AREAS.map(id => [id, id === 'greenway' && historical ? state.trailDeliveries?.deliveries || 0 : 0])),
      configs: Object.fromEntries(C.SKILLS.filter(d => d.options.length).map(d => [d.id, 'off'])), runtime: runtimeInitial() };
  }
  function validate(value, state) {
    if (!exact(value, ['version', 'revision', 'unlocked', 'ranks', 'highRanks', 'output', 'configs', 'runtime']) || value.version !== 1 || !Number.isSafeInteger(value.revision) || value.revision < 0) return false;
    if (!Array.isArray(value.unlocked) || value.unlocked.length > IDS.length || new Set(value.unlocked).size !== value.unlocked.length || value.unlocked.some(id => !IDS.includes(id))) return false;
    if (!exact(value.ranks, IDS) || !exact(value.highRanks, IDS) || IDS.some(id => !Number.isSafeInteger(value.ranks[id]) || !finite(value.ranks[id], 10) || !Number.isSafeInteger(value.highRanks[id]) || !finite(value.highRanks[id], 10) || value.highRanks[id] < value.ranks[id] || value.highRanks[id] > 0 && !value.unlocked.includes(id))) return false;
    if (!exact(value.output, AREAS) || AREAS.some(id => !finite(value.output[id]))) return false;
    const configurations = C.SKILLS.filter(d => d.options.length);
    if (!exact(value.configs, configurations.map(d => d.id)) || configurations.some(d => !d.options.some(o => o.value === value.configs[d.id]) || value.configs[d.id] !== 'off' && !value.unlocked.includes(d.id))) return false;
    const r = value.runtime;
    if (!exact(r, Object.keys(runtimeInitial())) || ['hopper', 'notebook', 'batchClock', 'drillWork', 'drillRemaining', 'shiftClock', 'templateWork', 'siteWork'].some(key => !finite(r[key])) || typeof r.convoyReady !== 'boolean' || typeof r.lastCommission !== 'string' || r.lastCommission.length > 128 || !Number.isSafeInteger(r.voyageSequence) || r.voyageSequence < 0 || !exact(r.batch, RESOURCES) || RESOURCES.some(id => !N.valid(r.batch[id]))) return false;
    if (!Array.isArray(r.voyageReceipts) || r.voyageReceipts.length > 2 || r.voyageReceipts.some(v => !exact(v, ['key', 'supplies', 'refund', 'domestic', 'bonded']) || typeof v.key !== 'string' || v.key.length > 512 || !finite(v.supplies) || !finite(v.refund, .25) || !finite(v.domestic, .35) || typeof v.bonded !== 'boolean')) return false;
    if (r.trip !== null && (!exact(r.trip, ['speed', 'cargo']) || !finite(r.trip.speed, 2) || r.trip.speed < 1 || !finite(r.trip.cargo, 2) || r.trip.cargo < .85)) return false;
    if (state?.expedition?.version === 3 && value.unlocked.some(id => !state.expedition.areas[definition(id).areaId])) return false;
    if (state?.expedition?.version === 3) {
      const manifests = (state.expedition.areas.harbor?.voyages || []).map(v => voyageKey(v));
      for (const receipt of r.voyageReceipts) {
        const index = manifests.indexOf(receipt.key);
        if (index < 0 || receipt.refund > 0 && !value.highRanks['return-cargo'] || receipt.domestic > 0 && !value.highRanks['exchange-houses'] || receipt.bonded && !value.highRanks['bonded-routes']) return false;
        manifests.splice(index, 1);
      }
    }
    return true;
  }
  function rank(state, id) { return active(state) ? state.areaSkills.ranks[id] || 0 : 0; }
  function value(state, id, from, to) {
    const r = rank(state, id), d = definition(id);
    if (!r || !d) return 0;
    const a = from === undefined ? d.from : from, b = to === undefined ? d.to : to;
    return a + (b - a) * Math.log(r) / Math.log(10);
  }
  function mode(state, id) { return rank(state, id) ? state.areaSkills.configs[id] || 'off' : 'off'; }
  const enabled = (state, id) => mode(state, id) !== 'off';
  const throughput = (...values) => 1 + Math.min(.75, values.reduce((sum, v) => sum + Math.max(0, v || 0), 0));
  const refund = (...values) => Math.min(.25, values.reduce((sum, v) => sum + Math.max(0, v || 0), 0));
  function count(state, areaId) { return areaId === 'greenway' ? Math.max(state.areaSkills?.output.greenway || 0, state.trailDeliveries?.deliveries || 0) : state.areaSkills?.output[areaId] || 0; }
  function foundationRequirement(state, areaId, trackIdOrIndex) {
    const defs = areaDef(areaId)?.tracks || [], index = typeof trackIdOrIndex === 'number' ? trackIdOrIndex : defs.findIndex(d => d.id === trackIdOrIndex), a = state.expedition?.areas[areaId], track = defs[index];
    if (!track || index < 0 || index > 2 || !a) return { met: false, learned: false, areaId, trackId: track?.id, requirements: [] };
    const learned = a.learned?.includes(track.id) || false, requirements = [];
    if (index > 0) {
      const prior = defs[index - 1], target = index === 1 ? 4 : 3, output = C.AREA[areaId].foundation[index - 1];
      requirements.push({ label: prior.name + ' Lv. ' + target, icon: prior.icon, current: a.ranks[prior.id] || 0, required: target, met: (a.ranks[prior.id] || 0) >= target });
      requirements.push({ label: output + ' ' + C.AREA[areaId].unit, icon: C.AREA[areaId].resource, current: count(state, areaId), required: output, met: count(state, areaId) + EPS >= output });
    }
    return { met: learned || requirements.every(r => r.met), learned, areaId, trackId: track.id, requirements };
  }
  function foundations(state, areaId) {
    const a = state.expedition?.areas[areaId];
    if (!a || state.expedition.version !== 3) return [];
    return areaDef(areaId).tracks.slice(0, 3).map(d => { const requirement = foundationRequirement(state, areaId, d.id); return { id: 'area:' + areaId + ':' + d.id, trackId: d.id, areaId, name: d.name, label: d.name, icon: d.icon, rank: a.ranks[d.id], maxRank: a.cap, state: requirement.learned ? 'learned' : requirement.met ? 'ready' : 'locked', prerequisites: requirement.requirements, functionalRole: C.AREA[areaId].roles[d.index], role: d.index, order: d.index, module: 0, moduleLabel: 'Foundations' }; });
  }
  function requirements(state, d) {
    const a = state.expedition?.areas[d.areaId], local = C.SKILLS.filter(v => v.areaId === d.areaId), out = [];
    out.push({ label: 'Discover ' + C.AREA[d.areaId].name, icon: C.AREA[d.areaId].resource, current: a ? 1 : 0, required: 1, met: !!a });
    const basics = areaDef(d.areaId).tracks.slice(0, 3), known = basics.filter(v => a?.learned?.includes(v.id) && (!state.upgradeTiers || state.upgradeTiers.claimed.includes('area:' + d.areaId + ':' + v.id))).length;
    out.push({ label: 'Learn all three foundations', icon: 'guild', current: known, required: 3, met: known === 3 });
    out.push({ label: d.outputRequired + ' ' + C.AREA[d.areaId].unit, icon: C.AREA[d.areaId].resource, current: count(state, d.areaId), required: d.outputRequired, met: count(state, d.areaId) + EPS >= d.outputRequired });
    if (d.order === 1 || d.order === 2) { const previous = local[d.order - 1]; out.push({ label: previous.name + ' Lv. 2', icon: previous.icon, current: state.areaSkills?.highRanks[previous.id] || 0, required: 2, met: (state.areaSkills?.highRanks[previous.id] || 0) >= 2 }); }
    if (d.project) { const project = D.PROJECTS.find(p => p.id === d.project), met = state.expedition?.projects?.includes(d.project) || false; out.push({ label: project.name, icon: 'research', current: met ? 1 : 0, required: 1, met }); }
    return out;
  }
  function eligible(state, id) { const d = definition(id); return active(state) && !!d && requirements(state, d).every(r => r.met); }
  function modes(state) {
    let max = 1;
    for (const b of D.BATCHES) if (state.lifetime.refits >= b.refits && state.lifetime.charters >= b.charters) max = Math.max(max, b.count);
    return D.BATCHES.filter(b => b.count <= max).map(b => b.count);
  }
  function cost(state, id, atRank) {
    const d = definition(id); if (!d) return {};
    const r = atRank === undefined ? rank(state, id) : atRank, discount = state.expedition?.renewed && r < (state.areaSkills?.highRanks[id] || 0) ? .5 : 1;
    const coins = Math.ceil(d.base * Math.pow(2.4, r) * discount), result = { coins: N.from(coins) };
    if (d.resource !== 'coins') result[d.resource] = N.from(Math.ceil(coins * .025));
    return result;
  }
  function quote(state, id, quantity = 1) {
    const d = definition(id), r = rank(state, id), q = { id, count: quantity, rank: r, rankAfter: r + quantity, costs: {}, valid: false, affordable: false, reason: '' };
    if (!d || !active(state) || !state.areaSkills.unlocked.includes(id)) { q.reason = 'Unlock this technique first.'; return q; }
    if (!Number.isSafeInteger(quantity) || !modes(state).includes(quantity)) { q.reason = 'Earn this exact purchase quantity first.'; return q; }
    if (r + quantity > 10) { q.reason = 'Only ' + (10 - r) + ' ranks remain. Choose a smaller quantity.'; return q; }
    for (let i = 0; i < quantity; i += 1) for (const [key, amount] of Object.entries(cost(state, id, r + i))) q.costs[key] = N.add(q.costs[key] || 0, amount);
    q.valid = true; q.affordable = Object.entries(q.costs).every(([key, amount]) => N.cmp(state.resources[key], amount) >= 0);
    if (!q.affordable) q.reason = 'Save for the entire ×' + quantity + ' purchase.';
    q.token = [state.run.id, state.areaSkills.revision, state.expedition.revision, id, r, quantity].join(':');
    return q;
  }
  function optionAllowed(state, id, option) {
    if (option === 'off') return true;
    const x = state.expedition, a = x?.areas;
    if (id === 'parallel-furnaces' && option !== 'balanced') return a?.quarry?.learned.includes('geology');
    if (id === 'continental-logistics') return option === 'trade' || a?.greenway?.learned.includes(option === 'freight' ? 'caravans' : 'scouts');
    if (id === 'celestial-calendar') return option === 'near' || a?.watchtower?.learned.includes(option === 'deep' ? 'optics' : 'forecasting');
    if (id === 'twin-manifests' && option === 'commerce') return x.projects.includes('guild-commerce');
    return true;
  }
  function configuration(state, d) {
    if (!d.options.length) return null;
    const selected = state.areaSkills?.configs[d.id] || 'off';
    return { id: d.id, skillId: d.id, areaId: d.areaId, label: d.name, selected, dormant: !rank(state, d.id), options: d.options.map(o => ({ ...o, id: o.value, selected: selected === o.value, disabled: !rank(state, d.id) || !optionAllowed(state, d.id, o.value), action: { type: 'area-skill-config', id: d.id, value: o.value } })) };
  }
  const parameterText = (d, v) => (Math.abs(d.from) <= 1 && Math.abs(d.to) <= 1 ? Math.round(v * 100) + '%' : Number(v.toFixed(2)).toString()) + ' ' + d.unit;
  function view(state) {
    if (!active(state)) return { items: [], ready: [], areas: [], operations: [] };
    const x = state.areaSkills, quantity = state.expedition.batch || 1;
    const items = C.SKILLS.map(d => {
      const owned = x.unlocked.includes(d.id), req = requirements(state, d), ready = !owned && req.every(r => r.met), q = quote(state, d.id, quantity), r = rank(state, d.id), fit = modes(state).filter(n => n <= 10 - r).pop();
      const revealed = owned || ready || d.module === 1 && req[1].met || !!d.project && state.expedition.projects.includes(d.project) || C.SKILLS.some(other => other.areaId === d.areaId && other.module === d.module && x.unlocked.includes(other.id));
      const nextRank = Math.min(10, r + (q.valid ? quantity : 1)), next = d.from + (d.to - d.from) * Math.log(Math.max(1, nextRank)) / Math.log(10), current = value(state, d.id);
      const config = configuration(state, d), impact = [{ metric: 'skill:' + d.id, areaId: d.areaId, label: d.name, current: r ? parameterText(d, current) : 'Inactive', next: parameterText(d, next), currentValue: current, nextValue: next, unit: d.unit }];
      const action = owned ? { type: 'area-skill-buy', id: d.id, count: quantity, quote: q.token } : { type: 'area-skill-unlock', id: d.id };
      return { ...d, id: 'skill:' + d.id, catalogId: 'skill:' + d.id, skillId: d.id, group: 'area', state: owned ? 'learned' : ready ? 'ready' : 'locked', ready, locked: !owned && !ready, owned, visible: !!state.expedition.areas[d.areaId] && revealed, rank: r, level: r, maxLevel: 10, maxed: r === 10,
        quantity, rankAfter: q.rankAfter, disabled: owned ? !q.valid || !q.affordable : !ready, reason: q.reason, requirements: req, dependencies: req,
        cost: Object.entries(q.costs).map(([resource, amount]) => ({ resource, amount, label: N.format(amount) + ' ' + resource, text: N.format(amount) + ' ' + resource })), costs: q.costs,
        description: d.effect, effectText: d.effect, impact, comparison: impact[0].current + ' → ' + impact[0].next, action, unlockAction: { type: 'area-skill-unlock', id: d.id },
        fittingQuantityAction: owned && quantity > 10 - r && fit ? { type: 'expedition-batch', count: fit } : null, configuration: config, configurations: config ? [config] : [], options: config?.options || [], sourceAreas: [d.areaId], targetAreas: [d.areaId] };
    });
    return { items, ready: items.filter(d => d.ready), areas: AREAS.filter(id => state.expedition.areas[id]).map(id => ({ id, ...C.AREA[id], output: count(state, id), foundations: foundations(state, id) })), operations: items.filter(d => d.owned && d.configuration).map(d => d.configuration), configurations: items.filter(d => d.owned && d.configuration).map(d => d.configuration) };
  }
  function sync(state) { if (active(state)) state.areaSkills.output.greenway = Math.max(state.areaSkills.output.greenway, state.trailDeliveries?.deliveries || 0); }
  function reset(state) {
    if (!state.areaSkills) return;
    const x = state.areaSkills;
    IDS.forEach(id => { x.highRanks[id] = Math.max(x.highRanks[id], x.ranks[id]); x.ranks[id] = 0; });
    x.runtime = runtimeInitial(); x.revision += 1;
  }
  function act(state, action) {
    const d = definition(action.id), x = state.areaSkills;
    if (!d || !active(state)) return { ok: false, message: 'Discover this area technique first.' };
    if (action.type === 'area-skill-unlock') {
      if (x.unlocked.includes(d.id) || !eligible(state, d.id)) return { ok: false, message: 'Finish the listed requirements before unlocking this technique.' };
      x.unlocked.push(d.id); x.revision += 1; state.expedition.revision += 1;
      return { ok: true, message: d.name + ' unlocked. Its first rank activates the technique.' };
    }
    if (action.type === 'area-skill-buy') {
      const q = quote(state, d.id, action.count === undefined ? state.expedition.batch || 1 : action.count);
      if (!q.valid || !q.affordable || action.quote !== undefined && action.quote !== q.token) return { ok: false, message: !q.valid || !q.affordable ? q.reason : 'This quote changed. Review its current price.' };
      for (const [key, amount] of Object.entries(q.costs)) state.resources[key] = N.sub(state.resources[key], amount);
      x.ranks[d.id] = q.rankAfter; x.highRanks[d.id] = Math.max(x.highRanks[d.id], q.rankAfter); x.revision += 1; state.expedition.revision += 1;
      return { ok: true, quantity: q.count, message: d.name + ' Lv. ' + q.rankAfter };
    }
    if (action.type === 'area-skill-config') {
      if (!rank(state, d.id) || !d.options.some(o => o.value === action.value) || !optionAllowed(state, d.id, action.value)) return { ok: false, message: 'Buy its first rank and earn that operation before selecting it.' };
      x.configs[d.id] = action.value;
      if (action.value !== 'off' && ['standard-tools', 'spare-parts'].includes(d.id)) x.configs[d.id === 'standard-tools' ? 'spare-parts' : 'standard-tools'] = 'off';
      if (d.id === 'batch-kilns' && action.value === 'off') settleBatch(state);
      x.revision += 1; state.expedition.revision += 1;
      return { ok: true, message: d.name + ' operation saved. Paid voyages retain their funded manifest.' };
    }
    return { ok: false, message: 'Choose a valid area technique action.' };
  }
  function credit(state, key, amount) { state.resources[key] = N.add(state.resources[key], amount); if (key === 'coins') state.lifetime.coins = N.add(state.lifetime.coins, amount); }
  function settleBatch(state) {
    if (!active(state)) return;
    const rt = state.areaSkills.runtime;
    for (const key of RESOURCES) { if (N.cmp(rt.batch[key], 0) > 0) credit(state, key, rt.batch[key]); rt.batch[key] = N.zero(); }
    rt.batchClock = 0;
  }
  // Boundary and settlement methods are called by Progression, never by Core a
  // second time. They use the same pre-step rate snapshot as the physical flow.
  function nextEvent(state, rates) {
    if (!active(state)) return Infinity;
    const rt = state.areaSkills.runtime, a = rates.areas;
    let next = Infinity;
    const limit = n => { if (Number.isFinite(n)) next = Math.min(next, Math.max(EPS, n)); };
    if (enabled(state, 'batch-kilns')) limit(value(state, 'batch-kilns', 5, 3) - rt.batchClock);
    if (enabled(state, 'shift-planning')) limit(value(state, 'shift-planning') - rt.shiftClock);
    if (rank(state, 'resonant-drills') && a.quarry?.actualPicks > EPS) limit((rt.drillRemaining > EPS ? rt.drillRemaining : 100 - rt.drillWork) / a.quarry.actualPicks);
    if (enabled(state, 'template-queue') && a.workshop?.flow > EPS) limit((value(state, 'template-queue') - rt.templateWork) / a.workshop.flow);
    if (enabled(state, 'site-catalogues') && a.ruins?.flow > EPS) limit((value(state, 'site-catalogues') - rt.siteWork) / a.ruins.flow);
    if (rt.hopper > EPS && a.workshop?.hopperDrain > EPS) limit(rt.hopper / a.workshop.hopperDrain);
    for (const id of AREAS.filter(id => id !== 'greenway' && id !== 'harbor')) {
      const flow = outputRate(rates, id); if (!(flow > EPS)) continue;
      // Only teaching a not-yet-learned foundation needs an exact production
      // boundary. Technique readiness does not alter rates. Splitting at every
      // future reveal would perturb released, inactive-technique trajectories.
      const local = state.expedition.areas[id], defs = areaDef(id).tracks;
      const thresholds = C.AREA[id].foundation.filter((_, index) => !local?.learned.includes(defs[index + 1].id));
      for (const target of thresholds) if (state.areaSkills.output[id] < target - EPS) limit((target - state.areaSkills.output[id]) / flow);
    }
    return next;
  }
  function outputRate(rates, id) { const a = rates.areas[id]; return !a ? 0 : id === 'quarry' ? a.actualFurnace : id === 'watchtower' ? a.baseSurvey || a.research : a.flow || 0; }
  function tick(state, seconds, rates) {
    if (!active(state)) return;
    const x = state.areaSkills, rt = x.runtime, a = rates.areas;
    for (const id of AREAS.filter(id => id !== 'greenway' && id !== 'harbor')) x.output[id] = Math.min(1e15, x.output[id] + Math.max(0, outputRate(rates, id)) * seconds);
    if (enabled(state, 'batch-kilns')) {
      rt.batchClock += seconds;
      for (const [key, rate] of Object.entries(a.quarry?.batchRates || {})) rt.batch[key] = N.add(rt.batch[key], N.mul(rate, seconds));
      if (rt.batchClock >= value(state, 'batch-kilns', 5, 3) - EPS) settleBatch(state);
    }
    if (rank(state, 'resonant-drills') && a.quarry) {
      const work = a.quarry.actualPicks * seconds;
      if (rt.drillRemaining > EPS) rt.drillRemaining = Math.max(0, rt.drillRemaining - work);
      else { rt.drillWork += work; if (rt.drillWork >= 100 - EPS) { rt.drillWork = 0; rt.drillRemaining = value(state, 'resonant-drills'); } }
    }
    if (enabled(state, 'shift-planning') && a.quarry) {
      rt.shiftClock += seconds;
      if (rt.shiftClock >= value(state, 'shift-planning') - EPS) { rt.shiftClock = 0; const q = state.expedition.areas.quarry; if (q.learned.includes('geology')) { const pressure = (q.buffers.input + q.buffers.output) / Math.max(EPS, 2 * a.quarry.capacity); if (pressure >= .75) q.choice = 'rich'; else if (pressure <= .25) q.choice = 'balanced'; } }
    }
    if (enabled(state, 'template-queue') && a.workshop) {
      rt.templateWork += a.workshop.flow * seconds;
      if (rt.templateWork >= value(state, 'template-queue') - EPS) { rt.templateWork = 0; const w = state.expedition.areas.workshop, options = ['supplies', 'tools'].concat(state.expedition.areas.watchtower?.learned.includes('optics') ? ['instruments'] : []); w.plans.templates[0] = options[(options.indexOf(w.plans.templates[0]) + 1) % options.length]; }
    }
    if (enabled(state, 'site-catalogues') && a.ruins) {
      rt.siteWork += a.ruins.flow * seconds;
      if (rt.siteWork >= value(state, 'site-catalogues') - EPS) { rt.siteWork = 0; const r = state.expedition.areas.ruins; r.plans.discovery = Object.keys(r.discoveries).sort((u, v) => r.discoveries[u] - r.discoveries[v] || u.localeCompare(v))[0]; }
    }
    if (rt.hopper > 0 && a.workshop?.hopperDrain) rt.hopper = Math.max(0, rt.hopper - a.workshop.hopperDrain * seconds);
    if (rank(state, 'field-notebooks') && a.watchtower && !rates.commissionId) rt.notebook = Math.min((a.watchtower.unboostedResearch || a.watchtower.research) * value(state, 'field-notebooks'), rt.notebook + (a.watchtower.unboostedResearch || a.watchtower.research) * seconds);
    sync(state);
  }
  function fundCommission(state, project) {
    if (!active(state) || !rank(state, 'field-notebooks') || !state.expedition.commission) return;
    const rt = state.areaSkills.runtime, grant = Math.min(rt.notebook, project.work * value(state, 'field-notebooks', .1, .25));
    state.expedition.commission.work += grant; rt.notebook -= grant; rt.lastCommission = project.id;
  }
  function trailSettings(state) {
    if (active(state) && state.areaSkills.runtime.trip) return state.areaSkills.runtime.trip;
    const express = enabled(state, 'express-routes'), tower = state.expedition?.areas?.watchtower;
    return { speed: express ? 1 / (1 - value(state, 'express-routes')) : 1,
      cargo: (express ? .85 : 1) * throughput(value(state, 'cargo-lashing'), tower?.plans?.assignments?.includes('trade') ? value(state, 'logistics-charts') : 0, domesticSupport(state)) };
  }
  function trailArrived(state, arrivals, rates) {
    if (!active(state)) return;
    state.areaSkills.output.greenway = Math.max(state.areaSkills.output.greenway, state.trailDeliveries?.deliveries || 0);
    if (rank(state, 'field-journals')) credit(state, 'maps', Math.max(0, rates.skillMaps || 0) * value(state, 'field-journals') * arrivals);
    const commission = state.expedition.commission;
    if (commission && rank(state, 'relay-runners')) { const p = D.PROJECTS.find(d => d.id === commission.id); commission.work = Math.min(p.work, commission.work + Math.max(0, rates.skillResearch || 0) * value(state, 'relay-runners') * arrivals); }
    state.areaSkills.runtime.trip = null;
  }
  function trailStarted(state) { if (active(state) && !state.areaSkills.runtime.trip) state.areaSkills.runtime.trip = { ...trailSettings(state) }; }
  function voyageKey(voyage) { return [voyage.target, voyage.port, voyage.supplies, voyage.convoy, ...Object.entries(voyage.payout).map(([k, v]) => k + ':' + N.toNumber(v))].join('|'); }
  function voyageLaunched(state, voyage) {
    if (!active(state)) return;
    const rt = state.areaSkills.runtime;
    const receipt = { key: voyageKey(voyage), supplies: voyage.supplies, refund: refund(value(state, 'return-cargo')), domestic: enabled(state, 'exchange-houses') ? value(state, 'exchange-houses') : 0, bonded: enabled(state, 'bonded-routes') };
    if (receipt.refund || receipt.domestic || receipt.bonded) rt.voyageReceipts.push(receipt);
    rt.voyageSequence += 1; rt.convoyReady = false;
  }
  function voyageArrived(state, voyage, nextSupply) {
    if (!active(state)) return;
    const x = state.areaSkills, rt = x.runtime, index = rt.voyageReceipts.findIndex(v => v.key === voyageKey(voyage)), receipt = index < 0 ? null : rt.voyageReceipts.splice(index, 1)[0];
    if (receipt) credit(state, 'provisions', receipt.supplies * receipt.refund);
    x.output.harbor += voyage.convoy || 1;
    rt.convoyReady = N.cmp(state.resources.provisions, nextSupply || 0) >= 0;
  }
  function domesticSupport(state) { return active(state) ? Math.max(0, ...state.areaSkills.runtime.voyageReceipts.map(v => v.domestic)) : 0; }
  return { Content: C, initial, validate, sync, reset, active, rank, value, mode, enabled, throughput, refund, count, foundations, foundationRequirement, requirements, eligible, cost, quote, view, act, configuration, nextEvent, tick, fundCommission, trailSettings, trailStarted, trailArrived, voyageLaunched, voyageArrived, domesticSupport, settleBatch };
});
