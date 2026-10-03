'use strict';

// Manual deterministic balance runner. No granted currency, purchases, ads,
// offline branch choices or automatic prestige. Outside the fast test suite.
const fs = require('node:fs');
const Core = require('../../js/games/wayfarers-guild/core.js');
const P = require('../../js/games/wayfarers-guild/progression.js');
const N = Core.Numbers;
const { claimTiers } = require('./helpers/wayfarers-progression.cjs');
const mode = process.argv[2] || 'reference';
if (!['reference', 'no-focus', 'missed', 'active', 'opening'].includes(mode)) throw new Error('Unknown simulation policy.');
if (mode === 'opening') {
  const state = Core.createState(0), individual = {}, stages = [];
  const production = () => Object.fromEntries(Object.entries(Core.getRates(state).gain).map(([id, value]) => [id, N.toNumber(value)]));
  let firstBuy = null, firstRefit = null, originalRates = null, eligibility = null, heldSince = null, sustained = null;
  for (let time = 0; time < 10800; time += 1) {
    claimTiers(state);
    const expedition = state.expedition;
    if (expedition.completed) {
      if (Core.getRefitPreview(state).available && firstRefit === null) {
        originalRates = production(); firstRefit = time;
        stages.push({ index: expedition.index, time });
        Core.act(state, { type: 'refit' });
        claimTiers(state);
        Core.act(state, { type: 'refit-upgrade', id: 'pace' });
        Core.act(state, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: false });
      } else if (firstRefit !== null && Core.getRefitPreview(state).available) {
        if (!eligibility) eligibility = { seconds: time - firstRefit, rates: production() };
      } else {
        stages.push({ index: expedition.index, time });
        Core.act(state, { type: 'expedition-next' });
      }
    }
    for (const definition of P.Content.PROJECTS) {
      const task = P.developmentTask(state, definition.id);
      if (task.open && !task.done && Object.entries(task.costs).every(([id, value]) => N.cmp(state.resources[id], value) >= 0)) {
        Core.act(state, { type: 'expedition-development', id: definition.id }); break;
      }
    }
    const offers = [];
    for (const [areaId, area] of Object.entries(expedition.areas)) for (const id of area.learned) {
      const quote = P.quote(state, areaId, id, 1);
      if (quote.valid && quote.affordable) offers.push({ areaId, id, quote });
    }
    offers.sort((a, b) => N.cmp(a.quote.costs.coins, b.quote.costs.coins));
    for (const offer of offers.slice(0, 3)) {
      if (Core.act(state, { type: 'expedition-buy', areaId: offer.areaId, id: offer.id }).ok && firstBuy === null) firstBuy = time;
    }
    if (originalRates) {
      const current = production();
      for (const [id, value] of Object.entries(originalRates)) if (value > 0 && current[id] >= value && individual[id] === undefined) individual[id] = time - firstRefit;
      const restored = Object.entries(originalRates).every(([id, value]) => current[id] >= value - 1e-9);
      if (!restored) heldSince = null;
      else if (heldSince === null) heldSince = time;
      if (heldSince !== null && time - heldSince >= 60) {
        sustained = { seconds: heldSince - firstRefit, confirmedSeconds: time - firstRefit, rates: current }; break;
      }
    }
    for (let remaining = 1; remaining > 1e-6;) {
      const result = Core.advance(state, remaining);
      if (!(result.seconds > 0)) throw new Error('The clock stopped.');
      remaining = result.pendingSeconds || 0;
    }
  }
  const result = { policy: 'Continuous first opening; cheapest three affordable learned upgrades each second, no Focus or grants. Invest first Notes in Pace and enable retained standing automation. No second reset. Recovery requires every initially positive canonical production rate to match its pre-Refit value for 60 consecutive seconds.', firstBuy, firstRefit, stages, originalRates, eligibility, individual, sustained, recoveryFraction: sustained ? sustained.seconds / firstRefit : null, validation: Core.validateState(state) };
  if (process.env.WAYFARERS_SIM_OUTPUT) fs.writeFileSync(process.env.WAYFARERS_SIM_OUTPUT, JSON.stringify(result, null, 2));
  else console.log(JSON.stringify(result, null, 2));
  process.exit(result.validation.valid && sustained ? 0 : 1);
}
const useFocus = mode !== 'no-focus';
const state = Core.createState(0), log = [], snapshots = [];
let now = 0, lastPrestige = -1e9, lastProjects = '', firstBuy = null, activeSeconds = 0;

function advance(seconds) {
  while (seconds > 1e-5) {
    const result = Core.advance(state, seconds);
    if (!(result.seconds > 0)) throw new Error('The clock stopped.');
    now += result.seconds;
    seconds = result.pendingSeconds || 0;
  }
  const projects = state.expedition.projects.join();
  if (projects !== lastProjects) {
    for (const id of state.expedition.projects) if (!log.some(event => event.event === 'learned' && event.id === id)) {
      log.push({ event: 'learned', id, day: now / 86400 });
    }
    lastProjects = projects;
  }
}

function decide() {
  // Tier claims are deliberate visit-time decisions, never performed by advance.
  claimTiers(state);
  const expedition = state.expedition;
  // At most one prestige per day. The first three Refits fund Notes, then
  // at least two Refits between eligible Charters balance permanent spending.
  if (now - lastPrestige > 86400 || !state.lifetime.refits) {
    const charter = Core.getCharterPreview(state).available && state.lifetime.refits >= 3 && state.chapter.refits >= 2;
    const type = charter ? 'charter' : Core.getRefitPreview(state).available ? 'refit' : null;
    if (type) {
      const result = Core.act(state, { type });
      if (!result.ok) throw new Error(result.message);
      lastPrestige = now;
      log.push({ event: type, day: now / 86400, count: state.lifetime[type === 'refit' ? 'refits' : 'charters'] });
    }
  }
  if (expedition.completed) Core.act(state, { type: 'expedition-next' });
  if (state.lifetime.refits && !expedition.automation.enabled) {
    Core.act(state, { type: 'expedition-automation', enabled: true, priority: 'balanced', dispatch: true });
  }
  const goal = P.Content.PROJECTS.find(definition => {
    const task = P.developmentTask(state, definition.id);
    return task.open && !task.done;
  });
  if (goal) {
    const task = P.developmentTask(state, goal.id);
    if (Object.entries(task.costs).every(([key, value]) => N.cmp(state.resources[key], value) >= 0)) {
      Core.act(state, { type: 'expedition-development', id: goal.id });
    } else if (state.lifetime.refits) {
      Core.act(state, { type: 'plan-goal', action: { type: 'expedition-development', id: goal.id } });
    }
  }
  const offers = [];
  for (const [areaId, area] of Object.entries(expedition.areas)) for (const id of area.learned) {
    const quote = P.quote(state, areaId, id, 1);
    if (quote.valid && quote.affordable) offers.push({ areaId, id, quote });
  }
  offers.sort((a, b) => N.cmp(a.quote.costs.coins, b.quote.costs.coins));
  for (const offer of offers.slice(0, 3)) {
    if (Core.act(state, { type: 'expedition-buy', areaId: offer.areaId, id: offer.id }).ok && firstBuy === null) firstBuy = now;
  }
  for (const definition of Core.Content.REFIT_UPGRADES) Core.act(state, { type: 'refit-upgrade', id: definition.id });
  for (const definition of Core.Content.LEGACY_UPGRADES) Core.act(state, { type: 'legacy-upgrade', id: definition.id });
  if (useFocus && expedition.focus.charges && !expedition.focus.remaining) {
    Core.act(state, { type: 'expedition-focus', areaId: expedition.commission ? 'watchtower' : expedition.projectArea, id: 'priority' });
  }
}

for (let day = 0; day < 56; day += 1) {
  const visits = mode === 'active' ? 144 : 3;
  for (let visit = 0; visit < visits; visit += 1) {
    if (mode === 'missed' && (day === 12 || day === 13)) continue;
    const start = day * 86400 + visit * (86400 / visits);
    if (now < start) advance(start - now);
    for (let remaining = 600; remaining > 0;) {
      const step = now < 60 ? 1 : Math.min(15, remaining);
      decide(); advance(step);
      activeSeconds += step; remaining -= step;
    }
  }
  const validation = Core.validateState(state);
  if (!validation.valid) throw new Error('Day ' + day + ': ' + validation.errors.join('; '));
  snapshots.push({
    day: day + 1, projects: state.expedition.projects.length,
    areas: Object.keys(state.expedition.areas), refits: state.lifetime.refits,
    charters: state.lifetime.charters, frontier: state.expedition.index,
    ranks: Object.fromEntries(Object.entries(state.expedition.areas).map(([id, area]) => [id, Math.max(...Object.values(area.ranks))])),
    rates: Object.fromEntries(Object.entries(Core.getRates(state).gain).map(([id, value]) => [id, N.toNumber(value)]))
  });
  console.log(JSON.stringify(snapshots[snapshots.length - 1]));
  if (state.expedition.projects.length === P.Content.PROJECTS.length) break;
}
const result = {
  policy: { mode, visitsPerDay: mode === 'active' ? 144 : 3, minutesPerVisit: 10, offlineDecisions: false, focus: useFocus },
  firstBuy, activeSeconds, log, snapshots, state
};
if (process.env.WAYFARERS_SIM_OUTPUT) fs.writeFileSync(process.env.WAYFARERS_SIM_OUTPUT, JSON.stringify(result, null, 2));
else console.log(JSON.stringify(result));
