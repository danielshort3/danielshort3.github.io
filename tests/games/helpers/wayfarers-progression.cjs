'use strict';

const Core = require('../../../js/games/wayfarers-guild/core.js');
const P = require('../../../js/games/wayfarers-guild/progression.js');
const Tiers = require('../../../js/games/wayfarers-guild/upgrade-tiers.js');
const N = Core.Numbers;
const clone = value => JSON.parse(JSON.stringify(value));
function advance(state, seconds) {
  while (seconds > 1e-6) {
    const result = Core.advance(state, seconds);
    if (!(result.seconds > 0)) throw new Error('The deterministic clock did not advance.');
    seconds = result.pendingSeconds || 0;
  }
}
function fund(state, amount = 1e14) {
  for (const id of ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps']) state.resources[id] = N.from(amount);
}
// Explicit decisions for funded behavior fixtures, never offline auto-claims.
function claimTiers(state) {
  Tiers.sync(state);
  for (let pass = 0; pass < 100; pass += 1) {
    const ready = Tiers.view(state).ready;
    if (!ready.length) return;
    for (const tier of ready) {
      const result = Core.act(state, tier.unlockAction);
      if (!result.ok) throw new Error(result.message);
    }
  }
  throw new Error('Tier claims did not settle.');
}
let cached;
// A deliberately funded behavior/renderer fixture, never a natural pacing
// witness. Areas and branches are created by their actual engine actions.
function mature(options = {}) {
  if (cached && !options.untilProject) return clone(cached);
  // Historical formula tests explicitly retain the shipped network economy.
  // New station witnesses use wayfarers-stations.cjs instead.
  const state = Core.createState(0);
  state.expedition.version = 3;
  state.stations = Core.Stations.initial(state);
  fund(state);
  for (let index = 0; index < 3; index += 1) {
    while (!state.expedition.completed || index < 2 && !P.canAdvance(state)) {
      claimTiers(state);
      for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned) {
        if (area.ranks[id] < 20) Core.act(state, { type: 'expedition-buy', areaId, id });
      }
      advance(state, 60);
    }
    if (index < 2) Core.act(state, { type: 'expedition-next' });
  }
  for (const definition of P.Content.PROJECTS) {
    fund(state);
    claimTiers(state);
    const started = Core.act(state, { type: 'expedition-development', id: definition.id });
    if (!started.ok) throw new Error(started.message);
    P.tick(state, definition.work / P.rawRates(state).researchRate + 1e-6);
    claimTiers(state);
    if (state.expedition.commission) throw new Error('Research fixture did not finish ' + definition.id);
    if (definition.id === options.untilProject) break;
  }
  state.lifetime.refits = 10;
  state.lifetime.charters = 1;
  state.guild.chapterProject.number = 1;
  state.premium.claimedMilestones.push('first-refit', 'first-charter');
  state.expedition.focus = { charges: 3, recharge: 0, active: null, remaining: 0, unlocked: true };
  fund(state);
  claimTiers(state);
  // New areas learn foundations from their own actual production. High funding
  // buys ranks, but cannot stand in for completed voyages or recovered finds.
  for (let pass = 0; pass < 720; pass += 1) {
    for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned.slice()) {
      while (area.ranks[id] < 25) {
        claimTiers(state);
        if (!P.quote(state, areaId, id, 1).valid) break;
        const bought = Core.act(state, { type: 'expedition-buy', areaId, id });
        if (!bought.ok) throw new Error('Mature fixture purchase ' + areaId + '/' + id + ': ' + bought.message);
      }
    }
    const established = P.Content.AREAS.filter(def => state.expedition.areas[def.id]).every(def => def.tracks.slice(0, 3).every(track => state.expedition.areas[def.id].ranks[track.id] >= 25) && state.expedition.areas[def.id].learned.every(id => state.expedition.areas[def.id].ranks[id] >= 25));
    if (established) break;
    advance(state, 60); claimTiers(state);
    if (pass === 719) throw new Error('Mature fixture did not establish every area foundation.');
  }
  const validation = Core.validateState(state);
  if (!validation.valid) throw new Error(validation.errors.join('; '));
  if (!options.untilProject) cached = clone(state);
  return state;
}
module.exports = { Core, P, N, clone, advance, fund, mature, claimTiers };
