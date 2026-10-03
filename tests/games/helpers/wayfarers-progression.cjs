'use strict';

const Core = require('../../../js/games/wayfarers-guild/core.js');
const P = require('../../../js/games/wayfarers-guild/progression.js');
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
let cached;
// A deliberately funded behavior/renderer fixture, never a natural pacing
// witness. Areas and branches are created by their actual engine actions.
function mature(options = {}) {
  if (cached && !options.untilProject) return clone(cached);
  const state = Core.createState(0);
  fund(state);
  for (let index = 0; index < 3; index += 1) {
    while (!state.expedition.completed) {
      for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned) {
        if (area.ranks[id] < 20) Core.act(state, { type: 'expedition-buy', areaId, id });
      }
      advance(state, 60);
    }
    if (index < 2) Core.act(state, { type: 'expedition-next' });
  }
  for (const definition of P.Content.PROJECTS) {
    fund(state);
    const started = Core.act(state, { type: 'expedition-development', id: definition.id });
    if (!started.ok) throw new Error(started.message);
    P.tick(state, definition.work / P.rawRates(state).researchRate + 1e-6);
    if (state.expedition.commission) throw new Error('Research fixture did not finish ' + definition.id);
    if (definition.id === options.untilProject) break;
  }
  state.lifetime.refits = 10;
  state.lifetime.charters = 1;
  state.guild.chapterProject.number = 1;
  state.premium.claimedMilestones.push('first-refit', 'first-charter');
  state.expedition.focus = { charges: 3, recharge: 0, active: null, remaining: 0, unlocked: true };
  fund(state);
  for (const [areaId, area] of Object.entries(state.expedition.areas)) for (const id of area.learned) {
    while (area.ranks[id] < 25) Core.act(state, { type: 'expedition-buy', areaId, id });
  }
  const validation = Core.validateState(state);
  if (!validation.valid) throw new Error(validation.errors.join('; '));
  if (!options.untilProject) cached = clone(state);
  return state;
}
module.exports = { Core, P, N, clone, advance, fund, mature };
