'use strict';
const H = require('./wayfarers-progression.cjs');
const Collection = require('../../../js/games/wayfarers-guild/collections.js');
const { Core, N } = H;
const act = (state, action) => { const result = Core.act(state, action); if (!result.ok) throw new Error(result.message); return result; };
// Explicitly funded diagnostic fixtures, not natural pacing evidence. Permanent
// projects, old Guild rooms, starter claims and scroll outcomes use real actions.
function advanced() {
  const state = H.mature();
  for (let i = 0; state.lifetime.highestRoute < 7 && i < 500; i += 1) {
    if (state.expedition.completed) act(state, { type: 'expedition-next' });
    H.advance(state, 60);
  }
  H.fund(state); H.claimTiers(state);
  if (!state.rooms.includes('study')) act(state, { type: 'project', id: 'study-foundation' });
  H.claimTiers(state);
  if (!state.rooms.includes('cartography')) act(state, { type: 'project', id: 'maps-foundation' });
  H.claimTiers(state);
  state.resources.crests = N.from(100); state.resources.notes = N.from(100);
  act(state, { type: 'capability', id: 'loadouts' });
  // Diagnostic ownership of one authored relic removes probabilistic fixture
  // startup; its equipment and kit still use the actual canonical controls.
  if (!state.luck.owned.includes('living-crucible')) state.luck.owned.push('living-crucible');
  act(state, { type: 'relic-equip', id: 'living-crucible' });
  act(state, { type: 'collection-unlock', kind: 'cards' });
  act(state, { type: 'collection-unlock', kind: 'equipment' });
  state.collection.scrolls.bold += 20;
  for (let i = 0; !state.collection.gear['trail-boots'].failed && i < 6; i += 1) {
    const action = { type: 'gear-scroll', id: 'trail-boots', scrollId: 'bold' };
    action.quote = Collection.quote(state, action).token; act(state, action);
  }
  if (!state.collection.gear['trail-boots'].failed) throw new Error('Diagnostic seed unexpectedly produced six successful Bold attempts.');
  H.advance(state, 7200); H.fund(state); H.claimTiers(state);
  const result = Core.validateState(state); if (!result.valid) throw new Error(result.errors.join('; '));
  return state;
}
module.exports = { advanced, projects: () => H.mature({ untilProject: 'rail-network' }) };
