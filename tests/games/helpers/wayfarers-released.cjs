'use strict';
const Core=require('../../../js/games/wayfarers-guild/core.js');
// Released economy witnesses use the preserved nested3 engine. This factory
// is local to historical contract tests; it never changes Core.createState or
// the current station game tested by wayfarers-guild-stations.test.cjs.
function createReleasedState(now) {
  const state=Core.createState(now);state.expedition.version=3;
  state.stations=Core.Stations.initial(state);
  return state;
}
module.exports={createReleasedState};
