'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

let cases = 0;
for (const fps of [30, 60, 120]) {
  for (const mapId of ['greenrootMeadow', 'thornpathThicket', 'rustcoilRuins', 'glacierSpine']) {
    for (const action of ['drop', 'climb-down', 'climbing-jump']) {
      const engine = createProjectStarfallEngine(null, data);
      engine.state.player.classId = 'fighter';
      engine.changeMap(mapId, { silent: true });
      const ladder = mapId === 'thornpathThicket'
        ? engine.runtime.climbables.find((entry) => entry.id === 'thornpathThicket_vine_relay_mid')
        : engine.runtime.climbables.find((entry) => engine.runtime.platforms[entry.topPlatformIndex]?.dropThrough && entry.h >= 100);
      assert(ladder, `${mapId}: fixture has a drop-through ladder top`);
      const player = engine.state.player;
      engine.placePlayerOnRuntimePlatform(ladder.topPlatformIndex, ladder.x + ladder.w / 2 - player.w / 2);
      const startY = player.y;
      if (action === 'climbing-jump') {
        Object.assign(player, { grounded: false, groundedPlatformId: '', groundedPlatformIndex: -1,
          climbing: true, climbableId: ladder.id, y: ladder.y + ladder.h / 2 - player.h / 2 });
      }
      engine.setInput('down', true);
      if (action !== 'climb-down') engine.setInput('jump', true);
      engine.updatePlayer(1 / fps);
      const context = `${mapId}/${action}/${fps}FPS`;
      if (action === 'drop') {
        assert(player.vy > 0 && player.y > startY, `${context}: grounded down+jump must move downward`);
        assert(!player.climbing && player.dropJumpConsumed, `${context}: drop wins before automatic ladder mounting`);
        assert.strictEqual(player.dropThroughPlatformId, ladder.topPlatformId, `${context}: the source floor identity is recorded`);
      } else if (action === 'climb-down') {
        assert(player.climbing && player.climbableId === ladder.id, `${context}: down alone still mounts the ladder`);
      } else {
        assert(!player.climbing && player.vy < 0, `${context}: an already climbing jump still exits upward`);
      }
      cases += 1;
    }
  }
}
console.log(`Starfall ladder input: ${cases} real movement cases preserve grounded drops, ladder descent and climbing jumps.`);
