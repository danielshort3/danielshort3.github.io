'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const movement = require('../../js/games/project-starfall/engine/movement.js');

let assertions = 0;
let clock = 1_800_000_000_000;
const originalNow = Date.now;
const check = (condition, message) => { assertions += 1; assert(condition, message); };

try {
  Date.now = () => clock;
  const source = { id: 'source', index: 1, x: 0, y: 300, w: 200, dropThrough: true };
  const alias = { id: 'alias', index: 2, x: 40, y: 300, w: 200, dropThrough: true };
  const lower = { id: 'lower', index: 3, x: 0, y: 301, w: 240, dropThrough: true };
  const body = { x: 60, y: 250, w: 40, h: 50, groundedPlatformId: source.id, groundedPlatformIndex: source.index };
  Object.assign(body, movement.createDropThroughState(body, 0.28, source, 10));
  check(movement.shouldSkipDropThroughPlatform(body, source, 10.1), 'source floor is ignored during the drop');
  check(movement.shouldSkipDropThroughPlatform(body, alias, 10.1), 'coincident overlapping floor is part of the same drop');
  check(!movement.shouldSkipDropThroughPlatform(body, lower, 10.1), 'a distinct floor only one pixel lower remains landable');
  check(!movement.shouldSkipDropThroughPlatform(body, { ...alias, dropThrough: false }, 10.1), 'solid ground is never ignored');
  check(!movement.shouldSkipDropThroughPlatform(body, { ...alias, x: 300 }, 10.1), 'a separate same-height ledge is not ignored');
  check(!movement.shouldSkipDropThroughPlatform(body, { ...alias, x: 60, w: 40, shape: 'slope', y: 300, y2: 320 }, 10.1),
    'a descending ramp meeting the source at one endpoint remains landable');
  check(!movement.shouldSkipDropThroughPlatform(body, alias, 10.29), 'coincident floor exclusion expires with the original drop');
  const straddledBody = { ...body, x: 180 };
  Object.assign(straddledBody, movement.createDropThroughState(straddledBody, 0.28, source, 10));
  const adjacent = { ...alias, x: 200 };
  check(movement.shouldSkipDropThroughPlatform(straddledBody, adjacent, 10.1), 'same-height adjacent floor under the straddled foot also drops');
  check(!movement.shouldSkipDropThroughPlatform(straddledBody, { ...adjacent, x: 220 }, 10.1), 'exclusion does not extend beyond the original foot span');

  const ramp = { id: 'ramp', index: 4, x: 0, y: 320, y2: 280, w: 200, shape: 'slope', dropThrough: true };
  const rampAlias = { ...ramp, id: 'ramp-alias', index: 5, x: 50, y: 310, y2: 270 };
  Object.assign(body, movement.createDropThroughState(body, 0.28, ramp, 10));
  check(movement.shouldSkipDropThroughPlatform(body, rampAlias, 10.1), 'coincident sloped definitions drop together');
  check(!movement.shouldSkipDropThroughPlatform(body, { ...rampAlias, y: 311, y2: 271 }, 10.1), 'a parallel lower ramp remains landable');
  Object.assign(straddledBody, movement.createDropThroughState(straddledBody, 0.28, ramp, 10));
  check(movement.shouldSkipDropThroughPlatform(straddledBody, { ...rampAlias, x: 200, y: 280, y2: 240 }, 10.1),
    'a straddled collinear ramp extension retains the source gradient');

  const lowerEngine = createProjectStarfallEngine(null, data);
  lowerEngine.runtime.platforms = [source, alias, lower];
  const falling = { x: 60, y: 252, previousY: 250, w: 40, h: 50, vx: 0, vy: 100 };
  Object.assign(falling, movement.createDropThroughState(falling, 0.28, source, clock / 1000));
  lowerEngine.resolvePlatforms(falling);
  check(falling.grounded && falling.groundedPlatformId === lower.id && falling.y + falling.h === 301,
    'real collision resolution lands on the nearest distinct lower floor');
  check(falling.dropThroughSurface === null && falling.dropThroughUntil === 0, 'landing clears all transient drop support');
  lowerEngine.runtime.platforms = [source, adjacent, lower];
  Object.assign(falling, { x: 180, y: 252, previousY: 250, grounded: false, vy: 100 },
    movement.createDropThroughState({ ...falling, x: 180 }, 0.28, source, clock / 1000));
  lowerEngine.resolvePlatforms(falling);
  check(falling.grounded && falling.groundedPlatformId === lower.id && falling.y + falling.h === 301,
    'real collision resolution clears a straddled adjacent seam and catches its lower floor');

  for (const fps of [30, 60, 120]) {
    for (const sourceId of ['rustcoil_switchworks_west_catwalk', 'rustcoil_switchworks_return_deck']) {
      const engine = createProjectStarfallEngine(null, data);
      engine.chooseClass('fighter');
      engine.changeMap('rustcoilRuins', { silent: true });
      const platform = engine.runtime.platforms.find(entry => entry.id === sourceId);
      engine.placePlayerOnRuntimePlatform(platform.index, 2360);
      const player = engine.state.player;
      engine.setInput('down', true);
      engine.setInput('jump', true);
      let landing = null;
      for (let frame = 0; frame < fps * 2; frame += 1) {
        clock += 1000 / fps;
        engine.updatePlayer(1 / fps);
        if (frame === 0) {
          engine.setInput('jump', false);
          engine.setInput('down', false);
          check(!player.grounded && !player.climbing && player.y + player.h > 700,
            `${sourceId}/${fps}FPS: one grounded down+jump clears both coplanar supports`);
        } else if (player.grounded) {
          landing = engine.getBodyPlatform(player);
          break;
        }
      }
      check(landing && landing.id === 'rustcoil_switchworks_conveyor_step' && player.y + player.h === 790,
        `${sourceId}/${fps}FPS: drop lands on the real intermediate lower step without repeated inputs`);
    }
  }
} finally {
  Date.now = originalNow;
}

console.log(`Starfall coincident support drops: ${assertions} assertions preserve lower floors, ramps and real Rustcoil drops at 30/60/120 FPS.`);
