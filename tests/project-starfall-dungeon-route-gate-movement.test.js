'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const movement = require('../js/games/project-starfall/engine/movement.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const clampToGate = movement.getDungeonRouteGateXClampPlan;

function getFlow(game) {
  return game.state.dungeons.currentRun.encounterFlow;
}

function getActiveBeatEnemies(game) {
  const flow = getFlow(game);
  return game.enemies.filter((enemy) =>
    enemy && enemy.hp > 0 && enemy.dungeonBeatId === flow.activeBeatId
  );
}

function createBrambleGame() {
  const game = createProjectStarfallEngine(null, Data);
  assert(game.chooseClass('fighter'));
  game.state.player.level = 25;
  game.state.player.advancedClassId = 'guardian';
  game.toastMessages = [];
  game.toast = (message) => {
    game.toastMessages.push(String(message || ''));
    return true;
  };
  game.recordProgressEvent = () => false;
  game.syncRosterUnlocks = () => false;
  game.awardProgressReward = () => true;
  assert(game.startDungeon('bramble_depths'));
  return game;
}

assert.strictEqual(typeof clampToGate, 'function',
  'the movement module should export the route-gate clamp used by every movement path');

assert.deepStrictEqual(
  clampToGate(1160, 40, 1200, 12),
  { x: 1148, blocked: true },
  'the first Bramble seal should stop a body before x=1200'
);
assert.deepStrictEqual(
  clampToGate(1148, 40, 1200, 12),
  { x: 1148, blocked: false },
  'resting exactly at the first padded boundary should not report another collision'
);
assert.deepStrictEqual(
  clampToGate(3160, 40, 3200, 12),
  { x: 3148, blocked: true },
  'the second Bramble seal should stop a body before x=3200'
);
assert.deepStrictEqual(
  clampToGate(3260, 40, 0, 12),
  { x: 3260, blocked: false },
  'gate x=0 should represent an open route'
);
assert.deepStrictEqual(
  clampToGate(3260, 40, null, 12),
  { x: 3260, blocked: false },
  'a missing route gate should leave movement unchanged'
);
assert.deepStrictEqual(
  clampToGate('3260', '40', '3200', '12'),
  { x: 3148, blocked: true },
  'finite serialized coordinates should normalize consistently'
);
assert.deepStrictEqual(
  clampToGate(520, -40, 500, -12),
  { x: 500, blocked: true },
  'negative widths and padding should normalize to zero'
);
assert.deepStrictEqual(
  clampToGate(800, 40, null, 12, { minX: 10, maxX: 700 }),
  { x: 700, blocked: false },
  'world bounds should normalize movement without inventing a route-gate collision'
);
assert.deepStrictEqual(
  clampToGate(Number.NaN, Number.POSITIVE_INFINITY, 1200, Number.NaN, { minX: 10, maxX: 3400 }),
  { x: 10, blocked: false },
  'non-finite movement inputs should fall back to safe finite values'
);

const game = createBrambleGame();
assert.strictEqual(game.getActiveDungeonRouteGateX(), 1200);
assert.strictEqual(game.effects.filter((effect) => effect.dungeonRouteGate).length, 1,
  'the first active beat should expose one visible route seal');

const restoredGateGame = createProjectStarfallEngine(null, Data);
game.state.player.x = 1280;
assert(restoredGateGame.restore(game.serialize()));
assert.strictEqual(
  restoredGateGame.state.player.x,
  1200 - restoredGateGame.state.player.w - 18,
  'restore should immediately move an out-of-bounds saved position behind the active seal'
);

let gateX = game.getActiveDungeonRouteGateX();
let gateLimit = gateX - game.state.player.w - 18;
game.state.player.x = gateX + 80;
game.updatePlayer(0);
assert.strictEqual(game.state.player.x, gateLimit,
  'normal movement finalization should clamp a restored position behind the first seal');

game.state.player.x = gateLimit - 8;
game.state.player.mobility = {
  direction: 1,
  speed: 1400,
  remaining: 400,
  preserveMomentumUntilGround: false
};
game.updatePlayer(0.2);
assert(game.state.player.x <= gateLimit && !game.state.player.mobility,
  'dash movement should stop and finish when it crosses the first seal');

game.state.player.x = gateLimit - 8;
game.state.player.facing = 1;
game.applySkillMovement({
  id: 'bramble_route_gate_test_blink',
  owner: 'mage',
  movementEffect: { mode: 'blink', distance: 500, duration: 0 }
}, 1, game.getStats(), null);
assert(game.state.player.x <= gateLimit,
  'Blink should use the same route-gate clamp as normal and dash movement');

game.nextDungeonRouteGateToastAt = 0;
game.toastMessages.length = 0;
game.applyDungeonRouteGateClamp(gateX + 100);
game.applyDungeonRouteGateClamp(gateX + 100);
assert.strictEqual(game.toastMessages.length, 1,
  'route-seal collision feedback should be throttled');

getActiveBeatEnemies(game).slice().forEach((enemy) => game.defeatEnemy(enemy));
assert.strictEqual(getFlow(game).activeBeatIndex, 1);
assert.strictEqual(game.getActiveDungeonRouteGateX(), 3200);
assert.strictEqual(game.effects.filter((effect) => effect.dungeonRouteGate).length, 1,
  'clearing the first beat should move the visible seal to the second gate');

gateX = game.getActiveDungeonRouteGateX();
gateLimit = gateX - game.state.player.w - 18;
const courtRope = game.runtime.climbables.find((climbable) => climbable.id === 'brambleDepths_rope_6');
assert(courtRope, 'the Court Gate ascent should retain its authored rope');
game.state.player.x = courtRope.x + courtRope.w / 2 - game.state.player.w / 2;
game.state.player.y = courtRope.y + courtRope.h / 2 - game.state.player.h / 2;
game.state.player.climbing = true;
game.state.player.climbableId = courtRope.id;
game.state.player.grounded = false;
game.input.up = true;
game.updatePlayer(1 / 60);
game.input.up = false;
assert(game.state.player.x <= gateLimit && !game.state.player.climbing,
  'climbing into the Court Gate seal should stop before the boundary and cleanly dismount');
game.state.player.climbing = false;
game.state.player.climbableId = '';

game.state.player.x = gateX + 80;
game.updatePlayer(0);
assert.strictEqual(game.state.player.x, gateLimit,
  'normal movement finalization should clamp restored positions behind the second seal');

getActiveBeatEnemies(game).slice().forEach((enemy) => game.defeatEnemy(enemy));
assert.strictEqual(getFlow(game).activeBeatIndex, 2);
assert.strictEqual(game.getActiveDungeonRouteGateX(), 0);
assert.strictEqual(game.effects.filter((effect) => effect.dungeonRouteGate).length, 0,
  'the route seal should disappear when Brambleking is revealed');
assert.deepStrictEqual(
  game.applyDungeonRouteGateClamp(3260),
  { x: 3260, blocked: false },
  'the former x=3200 seal should no longer constrain any movement path'
);

console.log('Project Starfall Bramble route-gate movement tests passed.');
