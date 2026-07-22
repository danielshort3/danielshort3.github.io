'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const movement = require('../js/games/project-starfall/engine/movement.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function assertNear(actual, expected, tolerance, message) {
  assert(Math.abs(actual - expected) <= tolerance,
    `${message}: expected ${expected}, received ${actual}`);
}

function simulateGroundFriction(fps, profile) {
  const player = { vx: 300, grounded: true };
  const frameCount = Math.round(fps * 0.1);
  const options = {
    groundAcceleration: 8.4,
    airAcceleration: 3.6,
    groundFrictionActive: 0.94,
    groundFrictionIdle: 0.76,
    airFriction: 0.96
  };
  for (let frame = 0; frame < frameCount; frame += 1) {
    const step = movement.getHorizontalMovementStepPlan(
      player,
      { speed: 280 },
      0,
      1 / fps,
      false,
      profile,
      options
    );
    player.vx = step.vx;
  }
  return player.vx;
}

assertNear(
  movement.getFrameRateIndependentRetention(0.76, 1 / 60, 60),
  0.76,
  1e-12,
  'one 60Hz frame should preserve the authored normal-ground retention'
);
assertNear(
  movement.getFrameRateIndependentBlendAlpha(0.08, 1 / 120, 60),
  1 - Math.sqrt(0.92),
  1e-12,
  'a 120Hz camera step should compose to the authored 60Hz blend'
);

const iceProfile = movement.getMapMovementProfile({ movementProfile: 'ice' });
const normalFrictionResults = [60, 120, 240].map((fps) => simulateGroundFriction(fps, null));
const iceFrictionResults = [60, 120, 240].map((fps) => simulateGroundFriction(fps, iceProfile));
normalFrictionResults.slice(1).forEach((value) => {
  assertNear(value, normalFrictionResults[0], 0.05,
    'normal-ground friction should retain the same velocity over equal wall time');
});
iceFrictionResults.slice(1).forEach((value) => {
  assertNear(value, iceFrictionResults[0], 0.05,
    'ice friction should retain the same velocity over equal wall time');
});
assert(iceFrictionResults[0] > normalFrictionResults[0] * 3,
  'ice should remain materially more momentum-preserving than normal ground');

function createEngine(classId) {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, {
    classId: classId || 'fighter',
    advancedClassId: '',
    x: 140,
    y: 360,
    w: 40,
    h: 74,
    facing: 1,
    grounded: true,
    vx: 0,
    vy: 0,
    hp: 1000,
    resource: 0,
    shield: 0,
    invulnerableUntil: 0,
    attackTimer: 0,
    combatLockUntil: 0,
    movementLockUntil: 0,
    climbing: false,
    climbMoving: false,
    climbableId: '',
    mobility: null
  });
  engine.resetPlayerFeelRuntime('test-setup');
  engine.enemies = [];
  engine.projectiles = [];
  engine.effects = [];
  engine.testAudioCues = [];
  engine.playAudioCue = (cue) => {
    engine.testAudioCues.push(cue);
    return true;
  };
  return engine;
}

function createEnemy() {
  const enemyData = data.ENEMIES[0];
  return {
    uid: 'player-feel-enemy',
    id: enemyData.id,
    name: enemyData.name,
    data: enemyData,
    x: 176,
    y: 372,
    w: 46,
    h: 46,
    hp: 10000,
    maxHp: 10000,
    defense: 0,
    level: 1,
    vx: 0,
    vy: 0,
    attackCd: 0,
    telegraph: 0,
    marked: 0,
    burning: 0,
    slowed: 0,
    staggered: 0,
    weakPoint: 0,
    packMarked: 0,
    runeLinked: 0,
    affixIds: [],
    facing: -1
  };
}

const originalRandom = Math.random;
try {
  Math.random = () => 0.99;

  const meleeEngine = createEngine('fighter');
  const meleeEnemy = createEnemy();
  meleeEngine.enemies = [meleeEnemy];
  const meleeStartHp = meleeEnemy.hp;
  const meleeStartResource = meleeEngine.state.player.resource;
  assert.strictEqual(meleeEngine.basicAttack(), true,
    'melee attack input should begin its authored windup');
  const meleePending = meleeEngine.pendingPlayerAttack;
  assert(meleePending, 'melee attack input should create one pending release');
  assertNear(meleePending.releaseAt - meleePending.startedAt, 0.09, 1e-6,
    'melee contact should release about 90ms into the animation');
  assertNear(meleeEngine.state.player.attackTimer - meleePending.startedAt, 0.38, 1e-6,
    'staging should preserve the authored melee start-to-start cadence');
  const meleeCooldownAtStart = meleeEngine.state.player.attackTimer;
  assert.strictEqual(meleeEnemy.hp, meleeStartHp,
    'melee damage should not resolve before the authored release');
  assert.strictEqual(meleeEngine.effects.length, 0,
    'melee FX should not play before contact');
  assert.deepStrictEqual(meleeEngine.testAudioCues, [],
    'melee audio should not play before contact');
  assert.strictEqual(
    meleeEngine.advancePendingBasicAttack(meleePending.releaseAt - 0.001),
    false,
    'melee contact should remain pending before its release timestamp'
  );
  assert.strictEqual(meleeEngine.advancePendingBasicAttack(meleePending.releaseAt), true,
    'melee contact should resolve at its release timestamp');
  assert(meleeEnemy.hp < meleeStartHp, 'melee release should damage an overlapping enemy');
  assert(meleeEngine.effects.length > 0, 'melee release should emit its contact FX');
  assert.deepStrictEqual(meleeEngine.testAudioCues, ['attack'],
    'melee release should emit attack audio exactly once');
  assert(meleeEngine.state.player.resource > meleeStartResource,
    'melee resource gain should happen at release');
  assert.strictEqual(meleeEngine.state.player.attackTimer, meleeCooldownAtStart,
    'release should not extend the start-to-start attack cadence');
  const meleeHpAfterRelease = meleeEnemy.hp;
  const meleeEffectCount = meleeEngine.effects.length;
  assert.strictEqual(meleeEngine.advancePendingBasicAttack(meleePending.releaseAt + 1), false,
    'a released melee attack should not resolve twice');
  assert.strictEqual(meleeEnemy.hp, meleeHpAfterRelease,
    'a second release check should not apply duplicate melee damage');
  assert.strictEqual(meleeEngine.effects.length, meleeEffectCount,
    'a second release check should not duplicate melee FX');

  const rangedEngine = createEngine('archer');
  assert.strictEqual(rangedEngine.basicAttack(), true,
    'ranged attack input should begin its authored windup');
  const rangedPending = rangedEngine.pendingPlayerAttack;
  assert(rangedPending, 'ranged attack input should create one pending release');
  assertNear(rangedPending.releaseAt - rangedPending.startedAt, 0.13, 1e-6,
    'ranged projectiles should release about 130ms into the animation');
  assertNear(rangedEngine.state.player.attackTimer - rangedPending.startedAt, 0.48, 1e-6,
    'staging should preserve the authored ranged start-to-start cadence');
  const rangedCooldownAtStart = rangedEngine.state.player.attackTimer;
  assert.strictEqual(rangedEngine.projectiles.length, 0,
    'ranged projectiles should not exist before release');
  assert.strictEqual(rangedEngine.effects.length, 0,
    'ranged release FX should not play during windup');
  assert.deepStrictEqual(rangedEngine.testAudioCues, [],
    'ranged audio should not play during windup');
  assert.strictEqual(rangedEngine.advancePendingBasicAttack(rangedPending.releaseAt), true,
    'ranged attack should release at its authored timestamp');
  assert.strictEqual(rangedEngine.projectiles.length, 1,
    'ranged release should create exactly one projectile');
  assert(rangedEngine.effects.length > 0, 'ranged release should emit its authored FX');
  assert.deepStrictEqual(rangedEngine.testAudioCues, ['attack'],
    'ranged release should emit attack audio exactly once');
  assert.strictEqual(rangedEngine.state.player.attackTimer, rangedCooldownAtStart,
    'ranged release should not extend the start-to-start cadence');
  assert.strictEqual(rangedEngine.advancePendingBasicAttack(rangedPending.releaseAt + 1), false,
    'a released ranged attack should not create a duplicate projectile');
  assert.strictEqual(rangedEngine.projectiles.length, 1,
    'a ranged attack should release exactly once');
} finally {
  Math.random = originalRandom;
}

const hitCancelEngine = createEngine('fighter');
assert.strictEqual(hitCancelEngine.basicAttack(), true,
  'player-hit cancellation setup should stage an attack');
assert(hitCancelEngine.pendingPlayerAttack, 'player-hit cancellation setup should have a pending release');
hitCancelEngine.damagePlayer(40, 'test strike');
assert.strictEqual(hitCancelEngine.pendingPlayerAttack, null,
  'taking a real hit should cancel a pending basic attack');
assert.strictEqual(hitCancelEngine.playerFeel.lastCancelReason, 'player-hit',
  'player-hit cancellation should record its reason');
assert.strictEqual(hitCancelEngine.advancePendingBasicAttack(Infinity), false,
  'a player-hit-cancelled attack should never release later');

const mapCancelEngine = createEngine('archer');
assert.strictEqual(mapCancelEngine.basicAttack(), true,
  'map cancellation setup should stage an attack');
const destinationMap = data.MAPS.find((map) => map.id !== mapCancelEngine.state.mapId);
assert(destinationMap, 'the map cancellation fixture needs a second map');
assert.strictEqual(mapCancelEngine.changeMap(destinationMap.id, { silent: true }), true,
  'the map cancellation fixture should complete a map change');
assert.strictEqual(mapCancelEngine.pendingPlayerAttack, null,
  'map changes should cancel pending attack releases');
assert.strictEqual(mapCancelEngine.playerFeel.lastCancelReason, 'map-change',
  'map cancellation should record its reason');
assert.strictEqual(mapCancelEngine.projectiles.length, 0,
  'a map-cancelled ranged attack should never create a projectile');

function placePlayerOnGround(engine) {
  const player = engine.state.player;
  const ground = engine.runtime.platforms[0];
  Object.assign(player, {
    x: ground.x + 100,
    y: ground.y - player.h,
    previousY: ground.y - player.h,
    vx: 0,
    vy: 0,
    grounded: true,
    groundedPlatformId: ground.id,
    groundedPlatformIndex: ground.index,
    climbing: false,
    climbMoving: false,
    climbableId: '',
    mobility: null,
    movementLockUntil: 0,
    combatLockUntil: 0
  });
  return ground;
}

const heldJumpEngine = createEngine('fighter');
placePlayerOnGround(heldJumpEngine);
heldJumpEngine.setInput('jump', true);
assert(heldJumpEngine.playerFeel.jumpBufferUntil > 0,
  'a rising jump edge should queue input');
heldJumpEngine.updatePlayer(1 / 60);
assert(heldJumpEngine.state.player.vy < 0 && !heldJumpEngine.state.player.grounded,
  'a queued grounded jump should launch the player');
assert.strictEqual(heldJumpEngine.playerFeel.jumpBufferUntil, 0,
  'a launched jump should consume its buffer');
placePlayerOnGround(heldJumpEngine);
heldJumpEngine.updatePlayer(1 / 60);
assert.strictEqual(heldJumpEngine.state.player.grounded, true,
  'holding jump through a later landing should not relaunch the player');
assert.strictEqual(heldJumpEngine.state.player.vy, 0,
  'a held jump without a new edge should settle on the platform');
heldJumpEngine.setInput('jump', true);
heldJumpEngine.updatePlayer(1 / 60);
assert.strictEqual(heldJumpEngine.state.player.grounded, true,
  'repeated keydown while held should not requeue jump input');

const coyoteEngine = createEngine('fighter');
placePlayerOnGround(coyoteEngine);
coyoteEngine.updatePlayer(1 / 240);
assert(coyoteEngine.playerFeel.coyoteUntil > 0,
  'ground contact should open a coyote-time window');
Object.assign(coyoteEngine.state.player, {
  y: 520,
  previousY: 520,
  grounded: false,
  groundedPlatformId: '',
  groundedPlatformIndex: -1,
  vy: 0
});
coyoteEngine.setInput('jump', true);
coyoteEngine.updatePlayer(1 / 60);
assert(coyoteEngine.state.player.vy < 0,
  'a rising jump edge just after leaving a ledge should use coyote time');
assert.strictEqual(coyoteEngine.playerFeel.coyoteUntil, 0,
  'a coyote jump should consume the window exactly once');

const expiredCoyoteEngine = createEngine('fighter');
Object.assign(expiredCoyoteEngine.state.player, {
  y: 520,
  previousY: 520,
  grounded: false,
  groundedPlatformId: '',
  groundedPlatformIndex: -1,
  vy: 0
});
expiredCoyoteEngine.playerFeel.coyoteUntil = 0;
expiredCoyoteEngine.setInput('jump', true);
expiredCoyoteEngine.updatePlayer(1 / 60);
assert(expiredCoyoteEngine.state.player.vy > 0,
  'an airborne jump edge outside coyote time should not launch the player');

const landingBufferEngine = createEngine('fighter');
const landingGround = landingBufferEngine.runtime.platforms[0];
const landingPlayer = landingBufferEngine.state.player;
Object.assign(landingPlayer, {
  x: landingGround.x + 100,
  y: landingGround.y - landingPlayer.h - 2,
  previousY: landingGround.y - landingPlayer.h - 2,
  vx: 0,
  vy: 200,
  grounded: false,
  groundedPlatformId: '',
  groundedPlatformIndex: -1
});
landingBufferEngine.playerFeel.coyoteUntil = 0;
const landingQueueTime = Date.now() / 1000;
const queuedLandingJumpUntil = landingBufferEngine.queueJumpInput(landingQueueTime);
assertNear(queuedLandingJumpUntil - landingQueueTime, 0.12, 1e-6,
  'the landing buffer should preserve a 120ms authored window');
landingBufferEngine.updatePlayer(1 / 60);
assert(landingPlayer.vy < 0 && !landingPlayer.grounded,
  'a jump queued just before landing should launch on contact');
assert.strictEqual(landingBufferEngine.playerFeel.jumpBufferUntil, 0,
  'a landing-buffer jump should be consumed exactly once');

function runCameraForEqualTime(fps) {
  const engine = createEngine('fighter');
  engine.runtime.worldWidth = 5000;
  engine.runtime.worldHeight = 3000;
  engine.state.player.x = 2500;
  engine.state.player.y = 1500;
  engine.camera.x = 0;
  engine.camera.y = 0;
  const frameCount = Math.round(fps * 0.25);
  for (let frame = 0; frame < frameCount; frame += 1) engine.updateCamera(1 / fps);
  return { x: engine.camera.x, y: engine.camera.y };
}

const camera60 = runCameraForEqualTime(60);
const camera120 = runCameraForEqualTime(120);
const camera240 = runCameraForEqualTime(240);
assertNear(camera120.x, camera60.x, 1e-8,
  'horizontal camera response should match at 60Hz and 120Hz over equal wall time');
assertNear(camera240.x, camera60.x, 1e-8,
  'horizontal camera response should match at 60Hz and 240Hz over equal wall time');
assertNear(camera120.y, camera60.y, 1e-8,
  'vertical camera response should match at 60Hz and 120Hz over equal wall time');
assertNear(camera240.y, camera60.y, 1e-8,
  'vertical camera response should match at 60Hz and 240Hz over equal wall time');

console.log('Project Starfall player-feel tests passed.');
