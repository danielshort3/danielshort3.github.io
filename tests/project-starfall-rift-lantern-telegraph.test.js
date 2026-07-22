'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function createEnemyFixture(enemyData) {
  return {
    uid: `test-${enemyData.id}`,
    id: enemyData.id,
    name: enemyData.name,
    data: enemyData,
    level: 1,
    x: 420,
    y: 300,
    w: 46,
    h: 46,
    vx: 0,
    vy: 0,
    facing: -1,
    grounded: false,
    hp: 100,
    maxHp: 100,
    defense: 0,
    damage: 12,
    speedScale: 1,
    attackCd: 0,
    telegraph: 0,
    state: 'idle',
    marked: 0,
    burning: 0,
    slowed: 0,
    staggered: 0,
    weakPoint: 0,
    packMarked: 0,
    runeLinked: 0,
    eliteAffixIds: [],
    animationState: 'idle',
    animationStartedAt: 0,
    animationDuration: 0,
    animationLoop: true,
    actionLockUntil: 0,
    hpBarUntil: 0,
    removeAt: 0
  };
}

const engine = createProjectStarfallEngine(null, data);
Object.assign(engine.state.player, {
  classId: 'fighter',
  x: 120,
  y: 320,
  w: 40,
  h: 74,
  hp: 200,
  maxHp: 200,
  facing: 1,
  grounded: true
});
engine.runtime.worldWidth = 1280;

const playerTarget = { kind: 'player', id: 'player', actor: engine.state.player };
engine.getCombatCharacters = () => [playerTarget];
engine.getEnemyAggroTarget = () => playerTarget;
engine.getPassiveOffscreenEnemyUpdateStride = () => 1;
engine.shouldDeferPassiveOffscreenEnemyUpdate = () => false;
engine.updateEnemyClimbing = () => false;
engine.updateEnemyPlatformJump = () => false;
engine.resolvePlatforms = () => false;
engine.recoverFallenBodyThroughTop = () => false;
engine.wakeEnemyOnContact = () => false;
engine.updateEnemyStuckState = () => false;
engine.buildEnemySpatialIndex = () => null;
engine.separateEnemies = () => false;
engine.refreshEnemySpatialIndexCenters = () => false;

const riftLanternData = data.ENEMIES.find((enemy) => enemy.id === 'riftLantern');
assert(riftLanternData, 'the Rift Lantern enemy definition should exist');
const riftLantern = createEnemyFixture(riftLanternData);
engine.enemies = [riftLantern];
engine.projectiles = [];

engine.updateEnemies(0.01);
assert.strictEqual(engine.projectiles.length, 0,
  'Rift Lantern should not create a projectile when its attack becomes ready');
assert.strictEqual(riftLantern.state, 'riftLanternWindup',
  'Rift Lantern should enter an explicit pre-fire windup state');
assert(riftLantern.telegraph > 0.5,
  'the pre-fire state should expose a clear half-second telegraph window');
const windupSnapshot = engine.getEnemyRenderSnapshot(riftLantern, null);
assert(windupSnapshot.telegraph > 0,
  'the render snapshot should expose the active pre-fire telegraph');
assert.strictEqual(windupSnapshot.animationState, 'telegraph',
  'the render snapshot should remain on the telegraph animation before firing');

engine.updateEnemies(0.25);
engine.updateEnemies(0.25);
assert.strictEqual(engine.projectiles.length, 0,
  'Rift Lantern should still have no projectile before the complete windup elapses');

engine.updateEnemies(0.03);
assert.strictEqual(engine.projectiles.length, 1,
  'Rift Lantern should create exactly one projectile after the windup elapses');
assert.strictEqual(engine.projectiles[0].sourceEnemyId, 'riftLantern');
assert.strictEqual(riftLantern.telegraph, 0,
  'the pre-fire telegraph should clear when the committed shot launches');
assert.strictEqual(riftLantern.state, 'idle');
assert.strictEqual(riftLantern.attackCd, 2.1,
  'the Rift Lantern special windup should retain its existing post-fire cadence');

engine.updateEnemies(0.1);
assert.strictEqual(engine.projectiles.length, 1,
  'the post-fire cooldown should prevent duplicate projectiles on later updates');

const genericFlyerData = data.ENEMIES.find((enemy) => enemy.behavior === 'flyer' && enemy.id !== 'riftLantern');
if (genericFlyerData) {
  engine.projectiles = [];
  const genericFlyer = createEnemyFixture(genericFlyerData);
  assert.strictEqual(engine.enemyProjectile(genericFlyer, 'firebolt', playerTarget), true);
  assert.strictEqual(engine.projectiles.length, 0,
    'generic projectile enemies should begin with a readable pre-fire window');
  assert(genericFlyer.telegraph > 0,
    'generic projectile enemies should expose their warning before release');
  assert(genericFlyer.pendingAttack && genericFlyer.pendingAttack.kind === 'projectile');
  genericFlyer.telegraph = 0;
  assert.strictEqual(engine.resolveEnemyPendingAttack(genericFlyer, [playerTarget]), true);
  assert.strictEqual(engine.projectiles.length, 1,
    'generic projectile enemies should release exactly once after their warning');
}

console.log('Project Starfall Rift Lantern telegraph tests passed.');
