'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  COUNTERPLAY_CONFIG,
  createRiftCounterplayState,
  createRiftCounterplaySnapshot,
  reduceRiftCounterplay
} = require('../js/games/project-starfall/engine/rift-counterplay.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function runSequence(counterplayId, events) {
  let state = createRiftCounterplayState({ mapId: 'endlessRift' });
  let result = null;
  events.forEach((event) => {
    result = reduceRiftCounterplay(state, Object.assign({ allowedIds: [counterplayId] }, event));
    state = result.state;
  });
  return { state, result };
}

const echoSequence = runSequence('interrupt_echo', [
  { type: 'direct_hit', now: 1, enemyKey: 'echo-1', x: 180, platformIndex: 0, damage: 8, maxHealth: 100, telegraphing: true }
]);
assert(echoSequence.result.activeIds.includes('interrupt_echo'), 'hitting a live telegraph should activate Break the Echo');

const spacingSequence = runSequence('control_spacing', [
  { type: 'direct_hit', now: 1, enemyKey: 'pack-a', x: 180, platformIndex: 0, damage: 5, maxHealth: 100 },
  { type: 'direct_hit', now: 1.4, enemyKey: 'pack-b', x: 310, platformIndex: 0, damage: 5, maxHealth: 100 }
]);
assert(spacingSequence.result.activeIds.includes('control_spacing'), 'hitting distinct enemies across one clean lane should activate Control the Spread');

const guardSequence = runSequence('break_guard', [
  { type: 'direct_hit', now: 1, enemyKey: 'guard-1', x: 220, platformIndex: 0, damage: 8, maxHealth: 100 },
  { type: 'direct_hit', now: 1.5, enemyKey: 'guard-1', x: 220, platformIndex: 0, damage: 8, maxHealth: 100 }
]);
assert(guardSequence.result.activeIds.includes('break_guard'), 'sustained damage on one durable target should activate Shatter the Guard');

const movementSequence = runSequence('rotate_hazards', [
  { type: 'move', now: 1, dx: 125, dy: 0, fromPlatformIndex: 0, toPlatformIndex: 0, inCombat: true, resource: 50, maxResource: 100 },
  { type: 'move', now: 1.5, dx: 125, dy: 0, fromPlatformIndex: 0, toPlatformIndex: 0, inCombat: true, resource: 50, maxResource: 100 }
]);
assert(movementSequence.result.activeIds.includes('rotate_hazards'), 'moving meaningful world distance during combat should activate Rotate the Heat');

const focusSequence = runSequence('break_focus', [
  { type: 'move', now: 1, dx: 40, dy: -72, fromPlatformIndex: 0, toPlatformIndex: 1, inCombat: true, resource: 50, maxResource: 100 },
  { type: 'direct_hit', now: 2, enemyKey: 'focus-1', x: 260, platformIndex: 1, damage: 8, maxHealth: 100 }
]);
assert(focusSequence.result.activeIds.includes('break_focus'), 'changing platforms and landing a follow-up hit should activate Break Their Focus');

const burstSequence = runSequence('burst_window', [
  { type: 'tick', now: 1, resource: 82, maxResource: 100 },
  { type: 'tick', now: 2.4, resource: 82, maxResource: 100 },
  { type: 'skill_spent', now: 2.5, skillId: 'test_skill', offensive: true, cost: 18, resourceBefore: 82, maxResource: 100 },
  { type: 'direct_hit', now: 3, enemyKey: 'burst-1', x: 260, platformIndex: 0, damage: 16, maxHealth: 100, skillId: 'test_skill' }
]);
assert(burstSequence.result.activeIds.includes('burst_window'), 'conserving resource before an offensive skill connects should activate Choose the Burst Window');

Object.entries({
  interrupt_echo: echoSequence,
  control_spacing: spacingSequence,
  break_guard: guardSequence,
  rotate_hazards: movementSequence,
  break_focus: focusSequence,
  burst_window: burstSequence
}).forEach(([counterplayId, sequence]) => {
  const activeSnapshot = createRiftCounterplaySnapshot(sequence.state, sequence.state.lastActivatedAt[counterplayId]);
  assert(activeSnapshot.activeIds.includes(counterplayId), `${counterplayId} should appear in its live snapshot`);
  const expiry = sequence.state.activeUntil[counterplayId] + 0.01;
  const expired = reduceRiftCounterplay(sequence.state, { type: 'tick', now: expiry, allowedIds: [counterplayId], resource: 0, maxResource: 100 });
  assert(!expired.activeIds.includes(counterplayId), `${counterplayId} should expire after its authored ${COUNTERPLAY_CONFIG[counterplayId].duration}s window`);
  const reset = reduceRiftCounterplay(sequence.state, { type: 'reset', now: expiry, mapId: 'starfallCrossing' });
  assert.deepStrictEqual(createRiftCounterplaySnapshot(reset.state, expiry).activeIds, [], `${counterplayId} should clear on map reset`);
});

const gated = runSequence('break_guard', [
  { type: 'direct_hit', now: 1, enemyKey: 'echo-2', x: 180, platformIndex: 0, damage: 8, maxHealth: 100, telegraphing: true }
]);
assert(!gated.result.activeIds.includes('interrupt_echo'), 'counterplay should not activate when its matching mutation is absent');

function createRiftEngine(mutationId) {
  const engine = createProjectStarfallEngine(null, data);
  engine.chooseClass('fighter');
  engine.state.player.level = 100;
  engine.changeMap('endlessRift', { silent: true });
  engine.enemies = [];
  engine.projectiles = [];
  engine.effects = [];
  engine.state.rift = Object.assign({}, engine.state.rift, {
    tier: 12,
    bestTier: 12,
    score: 0,
    mutationIds: [mutationId]
  });
  let fakeNow = 1;
  engine.getRiftCounterplayNow = () => fakeNow;
  engine.setRiftCounterplayTestTime = (value) => { fakeNow = Number(value); };
  engine.resetRiftCounterplay('endlessRift', fakeNow);
  engine.invalidateRiftRuntimeEffectsCache();
  return engine;
}

function createLivingRiftEnemy(engine, x, platformIndex) {
  const enemyData = data.ENEMIES.find((enemy) => enemy.id === 'riftAberration');
  assert(enemyData, 'counterplay integration requires the Rift Aberration');
  const enemy = engine.createEnemy(enemyData, { x, platformIndex: platformIndex || 0 });
  enemy.x = x;
  enemy.groundedPlatformIndex = platformIndex || 0;
  enemy.aggroUntil = 100;
  engine.enemies.push(enemy);
  engine.enemySpatialIndex = null;
  return enemy;
}

const echoEngine = createRiftEngine('echoing');
const echoEnemy = createLivingRiftEnemy(echoEngine, 260, 0);
echoEnemy.telegraph = 0.7;
echoEnemy.state = 'charging';
const echoSpawnDamage = echoEnemy.damage;
const echoBefore = echoEngine.getRiftRuntimeEffects();
assert.strictEqual(echoBefore.mutationEffects[0].countered, false, 'production Rift effects should begin uncountered');
echoEngine.damageEnemy(echoEnemy, 10, 'melee');
assert.strictEqual(echoEnemy.state, 'idle', 'a successful Break the Echo hit should cancel the enemy cast state');
assert(echoEnemy.attackCd >= 1.1, 'an interrupted enemy should receive a real attack delay');
const echoAfter = echoEngine.getRiftRuntimeEffects();
assert.strictEqual(echoAfter.mutationEffects[0].countered, true, 'production effects should automatically derive the live counterplay id');
assert.strictEqual(echoEngine.getRiftRuntimeEffects(), echoAfter, 'unchanged live counterplay should reuse the runtime-effects cache');
assert.strictEqual(echoEngine.getRiftRuntimeEffects({ counterplayIds: [] }).mutationEffects[0].countered, false,
  'an explicitly injected empty counterplay list should remain authoritative for tests and callers');
assert(echoEnemy.damage < echoSpawnDamage, 'already-spawned Rift enemies should lose mitigated mutation damage during counterplay');
assert(echoEngine.getRiftSnapshot().counterplay.activeIds.includes('interrupt_echo'), 'Rift snapshots should expose live counterplay status');
echoEngine.setRiftCounterplayTestTime(10);
echoEngine.recordRiftCounterplayEvent({ type: 'tick', resource: 0, maxResource: 100 });
assert.strictEqual(echoEngine.getRiftRuntimeEffects().mutationEffects[0].countered, false, 'expired production counterplay should leave the runtime cache');
assert(Math.abs(echoEnemy.damage - echoSpawnDamage) < 0.001, 'living enemy damage should restore when the counterplay window expires');

const spacingEngine = createRiftEngine('splintering');
const spacingEnemyA = createLivingRiftEnemy(spacingEngine, 250, 0);
const spacingEnemyB = createLivingRiftEnemy(spacingEngine, 390, 0);
spacingEngine.damageEnemy(spacingEnemyA, 5, 'melee');
spacingEngine.setRiftCounterplayTestTime(1.4);
spacingEngine.damageEnemy(spacingEnemyB, 5, 'melee');
assert(spacingEngine.getActiveRiftCounterplayIds().includes('control_spacing'), 'production direct-hit hooks should recognize controlled pack spacing');

const guardEngine = createRiftEngine('guarded');
const guardEnemy = createLivingRiftEnemy(guardEngine, 280, 0);
const guardedSpawnHealth = guardEnemy.maxHp;
const guardPressureDamage = Math.ceil(guardedSpawnHealth * 0.08);
guardEngine.damageEnemy(guardEnemy, guardPressureDamage, 'melee');
guardEngine.setRiftCounterplayTestTime(1.4);
guardEngine.damageEnemy(guardEnemy, guardPressureDamage, 'melee');
assert(guardEngine.getActiveRiftCounterplayIds().includes('break_guard'), 'production damage hooks should recognize sustained guard pressure');
assert(guardEnemy.maxHp < guardedSpawnHealth, 'guard counterplay should reduce living Rift enemy health pressure');

const movementEngine = createRiftEngine('burning');
const movementEnemy = createLivingRiftEnemy(movementEngine, 500, 0);
const movementPlayer = movementEngine.state.player;
movementPlayer.x = 120;
movementPlayer.groundedPlatformIndex = 0;
movementEngine.recordRiftCounterplayMovement({ x: 0, y: movementPlayer.y, platformIndex: 0 }, movementPlayer, 1);
movementPlayer.x = 260;
movementEngine.setRiftCounterplayTestTime(1.5);
movementEngine.recordRiftCounterplayMovement({ x: 120, y: movementPlayer.y, platformIndex: 0 }, movementPlayer, 1.5);
assert(movementEnemy.hp > 0 && movementEngine.getActiveRiftCounterplayIds().includes('rotate_hazards'),
  'production movement hooks should recognize actual combat rotation distance');

const focusEngine = createRiftEngine('focused');
const focusEnemy = createLivingRiftEnemy(focusEngine, 300, 1);
focusEngine.state.player.x = 220;
focusEngine.state.player.y = 280;
focusEngine.state.player.groundedPlatformIndex = 1;
focusEngine.recordRiftCounterplayMovement({ x: 180, y: 360, platformIndex: 0 }, focusEngine.state.player, 1);
focusEngine.setRiftCounterplayTestTime(1.8);
focusEngine.damageEnemy(focusEnemy, 8, 'melee');
assert(focusEngine.getActiveRiftCounterplayIds().includes('break_focus'), 'production movement plus hit hooks should recognize a lane-change follow-up');

const burstEngine = createRiftEngine('volatile');
const burstEnemy = createLivingRiftEnemy(burstEngine, burstEngine.state.player.x + 72, 0);
burstEnemy.y = burstEngine.state.player.y;
const burstStats = burstEngine.getStats();
burstEngine.state.player.mp = burstStats.maxMp;
burstEngine.recordRiftCounterplayEvent({ type: 'tick', now: 1, resource: burstStats.maxMp, maxResource: burstStats.maxMp });
burstEngine.setRiftCounterplayTestTime(2.4);
burstEngine.recordRiftCounterplayEvent({ type: 'tick', resource: burstStats.maxMp, maxResource: burstStats.maxMp });
burstEngine.state.skills.fighter_heavy_strike = Math.max(1, Number(burstEngine.state.skills.fighter_heavy_strike || 0));
assert.strictEqual(burstEngine.useSkill('fighter_heavy_strike', { silent: true }), true, 'the production burst sequence should accept a real offensive skill cast');
assert(burstEngine.getActiveRiftCounterplayIds().includes('burst_window'), 'the authoritative skill-spend and hit hooks should activate the saved-resource burst window');

console.log('Project Starfall Rift counterplay tests passed.');
