'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const visuals = require('../js/games/project-starfall/engine/visuals.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function createEngine() {
  const engine = createProjectStarfallEngine(null, data);
  const player = engine.state.player;
  const platform = engine.runtime.platforms[0];
  Object.assign(player, {
    classId: 'fighter',
    level: 20,
    x: 220,
    y: platform.y - 74,
    w: 40,
    h: 74,
    hp: 1000,
    maxHp: 1000,
    facing: 1,
    grounded: true,
    groundedPlatformId: platform.id,
    groundedPlatformIndex: platform.index,
    invulnerableUntil: 0,
    vx: 0,
    vy: 0
  });
  engine.runtime.worldWidth = Math.max(1280, Number(engine.runtime.worldWidth || 0));
  return engine;
}

function createEnemy(engine, enemyId, x) {
  const enemyData = data.ENEMIES.find((entry) => entry.id === enemyId);
  assert(enemyData, enemyId + ' should exist for the readability fixture');
  const platform = engine.runtime.platforms[0];
  const enemy = engine.createEnemy(enemyData, { x, platformIndex: 0, platformId: platform.id });
  Object.assign(enemy, {
    x,
    y: platform.y - enemy.h,
    grounded: true,
    groundedPlatformId: platform.id,
    groundedPlatformIndex: platform.index,
    attackCd: 0,
    telegraph: 0,
    state: 'idle',
    pendingAttack: null,
    attackRecovery: 0,
    staggered: 0,
    vx: 0,
    vy: 0
  });
  return enemy;
}

function playerTarget(engine) {
  return engine.getCombatCharacterByTarget('player', 'player');
}

function assertNear(actual, expected, message) {
  assert(Math.abs(actual - expected) < 1e-9,
    message + ': expected ' + expected + ', received ' + actual);
}

const engine = createEngine();
const player = engine.state.player;
const target = playerTarget(engine);

const melee = createEnemy(engine, 'slimelet', player.x + 38);
const hpBeforeWindup = player.hp;
assert.strictEqual(engine.enemyMelee(melee, target), true,
  'ordinary melee enemies should begin an authored windup');
assert.strictEqual(player.hp, hpBeforeWindup,
  'melee windup must not deal same-tick damage');
assert(melee.pendingAttack && melee.pendingAttack.kind === 'melee');
assert.strictEqual(melee.animationState, 'telegraph',
  'the telegraph row should play before melee commitment');
assert(engine.effects.some((effect) => effect.type === 'telegraph' && effect.sourceEnemyUid === melee.uid),
  'melee windup should create a spatial warning');
assert.strictEqual(engine.getEnemyRenderSnapshot(melee, null).animationState, 'telegraph');
const meleeWindup = melee.pendingAttack.windup;

melee.telegraph = 0.1;
assert.strictEqual(engine.resolveEnemyPendingAttack(melee, [target]), true);
assert.strictEqual(player.hp, hpBeforeWindup,
  'partial windup updates must remain harmless');
melee.telegraph = 0;
assert.strictEqual(engine.resolveEnemyPendingAttack(melee, [target]), true);
assert(player.hp < hpBeforeWindup, 'remaining in the melee arc should take one committed hit');
assert.strictEqual(melee.lastAttackOutcome, 'hit');
assert.strictEqual(melee.pendingAttack, null);
assert.strictEqual(melee.animationState, 'attack',
  'the attack row should begin at the contact event');
assertNear(melee.attackCd, 1.35 - meleeWindup,
  'melee commitment should credit its completed windup against the old cooldown');
assertNear(meleeWindup + melee.attackCd, 1.35,
  'ordinary melee start-to-start cadence should remain unchanged');
const hpAfterCommit = player.hp;
assert.strictEqual(engine.resolveEnemyPendingAttack(melee, [target]), false);
assert.strictEqual(player.hp, hpAfterCommit, 'a resolved intent must never commit twice');

player.invulnerableUntil = 0;
player.vx = 0;
player.vy = 0;
const whiffEnemy = createEnemy(engine, 'slimelet', player.x + 38);
assert.strictEqual(engine.enemyMelee(whiffEnemy, target), true);
player.x += 190;
const hpBeforeWhiff = player.hp;
whiffEnemy.telegraph = 0;
assert.strictEqual(engine.resolveEnemyPendingAttack(whiffEnemy, [target]), true);
assert.strictEqual(player.hp, hpBeforeWhiff,
  'leaving the committed melee arc should produce a real whiff');
assert.strictEqual(whiffEnemy.lastAttackOutcome, 'whiff');
assert(engine.effects.some((effect) => effect.type === 'slash' && effect.enemyFxId === whiffEnemy.id),
  'a whiff should still finish its readable attack animation and slash');

player.x = 220;
player.invulnerableUntil = 0;
const ranged = createEnemy(engine, 'banditThrower', player.x + 280);
engine.projectiles = [];
assert.strictEqual(engine.enemyProjectile(ranged, 'knife', target), true);
assert.strictEqual(engine.projectiles.length, 0,
  'ordinary ranged enemies should not launch on the windup frame');
assert(ranged.pendingAttack && ranged.pendingAttack.kind === 'projectile');
assert.strictEqual(ranged.animationState, 'telegraph');
const rangedWindup = ranged.pendingAttack.windup;
ranged.telegraph = 0;
assert.strictEqual(engine.resolveEnemyPendingAttack(ranged, [target]), true);
assert.strictEqual(engine.projectiles.length, 1,
  'a completed ranged windup should launch exactly one projectile');
assert.strictEqual(ranged.lastAttackOutcome, 'released');
assert.strictEqual(ranged.animationState, 'projectile');
assertNear(ranged.attackCd, 1.8 - rangedWindup,
  'knife commitment should credit its completed windup against the old cooldown');
assertNear(rangedWindup + ranged.attackCd, 1.8,
  'ordinary knife start-to-start cadence should remain unchanged');
assert.strictEqual(engine.resolveEnemyPendingAttack(ranged, [target]), false);
assert.strictEqual(engine.projectiles.length, 1,
  'later updates must not duplicate a committed projectile');

const fireboltEnemy = createEnemy(engine, 'emberWisp', player.x + 280);
engine.projectiles = [];
assert.strictEqual(engine.enemyProjectile(fireboltEnemy, 'firebolt', target), true);
const fireboltWindup = fireboltEnemy.pendingAttack.windup;
fireboltEnemy.telegraph = 0;
assert.strictEqual(engine.resolveEnemyPendingAttack(fireboltEnemy, [target]), true);
assertNear(fireboltEnemy.attackCd, 2.1 - fireboltWindup,
  'generic firebolt commitment should credit its completed windup against the old cooldown');
assertNear(fireboltWindup + fireboltEnemy.attackCd, 2.1,
  'ordinary firebolt start-to-start cadence should remain unchanged');

const lostTargetEnemy = createEnemy(engine, 'banditThrower', player.x + 280);
assert.strictEqual(engine.enemyProjectile(lostTargetEnemy, 'knife', target), true);
lostTargetEnemy.telegraph = 0;
assert.strictEqual(engine.resolveEnemyPendingAttack(lostTargetEnemy, []), true);
assert.strictEqual(lostTargetEnemy.pendingAttack, null,
  'losing the intended target should cancel a queued projectile');
assert(!engine.effects.some((effect) => effect.sourceEnemyUid === lostTargetEnemy.uid),
  'lost-target cancellation should remove its stale spatial warning');

const staggeredEnemy = createEnemy(engine, 'slimelet', player.x + 38);
assert.strictEqual(engine.enemyMelee(staggeredEnemy, target), true);
staggeredEnemy.staggered = 0.5;
assert.strictEqual(engine.resolveEnemyPendingAttack(staggeredEnemy, [target]), true);
assert.strictEqual(staggeredEnemy.pendingAttack, null,
  'stagger should cancel an ordinary attack windup');
assert(!engine.effects.some((effect) => effect.sourceEnemyUid === staggeredEnemy.uid));

const interruptedEnemy = createEnemy(engine, 'slimelet', player.x + 38);
assert.strictEqual(engine.enemyMelee(interruptedEnemy, target), true);
assert.strictEqual(engine.interruptRiftEnemyCast(interruptedEnemy), true);
assert.strictEqual(interruptedEnemy.pendingAttack, null);
assert.strictEqual(interruptedEnemy.telegraph, 0,
  'an explicit cast interrupt should clear the warning and intent together');
assert.strictEqual(interruptedEnemy.animationState, 'idle',
  'a direct interruption should not leave the actor frozen in its warning pose');
assert(!engine.effects.some((effect) => effect.sourceEnemyUid === interruptedEnemy.uid));

const defeatedEnemy = createEnemy(engine, 'slimelet', player.x + 38);
assert.strictEqual(engine.enemyMelee(defeatedEnemy, target), true);
engine.defeatEnemy(defeatedEnemy);
assert.strictEqual(defeatedEnemy.pendingAttack, null);
assert.strictEqual(defeatedEnemy.telegraph, 0,
  'defeat should cancel an unresolved attack');
assert(!engine.effects.some((effect) => effect.sourceEnemyUid === defeatedEnemy.uid));

const contactEngine = createEngine();
const contactPlayer = contactEngine.state.player;
const contactEnemy = createEnemy(contactEngine, 'slimelet', contactPlayer.x + 6);
contactEnemy.level = contactPlayer.level;
contactEnemy.wanderPauseUntil = Number.POSITIVE_INFINITY;
contactEngine.enemies = [contactEnemy];
contactEngine.getPassiveOffscreenEnemyUpdateStride = () => 1;
contactEngine.shouldDeferPassiveOffscreenEnemyUpdate = () => false;
contactEngine.updateEnemyClimbing = () => false;
contactEngine.updateEnemyPlatformJump = () => false;
contactEngine.recoverFallenBodyThroughTop = () => false;
contactEngine.updateEnemyStuckState = () => false;
const hpBeforeContact = contactPlayer.hp;
contactEngine.updateEnemies(0.01);
assert.strictEqual(contactPlayer.hp, hpBeforeContact,
  'neutral body overlap should alert an enemy without passive touch damage');
assert.strictEqual(contactEnemy.aggroTargetKind, 'player');
contactEngine.updateEnemies(0.01);
assert(contactEnemy.pendingAttack && contactEnemy.pendingAttack.kind === 'melee',
  'an alerted contact enemy should answer with a readable attack windup');
assert.strictEqual(contactPlayer.hp, hpBeforeContact);

const chargeEngine = createEngine();
const chargePlayer = chargeEngine.state.player;
chargePlayer.hp = 1000;
chargePlayer.maxHp = 1000;
chargePlayer.invulnerableUntil = 0;
const charger = createEnemy(chargeEngine, 'bristleBoar', chargePlayer.x + 24);
chargeEngine.enemies = [charger];
chargeEngine.getPassiveOffscreenEnemyUpdateStride = () => 1;
chargeEngine.shouldDeferPassiveOffscreenEnemyUpdate = () => false;
chargeEngine.updateEnemyClimbing = () => false;
chargeEngine.updateEnemyPlatformJump = () => false;
chargeEngine.recoverFallenBodyThroughTop = () => false;
chargeEngine.updateEnemyStuckState = () => false;
chargeEngine.setEnemyAggro(charger, playerTarget(chargeEngine), 'test');
assert.strictEqual(chargeEngine.beginEnemyCharge(charger), true);
charger.telegraph = 0;
const hpBeforeCharge = chargePlayer.hp;
chargeEngine.updateEnemies(0.01);
assert(chargePlayer.hp < hpBeforeCharge,
  'an explicitly telegraphed charge should preserve collision damage');

const telegraphPriority = visuals.getWorldEffectPriority({ type: 'telegraph', ttl: 0.1 });
assert(telegraphPriority > visuals.getWorldEffectPriority({ type: 'lootPickup', ttl: 1 }),
  'semantic warnings should outrank loot under visual pressure');
assert(telegraphPriority > visuals.getWorldEffectPriority({ type: 'slash', ttl: 1 }),
  'semantic warnings should outrank decorative combat effects');

console.log('Project Starfall professional enemy readability tests passed.');
