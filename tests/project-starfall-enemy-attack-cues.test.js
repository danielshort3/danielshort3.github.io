'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const visuals = require('../js/games/project-starfall/engine/visuals.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const EPSILON = 1e-9;

function assertApproximately(actual, expected, message) {
  assert(
    Math.abs(Number(actual) - Number(expected)) <= EPSILON,
    `${message}: expected ${expected}, received ${actual}`
  );
}

function createEngine(playerOverrides) {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, {
    classId: 'fighter',
    advancedClassId: '',
    x: 240,
    y: 320,
    w: 40,
    h: 74,
    facing: 1,
    grounded: true,
    vx: 0,
    vy: 0,
    hp: 180,
    maxHp: 180,
    shield: 0,
    invulnerableUntil: 0
  }, playerOverrides || {});
  engine.effects = [];
  engine.projectiles = [];
  engine.enemies = [];
  return engine;
}

function createEnemy(overrides) {
  return Object.assign({
    id: 'slimelet',
    uid: 'attack-cue-enemy',
    name: 'Slimelet',
    x: 200,
    y: 348,
    w: 46,
    h: 46,
    hp: 100,
    maxHp: 100,
    damage: 12,
    vx: 0,
    vy: 0,
    facing: 1,
    attackCd: 0,
    telegraph: 0,
    pendingAttack: null,
    data: {
      id: 'slimelet',
      name: 'Slimelet',
      behavior: 'melee',
      family: 'Plant',
      speed: 48
    }
  }, overrides || {});
}

function getPlayerTarget(engine) {
  const target = engine.getCombatCharacterByTarget('player', 'player');
  assert(target, 'configured player should be available as an enemy combat target');
  return target;
}

function getOnlyTelegraph(engine) {
  const warnings = engine.effects.filter((effect) => effect && effect.type === 'telegraph');
  assert.strictEqual(warnings.length, 1, 'starting one enemy attack should publish one warning');
  return warnings[0];
}

function assertWarningGeometry(enemy, warning, facing) {
  const centerX = Number(enemy.x) + Number(enemy.w) / 2;
  const expectedX = facing > 0 ? centerX : centerX - Number(warning.w);
  assert.strictEqual(warning.combatCritical, true, 'enemy attack warnings should be marked combat-critical');
  assert.strictEqual(warning.facing, facing, 'enemy attack warnings should retain the attack facing');
  assert(Number(warning.w) > 0, 'enemy attack warnings should expose a positive danger length');
  assertApproximately(warning.x, expectedX, 'warning should extend from the enemy toward its facing');
  assertApproximately(warning.duration, enemy.telegraph, 'warning duration should match the enemy cue');
}

function testMeleeCueDefersAndReleasesDamage() {
  const engine = createEngine();
  const enemy = createEnemy();
  engine.enemies = [enemy];
  const target = getPlayerTarget(engine);
  const hpBeforeCue = engine.state.player.hp;

  assert.strictEqual(engine.beginEnemyMelee(enemy, target), true, 'a valid melee cue should begin');
  assert.strictEqual(engine.state.player.hp, hpBeforeCue, 'melee damage must not land during the cue');
  assert.strictEqual(enemy.pendingAttack.kind, 'melee', 'melee cue should store a pending attack');
  assert.strictEqual(enemy.animationState, 'telegraph', 'melee cue should use the telegraph animation state');
  assert.strictEqual(
    engine.effects.filter((effect) => effect && effect.type === 'slash').length,
    0,
    'melee release FX must not appear during the cue'
  );
  assertWarningGeometry(enemy, getOnlyTelegraph(engine), 1);

  assert.strictEqual(
    engine.resolveEnemyPendingAttack(enemy, enemy.pendingAttack),
    true,
    'a completed melee cue should resolve'
  );
  assert(engine.state.player.hp < hpBeforeCue, 'an in-range melee release should damage the player');
  assert.strictEqual(enemy.pendingAttack, null, 'melee release should consume its pending attack');
  assert.strictEqual(enemy.telegraph, 0, 'melee release should clear the cue timer');
  assert.strictEqual(enemy.animationState, 'attack', 'melee release should enter the attack animation');
  assert.strictEqual(
    engine.effects.filter((effect) => effect && effect.type === 'slash').length,
    1,
    'one melee release should publish one slash effect'
  );
}

function testMeleeCueCanWhiffAfterDodge() {
  const engine = createEngine({ x: 150 });
  const enemy = createEnemy({ facing: -1 });
  engine.enemies = [enemy];
  const target = getPlayerTarget(engine);
  const hpBeforeCue = engine.state.player.hp;

  assert.strictEqual(engine.beginEnemyMelee(enemy, target), true, 'left-facing melee cue should begin');
  assert.strictEqual(engine.state.player.hp, hpBeforeCue, 'left-facing melee cue should not deal early damage');
  assertWarningGeometry(enemy, getOnlyTelegraph(engine), -1);

  engine.state.player.x = 40;
  assert.strictEqual(
    engine.resolveEnemyPendingAttack(enemy, enemy.pendingAttack),
    true,
    'a dodged melee cue should still finish its release'
  );
  assert.strictEqual(engine.state.player.hp, hpBeforeCue, 'leaving melee range before release should make the attack whiff');
  assert.strictEqual(enemy.pendingAttack, null, 'a whiffed melee release should still consume its pending attack');
}

function testProjectileCueReleasesOnceTowardStoredPoint() {
  const engine = createEngine({ x: 390, y: 300 });
  const enemy = createEnemy({ x: 180, y: 328, facing: 1 });
  engine.enemies = [enemy];
  const target = getPlayerTarget(engine);

  assert.strictEqual(engine.beginEnemyProjectile(enemy, 'thorn', target), true, 'a valid projectile cue should begin');
  assert.strictEqual(engine.projectiles.length, 0, 'enemy projectile must not exist during the cue');
  assert.strictEqual(enemy.pendingAttack.kind, 'projectile', 'projectile cue should store a pending attack');
  assert.strictEqual(enemy.animationState, 'telegraph', 'projectile cue should use the telegraph animation state');
  assertWarningGeometry(enemy, getOnlyTelegraph(engine), 1);

  const storedTarget = {
    x: enemy.pendingAttack.targetX,
    y: enemy.pendingAttack.targetY
  };
  engine.state.player.x = 700;
  engine.state.player.y = 120;

  assert.strictEqual(
    engine.resolveEnemyPendingAttack(enemy, enemy.pendingAttack),
    true,
    'a completed projectile cue should resolve'
  );
  assert.strictEqual(engine.projectiles.length, 1, 'one projectile cue should release exactly one projectile');
  assert.strictEqual(enemy.animationState, 'projectile', 'projectile release should enter the projectile animation');

  const projectile = engine.projectiles[0];
  const dx = storedTarget.x - projectile.x;
  const dy = storedTarget.y - projectile.y;
  const distance = Math.hypot(dx, dy);
  assert(distance > 0, 'stored projectile target should be distinct from its release point');
  assertApproximately(projectile.vx, dx / distance * 240, 'projectile horizontal velocity should aim at the stored point');
  assertApproximately(projectile.vy, dy / distance * 240, 'projectile vertical velocity should aim at the stored point');

  assert.strictEqual(
    engine.resolveEnemyPendingAttack(enemy),
    false,
    'a consumed projectile cue should not resolve a second time'
  );
  assert.strictEqual(engine.projectiles.length, 1, 'a consumed projectile cue must not duplicate its projectile');
}

function testCombatWarningsSurviveWorldEffectBudgeting() {
  const telegraph = { type: 'telegraph', ttl: 0.28, combatCritical: true };
  const bossTelegraph = { type: 'bossHazard', ttl: 0.8, telegraph: true };
  const cosmeticEffects = [
    { type: 'lootPickup', ttl: 1 },
    { type: 'upgradeResult', ttl: 1 },
    { type: 'recoveryPulse', ttl: 1 },
    { type: 'skillImpact', ttl: 1 },
    { type: 'slash', ttl: 1 }
  ];
  const effects = [telegraph].concat(cosmeticEffects, bossTelegraph);
  const entries = effects.map((effect, index) => ({ effect, index }));
  const selected = visuals.selectBudgetedEffectEntries(entries, 0, 2, 'world').effects;

  assert(
    visuals.getWorldEffectPriority(telegraph) > visuals.getWorldEffectPriority(cosmeticEffects[0]),
    'combat telegraphs should outrank loot and other cosmetic effects'
  );
  assert(selected.includes(telegraph), 'a constrained world-effect budget should retain the regular enemy telegraph');
  assert(selected.includes(bossTelegraph), 'a constrained world-effect budget should retain the boss telegraph');
}

testMeleeCueDefersAndReleasesDamage();
testMeleeCueCanWhiffAfterDodge();
testProjectileCueReleasesOnceTowardStoredPoint();
testCombatWarningsSurviveWorldEffectBudgeting();

console.log('Project Starfall enemy attack cue tests passed.');
