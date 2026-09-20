'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

const originalNow = Date.now;
let clock = 1_800_000_000_000;
Date.now = () => clock;

function fixture(id) {
  const engine = createProjectStarfallEngine(null, data);
  assert(engine.chooseClass('fighter'));
  assert(engine.changeMap('greenrootMeadow'));
  const platform = engine.runtime.platforms[0];
  const player = engine.state.player;
  Object.assign(player, {
    x: 1300, y: platform.y - player.h, hp: 100000, maxHp: 100000, level: 50,
    vx: 0, vy: 0, grounded: true, groundedPlatformId: platform.id,
    groundedPlatformIndex: platform.index, invulnerableUntil: 0
  });
  const enemy = engine.createEnemy(data.ENEMIES.find(entry => entry.id === id), { x: 1000, platformIndex: 0 });
  Object.assign(enemy, {
    x: 1000, y: platform.y - enemy.h, vx: 0, vy: 0, hp: 10000, maxHp: 10000,
    level: 50, elite: false, eliteAffixIds: [], attackCd: 0,
    telegraph: 0, state: 'idle', pendingAttack: null, staggered: 0,
    grounded: true, groundedPlatformId: platform.id, groundedPlatformIndex: platform.index,
    facing: 1
  });
  engine.enemies = [enemy];
  engine.effects = [];
  engine.projectiles = [];
  engine.playAudioCue = () => true;
  engine.setEnemyAggro(enemy, engine.getCombatCharacterByTarget('player', 'player'), 'test', 30);
  return { engine, enemy, player, platform };
}

function step(engine, seconds = 0.016) {
  clock += seconds * 1000;
  engine.updateEnemies(seconds);
}

function warnings(engine, enemy) {
  return engine.effects.filter(effect => effect.sourceEnemyUid === enemy.uid &&
    (effect.type === 'telegraph' || effect.telegraph === true || effect.phase === 'prepare'));
}

function testChargeCommitment() {
  for (const id of data.ENEMIES.filter(entry => entry.behavior === 'charger').map(entry => entry.id)) {
    const { engine, enemy, player } = fixture(id);
    const start = enemy.x;
    step(engine);
    assert.strictEqual(enemy.state, 'charging', `${id} starts its charge warning`);
    assert.strictEqual(warnings(engine, enemy).length, 1, `${id} owns its warning`);
    const warning = { x: warnings(engine, enemy)[0].x, w: warnings(engine, enemy)[0].w };
    for (let frame = 0; frame < 34; frame += 1) step(engine);
    assert.strictEqual(enemy.x, start, `${id} holds its position through preparation`);
    assert(enemy.telegraph > 0 && enemy.telegraph < 0.21, `${id} reaches the final commitment window`);
    player.x = 750;
    for (let frame = 0; frame < 14; frame += 1) step(engine);
    assert.strictEqual(enemy.facing, 1, `${id} cannot reverse after commitment`);
    assert(enemy.vx > 0 && enemy.x > start, `${id} charges along the warned direction`);
    assert.deepStrictEqual({ x: warnings(engine, enemy)[0].x, w: warnings(engine, enemy)[0].w }, warning,
      `${id} does not move its warning after commitment`);
    assert(warning.w >= enemy.w + 350, `${id} warns about the full authored charge distance`);
  }
}

function testSpecialAttackCancellation() {
  for (const id of ['bristleBoar', 'riftLantern']) {
    for (const reason of ['stagger', 'defeat', 'lostTarget', 'interrupt']) {
      const { engine, enemy, player } = fixture(id);
      const target = engine.getCombatCharacterByTarget('player', 'player');
      if (id === 'riftLantern') engine.beginRiftLanternProjectileWindup(enemy, target);
      else engine.beginEnemyCharge(enemy);
      assert.strictEqual(warnings(engine, enemy).length, 1, `${id} has a visible warning before ${reason}`);
      const start = enemy.x;
      if (reason === 'stagger') {
        enemy.staggered = 1;
        step(engine);
        assert.strictEqual(enemy.x, start, `${id} does not continue moving while staggered`);
      } else if (reason === 'defeat') {
        engine.defeatEnemy(enemy);
      } else if (reason === 'lostTarget') {
        player.x = 3600;
        step(engine);
      } else {
        engine.interruptRiftEnemyCast(enemy);
      }
      assert.notStrictEqual(enemy.state, 'charging', `${id} cancels charge on ${reason}`);
      assert.notStrictEqual(enemy.state, 'riftLanternWindup', `${id} cancels cast on ${reason}`);
      assert.strictEqual(warnings(engine, enemy).length, 0, `${id} clears its canceled warning on ${reason}`);
      assert.strictEqual(engine.projectiles.length, 0, `${id} does not release a canceled projectile`);
    }
  }
}

function testLanternAimCommitment() {
  const { engine, enemy, player } = fixture('riftLantern');
  engine.beginRiftLanternProjectileWindup(enemy, engine.getCombatCharacterByTarget('player', 'player'));
  const startX = enemy.x;
  const duration = enemy.telegraph;
  for (let frame = 0; frame < Math.ceil((duration - 0.18) / 0.016); frame += 1) step(engine);
  assert.strictEqual(enemy.x, startX, 'Rift Lantern holds position while gathering its shot');
  player.x = 750;
  while (!engine.projectiles.length && enemy.state === 'riftLanternWindup') step(engine);
  assert.strictEqual(engine.projectiles.length, 1, 'Rift Lantern releases one projectile after preparation');
  assert(engine.projectiles[0].vx > 0, 'Rift Lantern fires at the committed aim point when the player dodges behind it');
  assert.strictEqual(enemy.facing, 1, 'Rift Lantern keeps the committed facing at release');
  assert.strictEqual(warnings(engine, enemy).length, 0, 'Rift Lantern replaces its warning at release');
  assert(enemy.attackRecovery > 0, 'Rift Lantern has a recovery after release');
}

function testLethalBurnEndsUpdate() {
  const { engine, enemy } = fixture('slimelet');
  Object.assign(enemy, {
    hp: 1, burning: 0.5, burnDamageCarry: 4, burnTickElapsed: 1,
    wanderTargetX: enemy.x + 100, wanderUntil: clock / 1000 + 10,
    wanderPauseUntil: 0
  });
  const startX = enemy.x;
  step(engine);
  assert.strictEqual(enemy.hp, 0, 'the burn tick defeats the enemy');
  assert.strictEqual(enemy.x, startX, 'a lethally burned enemy cannot execute wander AI later in the same update');
  assert.strictEqual(enemy.vx, 0, 'a defeated enemy remains stopped');
  assert.strictEqual(enemy.animationState, 'defeat', 'lethal burn preserves the defeat pose');
}

function testBossWarningOwnership() {
  const { engine, enemy } = fixture('brambleking');
  const encounter = engine.getBossEncounterForEnemy(enemy);
  assert(encounter);
  const other = engine.createEnemy(enemy.data, { x: 1450, platformIndex: 0 });
  engine.enemies.push(other);
  const target = engine.getCombatCharacterByTarget('player', 'player');
  const phase = encounter.phases[0];
  engine.beginBossEncounterAction(enemy, encounter, phase, phase.actions[0], target);
  engine.beginBossEncounterAction(other, encounter, phase, phase.actions[0], target);
  assert.strictEqual(warnings(engine, enemy).length, 1);
  assert.strictEqual(warnings(engine, other).length, 1);
  const active = { type: 'bossHazard', sourceEnemyUid: enemy.uid, telegraph: false, ttl: 1 };
  engine.effects.push(active);
  engine.interruptRiftEnemyCast(enemy);
  assert.strictEqual(enemy.bossPendingAction, null, 'boss interruption cancels the pending action');
  assert.strictEqual(warnings(engine, enemy).length, 0, 'boss interruption clears its pending warning');
  assert.strictEqual(warnings(engine, other).length, 1, 'another boss of the same species keeps its warning');
  assert(engine.effects.includes(active), 'already active hazards are not silently removed with pending warnings');
  engine.defeatEnemy(other);
  assert.strictEqual(other.bossPendingAction, null, 'defeat clears the boss pending action');
  assert.strictEqual(warnings(engine, other).length, 0, 'defeat clears the boss warning');
}

function testBossStaggerCancellation() {
  const { engine, enemy } = fixture('brambleking');
  const encounter = engine.getBossEncounterForEnemy(enemy);
  const phase = encounter.phases[0];
  engine.beginBossEncounterAction(enemy, encounter, phase, phase.actions[0],
    engine.getCombatCharacterByTarget('player', 'player'));
  enemy.staggered = 1;
  step(engine);
  assert.strictEqual(enemy.bossPendingAction, null, 'stagger cancels boss attacks rather than consuming their warning and releasing late');
  assert.strictEqual(warnings(engine, enemy).length, 0, 'stagger clears the boss warning alongside its pending action');
}

function testBossPreparationKeepsItsRoute() {
  const { engine, enemy, player, platform } = fixture('brambleking');
  const upper = engine.runtime.platforms[1];
  const link = engine.findEnemyPlatformLink(platform, upper);
  assert(link && link.type === 'jump', 'the fixture must offer a real jump toward the player platform');
  enemy.x = link.exitX - enemy.w / 2;
  enemy.vx = 90;
  player.x = link.entryX - player.w / 2;
  player.y = upper.y - player.h;
  player.groundedPlatformId = upper.id;
  player.groundedPlatformIndex = upper.index;
  const encounter = engine.getBossEncounterForEnemy(enemy);
  const phase = encounter.phases[0];
  engine.beginBossEncounterAction(enemy, encounter, phase, phase.actions[0],
    engine.getCombatCharacterByTarget('player', 'player'));
  const pending = enemy.bossPendingAction;
  const startX = enemy.x;
  let routeCalls = 0;
  const originalRoute = engine.updateEnemyPlatformJump;
  engine.updateEnemyPlatformJump = function (...args) {
    routeCalls += 1;
    return originalRoute.apply(this, args);
  };
  step(engine);
  assert.strictEqual(routeCalls, 0, 'a boss preparing an action must not initiate a route jump or climb toward another platform');
  assert.strictEqual(enemy.bossPendingAction, pending, 'route suppression preserves the pending boss action');
  assert.strictEqual(enemy.climbing, false);
  assert.strictEqual(enemy.grounded, true, 'the boss stays on its current support during preparation');
  assert(Math.abs(enemy.vx - 90 * 0.76) < 1e-9, 'the boss retains its authored horizontal deceleration');
  assert(Math.abs(enemy.x - startX - 90 * 0.76 * 0.016) < 1e-9,
    'blocking platform routing must not freeze normal horizontal momentum during preparation');
}

function testNewRangedPreparationDoesNotRoute() {
  const { engine, enemy, player } = fixture('banditThrower');
  const upper = engine.runtime.platforms[1];
  player.x = 650;
  player.y = upper.y - player.h;
  player.groundedPlatformId = upper.id;
  player.groundedPlatformIndex = upper.index;
  let routeCalls = 0;
  const originalRoute = engine.updateEnemyPlatformJump;
  engine.updateEnemyPlatformJump = function (...args) {
    routeCalls += 1;
    return originalRoute.apply(this, args);
  };
  step(engine);
  assert(enemy.pendingAttack && enemy.pendingAttack.kind === 'projectile', 'the ranged enemy begins its attack during this update');
  assert.strictEqual(routeCalls, 0, 'a newly started projectile windup must suppress routing on its first tick');
  assert.strictEqual(enemy.grounded, true, 'the ranged enemy keeps its support during preparation');
}

function testBossLostTargetCancelsPendingAction() {
  const { engine, enemy, player } = fixture('brambleking');
  const encounter = engine.getBossEncounterForEnemy(enemy);
  const phase = encounter.phases[0];
  engine.beginBossEncounterAction(enemy, encounter, phase, phase.actions[0],
    engine.getCombatCharacterByTarget('player', 'player'));
  assert(enemy.bossPendingAction && warnings(engine, enemy).length === 1);
  player.x = 3600;
  step(engine);
  assert.strictEqual(enemy.bossPendingAction, null, 'losing the target cancels a pending boss action');
  assert.strictEqual(warnings(engine, enemy).length, 0, 'the abandoned warning disappears with its action');
  player.x = 1300;
  engine.setEnemyAggro(enemy, engine.getCombatCharacterByTarget('player', 'player'), 'test', 30);
  const health = player.hp;
  step(engine);
  assert.strictEqual(player.hp, health, 'reacquiring the target cannot release an expired-warning attack');
}

const checks = [testChargeCommitment, testSpecialAttackCancellation, testLanternAimCommitment,
  testLethalBurnEndsUpdate, testBossWarningOwnership, testBossStaggerCancellation, testBossPreparationKeepsItsRoute,
  testNewRangedPreparationDoesNotRoute, testBossLostTargetCancelsPendingAction];
try {
  const failures = [];
  for (const check of checks) {
    try {
      check();
      console.log(`PASS ${check.name}`);
    } catch (error) {
      failures.push(error);
      console.error(`FAIL ${check.name}: ${error.message}`);
    }
  }
  assert.strictEqual(failures.length, 0, `${failures.length} enemy behavior regression groups failed`);
  console.log('Project Starfall enemy behavior: charge preparation/commitment, special attack cancellation, locked projectile aim, lethal burn, and boss warning ownership passed.');
} finally {
  Date.now = originalNow;
}
