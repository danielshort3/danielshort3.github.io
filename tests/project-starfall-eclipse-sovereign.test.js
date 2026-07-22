'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const EngineVisuals = require('../js/games/project-starfall/engine/visuals.js');
const PixiRenderer = require('../js/games/project-starfall/project-starfall-renderer-pixi.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function createEncounterRuntime() {
  const engine = createProjectStarfallEngine(null, Data);
  assert.strictEqual(engine.chooseClass('fighter'), true);
  assert.strictEqual(engine.changeMap('eclipseThrone'), true);
  engine.enemies = [];
  const encounter = Data.BOSS_ENCOUNTERS.find((candidate) => candidate.id === 'eclipseSovereign');
  const enemyData = Data.ENEMIES.find((candidate) => candidate.id === 'eclipseSovereign');
  assert(encounter && enemyData);
  const boss = engine.createEnemy(enemyData, engine.runtime.spawnPoints[0]);
  boss.isEncounterBoss = true;
  boss.bossEncounterId = encounter.id;
  boss.hp = 1000000;
  boss.maxHp = 1000000;
  boss.bossPhaseIndex = 0;
  boss.bossActionIndex = 0;
  engine.enemies.push(boss);
  return { engine, encounter, boss };
}

function beginAction(runtime, actionId, phaseIndex) {
  const player = runtime.engine.getCombatCharacterByTarget('player', 'player');
  runtime.engine.beginBossEncounterAction(
    runtime.boss,
    runtime.encounter,
    runtime.encounter.phases[phaseIndex || 0],
    actionId,
    player
  );
  return runtime.boss.bossPendingAction;
}

function hasPlayer(hitTargets) {
  return hitTargets.some((character) => character && character.kind === 'player');
}

function createPixiSignature(actionId, reducedEffects) {
  const definitions = {
    solarFlare: { shape: 'pulse', label: 'SOLAR FLARE', radius: 230, polarity: 'danger', section: 'Solar Lane' },
    lunarMark: { shape: 'circle', label: 'LUNAR MARK', radius: 160, polarity: 'danger', section: 'Lunar Lane' },
    eclipseSigils: { shape: 'sigils', label: 'TOTALITY', radius: 170, polarity: 'safe', section: 'Eclipse Dais', hitRule: 'outsideRadius' }
  };
  const definition = definitions[actionId];
  const renderer = PixiRenderer.createRenderer({ data: Data });
  const operations = [];
  ['drawShape', 'drawLine', 'drawSolidRect', 'drawRectOutline', 'drawText'].forEach((method) => {
    renderer[method] = function recordOperation() {
      operations.push([method].concat(Array.from(arguments)));
      return true;
    };
  });
  renderer.renderWorldEffects({
    nowSec: 10,
    visualQuality: { reduceEffects: !!reducedEffects },
    worldEffects: [{
      type: 'bossHazard',
      actionId,
      shape: definition.shape,
      label: definition.label,
      x: 640,
      y: 420,
      r: definition.radius,
      ttl: 0.5,
      duration: 1,
      color: '#ffbe55',
      accentColor: '#7bdff2',
      variant: 'eclipse',
      telegraph: true,
      hazardPolarity: definition.polarity,
      hitRule: definition.hitRule || '',
      spatialSectionLabel: definition.section
    }]
  });
  return operations;
}

function main() {
  const encounter = Data.BOSS_ENCOUNTERS.find((candidate) => candidate.id === 'eclipseSovereign');
  assert(encounter);
  assert.strictEqual(encounter.introCombatDelay, 6.1);
  assert.strictEqual(encounter.phaseTransitionDelay, 2.2);
  assert.strictEqual(encounter.resetActionCycleOnPhase, true);
  assert(encounter.mechanic.includes('danger zones'));
  assert(encounter.mechanic.includes('regroup inside'));

  const profileRuntime = createEncounterRuntime();
  const solarProfile = profileRuntime.engine.getBossActionProfile('solarFlare', encounter);
  const lunarProfile = profileRuntime.engine.getBossActionProfile('lunarMark', encounter);
  const totalityProfile = profileRuntime.engine.getBossActionProfile('eclipseSigils', encounter);
  assert.strictEqual(solarProfile.telegraph, 1.25);
  assert.strictEqual(lunarProfile.telegraph, 1);
  assert.strictEqual(totalityProfile.telegraph, 1.55);
  assert.strictEqual(solarProfile.hitOrigin, 'target');
  assert.strictEqual(totalityProfile.hitRule, 'outsideRadius');
  assert.strictEqual(totalityProfile.hazardPolarity, 'safe');

  const solarRuntime = createEncounterRuntime();
  const solarPending = beginAction(solarRuntime, 'solarFlare', 0);
  assert.strictEqual(solarRuntime.engine.placePlayerOnRuntimePlatform(
    solarPending.spatialPlatformIndex,
    solarPending.targetX
  ), true);
  const solarHp = solarRuntime.engine.state.player.hp;
  const solarHits = solarRuntime.engine.applyBossHazardDamage(
    solarRuntime.boss,
    solarPending,
    solarPending.profile
  );
  assert(hasPlayer(solarHits), 'the visible Solar Lane marker should be the damaging zone');
  assert(solarRuntime.engine.state.player.hp < solarHp);

  const bossCenterRuntime = createEncounterRuntime();
  const bossCenterPending = beginAction(bossCenterRuntime, 'solarFlare', 0);
  const bossCenterPlayer = bossCenterRuntime.engine.state.player;
  bossCenterPlayer.x = bossCenterPending.bossX - bossCenterPlayer.w / 2;
  bossCenterPlayer.y = bossCenterPending.bossY - bossCenterPlayer.h / 2;
  const centerDistance = Math.hypot(
    bossCenterPending.bossX - bossCenterPending.targetX,
    bossCenterPending.bossY - bossCenterPending.targetY
  );
  assert(centerDistance > bossCenterPending.profile.radius);
  const bossCenterHp = bossCenterPlayer.hp;
  const bossCenterHits = bossCenterRuntime.engine.applyBossHazardDamage(
    bossCenterRuntime.boss,
    bossCenterPending,
    bossCenterPending.profile
  );
  assert(!hasPlayer(bossCenterHits), 'the boss center must not deal invisible Solar Flare damage');
  assert.strictEqual(bossCenterPlayer.hp, bossCenterHp);

  const safeRuntime = createEncounterRuntime();
  const totalityPending = beginAction(safeRuntime, 'eclipseSigils', 2);
  assert.strictEqual(totalityPending.spatialPlatformId, 'eclipse_throne_hop_01');
  assert.strictEqual(totalityPending.targetX, 2180);
  const totalityTelegraph = safeRuntime.engine.effects.find((effect) =>
    effect.type === 'bossHazard' && effect.actionId === 'eclipseSigils' && effect.telegraph);
  assert(totalityTelegraph);
  assert.strictEqual(totalityTelegraph.hazardPolarity, 'safe');
  assert.strictEqual(totalityTelegraph.hitRule, 'outsideRadius');
  assert.strictEqual(totalityTelegraph.spatialInstruction, 'Regroup on the eclipse dais without being hit.');
  assert.strictEqual(safeRuntime.engine.placePlayerOnRuntimePlatform(
    totalityPending.spatialPlatformIndex,
    totalityPending.targetX
  ), true);
  const safeHp = safeRuntime.engine.state.player.hp;
  const safeHits = safeRuntime.engine.applyBossHazardDamage(
    safeRuntime.boss,
    totalityPending,
    totalityPending.profile
  );
  assert(!hasPlayer(safeHits), 'the called Eclipse Dais must be safe during Totality');
  assert.strictEqual(safeRuntime.engine.state.player.hp, safeHp);
  safeRuntime.engine.resolveBossSpatialImmediateResponse(totalityPending, safeHits);
  assert.strictEqual(safeRuntime.engine.getBossSpatialResponseSummary().status, 'success');

  const outsideRuntime = createEncounterRuntime();
  const outsidePending = beginAction(outsideRuntime, 'eclipseSigils', 2);
  const lunarPlatform = outsideRuntime.engine.runtime.platforms.find((platform) =>
    platform.id === 'eclipse_throne_slope_03');
  assert(lunarPlatform);
  assert.strictEqual(outsideRuntime.engine.placePlayerOnRuntimePlatform(lunarPlatform.index, 2472), true);
  const outsideHp = outsideRuntime.engine.state.player.hp;
  const outsideHits = outsideRuntime.engine.applyBossHazardDamage(
    outsideRuntime.boss,
    outsidePending,
    outsidePending.profile
  );
  assert(hasPlayer(outsideHits), 'players outside the Totality safe zone should be hit');
  assert(outsideRuntime.engine.state.player.hp < outsideHp);
  outsideRuntime.engine.resolveBossSpatialImmediateResponse(outsidePending, outsideHits);
  assert.strictEqual(outsideRuntime.engine.getBossSpatialResponseSummary().status, 'failed');

  const transitionRuntime = createEncounterRuntime();
  const canceledPending = beginAction(transitionRuntime, 'solarFlare', 0);
  assert(transitionRuntime.engine.effects.some((effect) =>
    effect.type === 'bossHazard' && effect.actionId === canceledPending.actionId && effect.telegraph));
  transitionRuntime.boss.bossActionIndex = 9;
  transitionRuntime.boss.hp = transitionRuntime.boss.maxHp * 0.69;
  transitionRuntime.engine.updateBossEncounterPhase(transitionRuntime.boss, transitionRuntime.encounter);
  assert.strictEqual(transitionRuntime.boss.bossPhaseIndex, 1);
  assert.strictEqual(transitionRuntime.boss.bossActionIndex, 0);
  assert.strictEqual(transitionRuntime.boss.attackCd, 2.2);
  assert.strictEqual(transitionRuntime.boss.bossPendingAction, null);
  assert(!transitionRuntime.engine.effects.some((effect) =>
    effect.type === 'bossHazard' && effect.actionId === canceledPending.actionId && effect.telegraph),
  'phase changes should remove canceled warnings instead of leaving phantom decals');
  transitionRuntime.boss.bossActionIndex = 7;
  transitionRuntime.boss.hp = transitionRuntime.boss.maxHp * 0.37;
  const totalityPhase = transitionRuntime.engine.updateBossEncounterPhase(
    transitionRuntime.boss,
    transitionRuntime.encounter
  );
  assert.strictEqual(totalityPhase.actions[transitionRuntime.boss.bossActionIndex], 'eclipseSigils');
  assert.strictEqual(transitionRuntime.boss.attackCd, 2.2);

  const introEngine = createProjectStarfallEngine(null, Data);
  assert.strictEqual(introEngine.chooseClass('fighter'), true);
  assert.strictEqual(introEngine.enterBossEncounter('eclipseSovereign', { admin: true }), true);
  const introBoss = introEngine.enemies.find((enemy) => enemy.bossEncounterId === 'eclipseSovereign');
  assert(introBoss);
  assert.strictEqual(introBoss.attackCd, 6.1, 'the first boss call should wait until the intro card clears');

  const baseSpeed = 220;
  const reactionSeconds = 0.2;
  assert((solarProfile.telegraph - reactionSeconds) * baseSpeed >= solarProfile.radius);
  assert((lunarProfile.telegraph - reactionSeconds) * baseSpeed >= lunarProfile.radius);
  assert((encounter.phaseTransitionDelay + totalityProfile.telegraph - reactionSeconds) * baseSpeed >= 2960 - 2180);

  const sovereignAnimation = Data.ENEMY_ANIMATION_ASSETS.eclipseSovereign.states;
  assert.deepStrictEqual(sovereignAnimation.telegraph.holds, [3, 3, 4]);
  assert.strictEqual(sovereignAnimation.telegraph.fps, 8);
  assert.strictEqual(sovereignAnimation.telegraph.holds.reduce((sum, hold) => sum + hold, 0) / sovereignAnimation.telegraph.fps, 1.25);
  assert.strictEqual(sovereignAnimation.attack.holds.reduce((sum, hold) => sum + hold, 0) / sovereignAnimation.attack.fps, 0.5);
  assert.strictEqual(sovereignAnimation.buff.holds.reduce((sum, hold) => sum + hold, 0) / sovereignAnimation.buff.fps, 0.75);

  const warningStart = EngineVisuals.createBossHazardEffectDrawState({
    type: 'bossHazard',
    actionId: 'solarFlare',
    shape: 'pulse',
    telegraph: true,
    ttl: 1,
    duration: 1,
    r: 230
  });
  const warningImpact = EngineVisuals.createBossHazardEffectDrawState({
    type: 'bossHazard',
    actionId: 'solarFlare',
    shape: 'pulse',
    telegraph: true,
    ttl: 0.01,
    duration: 1,
    r: 230
  });
  assert(warningImpact.alpha > warningStart.alpha, 'critical warnings should intensify rather than fade before impact');

  const solarPixi = createPixiSignature('solarFlare', false);
  const lunarPixi = createPixiSignature('lunarMark', false);
  const totalityPixi = createPixiSignature('eclipseSigils', false);
  assert.notDeepStrictEqual(solarPixi, lunarPixi);
  assert.notDeepStrictEqual(lunarPixi, totalityPixi);
  assert.notDeepStrictEqual(solarPixi, totalityPixi);
  assert(totalityPixi.some((operation) => operation[0] === 'drawText' && operation[2] === 'SAFE: ECLIPSE DAIS'));
  assert(solarPixi.some((operation) => operation[0] === 'drawText' && operation[2] === 'DANGER: SOLAR LANE'));

  const reducedSolar = createPixiSignature('solarFlare', true);
  const reducedLunar = createPixiSignature('lunarMark', true);
  const reducedTotality = createPixiSignature('eclipseSigils', true);
  assert.notDeepStrictEqual(reducedSolar, reducedLunar);
  assert.notDeepStrictEqual(reducedLunar, reducedTotality);
  assert(reducedTotality.some((operation) => operation[0] === 'drawText' && operation[2] === 'SAFE: ECLIPSE DAIS'),
    'reduced effects must preserve the safe-zone boundary label');

  process.stdout.write('Project Starfall Eclipse Sovereign tests passed.\n');
}

main();
