'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const SUPPORTED_RESPONSE_TYPES = new Set([
  'avoidHazard',
  'reachSection',
  'clearAdds',
  'damageWindow',
  'dodgeProjectiles'
]);

Object.values(data.BOSS_SPATIAL_MECHANICS).forEach((mechanic) => {
  Object.values(mechanic.hooks).forEach((hook) => {
    assert(hook.responseCheck && SUPPORTED_RESPONSE_TYPES.has(hook.responseCheck.type),
      `${mechanic.id}:${hook.actionId} should declare a supported response contract`);
    assert(hook.responseCheck.label,
      `${mechanic.id}:${hook.actionId} should explain its measurable success condition`);
  });
});

function createDungeonEngine(mapId) {
  const engine = createProjectStarfallEngine(null, data);
  assert.strictEqual(engine.chooseClass('fighter'), true);
  assert.strictEqual(engine.changeMap(mapId), true, `${mapId} should load for boss-response testing`);
  engine.state.player.maxHp = 1000000;
  engine.state.player.hp = 1000000;
  engine.enemies = [];
  const dungeons = engine.getDungeonState();
  dungeons.activeDungeonId = `response_test_${mapId}`;
  dungeons.currentRun = {
    dungeonId: `response_test_${mapId}`,
    startedAt: Date.now(),
    completedAt: 0,
    bossDefeated: false,
    objectives: {},
    partyDefeats: 0
  };
  engine.ensureDungeonRunObjectives(dungeons.currentRun);
  return engine;
}

function spatialProgress(engine) {
  return engine.getDungeonState().currentRun.objectives.spatial_control.progress;
}

function createEncounterBoss(engine, bossId) {
  const enemyData = data.ENEMIES.find((enemy) => enemy.id === bossId);
  assert(enemyData, `${bossId} should exist`);
  const boss = engine.createEnemy(enemyData, engine.runtime.spawnPoints[0]);
  boss.isEncounterBoss = true;
  boss.bossEncounterId = bossId;
  engine.enemies.push(boss);
  const encounter = engine.getBossEncounterForEnemy(boss);
  assert(encounter, `${bossId} should resolve an encounter`);
  return { boss, encounter };
}

function beginAction(engine, boss, encounter, actionId) {
  engine.beginBossEncounterAction(
    boss,
    encounter,
    { id: 'responseTest', name: 'Response Test' },
    actionId,
    engine.getCombatCharacterByTarget('player', 'player')
  );
  assert(boss.bossPendingAction, `${actionId} should create a pending action`);
  return boss.bossPendingAction;
}

function placePlayerOnResponsePlatform(engine, pending) {
  const section = engine.runtime.spawnSections.find((entry) => entry.id === pending.spatialSectionId);
  const platform = engine.runtime.platforms.find((entry) => entry.id === pending.spatialPlatformId);
  assert(section && platform, 'the response should resolve to a real section and platform');
  const left = Math.max(Number(section.x || 0) + 24, Number(platform.x || 0) + 24);
  const right = Math.min(
    Number(section.x || 0) + Number(section.w || 0) - 24,
    Number(platform.x || 0) + Number(platform.w || 0) - 24
  );
  assert(right >= left, 'the called section should overlap its response platform');
  assert.strictEqual(engine.placePlayerOnRuntimePlatform(platform.index, (left + right) / 2), true);
}

function assertResponseAddPlacement(engine, pending, adds) {
  const section = engine.runtime.spawnSections.find((entry) => entry.id === pending.spatialSectionId);
  assert(section, `${pending.spatialSectionId} should resolve to a runtime section`);
  adds.forEach((add) => {
    const centerX = Number(add.x || 0) + Number(add.w || 0) / 2;
    assert(centerX >= Number(section.x || 0) && centerX <= Number(section.x || 0) + Number(section.w || 0),
      `${add.id} should spawn inside ${pending.spatialSectionId}`);
    assert.strictEqual(add.spawnSectionId, pending.spatialSectionId,
      `${add.id} should retain the called response section as its spawn section`);
    assert.strictEqual(add.groundedPlatformId, pending.spatialPlatformId,
      `${add.id} should use the allowed response platform`);
    assert.strictEqual(add.bossSpatialResponseSectionId, pending.spatialSectionId);
    assert.strictEqual(add.bossSpatialResponsePlatformId, pending.spatialPlatformId);
  });
}

Object.entries(data.BOSS_SPATIAL_MECHANICS).forEach(([mapId, mechanic]) => {
  const hook = mechanic.hooks.addWave;
  if (!hook) return;
  const engine = createProjectStarfallEngine(null, data);
  assert.strictEqual(engine.chooseClass('fighter'), true);
  assert.strictEqual(engine.changeMap(mapId), true, `${mapId} should load for add-response placement coverage`);
  const section = engine.runtime.spawnSections.find((entry) => entry.id === hook.sectionId);
  const platform = engine.getBossSpatialPlatformForSection(section, hook);
  assert(section && platform, `${mapId} should resolve its called add section to an allowed platform`);
  const response = {
    id: `placement-${mapId}`,
    sectionId: section.id,
    sectionLabel: section.label,
    platformId: platform.id,
    platformIndex: platform.index,
    targetTier: hook.targetTier
  };
  [0, 1].forEach((index) => {
    const spawn = engine.getBossSpatialAddSpawnPoint(response, index);
    assert(spawn, `${mapId} should produce a constrained response-add spawn`);
    assert.strictEqual(spawn.sectionId, section.id);
    assert.strictEqual(spawn.platformId, platform.id);
    assert(spawn.x >= Number(section.x || 0) && spawn.x < Number(section.x || 0) + Number(section.w || 0),
      `${mapId} response-add spawn ${index} should stay inside the called section`);
  });
});

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'gearSlam');
  assert.strictEqual(pending.spatialResponseCheck.type, 'avoidHazard');
  assert.strictEqual(engine.placePlayerOnRuntimePlatform(pending.spatialPlatformIndex, pending.targetX), true);
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 0,
    'standing in a resolved boss hazard should not earn Spatial Control progress');
  assert.strictEqual(engine.getBossSpatialResponseSummary().status, 'failed');
  assert.strictEqual(engine.getBossSpatialResponseSummary().failureReason, 'hazardHit');
}

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'gearSlam');
  const ground = engine.runtime.platforms[0];
  assert.strictEqual(engine.placePlayerOnRuntimePlatform(ground.index, engine.runtime.worldWidth - 180), true);
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 1,
    'clearing a marked boss hazard before impact should earn one Spatial Control response');
  assert.strictEqual(engine.getBossSpatialResponseSummary().status, 'success');
  assert.strictEqual(engine.getBossSpatialResponseSummary().progressAwarded, true);
}

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'corePulse');
  assert.strictEqual(pending.spatialResponseCheck.type, 'reachSection');
  assert.strictEqual(engine.placePlayerOnRuntimePlatform(0, 2100), true);
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 0,
    'surviving outside the called switch section should not count as answering it');
  assert.strictEqual(engine.getBossSpatialResponseSummary().failureReason, 'sectionMissed');
}

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'corePulse');
  placePlayerOnResponsePlatform(engine, pending);
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 1,
    'reaching the called switch platform without taking the hit should earn Spatial Control progress');
  const snapshot = engine.snapshot().bossEncounter;
  assert(snapshot.lastSpatialResponse && snapshot.lastSpatialResponse.status === 'success',
    'the boss snapshot should expose recent response feedback for the HUD');
}

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'addWave');
  assert.strictEqual(pending.spatialResponseCheck.type, 'clearAdds');
  engine.resolveBossEncounterAction(boss, encounter, pending);
  const responseAdds = engine.enemies.filter((enemy) => enemy.bossSpatialResponse);
  assert.strictEqual(responseAdds.length, 2, 'the full summoned wave should share one response contract');
  assertResponseAddPlacement(engine, pending, responseAdds);
  assert.strictEqual(spatialProgress(engine), 0, 'summoning adds should not count as clearing them');
  engine.defeatEnemy(responseAdds[0]);
  assert.strictEqual(spatialProgress(engine), 0, 'a partial add clear should not earn progress');
  engine.defeatEnemy(responseAdds[1]);
  assert.strictEqual(spatialProgress(engine), 1, 'defeating the full tagged wave should earn one response');
}

{
  const engine = createDungeonEngine('rimewardenVault');
  const { boss, encounter } = createEncounterBoss(engine, 'rimewarden');
  const pending = beginAction(engine, boss, encounter, 'addWave');
  assert(!engine.runtime.spawnPoints.some((spawn) => spawn.sectionId === pending.spatialSectionId),
    'the fallback regression should use a called section without a generic authored spawn point');
  engine.resolveBossEncounterAction(boss, encounter, pending);
  const responseAdds = engine.enemies.filter((enemy) => enemy.bossSpatialResponse);
  assert.strictEqual(responseAdds.length, 2,
    'the boss response system should synthesize constrained add positions when generic points are unavailable');
  assertResponseAddPlacement(engine, pending, responseAdds);
}

{
  const engine = createDungeonEngine('gearworksVault');
  const { boss, encounter } = createEncounterBoss(engine, 'quarryColossus');
  const pending = beginAction(engine, boss, encounter, 'plateExpose');
  assert.strictEqual(pending.spatialResponseCheck.type, 'damageWindow');
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 0, 'opening a burst window should not award progress by itself');
  assert(boss.bossSpatialDamageResponse && boss.bossSpatialDamageResponse.status === 'pending',
    'the exposed boss should retain a short actionable response window');
  assert(!engine.isCombatCharacterInBossSpatialSection(
    engine.getCombatCharacterByTarget('player', 'player'),
    pending
  ), 'the first expose hit should come from outside the called switch section');
  assert.strictEqual(engine.damageEnemy(boss, 12, 'melee', { attackerKind: 'player' }), true);
  assert.strictEqual(spatialProgress(engine), 0,
    'outside-section damage should not answer the called expose mechanic');
  assert.strictEqual(boss.bossSpatialDamageResponse.status, 'pending',
    'an outside hit should leave time to reposition and answer the active window');
  assert.strictEqual(boss.bossSpatialDamageResponse.lastRejectedReason, 'attackerOutsideSection');
  placePlayerOnResponsePlatform(engine, pending);
  assert.strictEqual(engine.damageEnemy(boss, 12, 'melee', { attackerKind: 'player' }), true);
  assert.strictEqual(spatialProgress(engine), 1,
    'direct damage from the called switch section should earn progress');
  engine.damageEnemy(boss, 12, 'melee', { attackerKind: 'player' });
  assert.strictEqual(spatialProgress(engine), 1, 'one expose window should never award duplicate responses');
  const expiredPending = beginAction(engine, boss, encounter, 'plateExpose');
  engine.resolveBossEncounterAction(boss, encounter, expiredPending);
  boss.bossSpatialDamageResponse.expiresAt = 0;
  assert.strictEqual(engine.pruneBossSpatialDamageResponse(boss), true,
    'an unanswered expose window should expire instead of remaining creditable indefinitely');
  assert.strictEqual(spatialProgress(engine), 1, 'an expired expose window should not earn progress');
  assert.strictEqual(engine.getBossSpatialResponseSummary().failureReason, 'damageWindowExpired');
}

{
  const engine = createDungeonEngine('stormbreakAerie');
  const { boss, encounter } = createEncounterBoss(engine, 'stormbreakRoc');
  const pending = beginAction(engine, boss, encounter, 'windBolt');
  assert.strictEqual(pending.spatialResponseCheck.type, 'dodgeProjectiles');
  const hpBefore = engine.state.player.hp;
  engine.resolveBossEncounterAction(boss, encounter, pending);
  assert.strictEqual(spatialProgress(engine), 0, 'launching a volley should not award progress');
  assert.strictEqual(engine.state.player.hp, hpBefore,
    'a projectile volley should not also apply an immediate invisible area hit');
  assert.strictEqual(engine.projectiles.length, 3);
  engine.projectiles.forEach((projectile) => {
    projectile.x = engine.runtime.worldWidth - 220;
    projectile.y = 80;
    projectile.vx = 0;
    projectile.vy = 0;
    projectile.ttl = 0.01;
  });
  engine.updateProjectiles(0.02);
  assert.strictEqual(spatialProgress(engine), 1,
    'letting every tagged projectile expire without a hit should earn one dodge response');
}

{
  const engine = createDungeonEngine('stormbreakAerie');
  const { boss, encounter } = createEncounterBoss(engine, 'stormbreakRoc');
  const pending = beginAction(engine, boss, encounter, 'windBolt');
  engine.resolveBossEncounterAction(boss, encounter, pending);
  engine.projectiles.forEach((projectile, index) => {
    projectile.vx = 0;
    projectile.vy = 0;
    projectile.ttl = index === 0 ? 1 : 0.01;
    projectile.x = index === 0 ? engine.state.player.x : engine.runtime.worldWidth - 220;
    projectile.y = index === 0 ? engine.state.player.y : 80;
  });
  engine.updateProjectiles(0.02);
  assert.strictEqual(spatialProgress(engine), 0,
    'taking any projectile in a tagged volley should fail that dodge response');
  assert.strictEqual(engine.getBossSpatialResponseSummary().failureReason, 'projectileHit');
}

console.log('Project Starfall boss response tests passed.');
