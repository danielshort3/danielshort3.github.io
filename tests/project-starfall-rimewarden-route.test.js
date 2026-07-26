'use strict';

const assert = require('assert');

const Data = require('../js/games/project-starfall/project-starfall-data.js');
const dungeonEngine = require('../js/games/project-starfall/engine/dungeons.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

const RIMEWARDEN = Object.freeze({
  dungeonId: 'rimewarden_sanctum',
  mapId: 'rimewardenSanctum',
  bossId: 'rimewarden',
  level: 58,
  beatIds: Object.freeze([
    'break_brute_gate',
    'silence_whiteout_shelf',
    'challenge_rimewarden'
  ]),
  sectionIds: Object.freeze([
    'rimewardenSanctum_brute_lane',
    'rimewardenSanctum_oracle_shelf',
    'rimewardenSanctum_sentinel_shelf'
  ]),
  gateXs: Object.freeze([1533, 3067, 0])
});

const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  rimewardenSanctum_brute_lane: Object.freeze([
    'rimewarden_sanctum_solid_lane_01',
    'rimewarden_sanctum_solid_lane_02'
  ]),
  rimewardenSanctum_oracle_shelf: Object.freeze([
    'rimewarden_sanctum_solid_lane_03',
    'rimewarden_sanctum_solid_lane_04',
    'rimewarden_sanctum_solid_lane_06'
  ]),
  rimewardenSanctum_sentinel_shelf: Object.freeze([
    'rimewarden_sanctum_solid_lane_05'
  ])
});

const EXPECTED_SPATIAL_ACTIONS = Object.freeze({
  iceShockwave: Object.freeze({
    sectionId: 'rimewardenSanctum_brute_lane',
    platformId: 'rimewarden_sanctum_solid_lane_01'
  }),
  whiteout: Object.freeze({
    sectionId: 'rimewardenSanctum_oracle_shelf',
    platformId: 'rimewarden_sanctum_solid_lane_06'
  }),
  iceWall: Object.freeze({
    sectionId: 'rimewardenSanctum_sentinel_shelf',
    platformId: 'rimewarden_sanctum_solid_lane_05'
  }),
  addWave: Object.freeze({
    sectionId: 'rimewardenSanctum_sentinel_shelf',
    platformId: 'rimewarden_sanctum_solid_lane_05'
  })
});

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function unique(values) {
  return Array.from(new Set((values || []).map(String).filter(Boolean)));
}

function sameMembers(actual, expected) {
  const actualValues = unique(actual).sort();
  const expectedValues = unique(expected).sort();
  return actualValues.length === expectedValues.length &&
    actualValues.every((value, index) => value === expectedValues[index]);
}

function getFlow(game) {
  return game.state.dungeons.currentRun.encounterFlow;
}

function getActiveBeatEnemies(game) {
  const flow = getFlow(game);
  return game.enemies.filter((enemy) =>
    enemy && enemy.hp > 0 && enemy.dungeonBeatId === flow.activeBeatId
  );
}

function prepareGame(game) {
  game.toastMessages = [];
  game.toast = (message) => {
    game.toastMessages.push(String(message || ''));
    return true;
  };
  game.recordProgressEvent = () => false;
  game.syncRosterUnlocks = () => false;
  game.awardProgressReward = () => true;
  return game;
}

function createLiveGame() {
  const game = prepareGame(createProjectStarfallEngine(null, Data));
  check(game.chooseClass('fighter'),
    'the Rimewarden route fixture should choose Fighter');
  game.state.player.level = RIMEWARDEN.level;
  game.state.player.advancedClassId = 'guardian';
  check(game.startDungeon(RIMEWARDEN.dungeonId),
    'the Rimewarden route fixture should enter the sanctum');
  return game;
}

function restoreGame(payload) {
  const game = prepareGame(createProjectStarfallEngine(null, Data));
  check(game.restore(payload),
    'the Rimewarden route fixture should restore');
  return game;
}

function assertEnemiesInsideSection(game, enemies, sectionId, message) {
  const section = game.runtime.spawnSections.find((entry) => entry.id === sectionId);
  check(!!section, `${sectionId} should resolve in the live runtime`);
  enemies.forEach((enemy) => {
    check(enemy.spawnSectionId === section.id &&
      enemy.x >= section.x &&
      enemy.x + enemy.w <= section.x + section.w,
    `${message}: ${enemy.id} should remain inside ${section.label}`);
    if (enemy.data && enemy.data.behavior !== 'boss') {
      const wander = game.getEnemyWanderBounds(enemy);
      check(wander &&
        wander.left >= section.x &&
        wander.right + enemy.w <= section.x + section.w,
      `${message}: ${enemy.id} wander bounds should remain inside ${section.label}`);
    }
  });
}

function triggerBossAddWave(game, boss) {
  const encounter = game.getBossEncounterForEnemy(boss);
  const phase = encounter && (encounter.phases || []).find((entry) =>
    Array.isArray(entry.actions) && entry.actions.includes('addWave')
  );
  const target = game.getCombatCharacterByTarget('player', 'player');
  const existing = new Set(game.enemies);
  check(!!encounter && !!phase && !!target,
    'Rimewarden should expose its authored add-wave phase');
  game.beginBossEncounterAction(boss, encounter, phase, 'addWave', target);
  const pending = boss.bossPendingAction;
  check(pending &&
    pending.spatialSectionId === EXPECTED_SPATIAL_ACTIONS.addWave.sectionId,
  'Rimewarden add-wave telegraph should target the Sentinel Shelf');
  game.resolveBossEncounterAction(boss, encounter, pending);
  return game.enemies.filter((enemy) =>
    enemy && !existing.has(enemy) && enemy.encounterMinion
  );
}

const dungeon = Data.DUNGEONS.find((entry) => entry.id === RIMEWARDEN.dungeonId);
const map = Data.MAPS.find((entry) => entry.id === RIMEWARDEN.mapId);
const bossData = Data.ENEMIES.find((entry) => entry.id === RIMEWARDEN.bossId);
const definition = dungeonEngine.getDungeonEncounterFlowDefinition(
  RIMEWARDEN.dungeonId,
  { data: Data }
);

check(!!dungeon && !!map && !!bossData && !!definition,
  'Rimewarden Sanctum should publish its map, boss, and staged dungeon route');
check(map.asset === 'img/project-starfall/maps/rimewarden-sanctum.webp' &&
  map.environment.terrain === 'rimewarden-sanctum' &&
  map.environment.props === 'rimewarden-sanctum' &&
  map.palette.join('|') === '#d7f3ff|#2f6fa6|#f7fbff',
'the route pass should preserve the playful icy painting, props, and frost palette');
check(bossData.asset === 'img/project-starfall/enemies/rimewarden.png' &&
  bossData.animation &&
  bossData.animation.sheet === 'img/project-starfall/animations/enemies/rimewarden-compact-sheet.png' &&
  bossData.animation.frameWidth === 128 &&
  bossData.animation.frameHeight === 128,
'Rimewarden should retain its compact playful sprite and proportions');
check(map.layoutStyle === 'dungeonArena' &&
  map.geometryGenerator === 'dungeonArena' &&
  map.arenaSkeleton === 'ice-wall-vault' &&
  map.platforms[0].w === 4600 &&
  map.platforms.length === 15 &&
  map.climbables.length === 6 &&
  map.movementProfile === 'ice',
'the staged route should preserve the compact sanctum skeleton, vertical lanes, and ice movement');
check(map.designIntent.priorityRedesign &&
  map.designIntent.implementationStatus === 'composition-route-v1' &&
  map.designIntent.visualIdentityTag === 'playful frost vault and ice-wall seal' &&
  map.designIntent.routeSummary.includes('Whiteout Shelf'),
'the design contract should describe the implemented route without changing its visual identity');

const routeSections = map.fieldComposition.routeSections;
check(routeSections.map((section) => `${section.label}:${section.x}:${section.w}`).join('|') ===
  'Brute Lane:0:1533|Oracle Shelf:1533:1534|Sentinel Shelf:3067:1533',
'the sanctum should read as three ordered route beats instead of decorative equal thirds');
check(routeSections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should follow the previous section`);
  return section.x + section.w;
}, 0) === map.platforms[0].w,
'the named route beats should cover the full sanctum without gaps or overlap');

const sectionsById = new Map(map.spawnSections.map((section) => [section.id, section]));
const groupsBySectionId = new Map(map.spawnGroups.map((group) => [group.sectionId, group]));
const claimedPlatformIds = new Set();
Object.entries(EXPECTED_SECTION_PLATFORMS).forEach(([sectionId, expectedPlatformIds]) => {
  const section = sectionsById.get(sectionId);
  const group = groupsBySectionId.get(sectionId);
  check(!!section && !!group,
    `${sectionId} should publish matching route and encounter sections`);
  check(sameMembers(section.platformIds, expectedPlatformIds) &&
    sameMembers(group.platformIds, expectedPlatformIds),
  `${sectionId} should own its explicit frost-lane platforms`);
  check(expectedPlatformIds.every((platformId) => {
    if (claimedPlatformIds.has(platformId)) return false;
    claimedPlatformIds.add(platformId);
    return map.platforms.some((platform) => platform.id === platformId);
  }), `${sectionId} should not share combat platforms with another route beat`);
  check(group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${sectionId} spawn territory should stay inside its named route beat`);
  check(!group.enemyWeights.some((entry) => entry.enemyId === RIMEWARDEN.bossId),
    `${sectionId} ordinary spawn weights should never include Rimewarden`);
});
check(claimedPlatformIds.size === 6,
  'the three route beats should own all six broad sanctum combat lanes exactly once');
check(map.spawnGroups.map((group) => group.population).join('|') === '4|4|1' &&
  map.spawnGroups.every((group) => group.maxPopulation === group.population) &&
  map.spawnGroups.reduce((sum, group) => sum + group.population, 0) === map.waveMax,
'the published encounter populations should mirror the staged 4-4-1 route budget');
check(map.spawnGroups[0].enemyWeights.some((entry) => entry.enemyId === 'rimebackBrute') &&
  map.spawnGroups[1].enemyWeights.some((entry) => entry.enemyId === 'icebloomOracle') &&
  map.spawnGroups[2].enemyWeights.some((entry) => entry.enemyId === 'glacierSentinel'),
'enemy roles should progress from grounded brutes to whiteout support and a sentinel seal');
check(map.spawnGroups[0].spawnBounds.minX - 100 >= map.portals[0].x,
  'the Glacier Return should retain a calm spawn-free arrival apron');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: Data.MAPS });
const reachablePlatforms = mapRuntime.getReachablePlatformIndices(runtime.platformGraph, 0);
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.issues.length === 0,
'the unchanged sanctum geometry should remain connected, loopable, and fully spawn-covered');
check(Array.from(claimedPlatformIds).every((platformId) => {
  const platform = runtime.platforms.find((entry) => entry.id === platformId);
  return platform && reachablePlatforms.has(platform.index);
}), 'every semantic encounter lane should remain reachable from the entrance floor');
const validation = validateMap(map);
check(validation.issues.length === 0 && validation.warnings.length === 0,
  `Rimewarden Sanctum should satisfy the shared map validator: ${
    validation.issues.concat(validation.warnings).join(' | ')
  }`);

check(definition.id === 'rimewarden_sanctum_route' &&
  definition.dungeonId === RIMEWARDEN.dungeonId &&
  definition.mapId === RIMEWARDEN.mapId &&
  definition.bossHpScale === 4,
'the route definition should bind a measured boss budget to the real sanctum');
check(definition.beats.map((beat) => beat.id).join('|') === RIMEWARDEN.beatIds.join('|') &&
  definition.beats.map((beat) => beat.kind).join('|') === 'combat|combat|boss' &&
  definition.beats.map((beat) => beat.sectionIds[0]).join('|') === RIMEWARDEN.sectionIds.join('|') &&
  definition.beats.map((beat) => beat.gateX).join('|') === RIMEWARDEN.gateXs.join('|'),
'the route should sequence Brute Gate, Whiteout Shelf, then Rimewarden');
check(definition.beats.map((beat) =>
  dungeonEngine.createDungeonEncounterBeatSlots(beat).length
).join('|') === '4|4|1',
'the route should stage exactly four enemies, four enemies, then one boss');
check(definition.beats.every((beat) => beat.entryGateX === 0 && beat.arenaMaxX === 0),
  'the boss route should reopen prior frost lanes for ice-ring and whiteout rotations');

const spatialHooks = Data.BOSS_SPATIAL_MECHANICS[RIMEWARDEN.mapId].hooks;
const spatialGame = createLiveGame();
Object.entries(EXPECTED_SPATIAL_ACTIONS).forEach(([actionId, expected]) => {
  const hook = spatialHooks[actionId];
  const section = hook && spatialGame.getRuntimeBossSpatialSection(hook);
  const platform = section && spatialGame.getBossSpatialPlatformForSection(section, hook);
  check(hook &&
    section &&
    section.id === expected.sectionId &&
    platform &&
    platform.id === expected.platformId,
  `${actionId} should resolve to its authored sanctum section and platform`);
});

const pureRun = dungeonEngine.createDungeonStartRunState(
  RIMEWARDEN.dungeonId,
  false,
  { data: Data, startedAt: 1000, nowMs: 1000 }
);
const firstBeat = definition.beats[0];
const firstSlots = dungeonEngine.createDungeonEncounterBeatSlots(firstBeat);
const wrongRun = dungeonEngine.recordDungeonEncounterEnemyDefeat(pureRun, {
  id: firstSlots[0].enemyId,
  dungeonBeatId: firstBeat.id,
  dungeonBeatSlotId: firstSlots[0].id,
  dungeonBeatDungeonId: RIMEWARDEN.dungeonId,
  dungeonBeatMapId: RIMEWARDEN.mapId,
  dungeonBeatRunStartedAt: pureRun.startedAt + 1
}, { data: Data, mapId: RIMEWARDEN.mapId, nowMs: 1100 });
check(!wrongRun.accepted && wrongRun.reason === 'wrong-run',
  'a stale or fabricated run tag should not advance the sanctum');
const wrongEnemy = dungeonEngine.recordDungeonEncounterEnemyDefeat(pureRun, {
  id: 'snowglareWisp',
  dungeonBeatId: firstBeat.id,
  dungeonBeatSlotId: firstSlots[0].id,
  dungeonBeatDungeonId: RIMEWARDEN.dungeonId,
  dungeonBeatMapId: RIMEWARDEN.mapId,
  dungeonBeatRunStartedAt: pureRun.startedAt
}, { data: Data, mapId: RIMEWARDEN.mapId, nowMs: 1101 });
check(!wrongEnemy.accepted && wrongEnemy.reason === 'wrong-enemy',
  'a matching slot tag should not credit the wrong frost enemy');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const game = createLiveGame();
  check(game.state.mapId === RIMEWARDEN.mapId &&
    getFlow(game).activeBeatIndex === 0 &&
    getFlow(game).spawnedBeatIds.join('|') === RIMEWARDEN.beatIds[0],
  'entering the sanctum should start only the Brute Gate beat');
  let activeEnemies = getActiveBeatEnemies(game);
  check(activeEnemies.length === 4 &&
    activeEnemies.map((enemy) => enemy.id).join('|') ===
      'rimebackBrute|frostlingScout|shardling|rimebackBrute',
  'the opening beat should spawn the authored four-enemy grounded guard');
  check(activeEnemies.every((enemy) =>
    enemy.dungeonBeatDungeonId === RIMEWARDEN.dungeonId &&
    enemy.dungeonBeatMapId === RIMEWARDEN.mapId &&
    enemy.dungeonBeatRunStartedAt === game.state.dungeons.currentRun.startedAt &&
    enemy.preventWaveRespawn
  ), 'opening enemies should carry exact run identity and no-respawn tags');
  assertEnemiesInsideSection(
    game,
    activeEnemies,
    RIMEWARDEN.sectionIds[0],
    'the Brute Gate beat'
  );
  check(!game.enemies.some((enemy) =>
    enemy.hp > 0 && enemy.id === RIMEWARDEN.bossId
  ), 'Rimewarden must not spawn at dungeon entry');
  check(game.getActiveDungeonRouteGateX() === RIMEWARDEN.gateXs[0],
    'the first ice seal should hold the forward route at x=1533');
  const firstHud = game.getDungeonSnapshot().activeDungeon.encounterFlow;
  check(firstHud &&
    firstHud.hud.title === 'Route 1/3' &&
    firstHud.hud.label === 'Break the Brute Gate' &&
    firstHud.hud.value === 0 &&
    firstHud.hud.goal === 4,
  'the dungeon HUD should communicate the first four-enemy route beat');
  game.spawnDungeonEncounterBeat(map);
  game.updateDungeonBossRespawns();
  check(getActiveBeatEnemies(game).length === 4,
    'route synchronization should not duplicate the opening guard');
  check(!game.completeDungeon(dungeon),
    'the dungeon should reject an early completion before the route is clear');

  const untaggedBoss = game.createEnemy(bossData, Object.assign(
    game.chooseBossSpawnPoint(0),
    { adminSpawned: true, preventWaveRespawn: true }
  ));
  game.enemies.push(untaggedBoss);
  game.defeatEnemy(untaggedBoss);
  check(getFlow(game).activeBeatIndex === 0 &&
    !game.state.dungeons.completedDungeonIds.includes(RIMEWARDEN.dungeonId),
  'an admin-spawned Rimewarden should not bypass the staged route');

  game.defeatEnemy(activeEnemies[0]);
  const partialRestore = restoreGame(game.serialize());
  check(getFlow(partialRestore).activeBeatIndex === 0 &&
    getFlow(partialRestore).defeatedSlotIds.length === 1 &&
    getActiveBeatEnemies(partialRestore).length === 3,
  'partial Brute Gate restore should recreate only the three undefeated slots');
  partialRestore.spawnDungeonEncounterBeat(map);
  check(getActiveBeatEnemies(partialRestore).length === 3,
    'restored opening spawns should remain idempotent');

  getActiveBeatEnemies(partialRestore).slice().forEach((enemy) =>
    partialRestore.defeatEnemy(enemy)
  );
  check(getFlow(partialRestore).activeBeatIndex === 1 &&
    getFlow(partialRestore).status === 'active' &&
    partialRestore.getActiveDungeonRouteGateX() === RIMEWARDEN.gateXs[1] &&
    !partialRestore.enemies.some((enemy) =>
      enemy.hp > 0 && enemy.id === RIMEWARDEN.bossId
    ),
  'clearing Brute Gate should move the forward seal to the Whiteout Shelf');
  activeEnemies = getActiveBeatEnemies(partialRestore);
  check(activeEnemies.length === 4 &&
    activeEnemies.map((enemy) => enemy.id).join('|') ===
      'icebloomOracle|snowglareWisp|glacierSentinel|icebloomOracle',
  'the second beat should spawn the authored oracle, wisp, and sentinel guard');
  assertEnemiesInsideSection(
    partialRestore,
    activeEnemies,
    RIMEWARDEN.sectionIds[1],
    'the Whiteout Shelf beat'
  );
  const secondHud = partialRestore.getDungeonSnapshot().activeDungeon.encounterFlow;
  check(secondHud &&
    secondHud.hud.title === 'Route 2/3' &&
    secondHud.hud.label === 'Silence the Whiteout Shelf' &&
    secondHud.hud.goal === 4,
  'the HUD should advance to the Whiteout Shelf objective');

  activeEnemies.slice().forEach((enemy) => partialRestore.defeatEnemy(enemy));
  check(getFlow(partialRestore).activeBeatIndex === 2 &&
    getFlow(partialRestore).status === 'boss' &&
    partialRestore.getActiveDungeonRouteGateX() === 0,
  'clearing the Whiteout Shelf should reveal the boss and reopen all frost lanes');
  const openBounds = partialRestore.getActiveDungeonRouteGateBounds();
  check(openBounds.minX === 0 && openBounds.maxX === 0,
    'the final boss beat should not seal players away from earlier spatial mechanics');
  activeEnemies = getActiveBeatEnemies(partialRestore);
  check(activeEnemies.length === 1 &&
    activeEnemies[0].id === RIMEWARDEN.bossId &&
    activeEnemies[0].encounterHpScale === definition.bossHpScale &&
    activeEnemies[0].maxHp > 40000,
  'the final beat should reveal one route-scaled Rimewarden');
  const boss = activeEnemies[0];
  assertEnemiesInsideSection(
    partialRestore,
    activeEnemies,
    RIMEWARDEN.sectionIds[2],
    'the Rimewarden reveal'
  );
  const bossHud = partialRestore.getDungeonSnapshot().activeDungeon.encounterFlow;
  check(bossHud &&
    bossHud.hud.title === 'Route 3/3' &&
    bossHud.hud.label === 'Challenge the Rimewarden' &&
    bossHud.hud.goal === 1,
  'the HUD should identify the one-target final reveal');
  const preArmHp = boss.hp;
  check(!partialRestore.damageEnemy(boss, 99999, 'route-test') &&
    boss.hp === preArmHp,
  'Rimewarden should remain invulnerable during the short reveal window');

  const bossRestore = restoreGame(partialRestore.serialize());
  activeEnemies = getActiveBeatEnemies(bossRestore);
  check(getFlow(bossRestore).activeBeatIndex === 2 &&
    activeEnemies.length === 1 &&
    activeEnemies[0].id === RIMEWARDEN.bossId,
  'restoring at the boss reveal should recreate exactly one tagged Rimewarden');
  bossRestore.spawnDungeonEncounterBeat(map);
  check(getActiveBeatEnemies(bossRestore).length === 1,
    'restored boss spawning should remain idempotent');

  const restoredBoss = getActiveBeatEnemies(bossRestore)[0];
  const bossAdds = triggerBossAddWave(bossRestore, restoredBoss);
  const sentinelSection = sectionsById.get(RIMEWARDEN.sectionIds[2]);
  check(bossAdds.length === 2 &&
    bossAdds.every((add) =>
      add.dungeonEncounterAdd &&
      add.dungeonEncounterFlowId === getFlow(bossRestore).id &&
      add.dungeonEncounterParentBeatId === definition.beats[2].id &&
      add.dungeonEncounterDungeonId === RIMEWARDEN.dungeonId &&
      add.dungeonEncounterMapId === RIMEWARDEN.mapId &&
      add.dungeonEncounterRunStartedAt === bossRestore.state.dungeons.currentRun.startedAt &&
      add.preventWaveRespawn &&
      !add.dungeonBeatSlotId &&
      add.spawnSectionId === sentinelSection.id &&
      add.x >= sentinelSection.x &&
      add.x + add.w <= sentinelSection.x + sentinelSection.w
    ),
  'Rimewarden adds should be Sentinel-owned, run-tagged, and unable to impersonate the boss slot');
  const defeatedBeforeAdd = getFlow(bossRestore).defeatedSlotIds.length;
  const pendingBeforeAdd = bossRestore.getWaveState(RIMEWARDEN.mapId).pending.length;
  bossRestore.defeatEnemy(bossAdds[0]);
  check(getFlow(bossRestore).activeBeatIndex === 2 &&
    getFlow(bossRestore).defeatedSlotIds.length === defeatedBeforeAdd &&
    bossRestore.getWaveState(RIMEWARDEN.mapId).pending.length === pendingBeforeAdd,
  'defeating a boss add should neither clear the boss slot nor queue a legacy replacement');

  restoredBoss.hostileArmedAt = 0;
  bossRestore.defeatEnemy(restoredBoss);
  check(getFlow(bossRestore).status === 'complete' &&
    bossRestore.state.dungeons.completedDungeonIds.includes(RIMEWARDEN.dungeonId) &&
    dungeonEngine.getDungeonEncounterCompletionBlockReason(
      RIMEWARDEN.dungeonId,
      bossRestore.state.dungeons.currentRun,
      { data: Data }
    ) === '',
  'defeating the one tagged Rimewarden should complete the dungeon exactly once');
  check(!bossRestore.enemies.some((enemy) =>
    enemy && enemy.hp > 0 && enemy.encounterMinion
  ), 'completing the route should purge surviving boss adds');
  bossRestore.updateDungeonBossRespawns();
  check(!bossRestore.enemies.some((enemy) =>
    enemy && enemy.hp > 0 && enemy.id === RIMEWARDEN.bossId
  ), 'the legacy respawn loop should not recreate Rimewarden after the clear');

  const completeRestore = restoreGame(bossRestore.serialize());
  check(getFlow(completeRestore).status === 'complete' &&
    getActiveBeatEnemies(completeRestore).length === 0,
  'a completed save should restore without recreating route enemies');
  check(completeRestore.startDungeon(RIMEWARDEN.dungeonId) &&
    getFlow(completeRestore).activeBeatIndex === 0 &&
    getActiveBeatEnemies(completeRestore).length === 4 &&
    !completeRestore.enemies.some((enemy) =>
      enemy.hp > 0 && enemy.id === RIMEWARDEN.bossId
    ),
  'the staged sanctum should remain immediately replayable from Brute Gate');
} finally {
  Math.random = originalRandom;
}

console.log(`Project Starfall Rimewarden route checks passed: ${checks}`);
