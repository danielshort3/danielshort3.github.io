'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const dungeonEngine = require('../js/games/project-starfall/engine/dungeons.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const BRAMBLE = Object.freeze({
  dungeonId: 'bramble_depths',
  mapId: 'brambleDepths',
  level: 25,
  bossIds: Object.freeze(['brambleking']),
  sectionIds: Object.freeze([
    'brambleDepths_ridge_return',
    'brambleDepths_root_lanes',
    'brambleDepths_court_gate'
  ]),
  gateXs: Object.freeze([1200, 3200, 0])
});

function getFlow(game) {
  return game.state.dungeons.currentRun.encounterFlow;
}

function getActiveBeatEnemies(game) {
  const flow = getFlow(game);
  return game.enemies.filter((enemy) =>
    enemy && enemy.hp > 0 && enemy.dungeonBeatId === flow.activeBeatId
  );
}

function assertEnemiesStayInsideSection(game, enemies, sectionId, message) {
  const section = map.spawnSections.find((entry) => entry.id === sectionId);
  assert(section, `${sectionId} should exist`);
  enemies.forEach((enemy) => {
    assert(
      enemy.x >= section.x &&
      enemy.x + enemy.w <= section.x + section.w,
      `${message}: ${enemy.id} at ${enemy.x}-${enemy.x + enemy.w} must stay inside ${section.x}-${section.x + section.w}`
    );
    if (enemy.data && enemy.data.behavior !== 'boss') {
      const wanderBounds = game.getEnemyWanderBounds(enemy);
      assert(
        wanderBounds.left >= section.x &&
        wanderBounds.right + enemy.w <= section.x + section.w,
        `${message}: ${enemy.id} wander bounds must remain inside the active encounter pocket`
      );
    }
  });
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
  assert(game.chooseClass('fighter'));
  game.state.player.level = BRAMBLE.level;
  game.state.player.advancedClassId = 'guardian';
  assert(game.startDungeon(BRAMBLE.dungeonId));
  return game;
}

function restoreGame(payload) {
  const game = prepareGame(createProjectStarfallEngine(null, Data));
  assert(game.restore(payload));
  return game;
}

const dungeon = Data.DUNGEONS.find((entry) => entry.id === BRAMBLE.dungeonId);
const map = Data.MAPS.find((entry) => entry.id === BRAMBLE.mapId);
const definition = dungeonEngine.getDungeonEncounterFlowDefinition(BRAMBLE.dungeonId, { data: Data });
assert(dungeon && map && definition, 'Bramble Depths should publish a staged route definition');
assert.strictEqual(definition.dungeonId, BRAMBLE.dungeonId);
assert.strictEqual(definition.mapId, BRAMBLE.mapId);
assert.strictEqual(definition.beats.length, 3);
assert.deepStrictEqual(definition.beats.map((beat) => beat.kind), ['combat', 'combat', 'boss']);
assert.deepStrictEqual(definition.beats.map((beat) => beat.sectionIds[0]), BRAMBLE.sectionIds);
assert.deepStrictEqual(definition.beats.map((beat) => beat.gateX), BRAMBLE.gateXs);
assert.deepStrictEqual(definition.beats.slice(0, 2).map((beat) => beat.enemyIds.length), [4, 4],
  'Bramble should stage two four-enemy clears before its boss');
assert.deepStrictEqual(definition.beats[2].bossIds, BRAMBLE.bossIds);

const mapSectionIds = new Set((map.spawnSections || []).map((section) => section.id));
definition.beats.forEach((beat) => {
  beat.sectionIds.forEach((sectionId) => {
    assert(mapSectionIds.has(sectionId), `${sectionId} should reference a semantic Bramble map section`);
  });
  dungeonEngine.getDungeonEncounterBeatEnemyIds(beat).forEach((enemyId) => {
    const enemy = Data.ENEMIES.find((entry) => entry.id === enemyId);
    assert(enemy, `${enemyId} should reference a published enemy`);
    if (beat.kind === 'boss') assert.strictEqual(enemy.behavior, 'boss');
  });
});

const pureRun = dungeonEngine.createDungeonStartRunState(BRAMBLE.dungeonId, false, {
  data: Data,
  startedAt: 1000
});
const firstBeat = definition.beats[0];
const firstSlots = dungeonEngine.createDungeonEncounterBeatSlots(firstBeat);
assert.strictEqual(firstSlots.length, 4);
const firstDefeat = dungeonEngine.recordDungeonEncounterEnemyDefeat(pureRun, {
  id: firstSlots[0].enemyId,
  dungeonBeatId: firstBeat.id,
  dungeonBeatSlotId: firstSlots[0].id,
  dungeonBeatDungeonId: BRAMBLE.dungeonId,
  dungeonBeatMapId: BRAMBLE.mapId,
  dungeonBeatRunStartedAt: pureRun.startedAt
}, { data: Data, mapId: BRAMBLE.mapId, nowMs: 1500 });
assert(firstDefeat.accepted && !firstDefeat.advanced,
  'one tagged defeat should count without advancing the first four-enemy beat');
assert.strictEqual(pureRun.encounterFlow.activeBeatIndex, 0);

const wrongRunDefeat = dungeonEngine.recordDungeonEncounterEnemyDefeat(pureRun, {
  id: firstSlots[1].enemyId,
  dungeonBeatId: firstBeat.id,
  dungeonBeatSlotId: firstSlots[1].id,
  dungeonBeatDungeonId: BRAMBLE.dungeonId,
  dungeonBeatMapId: BRAMBLE.mapId,
  dungeonBeatRunStartedAt: pureRun.startedAt + 1
}, { data: Data, mapId: BRAMBLE.mapId, nowMs: 1501 });
assert.strictEqual(wrongRunDefeat.accepted, false);
assert.strictEqual(wrongRunDefeat.reason, 'wrong-run',
  'route credit should require the exact active-run identity tag');
const wrongEnemyDefeat = dungeonEngine.recordDungeonEncounterEnemyDefeat(pureRun, {
  id: firstSlots[0].enemyId,
  dungeonBeatId: firstBeat.id,
  dungeonBeatSlotId: firstSlots[1].id,
  dungeonBeatDungeonId: BRAMBLE.dungeonId,
  dungeonBeatMapId: BRAMBLE.mapId,
  dungeonBeatRunStartedAt: pureRun.startedAt
}, { data: Data, mapId: BRAMBLE.mapId, nowMs: 1502 });
assert.strictEqual(wrongEnemyDefeat.accepted, false);
assert.strictEqual(wrongEnemyDefeat.reason, 'wrong-enemy',
  'a valid slot tag should not credit a different enemy definition');
assert.strictEqual(pureRun.encounterFlow.defeatedSlotIds.length, 1);

const restoredState = dungeonEngine.createDungeonState(JSON.parse(JSON.stringify({
  activeDungeonId: BRAMBLE.dungeonId,
  currentRun: pureRun
})), { data: Data });
assert.deepStrictEqual(restoredState.currentRun.encounterFlow, pureRun.encounterFlow,
  'Bramble route progress should normalize through JSON restore');
assert.notStrictEqual(
  restoredState.currentRun.encounterFlow.defeatedSlotIds,
  pureRun.encounterFlow.defeatedSlotIds,
  'restored route arrays should not alias the source save'
);

const runSnapshot = dungeonEngine.createDungeonRunSnapshot(pureRun);
runSnapshot.encounterFlow.defeatedSlotIds.push('snapshot_only');
assert(!pureRun.encounterFlow.defeatedSlotIds.includes('snapshot_only'),
  'run snapshots should deep-clone route progress');
const flowSnapshot = dungeonEngine.createDungeonEncounterFlowSnapshot(BRAMBLE.dungeonId, pureRun, {
  data: Data,
  nowMs: 1600
});
assert.strictEqual(flowSnapshot.activeGateX, 1200);
assert.strictEqual(flowSnapshot.hud.title, 'Route 1/3');
assert.strictEqual(flowSnapshot.hud.label, 'Break the Root Gate');
assert.strictEqual(flowSnapshot.hud.goal, 4);
assert.strictEqual(flowSnapshot.hud.value, 1);
assert.strictEqual(flowSnapshot.mapId, BRAMBLE.mapId);

const game = createLiveGame();
assert.strictEqual(game.state.mapId, BRAMBLE.mapId);
assert.strictEqual(getFlow(game).activeBeatIndex, 0);
assert.deepStrictEqual(getFlow(game).spawnedBeatIds, [definition.beats[0].id]);
assert(!game.enemies.some((enemy) => enemy.hp > 0 && BRAMBLE.bossIds.includes(enemy.id)),
  'Brambleking must not spawn at dungeon entry');

let activeEnemies = getActiveBeatEnemies(game);
assert.strictEqual(activeEnemies.length, 4);
assert(activeEnemies.every((enemy) =>
  enemy.dungeonBeatDungeonId === BRAMBLE.dungeonId &&
  enemy.dungeonBeatMapId === BRAMBLE.mapId &&
  enemy.dungeonBeatRunStartedAt === game.state.dungeons.currentRun.startedAt &&
  enemy.spawnSectionId === definition.beats[0].sectionIds[0] &&
  enemy.preventWaveRespawn
), 'the first four enemies should carry exact run and semantic-section tags');
assertEnemiesStayInsideSection(
  game,
  activeEnemies,
  definition.beats[0].sectionIds[0],
  'the first encounter should remain spatially inside Ridge Return'
);

game.spawnDungeonEncounterBeat(map);
game.updateDungeonBossRespawns();
assert.strictEqual(getActiveBeatEnemies(game).length, 4,
  'resyncing the active beat should not duplicate enemies');

assert.strictEqual(game.completeDungeon(dungeon), false,
  'the route director should reject a completion call before Brambleking falls');
const untaggedBossData = Data.ENEMIES.find((entry) => entry.id === BRAMBLE.bossIds[0]);
const untaggedBoss = game.createEnemy(untaggedBossData, Object.assign(game.chooseBossSpawnPoint(0), {
  adminSpawned: true,
  preventWaveRespawn: true
}));
game.enemies.push(untaggedBoss);
game.defeatEnemy(untaggedBoss);
assert(!game.state.dungeons.completedDungeonIds.includes(BRAMBLE.dungeonId));
assert.strictEqual(getFlow(game).activeBeatIndex, 0);
assert.strictEqual(getFlow(game).defeatedSlotIds.length, 0,
  'an untagged matching boss must not receive route or dungeon completion credit');

activeEnemies = getActiveBeatEnemies(game);
game.defeatEnemy(activeEnemies[0]);
assert.strictEqual(getFlow(game).activeBeatIndex, 0);
assert.strictEqual(getFlow(game).defeatedSlotIds.length, 1);
activeEnemies.slice(1).forEach((enemy) => game.defeatEnemy(enemy));
assert.strictEqual(getFlow(game).activeBeatIndex, 1);
assert.strictEqual(game.getActiveDungeonRouteGateX(), 3200);
assert(!game.enemies.some((enemy) => enemy.hp > 0 && BRAMBLE.bossIds.includes(enemy.id)));

activeEnemies = getActiveBeatEnemies(game);
assert.strictEqual(activeEnemies.length, 4);
assert(activeEnemies.every((enemy) =>
  enemy.spawnSectionId === definition.beats[1].sectionIds[0] &&
  enemy.dungeonBeatId === definition.beats[1].id
));
assertEnemiesStayInsideSection(
  game,
  activeEnemies,
  definition.beats[1].sectionIds[0],
  'the second encounter should remain spatially inside Root Lanes'
);
activeEnemies.slice().forEach((enemy) => game.defeatEnemy(enemy));
assert.strictEqual(getFlow(game).activeBeatIndex, 2);
assert.strictEqual(getFlow(game).status, 'boss');
assert.strictEqual(game.getActiveDungeonRouteGateX(), 0);

const bosses = game.enemies.filter((enemy) => enemy.hp > 0 && BRAMBLE.bossIds.includes(enemy.id));
assert.strictEqual(bosses.length, 1, 'the final route beat should spawn one Brambleking');
assert(bosses.every((enemy) =>
  enemy.dungeonBeatId === definition.beats[2].id &&
  enemy.dungeonBeatDungeonId === BRAMBLE.dungeonId &&
  enemy.dungeonBeatMapId === BRAMBLE.mapId &&
  enemy.dungeonBeatRunStartedAt === game.state.dungeons.currentRun.startedAt &&
  enemy.spawnSectionId === definition.beats[2].sectionIds[0]
), 'Brambleking should carry the final beat and active-run tags');
assertEnemiesStayInsideSection(
  game,
  bosses,
  definition.beats[2].sectionIds[0],
  'Brambleking should reveal inside Court Gate'
);
const wallNow = Date.now() / 1000;
assert(bosses[0].hostileArmedAt > wallNow + 1 && bosses[0].hostileArmedAt < wallNow + 5,
  'Brambleking should receive the staged reveal delay in the runtime clock domain');
const bossHp = bosses[0].hp;
assert.strictEqual(game.damageEnemy(bosses[0], 9999, 'route-test'), false);
assert.strictEqual(bosses[0].hp, bossHp,
  'Brambleking should be invulnerable during the intro arm window');
assert.strictEqual(game.completeDungeon(dungeon), false);

const courtSection = map.spawnSections.find((section) => section.id === definition.beats[2].sectionIds[0]);
bosses[0].hostileArmedAt = 0;
game.state.player.x = courtSection.x - 300;
game.state.player.y = bosses[0].y + bosses[0].h - game.state.player.h;
game.state.player.invulnerableUntil = Number.MAX_VALUE;
for (let frame = 0; frame < 600; frame += 1) game.updateEnemies(1 / 60);
assert(
  bosses[0].x >= courtSection.x + 18 &&
  bosses[0].x + bosses[0].w <= courtSection.x + courtSection.w - 18,
  'an aggroed Brambleking should remain inside the authored Court Gate encounter pocket'
);

game.defeatEnemy(bosses[0]);
assert(game.state.dungeons.completedDungeonIds.includes(BRAMBLE.dungeonId));
assert.strictEqual(getFlow(game).status, 'complete');
assert.strictEqual(dungeonEngine.getDungeonEncounterCompletionBlockReason(
  BRAMBLE.dungeonId,
  game.state.dungeons.currentRun,
  { data: Data }
), '');
game.updateDungeonBossRespawns();
assert(!game.enemies.some((enemy) => enemy.hp > 0 && BRAMBLE.bossIds.includes(enemy.id)),
  'the legacy respawn loop should not recreate Brambleking after a route clear');
assert(game.isDungeonBossRespawning(BRAMBLE.dungeonId),
  'the legacy reward cooldown should still be recorded after a clear');
assert(game.startDungeon(BRAMBLE.dungeonId),
  'the authored route itself should remain replayable during the legacy boss cooldown');
assert.strictEqual(getFlow(game).status, 'active');
assert.strictEqual(getFlow(game).activeBeatIndex, 0);
assert.strictEqual(getActiveBeatEnemies(game).length, 4);
assert(!game.enemies.some((enemy) => enemy.hp > 0 && BRAMBLE.bossIds.includes(enemy.id)),
  'an immediate replay should restart at Ridge Return instead of restoring a completed boss state');

const partialGame = createLiveGame();
partialGame.defeatEnemy(getActiveBeatEnemies(partialGame)[0]);
const restoredPartialGame = restoreGame(partialGame.serialize());
assert.strictEqual(getFlow(restoredPartialGame).activeBeatIndex, 0);
assert.strictEqual(getFlow(restoredPartialGame).defeatedSlotIds.length, 1);
assert.strictEqual(getActiveBeatEnemies(restoredPartialGame).length, 3,
  'partial restore should spawn only undefeated slots from the first beat');
restoredPartialGame.spawnDungeonEncounterBeat(map);
assert.strictEqual(getActiveBeatEnemies(restoredPartialGame).length, 3,
  'restored beat spawning should remain idempotent');

while (getFlow(restoredPartialGame).status !== 'complete') {
  const remaining = getActiveBeatEnemies(restoredPartialGame);
  assert(remaining.length, 'a restored route should expose every remaining active slot');
  remaining.slice().forEach((enemy) => restoredPartialGame.defeatEnemy(enemy));
}
const restoredCompleteGame = restoreGame(restoredPartialGame.serialize());
assert.strictEqual(getFlow(restoredCompleteGame).status, 'complete');
assert.strictEqual(
  restoredCompleteGame.enemies.filter((enemy) => enemy.hp > 0 && enemy.dungeonBeatId).length,
  0,
  'completed route restore should not spawn any route enemy or boss'
);

console.log('Project Starfall Bramble dungeon route director tests passed.');
