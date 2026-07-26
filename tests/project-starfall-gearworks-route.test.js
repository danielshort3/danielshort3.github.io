'use strict';

const assert = require('assert');

const Data = require('../js/games/project-starfall/project-starfall-data.js');
const dungeonEngine = require('../js/games/project-starfall/engine/dungeons.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

const GEARWORKS = Object.freeze({
  dungeonId: 'gearworks_vault',
  mapId: 'gearworksVault',
  level: 35,
  stationId: 'gearworks_master_switch',
  bosses: Object.freeze(['clockworkTitan', 'quarryColossus']),
  beatIds: Object.freeze([
    'clear_intake_lane',
    'disable_clockwork_titan',
    'prime_master_gear',
    'break_quarry_colossus'
  ]),
  sectionIds: Object.freeze([
    'gearworksVault_intake_tank_lane',
    'gearworksVault_titan_assembly',
    'gearworksVault_master_gear_switch',
    'gearworksVault_assembly_core'
  ]),
  gates: Object.freeze([1250, 2500, 3050, 0])
});

const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  gearworksVault_intake_tank_lane: Object.freeze([
    'gearworksVault_intake_lane',
    'gearworksVault_intake_catwalk'
  ]),
  gearworksVault_titan_assembly: Object.freeze([
    'gearworksVault_titan_floor',
    'gearworksVault_sentry_catwalk'
  ]),
  gearworksVault_master_gear_switch: Object.freeze([
    'gearworksVault_switch_approach',
    'gearworksVault_master_switch_shelf'
  ]),
  gearworksVault_assembly_core: Object.freeze([
    'gearworksVault_core_floor',
    'gearworksVault_core_catwalk'
  ])
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

function getLivingBosses(game, bossId) {
  return game.enemies.filter((enemy) =>
    enemy && enemy.hp > 0 && (!bossId || enemy.id === bossId)
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
  check(game.chooseClass('fighter'), 'the Gearworks route fixture should choose Fighter');
  game.state.player.level = GEARWORKS.level;
  game.state.player.advancedClassId = 'guardian';
  check(game.startDungeon(GEARWORKS.dungeonId), 'the Gearworks route fixture should enter the dungeon');
  return game;
}

function restoreGame(payload) {
  const game = prepareGame(createProjectStarfallEngine(null, Data));
  check(game.restore(payload), 'the Gearworks route fixture should restore');
  return game;
}

function makeTaggedEnemy(run, beat, slot) {
  return {
    id: slot.enemyId,
    dungeonBeatId: beat.id,
    dungeonBeatSlotId: slot.id,
    dungeonBeatDungeonId: GEARWORKS.dungeonId,
    dungeonBeatMapId: GEARWORKS.mapId,
    dungeonBeatRunStartedAt: run.startedAt
  };
}

function createRecordingCanvasContext() {
  const operations = [];
  const properties = {};
  const context = new Proxy({ operations }, {
    get(target, property) {
      if (property in target) return target[property];
      if (property in properties) return properties[property];
      if (property === 'measureText') {
        return (value) => ({ width: String(value || '').length * 6 });
      }
      return (...args) => {
        operations.push(['call', String(property), ...args]);
      };
    },
    set(target, property, value) {
      properties[property] = value;
      operations.push(['set', String(property), value]);
      return true;
    }
  });
  return context;
}

function getGearSwitchDrawOperations(game, station) {
  const context = createRecordingCanvasContext();
  const shouldReduceEffects = game.shouldReduceEffects;
  game.shouldReduceEffects = () => true;
  game.drawGearworksMasterSwitch(context, station);
  game.drawDungeonRouteStationEffects(context, map);
  game.shouldReduceEffects = shouldReduceEffects;
  return context.operations;
}

function getGearSwitchPrompt(game, station) {
  const player = game.state.player;
  const priorStation = player.activeStation;
  const priorPortal = player.activePortalId;
  const priorQuestNpc = player.activeQuestNpcId;
  player.activeStation = station.id;
  player.activePortalId = '';
  player.activeQuestNpcId = '';
  const prompt = hud.getStationPromptContext({
    state: { player },
    runtime: { stations: [station] },
    map: { stations: [] },
    dungeon: game.getDungeonSnapshot(),
    portals: [],
    questNpcs: { npcs: [] }
  }, {
    keyLabels: { interact: 'F' }
  });
  player.activeStation = priorStation;
  player.activePortalId = priorPortal;
  player.activeQuestNpcId = priorQuestNpc;
  return prompt;
}

function triggerBossAddWave(game, boss, sectionId) {
  const encounter = game.getBossEncounterForEnemy(boss);
  const phase = encounter && (encounter.phases || []).find((entry) =>
    Array.isArray(entry.actions) && entry.actions.includes('addWave')
  );
  const target = game.getCombatCharacterByTarget('player', 'player');
  const existing = new Set(game.enemies);
  check(!!encounter && !!phase && !!target,
    `${boss.id} should expose its authored add-wave encounter phase`);
  game.beginBossEncounterAction(boss, encounter, phase, 'addWave', target);
  const pending = boss.bossPendingAction;
  check(pending && pending.spatialSectionId === sectionId,
    `${boss.id} add-wave telegraph should target its active Gearworks room`);
  game.resolveBossEncounterAction(boss, encounter, pending);
  return game.enemies.filter((enemy) =>
    enemy && !existing.has(enemy) && enemy.encounterMinion
  );
}

function assertBossAddsOwnedByBeat(game, adds, beat, section, message) {
  check(adds.length === 2,
    `${message} should create exactly two authored adds`);
  check(adds.every((add) =>
    add.dungeonEncounterAdd &&
    add.dungeonEncounterFlowId === getFlow(game).id &&
    add.dungeonEncounterParentBeatId === beat.id &&
    add.dungeonEncounterDungeonId === GEARWORKS.dungeonId &&
    add.dungeonEncounterMapId === GEARWORKS.mapId &&
    add.dungeonEncounterRunStartedAt === game.state.dungeons.currentRun.startedAt &&
    add.preventWaveRespawn &&
    add.spawnSectionId === section.id &&
    add.x >= section.x &&
    add.x + add.w <= section.x + section.w
  ), `${message} adds should be section-owned, bounded, and excluded from legacy waves`);
  adds.forEach((add) => {
    const wander = game.getEnemyWanderBounds(add);
    check(wander &&
      wander.left >= section.x &&
      wander.right + add.w <= section.x + section.w,
    `${message} add ${add.id} should remain inside the ${section.label} encounter room`);
  });
  check(game.enemies.some((enemy) =>
    enemy && enemy.hp > 0 &&
    enemy.id === beat.bossIds[0] &&
    enemy.dungeonBeatSlotId
  ) &&
    adds.every((add) => !add.dungeonBeatSlotId),
  `${message} adds should not impersonate the route boss slot`);
}

function assertArenaEdgeClamps(game, section, label) {
  const player = game.state.player;
  player.x = section.x + 80;
  const armedPlan = game.applyDungeonRouteGateClamp(player.x, { silent: true });
  check(!armedPlan.blocked && armedPlan.x >= section.x + 18,
    `${label} should arm its entrance seal only after the player enters the room`);
  const leftPlan = game.applyDungeonRouteGateClamp(section.x - 240, { silent: true });
  const rightPlan = game.applyDungeonRouteGateClamp(section.x + section.w + 240, { silent: true });
  check(leftPlan.blocked &&
    leftPlan.x >= section.x + 18,
  `${label} should seal the player inside its left arena edge`);
  check(rightPlan.blocked &&
    rightPlan.x <= section.x + section.w - player.w - 18,
  `${label} should seal the player inside its right arena edge`);
  player.x = section.x - 240;
  game.updatePlayer(0);
  check(player.x >= section.x + 18,
    `${label} normal movement should enforce the left arena seal`);
  player.x = section.x + section.w + 240;
  game.updatePlayer(0);
  check(player.x <= section.x + section.w - player.w - 18,
    `${label} normal movement should enforce the right arena seal`);
}

const dungeon = Data.DUNGEONS.find((entry) => entry.id === GEARWORKS.dungeonId);
const map = Data.MAPS.find((entry) => entry.id === GEARWORKS.mapId);
const definition = dungeonEngine.getDungeonEncounterFlowDefinition(GEARWORKS.dungeonId, {
  data: Data
});

check(!!dungeon && !!map && !!definition,
  'Gearworks Vault should publish a staged encounter route');
check(map.asset === 'img/project-starfall/maps/gearworks-vault.webp' &&
  map.environment.terrain === 'gearworks-vault' &&
  map.environment.props === 'gearworks-vault' &&
  map.palette.join('|') === '#665b48|#7a8592|#29b3ad',
'the route pass should preserve Gearworks Vault\'s playful painting, machinery props, and brass-teal palette');
const gearworksBosses = GEARWORKS.bosses.map((bossId) =>
  Data.ENEMIES.find((enemy) => enemy.id === bossId)
);
check(gearworksBosses.every((boss) =>
  boss &&
  boss.asset === `img/project-starfall/enemies/${
    boss.id === 'clockworkTitan' ? 'clockwork-titan' : 'quarry-colossus'
  }.png` &&
  boss.animation &&
  boss.animation.sheet.endsWith('-compact-sheet.png') &&
  boss.animation.frameWidth === 128 &&
  boss.animation.frameHeight === 128
),
'the two construct bosses should retain their existing compact playful sprite assets and proportions');
check(map.layoutStyle === 'dungeonArena' &&
  map.geometryGenerator === 'dungeonArena' &&
  map.arenaSkeleton === 'staged-gearworks-route' &&
  map.platforms[0].w === 4600,
'Gearworks should retain its compact chibi factory scale while using an authored dungeon route');
check(map.designIntent.visualIdentityTag === 'playful brass and teal gear vault' &&
  map.designIntent.implementationStatus === 'geometry-route-v1' &&
  map.designIntent.routeSummary.includes('Master Gear Switch'),
'the design contract should describe the staged Gearworks route without replacing its playful identity');

const routeSections = map.fieldComposition.routeSections;
check(routeSections.map((section) => `${section.label}:${section.x}:${section.w}`).join('|') ===
  'Intake Tank Lane:0:1250|Titan Assembly:1250:1250|Master Gear Switch:2500:550|Assembly Core:3050:1550',
'the vault should read as four ordered map-design beats instead of three equal decorative thirds');
check(routeSections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should begin where the previous beat ends`);
  return section.x + section.w;
}, 0) === map.platforms[0].w,
'the four Gearworks route beats should cover the authored ground without gaps or overlap');

const sectionById = new Map(map.spawnSections.map((section) => [section.id, section]));
const groupBySectionId = new Map(map.spawnGroups.map((group) => [group.sectionId, group]));
const claimedPlatformIds = new Set();
Object.entries(EXPECTED_SECTION_PLATFORMS).forEach(([sectionId, platformIds]) => {
  const section = sectionById.get(sectionId);
  check(!!section, `${sectionId} should publish one semantic route section`);
  check(sameMembers(section.platformIds, platformIds),
    `${sectionId} should own its authored Gearworks platforms`);
  check(platformIds.every((platformId) => {
    if (claimedPlatformIds.has(platformId)) return false;
    claimedPlatformIds.add(platformId);
    return true;
  }), `${sectionId} should not share a platform with another route beat`);
});
check(claimedPlatformIds.size === 8 &&
  Array.from(claimedPlatformIds).every((platformId) =>
    map.platforms.some((platform) => platform.id === platformId)
  ),
'all eight semantic Gearworks platforms should resolve to real authored geometry');

const combatSectionIds = GEARWORKS.sectionIds.filter((sectionId) =>
  sectionId !== 'gearworksVault_master_gear_switch'
);
check(map.spawnGroups.length === 3 &&
  sameMembers(map.spawnGroups.map((group) => group.sectionId), combatSectionIds),
'Gearworks should publish combat only for the intake, Titan assembly, and final core');
map.spawnGroups.forEach((group) => {
  const section = sectionById.get(group.sectionId);
  check(!!section &&
    group.platformIds.length > 0 &&
    group.platformIds.every((platformId) => section.platformIds.includes(platformId)),
  `${group.sectionId} enemies should remain inside their semantic platform territory`);
  check(group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${group.sectionId} spawn bounds should stay inside the named route beat`);
});

const switchSection = sectionById.get('gearworksVault_master_gear_switch');
const switchPlatformIds = new Set(switchSection.platformIds);
const switchShelf = map.platforms.find((platform) =>
  platform.id === 'gearworksVault_master_switch_shelf'
);
const authoredSwitch = map.stations.find((station) => station.id === GEARWORKS.stationId);
const authoredSwitchPlatform = map.platforms[Number(authoredSwitch && authoredSwitch.platformIndex)];
check(switchShelf &&
  switchShelf.spawnDisabled === true,
'the Master Gear Switch should occupy an explicitly spawn-free objective shelf');
check(authoredSwitch &&
  authoredSwitch.name === 'Master Gear Switch' &&
  authoredSwitch.serviceRole === 'dungeon_objective' &&
  authoredSwitchPlatform === switchShelf &&
  authoredSwitch.x >= switchShelf.x + 24 &&
  authoredSwitch.x + 88 <= switchShelf.x + switchShelf.w - 24,
'the physical Master Gear Switch should be supported by its shelf with readable side clearance');
check(map.spawnGroups.every((group) =>
  group.sectionId !== switchSection.id &&
  group.platformIds.every((platformId) => !switchPlatformIds.has(platformId))
) &&
  map.spawnPoints.every((point) =>
    point.sectionId !== switchSection.id &&
    !switchPlatformIds.has(point.platformId)
  ),
'the switch shelf and its approach should contain no encounter group or spawn anchor');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: Data.MAPS });
const reachablePlatformIndices = mapRuntime.getReachablePlatformIndices(runtime.platformGraph, 0);
const runtimeSwitch = runtime.stations.find((station) => station.id === GEARWORKS.stationId);
check(Array.from(claimedPlatformIds).every((platformId) => {
  const platform = runtime.platforms.find((entry) => entry.id === platformId);
  return platform && reachablePlatformIndices.has(platform.index);
}), 'every semantic Gearworks platform should be traversable from the continuous factory floor');
check(runtimeSwitch &&
  runtimeSwitch.platformId === switchShelf.id &&
  reachablePlatformIndices.has(runtimeSwitch.platformIndex) &&
  runtimeSwitch.y + runtimeSwitch.h >= runtime.platforms[runtimeSwitch.platformIndex].y,
'the live Master Gear Switch should align to a reachable supporting shelf');
const validation = validateMap(map);
check(validation.issues.length === 0,
  `Gearworks should satisfy the shared map validator: ${validation.issues.join(' | ')}`);

check(definition.dungeonId === GEARWORKS.dungeonId &&
  definition.mapId === GEARWORKS.mapId &&
  definition.beats.length === 4,
'the Gearworks definition should bind four beats to the real dungeon map');
check(definition.beats.map((beat) => beat.id).join('|') === GEARWORKS.beatIds.join('|') &&
  definition.beats.map((beat) => beat.kind).join('|') === 'combat|boss|interaction|boss' &&
  definition.beats.map((beat) => beat.sectionIds[0]).join('|') === GEARWORKS.sectionIds.join('|') &&
  definition.beats.map((beat) => beat.gateX).join('|') === GEARWORKS.gates.join('|'),
'the route should sequence intake combat, Titan miniboss, gear switch, then Colossus boss');
check(definition.beats.map((beat) => beat.entryGateX).join('|') === '0|1250|0|3050',
  'both Gearworks bosses should publish a rear arena seal in addition to their forward room edge');
check(definition.beats[0].enemyIds.length === 4 &&
  sameMembers(definition.beats[1].bossIds, ['clockworkTitan']) &&
  sameMembers(definition.beats[2].stationIds, [GEARWORKS.stationId]) &&
  sameMembers(definition.beats[3].bossIds, ['quarryColossus']),
'each route beat should publish only the enemies, switch, or boss it owns');
definition.beats.forEach((beat) => {
  check(sectionById.has(beat.sectionIds[0]),
    `${beat.id} should target a published Gearworks section`);
});
const spatialHooks = Data.BOSS_SPATIAL_MECHANICS.gearworksVault.hooks;
['gearSlam', 'gearLane', 'plateExpose', 'overclock'].forEach((actionId) => {
  check(spatialHooks[actionId].sectionId === 'gearworksVault_titan_assembly',
    `${actionId} should stay inside the Clockwork Titan room before the switch opens`);
});
['rockfall', 'quakeAnchor', 'corePulse', 'addWave'].forEach((actionId) => {
  check(spatialHooks[actionId].sectionId === 'gearworksVault_assembly_core',
    `${actionId} should stay inside the opened Quarry Colossus room`);
});

const pureRun = dungeonEngine.createDungeonStartRunState(GEARWORKS.dungeonId, false, {
  data: Data,
  startedAt: 1000,
  nowMs: 1000
});
const firstSlots = dungeonEngine.createDungeonEncounterBeatSlots(definition.beats[0]);
firstSlots.forEach((slot, index) => {
  const result = dungeonEngine.recordDungeonEncounterEnemyDefeat(
    pureRun,
    makeTaggedEnemy(pureRun, definition.beats[0], slot),
    { data: Data, mapId: GEARWORKS.mapId, nowMs: 1100 + index }
  );
  check(result.accepted && result.advanced === (index === firstSlots.length - 1),
    'each intake slot should count once and only the final slot should advance');
});
check(pureRun.encounterFlow.activeBeatId === definition.beats[1].id &&
  pureRun.encounterFlow.status === 'boss',
'clearing the intake should reveal only the Clockwork Titan beat');

const titanSlot = dungeonEngine.createDungeonEncounterBeatSlots(definition.beats[1])[0];
const titanResult = dungeonEngine.recordDungeonEncounterEnemyDefeat(
  pureRun,
  makeTaggedEnemy(pureRun, definition.beats[1], titanSlot),
  { data: Data, mapId: GEARWORKS.mapId, nowMs: 1200 }
);
check(titanResult.accepted && titanResult.advanced &&
  !titanResult.complete &&
  !titanResult.bossRevealed &&
  pureRun.encounterFlow.status === 'interaction',
'defeating Clockwork Titan should advance to the physical switch instead of completing the dungeon');

const finalSlot = dungeonEngine.createDungeonEncounterBeatSlots(definition.beats[3])[0];
const earlyFinalResult = dungeonEngine.recordDungeonEncounterEnemyDefeat(
  pureRun,
  makeTaggedEnemy(pureRun, definition.beats[3], finalSlot),
  { data: Data, mapId: GEARWORKS.mapId, nowMs: 1300 }
);
check(!earlyFinalResult.accepted &&
  earlyFinalResult.reason === 'wrong-beat' &&
  pureRun.encounterFlow.activeBeatId === definition.beats[2].id,
'the Quarry Colossus cannot receive route credit before the Master Gear Switch is activated');

const interactionSnapshot = dungeonEngine.createDungeonEncounterFlowSnapshot(
  GEARWORKS.dungeonId,
  pureRun,
  { data: Data, nowMs: 1300 }
);
check(interactionSnapshot.status === 'interaction' &&
  interactionSnapshot.activeGateX === 3050 &&
  interactionSnapshot.hud.title === 'Route 3/4' &&
  interactionSnapshot.hud.kind === 'interaction' &&
  interactionSnapshot.hud.value === 0 &&
  interactionSnapshot.hud.goal === 1 &&
  interactionSnapshot.hud.status === '1 route objective remaining',
'the route snapshot should guide the player to one physical switch behind the third gate');

const restoredPureState = dungeonEngine.createDungeonState(JSON.parse(JSON.stringify({
  activeDungeonId: GEARWORKS.dungeonId,
  currentRun: pureRun
})), { data: Data });
check(restoredPureState.currentRun.encounterFlow.status === 'interaction' &&
  restoredPureState.currentRun.encounterFlow.activeBeatId === definition.beats[2].id &&
  restoredPureState.currentRun.encounterFlow.completedInteractionIds.length === 0,
'save normalization should preserve the unclaimed switch beat without revealing the final boss');
const pureRunSnapshot = dungeonEngine.createDungeonRunSnapshot(pureRun);
pureRunSnapshot.encounterFlow.completedInteractionIds.push('snapshot_only');
check(!pureRun.encounterFlow.completedInteractionIds.includes('snapshot_only'),
  'run snapshots should deep-clone interaction progress');

const switchResult = dungeonEngine.recordDungeonEncounterInteraction(
  pureRun,
  GEARWORKS.stationId,
  { data: Data, mapId: GEARWORKS.mapId, nowMs: 1400 }
);
check(switchResult.accepted &&
  switchResult.advanced &&
  switchResult.bossRevealed &&
  !switchResult.complete &&
  pureRun.encounterFlow.activeBeatId === definition.beats[3].id,
'activating the Master Gear Switch should reveal the Quarry Colossus exactly once');
const duplicateSwitchResult = dungeonEngine.recordDungeonEncounterInteraction(
  pureRun,
  GEARWORKS.stationId,
  { data: Data, mapId: GEARWORKS.mapId, nowMs: 1401 }
);
check(!duplicateSwitchResult.accepted &&
  pureRun.encounterFlow.completedInteractionIds.length === 1 &&
  pureRun.encounterFlow.activeBeatId === definition.beats[3].id,
'reusing the switch after it advances should not duplicate progress or skip the final boss');
check(!dungeonEngine.isDungeonEncounterFlowComplete(GEARWORKS.dungeonId, pureRun, { data: Data }),
  'switch activation alone should not complete Gearworks Vault');
const finalPureResult = dungeonEngine.recordDungeonEncounterEnemyDefeat(
  pureRun,
  makeTaggedEnemy(pureRun, definition.beats[3], finalSlot),
  { data: Data, mapId: GEARWORKS.mapId, nowMs: 1500 }
);
check(finalPureResult.accepted &&
  finalPureResult.complete &&
  dungeonEngine.isDungeonEncounterFlowComplete(GEARWORKS.dungeonId, pureRun, { data: Data }),
'only the tagged Quarry Colossus defeat should complete the staged route');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const game = createLiveGame();
  check(game.state.mapId === GEARWORKS.mapId &&
    getFlow(game).activeBeatIndex === 0 &&
    getFlow(game).spawnedBeatIds.join('|') === definition.beats[0].id,
  'Gearworks entry should start and spawn only the first intake beat');
  check(getActiveBeatEnemies(game).length === 4 &&
    !getLivingBosses(game, 'clockworkTitan').length &&
    !getLivingBosses(game, 'quarryColossus').length,
  'neither heavy boss should appear at dungeon entry');
  const entrySwitch = game.runtime.stations.find((station) =>
    station.id === GEARWORKS.stationId
  );
  const lockedSwitchDraw = getGearSwitchDrawOperations(game, entrySwitch);
  const lockedSwitchPrompt = getGearSwitchPrompt(game, entrySwitch);
  check(lockedSwitchPrompt &&
    lockedSwitchPrompt.hint === 'Route locked' &&
    lockedSwitchPrompt.kindLabel === 'Dungeon objective',
  'the untouched switch should clearly explain that its route is still locked');
  game.state.player.activeStation = GEARWORKS.stationId;
  game.state.player.activePortalId = '';
  check(!game.interact({ silent: true }) &&
    getFlow(game).activeBeatIndex === 0 &&
    !game.lastInteractionOpenedPanel,
  'using the locked switch should neither advance the route nor open a generic station panel');
  game.state.player.activeStation = '';
  check(game.getActiveDungeonRouteGateX() === 1250,
    'the opening route seal should hold the player inside the intake lane');
  game.spawnDungeonEncounterBeat(map);
  game.updateDungeonBossRespawns();
  check(getActiveBeatEnemies(game).length === 4,
    'resyncing the opening beat should not duplicate its four enemies');
  check(!game.completeDungeon(dungeon),
    'the dungeon clear should be blocked before any route beat is complete');

  getActiveBeatEnemies(game).slice().forEach((enemy) => game.defeatEnemy(enemy));
  check(getFlow(game).activeBeatIndex === 1 &&
    getFlow(game).status === 'boss' &&
    game.getActiveDungeonRouteGateX() === 2500,
  'clearing the intake should move the route seal to the far side of Titan Assembly');
  let activeEnemies = getActiveBeatEnemies(game);
  check(activeEnemies.length === 1 &&
    activeEnemies[0].id === 'clockworkTitan' &&
    !getLivingBosses(game, 'quarryColossus').length,
  'the second beat should spawn one Clockwork Titan and keep the Colossus sealed');
  game.spawnDungeonEncounterBeat(map);
  check(getActiveBeatEnemies(game).length === 1,
    'resyncing the Titan beat should remain idempotent');

  const titanSection = sectionById.get('gearworksVault_titan_assembly');
  assertArenaEdgeClamps(game, titanSection, 'Clockwork Titan Assembly');
  const titanFloor = game.runtime.platforms.find((platform) =>
    platform.id === 'gearworksVault_titan_floor'
  );
  game.placePlayerOnRuntimePlatform(titanFloor.index, titanFloor.x + titanFloor.w / 2);
  const titan = activeEnemies[0];
  const titanAdds = triggerBossAddWave(
    game,
    titan,
    'gearworksVault_titan_assembly'
  );
  assertBossAddsOwnedByBeat(
    game,
    titanAdds,
    definition.beats[1],
    titanSection,
    'Clockwork Titan add wave'
  );
  const titanDefeatedSlotsBeforeAdd = getFlow(game).defeatedSlotIds.length;
  const titanWavePendingBeforeAdd = game.getWaveState(GEARWORKS.mapId).pending.length;
  game.defeatEnemy(titanAdds[0]);
  check(getFlow(game).activeBeatIndex === 1 &&
    getFlow(game).defeatedSlotIds.length === titanDefeatedSlotsBeforeAdd,
  'defeating a Titan add should not receive boss-slot credit or advance the route');
  check(game.getWaveState(GEARWORKS.mapId).pending.length === titanWavePendingBeforeAdd,
    'defeating a Titan add should not enqueue a legacy wave replacement');

  game.defeatEnemy(activeEnemies[0]);
  check(getFlow(game).activeBeatIndex === 2 &&
    getFlow(game).status === 'interaction' &&
    game.getActiveDungeonRouteGateX() === 3050 &&
    !getLivingBosses(game, 'quarryColossus').length,
  'defeating Clockwork Titan should expose the switch beat, not the final boss');
  check(!game.enemies.some((enemy) =>
    enemy && enemy.hp > 0 && enemy.encounterMinion
  ), 'advancing past Clockwork Titan should purge its surviving add instead of leaking it into the switch beat');
  const liveInteractionSnapshot = game.getDungeonSnapshot().activeDungeon.encounterFlow;
  check(liveInteractionSnapshot &&
    liveInteractionSnapshot.hud.title === 'Route 3/4' &&
    liveInteractionSnapshot.hud.kind === 'interaction' &&
    liveInteractionSnapshot.hud.goal === 1,
  'the live HUD contract should clearly identify the one-step switch objective');
  game.spawnDungeonEncounterBeat(map);
  game.updateDungeonBossRespawns();
  check(!getLivingBosses(game, 'quarryColossus').length,
    'spawn and respawn synchronization must not bypass the switch gate');

  const switchSave = game.serialize();
  const switchRestore = restoreGame(switchSave);
  check(getFlow(switchRestore).activeBeatIndex === 2 &&
    getFlow(switchRestore).status === 'interaction' &&
    getFlow(switchRestore).completedInteractionIds.length === 0 &&
    !getLivingBosses(switchRestore, 'quarryColossus').length,
  'restoring at the switch should remain on the interaction beat with no leaked final boss');
  switchRestore.spawnDungeonEncounterBeat(map);
  check(!getLivingBosses(switchRestore, 'quarryColossus').length,
    'resyncing a restored switch beat should remain idempotent');

  const liveSwitch = switchRestore.runtime.stations.find((station) =>
    station.id === GEARWORKS.stationId
  );
  const readySwitchDraw = getGearSwitchDrawOperations(switchRestore, liveSwitch);
  const readySwitchPrompt = getGearSwitchPrompt(switchRestore, liveSwitch);
  check(readySwitchPrompt &&
    readySwitchPrompt.hint === 'F Activate' &&
    readySwitchPrompt.kindLabel === 'Dungeon objective - Ready',
  'the live switch prompt should become an explicit activation objective');
  check(switchRestore.placePlayerOnRuntimePlatform(liveSwitch.platformIndex, liveSwitch.x),
    'the fixture should place the player on the physical switch shelf');
  switchRestore.updateActiveStation();
  check(switchRestore.state.player.activeStation === GEARWORKS.stationId,
    'standing at the Master Gear Switch should select the dungeon objective');
  check(switchRestore.interact({ silent: true }),
    'interacting with the physical Master Gear Switch should advance the route');
  check(getFlow(switchRestore).activeBeatIndex === 3 &&
    getFlow(switchRestore).completedInteractionIds.join('|') === GEARWORKS.stationId &&
    game.getActiveDungeonRouteGateX() === 3050,
  'the switch should persist one completion record and leave the pre-interaction fixture unchanged');
  const primedSwitchDraw = getGearSwitchDrawOperations(switchRestore, liveSwitch);
  const primedSwitchPrompt = getGearSwitchPrompt(switchRestore, liveSwitch);
  check(primedSwitchPrompt &&
    primedSwitchPrompt.hint === 'Switch primed' &&
    primedSwitchPrompt.kindLabel === 'Dungeon objective',
  'the switch prompt should visibly settle into a completed primed state');
  const lockedSwitchSignature = JSON.stringify(lockedSwitchDraw);
  const readySwitchSignature = JSON.stringify(readySwitchDraw);
  const primedSwitchSignature = JSON.stringify(primedSwitchDraw);
  check(lockedSwitchSignature !== readySwitchSignature &&
    readySwitchSignature !== primedSwitchSignature &&
    lockedSwitchSignature !== primedSwitchSignature,
  'the physical switch should draw distinct locked, ready, and primed states');
  check([lockedSwitchSignature, readySwitchSignature, primedSwitchSignature].every((signature) =>
    signature.includes('#d8b65c') &&
    signature.includes('#29b3ad') &&
    signature.includes('GEAR')
  ), 'every switch state should preserve the existing playful brass, teal, and GEAR-machine art');
  const finalBossCountBeforeDuplicateSwitch = getLivingBosses(
    switchRestore,
    'quarryColossus'
  ).length;
  check(!switchRestore.interact({ silent: true }) &&
    getFlow(switchRestore).activeBeatIndex === 3 &&
    getLivingBosses(switchRestore, 'quarryColossus').length === finalBossCountBeforeDuplicateSwitch &&
    !switchRestore.lastInteractionOpenedPanel,
  'reusing the primed switch should give completed feedback without duplicating the boss or opening a panel');
  activeEnemies = getActiveBeatEnemies(switchRestore);
  check(activeEnemies.length === 1 &&
    activeEnemies[0].id === 'quarryColossus' &&
    switchRestore.getActiveDungeonRouteGateX() === 0,
  'the switch should reveal one Quarry Colossus and open the Assembly Core');
  switchRestore.spawnDungeonEncounterBeat(map);
  switchRestore.updateDungeonBossRespawns();
  check(getActiveBeatEnemies(switchRestore).length === 1,
    'repeated final-beat synchronization should not duplicate the Quarry Colossus');
  check(!switchRestore.completeDungeon(dungeon),
    'Gearworks should remain incomplete until the final boss falls');

  const finalSave = switchRestore.serialize();
  const finalRestore = restoreGame(finalSave);
  activeEnemies = getActiveBeatEnemies(finalRestore);
  check(getFlow(finalRestore).activeBeatIndex === 3 &&
    getFlow(finalRestore).completedInteractionIds.join('|') === GEARWORKS.stationId &&
    activeEnemies.length === 1 &&
    activeEnemies[0].id === 'quarryColossus',
  'restoring after switch activation should recreate exactly one tagged final boss');
  finalRestore.spawnDungeonEncounterBeat(map);
  check(getActiveBeatEnemies(finalRestore).length === 1,
    'restored final-boss spawning should remain idempotent');
  const coreSection = sectionById.get('gearworksVault_assembly_core');
  assertArenaEdgeClamps(finalRestore, coreSection, 'Quarry Colossus Assembly Core');
  const coreFloor = finalRestore.runtime.platforms.find((platform) =>
    platform.id === 'gearworksVault_core_floor'
  );
  finalRestore.placePlayerOnRuntimePlatform(coreFloor.index, coreFloor.x + coreFloor.w / 2);
  const colossus = activeEnemies[0];
  const colossusAdds = triggerBossAddWave(
    finalRestore,
    colossus,
    'gearworksVault_assembly_core'
  );
  assertBossAddsOwnedByBeat(
    finalRestore,
    colossusAdds,
    definition.beats[3],
    coreSection,
    'Quarry Colossus add wave'
  );
  const finalDefeatedSlotsBeforeAdd = getFlow(finalRestore).defeatedSlotIds.length;
  const finalWavePendingBeforeAdd = finalRestore.getWaveState(GEARWORKS.mapId).pending.length;
  finalRestore.defeatEnemy(colossusAdds[0]);
  check(getFlow(finalRestore).activeBeatIndex === 3 &&
    getFlow(finalRestore).defeatedSlotIds.length === finalDefeatedSlotsBeforeAdd,
  'defeating a Colossus add should not receive final-boss credit');
  check(finalRestore.getWaveState(GEARWORKS.mapId).pending.length === finalWavePendingBeforeAdd,
    'defeating a Colossus add should not enqueue a legacy wave replacement');
  finalRestore.defeatEnemy(activeEnemies[0]);
  check(getFlow(finalRestore).status === 'complete' &&
    finalRestore.state.dungeons.completedDungeonIds.includes(GEARWORKS.dungeonId) &&
    dungeonEngine.getDungeonEncounterCompletionBlockReason(
      GEARWORKS.dungeonId,
      finalRestore.state.dungeons.currentRun,
      { data: Data }
    ) === '',
  'defeating the one tagged Quarry Colossus should complete and reward Gearworks Vault');
  check(!finalRestore.enemies.some((enemy) =>
    enemy && enemy.hp > 0 && enemy.encounterMinion
  ), 'completing Gearworks should purge the final boss\'s surviving add');
  finalRestore.updateDungeonBossRespawns();
  check(!getLivingBosses(finalRestore, 'quarryColossus').length,
    'the legacy respawn loop should not recreate the final boss after a route clear');

  const partialGame = createLiveGame();
  const partialEnemies = getActiveBeatEnemies(partialGame);
  partialGame.defeatEnemy(partialEnemies[0]);
  const partialRestore = restoreGame(partialGame.serialize());
  check(getFlow(partialRestore).activeBeatIndex === 0 &&
    getFlow(partialRestore).defeatedSlotIds.length === 1 &&
    getActiveBeatEnemies(partialRestore).length === 3,
  'a partial intake restore should spawn only the three undefeated slots');
  partialRestore.spawnDungeonEncounterBeat(map);
  check(getActiveBeatEnemies(partialRestore).length === 3,
    'partial intake restoration should remain spawn-idempotent');
} finally {
  Math.random = originalRandom;
}

console.log(`Project Starfall Gearworks route checks passed: ${checks}`);
