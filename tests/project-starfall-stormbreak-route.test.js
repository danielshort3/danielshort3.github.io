'use strict';

const assert = require('assert');

const data = require('../js/games/project-starfall/data/index.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const { getLightningRodReadyPulseStyle } = require('../js/games/project-starfall/project-starfall-ui.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

const MAP_ID = 'stormbreakCliffs';
const STATION_ID = 'stormbreak_lightning_rod';
const OBJECTIVE_SECTION_ID = 'stormbreakCliffs_lightning_rod_objective';
const ACTIVE_SECTION_IDS = Object.freeze([
  'stormbreakCliffs_low_ram_lane',
  'stormbreakCliffs_mid_archer_bridge',
  'stormbreakCliffs_high_harrier_airspace'
]);
const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  stormbreakCliffs_low_ram_lane: Object.freeze([
    'stormbreak_cliffs_solid_lane_01',
    'stormbreak_cliffs_solid_lane_02',
    'stormbreak_cliffs_solid_lane_03'
  ]),
  stormbreakCliffs_mid_archer_bridge: Object.freeze([
    'stormbreak_cliffs_solid_lane_04',
    'stormbreak_cliffs_solid_lane_05',
    'stormbreak_cliffs_solid_lane_06'
  ]),
  stormbreakCliffs_high_harrier_airspace: Object.freeze([
    'stormbreak_cliffs_solid_lane_07',
    'stormbreak_cliffs_solid_lane_08',
    'stormbreak_cliffs_solid_lane_09',
    'stormbreak_cliffs_solid_lane_10'
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

function makeMechanicEnemy(sectionId, behavior) {
  return {
    spawnSectionId: sectionId,
    data: { behavior: behavior || 'ground' }
  };
}

function getSectionHits(snapshot) {
  return Object.fromEntries((snapshot.sections || []).map((section) => [section.id, section.hits]));
}

const map = data.MAPS.find((entry) => entry.id === MAP_ID);
const route = data.WORLD_ROUTES.find((entry) => entry.id === 'ascension');
const routeGoal = route && route.fieldGoals.find((entry) => entry.mapId === MAP_ID);

check(!!map && !!routeGoal,
  'Stormbreak Cliffs and its Ascension route goal should remain published');
check(hud.isStormbreakLightningRodStationId(STATION_ID) &&
  !hud.isStormbreakLightningRodStationId('future_map_objective') &&
  hud.getStormbreakLightningRodPresentation({
    mapModifiers: {
      mapMechanic: {
        id: 'future_map_objective',
        objectiveStationId: 'future_map_objective',
        objectiveReady: true
      }
    },
    runtime: {
      stations: [{
        id: 'future_map_objective',
        serviceRole: 'map_objective',
        x: 100,
        y: 100,
        w: 88,
        h: 56
      }]
    }
  }) === null,
'future map objectives should keep generic presentation instead of inheriting Stormbreak Rod behavior');
const reducedPulseStart = getLightningRodReadyPulseStyle(0, true);
const reducedPulseLater = getLightningRodReadyPulseStyle(900, true);
const animatedPulseLater = getLightningRodReadyPulseStyle(180, false);
check(JSON.stringify(reducedPulseStart) === JSON.stringify(reducedPulseLater) &&
  JSON.stringify(reducedPulseStart) !== JSON.stringify(animatedPulseLater),
'reduced-effects mode should retain a readable but completely static Lightning Rod marker');
check(map.asset === 'img/project-starfall/maps/stormbreak-cliffs.webp' &&
  map.environment.terrain === 'stormbreak-cliffs' &&
  map.environment.props === 'stormbreak-cliffs' &&
  map.palette.join('|') === '#4f6073|#91dbe8|#ffe16a',
'the route pass should preserve Stormbreak\'s playful painting, terrain, props, and palette');
check(map.layoutStyle === 'stormClimb' &&
  map.geometryGenerator === 'fieldLayout' &&
  map.platforms[0].w === 5200 &&
  map.worldHeight === 1260 &&
  map.authoredGroundY === 1120 &&
  map.platforms.length === 25 &&
  map.rampConnections.length === 6 &&
  map.climbables.length === 11,
'Stormbreak should keep its authored climb dimensions, platforms, ramps, and stairs');

const sections = map.fieldComposition.routeSections;
check(sections.map((section) => `${section.label}:${section.x}:${section.w}`).join('|') ===
  'Low Ram Lane:0:1300|Mid Archer Bridge:1300:1500|High Harrier Airspace:2800:1400|Lightning Rod Objective:4200:1000',
'the existing route should retain its four named map-design beats');
check(sections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === map.platforms[0].w,
'the four named route beats should cover the field without metadata gaps');
check(map.fieldComposition.landmarkBands.map((band) => band.label).join('|') ===
  'Ram Lane|Archer Bridge|Harrier Airspace|Lightning Rod' &&
  map.designIntent.visualIdentityTag === 'storm mast cliff climb' &&
  map.designIntent.routeSummary.includes('spawn-free lightning rod perch'),
'Stormbreak should keep its playful cliff landmarks while documenting the calm regroup finish');

const sectionsById = new Map(map.spawnSections.map((section) => [section.id, section]));
const groupsBySectionId = new Map(map.spawnGroups.map((group) => [group.sectionId, group]));
const claimedPlatformIds = new Set();
check(map.spawnGroups.length === 3 &&
  sameMembers(map.spawnGroups.map((group) => group.sectionId), ACTIVE_SECTION_IDS),
'Stormbreak should publish exactly the low, mid, and high combat groups');
Object.entries(EXPECTED_SECTION_PLATFORMS).forEach(([sectionId, expectedPlatformIds]) => {
  const section = sectionsById.get(sectionId);
  const group = groupsBySectionId.get(sectionId);
  check(!!section && !!group,
    `${sectionId} should publish a matching route section and encounter group`);
  check(sameMembers(section.platformIds, expectedPlatformIds),
    `${sectionId} should own its authored combat platforms`);
  check(sameMembers(group.platformIds, expectedPlatformIds),
    `${sectionId} encounters should stay inside their authored platform territory`);
  check(group.platformIds.every((platformId) => {
    if (claimedPlatformIds.has(platformId)) return false;
    claimedPlatformIds.add(platformId);
    return true;
  }), `${sectionId} should not share combat platforms with another group`);
  check(group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${sectionId} spawn bounds should stay inside its named route beat`);
});
check(map.spawnGroups.every((group) =>
  group.population === 12 &&
  group.maxPopulation === 15 &&
  group.partyBonusPerMember === 1
) &&
  map.spawnGroups.reduce((total, group) => total + group.population, 0) === 36 &&
  map.waveMax === routeGoal.count &&
  routeGoal.count === 36,
'one balanced 36-enemy solo circuit should fulfill the Ascension gate while retaining a bounded party-density budget');

const objectiveSection = sections.find((section) => section.label === 'Lightning Rod Objective');
const objectiveSpawnSection = sectionsById.get(OBJECTIVE_SECTION_ID);
const authoredStation = map.stations.find((station) => station.id === STATION_ID);
const authoredPortal = map.portals.find((portal) => portal.id === 'cliffs_stormbreak_aerie');
const perch = map.platforms.find((platform) => platform.id === 'stormbreakCliffs_aerie_perch');
const highApproach = map.platforms.find((platform) => platform.id === 'stormbreak_cliffs_solid_lane_09');
const perchStair = map.climbables.find((climbable) => climbable.id === 'stormbreakCliffs_storm_stair_11');
check(objectiveSection &&
  objectiveSection.stationId === STATION_ID &&
  objectiveSection.spawnFree === true &&
  objectiveSpawnSection &&
  sameMembers(objectiveSpawnSection.platformIds, [
    'stormbreakCliffs_aerie_perch',
    'stormbreak_cliffs_connector_07'
  ]),
'the objective metadata should bind the Lightning Rod to the existing spawn-free perch approach');
check(authoredStation &&
  authoredStation.name === 'Lightning Rod' &&
  authoredStation.x === 4760 &&
  authoredStation.platformIndex === 23 &&
  authoredStation.serviceRole === 'map_objective' &&
  authoredStation.serviceSummary.includes('Spawn-free regroup'),
'the perch should publish one explicit Lightning Rod interaction station');
check(perch &&
  perch.x === 4300 &&
  perch.y === 300 &&
  perch.w === 620 &&
  perch.spawnDisabled === true &&
  authoredPortal &&
  authoredPortal.x === 4550 &&
  authoredPortal.platformIndex === 23,
'the calm perch and Aerie gate should retain their authored geometry and placement');
check(highApproach &&
  perchStair &&
  perchStair.x >= Math.max(perch.x, highApproach.x) &&
  perchStair.x + perchStair.w <= Math.min(perch.x + perch.w, highApproach.x + highApproach.w) &&
  perchStair.y === perch.y &&
  perchStair.h === highApproach.y - perch.y,
'the existing storm stair should still connect the high approach directly to the Aerie perch');
const objectivePlatformIds = new Set(objectiveSpawnSection.platformIds);
check(map.spawnGroups.every((group) =>
  group.sectionId !== OBJECTIVE_SECTION_ID &&
  group.platformIds.every((platformId) => !objectivePlatformIds.has(platformId))
) &&
  map.spawnPoints.every((point) =>
    point.sectionId !== OBJECTIVE_SECTION_ID &&
    !objectivePlatformIds.has(point.platformId)
  ),
'the Lightning Rod objective territory should contain no encounter group or spawn anchor');
const highGroup = groupsBySectionId.get('stormbreakCliffs_high_harrier_airspace');
check(highGroup.spawnBounds.maxX <= 4190 &&
  authoredPortal.x - highGroup.spawnBounds.maxX >= 360 &&
  authoredStation.x - highGroup.spawnBounds.maxX >= 570,
'high-lane combat should stop before the portal and leave a readable walk to the rod');

const mechanicDefinition = data.MAP_MECHANIC_DEFINITIONS[MAP_ID];
check(mechanicDefinition &&
  mechanicDefinition.completionMode === 'objective-interaction' &&
  mechanicDefinition.objectiveStationId === STATION_ID &&
  mechanicDefinition.objectiveStationLabel === 'Lightning Rod' &&
  sameMembers(mechanicDefinition.activeSectionIds, ACTIVE_SECTION_IDS) &&
  mechanicDefinition.objectiveSectionId === OBJECTIVE_SECTION_ID &&
  mechanicDefinition.requiredUniqueSections === 3,
'the party objective should require all three combat jobs, then a deliberate Lightning Rod interaction');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: data.MAPS });
const runtimePortal = runtime.portals.find((portal) => portal.id === 'cliffs_stormbreak_aerie');
const runtimeStation = runtime.stations.find((station) => station.id === STATION_ID);
const reachablePlatformIndices = mapRuntime.getReachablePlatformIndices(runtime.platformGraph, 0);
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.reachableTierCount >= 4,
'the live Stormbreak route should remain connected, loopable, fully spawn-covered, and vertically varied');
check(runtimePortal &&
  runtimeStation &&
  runtimePortal.platformIndex === 23 &&
  runtimeStation.platformIndex === 23 &&
  runtimeStation.platformId === perch.id &&
  reachablePlatformIndices.has(runtimePortal.platformIndex),
'the Aerie gate and Lightning Rod should both remain reachable on the existing perch');
check(runtimeStation.x - runtimePortal.x >= 180 &&
  runtimeStation.x + runtimeStation.w <= perch.x + perch.w - 60,
'the station should have clear prompt space without overlapping the portal or perch edge');
check(runtime.spawnGroups.every((group) =>
  group.spawnPointIds.length > 0 &&
  group.spawnPointIds.every((spawnPointId) => {
    const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
    return point &&
      point.sectionId === group.sectionId &&
      group.platformIds.includes(point.platformId) &&
      point.x >= group.spawnBounds.minX &&
      point.x <= group.spawnBounds.maxX;
  })
), 'every Stormbreak encounter should use anchors inside its own bounded territory');

const validation = validateMap(map);
check(validation.issues.length === 0 && validation.warnings.length === 0,
  'Stormbreak should satisfy the shared geometry and authored-bound validator cleanly');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const routeEngine = createProjectStarfallEngine(null, data);
  check(routeEngine.chooseClass('fighter'),
    'the Stormbreak route fixture should choose Fighter');
  routeEngine.state.player.level = 100;
  check(routeEngine.changeMap(MAP_ID),
    'the route fixture should enter Stormbreak Cliffs');

  const liveEnemies = routeEngine.enemies.filter((enemy) => enemy && enemy.hp > 0);
  const liveCounts = liveEnemies.reduce((counts, enemy) => {
    counts[enemy.spawnGroupId] = (counts[enemy.spawnGroupId] || 0) + 1;
    return counts;
  }, {});
  check(liveEnemies.length === 36 &&
    ACTIVE_SECTION_IDS.every((sectionId) => liveCounts[sectionId] === 12),
  'the live opening wave should contain 12 enemies in each of the three combat jobs');
  routeEngine.getSpawnGroupPartySize = () => 4;
  check(routeEngine.getRuntimeSpawnGroups().every((group) =>
    routeEngine.getSpawnGroupPopulationTarget(group) === 15
  ) &&
    routeEngine.getWaveMax(map) === 45,
  'a four-player party should raise each combat job from 12 to 15 enemies instead of leaving party scaling inert');
  delete routeEngine.getSpawnGroupPartySize;
  check(liveEnemies.every((enemy) => {
    const group = groupsBySectionId.get(enemy.spawnSectionId);
    return group &&
      ACTIVE_SECTION_IDS.includes(enemy.spawnGroupId) &&
      group.platformIds.includes(enemy.spawnPlatformId) &&
      enemy.spawnPlatformId !== perch.id;
  }), 'live enemies should stay in their low, mid, or high territory and never occupy the perch');

  liveEnemies.slice(0, 35).forEach((enemy) => routeEngine.defeatEnemy(enemy));
  let routeStatus = routeEngine.getRouteFieldStatus(route, routeGoal);
  check(routeStatus.value === 35 &&
    routeStatus.goal === 36 &&
    !routeStatus.complete,
  'the final opening-wave enemy should remain visibly meaningful at 35/36');
  routeEngine.defeatEnemy(liveEnemies[35]);
  routeStatus = routeEngine.getRouteFieldStatus(route, routeGoal);
  check(routeStatus.value === 36 && routeStatus.complete,
    'clearing the real opening population should complete the Stormbreak route immediately');
  check(routeEngine.changeMap('stormbreakHaven'),
    'the route fixture should return to Stormbreak Haven');
  const observatoryPortal = routeEngine.runtime.portals.find((portal) =>
    portal.id === 'stormbreak_haven_observatory'
  );
  check(observatoryPortal &&
    routeEngine.getPortalBlockReason(observatoryPortal) === '',
  'the 36th opening-wave defeat should unlock Astral Observatory without a respawn wait');

  const mechanicEngine = createProjectStarfallEngine(null, data);
  check(mechanicEngine.chooseClass('fighter'),
    'the Lightning Rod fixture should choose Fighter');
  mechanicEngine.state.player.level = 100;
  check(mechanicEngine.changeMap(MAP_ID),
    'the Lightning Rod fixture should enter Stormbreak Cliffs');
  let mechanic = mechanicEngine.getMapMechanicSnapshot();
  check(mechanic.completionMode === 'objective-interaction' &&
    mechanic.objectiveTarget &&
    mechanic.objectiveTarget.type === 'station' &&
    mechanic.objectiveTarget.id === STATION_ID &&
    !mechanic.objectiveReady &&
    mechanic.objectiveCount === 0,
  'the mechanic HUD contract should identify an initially uncharged Lightning Rod target');

  mechanicEngine.recordMapMechanicDefeat(makeMechanicEnemy(ACTIVE_SECTION_IDS[2], 'flyer'));
  mechanicEngine.recordMapMechanicDefeat(makeMechanicEnemy(ACTIVE_SECTION_IDS[1]));
  mechanicEngine.recordMapMechanicDefeat(makeMechanicEnemy(ACTIVE_SECTION_IDS[0]));
  mechanic = mechanicEngine.getMapMechanicSnapshot();
  check(mechanic.progress === 5 &&
    mechanic.currentUniqueSections === 3 &&
    !mechanic.objectiveReady &&
    mechanic.objectiveCount === 0,
  'one pass through all three jobs should persist progress without completing the rod early');

  const partialHits = getSectionHits(mechanic);
  const partialSave = mechanicEngine.serialize();
  const partialRestore = createProjectStarfallEngine(null, data);
  check(partialRestore.restore(partialSave) &&
    partialRestore.state.mapId === MAP_ID,
  'partial Lightning Rod progress should restore on Stormbreak Cliffs');
  mechanic = partialRestore.getMapMechanicSnapshot();
  check(mechanic.progress === 5 &&
    mechanic.currentUniqueSections === 3 &&
    JSON.stringify(getSectionHits(mechanic)) === JSON.stringify(partialHits) &&
    !mechanic.objectiveReady,
  'save and reload should preserve section hits, unique jobs, and partial charge');

  partialRestore.recordMapMechanicDefeat(makeMechanicEnemy(ACTIVE_SECTION_IDS[2], 'flyer'));
  partialRestore.recordMapMechanicDefeat(makeMechanicEnemy(ACTIVE_SECTION_IDS[1]));
  mechanic = partialRestore.getMapMechanicSnapshot();
  check(mechanic.progress === 9 &&
    mechanic.progressPercent === 1 &&
    mechanic.objectiveReady &&
    mechanic.objectiveCount === 0 &&
    mechanic.completedCycles === 0,
  'meeting the combat goal should ready the rod without granting its reward remotely');
  const readyOverlay = partialRestore.getOverlaySnapshot({ openPanels: [] });
  const readyTracker = hud.getMapMechanicTrackerEntry(readyOverlay.mapModifiers.mapMechanic);
  const readyPresentation = hud.getStormbreakLightningRodPresentation(readyOverlay);
  check(readyTracker &&
    readyTracker.guideType === 'map' &&
    readyTracker.guideId === MAP_ID &&
    readyTracker.phase === 'ready' &&
    readyTracker.objectives[0].label === 'Tune the Lightning Rod' &&
    readyTracker.objectives[0].status === 'Ready',
  'the ordinary HUD snapshot should turn charged combat progress into a supported map-tracker objective');
  check(readyPresentation &&
    readyPresentation.ready &&
    readyPresentation.target &&
    readyPresentation.target.id === STATION_ID &&
    readyPresentation.target.platformId === perch.id,
  'the minimap presentation should resolve the charged rod from live station geometry');

  const readySave = partialRestore.serialize();
  const readyRestore = createProjectStarfallEngine(null, data);
  check(readyRestore.restore(readySave),
    'a charged Lightning Rod should survive save and reload');
  mechanic = readyRestore.getMapMechanicSnapshot();
  check(mechanic.objectiveReady &&
    mechanic.objectiveStationId === STATION_ID &&
    mechanic.objectiveStationLabel === 'Lightning Rod',
  'restored readiness should still point to the physical Lightning Rod');

  const liveStation = readyRestore.runtime.stations.find((station) => station.id === STATION_ID);
  readyRestore.state.player.x = liveStation.x;
  readyRestore.state.player.y = liveStation.y;
  readyRestore.updateActiveStation();
  check(readyRestore.state.player.activeStation === STATION_ID &&
    !readyRestore.state.player.activePortalId,
  'standing at the rod should select its prompt without colliding with the Aerie portal');
  const readyPrompt = hud.getStationPromptContext(
    readyRestore.getOverlaySnapshot({ openPanels: [] }),
    { keyLabels: { interact: 'F' } }
  );
  check(readyPrompt &&
    readyPrompt.title === 'Lightning Rod' &&
    readyPrompt.hint === 'F Tune' &&
    readyPrompt.kindLabel === 'Storm objective - Ready',
  'the in-world prompt should clearly name the ready rod and its single physical action');
  const currencyBeforeInteraction = readyRestore.state.player.currency;
  check(readyRestore.interact({ silent: true }),
    'interacting with a charged Lightning Rod should complete the objective');
  mechanic = readyRestore.getMapMechanicSnapshot();
  check(!mechanic.objectiveReady &&
    mechanic.progress === 0 &&
    mechanic.objectiveCount === 1 &&
    mechanic.completedCycles === 1 &&
    readyRestore.state.player.currency > currencyBeforeInteraction,
  'the physical rod interaction should reset charge and grant exactly one completed cycle reward');
  const tunedTracker = hud.getMapMechanicTrackerEntry(mechanic, {
    nowSeconds: mechanic.lastCompletedAt + 1
  });
  const nextCycleTracker = hud.getMapMechanicTrackerEntry(mechanic, {
    nowSeconds: mechanic.lastCompletedAt + 3
  });
  check(tunedTracker &&
    tunedTracker.phase === 'tuned' &&
    tunedTracker.objectives[0].complete &&
    nextCycleTracker &&
    nextCycleTracker.phase === 'lanes' &&
    !nextCycleTracker.objectives[0].complete,
  'the tracker should briefly confirm tuning, then return to the repeatable three-lane cycle');
  const currencyAfterInteraction = readyRestore.state.player.currency;
  check(!readyRestore.interact({ silent: true }) &&
    readyRestore.state.player.currency === currencyAfterInteraction &&
    readyRestore.getMapMechanicSnapshot().objectiveCount === 1,
  'an uncharged second interaction should not duplicate the Lightning Rod reward');

  const completedSave = readyRestore.serialize();
  const completedRestore = createProjectStarfallEngine(null, data);
  check(completedRestore.restore(completedSave) &&
    completedRestore.getMapMechanicSnapshot().objectiveCount === 1 &&
    completedRestore.getMapMechanicSnapshot().completedCycles === 1 &&
    !completedRestore.getMapMechanicSnapshot().objectiveReady,
  'completed Lightning Rod state should remain claimed and uncharged after reload');
} finally {
  Math.random = originalRandom;
}

console.log(`Project Starfall Stormbreak route checks passed: ${checks}`);
