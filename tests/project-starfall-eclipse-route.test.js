'use strict';

const assert = require('assert');

const data = require('../js/games/project-starfall/data/index.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

const MAP_ID = 'eclipseFrontier';
const ORDERED_SECTION_IDS = Object.freeze([
  'eclipseFrontier_solar_outpost',
  'eclipseFrontier_lunar_outpost',
  'eclipseFrontier_eclipse_gate',
  'eclipseFrontier_elite_pocket'
]);
const EXPECTED_SECTIONS = Object.freeze([
  Object.freeze({
    id: ORDERED_SECTION_IDS[0],
    label: 'Solar Outpost',
    x: 0,
    w: 1600,
    platformIds: Object.freeze([
      'eclipse_frontier_solid_lane_01',
      'eclipse_frontier_solid_lane_02',
      'eclipse_frontier_solid_lane_03'
    ])
  }),
  Object.freeze({
    id: ORDERED_SECTION_IDS[1],
    label: 'Lunar Outpost',
    x: 1600,
    w: 1200,
    platformIds: Object.freeze([
      'eclipse_frontier_solid_lane_04',
      'eclipse_frontier_solid_lane_05',
      'eclipse_frontier_solid_lane_06'
    ])
  }),
  Object.freeze({
    id: ORDERED_SECTION_IDS[2],
    label: 'Eclipse Gate',
    x: 2800,
    w: 1400,
    platformIds: Object.freeze([
      'eclipse_frontier_solid_lane_07',
      'eclipse_frontier_solid_lane_08',
      'eclipse_frontier_solid_lane_09'
    ])
  }),
  Object.freeze({
    id: ORDERED_SECTION_IDS[3],
    label: 'Elite Pocket',
    x: 4200,
    w: 1400,
    platformIds: Object.freeze([
      'eclipse_frontier_solid_lane_10',
      'eclipse_frontier_solid_lane_11',
      'eclipse_frontier_solid_lane_12'
    ])
  })
]);
const EXPECTED_GROUPS = Object.freeze({
  eclipseFrontier_solar_outpost: Object.freeze({
    label: 'Solar Sentinels',
    population: 8,
    maxPopulation: 10,
    respawnSeconds: 6,
    leash: 460,
    spawnBounds: Object.freeze({ minX: 320, maxX: 1500 }),
    enemyWeights: Object.freeze([
      Object.freeze({ enemyId: 'lumenSentinel', weight: 6 }),
      Object.freeze({ enemyId: 'indexScribe', weight: 2 }),
      Object.freeze({ enemyId: 'eclipseDuelist', weight: 2 })
    ])
  }),
  eclipseFrontier_lunar_outpost: Object.freeze({
    label: 'Lunar Motes',
    population: 7,
    maxPopulation: 9,
    respawnSeconds: 6,
    leash: 460,
    spawnBounds: Object.freeze({ minX: 1720, maxX: 2740 }),
    enemyWeights: Object.freeze([
      Object.freeze({ enemyId: 'voidMote', weight: 1 })
    ])
  }),
  eclipseFrontier_eclipse_gate: Object.freeze({
    label: 'Gate Duelists',
    population: 9,
    maxPopulation: 11,
    respawnSeconds: 7,
    leash: 500,
    spawnBounds: Object.freeze({ minX: 2920, maxX: 4060 }),
    enemyWeights: Object.freeze([
      Object.freeze({ enemyId: 'eclipseDuelist', weight: 1 })
    ])
  }),
  eclipseFrontier_elite_pocket: Object.freeze({
    label: 'Totality Elite Pocket',
    population: 10,
    maxPopulation: 12,
    respawnSeconds: 9,
    leash: 540,
    spawnBounds: Object.freeze({ minX: 4290, maxX: 5290 }),
    enemyWeights: Object.freeze([
      Object.freeze({ enemyId: 'eclipseDuelist', weight: 5 }),
      Object.freeze({ enemyId: 'crackedMimic', weight: 2 }),
      Object.freeze({ enemyId: 'voidMote', weight: 2 }),
      Object.freeze({ enemyId: 'lumenSentinel', weight: 1 })
    ])
  })
});

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function sameMembers(actual, expected) {
  const actualValues = Array.from(new Set((actual || []).map(String).filter(Boolean))).sort();
  const expectedValues = Array.from(new Set((expected || []).map(String).filter(Boolean))).sort();
  return actualValues.length === expectedValues.length &&
    actualValues.every((value, index) => value === expectedValues[index]);
}

function makeMechanicEnemy(sectionId, patch) {
  return Object.assign({
    spawnSectionId: sectionId,
    sectionId,
    data: { behavior: 'ground' }
  }, patch || {});
}

function createEclipseEngine() {
  const engine = createProjectStarfallEngine(null, data);
  assert(engine.chooseClass('fighter'), 'the Eclipse fixture should choose Fighter');
  engine.state.player.level = 100;
  assert(engine.changeMap(MAP_ID, { silent: true }), 'the Eclipse fixture should enter Eclipse Frontier');
  return engine;
}

function recordSectionDefeats(engine, sectionId, count) {
  return Array.from({ length: count }, () =>
    engine.recordMapMechanicDefeat(makeMechanicEnemy(sectionId)));
}

const map = data.MAPS.find((entry) => entry.id === MAP_ID);
const route = data.WORLD_ROUTES.find((entry) => entry.id === 'ascension');
const routeGoal = route && route.fieldGoals.find((entry) => entry.mapId === MAP_ID);
const patrolQuest = data.QUESTS.find((entry) => entry.id === 'eclipse_frontier_patrol');

check(!!map && !!routeGoal && !!patrolQuest,
  'Eclipse Frontier, its Ascension goal, and its patrol quest should remain published');
check(map.asset === 'img/project-starfall/maps/eclipse-frontier.webp' &&
  map.environment.terrain === 'eclipse-frontier' &&
  map.environment.props === 'eclipse-frontier' &&
  map.palette.join('|') === '#1f2330|#ffbe55|#7bdff2',
'the cleanup should preserve Eclipse Frontier\'s playful painting, terrain, props, and solar-lunar palette');
check(map.layoutStyle === 'astralStack' &&
  map.geometryGenerator === 'fieldLayout' &&
  map.compactWorldWidth === 5600 &&
  map.platforms[0].w === 5600 &&
  map.worldHeight === 1260 &&
  map.authoredGroundY === 1120,
'Eclipse Frontier should use the compact 5600-pixel astral-stack field without changing its vertical scale');
check(map.platforms.length === 36 &&
  map.rampConnections.length === 7 &&
  map.climbables.length === 16 &&
  map.spawnPoints.length === 12,
'the four-stack field should retain its complete authored platform, ramp, stair, and spawn-anchor geometry');

const routeSections = map.fieldComposition.routeSections;
check(routeSections.length === EXPECTED_SECTIONS.length &&
  routeSections.every((section, index) => {
    const expected = EXPECTED_SECTIONS[index];
    return section.label === expected.label &&
      section.x === expected.x &&
      section.w === expected.w &&
      sameMembers(section.platformIds, expected.platformIds);
  }),
'the map should publish the finalized Solar, Lunar, Gate, and Elite section spans and lane assignments');
check(routeSections.reduce((rightEdge, section) => {
  assert.strictEqual(section.x, rightEdge,
    `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === map.platforms[0].w,
'the four sections should cover the compact field contiguously without gaps or overlaps');

const platformsById = new Map(map.platforms.map((platform) => [platform.id, platform]));
EXPECTED_SECTIONS.forEach((expected) => {
  expected.platformIds.forEach((platformId) => {
    const platform = platformsById.get(platformId);
    check(platform &&
      platform.x >= expected.x &&
      platform.x + platform.w <= expected.x + expected.w,
    `${platformId} should remain physically inside ${expected.label}`);
  });
});
check(map.designIntent.visualIdentityTag === 'playful solar and lunar frontier outposts' &&
  map.designIntent.implementationStatus === 'geometry-mechanic-v1' &&
  map.designIntent.routeSummary.includes('Solar and Lunar outposts'),
'the design metadata should preserve the playful frontier identity and describe the implemented route');

const spawnSectionsById = new Map(map.spawnSections.map((section) => [section.id, section]));
const spawnGroupsById = new Map(map.spawnGroups.map((group) => [group.id, group]));
const claimedPlatforms = new Set();
check(map.spawnSections.length === 4 &&
  map.spawnGroups.length === 4 &&
  sameMembers(map.spawnSections.map((section) => section.id), ORDERED_SECTION_IDS) &&
  sameMembers(map.spawnGroups.map((group) => group.sectionId), ORDERED_SECTION_IDS),
'Eclipse Frontier should expose exactly one spawn section and encounter group per patrol stop');
EXPECTED_SECTIONS.forEach((expected) => {
  const spawnSection = spawnSectionsById.get(expected.id);
  const group = spawnGroupsById.get(expected.id);
  const profile = EXPECTED_GROUPS[expected.id];
  check(spawnSection &&
    spawnSection.label === expected.label &&
    spawnSection.x === expected.x &&
    spawnSection.w === expected.w &&
    sameMembers(spawnSection.platformIds, expected.platformIds),
  `${expected.label} spawn metadata should align with its authored section and lanes`);
  check(group &&
    group.sectionId === expected.id &&
    sameMembers(group.platformIds, expected.platformIds) &&
    group.platformIds.every((platformId) => {
      if (claimedPlatforms.has(platformId)) return false;
      claimedPlatforms.add(platformId);
      return true;
    }),
  `${expected.label} should exclusively own its three encounter platforms`);
  check(group.label === profile.label &&
    group.population === profile.population &&
    group.maxPopulation === profile.maxPopulation &&
    group.respawnSeconds === profile.respawnSeconds &&
    group.leash === profile.leash &&
    group.partyScaling === 'section-count' &&
    group.partyBonusPerMember === 1 &&
    JSON.stringify(group.spawnBounds) === JSON.stringify(profile.spawnBounds) &&
    JSON.stringify(group.enemyWeights) === JSON.stringify(profile.enemyWeights),
  `${expected.label} should retain its distinct bounded enemy and cadence profile`);
  check(group.spawnBounds.minX >= expected.x &&
    group.spawnBounds.maxX <= expected.x + expected.w,
  `${expected.label} spawn bounds should stay inside its named section`);
});
check(new Set(map.spawnGroups.map((group) => JSON.stringify({
  weights: group.enemyWeights,
  population: group.population,
  respawnSeconds: group.respawnSeconds,
  leash: group.leash
}))).size === 4,
'all four outposts should have genuinely distinct encounter profiles instead of cloned spawns');
check(map.spawnGroups.reduce((total, group) => total + group.population, 0) === 34 &&
  map.waveMax === 34 &&
  routeGoal.count === 40,
'the opening population should remain capped at 34 while the long-form Ascension clearance remains 40');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: data.MAPS });
const reachablePlatformIndices = mapRuntime.getReachablePlatformIndices(runtime.platformGraph, 0);
check(runtime.worldWidth === 5760 &&
  runtime.worldHeight === 1260 &&
  runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.reachableTierCount >= runtime.trainingRoute.requiredReachableTierCount,
'the compact live map should remain connected, loopable, fully spawn-covered, and vertically viable');
check(Object.values(runtime.trainingRoute.checks).every(Boolean) &&
  runtime.trainingRoute.minCombatLaneWidth >= 700,
'every runtime route-health guard should pass with broad combat lanes');
check(runtime.spawnGroups.every((group) =>
  group.spawnPointIds.length === 3 &&
  group.spawnPointIds.every((spawnPointId) => {
    const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
    return point &&
      point.sectionId === group.sectionId &&
      group.platformIds.includes(point.platformId) &&
      point.x >= group.spawnBounds.minX &&
      point.x <= group.spawnBounds.maxX;
  })
), 'each live encounter should use exactly three anchors inside its own section bounds');

const archivePortal = runtime.portals.find((portal) => portal.id === 'eclipse_archive');
const riftPortal = runtime.portals.find((portal) => portal.id === 'eclipse_rift');
const thronePortal = runtime.portals.find((portal) => portal.id === 'frontier_eclipse_throne');
check(archivePortal &&
  archivePortal.returnPortal === true &&
  archivePortal.destinationMapId === 'astralArchive' &&
  archivePortal.x === 110 &&
  archivePortal.platformIndex === 0,
'the Archive Return should remain on the safe left edge of the ground lane');
check(riftPortal &&
  riftPortal.destinationMapId === 'endlessRift' &&
  riftPortal.routeId === 'ascension' &&
  riftPortal.requiredMapId === MAP_ID &&
  riftPortal.x === map.platforms[0].w - 120 &&
  riftPortal.platformIndex === 0,
'the Endless Rift advance should sit on the safe right edge of the compact field');
check(thronePortal &&
  thronePortal.destinationMapId === 'eclipseThrone' &&
  thronePortal.bossPortal === true &&
  thronePortal.requiredQuestId === 'eclipse_echo' &&
  thronePortal.x === 4650 &&
  thronePortal.platformIndex === 34,
'the Eclipse Throne gate should remain on its authored upper perch');
check([archivePortal, riftPortal, thronePortal].every((portal) => {
  const platform = runtime.platforms[portal.platformIndex];
  return platform &&
    portal.x >= platform.x &&
    portal.x <= platform.x + platform.w &&
    reachablePlatformIndices.has(portal.platformIndex);
}), 'all three Eclipse portals should sit on valid, ground-reachable platforms');

const validation = validateMap(map);
check(validation.issues.length === 0 && validation.warnings.length === 0,
  'Eclipse Frontier should satisfy the shared geometry validator without issues or warnings');

const mechanicDefinition = data.MAP_MECHANIC_DEFINITIONS[MAP_ID];
check(mechanicDefinition &&
  mechanicDefinition.type === 'patrol-loop' &&
  mechanicDefinition.completionMode === 'automatic' &&
  mechanicDefinition.requiredSectionOrder === true &&
  mechanicDefinition.killsPerSection === 3 &&
  mechanicDefinition.eventKillGoal === 12 &&
  mechanicDefinition.requiredUniqueSections === 4 &&
  mechanicDefinition.repeatWarningThreshold === 13 &&
  mechanicDefinition.activeSectionIds.join('|') === ORDERED_SECTION_IDS.join('|'),
'the patrol should require an automatic ordered four-by-three Solar-to-Elite rotation with one-pack anti-camp grace');
check(mechanicDefinition.reward.currency === 150 &&
  mechanicDefinition.reward.materials.cubeFragment === 2 &&
  mechanicDefinition.objectiveSectionId === ORDERED_SECTION_IDS[3] &&
  mechanicDefinition.regroupSectionId === ORDERED_SECTION_IDS[3],
'the ordered patrol should finish and reward at the Elite Pocket');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const openingEngine = createEclipseEngine();
  const liveEnemies = openingEngine.enemies.filter((enemy) => enemy && enemy.hp > 0);
  const liveGroupCounts = {};
  const liveEnemyCounts = {};
  liveEnemies.forEach((enemy) => {
    liveGroupCounts[enemy.spawnGroupId] = (liveGroupCounts[enemy.spawnGroupId] || 0) + 1;
    liveEnemyCounts[enemy.data.id] = (liveEnemyCounts[enemy.data.id] || 0) + 1;
  });
  check(liveEnemies.length === 34 &&
    liveGroupCounts[ORDERED_SECTION_IDS[0]] === 8 &&
    liveGroupCounts[ORDERED_SECTION_IDS[1]] === 7 &&
    liveGroupCounts[ORDERED_SECTION_IDS[2]] === 9 &&
    liveGroupCounts[ORDERED_SECTION_IDS[3]] === 10,
  'the deterministic opening wave should populate all four patrol sections at their authored caps');
  const defeatObjectives = patrolQuest.objectives.filter((objective) =>
    objective.type === 'defeat' && objective.mapId === MAP_ID);
  check(defeatObjectives.length === 2 &&
    defeatObjectives.every((objective) =>
      Number(liveEnemyCounts[objective.enemyId] || 0) >= objective.count
    ) &&
    liveEnemies.filter((enemy) => enemy.spawnGroupId === ORDERED_SECTION_IDS[1])
      .every((enemy) => enemy.data.id === 'voidMote') &&
    liveEnemies.filter((enemy) => enemy.spawnGroupId === ORDERED_SECTION_IDS[2])
      .every((enemy) => enemy.data.id === 'eclipseDuelist'),
  'the Lunar and Gate rosters should guarantee the 7 Void Motes and 9 Duelists required by the patrol quest');
  check(liveEnemies.every((enemy) => {
    const group = spawnGroupsById.get(enemy.spawnGroupId);
    return group &&
      enemy.spawnSectionId === group.sectionId &&
      group.platformIds.includes(enemy.spawnPlatformId) &&
      enemy.x >= group.spawnBounds.minX &&
      enemy.x <= group.spawnBounds.maxX;
  }), 'every live opening enemy should remain inside its authored section, platform set, and horizontal bounds');

  const closedOverlay = openingEngine.getOverlaySnapshot({ openPanels: [] });
  const ascensionProgress = closedOverlay.routeProgress.routes.find((entry) => entry.id === 'ascension');
  const eclipseProgress = ascensionProgress &&
    ascensionProgress.fields.find((field) => field.mapId === MAP_ID);
  check(closedOverlay.worldMap.nodes.length === 0 &&
    ascensionProgress &&
    eclipseProgress &&
    eclipseProgress.value === 0 &&
    eclipseProgress.goal === 40 &&
    !eclipseProgress.complete,
  'route progress should remain available to the HUD while the world-map overlay is closed');

  const unifiedEntry = hud.getCurrentMapObjectiveTrackerEntry(closedOverlay);
  const trackerEntries = hud.getQuestTrackerEntries(closedOverlay);
  const eclipseTrackerEntries = trackerEntries.filter((entry) =>
    entry && entry.guideId === MAP_ID && (entry.currentMapObjective || entry.mapMechanic));
  check(unifiedEntry &&
    unifiedEntry.currentMapObjective === true &&
    unifiedEntry.guideType === 'map' &&
    unifiedEntry.guideId === MAP_ID &&
    unifiedEntry.phase === 'ordered' &&
    unifiedEntry.objectives.length === 2,
  'the ordinary HUD should expose one unified Eclipse map-objective entry');
  check(unifiedEntry.objectives[0].label === 'Secure Solar Outpost' &&
    unifiedEntry.objectives[0].value === 0 &&
    unifiedEntry.objectives[0].goal === 3 &&
    unifiedEntry.objectives[0].status === 'Patrol 1/4' &&
    unifiedEntry.objectives[1].label.includes('Route clearance') &&
    unifiedEntry.objectives[1].label.includes('Endless Rift') &&
    unifiedEntry.objectives[1].value === 0 &&
    unifiedEntry.objectives[1].goal === 40,
  'the unified entry should show both the next three-kill patrol stop and the persistent route gate');
  check(eclipseTrackerEntries.length === 1 &&
    eclipseTrackerEntries[0].objectives.length === 2,
  'quest tracking should not duplicate the Eclipse patrol and route-clearance objectives');
  check(openingEngine.setQuestGuideTarget(unifiedEntry.guideType, unifiedEntry.guideId, { noEmit: true }),
    'the patrol card should expose a valid map-mechanic guide action');
  const mechanicGuidance = openingEngine.getQuestGuidanceSnapshot();
  check(mechanicGuidance.targetType === 'map' &&
    mechanicGuidance.targetId === MAP_ID &&
    mechanicGuidance.sourceTitle === 'Eclipse Frontier' &&
    !mechanicGuidance.objectiveLabel.includes('Accept') &&
    !mechanicGuidance.sourceTitle.includes('Scout Hunt'),
  'clicking the patrol card should keep map-mechanic guidance instead of redirecting to the local hunt NPC');

  const weeklyAssignmentId = 'field_c:mapMechanic:eclipse_sigil_patrol';
  const weeklyOverlay = Object.assign({}, closedOverlay, {
    season: Object.assign({}, closedOverlay.season, {
      weeklyRoutes: {
        unlocked: true,
        complete: false,
        rewardGranted: false,
        completionCount: 0,
        assignments: [{
          id: weeklyAssignmentId,
          kind: 'mapMechanic',
          targetId: 'eclipse_sigil_patrol',
          mapId: MAP_ID,
          label: 'Complete Eclipse Sigil Patrol',
          guideType: 'map',
          guideId: MAP_ID,
          value: 0,
          goal: 1,
          complete: false
        }]
      }
    })
  });
  const weeklyEntries = hud.getQuestTrackerEntries(weeklyOverlay);
  const weeklyEclipseEntries = weeklyEntries.filter((entry) =>
    entry && entry.guideType === 'map' && entry.guideId === MAP_ID);
  check(weeklyEclipseEntries.length === 1 &&
    weeklyEclipseEntries[0].mapMechanic === true &&
    weeklyEclipseEntries[0].weeklyRoute === true &&
    weeklyEclipseEntries[0].assignmentId === weeklyAssignmentId &&
    weeklyEclipseEntries[0].title === 'Weekly · Eclipse Sigil Patrol',
  'a focused weekly patrol should merge into the live Eclipse objective card without a duplicate mechanic row');

  const provenanceEngine = createEclipseEngine();
  const provenanceBefore = provenanceEngine.getMapMechanicSnapshot();
  ['adminSpawned', 'temporarySpawn', 'adminDefeated'].forEach((flag) => {
    check(!provenanceEngine.recordMapMechanicDefeat(
      makeMechanicEnemy(ORDERED_SECTION_IDS[0], { [flag]: true })
    ), `${flag} enemies should be excluded from Eclipse patrol credit`);
  });
  const provenanceAfter = provenanceEngine.getMapMechanicSnapshot();
  check(provenanceAfter.progress === provenanceBefore.progress &&
    provenanceAfter.currentSectionKillCount === provenanceBefore.currentSectionKillCount &&
    provenanceAfter.sections.every((section) => section.hits === 0),
  'admin and temporary exclusions should leave all ordered patrol state untouched');

  const rejectionEngine = createEclipseEngine();
  const rejectedResults = recordSectionDefeats(rejectionEngine, ORDERED_SECTION_IDS[1], 5);
  const rejected = rejectionEngine.getMapMechanicSnapshot();
  check(rejectedResults.every(Boolean) &&
    rejected.progress === 0 &&
    rejected.currentSectionKillCount === 0 &&
    rejected.orderedSectionIds.length === 0 &&
    rejected.nextSectionId === ORDERED_SECTION_IDS[0] &&
    rejected.completedCycles === 0,
  'out-of-order Lunar camping should be observed but rejected from patrol progress and rewards');

  const mechanicEngine = createEclipseEngine();
  let mechanic = mechanicEngine.getMapMechanicSnapshot();
  check(mechanic.requiredSectionOrder === true &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[0] &&
    mechanic.nextSectionLabel === 'Solar Outpost' &&
    mechanic.currentSectionKillCount === 0 &&
    mechanic.orderedSectionIds.length === 0,
  'a new patrol should begin at an empty Solar Outpost step');

  recordSectionDefeats(mechanicEngine, ORDERED_SECTION_IDS[0], 2);
  mechanic = mechanicEngine.getMapMechanicSnapshot();
  check(mechanic.progress === 2 &&
    mechanic.currentSectionKillCount === 2 &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[0] &&
    mechanic.orderedSectionIds.length === 0,
  'two Solar defeats should remain two of three without advancing the patrol');
  recordSectionDefeats(mechanicEngine, ORDERED_SECTION_IDS[0], 1);
  recordSectionDefeats(mechanicEngine, ORDERED_SECTION_IDS[1], 1);
  mechanic = mechanicEngine.getMapMechanicSnapshot();
  check(mechanic.progress === 4 &&
    mechanic.currentSectionKillCount === 1 &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[1] &&
    mechanic.orderedSectionIds.join('|') === ORDERED_SECTION_IDS[0],
  'three Solar defeats should advance to Lunar, where one defeat should persist as one of three');

  const partialSave = mechanicEngine.serialize();
  const restoredEngine = createProjectStarfallEngine(null, data);
  check(restoredEngine.restore(partialSave) &&
    restoredEngine.state.mapId === MAP_ID,
  'the public restore API should reload a patrol saved mid-Lunar step');
  mechanic = restoredEngine.getMapMechanicSnapshot();
  check(mechanic.progress === 4 &&
    mechanic.currentSectionKillCount === 1 &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[1] &&
    mechanic.orderedSectionIds.join('|') === ORDERED_SECTION_IDS[0],
  'save and restore should preserve ordered section history, next stop, and partial three-kill credit');

  const currencyBefore = restoredEngine.state.player.currency;
  const fragmentsBefore = Number(restoredEngine.state.materials.cubeFragment || 0);
  recordSectionDefeats(restoredEngine, ORDERED_SECTION_IDS[1], 2);
  recordSectionDefeats(restoredEngine, ORDERED_SECTION_IDS[2], 3);
  recordSectionDefeats(restoredEngine, ORDERED_SECTION_IDS[3], 2);
  mechanic = restoredEngine.getMapMechanicSnapshot();
  check(mechanic.progress === 11 &&
    mechanic.currentSectionKillCount === 2 &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[3] &&
    mechanic.orderedSectionIds.join('|') === ORDERED_SECTION_IDS.slice(0, 3).join('|') &&
    mechanic.completedCycles === 0 &&
    restoredEngine.state.player.currency === currencyBefore,
  'the patrol should remain unrewarded at eleven of twelve with two Elite defeats');
  check(restoredEngine.recordMapMechanicDefeat(makeMechanicEnemy(ORDERED_SECTION_IDS[3])),
    'the twelfth in-order defeat should complete the patrol automatically');
  mechanic = restoredEngine.getMapMechanicSnapshot();
  check(mechanic.completedCycles === 1 &&
    mechanic.eventCount === 1 &&
    mechanic.progress === 0 &&
    mechanic.currentSectionKillCount === 0 &&
    mechanic.orderedSectionIds.length === 0 &&
    mechanic.nextSectionId === ORDERED_SECTION_IDS[0] &&
    mechanic.routeComplete === false,
  'completion should count once and reset the four-by-three route back to Solar');
  check(restoredEngine.state.player.currency === currencyBefore + 150 &&
    Number(restoredEngine.state.materials.cubeFragment || 0) === fragmentsBefore + 2,
  'a clean ordered patrol should grant exactly 150 currency and two Cube Fragments');

  const naturalClearEngine = createEclipseEngine();
  const naturalCurrencyBefore = naturalClearEngine.state.player.currency;
  const naturalFragmentsBefore = Number(naturalClearEngine.state.materials.cubeFragment || 0);
  EXPECTED_SECTIONS.forEach((section) => {
    recordSectionDefeats(naturalClearEngine, section.id, EXPECTED_GROUPS[section.id].population);
  });
  const naturalClear = naturalClearEngine.getMapMechanicSnapshot();
  check(naturalClear.completedCycles === 1 &&
    naturalClear.nextSectionId === ORDERED_SECTION_IDS[0] &&
    naturalClear.antiCampStacks === 0 &&
    naturalClear.rewardScale === 1,
  'clearing every enemy in one natural outpost circuit should reset to Solar without an anti-camp penalty');
  check(naturalClearEngine.state.player.currency === naturalCurrencyBefore + 150 &&
    Number(naturalClearEngine.state.materials.cubeFragment || 0) === naturalFragmentsBefore + 2,
  'a natural full-pack circuit should receive the complete patrol reward exactly once');
} finally {
  Math.random = originalRandom;
}

console.log(`Project Starfall Eclipse route checks passed: ${checks}`);
