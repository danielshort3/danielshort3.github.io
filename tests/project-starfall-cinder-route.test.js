'use strict';

const assert = require('assert');

const data = require('../js/games/project-starfall/project-starfall-data.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');
const { createMapBalanceReport } = require('./project-starfall-balance-harness.js');

const MAP_ID = 'cinderHollow';
const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  cinderHollow_ash_floor: Object.freeze([
    'cinderHollow_ash_floor_low',
    'cinderHollow_ash_overlook'
  ]),
  cinderHollow_vent_shortcut: Object.freeze([
    'cinderHollow_vent_floor',
    'cinderHollow_vent_shelf'
  ]),
  cinderHollow_wisp_turn: Object.freeze([
    'cinderHollow_wisp_recovery',
    'cinderHollow_wisp_turn_mid',
    'cinderHollow_wisp_turn_high'
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

function sectionToken(label) {
  return String(label || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

const map = data.MAPS.find((entry) => entry.id === MAP_ID);
check(!!map, 'Cinder Hollow should remain published');
check(map.asset === 'img/project-starfall/maps/cinder-hollow.webp' &&
  map.environment.terrain === 'cinder-hollow' &&
  map.environment.props === 'cinder-hollow' &&
  map.palette.join('|') === '#28272d|#f06b37|#9b4835',
'the route pass should preserve Cinder Hollow\'s playful painting, terrain, props, and palette');
check(map.layoutStyle === 'lavaShaft' &&
  map.geometryGenerator === 'priorityFieldV2' &&
  map.compactWorldWidth === 5200 &&
  map.platforms[0].w === 5200,
'Cinder Hollow should keep its lava-shaft identity inside the authored 5200px route');

const sections = map.fieldComposition.routeSections;
check(sections.map((section) => section.label).join('|') ===
  'Ash Floor|Vent Shortcut|Wisp Turn|Emberjaw Approach',
'Cinder Hollow should progress through four readable route beats');
check(sections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === 5200,
'the four route beats should cover Cinder Hollow without metadata gaps');

const sectionsById = new Map(map.spawnSections.map((section) => [section.id, section]));
const groupsBySectionId = new Map(map.spawnGroups.map((group) => [group.sectionId, group]));
const claimedPlatformIds = new Set();
Object.entries(EXPECTED_SECTION_PLATFORMS).forEach(([sectionId, expectedPlatformIds]) => {
  const section = sectionsById.get(sectionId);
  const group = groupsBySectionId.get(sectionId);
  check(!!section && !!group, `${sectionId} should publish a matching section and spawn group`);
  check(sameMembers(section.platformIds, expectedPlatformIds),
    `${sectionId} should own its authored combat platforms`);
  check(sameMembers(group.platformIds, expectedPlatformIds),
    `${sectionId} spawn group should stay inside its authored platform territory`);
  check(group.platformIds.every((platformId) => {
    if (claimedPlatformIds.has(platformId)) return false;
    claimedPlatformIds.add(platformId);
    return true;
  }), `${sectionId} should not share combat platforms with another section`);
  check(group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${sectionId} spawn bounds should stay inside the named route beat`);
});
check(map.spawnGroups.length === 3 &&
  map.spawnGroups.reduce((total, group) => total + group.population, 0) === 24 &&
  map.spawnGroups.every((group) => group.maxPopulation === group.population),
'three bounded eight-enemy encounters should fulfill one 24-enemy Cinder circuit');

const ashGroup = groupsBySectionId.get('cinderHollow_ash_floor');
const ventGroup = groupsBySectionId.get('cinderHollow_vent_shortcut');
const wispGroup = groupsBySectionId.get('cinderHollow_wisp_turn');
const ashEnemies = ashGroup.enemyWeights.map((entry) => entry.enemyId);
const ventEnemies = ventGroup.enemyWeights.map((entry) => entry.enemyId);
const wispEnemies = wispGroup.enemyWeights.map((entry) => entry.enemyId);
check(!ashEnemies.includes('emberWisp') &&
  !ventEnemies.includes('emberWisp') &&
  wispEnemies.includes('emberWisp') &&
  !wispEnemies.includes('ashCrawler'),
'enemy mechanics should progress from grounded packs to vent spitters and then a focused wisp chamber');
check(new Set(map.spawnGroups.flatMap((group) =>
  group.enemyWeights.map((entry) => entry.enemyId)
)).has('lavaTick') &&
  new Set(map.spawnGroups.flatMap((group) =>
    group.enemyWeights.map((entry) => entry.enemyId)
  )).has('cinderSpitter'),
'the authored groups should retain both Cinder Samples quest targets');

const bypass = map.platforms.find((platform) => platform.id === 'cinderHollow_vent_bypass');
check(bypass && bypass.spawnDisabled && bypass.climbableDisabled &&
  !claimedPlatformIds.has(bypass.id),
'the upper vent shelf should remain an optional, spawn-free shortcut');
check(map.climbables.length === 2,
  'ramps should carry the normal route while two intentional chains provide alternate vertical access');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: data.MAPS });
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.reachableTierCount >= 3,
'the live route should be connected, loopable, fully spawn-covered, and vertically varied');
check(runtime.spawnGroups.every((group) =>
  group.spawnPointIds.length > 0 &&
  group.spawnPointIds.every((spawnPointId) => {
    const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
    return point &&
      group.platformIds.includes(point.platformId) &&
      point.x >= group.spawnBounds.minX &&
      point.x <= group.spawnBounds.maxX;
  })
), 'every encounter should use spawn anchors inside its own bounded territory');

const pathfinder = map.questNpcs.find((npc) => npc.id === 'cinder_pathfinder');
const runtimePathfinder = runtime.questNpcs.find((npc) => npc.id === 'cinder_pathfinder');
const emberjawPortal = runtime.portals.find((portal) => portal.id === 'cinder_emberjaw');
check(pathfinder && runtimePathfinder && pathfinder.x === 4740 && runtimePathfinder.x === pathfinder.x,
  'the Pathfinder should be authored in bounds without silent runtime clamping');
check(emberjawPortal &&
  wispGroup.spawnBounds.maxX + 480 <= pathfinder.x &&
  pathfinder.x + 300 <= emberjawPortal.x,
'the last encounter, Pathfinder, and Emberjaw portal should have distinct interaction space');

const validation = validateMap(map);
check(validation.issues.length === 0 && validation.warnings.length === 0,
  'Cinder Hollow should satisfy the shared geometry and authored-bound validator cleanly');
const staleNpcValidation = validateMap(Object.assign({}, map, {
  questNpcs: [Object.assign({}, pathfinder, { x: 7460 })]
}));
check(staleNpcValidation.issues.some((issue) =>
  issue.includes('quest NPC cinder_pathfinder is authored outside platform')
), 'the shared validator should reject stale off-map quest NPC coordinates');

const balanceReport = createMapBalanceReport(data, createProjectStarfallEngine, {
  level: 50,
  rank: 10
});
const tuning = balanceReport.mapTuning.maps.find((entry) => entry.mapId === MAP_ID);
check(tuning &&
  tuning.metrics.travelSharePercent <= 30 &&
  tuning.metrics.classPerformanceSpreadPercent <= 30 &&
  tuning.warningIds.every((warningId) => warningId === 'classSpreadHigh'),
'the tuned route should remove excessive travel and keep the full-class spread within the intended map target');
check(tuning.metrics.abandonmentRiskIndex <= 20 &&
  tuning.metrics.repeatVisitationIndex >= 60,
'the tuned route should model lower abandonment friction without weakening repeat value');

const engine = createProjectStarfallEngine(null, data);
check(engine.chooseClass('fighter'), 'the Cinder quest fixture should choose Fighter');
engine.state.player.level = 25;
engine.state.progress.claimedQuestIds.push('trial_ready');
check(engine.changeMap('cinderRefuge'), 'the quest fixture should enter Cinder Refuge');
const envoy = engine.getQuestNpcSnapshot('cinderRefuge').npcs.find((npc) => npc.id === 'cinder_envoy');
engine.state.player.x = envoy.x - 4;
engine.state.player.y = envoy.y - engine.state.player.h + envoy.h;
engine.updateActiveStation();
check(engine.acceptQuestFromNpc('cinder_envoy', 'cinder_dispatch'),
  'Cinder Dispatch should be accepted from the Refuge envoy');
check(engine.changeMap(MAP_ID), 'the quest fixture should enter Cinder Hollow');
const livePathfinder = engine.getQuestNpcSnapshot(MAP_ID).npcs.find((npc) => npc.id === 'cinder_pathfinder');
engine.state.player.x = livePathfinder.x - 4;
engine.state.player.y = livePathfinder.y - engine.state.player.h + livePathfinder.h;
engine.updateActiveStation();
check(engine.state.player.activeQuestNpcId === 'cinder_pathfinder' &&
  engine.completeQuestTalkObjective('cinder_pathfinder', 'cinder_dispatch') &&
  engine.state.progress.completedQuestIds.includes('cinder_dispatch'),
'the in-bounds Pathfinder interaction should complete the mandatory dispatch handoff');

engine.state.player.x = emberjawPortal.x - 4;
engine.state.player.y = runtime.platforms[0].y - engine.state.player.h;
engine.updateActiveStation();
check(engine.state.player.activePortalId === 'cinder_emberjaw' &&
  engine.state.player.activeQuestNpcId !== 'cinder_pathfinder',
'the Emberjaw portal should remain separately reachable after the Pathfinder handoff');
check(engine.enemies.length === 24 &&
  engine.enemies.every((enemy) => {
    const group = runtime.spawnGroups.find((entry) => entry.id === enemy.spawnGroupId);
    return group &&
      enemy.spawnX >= group.spawnBounds.minX &&
      enemy.spawnX <= group.spawnBounds.maxX &&
      enemy.spawnX <= 4260;
  }),
'the live opening wave should stay inside the three combat territories and outside the final approach');

sections.forEach((section) => {
  const expectedId = `${MAP_ID}_${sectionToken(section.label)}`;
  check(sectionsById.has(expectedId), `${section.label} should publish the stable section id ${expectedId}`);
});

console.log(`Project Starfall Cinder route checks passed: ${checks}`);
