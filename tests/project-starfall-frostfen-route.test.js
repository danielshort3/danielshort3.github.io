'use strict';

const assert = require('assert');

const data = require('../js/games/project-starfall/project-starfall-data.js');
global.ProjectStarfallData = data;

const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

const MAP_ID = 'frostfenOutskirts';
const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  frostfenOutskirts_marsh_flats: Object.freeze([
    'frostfen_outskirts_solid_lane_01',
    'frostfen_outskirts_solid_lane_02',
    'frostfen_outskirts_solid_lane_03'
  ]),
  frostfenOutskirts_ice_shelves: Object.freeze([
    'frostfen_outskirts_solid_lane_04',
    'frostfen_outskirts_solid_lane_05',
    'frostfen_outskirts_solid_lane_06',
    'frostfen_outskirts_solid_lane_07',
    'frostfen_outskirts_solid_lane_08',
    'frostfen_outskirts_solid_lane_09'
  ]),
  frostfenOutskirts_oracle_grove: Object.freeze([
    'frostfen_outskirts_solid_lane_10',
    'frostfen_outskirts_solid_lane_11',
    'frostfen_outskirts_solid_lane_12'
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

function actorCenterX(actor) {
  return Number(actor.x || 0) + Number(actor.w || 0) / 2;
}

const map = data.MAPS.find((entry) => entry.id === MAP_ID);
const route = data.WORLD_ROUTES.find((entry) => entry.id === 'frostfen');
const routeGoal = route && route.fieldGoals.find((entry) => entry.mapId === MAP_ID);

check(!!map && !!routeGoal, 'Frostfen Outskirts and its route goal should remain published');
check(map.asset === 'img/project-starfall/maps/frostfen-outskirts.webp' &&
  map.environment.terrain === 'frostfen-outskirts' &&
  map.environment.props === 'frostfen-outskirts' &&
  map.palette.join('|') === '#d7f3ff|#5ca8e8|#f7fbff',
'the route pass should preserve Frostfen\'s playful painting, terrain, props, and palette');
check(map.layoutStyle === 'switchbackTerraces' &&
  map.geometryGenerator === 'fieldLayout' &&
  map.movementProfile === 'ice' &&
  map.platforms[0].w === 8400 &&
  map.platforms.length === 31 &&
  map.climbables.length === 12,
'Frostfen should keep its authored switchback proportions, platforms, ropes, and ice movement');

const sections = map.fieldComposition.routeSections;
check(sections.map((section) => `${section.label}:${section.x}:${section.w}`).join('|') ===
  'Marsh Flats:0:2200|Ice Shelves:2200:4000|Oracle Grove:6200:2200',
'the existing route should read as Marsh Flats, Ice Shelves, then Oracle Grove');
check(sections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === map.platforms[0].w,
'the three named route beats should cover Frostfen without metadata gaps');
check(map.fieldComposition.landmarkBands.map((band) => `${band.label}:${band.anchorX}`).join('|') ===
  'Frozen Marsh:1760|Ice Shelves:4020|Glacier Ascent:7920',
'existing Frostfen prop types should mark the marsh, shelves, and final ascent');
check(map.designIntent.routeSummary.includes('Glacier Ascent') &&
  map.designIntent.implementationStatus === 'composition-spawn-v1',
'Frostfen design intent should describe the forward route that is actually implemented');

const sectionsById = new Map(map.spawnSections.map((section) => [section.id, section]));
const groupsBySectionId = new Map(map.spawnGroups.map((group) => [group.sectionId, group]));
const claimedPlatformIds = new Set();
Object.entries(EXPECTED_SECTION_PLATFORMS).forEach(([sectionId, expectedPlatformIds]) => {
  const section = sectionsById.get(sectionId);
  const group = groupsBySectionId.get(sectionId);
  check(!!section && !!group, `${sectionId} should publish a matching section and encounter group`);
  check(sameMembers(section.platformIds, expectedPlatformIds),
    `${sectionId} should own its existing combat platforms`);
  check(sameMembers(group.platformIds, expectedPlatformIds),
    `${sectionId} encounters should stay inside their authored platform territory`);
  check(group.platformIds.every((platformId) => {
    if (claimedPlatformIds.has(platformId)) return false;
    claimedPlatformIds.add(platformId);
    return true;
  }), `${sectionId} should not share combat platforms with another section`);
  check(group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${sectionId} spawn bounds should stay inside its named route beat`);
});
check(map.spawnGroups.map((group) => group.population).join('|') === '10|11|11' &&
  map.spawnGroups.every((group) => group.maxPopulation === group.population) &&
  map.spawnGroups.reduce((total, group) => total + group.population, 0) === 32 &&
  map.waveMax === routeGoal.count &&
  routeGoal.count === 32,
'one 32-enemy circuit should fulfill the Frostfen gate goal without waiting for a respawn');

const marshGroup = groupsBySectionId.get('frostfenOutskirts_marsh_flats');
const iceGroup = groupsBySectionId.get('frostfenOutskirts_ice_shelves');
const oracleGroup = groupsBySectionId.get('frostfenOutskirts_oracle_grove');
const marshEnemies = marshGroup.enemyWeights.map((entry) => entry.enemyId);
const iceEnemies = iceGroup.enemyWeights.map((entry) => entry.enemyId);
const oracleEnemies = oracleGroup.enemyWeights.map((entry) => entry.enemyId);
check(!marshEnemies.includes('snowglareWisp') &&
  !marshEnemies.includes('icebloomOracle') &&
  iceEnemies.includes('snowglareWisp') &&
  !iceEnemies.includes('icebloomOracle') &&
  oracleEnemies.includes('icebloomOracle'),
'enemy mechanics should progress from grounded scouts to sliding wisps and a focused Oracle finish');

const runtime = mapRuntime.createMapRuntime(map, null, { maps: data.MAPS });
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.reachableTierCount >= 3,
'the live route should remain connected, loopable, fully spawn-covered, and vertically varied');
check(runtime.spawnGroups.every((group) =>
  group.spawnPointIds.length > 0 &&
  group.spawnPointIds.every((spawnPointId) => {
    const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
    return point &&
      group.platformIds.includes(point.platformId) &&
      point.x >= group.spawnBounds.minX &&
      point.x <= group.spawnBounds.maxX;
  })
), 'every Frostfen encounter should use spawn anchors inside its own bounded territory');

const authoredPortal = map.portals.find((portal) => portal.id === 'frostfen_glacier');
const runtimePortal = runtime.portals.find((portal) => portal.id === 'frostfen_glacier');
const finalPlatformRight = Math.max(...oracleGroup.platformIds.map((platformId) => {
  const platform = map.platforms.find((entry) => entry.id === platformId);
  return platform.x + platform.w;
}));
check(authoredPortal &&
  runtimePortal &&
  authoredPortal.x === 8280 &&
  runtimePortal.x === authoredPortal.x &&
  authoredPortal.x === map.platforms[0].x + map.platforms[0].w - 120,
'Glacier Ascent should be authored and rendered at the natural far-right route terminus');
check(oracleGroup.spawnBounds.maxX + 800 <= authoredPortal.x &&
  finalPlatformRight + 480 <= authoredPortal.x,
'Oracle Grove combat should end before a calm, readable approach to Glacier Ascent');
check(runtimePortal.roleLabel === 'right glacier ascent' &&
  runtimePortal.portalStyle === 'glacier lift',
'the moved gate should preserve its existing glacier-lift fiction');

const validation = validateMap(map);
check(validation.issues.length === 0 && validation.warnings.length === 0,
  'Frostfen should satisfy the shared geometry and authored-bound validator cleanly');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass('fighter'), 'the Frostfen route fixture should choose Fighter');
  engine.state.player.level = 60;
  engine.getRouteState().frostfen = { killsByMap: { ashglassPass: 34 } };
  check(engine.changeMap(MAP_ID), 'the route fixture should enter Frostfen Outskirts');

  const livePortal = engine.runtime.portals.find((portal) => portal.id === 'frostfen_glacier');
  const liveEnemies = engine.enemies.filter((enemy) => enemy && enemy.hp > 0);
  check(liveEnemies.length === 32,
    'the opening wave should contain the complete 32-enemy gate objective');
  check(engine.getPortalBlockReason(livePortal) ===
    'Frostfen Route: clear Frostfen Outskirts (0/32).',
  'Glacier Ascent should truthfully explain its opening route lock');

  check(engine.setWorldMapGuideTarget('glacierSpine'),
    'the world guide should accept Glacier Spine as a destination');
  const lockedGuidance = engine.getQuestGuidanceSnapshot();
  check(lockedGuidance.navigationTarget &&
    lockedGuidance.navigationTarget.portalId === 'frostfen_glacier' &&
    lockedGuidance.navigationTarget.label === 'Glacier Spine' &&
    lockedGuidance.navigationTarget.x === livePortal.x + livePortal.w / 2 &&
    !!lockedGuidance.navigationTarget.lockedReason,
  'locked navigation should point at the real far-right Glacier gate center');
  const minimapGuideUi = Object.create(ProjectStarfallUi.prototype);
  minimapGuideUi.snapshot = { questGuidance: lockedGuidance };
  check(JSON.stringify(minimapGuideUi.getMinimapPortalGuide()) === JSON.stringify({
    portalId: 'frostfen_glacier',
    portalLabel: 'Glacier Spine',
    locked: true
  }), 'the minimap should mirror the locked far-right Glacier guide');

  liveEnemies.slice(0, 31).forEach((enemy) => engine.defeatEnemy(enemy));
  check(engine.getPortalBlockReason(livePortal) ===
    'Frostfen Route: clear Frostfen Outskirts (31/32).',
  'the final enemy should remain visibly meaningful at 31/32');
  engine.defeatEnemy(liveEnemies[31]);
  check(engine.getPortalBlockReason(livePortal) === '',
    'clearing the real opening population should unlock Glacier Ascent immediately');

  const openGuidance = engine.getQuestGuidanceSnapshot();
  check(openGuidance.navigationTarget &&
    openGuidance.navigationTarget.portalId === 'frostfen_glacier' &&
    openGuidance.navigationTarget.x === livePortal.x + livePortal.w / 2 &&
    openGuidance.navigationTarget.lockedReason === '',
  'unlocked navigation should remain centered on the physical Glacier gate');
  minimapGuideUi.snapshot = { questGuidance: openGuidance };
  check(JSON.stringify(minimapGuideUi.getMinimapPortalGuide()) === JSON.stringify({
    portalId: 'frostfen_glacier',
    portalLabel: 'Glacier Spine',
    locked: false
  }), 'the minimap should remove locked styling as soon as the route is clear');

  const glacierPath = engine.getWorldMapPath(MAP_ID, 'glacierSpine');
  check(glacierPath &&
    glacierPath.steps.length === 1 &&
    glacierPath.steps[0].portalId === 'frostfen_glacier' &&
    glacierPath.lockedReason === '',
  'the world guide should expose one unlocked physical step to Glacier Spine');

  engine.state.player.x = livePortal.x + livePortal.w / 2 - engine.state.player.w / 2;
  engine.state.player.y = engine.runtime.platforms[0].y - engine.state.player.h;
  engine.updateActiveStation();
  check(engine.state.player.activePortalId === 'frostfen_glacier' &&
    engine.tryEnterActivePortal() &&
    engine.state.mapId === 'glacierSpine',
  'physically overlapping Glacier Ascent should travel to Glacier Spine');

  const returnPortal = engine.runtime.portals.find((portal) =>
    portal.id === 'glacier_frostfen_outskirts'
  );
  check(returnPortal &&
    engine.state.player.grounded &&
    engine.state.player.groundedPlatformId === returnPortal.platformId &&
    actorCenterX(engine.state.player) === returnPortal.x + returnPortal.w / 2,
  'Glacier Spine entry should land grounded and centered on the reciprocal return');
  engine.updateActiveStation();
  check(engine.state.player.activePortalId === 'glacier_frostfen_outskirts' &&
    engine.tryEnterActivePortal() &&
    engine.state.mapId === MAP_ID,
  'the reciprocal Tundra Return should travel back to Frostfen');

  const returnedPortal = engine.runtime.portals.find((portal) => portal.id === 'frostfen_glacier');
  check(returnedPortal &&
    engine.state.player.grounded &&
    engine.state.player.groundedPlatformId === returnedPortal.platformId &&
    actorCenterX(engine.state.player) === returnedPortal.x + returnedPortal.w / 2 &&
    returnedPortal.x === 8280,
  'return travel should land at the new far-right Frostfen gate');
} finally {
  Math.random = originalRandom;
}

console.log(`Project Starfall Frostfen route checks passed: ${checks}`);
