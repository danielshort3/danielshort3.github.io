'use strict';

const assert = require('assert');

const starfallData = require('../js/games/project-starfall/data/index.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const {
  validateMap,
  validateProjectStarfallMaps
} = require('../build/validate-project-starfall-maps.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function platformX(platform) {
  return Number(Array.isArray(platform) ? platform[0] : platform && platform.x) || 0;
}

function platformY(platform) {
  return Number(Array.isArray(platform) ? platform[1] : platform && platform.y) || 0;
}

function platformW(platform) {
  return Number(Array.isArray(platform) ? platform[2] : platform && platform.w) || 0;
}

function platformKind(platform) {
  if (platform && !Array.isArray(platform) && platform.shape === 'slope') return 'slope';
  return String(platform && !Array.isArray(platform) && platform.terrainVisual && platform.terrainVisual.kind || 'flat');
}

function sectionToken(label) {
  return String(label || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

const thornpath = starfallData.MAPS.find((map) => map.id === 'thornpathThicket');
check(!!thornpath, 'Thornpath Thicket should remain published');
check(thornpath.palette.join('|') === '#3f8f58|#5b3d2d|#c4475d' &&
  thornpath.asset === 'img/project-starfall/maps/thornpath-thicket.webp' &&
  thornpath.environment.terrain === 'thornpath-thicket' &&
  thornpath.environment.props === 'thornpath-thicket',
'the pass should preserve Thornpath’s playful palette, map painting, terrain, and prop family');
check(thornpath.layoutStyle === 'verticalCanopy' &&
  thornpath.geometryGenerator === 'priorityFieldV2' &&
  thornpath.compactWorldWidth === 5200,
'Thornpath should keep its vertical-canopy identity inside a deliberate 5200px route');
check(thornpath.levelRange.join(',') === '2,6' && thornpath.scaleEnemies === true,
  'Thornpath should meet normal post-Greenroot characters at a fair scaled level band');
check(thornpath.waveMax === 24,
  'one authored Thornpath circuit should use the 24-enemy Forest Route contract');

const ground = thornpath.platforms[0];
const nonGroundPlatforms = thornpath.platforms.slice(1);
const broadLanes = nonGroundPlatforms.filter((platform) =>
  platformKind(platform) === 'solidLane' && platformW(platform) >= 640
);
const slopes = nonGroundPlatforms.filter((platform) => platformKind(platform) === 'slope');
check(platformW(ground) === 5200 &&
  broadLanes.length === 12 &&
  slopes.length === 6 &&
  thornpath.climbables.length >= 12,
'the route should provide twelve roomy combat lanes, six readable ramps, and plentiful vine recovery');
check(Math.min(...slopes.map(platformX)) >= 600,
  'the first ramp should begin after the return portal and Thornpath Scout arrival apron');

const scout = thornpath.questNpcs.find((npc) => npc.id === 'thornpath_scout');
const returnPortal = thornpath.portals.find((portal) => portal.returnPortal);
const exits = thornpath.portals
  .filter((portal) => !portal.returnPortal)
  .sort((left, right) => left.x - right.x);
check(scout && returnPortal &&
  returnPortal.x < scout.x &&
  scout.x + 200 <= Math.min(...slopes.map(platformX)),
'the playful scout handoff should remain calm and unobstructed');
check(exits.map((portal) => portal.x).join(',') === '4920,5080' &&
  exits[1].x - exits[0].x >= 124 &&
  exits.every((portal) => portal.x + 58 <= platformX(ground) + platformW(ground) - 18),
'both destination choices should be distinct and truthfully authored on the final ground');

const sections = thornpath.fieldComposition.routeSections;
check(sections.map((section) => section.label).join('|') ===
  'Scout’s Clearing|Vine Tangle|Thorn Canopy|Deep Fork',
'Thornpath should progress through four named and readable route beats');
check(sections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === 5200,
'the four route beats should cover the compact map without gaps or off-map metadata');

const landmarks = thornpath.fieldComposition.landmarkBands;
const landmarkAnchors = landmarks.map((landmark) => Number(landmark.anchorX));
check(landmarkAnchors.join(',') === '560,1650,3100,4700' &&
  sections.every((section) => landmarks.some((landmark) =>
    landmark.anchorX >= section.x && landmark.anchorX < section.x + section.w
  )),
'each route beat should own a stable authored landmark anchor');

const broadLaneOverlaps = [];
broadLanes.forEach((lane, laneIndex) => {
  broadLanes.slice(laneIndex + 1).forEach((other) => {
    if (platformY(lane) !== platformY(other)) return;
    const overlap = Math.min(platformX(lane) + platformW(lane), platformX(other) + platformW(other)) -
      Math.max(platformX(lane), platformX(other));
    if (overlap > 0) broadLaneOverlaps.push(overlap);
  });
});
check(broadLaneOverlaps.length === 0,
  'same-height combat lanes should use clean handoffs instead of overlapping collision seams');

const runtime = mapRuntime.createMapRuntime(thornpath, null, { maps: starfallData.MAPS });
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1 &&
  runtime.trainingRoute.reachableTierCount >= 4,
'the canopy should remain connected, loopable, fully spawn-covered, and vertically varied at runtime');
check(runtime.spawnGroups.length === 4 &&
  runtime.spawnGroups.map((group) => group.population).join(',') === '5,6,7,6' &&
  runtime.spawnGroups.map((group) => group.maxPopulation).join(',') === '5,6,7,6' &&
  runtime.spawnGroups.map((group) => group.respawnSeconds).join(',') === '7,8,8,9',
'encounter pacing should grow through four bounded lessons without hidden population inflation');

const claimedPlatforms = new Set();
runtime.spawnGroups.forEach((group) => {
  const section = sections.find((entry) =>
    group.sectionId === `thornpathThicket_${sectionToken(entry.label)}`
  );
  check(group.platformIds.length > 0 && group.platformIds.every((platformId) => {
    if (claimedPlatforms.has(platformId)) return false;
    claimedPlatforms.add(platformId);
    const platform = runtime.platforms.find((entry) => entry.id === platformId);
    return platform &&
      platformKind(platform) === 'solidLane' &&
      platform.w >= 640;
  }), `${group.label} should exclusively own broad, flat combat lanes`);
  check(section &&
    group.spawnBounds &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${group.label} should keep its physical territory inside its named route beat`);
  check(group.spawnPointIds.length > 0 && group.spawnPointIds.every((spawnPointId) => {
    const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
    return point &&
      point.x >= group.spawnBounds.minX &&
      point.x <= group.spawnBounds.maxX;
  }), `${group.label} should own authored spawn anchors inside its physical bounds`);
});

check(runtime.spawnGroups[0].enemyWeights.every((entry) =>
  !['vineSnapper', 'briarStag'].includes(entry.enemyId)
) &&
  runtime.spawnGroups[1].enemyWeights.some((entry) => entry.enemyId === 'vineSnapper') &&
  !runtime.spawnGroups[1].enemyWeights.some((entry) => entry.enemyId === 'briarStag') &&
  runtime.spawnGroups[2].enemyWeights.some((entry) => entry.enemyId === 'briarStag'),
'enemy mechanics should unlock in order: basics, snapper, then the heavy briar charge');
check(exits[0].x - runtime.spawnGroups.at(-1).spawnBounds.maxX >= 480,
  'the final encounter should leave a calm 480px decision approach before either exit');

const forestRoute = starfallData.WORLD_ROUTES.find((route) => route.id === 'forest');
const routeGoal = forestRoute.fieldGoals.find((goal) => goal.mapId === thornpath.id);
const pathfinder = starfallData.ACCOMPLISHMENTS.find((entry) => entry.id === 'greenroot_pathfinder');
const accomplishmentGoal = pathfinder.objectives.find((objective) =>
  objective.mapId === thornpath.id
);
check(runtime.spawnGroups.reduce((total, group) => total + group.population, 0) === routeGoal.count &&
  routeGoal.count === accomplishmentGoal.count,
'one four-pocket circuit should satisfy both 24-kill Thornpath progression contracts');

const engine = createProjectStarfallEngine(null, starfallData);
engine.chooseClass('fighter');
engine.state.player.level = 2;
engine.changeMap(thornpath.id);
const initialPositionKeys = engine.enemies.map((enemy) =>
  `${enemy.spawnPlatformId}:${Math.round(Number(enemy.spawnX || 0))}`
);
check(engine.enemies.length === 24 &&
  new Set(initialPositionKeys).size === 24 &&
  engine.enemies.every((enemy) => enemy.level >= 2 && enemy.level <= 3),
'a level-2 arrival should get 24 distinct opening positions and fair level-2/3 enemies');
check(engine.enemies.every((enemy) => {
  const group = engine.runtime.spawnGroups.find((entry) => entry.id === enemy.spawnGroupId);
  const x = Number(enemy.spawnX || 0);
  return group &&
    x >= group.spawnBounds.minX &&
    x <= group.spawnBounds.maxX &&
    x >= runtime.spawnGroups[0].spawnBounds.minX &&
    x <= exits[0].x - 480;
}), 'live opening enemies should stay inside their named territory and portal safety margins');
check(engine.runtime.spawnGroups.every((group) => {
  const candidates = engine.createFieldRespawnCandidates({ spawnGroupId: group.id }, 0);
  return candidates.length > 0 && candidates.every((candidate) =>
    candidate.spawn.x >= group.spawnBounds.minX &&
    candidate.spawn.x <= group.spawnBounds.maxX &&
    group.platformIds.includes(candidate.spawn.platformId)
  );
}), 'replacement enemies should preserve the same bounded combat territories');
check(engine.runtime.portals.every((portal) => {
  const authored = thornpath.portals.find((entry) => entry.id === portal.id);
  return authored && authored.x === portal.x;
}), 'runtime portal placement should match source instead of silently clamping stale coordinates');

const mapValidation = validateMap(thornpath);
check(mapValidation.issues.length === 0 && mapValidation.warnings.length === 0,
  'Thornpath should satisfy the shared geometry and authored-bound validator without exceptions');
check(validateProjectStarfallMaps(starfallData, { includeWarnings: false }).issues.length === 0,
  'all published route portals and section metadata should remain inside their authored maps');

console.log(`Project Starfall Thornpath route checks passed: ${checks}`);
