'use strict';

const assert = require('assert');

const starfallData = require('../js/games/project-starfall/data/index.js');
const mapRuntime = require('../js/games/project-starfall/engine/map-runtime.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');
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

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

const greenroot = starfallData.MAPS.find((map) => map.id === 'greenrootMeadow');
check(!!greenroot, 'Greenroot Meadow should remain published');
check(greenroot.compactWorldWidth === 4200 && greenroot.waveMax === 24,
  'Greenroot should retain its compact world and legacy wave contract');
check(greenroot.scaleEnemies === true,
  'Greenroot should scale its mixed native roster to starter-character levels');

const nonGroundPlatforms = greenroot.platforms.slice(1);
const broadLanes = nonGroundPlatforms.filter((platform) =>
  platformKind(platform) === 'solidLane' && platformW(platform) >= 640
);
const slopes = nonGroundPlatforms.filter((platform) => platformKind(platform) === 'slope');
const connectors = nonGroundPlatforms.filter((platform) =>
  platformW(platform) >= 120 && platformW(platform) <= 320
);
check(greenroot.platforms.length >= 18 && broadLanes.length >= 9 &&
  slopes.length === 6 && connectors.length >= 6 && greenroot.climbables.length >= 9,
  'Greenroot should retain roomy combat lanes, six readable ramps, and optional rope recovery');
check(Math.min(...slopes.map(platformX)) >= 520,
  'the first ramp should begin after a readable flat arrival apron');

const entryPortal = greenroot.portals.find((portal) => portal.returnPortal);
const exitPortal = greenroot.portals.find((portal) => portal.id === 'greenroot_thornpath');
const guide = greenroot.questNpcs.find((npc) => npc.id === 'greenroot_guide');
const spawnXs = greenroot.spawnPoints.map((point) => Number(point.x || 0));
check(entryPortal && exitPortal && guide &&
  Math.min(...slopes.map(platformX)) >= guide.x + 200 &&
  Math.min(...spawnXs) >= guide.x + 400 &&
  Math.max(...spawnXs) <= exitPortal.x - 400,
  'arrival interactions and both portals should have calm, enemy-free approach space');

const sections = greenroot.fieldComposition.routeSections;
check(sections.map((section) => section.label).join('|') ===
  'Starter Pond Loop|Moss Lane Extension|Canopy Practice|Thornpath Gate',
  'Greenroot should progress through four contiguous, named route beats');
check(sections.reduce((right, section) => {
  assert.strictEqual(section.x, right, `${section.label} should start where the prior section ends`);
  return section.x + section.w;
}, 0) === 4200, 'Greenroot route sections should cover the full compact map without gaps');

const landmarkBands = greenroot.fieldComposition.landmarkBands;
const anchors = landmarkBands.map((band) => Number(band.anchorX));
check(anchors.every(Number.isFinite) &&
  new Set(anchors).size === anchors.length &&
  sections.every((section) => landmarkBands.some((band) =>
    Number(band.anchorX) >= section.x && Number(band.anchorX) < section.x + section.w
  )),
  'each route beat should own a stable authored visual landmark');

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

const laneAt = (x, y) => broadLanes.find((platform) =>
  platformY(platform) === y && platformX(platform) <= x && platformX(platform) + platformW(platform) >= x
);
const highRoute = nonGroundPlatforms
  .filter((platform) => platformY(platform) === 200 &&
    ['solidLane', 'connector'].includes(platformKind(platform)))
  .sort((left, right) => platformX(left) - platformX(right));
const highRouteMaxGap = highRoute.slice(1).reduce((maxGap, platform, index) =>
  Math.max(maxGap, platformX(platform) - (platformX(highRoute[index]) + platformW(highRoute[index]))), 0);
check(highRoute.length >= 4 &&
  platformX(highRoute[0]) <= 1600 &&
  platformX(highRoute.at(-1)) + platformW(highRoute.at(-1)) >= 4000 &&
  highRouteMaxGap <= 120,
  'the canopy shortcut should be a continuous, visible high route');
check(laneAt(2800, 456) && laneAt(2800, 328),
  'the canopy should offer a forgiving two-tier recovery drop');

const runtime = mapRuntime.createMapRuntime(greenroot, null, { maps: starfallData.MAPS });
check(runtime.trainingRoute.viable && runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1,
  'the authored route should remain fully connected, loopable, and spawn-covered at runtime');
check(runtime.spawnGroups.length === 4 &&
  runtime.spawnGroups.map((group) => group.population).join(',') === '4,4,5,5' &&
  runtime.spawnGroups.map((group) => group.maxPopulation).join(',') === '4,4,5,5' &&
  runtime.spawnGroups.map((group) => group.respawnSeconds).join(',') === '7,7,8,9',
  'encounter pacing should grow deliberately from two tutorial pockets into two mixed groups');

const claimedPlatforms = new Set();
runtime.spawnGroups.forEach((group) => {
  check(group.platformIds.every((platformId) => {
    if (claimedPlatforms.has(platformId)) return false;
    claimedPlatforms.add(platformId);
    return true;
  }), `${group.label} should exclusively own its combat platforms`);
  const section = sections.find((entry) => group.sectionId === `greenrootMeadow_${entry.label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')}`);
  check(group.platformIds.length > 0 &&
    group.spawnBounds &&
    section &&
    group.spawnBounds.minX >= section.x &&
    group.spawnBounds.maxX <= section.x + section.w,
  `${group.label} should have a physical territory inside its named route section`);
});
check(runtime.spawnGroups[0].enemyWeights.map((entry) => entry.enemyId).join(',') === 'slimelet' &&
  runtime.spawnGroups[1].enemyWeights.map((entry) => entry.enemyId).join(',') === 'dewSlime' &&
  runtime.spawnGroups[2].enemyWeights.some((entry) => entry.enemyId === 'thornSprout') &&
  runtime.spawnGroups[3].enemyWeights.some((entry) => entry.enemyId === 'mossback'),
  'enemy complexity should unlock in order instead of spiking in the arrival pocket');

const forestRoute = starfallData.WORLD_ROUTES.find((route) => route.id === 'forest');
const greenrootGoal = forestRoute.fieldGoals.find((goal) => goal.mapId === greenroot.id);
check(runtime.spawnGroups.reduce((total, group) => total + group.population, 0) === greenrootGoal.count,
  'one complete four-pocket circuit should match the 18-kill Forest Route goal');

const spreadEngine = createProjectStarfallEngine(null, starfallData);
spreadEngine.chooseClass('fighter');
spreadEngine.changeMap(greenroot.id);
const initialAreaKeys = spreadEngine.enemies.map((enemy) => spreadEngine.getFieldSpawnAreaKey(enemy));
const initialPositionKeys = spreadEngine.enemies.map((enemy) =>
  `${enemy.spawnPlatformId}:${Math.round(Number(enemy.spawnX || 0))}`
);
check(spreadEngine.enemies.length === 18 &&
  new Set(initialAreaKeys).size === 18 &&
  new Set(initialPositionKeys).size === 18,
  'the opening circuit should populate all 18 enemies across distinct physical areas');
check(spreadEngine.enemies.every((enemy) => {
  const group = spreadEngine.runtime.spawnGroups.find((entry) => entry.id === enemy.spawnGroupId);
  const section = sections.find((entry) => group && group.sectionId.endsWith(entry.label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')));
  const x = Number(enemy.spawnX || 0);
  return group && section &&
    x >= Math.max(group.spawnBounds.minX, section.x) &&
    x <= Math.min(group.spawnBounds.maxX, section.x + section.w) &&
    x >= guide.x + 400 &&
    x <= exitPortal.x - 400;
}), 'live opening enemies should stay inside their named section and both portal safety margins');
check(spreadEngine.runtime.spawnGroups.every((group) => {
  const candidates = spreadEngine.createFieldRespawnCandidates({ spawnGroupId: group.id }, 0);
  return candidates.length > 0 && candidates.every((candidate) =>
    candidate.spawn.x >= group.spawnBounds.minX &&
    candidate.spawn.x <= group.spawnBounds.maxX
  );
}), 'replacement candidates should preserve the same physical territory bounds');

const engine = createProjectStarfallEngine(null, starfallData);
engine.state.mapId = greenroot.id;
engine.state.player.level = 1;
const mossback = starfallData.ENEMIES.find((enemy) => enemy.id === 'mossback');
const gateGroup = runtime.spawnGroups.find((group) => group.label === 'Gate Guardians');
const scaledLevels = Array.from({ length: 40 }, () =>
  engine.createEnemy(mossback, {
    x: gateGroup.spawnBounds.minX,
    y: 200,
    platformIndex: gateGroup.platformIndices[0]
  }).level
);
check(scaledLevels.every((level) => level >= 1 && level <= 2),
  'even the late-pocket Mossback should scale to a fair level 1 starter range');

const mapValidation = validateMap(greenroot);
check(mapValidation.issues.length === 0 && mapValidation.warnings.length === 0,
  'Greenroot should satisfy the shared map geometry validator without exceptions');

console.log(`Project Starfall Greenroot route checks passed: ${checks}`);
