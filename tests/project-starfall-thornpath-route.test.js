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
  'Thornpath should preserve its quick early-game combat band before the first regional field');
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

const greenroot = starfallData.MAPS.find((map) => map.id === 'greenrootMeadow');
const rustcoilOutpost = starfallData.MAPS.find((map) => map.id === 'rustcoilOutpost');
const rustcoil = starfallData.MAPS.find((map) => map.id === 'rustcoilRuins');
const banditRidge = starfallData.MAPS.find((map) => map.id === 'banditRidgeCamp');
const rustcoilPortal = rustcoilOutpost.portals.find((portal) => portal.id === 'rustcoil_outpost_ruins');
const ridgePortal = thornpath.portals.find((portal) => portal.id === 'thornpath_bandit');
const rustcoilEdge = starfallData.WORLD_MAP_EDGES.find((edge) => edge.id === 'rustcoil_outpost_ruins');
const ridgeEdge = starfallData.WORLD_MAP_EDGES.find((edge) => edge.id === 'thornpath_bandit');
const fieldScoutQuest = starfallData.QUESTS.find((quest) => quest.id === 'field_scout');
const trialReadyQuest = starfallData.QUESTS.find((quest) => quest.id === 'trial_ready');
const rustcoilRelayQuest = starfallData.QUESTS.find((quest) => quest.id === 'rustcoil_relay');
const ridgeCourierQuest = starfallData.QUESTS.find((quest) => quest.id === 'ridge_courier');
const ridgeCleanupQuest = starfallData.QUESTS.find((quest) => quest.id === 'ridge_cleanup');
check(greenroot.levelRange[1] >= thornpath.levelRange[0] &&
  thornpath.levelRange[1] >= rustcoil.levelRange[0] &&
  rustcoil.levelRange[1] >= banditRidge.levelRange[0],
'the opening combat maps should use overlapping level bands instead of a level 6 to 12/18 progression hole');
check(rustcoilOutpost.levelRange[0] === 6 &&
  rustcoil.levelRange.join(',') === '6,20' &&
  rustcoil.scaleEnemies === true &&
  banditRidge.levelRange.join(',') === '12,30' &&
  banditRidge.scaleEnemies === true,
'Rustcoil and Bandit Ridge should preserve their maps and enemy rosters while scaling into survivable regional bands');
check(rustcoilPortal.requiredLevel === 6 &&
  rustcoilPortal.label === 'Rustcoil Ruins (Lv 6)' &&
  rustcoilEdge.requiredLevel === 6 &&
  ridgePortal.requiredLevel === 12 &&
  ridgePortal.label === 'Bandit Ridge (Lv 12)' &&
  ridgeEdge.requiredLevel === 12,
'physical portals and world-map edges should publish the same visible level gates');
check(fieldScoutQuest.nextQuestId === 'rustcoil_relay' &&
  trialReadyQuest.requiredQuestIds.includes('field_scout') &&
  rustcoilRelayQuest.requiredLevel === 6 &&
  ridgeCourierQuest.requiredLevel === 12 &&
  ridgeCleanupQuest.requiredLevel === 12,
'the post-Field-Scout handoff and regional quests should follow the repaired 6 then 12 progression milestones');

const originalRandom = Math.random;
Math.random = () => 0.5;
try {
  const routeEngine = createProjectStarfallEngine(null, starfallData);
  check(routeEngine.chooseClass('fighter') && routeEngine.changeMap(greenroot.id),
    'the early-route contract should begin with a playable Fighter in Greenroot');
  const greenrootEnemies = routeEngine.enemies.filter((enemy) => enemy && enemy.hp > 0);
  check(greenrootEnemies.length === 18,
    'the early-route contract should clear the actual 18-enemy Greenroot population');
  greenrootEnemies.forEach((enemy) => routeEngine.defeatEnemy(enemy));
  check(routeEngine.usePortal('greenroot_thornpath'),
    'clearing Greenroot should open the physical Thornpath route');
  const thornpathEnemies = routeEngine.enemies.filter((enemy) => enemy && enemy.hp > 0);
  check(thornpathEnemies.length === 24,
    'the early-route contract should clear the actual 24-enemy Thornpath population');
  thornpathEnemies.forEach((enemy) => routeEngine.defeatEnemy(enemy));
  check(routeEngine.state.player.level < 6,
    'the real opening populations should expose that the character is still below the first regional field band');

  routeEngine.state.progress.claimedQuestIds = Array.from(new Set(
    routeEngine.state.progress.claimedQuestIds.concat(['first_steps', 'field_scout'])
  ));
  const lockedRelay = routeEngine.getQuestAvailability('rustcoil_relay');
  const lockedCourier = routeEngine.getQuestAvailability('ridge_courier');
  check(lockedRelay.lockedReason === 'Reach Level 6 first.' &&
    lockedCourier.lockedReason === 'Reach Level 12 first.',
  'post-Field-Scout quests should advertise the next safe milestones instead of sending a low-level player to Ridge');
  check(routeEngine.getPortalBlockReason(routeEngine.runtime.portals.find((portal) => portal.id === 'thornpath_bandit')) === 'Level 12 required.',
    'Bandit Ridge should remain visibly locked below its survivable entry band');
  check(routeEngine.usePortal('thornpath_rustcoil_outpost') &&
    routeEngine.state.mapId === 'rustcoilOutpost' &&
    routeEngine.getPortalBlockReason(routeEngine.runtime.portals.find((portal) => portal.id === 'rustcoil_outpost_ruins')) === 'Level 6 required.' &&
    !routeEngine.usePortal('rustcoil_outpost_ruins'),
  'the safe outpost should remain visitable while Rustcoil combat truthfully blocks an underleveled character');

  routeEngine.state.player.level = 6;
  check(routeEngine.getQuestAvailability('rustcoil_relay').available &&
    routeEngine.usePortal('rustcoil_outpost_ruins') &&
    routeEngine.state.mapId === 'rustcoilRuins',
  'level 6 should unlock the first regional quest and physical Rustcoil field');
  check(routeEngine.enemies.length === 28 &&
    routeEngine.enemies.every((enemy) => enemy.level >= 6 && enemy.level <= 7),
  'a level-6 Rustcoil arrival should receive the real 28-enemy population scaled to levels 6/7');
  const scrapWardenData = starfallData.ENEMIES.find((enemy) => enemy.id === 'scrapWarden');
  const scaledWarden = routeEngine.createEnemy(scrapWardenData, routeEngine.runtime.spawnPoints[0]);
  check(scaledWarden.level >= 6 && scaledWarden.level <= 7,
    'Rustcoil scaling should prevent its native level-24 Scrap Warden from becoming an opening difficulty spike');

  check(routeEngine.usePortal('rustcoil_outpost_return') &&
    routeEngine.usePortal('rustcoil_outpost_thornpath') &&
    routeEngine.state.mapId === 'thornpathThicket',
  'the regional bridge should preserve the existing two-way physical route');
  routeEngine.state.player.level = 12;
  check(routeEngine.getQuestAvailability('ridge_courier').available &&
    routeEngine.usePortal('thornpath_bandit') &&
    routeEngine.state.mapId === 'banditRidgeCamp',
  'level 12 should align the Ridge courier handoff with the physical Bandit portal');
  check(routeEngine.enemies.length === 30 &&
    routeEngine.enemies.every((enemy) => enemy.level >= 12 && enemy.level <= 13),
  'a level-12 Bandit Ridge arrival should receive its real 30-enemy population at levels 12/13');

  routeEngine.state.progress.claimedQuestIds.push('ridge_courier');
  check(routeEngine.getQuestAvailability('ridge_cleanup').available,
    'the Ridge cleanup quest should become available at the same level as its scaled combat field');
} finally {
  Math.random = originalRandom;
}

const mapValidation = validateMap(thornpath);
check(mapValidation.issues.length === 0 && mapValidation.warnings.length === 0,
  'Thornpath should satisfy the shared geometry and authored-bound validator without exceptions');
check(validateProjectStarfallMaps(starfallData, { includeWarnings: false }).issues.length === 0,
  'all published route portals and section metadata should remain inside their authored maps');

console.log(`Project Starfall Thornpath route checks passed: ${checks}`);
