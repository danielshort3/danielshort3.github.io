'use strict';

const assert = require('assert');

const starfallData = require('../js/games/project-starfall/data/index.js');
const engineAssets = require('../js/games/project-starfall/engine/assets.js');
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

function sectionToken(label) {
  return String(label || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function emptyBounty(bounty) {
  return Number(bounty && bounty.currency || 0) === 0 &&
    Object.values(bounty && bounty.materials || {}).every((amount) => Number(amount || 0) === 0);
}

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

const endlessRift = starfallData.MAPS.find((map) => map.id === 'endlessRift');
check(!!endlessRift, 'Endless Rift should remain published');
check(endlessRift.palette.join('|') === '#191b2c|#7bdff2|#f06bff' &&
  endlessRift.asset === 'img/project-starfall/maps/endless-rift.webp' &&
  endlessRift.environment.terrain === 'endless-rift' &&
  endlessRift.environment.props === 'endless-rift',
'the route pass should preserve the Rift palette, background painting, terrain, and prop family');
check(endlessRift.layoutStyle === 'riftStack' &&
  endlessRift.geometryGenerator === 'fieldLayout' &&
  endlessRift.worldHeight === 1260 &&
  endlessRift.authoredGroundY === 1120 &&
  platformW(endlessRift.platforms[0]) === 5200,
'Endless Rift should keep its riftStack identity and authored 5200 by 1260 dimensions');
check(endlessRift.levelRange.join(',') === '100,100' &&
  endlessRift.endlessScaling === true &&
  endlessRift.waveMax === 36,
'the endgame level, endless-scaling, and 36-enemy encounter contracts should remain intact');

const returnPortal = endlessRift.portals.find((portal) => portal.id === 'rift_eclipse');
const watcher = endlessRift.questNpcs.find((npc) => npc.id === 'rift_watcher');
check(returnPortal &&
  returnPortal.returnPortal === true &&
  returnPortal.destinationMapId === 'eclipseFrontier' &&
  returnPortal.x === 110 &&
  returnPortal.platformIndex === 0,
'the Eclipse Return portal should retain its authored role and placement');
check(watcher &&
  watcher.name === 'Rift Watcher' &&
  watcher.asset === 'img/project-starfall/characters/generic-player.png' &&
  watcher.x === 520 &&
  watcher.platformIndex === 0 &&
  watcher.questIds.includes('rift_watch'),
'the playful Rift Watcher handoff should remain unchanged');

const nonGroundPlatforms = endlessRift.platforms.slice(1);
const slopes = nonGroundPlatforms.filter((platform) => platformKind(platform) === 'slope');
const corePlatform = nonGroundPlatforms.find((platform) => platform.id === 'endlessRift_core_dais');
check(!!corePlatform &&
  platformW(corePlatform) >= 640 &&
  corePlatform.spawnDisabled === true,
'the central Rift Core should be a broad, explicitly spawn-free dais');
check(slopes.length === 5 &&
  endlessRift.rampConnections.length === 5 &&
  endlessRift.climbables.length >= 14 &&
  endlessRift.climbables.every((climbable) => /^endlessRift_rune_stair_\d+$/.test(climbable.id)),
'the ring should retain five readable ramps and at least fourteen playful rune-stair recovery links');

const routeSections = endlessRift.fieldComposition.routeSections;
const quadrantSections = {
  sw: routeSections.find((section) => section.label === 'Southwest Rift Quadrant'),
  nw: routeSections.find((section) => section.label === 'Northwest Rift Quadrant'),
  ne: routeSections.find((section) => section.label === 'Northeast Rift Quadrant'),
  se: routeSections.find((section) => section.label === 'Southeast Rift Quadrant')
};
const coreSection = routeSections.find((section) => section.label === 'Rift Core Regroup');
check(Object.values(quadrantSections).every(Boolean) &&
  coreSection &&
  coreSection.platformIds.join(',') === 'endlessRift_core_dais',
'the field composition should describe four physical quadrants around one central regroup dais');

const sourcePlatformById = new Map(endlessRift.platforms.map((platform) => [platform.id, platform]));
Object.entries(quadrantSections).forEach(([quadrant, section]) => {
  const platforms = section.platformIds.map((platformId) => sourcePlatformById.get(platformId));
  check(section.platformIds.length === 3 &&
    section.platformIds.every((platformId) => new RegExp(`^endlessRift_${quadrant}_`).test(platformId)) &&
    platforms.every((platform) =>
      platform &&
      platformKind(platform) !== 'slope' &&
      platformW(platform) >= 640
    ),
  `${section.label} should own exactly three broad, flat, semantically named combat lanes`);
});

const coreCenterX = platformX(corePlatform) + platformW(corePlatform) / 2;
const coreY = platformY(corePlatform);
const quadrantPosition = Object.fromEntries(Object.entries(quadrantSections).map(([quadrant, section]) => {
  const platforms = section.platformIds.map((platformId) => sourcePlatformById.get(platformId));
  return [quadrant, {
    x: average(platforms.map((platform) => platformX(platform) + platformW(platform) / 2)),
    y: average(platforms.map(platformY))
  }];
}));
check(quadrantPosition.nw.x < coreCenterX &&
  quadrantPosition.sw.x < coreCenterX &&
  quadrantPosition.ne.x > coreCenterX &&
  quadrantPosition.se.x > coreCenterX,
'west and east quadrants should physically sit on their promised sides of the central core');
check(quadrantPosition.nw.y < coreY &&
  quadrantPosition.ne.y < coreY &&
  quadrantPosition.sw.y > coreY &&
  quadrantPosition.se.y > coreY,
'north quadrants should be above the core while south quadrants remain below it');

const runtime = mapRuntime.createMapRuntime(endlessRift, null, { maps: starfallData.MAPS });
check(runtime.trainingRoute.viable &&
  runtime.trainingRoute.loopable &&
  runtime.trainingRoute.issues.length === 0 &&
  runtime.trainingRoute.platformCoverage === 1,
'the authored Rift ring should remain viable, loopable, fully spawn-covered, and issue-free at runtime');

const semanticPlatformIds = Object.values(quadrantSections)
  .flatMap((section) => section.platformIds)
  .concat(corePlatform.id);
const semanticPlatformIndices = semanticPlatformIds.map((platformId) => {
  const platform = runtime.platforms.find((entry) => entry.id === platformId);
  assert(platform, `${platformId} should be published into runtime geometry`);
  return platform.index;
});
semanticPlatformIndices.forEach((startIndex) => {
  const reachable = new Set([startIndex]);
  const queue = [startIndex];
  while (queue.length) {
    const index = queue.shift();
    (runtime.platformGraph[index] || []).forEach((link) => {
      if (link.to === 0 || reachable.has(link.to)) return;
      reachable.add(link.to);
      queue.push(link.to);
    });
  }
  check(semanticPlatformIndices.every((index) => reachable.has(index)),
    `${runtime.platforms[startIndex].id} should reach every semantic Rift platform without using ground node 0`);
});

check(runtime.spawnGroups.length === 4 &&
  runtime.spawnGroups.every((group) => group.population === 9 && group.maxPopulation === 9) &&
  runtime.spawnGroups.reduce((total, group) => total + group.population, 0) === 36,
'the four exclusive quadrants should each carry nine enemies for the preserved 36-enemy cap');
const claimedSpawnPlatforms = new Set();
runtime.spawnGroups.forEach((group) => {
  const section = Object.values(quadrantSections).find((entry) =>
    group.sectionId === `endlessRift_${sectionToken(entry.label)}`
  );
  check(section &&
    group.platformIds.length === 3 &&
    group.platformIds.every((platformId) => {
      if (claimedSpawnPlatforms.has(platformId)) return false;
      claimedSpawnPlatforms.add(platformId);
      return section.platformIds.includes(platformId);
    }),
  `${group.label} should exclusively own the three lanes in its platform-authored quadrant`);
  check(group.spawnPointIds.length > 0 &&
    group.spawnPointIds.every((spawnPointId) => {
      const point = runtime.spawnPoints.find((entry) => entry.id === spawnPointId);
      return point &&
        section.platformIds.includes(point.platformId) &&
        point.sectionId === group.sectionId;
    }),
  `${group.label} spawn points should derive their section from their authored platform IDs`);
});
check(semanticPlatformIds
  .filter((platformId) => platformId !== corePlatform.id)
  .every((platformId) => claimedSpawnPlatforms.has(platformId)),
'all twelve quadrant combat lanes should be claimed exactly once');
check(!runtime.spawnGroups.some((group) =>
  group.platformIds.includes(corePlatform.id) ||
  group.sectionId === `endlessRift_${sectionToken(coreSection.label)}`
) &&
  !runtime.spawnPoints.some((point) =>
    point.platformId === corePlatform.id ||
    point.sectionId === `endlessRift_${sectionToken(coreSection.label)}`
  ),
'the central regroup dais should have no enemy group or spawn point');

const mapValidation = validateMap(endlessRift);
check(mapValidation.issues.length === 0 && mapValidation.warnings.length === 0,
  'Endless Rift should satisfy the shared geometry validator without exceptions or warnings');

const rimewarden = starfallData.MAPS.find((map) => map.id === 'rimewardenSanctum');
const rimeEnvironmentPaths = [];
engineAssets.collectMapEnvironmentAssetPaths(rimewarden, starfallData, rimeEnvironmentPaths);
check(rimeEnvironmentPaths.includes('img/project-starfall/environment/ramps/rimewarden-sanctum.png'),
  'critical map loading should include the playful Rimewarden ramp atlas instead of rendering brown fallback slopes');

function createRiftEngine(riftPatch) {
  const engine = createProjectStarfallEngine(null, starfallData);
  engine.chooseClass('fighter');
  engine.changeMap('endlessRift', { silent: true });
  if (riftPatch && typeof riftPatch === 'object') {
    const payload = engine.serialize();
    payload.state.rift = Object.assign({}, payload.state.rift, riftPatch);
    payload.state.mapId = 'endlessRift';
    check(engine.restore(payload), 'public restore should accept an authored active Rift run');
  }
  return engine;
}

const pressureEngine = createRiftEngine();
check(typeof pressureEngine.getRiftPressureProfile === 'function',
  'the engine should expose the public Rift pressure profile');
const tierOnePressure = pressureEngine.getRiftPressureProfile({ tier: 1, mutationIds: [], surgeActive: false });
const tierTenPressure = pressureEngine.getRiftPressureProfile({ tier: 10, mutationIds: [], surgeActive: false });
const tierTwentyPressure = pressureEngine.getRiftPressureProfile({ tier: 20, mutationIds: [], surgeActive: false });
check(tierOnePressure.enemyHpScale < tierTenPressure.enemyHpScale &&
  tierTenPressure.enemyHpScale < tierTwentyPressure.enemyHpScale &&
  tierOnePressure.enemyDamageScale < tierTenPressure.enemyDamageScale &&
  tierTenPressure.enemyDamageScale < tierTwentyPressure.enemyDamageScale,
'tiers 1, 10, and 20 should apply steadily increasing HP and damage pressure');
check([
  'enemyHpScale',
  'enemyDamageScale',
  'enemyDefenseScale',
  'eliteChanceBonus',
  'scoreScale',
  'rewardScale'
].every((key) => Number.isFinite(tierTwentyPressure[key])),
'the public pressure profile should expose finite combat and reward multipliers');

const deterministicA = createRiftEngine({ tier: 10, bestTier: 10, mutationIds: [] }).getRiftSnapshot();
const deterministicB = createRiftEngine({ tier: 10, bestTier: 10, mutationIds: [] }).getRiftSnapshot();
check(deterministicA.mutationIds.length > 0 &&
  deterministicA.mutationIds.join(',') === deterministicB.mutationIds.join(','),
'a Rift tier should draft the same non-empty mutation set across independent runs');

const mutationBaseline = pressureEngine.getRiftPressureProfile({
  tier: 10,
  mutationIds: [],
  surgeActive: false
});
const functionalMutation = starfallData.MUTATIONS.find((mutation) => {
  const profile = pressureEngine.getRiftPressureProfile({
    tier: 10,
    mutationIds: [mutation.id],
    surgeActive: false
  });
  return profile.enemyHpScale !== mutationBaseline.enemyHpScale ||
    profile.enemyDamageScale !== mutationBaseline.enemyDamageScale ||
    profile.enemyDefenseScale !== mutationBaseline.enemyDefenseScale ||
    profile.eliteChanceBonus !== mutationBaseline.eliteChanceBonus ||
    profile.scoreScale !== mutationBaseline.scoreScale ||
    profile.rewardScale !== mutationBaseline.rewardScale;
});
check(!!functionalMutation,
  'at least one authored mutation should change the public pressure profile rather than remain flavor text');

const hpMutation = starfallData.MUTATIONS.find((mutation) => Number(mutation.enemyHpScale || 1) > 1);
const damageMutation = starfallData.MUTATIONS.find((mutation) => Number(mutation.enemyDamageScale || 1) > 1);
check(!!hpMutation && !!damageMutation,
  'the Rift mutation pool should include observable HP and damage pressure choices');
const hpMutationEngine = createRiftEngine({
  tier: 10,
  bestTier: 10,
  mutationIds: [hpMutation.id]
});
const damageMutationEngine = createRiftEngine({
  tier: 10,
  bestTier: 10,
  mutationIds: [damageMutation.id]
});
const hpMutationPressure = hpMutationEngine.getRiftPressureProfile();
const damageMutationPressure = damageMutationEngine.getRiftPressureProfile();
const enemyData = starfallData.ENEMIES.find((enemy) => enemy.id === 'riftAberration');
const hpSpawnPlatform = hpMutationEngine.runtime.platforms.find((platform) =>
  hpMutationEngine.runtime.spawnGroups[0].platformIds.includes(platform.id)
);
const damageSpawnPlatform = damageMutationEngine.runtime.platforms.find((platform) =>
  damageMutationEngine.runtime.spawnGroups[0].platformIds.includes(platform.id)
);
const originalRandom = Math.random;
let hpMutationEnemy;
let damageMutationEnemy;
try {
  Math.random = () => 0.25;
  hpMutationEnemy = hpMutationEngine.createEnemy(enemyData, {
    x: hpSpawnPlatform.x + hpSpawnPlatform.w / 2,
    y: hpSpawnPlatform.y,
    platformIndex: hpSpawnPlatform.index,
    platformId: hpSpawnPlatform.id
  });
  Math.random = () => 0.25;
  damageMutationEnemy = damageMutationEngine.createEnemy(enemyData, {
    x: damageSpawnPlatform.x + damageSpawnPlatform.w / 2,
    y: damageSpawnPlatform.y,
    platformIndex: damageSpawnPlatform.index,
    platformId: damageSpawnPlatform.id
  });
} finally {
  Math.random = originalRandom;
}
const liveHpRatio = hpMutationEnemy.maxHp / damageMutationEnemy.maxHp;
const pressureHpRatio = hpMutationPressure.enemyHpScale / damageMutationPressure.enemyHpScale;
const liveDamageRatio = hpMutationEnemy.damage / damageMutationEnemy.damage;
const pressureDamageRatio = hpMutationPressure.enemyDamageScale / damageMutationPressure.enemyDamageScale;
check(Math.abs(liveHpRatio - pressureHpRatio) < 0.03 &&
  Math.abs(liveDamageRatio - pressureDamageRatio) < 0.03,
'live enemies created at the same tier and RNG should reflect the selected mutation pressure');

const riftDefinition = starfallData.MAP_MECHANIC_DEFINITIONS.endlessRift;
check(riftDefinition.requiredSectionOrder === true &&
  riftDefinition.killsPerSection === 3 &&
  riftDefinition.activeSectionIds.join('|') === [
    'endlessRift_southwest_rift_quadrant',
    'endlessRift_northwest_rift_quadrant',
    'endlessRift_northeast_rift_quadrant',
    'endlessRift_southeast_rift_quadrant'
  ].join('|') &&
  riftDefinition.rotationsPerTier === 3,
'the route contract should require three defeats per quadrant across three ordered southwest-to-southeast rotations per tier');

function recordSectionDefeat(engine, sectionId) {
  return engine.recordMapMechanicDefeat({
    spawnSectionId: sectionId,
    sectionId,
    data: { behavior: 'ground' }
  });
}

function completeRiftRotation(engine) {
  const before = engine.getMapMechanicSnapshot('endlessRift');
  const targetCycles = before.completedCycles + 1;
  for (let attempts = 0; attempts < 200; attempts += 1) {
    const snapshot = engine.getMapMechanicSnapshot('endlessRift');
    if (snapshot.completedCycles >= targetCycles) return snapshot;
    assert(riftDefinition.activeSectionIds.includes(snapshot.nextSectionId),
      `ordered Rift rotation should expose a valid next section, got ${snapshot.nextSectionId || 'none'}`);
    recordSectionDefeat(engine, snapshot.nextSectionId);
  }
  assert.fail('ordered Rift rotation should finish within 200 public defeat events');
}

function placePlayerAtRiftCore(engine) {
  const core = engine.runtime.platforms.find((platform) => platform.id === riftDefinition.corePlatformId);
  assert(core, 'Rift Core platform should exist at runtime');
  assert(engine.placePlayerOnRuntimePlatform(core.index, core.x + core.w / 2),
    'public platform placement should move the player onto the Rift Core');
  check(engine.isPlayerAtRiftCore(),
    'the engine should recognize the player standing on the authored Rift Core dais');
}

const pushEngine = createRiftEngine();
let orderedSnapshot = pushEngine.getMapMechanicSnapshot('endlessRift');
check(Array.isArray(orderedSnapshot.orderedSectionIds) &&
  orderedSnapshot.orderedSectionIds.length === 0 &&
  orderedSnapshot.nextSectionId === riftDefinition.activeSectionIds[0],
'a new Rift rotation should begin empty and point to the southwest quadrant');
recordSectionDefeat(pushEngine, riftDefinition.activeSectionIds[1]);
for (let campKill = 0; campKill < 7; campKill += 1) {
  recordSectionDefeat(pushEngine, riftDefinition.activeSectionIds[1]);
}
orderedSnapshot = pushEngine.getMapMechanicSnapshot('endlessRift');
check(orderedSnapshot.orderedSectionIds.length === 0 &&
  orderedSnapshot.nextSectionId === riftDefinition.activeSectionIds[0] &&
  orderedSnapshot.currentSectionKillCount === 0 &&
  orderedSnapshot.progress === 0,
'out-of-order quadrant camping should not bank kills toward rotation progress');

recordSectionDefeat(pushEngine, riftDefinition.activeSectionIds[0]);
recordSectionDefeat(pushEngine, riftDefinition.activeSectionIds[0]);
orderedSnapshot = pushEngine.getMapMechanicSnapshot('endlessRift');
check(orderedSnapshot.orderedSectionIds.length === 0 &&
  orderedSnapshot.nextSectionId === riftDefinition.activeSectionIds[0] &&
  orderedSnapshot.currentSectionKillCount === 2 &&
  orderedSnapshot.killsPerSection === 3 &&
  orderedSnapshot.progress === 2,
'the tracker should keep the active southwest pip at two of three until its quadrant quota is complete');
recordSectionDefeat(pushEngine, riftDefinition.activeSectionIds[0]);
orderedSnapshot = pushEngine.getMapMechanicSnapshot('endlessRift');
check(orderedSnapshot.orderedSectionIds.join(',') === riftDefinition.activeSectionIds[0] &&
  orderedSnapshot.nextSectionId === riftDefinition.activeSectionIds[1] &&
  orderedSnapshot.currentSectionKillCount === 0 &&
  orderedSnapshot.progress === 3,
'the route should advance to northwest only after three credited southwest defeats');

for (let cycle = 0; cycle < 3; cycle += 1) completeRiftRotation(pushEngine);
const prePush = pushEngine.getRiftSnapshot();
check(prePush.decisionPending === true &&
  prePush.rotationsThisTier === prePush.rotationsRequired &&
  !emptyBounty(prePush.unbankedBounty),
'three completed rotations should create a Push-or-Bank decision with an unbanked bounty');
const lockedMechanicBefore = pushEngine.getMapMechanicSnapshot('endlessRift');
const lockedBountyBefore = JSON.stringify(prePush.unbankedBounty);
const lockedDefeatResults = riftDefinition.activeSectionIds.flatMap((sectionId) =>
  Array.from({ length: riftDefinition.killsPerSection }, () => recordSectionDefeat(pushEngine, sectionId)));
const lockedMechanicAfter = pushEngine.getMapMechanicSnapshot('endlessRift');
const lockedRiftAfter = pushEngine.getRiftSnapshot();
check(lockedDefeatResults.every((result) => result === false) &&
  lockedMechanicAfter.completedCycles === lockedMechanicBefore.completedCycles &&
  lockedRiftAfter.rotationsThisTier === prePush.rotationsThisTier &&
  JSON.stringify(lockedRiftAfter.unbankedBounty) === lockedBountyBefore,
'the stabilized tier should freeze route credit and bounty until the player chooses Push or Bank');
const cachedAwayFromCore = pushEngine.getMapModifierSnapshot();
check(cachedAwayFromCore.rift.atCore === false &&
  cachedAwayFromCore.rift.canPush === false &&
  cachedAwayFromCore.rift.canBank === false,
'the cached Rift modifier snapshot should report Core actions unavailable while the player is away');
check(pushEngine.isPlayerAtRiftCore() === false &&
  pushEngine.pushRiftTier() === false &&
  pushEngine.getRiftSnapshot().tier === prePush.tier,
'Push should remain unavailable away from the central Rift Core');
placePlayerAtRiftCore(pushEngine);
const cachedAtCore = pushEngine.getMapModifierSnapshot();
check(cachedAtCore !== cachedAwayFromCore &&
  cachedAtCore.rift.atCore === true &&
  cachedAtCore.rift.canPush === true &&
  cachedAtCore.rift.canBank === true,
'entering the Rift Core should invalidate cached action flags and expose Push and Bank immediately');
pushEngine.state.player.grounded = false;
check(pushEngine.isPlayerAtRiftCore() === false,
  'stale Core platform IDs should not permit Push or Bank while the player is airborne');
pushEngine.state.player.grounded = true;
check(pushEngine.pushRiftTier() === true,
  'Push should succeed through the public API while the player is at the Core');
const postPush = pushEngine.getRiftSnapshot();
check(postPush.tier === prePush.tier + 1 &&
  postPush.bestTier >= postPush.tier &&
  postPush.score === 0 &&
  postPush.rotationsThisTier === 0 &&
  postPush.decisionPending === false,
'Push should advance one tier and reset only the completed tier progress');
check(postPush.unbankedBounty.currency === prePush.unbankedBounty.currency &&
  JSON.stringify(postPush.unbankedBounty.materials) === JSON.stringify(prePush.unbankedBounty.materials),
'Push should retain the full unbanked bounty for the risk-reward run');

const failCheckpoint = Math.max(postPush.bankedTier, postPush.checkpointTier);
check(pushEngine.failRiftRun() === true,
  'the public fail API should resolve an active pushed run');
const postFailure = pushEngine.getRiftSnapshot();
check(postFailure.tier === failCheckpoint &&
  postFailure.rotationsThisTier === 0 &&
  postFailure.decisionPending === false &&
  emptyBounty(postFailure.unbankedBounty),
'failure should return to the best banked checkpoint and clear unbanked run progress');

const bankEngine = createRiftEngine();
for (let cycle = 0; cycle < 3; cycle += 1) completeRiftRotation(bankEngine);
placePlayerAtRiftCore(bankEngine);
const preBank = bankEngine.getRiftSnapshot();
const currencyBeforeBank = bankEngine.state.player.currency;
const materialsBeforeBank = Object.assign({}, bankEngine.state.materials);
check(bankEngine.bankRiftRun() === true,
  'Bank should succeed through the public API while the player is at the Core');
const postBank = bankEngine.getRiftSnapshot();
check(bankEngine.state.mapId === 'eclipseFrontier' &&
  bankEngine.state.player.currency === currencyBeforeBank + preBank.unbankedBounty.currency &&
  Object.entries(preBank.unbankedBounty.materials).every(([materialId, amount]) =>
    Number(bankEngine.state.materials[materialId] || 0) ===
      Number(materialsBeforeBank[materialId] || 0) + Number(amount || 0)
  ),
'Bank should award the full bounty and return the player to Eclipse Frontier');
check(emptyBounty(postBank.unbankedBounty) &&
  postBank.bankedTier >= preBank.tier &&
  postBank.rotationsThisTier === 0 &&
  postBank.decisionPending === false,
'Bank should clear risk-state while preserving the secured tier');

function createBankReadyEngine() {
  const engine = createRiftEngine({
    tier: 1,
    bestTier: 1,
    bankedTier: 1,
    checkpointTier: 1,
    score: 500,
    rotationsThisTier: 3,
    decisionPending: true,
    unbankedBounty: {
      currency: 360,
      materials: { riftSplinter: 3 },
      consumables: {}
    }
  });
  placePlayerAtRiftCore(engine);
  return engine;
}

const invalidDestinationEngine = createBankReadyEngine();
const invalidDestinationCurrency = invalidDestinationEngine.state.player.currency;
const invalidDestinationBounty = invalidDestinationEngine.getRiftSnapshot().unbankedBounty;
check(invalidDestinationEngine.bankRiftRun({
  destinationMapId: 'not-a-map',
  silent: true
}) === false &&
  invalidDestinationEngine.state.mapId === 'endlessRift' &&
  invalidDestinationEngine.state.player.currency === invalidDestinationCurrency &&
  invalidDestinationEngine.getRiftSnapshot().decisionPending === true &&
  JSON.stringify(invalidDestinationEngine.getRiftSnapshot().unbankedBounty) === JSON.stringify(invalidDestinationBounty),
'a failed destination preflight should leave the complete bank-ready run untouched');

const fullInventoryEngine = createBankReadyEngine();
const etcCapacity = fullInventoryEngine.getInventoryCapacity('etc');
starfallData.MATERIAL_ITEMS
  .filter((item) => item.id !== 'riftSplinter')
  .slice(0, etcCapacity)
  .forEach((item) => {
    fullInventoryEngine.state.materials[item.id] = 1;
  });
fullInventoryEngine.reconcileInventorySlotOrder('etc');
check(fullInventoryEngine.getInventoryUsedSlots('etc') === etcCapacity &&
  fullInventoryEngine.canAddStackableInventoryItem('etc', 'riftSplinter', 3) === false,
'the full-inventory fixture should block the pending Rift Splinter stack');
const fullInventoryCurrency = fullInventoryEngine.state.player.currency;
const fullInventoryBounty = fullInventoryEngine.getRiftSnapshot().unbankedBounty;
check(fullInventoryEngine.bankRiftRun({ silent: true }) === false &&
  fullInventoryEngine.state.mapId === 'endlessRift' &&
  fullInventoryEngine.state.player.currency === fullInventoryCurrency &&
  fullInventoryEngine.getRiftSnapshot().decisionPending === true &&
  JSON.stringify(fullInventoryEngine.getRiftSnapshot().unbankedBounty) === JSON.stringify(fullInventoryBounty),
'Bank should be all-or-nothing when the player lacks inventory room for the bounty');

const retreatEngine = createRiftEngine();
completeRiftRotation(retreatEngine);
const retreatCurrency = retreatEngine.state.player.currency;
const retreatBounty = retreatEngine.getRiftSnapshot().unbankedBounty;
check(!emptyBounty(retreatBounty) &&
  retreatEngine.changeMap('eclipseFrontier', { silent: true }) === true &&
  retreatEngine.state.player.currency === retreatCurrency &&
  emptyBounty(retreatEngine.getRiftSnapshot().unbankedBounty),
'leaving through the return portal without a Core Bank should forfeit, not auto-award, the unbanked bounty');

const saveEngine = createRiftEngine();
completeRiftRotation(saveEngine);
let midRoute = saveEngine.getMapMechanicSnapshot('endlessRift');
recordSectionDefeat(saveEngine, midRoute.nextSectionId);
midRoute = saveEngine.getMapMechanicSnapshot('endlessRift');
const preSaveRift = saveEngine.getRiftSnapshot();
const payload = saveEngine.serialize();
const restoredEngine = createProjectStarfallEngine(null, starfallData);
check(restoredEngine.restore(payload),
  'the public restore API should load an active Rift run');
const restoredRift = restoredEngine.getRiftSnapshot();
const restoredRoute = restoredEngine.getMapMechanicSnapshot('endlessRift');
check(restoredEngine.state.mapId === 'endlessRift' &&
  [
    'tier',
    'bestTier',
    'bankedTier',
    'checkpointTier',
    'score',
    'nextTierScore',
    'rotationsThisTier',
    'rotationsRequired',
    'decisionPending'
  ].every((key) => restoredRift[key] === preSaveRift[key]) &&
  JSON.stringify(restoredRift.unbankedBounty) === JSON.stringify(preSaveRift.unbankedBounty) &&
  restoredRift.mutationIds.join(',') === preSaveRift.mutationIds.join(','),
'save and restore should preserve the complete active Rift tier, mutation, and bounty state');
check(restoredRoute.completedCycles === midRoute.completedCycles &&
  restoredRoute.progress === midRoute.progress &&
  restoredRoute.nextSectionId === midRoute.nextSectionId &&
  restoredRoute.currentSectionKillCount === midRoute.currentSectionKillCount &&
  restoredRoute.killsPerSection === midRoute.killsPerSection &&
  restoredRoute.orderedSectionIds.join(',') === midRoute.orderedSectionIds.join(','),
'save and restore should preserve the player mid-way through an ordered quadrant rotation');

console.log(`Project Starfall Endless Rift route checks passed: ${checks}`);
