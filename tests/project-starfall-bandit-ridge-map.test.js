'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const MapBuilders = require('../js/games/project-starfall/data/map-builders.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

function platformRight(platform) {
  return Number(platform.x || 0) + Number(platform.w || 0);
}

function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'banditRidgeCamp');
  assert(map, 'Bandit Ridge Camp should remain in the published map catalog');
  assert.strictEqual(map.platforms[0].w, 5400, 'the authored pass should preserve the 5400px world width');
  assert.strictEqual(map.geometryGenerator, 'banditRidgeAuthoredV2');
  assert.strictEqual(map.designIntent.implementationStatus, 'authored-ridge-v2');
  assert.strictEqual(map.designIntent.priorityRedesign, false);
  assert.strictEqual(typeof MapBuilders.makeBanditRidgeCampPlatforms, 'function');
  assert.strictEqual(typeof MapBuilders.makeBanditRidgeCampClimbables, 'function');

  const sections = map.fieldComposition.routeSections;
  assert.deepStrictEqual(
    sections.map((section) => [section.label, section.x, section.w]),
    [
      ['Lower Cutter Lane', 0, 1250],
      ['Middle Thrower Camp', 1250, 1450],
      ['High Rope Bridge', 2700, 1350],
      ['Campfire Regroup', 4050, 1350]
    ]
  );
  sections.forEach((section, index) => {
    assert.strictEqual(section.x, index ? sections[index - 1].x + sections[index - 1].w : 0,
      `${section.label} should begin where the previous route beat ends`);
  });
  assert.strictEqual(sections[sections.length - 1].x + sections[sections.length - 1].w, 5400);
  assert.strictEqual(sections[3].encounterMode, 'safe-regroup');
  assert.strictEqual(sections[3].routinePopulation, 0);
  assert(sections[3].safeRadius >= 200, 'the campfire should reserve a meaningful recovery radius');

  const landmarkBands = map.fieldComposition.landmarkBands;
  assert.strictEqual(landmarkBands.length, sections.length);
  landmarkBands.forEach((landmark, index) => {
    const section = sections[index];
    assert(landmark.x >= section.x && landmark.x + landmark.w <= section.x + section.w,
      `${landmark.label} should stay inside its authored route beat`);
  });

  const platformIds = map.platforms.map((platform) => platform.id);
  assert.strictEqual(new Set(platformIds).size, platformIds.length, 'authored platform ids should remain unique');
  const sectionById = new Map(map.spawnSections.map((section) => [section.id, section]));
  map.platforms.slice(1).forEach((platform) => {
    const section = sectionById.get(platform.sectionId);
    assert(section, `${platform.id} should declare its owning section`);
    assert(platform.x >= section.x && platformRight(platform) <= section.x + section.w,
      `${platform.id} should not cross a route-section boundary`);
  });

  const slopes = map.platforms.filter((platform) => platform.shape === 'slope');
  assert.strictEqual(slopes.length, 4, 'each route beat should use one deliberate slope');
  map.spawnSections.forEach((section) => {
    assert.strictEqual(slopes.filter((platform) => platform.sectionId === section.id).length, 1,
      `${section.label} should own exactly one authored slope`);
  });
  const broadCombatPlatforms = map.platforms.slice(1).filter((platform) =>
    platform.shape !== 'slope' && platform.w >= 640 && platform.sectionId !== 'banditRidgeCamp_campfire_regroup'
  );
  assert.strictEqual(broadCombatPlatforms.length, 8);
  ['banditRidgeCamp_lower_cutter_lane', 'banditRidgeCamp_middle_thrower_camp', 'banditRidgeCamp_high_rope_bridge']
    .forEach((sectionId) => {
      const flats = broadCombatPlatforms.filter((platform) => platform.sectionId === sectionId);
      assert(flats.length >= 2 && flats.some((platform) => platform.w >= 700),
        `${sectionId} should provide broad, readable combat footing`);
    });

  const normalizedSectionFingerprints = map.spawnSections.slice(0, 3).map((section) => map.platforms
    .filter((platform) => platform.sectionId === section.id && platform.shape !== 'slope')
    .map((platform) => [platform.x - section.x, platform.y, platform.w, platform.terrainVisual.kind].join(':'))
    .sort()
    .join('|'));
  assert.strictEqual(new Set(normalizedSectionFingerprints).size, normalizedSectionFingerprints.length,
    'combat beats should not be translated copies of one platform cluster');

  const spawnGroups = map.spawnGroups;
  assert.deepStrictEqual(
    spawnGroups.map((group) => group.sectionId),
    [
      'banditRidgeCamp_lower_cutter_lane',
      'banditRidgeCamp_middle_thrower_camp',
      'banditRidgeCamp_high_rope_bridge'
    ]
  );
  assert.strictEqual(spawnGroups.reduce((total, group) => total + group.population, 0), 30,
    'three combat territories should preserve the prior 30-enemy progression budget');
  assert.strictEqual(
    spawnGroups.filter((group) => group.sectionId === 'banditRidgeCamp_campfire_regroup')
      .reduce((total, group) => total + group.population, 0),
    0,
    'the campfire regroup should have no routine spawn population'
  );
  const ownedPlatformIds = new Set();
  spawnGroups.forEach((group) => {
    assert(group.platformIds.length >= 2, `${group.label} should own multiple deliberate spawn surfaces`);
    group.platformIds.forEach((platformId) => {
      assert(!ownedPlatformIds.has(platformId), `${platformId} should not be shared by two spawn groups`);
      ownedPlatformIds.add(platformId);
      const platform = map.platforms.find((candidate) => candidate.id === platformId);
      assert(platform, `${group.label} should reference a real authored platform`);
      assert.strictEqual(platform.sectionId, group.sectionId,
        `${platformId} should stay inside ${group.label}'s territory`);
      assert(map.spawnPoints.some((point) => point.platformId === platformId),
        `${platformId} should expose at least one valid spawn point`);
    });
  });

  const entrySpawnXs = map.spawnPoints
    .filter((point) => point.sectionId === 'banditRidgeCamp_lower_cutter_lane')
    .map((point) => point.x);
  assert(entrySpawnXs.length && Math.min(...entrySpawnXs) >= 350,
    'routine enemies should stay clear of the safe arrival pad');

  const ropeBridge = map.platforms.find((platform) => platform.id === 'bandit_ridge_camp_rope_bridge');
  const bridgeReturn = map.platforms.find((platform) => platform.id === 'bandit_ridge_camp_bridge_return_lane');
  assert(ropeBridge && bridgeReturn && ropeBridge.dropShortcut === true);
  assert.strictEqual(ropeBridge.dropTargetPlatformId, bridgeReturn.id);
  assert(Math.min(platformRight(ropeBridge), platformRight(bridgeReturn)) - Math.max(ropeBridge.x, bridgeReturn.x) >= 600,
    'the high bridge should overlap a broad, readable drop-return lane');
  assert(['banditRidgeCamp_rope_bridge_west', 'banditRidgeCamp_rope_bridge_east']
    .every((id) => map.climbables.some((climbable) => climbable.id === id)),
  'the rope bridge should have two authored access routes');

  const validation = validateMap(map);
  assert.deepStrictEqual(validation.issues, [], `map validation failed: ${validation.issues.join('; ')}`);

  process.stdout.write('Project Starfall Bandit Ridge map tests passed.\n');
}

main();
