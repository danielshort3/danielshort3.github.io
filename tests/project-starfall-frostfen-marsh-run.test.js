'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const MapBuilders = require('../js/games/project-starfall/data/map-builders.js');
const MapLayouts = require('../js/games/project-starfall/data/map-layouts.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { createMapBalanceReport } = require('./project-starfall-balance-harness.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

function platformRight(platform) {
  return Number(platform.x || 0) + Number(platform.w || 0);
}

function rectsOverlap(a, b) {
  return a.x < b.x + b.w &&
    a.x + a.w > b.x &&
    a.y < b.y + b.h &&
    a.y + a.h > b.y;
}

function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'frostfenOutskirts');
  assert(map, 'Frostfen Outskirts should remain in the published map catalog');
  assert.strictEqual(map.platforms[0].w, 5600, 'the Frozen Marsh Run should use its compact 5600px runtime width');
  assert.strictEqual(map.compactWorldWidth, 5600);
  assert.strictEqual(map.geometryGenerator, 'frostfenMarshRunV1');
  assert.strictEqual(map.movementProfile, 'ice', 'the authored route should preserve the live ice movement model');
  assert.strictEqual(map.designIntent.implementationStatus, 'frozen-marsh-run-v1');
  assert.strictEqual(map.designIntent.priorityRedesign, false);
  assert.strictEqual(map.designIntent.runtimeBoundsValidated, true);
  assert(MapLayouts.PRIORITY_FIELD_LAYOUT_IDS.includes('frostfenOutskirts'));
  assert.strictEqual(typeof MapBuilders.makeFrostfenMarshRunPlatforms, 'function');
  assert.strictEqual(typeof MapBuilders.makeFrostfenMarshRunClimbables, 'function');

  const sections = map.fieldComposition.routeSections;
  assert.deepStrictEqual(
    sections.map((section) => [section.label, section.x, section.w]),
    [
      ['Frozen Marsh', 0, 1500],
      ['Rimeglass Shelf', 1500, 2200],
      ['Oracle Grove', 3700, 1900]
    ]
  );
  sections.forEach((section, index) => {
    assert.strictEqual(section.x, index ? sections[index - 1].x + sections[index - 1].w : 0,
      `${section.label} should begin where the previous route beat ends`);
  });
  assert.strictEqual(sections[sections.length - 1].x + sections[sections.length - 1].w, map.platforms[0].w);

  const landmarkBands = map.fieldComposition.landmarkBands;
  assert.deepStrictEqual(
    landmarkBands.map((landmark) => landmark.label),
    ['Frozen Signal Wreck', 'Rimeglass Shelf', 'Oracle Bloom Grove']
  );
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
    assert(platformRight(platform) <= map.platforms[0].w, `${platform.id} should remain inside the runtime world`);
  });

  const slopes = map.platforms.filter((platform) => platform.shape === 'slope');
  assert.strictEqual(slopes.length, 3, 'each route beat should have one deliberate ice-run entry slope');
  map.spawnSections.forEach((section) => {
    assert.strictEqual(slopes.filter((platform) => platform.sectionId === section.id).length, 1,
      `${section.label} should own exactly one authored slope`);
    const broadFlats = map.platforms.filter((platform) =>
      platform.sectionId === section.id && platform.shape !== 'slope' && platform.w >= 640
    );
    assert(broadFlats.length >= 3, `${section.label} should expose at least three readable combat surfaces`);
  });

  const normalizedSectionFingerprints = map.spawnSections.map((section) => map.platforms
    .filter((platform) => platform.sectionId === section.id && platform.shape !== 'slope')
    .map((platform) => [platform.x - section.x, platform.y, platform.w, platform.terrainVisual.kind].join(':'))
    .sort()
    .join('|'));
  assert.strictEqual(new Set(normalizedSectionFingerprints).size, normalizedSectionFingerprints.length,
    'Frozen Marsh route beats should not be translated copies of one platform cluster');

  map.spawnSections.forEach((section) => {
    const points = map.spawnPoints.filter((point) => point.sectionId === section.id);
    assert(points.length >= 3, `${section.label} should own multiple spawn surfaces`);
  });
  map.spawnPoints.forEach((point) => {
    const platform = map.platforms[point.platformIndex];
    const section = sectionById.get(point.sectionId);
    assert(platform && section, `${point.id} should resolve to a platform and route section`);
    assert.strictEqual(point.platformId, platform.id);
    assert(point.x >= platform.x && point.x <= platformRight(platform), `${point.id} should sit on its platform`);
    assert(point.x >= section.x && point.x <= section.x + section.w, `${point.id} should sit inside its route section`);
  });

  assert(map.climbables.length >= 5 && map.climbables.length <= 7,
    'the route should use five to seven purposeful frost ladders instead of a repeated ladder grid');
  [
    'frostfenOutskirts_frost_ladder_marsh_windbreak',
    'frostfenOutskirts_frost_ladder_rimeglass_shelf',
    'frostfenOutskirts_frost_ladder_oracle_bloom'
  ].forEach((id) => assert(map.climbables.some((climbable) => climbable.id === id), `${id} should be authored`));

  const dropPerch = map.platforms.find((platform) => platform.id === 'frostfen_oracle_bloom_perch');
  const dropTarget = map.platforms.find((platform) => platform.id === dropPerch.dropTargetPlatformId);
  assert(dropPerch && dropTarget && dropPerch.dropShortcut === true);
  assert(Math.min(platformRight(dropPerch), platformRight(dropTarget)) - Math.max(dropPerch.x, dropTarget.x) >= 600,
    'the oracle perch should overlap a broad, readable drop-reset runway');

  const outgoingPortal = map.portals.find((portal) => portal.id === 'frostfen_glacier');
  assert(outgoingPortal, 'the authored route should preserve its Glacier Spine exit');
  assert.strictEqual(map.platforms[outgoingPortal.platformIndex].id, 'frostfen_oracle_exit_shelf');
  assert(outgoingPortal.x >= map.platforms[outgoingPortal.platformIndex].x &&
    outgoingPortal.x <= platformRight(map.platforms[outgoingPortal.platformIndex]),
  'the Glacier Spine relay should sit on its authored exit shelf');
  assert(outgoingPortal.portalStyle.includes('rime'), 'the exit should use Frostfen-specific relay fiction');

  const transitionEngine = createProjectStarfallEngine(null, Data);
  assert.strictEqual(transitionEngine.chooseClass('fighter'), true);
  assert.strictEqual(transitionEngine.changeMap('frostfenOutskirts'), true);
  assert.strictEqual(transitionEngine.runtime.id, 'frostfenOutskirts');
  assert.strictEqual(transitionEngine.runtime.worldWidth, 5760,
    'the live engine should add only its fixed 160px safety pad to the compact Frostfen route');
  const tracker = transitionEngine.runtime.questNpcs.find((npc) => npc.id === 'frostfen_tracker');
  assert(tracker, 'the Frostfen Tracker should resolve into the live map runtime');
  assert.strictEqual(tracker.platformIndex, 2);
  assert.strictEqual(tracker.platformId, 'frostfen_marsh_runway',
    'the tracker should stand on the authored runway instead of being sliced by it from ground level');
  const trackerPlatform = transitionEngine.runtime.platforms[tracker.platformIndex];
  assert.strictEqual(tracker.y + tracker.h, trackerPlatform.y,
    'the tracker feet should align with the flat runway surface');
  const trackerIntrusions = transitionEngine.runtime.platforms.filter((platform) =>
    platform.index !== tracker.platformIndex && rectsOverlap(tracker, platform)
  );
  assert.deepStrictEqual(trackerIntrusions.map((platform) => platform.id), [],
    'the tracker runtime rect should not overlap any non-owning platform');
  transitionEngine.state.routeProgress.frostfen = {
    killsByMap: {
      ashglassPass: 34,
      frostfenOutskirts: 32
    }
  };
  assert.strictEqual(transitionEngine.getPortalBlockReason(
    transitionEngine.runtime.portals.find((portal) => portal.id === 'frostfen_glacier')
  ), '', 'the Glacier Spine relay should unlock after the authored field goals are complete');
  assert.strictEqual(transitionEngine.usePortal('frostfen_glacier'), true,
    'the live Frostfen exit portal should complete a real engine map transition');
  assert.strictEqual(transitionEngine.state.mapId, 'glacierSpine');
  assert.strictEqual(transitionEngine.runtime.id, 'glacierSpine');

  const validation = validateMap(map);
  assert.deepStrictEqual(validation.issues, [], `map validation failed: ${validation.issues.join('; ')}`);
  assert.deepStrictEqual(validation.warnings, [], `map warnings remained: ${validation.warnings.join('; ')}`);

  const balanceReport = createMapBalanceReport(Data, createProjectStarfallEngine, { classIds: ['fighter'] });
  const tuning = balanceReport.mapTuning.maps.find((entry) => entry.mapId === map.id);
  assert(tuning && tuning.routeViable, 'the Frozen Marsh Run should preserve a viable repeatable route');
  assert.strictEqual(tuning.emptySectionCount, 0, 'all three authored route beats should be populated');
  assert.strictEqual(tuning.activeSpawnSectionCount, 3);
  assert(tuning.metrics.travelSharePercent <= balanceReport.mapTuning.warningThresholds.travelSharePercent,
    `Frostfen travel share should clear the guardrail: ${tuning.metrics.travelSharePercent}%`);
  assert(!tuning.warningIds.includes('travelShareHigh'),
    'the compact authored route should not retain the old high-travel warning');

  process.stdout.write('Project Starfall Frostfen Frozen Marsh Run tests passed.\n');
}

main();
