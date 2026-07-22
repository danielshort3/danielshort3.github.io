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

function canonicalTopology(map) {
  const ground = map.platforms[0];
  return map.platforms.slice(1)
    .map((platform) => [
      Number((platform.x / ground.w).toFixed(3)),
      Number(((ground.y - Math.min(platform.y, platform.y2 == null ? platform.y : platform.y2)) / 820).toFixed(3)),
      Number((platform.w / ground.w).toFixed(3)),
      platform.shape || 'flat',
      platform.terrainVisual && platform.terrainVisual.kind || ''
    ].join(':'))
    .sort()
    .join('|');
}

function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'rustcoilRuins');
  assert(map, 'Rustcoil Ruins should remain in the published map catalog');
  assert.strictEqual(map.platforms[0].w, 5200, 'the Fractured Orrery Circuit should use its compact 5200px width');
  assert.strictEqual(map.compactWorldWidth, 5200);
  assert.strictEqual(map.geometryGenerator, 'rustcoilOrreryCircuitV1');
  assert.strictEqual(map.layoutStyle, 'industrialStack');
  assert.strictEqual(map.designIntent.implementationStatus, 'fractured-orrery-circuit-v1');
  assert.strictEqual(map.designIntent.runtimeBoundsValidated, true);
  assert.strictEqual(map.routeStage, 'Fractured Orrery');
  assert.strictEqual(map.mapRoadName, 'Orrery Circuit');
  assert.strictEqual(map.landmark, 'warden starcoil');
  assert(map.purpose.includes('celestial machine'),
    'player-facing map copy should describe the authored identity instead of generic ruins');
  assert.strictEqual(map.designIntent.priorityRedesign, true,
    'the early-game circuit should retain active priority-polish weighting');
  assert(MapLayouts.PRIORITY_FIELD_LAYOUT_IDS.includes('rustcoilRuins'));
  assert.strictEqual(typeof MapBuilders.makeRustcoilOrreryCircuitPlatforms, 'function');
  assert.strictEqual(typeof MapBuilders.makeRustcoilOrreryCircuitClimbables, 'function');

  const sections = map.fieldComposition.routeSections;
  assert.deepStrictEqual(
    sections.map((section) => [section.label, section.x, section.w]),
    [
      ['Surveyor Yard', 0, 1500],
      ['Coil Switchworks', 1500, 1900],
      ['Warden Gearwell', 3400, 1800]
    ]
  );
  sections.forEach((section, index) => {
    assert.strictEqual(section.x, index ? sections[index - 1].x + sections[index - 1].w : 0,
      section.label + ' should begin where the previous route beat ends');
  });
  assert.strictEqual(sections[sections.length - 1].x + sections[sections.length - 1].w, map.platforms[0].w);
  assert.deepStrictEqual(
    map.fieldComposition.landmarkBands.map((landmark) => landmark.label),
    ['Fractured Orrery Yard', 'Coil Switchworks', 'Warden Starcoil']
  );

  const sectionById = new Map(map.spawnSections.map((section) => [section.id, section]));
  const platformIds = map.platforms.map((platform) => platform.id);
  assert.strictEqual(new Set(platformIds).size, platformIds.length, 'authored platform ids should remain unique');
  map.platforms.slice(1).forEach((platform) => {
    const section = sectionById.get(platform.sectionId);
    assert(section, platform.id + ' should declare its owning section');
    assert(platform.x >= section.x && platformRight(platform) <= section.x + section.w,
      platform.id + ' should stay inside ' + section.label);
    assert(platformRight(platform) <= map.platforms[0].w, platform.id + ' should remain inside the authored world');
  });

  const slopes = map.platforms.filter((platform) => platform.shape === 'slope');
  assert.strictEqual(slopes.length, 3, 'each circuit territory should have one deliberate ground entry');
  map.spawnSections.forEach((section) => {
    assert.strictEqual(slopes.filter((platform) => platform.sectionId === section.id).length, 1,
      section.label + ' should own exactly one authored slope');
    const broadFlats = map.platforms.filter((platform) =>
      platform.sectionId === section.id &&
      platform.shape !== 'slope' &&
      platform.terrainVisual.kind !== 'connector' &&
      platform.w >= 640
    );
    assert(broadFlats.length >= 3, section.label + ' should expose at least three readable combat surfaces');
  });

  const sectionFingerprints = map.spawnSections.map((section) => map.platforms
    .filter((platform) => platform.sectionId === section.id && platform.shape !== 'slope')
    .map((platform) => [
      platform.x - section.x,
      platform.y,
      platform.w,
      platform.terrainVisual.kind,
      platform.routeRole
    ].join(':'))
    .sort()
    .join('|'));
  assert.strictEqual(new Set(sectionFingerprints).size, sectionFingerprints.length,
    'the three territories should not be translated copies of one platform cluster');

  const rustcoilTopology = canonicalTopology(map);
  ['glacierSpine', 'astralArchive', 'eclipseFrontier'].forEach((mapId) => {
    const sibling = Data.MAPS.find((candidate) => candidate.id === mapId);
    assert(sibling, mapId + ' should remain available for topology comparison');
    assert.notStrictEqual(rustcoilTopology, canonicalTopology(sibling),
      'Rustcoil should not reuse ' + mapId + ' generic canonical topology');
  });

  map.spawnSections.forEach((section) => {
    const points = map.spawnPoints.filter((point) => point.sectionId === section.id);
    assert(points.length >= 3, section.label + ' should own at least three spawn placements');
  });
  map.spawnPoints.forEach((point) => {
    const platform = map.platforms[point.platformIndex];
    const section = sectionById.get(point.sectionId);
    assert(platform && section, point.id + ' should resolve to a platform and route section');
    assert.strictEqual(point.platformId, platform.id, point.id + ' should retain its stable platform id');
    assert(point.x >= platform.x && point.x <= platformRight(platform), point.id + ' should sit on its platform');
    assert(point.x >= section.x && point.x <= section.x + section.w, point.id + ' should sit inside its territory');
  });

  const climbableIds = [
    'rustcoilRuins_lift_yard_service',
    'rustcoilRuins_lift_yard_orrery',
    'rustcoilRuins_lift_switchworks_west',
    'rustcoilRuins_lift_switchworks_return',
    'rustcoilRuins_lift_switchworks_east',
    'rustcoilRuins_lift_warden_ring',
    'rustcoilRuins_lift_warden_relay',
    'rustcoilRuins_lift_warden_starcoil'
  ];
  assert.strictEqual(map.climbables.length, climbableIds.length,
    'the circuit should use eight purposeful lifts instead of a repeated ladder grid');
  climbableIds.forEach((id) => assert(map.climbables.some((climbable) => climbable.id === id), id + ' should be authored'));

  const dropPerch = map.platforms.find((platform) => platform.id === 'rustcoil_warden_starcoil_perch');
  const dropTarget = map.platforms.find((platform) => platform.id === dropPerch.dropTargetPlatformId);
  assert(dropPerch && dropTarget && dropPerch.dropShortcut === true);
  assert.strictEqual(dropTarget.id, 'rustcoil_warden_return_belt');
  assert(Math.min(platformRight(dropPerch), platformRight(dropTarget)) - Math.max(dropPerch.x, dropTarget.x) >= 700,
    'the starcoil perch should overlap a broad, readable return belt');

  const serviceBridge = map.platforms.find((platform) =>
    platform.id === 'rustcoil_warden_starcoil_service_bridge');
  assert(serviceBridge && serviceBridge.w === 600);
  assert.strictEqual(serviceBridge.terrainVisual.longSpan, true,
    'the only long connector should explicitly preserve its thin service-bridge silhouette');

  const runtimeEngine = createProjectStarfallEngine(null, Data);
  assert.strictEqual(runtimeEngine.chooseClass('fighter'), true);
  assert.strictEqual(runtimeEngine.changeMap('rustcoilRuins'), true);
  assert.strictEqual(runtimeEngine.runtime.id, 'rustcoilRuins');
  assert.strictEqual(runtimeEngine.runtime.worldWidth, 5360,
    'the live engine should add only its fixed 160px safety pad to the compact circuit');

  const surveyor = runtimeEngine.runtime.questNpcs.find((npc) => npc.id === 'ruins_surveyor');
  assert(surveyor, 'the Rustcoil Surveyor should remain in the live map runtime');
  assert.deepStrictEqual(surveyor.questIds, ['rustcoil_reclamation']);
  assert.strictEqual(surveyor.platformIndex, 0);
  assert.strictEqual(surveyor.platformId, 'rustcoil_orrery_circuit_ground');
  const surveyorPlatform = runtimeEngine.runtime.platforms[surveyor.platformIndex];
  assert.strictEqual(surveyor.y + surveyor.h, surveyorPlatform.y,
    'the Surveyor feet should align with the arrival ground');
  const surveyorIntrusions = runtimeEngine.runtime.platforms.filter((platform) =>
    platform.index !== surveyor.platformIndex && rectsOverlap(surveyor, platform)
  );
  assert.deepStrictEqual(surveyorIntrusions.map((platform) => platform.id), [],
    'the Surveyor arrival should remain clear of non-owning platforms');

  const returnPortal = runtimeEngine.runtime.portals.find((portal) => portal.id === 'rustcoil_outpost_return');
  assert(returnPortal, 'the Outpost Return portal should remain in the live map runtime');
  assert.strictEqual(returnPortal.destinationMapId, 'rustcoilOutpost');
  assert.strictEqual(returnPortal.platformId, 'rustcoil_orrery_circuit_ground');
  assert.strictEqual(returnPortal.portalStyle, 'gear gate');
  assert.strictEqual(runtimeEngine.usePortal(returnPortal.id), true,
    'the preserved return portal should complete a real engine transition');
  assert.strictEqual(runtimeEngine.state.mapId, 'rustcoilOutpost');

  const validation = validateMap(map);
  assert.deepStrictEqual(validation.issues, [], 'map validation failed: ' + validation.issues.join('; '));
  assert.deepStrictEqual(validation.warnings, [], 'map warnings remained: ' + validation.warnings.join('; '));

  const balanceReport = createMapBalanceReport(Data, createProjectStarfallEngine, {
    classIds: ['fighter', 'mage', 'archer']
  });
  const tuning = balanceReport.mapTuning.maps.find((entry) => entry.mapId === map.id);
  assert(tuning && tuning.routeViable, 'the Fractured Orrery Circuit should expose a viable repeatable route');
  assert.deepStrictEqual(tuning.routeIssueIds, []);
  assert.strictEqual(tuning.metrics.platformCoverage, 1);
  assert.strictEqual(tuning.activeSpawnSectionCount, 3);
  assert.strictEqual(tuning.emptySectionCount, 0);
  assert(tuning.metrics.travelSharePercent <= 26,
    'travel share exceeded the authored target: ' + tuning.metrics.travelSharePercent + '%');
  assert(tuning.metrics.nonCombatTraversalPercent <= 25,
    'noncombat traversal exceeded the authored target: ' + tuning.metrics.nonCombatTraversalPercent + '%');
  assert(tuning.metrics.abandonmentRiskIndex <= 12,
    'abandonment risk exceeded the authored target: ' + tuning.metrics.abandonmentRiskIndex);
  assert(tuning.metrics.repeatVisitationIndex >= 60,
    'repeat visitation missed the authored target: ' + tuning.metrics.repeatVisitationIndex);
  assert(tuning.metrics.classPerformanceSpreadPercent <= 20,
    'class spread exceeded the authored target: ' + tuning.metrics.classPerformanceSpreadPercent + '%');
  assert(tuning.metrics.partyOverlapPercent <= 20,
    'party overlap exceeded the authored target: ' + tuning.metrics.partyOverlapPercent + '%');
  assert.deepStrictEqual(tuning.warningIds, []);

  process.stdout.write('Project Starfall Rustcoil Fractured Orrery Circuit tests passed.\n');
}

main();
