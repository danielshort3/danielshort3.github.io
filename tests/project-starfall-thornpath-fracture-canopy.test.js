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

function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'thornpathThicket');
  assert(map, 'Thornpath Thicket should remain in the published map catalog');
  assert.strictEqual(map.platforms[0].w, 5200, 'the fracture canopy should use its actual 5200px runtime width');
  assert.strictEqual(map.geometryGenerator, 'thornpathFractureCanopyV1');
  assert.strictEqual(map.designIntent.implementationStatus, 'fracture-canopy-v1');
  assert.strictEqual(map.designIntent.priorityRedesign, false);
  assert.strictEqual(map.designIntent.runtimeBoundsValidated, true);
  assert(MapLayouts.PRIORITY_FIELD_LAYOUT_IDS.includes('thornpathThicket'));
  assert.strictEqual(typeof MapBuilders.makeThornpathFractureCanopyPlatforms, 'function');
  assert.strictEqual(typeof MapBuilders.makeThornpathFractureCanopyClimbables, 'function');

  const sections = map.fieldComposition.routeSections;
  assert.deepStrictEqual(
    sections.map((section) => [section.label, section.x, section.w]),
    [
      ['Meadow Return', 0, 1250],
      ['Fracture Canopy', 1250, 2200],
      ['Observatory Fork', 3450, 1750]
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
    ['Starstone Rootfall', 'Suspended Relay', 'Observatory Fork']
  );
  landmarkBands.forEach((landmark, index) => {
    const section = sections[index];
    assert(landmark.x >= section.x && landmark.x + landmark.w <= section.x + section.w,
      `${landmark.label} should stay inside its named route beat`);
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
  assert.strictEqual(slopes.length, 3, 'each route beat should use one deliberate entry slope');
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
    'route beats should not be translated copies of one platform cluster');

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

  const dropPerch = map.platforms.find((platform) => platform.id === 'thornpath_relay_shard_perch');
  const dropTarget = map.platforms.find((platform) => platform.id === dropPerch.dropTargetPlatformId);
  assert(dropPerch && dropTarget && dropPerch.dropShortcut === true);
  assert(Math.min(platformRight(dropPerch), platformRight(dropTarget)) - Math.max(dropPerch.x, dropTarget.x) >= 200,
    'the shard perch should overlap its readable drop-reset lane');
  [
    'thornpathThicket_vine_relay_perch',
    'thornpathThicket_vine_relay_perch_flank',
    'thornpathThicket_vine_ridge_branch',
    'thornpathThicket_vine_observatory_branch'
  ].forEach((id) => assert(map.climbables.some((climbable) => climbable.id === id), `${id} should be authored`));
  assert.strictEqual(map.climbables.length, 8,
    'the canopy should keep purposeful transfers without duplicating the three entry slopes or optional beacon drop');
  [
    'thornpathThicket_vine_rootfall_entry',
    'thornpathThicket_vine_relay_entry',
    'thornpathThicket_vine_fork_entry',
    'thornpathThicket_vine_fork_beacon'
  ].forEach((id) => assert(!map.climbables.some((climbable) => climbable.id === id), `${id} should remain removed`));

  const outgoingPortals = map.portals.filter((portal) => !portal.returnPortal);
  assert.deepStrictEqual(outgoingPortals.map((portal) => portal.id), ['thornpath_bandit', 'thornpath_rustcoil_outpost']);
  outgoingPortals.forEach((portal) => {
    const platform = map.platforms[portal.platformIndex];
    assert(platform, `${portal.id} should reference a real branch platform`);
    assert(portal.x >= platform.x && portal.x <= platformRight(platform), `${portal.id} should sit on its branch platform`);
    assert(portal.x <= map.platforms[0].w, `${portal.id} should remain inside the runtime world`);
    assert(portal.portalStyle.includes('starstone'), `${portal.id} should use Starfall-native portal fiction`);
  });
  assert.strictEqual(map.platforms[outgoingPortals[0].platformIndex].id, 'thornpath_fork_ridge_branch');
  assert.strictEqual(map.platforms[outgoingPortals[1].platformIndex].id, 'thornpath_fork_observatory_branch');

  const environment = Data.MAP_ENVIRONMENT_PROFILES.thornpathThicket;
  ['crystal', 'glow', 'sign'].forEach((kind) => assert(environment.propKinds.includes(kind)));
  ['grass', 'bush', 'flower'].forEach((kind) => assert(!environment.propKinds.includes(kind)));
  assert.strictEqual(environment.ramps, 'thornpath-thicket');

  const validation = validateMap(map);
  assert.deepStrictEqual(validation.issues, [], `map validation failed: ${validation.issues.join('; ')}`);
  const balanceReport = createMapBalanceReport(Data, createProjectStarfallEngine, { classIds: ['fighter'] });
  const tuning = balanceReport.mapTuning.maps.find((entry) => entry.mapId === map.id);
  assert(tuning && tuning.routeViable, 'the tuned canopy should preserve a viable loop');
  assert(tuning.metrics.travelSharePercent <= balanceReport.mapTuning.warningThresholds.travelSharePercent,
    `canopy travel share should clear the guardrail: ${tuning.metrics.travelSharePercent}%`);
  assert(!tuning.warningIds.includes('travelShareHigh'),
    'the authored canopy should not retain a high-travel warning after redundant vines are removed');

  const badPortalMap = Object.assign({}, map, {
    portals: map.portals.map((portal) => portal.id === 'thornpath_bandit'
      ? Object.assign({}, portal, { x: map.platforms[0].w + 1 })
      : portal)
  });
  assert(validateMap(badPortalMap).issues.some((issue) => issue.includes('portal thornpath_bandit is outside')),
    'runtime-bound validation should reject a stale off-map portal');

  const badSectionMap = Object.assign({}, map, {
    fieldComposition: Object.assign({}, map.fieldComposition, {
      routeSections: map.fieldComposition.routeSections.map((section, index) => index === 2
        ? Object.assign({}, section, { w: section.w + 1 })
        : section)
    })
  });
  assert(validateMap(badSectionMap).issues.some((issue) => issue.includes('route section Observatory Fork exceeds')),
    'map validation should reject a route section beyond the runtime width');

  process.stdout.write('Project Starfall Thornpath fracture canopy tests passed.\n');
}

main();
