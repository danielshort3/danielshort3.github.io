'use strict';

const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { createMapBalanceReport } = require('./project-starfall-balance-harness.js');

const MAP_ID = 'rimewardenVault';
const EXPECTED_SECTION_PLATFORMS = Object.freeze({
  rimewardenVault_brute_lane: Object.freeze([
    'rimewarden_vault_solid_lane_01',
    'rimewarden_vault_solid_lane_04'
  ]),
  rimewardenVault_oracle_shelf: Object.freeze([
    'rimewarden_vault_solid_lane_02',
    'rimewarden_vault_solid_lane_05'
  ]),
  rimewardenVault_sentinel_shelf: Object.freeze([
    'rimewarden_vault_solid_lane_03',
    'rimewarden_vault_solid_lane_06'
  ])
});
const EXPECTED_ACTION_SECTIONS = Object.freeze({
  iceShockwave: Object.freeze({
    sectionId: 'rimewardenVault_brute_lane',
    targetTier: 'ground',
    platformId: 'rimewarden_vault_solid_lane_01'
  }),
  whiteout: Object.freeze({
    sectionId: 'rimewardenVault_oracle_shelf',
    targetTier: 'mid',
    platformId: 'rimewarden_vault_solid_lane_05'
  }),
  iceWall: Object.freeze({
    sectionId: 'rimewardenVault_sentinel_shelf',
    targetTier: 'high',
    platformId: 'rimewarden_vault_solid_lane_06'
  }),
  addWave: Object.freeze({
    sectionId: 'rimewardenVault_sentinel_shelf',
    targetTier: 'high',
    platformId: 'rimewarden_vault_solid_lane_06'
  })
});

let checks = 0;
const failures = [];

function check(condition, message, details) {
  checks += 1;
  if (condition) return;
  failures.push(details ? `${message} (${details})` : message);
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

function intersection(left, right) {
  const rightValues = new Set(right || []);
  return unique(left).filter((value) => rightValues.has(value));
}

function platformCounts(values) {
  return (values || []).reduce((counts, value) => {
    const id = String(value || '');
    if (id) counts[id] = Number(counts[id] || 0) + 1;
    return counts;
  }, {});
}

function finish() {
  if (failures.length) {
    console.error(`Project Starfall Rimewarden platform-section contract failed: ${failures.length}/${checks} checks.`);
    failures.forEach((failure, index) => {
      console.error(`${index + 1}. ${failure}`);
    });
    process.exitCode = 1;
    return;
  }
  console.log(`Project Starfall Rimewarden platform-section contract passed: ${checks} checks.`);
}

function run() {
  const map = (data.MAPS || []).find((entry) => entry.id === MAP_ID);
  check(!!map, 'Rimewarden Vault should exist in the published map catalog');
  if (!map) {
    finish();
    return;
  }

  const sectionIds = Object.keys(EXPECTED_SECTION_PLATFORMS);
  const sectionsById = new Map((map.spawnSections || []).map((section) => [section.id, section]));
  const groupsBySectionId = new Map((map.spawnGroups || []).map((group) => [group.sectionId, group]));
  const publishedSectionPlatformIds = [];

  sectionIds.forEach((sectionId) => {
    const expectedPlatformIds = EXPECTED_SECTION_PLATFORMS[sectionId];
    const section = sectionsById.get(sectionId);
    check(!!section, `${sectionId} should publish as a spawn section`);
    if (!section) return;

    const sectionPlatformIds = (section.platformIds || []).map(String).filter(Boolean);
    const uniqueSectionPlatformIds = unique(sectionPlatformIds);
    publishedSectionPlatformIds.push(...sectionPlatformIds);
    check(sectionPlatformIds.length === 2 && uniqueSectionPlatformIds.length === 2,
      `${sectionId} should publish exactly two unique platform IDs`,
      `published ${JSON.stringify(sectionPlatformIds)}`);
    check(sameMembers(sectionPlatformIds, expectedPlatformIds),
      `${sectionId} should publish its intended vertical platform pair`,
      `expected ${JSON.stringify(expectedPlatformIds)}, received ${JSON.stringify(sectionPlatformIds)}`);

    const sectionPoints = (map.spawnPoints || []).filter((point) => point.sectionId === sectionId);
    const pointPlatformIds = sectionPoints.map((point) => point.platformId);
    check(sectionPoints.length === 2 && unique(pointPlatformIds).length === 2,
      `${sectionId} should own exactly two unique spawn points`,
      `received ${JSON.stringify(pointPlatformIds)}`);
    check(sameMembers(pointPlatformIds, expectedPlatformIds),
      `${sectionId} spawn points should use its intended vertical platform pair`,
      `expected ${JSON.stringify(expectedPlatformIds)}, received ${JSON.stringify(pointPlatformIds)}`);

    const group = groupsBySectionId.get(sectionId);
    check(!!group, `${sectionId} should publish a matching spawn group`);
    if (group) {
      check(sameMembers(group.platformIds, expectedPlatformIds),
        `${sectionId} spawn group should stay inside its intended platform territory`,
        `expected ${JSON.stringify(expectedPlatformIds)}, received ${JSON.stringify(group.platformIds || [])}`);
    }
  });

  const publishedCounts = platformCounts(publishedSectionPlatformIds);
  const spawnPlatformIds = (map.spawnPoints || []).map((point) => point.platformId);
  check(publishedSectionPlatformIds.length === 6 &&
    Object.keys(publishedCounts).length === 6 &&
    Object.values(publishedCounts).every((count) => count === 1),
  'the three sections should cover six platform IDs exactly once',
  `received ${JSON.stringify(publishedCounts)}`);
  check(spawnPlatformIds.length === 6 && sameMembers(publishedSectionPlatformIds, spawnPlatformIds),
    'the section platform contract should cover every Rimewarden spawn platform once',
    `section platforms ${JSON.stringify(publishedSectionPlatformIds)}, spawn platforms ${JSON.stringify(spawnPlatformIds)}`);

  for (let leftIndex = 0; leftIndex < sectionIds.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < sectionIds.length; rightIndex += 1) {
      const leftId = sectionIds[leftIndex];
      const rightId = sectionIds[rightIndex];
      const leftGroup = groupsBySectionId.get(leftId);
      const rightGroup = groupsBySectionId.get(rightId);
      if (!leftGroup || !rightGroup) continue;
      const sharedPlatforms = intersection(leftGroup.platformIds, rightGroup.platformIds);
      check(sharedPlatforms.length === 0,
        `${leftId} and ${rightId} spawn groups should be pairwise disjoint`,
        `shared ${JSON.stringify(sharedPlatforms)}`);
    }
  }

  const balanceReport = createMapBalanceReport(data, createProjectStarfallEngine, {
    level: 70,
    rank: 10,
    classIds: ['fighter']
  });
  const tuning = balanceReport &&
    balanceReport.mapTuning &&
    (balanceReport.mapTuning.maps || []).find((entry) => entry.mapId === MAP_ID);
  check(!!tuning, 'the balance harness should publish Rimewarden map-tuning metrics');
  if (tuning) {
    check(tuning.activeSpawnSectionCount === 3,
      'Rimewarden tuning should report all three sections active',
      `received ${tuning.activeSpawnSectionCount}`);
    check(tuning.emptySectionCount === 0,
      'Rimewarden tuning should report no empty authored sections',
      `received ${tuning.emptySectionCount}`);
    check(!(tuning.warningIds || []).includes('partyOverlapHigh'),
      'Rimewarden should not retain the high party-overlap warning',
      `warnings ${JSON.stringify(tuning.warningIds || [])}`);
  }

  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass('fighter') === true,
    'the boss-spatial fixture should choose Fighter');
  check(engine.changeMap(MAP_ID) === true,
    'the boss-spatial fixture should enter Rimewarden Vault');
  const mechanic = data.BOSS_SPATIAL_MECHANICS &&
    data.BOSS_SPATIAL_MECHANICS[MAP_ID];
  check(!!mechanic, 'Rimewarden should publish its boss spatial mechanic definition');
  if (mechanic) {
    Object.entries(EXPECTED_ACTION_SECTIONS).forEach(([actionId, expected]) => {
      const hook = mechanic.hooks && mechanic.hooks[actionId];
      check(!!hook, `${actionId} should publish a Rimewarden spatial hook`);
      if (!hook) return;
      check(hook.sectionId === expected.sectionId && hook.targetTier === expected.targetTier,
        `${actionId} should retain its intended section and vertical tier`,
        `received ${hook.sectionId}/${hook.targetTier}`);
      const section = engine.getRuntimeBossSpatialSection(hook);
      check(section && section.id === expected.sectionId,
        `${actionId} should resolve its intended runtime section`,
        `received ${section && section.id || 'none'}`);
      const platform = engine.getBossSpatialPlatformForSection(section, hook);
      check(platform && EXPECTED_SECTION_PLATFORMS[expected.sectionId].includes(platform.id),
        `${actionId} should resolve onto a platform in its intended vertical section`,
        `received ${platform && platform.id || 'none'}`);
      check(platform && platform.id === expected.platformId,
        `${actionId} should prefer the intended side of its authored platform pair`,
        `expected ${expected.platformId}, received ${platform && platform.id || 'none'}`);
    });

    check(engine.enterBossEncounter('rimewarden', { admin: true }) === true,
      'the add-wave fixture should enter the Rimewarden encounter');
    const boss = engine.enemies.find((enemy) => enemy.isEncounterBoss && enemy.id === 'rimewarden');
    const encounter = boss && engine.getBossEncounterForEnemy(boss);
    const enemiesBeforeAddWave = new Set(engine.enemies.map((enemy) => enemy.uid));
    if (boss && encounter) {
      engine.beginBossEncounterAction(
        boss,
        encounter,
        encounter.phases[1],
        'addWave',
        engine.getCombatCharacterByTarget('player', 'player')
      );
      const pending = boss.bossPendingAction;
      engine.resolveBossEncounterAction(boss, encounter, pending);
      const adds = engine.enemies.filter((enemy) =>
        enemy.encounterMinion && !enemiesBeforeAddWave.has(enemy.uid));
      check(adds.length === 2,
        'Rimewarden add wave should spawn two encounter minions',
        `received ${adds.length}`);
      adds.forEach((add) => {
        const platform = engine.runtime.platforms[add.spawnPlatformIndex];
        check(add.spawnSectionId === EXPECTED_ACTION_SECTIONS.addWave.sectionId &&
          EXPECTED_SECTION_PLATFORMS[EXPECTED_ACTION_SECTIONS.addWave.sectionId].includes(add.spawnPlatformId),
        `${add.id} should spawn on an authored Sentinel Shelf platform`,
        `received ${add.spawnSectionId}/${add.spawnPlatformId}`);
        check(platform &&
          add.x >= platform.x &&
          add.x + add.w <= platform.x + platform.w,
        `${add.id} should remain fully supported by its selected shelf`,
        `enemy ${add.x}-${add.x + add.w}, platform ${platform && platform.x}-${platform && platform.x + platform.w}`);
      });
    } else {
      check(false, 'the add-wave fixture should expose Rimewarden and its encounter definition');
    }
  }

  finish();
}

run();
