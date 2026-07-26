'use strict';

const assert = require('assert');

const data = require('../js/games/project-starfall/data/index.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const ARCHIVE_LANE_GROUPS = Object.freeze([
  Object.freeze([
    'astral_archive_solid_lane_01',
    'astral_archive_solid_lane_02',
    'astral_archive_solid_lane_03'
  ]),
  Object.freeze([
    'astral_archive_solid_lane_04',
    'astral_archive_solid_lane_05',
    'astral_archive_solid_lane_06'
  ]),
  Object.freeze([
    'astral_archive_solid_lane_07',
    'astral_archive_solid_lane_08',
    'astral_archive_solid_lane_09'
  ])
]);
const ARCHIVE_BRIDGES = Object.freeze([
  Object.freeze({
    id: 'astralArchive_west_rune_bridge_01',
    x: 1340,
    y: 846,
    w: 302
  }),
  Object.freeze({
    id: 'astralArchive_west_rune_bridge_02',
    x: 1642,
    y: 846,
    w: 301
  }),
  Object.freeze({
    id: 'astralArchive_west_rune_bridge_03',
    x: 1943,
    y: 846,
    w: 301
  }),
  Object.freeze({
    id: 'astralArchive_east_rune_bridge_01',
    x: 3389,
    y: 666,
    w: 254
  }),
  Object.freeze({
    id: 'astralArchive_east_rune_bridge_02',
    x: 3643,
    y: 666,
    w: 253
  })
]);
const STACK_SECTION_PLATFORMS = Object.freeze({
  astralStacks_left_stacks: Object.freeze([
    'astral_stacks_solid_lane_01',
    'astral_stacks_solid_lane_02',
    'astral_stacks_solid_lane_03'
  ]),
  astralStacks_center_rune_shelf: Object.freeze([
    'astralStacks_center_rune_shelf'
  ]),
  astralStacks_right_stacks: Object.freeze([
    'astral_stacks_solid_lane_04',
    'astral_stacks_solid_lane_05',
    'astral_stacks_solid_lane_06'
  ])
});
const STACK_ACTION_PLATFORMS = Object.freeze({
  runePages: Object.freeze({
    sectionId: 'astralStacks_center_rune_shelf',
    targetTier: 'mid',
    platformId: 'astralStacks_center_rune_shelf'
  }),
  memorySeal: Object.freeze({
    sectionId: 'astralStacks_left_stacks',
    targetTier: 'high',
    platformId: 'astral_stacks_solid_lane_03'
  }),
  mirrorEcho: Object.freeze({
    sectionId: 'astralStacks_right_stacks',
    targetTier: 'high',
    platformId: 'astral_stacks_solid_lane_06'
  }),
  addWave: Object.freeze({
    sectionId: 'astralStacks_left_stacks',
    targetTier: 'high',
    platformId: 'astral_stacks_solid_lane_03'
  })
});

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function getMap(mapId) {
  return (data.MAPS || []).find((map) => map.id === mapId);
}

function getQuest(questId) {
  return (data.QUESTS || []).find((quest) => quest.id === questId);
}

function platformX(platform) {
  return Number(Array.isArray(platform) ? platform[0] : platform && platform.x) || 0;
}

function platformY(platform) {
  return Number(Array.isArray(platform) ? platform[1] : platform && platform.y) || 0;
}

function platformW(platform) {
  return Number(Array.isArray(platform) ? platform[2] : platform && platform.w) || 0;
}

function unique(values) {
  return Array.from(new Set((values || []).map(String).filter(Boolean)));
}

function sameMembers(actual, expected) {
  const left = unique(actual).sort();
  const right = unique(expected).sort();
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function horizontalGap(left, right) {
  const leftStart = platformX(left);
  const leftEnd = leftStart + platformW(left);
  const rightStart = platformX(right);
  const rightEnd = rightStart + platformW(right);
  return Math.max(0, Math.max(leftStart, rightStart) - Math.min(leftEnd, rightEnd));
}

function canTraverseElevatedPlatforms(left, right) {
  return horizontalGap(left, right) <= 64 &&
    Math.abs(platformY(left) - platformY(right)) <= 190;
}

function getElevatedReachablePlatformIds(map, startId) {
  const elevatedPlatforms = (map.platforms || []).filter((platform, index) => {
    const kind = String(platform && !Array.isArray(platform) &&
      platform.terrainVisual && platform.terrainVisual.kind || '');
    return index > 0 && kind !== 'ground';
  });
  const platformById = new Map(elevatedPlatforms.map((platform) => [platform.id, platform]));
  const start = platformById.get(startId);
  if (!start) return new Set();

  const reached = new Set([start.id]);
  const queue = [start];
  while (queue.length) {
    const current = queue.shift();
    elevatedPlatforms.forEach((candidate) => {
      if (reached.has(candidate.id) ||
          !canTraverseElevatedPlatforms(current, candidate)) return;
      reached.add(candidate.id);
      queue.push(candidate);
    });
  }
  return reached;
}

function unlockQuest(engine, questId) {
  if (!engine.state.progress.completedQuestIds.includes(questId)) {
    engine.state.progress.completedQuestIds.push(questId);
  }
  if (!engine.state.progress.claimedQuestIds.includes(questId)) {
    engine.state.progress.claimedQuestIds.push(questId);
  }
}

const stormbreakRods = getQuest('stormbreak_rods');
const stormbreakLootObjectives = stormbreakRods &&
  stormbreakRods.objectives.filter((objective) => objective.type === 'loot');
check(stormbreakRods &&
  stormbreakLootObjectives.length === 1 &&
  stormbreakLootObjectives[0].materialId === 'stormFletching' &&
  stormbreakLootObjectives[0].count === 6,
'Stormbreak Rods should require six common Storm Fletchings instead of rare Prism Shards');

const astralIndexing = getQuest('astral_indexing');
const astralLootObjectives = astralIndexing &&
  astralIndexing.objectives.filter((objective) => objective.type === 'loot');
check(astralIndexing &&
  astralLootObjectives.length === 1 &&
  astralLootObjectives[0].materialId === 'runicPage' &&
  astralLootObjectives[0].count === 8,
'Astral Indexing should require eight thematic Runic Pages instead of rare Prism Shards');

const astralArchive = getMap('astralArchive');
const astralStacks = getMap('astralStacks');
check(!!astralArchive && !!astralStacks,
  'Astral Archive and Astral Stacks should remain published');

const astralEchoOwners = (data.MAPS || []).flatMap((map) =>
  (map.questNpcs || [])
    .filter((npc) => (npc.questIds || []).includes('astral_echo'))
    .map((npc) => ({ mapId: map.id, npcId: npc.id }))
);
check(astralEchoOwners.some((owner) =>
  owner.mapId === 'astralArchive' && owner.npcId === 'astral_scribe'),
'the Astral Scribe should offer Astral Echo before the boss portal');

const archivePlatformById = new Map(
  (astralArchive.platforms || []).map((platform) => [platform.id, platform])
);
ARCHIVE_BRIDGES.forEach((expected) => {
  const bridge = archivePlatformById.get(expected.id);
  check(bridge &&
    platformX(bridge) === expected.x &&
    platformY(bridge) === expected.y &&
    platformW(bridge) === expected.w &&
    bridge.spawnDisabled === true,
  `${expected.id} should publish its exact spawn-disabled elevated bridge geometry`);
  check(!(astralArchive.spawnPoints || []).some((point) => point.platformId === expected.id),
    `${expected.id} should never become an enemy spawn shelf`);
});

const archiveLanePlatformIds = ARCHIVE_LANE_GROUPS.flat();
const archiveLaneSections = (astralArchive.spawnSections || []).filter((section) =>
  (section.platformIds || []).some((platformId) => archiveLanePlatformIds.includes(platformId))
);
check(archiveLaneSections.length === 3,
  'Astral Archive should expose three authored combat-room sections');
ARCHIVE_LANE_GROUPS.forEach((expectedPlatformIds, index) => {
  const section = archiveLaneSections.find((candidate) =>
    sameMembers(candidate.platformIds, expectedPlatformIds)
  );
  check(!!section,
    `Astral Archive room ${index + 1} should own its intended three-lane tower`);
});
const publishedArchiveSectionPlatforms = archiveLaneSections.flatMap((section) =>
  section.platformIds || []
);
check(publishedArchiveSectionPlatforms.length === 9 &&
  sameMembers(publishedArchiveSectionPlatforms, archiveLanePlatformIds),
'the three Archive rooms should cover all nine combat lanes exactly once');
check((astralArchive.spawnPoints || []).every((point) => {
  const section = archiveLaneSections.find((candidate) => candidate.id === point.sectionId);
  return section && (section.platformIds || []).includes(point.platformId);
}), 'every Archive enemy spawn should stay inside its authored combat room');

const elevatedReachableIds = getElevatedReachablePlatformIds(
  astralArchive,
  ARCHIVE_LANE_GROUPS[0][0]
);
check(archiveLanePlatformIds.every((platformId) => elevatedReachableIds.has(platformId)) &&
  ARCHIVE_BRIDGES.every((bridge) => elevatedReachableIds.has(bridge.id)),
'the rune bridges should connect all three Archive combat rooms without using the ground');

const stacksPlatformById = new Map(
  (astralStacks.platforms || []).map((platform) => [platform.id, platform])
);
const centerRuneShelf = stacksPlatformById.get('astralStacks_center_rune_shelf');
check(centerRuneShelf &&
  platformX(centerRuneShelf) === 1740 &&
  platformY(centerRuneShelf) === 172 &&
  platformW(centerRuneShelf) === 760,
'Astral Stacks should publish one broad, readable center rune shelf');
check([
  'astral_stacks_connector_02',
  'astral_stacks_connector_03',
  'astral_stacks_hop_01'
].every((platformId) => !stacksPlatformById.has(platformId)),
'the center rune shelf should replace the three fragmented center hops');

const stacksSectionById = new Map(
  (astralStacks.spawnSections || []).map((section) => [section.id, section])
);
Object.entries(STACK_SECTION_PLATFORMS).forEach(([sectionId, expectedPlatformIds]) => {
  const section = stacksSectionById.get(sectionId);
  check(section && sameMembers(section.platformIds, expectedPlatformIds),
    `${sectionId} should bind only its intended arena shelves`);
  const sectionSpawnPlatformIds = (astralStacks.spawnPoints || [])
    .filter((point) => point.sectionId === sectionId)
    .map((point) => point.platformId);
  check(sameMembers(sectionSpawnPlatformIds, expectedPlatformIds),
    `${sectionId} spawn points should follow its explicit platform contract`);
});

const archivePortal = (astralArchive.portals || []).find((portal) =>
  portal.id === 'archive_astral_stacks'
);
check(archivePortal &&
  archivePortal.destinationMapId === 'astralStacks' &&
  archivePortal.bossEncounterId === 'astralArchivist' &&
  archivePortal.requiredLevel === 88 &&
  archivePortal.bossPortal === true,
'Astral Archive should publish an explicit Level 88 Astral Stacks boss portal');

const stacksEdge = (data.WORLD_MAP_EDGES || []).find((edge) =>
  edge.id === 'archive_astral_stacks'
);
check(stacksEdge &&
  stacksEdge.fromMapId === 'astralArchive' &&
  stacksEdge.toMapId === 'astralStacks' &&
  stacksEdge.type === 'dungeon' &&
  stacksEdge.requiredLevel === 88 &&
  stacksEdge.portalIds &&
  stacksEdge.portalIds.from === 'archive_astral_stacks' &&
  stacksEdge.portalIds.to === 'stacks_return',
'the world graph should truthfully link the Archive portal to the Astral Stacks return');

const blockedEngine = createProjectStarfallEngine(null, data);
check(blockedEngine.chooseClass('fighter') === true,
  'the Level 87 portal fixture should choose Fighter');
blockedEngine.state.player.level = 87;
check(blockedEngine.changeMap('astralArchive') === true,
  'the Level 87 portal fixture should enter Astral Archive');
const blockedPortal = blockedEngine.runtime.portals.find((portal) =>
  portal.id === 'archive_astral_stacks'
);
check(blockedPortal &&
  blockedEngine.getPortalBlockReason(blockedPortal) === 'Level 88 required.' &&
  blockedEngine.usePortal(blockedPortal.id) === false &&
  blockedEngine.state.mapId === 'astralArchive' &&
  !blockedEngine.enemies.some((enemy) => enemy.isEncounterBoss),
'the Stacks portal should remain closed at Level 87 without changing maps or spawning a boss');

const entryEngine = createProjectStarfallEngine(null, data);
check(entryEngine.chooseClass('fighter') === true,
  'the Level 88 portal fixture should choose Fighter');
entryEngine.state.player.level = 88;
unlockQuest(entryEngine, 'astral_indexing');
check(entryEngine.changeMap('astralArchive') === true,
  'the Level 88 portal fixture should enter Astral Archive');
const echoAvailability = entryEngine.getQuestAvailability('astral_echo');
check(echoAvailability.available &&
  echoAvailability.npcId === 'astral_scribe' &&
  echoAvailability.mapId === 'astralArchive',
'Astral Echo should become available from the Archive Scribe at Level 88');
check(entryEngine.startQuest('astral_echo') === true,
  'the portal fixture should accept Astral Echo before entry');
check(entryEngine.usePortal('archive_astral_stacks') === true &&
  entryEngine.state.mapId === 'astralStacks',
'the Level 88 Archive portal should enter the live Astral Stacks encounter');

const liveBoss = entryEngine.enemies.find((enemy) =>
  enemy.isEncounterBoss && enemy.id === 'astralArchivist' && enemy.hp > 0
);
const runtimeCenterShelf = entryEngine.runtime.platforms.find((platform) =>
  platform.id === 'astralStacks_center_rune_shelf'
);
check(liveBoss &&
  runtimeCenterShelf &&
  liveBoss.spawnPlatformId === runtimeCenterShelf.id &&
  liveBoss.spawnPlatformIndex === runtimeCenterShelf.index &&
  liveBoss.x >= runtimeCenterShelf.x &&
  liveBoss.x + liveBoss.w <= runtimeCenterShelf.x + runtimeCenterShelf.w,
'portal entry should spawn the live Astral Archivist fully supported by the center rune shelf');

const astralEchoSummary = entryEngine.getQuestSummary(getQuest('astral_echo'));
const travelObjective = astralEchoSummary &&
  astralEchoSummary.objectives.find((objective) => objective.id === 'reach_astral_stacks');
check(travelObjective && travelObjective.value === 1 && travelObjective.complete,
  'entering through the boss portal should immediately record Astral Echo travel progress');

const spatialMechanic = data.BOSS_SPATIAL_MECHANICS &&
  data.BOSS_SPATIAL_MECHANICS.astralStacks;
check(!!spatialMechanic,
  'Astral Stacks should retain its authored spatial mechanic definition');
Object.entries(STACK_ACTION_PLATFORMS).forEach(([actionId, expected]) => {
  const hook = spatialMechanic && spatialMechanic.hooks &&
    spatialMechanic.hooks[actionId];
  check(hook &&
    hook.sectionId === expected.sectionId &&
    hook.targetTier === expected.targetTier,
  `${actionId} should retain its intended Astral section and vertical tier`);
  const section = entryEngine.getRuntimeBossSpatialSection(hook);
  const platform = entryEngine.getBossSpatialPlatformForSection(section, hook);
  check(section &&
    section.id === expected.sectionId &&
    platform &&
    platform.id === expected.platformId,
  `${actionId} should resolve onto ${expected.platformId}`);
});

console.log(`Project Starfall Astral route checks passed: ${checks}`);
