const data = require('../js/games/project-starfall/data/index.js');
const {
  createProjectStarfallEngine
} = require('../js/games/project-starfall/project-starfall-engine.js');
const {
  validateWorldGraph
} = require('../build/validate-project-starfall-maps.js');
const {
  getReachablePlatformIndices
} = require('../js/games/project-starfall/engine/map-runtime.js');
const routeProgress = require('../js/games/project-starfall/engine/route-progress.js');

let checks = 0;

function check(condition, message) {
  checks += 1;
  if (!condition) throw new Error(message);
}

function getMap(mapId) {
  return (data.MAPS || []).find((map) => map && map.id === mapId) || null;
}

function getQuest(questId) {
  return (data.QUESTS || []).find((quest) => quest && quest.id === questId) || null;
}

function unlockQuestPrerequisites(engine, quest) {
  (quest.requiredQuestIds || []).forEach((questId) => {
    if (!engine.state.progress.completedQuestIds.includes(questId)) {
      engine.state.progress.completedQuestIds.push(questId);
    }
    if (!engine.state.progress.claimedQuestIds.includes(questId)) {
      engine.state.progress.claimedQuestIds.push(questId);
    }
  });
}

const ECHO_ROUTES = [
  {
    questId: 'brambleking_echo',
    sourceMapId: 'banditRidgeCamp',
    sourceNpcId: 'ridge_watch',
    targetMapId: 'bramblekingCourt',
    witnessNpcId: 'bramble_witness',
    portalId: 'bandit_brambleking_court',
    returnPortalId: 'court_return',
    encounterId: 'brambleking',
    bossId: 'brambleking',
    requiredLevel: 32,
    elevatedPerch: true
  },
  {
    questId: 'titan_foundry_echo',
    sourceMapId: 'orebackQuarry',
    sourceNpcId: 'quarry_foreman',
    targetMapId: 'titanFoundry',
    witnessNpcId: 'titan_witness',
    portalId: 'quarry_titan_foundry',
    returnPortalId: 'foundry_return',
    encounterId: 'clockworkTitan',
    bossId: 'clockworkTitan',
    requiredLevel: 50,
    elevatedPerch: true
  },
  {
    questId: 'deepcore_echo',
    sourceMapId: 'orebackQuarry',
    sourceNpcId: 'quarry_foreman',
    targetMapId: 'deepcoreCore',
    witnessNpcId: 'deepcore_witness',
    portalId: 'quarry_deepcore_core',
    returnPortalId: 'deepcore_return',
    encounterId: 'quarryColossus',
    bossId: 'quarryColossus',
    requiredLevel: 60,
    elevatedPerch: true
  },
  {
    questId: 'emberjaw_echo',
    sourceMapId: 'cinderHollow',
    sourceNpcId: 'cinder_pathfinder',
    targetMapId: 'emberjawFurnace',
    witnessNpcId: 'emberjaw_witness',
    portalId: 'cinder_emberjaw_furnace',
    returnPortalId: 'furnace_return',
    encounterId: 'emberjawGolem',
    bossId: 'emberjawGolem',
    requiredLevel: 42,
    elevatedPerch: true
  },
  {
    questId: 'rimewarden_echo',
    sourceMapId: 'glacierSpine',
    sourceNpcId: 'glacier_cartographer',
    targetMapId: 'rimewardenVault',
    witnessNpcId: 'rimewarden_witness',
    portalId: 'glacier_rimewarden_vault',
    returnPortalId: 'vault_return',
    encounterId: 'rimewarden',
    bossId: 'rimewarden',
    requiredLevel: 66,
    elevatedPerch: true
  },
  {
    questId: 'stormbreak_echo',
    sourceMapId: 'stormbreakCliffs',
    sourceNpcId: 'stormbreak_scout',
    targetMapId: 'stormbreakAerie',
    witnessNpcId: 'stormbreak_witness',
    portalId: 'cliffs_stormbreak_aerie',
    returnPortalId: 'aerie_return',
    encounterId: 'stormbreakRoc',
    bossId: 'stormbreakRoc',
    requiredLevel: 76,
    elevatedPerch: true
  },
  {
    questId: 'astral_echo',
    sourceMapId: 'astralArchive',
    sourceNpcId: 'astral_scribe',
    targetMapId: 'astralStacks',
    witnessNpcId: 'astral_witness',
    portalId: 'archive_astral_stacks',
    returnPortalId: 'stacks_return',
    encounterId: 'astralArchivist',
    bossId: 'astralArchivist',
    requiredLevel: 88,
    elevatedPerch: false
  },
  {
    questId: 'eclipse_echo',
    sourceMapId: 'eclipseFrontier',
    sourceNpcId: 'eclipse_envoy',
    targetMapId: 'eclipseThrone',
    witnessNpcId: 'eclipse_witness',
    portalId: 'frontier_eclipse_throne',
    returnPortalId: 'throne_return',
    encounterId: 'eclipseSovereign',
    bossId: 'eclipseSovereign',
    requiredLevel: 100,
    elevatedPerch: true
  }
];

const graphValidation = validateWorldGraph(data);
check(graphValidation.ok,
  `the public world graph should be complete: ${graphValidation.issues.join(' | ')}`);
check((data.MAPS || []).filter((map) => !map.adminOnly)
  .every((map) => graphValidation.reachableMapIds.includes(map.id)),
'every public map should be reachable from Starfall Crossing');

ECHO_ROUTES.forEach((route) => {
  const quest = getQuest(route.questId);
  const sourceMap = getMap(route.sourceMapId);
  const targetMap = getMap(route.targetMapId);
  const sourceNpc = sourceMap && (sourceMap.questNpcs || [])
    .find((npc) => npc.id === route.sourceNpcId);
  const witnessNpc = targetMap && (targetMap.questNpcs || [])
    .find((npc) => npc.id === route.witnessNpcId);
  const authoredPortal = sourceMap && (sourceMap.portals || [])
    .find((portal) => portal.id === route.portalId);
  const authoredReturn = targetMap && (targetMap.portals || [])
    .find((portal) => portal.id === route.returnPortalId);
  const edge = (data.WORLD_MAP_EDGES || [])
    .find((candidate) => candidate.id === route.portalId);

  check(quest &&
    quest.requiredLevel === route.requiredLevel &&
    quest.objectives.length === 2 &&
    quest.objectives[0].type === 'travel' &&
    quest.objectives[0].mapId === route.targetMapId &&
    quest.objectives[1].type === 'defeatBoss' &&
    quest.objectives[1].mapId === route.targetMapId &&
    quest.objectives[1].bossId === route.bossId,
  `${route.questId} should retain its travel-then-boss objective contract`);
  check(sourceNpc && sourceNpc.questIds.includes(route.questId),
    `${route.questId} should be offered before its boss portal`);
  check(witnessNpc && witnessNpc.questIds.includes(route.questId),
    `${route.questId} should remain claimable from its cleared arena`);
  check(authoredPortal &&
    authoredPortal.destinationMapId === route.targetMapId &&
    authoredPortal.bossEncounterId === route.encounterId &&
    authoredPortal.bossPortal === true &&
    authoredPortal.requiredLevel === route.requiredLevel &&
    authoredPortal.requiredQuestId === route.questId &&
    (route.elevatedPerch ? authoredPortal.platformIndex > 0 : authoredPortal.platformIndex === 0),
  `${route.portalId} should be a deliberate, quest-gated Echo entrance`);
  check(authoredReturn &&
    authoredReturn.returnPortal === true &&
    authoredReturn.destinationMapId === route.sourceMapId,
  `${route.returnPortalId} should return to the Echo's source field`);
  check(edge &&
    edge.fromMapId === route.sourceMapId &&
    edge.toMapId === route.targetMapId &&
    edge.type === 'dungeon' &&
    edge.requiredLevel === route.requiredLevel &&
    edge.requiredQuestId === route.questId &&
    edge.portalIds.from === route.portalId &&
    edge.portalIds.to === route.returnPortalId,
  `${route.portalId} should have a truthful world-map edge`);

  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass('fighter') === true,
    `${route.questId} gate fixture should choose Fighter`);
  engine.state.player.level = route.requiredLevel - 1;
  check(engine.changeMap(route.sourceMapId) === true,
    `${route.questId} gate fixture should enter its source field`);
  const runtimePortal = engine.runtime.portals.find((portal) => portal.id === route.portalId);
  const reachablePlatformIndices = getReachablePlatformIndices(engine.runtime.platformGraph, 0);
  check(runtimePortal && reachablePlatformIndices.has(runtimePortal.platformIndex),
    `${route.portalId} should be physically reachable from its source field route`);
  if (route.portalId === 'cliffs_stormbreak_aerie') {
    const portalPlatform = engine.runtime.platforms[runtimePortal.platformIndex];
    check(portalPlatform &&
      portalPlatform.id === 'stormbreakCliffs_aerie_perch' &&
      portalPlatform.spawnDisabled &&
      engine.runtime.climbables.some((climbable) =>
        climbable.id.includes('_storm_stair_') &&
        climbable.x >= portalPlatform.x &&
        climbable.x <= portalPlatform.x + portalPlatform.w
      ),
    'the Stormbreak Aerie gate should use a calm, storm-stair-connected lookout perch');
  }
  check(runtimePortal &&
    engine.getPortalBlockReason(runtimePortal) === `Level ${route.requiredLevel} required.`,
  `${route.portalId} should enforce its level before other locks`);

  engine.state.player.level = route.requiredLevel;
  const prerequisite = getQuest(quest.requiredQuestIds[0]);
  check(engine.getPortalBlockReason(runtimePortal) === `Complete ${prerequisite.title} first.`,
    `${route.portalId} should explain its missing prerequisite`);
  unlockQuestPrerequisites(engine, quest);
  check(engine.getPortalBlockReason(runtimePortal) === `Accept ${quest.title} first.`,
    `${route.portalId} should require accepting the available Echo`);
  const availability = engine.getQuestAvailability(route.questId);
  check(availability.available &&
    availability.npcId === route.sourceNpcId &&
    availability.mapId === route.sourceMapId,
  `${route.questId} should be available from its reachable source NPC`);
  check(engine.startQuest(route.questId) === true,
    `${route.questId} should start from its source field`);
  check(engine.getPortalBlockReason(runtimePortal) === '',
    `${route.portalId} should unlock for an active Echo`);

  const guidance = engine.getQuestGuidanceSnapshot();
  check(guidance.navigationTarget &&
    guidance.navigationTarget.portalId === route.portalId &&
    guidance.navigationTarget.destinationMapId === route.targetMapId &&
    guidance.navigationTarget.lockedReason === '',
  `${route.questId} guidance should mark its real encounter portal`);
  const worldPath = engine.getWorldMapPath(route.sourceMapId, route.targetMapId);
  check(worldPath &&
    worldPath.steps.length === 1 &&
    worldPath.steps[0].portalId === route.portalId &&
    worldPath.lockedReason === '',
  `${route.questId} should have one unlocked source-to-arena world step`);
  check(engine.usePortal(route.portalId) === true &&
    engine.state.mapId === route.targetMapId,
  `${route.portalId} should open ${route.targetMapId}`);
  const entrySummary = engine.getQuestSummary(quest);
  check(entrySummary.objectives[0].complete && !entrySummary.objectives[1].complete,
    `${route.questId} entry should complete travel before the boss`);

  const activeSave = engine.serialize();
  const restoredEngine = createProjectStarfallEngine(null, data);
  check(restoredEngine.restore(activeSave) === true &&
    restoredEngine.state.mapId === route.targetMapId,
  `${route.questId} should restore inside its active encounter`);
  const liveBoss = restoredEngine.enemies.find((enemy) =>
    enemy.isEncounterBoss && enemy.id === route.bossId && enemy.hp > 0
  );
  check(liveBoss &&
    restoredEngine.state.dungeons.currentRun &&
    restoredEngine.state.dungeons.currentRun.bossEncounterId === route.encounterId,
  `${route.questId} restore should rebuild its intended live boss`);
  restoredEngine.defeatEnemy(liveBoss);
  const completedSummary = restoredEngine.getQuestSummary(quest);
  const completedAvailability = restoredEngine.getQuestAvailability(route.questId);
  check(completedSummary.objectives.every((objective) => objective.complete) &&
    completedAvailability.claimable,
  `${route.questId} boss defeat should make the Echo reward claimable`);
  const localOwner = restoredEngine.getCurrentMapQuestOwner(route.questId);
  check(localOwner && localOwner.npcId === route.witnessNpcId,
    `${route.questId} should be claimable from its in-room witness`);

  const witness = restoredEngine.runtime.questNpcs.find((npc) => npc.id === route.witnessNpcId);
  restoredEngine.state.player.x = witness.x;
  restoredEngine.state.player.y = witness.y - restoredEngine.state.player.h;
  const currencyBeforeClaim = restoredEngine.state.player.currency;
  check(restoredEngine.claimQuestRewardFromNpc(route.witnessNpcId, route.questId) === true &&
    restoredEngine.state.player.currency > currencyBeforeClaim,
  `${route.questId} witness should grant its first-clear reward`);
  const currencyAfterClaim = restoredEngine.state.player.currency;
  check(restoredEngine.claimQuestRewardFromNpc(route.witnessNpcId, route.questId) === false &&
    restoredEngine.state.player.currency === currencyAfterClaim,
  `${route.questId} witness should never grant its reward twice`);
  check(restoredEngine.usePortal(route.returnPortalId) === true &&
    restoredEngine.state.mapId === route.sourceMapId,
  `${route.returnPortalId} should return the player to ${route.sourceMapId}`);

  const claimedSave = restoredEngine.serialize();
  const claimedEngine = createProjectStarfallEngine(null, data);
  check(claimedEngine.restore(claimedSave) === true,
    `${route.questId} claimed state should survive save and reload`);
  const replayPortal = claimedEngine.runtime.portals.find((portal) =>
    portal.id === route.portalId
  );
  check(claimedEngine.getQuestAvailability(route.questId).claimed &&
    replayPortal &&
    claimedEngine.getPortalBlockReason(replayPortal) === '',
  `${route.questId} should remain replayable after its one-time reward`);
});

const frostfenEngine = createProjectStarfallEngine(null, data);
check(frostfenEngine.chooseClass('fighter') === true,
  'the Frostfen route fixture should choose Fighter');
frostfenEngine.state.player.level = 100;
check(frostfenEngine.changeMap('frostfenOutskirts') === true,
  'the Frostfen route fixture should enter the Outskirts');
const glacierPath = frostfenEngine.getWorldMapPath('frostfenOutskirts', 'glacierSpine');
check(glacierPath &&
  glacierPath.steps.length === 1 &&
  glacierPath.steps[0].portalId === 'frostfen_glacier',
'Frostfen guidance should use the physical Outskirts-to-Glacier ascent');
check(!(getMap('frostfenCamp').portals || [])
  .some((portal) => portal.id === 'frostfen_camp_glacier') &&
  (getMap('glacierSpine').portals || []).some((portal) =>
    portal.id === 'glacier_frostfen_outskirts' &&
    portal.destinationMapId === 'frostfenOutskirts'
  ),
'Frostfen should remove the duplicate camp shortcut and expose a truthful Tundra return');

const disconnectedEngine = createProjectStarfallEngine(null, data);
check(disconnectedEngine.chooseClass('fighter') === true,
  'the disconnected guidance fixture should choose Fighter');
const disconnectedGuide = {
  active: true,
  complete: false,
  recommendedMapId: 'missingEchoMap',
  recommendedMapName: 'Missing Echo Map'
};
const disconnectedPortal = disconnectedEngine.getQuestNavigationPortal(disconnectedGuide);
const disconnectedTarget = disconnectedEngine.getQuestNavigationTarget(disconnectedGuide);
check(disconnectedPortal === null &&
  disconnectedTarget &&
  disconnectedTarget.kind === 'mapHint' &&
  !disconnectedTarget.portalId,
'unreachable explicit guidance should never select an unrelated shop or route portal');
check(disconnectedEngine.getQuestNavigationPortal({ active: true }) === null,
  'guidance without a destination should never invent a portal target');
check(routeProgress.getRouteForFieldMap('', { data }) === null &&
  routeProgress.getRouteForBossMap(null, { data }) === null &&
  routeProgress.getRouteForDungeon(undefined, { data }) === null,
'blank route lookup ids should never match routes with omitted metadata');

console.log(`Project Starfall Boss Echo route checks passed: ${checks}`);
