'use strict';

const data = require('../js/games/project-starfall/data/index.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

let checks = 0;
const failures = [];

function check(condition, message) {
  checks += 1;
  if (!condition) failures.push(message);
}

function createEngine(mapId) {
  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass('fighter'), 'retention fixtures should create a playable Fighter');
  if (mapId) check(engine.changeMap(mapId), `retention fixtures should enter ${mapId}`);
  return engine;
}

function getQuestSummary(engine, questId) {
  return engine.getProgressSnapshot().quests.find((quest) => quest.id === questId) || null;
}

function getQuestObjective(engine, questId, objectiveId) {
  const quest = getQuestSummary(engine, questId);
  return quest && quest.objectives.find((objective) => objective.id === objectiveId) || null;
}

function countActiveQuestId(engine, questId) {
  return engine.getProgressState().activeQuestIds.filter((id) => id === questId).length;
}

function getLiveEnemy(engine, enemyId) {
  return (engine.enemies || []).find((enemy) =>
    enemy &&
    enemy.id === enemyId &&
    !enemy.defeatedAt &&
    Number(enemy.hp || 0) > 0) || null;
}

function createCollectibleDrop(engine, suffix) {
  const player = engine.state.player;
  const drop = engine.dropLootItem({
    uid: `retention_loot_${suffix}`,
    id: 'upgrade_dust',
    kind: 'material',
    materialId: 'upgradeDust',
    name: 'Upgrade Dust',
    rarity: 'Common',
    quantity: 1,
    asset: data.ITEM_ASSETS.upgrade_dust
  }, null, {
    landX: player.x + player.w / 2,
    landY: player.y + player.h
  });
  if (!drop) return null;
  drop.airborne = false;
  drop.x = player.x + player.w / 2;
  drop.y = player.y + player.h;
  drop.vx = 0;
  drop.vy = 0;
  drop.settledAt = Date.now();
  engine.invalidateLootDropCaches();
  return drop;
}

function getStackCount(engine, itemId) {
  return Math.max(0, Number(engine.state.consumables && engine.state.consumables[itemId] || 0));
}

function getRatioSpread(values) {
  const positive = values.filter((value) => Number.isFinite(value) && value > 0);
  if (positive.length !== values.length || !positive.length) return Infinity;
  return Math.max(...positive) / Math.min(...positive);
}

function markHuntComplete(engine, mapId, goal) {
  const state = engine.getMapKillQuestState()[mapId];
  state.active = true;
  state.progress = goal;
  state.completedAt = Date.now();
  return state;
}

// First Steps should remain optional through travel and unrelated activity.
const travelOnlyEngine = createEngine('greenrootMeadow');
check(countActiveQuestId(travelOnlyEngine, 'first_steps') === 0 &&
  travelOnlyEngine.getQuestAvailability('first_steps').available,
'travel to Greenroot should expose First Steps without auto-activating it');

const dewEngine = createEngine('greenrootMeadow');
const dewSlime = getLiveEnemy(dewEngine, 'dewSlime');
check(!!dewSlime, 'Greenroot should spawn a live Dew Slime for the non-activation regression');
if (dewSlime) dewEngine.defeatEnemy(dewSlime);
check(countActiveQuestId(dewEngine, 'first_steps') === 0 &&
  getQuestObjective(dewEngine, 'first_steps', 'defeat_slimelets').value === 0,
'an unrelated Greenroot enemy should not auto-activate or advance First Steps');

const offMapLootEngine = createEngine('thornpathThicket');
const offMapDrop = createCollectibleDrop(offMapLootEngine, 'off_map');
check(!!offMapDrop && offMapLootEngine.lootItem(offMapDrop.uid),
  'the off-map regression should collect a real ground drop');
check(countActiveQuestId(offMapLootEngine, 'first_steps') === 0 &&
  getQuestObjective(offMapLootEngine, 'first_steps', 'loot_drop').value === 0,
'loot collected outside Greenroot should not auto-activate or advance First Steps');

// A meaningful first field action should auto-activate the starter quest and count itself once.
const slimeletEngine = createEngine('greenrootMeadow');
slimeletEngine.state.session.questGuide = { type: 'mapKill', id: 'greenrootMeadow' };
const firstSlimelet = getLiveEnemy(slimeletEngine, 'slimelet');
check(!!firstSlimelet, 'Greenroot should spawn a live Slimelet for starter-quest activation');
if (firstSlimelet) slimeletEngine.defeatEnemy(firstSlimelet);
let slimeletObjective = getQuestObjective(slimeletEngine, 'first_steps', 'defeat_slimelets');
check(countActiveQuestId(slimeletEngine, 'first_steps') === 1 &&
  slimeletObjective &&
  slimeletObjective.value === 1,
'the first real Greenroot Slimelet defeat should activate First Steps and count as 1/3');
check(slimeletEngine.state.session.questGuide.type === 'mapKill' &&
  slimeletEngine.state.session.questGuide.id === 'greenrootMeadow',
'auto-activating First Steps should preserve an explicit active guide target');

check(slimeletEngine.recordProgressEvent('defeat', {
  enemyId: 'slimelet',
  mapId: 'greenrootMeadow',
  count: 1
}), 'an active First Steps quest should accept a second matching defeat event');
slimeletObjective = getQuestObjective(slimeletEngine, 'first_steps', 'defeat_slimelets');
check(countActiveQuestId(slimeletEngine, 'first_steps') === 1 &&
  new Set(slimeletEngine.getProgressState().activeQuestIds).size === slimeletEngine.getProgressState().activeQuestIds.length &&
  slimeletObjective &&
  slimeletObjective.value === 2,
'subsequent Slimelet events should increment once without duplicating the active quest id');

if (typeof slimeletEngine.getCurrentMapQuestOwner === 'function') {
  const localOwner = slimeletEngine.getCurrentMapQuestOwner('first_steps');
  const localAvailability = slimeletEngine.getQuestAvailability('first_steps');
  check(localOwner &&
    localOwner.npcId === 'greenroot_guide' &&
    localOwner.mapId === 'greenrootMeadow' &&
    localAvailability.npcId === localOwner.npcId &&
    localAvailability.mapId === localOwner.mapId,
  'auto-assigned First Steps should resolve to the local Greenroot Guide while the player is in Greenroot');
}

const lootEngine = createEngine('greenrootMeadow');
const greenrootDrop = createCollectibleDrop(lootEngine, 'greenroot');
check(!!greenrootDrop && lootEngine.lootItem(greenrootDrop.uid),
  'the starter-loot regression should collect a real Greenroot ground drop');
let lootObjective = getQuestObjective(lootEngine, 'first_steps', 'loot_drop');
check(countActiveQuestId(lootEngine, 'first_steps') === 1 &&
  lootObjective &&
  lootObjective.value === 1 &&
  lootObjective.complete,
'the first real Greenroot pickup should activate First Steps and complete its 1/1 loot objective');
lootEngine.recordProgressEvent('loot', {
  kind: 'material',
  materialId: 'upgradeDust',
  itemId: 'upgrade_dust',
  mapId: 'greenrootMeadow',
  count: 1
});
lootObjective = getQuestObjective(lootEngine, 'first_steps', 'loot_drop');
check(countActiveQuestId(lootEngine, 'first_steps') === 1 &&
  lootObjective &&
  lootObjective.value === 1,
'later loot events should remain capped without duplicating First Steps');

const savedFirstSteps = slimeletEngine.serialize();
const restoredFirstStepsEngine = createProjectStarfallEngine(null, data);
check(restoredFirstStepsEngine.restore(savedFirstSteps),
  'an auto-assigned First Steps save should restore');
const restoredSlimeletObjective = getQuestObjective(restoredFirstStepsEngine, 'first_steps', 'defeat_slimelets');
check(countActiveQuestId(restoredFirstStepsEngine, 'first_steps') === 1 &&
  restoredSlimeletObjective &&
  restoredSlimeletObjective.value === 2,
'save/restore should preserve one active First Steps id and its exact Slimelet progress');

const claimedEngine = createEngine('greenrootMeadow');
claimedEngine.state.progress.completedQuestIds.push('first_steps');
claimedEngine.state.progress.claimedQuestIds.push('first_steps');
claimedEngine.recordProgressEvent('defeat', {
  enemyId: 'slimelet',
  mapId: 'greenrootMeadow',
  count: 1
});
claimedEngine.recordProgressEvent('loot', {
  kind: 'material',
  materialId: 'upgradeDust',
  itemId: 'upgrade_dust',
  mapId: 'greenrootMeadow',
  count: 1
});
check(countActiveQuestId(claimedEngine, 'first_steps') === 0 &&
  claimedEngine.getQuestAvailability('first_steps').claimed,
'matching events should never reactivate an already claimed First Steps quest');

// The tracker should merge overlapping onboarding and quest guidance into one useful card.
const trackerEngine = createEngine('greenrootMeadow');
const defeatStepIndex = data.ONBOARDING_STEPS.findIndex((step) => step.id === 'defeat_enemy');
trackerEngine.state.onboarding.completedIds = data.ONBOARDING_STEPS
  .slice(0, defeatStepIndex)
  .map((step) => step.id);
trackerEngine.onboardingSnapshotCache = null;
const trackerSlimelet = getLiveEnemy(trackerEngine, 'slimelet');
check(!!trackerSlimelet, 'the merged-tracker fixture should find a live Slimelet');
if (trackerSlimelet) trackerEngine.defeatEnemy(trackerSlimelet);
const trackerOnboarding = trackerEngine.getOnboardingSnapshot();
const trackerEntries = hud.getQuestTrackerEntries({
  onboarding: trackerOnboarding,
  progress: trackerEngine.getProgressTrackerSnapshot()
}, {
  keyLabels: { loot: 'G' }
});
const firstStepsEntries = trackerEntries.filter((entry) =>
  entry.guideId === 'first_steps' ||
  /first steps/i.test(String(entry.title || '')));
const mergedFirstSteps = firstStepsEntries.find((entry) =>
  entry.guideType === 'quest' &&
  entry.guideId === 'first_steps');
const mergedGuideHint = mergedFirstSteps && (mergedFirstSteps.objectives || [])
  .find((objective) => String(objective.id || '').startsWith('onboarding:'));
const mergedDefeatObjective = mergedFirstSteps && (mergedFirstSteps.objectives || [])
  .find((objective) => objective.id === 'defeat_slimelets');
check(trackerOnboarding.nextStep && trackerOnboarding.nextStep.id === 'loot_drop',
  'the merged-tracker fixture should advance onboarding to its loot step');
check(firstStepsEntries.length === 1 &&
  mergedFirstSteps &&
  !trackerEntries.some((entry) => entry.guideType === 'guide' && /first steps/i.test(String(entry.title || ''))),
'the HUD should expose one merged First Steps entry instead of separate guide and quest cards');
check(mergedGuideHint &&
  /G/.test(String(mergedGuideHint.label || '')) &&
  /loot|drop|pick/i.test(String(mergedGuideHint.label || '')),
'the merged First Steps entry should retain the rebound loot-key hint');
const unboundTrackerEntries = hud.getQuestTrackerEntries({
  onboarding: trackerOnboarding,
  progress: trackerEngine.getProgressTrackerSnapshot()
}, {
  keyLabels: { loot: 'Unbound' }
});
const unboundFirstSteps = unboundTrackerEntries.find((entry) =>
  entry.guideType === 'quest' &&
  entry.guideId === 'first_steps');
const unboundGuideHint = unboundFirstSteps && (unboundFirstSteps.objectives || [])
  .find((objective) => String(objective.id || '').startsWith('onboarding:'));
check(unboundGuideHint &&
  /bind Loot in Keybinds/i.test(String(unboundGuideHint.label || '')) &&
  !/hold Unbound/i.test(String(unboundGuideHint.label || '')),
'the merged First Steps entry should give a useful fallback when Loot is unbound');
check(mergedDefeatObjective &&
  mergedDefeatObjective.value === 1 &&
  mergedDefeatObjective.goal === 3,
'the merged First Steps entry should retain live quest progress alongside its guide hint');

// Every combat map should share the same capped Scout -> Ranger -> Veteran hunt ladder.
const huntEngine = createEngine();
const combatMaps = data.MAPS.filter((map) => map && !map.safeZone);
check(combatMaps.length > 0, 'retention checks should discover combat maps');

combatMaps.forEach((map) => {
  const huntState = huntEngine.getMapKillQuestState()[map.id];
  const baseGoal = huntEngine.getMapKillQuestBaseGoal(map);
  const rankCases = [
    { completions: 0, id: 'scout', label: 'Scout', index: 0, multiplier: 1, maxRank: false },
    { completions: 2, id: 'scout', label: 'Scout', index: 0, multiplier: 1, maxRank: false },
    { completions: 3, id: 'ranger', label: 'Ranger', index: 1, multiplier: 1.25, maxRank: false },
    { completions: 9, id: 'ranger', label: 'Ranger', index: 1, multiplier: 1.25, maxRank: false },
    { completions: 10, id: 'veteran', label: 'Veteran', index: 2, multiplier: 1.5, maxRank: true },
    { completions: 250, id: 'veteran', label: 'Veteran', index: 2, multiplier: 1.5, maxRank: true }
  ];
  const ranked = rankCases.map((rankCase) => {
    huntState.active = false;
    huntState.progress = 0;
    huntState.completedAt = 0;
    huntState.completions = rankCase.completions;
    const goal = huntEngine.getMapKillQuestGoal(map.id);
    const rank = huntEngine.getMapKillQuestRank(rankCase.completions);
    const summary = huntEngine.getMapKillQuestSnapshot(map.id);
    const reward = huntEngine.getMapKillQuestRewards(map, { goal });
    check(rank.id === rankCase.id &&
      rank.label === rankCase.label &&
      rank.rankIndex === rankCase.index &&
      rank.maxRank === rankCase.maxRank,
    `${map.id} should resolve completion ${rankCase.completions} to the expected capped hunt rank`);
    check(goal === Math.ceil(baseGoal * rankCase.multiplier),
      `${map.id} ${rankCase.label} Hunt should use its bounded goal multiplier`);
    check(summary &&
      summary.rankId === rank.id &&
      summary.rankLabel === rank.label &&
      summary.rankIndex === rank.rankIndex &&
      summary.maxRank === rank.maxRank &&
      summary.goalMultiplier === rankCase.multiplier &&
      summary.goal === goal &&
      summary.objectives[0].goal === goal,
    `${map.id} ${rankCase.label} Hunt snapshot should publish consistent rank and goal fields`);
    check(typeof summary.rewardSummary === 'string' &&
      /XP/i.test(summary.rewardSummary) &&
      /coin/i.test(summary.rewardSummary),
    `${map.id} ${rankCase.label} Hunt should summarize its repeatable XP and coin rewards`);
    check(Number(reward.xp || 0) > 0 && Number(reward.currency || 0) > 0,
      `${map.id} ${rankCase.label} Hunt should retain positive XP and coin rewards`);
    return { goal, reward };
  });

  check(ranked[1].goal === ranked[0].goal &&
    ranked[3].goal === ranked[2].goal &&
    ranked[5].goal === ranked[4].goal,
  `${map.id} Hunt goals should remain stable within each rank`);
  check(ranked[5].goal === ranked[4].goal,
    `${map.id} Hunt goals should remain capped at Veteran after many completions`);
  const efficiencyRanks = [ranked[0], ranked[2], ranked[4]];
  const xpEfficiency = efficiencyRanks.map((entry) => Number(entry.reward.xp || 0) / entry.goal);
  const coinEfficiency = efficiencyRanks.map((entry) => Number(entry.reward.currency || 0) / entry.goal);
  check(getRatioSpread(xpEfficiency) <= 1.12,
    `${map.id} Hunt XP per defeat should remain stable across ranks`);
  check(getRatioSpread(coinEfficiency) <= 1.12,
    `${map.id} Hunt coin efficiency should remain stable across ranks`);
});

// Claiming a hunt should add the advertised XP, coins, and manual, then unlock reacceptance at the next rank.
const rewardEngine = createEngine('greenrootMeadow');
const greenrootMap = data.MAPS.find((map) => map.id === 'greenrootMeadow');
const scoutHunt = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
const rewardPlan = typeof rewardEngine.getMapKillQuestRewardPlan === 'function'
  ? rewardEngine.getMapKillQuestRewardPlan(greenrootMap, {
      completionCount: 1,
      goal: scoutHunt.goal
    })
  : Object.assign({}, rewardEngine.getMapKillQuestRewards(greenrootMap, { goal: scoutHunt.goal }), { manualId: '' });
check(Number(rewardPlan.xp || 0) > 0 &&
  Number(rewardPlan.currency || 0) > 0 &&
  !!rewardPlan.manualId,
'the first Scout Hunt reward plan should include additive XP, coins, and a skill manual');
check(/XP/i.test(scoutHunt.rewardSummary) &&
  /coin/i.test(scoutHunt.rewardSummary) &&
  /manual/i.test(scoutHunt.rewardSummary),
'the Scout Hunt summary should advertise all additive reward components');

markHuntComplete(rewardEngine, greenrootMap.id, scoutHunt.goal);
const xpBeforeClaim = rewardEngine.state.player.xp;
const coinsBeforeClaim = rewardEngine.state.player.currency;
const manualBeforeClaim = getStackCount(rewardEngine, rewardPlan.manualId);
check(rewardEngine.claimMapKillQuestReward(greenrootMap.id),
  'a completed Scout Hunt should be claimable');
check(rewardEngine.state.player.xp === xpBeforeClaim + rewardPlan.xp &&
  rewardEngine.state.player.currency === coinsBeforeClaim + rewardPlan.currency &&
  getStackCount(rewardEngine, rewardPlan.manualId) === manualBeforeClaim + 1,
'claiming the Scout Hunt should grant XP and coins in addition to its manual');

const secondScout = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
const secondPlan = rewardEngine.getMapKillQuestRewardPlan(greenrootMap, {
  completionCount: 2,
  goal: secondScout.goal
});
check(secondScout.rankId === 'scout' &&
  secondScout.completions === 1 &&
  secondScout.goal === scoutHunt.goal &&
  !secondPlan.manualId &&
  !/manual/i.test(secondScout.rewardSummary),
'the second Scout clear should preview XP and coins without inventing a non-milestone manual');
check(rewardEngine.startMapKillQuest(greenrootMap.id),
  'the second Scout Hunt should be reacceptable');
markHuntComplete(rewardEngine, greenrootMap.id, secondScout.goal);
const manualBeforeSecondClear = getStackCount(rewardEngine, rewardPlan.manualId);
check(rewardEngine.claimMapKillQuestReward(greenrootMap.id) &&
  getStackCount(rewardEngine, rewardPlan.manualId) === manualBeforeSecondClear,
'the second Scout claim should match its no-manual preview');

const promotionScout = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
const rangerPromotionPlan = rewardEngine.getMapKillQuestRewardPlan(greenrootMap, {
  completionCount: 3,
  goal: promotionScout.goal
});
check(promotionScout.rankId === 'scout' &&
  promotionScout.completions === 2 &&
  !!rangerPromotionPlan.manualId &&
  /manual/i.test(promotionScout.rewardSummary),
'the third Scout clear should preview the Ranger-promotion manual');
check(rewardEngine.startMapKillQuest(greenrootMap.id),
  'the Ranger-promotion Scout Hunt should be reacceptable');
markHuntComplete(rewardEngine, greenrootMap.id, promotionScout.goal);
const manualBeforeRangerPromotion = getStackCount(rewardEngine, rangerPromotionPlan.manualId);
check(rewardEngine.claimMapKillQuestReward(greenrootMap.id) &&
  getStackCount(rewardEngine, rangerPromotionPlan.manualId) === manualBeforeRangerPromotion + 1,
'the Ranger-promotion claim should deliver the manual advertised in its preview');
const rangerHunt = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
check(rangerHunt.rankId === 'ranger' &&
  rangerHunt.completions === 3 &&
  rangerHunt.goal > scoutHunt.goal &&
  rewardEngine.startMapKillQuest(greenrootMap.id) &&
  rewardEngine.getMapKillQuestSnapshot(greenrootMap.id).active,
'the third clear should unlock a reacceptable, higher-goal Ranger Hunt');

const veteranPromotionState = rewardEngine.getMapKillQuestState()[greenrootMap.id];
veteranPromotionState.completions = 9;
veteranPromotionState.active = false;
veteranPromotionState.progress = 0;
veteranPromotionState.completedAt = 0;
const veteranPromotionHunt = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
const veteranPromotionPlan = rewardEngine.getMapKillQuestRewardPlan(greenrootMap, {
  completionCount: 10,
  goal: veteranPromotionHunt.goal
});
check(veteranPromotionHunt.rankId === 'ranger' &&
  veteranPromotionHunt.completions === 9 &&
  !!veteranPromotionPlan.manualId &&
  /manual/i.test(veteranPromotionHunt.rewardSummary),
'the tenth clear should preview the Veteran-promotion manual');
check(rewardEngine.startMapKillQuest(greenrootMap.id),
  'the Veteran-promotion Ranger Hunt should be reacceptable');
markHuntComplete(rewardEngine, greenrootMap.id, veteranPromotionHunt.goal);
const manualBeforeVeteranPromotion = getStackCount(rewardEngine, veteranPromotionPlan.manualId);
check(rewardEngine.claimMapKillQuestReward(greenrootMap.id) &&
  getStackCount(rewardEngine, veteranPromotionPlan.manualId) === manualBeforeVeteranPromotion + 1,
'the Veteran-promotion claim should deliver the manual advertised in its preview');
const veteranHunt = rewardEngine.getMapKillQuestSnapshot(greenrootMap.id);
check(veteranHunt.rankId === 'veteran' &&
  veteranHunt.maxRank &&
  veteranHunt.completions === 10 &&
  veteranHunt.goal > rangerHunt.goal &&
  rewardEngine.startMapKillQuest(greenrootMap.id) &&
  rewardEngine.getMapKillQuestSnapshot(greenrootMap.id).active,
'the tenth clear should unlock a reacceptable, capped Veteran Hunt');

const fullInventoryEngine = createEngine('greenrootMeadow');
const fullInventoryHunt = fullInventoryEngine.getMapKillQuestSnapshot(greenrootMap.id);
const fullInventoryPlan = fullInventoryEngine.getMapKillQuestRewardPlan(greenrootMap, {
  completionCount: 1,
  goal: fullInventoryHunt.goal
});
const usableCapacity = fullInventoryEngine.getInventoryCapacity('usable');
fullInventoryEngine.state.consumables = Object.fromEntries(
  Array.from({ length: usableCapacity }, (_, index) => [`retention_full_use_${index}`, 1])
);
check(!!fullInventoryPlan.manualId &&
  !fullInventoryEngine.canAddStackableInventoryItem('usable', fullInventoryPlan.manualId, 1),
'the blocked-claim fixture should fill every usable slot before a promised manual');
const fullInventoryState = fullInventoryEngine.getMapKillQuestState()[greenrootMap.id];
fullInventoryState.active = true;
fullInventoryState.progress = fullInventoryHunt.goal;
fullInventoryState.completedAt = Date.now();
const blockedClaimBefore = {
  completions: fullInventoryState.completions,
  progress: fullInventoryState.progress,
  active: fullInventoryState.active,
  completedAt: fullInventoryState.completedAt,
  xp: fullInventoryEngine.state.player.xp,
  currency: fullInventoryEngine.state.player.currency
};
check(!fullInventoryEngine.claimMapKillQuestReward(greenrootMap.id),
  'a hunt claim should block when the inventory cannot accept its advertised manual');
const blockedClaimAfter = fullInventoryEngine.getMapKillQuestState()[greenrootMap.id];
check(blockedClaimAfter.completions === blockedClaimBefore.completions &&
  blockedClaimAfter.progress === blockedClaimBefore.progress &&
  blockedClaimAfter.active === blockedClaimBefore.active &&
  blockedClaimAfter.completedAt === blockedClaimBefore.completedAt &&
  fullInventoryEngine.state.player.xp === blockedClaimBefore.xp &&
  fullInventoryEngine.state.player.currency === blockedClaimBefore.currency &&
  fullInventoryEngine.getMapKillQuestSnapshot(greenrootMap.id).claimable,
'a blocked manual claim should retain the completed hunt and award nothing');

const highCompletionEngine = createEngine('greenrootMeadow');
const highState = highCompletionEngine.getMapKillQuestState()[greenrootMap.id];
highState.completions = 137;
highState.active = true;
highState.completedAt = 0;
const highGoal = highCompletionEngine.getMapKillQuestGoal(greenrootMap.id);
highState.progress = highGoal - 1;
highState.lastCompletedAt = Date.now() - 5000;
const highCompletionSave = highCompletionEngine.serialize();
const restoredHighCompletionEngine = createProjectStarfallEngine(null, data);
check(restoredHighCompletionEngine.restore(highCompletionSave),
  'a high-completion hunt save should restore');
const restoredHighState = restoredHighCompletionEngine.getMapKillQuestState()[greenrootMap.id];
const restoredHighHunt = restoredHighCompletionEngine.getMapKillQuestSnapshot(greenrootMap.id);
check(restoredHighState.completions === 137 &&
  restoredHighState.active &&
  restoredHighState.progress === highGoal - 1 &&
  restoredHighHunt.rankId === 'veteran' &&
  restoredHighHunt.maxRank &&
  restoredHighHunt.goal === highGoal,
'high-completion save/restore should preserve hunt state while keeping the goal capped at Veteran');

if (failures.length) {
  console.error(`Project Starfall retention-loop checks failed: ${failures.length}/${checks}`);
  failures.forEach((message, index) => console.error(`${index + 1}. ${message}`));
  process.exitCode = 1;
} else {
  console.log(`Project Starfall retention-loop checks passed: ${checks}`);
}
