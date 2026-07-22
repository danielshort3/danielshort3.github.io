'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const seasonEngine = require('../js/games/project-starfall/engine/season.js');
const cashShopEngine = require('../js/games/project-starfall/engine/cash-shop.js');
const progressObjectives = require('../js/games/project-starfall/engine/progress-objectives.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const season = data.SEASONS.find((entry) => entry && entry.active);
assert(season, 'an active season should exist');
assert.strictEqual(season.cadence, 'weekly', 'the active operation should declare a real weekly cadence');
assert.strictEqual(season.resetDayUtc, 1, 'weekly operations should reset on Monday UTC');
assert.strictEqual(season.resetHourUtc, 0, 'weekly operations should reset at 00:00 UTC');
assert((season.objectives || []).every((objective) => ['defeat', 'defeatBoss', 'dungeonComplete'].includes(objective.type)),
  'every weekly objective should be repeatable instead of depending on one-time character progression');
assert(season.firstCompletionRewards && season.firstCompletionRewards.cosmeticId === 'founder_spark',
  'the unique prestige cosmetic should be separated from the recurring weekly cache');
assert(!season.rewards.cosmeticId, 'the renewable reward should not pretend to grant the same unique cosmetic every week');

const referenceNow = Date.UTC(2026, 6, 22, 12, 0, 0);
const referenceStart = Date.UTC(2026, 6, 20, 0, 0, 0);
const referenceEnd = Date.UTC(2026, 6, 27, 0, 0, 0);
const referenceCycle = seasonEngine.getSeasonCycleWindow(season, { nowMs: referenceNow });
assert.strictEqual(referenceCycle.startedAt, referenceStart, 'the cycle should start at Monday 00:00 UTC');
assert.strictEqual(referenceCycle.endsAt, referenceEnd, 'the cycle should end exactly seven days later');
assert.strictEqual(referenceCycle.cycleId, `${season.id}:weekly:${referenceStart}`,
  'the persisted cycle id should be deterministic and season-scoped');
const cashShopWeek = cashShopEngine.getCashShopWeekWindow(referenceNow);
assert.strictEqual(cashShopWeek.startedAt, referenceCycle.startedAt,
  'cash-shop purchase limits and weekly operations should start together');
assert.strictEqual(cashShopWeek.endsAt, referenceCycle.endsAt,
  'cash-shop purchase limits and weekly operations should reset together');
assert.strictEqual(cashShopEngine.getCashShopWeekId(referenceEnd - 1), cashShopWeek.weekId);
assert.strictEqual(cashShopEngine.getCashShopWeekId(referenceEnd), cashShopWeek.weekId + 1,
  'the weekly purchase limiter should roll exactly at Monday 00:00 UTC');

const legacyCashShopWeekId = Math.floor(referenceNow / (7 * 24 * 60 * 60 * 1000));
const migratedCashShop = cashShopEngine.createCashShopState({
  starTokens: 415,
  purchasedItemIds: ['guard_tonic_pack'],
  purchaseWeekId: legacyCashShopWeekId,
  purchaseCountsByWeek: { guard_tonic_pack: 2 }
}, { nowMs: referenceNow });
assert.strictEqual(migratedCashShop.purchaseWeekSchema, 2);
assert.strictEqual(migratedCashShop.purchaseWeekId, cashShopWeek.weekId);
assert.strictEqual(migratedCashShop.purchaseCountsByWeek.guard_tonic_pack, 2,
  'the one-time reset-schema migration should conservatively preserve current legacy limits');
assert.strictEqual(migratedCashShop.starTokens, 415);
assert.deepStrictEqual(migratedCashShop.purchasedItemIds, ['guard_tonic_pack']);
const nextCashShopWeekId = cashShopEngine.syncCashShopPurchaseWeek(migratedCashShop, { nowMs: referenceEnd });
assert.strictEqual(nextCashShopWeekId, cashShopWeek.weekId + 1);
assert.deepStrictEqual(migratedCashShop.purchaseCountsByWeek, {},
  'Monday rollover should clear only weekly counts while keeping durable shop state');
assert.strictEqual(migratedCashShop.starTokens, 415);
assert.deepStrictEqual(migratedCashShop.purchasedItemIds, ['guard_tonic_pack']);
const cashShopSnapshot = cashShopEngine.createCashShopSnapshot(migratedCashShop, { cosmetics: [] }, {
  data,
  nowMs: referenceEnd
});
assert.strictEqual(cashShopSnapshot.cadenceLabel, 'Weekly purchase limits');
assert.strictEqual(cashShopSnapshot.resetScheduleLabel, 'Monday 00:00 UTC');
assert.strictEqual(cashShopSnapshot.resetAt, referenceEnd + 7 * 24 * 60 * 60 * 1000);

const futureNow = referenceEnd + 24 * 60 * 60 * 1000;
const futureWeek = cashShopEngine.getCashShopWeekWindow(futureNow);
const rollbackLimitedCashShop = cashShopEngine.createCashShopState({
  starTokens: 415,
  purchasedItemIds: ['guard_tonic_pack'],
  purchaseWeekSchema: 2,
  purchaseWeekId: futureWeek.weekId,
  purchaseCountsByWeek: { guard_tonic_pack: 2 }
}, { nowMs: futureNow });
assert.strictEqual(cashShopEngine.syncCashShopPurchaseWeek(rollbackLimitedCashShop, { nowMs: referenceNow }), futureWeek.weekId,
  'clock rollback should keep the newest stored limiter week authoritative');
assert.strictEqual(rollbackLimitedCashShop.purchaseWeekId, futureWeek.weekId);
assert.strictEqual(rollbackLimitedCashShop.purchaseCountsByWeek.guard_tonic_pack, 2,
  'future-to-prior clock movement should not clear weekly purchase counts');
const normalizedRollbackLimitedCashShop = cashShopEngine.createCashShopState(rollbackLimitedCashShop, { nowMs: referenceNow });
assert.strictEqual(normalizedRollbackLimitedCashShop.purchaseWeekId, futureWeek.weekId,
  'save normalization during clock rollback should remain monotonic');
assert.strictEqual(normalizedRollbackLimitedCashShop.purchaseCountsByWeek.guard_tonic_pack, 2);
const rollbackCashShopSnapshot = cashShopEngine.createCashShopSnapshot(
  normalizedRollbackLimitedCashShop,
  { cosmetics: [] },
  { data, nowMs: referenceNow }
);
const rollbackGuardPack = rollbackCashShopSnapshot.items.find((item) => item.id === 'guard_tonic_pack');
assert(rollbackGuardPack && rollbackGuardPack.purchaseCount === 2 && rollbackGuardPack.remainingPurchases === 1,
  'a rollback-time snapshot should preserve the active purchase limit');
assert.strictEqual(rollbackCashShopSnapshot.resetAt, futureWeek.endsAt,
  'rollback-time reset messaging should describe the monotonic stored week');
assert.strictEqual(cashShopEngine.syncCashShopPurchaseWeek(normalizedRollbackLimitedCashShop, { nowMs: futureNow }), futureWeek.weekId);
assert.strictEqual(normalizedRollbackLimitedCashShop.purchaseCountsByWeek.guard_tonic_pack, 2,
  'returning from prior time to the stored future week should not grant a second reset');
assert.strictEqual(cashShopEngine.syncCashShopPurchaseWeek(normalizedRollbackLimitedCashShop, { nowMs: futureWeek.endsAt }), futureWeek.weekId + 1);
assert.deepStrictEqual(normalizedRollbackLimitedCashShop.purchaseCountsByWeek, {},
  'counts should clear exactly once when time advances beyond the newest stored week');

const rollbackCashShopEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(rollbackCashShopEngine.chooseClass('fighter'), true);
rollbackCashShopEngine.state.cashShop = {
  starTokens: 415,
  purchasedItemIds: ['guard_tonic_pack'],
  purchaseWeekSchema: 2,
  purchaseWeekId: futureWeek.weekId,
  purchaseCountsByWeek: { guard_tonic_pack: 2 }
};
const rollbackEngineSnapshot = rollbackCashShopEngine.getCashShopSnapshot({ nowMs: referenceNow });
const rollbackEngineGuardPack = rollbackEngineSnapshot.items.find((item) => item.id === 'guard_tonic_pack');
assert(rollbackEngineGuardPack && rollbackEngineGuardPack.purchaseCount === 2 && rollbackEngineGuardPack.remainingPurchases === 1,
  'the integrated engine snapshot should not clear limits when the clock moves to a prior week');
rollbackCashShopEngine.getCashShopState({ nowMs: futureNow });
assert.strictEqual(rollbackCashShopEngine.state.cashShop.purchaseCountsByWeek.guard_tonic_pack, 2,
  'the integrated engine should preserve limits when the clock returns to the newest stored week');
rollbackCashShopEngine.getCashShopState({ nowMs: futureWeek.endsAt });
assert.deepStrictEqual(rollbackCashShopEngine.state.cashShop.purchaseCountsByWeek, {},
  'the integrated engine should clear limits at the next genuine forward boundary');

const legacyState = seasonEngine.createSeasonState({
  activeSeasonId: season.id,
  objectiveValues: { field_patrol: 60, field_bosses: 2, dungeon_clears: 2 },
  claimedRewardIds: [season.id]
}, { data, nowMs: referenceNow });
assert.deepStrictEqual(legacyState.objectiveValues, { field_patrol: 60, field_bosses: 2, dungeon_clears: 2 },
  'legacy progress should remain intact in its migration cycle');
assert(legacyState.claimedCycleIds.includes(referenceCycle.cycleId),
  'a legacy lifetime claim should migrate into the current cycle instead of granting a duplicate reward');
assert.strictEqual(legacyState.totalRewardsClaimed, 1, 'legacy claim history should preserve first-clear ownership');
const legacySnapshot = seasonEngine.createSeasonSnapshot(legacyState, season, {
  data,
  nowMs: referenceNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(legacySnapshot.rewardClaimed, true);
assert.strictEqual(legacySnapshot.firstCompletionRewardAvailable, false);
assert.strictEqual(legacySnapshot.resetScheduleLabel, 'Monday 00:00 UTC');
assert(legacySnapshot.resetLabel.startsWith('Resets in '), 'the player-facing snapshot should expose reset timing');
const preWeeklyClaim = seasonEngine.createSeasonState({
  activeSeasonId: season.id,
  objectiveValues: { field_bosses: 2, dungeon_clears: 2, advanced_path: 1 },
  claimedRewardIds: [season.id]
}, { data, nowMs: referenceNow });
const preWeeklySnapshot = seasonEngine.createSeasonSnapshot(preWeeklyClaim, season, {
  data,
  nowMs: referenceNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(preWeeklySnapshot.complete, true,
  'a claimed pre-weekly save should migrate to a coherent completed-and-claimed current-cycle display');
assert.strictEqual(preWeeklyClaim.objectiveValues.advanced_path, 1,
  'migration should preserve historical objective evidence even when the weekly objective set changes');

const nextCycleNow = referenceEnd + 1;
const rolledState = seasonEngine.createSeasonState(legacyState, { data, nowMs: nextCycleNow });
assert.notStrictEqual(rolledState.cycleId, legacyState.cycleId, 'crossing the reset boundary should create a new cycle');
assert.deepStrictEqual(rolledState.objectiveValues, {}, 'weekly objective progress should reset at the boundary');
assert.strictEqual(rolledState.totalRewardsClaimed, 1, 'weekly reset should preserve lifetime completion history');
const rolledSnapshot = seasonEngine.createSeasonSnapshot(rolledState, season, {
  data,
  nowMs: nextCycleNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(rolledSnapshot.rewardClaimed, false, 'the new weekly cache should be claimable again');
assert.strictEqual(rolledSnapshot.firstCompletionRewardAvailable, false,
  'the one-time prestige reward should not reset with the renewable cache');

const futureSeason = Object.freeze({
  id: 'fracture_watch_two',
  name: 'Fracture Watch II',
  active: true,
  cadence: 'weekly',
  resetDayUtc: 1,
  resetHourUtc: 0,
  objectives: Object.freeze([{ id: 'field_patrol_two', type: 'defeat', count: 20 }]),
  rewards: Object.freeze({ starTokens: 100 }),
  firstCompletionRewards: Object.freeze({ cosmeticId: 'constellation_trail' })
});
const transitionedData = Object.assign({}, data, {
  SEASONS: [futureSeason, Object.assign({}, season, { active: false })]
});
const transitionedState = seasonEngine.createSeasonState(rolledState, { data: transitionedData, nowMs: nextCycleNow });
const transitionedSnapshot = seasonEngine.createSeasonSnapshot(transitionedState, futureSeason, {
  data: transitionedData,
  nowMs: nextCycleNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(transitionedState.activeSeasonId, futureSeason.id, 'an inactive saved season should hand off to the live season');
assert.strictEqual(transitionedSnapshot.seasonRewardsClaimed, 0,
  'reward history should be tracked per season rather than suppressing future first-clear rewards');
assert.strictEqual(transitionedSnapshot.firstCompletionRewardAvailable, true);
assert.strictEqual(transitionedSnapshot.directiveChoices.length, 0,
  'a future season should not display Greenroot directives tied to the prior reward envelope');
assert.strictEqual(transitionedSnapshot.directiveSelectionRequired, false,
  'a season without authored directives should continue using its own objective set');
assert.strictEqual(transitionedSnapshot.objectives.length, futureSeason.objectives.length);

rolledState.objectiveValues.field_patrol = 17;
const rollbackSafeState = seasonEngine.createSeasonState(rolledState, { data, nowMs: referenceNow });
assert.strictEqual(rollbackSafeState.cycleId, rolledState.cycleId,
  'moving the local clock backward should not reopen an earlier reward cycle');
assert.strictEqual(rollbackSafeState.objectiveValues.field_patrol, 17,
  'clock rollback protection should retain the newest known cycle progress');

const legacyRestoreSource = createProjectStarfallEngine(null, data);
assert.strictEqual(legacyRestoreSource.chooseClass('fighter'), true);
const legacyPayload = legacyRestoreSource.serialize();
legacyPayload.state.season = {
  activeSeasonId: season.id,
  objectiveValues: { field_patrol: 60, field_bosses: 2, dungeon_clears: 2 },
  claimedRewardIds: [season.id]
};
const legacyRestoreTarget = createProjectStarfallEngine(null, data);
assert.strictEqual(legacyRestoreTarget.restore(legacyPayload), true, 'a lifetime-season save should restore');
const restoredLegacySnapshot = legacyRestoreTarget.getSeasonSnapshot();
assert.strictEqual(restoredLegacySnapshot.rewardClaimed, true,
  'restoring a claimed legacy save should not duplicate the current weekly reward');
assert.strictEqual(restoredLegacySnapshot.complete, true, 'restoring should preserve legacy objective progress');
assert.strictEqual(legacyRestoreTarget.state.season.totalRewardsClaimed, 1);
const liveNow = Date.now();
legacyPayload.state.cashShop = {
  starTokens: 415,
  purchasedItemIds: ['guard_tonic_pack'],
  purchaseWeekId: Math.floor(liveNow / (7 * 24 * 60 * 60 * 1000)),
  purchaseCountsByWeek: { guard_tonic_pack: 2 }
};
const legacyCashShopRestore = createProjectStarfallEngine(null, data);
assert.strictEqual(legacyCashShopRestore.restore(legacyPayload), true, 'a legacy weekly-limit save should restore');
assert.strictEqual(legacyCashShopRestore.state.cashShop.purchaseWeekSchema, 2);
assert.strictEqual(legacyCashShopRestore.state.cashShop.purchaseCountsByWeek.guard_tonic_pack, 2);
assert.strictEqual(legacyCashShopRestore.state.cashShop.starTokens, 415);
assert.deepStrictEqual(legacyCashShopRestore.state.cashShop.purchasedItemIds, ['guard_tonic_pack']);

const engine = createProjectStarfallEngine(null, data);
assert.strictEqual(engine.chooseClass('fighter'), true, 'the recurring-loop test character should initialize');
engine.state.player.level = 25;
const fighterAdvancedId = Object.keys(data.ADVANCED_CLASSES || {})
  .find((advancedId) => data.ADVANCED_CLASSES[advancedId].baseClass === 'fighter');
assert(fighterAdvancedId, 'the test character should have an eligible advanced branch');
engine.state.player.advancedClassId = fighterAdvancedId;
const forestRoute = data.WORLD_ROUTES.find((route) => route.id === 'forest');
assert(forestRoute, 'the Greenroot directive route should exist');
(forestRoute.fieldGoals || []).forEach((field) => {
  engine.state.routeProgress.forest.killsByMap[field.mapId] = field.count;
});
const currentNow = Date.now();
const currentCycle = seasonEngine.getSeasonCycleWindow(season, { nowMs: currentNow });
engine.state.season = seasonEngine.createSeasonState(null, { data, nowMs: currentNow });
const directiveChoiceSnapshot = engine.getSeasonSnapshot({ nowMs: currentNow });
assert.strictEqual(directiveChoiceSnapshot.directiveSelectionRequired, true);
assert.strictEqual(directiveChoiceSnapshot.directiveChoices.length, 3);
assert(directiveChoiceSnapshot.directiveChoices.every((choice) => choice.canSelect),
  'a qualified player should receive three equally eligible weekly approaches');
assert.strictEqual(
  engine.recordSeasonEvent('defeatBoss', { bossId: 'brambleking', count: 7 }, { nowMs: currentNow }),
  false,
  'weekly progress should wait for an explicit directive choice'
);
assert.strictEqual(engine.selectSeasonDirective('greenroot_echo_breach', { nowMs: currentNow }), true);
const directiveAtlas = engine.getWorldMapSnapshot();
assert.strictEqual(directiveAtlas.directive.id, 'greenroot_echo_breach');
assert.deepStrictEqual(
  directiveAtlas.nodes.filter((node) => node.directiveRoute).map((node) => node.mapId),
  ['greenrootMeadow', 'thornpathThicket', 'banditRidgeCamp', 'brambleDepths'],
  'the chosen operation should mark its authored route on the Atlas'
);
assert.strictEqual(directiveAtlas.edges.filter((edge) => edge.directiveRoute).length, 3,
  'the chosen operation route should form a readable contiguous lane');
assert.strictEqual(
  engine.recordSeasonEvent('defeatBoss', { bossId: 'brambleking', count: 7 }, { nowMs: currentNow }),
  true
);
assert.strictEqual(engine.getSeasonSnapshot({ nowMs: currentNow }).complete, true,
  'the selected repeatable approach should complete the weekly operation');
const tokensBeforeFirstClaim = engine.getCashShopState().starTokens;
assert.strictEqual(engine.claimSeasonReward({ nowMs: currentNow }), true, 'the completed weekly cache should claim once');
assert.strictEqual(engine.claimSeasonReward({ nowMs: currentNow }), false, 'the same weekly cache should not claim twice');
assert.strictEqual(engine.getCashShopState().starTokens, tokensBeforeFirstClaim + season.rewards.starTokens);
assert(engine.state.cosmetics.unlockedIds.includes('founder_spark'), 'the first weekly clear should unlock Founder Spark');
assert.strictEqual(engine.state.season.stabilizationByAreaId.greenroot, 1,
  'the first directive clear should leave one visual-only Atlas stabilization seal');
const stabilizedAtlas = engine.getWorldMapSnapshot();
assert.strictEqual(stabilizedAtlas.areas.find((area) => area.id === 'greenroot').stabilizationLevel, 1,
  'claimed stabilization should be visible in the Atlas snapshot');
const claimedPayload = engine.serialize();
const claimedRestore = createProjectStarfallEngine(null, data);
assert.strictEqual(claimedRestore.restore(claimedPayload), true, 'a cycle-aware season save should restore');
assert.strictEqual(claimedRestore.getSeasonSnapshot({ nowMs: currentNow }).rewardClaimed, true,
  'a current-cycle claim should survive save and restore');
assert.strictEqual(claimedRestore.state.season.totalRewardsClaimed, 1);
assert.strictEqual(claimedRestore.state.season.stabilizationByAreaId.greenroot, 1,
  'Atlas stabilization should survive save and restore');

const secondCycleNow = currentCycle.endsAt + 1;
const secondSnapshot = engine.getSeasonSnapshot({ nowMs: secondCycleNow });
assert.strictEqual(secondSnapshot.rewardClaimed, false, 'the next Monday should reopen the renewable reward');
assert.strictEqual(secondSnapshot.complete, false, 'the next Monday should clear weekly objective progress');
assert.strictEqual(secondSnapshot.totalRewardsClaimed, 1);
assert.strictEqual(secondSnapshot.selectedDirectiveId, '', 'the next cycle should request a fresh approach');
assert.strictEqual(engine.selectSeasonDirective('greenroot_echo_breach', { nowMs: secondCycleNow }), true);
assert.strictEqual(
  engine.recordSeasonEvent('defeatBoss', { bossId: 'brambleking', count: 7 }, { nowMs: secondCycleNow }),
  true
);
assert.strictEqual(engine.claimSeasonReward({ nowMs: secondCycleNow }), true,
  'the fully completed operation should claim again in the next cycle');
assert.strictEqual(engine.getCashShopState().starTokens, tokensBeforeFirstClaim + season.rewards.starTokens * 2,
  'recurring rewards should be granted once per completed weekly cycle');
assert.strictEqual(engine.state.season.totalRewardsClaimed, 2);
assert.strictEqual(engine.state.season.stabilizationByAreaId.greenroot, 2,
  'successive weekly clears should build durable visual Atlas history without power');
assert.strictEqual(engine.state.season.claimedCycleIds.length, 2);
assert.strictEqual(engine.state.season.rewardCountsBySeason[season.id], 2,
  'per-season claim history should distinguish renewable clears from future season prestige');

const rolloverEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(rolloverEngine.chooseClass('fighter'), true);
rolloverEngine.state.player.level = 25;
rolloverEngine.state.season = seasonEngine.createSeasonState(null, { data, nowMs: referenceNow });
const originalDateNow = Date.now;
try {
  Date.now = () => referenceNow;
  const rolloverSelection = seasonEngine.selectSeasonDirective(
    rolloverEngine.state.season, season, 'greenroot_echo_breach',
    { data, nowMs: referenceNow, playerLevel: 25 }
  );
  assert.strictEqual(rolloverSelection.ok, true);
  assert.strictEqual(rolloverEngine.getWorldMapSnapshot().directive.id, 'greenroot_echo_breach');
  const cachedOverlay = rolloverEngine.getOverlaySnapshot({ openPanels: ['beta', 'worldmap'] });
  assert.strictEqual(cachedOverlay.season.selectedDirectiveId, 'greenroot_echo_breach');
  Date.now = () => referenceEnd + 1;
  const rolledAtlas = rolloverEngine.getWorldMapSnapshot();
  assert.strictEqual(rolledAtlas.directive, null,
    'an Atlas read after weekly rollover should not retain the prior cycle directive');
  const rolledOverlay = rolloverEngine.getOverlaySnapshot({ openPanels: ['beta', 'worldmap'] });
  assert.notStrictEqual(rolledOverlay, cachedOverlay,
    'the live overlay cache should invalidate when the weekly cycle changes');
  assert.strictEqual(rolledOverlay.season.selectedDirectiveId, '');
  assert.strictEqual(rolledOverlay.worldMap.directive, null);
} finally {
  Date.now = originalDateNow;
}

const map = data.MAPS.find((entry) => entry.id === 'greenrootMeadow');
assert(map, 'Starfall Verge should exist for hunt-scaling tests');
const baseGoal = engine.getMapKillQuestBaseGoal(map);
const expectedGoals = [0, 1, 2, 3, 4, 5, 20, 1000].map((completions) =>
  progressObjectives.getMapKillQuestGoal(baseGoal, completions));
assert.deepStrictEqual(expectedGoals.slice(0, 6), [
  baseGoal,
  Math.ceil(baseGoal * 1.1 - 1e-9),
  Math.ceil(baseGoal * 1.2 - 1e-9),
  Math.ceil(baseGoal * 1.3 - 1e-9),
  Math.ceil(baseGoal * 1.4 - 1e-9),
  Math.ceil(baseGoal * 1.5 - 1e-9)
], 'repeat hunts should ramp by a readable ten percent mastery step');
assert.strictEqual(expectedGoals[5], expectedGoals[6], 'hunt scaling should stop after five completions');
assert.strictEqual(expectedGoals[6], expectedGoals[7], 'even extreme legacy completion counts should stay at the cap');
assert(expectedGoals.every((goal) => goal <= Math.ceil(baseGoal * 1.5)),
  'no repeatable hunt should exceed 150 percent of its authored base objective');

engine.state.mapKillQuests.greenrootMeadow.completions = 1000;
const cappedProfile = engine.getMapKillQuestGoalProfile('greenrootMeadow');
assert.strictEqual(cappedProfile.capped, true);
assert.strictEqual(cappedProfile.masteryTier, 5);
assert.strictEqual(engine.getMapKillQuestGoal('greenrootMeadow'), Math.ceil(baseGoal * 1.5 - 1e-9));
const cappedSummary = engine.getMapKillQuestSnapshot('greenrootMeadow');
assert.strictEqual(cappedSummary.scalingCapped, true);
assert.strictEqual(cappedSummary.maxMasteryTier, 5);
const baseRewards = engine.getMapKillQuestRewards(map, { goal: baseGoal });
const cappedRewards = engine.getMapKillQuestRewards(map, { goal: cappedProfile.goal });
assert(cappedRewards.xp > baseRewards.xp && cappedRewards.currency >= baseRewards.currency,
  'the capped increase in effort should retain proportional XP and currency value');

const normalizedLegacyHunts = progressObjectives.createMapKillQuestState({
  greenrootMeadow: { active: true, progress: 12, completions: 1000, completedAt: 0, lastCompletedAt: 123 }
}, { data });
assert.strictEqual(normalizedLegacyHunts.greenrootMeadow.progress, 12);
assert.strictEqual(normalizedLegacyHunts.greenrootMeadow.completions, 1000,
  'migration should preserve legitimate historical hunt completions even though their goal is now capped');

const root = path.resolve(__dirname, '..');
const engineSource = fs.readFileSync(path.join(root, 'js/games/project-starfall/project-starfall-engine.js'), 'utf8');
const uiSource = fs.readFileSync(path.join(root, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(!/base\s*\*\s*Math\.pow\(1\.5,\s*Math\.max\(0,\s*Number\(state\.completions/.test(engineSource),
  'the former uncapped exponential hunt formula should not remain in runtime code');
assert(uiSource.includes('season.cadenceLabel') && uiSource.includes('season.resetScheduleLabel') && uiSource.includes('Weekly reward:'),
  'the season panel should disclose cadence, reset timing, and the renewable reward');
assert(uiSource.includes('cashShop.cadenceLabel') && uiSource.includes('cashShop.resetScheduleLabel'),
  'the cash-shop panel should disclose the same weekly reset schedule');

console.log('Project Starfall recurring loop tests passed.');
