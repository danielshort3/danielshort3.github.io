'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const directiveData = require('../js/games/project-starfall/data/fracture-directives.js');
const seasonEngine = require('../js/games/project-starfall/engine/season.js');
const progressObjectives = require('../js/games/project-starfall/engine/progress-objectives.js');

global.ProjectStarfallData = data;
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');
const season = data.SEASONS.find((entry) => entry && entry.active);
const directives = data.FRACTURE_DIRECTIVES || [];
const referenceNow = Date.UTC(2026, 6, 22, 12, 0, 0);
const objectiveMinutes = Object.freeze({
  defeat: 0.35,
  defeatBoss: 10,
  dungeonComplete: 15
});

function createEventPlan(state, type, payload, nowMs, extraOptions) {
  return seasonEngine.createSeasonEventPlan(state, season, type, payload, Object.assign({
    data,
    nowMs,
    requireDirectiveSelection: true,
    getSeasonObjectivesByType: seasonEngine.getSeasonObjectivesByType,
    matchObjective: progressObjectives.objectiveMatchesEvent,
    getObjectiveKey: progressObjectives.getObjectiveKey,
    getObjectiveGoal: progressObjectives.getObjectiveGoal
  }, extraOptions || {}));
}

function applyEventPlan(state, plan) {
  (plan.changes || []).forEach((change) => {
    state.objectiveValues[change.key] = change.next;
  });
}

function completeSelectedDirective(state, nowMs) {
  const directive = seasonEngine.getSelectedSeasonDirective(state, { data });
  assert(directive, 'a directive should be selected before completing its objectives');
  (directive.objectives || []).forEach((objective) => {
    const payload = {
      count: progressObjectives.getObjectiveGoal(objective),
      mapId: objective.mapId,
      bossId: objective.bossId,
      dungeonId: objective.dungeonId
    };
    const plan = createEventPlan(state, objective.type, payload, nowMs);
    assert.strictEqual(plan.changed, true, `${objective.id} should accept its matching existing event`);
    applyEventPlan(state, plan);
  });
}

assert(season, 'an active season should exist');
assert.strictEqual(directives.length, 3, 'the first directive milestone should expose exactly three choices');
assert.strictEqual(directiveData.FRACTURE_DIRECTIVES, directives,
  'the aggregate data index should expose the authored directive catalog');
assert.strictEqual(new Set(directives.map((directive) => directive.id)).size, directives.length,
  'directive ids should be unique');
assert.strictEqual(new Set(directives.map((directive) => directive.playstyle)).size, directives.length,
  'each first-milestone choice should support a distinct playstyle');
assert.deepStrictEqual(
  Array.from(new Set(directives.flatMap((directive) => directive.objectives.map((objective) => objective.type)))).sort(),
  ['defeat', 'defeatBoss', 'dungeonComplete'],
  'directives should use only the three existing repeatable progress events'
);

const knownAreaIds = new Set((data.WORLD_AREAS || []).map((area) => area.id));
const knownRouteIds = new Set((data.WORLD_ROUTES || []).map((route) => route.id));
const knownMapIds = new Set((data.MAPS || []).map((map) => map.id));
const knownBossIds = new Set((data.BOSS_ENCOUNTERS || []).map((boss) => boss.bossId || boss.id));
const knownDungeonIds = new Set((data.DUNGEONS || []).map((dungeon) => dungeon.id));
directives.forEach((directive) => {
  assert.strictEqual(directive.seasonId, season.id, `${directive.id} should be scoped to its reward season`);
  assert(knownAreaIds.has(directive.areaId), `${directive.id} should reference a real Atlas area`);
  assert(knownRouteIds.has(directive.routeId), `${directive.id} should reference a real Atlas route`);
  (directive.mapIds || []).forEach((mapId) => assert(knownMapIds.has(mapId), `${directive.id} should reference map ${mapId}`));
  (directive.requiredMapIds || []).forEach((mapId) => assert(knownMapIds.has(mapId), `${directive.id} should require real map ${mapId}`));
  (directive.requiredDungeonIds || []).forEach((dungeonId) =>
    assert(knownDungeonIds.has(dungeonId), `${directive.id} should require real dungeon ${dungeonId}`));
  (directive.objectives || []).forEach((objective) => {
    if (objective.mapId) assert(knownMapIds.has(objective.mapId), `${objective.id} should reference a real map`);
    if (objective.bossId) assert(knownBossIds.has(objective.bossId), `${objective.id} should reference a real boss`);
    if (objective.dungeonId) assert(knownDungeonIds.has(objective.dungeonId), `${objective.id} should reference a real dungeon`);
  });
  const estimatedMinutes = (directive.objectives || []).reduce((total, objective) =>
    total + progressObjectives.getObjectiveGoal(objective) * objectiveMinutes[objective.type], 0);
  assert(estimatedMinutes >= 60 && estimatedMinutes <= 90,
    `${directive.id} should fit the 60-90 minute weekly choice window`);
  assert(Math.abs(estimatedMinutes - directive.estimatedMinutes) < 0.01,
    `${directive.id} should disclose its modeled completion time`);
  assert.deepStrictEqual(directive.rewards, season.rewards,
    `${directive.id} should preserve the existing weekly reward envelope`);
  assert.strictEqual(directive.stabilization.visualOnly, true,
    `${directive.id} stabilization should be visual-only`);
  assert.strictEqual(directive.stabilization.maxSeals, 3,
    `${directive.id} stabilization should cap at three seals`);
  assert(!Object.prototype.hasOwnProperty.call(directive.rewards, 'permanentStats'),
    `${directive.id} should not award permanent power`);
  assert(!Object.prototype.hasOwnProperty.call(directive.rewards.materials || {}, 'riftSplinters'),
    `${directive.id} should not add a Rift Splinter faucet`);
});

const root = path.resolve(__dirname, '..');
const entrySource = fs.readFileSync(path.join(root, 'build/entries/project-starfall.entry.js'), 'utf8');
const directiveImportIndex = entrySource.indexOf("data/fracture-directives.js");
assert(directiveImportIndex > entrySource.indexOf("data/commerce.js"),
  'the source bundle should load directive data after the season reward catalog');
assert(directiveImportIndex < entrySource.indexOf("data/index.js"),
  'the source bundle should load directive data before the aggregate index');

const betaOnlySeason = {
  stabilizationByAreaId: {
    rustcoil: 0,
    greenroot: 2
  },
  stabilizationMax: 3
};
const betaOnlyUi = {
  snapshot: { season: betaOnlySeason },
  getFractureStabilizationState: ProjectStarfallUi.prototype.getFractureStabilizationState
};
const betaOnlyStabilization = ProjectStarfallUi.prototype.getFractureStabilizationState.call(
  betaOnlyUi,
  betaOnlySeason,
  null
);
assert.deepStrictEqual(betaOnlyStabilization, { areaId: 'greenroot', label: 'Greenroot relay stabilization', level: 2, max: 3 },
  'the beta overlay should preserve and label a nonzero stabilized area without an active directive or world map snapshot');
assert(ProjectStarfallUi.prototype.renderFractureStabilizationSeals.call(betaOnlyUi, betaOnlySeason, null).includes('Greenroot relay stabilization: 2 of 3 seals'));
const initialState = seasonEngine.createSeasonState(null, { data, nowMs: referenceNow });
assert.strictEqual(initialState.selectedDirectiveId, '');
assert.deepStrictEqual(initialState.stabilizationByAreaId, {});
assert.deepStrictEqual(initialState.stabilizedCycleIds, []);
const underLevelChoices = seasonEngine.getSeasonDirectiveChoices(initialState, season, {
  data,
  nowMs: referenceNow,
  playerLevel: 24
});
assert(underLevelChoices.every((choice) => !choice.eligible && choice.lockedReason.includes('level 25')),
  'directive eligibility should explain the shared level gate');
const availableChoices = seasonEngine.getSeasonDirectiveChoices(initialState, season, {
  data,
  nowMs: referenceNow,
  playerLevel: 25
});
assert.strictEqual(availableChoices.length, 3);
assert(availableChoices.every((choice) => choice.eligible && choice.canSelect),
  'all three playstyle choices should be available at the milestone gate when routes are not constrained');
const futureSeason = Object.freeze({
  id: 'future_fracture_watch',
  active: true,
  cadence: 'weekly',
  objectives: Object.freeze([{ id: 'future_patrol', type: 'defeat', count: 20 }]),
  rewards: Object.freeze({ starTokens: 100 })
});
const futureData = Object.assign({}, data, {
  SEASONS: [futureSeason, Object.assign({}, season, { active: false })]
});
const futureState = seasonEngine.createSeasonState(
  Object.assign({}, initialState, { selectedDirectiveId: 'greenroot_echo_breach' }),
  { data: futureData, nowMs: referenceNow }
);
const futureChoices = seasonEngine.getSeasonDirectiveChoices(futureState, futureSeason, {
  data: futureData,
  nowMs: referenceNow,
  playerLevel: 25
});
assert.deepStrictEqual(futureChoices, [], 'a future season should not inherit another season\'s directive catalog');
assert.strictEqual(futureState.selectedDirectiveId, '',
  'season transition should clear an out-of-scope directive selection');

const noSelectionPlan = createEventPlan(initialState, 'defeat', {
  mapId: 'greenrootMeadow',
  count: 72
}, referenceNow);
assert.strictEqual(noSelectionPlan.changed, false,
  'opted-in directive progress should not start before the player selects a choice');
const legacyPlan = seasonEngine.createSeasonEventPlan(initialState, season, 'defeat', { count: 1 }, {
  data,
  nowMs: referenceNow,
  getSeasonObjectivesByType: seasonEngine.getSeasonObjectivesByType,
  matchObjective: progressObjectives.objectiveMatchesEvent,
  getObjectiveKey: progressObjectives.getObjectiveKey,
  getObjectiveGoal: progressObjectives.getObjectiveGoal
});
assert.strictEqual(legacyPlan.changed, true,
  'callers that have not integrated directives should retain the legacy fixed checklist');
assert.strictEqual(legacyPlan.changes[0].key, 'field_patrol');

const surveySelection = seasonEngine.selectSeasonDirective(
  initialState,
  season,
  'greenroot_relay_survey',
  { data, nowMs: referenceNow, playerLevel: 25 }
);
assert.strictEqual(surveySelection.ok, true);
assert.strictEqual(surveySelection.changed, true);
assert.strictEqual(initialState.selectedDirectiveId, 'greenroot_relay_survey');
const wrongMapPlan = createEventPlan(initialState, 'defeat', { mapId: 'rustcoilRuins', count: 72 }, referenceNow);
assert.strictEqual(wrongMapPlan.changed, false, 'unrelated maps should not advance the selected field survey');
const surveyPlan = createEventPlan(initialState, 'defeat', { mapId: 'greenrootMeadow', count: 12 }, referenceNow);
assert.strictEqual(surveyPlan.changed, true);
assert.strictEqual(surveyPlan.selectedDirectiveId, 'greenroot_relay_survey');
applyEventPlan(initialState, surveyPlan);
assert.strictEqual(initialState.objectiveValues.directive_greenroot_meadow_survey, 12);
const lockedSwitch = seasonEngine.selectSeasonDirective(
  initialState,
  season,
  'greenroot_echo_breach',
  { data, nowMs: referenceNow, playerLevel: 25 }
);
assert.strictEqual(lockedSwitch.ok, false);
assert.strictEqual(lockedSwitch.reason, 'progressLocked', 'selected progress should lock switching for the cycle');
assert.strictEqual(initialState.selectedDirectiveId, 'greenroot_relay_survey');
const idempotentSelection = seasonEngine.selectSeasonDirective(
  initialState,
  season,
  'greenroot_relay_survey',
  { data, nowMs: referenceNow, playerLevel: 25 }
);
assert.strictEqual(idempotentSelection.ok, true);
assert.strictEqual(idempotentSelection.changed, false);

const surveySnapshot = seasonEngine.createSeasonSnapshot(initialState, season, {
  data,
  nowMs: referenceNow,
  requireDirectiveSelection: true,
  playerLevel: 25,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(surveySnapshot.selectedDirectiveId, 'greenroot_relay_survey');
assert.strictEqual(surveySnapshot.objectives.length, 3);
assert.strictEqual(surveySnapshot.objectives[0].value, 12);
assert.strictEqual(surveySnapshot.directiveProgressStarted, true);
assert.strictEqual(surveySnapshot.legacyObjectivesActive, false);

const legacyProgressState = seasonEngine.createSeasonState({
  activeSeasonId: season.id,
  cycleId: initialState.cycleId,
  cycleStartedAt: initialState.cycleStartedAt,
  cycleEndsAt: initialState.cycleEndsAt,
  objectiveValues: { field_patrol: 1 }
}, { data, nowMs: referenceNow });
const legacySelection = seasonEngine.selectSeasonDirective(
  legacyProgressState,
  season,
  'greenroot_echo_breach',
  { data, nowMs: referenceNow, playerLevel: 25 }
);
assert.strictEqual(legacySelection.reason, 'progressLocked',
  'mid-cycle legacy progress should not be silently discarded during integration');

let stabilizationState = seasonEngine.createSeasonState(null, { data, nowMs: referenceNow });
let cycleNow = referenceNow;
for (let expectedSeal = 1; expectedSeal <= 3; expectedSeal += 1) {
  if (expectedSeal > 1) {
    cycleNow = stabilizationState.cycleEndsAt + 1;
    stabilizationState = seasonEngine.createSeasonState(stabilizationState, { data, nowMs: cycleNow });
    assert.strictEqual(stabilizationState.selectedDirectiveId, '', 'weekly rollover should clear directive selection');
    assert.deepStrictEqual(stabilizationState.objectiveValues, {}, 'weekly rollover should clear directive progress');
    assert.strictEqual(stabilizationState.stabilizationByAreaId.greenroot, expectedSeal - 1,
      'weekly rollover should preserve durable visual stabilization');
  }
  const selection = seasonEngine.selectSeasonDirective(
    stabilizationState,
    season,
    'greenroot_echo_breach',
    { data, nowMs: cycleNow, playerLevel: 25 }
  );
  assert.strictEqual(selection.ok, true);
  completeSelectedDirective(stabilizationState, cycleNow);
  const plan = seasonEngine.createSeasonStabilizationPlan(stabilizationState, season, {
    data,
    nowMs: cycleNow,
    createObjectiveStatuses: progressObjectives.createObjectiveStatuses
  });
  assert.strictEqual(plan.changed, true);
  assert.strictEqual(plan.before, expectedSeal - 1);
  assert.strictEqual(plan.next, expectedSeal);
  assert.strictEqual(plan.maxSeals, 3);
  assert.strictEqual(seasonEngine.applySeasonStabilizationPlan(stabilizationState, plan), true);
  assert.strictEqual(seasonEngine.applySeasonStabilizationPlan(stabilizationState, plan), false,
    'the same cycle should never apply stabilization twice');
  assert.strictEqual(stabilizationState.stabilizationByAreaId.greenroot, expectedSeal);
}

const serializedState = JSON.parse(JSON.stringify(stabilizationState));
const restoredState = seasonEngine.createSeasonState(serializedState, { data, nowMs: cycleNow });
assert.strictEqual(restoredState.selectedDirectiveId, 'greenroot_echo_breach');
assert.strictEqual(restoredState.stabilizationByAreaId.greenroot, 3);
assert.strictEqual(restoredState.stabilizedCycleIds.length, 3,
  'directive stabilization history should survive save normalization');
const nextCycleNow = restoredState.cycleEndsAt + 1;
const cappedState = seasonEngine.createSeasonState(restoredState, { data, nowMs: nextCycleNow });
seasonEngine.selectSeasonDirective(cappedState, season, 'greenroot_echo_breach', {
  data,
  nowMs: nextCycleNow,
  playerLevel: 25
});
completeSelectedDirective(cappedState, nextCycleNow);
const cappedPlan = seasonEngine.createSeasonStabilizationPlan(cappedState, season, {
  data,
  nowMs: nextCycleNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(cappedPlan.changed, false);
assert.strictEqual(cappedPlan.reason, 'stabilizationCapped');
assert.strictEqual(cappedState.stabilizationByAreaId.greenroot, 3);
assert(!Object.prototype.hasOwnProperty.call(cappedState, 'permanentStats'),
  'the directive state should never create a permanent-power channel');

const legacyClaimState = seasonEngine.createSeasonState({
  activeSeasonId: season.id,
  objectiveValues: { field_patrol: 60, field_bosses: 2, dungeon_clears: 2 },
  claimedRewardIds: [season.id]
}, { data, nowMs: referenceNow });
assert.strictEqual(legacyClaimState.selectedDirectiveId, '');
assert.deepStrictEqual(legacyClaimState.stabilizationByAreaId, {});
const legacySnapshot = seasonEngine.createSeasonSnapshot(legacyClaimState, season, {
  data,
  nowMs: referenceNow,
  createObjectiveStatuses: progressObjectives.createObjectiveStatuses
});
assert.strictEqual(legacySnapshot.complete, true);
assert.strictEqual(legacySnapshot.rewardClaimed, true,
  'claimed fixed-checklist saves should remain coherent and should not duplicate rewards');
assert.strictEqual(legacySnapshot.objectives.length, season.objectives.length,
  'legacy saves should keep the fixed checklist until the integrated engine opts into selection');

console.log('Project Starfall Fracture Directives tests passed.');
