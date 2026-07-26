'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/data/index.js');
const weeklyRoutes = require('../js/games/project-starfall/engine/weekly-routes.js');
const questUi = require('../js/games/project-starfall/ui/quests.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const DAY_MS = 24 * 60 * 60 * 1000;
const WEEK_MS = 7 * DAY_MS;
const WEEK_ONE_MS = Date.UTC(2026, 6, 20, 0, 0, 0);
const WEEK_TWO_MS = WEEK_ONE_MS + WEEK_MS;
const CONFIG = data.WEEKLY_STAR_ROUTES_CONFIG;

let checks = 0;

function check(condition, message) {
  checks += 1;
  assert.ok(condition, message);
}

function checkEqual(actual, expected, message) {
  checks += 1;
  assert.strictEqual(actual, expected, message);
}

function checkDeepEqual(actual, expected, message) {
  checks += 1;
  assert.deepStrictEqual(actual, expected, message);
}

function createCandidateFixture() {
  return {
    field: [
      { mapId: 'greenrootMeadow', name: 'Greenroot Meadow Hunt', summary: 'Complete the Greenroot hunt.', guideType: 'mapKill', guideId: 'greenrootMeadow' },
      { mapId: 'thornpathThicket', name: 'Thornpath Thicket Hunt', summary: 'Complete the Thornpath hunt.', guideType: 'mapKill', guideId: 'thornpathThicket' },
      { mapId: 'rustcoilRuins', name: 'Rustcoil Ruins Hunt', summary: 'Complete the Rustcoil hunt.', guideType: 'mapKill', guideId: 'rustcoilRuins' },
      { mapId: 'orebackQuarry', name: 'Oreback Quarry Hunt', summary: 'Complete the Oreback hunt.', guideType: 'mapKill', guideId: 'orebackQuarry' },
      { mapId: 'cinderHollow', name: 'Cinder Hollow Hunt', summary: 'Complete the Cinder hunt.', guideType: 'mapKill', guideId: 'cinderHollow' },
      { mapId: 'frostfenOutskirts', name: 'Frostfen Outskirts Hunt', summary: 'Complete the Frostfen hunt.', guideType: 'mapKill', guideId: 'frostfenOutskirts' }
    ],
    mechanic: [
      { mapMechanicId: 'greenroot_bloom', mapId: 'greenrootMeadow', name: 'Greenroot Bloom Cycle', summary: 'Complete a bloom cycle.', guideType: 'map', guideId: 'greenrootMeadow' },
      { mapMechanicId: 'rustcoil_overdrive', mapId: 'rustcoilRuins', name: 'Rustcoil Overdrive', summary: 'Complete an overdrive cycle.', guideType: 'map', guideId: 'rustcoilRuins' },
      { mapMechanicId: 'cinder_pressure', mapId: 'cinderHollow', name: 'Cinder Pressure Cycle', summary: 'Complete a pressure cycle.', guideType: 'map', guideId: 'cinderHollow' }
    ],
    dungeon: [
      { dungeonId: 'bramble_depths', mapId: 'brambleDepths', name: 'Bramble Depths', summary: 'Clear Bramble Depths.', guideType: 'dungeon', guideId: 'bramble_depths' },
      { dungeonId: 'gearworks_vault', mapId: 'gearworksVault', name: 'Gearworks Vault', summary: 'Clear Gearworks Vault.', guideType: 'dungeon', guideId: 'gearworks_vault' },
      { dungeonId: 'emberjaw_lair', mapId: 'emberjawLair', name: 'Emberjaw Lair', summary: 'Clear Emberjaw Lair.', guideType: 'dungeon', guideId: 'emberjaw_lair' },
      { dungeonId: 'rimewarden_sanctum', mapId: 'rimewardenSanctum', name: 'Rimewarden Sanctum', summary: 'Clear Rimewarden Sanctum.', guideType: 'dungeon', guideId: 'rimewarden_sanctum' }
    ]
  };
}

function reverseCandidateFixture(candidates) {
  return Object.fromEntries(Object.entries(candidates).map(([key, values]) => [key, values.slice().reverse()]));
}

function createUnlockedSeasonState(weeklyState) {
  return {
    activeSeasonId: 'beta_foundations',
    objectiveValues: {},
    claimedRewardIds: ['beta_foundations'],
    weeklyRoutes: weeklyState || weeklyRoutes.createWeeklyRouteState(null, { config: CONFIG })
  };
}

function initializeWeek(nowMs, options) {
  const candidates = options && options.candidates || createCandidateFixture();
  const seasonState = options && options.seasonState || createUnlockedSeasonState();
  return weeklyRoutes.reconcileWeeklyRouteState(seasonState, candidates, {
    config: CONFIG,
    nowMs,
    seasonState
  });
}

function payloadForAssignment(assignment, eventKey, count) {
  const payload = {
    eventKey,
    count: count == null ? 1 : count
  };
  if (assignment.kind === 'mapHunt') {
    payload.mapId = assignment.targetId;
  } else if (assignment.kind === 'mapMechanic') {
    payload.mapMechanicId = assignment.targetId;
    payload.mapId = assignment.mapId;
  } else if (assignment.kind === 'dungeon') {
    payload.dungeonId = assignment.targetId;
    payload.mapId = assignment.mapId;
    payload.runId = `run_${eventKey}`;
  }
  return payload;
}

function applyAssignmentEvent(state, assignment, eventKey, options) {
  const settings = options || {};
  return weeklyRoutes.createWeeklyRouteEventPlan(
    state,
    assignment.type,
    payloadForAssignment(assignment, eventKey, settings.count),
    {
      config: settings.config || CONFIG,
      candidates: settings.candidates || createCandidateFixture(),
      nowMs: settings.nowMs == null ? WEEK_ONE_MS + DAY_MS : settings.nowMs,
      unlocked: settings.unlocked == null ? true : settings.unlocked
    }
  );
}

// Weekly reset calculations must be pure, UTC-based, and stable across the boundary.
checkEqual(
  weeklyRoutes.getWeeklyRouteWeekStartMs(WEEK_ONE_MS),
  WEEK_ONE_MS,
  'Monday 00:00 UTC should be the exact weekly route boundary'
);
checkEqual(
  weeklyRoutes.getWeeklyRouteWeekStartMs(WEEK_ONE_MS - 1),
  WEEK_ONE_MS - WEEK_MS,
  'the final millisecond before Monday should belong to the previous UTC week'
);
checkEqual(
  weeklyRoutes.getWeeklyRouteWeekStartMs(WEEK_ONE_MS + WEEK_MS - 1),
  WEEK_ONE_MS,
  'the final millisecond before reset should remain in the current UTC week'
);
checkEqual(
  weeklyRoutes.getWeeklyRouteWeekStartMs(WEEK_TWO_MS),
  WEEK_TWO_MS,
  'the reset boundary should advance the week exactly once'
);
checkEqual(
  weeklyRoutes.getWeeklyRouteWeekKey(WEEK_ONE_MS + 3 * DAY_MS),
  '2026-07-20',
  'the weekly route key should use the Monday UTC calendar date'
);
checkEqual(
  weeklyRoutes.getWeeklyRouteResetAt(WEEK_ONE_MS + 3 * DAY_MS),
  WEEK_TWO_MS,
  'resetAt should be the next Monday 00:00 UTC'
);

// Assignment selection must be deterministic and independent of candidate input order.
const candidates = createCandidateFixture();
const assignments = weeklyRoutes.createWeeklyRouteAssignments(candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS
});
const repeatedAssignments = weeklyRoutes.createWeeklyRouteAssignments(candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS + 5 * DAY_MS
});
const reorderedAssignments = weeklyRoutes.createWeeklyRouteAssignments(reverseCandidateFixture(candidates), {
  config: CONFIG,
  nowMs: WEEK_ONE_MS
});
checkDeepEqual(repeatedAssignments, assignments,
  'every timestamp in one week should resolve to the same assignments');
checkDeepEqual(reorderedAssignments, assignments,
  'candidate input order should not change deterministic weekly assignments');
checkEqual(assignments.length, 4,
  'a complete candidate pool should fill all four weekly slots');
checkDeepEqual(assignments.map((assignment) => assignment.slotId), ['field_a', 'field_b', 'challenge', 'dungeon'],
  'weekly assignments should retain the configured presentation order');
checkDeepEqual(assignments.map((assignment) => assignment.kind), ['mapHunt', 'mapHunt', 'mapMechanic', 'dungeon'],
  'the normal rotation should include two hunts, one mechanic, and one dungeon');
check(assignments.every((assignment) =>
  assignment.summary &&
  assignment.guideType &&
  assignment.guideId),
'assignment generation should preserve the summary and guidance metadata required by the live UI');
checkEqual(new Set(assignments.map((assignment) => assignment.id)).size, assignments.length,
  'weekly assignment ids should be unique');
check(assignments[0].mapId !== assignments[1].mapId,
  'the two field slots should select distinct maps');
checkEqual(new Set(assignments.map((assignment) => `${assignment.kind}:${assignment.targetId}`)).size, assignments.length,
  'the rotation should not repeat a target within the same objective kind');

const weeklySignatures = new Set(Array.from({ length: 12 }, (_, index) => {
  return weeklyRoutes.createWeeklyRouteAssignments(candidates, {
    config: CONFIG,
    nowMs: WEEK_ONE_MS + index * WEEK_MS
  }).map((assignment) => assignment.id).join('|');
}));
check(weeklySignatures.size > 1,
  'deterministic assignment selection should still rotate content across weeks');

const fallbackAssignments = weeklyRoutes.createWeeklyRouteAssignments({
  field: candidates.field,
  mechanic: [],
  dungeon: []
}, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS
});
checkEqual(fallbackAssignments.length, 4,
  'hunt fallbacks should keep the weekly route playable when mechanic and dungeon pools are empty');
check(fallbackAssignments.every((assignment) => assignment.kind === 'mapHunt'),
  'challenge and dungeon slots should use their configured hunt fallback');
checkEqual(new Set(fallbackAssignments.map((assignment) => assignment.mapId)).size, 4,
  'fallback hunts should remain distinct when enough maps are available');

// Saved state normalization must reject stale content, dedupe slots/events, and avoid aliases.
const firstAssignment = assignments[0];
const normalizationInput = {
  unlocked: true,
  weekStartMs: WEEK_ONE_MS + 2 * DAY_MS,
  assignments: [
    Object.assign({}, firstAssignment, { goal: 3 }),
    Object.assign({}, assignments[1]),
    Object.assign({}, assignments[1], { targetId: 'duplicate_slot_target' }),
    { slotId: 'unknown_slot', kind: 'mapHunt', targetId: 'greenrootMeadow', goal: 1 },
    { slotId: 'challenge', kind: 'unknown_kind', targetId: 'bad', goal: 1 }
  ],
  objectiveValues: {
    [firstAssignment.id]: 99,
    [assignments[1].id]: -4,
    unknown_objective: 100
  },
  creditedEventKeys: Array.from({ length: 40 }, (_, index) => `event_${index}`).concat('event_39'),
  rewardGrantedWeekStartMs: WEEK_ONE_MS + 4 * DAY_MS,
  completedWeekCount: 7.8
};
const normalized = weeklyRoutes.createWeeklyRouteState(normalizationInput, { config: CONFIG });
checkEqual(normalized.weekStartMs, WEEK_ONE_MS,
  'saved timestamps should normalize to their Monday UTC boundary');
checkEqual(normalized.rewardGrantedWeekStartMs, WEEK_ONE_MS,
  'saved reward timestamps should normalize to their Monday UTC boundary');
checkEqual(normalized.assignments.length, 2,
  'normalization should keep one valid assignment per configured slot');
checkEqual(normalized.assignments[0].goal, 3,
  'normalization should preserve a positive authored assignment goal');
checkEqual(normalized.objectiveValues[firstAssignment.id], 3,
  'tampered objective progress above an assignment goal should clamp to that goal');
checkEqual(normalized.objectiveValues[assignments[1].id], 0,
  'negative objective progress should normalize to zero');
check(!Object.prototype.hasOwnProperty.call(normalized.objectiveValues, 'unknown_objective'),
  'progress for assignments no longer in the rotation should be discarded');
checkEqual(normalized.creditedEventKeys.length, CONFIG.eventKeyLimit,
  'credited event history should remain bounded');
checkEqual(new Set(normalized.creditedEventKeys).size, normalized.creditedEventKeys.length,
  'credited event history should be deduplicated');
checkEqual(normalized.completedWeekCount, 7,
  'completed week count should normalize to a non-negative integer');
normalized.assignments[0].label = 'Changed in normalized state';
check(normalizationInput.assignments[0].label !== normalized.assignments[0].label,
  'normalized assignments should not alias the save payload');

// Unlock, initialization, rollover, and backward-clock protection.
const lockedState = {
  activeSeasonId: 'beta_foundations',
  objectiveValues: {},
  claimedRewardIds: [],
  weeklyRoutes: weeklyRoutes.createWeeklyRouteState(null, { config: CONFIG })
};
const lockedReconciliation = weeklyRoutes.reconcileWeeklyRouteState(lockedState, candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS,
  seasonState: lockedState,
  seasonSnapshot: { activeSeason: { id: 'beta_foundations' }, complete: false }
});
check(!lockedReconciliation.state.unlocked &&
  !lockedReconciliation.initialized &&
  lockedReconciliation.state.assignments.length === 0,
'an incomplete unlock season should leave Weekly Star Routes locked and uninitialized');

const completedSeasonState = Object.assign({}, lockedState, {
  weeklyRoutes: weeklyRoutes.createWeeklyRouteState(null, { config: CONFIG })
});
const completedSeasonUnlock = weeklyRoutes.reconcileWeeklyRouteState(completedSeasonState, candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS,
  seasonState: completedSeasonState,
  seasonSnapshot: { activeSeason: { id: 'beta_foundations' }, complete: true }
});
check(completedSeasonUnlock.state.unlocked && completedSeasonUnlock.initialized,
  'completing Beta Foundations should permanently initialize Weekly Star Routes before reward claim');

const initialized = initializeWeek(WEEK_ONE_MS);
check(initialized.state.unlocked &&
  initialized.initialized &&
  !initialized.rolledOver &&
  initialized.state.weekStartMs === WEEK_ONE_MS &&
  initialized.state.assignments.length === 4,
'a claimed Beta Foundations reward should initialize the current weekly rotation');
initialized.state.objectiveValues[initialized.state.assignments[0].id] = 1;
initialized.state.creditedEventKeys = ['hunt:preserved'];
const sameWeek = weeklyRoutes.reconcileWeeklyRouteState(initialized.state, candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS + 6 * DAY_MS,
  unlocked: true
});
check(!sameWeek.rolledOver &&
  sameWeek.state.objectiveValues[initialized.state.assignments[0].id] === 1 &&
  sameWeek.state.creditedEventKeys.includes('hunt:preserved'),
'same-week reconciliation should preserve assignments, progress, and credited events');

const removedAssignment = sameWeek.state.assignments[0];
const candidatesWithoutRemovedTarget = {
  field: candidates.field.filter((candidate) => candidate.mapId !== removedAssignment.targetId),
  mechanic: candidates.mechanic.filter((candidate) => candidate.mapMechanicId !== removedAssignment.targetId),
  dungeon: candidates.dungeon.filter((candidate) => candidate.dungeonId !== removedAssignment.targetId)
};
const validReplacementTargets = new Set(
  candidatesWithoutRemovedTarget.field.map((candidate) => `mapHunt:${candidate.mapId}`)
    .concat(candidatesWithoutRemovedTarget.mechanic.map((candidate) => `mapMechanic:${candidate.mapMechanicId}`))
    .concat(candidatesWithoutRemovedTarget.dungeon.map((candidate) => `dungeon:${candidate.dungeonId}`))
);
const contentRepaired = weeklyRoutes.reconcileWeeklyRouteState(sameWeek.state, candidatesWithoutRemovedTarget, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS + 6 * DAY_MS,
  unlocked: true,
  isAssignmentValid: (assignment) => validReplacementTargets.has(`${assignment.kind}:${assignment.targetId}`)
});
check(contentRepaired.replacedAssignmentIds.includes(removedAssignment.id) &&
  !contentRepaired.state.assignments.some((assignment) => assignment.targetId === removedAssignment.targetId) &&
  !Object.prototype.hasOwnProperty.call(contentRepaired.state.objectiveValues, removedAssignment.id),
'same-week reconciliation should replace removed content and discard only its stale progress');

const rolled = weeklyRoutes.reconcileWeeklyRouteState(sameWeek.state, candidates, {
  config: CONFIG,
  nowMs: WEEK_TWO_MS,
  unlocked: true
});
check(rolled.rolledOver &&
  rolled.state.weekStartMs === WEEK_TWO_MS &&
  Object.keys(rolled.state.objectiveValues).length === 0 &&
  rolled.state.creditedEventKeys.length === 0,
'crossing the UTC reset should clear only weekly progress and dedupe history');
const rollback = weeklyRoutes.reconcileWeeklyRouteState(rolled.state, candidates, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS + DAY_MS,
  unlocked: true
});
check(rollback.clockGuarded &&
  !rollback.rolledOver &&
  rollback.state.weekStartMs === WEEK_TWO_MS &&
  rollback.weekKey === '2026-07-27',
'a backward local clock must not restore an older rotation or reopen its reward');

const permanentlyUnlocked = weeklyRoutes.isWeeklyRouteUnlocked({
  weeklyRoutes: Object.assign({}, rolled.state, { unlocked: true }),
  claimedRewardIds: []
}, { config: CONFIG });
check(permanentlyUnlocked,
  'the unlock flag should remain sufficient after the one-time season is no longer active');

// Event matching, deduplication, and capping.
const eventWeek = initializeWeek(WEEK_ONE_MS).state;
const mapAssignment = eventWeek.assignments.find((assignment) => assignment.kind === 'mapHunt');
const mechanicAssignment = eventWeek.assignments.find((assignment) => assignment.kind === 'mapMechanic');
const dungeonAssignment = eventWeek.assignments.find((assignment) => assignment.kind === 'dungeon');

let eventPlan = weeklyRoutes.createWeeklyRouteEventPlan(eventWeek, mapAssignment.type, {
  mapId: 'wrong_map',
  eventKey: 'wrong-map'
}, {
  config: CONFIG,
  candidates,
  nowMs: WEEK_ONE_MS + DAY_MS,
  unlocked: true
});
check(!eventPlan.credited && eventPlan.reason === 'no-match' &&
  !eventPlan.state.creditedEventKeys.includes('wrong-map'),
'a nonmatching event should not consume its dedupe key');

eventPlan = weeklyRoutes.createWeeklyRouteEventPlan(eventPlan.state, mapAssignment.type, {
  mapId: mapAssignment.targetId
}, {
  config: CONFIG,
  candidates,
  nowMs: WEEK_ONE_MS + DAY_MS,
  unlocked: true
});
check(!eventPlan.credited && eventPlan.reason === 'missing-event-key',
  'weekly progress should require a stable source event key');

if (dungeonAssignment) {
  eventPlan = weeklyRoutes.createWeeklyRouteEventPlan(eventPlan.state, dungeonAssignment.type, {
    dungeonId: dungeonAssignment.targetId,
    mapId: dungeonAssignment.mapId,
    eventKey: 'dungeon-without-run'
  }, {
    config: CONFIG,
    candidates,
    nowMs: WEEK_ONE_MS + DAY_MS,
    unlocked: true
  });
  check(!eventPlan.credited && eventPlan.reason === 'no-match',
    'a dungeon completion without a real run id should not advance the weekly route');
}

eventPlan = applyAssignmentEvent(eventPlan.state, mapAssignment, 'map-hunt-once');
check(eventPlan.credited &&
  eventPlan.reason === 'credited' &&
  eventPlan.state.objectiveValues[mapAssignment.id] === mapAssignment.goal,
'a matching map-hunt claim should credit and cap its assigned row');
const creditedState = eventPlan.state;
const duplicatePlan = applyAssignmentEvent(creditedState, mechanicAssignment || mapAssignment, 'map-hunt-once');
check(!duplicatePlan.credited &&
  duplicatePlan.reason === 'duplicate-event' &&
  duplicatePlan.completionCount === 1,
'the same source event key should never credit a second assignment');
const alreadyComplete = applyAssignmentEvent(duplicatePlan.state, mapAssignment, 'map-hunt-fresh-key');
check(!alreadyComplete.credited &&
  alreadyComplete.reason === 'already-complete' &&
  !alreadyComplete.state.creditedEventKeys.includes('map-hunt-fresh-key'),
'fresh events for a finished row should not pollute dedupe history');

const multiCountConfig = Object.assign({}, CONFIG, {
  completionGoal: 2,
  eventKeyLimit: 3,
  slots: [CONFIG.slots[0]]
});
const multiAssignment = Object.assign({}, mapAssignment, {
  id: `field_a:mapHunt:${mapAssignment.targetId}`,
  slotId: 'field_a',
  kind: 'mapHunt',
  type: 'mapHuntClaim',
  goal: 5
});
let multiState = weeklyRoutes.createWeeklyRouteState({
  unlocked: true,
  weekStartMs: WEEK_ONE_MS,
  assignments: [multiAssignment],
  objectiveValues: {},
  creditedEventKeys: []
}, { config: multiCountConfig });
let multiPlan = null;
for (let index = 1; index <= 4; index += 1) {
  multiPlan = weeklyRoutes.createWeeklyRouteEventPlan(multiState, 'mapHuntClaim', {
    mapId: multiAssignment.targetId,
    eventKey: `multi-${index}`,
    count: 1
  }, {
    config: multiCountConfig,
    candidates,
    nowMs: WEEK_ONE_MS,
    unlocked: true
  });
  multiState = multiPlan.state;
}
check(multiPlan.credited &&
  multiPlan.state.objectiveValues[multiAssignment.id] === 4 &&
  multiPlan.state.creditedEventKeys.length === 3 &&
  !multiPlan.state.creditedEventKeys.includes('multi-1'),
'credited progress should advance by count while retaining only the configured newest event keys');
multiPlan = weeklyRoutes.createWeeklyRouteEventPlan(multiPlan.state, 'mapHuntClaim', {
  mapId: multiAssignment.targetId,
  eventKey: 'multi-5',
  count: 99
}, {
  config: multiCountConfig,
  candidates,
  nowMs: WEEK_ONE_MS,
  unlocked: true
});
checkEqual(multiPlan.state.objectiveValues[multiAssignment.id], 5,
  'multi-step weekly progress should cap at its authored goal');
checkDeepEqual(multiPlan.state.creditedEventKeys, ['multi-3', 'multi-4', 'multi-5'],
  'event-key eviction should be deterministic after capping progress');

// Completing any three of four rows should auto-award once; the fourth is optional.
let rewardState = initializeWeek(WEEK_ONE_MS).state;
const rewardAssignments = rewardState.assignments.slice();
const rewardPlans = [];
for (let index = 0; index < 3; index += 1) {
  const plan = applyAssignmentEvent(rewardState, rewardAssignments[index], `reward-${index}`);
  rewardPlans.push(plan);
  rewardState = plan.state;
}
check(!rewardPlans[0].rewardGranted &&
  !rewardPlans[1].rewardGranted &&
  rewardPlans[2].rewardGranted,
'the reward should be granted on the third completed row, not before');
checkDeepEqual(rewardPlans[2].reward, {
  currency: 400,
  starTokens: 75
}, 'the third-row event should return the exact bounded weekly reward');
check(rewardState.rewardGrantedWeekStartMs === WEEK_ONE_MS &&
  rewardState.completedWeekCount === 1,
'granting the weekly reward should persist its week and lifetime completion count');
const fourthPlan = applyAssignmentEvent(rewardState, rewardAssignments[3], 'reward-fourth');
check(fourthPlan.credited &&
  fourthPlan.completionCount === 4 &&
  !fourthPlan.rewardGranted &&
  fourthPlan.state.completedWeekCount === 1,
'the optional fourth row should remain completable without granting the reward twice');

let alternateRewardState = initializeWeek(WEEK_ONE_MS).state;
const alternateOrder = [0, 1, 3];
let alternateRewardPlan = null;
alternateOrder.forEach((assignmentIndex, eventIndex) => {
  alternateRewardPlan = applyAssignmentEvent(
    alternateRewardState,
    alternateRewardState.assignments[assignmentIndex],
    `alternate-${eventIndex}`
  );
  alternateRewardState = alternateRewardPlan.state;
});
check(alternateRewardPlan.rewardGranted &&
  alternateRewardPlan.completionCount === 3 &&
  !alternateRewardState.objectiveValues[alternateRewardState.assignments[2].id],
'any three rows should award, including a route that skips the mechanic challenge');

const completedSnapshot = weeklyRoutes.createWeeklyRouteSnapshot(fourthPlan.state, {
  config: CONFIG,
  nowMs: WEEK_ONE_MS + 2 * DAY_MS
});
check(completedSnapshot.complete &&
  completedSnapshot.completionCount === 4 &&
  completedSnapshot.completionGoal === 3 &&
  completedSnapshot.rewardGranted &&
  completedSnapshot.remainingMs === 5 * DAY_MS &&
  completedSnapshot.assignments.every((assignment) => assignment.complete),
'the weekly snapshot should expose completion, reset timing, and every row without recomputing rewards');
completedSnapshot.assignments[0].label = 'Snapshot mutation';
check(fourthPlan.state.assignments[0].label !== completedSnapshot.assignments[0].label,
  'weekly snapshots should not alias saved assignments');

const secondWeek = weeklyRoutes.reconcileWeeklyRouteState(fourthPlan.state, candidates, {
  config: CONFIG,
  nowMs: WEEK_TWO_MS,
  unlocked: true
}).state;
let secondWeekState = secondWeek;
let secondWeekReward = null;
for (let index = 0; index < 3; index += 1) {
  const plan = applyAssignmentEvent(secondWeekState, secondWeekState.assignments[index], `week-two-${index}`, {
    nowMs: WEEK_TWO_MS
  });
  secondWeekState = plan.state;
  if (plan.rewardGranted) secondWeekReward = plan;
}
check(secondWeekReward &&
  secondWeekState.rewardGrantedWeekStartMs === WEEK_TWO_MS &&
  secondWeekState.completedWeekCount === 2,
'a genuinely new UTC week should be able to award once and advance lifetime completions');

// Runtime integration: canonical candidate filtering, event routing, auto-award, and persistence.
const contentEngine = createProjectStarfallEngine(null, data);
check(contentEngine.chooseClass('fighter'),
  'the runtime content fixture should create a playable Fighter');
const runtimeCandidates = contentEngine.getWeeklyRouteCandidates();
checkDeepEqual(Object.keys(runtimeCandidates), ['field', 'mechanic', 'dungeon'],
  'the runtime should expose the canonical weekly candidate groups');
check(runtimeCandidates.field.length > 0,
  'a new character should have at least one reachable field-hunt candidate');
runtimeCandidates.field.forEach((candidate) => {
  const map = data.MAPS.find((entry) => entry.id === candidate.mapId);
  check(map &&
    !map.safeZone &&
    !map.adminOnly &&
    !map.isDungeon &&
    !map.dungeonId &&
    Array.isArray(map.enemies) &&
    map.enemies.length > 0 &&
    candidate.kind === 'mapHunt' &&
    candidate.guideType === 'mapKill' &&
    candidate.guideId === map.id,
  `field candidate ${candidate.targetId} should be a reachable combat map with hunt guidance`);
});
runtimeCandidates.mechanic.forEach((candidate) => {
  const definition = Object.values(data.MAP_MECHANIC_DEFINITIONS || {})
    .find((entry) => entry && entry.id === candidate.targetId);
  check(definition &&
    definition.mapId === candidate.mapId &&
    runtimeCandidates.field.some((field) => field.mapId === candidate.mapId) &&
    candidate.kind === 'mapMechanic' &&
    candidate.guideType === 'map',
  `mechanic candidate ${candidate.targetId} should belong to an eligible field map`);
});
runtimeCandidates.dungeon.forEach((candidate) => {
  const dungeon = data.DUNGEONS.find((entry) => entry.id === candidate.targetId);
  const map = dungeon && data.MAPS.find((entry) => entry.id === dungeon.mapId);
  check(dungeon &&
    map &&
    candidate.mapId === map.id &&
    candidate.kind === 'dungeon' &&
    candidate.guideType === 'dungeon' &&
    candidate.guideId === dungeon.id,
  `dungeon candidate ${candidate.targetId} should reference a real, guided dungeon`);
});

const provenanceEngine = createProjectStarfallEngine(null, data);
check(provenanceEngine.chooseClass('fighter'), 'the provenance fixture should create a Fighter');
provenanceEngine.state.mapId = 'greenrootMeadow';
check(provenanceEngine.startMapKillQuest('greenrootMeadow'),
'the provenance fixture should start a normal local hunt');
['adminSpawned', 'temporarySpawn', 'adminDefeated'].forEach((flag) => {
  const enemy = { data: { behavior: 'walker' }, [flag]: true };
  check(!provenanceEngine.recordMapKillQuestDefeat(enemy) &&
    provenanceEngine.getMapKillQuestState().greenrootMeadow.progress === 0,
  `${flag} enemies should not advance local hunts`);
});
provenanceEngine.recordMapKillQuestDefeat({ data: { behavior: 'walker' } });
check(provenanceEngine.getMapKillQuestState().greenrootMeadow.progress === 1,
  'a normal production enemy should still advance the local hunt');
provenanceEngine.state.mapId = 'orebackQuarry';
check(!provenanceEngine.recordMapMechanicDefeat({
  data: { behavior: 'walker' },
  adminSpawned: true
}), 'admin-spawned enemies should not advance authored map mechanics');

const unlockEventEngine = createProjectStarfallEngine(null, data);
check(unlockEventEngine.chooseClass('fighter'),
  'the unlock-event fixture should create a playable Fighter');
unlockEventEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
unlockEventEngine.state.season.claimedRewardIds.push('beta_foundations');
const unlockEventAssignment = weeklyRoutes.createWeeklyRouteAssignments(createCandidateFixture(), {
  config: CONFIG,
  nowMs: WEEK_ONE_MS
})[0];
check(unlockEventEngine.recordProgressEvent(
  unlockEventAssignment.type,
  payloadForAssignment(unlockEventAssignment, 'first-unlock-event'),
  { nowMs: WEEK_ONE_MS, audio: false, noEmit: true }
), 'the first post-unlock event should initialize the weekly card');
const unlockEventSnapshot = unlockEventEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
check(unlockEventSnapshot.assignments.length === 4 &&
  unlockEventSnapshot.completionCount === 0 &&
  unlockEventEngine.state.season.weeklyRoutes.creditedEventKeys.length === 0,
'the event that first initializes Weekly Star Routes should not retroactively credit its new row');

const stableLowCandidates = {
  field: ['greenrootMeadow', 'thornpathThicket', 'rustcoilRuins', 'cinderHollow'].map((mapId) => ({
    mapId,
    guideType: 'mapKill',
    guideId: mapId
  })),
  mechanic: [{
    mapMechanicId: 'oreback_material_rush',
    mapId: 'orebackQuarry',
    guideType: 'map',
    guideId: 'orebackQuarry'
  }],
  dungeon: [{
    dungeonId: 'bramble_depths',
    mapId: 'brambleDepths',
    guideType: 'dungeon',
    guideId: 'bramble_depths'
  }]
};
const stableHighCandidates = {
  field: ['ashglassPass', 'frostfenOutskirts', 'glacierSpine', 'stormbreakCliffs'].map((mapId) => ({
    mapId,
    guideType: 'mapKill',
    guideId: mapId
  })),
  mechanic: [{
    mapMechanicId: 'stormbreak_lightning_rod',
    mapId: 'stormbreakCliffs',
    guideType: 'map',
    guideId: 'stormbreakCliffs'
  }],
  dungeon: [{
    dungeonId: 'rimewarden_sanctum',
    mapId: 'rimewardenSanctum',
    guideType: 'dungeon',
    guideId: 'rimewarden_sanctum'
  }]
};
const stableRuntimeEngine = createProjectStarfallEngine(null, data);
check(stableRuntimeEngine.chooseClass('fighter'),
  'the same-week stability fixture should create a playable Fighter');
stableRuntimeEngine.state.season.claimedRewardIds.push('beta_foundations');
stableRuntimeEngine.getWeeklyRouteCandidates = () => stableLowCandidates;
const stableBefore = stableRuntimeEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
const stableFirst = stableBefore.assignments[0];
check(stableRuntimeEngine.recordProgressEvent(
  stableFirst.type,
  payloadForAssignment(stableFirst, 'stable-level-event'),
  { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
), 'the same-week stability fixture should credit its first row');
stableRuntimeEngine.state.player.level = 80;
stableRuntimeEngine.getWeeklyRouteCandidates = () => stableHighCandidates;
const stableAfter = stableRuntimeEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
checkDeepEqual(
  stableAfter.assignments.map((assignment) => assignment.id),
  stableBefore.assignments.map((assignment) => assignment.id),
  'leveling and opening newer content should not reroll persisted assignments midweek'
);
check(stableAfter.completionCount === 1 &&
  stableAfter.assignments[0].complete,
'same-week candidate changes should preserve completed-row progress');

const integrationEngine = createProjectStarfallEngine(null, data);
check(integrationEngine.chooseClass('fighter'),
  'the runtime integration fixture should create a playable Fighter');
integrationEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
const lockedRuntimeSnapshot = integrationEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
check(!lockedRuntimeSnapshot.unlocked &&
  lockedRuntimeSnapshot.assignments.length === 0 &&
  /Beta Foundations/.test(lockedRuntimeSnapshot.lockedReason),
'the runtime snapshot should explain the one-time Beta Foundations unlock');

const routeLimitedEngine = createProjectStarfallEngine(null, data);
check(routeLimitedEngine.chooseClass('fighter'),
  'the route-limited presentation fixture should create a playable Fighter');
routeLimitedEngine.state.season.claimedRewardIds.push('beta_foundations');
routeLimitedEngine.getWeeklyRouteCandidates = () => ({
  field: createCandidateFixture().field.slice(0, 2),
  mechanic: createCandidateFixture().mechanic.slice(0, 1),
  dungeon: []
});
const routeLimitedSnapshot = routeLimitedEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
const routeLimitedPresentation = questUi.getWeeklyStarRoutePresentation({
  weeklyRoutes: routeLimitedSnapshot
}, { nowMs: WEEK_ONE_MS });
check(!routeLimitedSnapshot.unlocked &&
  routeLimitedSnapshot.permanentlyUnlocked &&
  routeLimitedSnapshot.assignments.length === 3 &&
  /connected routes/.test(routeLimitedSnapshot.lockedReason),
'a permanently unlocked save with too few authored targets should explain that more connected routes are needed');
check(routeLimitedPresentation &&
  routeLimitedPresentation.permanentlyUnlocked &&
  routeLimitedPresentation.lockedReason === routeLimitedSnapshot.lockedReason,
'the Quests card presentation should preserve the route-readiness reason instead of showing a generic lock');
const routeLimitedCoinsBefore = routeLimitedEngine.state.player.currency;
const routeLimitedTokensBefore = routeLimitedEngine.state.cashShop.starTokens;
routeLimitedSnapshot.assignments.forEach((assignment, index) => {
  routeLimitedEngine.recordProgressEvent(
    assignment.type,
    payloadForAssignment(assignment, `route-limited-${index}`),
    { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
  );
});
const routeLimitedAfterEvents = routeLimitedEngine.getWeeklyRouteSnapshot({
  nowMs: WEEK_ONE_MS + DAY_MS
});
check(routeLimitedAfterEvents.completionCount === 0 &&
  routeLimitedAfterEvents.assignments.every((assignment) => !assignment.complete) &&
  routeLimitedEngine.state.season.weeklyRoutes.creditedEventKeys.length === 0,
'a route-limited weekly card must not record progress invisibly');
check(!routeLimitedAfterEvents.rewardGranted &&
  routeLimitedEngine.state.player.currency === routeLimitedCoinsBefore &&
  routeLimitedEngine.state.cashShop.starTokens === routeLimitedTokensBefore,
'a route-limited weekly card must not award coins or Star Tokens');

function createLegacyAdvancedPayload(options) {
  const settings = options || {};
  const fixture = createProjectStarfallEngine(null, data);
  const baseClassId = settings.baseClassId || 'fighter';
  assert.ok(fixture.chooseClass(baseClassId), 'legacy advanced-class payload fixture should choose its base class');
  const payload = fixture.serialize();
  payload.state.player.level = 35;
  payload.state.player.classId = baseClassId;
  payload.state.player.advancedClassId = settings.advancedClassId || 'guardian';
  payload.state.season = {
    activeSeasonId: settings.activeSeasonId || 'beta_foundations',
    objectiveValues: Object.assign({}, settings.objectiveValues || {
      field_bosses: 2,
      dungeon_clears: 2
    }),
    claimedRewardIds: []
  };
  return payload;
}

const legacyAdvancedPayload = createLegacyAdvancedPayload();
const legacyCurrencyBefore = legacyAdvancedPayload.state.player.currency;
const legacyTokensBefore = legacyAdvancedPayload.state.cashShop.starTokens;
const legacyClaimsBefore = legacyAdvancedPayload.state.season.claimedRewardIds.slice();
const legacyAdvancedEngine = createProjectStarfallEngine(null, data);
legacyAdvancedEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(legacyAdvancedEngine.restore(legacyAdvancedPayload),
  'a real legacy advanced-class payload should restore through the public engine');
checkEqual(legacyAdvancedEngine.state.season.objectiveValues.advanced_path, 1,
  'restore-time migration should immediately backfill the authored advanced-class milestone');
checkEqual(legacyAdvancedEngine.serialize().state.season.objectiveValues.advanced_path, 1,
  'the restored permanent milestone should persist in the next save');
const legacyAdvancedSnapshot = legacyAdvancedEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS });
check(legacyAdvancedSnapshot.complete &&
  legacyAdvancedSnapshot.weeklyRoutes.permanentlyUnlocked &&
  legacyAdvancedSnapshot.weeklyRoutes.assignments.length === 4,
'a valid legacy Guardian should complete Beta Foundations and unlock weekly routes');
check(legacyAdvancedEngine.state.player.currency === legacyCurrencyBefore &&
  legacyAdvancedEngine.state.cashShop.starTokens === legacyTokensBefore &&
  !legacyAdvancedSnapshot.rewardClaimed,
'permanent-milestone migration should never grant currency, Star Tokens, or the season reward');
checkDeepEqual(legacyAdvancedEngine.state.season.claimedRewardIds, legacyClaimsBefore,
  'permanent-milestone migration should not mutate season reward claims');
legacyAdvancedEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
const repeatedLegacySave = legacyAdvancedEngine.serialize();
check(repeatedLegacySave.state.season.objectiveValues.advanced_path === 1 &&
  repeatedLegacySave.state.player.currency === legacyCurrencyBefore &&
  repeatedLegacySave.state.cashShop.starTokens === legacyTokensBefore,
'later reads and saves should keep the migration idempotent and reward-free');

const partialLegacyEngine = createProjectStarfallEngine(null, data);
partialLegacyEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(partialLegacyEngine.restore(createLegacyAdvancedPayload({
  objectiveValues: { field_bosses: 2 }
})), 'a partially complete legacy Beta payload should restore');
const partialLegacySnapshot = partialLegacyEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS });
check(partialLegacyEngine.state.season.objectiveValues.advanced_path === 1 &&
  !partialLegacySnapshot.complete &&
  !partialLegacySnapshot.weeklyRoutes.permanentlyUnlocked,
'backfilling one permanent fact should not prematurely complete the season or unlock weekly routes');

[
  { advancedClassId: 'bogus', label: 'unknown advanced class' },
  { advancedClassId: 'fireMage', label: 'cross-class advanced path' }
].forEach((fixture) => {
  const invalidLegacyEngine = createProjectStarfallEngine(null, data);
  invalidLegacyEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
  check(invalidLegacyEngine.restore(createLegacyAdvancedPayload(fixture)),
    `${fixture.label} payload should still restore safely`);
  const invalidLegacySnapshot = invalidLegacyEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS });
  check(!Object.prototype.hasOwnProperty.call(
    invalidLegacyEngine.state.season.objectiveValues,
    'advanced_path'
  ) &&
    !invalidLegacySnapshot.complete &&
    !invalidLegacySnapshot.weeklyRoutes.permanentlyUnlocked,
  `${fixture.label} must not receive Beta advancement credit or unlock weekly routes`);
});

const futureSeasonEngine = createProjectStarfallEngine(null, data);
futureSeasonEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(futureSeasonEngine.restore(createLegacyAdvancedPayload({
  activeSeasonId: 'future_season'
})), 'a legacy payload naming a future season should restore safely');
const futureSeasonSnapshot = futureSeasonEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS });
check(!Object.prototype.hasOwnProperty.call(
  futureSeasonEngine.state.season.objectiveValues,
  'advanced_path'
) &&
  !futureSeasonSnapshot.complete &&
  !futureSeasonSnapshot.weeklyRoutes.permanentlyUnlocked,
'a non-Beta active season must not receive Beta advancement credit or unlock weekly routes');

integrationEngine.state.season.claimedRewardIds.push('beta_foundations');
const initializedRuntimeSnapshot = integrationEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
check(initializedRuntimeSnapshot.unlocked &&
  initializedRuntimeSnapshot.assignments.length === 4 &&
  initializedRuntimeSnapshot.nextObjectiveId === initializedRuntimeSnapshot.assignments[0].id &&
  initializedRuntimeSnapshot.nextGuide &&
  initializedRuntimeSnapshot.nextGuide.type === initializedRuntimeSnapshot.assignments[0].guideType &&
  initializedRuntimeSnapshot.nextGuide.id === initializedRuntimeSnapshot.assignments[0].guideId &&
  initializedRuntimeSnapshot.nextGuide.assignmentId === initializedRuntimeSnapshot.assignments[0].id &&
  initializedRuntimeSnapshot.rewardSummary === '400 coins, 75 Star Tokens',
'the unlocked runtime snapshot should expose four guided rows and its exact inventory-independent reward');
const nestedSeasonSnapshot = integrationEngine.getSeasonSnapshot({ nowMs: WEEK_ONE_MS });
check(nestedSeasonSnapshot.weeklyRoutes &&
  nestedSeasonSnapshot.weeklyRoutes.weekKey === '2026-07-20',
'the season snapshot should publish the same fixed weekly route state');

const integrationAssignments = initializedRuntimeSnapshot.assignments;
const secondHuntAssignment = integrationAssignments.find((assignment, index) =>
  index > 0 && assignment.guideType === 'mapKill'
);
const dungeonGuideAssignment = integrationAssignments.find((assignment) =>
  assignment.guideType === 'dungeon'
);
check(secondHuntAssignment && dungeonGuideAssignment,
  'the weekly card fixture should expose a non-first hunt and a dungeon choice');
const fullCardPresentation = questUi.getWeeklyStarRoutePresentation({
  weeklyRoutes: initializedRuntimeSnapshot
}, {
  nowMs: WEEK_ONE_MS,
  guidance: integrationEngine.getQuestGuidanceSnapshot()
});
check(fullCardPresentation &&
  fullCardPresentation.visibleAssignments.length === 4 &&
  fullCardPresentation.hiddenCount === 0,
'the weekly card should keep all four 3-of-4 choices visible and selectable');
const secondHuntAction = questUi.getQuestPanelRegionAction({
  type: 'quest-guide',
  guideType: secondHuntAssignment.guideType,
  guideId: secondHuntAssignment.guideId,
  assignmentId: secondHuntAssignment.id
});
check(secondHuntAction.handled &&
  secondHuntAction.guideType === 'mapKill' &&
  secondHuntAction.assignmentId === secondHuntAssignment.id,
'weekly hunt clicks should preserve hunt semantics and assignment identity');

integrationEngine.state.session.worldMapSelectedId = integrationAssignments[0].mapId;
check(integrationEngine.setWeeklyRouteGuideTarget(secondHuntAssignment.id, {
  nowMs: WEEK_ONE_MS
}), 'a player should be able to focus the non-first weekly hunt');
const focusedHuntGuidance = integrationEngine.getQuestGuidanceSnapshot();
const focusedHuntSnapshot = integrationEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS });
const focusedHuntTracker = hud.getWeeklyStarRouteTrackerEntry({
  weeklyRoutes: focusedHuntSnapshot
}, focusedHuntGuidance);
const focusedHuntPresentation = questUi.getWeeklyStarRoutePresentation({
  weeklyRoutes: focusedHuntSnapshot
}, {
  nowMs: WEEK_ONE_MS,
  guidance: focusedHuntGuidance
});
check(integrationEngine.state.session.questGuide.type === 'mapKill' &&
  integrationEngine.state.session.questGuide.id === secondHuntAssignment.guideId &&
  integrationEngine.state.session.questGuide.assignmentId === secondHuntAssignment.id &&
  focusedHuntGuidance.targetType === 'mapKill' &&
  focusedHuntGuidance.assignmentId === secondHuntAssignment.id,
'the focused hunt should remain an end-to-end map-hunt guide instead of degrading into travel');
check(focusedHuntSnapshot.nextObjectiveId === secondHuntAssignment.id &&
  focusedHuntSnapshot.nextGuide.type === 'mapKill' &&
  focusedHuntSnapshot.nextGuide.id === secondHuntAssignment.guideId &&
  focusedHuntSnapshot.nextGuide.assignmentId === secondHuntAssignment.id &&
  focusedHuntTracker &&
  focusedHuntTracker.assignmentId === secondHuntAssignment.id,
'the weekly snapshot and HUD should honor the exact non-first assignment the player selected');
check(integrationEngine.getWorldMapSnapshot().selectedMapId === secondHuntAssignment.mapId &&
  focusedHuntPresentation.visibleAssignments.find((assignment) =>
    assignment.id === secondHuntAssignment.id
  ).focused,
'weekly focus should synchronize the world map and visibly mark the selected card row');

const focusedHuntSave = integrationEngine.serialize();
const focusedHuntRestoredEngine = createProjectStarfallEngine(null, data);
focusedHuntRestoredEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(focusedHuntRestoredEngine.restore(focusedHuntSave),
  'a save with a non-first weekly focus should restore');
const restoredFocusedHuntSnapshot = focusedHuntRestoredEngine.getWeeklyRouteSnapshot({
  nowMs: WEEK_ONE_MS
});
check(focusedHuntRestoredEngine.state.session.questGuide.assignmentId === secondHuntAssignment.id &&
  restoredFocusedHuntSnapshot.nextGuide.assignmentId === secondHuntAssignment.id &&
  focusedHuntRestoredEngine.getQuestGuidanceSnapshot().targetType === 'mapKill',
'save/restore should retain weekly assignment identity and hunt-specific guidance');

check(integrationEngine.setWeeklyRouteGuideTarget(dungeonGuideAssignment.id, {
  nowMs: WEEK_ONE_MS
}), 'a player should be able to focus the non-first weekly dungeon');
check(integrationEngine.state.session.questGuide.type === 'dungeon' &&
  integrationEngine.state.session.questGuide.id === dungeonGuideAssignment.guideId &&
  integrationEngine.state.session.questGuide.assignmentId === dungeonGuideAssignment.id &&
  integrationEngine.getWorldMapSnapshot().selectedMapId === dungeonGuideAssignment.mapId,
'dungeon focus should keep its dungeon identity and synchronize the destination map');
check(integrationEngine.setWorldMapGuideTarget(integrationAssignments[0].mapId) &&
  integrationEngine.state.session.questGuide.type === 'map' &&
  integrationEngine.state.session.questGuide.assignmentId === '',
'generic World Map guidance should remain plain travel and clear weekly focus');
check(integrationEngine.setWeeklyRouteGuideTarget(secondHuntAssignment.id, {
  nowMs: WEEK_ONE_MS
}), 'the completion handoff fixture should restore the selected second hunt');

for (let index = 0; index < 2; index += 1) {
  check(integrationEngine.recordProgressEvent(
    integrationAssignments[index].type,
    payloadForAssignment(integrationAssignments[index], `runtime-${index}`),
    { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
  ), `runtime event ${index + 1} should route through recordProgressEvent`);
}
const partialRuntimeSnapshot = integrationEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
check(partialRuntimeSnapshot.completionCount === 2 &&
  !partialRuntimeSnapshot.complete &&
  !partialRuntimeSnapshot.rewardGranted &&
  partialRuntimeSnapshot.nextObjectiveId === integrationAssignments[2].id,
'two real progress events should persist partial weekly state and advance guidance');
check(integrationEngine.state.session.questGuide.assignmentId === integrationAssignments[2].id &&
  partialRuntimeSnapshot.nextGuide.assignmentId === integrationAssignments[2].id,
'completing the selected weekly row should advance focus to the first unfinished assignment');

const partialSave = integrationEngine.serialize();
const partialRestoredEngine = createProjectStarfallEngine(null, data);
partialRestoredEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(partialRestoredEngine.restore(partialSave),
  'a partial Weekly Star Route save should restore through the public engine');
const partialRestoredSnapshot = partialRestoredEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
check(partialRestoredSnapshot.completionCount === 2 &&
  partialRestoredSnapshot.assignments.filter((assignment) => assignment.complete).length === 2,
'save/restore should preserve the exact two completed weekly rows');

const inventoryBeforeCapacityProbe = integrationEngine.state.inventory;
const consumablesBeforeCapacityProbe = integrationEngine.state.consumables;
const materialsBeforeCapacityProbe = integrationEngine.state.materials;
integrationEngine.state.inventory = Array.from(
  { length: integrationEngine.getInventoryCapacity('equipment') },
  (_, index) => ({ uid: `weekly_capacity_${index}` })
);
integrationEngine.state.consumables = {
  minor_health_potion: integrationEngine.getStackCap('usable', 'minor_health_potion') *
    integrationEngine.getInventoryCapacity('usable')
};
integrationEngine.state.materials = {
  upgradeDust: integrationEngine.getStackCap('etc', 'upgradeDust') *
    integrationEngine.getInventoryCapacity('etc')
};
check(integrationEngine.getInventoryUsedSlots('equipment') === integrationEngine.getInventoryCapacity('equipment') &&
  integrationEngine.getInventoryUsedSlots('usable') === integrationEngine.getInventoryCapacity('usable') &&
  integrationEngine.getInventoryUsedSlots('etc') === integrationEngine.getInventoryCapacity('etc') &&
  integrationEngine.getProgressRewardInventoryBlockReason(CONFIG.reward) === '',
'the currency-only weekly reward should remain atomic and claimable with item inventories full');

const runtimeCoinsBefore = integrationEngine.state.player.currency;
const runtimeTokensBefore = integrationEngine.state.cashShop.starTokens;
check(integrationEngine.recordProgressEvent(
  integrationAssignments[2].type,
  payloadForAssignment(integrationAssignments[2], 'runtime-2'),
  { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
), 'the third runtime event should complete and auto-award the weekly route');
check(integrationEngine.state.player.currency === runtimeCoinsBefore + 400 &&
  integrationEngine.state.cashShop.starTokens === runtimeTokensBefore + 75 &&
  integrationEngine.state.season.weeklyRoutes.completedWeekCount === 1,
'the runtime should grant the exact weekly currencies once on the third row');
check(!integrationEngine.recordProgressEvent(
  integrationAssignments[2].type,
  payloadForAssignment(integrationAssignments[2], 'runtime-2'),
  { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
) &&
  integrationEngine.state.player.currency === runtimeCoinsBefore + 400 &&
  integrationEngine.state.cashShop.starTokens === runtimeTokensBefore + 75,
'replaying the third-row event key should not mutate progress or duplicate runtime rewards');

integrationEngine.state.inventory = inventoryBeforeCapacityProbe;
integrationEngine.state.consumables = consumablesBeforeCapacityProbe;
integrationEngine.state.materials = materialsBeforeCapacityProbe;
check(integrationEngine.recordProgressEvent(
  integrationAssignments[3].type,
  payloadForAssignment(integrationAssignments[3], 'runtime-3'),
  { nowMs: WEEK_ONE_MS + DAY_MS, audio: false, noEmit: true }
), 'the optional fourth runtime row should remain completable');
check(integrationEngine.state.player.currency === runtimeCoinsBefore + 400 &&
  integrationEngine.state.cashShop.starTokens === runtimeTokensBefore + 75 &&
  integrationEngine.state.season.weeklyRoutes.completedWeekCount === 1,
'the optional fourth row should not duplicate the runtime reward');

const completedRuntimeSave = integrationEngine.serialize();
const completedRestoredEngine = createProjectStarfallEngine(null, data);
completedRestoredEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(completedRestoredEngine.restore(completedRuntimeSave),
  'a rewarded Weekly Star Route save should restore through the public engine');
const completedRestoredSnapshot = completedRestoredEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
check(completedRestoredSnapshot.rewardGranted &&
  completedRestoredSnapshot.completionCount === 4 &&
  completedRestoredSnapshot.completedWeekCount === 1,
'save/restore should retain the rewarded week and all completed rows');

const legacyRuntimeSave = integrationEngine.serialize();
delete legacyRuntimeSave.state.season.weeklyRoutes;
const migratedWeeklyEngine = createProjectStarfallEngine(null, data);
migratedWeeklyEngine.getWeeklyRouteCandidates = () => createCandidateFixture();
check(migratedWeeklyEngine.restore(legacyRuntimeSave),
  'a legacy save without weekly route state should restore');
const migratedWeeklySnapshot = migratedWeeklyEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
check(migratedWeeklySnapshot.unlocked &&
  migratedWeeklySnapshot.assignments.length === 4 &&
  migratedWeeklySnapshot.completionCount === 0 &&
  !migratedWeeklySnapshot.rewardGranted,
'legacy restore should initialize a clean current-week route from the permanent season unlock');

const rolledRuntimeSnapshot = completedRestoredEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_TWO_MS });
check(rolledRuntimeSnapshot.weekKey === '2026-07-27' &&
  rolledRuntimeSnapshot.completionCount === 0 &&
  !rolledRuntimeSnapshot.rewardGranted &&
  rolledRuntimeSnapshot.completedWeekCount === 1,
'runtime snapshot access in a new week should roll progress without losing lifetime completions');
const guardedRuntimeSnapshot = completedRestoredEngine.getWeeklyRouteSnapshot({ nowMs: WEEK_ONE_MS + DAY_MS });
check(guardedRuntimeSnapshot.clockGuarded &&
  guardedRuntimeSnapshot.weekKey === '2026-07-27' &&
  guardedRuntimeSnapshot.completionCount === 0,
'runtime snapshots should preserve the newer rotation when the supplied clock moves backward');

// Content policy: inventory-independent currency only, with no power/RNG/storage shortcut.
check(CONFIG &&
  CONFIG.id === 'weekly_star_routes' &&
  CONFIG.completionGoal === 3 &&
  CONFIG.eventKeyLimit === 32,
'the public data config should define the expected weekly identity, 3-of-4 goal, and dedupe bound');
checkDeepEqual(CONFIG.slots.map((slot) => slot.id), ['field_a', 'field_b', 'challenge', 'dungeon'],
  'the public config should preserve four stable assignment slots');
checkDeepEqual(CONFIG.reward, {
  currency: 400,
  starTokens: 75
}, 'the public weekly reward should match the tested economy budget');
const serializedReward = JSON.stringify(CONFIG.reward);
[
  'items',
  'cards',
  'consumables',
  'materials',
  'permanentStats',
  'cosmeticId',
  'slot_coupon',
  'advanced_skill_manual',
  'potential_cube',
  'random',
  'rarity'
].forEach((forbiddenKey) => {
  check(!serializedReward.includes(forbiddenKey),
    `weekly rewards should exclude ${forbiddenKey} power, RNG, or capacity shortcuts`);
});

console.log(`Project Starfall Weekly Star Routes checks passed: ${checks}`);
