'use strict';

const assert = require('assert');

global.ProjectStarfallData = require('../js/games/project-starfall/data/index.js');
const data = global.ProjectStarfallData;
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { getClassTrialActionPresentation } = require('../js/games/project-starfall/ui/quests.js');

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function getToastMessage(payload) {
  return typeof payload === 'string' ? payload : String(payload && payload.message || '');
}

function createFixture(classId) {
  const engine = createProjectStarfallEngine(null, data);
  const toasts = [];
  let changes = 0;
  engine.onToast = (payload) => toasts.push(getToastMessage(payload));
  engine.onChange = () => {
    changes += 1;
  };
  check(engine.chooseClass(classId), `${classId} defeat fixture should choose its class`);
  engine.state.player.level = 20;
  engine.invalidateStatsCache();
  const stats = engine.getStats();
  engine.state.player.hp = stats.maxHp;
  engine.state.player.mp = stats.maxMp;
  engine.state.player.resource = stats.secondaryResourceMax;
  return {
    engine,
    toasts,
    resetSignals() {
      toasts.length = 0;
      changes = 0;
    },
    getChanges() {
      return changes;
    }
  };
}

function getDeathCount(engine) {
  return Number(engine.ensureCombatMetrics().deaths || 0);
}

function defeatPlayer(engine, source) {
  const stats = engine.getStats();
  const deathsBefore = getDeathCount(engine);
  engine.state.player.invulnerableUntil = 0;
  engine.damagePlayer(Number.MAX_SAFE_INTEGER, source);
  return { stats, deathsBefore };
}

function checkRecoveryVitals(engine, stats, label) {
  const player = engine.state.player;
  check(player.hp === Math.max(1, Math.round(stats.maxHp * 0.45)),
    `${label} recovery should restore 45% HP`);
  check(player.mp === Math.round(stats.maxMp * 0.35),
    `${label} recovery should restore 35% MP`);
  check(player.resource === Math.round(stats.secondaryResourceMax * 0.35),
    `${label} recovery should restore 35% class resource`);
}

data.CLASS_TRIALS.forEach((trial) => {
  const fixture = createFixture(trial.baseClass);
  const { engine, toasts } = fixture;
  check(engine.startClassTrial(trial.id),
    `${trial.title} should start before its defeat-recovery check`);
  check(engine.runtime && engine.runtime.isTrialInstance &&
    engine.getTrialInstanceState().trialId === trial.id,
  `${trial.title} should run inside its authored trial instance`);

  const firstObjective = trial.objectives[0];
  check(engine.recordProgressEvent('defeat', {
    enemyId: firstObjective.enemyId,
    mapId: engine.state.mapId
  }), `${trial.title} should record partial attempt progress`);
  check(Number(
    engine.getProgressState().trialProgress[trial.id].objectiveValues[firstObjective.id] || 0
  ) === 1, `${trial.title} should retain partial progress before defeat`);

  fixture.resetSignals();
  const { stats, deathsBefore } = defeatPlayer(engine, `${trial.title} foe`);
  const progress = engine.getProgressState();
  const player = engine.state.player;
  const trialProgress = progress.trialProgress[trial.id];

  check(engine.state.mapId === 'starfallCrossing' &&
    engine.runtime &&
    engine.runtime.id === 'starfallCrossing' &&
    !engine.runtime.isTrialInstance,
  `${trial.title} defeat should restore the real Starfall Crossing runtime`);
  check(!progress.trialInstance.active &&
    progress.activeTrialId === trial.id &&
    !progress.completedTrials[trial.advancedId],
  `${trial.title} defeat should close only the failed instance and preserve retry eligibility`);
  check(trialProgress &&
    Object.keys(trialProgress.objectiveValues || {}).length === 0 &&
    Number(trialProgress.completedAt || 0) === 0,
  `${trial.title} defeat should reset attempt objectives without completing the trial`);
  check(player.grounded &&
    engine.runtime.platforms.some((platform) => platform.id === player.groundedPlatformId),
  `${trial.title} defeat should use a grounded hometown return placement`);
  check(engine.enemies.length === 0,
    `${trial.title} defeat should leave no trial enemies in the safe hometown`);
  check(getDeathCount(engine) === deathsBefore + 1,
    `${trial.title} defeat should increment the death metric once`);
  checkRecoveryVitals(engine, stats, trial.title);
  check(toasts.length === 1 &&
    toasts[0].includes(`${trial.title} attempt reset.`) &&
    toasts[0].includes('retry from Quests') &&
    !toasts[0].includes('Finish the advancement trial'),
  `${trial.title} defeat should publish one coherent recovery message`);
  check(fixture.getChanges() === 1,
    `${trial.title} defeat should publish one consolidated state update`);

  const retryPresentation = getClassTrialActionPresentation(
    engine.getProgressSnapshot().trials.find((candidate) => candidate.id === trial.id),
    engine.getProgressSnapshot(),
    engine.state.player
  );
  check(retryPresentation.retryReady &&
    !retryPresentation.disabled &&
    retryPresentation.buttonLabel === 'Retry' &&
    retryPresentation.statusLabel === 'Ready to retry' &&
    retryPresentation.panelHeading === 'Trial Ready to Retry',
  `${trial.title} defeat should expose a clear, enabled retry action`);
  const retryGuidance = engine.getObjectiveGuidance(
    'trial',
    trial.id,
    trial.title,
    firstObjective,
    trial
  );
  check(!retryGuidance.recommendedMapId &&
    retryGuidance.targetEnemyIds.length === 0 &&
    retryGuidance.hint.includes('choose Retry'),
  `${trial.title} recovery guidance should point to the retry action, not field enemies`);

  (trial.objectives || []).forEach((objective) => {
    const count = Math.max(1, Number(objective.count || 1));
    for (let index = 0; index < count; index += 1) {
      engine.recordProgressEvent(objective.type, {
        enemyId: objective.enemyId,
        bossId: objective.bossId,
        mapId: objective.mapId || trial.mapId
      }, { noEmit: true, audio: false });
    }
  });
  check(Object.keys(engine.getProgressState().trialProgress[trial.id].objectiveValues || {}).length === 0 &&
    !engine.getProgressState().completedTrials[trial.advancedId],
  `${trial.title} recovery should reject matching kills outside its trial instance`);

  fixture.resetSignals();
  check(engine.startAdvancementTrialFromQuest(),
    `${trial.title} should remain immediately retryable`);
  const retryProgress = engine.getProgressState().trialProgress[trial.id];
  check(engine.runtime.isTrialInstance &&
    engine.getTrialInstanceState().trialId === trial.id &&
    Object.keys(retryProgress.objectiveValues || {}).length === 0,
  `${trial.title} retry should open a fresh instance with clean attempt progress`);
  const runningPresentation = getClassTrialActionPresentation(
    engine.getProgressSnapshot().trials.find((candidate) => candidate.id === trial.id),
    engine.getProgressSnapshot(),
    engine.state.player
  );
  check(runningPresentation.running &&
    runningPresentation.disabled &&
    runningPresentation.buttonLabel === 'Active' &&
    runningPresentation.statusLabel === 'In progress',
  `${trial.title} should disable duplicate starts while its retry instance is running`);
  const alternateTrial = data.CLASS_TRIALS.find((candidate) =>
    candidate.baseClass === trial.baseClass && candidate.id !== trial.id);
  const retrySnapshot = engine.getProgressSnapshot();
  const alternatePresentation = getClassTrialActionPresentation(
    retrySnapshot.trials.find((candidate) => candidate.id === alternateTrial.id),
    retrySnapshot,
    engine.state.player
  );
  check(alternatePresentation.blockedByTrialInstance &&
    alternatePresentation.disabled &&
    alternatePresentation.tooltipAction.includes('Finish the current class trial'),
  `${trial.title} should visibly lock alternate trials while its instance is running`);
  check(!engine.startClassTrial(alternateTrial.id) &&
    engine.getTrialInstanceState().active &&
    engine.getTrialInstanceState().trialId === trial.id &&
    engine.getProgressState().activeTrialId === trial.id,
  `${trial.title} should reject attempts to overwrite its running instance`);
});

[
  {
    label: 'field',
    mapId: 'greenrootMeadow',
    source: 'Dew Slime',
    expectedMessage: 'Recovered at Starfall Crossing after Dew Slime.'
  },
  {
    label: 'dungeon',
    mapId: 'emberjawLair',
    source: 'Emberjaw Sentry',
    expectedMessage: 'Recovered at Starfall Crossing after Emberjaw Sentry.'
  }
].forEach((scenario) => {
  const fixture = createFixture('fighter');
  const { engine, toasts } = fixture;
  if (scenario.label === 'dungeon') {
    engine.state.player.level = 25;
    engine.state.player.advancedClassId = 'guardian';
    engine.invalidateStatsCache();
    check(engine.startDungeon('emberjaw_lair', { silent: true }),
      `${scenario.label} defeat fixture should start a real dungeon run`);
    check(engine.getDungeonState().activeDungeonId === 'emberjaw_lair' &&
      engine.getDungeonState().currentRun &&
      engine.getDungeonState().currentRun.dungeonId === 'emberjaw_lair',
    `${scenario.label} defeat fixture should hold active run state before defeat`);
  } else {
    check(engine.changeMap(scenario.mapId, { silent: true }),
      `${scenario.label} defeat fixture should enter ${scenario.mapId}`);
  }
  fixture.resetSignals();
  const { stats, deathsBefore } = defeatPlayer(engine, scenario.source);
  check(engine.state.mapId === 'starfallCrossing' &&
    engine.runtime.id === 'starfallCrossing' &&
    !engine.getTrialInstanceState().active,
  `${scenario.label} defeat should return to the normal hometown runtime`);
  checkRecoveryVitals(engine, stats, scenario.label);
  check(getDeathCount(engine) === deathsBefore + 1,
    `${scenario.label} defeat should increment the death metric once`);
  if (scenario.label === 'dungeon') {
    check(!engine.getDungeonState().activeDungeonId &&
      !engine.getDungeonState().currentRun,
    'dungeon defeat should clear the abandoned run state');
  }
  check(toasts.length === 1 && toasts[0] === scenario.expectedMessage,
    `${scenario.label} defeat should suppress the generic map-loaded message`);
  check(fixture.getChanges() === 1,
    `${scenario.label} defeat should publish one consolidated state update`);
});

{
  const fixture = createFixture('mage');
  const { engine, toasts } = fixture;
  check(engine.changeMap('endlessRift', { silent: true }),
    'Rift defeat fixture should enter the Endless Rift');
  engine.state.rift.tier = 4;
  engine.state.rift.bankedTier = 2;
  engine.state.rift.checkpointTier = 2;
  engine.state.rift.score = 250;
  engine.state.rift.unbankedBounty = {
    currency: 175,
    materials: { upgradeDust: 3 },
    consumables: {}
  };
  fixture.resetSignals();
  const { stats, deathsBefore } = defeatPlayer(engine, 'Rift Warden');
  const bounty = engine.state.rift.unbankedBounty || {};
  check(engine.state.mapId === 'starfallCrossing' &&
    engine.runtime.id === 'starfallCrossing',
  'Rift defeat should return to the normal hometown runtime');
  check(Number(engine.state.rift.tier || 0) === 2 &&
    Number(bounty.currency || 0) === 0 &&
    Object.keys(bounty.materials || {}).length === 0 &&
    Object.keys(bounty.consumables || {}).length === 0,
  'Rift defeat should reset to the secured checkpoint and clear unbanked bounty');
  checkRecoveryVitals(engine, stats, 'Rift');
  check(getDeathCount(engine) === deathsBefore + 1,
    'Rift defeat should increment the death metric once');
  check(toasts.length === 1 &&
    toasts[0] === 'Rift run fractured. Recovered at Starfall Crossing - unbanked bounty lost.',
  'Rift defeat should publish one complete failure-and-recovery message');
  check(fixture.getChanges() === 1,
    'Rift defeat should publish one consolidated state update');
}

{
  const fixture = createFixture('fighter');
  const { engine } = fixture;
  const trial = data.CLASS_TRIALS.find((candidate) => candidate.baseClass === 'fighter');
  check(engine.startClassTrial(trial.id),
    'Worldwright map-travel fixture should start inside a real advancement trial');
  const firstObjective = trial.objectives[0];
  check(engine.recordProgressEvent('defeat', {
    enemyId: firstObjective.enemyId,
    mapId: engine.state.mapId
  }), 'Worldwright map-travel fixture should record partial trial progress');

  const invalidResult = engine.executeAdminCommand('tp map missingMap');
  check(!invalidResult.ok &&
    invalidResult.message === 'Map not found: missingMap.' &&
    engine.getTrialInstanceState().active,
  'an invalid Worldwright destination should report not found without abandoning the active trial');

  const travelResult = engine.executeAdminCommand('tp map frostfenOutskirts 8289');
  const trialProgress = engine.getProgressState().trialProgress[trial.id];
  check(travelResult.ok &&
    travelResult.message === 'Teleported to Frostfen Outskirts.' &&
    engine.state.mapId === 'frostfenOutskirts' &&
    engine.runtime.id === 'frostfenOutskirts',
  'Worldwright should leave an advancement instance and complete valid map travel');
  check(!engine.getTrialInstanceState().active &&
    engine.getProgressState().activeTrialId === trial.id &&
    !engine.getProgressState().completedTrials[trial.advancedId] &&
    Number(trialProgress.objectiveValues[firstObjective.id] || 0) === 1,
  'Worldwright map travel should close only the instance while preserving retry progress and eligibility');
  check(Math.round(engine.state.player.x) === 8289,
    'Worldwright map travel should still apply the requested map position after leaving a trial');
}

{
  const fixture = createFixture('fighter');
  const { engine } = fixture;
  const trial = data.CLASS_TRIALS.find((candidate) => candidate.baseClass === 'fighter');
  check(engine.startClassTrial(trial.id),
    'Worldwright boss-travel fixture should start inside a real advancement trial');
  const trialRuntimeId = engine.runtime.id;
  const invalidResult = engine.executeAdminCommand('tp boss missingBoss');
  check(!invalidResult.ok &&
    invalidResult.message === 'Boss not found: missingBoss.' &&
    engine.getTrialInstanceState().active &&
    engine.runtime.id === trialRuntimeId,
  'an invalid Worldwright boss should leave the active trial runtime intact');
  engine.state.player.combatLockUntil = Number.MAX_SAFE_INTEGER;
  engine.state.player.movementLockUntil = Number.MAX_SAFE_INTEGER;
  engine.state.player.climbing = true;
  engine.state.player.climbMoving = true;
  engine.state.player.climbableId = 'trial-rope';
  const travelResult = engine.executeAdminCommand('tp boss rimewarden');
  check(travelResult.ok &&
    travelResult.message === 'Teleported to boss rimewarden.' &&
    engine.state.mapId === 'rimewardenVault' &&
    engine.runtime.id === 'rimewardenVault',
  'Worldwright boss travel should leave an advancement instance and enter the requested encounter');
  check(!engine.getTrialInstanceState().active &&
    engine.enemies.some((enemy) => enemy && enemy.id === 'rimewarden'),
  'Worldwright boss travel should clear stale trial state and spawn the requested boss');
  check(!engine.state.player.climbing &&
    !engine.state.player.climbMoving &&
    !engine.state.player.climbableId &&
    engine.state.player.combatLockUntil === 0 &&
    engine.state.player.movementLockUntil === 0,
  'Worldwright boss travel should clear stale climbing, combat, and movement locks atomically');
}

console.log(`Project Starfall defeat recovery tests passed: ${checks} checks.`);
