'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  RIFT_FIRST_CLEAR_REWARD,
  RIFT_OPERATION_DURATION_MS,
  createRiftState,
  createRiftSnapshot,
  finishRiftOperation,
  getRiftOperationWeekWindow,
  startRiftOperation
} = require('../js/games/project-starfall/engine/map-mechanics.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const NOW = Date.UTC(2026, 6, 22, 18, 0, 0);

const migrated = createRiftState({
  tier: 14,
  bestTier: 18,
  score: 321,
  mutationIds: ['burning'],
  startedAt: NOW - 5000,
  mapMechanics: {}
}, { data, nowMs: NOW });
assert.strictEqual(migrated.operationVersion, 1, 'legacy Rift saves should migrate to the operation schema');
assert.strictEqual(migrated.active, false, 'legacy lifetime progress should not become an active timed run');
assert.strictEqual(migrated.tier, 1, 'legacy lifetime tiers should not leak into a new run');
assert.strictEqual(migrated.score, 0, 'legacy lifetime score should not leak into a new run');
assert.strictEqual(migrated.personalBestTier, 18, 'legacy best tier should survive as a personal record');
assert.deepStrictEqual(migrated.mutationIds, [], 'legacy mutation state should not leak into a new run');

let operation = startRiftOperation(migrated, { data, nowMs: NOW });
assert.strictEqual(operation.active, true, 'starting an operation should create an active run');
assert.strictEqual(operation.tier, 1, 'each operation should start at tier one');
assert.strictEqual(operation.score, 0, 'each operation should start with an empty tier score');
assert.strictEqual(operation.runScore, 0, 'each operation should start with an empty run score');
assert.strictEqual(operation.endsAt - operation.startedAt, RIFT_OPERATION_DURATION_MS,
  'operations should have one finite authored duration');

operation.tier = 4;
operation.score = 125;
operation.runScore = 2425;
operation.kills = 31;
const midRunSnapshot = createRiftSnapshot(operation, ['burning'], { data, nowMs: NOW + 90000 });
assert.strictEqual(midRunSnapshot.active, true, 'the snapshot should expose active run state');
assert.strictEqual(midRunSnapshot.remainingMs, RIFT_OPERATION_DURATION_MS - 90000,
  'the snapshot should expose a live remaining timer');
assert.strictEqual(midRunSnapshot.runScore, 2425, 'the snapshot should expose total run score');
assert.strictEqual(midRunSnapshot.personalBest.tier, 18, 'the snapshot should expose migrated personal bests');
assert.strictEqual(midRunSnapshot.weeklyRewardAvailable, true, 'the snapshot should expose first-clear availability');

const death = finishRiftOperation(operation, 'death', { data, nowMs: NOW + 95000 });
assert.strictEqual(death.changed, true, 'death should finish an active operation');
assert.strictEqual(death.summary.reason, 'death', 'death should be persisted as the run outcome');
assert.strictEqual(death.summary.cleared, false, 'death should not count as a clear');
assert.strictEqual(death.reward, null, 'death should not grant the weekly first-clear reward');
assert.strictEqual(death.state.personalBestScore, 2425, 'a finished run should persist its personal best score');
assert.strictEqual(death.state.weekly.bestScore, 2425, 'a finished run should persist its weekly best score');

operation = startRiftOperation(death.state, { data, nowMs: NOW + 100000 });
operation.tier = 5;
operation.runScore = 4100;
operation.kills = 48;
const firstClear = finishRiftOperation(operation, 'timeout', { data, nowMs: operation.endsAt });
assert.deepStrictEqual(firstClear.reward, RIFT_FIRST_CLEAR_REWARD, 'the first weekly clear should grant the capped reward');
assert.strictEqual(firstClear.state.weekly.rewardClaimed, true, 'the weekly reward should be marked claimed atomically');
assert.strictEqual(firstClear.state.weekly.clears, 1, 'the weekly state should count completed operations');
assert.strictEqual(firstClear.state.weekly.bestScore, 4100, 'the weekly record should keep the stronger run');

operation = startRiftOperation(firstClear.state, { data, nowMs: operation.endsAt + 1000 });
const repeatClear = finishRiftOperation(operation, 'timeout', { data, nowMs: operation.endsAt });
assert.strictEqual(repeatClear.reward, null, 'additional clears in the same week should not repeat the reward');
assert.strictEqual(repeatClear.state.weekly.clears, 2, 'repeat clears should still count toward weekly history');

const nextWeekNow = repeatClear.state.weekly.endsAt + 1000;
operation = startRiftOperation(repeatClear.state, { data, nowMs: nextWeekNow });
assert.strictEqual(operation.weekly.rewardClaimed, false, 'a newer weekly cycle should reopen the first-clear reward');
assert.strictEqual(operation.weekly.bestScore, 0, 'a newer weekly cycle should reset the weekly record');
const nextWeekClear = finishRiftOperation(operation, 'timeout', { data, nowMs: operation.endsAt });
assert.deepStrictEqual(nextWeekClear.reward, RIFT_FIRST_CLEAR_REWARD, 'the next weekly cycle should grant one new reward');

const rolledBack = createRiftState(nextWeekClear.state, { data, nowMs: NOW });
assert.strictEqual(rolledBack.weekly.cycleId, nextWeekClear.state.weekly.cycleId,
  'clock rollback should not reopen an older weekly reward window');
assert.strictEqual(rolledBack.weekly.rewardClaimed, true,
  'clock rollback should preserve the claimed state of the latest observed week');
assert(getRiftOperationWeekWindow(NOW).startedAt < rolledBack.weekly.startedAt,
  'the rollback fixture should exercise a genuinely older local week');

function createEngine() {
  const engine = createProjectStarfallEngine(null, data);
  if (!engine.state.player.classId) engine.chooseClass('fighter');
  engine.state.player.level = 100;
  return engine;
}

const engineNow = getRiftOperationWeekWindow(Date.now()).startedAt + 24 * 60 * 60 * 1000;
const engine = createEngine();
assert.strictEqual(engine.changeMap('endlessRift', { silent: true, riftNowMs: engineNow }), true,
  'entering the Rift should start an operation');
assert.strictEqual(engine.state.rift.active, true, 'map entry should leave one active operation');
const firstRunId = engine.state.rift.runId;
engine.addRiftOperationScore(750, { kills: 2 });
assert.strictEqual(engine.state.rift.tier, 2, 'run score should advance operation tiers');
assert.strictEqual(engine.state.rift.score, 250, 'tier advancement should retain only run-local overflow');
assert.strictEqual(engine.state.rift.runScore, 750, 'tier advancement should retain total run score');

const saved = engine.serialize();
const restored = createEngine();
assert.strictEqual(restored.restore(saved), true, 'active Rift state should restore from a normal save payload');
assert.strictEqual(restored.state.rift.active, true, 'a still-live saved operation should remain active after restore');
assert.strictEqual(restored.state.rift.runId, firstRunId, 'save restore should preserve the active run identity');
assert.strictEqual(restored.state.rift.runScore, 750, 'save restore should preserve active run progress');

restored.changeMap('starfallCrossing', { silent: true, riftNowMs: engineNow + 2000 });
assert.strictEqual(restored.state.rift.active, false, 'explicit map exit should end the operation');
assert.strictEqual(restored.state.rift.lastRun.reason, 'exit', 'explicit map exit should persist its outcome');
assert.strictEqual(restored.state.rift.personalBestScore, 750, 'explicit map exit should persist the completed score record');

restored.changeMap('endlessRift', { silent: true, riftNowMs: engineNow + 3000 });
assert.notStrictEqual(restored.state.rift.runId, firstRunId, 're-entry should start a distinct operation');
assert.strictEqual(restored.state.rift.tier, 1, 're-entry should reset run-local tier');
assert.strictEqual(restored.state.rift.runScore, 0, 're-entry should reset run-local score');
const bossEncounter = data.BOSS_ENCOUNTERS[0];
assert(bossEncounter && restored.enterBossEncounter(bossEncounter.id, { silent: true }),
  'direct boss-instance travel should remain available from the Rift');
assert.strictEqual(restored.state.rift.active, false, 'direct boss-instance travel should end the operation');
assert.strictEqual(restored.state.rift.lastRun.reason, 'exit', 'direct boss-instance travel should persist an exit outcome');
restored.changeMap('endlessRift', { silent: true, riftNowMs: engineNow + 4000 });
restored.state.player.invulnerableUntil = 0;
restored.damagePlayer(1000000000, 'test hazard');
assert.strictEqual(restored.state.mapId, 'starfallCrossing', 'death should return the player to Starfall Crossing');
assert.strictEqual(restored.state.rift.lastRun.reason, 'death', 'death should persist a failed operation outcome');

const clearStart = engineNow + 10000;
restored.changeMap('endlessRift', { silent: true, riftNowMs: clearStart });
const currencyBeforeReward = restored.state.player.currency;
const splintersBeforeReward = Number(restored.state.materials.riftSplinter || 0);
const fragmentsBeforeReward = Number(restored.state.materials.cubeFragment || 0);
const clearEndsAt = restored.state.rift.endsAt;
assert.strictEqual(restored.updateRiftOperation({ nowMs: clearEndsAt + 1, silent: true }), true,
  'timeout should finish an active operation');
assert.strictEqual(restored.state.mapId, 'starfallCrossing', 'timeout should return the player to Starfall Crossing');
assert.strictEqual(restored.state.rift.lastRun.reason, 'timeout', 'timeout should persist a clear outcome');
assert.strictEqual(restored.state.player.currency - currencyBeforeReward, RIFT_FIRST_CLEAR_REWARD.currency,
  'timeout should grant the weekly reward currency once');
assert.strictEqual(Number(restored.state.materials.riftSplinter || 0) - splintersBeforeReward,
  RIFT_FIRST_CLEAR_REWARD.materials.riftSplinter, 'timeout should grant the weekly Rift materials once');
assert.strictEqual(Number(restored.state.materials.cubeFragment || 0) - fragmentsBeforeReward,
  RIFT_FIRST_CLEAR_REWARD.materials.cubeFragment, 'timeout should grant the weekly cube materials once');

const rewardedCurrency = restored.state.player.currency;
restored.changeMap('endlessRift', { silent: true, riftNowMs: clearEndsAt + 2000 });
const repeatEndsAt = restored.state.rift.endsAt;
restored.updateRiftOperation({ nowMs: repeatEndsAt + 1, silent: true });
assert.strictEqual(restored.state.player.currency, rewardedCurrency,
  'a second engine clear in the same week should not duplicate the weekly reward');

console.log('Project Starfall Rift operation lifecycle tests passed.');
