'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const hud = require('../js/games/project-starfall/ui/hud.js');

function getDungeonTracker(activeDungeon) {
  return hud.getQuestTrackerEntries({
    progress: { activeQuest: null, activeTrial: null, claimableQuests: [] },
    dungeon: { activeDungeon },
    onboarding: { hidden: true }
  }).find((entry) => entry.guideType === 'dungeon');
}

function createBrambleDungeon(encounterFlow) {
  return {
    id: 'bramble_depths',
    name: 'Bramble Depths',
    bossName: 'Brambleking',
    runComplete: false,
    encounterFlow
  };
}

const rootGateTracker = getDungeonTracker(createBrambleDungeon({
  status: 'active',
  hud: {
    title: 'Route 1/3',
    label: 'Break the Root Gate',
    status: '4 enemies remaining',
    value: 0,
    goal: 4,
    complete: false,
    activeGateX: 1200
  }
}));
assert(rootGateTracker, 'the active Bramble route should remain guideable in the quest tracker');
assert.strictEqual(rootGateTracker.title, 'Route 1/3');
assert.deepStrictEqual(rootGateTracker.objectives, [{
  label: 'Break the Root Gate',
  value: 0,
  goal: 4,
  complete: false,
  status: '4 enemies remaining'
}], 'the active route should use one concise lead objective');

const brambleSealTracker = getDungeonTracker(createBrambleDungeon({
  status: 'active',
  hud: {
    title: 'Route 2/3',
    label: 'Purge the Bramble Seal',
    status: '2 enemies remaining',
    value: 2,
    goal: 4,
    complete: false,
    activeGateX: 3200
  }
}));
assert.strictEqual(brambleSealTracker.title, 'Route 2/3');
assert.deepStrictEqual(brambleSealTracker.objectives, [{
  label: 'Purge the Bramble Seal',
  value: 2,
  goal: 4,
  complete: false,
  status: '2 enemies remaining'
}]);

const bossIntroTracker = getDungeonTracker(createBrambleDungeon({
  status: 'boss_intro',
  bossReveal: { active: true, introActive: true, remainingMs: 2600 },
  hud: {
    title: 'Route 3/3',
    label: 'Challenge the Brambleking',
    status: 'Boss arming in 3s',
    value: 0,
    goal: 1,
    complete: false,
    activeGateX: 0
  }
}));
assert.strictEqual(bossIntroTracker.title, 'Route 3/3');
assert.deepStrictEqual(bossIntroTracker.objectives, [{
  label: 'Challenge the Brambleking',
  value: 0,
  goal: 1,
  complete: false,
  status: 'Boss arming in 3s'
}], 'the boss beat should replace the generic boss directive with its live reveal status');
assert(!bossIntroTracker.objectives.some((objective) => objective.label === 'Defeat Brambleking'),
  'the generic boss objective should not compete with active route direction');

const legacyTracker = getDungeonTracker(createBrambleDungeon(null));
assert.strictEqual(legacyTracker.title, 'Bramble Depths');
assert.deepStrictEqual(legacyTracker.objectives, [{
  label: 'Defeat Brambleking',
  value: 0,
  goal: 1,
  complete: false
}], 'a dungeon without encounter flow should retain the generic boss directive');

const incompleteFlowTracker = getDungeonTracker(createBrambleDungeon({ status: 'active' }));
assert.strictEqual(incompleteFlowTracker.title, 'Bramble Depths');
assert.strictEqual(incompleteFlowTracker.objectives[0].label, 'Defeat Brambleking',
  'route presentation should activate only when the engine publishes a HUD contract');

const completedTracker = getDungeonTracker(Object.assign(createBrambleDungeon({
  status: 'complete',
  hud: {
    title: 'Expedition Route Complete',
    label: 'Expedition route secured',
    status: 'Expedition route secured',
    value: 3,
    goal: 3,
    complete: true
  }
}), { runComplete: true }));
assert.strictEqual(completedTracker.title, 'Expedition Route Complete');
assert.deepStrictEqual(completedTracker.objectives, [{
  label: 'Expedition route secured',
  value: 3,
  goal: 3,
  complete: true,
  status: 'Expedition route secured'
}], 'a completed route should not fall back to an incorrect defeat-the-boss instruction');

const hudSource = fs.readFileSync(path.join(
  __dirname,
  '..',
  'js',
  'games',
  'project-starfall',
  'ui',
  'hud.js'
), 'utf8');
const uiSource = fs.readFileSync(path.join(
  __dirname,
  '..',
  'js',
  'games',
  'project-starfall',
  'project-starfall-ui.js'
), 'utf8');
assert(hudSource.includes('activeDungeon.encounterFlow.hud'),
  'the HUD helper should consume the engine-owned route presentation contract');
[
  'bramble_depths',
  'Break the Root Gate',
  'Purge the Bramble Seal',
  'Challenge the Brambleking'
].forEach((routeSpecificText) => {
  assert(!hudSource.includes(routeSpecificText),
    `the shared HUD should not hard-code Bramble route text: ${routeSpecificText}`);
});
assert(uiSource.includes("getHudWidgetHelper('getQuestTrackerEntries')"),
  'the shipped UI should delegate quest tracker construction to the tested HUD helper');

console.log('Project Starfall Bramble route HUD checks passed.');
