'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const steps = data.ONBOARDING_STEPS;
assert.strictEqual(steps.length, 6, 'the first expedition guide should contain six focused beats');
assert.strictEqual(new Set(steps.map((step) => step.id)).size, steps.length,
  'first expedition guide step ids should be unique');
assert.deepStrictEqual(steps.map((step) => step.event), [
  'move',
  'attack',
  'travel',
  'defeat',
  'loot',
  'questClaim'
]);

const travelStep = steps.find((step) => step.id === 'travel_greenroot');
const defeatStep = steps.find((step) => step.id === 'defeat_enemy');
const lootStep = steps.find((step) => step.id === 'loot_drop');
const claimStep = steps.find((step) => step.id === 'claim_first_steps');
assert.strictEqual(travelStep.mapId, 'greenrootMeadow', 'travel should require Starfall Verge');
assert.strictEqual(defeatStep.mapId, 'greenrootMeadow', 'the first defeat should happen in Starfall Verge');
assert.strictEqual(defeatStep.enemyId, 'glassback', 'the first defeat should require a Glassback');
assert.strictEqual(defeatStep.count, 3, 'the guided Glassback target should match the quest objective');
assert.strictEqual(lootStep.mapId, 'greenrootMeadow', 'the first field drop should come from Starfall Verge');
assert.strictEqual(claimStep.questId, 'first_steps', 'the report step should require the First Expedition claim');

const engine = createProjectStarfallEngine(null, data);
assert.strictEqual(engine.chooseClass('fighter'), true, 'a valid new character should be created');
assert.strictEqual(engine.state.progress.activeQuestId, 'first_steps',
  'a new character should automatically start First Expedition');
assert.deepStrictEqual(engine.state.session.questGuide, { type: 'quest', id: 'first_steps' },
  'First Expedition should be the pinned quest guidance target');

function expectNextStep(id) {
  const snapshot = engine.getOnboardingSnapshot();
  assert.strictEqual(snapshot.nextStep && snapshot.nextStep.id, id);
}

expectNextStep('learn_move');
assert.strictEqual(engine.recordOnboardingEvent('defeat', {
  enemyId: 'riftLantern',
  mapId: 'greenrootMeadow'
}, { silent: true }), false, 'another opening enemy should not satisfy the Glassback beat');
assert.strictEqual(engine.recordOnboardingEvent('travel', {
  mapId: 'thornpathThicket'
}, { silent: true }), false, 'travel to another map should not satisfy the Verge beat');
assert.strictEqual(engine.recordOnboardingEvent('loot', {
  mapId: 'starfallCrossing',
  kind: 'material'
}, { silent: true }), false, 'town inventory changes should not satisfy the field-drop beat');
assert.strictEqual(engine.recordOnboardingEvent('questClaim', {
  questId: 'field_scout'
}, { silent: true }), false, 'another quest claim should not satisfy the Quartermaster report');

assert.strictEqual(engine.recordOnboardingEvent('move', { input: 'right' }, { silent: true }), true);
expectNextStep('learn_attack');
assert.strictEqual(engine.recordOnboardingEvent('attack', { action: 'attack' }, { silent: true }), true);
expectNextStep('travel_greenroot');
assert.strictEqual(engine.recordOnboardingEvent('travel', { mapId: 'greenrootMeadow' }, { silent: true }), true);
expectNextStep('defeat_enemy');
assert.strictEqual(engine.recordOnboardingEvent('defeat', {
  enemyId: 'glassback',
  mapId: 'greenrootMeadow'
}, { silent: true }), true);
expectNextStep('defeat_enemy');
assert.strictEqual(engine.getOnboardingSnapshot().nextStep.progress, 1);
assert.strictEqual(engine.getOnboardingSnapshot().nextStep.goal, 3);
assert.strictEqual(engine.recordOnboardingEvent('defeat', {
  enemyId: 'glassback',
  mapId: 'greenrootMeadow'
}, { silent: true }), true);
expectNextStep('defeat_enemy');
assert.strictEqual(engine.getOnboardingSnapshot().nextStep.progress, 2);
assert.strictEqual(engine.recordOnboardingEvent('defeat', {
  enemyId: 'glassback',
  mapId: 'greenrootMeadow'
}, { silent: true }), true);
expectNextStep('loot_drop');
assert.strictEqual(engine.recordOnboardingEvent('loot', {
  mapId: 'greenrootMeadow',
  kind: 'material',
  materialId: 'starGlassChip'
}, { silent: true }), true);
expectNextStep('claim_first_steps');

engine.recordProgressEvent('travel', { mapId: 'greenrootMeadow' }, { noEmit: true });
for (let index = 0; index < 3; index += 1) {
  engine.recordProgressEvent('defeat', {
    enemyId: 'glassback',
    mapId: 'greenrootMeadow',
    count: 1
  }, { noEmit: true });
}
engine.recordProgressEvent('loot', {
  kind: 'material',
  materialId: 'starGlassChip',
  itemId: 'starGlassChip',
  mapId: 'greenrootMeadow',
  count: 1
}, { noEmit: true });
assert(engine.state.progress.completedQuestIds.includes('first_steps'),
  'normal travel, defeat, and loot events should complete First Expedition');

assert.strictEqual(engine.claimQuestReward('first_steps'), true,
  'the completed First Expedition reward should be claimable once');
const claimedCurrency = engine.state.player.currency;
assert.strictEqual(engine.getOnboardingSnapshot().nextStep, null,
  'claiming First Expedition should complete the final guide beat');
assert.strictEqual(engine.getOnboardingSnapshot().completeCount, 6);
assert.strictEqual(engine.claimQuestReward('first_steps'), false,
  'a duplicate First Expedition reward claim should remain blocked');
assert.strictEqual(engine.state.player.currency, claimedCurrency,
  'a blocked duplicate claim should not award currency again');
assert.strictEqual(engine.state.progress.claimedQuestIds.filter((id) => id === 'first_steps').length, 1,
  'First Expedition should have one claimed-reward record');

const legacyEngine = createProjectStarfallEngine(null, data);
legacyEngine.chooseClass('mage');
legacyEngine.state.onboarding = {
  hidden: true,
  completedIds: ['learn_move', 'open_worldmap']
};
legacyEngine.state.progress.activeQuestId = '';
legacyEngine.state.progress.completedQuestIds = ['first_steps'];
legacyEngine.state.progress.claimedQuestIds = ['first_steps'];
const legacySnapshot = legacyEngine.getOnboardingSnapshot();
assert.strictEqual(legacySnapshot.hidden, true, 'an existing hidden-guide preference should be preserved');
assert.strictEqual(legacySnapshot.completeCount, 6,
  'an existing First Expedition claim should backfill every new guide beat without another reward claim');
assert.deepStrictEqual(new Set(legacySnapshot.completedIds), new Set(steps.map((step) => step.id)),
  'claimed existing saves should not regress to movement or combat tutorial prompts');
assert.strictEqual(legacySnapshot.nextStep, null);

console.log('Project Starfall guided First Expedition tests passed.');
