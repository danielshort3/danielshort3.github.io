'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function createProgressionEngine(level) {
  const engine = createProjectStarfallEngine(null, data);
  assert.strictEqual(engine.chooseClass('fighter'), true, 'the test character should initialize');
  engine.state.player.level = level;
  engine.state.progress.activeQuestId = '';
  engine.state.progress.completedQuestIds = ['first_steps', 'field_scout'];
  engine.state.progress.claimedQuestIds = ['first_steps', 'field_scout'];
  return engine;
}

function placeAtQuestNpc(engine, npcId) {
  const npc = engine.getQuestNpcById(npcId);
  assert(npc, `${npcId} should exist on the current map`);
  engine.state.player.x = npc.x;
  engine.state.player.y = npc.y;
  return npc;
}

const advancementQuest = data.QUESTS.find((quest) => quest.id === 'trial_ready');
assert(advancementQuest, 'Ready for Advancement should exist');
assert.strictEqual(advancementQuest.requiredLevel, 20,
  'Ready for Advancement should be unavailable before the trial level');

const gatedEngine = createProgressionEngine(1);
for (let level = 1; level < 20; level += 1) {
  gatedEngine.state.player.level = level;
  const availability = gatedEngine.getQuestAvailability('trial_ready');
  assert.strictEqual(availability.available, false, `level ${level} should not unlock the advancement quest`);
  assert.strictEqual(availability.locked, true, `level ${level} should report the advancement quest as locked`);
  assert.strictEqual(availability.lockedReason, 'Reach Level 20 first.');
  assert.strictEqual(gatedEngine.startQuest('trial_ready'), false,
    `level ${level} should not start an unavailable advancement quest`);
  assert.strictEqual(gatedEngine.state.progress.activeQuestId, '',
    `level ${level} should leave the active quest slot open`);
}
placeAtQuestNpc(gatedEngine, 'crossing_class_master');
assert.strictEqual(gatedEngine.acceptQuestFromNpc('crossing_class_master', 'trial_ready'), false,
  'the Class Master should not accept the advancement quest below level 20');
assert.strictEqual(gatedEngine.state.progress.activeQuestId, '');

const levelTwentyEngine = createProgressionEngine(20);
placeAtQuestNpc(levelTwentyEngine, 'crossing_class_master');
assert.strictEqual(levelTwentyEngine.getQuestAvailability('trial_ready').available, true,
  'level 20 should unlock the advancement quest');
assert.strictEqual(levelTwentyEngine.acceptQuestFromNpc('crossing_class_master', 'trial_ready'), true,
  'the Class Master should accept the advancement quest at level 20');
assert.strictEqual(levelTwentyEngine.state.progress.activeQuestId, 'trial_ready');
assert(levelTwentyEngine.state.progress.activeTrialId,
  'accepting Ready for Advancement should start a real class trial');
assert.strictEqual(levelTwentyEngine.getTrialInstanceSnapshot().active, true,
  'the level-20 handoff should enter the trial instance');

const legacySource = createProgressionEngine(8);
legacySource.state.progress.activeQuestId = 'trial_ready';
legacySource.state.progress.questProgress = {
  trial_ready: {
    objectiveValues: { reach_20: 8, complete_trial: 0 },
    completedAt: 0
  },
  greenroot_samples: {
    objectiveValues: { defeat_glassbacks: 2, collect_star_glass: 1 },
    completedAt: 0
  }
};
legacySource.state.session.questGuide = { type: 'quest', id: 'trial_ready' };
const legacyPayload = legacySource.serialize();
const legacyCompleted = legacyPayload.state.progress.completedQuestIds.slice();
const legacyClaimed = legacyPayload.state.progress.claimedQuestIds.slice();
const legacyQuestProgress = JSON.parse(JSON.stringify(legacyPayload.state.progress.questProgress));

const migratedEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(migratedEngine.restore(legacyPayload), true, 'the legacy save should restore');
assert.strictEqual(migratedEngine.state.progress.activeQuestId, '',
  'an under-level legacy advancement quest should return to its locked-at-20 state');
assert.deepStrictEqual(migratedEngine.state.progress.completedQuestIds, legacyCompleted,
  'legacy completed quest records should be preserved');
assert.deepStrictEqual(migratedEngine.state.progress.claimedQuestIds, legacyClaimed,
  'legacy claimed quest records should be preserved');
Object.entries(legacyQuestProgress).forEach(([questId, entry]) => {
  assert.deepStrictEqual(migratedEngine.state.progress.questProgress[questId], entry,
    `${questId} legacy objective progress should be preserved`);
});
assert.strictEqual(migratedEngine.getQuestAvailability('trial_ready').lockedReason, 'Reach Level 20 first.');
assert.strictEqual(migratedEngine.getQuestAvailability('greenroot_samples').available, true,
  'repairing the invalid active quest should unblock eligible regional quests');
migratedEngine.state.player.level = 20;
assert.strictEqual(migratedEngine.getQuestAvailability('trial_ready').available, true,
  'the repaired legacy quest should become normally available at level 20');
assert.strictEqual(migratedEngine.state.progress.activeQuestId, '',
  'reaching level 20 should not silently reactivate or auto-claim the repaired quest');

const handoffEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(handoffEngine.chooseClass('fighter'), true);
assert.strictEqual(handoffEngine.travelToMap('greenrootMeadow'), true,
  'Starfall Verge should be reachable for the first quest handoff');
handoffEngine.state.progress.activeQuestId = '';
handoffEngine.state.progress.completedQuestIds = ['first_steps'];
handoffEngine.state.progress.claimedQuestIds = [];
assert.strictEqual(handoffEngine.claimQuestReward('first_steps'), true,
  'First Expedition should award its completed reward');
assert.deepStrictEqual(handoffEngine.state.session.questGuide, { type: 'quest', id: 'greenroot_samples' },
  'claiming First Expedition should pin the immediately actionable Verge follow-up');

const guidance = handoffEngine.getQuestGuidanceSnapshot();
assert.strictEqual(guidance.active, true);
assert.strictEqual(guidance.targetType, 'quest');
assert.strictEqual(guidance.targetId, 'greenroot_samples');
assert.strictEqual(guidance.objectiveType, 'talk',
  'an unaccepted quest should guide the player to its quest giver');
assert.strictEqual(guidance.targetNpcId, 'greenroot_guide');
assert.strictEqual(guidance.recommendedMapId, 'greenrootMeadow');
assert(guidance.navigationTarget && guidance.navigationTarget.kind === 'npc',
  'the handoff should resolve to a real NPC navigation target');

const handoffPayload = handoffEngine.serialize();
const restoredHandoffEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(restoredHandoffEngine.restore(handoffPayload), true,
  'the corrected handoff save should restore');
assert.deepStrictEqual(restoredHandoffEngine.state.session.questGuide, { type: 'quest', id: 'greenroot_samples' },
  'the actionable handoff should persist through save and restore');
const restoredGuidance = restoredHandoffEngine.getQuestGuidanceSnapshot();
assert.strictEqual(restoredGuidance.targetNpcId, 'greenroot_guide');
assert.strictEqual(restoredGuidance.recommendedMapId, 'greenrootMeadow');
assert(restoredGuidance.navigationTarget && restoredGuidance.navigationTarget.kind === 'npc',
  'restored guidance should still resolve to the quest giver');

const fieldScoutEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(fieldScoutEngine.chooseClass('fighter'), true);
assert.strictEqual(fieldScoutEngine.changeMap('thornpathThicket'), true,
  'the regression character should stand beside the Thornpath quest giver');
fieldScoutEngine.state.player.level = 8;
fieldScoutEngine.state.progress.activeQuestId = '';
fieldScoutEngine.state.progress.completedQuestIds = ['first_steps', 'field_scout'];
fieldScoutEngine.state.progress.claimedQuestIds = ['first_steps'];
const fieldScoutToasts = [];
fieldScoutEngine.setToastHandler((message) => {
  fieldScoutToasts.push(typeof message === 'string' ? message : message && message.message || '');
});
assert.strictEqual(fieldScoutEngine.claimQuestReward('field_scout'), true,
  'the level-8 field scout reward should be claimable');
assert.strictEqual(fieldScoutEngine.getQuestAvailability('trial_ready').lockedReason, 'Reach Level 20 first.',
  'the advancement quest should remain locked after the level-8 field scout claim');
assert(!fieldScoutToasts.some((message) => message.includes('Ready for Advancement is available')),
  'the reward toast should not advertise a locked advancement quest');
assert(fieldScoutToasts.some((message) => message.includes('Courier to the Ridge is available from Thornpath Scout.')),
  'the reward toast should advertise the live regional handoff');
assert.deepStrictEqual(fieldScoutEngine.state.session.questGuide, { type: 'quest', id: 'ridge_courier' },
  'the level-8 handoff should pin an available regional quest');
const fieldScoutGuidance = fieldScoutEngine.getQuestGuidanceSnapshot();
assert.strictEqual(fieldScoutGuidance.active, true);
assert.strictEqual(fieldScoutGuidance.targetId, 'ridge_courier');
assert.strictEqual(fieldScoutGuidance.objectiveType, 'talk');
assert.strictEqual(fieldScoutGuidance.targetNpcId, 'thornpath_scout');
assert.strictEqual(fieldScoutGuidance.recommendedMapId, 'thornpathThicket');
assert(fieldScoutGuidance.navigationTarget && fieldScoutGuidance.navigationTarget.kind === 'npc',
  'the field scout handoff should resolve to the real Thornpath Scout NPC');

console.log('Project Starfall quest handoff tests passed.');
