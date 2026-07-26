'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const data = require('../js/games/project-starfall/data/index.js');
const progressObjectives = require('../js/games/project-starfall/engine/progress-objectives.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const hud = require('../js/games/project-starfall/ui/hud.js');

const root = path.resolve(__dirname, '..');
let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function getQuest(questId) {
  return data.QUESTS.find((quest) => quest.id === questId);
}

function createQuestEngine(level) {
  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass('fighter'), 'quest regression engines should choose a class');
  if (level) engine.state.player.level = level;
  return engine;
}

function unlockQuests(engine, questIds) {
  questIds.forEach((questId) => {
    if (!engine.state.progress.completedQuestIds.includes(questId)) {
      engine.state.progress.completedQuestIds.push(questId);
    }
    if (!engine.state.progress.claimedQuestIds.includes(questId)) {
      engine.state.progress.claimedQuestIds.push(questId);
    }
  });
}

function moveToQuestNpc(engine, mapId, npcId) {
  check(engine.changeMap(mapId), `quest regression should travel to ${mapId}`);
  const npc = engine.getQuestNpcSnapshot(mapId).npcs.find((candidate) => candidate.id === npcId);
  check(!!npc, `${mapId} should expose ${npcId}`);
  engine.state.player.x = npc.x - 4;
  engine.state.player.y = npc.y - engine.state.player.h + npc.h;
  engine.updateActiveStation();
  return npc;
}

const trialReady = getQuest('trial_ready');
check(trialReady && trialReady.requiredLevel === 20,
  'Ready for Advancement should advertise the same Level 20 gate as its objective');

const migratedProgress = progressObjectives.createProgressState({ activeQuestId: ' first_steps ' });
check(migratedProgress.activeQuestIds.join('|') === 'first_steps',
  'legacy singular active quest saves should migrate into the canonical active quest array');
check(migratedProgress.activeQuestId === 'first_steps' &&
  !Object.keys(migratedProgress).includes('activeQuestId'),
  'the legacy active quest accessor should remain readable without being serialized');

const normalizedProgress = progressObjectives.createProgressState({
  activeQuestIds: [' field_scout ', 'field_scout', '', 'greenroot_samples'],
  activeQuestId: 'first_steps'
});
check(normalizedProgress.activeQuestIds.join('|') === 'field_scout|greenroot_samples',
  'canonical active quest arrays should trim, deduplicate, and take precedence over stale legacy state');

const legacySaveEngine = createQuestEngine();
const legacyPayload = legacySaveEngine.serialize();
delete legacyPayload.state.progress.activeQuestIds;
legacyPayload.state.progress.activeQuestId = ' first_steps ';
const migratedSaveEngine = createProjectStarfallEngine(null, data);
check(migratedSaveEngine.restore(legacyPayload) &&
  migratedSaveEngine.state.progress.activeQuestIds.join('|') === 'first_steps',
  'public restore should migrate a legacy active quest into the concurrent journal');
const canonicalPayload = migratedSaveEngine.serialize();
check(canonicalPayload.state.progress.activeQuestIds.join('|') === 'first_steps' &&
  !Object.prototype.hasOwnProperty.call(canonicalPayload.state.progress, 'activeQuestId'),
  'migrated saves should persist only the canonical active quest array');
const reloadedSaveEngine = createProjectStarfallEngine(null, data);
check(reloadedSaveEngine.restore(canonicalPayload) &&
  reloadedSaveEngine.state.progress.activeQuestIds.join('|') === 'first_steps' &&
  reloadedSaveEngine.getProgressSnapshot().activeQuest.id === 'first_steps',
  'canonical concurrent quest state should survive a second save and reload');

const gatedEngine = createQuestEngine(19);
unlockQuests(gatedEngine, ['first_steps', 'field_scout']);
check(gatedEngine.getQuestAvailability('trial_ready').locked &&
  gatedEngine.getQuestAvailability('trial_ready').lockedReason === 'Reach Level 20 first.' &&
  !gatedEngine.startQuest('trial_ready'),
  'Ready for Advancement should remain unavailable before Level 20');

const handoffEngine = createQuestEngine();
check(handoffEngine.startQuest('first_steps'),
  'the first journey quest should start for the handoff regression');
handoffEngine.state.progress.completedQuestIds.push('first_steps');
check(handoffEngine.claimQuestReward('first_steps'),
  'the first journey reward should be claimable in the handoff regression');
const fieldScoutAvailability = handoffEngine.getQuestAvailability('field_scout');
const fieldScoutGuidance = handoffEngine.getQuestGuidanceSnapshot();
check(handoffEngine.state.session.questGuide.type === 'quest' &&
  handoffEngine.state.session.questGuide.id === 'field_scout' &&
  fieldScoutAvailability.available &&
  !fieldScoutAvailability.active &&
  !handoffEngine.state.progress.activeQuestIds.includes('field_scout'),
  'claiming the focused first quest should guide its available successor without auto-accepting it');
check(fieldScoutGuidance.active &&
  fieldScoutGuidance.targetId === 'field_scout' &&
  fieldScoutGuidance.objectiveType === 'talk' &&
  fieldScoutGuidance.recommendedMapId === 'thornpathThicket' &&
  fieldScoutGuidance.targetNpcId === 'thornpath_scout' &&
  fieldScoutGuidance.objectiveLabel === 'Accept Thornpath Field Scout from Thornpath Scout',
  'available successor guidance should route the player to its quest NPC and map');

moveToQuestNpc(handoffEngine, 'thornpathThicket', 'thornpath_scout');
check(handoffEngine.acceptQuestFromNpc('thornpath_scout', 'field_scout'),
  'the guided successor should still require explicit NPC acceptance');
const acceptedFieldScoutGuidance = handoffEngine.getQuestGuidanceSnapshot();
check(handoffEngine.state.progress.activeQuestIds.includes('field_scout') &&
  acceptedFieldScoutGuidance.targetId === 'field_scout' &&
  acceptedFieldScoutGuidance.objectiveType === 'defeat' &&
  acceptedFieldScoutGuidance.targetEnemyIds.includes('mossback'),
  'accepting the guided successor should advance guidance from its NPC handoff to its first incomplete objective');

const lockedHandoffEngine = createQuestEngine(3);
unlockQuests(lockedHandoffEngine, ['first_steps']);
check(lockedHandoffEngine.startQuest('field_scout'),
  'the field scout quest should start for the locked-successor regression');
lockedHandoffEngine.state.progress.completedQuestIds.push('field_scout');
const lockedHandoffMessages = [];
lockedHandoffEngine.toast = (message) => {
  lockedHandoffMessages.push(String(message || ''));
};
check(lockedHandoffEngine.claimQuestReward('field_scout'),
  'the field scout reward should be claimable before the advancement level gate');
const lockedTrialAvailability = lockedHandoffEngine.getQuestAvailability('trial_ready');
check(lockedTrialAvailability.locked &&
  lockedTrialAvailability.lockedReason === 'Reach Level 20 first.' &&
  lockedHandoffEngine.state.session.questGuide.type === '' &&
  lockedHandoffMessages.some((message) =>
    message.includes('Ready for Advancement remains locked: Reach Level 20 first.')) &&
  !lockedHandoffMessages.some((message) =>
    message.includes('Ready for Advancement is available')),
  'claiming a quest should report its successor level gate instead of falsely calling the quest available');

const concurrentEngine = createQuestEngine();
unlockQuests(concurrentEngine, ['first_steps']);
check(concurrentEngine.startQuest('field_scout') &&
  concurrentEngine.startQuest('greenroot_samples'),
  'two independently available quests should be accepted without a global journal lock');
check(concurrentEngine.state.progress.activeQuestIds.join('|') === 'field_scout|greenroot_samples' &&
  concurrentEngine.getQuestAvailability('field_scout').active &&
  concurrentEngine.getQuestAvailability('greenroot_samples').active,
  'both accepted quests should remain active in acceptance order');

let concurrentSnapshot = concurrentEngine.getProgressSnapshot();
check(concurrentSnapshot.activeQuests.map((quest) => quest.id).join('|') === 'field_scout|greenroot_samples' &&
  concurrentSnapshot.activeQuest.id === 'greenroot_samples' &&
  concurrentSnapshot.activeQuests.find((quest) => quest.id === 'greenroot_samples').focused,
  'the journal should expose every accepted quest while preserving one explicit focus');

const trackerEntries = hud.getQuestTrackerEntries({
  onboarding: { hidden: true },
  progress: concurrentEngine.getProgressTrackerSnapshot()
});
const trackedAcceptedQuests = trackerEntries.filter((entry) => entry.guideType === 'quest');
check(trackedAcceptedQuests.length === 1 &&
  trackedAcceptedQuests[0].guideId === 'greenroot_samples',
  'the HUD tracker should show only the focused accepted quest');

check(concurrentEngine.recordProgressEvent('defeat', {
  enemyId: 'mossback',
  mapId: 'thornpathThicket',
  count: 1
}), 'a non-focused accepted quest should receive matching progress events');
check(concurrentEngine.getQuestSummary(getQuest('field_scout')).objectives
  .find((objective) => objective.id === 'defeat_mossbacks').value === 1,
  'the non-focused field quest should retain its Mossback progress');

check(concurrentEngine.recordProgressEvent('defeat', {
  enemyId: 'dewSlime',
  mapId: 'greenrootMeadow',
  count: 1
}), 'the focused side quest should receive its own matching progress events');
check(concurrentEngine.getQuestSummary(getQuest('greenroot_samples')).objectives
  .find((objective) => objective.id === 'defeat_dew_slimes').value === 1,
  'the focused Greenroot quest should retain its Dew Slime progress');

concurrentEngine.recordProgressEvent('travel', { mapId: 'thornpathThicket' });
concurrentEngine.recordProgressEvent('defeat', {
  enemyId: 'mossback',
  mapId: 'thornpathThicket',
  count: 1
});
concurrentEngine.recordProgressEvent('defeat', {
  enemyId: 'thornSprout',
  mapId: 'thornpathThicket',
  count: 2
});
check(concurrentEngine.state.progress.completedQuestIds.includes('field_scout') &&
  concurrentEngine.state.progress.activeQuestIds.join('|') === 'greenroot_samples',
  'completing one accepted quest should remove only that quest from the active journal');
check(concurrentEngine.getQuestSummary(getQuest('greenroot_samples')).objectives
  .find((objective) => objective.id === 'defeat_dew_slimes').value === 1,
  'completing another quest should not reset the surviving quest progress');
check(concurrentEngine.claimQuestReward('field_scout') &&
  concurrentEngine.state.progress.claimedQuestIds.includes('field_scout') &&
  concurrentEngine.state.progress.activeQuestIds.join('|') === 'greenroot_samples',
  'claiming one completed quest should leave the surviving accepted quest untouched');

const npcEngine = createQuestEngine(20);
unlockQuests(npcEngine, ['first_steps', 'field_scout']);
check(npcEngine.startQuest('trial_ready'),
  'the Level 20 advancement quest should be accepted alongside regional quests');
moveToQuestNpc(npcEngine, 'rustcoilOutpost', 'rustcoil_foreman');
check(npcEngine.getQuestNpcPrompt('rustcoil_foreman', 'accept', 'rustcoil_relay') &&
  npcEngine.acceptQuestFromNpc('rustcoil_foreman', 'rustcoil_relay') &&
  npcEngine.state.progress.activeQuestIds.join('|') === 'trial_ready|rustcoil_relay',
  'NPC acceptance should add a second quest without replacing the first');

moveToQuestNpc(npcEngine, 'rustcoilRuins', 'ruins_surveyor');
const talkPrompt = npcEngine.getQuestNpcPrompt('ruins_surveyor', 'talk', 'rustcoil_relay');
check(talkPrompt && talkPrompt.questId === 'rustcoil_relay' &&
  npcEngine.completeQuestTalkObjective('ruins_surveyor', 'rustcoil_relay'),
  'NPC talk routing should resolve the requested objective among concurrent quests');
check(npcEngine.state.progress.completedQuestIds.includes('rustcoil_relay') &&
  npcEngine.state.progress.activeQuestIds.join('|') === 'trial_ready',
  'talk completion should isolate the completed quest and preserve the advancement quest');
check(!npcEngine.claimQuestRewardFromNpc('ruins_surveyor', 'rustcoil_relay'),
  'the talk target should not become the completed quest reward owner');

moveToQuestNpc(npcEngine, 'rustcoilOutpost', 'rustcoil_foreman');
check(npcEngine.getQuestNpcPrompt('rustcoil_foreman', 'claim', 'rustcoil_relay') &&
  npcEngine.claimQuestRewardFromNpc('rustcoil_foreman', 'rustcoil_relay'),
  'NPC claim routing should remain anchored to the owning quest giver');
check(npcEngine.state.progress.activeQuestIds.join('|') === 'trial_ready' &&
  npcEngine.state.session.questGuide.type === 'quest' &&
  npcEngine.state.session.questGuide.id === 'trial_ready' &&
  npcEngine.getProgressTrackerSnapshot().activeQuest.id === 'trial_ready',
  'claiming the focused quest should move the single guide focus to the surviving quest');

const uiSource = fs.readFileSync(
  path.join(root, 'js/games/project-starfall/project-starfall-ui.js'),
  'utf8'
);
check(uiSource.includes("'Accepted Quests'") &&
  uiSource.includes('progress.activeQuests') &&
  uiSource.includes("type: 'quest-guide'"),
  'the quest journal should render every accepted quest as a selectable HUD focus');

console.log(`Project Starfall concurrent quest checks passed: ${checks}`);
