'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const hud = require(path.join(ROOT, 'js/games/project-starfall/ui/hud.js'));
const data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const { createProjectStarfallEngine } = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-engine.js'));

function withPlayer(bossEncounter) {
  return {
    state: { player: { classId: 'fighter', level: 30, hp: 500 } },
    stats: { maxHp: 500 },
    classData: { name: 'Fighter' },
    bossEncounter
  };
}

const callBoss = {
  active: true,
  bossName: 'Quarry Colossus',
  hpRatio: 0.8,
  phaseCount: 3,
  phaseIndex: 1,
  phaseName: 'Overclock',
  pendingActionId: 'plateExpose',
  pendingActionLabel: 'OPEN PLATES',
  pendingActionProgress: 0.4,
  pendingSpatialResponseType: 'damageWindow',
  pendingSpatialSectionLabel: 'Gear Switch Shelf',
  pendingSpatialInstruction: 'Reach the called switch section and damage the boss while its plates are open.'
};

const callFeedback = hud.getBossResponseFeedbackMetadata(callBoss);
assert(callFeedback, 'a spatial telegraph should produce boss-call feedback');
assert.strictEqual(callFeedback.tone, 'call');
assert.strictEqual(callFeedback.title, 'Prepare counter - Gear Switch Shelf');
assert(callFeedback.detail.includes('damage the boss'), 'the call should retain the measurable response instruction');

const callHud = hud.getCanvasBossEncounterHudMetadata(callBoss, 1280);
assert(callHud.feedback && callHud.box.h > 72, 'the canvas boss HUD should reserve a dedicated response ribbon');
assert.strictEqual(callHud.feedback.titleText.value, callFeedback.title);
assert.strictEqual(callHud.feedback.detailText.value, callFeedback.detail);

const drawnTexts = [];
const drawContext = {
  save() {},
  restore() {},
  fillRect() {},
  fillStyle: '',
  shadowColor: '',
  shadowBlur: 0
};
assert.strictEqual(hud.drawCanvasBossEncounterHud(drawContext, callBoss, 1280, {
  drawRoundRect() {},
  drawCanvasText(entry) { drawnTexts.push(entry.value); }
}), true);
assert(drawnTexts.includes('BOSS CALL') && drawnTexts.includes('Prepare counter - Gear Switch Shelf'),
  'the canvas renderer should draw both the response state and actionable title');

const introWithCall = hud.getCanvasBossEncounterOverlayMetadata(Object.assign({}, callBoss, {
  intro: { text: 'The foundry wakes.' }
}), 1280, 806);
assert.strictEqual(introWithCall.box.y, 136, 'the intro card should move below an active response ribbon');

const callAnnouncement = hud.getHudStatusAnnouncement(withPlayer(callBoss));
const progressedCallAnnouncement = hud.getHudStatusAnnouncement(withPlayer(Object.assign({}, callBoss, {
  pendingActionProgress: 0.85
})));
assert(callAnnouncement.message.includes('Boss response: Boss call.'), 'the polite HUD status should announce a new boss call');
assert.strictEqual(callAnnouncement.signature, progressedCallAnnouncement.signature,
  'telegraph animation progress should not repeatedly announce the same boss call');

const newCallAfterSuccess = hud.getBossResponseFeedbackMetadata(Object.assign({}, callBoss, {
  pendingSpatialSectionLabel: 'Current Call Section',
  lastSpatialResponse: {
    id: 'old-success',
    status: 'success',
    sectionLabel: 'Old Response Section'
  }
}));
assert.strictEqual(newCallAfterSuccess.tone, 'call');
assert(newCallAfterSuccess.title.includes('Current Call Section'), 'the current call should use its own target section');
assert(!newCallAfterSuccess.title.includes('Old Response Section'), 'recent terminal feedback must not leak a stale section into a new call');

const addsTwo = Object.assign({}, callBoss, {
  pendingActionId: '',
  pendingSpatialResponseType: '',
  lastSpatialResponse: {
    id: 'response-adds',
    status: 'pending',
    type: 'clearAdds',
    label: 'Sentry Catwalk Adds',
    instruction: 'Defeat the full summoned wave.',
    sectionLabel: 'Sentry Catwalk',
    remaining: 2,
    total: 2
  }
});
const addsOne = Object.assign({}, addsTwo, {
  lastSpatialResponse: Object.assign({}, addsTwo.lastSpatialResponse, { remaining: 1 })
});
const addsTwoFeedback = hud.getBossResponseFeedbackMetadata(addsTwo);
const addsOneFeedback = hud.getBossResponseFeedbackMetadata(addsOne);
assert.strictEqual(addsTwoFeedback.title, 'Clear adds - 2 remaining');
assert.strictEqual(addsOneFeedback.title, 'Clear adds - 1 remaining');
assert.notStrictEqual(addsTwoFeedback.announcementKey, addsOneFeedback.announcementKey,
  'meaningful add-clear progress should create one new accessibility milestone');

const wrongSection = Object.assign({}, callBoss, {
  pendingActionId: '',
  pendingSpatialResponseType: '',
  lastSpatialResponse: {
    id: 'response-window',
    status: 'pending',
    type: 'damageWindow',
    sectionLabel: 'Gear Switch Shelf',
    instruction: 'Damage the boss from the called shelf.',
    lastRejectedReason: 'attackerOutsideSection'
  }
});
const rejectionFeedback = hud.getBossResponseFeedbackMetadata(wrongSection);
assert.strictEqual(rejectionFeedback.tone, 'rejected');
assert.strictEqual(rejectionFeedback.label, 'Wrong position');
assert.strictEqual(rejectionFeedback.title, 'Attack from Gear Switch Shelf');
assert(rejectionFeedback.detail.includes('before the counter window closes'),
  'a rejected hit should tell the player how to recover during the same window');

const compromisedVolley = Object.assign({}, callBoss, {
  pendingActionId: '',
  pendingSpatialResponseType: '',
  lastSpatialResponse: {
    id: 'response-volley',
    status: 'pending',
    type: 'dodgeProjectiles',
    remaining: 2,
    total: 3,
    failed: true
  }
});
const volleyFeedback = hud.getBossResponseFeedbackMetadata(compromisedVolley);
assert.strictEqual(volleyFeedback.label, 'Volley compromised');
assert(volleyFeedback.title.includes('2 projectiles remaining'), 'volley feedback should expose unresolved projectiles');

[
  ['hazardHit', 'Hazard hit', 'Leave the marked impact'],
  ['sectionMissed', 'Missed Gear Switch Shelf', 'Rotate to Gear Switch Shelf'],
  ['damageWindowExpired', 'Counter window missed', 'Deal direct damage from Gear Switch Shelf'],
  ['projectileHit', 'Volley hit', 'Avoid every tagged projectile']
].forEach(([failureReason, expectedTitle, expectedGuidance]) => {
  const feedback = hud.getBossResponseFeedbackMetadata(Object.assign({}, callBoss, {
    pendingActionId: '',
    pendingSpatialResponseType: '',
    lastSpatialResponse: {
      id: `failed-${failureReason}`,
      status: 'failed',
      type: 'avoidHazard',
      sectionLabel: 'Gear Switch Shelf',
      failureReason
    }
  }));
  assert.strictEqual(feedback.tone, 'failed');
  assert.strictEqual(feedback.title, expectedTitle);
  assert(feedback.detail.includes(expectedGuidance), `${failureReason} should include actionable recovery copy`);
});

const successFeedback = hud.getBossResponseFeedbackMetadata(Object.assign({}, callBoss, {
  pendingActionId: '',
  pendingSpatialResponseType: '',
  lastSpatialResponse: {
    id: 'response-success',
    status: 'success',
    label: 'Gear Switch Shelf',
    progressAwarded: true
  }
}));
assert.strictEqual(successFeedback.tone, 'success');
assert(successFeedback.detail.includes('Resume pressure'), 'success feedback should clearly return the player to damage');

const engine = createProjectStarfallEngine(null, data);
assert.strictEqual(engine.chooseClass('fighter'), true);
assert.strictEqual(engine.changeMap('gearworksVault'), true);
engine.enemies = [];
const enemyData = data.ENEMIES.find((enemy) => enemy.id === 'quarryColossus');
assert(enemyData, 'Quarry Colossus should exist for response-feedback integration coverage');
const boss = engine.createEnemy(enemyData, engine.runtime.spawnPoints[0]);
boss.isEncounterBoss = true;
boss.bossEncounterId = 'quarryColossus';
boss.hp = 1000000;
boss.maxHp = 1000000;
engine.enemies.push(boss);
const encounter = engine.getBossEncounterForEnemy(boss);
assert(encounter, 'Quarry Colossus encounter should resolve');
engine.beginBossEncounterAction(
  boss,
  encounter,
  { id: 'feedback', name: 'Feedback' },
  'plateExpose',
  engine.getCombatCharacterByTarget('player', 'player')
);
let bossSnapshot = engine.getBossEncounterSnapshot();
assert(bossSnapshot.pendingSpatialInstruction.includes('damage the boss'),
  'the engine snapshot should carry the measurable call instruction');
const pending = boss.bossPendingAction;
engine.resolveBossEncounterAction(boss, encounter, pending);
let responseSnapshot = engine.getBossSpatialResponseSummary();
assert(responseSnapshot && responseSnapshot.status === 'pending',
  'a live response window should remain visible in the HUD summary');
assert(responseSnapshot.remainingSeconds > 0, 'timed response windows should expose their remaining duration');

const responsePlatform = engine.runtime.platforms.find((platform) => platform.id === pending.spatialPlatformId);
const outsidePlatform = engine.runtime.platforms.find((platform) => platform.id !== pending.spatialPlatformId);
assert(responsePlatform && outsidePlatform, 'the test arena should provide distinct outside and response platforms');
assert.strictEqual(engine.placePlayerOnRuntimePlatform(outsidePlatform.index, outsidePlatform.x + 36), true);
const rejectionToasts = [];
engine.setToastHandler((message) => rejectionToasts.push(typeof message === 'string' ? message : message.message));
engine.damageEnemy(boss, 10, 'melee', { attackerKind: 'player' });
engine.damageEnemy(boss, 10, 'melee', { attackerKind: 'player' });
responseSnapshot = engine.getBossSpatialResponseSummary();
assert.strictEqual(responseSnapshot.lastRejectedReason, 'attackerOutsideSection');
assert.strictEqual(rejectionToasts.filter((message) => String(message).includes('Counter rejected')).length, 1,
  'repeated wrong-section hits should be throttled instead of spamming toasts');

const responseSection = engine.runtime.spawnSections.find((section) => section.id === pending.spatialSectionId);
assert(responseSection, 'the response section should resolve');
const responseLeft = Math.max(Number(responsePlatform.x || 0) + 36, Number(responseSection.x || 0) + 36);
const responseRight = Math.min(
  Number(responsePlatform.x || 0) + Number(responsePlatform.w || 0) - 36,
  Number(responseSection.x || 0) + Number(responseSection.w || 0) - 36
);
assert(responseRight >= responseLeft, 'the response section and platform should overlap');
assert.strictEqual(engine.placePlayerOnRuntimePlatform(responsePlatform.index, (responseLeft + responseRight) / 2), true);
engine.damageEnemy(boss, 10, 'melee', { attackerKind: 'player' });
responseSnapshot = engine.getBossSpatialResponseSummary();
assert.strictEqual(responseSnapshot.status, 'success', 'a valid counter should replace rejection feedback with success');

const hudCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/hud.css'), 'utf8');
const responsiveCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/responsive.css'), 'utf8');
const uiCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(hudCss.includes('.project-starfall-boss-response'), 'the semantic HUD should style the response state as a dedicated ribbon');
assert(responsiveCss.includes('.project-starfall-boss-response'), 'the response ribbon should have a narrow-screen layout');
assert(uiCode.includes("getHudBossEncounterHelper('getBossResponseFeedbackMetadata')"),
  'the semantic boss HUD should consume the same response presentation as the canvas HUD');
assert(uiCode.includes('this.updateHudStatusAnnouncement(this.snapshot)'),
  'live canvas boss-state changes should reach the dedicated accessibility status region');

console.log('Project Starfall boss response feedback tests passed.');
