'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const feedback = require('../js/games/project-starfall/engine/combat-feedback.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const normalProfile = feedback.getBasicHitFeedbackProfile({ reducedEffects: false });
assert.strictEqual(normalProfile.hitstopMs, 45, 'a standard basic hit should hold its contact frame for 45ms');
assert(normalProfile.impulseStrengthPx > 0 && normalProfile.impulseStrengthPx <= 4,
  'the normal camera impulse should remain restrained');

const reducedProfile = feedback.getBasicHitFeedbackProfile({ reducedEffects: true });
assert(reducedProfile.hitstopMs < normalProfile.hitstopMs,
  'reduced effects should substantially shorten hitstop');
assert.strictEqual(reducedProfile.impulseStrengthPx, 0,
  'reduced effects should remove camera movement');

let state = feedback.triggerBasicHitFeedback(feedback.createCombatFeedbackState(), { direction: 1 });
state = feedback.triggerBasicHitFeedback(state, { direction: 1 });
assert.strictEqual(state.hitstopRemainingMs, 45,
  'multi-target contact in one swing should coalesce rather than stack hitstop');

const firstFrame = feedback.advanceCombatFeedback(state, 16);
assert.strictEqual(firstFrame.simulationScale, 0, 'simulation should hold while hitstop covers the full frame');
const secondFrame = feedback.advanceCombatFeedback(firstFrame.state, 16);
const contactReleaseFrame = feedback.advanceCombatFeedback(secondFrame.state, 16);
assert(contactReleaseFrame.simulationScale > 0 && contactReleaseFrame.simulationScale < 1,
  'the release frame should preserve the fraction of time left after the 45ms hold');
assert.strictEqual(contactReleaseFrame.state.hitstopRemainingMs, 0,
  'hitstop should be fully consumed after the release frame');

const initialImpulse = feedback.getCameraImpulseOffset(state);
assert(Math.abs(initialImpulse.x) <= feedback.CAMERA_IMPULSE_STRENGTH_PX,
  'camera impulse should never exceed its authored pixel cap');
const settledImpulse = feedback.getCameraImpulseOffset(
  feedback.advanceCombatFeedback(state, feedback.BASIC_HITSTOP_MS + feedback.CAMERA_IMPULSE_DURATION_MS + 1).state
);
assert.deepStrictEqual(settledImpulse, { x: 0, y: 0 }, 'camera impulse should settle completely');

const reaction = feedback.createEnemyHitReaction({ startedAtMs: 100, direction: 1 });
const contactReaction = feedback.getEnemyHitReactionState(reaction, 100);
assert(contactReaction.translateX < 0, 'an enemy should recoil away from a right-facing hit');
assert(contactReaction.scaleY < 1 && contactReaction.scaleX > 1,
  'contact should briefly squash the enemy while preserving its foot baseline');
const transformedBox = feedback.applyEnemyHitReactionToBox({ x: 20, y: 40, w: 80, h: 80 }, contactReaction);
assert.strictEqual(Math.round(transformedBox.y + transformedBox.h), 120,
  'squash should keep the enemy planted on its original foot baseline');

const reducedReaction = feedback.getEnemyHitReactionState(
  feedback.createEnemyHitReaction({ startedAtMs: 100, direction: 1, reducedEffects: true }),
  100
);
assert.strictEqual(reducedReaction.translateX, 0, 'reduced effects should remove enemy recoil');
assert.strictEqual(reducedReaction.scaleX, 1, 'reduced effects should remove enemy horizontal squash');
assert.strictEqual(reducedReaction.scaleY, 1, 'reduced effects should remove enemy vertical squash');
assert(reducedReaction.flashAlpha > 0 && reducedReaction.flashAlpha < contactReaction.flashAlpha,
  'reduced effects should retain a quieter non-motion hit confirmation');

function createBasicAttackEngine(reducedEffects) {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, {
    classId: 'fighter',
    advancedClassId: '',
    x: 140,
    y: 360,
    w: 40,
    h: 74,
    facing: 1,
    grounded: true,
    vx: 0,
    vy: 0,
    attackTimer: 0,
    combatLockUntil: 0
  });
  engine.userSettings.accessibility.reducedEffects = !!reducedEffects;
  const enemyData = data.ENEMIES[0];
  const enemy = {
    id: enemyData.id,
    name: enemyData.name,
    data: enemyData,
    x: 176,
    y: 372,
    w: 46,
    h: 46,
    hp: 100,
    maxHp: 100,
    defense: 0,
    level: 1,
    vx: 0,
    vy: 0,
    attackCd: 0,
    telegraph: 0,
    marked: 0,
    burning: 0,
    slowed: 0,
    staggered: 0,
    weakPoint: 0,
    packMarked: 0,
    runeLinked: 0,
    affixIds: [],
    facing: -1
  };
  engine.enemies = [enemy];
  return { engine, enemy };
}

const normalCombat = createBasicAttackEngine(false);
const originalRandom = Math.random;
try {
  Math.random = () => 0.99;
  assert.strictEqual(normalCombat.engine.basicAttack(), true, 'the focused melee fixture should execute a basic attack');
  assert(normalCombat.engine.pendingPlayerAttack, 'the focused melee fixture should stage its contact frame');
  assert.strictEqual(normalCombat.engine.advancePendingBasicAttack(normalCombat.engine.pendingPlayerAttack.releaseAt), true,
    'the focused melee fixture should release at its authored contact frame');
} finally {
  Math.random = originalRandom;
}
assert(normalCombat.enemy.hp < normalCombat.enemy.maxHp, 'the basic attack should make contact');
assert.strictEqual(normalCombat.engine.combatFeedback.hitstopRemainingMs, 45,
  'engine basic-hit contact should trigger the authored hitstop');
assert(normalCombat.enemy.basicHitReaction, 'engine basic-hit contact should register an enemy reaction');
const reactionStart = normalCombat.enemy.basicHitReaction.startedAtMs;
const animationStart = normalCombat.enemy.animationStartedAt;
const heldFrame = normalCombat.engine.advanceCombatFeedback(16);
assert.strictEqual(heldFrame.simulationScale, 0, 'the engine should hold simulation on the contact frame');
assert.strictEqual(normalCombat.enemy.basicHitReaction.startedAtMs, reactionStart + 16,
  'enemy reaction timing should stay pinned to contact while simulation is held');
assert.strictEqual(normalCombat.enemy.animationStartedAt, animationStart + 0.016,
  'the authored enemy hit frame should stay pinned during hitstop');

const reducedCombat = createBasicAttackEngine(true);
try {
  Math.random = () => 0.99;
  assert.strictEqual(reducedCombat.engine.basicAttack(), true, 'reduced-effects combat should still execute normally');
  assert(reducedCombat.engine.pendingPlayerAttack, 'reduced-effects combat should stage its contact frame');
  assert.strictEqual(reducedCombat.engine.advancePendingBasicAttack(reducedCombat.engine.pendingPlayerAttack.releaseAt), true,
    'reduced-effects combat should release at its authored contact frame');
} finally {
  Math.random = originalRandom;
}
assert.strictEqual(reducedCombat.engine.combatFeedback.hitstopRemainingMs, feedback.REDUCED_BASIC_HITSTOP_MS,
  'the engine should honor the reduced-effects hitstop profile');
assert.deepStrictEqual(reducedCombat.engine.getCombatFeedbackCameraOffset(), { x: 0, y: 0 },
  'the engine should expose no camera motion in reduced-effects mode');

console.log('Project Starfall combat feedback tests passed.');
