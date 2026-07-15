'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const visuals = require('../js/games/project-starfall/engine/visuals.js');
const movement = require('../js/games/project-starfall/engine/movement.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

function getSkill(skillId) {
  return data.SKILLS.find((skill) => skill.id === skillId);
}

function getSkillSheet(skillId) {
  return data.SKILL_FX_ANIMATION_ASSETS[skillId].sheet;
}

function createEngine() {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, {
    classId: 'fighter',
    advancedClassId: 'guardian',
    x: 140,
    y: 320,
    w: 40,
    h: 74,
    facing: 1,
    grounded: true,
    vx: 0,
    vy: 0
  });
  engine.runtime.worldWidth = Math.max(1280, Number(engine.runtime.worldWidth || 0));
  return engine;
}

const groundSlamAnimation = data.SKILL_FX_ANIMATION_ASSETS.fighter_ground_slam;
assert.strictEqual(groundSlamAnimation.states.cast.loop, false, 'skill casts should be one-shot animations');
assert.strictEqual(groundSlamAnimation.states.impact.loop, false, 'skill impacts should be one-shot animations');
assert.strictEqual(groundSlamAnimation.states.projectile.loop, true, 'in-flight projectile effects should keep looping');
assert(groundSlamAnimation.states.projectile.loopDelay > 0, 'looping projectile effects should keep a loop delay');

const duration = 0.3;
const impactFrames = Array.from({ length: 6 }, (_, frameIndex) => {
  const progress = (frameIndex + 0.5) / 6;
  const ttl = duration * (1 - progress);
  return visuals.createTimedCombatFxAnimationFrame(groundSlamAnimation, 'impact', ttl, duration).frameIndex;
});
assert.deepStrictEqual(impactFrames, [0, 1, 2, 3, 4, 5], 'one-shot timing should visit every authored frame exactly once');

const finalFrameStart = visuals.createEffectCombatFxDrawState({
  type: 'skillImpact',
  ttl: duration / 6,
  duration
}, 'impact', { frameDef: groundSlamAnimation.states.impact });
assert.strictEqual(finalFrameStart.oneShot, true, 'impact draw state should identify one-shot playback');
assert(finalFrameStart.alpha >= 0.99, 'the final one-shot frame should begin fully visible before its fade');
assert.strictEqual(visuals.getEffectCombatFxState({ type: 'partyBuff' }), 'cast', 'party buffs should resolve the cast row in every renderer');

const blinkSkill = getSkill('mage_blink');
const blinkPlayer = { x: 80, y: 260, w: 40, h: 74, facing: 1, grounded: true, vx: 0, vy: 0 };
const blinkApplication = movement.getSkillMovementApplicationPlan(blinkSkill, 1, blinkPlayer);
const blinkStart = movement.getSkillMovementBlinkStartEffectPlan(blinkApplication);
const blinkAction = movement.getSkillMovementActionEffectPlan(blinkSkill, blinkApplication);
assert.strictEqual(blinkStart.skillId, blinkSkill.id, 'blink start effects should retain their source skill sheet id');
assert.strictEqual(blinkStart.combatFxState, 'cast', 'blink start effects should use the cast row');
assert.strictEqual(blinkAction.options.skillId, blinkSkill.id, 'mobility action effects should retain their source skill sheet id');
assert.strictEqual(blinkAction.options.combatFxState, 'cast', 'blink action effects should use the cast row');

const engine = createEngine();
engine.getActivePrototypePartyMembers = () => [{ classId: 'stormMage' }];
const selectedPaths = [];
engine.collectCurrentSkillFxAssetPaths(selectedPaths);
assert(selectedPaths.includes(getSkillSheet('fighter_ground_slam')), 'current base-class skill FX should be selected for preload');
assert(selectedPaths.includes(getSkillSheet('guardian_shield_wall')), 'current advanced-class party FX should be selected for preload');
assert(selectedPaths.includes(getSkillSheet('storm_mage_stormfront')), 'active party-member advanced skill FX should be selected for preload');
assert(selectedPaths.includes(getSkillSheet('mage_magic_bolt')), 'active party-member base skill FX should be selected for preload');
assert(!selectedPaths.includes(getSkillSheet('archer_quick_shot')), 'unrelated class skill FX should stay out of the preload set');

const mapPaths = engine.getMapAssetSet(engine.state.mapId);
const pixiPaths = engine.getCurrentPixiAssetSet();
[mapPaths, pixiPaths].forEach((paths) => {
  assert(paths.includes(getSkillSheet('fighter_ground_slam')), 'Canvas and Pixi asset sets should include current class skill FX');
  assert(paths.includes(getSkillSheet('storm_mage_stormfront')), 'Canvas and Pixi asset sets should include active party skill FX');
  assert(!paths.includes(getSkillSheet('archer_quick_shot')), 'Canvas and Pixi asset sets should exclude unrelated class skill FX');
});

engine.effects = [];
const groundSlamSkill = getSkill('fighter_ground_slam');
engine.pushSkillImpactEffect(220, 330, groundSlamSkill, { ttl: duration, duration });
const delayedImpact = engine.effects[0];
assert(delayedImpact.activationDelay > 0, 'contact FX should wait for the player/equipment contact frame');
const initialTtl = delayedImpact.ttl;
engine.updateEffects(delayedImpact.activationDelay / 2);
assert.strictEqual(delayedImpact.ttl, initialTtl, 'contact delay should not consume the visible FX lifetime');
assert.strictEqual(engine.getWorldEffectRenderSnapshot(delayedImpact), null, 'pending contact FX should not render early');
engine.updateEffects(delayedImpact.activationDelay + 0.01);
assert(delayedImpact.ttl < initialTtl, 'contact FX should start advancing after the contact delay');
const impactSnapshot = engine.getWorldEffectRenderSnapshot(delayedImpact);
assert(impactSnapshot.animationFrame && impactSnapshot.combatFxDrawState, 'renderer snapshots should share frame and draw-state data');
assert.strictEqual(impactSnapshot.combatFxDrawState.size, visuals.createEffectCombatFxDrawState(
  delayedImpact,
  'impact',
  { frameDef: groundSlamAnimation.states.impact }
).size, 'Canvas and Pixi should consume the same combat FX sizing state');

engine.effects = [];
engine.pushBuffCastEffect(getSkill('guardian_shield_wall'), 'shieldWall', { ttl: 0.6, duration: 0.6 });
const buffEffect = engine.effects[0];
const buffSnapshot = engine.getWorldEffectRenderSnapshot(buffEffect);
assert.strictEqual(buffEffect.combatFxState, 'cast', 'party buff effects should explicitly carry the cast state');
assert.strictEqual(buffSnapshot.animationFrame.row, groundSlamAnimation.states.cast.row, 'party buffs should select the cast sheet row');

console.log('Project Starfall skill FX runtime tests passed.');
