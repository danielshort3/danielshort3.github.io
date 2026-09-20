'use strict';

const assert = require('assert');
const Data = require('../../js/games/project-starfall/project-starfall-data');
const Visuals = require('../../js/games/project-starfall/engine/visuals');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine');
const engine = createProjectStarfallEngine(null, Data);
const platform = engine.runtime.platforms[0];
Object.assign(engine.state.player, { classId: 'fighter', level: 20, x: 220, y: platform.y - 74, w: 40, h: 74, hp: 1000, mp: 1000, facing: 1, grounded: true });
function enemy(id, x) {
  const result = engine.createEnemy(Data.ENEMIES.find((entry) => entry.id === id), { x, platformIndex: 0, platformId: platform.id });
  Object.assign(result, { x, y: platform.y - result.h, attackCd: 0, telegraph: 0, staggered: 0, pendingAttack: null, vx: 0, vy: 0 });
  return result;
}
const oracle = enemy('icebloomOracle', 400), recipient = enemy('slimelet', 440);
for (const id of ['icebloomOracle', 'glowcapHealer', 'cloudcallAcolyte']) {
  const cast = Data.ENEMY_ANIMATION_ASSETS[id].states.buff;
  assert.strictEqual(engine.getWeightedAnimationFrameIndex(cast, 0.349999), 3, `${id} remains in preparation immediately before its healing event.`);
  assert.strictEqual(engine.getWeightedAnimationFrameIndex(cast, 0.35), 4, `${id} reaches the restorative pose at the HP event.`);
}
recipient.hp = Math.floor(recipient.maxHp / 2);
engine.enemies = [oracle, recipient];
const before = recipient.hp;
assert(engine.healNearby(oracle));
assert.strictEqual(recipient.hp, before, 'Healing cannot precede its visible restorative pulse.');
oracle.telegraph = 0.001;
engine.resolveEnemyPendingAttack(oracle, []);
assert.strictEqual(recipient.hp, before, 'The last preparation millisecond remains harmless.');
oracle.telegraph = 0;
engine.resolveEnemyPendingAttack(oracle, []);
assert(recipient.hp > before);
assert(engine.effects.some((effect) => effect.recipientEnemyUid === recipient.uid && effect.recoveryKind === 'heal' && effect.color === '#62D995'));
assert(!engine.effects.some((effect) => effect.phase === 'prepare' && effect.sourceEnemyUid === oracle.uid));
const healed = recipient.hp;
oracle.attackRecovery = 0; oracle.attackCd = 0;
assert(engine.healNearby(oracle));
oracle.staggered = 0.2;
engine.resolveEnemyPendingAttack(oracle, []);
assert.strictEqual(recipient.hp, healed, 'An interrupted cast grants no healing.');
assert.strictEqual(oracle.pendingAttack, null);
assert(!engine.effects.some((effect) => effect.phase === 'prepare' && effect.sourceEnemyUid === oracle.uid));

const thrower = enemy('banditThrower', 700);
for (const [id, windup, commitment] of [['slimelet', 0.42, 0.2], ['banditThrower', 0.54, 0.2], ['brambleking', 1, 0.3]]) {
  const actor = enemy(id, 500);
  actor.pendingAttack = { windup };
  actor.animationDuration = windup;
  const animation = Data.ENEMY_ANIMATION_ASSETS[id];
  actor.telegraph = commitment + 0.001;
  assert(engine.getAnimationFrame(animation, 'telegraph', actor).frameIndex < animation.states.telegraph.frames - 1, `${id} must finish preparation before its commitment pose.`);
  for (const remaining of [commitment, commitment / 2, 0.001]) {
    actor.telegraph = remaining;
    assert.strictEqual(engine.getAnimationFrame(animation, 'telegraph', actor).frameIndex, animation.states.telegraph.frames - 1, `${id} must hold a visible commitment through activation.`);
  }
}
const target = engine.getCombatCharacterByTarget('player', 'player');
assert(engine.beginEnemyAttackWindup(thrower, target, { kind: 'projectile', projectileType: 'knife' }));
const locked = { ...thrower.pendingAttack.aimPoint };
engine.state.player.x = 940;
thrower.telegraph = 0;
engine.resolveEnemyPendingAttack(thrower, [target]);
assert(locked.x < thrower.x && engine.projectiles.at(-1).vx < 0, 'A committed projectile cannot silently retarget behind the enemy.');

const skill = Data.SKILLS.find((entry) => entry.id === 'fighter_heavy_strike');
{
  const actionEngine = createProjectStarfallEngine(null, Data);
  Object.assign(actionEngine.state.player, { classId: 'fighter', level: 20, x: 220, y: 360, facing: 1, hp: 1000, mp: 1000, grounded: true });
  actionEngine.state.skills.fighter_heavy_strike = 1;
  const victim = actionEngine.createEnemy(Data.ENEMIES.find((entry) => entry.id === 'slimelet'), { x: 300, platformIndex: 0 });
  Object.assign(victim, { x: 300, y: 380, hp: 10000, maxHp: 10000, defense: 0, vx: 0, vy: 0 });
  actionEngine.enemies = [victim];
  assert(actionEngine.useSkill('fighter_heavy_strike'));
  assert.strictEqual(victim.hp, 10000, 'An accepted attack begins preparation without dealing damage.');
  actionEngine.updatePendingSkillActions(0.16);
  assert.strictEqual(victim.hp, 10000, 'Damage remains pending until the visible release pose.');
  actionEngine.updatePendingSkillActions(1 / 6 - 0.16);
  assert(victim.hp < 10000, 'The real Heavy Strike damage path resolves at contact.');
  assert(actionEngine.effects.some((effect) => ['skillImpact', 'playerActionTrail'].includes(effect.type) && !effect.activationDelay), 'Contact damage and its visual feedback become active together.');
  const hpAtContact = victim.hp;
  actionEngine.updatePendingSkillActions(1);
  assert.strictEqual(victim.hp, hpAtContact, 'The real damage path cannot commit twice.');
}
const player = engine.state.player;
player.animationState = 'skill'; player.skillCastSerial = 1;
let resolutions = 0;
const originalResolve = engine.resolvePreparedSkill;
engine.resolvePreparedSkill = () => { resolutions += 1; };
engine.pendingSkillActions = [{ remaining: 1 / 6, player, runtime: engine.runtime, serial: 1, skill, rank: 1, stats: engine.getStats() }];
engine.updatePendingSkillActions(0.15);
assert.strictEqual(resolutions, 0);
engine.updatePendingSkillActions(1 / 60);
assert.strictEqual(resolutions, 1, 'Skill resolution occurs at the authored contact frame.');
engine.updatePendingSkillActions(1);
assert.strictEqual(resolutions, 1, 'Contact commits once.');
engine.pendingSkillActions = [{ remaining: 1 / 6, player, runtime: engine.runtime, serial: 1, skill }];
engine.effects.push({ type: 'skillCast', playerCastSerial: 1 });
player.animationState = 'hit';
engine.updatePendingSkillActions(0.2);
assert.strictEqual(resolutions, 1);
assert(!engine.effects.some((effect) => effect.playerCastSerial === 1));
engine.resolvePreparedSkill = originalResolve;
engine.effects = [];
engine.pushSkillImpactEffect(100, 100, skill);
assert.strictEqual(engine.effects[0].activationDelay, 0, 'Impact FX has no second hidden delay after contact.');

const heal = Visuals.createSemanticRecoveryDrawState({ recoveryKind: 'heal', ownership: 'enemy', ttl: 0.4, duration: 0.5 });
for (const actionId of ['solarFlare', 'lunarMark', 'ordinaryArea']) {
  const warning = Visuals.createBossHazardEffectDrawState({ actionId, shape: 'circle', color: '#ffbe55', r: 90, ttl: 0.2, duration: 1, telegraph: true });
  const active = Visuals.createBossHazardEffectDrawState({ actionId, shape: 'circle', color: '#7bdff2', r: 90, ttl: 0.8, duration: 1 });
  assert.strictEqual(warning.primaryColor, '#F06A60', 'Elemental identity cannot replace the danger category color.');
  assert.strictEqual(active.primaryColor, '#F06A60');
  assert.strictEqual(warning.boundaryWidth, active.boundaryWidth, 'Warning footprint stays exact through activation.');
}
assert.strictEqual(Visuals.createBossHazardEffectDrawState({ actionId: 'eclipseSigils', hazardPolarity: 'safe' }).primaryColor, '#63D7E8');
const resource = Visuals.createSemanticRecoveryDrawState({ recoveryKind: 'resource', ttl: 0.4, duration: 0.5 });
assert.strictEqual(heal.color, '#62D995');
assert.strictEqual(resource.color, '#668FFF');
assert.strictEqual(heal.orbs.length, 0);
assert(resource.orbs.length > 0, 'Resource droplets and healing plus marks remain distinct without hue.');
assert(heal.lines.length < resource.lines.length, 'Enemy recovery uses a segmented outer contour.');
console.log('Starfall visual-language timing, cancellation, locked aim, and shared renderer cues passed.');
