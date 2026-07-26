'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

function withMockClock(startMs, callback) {
  const originalDateNow = Date.now;
  const originalPerformance = Object.getOwnPropertyDescriptor(globalThis, 'performance');
  let currentMs = Number(startMs) || 0;
  Date.now = () => currentMs;
  Object.defineProperty(globalThis, 'performance', {
    configurable: true,
    writable: true,
    value: {
      now: () => currentMs
    }
  });
  try {
    return callback({
      advanceMs(deltaMs) {
        currentMs += Number(deltaMs) || 0;
        return currentMs;
      },
      nowMs() {
        return currentMs;
      }
    });
  } finally {
    Date.now = originalDateNow;
    if (originalPerformance) {
      Object.defineProperty(globalThis, 'performance', originalPerformance);
    } else {
      delete globalThis.performance;
    }
  }
}

withMockClock(1_000_000, (clock) => {
  const animationModule = require('../js/games/project-starfall/data/animations.js');
  const equipmentVisuals = require('../js/games/project-starfall/data/equipment-visuals.js');
  const data = require('../js/games/project-starfall/project-starfall-data.js');
  const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

  let checks = 0;

  function check(condition, message) {
    assert(condition, message);
    checks += 1;
  }

  function approximatelyEqual(actual, expected, tolerance = 0.000001) {
    return Math.abs(Number(actual) - Number(expected)) <= tolerance;
  }

  function getContactDelay(frameDefinition) {
    return Number(frameDefinition.contactFrame) / Number(frameDefinition.fps);
  }

  const animationData = animationModule.createAnimationData();
  const basicAnimation = animationData.PLAYER_ANIMATION_CONFIG.basic;
  const skillAnimation = animationData.PLAYER_ANIMATION_CONFIG.skill;
  check(basicAnimation.contactFrame === 2 &&
    basicAnimation.contactFrame >= 0 &&
    basicAnimation.contactFrame < basicAnimation.frames,
  'basic attacks should declare a valid authored contact frame');
  check(skillAnimation.contactFrame === 2 &&
    skillAnimation.contactFrame >= 0 &&
    skillAnimation.contactFrame < skillAnimation.frames,
  'skills should declare a valid authored contact frame');
  check(approximatelyEqual(getContactDelay(basicAnimation), 0.125),
    'the six-frame basic animation should contact at 125ms');
  check(approximatelyEqual(getContactDelay(skillAnimation), 1 / 6),
    'the six-frame skill animation should contact at one sixth of a second');
  check(equipmentVisuals.FIGHTER_RIG_ANIMATION_STATES.basic.contactFrame === 2 &&
    equipmentVisuals.FIGHTER_RIG_ANIMATION_STATES.skill.contactFrame === 2,
  'the layered Fighter rig should preserve the same authored contact event');
  ['fighter', 'mage', 'archer'].forEach((classId) => {
    const animation = data.BASE_CLASSES[classId].animation;
    check(animation.states.basic.contactFrame === basicAnimation.contactFrame &&
      animation.states.skill.contactFrame === skillAnimation.contactFrame,
    `${classId} runtime animation data should carry both contact frames`);
  });

  const uiSource = fs.readFileSync(path.join(
    __dirname,
    '../js/games/project-starfall/project-starfall-ui.js'
  ), 'utf8');
  [
    'basicAttack({ silent: true, fromHeldInput: true, bufferOnBlock: true, contactSynced: true })',
    'useSkill(action.skillId, { bufferOnBlock: true, contactSynced: true })',
    'useSkill(skillActivationAction.skillId, { bufferOnBlock: true, contactSynced: true })',
    'basicAttack({ bufferOnBlock: true, contactSynced: true })'
  ].forEach((snippet) => {
    check(uiSource.includes(snippet), `live UI input should retain contact synchronization: ${snippet}`);
  });

  function createFixture(classId, options) {
    const settings = options || {};
    const engine = createProjectStarfallEngine(null, data);
    assert.strictEqual(engine.chooseClass(classId), true, `${classId} fixture should choose its class`);
    const player = engine.state.player;
    Object.assign(player, {
      x: 100,
      y: 300,
      previousY: 300,
      w: 40,
      h: 74,
      facing: 1,
      vx: 0,
      vy: 0,
      hp: 999,
      maxHp: 999,
      mp: 999,
      maxMp: 999,
      shield: 0,
      grounded: true,
      climbing: false,
      attackTimer: 0,
      skillTimer: 0,
      combatLockUntil: 0,
      movementLockUntil: 0,
      animationLockUntil: 0,
      skillCooldowns: {}
    });
    engine.effects = [];
    engine.projectiles = [];
    engine.chainPulses = [];
    engine.enemies = [];
    engine.updatePassiveRegen = () => {};
    engine.updateEnemies = () => {};
    engine.updateProjectiles = () => {};
    if (settings.withEnemy) {
      const enemyData = data.ENEMIES.find((enemy) => enemy.id === 'slimelet');
      const enemy = engine.createEnemy(enemyData, {
        x: 150,
        platformIndex: 0
      });
      Object.assign(enemy, {
        x: 150,
        y: 312,
        lastX: 150,
        hp: 100000,
        maxHp: 100000,
        defense: 0,
        vx: 0,
        vy: 0,
        attackCd: 999,
        telegraph: 0,
        pendingAttack: null
      });
      engine.enemies = [enemy];
      return { engine, player, enemy };
    }
    return { engine, player, enemy: null };
  }

  function advance(engine, seconds) {
    const delta = Number(seconds) || 0;
    clock.advanceMs(delta * 1000);
    engine.update(delta);
  }

  const basicDelay = getContactDelay(basicAnimation);
  const skillDelay = getContactDelay(skillAnimation);

  {
    const { engine, player, enemy } = createFixture('fighter', { withEnemy: true });
    const hpBefore = enemy.hp;
    check(engine.basicAttack({ contactSynced: true }),
      'a synchronized Fighter basic attack should validate at input time');
    check(player.animationState === 'basic' && enemy.hp === hpBefore,
      'a Fighter basic attack should animate immediately without dealing early damage');
    advance(engine, basicDelay - 0.001);
    check(enemy.hp === hpBefore,
      'a Fighter basic attack should not damage before its authored contact frame');
    advance(engine, 0.002);
    check(enemy.hp < hpBefore,
      'a Fighter basic attack should damage at its authored contact frame');
    const hpAfterContact = enemy.hp;
    advance(engine, 0.05);
    check(enemy.hp === hpAfterContact,
      'a Fighter basic attack should resolve its contact only once');
  }

  ['mage', 'archer'].forEach((classId) => {
    const { engine, player } = createFixture(classId);
    check(engine.basicAttack({ contactSynced: true }),
      `a synchronized ${classId} basic attack should validate at input time`);
    check(player.animationState === 'basic' && engine.projectiles.length === 0,
      `a ${classId} basic attack should animate without spawning an early projectile`);
    advance(engine, basicDelay - 0.001);
    check(engine.projectiles.length === 0,
      `a ${classId} basic projectile should wait for the authored contact frame`);
    advance(engine, 0.002);
    check(engine.projectiles.length === 1,
      `a ${classId} basic projectile should spawn at the authored contact frame`);
    advance(engine, 0.05);
    check(engine.projectiles.length === 1,
      `a ${classId} basic projectile should spawn only once`);
  });

  {
    const { engine, player, enemy } = createFixture('fighter', { withEnemy: true });
    engine.state.skills.fighter_heavy_strike = 1;
    const hpBefore = enemy.hp;
    const mpBefore = player.mp;
    check(engine.useSkill('fighter_heavy_strike', { contactSynced: true }),
      'Heavy Strike should validate at input time');
    check(player.mp < mpBefore &&
      player.animationState === 'skill' &&
      enemy.hp === hpBefore,
    'Heavy Strike should spend MP and animate immediately without dealing early damage');
    advance(engine, skillDelay - 0.001);
    check(enemy.hp === hpBefore &&
      !engine.effects.some((effect) => effect && effect.skillId === 'fighter_heavy_strike'),
    'Heavy Strike should not publish damage or skill FX before the skill contact frame');
    advance(engine, 0.002);
    check(enemy.hp < hpBefore,
      'Heavy Strike should damage at the skill contact frame');
    check(engine.effects.some((effect) =>
      effect &&
      effect.skillId === 'fighter_heavy_strike' &&
      Number(effect.activationDelay || 0) === 0
    ), 'Heavy Strike FX should begin on the resolved contact without a second delay');
    const hpAfterContact = enemy.hp;
    advance(engine, 0.05);
    check(enemy.hp === hpAfterContact,
      'Heavy Strike should resolve its contact only once');
  }

  [
    { classId: 'mage', skillId: 'mage_magic_bolt', projectileCount: 1 },
    { classId: 'archer', skillId: 'archer_quick_shot', projectileCount: 2 }
  ].forEach(({ classId, skillId, projectileCount }) => {
    const { engine, player } = createFixture(classId);
    engine.state.skills[skillId] = 1;
    const mpBefore = player.mp;
    check(engine.useSkill(skillId, { contactSynced: true }),
      `${skillId} should validate at input time`);
    check(player.mp < mpBefore &&
      player.animationState === 'skill' &&
      engine.projectiles.length === 0,
    `${skillId} should spend MP and animate without spawning early projectiles`);
    advance(engine, skillDelay - 0.001);
    check(engine.projectiles.length === 0,
      `${skillId} projectiles should wait for the skill contact frame`);
    advance(engine, 0.002);
    check(engine.projectiles.length === projectileCount,
      `${skillId} should spawn its authored projectile count at contact`);
    advance(engine, 0.05);
    check(engine.projectiles.length === projectileCount,
      `${skillId} should resolve its projectile contact only once`);
  });

  {
    const { engine, player } = createFixture('fighter');
    engine.state.skills.fighter_heavy_strike = 3;
    engine.state.skills.fighter_guard = 1;
    check(engine.useSkill('fighter_guard', { contactSynced: true }),
      'Guard should remain usable through a contact-synchronized input path');
    check(player.shield > 0,
      'non-offensive defensive skills should still resolve immediately');
  }

  {
    const { engine, player } = createFixture('archer');
    engine.state.skills.archer_quick_shot = 3;
    engine.state.skills.archer_marked_shot = 3;
    engine.state.skills.archer_eagle_stance = 1;
    check(engine.useSkill('archer_eagle_stance', { contactSynced: true }),
      'Eagle Stance should remain usable through a contact-synchronized input path');
    check(player.buffs.eagleEye > clock.nowMs() / 1000,
      'non-offensive buff skills should still resolve immediately');
  }

  {
    const { engine, player, enemy } = createFixture('fighter', { withEnemy: true });
    const hpBefore = enemy.hp;
    check(engine.basicAttack({ contactSynced: true }),
      'the cancellation fixture should queue a synchronized basic attack');
    player.hp = 0;
    advance(engine, basicDelay + 0.01);
    check(enemy.hp === hpBefore,
      'defeat before contact should cancel pending offensive damage');
    player.hp = player.maxHp;
    advance(engine, basicDelay + 0.01);
    check(enemy.hp === hpBefore,
      'a canceled player action should not resolve later after revival');
  }

  {
    const { engine, enemy } = createFixture('fighter', { withEnemy: true });
    const hpBefore = enemy.hp;
    check(engine.basicAttack({ contactSynced: true }),
      'the dodge fixture should queue a synchronized Fighter basic attack');
    enemy.x += 500;
    advance(engine, basicDelay + 0.01);
    check(enemy.hp === hpBefore,
      'a target that leaves the captured melee lane before contact should make the attack whiff');
  }

  {
    const { engine } = createFixture('mage');
    check(engine.basicAttack({ contactSynced: true }),
      'the travel fixture should queue a synchronized Mage basic attack');
    check(engine.changeMap('greenrootMeadow', { silent: true }),
      'the travel fixture should change maps before contact');
    advance(engine, basicDelay + 0.01);
    check(!engine.pendingPlayerAction && engine.projectiles.length === 0,
      'map travel should cancel pending player contact and prevent a late projectile');
  }

  console.log(`Project Starfall player contact timing tests passed: ${checks} checks.`);
});
