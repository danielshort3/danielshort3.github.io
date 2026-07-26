'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const input = require('../js/games/project-starfall/ui/input.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const SKILL_ID = 'fighter_heavy_strike';
const BUFFER_SECONDS = 0.12;

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function approximatelyEqual(actual, expected, tolerance = 0.000001) {
  return Math.abs(Number(actual) - Number(expected)) <= tolerance;
}

function withMockNow(startMs, callback) {
  const originalNow = Date.now;
  let currentMs = Number(startMs) || 0;
  Date.now = () => currentMs;
  try {
    return callback((deltaMs) => {
      currentMs += Number(deltaMs) || 0;
      return currentMs;
    });
  } finally {
    Date.now = originalNow;
  }
}

function createFighterEngine() {
  const engine = createProjectStarfallEngine(null, data);
  assert.strictEqual(engine.chooseClass('fighter'), true, 'combat buffer fixture should choose Fighter');
  engine.state.skills[SKILL_ID] = 1;
  Object.assign(engine.state.player, {
    mp: 999,
    grounded: true,
    climbing: false,
    attackTimer: 0,
    combatLockUntil: 0,
    movementLockUntil: 0,
    skillCooldowns: {}
  });
  engine.enemies = [];
  return engine;
}

function countSuccessfulCombatActions(engine) {
  const counts = { basic: 0, skill: 0 };
  const originalBasicAttack = engine.basicAttack.bind(engine);
  const originalUseSkill = engine.useSkill.bind(engine);

  engine.basicAttack = function countedBasicAttack(...args) {
    const used = originalBasicAttack(...args);
    if (used) counts.basic += 1;
    return used;
  };
  engine.useSkill = function countedUseSkill(...args) {
    const used = originalUseSkill(...args);
    if (used) counts.skill += 1;
    return used;
  };

  return counts;
}

function getFreshAttackMetadata() {
  return input.getAttackKeyInputMetadata('KeyX', true, {
    heldAttackKeys: new Set(),
    repeat: false
  });
}

function getFreshSkillMetadata() {
  return input.getSkillKeyInputMetadata({ skillId: SKILL_ID }, 'Digit1', true, {
    heldSkillKeys: new Map(),
    repeat: false
  });
}

const freshAttack = getFreshAttackMetadata();
check(freshAttack.shouldBasicAttack &&
  freshAttack.basicAttackOptions &&
  freshAttack.basicAttackOptions.bufferOnBlock === true,
'fresh keyboard attacks should opt into the short engine buffer');
check(freshAttack.basicAttackOptions.contactSynced === true,
  'fresh keyboard attacks should opt into authored contact timing');

const repeatedAttack = input.getAttackKeyInputMetadata('KeyX', true, {
  heldAttackKeys: freshAttack.heldAttackKeys,
  repeat: true
});
check(!repeatedAttack.shouldBasicAttack,
  'keyboard repeat should not enqueue another fresh attack');

const freshSkill = getFreshSkillMetadata();
check(freshSkill.shouldUseSkill &&
  freshSkill.skillOptions &&
  freshSkill.skillOptions.bufferOnBlock === true,
'fresh keyboard skills should opt into the short engine buffer');
check(freshSkill.skillOptions.contactSynced === true,
  'fresh keyboard skills should opt into authored contact timing');

const repeatedSkill = input.getSkillKeyInputMetadata({ skillId: SKILL_ID }, 'Digit1', true, {
  heldSkillKeys: freshSkill.heldSkillKeys,
  repeat: true
});
check(!repeatedSkill.shouldUseSkill,
  'keyboard repeat should not enqueue another fresh skill');

const pointerAttack = input.getDomAttackButtonPointerAction({
  handled: true,
  actionId: 'attack',
  disabled: false,
  target: {}
}, {
  inRoot: true
});
check(pointerAttack.shouldBasicAttack &&
  pointerAttack.basicAttackOptions &&
  pointerAttack.basicAttackOptions.bufferOnBlock === true,
'fresh pointer attacks should opt into the same engine buffer');
check(pointerAttack.basicAttackOptions.contactSynced === true,
  'fresh pointer attacks should opt into authored contact timing');

withMockNow(1_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const metadata = getFreshAttackMetadata();
  const now = Date.now() / 1000;
  player.attackTimer = now + 0.08;

  engine.setInput('attack', metadata.attackInput);
  check(!engine.basicAttack(metadata.basicAttackOptions),
    'a fresh basic attack should queue instead of firing during a near-ready cooldown');
  engine.setInput('attack', false);

  const buffered = engine.combatInputBuffer;
  check(buffered &&
    buffered.kind === 'basic' &&
    approximatelyEqual(buffered.expiresAt - buffered.queuedAt, BUFFER_SECONDS),
  'a blocked fresh basic attack should retain the full 120ms buffer window');

  advanceClock(79);
  check(!engine.updateCombatInputBuffer() &&
    counts.basic === 0 &&
    engine.combatInputBuffer === buffered,
  'a buffered basic attack should remain pending until its cooldown is ready');

  advanceClock(1);
  check(engine.updateCombatInputBuffer() &&
    counts.basic === 1 &&
    engine.combatInputBuffer === null,
  'a buffered basic attack should execute once when its cooldown becomes ready');
  check(!engine.updateCombatInputBuffer() && counts.basic === 1,
    'a consumed basic attack buffer should not execute twice');
});

withMockNow(2_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const metadata = getFreshSkillMetadata();
  const now = Date.now() / 1000;
  player.skillCooldowns[SKILL_ID] = now + 0.07;

  check(!engine.useSkill(SKILL_ID, metadata.skillOptions),
    'a fresh skill should queue instead of firing during a near-ready cooldown');
  check(engine.combatInputBuffer &&
    engine.combatInputBuffer.kind === 'skill' &&
    engine.combatInputBuffer.skillId === SKILL_ID,
  'a cooldown-blocked skill should retain its normalized skill id');

  const mpBefore = player.mp;
  advanceClock(70);
  check(engine.updateCombatInputBuffer() &&
    counts.skill === 1 &&
    player.mp < mpBefore &&
    engine.combatInputBuffer === null,
  'a cooldown-buffered skill should execute once when ready');
  check(!engine.updateCombatInputBuffer() && counts.skill === 1,
    'a consumed skill cooldown buffer should not execute twice');
});

withMockNow(3_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const metadata = getFreshSkillMetadata();
  const now = Date.now() / 1000;
  player.combatLockUntil = now + 0.09;

  check(!engine.useSkill(SKILL_ID, metadata.skillOptions) &&
    engine.combatInputBuffer &&
    engine.combatInputBuffer.kind === 'skill',
  'a fresh skill should queue behind a near-ready combat lock');

  advanceClock(89);
  check(!engine.updateCombatInputBuffer() && counts.skill === 0,
    'a combat-lock-buffered skill should not execute early');
  advanceClock(1);
  check(engine.updateCombatInputBuffer() &&
    counts.skill === 1 &&
    engine.combatInputBuffer === null,
  'a combat-lock-buffered skill should execute once when the lock ends');
});

withMockNow(4_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const now = Date.now() / 1000;
  player.attackTimer = now + 0.25;

  check(!engine.basicAttack(getFreshAttackMetadata().basicAttackOptions) &&
    engine.combatInputBuffer &&
    engine.combatInputBuffer.kind === 'basic',
  'a blocked fresh attack should create an expiring buffer entry');

  advanceClock(121);
  check(!engine.updateCombatInputBuffer() &&
    counts.basic === 0 &&
    engine.combatInputBuffer === null,
  'a combat input should expire after the 120ms window');

  player.attackTimer = 0;
  check(!engine.updateCombatInputBuffer() && counts.basic === 0,
    'an expired input should not execute after its cooldown later becomes ready');
});

withMockNow(5_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const now = Date.now() / 1000;
  player.combatLockUntil = now + 0.08;

  check(!engine.basicAttack(getFreshAttackMetadata().basicAttackOptions),
    'latest-input fixture should initially queue a basic attack');
  check(!engine.useSkill(SKILL_ID, getFreshSkillMetadata().skillOptions) &&
    engine.combatInputBuffer &&
    engine.combatInputBuffer.kind === 'skill',
  'a newer skill press should replace an older buffered basic attack');

  advanceClock(80);
  check(engine.updateCombatInputBuffer() &&
    counts.skill === 1 &&
    counts.basic === 0,
  'latest-input-wins should execute only the replacement skill');
});

withMockNow(6_000_000, (advanceClock) => {
  const engine = createFighterEngine();
  const player = engine.state.player;
  const counts = countSuccessfulCombatActions(engine);
  const now = Date.now() / 1000;
  player.combatLockUntil = now + 0.08;

  check(!engine.useSkill(SKILL_ID, getFreshSkillMetadata().skillOptions),
    'reverse latest-input fixture should initially queue a skill');
  check(!engine.basicAttack(getFreshAttackMetadata().basicAttackOptions) &&
    engine.combatInputBuffer &&
    engine.combatInputBuffer.kind === 'basic',
  'a newer basic press should replace an older buffered skill');

  advanceClock(80);
  check(engine.updateCombatInputBuffer() &&
    counts.basic === 1 &&
    counts.skill === 0,
  'latest-input-wins should execute only the replacement basic attack');
});

withMockNow(7_000_000, (advanceClock) => {
  const attackEngine = createFighterEngine();
  const attackCounts = countSuccessfulCombatActions(attackEngine);
  const attackPlayer = attackEngine.state.player;
  const now = Date.now() / 1000;
  attackPlayer.attackTimer = now + 0.08;
  attackEngine.setInput('attack', true);

  attackEngine.updateHeldAttack();
  check(attackEngine.combatInputBuffer === null && attackCounts.basic === 0,
    'held basic retries should not create fresh-input buffer entries');
  advanceClock(80);
  attackEngine.updateHeldAttack();
  check(attackCounts.basic === 1 && attackEngine.combatInputBuffer === null,
    'held basic input should keep its existing retry behavior without using the buffer');
  attackEngine.setInput('attack', false);

  const skillEngine = createFighterEngine();
  const skillCounts = countSuccessfulCombatActions(skillEngine);
  const skillPlayer = skillEngine.state.player;
  skillPlayer.combatLockUntil = Date.now() / 1000 + 0.08;
  skillEngine.setHeldSkill(SKILL_ID, true);

  skillEngine.updateHeldSkills();
  check(skillEngine.combatInputBuffer === null && skillCounts.skill === 0,
    'held skill retries should not create fresh-input buffer entries');
  advanceClock(120);
  skillEngine.updateHeldSkills();
  check(skillCounts.skill === 1 && skillEngine.combatInputBuffer === null,
    'held skills should keep their existing retry cadence without using the buffer');
});

withMockNow(8_000_000, () => {
  const engine = createFighterEngine();
  engine.setHeldSkill(SKILL_ID, true);
  engine.heldSkillRetryAt[SKILL_ID] = Date.now() / 1000 + 1;
  engine.heldAirborneSkillRetryIds = {};
  engine.heldAirborneSkillRetryIds[SKILL_ID] = true;
  engine.queueCombatInput('skill', SKILL_ID);

  check(engine.heldSkillIds.has(SKILL_ID) && engine.combatInputBuffer,
    'clear-held fixture should start with held and buffered skill state');
  engine.clearHeldSkills();
  check(engine.heldSkillIds.size === 0 &&
    Object.keys(engine.heldSkillRetryAt).length === 0 &&
    Object.keys(engine.heldAirborneSkillRetryIds).length === 0 &&
    engine.combatInputBuffer === null,
  'clearHeldSkills should cancel held retries and any pending combat input');
});

console.log(`Project Starfall combat input buffer checks passed: ${checks}`);
