'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  createRiftRuntimeEffects,
  createRiftSnapshot,
  createRiftTierRuntimeEffects
} = require('../js/games/project-starfall/engine/map-mechanics.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const RUNTIME_SCALE_KEYS = [
  'enemyHealthScale',
  'enemyDamageScale',
  'playerDamageScale',
  'playerResourceCostScale',
  'rewardScale'
];

assert.strictEqual(data.MUTATIONS.length, 6, 'the Endless Rift should keep six authored mutation foundations');

const tierTwelveBaseline = createRiftRuntimeEffects({ tier: 12, mutationIds: [] }, [], { data });
data.MUTATIONS.forEach((mutation) => {
  assert(mutation.danger && typeof mutation.danger === 'object', `${mutation.id} should define structured danger`);
  assert(mutation.upside && typeof mutation.upside === 'object', `${mutation.id} should define structured upside`);
  assert(mutation.counterplay && mutation.counterplay.id, `${mutation.id} should define actionable counterplay`);
  assert(mutation.counterplay.summary, `${mutation.id} counterplay should explain the player response`);
  assert(mutation.counterplay.dangerMitigation > 0, `${mutation.id} counterplay should mitigate authored danger`);
  assert(mutation.rewardScale > 1, `${mutation.id} should compensate added pressure with reward scaling`);

  const active = createRiftRuntimeEffects({ tier: 12, mutationIds: [mutation.id] }, [mutation.id], { data });
  const changedKeys = RUNTIME_SCALE_KEYS.filter((key) => active[key] !== tierTwelveBaseline[key]);
  assert(changedKeys.length > 0, `${mutation.id} should change at least one runtime output`);
  assert(active.rewardScale > tierTwelveBaseline.rewardScale, `${mutation.id} should increase runtime rewards`);
  assert.strictEqual(active.mutationEffects.length, 1, `${mutation.id} should expose one applied runtime effect`);
  assert.strictEqual(active.mutationEffects[0].countered, false, `${mutation.id} should begin uncountered`);

  const countered = createRiftRuntimeEffects({ tier: 12, mutationIds: [mutation.id] }, [mutation.id], {
    data,
    counterplayIds: [mutation.counterplay.id]
  });
  assert.strictEqual(countered.mutationEffects[0].countered, true, `${mutation.id} should recognize its counterplay id`);
  assert(countered.enemyHealthScale <= active.enemyHealthScale, `${mutation.id} counterplay should not add enemy health pressure`);
  assert(countered.enemyDamageScale <= active.enemyDamageScale, `${mutation.id} counterplay should not add enemy damage pressure`);
  assert(countered.enemyHealthScale < active.enemyHealthScale || countered.enemyDamageScale < active.enemyDamageScale,
    `${mutation.id} counterplay should reduce at least one danger output`);
});

let previousTierEffects = createRiftTierRuntimeEffects(1);
for (let tier = 2; tier <= 40; tier += 1) {
  const nextTierEffects = createRiftTierRuntimeEffects(tier);
  assert(nextTierEffects.enemyHealthScale > previousTierEffects.enemyHealthScale, `tier ${tier} should increase enemy health pressure`);
  assert(nextTierEffects.enemyDamageScale > previousTierEffects.enemyDamageScale, `tier ${tier} should increase enemy damage pressure`);
  assert(nextTierEffects.rewardScale > previousTierEffects.rewardScale, `tier ${tier} should increase rewards`);
  assert(nextTierEffects.mutationPotencyScale >= previousTierEffects.mutationPotencyScale, `tier ${tier} should not reduce mutation potency`);
  previousTierEffects = nextTierEffects;
}

const allMutationIds = data.MUTATIONS.map((mutation) => mutation.id);
const lowTierEffects = createRiftRuntimeEffects({ tier: 3, mutationIds: allMutationIds }, allMutationIds, { data });
const highTierEffects = createRiftRuntimeEffects({ tier: 18, mutationIds: allMutationIds }, allMutationIds, { data });
assert(highTierEffects.enemyHealthScale > lowTierEffects.enemyHealthScale, 'higher tiers should increase aggregate enemy health');
assert(highTierEffects.enemyDamageScale > lowTierEffects.enemyDamageScale, 'higher tiers should increase aggregate enemy damage');
assert(highTierEffects.rewardScale > lowTierEffects.rewardScale, 'higher tiers should increase aggregate rewards');

const snapshot = createRiftSnapshot({ tier: 12, bestTier: 14, score: 321, mutationIds: ['burning'] }, ['burning'], { data });
assert.deepStrictEqual(snapshot.mutationIds, ['burning'], 'Rift snapshots should preserve the active mutation set');
assert.strictEqual(snapshot.runtimeEffects.tier, 12, 'Rift snapshots should expose the runtime tier');
assert(snapshot.runtimeEffects.enemyDamageScale > 1, 'Rift snapshots should expose applied challenge scaling');
assert(snapshot.runtimeEffects.rewardScale > 1, 'Rift snapshots should expose applied reward scaling');

function createRiftEngine(tier, mutationIds) {
  const engine = createProjectStarfallEngine(null, data);
  if (!engine.state.player.classId) engine.chooseClass('fighter');
  engine.state.player.level = 100;
  engine.state.mapId = 'endlessRift';
  engine.state.rift = Object.assign({}, engine.state.rift, {
    tier,
    bestTier: tier,
    score: 0,
    mutationIds: mutationIds.slice()
  });
  engine.getMapModifierScale = () => 1;
  engine.getMapModifierBonus = () => 0;
  engine.getMapMechanicRewardScale = () => 1;
  return engine;
}

const lowTierEngine = createRiftEngine(3, ['burning']);
const highTierEngine = createRiftEngine(18, ['burning']);
const riftEnemyData = data.ENEMIES.find((enemy) => enemy.id === 'riftAberration');
assert(riftEnemyData, 'the runtime integration test requires the Rift Aberration enemy');

const originalRandom = Math.random;
let lowTierEnemy;
let highTierEnemy;
try {
  Math.random = () => 0.99;
  lowTierEnemy = lowTierEngine.createEnemy(riftEnemyData, { x: 520, platformIndex: 0 });
  highTierEnemy = highTierEngine.createEnemy(riftEnemyData, { x: 520, platformIndex: 0 });
} finally {
  Math.random = originalRandom;
}
assert(highTierEnemy.maxHp > lowTierEnemy.maxHp, 'actual Rift enemy creation should apply monotonic tier health scaling');
assert(highTierEnemy.damage > lowTierEnemy.damage, 'actual Rift enemy creation should apply monotonic tier damage scaling');
assert.strictEqual(highTierEnemy.riftTier, 18, 'spawned Rift enemies should retain the tier that scaled them');
assert.deepStrictEqual(highTierEnemy.riftMutationIds, ['burning'], 'spawned Rift enemies should retain their mutation context');

const guardedEngine = createRiftEngine(12, ['guarded']);
const volatileEngine = createRiftEngine(12, ['volatile']);
const resourceSkill = data.SKILLS.find((skill) => Number(skill.resourceCost || 0) >= 10);
assert(resourceSkill, 'the runtime integration test requires a resource-spending skill');
const resourceStats = { resourceCostReductionPercent: 0 };
const guardedCost = guardedEngine.getRuntimeSkillResourceCost(resourceSkill, 1, null, resourceStats);
const volatileCost = volatileEngine.getRuntimeSkillResourceCost(resourceSkill, 1, null, resourceStats);
assert(guardedCost < volatileCost, 'Rift upside and danger should alter actual skill resource costs');

let baselineDamage;
let burningDamage;
try {
  Math.random = () => 0.5;
  const nonRiftEngine = createProjectStarfallEngine(null, data);
  if (!nonRiftEngine.state.player.classId) nonRiftEngine.chooseClass('fighter');
  baselineDamage = nonRiftEngine.rollDamage(100, null);
  burningDamage = lowTierEngine.rollDamage(100, null);
} finally {
  Math.random = originalRandom;
}
assert(burningDamage > baselineDamage, 'Rift upside should alter actual outgoing player damage');

const lowRewardStart = { xp: lowTierEngine.state.player.xp, currency: lowTierEngine.state.player.currency };
const highRewardStart = { xp: highTierEngine.state.player.xp, currency: highTierEngine.state.player.currency };
try {
  Math.random = () => 0.99;
  lowTierEngine.defeatEnemy(lowTierEnemy);
  highTierEngine.defeatEnemy(highTierEnemy);
} finally {
  Math.random = originalRandom;
}
const lowXpReward = lowTierEngine.state.player.xp - lowRewardStart.xp;
const highXpReward = highTierEngine.state.player.xp - highRewardStart.xp;
const lowCurrencyReward = lowTierEngine.state.player.currency - lowRewardStart.currency;
const highCurrencyReward = highTierEngine.state.player.currency - highRewardStart.currency;
assert(highXpReward > lowXpReward, 'actual Rift XP rewards should increase with tier pressure');
assert(highCurrencyReward > lowCurrencyReward, 'actual Rift currency rewards should increase with tier pressure');
assert(highTierEngine.state.rift.score > lowTierEngine.state.rift.score, 'actual Rift score rewards should increase with tier pressure');

const engineSnapshot = highTierEngine.getRiftSnapshot();
assert(engineSnapshot.runtimeEffects, 'engine Rift snapshots should include runtime-effect evidence');
assert.strictEqual(engineSnapshot.runtimeEffects.tier, 18, 'engine Rift snapshots should report the active runtime tier');

console.log('Project Starfall Rift mutation runtime tests passed.');
