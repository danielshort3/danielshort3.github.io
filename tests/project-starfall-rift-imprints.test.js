'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const modifierSystem = require('../js/games/project-starfall/engine/skill-modifiers.js');
const skillUi = require('../js/games/project-starfall/ui/skill-state.js');
const { RIFT_FIRST_CLEAR_REWARD } = require('../js/games/project-starfall/engine/map-mechanics.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { createRetentionHealthReport } = require('./project-starfall-balance-harness.js');

const advancedClassIds = Object.keys(data.ADVANCED_CLASSES || {}).sort();
assert.strictEqual(advancedClassIds.length, 9, 'the Rift Imprint matrix should cover all nine advanced classes');

const choicesByClass = new Map();
advancedClassIds.forEach((classId) => {
  const signatureSkill = (data.SKILLS || []).find((skill) => skill && skill.owner === classId && skill.primaryTraining);
  assert(signatureSkill, `${classId} should have one primary-training signature skill`);
  const modifiers = (data.SKILL_MODIFIERS || []).filter((modifier) => modifier && modifier.skillId === signatureSkill.id);
  const baseline = modifiers.filter((modifier) => modifier.unlockSource !== 'rift');
  const rift = modifiers.filter((modifier) => modifier.unlockSource === 'rift');
  assert.strictEqual(baseline.length, 1, `${classId} should have one automatic signature baseline`);
  assert.strictEqual(rift.length, 1, `${classId} should have one paid horizontal Rift alternative`);
  assert.deepStrictEqual(rift[0].unlockCost, { materialId: 'riftSplinter', amount: 24 }, `${classId} should use the authored Rift Splinter price`);
  assert.strictEqual(rift[0].unlockLevel, 100, `${classId} should unlock Rift shaping at level 100`);
  choicesByClass.set(classId, { signatureSkill, baseline: baseline[0], rift: rift[0] });
});

const pristineModifierState = modifierSystem.createSkillModifierState(null, { data });
const automaticIds = modifierSystem.createUnlockedSkillModifierIds(pristineModifierState, {
  level: 100,
  advancedClassId: 'guardian'
}, {
  data,
  getSkillRank: () => 10
});
assert.strictEqual((data.SKILL_MODIFIERS || []).filter((modifier) => modifier.unlockSource === 'rift' && automaticIds.includes(modifier.id)).length, 0,
  'paid Rift Imprints should never auto-unlock from level or skill rank');
assert(automaticIds.includes(choicesByClass.get('berserker').baseline.id), 'Berserker should receive a sensible automatic baseline');
assert(automaticIds.includes(choicesByClass.get('duelist').baseline.id), 'Duelist should receive a sensible automatic baseline');

function createAdvancedEngine(classId) {
  const advanced = data.ADVANCED_CLASSES[classId];
  const choice = choicesByClass.get(classId);
  const engine = createProjectStarfallEngine(null, data);
  if (!engine.state.player.classId) engine.chooseClass(advanced.baseClass);
  engine.state.player.level = 100;
  engine.state.player.advancedClassId = classId;
  engine.state.skills[choice.signatureSkill.id] = 1;
  engine.state.mapId = 'starfallCrossing';
  engine.state.rift.active = false;
  return engine;
}

const guardianChoice = choicesByClass.get('guardian');
const purchaseEngine = createAdvancedEngine('guardian');
purchaseEngine.state.materials.riftSplinter = 23;
assert.strictEqual(purchaseEngine.unlockSkillModifier(guardianChoice.rift.id), false, 'an unaffordable Imprint purchase should fail');
assert.strictEqual(purchaseEngine.state.materials.riftSplinter, 23, 'a failed purchase should not spend any material');
assert(!purchaseEngine.state.skillModifiers.unlockedModifierIds.includes(guardianChoice.rift.id), 'a failed purchase should not unlock the Imprint');

purchaseEngine.state.materials.riftSplinter = 24;
assert.strictEqual(purchaseEngine.unlockSkillModifier(guardianChoice.rift.id), true, 'an eligible player should be able to shape a Rift Imprint');
assert.strictEqual(purchaseEngine.state.materials.riftSplinter, 0, 'a successful purchase should spend exactly 24 Rift Splinters once');
assert(purchaseEngine.state.skillModifiers.unlockedModifierIds.includes(guardianChoice.rift.id), 'a successful purchase should persist the paid unlock');
assert.strictEqual(purchaseEngine.state.skillModifiers.activeBySkillId[guardianChoice.signatureSkill.id], guardianChoice.rift.id,
  'a newly shaped Rift Imprint should equip atomically');
assert.strictEqual(purchaseEngine.unlockSkillModifier(guardianChoice.rift.id), false, 'a duplicate purchase should be rejected');
assert.strictEqual(purchaseEngine.state.materials.riftSplinter, 0, 'a duplicate purchase should never double-spend');

assert.strictEqual(purchaseEngine.selectSkillModifier(guardianChoice.baseline.id), true, 'unlocked Imprints should switch freely in a safe zone');
assert.strictEqual(purchaseEngine.state.materials.riftSplinter, 0, 'free switching should not spend Rift Splinters');
assert.strictEqual(purchaseEngine.state.skillModifiers.activeBySkillId[guardianChoice.signatureSkill.id], guardianChoice.baseline.id,
  'the selected baseline should become active');
assert.strictEqual(purchaseEngine.selectSkillModifier(guardianChoice.rift.id), true, 'the paid path should remain freely selectable after purchase');

const fieldMap = (data.MAPS || []).find((map) => map && !map.safeZone && !map.adminOnly);
assert(fieldMap, 'the field-lock test requires one combat map');
purchaseEngine.state.mapId = fieldMap.id;
assert.strictEqual(purchaseEngine.selectSkillModifier(guardianChoice.baseline.id), false, 'Imprint switching should be blocked outside safe zones');
assert.strictEqual(purchaseEngine.state.skillModifiers.activeBySkillId[guardianChoice.signatureSkill.id], guardianChoice.rift.id,
  'a blocked field switch should preserve the active path');

const activeRiftEngine = createAdvancedEngine('guardian');
activeRiftEngine.state.materials.riftSplinter = 24;
activeRiftEngine.state.rift.active = true;
assert.strictEqual(activeRiftEngine.unlockSkillModifier(guardianChoice.rift.id), false, 'an active Rift run should block shaping even in a safe-zone save state');
assert.strictEqual(activeRiftEngine.state.materials.riftSplinter, 24, 'the active-Rift lock should remain atomic');

const wrongClassEngine = createAdvancedEngine('berserker');
wrongClassEngine.state.materials.riftSplinter = 24;
assert.strictEqual(wrongClassEngine.unlockSkillModifier(guardianChoice.rift.id), false, 'a different advanced class should not unlock another class path');
assert.strictEqual(wrongClassEngine.state.materials.riftSplinter, 24, 'wrong-class validation should happen before spending');

const restoredEngine = createProjectStarfallEngine(null, data);
assert.strictEqual(restoredEngine.restore(JSON.parse(JSON.stringify(purchaseEngine.serialize()))), true, 'an Imprint save should restore successfully');
assert(restoredEngine.state.skillModifiers.unlockedModifierIds.includes(guardianChoice.rift.id), 'paid unlocks should survive save restoration without a new schema field');
assert.strictEqual(restoredEngine.state.skillModifiers.activeBySkillId[guardianChoice.signatureSkill.id], guardianChoice.rift.id,
  'the selected path should survive save restoration');

function getGuardianPaidSnapshot(engine) {
  return engine.getSkillModifierSnapshot().modifiers.find((modifier) => modifier.id === guardianChoice.rift.id);
}

const snapshotEngine = createAdvancedEngine('guardian');
snapshotEngine.state.materials.riftSplinter = 23;
let paidSnapshot = getGuardianPaidSnapshot(snapshotEngine);
assert.strictEqual(paidSnapshot.affordable, false, 'the snapshot should expose insufficient Rift Splinters');
assert.strictEqual(paidSnapshot.actionState, 'locked', 'an unaffordable paid path should expose a locked action state');
assert(/Requires 24 Rift Splinters/.test(paidSnapshot.lockedReason), 'the locked snapshot should explain its material requirement');

snapshotEngine.state.materials.riftSplinter = 24;
paidSnapshot = getGuardianPaidSnapshot(snapshotEngine);
assert.strictEqual(paidSnapshot.affordable, true, 'the snapshot should refresh when the material balance changes');
assert.strictEqual(paidSnapshot.actionState, 'unlock', 'an affordable eligible path should expose the unlock action');
assert.strictEqual(paidSnapshot.canUnlock, true, 'the snapshot should explicitly expose unlock availability');

snapshotEngine.state.mapId = fieldMap.id;
paidSnapshot = getGuardianPaidSnapshot(snapshotEngine);
assert.strictEqual(paidSnapshot.actionState, 'locked', 'field maps should replace the purchase action with a lock');
assert(/safe zone/i.test(paidSnapshot.lockedReason), 'the field lock should tell the player where shaping is available');

function fakeTarget(attributes) {
  return { getAttribute: (name) => attributes[name] || null };
}

assert.deepStrictEqual(skillUi.getSkillPanelDomAction(fakeTarget({
  'data-starfall-skill-modifier-unlock': guardianChoice.rift.id
})), {
  handled: true,
  type: 'unlockSkillModifier',
  modifierId: guardianChoice.rift.id
}, 'the DOM action router should identify Rift Imprint unlock buttons');
assert.deepStrictEqual(skillUi.getSkillPanelDomAction(fakeTarget({
  'data-starfall-skill-modifier-select': guardianChoice.baseline.id
})), {
  handled: true,
  type: 'selectSkillModifier',
  modifierId: guardianChoice.baseline.id
}, 'the DOM action router should identify free switching buttons');
assert.deepStrictEqual(skillUi.getSkillPanelRegionAction({
  type: 'skill-modifier-unlock',
  modifierId: guardianChoice.rift.id
}), {
  handled: true,
  type: 'unlockSkillModifier',
  modifierId: guardianChoice.rift.id
}, 'the canvas router should identify Rift Imprint unlock regions');
assert.deepStrictEqual(skillUi.getSkillPanelRegionAction({
  type: 'skill-modifier-select',
  modifierId: guardianChoice.baseline.id
}), {
  handled: true,
  type: 'selectSkillModifier',
  modifierId: guardianChoice.baseline.id
}, 'the canvas router should identify Imprint selection regions');

const uiSource = fs.readFileSync(path.join(__dirname, '../js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(uiSource.includes('aria-labelledby="project-starfall-imprints-title"') && uiSource.includes('aria-describedby="'),
  'the DOM Imprint surface should have a named section and described controls');
assert(uiSource.includes("type: 'skill-modifier-unlock'") && uiSource.includes("type: 'skill-modifier-select'"),
  'the canvas Imprint surface should expose unlock and select hit regions');

assert.strictEqual(RIFT_FIRST_CLEAR_REWARD.materials.riftSplinter, 12, 'one weekly Rift clear should visibly fund half an Imprint');
assert.strictEqual(RIFT_FIRST_CLEAR_REWARD.materials.riftSplinter * 2, guardianChoice.rift.unlockCost.amount,
  'two first-clear reward amounts should map cleanly to one authored Imprint cost');

const retention = createRetentionHealthReport(data);
assert.strictEqual(retention.advancedBuildChoiceCoverageCount, 9, 'retention analysis should require a real horizontal choice for every advanced class');
assert.deepStrictEqual(retention.missingAdvancedBuildChoiceClassIds, [], 'the advanced build-choice matrix should have no uncovered class');
assert.strictEqual(retention.riftImprintSinkCount, 9, 'retention analysis should count every Rift Splinter build sink');
assert.deepStrictEqual(retention.orphanRewardMaterialIds, [], 'Rift reward materials should not be orphaned from long-term sinks');

function measureRuntimeThroughput(engine, choice, modifier, flags) {
  engine.state.skillModifiers.unlockedModifierIds = Array.from(new Set(engine.state.skillModifiers.unlockedModifierIds.concat(modifier.id)));
  engine.state.skillModifiers.activeBySkillId[choice.signatureSkill.id] = modifier.id;
  const enemy = {
    brokenUntil: flags.broken ? Number.POSITIVE_INFINITY : 0,
    marked: flags.marked ? 1 : 0,
    weakPoint: 0
  };
  const damageScale = engine.getSkillModifierDamageScale(choice.signatureSkill, enemy);
  const cooldown = engine.getSkillCooldownDuration(choice.signatureSkill, modifier, {
    cooldownRecoveryPercent: 0,
    mobilityCooldownPercent: 0
  });
  return damageScale / cooldown;
}

const scenarioWeights = [
  { marked: false, broken: false, weight: 0.44 },
  { marked: true, broken: false, weight: 0.36 },
  { marked: false, broken: true, weight: 0.11 },
  { marked: true, broken: true, weight: 0.09 }
];

advancedClassIds.forEach((classId) => {
  const choice = choicesByClass.get(classId);
  const engine = createAdvancedEngine(classId);
  const average = (modifier) => scenarioWeights.reduce((sum, scenario) =>
    sum + measureRuntimeThroughput(engine, choice, modifier, scenario) * scenario.weight, 0);
  const baselineAverage = average(choice.baseline);
  const riftAverage = average(choice.rift);
  const baselineBest = measureRuntimeThroughput(engine, choice, choice.baseline, { marked: true, broken: true });
  const riftBest = measureRuntimeThroughput(engine, choice, choice.rift, { marked: true, broken: true });
  const averageDelta = riftAverage / baselineAverage - 1;
  const bestDelta = riftBest / baselineBest - 1;
  assert(Math.abs(averageDelta) <= 0.050001,
    `${classId} Rift path average runtime throughput should remain within 5% of baseline (actual ${(averageDelta * 100).toFixed(2)}%)`);
  assert(Math.abs(bestDelta) <= 0.100001,
    `${classId} Rift path best-case runtime throughput should remain within 10% of baseline (actual ${(bestDelta * 100).toFixed(2)}%)`);
});

function measureSummedRuntimeLines(modifier) {
  const choice = choicesByClass.get('stormMage');
  const engine = createAdvancedEngine('stormMage');
  let total = 0;
  let lineCalls = 0;
  engine.getSkillModifierForSkill = () => modifier;
  engine.getStats = () => ({ skillEffectPercent: 0 });
  engine.rollDamageResult = (amount) => ({ amount, critical: false });
  engine.damageEnemy = (enemy, amount) => {
    total += amount;
    lineCalls += 1;
  };
  engine.pushSkillImpactEffect = () => {};
  engine.applySkillHitEffects = () => {};
  engine.applySkillModifierHitEffects = () => {};
  engine.applyBossBreakProgress = () => {};
  engine.damageEnemyWithSkillLines({ x: 0, y: 0, w: 40, h: 40, uid: 'line-test' }, 100, choice.signatureSkill);
  return { total, lineCalls };
}

const conductorLines = measureSummedRuntimeLines(choicesByClass.get('stormMage').baseline);
const stormglassLines = measureSummedRuntimeLines(choicesByClass.get('stormMage').rift);
assert.strictEqual(conductorLines.lineCalls, 4, 'Conductor should retain its authored fourth presentation line');
assert.strictEqual(stormglassLines.lineCalls, 3, 'Stormglass should trade the presentation line for a heavier direct hit');
assert(Math.abs(conductorLines.total - 58) < 0.0001, 'extra lines should divide, rather than multiply, the same runtime damage total');
assert(Math.abs(stormglassLines.total - 58 * 1.035) < 0.0001, 'Stormglass runtime damage should apply its measured 3.5% tradeoff');

console.log('Project Starfall Rift Imprints tests passed.');
