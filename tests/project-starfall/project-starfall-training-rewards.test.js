'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const engine = createProjectStarfallEngine(null, data);
engine.state.player.classId = 'fighter';
engine.playAudioCue = () => true;
const ordinary = data.MAPS.find(map => map.id === 'orebackQuarry');
const { getMonsterXp, getTrainingXpMultiplier } = require('../../js/games/project-starfall/engine/combat-formulas.js');
const enemy = { data: { behavior: 'melee' } };
engine.changeMap(ordinary.id, { silent: true });
assert.strictEqual(engine.getTrainingXpMultiplier(enemy), ordinary.trainingXpMultiplier || 1);
for (const [value, expected] of [[undefined, 1], [.9, .9], [1.1, 1.1], [5, 1.15], [.1, .85], [NaN, 1]]) {
  assert.strictEqual(getTrainingXpMultiplier({ ...ordinary, trainingXpMultiplier: value }, false, false), expected);
}
const authored = { ...ordinary, trainingXpMultiplier: 1.15 };
assert.strictEqual(getTrainingXpMultiplier(authored, true, false), 1, 'boss rewards unchanged');
assert.strictEqual(getTrainingXpMultiplier(authored, false, true), 1, 'trial rewards unchanged');
for (const map of data.MAPS.filter(map => map.safeZone || map.isDungeon || map.bossRoom || map.layoutRole === 'endlessField')) {
  assert.strictEqual(getTrainingXpMultiplier({ ...map, trainingXpMultiplier: 1.15 }, false, false), 1, map.id + ': special reward context unchanged');
}
const cinder = data.MAPS.find(map => map.id === 'cinderHollow');
assert.strictEqual(cinder.trainingXpMultiplier, 0.93, 'the measured Cinder adjustment belongs to authored map data');
assert.deepStrictEqual(data.MAPS.filter(map => getTrainingXpMultiplier(map, false, false) !== 1).map(map => map.id), ['cinderHollow'], 'this adjustment must not alter other field rewards');
engine.state.player.level = 28;
engine.checkLevelUp = () => {};
engine.changeMap(cinder.id, { silent: true });
const rewardEnemy = engine.createEnemy(data.ENEMIES.find(entry => entry.id === 'lavaTick'), { x: 1100, platformIndex: 0 });
rewardEnemy.elite = false;
rewardEnemy.eliteAffixIds = [];
const rewardStart = { xp: engine.state.player.xp, currency: engine.state.player.currency };
const expectedXp = Math.max(1, Math.round(getMonsterXp(rewardEnemy.level, rewardEnemy.data) * 0.93));
engine.defeatEnemy(rewardEnemy);
assert.strictEqual(engine.state.player.xp - rewardStart.xp, expectedXp, 'real ordinary kills apply the authored scalar with per-kill integer rounding');
assert.strictEqual(engine.state.player.currency - rewardStart.currency, 14 + rewardEnemy.level * 4, 'the XP adjustment does not scale earned currency');
assert.strictEqual(engine.getTrainingXpMultiplier({ data: { behavior: 'boss' } }), 1, 'Cinder boss rewards remain excluded');
const eliteEnemy = engine.createEnemy(data.ENEMIES.find(entry => entry.id === 'lavaTick'), { x: 1300, platformIndex: 0 });
eliteEnemy.elite = true;
eliteEnemy.eliteAffixIds = [];
const eliteXpStart = engine.state.player.xp;
assert.strictEqual(engine.getTrainingXpMultiplier(eliteEnemy), 1, 'the engine protects random elites through its boss-or-elite reward predicate');
engine.defeatEnemy(eliteEnemy);
assert.strictEqual(engine.state.player.xp - eliteXpStart, getMonsterXp(eliteEnemy.level, eliteEnemy.data), 'real elite kills retain full XP despite the Cinder normal-kill adjustment');
engine.runtime.isTrialInstance = true;
assert.strictEqual(engine.getTrainingXpMultiplier(enemy), 1, 'Cinder trial rewards remain excluded');
console.log('Starfall training reward scope/default/bounds passed.');
