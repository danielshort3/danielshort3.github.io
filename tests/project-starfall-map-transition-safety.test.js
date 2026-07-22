'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const engine = createProjectStarfallEngine(null, data);
assert.strictEqual(engine.chooseClass('fighter'), true);
assert.strictEqual(engine.changeMap('banditRidgeCamp'), true);

const ground = engine.runtime.platforms[0];
const enemyData = data.ENEMIES.find((enemy) => enemy.id === 'banditCutter');
assert(enemyData, 'Bandit Cutter should exist for the lethal-hit transition regression');

const enemy = engine.createEnemy(enemyData, {
  x: 500,
  platformIndex: ground.index,
  platformId: ground.id
});
const player = engine.state.player;
Object.assign(player, {
  x: enemy.x - 10,
  y: enemy.y + enemy.h - player.h,
  hp: 1,
  shield: 0,
  invulnerableUntil: 0,
  grounded: true,
  groundedPlatformId: ground.id,
  groundedPlatformIndex: ground.index
});
Object.assign(enemy, {
  damage: 999999,
  attackCd: 0,
  x: player.x + 10,
  y: player.y + player.h - enemy.h
});

engine.enemies = [enemy];
engine.setEnemyAggro(
  enemy,
  engine.getCombatCharacterByTarget('player', 'player'),
  'transition-regression',
  10
);
const staleBanditEnemies = engine.enemies;
const staleBanditRuntime = engine.runtime;

engine.updateEnemies(0.016);

assert.strictEqual(engine.state.mapId, 'banditRidgeCamp',
  'the telegraphed melee windup should not recover the player before its impact frame');
assert(enemy.pendingAttack &&
  enemy.pendingAttack.kind === 'melee' &&
  enemy.animationState === 'telegraph',
  'the lethal Bandit Cutter attack should enter the normal readable melee windup');
engine.updateEnemies(Number(enemy.pendingAttack.windup || 0) + 0.01);

assert.strictEqual(engine.state.mapId, 'starfallCrossing',
  'lethal enemy damage should recover the player at Starfall Crossing');
assert.strictEqual(engine.runtime.id, 'starfallCrossing');
assert.notStrictEqual(engine.runtime, staleBanditRuntime,
  'recovery should keep ownership of the newly-created town runtime');
assert.notStrictEqual(engine.enemies, staleBanditEnemies,
  'the completed Bandit Ridge enemy frame must not reclaim the town enemy list');
assert.deepStrictEqual(engine.enemies, [],
  'Bandit enemies must not remain active around the Starfall Crossing shops');

const bossEntryTransitions = [];
engine.pruneRendererMapDerivedCaches = (previousMapId) => {
  bossEntryTransitions.push(`prune:${previousMapId}`);
  return null;
};
engine.maybeSchedulePixiPrewarm = (options) => {
  bossEntryTransitions.push(`prewarm:${options && options.force === true}`);
};
engine.queueCurrentAssetPreload = (label) => {
  bossEntryTransitions.push(`preload:${label}`);
  return Promise.resolve(true);
};
assert.strictEqual(engine.enterBossEncounter('eclipseSovereign', { admin: true }), true,
  'the Eclipse Sovereign encounter should open from a normal map state');
assert.strictEqual(engine.state.mapId, 'eclipseThrone');
assert(engine.runtime.asset.endsWith('/maps/eclipse-throne-v2.webp'),
  'boss entry should switch to the authored Eclipse panorama');
assert(bossEntryTransitions.includes('prune:starfallCrossing'),
  'boss entry should prune stale map-derived renderer textures');
assert(bossEntryTransitions.includes('prewarm:true'),
  'boss entry should schedule the same immediate Pixi prewarm as normal map travel');
assert(bossEntryTransitions.includes('preload:boss-entry:eclipseThrone'),
  'boss entry should queue its background, environment, and actor assets');

console.log('Project Starfall map-transition safety regression checks passed.');
