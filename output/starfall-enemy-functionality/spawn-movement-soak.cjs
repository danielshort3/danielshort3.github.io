'use strict';
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const engine = createProjectStarfallEngine(null, data);
engine.state.player.classId = 'fighter';
engine.playAudioCue = () => true;
let clock = 1900000000000;
let seed = 1234;
Date.now = () => clock;
Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
const issues = [];
let actors = 0;
const surfaceY = (platform, x) => platform.shape === 'slope'
  ? platform.y + (platform.y2 - platform.y) * Math.max(0, Math.min(1, (x - platform.x) / platform.w))
  : platform.y;
const position = actor => ({ x: actor.x, y: actor.y, w: actor.w, h: actor.h, vx: actor.vx, vy: actor.vy,
  grounded: actor.grounded, groundedPlatformIndex: actor.groundedPlatformIndex, climbing: actor.climbing,
  dropThroughUntil: actor.dropThroughUntil, pathTargetPlatformIndex: actor.pathTargetPlatformIndex });
for (const map of data.MAPS) {
  engine.changeMap(map.id, { silent: true });
  engine.state.player.hp = engine.state.player.maxHp = 1e9;
  const originalActors = engine.enemies.slice();
  actors += originalActors.length;
  for (let stage = 0; stage < 2; stage += 1) {
    if (stage && originalActors.length) {
      const target = originalActors[Math.floor(originalActors.length / 2)];
      Object.assign(engine.state.player, { x: target.x + 130, y: target.y + target.h - engine.state.player.h });
      engine.enemies.forEach(enemy => engine.setEnemyAggro(enemy, engine.getCombatCharacterByTarget('player', 'player'), 'audit', 20));
    }
    for (let tick = 0; tick < 240; tick += 1) {
      clock += 1000 / 60;
      const previous = new Map(engine.enemies.map(enemy => [enemy.uid, position(enemy)]));
      engine.updateEnemies(1 / 60);
      engine.updateWaveSpawns();
      engine.updateDungeonBossRespawns();
      for (const enemy of engine.enemies) {
        const record = { map: map.id, stage, tick, id: enemy.id, before: previous.get(enemy.uid), after: position(enemy),
          player: position(engine.state.player) };
        if (![enemy.x, enemy.y, enemy.vx, enemy.vy].every(Number.isFinite)) issues.push({ ...record, issue: 'nonfinite' });
        if (enemy.x < -1 || enemy.x + enemy.w > engine.runtime.worldWidth + 1 || enemy.y < -1 ||
          enemy.y + enemy.h > engine.runtime.worldHeight + 50) issues.push({ ...record, issue: 'bounds' });
        if (enemy.grounded && enemy.data.behavior !== 'flyer' && !enemy.climbing) {
          const platform = engine.runtime.platforms[enemy.groundedPlatformIndex];
          if (!platform || platform.id !== enemy.groundedPlatformId) issues.push({ ...record, issue: 'stale-platform' });
          else {
            const delta = surfaceY(platform, enemy.x + enemy.w / 2) - enemy.y - enemy.h;
            if (Math.abs(delta) > 3) issues.push({ ...record, issue: 'foot-delta', delta, platform });
          }
        }
      }
    }
  }
}
console.log(JSON.stringify({ maps: data.MAPS.length, actors, secondsPerMap: 8, issues }, null, 2));
