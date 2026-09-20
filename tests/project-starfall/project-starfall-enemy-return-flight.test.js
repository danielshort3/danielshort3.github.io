'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

const originalNow = Date.now;
let clock = 1800000000000;
Date.now = () => clock;

function fixture(id, x = 500) {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, { classId: 'fighter', level: 200, x: 100, hp: 100000, maxHp: 100000 });
  const enemy = engine.createEnemy(data.ENEMIES.find(entry => entry.id === id), { x, platformIndex: 0 });
  Object.assign(enemy, { level: 1, speedScale: 1, attackCd: 10000, eliteAffixIds: [] });
  engine.enemies = [enemy];
  return { engine, enemy };
}

try {
  for (const fps of [30, 60, 120]) {
    const { engine, enemy } = fixture('slimelet');
    const homeBounds = engine.getEnemyWanderBounds(enemy);
    enemy.x = homeBounds.right + 220;
    enemy.staggered = 0.1;
    const start = enemy.x;
    let maxStep = 0;
    for (let frame = 0; frame < fps * 7; frame += 1) {
      clock += 1000 / fps;
      const before = enemy.x;
      engine.updateEnemies(1 / fps);
      maxStep = Math.max(maxStep, Math.abs(enemy.x - before));
      assert(enemy.grounded, 'an enemy returning along a flat lane stays grounded');
    }
    assert(maxStep <= enemy.data.speed / fps + 0.01,
      `lost aggro must not snap 220 pixels into the patrol box at ${fps} Hz (step=${maxStep})`);
    assert(enemy.x < start - 200 && enemy.x >= homeBounds.left && enemy.x <= homeBounds.right,
      'a disengaged enemy walks back into its home patrol range');
  }

  for (const definition of data.ENEMIES.filter(entry => entry.behavior === 'flyer')) {
    const { engine, enemy } = fixture(definition.id);
    const homeY = enemy.spawnY;
    enemy.vy = 500;
    enemy.grounded = true;
    for (let frame = 0; frame < 60 * 12; frame += 1) {
      clock += 1000 / 60;
      const before = enemy.y;
      engine.updateEnemies(1 / 60);
      assert(!enemy.grounded && enemy.groundedPlatformId === '' && enemy.groundedPlatformIndex === -1,
        `${definition.id}: flyers never acquire a ground body from platform resolution`);
      assert(Math.abs(enemy.y - before) <= Math.max(30, definition.speed * 0.72) / 60 + 0.01,
        `${definition.id}: vertical movement is bounded, including stale velocity recovery`);
      assert(Math.abs(enemy.y - homeY) <= 9.01, `${definition.id}: passive hover must not accumulate vertical drift`);
    }
    const p = engine.state.player;
    Object.assign(p, { x: enemy.x + 300, y: enemy.y + 210 });
    Object.assign(enemy, { aggroTargetKind: 'player', aggroTargetId: 'player', aggroUntil: Infinity });
    const initialY = enemy.y;
    for (let frame = 0; frame < 60 * 5; frame += 1) {
      clock += 1000 / 60;
      engine.updateEnemies(1 / 60);
    }
    assert(enemy.y > initialY + 100, `${definition.id}: a flyer smoothly follows a target to a lower lane`);
    assert(Math.abs(enemy.y - (p.y - 56)) <= 10, `${definition.id}: chase maintains a stable hover height`);
    enemy.vy = 500;
    const lockedY = enemy.y;
    engine.updateEnemyFlight(enemy, p, 1 / 60, clock / 1000, true);
    assert.strictEqual(enemy.vy, 0, `${definition.id}: windup or stagger suspends vertical steering`);
    assert.strictEqual(enemy.y, lockedY);
  }
  console.log('Project Starfall return-home and flyer movement tests passed (3 frame rates, all 5 flyer types).');
} finally {
  Date.now = originalNow;
}
