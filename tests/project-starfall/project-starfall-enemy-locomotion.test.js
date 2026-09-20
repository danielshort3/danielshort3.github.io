'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { createMapRuntime } = require('../../js/games/project-starfall/engine/map-runtime.js');
const { getPlatformSurfaceY } = require('../../js/games/project-starfall/core/geometry.js');

let checks = 0;
function check(condition, message) {
  checks += 1;
  assert(condition, message);
}

function fixture(map, connectionIndex, descending, enemyId = 'lavaTick') {
  const engine = createProjectStarfallEngine(null, data);
  engine.runtime = createMapRuntime(map, null, { maps: data.MAPS });
  engine.state.mapId = map.id;
  const connection = engine.runtime.rampConnections[connectionIndex];
  const source = engine.runtime.platforms[descending ? connection.upperPlatformIndex : connection.lowerPlatformIndex];
  const destination = engine.runtime.platforms[descending ? connection.lowerPlatformIndex : connection.upperPlatformIndex];
  const startSeam = descending ? connection.upperX : connection.lowerX;
  const endSeam = descending ? connection.lowerX : connection.upperX;
  const direction = Math.sign(endSeam - startSeam);
  const definition = data.ENEMIES.find((entry) => entry.id === enemyId);
  const enemy = engine.createEnemy(definition, { x: startSeam, y: getPlatformSurfaceY(source, startSeam), platformIndex: source.index });
  const clampCenter = (x, platform, width) => Math.max(12 + width / 2, platform.x + width / 2 + 2,
    Math.min(engine.runtime.worldWidth - width / 2 - 12, platform.x + platform.w - width / 2 - 2, x));
  const startCenter = clampCenter(startSeam - direction * 70, source, enemy.w);
  Object.assign(enemy, {
    x: startCenter - enemy.w / 2, y: getPlatformSurfaceY(source, startCenter) - enemy.h,
    grounded: true, groundedPlatformId: source.id, groundedPlatformIndex: source.index,
    vx: 0, vy: 0, speedScale: 1, elite: false, eliteAffixIds: [], attackCd: 10000,
    aggroTargetKind: 'player', aggroTargetId: 'player', aggroUntil: Infinity
  });
  const player = engine.state.player;
  const targetCenter = clampCenter(endSeam + direction * 100, destination, player.w);
  Object.assign(player, {
    classId: 'fighter', x: targetCenter - player.w / 2,
    y: getPlatformSurfaceY(destination, targetCenter) - player.h,
    grounded: true, groundedPlatformId: destination.id, groundedPlatformIndex: destination.index,
    hp: 100000, maxHp: 100000, invulnerableUntil: Infinity
  });
  engine.enemies = [enemy];
  return { engine, enemy, source, destination, connection, direction };
}

function traverse(map, connectionIndex, descending, enemyId = 'lavaTick', fps = 60) {
  const { engine, enemy, destination, connection, direction } = fixture(map, connectionIndex, descending, enemyId);
  const label = `${map.id}/${connection.id}/${enemyId}/${descending ? 'down' : 'up'}/${fps}Hz`;
  let reached = false;
  let enteredRamp = false;
  for (let frame = 0; frame < fps * 9; frame += 1) {
    const previousX = enemy.x;
    const previousY = enemy.y;
    engine.frameId += 1;
    engine.updateEnemies(1 / fps);
    const step = enemy.x - previousX;
    check(Number.isFinite(enemy.x) && Number.isFinite(enemy.y), `${label}: finite world position`);
    check(!enteredRamp || step * direction >= -0.001, `${label}: reversed on frame ${frame} by ${step}`);
    check(Math.abs(step) <= Math.max(150, enemy.data.speed) / fps + 0.001, `${label}: teleported on frame ${frame} by ${step}`);
    check(enemy.grounded, `${label}: lost ramp contact on frame ${frame} at ${enemy.x}, ${enemy.y}`);
    const platform = engine.runtime.platforms[enemy.groundedPlatformIndex];
    if (platform.index === connection.rampPlatformIndex) enteredRamp = true;
    check(Math.abs(enemy.y + enemy.h - getPlatformSurfaceY(platform, enemy.x + enemy.w / 2)) < 0.001,
      `${label}: feet must remain on the selected surface`);
    check(Math.abs(enemy.y - previousY) <= 10, `${label}: vertical jump at ramp seam`);
    if (enteredRamp && enemy.groundedPlatformId === destination.id) {
      reached = true;
      break;
    }
  }
  check(reached, `${label}: did not reach player on destination platform; ended at ${enemy.x}, ${enemy.y}`);
}

function testMovingTargetAndCrowds(map) {
  for (const index of [0, 2]) {
    const { engine, enemy, connection, direction, destination } = fixture(map, index, true);
    const ramp = engine.runtime.platforms[connection.rampPlatformIndex];
    const player = engine.state.player;
    let playerCenter = connection.rampUpperX + direction * 90;
    const finalCenter = player.x + player.w / 2;
    for (let frame = 0; frame < 210; frame += 1) {
      playerCenter += direction * 125 / 60;
      if ((finalCenter - playerCenter) * direction < 0) playerCenter = finalCenter;
      const support = playerCenter >= ramp.x && playerCenter <= ramp.x + ramp.w ? ramp : destination;
      Object.assign(player, { x: playerCenter - player.w / 2, y: getPlatformSurfaceY(support, playerCenter) - player.h,
        groundedPlatformId: support.id, groundedPlatformIndex: support.index });
      const before = enemy.x;
      engine.frameId += 1;
      engine.updateEnemies(1 / 60);
      check((enemy.x - before) * direction >= -0.001, 'following a player down a ramp must not reverse each frame');
      check(enemy.grounded, 'following a moving player must retain slope contact');
      if (Math.abs(enemy.vx) > 0.01) check(enemy.facing === direction, 'downhill movement must retain its facing');
    }

    const center = ramp.x + ramp.w / 2;
    Object.assign(enemy, { x: center - enemy.w / 2, y: getPlatformSurfaceY(ramp, center) - enemy.h,
      grounded: true, groundedPlatformId: ramp.id, groundedPlatformIndex: ramp.index, groundRoutePlatformIndex: -1 });
    engine.enemies = [enemy, ...Array.from({ length: 5 }, (_, offset) => Object.assign({}, enemy, {
      uid: `slope-pack-${offset}`, x: enemy.x + offset * 0.2
    }))];
    for (const member of engine.enemies) member.y = getPlatformSurfaceY(ramp, member.x + member.w / 2) - member.h;
    for (let frame = 0; frame < 180; frame += 1) {
      engine.frameId += 1;
      engine.updateEnemies(1 / 60);
      for (const member of engine.enemies) {
        check(member.grounded, 'crowd separation must not dislodge a descending enemy');
        const platform = engine.runtime.platforms[member.groundedPlatformIndex];
        check(Math.abs(member.y + member.h - getPlatformSurfaceY(platform, member.x + member.w / 2)) < 0.001,
          'crowd separation must update height along a slope, including after the physics pass');
      }
    }
  }
}

function testFallsAndOneWayPlatforms(map) {
  const { engine, enemy } = fixture(map, 0, true);
  const top = { id: 'upper-landing', index: 0, x: 100, y: 300, w: 600, h: 24, dropThrough: true };
  const low = { id: 'lower-landing', index: 1, x: 100, y: 370, w: 600, h: 24, dropThrough: true };
  engine.runtime.platforms = [top, low];
  for (const preferred of [0, 1, -1]) {
    Object.assign(enemy, { x: 300, y: 400 - enemy.h, previousY: 260 - enemy.h, vy: 900,
      grounded: false, groundedPlatformId: '', groundedPlatformIndex: preferred });
    engine.resolvePlatforms(enemy);
    check(enemy.groundedPlatformId === top.id, 'a fast fall lands on the first crossed shelf regardless of old support index');
    check(enemy.y + enemy.h === top.y, 'fast falling must not tunnel to the lower shelf');
  }
  Object.assign(enemy, { x: 300, y: top.y - enemy.h, previousY: top.y - enemy.h,
    vy: -300, grounded: false, groundedPlatformId: '', groundedPlatformIndex: top.index });
  enemy.y -= 5;
  engine.resolveEnemyGroundMovement(enemy, enemy.x, top.y - enemy.h, true);
  check(!enemy.grounded && enemy.vy === -300, 'jump or upward knockback must leave ground adhesion');
  enemy.y = top.y - enemy.h + 3;
  enemy.previousY = top.y - enemy.h;
  enemy.vy = 180;
  engine.beginDropThrough(enemy, 1, top);
  engine.resolveEnemyGroundMovement(enemy, enemy.x, top.y - enemy.h, true);
  check(!enemy.grounded, 'an explicit drop must pass through its source shelf');
  enemy.previousY = enemy.y;
  enemy.y = low.y - enemy.h + 4;
  engine.resolveEnemyGroundMovement(enemy, enemy.x, enemy.previousY, false);
  check(enemy.groundedPlatformId === low.id, 'drop-through excludes only its source, so the lower shelf catches the enemy');
}

function testRampSlowing(map) {
  const baseline = fixture(map, 0, true);
  const slowed = fixture(map, 0, true);
  slowed.enemy.slowed = 2;
  const baselineX = baseline.enemy.x;
  const slowedX = slowed.enemy.x;
  for (let frame = 0; frame < 60; frame += 1) {
    baseline.engine.updateEnemies(1 / 60);
    slowed.engine.updateEnemies(1 / 60);
  }
  check(Math.abs((slowed.enemy.x - slowedX) / (baseline.enemy.x - baselineX) - 0.45) < 0.001,
    'slow applies to the whole ramp route speed, including its minimum walking speed');
}

function testActualPlayerRampInput(map) {
  for (const index of [0, 2]) {
    const { engine, enemy, connection, direction } = fixture(map, index, true);
    const player = engine.state.player;
    const upper = engine.runtime.platforms[connection.upperPlatformIndex];
    Object.assign(player, { x: connection.upperX - direction * 25 - player.w / 2,
      y: connection.upperY - player.h, grounded: true, groundedPlatformId: upper.id,
      groundedPlatformIndex: upper.index, vx: 0, vy: 0 });
    engine.setInput(direction < 0 ? 'left' : 'right', true);
    for (let frame = 0; frame < 200; frame += 1) {
      const previousX = enemy.x;
      engine.frameId += 1;
      // Exercise the real keyboard-input player physics and natural platform
      // transitions, rather than only placing a target on each ramp surface.
      engine.updatePlayer(1 / 60);
      engine.updateEnemies(1 / 60);
      check((enemy.x - previousX) * direction >= -0.001, 'natural player ramp movement must not make the pursuing enemy reverse');
      check(enemy.grounded, 'natural player ramp movement must not interrupt enemy ground support');
    }
    engine.setInput(direction < 0 ? 'left' : 'right', false);
  }
}

function main() {
  let rampCount = 0;
  for (const map of data.MAPS) {
    if (!(map.enemies || []).length && !map.isDungeon) continue;
    const runtime = createMapRuntime(map, null, { maps: data.MAPS });
    for (let index = 0; index < runtime.rampConnections.length; index += 1) {
      traverse(map, index, true);
      traverse(map, index, false);
      rampCount += 1;
    }
  }
  const meadow = data.MAPS.find((map) => map.id === 'greenrootMeadow');
  for (const enemyId of ['faultSkitter', 'lavaTick', 'quarryColossus']) {
    for (const fps of [30, 60, 120]) {
      for (const descending of [false, true]) {
        for (const index of [0, 2]) traverse(meadow, index, descending, enemyId, fps);
      }
    }
  }
  testMovingTargetAndCrowds(meadow);
  testFallsAndOneWayPlatforms(meadow);
  testRampSlowing(meadow);
  testActualPlayerRampInput(meadow);
  console.log(`Project Starfall enemy locomotion: ${checks} checks passed across ${rampCount} authored combat ramps, both directions, body sizes, and 30/60/120Hz.`);
}

if (require.main === module) main();
module.exports = { fixture, traverse, main };
