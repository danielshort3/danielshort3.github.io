'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

const originalRandom = Math.random;
const originalNow = Date.now;
let clock = 1900000000000;
let seed = 0x51a7f411;
let assertions = 0;
let arrivals = 0;
let actors = 0;

function check(condition, message) {
  assertions += 1;
  assert(condition, message);
}

function overlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function surfaceY(platform, x) {
  if (platform.shape !== 'slope') return platform.y;
  const ratio = Math.max(0, Math.min(1, (x - platform.x) / platform.w));
  return platform.y + (platform.y2 - platform.y) * ratio;
}

function createEngine() {
  const engine = createProjectStarfallEngine(null, data);
  engine.state.player.classId = 'fighter';
  engine.playAudioCue = () => true;
  return engine;
}

try {
  Date.now = () => clock;
  Math.random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const engine = createEngine();

  // Exercise real map-entry placement, authored groups and all portal arrivals,
  // including reverse travel. Quotas include slots intentionally deferred while
  // the player occupies the only safe space in a territory.
  for (const map of data.MAPS) {
    const entries = [{}, ...(map.portals || []).map(portal => ({ entryPortalId: portal.id }))];
    for (const entry of entries) {
      engine.resetWaveReplacementQueues({ silent: true });
      check(engine.changeMap(map.id, { silent: true, ...entry }), `${map.id}: enters successfully`);
      const label = `${map.id}/${entry.entryPortalId || 'default'}`;
      const wave = engine.getWaveState();
      arrivals += 1;
      check(engine.enemies.length + wave.pending.length === engine.getWaveMax(map), `${label}: preserves population quota`);
      check(!map.safeZone || engine.enemies.length === 0, `${label}: town remains safe`);
      const bossIds = engine.enemies.filter(enemy => enemy.data.behavior === 'boss').map(enemy => enemy.id);
      check(new Set(bossIds).size === bossIds.length, `${label}: bosses are unique`);
      for (const enemy of engine.enemies) {
        actors += 1;
        const platform = engine.runtime.platforms[enemy.spawnPlatformIndex];
        check(platform && platform.id === enemy.spawnPlatformId, `${label}/${enemy.id}: known foothold`);
        check(enemy.x >= platform.x && enemy.x + enemy.w <= platform.x + platform.w,
          `${label}/${enemy.id}: entire terrain body fits on its platform`);
        check(enemy.x >= 0 && enemy.x + enemy.w <= engine.runtime.worldWidth && enemy.y >= 0 &&
          enemy.y + enemy.h <= engine.runtime.worldHeight, `${label}/${enemy.id}: inside world`);
        check(platform.shape !== 'slope', `${label}/${enemy.id}: spawns on a flat`);
        if (enemy.data.behavior !== 'flyer') {
          check(Math.abs(enemy.y + enemy.h - surfaceY(platform, enemy.x + enemy.w / 2)) < 0.001,
            `${label}/${enemy.id}: feet match its assigned floor`);
        }
        check(!overlap(enemy, engine.state.player), `${label}/${enemy.id}: never appears inside the arriving player`);
        const group = engine.getRuntimeSpawnGroupById(enemy.spawnGroupId);
        if (group) {
          check(group.platformIds.includes(platform.id), `${label}/${enemy.id}: within authored territory`);
          check(group.enemyWeights.some(entry => entry.enemyId === enemy.id), `${label}/${enemy.id}: correct encounter roster`);
        }
      }
      for (const group of engine.isTrainingRespawnMap(map) ? engine.getRuntimeSpawnGroups(map) : []) {
        const pending = wave.pending.filter(entry => entry.spawnGroupId === group.id).length;
        check(engine.getSpawnGroupAliveCount(group.id) + pending === engine.getSpawnGroupPopulationTarget(group),
          `${label}/${group.id}: group quota preserved`);
      }
      const uids = engine.enemies.map(enemy => enemy.uid);
      const pendingCount = wave.pending.length;
      engine.spawnEnemiesIfNeeded();
      check(JSON.stringify(engine.enemies.map(enemy => enemy.uid)) === JSON.stringify(uids), `${label}: repeated initialization does not duplicate enemies`);
      check(wave.pending.length === pendingCount, `${label}: repeated initialization does not duplicate deferred slots`);
    }
  }

  // The old weighted selection could place every dungeon enemy at one point
  // under repeated random values. Reservation must work independently of luck.
  Math.random = () => 0.5;
  for (const map of data.MAPS.filter(map => map.isDungeon)) {
    engine.resetWaveReplacementQueues({ silent: true });
    engine.changeMap(map.id, { silent: true });
    engine.enemies.forEach((enemy, index) => {
      engine.enemies.slice(index + 1).forEach(other => {
        check(!overlap(enemy, other), `${map.id}: ${enemy.id} and ${other.id} have distinct initial positions`);
      });
    });
  }

  // Real reverse-entry regressions previously placed enemies directly on the
  // portal: Thornpath's two eastern entrances and Frostfen's glacier entrance.
  for (const [mapId, portalId] of [
    ['thornpathThicket', 'thornpath_bandit'],
    ['thornpathThicket', 'thornpath_rustcoil_outpost'],
    ['frostfenOutskirts', 'frostfen_glacier']
  ]) {
    engine.resetWaveReplacementQueues({ silent: true });
    engine.changeMap(mapId, { silent: true, entryPortalId: portalId });
    const player = engine.state.player;
    engine.enemies.filter(enemy => enemy.data.behavior !== 'flyer').forEach(enemy => {
      check(Math.abs(enemy.x + enemy.w / 2 - player.x - player.w / 2) >= 260 ||
        Math.abs(enemy.y + enemy.h - player.y - player.h) >= 160,
      `${portalId}: safe same-lane arrival distance`);
    });
  }

  // Occupied territory must defer population, then fill through the usual
  // cadence, keeping the slot queued if the player is still standing there.
  const blocked = createEngine();
  blocked.changeMap('greenrootMeadow', { silent: true });
  const territory = { ...blocked.runtime.platforms[1], index: 0, x: 100, w: 200 };
  const group = {
    ...blocked.runtime.spawnGroups[0], population: 1, maxPopulation: 1,
    partyScaling: 'none', platformIds: [territory.id], platformIndices: [0]
  };
  blocked.runtime.platforms = [territory];
  blocked.runtime.spawnGroups = [group];
  blocked.runtime.spawnPoints = [{ id: 'occupied', x: 190, y: territory.y, platformIndex: 0, platformId: territory.id }];
  blocked.enemies = [];
  blocked.resetWaveReplacementQueues({ silent: true });
  Object.assign(blocked.state.player, { x: 180, y: territory.y - blocked.state.player.h, attackTimer: 0, skillTimer: 0 });
  blocked.spawnInitialEnemies();
  const blockedWave = blocked.getWaveState();
  check(blocked.enemies.length === 0 && blockedWave.pending.length === 1, 'occupied territory defers its population slot');
  blocked.spawnInitialEnemies();
  check(blockedWave.pending.length === 1, 'blocked initialization is idempotent');
  clock += (group.respawnSeconds + 1) * 1000;
  blocked.updateWaveSpawns();
  check(blocked.enemies.length === 0 && blockedWave.pending.length === 1, 'blocked respawn waits while the player remains there');
  blocked.state.player.x = 900;
  clock += 2000;
  blocked.updateWaveSpawns();
  check(blocked.enemies.length === 1 && blockedWave.pending.length === 0, 'vacated territory fills its deferred slot');
  const respawned = blocked.enemies[0];
  check(respawned.spawnPlatformId === territory.id && respawned.x + respawned.w <= territory.x + territory.w,
    'deferred spawn stays fully on its authored platform');
  blocked.defeatEnemy(respawned);
  blocked.defeatEnemy(respawned);
  check(blockedWave.pending.length === 1, 'one defeat queues one replacement');
  blocked.enemies = [];
  blocked.updateWaveSpawns();
  check(blocked.enemies.length === 0, 'replacement never skips the group respawn delay');
  clock += (group.respawnSeconds + 1) * 1000;
  blocked.updateWaveSpawns();
  check(blocked.enemies.length === 1 && blockedWave.pending.length === 0, 'ordinary enemy replacement completes after its cadence');

  // Boss cooldown reserves the boss's slot rather than replacing it with an
  // extra ordinary enemy and overpopulating the room when the boss returns.
  const dungeon = createEngine();
  const dungeonMap = data.MAPS.find(map => map.id === 'brambleDepths');
  dungeon.getDungeonState().bossRespawnAt[dungeonMap.dungeonId] = clock + 60000;
  dungeon.changeMap(dungeonMap.id, { silent: true });
  const bossCount = dungeon.getDungeonBossIds(dungeonMap).length;
  check(dungeon.enemies.length === dungeon.getWaveMax(dungeonMap) - bossCount, 'cooldown reserves boss population slots');
  check(dungeon.enemies.every(enemy => enemy.data.behavior !== 'boss'), 'boss remains absent until cooldown expires');
  clock += 61000;
  dungeon.updateDungeonBossRespawns();
  dungeon.updateDungeonBossRespawns();
  check(dungeon.enemies.length === dungeon.getWaveMax(dungeonMap), 'returning boss restores the room quota exactly once');
  const wave = dungeon.getWaveState();
  const queuedBeforeAdd = wave.pending.length;
  check(dungeon.spawnBossEncounterAdd({ adds: ['slimelet'] }, 0, { quiet: true }), 'boss can summon an encounter add');
  const add = dungeon.enemies[dungeon.enemies.length - 1];
  dungeon.defeatEnemy(add);
  check(wave.pending.length === queuedBeforeAdd, 'summoned adds never become permanent wave replacements');

  // Placement is body-aware, even for the widest boss at a platform edge.
  for (const definition of data.ENEMIES) {
    const platform = dungeon.runtime.platforms.find(platform => platform.shape !== 'slope' && platform.index > 0);
    const enemy = dungeon.createEnemy(definition, { x: platform.x + platform.w - 1, platformIndex: platform.index });
    check(enemy.x >= platform.x && enemy.x + enemy.w <= platform.x + platform.w,
      `${definition.id}: right-edge spawn supports the whole body`);
  }
  console.log(`Project Starfall enemy spawning passed: ${assertions} assertions, ${data.MAPS.length} maps, ${arrivals} arrivals, ${actors} actors.`);
} finally {
  Math.random = originalRandom;
  Date.now = originalNow;
}
