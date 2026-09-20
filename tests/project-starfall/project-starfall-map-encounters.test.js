'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { createMapRuntime, findPlatformRouteLink, isPlatformJumpLinkTraversable } = require('../../js/games/project-starfall/engine/map-runtime.js');
const { getPlatformSurfaceY } = require('../../js/games/project-starfall/core/geometry.js');

const originalNow = Date.now;
const originalRandom = Math.random;
let clock = 1_800_000_000_000;
let assertions = 0;
function check(condition, message) { assertions += 1; assert(condition, message); }
function runtimeEngine(mapId = 'greenrootMeadow') {
  const engine = createProjectStarfallEngine(null, data);
  engine.state.player.classId = 'fighter';
  engine.changeMap(mapId, { silent: true });
  engine.effects = [];
  return engine;
}
function actor(engine, x, policy) {
  const enemy = engine.createEnemy(data.ENEMIES.find(entry => entry.id === 'lavaTick'), { x, platformIndex: 0,
    spawnActorTraversal: policy, spawnGroupLeash: 700 });
  Object.assign(enemy, { hp: 10000, maxHp: 10000, level: 1, attackCd: 10000, eliteAffixIds: [], speedScale: 1 });
  return enemy;
}

try {
  Date.now = () => clock;
  let seed = 0x51a7f411;
  Math.random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const audit = [];
  const connectorTraversalDiagnostics = [];
  const engine = runtimeEngine();
  for (const map of data.MAPS) {
    engine.changeMap(map.id, { silent: true });
    const runtime = engine.runtime;
    const unreachable = runtime.platforms.filter(platform => platform.index > 0 &&
      !findPlatformRouteLink(runtime.platformGraph, 0, platform.index));
    const noReturn = runtime.platforms.filter(platform => platform.index > 0 &&
      !findPlatformRouteLink(runtime.platformGraph, platform.index, 0));
    check(unreachable.length === 0, `${map.id}: every playable platform has an arrival route`);
    check(noReturn.length === 0, `${map.id}: every playable platform has a return route`);
    const ordinaryJumpPolicy = { allowLink: link => isPlatformJumpLinkTraversable(link, runtime.platforms,
      { jumpVelocity: 510, moveSpeed: 240, bodyWidth: 40, fps: 30 }) };
    const jumpLinks = runtime.platformGraph.flat().filter(link => link.type === 'jump');
    const ordinaryJumps = jumpLinks.filter(ordinaryJumpPolicy.allowLink);
    for (const link of ordinaryJumps) {
      for (const fps of [60, 120]) check(isPlatformJumpLinkTraversable(link, runtime.platforms,
        { jumpVelocity: 510, moveSpeed: 240, bodyWidth: 40, fps }),
      `${map.id}/${link.from}-${link.to}: conservative jump remains capable at ${fps}FPS`);
    }
    for (const platform of runtime.platforms.slice(1)) {
      check(findPlatformRouteLink(runtime.platformGraph, 0, platform.index, ordinaryJumpPolicy),
        `${map.id}/${platform.id}: ordinary actors can arrive without an impossible jump`);
      check(findPlatformRouteLink(runtime.platformGraph, platform.index, 0, ordinaryJumpPolicy),
        `${map.id}/${platform.id}: ordinary actors have a physically capable return route`);
    }
    for (const portal of runtime.portals) {
      check(runtime.platforms[portal.platformIndex], `${map.id}/${portal.id}: exit has a valid foothold`);
    }
    for (const climbable of runtime.climbables) {
      const centerX = climbable.x + climbable.w / 2;
      for (const endpoint of ['top', 'bottom']) {
        const platform = runtime.platforms[climbable[`${endpoint}PlatformIndex`]];
        check(platform && centerX >= platform.x && centerX <= platform.x + platform.w,
          `${map.id}/${climbable.id}: ${endpoint} graph endpoint has actual footing beneath it`);
      }
    }
    const correctedAlignmentIds = new Set([
      'rustcoilOutpost_stair_market_roofwalk', 'rustcoilOutpost_stair_gate_roof',
      'frostfenCamp_stair_market_roofwalk', 'frostfenCamp_stair_gate_roof',
      'cinderHollow_chain_11', 'banditRidgeCamp_rope_thrower_east_main'
    ]);
    for (const climbable of runtime.climbables.filter(entry => entry.id.includes('_perch_') || entry.id.endsWith('_beacon_regroup') || entry.id.startsWith('banditAnimationLab_rope_stand_') || correctedAlignmentIds.has(entry.id))) {
      const bottom = runtime.platforms[climbable.bottomPlatformIndex];
      const top = runtime.platforms[climbable.topPlatformIndex];
      const player = engine.state.player;
      check(bottom && top, `${map.id}/${climbable.id}: access link joins two real platforms`);
      for (const fps of [30, 60, 120]) {
        engine.placePlayerOnRuntimePlatform(bottom.index, climbable.x + climbable.w / 2 - player.w / 2);
        engine.setInput('up', true);
        let reached = false;
        for (let frame = 0; frame < fps * 6; frame += 1) {
          clock += 1000 / fps;
          engine.updatePlayer(1 / fps);
          if (player.grounded && player.groundedPlatformId === top.id) { reached = true; break; }
        }
        engine.setInput('up', false);
        check(reached, `${map.id}/${climbable.id}: real player input reaches the added access at ${fps}FPS`);
      }
    }
    // Added bridges do not invalidate saved footholds. Exercise the runtime's
    // position restoration at every old platform, including sloping surfaces.
    for (const platform of runtime.platforms.filter(entry => !/_hunting_(bridge|ramp)_/.test(entry.id))) {
      const previous = { ...engine.state.player, x: platform.x + (platform.w - engine.state.player.w) / 2,
        y: getPlatformSurfaceY(platform, platform.x + platform.w / 2) - engine.state.player.h };
      engine.placePlayerAtChannelPosition(previous);
      const body = engine.state.player;
      const support = runtime.platforms[body.groundedPlatformIndex];
      check(support && Number.isFinite(body.x) && Number.isFinite(body.y), `${map.id}/${platform.id}: restored position is valid`);
      check(Math.abs(body.y + body.h - getPlatformSurfaceY(support, body.x + body.w / 2)) < 0.001,
        `${map.id}/${platform.id}: restored feet sit on terrain`);
    }
    const groups = map.spawnGroups || [];
    const huntingConnectors = runtime.platforms.filter(platform => /_hunting_(bridge|ramp)_/.test(platform.id));
    for (const connector of huntingConnectors) {
      check(map.platforms[connector.index].noRoutineSpawns === true &&
        !map.spawnPoints.some(point => point.platformIndex === connector.index) &&
        groups.every(group => !group.platformIds.includes(connector.id)),
      `${map.id}/${connector.id}: hunting connector preserves combat spawn footprints`);
    }
    if (map.trainingRoute) {
      const route = map.trainingRoute;
      check(JSON.stringify(runtime.trainingRoute.mainPlatformIds) === JSON.stringify(route.mainPlatformIds), `${map.id}: runtime exposes the authored ordinary circuit`);
      check(JSON.stringify(runtime.trainingRoute.optionalPlatformIds) === JSON.stringify(route.optionalPlatformIds), `${map.id}: runtime exposes the authored optional circuit`);
      check(runtime.platforms.some(platform => platform.id === route.regroupPlatformId), `${map.id}: regroup has actual footing`);
      check(groups.every(group => !group.platformIds.includes(route.regroupPlatformId)), `${map.id}: regroup has no routine spawn owner`);
      const mainRegroup = runtime.platforms.find(platform => platform.id === route.mainRegroupPlatformId);
      check(mainRegroup && route.mainRegroupX >= mainRegroup.x && route.mainRegroupX <= mainRegroup.x + mainRegroup.w,
        `${map.id}: ordinary loop has a real low-pressure arrival return`);
      check(groups.every(group => !group.platformIds.includes(route.mainRegroupPlatformId)), `${map.id}: arrival return has no routine spawn owner`);
      check(groups.some(group => group.id === route.optionalGroupId), `${map.id}: optional encounter is real`);
      for (const scope of ['main', 'optional']) {
        const ids = route[`${scope}PlatformIds`];
        check(Array.isArray(ids) && ids.length >= 3, `${map.id}/${scope}: circuit connects several combat pockets`);
        check(new Set(ids).size === ids.length, `${map.id}/${scope}: circuit does not inflate coverage with duplicate goals`);
        for (let index = 0; index < ids.length; index += 1) {
          const from = runtime.platforms.find(platform => platform.id === ids[index]);
          const to = runtime.platforms.find(platform => platform.id === ids[(index + 1) % ids.length]);
          check(from && to && groups.some(group => group.platformIds.includes(from.id)), `${map.id}/${scope}/${ids[index]}: goal is a real authored combat pocket`);
          check(findPlatformRouteLink(runtime.platformGraph, from.index, to.index, ordinaryJumpPolicy),
            `${map.id}/${scope}/${from.id}: next leg including loop closure is physically reachable`);
        }
      }
      const optionalGroup = groups.find(group => group.id === route.optionalGroupId);
      check(route.optionalPlatformIds.some(id => optionalGroup.platformIds.includes(id)), `${map.id}: optional circuit reaches its challenge encounter`);
      check(route.optionalPlatformIds.some(id => !route.mainPlatformIds.includes(id)), `${map.id}: challenge adds a distinct branch`);
      check(new Set(groups.map(group => JSON.stringify(group.enemyWeights))).size > 1, `${map.id}: encounter sections have distinct rosters`);
      check(groups.length >= 3, `${map.id}: connected loop contains several combat pockets`);
      check(groups.every(group => !group.actorTraversal.stayInTerritory), `${map.id}: mobile groups may use nearby connected lanes`);
    }
    audit.push({ mapId: map.id, name: map.name, role: map.layoutRole, levelRange: map.levelRange,
      route: map.designIntent && map.designIntent.routeSummary || (map.safeZone ? 'Safe town/service traversal' : 'Encounter arena'),
      platformCount: runtime.platforms.length, climbableCount: runtime.climbables.length,
      huntingConnectors: huntingConnectors.map(platform => ({ id: platform.id, x: platform.x, y: platform.y,
        y2: platform.y2, w: platform.w, shape: platform.shape || 'flat' })),
      connectedComponents: 1, unreachable: unreachable.map(entry => entry.id),
      jumpCapability: { rawGraphLinks: jumpLinks.length, ordinaryActorAllowed: ordinaryJumps.length,
        filteredForOrdinaryActor: jumpLinks.length - ordinaryJumps.length, allPlatformsReachable: true,
        basis: '510 jump velocity, 240 movement speed, 40px body, conservative30FPS integration with no snap tolerance' },
      noReturn: noReturn.map(entry => entry.id), exits: runtime.portals.map(portal => ({ id: portal.id, platformId: runtime.platforms[portal.platformIndex].id })),
      trainingRoute: map.trainingRoute || null,
      groups: groups.map(group => ({ id: group.id, label: group.label, platformIds: group.platformIds,
        population: group.population, threeMemberPopulation: group.partyScaling === 'section-count'
          ? Math.min(group.maxPopulation, group.population + group.partyBonusPerMember * 2) : group.population,
        respawnSeconds: group.respawnSeconds, enemyWeights: group.enemyWeights, enemyMaxAlive: group.enemyMaxAlive,
        traversal: group.actorTraversal, leash: group.leash })) });
  }

  for (const [mapId, population, respawnSeconds, leash] of [
    ['greenrootMeadow', 2, 7, 420], ['cinderHollow', 7, 6, 560], ['banditRidgeCamp', 10, 5, 420]
  ]) {
    const map = data.MAPS.find(entry => entry.id === mapId);
    const group = map.spawnGroups.find(entry => entry.id === map.trainingRoute.optionalGroupId);
    const totalWeight = group.enemyWeights.reduce((total, entry) => total + entry.weight, 0);
    const supportedWeight = group.enemyWeights.reduce((total, entry) => {
      const enemy = data.ENEMIES.find(candidate => candidate.id === entry.enemyId);
      return total + (enemy.levelRange[1] >= map.levelRange[1] - 2 ? entry.weight : 0);
    }, 0);
    check(supportedWeight / totalWeight >= 0.75, `${mapId}: optional branch does not become mostly underleveled at the top of its unscaled map range`);
    check(group.population === population && group.respawnSeconds === respawnSeconds && group.leash === leash,
      `${mapId}: optional roster tuning preserves population, cadence and pursuit bounds`);
  }
  check(data.MAPS.find(map => map.id === 'cinderHollow').spawnGroups.find(group => group.id === 'cinderHollow_flyer_turns').label === 'Ember Crossfire',
    'Cinder mixed projectile/skirmisher branch uses its descriptive label while preserving the original group identity');
  const meadow = data.MAPS.find(map => map.id === 'greenrootMeadow');
  const meadowBasin = meadow.spawnGroups.find(group => group.id === 'greenrootMeadow_glass_basin');
  check(meadow.spawnGroups.reduce((sum, group) => sum + group.population, 0) === 11 &&
    meadow.spawnGroups.find(group => group.id === 'greenrootMeadow_arrival_shelf').population === 2 &&
    meadowBasin.population === 4 && meadowBasin.respawnSeconds === 6 && meadowBasin.leash === 460,
  'Meadow adds one basin enemy while preserving the quiet arrival and replacement cadence');
  const basinPoints = meadow.spawnPoints.filter(point => point.sectionId === meadowBasin.sectionId);
  check(basinPoints.length === 2 && basinPoints.every(point => point.weight === 1) &&
    basinPoints.some(point => point.id === 'spawn_3' && point.x === 1620) &&
    basinPoints.some(point => point.id === 'spawn_4' && point.x === 920),
  'Meadow basin shares spawns equally between its unchanged lower and upper points');
  const quarryGroups = data.MAPS.find(map => map.id === 'orebackQuarry').spawnGroups;
  const cartGroup = quarryGroups.find(group => group.id === 'orebackQuarry_ore_cart_lane');
  const supportGroup = quarryGroups.find(group => group.id === 'orebackQuarry_mushroom_pocket');
  check(JSON.stringify(cartGroup.enemyWeights) === JSON.stringify([{ enemyId: 'orebackBeetle', weight: 8 }, { enemyId: 'scrapWarden', weight: 2 }]) &&
    cartGroup.population === 8 && cartGroup.respawnSeconds === 5,
  'Quarry cart roster uses level-appropriate beetles and wardens without more spawn slots');
  check(JSON.stringify(supportGroup.enemyWeights) === JSON.stringify([{ enemyId: 'glowcapHealer', weight: 2 }, { enemyId: 'orebackBeetle', weight: 8 }]) &&
    supportGroup.enemyMaxAlive.glowcapHealer === 1 && supportGroup.population === 7 && supportGroup.respawnSeconds === 6,
  'Quarry support replaces underleveled ratchets while retaining one healer and the authored cadence');

  for (const classId of ['fighter', 'mage', 'archer']) {
    for (const fps of [30, 60, 120]) {
      const jumpEngine = runtimeEngine();
      jumpEngine.chooseClass(classId);
      jumpEngine.runtime.platforms = [
        { id: 'jump_start', index: 0, x: 0, y: 520, w: 1200, h: 30 },
        { id: 'jump_landing', index: 1, x: 300, y: 450, w: 600, h: 22, dropThrough: true }
      ];
      const jump = { type: 'jump', from: 0, to: 1, exitX: 600, entryX: 600 };
      const stats = jumpEngine.getStats();
      const options = { jumpVelocity: stats.jump, moveSpeed: stats.speed, bodyWidth: 40, fps: 30 };
      check(isPlatformJumpLinkTraversable(jump, jumpEngine.runtime.platforms, options), 'ordinary70px jump is permitted');
      jumpEngine.placePlayerOnRuntimePlatform(0, 580);
      jumpEngine.setInput('jump', true);
      let landed = false;
      for (let frame = 0; frame < fps * 2; frame += 1) {
        clock += 1000 / fps;
        jumpEngine.updatePlayer(1 / fps);
        jumpEngine.setInput('jump', false);
        if (jumpEngine.state.player.groundedPlatformId === 'jump_landing') { landed = true; break; }
      }
      check(landed, `${classId}/${fps}FPS: a permitted ordinary jump lands using real input`);
      jumpEngine.runtime.platforms[1].y = 430;
      check(!isPlatformJumpLinkTraversable(jump, jumpEngine.runtime.platforms, options),
        `${classId}/${fps}FPS: marginal90px jump is filtered consistently for all supported rates`);
    }
  }

  for (const fps of [30, 60, 120]) {
    const jumpEngine = runtimeEngine();
    const platforms = [
      { id: 'actor_jump_start', index: 0, x: 0, y: 520, w: 1200, h: 30 },
      { id: 'actor_jump_landing', index: 1, x: 300, y: 420, w: 600, h: 22, dropThrough: true }
    ];
    jumpEngine.runtime.platforms = platforms;
    jumpEngine.runtime.platformGraph = [[{ from: 0, to: 1, type: 'jump', exitX: 600, entryX: 600 }], []];
    const enemy = actor(jumpEngine, 600, { allowRamps: true, allowLadders: false, stayInTerritory: false });
    Object.assign(enemy, { x: 600 - enemy.w / 2, y: 520 - enemy.h, grounded: true,
      groundedPlatformIndex: 0, groundedPlatformId: platforms[0].id, spawnPlatformId: platforms[0].id,
      spawnPlatformIndex: 0, spawnX: 600 - enemy.w / 2, spawnY: 520 - enemy.h, wanderLeash: 700 });
    jumpEngine.enemies = [enemy];
    jumpEngine.placePlayerOnRuntimePlatform(1, 580);
    check(jumpEngine.canEnemyTraverseJumpLink(enemy, jumpEngine.runtime.platformGraph[0][0]), 'enemy can perform100px jump');
    check(jumpEngine.setEnemyAggro(enemy, jumpEngine.getCombatCharacterByTarget('player', 'player'), 'test', 60), 'jump target is in leash');
    let enemyLanded = false;
    for (let frame = 0; frame < fps * 3; frame += 1) {
      clock += 1000 / fps;
      jumpEngine.updateEnemies(1 / fps);
      if (enemy.groundedPlatformId === platforms[1].id) { enemyLanded = true; break; }
    }
    check(enemyLanded, `${fps}FPS: real enemy chase lands its permitted jump without overshooting`);
    const companion = { x: 580, y: 450, w: 40, h: 70, vx: 0, vy: 0, grounded: true,
      groundedPlatformId: platforms[0].id, groundedPlatformIndex: 0, facing: 1 };
    let companionLanded = false;
    for (let frame = 0; frame < fps * 3; frame += 1) {
      clock += 1000 / fps;
      jumpEngine.moveGroundBodyToward(companion, { x: 580, y: 350, w: 40, h: 70, platformId: platforms[1].id }, 220, 1 / fps);
      if (companion.groundedPlatformId === platforms[1].id) { companionLanded = true; break; }
    }
    check(companionLanded, `${fps}FPS: real companion movement lands its permitted jump`);
    platforms[1].y = 392;
    check(!jumpEngine.canEnemyTraverseJumpLink(enemy, jumpEngine.runtime.platformGraph[0][0]),
      `${fps}FPS:128px jump is excluded for enemies instead of repeated impossible attempts`);
  }

  for (const mapId of ['glacierSpine', 'astralArchive', 'eclipseFrontier', 'cinderHollow', 'orebackQuarry', 'ashglassPass', 'stormbreakCliffs']) {
    for (const fps of [30, 60, 120]) {
      for (const direction of [-1, 1]) {
        const bridgeEngine = runtimeEngine(mapId);
        for (const bridge of bridgeEngine.runtime.platforms.filter(entry => entry.id.includes('_hunting_bridge_'))) {
          const player = bridgeEngine.state.player;
          const start = direction > 0 ? bridge.x - 30 : bridge.x + bridge.w + 30;
          const goal = direction > 0 ? bridge.x + bridge.w + 24 : bridge.x - 24;
          const support = bridgeEngine.runtime.platforms.find(entry => entry !== bridge && entry.y === bridge.y &&
            start >= entry.x && start <= entry.x + entry.w);
          check(support, `${bridge.id}: original lane touches the bridge entrance`);
          bridgeEngine.placePlayerOnRuntimePlatform(support.index, start - player.w / 2);
          bridgeEngine.setInput(direction > 0 ? 'right' : 'left', true);
          let crossed = false;
          for (let frame = 0; frame < fps * 12; frame += 1) {
            clock += 1000 / fps;
            bridgeEngine.updatePlayer(1 / fps);
            check(player.grounded && Math.abs(player.y + player.h - bridge.y) < 0.01,
              `${bridge.id}/${fps}Hz: crossing the bridge preserves continuous footing`);
            if ((player.x + player.w / 2 - goal) * direction >= 0) { crossed = true; break; }
          }
          bridgeEngine.setInput(direction > 0 ? 'right' : 'left', false);
          check(crossed, `${bridge.id}/${fps}Hz: player traverses both bridge seams`);
          connectorTraversalDiagnostics.push({ mapId, platformId: bridge.id, fps, direction, crossed, continuousGrounding: true });
        }
      }
    }
  }

  // Cinder's raised regroup shelf needs a shallow ramp, not a second deck just
  // 40px beneath it. Descending player motion follows gravity, so unlike flat
  // bridges it can briefly be airborne; retain that diagnostic explicitly.
  for (const fps of [30, 60, 120]) {
    for (const direction of [-1, 1]) {
      const rampEngine = runtimeEngine('cinderHollow');
      const ramp = rampEngine.runtime.platforms.find(platform => platform.id === 'cinder_hollow_hunting_ramp_1');
      const player = rampEngine.state.player;
      const support = rampEngine.runtime.platforms.find(platform => platform.id ===
        (direction > 0 ? 'cinder_hollow_solid_lane_01' : 'cinder_hollow_solid_lane_10'));
      const destinationId = direction > 0 ? 'cinder_hollow_solid_lane_10' : 'cinder_hollow_solid_lane_01';
      const start = direction > 0 ? ramp.x - 35 : ramp.x + ramp.w + 35;
      const goal = direction > 0 ? ramp.x + ramp.w + 35 : ramp.x - 35;
      rampEngine.placePlayerOnRuntimePlatform(support.index, start - player.w / 2);
      rampEngine.setInput(direction > 0 ? 'right' : 'left', true);
      let crossed = false;
      let airborneFrames = 0;
      let maximumSurfaceGap = 0;
      for (let frame = 0; frame < fps * 6; frame += 1) {
        const previousX = player.x;
        clock += 1000 / fps;
        rampEngine.updatePlayer(1 / fps);
        check((player.x - previousX) * direction >= -0.001, 'shallow ramp input must not reverse direction');
        const center = player.x + player.w / 2;
        if (!player.grounded) airborneFrames += 1;
        if (center > ramp.x + 20 && center < ramp.x + ramp.w - 20) {
          maximumSurfaceGap = Math.max(maximumSurfaceGap, Math.abs(player.y + player.h - getPlatformSurfaceY(ramp, center)));
        }
        if ((center - goal) * direction >= 0) { crossed = true; break; }
      }
      check(crossed && player.grounded && player.groundedPlatformId === destinationId,
        `${fps}FPS: shallow Cinder connection reaches the original destination in both directions`);
      check(maximumSurfaceGap < 5, `${fps}FPS: shallow connection stays close to its visible surface without falling through`);
      connectorTraversalDiagnostics.push({ mapId: 'cinderHollow', platformId: ramp.id, fps, direction,
        crossed, airborneFrames, maximumSurfaceGap: Math.round(maximumSurfaceGap * 1000) / 1000,
        note: 'Player gravity can produce intermittent airborne flags while descending; this is not a continuous-grounding assertion.' });
    }
  }

  // Policy changes must not reuse a route cached for a more permissive actor.
  const policyEngine = runtimeEngine();
  policyEngine.runtime.platforms = [
    { id: 'ground', index: 0, x: 0, y: 600, w: 1200, h: 24, shape: 'flat' },
    { id: 'upper', index: 1, x: 500, y: 420, w: 500, h: 24, shape: 'flat' },
    { id: 'ramp', index: 2, x: 300, y: 600, y2: 420, w: 300, h: 24, shape: 'slope' }
  ];
  policyEngine.runtime.climbables = [{ id: 'test_ladder', x: 550, y: 420, w: 30, h: 180 }];
  const [ground, upper, ramp] = policyEngine.runtime.platforms;
  const ladder = { from: 0, to: 1, type: 'ladder-up', exitX: 565, entryX: 565, climbableId: 'test_ladder' };
  policyEngine.runtime.platformGraph = [[ladder], [{ from: 1, to: 0, type: 'ladder-down', exitX: 565, entryX: 565, climbableId: 'test_ladder' }], []];
  const climber = actor(policyEngine, 560, { allowLadders: true, allowRamps: true, stayInTerritory: false });
  const walker = actor(policyEngine, 560, { allowLadders: false, allowRamps: true, stayInTerritory: false });
  check(policyEngine.findEnemyPlatformLink(ground, upper, climber) === ladder, 'permitted ladder route is found');
  check(policyEngine.findEnemyPlatformLink(ground, upper, walker) === null, 'ladder exclusion does not reuse climber cache');
  policyEngine.updateEnemyPlatformJump(walker, { x: 565, y: 420, w: 40, h: 74 }, 1, clock / 1000, upper);
  check(!walker.climbing && walker.grounded && walker.vx === 0, 'no proximity jump bypasses a denied route');
  climber.x = 565 - climber.w / 2;
  policyEngine.updateEnemyPlatformJump(climber, { x: 565, y: 420, w: 40, h: 74 }, 1, clock / 1000, upper);
  check(climber.climbing, 'permitted climber enters the real climbing state');
  for (let frame = 0; frame < 180; frame += 1) policyEngine.updateEnemyClimbing(climber, 1 / 60, 1);
  check(climber.grounded && climber.groundedPlatformId === upper.id, 'climber arrives on the destination support');
  policyEngine.runtime.platformGraph = [
    [{ from: 0, to: 2, type: 'ramp-up', exitX: 300, entryX: 300 }], [],
    [{ from: 2, to: 1, type: 'ramp-up', exitX: 600, entryX: 600 }]
  ];
  const noRamp = actor(policyEngine, 400, { allowLadders: false, allowRamps: false, stayInTerritory: false });
  check(policyEngine.findEnemyPlatformLink(ground, upper, walker), 'permitted ramp provides nearby lane pursuit');
  check(!policyEngine.findEnemyPlatformLink(ground, upper, noRamp), 'forbidden ramp is filtered, including generic slope links');
  policyEngine.runtime.spawnGroups = [{ id: 'strict', platformIds: [ground.id] }];
  const strict = actor(policyEngine, 400, { allowLadders: true, allowRamps: true, stayInTerritory: true });
  strict.spawnGroupId = 'strict';
  check(!policyEngine.findEnemyPlatformLink(ground, upper, strict), 'strict territory cannot enter an unrelated broad lane');

  for (const fps of [30, 60, 120]) {
    const returnEngine = runtimeEngine();
    const enemy = actor(returnEngine, 1500, { allowLadders: false, allowRamps: true, stayInTerritory: false });
    enemy.wanderLeash = 240;
    returnEngine.enemies = [enemy];
    const player = returnEngine.state.player;
    Object.assign(player, { hp: 100000, level: 200, x: enemy.spawnX + 150, y: enemy.y, grounded: true,
      groundedPlatformId: enemy.spawnPlatformId, groundedPlatformIndex: enemy.spawnPlatformIndex });
    check(returnEngine.setEnemyAggro(enemy, returnEngine.getCombatCharacterByTarget('player', 'player'), 'test', 60), 'inside-leash target is acquired');
    enemy.x = enemy.spawnX + 300;
    player.x = enemy.spawnX + 520;
    check(!returnEngine.getEnemyAggroTarget(enemy, clock / 1000) && enemy.leashReturning, 'leash breach begins walk home');
    let returned = false;
    for (let frame = 0; frame < fps * 8; frame += 1) {
      clock += 1000 / fps;
      player.x = enemy.x + 20;
      check(!returnEngine.setEnemyAggro(enemy, returnEngine.getCombatCharacterByTarget('player', 'player'), 'contact', 60),
        'contact cannot repeatedly reacquire an enemy walking home');
      const before = enemy.x;
      returnEngine.updateEnemies(1 / fps);
      check(Math.abs(enemy.x - before) <= enemy.data.speed / fps + 0.01, `${fps}Hz return movement does not snap`);
      if (!enemy.leashReturning) { returned = true; break; }
    }
    check(returned && Math.abs(enemy.x - enemy.spawnX) <= 50, `${fps}Hz enemy reaches home before reacquisition`);
    clock += 1100;
    check(returnEngine.setEnemyAggro(enemy, returnEngine.getCombatCharacterByTarget('player', 'player'), 'combat', 60), 'settled enemy can be engaged again');
  }

  const quarry = runtimeEngine('orebackQuarry');
  const group = quarry.getRuntimeSpawnGroups().find(entry => entry.id.endsWith('mushroom_pocket'));
  quarry.enemies = [];
  Math.random = () => 0;
  for (let index = 0; index < 80; index += 1) {
    const id = quarry.getSpawnGroupEnemyId(group, index);
    quarry.enemies.push({ id, hp: 100, spawnGroupId: group.id });
    check(quarry.enemies.filter(entry => entry.id === 'glowcapHealer' && entry.hp > 0).length <= 1,
      'support cap holds under adversarial random selection and replacement batches');
    if (index % 7 === 0) quarry.enemies[quarry.enemies.length - 1].hp = 0;
  }
  const glacier = runtimeEngine('glacierSpine');
  const soloPopulation = glacier.getRuntimeSpawnGroups().reduce((sum, entry) => sum + glacier.getSpawnGroupPopulationTarget(entry), 0);
  glacier.getActivePrototypePartyMembers = () => [{ hp: 100 }, { hp: 100 }];
  const partyPopulation = glacier.getRuntimeSpawnGroups().reduce((sum, entry) => sum + glacier.getSpawnGroupPopulationTarget(entry), 0);
  check(soloPopulation === 32 && partyPopulation === 40, 'Glacier adds eight party slots across three circuits');
  const support = quarry.createEnemy(data.ENEMIES.find(entry => entry.id === 'glowcapHealer'), { x: 1000, platformIndex: 0 });
  const otherSupport = quarry.createEnemy(data.ENEMIES.find(entry => entry.id === 'glowcapHealer'), { x: 1040, platformIndex: 0 });
  const guard = quarry.createEnemy(data.ENEMIES.find(entry => entry.id === 'orebackBeetle'), { x: 1080, platformIndex: 0 });
  const lowerGuard = quarry.createEnemy(data.ENEMIES.find(entry => entry.id === 'orebackBeetle'), { x: 1080, platformIndex: 0 });
  for (const recipient of [otherSupport, guard, lowerGuard]) recipient.hp = Math.floor(recipient.maxHp * 0.5);
  lowerGuard.y += 320;
  quarry.enemies = [support, otherSupport, guard, lowerGuard];
  check(quarry.healNearby(support), 'injured frontline starts the visible support preparation');
  check(support.pendingAttack.recipients.length === 1 && support.pendingAttack.recipients[0] === guard,
    'recipients exclude other healers and unrelated floors outside the two-dimensional radius');
  const guardBefore = guard.hp;
  quarry.resolveEnemyPendingAttack(support, []);
  check(guard.hp === guardBefore, 'support cannot restore HP during preparation');
  support.telegraph = 0;
  quarry.resolveEnemyPendingAttack(support, []);
  check(guard.hp > guardBefore && otherSupport.hp < otherSupport.maxHp && lowerGuard.hp < lowerGuard.maxHp,
    'the restorative pulse heals only the declared nearby frontline recipient');
  if (process.argv.includes('--report')) {
    const folder = path.join(process.env.TEMP || process.cwd(), 'starfall-encounters');
    fs.mkdirSync(folder, { recursive: true });
    fs.writeFileSync(path.join(folder, 'map-encounter-audit.json'), JSON.stringify({
      note: 'Structural route and runtime placement audit; visual quality and measured training are separate evidence.',
      assertions, maps: audit, connectorTraversalDiagnostics,
      totals: { mapCount: audit.length, platformCount: audit.reduce((sum, map) => sum + map.platformCount, 0),
        climbableCount: audit.reduce((sum, map) => sum + map.climbableCount, 0),
        huntingBridgeCount: audit.reduce((sum, map) => sum + map.huntingConnectors.filter(platform => platform.shape !== 'slope').length, 0),
        huntingRampCount: audit.reduce((sum, map) => sum + map.huntingConnectors.filter(platform => platform.shape === 'slope').length, 0),
        actualConnectorTraversals: connectorTraversalDiagnostics.length }
    }, null, 2));
  }
  console.log(`Project Starfall map encounters: ${assertions} assertions, 56 connected map graphs, 13 authored field loops, permissions, leash returns, support caps and saved-position recovery passed.`);
} finally {
  Date.now = originalNow;
  Math.random = originalRandom;
}
