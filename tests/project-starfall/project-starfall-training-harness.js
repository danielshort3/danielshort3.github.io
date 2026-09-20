'use strict';

// This harness drives the actual runtime through player inputs. It never moves
// actors, removes enemies, awards damage/loot, or replaces combat calculations.
const skills = require('../../js/games/project-starfall/engine/skills.js');
const movement = require('../../js/games/project-starfall/engine/movement.js');
const { isPlatformJumpLinkTraversable } = require('../../js/games/project-starfall/engine/map-runtime.js');
const { ALL_CLASS_IDS, CLASS_ROTATIONS, ADVANCED_BASE_FILLERS, getMapSpawnPopulation } = require('./project-starfall-balance-harness.js');

const TRAINING_SEEDS = Object.freeze([0x51a7f411, 0x1c0ffee, 0x6d2b79f5]);
const TRAINING_PROTOCOL = Object.freeze({ warmupSeconds: 60, measuredSeconds: 300, fps: 30, seeds: TRAINING_SEEDS });
const ROUND = (value) => Math.round(value * 1000) / 1000;
const BASE_TIME = Date.UTC(2026, 8, 19, 12);

function getPublicTrainingMaps(data) {
  return data.MAPS.filter((map) => !map.safeZone && !map.adminOnly && !map.isDungeon && Array.isArray(map.spawnGroups) && map.spawnGroups.length);
}

function getEligibleTrainingClasses(data, level) {
  return ALL_CLASS_IDS.filter((id) => !data.ADVANCED_CLASSES[id] || level >= data.ADVANCED_CLASSES[id].levelRequirement);
}

function getTrainingCohorts(data) {
  const maps = getPublicTrainingMaps(data);
  const cohorts = [];
  const included = new Set();
  // Pairwise overlap creates honest same-level comparisons, not comparisons of
  // each map's unrelated midpoint. Identical cohorts are deduplicated.
  for (let a = 0; a < maps.length; a += 1) {
    for (let b = a + 1; b < maps.length; b += 1) {
      const low = Math.max(maps[a].levelRange[0], maps[b].levelRange[0]);
      const high = Math.min(maps[a].levelRange[1], maps[b].levelRange[1]);
      if (low > high) continue;
      const level = Math.round((low + high) / 2);
      const mapIds = maps.filter((map) => map.levelRange[0] <= level && map.levelRange[1] >= level).map((map) => map.id);
      const key = `${level}:${mapIds.join(',')}`;
      if (included.has(key)) continue;
      included.add(key);
      cohorts.push({ id: `level-${level}`, level, mapIds });
    }
  }
  maps.filter((map) => !cohorts.some((cohort) => cohort.mapIds.includes(map.id))).forEach((map) => {
    cohorts.push({ id: `${map.id}-standalone`, level: Math.round((map.levelRange[0] + map.levelRange[1]) / 2), mapIds: [map.id] });
  });
  return cohorts.sort((a, b) => a.level - b.level);
}

function prepareTrainingPlayer(data, engine, classId, level) {
  const baseId = data.ADVANCED_CLASSES[classId] ? data.ADVANCED_CLASSES[classId].baseClass : classId;
  if (!data.BASE_CLASSES[baseId] || !getEligibleTrainingClasses(data, level).includes(classId)) throw new Error(`Ineligible class ${classId} at level ${level}`);
  engine.chooseClass(baseId);
  const player = engine.state.player;
  player.level = level;
  player.advancedClassId = classId === baseId ? '' : classId;
  // Benchmark setup grants only the normal level-earned budget. rankSkill still
  // enforces prerequisite and per-skill caps; no manuals or quest SP are granted.
  const budget = skills.getSkillPointBudget(player, player.advancedClassId, { data });
  player.baseSkillPoints = budget.base;
  player.advancedSkillPoints = budget.advanced;
  player.skillPoints = budget.base + budget.advanced;
  engine.state.skills = Object.assign({}, skills.createDefaultRanks(baseId, { data }), player.advancedClassId ? skills.createDefaultRanks(classId, { data }) : {});
  const accessible = data.SKILLS.filter((skill) => skill.owner === baseId || skill.owner === classId);
  for (let round = 0; round < 30; round += 1) {
    let changed = false;
    accessible.forEach((skill) => { if (engine.rankSkill(skill.id)) changed = true; });
    if (!changed) break;
  }
  // Equal shop budget at each level, identical selection rule for each class.
  // Use real, unupgraded items; never manufacture stats or rarity bonuses.
  const equipmentBudget = Math.max(0, level - 1) * 160;
  let remaining = equipmentBudget;
  const equipment = {};
  const slots = ['weapon', 'chest', 'head', 'boots', 'gloves', 'ring', 'offhand'];
  slots.forEach((slot) => {
    const item = data.SHOP_ITEMS.filter((candidate) => candidate.slot === slot && candidate.level <= level && Number(candidate.cost || 0) <= remaining && ['any', baseId, classId].includes(candidate.classId))
      .sort((a, b) => b.level - a.level || Number(b.cost || 0) - Number(a.cost || 0) || a.id.localeCompare(b.id))[0];
    if (!item) return;
    equipment[slot] = Object.assign({}, JSON.parse(JSON.stringify(item)), { uid: `training-${item.id}`, upgradeLevel: 0 });
    remaining -= Number(item.cost || 0);
  });
  engine.state.equipment = equipment;
  engine.state.inventory = [];
  engine.state.adminSettings = { xpRate: 1, dropRate: 1 };
  engine.state.pet = { enabled: false };
  const tier = level >= 70 ? 'superior' : level >= 40 ? 'greater' : level >= 15 ? 'standard' : 'minor';
  const potionIds = [`${tier}_health_potion`, `${tier}_resource_tonic`];
  engine.state.consumables = Object.fromEntries(potionIds.map((id) => [id, 99]));
  engine.invalidateStatsCache();
  const stats = engine.getStats();
  player.hp = stats.maxHp;
  player.mp = stats.maxMp;
  player.resource = stats.secondaryResourceMax;
  player.xp = 0;
  // Level and loadout are experimental controls. XP still goes through the
  // production award and telemetry paths; disable only level-up conversion.
  engine.checkLevelUp = () => {};
  // Mastery milestones grant permanent combat stats. Freeze their progression
  // along with level; otherwise faster maps change the benchmark's loadout.
  engine.addClassMasteryXp = () => 0;
  const rotationIds = [...(CLASS_ROTATIONS[classId] || []), ...(ADVANCED_BASE_FILLERS[classId] || [])];
  const resourceCost = (skill) => engine.getRuntimeSkillResourceCost(skill, engine.state.skills[skill.id], engine.getSkillModifierForSkill(skill), stats);
  const attackMpReserve = Math.max(0, ...accessible.filter((skill) => rotationIds.includes(skill.id) && engine.state.skills[skill.id] > 0).map(resourceCost));
  return { baseId, equipmentBudget, equipmentSpent: equipmentBudget - remaining, equipmentIds: Object.values(equipment).map((item) => item.id), skills: { ...engine.state.skills }, skillBudget: budget, potionIds, attackMpReserve,
    mobilitySkills: accessible.filter((skill) => skill.movementEffect && skill.movementEffect.mode !== 'leap' && engine.state.skills[skill.id] > 0).map((skill) => ({ id: skill.id, movementEffect: skill.movementEffect, mpCost: resourceCost(skill) })) };
}

function useTrainingConsumables(engine, loadout) {
  const stats = engine.getStats();
  const player = engine.state.player;
  if (player.hp < stats.maxHp * 0.5) engine.useConsumable(loadout.potionIds[0], { silent: true });
  // Every class spends MP on skills; Momentum/Focus are separate meters.
  if (player.mp < stats.maxMp * 0.25) engine.useConsumable(loadout.potionIds[1], { silent: true });
}

function assessTrainingLoot(engine, admission, item) {
  const quantity = Math.max(1, Number(admission.quantity || 1));
  const equipment = admission.kind === 'equipment';
  return { equipmentCollected: equipment ? quantity : 0, equippableEquipment: equipment && engine.canUseItem(item) ? quantity : 0,
    potentialEquipmentResale: equipment ? engine.getItemSellValue(item) * quantity : 0,
    materialUnits: admission.kind === 'material' ? quantity : 0, consumableUnits: admission.kind === 'consumable' ? quantity : 0,
    cardUnits: admission.kind === 'card' ? quantity : 0 };
}

function getSurfaceY(platform, x) {
  if (platform.shape !== 'slope') return platform.y;
  return platform.y + (platform.y2 - platform.y) * Math.max(0, Math.min(1, (x - platform.x) / platform.w));
}

function getTrainingLinkLandingX(link, platforms, bodyWidth) {
  const next = platforms[link.to];
  const inset = Math.min(next.w / 2, bodyWidth / 2 + 2);
  return Math.max(next.x + inset, Math.min(next.x + next.w - inset, link.entryX));
}

function getTrainingRampDirection(link, from, to) {
  const ramp = from.shape === 'slope' ? from : to.shape === 'slope' ? to : null;
  return ramp && link.type.startsWith('ramp') ? Math.sign(ramp.y2 - ramp.y) * (link.type === 'ramp-up' ? -1 : 1) : 0;
}

function isTrainingDropTraversable(link, platforms, stats) {
  if (link.type !== 'drop') return true;
  const launchX = link.exitX;
  const landingX = getTrainingLinkLandingX(link, platforms, stats.bodyWidth);
  const fall = getSurfaceY(platforms[link.to], landingX) - getSurfaceY(platforms[link.from], launchX);
  if (fall <= 0) return false;
  let distance = 0;
  let velocity = 0;
  let duration = 0;
  while (distance < fall && duration < 3) { velocity += 1600 / 30; distance += velocity / 30; duration += 1 / 30; }
  return Math.abs(landingX - launchX) <= Math.max(0, duration - 0.1) * stats.speed;
}

function prepareTrainingDropLink(link, platforms, climbables, stats) {
  if (link.type !== 'drop') return link;
  const source = platforms[link.from];
  const half = stats.bodyWidth / 2;
  const candidates = [0, -90, 90, -150, 150, -220, 220].map((offset) => Math.max(source.x + half + 2, Math.min(source.x + source.w - half - 2, link.exitX + offset)));
  for (const exitX of candidates) {
    const sourceY = getSurfaceY(source, exitX);
    const overlapsLadder = climbables.some((climbable) => sourceY >= climbable.y - 20 && sourceY <= climbable.y + climbable.h + 20 && Math.abs(exitX - climbable.x - climbable.w / 2) < half + climbable.w / 2 + 40);
    if (overlapsLadder) continue;
    const adjusted = { ...link, exitX, entryX: getTrainingLinkLandingX({ ...link, entryX: exitX }, platforms, stats.bodyWidth) };
    if (isTrainingDropTraversable(adjusted, platforms, stats)) return adjusted;
  }
  const landingX = getTrainingLinkLandingX(link, platforms, stats.bodyWidth);
  // Jumping toward a lower, overlapping floor would land on the source again.
  // Across a real gap, normal jump input can supply a safe longer flight.
  if (landingX - half >= source.x + source.w || landingX + half <= source.x) return { ...link, type: 'jump', authoredType: 'drop' };
  return null;
}

function findTrainingRoute(graph, platforms, fromIndex, toIndex, startX, targetX, stats) {
  if (fromIndex < 0 || toIndex < 0) return { link: null, seconds: Infinity };
  const speed = Math.max(1, stats.speed);
  const walkTime = (platform, a, b) => Math.hypot(b - a, getSurfaceY(platform, b) - getSurfaceY(platform, a)) / speed;
  const queue = [{ index: fromIndex, x: startX, first: null, cost: 0 }];
  const bestCost = new Map();
  let result = { link: null, seconds: Infinity };
  while (queue.length) {
    queue.sort((a, b) => a.cost - b.cost);
    const current = queue.shift();
    if (current.cost >= result.seconds) break;
    const key = `${current.index}:${Math.round(current.x)}`;
    if (current.cost > (bestCost.get(key) ?? Infinity)) continue;
    if (current.index === toIndex) {
      const seconds = current.cost + walkTime(platforms[toIndex], current.x, targetX);
      if (seconds < result.seconds) result = { link: current.first, seconds };
    }
    for (const link of graph[current.index] || []) {
      const from = platforms[current.index];
      const next = platforms[link.to];
      let entryX = link.type === 'drop' || link.type === 'jump' ? getTrainingLinkLandingX(link, platforms, stats.bodyWidth) : link.entryX;
      if (link.type === 'walk' && from.shape !== 'slope' && next.shape !== 'slope') {
        const direction = Math.sign(next.x + next.w / 2 - from.x - from.w / 2) || 1;
        const handoff = direction > 0 ? from.x + from.w + stats.bodyWidth / 2 + 10 : from.x - stats.bodyWidth / 2 - 10;
        const inset = Math.min(next.w / 2, stats.bodyWidth / 2 + 10);
        entryX = Math.max(next.x + inset, Math.min(next.x + next.w - inset, handoff));
      }
      const vertical = Math.abs(getSurfaceY(next, entryX) - getSurfaceY(from, link.exitX));
      const transition = link.type.startsWith('ladder') ? vertical / 180 + 0.3 : link.type === 'jump' ? Math.max(0.65, Math.abs(entryX - link.exitX) / speed) : link.type === 'drop' ? Math.max(0.2, Math.sqrt(2 * vertical / 1600)) : Math.hypot(entryX - link.exitX, vertical) / speed;
      const cost = current.cost + walkTime(from, current.x, link.exitX) + transition;
      const nextKey = `${link.to}:${Math.round(entryX)}`;
      if (cost >= (bestCost.get(nextKey) ?? Infinity)) continue;
      bestCost.set(nextKey, cost);
      queue.push({ index: link.to, x: entryX, first: current.first || link, cost });
    }
  }
  return result;
}

function classifyTrainingPhase(options) {
  if (options.recovery) return 'recovery';
  if (options.combat) return 'combat';
  if (!options.hasRoute) return 'idle';
  if (!options.navigationInput && !options.moving && options.atCombatPocket && !options.localEnemyAlive && options.pendingLocalRespawn) return 'respawnWaiting';
  return 'travel';
}

function selectTrainingRoute(map, runtime, options) {
  const scope = options.routeScope || 'main';
  if (!['main', 'full', 'optional'].includes(scope)) throw new Error('Training route must be main, full or optional');
  const plan = options.routePlan || map.trainingRoute || {};
  const authored = scope === 'full' ? runtime.trainingRoute.routePlatformIds : plan[scope === 'main' ? 'mainPlatformIds' : 'optionalPlatformIds'];
  const fallback = !Array.isArray(authored) || !authored.length;
  const plannedPlatformIds = fallback ? [...runtime.trainingRoute.routePlatformIds] : [...authored];
  const missingPlatformIds = plannedPlatformIds.filter((id) => !runtime.platforms.some((platform) => platform.id === id));
  return { scope, fallback, plannedPlatformIds, missingPlatformIds,
    platformIds: plannedPlatformIds.filter((id) => !missingPlatformIds.includes(id)) };
}

function createTrainingController(engine, classId, loadout, portalGoal, selectedRoute) {
  const movementStats = engine.getStats();
  const routeStats = { speed: movementStats.speed, bodyWidth: engine.state.player.w };
  const jumpOptions = {
    jumpVelocity: movementStats.jump, moveSpeed: movementStats.speed, bodyWidth: engine.state.player.w, gravity: 1600, fps: 30
  };
  const graph = engine.runtime.platformGraph.map((links) => links.map((link) => prepareTrainingDropLink(link, engine.runtime.platforms, engine.runtime.climbables, routeStats))
    .filter((link) => link && isPlatformJumpLinkTraversable(link, engine.runtime.platforms, jumpOptions) && !(link.type === 'walk' && links.some((other) => other.to === link.to && other.type.startsWith('ramp')))));
  const portalPlatform = portalGoal && engine.runtime.platforms.find((platform) => platform.id === portalGoal.platformId);
  const route = portalPlatform ? [{ ...portalPlatform, x: portalGoal.x, w: portalGoal.w }] : (selectedRoute ? selectedRoute.platformIds : engine.runtime.trainingRoute.routePlatformIds).map((id) => engine.runtime.platforms.find((platform) => platform.id === id)).filter(Boolean);
  const rotation = [...(CLASS_ROTATIONS[classId] || []), ...(ADVANCED_BASE_FILLERS[classId] || [])];
  let routeIndex = 0;
  let arrivedAt = null;
  let nextDecision = 0;
  let nextJump = 0;
  let cachedTarget = null;
  let goal = route[0];
  let climbDestination = null;
  let climbCommand = null;
  let airTransfer = null;
  let passingEncounterUntil = 0;
  let passingEncounterUsed = false;
  const visited = new Set();
  let routeCycles = 0;
  let completedGoalsInOrder = 0;
  let routeAdvances = 0;
  let mobilityCasts = 0;
  let stuckSeconds = 0;
  let progressGoalId = '';
  let bestGoalDistance = Infinity;
  let lastProgressAt = 0;
  const routeCostCache = new Map();

  function physicalRoute(fromIndex, toIndex, startX, targetX) {
    const from = engine.runtime.platforms[fromIndex];
    const to = engine.runtime.platforms[toIndex];
    if (!from || !to) return { link: null, seconds: Infinity };
    const x = startX ?? from.x + from.w / 2;
    const goalX = targetX ?? to.x + to.w / 2;
    const key = `${fromIndex}:${toIndex}:${Math.round(x / 16)}:${Math.round(goalX / 16)}`;
    if (routeCostCache.has(key)) return routeCostCache.get(key);
    const result = findTrainingRoute(graph, engine.runtime.platforms, fromIndex, toIndex, x, goalX, routeStats);
    routeCostCache.set(key, result);
    return result;
  }

  function advanceRoute() {
    completedGoalsInOrder = arrivedAt !== null ? completedGoalsInOrder + 1 : 0;
    if (completedGoalsInOrder >= route.length) { routeCycles += 1; completedGoalsInOrder = 0; }
    routeIndex = (routeIndex + 1) % Math.max(1, route.length);
    routeAdvances += 1;
    arrivedAt = null;
    goal = route[routeIndex];
    passingEncounterUntil = 0;
    passingEncounterUsed = false;
  }

  function drive(elapsed, dt) {
    const player = engine.state.player;
    const input = engine.input;
    input.left = input.right = input.up = input.down = input.attack = input.jump = false;
    input.loot = true;
    if (!goal) return { phase: 'idle', target: null };
    const platform = engine.getBodyPlatform(player);
    if (airTransfer && (elapsed >= airTransfer.until || player.grounded && platform && platform.id === airTransfer.platform.id)) airTransfer = null;
    if (player.grounded && platform && route.includes(platform)) {
      visited.add(platform.id);
    }
    const live = engine.enemies.filter((enemy) => enemy.hp > 0);
    const px = player.x + player.w / 2;
    const py = player.y + player.h / 2;
    const melee = loadout.baseId === 'fighter';
    const range = melee ? 92 : classId === 'trapper' ? 205 : classId === 'runeMage' ? 255 : Math.max(255, Number(movementStats.range || 255));
    const shootable = (enemy) => Math.abs(enemy.x + enemy.w / 2 - px) <= range && Math.abs(enemy.y + enemy.h / 2 - py) <= (melee ? 90 : 65);
    if (elapsed >= nextDecision || cachedTarget && cachedTarget.hp <= 0) {
      nextDecision = elapsed + 0.15;
      const local = live.filter((enemy) => {
        if (Math.abs(enemy.x + enemy.w / 2 - px) >= 550 || Math.abs(enemy.y + enemy.h / 2 - py) >= 160) return false;
        if (shootable(enemy)) return true;
        const enemyPlatform = engine.getBodyPlatform(enemy);
        return platform && enemyPlatform && physicalRoute(platform.index, enemyPlatform.index, px, enemy.x + enemy.w / 2).seconds <= 6;
      });
      const score = (enemy) => Math.hypot(enemy.x - player.x, (enemy.y - player.y) * 1.8) - (enemy.data.behavior === 'healer' ? 70 : 0);
      local.sort((a, b) => Number(shootable(b)) - Number(shootable(a)) || score(a) - score(b));
      const candidate = local[0] || null;
      // Finish a reachable living target. In particular, never abandon an
      // immediately shootable enemy for a nearer enemy on a different floor.
      if (!local.includes(cachedTarget) || !player.climbing && candidate && !shootable(cachedTarget) && shootable(candidate)) cachedTarget = candidate;
    }
    // Keep the encounter time budget after knockback or pursuing a target off
    // its platform. Otherwise ranged actors can camp forever beside the goal.
    if (arrivedAt !== null && elapsed - arrivedAt > 18) advanceRoute();
    if (player.grounded && platform && platform.id === goal.id) {
      if (arrivedAt === null) arrivedAt = elapsed;
      if (!cachedTarget && elapsed - arrivedAt > 1) advanceRoute();
    }
    const target = cachedTarget;
    const dx = target ? target.x + target.w / 2 - px : 0;
    const dy = target ? target.y + target.h / 2 - py : 0;
    const engaging = !portalGoal && (arrivedAt !== null || platform && platform.id === goal.id);
    const withinWeaponReach = target && Math.abs(dx) <= range && Math.abs(dy) <= (melee ? 90 : 65);
    // Attack enemies encountered along the route instead of walking through
    // them. Bound each passing encounter so repeated nearby respawns cannot
    // indefinitely prevent reaching the next authored combat pocket.
    if (!portalGoal && !engaging && withinWeaponReach && !passingEncounterUsed && !airTransfer && !player.climbing) {
      passingEncounterUntil = elapsed + 4;
      passingEncounterUsed = true;
    }
    const passingEncounter = !portalGoal && !engaging && elapsed < passingEncounterUntil;
    const inRange = !airTransfer && (engaging || passingEncounter) && withinWeaponReach;
    let desiredX = goal.x + goal.w / 2;
    let desiredPlatform = goal;
    if (target && engaging) {
      desiredX = target.x + target.w / 2 - Math.sign(dx || 1) * (melee ? 60 : 180);
      desiredPlatform = engine.getBodyPlatform(target) || goal;
    }
    let moving = !inRange;
    if (inRange && !player.climbing) {
      player.facing = dx >= 0 ? 1 : -1;
      // Skills, costs, cooldowns, anticipation and target masks are production.
      let cast = false;
      for (const id of rotation) {
        if (id === 'fire_mage_heat_vent' && Math.hypot(Math.abs(dx) - 138, dy) > 150 + Number(engine.state.skills[id] || 0) * 3) continue;
        if (engine.state.skills[id] > 0 && engine.useSkill(id, { silent: true, suppressCombatEmit: true })) { cast = true; break; }
      }
      input.attack = !cast;
    } else if (airTransfer) {
      desiredX = airTransfer.x;
      // A graph drop can cross intervening one-way ramps. Continue the user's
      // down+jump input through those surfaces until the intended floor lands.
      if (airTransfer.type === 'drop' && player.grounded && elapsed >= nextJump) {
        input.down = input.jump = true;
        engine.queueJumpInput();
        nextJump = elapsed + 0.35;
      }
    } else if (platform && platform.id !== desiredPlatform.id) {
      const link = physicalRoute(platform.index, desiredPlatform.index, px, desiredX).link;
      if (link) {
        const next = engine.runtime.platforms[link.to];
        desiredX = link.exitX;
        if (link.type.startsWith('ladder')) {
          const climbable = engine.runtime.climbables.find((entry) => entry.id === link.climbableId);
          if (climbable) desiredX = climbable.x + climbable.w / 2;
          if (Math.abs(px - desiredX) < 18) {
            input.up = link.type === 'ladder-up';
            input.down = !input.up;
            moving = false;
            if (!player.climbing) {
              climbDestination = next;
              climbCommand = { id: link.climbableId, up: input.up, destination: next };
            }
          }
        } else if (link.type === 'jump' && Math.abs(px - desiredX) < 12 && player.grounded && elapsed >= nextJump) {
          input.jump = true;
          engine.queueJumpInput();
          nextJump = elapsed + 0.65;
          desiredX = Math.max(next.x + player.w / 2 + 2, Math.min(next.x + next.w - player.w / 2 - 2, link.entryX));
          airTransfer = { type: 'jump', platform: next, x: desiredX, until: elapsed + 2.5 };
        } else if (link.type === 'drop' && Math.abs(px - desiredX) < 25 && player.grounded && elapsed >= nextJump) {
          input.down = input.jump = true;
          engine.queueJumpInput();
          nextJump = elapsed + 0.65;
          desiredX = getTrainingLinkLandingX(link, engine.runtime.platforms, player.w);
          airTransfer = { type: 'drop', platform: next, x: desiredX, until: elapsed + 3 };
        } else if (link.type === 'walk' || link.type.startsWith('ramp')) {
          const direction = getTrainingRampDirection(link, platform, next) || Math.sign(link.entryX - link.exitX) || (platform.shape === 'slope' ? Math.sign(link.exitX - platform.x - platform.w / 2) : 0) || Math.sign(next.x + next.w / 2 - (platform.x + platform.w / 2)) || Math.sign(link.entryX - px) || 1;
          desiredX = Math.abs(px - link.exitX) <= player.w + 60 ? link.entryX + direction * (player.w + 48) : link.exitX;
          if (link.type === 'walk' && platform.shape !== 'slope' && next.shape !== 'slope' && Math.abs(px - link.exitX) <= player.w + 60) {
            const inset = Math.min(next.w / 2, player.w / 2 + 10);
            const handoffX = direction > 0 ? platform.x + platform.w + player.w / 2 + 10 : platform.x - player.w / 2 - 10;
            desiredX = Math.max(next.x + inset, Math.min(next.x + next.w - inset, handoffX));
          }
          // A broad upper one-way floor can cover the descending ramp. Walking
          // left stays on that floor; use the player's real drop-through input
          // to transfer onto the ramp beneath it.
          if (link.type === 'ramp-down' && next.shape === 'slope' && platform.shape !== 'slope' && player.grounded &&
              px > next.x && px < next.x + next.w && getSurfaceY(next, px) > getSurfaceY(platform, px) + 2 && elapsed >= nextJump) {
            input.down = input.jump = true;
            engine.queueJumpInput();
            nextJump = elapsed + 0.65;
            desiredX = next.x + next.w / 2;
            airTransfer = { type: 'drop', platform: next, x: desiredX, until: elapsed + 3 };
          }
          // A knockback can leave the player underneath a one-way ramp. Use the
          // normal jump input to remount it; never force a surface snap.
          if (link.type === 'ramp-up' && next.shape === 'slope' && px >= next.x && px <= next.x + next.w && player.y + player.h > getSurfaceY(next, px) + 10 && player.grounded && elapsed >= nextJump) {
            input.jump = true;
            engine.queueJumpInput();
            nextJump = elapsed + 0.8;
            desiredX = next.x + next.w / 2;
          }
        }
      }
    }
    if (player.climbing) {
      const activeClimbable = engine.runtime.climbables.find((entry) => entry.id === player.climbableId);
      if (activeClimbable && (!climbCommand || climbCommand.id !== activeClimbable.id)) {
        const endpoints = [activeClimbable.topPlatformIndex, activeClimbable.bottomPlatformIndex]
          .map((index) => engine.runtime.platforms[index]).filter(Boolean);
        endpoints.sort((a, b) => physicalRoute(a.index, desiredPlatform.index, px, desiredX).seconds - physicalRoute(b.index, desiredPlatform.index, px, desiredX).seconds ||
          Math.abs(getSurfaceY(a, px) - player.y - player.h) - Math.abs(getSurfaceY(b, px) - player.y - player.h));
        climbDestination = endpoints[0] || desiredPlatform;
        climbCommand = { id: activeClimbable.id, up: climbDestination.index === activeClimbable.topPlatformIndex, destination: climbDestination };
      }
      // Hold the commanded direction through the endpoint until the runtime
      // actually dismounts. Comparing feet against the endpoint flips direction
      // one frame too early and traps the actor between overlapping ladders.
      input.up = climbCommand ? climbCommand.up : input.up;
      input.down = !input.up;
      moving = false;
    }
    if (moving) { input.left = px > desiredX + 7; input.right = px < desiredX - 7; }
    if (moving && !engaging && player.grounded && !player.climbing && !airTransfer && player.mp > movementStats.maxMp * 0.5 && platform && platform.shape !== 'slope') {
      const direction = Math.sign(desiredX - px);
      for (const skill of loadout.mobilitySkills || []) {
        if (player.mp - skill.mpCost < loadout.attackMpReserve) continue;
        const facing = skill.movementEffect.direction === 'backward' ? -direction : direction;
        const application = movement.getSkillMovementApplicationPlan(skill, engine.state.skills[skill.id], { ...player, facing });
        const landingX = px + direction * application.distance;
        // Use only legal horizontal movement over verified continuous footing;
        // runtime costs, cooldowns, movement and collisions remain authoritative.
        if (Math.abs(desiredX - px) < application.distance + 40 || landingX < platform.x + player.w / 2 + 12 || landingX > platform.x + platform.w - player.w / 2 - 12) continue;
        player.facing = facing;
        if (engine.useSkill(skill.id, { silent: true, suppressCombatEmit: true })) { mobilityCasts += 1; break; }
      }
    }
    const progressPlatform = player.climbing && climbCommand ? climbCommand.destination : platform;
    const progressDestination = engaging && target ? desiredPlatform : goal;
    const progressKey = engaging && target ? `enemy:${target.uid}` : `route:${goal.id}`;
    const goalDistance = progressPlatform ? physicalRoute(progressPlatform.index, progressDestination.index, px, engaging && target ? desiredX : goal.x + goal.w / 2).seconds * routeStats.speed +
      (player.climbing ? Math.abs(player.y + player.h - getSurfaceY(progressPlatform, px)) : 0) : Infinity;
    if (progressGoalId !== progressKey || goalDistance < bestGoalDistance - 16 || inRange) {
      progressGoalId = progressKey;
      bestGoalDistance = goalDistance;
      lastProgressAt = elapsed;
    } else if (!inRange && elapsed - lastProgressAt > 3) {
      stuckSeconds += dt;
      // Retry a different real route; never snap/teleport the player.
      if (elapsed - lastProgressAt > 8) { advanceRoute(); lastProgressAt = elapsed; }
    }
    useTrainingConsumables(engine, loadout);
    // Count pursuit as combat only after the runtime establishes aggro (or the
    // player is already in weapon range). Merely approaching a pack is travel.
    const activeEngagement = (engaging || passingEncounter) && target && (inRange || target.aggroTargetKind && Number(target.aggroUntil || 0) > Date.now() / 1000);
    const atCombatPocket = arrivedAt !== null && player.grounded && platform && platform.id === goal.id;
    const navigationInput = !!(input.left || input.right || input.up || input.down || input.jump);
    const pendingLocalRespawn = !target && atCombatPocket && !navigationInput && engine.getWaveState(engine.state.mapId).pending.some((entry) => entry.spawnPlatformId === goal.id);
    return { phase: activeEngagement ? 'combat' : 'travel', target, pursuit: !!(activeEngagement && !inRange), outsideWeaponRange: !inRange,
      navigationInput, atCombatPocket, localEnemyAlive: !!target, pendingLocalRespawn,
      navigation: { platformId: platform && platform.id, goalId: goal.id, desiredX: ROUND(desiredX), left: input.left, right: input.right, up: input.up, down: input.down, climbing: player.climbing, climbDestination: climbDestination && climbDestination.id } };
  }
  return { drive, resetMeasurement: () => { visited.clear(); completedGoalsInOrder = 0; routeCycles = 0; routeAdvances = 0; mobilityCasts = 0; stuckSeconds = 0; },
    summary: () => ({ ...(selectedRoute || {}), routeCycles, routeAdvances, mobilityCasts, routePlatformCount: route.length, visitedPlatformIds: [...visited], stuckSeconds: ROUND(stuckSeconds) }) };
}

function findRecoveryPortal(data, fromMapId, destinationMapId) {
  const queue = [{ id: fromMapId, first: null }];
  const visited = new Set([fromMapId]);
  while (queue.length) {
    const current = queue.shift();
    if (current.id === destinationMapId) return current.first;
    const map = data.MAPS.find((entry) => entry.id === current.id);
    for (const portal of map && map.portals || []) {
      if (!portal.destinationMapId || portal.dungeonId || visited.has(portal.destinationMapId)) continue;
      visited.add(portal.destinationMapId);
      queue.push({ id: portal.destinationMapId, first: current.first || portal.id });
    }
  }
  return null;
}

function runTrainingScenario(data, createEngine, options) {
  const map = data.MAPS.find((entry) => entry.id === options.mapId);
  if (!map || !getPublicTrainingMaps(data).includes(map)) throw new Error(`Not a public training field: ${options.mapId}`);
  const level = Number(options.level);
  if (!Number.isInteger(level) || level < map.levelRange[0] || level > map.levelRange[1]) throw new Error(`Level ${level} outside ${map.id} range`);
  const warmupSeconds = options.warmupSeconds ?? TRAINING_PROTOCOL.warmupSeconds;
  const measuredSeconds = options.measuredSeconds ?? TRAINING_PROTOCOL.measuredSeconds;
  const fps = options.fps ?? TRAINING_PROTOCOL.fps;
  if (![30, 60, 120].includes(fps)) throw new Error('Training FPS must be 30, 60 or 120');
  if (!Number.isFinite(warmupSeconds) || warmupSeconds < 0 || !Number.isFinite(measuredSeconds) || measuredSeconds <= 0) throw new Error('Training windows must have nonnegative warmup and positive measured duration');
  const seed = Number(options.seed ?? TRAINING_SEEDS[0]) >>> 0;
  const classId = options.classId || 'fighter';
  const party = !!options.party;
  if (party && classId !== 'fighter') throw new Error('Fixed companion party requires fighter leader');
  let randomState = seed;
  let clockMs = BASE_TIME;
  const originalRandom = Math.random;
  const originalNow = Date.now;
  const performanceClock = globalThis.performance;
  const originalPerformanceNow = performanceClock && Object.getOwnPropertyDescriptor(performanceClock, 'now');
  try {
    Date.now = () => clockMs;
    if (performanceClock) Object.defineProperty(performanceClock, 'now', { configurable: true, value: () => clockMs - BASE_TIME });
    Math.random = () => { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; };
    const engine = createEngine(null, data);
    engine.playAudioCue = () => true;
    const loadout = prepareTrainingPlayer(data, engine, classId, level);
    if (party) {
      engine.state.party.members = ['archer', 'mage'].map((id, slot) => ({ id: `training-${id}`, classId: id, name: id, level, slot }));
    }
    engine.changeMap(map.id, { silent: true });
    const selectedRoute = selectTrainingRoute(map, engine.runtime, options);
    const controller = createTrainingController(engine, classId, loadout, null, selectedRoute);
    let recoveryController = null;
    let recoveryPortal = null;
    let recoveryMapId = '';
    let measuring = false;
    const totals = {};
    const lootKinds = {};
    const collectedItems = {};
    const usefulLoot = { equipmentCollected: 0, equippableEquipment: 0, potentialEquipmentResale: 0, materialUnits: 0, consumableUnits: 0, cardUnits: 0 };
    const killsByEnemy = {};
    const killsByGroup = {};
    const metric = engine.recordCombatMetric.bind(engine);
    engine.recordCombatMetric = (kind, amount) => {
      if (measuring) totals[kind] = (totals[kind] || 0) + Math.max(0, Number(amount || 0));
      return metric(kind, amount);
    };
    const pickup = engine.recordCombatLootPickup.bind(engine);
    engine.recordCombatLootPickup = (kind, quantity) => {
      if (measuring) lootKinds[kind] = (lootKinds[kind] || 0) + Math.max(1, Number(quantity || 1));
      return pickup(kind, quantity);
    };
    const lootItem = engine.lootItem.bind(engine);
    engine.lootItem = (uid, settings) => {
      const drop = engine.findCurrentMapLootDropByUid(uid);
      const admission = drop ? engine.getLootDropInventoryAdmission(drop) : null;
      const result = lootItem(uid, settings);
      if (measuring && result && drop && admission) {
        const id = admission.materialId || admission.consumableId || admission.cardId || drop.item.id || admission.kind;
        const key = `${admission.kind}:${id}`;
        collectedItems[key] = (collectedItems[key] || 0) + Math.max(1, Number(admission.quantity || 1));
        Object.entries(assessTrainingLoot(engine, admission, drop.item)).forEach(([name, value]) => { usefulLoot[name] += value; });
      }
      return result;
    };
    const killed = engine.recordCombatKill.bind(engine);
    engine.recordCombatKill = (enemy) => {
      if (measuring) {
        killsByEnemy[enemy.id] = (killsByEnemy[enemy.id] || 0) + 1;
        killsByGroup[enemy.spawnGroupId || 'ungrouped'] = (killsByGroup[enemy.spawnGroupId || 'ungrouped'] || 0) + 1;
      }
      return killed(enemy);
    };
    const resolveAttack = engine.resolveEnemyPendingAttack.bind(engine);
    let enemyHealing = 0;
    let healingPulses = 0;
    engine.resolveEnemyPendingAttack = (enemy, characters) => {
      const recipients = enemy.pendingAttack && enemy.pendingAttack.kind === 'heal' && Number(enemy.telegraph || 0) <= 0
        ? new Map(enemy.pendingAttack.recipients.map((recipient) => [recipient, recipient.hp])) : null;
      const result = resolveAttack(enemy, characters);
      if (measuring && recipients && enemy.lastAttackOutcome === 'healed') {
        healingPulses += 1;
        recipients.forEach((hp, recipient) => { enemyHealing += Math.max(0, recipient.hp - hp); });
      }
      return result;
    };
    const phases = { combat: 0, travel: 0, respawnWaiting: 0, idle: 0, recovery: 0 };
    const rawMotion = { movingSeconds: 0, engagedPursuitSeconds: 0, outsideWeaponRangeSeconds: 0 };
    const measuredFrames = Math.round(measuredSeconds * fps);
    const warmupFrames = Math.round(warmupSeconds * fps);
    let partyXpBaseline = 0;
    let partyDeaths = 0;
    let partyDownMemberSeconds = 0;
    const partyDownsByMember = {};
    const partyDownSecondsByMember = {};
    const deadMembers = new Set();
    let invalidCoordinates = 0;
    const samples = [];
    for (let frame = 0; frame < warmupFrames + measuredFrames; frame += 1) {
      const elapsed = frame / fps;
      clockMs = BASE_TIME + frame * 1000 / fps;
      if (frame === warmupFrames) {
        measuring = true;
        controller.resetMeasurement();
        partyXpBaseline = engine.getActivePrototypePartyMembers().reduce((sum, member) => sum + Number(member.sharedXp || 0), 0);
      }
      const before = { x: engine.state.player.x, y: engine.state.player.y };
      let control;
      if (engine.state.mapId === map.id) control = controller.drive(elapsed, 1 / fps);
      else {
        if (recoveryMapId !== engine.state.mapId) {
          recoveryMapId = engine.state.mapId;
          const portalId = findRecoveryPortal(data, recoveryMapId, map.id);
          recoveryPortal = engine.runtime.portals.find((portal) => portal.id === portalId);
          recoveryController = recoveryPortal ? createTrainingController(engine, classId, loadout, recoveryPortal) : null;
        }
        control = recoveryController ? recoveryController.drive(elapsed, 1 / fps) : {};
        control.phase = 'recovery';
        if (!recoveryController) Object.keys(engine.input).forEach((id) => { engine.input[id] = false; });
        // Portal use is only attempted at its real activation zone; production
        // eligibility still applies. No map teleport or reset after death.
        if (recoveryPortal && engine.state.player.activePortalId === recoveryPortal.id) engine.usePortal(recoveryPortal.id);
      }
      if (measuring && control.pursuit) rawMotion.engagedPursuitSeconds += 1 / fps;
      if (measuring && control.outsideWeaponRange) rawMotion.outsideWeaponRangeSeconds += 1 / fps;
      engine.frameId += 1;
      engine.invalidateFrameStatsCache();
      engine.update(1 / fps);
      const moving = Math.hypot(engine.state.player.x - before.x, engine.state.player.y - before.y) > 0.1;
      const phase = classifyTrainingPhase({ ...control, moving, hasRoute: control.phase !== 'idle', combat: control.phase === 'combat', recovery: control.phase === 'recovery' });
      if (measuring) phases[phase] += 1 / fps;
      if (measuring && moving) rawMotion.movingSeconds += 1 / fps;
      if (![engine.state.player.x, engine.state.player.y, ...engine.enemies.flatMap((enemy) => [enemy.x, enemy.y])].every(Number.isFinite)) invalidCoordinates += 1;
      engine.getActivePrototypePartyMembers().forEach((member) => {
        if (member.mode === 'down' && measuring) {
          partyDownMemberSeconds += 1 / fps;
          partyDownSecondsByMember[member.id] = (partyDownSecondsByMember[member.id] || 0) + 1 / fps;
        }
        if (member.mode === 'down' && !deadMembers.has(member.id)) {
          if (measuring) {
            partyDeaths += 1;
            partyDownsByMember[member.id] = (partyDownsByMember[member.id] || 0) + 1;
          }
          deadMembers.add(member.id);
        }
        if (member.mode !== 'down') deadMembers.delete(member.id);
      });
      if (frame % (fps * 30) === 0) samples.push({ seconds: ROUND(elapsed), mapId: engine.state.mapId, x: ROUND(engine.state.player.x), y: ROUND(engine.state.player.y), hp: ROUND(engine.state.player.hp), kills: totals.kill || 0, phase, navigation: control.navigation });
    }
    const partyXp = engine.getActivePrototypePartyMembers().reduce((sum, member) => sum + Number(member.sharedXp || 0), 0) - partyXpBaseline;
    const minutes = measuredSeconds / 60;
    return {
      evidenceKind: 'observed-engine', mapId: map.id, classId, party, level, seed, routeScope: selectedRoute.scope,
      warmupSeconds, measuredSeconds, fps, loadout,
      population: getMapSpawnPopulation(map, party ? 3 : 1),
      totals, killsByEnemy, killsByGroup, lootKinds, collectedItems, partyDeaths,
      partyDownMemberSeconds: ROUND(partyDownMemberSeconds), partyDownsByMember,
      partyDownSecondsByMember: Object.fromEntries(Object.entries(partyDownSecondsByMember).map(([id, seconds]) => [id, ROUND(seconds)])),
      enemyHealing, healingPulses,
      leaderXpPerMinute: ROUND((totals.xp || 0) / minutes),
      companionXpCredits: partyXp,
      aggregateXpCreditsPerMinute: ROUND(((totals.xp || 0) + partyXp) / minutes),
      killsPerMinute: ROUND((totals.kill || 0) / minutes),
      damageTakenPerMinute: ROUND((totals.damageTaken || 0) / minutes),
      leaderDamageTakenPerMinute: ROUND((totals.damageTaken || 0) / minutes),
      consumableCostPerMinute: ROUND((totals.potionCost || 0) / minutes),
      phases: Object.fromEntries(Object.entries(phases).map(([key, value]) => [key, ROUND(value)])),
      rawMotion: Object.fromEntries(Object.entries(rawMotion).map(([key, value]) => [key, ROUND(value)])),
      travelPercent: ROUND(phases.travel / measuredSeconds * 100),
      respawnWaitingPercent: ROUND(phases.respawnWaiting / measuredSeconds * 100),
      recoveryPercent: ROUND(phases.recovery / measuredSeconds * 100),
      route: controller.summary(), usefulLoot, invalidCoordinates, samples,
      finalMapId: engine.state.mapId,
      limitations: [
        'Deterministic input bot; imperfect route/combat choices can limit observed throughput. Route coverage and stuck time are reported.',
        'Character level/loadout/mastery fixed; earned XP counted without level-ups or permanent mastery gains. No manual rewards, coupons, account bonuses, upgrades or admin boosts.',
        'Death uses the real town recovery; the bot walks through eligible portals back to the field. All return time counts against throughput.',
        'Travel excludes active local-pack pursuit, which is reported separately along with raw moving time and time outside weapon reach.',
        'Respawn waiting requires a stationary player at the current pocket, no nearby live target, and an actual pending replacement on that platform; walking between pockets remains travel.',
        'Damage taken is production leader damage telemetry, including absorbed damage; it is not combined party HP loss.',
        'Equippable loot passes the real class/level rules but is not necessarily an upgrade. Potential equipment resale is an appraisal only; no items are sold and it is not earned currency or net profit.',
        ...(party ? ['Fighter leader plus runtime archer/mage companions; companions use authored AI stats/skill ranks, not human equipment budgets. Companion XP credits are reported separately.',
          'Companion down counts record transitions during measurement. Down member-seconds sum each companion time in the real down state, including ongoing recovery from warmup; two companions down together contribute two member-seconds per second.'] : [])
      ]
    };
  } finally {
    Math.random = originalRandom;
    Date.now = originalNow;
    if (performanceClock) {
      if (originalPerformanceNow) Object.defineProperty(performanceClock, 'now', originalPerformanceNow);
      else delete performanceClock.now;
    }
  }
}

function summarizeTrainingRuns(runs) {
  const grouped = new Map();
  runs.forEach((run) => {
    const key = `${run.routeScope || 'full'}:${run.level}:${run.mapId}:${run.party ? 'party' : run.classId}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(run);
  });
  const rows = [...grouped.values()].map((entries) => {
    const first = entries[0];
    const mean = (key) => ROUND(entries.reduce((sum, run) => sum + run[key], 0) / entries.length);
    const seedSet = new Set(entries.map((run) => run.seed));
    const eligibleForAcceptance = entries.every((run) => run.warmupSeconds === 60 && run.measuredSeconds === 300 && run.fps === first.fps && run.invalidCoordinates === 0 && !run.route.fallback && !(run.route.missingPlatformIds || []).length && run.route.visitedPlatformIds.length >= run.route.routePlatformCount * 0.75 && run.route.stuckSeconds <= 18) && entries.length === TRAINING_SEEDS.length && TRAINING_SEEDS.every((seed) => seedSet.has(seed));
    return { mapId: first.mapId, level: first.level, classId: first.classId, party: first.party, routeScope: first.routeScope || 'full', seeds: entries.length, eligibleForAcceptance,
      leaderXpPerMinute: mean('leaderXpPerMinute'), aggregateXpCreditsPerMinute: mean('aggregateXpCreditsPerMinute'), killsPerMinute: mean('killsPerMinute'),
      travelPercent: mean('travelPercent'), respawnWaitingPercent: mean('respawnWaitingPercent'), recoveryPercent: mean('recoveryPercent'),
      damageTakenPerMinute: mean('damageTakenPerMinute'), consumableCostPerMinute: mean('consumableCostPerMinute'),
      deaths: entries.reduce((sum, run) => sum + Number(run.totals.death || 0), 0),
      companionDowns: entries.reduce((sum, run) => sum + Number(run.partyDeaths || 0), 0),
      companionDownMemberSeconds: ROUND(entries.reduce((sum, run) => sum + Number(run.partyDownMemberSeconds || 0), 0) / entries.length),
      visitedPlatformCount: ROUND(entries.reduce((sum, run) => sum + run.route.visitedPlatformIds.length, 0) / entries.length),
      routePlatformCount: first.route.routePlatformCount,
      stuckSeconds: ROUND(entries.reduce((sum, run) => sum + run.route.stuckSeconds, 0) / entries.length)
    };
  });
  return rows.map((row) => {
    const peers = rows.filter((peer) => peer.level === row.level && peer.classId === row.classId && peer.party === row.party && peer.routeScope === row.routeScope);
    const sorted = peers.map((peer) => peer.leaderXpPerMinute).sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    return { ...row, cohortMapCount: peers.length, cohortMedianXpPerMinute: median, deviationFromCohortMedianPercent: median ? ROUND((row.leaderXpPerMinute / median - 1) * 100) : null,
      pacingWithinTarget: row.travelPercent <= 30 && row.respawnWaitingPercent <= 10,
      acceptanceNote: row.eligibleForAcceptance ? 'Measured route coverage and sample protocol sufficient for review; class/map balance still requires cohort judgment.' : 'Diagnostic only: incomplete route coverage, bot stalls, fewer than three seeds or nonstandard sample duration.' };
  });
}

module.exports = { TRAINING_PROTOCOL, TRAINING_SEEDS, getPublicTrainingMaps, getEligibleTrainingClasses, getTrainingCohorts, prepareTrainingPlayer, useTrainingConsumables, assessTrainingLoot, classifyTrainingPhase, selectTrainingRoute, getTrainingLinkLandingX, getTrainingRampDirection, isTrainingDropTraversable, prepareTrainingDropLink, findTrainingRoute, runTrainingScenario, summarizeTrainingRuns };
