(function initProjectStarfallEngineDungeons(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  const DEFAULT_BOSS_RESPAWN_SECONDS = 30;

  function getDungeonData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getDungeonObjectiveDefinitions(dungeonId, options) {
    const data = getDungeonData(options);
    const dungeon = getById(data.DUNGEONS || [], dungeonId);
    if (!dungeon) return [];
    return data.DUNGEON_OBJECTIVES || [];
  }

  function getActiveDungeonDefinition(dungeons, options) {
    const data = getDungeonData(options);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : {};
    const id = normalizeId(state.activeDungeonId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.id) === id) || null;
  }

  function getDungeonDefinitionById(dungeonId, options) {
    const data = getDungeonData(options);
    const id = normalizeId(dungeonId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.id) === id) || null;
  }

  function getDungeonBossIds(map, options) {
    if (!map || !map.isDungeon) return [];
    const dungeon = map.dungeonId ? getDungeonDefinitionById(map.dungeonId, options) : null;
    const configured = map.bossId
      ? [map.bossId]
      : dungeon && Array.isArray(dungeon.bossIds) && dungeon.bossIds.length
      ? dungeon.bossIds
      : dungeon && dungeon.bossId
        ? [dungeon.bossId]
        : [];
    return Array.from(new Set(configured.filter(Boolean)));
  }

  function getDungeonDefinitionByMapId(mapId, options) {
    const data = getDungeonData(options);
    const id = normalizeId(mapId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.mapId) === id) || null;
  }

  function createDungeonObjectiveRunState(dungeonId, value, options) {
    const data = getDungeonData(options);
    const source = value && typeof value === 'object' ? value : {};
    const configured = data.DUNGEON_OBJECTIVES || [];
    return configured.reduce((objectives, objective) => {
      const id = normalizeId(objective && objective.id);
      if (!id) return objectives;
      const entry = source[id] && typeof source[id] === 'object' ? source[id] : {};
      const goal = Math.max(1, Number(objective.goal || 1) || 1);
      objectives[id] = {
        progress: clamp(Number(entry.progress || 0), 0, goal),
        complete: !!entry.complete || Number(entry.progress || 0) >= goal,
        failed: !!entry.failed,
        claimed: !!entry.claimed
      };
      return objectives;
    }, {});
  }

  function ensureDungeonRunObjectives(run, options) {
    if (!run || !run.dungeonId) return {};
    run.objectives = createDungeonObjectiveRunState(run.dungeonId, run.objectives, options);
    return run.objectives;
  }

  function recordDungeonObjectiveRunProgress(run, type, amount, options) {
    if (!run || run.completedAt || !run.dungeonId) return false;
    const data = getDungeonData(options);
    const objectives = ensureDungeonRunObjectives(run, options);
    let changed = false;
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== type) return;
      const entry = objectives[objective.id];
      if (!entry || entry.failed || entry.complete) return;
      const goal = Math.max(1, Number(objective.goal || 1));
      entry.progress = clamp(Number(entry.progress || 0) + Math.max(0, Number(amount || 1)), 0, goal);
      entry.complete = entry.progress >= goal;
      changed = true;
    });
    return changed;
  }

  function failDungeonObjectiveRun(run, type, options) {
    if (!run || run.completedAt || !run.dungeonId) return false;
    const data = getDungeonData(options);
    const objectives = ensureDungeonRunObjectives(run, options);
    let changed = false;
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== type) return;
      const entry = objectives[objective.id];
      if (!entry || entry.complete) return;
      entry.failed = true;
      changed = true;
    });
    return changed;
  }

  function finalizeDungeonObjectiveRunState(run, options) {
    if (!run || !run.dungeonId) return;
    const settings = options || {};
    const data = getDungeonData(settings);
    const nowMs = Object.prototype.hasOwnProperty.call(settings, 'nowMs')
      ? Number(settings.nowMs || 0)
      : Date.now();
    const fallbackStartedAt = Object.prototype.hasOwnProperty.call(settings, 'fallbackStartedAt')
      ? Number(settings.fallbackStartedAt || 0)
      : Date.now();
    const elapsed = Math.max(0, (nowMs - Number(run.startedAt || fallbackStartedAt)) / 1000);
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== 'timedClear') return;
      const entry = ensureDungeonRunObjectives(run, settings)[objective.id];
      if (!entry) return;
      const goal = Math.max(1, Number(objective.goal || 1));
      entry.progress = Math.min(goal, elapsed <= goal ? goal : 0);
      entry.complete = elapsed <= goal;
      entry.failed = elapsed > goal;
    });
    const survival = (data.DUNGEON_OBJECTIVES || []).find((objective) => objective.type === 'partySurvival');
    if (survival) {
      const entry = ensureDungeonRunObjectives(run, settings)[survival.id];
      if (entry) {
        entry.progress = Number(run.partyDefeats || 0) > 0 ? 0 : 1;
        entry.complete = Number(run.partyDefeats || 0) <= 0;
        entry.failed = Number(run.partyDefeats || 0) > 0;
      }
    }
  }

  function awardDungeonObjectiveRunRewards(run, awardReward, options) {
    if (!run || !run.objectives || typeof awardReward !== 'function') return;
    const data = getDungeonData(options);
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      const entry = run.objectives[objective.id];
      if (!entry || !entry.complete || entry.claimed || !objective.reward) return;
      awardReward(objective.reward);
      entry.claimed = true;
    });
  }

  function createDungeonState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const completedDungeonIds = Array.isArray(source.completedDungeonIds)
      ? source.completedDungeonIds.map(normalizeId).filter(Boolean)
      : [];
    const completionCounts = {};
    Object.entries(source.completionCounts || {}).forEach(([id, count]) => {
      const key = normalizeId(id);
      if (key) completionCounts[key] = Math.max(0, Math.floor(Number(count) || 0));
    });
    const bossRespawnAt = {};
    Object.entries(source.bossRespawnAt || {}).forEach(([id, value]) => {
      const key = normalizeId(id);
      const timestamp = Number(value || 0);
      if (key && timestamp > 0) bossRespawnAt[key] = timestamp;
    });
    const currentRun = source.currentRun && typeof source.currentRun === 'object'
      ? {
          dungeonId: normalizeId(source.currentRun.dungeonId),
          startedAt: Number(source.currentRun.startedAt || 0),
          completedAt: Number(source.currentRun.completedAt || 0),
          bossDefeated: !!source.currentRun.bossDefeated,
          bossEncounterId: normalizeId(source.currentRun.bossEncounterId),
          adminEncounter: !!source.currentRun.adminEncounter,
          objectives: createDungeonObjectiveRunState(source.currentRun.dungeonId, source.currentRun.objectives, options),
          partyDefeats: Math.max(0, Math.floor(Number(source.currentRun.partyDefeats || 0) || 0))
        }
      : null;
    return {
      activeDungeonId: normalizeId(source.activeDungeonId),
      currentRun: currentRun && currentRun.dungeonId ? currentRun : null,
      completedDungeonIds: Array.from(new Set(completedDungeonIds)),
      completionCounts,
      bossRespawnAt,
      lastCompletedAt: Number(source.lastCompletedAt || 0)
    };
  }

  function createDungeonStartBlockReason(dungeon, player) {
    const activePlayer = player || {};
    if (!dungeon) return 'Dungeon is unavailable.';
    if (!activePlayer.classId) return 'Choose a class first.';
    if (activePlayer.level < Number(dungeon.levelRequirement || 1)) return `Level ${dungeon.levelRequirement} required.`;
    if (dungeon.requiresAdvancedClass && !activePlayer.advancedClassId) return 'Choose an advanced class first.';
    return '';
  }

  function createDungeonBossRespawnInfo(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const respawnAt = id ? Number(state.bossRespawnAt && state.bossRespawnAt[id] || 0) : 0;
    const remainingMs = Math.max(0, respawnAt - Number(settings.nowMs || 0));
    const bossRespawnSeconds = Math.max(0, Number(settings.bossRespawnSeconds || DEFAULT_BOSS_RESPAWN_SECONDS));
    return {
      respawnAt,
      remainingMs,
      remaining: Math.min(bossRespawnSeconds, remainingMs / 1000),
      respawning: remainingMs > 0
    };
  }

  function isDungeonBossRespawning(dungeonId, dungeons, options) {
    return createDungeonBossRespawnInfo(dungeonId, dungeons, options).respawning;
  }

  function refreshDungeonBossRespawnState(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    if (!id) return false;
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const respawnAt = Number(state.bossRespawnAt && state.bossRespawnAt[id] || 0);
    if (!respawnAt || Number(settings.nowMs || 0) < respawnAt) return false;
    delete state.bossRespawnAt[id];
    state.activeDungeonId = id;
    state.currentRun = createDungeonStartRunState(id, false, settings);
    return true;
  }

  function createDungeonStartRunState(dungeonId, bossRespawning, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    const respawning = !!bossRespawning;
    return {
      dungeonId: id,
      startedAt: Number(settings.startedAt || 0),
      completedAt: respawning ? Number(settings.completedAt || 0) : 0,
      bossDefeated: respawning,
      objectives: createDungeonObjectiveRunState(id, null, settings),
      partyDefeats: 0
    };
  }

  function createMapChangeDungeonRunState(dungeonId, bossRespawning, options) {
    return createDungeonStartRunState(dungeonId, bossRespawning, options);
  }

  function startDungeonState(dungeonId, dungeons, bossRespawning, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createDungeonStartRunState(dungeonId, bossRespawning, settings);
    return state.currentRun;
  }

  function transitionDungeonMapState(dungeonId, dungeons, run, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createDungeonMapTransitionRunState(dungeonId, run, settings);
    return state.currentRun;
  }

  function changeMapDungeonState(dungeonId, dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    if (dungeonId) {
      state.activeDungeonId = dungeonId;
      if (!state.currentRun || state.currentRun.dungeonId !== state.activeDungeonId) {
        state.currentRun = createMapChangeDungeonRunState(state.activeDungeonId, !!settings.bossRespawning, settings);
      }
    } else {
      state.activeDungeonId = '';
      state.currentRun = null;
    }
    return state.currentRun;
  }

  function createDungeonCompletionRunState(dungeonId, currentRun, options) {
    const settings = options || {};
    const source = currentRun && typeof currentRun === 'object' ? currentRun : {};
    const existingRun = settings.existingRun && typeof settings.existingRun === 'object' ? settings.existingRun : null;
    const id = normalizeId(dungeonId);
    const completedAt = Number(settings.completedAt || 0);
    return Object.assign({}, source, {
      dungeonId: id,
      startedAt: Number(source.startedAt || completedAt),
      completedAt,
      bossDefeated: true,
      objectives: existingRun ? existingRun.objectives : createDungeonObjectiveRunState(id, null, settings),
      partyDefeats: existingRun ? Number(existingRun.partyDefeats || 0) : 0
    });
  }

  function completeDungeonState(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = dungeonId;
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    if (!Array.isArray(state.completedDungeonIds)) state.completedDungeonIds = [];
    if (!state.completionCounts || typeof state.completionCounts !== 'object') state.completionCounts = {};
    if (!state.completedDungeonIds.includes(id)) state.completedDungeonIds.push(id);
    state.completionCounts[id] = Math.max(0, Number(state.completionCounts[id] || 0)) + 1;
    const completedAt = Number(settings.completedAt || 0);
    state.lastCompletedAt = completedAt;
    state.activeDungeonId = id;
    state.bossRespawnAt = state.bossRespawnAt && typeof state.bossRespawnAt === 'object' ? state.bossRespawnAt : {};
    const bossRespawnSeconds = Math.max(0, Number(settings.bossRespawnSeconds || DEFAULT_BOSS_RESPAWN_SECONDS));
    state.bossRespawnAt[id] = completedAt + bossRespawnSeconds * 1000;
    state.currentRun = createDungeonCompletionRunState(id, state.currentRun, settings);
    return state;
  }

  function createBossEncounterCompletionRunState(dungeonId, currentRun, options) {
    const settings = options || {};
    const source = currentRun && typeof currentRun === 'object' ? currentRun : {};
    const completedAt = Number(settings.completedAt || 0);
    return Object.assign({}, source, {
      dungeonId,
      startedAt: Number(source.startedAt || completedAt),
      completedAt,
      bossDefeated: true,
      bossEncounterId: settings.bossEncounterId,
      adminEncounter: true
    });
  }

  function completeBossEncounterState(dungeonId, dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createBossEncounterCompletionRunState(state.activeDungeonId, state.currentRun, settings);
    return state;
  }

  function createBossEncounterStartRunState(dungeonId, bossEncounterId, adminEncounter, options) {
    const settings = options || {};
    return {
      dungeonId,
      startedAt: Number(settings.startedAt || 0),
      completedAt: 0,
      bossDefeated: false,
      bossEncounterId,
      adminEncounter: !!adminEncounter
    };
  }

  function startBossEncounterState(dungeonId, dungeons, bossEncounterId, adminEncounter, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createBossEncounterStartRunState(state.activeDungeonId, bossEncounterId, adminEncounter, settings);
    return state.currentRun;
  }

  function createDungeonMapTransitionRunState(dungeonId, run, options) {
    const settings = options || {};
    const source = run && typeof run === 'object' ? run : {};
    const id = normalizeId(dungeonId);
    return Object.assign({}, source, {
      objectives: createDungeonObjectiveRunState(id, source.objectives, settings),
      partyDefeats: Number(source.partyDefeats || 0)
    });
  }

  function createDungeonObjectiveSnapshots(objectives, options) {
    const data = getDungeonData(options);
    return Object.entries(objectives || {}).map(([id, entry]) => {
      const objective = getById(data.DUNGEON_OBJECTIVES || [], id) || {};
      const goal = Math.max(1, Number(objective.goal || 1));
      return Object.assign({}, objective, entry, {
        id,
        goal,
        progress: clamp(Number(entry && entry.progress || 0), 0, goal)
      });
    });
  }

  function createDungeonSummary(dungeon, dungeons, objectives, options) {
    if (!dungeon) return null;
    const data = getDungeonData(options);
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const map = getById(data.MAPS || [], dungeon.mapId);
    const boss = getById(data.ENEMIES || [], dungeon.bossId);
    const count = Math.max(0, Number(state.completionCounts && state.completionCounts[dungeon.id] || 0));
    const respawn = createDungeonBossRespawnInfo(dungeon.id, state, settings);
    const run = state.currentRun && state.currentRun.dungeonId === dungeon.id ? state.currentRun : null;
    return {
      id: dungeon.id,
      name: dungeon.name,
      summary: dungeon.summary,
      mapId: dungeon.mapId,
      mapName: map ? map.name : dungeon.mapId,
      bossId: dungeon.bossId,
      bossName: boss ? boss.name : dungeon.bossId,
      levelRequirement: dungeon.levelRequirement,
      recommendedPartySize: dungeon.recommendedPartySize,
      requiresAdvancedClass: !!dungeon.requiresAdvancedClass,
      active: state.activeDungeonId === dungeon.id,
      complete: (state.completedDungeonIds || []).includes(dungeon.id),
      completionCount: count,
      bossRespawnAt: respawn.respawnAt,
      bossRespawnRemaining: respawn.remaining,
      bossRespawning: respawn.respawning,
      lockedReason: createDungeonStartBlockReason(dungeon, settings.player),
      objectives: run ? createDungeonObjectiveSnapshots(objectives || run.objectives, settings) : []
    };
  }

  function createDungeonSummaryFromState(dungeon, dungeons, options) {
    if (!dungeon) return null;
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const run = state.currentRun && state.currentRun.dungeonId === dungeon.id ? state.currentRun : null;
    const objectives = run ? ensureDungeonRunObjectives(run, settings) : null;
    return createDungeonSummary(dungeon, state, objectives, settings);
  }

  function createDungeonSnapshot(dungeons, activeDungeonSummary, dungeonSummaries) {
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState();
    return {
      activeDungeon: activeDungeonSummary || null,
      dungeons: Array.isArray(dungeonSummaries) ? dungeonSummaries : [],
      completedDungeonIds: (state.completedDungeonIds || []).slice(),
      completionCounts: Object.assign({}, state.completionCounts),
      bossRespawnAt: Object.assign({}, state.bossRespawnAt || {}),
      currentRun: state.currentRun ? Object.assign({}, state.currentRun) : null
    };
  }

  function getDungeonSnapshotSummaryOptions(options) {
    const settings = options || {};
    const nowMs = typeof settings.nowMsProvider === 'function'
      ? settings.nowMsProvider()
      : Number(settings.nowMs || 0);
    return Object.assign({}, settings, { nowMs });
  }

  function createDungeonSnapshotFromState(dungeons, options) {
    const settings = options || {};
    const data = getDungeonData(settings);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const activeDungeon = getActiveDungeonDefinition(state, settings);
    const activeSummary = activeDungeon
      ? createDungeonSummaryFromState(activeDungeon, state, getDungeonSnapshotSummaryOptions(settings))
      : null;
    const summaries = (data.DUNGEONS || []).map((dungeon) =>
      createDungeonSummaryFromState(dungeon, state, getDungeonSnapshotSummaryOptions(settings))
    );
    return createDungeonSnapshot(state, activeSummary, summaries);
  }

  function createDungeonTrackerSnapshot(dungeons, activeDungeonSummary) {
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState();
    return {
      activeDungeon: activeDungeonSummary || null,
      dungeons: [],
      completedDungeonIds: [],
      completionCounts: {},
      bossRespawnAt: {},
      currentRun: state.currentRun ? Object.assign({}, state.currentRun) : null
    };
  }

  function createDungeonTrackerSnapshotFromState(dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const activeDungeon = getDungeonDefinitionById(state.activeDungeonId, settings);
    const activeSummary = activeDungeon
      ? createDungeonSummaryFromState(activeDungeon, state, getDungeonSnapshotSummaryOptions(settings))
      : null;
    return createDungeonTrackerSnapshot(state, activeSummary);
  }

  const api = {
    DEFAULT_BOSS_RESPAWN_SECONDS,
    getDungeonObjectiveDefinitions,
    getActiveDungeonDefinition,
    getDungeonDefinitionById,
    getDungeonBossIds,
    getDungeonDefinitionByMapId,
    createDungeonObjectiveRunState,
    ensureDungeonRunObjectives,
    recordDungeonObjectiveRunProgress,
    failDungeonObjectiveRun,
    finalizeDungeonObjectiveRunState,
    awardDungeonObjectiveRunRewards,
    createDungeonState,
    createDungeonStartBlockReason,
    createDungeonBossRespawnInfo,
    isDungeonBossRespawning,
    refreshDungeonBossRespawnState,
    createDungeonStartRunState,
    createMapChangeDungeonRunState,
    startDungeonState,
    transitionDungeonMapState,
    changeMapDungeonState,
    createDungeonCompletionRunState,
    completeDungeonState,
    createBossEncounterCompletionRunState,
    completeBossEncounterState,
    createBossEncounterStartRunState,
    startBossEncounterState,
    createDungeonMapTransitionRunState,
    createDungeonObjectiveSnapshots,
    createDungeonSummary,
    createDungeonSummaryFromState,
    createDungeonSnapshot,
    createDungeonSnapshotFromState,
    createDungeonTrackerSnapshot,
    createDungeonTrackerSnapshotFromState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.dungeons = Object.assign({}, modules.dungeons || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
