(function initProjectStarfallEngineSeason(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    const source = Array.isArray(items) ? items : [];
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] && source[index].id === id) return source[index];
    }
    return null;
  };

  const seasonObjectiveTypeMapCache = new Map();

  function getSeasonData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getDefaultSeasonId(options) {
    const data = getSeasonData(options);
    const active = (data.SEASONS || []).find((season) => season && season.active);
    return active ? active.id : ((data.SEASONS || [])[0] || {}).id || '';
  }

  function createSeasonState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const objectiveValues = {};
    Object.entries(source.objectiveValues || {}).forEach(([id, count]) => {
      const key = normalizeId(id);
      if (key) objectiveValues[key] = Math.max(0, Math.floor(Number(count) || 0));
    });
    const claimedRewardIds = Array.isArray(source.claimedRewardIds)
      ? source.claimedRewardIds.map(normalizeId).filter(Boolean)
      : [];
    return {
      activeSeasonId: normalizeId(source.activeSeasonId) || getDefaultSeasonId(options),
      objectiveValues,
      claimedRewardIds: Array.from(new Set(claimedRewardIds))
    };
  }

  function getActiveSeasonData(state, options) {
    const data = getSeasonData(options);
    const source = state && typeof state === 'object' ? state : {};
    const season = source.activeSeasonId ? getById(data.SEASONS || [], source.activeSeasonId) : null;
    return season || getById(data.SEASONS || [], getDefaultSeasonId(options));
  }

  function getSeasonObjectivesByType(season, type) {
    const seasonId = normalizeId(season && season.id);
    const objectiveType = normalizeId(type);
    if (!seasonId || !objectiveType) return [];
    if (!seasonObjectiveTypeMapCache.has(seasonId)) {
      const objectiveMap = new Map();
      (season.objectives || []).forEach((objective, index) => {
        const id = normalizeId(objective && objective.type);
        if (!id) return;
        if (!objectiveMap.has(id)) objectiveMap.set(id, []);
        objectiveMap.get(id).push({ objective, index });
      });
      seasonObjectiveTypeMapCache.set(seasonId, objectiveMap);
    }
    const objectiveMap = seasonObjectiveTypeMapCache.get(seasonId);
    return objectiveMap ? objectiveMap.get(objectiveType) || [] : [];
  }

  function createSeasonSnapshot(state, season, options) {
    const settings = options || {};
    if (!season) return { activeSeason: null, objectives: [], complete: false, rewardClaimed: false };
    const createObjectiveStatuses = settings.createObjectiveStatuses || function createObjectiveStatusesFallback() {
      return [];
    };
    const source = state && typeof state === 'object' ? state : {};
    const entry = { objectiveValues: source.objectiveValues || {} };
    const objectives = createObjectiveStatuses(entry, season.objectives || []);
    return {
      activeSeason: season,
      objectives,
      complete: objectives.length > 0 && objectives.every((objective) => objective.complete),
      rewardClaimed: Array.isArray(source.claimedRewardIds) && source.claimedRewardIds.includes(season.id)
    };
  }

  function createSeasonEventPlan(state, season, type, payload, options) {
    const settings = options || {};
    const source = state && typeof state === 'object' ? state : {};
    if (!season || Array.isArray(source.claimedRewardIds) && source.claimedRewardIds.includes(season.id) && !(season.objectives || []).length) {
      return { changed: false, changes: [] };
    }
    const readObjectives = settings.getSeasonObjectivesByType || getSeasonObjectivesByType;
    const matchObjective = settings.matchObjective || function matchObjectiveFallback() {
      return false;
    };
    const getObjectiveKey = settings.getObjectiveKey || function getObjectiveKeyFallback(objective, index) {
      return normalizeId(objective && objective.id) || `${normalizeId(objective && objective.type) || 'objective'}_${index}`;
    };
    const getObjectiveGoal = settings.getObjectiveGoal || function getObjectiveGoalFallback(objective) {
      return Math.max(1, Number(objective && (objective.count || objective.level) || 1));
    };
    const objectives = readObjectives(season, type);
    if (!objectives.length) return { changed: false, changes: [] };
    const objectiveValues = source.objectiveValues || {};
    const changes = [];
    objectives.forEach(({ objective, index }) => {
      if (!matchObjective(objective, type, payload)) return;
      const key = getObjectiveKey(objective, index);
      const goal = getObjectiveGoal(objective);
      const before = Math.max(0, Number(objectiveValues[key]) || 0);
      const next = Math.min(goal, before + Math.max(1, Number(payload && payload.count || 1)));
      if (next !== before) changes.push({ key, next });
    });
    return {
      changed: changes.length > 0,
      changes
    };
  }

  const api = {
    getDefaultSeasonId,
    createSeasonState,
    getActiveSeasonData,
    getSeasonObjectivesByType,
    createSeasonSnapshot,
    createSeasonEventPlan
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.season = Object.assign({}, modules.season || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
