(function initProjectStarfallEngineAccomplishments(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getAccomplishmentData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createProgressEntry(value) {
    const source = value && typeof value === 'object' ? value : {};
    const objectiveValues = source.objectiveValues && typeof source.objectiveValues === 'object' ? source.objectiveValues : {};
    return {
      objectiveValues: Object.keys(objectiveValues).reduce((values, key) => {
        values[key] = Math.max(0, Number(objectiveValues[key]) || 0);
        return values;
      }, {}),
      completedAt: Number(source.completedAt || 0)
    };
  }

  function ensureAccomplishmentEntry(state, accomplishmentId) {
    const source = state && typeof state === 'object' ? state : {};
    source.accomplishmentProgress = source.accomplishmentProgress && typeof source.accomplishmentProgress === 'object'
      ? source.accomplishmentProgress
      : {};
    const id = normalizeId(accomplishmentId);
    if (!source.accomplishmentProgress[id]) source.accomplishmentProgress[id] = createProgressEntry();
    return source.accomplishmentProgress[id];
  }

  function createAccomplishmentState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const accomplishmentProgress = {};
    Object.entries(source.accomplishmentProgress || {}).forEach(([id, entry]) => {
      accomplishmentProgress[id] = createProgressEntry(entry);
    });
    const claimedIds = Array.isArray(source.claimedIds)
      ? source.claimedIds.map(normalizeId).filter(Boolean)
      : [];
    return {
      accomplishmentProgress,
      claimedIds: Array.from(new Set(claimedIds))
    };
  }

  function createAccomplishmentSyncStateKey(player, mapId, progress, dungeons, options) {
    const data = getAccomplishmentData(options);
    const sourcePlayer = player && typeof player === 'object' ? player : {};
    const sourceProgress = progress && typeof progress === 'object' ? progress : {};
    const sourceDungeons = dungeons && typeof dungeons === 'object' ? dungeons : {};
    const completedTrials = Object.keys(sourceProgress.completedTrials || {})
      .filter((id) => sourceProgress.completedTrials[id])
      .map(normalizeId)
      .sort()
      .join(',');
    const completedDungeons = Array.isArray(sourceDungeons.completedDungeonIds)
      ? sourceDungeons.completedDungeonIds.map(normalizeId).filter(Boolean).sort().join(',')
      : '';
    return [
      data.ACCOMPLISHMENTS || [],
      (data.ACCOMPLISHMENTS || []).length,
      Math.max(1, Math.floor(Number(sourcePlayer.level || 1) || 1)),
      normalizeId(mapId),
      normalizeId(sourcePlayer.classId),
      normalizeId(sourcePlayer.advancedClassId),
      completedTrials,
      completedDungeons
    ];
  }

  function createAccomplishmentSummary(accomplishment, state, options) {
    const settings = options || {};
    if (!accomplishment) return null;
    const source = state && typeof state === 'object' ? state : createAccomplishmentState();
    const createObjectiveStatuses = settings.createObjectiveStatuses || function createObjectiveStatusesFallback() {
      return [];
    };
    const entry = ensureAccomplishmentEntry(source, accomplishment.id);
    const objectives = createObjectiveStatuses(entry, accomplishment.objectives);
    const complete = objectives.length > 0 && objectives.every((objective) => objective.complete);
    const claimed = Array.isArray(source.claimedIds) && source.claimedIds.includes(accomplishment.id);
    return Object.assign({}, accomplishment, {
      objectives,
      complete,
      claimed,
      claimable: complete && !claimed
    });
  }

  function getAccomplishmentSnapshotCacheKey(state, permanentStats, options) {
    const data = getAccomplishmentData(options);
    const source = state && typeof state === 'object' ? state : {};
    const progress = source.accomplishmentProgress && typeof source.accomplishmentProgress === 'object' ? source.accomplishmentProgress : {};
    const progressKey = Object.keys(progress).sort().map((id) => {
      const entry = progress[id] || {};
      const values = entry.objectiveValues && typeof entry.objectiveValues === 'object' ? entry.objectiveValues : {};
      const valueKey = Object.keys(values).sort().map((key) => `${normalizeId(key)}:${Number(values[key] || 0)}`).join(',');
      return `${normalizeId(id)}:${entry.complete ? 1 : 0}:${Number(entry.completedAt || 0)}:${valueKey}`;
    }).join('|');
    const claimed = Array.isArray(source.claimedIds)
      ? source.claimedIds.map(normalizeId).filter(Boolean).sort().join(',')
      : '';
    const stats = permanentStats && typeof permanentStats === 'object' ? permanentStats : {};
    const permanentKey = Object.keys(stats).sort().map((key) => `${normalizeId(key)}:${Number(stats[key] || 0)}`).join(',');
    return [
      data.ACCOMPLISHMENTS || [],
      (data.ACCOMPLISHMENTS || []).length,
      progressKey,
      claimed,
      permanentKey
    ];
  }

  function createAccomplishmentSnapshot(accomplishments, state, permanentStats, options) {
    const source = state && typeof state === 'object' ? state : createAccomplishmentState();
    const list = Array.isArray(accomplishments) ? accomplishments : [];
    const summaries = list.map((accomplishment) => createAccomplishmentSummary(accomplishment, source, options));
    return {
      accomplishments: summaries,
      claimableCount: summaries.filter((item) => item && item.claimable).length,
      claimedIds: Array.isArray(source.claimedIds) ? source.claimedIds.slice() : [],
      permanentStats: Object.assign({}, permanentStats || {})
    };
  }

  const api = {
    createProgressEntry,
    ensureAccomplishmentEntry,
    createAccomplishmentState,
    createAccomplishmentSyncStateKey,
    createAccomplishmentSummary,
    getAccomplishmentSnapshotCacheKey,
    createAccomplishmentSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.accomplishments = Object.assign({}, modules.accomplishments || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
