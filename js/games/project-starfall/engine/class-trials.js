(function initProjectStarfallEngineClassTrials(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (Array.isArray(items) ? items : []).find((item) => item && item.id === id) || null;
  };

  function getClassTrialData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createTrialInstanceState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const trialId = normalizeId(source.trialId);
    return {
      active: !!source.active && !!trialId,
      trialId,
      returnMapId: normalizeId(source.returnMapId) || 'starfallCrossing',
      returnX: Number.isFinite(Number(source.returnX)) ? Number(source.returnX) : 1845,
      returnY: Number.isFinite(Number(source.returnY)) ? Number(source.returnY) : 360,
      startedAt: Number(source.startedAt || 0)
    };
  }

  function getActiveClassTrial(progress, options) {
    const data = getClassTrialData(options);
    const state = progress && typeof progress === 'object' ? progress : {};
    return getById(data.CLASS_TRIALS || [], normalizeId(state.activeTrialId));
  }

  function isClassTrialComplete(progress, advancedId) {
    const state = progress && typeof progress === 'object' ? progress : {};
    return !!(state.completedTrials && state.completedTrials[advancedId]);
  }

  function isTrialInstanceActive(instance, options) {
    return createTrialInstanceSnapshot(instance, options).active;
  }

  function createTrialInstanceSnapshot(instance, options) {
    const settings = options || {};
    const data = getClassTrialData(settings);
    const state = createTrialInstanceState(instance);
    const trial = state.active ? getById(data.CLASS_TRIALS || [], state.trialId) : null;
    const runtime = settings.runtime || {};
    return {
      active: !!(state.active && trial),
      trialId: state.trialId,
      returnMapId: state.returnMapId,
      runtimeMapId: runtime && runtime.isTrialInstance ? runtime.id : '',
      runtimeMapName: runtime && runtime.isTrialInstance ? runtime.name : '',
      title: trial ? trial.title : ''
    };
  }

  function getDefaultClassTrialForPlayer(player, progress, options) {
    const data = getClassTrialData(options);
    const activePlayer = player && typeof player === 'object' ? player : {};
    const state = progress && typeof progress === 'object' ? progress : {};
    if (!activePlayer.classId || activePlayer.advancedClassId) return null;
    const completedTrials = state.completedTrials && typeof state.completedTrials === 'object'
      ? state.completedTrials
      : {};
    return (data.CLASS_TRIALS || []).find((trial) =>
      trial &&
      trial.baseClass === activePlayer.classId &&
      !completedTrials[trial.advancedId] &&
      Number(activePlayer.level || 1) >= Number(trial.levelRequirement || 20)) || null;
  }

  const api = {
    createTrialInstanceState,
    getActiveClassTrial,
    isClassTrialComplete,
    isTrialInstanceActive,
    createTrialInstanceSnapshot,
    getDefaultClassTrialForPlayer
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.classTrials = Object.assign({}, modules.classTrials || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
