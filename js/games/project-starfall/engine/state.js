(function initProjectStarfallEngineState(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreSettings = (typeof require === 'function' ? require('../core/settings.js') : null) || global.ProjectStarfallCore || {};
  const EngineModules = global.ProjectStarfallEngineModules || {};
  const EnginePerformance = (typeof require === 'function' ? require('./performance.js') : null) || EngineModules.performance || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const normalizeAdminRateSetting = CoreSettings.normalizeAdminRate || function normalizeAdminRateFallback(value, fallback, min, max) {
    const low = Number.isFinite(Number(min)) ? Number(min) : 1;
    const high = Number.isFinite(Number(max)) ? Number(max) : 1000;
    const defaultValue = Number.isFinite(Number(fallback)) ? Number(fallback) : low;
    return Math.max(low, Math.min(high, Math.round(Number(value) || defaultValue)));
  };
  const normalizePerformanceDebugModeSetting = CoreSettings.normalizePerformanceDebugMode || function normalizePerformanceDebugModeFallback(value, modes, fallback) {
    const source = Array.isArray(modes) ? modes : [];
    const defaultValue = String(fallback || 'off');
    const mode = String(value || defaultValue);
    return source.includes(mode) ? mode : defaultValue;
  };

  const ADMIN_RATE_MIN = 1;
  const ADMIN_RATE_MAX = 1000;
  const DEFAULT_ADMIN_SETTINGS = Object.freeze({
    xpRate: 1,
    dropRate: 1,
    performanceDebugMode: 'off'
  });
  const PERFORMANCE_DEBUG_MODES = EnginePerformance.PERFORMANCE_DEBUG_MODES || Object.freeze(['off', 'fps', 'breakdown']);

  function createOnboardingState(value, data) {
    const source = value && typeof value === 'object' ? value : {};
    const sourceData = data || {};
    const stepIds = new Set((sourceData.ONBOARDING_STEPS || []).map((step) => step.id));
    const completedIds = Array.isArray(source.completedIds)
      ? source.completedIds.map(normalizeId).filter((id) => stepIds.has(id))
      : [];
    const progressById = {};
    Object.entries(source.progressById && typeof source.progressById === 'object' ? source.progressById : {}).forEach(([rawId, rawValue]) => {
      const id = normalizeId(rawId);
      const value = Math.max(0, Math.floor(Number(rawValue) || 0));
      if (id && stepIds.has(id) && value > 0 && !completedIds.includes(id)) progressById[id] = value;
    });
    return {
      hidden: !!source.hidden,
      completedIds: Array.from(new Set(completedIds)),
      progressById
    };
  }

  function onboardingStepMatchesEvent(step, type, payload) {
    if (!step || step.event !== type) return false;
    const data = payload || {};
    if (step.mapId && step.mapId !== data.mapId) return false;
    if (step.panelId && step.panelId !== data.panelId) return false;
    if (step.stationId && step.stationId !== data.stationId) return false;
    if (step.skillId && step.skillId !== data.skillId) return false;
    if (step.owner && step.owner !== data.owner) return false;
    if (step.advancedId && step.advancedId !== data.advancedId) return false;
    if (step.dungeonId && step.dungeonId !== data.dungeonId) return false;
    if (step.kind && step.kind !== data.kind) return false;
    return true;
  }

  function getOnboardingSnapshotCacheKey(onboarding, options) {
    const source = onboarding && typeof onboarding === 'object' ? onboarding : createOnboardingState(null);
    const settings = options || {};
    const revisions = settings.revisions || {};
    const steps = Array.isArray(settings.steps) ? settings.steps : [];
    let completedIds = '';
    if (Array.isArray(source.completedIds)) {
      source.completedIds.forEach((rawId) => {
        const id = normalizeId(rawId);
        if (id) completedIds += `${completedIds ? ',' : ''}${id}`;
      });
    }
    const progressById = Object.entries(source.progressById && typeof source.progressById === 'object' ? source.progressById : {})
      .map(([id, value]) => `${normalizeId(id)}:${Math.max(0, Math.floor(Number(value) || 0))}`)
      .filter((entry) => !entry.startsWith(':0'))
      .sort()
      .join(',');
    return [
      Number(revisions.session || 0),
      source.hidden ? 1 : 0,
      completedIds,
      progressById,
      steps.length
    ].join('|');
  }

  function createOnboardingSnapshot(onboarding, steps) {
    const source = onboarding && typeof onboarding === 'object' ? onboarding : createOnboardingState(null);
    const completed = new Set(source.completedIds || []);
    const progressById = source.progressById && typeof source.progressById === 'object' ? source.progressById : {};
    const stepSummaries = (steps || []).map((step) => {
      const goal = Math.max(1, Math.floor(Number(step && step.count || 1) || 1));
      const complete = completed.has(step.id);
      return Object.assign({}, step, {
        complete,
        progress: complete ? goal : Math.min(goal, Math.max(0, Math.floor(Number(progressById[step.id]) || 0))),
        goal
      });
    });
    return {
      hidden: !!source.hidden,
      completedIds: Array.isArray(source.completedIds) ? source.completedIds.slice() : [],
      completeCount: completed.size,
      total: stepSummaries.length,
      steps: stepSummaries,
      nextStep: stepSummaries.find((step) => !step.complete) || null
    };
  }

  function createOnboardingEventPlan(onboarding, candidates, type, payload) {
    const source = onboarding && typeof onboarding === 'object' ? onboarding : createOnboardingState(null);
    const completed = new Set(source.completedIds || []);
    const progressById = Object.assign({}, source.progressById && typeof source.progressById === 'object' ? source.progressById : {});
    const matches = (candidates || []).filter((step) =>
      !completed.has(step.id) &&
      onboardingStepMatchesEvent(step, type, payload || {}));
    if (!matches.length) return { changed: false, completedIds: source.completedIds || [], progressById };
    const increment = Math.max(1, Math.floor(Number(payload && payload.count || 1) || 1));
    matches.forEach((step) => {
      const goal = Math.max(1, Math.floor(Number(step && step.count || 1) || 1));
      const progress = Math.min(goal, Math.max(0, Math.floor(Number(progressById[step.id]) || 0)) + increment);
      if (progress >= goal) {
        completed.add(step.id);
        delete progressById[step.id];
      } else {
        progressById[step.id] = progress;
      }
    });
    return {
      changed: true,
      completedIds: Array.from(completed),
      progressById
    };
  }

  function createAudioState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const volume = clamp(Number(source.volume == null ? 0.42 : source.volume), 0, 1);
    return {
      enabled: !!source.enabled,
      volume
    };
  }

  function createClassMechanicsState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const links = Array.isArray(source.runeLinkIds) ? source.runeLinkIds.map(normalizeId).filter(Boolean).slice(0, 4) : [];
    return {
      fighterComboUid: normalizeId(source.fighterComboUid),
      fighterComboStacks: clamp(Number(source.fighterComboStacks || 0), 0, 3),
      fighterComboExpiresAt: Number(source.fighterComboExpiresAt || 0),
      guardianImpact: clamp(Number(source.guardianImpact || 0), 0, 120),
      duelistTargetUid: normalizeId(source.duelistTargetUid),
      duelistTempo: clamp(Number(source.duelistTempo || 0), 0, 5),
      duelistTempoExpiresAt: Number(source.duelistTempoExpiresAt || 0),
      beastMarkUid: normalizeId(source.beastMarkUid),
      beastMarkExpiresAt: Number(source.beastMarkExpiresAt || 0),
      runeLinkIds: links
    };
  }

  function normalizeAdminRate(value, fallback) {
    return normalizeAdminRateSetting(value, fallback || DEFAULT_ADMIN_SETTINGS.xpRate, ADMIN_RATE_MIN, ADMIN_RATE_MAX);
  }

  function createAdminSettings(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const settings = options || {};
    const defaults = settings.defaults || DEFAULT_ADMIN_SETTINGS;
    const modes = settings.performanceDebugModes || PERFORMANCE_DEBUG_MODES;
    return {
      xpRate: normalizeAdminRateSetting(source.xpRate, defaults.xpRate, ADMIN_RATE_MIN, ADMIN_RATE_MAX),
      dropRate: normalizeAdminRateSetting(source.dropRate, defaults.dropRate, ADMIN_RATE_MIN, ADMIN_RATE_MAX),
      performanceDebugMode: normalizePerformanceDebugModeSetting(source.performanceDebugMode, modes, defaults.performanceDebugMode)
    };
  }

  const api = {
    ADMIN_RATE_MIN,
    ADMIN_RATE_MAX,
    DEFAULT_ADMIN_SETTINGS,
    createOnboardingState,
    onboardingStepMatchesEvent,
    getOnboardingSnapshotCacheKey,
    createOnboardingSnapshot,
    createOnboardingEventPlan,
    createAudioState,
    createClassMechanicsState,
    normalizeAdminRate,
    createAdminSettings
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.state = Object.assign({}, modules.state || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
