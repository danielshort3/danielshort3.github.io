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
  const DAY_MS = 24 * 60 * 60 * 1000;
  const WEEK_MS = 7 * DAY_MS;
  const MAX_CLAIMED_CYCLE_HISTORY = 104;
  const MAX_STABILIZED_CYCLE_HISTORY = 104;
  const DEFAULT_MAX_STABILIZATION = 3;
  const WEEKDAY_LABELS = Object.freeze(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']);

  function getSeasonData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getDefaultSeasonId(options) {
    const data = getSeasonData(options);
    const active = (data.SEASONS || []).find((season) => season && season.active);
    return active ? active.id : ((data.SEASONS || [])[0] || {}).id || '';
  }

  function getSeasonCycleWindow(season, options) {
    const settings = options || {};
    const seasonId = normalizeId(season && season.id);
    const cadence = normalizeId(season && season.cadence) || 'lifetime';
    const nowMs = Number.isFinite(Number(settings.nowMs))
      ? Math.max(0, Number(settings.nowMs))
      : Date.now();
    if (!seasonId || cadence !== 'weekly') {
      return {
        cadence,
        cycleId: seasonId ? `${seasonId}:lifetime` : '',
        startedAt: 0,
        endsAt: 0,
        resetDayUtc: 0,
        resetHourUtc: 0
      };
    }

    const resetDayUtc = Math.min(6, Math.max(0, Math.floor(Number(season.resetDayUtc == null ? 1 : season.resetDayUtc) || 0)));
    const resetHourUtc = Math.min(23, Math.max(0, Math.floor(Number(season.resetHourUtc || 0) || 0)));
    const now = new Date(nowMs);
    const dayDelta = (now.getUTCDay() - resetDayUtc + 7) % 7;
    let startedAt = Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate() - dayDelta,
      resetHourUtc,
      0,
      0,
      0
    );
    if (startedAt > nowMs) startedAt -= WEEK_MS;
    return {
      cadence,
      cycleId: `${seasonId}:weekly:${startedAt}`,
      startedAt,
      endsAt: startedAt + WEEK_MS,
      resetDayUtc,
      resetHourUtc
    };
  }

  function getSeasonCycleClaimId(season, options) {
    return getSeasonCycleWindow(season, options).cycleId;
  }

  function normalizeObjectiveValues(value) {
    const objectiveValues = {};
    Object.entries(value && typeof value === 'object' ? value : {}).forEach(([id, count]) => {
      const key = normalizeId(id);
      if (key) objectiveValues[key] = Math.max(0, Math.floor(Number(count) || 0));
    });
    return objectiveValues;
  }

  function normalizeClaimedCycleIds(value) {
    const ids = Array.isArray(value) ? value.map(normalizeId).filter(Boolean) : [];
    return Array.from(new Set(ids)).slice(-MAX_CLAIMED_CYCLE_HISTORY);
  }

  function normalizeRewardCountsBySeason(value) {
    const counts = {};
    Object.entries(value && typeof value === 'object' ? value : {}).forEach(([seasonId, count]) => {
      const id = normalizeId(seasonId);
      const amount = Math.max(0, Math.floor(Number(count) || 0));
      if (id && amount > 0) counts[id] = amount;
    });
    return counts;
  }

  function getFractureDirectives(options) {
    const settings = options || {};
    const data = getSeasonData(settings);
    const seasonId = normalizeId(settings.seasonId || settings.season && settings.season.id);
    const directives = Array.isArray(data.FRACTURE_DIRECTIVES) ? data.FRACTURE_DIRECTIVES.filter(Boolean) : [];
    return seasonId
      ? directives.filter((directive) => normalizeId(directive && directive.seasonId) === seasonId)
      : directives;
  }

  function getFractureDirectiveById(directiveId, options) {
    return getById(getFractureDirectives(options), normalizeId(directiveId));
  }

  function getSeasonDirectiveOptions(state, season, options) {
    const settings = options || {};
    const seasonId = normalizeId(season && season.id)
      || normalizeId(settings.seasonId || settings.season && settings.season.id)
      || normalizeId(state && state.activeSeasonId)
      || normalizeId(getDefaultSeasonId(settings));
    return Object.assign({}, settings, { seasonId });
  }

  function getSeasonDirectiveMaxStabilization(directive, options) {
    const data = getSeasonData(options);
    const configured = Number(directive && directive.stabilization && directive.stabilization.maxSeals
      || data.FRACTURE_DIRECTIVE_MAX_STABILIZATION
      || DEFAULT_MAX_STABILIZATION);
    return Math.min(DEFAULT_MAX_STABILIZATION, Math.max(1, Math.floor(configured || DEFAULT_MAX_STABILIZATION)));
  }

  function normalizeStabilizationByAreaId(value, options) {
    const output = {};
    Object.entries(value && typeof value === 'object' ? value : {}).forEach(([areaId, count]) => {
      const id = normalizeId(areaId);
      const amount = Math.min(
        getSeasonDirectiveMaxStabilization(null, options),
        Math.max(0, Math.floor(Number(count) || 0))
      );
      if (id && amount > 0) output[id] = amount;
    });
    return output;
  }

  function normalizeStabilizedCycleIds(value) {
    const ids = Array.isArray(value) ? value.map(normalizeId).filter(Boolean) : [];
    return Array.from(new Set(ids)).slice(-MAX_STABILIZED_CYCLE_HISTORY);
  }

  function getObjectiveKeyForSeason(objective, index, options) {
    const settings = options || {};
    if (typeof settings.getObjectiveKey === 'function') return settings.getObjectiveKey(objective, index);
    return normalizeId(objective && objective.id) || `${normalizeId(objective && objective.type) || 'objective'}_${index}`;
  }

  function hasObjectiveProgress(state, objectives, options) {
    const source = state && typeof state === 'object' ? state : {};
    const values = source.objectiveValues && typeof source.objectiveValues === 'object' ? source.objectiveValues : {};
    return (objectives || []).some((objective, index) =>
      Math.max(0, Number(values[getObjectiveKeyForSeason(objective, index, options)] || 0)) > 0);
  }

  function resolveSeasonCycleWindow(source, season, options) {
    const current = getSeasonCycleWindow(season, options);
    const savedCycleId = normalizeId(source && source.cycleId);
    const savedStartedAt = Math.max(0, Number(source && source.cycleStartedAt || 0) || 0);
    const savedEndsAt = Math.max(0, Number(source && source.cycleEndsAt || 0) || 0);
    const seasonPrefix = `${normalizeId(season && season.id)}:`;
    const savedCycleIsValid = savedCycleId.startsWith(seasonPrefix) && savedEndsAt > savedStartedAt;
    if (savedCycleIsValid && savedStartedAt > current.startedAt) {
      return Object.assign({}, current, {
        cycleId: savedCycleId,
        startedAt: savedStartedAt,
        endsAt: savedEndsAt
      });
    }
    return current;
  }

  function createSeasonState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const claimedRewardIds = Array.isArray(source.claimedRewardIds)
      ? source.claimedRewardIds.map(normalizeId).filter(Boolean)
      : [];
    const activeSeason = getActiveSeasonData(source, options);
    const activeSeasonId = normalizeId(activeSeason && activeSeason.id) || normalizeId(source.activeSeasonId) || getDefaultSeasonId(options);
    const cycle = resolveSeasonCycleWindow(source, activeSeason, options);
    const savedCycleId = normalizeId(source.cycleId);
    const legacyState = !savedCycleId;
    const currentCycle = legacyState || savedCycleId === cycle.cycleId;
    const savedDirectiveId = currentCycle ? normalizeId(source.selectedDirectiveId) : '';
    const directiveOptions = getSeasonDirectiveOptions(source, activeSeason, options);
    const directives = getFractureDirectives(directiveOptions);
    const selectedDirectiveId = savedDirectiveId && getById(directives, savedDirectiveId)
      ? savedDirectiveId
      : '';
    const claimedCycleIds = normalizeClaimedCycleIds(source.claimedCycleIds);
    if (legacyState && activeSeasonId && claimedRewardIds.includes(activeSeasonId) && cycle.cycleId && !claimedCycleIds.includes(cycle.cycleId)) {
      claimedCycleIds.push(cycle.cycleId);
    }
    const objectiveValues = currentCycle ? normalizeObjectiveValues(source.objectiveValues) : {};
    if (legacyState && claimedRewardIds.includes(activeSeasonId) && activeSeason) {
      (activeSeason.objectives || []).forEach((objective, index) => {
        const key = normalizeId(objective && objective.id) || `${normalizeId(objective && objective.type) || 'objective'}_${index}`;
        const goal = Math.max(1, Number(objective && (objective.type === 'level' ? objective.level || objective.count : objective.count) || 1));
        objectiveValues[key] = Math.max(goal, Number(objectiveValues[key] || 0));
      });
    }
    const rewardCountsBySeason = normalizeRewardCountsBySeason(source.rewardCountsBySeason);
    claimedRewardIds.forEach((seasonId) => {
      rewardCountsBySeason[seasonId] = Math.max(1, Number(rewardCountsBySeason[seasonId] || 0));
    });
    const savedTotalClaims = Math.max(0, Math.floor(Number(source.totalRewardsClaimed || 0) || 0));
    if (!Object.keys(rewardCountsBySeason).length && savedTotalClaims > 0 && activeSeasonId) {
      rewardCountsBySeason[activeSeasonId] = savedTotalClaims;
    }
    const inferredClaims = Math.max(
      claimedCycleIds.length,
      Object.values(rewardCountsBySeason).reduce((sum, count) => sum + Math.max(0, Number(count) || 0), 0)
    );
    return {
      activeSeasonId,
      cycleId: cycle.cycleId,
      cycleStartedAt: cycle.startedAt,
      cycleEndsAt: cycle.endsAt,
      selectedDirectiveId,
      objectiveValues,
      claimedRewardIds: Array.from(new Set(claimedRewardIds)),
      claimedCycleIds: normalizeClaimedCycleIds(claimedCycleIds),
      stabilizationByAreaId: normalizeStabilizationByAreaId(source.stabilizationByAreaId, options),
      stabilizedCycleIds: normalizeStabilizedCycleIds(source.stabilizedCycleIds),
      rewardCountsBySeason,
      totalRewardsClaimed: Math.max(inferredClaims, savedTotalClaims),
      lastRewardClaimedAt: Math.max(0, Number(source.lastRewardClaimedAt || 0) || 0)
    };
  }

  function syncSeasonState(value, season, options) {
    const source = value && typeof value === 'object' ? value : {};
    const settings = Object.assign({}, options || {});
    if (season && season.id) {
      source.activeSeasonId = season.id;
      const data = getSeasonData(settings);
      if (!(data.SEASONS || []).some((entry) => entry && entry.id === season.id)) {
        settings.data = Object.assign({}, data, { SEASONS: [season].concat(data.SEASONS || []) });
      }
    }
    const normalized = createSeasonState(source, settings);
    Object.keys(source).forEach((key) => {
      if (!Object.prototype.hasOwnProperty.call(normalized, key)) delete source[key];
    });
    Object.assign(source, normalized);
    return source;
  }

  function getActiveSeasonData(state, options) {
    const data = getSeasonData(options);
    const source = state && typeof state === 'object' ? state : {};
    const season = source.activeSeasonId ? getById(data.SEASONS || [], source.activeSeasonId) : null;
    return season && season.active !== false
      ? season
      : getById(data.SEASONS || [], getDefaultSeasonId(options));
  }

  function getSelectedSeasonDirective(state, options) {
    const source = state && typeof state === 'object' ? state : {};
    return getFractureDirectiveById(source.selectedDirectiveId, getSeasonDirectiveOptions(source, null, options));
  }

  function getDirectiveEligibility(directive, options) {
    const settings = options || {};
    if (!directive || !normalizeId(directive.id)) return { eligible: false, lockedReason: 'Directive unavailable.' };
    if (directive.active === false) return { eligible: false, lockedReason: 'Directive is not active this cycle.' };
    const playerLevelValue = settings.playerLevel != null
      ? settings.playerLevel
      : settings.player && settings.player.level != null
        ? settings.player.level
        : null;
    const playerLevel = playerLevelValue == null ? null : Math.max(0, Math.floor(Number(playerLevelValue) || 0));
    const minLevel = Math.max(1, Math.floor(Number(directive.minLevel || 1) || 1));
    if (playerLevel != null && playerLevel < minLevel) {
      return { eligible: false, lockedReason: `Reach level ${minLevel} to accept this directive.` };
    }
    if (Array.isArray(settings.unlockedMapIds)) {
      const unlockedMapIds = new Set(settings.unlockedMapIds.map(normalizeId).filter(Boolean));
      const missingMapId = (directive.requiredMapIds || []).map(normalizeId).find((mapId) => mapId && !unlockedMapIds.has(mapId));
      if (missingMapId) return { eligible: false, lockedReason: 'Open the full directive route on the Atlas first.' };
    }
    if (Array.isArray(settings.unlockedDungeonIds)) {
      const unlockedDungeonIds = new Set(settings.unlockedDungeonIds.map(normalizeId).filter(Boolean));
      const missingDungeonId = (directive.requiredDungeonIds || []).map(normalizeId).find((dungeonId) => dungeonId && !unlockedDungeonIds.has(dungeonId));
      if (missingDungeonId) return { eligible: false, lockedReason: 'Unlock the required expedition first.' };
    }
    return { eligible: true, lockedReason: '' };
  }

  function getSeasonEffectiveObjectives(state, season, options) {
    const settings = getSeasonDirectiveOptions(state, season, options);
    const directive = getSelectedSeasonDirective(state, settings);
    if (directive) return directive.objectives || [];
    if (settings.requireDirectiveSelection && getFractureDirectives(settings).length) return [];
    return season && season.objectives || [];
  }

  function getSeasonObjectiveOwner(state, season, options) {
    const settings = getSeasonDirectiveOptions(state, season, options);
    const directive = getSelectedSeasonDirective(state, settings);
    if (!directive) {
      return settings.requireDirectiveSelection && getFractureDirectives(settings).length
        ? { id: `${normalizeId(season && season.id)}:directive:unselected`, objectives: [] }
        : season;
    }
    return {
      id: `${normalizeId(season && season.id)}:directive:${directive.id}`,
      objectives: directive.objectives || []
    };
  }

  function getSeasonDirectiveProgressStarted(state, season, options) {
    const directive = getSelectedSeasonDirective(state, getSeasonDirectiveOptions(state, season, options));
    const objectives = directive ? directive.objectives || [] : season && season.objectives || [];
    return hasObjectiveProgress(state, objectives, options);
  }

  function getSeasonDirectiveChoices(state, season, options) {
    const source = state && typeof state === 'object' ? state : {};
    const settings = getSeasonDirectiveOptions(source, season, options);
    const selected = getSelectedSeasonDirective(source, settings);
    const progressStarted = getSeasonDirectiveProgressStarted(source, season, settings);
    const cycle = resolveSeasonCycleWindow(source, season, settings);
    const rewardClaimed = Array.isArray(source.claimedCycleIds) && source.claimedCycleIds.includes(cycle.cycleId);
    return getFractureDirectives(settings).map((directive) => {
      const eligibility = getDirectiveEligibility(directive, settings);
      const isSelected = !!selected && selected.id === directive.id;
      const switchingLocked = progressStarted && !isSelected;
      const canSelect = eligibility.eligible && !rewardClaimed && !switchingLocked;
      let lockedReason = eligibility.lockedReason;
      if (!lockedReason && rewardClaimed) lockedReason = 'This cycle reward has already been claimed.';
      if (!lockedReason && switchingLocked) {
        lockedReason = selected
          ? 'Progress has started; this cycle directive is locked.'
          : 'Legacy weekly progress has started; finish this cycle before choosing a directive.';
      }
      return Object.assign({}, directive, {
        selected: isSelected,
        eligible: eligibility.eligible,
        canSelect,
        selectionLocked: switchingLocked,
        progressStarted: isSelected && progressStarted,
        lockedReason
      });
    });
  }

  function clearSeasonDirectiveObjectiveValues(state, options) {
    const source = state && typeof state === 'object' ? state : {};
    const values = source.objectiveValues && typeof source.objectiveValues === 'object' ? source.objectiveValues : {};
    getFractureDirectives(getSeasonDirectiveOptions(source, null, options)).forEach((directive) => {
      (directive.objectives || []).forEach((objective, index) => {
        delete values[getObjectiveKeyForSeason(objective, index, options)];
      });
    });
    source.objectiveValues = values;
  }

  function selectSeasonDirective(state, season, directiveId, options) {
    const source = syncSeasonState(state, season, options || {});
    const settings = getSeasonDirectiveOptions(source, season, options);
    const targetId = normalizeId(directiveId);
    const directive = getFractureDirectiveById(targetId, settings);
    if (!season || !directive) {
      return { ok: false, changed: false, reason: 'invalidDirective', directive: null, state: source };
    }
    const selected = getSelectedSeasonDirective(source, settings);
    if (selected && selected.id === directive.id) {
      return { ok: true, changed: false, reason: 'alreadySelected', directive, state: source };
    }
    const choice = getSeasonDirectiveChoices(source, season, settings).find((entry) => entry.id === directive.id);
    if (!choice || !choice.canSelect) {
      return {
        ok: false,
        changed: false,
        reason: choice && choice.selectionLocked ? 'progressLocked' : choice && !choice.eligible ? 'ineligible' : 'cycleClosed',
        lockedReason: choice && choice.lockedReason || 'Directive cannot be selected.',
        directive,
        state: source
      };
    }
    clearSeasonDirectiveObjectiveValues(source, settings);
    source.selectedDirectiveId = directive.id;
    return { ok: true, changed: true, reason: 'selected', directive, state: source };
  }

  function getSeasonObjectivesByType(season, type) {
    const seasonId = normalizeId(season && season.id);
    const objectiveType = normalizeId(type);
    if (!seasonId || !objectiveType) return [];
    const objectiveSignature = (season.objectives || []).map((objective, index) => [
      normalizeId(objective && objective.id) || index,
      normalizeId(objective && objective.type),
      normalizeId(objective && objective.mapId),
      normalizeId(objective && objective.bossId),
      normalizeId(objective && objective.dungeonId),
      Math.max(1, Number(objective && objective.count || 1))
    ].join(':')).join('|');
    const cacheKey = `${seasonId}:${objectiveSignature}`;
    if (!seasonObjectiveTypeMapCache.has(cacheKey)) {
      const objectiveMap = new Map();
      (season.objectives || []).forEach((objective, index) => {
        const id = normalizeId(objective && objective.type);
        if (!id) return;
        if (!objectiveMap.has(id)) objectiveMap.set(id, []);
        objectiveMap.get(id).push({ objective, index });
      });
      seasonObjectiveTypeMapCache.set(cacheKey, objectiveMap);
    }
    const objectiveMap = seasonObjectiveTypeMapCache.get(cacheKey);
    return objectiveMap ? objectiveMap.get(objectiveType) || [] : [];
  }

  function createSeasonSnapshot(state, season, options) {
    if (!season) return { activeSeason: null, objectives: [], complete: false, rewardClaimed: false };
    const settings = getSeasonDirectiveOptions(state, season, options);
    const createObjectiveStatuses = settings.createObjectiveStatuses || function createObjectiveStatusesFallback() {
      return [];
    };
    const source = syncSeasonState(state, season, settings);
    const entry = { objectiveValues: source.objectiveValues || {} };
    const selectedDirective = getSelectedSeasonDirective(source, settings);
    const effectiveObjectives = getSeasonEffectiveObjectives(source, season, settings);
    const objectives = createObjectiveStatuses(entry, effectiveObjectives);
    const directiveChoices = getSeasonDirectiveChoices(source, season, settings);
    const stabilizationByAreaId = Object.assign({}, source.stabilizationByAreaId || {});
    const selectedAreaId = normalizeId(selectedDirective && selectedDirective.areaId);
    const stabilizationMax = getSeasonDirectiveMaxStabilization(selectedDirective, settings);
    const cycle = resolveSeasonCycleWindow(source, season, settings);
    const rewardClaimed = Array.isArray(source.claimedCycleIds) && source.claimedCycleIds.includes(cycle.cycleId);
    const nowMs = Number.isFinite(Number(settings.nowMs)) ? Math.max(0, Number(settings.nowMs)) : Date.now();
    const remainingMs = cycle.endsAt ? Math.max(0, cycle.endsAt - nowMs) : 0;
    const remainingDays = Math.floor(remainingMs / DAY_MS);
    const remainingHours = Math.floor((remainingMs % DAY_MS) / (60 * 60 * 1000));
    const remainingMinutes = Math.max(1, Math.ceil((remainingMs % (60 * 60 * 1000)) / (60 * 1000)));
    const remainingLabel = remainingDays > 0
      ? `${remainingDays}d ${remainingHours}h`
      : remainingHours > 0
        ? `${remainingHours}h ${remainingMinutes}m`
        : `${remainingMinutes}m`;
    const seasonRewardsClaimed = Math.max(0, Math.floor(Number(source.rewardCountsBySeason && source.rewardCountsBySeason[season.id] || 0) || 0));
    return {
      activeSeason: season,
      directiveChoices,
      selectedDirective,
      selectedDirectiveId: selectedDirective ? selectedDirective.id : '',
      directiveSelectionRequired: !!settings.requireDirectiveSelection && directiveChoices.length > 0 && !selectedDirective,
      legacyObjectivesActive: !selectedDirective && effectiveObjectives === (season.objectives || []),
      directiveProgressStarted: getSeasonDirectiveProgressStarted(source, season, settings),
      stabilizationByAreaId,
      selectedAreaStabilization: selectedAreaId ? Math.max(0, Number(stabilizationByAreaId[selectedAreaId] || 0)) : 0,
      stabilizationMax,
      objectives,
      complete: objectives.length > 0 && objectives.every((objective) => objective.complete),
      rewardClaimed,
      cadence: cycle.cadence,
      cadenceLabel: cycle.cadence === 'weekly' ? 'Weekly operation' : 'Season operation',
      cycleId: cycle.cycleId,
      cycleStartedAt: cycle.startedAt,
      cycleEndsAt: cycle.endsAt,
      resetAt: cycle.endsAt,
      resetLabel: cycle.endsAt ? `Resets in ${remainingLabel}` : '',
      resetScheduleLabel: cycle.cadence === 'weekly'
        ? `${WEEKDAY_LABELS[cycle.resetDayUtc]} ${String(cycle.resetHourUtc).padStart(2, '0')}:00 UTC`
        : '',
      totalRewardsClaimed: Math.max(0, Math.floor(Number(source.totalRewardsClaimed || 0) || 0)),
      seasonRewardsClaimed,
      firstCompletionRewardAvailable: seasonRewardsClaimed === 0 && !!season.firstCompletionRewards
    };
  }

  function createSeasonEventPlan(state, season, type, payload, options) {
    const settings = options || {};
    const source = syncSeasonState(state, season, settings);
    if (!season || Array.isArray(source.claimedCycleIds) && source.claimedCycleIds.includes(source.cycleId)) {
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
    const objectiveOwner = getSeasonObjectiveOwner(source, season, settings);
    const objectives = readObjectives(objectiveOwner, type);
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
      changes,
      selectedDirectiveId: normalizeId(source.selectedDirectiveId)
    };
  }

  function createSeasonStabilizationPlan(state, season, options) {
    const settings = options || {};
    const source = syncSeasonState(state, season, settings);
    const directive = getSelectedSeasonDirective(source, settings);
    const cycleId = normalizeId(source.cycleId);
    const areaId = normalizeId(directive && directive.areaId);
    const maxSeals = getSeasonDirectiveMaxStabilization(directive, settings);
    const before = areaId ? Math.max(0, Number(source.stabilizationByAreaId && source.stabilizationByAreaId[areaId] || 0)) : 0;
    const basePlan = {
      changed: false,
      reason: '',
      cycleId,
      directiveId: normalizeId(directive && directive.id),
      areaId,
      before,
      next: before,
      maxSeals
    };
    if (!season || !directive || !cycleId || !areaId) return Object.assign(basePlan, { reason: 'noDirectiveSelected' });
    if (Array.isArray(source.claimedCycleIds) && source.claimedCycleIds.includes(cycleId)) {
      return Object.assign(basePlan, { reason: 'cycleClosed' });
    }
    if (Array.isArray(source.stabilizedCycleIds) && source.stabilizedCycleIds.includes(cycleId)) {
      return Object.assign(basePlan, { reason: 'alreadyStabilized' });
    }
    const entry = { objectiveValues: source.objectiveValues || {} };
    const createObjectiveStatuses = settings.createObjectiveStatuses;
    const statuses = typeof createObjectiveStatuses === 'function'
      ? createObjectiveStatuses(entry, directive.objectives || [])
      : (directive.objectives || []).map((objective, index) => {
        const value = Math.max(0, Number(entry.objectiveValues[getObjectiveKeyForSeason(objective, index, settings)] || 0));
        const goal = Math.max(1, Number(objective && (objective.count || objective.level) || 1));
        return { complete: value >= goal };
      });
    if (!statuses.length || !statuses.every((status) => status && status.complete)) {
      return Object.assign(basePlan, { reason: 'incomplete' });
    }
    if (before >= maxSeals) return Object.assign(basePlan, { reason: 'stabilizationCapped' });
    const next = Math.min(maxSeals, before + 1);
    const stabilizationByAreaId = Object.assign({}, source.stabilizationByAreaId || {}, { [areaId]: next });
    const stabilizedCycleIds = normalizeStabilizedCycleIds([].concat(source.stabilizedCycleIds || [], cycleId));
    return Object.assign(basePlan, {
      changed: next > before,
      reason: next > before ? 'stabilized' : 'stabilizationCapped',
      next,
      stabilizationByAreaId,
      stabilizedCycleIds
    });
  }

  function applySeasonStabilizationPlan(state, plan) {
    const source = state && typeof state === 'object' ? state : {};
    const change = plan && typeof plan === 'object' ? plan : {};
    const cycleId = normalizeId(change.cycleId);
    const directiveId = normalizeId(change.directiveId);
    const areaId = normalizeId(change.areaId);
    if (!change.changed || !cycleId || !directiveId || !areaId) return false;
    if (normalizeId(source.cycleId) !== cycleId || normalizeId(source.selectedDirectiveId) !== directiveId) return false;
    if (Array.isArray(source.stabilizedCycleIds) && source.stabilizedCycleIds.includes(cycleId)) return false;
    const maxSeals = Math.min(DEFAULT_MAX_STABILIZATION, Math.max(1, Math.floor(Number(change.maxSeals) || DEFAULT_MAX_STABILIZATION)));
    const current = Math.min(maxSeals, Math.max(0, Math.floor(Number(source.stabilizationByAreaId && source.stabilizationByAreaId[areaId] || 0))));
    const next = Math.min(maxSeals, Math.max(current, Math.floor(Number(change.next) || 0)));
    source.stabilizationByAreaId = Object.assign({}, source.stabilizationByAreaId || {}, { [areaId]: next });
    source.stabilizedCycleIds = normalizeStabilizedCycleIds([].concat(source.stabilizedCycleIds || [], cycleId));
    return true;
  }

  const api = {
    getDefaultSeasonId,
    getSeasonCycleWindow,
    getSeasonCycleClaimId,
    createSeasonState,
    syncSeasonState,
    getActiveSeasonData,
    getFractureDirectives,
    getFractureDirectiveById,
    getSeasonDirectiveMaxStabilization,
    getSelectedSeasonDirective,
    getSeasonEffectiveObjectives,
    getSeasonDirectiveProgressStarted,
    getSeasonDirectiveChoices,
    selectSeasonDirective,
    getSeasonObjectivesByType,
    createSeasonSnapshot,
    createSeasonEventPlan,
    createSeasonStabilizationPlan,
    applySeasonStabilizationPlan
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.season = Object.assign({}, modules.season || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
