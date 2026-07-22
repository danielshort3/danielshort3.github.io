(function initProjectStarfallEngineMapMechanics(global) {
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
  const seededUnit = CoreMath.seededUnit || function seededUnitFallback(seed, salt) {
    let hash = 2166136261;
    const text = `${String(seed || '')}:${String(salt || '')}`;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    hash ^= hash << 13;
    hash ^= hash >>> 17;
    hash ^= hash << 5;
    return ((hash >>> 0) % 10000) / 10000;
  };
  const seededPick = CoreMath.seededPick || function seededPickFallback(items, seed, salt) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return '';
    return options[Math.floor(seededUnit(seed, salt) * options.length) % options.length];
  };
  const RIFT_OPERATION_VERSION = 1;
  const RIFT_OPERATION_DURATION_MS = 6 * 60 * 1000;
  const RIFT_OPERATION_WEEK_MS = 7 * 24 * 60 * 60 * 1000;
  const RIFT_OPERATION_WEEK_ANCHOR_MS = Date.UTC(1970, 0, 5);
  const RIFT_FIRST_CLEAR_REWARD = Object.freeze({
    currency: 1800,
    materials: Object.freeze({ riftSplinter: 12, cubeFragment: 3 })
  });

  function getMapMechanicData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function roundRiftScale(value) {
    return Math.round(Math.max(0, Number(value || 0)) * 10000) / 10000;
  }

  function createRiftTierRuntimeEffects(tier) {
    const normalizedTier = Math.max(1, Math.floor(Number(tier || 1) || 1));
    const tierSteps = normalizedTier - 1;
    return {
      tier: normalizedTier,
      mutationPotencyScale: roundRiftScale(1 + Math.min(1, tierSteps * 0.02)),
      enemyHealthScale: roundRiftScale(1 + tierSteps * 0.045),
      enemyDamageScale: roundRiftScale(1 + tierSteps * 0.03),
      rewardScale: roundRiftScale(1 + tierSteps * 0.035)
    };
  }

  function getRiftMutationScale(source, key, potency, influence) {
    const authoredScale = Math.max(0.1, Number(source && source[key] || 1) || 1);
    return Math.max(0.1, 1 + (authoredScale - 1) * Math.max(1, Number(potency || 1)) * Math.max(0, Number(influence == null ? 1 : influence)));
  }

  function getMapMechanicDefinitionById(mapId, options) {
    const data = getMapMechanicData(options);
    const id = normalizeId(mapId);
    return id && data.MAP_MECHANIC_DEFINITIONS && data.MAP_MECHANIC_DEFINITIONS[id] || null;
  }

  function normalizeMapMechanicSectionId(definition, sectionId) {
    const id = normalizeId(sectionId);
    if (!definition || !id) return '';
    const sections = Array.isArray(definition.sections) ? definition.sections : [];
    return sections.some((section) => section && section.id === id) ? id : '';
  }

  function getDefaultMapMechanicSectionId(definition) {
    if (!definition) return '';
    const activeIds = Array.isArray(definition.activeSectionIds) ? definition.activeSectionIds : [];
    return activeIds.find((sectionId) => normalizeMapMechanicSectionId(definition, sectionId)) ||
      normalizeMapMechanicSectionId(definition, definition.objectiveSectionId) ||
      (definition.sections && definition.sections[0] && definition.sections[0].id || '');
  }

  function createMapMechanicEntryState(definition, value) {
    const source = value && typeof value === 'object' ? value : {};
    const activeSectionId = normalizeMapMechanicSectionId(definition, source.activeSectionId) || getDefaultMapMechanicSectionId(definition);
    const sectionHits = {};
    Object.entries(source.sectionHits || {}).forEach(([sectionId, amount]) => {
      const id = normalizeMapMechanicSectionId(definition, sectionId);
      if (id) sectionHits[id] = Math.max(0, Math.floor(Number(amount || 0) || 0));
    });
    const cycleSectionIds = Array.isArray(source.cycleSectionIds)
      ? source.cycleSectionIds.map((sectionId) => normalizeMapMechanicSectionId(definition, sectionId)).filter(Boolean)
      : [];
    const activeIds = Array.isArray(definition && definition.activeSectionIds) ? definition.activeSectionIds : [];
    const activeSectionIndex = activeIds.findIndex((sectionId) => sectionId === activeSectionId);
    return {
      activeSectionId,
      activeSectionIndex: activeSectionIndex >= 0 ? activeSectionIndex : 0,
      progress: Math.max(0, Number(source.progress || 0)),
      completedCycles: Math.max(0, Math.floor(Number(source.completedCycles || 0) || 0)),
      eventCount: Math.max(0, Math.floor(Number(source.eventCount || 0) || 0)),
      objectiveCount: Math.max(0, Math.floor(Number(source.objectiveCount || 0) || 0)),
      surgeCount: Math.max(0, Math.floor(Number(source.surgeCount || 0) || 0)),
      surgeActiveUntil: Math.max(0, Number(source.surgeActiveUntil || 0)),
      lastSectionId: normalizeMapMechanicSectionId(definition, source.lastSectionId),
      repeatCount: Math.max(0, Math.floor(Number(source.repeatCount || 0) || 0)),
      antiCampStacks: clamp(Math.floor(Number(source.antiCampStacks || 0) || 0), 0, 8),
      rewardScale: clamp(Number(source.rewardScale || 1) || 1, Number(definition && definition.minimumRewardScale || 0.5), 1),
      sectionHits,
      cycleSectionIds: Array.from(new Set(cycleSectionIds)),
      lastCompletedAt: Math.max(0, Number(source.lastCompletedAt || 0))
    };
  }

  function createMapMechanicState(value, options) {
    const data = getMapMechanicData(options);
    const source = value && typeof value === 'object' ? value : {};
    const sourceByMapId = source.byMapId && typeof source.byMapId === 'object'
      ? source.byMapId
      : source.mapMechanicsByMapId && typeof source.mapMechanicsByMapId === 'object'
        ? source.mapMechanicsByMapId
        : source;
    const byMapId = {};
    Object.entries(data.MAP_MECHANIC_DEFINITIONS || {}).forEach(([mapId, definition]) => {
      byMapId[mapId] = createMapMechanicEntryState(definition, sourceByMapId[mapId]);
    });
    return { byMapId };
  }

  function normalizeRiftOperationNow(nowMs) {
    const value = Number(nowMs);
    return Number.isFinite(value) && value >= 0 ? Math.floor(value) : Date.now();
  }

  function getRiftOperationWeekWindow(nowMs) {
    const now = normalizeRiftOperationNow(nowMs);
    const weekIndex = Math.floor((now - RIFT_OPERATION_WEEK_ANCHOR_MS) / RIFT_OPERATION_WEEK_MS);
    const startedAt = RIFT_OPERATION_WEEK_ANCHOR_MS + weekIndex * RIFT_OPERATION_WEEK_MS;
    const endsAt = startedAt + RIFT_OPERATION_WEEK_MS;
    return {
      cycleId: `rift-week-${startedAt}`,
      startedAt,
      endsAt
    };
  }

  function createRiftWeeklyState(value, nowMs) {
    const source = value && typeof value === 'object' ? value : {};
    const current = getRiftOperationWeekWindow(nowMs);
    const sourceStartedAt = Math.max(0, Math.floor(Number(source.startedAt || 0) || 0));
    const preserveSource = sourceStartedAt >= current.startedAt;
    if (!preserveSource) {
      return {
        cycleId: current.cycleId,
        startedAt: current.startedAt,
        endsAt: current.endsAt,
        bestTier: 1,
        bestScore: 0,
        runs: 0,
        clears: 0,
        rewardClaimed: false,
        rewardClaimedAt: 0
      };
    }
    const startedAt = sourceStartedAt;
    const endsAt = Math.max(startedAt + RIFT_OPERATION_WEEK_MS, Math.floor(Number(source.endsAt || 0) || 0));
    return {
      cycleId: normalizeId(source.cycleId) || `rift-week-${startedAt}`,
      startedAt,
      endsAt,
      bestTier: Math.max(1, Math.floor(Number(source.bestTier || 1) || 1)),
      bestScore: Math.max(0, Math.floor(Number(source.bestScore || 0) || 0)),
      runs: Math.max(0, Math.floor(Number(source.runs || 0) || 0)),
      clears: Math.max(0, Math.floor(Number(source.clears || 0) || 0)),
      rewardClaimed: source.rewardClaimed === true,
      rewardClaimedAt: Math.max(0, Math.floor(Number(source.rewardClaimedAt || 0) || 0))
    };
  }

  function createRiftRunSummary(value) {
    const source = value && typeof value === 'object' ? value : null;
    if (!source) return null;
    const startedAt = Math.max(0, Math.floor(Number(source.startedAt || 0) || 0));
    const endedAt = Math.max(startedAt, Math.floor(Number(source.endedAt || startedAt) || startedAt));
    const reason = normalizeId(source.reason) || 'exit';
    return {
      runId: normalizeId(source.runId),
      status: normalizeId(source.status) || (reason === 'timeout' ? 'cleared' : 'ended'),
      reason,
      cleared: source.cleared === true || reason === 'timeout',
      startedAt,
      endedAt,
      elapsedMs: Math.max(0, Math.floor(Number(source.elapsedMs || endedAt - startedAt) || 0)),
      tier: Math.max(1, Math.floor(Number(source.tier || 1) || 1)),
      score: Math.max(0, Math.floor(Number(source.score || 0) || 0)),
      kills: Math.max(0, Math.floor(Number(source.kills || 0) || 0))
    };
  }

  function createRiftState(value, options) {
    const data = getMapMechanicData(options);
    const settings = options || {};
    const source = value && typeof value === 'object' ? value : {};
    const validMutationIds = new Set((data.MUTATIONS || []).map((mutation) => mutation.id));
    const version = Math.max(0, Math.floor(Number(source.operationVersion || 0) || 0));
    const legacyLifetimeState = version < RIFT_OPERATION_VERSION &&
      Object.prototype.hasOwnProperty.call(source, 'startedAt') &&
      !Object.prototype.hasOwnProperty.call(source, 'active');
    const sourceTier = Math.max(1, Math.floor(Number(source.tier || 1) || 1));
    const tier = legacyLifetimeState ? 1 : sourceTier;
    const mutationIds = Array.isArray(source.mutationIds)
      ? source.mutationIds.map(normalizeId).filter((id) => validMutationIds.has(id))
      : [];
    const active = !legacyLifetimeState && source.active === true &&
      Number(source.startedAt || 0) > 0 && Number(source.endsAt || 0) > Number(source.startedAt || 0);
    const personalBestTier = Math.max(
      1,
      Math.floor(Number(source.personalBestTier || 1) || 1),
      Math.floor(Number(source.bestTier || 1) || 1),
      sourceTier
    );
    const nowMs = normalizeRiftOperationNow(settings.nowMs);
    return {
      operationVersion: RIFT_OPERATION_VERSION,
      active,
      status: active ? 'active' : legacyLifetimeState ? 'migrated' : normalizeId(source.status) || 'idle',
      runId: active ? normalizeId(source.runId) : '',
      tier,
      bestTier: personalBestTier,
      personalBestTier,
      personalBestScore: Math.max(0, Math.floor(Number(source.personalBestScore || 0) || 0)),
      score: legacyLifetimeState ? 0 : Math.max(0, Math.floor(Number(source.score || 0) || 0)),
      runScore: legacyLifetimeState ? 0 : Math.max(0, Math.floor(Number(source.runScore || 0) || 0)),
      kills: legacyLifetimeState ? 0 : Math.max(0, Math.floor(Number(source.kills || 0) || 0)),
      mutationIds: legacyLifetimeState ? [] : Array.from(new Set(mutationIds)).slice(0, 3),
      startedAt: active ? Math.max(0, Math.floor(Number(source.startedAt || 0) || 0)) : 0,
      endsAt: active ? Math.max(0, Math.floor(Number(source.endsAt || 0) || 0)) : 0,
      durationMs: active
        ? Math.max(1000, Math.floor(Number(source.durationMs || Number(source.endsAt || 0) - Number(source.startedAt || 0)) || RIFT_OPERATION_DURATION_MS))
        : RIFT_OPERATION_DURATION_MS,
      runCount: Math.max(0, Math.floor(Number(source.runCount || 0) || 0)),
      clearedRuns: Math.max(0, Math.floor(Number(source.clearedRuns || 0) || 0)),
      lastRun: createRiftRunSummary(source.lastRun),
      weekly: createRiftWeeklyState(source.weekly, nowMs),
      mapMechanics: createMapMechanicState(source.mapMechanics || source.mapMechanicsByMapId, options)
    };
  }

  function startRiftOperation(rift, options) {
    const settings = options || {};
    const nowMs = normalizeRiftOperationNow(settings.nowMs);
    const state = createRiftState(rift, Object.assign({}, settings, { nowMs }));
    const durationMs = Math.max(1000, Math.floor(Number(settings.durationMs || RIFT_OPERATION_DURATION_MS) || RIFT_OPERATION_DURATION_MS));
    const runCount = state.runCount + 1;
    const startTier = Math.max(1, Math.floor(Number(settings.startTier || 1) || 1));
    return Object.assign({}, state, {
      active: true,
      status: 'active',
      runId: normalizeId(settings.runId) || `rift-run-${nowMs}-${runCount}`,
      tier: startTier,
      score: 0,
      runScore: 0,
      kills: 0,
      mutationIds: [],
      startedAt: nowMs,
      endsAt: nowMs + durationMs,
      durationMs,
      runCount,
      weekly: createRiftWeeklyState(state.weekly, nowMs)
    });
  }

  function finishRiftOperation(rift, reason, options) {
    const settings = options || {};
    const nowMs = normalizeRiftOperationNow(settings.nowMs);
    const state = createRiftState(rift, Object.assign({}, settings, { nowMs }));
    if (!state.active) return { changed: false, state, summary: state.lastRun, reward: null };
    const normalizedReason = normalizeId(reason) || 'exit';
    const cleared = normalizedReason === 'timeout' || normalizedReason === 'clear';
    const endedAt = Math.max(state.startedAt, nowMs);
    const summary = createRiftRunSummary({
      runId: state.runId,
      status: cleared ? 'cleared' : normalizedReason === 'death' ? 'failed' : 'ended',
      reason: normalizedReason,
      cleared,
      startedAt: state.startedAt,
      endedAt,
      elapsedMs: Math.min(state.durationMs, Math.max(0, endedAt - state.startedAt)),
      tier: state.tier,
      score: state.runScore,
      kills: state.kills
    });
    const weekly = createRiftWeeklyState(state.weekly, endedAt);
    weekly.runs += 1;
    if (cleared) weekly.clears += 1;
    weekly.bestTier = Math.max(weekly.bestTier, summary.tier);
    weekly.bestScore = Math.max(weekly.bestScore, summary.score);
    let reward = null;
    if (cleared && !weekly.rewardClaimed) {
      weekly.rewardClaimed = true;
      weekly.rewardClaimedAt = endedAt;
      reward = RIFT_FIRST_CLEAR_REWARD;
    }
    const personalBestTier = Math.max(state.personalBestTier, summary.tier);
    const personalBestScore = Math.max(state.personalBestScore, summary.score);
    return {
      changed: true,
      summary,
      reward,
      state: Object.assign({}, state, {
        active: false,
        status: summary.status,
        runId: '',
        bestTier: personalBestTier,
        personalBestTier,
        personalBestScore,
        startedAt: 0,
        endsAt: 0,
        clearedRuns: state.clearedRuns + (cleared ? 1 : 0),
        lastRun: summary,
        weekly
      })
    };
  }

  function getMapMechanicSection(definition, sectionId) {
    const id = normalizeMapMechanicSectionId(definition, sectionId);
    return id && (definition.sections || []).find((section) => section && section.id === id) || null;
  }

  function getMapMechanicSectionWeight(definition, sectionId) {
    const section = getMapMechanicSection(definition, sectionId);
    return Math.max(0.25, Number(section && section.weight || 1) || 1);
  }

  function getMapMechanicRewardScale(entry, definition) {
    if (!entry || !definition) return 1;
    return clamp(Number(entry.rewardScale || 1), Number(definition.minimumRewardScale || 0.5), 1);
  }

  function clonePlain(value) {
    if (!value || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(clonePlain);
    return Object.entries(value).reduce((copy, [key, item]) => {
      copy[key] = clonePlain(item);
      return copy;
    }, {});
  }

  function createScaledMapMechanicReward(reward, scale) {
    const result = clonePlain(reward || {});
    const rewardScale = clamp(Number(scale || 1), 0.1, 1);
    if (rewardScale >= 0.999) return result;
    if (result.currency) result.currency = Math.max(1, Math.round(Number(result.currency || 0) * rewardScale));
    Object.keys(result.materials || {}).forEach((materialId) => {
      result.materials[materialId] = Math.max(1, Math.round(Number(result.materials[materialId] || 0) * rewardScale));
    });
    Object.keys(result.consumables || {}).forEach((consumableId) => {
      result.consumables[consumableId] = Math.max(1, Math.round(Number(result.consumables[consumableId] || 0) * rewardScale));
    });
    return result;
  }

  function createRiftMutationIds(rift, options) {
    const data = getMapMechanicData(options);
    const state = createRiftState(rift, options);
    if (state.mutationIds.length) return state.mutationIds.slice();
    const mutations = data.MUTATIONS || [];
    const tier = Math.max(1, Number(state.tier || 1));
    const count = clamp(1 + Math.floor(tier / 12), 1, 3);
    const ids = [];
    for (let index = 0; index < count; index += 1) {
      const pick = seededPick(
        mutations.filter((mutation) => !ids.includes(mutation.id)),
        `rift:${state.runId || 'legacy'}:${tier}`,
        index
      );
      if (pick && pick.id) ids.push(pick.id);
    }
    return ids;
  }

  function createRiftRuntimeEffects(rift, mutationIds, options) {
    const data = getMapMechanicData(options);
    const settings = options || {};
    const state = createRiftState(rift, options);
    const ids = Array.from(new Set(Array.isArray(mutationIds)
      ? mutationIds.map(normalizeId).filter(Boolean)
      : createRiftMutationIds(state, options)));
    const counterplayIds = new Set((Array.isArray(settings.counterplayIds) ? settings.counterplayIds : []).map(normalizeId).filter(Boolean));
    const tierEffects = createRiftTierRuntimeEffects(state.tier);
    let mutationEnemyHealthScale = 1;
    let mutationEnemyDamageScale = 1;
    let playerDamageScale = 1;
    let playerResourceCostScale = 1;
    let mutationRewardScale = 1;
    const mutationEffects = ids.map((id) => getById(data.MUTATIONS || [], id)).filter(Boolean).map((mutation) => {
      const counterplay = mutation.counterplay || {};
      const countered = !!(counterplay.id && counterplayIds.has(normalizeId(counterplay.id)));
      const dangerMitigation = countered ? clamp(Number(counterplay.dangerMitigation || 0), 0, 0.75) : 0;
      const dangerInfluence = 1 - dangerMitigation;
      const enemyHealthScale = getRiftMutationScale(mutation.danger, 'enemyHealthScale', tierEffects.mutationPotencyScale, dangerInfluence);
      const enemyDamageScale = getRiftMutationScale(mutation.danger, 'enemyDamageScale', tierEffects.mutationPotencyScale, dangerInfluence);
      const dangerResourceCostScale = getRiftMutationScale(mutation.danger, 'playerResourceCostScale', tierEffects.mutationPotencyScale, dangerInfluence);
      const upsideDamageScale = getRiftMutationScale(mutation.upside, 'playerDamageScale', tierEffects.mutationPotencyScale, 1);
      const upsideResourceCostScale = getRiftMutationScale(mutation.upside, 'playerResourceCostScale', tierEffects.mutationPotencyScale, 1);
      const rewardScale = getRiftMutationScale({ rewardScale: mutation.rewardScale }, 'rewardScale', tierEffects.mutationPotencyScale, 1);
      mutationEnemyHealthScale *= enemyHealthScale;
      mutationEnemyDamageScale *= enemyDamageScale;
      playerDamageScale *= upsideDamageScale;
      playerResourceCostScale *= dangerResourceCostScale * upsideResourceCostScale;
      mutationRewardScale *= rewardScale;
      return {
        id: mutation.id,
        name: mutation.name,
        effect: mutation.effect,
        countered,
        dangerMitigation: roundRiftScale(dangerMitigation),
        counterplay: Object.assign({}, counterplay),
        appliedScales: {
          enemyHealthScale: roundRiftScale(enemyHealthScale),
          enemyDamageScale: roundRiftScale(enemyDamageScale),
          playerDamageScale: roundRiftScale(upsideDamageScale),
          playerResourceCostScale: roundRiftScale(dangerResourceCostScale * upsideResourceCostScale),
          rewardScale: roundRiftScale(rewardScale)
        }
      };
    });
    return {
      tier: tierEffects.tier,
      mutationPotencyScale: tierEffects.mutationPotencyScale,
      mutationIds: mutationEffects.map((mutation) => mutation.id),
      mutationEffects,
      counterplayIds: Array.from(counterplayIds),
      enemyHealthScale: roundRiftScale(tierEffects.enemyHealthScale * mutationEnemyHealthScale),
      enemyDamageScale: roundRiftScale(tierEffects.enemyDamageScale * mutationEnemyDamageScale),
      playerDamageScale: roundRiftScale(playerDamageScale),
      playerResourceCostScale: roundRiftScale(clamp(playerResourceCostScale, 0.65, 1.5)),
      rewardScale: roundRiftScale(tierEffects.rewardScale * mutationRewardScale)
    };
  }

  function createRiftSnapshot(rift, mutationIds, options) {
    const data = getMapMechanicData(options);
    const settings = options || {};
    const nowMs = normalizeRiftOperationNow(settings.nowMs);
    const state = createRiftState(rift, Object.assign({}, settings, { nowMs }));
    const tier = Math.max(1, Number(state.tier || 1));
    const ids = Array.isArray(mutationIds)
      ? mutationIds.map(normalizeId).filter(Boolean)
      : createRiftMutationIds(state, options);
    const remainingMs = state.active ? Math.max(0, state.endsAt - nowMs) : 0;
    const elapsedMs = state.active ? Math.max(0, Math.min(state.durationMs, nowMs - state.startedAt)) : 0;
    return {
      operationVersion: state.operationVersion,
      active: state.active,
      status: state.status,
      runId: state.runId,
      tier,
      bestTier: Math.max(tier, Number(state.bestTier || tier)),
      score: Math.max(0, Number(state.score || 0)),
      runScore: Math.max(0, Number(state.runScore || 0)),
      kills: Math.max(0, Number(state.kills || 0)),
      nextTierScore: Math.max(500, tier * 500),
      startedAt: state.startedAt,
      endsAt: state.endsAt,
      durationMs: state.durationMs,
      elapsedMs,
      remainingMs,
      remainingSeconds: Math.ceil(remainingMs / 1000),
      timeExpired: state.active && remainingMs <= 0,
      progress: state.active ? clamp(elapsedMs / Math.max(1, state.durationMs), 0, 1) : 0,
      personalBest: {
        tier: state.personalBestTier,
        score: state.personalBestScore
      },
      weeklyBest: {
        cycleId: state.weekly.cycleId,
        startedAt: state.weekly.startedAt,
        endsAt: state.weekly.endsAt,
        tier: state.weekly.bestTier,
        score: state.weekly.bestScore,
        runs: state.weekly.runs,
        clears: state.weekly.clears
      },
      weeklyRewardClaimed: state.weekly.rewardClaimed,
      weeklyRewardAvailable: !state.weekly.rewardClaimed,
      weeklyReward: clonePlain(RIFT_FIRST_CLEAR_REWARD),
      lastRun: createRiftRunSummary(state.lastRun),
      mutationIds: ids.slice(),
      mutations: ids.map((id) => getById(data.MUTATIONS || [], id)).filter(Boolean),
      runtimeEffects: createRiftRuntimeEffects(state, ids, settings)
    };
  }

  function createMapMechanicSnapshot(definition, entry, options) {
    const settings = options || {};
    if (!definition) return { active: false, mapId: normalizeId(settings.mapId || settings.currentMapId) };
    const state = createMapMechanicEntryState(definition, entry);
    const activeSection = getMapMechanicSection(definition, state.activeSectionId);
    const objectiveSection = getMapMechanicSection(definition, definition.objectiveSectionId);
    const regroupSection = getMapMechanicSection(definition, definition.regroupSectionId);
    const goal = Math.max(1, Number(definition.eventKillGoal || 1));
    const progress = Math.max(0, Number(state.progress || 0));
    const now = Number(settings.nowSeconds || 0);
    const rewardScale = Number.isFinite(Number(settings.rewardScale))
      ? Number(settings.rewardScale)
      : getMapMechanicRewardScale(state, definition);
    return {
      active: true,
      id: definition.id,
      mapId: definition.mapId,
      type: definition.type,
      label: definition.label,
      summary: definition.summary,
      partyRoleHook: definition.partyRoleHook,
      rewardAbuseControl: definition.rewardAbuseControl,
      activeSectionId: activeSection && activeSection.id || '',
      activeSectionLabel: activeSection && activeSection.label || '',
      objectiveSectionId: objectiveSection && objectiveSection.id || '',
      objectiveSectionLabel: objectiveSection && objectiveSection.label || '',
      regroupSectionId: regroupSection && regroupSection.id || '',
      regroupSectionLabel: regroupSection && regroupSection.label || '',
      progress,
      goal,
      progressPercent: clamp(progress / goal, 0, 1),
      requiredUniqueSections: Math.max(1, Number(definition.requiredUniqueSections || 1)),
      currentUniqueSections: state.cycleSectionIds ? state.cycleSectionIds.length : 0,
      completedCycles: Math.max(0, Number(state.completedCycles || 0)),
      eventCount: Math.max(0, Number(state.eventCount || 0)),
      objectiveCount: Math.max(0, Number(state.objectiveCount || 0)),
      surgeCount: Math.max(0, Number(state.surgeCount || 0)),
      surgeActive: !!(Number(state.surgeActiveUntil || 0) > now),
      surgeActiveUntil: Math.max(0, Number(state.surgeActiveUntil || 0)),
      antiCampStacks: Math.max(0, Number(state.antiCampStacks || 0)),
      repeatCount: Math.max(0, Number(state.repeatCount || 0)),
      rewardScale,
      lastCompletedAt: Math.max(0, Number(state.lastCompletedAt || 0)),
      sections: (definition.sections || []).map((section) => Object.assign({}, section, {
        hits: Math.max(0, Number(state.sectionHits && state.sectionHits[section.id] || 0)),
        active: !!(activeSection && activeSection.id === section.id),
        objective: !!(objectiveSection && objectiveSection.id === section.id),
        regroup: !!(regroupSection && regroupSection.id === section.id)
      }))
    };
  }

  const api = {
    RIFT_OPERATION_VERSION,
    RIFT_OPERATION_DURATION_MS,
    RIFT_FIRST_CLEAR_REWARD,
    getMapMechanicDefinitionById,
    normalizeMapMechanicSectionId,
    getDefaultMapMechanicSectionId,
    createMapMechanicEntryState,
    createMapMechanicState,
    createRiftState,
    getRiftOperationWeekWindow,
    createRiftWeeklyState,
    startRiftOperation,
    finishRiftOperation,
    getMapMechanicSection,
    getMapMechanicSectionWeight,
    getMapMechanicRewardScale,
    createScaledMapMechanicReward,
    createRiftMutationIds,
    createRiftTierRuntimeEffects,
    createRiftRuntimeEffects,
    createRiftSnapshot,
    createMapMechanicSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.mapMechanics = Object.assign({}, modules.mapMechanics || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
