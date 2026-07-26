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

  function getMapMechanicData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
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
    const orderedSectionIds = Array.isArray(source.orderedSectionIds)
      ? source.orderedSectionIds.map((sectionId) => normalizeMapMechanicSectionId(definition, sectionId)).filter(Boolean)
      : [];
    const activeIds = Array.isArray(definition && definition.activeSectionIds) ? definition.activeSectionIds : [];
    const activeSectionIndex = activeIds.findIndex((sectionId) => sectionId === activeSectionId);
    const nextSectionIndex = clamp(
      Math.floor(Number(source.nextSectionIndex == null ? orderedSectionIds.length : source.nextSectionIndex) || 0),
      0,
      Math.max(0, activeIds.length - 1)
    );
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
      cycleKillCount: Math.max(0, Math.floor(Number(source.cycleKillCount || 0) || 0)),
      currentSectionKillCount: Math.max(0, Math.floor(Number(source.currentSectionKillCount || 0) || 0)),
      orderedSectionIds: orderedSectionIds.slice(0, activeIds.length),
      nextSectionIndex,
      routeComplete: !!source.routeComplete,
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

  function createRiftBounty(value) {
    const source = value && typeof value === 'object' ? value : {};
    const normalizeCounts = (counts) => Object.entries(counts && typeof counts === 'object' ? counts : {}).reduce((result, [id, amount]) => {
      const key = normalizeId(id);
      const count = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (key && count) result[key] = count;
      return result;
    }, {});
    return {
      currency: Math.max(0, Math.floor(Number(source.currency || 0) || 0)),
      materials: normalizeCounts(source.materials),
      consumables: normalizeCounts(source.consumables)
    };
  }

  function mergeRiftBounty(value, reward, scale) {
    const bounty = createRiftBounty(value);
    const source = reward && typeof reward === 'object' ? reward : {};
    const rewardScale = Math.max(0, Number(scale == null ? 1 : scale) || 0);
    bounty.currency += Math.max(0, Math.round(Number(source.currency || 0) * rewardScale));
    ['materials', 'consumables'].forEach((kind) => {
      Object.entries(source[kind] || {}).forEach(([id, amount]) => {
        const key = normalizeId(id);
        const count = Math.max(0, Math.round(Number(amount || 0) * rewardScale));
        if (key && count) bounty[kind][key] = Math.max(0, Number(bounty[kind][key] || 0)) + count;
      });
    });
    return bounty;
  }

  function getRiftTierScoreTarget(tier) {
    const level = Math.max(1, Math.floor(Number(tier || 1) || 1));
    return Math.min(2500, 500 + (level - 1) * 50);
  }

  function createRiftState(value, options) {
    const data = getMapMechanicData(options);
    const source = value && typeof value === 'object' ? value : {};
    const validMutationIds = new Set((data.MUTATIONS || []).map((mutation) => mutation.id));
    const tier = Math.max(1, Math.floor(Number(source.tier || 1) || 1));
    const bankedTier = Math.max(1, Math.floor(Number(source.bankedTier || tier) || tier));
    const checkpointTier = Math.max(1, Math.floor(Number(source.checkpointTier || 1) || 1));
    const mutationIds = Array.isArray(source.mutationIds)
      ? source.mutationIds.map(normalizeId).filter((id) => validMutationIds.has(id))
      : [];
    return {
      tier,
      bestTier: Math.max(tier, bankedTier, checkpointTier, Math.floor(Number(source.bestTier || tier) || tier)),
      bankedTier,
      checkpointTier,
      score: Math.max(0, Math.floor(Number(source.score || 0) || 0)),
      rotationsThisTier: Math.max(0, Math.floor(Number(source.rotationsThisTier || 0) || 0)),
      decisionPending: !!source.decisionPending,
      unbankedBounty: createRiftBounty(source.unbankedBounty || source.bounty),
      mutationIds: Array.from(new Set(mutationIds)).slice(0, 3),
      startedAt: Number(source.startedAt || 0),
      mapMechanics: createMapMechanicState(source.mapMechanics || source.mapMechanicsByMapId, options)
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
    const count = clamp(1 + Math.floor((tier - 1) / 5), 1, 3);
    const ids = [];
    for (let index = 0; index < count; index += 1) {
      const pick = seededPick(mutations.filter((mutation) => !ids.includes(mutation.id)), `rift:${tier}`, index);
      if (pick && pick.id) ids.push(pick.id);
    }
    return ids;
  }

  function createRiftPressureProfile(rift, mutationIds, options) {
    const settings = options || {};
    const data = getMapMechanicData(settings);
    const state = createRiftState(Object.assign({}, rift || {}, {
      tier: settings.tier == null ? rift && rift.tier : settings.tier
    }), settings);
    const tier = Math.max(1, Math.floor(Number(state.tier || 1) || 1));
    const pressure = tier - 1;
    const ids = Array.isArray(mutationIds)
      ? mutationIds.map(normalizeId).filter(Boolean)
      : Array.isArray(settings.mutationIds)
        ? settings.mutationIds.map(normalizeId).filter(Boolean)
        : createRiftMutationIds(state, settings);
    const multipliers = {
      enemyHpScale: Math.min(2.75, 1 + pressure * 0.075),
      enemyDamageScale: Math.min(1.85, 1 + pressure * 0.035),
      enemyDefenseScale: Math.min(1.6, 1 + pressure * 0.015),
      enemySpeedScale: 1,
      eliteChanceBonus: Math.min(0.22, pressure * 0.008),
      scoreScale: Math.min(1.75, 1 + pressure * 0.015),
      rewardScale: Math.min(1.8, 1 + pressure * 0.02)
    };
    ids.forEach((id) => {
      const mutation = getById(data.MUTATIONS || [], id);
      if (!mutation) return;
      ['enemyHpScale', 'enemyDamageScale', 'enemyDefenseScale', 'enemySpeedScale', 'scoreScale', 'rewardScale'].forEach((key) => {
        multipliers[key] *= Math.max(0.1, Number(mutation[key] || 1) || 1);
      });
      multipliers.eliteChanceBonus += Math.max(0, Number(mutation.eliteChanceBonus || 0));
    });
    const mechanic = state.mapMechanics && state.mapMechanics.byMapId && state.mapMechanics.byMapId.endlessRift || {};
    const now = Number(settings.nowSeconds == null ? Date.now() / 1000 : settings.nowSeconds);
    const surgeActive = settings.surgeActive == null
      ? Number(mechanic.surgeActiveUntil || 0) > now
      : !!settings.surgeActive;
    if (surgeActive) {
      const definition = data.MAP_MECHANIC_DEFINITIONS && data.MAP_MECHANIC_DEFINITIONS.endlessRift || {};
      multipliers.eliteChanceBonus += 0.12;
      multipliers.scoreScale *= Math.max(1, Number(definition.surgeScoreScale || 1.35));
    }
    return {
      tier,
      mutationIds: ids,
      surgeActive,
      enemyHpScale: Number(multipliers.enemyHpScale.toFixed(4)),
      enemyDamageScale: Number(multipliers.enemyDamageScale.toFixed(4)),
      enemyDefenseScale: Number(multipliers.enemyDefenseScale.toFixed(4)),
      enemySpeedScale: Number(multipliers.enemySpeedScale.toFixed(4)),
      eliteChanceBonus: Number(Math.min(0.5, multipliers.eliteChanceBonus).toFixed(4)),
      scoreScale: Number(multipliers.scoreScale.toFixed(4)),
      rewardScale: Number(multipliers.rewardScale.toFixed(4))
    };
  }

  function createRiftSnapshot(rift, mutationIds, options) {
    const data = getMapMechanicData(options);
    const state = createRiftState(rift, options);
    const tier = Math.max(1, Number(state.tier || 1));
    const ids = Array.isArray(mutationIds)
      ? mutationIds.map(normalizeId).filter(Boolean)
      : createRiftMutationIds(state, options);
    const definition = data.MAP_MECHANIC_DEFINITIONS && data.MAP_MECHANIC_DEFINITIONS.endlessRift || {};
    const rotationsRequired = Math.max(1, Number(definition.rotationsPerTier || 3));
    const pressure = createRiftPressureProfile(state, ids, options);
    const mechanic = state.mapMechanics && state.mapMechanics.byMapId && state.mapMechanics.byMapId.endlessRift || {};
    const now = Number(!options || options.nowSeconds == null ? Date.now() / 1000 : options.nowSeconds);
    const surgeActiveUntil = Math.max(0, Number(mechanic.surgeActiveUntil || 0));
    return {
      tier,
      bestTier: Math.max(tier, Number(state.bestTier || tier)),
      bankedTier: Math.max(1, Number(state.bankedTier || tier)),
      checkpointTier: Math.max(1, Number(state.checkpointTier || 1)),
      score: Math.max(0, Number(state.score || 0)),
      nextTierScore: getRiftTierScoreTarget(tier),
      rotationsThisTier: Math.max(0, Number(state.rotationsThisTier || 0)),
      rotationsRequired,
      decisionPending: !!state.decisionPending,
      unbankedBounty: createRiftBounty(state.unbankedBounty),
      mutationIds: ids.slice(),
      mutations: ids.map((id) => getById(data.MUTATIONS || [], id)).filter(Boolean),
      pressure,
      surgeActiveUntil,
      surgeSecondsRemaining: Math.max(0, Math.ceil(surgeActiveUntil - now))
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
      orderedSectionIds: state.orderedSectionIds ? state.orderedSectionIds.slice() : [],
      nextSectionId: state.activeSectionId,
      nextSectionLabel: activeSection && activeSection.label || '',
      currentSectionKillCount: Math.max(0, Number(state.currentSectionKillCount || 0)),
      killsPerSection: Math.max(1, Number(definition.killsPerSection || Math.ceil(goal / Math.max(1, (definition.activeSectionIds || []).length)))),
      routeComplete: !!state.routeComplete,
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
    getMapMechanicDefinitionById,
    normalizeMapMechanicSectionId,
    getDefaultMapMechanicSectionId,
    createMapMechanicEntryState,
    createMapMechanicState,
    createRiftBounty,
    mergeRiftBounty,
    getRiftTierScoreTarget,
    createRiftState,
    getMapMechanicSection,
    getMapMechanicSectionWeight,
    getMapMechanicRewardScale,
    createScaledMapMechanicReward,
    createRiftMutationIds,
    createRiftPressureProfile,
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
