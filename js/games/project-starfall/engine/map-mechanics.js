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

  function createRiftState(value, options) {
    const data = getMapMechanicData(options);
    const source = value && typeof value === 'object' ? value : {};
    const validMutationIds = new Set((data.MUTATIONS || []).map((mutation) => mutation.id));
    const tier = Math.max(1, Math.floor(Number(source.tier || 1) || 1));
    const mutationIds = Array.isArray(source.mutationIds)
      ? source.mutationIds.map(normalizeId).filter((id) => validMutationIds.has(id))
      : [];
    return {
      tier,
      bestTier: Math.max(tier, Math.floor(Number(source.bestTier || tier) || tier)),
      score: Math.max(0, Math.floor(Number(source.score || 0) || 0)),
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
    const count = clamp(1 + Math.floor(tier / 12), 1, 3);
    const ids = [];
    for (let index = 0; index < count; index += 1) {
      const pick = seededPick(mutations.filter((mutation) => !ids.includes(mutation.id)), `rift:${tier}`, index);
      if (pick && pick.id) ids.push(pick.id);
    }
    return ids;
  }

  function createRiftSnapshot(rift, mutationIds, options) {
    const data = getMapMechanicData(options);
    const state = createRiftState(rift, options);
    const tier = Math.max(1, Number(state.tier || 1));
    const ids = Array.isArray(mutationIds)
      ? mutationIds.map(normalizeId).filter(Boolean)
      : createRiftMutationIds(state, options);
    return {
      tier,
      bestTier: Math.max(tier, Number(state.bestTier || tier)),
      score: Math.max(0, Number(state.score || 0)),
      nextTierScore: Math.max(500, tier * 500),
      mutationIds: ids.slice(),
      mutations: ids.map((id) => getById(data.MUTATIONS || [], id)).filter(Boolean)
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
    getMapMechanicDefinitionById,
    normalizeMapMechanicSectionId,
    getDefaultMapMechanicSectionId,
    createMapMechanicEntryState,
    createMapMechanicState,
    createRiftState,
    getMapMechanicSection,
    getMapMechanicSectionWeight,
    getMapMechanicRewardScale,
    createScaledMapMechanicReward,
    createRiftMutationIds,
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
