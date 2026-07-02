(function initProjectStarfallEngineSpecializations(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getSpecializationData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createSpecializationState(value, options) {
    const data = getSpecializationData(options);
    const source = value && typeof value === 'object' ? value : {};
    const selectedByAdvancedId = {};
    Object.entries(source.selectedByAdvancedId || {}).forEach(([advancedId, specializationId]) => {
      const advancedKey = normalizeId(advancedId);
      const specKey = normalizeId(specializationId);
      const specialization = getById(data.SPECIALIZATIONS || [], specKey);
      if (advancedKey && specialization && specialization.advancedId === advancedKey) {
        selectedByAdvancedId[advancedKey] = specKey;
      }
    });
    return { selectedByAdvancedId };
  }

  function createSpecializationBonuses(player, specializations, options) {
    const data = getSpecializationData(options);
    const activePlayer = player || {};
    const state = specializations && typeof specializations === 'object'
      ? specializations
      : createSpecializationState(null, options);
    const specializationId = state.selectedByAdvancedId && state.selectedByAdvancedId[activePlayer.advancedClassId];
    const specialization = getById(data.SPECIALIZATIONS || [], specializationId);
    if (!specialization || specialization.advancedId !== activePlayer.advancedClassId) return {};
    return Object.entries(specialization.statBonuses || {}).reduce((stats, [key, value]) => {
      stats[key] = Number(value || 0);
      return stats;
    }, {});
  }

  function getSpecializationLockReason(specialization, player, options) {
    const data = getSpecializationData(options);
    const activePlayer = player || {};
    if (!specialization) return 'Specialization is unavailable.';
    if (!activePlayer.advancedClassId) return 'Choose an advanced class first.';
    if (specialization.advancedId !== activePlayer.advancedClassId) return 'Different advanced class.';
    const levelRequirement = specialization.levelRequirement || data.SPECIALIZATION_LEVEL || 60;
    if (activePlayer.level < Number(levelRequirement)) {
      return `Level ${levelRequirement} required.`;
    }
    return '';
  }

  function createSpecializationSnapshot(player, specializations, options) {
    const data = getSpecializationData(options);
    const activePlayer = player || {};
    const state = specializations && typeof specializations === 'object'
      ? specializations
      : createSpecializationState(null, options);
    const selectedByAdvancedId = Object.assign({}, state.selectedByAdvancedId || {});
    return {
      levelRequirement: Number(data.SPECIALIZATION_LEVEL || 60),
      selectedByAdvancedId,
      selectedId: selectedByAdvancedId[activePlayer.advancedClassId] || '',
      specializations: (data.SPECIALIZATIONS || []).map((specialization) => Object.assign({}, specialization, {
        available: !!(activePlayer.advancedClassId && specialization.advancedId === activePlayer.advancedClassId),
        selected: selectedByAdvancedId[specialization.advancedId] === specialization.id,
        lockedReason: getSpecializationLockReason(specialization, activePlayer, options)
      }))
    };
  }

  function createSpecializationChoicePlan(specialization, player, options) {
    const lockReason = getSpecializationLockReason(specialization, player, options);
    if (lockReason) {
      return {
        ok: false,
        reason: 'locked',
        toast: lockReason
      };
    }
    return {
      ok: true,
      advancedId: specialization.advancedId,
      specializationId: specialization.id,
      toast: `${specialization.name} specialization active.`
    };
  }

  const api = {
    createSpecializationState,
    createSpecializationBonuses,
    createSpecializationSnapshot,
    getSpecializationLockReason,
    createSpecializationChoicePlan
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.specializations = Object.assign({}, modules.specializations || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
