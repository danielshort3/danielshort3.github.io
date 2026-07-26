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

  function getSpecializationRespecCost(options) {
    const data = getSpecializationData(options);
    return Math.max(0, Math.floor(Number(data.SPECIALIZATION_RESPEC_COST || 0) || 0));
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

  function getActiveSpecialization(player, specializations, options) {
    const data = getSpecializationData(options);
    const activePlayer = player || {};
    const state = specializations && typeof specializations === 'object'
      ? specializations
      : createSpecializationState(null, options);
    const specializationId = state.selectedByAdvancedId && state.selectedByAdvancedId[activePlayer.advancedClassId];
    const specialization = getById(data.SPECIALIZATIONS || [], specializationId);
    return specialization && specialization.advancedId === activePlayer.advancedClassId
      ? specialization
      : null;
  }

  function getSpecializationSkillModifier(skill, player, specializations, options) {
    const specialization = getActiveSpecialization(player, specializations, options);
    if (!specialization || !skill || !skill.id || skill.owner !== specialization.advancedId) return null;
    const modifier = specialization.skillModifiers && specialization.skillModifiers[skill.id];
    return modifier && typeof modifier === 'object'
      ? Object.assign({ sourceSpecializationId: specialization.id }, modifier)
      : null;
  }

  function createSpecializationBonuses(player, specializations, options) {
    const specialization = getActiveSpecialization(player, specializations, options);
    if (!specialization) return {};
    return Object.entries(specialization.statBonuses || {}).reduce((stats, [key, value]) => {
      stats[key] = Number(value || 0);
      return stats;
    }, {});
  }

  function getSpecializationLockReason(specialization, player, options) {
    const data = getSpecializationData(options);
    const settings = options || {};
    const activePlayer = player || {};
    if (!specialization) return 'Specialization is unavailable.';
    if (!activePlayer.advancedClassId) return 'Choose an advanced class first.';
    if (specialization.advancedId !== activePlayer.advancedClassId) return 'Different advanced class.';
    const levelRequirement = specialization.levelRequirement || data.SPECIALIZATION_LEVEL || 60;
    if (activePlayer.level < Number(levelRequirement)) {
      return `Level ${levelRequirement} required.`;
    }
    const state = settings.specializations && typeof settings.specializations === 'object'
      ? settings.specializations
      : createSpecializationState(null, options);
    const selectedId = state.selectedByAdvancedId && state.selectedByAdvancedId[activePlayer.advancedClassId] || '';
    if (selectedId === specialization.id) return '';
    if (settings.trialActive) return 'Finish the active class trial before choosing a path.';
    if (settings.dungeonActive) return 'Finish or leave the active dungeon before choosing a path.';
    if (settings.riftActive) return 'Bank or end the Rift run before choosing a path.';
    if (!settings.safeZone) return 'Return to a town to choose a path.';
    const respecCost = selectedId ? getSpecializationRespecCost(options) : 0;
    if (respecCost > Math.max(0, Number(settings.currency == null ? activePlayer.currency : settings.currency) || 0)) {
      return `Need ${respecCost.toLocaleString('en-US')} coins to switch paths.`;
    }
    return '';
  }

  function createSpecializationSnapshot(player, specializations, options) {
    const data = getSpecializationData(options);
    const settings = options || {};
    const activePlayer = player || {};
    const state = specializations && typeof specializations === 'object'
      ? specializations
      : createSpecializationState(null, options);
    const selectedByAdvancedId = Object.assign({}, state.selectedByAdvancedId || {});
    const selectedId = selectedByAdvancedId[activePlayer.advancedClassId] || '';
    const selected = getById(data.SPECIALIZATIONS || [], selectedId);
    const levelRequirement = Number(data.SPECIALIZATION_LEVEL || 60);
    const respecCost = getSpecializationRespecCost(options);
    return {
      levelRequirement,
      respecCost,
      safeZone: !!settings.safeZone,
      mapName: String(settings.mapName || ''),
      selectedByAdvancedId,
      selectedId,
      selectedName: selected && selected.name || '',
      selectionPending: !!(activePlayer.advancedClassId && Number(activePlayer.level || 1) >= levelRequirement && !selectedId),
      specializations: (data.SPECIALIZATIONS || []).map((specialization) => Object.assign({}, specialization, {
        available: !!(activePlayer.advancedClassId && specialization.advancedId === activePlayer.advancedClassId),
        selected: selectedByAdvancedId[specialization.advancedId] === specialization.id,
        lockedReason: getSpecializationLockReason(specialization, activePlayer, Object.assign({}, settings, {
          data,
          specializations: state
        })),
        actionLabel: selectedByAdvancedId[specialization.advancedId] === specialization.id
          ? 'Active'
          : selectedId
            ? 'Switch'
            : 'Choose',
        choiceCost: selectedId && selectedId !== specialization.id ? respecCost : 0
      }))
    };
  }

  function createSpecializationChoicePlan(specialization, player, options) {
    const settings = options || {};
    const activePlayer = player || {};
    const state = settings.specializations && typeof settings.specializations === 'object'
      ? settings.specializations
      : createSpecializationState(null, options);
    const selectedId = state.selectedByAdvancedId && state.selectedByAdvancedId[activePlayer.advancedClassId] || '';
    if (specialization && selectedId === specialization.id) {
      return {
        ok: false,
        reason: 'alreadySelected',
        toast: `${specialization.name} is already active.`
      };
    }
    const lockReason = getSpecializationLockReason(specialization, activePlayer, Object.assign({}, settings, {
      specializations: state
    }));
    if (lockReason) {
      return {
        ok: false,
        reason: 'locked',
        toast: lockReason
      };
    }
    const cost = selectedId ? getSpecializationRespecCost(options) : 0;
    if (!settings.confirmed) {
      return {
        ok: false,
        reason: 'confirmation',
        requiresConfirmation: true,
        advancedId: specialization.advancedId,
        specializationId: specialization.id,
        previousSpecializationId: selectedId,
        cost,
        toast: `Confirm ${selectedId ? 'the path switch' : 'this specialization'} first.`
      };
    }
    return {
      ok: true,
      advancedId: specialization.advancedId,
      specializationId: specialization.id,
      previousSpecializationId: selectedId,
      cost,
      toast: selectedId
        ? `Switched to ${specialization.name}.`
        : `${specialization.name} specialization active.`
    };
  }

  const api = {
    createSpecializationState,
    getActiveSpecialization,
    getSpecializationSkillModifier,
    createSpecializationBonuses,
    createSpecializationSnapshot,
    getSpecializationLockReason,
    createSpecializationChoicePlan,
    getSpecializationRespecCost
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.specializations = Object.assign({}, modules.specializations || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
