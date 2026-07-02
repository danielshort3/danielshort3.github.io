(function initProjectStarfallEngineUpgrades(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getUpgradeRangeIndex(upgradeLevel) {
    const level = Number(upgradeLevel) || 0;
    if (level <= 5) return 0;
    if (level <= 10) return 1;
    if (level <= 15) return 2;
    return 3;
  }

  function getUpgradeDustCost(upgradeLevel, data) {
    const rangeIndex = getUpgradeRangeIndex(upgradeLevel);
    const sourceData = data || {};
    const costs = Array.isArray(sourceData.UPGRADE_DUST_COST_BY_RANGE) ? sourceData.UPGRADE_DUST_COST_BY_RANGE : [];
    return Math.max(1, Math.floor(Number(costs[rangeIndex]) || (1 + Math.floor((Number(upgradeLevel) || 0) / 4))));
  }

  function getUpgradeFailureSalvageDust(cost) {
    return Math.max(1, Math.floor(Math.max(1, Number(cost || 1)) / 2));
  }

  function getUpgradeAideDefinition(aideId, data) {
    const id = normalizeId(aideId);
    const sourceData = data || {};
    return (sourceData.UPGRADE_AIDES || []).find((aide) => normalizeId(aide && (aide.id || aide.materialId)) === id) || null;
  }

  function normalizeUpgradeAideIds(aideIds, data) {
    const selected = Array.isArray(aideIds) ? aideIds : Object.values(aideIds || {});
    const byType = {};
    selected.forEach((aideId) => {
      const aide = getUpgradeAideDefinition(aideId, data);
      if (aide && aide.type && !byType[aide.type]) byType[aide.type] = aide.materialId || aide.id;
    });
    return Object.values(byType);
  }

  function getUpgradeAides(aideIds, data) {
    return normalizeUpgradeAideIds(aideIds, data)
      .map((aideId) => getUpgradeAideDefinition(aideId, data))
      .filter(Boolean);
  }

  function getUpgradeSuccessBonus(aides) {
    return (aides || []).reduce((sum, aide) => sum + Math.max(0, Number(aide.successBonus || 0)), 0);
  }

  function getUpgradeEnhancementBonus(aides) {
    return (aides || []).reduce((sum, aide) => sum + Math.max(0, Math.floor(Number(aide.successUpgradeBonus || 0) || 0)), 0);
  }

  function hasUpgradeDestroyProtection(aides) {
    return (aides || []).some((aide) => !!aide.protectsDestroy);
  }

  function getModifiedUpgradeOutcomes(upgradeLevel, aideIds, data) {
    const rangeIndex = getUpgradeRangeIndex(upgradeLevel);
    const sourceData = data || {};
    const aides = getUpgradeAides(aideIds, sourceData);
    const outcomes = (sourceData.UPGRADE_OUTCOMES || []).map((outcome) => Object.assign({}, outcome, {
      weight: Math.max(0, Number(outcome.weightByRange && outcome.weightByRange[rangeIndex]) || 0)
    }));
    const totalWeight = outcomes.reduce((sum, outcome) => sum + outcome.weight, 0);
    const successBonusWeight = totalWeight * getUpgradeSuccessBonus(aides) / 100;
    if (successBonusWeight > 0) {
      const success = outcomes.find((outcome) => outcome.id === 'success');
      const penalties = outcomes.filter((outcome) => outcome.id !== 'success' && outcome.weight > 0);
      const penaltyTotal = penalties.reduce((sum, outcome) => sum + outcome.weight, 0);
      const applied = Math.min(successBonusWeight, penaltyTotal);
      if (success && applied > 0 && penaltyTotal > 0) {
        success.weight += applied;
        penalties.forEach((outcome) => {
          outcome.weight = Math.max(0, outcome.weight - applied * (outcome.weight / penaltyTotal));
        });
      }
    }
    return outcomes.filter((outcome) => outcome.weight > 0);
  }

  function upgradeHasDestroyChance(upgradeLevel, aideIds, data) {
    return getModifiedUpgradeOutcomes(upgradeLevel, aideIds, data)
      .some((outcome) => outcome.id === 'destroy' && Number(outcome.weight || 0) > 0);
  }

  const api = {
    getUpgradeRangeIndex,
    getUpgradeDustCost,
    getUpgradeFailureSalvageDust,
    getUpgradeAideDefinition,
    normalizeUpgradeAideIds,
    getUpgradeAides,
    getUpgradeSuccessBonus,
    getUpgradeEnhancementBonus,
    hasUpgradeDestroyProtection,
    getModifiedUpgradeOutcomes,
    upgradeHasDestroyChance
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.upgrades = Object.assign({}, modules.upgrades || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
