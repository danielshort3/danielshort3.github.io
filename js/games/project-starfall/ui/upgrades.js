(function initProjectStarfallUiUpgrades(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiCanvasWindows = (typeof require === 'function' ? require('./canvas-windows.js') : null) || UiModules.canvasWindows || {};

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function getUpgradeAidesById(data) {
    const source = data || Data;
    return Object.freeze((source.UPGRADE_AIDES || []).reduce((map, aide) => {
      const id = String(aide && (aide.id || aide.materialId) || '').trim();
      const materialId = String(aide && aide.materialId || '').trim();
      if (id) map[id] = aide;
      if (materialId) map[materialId] = aide;
      return map;
    }, {}));
  }

  const UPGRADE_AIDES_BY_ID = getUpgradeAidesById(Data);

  function getUpgradeRangeIndex(upgradeLevel) {
    const level = Number(upgradeLevel || 0);
    if (level <= 5) return 0;
    if (level <= 10) return 1;
    if (level <= 15) return 2;
    return 3;
  }

  function getUpgradeDustCost(upgradeLevel, data) {
    const rangeIndex = getUpgradeRangeIndex(upgradeLevel);
    const sourceData = data || Data;
    const costs = Array.isArray(sourceData.UPGRADE_DUST_COST_BY_RANGE) ? sourceData.UPGRADE_DUST_COST_BY_RANGE : [];
    return Math.max(1, Math.floor(Number(costs[rangeIndex]) || (1 + Math.floor((Number(upgradeLevel) || 0) / 4))));
  }

  function formatChance(percent) {
    const rounded = Math.round((Number(percent) || 0) * 10) / 10;
    return `${rounded % 1 === 0 ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
  }

  function getUpgradeAideDefinition(aideId, options) {
    const id = String(aideId || '').trim();
    const aidesById = options && options.upgradeAidesById || UPGRADE_AIDES_BY_ID;
    return aidesById[id] || null;
  }

  function normalizeUpgradeAideIds(aideIds, options) {
    const selected = Array.isArray(aideIds) ? aideIds : Object.values(aideIds || {});
    const byType = {};
    selected.forEach((aideId) => {
      const aide = getUpgradeAideDefinition(aideId, options);
      if (aide && aide.type && !byType[aide.type]) byType[aide.type] = aide.materialId || aide.id;
    });
    return Object.values(byType);
  }

  function getUpgradeAides(aideIds, options) {
    return normalizeUpgradeAideIds(aideIds, options)
      .map((aideId) => getUpgradeAideDefinition(aideId, options))
      .filter(Boolean);
  }

  function getUpgradeSuccessBonus(aideIds, options) {
    return getUpgradeAides(aideIds, options)
      .reduce((sum, aide) => sum + Math.max(0, Number(aide.successBonus || 0)), 0);
  }

  function getUpgradeEnhancementBonus(aideIds, options) {
    return getUpgradeAides(aideIds, options)
      .reduce((sum, aide) => sum + Math.max(0, Math.floor(Number(aide.successUpgradeBonus || 0) || 0)), 0);
  }

  function hasUpgradeDestroyProtection(aideIds, options) {
    return getUpgradeAides(aideIds, options).some((aide) => !!aide.protectsDestroy);
  }

  function upgradePreviewHasDestroyChance(preview) {
    return !!(preview && preview.outcomes || []).some((outcome) => outcome.id === 'destroy' && Number(outcome.chance || 0) > 0);
  }

  function getUpgradePreview(item, aideIds, options) {
    const settings = options || {};
    const data = settings.data || Data;
    const upgrade = Number(item && item.upgrade || 0);
    const rangeIndex = getUpgradeRangeIndex(upgrade);
    const outcomes = (data.UPGRADE_OUTCOMES || [])
      .map((outcome) => Object.assign({}, outcome, {
        weight: Math.max(0, Number(outcome.weightByRange && outcome.weightByRange[rangeIndex]) || 0)
      }))
      .filter((outcome) => outcome.weight > 0);
    const totalWeight = outcomes.reduce((sum, outcome) => sum + outcome.weight, 0);
    const successBonusWeight = totalWeight * getUpgradeSuccessBonus(aideIds, settings) / 100;
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
    const adjustedTotal = outcomes.reduce((sum, outcome) => sum + outcome.weight, 0);
    return {
      cost: getUpgradeDustCost(upgrade, data),
      rangeIndex,
      outcomes: outcomes.map((outcome) => Object.assign({}, outcome, {
        chance: adjustedTotal ? outcome.weight / adjustedTotal * 100 : 0
      }))
    };
  }

  function getUpgradeMaterialRequirements(snapshot, preview, options) {
    const settings = options || {};
    const data = settings.data || Data;
    const materialDisplayMeta = settings.materialDisplayMeta || {};
    const getMaterialAssetId = settings.getMaterialAssetId || function getMaterialAssetIdFallback(materialId) {
      return String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase();
    };
    const materials = snapshot && snapshot.state && snapshot.state.materials || {};
    const requiredDust = Math.max(1, Math.floor(Number(preview && preview.cost || 1) || 1));
    const dustMeta = materialDisplayMeta.upgradeDust || {};
    return [{
      materialId: 'upgradeDust',
      required: requiredDust,
      owned: Math.max(0, Math.floor(Number(materials.upgradeDust || 0) || 0)),
      name: dustMeta.name,
      icon: dustMeta.icon,
      asset: data.ITEM_ASSETS && data.ITEM_ASSETS[getMaterialAssetId('upgradeDust')] || ''
    }];
  }

  function upgradeRequirementsMet(requirements) {
    return (requirements || []).every((requirement) => Number(requirement.owned || 0) >= Number(requirement.required || 0));
  }

  function formatUpgradeChanceSummary(preview, limit) {
    const max = Math.max(1, Number(limit) || 3);
    return (preview.outcomes || [])
      .slice(0, max)
      .map((outcome) => `${outcome.label} ${formatChance(outcome.chance)}`)
      .join(' | ');
  }

  function getUpgradeBeforeAfterSummary(item, aideIds, options) {
    const settings = options || {};
    const getItemStrength = settings.getItemStrength || function getItemStrengthFallback() { return 0; };
    const getEffectiveStats = settings.getEffectiveStats || function getEffectiveStatsFallback() { return {}; };
    const compactStats = settings.compactStats || function compactStatsFallback() { return ''; };
    const upgrade = Number(item && item.upgrade || 0);
    const current = getItemStrength(item);
    const successItem = Object.assign({}, item, { upgrade: Math.min(20, upgrade + 1 + getUpgradeEnhancementBonus(aideIds, settings)) });
    const failItem = Object.assign({}, item, { upgrade: Math.max(0, upgrade - 1) });
    const next = getItemStrength(successItem);
    const fail = getItemStrength(failItem);
    const currentStats = compactStats(getEffectiveStats(item), 2) || 'No stats';
    const nextStats = compactStats(getEffectiveStats(successItem), 2) || 'No stats';
    const failStats = compactStats(getEffectiveStats(failItem), 2) || 'No stats';
    return {
      current,
      next,
      fail,
      currentStats,
      nextStats,
      failStats,
      delta: next - current
    };
  }

  function getUpgradeResultItem(item, outcome, aideIds, options) {
    if (!item || !outcome || outcome.id === 'destroy') return null;
    const upgrade = Number(item.upgrade || 0);
    const next = Object.assign({}, item, {
      stats: Object.assign({}, item.stats || {})
    });
    if (outcome.id === 'success') next.upgrade = upgrade + 1 + getUpgradeEnhancementBonus(aideIds, options);
    else if (outcome.id === 'fail') next.upgrade = Math.max(0, upgrade - 1);
    else next.upgrade = upgrade;
    next.upgrade = Math.min(20, Math.max(0, next.upgrade || 0));
    return next;
  }

  function getUpgradeOutcomeResult(snapshot, item, outcome, aideIds, options) {
    const settings = options || {};
    const getItemStrength = settings.getItemStrength || function getItemStrengthFallback() { return 0; };
    const getEffectiveStats = settings.getEffectiveStats || function getEffectiveStatsFallback() { return {}; };
    const compactStats = settings.compactStats || function compactStatsFallback() { return ''; };
    const currentUpgrade = Number(item && item.upgrade || 0);
    const projected = getUpgradeResultItem(item, outcome, aideIds, settings);
    if (!item || !outcome) {
      return { result: 'No result available.', detail: '' };
    }
    if (outcome.id === 'destroy') {
      if (hasUpgradeDestroyProtection(aideIds, settings)) {
        const failUpgrade = Math.max(0, currentUpgrade - 1);
        const failItem = Object.assign({}, item, { upgrade: failUpgrade });
        return {
          result: `Protected, drops to +${failUpgrade}.`,
          detail: compactStats(getEffectiveStats(failItem), 2) || 'The item survives this result.'
        };
      }
      return {
        result: 'Item is destroyed.',
        detail: 'The gear leaves inventory/equipment after this result.'
      };
    }
    if (outcome.id === 'fail') {
      const failUpgrade = Math.max(0, currentUpgrade - 1);
      const failItem = Object.assign({}, item, { upgrade: failUpgrade });
      return {
        result: `Drops to +${failUpgrade} with PWR ${getItemStrength(failItem)}.`,
        detail: failUpgrade === currentUpgrade ? 'Already +0, so stats stay at the floor.' : compactStats(getEffectiveStats(failItem), 2)
      };
    }
    const stats = compactStats(getEffectiveStats(projected), 2) || 'No stats';
    return {
      result: `Becomes +${projected.upgrade} with PWR ${getItemStrength(projected)}.`,
      detail: stats
    };
  }

  function createUpgradeUiHelpers(options) {
    const settings = options || {};
    const data = settings.data || Data;
    const sharedOptions = Object.assign({}, settings, {
      data,
      upgradeAidesById: settings.upgradeAidesById || getUpgradeAidesById(data)
    });
    return Object.freeze({
      getUpgradeRangeIndex,
      getUpgradeDustCost: (upgradeLevel) => getUpgradeDustCost(upgradeLevel, data),
      formatChance,
      getUpgradeAideDefinition: (aideId) => getUpgradeAideDefinition(aideId, sharedOptions),
      normalizeUpgradeAideIds: (aideIds) => normalizeUpgradeAideIds(aideIds, sharedOptions),
      getUpgradeAides: (aideIds) => getUpgradeAides(aideIds, sharedOptions),
      getUpgradeSuccessBonus: (aideIds) => getUpgradeSuccessBonus(aideIds, sharedOptions),
      getUpgradeEnhancementBonus: (aideIds) => getUpgradeEnhancementBonus(aideIds, sharedOptions),
      hasUpgradeDestroyProtection: (aideIds) => hasUpgradeDestroyProtection(aideIds, sharedOptions),
      upgradePreviewHasDestroyChance,
      getUpgradePreview: (item, aideIds) => getUpgradePreview(item, aideIds, sharedOptions),
      getUpgradeMaterialRequirements: (snapshot, preview) => getUpgradeMaterialRequirements(snapshot, preview, sharedOptions),
      upgradeRequirementsMet,
      formatUpgradeChanceSummary,
      getUpgradeBeforeAfterSummary: (item, aideIds) => getUpgradeBeforeAfterSummary(item, aideIds, sharedOptions),
      getUpgradeResultItem: (item, outcome, aideIds) => getUpgradeResultItem(item, outcome, aideIds, sharedOptions),
      getUpgradeOutcomeResult: (snapshot, item, outcome, aideIds) => getUpgradeOutcomeResult(snapshot, item, outcome, aideIds, sharedOptions),
      getAllItems,
      getUpgradeItemGroups: (snapshot, options) => getUpgradeItemGroups(snapshot, Object.assign({}, sharedOptions, options || {})),
      getSelectedUpgradeItem: (snapshot, items, options) => getSelectedUpgradeItem(snapshot, items, Object.assign({}, sharedOptions, options || {})),
      getUpgradePromptBox: (width, height, state, options) => getUpgradePromptBox(width, height, state, Object.assign({}, sharedOptions, options || {})),
      getUpgradeStationDomAction,
      getUpgradeStationRegionAction,
      getUpgradePromptPointerAction,
      getUpgradeConfirmDomAction
    });
  }

  function getAllItems(snapshot) {
    const equipped = Object.values(snapshot.state.equipment || {}).filter(Boolean);
    return [...equipped, ...(snapshot.state.inventory || [])];
  }

  function getUpgradeItemGroups(snapshot, options) {
    const settings = options || {};
    const equipmentSlotDisplayOrder = settings.equipmentSlotDisplayOrder || [];
    const equipment = snapshot && snapshot.state && snapshot.state.equipment || {};
    const equipped = equipmentSlotDisplayOrder
      .map((slot) => equipment[slot])
      .filter(Boolean);
    const inventory = (snapshot && snapshot.state && snapshot.state.inventory || []).filter(Boolean);
    return [
      { id: 'equipped', label: 'Equipped Items', items: equipped },
      { id: 'inventory', label: 'Inventory Items', items: inventory }
    ];
  }

  function getSelectedUpgradeItem(snapshot, items, options) {
    const list = items || getAllItems(snapshot);
    const state = snapshot && snapshot.state || {};
    const selectedUpgradeUid = state.selectedUpgradeUid || state.selectedInventoryUid || '';
    return list.find((item) => item && item.uid === selectedUpgradeUid) || list[0] || null;
  }

  function getUpgradePromptBox(width, height, state, options) {
    const settings = options || {};
    const boxW = Math.min(390, Math.max(310, width - 32));
    const boxH = 360;
    const bottomLimit = Number(settings.bottomLimit || height);
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, userPlaced: false };
    const getCenteredPromptBox = typeof settings.getCenteredPromptBox === 'function'
      ? settings.getCenteredPromptBox
      : UiCanvasWindows.getCenteredPromptBox;
    if (getCenteredPromptBox) {
      return getCenteredPromptBox(width, height, target, {
        minWidth: 310,
        maxWidth: 390,
        height: boxH,
        bottomLimit,
        defaultYMin: 18,
        constrainDefaultY: false
      });
    }
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      target.y = Math.round(Math.max(18, (bottomLimit - boxH) / 2));
    }
    target.x = clamp(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clamp(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getUpgradeStationDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const selectUpgradeUid = getAttribute('data-starfall-select-upgrade');
    if (selectUpgradeUid) return { handled: true, type: 'openUpgradePrompt', uid: selectUpgradeUid };
    if (hasAttribute('data-starfall-upgrade-drop-zone')) return { handled: true, type: 'openGearPicker', mode: 'upgrade' };
    const upgradeAideId = getAttribute('data-starfall-upgrade-aide');
    if (upgradeAideId) return { handled: true, type: 'toggleAide', aideId: upgradeAideId };
    if (hasAttribute('data-starfall-upgrade-close')) return { handled: true, type: 'closeUpgradePrompt' };
    return { handled: false, type: '' };
  }

  function getUpgradeStationRegionAction(region) {
    const source = region || {};
    if (source.type === 'select-upgrade-item') return { handled: true, type: 'openUpgradePrompt', uid: source.uid };
    if (source.type === 'upgrade-item') return { handled: true, type: 'confirmUpgradeItem', uid: source.uid };
    if (source.type === 'upgrade-prompt-close') return { handled: true, type: 'closeUpgradePrompt' };
    if (source.type === 'upgrade-aide-toggle') return { handled: true, type: 'toggleAide', aideId: source.aideId };
    if (source.type === 'upgrade-prompt-confirm') return { handled: true, type: 'confirmUpgradePrompt', uid: source.uid };
    return { handled: false, type: '' };
  }

  function getUpgradePromptPointerAction(region) {
    const source = region || {};
    if (source.type === 'upgrade-prompt-header') {
      return {
        handled: true,
        type: 'startUpgradePromptDrag',
        dragKey: 'upgradePromptDrag',
        boxType: 'upgradePrompt',
        activePanel: 'upgradePrompt',
        shouldPreventDefault: true
      };
    }
    return {
      handled: false,
      type: '',
      dragKey: '',
      boxType: '',
      activePanel: '',
      shouldPreventDefault: false
    };
  }

  function getUpgradeConfirmDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const uid = getAttribute('data-starfall-upgrade-confirm');
    if (uid) return { handled: true, type: 'confirmUpgradePrompt', uid };
    return { handled: false, type: '' };
  }

  const api = {
    UPGRADE_AIDES_BY_ID,
    getUpgradeAidesById,
    getUpgradeRangeIndex,
    getUpgradeDustCost,
    formatChance,
    getUpgradeAideDefinition,
    normalizeUpgradeAideIds,
    getUpgradeAides,
    getUpgradeSuccessBonus,
    getUpgradeEnhancementBonus,
    hasUpgradeDestroyProtection,
    upgradePreviewHasDestroyChance,
    getUpgradePreview,
    getUpgradeMaterialRequirements,
    upgradeRequirementsMet,
    formatUpgradeChanceSummary,
    getUpgradeBeforeAfterSummary,
    getUpgradeResultItem,
    getUpgradeOutcomeResult,
    createUpgradeUiHelpers,
    getAllItems,
    getUpgradeItemGroups,
    getSelectedUpgradeItem,
    getUpgradePromptBox,
    getUpgradeStationDomAction,
    getUpgradeStationRegionAction,
    getUpgradePromptPointerAction,
    getUpgradeConfirmDomAction
  };

  const modules = UiModules;
  modules.upgrades = Object.assign({}, modules.upgrades || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
