(function initProjectStarfallUiCanvasHover(global) {
  'use strict';

  function getEmptyCanvasHoverTarget() {
    return { type: '', key: '' };
  }

  function getInfoHoverTarget(region, overrides = {}) {
    const source = region || {};
    const panelId = source.panelId || overrides.sourcePanel || '';
    const title = source.tooltipTitle || overrides.tooltipTitle || source.label || '';
    const subtitle = source.tooltipSubtitle || overrides.tooltipSubtitle || '';
    const lines = Array.isArray(source.tooltipLines)
      ? source.tooltipLines.slice()
      : Array.isArray(overrides.tooltipLines)
        ? overrides.tooltipLines.slice()
        : [];
    const key = overrides.key || source.tooltipKey || [
      'info',
      panelId,
      source.type || '',
      source.questId || source.trialId || source.dungeonId || source.accomplishmentId || source.enemyId || source.mapId || source.guideId || source.traitId || source.advancedId || title
    ].filter(Boolean).join(':');
    return Object.assign({
      type: 'info',
      key,
      sourcePanel: panelId,
      title,
      subtitle,
      lines,
      accent: source.tooltipAccent || overrides.tooltipAccent || ''
    }, overrides, {
      title,
      subtitle,
      lines,
      accent: source.tooltipAccent || overrides.tooltipAccent || ''
    });
  }

  function getCanvasHoverTargetAt(point, options) {
    const settings = options || {};
    const openWindows = Array.isArray(settings.openWindows) ? settings.openWindows : [];
    const findHoverRegion = typeof settings.findHoverRegion === 'function'
      ? settings.findHoverRegion
      : function findHoverRegionFallback() { return null; };
    const getBindableAction = typeof settings.getBindableAction === 'function'
      ? settings.getBindableAction
      : function getBindableActionFallback() { return null; };
    const getInfoTarget = typeof settings.getInfoHoverTarget === 'function'
      ? settings.getInfoHoverTarget
      : getInfoHoverTarget;

    if (openWindows.includes('skills')) {
      const skillRegion = findHoverRegion((item) => item.type === 'skill-card');
      if (skillRegion && skillRegion.skillId) return { type: 'skill', key: `skill:${skillRegion.skillId}`, sourcePanel: 'skills', skillId: skillRegion.skillId };
    }
    if (openWindows.includes('keybinds')) {
      const bindRegion = findHoverRegion((item) => item.type === 'bind-action' || item.type === 'key-target');
      const actionId = bindRegion && (bindRegion.actionId || bindRegion.ownerId) || '';
      const action = actionId ? getBindableAction(actionId) : null;
      if (action && action.type === 'skill' && action.skillId) return { type: 'skill', key: `skill:${action.skillId}`, sourcePanel: 'keybinds', skillId: action.skillId };
      if (action && action.type === 'item' && action.itemId) return { type: 'consumable', key: `consumable:${action.itemId}`, sourcePanel: 'keybinds', itemId: action.itemId };
    }
    if (openWindows.includes('inventory')) {
      const sellRuleRegion = findHoverRegion((item) => item.type === 'inventory-sell-rule' || item.type === 'inventory-sell-rarity' || item.type === 'inventory-sell-settings-toggle' || item.type === 'bulk-sell-weak' || item.type === 'inventory-sell-reset');
      if (sellRuleRegion) return {
        type: 'sellRule',
        key: `sellRule:${sellRuleRegion.type}:${sellRuleRegion.ruleId || sellRuleRegion.rarity || ''}`,
        sourcePanel: 'inventory',
        label: sellRuleRegion.label || '',
        tooltip: sellRuleRegion.tooltip || ''
      };
      const cardRegion = findHoverRegion((item) => item.type === 'card-item' || item.type === 'card-equip');
      if (cardRegion && cardRegion.uid) return { type: 'card', key: `card:${cardRegion.uid}`, sourcePanel: 'inventory', uid: cardRegion.uid, slotIndex: cardRegion.slotIndex };
      const itemRegion = findHoverRegion((item) => item.type === 'inventory-item');
      if (itemRegion && itemRegion.uid) return { type: 'item', key: `item:${itemRegion.uid}`, sourcePanel: 'inventory', uid: itemRegion.uid };
      const consumableRegion = findHoverRegion((item) => item.type === 'consumable-item');
      if (consumableRegion && consumableRegion.itemId) return { type: 'consumable', key: `consumable:${consumableRegion.itemId}`, sourcePanel: 'inventory', itemId: consumableRegion.itemId };
      const etcRegion = findHoverRegion((item) => item.type === 'etc-item');
      if (etcRegion && etcRegion.materialId) return { type: 'etc', key: `etc:${etcRegion.materialId}`, sourcePanel: 'inventory', materialId: etcRegion.materialId };
    }
    if (openWindows.includes('storage')) {
      const itemRegion = findHoverRegion((item) => item.type === 'storage-item');
      if (itemRegion && itemRegion.uid) return { type: 'item', key: `storage-item:${itemRegion.uid}`, sourcePanel: 'storage', uid: itemRegion.uid };
      const consumableRegion = findHoverRegion((item) => item.type === 'storage-consumable-item');
      if (consumableRegion && consumableRegion.itemId) return { type: 'consumable', key: `storage-consumable:${consumableRegion.itemId}`, sourcePanel: 'storage', itemId: consumableRegion.itemId };
      const etcRegion = findHoverRegion((item) => item.type === 'storage-etc-item');
      if (etcRegion && etcRegion.materialId) return { type: 'etc', key: `storage-etc:${etcRegion.materialId}`, sourcePanel: 'storage', materialId: etcRegion.materialId };
    }
    if (openWindows.includes('character')) {
      const statRegion = findHoverRegion((item) => item.type === 'equipment-stat');
      if (statRegion && statRegion.statKey) return { type: 'equipmentStat', key: `characterStat:${statRegion.statKey}`, sourcePanel: 'character', statKey: statRegion.statKey, label: statRegion.label, value: statRegion.value };
    }
    if (openWindows.includes('equipment')) {
      const cardRegion = findHoverRegion((item) => item.type === 'card-deck-slot' || item.type === 'card-item' || item.type === 'card-equip');
      if (cardRegion && cardRegion.uid) return { type: 'card', key: `equipmentCard:${cardRegion.uid}:${cardRegion.slotIndex == null ? '' : cardRegion.slotIndex}`, sourcePanel: 'equipment', uid: cardRegion.uid, slotIndex: cardRegion.slotIndex };
      const statRegion = findHoverRegion((item) => item.type === 'equipment-stat');
      if (statRegion && statRegion.statKey) return { type: 'equipmentStat', key: `equipmentStat:${statRegion.statKey}`, sourcePanel: 'equipment', statKey: statRegion.statKey, label: statRegion.label, value: statRegion.value };
      const equipmentRegion = findHoverRegion((item) => item.type === 'equipment-item');
      if (equipmentRegion && equipmentRegion.slot) return { type: 'item', key: `equipment:${equipmentRegion.slot}`, sourcePanel: 'equipment', slot: equipmentRegion.slot };
    }
    if (openWindows.includes('worldmap')) {
      const nodeRegion = findHoverRegion((item) => item.type === 'world-map-node');
      if (nodeRegion && nodeRegion.mapId) return getInfoTarget(nodeRegion, {
        type: 'worldMapNode',
        key: `worldMapNode:${nodeRegion.mapId}`,
        sourcePanel: 'worldmap',
        mapId: nodeRegion.mapId
      });
    }
    const infoRegion = findHoverRegion((item) => item.tooltipTitle || item.tooltipSubtitle || item.tooltipLines);
    if (infoRegion) return getInfoTarget(infoRegion);
    if (settings.upgradePromptOpen) {
      const upgradeAideRegion = findHoverRegion((item) => item.type === 'upgrade-aide-toggle');
      if (upgradeAideRegion && upgradeAideRegion.aideId) {
        return { type: 'upgradeAide', key: `upgradeAide:${upgradeAideRegion.aideId}:${upgradeAideRegion.owned || 0}:${upgradeAideRegion.disabledReason || ''}`, sourcePanel: 'upgrade', aideId: upgradeAideRegion.aideId };
      }
    }
    return getEmptyCanvasHoverTarget();
  }

  function getActiveCanvasHoverTooltipRenderer(hoverTarget) {
    const hoverType = hoverTarget && hoverTarget.type || '';
    if (hoverType === 'skill') return 'skill';
    if (hoverType === 'equipmentStat') return 'equipmentStat';
    if (hoverType === 'sellRule') return 'sellRule';
    if (hoverType === 'item') return 'item';
    if (hoverType === 'card') return 'card';
    if (hoverType === 'consumable') return 'consumable';
    if (hoverType === 'etc') return 'etc';
    if (hoverType === 'upgradeAide') return 'upgradeAide';
    if (hoverType === 'info' || hoverType === 'worldMapNode') return 'info';
    return '';
  }

  function getCanvasHoverRegionCacheKey(options) {
    const settings = options || {};
    const openWindows = Array.isArray(settings.openWindows) ? settings.openWindows : [];
    const windowState = settings.windowState || {};
    const windows = openWindows.map((panelId) => {
      const win = windowState && windowState[panelId];
      return win ? [
        panelId,
        Math.round(Number(win.x || 0)),
        Math.round(Number(win.y || 0)),
        Math.round(Number(win.w || 0)),
        Math.round(Number(win.h || 0)),
        Math.round(Number(win.scroll || 0)),
        Number(win.z || 0)
      ].join(':') : panelId;
    }).join('|');
    const canvasHitRegionCount = Array.isArray(settings.canvasHitRegions)
      ? settings.canvasHitRegions.length
      : Number(settings.canvasHitRegionCount || 0);
    return [
      windows,
      canvasHitRegionCount,
      settings.overlayModalKey || '',
      settings.isCommandOpen ? 'command' : '',
      settings.inventorySellSettingsOpen ? 'sellSettings' : '',
      settings.storageTab || '',
      settings.petPotionPickerKind || ''
    ].join('~');
  }

  function getClearedCanvasHoverMeta() {
    return { x: NaN, y: NaN, key: '', target: getEmptyCanvasHoverTarget() };
  }

  function getCanvasPointerLeaveAction() {
    return {
      handled: true,
      pointer: { x: -9999, y: -9999 },
      hoverTarget: getEmptyCanvasHoverTarget(),
      shouldCancelPointer: true
    };
  }

  function getCanvasHoverRefreshForDraw(point, cache, options) {
    const settings = options || {};
    const hoverPoint = point || { x: -9999, y: -9999 };
    const hoverKey = settings.hoverKey || '';
    const sourceCache = cache || {};
    if (sourceCache.key === hoverKey && sourceCache.x === hoverPoint.x && sourceCache.y === hoverPoint.y) {
      return {
        target: sourceCache.target || getEmptyCanvasHoverTarget(),
        cache: sourceCache,
        reused: true
      };
    }
    const target = typeof settings.getCanvasHoverTargetAt === 'function'
      ? settings.getCanvasHoverTargetAt(hoverPoint)
      : getEmptyCanvasHoverTarget();
    return {
      target,
      cache: { x: hoverPoint.x, y: hoverPoint.y, key: hoverKey, target },
      reused: false
    };
  }

  function getCanvasHoverTargetUpdate(currentTarget, point, options) {
    const settings = options || {};
    const hoverPoint = point || {};
    const nextTarget = typeof settings.getCanvasHoverTargetAt === 'function'
      ? settings.getCanvasHoverTargetAt(hoverPoint)
      : getEmptyCanvasHoverTarget();
    const cache = {
      x: hoverPoint && hoverPoint.x,
      y: hoverPoint && hoverPoint.y,
      key: settings.hoverKey || '',
      target: nextTarget
    };
    const previousKey = currentTarget && currentTarget.key || '';
    const nextKey = nextTarget.key || '';
    return {
      target: nextTarget,
      cache,
      changed: previousKey !== nextKey
    };
  }

  function createCanvasHoverUiHelpers() {
    return Object.freeze({
      getEmptyCanvasHoverTarget,
      getInfoHoverTarget,
      getCanvasHoverTargetAt,
      getActiveCanvasHoverTooltipRenderer,
      getCanvasHoverRegionCacheKey,
      getClearedCanvasHoverMeta,
      getCanvasPointerLeaveAction,
      getCanvasHoverRefreshForDraw,
      getCanvasHoverTargetUpdate
    });
  }

  const api = {
    createCanvasHoverUiHelpers,
    getEmptyCanvasHoverTarget,
    getInfoHoverTarget,
    getCanvasHoverTargetAt,
    getActiveCanvasHoverTooltipRenderer,
    getCanvasHoverRegionCacheKey,
    getClearedCanvasHoverMeta,
    getCanvasPointerLeaveAction,
    getCanvasHoverRefreshForDraw,
    getCanvasHoverTargetUpdate
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.canvasHover = Object.assign({}, modules.canvasHover || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
