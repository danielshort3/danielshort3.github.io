(function initProjectStarfallUiCanvasRegions(global) {
  'use strict';

  const CANVAS_MODAL_BLOCKER_TYPES = new Set([
    'quest-prompt-shell',
    'gear-picker-shell',
    'upgrade-prompt-shell',
    'potential-prompt-shell',
    'potential-help-shell',
    'shard-craft-shell',
    'drop-quantity-shell',
    'admin-number-shell',
    'confirm-prompt-shell'
  ]);
  const CANVAS_CLIP_BYPASS_TYPES = new Set(['window-shell', 'window-header', 'window-body', 'close-window']);

  function pointInRectFallback(point, rect) {
    return !!(point && rect && point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h);
  }

  function getPointInRect(options) {
    return options && typeof options.pointInRect === 'function'
      ? options.pointInRect
      : pointInRectFallback;
  }

  function isHudTooltipCanvasRegion(region) {
    return !!region && (region.type === 'hud-buff' || region.type === 'hud-cooldown');
  }

  function getCanvasHudTooltipRegionCacheValue(regions, cache, options) {
    const sourceRegions = Array.isArray(regions) ? regions : [];
    const isHudTooltipRegion = options && typeof options.isHudTooltipCanvasRegion === 'function'
      ? options.isHudTooltipCanvasRegion
      : isHudTooltipCanvasRegion;
    if (!sourceRegions.length) {
      return {
        hasHudTooltipRegions: false,
        cache: { regions: sourceRegions, length: sourceRegions.length, hasHudTooltipRegions: false }
      };
    }
    if (cache && cache.regions === sourceRegions && cache.length === sourceRegions.length) {
      return { hasHudTooltipRegions: !!cache.hasHudTooltipRegions, cache };
    }
    let hasHudTooltipRegions = false;
    for (let index = 0; index < sourceRegions.length; index += 1) {
      if (isHudTooltipRegion(sourceRegions[index])) {
        hasHudTooltipRegions = true;
        break;
      }
    }
    return {
      hasHudTooltipRegions,
      cache: { regions: sourceRegions, length: sourceRegions.length, hasHudTooltipRegions }
    };
  }

  function getCanvasHudTooltipRegionCacheForAdd(entry, regions, cache, options) {
    const sourceRegions = Array.isArray(regions) ? regions : [];
    const isHudTooltipRegion = options && typeof options.isHudTooltipCanvasRegion === 'function'
      ? options.isHudTooltipCanvasRegion
      : isHudTooltipCanvasRegion;
    const hasHudTooltipRegion = isHudTooltipRegion(entry);
    if (cache && cache.regions === sourceRegions && cache.length === sourceRegions.length - 1) {
      cache.length = sourceRegions.length;
      if (hasHudTooltipRegion) cache.hasHudTooltipRegions = true;
      return cache;
    }
    if (hasHudTooltipRegion) {
      return { regions: sourceRegions, length: sourceRegions.length, hasHudTooltipRegions: true };
    }
    return null;
  }

  function getCanvasRegionEntry(region, options) {
    const settings = options || {};
    const entry = Object.assign({}, region);
    if (settings.currentCanvasPanelId && !entry.panelId && !String(entry.type || '').startsWith('menu-')) {
      entry.panelId = settings.currentCanvasPanelId;
    }
    const windowState = settings.windowState || {};
    if (entry.panelId && typeof entry.z === 'undefined' && windowState[entry.panelId]) {
      entry.z = windowState[entry.panelId].z || 0;
    }
    const clip = settings.activeCanvasClip;
    if (clip && !CANVAS_CLIP_BYPASS_TYPES.has(entry.type)) {
      const clippedX = Math.max(entry.x, clip.x);
      const clippedY = Math.max(entry.y, clip.y);
      const clippedRight = Math.min(entry.x + entry.w, clip.x + clip.w);
      const clippedBottom = Math.min(entry.y + entry.h, clip.y + clip.h);
      if (clippedRight <= clippedX || clippedBottom <= clippedY) {
        return { entry, shouldAdd: false };
      }
      entry.x = clippedX;
      entry.y = clippedY;
      entry.w = clippedRight - clippedX;
      entry.h = clippedBottom - clippedY;
    }
    return { entry, shouldAdd: true };
  }

  function isCanvasModalBlocker(region) {
    return !!region && CANVAS_MODAL_BLOCKER_TYPES.has(region.type);
  }

  function isCanvasBlockingRegion(region) {
    const type = String(region && region.type || '');
    return !!(
      isCanvasModalBlocker(region) ||
      type === 'window-shell' ||
      type === 'menu-shell' ||
      type === 'item-context-shell'
    );
  }

  function getCanvasBlockingRegions(regions, cache, options) {
    const sourceRegions = Array.isArray(regions) ? regions : [];
    const isBlockingRegion = options && typeof options.isCanvasBlockingRegion === 'function'
      ? options.isCanvasBlockingRegion
      : isCanvasBlockingRegion;
    if (!sourceRegions.length) return { blockers: [], cache };
    if (cache && cache.regions === sourceRegions && cache.length === sourceRegions.length) {
      return { blockers: cache.blockers, cache };
    }
    const blockers = [];
    for (let index = 0; index < sourceRegions.length; index += 1) {
      const region = sourceRegions[index];
      if (isBlockingRegion(region)) blockers.push(region);
    }
    return { blockers, cache: { regions: sourceRegions, length: sourceRegions.length, blockers } };
  }

  function getActiveCanvasModalBlocker(blockers, options) {
    const sourceBlockers = Array.isArray(blockers) ? blockers : [];
    const isModalBlocker = options && typeof options.isCanvasModalBlocker === 'function'
      ? options.isCanvasModalBlocker
      : isCanvasModalBlocker;
    for (let index = sourceBlockers.length - 1; index >= 0; index -= 1) {
      const region = sourceBlockers[index];
      if (isModalBlocker(region)) return region;
    }
    return null;
  }

  function canvasRegionBelongsToBlocker(region, blocker) {
    const type = String(region && region.type || '');
    const blockerType = String(blocker && blocker.type || '');
    if (!blockerType) return true;
    if (blockerType === 'gear-picker-shell') return type.startsWith('gear-picker');
    if (blockerType === 'quest-prompt-shell') return type.startsWith('quest-prompt');
    if (blockerType === 'upgrade-prompt-shell') return type.startsWith('upgrade-prompt') || type === 'upgrade-aide-toggle' || type === 'gear-picker-open';
    if (blockerType === 'potential-prompt-shell') return type.startsWith('potential-prompt') || type === 'potential-choice' || type === 'potential-repeat' || type === 'potential-line-upgrade' || type === 'potential-auto-toggle' || type === 'potential-auto-close' || type === 'potential-auto-stat-cycle' || type === 'potential-auto-tier-cycle' || type === 'potential-auto-repeat' || type === 'potential-auto-run' || type === 'gear-picker-open';
    if (blockerType === 'potential-help-shell') return type.startsWith('potential-help');
    if (blockerType === 'shard-craft-shell') return type.startsWith('shard-craft') || type === 'combine-cube-fragments' || type === 'combine-preservation-cube-fragments';
    if (blockerType === 'drop-quantity-shell') return type.startsWith('drop-quantity');
    if (blockerType === 'admin-number-shell') return type.startsWith('admin-number');
    if (blockerType === 'confirm-prompt-shell') return type.startsWith('confirm-prompt');
    if (blockerType === 'item-context-shell') return type.startsWith('item-context');
    if (blockerType === 'menu-shell') return type.startsWith('menu-');
    if (blockerType === 'window-shell') {
      if (region && region.panelId) return region.panelId === blocker.panelId;
      return type.startsWith('menu-');
    }
    return true;
  }

  function isCanvasCommandRegion(region) {
    const type = String(region && region.type || '');
    return type === 'menu-shell' || type.startsWith('menu-');
  }

  function getCanvasRegionPriority(region) {
    const priority = Number(region && region.priority);
    return Number.isFinite(priority) ? priority : 0;
  }

  function getCachedRegionQuery(cache, regions, queryX, queryY) {
    return cache &&
      cache.regions === regions &&
      cache.length === regions.length &&
      cache.x === queryX &&
      cache.y === queryY
        ? cache
        : null;
  }

  function getCanvasRegionsAtPoint(point, options) {
    const settings = options || {};
    const regions = Array.isArray(settings.regions) ? settings.regions : [];
    if (!regions.length) return { hits: [], cache: settings.cache || null };
    const queryPoint = point || {};
    const queryX = Number(queryPoint.x || 0);
    const queryY = Number(queryPoint.y || 0);
    const cachedForPoint = getCachedRegionQuery(settings.cache, regions, queryX, queryY);
    if (cachedForPoint && Array.isArray(cachedForPoint.hits)) {
      return { hits: cachedForPoint.hits, cache: cachedForPoint };
    }
    const pointInRect = getPointInRect(settings);
    const belongsToBlocker = typeof settings.canvasRegionBelongsToBlocker === 'function'
      ? settings.canvasRegionBelongsToBlocker
      : canvasRegionBelongsToBlocker;
    const getPriority = typeof settings.getCanvasRegionPriority === 'function'
      ? settings.getCanvasRegionPriority
      : getCanvasRegionPriority;
    const topBlocker = cachedForPoint
      ? cachedForPoint.topBlocker
      : typeof settings.getTopmostCanvasBlockerAt === 'function'
        ? settings.getTopmostCanvasBlockerAt(queryPoint)
        : settings.topBlocker || null;
    const hits = [];
    for (let index = regions.length - 1; index >= 0; index -= 1) {
      const region = regions[index];
      if (!pointInRect(queryPoint, region)) continue;
      if (topBlocker && !belongsToBlocker(region, topBlocker)) continue;
      hits.push(region);
    }
    if (hits.length > 1) {
      hits.sort((a, b) => getPriority(b) - getPriority(a));
    }
    const nextCache = {
      regions,
      length: regions.length,
      x: queryX,
      y: queryY,
      topBlocker,
      hits,
      topHit: hits[0] || null,
      topHitComputed: true
    };
    return { hits, cache: nextCache };
  }

  function findCanvasRegionInList(regions, filter) {
    const list = Array.isArray(regions) ? regions : [];
    for (let index = 0; index < list.length; index += 1) {
      const region = list[index];
      if (!filter || filter(region)) return region;
    }
    return null;
  }

  function findCanvasRegion(point, options) {
    const settings = options || {};
    const regions = Array.isArray(settings.regions) ? settings.regions : [];
    if (!regions.length) return { region: null, cache: settings.cache || null };
    const queryPoint = point || {};
    const queryX = Number(queryPoint.x || 0);
    const queryY = Number(queryPoint.y || 0);
    const filter = typeof settings.filter === 'function' ? settings.filter : null;
    const cachedForPoint = getCachedRegionQuery(settings.cache, regions, queryX, queryY);
    if (cachedForPoint && Array.isArray(cachedForPoint.hits)) {
      return { region: findCanvasRegionInList(cachedForPoint.hits, filter), cache: cachedForPoint };
    }
    if (!filter && cachedForPoint && cachedForPoint.topHitComputed) {
      return { region: cachedForPoint.topHit || null, cache: cachedForPoint };
    }
    const pointInRect = getPointInRect(settings);
    const belongsToBlocker = typeof settings.canvasRegionBelongsToBlocker === 'function'
      ? settings.canvasRegionBelongsToBlocker
      : canvasRegionBelongsToBlocker;
    const getPriority = typeof settings.getCanvasRegionPriority === 'function'
      ? settings.getCanvasRegionPriority
      : getCanvasRegionPriority;
    const topBlocker = cachedForPoint
      ? cachedForPoint.topBlocker
      : typeof settings.getTopmostCanvasBlockerAt === 'function'
        ? settings.getTopmostCanvasBlockerAt(queryPoint)
        : settings.topBlocker || null;
    let best = null;
    let bestPriority = -Infinity;
    for (let index = regions.length - 1; index >= 0; index -= 1) {
      const region = regions[index];
      if (!pointInRect(queryPoint, region)) continue;
      if (topBlocker && !belongsToBlocker(region, topBlocker)) continue;
      if (filter && !filter(region)) continue;
      const priority = getPriority(region);
      if (!best || priority > bestPriority) {
        best = region;
        bestPriority = priority;
      }
    }
    const topHitComputed = !filter || !!(cachedForPoint && cachedForPoint.topHitComputed);
    const nextCache = {
      regions,
      length: regions.length,
      x: queryX,
      y: queryY,
      topBlocker,
      hits: null,
      topHit: !filter ? best : (cachedForPoint && cachedForPoint.topHit || null),
      topHitComputed
    };
    return { region: best, cache: nextCache };
  }

  function getCanvasWheelRegionsAtPoint(point, options) {
    const settings = options || {};
    const regions = Array.isArray(settings.regions) ? settings.regions : [];
    const queryPoint = point || {};
    const pointInRect = getPointInRect(settings);
    const belongsToBlocker = typeof settings.canvasRegionBelongsToBlocker === 'function'
      ? settings.canvasRegionBelongsToBlocker
      : canvasRegionBelongsToBlocker;
    const getPriority = typeof settings.getCanvasRegionPriority === 'function'
      ? settings.getCanvasRegionPriority
      : getCanvasRegionPriority;
    let potentialHelpRegion = null;
    let potentialHelpPriority = -Infinity;
    let pickerRegion = null;
    let pickerPriority = -Infinity;
    let monsterListRegion = null;
    let monsterListPriority = -Infinity;
    let assetCategoryRegion = null;
    let assetCategoryPriority = -Infinity;
    let assetEntryRegion = null;
    let assetEntryPriority = -Infinity;
    let inventoryGridRegion = null;
    let inventoryGridPriority = -Infinity;
    let storageGridRegion = null;
    let storageGridPriority = -Infinity;
    let windowBodyRegion = null;
    let windowBodyPriority = -Infinity;
    const emptyResult = () => ({
      potentialHelpRegion,
      pickerRegion,
      monsterListRegion,
      assetCategoryRegion,
      assetEntryRegion,
      inventoryGridRegion,
      storageGridRegion,
      windowBodyRegion
    });
    if (!regions.length) return emptyResult();
    const topBlocker = typeof settings.getTopmostCanvasBlockerAt === 'function'
      ? settings.getTopmostCanvasBlockerAt(queryPoint)
      : settings.topBlocker || null;
    for (let index = regions.length - 1; index >= 0; index -= 1) {
      const region = regions[index];
      if (!pointInRect(queryPoint, region)) continue;
      if (topBlocker && !belongsToBlocker(region, topBlocker)) continue;
      const priority = getPriority(region);
      if (region.type === 'potential-help-body') {
        if (!potentialHelpRegion || priority > potentialHelpPriority) {
          potentialHelpRegion = region;
          potentialHelpPriority = priority;
        }
      } else if (region.type === 'gear-picker-body') {
        if (!pickerRegion || priority > pickerPriority) {
          pickerRegion = region;
          pickerPriority = priority;
        }
      } else if (region.type === 'monster-guide-list-scroll') {
        if (!monsterListRegion || priority > monsterListPriority) {
          monsterListRegion = region;
          monsterListPriority = priority;
        }
      } else if (region.type === 'asset-preview-category-scroll') {
        if (!assetCategoryRegion || priority > assetCategoryPriority) {
          assetCategoryRegion = region;
          assetCategoryPriority = priority;
        }
      } else if (region.type === 'asset-preview-entry-scroll') {
        if (!assetEntryRegion || priority > assetEntryPriority) {
          assetEntryRegion = region;
          assetEntryPriority = priority;
        }
      } else if (region.type === 'inventory-grid-scroll') {
        if (!inventoryGridRegion || priority > inventoryGridPriority) {
          inventoryGridRegion = region;
          inventoryGridPriority = priority;
        }
      } else if (region.type === 'storage-grid-scroll') {
        if (!storageGridRegion || priority > storageGridPriority) {
          storageGridRegion = region;
          storageGridPriority = priority;
        }
      } else if (region.type === 'window-body') {
        if (!windowBodyRegion || priority > windowBodyPriority) {
          windowBodyRegion = region;
          windowBodyPriority = priority;
        }
      }
    }
    return emptyResult();
  }

  function getCanvasEmptyRegionReleaseAction(options) {
    const settings = options || {};
    const shouldBlurMonsterGuideSearch = !!settings.monsterGuideSearchFocused;
    return {
      handled: true,
      type: 'emptyRegionRelease',
      shouldClearSelectedBind: true,
      shouldBlurMonsterGuideSearch,
      shouldDraw: shouldBlurMonsterGuideSearch,
      drawOptions: shouldBlurMonsterGuideSearch ? { force: true } : null,
      shouldReturnEarly: true
    };
  }

  function getCanvasPointerReleaseGateAction(region, options) {
    const settings = options || {};
    if (region && region.type === 'menu-dismiss') {
      return {
        handled: true,
        type: 'menuDismissRelease',
        shouldPreventDefault: true,
        shouldDraw: false,
        drawOptions: null,
        shouldReturnEarly: true
      };
    }
    if (settings.bypassedModal) {
      return {
        handled: true,
        type: 'modalBypassRelease',
        shouldPreventDefault: true,
        shouldDraw: false,
        drawOptions: null,
        shouldReturnEarly: true
      };
    }
    if (settings.consumedPlinkoHoldClick) {
      return {
        handled: true,
        type: 'consumedPlinkoHoldRelease',
        shouldPreventDefault: true,
        shouldDraw: true,
        drawOptions: null,
        shouldReturnEarly: true
      };
    }
    return {
      handled: false,
      type: '',
      shouldPreventDefault: false,
      shouldDraw: false,
      drawOptions: null,
      shouldReturnEarly: false
    };
  }

  function getCanvasPointerCancelDragCleanupAction() {
    return {
      handled: true,
      type: 'pointerCancelDragCleanup',
      shouldStopPlinkoDropHold: true,
      shouldClearActivePanel: true,
      dragKeys: ['canvasDrag', 'canvasBindDrag', 'canvasGearDrag', 'canvasInventoryDrag']
    };
  }

  function getCanvasPointerCancelFinalCleanupAction() {
    return {
      handled: true,
      type: 'pointerCancelFinalCleanup',
      shouldClearCanvasDown: true,
      shouldClearCanvasDownModalBypass: true,
      clickKeys: ['canvasInventoryClick'],
      shouldDraw: true,
      drawOptions: null
    };
  }

  function getCanvasContextMenuStartAction(options) {
    const settings = options || {};
    if (!settings.hasEvent) {
      return {
        handled: false,
        type: '',
        shouldPreventDefault: false,
        shouldStopPropagation: false,
        shouldCloseItemContextMenu: false,
        closeOptions: null
      };
    }
    return {
      handled: true,
      type: 'contextMenuStart',
      shouldPreventDefault: true,
      shouldStopPropagation: true,
      shouldCloseItemContextMenu: !!settings.hasItemContextMenu,
      closeOptions: { skipCanvas: true }
    };
  }

  function getCanvasContextMenuRegionHitAction(region) {
    if (!region) {
      return {
        handled: false,
        type: '',
        shouldFocusCanvas: false,
        shouldReturnEarly: false
      };
    }
    return {
      handled: true,
      type: 'focusCanvasRegion',
      shouldFocusCanvas: true,
      shouldReturnEarly: true
    };
  }

  function getCanvasContextMenuPortalAction(portal, didStartTransition) {
    if (!portal || !didStartTransition) {
      return {
        handled: false,
        type: '',
        portalId: portal && portal.id || '',
        shouldFocusCanvas: false,
        shouldReturnEarly: false
      };
    }
    return {
      handled: true,
      type: 'portalTransition',
      portalId: portal.id,
      shouldFocusCanvas: true,
      shouldReturnEarly: true
    };
  }

  function getCanvasContextInteractionRequestAction(event, options) {
    const settings = options || {};
    const hasContextInteractAtCanvasPoint = !!settings.hasContextInteractAtCanvasPoint;
    return {
      handled: true,
      type: hasContextInteractAtCanvasPoint ? 'contextInteractAtCanvasPoint' : 'interactFallback',
      x: event && event.clientX,
      y: event && event.clientY,
      shouldCallContextInteract: hasContextInteractAtCanvasPoint,
      shouldCallInteractFallback: !hasContextInteractAtCanvasPoint,
      fallbackPanel: true
    };
  }

  function getCanvasContextInteractionResultAction(result, options) {
    const settings = options || {};
    if (result && result.panel) {
      const panelId = result.panelId || settings.lastInteractionPanelId || settings.engineSelectedPanelId || settings.snapshotSelectedPanelId || '';
      return {
        handled: true,
        type: panelId ? 'openPanel' : 'panelResultNoPanel',
        panelId,
        questPrompt: null,
        shouldOpenPanel: !!panelId,
        shouldSetQuestPrompt: false,
        shouldFocusCanvas: false
      };
    }
    if (result && result.questPrompt) {
      return {
        handled: true,
        type: 'setQuestPrompt',
        panelId: '',
        questPrompt: result.questPrompt,
        shouldOpenPanel: false,
        shouldSetQuestPrompt: true,
        shouldFocusCanvas: false
      };
    }
    return {
      handled: true,
      type: 'focusCanvas',
      panelId: '',
      questPrompt: null,
      shouldOpenPanel: false,
      shouldSetQuestPrompt: false,
      shouldFocusCanvas: true
    };
  }

  function createCanvasRegionUiHelpers() {
    return Object.freeze({
      isHudTooltipCanvasRegion,
      getCanvasHudTooltipRegionCacheValue,
      getCanvasHudTooltipRegionCacheForAdd,
      getCanvasRegionEntry,
      isCanvasModalBlocker,
      isCanvasBlockingRegion,
      getCanvasBlockingRegions,
      getActiveCanvasModalBlocker,
      canvasRegionBelongsToBlocker,
      isCanvasCommandRegion,
      getCanvasRegionPriority,
      getCanvasRegionsAtPoint,
      findCanvasRegionInList,
      findCanvasRegion,
      getCanvasWheelRegionsAtPoint,
      getCanvasEmptyRegionReleaseAction,
      getCanvasPointerReleaseGateAction,
      getCanvasPointerCancelDragCleanupAction,
      getCanvasPointerCancelFinalCleanupAction,
      getCanvasContextMenuStartAction,
      getCanvasContextMenuRegionHitAction,
      getCanvasContextMenuPortalAction,
      getCanvasContextInteractionRequestAction,
      getCanvasContextInteractionResultAction
    });
  }

  const api = {
    CANVAS_MODAL_BLOCKER_TYPES,
    CANVAS_CLIP_BYPASS_TYPES,
    createCanvasRegionUiHelpers,
    isHudTooltipCanvasRegion,
    getCanvasHudTooltipRegionCacheValue,
    getCanvasHudTooltipRegionCacheForAdd,
    getCanvasRegionEntry,
    isCanvasModalBlocker,
    isCanvasBlockingRegion,
    getCanvasBlockingRegions,
    getActiveCanvasModalBlocker,
    canvasRegionBelongsToBlocker,
    isCanvasCommandRegion,
    getCanvasRegionPriority,
    getCanvasRegionsAtPoint,
    findCanvasRegionInList,
    findCanvasRegion,
    getCanvasWheelRegionsAtPoint,
    getCanvasEmptyRegionReleaseAction,
    getCanvasPointerReleaseGateAction,
    getCanvasPointerCancelDragCleanupAction,
    getCanvasPointerCancelFinalCleanupAction,
    getCanvasContextMenuStartAction,
    getCanvasContextMenuRegionHitAction,
    getCanvasContextMenuPortalAction,
    getCanvasContextInteractionRequestAction,
    getCanvasContextInteractionResultAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.canvasRegions = Object.assign({}, modules.canvasRegions || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
