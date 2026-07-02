(function initProjectStarfallUiCanvasWheel(global) {
  'use strict';

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeInventoryTabFallback(value) {
    const id = String(value || '').trim();
    return ['equipment', 'usable', 'etc', 'cards'].includes(id) ? id : 'equipment';
  }

  function normalizeAssetPreviewCategoryFallback(value) {
    const id = String(value || '').trim();
    return id || 'all';
  }

  function getInventoryCanvasWheelDelta(event, options) {
    const settings = options || {};
    const inventoryCanvasCellSize = Number(settings.inventoryCanvasCellSize || 36);
    const clamp = typeof settings.clamp === 'function' ? settings.clamp : clampValue;
    const modeScale = event && event.deltaMode === 1
      ? 16
      : event && event.deltaMode === 2
        ? inventoryCanvasCellSize * 3
        : 1;
    const raw = Number(event && event.deltaY || 0) * modeScale;
    const maxStep = inventoryCanvasCellSize * 0.7;
    return clamp(raw * 0.65, -maxStep, maxStep);
  }

  function getCanvasWheelRoute(wheelRegions, options) {
    const regions = wheelRegions || {};
    const settings = options || {};
    const event = settings.event || {};
    const deltaY = Number(event.deltaY || 0);
    const normalizeInventoryTab = typeof settings.normalizeInventoryTab === 'function'
      ? settings.normalizeInventoryTab
      : normalizeInventoryTabFallback;
    const normalizeAssetPreviewCategory = typeof settings.normalizeAssetPreviewCategory === 'function'
      ? settings.normalizeAssetPreviewCategory
      : normalizeAssetPreviewCategoryFallback;
    const getInventoryWheelDelta = typeof settings.getInventoryCanvasWheelDelta === 'function'
      ? settings.getInventoryCanvasWheelDelta
      : (inputEvent) => getInventoryCanvasWheelDelta(inputEvent, settings);

    if (regions.potentialHelpRegion && settings.hasPotentialHelpWindow) {
      return { type: 'potentialHelp', region: regions.potentialHelpRegion, deltaY, preventDefault: true, drawOptions: null };
    }
    if (regions.pickerRegion && settings.hasGearPickerWindow) {
      return { type: 'gearPicker', region: regions.pickerRegion, deltaY, preventDefault: true, drawOptions: null };
    }
    if (regions.monsterListRegion) {
      return { type: 'monsterList', region: regions.monsterListRegion, deltaY, preventDefault: true, drawOptions: { coalesce: true } };
    }
    if (regions.assetCategoryRegion) {
      return { type: 'assetCategory', region: regions.assetCategoryRegion, deltaY, preventDefault: true, drawOptions: null };
    }
    if (regions.assetEntryRegion) {
      return {
        type: 'assetEntry',
        region: regions.assetEntryRegion,
        category: normalizeAssetPreviewCategory(regions.assetEntryRegion.category),
        deltaY,
        preventDefault: true,
        drawOptions: null
      };
    }
    if (regions.inventoryGridRegion) {
      return {
        type: 'inventoryGrid',
        region: regions.inventoryGridRegion,
        tab: normalizeInventoryTab(regions.inventoryGridRegion.tabId),
        deltaY: getInventoryWheelDelta(event),
        preventDefault: true,
        drawOptions: { coalesce: true }
      };
    }
    if (regions.storageGridRegion) {
      return {
        type: 'storageGrid',
        region: regions.storageGridRegion,
        tab: normalizeInventoryTab(regions.storageGridRegion.tabId),
        deltaY,
        preventDefault: true,
        drawOptions: null
      };
    }
    const region = regions.windowBodyRegion;
    if (!region) return { type: '', handled: false };
    if (region.panelId === 'inventory' || region.panelId === 'storage') {
      return { type: 'windowBodyBlocked', region, preventDefault: true, drawOptions: null };
    }
    return { type: 'windowBody', region, panelId: region.panelId, deltaY, preventDefault: true, drawOptions: null };
  }

  function createCanvasWheelUiHelpers(options) {
    const defaults = options || {};
    function getWheelOptions(callOptions) {
      return Object.assign({}, defaults, callOptions || {});
    }
    return Object.freeze({
      getInventoryCanvasWheelDelta: (event, callOptions) => getInventoryCanvasWheelDelta(event, getWheelOptions(callOptions)),
      getCanvasWheelRoute: (wheelRegions, callOptions) => getCanvasWheelRoute(wheelRegions, getWheelOptions(callOptions))
    });
  }

  const api = {
    createCanvasWheelUiHelpers,
    getInventoryCanvasWheelDelta,
    getCanvasWheelRoute
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.canvasWheel = Object.assign({}, modules.canvasWheel || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
