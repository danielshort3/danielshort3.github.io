(function initProjectStarfallUiPanelCache(global) {
  'use strict';

  const UiModules = global.ProjectStarfallUiModules || {};
  const UiCanvasWindows = (typeof require === 'function' ? require('./canvas-windows.js') : null) || UiModules.canvasWindows || {};

  const DEFAULT_PANEL_CACHE_PREWARM_ORDER = Object.freeze([
    'character',
    'equipment',
    'skills',
    'quests',
    'guide',
    'partyPanel'
  ]);

  function createPanelCachePrewarmOrder(panelIds) {
    return Object.freeze(Array.isArray(panelIds) ? panelIds.slice() : []);
  }

  const PANEL_CACHE_PREWARM_ORDER = createPanelCachePrewarmOrder(DEFAULT_PANEL_CACHE_PREWARM_ORDER);
  const PANEL_CACHE_PREWARM_IDLE_TIMEOUT_MS = 160;
  const DEFAULT_INVENTORY_MARKUP_PREWARM_ORDER = Object.freeze(['equipment']);

  function createInventoryMarkupPrewarmOrder(tabIds) {
    return Object.freeze(Array.isArray(tabIds) ? tabIds.slice() : []);
  }

  const INVENTORY_MARKUP_PREWARM_ORDER = createInventoryMarkupPrewarmOrder(DEFAULT_INVENTORY_MARKUP_PREWARM_ORDER);
  const INVENTORY_MARKUP_PREWARM_BATCH_SIZE = 24;
  const INVENTORY_MARKUP_PREWARM_SLOT_BUDGET = 48;
  const INVENTORY_MARKUP_PREWARM_MIN_IDLE_MS = 3;
  const DEFAULT_CANVAS_PANEL_CACHEABLE_IDS = Object.freeze([
    'storage', 'equipment', 'character', 'skills', 'quests',
    'pet', 'monsters', 'partyPanel', 'shop', 'upgrade',
    'plinko', 'daily', 'cashShop', 'beta', 'guide', 'keybinds'
  ]);

  function createCanvasPanelCacheableIds(panelIds) {
    return new Set(Array.isArray(panelIds) ? panelIds : []);
  }

  const CANVAS_PANEL_CACHEABLE_IDS = createCanvasPanelCacheableIds(DEFAULT_CANVAS_PANEL_CACHEABLE_IDS);
  const CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT = 4;
  const DEFAULT_CANVAS_PANEL_CACHE_ENTRY_LIMITS = Object.freeze({
    inventory: 12,
    storage: 8,
    skills: 6,
    quests: 6,
    monsters: 6,
    shop: 6,
    upgrade: 6,
    daily: 6,
    guide: 6
  });

  function createCanvasPanelCacheEntryLimits(limits) {
    const source = limits && typeof limits === 'object' ? limits : {};
    return Object.freeze(Object.assign({}, source));
  }

  const CANVAS_PANEL_CACHE_ENTRY_LIMITS = createCanvasPanelCacheEntryLimits(DEFAULT_CANVAS_PANEL_CACHE_ENTRY_LIMITS);
  const DEFAULT_CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS = Object.freeze(['inventory', 'equipment', 'skills', 'worldmap']);

  function createCanvasOverlayPanelCacheBypassIds(panelIds) {
    return new Set(Array.isArray(panelIds) ? panelIds : []);
  }

  const CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS = createCanvasOverlayPanelCacheBypassIds(DEFAULT_CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS);
  const INVENTORY_TILE_CACHE_WARM_BUDGET = 8;
  const CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT = 180;

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isCanvasPanelCacheable(panelId) {
    return CANVAS_PANEL_CACHEABLE_IDS.has(panelId);
  }

  function getCanvasPanelCacheEntryLimit(panelId) {
    return CANVAS_PANEL_CACHE_ENTRY_LIMITS[panelId] || CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT;
  }

  function getCanvasPanelLayerCacheStore(panelId, options) {
    const id = String(panelId || '');
    const settings = options || {};
    const caches = settings.caches || {};
    if (!id) return null;
    const current = caches[id];
    if (current && current.entries instanceof Map) return current;
    const store = { entries: new Map() };
    if (current && current.canvas && current.key) store.entries.set(current.key, current);
    caches[id] = store;
    return store;
  }

  function trimCanvasPanelLayerCacheStore(store, limit) {
    if (!store || !(store.entries instanceof Map)) return;
    const maxEntries = Math.max(1, Math.floor(Number(limit || CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT) || CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT));
    while (store.entries.size > maxEntries) {
      const firstKey = store.entries.keys().next().value;
      store.entries.delete(firstKey);
    }
  }

  function getDeferredCanvasPanelBodyCacheRequest(panelId, bodyW, bodyH, scrollOffset, cacheKey, options) {
    const settings = options || {};
    const id = String(panelId || '');
    const cacheable = typeof settings.isCanvasPanelCacheable === 'function'
      ? settings.isCanvasPanelCacheable(id)
      : isCanvasPanelCacheable(id);
    if (!id || !cacheKey || !cacheable) return null;
    const store = settings.store || null;
    if (store && store.entries && store.entries.has(cacheKey)) return null;
    const queuedKeys = settings.queuedKeys || null;
    if (queuedKeys && queuedKeys.has(cacheKey)) return null;
    return {
      panelId: id,
      bodyW: Math.max(1, Math.round(Number(bodyW || 1))),
      bodyH: Math.max(1, Math.round(Number(bodyH || 1))),
      scrollOffset: Number(scrollOffset || 0),
      cacheKey
    };
  }

  function getSkillsContentCooldownCacheKey(cooldowns) {
    const source = Array.isArray(cooldowns) ? cooldowns : [];
    return source
      .filter((cooldown) => cooldown && cooldown.skillId)
      .map((cooldown) => `${cooldown.skillId}:${Math.ceil(Math.max(0, Number(cooldown.remaining || 0)) * 4)}`)
      .join(',');
  }

  function getSkillsContentCacheKey(bodyW, bodyH, scrollOffset, options) {
    const settings = options || {};
    const assetReadyCacheKey = typeof settings.getAssetReadyCacheKey === 'function'
      ? settings.getAssetReadyCacheKey()
      : settings.assetReadyCacheKey || '';
    const panelStateKey = typeof settings.getPanelStateKey === 'function'
      ? settings.getPanelStateKey('skills')
      : settings.panelStateKey || '';
    const cooldownCacheKey = typeof settings.getCooldownCacheKey === 'function'
      ? settings.getCooldownCacheKey()
      : settings.cooldownCacheKey || '';
    return [
      'skillsContent',
      Math.round(Number(bodyW || 0)),
      Math.round(Number(bodyH || 0)),
      Math.round(Number(scrollOffset || 0)),
      assetReadyCacheKey,
      panelStateKey,
      cooldownCacheKey
    ].join('::');
  }

  function isInventoryCanvasTileLayerCache(cacheName, options) {
    const settings = options || {};
    return settings.currentCanvasPanelId === 'inventory' && String(cacheName || '').startsWith('inventory');
  }

  function getCanvasTileLayerWarmBudget(windows, options) {
    const settings = options || {};
    const sourceWindows = Array.isArray(windows) ? windows : settings.openWindows || [];
    const open = Array.isArray(windows)
      ? sourceWindows.some((entry) => entry && entry.id === 'inventory')
      : sourceWindows.includes('inventory');
    if (!open) return 0;
    if (
      settings.canvasInventoryDrag ||
      settings.canvasGearDrag ||
      settings.canvasDrag ||
      settings.canvasResizeDrag
    ) return 0;
    return Number(settings.warmBudgetLimit || INVENTORY_TILE_CACHE_WARM_BUDGET);
  }

  function getCanvasTileLayerCacheDecision(cacheName, options) {
    const settings = options || {};
    if (settings.isBuildingCanvasPanelCache) return { bypass: true, nextWarmBudget: settings.currentWarmBudget, consumeWarmBudget: false };
    const isInventoryTileCache = typeof settings.isInventoryCanvasTileLayerCache === 'function'
      ? settings.isInventoryCanvasTileLayerCache(cacheName)
      : isInventoryCanvasTileLayerCache(cacheName, settings);
    if (!isInventoryTileCache) return { bypass: false, nextWarmBudget: settings.currentWarmBudget, consumeWarmBudget: false };
    const budget = Math.max(0, Math.floor(Number(settings.currentWarmBudget || 0) || 0));
    if (budget <= 0) return { bypass: true, nextWarmBudget: settings.currentWarmBudget, consumeWarmBudget: false };
    return { bypass: false, nextWarmBudget: budget - 1, consumeWarmBudget: true };
  }

  function getCanvasTileLayerCacheKey(width, height, assetReadyCacheKey, key) {
    return `${Math.max(1, Math.round(Number(width || 1)))}x${Math.max(1, Math.round(Number(height || 1)))}:${String(assetReadyCacheKey || '')}:${String(key || '')}`;
  }

  function getCanvasTileLayerCacheStore(cacheName, options) {
    const name = String(cacheName || '');
    const settings = options || {};
    const caches = settings.caches || {};
    if (!name) return null;
    if (!(caches[name] instanceof Map)) caches[name] = new Map();
    return caches[name];
  }

  function trimCanvasTileLayerCacheStore(cache, limit) {
    if (!(cache instanceof Map)) return;
    const maxEntries = Math.max(1, Math.floor(Number(limit || CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT) || CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT));
    while (cache.size > maxEntries) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
  }

  function getPanelPrewarmTarget(panelId, options) {
    const settings = options || {};
    const normalizeInventoryTab = settings.normalizeInventoryTab || function normalizeInventoryTabFallback(value) {
      return String(value || '').trim() || 'equipment';
    };
    const raw = String(panelId || '');
    const parts = raw.split(':');
    const id = String(parts[0] || '').trim();
    return {
      id,
      inventoryTab: id === 'inventory' && parts[1] ? normalizeInventoryTab(parts[1]) : ''
    };
  }

  function getCanvasPanelCacheKey(panelId, bodyW, bodyH, scrollOffset, options) {
    const settings = options || {};
    const id = String(panelId || '');
    const cacheable = typeof settings.isCanvasPanelCacheable === 'function'
      ? settings.isCanvasPanelCacheable(id)
      : isCanvasPanelCacheable(id);
    if (!cacheable) return '';
    if (id === 'plinko' && settings.hasPlinkoAnimations) return '';
    const assetReadyCacheKey = typeof settings.getAssetReadyCacheKey === 'function'
      ? settings.getAssetReadyCacheKey()
      : settings.assetReadyCacheKey || '';
    const panelStateKey = typeof settings.getPanelStateKey === 'function'
      ? settings.getPanelStateKey(id)
      : settings.panelStateKey || '';
    return [
      id,
      Math.round(Number(bodyW || 0)),
      Math.round(Number(bodyH || 0)),
      Math.round(Number(scrollOffset || 0)),
      assetReadyCacheKey,
      panelStateKey
    ].join('::');
  }

  function getOptionValue(settings, key, fallback) {
    return Object.prototype.hasOwnProperty.call(settings, key) ? settings[key] : fallback;
  }

  function getPanelSnapshotRevision(panelId, settings) {
    return typeof settings.getPanelSnapshotRevision === 'function'
      ? settings.getPanelSnapshotRevision(panelId)
      : getOptionValue(settings, 'revisionKey', '');
  }

  function getCanvasPanelCacheStateKey(panelId, state, options) {
    const id = String(panelId || '');
    const source = state || {};
    const settings = options || {};
    const player = source.player || {};
    const session = source.session || {};
    const revisionKey = () => getPanelSnapshotRevision(id, settings);
    if (id === 'equipment' || id === 'character') {
      const getEquipmentKey = typeof settings.getEquipmentKey === 'function'
        ? settings.getEquipmentKey
        : () => settings.equipmentKey || '';
      const normalizeCharacterPanelTab = typeof settings.normalizeCharacterPanelTab === 'function'
        ? settings.normalizeCharacterPanelTab
        : (value) => String(value || '').trim();
      return [
        revisionKey(),
        getEquipmentKey(source.equipment || {}),
        source.selectedInventoryUid || '',
        player.level || 0,
        player.classId || '',
        player.advancedClassId || '',
        id === 'character' ? normalizeCharacterPanelTab(settings.characterPanelTab) : ''
      ].join('|');
    }
    if (id === 'skills') {
      return [
        revisionKey(),
        settings.activeSkillOwner || '',
        player.skillPoints || 0,
        player.baseSkillPoints || 0,
        player.advancedSkillPoints || 0,
        player.classId || '',
        player.advancedClassId || ''
      ].join('|');
    }
    if (id === 'pet') {
      return [
        revisionKey(),
        settings.petPotionPickerKind || ''
      ].join('|');
    }
    if (id === 'worldmap') {
      const questTrackerState = settings.questTrackerState || {};
      return [
        revisionKey(),
        source.mapId || '',
        questTrackerState.compact ? 1 : 0
      ].join('|');
    }
    if (id === 'monsters') {
      return [
        revisionKey(),
        source.monsterGuide && source.monsterGuide.selectedEnemyId || '',
        settings.monsterGuideSearchQuery || '',
        settings.monsterGuideFilter || 'all',
        settings.monsterGuideSearchFocused ? 1 : 0,
        Math.round(Number(settings.monsterGuideListScroll || 0))
      ].join('|');
    }
    if (id === 'quests') {
      return [revisionKey(), session.questTab || 'quests'].join('|');
    }
    if (id === 'partyPanel') {
      return [revisionKey(), source.party && source.party.commandId || ''].join('|');
    }
    if (id === 'shop') {
      return [
        revisionKey(),
        player.activeStation || '',
        player.currency || 0,
        player.level || 0,
        player.classId || '',
        player.advancedClassId || ''
      ].join('|');
    }
    if (id === 'upgrade') {
      return [
        revisionKey(),
        (source.inventory || []).length,
        typeof settings.getEquipmentKey === 'function' ? settings.getEquipmentKey(source.equipment || {}) : settings.equipmentKey || '',
        source.selectedInventoryUid || '',
        settings.upgradePromptUid || '',
        Object.keys(settings.upgradeAideSelections || {}).sort().join(',')
      ].join('|');
    }
    if (id === 'cashShop' || id === 'beta' || id === 'guide') {
      return String(revisionKey());
    }
    if (id === 'keybinds') {
      return [
        revisionKey(),
        settings.selectedBindActionId || '',
        settings.draggingBindActionId || ''
      ].join('|');
    }
    return null;
  }

  function getDailyPanelCacheStateKey(daily, options) {
    const source = daily || {};
    const settings = options || {};
    return [
      getOptionValue(settings, 'revisionKey', ''),
      source.todayKey || '',
      source.lastClaimedDateKey || '',
      source.totalClaimedDays || 0,
      source.streak || 0,
      source.cycleDay || 0,
      source.claimable ? 1 : 0,
      source.claimedToday ? 1 : 0,
      source.disabledReason || '',
      (source.claimMilestones || []).map((milestone) => milestone.id).join(','),
      (source.milestones || []).map((milestone) => `${milestone.id}:${milestone.claimed ? 1 : 0}:${milestone.claimable ? 1 : 0}:${milestone.progress || 0}`).join(',')
    ].join('|');
  }

  function getPlinkoPanelCacheStateKey(plinko, options) {
    const source = plinko || {};
    const settings = options || {};
    const buyQuantityByBall = settings.buyQuantityByBall || {};
    const dropHold = settings.dropHold || null;
    const buyQuantities = Object.keys(buyQuantityByBall).sort()
      .map((ballId) => `${ballId}:${buyQuantityByBall[ballId]}`)
      .join(',');
    return [
      getOptionValue(settings, 'revisionKey', ''),
      source.selectedBallId || '',
      buyQuantities,
      dropHold && dropHold.active ? `${dropHold.ballId}:held` : '',
      Number(source.pity || 0),
      Number(source.activeDropCount || 0),
      source.dropDisabledReason || '',
      (source.balls || []).map((ball) => `${ball.id}:${ball.count}`).join(','),
      (source.pendingDrops || []).map((drop) => `${drop.id}:${drop.slotIndex}`).join(','),
      (source.prizeTray || []).map((entry) => `${entry.id}:${entry.claimable ? 1 : 0}`).join(','),
      (source.lastRewards || []).map((entry) => `${entry.createdAt}:${entry.slotId}`).join(',')
    ].join('|');
  }

  function hasBlockingPanelPrewarmUi(state) {
    const source = state || {};
    return !!(
      source.isCommandOpen ||
      source.openWindows && source.openWindows.length ||
      source.upgradePromptOpen ||
      source.potentialPromptOpen ||
      source.potentialHelpOpen ||
      source.shardCraftPromptOpen ||
      source.questPrompt ||
      source.dropQuantityPrompt ||
      source.adminNumberPrompt ||
      source.confirmPrompt ||
      source.gearPickerContext
    );
  }

  function canPrewarmPanelCaches(options) {
    const settings = options || {};
    const engine = settings.engine || null;
    const state = settings.state || {};
    const hasBlockingUi = typeof settings.hasBlockingPanelPrewarmUi === 'function'
      ? settings.hasBlockingPanelPrewarmUi
      : () => hasBlockingPanelPrewarmUi(settings.uiState);
    const isAssetPrewarmReady = typeof settings.isAssetPrewarmReady === 'function'
      ? settings.isAssetPrewarmReady
      : () => settings.assetPrewarmReady !== false;
    return !!(
      engine &&
      typeof engine.getOverlaySnapshot === 'function' &&
      state.player &&
      state.player.classId &&
      !(engine.shouldPausePixiIdlePrewarm && engine.shouldPausePixiIdlePrewarm()) &&
      !hasBlockingUi() &&
      isAssetPrewarmReady()
    );
  }

  function getPanelPrewarmBatchKey(options) {
    const settings = options || {};
    const state = settings.state || {};
    const player = state.player || {};
    const getPanelSnapshotRevision = typeof settings.getPanelSnapshotRevision === 'function'
      ? settings.getPanelSnapshotRevision
      : () => '';
    const assetReadyCacheKey = typeof settings.getAssetReadyCacheKey === 'function'
      ? settings.getAssetReadyCacheKey()
      : settings.assetReadyCacheKey || '';
    const viewportWidth = settings.viewportWidth;
    const viewportHeight = settings.viewportHeight;
    const fallbackWidth = settings.fallbackViewportWidth;
    const fallbackHeight = settings.fallbackViewportHeight;
    return [
      getPanelSnapshotRevision('character'),
      getPanelSnapshotRevision('inventory'),
      getPanelSnapshotRevision('equipment'),
      getPanelSnapshotRevision('skills'),
      getPanelSnapshotRevision('quests'),
      getPanelSnapshotRevision('worldmap'),
      getPanelSnapshotRevision('guide'),
      getPanelSnapshotRevision('partyPanel'),
      assetReadyCacheKey,
      Math.round(Number(viewportWidth || fallbackWidth || 0)),
      Math.round(Number(viewportHeight || fallbackHeight || 0)),
      state.mapId || '',
      player.classId || '',
      player.advancedClassId || '',
      player.level || 0
    ].join('|');
  }

  function getPanelPrewarmMetrics(panelId, options) {
    const settings = options || {};
    const getTarget = typeof settings.getPanelPrewarmTarget === 'function'
      ? settings.getPanelPrewarmTarget
      : (id) => getPanelPrewarmTarget(id, settings);
    const getDefaults = typeof settings.getWindowDefaults === 'function'
      ? settings.getWindowDefaults
      : (id) => UiCanvasWindows.getWindowDefaults(id, settings);
    const clamp = typeof settings.clamp === 'function' ? settings.clamp : clampValue;
    const target = getTarget(panelId);
    const defaults = getDefaults(target.id || panelId) || {};
    const fallbackWidth = settings.fallbackViewportWidth;
    const fallbackHeight = settings.fallbackViewportHeight;
    const width = settings.viewportWidth;
    const height = settings.viewportHeight;
    const bottomLimit = typeof settings.getCanvasUiBottom === 'function'
      ? settings.getCanvasUiBottom(width, height, 8)
      : settings.bottomLimit;
    const panelW = clamp(Number(defaults.w || 420), 180, Math.max(180, Number(width || fallbackWidth || 0) - 24));
    const panelH = clamp(Number(defaults.h || 360), 120, Math.max(120, Number(bottomLimit || height) - 16));
    return {
      bodyW: Math.max(1, Math.round(panelW - 28)),
      bodyH: Math.max(1, Math.round(panelH - 58))
    };
  }

  function getInventoryMarkupPrewarmBatchKey(options) {
    const settings = options || {};
    const inventoryCacheKey = typeof settings.getInventoryCacheKey === 'function'
      ? settings.getInventoryCacheKey('equipment')
      : settings.inventoryCacheKey || '';
    const assetReadyCacheKey = typeof settings.getAssetReadyCacheKey === 'function'
      ? settings.getAssetReadyCacheKey()
      : settings.assetReadyCacheKey || '';
    return [
      'inventoryMarkup',
      inventoryCacheKey,
      assetReadyCacheKey
    ].join('|');
  }

  function createPanelCacheUiHelpers() {
    return Object.freeze({
      isCanvasPanelCacheable,
      getCanvasPanelCacheEntryLimit,
      getCanvasPanelLayerCacheStore,
      trimCanvasPanelLayerCacheStore,
      getDeferredCanvasPanelBodyCacheRequest,
      getSkillsContentCooldownCacheKey,
      getSkillsContentCacheKey,
      isInventoryCanvasTileLayerCache,
      getCanvasTileLayerWarmBudget,
      getCanvasTileLayerCacheDecision,
      getCanvasTileLayerCacheKey,
      getCanvasTileLayerCacheStore,
      trimCanvasTileLayerCacheStore,
      getPanelPrewarmTarget,
      getCanvasPanelCacheKey,
      getCanvasPanelCacheStateKey,
      getDailyPanelCacheStateKey,
      getPlinkoPanelCacheStateKey,
      hasBlockingPanelPrewarmUi,
      canPrewarmPanelCaches,
      getPanelPrewarmBatchKey,
      getPanelPrewarmMetrics,
      getInventoryMarkupPrewarmBatchKey
    });
  }

  const api = {
    DEFAULT_PANEL_CACHE_PREWARM_ORDER,
    PANEL_CACHE_PREWARM_ORDER,
    createPanelCachePrewarmOrder,
    PANEL_CACHE_PREWARM_IDLE_TIMEOUT_MS,
    DEFAULT_INVENTORY_MARKUP_PREWARM_ORDER,
    INVENTORY_MARKUP_PREWARM_ORDER,
    createInventoryMarkupPrewarmOrder,
    INVENTORY_MARKUP_PREWARM_BATCH_SIZE,
    INVENTORY_MARKUP_PREWARM_SLOT_BUDGET,
    INVENTORY_MARKUP_PREWARM_MIN_IDLE_MS,
    DEFAULT_CANVAS_PANEL_CACHEABLE_IDS,
    CANVAS_PANEL_CACHEABLE_IDS,
    createCanvasPanelCacheableIds,
    CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT,
    DEFAULT_CANVAS_PANEL_CACHE_ENTRY_LIMITS,
    CANVAS_PANEL_CACHE_ENTRY_LIMITS,
    createCanvasPanelCacheEntryLimits,
    DEFAULT_CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS,
    CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS,
    createCanvasOverlayPanelCacheBypassIds,
    INVENTORY_TILE_CACHE_WARM_BUDGET,
    CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT,
    isCanvasPanelCacheable,
    getCanvasPanelCacheEntryLimit,
    getCanvasPanelLayerCacheStore,
    trimCanvasPanelLayerCacheStore,
    getDeferredCanvasPanelBodyCacheRequest,
    getSkillsContentCooldownCacheKey,
    getSkillsContentCacheKey,
    isInventoryCanvasTileLayerCache,
    getCanvasTileLayerWarmBudget,
    getCanvasTileLayerCacheDecision,
    getCanvasTileLayerCacheKey,
    getCanvasTileLayerCacheStore,
    trimCanvasTileLayerCacheStore,
    getPanelPrewarmTarget,
    getCanvasPanelCacheKey,
    getCanvasPanelCacheStateKey,
    getDailyPanelCacheStateKey,
    getPlinkoPanelCacheStateKey,
    hasBlockingPanelPrewarmUi,
    canPrewarmPanelCaches,
    getPanelPrewarmBatchKey,
    getPanelPrewarmMetrics,
    getInventoryMarkupPrewarmBatchKey,
    createPanelCacheUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.panelCache = Object.assign({}, modules.panelCache || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
