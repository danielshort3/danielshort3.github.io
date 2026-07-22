(function initProjectStarfallUiPanels(global) {
  'use strict';

  const UiModules = global.ProjectStarfallUiModules || {};
  const UiCanvasWindows = (typeof require === 'function' ? require('./canvas-windows.js') : null) || UiModules.canvasWindows || {};
  const UiPanelCache = (typeof require === 'function' ? require('./panel-cache.js') : null) || UiModules.panelCache || {};
  const PANEL_IDS = Object.freeze([
    'character', 'equipment', 'partyPanel', 'pet', 'worldmap', 'monsters',
    'inventory', 'storage', 'skills', 'quests', 'shop', 'upgrade', 'plinko',
    'daily', 'cashShop', 'beta', 'guide', 'log', 'keybinds', 'settings',
    'admin', 'worldwright', 'assetPreview'
  ]);
  const HUD_QUICK_PANEL_IDS = Object.freeze(['inventory', 'equipment', 'skills', 'worldmap']);
  const PANEL_CACHE_PREWARM_ORDER = UiPanelCache.PANEL_CACHE_PREWARM_ORDER;
  const CANVAS_PANEL_CACHEABLE_IDS = UiPanelCache.CANVAS_PANEL_CACHEABLE_IDS;
  const CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT = UiPanelCache.CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT;
  const CANVAS_PANEL_CACHE_ENTRY_LIMITS = UiPanelCache.CANVAS_PANEL_CACHE_ENTRY_LIMITS;
  const CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS = UiPanelCache.CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS;
  const INVENTORY_TILE_CACHE_WARM_BUDGET = UiPanelCache.INVENTORY_TILE_CACHE_WARM_BUDGET;
  const CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT = UiPanelCache.CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT;
  const HUD_REFRESH_DOMAINS = Object.freeze(['hud', 'equipment', 'cards', 'skills', 'party', 'pet', 'settings']);
  const UI_CHANGE_HUD_REFRESH_DOMAINS = Object.freeze(['hud', 'equipment', 'cards', 'skills', 'party', 'pet']);
  const UI_CHANGE_LAZY_REFRESH_DOMAINS = new Set(['world', 'quests', 'guide', 'monsterGuide', 'debug', 'inventory', 'equipment', 'cards', 'shop', 'party', 'pet', 'daily', 'season']);
  const DOM_ATTACK_ACTION_BUTTON_ATTRIBUTES = Object.freeze([
    'data-starfall-action'
  ]);
  const DOM_ATTACK_ACTION_BUTTON_SELECTOR = '[data-starfall-action="attack"]';
  const DOM_ROOT_ELEMENT_SELECTORS = Object.freeze({
    stage: '.project-starfall-canvas-wrap',
    loader: '[data-starfall-loader]',
    loaderBar: '[data-starfall-loader-bar]',
    loaderPercent: '[data-starfall-loader-percent]',
    loaderStatus: '[data-starfall-loader-status]',
    startScreen: '[data-starfall-start-screen]',
    classSelect: '[data-starfall-class-select]',
    touchControls: '[data-starfall-touch-controls]',
    hud: '[data-starfall-hud]',
    commandMenu: '[data-starfall-command-menu]',
    commandToggle: '[data-starfall-command-toggle]',
    toast: '[data-starfall-toast]',
    station: '[data-starfall-station]',
    modalHeaderActions: '[data-starfall-modal-header-actions]',
    canvas: '#project-starfall-canvas'
  });
  const DOM_PANEL_RENDER_METHODS = Object.freeze({
    skills: 'renderSkillsPanel',
    equipment: 'renderEquipmentPanel',
    partyPanel: 'renderPartyPanel',
    pet: 'renderPetPanel',
    worldmap: 'renderWorldMapPanel',
    inventory: 'renderInventoryPanel',
    storage: 'renderStoragePanel',
    shop: 'renderShopPanel',
    upgrade: 'renderUpgradePanel',
    plinko: 'renderPlinkoPanel',
    daily: 'renderDailyLoginPanel',
    cashShop: 'renderCashShopPanel',
    beta: 'renderFractureOpsPanel',
    guide: 'renderGuidePanel',
    log: 'renderLogPanel',
    keybinds: 'renderKeybindsPanel',
    settings: 'renderSettingsPanel',
    admin: 'renderAdminPanel',
    worldwright: 'renderAdminConsole',
    assetPreview: 'renderAssetPreviewPanel'
  });
  const DOM_PANEL_MODAL_CLASS_TOGGLES = Object.freeze([
    Object.freeze({ className: 'is-inventory-modal', panelId: 'inventory' }),
    Object.freeze({ className: 'is-storage-modal', panelId: 'storage' })
  ]);
  const DOM_PANEL_BODY_CLASS_TOGGLES = Object.freeze([
    Object.freeze({ className: 'is-inventory-panel', panelId: 'inventory' }),
    Object.freeze({ className: 'is-storage-panel', panelId: 'storage' }),
    Object.freeze({ className: 'is-skills-panel', panelId: 'skills' }),
    Object.freeze({ className: 'is-plinko-panel', panelId: 'plinko' })
  ]);

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizePanelId(panelId, fallback) {
    const id = String(panelId || '').trim();
    if (PANEL_IDS.includes(id)) return id;
    return fallback && PANEL_IDS.includes(fallback) ? fallback : 'character';
  }

  function getRootElements(root, selectors) {
    const source = root || null;
    const selectorMap = selectors || DOM_ROOT_ELEMENT_SELECTORS;
    return Object.keys(DOM_ROOT_ELEMENT_SELECTORS).reduce((elements, key) => {
      const selector = selectorMap[key] || DOM_ROOT_ELEMENT_SELECTORS[key];
      elements[key] = source && typeof source.querySelector === 'function'
        ? source.querySelector(selector)
        : null;
      return elements;
    }, {});
  }

  const getWindowTitle = UiCanvasWindows.getWindowTitle;
  const getWindowDefaults = UiCanvasWindows.getWindowDefaults;

  function getPanelRefreshDomains(panelId, fallbackPanelId) {
    const id = String(panelId || fallbackPanelId || '');
    if (id === 'inventory') return ['session', 'inventory', 'equipment', 'cards'];
    if (id === 'storage') return ['session', 'inventory', 'equipment', 'cards'];
    if (id === 'equipment') return ['session', 'equipment', 'cards', 'inventory'];
    if (id === 'character') return ['hud', 'session', 'equipment', 'cards', 'skills'];
    if (id === 'skills') return ['session', 'skills', 'hud'];
    if (id === 'quests') return ['session', 'quests'];
    if (id === 'worldmap') return ['session', 'world', 'quests', 'season'];
    if (id === 'monsters') return ['session', 'monsterGuide', 'guide'];
    if (id === 'partyPanel') return ['session', 'party'];
    if (id === 'pet') return ['session', 'pet', 'inventory'];
    if (id === 'daily') return ['session', 'daily', 'inventory', 'cards', 'shop', 'hud'];
    if (id === 'beta') return ['session', 'season', 'world', 'quests', 'shop', 'inventory', 'hud'];
    if (id === 'shop' || id === 'cashShop' || id === 'plinko') return ['session', 'shop', 'inventory', 'hud'];
    if (id === 'guide') return ['session', 'guide', 'skills', 'world', 'quests'];
    if (id === 'keybinds' || id === 'settings' || id === 'admin') return ['session', 'settings', 'debug'];
    return ['session', 'hud'];
  }

  function getDomPanelRenderMethod(panelId) {
    return DOM_PANEL_RENDER_METHODS[String(panelId || '')] || 'renderCharacterPanel';
  }

  function getDomPanelPresentation(panelId) {
    const id = String(panelId || '');
    return {
      title: typeof getWindowTitle === 'function' ? getWindowTitle(id) : 'Project Starfall',
      kicker: 'Project Starfall',
      headerActions: id === 'inventory' ? 'inventorySort' : '',
      modalClassToggles: DOM_PANEL_MODAL_CLASS_TOGGLES.map((entry) => ({
        className: entry.className,
        active: id === entry.panelId
      })),
      bodyClassToggles: DOM_PANEL_BODY_CLASS_TOGGLES.map((entry) => ({
        className: entry.className,
        active: id === entry.panelId
      }))
    };
  }

  function getDomPanelRenderCacheKey(panelId, options) {
    const id = String(panelId || '');
    if (id !== 'inventory') return '';
    const settings = options || {};
    const potentialPromptResult = settings.potentialPromptResult || null;
    return [
      id,
      typeof settings.getCanvasPanelCacheStateKey === 'function' ? settings.getCanvasPanelCacheStateKey(id) : settings.canvasPanelCacheStateKey || '',
      typeof settings.getOverlayModalSnapshotKey === 'function' ? settings.getOverlayModalSnapshotKey() : settings.overlayModalSnapshotKey || '',
      potentialPromptResult && typeof settings.getPotentialPromptChoiceCacheKey === 'function' ? settings.getPotentialPromptChoiceCacheKey(potentialPromptResult) : '',
      settings.potentialAutoPanelOpen ? 1 : 0,
      JSON.stringify(settings.potentialAutoTarget || {}),
      settings.shardCraftPromptOpen ? `${settings.shardCraftRecipe || ''}:${settings.shardCraftQuantity || 1}` : ''
    ].join('|');
  }

  function shouldSkipDomPanelRender(cache, panelId, body, key) {
    return !!(key && cache && cache.panelId === panelId && cache.body === body && cache.key === key);
  }

  function getRememberedDomPanelRenderCache(panelId, body, key) {
    return key ? { panelId, body, key } : null;
  }

  function shouldUpdateModalHeaderActions(lastElement, lastHtml, element, html) {
    if (!element) return false;
    const markup = String(html || '');
    return !(lastElement === element && lastHtml === markup);
  }

  function getRememberedModalHeaderActionsState(element, html) {
    return {
      element,
      html: String(html || '')
    };
  }

  function shouldRefreshHudForDomains(domains) {
    const set = new Set(domains || []);
    return HUD_REFRESH_DOMAINS.some((domain) => set.has(domain));
  }

  function shouldRefreshPanelForDomains(domains, panelId, options) {
    const settings = options || {};
    if (!settings.isModalOpen) return false;
    const getDomains = typeof settings.getPanelRefreshDomains === 'function'
      ? settings.getPanelRefreshDomains
      : (id) => getPanelRefreshDomains(id, settings.fallbackPanelId);
    const set = new Set(domains || []);
    return getDomains(panelId).some((domain) => set.has(domain));
  }

  function getUiChangeDerivedRefreshDomains(domains, options) {
    const settings = options || {};
    const source = Array.isArray(domains) ? domains : [];
    const getDomains = typeof settings.getPanelRefreshDomains === 'function'
      ? settings.getPanelRefreshDomains
      : (id) => getPanelRefreshDomains(id, settings.activePanel);
    const visiblePanelDomains = settings.isModalOpen ? new Set(getDomains(settings.activePanel)) : new Set();
    const refreshDomains = source.filter((domain) => !UI_CHANGE_LAZY_REFRESH_DOMAINS.has(domain) || visiblePanelDomains.has(domain));
    const hasHudRefresh = refreshDomains.some((domain) => UI_CHANGE_HUD_REFRESH_DOMAINS.includes(domain));
    if (shouldRefreshHudForDomains(source) && !hasHudRefresh) refreshDomains.unshift('hud');
    return Array.from(new Set(refreshDomains));
  }

  function getQueuedUiRefreshState(currentRefresh, options) {
    const settings = options || {};
    const domains = Array.isArray(settings.domains) ? settings.domains : settings.domains ? [settings.domains] : [];
    const current = currentRefresh || {
      domains: new Set(),
      hud: false,
      panel: false,
      command: false,
      draw: false,
      focus: null
    };
    domains.forEach((domain) => current.domains.add(String(domain || '')));
    current.hud = current.hud || !!settings.hud;
    current.panel = current.panel || !!settings.panel;
    current.command = current.command || !!settings.command;
    current.draw = current.draw || settings.draw !== false;
    if (settings.focus) current.focus = settings.focus;
    return current;
  }

  function getUiChangeRefreshRequest(domains, patchRefresh, options) {
    const settings = options || {};
    const sourceDomains = Array.isArray(domains) ? domains : [];
    const patches = patchRefresh || {};
    const hud = patches.hud || shouldRefreshHudForDomains(sourceDomains);
    const panel = patches.panel || shouldRefreshPanelForDomains(sourceDomains, settings.activePanel, {
      isModalOpen: settings.isModalOpen,
      getPanelRefreshDomains: settings.getPanelRefreshDomains,
      fallbackPanelId: settings.activePanel
    });
    return {
      domains: sourceDomains,
      hud,
      panel,
      command: sourceDomains.includes('session'),
      draw: true
    };
  }

  function getPanelDerivedSnapshotUpdate(engine, domains) {
    const source = engine || {};
    const set = new Set(domains || []);
    const update = {};
    if (set.has('cards') && typeof source.getCardSnapshot === 'function') {
      update.cards = source.getCardSnapshot();
    }
    if (set.has('skills') && typeof source.getSkillsSnapshot === 'function') {
      update.skills = source.getSkillsSnapshot();
    }
    if (set.has('shop')) {
      if (typeof source.getPlinkoSnapshot === 'function') update.plinko = source.getPlinkoSnapshot();
      if (typeof source.getCashShopSnapshot === 'function') update.cashShop = source.getCashShopSnapshot();
    }
    if (set.has('daily') && typeof source.getDailyLoginSnapshot === 'function') {
      update.dailyLogin = source.getDailyLoginSnapshot();
    }
    if ((set.has('season') || set.has('shop')) && typeof source.getSeasonSnapshot === 'function') {
      update.season = source.getSeasonSnapshot();
    }
    if (set.has('party') && typeof source.getPartySnapshot === 'function') {
      update.party = source.getPartySnapshot();
    }
    if (set.has('pet') && typeof source.getPetSnapshot === 'function') {
      update.pet = source.getPetSnapshot();
    }
    return update;
  }

  const getEnsuredWindowState = UiCanvasWindows.getEnsuredWindowState;
  const getCanvasWindowDrawEntries = UiCanvasWindows.getCanvasWindowDrawEntries;
  const getTopWindowId = UiCanvasWindows.getTopWindowId;
  const getClosedWindowState = UiCanvasWindows.getClosedWindowState;
  const getRaisedWindowState = UiCanvasWindows.getRaisedWindowState;

  function getOpenPanelAction(panelId, options) {
    const settings = options || {};
    const nextPanel = normalizePanelId(panelId, settings.fallback || 'character');
    return {
      action: nextPanel === 'upgrade' ? 'openUpgradePrompt' : 'openPanel',
      panelId: nextPanel
    };
  }

  function getTogglePanelAction(panelId, options) {
    const settings = options || {};
    const nextPanel = normalizePanelId(panelId, settings.fallback || 'character');
    if (nextPanel === 'upgrade' && settings.upgradePromptOpen) {
      return { action: 'closeUpgradePrompt', panelId: nextPanel, opened: false };
    }
    if ((Array.isArray(settings.openWindows) ? settings.openWindows : []).includes(nextPanel)) {
      return { action: 'closePanel', panelId: nextPanel, opened: false };
    }
    return { action: 'openPanel', panelId: nextPanel, opened: true };
  }

  function getCommandPanelOpenState(forceOpen, isCommandOpen) {
    return typeof forceOpen === 'boolean' ? forceOpen : !isCommandOpen;
  }

  function getCommandMenuCanvasPointerAction(isCommandOpen, isCommandRegion, point) {
    if (!isCommandOpen || isCommandRegion) {
      return {
        handled: false,
        type: '',
        canvasDownRegion: null,
        shouldPreventDefault: false
      };
    }
    const sourcePoint = point || {};
    return {
      handled: true,
      type: 'dismissCommandMenu',
      canvasDownRegion: {
        type: 'menu-dismiss',
        x: Number(sourcePoint.x) - 1,
        y: Number(sourcePoint.y) - 1,
        w: 2,
        h: 2
      },
      shouldPreventDefault: true
    };
  }

  function getCommandMenuRegionAction(region) {
    const source = region || {};
    if (source.type === 'menu-panel') {
      return { handled: true, type: 'togglePanel', panelId: source.panelId };
    }
    if (source.type === 'menu-action') {
      const actionId = source.action;
      if (actionId === 'menu') return { handled: true, type: 'handleMenuAction', actionId };
      if (actionId === 'changeChannel') {
        return { handled: true, type: 'changeChannel', channelId: source.channelId };
      }
      return { handled: true, type: 'handleAction', actionId };
    }
    return { handled: false, type: '' };
  }

  function getPanelShellDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (hasAttribute('data-starfall-command-toggle')) return { handled: true, type: 'toggleCommand' };
    if (hasAttribute('data-starfall-close')) return { handled: true, type: 'closePanel' };
    const channelId = getAttribute('data-starfall-command-channel');
    if (channelId) return { handled: true, type: 'changeChannel', channelId };
    const panelId = getAttribute('data-starfall-open-panel');
    if (panelId) return { handled: true, type: 'togglePanel', panelId };
    return { handled: false, type: '' };
  }

  function getDailyPanelDomAction(target) {
    const source = target || null;
    if (source && typeof source.hasAttribute === 'function' && source.hasAttribute('data-starfall-daily-login-claim')) {
      return { handled: true, type: 'claimDailyLoginReward' };
    }
    return { handled: false, type: '' };
  }

  function getDailyPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'daily-login-claim') return { handled: true, type: 'claimDailyLoginReward' };
    return { handled: false, type: '' };
  }

  function getFractureOpsPanelDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const directiveId = getAttribute('data-starfall-directive-select');
    if (directiveId) return { handled: true, type: 'selectSeasonDirective', directiveId };
    if (hasAttribute('data-starfall-season-claim')) return { handled: true, type: 'claimSeasonReward' };
    return { handled: false, type: '' };
  }

  function getFractureOpsPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'directive-select' && source.directiveId) {
      return { handled: true, type: 'selectSeasonDirective', directiveId: source.directiveId };
    }
    if (source.type === 'season-claim') return { handled: true, type: 'claimSeasonReward' };
    return { handled: false, type: '' };
  }

  function getActionButtonDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const actionId = getAttribute('data-starfall-action');
    if (!actionId) return { handled: false, type: '' };
    if (actionId === 'attack') return { handled: true, type: 'focusOnly', actionId };
    return { handled: true, type: 'handleAction', actionId };
  }

  function getAttackActionButtonTarget(target, selector) {
    const source = target && typeof target.closest === 'function'
      ? target.closest(selector || DOM_ATTACK_ACTION_BUTTON_SELECTOR)
      : null;
    return source || null;
  }

  function getActionButtonPointerDomAction(target) {
    const source = getAttackActionButtonTarget(target);
    if (!source) return { handled: false, type: '', target: null };
    return {
      handled: true,
      type: 'pressAttack',
      actionId: 'attack',
      target: source,
      disabled: !!source.disabled
    };
  }

  const getCenteredPromptBox = UiCanvasWindows.getCenteredPromptBox;
  const isCanvasPanelCacheable = UiPanelCache.isCanvasPanelCacheable;
  const getCanvasPanelCacheEntryLimit = UiPanelCache.getCanvasPanelCacheEntryLimit;
  const getCanvasPanelLayerCacheStore = UiPanelCache.getCanvasPanelLayerCacheStore;
  const trimCanvasPanelLayerCacheStore = UiPanelCache.trimCanvasPanelLayerCacheStore;
  const getDeferredCanvasPanelBodyCacheRequest = UiPanelCache.getDeferredCanvasPanelBodyCacheRequest;
  const getSkillsContentCooldownCacheKey = UiPanelCache.getSkillsContentCooldownCacheKey;
  const getSkillsContentCacheKey = UiPanelCache.getSkillsContentCacheKey;
  const isInventoryCanvasTileLayerCache = UiPanelCache.isInventoryCanvasTileLayerCache;
  const getCanvasTileLayerWarmBudget = UiPanelCache.getCanvasTileLayerWarmBudget;
  const getCanvasTileLayerCacheDecision = UiPanelCache.getCanvasTileLayerCacheDecision;
  const getCanvasTileLayerCacheKey = UiPanelCache.getCanvasTileLayerCacheKey;
  const getCanvasTileLayerCacheStore = UiPanelCache.getCanvasTileLayerCacheStore;
  const trimCanvasTileLayerCacheStore = UiPanelCache.trimCanvasTileLayerCacheStore;
  const getPanelPrewarmTarget = UiPanelCache.getPanelPrewarmTarget;
  const getCanvasPanelCacheKey = UiPanelCache.getCanvasPanelCacheKey;
  const getCanvasPanelCacheStateKey = UiPanelCache.getCanvasPanelCacheStateKey;
  const getDailyPanelCacheStateKey = UiPanelCache.getDailyPanelCacheStateKey;
  const getPlinkoPanelCacheStateKey = UiPanelCache.getPlinkoPanelCacheStateKey;
  const hasBlockingPanelPrewarmUi = UiPanelCache.hasBlockingPanelPrewarmUi;
  const canPrewarmPanelCaches = UiPanelCache.canPrewarmPanelCaches;
  const getPanelPrewarmBatchKey = UiPanelCache.getPanelPrewarmBatchKey;
  const getPanelPrewarmMetrics = UiPanelCache.getPanelPrewarmMetrics;
  const getInventoryMarkupPrewarmBatchKey = UiPanelCache.getInventoryMarkupPrewarmBatchKey;

  function createPanelDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_ATTACK_ACTION_BUTTON_ATTRIBUTES,
      DOM_ATTACK_ACTION_BUTTON_SELECTOR,
      DOM_ROOT_ELEMENT_SELECTORS
    });
  }

  function createPanelRefreshUiHelpers() {
    return Object.freeze({
      getPanelRefreshDomains,
      shouldRefreshHudForDomains,
      shouldRefreshPanelForDomains,
      getUiChangeDerivedRefreshDomains,
      getQueuedUiRefreshState,
      getUiChangeRefreshRequest,
      getPanelDerivedSnapshotUpdate
    });
  }

  function createPanelDomRenderUiHelpers() {
    return Object.freeze({
      getRootElements,
      getDomPanelRenderMethod,
      getDomPanelPresentation,
      getDomPanelRenderCacheKey,
      shouldSkipDomPanelRender,
      getRememberedDomPanelRenderCache,
      shouldUpdateModalHeaderActions,
      getRememberedModalHeaderActionsState
    });
  }

  function createPanelInteractionUiHelpers() {
    return Object.freeze({
      normalizePanelId,
      getOpenPanelAction,
      getTogglePanelAction,
      getCommandPanelOpenState,
      getCommandMenuCanvasPointerAction,
      getCommandMenuRegionAction,
      getPanelShellDomAction,
      getDailyPanelDomAction,
      getDailyPanelRegionAction,
      getFractureOpsPanelDomAction,
      getFractureOpsPanelRegionAction,
      getActionButtonDomAction,
      getAttackActionButtonTarget,
      getActionButtonPointerDomAction
    });
  }

  const api = {
    DOM_ATTACK_ACTION_BUTTON_ATTRIBUTES,
    DOM_ATTACK_ACTION_BUTTON_SELECTOR,
    DOM_ROOT_ELEMENT_SELECTORS,
    PANEL_IDS,
    HUD_QUICK_PANEL_IDS,
    PANEL_CACHE_PREWARM_ORDER,
    CANVAS_PANEL_CACHEABLE_IDS,
    CANVAS_PANEL_CACHE_DEFAULT_ENTRY_LIMIT,
    CANVAS_PANEL_CACHE_ENTRY_LIMITS,
    CANVAS_OVERLAY_PANEL_CACHE_BYPASS_IDS,
    INVENTORY_TILE_CACHE_WARM_BUDGET,
    CANVAS_TILE_LAYER_CACHE_ENTRY_LIMIT,
    normalizePanelId,
    getRootElements,
    getWindowTitle,
    getWindowDefaults,
    getPanelRefreshDomains,
    getDomPanelRenderMethod,
    getDomPanelPresentation,
    getDomPanelRenderCacheKey,
    shouldSkipDomPanelRender,
    getRememberedDomPanelRenderCache,
    shouldUpdateModalHeaderActions,
    getRememberedModalHeaderActionsState,
    shouldRefreshHudForDomains,
    shouldRefreshPanelForDomains,
    getUiChangeDerivedRefreshDomains,
    getQueuedUiRefreshState,
    getUiChangeRefreshRequest,
    getPanelDerivedSnapshotUpdate,
    getEnsuredWindowState,
    getCanvasWindowDrawEntries,
    getTopWindowId,
    getClosedWindowState,
    getRaisedWindowState,
    getOpenPanelAction,
    getTogglePanelAction,
    getCommandPanelOpenState,
    getCommandMenuCanvasPointerAction,
    getCommandMenuRegionAction,
    getPanelShellDomAction,
    getDailyPanelDomAction,
    getDailyPanelRegionAction,
    getFractureOpsPanelDomAction,
    getFractureOpsPanelRegionAction,
    getActionButtonDomAction,
    getAttackActionButtonTarget,
    getActionButtonPointerDomAction,
    createPanelDomSelectorUiHelpers,
    createPanelRefreshUiHelpers,
    createPanelDomRenderUiHelpers,
    createPanelInteractionUiHelpers,
    getCenteredPromptBox,
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
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.panels = Object.assign({}, modules.panels || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
