(function initProjectStarfallUiCanvasWindows(global) {
  'use strict';

  const WINDOW_TITLES = Object.freeze({
    character: 'Character',
    equipment: 'Equipment',
    partyPanel: 'Party',
    pet: 'Pet',
    worldmap: 'World Map',
    monsters: 'Monster Guide',
    skills: 'Skill Tree',
    quests: 'Quests & Trials',
    inventory: 'Inventory',
    storage: 'Storage',
    shop: 'Shop',
    upgrade: 'Upgrade Station',
    plinko: 'Starfall Plinko',
    daily: 'Daily Rewards',
    cashShop: 'Cash Shop',
    beta: 'Beta Systems',
    guide: 'Guide',
    log: 'Session Log',
    keybinds: 'Keybinds',
    settings: 'Settings',
    admin: 'Admin Settings',
    worldwright: 'Worldwright Console',
    assetPreview: 'Asset Preview'
  });

  const WINDOW_DEFAULTS = Object.freeze({
    admin: Object.freeze({ x: 184, y: 36, w: 520, h: 630 }),
    worldwright: Object.freeze({ x: 124, y: 44, w: 720, h: 500 }),
    assetPreview: Object.freeze({ x: 88, y: 32, w: 760, h: 560 }),
    settings: Object.freeze({ x: 192, y: 44, w: 520, h: 490 }),
    keybinds: Object.freeze({ x: 142, y: 24, w: 760, h: 500 }),
    storage: Object.freeze({ x: 112, y: 44, w: 700, h: 620 }),
    pet: Object.freeze({ x: 158, y: 62, w: 380, h: 380 }),
    cashShop: Object.freeze({ x: 372, y: 30, w: 520, h: 620 }),
    daily: Object.freeze({ x: 324, y: 30, w: 620, h: 620 }),
    beta: Object.freeze({ x: 418, y: 36, w: 480, h: 600 }),
    guide: Object.freeze({ x: 136, y: 34, w: 600, h: 640 }),
    partyPanel: Object.freeze({ x: 156, y: 58, w: 500, h: 580 }),
    equipment: Object.freeze({ x: 128, y: 54, w: 456, h: 460 }),
    shop: Object.freeze({ x: 96, y: 54, w: 760, h: 430 }),
    upgrade: Object.freeze({ x: 96, y: 54, w: 430, h: 430 }),
    plinko: Object.freeze({ x: 64, y: 30, w: 890, h: 560 }),
    worldmap: Object.freeze({ x: 50, y: 22, w: 980, h: 640 }),
    monsters: Object.freeze({ x: 112, y: 38, w: 520, h: 560 }),
    skills: Object.freeze({ x: 96, y: 46, w: 520, h: 470 }),
    quests: Object.freeze({ x: 132, y: 34, w: 430, h: 530 }),
    character: Object.freeze({ x: 122, y: 48, w: 560, h: 500 })
  });

  const FALLBACK_WINDOW_DEFAULTS = Object.freeze({ x: 146, y: 62, w: 660, h: 410 });
  const PROMPT_WINDOW_DRAG_ENTRIES = Object.freeze([
    Object.freeze({ dragKey: 'upgradePromptDrag', stateName: 'upgradePromptWindow', boxType: 'upgradePrompt', activePanel: 'upgradePrompt' }),
    Object.freeze({ dragKey: 'potentialPromptDrag', stateName: 'potentialPromptWindow', boxType: 'potentialPrompt', activePanel: 'potentialPrompt' }),
    Object.freeze({ dragKey: 'shardCraftPromptDrag', stateName: 'shardCraftPromptWindow', boxType: 'shardCraftPrompt', activePanel: 'shardCraftPrompt' }),
    Object.freeze({ dragKey: 'questPromptDrag', stateName: 'questPromptWindow', boxType: 'questPrompt', activePanel: 'questPrompt' }),
    Object.freeze({ dragKey: 'dropQuantityPromptDrag', stateName: 'dropQuantityPromptWindow', boxType: 'dropQuantityPrompt', activePanel: 'dropQuantityPrompt' }),
    Object.freeze({ dragKey: 'adminNumberPromptDrag', stateName: 'adminNumberPromptWindow', boxType: 'adminNumberPrompt', activePanel: 'adminNumberPrompt' }),
    Object.freeze({ dragKey: 'confirmPromptDrag', stateName: 'confirmPromptWindow', boxType: 'confirmPrompt', activePanel: 'confirmPrompt' })
  ]);
  const PROMPT_WINDOW_DRAG_KEYS = Object.freeze(PROMPT_WINDOW_DRAG_ENTRIES.map((entry) => entry.dragKey));

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function getWindowTitle(panelId) {
    return WINDOW_TITLES[panelId] || 'Project Starfall';
  }

  function getWindowDefaults(panelId, options) {
    const settings = options || {};
    if (panelId === 'inventory') {
      return { x: 344, y: 18, w: 300, h: 188 + Math.max(0, Number(settings.inventoryGridHeight || 0)) };
    }
    return Object.assign({}, WINDOW_DEFAULTS[panelId] || FALLBACK_WINDOW_DEFAULTS);
  }

  function getEnsuredWindowState(existingWindow, defaults, currentWindowZ) {
    if (existingWindow) {
      return {
        win: existingWindow,
        windowZ: currentWindowZ,
        created: false
      };
    }
    const windowZ = Number(currentWindowZ || 0) + 1;
    return {
      win: Object.assign({}, defaults || {}, { scroll: 0, z: windowZ }),
      windowZ,
      created: true
    };
  }

  function getCanvasWindowDrawEntries(panelIds, options) {
    const settings = options || {};
    const getWindow = typeof settings.getWindow === 'function' ? settings.getWindow : () => null;
    return (Array.isArray(panelIds) ? panelIds : [])
      .map((id) => ({ id, win: getWindow(id) }))
      .sort((a, b) => Number(a.win && a.win.z || 0) - Number(b.win && b.win.z || 0));
  }

  function getTopWindowId(panelIds, windowState) {
    const state = windowState || {};
    const top = (Array.isArray(panelIds) ? panelIds : [])
      .map((id) => ({ id, z: (state[id] || {}).z || 0 }))
      .sort((a, b) => a.z - b.z)
      .pop();
    return top ? top.id : '';
  }

  function getClosedWindowState(panelIds, closeId) {
    const id = String(closeId || '');
    const openWindows = (Array.isArray(panelIds) ? panelIds : []).filter((panelId) => panelId !== id);
    return {
      openWindows,
      isModalOpen: openWindows.length > 0,
      activePanel: openWindows.length ? openWindows[openWindows.length - 1] : id
    };
  }

  function getRaisedWindowState(panelIds, panelId, currentWindowZ) {
    const id = String(panelId || '');
    const windowZ = Number(currentWindowZ || 0) + 1;
    return {
      windowZ,
      openWindows: (Array.isArray(panelIds) ? panelIds : []).filter((openId) => openId !== id).concat(id),
      activePanel: id
    };
  }

  function getCanvasWindowRegionAction(region) {
    const source = region || {};
    if (source.type === 'close-window') return { handled: true, type: 'closePanel', panelId: source.panelId };
    return { handled: false, type: '' };
  }

  function getCanvasWindowPointerAction(region) {
    const source = region || {};
    if (source.type === 'window-header') return { handled: true, type: 'startWindowDrag', panelId: source.panelId };
    return { handled: false, type: '' };
  }

  function getCanvasWindowRaiseAction(region) {
    const source = region || {};
    if (!source.panelId || String(source.type || '').startsWith('menu-')) {
      return { handled: false, type: '', panelId: '' };
    }
    return { handled: true, type: 'raiseWindow', panelId: source.panelId };
  }

  function getCanvasWindowDragUpdate(point, drag, win, bounds) {
    const sourcePoint = point || {};
    const sourceDrag = drag || {};
    const sourceWindow = win || {};
    const settings = bounds || {};
    const width = Number(settings.width || 0);
    const bottomLimit = Number(settings.bottomLimit || settings.height || 0);
    const windowW = Number(sourceWindow.w || 0);
    const windowH = Number(sourceWindow.h || 0);
    return {
      x: clampValue(Number(sourcePoint.x || 0) - Number(sourceDrag.dx || 0), 8, Math.max(8, width - windowW - 8)),
      y: clampValue(Number(sourcePoint.y || 0) - Number(sourceDrag.dy || 0), 8, Math.max(8, bottomLimit - windowH))
    };
  }

  function getCanvasWindowReleaseAction(drag) {
    if (!drag) {
      return {
        handled: false,
        shouldClearDrag: false,
        shouldClearActivePanel: false,
        shouldDraw: false
      };
    }
    return {
      handled: true,
      shouldClearDrag: true,
      shouldClearActivePanel: true,
      shouldDraw: true
    };
  }

  function getCenteredPromptBox(width, height, state, options) {
    const settings = options || {};
    const horizontalInset = Object.prototype.hasOwnProperty.call(settings, 'horizontalInset')
      ? Number(settings.horizontalInset)
      : 32;
    const minWidth = Number(settings.minWidth || 286);
    const maxWidth = Number(settings.maxWidth || 360);
    const boxW = Math.min(maxWidth, Math.max(minWidth, width - horizontalInset));
    const boxH = Number(settings.height || 0);
    const bottomLimit = Number(settings.bottomLimit || height);
    const defaultYMin = Object.prototype.hasOwnProperty.call(settings, 'defaultYMin')
      ? Number(settings.defaultYMin)
      : 16;
    const defaultBottomInset = Object.prototype.hasOwnProperty.call(settings, 'defaultBottomInset')
      ? Number(settings.defaultBottomInset)
      : 8;
    const defaultYOffset = Object.prototype.hasOwnProperty.call(settings, 'defaultYOffset')
      ? Number(settings.defaultYOffset)
      : 0;
    const shouldConstrainDefaultY = settings.constrainDefaultY !== false;
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, userPlaced: false };
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      const centeredY = (bottomLimit - boxH) / 2 + defaultYOffset;
      target.y = Math.round(Math.max(defaultYMin, shouldConstrainDefaultY
        ? Math.min(bottomLimit - boxH - defaultBottomInset, centeredY)
        : centeredY));
    }
    target.x = clampValue(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clampValue(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getPromptWindowDragUpdate(point, drag, box, bounds) {
    return getCanvasWindowDragUpdate(point, drag, box, bounds);
  }

  function getPromptWindowMoveAction(state) {
    const source = state || {};
    const entry = PROMPT_WINDOW_DRAG_ENTRIES.find((item) => !!source[item.dragKey]) || null;
    if (!entry) {
      return {
        handled: false,
        dragKey: '',
        drag: null,
        stateName: '',
        boxType: '',
        activePanel: '',
        shouldPreventDefault: false
      };
    }
    return {
      handled: true,
      dragKey: entry.dragKey,
      drag: source[entry.dragKey],
      stateName: entry.stateName,
      boxType: entry.boxType,
      activePanel: entry.activePanel,
      shouldPreventDefault: true
    };
  }

  function getPromptWindowReleaseAction(state) {
    const source = state || {};
    const dragKey = PROMPT_WINDOW_DRAG_KEYS.find((key) => !!source[key]) || '';
    if (!dragKey) return { handled: false, dragKey: '', shouldClearActivePanel: false, shouldDraw: false };
    return { handled: true, dragKey, shouldClearActivePanel: true, shouldDraw: true };
  }

  function getPromptWindowCancelAction() {
    return {
      handled: true,
      dragKeys: PROMPT_WINDOW_DRAG_KEYS.slice(),
      shouldDraw: true
    };
  }

  function createCanvasWindowUiHelpers() {
    return Object.freeze({
      getWindowTitle,
      getWindowDefaults,
      getEnsuredWindowState,
      getCanvasWindowDrawEntries,
      getTopWindowId,
      getClosedWindowState,
      getRaisedWindowState,
      getCanvasWindowRegionAction,
      getCanvasWindowPointerAction,
      getCanvasWindowRaiseAction,
      getCanvasWindowDragUpdate,
      getCanvasWindowReleaseAction,
      getCenteredPromptBox,
      getPromptWindowDragUpdate,
      getPromptWindowMoveAction,
      getPromptWindowReleaseAction,
      getPromptWindowCancelAction
    });
  }

  const api = {
    WINDOW_TITLES,
    WINDOW_DEFAULTS,
    FALLBACK_WINDOW_DEFAULTS,
    createCanvasWindowUiHelpers,
    getWindowTitle,
    getWindowDefaults,
    getEnsuredWindowState,
    getCanvasWindowDrawEntries,
    getTopWindowId,
    getClosedWindowState,
    getRaisedWindowState,
    getCanvasWindowRegionAction,
    getCanvasWindowPointerAction,
    getCanvasWindowRaiseAction,
    getCanvasWindowDragUpdate,
    getCanvasWindowReleaseAction,
    getCenteredPromptBox,
    getPromptWindowDragUpdate,
    getPromptWindowMoveAction,
    getPromptWindowReleaseAction,
    getPromptWindowCancelAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.canvasWindows = Object.assign({}, modules.canvasWindows || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
