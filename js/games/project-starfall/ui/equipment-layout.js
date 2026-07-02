(function initProjectStarfallUiEquipmentLayout(global) {
  'use strict';

  const DEFAULT_SLOT_META = Object.freeze({
    weapon: { label: 'Weapon', icon: 'WPN' },
    offhand: { label: 'Offhand', icon: 'OFF' },
    head: { label: 'Head', icon: 'HD' },
    chest: { label: 'Chest', icon: 'CH' },
    gloves: { label: 'Gloves', icon: 'GLV' },
    boots: { label: 'Boots', icon: 'BT' },
    ring: { label: 'Ring', icon: 'RG' },
    amulet: { label: 'Amulet', icon: 'AM' }
  });
  const EQUIPMENT_SLOT_DISPLAY_ORDER = Object.freeze(['weapon', 'offhand', 'head', 'chest', 'gloves', 'boots', 'ring', 'amulet']);
  const CHARACTER_PANEL_TABS = Object.freeze([
    { id: 'overview', label: 'Overview' },
    { id: 'stats', label: 'Stats' },
    { id: 'growth', label: 'Growth' },
    { id: 'class', label: 'Class' }
  ]);
  const EQUIPMENT_GRID_COLUMNS = 5;
  const EQUIPMENT_GRID_CELL_SIZE = 74;
  const EQUIPMENT_GRID_GAP = 8;
  const EQUIPMENT_GRID_LAYOUT = Object.freeze([
    Object.freeze([
      Object.freeze({ slot: 'ring' }),
      Object.freeze({ placeholder: 'insignia-left', icon: '.', label: 'Visual slot' }),
      Object.freeze({ slot: 'head' }),
      Object.freeze({ placeholder: 'insignia-right', icon: '.', label: 'Visual slot' }),
      Object.freeze({ slot: 'amulet' })
    ]),
    Object.freeze([
      Object.freeze({ placeholder: 'upper-left', icon: '', label: 'Inactive slot' }),
      Object.freeze({ slot: 'weapon' }),
      Object.freeze({ slot: 'chest' }),
      Object.freeze({ slot: 'offhand' }),
      Object.freeze({ placeholder: 'upper-right', icon: '', label: 'Inactive slot' })
    ]),
    Object.freeze([
      Object.freeze({ placeholder: 'mid-left', icon: '', label: 'Inactive slot' }),
      Object.freeze({ slot: 'gloves' }),
      Object.freeze({ placeholder: 'core', icon: '*', label: 'Visual slot' }),
      Object.freeze({ placeholder: 'mid-right', icon: '', label: 'Inactive slot' }),
      Object.freeze({ placeholder: 'lower-right', icon: '', label: 'Inactive slot' })
    ]),
    Object.freeze([
      Object.freeze({ placeholder: 'foot-left', icon: '', label: 'Inactive slot' }),
      Object.freeze({ placeholder: 'foot-mid-left', icon: '', label: 'Inactive slot' }),
      Object.freeze({ slot: 'boots' }),
      Object.freeze({ placeholder: 'foot-mid-right', icon: '', label: 'Inactive slot' }),
      Object.freeze({ placeholder: 'foot-right', icon: '', label: 'Inactive slot' })
    ])
  ]);

  function normalizeCharacterPanelTab(value) {
    const id = String(value || '').trim();
    return CHARACTER_PANEL_TABS.some((option) => option.id === id) ? id : 'overview';
  }

  function getCharacterPanelTabDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const tabId = getAttribute('data-starfall-character-tab');
    if (tabId) return { handled: true, type: 'setTab', tabId };
    return { handled: false, type: '' };
  }

  function getEquipmentPanelTabDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const tabId = getAttribute('data-starfall-equipment-tab');
    if (tabId) return { handled: true, type: 'setTab', tabId };
    return { handled: false, type: '' };
  }

  function getCharacterPanelTabRegionAction(region) {
    const source = region || {};
    if (source.type === 'character-tab') return { handled: true, type: 'setTab', tabId: source.tabId };
    return { handled: false, type: '' };
  }

  function getEquipmentPanelTabRegionAction(region) {
    const source = region || {};
    if (source.type === 'equipment-tab') return { handled: true, type: 'setTab', tabId: source.tabId };
    return { handled: false, type: '' };
  }

  function getSlotMeta(slot, options) {
    const settings = options || {};
    const data = settings.data || {};
    const formatStatName = settings.formatStatName || function formatStatNameFallback(key) {
      return String(key || '')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, (letter) => letter.toUpperCase());
    };
    const metadata = data.EQUIPMENT_SLOT_META || DEFAULT_SLOT_META;
    return metadata[slot] || { label: formatStatName(slot), icon: String(slot || 'item').slice(0, 3).toUpperCase() };
  }

  function getCanvasPanelEquipmentKey(equipment) {
    return Object.keys(equipment || {})
      .sort()
      .map((slot) => `${slot}:${equipment[slot] && equipment[slot].uid || ''}`)
      .join(',');
  }

  function createEquipmentLayoutUiHelpers(options) {
    const settings = options || {};
    return Object.freeze({
      normalizeCharacterPanelTab,
      getCharacterPanelTabDomAction,
      getEquipmentPanelTabDomAction,
      getCharacterPanelTabRegionAction,
      getEquipmentPanelTabRegionAction,
      getSlotMeta: (slot) => getSlotMeta(slot, settings),
      getCanvasPanelEquipmentKey
    });
  }

  const api = {
    DEFAULT_SLOT_META,
    EQUIPMENT_SLOT_DISPLAY_ORDER,
    CHARACTER_PANEL_TABS,
    EQUIPMENT_GRID_COLUMNS,
    EQUIPMENT_GRID_CELL_SIZE,
    EQUIPMENT_GRID_GAP,
    EQUIPMENT_GRID_LAYOUT,
    normalizeCharacterPanelTab,
    getCharacterPanelTabDomAction,
    getEquipmentPanelTabDomAction,
    getCharacterPanelTabRegionAction,
    getEquipmentPanelTabRegionAction,
    getSlotMeta,
    getCanvasPanelEquipmentKey,
    createEquipmentLayoutUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.equipmentLayout = Object.assign({}, modules.equipmentLayout || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
