(function initProjectStarfallUiInventoryConfig(global) {
  'use strict';

  const INVENTORY_DOUBLE_CLICK_MS = 360;
  const INVENTORY_DOUBLE_CLICK_DISTANCE = 8;
  const INVENTORY_COLUMNS = 4;
  const INVENTORY_ROWS = 6;
  const INVENTORY_SECTION_SIZE = 36;
  const INVENTORY_STORAGE_COLUMNS = 6;
  const INVENTORY_COUPON_MAX_SECTIONS = 8;
  const INVENTORY_MAX_SECTIONS = 8;
  const STORAGE_STACK_CAP_MULTIPLIER = 4;
  const DEFAULT_USABLE_STACK_CAP = 100;
  const DEFAULT_ETC_STACK_CAP = 250;
  const STACKABLE_SLOT_LIMITS = Object.freeze({
    usable: Object.freeze({
      admin_worldwright_console: 1,
      pet_whistle: 1,
      minor_health_potion: 100,
      minor_resource_tonic: 100,
      camp_ration: 100,
      guard_tonic: 50,
      swiftstep_oil: 50,
      magnet_charm: 50,
      town_return_scroll: 30,
      equipment_slot_coupon: 25,
      usable_slot_coupon: 25,
      etc_slot_coupon: 25,
      card_slot_coupon: 25,
      potential_cube: 25,
      preservation_cube: 25,
      line_catalyst: 25,
      plinko_ball_basic: 250,
      plinko_ball_polished: 150,
      plinko_ball_meteor: 75,
      base_skill_manual: 25,
      advanced_skill_manual: 25,
      skill_reset_scroll: 25,
      stat_reset_scroll: 25
    }),
    etc: Object.freeze({
      upgradeDust: 500,
      gelDrop: 250,
      oreChunks: 250,
      upgradeCatalyst: 100,
      wardingScroll: 100,
      cubeFragment: 100,
      refinementCore: 50
    })
  });

  const INVENTORY_CANVAS_VISIBLE_ROWS = 6;
  const INVENTORY_CANVAS_CELL_SIZE = 58;
  const INVENTORY_CANVAS_CELL_GAP = 5;
  const INVENTORY_CANVAS_GRID_HEIGHT = INVENTORY_CANVAS_VISIBLE_ROWS * INVENTORY_CANVAS_CELL_SIZE + (INVENTORY_CANVAS_VISIBLE_ROWS - 1) * INVENTORY_CANVAS_CELL_GAP;

  const EQUIPMENT_INVENTORY_SORT_OPTIONS = Object.freeze([
    { id: 'powerDesc', label: 'Power Impact', shortLabel: 'Power' },
    { id: 'powerAsc', label: 'Weakest First', shortLabel: 'Weak' },
    { id: 'type', label: 'Item Type', shortLabel: 'Type' },
    { id: 'newest', label: 'Newest', shortLabel: 'New' }
  ]);
  const STACKABLE_INVENTORY_SORT_OPTIONS = Object.freeze([
    { id: 'type', label: 'Item Type', shortLabel: 'Type' },
    { id: 'value', label: 'Value', shortLabel: 'Value' },
    { id: 'quantity', label: 'Quantity', shortLabel: 'Qty' },
    { id: 'name', label: 'Name', shortLabel: 'Name' }
  ]);
  const CARD_INVENTORY_SORT_OPTIONS = Object.freeze([
    { id: 'value', label: 'Rarity/Rank', shortLabel: 'Value' },
    { id: 'newest', label: 'Newest', shortLabel: 'New' },
    { id: 'name', label: 'Name', shortLabel: 'Name' }
  ]);

  function createInventorySortOptions(equipmentOptions, stackableOptions) {
    const equipment = Array.isArray(equipmentOptions) ? equipmentOptions : [];
    const stackable = Array.isArray(stackableOptions) ? stackableOptions : [];
    return Object.freeze(equipment.concat(
      stackable.filter((option) => !equipment.some((item) => item.id === option.id))
    ));
  }

  const INVENTORY_SORT_OPTIONS = createInventorySortOptions(EQUIPMENT_INVENTORY_SORT_OPTIONS, STACKABLE_INVENTORY_SORT_OPTIONS);
  const INVENTORY_TAB_OPTIONS = Object.freeze([
    { id: 'equipment', label: 'Equipment' },
    { id: 'usable', label: 'Usable' },
    { id: 'etc', label: 'Etc' },
    { id: 'cards', label: 'Cards' }
  ]);

  function createStorageTabOptions(tabOptions) {
    return Object.freeze((Array.isArray(tabOptions) ? tabOptions : []).filter((option) => option.id !== 'cards'));
  }

  const STORAGE_TAB_OPTIONS = createStorageTabOptions(INVENTORY_TAB_OPTIONS);
  const INVENTORY_SELL_RULE_TOOLTIPS = Object.freeze({
    autoSell: 'When enabled, matching newly looted equipment is sold immediately on pickup by you or your pet.',
    matchWeak: 'Matches unlocked equipment with power impact less than or equal to your currently equipped item in that slot.',
    matchNonClass: 'Matches unlocked equipment your current base or advanced class cannot use.',
    rarity: 'Matches unlocked equipment of this rarity for Bulk Sell and Auto Sell.',
    reset: 'Restore the default Sell Rules.',
    bulk: 'Sell unlocked inventory gear matching the enabled filters. Equipped and locked gear are never sold.'
  });

  function normalizeInventoryTab(value) {
    const id = String(value || '').trim();
    return INVENTORY_TAB_OPTIONS.some((option) => option.id === id) ? id : 'equipment';
  }

  function normalizeStorageTab(value) {
    const id = String(value || '').trim();
    return STORAGE_TAB_OPTIONS.some((option) => option.id === id) ? id : 'equipment';
  }

  function getInventorySortOptions(tabId) {
    const tab = normalizeInventoryTab(tabId);
    if (tab === 'equipment') return EQUIPMENT_INVENTORY_SORT_OPTIONS;
    if (tab === 'cards') return CARD_INVENTORY_SORT_OPTIONS;
    return STACKABLE_INVENTORY_SORT_OPTIONS;
  }

  function getInventorySortLabel(sortId, tabId) {
    const options = getInventorySortOptions(tabId);
    const option = options.find((item) => item.id === sortId) ||
      INVENTORY_SORT_OPTIONS.find((item) => item.id === sortId) ||
      options[0];
    return option.shortLabel || option.label;
  }

  function normalizeInventorySortChoice(sortId, fallback, tabId) {
    const options = getInventorySortOptions(tabId);
    const id = String(sortId || '').trim();
    if (options.some((item) => item.id === id)) return id;
    return fallback && options.some((item) => item.id === fallback) ? fallback : options[0].id;
  }

  function getStackCapForItem(tabId, itemId, options) {
    const settings = options || {};
    const tab = normalizeInventoryTab(tabId);
    const id = String(itemId || '').trim();
    const engine = settings.engine;
    if (engine && typeof engine.getStackCap === 'function') return engine.getStackCap(tab, id);
    if (tab === 'usable') {
      if (STACKABLE_SLOT_LIMITS.usable[id]) return STACKABLE_SLOT_LIMITS.usable[id];
      const getConsumableItem = settings.getConsumableItem || function getConsumableItemFallback() { return null; };
      const item = getConsumableItem(id);
      if (item && item.rateBuffType) return 25;
      if (item && (item.potentialCube || item.preservationCube || item.lineCatalyst || item.inventorySectionCoupon || item.skillPointPool || item.resetSkillPoints || item.resetStatUpgrades || item.returnMapId)) return 25;
      if (item && item.buffId) return 50;
      return DEFAULT_USABLE_STACK_CAP;
    }
    if (tab === 'etc') return STACKABLE_SLOT_LIMITS.etc[id] || DEFAULT_ETC_STACK_CAP;
    return 1;
  }

  function getStackSlotCountForCount(tabId, itemId, count, options) {
    const quantity = Math.max(0, Math.floor(Number(count) || 0));
    if (quantity <= 0) return 0;
    return Math.ceil(quantity / Math.max(1, getStackCapForItem(tabId, itemId, options)));
  }

  function getStorageStackCapForItem(tabId, itemId, options) {
    const tab = normalizeInventoryTab(tabId);
    if (tab === 'equipment') return 1;
    return Math.max(1, getStackCapForItem(tab, itemId, options) * STORAGE_STACK_CAP_MULTIPLIER);
  }

  function getStorageStackSlotCountForCount(tabId, itemId, count, options) {
    const quantity = Math.max(0, Math.floor(Number(count) || 0));
    if (quantity <= 0) return 0;
    return Math.ceil(quantity / Math.max(1, getStorageStackCapForItem(tabId, itemId, options)));
  }

  function createStackedSlotEntries(tabId, itemId, count, createEntry, options) {
    const entries = [];
    appendStackedSlotEntries(entries, tabId, itemId, count, createEntry, options);
    return entries;
  }

  function appendStackedSlotEntries(target, tabId, itemId, count, createEntry, options, maxEntries) {
    const tab = normalizeInventoryTab(tabId);
    const id = String(itemId || '').trim();
    const limit = Number.isFinite(Number(maxEntries)) ? Math.max(0, Math.floor(Number(maxEntries))) : Infinity;
    if (target.length >= limit) return target;
    const total = Math.max(0, Math.floor(Number(count) || 0));
    const cap = Math.max(1, getStackCapForItem(tab, id, options));
    const slots = total > 0 ? Math.ceil(total / cap) : 0;
    const visibleSlots = Math.min(slots, Math.max(0, limit - target.length));
    for (let index = 0; index < visibleSlots; index += 1) {
      const stackCount = Math.min(cap, Math.max(0, total - index * cap));
      const base = typeof createEntry === 'function' ? createEntry(id, stackCount, index, slots) : { id };
      target.push(Object.assign({}, base || { id }, {
        id: base && base.id ? base.id : id,
        stackKey: `${tab}:${id}:${index}`,
        stackIndex: index,
        stackSlots: slots,
        stackCount,
        count: stackCount,
        totalCount: total,
        maxStack: cap
      }));
    }
    return target;
  }

  function createStorageStackedSlotEntries(tabId, itemId, count, createEntry, options) {
    const entries = [];
    appendStorageStackedSlotEntries(entries, tabId, itemId, count, createEntry, options);
    return entries;
  }

  function appendStorageStackedSlotEntries(target, tabId, itemId, count, createEntry, options, maxEntries) {
    const tab = normalizeInventoryTab(tabId);
    const id = String(itemId || '').trim();
    const limit = Number.isFinite(Number(maxEntries)) ? Math.max(0, Math.floor(Number(maxEntries))) : Infinity;
    if (target.length >= limit) return target;
    const total = Math.max(0, Math.floor(Number(count) || 0));
    const cap = Math.max(1, getStorageStackCapForItem(tab, id, options));
    const slots = total > 0 ? Math.ceil(total / cap) : 0;
    const visibleSlots = Math.min(slots, Math.max(0, limit - target.length));
    for (let index = 0; index < visibleSlots; index += 1) {
      const stackCount = Math.min(cap, Math.max(0, total - index * cap));
      const base = typeof createEntry === 'function' ? createEntry(id, stackCount, index, slots) : { id };
      target.push(Object.assign({}, base || { id }, {
        id: base && base.id ? base.id : id,
        stackKey: `storage:${tab}:${id}:${index}`,
        stackIndex: index,
        stackSlots: slots,
        stackCount,
        count: stackCount,
        totalCount: total,
        maxStack: cap
      }));
    }
    return target;
  }

  function getInventorySectionCouponId(tabId) {
    const tab = normalizeInventoryTab(tabId);
    if (tab === 'cards') return 'card_slot_coupon';
    if (tab === 'usable') return 'usable_slot_coupon';
    if (tab === 'etc') return 'etc_slot_coupon';
    return 'equipment_slot_coupon';
  }

  function getInventoryTabLabel(tabId) {
    const tab = normalizeInventoryTab(tabId);
    const option = INVENTORY_TAB_OPTIONS.find((item) => item.id === tab);
    return option ? option.label : 'Equipment';
  }

  function normalizeInventorySectionCount(value) {
    const count = Math.floor(Number(value || 1) || 1);
    return Math.max(1, Math.min(INVENTORY_MAX_SECTIONS, count));
  }

  function getInventorySectionCount(snapshot, tabId) {
    const tab = normalizeInventoryTab(tabId);
    const sections = snapshot && snapshot.state && snapshot.state.inventorySections || {};
    const legacySlots = snapshot && snapshot.state && snapshot.state.inventorySlots || {};
    const fallback = tab === 'equipment' && legacySlots.equipment
      ? Math.ceil(Math.max(0, Number(legacySlots.equipment || 0)) / INVENTORY_SECTION_SIZE)
      : 1;
    return normalizeInventorySectionCount(sections[tab] || fallback);
  }

  function getInventoryUnlockedCapacity(snapshot, tabId) {
    return getInventorySectionCount(snapshot, tabId) * INVENTORY_SECTION_SIZE;
  }

  function getInventoryPreviewCapacity() {
    return INVENTORY_MAX_SECTIONS * INVENTORY_SECTION_SIZE;
  }

  function countPositiveStackEntries(source) {
    if (!source || typeof source !== 'object') return 0;
    let count = 0;
    const keys = Object.keys(source);
    for (let index = 0; index < keys.length; index += 1) {
      if (Math.max(0, Math.floor(Number(source[keys[index]]) || 0)) > 0) count += 1;
    }
    return count;
  }

  function hasEnumerableKeys(source) {
    if (!source || typeof source !== 'object') return false;
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) return true;
    }
    return false;
  }

  function getLastPopulatedSlotIndex(entries) {
    return (Array.isArray(entries) ? entries : []).reduce((lastIndex, entry, index) => entry ? index : lastIndex, -1);
  }

  function getInventoryGridVisibleSlotCount(entries, capacity) {
    const unlockedCapacity = Math.max(0, Math.floor(Number(capacity || 0) || 0));
    const previewCapacity = getInventoryPreviewCapacity();
    const populatedSlots = Math.min(previewCapacity, getLastPopulatedSlotIndex(entries) + 1);
    return Math.max(previewCapacity, unlockedCapacity, populatedSlots);
  }

  function getFirstAvailableSlotIndex(entries, capacity) {
    const source = Array.isArray(entries) ? entries : [];
    const unlockedCapacity = Math.max(0, Math.floor(Number(capacity || 0) || 0));
    for (let index = 0; index < unlockedCapacity; index += 1) {
      if (!source[index]) return index;
    }
    return Math.max(0, Math.min(Math.max(0, unlockedCapacity - 1), source.length));
  }

  function isInventoryPreviewSlotLocked(slotIndex, capacity) {
    const index = Math.max(0, Math.floor(Number(slotIndex || 0) || 0));
    const unlockedCapacity = Math.max(0, Math.floor(Number(capacity || 0) || 0));
    return index >= unlockedCapacity && index < getInventoryPreviewCapacity();
  }

  function getInventorySlotPurchaseOffer(snapshot, tabId) {
    const tab = normalizeInventoryTab(tabId);
    const offers = snapshot && snapshot.inventorySlotExpansion || {};
    const offer = offers[tab] && typeof offers[tab] === 'object' ? offers[tab] : null;
    return offer || {
      tab,
      available: false,
      reason: 'Visit Slot Broker',
      cost: 0,
      slotsAdded: INVENTORY_SECTION_SIZE
    };
  }

  function getInventoryPanelDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (hasAttribute('data-starfall-apply-inventory-sort')) {
      return { handled: true, type: 'applySort', tabId: getAttribute('data-starfall-apply-inventory-sort') };
    }
    const sortId = getAttribute('data-starfall-inventory-sort');
    if (sortId) return { handled: true, type: 'setSort', sortId, tabId: getAttribute('data-starfall-inventory-sort-tab') };
    const inventoryTab = getAttribute('data-starfall-inventory-tab');
    if (inventoryTab) return { handled: true, type: 'selectInventoryTab', tabId: inventoryTab };
    if (hasAttribute('data-starfall-inventory-locked')) {
      return { handled: true, type: 'openLockedSection', tabId: getAttribute('data-starfall-inventory-slot-tab') };
    }
    const storageTab = getAttribute('data-starfall-storage-tab');
    if (storageTab) return { handled: true, type: 'selectStorageTab', tabId: storageTab };
    const sectionUnlockTab = getAttribute('data-starfall-inventory-section-unlock');
    if (sectionUnlockTab) return { handled: true, type: 'openSectionUnlock', tabId: sectionUnlockTab };
    const slotPurchaseTab = getAttribute('data-starfall-inventory-slot-purchase');
    if (slotPurchaseTab) return { handled: true, type: 'purchaseSlots', tabId: slotPurchaseTab };
    if (hasAttribute('data-starfall-inventory-sell-settings')) return { handled: true, type: 'toggleSellSettings' };
    const sellRule = getAttribute('data-starfall-inventory-sell-toggle');
    if (sellRule) return { handled: true, type: 'toggleSellRule', rule: sellRule };
    const sellRarity = getAttribute('data-starfall-inventory-sell-rarity');
    if (sellRarity) return { handled: true, type: 'toggleSellRarity', rarity: sellRarity };
    if (hasAttribute('data-starfall-inventory-sell-reset')) return { handled: true, type: 'resetSellRules' };
    return { handled: false, type: '' };
  }

  function getInventoryPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'inventory-sort') return { handled: true, type: 'cycleSort', tabId: source.tabId };
    if (source.type === 'inventory-sort-apply') return { handled: true, type: 'applySort', tabId: source.tabId };
    if (source.type === 'inventory-tab') return { handled: true, type: 'selectInventoryTab', tabId: source.tabId };
    if (source.type === 'storage-tab') return { handled: true, type: 'selectStorageTab', tabId: source.tabId };
    if (source.type === 'inventory-slot' && source.locked) {
      return { handled: true, type: 'openLockedSection', tabId: source.tabId };
    }
    if (source.type === 'inventory-section-unlock') return { handled: true, type: 'openSectionUnlock', tabId: source.tabId };
    if (source.type === 'inventory-slot-purchase') return { handled: true, type: 'purchaseSlots', tabId: source.tabId };
    if (source.type === 'inventory-sell-settings-toggle') return { handled: true, type: 'toggleSellSettings' };
    if (source.type === 'inventory-sell-rule') return { handled: true, type: 'toggleSellRule', rule: source.ruleId };
    if (source.type === 'inventory-sell-rarity') return { handled: true, type: 'toggleSellRarity', rarity: source.rarity };
    if (source.type === 'inventory-sell-reset') return { handled: true, type: 'resetSellRules' };
    return { handled: false, type: '' };
  }

  function getInventorySortInputDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (!hasAttribute('data-starfall-inventory-sort-choice')) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'setPendingInventorySort',
      sortId: source && source.value,
      tabId: getAttribute('data-starfall-inventory-sort-tab')
    };
  }

  function isInventorySlotBrokerActive(snapshot) {
    return !!(snapshot && snapshot.state && snapshot.state.player && snapshot.state.player.activeStation === 'slots');
  }

  function getCanvasPanelCacheInventoryKey(tabId, options) {
    const settings = options || {};
    const snapshot = settings.snapshot || {};
    const state = snapshot.state || {};
    const tab = normalizeInventoryTab(tabId);
    const getRevisionKey = typeof settings.getCanvasInventoryPanelRevisionKey === 'function'
      ? settings.getCanvasInventoryPanelRevisionKey
      : () => '';
    const getPositiveStackEntryCount = typeof settings.getPositiveStackEntryCount === 'function'
      ? settings.getPositiveStackEntryCount
      : (cacheId, source) => countPositiveStackEntries(source);
    const getPendingInventorySortId = typeof settings.getPendingInventorySortId === 'function'
      ? settings.getPendingInventorySortId
      : () => '';
    const revisionKey = getRevisionKey(tab);
    const stackSource = tab === 'usable' ? state.consumables : state.materials;
    const hasRevisions = hasEnumerableKeys(snapshot.domainRevisions);
    const populatedSlots = tab === 'equipment'
      ? (snapshot.inventory || state.inventory || []).length
      : getPositiveStackEntryCount(`inventory:${tab}`, stackSource, revisionKey, hasRevisions);
    return [
      revisionKey,
      tab,
      populatedSlots,
      getInventorySectionCount(snapshot, tab),
      state.inventorySlots && state.inventorySlots[tab] || 0,
      state.selectedInventoryUid || '',
      getPendingInventorySortId(tab),
      settings.inventorySellSettingsOpen ? 1 : 0,
      Math.round(Number(settings.inventoryCanvasScrollByTab && settings.inventoryCanvasScrollByTab[tab] || 0))
    ].join('|');
  }

  function createInventoryConfigUiHelpers() {
    return Object.freeze({
      normalizeInventoryTab,
      normalizeStorageTab,
      getInventorySortOptions,
      getInventorySortLabel,
      normalizeInventorySortChoice,
      getStackCapForItem,
      getStackSlotCountForCount,
      getStorageStackCapForItem,
      getStorageStackSlotCountForCount,
      createStackedSlotEntries,
      appendStackedSlotEntries,
      createStorageStackedSlotEntries,
      appendStorageStackedSlotEntries,
      getInventorySectionCouponId,
      getInventoryTabLabel,
      normalizeInventorySectionCount,
      getInventorySectionCount,
      getInventoryUnlockedCapacity,
      getInventoryPreviewCapacity,
      countPositiveStackEntries,
      hasEnumerableKeys,
      getLastPopulatedSlotIndex,
      getInventoryGridVisibleSlotCount,
      getFirstAvailableSlotIndex,
      isInventoryPreviewSlotLocked,
      getInventorySlotPurchaseOffer,
      getInventoryPanelDomAction,
      getInventoryPanelRegionAction,
      getInventorySortInputDomAction,
      isInventorySlotBrokerActive,
      getCanvasPanelCacheInventoryKey
    });
  }

  const api = {
    INVENTORY_DOUBLE_CLICK_MS,
    INVENTORY_DOUBLE_CLICK_DISTANCE,
    INVENTORY_COLUMNS,
    INVENTORY_ROWS,
    INVENTORY_SECTION_SIZE,
    INVENTORY_STORAGE_COLUMNS,
    INVENTORY_COUPON_MAX_SECTIONS,
    INVENTORY_MAX_SECTIONS,
    STORAGE_STACK_CAP_MULTIPLIER,
    DEFAULT_USABLE_STACK_CAP,
    DEFAULT_ETC_STACK_CAP,
    STACKABLE_SLOT_LIMITS,
    INVENTORY_CANVAS_VISIBLE_ROWS,
    INVENTORY_CANVAS_CELL_SIZE,
    INVENTORY_CANVAS_CELL_GAP,
    INVENTORY_CANVAS_GRID_HEIGHT,
    EQUIPMENT_INVENTORY_SORT_OPTIONS,
    STACKABLE_INVENTORY_SORT_OPTIONS,
    CARD_INVENTORY_SORT_OPTIONS,
    INVENTORY_SORT_OPTIONS,
    createInventorySortOptions,
    INVENTORY_TAB_OPTIONS,
    STORAGE_TAB_OPTIONS,
    createStorageTabOptions,
    INVENTORY_SELL_RULE_TOOLTIPS,
    normalizeInventoryTab,
    normalizeStorageTab,
    getInventorySortOptions,
    getInventorySortLabel,
    normalizeInventorySortChoice,
    getStackCapForItem,
    getStackSlotCountForCount,
    getStorageStackCapForItem,
    getStorageStackSlotCountForCount,
    createStackedSlotEntries,
    appendStackedSlotEntries,
    createStorageStackedSlotEntries,
    appendStorageStackedSlotEntries,
    getInventorySectionCouponId,
    getInventoryTabLabel,
    normalizeInventorySectionCount,
    getInventorySectionCount,
    getInventoryUnlockedCapacity,
    getInventoryPreviewCapacity,
    countPositiveStackEntries,
    hasEnumerableKeys,
    getLastPopulatedSlotIndex,
    getInventoryGridVisibleSlotCount,
    getFirstAvailableSlotIndex,
    isInventoryPreviewSlotLocked,
    getInventorySlotPurchaseOffer,
    getInventoryPanelDomAction,
    getInventoryPanelRegionAction,
    getInventorySortInputDomAction,
    isInventorySlotBrokerActive,
    getCanvasPanelCacheInventoryKey,
    createInventoryConfigUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.inventoryConfig = Object.assign({}, modules.inventoryConfig || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
