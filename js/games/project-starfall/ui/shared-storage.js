(function initProjectStarfallUiSharedStorage(global) {
  'use strict';

  const SHARED_STORAGE_NORMALIZED_KEY = '__projectStarfallSharedStorageNormalized';

  function clonePlainFallback(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  function normalizeIdFallback(value) {
    return String(value || '').trim();
  }

  function normalizeInventorySectionCountFallback(value) {
    return Math.max(1, Math.floor(Number(value || 1) || 1));
  }

  function getOptions(options) {
    const settings = options || {};
    return {
      characterSlotCount: Math.max(1, Math.floor(Number(settings.characterSlotCount || 8) || 8)),
      clonePlain: settings.clonePlain || clonePlainFallback,
      createCharacterRecord: settings.createCharacterRecord || ((slotId, payload) => payload || null),
      createEmptyCharacterSlot: settings.createEmptyCharacterSlot || ((index) => ({ slotId: `slot_${index + 1}`, index, character: null })),
      normalizeId: settings.normalizeId || normalizeIdFallback,
      normalizeInventorySectionCount: settings.normalizeInventorySectionCount || normalizeInventorySectionCountFallback
    };
  }

  function createSharedStorageSections(value, options) {
    const settings = getOptions(options);
    const source = value && typeof value === 'object' ? value : {};
    return {
      equipment: settings.normalizeInventorySectionCount(source.equipment),
      usable: settings.normalizeInventorySectionCount(source.usable),
      etc: settings.normalizeInventorySectionCount(source.etc)
    };
  }

  function createSharedStorageSlotOrder(value, options) {
    const settings = getOptions(options);
    const source = value && typeof value === 'object' ? value : {};
    return {
      equipment: Array.isArray(source.equipment) ? source.equipment.map(settings.normalizeId).filter(Boolean) : [],
      usable: Array.isArray(source.usable) ? source.usable.map(settings.normalizeId).filter(Boolean) : [],
      etc: Array.isArray(source.etc) ? source.etc.map(settings.normalizeId).filter(Boolean) : []
    };
  }

  function normalizeSharedStorageCounts(value, options) {
    const settings = getOptions(options);
    const source = value && typeof value === 'object' ? value : {};
    return Object.keys(source).reduce((counts, key) => {
      const id = settings.normalizeId(key);
      const count = Math.max(0, Math.floor(Number(source[key]) || 0));
      if (id && count > 0) counts[id] = count;
      return counts;
    }, {});
  }

  function createEmptySharedStorage(options) {
    return {
      version: 1,
      inventory: [],
      consumables: {},
      materials: {},
      inventorySections: createSharedStorageSections(null, options),
      inventorySlotOrder: createSharedStorageSlotOrder(null, options)
    };
  }

  function markSharedStorageNormalized(storage) {
    if (!storage || typeof storage !== 'object') return storage;
    try {
      Object.defineProperty(storage, SHARED_STORAGE_NORMALIZED_KEY, {
        value: true,
        configurable: true
      });
    } catch (err) {
      storage[SHARED_STORAGE_NORMALIZED_KEY] = true;
    }
    return storage;
  }

  function isSharedStorageNormalized(storage) {
    return !!(storage &&
      typeof storage === 'object' &&
      storage[SHARED_STORAGE_NORMALIZED_KEY] === true &&
      Array.isArray(storage.inventory) &&
      storage.consumables && typeof storage.consumables === 'object' &&
      storage.materials && typeof storage.materials === 'object' &&
      storage.inventorySections && typeof storage.inventorySections === 'object' &&
      storage.inventorySlotOrder && typeof storage.inventorySlotOrder === 'object');
  }

  function normalizeSharedStorage(source, options) {
    const settings = getOptions(options);
    const storage = createEmptySharedStorage(settings);
    const data = source && typeof source === 'object' ? source : {};
    storage.version = Math.max(1, Math.floor(Number(data.version || 1) || 1));
    storage.inventory = Array.isArray(data.inventory)
      ? data.inventory.filter((item) => item && typeof item === 'object' && settings.normalizeId(item.uid)).map((item) => settings.clonePlain(item))
      : [];
    storage.consumables = normalizeSharedStorageCounts(data.consumables, settings);
    storage.materials = normalizeSharedStorageCounts(data.materials, settings);
    storage.inventorySections = createSharedStorageSections(data.inventorySections, settings);
    storage.inventorySlotOrder = createSharedStorageSlotOrder(data.inventorySlotOrder, settings);
    return markSharedStorageNormalized(storage);
  }

  function createEmptyCharacterRoster(options) {
    const settings = getOptions(options);
    return {
      version: 1,
      activeSlotId: '',
      sharedStorage: createEmptySharedStorage(settings),
      slots: Array.from({ length: settings.characterSlotCount }, (_, index) => settings.createEmptyCharacterSlot(index))
    };
  }

  function normalizeCharacterRoster(source, options) {
    const settings = getOptions(options);
    const roster = createEmptyCharacterRoster(settings);
    if (!source || typeof source !== 'object') return roster;
    roster.activeSlotId = String(source.activeSlotId || '');
    roster.sharedStorage = normalizeSharedStorage(source.sharedStorage, settings);
    const sourceSlots = Array.isArray(source.slots) ? source.slots : [];
    roster.slots = roster.slots.map((emptySlot, index) => {
      const sourceSlot = sourceSlots[index] || sourceSlots.find((slot) => slot && slot.slotId === emptySlot.slotId) || null;
      if (!sourceSlot || !sourceSlot.character) return emptySlot;
      return {
        slotId: emptySlot.slotId,
        index,
        character: settings.createCharacterRecord(emptySlot.slotId, sourceSlot.character.payload || sourceSlot.character)
      };
    });
    if (!roster.slots.some((slot) => slot.slotId === roster.activeSlotId && slot.character)) roster.activeSlotId = '';
    return roster;
  }

  function getCanvasPanelCacheStorageKey(tabId, options) {
    const settings = options || {};
    const normalizeInventoryTab = settings.normalizeInventoryTab || function normalizeInventoryTabFallback(value) {
      return String(value || '').trim() || 'equipment';
    };
    const getRevisionKey = typeof settings.getCanvasPanelSnapshotRevision === 'function'
      ? settings.getCanvasPanelSnapshotRevision
      : () => '';
    const getUsedSlots = typeof settings.getSharedStorageUsedSlots === 'function'
      ? settings.getSharedStorageUsedSlots
      : () => 0;
    const tab = normalizeInventoryTab(tabId || settings.storageTab);
    const storage = settings.storage && typeof settings.storage === 'object' ? settings.storage : {};
    return [
      getRevisionKey('storage'),
      tab,
      getUsedSlots(tab),
      storage.inventorySections && storage.inventorySections[tab] || 0,
      storage.inventorySlotOrder && storage.inventorySlotOrder[tab] && storage.inventorySlotOrder[tab].length || 0,
      Math.round(Number(settings.storageCanvasScrollByTab && settings.storageCanvasScrollByTab[tab] || 0))
    ].join('|');
  }

  function createSharedStorageUiHelpers(options) {
    const settings = options || {};
    const normalizedOptions = getOptions(settings);
    return Object.freeze({
      createSharedStorageSections: (value) => createSharedStorageSections(value, normalizedOptions),
      createSharedStorageSlotOrder: (value) => createSharedStorageSlotOrder(value, normalizedOptions),
      normalizeSharedStorageCounts: (value) => normalizeSharedStorageCounts(value, normalizedOptions),
      createEmptySharedStorage: () => createEmptySharedStorage(normalizedOptions),
      markSharedStorageNormalized,
      isSharedStorageNormalized,
      normalizeSharedStorage: (source) => normalizeSharedStorage(source, normalizedOptions),
      createEmptyCharacterRoster: () => createEmptyCharacterRoster(normalizedOptions),
      normalizeCharacterRoster: (source) => normalizeCharacterRoster(source, normalizedOptions),
      getCanvasPanelCacheStorageKey: (tabId, overrides) => getCanvasPanelCacheStorageKey(tabId, Object.assign({}, settings, overrides || {}))
    });
  }

  const api = {
    SHARED_STORAGE_NORMALIZED_KEY,
    createSharedStorageSections,
    createSharedStorageSlotOrder,
    normalizeSharedStorageCounts,
    createEmptySharedStorage,
    markSharedStorageNormalized,
    isSharedStorageNormalized,
    normalizeSharedStorage,
    createEmptyCharacterRoster,
    normalizeCharacterRoster,
    getCanvasPanelCacheStorageKey,
    createSharedStorageUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.sharedStorage = Object.assign({}, modules.sharedStorage || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
