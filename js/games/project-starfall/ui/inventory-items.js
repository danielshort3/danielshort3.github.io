(function initProjectStarfallUiInventoryItems(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiFormatting = (typeof require === 'function' ? require('./formatting.js') : null) || UiModules.formatting || {};
  const formatStatName = UiFormatting.formatStatName || function formatStatNameFallback(key) {
    return String(key || '')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (match) => match.toUpperCase());
  };

  const DEFAULT_MATERIAL_DISPLAY_ITEMS = Object.freeze([
    Object.freeze({ id: 'upgradeDust', materialId: 'upgradeDust', assetId: 'upgrade_dust', name: 'Upgrade Dust', icon: 'UD', rarity: 'Uncommon' }),
    Object.freeze({ id: 'upgradeCatalyst', materialId: 'upgradeCatalyst', assetId: 'upgrade_catalyst', name: 'Upgrade Catalyst', icon: 'UC', rarity: 'Rare' }),
    Object.freeze({ id: 'wardingScroll', materialId: 'wardingScroll', assetId: 'warding_scroll', name: 'Warding Scroll', icon: 'WS', rarity: 'Rare' }),
    Object.freeze({ id: 'refinementCore', materialId: 'refinementCore', assetId: 'refinement_core', name: 'Refinement Core', icon: 'RC', rarity: 'Epic' }),
    Object.freeze({ id: 'cubeFragment', materialId: 'cubeFragment', assetId: 'cube_fragment', name: 'Prism Shard', icon: 'PS', rarity: 'Rare' }),
    Object.freeze({ id: 'whiteStarCard', materialId: 'whiteStarCard', assetId: 'white_star_card', name: 'White Star Card', icon: 'WSC', rarity: 'Common' }),
    Object.freeze({ id: 'greenStarCard', materialId: 'greenStarCard', assetId: 'green_star_card', name: 'Green Star Card', icon: 'GSC', rarity: 'Uncommon' }),
    Object.freeze({ id: 'blueStarCard', materialId: 'blueStarCard', assetId: 'blue_star_card', name: 'Blue Star Card', icon: 'BSC', rarity: 'Rare' }),
    Object.freeze({ id: 'purpleStarCard', materialId: 'purpleStarCard', assetId: 'purple_star_card', name: 'Purple Star Card', icon: 'PSC', rarity: 'Epic' }),
    Object.freeze({ id: 'orangeStarCard', materialId: 'orangeStarCard', assetId: 'orange_star_card', name: 'Orange Star Card', icon: 'OSC', rarity: 'Relic' }),
    Object.freeze({ id: 'gelDrop', materialId: 'gelDrop', assetId: 'gel_drop', name: 'Gel Drop', icon: 'GD', rarity: 'Common' }),
    Object.freeze({ id: 'oreChunks', materialId: 'oreChunks', assetId: 'ore_chunks', name: 'Ore Chunks', icon: 'OR', rarity: 'Common' })
  ]);

  function getMaterialDisplayItems(data) {
    const source = data || Data;
    return Object.freeze(Array.isArray(source.MATERIAL_ITEMS) && source.MATERIAL_ITEMS.length ? source.MATERIAL_ITEMS : DEFAULT_MATERIAL_DISPLAY_ITEMS);
  }

  function getMaterialDisplayOrder(items) {
    const source = Array.isArray(items) ? items : MATERIAL_DISPLAY_ITEMS;
    return Object.freeze(source.map((item) => String(item.materialId || item.id || '')).filter(Boolean));
  }

  function getMaterialDisplayOrderSet(order) {
    const source = Array.isArray(order) ? order : MATERIAL_DISPLAY_ORDER;
    return new Set(source);
  }

  function getMaterialDisplayMeta(items, options) {
    const source = Array.isArray(items) ? items : MATERIAL_DISPLAY_ITEMS;
    const formatName = options && options.formatStatName || formatStatName;
    return Object.freeze(source.reduce((meta, item) => {
      const id = String(item.materialId || item.id || '');
      if (!id) return meta;
      meta[id] = Object.freeze({
        id: item.assetId || id.replace(/([A-Z])/g, '_$1').toLowerCase(),
        name: item.name || formatName(id),
        icon: item.icon || id.slice(0, 3).toUpperCase(),
        rarity: item.rarity || 'Common'
      });
      return meta;
    }, {}));
  }

  function getConsumableItemsById(data) {
    const source = data || Data;
    return Object.freeze((source.CONSUMABLE_ITEMS || []).reduce((map, item) => {
      const id = String(item && item.id || '');
      if (id) map[id] = item;
      return map;
    }, {}));
  }

  function getConsumableItemOrder(data) {
    const source = data || Data;
    return Object.freeze((source.CONSUMABLE_ITEMS || [])
      .map((item) => String(item && item.id || ''))
      .filter(Boolean));
  }

  function getConsumableItemOrderIndex(order) {
    const source = Array.isArray(order) ? order : CONSUMABLE_ITEM_ORDER;
    return Object.freeze(source.reduce((map, id, index) => {
      map[id] = index;
      return map;
    }, {}));
  }

  const MATERIAL_DISPLAY_ITEMS = getMaterialDisplayItems(Data);
  const MATERIAL_DISPLAY_ORDER = getMaterialDisplayOrder(MATERIAL_DISPLAY_ITEMS);
  const MATERIAL_DISPLAY_ORDER_SET = getMaterialDisplayOrderSet(MATERIAL_DISPLAY_ORDER);
  const MATERIAL_DISPLAY_META = getMaterialDisplayMeta(MATERIAL_DISPLAY_ITEMS);
  const CONSUMABLE_ITEMS_BY_ID = getConsumableItemsById(Data);
  const CONSUMABLE_ITEM_ORDER = getConsumableItemOrder(Data);
  const CONSUMABLE_ITEM_ORDER_INDEX = getConsumableItemOrderIndex(CONSUMABLE_ITEM_ORDER);

  function getMaterialAssetId(materialId, options) {
    const materialMeta = options && options.materialDisplayMeta || MATERIAL_DISPLAY_META;
    const meta = materialMeta[materialId];
    if (meta && meta.id) return meta.id;
    return String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase();
  }

  function getMaterialInventoryEntry(materials, materialId, options) {
    const id = String(materialId || '').trim();
    if (!id) return null;
    const source = materials && typeof materials === 'object' ? materials : {};
    const count = Math.max(0, Math.floor(Number(source[id]) || 0));
    if (count <= 0) return null;
    const data = options && options.data || Data;
    const materialMeta = options && options.materialDisplayMeta || MATERIAL_DISPLAY_META;
    const formatName = options && options.formatStatName || formatStatName;
    const meta = materialMeta[id] || {};
    const assetId = getMaterialAssetId(id, { materialDisplayMeta: materialMeta });
    return {
      id,
      count,
      name: meta.name || formatName(id),
      icon: meta.icon || String(id || 'mat').slice(0, 3).toUpperCase(),
      rarity: meta.rarity || 'Common',
      asset: data.ITEM_ASSETS && data.ITEM_ASSETS[assetId] || ''
    };
  }

  function getMaterialInventoryEntries(materials, options) {
    const source = materials && typeof materials === 'object' ? materials : {};
    const order = options && options.materialDisplayOrder || MATERIAL_DISPLAY_ORDER;
    const orderSet = options && options.materialDisplayOrderSet || MATERIAL_DISPLAY_ORDER_SET;
    const formatName = options && options.formatStatName || formatStatName;
    const orderedIds = order.concat(
      Object.keys(source)
        .filter((id) => !orderSet.has(id))
        .sort((a, b) => formatName(a).localeCompare(formatName(b)))
    );
    return orderedIds
      .map((id) => getMaterialInventoryEntry(source, id, options))
      .filter(Boolean);
  }

  function getConsumableItem(itemId, options) {
    const itemsById = options && options.consumableItemsById || CONSUMABLE_ITEMS_BY_ID;
    return itemsById[String(itemId || '')] || null;
  }

  function getOwnedConsumableIds(counts, options) {
    const source = counts && typeof counts === 'object' ? counts : {};
    const countIds = Object.keys(source);
    const order = options && options.consumableItemOrder || CONSUMABLE_ITEM_ORDER;
    const orderIndex = options && options.consumableItemOrderIndex || CONSUMABLE_ITEM_ORDER_INDEX;
    const itemsById = options && options.consumableItemsById || CONSUMABLE_ITEMS_BY_ID;
    if (countIds.length * 2 >= order.length) {
      return order.filter((id) => Number(source[id] || 0) > 0 && itemsById[id]);
    }
    return countIds
      .filter((id) => Number(source[id] || 0) > 0 && itemsById[id])
      .sort((a, b) => {
        const orderA = orderIndex[a];
        const orderB = orderIndex[b];
        return (orderA == null ? Number.MAX_SAFE_INTEGER : orderA) - (orderB == null ? Number.MAX_SAFE_INTEGER : orderB);
      });
  }

  function getCanvasPanelCacheItemKey(item, cache) {
    if (!item) return '';
    const keyCache = item && typeof item === 'object' && cache && typeof cache.get === 'function' && typeof cache.set === 'function'
      ? cache
      : null;
    const cached = keyCache ? keyCache.get(item) : null;
    if (typeof cached === 'string') return cached;
    const stats = item.stats || item.bonusStats || {};
    const potential = item.potential || item.attunement || {};
    const key = [
      item.uid || '',
      item.id || '',
      item.slot || '',
      item.rarity || '',
      item.level || 0,
      item.upgrade || 0,
      item.locked ? 1 : 0,
      item.count || 0,
      item.asset || '',
      JSON.stringify(stats),
      JSON.stringify(potential),
      JSON.stringify(item.set || item.setId || '')
    ].join(':');
    if (keyCache) keyCache.set(item, key);
    return key;
  }

  function getCanvasPanelCacheConsumableKey(item) {
    if (!item) return '';
    return [item.id || '', item.count || 0, item.asset || '', item.cooldown || 0, item.type || ''].join(':');
  }

  function getCanvasPanelCacheMaterialKey(item) {
    if (!item) return '';
    return [item.id || '', item.count || 0, item.asset || '', item.value || 0, item.type || ''].join(':');
  }

  function getInventoryItemDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const lockUid = getAttribute('data-starfall-toggle-lock');
    if (lockUid) return { handled: true, type: 'toggleLock', uid: lockUid };
    if (hasAttribute('data-starfall-bulk-sell')) return { handled: true, type: 'bulkSell' };
    return { handled: false, type: '' };
  }

  function getInventoryConsumableDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const selectableConsumableId = getAttribute('data-starfall-select-consumable');
    if (selectableConsumableId) return { handled: true, type: 'selectConsumable', itemId: selectableConsumableId };
    const consumableId = getAttribute('data-starfall-use-consumable');
    if (consumableId) return { handled: true, type: 'useConsumable', itemId: consumableId };
    return { handled: false, type: '' };
  }

  function getInventoryEquipmentDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const equipUid = getAttribute('data-starfall-equip');
    if (equipUid) return { handled: true, type: 'equip', uid: equipUid };
    const unequipUid = getAttribute('data-starfall-unequip');
    if (unequipUid) return { handled: true, type: 'unequip', uid: unequipUid };
    const selectUid = getAttribute('data-starfall-select-item');
    if (selectUid) return { handled: true, type: 'selectItem', uid: selectUid };
    const selectEtcId = getAttribute('data-starfall-select-etc');
    if (selectEtcId) return { handled: true, type: 'selectEtc', itemId: selectEtcId };
    return { handled: false, type: '' };
  }

  function getInventoryItemRegionAction(region) {
    const source = region || {};
    if (source.type === 'toggle-item-lock') return { handled: true, type: 'toggleLock', uid: source.uid };
    if (source.type === 'bulk-sell-weak') return { handled: true, type: 'bulkSell' };
    if (
      source.type === 'equip-item' ||
      source.type === 'inventory-item' ||
      source.type === 'storage-item'
    ) {
      return { handled: true, type: 'ignoreItemActivation' };
    }
    if (
      source.type === 'use-consumable' ||
      source.type === 'consumable-item' ||
      source.type === 'etc-item' ||
      source.type === 'storage-consumable-item' ||
      source.type === 'storage-etc-item'
    ) {
      return { handled: true, type: 'ignoreConsumableActivation' };
    }
    return { handled: false, type: '' };
  }

  function getInventoryDerivedSnapshotUpdate(engine) {
    const source = engine || {};
    const update = {};
    if (typeof source.getOrderedInventoryItems === 'function') {
      update.inventory = source.getOrderedInventoryItems();
    }
    if (typeof source.getInventorySlotExpansionSnapshot === 'function') {
      update.inventorySlotExpansion = source.getInventorySlotExpansionSnapshot();
    }
    if (typeof source.getInventorySellPreview === 'function') {
      const preview = source.getInventorySellPreview();
      update.inventorySellPreview = preview;
      update.weakInventorySell = preview;
    }
    if (typeof source.getInventorySellRules === 'function') {
      update.inventorySellRules = source.getInventorySellRules();
    }
    return update;
  }

  function createInventoryItemUiHelpers() {
    return Object.freeze({
      getMaterialAssetId,
      getMaterialInventoryEntry,
      getMaterialInventoryEntries,
      getConsumableItem,
      getOwnedConsumableIds,
      getCanvasPanelCacheItemKey,
      getCanvasPanelCacheConsumableKey,
      getCanvasPanelCacheMaterialKey,
      getInventoryItemDomAction,
      getInventoryConsumableDomAction,
      getInventoryEquipmentDomAction,
      getInventoryItemRegionAction,
      getInventoryDerivedSnapshotUpdate
    });
  }

  const api = {
    DEFAULT_MATERIAL_DISPLAY_ITEMS,
    MATERIAL_DISPLAY_ITEMS,
    MATERIAL_DISPLAY_ORDER,
    MATERIAL_DISPLAY_ORDER_SET,
    MATERIAL_DISPLAY_META,
    CONSUMABLE_ITEMS_BY_ID,
    CONSUMABLE_ITEM_ORDER,
    CONSUMABLE_ITEM_ORDER_INDEX,
    getMaterialDisplayItems,
    getMaterialDisplayOrder,
    getMaterialDisplayOrderSet,
    getMaterialDisplayMeta,
    getConsumableItemsById,
    getConsumableItemOrder,
    getConsumableItemOrderIndex,
    getMaterialAssetId,
    getMaterialInventoryEntry,
    getMaterialInventoryEntries,
    getConsumableItem,
    getOwnedConsumableIds,
    getCanvasPanelCacheItemKey,
    getCanvasPanelCacheConsumableKey,
    getCanvasPanelCacheMaterialKey,
    getInventoryItemDomAction,
    getInventoryConsumableDomAction,
    getInventoryEquipmentDomAction,
    getInventoryItemRegionAction,
    getInventoryDerivedSnapshotUpdate,
    createInventoryItemUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.inventoryItems = Object.assign({}, modules.inventoryItems || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
