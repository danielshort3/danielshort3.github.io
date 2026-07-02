(function initProjectStarfallEngineInventory(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const INVENTORY_SECTION_SIZE = 36;
  const INVENTORY_COUPON_MAX_SECTIONS = 8;
  const INVENTORY_MAX_SECTIONS = 8;
  const INVENTORY_SLOT_COIN_PURCHASE_BASE = 1;
  const INVENTORY_SLOT_COIN_PURCHASE_GROWTH = 1.16;
  const DEFAULT_EQUIPMENT_INVENTORY_SLOTS = INVENTORY_SECTION_SIZE;
  const MIN_EQUIPMENT_INVENTORY_SLOTS = INVENTORY_SECTION_SIZE;
  const MAX_EQUIPMENT_INVENTORY_SLOTS = INVENTORY_SECTION_SIZE * INVENTORY_MAX_SECTIONS;
  const EQUIPMENT_SLOT_COUPON_SIZE = INVENTORY_SECTION_SIZE;
  const BULK_SELL_VALUE_RATE = 0.25;
  const DEFAULT_USABLE_STACK_CAP = 100;
  const DEFAULT_ETC_STACK_CAP = 250;
  const STACKABLE_SLOT_LIMITS = Object.freeze({
    usable: Object.freeze({
      admin_worldwright_console: 1,
      pet_whistle: 1,
      minor_health_potion: 100,
      minor_resource_tonic: 100,
      camp_ration: 100,
      standard_health_potion: 100,
      standard_resource_tonic: 100,
      field_ration: 100,
      greater_health_potion: 100,
      greater_resource_tonic: 100,
      expedition_ration: 100,
      superior_health_potion: 100,
      superior_resource_tonic: 100,
      hero_ration: 100,
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
  const INVENTORY_SELL_RARITIES = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic']);
  const INVENTORY_SELL_THRESHOLD_MIN = -9999;
  const INVENTORY_SELL_THRESHOLD_MAX = 9999;
  const DEFAULT_INVENTORY_SELL_RARITIES = Object.freeze(INVENTORY_SELL_RARITIES.reduce((rarities, rarity) => {
    rarities[rarity] = false;
    return rarities;
  }, {}));
  const DEFAULT_INVENTORY_SELL_RULES = Object.freeze({
    autoSell: false,
    matchWeak: true,
    matchNonClass: false,
    rarities: DEFAULT_INVENTORY_SELL_RARITIES
  });
  const INVENTORY_SORT_OPTIONS = Object.freeze(['powerDesc', 'powerAsc', 'type', 'newest', 'value', 'quantity', 'name']);
  const INVENTORY_SORT_OPTIONS_BY_TAB = Object.freeze({
    equipment: Object.freeze(['powerDesc', 'powerAsc', 'type', 'newest']),
    usable: Object.freeze(['type', 'value', 'quantity', 'name']),
    etc: Object.freeze(['type', 'value', 'quantity', 'name']),
    cards: Object.freeze(['value', 'newest', 'name'])
  });
  const INVENTORY_TAB_OPTIONS = Object.freeze(['equipment', 'usable', 'etc', 'cards']);
  const RECOVERY_CONSUMABLE_SORT_ORDER = Object.freeze({
    minor_health_potion: 0,
    standard_health_potion: 1,
    greater_health_potion: 2,
    superior_health_potion: 3,
    minor_resource_tonic: 100,
    standard_resource_tonic: 101,
    greater_resource_tonic: 102,
    superior_resource_tonic: 103,
    camp_ration: 200,
    field_ration: 201,
    expedition_ration: 202,
    hero_ration: 203
  });
  const UTILITY_CONSUMABLE_SORT_ORDER = Object.freeze({
    guard_tonic: 0,
    swiftstep_oil: 1,
    magnet_charm: 2
  });
  const PLINKO_BALL_SORT_ORDER = Object.freeze({
    plinko_ball_basic: 0,
    plinko_ball_polished: 1,
    plinko_ball_meteor: 2
  });
  const SLOT_COUPON_SORT_ORDER = Object.freeze({
    equipment_slot_coupon: 0,
    usable_slot_coupon: 1,
    etc_slot_coupon: 2,
    card_slot_coupon: 3
  });
  const PRISM_CONSUMABLE_SORT_ORDER = Object.freeze({
    potential_cube: 0,
    preservation_cube: 1,
    line_catalyst: 2
  });
  const MANUAL_CONSUMABLE_SORT_ORDER = Object.freeze({
    base_skill_manual: 0,
    advanced_skill_manual: 1,
    skill_reset_scroll: 2,
    stat_reset_scroll: 3
  });
  const UPGRADE_AIDE_MATERIAL_SORT_ORDER = Object.freeze({
    refinementCore: 0,
    wardingScroll: 1,
    upgradeCatalyst: 2
  });
  const STAR_CARD_MATERIAL_SORT_ORDER = Object.freeze({
    whiteStarCard: 0,
    greenStarCard: 1,
    blueStarCard: 2,
    purpleStarCard: 3,
    orangeStarCard: 4
  });
  const MATERIAL_VALUE_PRIORITY = Object.freeze({
    refinementCore: 0,
    wardingScroll: 1,
    upgradeCatalyst: 2,
    cubeFragment: 3,
    upgradeDust: 4,
    oreChunks: 5,
    gelDrop: 6
  });

  function getMappedSortOrder(map, id, fallback) {
    const key = normalizeId(id);
    return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : fallback;
  }

  function isRecoveryConsumable(item) {
    return !!(item && (item.hpPercent || item.resourcePercent || item.hpFlat || item.resourceFlat));
  }

  function getRecoveryConsumableFallbackOrder(item) {
    if (!item) return 999;
    const hp = Math.max(0, Number(item.hpFlat || item.hpPercent || 0) || 0);
    const resource = Math.max(0, Number(item.resourceFlat || item.resourcePercent || 0) || 0);
    if (hp > 0 && resource <= 0) return 50 + hp;
    if (resource > 0 && hp <= 0) return 150 + resource;
    if (hp > 0 || resource > 0) return 250 + hp + resource;
    return 999;
  }

  function getConsumableSortType(item) {
    if (!item) return 99;
    if (isRecoveryConsumable(item)) return 0;
    if (normalizeId(item.rateBuffType) === 'xp') return 1;
    if (normalizeId(item.rateBuffType) === 'drop') return 2;
    if (item.buffId) return 3;
    if (item.plinkoBall) return 4;
    if (item.returnMapId) return 5;
    if (item.inventorySectionCoupon) return 6;
    if (item.potentialCube || item.preservationCube || item.lineCatalyst) return 7;
    if (item.petUnlock) return 8;
    if (item.skillPointPool || item.resetSkillPoints || item.resetStatUpgrades) return 9;
    if (item.adminOnly) return 10;
    return 99;
  }

  function getConsumableSortSubPriority(item) {
    if (!item) return 0;
    const id = normalizeId(item.id);
    if (isRecoveryConsumable(item)) {
      return getMappedSortOrder(RECOVERY_CONSUMABLE_SORT_ORDER, id, getRecoveryConsumableFallbackOrder(item));
    }
    if (item.rateBuffType) return Math.max(0, Math.round(Number(item.rateMultiplier || 1) * 100));
    if (item.buffId) return getMappedSortOrder(UTILITY_CONSUMABLE_SORT_ORDER, id, 99);
    if (item.plinkoBall) return getMappedSortOrder(PLINKO_BALL_SORT_ORDER, id, 99);
    if (item.inventorySectionCoupon) return getMappedSortOrder(SLOT_COUPON_SORT_ORDER, id, 99);
    if (item.potentialCube || item.preservationCube || item.lineCatalyst) return getMappedSortOrder(PRISM_CONSUMABLE_SORT_ORDER, id, 99);
    if (item.skillPointPool || item.resetSkillPoints || item.resetStatUpgrades) return getMappedSortOrder(MANUAL_CONSUMABLE_SORT_ORDER, id, 99);
    return 0;
  }

  function getConsumableValuePriority(item) {
    if (!item) return 99;
    if (item.preservationCube) return 0;
    if (item.potentialCube) return 1;
    if (item.lineCatalyst) return 2;
    if (item.skillPointPool) return 2;
    if (item.resetSkillPoints || item.resetStatUpgrades) return 3;
    if (item.inventorySectionCoupon) return 4;
    if (item.petUnlock) return 5;
    if (item.buffId) return 6;
    if (item.returnMapId) return 7;
    if (item.hpPercent || item.resourcePercent || item.hpFlat || item.resourceFlat) return 8;
    return 99;
  }

  function getMaterialSortType(materialId, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const cubeFragmentMaterialId = normalizeId(settings.cubeFragmentMaterialId || 'cubeFragment');
    const id = normalizeId(materialId);
    if (Object.prototype.hasOwnProperty.call(UPGRADE_AIDE_MATERIAL_SORT_ORDER, id)) return 0;
    if (id === cubeFragmentMaterialId) return 1;
    if (id === 'upgradeDust') return 2;
    if (Object.prototype.hasOwnProperty.call(STAR_CARD_MATERIAL_SORT_ORDER, id)) return 3;
    if (materialDropDefinitions[id]) return 4;
    return 99;
  }

  function getMaterialSortSubPriority(materialId, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const materialSourceIndexById = settings.materialSourceIndexById || {};
    const cubeFragmentMaterialId = normalizeId(settings.cubeFragmentMaterialId || 'cubeFragment');
    const id = normalizeId(materialId);
    if (Object.prototype.hasOwnProperty.call(UPGRADE_AIDE_MATERIAL_SORT_ORDER, id)) return UPGRADE_AIDE_MATERIAL_SORT_ORDER[id];
    if (id === cubeFragmentMaterialId || id === 'upgradeDust') return 0;
    if (Object.prototype.hasOwnProperty.call(STAR_CARD_MATERIAL_SORT_ORDER, id)) return STAR_CARD_MATERIAL_SORT_ORDER[id];
    if (materialDropDefinitions[id]) return getMappedSortOrder(materialSourceIndexById, id, 999);
    return 999;
  }

  function getMaterialValuePriority(materialId) {
    const id = normalizeId(materialId);
    return Object.prototype.hasOwnProperty.call(MATERIAL_VALUE_PRIORITY, id) ? MATERIAL_VALUE_PRIORITY[id] : 99;
  }

  function getMaterialSortMeta(materialId, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const id = normalizeId(materialId);
    const meta = materialDropDefinitions[id] || {};
    return {
      id,
      name: meta.name || String(id || '').replace(/([A-Z])/g, ' $1').replace(/^./, (letter) => letter.toUpperCase()),
      rarity: meta.rarity || 'Common'
    };
  }

  function normalizeInventoryTab(value) {
    const id = normalizeId(value);
    return INVENTORY_TAB_OPTIONS.includes(id) ? id : 'equipment';
  }

  function getInventoryTabLabel(tabId) {
    const labels = { equipment: 'Equipment', usable: 'Usable', etc: 'Etc', cards: 'Cards' };
    return labels[normalizeInventoryTab(tabId)] || 'Inventory';
  }

  function getInventorySortOptions(tabId) {
    return INVENTORY_SORT_OPTIONS_BY_TAB[normalizeInventoryTab(tabId)] || INVENTORY_SORT_OPTIONS_BY_TAB.equipment;
  }

  function normalizeInventorySort(value, tabId) {
    const options = getInventorySortOptions(tabId);
    const id = normalizeId(value);
    return options.includes(id) ? id : options[0] || 'powerDesc';
  }

  function getStackableItemCap(tabId, itemId, options) {
    const settings = options || {};
    const tab = normalizeInventoryTab(tabId);
    const id = normalizeId(itemId);
    if (tab === 'usable') {
      if (STACKABLE_SLOT_LIMITS.usable[id]) return STACKABLE_SLOT_LIMITS.usable[id];
      const item = typeof settings.getConsumableDefinitionById === 'function'
        ? settings.getConsumableDefinitionById(id)
        : settings.consumableDefinition;
      if (item && item.rateBuffType) return 25;
      if (item && (item.potentialCube || item.preservationCube || item.inventorySectionCoupon || item.skillPointPool || item.resetSkillPoints || item.resetStatUpgrades || item.returnMapId)) return 25;
      if (item && item.buffId) return 50;
      return DEFAULT_USABLE_STACK_CAP;
    }
    if (tab === 'etc') return STACKABLE_SLOT_LIMITS.etc[id] || DEFAULT_ETC_STACK_CAP;
    return 1;
  }

  function getStackSlotCount(tabId, itemId, count, options) {
    const quantity = Math.max(0, Math.floor(Number(count) || 0));
    if (quantity <= 0) return 0;
    return Math.ceil(quantity / Math.max(1, getStackableItemCap(tabId, itemId, options)));
  }

  function createStackedSlotEntries(tabId, itemId, count, createEntry, options) {
    const tab = normalizeInventoryTab(tabId);
    const id = normalizeId(itemId);
    const total = Math.max(0, Math.floor(Number(count) || 0));
    const cap = getStackableItemCap(tab, id, options);
    const slots = getStackSlotCount(tab, id, total, options);
    return Array.from({ length: slots }, (_, index) => {
      const stackCount = Math.min(cap, Math.max(0, total - index * cap));
      const base = typeof createEntry === 'function' ? createEntry(id, stackCount, index, slots) : { id };
      return Object.assign({}, base || { id }, {
        id: base && base.id ? base.id : id,
        stackKey: `${tab}:${id}:${index}`,
        stackIndex: index,
        stackSlots: slots,
        stackCount,
        count: stackCount,
        totalCount: total,
        maxStack: cap
      });
    });
  }

  function getInventoryData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createConsumableState(value, includeStarter, options) {
    const data = getInventoryData(options);
    const source = value && typeof value === 'object' ? value : {};
    const consumables = {};
    (data.CONSUMABLE_ITEMS || []).forEach((item) => {
      const starter = includeStarter ? Number((data.STARTER_CONSUMABLES || {})[item.id] || 0) : 0;
      consumables[item.id] = Math.max(0, Math.floor(Number(source[item.id] != null ? source[item.id] : starter) || 0));
    });
    Object.keys(source).forEach((id) => {
      if (Object.prototype.hasOwnProperty.call(consumables, id)) return;
      if (id === 'repair_seal') return;
      consumables[id] = Math.max(0, Math.floor(Number(source[id]) || 0));
    });
    return consumables;
  }

  function ensureAdminConsoleItem(state, options) {
    const settings = options || {};
    const data = getInventoryData(settings);
    const itemId = normalizeId(settings.adminConsoleItemId || 'admin_worldwright_console');
    if (!state || typeof state !== 'object' || !itemId) return state;
    const hasConsoleDefinition = (data.CONSUMABLE_ITEMS || []).some((item) => item && item.id === itemId);
    if (!hasConsoleDefinition) return state;
    state.consumables = createConsumableState(state.consumables, false, settings);
    state.consumables[itemId] = Math.max(1, Math.floor(Number(state.consumables[itemId] || 0) || 0));
    return state;
  }

  function createMaterialStarterMap(options) {
    const settings = options || {};
    if (settings.materialStarters && typeof settings.materialStarters === 'object') return settings.materialStarters;
    const data = getInventoryData(settings);
    return (data.MATERIAL_ITEMS || []).reduce((starters, item) => {
      const materialId = normalizeId(item && (item.materialId || item.id));
      if (materialId) starters[materialId] = Math.max(0, Math.floor(Number(item.starterQuantity || 0) || 0));
      return starters;
    }, {});
  }

  function createMaterialState(value, includeStarter, options) {
    const source = value && typeof value === 'object' ? value : {};
    const materialStarters = createMaterialStarterMap(options);
    const materials = {};
    Object.keys(materialStarters).forEach((key) => {
      const starter = includeStarter ? Number(materialStarters[key] || 0) : 0;
      materials[key] = Math.max(0, Math.floor(Number(source[key] != null ? source[key] : starter) || 0));
    });
    Object.keys(source).forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(materials, key)) return;
      if (key === 'fractureDust') return;
      materials[key] = Math.max(0, Math.floor(Number(source[key]) || 0));
    });
    return materials;
  }

  function createInventorySlots(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      equipment: clamp(
        Math.floor(Number(source.equipment || DEFAULT_EQUIPMENT_INVENTORY_SLOTS) || DEFAULT_EQUIPMENT_INVENTORY_SLOTS),
        MIN_EQUIPMENT_INVENTORY_SLOTS,
        MAX_EQUIPMENT_INVENTORY_SLOTS
      )
    };
  }

  function normalizeInventorySellRarity(value) {
    const id = normalizeId(value).toLowerCase();
    return INVENTORY_SELL_RARITIES.find((rarity) => normalizeId(rarity).toLowerCase() === id) || '';
  }

  function normalizeInventorySellThreshold(value) {
    return clamp(
      Math.round(Number(value == null || value === '' ? 0 : value) || 0),
      INVENTORY_SELL_THRESHOLD_MIN,
      INVENTORY_SELL_THRESHOLD_MAX
    );
  }

  function createInventorySellRarityMap(value) {
    const source = value && typeof value === 'object' ? value : {};
    return INVENTORY_SELL_RARITIES.reduce((rarities, rarity) => {
      const lowerKey = normalizeId(rarity).toLowerCase();
      const sourceKey = Object.keys(source).find((key) => normalizeId(key).toLowerCase() === lowerKey);
      const direct = sourceKey ? source[sourceKey] : false;
      rarities[rarity] = !!direct;
      return rarities;
    }, {});
  }

  function createInventorySellRules(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      autoSell: Object.prototype.hasOwnProperty.call(source, 'autoSell') ? !!source.autoSell : DEFAULT_INVENTORY_SELL_RULES.autoSell,
      matchWeak: Object.prototype.hasOwnProperty.call(source, 'matchWeak') ? !!source.matchWeak : DEFAULT_INVENTORY_SELL_RULES.matchWeak,
      matchNonClass: Object.prototype.hasOwnProperty.call(source, 'matchNonClass') ? !!source.matchNonClass : DEFAULT_INVENTORY_SELL_RULES.matchNonClass,
      rarities: createInventorySellRarityMap(source.rarities)
    };
  }

  function getInventorySlotCoinCost(tabId, slotNumber) {
    const tab = normalizeInventoryTab(tabId);
    const tabMultiplier = tab === 'equipment' ? 1.15 : tab === 'usable' ? 1 : tab === 'cards' ? 1.05 : 0.9;
    const slotIndex = Math.max(0, Math.floor(Number(slotNumber || 1) || 1) - INVENTORY_SECTION_SIZE - 1);
    return Math.max(1, Math.round(INVENTORY_SLOT_COIN_PURCHASE_BASE * tabMultiplier * Math.pow(INVENTORY_SLOT_COIN_PURCHASE_GROWTH, slotIndex)));
  }

  function getInventorySectionCoinCost(tabId, currentSections) {
    const sections = clamp(Math.floor(Number(currentSections || 1) || 1), 1, INVENTORY_MAX_SECTIONS);
    const currentSlots = sections * INVENTORY_SECTION_SIZE;
    let cost = 0;
    for (let offset = 1; offset <= INVENTORY_SECTION_SIZE; offset += 1) {
      cost += getInventorySlotCoinCost(tabId, currentSlots + offset);
    }
    return cost;
  }

  function normalizeInventorySectionCount(value) {
    return clamp(Math.floor(Number(value || 1) || 1), 1, INVENTORY_MAX_SECTIONS);
  }

  function getInventorySectionsFromSlotCount(value) {
    return normalizeInventorySectionCount(Math.ceil(Math.max(0, Number(value || 0)) / INVENTORY_SECTION_SIZE));
  }

  function createInventorySections(value, legacySlots) {
    const source = value && typeof value === 'object' ? value : {};
    const slots = legacySlots && typeof legacySlots === 'object' ? legacySlots : {};
    const equipmentFallback = slots.equipment
      ? getInventorySectionsFromSlotCount(slots.equipment)
      : 1;
    return {
      equipment: normalizeInventorySectionCount(source.equipment || equipmentFallback),
      usable: normalizeInventorySectionCount(source.usable),
      etc: normalizeInventorySectionCount(source.etc),
      cards: normalizeInventorySectionCount(source.cards)
    };
  }

  function createInventorySectionSession(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      equipment: Math.max(0, Math.floor(Number(source.equipment || 0) || 0)),
      usable: Math.max(0, Math.floor(Number(source.usable || 0) || 0)),
      etc: Math.max(0, Math.floor(Number(source.etc || 0) || 0)),
      cards: Math.max(0, Math.floor(Number(source.cards || 0) || 0))
    };
  }

  function createInventorySlotOrder(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      equipment: Array.isArray(source.equipment) ? source.equipment.map(normalizeId) : [],
      usable: Array.isArray(source.usable) ? source.usable.map(normalizeId) : [],
      etc: Array.isArray(source.etc) ? source.etc.map(normalizeId) : [],
      cards: Array.isArray(source.cards) ? source.cards.map(normalizeId) : []
    };
  }

  function countStackedInventorySlots(tabId, source, options) {
    const settings = options || {};
    const tab = normalizeInventoryTab(tabId);
    const counts = source || {};
    const getSlotCount = typeof settings.getStackSlotCount === 'function'
      ? settings.getStackSlotCount
      : (slotTab, id, count) => getStackSlotCount(slotTab, id, count, settings);
    let total = 0;
    for (const rawId in counts) {
      if (!Object.prototype.hasOwnProperty.call(counts, rawId)) continue;
      if (typeof settings.isVisibleItemId === 'function' && !settings.isVisibleItemId(tab, rawId)) continue;
      const count = Math.max(0, Math.floor(Number(counts[rawId]) || 0));
      if (count <= 0) continue;
      total += getSlotCount(tab, rawId, count);
    }
    return total;
  }

  function clonePlainValue(value, options) {
    const settings = options || {};
    const clonePlain = typeof settings.clonePlain === 'function'
      ? settings.clonePlain
      : (input) => JSON.parse(JSON.stringify(input));
    return clonePlain(value);
  }

  function getShopItemDefinition(itemId, options) {
    const settings = options || {};
    const id = normalizeId(itemId);
    if (typeof settings.getShopItemById === 'function') return settings.getShopItemById(id);
    return (settings.shopItems || []).find((item) => normalizeId(item && item.id) === id) || null;
  }

  function createShopEntrySnapshot(entry, vendor, options) {
    const settings = options || {};
    const source = entry && typeof entry === 'object' ? entry : {};
    const player = settings.player || {};
    const kind = normalizeId(source.kind || source.type);
    const vendorId = vendor && vendor.id || '';
    if (kind === 'equipment') {
      const item = getShopItemDefinition(source.itemId, settings);
      const cost = Math.max(0, Number(source.cost || item && item.cost || 0));
      let disabledReason = '';
      if (!item) disabledReason = 'Unavailable.';
      else if (typeof settings.canUseItem === 'function' && !settings.canUseItem(item)) disabledReason = Number(player.level || 1) < Number(item.level || 1) ? `Level ${item.level} required.` : 'Wrong class.';
      else if (Number(player.currency || 0) < cost) disabledReason = 'Not enough coins.';
      else if (typeof settings.canAddEquipmentToInventory === 'function' && !settings.canAddEquipmentToInventory(1)) disabledReason = 'Equipment inventory is full.';
      return {
        id: normalizeId(source.id || source.itemId),
        kind: 'equipment',
        itemId: item && item.id || normalizeId(source.itemId),
        name: item && item.name || normalizeId(source.itemId),
        cost,
        quantity: 1,
        item: item ? clonePlainValue(item, settings) : null,
        vendorId,
        disabled: !!disabledReason,
        disabledReason
      };
    }
    if (kind === 'consumable') {
      const item = typeof settings.getConsumableDefinitionById === 'function'
        ? settings.getConsumableDefinitionById(source.consumableId)
        : null;
      const quantity = Math.max(1, Math.floor(Number(source.quantity || 1) || 1));
      const cost = Math.max(0, Number(source.cost || item && item.cost || 0));
      let disabledReason = '';
      if (!item) disabledReason = 'Unavailable.';
      else if (Number(player.currency || 0) < cost) disabledReason = 'Not enough coins.';
      else if (typeof settings.canAddStackableInventoryItem === 'function' && !settings.canAddStackableInventoryItem('usable', item.id, quantity)) disabledReason = 'Use inventory is full.';
      return {
        id: normalizeId(source.id || source.consumableId),
        kind: 'consumable',
        consumableId: item && item.id || normalizeId(source.consumableId),
        name: item && item.name || normalizeId(source.consumableId),
        cost,
        quantity,
        item: item ? clonePlainValue(item, settings) : null,
        vendorId,
        disabled: !!disabledReason,
        disabledReason
      };
    }
    if (kind === 'bundle') {
      const reward = clonePlainValue(source.reward || {}, settings);
      const cost = Math.max(0, Number(source.cost || 0));
      let disabledReason = '';
      if (Number(player.currency || 0) < cost) disabledReason = 'Not enough coins.';
      else if (typeof settings.getShopRewardInventoryBlockReason === 'function') disabledReason = settings.getShopRewardInventoryBlockReason(reward);
      return {
        id: normalizeId(source.id),
        kind: 'bundle',
        name: source.name || 'Shop Bundle',
        cost,
        quantity: 1,
        summary: source.summary || (typeof settings.formatRewardSummary === 'function' ? settings.formatRewardSummary(reward) : ''),
        reward,
        vendorId,
        disabled: !!disabledReason,
        disabledReason
      };
    }
    return {
      id: normalizeId(source.id),
      kind,
      name: 'Unavailable',
      cost: 0,
      quantity: 1,
      vendorId,
      disabled: true,
      disabledReason: 'Unavailable.'
    };
  }

  function createShopSellItemSnapshot(item, options) {
    const settings = options || {};
    const player = settings.player || {};
    const snapshot = clonePlainValue(item || {}, settings);
    return Object.assign(snapshot, {
      sellValue: typeof settings.getItemSellValue === 'function' ? settings.getItemSellValue(item) : 0,
      powerDelta: typeof settings.getInventoryItemPowerDelta === 'function' ? settings.getInventoryItemPowerDelta(item) : 0,
      classMatch: typeof settings.itemMatchesPlayerClass === 'function' ? settings.itemMatchesPlayerClass(item, player) : true
    });
  }

  function createInventoryProtectedUidSet(equipment) {
    const protectedUids = new Set();
    Object.values(equipment || {}).forEach((item) => {
      const uid = normalizeId(item && item.uid);
      if (uid) protectedUids.add(uid);
    });
    return protectedUids;
  }

  function isInventorySellProtected(item, protectedUids, options) {
    const settings = options || {};
    if (!item || !item.uid || item.locked) return true;
    const itemUid = normalizeId(item.uid);
    if (protectedUids && typeof protectedUids.has === 'function') return protectedUids.has(itemUid);
    return Object.values(settings.equipment || {}).some((equipped) => equipped && normalizeId(equipped.uid) === itemUid);
  }

  function createShopSellInventoryItems(items, mode, options) {
    const settings = options || {};
    const player = settings.player || {};
    const protectedUids = settings.protectedUids;
    const sellMode = normalizeId(mode || 'weak');
    return (Array.isArray(items) ? items : []).filter((item) => {
      const protectedItem = typeof settings.isInventorySellProtected === 'function'
        ? settings.isInventorySellProtected(item, protectedUids)
        : isInventorySellProtected(item, protectedUids, settings);
      if (protectedItem) return false;
      if (sellMode === 'all') return true;
      const classMatch = typeof settings.itemMatchesPlayerClass === 'function'
        ? settings.itemMatchesPlayerClass(item, player)
        : true;
      if (sellMode === 'nonClass') return !classMatch;
      const powerDelta = typeof settings.getInventoryItemPowerDelta === 'function'
        ? settings.getInventoryItemPowerDelta(item)
        : Number(item && item.powerDelta || 0);
      return powerDelta <= 0;
    });
  }

  function createShopSellPreview(items, options) {
    const settings = options || {};
    const source = Array.isArray(items) ? items : [];
    const getItemSellValue = typeof settings.getItemSellValue === 'function'
      ? settings.getItemSellValue
      : () => 0;
    return {
      count: source.length,
      coins: source.reduce((sum, item) => sum + getItemSellValue(item), 0)
    };
  }

  function createShopSellItemSnapshots(items, options) {
    const settings = options || {};
    const source = Array.isArray(items) ? items : [];
    const createSnapshot = typeof settings.createShopSellItemSnapshot === 'function'
      ? settings.createShopSellItemSnapshot
      : (item) => createShopSellItemSnapshot(item, settings);
    return source.map((item) => createSnapshot(item));
  }

  function createShopVendorSnapshot(activeVendorId, vendor, options) {
    const settings = options || {};
    if (!vendor) {
      return {
        activeVendorId: normalizeId(activeVendorId),
        vendor: null,
        entries: [],
        sell: { weak: { count: 0, coins: 0 }, nonClass: { count: 0, coins: 0 }, items: [] }
      };
    }
    const createEntrySnapshot = typeof settings.createShopEntrySnapshot === 'function'
      ? settings.createShopEntrySnapshot
      : (entry, sourceVendor) => createShopEntrySnapshot(entry, sourceVendor, settings);
    const getShopSellPreview = typeof settings.getShopSellPreview === 'function'
      ? settings.getShopSellPreview
      : () => ({ count: 0, coins: 0 });
    const getShopSellItemSnapshots = typeof settings.getShopSellItemSnapshots === 'function'
      ? settings.getShopSellItemSnapshots
      : () => [];
    return {
      activeVendorId: vendor.id,
      vendor: clonePlainValue(vendor, settings),
      entries: (vendor.entries || []).map((entry) => createEntrySnapshot(entry, vendor)),
      sell: {
        weak: getShopSellPreview('weak'),
        nonClass: getShopSellPreview('nonClass'),
        items: getShopSellItemSnapshots()
      }
    };
  }

  function createInventorySellMatch(item, rules, options) {
    const settings = options || {};
    const config = createInventorySellRules(rules || settings.rules);
    const player = settings.player || {};
    const protectedUids = settings.protectedUids;
    const protectedItem = typeof settings.isInventorySellProtected === 'function'
      ? settings.isInventorySellProtected(item, protectedUids)
      : isInventorySellProtected(item, protectedUids, settings);
    if (protectedItem) {
      return { sellable: false, reasons: [] };
    }
    const reasons = [];
    const powerDelta = typeof settings.getInventoryItemPowerDelta === 'function'
      ? settings.getInventoryItemPowerDelta(item)
      : Number(item && item.powerDelta || 0);
    const classMatch = typeof settings.itemMatchesPlayerClass === 'function'
      ? settings.itemMatchesPlayerClass(item, player)
      : true;
    if (config.matchWeak && powerDelta <= 0) reasons.push('weak');
    if (config.matchNonClass && !classMatch) reasons.push('nonClass');
    const rarity = normalizeInventorySellRarity(item && item.rarity);
    if (rarity && config.rarities[rarity]) reasons.push('rarity');
    return { sellable: reasons.length > 0, reasons };
  }

  function createBulkSellInventoryItems(items, rules, options) {
    const settings = options || {};
    const source = Array.isArray(items) ? items : [];
    const config = createInventorySellRules(rules || settings.rules);
    return source.filter((item) => createInventorySellMatch(item, config, settings).sellable);
  }

  function createInventorySellPreview(items, options) {
    const settings = options || {};
    const source = Array.isArray(items) ? items : [];
    const getItemSellValue = typeof settings.getItemSellValue === 'function'
      ? settings.getItemSellValue
      : () => 0;
    return {
      count: source.length,
      coins: source.reduce((sum, item) => sum + getItemSellValue(item), 0)
    };
  }

  const api = {
    INVENTORY_SECTION_SIZE,
    INVENTORY_COUPON_MAX_SECTIONS,
    INVENTORY_MAX_SECTIONS,
    INVENTORY_SLOT_COIN_PURCHASE_BASE,
    INVENTORY_SLOT_COIN_PURCHASE_GROWTH,
    DEFAULT_EQUIPMENT_INVENTORY_SLOTS,
    MIN_EQUIPMENT_INVENTORY_SLOTS,
    MAX_EQUIPMENT_INVENTORY_SLOTS,
    EQUIPMENT_SLOT_COUPON_SIZE,
    BULK_SELL_VALUE_RATE,
    DEFAULT_USABLE_STACK_CAP,
    DEFAULT_ETC_STACK_CAP,
    STACKABLE_SLOT_LIMITS,
    INVENTORY_SELL_RARITIES,
    INVENTORY_SELL_THRESHOLD_MIN,
    INVENTORY_SELL_THRESHOLD_MAX,
    DEFAULT_INVENTORY_SELL_RARITIES,
    DEFAULT_INVENTORY_SELL_RULES,
    INVENTORY_SORT_OPTIONS,
    INVENTORY_SORT_OPTIONS_BY_TAB,
    INVENTORY_TAB_OPTIONS,
    RECOVERY_CONSUMABLE_SORT_ORDER,
    UTILITY_CONSUMABLE_SORT_ORDER,
    PLINKO_BALL_SORT_ORDER,
    SLOT_COUPON_SORT_ORDER,
    PRISM_CONSUMABLE_SORT_ORDER,
    MANUAL_CONSUMABLE_SORT_ORDER,
    UPGRADE_AIDE_MATERIAL_SORT_ORDER,
    STAR_CARD_MATERIAL_SORT_ORDER,
    MATERIAL_VALUE_PRIORITY,
    getMappedSortOrder,
    isRecoveryConsumable,
    getRecoveryConsumableFallbackOrder,
    getConsumableSortType,
    getConsumableSortSubPriority,
    getConsumableValuePriority,
    getMaterialSortType,
    getMaterialSortSubPriority,
    getMaterialValuePriority,
    getMaterialSortMeta,
    normalizeInventoryTab,
    getInventoryTabLabel,
    getInventorySortOptions,
    normalizeInventorySort,
    getStackableItemCap,
    getStackSlotCount,
    createStackedSlotEntries,
    createConsumableState,
    ensureAdminConsoleItem,
    createMaterialState,
    createInventorySlots,
    normalizeInventorySellRarity,
    normalizeInventorySellThreshold,
    createInventorySellRarityMap,
    createInventorySellRules,
    getInventorySlotCoinCost,
    getInventorySectionCoinCost,
    normalizeInventorySectionCount,
    getInventorySectionsFromSlotCount,
    createInventorySections,
    createInventorySectionSession,
    createInventorySlotOrder,
    countStackedInventorySlots,
    createShopEntrySnapshot,
    createShopSellItemSnapshot,
    createInventoryProtectedUidSet,
    isInventorySellProtected,
    createShopSellInventoryItems,
    createShopSellPreview,
    createShopSellItemSnapshots,
    createShopVendorSnapshot,
    createInventorySellMatch,
    createBulkSellInventoryItems,
    createInventorySellPreview
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.inventory = Object.assign({}, modules.inventory || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
