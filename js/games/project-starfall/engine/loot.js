(function initProjectStarfallEngineLoot(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const rectOverlapsBox = CoreMath.rectOverlapsBox || function rectOverlapsBoxFallback(rect, x, y, w, h) {
    return !!rect && x < rect.x + rect.w && x + w > rect.x && y < rect.y + rect.h && y + h > rect.y;
  };
  const rectsOverlap = CoreMath.rectsOverlap || function rectsOverlapFallback(a, b) {
    return !!(a && b &&
      a.x < b.x + b.w &&
      a.x + a.w > b.x &&
      a.y < b.y + b.h &&
      a.y + a.h > b.y);
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const POTION_DROP_TIERS = Object.freeze([
    Object.freeze({ minScore: 0, rarity: 'Common', hp: 'minor_health_potion', mp: 'minor_resource_tonic', hybrid: 'camp_ration' }),
    Object.freeze({ minScore: 31, rarity: 'Uncommon', hp: 'standard_health_potion', mp: 'standard_resource_tonic', hybrid: 'field_ration' }),
    Object.freeze({ minScore: 59, rarity: 'Rare', hp: 'greater_health_potion', mp: 'greater_resource_tonic', hybrid: 'expedition_ration' }),
    Object.freeze({ minScore: 87, rarity: 'Epic', hp: 'superior_health_potion', mp: 'superior_resource_tonic', hybrid: 'hero_ration' })
  ]);

  function getEnemyData(enemy) {
    return enemy && enemy.data ? enemy.data : enemy;
  }

  function defaultIsEliteMonster(enemy) {
    const id = String(enemy && (enemy.id || enemy.enemyId) || '');
    return !!(enemy && enemy.elite) || id === 'crackedMimic' || id === 'emberjawGolem';
  }

  function defaultIsBossMonster(enemy) {
    const enemyData = getEnemyData(enemy);
    return !!(enemyData && enemyData.behavior === 'boss');
  }

  function getMonsterDropPoolConfig(enemyData) {
    const config = enemyData && enemyData.dropPool;
    return config && typeof config === 'object' ? config : null;
  }

  function normalizeDropWeight(value, fallback) {
    return Math.max(1, Math.round(Number(value == null ? fallback : value) || fallback || 1));
  }

  function normalizeDropChance(value, fallback) {
    const chance = Number(value == null ? fallback : value);
    return Number.isFinite(chance) ? clamp(chance, 0, 1) : clamp(Number(fallback || 0), 0, 1);
  }

  function getLootData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getDropEconomyValue(path, fallback, options) {
    const data = getLootData(options);
    const parts = String(path || '').split('.').filter(Boolean);
    let value = data.DROP_ECONOMY || {};
    for (const part of parts) {
      if (!value || typeof value !== 'object' || !Object.prototype.hasOwnProperty.call(value, part)) return fallback;
      value = value[part];
    }
    return value == null ? fallback : value;
  }

  function getDropEconomyNumber(path, fallback, options) {
    const value = Number(getDropEconomyValue(path, fallback, options));
    return Number.isFinite(value) ? value : fallback;
  }

  function weightedItem(items) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return null;
    const total = options.reduce((sum, item) => sum + Math.max(1, Number(item.weight || 1)), 0);
    let roll = Math.random() * total;
    for (let index = 0; index < options.length; index += 1) {
      const item = options[index];
      roll -= Math.max(1, Number(item.weight || 1));
      if (roll <= 0) return item;
    }
    return options[options.length - 1];
  }

  function randomInt(min, max) {
    const low = Math.ceil(Number(min) || 0);
    const high = Math.floor(Number(max) || low);
    if (high <= low) return low;
    return low + Math.floor(Math.random() * (high - low + 1));
  }

  function getCoinStackAssetId(amount) {
    const count = Math.max(1, Math.floor(Number(amount) || 1));
    if (count >= 1000) return 'coins_huge';
    if (count >= 200) return 'coins_large';
    if (count >= 50) return 'coins_medium';
    return 'coins_small';
  }

  function describeLootItem(item) {
    if (!item) return 'loot';
    if (item.kind === 'currency') return `${Math.max(1, Math.floor(Number(item.amount || item.quantity) || 1))} coins`;
    const quantity = Math.max(1, Math.floor(Number(item.quantity || 1) || 1));
    const stack = quantity > 1 ? `${quantity}x ` : '';
    const rarity = item.rarity ? `${item.rarity} ` : '';
    return `${stack}${rarity}${item.name || 'loot'}`;
  }

  function makeMaterialDrop(materialId, quantity, config, options) {
    const settings = options || {};
    const definitions = settings.materialDropDefinitions || {};
    const createReward = typeof settings.createLootRewardItem === 'function'
      ? settings.createLootRewardItem
      : () => null;
    const definition = definitions[materialId];
    const dropSettings = config || {};
    if (!definition) return null;
    return createReward({
      id: definition.id,
      kind: 'material',
      materialId: definition.materialId,
      name: definition.name,
      icon: definition.icon,
      rarity: dropSettings.rarity || definition.rarity,
      quantity,
      source: dropSettings.source || 'Monster drop'
    });
  }

  function makeCardDrop(cardId, config, options) {
    const settings = options || {};
    const getCard = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : () => null;
    const createReward = typeof settings.createLootRewardItem === 'function'
      ? settings.createLootRewardItem
      : () => null;
    const definition = getCard(cardId);
    const dropSettings = config || {};
    if (!definition) return null;
    return createReward({
      id: definition.id,
      kind: 'card',
      cardId: definition.id,
      name: definition.name,
      icon: definition.icon || 'CD',
      rarity: dropSettings.rarity || definition.rarity || 'Common',
      quantity: 1,
      source: dropSettings.source || 'Monster drop'
    });
  }

  function makeConsumableDrop(consumableId, quantity, config, options) {
    const settings = options || {};
    const getConsumable = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : () => null;
    const createReward = typeof settings.createLootRewardItem === 'function'
      ? settings.createLootRewardItem
      : () => null;
    const base = getConsumable(consumableId);
    const dropSettings = config || {};
    if (!base) return null;
    return createReward({
      id: base.id,
      kind: 'consumable',
      consumableId: base.id,
      name: base.name,
      icon: base.icon || 'IT',
      rarity: dropSettings.rarity || 'Common',
      quantity,
      source: dropSettings.source || 'Monster drop'
    });
  }

  function makeCurrencyDrop(enemy, quantity, options) {
    const settings = options || {};
    const createReward = typeof settings.createLootRewardItem === 'function'
      ? settings.createLootRewardItem
      : () => null;
    const getCoinAssetId = typeof settings.getCoinStackAssetId === 'function'
      ? settings.getCoinStackAssetId
      : getCoinStackAssetId;
    const level = Math.max(1, Number(enemy && enemy.level) || 1);
    const amount = Math.max(1, Math.round(Number(quantity) || (10 + level * 3 + (enemy && enemy.elite ? 60 : 0))));
    return createReward({
      id: 'coins',
      kind: 'currency',
      assetId: getCoinAssetId(amount),
      name: 'Coin Pouch',
      icon: '$',
      rarity: enemy && enemy.elite ? 'Uncommon' : 'Common',
      quantity: amount,
      amount,
      source: 'Monster drop'
    });
  }

  function getLootStackQuantity(item) {
    return Math.max(1, Math.floor(Number(item && (item.quantity || item.amount) || 1) || 1));
  }

  function getLootStackAmount(item) {
    return Math.max(1, Math.floor(Number(item && (item.amount || item.quantity) || 1) || 1));
  }

  function isStackableLootItem(item) {
    const kind = normalizeId(item && item.kind);
    return kind === 'material' || kind === 'consumable' || kind === 'currency';
  }

  function isPriorityStackableLootItem(item) {
    const kind = normalizeId(item && item.kind);
    return kind === 'material' || kind === 'consumable';
  }

  function getLootStackKey(item) {
    const kind = normalizeId(item && item.kind);
    if (kind === 'material') return `material:${normalizeId(item.materialId || item.id)}:${item.rarity || ''}`;
    if (kind === 'consumable') return `consumable:${normalizeId(item.consumableId || item.id)}:${item.rarity || ''}`;
    if (kind === 'currency') return `currency:${normalizeId(item.id || 'coins')}:${item.rarity || ''}`;
    return '';
  }

  function mergeStackableLootItem(target, item, options) {
    const settings = options || {};
    if (!target || !item) return target;
    const getQuantity = typeof settings.getLootStackQuantity === 'function'
      ? settings.getLootStackQuantity
      : getLootStackQuantity;
    const getAmount = typeof settings.getLootStackAmount === 'function'
      ? settings.getLootStackAmount
      : getLootStackAmount;
    target.quantity = getQuantity(target) + getQuantity(item);
    target.amount = getAmount(target) + getAmount(item);
    if (normalizeId(target.kind) === 'currency') {
      const getCoinAssetId = typeof settings.getCoinStackAssetId === 'function'
        ? settings.getCoinStackAssetId
        : null;
      const getAsset = typeof settings.getItemAsset === 'function'
        ? settings.getItemAsset
        : null;
      target.quantity = target.amount;
      if (getCoinAssetId) target.assetId = getCoinAssetId(target.amount);
      if (getAsset) target.asset = getAsset(target);
    }
    return target;
  }

  function prioritizeMonsterLootBatch(equipment, stackableItems, options) {
    const settings = options || {};
    const cap = Math.max(0, Math.floor(Number(settings.groundLootDropCap == null ? 50 : settings.groundLootDropCap) || 0));
    const isPriority = typeof settings.isPriorityStackableLootItem === 'function'
      ? settings.isPriorityStackableLootItem
      : isPriorityStackableLootItem;
    const gearDrops = (equipment || []).filter(Boolean);
    const priorityStacks = [];
    const currencyStacks = [];
    (stackableItems || []).filter(Boolean).forEach((item) => {
      if (isPriority(item)) priorityStacks.push(item);
      else currencyStacks.push(item);
    });
    const prioritized = priorityStacks.slice(0, cap);
    const remainingAfterPriority = Math.max(0, cap - prioritized.length);
    const selectedGear = gearDrops.slice(0, remainingAfterPriority);
    const remainingAfterGear = Math.max(0, remainingAfterPriority - selectedGear.length);
    const selectedCurrency = currencyStacks.slice(0, remainingAfterGear);
    return prioritized.concat(selectedGear, selectedCurrency);
  }

  function enforceLootDropCap(source, options) {
    const settings = options || {};
    const cap = Math.max(0, Math.floor(Number(settings.groundLootDropCap == null ? 50 : settings.groundLootDropCap) || 0));
    const sourceIsArray = Array.isArray(source);
    let drops = sourceIsArray ? source : [];
    for (let index = 0; index < drops.length; index += 1) {
      if (!drops[index]) {
        drops = drops.filter(Boolean);
        break;
      }
    }
    if (drops.length <= cap) {
      return {
        drops,
        removed: 0,
        changed: !sourceIsArray || drops !== source
      };
    }
    if (drops.length === cap + 1) {
      let oldestIndex = 0;
      for (let index = 1; index < drops.length; index += 1) {
        const drop = drops[index];
        const oldest = drops[oldestIndex];
        const dropCreatedAt = Number(drop && drop.createdAt || 0);
        const oldestCreatedAt = Number(oldest && oldest.createdAt || 0);
        if (dropCreatedAt < oldestCreatedAt || dropCreatedAt === oldestCreatedAt && index < oldestIndex) {
          oldestIndex = index;
        }
      }
      drops.splice(oldestIndex, 1);
      return {
        drops,
        removed: 1,
        changed: true
      };
    }
    const keep = new Set(drops
      .map((drop, index) => ({ drop, index }))
      .sort((a, b) => Number(b.drop.createdAt || 0) - Number(a.drop.createdAt || 0) || b.index - a.index)
      .slice(0, cap)
      .map((entry) => entry.drop));
    const cappedDrops = drops.filter((drop) => keep.has(drop));
    return {
      drops: cappedDrops,
      removed: drops.length - cappedDrops.length,
      changed: true
    };
  }

  function createLootDropCacheContext(state, channelId) {
    if (!state) return null;
    const source = Array.isArray(state.lootDrops) ? state.lootDrops : (state.lootDrops = []);
    return {
      source,
      length: source.length,
      mapId: state.mapId,
      channelId
    };
  }

  function lootDropCacheMatches(cache, context) {
    return !!(cache &&
      context &&
      cache.source === context.source &&
      cache.length === context.length &&
      cache.mapId === context.mapId &&
      cache.channelId === context.channelId);
  }

  function createLootDropUidMapCache(context, drops, options) {
    if (!context) return null;
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const list = Array.isArray(drops) ? drops : [];
    const result = new Map();
    for (let index = 0; index < list.length; index += 1) {
      const drop = list[index];
      const uid = normalize(drop && drop.uid);
      if (uid && !result.has(uid)) result.set(uid, drop);
    }
    return {
      source: context.source,
      length: context.length,
      mapId: context.mapId,
      channelId: context.channelId,
      result
    };
  }

  function createCurrentMapLootDropCaches(context, options) {
    if (!context) return null;
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const normalizeChannel = typeof settings.normalizeMapChannelId === 'function'
      ? settings.normalizeMapChannelId
      : function normalizeMapChannelIdFallback(channelId) {
          return normalize(channelId);
    };
    const source = Array.isArray(context.source) ? context.source : [];
    const result = [];
    for (let index = 0; index < source.length; index += 1) {
      const drop = source[index];
      if (!drop || drop.mapId !== context.mapId) continue;
      if (normalizeChannel(drop.channelId) === context.channelId) {
        result.push(drop);
      }
    }
    return {
      listCache: {
        source,
        length: context.length,
        mapId: context.mapId,
        channelId: context.channelId,
        result
      },
      uidMapCache: createLootDropUidMapCache(context, result, { normalizeId: normalize })
    };
  }

  function removeLootDropByUid(source, uid, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const id = normalize(uid);
    const drops = Array.isArray(source) ? source : null;
    if (!id || !drops || !drops.length) {
      return {
        drops,
        removed: false,
        removedCount: 0
      };
    }
    let writeIndex = 0;
    let removedCount = 0;
    for (let index = 0; index < drops.length; index += 1) {
      const drop = drops[index];
      if (drop && drop.uid === id) {
        removedCount += 1;
        continue;
      }
      drops[writeIndex] = drop;
      writeIndex += 1;
    }
    if (removedCount > 0) drops.length = writeIndex;
    return {
      drops,
      removed: removedCount > 0,
      removedCount
    };
  }

  function createLootInventoryAdmissionContext() {
    return {
      capacities: {},
      usedSlots: {},
      stackableAddCapacities: {
        usable: new Map(),
        etc: new Map()
      }
    };
  }

  function normalizeLootInventoryTab(tabId, options) {
    const settings = options || {};
    const normalizeTab = typeof settings.normalizeInventoryTab === 'function'
      ? settings.normalizeInventoryTab
      : function normalizeInventoryTabFallback(value) {
          const tab = normalizeId(value);
          return tab || 'equipment';
        };
    return normalizeTab(tabId);
  }

  function getLootAdmissionContextCapacity(context, tabId, options) {
    const settings = options || {};
    const tab = normalizeLootInventoryTab(tabId, settings);
    const getCapacity = typeof settings.getInventoryCapacity === 'function'
      ? settings.getInventoryCapacity
      : function getInventoryCapacityFallback() {
          return 0;
        };
    if (!context || typeof context !== 'object') return Math.max(0, Math.floor(Number(getCapacity(tab) || 0) || 0));
    context.capacities = context.capacities && typeof context.capacities === 'object' ? context.capacities : {};
    if (!Object.prototype.hasOwnProperty.call(context.capacities, tab)) {
      context.capacities[tab] = getCapacity(tab);
    }
    return Math.max(0, Math.floor(Number(context.capacities[tab] || 0) || 0));
  }

  function getLootAdmissionContextUsedSlots(context, tabId, options) {
    const settings = options || {};
    const tab = normalizeLootInventoryTab(tabId, settings);
    const getUsedSlots = typeof settings.getInventoryUsedSlots === 'function'
      ? settings.getInventoryUsedSlots
      : function getInventoryUsedSlotsFallback() {
          return 0;
        };
    if (!context || typeof context !== 'object') return Math.max(0, Math.floor(Number(getUsedSlots(tab) || 0) || 0));
    context.usedSlots = context.usedSlots && typeof context.usedSlots === 'object' ? context.usedSlots : {};
    if (!Object.prototype.hasOwnProperty.call(context.usedSlots, tab)) {
      context.usedSlots[tab] = getUsedSlots(tab);
    }
    return Math.max(0, Math.floor(Number(context.usedSlots[tab] || 0) || 0));
  }

  function getStackableInventoryAddCapacityForAdmission(tabId, itemId, context, options) {
    const settings = options || {};
    const tab = normalizeLootInventoryTab(tabId, settings);
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const id = normalize(itemId);
    if (tab === 'equipment' || !id) return 0;
    const getDirectCapacity = typeof settings.getStackableInventoryAddCapacity === 'function'
      ? settings.getStackableInventoryAddCapacity
      : function getStackableInventoryAddCapacityFallback() {
          return 0;
        };
    if (!context || typeof context !== 'object') return getDirectCapacity(tab, id);
    context.stackableAddCapacities = context.stackableAddCapacities && typeof context.stackableAddCapacities === 'object'
      ? context.stackableAddCapacities
      : {};
    let cache = context.stackableAddCapacities[tab];
    if (!cache || typeof cache.get !== 'function' || typeof cache.set !== 'function') {
      cache = new Map();
      context.stackableAddCapacities[tab] = cache;
    }
    if (cache.has(id)) return cache.get(id);
    const getUsedSlots = typeof settings.getLootAdmissionContextUsedSlots === 'function'
      ? settings.getLootAdmissionContextUsedSlots
      : (targetContext, targetTab) => getLootAdmissionContextUsedSlots(targetContext, targetTab, settings);
    const getCapacity = typeof settings.getLootAdmissionContextCapacity === 'function'
      ? settings.getLootAdmissionContextCapacity
      : (targetContext, targetTab) => getLootAdmissionContextCapacity(targetContext, targetTab, settings);
    const getCapacityFromUsed = typeof settings.getStackableInventoryAddCapacityFromUsed === 'function'
      ? settings.getStackableInventoryAddCapacityFromUsed
      : function getStackableInventoryAddCapacityFromUsedFallback() {
          return 0;
        };
    const usedSlots = getUsedSlots(context, tab);
    const capacity = getCapacity(context, tab);
    const addCapacity = getCapacityFromUsed(tab, id, usedSlots, capacity);
    cache.set(id, addCapacity);
    return addCapacity;
  }

  function createLootDropAdmissionResult(values) {
    const source = values || {};
    return {
      allowed: !!source.allowed,
      kind: source.kind || '',
      quantity: Math.max(0, Math.floor(Number(source.quantity || 0) || 0)),
      materialId: source.materialId || '',
      consumableId: source.consumableId || '',
      cardId: source.cardId || '',
      autoSell: !!source.autoSell,
      fullMessage: source.fullMessage || ''
    };
  }

  function createLootDropInventoryAdmission(drop, options) {
    const settings = options || {};
    const admissionContext = settings.inventoryAdmissionContext || settings.admissionContext || null;
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const canAddStackable = typeof settings.canAddStackableInventoryItem === 'function'
      ? settings.canAddStackableInventoryItem
      : function canAddStackableInventoryItemFallback() {
          return false;
        };
    const canAddCard = typeof settings.canAddCardToInventory === 'function'
      ? settings.canAddCardToInventory
      : function canAddCardToInventoryFallback() {
          return false;
        };
    const getCard = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : function getCardDefinitionFallback() {
          return null;
        };
    const shouldAutoSell = typeof settings.shouldAutoSellLootItem === 'function'
      ? settings.shouldAutoSellLootItem
      : function shouldAutoSellLootItemFallback() {
          return false;
        };
    const getUsedSlots = typeof settings.getLootAdmissionContextUsedSlots === 'function'
      ? settings.getLootAdmissionContextUsedSlots
      : (targetContext, targetTab) => getLootAdmissionContextUsedSlots(targetContext, targetTab, settings);
    const getCapacity = typeof settings.getLootAdmissionContextCapacity === 'function'
      ? settings.getLootAdmissionContextCapacity
      : (targetContext, targetTab) => getLootAdmissionContextCapacity(targetContext, targetTab, settings);
    const getEquipmentCount = typeof settings.getEquipmentInventoryCount === 'function'
      ? settings.getEquipmentInventoryCount
      : function getEquipmentInventoryCountFallback() {
          return 0;
        };
    const getInventoryCapacity = typeof settings.getInventoryCapacity === 'function'
      ? settings.getInventoryCapacity
      : function getInventoryCapacityFallback() {
          return 0;
        };
    if (!drop || !drop.item) return createLootDropAdmissionResult();
    const item = drop.item;
    const rawKind = normalize(item.kind || 'equipment');
    const kind = rawKind === 'material' || rawKind === 'consumable' || rawKind === 'currency' || rawKind === 'card' ? rawKind : 'equipment';
    const quantity = Math.max(1, Math.floor(Number(item.quantity || item.amount || 1) || 1));
    const materialId = kind === 'material' ? normalize(item.materialId) : '';
    const consumableId = kind === 'consumable' ? normalize(item.consumableId || item.id) : '';
    const cardId = kind === 'card' ? normalize(item.cardId || item.id) : '';
    if (kind === 'currency') return createLootDropAdmissionResult({ allowed: true, kind, quantity });
    if (kind === 'material') {
      if (!materialId) return createLootDropAdmissionResult({ kind, quantity });
      return createLootDropAdmissionResult({
        allowed: canAddStackable('etc', materialId, quantity, { admissionContext }),
        kind,
        quantity,
        materialId,
        fullMessage: 'Etc inventory is full.'
      });
    }
    if (kind === 'consumable') {
      if (!consumableId) return createLootDropAdmissionResult({ kind, quantity });
      return createLootDropAdmissionResult({
        allowed: canAddStackable('usable', consumableId, quantity, { admissionContext }),
        kind,
        quantity,
        consumableId,
        fullMessage: 'Usable inventory is full.'
      });
    }
    if (kind === 'card') {
      return createLootDropAdmissionResult({
        allowed: !!(cardId && getCard(cardId) && canAddCard(cardId)),
        kind,
        quantity,
        cardId,
        fullMessage: 'Card is unavailable.'
      });
    }
    const autoSell = shouldAutoSell(drop, item);
    const equipmentHasRoom = admissionContext
      ? getUsedSlots(admissionContext, 'equipment') < getCapacity(admissionContext, 'equipment')
      : getEquipmentCount() < getInventoryCapacity('equipment');
    return createLootDropAdmissionResult({
      allowed: autoSell || equipmentHasRoom,
      kind,
      quantity,
      autoSell,
      fullMessage: 'Equipment inventory is full.'
    });
  }

  function formatLootInteger(value, options) {
    const settings = options || {};
    if (typeof settings.formatIntegerWithCommas === 'function') return settings.formatIntegerWithCommas(value);
    const amount = Math.floor(Number(value || 0) || 0);
    return String(amount).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function getLootMaterialDefinition(materialId, options) {
    const settings = options || {};
    if (typeof settings.getMaterialDefinition === 'function') return settings.getMaterialDefinition(materialId) || null;
    const definitions = settings.materialDropDefinitions || {};
    return definitions && definitions[materialId] || null;
  }

  function createLootRewardToastMessage(item, admission, options) {
    const settings = options || {};
    const kind = admission && admission.kind || item && item.kind || '';
    if (kind === 'equipment' && settings.autoSell) {
      return `Sold ${item && item.name || 'gear'} +${formatLootInteger(settings.coins, settings)} coins`;
    }
    if (kind === 'currency') {
      const amount = Math.max(1, Math.floor(Number(item && (item.amount || item.quantity) || admission && admission.quantity) || 1));
      return `+${formatLootInteger(amount, settings)} coins`;
    }
    const quantity = Math.max(1, Math.floor(Number(admission && admission.quantity || item && item.quantity || 1) || 1));
    if (kind === 'material') {
      const material = getLootMaterialDefinition(admission && admission.materialId, settings) || {};
      return `+${formatLootInteger(quantity, settings)} ${material.name || item && item.name || 'material'}`;
    }
    if (kind === 'consumable') {
      const getConsumable = typeof settings.getConsumableDefinition === 'function'
        ? settings.getConsumableDefinition
        : function getConsumableDefinitionFallback() {
            return null;
          };
      const consumable = getConsumable(admission && admission.consumableId);
      return `+${formatLootInteger(quantity, settings)} ${consumable && consumable.name || item && item.name || 'item'}`;
    }
    if (kind === 'card') {
      const getCard = typeof settings.getCardDefinition === 'function'
        ? settings.getCardDefinition
        : function getCardDefinitionFallback() {
            return null;
          };
      const card = getCard(admission && admission.cardId || item && item.cardId);
      return `+${card && card.name || item && item.name || 'Card'} card`;
    }
    return item && item.name || 'Loot acquired';
  }

  function canPetLootDrop(drop, filters, options) {
    const settings = options || {};
    const isCollectible = typeof settings.isLootCollectible === 'function' ? settings.isLootCollectible : isLootCollectible;
    if (!isCollectible(drop) || !drop.item) return false;
    if (drop.playerDropped || String(drop.item.source || '') === 'Player drop') return false;
    const activeFilters = filters && typeof filters === 'object' ? filters : {};
    const item = drop.item;
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const kind = normalize(item.kind || 'equipment');
    if (kind === 'currency') return !!activeFilters.currency;
    if (kind === 'consumable') return !!activeFilters.consumables;
    if (kind === 'material') return !!activeFilters.materials;
    if (kind === 'card') return !!activeFilters.cards;
    if (!activeFilters.equipment) return false;
    const rarityOrder = Array.isArray(settings.rarityOrder) && settings.rarityOrder.length
      ? settings.rarityOrder
      : ['Common', 'Uncommon', 'Rare', 'Epic', 'Relic'];
    const itemRarityIndex = Math.max(0, rarityOrder.indexOf(item.rarity || 'Common'));
    const minimumIndex = Math.max(0, rarityOrder.indexOf(activeFilters.minEquipmentRarity || 'Common'));
    return itemRarityIndex >= minimumIndex;
  }

  function createPetLootClusterCounts(candidates, options) {
    const settings = options || {};
    const clusterRadius = Math.max(0, Number(settings.clusterRadius == null ? 220 : settings.clusterRadius) || 0);
    const counts = new Map();
    const groups = new Map();
    (candidates || []).forEach((entry) => {
      const key = entry && entry.platformKey;
      if (!key) {
        counts.set(entry, 0);
        return;
      }
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(entry);
    });
    groups.forEach((sorted) => {
      sorted.sort((a, b) => Number(a.dropX || 0) - Number(b.dropX || 0));
      let start = 0;
      let end = 0;
      for (let index = 0; index < sorted.length; index += 1) {
        const entry = sorted[index];
        const x = Number(entry.dropX || 0);
        while (start < sorted.length && x - Number(sorted[start].dropX || 0) > clusterRadius) start += 1;
        if (end < index) end = index;
        while (end + 1 < sorted.length && Number(sorted[end + 1].dropX || 0) - x <= clusterRadius) end += 1;
        counts.set(entry, Math.max(0, end - start));
      }
    });
    return counts;
  }

  function scorePetLootCandidate(entry, candidates, context, clusterCounts, options) {
    const settings = options || {};
    const clusterRadius = Math.max(0, Number(settings.clusterRadius == null ? 220 : settings.clusterRadius) || 0);
    const clusterBonusValue = Math.max(0, Number(settings.clusterBonus == null ? 42 : settings.clusterBonus) || 0);
    const clusterBonusCap = Math.max(0, Number(settings.clusterBonusCap == null ? 150 : settings.clusterBonusCap) || 0);
    const platformSwitchPenalty = Math.max(0, Number(settings.platformSwitchPenalty == null ? 260 : settings.platformSwitchPenalty) || 0);
    const verticalWeight = Math.max(0, Number(settings.verticalWeight == null ? 0.9 : settings.verticalWeight) || 0);
    const playerDistanceWeight = Math.max(0, Number(settings.playerDistanceWeight == null ? 0.08 : settings.playerDistanceWeight) || 0);
    const scoringContext = context || {};
    const currentPlatformKey = scoringContext.currentPlatformKey || '';
    const platformSwitch = currentPlatformKey && entry && entry.platformKey && entry.platformKey !== currentPlatformKey
      ? platformSwitchPenalty
      : 0;
    const verticalCost = Math.abs(Number(entry && (entry.targetY || entry.dropY) || 0) - Number(scoringContext.petY || 0)) * verticalWeight;
    const horizontalCost = Math.abs(Number(entry && entry.dropX || 0) - Number(scoringContext.petX || 0));
    const nearbySamePlatform = clusterCounts && clusterCounts.has(entry)
      ? Number(clusterCounts.get(entry) || 0)
      : (candidates || []).filter((candidate) =>
        candidate !== entry &&
        candidate &&
        entry &&
        candidate.platformKey &&
        candidate.platformKey === entry.platformKey &&
        Math.abs(Number(candidate.dropX || 0) - Number(entry.dropX || 0)) <= clusterRadius).length;
    const clusterBonus = Math.min(clusterBonusCap, nearbySamePlatform * clusterBonusValue);
    return horizontalCost + verticalCost + platformSwitch + Number(entry && entry.playerDistance || 0) * playerDistanceWeight - clusterBonus;
  }

  function selectPetLootTargetCandidate(candidates, options) {
    const settings = options || {};
    const list = Array.isArray(candidates) ? candidates : [];
    const routeKey = String(settings.routeKey || '');
    let useRouteCandidates = false;
    if (routeKey) {
      for (let index = 0; index < list.length; index += 1) {
        if (list[index] && list[index].platformKey === routeKey) {
          useRouteCandidates = true;
          break;
        }
      }
    }
    const context = settings.context || {};
    const clusterCounts = settings.clusterCounts || createPetLootClusterCounts(list, settings);
    const scoreCandidate = typeof settings.scorePetLootCandidate === 'function'
      ? settings.scorePetLootCandidate
      : (entry, targetCandidates, targetContext, targetClusterCounts) => scorePetLootCandidate(entry, targetCandidates, targetContext, targetClusterCounts, settings);
    let match = null;
    let matchScore = Infinity;
    for (let index = 0; index < list.length; index += 1) {
      const entry = list[index];
      if (!entry) continue;
      if (useRouteCandidates && entry.platformKey !== routeKey) continue;
      const score = scoreCandidate(entry, list, context, clusterCounts);
      if (!match ||
        score < matchScore ||
        score === matchScore && Number(entry.playerDistance || 0) < Number(match.playerDistance || 0) ||
        score === matchScore && Number(entry.playerDistance || 0) === Number(match.playerDistance || 0) && Number(entry.drop && entry.drop.createdAt || 0) < Number(match.drop && match.drop.createdAt || 0)) {
        match = entry;
        matchScore = score;
      }
    }
    return {
      entry: match,
      score: matchScore,
      useRouteCandidates
    };
  }

  function createPetLootTouchBox(runtime, options) {
    const settings = options || {};
    const halfWidth = Math.max(0, Number(settings.halfWidth == null ? 34 : settings.halfWidth) || 0);
    const topOffset = Math.max(0, Number(settings.topOffset == null ? 68 : settings.topOffset) || 0);
    const width = Math.max(0, Number(settings.width == null ? 72 : settings.width) || 0);
    const height = Math.max(0, Number(settings.height == null ? 82 : settings.height) || 0);
    return {
      x: Number(runtime && runtime.x || 0) - halfWidth,
      y: Number(runtime && runtime.y || 0) - topOffset,
      w: width,
      h: height
    };
  }

  function petLootTouchBoxOverlapsDrop(touchBox, drop, options) {
    if (!touchBox || !drop) return false;
    const settings = options || {};
    const overlaps = typeof settings.rectsOverlap === 'function' ? settings.rectsOverlap : rectsOverlap;
    const drawBox = typeof settings.createLootDrawBox === 'function'
      ? settings.createLootDrawBox(drop)
      : createLootDrawBox(drop, settings);
    return overlaps(touchBox, drawBox);
  }

  function lootNowSeconds(options) {
    const settings = options || {};
    return typeof settings.nowSeconds === 'function' ? settings.nowSeconds() : Date.now() / 1000;
  }

  function lootItemShouldShowTierAura(item, options) {
    const settings = options || {};
    if (typeof settings.itemShouldShowTierAura === 'function') return settings.itemShouldShowTierAura(item);
    const kind = normalizeId(item && item.kind || '');
    return kind === 'equipment' || kind === 'card' || (!kind && !!(item && item.slot));
  }

  function isLootCollectible(drop) {
    return !!(drop && !drop.airborne && (Number(drop.settledAt || 0) > 0 || Number(drop.createdAt || 0) > 0));
  }

  function createLootDrawBox(drop, options) {
    const settings = options || {};
    const hover = drop.airborne ? 0 : Math.sin(lootNowSeconds(settings) * 3.2 + Number(drop.seed || 0)) * 5;
    const size = lootItemShouldShowTierAura(drop && drop.item, settings) ? 44 : 58;
    const lift = lootItemShouldShowTierAura(drop && drop.item, settings) ? 36 : 48;
    return {
      x: drop.x - size / 2,
      y: drop.y - lift + hover,
      w: size,
      h: size
    };
  }

  function lootDrawBoxOverlapsBox(drop, box, hoverPhase, options) {
    if (!drop || !box) return false;
    const settings = options || {};
    const overlapsBox = typeof settings.rectOverlapsBox === 'function' ? settings.rectOverlapsBox : rectOverlapsBox;
    const size = lootItemShouldShowTierAura(drop && drop.item, settings) ? 44 : 58;
    const lift = lootItemShouldShowTierAura(drop && drop.item, settings) ? 36 : 48;
    const phase = Number.isFinite(Number(hoverPhase)) ? Number(hoverPhase) : lootNowSeconds(settings) * 3.2;
    const hover = drop.airborne ? 0 : Math.sin(phase + Number(drop.seed || 0)) * 5;
    return overlapsBox(
      box,
      Number(drop.x || 0) - size / 2,
      Number(drop.y || 0) - lift + hover,
      size,
      size
    );
  }

  function lootDropCouldOverlapBox(drop, box, options) {
    if (!drop || !box) return false;
    const settings = options || {};
    const broadHalfWidth = Math.max(0, Number(settings.broadHalfWidth == null ? 32 : settings.broadHalfWidth) || 0);
    const broadTop = Math.max(0, Number(settings.broadTop == null ? 53 : settings.broadTop) || 0);
    const broadBottom = Math.max(0, Number(settings.broadBottom == null ? 15 : settings.broadBottom) || 0);
    const dropX = Number(drop.x || 0);
    const dropY = Number(drop.y || 0);
    return dropX >= Number(box.x || 0) - broadHalfWidth &&
      dropX <= Number(box.x || 0) + Number(box.w || 0) + broadHalfWidth &&
      dropY >= Number(box.y || 0) - broadBottom &&
      dropY <= Number(box.y || 0) + Number(box.h || 0) + broadTop;
  }

  function getMonsterMaterialDropQuantity(materialId, level, enemyData, dropEntry, options) {
    const settings = options || {};
    const entry = dropEntry || {};
    const randomInteger = typeof settings.randomInt === 'function'
      ? settings.randomInt
      : randomInt;
    if (entry.minQuantity != null || entry.maxQuantity != null) {
      const min = Math.max(1, Math.floor(Number(entry.minQuantity || 1) || 1));
      const max = Math.max(min, Math.floor(Number(entry.maxQuantity || min) || min));
      return randomInteger(min, max);
    }
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const normalizedLevel = Math.max(1, Math.floor(Number(level) || 1));
    const id = normalize(materialId);
    const drops = Array.isArray(enemyData && enemyData.drops) ? enemyData.drops : [];
    const hasDrop = (label) => drops.includes(label);
    if (id === 'upgradeDust') {
      return randomInteger(1 + Math.floor(normalizedLevel / 12), 2 + Math.floor(normalizedLevel / 10) + (hasDrop('Upgrade Catalyst') || hasDrop('Rare catalyst') ? 1 : 0));
    }
    if (id === 'upgradeCatalyst' || id === 'wardingScroll' || id === 'refinementCore' || id === 'cubeFragment') return 1;
    if (id === 'oreChunks') return randomInteger(1, 3);
    if (id === 'gelDrop') return randomInteger(1, 2);
    if (isBoss(enemyData)) return 1;
    return randomInteger(1, normalizedLevel >= 45 ? 2 : 1);
  }

  function makeLootItemForPoolEntry(selection, enemy, player, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const level = Math.max(1, Number(enemy && enemy.level) || Number(player && player.level) || 1);
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const randomInteger = typeof settings.randomInt === 'function' ? settings.randomInt : randomInt;
    const getMaterialQuantity = typeof settings.getMonsterMaterialDropQuantity === 'function'
      ? settings.getMonsterMaterialDropQuantity
      : (materialId, targetLevel, targetEnemyData, dropEntry) => getMonsterMaterialDropQuantity(materialId, targetLevel, targetEnemyData, dropEntry, settings);
    const makeMaterial = typeof settings.makeMaterialDrop === 'function' ? settings.makeMaterialDrop : () => null;
    const makeCurrency = typeof settings.makeCurrencyDrop === 'function' ? settings.makeCurrencyDrop : () => null;
    const makeConsumable = typeof settings.makeConsumableDrop === 'function' ? settings.makeConsumableDrop : () => null;
    const makeCard = typeof settings.makeCardDrop === 'function' ? settings.makeCardDrop : () => null;
    const makeSpecificEquipment = typeof settings.makeSpecificDroppedItem === 'function' ? settings.makeSpecificDroppedItem : () => null;
    const makeEquipment = typeof settings.makeDroppedItem === 'function' ? settings.makeDroppedItem : () => null;
    const getCard = typeof settings.getCardDefinition === 'function' ? settings.getCardDefinition : () => null;
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const elite = isElite(enemy);
    switch (selection && selection.type) {
      case 'material':
        return makeMaterial(selection.materialId, getMaterialQuantity(selection.materialId, level, enemyData, selection), {
          rarity: selection.rarity || materialDropDefinitions[selection.materialId] && materialDropDefinitions[selection.materialId].rarity
        });
      case 'upgradeDust':
        return makeMaterial('upgradeDust', getMaterialQuantity('upgradeDust', level, enemyData, selection), { rarity: selection.rarity || 'Uncommon' });
      case 'upgradeCatalyst':
        return makeMaterial('upgradeCatalyst', 1);
      case 'wardingScroll':
        return makeMaterial('wardingScroll', 1);
      case 'refinementCore':
        return makeMaterial('refinementCore', 1);
      case 'gelDrop':
        return makeMaterial('gelDrop', randomInteger(1, 2));
      case 'oreChunks':
        return makeMaterial('oreChunks', randomInteger(1, 3));
      case 'currency':
        return makeCurrency(enemy, Math.round(10 + level * 3 + (elite ? 65 : 0) + randomInteger(0, 8 + level)));
      case 'consumable':
        return makeConsumable(selection.consumableId, 1, { rarity: selection.rarity || 'Common' });
      case 'card':
        return makeCard(selection.cardId, { rarity: selection.rarity || getCard(selection.cardId) && getCard(selection.cardId).rarity });
      case 'base_skill_manual':
        return makeConsumable('base_skill_manual', 1, { rarity: 'Rare' });
      case 'advanced_skill_manual':
        return makeConsumable('advanced_skill_manual', 1, { rarity: 'Rare' });
      case 'skill_reset_scroll':
        return makeConsumable('skill_reset_scroll', 1, { rarity: elite ? 'Epic' : 'Rare' });
      case 'equipment':
        if (selection.itemId) return makeSpecificEquipment(selection.itemId, enemyData || {}, player || {}, selection);
        return makeEquipment(enemyData || {}, player || {});
      default:
        return null;
    }
  }

  function makeDroppedLoot(enemy, player, state, options) {
    const settings = options || {};
    const chooseWeighted = typeof settings.weightedItem === 'function' ? settings.weightedItem : weightedItem;
    const getPool = typeof settings.getMonsterLootPool === 'function'
      ? settings.getMonsterLootPool
      : (targetEnemy, targetState) => getMonsterLootPool(targetEnemy, targetState, settings);
    const makeItem = typeof settings.makeLootItemForPoolEntry === 'function'
      ? settings.makeLootItemForPoolEntry
      : (selection, targetEnemy, targetPlayer) => makeLootItemForPoolEntry(selection, targetEnemy, targetPlayer, settings);
    return makeItem(chooseWeighted(getPool(enemy, state)), enemy, player);
  }

  function makeMonsterGlobalRareDrop(enemy, player, state, options) {
    const settings = options || {};
    const chooseWeighted = typeof settings.weightedItem === 'function' ? settings.weightedItem : weightedItem;
    const getPool = typeof settings.getMonsterGlobalRarePool === 'function'
      ? settings.getMonsterGlobalRarePool
      : (targetEnemy, targetState) => getMonsterGlobalRarePool(targetEnemy, targetState, settings);
    const makeItem = typeof settings.makeLootItemForPoolEntry === 'function'
      ? settings.makeLootItemForPoolEntry
      : (selection, targetEnemy, targetPlayer) => makeLootItemForPoolEntry(selection, targetEnemy, targetPlayer, settings);
    return makeItem(chooseWeighted(getPool(enemy, state)), enemy, player);
  }

  function getMonsterPotionTierScore(enemy, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const range = Array.isArray(enemyData && enemyData.levelRange) ? enemyData.levelRange : [];
    const rangeMax = range.reduce((max, value) => {
      const level = Number(value);
      return Number.isFinite(level) ? Math.max(max, level) : max;
    }, 0);
    const actualLevel = Number(enemy && enemy.level || enemyData && enemyData.level || 0);
    let score = Math.max(1, rangeMax, Number.isFinite(actualLevel) ? actualLevel : 0);
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const id = normalize(enemyData && enemyData.id || enemy && (enemy.id || enemy.enemyId));
    if (isBoss(enemy)) score += 25;
    if (isElite(enemy) || enemyData && enemyData.behavior === 'elite' || id === 'riftAberration') score += 15;
    return score;
  }

  function getMonsterPotionTierIndex(enemy, options) {
    const settings = options || {};
    const tiers = Array.isArray(settings.potionDropTiers) ? settings.potionDropTiers : POTION_DROP_TIERS;
    const getScore = typeof settings.getMonsterPotionTierScore === 'function'
      ? settings.getMonsterPotionTierScore
      : (targetEnemy) => getMonsterPotionTierScore(targetEnemy, settings);
    const score = getScore(enemy);
    let tierIndex = 0;
    tiers.forEach((tier, index) => {
      if (score >= tier.minScore) tierIndex = index;
    });
    return tierIndex;
  }

  function getMonsterBasicConsumablePool(enemy, options) {
    const settings = options || {};
    const tiers = Array.isArray(settings.potionDropTiers) ? settings.potionDropTiers : POTION_DROP_TIERS;
    const getTierIndex = typeof settings.getMonsterPotionTierIndex === 'function'
      ? settings.getMonsterPotionTierIndex
      : (targetEnemy) => getMonsterPotionTierIndex(targetEnemy, settings);
    const tierIndex = getTierIndex(enemy);
    const tier = tiers[tierIndex] || tiers[0];
    const pool = [
      { type: 'consumable', consumableId: tier.hp, weight: 12, rarity: tier.rarity },
      { type: 'consumable', consumableId: tier.mp, weight: 12, rarity: tier.rarity },
      { type: 'consumable', consumableId: tier.hybrid, weight: 6, rarity: tier.rarity }
    ];
    if (tierIndex > 0) {
      const lowerTier = tiers[tierIndex - 1];
      pool.push(
        { type: 'consumable', consumableId: lowerTier.hp, weight: 3, rarity: lowerTier.rarity },
        { type: 'consumable', consumableId: lowerTier.mp, weight: 3, rarity: lowerTier.rarity },
        { type: 'consumable', consumableId: lowerTier.hybrid, weight: 2, rarity: lowerTier.rarity }
      );
    }
    return pool.concat([
      { type: 'consumable', consumableId: 'town_return_scroll', weight: 3 },
      { type: 'consumable', consumableId: 'guard_tonic', weight: 4, rarity: 'Uncommon' },
      { type: 'consumable', consumableId: 'swiftstep_oil', weight: 4, rarity: 'Uncommon' },
      { type: 'consumable', consumableId: 'magnet_charm', weight: 3, rarity: 'Uncommon' }
    ]);
  }

  function getPreferredSkillManualDropId(state, options) {
    const settings = options || {};
    const isSkillPointPoolAtTarget = typeof settings.isSkillPointPoolAtTarget === 'function'
      ? settings.isSkillPointPoolAtTarget
      : function isSkillPointPoolAtTargetFallback() {
          return true;
        };
    if (!state || !state.player || !state.player.classId) return '';
    if (!isSkillPointPoolAtTarget(state, 'baseSkillPoints')) return 'base_skill_manual';
    if (state.player.advancedClassId && !isSkillPointPoolAtTarget(state, 'advancedSkillPoints')) return 'advanced_skill_manual';
    return '';
  }

  function getMonsterBaseDropChance(enemy, options) {
    const settings = options || {};
    const getEconomyNumber = typeof settings.getDropEconomyNumber === 'function'
      ? settings.getDropEconomyNumber
      : function getDropEconomyNumberFallback(path, fallback) {
          return fallback;
        };
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    if (isBoss(enemy)) return clamp(getEconomyNumber('bossLootChance', 0.7), 0, 0.96);
    return isElite(enemy)
      ? clamp(getEconomyNumber('eliteDropChance', 0.45), 0, 0.96)
      : clamp(getEconomyNumber('normalDropChance', 0.12), 0, 0.96);
  }

  function getDropMaterialIdForLabel(label, options) {
    const settings = options || {};
    const labels = settings.monsterDropMaterialLabels || {};
    return labels[String(label || '').trim()] || '';
  }

  function getMonsterMaterialDropEntries(enemyData, options) {
    const settings = options || {};
    const definitions = settings.materialDropDefinitions || {};
    const getMaterialId = typeof settings.getDropMaterialIdForLabel === 'function'
      ? settings.getDropMaterialIdForLabel
      : (label) => getDropMaterialIdForLabel(label, settings);
    const drops = Array.isArray(enemyData && enemyData.drops) ? enemyData.drops : [];
    return drops
      .map((label) => ({ label, materialId: getMaterialId(label) }))
      .filter((entry) => entry.materialId && definitions[entry.materialId]);
  }

  function getEnemyPrimaryMaterialId(enemyData, options) {
    const settings = options || {};
    const definitions = settings.materialDropDefinitions || {};
    const getEntries = typeof settings.getMonsterMaterialDropEntries === 'function'
      ? settings.getMonsterMaterialDropEntries
      : (targetEnemyData) => getMonsterMaterialDropEntries(targetEnemyData, settings);
    const entries = getEntries(enemyData);
    const primaryEntries = entries.filter((entry) => {
      const material = definitions[entry.materialId];
      return material && material.primaryDrop !== false;
    });
    const specific = primaryEntries.find((entry) => {
      const material = definitions[entry.materialId];
      return material && !material.genericDrop;
    });
    return specific && specific.materialId || primaryEntries[0] && primaryEntries[0].materialId || entries[0] && entries[0].materialId || '';
  }

  function getMonsterDropTableBaseChance(tableId, enemy, options) {
    const settings = options || {};
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const getEconomyNumber = typeof settings.getDropEconomyNumber === 'function'
      ? settings.getDropEconomyNumber
      : () => 0;
    if (tableId === 'primaryEtc') return 1;
    if (tableId === 'rareValuables') {
      return typeof settings.getMonsterGlobalRareBaseChance === 'function'
        ? settings.getMonsterGlobalRareBaseChance(enemy)
        : 0;
    }
    const getBaseDropChance = typeof settings.getMonsterBaseDropChance === 'function'
      ? settings.getMonsterBaseDropChance
      : () => 0;
    const baseDropChance = getBaseDropChance(enemy);
    if (tableId === 'plinkoBalls') {
      const getPlinkoChance = typeof settings.getMonsterPlinkoBallBaseChance === 'function'
        ? settings.getMonsterPlinkoBallBaseChance
        : () => 0;
      return getPlinkoChance(enemy) * baseDropChance;
    }
    return normalizeChance(getEconomyNumber(`dropTableChances.${tableId}`, 0), 0) * baseDropChance;
  }

  function getMonsterDropTableCap(tableId, options) {
    const settings = options || {};
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const getEconomyNumber = typeof settings.getDropEconomyNumber === 'function'
      ? settings.getDropEconomyNumber
      : () => 1;
    if (tableId === 'primaryEtc') return 1;
    return normalizeChance(getEconomyNumber(`dropTableCaps.${tableId}`, 1), 1);
  }

  function getMonsterDropTableMultiplier(engine, bonus, options) {
    const settings = options || {};
    const dropBonus = Math.max(0, Number(bonus || 0));
    const couponMultiplier = settings.couponMultiplier != null
      ? Math.max(0, Number(settings.couponMultiplier) || 0)
      : engine && typeof engine.getRateCouponMultiplier === 'function'
      ? Math.max(0, Number(engine.getRateCouponMultiplier('drop')) || 0)
      : 1;
    const adminMultiplier = settings.adminMultiplier != null
      ? Math.max(1, Number(settings.adminMultiplier) || 1)
      : engine && typeof engine.getAdminRate === 'function'
      ? Math.max(1, Number(engine.getAdminRate('dropRate')) || 1)
      : 1;
    return (1 + dropBonus) * couponMultiplier * adminMultiplier;
  }

  function getEffectiveMonsterDropTableChance(table, enemy, bonus, engine, multiplier, options) {
    const settings = options || {};
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const getMultiplier = typeof settings.getMonsterDropTableMultiplier === 'function'
      ? settings.getMonsterDropTableMultiplier
      : getMonsterDropTableMultiplier;
    if (!table) return 0;
    const baseChance = normalizeChance(table.chance, 0);
    if (table.guaranteed) return baseChance;
    const cap = normalizeChance(table.cap, 1);
    const tableMultiplier = Number.isFinite(Number(multiplier))
      ? Math.max(0, Number(multiplier))
      : getMultiplier(engine, bonus, settings);
    return clamp(baseChance * tableMultiplier, 0, cap);
  }

  function normalizeDropTableEntries(entries, tableId, tableLabel, tableChance, options) {
    const settings = options || {};
    const normalizeWeight = typeof settings.normalizeDropWeight === 'function'
      ? settings.normalizeDropWeight
      : normalizeDropWeight;
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const normalizedEntries = (entries || []).filter(Boolean).map((entry) => Object.assign({}, entry, {
      weight: normalizeWeight(entry.weight, 1)
    }));
    const totalWeight = normalizedEntries.reduce((sum, entry) => sum + normalizeWeight(entry.weight, 1), 0);
    return normalizedEntries.map((entry) => {
      const chance = totalWeight > 0 ? normalizeWeight(entry.weight, 1) / totalWeight : 0;
      return Object.assign({}, entry, {
        tableId,
        tableLabel,
        sourcePool: tableId,
        chance,
        chancePerKill: chance * normalizeChance(tableChance, 0)
      });
    });
  }

  function createMonsterDropTable(tableId, label, chance, entries, options) {
    const settings = options || {};
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const normalizeWeight = typeof settings.normalizeDropWeight === 'function'
      ? settings.normalizeDropWeight
      : normalizeDropWeight;
    const normalizeEntries = typeof settings.normalizeDropTableEntries === 'function'
      ? settings.normalizeDropTableEntries
      : (targetEntries, targetTableId, targetLabel, targetChance) => normalizeDropTableEntries(targetEntries, targetTableId, targetLabel, targetChance, settings);
    const tableChance = normalizeChance(chance, 0);
    const tableEntries = normalizeEntries(entries, tableId, label, tableChance);
    if (!tableEntries.length) return null;
    const getCap = typeof settings.getMonsterDropTableCap === 'function'
      ? settings.getMonsterDropTableCap
      : () => 1;
    return {
      id: tableId,
      label,
      chance: tableChance,
      cap: settings.cap == null ? getCap(tableId) : normalizeChance(settings.cap, 1),
      guaranteed: !!settings.guaranteed,
      totalWeight: tableEntries.reduce((sum, entry) => sum + normalizeWeight(entry.weight, 1), 0),
      entries: tableEntries
    };
  }

  function weightedDropTableEntry(table, options) {
    const settings = options || {};
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const entries = table && Array.isArray(table.entries) ? table.entries : [];
    if (!entries.length) return null;
    let roll = random() * Math.max(1, Number(table.totalWeight || 0) || entries.length);
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      roll -= Math.max(1, Number(entry && entry.weight || 1));
      if (roll <= 0) return entry;
    }
    return entries[entries.length - 1];
  }

  function getMonsterPrimaryEtcEntry(enemyData, options) {
    const settings = options || {};
    const config = getMonsterDropPoolConfig(enemyData);
    const materials = Array.isArray(config && config.materials) ? config.materials : [];
    if (materials[0] && materials[0].materialId) {
      return Object.assign({}, materials[0], {
        type: 'material',
        weight: 1
      });
    }
    const getPrimaryMaterialId = typeof settings.getEnemyPrimaryMaterialId === 'function'
      ? settings.getEnemyPrimaryMaterialId
      : () => '';
    const materialId = getPrimaryMaterialId(enemyData);
    return materialId ? { type: 'material', materialId, weight: 1 } : null;
  }

  function getMonsterBonusMaterialEntries(enemyData, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const config = getMonsterDropPoolConfig(enemyData);
    const materials = Array.isArray(config && config.materials) ? config.materials : [];
    return materials.slice(1)
      .filter((entry) => entry && entry.materialId && materialDropDefinitions[entry.materialId])
      .map((entry) => Object.assign({}, entry, {
        type: 'material',
        weight: normalizeDropWeight(entry.weight, 1)
      }));
  }

  function getMonsterEquipmentDropLevelCap(enemyData, options) {
    const settings = options || {};
    const enemy = settings.enemy || enemyData;
    const actualLevel = Number(enemy && enemy.level);
    const range = Array.isArray(enemyData && enemyData.levelRange) ? enemyData.levelRange : [];
    const rangeMax = range.reduce((max, value) => {
      const level = Number(value);
      return Number.isFinite(level) ? Math.max(max, level) : max;
    }, 0);
    const referenceLevel = Number.isFinite(actualLevel) && actualLevel > 0
      ? actualLevel
      : Math.max(1, rangeMax, Number(enemyData && enemyData.level) || 1);
    const leeway = Math.max(0, Math.floor(Number(settings.equipmentLevelLeeway == null ? 8 : settings.equipmentLevelLeeway) || 0));
    return Math.max(1, Math.floor(referenceLevel)) + leeway;
  }

  function getMonsterEquipmentTableEntries(enemyData, options) {
    const settings = options || {};
    const getDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : () => null;
    const getWeight = typeof settings.getMonsterEquipmentDropWeight === 'function'
      ? settings.getMonsterEquipmentDropWeight
      : (entry) => normalizeDropWeight(entry && entry.weight, 1);
    const matchesPlayerClass = typeof settings.itemMatchesPlayerClass === 'function'
      ? settings.itemMatchesPlayerClass
      : () => true;
    const state = settings.state || null;
    const player = state && state.player || null;
    const levelCap = getMonsterEquipmentDropLevelCap(enemyData, settings);
    const config = getMonsterDropPoolConfig(enemyData);
    const equipment = Array.isArray(config && config.equipment) ? config.equipment : [];
    return equipment
      .map((entry) => ({ entry, base: entry && entry.itemId ? getDefinition(entry.itemId) : null }))
      .filter(({ base }) => {
        if (!base || Math.max(1, Number(base.level) || 1) > levelCap) return false;
        if (normalizeId(base.slot) === 'weapon' && player && player.classId && !matchesPlayerClass(base, player)) return false;
        return true;
      })
      .map(({ entry }) => Object.assign({}, entry, {
        type: 'equipment',
        weight: normalizeDropWeight(getWeight(entry, state), 1)
      }));
  }

  function getMonsterCardTableEntries(enemyData, options) {
    const settings = options || {};
    const getDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : () => null;
    const config = getMonsterDropPoolConfig(enemyData);
    const cards = Array.isArray(config && config.cards) ? config.cards : [];
    return cards
      .filter((entry) => entry && entry.cardId && getDefinition(entry.cardId))
      .map((entry) => Object.assign({}, entry, {
        type: 'card',
        weight: normalizeDropWeight(entry.weight, 1)
      }));
  }

  function getMonsterPotionTableEntries(enemy, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const config = getMonsterDropPoolConfig(enemyData);
    if (config && config.basicConsumables === false) return [];
    const getConsumable = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : () => null;
    const getBasicPool = typeof settings.getMonsterBasicConsumablePool === 'function'
      ? settings.getMonsterBasicConsumablePool
      : (targetEnemy) => getMonsterBasicConsumablePool(targetEnemy, settings);
    return getBasicPool(enemy)
      .filter((entry) => entry && entry.consumableId && getConsumable(entry.consumableId))
      .map((entry) => Object.assign({}, entry, {
        type: 'consumable',
        weight: normalizeDropWeight(entry.weight, 1)
      }));
  }

  function getMonsterPlinkoBallBaseChance(enemy, options) {
    const settings = options || {};
    const plinkoBalls = Array.isArray(settings.plinkoBalls) ? settings.plinkoBalls : [];
    if (!plinkoBalls.length) return 0;
    const enemyData = getEnemyData(enemy);
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const normalizeChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : normalizeDropChance;
    const getEconomyNumber = typeof settings.getDropEconomyNumber === 'function'
      ? settings.getDropEconomyNumber
      : () => 0;
    const id = normalize(enemyData && enemyData.id || enemy && enemy.id);
    if (isBoss(enemyData)) return 0.55;
    if (id === 'crackedMimic' || id === 'riftAberration') return 0.18;
    if (isElite(enemy)) return 0.1;
    return normalizeChance(getEconomyNumber('dropTableChances.plinkoBalls', 0.04), 0.04);
  }

  function getMonsterPlinkoBallTableEntries(enemy, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const getConsumable = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : () => null;
    const basicBallId = settings.basicBallId || 'plinko_ball_basic';
    const polishedBallId = settings.polishedBallId || 'plinko_ball_polished';
    const meteorBallId = settings.meteorBallId || 'plinko_ball_meteor';
    const id = normalize(enemyData && enemyData.id || enemy && enemy.id);
    const boss = isBoss(enemyData);
    const specialElite = id === 'crackedMimic' || id === 'riftAberration';
    const elite = isElite(enemy);
    const weights = boss
      ? { [basicBallId]: 3, [polishedBallId]: 12, [meteorBallId]: 3 }
      : specialElite
        ? { [basicBallId]: 3, [polishedBallId]: 7, [meteorBallId]: 1 }
        : elite
          ? { [basicBallId]: 8, [polishedBallId]: 2 }
          : { [basicBallId]: 1 };
    return Object.entries(weights)
      .map(([consumableId, weight]) => {
        const item = getConsumable(consumableId);
        if (!item) return null;
        return {
          type: 'consumable',
          consumableId,
          rarity: item.rarity || 'Uncommon',
          weight
        };
      })
      .filter(Boolean);
  }

  function getLegacyMonsterLootPool(enemy, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const drops = Array.isArray(enemyData && enemyData.drops) ? enemyData.drops : [];
    const hasDrop = (label) => drops.includes(label);
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const getPrimaryMaterialId = typeof settings.getEnemyPrimaryMaterialId === 'function'
      ? settings.getEnemyPrimaryMaterialId
      : () => '';
    const getBasicConsumables = typeof settings.getMonsterBasicConsumablePool === 'function'
      ? settings.getMonsterBasicConsumablePool
      : (targetEnemy) => getMonsterBasicConsumablePool(targetEnemy, settings);
    const elite = isElite(enemy);
    const primaryMaterialId = getPrimaryMaterialId(enemyData);
    const hasCatalyst = hasDrop('Upgrade Catalyst') || hasDrop('Rare catalyst');
    const pool = [{ type: 'currency', weight: hasDrop('Currency burst') ? 58 : 42, sourcePool: 'monster' }];
    if (primaryMaterialId) pool.push({ type: 'material', materialId: primaryMaterialId, weight: 40, sourcePool: 'monster' });
    getBasicConsumables(enemy, state).forEach((entry) => pool.push(Object.assign({ sourcePool: 'monster' }, entry)));
    pool.push({ type: 'equipment', weight: hasDrop('Rare gear') ? 11 : 7, sourcePool: 'monster' });
    if (hasDrop('Upgrade Dust') || hasCatalyst) pool.push({ type: 'material', materialId: 'upgradeDust', weight: hasCatalyst ? 9 : 6, sourcePool: 'monster' });
    if (hasCatalyst) pool.push({ type: 'material', materialId: 'upgradeCatalyst', weight: 5, sourcePool: 'monster' });
    if (hasDrop('Prism Shards')) pool.push({ type: 'material', materialId: 'cubeFragment', weight: 2, sourcePool: 'monster' });
    if (hasCatalyst || elite) {
      pool.push({ type: 'material', materialId: 'wardingScroll', weight: elite ? 2 : 1, sourcePool: 'monster' });
      pool.push({ type: 'material', materialId: 'refinementCore', weight: elite ? 2 : 1, sourcePool: 'monster' });
    }
    return pool;
  }

  function getMonsterLootPool(enemy, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const getConfig = typeof settings.getMonsterDropPoolConfig === 'function'
      ? settings.getMonsterDropPoolConfig
      : getMonsterDropPoolConfig;
    const getLegacyPool = typeof settings.getLegacyMonsterLootPool === 'function'
      ? settings.getLegacyMonsterLootPool
      : (targetEnemy, targetState) => getLegacyMonsterLootPool(targetEnemy, targetState, settings);
    const getTables = typeof settings.getMonsterDropTables === 'function'
      ? settings.getMonsterDropTables
      : (targetEnemy, targetState) => createMonsterDropTables(targetEnemy, targetState, settings);
    const config = getConfig(enemyData);
    if (!config) return getLegacyPool(enemy, state);
    return getTables(enemy, state).reduce((entries, table) => entries.concat(table.entries || []), []);
  }

  function monsterDropPoolHasDrops(enemyData) {
    const config = getMonsterDropPoolConfig(enemyData);
    if (!config) return true;
    return !!((config.materials || []).length || (config.equipment || []).length || (config.consumables || []).length || (config.cards || []).length || Number(config.currencyWeight || 0) > 0);
  }

  function getMonsterDropTableCacheKey(enemy, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const player = state && state.player || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const getManualId = typeof settings.getPreferredSkillManualDropId === 'function'
      ? settings.getPreferredSkillManualDropId
      : () => '';
    return [
      normalize(enemyData && enemyData.id || enemy && enemy.id),
      Math.max(0, Math.floor(Number(enemy && enemy.level || enemyData && enemyData.level || 0) || 0)),
      isElite(enemy) ? 1 : 0,
      normalize(player.classId),
      normalize(player.advancedClassId),
      getManualId(state)
    ].join('|');
  }

  function createMonsterDropTables(enemy, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const createTable = typeof settings.createMonsterDropTable === 'function'
      ? settings.createMonsterDropTable
      : createMonsterDropTable;
    const getPrimaryEntry = typeof settings.getMonsterPrimaryEtcEntry === 'function'
      ? settings.getMonsterPrimaryEtcEntry
      : getMonsterPrimaryEtcEntry;
    const hasPoolDrops = typeof settings.monsterDropPoolHasDrops === 'function'
      ? settings.monsterDropPoolHasDrops
      : monsterDropPoolHasDrops;
    const getBaseChance = typeof settings.getMonsterDropTableBaseChance === 'function'
      ? settings.getMonsterDropTableBaseChance
      : (tableId, targetEnemy) => getMonsterDropTableBaseChance(tableId, targetEnemy, settings);
    const getPotionEntries = typeof settings.getMonsterPotionTableEntries === 'function'
      ? settings.getMonsterPotionTableEntries
      : (targetEnemy) => getMonsterPotionTableEntries(targetEnemy, settings);
    const getEquipmentEntries = typeof settings.getMonsterEquipmentTableEntries === 'function'
      ? settings.getMonsterEquipmentTableEntries
      : (targetEnemyData, targetEnemy, targetState) => getMonsterEquipmentTableEntries(targetEnemyData, Object.assign({}, settings, {
          enemy: targetEnemy,
          state: targetState
        }));
    const getCardEntries = typeof settings.getMonsterCardTableEntries === 'function'
      ? settings.getMonsterCardTableEntries
      : (targetEnemyData) => getMonsterCardTableEntries(targetEnemyData, settings);
    const getBonusMaterialEntries = typeof settings.getMonsterBonusMaterialEntries === 'function'
      ? settings.getMonsterBonusMaterialEntries
      : (targetEnemyData) => getMonsterBonusMaterialEntries(targetEnemyData, settings);
    const getPlinkoEntries = typeof settings.getMonsterPlinkoBallTableEntries === 'function'
      ? settings.getMonsterPlinkoBallTableEntries
      : () => [];
    const getRareEntries = typeof settings.getMonsterRareValuableTableEntries === 'function'
      ? settings.getMonsterRareValuableTableEntries
      : () => [];
    const tables = [];
    const primaryEntry = getPrimaryEntry(enemyData);
    if (primaryEntry) {
      tables.push(createTable('primaryEtc', 'Guaranteed Etc', 1, [primaryEntry], { guaranteed: true, cap: 1 }));
    }
    const config = getMonsterDropPoolConfig(enemyData);
    const hasDrops = hasPoolDrops(enemyData);
    if (hasDrops && (!config || Number(config.currencyWeight || 0) > 0)) {
      tables.push(createTable('coins', 'Coins', getBaseChance('coins', enemyData), [{ type: 'currency', weight: 1 }]));
    }
    if (hasDrops) {
      tables.push(createTable('potions', 'Potions', getBaseChance('potions', enemyData), getPotionEntries(enemy)));
      tables.push(createTable('equipment', 'Equipment', getBaseChance('equipment', enemyData), getEquipmentEntries(enemyData, enemy, state)));
      tables.push(createTable('cards', 'Cards', getBaseChance('cards', enemyData), getCardEntries(enemyData)));
      tables.push(createTable('bonusMaterials', 'Bonus Materials', getBaseChance('bonusMaterials', enemyData), getBonusMaterialEntries(enemyData)));
      tables.push(createTable('plinkoBalls', 'Plinko Balls', getBaseChance('plinkoBalls', enemyData), getPlinkoEntries(enemy)));
    }
    tables.push(createTable('rareValuables', 'Rare Valuables', getBaseChance('rareValuables', enemyData), getRareEntries(enemyData, state)));
    return tables.filter(Boolean);
  }

  function getMonsterGlobalRareBaseChance(enemy, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const getConfig = typeof settings.getMonsterDropPoolConfig === 'function'
      ? settings.getMonsterDropPoolConfig
      : getMonsterDropPoolConfig;
    const dropPool = getConfig(enemyData);
    if (dropPool && dropPool.globalRareEligible === false) return 0;
    const getEconomyValue = typeof settings.getDropEconomyValue === 'function'
      ? settings.getDropEconomyValue
      : () => ({});
    const chances = getEconomyValue('globalRareChance', {}) || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const id = normalize(enemyData && enemyData.id || enemy && enemy.id);
    if (isBoss(enemyData)) return clamp(Number(chances.boss || 0.015), 0, 0.2);
    if (id === 'crackedMimic' || id === 'riftAberration') return clamp(Number(chances.specialElite || 0.02), 0, 0.2);
    if (isElite(enemy)) return clamp(Number(chances.elite || 0.0075), 0, 0.2);
    return clamp(Number(chances.normal || 0.0025), 0, 0.2);
  }

  function getMonsterGlobalRarePool(enemy, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const getBaseChance = typeof settings.getMonsterGlobalRareBaseChance === 'function'
      ? settings.getMonsterGlobalRareBaseChance
      : (targetEnemy) => getMonsterGlobalRareBaseChance(targetEnemy, settings);
    if (getBaseChance(enemyData) <= 0) return [];
    const isBoss = typeof settings.isBossMonster === 'function' ? settings.isBossMonster : defaultIsBossMonster;
    const isElite = typeof settings.isEliteMonster === 'function' ? settings.isEliteMonster : defaultIsEliteMonster;
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const getManualId = typeof settings.getPreferredSkillManualDropId === 'function'
      ? settings.getPreferredSkillManualDropId
      : () => '';
    const getConsumable = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : () => null;
    const potentialCubeId = settings.potentialCubeId || 'potential_cube';
    const preservationCubeId = settings.preservationCubeId || 'preservation_cube';
    const elite = isElite(enemy);
    const boss = isBoss(enemyData);
    const manualId = getManualId(state);
    const pool = [
      { type: 'consumable', consumableId: potentialCubeId, weight: elite || boss ? 18 : 14, rarity: 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: preservationCubeId, weight: elite || boss ? 8 : 5, rarity: 'Epic', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'xp_coupon_1_2_1h', weight: 12, rarity: 'Uncommon', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'drop_coupon_1_2_1h', weight: 12, rarity: 'Uncommon', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'equipment_slot_coupon', weight: 5, rarity: 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'usable_slot_coupon', weight: 5, rarity: 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'etc_slot_coupon', weight: 5, rarity: 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'card_slot_coupon', weight: 5, rarity: 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'skill_reset_scroll', weight: elite || boss ? 5 : 3, rarity: elite || boss ? 'Epic' : 'Rare', sourcePool: 'globalRare' },
      { type: 'consumable', consumableId: 'stat_reset_scroll', weight: 3, rarity: 'Rare', sourcePool: 'globalRare' }
    ];
    if (manualId) pool.push({ type: 'consumable', consumableId: manualId, weight: elite || boss ? 9 : 7, rarity: 'Rare', sourcePool: 'globalRare' });
    if (elite || boss) {
      pool.push({ type: 'consumable', consumableId: 'xp_coupon_1_5_1h', weight: 7, rarity: 'Rare', sourcePool: 'globalRare' });
      pool.push({ type: 'consumable', consumableId: 'drop_coupon_1_5_1h', weight: 7, rarity: 'Rare', sourcePool: 'globalRare' });
    }
    if (boss || normalize(enemyData && enemyData.id) === 'crackedMimic') {
      pool.push({ type: 'consumable', consumableId: 'xp_coupon_2_0_1h', weight: 4, rarity: 'Epic', sourcePool: 'globalRare' });
      pool.push({ type: 'consumable', consumableId: 'drop_coupon_2_0_1h', weight: 4, rarity: 'Epic', sourcePool: 'globalRare' });
    }
    return pool.filter((entry) => getConsumable(entry.consumableId));
  }

  function getMonsterRareValuableTableEntries(enemy, state, options) {
    const settings = options || {};
    const getPool = typeof settings.getMonsterGlobalRarePool === 'function'
      ? settings.getMonsterGlobalRarePool
      : (targetEnemy, targetState) => getMonsterGlobalRarePool(targetEnemy, targetState, settings);
    const normalizeWeight = typeof settings.normalizeDropWeight === 'function'
      ? settings.normalizeDropWeight
      : normalizeDropWeight;
    return getPool(enemy, state).map((entry) => Object.assign({}, entry, {
      weight: normalizeWeight(entry.weight, 1)
    }));
  }

  function rollEnemyDropCount(enemy, state, engine, bonus, options) {
    const settings = options || {};
    const getTables = typeof settings.getMonsterDropTables === 'function'
      ? settings.getMonsterDropTables
      : (targetEnemy, targetState) => createMonsterDropTables(targetEnemy, targetState, settings);
    const getMultiplier = typeof settings.getMonsterDropTableMultiplier === 'function'
      ? settings.getMonsterDropTableMultiplier
      : getMonsterDropTableMultiplier;
    const getEffectiveChance = typeof settings.getEffectiveMonsterDropTableChance === 'function'
      ? settings.getEffectiveMonsterDropTableChance
      : getEffectiveMonsterDropTableChance;
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const dropTables = getTables(enemy, state);
    const dropTableMultiplier = getMultiplier(engine, bonus);
    let count = 0;
    for (let index = 0; index < dropTables.length; index += 1) {
      const table = dropTables[index];
      if (!table || table.guaranteed || table.id === 'rareValuables') continue;
      if (random() < getEffectiveChance(table, enemy, bonus, engine, dropTableMultiplier)) count += 1;
    }
    return count;
  }

  function rollMonsterGlobalRareDropCount(enemy, state, engine, bonus, options) {
    const settings = options || {};
    const getTables = typeof settings.getMonsterDropTables === 'function'
      ? settings.getMonsterDropTables
      : (targetEnemy, targetState) => createMonsterDropTables(targetEnemy, targetState, settings);
    const getMultiplier = typeof settings.getMonsterDropTableMultiplier === 'function'
      ? settings.getMonsterDropTableMultiplier
      : getMonsterDropTableMultiplier;
    const getEffectiveChance = typeof settings.getEffectiveMonsterDropTableChance === 'function'
      ? settings.getEffectiveMonsterDropTableChance
      : getEffectiveMonsterDropTableChance;
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const dropTables = getTables(enemy, state);
    const dropTableMultiplier = getMultiplier(engine, bonus);
    for (let index = 0; index < dropTables.length; index += 1) {
      const table = dropTables[index];
      if (!table || table.id !== 'rareValuables') continue;
      return random() < getEffectiveChance(table, enemy, bonus, engine, dropTableMultiplier) ? 1 : 0;
    }
    return 0;
  }

  function ensureBossSetMisses(state) {
    const targetState = state && typeof state === 'object' ? state : {};
    targetState.bossSetMissesByBossId = targetState.bossSetMissesByBossId && typeof targetState.bossSetMissesByBossId === 'object'
      ? targetState.bossSetMissesByBossId
      : {};
    return targetState.bossSetMissesByBossId;
  }

  function getBossSetMissCount(bossId, state, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const ensureMisses = typeof settings.ensureBossSetMisses === 'function'
      ? settings.ensureBossSetMisses
      : ensureBossSetMisses;
    const id = normalize(bossId);
    if (!id) return 0;
    const misses = ensureMisses(state);
    return Math.max(0, Math.floor(Number(misses[id]) || 0));
  }

  function getBossSetPityBonus(source, state, options) {
    if (!source || !source.bossId) return 0;
    const settings = options || {};
    const getMissCount = typeof settings.getBossSetMissCount === 'function'
      ? settings.getBossSetMissCount
      : (bossId) => getBossSetMissCount(bossId, state, settings);
    const getEconomyNumber = typeof settings.getDropEconomyNumber === 'function'
      ? settings.getDropEconomyNumber
      : function getDropEconomyNumberFallback(path, fallback) {
          return fallback;
        };
    const clampValue = typeof settings.clamp === 'function' ? settings.clamp : clamp;
    const rarity = String(source.rarity || '').toLowerCase();
    const misses = getMissCount(source.bossId);
    const start = Math.max(0, Math.floor(getEconomyNumber(`bossPity.${rarity}Start`, rarity === 'relic' ? 10 : 8)));
    if (misses < start) return 0;
    const step = getEconomyNumber(`bossPity.${rarity}Step`, rarity === 'relic' ? 0.0075 : 0.01);
    const max = getEconomyNumber(`bossPity.${rarity}Max`, rarity === 'relic' ? 0.09 : 0.1);
    return clampValue((misses - start + 1) * step, 0, max);
  }

  function getBossSetDropChance(source, state, options) {
    if (!source) return 0;
    const settings = options || {};
    const getPityBonus = typeof settings.getBossSetPityBonus === 'function'
      ? settings.getBossSetPityBonus
      : (targetSource) => getBossSetPityBonus(targetSource, state, settings);
    const clampValue = typeof settings.clamp === 'function' ? settings.clamp : clamp;
    return clampValue(Math.max(0, Number(source.dropChance || 0)) + getPityBonus(source), 0, 0.95);
  }

  function recordBossSetDropRoll(source, dropped, state, options) {
    if (!source || !source.bossId) return null;
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const ensureMisses = typeof settings.ensureBossSetMisses === 'function'
      ? settings.ensureBossSetMisses
      : ensureBossSetMisses;
    const getMissCount = typeof settings.getBossSetMissCount === 'function'
      ? settings.getBossSetMissCount
      : (bossId) => getBossSetMissCount(bossId, state, settings);
    const id = normalize(source.bossId);
    if (!id) return null;
    const misses = ensureMisses(state);
    misses[id] = dropped ? 0 : getMissCount(id) + 1;
    return misses[id];
  }

  function createBossDropPreview(encounter, state, options) {
    if (!encounter) return { setId: '', setName: '', rarity: '', dropChance: 0, label: 'Boss loot' };
    const settings = options || {};
    const getBossSource = typeof settings.getBossEquipmentSource === 'function'
      ? settings.getBossEquipmentSource
      : function getBossEquipmentSourceFallback() {
          return null;
        };
    const getSetDefinition = typeof settings.getEquipmentSetDefinition === 'function'
      ? settings.getEquipmentSetDefinition
      : function getEquipmentSetDefinitionFallback() {
          return null;
        };
    const getDropChance = typeof settings.getBossSetDropChance === 'function'
      ? settings.getBossSetDropChance
      : (source) => getBossSetDropChance(source, state, settings);
    const getPityBonus = typeof settings.getBossSetPityBonus === 'function'
      ? settings.getBossSetPityBonus
      : (source) => getBossSetPityBonus(source, state, settings);
    const getMissCount = typeof settings.getBossSetMissCount === 'function'
      ? settings.getBossSetMissCount
      : (bossId) => getBossSetMissCount(bossId, state, settings);
    const source = getBossSource(encounter.bossId) || {};
    const setId = encounter.setId || source.setId || '';
    const set = setId ? getSetDefinition(setId) : null;
    const rarity = source.rarity || '';
    const chance = getDropChance(source);
    return {
      setId,
      setName: set && set.name || '',
      rarity,
      dropChance: chance,
      baseDropChance: Math.max(0, Number(source.dropChance || 0)),
      pityBonus: getPityBonus(source),
      pityMisses: getMissCount(source.bossId),
      label: set && rarity
        ? `${rarity} ${set.name} set`
        : set
          ? `${set.name} set`
          : 'Boss materials and rare loot'
    };
  }

  function makeBossDroppedItem(enemy, player, state, options) {
    const settings = options || {};
    const enemyData = getEnemyData(enemy);
    const getBossSource = typeof settings.getBossEquipmentSource === 'function'
      ? settings.getBossEquipmentSource
      : function getBossEquipmentSourceFallback() {
          return null;
        };
    const bossEquipmentItems = Array.isArray(settings.bossEquipmentItems) ? settings.bossEquipmentItems : [];
    const getWeight = typeof settings.getBossEquipmentDropWeight === 'function'
      ? settings.getBossEquipmentDropWeight
      : function getBossEquipmentDropWeightFallback() {
          return 1;
        };
    const chooseWeighted = typeof settings.weightedItem === 'function' ? settings.weightedItem : weightedItem;
    const createItem = typeof settings.makeItem === 'function'
      ? settings.makeItem
      : function makeItemFallback() {
          return null;
        };
    const assignTrait = typeof settings.maybeAssignGearTrait === 'function'
      ? settings.maybeAssignGearTrait
      : function maybeAssignGearTraitFallback(item) {
          return item;
        };
    const source = getBossSource(enemyData && enemyData.id);
    if (!source) return null;
    const candidates = bossEquipmentItems.filter((item) => {
      if (!item || item.setId !== source.setId) return false;
      return true;
    });
    const weightedCandidates = candidates.map((item) => ({
      item,
      weight: getWeight(item, player, state)
    }));
    const selected = chooseWeighted(weightedCandidates);
    const base = selected && selected.item || null;
    if (!base) return null;
    return assignTrait(createItem(base, {
      rarity: base.rarity || source.rarity,
      upgrade: base.rarity === 'Relic' ? 3 : 2,
      randomizeStats: true,
      source: `${source.name || enemyData.name || 'Boss'} drop`
    }), 0.9);
  }

  function makeDroppedItem(enemy, player, options) {
    const settings = options || {};
    const getCatalog = typeof settings.getGenericEquipmentDropCatalog === 'function'
      ? settings.getGenericEquipmentDropCatalog
      : function getGenericEquipmentDropCatalogFallback() {
          return [];
        };
    const getWeight = typeof settings.getEquipmentDropClassWeight === 'function'
      ? settings.getEquipmentDropClassWeight
      : function getEquipmentDropClassWeightFallback() {
          return 1;
        };
    const chooseWeighted = typeof settings.weightedItem === 'function' ? settings.weightedItem : weightedItem;
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const createItem = typeof settings.makeItem === 'function'
      ? settings.makeItem
      : function makeItemFallback() {
          return null;
        };
    const assignTrait = typeof settings.maybeAssignGearTrait === 'function'
      ? settings.maybeAssignGearTrait
      : function maybeAssignGearTraitFallback(item) {
          return item;
        };
    const chooseRandomItem = typeof settings.randItem === 'function' ? settings.randItem : weightedItem;
    const matchesPlayerClass = typeof settings.itemMatchesPlayerClass === 'function'
      ? settings.itemMatchesPlayerClass
      : () => true;
    const mutations = Array.isArray(settings.mutations) ? settings.mutations : [];
    const level = Math.max(1, Number(player && player.level) || Number(enemy && enemy.level) || 1);
    const catalog = getCatalog();
    const dropItems = catalog.filter((item) => {
      if (!item || item.level > level + 8) return false;
      if (normalizeId(item.slot) === 'weapon' && player && player.classId && !matchesPlayerClass(item, player)) return false;
      return true;
    });
    const weightedDropItems = dropItems.map((item) => ({
      item,
      weight: getWeight(item, player)
    }));
    const selected = weightedDropItems.length ? chooseWeighted(weightedDropItems) : null;
    const base = selected && selected.item || dropItems[0];
    if (!base) return null;
    const rarityRoll = random();
    let rarity = base.rarity;
    if (rarityRoll > 0.96) rarity = 'Epic';
    else if (rarityRoll > 0.78) rarity = 'Rare';
    else if (rarityRoll > 0.48 && rarity === 'Common') rarity = 'Uncommon';
    const item = createItem(base, {
      rarity,
      upgrade: rarity === 'Epic' ? 2 : rarity === 'Rare' ? 1 : 0,
      randomizeStats: true,
      source: `${enemy.name} drop`,
      levelBoost: enemy.id === 'emberjawGolem' || enemy.id === 'crackedMimic' ? 0.25 : 0.08
    });
    if (rarity === 'Epic' && mutations.length) {
      item.mutation = chooseRandomItem(mutations).id;
    }
    assignTrait(item, rarity === 'Epic' ? 0.7 : rarity === 'Rare' ? 0.45 : rarity === 'Uncommon' ? 0.18 : 0);
    return item;
  }

  function makeSpecificDroppedItem(itemId, enemy, player, dropEntry, options) {
    const settings = options || {};
    const getEquipmentDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : function getEquipmentDefinitionFallback() {
          return null;
        };
    const makeGenericDrop = typeof settings.makeDroppedItem === 'function'
      ? settings.makeDroppedItem
      : (targetEnemy, targetPlayer) => makeDroppedItem(targetEnemy, targetPlayer, settings);
    const createItem = typeof settings.makeItem === 'function'
      ? settings.makeItem
      : function makeItemFallback() {
          return null;
        };
    const assignTrait = typeof settings.maybeAssignGearTrait === 'function'
      ? settings.maybeAssignGearTrait
      : function maybeAssignGearTraitFallback(item) {
          return item;
        };
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : [];
    const base = getEquipmentDefinition(itemId);
    if (!base) return makeGenericDrop(enemy, player);
    const entry = dropEntry || {};
    const rarity = entry.rarity || base.rarity || 'Common';
    const rarityIndex = Math.max(0, rarityOrder.indexOf(rarity));
    const item = createItem(base, {
      rarity,
      upgrade: rarityIndex >= 3 ? 2 : rarityIndex >= 2 ? 1 : 0,
      randomizeStats: true,
      source: `${enemy && enemy.name || 'Monster'} drop`,
      levelBoost: enemy && (enemy.id === 'crackedMimic' || enemy.id === 'riftAberration') ? 0.2 : 0.08
    });
    return assignTrait(item, rarity === 'Epic' ? 0.7 : rarity === 'Rare' ? 0.45 : rarity === 'Uncommon' ? 0.18 : 0);
  }

  const api = {
    POTION_DROP_TIERS,
    getMonsterDropPoolConfig,
    normalizeDropWeight,
    normalizeDropChance,
    getDropEconomyValue,
    getDropEconomyNumber,
    getCoinStackAssetId,
    describeLootItem,
    makeMaterialDrop,
    makeCardDrop,
    makeConsumableDrop,
    makeCurrencyDrop,
    getLootStackQuantity,
    getLootStackAmount,
    isStackableLootItem,
    isPriorityStackableLootItem,
    getLootStackKey,
    mergeStackableLootItem,
    prioritizeMonsterLootBatch,
    enforceLootDropCap,
    createLootDropCacheContext,
    lootDropCacheMatches,
    createLootDropUidMapCache,
    createCurrentMapLootDropCaches,
    removeLootDropByUid,
    createLootInventoryAdmissionContext,
    getLootAdmissionContextCapacity,
    getLootAdmissionContextUsedSlots,
    getStackableInventoryAddCapacityForAdmission,
    createLootDropInventoryAdmission,
    createLootRewardToastMessage,
    canPetLootDrop,
    createPetLootClusterCounts,
    scorePetLootCandidate,
    selectPetLootTargetCandidate,
    createPetLootTouchBox,
    petLootTouchBoxOverlapsDrop,
    isLootCollectible,
    createLootDrawBox,
    lootDrawBoxOverlapsBox,
    lootDropCouldOverlapBox,
    getMonsterMaterialDropQuantity,
    makeLootItemForPoolEntry,
    makeDroppedLoot,
    makeMonsterGlobalRareDrop,
    getMonsterPotionTierScore,
    getMonsterPotionTierIndex,
    getMonsterBasicConsumablePool,
    getPreferredSkillManualDropId,
    getMonsterBaseDropChance,
    getDropMaterialIdForLabel,
    getMonsterMaterialDropEntries,
    getEnemyPrimaryMaterialId,
    getMonsterDropTableBaseChance,
    getMonsterDropTableCap,
    getMonsterDropTableMultiplier,
    getEffectiveMonsterDropTableChance,
    normalizeDropTableEntries,
    createMonsterDropTable,
    weightedDropTableEntry,
    getMonsterPrimaryEtcEntry,
    getMonsterBonusMaterialEntries,
    getMonsterEquipmentDropLevelCap,
    getMonsterEquipmentTableEntries,
    getMonsterCardTableEntries,
    getMonsterPotionTableEntries,
    getMonsterPlinkoBallBaseChance,
    getMonsterPlinkoBallTableEntries,
    getLegacyMonsterLootPool,
    getMonsterLootPool,
    monsterDropPoolHasDrops,
    getMonsterDropTableCacheKey,
    createMonsterDropTables,
    getMonsterGlobalRareBaseChance,
    getMonsterGlobalRarePool,
    getMonsterRareValuableTableEntries,
    rollEnemyDropCount,
    rollMonsterGlobalRareDropCount,
    ensureBossSetMisses,
    getBossSetMissCount,
    getBossSetPityBonus,
    getBossSetDropChance,
    recordBossSetDropRoll,
    createBossDropPreview,
    makeBossDroppedItem,
    makeDroppedItem,
    makeSpecificDroppedItem
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.loot = Object.assign({}, modules.loot || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
