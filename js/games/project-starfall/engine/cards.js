(function initProjectStarfallEngineCards(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const CARD_DECK_COUNT = 4;
  const CARD_DECK_SLOT_COUNT = 6;
  const CARD_UPGRADE_COPY_COUNT = 3;
  const CARD_SELL_RARITY_VALUES = Object.freeze({
    Common: 18,
    Uncommon: 36,
    Rare: 72,
    Epic: 144,
    Relic: 288
  });
  const DEFAULT_STAR_CARD_MATERIAL_IDS = Object.freeze({
    white: 'whiteStarCard',
    green: 'greenStarCard',
    blue: 'blueStarCard',
    purple: 'purpleStarCard',
    orange: 'orangeStarCard'
  });

  function createStarCardMaterialIds(data) {
    const sourceData = data || {};
    return Object.freeze(Object.assign({}, DEFAULT_STAR_CARD_MATERIAL_IDS, sourceData.STAR_CARD_MATERIAL_IDS || {}));
  }

  function createStarCardMaterialOrder(materialIds) {
    const ids = materialIds || DEFAULT_STAR_CARD_MATERIAL_IDS;
    return Object.freeze([
      ids.white,
      ids.green,
      ids.blue,
      ids.purple,
      ids.orange
    ]);
  }

  function createCardStarUpgradeCosts(materialIds) {
    const ids = materialIds || DEFAULT_STAR_CARD_MATERIAL_IDS;
    return Object.freeze({
      2: Object.freeze({ [ids.white]: 3, [ids.green]: 1 }),
      3: Object.freeze({ [ids.white]: 10, [ids.green]: 3, [ids.blue]: 1 }),
      4: Object.freeze({ [ids.white]: 20, [ids.green]: 4, [ids.blue]: 2, [ids.purple]: 1 }),
      5: Object.freeze({ [ids.white]: 100, [ids.green]: 20, [ids.blue]: 4, [ids.purple]: 2, [ids.orange]: 1 })
    });
  }

  function createStarCardExchangeRates(materialIds) {
    const ids = materialIds || DEFAULT_STAR_CARD_MATERIAL_IDS;
    return Object.freeze([
      Object.freeze({ from: ids.white, to: ids.green, rate: 5 }),
      Object.freeze({ from: ids.green, to: ids.blue, rate: 4 }),
      Object.freeze({ from: ids.blue, to: ids.purple, rate: 3 }),
      Object.freeze({ from: ids.purple, to: ids.orange, rate: 2 })
    ]);
  }

  function normalizeCardMaterialCost(cost) {
    return Object.entries(cost || {}).reduce((normalized, [id, amount]) => {
      const materialId = normalizeId(id);
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (materialId && value > 0) normalized[materialId] = value;
      return normalized;
    }, {});
  }

  function multiplyCardMaterialCost(cost, quantity) {
    const amount = Math.max(1, Math.floor(Number(quantity || 1) || 1));
    return Object.entries(normalizeCardMaterialCost(cost)).reduce((scaled, [id, value]) => {
      scaled[id] = value * amount;
      return scaled;
    }, {});
  }

  function getCardMaterialCostCounts(materials, cost, options) {
    const settings = options || {};
    const source = materials && typeof materials === 'object' ? materials : {};
    const getName = typeof settings.getMaterialDisplayName === 'function'
      ? settings.getMaterialDisplayName
      : (materialId) => String(materialId || '');
    const getAssetId = typeof settings.getMaterialAssetIdForMaterial === 'function'
      ? settings.getMaterialAssetIdForMaterial
      : (materialId) => materialId;
    return Object.entries(normalizeCardMaterialCost(cost)).reduce((counts, [id, amount]) => {
      const owned = Math.max(0, Math.floor(Number(source[id] || 0) || 0));
      counts[id] = {
        id,
        name: getName(id),
        assetId: getAssetId(id),
        cost: amount,
        owned,
        missing: Math.max(0, amount - owned)
      };
      return counts;
    }, {});
  }

  function canPayCardMaterialCost(materials, cost, options) {
    return Object.values(getCardMaterialCostCounts(materials, cost, options)).every((entry) => entry.owned >= entry.cost);
  }

  function getMaxAffordableCardMaterialCost(materials, cost, options) {
    const entries = Object.values(getCardMaterialCostCounts(materials, cost, options));
    if (!entries.length) return 0;
    return entries.reduce((max, entry) => Math.min(max, Math.floor(entry.owned / Math.max(1, entry.cost))), Number.MAX_SAFE_INTEGER);
  }

  function addCardMaterialCost(target, cost) {
    const output = target || {};
    Object.entries(normalizeCardMaterialCost(cost)).forEach(([id, amount]) => {
      output[id] = Number(output[id] || 0) + amount;
    });
    return output;
  }

  function getCardMaterialCostLabel(cost, options) {
    const settings = options || {};
    const entries = Object.entries(normalizeCardMaterialCost(cost));
    const formatInteger = typeof settings.formatIntegerWithCommas === 'function'
      ? settings.formatIntegerWithCommas
      : (value) => String(Math.round(Number(value) || 0));
    const getName = typeof settings.getMaterialDisplayName === 'function'
      ? settings.getMaterialDisplayName
      : (materialId) => String(materialId || '');
    if (!entries.length) return 'No Star Cards';
    return entries
      .map(([id, amount]) => `${formatInteger(amount)} ${getName(id)}`)
      .join(', ');
  }

  function createMissingCardPreview(options) {
    const settings = options || {};
    return {
      card: null,
      definition: null,
      available: 0,
      needed: Math.max(1, Math.floor(Number(settings.needed || 1) || 1)),
      maxRank: 5,
      canUpgrade: false,
      canCombine: false,
      maxCreate: 0,
      outputRank: 0,
      cost: {},
      costEntries: [],
      missing: [],
      costLabel: '',
      reason: 'Missing card'
    };
  }

  function createCardCombinePreview(card, definition, options) {
    const settings = options || {};
    const upgradeCopyCount = Math.max(1, Math.floor(Number(settings.upgradeCopyCount || CARD_UPGRADE_COPY_COUNT) || CARD_UPGRADE_COPY_COUNT));
    if (!card || !definition) return createMissingCardPreview({ needed: upgradeCopyCount });
    const maxRank = Math.max(1, Math.floor(Number(definition.maxRank || 5) || 5));
    const available = Math.max(1, Math.floor(Number(card.quantity || 1) || 1));
    if (card.rank >= maxRank) {
      return { card, definition, available, needed: upgradeCopyCount, maxRank, canUpgrade: false, canCombine: false, maxCreate: 0, outputRank: maxRank, cost: {}, costEntries: [], missing: [], costLabel: `${upgradeCopyCount} copies`, reason: 'Max rank' };
    }
    const includeLocked = settings.includeLocked === true || settings.ignoreLocked === true;
    const maxCreate = card.locked && !includeLocked ? 0 : Math.floor(available / upgradeCopyCount);
    const outputRank = normalizeCardRank(Number(card.rank || 1) + 1, definition);
    return {
      card,
      definition,
      available,
      needed: upgradeCopyCount,
      maxRank,
      canUpgrade: maxCreate > 0,
      canCombine: maxCreate > 0,
      maxCreate,
      outputRank,
      cost: {},
      costEntries: [],
      missing: [],
      costLabel: `${upgradeCopyCount} copies`,
      reason: card.locked && !includeLocked ? 'Unlock before combining' : maxCreate > 0 ? '' : `Need ${upgradeCopyCount} cards`
    };
  }

  function createCardUpgradePreview(card, definition, materials, options) {
    const settings = options || {};
    if (!card || !definition) return createMissingCardPreview({ needed: 1 });
    const maxRank = Math.max(1, Math.floor(Number(definition.maxRank || 5) || 5));
    const available = Math.max(1, Math.floor(Number(card.quantity || 1) || 1));
    if (card.rank >= maxRank) {
      return { card, definition, available, needed: 1, maxRank, canUpgrade: false, canCombine: false, maxCreate: 0, outputRank: maxRank, cost: {}, costEntries: [], missing: [], costLabel: '', reason: 'Max rank' };
    }
    const outputRank = normalizeCardRank(Number(card.rank || 1) + 1, definition);
    const cost = typeof settings.getCardUpgradeCostForRank === 'function'
      ? settings.getCardUpgradeCostForRank(card.rank)
      : getCardUpgradeCostForRank(card.rank, settings);
    const costEntries = Object.values(typeof settings.getMaterialCostCounts === 'function'
      ? settings.getMaterialCostCounts(materials, cost)
      : getCardMaterialCostCounts(materials, cost, settings));
    const missing = costEntries.filter((entry) => entry.missing > 0);
    const maxAffordable = typeof settings.getMaxAffordableMaterialCost === 'function'
      ? settings.getMaxAffordableMaterialCost(materials, cost)
      : getMaxAffordableCardMaterialCost(materials, cost, settings);
    const maxCreate = card.locked || !costEntries.length ? 0 : Math.min(available, maxAffordable);
    const costLabel = typeof settings.getMaterialCostLabel === 'function'
      ? settings.getMaterialCostLabel(cost)
      : getCardMaterialCostLabel(cost, settings);
    const reason = card.locked
      ? 'Unlock before combining'
      : !costEntries.length
        ? 'No Star Card recipe'
        : maxCreate > 0 ? '' : `Need ${missing.map((entry) => `${entry.missing} ${entry.name}`).join(', ') || 'more Star Cards'}`;
    return {
      card,
      definition,
      available,
      needed: 1,
      maxRank,
      canUpgrade: maxCreate > 0,
      canCombine: maxCreate > 0,
      maxCreate,
      maxAffordable,
      outputRank,
      cost,
      costEntries,
      missing,
      costLabel,
      reason
    };
  }

  function createCardBulkUpgradePreview(stacks, options) {
    const settings = options || {};
    const includeLocked = settings.includeLocked === true || settings.ignoreLocked === true;
    const upgradeCopyCount = Math.max(1, Math.floor(Number(settings.upgradeCopyCount || CARD_UPGRADE_COPY_COUNT) || CARD_UPGRADE_COPY_COUNT));
    const capacity = Math.max(0, Math.floor(Number(settings.capacity || 0) || 0));
    const simulatedStacks = (Array.isArray(stacks) ? stacks : [])
      .map((source) => source && typeof source === 'object' ? {
        cardId: normalizeId(source.cardId),
        rank: Math.max(1, Math.floor(Number(source.rank || 1) || 1)),
        quantity: Math.max(1, Math.floor(Number(source.quantity || 1) || 1)),
        locked: !!source.locked,
        acquiredAt: Number(source.acquiredAt || 0),
        maxRank: Math.max(1, Math.floor(Number(source.maxRank || 5) || 5))
      } : null)
      .filter((source) => source && source.cardId);
    const affected = new Set();
    let totalCombines = 0;
    let consumedCards = 0;
    let createdCards = 0;
    let changed = true;
    let guard = 0;
    while (changed && guard < 1000) {
      guard += 1;
      changed = false;
      simulatedStacks.sort((a, b) => String(a.cardId).localeCompare(String(b.cardId)) || Number(a.rank || 1) - Number(b.rank || 1) || Number(a.acquiredAt || 0) - Number(b.acquiredAt || 0));
      for (let index = 0; index < simulatedStacks.length; index += 1) {
        const source = simulatedStacks[index];
        if (!source || source.locked && !includeLocked || source.rank >= source.maxRank) continue;
        const quantity = Math.floor(Math.max(0, Number(source.quantity || 0)) / upgradeCopyCount);
        if (quantity <= 0) continue;
        const outputRank = Math.min(source.maxRank, Number(source.rank || 1) + 1);
        const consumed = quantity * upgradeCopyCount;
        const remaining = Math.max(0, Number(source.quantity || 0) - consumed);
        const output = simulatedStacks.find((card, outputIndex) => outputIndex !== index && card && card.cardId === source.cardId && Number(card.rank || 1) === outputRank) || null;
        if (remaining > 0 && !output && simulatedStacks.length >= capacity) continue;
        affected.add(`${source.cardId}|${source.rank}`);
        totalCombines += quantity;
        consumedCards += consumed;
        createdCards += quantity;
        source.quantity = remaining;
        if (output) {
          output.quantity = Math.max(1, Math.floor(Number(output.quantity || 1) || 1)) + quantity;
          output.locked = !!(output.locked || source.locked);
        } else if (remaining <= 0) {
          source.rank = outputRank;
          source.quantity = quantity;
        } else {
          simulatedStacks.push({
            cardId: source.cardId,
            rank: outputRank,
            quantity,
            locked: !!source.locked,
            acquiredAt: Number(source.acquiredAt || Date.now()),
            maxRank: source.maxRank
          });
        }
        if (source.quantity <= 0) simulatedStacks.splice(index, 1);
        changed = true;
        break;
      }
    }
    return {
      canUpgrade: totalCombines > 0,
      canCombine: totalCombines > 0,
      totalUpgrades: totalCombines,
      totalCombines,
      consumedCards,
      createdCards,
      upgradedCards: createdCards,
      affectedStacks: Array.from(affected),
      affectedStackCount: affected.size,
      includeLocked,
      reason: totalCombines > 0 ? '' : 'No card stacks can be upgraded'
    };
  }

  function getCardUpgradeCostForRank(rank, options) {
    const settings = options || {};
    const outputRank = Math.max(2, Math.floor(Number(rank || 1) || 1) + 1);
    const costs = settings.cardStarUpgradeCosts || createCardStarUpgradeCosts(settings.materialIds);
    const normalizer = typeof settings.normalizeMaterialCost === 'function'
      ? settings.normalizeMaterialCost
      : normalizeCardMaterialCost;
    return normalizer(costs[outputRank] || {});
  }

  function createStarCardWallet(materials, options) {
    const settings = options || {};
    const source = materials && typeof materials === 'object' ? materials : {};
    const materialOrder = Array.isArray(settings.materialOrder) ? settings.materialOrder : createStarCardMaterialOrder(settings.materialIds);
    const materialDefinitions = settings.materialDefinitions || {};
    const itemAssets = settings.itemAssets || {};
    const getAssetId = typeof settings.getMaterialAssetIdForMaterial === 'function'
      ? settings.getMaterialAssetIdForMaterial
      : (materialId) => materialId;
    const formatName = typeof settings.formatMaterialName === 'function'
      ? settings.formatMaterialName
      : (materialId) => String(materialId || '');
    return materialOrder.map((materialId) => {
      const material = materialDefinitions[materialId] || {};
      const assetId = getAssetId(materialId);
      return {
        id: materialId,
        materialId,
        assetId,
        asset: itemAssets && itemAssets[assetId] || '',
        name: material.name || formatName(materialId),
        icon: material.icon || String(materialId || '').slice(0, 3).toUpperCase(),
        rarity: material.rarity || 'Common',
        count: Math.max(0, Math.floor(Number(source[materialId] || 0) || 0))
      };
    });
  }

  function getStarCardExchangeRate(materialId, options) {
    const settings = options || {};
    const rates = Array.isArray(settings.exchangeRates) ? settings.exchangeRates : createStarCardExchangeRates(settings.materialIds);
    const fromId = normalizeId(materialId);
    return rates.find((entry) => normalizeId(entry && entry.from) === fromId) || null;
  }

  function createStarCardExchangeSnapshot(wallet, options) {
    const settings = options || {};
    const sourceWallet = Array.isArray(wallet) ? wallet : [];
    const byId = new Map(sourceWallet.map((entry) => [entry.id, entry]));
    const rates = Array.isArray(settings.exchangeRates) ? settings.exchangeRates : createStarCardExchangeRates(settings.materialIds);
    const getMaterialDisplayName = typeof settings.getMaterialDisplayName === 'function'
      ? settings.getMaterialDisplayName
      : (materialId) => String(materialId || '');
    return rates.map((rate) => {
      const from = byId.get(rate.from) || { id: rate.from, name: getMaterialDisplayName(rate.from), count: 0 };
      const to = byId.get(rate.to) || { id: rate.to, name: getMaterialDisplayName(rate.to), count: 0 };
      const maxCreate = Math.floor(Math.max(0, Number(from.count || 0)) / Math.max(1, Number(rate.rate || 1)));
      return {
        from,
        to,
        rate: Math.max(1, Math.floor(Number(rate.rate || 1) || 1)),
        available: Math.max(0, Number(from.count || 0)),
        maxCreate,
        canExchange: maxCreate > 0,
        label: `${rate.rate} ${from.name} -> 1 ${to.name}`
      };
    });
  }

  function createCardDefinitionLookup(data) {
    const sourceData = data || {};
    const source = Array.isArray(sourceData.CARD_DEFINITIONS) ? sourceData.CARD_DEFINITIONS : [];
    const byId = new Map();
    const byRarity = new Map();
    for (let index = 0; index < source.length; index += 1) {
      const card = source[index];
      const id = normalizeId(card && card.id);
      if (!id) continue;
      byId.set(id, card);
      const rarity = String(card.rarity || 'Common');
      const list = byRarity.get(rarity) || [];
      list.push(card);
      byRarity.set(rarity, list);
    }
    byRarity.forEach((list, rarity) => {
      byRarity.set(rarity, Object.freeze(list.slice()));
    });
    return { source, byId, byRarity };
  }

  function getCardLookup(options) {
    const settings = options || {};
    return settings.lookup || createCardDefinitionLookup(settings.data);
  }

  function getCardDefinition(cardId, options) {
    const lookup = getCardLookup(options);
    return lookup.byId.get(normalizeId(cardId)) || null;
  }

  function getCardDefinitionsByRarity(rarity, options) {
    const lookup = getCardLookup(options);
    return lookup.byRarity.get(String(rarity || 'Common')) || [];
  }

  function normalizeCardRank(value, definition) {
    const maxRank = Math.max(1, Math.floor(Number(definition && definition.maxRank || 5) || 5));
    return clamp(Math.floor(Number(value || 1) || 1), 1, maxRank);
  }

  function resolveCardDefinition(cardId, options) {
    const settings = options || {};
    if (typeof settings.getCardDefinition === 'function') return settings.getCardDefinition(cardId);
    return getCardDefinition(cardId, settings);
  }

  function createCardInstance(cardId, options) {
    const settings = options || {};
    const definition = resolveCardDefinition(cardId, settings);
    if (!definition) return null;
    const createUid = typeof settings.createCardUid === 'function' ? settings.createCardUid : createFallbackCardUid;
    return {
      uid: normalizeId(settings.uid) || createUid(definition.id),
      cardId: definition.id,
      rank: normalizeCardRank(settings.rank, definition),
      quantity: Math.max(1, Math.floor(Number(settings.quantity || settings.count || 1) || 1)),
      locked: !!settings.locked,
      acquiredAt: Math.max(1, Math.floor(Number(settings.acquiredAt || Date.now()) || Date.now()))
    };
  }

  function getCardDisplayName(card, options) {
    const definition = resolveCardDefinition(card && card.cardId || card, options);
    return definition ? definition.name : 'Card';
  }

  function createCardDefinitionSnapshot(definition, options) {
    const settings = options || {};
    if (!definition) return null;
    const clonePlain = typeof settings.clonePlain === 'function'
      ? settings.clonePlain
      : (value) => JSON.parse(JSON.stringify(value));
    const cardAssets = settings.cardAssets || {};
    return Object.assign(clonePlain(definition), { asset: cardAssets[definition.id] || '' });
  }

  function createCardUpgradeCostEntrySnapshots(costEntries, options) {
    const settings = options || {};
    const itemAssets = settings.itemAssets || {};
    return (costEntries || []).map((entry) => Object.assign({}, entry, {
      asset: itemAssets && itemAssets[entry.assetId] || ''
    }));
  }

  function createCardCombineSnapshot(combinePreview, card, definition, quantity, options) {
    const settings = options || {};
    const preview = combinePreview || {};
    const upgradeCopyCount = Math.max(1, Math.floor(Number(settings.upgradeCopyCount || CARD_UPGRADE_COPY_COUNT) || CARD_UPGRADE_COPY_COUNT));
    const availableQuantity = Math.max(1, Math.floor(Number(quantity || card && card.quantity || 1) || 1));
    return {
      canCombine: !!preview.canCombine,
      available: Number(preview.available || availableQuantity),
      needed: Number(preview.needed || upgradeCopyCount),
      maxRank: Number(preview.maxRank || definition && definition.maxRank || 5),
      maxCreate: Number(preview.maxCreate || 0),
      outputRank: Number(preview.outputRank || card && card.rank || 1),
      cost: {},
      costEntries: [],
      missing: [],
      costLabel: preview.costLabel || `${upgradeCopyCount} copies`,
      reason: preview.reason || ''
    };
  }

  function createCardUpgradeSnapshot(upgradePreview, card, definition, quantity, costEntries, options) {
    const settings = options || {};
    const preview = upgradePreview || {};
    const clonePlain = typeof settings.clonePlain === 'function'
      ? settings.clonePlain
      : (value) => JSON.parse(JSON.stringify(value));
    const entries = Array.isArray(costEntries) ? costEntries : preview.costEntries || [];
    const availableQuantity = Math.max(1, Math.floor(Number(quantity || card && card.quantity || 1) || 1));
    return {
      canUpgrade: !!preview.canUpgrade,
      available: Number(preview.available || availableQuantity),
      needed: Number(preview.needed || 1),
      maxRank: Number(preview.maxRank || definition && definition.maxRank || 5),
      maxCreate: Number(preview.maxCreate || 0),
      outputRank: Number(preview.outputRank || card && card.rank || 1),
      cost: clonePlain(preview.cost || {}),
      costEntries: entries,
      missing: entries.filter((entry) => Number(entry.missing || 0) > 0),
      costLabel: preview.costLabel || '',
      reason: preview.reason || ''
    };
  }

  function createCardDeckSlots(value, options) {
    const settings = options || {};
    const slotCount = Math.max(1, Math.floor(Number(settings.deckSlotCount || CARD_DECK_SLOT_COUNT) || CARD_DECK_SLOT_COUNT));
    const source = value && typeof value === 'object' && Array.isArray(value.slots)
      ? value.slots
      : Array.isArray(value) ? value : [];
    return Array.from({ length: slotCount }, (_, index) => normalizeId(source[index]));
  }

  function getDefaultCardDeckId(index) {
    return `deck_${Math.max(1, Math.floor(Number(index || 0) || 0) + 1)}`;
  }

  function createCardDeckState(value, index, uidMap, byUid, options) {
    const source = value && typeof value === 'object' ? value : {};
    const fallbackId = getDefaultCardDeckId(index);
    const id = normalizeId(source.id) || fallbackId;
    const name = String(source.name || `Deck ${Math.max(1, Math.floor(Number(index || 0) || 0) + 1)}`);
    const seenCardIds = new Set();
    const slots = createCardDeckSlots(source, options).map((uid) => {
      uid = uidMap.get(normalizeId(uid)) || uid;
      const card = byUid.get(normalizeId(uid));
      if (!card || seenCardIds.has(card.cardId)) return '';
      seenCardIds.add(card.cardId);
      return card.uid;
    });
    return { id, name, slots };
  }

  function getCardCollection(cards) {
    if (!cards || typeof cards !== 'object') return [];
    if (Array.isArray(cards.collection)) return cards.collection;
    if (Array.isArray(cards.inventory)) return cards.inventory;
    return [];
  }

  function getCardDecks(cards) {
    if (!cards || typeof cards !== 'object') return [];
    if (Array.isArray(cards.decks)) return cards.decks;
    if (cards.deck && typeof cards.deck === 'object') return [cards.deck];
    return [];
  }

  function createCardDeckUidSet(deck) {
    return new Set((deck && deck.slots || []).map(normalizeId).filter(Boolean));
  }

  function createAllCardDeckUidSet(cards) {
    return getCardDecks(cards).reduce((uids, deck) => {
      (deck && deck.slots || []).forEach((uid) => {
        const id = normalizeId(uid);
        if (id) uids.add(id);
      });
      return uids;
    }, new Set());
  }

  function createCardDeckIdsByUid(cards) {
    return getCardDecks(cards).reduce((map, deck) => {
      const deckId = normalizeId(deck && deck.id);
      (deck && deck.slots || []).forEach((uid) => {
        const id = normalizeId(uid);
        if (!id) return;
        if (!map.has(id)) map.set(id, []);
        map.get(id).push(deckId);
      });
      return map;
    }, new Map());
  }

  function createCardDeckSnapshot(deck, cardsByUid, options) {
    const settings = options || {};
    const byUid = cardsByUid && typeof cardsByUid.get === 'function' ? cardsByUid : new Map();
    const activeDeckId = normalizeId(settings.activeDeckId);
    const slotCount = Math.max(1, Math.floor(Number(settings.deckSlotCount || CARD_DECK_SLOT_COUNT) || CARD_DECK_SLOT_COUNT));
    const cloneStats = typeof settings.cloneStats === 'function'
      ? settings.cloneStats
      : (stats) => Object.assign({}, stats || {});
    const id = normalizeId(deck && deck.id);
    return {
      id,
      name: deck && deck.name || 'Deck',
      active: id === activeDeckId,
      slots: Array.from({ length: slotCount }, (_, index) => {
        const uid = deck && deck.slots ? normalizeId(deck.slots[index]) : '';
        const card = byUid.get(uid);
        return card ? {
          index,
          uid,
          card,
          stats: cloneStats(card.stats)
        } : { index, uid: '', card: null, stats: {} };
      })
    };
  }

  function replaceCardDeckReferences(cards, sourceUid, targetUid, options) {
    const settings = options || {};
    const sourceId = normalizeId(sourceUid);
    const targetId = normalizeId(targetUid);
    if (!sourceId) return false;
    const decks = Array.isArray(cards) ? cards : getCardDecks(cards);
    let changed = false;
    decks.forEach((deck) => {
      const slots = createCardDeckSlots(deck, settings);
      const next = slots.map((slotUid) => normalizeId(slotUid) === sourceId ? targetId : slotUid);
      if (next.join('|') !== slots.join('|')) {
        deck.slots = next;
        changed = true;
      }
    });
    return changed;
  }

  function sortCardInventorySlotIds(ids, cards, sortId, options) {
    const settings = options || {};
    const mode = String(sortId || 'value');
    const collection = Array.isArray(cards) ? cards : getCardCollection(cards);
    const byUid = new Map(collection.map((card) => [normalizeId(card && card.uid), card]));
    const resolveDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : (cardId) => getCardDefinition(cardId, settings);
    const getRarityRank = typeof settings.getRarityRank === 'function'
      ? settings.getRarityRank
      : () => 0;
    return (Array.isArray(ids) ? ids : [])
      .map((rawId, index) => {
        const id = normalizeId(rawId);
        const card = byUid.get(id) || {};
        const definition = resolveDefinition(card.cardId) || {};
        return {
          id,
          index,
          name: definition.name || id,
          rarity: getRarityRank(definition.rarity),
          rank: Math.max(1, Number(card.rank || 1)),
          acquiredAt: Number(card.acquiredAt || 0)
        };
      })
      .sort((a, b) => {
        if (mode === 'newest') return b.acquiredAt - a.acquiredAt || b.rank - a.rank || a.name.localeCompare(b.name) || a.index - b.index;
        if (mode === 'name') return a.name.localeCompare(b.name) || b.rank - a.rank || a.index - b.index;
        return b.rarity - a.rarity || b.rank - a.rank || a.name.localeCompare(b.name) || a.index - b.index;
      })
      .map((entry) => entry.id);
  }

  function createCardUidLookup(cards, options) {
    const settings = options || {};
    const collection = Array.isArray(cards) ? cards : getCardCollection(cards);
    const resolveDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : (cardId) => getCardDefinition(cardId, settings);
    const byUid = new Map();
    const byStack = new Map();
    for (let index = 0; index < collection.length; index += 1) {
      const card = collection[index];
      const id = normalizeId(card && card.uid);
      if (id) byUid.set(id, card);
      const cardId = normalizeId(card && card.cardId);
      if (cardId) {
        const definition = resolveDefinition(cardId);
        const rank = normalizeCardRank(card && card.rank, definition);
        const stackKey = `${cardId}|${rank}`;
        if (!byStack.has(stackKey)) byStack.set(stackKey, card);
      }
    }
    return {
      collection,
      length: collection.length,
      revision: Math.max(0, Number(settings.revision || 0) || 0),
      byUid,
      byStack
    };
  }

  function getActiveCardDeck(cards, deckId) {
    const decks = getCardDecks(cards);
    if (!decks.length) return null;
    const requestedId = normalizeId(deckId || cards && cards.activeDeckId);
    return decks.find((deck) => normalizeId(deck && deck.id) === requestedId) || decks[0] || null;
  }

  function createFallbackCardUid(cardId) {
    return `card_${normalizeId(cardId) || 'unknown'}_${Date.now().toString(36)}`;
  }

  function normalizeCardInstance(card, index, options) {
    const settings = options || {};
    const source = card && typeof card === 'object' ? card : {};
    const definition = resolveCardDefinition(source.cardId || source.id, settings);
    if (!definition) return null;
    const createUid = typeof settings.createCardUid === 'function' ? settings.createCardUid : createFallbackCardUid;
    const normalized = {
      uid: normalizeId(source.uid) || createUid(definition.id),
      cardId: definition.id,
      rank: normalizeCardRank(source.rank, definition),
      quantity: Math.max(1, Math.floor(Number(source.quantity || source.count || 1) || 1)),
      locked: !!source.locked,
      acquiredAt: Math.max(1, Math.floor(Number(source.acquiredAt || Date.now() + Number(index || 0)) || Date.now()))
    };
    if (!settings.preserveReferences) return normalized;
    Object.assign(source, normalized);
    return source;
  }

  function createCardState(value, options) {
    const settings = options || {};
    const source = value && typeof value === 'object' ? value : {};
    const rawCollection = Array.isArray(source.collection)
      ? source.collection
      : Array.isArray(source.inventory)
        ? source.inventory
        : Array.isArray(source.cards) ? source.cards : [];
    const uidMap = new Map();
    const collection = rawCollection
      .map((card, index) => normalizeCardInstance(card, index, settings))
      .filter(Boolean)
      .reduce((stacks, card) => {
        const key = `${card.cardId}|${Number(card.rank || 1)}`;
        const existing = stacks.find((entry) => `${entry.cardId}|${Number(entry.rank || 1)}` === key);
        if (existing) {
          existing.quantity = Math.max(1, Math.floor(Number(existing.quantity || 1) || 1)) + Math.max(1, Math.floor(Number(card.quantity || 1) || 1));
          existing.locked = !!(existing.locked || card.locked);
          existing.acquiredAt = Math.min(Number(existing.acquiredAt || card.acquiredAt), Number(card.acquiredAt || existing.acquiredAt));
          uidMap.set(normalizeId(card.uid), existing.uid);
        } else {
          stacks.push(card);
          uidMap.set(normalizeId(card.uid), card.uid);
        }
        return stacks;
      }, []);
    const byUid = new Map(collection.map((card) => [normalizeId(card.uid), card]));
    const rawDecks = Array.isArray(source.decks) && source.decks.length
      ? source.decks
      : source.deck && typeof source.deck === 'object'
        ? [source.deck]
        : [];
    const deckCount = Math.max(1, Math.floor(Number(settings.deckCount || CARD_DECK_COUNT) || CARD_DECK_COUNT));
    const decks = Array.from({ length: deckCount }, (_, index) => {
      const deck = createCardDeckState(rawDecks[index], index, uidMap, byUid, settings);
      const defaultId = getDefaultCardDeckId(index);
      deck.id = normalizeId(deck.id) || defaultId;
      return deck;
    });
    const usedDeckIds = new Set();
    decks.forEach((deck, index) => {
      const fallbackId = getDefaultCardDeckId(index);
      let id = normalizeId(deck.id) || fallbackId;
      if (usedDeckIds.has(id)) id = fallbackId;
      while (usedDeckIds.has(id)) id = `${fallbackId}_${usedDeckIds.size + 1}`;
      deck.id = id;
      usedDeckIds.add(id);
    });
    const requestedActiveDeckId = normalizeId(source.activeDeckId);
    const activeDeck = decks.find((deck) => normalizeId(deck.id) === requestedActiveDeckId) || decks[0] || null;
    const mappedSelectedUid = uidMap.get(normalizeId(source.selectedUid)) || normalizeId(source.selectedUid);
    const selectedUid = byUid.has(mappedSelectedUid) ? mappedSelectedUid : '';
    return {
      collection,
      inventory: collection,
      decks,
      activeDeckId: activeDeck ? activeDeck.id : getDefaultCardDeckId(0),
      deck: activeDeck || { id: getDefaultCardDeckId(0), name: 'Deck 1', slots: createCardDeckSlots(null, settings) },
      selectedUid
    };
  }

  function getCardInstanceStats(card, options) {
    const definition = resolveCardDefinition(card && card.cardId, options);
    if (!definition) return {};
    const rank = normalizeCardRank(card && card.rank, definition);
    const scale = 1 + Math.max(0, rank - 1) * Math.max(0, Number(definition.rankScale == null ? 0.35 : definition.rankScale) || 0);
    return Object.entries(definition.baseStats || {}).reduce((stats, [key, value]) => {
      const amount = Math.round(Number(value || 0) * scale);
      if (amount) stats[key] = amount;
      return stats;
    }, {});
  }

  function getCardSellValueForRank(card, definition, options) {
    const settings = options || {};
    const values = settings.sellRarityValues || CARD_SELL_RARITY_VALUES;
    const rarity = definition && definition.rarity || 'Common';
    const base = values[rarity] || values.Common;
    const rank = normalizeCardRank(card && card.rank, definition);
    return Math.max(1, Math.round(base * (1 + Math.max(0, rank - 1) * 0.6)));
  }

  function createCardSellPreview(card, definition, options) {
    const settings = options || {};
    if (!card || !definition) return { canSell: false, count: 0, coins: 0, maxQuantity: 0, valueEach: 0, reason: 'Missing card' };
    const quantity = Math.max(1, Math.floor(Number(card.quantity || 1) || 1));
    const valueEach = typeof settings.getCardSellValue === 'function'
      ? settings.getCardSellValue(card, definition)
      : getCardSellValueForRank(card, definition, settings);
    const inDeck = typeof settings.isCardInAnyDeck === 'function'
      ? settings.isCardInAnyDeck(card.uid)
      : !!settings.inDeck;
    const canSell = !card.locked && !inDeck;
    return {
      canSell,
      count: quantity,
      coins: valueEach * quantity,
      maxQuantity: canSell ? quantity : 0,
      valueEach,
      reason: card.locked ? 'Unlock before selling' : inDeck ? 'Remove from every deck before selling' : ''
    };
  }

  function getEquippedCardStatsFromState(state, options) {
    const settings = options || {};
    const cards = createCardState(state && state.cards, settings);
    const byUid = new Map(getCardCollection(cards).map((card) => [normalizeId(card.uid), card]));
    const deck = getActiveCardDeck(cards);
    return (deck && deck.slots || []).reduce((stats, uid) => {
      const card = byUid.get(normalizeId(uid));
      if (!card) return stats;
      const cardStats = getCardInstanceStats(card, settings);
      if (typeof settings.addStats === 'function') settings.addStats(stats, cardStats);
      else {
        Object.entries(cardStats).forEach(([key, value]) => {
          const amount = Math.round(Number(value || 0) || 0);
          if (amount) stats[key] = (stats[key] || 0) + amount;
        });
      }
      return stats;
    }, {});
  }

  const api = {
    CARD_DECK_COUNT,
    CARD_DECK_SLOT_COUNT,
    CARD_UPGRADE_COPY_COUNT,
    CARD_SELL_RARITY_VALUES,
    DEFAULT_STAR_CARD_MATERIAL_IDS,
    createStarCardMaterialIds,
    createStarCardMaterialOrder,
    createCardStarUpgradeCosts,
    createStarCardExchangeRates,
    normalizeCardMaterialCost,
    multiplyCardMaterialCost,
    getCardMaterialCostCounts,
    canPayCardMaterialCost,
    getMaxAffordableCardMaterialCost,
    addCardMaterialCost,
    getCardMaterialCostLabel,
    createMissingCardPreview,
    createCardCombinePreview,
    createCardUpgradePreview,
    createCardBulkUpgradePreview,
    getCardUpgradeCostForRank,
    createStarCardWallet,
    getStarCardExchangeRate,
    createStarCardExchangeSnapshot,
    createCardDefinitionLookup,
    getCardDefinition,
    getCardDefinitionsByRarity,
    normalizeCardRank,
    createCardInstance,
    getCardDisplayName,
    createCardDefinitionSnapshot,
    createCardUpgradeCostEntrySnapshots,
    createCardCombineSnapshot,
    createCardUpgradeSnapshot,
    normalizeCardInstance,
    createCardDeckSlots,
    getDefaultCardDeckId,
    createCardDeckState,
    getCardCollection,
    getCardDecks,
    createCardDeckUidSet,
    createAllCardDeckUidSet,
    createCardDeckIdsByUid,
    createCardDeckSnapshot,
    replaceCardDeckReferences,
    sortCardInventorySlotIds,
    createCardUidLookup,
    getActiveCardDeck,
    createCardState,
    getCardInstanceStats,
    getCardSellValueForRank,
    createCardSellPreview,
    getEquippedCardStatsFromState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.cards = Object.assign({}, modules.cards || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
