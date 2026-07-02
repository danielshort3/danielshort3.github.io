(function initProjectStarfallEngineCashShop(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getCashShopData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getCashShopWeekId(nowMs) {
    const ms = Number.isFinite(nowMs) ? nowMs : Date.now();
    return Math.floor(Math.max(0, ms) / (7 * 24 * 60 * 60 * 1000));
  }

  function createCashShopState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const purchasedItemIds = Array.isArray(source.purchasedItemIds)
      ? source.purchasedItemIds.map(normalizeId).filter(Boolean)
      : [];
    const purchaseCounts = {};
    Object.entries(source.purchaseCountsByWeek || {}).forEach(([itemId, count]) => {
      const key = normalizeId(itemId);
      if (key) purchaseCounts[key] = Math.max(0, Math.floor(Number(count || 0) || 0));
    });
    const currentWeekId = getCashShopWeekId(options && options.nowMs);
    const savedWeekId = Number.isFinite(Number(source.purchaseWeekId)) ? Math.floor(Number(source.purchaseWeekId)) : currentWeekId;
    return {
      starTokens: Math.max(0, Math.floor(Number(source.starTokens || 0) || 0)),
      purchasedItemIds: Array.from(new Set(purchasedItemIds)),
      purchaseWeekId: currentWeekId,
      purchaseCountsByWeek: savedWeekId === currentWeekId ? purchaseCounts : {}
    };
  }

  function syncCashShopPurchaseWeek(shop, options) {
    const source = shop && typeof shop === 'object' ? shop : {};
    const currentWeekId = getCashShopWeekId(options && options.nowMs);
    if (Number(source.purchaseWeekId || 0) !== currentWeekId) {
      source.purchaseWeekId = currentWeekId;
      source.purchaseCountsByWeek = {};
    } else if (!source.purchaseCountsByWeek || typeof source.purchaseCountsByWeek !== 'object') {
      source.purchaseCountsByWeek = {};
    }
    return currentWeekId;
  }

  function getCashShopPurchaseCount(itemId, shop, options) {
    const source = shop && typeof shop === 'object' ? shop : {};
    syncCashShopPurchaseWeek(source, options);
    const id = normalizeId(itemId);
    return id ? Math.max(0, Math.floor(Number(source.purchaseCountsByWeek[id] || 0) || 0)) : 0;
  }

  function getCashShopSnapshotCacheKey(shop, cosmeticSnapshotKey, options) {
    const data = getCashShopData(options);
    const source = shop && typeof shop === 'object' ? shop : {};
    const counts = source.purchaseCountsByWeek && typeof source.purchaseCountsByWeek === 'object' ? source.purchaseCountsByWeek : {};
    const countKey = Object.keys(counts).sort().map((id) => `${normalizeId(id)}:${Math.max(0, Math.floor(Number(counts[id] || 0) || 0))}`).join(',');
    const purchased = Array.isArray(source.purchasedItemIds)
      ? source.purchasedItemIds.map(normalizeId).filter(Boolean).sort().join(',')
      : '';
    const cosmeticKey = Array.isArray(cosmeticSnapshotKey) ? cosmeticSnapshotKey.slice(2).join('|') : '';
    return [
      data.CASH_SHOP_CATEGORIES || [],
      (data.CASH_SHOP_CATEGORIES || []).length,
      data.CASH_SHOP_ITEMS || [],
      (data.CASH_SHOP_ITEMS || []).length,
      Math.max(0, Math.floor(Number(source.starTokens || 0) || 0)),
      Math.floor(Number(source.purchaseWeekId || 0) || 0),
      countKey,
      purchased,
      cosmeticKey
    ];
  }

  function createCashShopSnapshot(shop, cosmeticSnapshot, options) {
    const data = getCashShopData(options);
    const source = shop && typeof shop === 'object' ? shop : {};
    const cosmetics = cosmeticSnapshot && Array.isArray(cosmeticSnapshot.cosmetics) ? cosmeticSnapshot.cosmetics : [];
    const cosmeticById = cosmetics.reduce((map, cosmetic) => {
      map[cosmetic.id] = cosmetic;
      return map;
    }, {});
    return {
      currencyId: 'starTokens',
      currencyName: 'Star Tokens',
      balance: Math.max(0, Math.floor(Number(source.starTokens || 0) || 0)),
      categories: (data.CASH_SHOP_CATEGORIES || []).map((category) => Object.assign({}, category)),
      items: (data.CASH_SHOP_ITEMS || []).map((item) => {
        const cosmetic = item.cosmeticId ? cosmeticById[item.cosmeticId] : null;
        const purchaseCount = getCashShopPurchaseCount(item.id, source, options);
        const weeklyLimit = Math.max(0, Math.floor(Number(item.weeklyLimit || 0) || 0));
        const owned = !!(cosmetic && cosmetic.unlocked);
        const equipped = !!(cosmetic && cosmetic.equipped);
        let lockedReason = '';
        if (weeklyLimit && purchaseCount >= weeklyLimit) lockedReason = 'Weekly limit reached.';
        if (item.kind === 'cosmetic' && owned) lockedReason = '';
        return Object.assign({}, item, {
          owned,
          equipped,
          cosmetic,
          purchaseCount,
          remainingPurchases: weeklyLimit ? Math.max(0, weeklyLimit - purchaseCount) : null,
          lockedReason,
          earnableInGame: !!((item.tags || []).includes('Earnable In Game') || (item.earnableSources || []).length)
        });
      })
    };
  }

  const api = {
    getCashShopWeekId,
    createCashShopState,
    syncCashShopPurchaseWeek,
    getCashShopPurchaseCount,
    getCashShopSnapshotCacheKey,
    createCashShopSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.cashShop = Object.assign({}, modules.cashShop || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
