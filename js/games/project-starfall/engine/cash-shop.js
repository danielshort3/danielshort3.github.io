(function initProjectStarfallEngineCashShop(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const DAY_MS = 24 * 60 * 60 * 1000;
  const WEEK_MS = 7 * DAY_MS;
  const MONDAY_EPOCH_OFFSET_MS = 3 * DAY_MS;
  const CASH_SHOP_WEEK_SCHEMA = 2;

  function getCashShopData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createCashShopWeekWindow(weekId) {
    const normalizedWeekId = Math.max(0, Math.floor(Number(weekId) || 0));
    const startedAt = normalizedWeekId * WEEK_MS - MONDAY_EPOCH_OFFSET_MS;
    return {
      weekId: normalizedWeekId,
      startedAt,
      endsAt: startedAt + WEEK_MS,
      resetDayUtc: 1,
      resetHourUtc: 0
    };
  }

  function getCashShopWeekWindow(nowMs) {
    const ms = Number.isFinite(Number(nowMs)) ? Math.max(0, Number(nowMs)) : Date.now();
    const weekId = Math.floor((ms + MONDAY_EPOCH_OFFSET_MS) / WEEK_MS);
    return createCashShopWeekWindow(weekId);
  }

  function getCashShopWeekId(nowMs) {
    return getCashShopWeekWindow(nowMs).weekId;
  }

  function getLegacyCashShopWeekId(nowMs) {
    const ms = Number.isFinite(Number(nowMs)) ? Math.max(0, Number(nowMs)) : Date.now();
    return Math.floor(ms / WEEK_MS);
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
    const nowMs = options && options.nowMs;
    const currentWeekId = getCashShopWeekId(nowMs);
    const hasSavedWeekId = Number.isFinite(Number(source.purchaseWeekId));
    const savedWeekId = hasSavedWeekId ? Math.max(0, Math.floor(Number(source.purchaseWeekId))) : currentWeekId;
    const savedSchema = Math.max(0, Math.floor(Number(source.purchaseWeekSchema || 0) || 0));
    const cycleAwareState = savedSchema >= CASH_SHOP_WEEK_SCHEMA;
    const effectiveWeekId = cycleAwareState ? Math.max(savedWeekId, currentWeekId) : currentWeekId;
    const preserveCounts = !hasSavedWeekId || (cycleAwareState
      ? savedWeekId >= currentWeekId
      : savedWeekId === getLegacyCashShopWeekId(nowMs));
    return {
      starTokens: Math.max(0, Math.floor(Number(source.starTokens || 0) || 0)),
      purchasedItemIds: Array.from(new Set(purchasedItemIds)),
      purchaseWeekSchema: CASH_SHOP_WEEK_SCHEMA,
      purchaseWeekId: effectiveWeekId,
      purchaseCountsByWeek: preserveCounts ? purchaseCounts : {}
    };
  }

  function syncCashShopPurchaseWeek(shop, options) {
    const source = shop && typeof shop === 'object' ? shop : {};
    const nowMs = options && options.nowMs;
    const currentWeekId = getCashShopWeekId(nowMs);
    const hasSavedWeekId = Number.isFinite(Number(source.purchaseWeekId));
    const savedWeekId = hasSavedWeekId
      ? Math.max(0, Math.floor(Number(source.purchaseWeekId)))
      : currentWeekId;
    const savedSchema = Math.max(0, Math.floor(Number(source.purchaseWeekSchema || 0) || 0));
    const legacyCurrent = savedSchema < CASH_SHOP_WEEK_SCHEMA &&
      (!hasSavedWeekId || savedWeekId === getLegacyCashShopWeekId(nowMs));
    if (savedSchema < CASH_SHOP_WEEK_SCHEMA) {
      source.purchaseWeekSchema = CASH_SHOP_WEEK_SCHEMA;
      source.purchaseWeekId = currentWeekId;
      source.purchaseCountsByWeek = legacyCurrent && source.purchaseCountsByWeek && typeof source.purchaseCountsByWeek === 'object'
        ? source.purchaseCountsByWeek
        : {};
    } else if (currentWeekId > savedWeekId) {
      source.purchaseWeekId = currentWeekId;
      source.purchaseCountsByWeek = {};
    } else if (!source.purchaseCountsByWeek || typeof source.purchaseCountsByWeek !== 'object') {
      source.purchaseCountsByWeek = {};
    }
    return Math.max(currentWeekId, Math.floor(Number(source.purchaseWeekId || 0) || 0));
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
      Math.floor(Number(source.purchaseWeekSchema || 0) || 0),
      Math.floor(Number(source.purchaseWeekId || 0) || 0),
      countKey,
      purchased,
      cosmeticKey
    ];
  }

  function createCashShopSnapshot(shop, cosmeticSnapshot, options) {
    const data = getCashShopData(options);
    const source = shop && typeof shop === 'object' ? shop : {};
    const nowMs = options && options.nowMs;
    syncCashShopPurchaseWeek(source, { nowMs });
    const week = createCashShopWeekWindow(source.purchaseWeekId);
    const cosmetics = cosmeticSnapshot && Array.isArray(cosmeticSnapshot.cosmetics) ? cosmeticSnapshot.cosmetics : [];
    const cosmeticById = cosmetics.reduce((map, cosmetic) => {
      map[cosmetic.id] = cosmetic;
      return map;
    }, {});
    return {
      currencyId: 'starTokens',
      currencyName: 'Star Tokens',
      balance: Math.max(0, Math.floor(Number(source.starTokens || 0) || 0)),
      cadenceLabel: 'Weekly purchase limits',
      resetAt: week.endsAt,
      resetScheduleLabel: 'Monday 00:00 UTC',
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
    getCashShopWeekWindow,
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
