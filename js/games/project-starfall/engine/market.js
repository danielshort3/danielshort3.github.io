(function initProjectStarfallEngineMarket(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getMarketData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createMarketState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const purchasedListingIds = Array.isArray(source.purchasedListingIds)
      ? source.purchasedListingIds.map(normalizeId).filter(Boolean)
      : [];
    return { purchasedListingIds: Array.from(new Set(purchasedListingIds)) };
  }

  function createMarketSnapshot(market, options) {
    const data = getMarketData(options);
    const state = createMarketState(market);
    const purchased = new Set(state.purchasedListingIds || []);
    return {
      listings: (data.MARKET_LISTINGS || []).map((listing) => Object.assign({}, listing, {
        purchased: purchased.has(listing.id),
        lockedReason: listing.once && purchased.has(listing.id) ? 'Already purchased.' : ''
      }))
    };
  }

  function createMarketPurchasePlan(listing, market, playerCurrency) {
    if (!listing) return { ok: false, reason: 'missing' };
    const state = createMarketState(market);
    if (listing.once && state.purchasedListingIds.includes(listing.id)) {
      return {
        ok: false,
        reason: 'purchased',
        toast: `${listing.name} is already purchased.`
      };
    }
    const cost = Math.max(0, Number(listing.cost || 0));
    if (playerCurrency < cost) {
      return {
        ok: false,
        reason: 'currency',
        cost,
        toast: 'Not enough coins.'
      };
    }
    return {
      ok: true,
      listingId: listing.id,
      cost,
      reward: listing.reward,
      shouldRecordPurchase: !!listing.once,
      toast: `Bought ${listing.name}.`
    };
  }

  const api = {
    createMarketState,
    createMarketSnapshot,
    createMarketPurchasePlan
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.market = Object.assign({}, modules.market || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
