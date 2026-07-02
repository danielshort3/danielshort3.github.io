(function initProjectStarfallUiShop(global) {
  'use strict';

  function getShopPanelDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const vendorEntryId = getAttribute('data-starfall-shop-buy-entry');
    if (vendorEntryId) return { handled: true, type: 'buyVendorEntry', entryId: vendorEntryId };
    const sellItemUid = getAttribute('data-starfall-shop-sell-item');
    if (sellItemUid) return { handled: true, type: 'sellItem', uid: sellItemUid };
    const sellBulkMode = getAttribute('data-starfall-shop-sell-bulk');
    if (sellBulkMode) return { handled: true, type: 'sellBulk', mode: sellBulkMode };
    const buyItemId = getAttribute('data-starfall-buy');
    if (buyItemId) return { handled: true, type: 'buyItem', itemId: buyItemId };
    const cashShopItemId = getAttribute('data-starfall-cash-shop-buy');
    if (cashShopItemId) return { handled: true, type: 'buyCashShopItem', itemId: cashShopItemId };
    return { handled: false, type: '' };
  }

  function getShopPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'shop-buy-entry') return { handled: true, type: 'buyVendorEntry', entryId: source.entryId };
    if (source.type === 'shop-sell-item') return { handled: true, type: 'sellItem', uid: source.uid };
    if (source.type === 'shop-sell-bulk') return { handled: true, type: 'sellBulk', mode: source.mode };
    if (source.type === 'buy-item') return { handled: true, type: 'buyItem', itemId: source.itemId };
    if (source.type === 'cash-shop-buy') return { handled: true, type: 'buyCashShopItem', itemId: source.itemId };
    if (source.type === 'market-buy') return { handled: true, type: 'buyMarketListing', listingId: source.listingId };
    if (source.type === 'cosmetic-buy') return { handled: true, type: 'buyCosmetic', cosmeticId: source.cosmeticId };
    if (source.type === 'cosmetic-equip') return { handled: true, type: 'equipCosmetic', cosmeticId: source.cosmeticId };
    if (source.type === 'season-claim') return { handled: true, type: 'claimSeasonReward' };
    return { handled: false, type: '' };
  }

  function createShopPanelUiHelpers() {
    return Object.freeze({
      getShopPanelDomAction,
      getShopPanelRegionAction
    });
  }

  const api = {
    createShopPanelUiHelpers,
    getShopPanelDomAction,
    getShopPanelRegionAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.shop = Object.assign({}, modules.shop || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
