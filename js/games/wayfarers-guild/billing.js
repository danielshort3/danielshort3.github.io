(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersBilling = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PRODUCTS = [
    { productId: 'wayfarers_starshards_20', shards: 20, title: 'Pocket of Starshards' },
    { productId: 'wayfarers_starshards_5', shards: 5, title: 'Few Starshards' },
    { productId: 'wayfarers_starshards_1', shards: 1, title: 'One Starshard' }
  ];
  const ITEMS = ['compass', 'artisan', 'scholar', 'banner-amber', 'banner-moon'];
  const ITEM_COSTS = { compass: 10, artisan: 10, scholar: 10, 'banner-amber': 5, 'banner-moon': 5 };
  const WEB_MESSAGE = 'Currency packs are available through Google Play in the Android app. You can earn and spend Starshards here without purchases.';
  const validWallet = wallet => wallet && typeof wallet === 'object' && /^[a-f0-9]{64}$/.test(wallet.walletId) &&
    ['balance', 'debt', 'revision'].every(key => Number.isSafeInteger(wallet[key]) && wallet[key] >= 0) &&
    Array.isArray(wallet.owned) && wallet.owned.every(id => ITEMS.includes(id)) && new Set(wallet.owned).size === wallet.owned.length;

  function getPurchaseBudget(context) {
    const value = context || {};
    const owned = new Set(Array.isArray(value.owned) ? value.owned : []);
    const prices = ITEMS.filter(id => !owned.has(id)).map(id => ITEM_COSTS[id]);
    const remainingCost = prices.reduce((sum, price) => sum + price, 0);
    const balance = key => Number.isFinite(value[key]) ? Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, value[key])) : 0;
    // A keepsake uses one wallet in full. Find the best whole-item allocation
    // to earned shards rather than pretending two partial balances can mix.
    let usableEarned = 0;
    for (let mask = 0; mask < (1 << prices.length); mask += 1) {
      const spent = prices.reduce((sum, price, index) => sum + (mask & (1 << index) ? price : 0), 0);
      if (spent <= balance('earnedBalance')) usableEarned = Math.max(usableEarned, spent);
    }
    const debt = balance('debt');
    const catalogShortfall = debt && value.retainedKnown === false ? 0 : Math.max(0, remainingCost - usableEarned - balance('paidBalance'));
    const shortfall = debt + catalogShortfall;
    const reason = debt ? 'New paid shards first settle ' + debt + ' refunded shards. Remaining useful top-up: ' + shortfall + '.' : remainingCost === 0 ? 'You own the entire catalog.' : shortfall === 0 ? 'Your existing wallets can cover every remaining keepsake.' : 'Keepsakes use one wallet in full. Useful paid top-up: ' + shortfall + ' shards.';
    return { remainingCost, shortfall, debt, usableEarned, earnedUnallocated: Math.max(0, balance('earnedBalance') - usableEarned), reason, availableProducts: PRODUCTS.filter(product => product.shards <= shortfall).map(product => product.productId) };
  }

  function createClient(options) {
    const settings = options || {};
    const host = settings.root || globalThis;
    const listeners = new Set();
    const pending = new Map();
    let sequence = 0;
    let walletSequence = 0;
    let destroyed = false;
    let catalogContext = { owned: [], earnedBalance: 0 };
    let snapshot = { native: false, hydrated: false, freePlaySafe: false, available: false, configured: false, pending: false, pendingCount: 0, walletId: '', revision: -1, balance: 0, debt: 0, owned: [], catalogOwned: [], retainedKnown: false, products: PRODUCTS.map(item => Object.assign({}, item, { available: false, price: '' })), message: WEB_MESSAGE };
    const native = () => !destroyed && host.WayfarersPlayBilling && typeof host.WayfarersPlayBilling.postMessage === 'function';
    const copy = value => JSON.parse(JSON.stringify(value));
    function catalogSnapshot() {
      const next = copy(snapshot);
      next.budget = getPurchaseBudget({ owned: [...catalogContext.owned, ...snapshot.catalogOwned], earnedBalance: catalogContext.earnedBalance, paidBalance: snapshot.balance, debt: snapshot.debt, retainedKnown: snapshot.retainedKnown });
      next.products.forEach(product => {
        product.budgetAvailable = next.budget.availableProducts.includes(product.productId) && !snapshot.pending;
        product.blockedReason = snapshot.pending ? 'Finish or restore the pending Google Play purchase before buying another pack.' : product.budgetAvailable ? '' : next.budget.reason;
        product.available = product.available && product.budgetAvailable && snapshot.hydrated;
      });
      return next;
    }
    function publish(data) {
      if (!data || typeof data !== 'object' || destroyed) return;
      const wallet = data.wallet && typeof data.wallet === 'object' ? data.wallet : data;
      const next = Object.assign({}, snapshot, { native: !!native() });
      if (typeof data.available === 'boolean') next.available = data.available;
      if (typeof data.configured === 'boolean') next.configured = data.configured;
      if (typeof data.freePlaySafe === 'boolean') next.freePlaySafe = data.freePlaySafe;
      if (typeof data.pending === 'boolean') next.pending = data.pending;
      if (Number.isSafeInteger(data.pendingCount) && data.pendingCount >= 0) next.pendingCount = data.pendingCount;
      if (validWallet(wallet)) {
        next.hydrated = true;
        next.balance = wallet.balance;
        next.debt = wallet.debt;
        next.walletId = wallet.walletId;
        next.revision = wallet.revision;
        next.owned = wallet.debt ? [] : wallet.owned.slice();
        next.retainedKnown = Array.isArray(wallet.catalogOwned) && wallet.catalogOwned.every(id => ITEMS.includes(id)) && new Set(wallet.catalogOwned).size === wallet.catalogOwned.length && wallet.owned.every(id => wallet.catalogOwned.includes(id));
        next.catalogOwned = next.retainedKnown ? wallet.catalogOwned.slice() : wallet.owned.slice();
      }
      if (typeof data.message === 'string') next.message = data.message.slice(0, 500);
      if (Array.isArray(data.products)) next.products = PRODUCTS.map(item => {
        const product = data.products.find(entry => entry && entry.productId === item.productId);
        return Object.assign({}, item, { available: !!(product && product.available !== false && typeof product.price === 'string' && product.price), price: product && typeof product.price === 'string' ? product.price.slice(0, 80) : '' });
      });
      snapshot = next;
      listeners.forEach(listener => listener(catalogSnapshot()));
    }
    function call(method, args) {
      if (!native()) return Promise.reject(new Error(WEB_MESSAGE));
      const id = 'wg-' + Date.now().toString(36) + '-' + (++sequence);
      return new Promise((resolve, reject) => {
        const timeout = host.setTimeout(() => {
          pending.delete(id);
          reject(new Error('Google Play has not responded yet. Refresh the shop or check pending purchases before trying again.'));
        }, settings.timeoutMs || 45000);
        pending.set(id, { resolve, reject, timeout, method, walletSequence });
        try { host.WayfarersPlayBilling.postMessage(JSON.stringify({ id, method, args: args || {} })); }
        catch (error) { host.clearTimeout(timeout); pending.delete(id); reject(new Error('Google Play could not open. Your guild has not changed.')); }
      });
    }
    function receive(event) {
      const detail = event && event.detail;
      if (!detail || typeof detail !== 'object' || typeof detail.id !== 'string') return;
      const request = pending.get(detail.id);
      if (!request && detail.id !== 'event') return;
      const data = detail.data;
      const wallet = data && data.wallet && typeof data.wallet === 'object' ? data.wallet : data;
      const hasWallet = validWallet(wallet);
      const sameWallet = wallet && (!wallet.walletId || wallet.walletId === snapshot.walletId);
      const olderRevision = hasWallet && sameWallet && Number.isSafeInteger(wallet.revision) && wallet.revision < snapshot.revision;
      const newerRevision = hasWallet && sameWallet && Number.isSafeInteger(wallet.revision) && wallet.revision > snapshot.revision;
      const olderRequest = hasWallet && request && request.method !== 'restoreWallet' && request.walletSequence < walletSequence && !newerRevision;
      if (detail.ok === true && data && !olderRevision && !olderRequest) {
        publish(data);
        if (hasWallet) walletSequence += 1;
      }
      if (!request) return;
      host.clearTimeout(request.timeout);
      pending.delete(detail.id);
      if (detail.ok === true) request.resolve(detail.data || {});
      else request.reject(new Error(typeof detail.error === 'string' ? detail.error.slice(0, 500) : 'Google Play could not complete this request. Refresh to check its status.'));
    }
    host.addEventListener('wayfarers:billing', receive);
    return {
      snapshot: catalogSnapshot,
      setCatalogContext(context) {
        const value = context || {};
        catalogContext = { owned: Array.isArray(value.owned) ? [...new Set(value.owned.filter(id => ITEMS.includes(id)))] : [], earnedBalance: Number.isFinite(value.earnedBalance) ? Math.max(0, value.earnedBalance) : 0 };
        return catalogSnapshot();
      },
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      async refresh() {
        if (!native()) return catalogSnapshot();
        publish({ message: 'Connecting to Google Play…' });
        await call('status');
        return catalogSnapshot();
      },
      async purchase(productId) {
        if (!PRODUCTS.some(item => item.productId === productId)) throw new Error('Unknown currency pack.');
        if (!snapshot.available || !snapshot.configured || !snapshot.products.some(item => item.productId === productId && item.available)) throw new Error('This pack is not available from Google Play yet.');
        const product = catalogSnapshot().products.find(item => item.productId === productId);
        if (!snapshot.hydrated || !product.budgetAvailable) throw new Error(product.blockedReason || 'Wait for your verified purchase wallet.');
        return call('purchase', { productId, earnedOwned: catalogContext.owned, earnedBalance: catalogContext.earnedBalance });
      },
      async spend(itemId) {
        if (!ITEMS.includes(itemId)) throw new Error('Unknown Starshard item.');
        return call('spend', { itemId, requestId: host.crypto && typeof host.crypto.randomUUID === 'function' ? host.crypto.randomUUID() : 'wg-' + Date.now().toString(36) + '-' + (++sequence) });
      },
      restore: () => call('restore'),
      backupWallet: () => call('backupWallet'),
      restoreWallet: () => call('restoreWallet'),
      destroy() {
        destroyed = true;
        host.removeEventListener('wayfarers:billing', receive);
        pending.forEach(request => { host.clearTimeout(request.timeout); request.reject(new Error('The guild shop was closed. Reopen it to check purchase status.')); });
        pending.clear();
        listeners.clear();
      }
    };
  }
  return { PRODUCTS, ITEMS, ITEM_COSTS, WEB_MESSAGE, getPurchaseBudget, createClient };
});
