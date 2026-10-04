(function (root, factory) {
  'use strict';
  const api = factory(typeof module === 'object' && module.exports ? require('./numbers.js') : root.WayfarersNumbers);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersProgressionPurchases = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N) {
  'use strict';
  const COUNTS = [1, 5, 10, 25, 100];
  const TYPES = { buy: ['upgrades', 'UPGRADES'], 'refit-upgrade': ['refitUpgrades', 'REFIT_UPGRADES'], 'legacy-upgrade': ['legacy', 'LEGACY_UPGRADES'] };
  function quote(state, action, count, dependencies) {
    const result = { valid: false, affordable: false, reason: '', costs: {}, count, rank: 0, rankAfter: 0 };
    const info = action && TYPES[action.type];
    if (![3,4].includes(state.expedition?.version) || !info) { result.reason = 'This purchase is not repeatable.'; return result; }
    const [map, catalog] = info;
    const definition = dependencies.Content[catalog].find(item => item.id === action.id);
    if (!definition || !dependencies.isOpen(state, action)) { result.reason = 'Unlock this improvement first.'; return result; }
    result.rank = state[map][action.id];
    result.rankAfter = result.rank + (Number.isSafeInteger(count) ? count : 0);
    if (!COUNTS.includes(count) || !dependencies.batchModes(state).some(mode => mode.count === count && mode.unlocked)) {
      result.reason = 'Earn this exact purchase quantity first.'; return result;
    }
    if (result.rankAfter > 1000000) { result.reason = 'This batch exceeds the upgrade limit. Choose a smaller quantity.'; return result; }
    const preview = { ...state, [map]: { ...state[map] } };
    for (let index = 0; index < count; index += 1) {
      preview[map][action.id] = result.rank + index;
      const cost = action.type === 'buy' ? dependencies.upgradeCost(preview, definition)
        : { [action.type === 'refit-upgrade' ? 'notes' : 'crests']: N.mul(definition.base, N.pow(definition.scale, preview[map][action.id])) };
      for (const [id, amount] of Object.entries(cost)) result.costs[id] = N.add(result.costs[id] || 0, amount);
    }
    result.valid = true;
    result.affordable = Object.entries(result.costs).every(([id, amount]) => N.cmp(state.resources[id], amount) >= 0);
    if (!result.affordable) result.reason = 'Save for the entire ×' + count + ' batch.';
    result.token = [state.run.id, state.expedition.revision, action.type, action.id, result.rank, count, JSON.stringify(result.costs)].join(':');
    return result;
  }
  function commit(state, action, dependencies) {
    const count = action.count === undefined ? 1 : action.count;
    const offer = quote(state, action, count, dependencies);
    if (!offer.valid || !offer.affordable) return { ok: false, message: offer.reason };
    if (action.quote !== undefined && action.quote !== offer.token) return { ok: false, message: 'This quote changed. Review its refreshed cost and effect.' };
    const [map, catalog] = TYPES[action.type];
    const definition = dependencies.Content[catalog].find(item => item.id === action.id);
    // No intermediate advancement or callback: the whole batch is one transaction.
    for (const [id, amount] of Object.entries(offer.costs)) state.resources[id] = N.sub(state.resources[id], amount);
    state[map][action.id] = offer.rankAfter;
    state.expedition.revision += 1;
    return { ok: true, quantity: count, equipmentRanks: action.type === 'buy' && definition.equipment ? count : 0,
      message: definition.name + ' +' + count + ' · rank ' + offer.rankAfter };
  }
  return { quote, commit };
});
