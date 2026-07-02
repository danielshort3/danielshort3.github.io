(function initProjectStarfallUiCards(global) {
  'use strict';

  function getCardPanelDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const deckId = getAttribute('data-starfall-card-deck');
    if (deckId) return { handled: true, type: 'setDeck', deckId };
    const equipUid = getAttribute('data-starfall-card-equip');
    if (equipUid) return { handled: true, type: 'equip', uid: equipUid, slot: getAttribute('data-starfall-card-slot') };
    const unequipSlot = getAttribute('data-starfall-card-unequip');
    if (unequipSlot) return { handled: true, type: 'unequip', slot: unequipSlot };
    const upgradeUid = getAttribute('data-starfall-card-upgrade');
    if (upgradeUid) return { handled: true, type: 'upgrade', uid: upgradeUid };
    if (hasAttribute('data-starfall-card-upgrade-all')) return { handled: true, type: 'upgradeAll' };
    const sellUid = getAttribute('data-starfall-card-sell');
    if (sellUid) return { handled: true, type: 'sell', uid: sellUid };
    const starCardExchangeId = getAttribute('data-starfall-star-card-exchange');
    if (starCardExchangeId) return { handled: true, type: 'exchangeStarCard', materialId: starCardExchangeId };
    const lockUid = getAttribute('data-starfall-card-lock');
    if (lockUid) return { handled: true, type: 'toggleLock', uid: lockUid };
    return { handled: false, type: '' };
  }

  function getCardPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'card-deck-select') return { handled: true, type: 'setDeck', deckId: source.deckId };
    if (source.type === 'card-equip') return { handled: true, type: 'equip', uid: source.uid, slot: source.slotIndex };
    if (source.type === 'card-unequip') return { handled: true, type: 'unequip', slot: source.slotIndex };
    if (source.type === 'card-upgrade') return { handled: true, type: 'upgrade', uid: source.uid };
    if (source.type === 'card-upgrade-all') return { handled: true, type: 'upgradeAll' };
    if (source.type === 'card-item' || source.type === 'card-deck-slot') return { handled: true, type: 'selectCard', uid: source.uid };
    return { handled: false, type: '' };
  }

  function getCardSelectionDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const uid = getAttribute('data-starfall-select-card');
    if (uid) return { handled: true, type: 'selectCard', uid };
    return { handled: false, type: '' };
  }

  function createCardPanelUiHelpers() {
    return Object.freeze({
      getCardPanelDomAction,
      getCardPanelRegionAction,
      getCardSelectionDomAction
    });
  }

  const api = {
    createCardPanelUiHelpers,
    getCardPanelDomAction,
    getCardPanelRegionAction,
    getCardSelectionDomAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.cards = Object.assign({}, modules.cards || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
