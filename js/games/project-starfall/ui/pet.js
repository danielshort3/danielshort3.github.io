(function initProjectStarfallUiPet(global) {
  'use strict';

  function getPetPanelDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const settingId = getAttribute('data-starfall-pet-toggle');
    if (settingId) return { handled: true, type: 'toggleSetting', settingId };
    const thresholdKind = getAttribute('data-starfall-pet-threshold');
    if (thresholdKind) {
      return {
        handled: true,
        type: 'adjustThreshold',
        kind: thresholdKind,
        delta: Number(getAttribute('data-starfall-pet-delta') || 0)
      };
    }
    const pickerKind = getAttribute('data-starfall-pet-picker');
    if (pickerKind) return { handled: true, type: 'togglePotionPicker', kind: pickerKind };
    const pickKind = getAttribute('data-starfall-pet-pick');
    if (pickKind) return { handled: true, type: 'pickPotion', kind: pickKind, itemId: getAttribute('data-starfall-pet-item') };
    const filterId = getAttribute('data-starfall-pet-loot-filter');
    if (filterId) return { handled: true, type: 'toggleLootFilter', filterId };
    return { handled: false, type: '' };
  }

  function getPetInputDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const value = source && Object.prototype.hasOwnProperty.call(source, 'value') ? source.value : undefined;
    if (hasAttribute('data-starfall-pet-min-rarity')) {
      return { handled: true, type: 'setMinEquipmentRarity', value };
    }
    const potionKind = getAttribute('data-starfall-pet-potion');
    if (potionKind) return { handled: true, type: 'setPotion', kind: potionKind, value };
    return { handled: false, type: '' };
  }

  function getPetPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'pet-toggle') {
      return { handled: true, type: 'toggleSetting', settingId: source.settingId };
    }
    if (source.type === 'pet-threshold') {
      return { handled: true, type: 'adjustThreshold', kind: source.kind, delta: source.delta };
    }
    if (source.type === 'pet-potion') {
      return { handled: true, type: 'cyclePotion', kind: source.kind };
    }
    if (source.type === 'pet-loot-filter') {
      return { handled: true, type: 'toggleLootFilter', filterId: source.filterId };
    }
    if (source.type === 'pet-min-rarity') {
      return { handled: true, type: 'cycleMinEquipmentRarity' };
    }
    return { handled: false, type: '' };
  }

  function createPetPanelUiHelpers() {
    return Object.freeze({
      getPetPanelDomAction,
      getPetInputDomAction,
      getPetPanelRegionAction
    });
  }

  const api = {
    createPetPanelUiHelpers,
    getPetPanelDomAction,
    getPetInputDomAction,
    getPetPanelRegionAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.pet = Object.assign({}, modules.pet || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
