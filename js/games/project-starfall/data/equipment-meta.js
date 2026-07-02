(function initProjectStarfallDataEquipmentMeta(global) {
  'use strict';

  const EQUIPMENT_SLOTS = Object.freeze(['weapon', 'offhand', 'head', 'chest', 'gloves', 'boots', 'ring', 'amulet']);

  const EQUIPMENT_SLOT_META = Object.freeze({
    weapon: Object.freeze({ label: 'Weapon', icon: 'WPN' }),
    offhand: Object.freeze({ label: 'Offhand', icon: 'OFF' }),
    head: Object.freeze({ label: 'Head', icon: 'HD' }),
    chest: Object.freeze({ label: 'Chest', icon: 'CH' }),
    gloves: Object.freeze({ label: 'Gloves', icon: 'GLV' }),
    boots: Object.freeze({ label: 'Boots', icon: 'BT' }),
    ring: Object.freeze({ label: 'Ring', icon: 'RG' }),
    amulet: Object.freeze({ label: 'Amulet', icon: 'AM' })
  });

  const api = {
    EQUIPMENT_SLOTS,
    EQUIPMENT_SLOT_META
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentMeta = Object.assign({}, modules.equipmentMeta || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
