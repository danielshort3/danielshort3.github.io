(function initProjectStarfallDataDungeons(global) {
  'use strict';

  const DUNGEONS = Object.freeze([
    Object.freeze({
      id: 'bramble_depths',
      name: 'Bramble Depths',
      summary: 'An early MVP Alpha dungeon built around control, fire payoff, and vertical add pressure.',
      mapId: 'brambleDepths',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'brambleking',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 360, currency: 180, materials: Object.freeze({ upgradeDust: 5, gelDrop: 2 }) })
    }),
    Object.freeze({
      id: 'emberjaw_lair',
      name: 'Emberjaw Lair',
      summary: 'A compact party-style dungeon that culminates in the Emberjaw Golem boss fight.',
      mapId: 'emberjawLair',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'emberjawGolem',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 420, currency: 220, materials: Object.freeze({ upgradeDust: 6, upgradeCatalyst: 1 }) })
    }),
    Object.freeze({
      id: 'gearworks_vault',
      name: 'Gearworks Vault',
      summary: 'A late prototype dungeon that checks armor break, add control, and boss uptime.',
      mapId: 'gearworksVault',
      levelRequirement: 35,
      recommendedPartySize: 4,
      bossId: 'quarryColossus',
      bossIds: Object.freeze(['clockworkTitan', 'quarryColossus']),
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 620, currency: 320, materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 3 }) })
    }),
    Object.freeze({
      id: 'rimewarden_sanctum',
      name: 'Rimewarden Sanctum',
      summary: 'A Frostfen dungeon built around slick footing, frost flyers, and Rimewarden arena control.',
      mapId: 'rimewardenSanctum',
      levelRequirement: 58,
      recommendedPartySize: 4,
      bossId: 'rimewarden',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 760, currency: 380, materials: Object.freeze({ upgradeDust: 9, upgradeCatalyst: 3, refinementCore: 1 }) })
    })
  ]);

  const api = {
    DUNGEONS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.dungeons = Object.assign({}, modules.dungeons || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
