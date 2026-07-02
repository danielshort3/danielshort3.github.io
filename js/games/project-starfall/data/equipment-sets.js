(function initProjectStarfallDataEquipmentSets(global) {
  'use strict';

  const EQUIPMENT_SETS = Object.freeze([
    Object.freeze({ id: 'thorncrown_regalia', name: 'Thorncrown Regalia', bossId: 'brambleking', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ hp: 120, defense: 10 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 10, resourceGain: 4 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ armorBreak: 8, areaDamage: 6 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 6, crit: 4 }) })
    ]) }),
    Object.freeze({ id: 'furnaceheart_arsenal', name: 'Furnaceheart Arsenal', bossId: 'emberjawGolem', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ burnDamage: 12 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 14, resourceGain: 5 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ areaDamage: 10, defense: 12 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 8, critDamage: 16 }) })
    ]) }),
    Object.freeze({ id: 'titanwork_aegis', name: 'Titanwork Aegis', bossId: 'clockworkTitan', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ defense: 20, block: 5 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ armorBreak: 12, resourceGain: 6 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ power: 16, crit: 4 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 8, hp: 250 }) })
    ]) }),
    Object.freeze({ id: 'deepcore_colossus', name: 'Deepcore Colossus', bossId: 'quarryColossus', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ defense: 28, hp: 220 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 22, armorBreak: 14 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ critDamage: 18, block: 5 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 10, areaDamage: 10 }) })
    ]) }),
    Object.freeze({ id: 'stormcaller_tempest', name: 'Stormcaller Tempest', bossId: 'stormbreakRoc', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ speed: 24, avoid: 6 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ crit: 8, range: 42 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ areaDamage: 14, resourceGain: 8 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 10, critDamage: 24 }) })
    ]) }),
    Object.freeze({ id: 'astral_index', name: 'Astral Index', bossId: 'astralArchivist', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ mpMax: 120, resourceMax: 18 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 24, resourceGain: 10 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ range: 50, areaDamage: 16 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 12, crit: 8 }) })
    ]) }),
    Object.freeze({ id: 'eclipse_paragon', name: 'Eclipse Paragon', bossId: 'eclipseSovereign', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ power: 26, defense: 24 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ crit: 10, resourceGain: 10 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ hp: 420, critDamage: 28 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 14, areaDamage: 18 }) })
    ]) })
  ]);

  const api = {
    EQUIPMENT_SETS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentSets = Object.assign({}, modules.equipmentSets || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
