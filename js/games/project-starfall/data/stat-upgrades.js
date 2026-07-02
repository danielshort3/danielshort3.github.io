(function initProjectStarfallDataStatUpgrades(global) {
  'use strict';

  const STAT_UPGRADE_DEFINITIONS = Object.freeze([
    Object.freeze({ id: 'might', name: 'Might', summary: 'Raises direct attack power.', statBonuses: Object.freeze({ power: 1 }) }),
    Object.freeze({ id: 'vitality', name: 'Vitality', summary: 'Raises maximum HP.', statBonuses: Object.freeze({ hp: 12 }) }),
    Object.freeze({ id: 'guard', name: 'Guard', summary: 'Improves mitigation and blocking.', statBonuses: Object.freeze({ defense: 0.5, block: 0.03 }) }),
    Object.freeze({ id: 'focus', name: 'Focus', summary: 'Raises MP and class-resource gain.', statBonuses: Object.freeze({ mpMax: 8, resourceGain: 0.05 }) }),
    Object.freeze({ id: 'agility', name: 'Agility', summary: 'Improves movement and avoidance.', statBonuses: Object.freeze({ speed: 0.35, avoid: 0.05 }) }),
    Object.freeze({ id: 'precision', name: 'Precision', summary: 'Improves crit chance and crit damage.', statBonuses: Object.freeze({ crit: 0.05, critDamage: 0.12 }) })
  ]);

  const api = {
    STAT_UPGRADE_DEFINITIONS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.statUpgrades = Object.assign({}, modules.statUpgrades || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
