(function initProjectStarfallDataMapRules(global) {
  'use strict';

  const MUTATIONS = Object.freeze([
    Object.freeze({ id: 'echoing', name: 'Echoing', effect: 'Skills have a chance to repeat at reduced power.' }),
    Object.freeze({ id: 'splintering', name: 'Splintering', effect: 'Attacks can split into smaller secondary hits.' }),
    Object.freeze({ id: 'guarded', name: 'Guarded', effect: 'Defensive skills grant minor resource.' }),
    Object.freeze({ id: 'burning', name: 'Burning', effect: 'Attacks can apply burn.' }),
    Object.freeze({ id: 'focused', name: 'Focused', effect: 'Marks, runes, or weak points last longer.' }),
    Object.freeze({ id: 'volatile', name: 'Volatile', effect: 'Higher damage but increased resource cost.' })
  ]);

  const MAP_MODIFIERS = Object.freeze([
    Object.freeze({ id: 'overgrown', name: 'Overgrown Lanes', summary: 'Dense terrain slows monsters but improves material yield.', mapTypes: Object.freeze(['field', 'dungeon']), enemySpeedScale: 0.94, lootBonus: 0.08, xpBonus: 0.03 }),
    Object.freeze({ id: 'glass_cannon', name: 'Glass Cannon Packs', summary: 'Enemies hit harder but have lower armor and grant more XP.', mapTypes: Object.freeze(['field', 'rift']), enemyDamageScale: 1.1, enemyDefenseScale: 0.88, xpBonus: 0.08 }),
    Object.freeze({ id: 'treasure_wind', name: 'Treasure Wind', summary: 'More elites and better drop rolls appear while field pressure rises.', mapTypes: Object.freeze(['field', 'rift']), eliteChanceBonus: 0.08, lootBonus: 0.12, enemyHpScale: 1.04 }),
    Object.freeze({ id: 'unstable_floor', name: 'Unstable Floor', summary: 'Dungeon enemies move faster and bosses build break gauge faster.', mapTypes: Object.freeze(['dungeon', 'rift']), enemySpeedScale: 1.08, bossBreakBonus: 0.12, currencyBonus: 0.08 }),
    Object.freeze({ id: 'thick_hide', name: 'Thick Hide', summary: 'Enemies are tougher, but break and armor tools pay out extra.', mapTypes: Object.freeze(['field', 'dungeon', 'rift']), enemyHpScale: 1.08, enemyDefenseScale: 1.08, breakRewardBonus: 0.15 }),
    Object.freeze({ id: 'lucent_cache', name: 'Lucent Cache', summary: 'Target-farm drops and Monster Guide research advance faster.', mapTypes: Object.freeze(['field', 'dungeon', 'rift']), targetFarmBonus: 0.14, researchBonus: 1 })
  ]);

  const ELITE_AFFIXES = Object.freeze([
    Object.freeze({ id: 'bulwark', name: 'Bulwark', summary: 'Higher HP and defense, vulnerable to break effects.', hpScale: 1.22, defenseScale: 1.16, breakTakenScale: 1.18 }),
    Object.freeze({ id: 'swift', name: 'Swift', summary: 'Moves and attacks faster, drops extra currency.', speedScale: 1.18, attackCooldownScale: 0.9, currencyBonus: 0.18 }),
    Object.freeze({ id: 'volatile', name: 'Volatile', summary: 'Deals more damage and grants extra XP.', damageScale: 1.14, xpBonus: 0.18 }),
    Object.freeze({ id: 'mender', name: 'Mender', summary: 'Periodically stabilizes nearby packs and drops more consumables.', hpScale: 1.12, lootBonus: 0.08 }),
    Object.freeze({ id: 'marked', name: 'Marked', summary: 'Starts with a weak point and improves target-farm progress.', weakPointDuration: 5, targetFarmBonus: 0.12 }),
    Object.freeze({ id: 'riftbound', name: 'Riftbound', summary: 'Rift elites scale harder and improve ladder score.', hpScale: 1.18, damageScale: 1.08, riftScoreBonus: 0.2 })
  ]);

  const api = {
    MUTATIONS,
    MAP_MODIFIERS,
    ELITE_AFFIXES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapRules = Object.assign({}, modules.mapRules || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
