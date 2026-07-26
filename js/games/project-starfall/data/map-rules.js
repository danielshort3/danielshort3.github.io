(function initProjectStarfallDataMapRules(global) {
  'use strict';

  const MUTATIONS = Object.freeze([
    Object.freeze({ id: 'echoing', name: 'Echoing', effect: 'Rift packs move 8% faster and grant 12% more stability.', enemySpeedScale: 1.08, scoreScale: 1.12 }),
    Object.freeze({ id: 'splintering', name: 'Splintering', effect: 'Rift packs gain 10% HP and add 18% more to the unbanked bounty.', enemyHpScale: 1.1, rewardScale: 1.18 }),
    Object.freeze({ id: 'guarded', name: 'Guarded', effect: 'Rift packs gain 16% defense and grant 12% more stability.', enemyDefenseScale: 1.16, scoreScale: 1.12 }),
    Object.freeze({ id: 'burning', name: 'Burning', effect: 'Rift packs deal 10% more damage and add 12% more to the unbanked bounty.', enemyDamageScale: 1.1, rewardScale: 1.12 }),
    Object.freeze({ id: 'focused', name: 'Focused', effect: 'Elite pressure rises by 8% and elites grant 10% more stability.', eliteChanceBonus: 0.08, scoreScale: 1.1 }),
    Object.freeze({ id: 'volatile', name: 'Volatile', effect: 'Rift packs deal 15% more damage and grant 20% more stability.', enemyDamageScale: 1.15, scoreScale: 1.2 })
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
