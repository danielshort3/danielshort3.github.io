(function initProjectStarfallDataUpgrades(global) {
  'use strict';

  const UPGRADE_OUTCOMES = Object.freeze([
    { id: 'success', label: 'Success', weightByRange: [90, 64, 39, 22], effect: 'Increase upgrade level by +1.' },
    { id: 'fail', label: 'Failure', weightByRange: [10, 36, 53, 62], effect: 'Decrease upgrade level by -1. Upgrade Dust is consumed.' },
    { id: 'destroy', label: 'Destroy', weightByRange: [0, 0, 8, 16], effect: 'Destroy the item. Upgrade Dust is consumed.' }
  ]);

  const UPGRADE_DUST_COST_BY_RANGE = Object.freeze([1, 3, 6, 10]);

  const UPGRADE_AIDES = Object.freeze([
    Object.freeze({ id: 'upgradeCatalyst', materialId: 'upgradeCatalyst', type: 'chance', name: 'Upgrade Catalyst', icon: 'UC', rarity: 'Rare', successBonus: 10, summary: 'Adds +10% success chance to the next upgrade attempt.' }),
    Object.freeze({ id: 'wardingScroll', materialId: 'wardingScroll', type: 'protection', name: 'Warding Scroll', icon: 'WS', rarity: 'Rare', protectsDestroy: true, summary: 'Turns a destroy result into a failure so the item survives.' }),
    Object.freeze({ id: 'refinementCore', materialId: 'refinementCore', type: 'enhancement', name: 'Refinement Core', icon: 'RC', rarity: 'Epic', successUpgradeBonus: 1, summary: 'Successful upgrades gain +2 instead of +1.' })
  ]);

  const POTENTIAL_TIERS = Object.freeze([
    Object.freeze({ id: 'rare', name: 'Rare Attunement', nextTier: 'epic', tierUpChance: 0.12 }),
    Object.freeze({ id: 'epic', name: 'Epic Attunement', nextTier: 'relic', tierUpChance: 0.04 }),
    Object.freeze({ id: 'relic', name: 'Relic Attunement', nextTier: 'mythic', tierUpChance: 0.015 }),
    Object.freeze({ id: 'mythic', name: 'Mythic Attunement', nextTier: 'ascendant', tierUpChance: 0.005 }),
    Object.freeze({ id: 'ascendant', name: 'Ascendant Attunement', nextTier: 'celestial', tierUpChance: 0.0015 }),
    Object.freeze({ id: 'celestial', name: 'Celestial Attunement', nextTier: '', tierUpChance: 0 })
  ]);

  const POTENTIAL_LINE_POOLS = Object.freeze([
    Object.freeze({ stat: 'powerPercent', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [1, 3], epic: [4, 6], relic: [7, 11], mythic: [12, 15], ascendant: [16, 20], celestial: [21, 28] }) }),
    Object.freeze({ stat: 'attackDamagePercent', slots: Object.freeze(['weapon', 'gloves']), values: Object.freeze({ rare: [1, 3], epic: [3, 5], relic: [6, 10], mythic: [11, 14], ascendant: [15, 19], celestial: [20, 26] }) }),
    Object.freeze({ stat: 'bossDamagePercent', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'eliteDamagePercent', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'areaDamage', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'burnDamage', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'armorBreak', slots: Object.freeze(['weapon', 'gloves']), values: Object.freeze({ rare: [2, 4], epic: [5, 8], relic: [9, 14], mythic: [15, 20], ascendant: [21, 27], celestial: [28, 36] }) }),
    Object.freeze({ stat: 'maxMpPercent', slots: Object.freeze(['offhand', 'head', 'amulet']), values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 13], mythic: [14, 17], ascendant: [18, 23], celestial: [24, 32] }) }),
    Object.freeze({ stat: 'mpRecoveryPercent', slots: Object.freeze(['offhand', 'ring', 'amulet']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'resourceGainPercent', slots: Object.freeze(['offhand', 'ring']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'resourceMax', slots: Object.freeze(['offhand', 'ring']), values: Object.freeze({ rare: [2, 4], epic: [5, 8], relic: [9, 14], mythic: [15, 20], ascendant: [21, 27], celestial: [28, 38] }) }),
    Object.freeze({ stat: 'resourceCostReductionPercent', slots: Object.freeze(['offhand', 'amulet']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'shieldStrengthPercent', slots: Object.freeze(['offhand', 'chest']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'runeDuration', slots: Object.freeze(['offhand', 'amulet']), values: Object.freeze({ rare: [1, 1], epic: [1, 2], relic: [2, 3], mythic: [4, 5], ascendant: [6, 7], celestial: [8, 9] }) }),
    Object.freeze({ stat: 'block', slots: Object.freeze(['offhand', 'chest']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'crit', slots: Object.freeze(['head']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'weakPointDuration', slots: Object.freeze(['head', 'amulet']), values: Object.freeze({ rare: [1, 1], epic: [1, 2], relic: [2, 3], mythic: [4, 5], ascendant: [6, 7], celestial: [8, 9] }) }),
    Object.freeze({ stat: 'markDuration', slots: Object.freeze(['head']), values: Object.freeze({ rare: [1, 1], epic: [1, 2], relic: [2, 3], mythic: [4, 5], ascendant: [6, 7], celestial: [8, 9] }) }),
    Object.freeze({ stat: 'cooldownRecoveryPercent', slots: Object.freeze(['head', 'amulet']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'buffDurationPercent', slots: Object.freeze(['head', 'amulet']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'maxHpPercent', slots: Object.freeze(['chest']), values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 13], mythic: [14, 17], ascendant: [18, 23], celestial: [24, 32] }) }),
    Object.freeze({ stat: 'defensePercent', slots: Object.freeze(['chest']), values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 13], mythic: [14, 17], ascendant: [18, 23], celestial: [24, 32] }) }),
    Object.freeze({ stat: 'hpRecoveryPercent', slots: Object.freeze(['chest', 'boots', 'ring']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'damageReductionPercent', slots: Object.freeze(['chest']), values: Object.freeze({ rare: [1, 1], epic: [2, 3], relic: [4, 6], mythic: [7, 9], ascendant: [10, 12], celestial: [13, 16] }) }),
    Object.freeze({ stat: 'potionEffectPercent', slots: Object.freeze(['chest']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'critDamage', slots: Object.freeze(['gloves']), values: Object.freeze({ rare: [4, 7], epic: [8, 13], relic: [14, 23], mythic: [24, 31], ascendant: [32, 40], celestial: [42, 54] }) }),
    Object.freeze({ stat: 'damageFloor', slots: Object.freeze(['gloves', 'ring']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'trapDamage', slots: Object.freeze(['gloves']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'trapSpeed', slots: Object.freeze(['gloves', 'boots']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'executeDamagePercent', slots: Object.freeze(['gloves']), values: Object.freeze({ rare: [2, 4], epic: [5, 9], relic: [10, 16], mythic: [18, 23], ascendant: [25, 31], celestial: [33, 42] }) }),
    Object.freeze({ stat: 'speed', slots: Object.freeze(['boots']), values: Object.freeze({ rare: [4, 8], epic: [9, 15], relic: [16, 25], mythic: [28, 38], ascendant: [40, 52], celestial: [54, 72] }) }),
    Object.freeze({ stat: 'avoid', slots: Object.freeze(['boots', 'ring']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 17], celestial: [18, 24] }) }),
    Object.freeze({ stat: 'skillEffectPercent', slots: Object.freeze(['boots']), values: Object.freeze({ rare: [1, 3], epic: [4, 6], relic: [7, 11], mythic: [12, 15], ascendant: [16, 20], celestial: [21, 28] }) }),
    Object.freeze({ stat: 'mobilityCooldownPercent', slots: Object.freeze(['boots']), values: Object.freeze({ rare: [1, 3], epic: [4, 6], relic: [7, 11], mythic: [12, 15], ascendant: [16, 20], celestial: [21, 28] }) }),
    Object.freeze({ stat: 'mobilityWindowPercent', slots: Object.freeze(['boots']), values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [11, 17], mythic: [18, 24], ascendant: [25, 32], celestial: [34, 45] }) }),
    Object.freeze({ stat: 'hpOnHit', slots: Object.freeze(['ring']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 16], celestial: [17, 22] }) }),
    Object.freeze({ stat: 'mpOnHit', slots: Object.freeze(['ring']), values: Object.freeze({ rare: [1, 2], epic: [3, 5], relic: [6, 8], mythic: [9, 12], ascendant: [13, 16], celestial: [17, 22] }) }),
    Object.freeze({ stat: 'buffEffectPercent', slots: Object.freeze(['amulet']), values: Object.freeze({ rare: [1, 3], epic: [4, 6], relic: [7, 11], mythic: [12, 15], ascendant: [16, 20], celestial: [21, 28] }) })
  ]);

  const api = {
    UPGRADE_OUTCOMES,
    UPGRADE_DUST_COST_BY_RANGE,
    UPGRADE_AIDES,
    POTENTIAL_TIERS,
    POTENTIAL_LINE_POOLS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.upgrades = Object.assign({}, modules.upgrades || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
