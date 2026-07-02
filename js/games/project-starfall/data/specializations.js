(function initProjectStarfallDataSpecializations(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const SPECIALIZATION_LEVEL = DataAssets.SPECIALIZATION_LEVEL;

  const ROSTER_TRAITS = Object.freeze([
    Object.freeze({ id: 'guardian_bulwark', name: 'Guardian Bulwark', sourceAdvancedId: 'guardian', summary: 'A roster bonus from proving the Guardian path.', statBonuses: Object.freeze({ hp: 60, defense: 4 }) }),
    Object.freeze({ id: 'berserker_fervor', name: 'Berserker Fervor', sourceAdvancedId: 'berserker', summary: 'A roster bonus from proving the Berserker path.', statBonuses: Object.freeze({ power: 3, resourceGain: 2 }) }),
    Object.freeze({ id: 'duelist_tempo', name: 'Duelist Tempo', sourceAdvancedId: 'duelist', summary: 'A roster bonus from proving the Duelist path.', statBonuses: Object.freeze({ speed: 6, crit: 2 }) }),
    Object.freeze({ id: 'fire_mage_kindling', name: 'Kindling Memory', sourceAdvancedId: 'fireMage', summary: 'A roster bonus from proving the Fire Mage path.', statBonuses: Object.freeze({ power: 2, burnDamage: 5 }) }),
    Object.freeze({ id: 'rune_mage_pattern', name: 'Pattern Memory', sourceAdvancedId: 'runeMage', summary: 'A roster bonus from proving the Rune Mage path.', statBonuses: Object.freeze({ mpMax: 35, resourceGain: 3 }) }),
    Object.freeze({ id: 'storm_mage_charge', name: 'Storm Charge', sourceAdvancedId: 'stormMage', summary: 'A roster bonus from proving the Storm Mage path.', statBonuses: Object.freeze({ mpMax: 30, areaDamage: 3 }) }),
    Object.freeze({ id: 'sniper_focus', name: 'Sniper Focus', sourceAdvancedId: 'sniper', summary: 'A roster bonus from proving the Sniper path.', statBonuses: Object.freeze({ crit: 3, range: 18 }) }),
    Object.freeze({ id: 'trapper_routes', name: 'Trapper Routes', sourceAdvancedId: 'trapper', summary: 'A roster bonus from proving the Trapper path.', statBonuses: Object.freeze({ speed: 8, trapDamage: 4 }) }),
    Object.freeze({ id: 'beast_archer_bond', name: 'Pack Bond', sourceAdvancedId: 'beastArcher', summary: 'A roster bonus from proving the Beast Archer path.', statBonuses: Object.freeze({ hp: 35, resourceGain: 3 }) }),
    Object.freeze({ id: 'dungeon_veteran', name: 'Dungeon Veteran', sourceDungeonId: 'emberjaw_lair', summary: 'Unlocked by clearing Emberjaw Lair.', statBonuses: Object.freeze({ hp: 40, power: 2 }) }),
    Object.freeze({ id: 'vaultbreaker', name: 'Vaultbreaker', sourceDungeonId: 'gearworks_vault', summary: 'Unlocked by clearing Gearworks Vault.', statBonuses: Object.freeze({ defense: 3, armorBreak: 4 }) })
  ]);

  const CLASS_TRIALS = Object.freeze([
    Object.freeze({
      id: 'guardian_trial',
      advancedId: 'guardian',
      baseClass: 'fighter',
      title: 'Guardian Trial: Hold the Line',
      summary: 'Break sturdy beasts in Thornpath to prove you can turn pressure into protection.',
      levelRequirement: 20,
      mapId: 'thornpathThicket',
      objectives: Object.freeze([
        Object.freeze({ id: 'mossbacks', type: 'defeat', enemyId: 'mossback', count: 3, label: 'Defeat 3 Mossbacks' }),
        Object.freeze({ id: 'boars', type: 'defeat', enemyId: 'bristleBoar', count: 2, label: 'Defeat 2 Bristle Boars' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'berserker_trial',
      advancedId: 'berserker',
      baseClass: 'fighter',
      title: 'Berserker Trial: Blood Rush',
      summary: 'Hunt fast Dust Imps and prove you can stay aggressive under pressure.',
      levelRequirement: 20,
      mapId: 'rustcoilRuins',
      objectives: Object.freeze([
        Object.freeze({ id: 'dust_imps', type: 'defeat', enemyId: 'dustImp', count: 5, label: 'Defeat 5 Dust Imps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 1, label: 'Defeat 1 Clockbug' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'duelist_trial',
      advancedId: 'duelist',
      baseClass: 'fighter',
      title: 'Duelist Trial: Clean Openings',
      summary: 'Challenge Bandit Ridge and prove you can create precise openings under pressure.',
      levelRequirement: 20,
      mapId: 'banditRidgeCamp',
      objectives: Object.freeze([
        Object.freeze({ id: 'cutters', type: 'defeat', enemyId: 'banditCutter', count: 3, label: 'Defeat 3 Bandit Cutters' }),
        Object.freeze({ id: 'throwers', type: 'defeat', enemyId: 'banditThrower', count: 2, label: 'Defeat 2 Bandit Throwers' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'fire_mage_trial',
      advancedId: 'fireMage',
      baseClass: 'mage',
      title: 'Fire Mage Trial: Ember Control',
      summary: 'Challenge Cinder Hollow spirits and prove you can manage explosive area pressure.',
      levelRequirement: 20,
      mapId: 'cinderHollow',
      objectives: Object.freeze([
        Object.freeze({ id: 'wisps', type: 'defeat', enemyId: 'emberWisp', count: 5, label: 'Defeat 5 Ember Wisps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 1, label: 'Defeat 1 Clockbug' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'rune_mage_trial',
      advancedId: 'runeMage',
      baseClass: 'mage',
      title: 'Rune Mage Trial: Pattern Study',
      summary: 'Study construct movement in Rustcoil and prove you can handle setup combat.',
      levelRequirement: 20,
      mapId: 'rustcoilRuins',
      objectives: Object.freeze([
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 4, label: 'Defeat 4 Clockbugs' }),
        Object.freeze({ id: 'thorn_sprouts', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'storm_mage_trial',
      advancedId: 'stormMage',
      baseClass: 'mage',
      title: 'Storm Mage Trial: Conductive Lines',
      summary: 'Fight constructs and ember spirits while learning to chain damage through crowded lanes.',
      levelRequirement: 20,
      mapId: 'cinderHollow',
      objectives: Object.freeze([
        Object.freeze({ id: 'wisps', type: 'defeat', enemyId: 'emberWisp', count: 4, label: 'Defeat 4 Ember Wisps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 2, label: 'Defeat 2 Clockbugs' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'sniper_trial',
      advancedId: 'sniper',
      baseClass: 'archer',
      title: 'Sniper Trial: Priority Targets',
      summary: 'Pick off Bandit Throwers from ridge platforms to prove precision target control.',
      levelRequirement: 20,
      mapId: 'banditRidgeCamp',
      objectives: Object.freeze([
        Object.freeze({ id: 'throwers', type: 'defeat', enemyId: 'banditThrower', count: 4, label: 'Defeat 4 Bandit Throwers' }),
        Object.freeze({ id: 'cutters', type: 'defeat', enemyId: 'banditCutter', count: 2, label: 'Defeat 2 Bandit Cutters' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'trapper_trial',
      advancedId: 'trapper',
      baseClass: 'archer',
      title: 'Trapper Trial: Route Control',
      summary: 'Stop charging beasts in Thornpath and prove you can control enemy lanes.',
      levelRequirement: 20,
      mapId: 'thornpathThicket',
      objectives: Object.freeze([
        Object.freeze({ id: 'boars', type: 'defeat', enemyId: 'bristleBoar', count: 4, label: 'Defeat 4 Bristle Boars' }),
        Object.freeze({ id: 'thorn_sprouts', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'beast_archer_trial',
      advancedId: 'beastArcher',
      baseClass: 'archer',
      title: 'Beast Archer Trial: Pack Routes',
      summary: 'Clear Bramble lanes and quarry supports while proving you can coordinate companion pressure.',
      levelRequirement: 20,
      mapId: 'orebackQuarry',
      objectives: Object.freeze([
        Object.freeze({ id: 'beetles', type: 'defeat', enemyId: 'orebackBeetle', count: 3, label: 'Defeat 3 Oreback Beetles' }),
        Object.freeze({ id: 'healers', type: 'defeat', enemyId: 'glowcapHealer', count: 2, label: 'Defeat 2 Glowcap Healers' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    })
  ]);

  const SPECIALIZATIONS = Object.freeze([
    Object.freeze({ id: 'guardian_aegis_captain', advancedId: 'guardian', name: 'Aegis Captain', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Leans into shields, boss control, and party mitigation.', statBonuses: Object.freeze({ hp: 120, defense: 8, block: 4 }) }),
    Object.freeze({ id: 'berserker_crimson_reaver', advancedId: 'berserker', name: 'Crimson Reaver', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Turns missing health and risk windows into sustained damage.', statBonuses: Object.freeze({ power: 8, resourceGain: 4 }) }),
    Object.freeze({ id: 'duelist_blade_dancer', advancedId: 'duelist', name: 'Blade Dancer', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Improves haste, precision consistency, and short repositioning windows.', statBonuses: Object.freeze({ speed: 16, crit: 5, critDamage: 10 }) }),
    Object.freeze({ id: 'fire_mage_ash_caller', advancedId: 'fireMage', name: 'Ash Caller', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Focuses Heat into larger burn fields and boss burst windows.', statBonuses: Object.freeze({ power: 7, burnDamage: 12, areaDamage: 4 }) }),
    Object.freeze({ id: 'rune_mage_seal_architect', advancedId: 'runeMage', name: 'Seal Architect', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Strengthens rune duration, MP capacity, and field control.', statBonuses: Object.freeze({ mpMax: 70, runeDuration: 10, resourceGain: 4 }) }),
    Object.freeze({ id: 'storm_mage_tempest_weaver', advancedId: 'stormMage', name: 'Tempest Weaver', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Adds reliable area damage and MP depth to lightning chains.', statBonuses: Object.freeze({ mpMax: 55, areaDamage: 8, speed: 8 }) }),
    Object.freeze({ id: 'sniper_deadeye_commander', advancedId: 'sniper', name: 'Deadeye Commander', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Rewards long-range weak point uptime with heavy precision scaling.', statBonuses: Object.freeze({ crit: 7, critDamage: 14, range: 34 }) }),
    Object.freeze({ id: 'trapper_field_engineer', advancedId: 'trapper', name: 'Field Engineer', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Improves trap setup speed, trap payoff, and controlled kill zones.', statBonuses: Object.freeze({ trapSpeed: 12, trapDamage: 10, defense: 3 }) }),
    Object.freeze({ id: 'beast_archer_pack_warden', advancedId: 'beastArcher', name: 'Pack Warden', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Builds companion sustain, Bond generation, and flexible party support.', statBonuses: Object.freeze({ hp: 85, resourceGain: 5, avoid: 3 }) })
  ]);

  const api = {
    ROSTER_TRAITS,
    CLASS_TRIALS,
    SPECIALIZATIONS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.specializations = Object.assign({}, modules.specializations || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
