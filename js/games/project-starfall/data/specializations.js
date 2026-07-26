(function initProjectStarfallDataSpecializations(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const SPECIALIZATION_LEVEL = DataAssets.SPECIALIZATION_LEVEL;
  const SPECIALIZATION_RESPEC_COST = 0;

  function freezeSpecializationSkillModifiers(value) {
    return Object.freeze(Object.entries(value || {}).reduce((result, [skillId, modifier]) => {
      result[skillId] = Object.freeze(Object.assign({}, modifier || {}));
      return result;
    }, {}));
  }

  function specialization(config) {
    const source = config || {};
    return Object.freeze(Object.assign({}, source, {
      levelRequirement: SPECIALIZATION_LEVEL,
      statBonuses: Object.freeze(Object.assign({}, source.statBonuses || {})),
      skillModifiers: freezeSpecializationSkillModifiers(source.skillModifiers)
    }));
  }

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
    specialization({
      id: 'guardian_aegis_captain',
      advancedId: 'guardian',
      name: 'Aegis Captain',
      badge: 'AEG',
      role: 'Mitigation / shield rhythm',
      summary: 'Keeps protective skills available for steady Stored Impact play.',
      mechanic: 'Impact Guard and Oath Barrier cost less and recover sooner.',
      tradeoff: 'Gives up the counter damage and boss break of Impact Marshal.',
      statBonuses: { hp: 100, defense: 6, shieldStrengthPercent: 10 },
      skillModifiers: {
        guardian_impact_guard: { cooldownScale: 0.94, resourceCostScale: 0.94 },
        guardian_oath_barrier: { cooldownScale: 0.94, resourceCostScale: 0.94 }
      }
    }),
    specialization({
      id: 'guardian_impact_marshal',
      advancedId: 'guardian',
      name: 'Impact Marshal',
      badge: 'IMP',
      role: 'Counter / boss break',
      summary: 'Turns guarded openings into harder retaliation and break pressure.',
      mechanic: 'Shield Bash, Retaliation Wave, and Guardian\'s Verdict hit harder into break windows.',
      tradeoff: 'Gives up Aegis Captain\'s health, shields, and defensive skill uptime.',
      statBonuses: { power: 8, armorBreak: 5, bossDamagePercent: 2 },
      skillModifiers: {
        guardian_shield_bash: { damageScale: 1.02, breakScale: 1.08 },
        guardian_retaliation_wave: { damageScale: 1.03, brokenDamageScale: 1.04, resourceCostScale: 1.04 },
        guardian_verdict: { damageScale: 1.04, breakScale: 1.08, resourceCostScale: 1.05 }
      }
    }),
    specialization({
      id: 'berserker_crimson_reaver',
      advancedId: 'berserker',
      name: 'Crimson Reaver',
      badge: 'REV',
      role: 'Low-health sustain / bossing',
      summary: 'Supports the Berserker\'s missing-health damage loop with more reliable recovery.',
      mechanic: 'Blood Cleave stays efficient and Crimson Recovery cycles sooner.',
      tradeoff: 'Has less pack durability and area pressure than Warhowl Ravager.',
      statBonuses: { power: 8, resourceGain: 4, hp: 50 },
      skillModifiers: {
        berserker_blood_cleave: { resourceCostScale: 0.95 },
        berserker_crimson_recovery: { cooldownScale: 0.92, resourceCostScale: 0.92 }
      }
    }),
    specialization({
      id: 'berserker_warhowl_ravager',
      advancedId: 'berserker',
      name: 'Warhowl Ravager',
      badge: 'HOW',
      role: 'Pack bruiser / safer pressure',
      summary: 'Trades maximum boss risk for sturdier, repeatable crowd fighting.',
      mechanic: 'Blood Cleave and Reckless Leap cycle faster through crowded lanes.',
      tradeoff: 'Has a lower single-target ceiling than Crimson Reaver.',
      statBonuses: { hp: 120, defense: 6, areaDamage: 4 },
      skillModifiers: {
        berserker_blood_cleave: { damageScale: 0.96, cooldownScale: 0.94, resourceCostScale: 0.93 },
        berserker_reckless_leap: { cooldownScale: 0.9, resourceCostScale: 0.92 }
      }
    }),
    specialization({
      id: 'duelist_blade_dancer',
      advancedId: 'duelist',
      name: 'Blade Dancer',
      badge: 'BLD',
      role: 'Mobility / Tempo consistency',
      summary: 'Keeps quick cuts and repositioning fluid across changing targets.',
      mechanic: 'Quick Cut costs less while Flash Step recovers sooner.',
      tradeoff: 'Gives up Riposte Ace\'s marked-target damage and boss break.',
      statBonuses: { speed: 16, crit: 5, critDamage: 10 },
      skillModifiers: {
        duelist_quick_cut: { resourceCostScale: 0.95 },
        duelist_flash_step: { cooldownScale: 0.88, resourceCostScale: 0.92 }
      }
    }),
    specialization({
      id: 'duelist_riposte_ace',
      advancedId: 'duelist',
      name: 'Riposte Ace',
      badge: 'RIP',
      role: 'Focused dueling / boss control',
      summary: 'Commits to one opening and converts setup into harder precision cuts.',
      mechanic: 'Quick Cut gains marked-target damage and stronger break pressure.',
      tradeoff: 'Gives up Blade Dancer\'s speed, crit scaling, and mobility uptime.',
      statBonuses: { power: 5, armorBreak: 5, crit: 4 },
      skillModifiers: {
        duelist_quick_cut: { damageScale: 1.02, markedDamageScale: 1.04, breakScale: 1.08, resourceCostScale: 1.04 },
        duelist_flash_step: { cooldownScale: 1.04 }
      }
    }),
    specialization({
      id: 'fire_mage_ash_caller',
      advancedId: 'fireMage',
      name: 'Ash Caller',
      badge: 'ASH',
      role: 'Burn spread / mobbing',
      summary: 'Favors wide burn routes and repeat Wildfire casts through dense packs.',
      mechanic: 'Fireball costs less while Wildfire hits harder and recovers sooner.',
      tradeoff: 'Gives up Cinder Savant\'s single-target Heat cashout.',
      statBonuses: { burnDamage: 10, areaDamage: 5, resourceGain: 2 },
      skillModifiers: {
        fire_mage_fireball: { damageScale: 1.02, resourceCostScale: 0.96 },
        fire_mage_wildfire: { damageScale: 1.03, cooldownScale: 0.97, resourceCostScale: 0.95 }
      }
    }),
    specialization({
      id: 'fire_mage_cinder_savant',
      advancedId: 'fireMage',
      name: 'Cinder Savant',
      badge: 'CIN',
      role: 'Heat cashout / bossing',
      summary: 'Builds toward a deliberate marked-target Inferno Burst.',
      mechanic: 'Burning Mark and Inferno Burst gain stronger marked-target payoff.',
      tradeoff: 'Gives up Ash Caller\'s area damage, resource gain, and Wildfire cadence.',
      statBonuses: { power: 7, burnDamage: 6, bossDamagePercent: 3 },
      skillModifiers: {
        fire_mage_burning_mark: { damageScale: 1.03, markedDamageScale: 1.03 },
        fire_mage_inferno_burst: { damageScale: 1.04, markedDamageScale: 1.06, resourceCostScale: 1.05 }
      }
    }),
    specialization({
      id: 'rune_mage_seal_architect',
      advancedId: 'runeMage',
      name: 'Seal Architect',
      badge: 'SEA',
      role: 'Field setup / control',
      summary: 'Keeps wide rune fields and control tools active with a deeper MP plan.',
      mechanic: 'Ground Glyph and Mana Seal cost less, recover sooner, and hold enemies longer.',
      tradeoff: 'Gives up Runebreaker\'s detonation damage and boss break.',
      statBonuses: { mpMax: 50, runeDuration: 8, resourceGain: 4 },
      skillModifiers: {
        rune_mage_ground_glyph: { cooldownScale: 0.94, resourceCostScale: 0.92, slowDuration: 6 },
        rune_mage_mana_seal: { cooldownScale: 0.94, resourceCostScale: 0.92, slowDuration: 6 }
      }
    }),
    specialization({
      id: 'rune_mage_runebreaker',
      advancedId: 'runeMage',
      name: 'Runebreaker',
      badge: 'BRK',
      role: 'Detonation / boss break',
      summary: 'Trades field uptime for sharper rune detonations against prepared targets.',
      mechanic: 'Arcane Link and Rune Detonation gain marked-target damage and break pressure.',
      tradeoff: 'Gives up Seal Architect\'s MP, rune duration, and field-control cadence.',
      statBonuses: { power: 6, armorBreak: 6, bossDamagePercent: 3 },
      skillModifiers: {
        rune_mage_arcane_link: { damageScale: 1.02, markedDamageScale: 1.03 },
        rune_mage_rune_detonation: { damageScale: 1.04, markedDamageScale: 1.06, breakScale: 1.08, resourceCostScale: 1.05 }
      }
    }),
    specialization({
      id: 'storm_mage_tempest_weaver',
      advancedId: 'stormMage',
      name: 'Tempest Weaver',
      badge: 'TMP',
      role: 'Chain sustain / dense packs',
      summary: 'Keeps lightning moving through crowded lanes with a forgiving MP rhythm.',
      mechanic: 'Chain Bolt costs less and recovers faster during repeated pack clears.',
      tradeoff: 'Gives up Stormlance\'s marked-target and boss damage.',
      statBonuses: { mpMax: 45, resourceCostReductionPercent: 4, speed: 10 },
      skillModifiers: {
        storm_mage_chain_bolt: { damageScale: 0.98, cooldownScale: 0.98, resourceCostScale: 0.9 }
      }
    }),
    specialization({
      id: 'storm_mage_stormlance',
      advancedId: 'stormMage',
      name: 'Stormlance',
      badge: 'LNC',
      role: 'Single-target lightning / mobility',
      summary: 'Uses marks and fast repositioning to focus lightning onto priority targets.',
      mechanic: 'Chain Bolt hits marked targets harder and Static Shift recovers sooner.',
      tradeoff: 'Gives up Tempest Weaver\'s MP depth and chain-casting efficiency.',
      statBonuses: { power: 6, bossDamagePercent: 4, crit: 3, range: 50 },
      skillModifiers: {
        storm_mage_chain_bolt: { damageScale: 0.97, markedDamageScale: 1.06, resourceCostScale: 1.05 },
        storm_mage_static_shift: { cooldownScale: 0.88, resourceCostScale: 0.94 }
      }
    }),
    specialization({
      id: 'sniper_deadeye_commander',
      advancedId: 'sniper',
      name: 'Deadeye Commander',
      badge: 'DED',
      role: 'Weak-point cashout / bossing',
      summary: 'Rewards patient mark setup with heavier precision shots.',
      mechanic: 'Aimed Shot, Execution Shot, and One Perfect Shot gain marked-target payoff.',
      tradeoff: 'Gives up Ridge Runner\'s mobility, elite damage, and lane control.',
      statBonuses: { crit: 5, critDamage: 10, bossDamagePercent: 4, range: 60 },
      skillModifiers: {
        sniper_aimed_shot: { markedDamageScale: 1.02, resourceCostScale: 1.03 },
        sniper_execution_shot: { markedDamageScale: 1.03, resourceCostScale: 1.04 },
        sniper_one_perfect_shot: { markedDamageScale: 1.03, resourceCostScale: 1.04 }
      }
    }),
    specialization({
      id: 'sniper_ridge_runner',
      advancedId: 'sniper',
      name: 'Ridge Runner',
      badge: 'RDG',
      role: 'Mobile elite hunter / lanes',
      summary: 'Moves between firing lanes quickly and keeps armor-piercing shots available.',
      mechanic: 'Combat Roll and Pierce Armor recover sooner, with stronger lane damage.',
      tradeoff: 'Gives up Deadeye Commander\'s boss and weak-point ceiling.',
      statBonuses: { speed: 12, range: 100, eliteDamagePercent: 5, avoid: 2, mobilityCooldownPercent: 5 },
      skillModifiers: {
        sniper_combat_roll: { cooldownScale: 0.86, resourceCostScale: 0.92 },
        sniper_pierce_armor: { damageScale: 1.03, cooldownScale: 0.96, resourceCostScale: 0.96 }
      }
    }),
    specialization({
      id: 'trapper_field_engineer',
      advancedId: 'trapper',
      name: 'Field Engineer',
      badge: 'FLD',
      role: 'Persistent trap network / control',
      summary: 'Maintains a reliable web of quick, efficient traps through long fights.',
      mechanic: 'Snare Trap, Spike Trap, and Tripwire cost less; Snare holds foes longer.',
      tradeoff: 'Gives up Quarry Saboteur\'s active detonation and boss break.',
      statBonuses: { trapDamage: 8, defense: 8, cooldownRecoveryPercent: 3 },
      skillModifiers: {
        trapper_snare_trap: { resourceCostScale: 0.94, slowDuration: 6 },
        trapper_spike_trap: { resourceCostScale: 0.94 },
        trapper_tripwire: { resourceCostScale: 0.94 }
      }
    }),
    specialization({
      id: 'trapper_quarry_saboteur',
      advancedId: 'trapper',
      name: 'Quarry Saboteur',
      badge: 'SAB',
      role: 'Active demolition / boss break',
      summary: 'Commits trap resources to deliberate detonation bursts.',
      mechanic: 'Detonate and Kill Zone hit harder and add break pressure.',
      tradeoff: 'Gives up Field Engineer\'s defense and sustained trap cadence.',
      statBonuses: { power: 5, armorBreak: 6, bossDamagePercent: 3, resourceGain: 1 },
      skillModifiers: {
        trapper_detonate: { damageScale: 1.06, breakScale: 1.08, resourceCostScale: 1.06 },
        trapper_kill_zone: { damageScale: 1.04, breakScale: 1.05, cooldownScale: 1.04, resourceCostScale: 1.06 }
      }
    }),
    specialization({
      id: 'beast_archer_pack_warden',
      advancedId: 'beastArcher',
      name: 'Pack Warden',
      badge: 'PAC',
      role: 'Sustain / mark continuity',
      summary: 'Keeps companion pressure and repositioning comfortable over long sessions.',
      mechanic: 'Companion Strike costs less while Pounce Roll costs less and recovers sooner.',
      tradeoff: 'Gives up Alpha Hunter\'s focused-target damage and armor break.',
      statBonuses: { hp: 100, defense: 4, buffEffectPercent: 2, mpRecoveryPercent: 20 },
      skillModifiers: {
        beast_archer_companion_strike: { resourceCostScale: 0.92 },
        beast_archer_pounce_roll: { cooldownScale: 0.9, resourceCostScale: 0.92 }
      }
    }),
    specialization({
      id: 'beast_archer_alpha_hunter',
      advancedId: 'beastArcher',
      name: 'Alpha Hunter',
      badge: 'ALP',
      role: 'Focused-target offense / hunt',
      summary: 'Commits the pack to one marked target for a sharper damage route.',
      mechanic: 'Companion Strike gains strong marked-target damage at a higher resource cost.',
      tradeoff: 'Gives up Pack Warden\'s health, defense, and casting comfort.',
      statBonuses: { power: 6, crit: 4, armorBreak: 4, range: 20 },
      skillModifiers: {
        beast_archer_companion_strike: { damageScale: 0.98, markedDamageScale: 1.08, resourceCostScale: 1.05 },
        beast_archer_pounce_roll: { cooldownScale: 1.04 }
      }
    })
  ]);

  const api = {
    ROSTER_TRAITS,
    CLASS_TRIALS,
    SPECIALIZATIONS,
    SPECIALIZATION_RESPEC_COST
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.specializations = Object.assign({}, modules.specializations || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
