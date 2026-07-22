(function initProjectStarfallDataCombatModifiers(global) {
  'use strict';

  const BOSS_BREAK_PROFILES = Object.freeze([
    Object.freeze({ bossId: 'brambleking', name: 'Thorn Heart', maxGauge: 140, duration: 4.2, damageTakenScale: 1.18, reward: Object.freeze({ materials: Object.freeze({ gelDrop: 1 }) }) }),
    Object.freeze({ bossId: 'clockworkTitan', name: 'Gear Exposure', maxGauge: 165, duration: 4.4, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ oreChunks: 1 }) }) }),
    Object.freeze({ bossId: 'quarryColossus', name: 'Ore Shell Crack', maxGauge: 180, duration: 4.6, damageTakenScale: 1.22, reward: Object.freeze({ materials: Object.freeze({ oreChunks: 2 }) }) }),
    Object.freeze({ bossId: 'emberjawGolem', name: 'Overheat Break', maxGauge: 155, duration: 4.8, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 1 }) }) }),
    Object.freeze({ bossId: 'rimewarden', name: 'Frostplate Break', maxGauge: 175, duration: 4.5, damageTakenScale: 1.19, reward: Object.freeze({ materials: Object.freeze({ refinementCore: 1 }) }) }),
    Object.freeze({ bossId: 'stormbreakRoc', name: 'Wing Stagger', maxGauge: 170, duration: 4.1, damageTakenScale: 1.18, reward: Object.freeze({ currency: 120 }) }),
    Object.freeze({ bossId: 'astralArchivist', name: 'Page Lockout', maxGauge: 185, duration: 4.3, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ cubeFragment: 2 }) }) }),
    Object.freeze({ bossId: 'eclipseSovereign', name: 'Totality Fracture', maxGauge: 210, duration: 4.7, damageTakenScale: 1.24, reward: Object.freeze({ materials: Object.freeze({ cubeFragment: 3 }) }) })
  ]);

  const RIFT_IMPRINT_COST = Object.freeze({ materialId: 'riftSplinter', amount: 24 });

  const SKILL_MODIFIERS = Object.freeze([
    Object.freeze({ id: 'heavy_strike_guardbreaker', skillId: 'fighter_heavy_strike', name: 'Guardbreaker', summary: 'Heavy Strike builds more break gauge on bosses.', unlockLevel: 8, breakScale: 1.35, damageScale: 1.04 }),
    Object.freeze({ id: 'ground_slam_aftershock', skillId: 'fighter_ground_slam', name: 'Aftershock', summary: 'Ground Slam gains area damage against clustered targets.', unlockLevel: 14, damageScale: 1.06, extraLines: 1 }),
    Object.freeze({ id: 'magic_bolt_arc', skillId: 'mage_magic_bolt', name: 'Arc Primer', summary: 'Magic Bolt marks targets briefly for follow-up skills.', unlockLevel: 8, markDuration: 2.5, damageScale: 1.03 }),
    Object.freeze({ id: 'chain_bolt_conductor', skillId: 'storm_mage_chain_bolt', name: 'Conductor', summary: 'Chain Bolt adds one extra line and slightly cheaper casts.', unlockLevel: 25, extraLines: 1, resourceCostScale: 0.92 }),
    Object.freeze({ id: 'fireball_ember_core', skillId: 'fire_mage_fireball', name: 'Ember Core', summary: 'Fireball burns longer and hits broken bosses harder.', unlockLevel: 25, burnDuration: 2, brokenDamageScale: 1.12 }),
    Object.freeze({ id: 'rune_mark_anchor', skillId: 'rune_mage_rune_mark', name: 'Anchor Rune', summary: 'Rune Mark lasts longer and improves link detonation.', unlockLevel: 25, markDuration: 2, runeDuration: 2 }),
    Object.freeze({ id: 'quick_shot_fletching', skillId: 'archer_quick_shot', name: 'Fine Fletching', summary: 'Quick Shot gets smoother single-target uptime.', unlockLevel: 8, cooldownScale: 0.94, damageScale: 1.02 }),
    Object.freeze({ id: 'piercing_arrow_broadhead', skillId: 'archer_piercing_arrow', name: 'Broadhead', summary: 'Piercing Arrow builds break and weak-point pressure.', unlockLevel: 14, breakScale: 1.22, weakPointDuration: 2 }),
    Object.freeze({ id: 'sniper_aimed_shot_deadeye', skillId: 'sniper_aimed_shot', name: 'Deadeye Window', summary: 'Aimed Shot deals more damage to marked or broken targets.', unlockLevel: 25, markedDamageScale: 1.12, brokenDamageScale: 1.12 }),
    Object.freeze({ id: 'trapper_snare_relay', skillId: 'trapper_snare_trap', name: 'Snare Relay', summary: 'Snare Trap arms faster and slows elites longer.', unlockLevel: 25, cooldownScale: 0.94, slowDuration: 2 }),
    Object.freeze({ id: 'guardian_bash_shatter', skillId: 'guardian_shield_bash', name: 'Shatter Bash', summary: 'Shield Bash has a stronger boss break role.', unlockLevel: 25, breakScale: 1.45 }),
    Object.freeze({ id: 'berserker_blood_cleave_crimson_edge', skillId: 'berserker_blood_cleave', name: 'Crimson Edge', summary: 'Blood Cleave favors deliberate boss pressure, with stronger hits into broken targets.', unlockLevel: 25, damageScale: 1.04, brokenDamageScale: 1.04 }),
    Object.freeze({ id: 'duelist_quick_cut_tempo_edge', skillId: 'duelist_quick_cut', name: 'Tempo Edge', summary: 'Quick Cut gains a measured blend of speed and precision damage.', unlockLevel: 25, cooldownScale: 0.98, damageScale: 1.02 }),
    Object.freeze({ id: 'beast_companion_hunt', skillId: 'beast_archer_companion_strike', name: 'Hunt Signal', summary: 'Companion Strike advances target-farm streaks faster.', unlockLevel: 25, targetFarmBonus: 0.12, damageScale: 1.04 }),
    Object.freeze({ id: 'guardian_bash_bulwark_echo', skillId: 'guardian_shield_bash', name: 'Bulwark Echo', summary: 'Trade some break specialization for steadier Shield Bash damage and cadence.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.025, cooldownScale: 0.985 }),
    Object.freeze({ id: 'berserker_blood_cleave_redline', skillId: 'berserker_blood_cleave', name: 'Redline Rhythm', summary: 'Trade broken-target payoff for faster, steadier Blood Cleave pressure.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.01, cooldownScale: 0.965 }),
    Object.freeze({ id: 'duelist_quick_cut_riposte_rhythm', skillId: 'duelist_quick_cut', name: 'Riposte Rhythm', summary: 'Trade attack cadence for slightly heavier Quick Cut openings and cheaper follow-through.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.04, resourceCostScale: 0.96 }),
    Object.freeze({ id: 'fireball_flashpoint', skillId: 'fire_mage_fireball', name: 'Flashpoint', summary: 'Trade burn and break-window setup for reliable immediate Fireball damage.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.025 }),
    Object.freeze({ id: 'rune_mark_resonant_script', skillId: 'rune_mage_rune_mark', name: 'Resonant Script', summary: 'Trade longer rune setup for a harder first Rune Mark hit at a modest mana premium.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.03, resourceCostScale: 1.06 }),
    Object.freeze({ id: 'chain_bolt_stormglass', skillId: 'storm_mage_chain_bolt', name: 'Stormglass', summary: 'Trade efficient chaining for stronger first-target Chain Bolt pressure.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.035, resourceCostScale: 1.04 }),
    Object.freeze({ id: 'sniper_aimed_shot_snap_sight', skillId: 'sniper_aimed_shot', name: 'Snap Sight', summary: 'Retain marked-target precision while trading broken-target burst for steadier Aimed Shot follow-ups.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.02, cooldownScale: 0.97, markedDamageScale: 1.12 }),
    Object.freeze({ id: 'trapper_snare_tripcoil', skillId: 'trapper_snare_trap', name: 'Tripcoil', summary: 'Trade extended control for a harder-hitting Snare Trap cadence.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.04, cooldownScale: 0.98 }),
    Object.freeze({ id: 'beast_companion_pack_tempo', skillId: 'beast_archer_companion_strike', name: 'Pack Tempo', summary: 'Trade target-farm momentum for faster Companion Strike pressure.', unlockLevel: 100, unlockSource: 'rift', unlockCost: RIFT_IMPRINT_COST, damageScale: 1.01, cooldownScale: 0.97 })
  ]);

  const GEAR_TRAITS = Object.freeze([
    Object.freeze({ id: 'bossbreaker', name: 'Bossbreaker', summary: 'Boosts boss and break-window damage.', slots: Object.freeze(['weapon', 'offhand']), rarity: 'Rare', statBonuses: Object.freeze({ armorBreak: 4, attackDamagePercent: 2 }) }),
    Object.freeze({ id: 'rift_hunter', name: 'Rift Hunter', summary: 'Improves Endless Rift score and elite rewards.', slots: Object.freeze(['weapon', 'gloves', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ power: 3, crit: 1 }) }),
    Object.freeze({ id: 'field_sweeper', name: 'Field Sweeper', summary: 'Improves mobbing and target-farm drops.', slots: Object.freeze(['weapon', 'chest', 'ring']), rarity: 'Uncommon', statBonuses: Object.freeze({ areaDamage: 3, resourceGain: 1 }) }),
    Object.freeze({ id: 'warded', name: 'Warded', summary: 'Adds survival stats for dungeon objectives.', slots: Object.freeze(['chest', 'head', 'boots']), rarity: 'Uncommon', statBonuses: Object.freeze({ hp: 35, defense: 2 }) }),
    Object.freeze({ id: 'quickstep', name: 'Quickstep', summary: 'Improves movement uptime and party spread commands.', slots: Object.freeze(['boots']), rarity: 'Uncommon', statBonuses: Object.freeze({ speed: 8, avoid: 1 }) }),
    Object.freeze({ id: 'focus_lens', name: 'Focus Lens', summary: 'Improves marks, weak points, and precision skills.', slots: Object.freeze(['offhand', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ crit: 2, critDamage: 5, weakPointDuration: 1 }) }),
    Object.freeze({ id: 'resource_loop', name: 'Resource Loop', summary: 'Raises MP and secondary-resource uptime.', slots: Object.freeze(['offhand', 'chest', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ mpMax: 28, resourceGainPercent: 3 }) }),
    Object.freeze({ id: 'party_standard', name: 'Party Standard', summary: 'Improves AI ally buffs and party objective safety.', slots: Object.freeze(['head', 'chest']), rarity: 'Epic', statBonuses: Object.freeze({ hp: 40, power: 2, defense: 2 }) })
  ]);

  const api = {
    BOSS_BREAK_PROFILES,
    RIFT_IMPRINT_COST,
    SKILL_MODIFIERS,
    GEAR_TRAITS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.combatModifiers = Object.assign({}, modules.combatModifiers || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
