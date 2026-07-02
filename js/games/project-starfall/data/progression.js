(function initProjectStarfallDataProgression(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};

  function createClassMasteryTracks(classFileIds) {
    return Object.freeze(Object.keys(classFileIds || {}).map((classId) => Object.freeze({
      classId,
      name: `${classId.replace(/([A-Z])/g, ' $1').replace(/^./, (letter) => letter.toUpperCase())} Mastery`,
      milestones: Object.freeze([
        Object.freeze({ level: 1, xp: 0, statBonuses: Object.freeze({}) }),
        Object.freeze({ level: 2, xp: 2500, statBonuses: Object.freeze({ power: 1 }) }),
        Object.freeze({ level: 3, xp: 8500, statBonuses: Object.freeze({ hp: 30, resourceGain: 1 }) }),
        Object.freeze({ level: 4, xp: 18000, statBonuses: Object.freeze({ attackDamagePercent: 1, defense: 1 }) }),
        Object.freeze({ level: 5, xp: 36000, statBonuses: Object.freeze({ power: 2, crit: 1 }) })
      ])
    })));
  }

  const DUNGEON_OBJECTIVES = Object.freeze([
    Object.freeze({ id: 'break_boss', name: 'Break the Boss', summary: 'Trigger at least one boss break window before the clear.', type: 'bossBreak', goal: 1, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 2 }) }) }),
    Object.freeze({ id: 'elite_control', name: 'Elite Control', summary: 'Defeat an elite or affixed enemy during the run.', type: 'defeatElite', goal: 1, reward: Object.freeze({ currency: 80 }) }),
    Object.freeze({ id: 'party_survival', name: 'Keep the Party Up', summary: 'Clear with no visible AI ally defeated during the run.', type: 'partySurvival', goal: 1, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 1 }) }) }),
    Object.freeze({ id: 'spatial_control', name: 'Spatial Control', summary: 'Answer dungeon boss lane, switch, vent, wall, perch, archive, or zone calls during the run.', type: 'spatialMechanic', goal: 3, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 2 }), currency: 60 }) }),
    Object.freeze({ id: 'clear_adds', name: 'Clean Sweep', summary: 'Defeat enough dungeon adds before the boss falls.', type: 'defeatDungeonEnemy', goal: 8, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 3 }) }) }),
    Object.freeze({ id: 'swift_clear', name: 'Swift Clear', summary: 'Clear the dungeon inside the bonus timer.', type: 'timedClear', goal: 420, reward: Object.freeze({ currency: 150 }) })
  ]);

  const ROSTER_SYNERGIES = Object.freeze([
    Object.freeze({ id: 'frontline_pair', name: 'Frontline Pair', summary: 'Guardian plus Berserker roster traits improve durable offense.', requiredTraitIds: Object.freeze(['guardian_bulwark', 'berserker_fervor']), statBonuses: Object.freeze({ hp: 35, power: 2 }) }),
    Object.freeze({ id: 'arcane_weather', name: 'Arcane Weather', summary: 'Fire, Rune, and Storm memories improve spell area damage.', requiredTraitIds: Object.freeze(['fire_mage_kindling', 'rune_mage_pattern', 'storm_mage_charge']), statBonuses: Object.freeze({ areaDamage: 4, mpMax: 25 }) }),
    Object.freeze({ id: 'hunter_net', name: 'Hunter Net', summary: 'Sniper, Trapper, and Beast Archer memories improve target farming.', requiredTraitIds: Object.freeze(['sniper_focus', 'trapper_routes', 'beast_archer_bond']), statBonuses: Object.freeze({ crit: 2, range: 12 }) }),
    Object.freeze({ id: 'dungeon_vanguard', name: 'Dungeon Vanguard', summary: 'Dungeon Veteran plus Vaultbreaker improves boss dungeons.', requiredTraitIds: Object.freeze(['dungeon_veteran', 'vaultbreaker']), statBonuses: Object.freeze({ defense: 2, armorBreak: 3 }) })
  ]);

  const PARTY_COMMANDS = Object.freeze([
    Object.freeze({ id: 'balanced', name: 'Balanced', summary: 'Allies split targets naturally and use normal assists.', powerScale: 0, defenseScale: 0 }),
    Object.freeze({ id: 'focus_boss', name: 'Focus Boss', summary: 'Allies prefer bosses, marks, and break windows.', preferBoss: true, powerScale: 0.04 }),
    Object.freeze({ id: 'spread_clear', name: 'Spread Clear', summary: 'Allies split across mobs more aggressively for field farming.', splitTargets: true, areaScale: 0.04 }),
    Object.freeze({ id: 'guard_player', name: 'Guard Player', summary: 'Allies favor defensive assists and party-survival objectives.', defenseScale: 0.05, shieldScale: 0.08 }),
    Object.freeze({ id: 'burst_window', name: 'Burst Window', summary: 'Allies save pressure for marked and broken targets.', preferMarked: true, powerScale: 0.06, cooldownScale: 1.08 })
  ]);

  function createProgressionData(options) {
    const settings = options || {};
    const classFileIds = settings.classFileIds || DataAssets.CLASS_FILE_IDS || {};
    return Object.freeze({
      CLASS_MASTERY_TRACKS: createClassMasteryTracks(classFileIds),
      DUNGEON_OBJECTIVES,
      ROSTER_SYNERGIES,
      PARTY_COMMANDS
    });
  }

  const defaultProgressionData = createProgressionData();
  const api = Object.assign({
    createClassMasteryTracks,
    createProgressionData
  }, defaultProgressionData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.progression = Object.assign({}, modules.progression || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
