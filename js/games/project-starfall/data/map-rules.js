(function initProjectStarfallDataMapRules(global) {
  'use strict';

  function createRiftMutation(definition) {
    const source = definition || {};
    return Object.freeze({
      id: source.id,
      name: source.name,
      effect: source.effect,
      danger: Object.freeze(Object.assign({
        enemyHealthScale: 1,
        enemyDamageScale: 1,
        playerResourceCostScale: 1
      }, source.danger || {})),
      upside: Object.freeze(Object.assign({
        playerDamageScale: 1,
        playerResourceCostScale: 1
      }, source.upside || {})),
      counterplay: Object.freeze(Object.assign({
        id: '',
        label: '',
        summary: '',
        dangerMitigation: 0
      }, source.counterplay || {})),
      rewardScale: Math.max(1, Number(source.rewardScale || 1))
    });
  }

  const MUTATIONS = Object.freeze([
    createRiftMutation({
      id: 'echoing',
      name: 'Echoing',
      effect: 'Rift attacks echo as heavier pressure while player offense resonates for bonus damage.',
      danger: { enemyDamageScale: 1.06 },
      upside: { playerDamageScale: 1.04 },
      counterplay: { id: 'interrupt_echo', label: 'Break the Echo', summary: 'Interrupt or stagger dangerous casts before their echo resolves.', dangerMitigation: 0.35 },
      rewardScale: 1.05
    }),
    createRiftMutation({
      id: 'splintering',
      name: 'Splintering',
      effect: 'Splintered enemies are tougher and hit harder, while player attacks gain extra force.',
      danger: { enemyHealthScale: 1.03, enemyDamageScale: 1.08 },
      upside: { playerDamageScale: 1.055 },
      counterplay: { id: 'control_spacing', label: 'Control the Spread', summary: 'Keep packs controlled and fight from clean lanes instead of stacking hazards.', dangerMitigation: 0.3 },
      rewardScale: 1.065
    }),
    createRiftMutation({
      id: 'guarded',
      name: 'Guarded',
      effect: 'Enemies gain durable Rift guards while disciplined players spend less resource.',
      danger: { enemyHealthScale: 1.12 },
      upside: { playerResourceCostScale: 0.94 },
      counterplay: { id: 'break_guard', label: 'Shatter the Guard', summary: 'Use break, armor pressure, and coordinated burst to remove durable targets quickly.', dangerMitigation: 0.4 },
      rewardScale: 1.06
    }),
    createRiftMutation({
      id: 'burning',
      name: 'Burning',
      effect: 'Burning pressure raises incoming danger while empowering committed offense.',
      danger: { enemyDamageScale: 1.11 },
      upside: { playerDamageScale: 1.065 },
      counterplay: { id: 'rotate_hazards', label: 'Rotate the Heat', summary: 'Make a decisive, damage-free combat rotation instead of extending the same trade.', dangerMitigation: 0.35 },
      rewardScale: 1.075
    }),
    createRiftMutation({
      id: 'focused',
      name: 'Focused',
      effect: 'Focused enemies sustain pressure longer while precise players gain efficient damage.',
      danger: { enemyHealthScale: 1.05, enemyDamageScale: 1.055 },
      upside: { playerDamageScale: 1.03, playerResourceCostScale: 0.96 },
      counterplay: { id: 'break_focus', label: 'Break Their Focus', summary: 'Change lanes, stagger priority targets, and deny uninterrupted pressure.', dangerMitigation: 0.3 },
      rewardScale: 1.065
    }),
    createRiftMutation({
      id: 'volatile',
      name: 'Volatile',
      effect: 'Both sides deal more damage, and player skills cost more resource during the surge.',
      danger: { enemyDamageScale: 1.16, playerResourceCostScale: 1.08 },
      upside: { playerDamageScale: 1.12 },
      counterplay: { id: 'burst_window', label: 'Choose the Burst Window', summary: 'Save resource and defensive tools for a short, decisive damage window.', dangerMitigation: 0.25 },
      rewardScale: 1.1
    })
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
