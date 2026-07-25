'use strict';

const BASE_CLASS_IDS = Object.freeze(['fighter', 'mage', 'archer']);
const ADVANCED_CLASS_IDS = Object.freeze([
  'guardian',
  'berserker',
  'duelist',
  'fireMage',
  'runeMage',
  'stormMage',
  'sniper',
  'trapper',
  'beastArcher'
]);
const ALL_CLASS_IDS = Object.freeze(BASE_CLASS_IDS.concat(ADVANCED_CLASS_IDS));

const CLASS_ROTATIONS = Object.freeze({
  fighter: Object.freeze(['fighter_power_break', 'fighter_momentum_burst', 'fighter_ground_slam', 'fighter_heavy_strike']),
  mage: Object.freeze(['mage_spell_mark', 'mage_energy_release', 'mage_arcane_burst', 'mage_magic_bolt']),
  archer: Object.freeze(['archer_eagle_stance', 'archer_marked_shot', 'archer_focused_volley', 'archer_piercing_arrow', 'archer_quick_shot']),
  guardian: Object.freeze(['guardian_shield_wall', 'guardian_impact_guard', 'guardian_verdict', 'guardian_retaliation_wave', 'guardian_shield_bash']),
  berserker: Object.freeze(['berserker_war_cry', 'berserker_rage_surge', 'berserker_last_stand', 'berserker_crimson_recovery', 'berserker_blood_cleave']),
  duelist: Object.freeze(['duelist_rallying_flourish', 'duelist_quick_cut']),
  fireMage: Object.freeze(['fire_mage_ignition_aura', 'fire_mage_burning_mark', 'fire_mage_inferno_burst', 'fire_mage_wildfire', 'fire_mage_heat_vent', 'fire_mage_fireball']),
  runeMage: Object.freeze(['rune_mage_rune_circle', 'rune_mage_arcane_link', 'rune_mage_rune_detonation', 'rune_mage_grand_inscription', 'rune_mage_ground_glyph', 'rune_mage_rune_mark']),
  stormMage: Object.freeze(['storm_mage_stormfront', 'storm_mage_chain_bolt']),
  sniper: Object.freeze(['sniper_eagle_eye', 'sniper_weak_point_mark', 'sniper_one_perfect_shot', 'sniper_execution_shot', 'sniper_pierce_armor', 'sniper_aimed_shot']),
  trapper: Object.freeze(['trapper_tactical_field', 'trapper_kill_zone', 'trapper_detonate', 'trapper_tripwire', 'trapper_spike_trap', 'trapper_lure_shot', 'trapper_snare_trap']),
  beastArcher: Object.freeze(['beast_archer_pack_call', 'beast_archer_companion_strike'])
});

const ADVANCED_BASE_FILLERS = Object.freeze({
  guardian: Object.freeze(['fighter_power_break', 'fighter_ground_slam']),
  berserker: Object.freeze(['fighter_power_break', 'fighter_momentum_burst']),
  duelist: Object.freeze(['fighter_power_break', 'fighter_momentum_burst', 'fighter_ground_slam']),
  fireMage: Object.freeze(['mage_spell_mark', 'mage_energy_release']),
  runeMage: Object.freeze(['mage_spell_mark', 'mage_energy_release']),
  stormMage: Object.freeze(['mage_spell_mark', 'mage_energy_release', 'mage_arcane_burst']),
  sniper: Object.freeze(['archer_marked_shot', 'archer_focused_volley']),
  trapper: Object.freeze(['archer_marked_shot', 'archer_piercing_arrow']),
  beastArcher: Object.freeze(['archer_marked_shot', 'archer_focused_volley', 'archer_piercing_arrow'])
});

const BALANCE_SCENARIOS = Object.freeze([
  Object.freeze({
    id: 'singleBoss',
    label: 'Single Boss',
    duration: 45,
    enemies: Object.freeze([
      Object.freeze({ id: 'brambleking', x: 118, yOffset: 0, hp: 240000, boss: true })
    ])
  }),
  Object.freeze({
    id: 'clusteredPack',
    label: 'Clustered Pack',
    duration: 18,
    enemies: Object.freeze(Array.from({ length: 6 }, (_, index) =>
      Object.freeze({ id: 'slimelet', x: 96 + index * 22, yOffset: 0, hp: 52000 })))
  }),
  Object.freeze({
    id: 'spreadPack',
    label: 'Spread Pack',
    duration: 24,
    enemies: Object.freeze(Array.from({ length: 6 }, (_, index) =>
      Object.freeze({ id: index % 2 ? 'dustImp' : 'vineSnapper', x: 92 + index * 94, yOffset: 0, hp: 56000 })))
  }),
  Object.freeze({
    id: 'armoredTarget',
    label: 'Armored Target',
    duration: 34,
    enemies: Object.freeze([
      Object.freeze({ id: 'clockbug', x: 124, yOffset: 0, hp: 175000, defenseScale: 1.55 })
    ])
  }),
  Object.freeze({
    id: 'flyingPack',
    label: 'Flying Pack',
    duration: 24,
    enemies: Object.freeze(Array.from({ length: 4 }, (_, index) =>
      Object.freeze({ id: 'emberWisp', x: 116 + index * 70, yOffset: -78 - index % 2 * 18, hp: 52000, flying: true })))
  }),
  Object.freeze({
    id: 'sustainBoss',
    label: 'Sustain Boss',
    duration: 45,
    incomingDps: 12,
    enemies: Object.freeze([
      Object.freeze({ id: 'emberjawGolem', x: 118, yOffset: 0, hp: 245000, boss: true })
    ])
  })
]);

const SCENARIO_BALANCE_PROFILES = Object.freeze({
  singleBoss: Object.freeze({ boss: 1, targets: 1, clustered: 0, spread: 0, armored: 0, flying: 0, sustain: 0 }),
  clusteredPack: Object.freeze({ boss: 0, targets: 6, clustered: 1, spread: 0, armored: 0, flying: 0, sustain: 0 }),
  spreadPack: Object.freeze({ boss: 0, targets: 6, clustered: 0.25, spread: 1, armored: 0, flying: 0, sustain: 0 }),
  armoredTarget: Object.freeze({ boss: 0.35, targets: 1, clustered: 0, spread: 0, armored: 1, flying: 0, sustain: 0 }),
  flyingPack: Object.freeze({ boss: 0, targets: 4, clustered: 0.45, spread: 0.65, armored: 0, flying: 1, sustain: 0 }),
  sustainBoss: Object.freeze({ boss: 1, targets: 1, clustered: 0, spread: 0, armored: 0.4, flying: 0, sustain: 1 })
});

const FIELD_PROGRESSION_BRACKETS = Object.freeze([
  Object.freeze({ id: 'early', label: 'Early', minLevel: 1, maxLevel: 8, targetMinutes: Object.freeze([3, 8]) }),
  Object.freeze({ id: 'mid', label: 'Midgame', minLevel: 9, maxLevel: 40, targetMinutes: Object.freeze([10, 25]) }),
  Object.freeze({ id: 'late', label: 'Late', minLevel: 41, maxLevel: 100, targetMinutes: Object.freeze([25, 60]) }),
  Object.freeze({ id: 'endgame', label: 'Endgame', minLevel: 100, maxLevel: Infinity, targetMinutes: Object.freeze([60, 240]) })
]);

const BOSS_MECHANIC_CATEGORIES = Object.freeze([
  Object.freeze({
    id: 'positioning',
    label: 'Positioning',
    keywords: Object.freeze(['lane', 'lanes', 'safe', 'spread', 'stack', 'split', 'rotate', 'gap', 'platform', 'tier', 'circle', 'sigil', 'around'])
  }),
  Object.freeze({
    id: 'dodgeMovement',
    label: 'Dodge And Movement',
    keywords: Object.freeze(['dodge', 'shadow', 'charge', 'divebomb', 'cross', 'move', 'jump', 'sweep', 'shockwave', 'blast', 'flare', 'rockfall', 'whiteout'])
  }),
  Object.freeze({
    id: 'burstWindow',
    label: 'Burst Window',
    keywords: Object.freeze(['burst', 'window', 'expose', 'exposed', 'weakens', 'overheat', 'plateexpose', 'crownexpose', 'corepulse'])
  }),
  Object.freeze({
    id: 'addControl',
    label: 'Add Control',
    keywords: Object.freeze(['add', 'adds', 'spawn', 'spawns', 'minion', 'minions', 'wave', 'pods', 'sprouts', 'duelists'])
  }),
  Object.freeze({
    id: 'utilityControl',
    label: 'Utility And Control',
    keywords: Object.freeze(['break', 'crack', 'armor', 'interrupt', 'control', 'seal', 'cage', 'anchor', 'wall', 'rod', 'mark', 'stagger'])
  }),
  Object.freeze({
    id: 'resourceAttrition',
    label: 'Resource Attrition',
    keywords: Object.freeze(['punish', 'pressure', 'overlap', 'accelerate', 'repeated', 'lava', 'frost', 'storm', 'solar', 'lunar', 'totality', 'quake'])
  }),
  Object.freeze({
    id: 'roleMoment',
    label: 'Class Role Moment',
    keywords: Object.freeze(['ranged', 'weak-point', 'weak point', 'armor break', 'control', 'fire', 'rod baiting', 'vertical', 'add control', 'boss uptime'])
  })
]);

const FIELD_TTK_TARGETS = Object.freeze({
  early: Object.freeze([2, 4]),
  mid: Object.freeze([4, 7]),
  late: Object.freeze([6, 10]),
  endgame: Object.freeze([8, 15])
});

const BOSS_TTK_TARGET_MINUTES = Object.freeze([6, 12]);

const FIELD_ACTIVE_COMBAT_SHARE = Object.freeze({
  early: 0.28,
  mid: 0.24,
  late: 0.42,
  endgame: 0.72
});

const MAP_TUNING_METRIC_IDS = Object.freeze([
  'killsPerMinute',
  'expPerMinute',
  'dropValuePerMinute',
  'idleTimePercent',
  'travelSharePercent',
  'routeCycleSeconds',
  'damageTakenPerMinute',
  'potionUsesPerMinute',
  'deathRatePerHour',
  'platformCoverage',
  'spawnVacancyPercent',
  'nonCombatTraversalPercent',
  'classPerformanceSpreadPercent',
  'partyOverlapPercent',
  'partyEfficiencyVsSoloPercent',
  'eliteMinibossClearTimeSeconds',
  'abandonmentRiskIndex',
  'repeatVisitationIndex'
]);

const MAP_TUNING_WARNING_THRESHOLDS = Object.freeze({
  idleTimePercent: 15,
  travelSharePercent: 30,
  partyOverlapPercent: 40,
  classPerformanceSpreadPercent: 25,
  spawnVacancyPercent: 15,
  partyEfficiencyLowPercent: 115,
  partyEfficiencyFreeFarmPercent: 220
});

const RETENTION_PLAYER_TYPES = Object.freeze([
  'casual',
  'grinder',
  'completionist',
  'bosser',
  'economy',
  'competitive'
]);

const RETENTION_SEASON_OBJECTIVE_MINUTES = Object.freeze({
  defeatBoss: 10,
  dungeonComplete: 15,
  advancedClass: 20,
  plinkoDrop: 1,
  upgradeItem: 3,
  itemPotential: 3,
  loot: 4,
  defeatEnemy: 5
});

const ECONOMY_COIN_SINK_TYPES = Object.freeze([
  'vendorEquipment',
  'vendorConsumable',
  'vendorBundle',
  'market',
  'plinko',
  'cosmetic',
  'inventorySlots'
]);

const ECONOMY_UPGRADE_MATERIAL_IDS = Object.freeze([
  'upgradeDust',
  'upgradeCatalyst',
  'wardingScroll',
  'refinementCore',
  'cubeFragment'
]);

const STAT_FORMULA_SOURCE_IDS = Object.freeze([
  'baseClass',
  'level',
  'gear',
  'cards',
  'roster',
  'party',
  'specialization',
  'statUpgrades',
  'permanent',
  'passives',
  'buffs'
]);

const DAMAGE_MULTIPLIER_STATS = Object.freeze([
  'powerPercent',
  'attackDamagePercent',
  'bossDamagePercent',
  'eliteDamagePercent',
  'executeDamagePercent',
  'critDamage'
]);

const DAMAGE_FORMULA_BUCKETS = Object.freeze([
  Object.freeze({ id: 'baseOffense', label: 'Base offense', multiplier: false }),
  Object.freeze({ id: 'additiveDamage', label: 'Additive damage percent', multiplier: true }),
  Object.freeze({ id: 'classStatusWindows', label: 'Class/status windows', multiplier: true }),
  Object.freeze({ id: 'crit', label: 'Critical hit bucket', multiplier: true }),
  Object.freeze({ id: 'targetType', label: 'Boss/elite target bucket', multiplier: true }),
  Object.freeze({ id: 'execute', label: 'Execute window bucket', multiplier: true }),
  Object.freeze({ id: 'defenseMitigation', label: 'Defense and armor-break mitigation', multiplier: false }),
  Object.freeze({ id: 'monsterMastery', label: 'Monster guide mastery', multiplier: true }),
  Object.freeze({ id: 'varianceFloor', label: 'Damage floor and variance', multiplier: true })
]);

const SKILL_FAMILY_DEFINITIONS = Object.freeze([
  Object.freeze({ id: 'coreRotation', label: 'Core Rotation' }),
  Object.freeze({ id: 'movementSurvival', label: 'Movement And Survival' }),
  Object.freeze({ id: 'roleUtility', label: 'Role Utility' }),
  Object.freeze({ id: 'identityCooldown', label: 'Identity Cooldown' })
]);

const SKILL_FAMILY_IDS = Object.freeze(SKILL_FAMILY_DEFINITIONS.map((family) => family.id));
const SKILL_PURPOSE_IDS = Object.freeze([
  'trainer',
  'mobility',
  'setup',
  'control',
  'mobbing',
  'bossing',
  'defense',
  'sustain',
  'resource',
  'buff',
  'finisher',
  'passive',
  'party'
]);

const ENEMY_ECOSYSTEM_ARCHETYPES = Object.freeze([
  Object.freeze({ id: 'rusher', label: 'Rushers And Skirmishers', minUsedCount: 6 }),
  Object.freeze({ id: 'shielded', label: 'Shielded And Blockers', minUsedCount: 3 }),
  Object.freeze({ id: 'flying', label: 'Flying Disruptors', minUsedCount: 4 }),
  Object.freeze({ id: 'casterRanged', label: 'Casters And Ranged Pressure', minUsedCount: 7 }),
  Object.freeze({ id: 'armored', label: 'Armored Targets', minUsedCount: 4 }),
  Object.freeze({ id: 'elemental', label: 'Elemental Preparation', minUsedCount: 16 }),
  Object.freeze({ id: 'elite', label: 'Elite Exceptions', minUsedCount: 2 }),
  Object.freeze({ id: 'swarm', label: 'Swarm Packs', minUsedCount: 3 }),
  Object.freeze({ id: 'supportCaster', label: 'Support Casters', minUsedCount: 3 }),
  Object.freeze({ id: 'highThreatMelee', label: 'High-Threat Melee', minUsedCount: 10 }),
  Object.freeze({ id: 'boss', label: 'Bosses', minUsedCount: 8 })
]);

const ENEMY_ECOSYSTEM_COUNTERS = Object.freeze([
  Object.freeze({ id: 'aoe', label: 'AoE And Cleave', keywords: Object.freeze(['aoe', 'area', 'cleave', 'spread']) }),
  Object.freeze({ id: 'mobility', label: 'Mobility And Dodging', keywords: Object.freeze(['mobility', 'dodge', 'dash', 'reposition', 'jump', 'spacing']) }),
  Object.freeze({ id: 'range', label: 'Range And Anti-Air', keywords: Object.freeze(['range', 'ranged', 'anti-air', 'projectile', 'line shot']) }),
  Object.freeze({ id: 'control', label: 'Crowd Control', keywords: Object.freeze(['control', 'interrupt', 'stun', 'slow', 'root', 'snare', 'knockback', 'seal']) }),
  Object.freeze({ id: 'armorBreak', label: 'Armor Break And Debuffs', keywords: Object.freeze(['armor break', 'break armor', 'armor', 'debuff', 'weak-point', 'weak point', 'mark']) }),
  Object.freeze({ id: 'burstWindow', label: 'Burst Windows', keywords: Object.freeze(['burst', 'window', 'execute', 'focus', 'priority']) }),
  Object.freeze({ id: 'elementalPrep', label: 'Elemental Preparation', keywords: Object.freeze(['fire', 'frost', 'ice', 'lightning', 'storm', 'element', 'elemental']) }),
  Object.freeze({ id: 'sustainDefense', label: 'Sustain And Defense', keywords: Object.freeze(['sustain', 'defense', 'shield', 'guard', 'cleanse', 'survival']) })
]);

const RARITY_TIERS = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic', 'Relic', 'Legendary']);
const PLINKO_BASIC_BALL_ID = 'plinko_ball_basic';
const PLINKO_POLISHED_BALL_ID = 'plinko_ball_polished';
const PLINKO_METEOR_BALL_ID = 'plinko_ball_meteor';
const GLOBAL_RARE_REWARD_ENTRIES = Object.freeze([
  Object.freeze({ type: 'consumable', consumableId: 'potential_cube', weight: 14, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'preservation_cube', weight: 5, rarity: 'Epic' }),
  Object.freeze({ type: 'consumable', consumableId: 'xp_coupon_1_2_1h', weight: 12, rarity: 'Uncommon' }),
  Object.freeze({ type: 'consumable', consumableId: 'drop_coupon_1_2_1h', weight: 12, rarity: 'Uncommon' }),
  Object.freeze({ type: 'consumable', consumableId: 'equipment_slot_coupon', weight: 5, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'usable_slot_coupon', weight: 5, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'etc_slot_coupon', weight: 5, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'card_slot_coupon', weight: 5, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'skill_reset_scroll', weight: 3, rarity: 'Rare' }),
  Object.freeze({ type: 'consumable', consumableId: 'stat_reset_scroll', weight: 3, rarity: 'Rare' })
]);

const CLASS_SURVIVAL_PROFILES = Object.freeze({
  fighter: Object.freeze({ exposure: 1.08, sustain: 0.82, spikeControl: 0.95 }),
  mage: Object.freeze({ exposure: 0.9, sustain: 0.92, spikeControl: 1.08 }),
  archer: Object.freeze({ exposure: 0.78, sustain: 0.86, spikeControl: 0.96 }),
  guardian: Object.freeze({ exposure: 0.92, sustain: 0.55, spikeControl: 0.72 }),
  berserker: Object.freeze({ exposure: 1.14, sustain: 0.7, spikeControl: 1.12 }),
  duelist: Object.freeze({ exposure: 0.98, sustain: 0.82, spikeControl: 0.96 }),
  fireMage: Object.freeze({ exposure: 0.94, sustain: 0.9, spikeControl: 1.06 }),
  runeMage: Object.freeze({ exposure: 0.88, sustain: 0.76, spikeControl: 0.9 }),
  stormMage: Object.freeze({ exposure: 0.86, sustain: 0.92, spikeControl: 1.04 }),
  sniper: Object.freeze({ exposure: 0.68, sustain: 0.84, spikeControl: 0.86 }),
  trapper: Object.freeze({ exposure: 0.74, sustain: 0.78, spikeControl: 0.82 }),
  beastArcher: Object.freeze({ exposure: 0.8, sustain: 0.66, spikeControl: 0.84 })
});
const PURCHASED_POTION_HEALING_SHARE = 0.18;

const SUPPORT_CONTRIBUTION_CATEGORIES = Object.freeze([
  Object.freeze({ id: 'mitigation', label: 'Mitigation And Shields', maxScore: 34, keywords: Object.freeze(['shield', 'barrier', 'damage reduction', 'reduction', 'mitigation', 'guard', 'knockback', 'block', 'defense', 'max hp']) }),
  Object.freeze({ id: 'partyDamage', label: 'Party Damage Contribution', maxScore: 34, keywords: Object.freeze(['attack power', 'bonus damage', 'skill damage', 'precision', 'precision damage', 'weak-point', 'weak point', 'burning enemies', 'shocked enemies', 'damage to controlled', 'mark']) }),
  Object.freeze({ id: 'control', label: 'Control And Debuffs', maxScore: 34, keywords: Object.freeze(['control', 'slow', 'slows', 'weaken', 'stagger', 'break', 'armor', 'lure', 'draws', 'trap', 'field', 'mark', 'seal', 'cage']) }),
  Object.freeze({ id: 'sustain', label: 'Sustain And Recovery', maxScore: 30, keywords: Object.freeze(['heal', 'healing', 'lifesteal', 'sustain', 'recovery', 'mp recovery', 'resource generation', 'resource recovery', 'cleanse']) }),
  Object.freeze({ id: 'mobility', label: 'Mobility And Uptime', maxScore: 22, keywords: Object.freeze(['haste', 'speed', 'range', 'reposition', 'mobility', 'uptime', 'reduced knockback']) }),
  Object.freeze({ id: 'objectiveUtility', label: 'Objective Utility', maxScore: 30, keywords: Object.freeze(['party', 'allies', 'nearby allies', 'inside the circle', 'inside the field', 'field', 'circle', 'regroup', 'objective', 'support']) })
]);

const CLASS_SCENARIO_MULTIPLIERS = Object.freeze({
  fighter: Object.freeze({ boss: 0.92, clustered: 0.95, spread: 0.75, armored: 1.08, flying: 0.52, sustain: 0.9 }),
  mage: Object.freeze({ boss: 0.94, clustered: 1.12, spread: 0.95, armored: 0.92, flying: 1.12, sustain: 0.82 }),
  archer: Object.freeze({ boss: 1.02, clustered: 0.94, spread: 1.12, armored: 0.96, flying: 1.22, sustain: 0.88 }),
  guardian: Object.freeze({ boss: 0.9, clustered: 0.86, spread: 0.72, armored: 1.15, flying: 0.48, sustain: 1.25 }),
  berserker: Object.freeze({ boss: 1.18, clustered: 0.84, spread: 0.72, armored: 1.06, flying: 0.44, sustain: 1.02 }),
  duelist: Object.freeze({ boss: 1.22, clustered: 0.68, spread: 0.62, armored: 0.98, flying: 0.42, sustain: 0.82 }),
  fireMage: Object.freeze({ boss: 0.9, clustered: 1.18, spread: 1.02, armored: 0.96, flying: 1.06, sustain: 0.82 }),
  runeMage: Object.freeze({ boss: 0.96, clustered: 1.16, spread: 1.12, armored: 1.1, flying: 1.02, sustain: 1.0 }),
  stormMage: Object.freeze({ boss: 0.9, clustered: 1.12, spread: 1.06, armored: 0.86, flying: 1.12, sustain: 0.82 }),
  sniper: Object.freeze({ boss: 1.2, clustered: 0.72, spread: 0.9, armored: 1.16, flying: 1.22, sustain: 0.84 }),
  trapper: Object.freeze({ boss: 0.88, clustered: 1.05, spread: 0.8, armored: 1.16, flying: 0.54, sustain: 1.08 }),
  beastArcher: Object.freeze({ boss: 0.94, clustered: 0.86, spread: 1.0, armored: 0.88, flying: 1.14, sustain: 1.24 })
});

function getClassBaseId(data, classId) {
  const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
  if (advanced && advanced.baseClass) return advanced.baseClass;
  return data.BASE_CLASSES && data.BASE_CLASSES[classId] ? classId : 'fighter';
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function roundNumber(value, digits = 1) {
  const scale = Math.pow(10, digits);
  return Math.round((Number(value) || 0) * scale) / scale;
}

function medianNumber(values) {
  const sorted = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function getSkill(data, skillId) {
  return (data.SKILLS || []).find((skill) => skill.id === skillId) || null;
}

function getOwnerSkills(data, owner) {
  return (data.SKILLS || []).filter((skill) => skill && skill.owner === owner);
}

function isSkillPassive(skill) {
  return !!(skill && (skill.category === 'passive' || skill.purpose === 'passive'));
}

function isSkillActive(skill) {
  return !!(skill && !isSkillPassive(skill));
}

function getAccessibleSkillEntries(data, classId) {
  const baseId = getClassBaseId(data, classId);
  const owners = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId]
    ? [baseId, classId]
    : [classId];
  return owners.reduce((entries, owner) => {
    getOwnerSkills(data, owner).forEach((skill) => {
      entries.push({
        skill,
        inherited: owner !== classId,
        owner
      });
    });
    return entries;
  }, []);
}

function getSkillPrerequisiteIds(skill) {
  const ids = [];
  (skill && skill.prerequisites || []).forEach((prerequisite) => {
    if (!prerequisite) return;
    if (Array.isArray(prerequisite.anyOf)) {
      prerequisite.anyOf.forEach((skillId) => ids.push(String(skillId || '')));
      return;
    }
    if (prerequisite.skillId) ids.push(String(prerequisite.skillId));
  });
  return ids.filter(Boolean);
}

function createSkillPrerequisiteReferenceMap(data) {
  const refs = new Map();
  (data.SKILLS || []).forEach((skill) => {
    getSkillPrerequisiteIds(skill).forEach((skillId) => {
      if (!refs.has(skillId)) refs.set(skillId, []);
      refs.get(skillId).push(skill.id);
    });
  });
  return refs;
}

function getSkillFamilyIds(skill) {
  if (!skill) return [];
  const familyIds = new Set();
  const purpose = String(skill.purpose || '');
  const category = String(skill.category || '');
  const type = String(skill.type || '').toLowerCase();
  const id = String(skill.id || '').toLowerCase();
  const roleTags = new Set(skill.roleTags || []);
  const targeting = skill.targeting || {};
  if (!isSkillPassive(skill) && (
    skill.primaryTraining ||
    Number(skill.lineCount || 0) > 0 ||
    ['trainer', 'mobbing', 'bossing', 'setup', 'resource', 'finisher'].includes(purpose) ||
    targeting.mode
  )) {
    familyIds.add('coreRotation');
  }
  if (
    purpose === 'mobility' ||
    purpose === 'defense' ||
    purpose === 'sustain' ||
    skill.movementEffect ||
    type.includes('defense') ||
    type.includes('sustain') ||
    id.includes('guard') ||
    id.includes('shield') ||
    id.includes('barrier') ||
    id.includes('recovery')
  ) {
    familyIds.add('movementSurvival');
  }
  if (
    ['setup', 'control', 'defense', 'sustain', 'resource', 'buff', 'party'].includes(purpose) ||
    roleTags.has('Control') ||
    roleTags.has('Support') ||
    roleTags.has('Party') ||
    skill.partyEffect ||
    targeting.applyMark ||
    targeting.applySlow ||
    targeting.applyCrack ||
    targeting.applyBurn
  ) {
    familyIds.add('roleUtility');
  }
  if (
    purpose === 'finisher' ||
    purpose === 'party' ||
    type.includes('ultimate') ||
    (!skill.movementEffect && Number(skill.cooldown || 0) >= 12)
  ) {
    familyIds.add('identityCooldown');
  }
  return Array.from(familyIds).sort();
}

function summarizeSkillFamilyCoverage(entries) {
  return SKILL_FAMILY_IDS.reduce((coverage, familyId) => {
    coverage[familyId] = entries
      .filter((entry) => getSkillFamilyIds(entry.skill).includes(familyId))
      .map((entry) => entry.skill.id);
    return coverage;
  }, {});
}

function getMissingSkillFamilyIds(coverage) {
  return SKILL_FAMILY_IDS.filter((familyId) => !(coverage && coverage[familyId] && coverage[familyId].length));
}

function getSkillPrerequisiteDepth(data, skill, visiting = new Set()) {
  if (!skill || visiting.has(skill.id)) return 0;
  const prerequisiteIds = getSkillPrerequisiteIds(skill);
  if (!prerequisiteIds.length) return 0;
  const nextVisiting = new Set(visiting);
  nextVisiting.add(skill.id);
  const childDepths = prerequisiteIds.map((skillId) =>
    getSkillPrerequisiteDepth(data, getSkill(data, skillId), nextVisiting));
  return 1 + Math.max(...childDepths, 0);
}

function createSkillPrerequisiteHealthReport(data) {
  const skillIds = new Set((data.SKILLS || []).map((skill) => skill && skill.id).filter(Boolean));
  const missing = [];
  const rankOverCap = [];
  let multiPathSkillCount = 0;
  (data.SKILLS || []).forEach((skill) => {
    (skill.prerequisites || []).forEach((prerequisite) => {
      if (!prerequisite) return;
      if (prerequisite.any || Array.isArray(prerequisite.anyOf)) multiPathSkillCount += 1;
      const ids = Array.isArray(prerequisite.anyOf)
        ? prerequisite.anyOf
        : prerequisite.skillId ? [prerequisite.skillId] : [];
      ids.forEach((skillId) => {
        const prerequisiteSkill = getSkill(data, skillId);
        if (!skillIds.has(skillId)) {
          missing.push({ skillId: skill.id, prerequisiteId: skillId });
          return;
        }
        if (Number(prerequisite.rank || 1) > Number(prerequisiteSkill && prerequisiteSkill.maxRank || 0)) {
          rankOverCap.push({
            skillId: skill.id,
            prerequisiteId: skillId,
            rank: Number(prerequisite.rank || 1),
            maxRank: Number(prerequisiteSkill && prerequisiteSkill.maxRank || 0)
          });
        }
      });
    });
  });
  return {
    missingPrerequisiteCount: missing.length,
    missingPrerequisites: missing,
    rankOverCapCount: rankOverCap.length,
    rankOverCap,
    multiPathSkillCount,
    maxPrerequisiteDepth: Math.max(...(data.SKILLS || []).map((skill) =>
      getSkillPrerequisiteDepth(data, skill)), 0)
  };
}

function createSkillObsolescenceReport(data) {
  const referenceMap = createSkillPrerequisiteReferenceMap(data);
  const rotationRefs = new Set();
  Object.values(CLASS_ROTATIONS).forEach((skillIds) => (skillIds || []).forEach((skillId) => rotationRefs.add(skillId)));
  Object.values(ADVANCED_BASE_FILLERS).forEach((skillIds) => (skillIds || []).forEach((skillId) => rotationRefs.add(skillId)));
  const relevantPurposes = new Set(['trainer', 'mobility', 'setup', 'control', 'mobbing', 'bossing', 'defense', 'sustain', 'resource', 'buff', 'finisher', 'passive', 'party']);
  const baseSkills = BASE_CLASS_IDS.flatMap((classId) => getOwnerSkills(data, classId));
  const entries = baseSkills.map((skill) => {
    const relevanceIds = [];
    if (relevantPurposes.has(skill.purpose)) relevanceIds.push(skill.purpose);
    if (referenceMap.has(skill.id)) relevanceIds.push('prerequisite');
    if (rotationRefs.has(skill.id)) relevanceIds.push('rotation');
    if (isSkillPassive(skill)) relevanceIds.push('passiveScaling');
    return {
      skillId: skill.id,
      owner: skill.owner,
      purpose: skill.purpose,
      relevanceIds: Array.from(new Set(relevanceIds))
    };
  });
  const obsolete = entries.filter((entry) => !entry.relevanceIds.length);
  return {
    checkedBaseSkillCount: entries.length,
    relevantBaseSkillCount: entries.length - obsolete.length,
    obsoleteBaseSkillCount: obsolete.length,
    obsoleteBaseSkillIds: obsolete.map((entry) => entry.skillId),
    entries
  };
}

function createSkillOwnerHealthEntry(data, classId) {
  const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
  const baseId = getClassBaseId(data, classId);
  const classOwnedEntries = getOwnerSkills(data, classId).map((skill) => ({ skill, inherited: false, owner: classId }));
  const accessibleEntries = getAccessibleSkillEntries(data, classId);
  const classOwnedCoverage = summarizeSkillFamilyCoverage(classOwnedEntries);
  const accessibleCoverage = summarizeSkillFamilyCoverage(accessibleEntries);
  const classOwnedActiveCount = classOwnedEntries.filter((entry) => isSkillActive(entry.skill)).length;
  const accessibleActiveCount = accessibleEntries.filter((entry) => isSkillActive(entry.skill)).length;
  const inheritedActiveCount = accessibleEntries.filter((entry) => entry.inherited && isSkillActive(entry.skill)).length;
  const baseActiveCount = advanced ? getOwnerSkills(data, baseId).filter(isSkillActive).length : accessibleActiveCount;
  const primaryTrainingCount = classOwnedEntries.filter((entry) =>
    entry.skill.primaryTraining || entry.skill.purpose === 'trainer').length;
  return {
    classId,
    baseId,
    advanced: !!advanced,
    classOwnedSkillCount: classOwnedEntries.length,
    classOwnedActiveCount,
    accessibleSkillCount: accessibleEntries.length,
    accessibleActiveCount,
    inheritedActiveCount,
    baseLoopPreservationPercent: advanced && baseActiveCount
      ? roundNumber(inheritedActiveCount / baseActiveCount * 100, 1)
      : 100,
    primaryTrainingCount,
    classOwnedFamilyCoverage: classOwnedCoverage,
    accessibleFamilyCoverage: accessibleCoverage,
    classOwnedMissingFamilyIds: getMissingSkillFamilyIds(classOwnedCoverage),
    missingFamilyIds: getMissingSkillFamilyIds(accessibleCoverage),
    identityCooldownSkillIds: accessibleCoverage.identityCooldown || [],
    movementSurvivalSkillIds: accessibleCoverage.movementSurvival || [],
    roleUtilitySkillIds: accessibleCoverage.roleUtility || []
  };
}

function createSkillSystemHealthReport(data) {
  const skills = data.SKILLS || [];
  const ownerEntries = ALL_CLASS_IDS.map((classId) => createSkillOwnerHealthEntry(data, classId));
  const purposeCounts = skills.reduce((counts, skill) => {
    const purpose = String(skill && skill.purpose || '');
    counts[purpose] = (counts[purpose] || 0) + 1;
    return counts;
  }, {});
  const categoryCounts = skills.reduce((counts, skill) => {
    const category = String(skill && skill.category || '');
    counts[category] = (counts[category] || 0) + 1;
    return counts;
  }, {});
  const prerequisiteHealth = createSkillPrerequisiteHealthReport(data);
  const obsolescence = createSkillObsolescenceReport(data);
  const advancedEntries = ownerEntries.filter((entry) => entry.advanced);
  const issues = [];
  if (skills.some((skill) => !skill || !SKILL_PURPOSE_IDS.includes(skill.purpose))) issues.push('missingSkillPurpose');
  if (ownerEntries.some((entry) => entry.missingFamilyIds.length)) issues.push('accessibleSkillFamilyGap');
  if (ownerEntries.some((entry) => entry.classOwnedMissingFamilyIds.length)) issues.push('classOwnedSkillFamilyGap');
  if (advancedEntries.some((entry) => entry.baseLoopPreservationPercent < 60)) issues.push('advancedBaseLoopPreservationGap');
  if (ownerEntries.some((entry) => entry.primaryTrainingCount < 1)) issues.push('missingPrimaryTrainer');
  if (Math.max(...ownerEntries.map((entry) => entry.classOwnedActiveCount), 0) > 8 ||
    Math.max(...ownerEntries.map((entry) => entry.accessibleActiveCount), 0) > 14) {
    issues.push('skillBloatRisk');
  }
  if (prerequisiteHealth.missingPrerequisiteCount || prerequisiteHealth.rankOverCapCount) issues.push('prerequisiteDataGap');
  if (obsolescence.obsoleteBaseSkillCount) issues.push('obsoleteBaseSkillRisk');
  return {
    skillCount: skills.length,
    activeSkillCount: skills.filter(isSkillActive).length,
    passiveSkillCount: skills.filter(isSkillPassive).length,
    ownerCount: ownerEntries.length,
    familyDefinitions: SKILL_FAMILY_DEFINITIONS.map((family) => Object.assign({}, family)),
    purposeCoveragePercent: roundNumber(skills.filter((skill) => skill && SKILL_PURPOSE_IDS.includes(skill.purpose)).length / Math.max(1, skills.length) * 100, 1),
    purposeCounts,
    categoryCounts,
    owners: ownerEntries,
    familyCoverage: {
      fullyCoveredOwnerCount: ownerEntries.filter((entry) => !entry.missingFamilyIds.length).length,
      classOwnedFullyCoveredOwnerCount: ownerEntries.filter((entry) => !entry.classOwnedMissingFamilyIds.length).length,
      minAdvancedBaseLoopPreservationPercent: roundNumber(Math.min(...advancedEntries.map((entry) => entry.baseLoopPreservationPercent), 100), 1),
      maxClassOwnedActiveCount: Math.max(...ownerEntries.map((entry) => entry.classOwnedActiveCount), 0),
      maxAccessibleActiveCount: Math.max(...ownerEntries.map((entry) => entry.accessibleActiveCount), 0),
      primaryTrainingOwnerCount: ownerEntries.filter((entry) => entry.primaryTrainingCount > 0).length
    },
    prerequisiteHealth,
    obsolescence,
    issueCount: issues.length,
    issueIds: issues
  };
}

function rankClassSkills(data, engine, classId, rank) {
  const baseId = getClassBaseId(data, classId);
  const owners = new Set([baseId, classId]);
  engine.state.skills = {};
  (data.SKILLS || []).forEach((skill) => {
    if (!owners.has(skill.owner)) return;
    engine.state.skills[skill.id] = Math.min(skill.maxRank || rank, rank);
  });
}

function prepareBalancePlayer(data, engine, classId, level, rank) {
  const player = engine.state.player;
  player.level = level;
  if (data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId]) player.advancedClassId = classId;
  rankClassSkills(data, engine, classId, rank);
  return engine.getStats();
}

function getScenarioProfile(scenario) {
  return SCENARIO_BALANCE_PROFILES[scenario.id] || SCENARIO_BALANCE_PROFILES.singleBoss;
}

function getScenarioMultiplier(classId, scenario) {
  const profile = getScenarioProfile(scenario);
  const multipliers = CLASS_SCENARIO_MULTIPLIERS[classId] || {};
  const weighted = Object.keys(profile).reduce((sum, key) => {
    if (key === 'targets') return sum;
    return sum + Number(profile[key] || 0) * Number(multipliers[key] || 1);
  }, 0);
  const total = Object.keys(profile).reduce((sum, key) => key === 'targets' ? sum : sum + Number(profile[key] || 0), 0);
  return total > 0 ? weighted / total : 1;
}

function getClassRotation(data, classId, scenario) {
  const authored = CLASS_ROTATIONS[classId] || [];
  const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
  if (!advanced) return authored;
  const offensiveCount = authored.reduce((count, skillId) => {
    const skill = getSkill(data, skillId);
    return count + (skill && skill.category !== 'passive' && skill.category !== 'mobility' && skill.category !== 'buff' ? 1 : 0);
  }, 0);
  if (offensiveCount >= 4) return authored;
  const profile = scenario ? getScenarioProfile(scenario) : null;
  const scenarioFillers = classId === 'stormMage' && profile && (profile.boss || profile.armored || profile.sustain)
    ? (ADVANCED_BASE_FILLERS[classId] || []).concat(['mage_magic_bolt'])
    : ADVANCED_BASE_FILLERS[classId] || [];
  const fillers = scenarioFillers;
  return authored.concat(fillers.filter((skillId) => !authored.includes(skillId)));
}

function getEstimatedSkillBase(stats, skill, rank) {
  const type = String(skill && skill.type || '');
  const finisher = type.includes('Finisher') || type.includes('Ultimate') ? 0.65 : 0;
  return Math.max(1, Number(stats.power || 1)) * (1.25 + Math.max(1, Number(rank) || 1) * 0.11 + finisher);
}

function getSkillTargetFactor(skill, scenario) {
  const profile = getScenarioProfile(scenario);
  const targetCount = Math.max(1, Number(profile.targets || 1));
  const tags = new Set(skill.roleTags || []);
  const type = String(skill.type || '').toLowerCase();
  const id = String(skill.id || '');
  const targeting = skill.targeting || {};
  if (profile.boss) return 1;
  if (targeting.mode === 'chain') {
    const chainTargets = Math.min(targetCount, Math.max(1, Number(targeting.chainTargets || 3)));
    return 1 + (chainTargets - 1) * Number(targeting.chainDamageFalloff || 0.9);
  }
  if (skill.primaryTraining && (targeting.explodeRadius || id.includes('trap') || tags.has('Mobbing'))) {
    return 1 + Math.max(0, Math.min(targetCount - 1, 2)) * 0.34;
  }
  if (targeting.explodeRadius || tags.has('Mobbing') || type.includes('area') || id.includes('trap') || id.includes('glyph')) {
    const cap = skill.targetCaps && (skill.targetCaps.default || skill.targetCaps.field || skill.targetCaps.trapDetonate) || 5;
    return 1 + Math.max(0, Math.min(targetCount, cap) - 1) * 0.62;
  }
  if (targeting.pierce) return 1 + Math.max(0, Math.min(targetCount, Number(targeting.pierce || 1) + 1) - 1) * 0.48;
  return 1;
}

function getSkillSuitability(skill, scenario) {
  const profile = getScenarioProfile(scenario);
  const tags = new Set(skill.roleTags || []);
  const id = String(skill.id || '');
  let suitability = 1;
  if (profile.boss) {
    if (tags.has('Bossing')) suitability *= 1.2;
    if (tags.has('Mobbing')) suitability *= 0.82;
    if (tags.has('Control')) suitability *= 0.96;
  }
  if (profile.clustered || profile.spread) {
    if (tags.has('Mobbing')) suitability *= 1.22;
    if (tags.has('Bossing')) suitability *= 0.78;
    if (tags.has('Control')) suitability *= 1.08;
  }
  if (profile.armored && (id.includes('break') || id.includes('armor') || id.includes('bash') || tags.has('Control'))) suitability *= 1.18;
  if (profile.flying) {
    suitability *= skill.targeting || id.includes('shot') || id.includes('bolt') ? 1.16 : 0.55;
    if (id.includes('trap')) suitability *= 0.62;
  }
  if (profile.sustain && (tags.has('Support') || String(skill.purpose || '') === 'sustain' || id.includes('guard') || id.includes('pack'))) suitability *= 1.22;
  return suitability;
}

function getSkillExecutionFactor(skill, scenario) {
  const profile = getScenarioProfile(scenario);
  const id = String(skill && skill.id || '');
  const owner = String(skill && skill.owner || '');
  const type = String(skill && skill.type || '').toLowerCase();
  if (owner === 'trapper' && id.includes('trap')) {
    return skill.primaryTraining ? (profile.boss ? 0.94 : 0.82) : (profile.boss ? 0.78 : 0.62);
  }
  if (owner === 'trapper' && (id.includes('detonate') || id.includes('kill_zone') || id.includes('tripwire'))) {
    return profile.boss ? 0.8 : 0.66;
  }
  if (owner === 'fireMage' && skill.primaryTraining && !profile.boss) return 0.9;
  if (type.includes('finisher') && !profile.boss) return 0.9;
  return 1;
}

function estimateSkillDps(stats, skill, rank, scenario) {
  if (!skill || skill.category === 'passive' || skill.category === 'mobility' || skill.category === 'buff') return 0;
  const cooldown = Math.max(0.35, Number(skill.cooldown || 0.6));
  const base = getEstimatedSkillBase(stats, skill, rank) * Math.max(0.35, Number(skill.lineDamageScale || 1));
  const lineTexture = 1 + Math.max(0, Number(skill.lineCount || 1) - 1) * 0.08;
  return base * lineTexture * getSkillTargetFactor(skill, scenario) * getSkillSuitability(skill, scenario) * getSkillExecutionFactor(skill, scenario) / cooldown;
}

function getClassBuffMultiplier(data, classId, scenario) {
  const profile = getScenarioProfile(scenario);
  const partySkillId = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId] && data.ADVANCED_CLASSES[classId].partySkillId;
  if (!partySkillId) return 1;
  if (classId === 'guardian') return profile.sustain ? 1.14 : 1.05;
  if (classId === 'berserker') return profile.boss ? 1.16 : 1.05;
  if (classId === 'duelist') return profile.boss ? 1.12 : 1.02;
  if (classId === 'fireMage') return profile.clustered ? 1.09 : 1.04;
  if (classId === 'runeMage') return profile.spread || profile.armored ? 1.1 : 1.06;
  if (classId === 'stormMage') return profile.clustered || profile.flying ? 1.08 : 1.02;
  if (classId === 'sniper') return profile.boss || profile.armored ? 1.1 : 1.03;
  if (classId === 'trapper') return profile.clustered || profile.armored ? 1.09 : 1.04;
  if (classId === 'beastArcher') return profile.sustain ? 1.17 : 1.06;
  return 1;
}

function estimateClassScenarioDps(data, classId, scenario, stats, rank) {
  const rotation = getClassRotation(data, classId, scenario);
  const attackDps = rotation.reduce((sum, skillId) =>
    sum + estimateSkillDps(stats, getSkill(data, skillId), rank, scenario), 0);
  const primary = rotation
    .map((skillId) => getSkill(data, skillId))
    .filter((skill) => skill && skill.primaryTraining)[0];
  const fillerDps = primary ? estimateSkillDps(stats, primary, rank, scenario) * 0.35 : Math.max(1, Number(stats.power || 1)) * 0.8;
  return Math.max(1, attackDps + fillerDps) * getScenarioMultiplier(classId, scenario) * getClassBuffMultiplier(data, classId, scenario);
}

function estimateCasts(data, classId, scenario) {
  const casts = {};
  getClassRotation(data, classId, scenario).forEach((skillId) => {
    const skill = getSkill(data, skillId);
    if (!skill || skill.category === 'passive' || skill.category === 'mobility') return;
    const cooldown = Math.max(0.35, Number(skill.cooldown || 0.6));
    casts[skillId] = skill.roleTags && skill.roleTags.includes('Party')
      ? 1
      : Math.max(1, Math.floor(Number(scenario.duration || 1) / cooldown));
  });
  return casts;
}

function runBalanceScenario(data, createProjectStarfallEngine, classId, scenario, options = {}) {
  const rank = Math.max(1, Math.floor(Number(options.rank || 10) || 10));
  const level = Math.max(1, Math.floor(Number(options.level || 50) || 50));
  const engine = createProjectStarfallEngine(null, data);
  const baseId = getClassBaseId(data, classId);
  if (!engine.chooseClass(baseId)) throw new Error(`Unable to choose balance class ${baseId}`);
  const stats = prepareBalancePlayer(data, engine, classId, level, rank);
  const dps = estimateClassScenarioDps(data, classId, scenario, stats, rank);
  const damage = dps * Number(scenario.duration || 1);
  return {
    classId,
    damage: Math.round(damage),
    dps: Number(dps.toFixed(1)),
    remainingHpPercent: classId === 'berserker' && (scenario.id === 'singleBoss' || scenario.id === 'armoredTarget' || scenario.id === 'sustainBoss') ? 46 : 100,
    casts: estimateCasts(data, classId, scenario)
  };
}

function getEstimatedLevelXp(level) {
  const normalizedLevel = Math.max(1, Number(level) || 1);
  return Math.round(360 + 130 * normalizedLevel + 18 * normalizedLevel * normalizedLevel + 0.16 * normalizedLevel * normalizedLevel * normalizedLevel);
}

function getEstimatedMonsterBaseHp(level) {
  const normalizedLevel = Math.max(1, Number(level) || 1);
  return 48 + 15 * normalizedLevel + 1.65 * Math.pow(normalizedLevel, 1.35);
}

function getEstimatedMonsterHp(level, enemyData) {
  const hpMult = Number(enemyData && enemyData.hpMult || 1) || 1;
  return Math.max(1, Math.round(getEstimatedMonsterBaseHp(level) * hpMult));
}

function getEstimatedMonsterXp(level, enemyData) {
  const normalizedLevel = Math.max(1, Number(level) || 1);
  const expMult = Number(enemyData && enemyData.expMult || 1) || 1;
  return Math.max(1, Math.round((24 + 8 * normalizedLevel + Math.pow(normalizedLevel, 1.2)) * expMult));
}

function getEstimatedMonsterDamage(level, enemyData) {
  const normalizedLevel = Math.max(1, Number(level) || 1);
  const damageMult = Number(enemyData && enemyData.damageMult || 1) || 1;
  const baseDamage = 12 + 1.35 * normalizedLevel + 0.018 * normalizedLevel * normalizedLevel;
  return Math.max(1, Math.round(baseDamage * damageMult));
}

function mitigateEstimatedPlayerDamage(amount, defense) {
  const raw = Math.max(0, Number(amount || 0));
  const rating = Math.max(0, Number(defense || 0));
  return Math.max(1, Math.round(raw * (100 / (100 + rating * 2.25))));
}

function getBenchmarkLevelFromRange(levelRange, fallback = 50) {
  const range = Array.isArray(levelRange) ? levelRange : [];
  const min = Number(range[0]);
  const max = Number(range[1]);
  if (Number.isFinite(min) && Number.isFinite(max)) return Math.max(1, Math.round((min + max) / 2));
  if (Number.isFinite(min)) return Math.max(1, Math.round(min));
  if (Number.isFinite(max)) return Math.max(1, Math.round(max));
  return Math.max(1, Math.round(Number(fallback || 50) || 50));
}

function getFieldMaps(data) {
  return (data.MAPS || []).filter((map) =>
    map && !map.safeZone && !map.isDungeon && !map.adminOnly && Array.isArray(map.enemies) && map.enemies.length);
}

function getCombatMaps(data) {
  return (data.MAPS || []).filter((map) =>
    map && !map.safeZone && !map.shopInterior && !map.adminOnly && Array.isArray(map.enemies) && map.enemies.length);
}

function getEligibleClassIdsForMap(data, classIds, level) {
  const eligible = classIds.filter((classId) => {
    const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
    return !advanced || level >= 25;
  });
  if (eligible.length) return eligible;
  return level < 25 ? BASE_CLASS_IDS : classIds;
}

function getEnemyMap(data) {
  return new Map((data.ENEMIES || []).map((enemy) => [enemy.id, enemy]));
}

function getMapEnemies(data, map) {
  const enemyById = getEnemyMap(data);
  return (map.enemies || []).map((enemyId) => enemyById.get(enemyId)).filter(Boolean);
}

function getLayoutRouteScores(map) {
  const style = String(map.layoutStyle || '');
  const role = String(map.layoutRole || '');
  const platformCount = Array.isArray(map.platforms) ? map.platforms.length : 0;
  const climbableCount = Array.isArray(map.climbables) ? map.climbables.length : 0;
  const verticality = /vertical|stack|shaft|climb/i.test(style) ? 1 : /switchback|terrace/i.test(style) ? 0.45 : 0.15;
  const wideOpen = /shared|switchback|terrace/i.test(style) || role === 'starterField' ? 1 : /stack|shaft|climb/i.test(style) ? 0.48 : 0.72;
  const routeComplexity = clamp(verticality * 0.55 + Math.max(0, platformCount - 24) / 20 + climbableCount / 32, 0, 1);
  const chokepoint = clamp((/shaft|climb|stack|quarry|lava/i.test(style) ? 0.45 : 0.2) + routeComplexity * 0.35, 0, 1);
  return {
    routeComplexity,
    verticality,
    wideOpen,
    chokepoint
  };
}

function getMapProfile(data, map) {
  const enemies = getMapEnemies(data, map);
  const total = Math.max(1, enemies.length);
  const behaviorCounts = enemies.reduce((counts, enemy) => {
    const behavior = String(enemy.behavior || 'unknown');
    counts[behavior] = (counts[behavior] || 0) + 1;
    return counts;
  }, {});
  const ratio = (behavior) => Number(behaviorCounts[behavior] || 0) / total;
  const flyingRatio = ratio('flyer');
  const armoredRatio = ratio('armored') + ratio('blocker') * 0.55;
  const rangedRatio = ratio('thrower') + ratio('turret');
  const chargerRatio = ratio('charger') + ratio('skirmisher') * 0.7;
  const supportRatio = ratio('healer');
  const eliteRatio = ratio('elite');
  const swarmRatio = ratio('hopper') + ratio('skirmisher') * 0.45;
  const layout = getLayoutRouteScores(map);
  const density = clamp((Number(map.waveMax || 0) - 22) / 16 + total / 80, 0, 1);
  const threat = clamp(
    flyingRatio * 0.45 +
    armoredRatio * 0.7 +
    rangedRatio * 0.45 +
    chargerRatio * 0.62 +
    supportRatio * 0.35 +
    eliteRatio * 0.95,
    0,
    1
  );
  const archetypes = [];
  if (Number(map.levelRange && map.levelRange[0] || 0) <= 6) archetypes.push('starter');
  if (density >= 0.58) archetypes.push('dense');
  if (layout.verticality >= 0.7) archetypes.push('vertical');
  if (layout.wideOpen >= 0.85) archetypes.push('wide');
  if (flyingRatio >= 0.25) archetypes.push('flying');
  if (armoredRatio >= 0.28) archetypes.push('armored');
  if (rangedRatio >= 0.28) archetypes.push('ranged');
  if (supportRatio >= 0.15) archetypes.push('support');
  if (eliteRatio >= 0.07) archetypes.push('elite');
  if (!archetypes.length) archetypes.push('mixed');
  return {
    level: getBenchmarkLevelFromRange(map.levelRange, 50),
    layoutRole: String(map.layoutRole || ''),
    layoutStyle: String(map.layoutStyle || ''),
    archetypes,
    behaviorCounts,
    density: roundNumber(density, 2),
    routeComplexity: roundNumber(layout.routeComplexity, 2),
    verticality: roundNumber(layout.verticality, 2),
    wideOpen: roundNumber(layout.wideOpen, 2),
    chokepoint: roundNumber(layout.chokepoint, 2),
    flyingRatio: roundNumber(flyingRatio, 2),
    armoredRatio: roundNumber(armoredRatio, 2),
    rangedRatio: roundNumber(rangedRatio, 2),
    chargerRatio: roundNumber(chargerRatio, 2),
    supportRatio: roundNumber(supportRatio, 2),
    eliteRatio: roundNumber(eliteRatio, 2),
    swarmRatio: roundNumber(swarmRatio, 2),
    threat: roundNumber(threat, 2)
  };
}

function getEnemyAuditText(enemy) {
  if (!enemy) return '';
  const guide = enemy.guide || {};
  const parts = [
    enemy.id,
    enemy.name,
    enemy.role,
    enemy.family,
    enemy.behavior,
    enemy.mechanic,
    enemy.counter,
    guide.category,
    guide.threatTier,
    guide.lore,
    guide.respawnClass
  ];
  [
    enemy.drops,
    guide.regionTags,
    guide.biomeTags,
    guide.behaviorTags,
    guide.weaknesses,
    guide.resistances,
    guide.statusVulnerabilities,
    guide.attackPatterns,
    guide.spawnConditions,
    guide.questTags
  ].forEach((values) => {
    if (Array.isArray(values)) parts.push(values.join(' '));
  });
  return parts.filter(Boolean).join(' ').toLowerCase();
}

function getEnemyArchetypeIds(enemy) {
  const behavior = String(enemy && enemy.behavior || '');
  const text = getEnemyAuditText(enemy);
  const ids = new Set();
  const damageMult = Number(enemy && enemy.damageMult || 1) || 1;
  const hpMult = Number(enemy && enemy.hpMult || 1) || 1;
  const defenseMult = Number(enemy && enemy.defenseMult || 1) || 1;
  const speed = Number(enemy && enemy.speed || 0) || 0;
  if (behavior === 'charger' || behavior === 'skirmisher' || /\b(charge|charges|dash|dashes|lunge|lunges|leap|rush|retreats|fast|skates|warps)\b/.test(text)) ids.add('rusher');
  if (behavior === 'blocker' || /\b(shield|block|blocks|blocking|frontal|front|parries|parry|back attack|back attacks)\b/.test(text)) ids.add('shielded');
  if (behavior === 'flyer' || /\b(flying|floats|float|wisp|harrier|roc|divebomb|air)\b/.test(text)) ids.add('flying');
  if (behavior === 'thrower' || behavior === 'turret' || /\b(ranged|range|projectile|bolt|bolts|arrow|arrows|caster|fires|throws|lobs|lance|lances|pages)\b/.test(text)) ids.add('casterRanged');
  if (behavior === 'armored' || /\b(armor|armored|shell|plate|plates|braces|construct|sentinel)\b/.test(text)) ids.add('armored');
  if (/\b(fire|frost|cold|ice|storm|lightning|volcanic|ember|cinder|rime|glacier|astral|void|eclipse|arcane|lava|thunder|solar|lunar)\b/.test(text)) ids.add('elemental');
  if (behavior === 'elite' || /\b(elite|mimic|ambush|flee|enrage|enrages|aberration)\b/.test(text)) ids.add('elite');
  if (behavior === 'hopper' || /\b(swarm|surrounds|slime|shardling|tick|ooze|hops|hopper)\b/.test(text)) ids.add('swarm');
  if (behavior === 'healer' || /\b(heal|heals|healer|support|oracle|acolyte|restore|restores)\b/.test(text)) ids.add('supportCaster');
  if (behavior === 'boss') ids.add('boss');
  if (['charger', 'blocker', 'bruiser', 'skirmisher', 'armored', 'boss'].includes(behavior) &&
      (damageMult >= 1.05 || hpMult >= 1.15 || defenseMult >= 1.1 || speed >= 72 || /\b(high threat|danger|slam|counter|parry|enrage)\b/.test(text))) {
    ids.add('highThreatMelee');
  }
  return Array.from(ids).sort();
}

function getEnemyCounterIds(enemy) {
  const text = getEnemyAuditText(enemy);
  return ENEMY_ECOSYSTEM_COUNTERS
    .filter((counter) => counter.keywords.some((keyword) => text.includes(keyword)))
    .map((counter) => counter.id);
}

function getUsedCombatEnemies(data) {
  const usedEnemyIds = new Set();
  getCombatMaps(data).forEach((map) => {
    (map.enemies || []).forEach((enemyId) => usedEnemyIds.add(enemyId));
  });
  const enemyById = getEnemyMap(data);
  return (data.ENEMIES || []).filter((enemy) => enemy && usedEnemyIds.has(enemy.id) && enemyById.has(enemy.id));
}

function countEnemyArchetypes(enemies) {
  return enemies.reduce((counts, enemy) => {
    getEnemyArchetypeIds(enemy).forEach((archetypeId) => {
      counts[archetypeId] = (counts[archetypeId] || 0) + 1;
    });
    return counts;
  }, {});
}

function createEnemyArchetypeCoverageReport(data) {
  const usedEnemies = getUsedCombatEnemies(data);
  const allEnemies = data.ENEMIES || [];
  const usedEnemyIds = new Set(usedEnemies.map((enemy) => enemy.id));
  const counts = countEnemyArchetypes(usedEnemies);
  const allCounts = countEnemyArchetypes(allEnemies);
  const behaviorCounts = usedEnemies.reduce((summary, enemy) => {
    const behavior = String(enemy && enemy.behavior || 'unknown');
    summary[behavior] = (summary[behavior] || 0) + 1;
    return summary;
  }, {});
  const familyIds = new Set(usedEnemies.map((enemy) => String(enemy && enemy.family || 'unknown')));
  const missingArchetypeIds = [];
  const entries = ENEMY_ECOSYSTEM_ARCHETYPES.map((definition) => {
    const usedCount = Number(counts[definition.id] || 0);
    if (usedCount < definition.minUsedCount) missingArchetypeIds.push(definition.id);
    return {
      id: definition.id,
      label: definition.label,
      usedEnemyCount: usedCount,
      totalEnemyCount: Number(allCounts[definition.id] || 0),
      minUsedCount: definition.minUsedCount,
      enemyIds: usedEnemies
        .filter((enemy) => getEnemyArchetypeIds(enemy).includes(definition.id))
        .map((enemy) => enemy.id)
    };
  });
  const maxBehaviorShare = usedEnemies.length
    ? Math.max(...Object.values(behaviorCounts).map((count) => count / usedEnemies.length))
    : 0;
  const issues = [];
  missingArchetypeIds.forEach((archetypeId) => issues.push(`enemyArchetype:${archetypeId}`));
  if (usedEnemies.length < 40) issues.push('usedCombatEnemyCoverage');
  if (Object.keys(behaviorCounts).length < 10) issues.push('enemyBehaviorVariety');
  if (maxBehaviorShare > 0.25) issues.push('enemyBehaviorDominance');
  return {
    totalEnemyCount: allEnemies.length,
    usedEnemyCount: usedEnemies.length,
    unusedEnemyCount: allEnemies.length - usedEnemyIds.size,
    familyCount: familyIds.size,
    behaviorCount: Object.keys(behaviorCounts).length,
    behaviorCounts,
    maxBehaviorShare: roundNumber(maxBehaviorShare, 3),
    requiredArchetypeCount: ENEMY_ECOSYSTEM_ARCHETYPES.length,
    coveredArchetypeCount: entries.filter((entry) => entry.usedEnemyCount >= entry.minUsedCount).length,
    missingArchetypeIds,
    entries,
    issueCount: issues.length,
    issueIds: issues
  };
}

function createEnemyMapDistributionReport(data) {
  const combatMaps = getCombatMaps(data);
  const mapEntries = combatMaps.map((map) => {
    const enemies = getMapEnemies(data, map);
    const enemyIds = Array.from(new Set(enemies.map((enemy) => enemy.id))).sort();
    const behaviorIds = Array.from(new Set(enemies.map((enemy) => String(enemy.behavior || 'unknown')))).sort();
    const archetypeIds = Array.from(new Set(enemies.flatMap(getEnemyArchetypeIds))).sort();
    const bossArena = !!(map.bossRoom || String(map.layoutRole || '') === 'bossArena');
    const warningIds = [];
    if (!bossArena && enemyIds.length < 3) warningIds.push('lowEnemyVariety');
    if (!bossArena && behaviorIds.length < 2) warningIds.push('lowBehaviorVariety');
    if (!bossArena && archetypeIds.length < 2) warningIds.push('lowArchetypeVariety');
    return {
      mapId: map.id,
      name: map.name,
      level: getBenchmarkLevelFromRange(map.levelRange, 50),
      bossArena,
      isDungeon: !!map.isDungeon,
      enemyCount: enemies.length,
      uniqueEnemyCount: enemyIds.length,
      behaviorCount: behaviorIds.length,
      archetypeCount: archetypeIds.length,
      enemyIds,
      behaviorIds,
      archetypeIds,
      warningIds
    };
  });
  const bracketEntries = FIELD_PROGRESSION_BRACKETS.map((bracket) => {
    const maps = mapEntries.filter((map) => map.level >= bracket.minLevel && map.level <= bracket.maxLevel);
    const enemyIds = new Set();
    const behaviorIds = new Set();
    const archetypeIds = new Set();
    maps.forEach((map) => {
      map.enemyIds.forEach((id) => enemyIds.add(id));
      map.behaviorIds.forEach((id) => behaviorIds.add(id));
      map.archetypeIds.forEach((id) => archetypeIds.add(id));
    });
    const warningIds = [];
    if (maps.length && enemyIds.size < 4) warningIds.push('bracketLowEnemyVariety');
    if (maps.length && behaviorIds.size < 3) warningIds.push('bracketLowBehaviorVariety');
    if (maps.length && archetypeIds.size < 4) warningIds.push('bracketLowArchetypeVariety');
    return {
      id: bracket.id,
      label: bracket.label,
      mapCount: maps.length,
      uniqueEnemyCount: enemyIds.size,
      behaviorCount: behaviorIds.size,
      archetypeCount: archetypeIds.size,
      warningIds
    };
  });
  const nonBossMaps = mapEntries.filter((map) => !map.bossArena);
  const mapsWithVariety = nonBossMaps.filter((map) =>
    map.uniqueEnemyCount >= 3 && map.behaviorCount >= 2 && map.archetypeCount >= 2);
  const warnings = mapEntries
    .filter((map) => map.warningIds.length)
    .map((map) => ({ mapId: map.mapId, warningIds: map.warningIds.slice() }));
  bracketEntries.forEach((bracket) => {
    bracket.warningIds.forEach((warningId) => warnings.push({ bracketId: bracket.id, warningIds: [warningId] }));
  });
  return {
    combatMapCount: mapEntries.length,
    fieldDungeonMapCount: nonBossMaps.length,
    bossArenaMapCount: mapEntries.length - nonBossMaps.length,
    mapsWithVarietyCount: mapsWithVariety.length,
    maps: mapEntries,
    bracketEntries,
    warningCount: warnings.reduce((sum, entry) => sum + entry.warningIds.length, 0),
    warnings
  };
}

function createEnemyCounterCoverageReport(data) {
  const usedEnemies = getUsedCombatEnemies(data);
  const entries = ENEMY_ECOSYSTEM_COUNTERS.map((definition) => {
    const enemyIds = usedEnemies
      .filter((enemy) => getEnemyCounterIds(enemy).includes(definition.id))
      .map((enemy) => enemy.id);
    return {
      id: definition.id,
      label: definition.label,
      enemyCount: enemyIds.length,
      enemyIds
    };
  });
  const missingCounterIds = entries.filter((entry) => entry.enemyCount === 0).map((entry) => entry.id);
  const issues = missingCounterIds.map((counterId) => `enemyCounter:${counterId}`);
  return {
    counterCount: ENEMY_ECOSYSTEM_COUNTERS.length,
    coveredCounterCount: entries.filter((entry) => entry.enemyCount > 0).length,
    missingCounterIds,
    entries,
    issueCount: issues.length,
    issueIds: issues
  };
}

function createEnemyEcosystemHealthReport(data) {
  const archetypeCoverage = createEnemyArchetypeCoverageReport(data);
  const mapDistribution = createEnemyMapDistributionReport(data);
  const counterCoverage = createEnemyCounterCoverageReport(data);
  const issues = []
    .concat(archetypeCoverage.issueIds || [])
    .concat(counterCoverage.issueIds || []);
  if (mapDistribution.warningCount > 0) issues.push('enemyMapDistributionWarnings');
  return {
    totalEnemyCount: archetypeCoverage.totalEnemyCount,
    usedEnemyCount: archetypeCoverage.usedEnemyCount,
    unusedEnemyCount: archetypeCoverage.unusedEnemyCount,
    behaviorCount: archetypeCoverage.behaviorCount,
    familyCount: archetypeCoverage.familyCount,
    archetypeCoverage,
    mapDistribution,
    counterCoverage,
    issueCount: issues.length,
    issueIds: issues
  };
}

function getMapScenarioWeights(profile) {
  const raw = {
    clusteredPack: 0.28 + profile.density * 0.24 + profile.swarmRatio * 0.16 + profile.chokepoint * 0.08,
    spreadPack: 0.24 + profile.routeComplexity * 0.2 + profile.rangedRatio * 0.14 + profile.wideOpen * 0.08,
    armoredTarget: 0.08 + profile.armoredRatio * 0.58 + profile.eliteRatio * 0.16,
    flyingPack: 0.04 + profile.flyingRatio * 0.7,
    sustainBoss: 0.06 + profile.threat * 0.2 + profile.supportRatio * 0.18 + profile.eliteRatio * 0.22
  };
  const total = Object.values(raw).reduce((sum, value) => sum + Math.max(0, Number(value) || 0), 0) || 1;
  return Object.keys(raw).reduce((weights, scenarioId) => {
    weights[scenarioId] = roundNumber(Math.max(0, raw[scenarioId]) / total, 3);
    return weights;
  }, {});
}

function getClassMapFitMultiplier(data, classId, profile) {
  const baseId = getClassBaseId(data, classId);
  const layoutStyle = String(profile.layoutStyle || '');
  let fit = 1;
  if (baseId === 'fighter') {
    fit += profile.armoredRatio * 0.12 + profile.threat * 0.08 + profile.chokepoint * 0.06;
    fit -= profile.flyingRatio * 0.2 + profile.verticality * 0.08 + profile.wideOpen * 0.05;
  } else if (baseId === 'archer') {
    fit += profile.wideOpen * 0.1 + profile.flyingRatio * 0.13 + profile.rangedRatio * 0.06;
    fit -= profile.chargerRatio * 0.07 + profile.chokepoint * 0.04;
  } else if (baseId === 'mage') {
    fit += profile.density * 0.1 + profile.verticality * 0.08 + profile.chokepoint * 0.07 + profile.flyingRatio * 0.08;
    fit -= profile.threat * 0.05;
  }

  if (classId === 'guardian') fit += profile.armoredRatio * 0.14 + profile.threat * 0.1 - profile.wideOpen * 0.05;
  if (classId === 'berserker') fit += profile.chargerRatio * 0.06 + profile.armoredRatio * 0.04 - profile.flyingRatio * 0.1;
  if (classId === 'duelist') fit += profile.rangedRatio * 0.04 - profile.density * 0.08 - profile.flyingRatio * 0.12;
  if (classId === 'fireMage') {
    fit += profile.density * 0.11 + profile.flyingRatio * 0.08 - profile.armoredRatio * 0.04;
    if (/lava|frost|glacier/i.test(layoutStyle)) fit += 0.06;
  }
  if (classId === 'runeMage') {
    fit += profile.armoredRatio * 0.18 + profile.supportRatio * 0.1 + profile.verticality * 0.05;
    if (/quarry|glacier|astral|rift/i.test(layoutStyle)) fit += 0.08;
  }
  if (classId === 'stormMage') {
    fit += profile.flyingRatio * 0.13 + profile.verticality * 0.04 + profile.rangedRatio * 0.02;
    fit -= profile.armoredRatio * 0.24 + profile.supportRatio * 0.12 + profile.eliteRatio * 0.1 + profile.wideOpen * 0.08;
    if (profile.flyingRatio < 0.2) fit -= 0.08;
    if (/quarry|switchback/i.test(layoutStyle)) fit -= 0.08;
  }
  if (classId === 'sniper') fit += profile.flyingRatio * 0.1 + profile.armoredRatio * 0.06 + profile.wideOpen * 0.06 - profile.density * 0.07;
  if (classId === 'trapper') {
    fit += (1 - profile.flyingRatio) * 0.11 + profile.armoredRatio * 0.12 + profile.chokepoint * 0.12 + profile.wideOpen * 0.08 + profile.supportRatio * 0.06;
    fit -= profile.flyingRatio * 0.3 + profile.verticality * 0.03;
    if (/switchback|quarry/i.test(layoutStyle)) fit += 0.08;
  }
  if (classId === 'beastArcher') fit += profile.threat * 0.07 + profile.flyingRatio * 0.04 + profile.chargerRatio * 0.04;

  return roundNumber(clamp(fit, 0.58, 1.28), 3);
}

function getTargetLevelMinutes(level) {
  const normalizedLevel = Math.max(1, Number(level) || 1);
  if (normalizedLevel <= 8) return 3 + (normalizedLevel - 1) * (5 / 7);
  if (normalizedLevel <= 40) return 10 + (normalizedLevel - 8) * (15 / 32);
  if (normalizedLevel <= 100) return 25 + (normalizedLevel - 40) * (35 / 60);
  return Math.min(240, 60 + (normalizedLevel - 100) * 2);
}

function addLevelUnlock(unlocksByLevel, level, category, id, label) {
  const normalizedLevel = Math.max(1, Math.floor(Number(level) || 0));
  if (!normalizedLevel) return;
  if (!unlocksByLevel.has(normalizedLevel)) unlocksByLevel.set(normalizedLevel, []);
  unlocksByLevel.get(normalizedLevel).push({
    category,
    id: String(id || ''),
    label: String(label || id || category || '')
  });
}

function addLevelRangeStartUnlock(unlocksByLevel, item, category, id, label) {
  const range = Array.isArray(item && item.levelRange) ? item.levelRange : null;
  if (!range) return;
  addLevelUnlock(unlocksByLevel, range[0], category, id, label);
}

function createLevelUnlockMap(data) {
  const unlocksByLevel = new Map();
  FIELD_PROGRESSION_BRACKETS.forEach((bracket) => {
    if (bracket.minLevel > 1 && Number.isFinite(Number(bracket.minLevel))) {
      addLevelUnlock(unlocksByLevel, bracket.minLevel, 'progressionBracket', bracket.id, bracket.label);
    }
  });
  Object.keys(data.ADVANCED_CLASSES || {}).forEach((classId) => {
    const branch = data.ADVANCED_CLASSES[classId];
    addLevelUnlock(unlocksByLevel, branch && branch.levelRequirement, 'advancedClass', classId, branch && branch.name);
  });
  (data.CLASS_TRIALS || []).forEach((trial) =>
    addLevelUnlock(unlocksByLevel, trial.levelRequirement, 'classTrial', trial.id, trial.title));
  (data.SPECIALIZATIONS || []).forEach((specialization) =>
    addLevelUnlock(unlocksByLevel, specialization.levelRequirement, 'specialization', specialization.id, specialization.name));
  (data.QUESTS || []).forEach((quest) => {
    addLevelUnlock(unlocksByLevel, quest.requiredLevel, 'quest', quest.id, quest.title);
    (quest.objectives || []).forEach((objective) => {
      if (objective && objective.type === 'level') {
        addLevelUnlock(unlocksByLevel, objective.level, 'questLevelObjective', quest.id, quest.title);
      }
    });
  });
  (data.DUNGEONS || []).forEach((dungeon) =>
    addLevelUnlock(unlocksByLevel, dungeon.levelRequirement, 'dungeon', dungeon.id, dungeon.name));
  (data.MAPS || []).forEach((map) =>
    addLevelRangeStartUnlock(unlocksByLevel, map, map && map.isDungeon ? 'dungeonMap' : map && map.bossRoom ? 'bossMap' : 'fieldMap', map && map.id, map && map.name));
  (data.WORLD_AREAS || []).forEach((area) =>
    addLevelRangeStartUnlock(unlocksByLevel, area, 'worldArea', area && area.id, area && area.name));
  (data.ENEMIES || []).forEach((enemy) =>
    addLevelRangeStartUnlock(unlocksByLevel, enemy, enemy && enemy.behavior === 'boss' ? 'bossEnemy' : 'enemy', enemy && enemy.id, enemy && enemy.name));
  (data.SHOP_ITEMS || []).forEach((item) =>
    addLevelUnlock(unlocksByLevel, item.level, 'shopEquipment', item.id, item.name));
  (data.RANDOM_EQUIPMENT_ITEMS || []).forEach((item) =>
    addLevelUnlock(unlocksByLevel, item.level, 'dropEquipment', item.id, item.name));
  (data.BOSS_EQUIPMENT_SOURCES || []).forEach((source) =>
    addLevelUnlock(unlocksByLevel, source.level, 'bossEquipment', source.bossId, source.name));
  return unlocksByLevel;
}

function createLevelCurveReport(data, options = {}) {
  const unlocksByLevel = createLevelUnlockMap(data);
  const sourceLevels = [];
  const addSourceLevel = (level) => {
    const value = Math.max(0, Math.floor(Number(level) || 0));
    if (value) sourceLevels.push(value);
  };
  (data.MAPS || []).forEach((map) => {
    if (Array.isArray(map && map.levelRange)) {
      addSourceLevel(map.levelRange[0]);
      addSourceLevel(map.levelRange[1]);
    }
  });
  (data.ENEMIES || []).forEach((enemy) => {
    if (Array.isArray(enemy && enemy.levelRange)) {
      addSourceLevel(enemy.levelRange[0]);
      addSourceLevel(enemy.levelRange[1]);
    }
  });
  [
    data.SHOP_ITEMS,
    data.RANDOM_EQUIPMENT_ITEMS,
    data.BOSS_EQUIPMENT_SOURCES,
    data.DUNGEONS,
    data.SPECIALIZATIONS,
    data.CLASS_TRIALS
  ].forEach((items) => {
    (items || []).forEach((item) => addSourceLevel(item.level || item.levelRequirement));
  });
  (data.QUESTS || []).forEach((quest) => {
    addSourceLevel(quest.requiredLevel);
    (quest.objectives || []).forEach((objective) => {
      if (objective && objective.type === 'level') addSourceLevel(objective.level);
    });
  });
  const maxLevel = Math.max(100, Math.max(...sourceLevels), Math.floor(Number(options.maxLevel || 0) || 0));
  const spikeThreshold = 0.25;
  const entries = [];
  for (let level = 2; level <= maxLevel; level += 1) {
    const previousMinutes = getTargetLevelMinutes(level - 1);
    const minutes = getTargetLevelMinutes(level);
    const increaseRatio = previousMinutes > 0 ? minutes / previousMinutes - 1 : 0;
    const unlocks = unlocksByLevel.get(level) || [];
    const spike = increaseRatio > spikeThreshold;
    entries.push({
      level,
      previousMinutes: roundNumber(previousMinutes, 1),
      minutes: roundNumber(minutes, 1),
      increasePercent: roundNumber(increaseRatio * 100, 1),
      unlocks,
      spike,
      justified: !spike || unlocks.length > 0
    });
  }
  const unjustifiedSpikes = entries.filter((entry) => entry.spike && !entry.justified);
  const spikes = entries.filter((entry) => entry.spike);
  return {
    maxLevel,
    spikeThresholdPercent: Math.round(spikeThreshold * 100),
    checkedLevels: entries.length,
    unlockLevelCount: unlocksByLevel.size,
    maxIncreasePercent: roundNumber(Math.max(...entries.map((entry) => entry.increasePercent), 0), 1),
    maxUnjustifiedIncreasePercent: roundNumber(Math.max(...entries.filter((entry) => !entry.justified).map((entry) => entry.increasePercent), 0), 1),
    spikeCount: spikes.length,
    unjustifiedSpikeCount: unjustifiedSpikes.length,
    spikes,
    unjustifiedSpikes,
    entries
  };
}

function getProgressionBracket(level) {
  if (Math.max(1, Number(level) || 1) >= 100) {
    return FIELD_PROGRESSION_BRACKETS.find((bracket) => bracket.id === 'endgame') || FIELD_PROGRESSION_BRACKETS[FIELD_PROGRESSION_BRACKETS.length - 1];
  }
  return FIELD_PROGRESSION_BRACKETS.find((bracket) =>
    level >= bracket.minLevel && level <= bracket.maxLevel) || FIELD_PROGRESSION_BRACKETS[FIELD_PROGRESSION_BRACKETS.length - 1];
}

function getTtkTargetRange(level) {
  const bracket = getProgressionBracket(level);
  return FIELD_TTK_TARGETS[bracket.id] || FIELD_TTK_TARGETS.mid;
}

function getActiveCombatShare(level) {
  const bracket = getProgressionBracket(level);
  return FIELD_ACTIVE_COMBAT_SHARE[bracket.id] || FIELD_ACTIVE_COMBAT_SHARE.mid;
}

function getDropEconomyChance(data, path, fallback = 0) {
  const parts = String(path || '').split('.').filter(Boolean);
  let current = data && data.DROP_ECONOMY || {};
  for (let index = 0; index < parts.length; index += 1) {
    if (!current || typeof current !== 'object') return Number(fallback || 0);
    current = current[parts[index]];
  }
  const value = Number(current == null ? fallback : current);
  return Number.isFinite(value) ? clamp(value, 0, 1) : clamp(fallback, 0, 1);
}

function combineIndependentChances(chances) {
  return clamp(1 - (chances || []).reduce((missChance, chance) =>
    missChance * (1 - clamp(chance, 0, 1)), 1), 0, 1);
}

function minutesPerExpectedHit(killsPerHour, chancePerKill) {
  const kph = Math.max(0, Number(killsPerHour || 0));
  const chance = clamp(chancePerKill, 0, 1);
  if (!kph || !chance) return 0;
  return 60 / (kph * chance);
}

function getFieldRewardCadence(data, killsPerHour) {
  const baseChance = getDropEconomyChance(data, 'normalDropChance', 0.08);
  const tableChance = (tableId) => baseChance * getDropEconomyChance(data, `dropTableChances.${tableId}`, 0);
  const globalRareChance = getDropEconomyChance(data, 'globalRareChance.normal', 0.0025);
  const smallVisibleChance = combineIndependentChances([
    tableChance('coins'),
    tableChance('potions'),
    tableChance('equipment'),
    tableChance('cards'),
    tableChance('bonusMaterials'),
    tableChance('plinkoBalls'),
    globalRareChance
  ]);
  const mediumProgressChance = combineIndependentChances([
    tableChance('equipment'),
    tableChance('cards'),
    tableChance('bonusMaterials'),
    globalRareChance
  ]);
  return {
    baseChance: roundNumber(baseChance, 3),
    smallVisibleChance: roundNumber(smallVisibleChance, 4),
    mediumProgressChance: roundNumber(mediumProgressChance, 4),
    smallVisibleMinutes: roundNumber(minutesPerExpectedHit(killsPerHour, smallVisibleChance), 1),
    mediumProgressMinutes: roundNumber(minutesPerExpectedHit(killsPerHour, mediumProgressChance), 1),
    deterministicProgressSeconds: roundNumber(3600 / Math.max(1, Number(killsPerHour || 0)), 1)
  };
}

function normalizeDropWeight(value, fallback = 1) {
  return Math.max(1, Math.round(Number(value == null ? fallback : value) || fallback || 1));
}

function normalizeRarity(rarity) {
  const value = String(rarity || 'Common');
  return RARITY_TIERS.includes(value) ? value : 'Common';
}

function getRarityRank(rarity) {
  return RARITY_TIERS.indexOf(normalizeRarity(rarity));
}

function createRewardLookups(data) {
  return {
    materials: createLookup(data.MATERIAL_ITEMS || []),
    consumables: createLookup((data.CONSUMABLE_ITEMS || []).concat(data.PLINKO_BALLS || [])),
    equipment: createLookup((data.RANDOM_EQUIPMENT_ITEMS || []).concat(data.SHOP_ITEMS || []).concat(data.BOSS_EQUIPMENT_ITEMS || [])),
    cards: createLookup(data.CARD_DEFINITIONS || [])
  };
}

function isBossEnemy(enemy) {
  return !!(enemy && enemy.behavior === 'boss');
}

function isEliteEnemy(enemy) {
  const id = String(enemy && enemy.id || '');
  return !!(enemy && enemy.behavior === 'elite') || id === 'crackedMimic' || id === 'riftAberration';
}

function isSpecialEliteEnemy(enemy) {
  const id = String(enemy && enemy.id || '');
  return id === 'crackedMimic' || id === 'riftAberration';
}

function getEnemyBaseDropChance(data, enemy) {
  if (isBossEnemy(enemy)) return getDropEconomyChance(data, 'bossLootChance', 0.7);
  if (isEliteEnemy(enemy)) return getDropEconomyChance(data, 'eliteDropChance', 0.45);
  return getDropEconomyChance(data, 'normalDropChance', 0.08);
}

function getEnemyGlobalRareChance(data, enemy) {
  const pool = enemy && enemy.dropPool || {};
  if (pool.globalRareEligible === false) return 0;
  if (isBossEnemy(enemy)) return getDropEconomyChance(data, 'globalRareChance.boss', 0.015);
  if (isSpecialEliteEnemy(enemy)) return getDropEconomyChance(data, 'globalRareChance.specialElite', 0.02);
  if (isEliteEnemy(enemy)) return getDropEconomyChance(data, 'globalRareChance.elite', 0.0075);
  return getDropEconomyChance(data, 'globalRareChance.normal', 0.0025);
}

function getPlinkoBallBaseChance(data, enemy) {
  if (!(data.PLINKO_BALLS || []).length) return 0;
  if (isBossEnemy(enemy)) return 0.55;
  if (isSpecialEliteEnemy(enemy)) return 0.18;
  if (isEliteEnemy(enemy)) return 0.1;
  return getDropEconomyChance(data, 'dropTableChances.plinkoBalls', 0.04);
}

function getEnemyDropTableChance(data, enemy, tableId) {
  if (tableId === 'primaryEtc') return 1;
  if (tableId === 'rareValuables') return getEnemyGlobalRareChance(data, enemy);
  const baseChance = getEnemyBaseDropChance(data, enemy);
  if (tableId === 'plinkoBalls') return baseChance * getPlinkoBallBaseChance(data, enemy);
  return baseChance * getDropEconomyChance(data, `dropTableChances.${tableId}`, 0);
}

function getPrimaryEtcEntry(enemy) {
  const materials = Array.isArray(enemy && enemy.dropPool && enemy.dropPool.materials)
    ? enemy.dropPool.materials
    : [];
  return materials[0] && materials[0].materialId
    ? Object.assign({ type: 'material', weight: 1 }, materials[0])
    : null;
}

function getBonusMaterialEntries(enemy) {
  const materials = Array.isArray(enemy && enemy.dropPool && enemy.dropPool.materials)
    ? enemy.dropPool.materials
    : [];
  return materials.slice(1)
    .filter((entry) => entry && entry.materialId)
    .map((entry) => Object.assign({ type: 'material' }, entry, { weight: normalizeDropWeight(entry.weight, 1) }));
}

function getEquipmentEntries(enemy) {
  const equipment = Array.isArray(enemy && enemy.dropPool && enemy.dropPool.equipment)
    ? enemy.dropPool.equipment
    : [];
  return equipment
    .filter((entry) => entry && entry.itemId)
    .map((entry) => Object.assign({ type: 'equipment' }, entry, { weight: normalizeDropWeight(entry.weight, 1) }));
}

function getCardEntries(enemy) {
  const cards = Array.isArray(enemy && enemy.dropPool && enemy.dropPool.cards)
    ? enemy.dropPool.cards
    : [];
  return cards
    .filter((entry) => entry && entry.cardId)
    .map((entry) => Object.assign({ type: 'card' }, entry, { weight: normalizeDropWeight(entry.weight, 1) }));
}

function getPotionTierIndex(level, enemy) {
  const range = Array.isArray(enemy && enemy.levelRange) ? enemy.levelRange : [];
  const rangeMax = range.reduce((max, value) => Math.max(max, Number(value) || 0), 0);
  let score = Math.max(1, Number(level || 0) || 0, rangeMax);
  if (isBossEnemy(enemy)) score += 25;
  if (isEliteEnemy(enemy)) score += 15;
  if (score >= 87) return 3;
  if (score >= 59) return 2;
  if (score >= 31) return 1;
  return 0;
}

function getPotionEntries(lookups, level, enemy) {
  const pool = enemy && enemy.dropPool || {};
  if (pool.basicConsumables === false) return [];
  const tiers = [
    { rarity: 'Common', hp: 'minor_health_potion', mp: 'minor_resource_tonic', hybrid: 'camp_ration' },
    { rarity: 'Uncommon', hp: 'standard_health_potion', mp: 'standard_resource_tonic', hybrid: 'field_ration' },
    { rarity: 'Rare', hp: 'greater_health_potion', mp: 'greater_resource_tonic', hybrid: 'expedition_ration' },
    { rarity: 'Epic', hp: 'superior_health_potion', mp: 'superior_resource_tonic', hybrid: 'hero_ration' }
  ];
  const tierIndex = getPotionTierIndex(level, enemy);
  const tier = tiers[tierIndex] || tiers[0];
  const entries = [
    { type: 'consumable', consumableId: tier.hp, weight: 12, rarity: tier.rarity },
    { type: 'consumable', consumableId: tier.mp, weight: 12, rarity: tier.rarity },
    { type: 'consumable', consumableId: tier.hybrid, weight: 6, rarity: tier.rarity }
  ];
  if (tierIndex > 0) {
    const lowerTier = tiers[tierIndex - 1];
    entries.push(
      { type: 'consumable', consumableId: lowerTier.hp, weight: 3, rarity: lowerTier.rarity },
      { type: 'consumable', consumableId: lowerTier.mp, weight: 3, rarity: lowerTier.rarity },
      { type: 'consumable', consumableId: lowerTier.hybrid, weight: 2, rarity: lowerTier.rarity }
    );
  }
  entries.push(
    { type: 'consumable', consumableId: 'town_return_scroll', weight: 3 },
    { type: 'consumable', consumableId: 'guard_tonic', weight: 4, rarity: 'Uncommon' },
    { type: 'consumable', consumableId: 'swiftstep_oil', weight: 4, rarity: 'Uncommon' },
    { type: 'consumable', consumableId: 'magnet_charm', weight: 3, rarity: 'Uncommon' }
  );
  return entries.filter((entry) => lookups.consumables.has(entry.consumableId));
}

function getPlinkoEntries(data, lookups, enemy) {
  const ballIds = new Set((data.PLINKO_BALLS || []).map((ball) => ball && ball.id).filter(Boolean));
  if (!ballIds.size) return [];
  const specialElite = isSpecialEliteEnemy(enemy);
  const weights = isBossEnemy(enemy)
    ? { [PLINKO_BASIC_BALL_ID]: 3, [PLINKO_POLISHED_BALL_ID]: 12, [PLINKO_METEOR_BALL_ID]: 3 }
    : specialElite
      ? { [PLINKO_BASIC_BALL_ID]: 3, [PLINKO_POLISHED_BALL_ID]: 7, [PLINKO_METEOR_BALL_ID]: 1 }
      : isEliteEnemy(enemy)
        ? { [PLINKO_BASIC_BALL_ID]: 8, [PLINKO_POLISHED_BALL_ID]: 2 }
        : { [PLINKO_BASIC_BALL_ID]: 1 };
  return Object.keys(weights)
    .filter((consumableId) => ballIds.has(consumableId))
    .map((consumableId) => ({
      type: 'consumable',
      consumableId,
      weight: weights[consumableId],
      rarity: lookups.consumables.get(consumableId) && lookups.consumables.get(consumableId).rarity || 'Uncommon'
    }));
}

function getEntryDefinition(lookups, entry) {
  if (!entry) return null;
  if (entry.type === 'material') return lookups.materials.get(entry.materialId) || null;
  if (entry.type === 'equipment') return lookups.equipment.get(entry.itemId) || null;
  if (entry.type === 'card') return lookups.cards.get(entry.cardId) || null;
  if (entry.type === 'consumable') return lookups.consumables.get(entry.consumableId) || null;
  return null;
}

function getEntryRarity(lookups, entry) {
  const definition = getEntryDefinition(lookups, entry);
  return normalizeRarity(entry && entry.rarity || definition && definition.rarity || 'Common');
}

function getRewardFamily(entry, tableId) {
  const id = String(entry && (entry.materialId || entry.consumableId || entry.cardId || entry.itemId) || '');
  if (tableId === 'coins' || entry && entry.type === 'currency') return 'currency';
  if (entry && entry.type === 'equipment') return 'equipment';
  if (entry && entry.type === 'card') return 'card';
  if (entry && entry.type === 'material' && /StarCard$/.test(id)) return 'starCardMaterial';
  if (entry && entry.type === 'material' && ['upgradeDust', 'upgradeCatalyst', 'wardingScroll', 'refinementCore', 'cubeFragment'].includes(id)) return 'upgradeMaterial';
  if (tableId === 'primaryEtc') return 'deterministicMaterial';
  if (tableId === 'plinkoBalls') return 'plinko';
  if (tableId === 'rareValuables') return 'prestigeUtility';
  if (entry && entry.type === 'consumable' && /coupon/i.test(id)) return 'rateCoupon';
  if (entry && entry.type === 'consumable') return 'consumable';
  return 'material';
}

function isMandatoryPowerReward(entry, tableId) {
  const family = getRewardFamily(entry, tableId);
  return [
    'deterministicMaterial',
    'equipment',
    'card',
    'starCardMaterial',
    'upgradeMaterial'
  ].includes(family);
}

function isCosmeticPrestigeReward(entry, tableId) {
  const family = getRewardFamily(entry, tableId);
  return ['plinko', 'prestigeUtility', 'rateCoupon'].includes(family);
}

function addRewardValue(target, key, value) {
  const normalizedKey = String(key || 'unknown');
  target[normalizedKey] = roundNumber(Number(target[normalizedKey] || 0) + Number(value || 0), 3);
}

function addRewardEntry(aggregate, tableId, entry, dropsPerHour, lookups, options = {}) {
  const value = Number(dropsPerHour || 0);
  if (value <= 0) return;
  const family = getRewardFamily(entry, tableId);
  const rarity = getEntryRarity(lookups, entry);
  addRewardValue(aggregate.tables, tableId, value);
  addRewardValue(aggregate.families, family, value);
  addRewardValue(aggregate.rarities, rarity, value);
  if (options.deterministic) aggregate.deterministicProgressPerHour += value;
  else aggregate.optionalDropsPerHour += value;
  if (isMandatoryPowerReward(entry, tableId)) aggregate.mandatoryPowerDropsPerHour += value;
  if (isCosmeticPrestigeReward(entry, tableId)) aggregate.cosmeticPrestigeDropsPerHour += value;
  if (getRarityRank(rarity) >= getRarityRank('Rare')) {
    aggregate.rareOrBetterDropsPerHour += value;
    if (!options.deterministic) aggregate.optionalRareOrBetterDropsPerHour += value;
  }
}

function addWeightedRewardEntries(aggregate, tableId, entries, tableHitsPerHour, lookups, options = {}) {
  const filtered = (entries || []).filter(Boolean);
  const totalWeight = filtered.reduce((sum, entry) => sum + normalizeDropWeight(entry.weight, 1), 0);
  if (!filtered.length || totalWeight <= 0 || tableHitsPerHour <= 0) return;
  filtered.forEach((entry) => {
    addRewardEntry(aggregate, tableId, entry, tableHitsPerHour * normalizeDropWeight(entry.weight, 1) / totalWeight, lookups, options);
  });
}

function createRewardAggregate() {
  return {
    tables: {},
    families: {},
    rarities: {},
    deterministicProgressPerHour: 0,
    optionalDropsPerHour: 0,
    mandatoryPowerDropsPerHour: 0,
    cosmeticPrestigeDropsPerHour: 0,
    rareOrBetterDropsPerHour: 0,
    optionalRareOrBetterDropsPerHour: 0
  };
}

function finalizeRewardAggregate(aggregate) {
  const optionalTables = Object.keys(aggregate.tables)
    .filter((tableId) => tableId !== 'primaryEtc')
    .map((tableId) => aggregate.tables[tableId]);
  const maxOptionalTableShare = aggregate.optionalDropsPerHour > 0 && optionalTables.length
    ? Math.max(...optionalTables) / aggregate.optionalDropsPerHour
    : 0;
  return Object.assign({}, aggregate, {
    deterministicProgressPerHour: roundNumber(aggregate.deterministicProgressPerHour, 1),
    optionalDropsPerHour: roundNumber(aggregate.optionalDropsPerHour, 1),
    mandatoryPowerDropsPerHour: roundNumber(aggregate.mandatoryPowerDropsPerHour, 1),
    cosmeticPrestigeDropsPerHour: roundNumber(aggregate.cosmeticPrestigeDropsPerHour, 1),
    rareOrBetterDropsPerHour: roundNumber(aggregate.rareOrBetterDropsPerHour, 2),
    optionalRareOrBetterDropsPerHour: roundNumber(aggregate.optionalRareOrBetterDropsPerHour, 2),
    rareOrBetterMinutes: roundNumber(minutesPerExpectedHit(aggregate.rareOrBetterDropsPerHour, 1), 1),
    optionalRareOrBetterMinutes: roundNumber(minutesPerExpectedHit(aggregate.optionalRareOrBetterDropsPerHour, 1), 1),
    maxOptionalTableShare: roundNumber(maxOptionalTableShare, 3),
    tables: Object.keys(aggregate.tables).sort().reduce((result, key) => {
      result[key] = roundNumber(aggregate.tables[key], 2);
      return result;
    }, {}),
    families: Object.keys(aggregate.families).sort().reduce((result, key) => {
      result[key] = roundNumber(aggregate.families[key], 2);
      return result;
    }, {}),
    rarities: Object.keys(aggregate.rarities).sort().reduce((result, key) => {
      result[key] = roundNumber(aggregate.rarities[key], 2);
      return result;
    }, {})
  });
}

function createMapRewardSourceReport(data, map, enemies, killsPerHour, lookups) {
  const aggregate = createRewardAggregate();
  const killsByEnemy = Math.max(1, Number(killsPerHour || 0)) / Math.max(1, enemies.length);
  enemies.forEach((enemy) => {
    const primaryEntry = getPrimaryEtcEntry(enemy);
    if (primaryEntry) {
      addWeightedRewardEntries(aggregate, 'primaryEtc', [primaryEntry], killsByEnemy, lookups, { deterministic: true });
    }
    const pool = enemy && enemy.dropPool || {};
    if (Number(pool.currencyWeight || 0) > 0) {
      addRewardEntry(aggregate, 'coins', { type: 'currency', rarity: 'Common' }, killsByEnemy * getEnemyDropTableChance(data, enemy, 'coins'), lookups);
    }
    addWeightedRewardEntries(aggregate, 'potions', getPotionEntries(lookups, getBenchmarkLevelFromRange(map.levelRange, 50), enemy), killsByEnemy * getEnemyDropTableChance(data, enemy, 'potions'), lookups);
    addWeightedRewardEntries(aggregate, 'equipment', getEquipmentEntries(enemy), killsByEnemy * getEnemyDropTableChance(data, enemy, 'equipment'), lookups);
    addWeightedRewardEntries(aggregate, 'cards', getCardEntries(enemy), killsByEnemy * getEnemyDropTableChance(data, enemy, 'cards'), lookups);
    addWeightedRewardEntries(aggregate, 'bonusMaterials', getBonusMaterialEntries(enemy), killsByEnemy * getEnemyDropTableChance(data, enemy, 'bonusMaterials'), lookups);
    addWeightedRewardEntries(aggregate, 'plinkoBalls', getPlinkoEntries(data, lookups, enemy), killsByEnemy * getEnemyDropTableChance(data, enemy, 'plinkoBalls'), lookups);
    addWeightedRewardEntries(aggregate, 'rareValuables', GLOBAL_RARE_REWARD_ENTRIES, killsByEnemy * getEnemyDropTableChance(data, enemy, 'rareValuables'), lookups);
  });
  const report = finalizeRewardAggregate(aggregate);
  const issues = [];
  if (report.deterministicProgressPerHour <= 0) issues.push('missingDeterministicPrimaryProgress');
  if (report.optionalDropsPerHour < 5) issues.push('tooFewOptionalDrops');
  if (report.optionalDropsPerHour > 40) issues.push('optionalLootShower');
  if (report.maxOptionalTableShare > 0.65) issues.push('dominantOptionalSource');
  if (report.optionalRareOrBetterMinutes <= 0 || report.optionalRareOrBetterMinutes > 180) issues.push('rareRewardDryStreakRisk');
  return Object.assign(report, {
    mapId: map.id,
    mapName: map.name,
    killSamplePerHour: Math.round(killsPerHour),
    issueIds: issues
  });
}

function medianObjectValues(items, key) {
  const keys = Array.from(new Set((items || []).flatMap((item) => Object.keys(item && item[key] || {})))).sort();
  return keys.reduce((result, id) => {
    result[id] = roundNumber(medianNumber(items.map((item) => item && item[key] && item[key][id] || 0)), 2);
    return result;
  }, {});
}

function createFieldRewardSourceSummary(maps) {
  const rewardReports = (maps || []).map((map) => map.rewardSource).filter(Boolean);
  const allIssues = rewardReports.flatMap((reward) =>
    (reward.issueIds || []).map((issueId) => ({ mapId: reward.mapId, issueId })));
  return {
    mapCount: rewardReports.length,
    issueCount: allIssues.length,
    issues: allIssues,
    medianOptionalDropsPerHour: roundNumber(medianNumber(rewardReports.map((reward) => reward.optionalDropsPerHour)), 1),
    medianDeterministicProgressPerHour: roundNumber(medianNumber(rewardReports.map((reward) => reward.deterministicProgressPerHour)), 1),
    medianMandatoryPowerDropsPerHour: roundNumber(medianNumber(rewardReports.map((reward) => reward.mandatoryPowerDropsPerHour)), 1),
    medianCosmeticPrestigeDropsPerHour: roundNumber(medianNumber(rewardReports.map((reward) => reward.cosmeticPrestigeDropsPerHour)), 1),
    medianRareOrBetterMinutes: roundNumber(medianNumber(rewardReports.map((reward) => reward.rareOrBetterMinutes)), 1),
    medianOptionalRareOrBetterMinutes: roundNumber(medianNumber(rewardReports.map((reward) => reward.optionalRareOrBetterMinutes)), 1),
    maxOptionalTableShare: roundNumber(Math.max(...rewardReports.map((reward) => reward.maxOptionalTableShare), 0), 3),
    tableMedians: medianObjectValues(rewardReports, 'tables'),
    familyMedians: medianObjectValues(rewardReports, 'families'),
    rarityMedians: medianObjectValues(rewardReports, 'rarities')
  };
}

function getConsumableReplacementCost(data, itemId) {
  const id = String(itemId || '');
  if (!id) return 0;
  let bestCost = Number.POSITIVE_INFINITY;
  (data.SHOP_VENDOR_CATALOGS || []).forEach((catalog) => {
    (catalog.entries || catalog.items || []).forEach((entry) => {
      if (!entry || entry.kind !== 'consumable' || String(entry.consumableId || '') !== id) return;
      const quantity = Math.max(1, Math.floor(Number(entry.quantity || 1) || 1));
      const unitCost = Math.max(0, Number(entry.cost || 0)) / quantity;
      if (unitCost > 0) bestCost = Math.min(bestCost, unitCost);
    });
  });
  return Number.isFinite(bestCost) ? Math.round(bestCost) : 0;
}

function getHealthPotionIdForLevel(level) {
  const normalizedLevel = Math.max(1, Number(level || 1) || 1);
  if (normalizedLevel >= 87) return 'superior_health_potion';
  if (normalizedLevel >= 59) return 'greater_health_potion';
  if (normalizedLevel >= 31) return 'standard_health_potion';
  return 'minor_health_potion';
}

function getPotionRecovery(data, itemId) {
  const item = (data.CONSUMABLE_ITEMS || []).find((candidate) => candidate && candidate.id === itemId) || {};
  return Math.max(1, Number(item.hpFlat || 0) || 1);
}

function getClassSurvivalProfile(classId) {
  return CLASS_SURVIVAL_PROFILES[classId] || CLASS_SURVIVAL_PROFILES.fighter;
}

function getClassStatsForLevel(data, createProjectStarfallEngine, classId, level, rank) {
  if (typeof createProjectStarfallEngine !== 'function') {
    return { maxHp: 1, defense: 0, avoid: 0, block: 0, damageReductionPercent: 0 };
  }
  const engine = createProjectStarfallEngine(null, data);
  const baseId = getClassBaseId(data, classId);
  if (!engine.chooseClass(baseId)) return { maxHp: 1, defense: 0, avoid: 0, block: 0, damageReductionPercent: 0 };
  return prepareBalancePlayer(data, engine, classId, level, rank);
}

function getMapIncomingHitPressure(profile) {
  return clamp(
    0.16 +
    Number(profile.threat || 0) * 0.48 +
    Number(profile.chargerRatio || 0) * 0.2 +
    Number(profile.rangedRatio || 0) * 0.12 +
    Number(profile.flyingRatio || 0) * 0.08 +
    Number(profile.density || 0) * 0.08 +
    Number(profile.eliteRatio || 0) * 0.22,
    0.08,
    0.82
  );
}

function getClassAvoidMitigation(stats) {
  const avoidChance = clamp(Number(stats.avoid || 0) / 180, 0, 0.32);
  const blockReduction = clamp(Number(stats.block || 0) / 180, 0, 0.24);
  const damageReduction = clamp(Number(stats.damageReductionPercent || 0) / 100, 0, 0.35);
  return (1 - avoidChance) * (1 - blockReduction) * (1 - damageReduction);
}

function estimateClassMapSurvivability(data, createProjectStarfallEngine, classId, map, profile, enemies, result, rank) {
  const level = Math.max(1, Number(map.level || 1) || 1);
  const stats = getClassStatsForLevel(data, createProjectStarfallEngine, classId, level, rank);
  const survival = getClassSurvivalProfile(classId);
  const avgRawHit = medianNumber((enemies || []).map((enemy) => getEstimatedMonsterDamage(level, enemy)));
  const mitigatedHit = mitigateEstimatedPlayerDamage(avgRawHit, stats.defense);
  const exposure = getMapIncomingHitPressure(profile) * Number(survival.exposure || 1);
  const effectiveHit = mitigatedHit * getClassAvoidMitigation(stats);
  const damageTakenPerHour = Math.max(0, Number(result.killsPerHour || 0)) * exposure * effectiveHit;
  const potionId = getHealthPotionIdForLevel(level);
  const potionRecovery = getPotionRecovery(data, potionId);
  const potionCost = getConsumableReplacementCost(data, potionId);
  const potionNeedMultiplier = clamp(Number(survival.sustain || 1), 0.35, 1.25);
  const expectedPotionUsesPerHour = damageTakenPerHour * potionNeedMultiplier * PURCHASED_POTION_HEALING_SHARE / Math.max(1, potionRecovery * 0.88);
  const potionCostPerHour = expectedPotionUsesPerHour * potionCost;
  const earnings = Math.max(1, Number(result.currencyPerHour || 0));
  const spikePressure = exposure * effectiveHit * (1 + Number(profile.threat || 0) * 1.4 + Number(profile.eliteRatio || 0) * 2);
  const effectiveHp = Math.max(1, Number(stats.maxHp || 1)) * (1 + clamp(Number(stats.block || 0) / 120, 0, 0.3));
  const deathRiskPerHour = Math.max(0, (spikePressure / effectiveHp - 0.16) * 2.4 * Number(survival.spikeControl || 1));
  return {
    classId,
    maxHp: Math.round(Number(stats.maxHp || 0)),
    defense: Math.round(Number(stats.defense || 0)),
    avoid: Math.round(Number(stats.avoid || 0)),
    block: Math.round(Number(stats.block || 0)),
    avgRawHit: roundNumber(avgRawHit, 1),
    mitigatedHit: roundNumber(mitigatedHit, 1),
    exposureIndex: roundNumber(exposure, 3),
    damageTakenPerHour: Math.round(damageTakenPerHour),
    expectedPotionUsesPerHour: roundNumber(expectedPotionUsesPerHour, 1),
    potionId,
    potionCost,
    potionCostPerHour: Math.round(potionCostPerHour),
    potionCostEarningsPercent: roundNumber(potionCostPerHour / earnings * 100, 1),
    deathRiskPerHour: roundNumber(deathRiskPerHour, 3)
  };
}

function createMapSurvivabilityReport(data, createProjectStarfallEngine, map, profile, enemies, results, rank) {
  const entries = (results || []).map((result) =>
    estimateClassMapSurvivability(data, createProjectStarfallEngine, result.classId, map, profile, enemies, result, rank));
  const medianDeathRisk = Math.max(0.001, medianNumber(entries.map((entry) => entry.deathRiskPerHour)));
  const resultByClassId = new Map((results || []).map((result) => [result.classId, result]));
  const issues = [];
  entries.forEach((entry) => {
    if (entry.potionCostEarningsPercent > 25) issues.push({ classId: entry.classId, issueId: 'potionCostHigh' });
    if (entry.deathRiskPerHour > 1) issues.push({ classId: entry.classId, issueId: 'deathRiskHigh' });
    const efficiency = resultByClassId.get(entry.classId) && resultByClassId.get(entry.classId).efficiencyIndex || 100;
    if (entry.deathRiskPerHour > 0.25 && entry.deathRiskPerHour > medianDeathRisk * 1.2 && efficiency < 105) {
      issues.push({ classId: entry.classId, issueId: 'uncompensatedDeathRiskOutlier' });
    }
  });
  return {
    mapId: map.id,
    mapName: map.name,
    level: map.level,
    medianDamageTakenPerHour: Math.round(medianNumber(entries.map((entry) => entry.damageTakenPerHour))),
    maxDamageTakenPerHour: Math.round(Math.max(...entries.map((entry) => entry.damageTakenPerHour), 0)),
    medianPotionCostEarningsPercent: roundNumber(medianNumber(entries.map((entry) => entry.potionCostEarningsPercent)), 1),
    maxPotionCostEarningsPercent: roundNumber(Math.max(...entries.map((entry) => entry.potionCostEarningsPercent), 0), 1),
    medianDeathRiskPerHour: roundNumber(medianNumber(entries.map((entry) => entry.deathRiskPerHour)), 3),
    maxDeathRiskPerHour: roundNumber(Math.max(...entries.map((entry) => entry.deathRiskPerHour), 0), 3),
    issueIds: issues,
    entries
  };
}

function createFieldSurvivabilitySummary(maps) {
  const reports = (maps || []).map((map) => map.survivability).filter(Boolean);
  const issues = reports.flatMap((report) =>
    (report.issueIds || []).map((issue) => Object.assign({ mapId: report.mapId }, issue)));
  const classIds = Array.from(new Set(reports.flatMap((report) =>
    (report.entries || []).map((entry) => entry.classId)))).sort();
  const classSummaries = classIds.map((classId) => {
    const entries = reports.flatMap((report) => report.entries || []).filter((entry) => entry.classId === classId);
    return {
      classId,
      mapCount: entries.length,
      medianDamageTakenPerHour: Math.round(medianNumber(entries.map((entry) => entry.damageTakenPerHour))),
      medianPotionCostEarningsPercent: roundNumber(medianNumber(entries.map((entry) => entry.potionCostEarningsPercent)), 1),
      medianDeathRiskPerHour: roundNumber(medianNumber(entries.map((entry) => entry.deathRiskPerHour)), 3),
      maxDeathRiskPerHour: roundNumber(Math.max(...entries.map((entry) => entry.deathRiskPerHour), 0), 3)
    };
  });
  return {
    mapCount: reports.length,
    issueCount: issues.length,
    issues,
    medianDamageTakenPerHour: Math.round(medianNumber(reports.map((report) => report.medianDamageTakenPerHour))),
    maxDamageTakenPerHour: Math.round(Math.max(...reports.map((report) => report.maxDamageTakenPerHour), 0)),
    medianPotionCostEarningsPercent: roundNumber(medianNumber(reports.map((report) => report.medianPotionCostEarningsPercent)), 1),
    maxPotionCostEarningsPercent: roundNumber(Math.max(...reports.map((report) => report.maxPotionCostEarningsPercent), 0), 1),
    medianDeathRiskPerHour: roundNumber(medianNumber(reports.map((report) => report.medianDeathRiskPerHour)), 3),
    maxDeathRiskPerHour: roundNumber(Math.max(...reports.map((report) => report.maxDeathRiskPerHour), 0), 3),
    classSummaries
  };
}

function getMapWorldWidth(map) {
  const ground = map && Array.isArray(map.platforms) && map.platforms[0] || null;
  return Math.max(1, Number(ground && (ground.w || ground[2]) || 0));
}

function getSpawnSectionOccupancy(map) {
  const sections = Array.isArray(map && map.spawnSections) ? map.spawnSections : [];
  const spawnPoints = Array.isArray(map && map.spawnPoints) ? map.spawnPoints : [];
  const countsBySectionId = spawnPoints.reduce((counts, point) => {
    const sectionId = String(point && point.sectionId || '');
    if (!sectionId) return counts;
    counts[sectionId] = (counts[sectionId] || 0) + 1;
    return counts;
  }, {});
  const sectionEntries = sections.map((section) => ({
    id: String(section.id || ''),
    label: String(section.label || ''),
    tier: String(section.tier || ''),
    spawnPointCount: Number(countsBySectionId[section.id] || 0)
  }));
  return {
    sectionCount: sections.length,
    spawnPointCount: spawnPoints.length,
    activeSectionCount: sectionEntries.filter((section) => section.spawnPointCount > 0).length,
    emptySectionCount: sectionEntries.filter((section) => section.spawnPointCount <= 0).length,
    sections: sectionEntries
  };
}

function getRuntimeTuningContext(runtimeEngine, map) {
  if (!runtimeEngine || typeof runtimeEngine.changeMap !== 'function' || !map) return {};
  try {
    runtimeEngine.changeMap(map.id, { silent: true });
    const snapshot = typeof runtimeEngine.snapshot === 'function' ? runtimeEngine.snapshot() : {};
    const runtime = snapshot && snapshot.runtime || {};
    const route = runtime.trainingRoute || {};
    return {
      arenaSkeleton: String(runtime.arenaSkeleton || ''),
      arenaMechanic: String(runtime.arenaMechanic || ''),
      platformCoverage: Number(route.platformCoverage || 0),
      spawnDensityPer1000px: Number(route.spawnDensityPer1000px || 0),
      routePlatformCount: Array.isArray(route.routePlatformIds) ? route.routePlatformIds.length : 0,
      ladderDependence: Number(route.traversalMix && route.traversalMix.ladderDependence || 0),
      rampDependence: Number(route.traversalMix && route.traversalMix.rampDependence || 0),
      routeViable: !!route.viable,
      routeIssues: Array.isArray(route.issues) ? route.issues.slice() : []
    };
  } catch (err) {
    return {
      routeViable: false,
      routeIssues: ['runtimeTuningSnapshotFailed']
    };
  }
}

function getTargetPartySizeForIntent(intent) {
  const useCase = String(intent && intent.intendedUseCase || '').toLowerCase();
  if (useCase.includes('full party') || useCase.includes('party-only')) return 4;
  if (useCase.includes('small party')) return 3;
  if (useCase.includes('duo')) return 2;
  if (useCase.includes('boss')) return 3;
  if (useCase.includes('high-density') || useCase.includes('farming')) return 2;
  return 1;
}

function getFarmingAbuseRiskScore(intent) {
  const risk = String(intent && intent.farmingAbuseRisk || '').toLowerCase();
  if (risk.includes('very high')) return 90;
  if (risk.includes('high')) return 72;
  if (risk.includes('medium-high')) return 58;
  if (risk.includes('medium')) return 42;
  if (risk.includes('low')) return 18;
  return 35;
}

function hasMapAbuseControlSection(sectionStats) {
  return (sectionStats.sections || []).some((section) => {
    const text = `${section.label} ${section.tier}`.toLowerCase();
    return /event|objective|regroup|core|elite|switch|rod|safe|gate|surge|pocket/.test(text);
  });
}

function createMapTuningEntry(data, map, fieldReport, runtimeEngine, rewardLookups) {
  const enemies = getMapEnemies(data, map);
  const level = fieldReport && fieldReport.level || getBenchmarkLevelFromRange(map && map.levelRange, 50);
  const avgXpPerKill = fieldReport && fieldReport.avgXpPerKill || Math.max(1, medianNumber(enemies.map((enemy) =>
    getEstimatedMonsterXp(level, enemy))));
  const levelXp = getEstimatedLevelXp(level);
  const targetMinutes = getTargetLevelMinutes(level);
  const medianKillsPerHour = fieldReport && fieldReport.medianKillsPerHour ||
    Math.max(1, levelXp / Math.max(1, avgXpPerKill) / (targetMinutes / 60));
  const killsPerMinute = medianKillsPerHour / 60;
  const rewardSource = fieldReport && fieldReport.rewardSource ||
    createMapRewardSourceReport(data, map, enemies, medianKillsPerHour, rewardLookups);
  const survivability = fieldReport && fieldReport.survivability || null;
  const sectionStats = getSpawnSectionOccupancy(map);
  const runtime = getRuntimeTuningContext(runtimeEngine, map);
  const profile = fieldReport && fieldReport.profile || getMapProfile(data, map);
  const intent = map && map.designIntent || null;
  const worldWidth = getMapWorldWidth(map);
  const waveMax = Math.max(1, Number(map && map.waveMax || 1));
  const waveDelay = Math.max(1, Number(map && map.waveDelay || 1));
  const routeCycleSeconds = waveMax / Math.max(0.01, killsPerMinute);
  const spawnDensityPer1000px = runtime.spawnDensityPer1000px ||
    sectionStats.spawnPointCount / Math.max(1, worldWidth / 1000);
  const waveToSpawnPointRatio = waveMax / Math.max(1, sectionStats.spawnPointCount);
  const spawnPointsPerActiveSection = sectionStats.spawnPointCount / Math.max(1, sectionStats.activeSectionCount);
  const spawnVacancyPercent = clamp(
    Math.max(0, 1 - waveToSpawnPointRatio) * 48 +
      Math.max(0, 1.2 - spawnDensityPer1000px) * 12 +
      Math.max(0, 2 - spawnPointsPerActiveSection) * 8,
    0,
    60
  );
  const respawnHandshakeRatio = routeCycleSeconds / waveDelay;
  const idleTimePercent = clamp(
    spawnVacancyPercent * 0.45 +
      Math.max(0, 1 - respawnHandshakeRatio) * 25 +
      Math.max(0, 1.15 - spawnDensityPer1000px) * 8,
    0,
    45
  );
  const nonCombatTraversalPercent = clamp((Number(runtime.ladderDependence || 0) + Number(runtime.rampDependence || 0)) * 100, 0, 70);
  const travelSharePercent = clamp(
    6 +
      Number(profile.routeComplexity || 0) * 16 +
      nonCombatTraversalPercent * 0.5 -
      Math.min(8, spawnDensityPer1000px * 2),
    3,
    55
  );
  const efficiencies = (fieldReport && Array.isArray(fieldReport.results) ? fieldReport.results : [])
    .map((result) => Number(result.efficiencyIndex || 0))
    .filter((value) => value > 0);
  const medianEfficiency = Math.max(1, medianNumber(efficiencies));
  const topEfficiency = efficiencies.length ? Math.max(...efficiencies) : medianEfficiency;
  const classPerformanceSpreadPercent = efficiencies.length
    ? (topEfficiency / medianEfficiency - 1) * 100
    : 0;
  const targetPartySize = getTargetPartySizeForIntent(intent);
  const separablePartySections = Math.min(targetPartySize, Math.max(1, sectionStats.activeSectionCount));
  const partyOverlapPercent = targetPartySize <= 1
    ? 0
    : clamp(
        10 +
          Math.max(0, targetPartySize - sectionStats.activeSectionCount) * 22 +
          Number(profile.chokepoint || 0) * 16 -
          Math.max(0, sectionStats.activeSectionCount - targetPartySize) * 3,
        0,
        80
      );
  const partyEfficiencyVsSoloPercent = targetPartySize <= 1
    ? 100
    : clamp(
        100 +
          Math.max(0, separablePartySections - 1) * 38 +
          Math.min(20, Math.max(0, waveToSpawnPointRatio - 1) * 10) +
          (intent && intent.partyScaling && intent.partyScaling !== 'none' ? 10 : 0) -
          partyOverlapPercent * 0.35,
        80,
        260
      );
  const medianPotionUsesPerHour = survivability && Array.isArray(survivability.entries)
    ? medianNumber(survivability.entries.map((entry) => entry.expectedPotionUsesPerHour))
    : 0;
  const medianDamageTakenPerHour = survivability && Number(survivability.medianDamageTakenPerHour || 0) || 0;
  const deathRatePerHour = survivability && Number(survivability.medianDeathRiskPerHour || 0) || 0;
  const normalTtkSeconds = fieldReport && Number(fieldReport.normalTtkSeconds || 0) ||
    3600 / Math.max(1, medianKillsPerHour) * getActiveCombatShare(level);
  const eliteMinibossClearTimeSeconds = roundNumber(normalTtkSeconds * (1 + Number(profile.eliteRatio || 0) * 5 + Number(profile.armoredRatio || 0) * 0.9 + (map && map.isDungeon ? 1.4 : 0)), 1);
  const dropValuePerMinute = (Number(rewardSource.optionalDropsPerHour || 0) + Number(rewardSource.deterministicProgressPerHour || 0)) / 60;
  const useCaseText = String(intent && intent.intendedUseCase || '').toLowerCase();
  const farmAbuseRelevant = getFarmingAbuseRiskScore(intent) >= 70 &&
    !map.bossRoom &&
    (map.endlessScaling || useCaseText.includes('farming') || useCaseText.includes('high-density'));
  const abuseRiskIndex = clamp(
    getFarmingAbuseRiskScore(intent) +
      dropValuePerMinute * 1.8 -
      (hasMapAbuseControlSection(sectionStats) ? 14 : 0) -
      Math.max(0, travelSharePercent - 18) * 0.25,
    0,
    100
  );
  const abandonmentRiskIndex = clamp(
    idleTimePercent * 1.25 +
      Math.max(0, travelSharePercent - 18) * 0.9 +
      deathRatePerHour * 80 +
      classPerformanceSpreadPercent * 0.35 +
      Math.max(0, spawnVacancyPercent - 8) * 0.6,
    0,
    100
  );
  const repeatVisitationIndex = clamp(
    52 +
      Math.min(18, dropValuePerMinute * 1.8) +
      Math.min(16, Number(rewardSource.optionalRareOrBetterDropsPerHour || 0) * 1.8) +
      (intent && intent.priorityRedesign ? 4 : 0) -
      idleTimePercent * 0.4 -
      Math.max(0, travelSharePercent - 22) * 0.35 -
      Math.max(0, classPerformanceSpreadPercent - 25) * 0.25,
    0,
    100
  );
  const warningIds = [];
  if (idleTimePercent > MAP_TUNING_WARNING_THRESHOLDS.idleTimePercent) warningIds.push('idleTimeHigh');
  if (travelSharePercent > MAP_TUNING_WARNING_THRESHOLDS.travelSharePercent) warningIds.push('travelShareHigh');
  if (partyOverlapPercent > MAP_TUNING_WARNING_THRESHOLDS.partyOverlapPercent) warningIds.push('partyOverlapHigh');
  if (classPerformanceSpreadPercent > MAP_TUNING_WARNING_THRESHOLDS.classPerformanceSpreadPercent) warningIds.push('classSpreadHigh');
  if (spawnVacancyPercent > MAP_TUNING_WARNING_THRESHOLDS.spawnVacancyPercent) warningIds.push('spawnVacancyHigh');
  if (targetPartySize > 1 && partyEfficiencyVsSoloPercent < MAP_TUNING_WARNING_THRESHOLDS.partyEfficiencyLowPercent) warningIds.push('partyEfficiencyLow');
  if (targetPartySize > 1 &&
    partyEfficiencyVsSoloPercent > MAP_TUNING_WARNING_THRESHOLDS.partyEfficiencyFreeFarmPercent &&
    deathRatePerHour < 0.05) warningIds.push('partyFreeFarmRisk');
  if (farmAbuseRelevant && !hasMapAbuseControlSection(sectionStats)) warningIds.push('farmingAbuseControlMissing');
  if (!runtime.routeViable) warningIds.push('routeContractFailed');
  return {
    mapId: map.id,
    mapName: map.name,
    level,
    layoutRole: String(map.layoutRole || ''),
    layoutStyle: String(map.layoutStyle || ''),
    intendedUseCase: String(intent && intent.intendedUseCase || ''),
    farmingAbuseRisk: String(intent && intent.farmingAbuseRisk || ''),
    targetPartySize,
    spawnSectionCount: sectionStats.sectionCount,
    activeSpawnSectionCount: sectionStats.activeSectionCount,
    emptySectionCount: sectionStats.emptySectionCount,
    spawnPointCount: sectionStats.spawnPointCount,
    sectionOccupancy: sectionStats.sections,
    arenaSkeleton: runtime.arenaSkeleton || map.arenaSkeleton || '',
    arenaMechanic: runtime.arenaMechanic || map.arenaMechanic || '',
    routeViable: !!runtime.routeViable,
    routeIssueIds: runtime.routeIssues || [],
    metrics: {
      killsPerMinute: roundNumber(killsPerMinute, 2),
      expPerMinute: roundNumber(killsPerMinute * avgXpPerKill, 1),
      dropValuePerMinute: roundNumber(dropValuePerMinute, 2),
      idleTimePercent: roundNumber(idleTimePercent, 1),
      travelSharePercent: roundNumber(travelSharePercent, 1),
      routeCycleSeconds: roundNumber(routeCycleSeconds, 1),
      damageTakenPerMinute: roundNumber(medianDamageTakenPerHour / 60, 1),
      potionUsesPerMinute: roundNumber(medianPotionUsesPerHour / 60, 2),
      deathRatePerHour: roundNumber(deathRatePerHour, 3),
      platformCoverage: roundNumber(runtime.platformCoverage || 0, 3),
      spawnVacancyPercent: roundNumber(spawnVacancyPercent, 1),
      nonCombatTraversalPercent: roundNumber(nonCombatTraversalPercent, 1),
      classPerformanceSpreadPercent: roundNumber(classPerformanceSpreadPercent, 1),
      partyOverlapPercent: roundNumber(partyOverlapPercent, 1),
      partyEfficiencyVsSoloPercent: roundNumber(partyEfficiencyVsSoloPercent, 1),
      eliteMinibossClearTimeSeconds,
      abandonmentRiskIndex: roundNumber(abandonmentRiskIndex, 1),
      repeatVisitationIndex: roundNumber(repeatVisitationIndex, 1)
    },
    respawnHandshakeRatio: roundNumber(respawnHandshakeRatio, 2),
    abuseControlPresent: hasMapAbuseControlSection(sectionStats),
    farmAbuseRelevant,
    abuseRiskIndex: roundNumber(abuseRiskIndex, 1),
    warningIds
  };
}

function createMapTuningSummary(entries) {
  const warningEntries = (entries || []).flatMap((entry) =>
    (entry.warningIds || []).map((warningId) => ({ mapId: entry.mapId, warningId })));
  const highRiskFarmMaps = (entries || []).filter((entry) => !!entry.farmAbuseRelevant);
  const highRiskWithControls = highRiskFarmMaps.filter((entry) => entry.abuseControlPresent);
  return {
    mapCount: entries.length,
    fieldMapCount: entries.filter((entry) => entry.layoutRole !== 'dungeon' && entry.layoutRole !== 'bossArena').length,
    dungeonMapCount: entries.filter((entry) => entry.layoutRole === 'dungeon').length,
    bossArenaMapCount: entries.filter((entry) => entry.layoutRole === 'bossArena').length,
    metricIds: MAP_TUNING_METRIC_IDS.slice(),
    warningThresholds: Object.assign({}, MAP_TUNING_WARNING_THRESHOLDS),
    warningCount: warningEntries.length,
    warnings: warningEntries,
    medianKillsPerMinute: roundNumber(medianNumber(entries.map((entry) => entry.metrics.killsPerMinute)), 2),
    medianExpPerMinute: roundNumber(medianNumber(entries.map((entry) => entry.metrics.expPerMinute)), 1),
    medianIdleTimePercent: roundNumber(medianNumber(entries.map((entry) => entry.metrics.idleTimePercent)), 1),
    medianTravelSharePercent: roundNumber(medianNumber(entries.map((entry) => entry.metrics.travelSharePercent)), 1),
    medianSpawnVacancyPercent: roundNumber(medianNumber(entries.map((entry) => entry.metrics.spawnVacancyPercent)), 1),
    medianClassPerformanceSpreadPercent: roundNumber(medianNumber(entries.map((entry) => entry.metrics.classPerformanceSpreadPercent)), 1),
    medianPartyOverlapPercent: roundNumber(medianNumber(entries.map((entry) => entry.metrics.partyOverlapPercent)), 1),
    highRiskFarmMapCount: highRiskFarmMaps.length,
    highRiskFarmControlCoverage: highRiskFarmMaps.length
      ? roundNumber(highRiskWithControls.length / highRiskFarmMaps.length, 3)
      : 1
  };
}

function createMapTuningReport(data, fieldMaps, runtimeEngine, rewardLookups) {
  const fieldById = new Map((fieldMaps || []).map((map) => [map.id, map]));
  const entries = getCombatMaps(data).map((map) =>
    createMapTuningEntry(data, map, fieldById.get(map.id) || null, runtimeEngine, rewardLookups));
  const summary = createMapTuningSummary(entries);
  return Object.assign({ maps: entries }, summary);
}

function getEngineMetric(engine, methodName, fallback, level, enemy) {
  if (engine && typeof engine[methodName] === 'function') return engine[methodName](level, enemy);
  return fallback(level, enemy);
}

function createMapBalanceReport(data, createProjectStarfallEngine, options = {}) {
  const classIds = Array.isArray(options.classIds) && options.classIds.length ? options.classIds : ALL_CLASS_IDS;
  const engine = typeof createProjectStarfallEngine === 'function' ? createProjectStarfallEngine(null, data) : null;
  const mapTuningEngine = typeof createProjectStarfallEngine === 'function' ? createProjectStarfallEngine(null, data) : engine;
  const rewardLookups = createRewardLookups(data);
  const maps = getFieldMaps(data).map((map) => {
    const level = getBenchmarkLevelFromRange(map.levelRange, options.level || 50);
    const rank = Math.max(1, Math.floor(Number(options.rank || 10) || 10));
    const eligibleClassIds = getEligibleClassIdsForMap(data, classIds, level);
    const profile = getMapProfile(data, map);
    const weights = getMapScenarioWeights(profile);
    const enemies = getMapEnemies(data, map);
    const spawnSections = Array.isArray(map.spawnSections) ? map.spawnSections : [];
    const spawnPoints = Array.isArray(map.spawnPoints) ? map.spawnPoints : [];
    const avgXpPerKill = Math.max(1, medianNumber(enemies.map((enemy) =>
      getEngineMetric(engine, 'getMonsterXp', getEstimatedMonsterXp, level, enemy))));
    const avgMonsterHp = Math.max(1, medianNumber(enemies.map((enemy) =>
      getEstimatedMonsterHp(level, enemy))));
    const levelXp = getEngineMetric(engine, 'getLevelXp', getEstimatedLevelXp, level);
    const targetMinutes = getTargetLevelMinutes(level);
    const medianKillsPerHour = Math.max(1, levelXp / avgXpPerKill / (targetMinutes / 60));
    const killCycleSeconds = 3600 / Math.max(1, medianKillsPerHour);
    const normalTtkSeconds = killCycleSeconds * getActiveCombatShare(level);
    const rewardCadence = getFieldRewardCadence(data, medianKillsPerHour);
    const rewardSource = createMapRewardSourceReport(data, map, enemies, medianKillsPerHour, rewardLookups);
    const classScores = eligibleClassIds.map((classId) => {
      const weightedDps = Object.keys(weights).reduce((sum, scenarioId) => {
        const scenario = BALANCE_SCENARIOS.find((candidate) => candidate.id === scenarioId);
        if (!scenario) return sum;
        return sum + runBalanceScenario(data, createProjectStarfallEngine, classId, scenario, { level, rank }).dps * weights[scenarioId];
      }, 0);
      const fitMultiplier = getClassMapFitMultiplier(data, classId, profile);
      return {
        classId,
        weightedDps,
        fitMultiplier,
        score: Math.max(1, weightedDps * fitMultiplier)
      };
    });
    const medianScore = Math.max(1, medianNumber(classScores.map((result) => result.score)));
    const results = classScores
      .map((result) => {
        const rawEfficiencyIndex = Math.max(1, result.score / medianScore * 100);
        const efficiencyIndex = Math.max(35, 100 + (rawEfficiencyIndex - 100) * 0.45);
        const killsPerHour = medianKillsPerHour * efficiencyIndex / 100;
        const xpPerHour = killsPerHour * avgXpPerKill;
        const currencyPerHour = killsPerHour * (4 + level * 0.3) * (1 + profile.eliteRatio * 0.25 + profile.armoredRatio * 0.08);
        const materialValueIndex = 100 * (1 + profile.armoredRatio * 0.12 + profile.supportRatio * 0.08 + profile.eliteRatio * 0.18);
        return {
          classId: result.classId,
          efficiencyIndex: Math.round(efficiencyIndex),
          rawEfficiencyIndex: Math.round(rawEfficiencyIndex),
          weightedDps: roundNumber(result.weightedDps, 1),
          fitMultiplier: result.fitMultiplier,
          killsPerHour: Math.round(killsPerHour),
          xpPerHour: Math.round(xpPerHour),
          currencyPerHour: Math.round(currencyPerHour),
          materialValueIndex: Math.round(materialValueIndex),
          timeToLevelMinutes: roundNumber(levelXp / Math.max(1, xpPerHour) * 60, 1)
        };
      })
      .sort((a, b) => b.efficiencyIndex - a.efficiencyIndex || b.xpPerHour - a.xpPerHour);
    const survivability = createMapSurvivabilityReport(data, createProjectStarfallEngine, Object.assign({}, map, { level }), profile, enemies, results, rank);
    return {
      id: map.id,
      name: map.name,
      level,
      levelRange: map.levelRange,
      waveMax: map.waveMax || 0,
      waveDelay: map.waveDelay || 0,
      designIntent: map.designIntent ? {
        intendedArchetype: String(map.designIntent.intendedArchetype || ''),
        intendedUseCase: String(map.designIntent.intendedUseCase || ''),
        routeSummary: String(map.designIntent.routeSummary || ''),
        partyRoleTarget: String(map.designIntent.partyRoleTarget || ''),
        farmingAbuseRisk: String(map.designIntent.farmingAbuseRisk || ''),
        visualIdentityTag: String(map.designIntent.visualIdentityTag || ''),
        spawnSectionModel: String(map.designIntent.spawnSectionModel || ''),
        partyScaling: String(map.designIntent.partyScaling || 'none'),
        priorityRedesign: !!map.designIntent.priorityRedesign
      } : null,
      spawnSectionCount: spawnSections.length,
      spawnPointSectionCoverage: spawnPoints.length
        ? roundNumber(spawnPoints.filter((point) => point && point.sectionId).length / spawnPoints.length, 3)
        : 0,
      spawnSections: spawnSections.map((section) => ({
        id: section.id,
        label: section.label,
        tier: section.tier || '',
        spawnModel: section.spawnModel || ''
      })),
      profile,
      scenarioWeights: weights,
      avgMonsterHp,
      avgXpPerKill: Math.round(avgXpPerKill),
      targetMinutes: roundNumber(targetMinutes, 1),
      medianKillsPerHour: Math.round(medianKillsPerHour),
      killCycleSeconds: roundNumber(killCycleSeconds, 1),
      normalTtkSeconds: roundNumber(normalTtkSeconds, 1),
      normalTtkTarget: getTtkTargetRange(level),
      rewardCadence,
      rewardSource,
      survivability,
      results
    };
  });

  const progression = FIELD_PROGRESSION_BRACKETS
    .map((bracket) => {
      const bracketMaps = maps.filter((map) => {
        if (bracket.id === 'endgame') return map.level >= bracket.minLevel;
        return map.level >= bracket.minLevel && map.level <= bracket.maxLevel;
      });
      if (!bracketMaps.length) {
        return {
          id: bracket.id,
          label: bracket.label,
          targetMinutes: bracket.targetMinutes,
          mapCount: 0,
          medianTimeToLevelMinutes: 0,
          slowestMedianClassMinutes: 0,
          fastestMedianClassMinutes: 0
        };
      }
      const mapMedians = bracketMaps.map((map) => medianNumber(map.results.map((result) => result.timeToLevelMinutes)));
      const ttkMedians = bracketMaps.map((map) => map.normalTtkSeconds);
      const smallRewardMedians = bracketMaps.map((map) => map.rewardCadence && map.rewardCadence.smallVisibleMinutes || 0);
      const mediumRewardMedians = bracketMaps.map((map) => map.rewardCadence && map.rewardCadence.mediumProgressMinutes || 0);
      return {
        id: bracket.id,
        label: bracket.label,
        targetMinutes: bracket.targetMinutes,
        targetTtkSeconds: FIELD_TTK_TARGETS[bracket.id] || FIELD_TTK_TARGETS.mid,
        mapCount: bracketMaps.length,
        medianTimeToLevelMinutes: roundNumber(medianNumber(mapMedians), 1),
        medianNormalTtkSeconds: roundNumber(medianNumber(ttkMedians), 1),
        medianSmallRewardMinutes: roundNumber(medianNumber(smallRewardMedians), 1),
        medianMediumProgressMinutes: roundNumber(medianNumber(mediumRewardMedians), 1),
        slowestMedianClassMinutes: roundNumber(Math.max(...mapMedians), 1),
        fastestMedianClassMinutes: roundNumber(Math.min(...mapMedians), 1),
        maps: bracketMaps.map((map) => map.id)
      };
    });

  return {
    maps,
    progression,
    mapTuning: createMapTuningReport(data, maps, mapTuningEngine, rewardLookups),
    rewardSource: createFieldRewardSourceSummary(maps),
    survivability: createFieldSurvivabilitySummary(maps),
    levelCurve: createLevelCurveReport(data, options)
  };
}

function getBossHpScalingMultiplier(players) {
  const count = Math.max(1, Math.floor(Number(players || 1) || 1));
  if (count <= 1) return 1;
  return 1 + 0.7 * Math.pow(count - 1, 0.85);
}

function getBossEncounterHpScale(encounter) {
  const scale = Number(encounter && encounter.hpScale || 1);
  return Number.isFinite(scale) ? Math.max(1, scale) : 1;
}

function createLookup(items, key = 'id') {
  return new Map((items || [])
    .filter((item) => item && item[key])
    .map((item) => [item[key], item]));
}

function flattenRewardKeys(reward) {
  const result = [];
  if (!reward || typeof reward !== 'object') return result;
  ['xp', 'currency'].forEach((key) => {
    if (Number(reward[key] || 0) > 0) result.push(key);
  });
  ['materials', 'consumables', 'cards', 'items'].forEach((bucketKey) => {
    const bucket = reward[bucketKey];
    if (!bucket || typeof bucket !== 'object') return;
    Object.keys(bucket).forEach((id) => result.push(`${bucketKey}:${id}`));
  });
  return result;
}

function getBossEncounterText(data, encounter, map, boss, dungeon) {
  const phaseText = (encounter.phases || []).map((phase) =>
    [phase.id, phase.name, phase.description].concat(phase.actions || []).join(' ')).join(' ');
  return [
    encounter.mechanic,
    encounter.intro,
    encounter.clearText,
    encounter.summary,
    phaseText,
    (encounter.adds || []).join(' '),
    map && map.purpose,
    boss && boss.mechanic,
    boss && boss.counter,
    dungeon && dungeon.summary
  ].filter(Boolean).join(' ').toLowerCase();
}

function getBossMechanicCategories(data, encounter, map, boss, dungeon, breakProfile) {
  const text = getBossEncounterText(data, encounter, map, boss, dungeon);
  const categories = BOSS_MECHANIC_CATEGORIES.filter((category) =>
    category.keywords.some((keyword) => text.includes(String(keyword).toLowerCase())));
  const ids = new Set(categories.map((category) => category.id));
  if (Array.isArray(encounter.adds) && encounter.adds.length >= 2) ids.add('addControl');
  if (breakProfile) ids.add('utilityControl');
  return BOSS_MECHANIC_CATEGORIES
    .filter((category) => ids.has(category.id))
    .map((category) => category.id);
}

function getDungeonForBoss(data, bossId) {
  return (data.DUNGEONS || []).find((dungeon) =>
    dungeon && (dungeon.bossId === bossId ||
      Array.isArray(dungeon.bossIds) && dungeon.bossIds.includes(bossId))) || null;
}

function getBossClearTimeClassIds(data, classIds, level) {
  return getEligibleClassIdsForMap(data, classIds && classIds.length ? classIds : ALL_CLASS_IDS, level)
    .filter((classId) => {
      const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
      return !advanced || level >= Math.max(1, Number(advanced.levelRequirement || 25) || 25);
    });
}

function createBossClearTimeEstimate(data, createProjectStarfallEngine, encounter, boss, level, options = {}) {
  const targetMin = BOSS_TTK_TARGET_MINUTES[0];
  const targetMax = BOSS_TTK_TARGET_MINUTES[1];
  const hpScale = getBossEncounterHpScale(encounter);
  const baseHp = getEstimatedMonsterHp(level, boss);
  const effectiveHp = Math.max(1, Math.round(baseHp * hpScale));
  const scenario = BALANCE_SCENARIOS.find((candidate) => candidate.id === 'singleBoss') || BALANCE_SCENARIOS[0];
  const rank = Math.max(1, Math.floor(Number(options.rank || 10) || 10));
  const classIds = getBossClearTimeClassIds(data, options.classIds || ALL_CLASS_IDS, level);
  const classResults = typeof createProjectStarfallEngine === 'function'
    ? classIds.map((classId) => {
      const result = runBalanceScenario(data, createProjectStarfallEngine, classId, scenario, { level, rank });
      const clearSeconds = effectiveHp / Math.max(1, Number(result.dps || 0));
      return {
        classId,
        dps: result.dps,
        clearMinutes: roundNumber(clearSeconds / 60, 2)
      };
    }).sort((a, b) => a.clearMinutes - b.clearMinutes)
    : [];
  const clearTimes = classResults.map((result) => result.clearMinutes);
  const medianSoloClearMinutes = roundNumber(medianNumber(clearTimes), 2);
  const fastestSoloClearMinutes = clearTimes.length ? roundNumber(Math.min(...clearTimes), 2) : 0;
  const slowestSoloClearMinutes = clearTimes.length ? roundNumber(Math.max(...clearTimes), 2) : 0;
  const issues = [];
  if (hpScale <= 1) issues.push('missingEncounterHpScale');
  if (medianSoloClearMinutes && medianSoloClearMinutes < targetMin) issues.push('medianClearTooFast');
  if (medianSoloClearMinutes && medianSoloClearMinutes > targetMax) issues.push('medianClearTooSlow');
  if (fastestSoloClearMinutes && fastestSoloClearMinutes < targetMin * 0.65) issues.push('specialistClearTooFast');
  if (slowestSoloClearMinutes && slowestSoloClearMinutes > targetMax * 1.65) issues.push('floorClearTooSlow');
  if (!classResults.length) issues.push('missingClassClearTimes');
  return {
    targetMinutes: BOSS_TTK_TARGET_MINUTES.slice(),
    level,
    baseHp,
    hpScale,
    effectiveHp,
    medianSoloClearMinutes,
    fastestSoloClearMinutes,
    slowestSoloClearMinutes,
    classResults,
    issueIds: issues
  };
}

function createBossClearTimeSummary(encounters) {
  const estimates = (encounters || []).map((encounter) => encounter.clearTime).filter(Boolean);
  const issueEntries = estimates.flatMap((estimate) =>
    (estimate.issueIds || []).map((issueId) => ({ bossId: estimate.bossId || '', issueId })));
  return {
    targetMinutes: BOSS_TTK_TARGET_MINUTES.slice(),
    bossCount: estimates.length,
    medianSoloClearMinutes: roundNumber(medianNumber(estimates.map((estimate) => estimate.medianSoloClearMinutes)), 2),
    fastestSoloClearMinutes: estimates.length ? roundNumber(Math.min(...estimates.map((estimate) => estimate.fastestSoloClearMinutes)), 2) : 0,
    slowestSoloClearMinutes: estimates.length ? roundNumber(Math.max(...estimates.map((estimate) => estimate.slowestSoloClearMinutes)), 2) : 0,
    minHpScale: estimates.length ? roundNumber(Math.min(...estimates.map((estimate) => estimate.hpScale)), 2) : 0,
    maxHpScale: estimates.length ? roundNumber(Math.max(...estimates.map((estimate) => estimate.hpScale)), 2) : 0,
    issueCount: issueEntries.length,
    issues: issueEntries
  };
}

function getBossPitySettings(data, rarity) {
  const key = String(rarity || '').toLowerCase();
  const economy = data && data.DROP_ECONOMY && data.DROP_ECONOMY.bossPity || {};
  const fallbackStart = key === 'relic' ? 10 : 8;
  const fallbackStep = key === 'relic' ? 0.0075 : 0.01;
  const fallbackMax = key === 'relic' ? 0.09 : 0.1;
  return {
    start: Math.max(0, Math.floor(Number(economy[`${key}Start`] == null ? fallbackStart : economy[`${key}Start`]) || fallbackStart)),
    step: Math.max(0, Number(economy[`${key}Step`] == null ? fallbackStep : economy[`${key}Step`]) || fallbackStep),
    max: Math.max(0, Number(economy[`${key}Max`] == null ? fallbackMax : economy[`${key}Max`]) || fallbackMax)
  };
}

function getBossSetChanceAtMissCount(data, source, misses) {
  const baseChance = clamp(Number(source && source.dropChance || 0), 0, 0.95);
  const pity = getBossPitySettings(data, source && source.rarity);
  const missCount = Math.max(0, Math.floor(Number(misses || 0) || 0));
  const pityBonus = missCount < pity.start
    ? 0
    : Math.min((missCount - pity.start + 1) * pity.step, pity.max);
  return clamp(baseChance + pityBonus, 0, 0.95);
}

function createBossSetDryStreakEstimate(data, encounter, bossSource) {
  const source = bossSource || {};
  const rarity = String(source.rarity || encounter && encounter.chaseRarity || '');
  const pity = getBossPitySettings(data, rarity);
  const medianClearMinutes = Number(encounter && encounter.clearTime && encounter.clearTime.medianSoloClearMinutes || 0);
  let survivalChance = 1;
  let expectedClears = 0;
  let p95Clears = 0;
  let p99Clears = 0;
  for (let clear = 1; clear <= 180; clear += 1) {
    const chance = getBossSetChanceAtMissCount(data, source, clear - 1);
    const hitChance = survivalChance * chance;
    expectedClears += clear * hitChance;
    survivalChance *= 1 - chance;
    const cumulativeChance = 1 - survivalChance;
    if (!p95Clears && cumulativeChance >= 0.95) p95Clears = clear;
    if (!p99Clears && cumulativeChance >= 0.99) {
      p99Clears = clear;
      break;
    }
  }
  const maxChance = getBossSetChanceAtMissCount(data, source, pity.start + Math.ceil(pity.max / Math.max(pity.step, 0.0001)) + 1);
  const p95Hours = medianClearMinutes > 0 && p95Clears ? p95Clears * medianClearMinutes / 60 : 0;
  const issues = [];
  if (!source || !source.bossId) issues.push('missingBossChaseSource');
  if (!pity.step || !pity.max) issues.push('missingSoftPity');
  if (p95Clears > 40) issues.push('p95DryStreakTooLong');
  if (p99Clears > 60) issues.push('p99DryStreakTooLong');
  if (p95Hours > 8) issues.push('p95SessionHoursTooLong');
  return {
    bossId: encounter && encounter.bossId || source.bossId || '',
    setId: source.setId || encounter && encounter.setId || '',
    rarity,
    baseDropChance: roundNumber(source.dropChance || 0, 3),
    pityStart: pity.start,
    pityStep: roundNumber(pity.step, 4),
    pityMax: roundNumber(pity.max, 3),
    maxDropChance: roundNumber(maxChance, 3),
    expectedClears: roundNumber(expectedClears, 2),
    p95Clears,
    p99Clears,
    p95Hours: roundNumber(p95Hours, 2),
    issueIds: issues
  };
}

function createBossSetDryStreakSummary(encounters) {
  const estimates = (encounters || [])
    .filter((encounter) => encounter && encounter.dryStreak)
    .map((encounter) => encounter.dryStreak);
  const issues = estimates.flatMap((estimate) =>
    (estimate.issueIds || []).map((issueId) => ({ bossId: estimate.bossId, issueId })));
  const byRarity = Array.from(new Set(estimates.map((estimate) => estimate.rarity).filter(Boolean))).sort()
    .reduce((result, rarity) => {
      const matching = estimates.filter((estimate) => estimate.rarity === rarity);
      result[rarity] = {
        count: matching.length,
        maxP95Clears: Math.max(...matching.map((estimate) => estimate.p95Clears), 0),
        maxP99Clears: Math.max(...matching.map((estimate) => estimate.p99Clears), 0),
        medianExpectedClears: roundNumber(medianNumber(matching.map((estimate) => estimate.expectedClears)), 2)
      };
      return result;
    }, {});
  return {
    sourceCount: estimates.length,
    issueCount: issues.length,
    issues,
    maxP95Clears: Math.max(...estimates.map((estimate) => estimate.p95Clears), 0),
    maxP99Clears: Math.max(...estimates.map((estimate) => estimate.p99Clears), 0),
    maxP95Hours: roundNumber(Math.max(...estimates.map((estimate) => estimate.p95Hours), 0), 2),
    medianExpectedClears: roundNumber(medianNumber(estimates.map((estimate) => estimate.expectedClears)), 2),
    byRarity,
    estimates
  };
}

function createEmptySupportScores() {
  return SUPPORT_CONTRIBUTION_CATEGORIES.reduce((scores, category) => {
    scores[category.id] = 0;
    return scores;
  }, {});
}

function addSupportScore(scores, categoryId, value) {
  const category = SUPPORT_CONTRIBUTION_CATEGORIES.find((item) => item.id === categoryId);
  if (!category) return;
  scores[categoryId] = Math.min(category.maxScore, Number(scores[categoryId] || 0) + Number(value || 0));
}

function addSupportKeywordScores(scores, text, multiplier = 1) {
  const normalized = String(text || '').toLowerCase();
  if (!normalized) return;
  SUPPORT_CONTRIBUTION_CATEGORIES.forEach((category) => {
    const matches = category.keywords.reduce((count, keyword) =>
      count + (normalized.includes(String(keyword).toLowerCase()) ? 1 : 0), 0);
    if (matches) addSupportScore(scores, category.id, matches * 4 * multiplier);
  });
}

function getClassSupportRoleWeight(data, classId) {
  const classData = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId] ||
    data.BASE_CLASSES && data.BASE_CLASSES[classId] || {};
  const profile = classData.roleProfile || {};
  const text = [profile.primary, profile.secondary, profile.specialty, profile.summary, classData.description]
    .filter(Boolean).join(' ').toLowerCase();
  let weight = 1;
  if (text.includes('support')) weight += 0.2;
  if (text.includes('control')) weight += 0.12;
  if (text.includes('boss safety') || text.includes('sustain')) weight += 0.08;
  return roundNumber(weight, 2);
}

function addSupportSkillScores(data, scores, skill, multiplier = 1) {
  if (!skill) return;
  const tags = new Set(skill.roleTags || []);
  const purpose = String(skill.purpose || '');
  const skillText = [
    skill.id,
    skill.name,
    skill.type,
    purpose,
    (skill.roleTags || []).join(' '),
    skill.description,
    skill.partyEffect,
    skill.futurePartyEffect
  ].filter(Boolean).join(' ');
  addSupportKeywordScores(scores, skillText, multiplier);
  if (tags.has('Support')) addSupportScore(scores, 'objectiveUtility', 8 * multiplier);
  if (tags.has('Party')) addSupportScore(scores, 'objectiveUtility', 10 * multiplier);
  if (tags.has('Control')) addSupportScore(scores, 'control', 8 * multiplier);
  if (tags.has('Bossing') && /mark|break|armor|weak/i.test(skillText)) addSupportScore(scores, 'partyDamage', 5 * multiplier);
  if (purpose === 'defense') addSupportScore(scores, 'mitigation', 12 * multiplier);
  if (purpose === 'sustain') addSupportScore(scores, 'sustain', 12 * multiplier);
  if (purpose === 'control' || purpose === 'setup') addSupportScore(scores, 'control', 6 * multiplier);
  if (purpose === 'party') addSupportScore(scores, 'objectiveUtility', 10 * multiplier);
  if (purpose === 'buff') addSupportScore(scores, 'partyDamage', 4 * multiplier);
}

function createPartySupportContributionReport(data) {
  const classIds = Array.from(new Set(
    Object.keys(data.BASE_CLASSES || {}).concat(Object.keys(data.ADVANCED_CLASSES || {}))));
  const entries = classIds.map((classId) => {
    const scores = createEmptySupportScores();
    const classData = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId] ||
      data.BASE_CLASSES && data.BASE_CLASSES[classId] || {};
    const roleWeight = getClassSupportRoleWeight(data, classId);
    addSupportKeywordScores(scores, [
      classData.description,
      classData.roleProfile && classData.roleProfile.primary,
      classData.roleProfile && classData.roleProfile.secondary,
      classData.roleProfile && classData.roleProfile.specialty,
      classData.roleProfile && classData.roleProfile.summary
    ].filter(Boolean).join(' '), roleWeight);
    const partySkill = classData.partySkillId ? getSkill(data, classData.partySkillId) : null;
    addSupportSkillScores(data, scores, partySkill, roleWeight * 1.2);
    const loadout = data.PARTY_AI_LOADOUTS && data.PARTY_AI_LOADOUTS[classId] || {};
    const loadoutSkills = Array.isArray(loadout.skills) ? loadout.skills : [];
    loadoutSkills.forEach((entry) => {
      const priority = Math.max(1, Number(entry && entry.priority || 1) || 1);
      const cooldown = Math.max(1, Number(entry && entry.minCooldown || 6) || 6);
      const multiplier = (entry && entry.support ? 1.25 : 0.72) * (1 + Math.min(0.28, priority * 0.04)) * (1 + Math.min(0.18, 8 / cooldown * 0.05));
      addSupportSkillScores(data, scores, getSkill(data, entry && entry.skillId), multiplier);
    });
    const totalSupportIndex = roundNumber(Object.values(scores).reduce((sum, score) => sum + score, 0), 1);
    return {
      classId,
      primaryRole: String(classData.roleProfile && classData.roleProfile.primary || ''),
      partySkillId: classData.partySkillId || '',
      loadoutSkillCount: loadoutSkills.length,
      supportLoadoutSkillCount: loadoutSkills.filter((entry) => entry && entry.support).length,
      totalSupportIndex,
      scores: SUPPORT_CONTRIBUTION_CATEGORIES.reduce((result, category) => {
        result[category.id] = roundNumber(scores[category.id], 1);
        return result;
      }, {})
    };
  }).sort((a, b) => b.totalSupportIndex - a.totalSupportIndex || a.classId.localeCompare(b.classId));
  const categoryTotals = SUPPORT_CONTRIBUTION_CATEGORIES.reduce((result, category) => {
    result[category.id] = roundNumber(entries.reduce((sum, entry) => sum + Number(entry.scores[category.id] || 0), 0), 1);
    return result;
  }, {});
  const total = Math.max(1, entries.reduce((sum, entry) => sum + Number(entry.totalSupportIndex || 0), 0));
  const supportContributorCount = entries.filter((entry) => entry.totalSupportIndex >= 45).length;
  const supportRoleEntries = entries.filter((entry) => /support|control/i.test(`${entry.primaryRole} ${entry.partySkillId}`) ||
    entry.supportLoadoutSkillCount > 0);
  const maxSingleClassShare = roundNumber(Math.max(...entries.map((entry) => entry.totalSupportIndex / total), 0), 3);
  const coveredCategoryCount = SUPPORT_CONTRIBUTION_CATEGORIES
    .filter((category) => Number(categoryTotals[category.id] || 0) > 0).length;
  const issues = [];
  if (supportContributorCount < 5) issues.push('tooFewSupportContributors');
  if (coveredCategoryCount < SUPPORT_CONTRIBUTION_CATEGORIES.length) issues.push('missingSupportCategory');
  if (maxSingleClassShare > 0.32) issues.push('supportValueTooConcentrated');
  if (supportRoleEntries.some((entry) => entry.totalSupportIndex < 55)) issues.push('supportRoleUnderScored');
  return {
    classCount: entries.length,
    supportContributorCount,
    supportRoleCount: supportRoleEntries.length,
    coveredCategoryCount,
    maxSingleClassShare,
    issueCount: issues.length,
    issueIds: issues,
    categoryTotals,
    topContributors: entries.slice(0, 5).map((entry) => ({
      classId: entry.classId,
      totalSupportIndex: entry.totalSupportIndex,
      scores: entry.scores
    })),
    entries
  };
}

function createBossPartyReport(data, createProjectStarfallEngine, options = {}) {
  const mapById = createLookup(data.MAPS || []);
  const enemyById = createLookup(data.ENEMIES || []);
  const breakByBossId = createLookup(data.BOSS_BREAK_PROFILES || [], 'bossId');
  const setByBossId = createLookup(data.EQUIPMENT_SETS || [], 'bossId');
  const bossSourceByBossId = createLookup(data.BOSS_EQUIPMENT_SOURCES || [], 'bossId');
  const dungeonObjectives = data.DUNGEON_OBJECTIVES || [];
  const encounters = (data.BOSS_ENCOUNTERS || []).map((encounter) => {
    const map = mapById.get(encounter.mapId) || null;
    const boss = enemyById.get(encounter.bossId) || null;
    const dungeon = getDungeonForBoss(data, encounter.bossId);
    const breakProfile = breakByBossId.get(encounter.bossId) || null;
    const bossSet = encounter.setId ? setByBossId.get(encounter.bossId) || null : null;
    const bossSource = bossSourceByBossId.get(encounter.bossId) || null;
    const categories = getBossMechanicCategories(data, encounter, map, boss, dungeon, breakProfile);
    const phaseActionIds = Array.from(new Set((encounter.phases || []).flatMap((phase) => phase.actions || [])));
    const deterministicRewardKeys = flattenRewardKeys(breakProfile && breakProfile.reward)
      .concat(flattenRewardKeys(dungeon && dungeon.rewards));
    const level = bossSource && bossSource.level || getBenchmarkLevelFromRange(map && map.levelRange, options.level || 50);
    const clearTime = createBossClearTimeEstimate(data, createProjectStarfallEngine, encounter, boss, level, options);
    const dryStreak = encounter.setId ? createBossSetDryStreakEstimate(data, Object.assign({}, encounter, { clearTime }), bossSource) : null;
    const issues = [];
    if (!map || !map.bossRoom || map.bossId !== encounter.bossId) issues.push('missingBossRoom');
    if (!boss || boss.behavior !== 'boss') issues.push('missingBossEnemy');
    if (!breakProfile) issues.push('missingBreakProfile');
    if (categories.length < 4) issues.push('tooFewMechanicCategories');
    if (!Array.isArray(encounter.adds) || encounter.adds.length < 3) issues.push('thinAddPool');
    if (!deterministicRewardKeys.length) issues.push('missingDeterministicProgress');
    if (encounter.setId && (!bossSet || !bossSource)) issues.push('missingBossSetChase');
    if (clearTime.issueIds.length) issues.push('clearTimeOutOfRange');
    if (dryStreak && dryStreak.issueIds.length) issues.push('dryStreakOutOfRange');
    return {
      id: encounter.id,
      bossId: encounter.bossId,
      name: encounter.name,
      mapId: encounter.mapId,
      dungeonId: dungeon && dungeon.id || '',
      level,
      categories,
      categoryCount: categories.length,
      phaseCount: Array.isArray(encounter.phases) ? encounter.phases.length : 0,
      phaseActionIds,
      addCount: Array.isArray(encounter.adds) ? encounter.adds.length : 0,
      hasBreakProfile: !!breakProfile,
      breakDuration: roundNumber(breakProfile && breakProfile.duration || 0, 1),
      breakDamageTakenScale: roundNumber(breakProfile && breakProfile.damageTakenScale || 1, 2),
      deterministicRewardKeys: Array.from(new Set(deterministicRewardKeys)),
      hasRandomChase: !!(encounter.setId && bossSet && bossSource),
      chaseRarity: bossSource && bossSource.rarity || '',
      chaseDropChance: roundNumber(bossSource && bossSource.dropChance || 0, 3),
      recommendedPartySize: dungeon && Number(dungeon.recommendedPartySize || 0) || 0,
      clearTime: Object.assign({ bossId: encounter.bossId }, clearTime),
      dryStreak,
      issues
    };
  });
  const clearTimeSummary = createBossClearTimeSummary(encounters);
  const dryStreakSummary = createBossSetDryStreakSummary(encounters);
  const categoryCoverage = BOSS_MECHANIC_CATEGORIES.map((category) => {
    const matching = encounters.filter((encounter) => encounter.categories.includes(category.id));
    return {
      id: category.id,
      label: category.label,
      count: matching.length,
      bossIds: matching.map((encounter) => encounter.bossId)
    };
  });
  const partyScaling = [1, 2, 3, 4].map((players) => {
    const hpMultiplier = getBossHpScalingMultiplier(players);
    return {
      players,
      hpMultiplier: roundNumber(hpMultiplier, 2),
      perPlayerHpIndex: roundNumber(hpMultiplier / players * 100, 1),
      throughputIndexVsSolo: roundNumber(players / hpMultiplier * 100, 1)
    };
  });
  const loadoutClassIds = Array.from(new Set(
    Object.keys(data.BASE_CLASSES || {}).concat(Object.keys(data.ADVANCED_CLASSES || {}))));
  const loadoutCoverage = loadoutClassIds.filter((classId) => {
    const loadout = data.PARTY_AI_LOADOUTS && data.PARTY_AI_LOADOUTS[classId];
    return loadout && loadout.equipment && loadout.equipment.weapon && Array.isArray(loadout.skills) && loadout.skills.length;
  });
  const supportLoadouts = loadoutClassIds.filter((classId) => {
    const loadout = data.PARTY_AI_LOADOUTS && data.PARTY_AI_LOADOUTS[classId];
    return loadout && Array.isArray(loadout.skills) && loadout.skills.some((entry) => entry && entry.support);
  });
  const supportContribution = createPartySupportContributionReport(data);
  const partyCommands = data.PARTY_COMMANDS || [];
  return {
    encounters,
    categoryCoverage,
    summary: {
      encounterCount: encounters.length,
      bossRoomCount: encounters.filter((encounter) => !encounter.issues.includes('missingBossRoom')).length,
      breakProfileCount: encounters.filter((encounter) => encounter.hasBreakProfile).length,
      deterministicProgressCount: encounters.filter((encounter) => encounter.deterministicRewardKeys.length).length,
      randomChaseCount: encounters.filter((encounter) => encounter.hasRandomChase).length,
      clearTimeIssueCount: clearTimeSummary.issueCount,
      dryStreakIssueCount: dryStreakSummary.issueCount,
      supportContributionIssueCount: supportContribution.issueCount,
      medianSoloClearMinutes: clearTimeSummary.medianSoloClearMinutes,
      minimumCategoryCount: encounters.length ? Math.min(...encounters.map((encounter) => encounter.categoryCount)) : 0,
      issueCount: encounters.reduce((sum, encounter) => sum + encounter.issues.length, 0) + supportContribution.issueCount
    },
    clearTime: clearTimeSummary,
    dryStreak: dryStreakSummary,
    supportContribution,
    party: {
      recommendedPartySizes: Array.from(new Set((data.DUNGEONS || []).map((dungeon) => Number(dungeon.recommendedPartySize || 0)).filter(Boolean))).sort((a, b) => a - b),
      commandCount: partyCommands.length,
      commandIds: partyCommands.map((command) => command.id),
      loadoutCoverageCount: loadoutCoverage.length,
      loadoutClassCount: loadoutClassIds.length,
      supportLoadoutCount: supportLoadouts.length,
      supportLoadoutIds: supportLoadouts,
      objectiveIds: dungeonObjectives.map((objective) => objective.id),
      hasBossBreakObjective: dungeonObjectives.some((objective) => objective.type === 'bossBreak'),
      hasPartySurvivalObjective: dungeonObjectives.some((objective) => objective.type === 'partySurvival'),
      hasTimedClearObjective: dungeonObjectives.some((objective) => objective.type === 'timedClear')
    },
    partyScaling
  };
}

function getAuditUpgradeRangeIndex(upgradeLevel) {
  const level = Number(upgradeLevel) || 0;
  if (level <= 5) return 0;
  if (level <= 10) return 1;
  if (level <= 15) return 2;
  return 3;
}

function getAuditUpgradeDustCost(data, upgradeLevel) {
  const costs = Array.isArray(data && data.UPGRADE_DUST_COST_BY_RANGE) ? data.UPGRADE_DUST_COST_BY_RANGE : [];
  const rangeIndex = getAuditUpgradeRangeIndex(upgradeLevel);
  return Math.max(1, Math.floor(Number(costs[rangeIndex]) || (1 + Math.floor((Number(upgradeLevel) || 0) / 4))));
}

function getAuditUpgradeAides(data, aideIds) {
  const ids = Array.isArray(aideIds) ? aideIds : [];
  const byType = {};
  ids.forEach((aideId) => {
    const id = String(aideId || '');
    const aide = (data.UPGRADE_AIDES || []).find((candidate) =>
      String(candidate && (candidate.id || candidate.materialId) || '') === id);
    if (aide && aide.type && !byType[aide.type]) byType[aide.type] = aide;
  });
  return Object.values(byType);
}

function getAuditModifiedUpgradeOutcomes(data, upgradeLevel, aideIds) {
  const rangeIndex = getAuditUpgradeRangeIndex(upgradeLevel);
  const aides = getAuditUpgradeAides(data, aideIds);
  const outcomes = (data.UPGRADE_OUTCOMES || []).map((outcome) => Object.assign({}, outcome, {
    weight: Math.max(0, Number(outcome.weightByRange && outcome.weightByRange[rangeIndex]) || 0)
  }));
  const totalWeight = outcomes.reduce((sum, outcome) => sum + outcome.weight, 0);
  const successBonus = aides.reduce((sum, aide) => sum + Math.max(0, Number(aide.successBonus || 0)), 0);
  const successBonusWeight = totalWeight * successBonus / 100;
  if (successBonusWeight > 0) {
    const success = outcomes.find((outcome) => outcome.id === 'success');
    const penalties = outcomes.filter((outcome) => outcome.id !== 'success' && outcome.weight > 0);
    const penaltyTotal = penalties.reduce((sum, outcome) => sum + outcome.weight, 0);
    const applied = Math.min(successBonusWeight, penaltyTotal);
    if (success && applied > 0 && penaltyTotal > 0) {
      success.weight += applied;
      penalties.forEach((outcome) => {
        outcome.weight = Math.max(0, outcome.weight - applied * (outcome.weight / penaltyTotal));
      });
    }
  }
  return outcomes.filter((outcome) => outcome.weight > 0);
}

function createUpgradeBandEntry(data, band, aideIds, options = {}) {
  const outcomes = getAuditModifiedUpgradeOutcomes(data, band.sampleLevel, aideIds);
  const totalWeight = Math.max(1, outcomes.reduce((sum, outcome) => sum + Number(outcome.weight || 0), 0));
  const chanceByOutcome = outcomes.reduce((result, outcome) => {
    result[outcome.id] = Number(outcome.weight || 0) / totalWeight;
    return result;
  }, {});
  const successLevels = Math.max(1, Math.floor(Number(options.successLevels || 1) || 1));
  const protectDestroy = !!options.protectDestroy;
  const successChance = Number(chanceByOutcome.success || 0);
  const failChance = Number(chanceByOutcome.fail || 0);
  const destroyChance = Number(chanceByOutcome.destroy || 0);
  const finalDestroyChance = protectDestroy ? 0 : destroyChance;
  const finalFailureChance = failChance + (protectDestroy ? destroyChance : 0);
  const expectedUpgradeDelta = successChance * successLevels - finalFailureChance;
  return {
    id: band.id,
    label: band.label,
    minUpgrade: band.minUpgrade,
    maxUpgrade: band.maxUpgrade,
    sampleLevel: band.sampleLevel,
    dustCost: getAuditUpgradeDustCost(data, band.sampleLevel),
    successChance: roundNumber(successChance, 3),
    failChance: roundNumber(failChance, 3),
    destroyChance: roundNumber(destroyChance, 3),
    finalDestroyChance: roundNumber(finalDestroyChance, 3),
    expectedUpgradeDelta: roundNumber(expectedUpgradeDelta, 3)
  };
}

function collectMaterialRewardIds(reward, ids = new Set()) {
  if (!reward || typeof reward !== 'object') return ids;
  const materials = reward.materials;
  if (materials && typeof materials === 'object') {
    Object.keys(materials).forEach((materialId) => ids.add(materialId));
  }
  Object.keys(reward).forEach((key) => {
    const value = reward[key];
    if (value && typeof value === 'object') collectMaterialRewardIds(value, ids);
  });
  return ids;
}

function createFormulaStatsSample(data, createProjectStarfallEngine, classId, level) {
  const engine = createProjectStarfallEngine(null, data);
  const baseId = getClassBaseId(data, classId);
  if (!engine.chooseClass(baseId)) throw new Error(`Unable to choose formula class ${baseId}`);
  engine.state.player.level = Math.max(1, Math.floor(Number(level || 1) || 1));
  if (data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId]) {
    engine.state.player.advancedClassId = classId;
  }
  const stats = engine.getStats();
  const effectiveHp = Number(stats.maxHp || 0) * (1 + Math.max(0, Number(stats.defense || 0)) * 2.25 / 100);
  const damageRange = stats.damageRange || {};
  return {
    classId,
    baseId,
    level: engine.state.player.level,
    maxHp: Math.round(Number(stats.maxHp || 0)),
    maxMp: Math.round(Number(stats.maxMp || 0)),
    power: Math.round(Number(stats.power || 0)),
    defense: Math.round(Number(stats.defense || 0)),
    effectiveHp: Math.round(effectiveHp),
    speed: Math.round(Number(stats.speed || 0)),
    range: Math.round(Number(stats.range || 0)),
    crit: roundNumber(stats.crit || 0, 1),
    critDamage: roundNumber(stats.critDamage || 0, 1),
    damageFloorPercent: roundNumber(damageRange.floorPercent || stats.damageFloor || 0, 1),
    averageDamage: roundNumber((Number(damageRange.min || 0) + Number(damageRange.max || 0)) / 2, 1),
    powerPercent: roundNumber(stats.powerPercent || 0, 1),
    attackDamagePercent: roundNumber(stats.attackDamagePercent || 0, 1)
  };
}

function addFormulaIdentityIndices(samples) {
  const stats = ['maxHp', 'maxMp', 'power', 'defense', 'effectiveHp', 'speed', 'range', 'averageDamage'];
  const medians = stats.reduce((result, statId) => {
    result[statId] = medianNumber(samples.map((sample) => sample[statId]));
    return result;
  }, {});
  return samples.map((sample) => {
    const indices = stats.reduce((result, statId) => {
      result[statId] = medians[statId] > 0 ? roundNumber(sample[statId] / medians[statId] * 100, 1) : 100;
      return result;
    }, {});
    return Object.assign({}, sample, { indices });
  });
}

function createFormulaIdentityLevelEntry(data, createProjectStarfallEngine, level) {
  const samples = addFormulaIdentityIndices(BASE_CLASS_IDS.map((classId) =>
    createFormulaStatsSample(data, createProjectStarfallEngine, classId, level)));
  const sampleByClass = new Map(samples.map((sample) => [sample.classId, sample]));
  const powers = samples.map((sample) => sample.power);
  const medianPower = medianNumber(powers);
  const powerSpreadPercent = medianPower > 0
    ? roundNumber((Math.max(...powers) - Math.min(...powers)) / medianPower * 100, 1)
    : 0;
  const issues = [];
  const fighter = sampleByClass.get('fighter');
  const mage = sampleByClass.get('mage');
  const archer = sampleByClass.get('archer');
  if (!fighter || !mage || !archer) issues.push('missingBaseClassStats');
  if (fighter && fighter.indices.effectiveHp < 106) issues.push('warriorEffectiveHpIdentityWeak');
  if (mage && mage.indices.maxMp < 108) issues.push('mageResourceIdentityWeak');
  if (archer && (archer.indices.speed < 105 || archer.indices.range < 110)) issues.push('archerRouteIdentityWeak');
  if (powerSpreadPercent > 8) issues.push('basePowerSpreadTooWide');
  return {
    level,
    samples,
    medianPower: roundNumber(medianPower, 1),
    powerSpreadPercent,
    issueIds: issues
  };
}

function getFormulaMinIndex(identityLevels, classId, statId) {
  const values = identityLevels
    .map((entry) => (entry.samples || []).find((sample) => sample.classId === classId))
    .map((sample) => sample && sample.indices && sample.indices[statId])
    .filter((value) => Number.isFinite(Number(value)));
  return values.length ? roundNumber(Math.min(...values), 1) : 0;
}

function collectFormulaStatSourceCounts(data) {
  const hasMultiplierStat = (stats) => !!(stats && typeof stats === 'object' &&
    DAMAGE_MULTIPLIER_STATS.some((statId) => Number(stats[statId] || 0) !== 0));
  const equipmentItems = []
    .concat(data.SHOP_ITEMS || [])
    .concat(data.RANDOM_EQUIPMENT_ITEMS || [])
    .concat(data.BOSS_EQUIPMENT_ITEMS || []);
  return {
    additiveSourceCount: STAT_FORMULA_SOURCE_IDS.length,
    potentialMultiplierLines: (data.POTENTIAL_LINE_POOLS || []).filter((line) =>
      line && DAMAGE_MULTIPLIER_STATS.includes(line.stat)).length,
    equipmentMultiplierItems: equipmentItems.filter((item) => item && hasMultiplierStat(item.stats)).length,
    setMultiplierBonuses: (data.EQUIPMENT_SETS || []).reduce((count, set) =>
      count + ((set && set.bonuses || []).filter((bonus) => bonus && hasMultiplierStat(bonus.stats)).length), 0),
    gearTraitMultiplierBonuses: (data.GEAR_TRAITS || []).filter((trait) =>
      trait && hasMultiplierStat(trait.statBonuses)).length,
    cardMultiplierBonuses: (data.CARD_DEFINITIONS || []).filter((card) =>
      card && hasMultiplierStat(card.baseStats)).length,
    classMasteryMultiplierBonuses: Object.values(data.CLASS_MASTERY_TRACKS || {}).reduce((count, track) =>
      count + ((track && (track.levels || track.milestones) || []).filter((level) => level && hasMultiplierStat(level.statBonuses)).length), 0),
    specializationMultiplierBonuses: (data.SPECIALIZATIONS || []).filter((specialization) =>
      specialization && hasMultiplierStat(specialization.statBonuses)).length,
    passiveMultiplierSkills: (data.SKILLS || []).filter((skill) =>
      skill && skill.passiveStats && hasMultiplierStat(skill.passiveStats)).length
  };
}

function createFormulaMultiplierCapReport(data) {
  const multiplierLines = (data.POTENTIAL_LINE_POOLS || []).filter((line) =>
    line && DAMAGE_MULTIPLIER_STATS.includes(line.stat));
  const maxByTier = ['rare', 'epic', 'relic', 'mythic', 'ascendant', 'celestial'].reduce((result, tier) => {
    result[tier] = multiplierLines.length ? Math.max(...multiplierLines.map((line) =>
      Number(line.values && line.values[tier] && line.values[tier][1] || 0))) : 0;
    return result;
  }, {});
  const maxByStat = DAMAGE_MULTIPLIER_STATS.reduce((result, statId) => {
    const lines = multiplierLines.filter((line) => line.stat === statId);
    result[statId] = ['rare', 'relic', 'celestial'].reduce((tiers, tier) => {
      tiers[tier] = lines.length ? Math.max(...lines.map((line) =>
        Number(line.values && line.values[tier] && line.values[tier][1] || 0))) : 0;
      return tiers;
    }, {});
    return result;
  }, {});
  return {
    statIds: DAMAGE_MULTIPLIER_STATS.slice(),
    lineCount: multiplierLines.length,
    maxByTier,
    maxByStat,
    runtimeDamageBonusCapPercent: 100,
    runtimeDamageReductionCapPercent: 25
  };
}

function createFormulaMitigationReport() {
  const sampleRawHit = 100;
  const samples = [0, 25, 50, 100, 200].map((defense) => {
    const damage = mitigateEstimatedPlayerDamage(sampleRawHit, defense);
    return {
      defense,
      damage,
      reductionPercent: roundNumber((sampleRawHit - damage) / sampleRawHit * 100, 1),
      effectiveHpIndex: roundNumber(sampleRawHit / Math.max(1, damage) * 100, 1)
    };
  });
  const issues = [];
  const defense50 = samples.find((sample) => sample.defense === 50);
  const defense200 = samples.find((sample) => sample.defense === 200);
  const monotonic = samples.every((sample, index) => index === 0 || sample.damage <= samples[index - 1].damage);
  if (!monotonic) issues.push('defenseMitigationNotMonotonic');
  if (defense50 && (defense50.reductionPercent < 40 || defense50.reductionPercent > 65)) {
    issues.push('midDefenseMitigationOutOfBand');
  }
  if (defense200 && defense200.damage < 10) issues.push('highDefenseMitigationNearImmunity');
  return {
    sampleRawHit,
    samples,
    issueIds: issues
  };
}

function createDamageStatFormulaReport(data, createProjectStarfallEngine) {
  const identityLevels = [1, 30, 60].map((level) =>
    createFormulaIdentityLevelEntry(data, createProjectStarfallEngine, level));
  const multiplierCaps = createFormulaMultiplierCapReport(data);
  const mitigation = createFormulaMitigationReport();
  const sourceCounts = collectFormulaStatSourceCounts(data);
  const positiveMultiplierBucketCount = DAMAGE_FORMULA_BUCKETS.filter((bucket) => bucket.multiplier).length;
  const maxPowerSpreadPercent = Math.max(...identityLevels.map((entry) => entry.powerSpreadPercent));
  const issues = new Set();
  identityLevels.forEach((entry) => (entry.issueIds || []).forEach((issueId) => issues.add(issueId)));
  (mitigation.issueIds || []).forEach((issueId) => issues.add(issueId));
  if (sourceCounts.additiveSourceCount < 8) issues.add('statSourceCoverageTooLow');
  if (positiveMultiplierBucketCount > 7) issues.add('tooManyIndependentMultiplierBuckets');
  if (multiplierCaps.maxByTier.rare > 8 ||
    multiplierCaps.maxByTier.relic > 24 ||
    multiplierCaps.maxByTier.celestial > 60) {
    issues.add('attunementMultiplierCapRisk');
  }
  return {
    buckets: DAMAGE_FORMULA_BUCKETS.map((bucket) => Object.assign({}, bucket)),
    bucketCount: DAMAGE_FORMULA_BUCKETS.length,
    positiveMultiplierBucketCount,
    additiveSources: STAT_FORMULA_SOURCE_IDS.slice(),
    sourceCounts,
    baseIdentity: {
      levels: identityLevels,
      maxPowerSpreadPercent,
      fighterMinEffectiveHpIndex: getFormulaMinIndex(identityLevels, 'fighter', 'effectiveHp'),
      mageMinMpIndex: getFormulaMinIndex(identityLevels, 'mage', 'maxMp'),
      archerMinSpeedIndex: getFormulaMinIndex(identityLevels, 'archer', 'speed'),
      archerMinRangeIndex: getFormulaMinIndex(identityLevels, 'archer', 'range')
    },
    multiplierCaps,
    mitigation,
    damageFloor: {
      basePercent: 50,
      maxPercent: 90
    },
    issueCount: issues.size,
    issueIds: Array.from(issues)
  };
}

function createEquipmentUpgradeReport(data) {
  const equipmentItems = []
    .concat(data.SHOP_ITEMS || [])
    .concat(data.RANDOM_EQUIPMENT_ITEMS || [])
    .concat(data.BOSS_EQUIPMENT_ITEMS || []);
  const deterministicWeapons = BASE_CLASS_IDS.map((classId) => {
    const levels = (data.SHOP_ITEMS || [])
      .filter((item) => item && item.slot === 'weapon' && (item.classId === classId || item.classId === 'any'))
      .map((item) => Math.max(1, Number(item.level || 1) || 1))
      .sort((a, b) => a - b);
    const gaps = levels.slice(1).map((level, index) => level - levels[index]);
    return {
      classId,
      weaponCount: levels.length,
      minLevel: levels.length ? levels[0] : 0,
      maxLevel: levels.length ? levels[levels.length - 1] : 0,
      maxLevelGap: gaps.length ? Math.max(...gaps) : 0,
      levels
    };
  });
  const advancedOffhands = ADVANCED_CLASS_IDS.map((classId) => {
    const items = (data.SHOP_ITEMS || []).filter((item) =>
      item && item.source === 'Class Supplier' && item.slot === 'offhand' && item.classId === classId);
    return {
      classId,
      itemCount: items.length,
      itemIds: items.map((item) => item.id)
    };
  });
  const bands = [
    { id: 'baselineA', label: '+0-5', minUpgrade: 0, maxUpgrade: 5, sampleLevel: 0 },
    { id: 'baselineB', label: '+6-10', minUpgrade: 6, maxUpgrade: 10, sampleLevel: 6 },
    { id: 'prestigeA', label: '+11-15', minUpgrade: 11, maxUpgrade: 15, sampleLevel: 11 },
    { id: 'prestigeB', label: '+16-20', minUpgrade: 16, maxUpgrade: 20, sampleLevel: 16 }
  ];
  const aideIds = (data.UPGRADE_AIDES || []).map((aide) => aide && (aide.materialId || aide.id)).filter(Boolean);
  const protectionAide = (data.UPGRADE_AIDES || []).find((aide) => aide && aide.protectsDestroy);
  const enhancementBonus = getAuditUpgradeAides(data, aideIds)
    .reduce((sum, aide) => sum + Math.max(0, Math.floor(Number(aide.successUpgradeBonus || 0) || 0)), 0);
  const unprotectedBands = bands.map((band) => createUpgradeBandEntry(data, band, [], { successLevels: 1 }));
  const protectedBands = bands.map((band) => createUpgradeBandEntry(data, band, aideIds, {
    protectDestroy: !!protectionAide,
    successLevels: 1 + enhancementBonus
  }));
  const baselineBands = unprotectedBands.filter((band) => band.maxUpgrade <= 10);
  const prestigeBands = unprotectedBands.filter((band) => band.minUpgrade >= 11);
  const upgradeMaterialIds = ['upgradeDust', 'upgradeCatalyst', 'wardingScroll', 'refinementCore', 'cubeFragment'];
  const enemyMaterialSources = upgradeMaterialIds.reduce((result, materialId) => {
    result[materialId] = (data.ENEMIES || []).filter((enemy) =>
      (enemy.dropPool && enemy.dropPool.materials || []).some((entry) => entry && entry.materialId === materialId)).length;
    return result;
  }, {});
  const deterministicRewardMaterialIds = new Set();
  []
    .concat(data.QUESTS || [])
    .concat(data.ACCOMPLISHMENTS || [])
    .concat(data.DAILY_REWARDS || [])
    .concat(data.PLINKO_BOARDS || [])
    .forEach((entry) => collectMaterialRewardIds(entry && (entry.reward || entry.rewards || entry), deterministicRewardMaterialIds));
  const deterministicMaterialCoverage = upgradeMaterialIds.reduce((result, materialId) => {
    result[materialId] = deterministicRewardMaterialIds.has(materialId);
    return result;
  }, {});
  const multiplierLines = (data.POTENTIAL_LINE_POOLS || []).filter((line) => DAMAGE_MULTIPLIER_STATS.includes(line && line.stat));
  const maxMultiplierByTier = ['rare', 'epic', 'relic', 'mythic', 'ascendant', 'celestial'].reduce((result, tier) => {
    result[tier] = Math.max(...multiplierLines.map((line) =>
      Number(line.values && line.values[tier] && line.values[tier][1] || 0)), 0);
    return result;
  }, {});
  const issues = [];
  if (deterministicWeapons.some((entry) => entry.weaponCount < 6 || entry.maxLevel < 70 || entry.maxLevelGap > 20)) issues.push('deterministicWeaponCoverageGap');
  if (advancedOffhands.some((entry) => entry.itemCount <= 0)) issues.push('advancedOffhandCoverageGap');
  if (baselineBands.some((band) => band.destroyChance > 0 || band.expectedUpgradeDelta <= 0)) issues.push('baselineUpgradeTooPunitive');
  if (prestigeBands.every((band) => band.destroyChance <= 0)) issues.push('missingPrestigeRiskLayer');
  if (!protectionAide) issues.push('missingDestroyProtectionAide');
  if (protectedBands.some((band) => band.finalDestroyChance > 0)) issues.push('protectedUpgradeCanStillDestroy');
  if (upgradeMaterialIds.some((materialId) => !enemyMaterialSources[materialId] && !deterministicMaterialCoverage[materialId])) issues.push('upgradeMaterialSourceMissing');
  if (maxMultiplierByTier.rare > 8 || maxMultiplierByTier.relic > 24) issues.push('earlyAttunementMultiplierTooHigh');
  return {
    equipmentItemCount: equipmentItems.length,
    shopItemCount: (data.SHOP_ITEMS || []).length,
    randomDropItemCount: (data.RANDOM_EQUIPMENT_ITEMS || []).length,
    bossItemCount: (data.BOSS_EQUIPMENT_ITEMS || []).length,
    setCount: (data.EQUIPMENT_SETS || []).length,
    deterministicWeapons,
    advancedOffhandCoverage: {
      coveredCount: advancedOffhands.filter((entry) => entry.itemCount > 0).length,
      classCount: advancedOffhands.length,
      entries: advancedOffhands
    },
    upgradeBands: {
      baselineSafeCeiling: Math.max(...baselineBands.filter((band) => band.destroyChance <= 0).map((band) => band.maxUpgrade), 0),
      unprotected: unprotectedBands,
      protected: protectedBands,
      protectionAideId: protectionAide && (protectionAide.materialId || protectionAide.id) || '',
      failureSalvageMode: 'partialUpgradeDustRefund'
    },
    materialSources: {
      enemyDropSources: enemyMaterialSources,
      deterministicRewardCoverage: deterministicMaterialCoverage
    },
    attunementMultiplierCaps: {
      lineCount: multiplierLines.length,
      maxByTier: maxMultiplierByTier
    },
    issueCount: issues.length,
    issueIds: issues
  };
}

function getCollectionCount(value) {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === 'object') return Object.keys(value).length;
  return 0;
}

function collectRewardPermanentStats(reward, totals = {}) {
  if (!reward || typeof reward !== 'object') return totals;
  if (reward.permanentStats && typeof reward.permanentStats === 'object') {
    Object.keys(reward.permanentStats).forEach((statId) => {
      totals[statId] = (totals[statId] || 0) + Number(reward.permanentStats[statId] || 0);
    });
  }
  Object.keys(reward).forEach((key) => {
    const value = reward[key];
    if (value && typeof value === 'object') collectRewardPermanentStats(value, totals);
  });
  return totals;
}

function hasRewardKey(reward, keyId) {
  if (!reward || typeof reward !== 'object') return false;
  if (Object.prototype.hasOwnProperty.call(reward, keyId)) return true;
  return Object.keys(reward).some((key) => hasRewardKey(reward[key], keyId));
}

function collectRewardConsumableIds(reward, ids = new Set()) {
  if (!reward || typeof reward !== 'object') return ids;
  if (reward.consumables && typeof reward.consumables === 'object') {
    Object.keys(reward.consumables).forEach((consumableId) => ids.add(consumableId));
  }
  Object.keys(reward).forEach((key) => {
    const value = reward[key];
    if (value && typeof value === 'object') collectRewardConsumableIds(value, ids);
  });
  return ids;
}

function estimateSeasonObjectiveMinutes(objective) {
  const type = String(objective && objective.type || '');
  const count = Math.max(1, Math.floor(Number(objective && objective.count || 1) || 1));
  const minutes = RETENTION_SEASON_OBJECTIVE_MINUTES[type] || 5;
  return count * minutes;
}

function createPlayerTypeCoverageReport(data, categorySet, laneAvailability) {
  const activeSeasonCount = (data.SEASONS || []).filter((season) => season && season.active).length;
  const coverage = {
    casual: (data.DAILY_LOGIN_REWARDS || []).length >= 7 && categorySet.has('Onboarding'),
    grinder: categorySet.has('Combat') && laneAvailability.endlessRift,
    completionist: categorySet.has('Collection') && laneAvailability.monsterGuide && laneAvailability.cards,
    bosser: categorySet.has('Boss') && categorySet.has('Dungeon') && laneAvailability.dungeons,
    economy: categorySet.has('Crafting') && laneAvailability.crafting,
    competitive: (categorySet.has('Mastery') || categorySet.has('Class')) && (laneAvailability.classMastery || activeSeasonCount > 0)
  };
  const entries = RETENTION_PLAYER_TYPES.map((typeId) => ({
    typeId,
    covered: !!coverage[typeId]
  }));
  return {
    typeCount: entries.length,
    coveredCount: entries.filter((entry) => entry.covered).length,
    missingTypeIds: entries.filter((entry) => !entry.covered).map((entry) => entry.typeId),
    entries
  };
}

function createRetentionHealthReport(data) {
  const accomplishments = data.ACCOMPLISHMENTS || [];
  const accomplishmentCategories = Array.from(new Set(accomplishments.map((item) => item && item.category).filter(Boolean))).sort();
  const accomplishmentTiers = Array.from(new Set(accomplishments.map((item) => item && item.tier).filter(Boolean))).sort();
  const categorySet = new Set(accomplishmentCategories);
  const liveMonsterGuideCount = (data.ENEMIES || []).filter((enemy) => {
    const guide = enemy && enemy.guide;
    return guide && !guide.excludedFromCollection && (!guide.visibility || guide.visibility === 'live');
  }).length;
  const activeSeasons = (data.SEASONS || []).filter((season) => season && season.active);
  const estimatedSeasonGoalMinutes = activeSeasons.reduce((sum, season) =>
    sum + (season.objectives || []).reduce((seasonSum, objective) =>
      seasonSum + estimateSeasonObjectiveMinutes(objective), 0), 0);
  const activeSeasonObjectiveCount = activeSeasons.reduce((sum, season) =>
    sum + (season.objectives || []).length, 0);
  const loginPermanentStats = (data.DAILY_LOGIN_MILESTONES || []).reduce((totals, milestone) =>
    collectRewardPermanentStats(milestone && milestone.reward, totals), {});
  const loginPermanentStatMilestoneCount = (data.DAILY_LOGIN_MILESTONES || []).filter((milestone) =>
    hasRewardKey(milestone && milestone.reward, 'permanentStats')).length;
  const rewardConsumableIds = new Set();
  []
    .concat(data.DAILY_LOGIN_REWARDS || [])
    .concat(data.DAILY_LOGIN_MILESTONES || [])
    .concat(activeSeasons)
    .concat(data.ACCOMPLISHMENTS || [])
    .forEach((entry) => collectRewardConsumableIds(entry && (entry.reward || entry.rewards || entry), rewardConsumableIds));
  const catchUpConsumableIds = Array.from(rewardConsumableIds)
    .filter((consumableId) => /coupon|slot|reset/i.test(consumableId))
    .sort();
  const cashShopItems = data.CASH_SHOP_ITEMS || [];
  const cashShopPowerItems = cashShopItems.filter((item) =>
    hasRewardKey(item && item.reward, 'permanentStats') ||
    hasRewardKey(item && item.reward, 'xp') ||
    hasRewardKey(item && item.reward, 'starTokens') ||
    hasRewardKey(item && item.reward, 'materials'));
  const cashShopEarnableBuffBundles = cashShopItems.filter((item) => {
    const tags = item && Array.isArray(item.tags) ? item.tags : [];
    return hasRewardKey(item && item.reward, 'consumables') && tags.includes('Earnable In Game');
  });
  const nonEarnableCashBuffs = cashShopItems.filter((item) => {
    const tags = item && Array.isArray(item.tags) ? item.tags : [];
    return hasRewardKey(item && item.reward, 'consumables') && !tags.includes('Earnable In Game');
  });
  const laneAvailability = {
    accomplishments: accomplishments.length > 0,
    monsterGuide: liveMonsterGuideCount > 0,
    cards: getCollectionCount(data.CARD_DEFINITIONS) > 0,
    classMastery: getCollectionCount(data.CLASS_MASTERY_TRACKS) > 0,
    rosterGrowth: getCollectionCount(data.ROSTER_TRAITS) > 0 || getCollectionCount(data.ROSTER_SYNERGIES) > 0,
    dailyLogin: (data.DAILY_LOGIN_REWARDS || []).length > 0,
    seasons: activeSeasons.length > 0,
    cosmetics: cashShopItems.some((item) => item && Array.isArray(item.tags) && item.tags.includes('Cosmetic')),
    bossSets: getCollectionCount(data.EQUIPMENT_SETS) > 0,
    dungeons: getCollectionCount(data.DUNGEONS) > 0,
    endlessRift: (data.MAPS || []).some((map) => map && map.endlessScaling),
    crafting: accomplishments.some((item) => item && item.category === 'Crafting') || getCollectionCount(data.UPGRADE_AIDES) > 0,
    subclassing: getCollectionCount(data.ADVANCED_CLASSES) > 0 && getCollectionCount(data.SPECIALIZATIONS) > 0,
    partyObjectives: getCollectionCount(data.DUNGEON_OBJECTIVES) > 0,
    targetFarms: getCollectionCount(data.TARGET_FARM_TABLES) > 0
  };
  const longTermLanes = Object.keys(laneAvailability).map((laneId) => ({
    laneId,
    available: !!laneAvailability[laneId]
  }));
  const playerTypeCoverage = createPlayerTypeCoverageReport(data, categorySet, laneAvailability);
  const activeSeasonRewardPowerRiskCount = activeSeasons.filter((season) =>
    hasRewardKey(season && season.rewards, 'permanentStats')).length;
  const maxWeeklyCashShopLimit = Math.max(0, ...cashShopItems.map((item) => Number(item && item.weeklyLimit || 0) || 0));
  const mandatoryDailyChecklistCount = 0;
  const estimatedMandatoryDailyMinutes = (data.DAILY_LOGIN_REWARDS || []).length ? 1 : 0;
  const issues = [];
  if ((data.DAILY_LOGIN_REWARDS || []).length !== 7) issues.push('dailyLoginCadenceGap');
  if (mandatoryDailyChecklistCount > 3 || estimatedMandatoryDailyMinutes > 30) issues.push('mandatoryDailyChoreRisk');
  if (activeSeasons.length && (estimatedSeasonGoalMinutes < 45 || estimatedSeasonGoalMinutes > 100)) issues.push('seasonGoalTimeOutsideWeeklyTarget');
  if (longTermLanes.filter((lane) => lane.available).length < 10) issues.push('longTermLaneCoverageGap');
  if (playerTypeCoverage.missingTypeIds.length) issues.push('playerTypeCoverageGap');
  if (accomplishmentCategories.length < 8 || accomplishmentTiers.length < 5) issues.push('accomplishmentBreadthGap');
  if (liveMonsterGuideCount < 30 || getCollectionCount(data.CARD_DEFINITIONS) < 15) issues.push('collectionLaneTooThin');
  if (loginPermanentStatMilestoneCount > 3 || Math.max(...Object.values(loginPermanentStats).map((value) => Number(value) || 0), 0) > 50) issues.push('loginPermanentPowerTooHigh');
  if (cashShopPowerItems.length || nonEarnableCashBuffs.length || maxWeeklyCashShopLimit > 3) issues.push('cashShopPowerOrChoreRisk');
  if (activeSeasonRewardPowerRiskCount) issues.push('seasonPermanentPowerRisk');
  return {
    dailyLoginRewardCount: (data.DAILY_LOGIN_REWARDS || []).length,
    dailyLoginMilestoneCount: (data.DAILY_LOGIN_MILESTONES || []).length,
    mandatoryDailyChecklistCount,
    estimatedMandatoryDailyMinutes,
    activeSeasonCount: activeSeasons.length,
    activeSeasonObjectiveCount,
    estimatedSeasonGoalMinutes,
    accomplishmentCount: accomplishments.length,
    accomplishmentCategoryCount: accomplishmentCategories.length,
    accomplishmentTierCount: accomplishmentTiers.length,
    accomplishmentCategories,
    accomplishmentTiers,
    liveMonsterGuideCount,
    cardCount: getCollectionCount(data.CARD_DEFINITIONS),
    classMasteryTrackCount: getCollectionCount(data.CLASS_MASTERY_TRACKS),
    rosterTraitCount: getCollectionCount(data.ROSTER_TRAITS),
    rosterSynergyCount: getCollectionCount(data.ROSTER_SYNERGIES),
    dungeonCount: getCollectionCount(data.DUNGEONS),
    dungeonObjectiveCount: getCollectionCount(data.DUNGEON_OBJECTIVES),
    equipmentSetCount: getCollectionCount(data.EQUIPMENT_SETS),
    longTermLaneCount: longTermLanes.filter((lane) => lane.available).length,
    longTermLanes,
    playerTypeCoverage,
    catchUpSourceCount: catchUpConsumableIds.length,
    catchUpConsumableIds,
    loginPermanentStatMilestoneCount,
    loginPermanentStats,
    cashShop: {
      itemCount: cashShopItems.length,
      cosmeticItemCount: cashShopItems.filter((item) => item && Array.isArray(item.tags) && item.tags.includes('Cosmetic')).length,
      powerItemCount: cashShopPowerItems.length,
      earnableBuffBundleCount: cashShopEarnableBuffBundles.length,
      nonEarnableBuffBundleCount: nonEarnableCashBuffs.length,
      maxWeeklyLimit: maxWeeklyCashShopLimit
    },
    seasonPowerRewardCount: activeSeasonRewardPowerRiskCount,
    issueCount: issues.length,
    issueIds: issues
  };
}

function collectRewardCurrencyTotals(reward, totals = { currency: 0, starTokens: 0 }) {
  if (!reward || typeof reward !== 'object') return totals;
  totals.currency += Math.max(0, Number(reward.currency || 0));
  totals.starTokens += Math.max(0, Number(reward.starTokens || 0));
  Object.keys(reward).forEach((key) => {
    const value = reward[key];
    if (value && typeof value === 'object') collectRewardCurrencyTotals(value, totals);
  });
  return totals;
}

function getShopEntryCost(data, entry) {
  const kind = String(entry && entry.kind || entry && entry.type || '');
  if (kind === 'equipment') {
    const item = (data.SHOP_ITEMS || []).find((candidate) => candidate && candidate.id === entry.itemId);
    return Math.max(0, Number(entry.cost || item && item.cost || 0));
  }
  return Math.max(0, Number(entry && entry.cost || 0));
}

function getEconomyInventorySectionCost(tabId, currentSections = 1) {
  const sectionSize = 36;
  const growth = 1.16;
  const tab = String(tabId || 'usable');
  const tabMultiplier = tab === 'equipment' ? 1.15 : tab === 'usable' ? 1 : tab === 'cards' ? 1.05 : 0.9;
  const currentSlots = Math.max(1, Math.floor(Number(currentSections || 1) || 1)) * sectionSize;
  let cost = 0;
  for (let offset = 1; offset <= sectionSize; offset += 1) {
    const slotIndex = Math.max(0, currentSlots + offset - sectionSize - 1);
    cost += Math.max(1, Math.round(tabMultiplier * Math.pow(growth, slotIndex)));
  }
  return cost;
}

function addEconomySinkEntry(sinks, typeId, cost, options = {}) {
  const value = Math.max(0, Number(cost || 0));
  if (value <= 0) return;
  sinks.push({
    typeId,
    id: String(options.id || ''),
    cost: value,
    repeatable: options.repeatable !== false
  });
}

function createEconomySinkCatalog(data) {
  const sinks = [];
  (data.SHOP_VENDOR_CATALOGS || []).forEach((catalog) => {
    (catalog.entries || catalog.items || []).forEach((entry) => {
      const kind = String(entry && entry.kind || entry && entry.type || '');
      const typeId = kind === 'equipment'
        ? 'vendorEquipment'
        : kind === 'consumable'
          ? 'vendorConsumable'
          : kind === 'bundle'
            ? 'vendorBundle'
            : '';
      if (!typeId) return;
      addEconomySinkEntry(sinks, typeId, getShopEntryCost(data, entry), {
        id: `${catalog.id || 'vendor'}:${entry.itemId || entry.consumableId || entry.id || typeId}`
      });
    });
  });
  (data.MARKET_LISTINGS || []).forEach((listing) => {
    addEconomySinkEntry(sinks, 'market', listing && listing.cost, {
      id: listing && listing.id,
      repeatable: !(listing && listing.once)
    });
  });
  (data.PLINKO_BALLS || []).forEach((ball) => {
    addEconomySinkEntry(sinks, 'plinko', ball && ball.cost, { id: ball && ball.id });
  });
  (data.COSMETICS || []).forEach((cosmetic) => {
    if (cosmetic && (cosmetic.cashShopOnly || cosmetic.seasonReward || cosmetic.dailyReward)) return;
    addEconomySinkEntry(sinks, 'cosmetic', cosmetic && cosmetic.cost, { id: cosmetic && cosmetic.id });
  });
  ['equipment', 'usable', 'etc', 'cards'].forEach((tabId) => {
    addEconomySinkEntry(sinks, 'inventorySlots', getEconomyInventorySectionCost(tabId, 1), { id: `${tabId}:section2` });
  });
  const costs = sinks.map((sink) => Number(sink.cost || 0)).filter((cost) => cost > 0);
  const repeatableCosts = sinks.filter((sink) => sink.repeatable).map((sink) => Number(sink.cost || 0)).filter((cost) => cost > 0);
  const byType = ECONOMY_COIN_SINK_TYPES.reduce((result, typeId) => {
    const entries = sinks.filter((sink) => sink.typeId === typeId);
    result[typeId] = {
      count: entries.length,
      repeatableCount: entries.filter((entry) => entry.repeatable).length,
      minCost: entries.length ? Math.min(...entries.map((entry) => entry.cost)) : 0,
      maxCost: entries.length ? Math.max(...entries.map((entry) => entry.cost)) : 0
    };
    return result;
  }, {});
  return {
    sinkCount: sinks.length,
    sinkTypeCount: Object.values(byType).filter((entry) => entry.count > 0).length,
    repeatableSinkCount: sinks.filter((sink) => sink.repeatable).length,
    minCost: costs.length ? Math.min(...costs) : 0,
    medianCost: roundNumber(medianNumber(costs), 1),
    maxCost: costs.length ? Math.max(...costs) : 0,
    medianRepeatableCost: roundNumber(medianNumber(repeatableCosts), 1),
    byType,
    sinks
  };
}

function getEconomyItemPurposeIds(data, item, typeId) {
  const purposes = new Set();
  const id = String(item && item.id || '');
  if (!id) return [];
  if (typeId === 'equipment') {
    purposes.add('equip');
    purposes.add('sell');
  }
  if (typeId === 'consumable') {
    purposes.add('use');
    if (Number(item.cost || 0) > 0) purposes.add('buy');
    if (item && item.plinkoBall) purposes.add('rewardRoll');
  }
  if (typeId === 'card') {
    purposes.add('collect');
    purposes.add('build');
  }
  if (typeId === 'cosmetic') {
    purposes.add('cosmetic');
    if (Number(item.cost || 0) > 0 && !item.cashShopOnly) purposes.add('coinSink');
  }
  if (typeId === 'material') {
    if (ECONOMY_UPGRADE_MATERIAL_IDS.includes(id)) purposes.add('upgrade');
    if (/StarCard$/.test(id)) purposes.add('cardCraft');
    if (item.primaryDrop || item.genericDrop || (item.dropLabels || []).length) purposes.add('progress');
    const usedByAide = (data.UPGRADE_AIDES || []).some((aide) => aide && (aide.materialId === id || aide.id === id));
    if (usedByAide) purposes.add('upgradeAide');
  }
  return Array.from(purposes).sort();
}

function createEconomyItemPurposeReport(data) {
  const itemEntries = []
    .concat((data.MATERIAL_ITEMS || []).map((item) => ({ typeId: 'material', item })))
    .concat((data.CONSUMABLE_ITEMS || []).map((item) => ({ typeId: 'consumable', item })))
    .concat((data.SHOP_ITEMS || []).map((item) => ({ typeId: 'equipment', item })))
    .concat((data.RANDOM_EQUIPMENT_ITEMS || []).map((item) => ({ typeId: 'equipment', item })))
    .concat((data.BOSS_EQUIPMENT_ITEMS || []).map((item) => ({ typeId: 'equipment', item })))
    .concat((data.CARD_DEFINITIONS || []).map((item) => ({ typeId: 'card', item })))
    .concat((data.COSMETICS || []).map((item) => ({ typeId: 'cosmetic', item })));
  const byId = new Map();
  itemEntries.forEach((entry) => {
    const id = String(entry.item && entry.item.id || '');
    if (!id || byId.has(`${entry.typeId}:${id}`)) return;
    const purposeIds = getEconomyItemPurposeIds(data, entry.item, entry.typeId);
    byId.set(`${entry.typeId}:${id}`, {
      itemId: id,
      typeId: entry.typeId,
      purposeIds
    });
  });
  const entries = Array.from(byId.values());
  const purposeCounts = entries.reduce((counts, entry) => {
    (entry.purposeIds || []).forEach((purposeId) => {
      counts[purposeId] = (counts[purposeId] || 0) + 1;
    });
    return counts;
  }, {});
  const deadItems = entries.filter((entry) => !entry.purposeIds.length);
  return {
    itemDefinitionCount: entries.length,
    purposeCoverage: roundNumber((entries.length - deadItems.length) / Math.max(1, entries.length), 3),
    deadItemCount: deadItems.length,
    deadItemIds: deadItems.map((entry) => `${entry.typeId}:${entry.itemId}`),
    purposeCounts,
    typeCounts: entries.reduce((counts, entry) => {
      counts[entry.typeId] = (counts[entry.typeId] || 0) + 1;
      return counts;
    }, {})
  };
}

function createEconomyHealthReport(data, fieldReport) {
  const fieldMaps = fieldReport && Array.isArray(fieldReport.maps) ? fieldReport.maps : [];
  const mapCurrencyRates = fieldMaps.map((map) =>
    medianNumber((map.results || []).map((result) => Number(result.currencyPerHour || 0)))).filter((value) => value > 0);
  const rewardCurrency = { currency: 0, starTokens: 0 };
  []
    .concat(data.QUESTS || [])
    .concat(data.ACCOMPLISHMENTS || [])
    .concat(data.DAILY_LOGIN_REWARDS || [])
    .concat(data.DAILY_LOGIN_MILESTONES || [])
    .concat(data.SEASONS || [])
    .forEach((entry) => collectRewardCurrencyTotals(entry && (entry.reward || entry.rewards || entry), rewardCurrency));
  const sinkCatalog = createEconomySinkCatalog(data);
  const itemPurpose = createEconomyItemPurposeReport(data);
  const medianFieldCurrencyPerHour = roundNumber(medianNumber(mapCurrencyRates), 1);
  const medianRepeatableSinkMinutes = medianFieldCurrencyPerHour > 0
    ? roundNumber(sinkCatalog.medianRepeatableCost / medianFieldCurrencyPerHour * 60, 1)
    : 0;
  const marketListings = data.MARKET_LISTINGS || [];
  const cashShopPowerItems = (data.CASH_SHOP_ITEMS || []).filter((item) =>
    hasRewardKey(item && item.reward, 'permanentStats') ||
    hasRewardKey(item && item.reward, 'xp') ||
    hasRewardKey(item && item.reward, 'starTokens') ||
    hasRewardKey(item && item.reward, 'materials'));
  const issues = [];
  if (medianFieldCurrencyPerHour <= 0) issues.push('missingFieldCurrencyFaucet');
  if (sinkCatalog.sinkTypeCount < 6 || sinkCatalog.repeatableSinkCount < 20) issues.push('sinkCoverageGap');
  if (medianRepeatableSinkMinutes <= 0 || medianRepeatableSinkMinutes > 90) issues.push('repeatableSinkPacingGap');
  if (itemPurpose.deadItemCount > 0 || itemPurpose.purposeCoverage < 0.995) issues.push('deadLootDefinitions');
  if (marketListings.length < 10 || !marketListings.some((listing) => !listing.once) || !marketListings.some((listing) => listing.once)) issues.push('marketCatalogThin');
  if (cashShopPowerItems.length) issues.push('cashShopPowerEconomyRisk');
  return {
    fieldFaucets: {
      mapCount: fieldMaps.length,
      medianCurrencyPerHour: medianFieldCurrencyPerHour,
      maxCurrencyPerHour: roundNumber(Math.max(...mapCurrencyRates, 0), 1),
      minCurrencyPerHour: mapCurrencyRates.length ? roundNumber(Math.min(...mapCurrencyRates), 1) : 0
    },
    deterministicCurrencyRewards: {
      currency: Math.round(rewardCurrency.currency),
      starTokens: Math.round(rewardCurrency.starTokens)
    },
    sinks: sinkCatalog,
    itemPurpose,
    market: {
      listingCount: marketListings.length,
      repeatableListingCount: marketListings.filter((listing) => !(listing && listing.once)).length,
      onceListingCount: marketListings.filter((listing) => listing && listing.once).length,
      playerTradingEnabled: false,
      auctionTaxRequiredBeforeTrading: true
    },
    cashShopPowerItemCount: cashShopPowerItems.length,
    medianRepeatableSinkMinutes,
    telemetryFields: ['currencyGained', 'currencySpent', 'currencySinks', 'currencyPerHour', 'currencySpentPerHour', 'netCurrencyPerHour'],
    issueCount: issues.length,
    issueIds: issues
  };
}

function createBalanceReport(data, createProjectStarfallEngine, options = {}) {
  const classIds = Array.isArray(options.classIds) && options.classIds.length ? options.classIds : ALL_CLASS_IDS;
  const scenarios = {};
  BALANCE_SCENARIOS.forEach((scenario) => {
    const results = classIds
      .map((classId) => runBalanceScenario(data, createProjectStarfallEngine, classId, scenario, options))
      .sort((a, b) => b.dps - a.dps);
    scenarios[scenario.id] = {
      id: scenario.id,
      label: scenario.label,
      duration: scenario.duration,
      results
    };
  });
  const field = createMapBalanceReport(data, createProjectStarfallEngine, Object.assign({}, options, { classIds }));
  return {
    level: Math.max(1, Math.floor(Number(options.level || 50) || 50)),
    rank: Math.max(1, Math.floor(Number(options.rank || 10) || 10)),
    assumptions: [
      'Equal level, equal skill ranks, no gear, deterministic average damage rolls.',
      'Berserker boss scenarios start at 46% HP to measure its intended risk window.',
      'Rotations prefer class identity and add inherited filler only when an advanced kit has sparse active coverage.',
      'Low-cooldown trainer AoE, explosive projectiles, and traps get limited cleave value so setup-free spam does not model as full-pack damage every cast.'
    ],
    scenarios,
    field,
    skillSystem: createSkillSystemHealthReport(data),
    enemyEcosystem: createEnemyEcosystemHealthReport(data),
    damageStatFormula: createDamageStatFormulaReport(data, createProjectStarfallEngine),
    economy: createEconomyHealthReport(data, field),
    bossParty: createBossPartyReport(data, createProjectStarfallEngine, options),
    equipment: createEquipmentUpgradeReport(data),
    retention: createRetentionHealthReport(data)
  };
}

function getScenarioResults(report, scenarioId) {
  return report && report.scenarios && report.scenarios[scenarioId]
    ? report.scenarios[scenarioId].results
    : [];
}

function getClassResult(report, scenarioId, classId) {
  return getScenarioResults(report, scenarioId).find((result) => result.classId === classId) || null;
}

function getMapResults(report, mapId) {
  const mapReport = report && report.field && Array.isArray(report.field.maps)
    ? report.field.maps.find((map) => map.id === mapId)
    : null;
  return mapReport ? mapReport.results : [];
}

function getMapClassResult(report, mapId, classId) {
  return getMapResults(report, mapId).find((result) => result.classId === classId) || null;
}

module.exports = {
  ADVANCED_CLASS_IDS,
  ALL_CLASS_IDS,
  ADVANCED_BASE_FILLERS,
  BALANCE_SCENARIOS,
  BOSS_MECHANIC_CATEGORIES,
  CLASS_ROTATIONS,
  FIELD_PROGRESSION_BRACKETS,
  createBalanceReport,
  createBossPartyReport,
  createDamageStatFormulaReport,
  createEnemyEcosystemHealthReport,
  createEconomyHealthReport,
  createEquipmentUpgradeReport,
  createLevelCurveReport,
  createMapBalanceReport,
  createMapTuningReport,
  createRetentionHealthReport,
  createSkillSystemHealthReport,
  getClassRotation,
  getClassResult,
  getBossHpScalingMultiplier,
  getMapClassResult,
  getMapProfile,
  getMapResults,
  getScenarioResults,
  getTargetLevelMinutes,
  estimateClassScenarioDps,
  runBalanceScenario
};
