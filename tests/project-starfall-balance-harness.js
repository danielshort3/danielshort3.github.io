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

const CLASS_SCENARIO_MULTIPLIERS = Object.freeze({
  fighter: Object.freeze({ boss: 0.92, clustered: 0.95, spread: 0.75, armored: 1.08, flying: 0.52, sustain: 0.9 }),
  mage: Object.freeze({ boss: 0.94, clustered: 1.12, spread: 0.95, armored: 0.92, flying: 1.12, sustain: 0.82 }),
  archer: Object.freeze({ boss: 1.02, clustered: 0.94, spread: 1.12, armored: 0.96, flying: 1.22, sustain: 0.88 }),
  guardian: Object.freeze({ boss: 0.86, clustered: 0.86, spread: 0.72, armored: 1.18, flying: 0.48, sustain: 1.35 }),
  berserker: Object.freeze({ boss: 1.18, clustered: 0.84, spread: 0.72, armored: 1.06, flying: 0.44, sustain: 1.02 }),
  duelist: Object.freeze({ boss: 1.22, clustered: 0.68, spread: 0.62, armored: 0.98, flying: 0.42, sustain: 0.82 }),
  fireMage: Object.freeze({ boss: 0.9, clustered: 1.34, spread: 1.08, armored: 0.96, flying: 1.12, sustain: 0.82 }),
  runeMage: Object.freeze({ boss: 0.96, clustered: 1.16, spread: 1.12, armored: 1.1, flying: 1.02, sustain: 1.0 }),
  stormMage: Object.freeze({ boss: 0.72, clustered: 1.28, spread: 1.18, armored: 0.84, flying: 1.28, sustain: 0.76 }),
  sniper: Object.freeze({ boss: 1.34, clustered: 0.72, spread: 0.9, armored: 1.2, flying: 1.32, sustain: 0.84 }),
  trapper: Object.freeze({ boss: 0.84, clustered: 1.22, spread: 0.86, armored: 1.22, flying: 0.58, sustain: 1.08 }),
  beastArcher: Object.freeze({ boss: 0.94, clustered: 0.86, spread: 1.0, armored: 0.88, flying: 1.14, sustain: 1.24 })
});

function getClassBaseId(data, classId) {
  const advanced = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId];
  if (advanced && advanced.baseClass) return advanced.baseClass;
  return data.BASE_CLASSES && data.BASE_CLASSES[classId] ? classId : 'fighter';
}

function getSkill(data, skillId) {
  return (data.SKILLS || []).find((skill) => skill.id === skillId) || null;
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

function estimateSkillDps(stats, skill, rank, scenario) {
  if (!skill || skill.category === 'passive' || skill.category === 'mobility' || skill.category === 'buff') return 0;
  const cooldown = Math.max(0.35, Number(skill.cooldown || 0.6));
  const base = getEstimatedSkillBase(stats, skill, rank) * Math.max(0.35, Number(skill.lineDamageScale || 1));
  const lineTexture = 1 + Math.max(0, Number(skill.lineCount || 1) - 1) * 0.08;
  return base * lineTexture * getSkillTargetFactor(skill, scenario) * getSkillSuitability(skill, scenario) / cooldown;
}

function getClassBuffMultiplier(data, classId, scenario) {
  const profile = getScenarioProfile(scenario);
  const partySkillId = data.ADVANCED_CLASSES && data.ADVANCED_CLASSES[classId] && data.ADVANCED_CLASSES[classId].partySkillId;
  if (!partySkillId) return 1;
  if (classId === 'guardian') return profile.sustain ? 1.18 : 1.05;
  if (classId === 'berserker') return profile.boss ? 1.16 : 1.05;
  if (classId === 'duelist') return profile.boss ? 1.12 : 1.02;
  if (classId === 'fireMage') return profile.clustered ? 1.14 : 1.04;
  if (classId === 'runeMage') return profile.spread || profile.armored ? 1.1 : 1.06;
  if (classId === 'stormMage') return profile.clustered || profile.flying ? 1.12 : 1.02;
  if (classId === 'sniper') return profile.boss || profile.armored ? 1.14 : 1.03;
  if (classId === 'trapper') return profile.clustered || profile.armored ? 1.13 : 1.04;
  if (classId === 'beastArcher') return profile.sustain ? 1.17 : 1.06;
  return 1;
}

function estimateClassScenarioDps(data, classId, scenario, stats, rank) {
  const rotation = CLASS_ROTATIONS[classId] || [];
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
  (CLASS_ROTATIONS[classId] || []).forEach((skillId) => {
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
  return {
    level: Math.max(1, Math.floor(Number(options.level || 50) || 50)),
    rank: Math.max(1, Math.floor(Number(options.rank || 10) || 10)),
    assumptions: [
      'Equal level, equal skill ranks, no gear, deterministic average damage rolls.',
      'Berserker boss scenarios start at 46% HP to measure its intended risk window.',
      'Rotations prefer each class identity instead of using every inherited base skill.'
    ],
    scenarios
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

module.exports = {
  ADVANCED_CLASS_IDS,
  ALL_CLASS_IDS,
  BALANCE_SCENARIOS,
  CLASS_ROTATIONS,
  createBalanceReport,
  getClassResult,
  getScenarioResults,
  runBalanceScenario
};
