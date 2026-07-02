(function initProjectStarfallEngineCombatFormulas(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  const DAMAGE_FLOOR_BASE_PERCENT = 50;
  const DAMAGE_FLOOR_MAX_PERCENT = 90;
  const PASSIVE_HP_REGEN_RATE = 0.0015;
  const PASSIVE_MP_REGEN_RATE = 0.003;
  const ATTUNEMENT_DAMAGE_BONUS_CAP = 100;
  const ATTUNEMENT_ARMOR_BREAK_DAMAGE_CAP = 50;
  const ATTUNEMENT_ARMOR_BREAK_GAUGE_CAP = 75;
  const ATTUNEMENT_RESOURCE_COST_REDUCTION_CAP = 35;
  const ATTUNEMENT_COOLDOWN_RECOVERY_CAP = 35;
  const ATTUNEMENT_SHIELD_STRENGTH_CAP = 75;
  const ATTUNEMENT_BUFF_DURATION_CAP = 60;
  const ATTUNEMENT_DAMAGE_REDUCTION_CAP = 25;
  const ATTUNEMENT_POTION_EFFECT_CAP = 75;
  const ATTUNEMENT_TRAP_ARM_SPEED_CAP = 60;
  const ATTUNEMENT_MOBILITY_WINDOW_CAP = 75;

  function getDamageFloorPercent(bonus) {
    return clamp(DAMAGE_FLOOR_BASE_PERCENT + Math.max(0, Number(bonus || 0)), DAMAGE_FLOOR_BASE_PERCENT, DAMAGE_FLOOR_MAX_PERCENT);
  }

  function makeDamageRange(power, floorPercent) {
    const max = Math.max(1, Math.round(Number(power) || 1));
    const floor = getDamageFloorPercent(Number(floorPercent || DAMAGE_FLOOR_BASE_PERCENT) - DAMAGE_FLOOR_BASE_PERCENT);
    return {
      min: Math.max(1, Math.round(max * floor / 100)),
      max,
      floorPercent: floor
    };
  }

  function getMonsterDamage(level, enemyData) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    const damageMult = Number(enemyData && enemyData.damageMult || 1) || 1;
    const baseDamage = 12 + 1.35 * normalizedLevel + 0.018 * normalizedLevel * normalizedLevel;
    return Math.max(1, Math.round(baseDamage * damageMult));
  }

  function mitigatePlayerDamage(amount, defense) {
    const raw = Math.max(0, Number(amount || 0));
    const rating = Math.max(0, Number(defense || 0));
    return Math.max(1, Math.round(raw * (100 / (100 + rating * 2.25))));
  }

  function createDamageRollResult(amount, options) {
    const settings = options || {};
    const stats = settings.stats || {};
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const critChance = clamp(Number(settings.critChance != null ? settings.critChance : stats.crit || 0), 0, 70);
    const critical = settings.critical != null ? !!settings.critical : critChance > 0 && random() * 100 < critChance;
    const critBonus = critical ? 1.5 + Number(stats.critDamage || 0) / 100 : 1;
    const attackDamageBonus = 1 + Math.max(0, Number(stats.attackDamagePercent || 0)) / 100;
    const markBonus = Number(settings.markBonus || 1) || 1;
    const roleBonus = Number(settings.roleBonus || 1) || 1;
    const bossDamageBonus = Number(settings.bossDamageBonus || 1) || 1;
    const eliteDamageBonus = Number(settings.eliteDamageBonus || 1) || 1;
    const executeDamageBonus = Number(settings.executeDamageBonus || 1) || 1;
    const targetDefense = Number(settings.targetDefense || 0) || 0;
    const monsterMastery = Number(settings.monsterMastery || 0) || 0;
    const baseMaxDamage = Math.max(1, Math.round((Number(amount) || 1) * markBonus * roleBonus * critBonus * attackDamageBonus * bossDamageBonus * eliteDamageBonus * executeDamageBonus - targetDefense));
    const maxDamage = Math.max(1, Math.round(baseMaxDamage * (1 + monsterMastery)));
    const floorPercent = clamp(Number(stats.damageRange && stats.damageRange.floorPercent || stats.damageFloor || DAMAGE_FLOOR_BASE_PERCENT), DAMAGE_FLOOR_BASE_PERCENT, DAMAGE_FLOOR_MAX_PERCENT);
    const minDamage = maxDamage * floorPercent / 100;
    return {
      amount: Math.max(1, Math.round(minDamage + (maxDamage - minDamage) * random())),
      critical
    };
  }

  function getAttunementBonusScale(stats, key, cap) {
    const value = Math.max(0, Number(stats && stats[key] || 0));
    return 1 + clamp(value, 0, Math.max(0, Number(cap || 100))) / 100;
  }

  function getAttunementReductionScale(stats, key, cap) {
    const value = Math.max(0, Number(stats && stats[key] || 0));
    return 1 - clamp(value, 0, Math.max(0, Number(cap || 40))) / 100;
  }

  function getScaledShieldAmount(amount, stats) {
    return Math.max(0, Math.round(Math.max(0, Number(amount || 0)) * getAttunementBonusScale(stats, 'shieldStrengthPercent', ATTUNEMENT_SHIELD_STRENGTH_CAP)));
  }

  function getSkillResourceCost(skill, rank, modifier, stats) {
    const baseCost = Math.max(0, Number(skill && skill.resourceCost || 0));
    const rankScale = 1 - Math.min(Math.max(0, Number(rank || 0)), 10) * 0.015;
    const modifierScale = Number(modifier && modifier.resourceCostScale || 1);
    return Math.max(0, Math.round(baseCost * rankScale * modifierScale * getAttunementReductionScale(stats, 'resourceCostReductionPercent', ATTUNEMENT_RESOURCE_COST_REDUCTION_CAP)));
  }

  function getSkillCooldownDuration(skill, modifier, stats, options) {
    const settings = options || {};
    const baseCooldown = Number(skill && skill.cooldown || 0.6);
    const modifierScale = Number(modifier && modifier.cooldownScale || 1);
    const cooldownKey = settings.mobility ? 'mobilityCooldownPercent' : 'cooldownRecoveryPercent';
    const cooldownScale = getAttunementReductionScale(stats, cooldownKey, ATTUNEMENT_COOLDOWN_RECOVERY_CAP);
    return Math.max(0.25, baseCooldown * modifierScale * cooldownScale);
  }

  function addStatDuration(baseDuration, stats, key) {
    return Math.max(0.1, Number(baseDuration || 0) + Math.max(0, Number(stats && stats[key] || 0)));
  }

  function scaleBuffDuration(baseDuration, stats) {
    return Math.max(0.2, Number(baseDuration || 0) * getAttunementBonusScale(stats, 'buffDurationPercent', ATTUNEMENT_BUFF_DURATION_CAP));
  }

  function getPassiveRecoveryTickAmounts(stats, interval) {
    const sourceStats = stats || {};
    const hpRecoveryScale = 1 + Math.max(0, Number(sourceStats.hpRecoveryPercent || 0)) / 100;
    const mpRecoveryScale = 1 + Math.max(0, Number(sourceStats.mpRecoveryPercent || 0)) / 100;
    return {
      hp: Math.max(1, sourceStats.maxHp * PASSIVE_HP_REGEN_RATE * hpRecoveryScale) * interval,
      mp: Math.max(1, sourceStats.maxMp * PASSIVE_MP_REGEN_RATE * mpRecoveryScale) * interval
    };
  }

  function getPlayerOnHitRecoveryAmounts(stats, options) {
    const settings = options || {};
    if (Number(settings.lineCount || 1) > 1 && Number(settings.lineIndex || 0) > 0) return { hp: 0, mp: 0 };
    return {
      hp: Math.max(0, Number(stats && stats.hpOnHit || 0)),
      mp: Math.max(0, Number(stats && stats.mpOnHit || 0))
    };
  }

  function getLevelXp(level) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    return Math.round(360 + 130 * normalizedLevel + 18 * normalizedLevel * normalizedLevel + 0.16 * normalizedLevel * normalizedLevel * normalizedLevel);
  }

  function getMonsterBaseHp(level) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    return 48 + 15 * normalizedLevel + 1.65 * Math.pow(normalizedLevel, 1.35);
  }

  function getMonsterHp(level, enemyData) {
    const hpMult = Number(enemyData && enemyData.hpMult || 1) || 1;
    return Math.max(1, Math.round(getMonsterBaseHp(level) * hpMult));
  }

  function getMonsterDefense(level, enemyData) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    const defenseMult = Number(enemyData && enemyData.defenseMult || 1) || 1;
    return Math.max(0, Math.round((3 + normalizedLevel * 0.55) * defenseMult));
  }

  function getMonsterXp(level, enemyData) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    const expMult = Number(enemyData && enemyData.expMult || 1) || 1;
    return Math.max(1, Math.round((24 + 8 * normalizedLevel + Math.pow(normalizedLevel, 1.2)) * expMult));
  }

  const api = {
    DAMAGE_FLOOR_BASE_PERCENT,
    DAMAGE_FLOOR_MAX_PERCENT,
    PASSIVE_HP_REGEN_RATE,
    PASSIVE_MP_REGEN_RATE,
    ATTUNEMENT_DAMAGE_BONUS_CAP,
    ATTUNEMENT_ARMOR_BREAK_DAMAGE_CAP,
    ATTUNEMENT_ARMOR_BREAK_GAUGE_CAP,
    ATTUNEMENT_RESOURCE_COST_REDUCTION_CAP,
    ATTUNEMENT_COOLDOWN_RECOVERY_CAP,
    ATTUNEMENT_SHIELD_STRENGTH_CAP,
    ATTUNEMENT_BUFF_DURATION_CAP,
    ATTUNEMENT_DAMAGE_REDUCTION_CAP,
    ATTUNEMENT_POTION_EFFECT_CAP,
    ATTUNEMENT_TRAP_ARM_SPEED_CAP,
    ATTUNEMENT_MOBILITY_WINDOW_CAP,
    getDamageFloorPercent,
    makeDamageRange,
    getMonsterDamage,
    mitigatePlayerDamage,
    createDamageRollResult,
    getAttunementBonusScale,
    getAttunementReductionScale,
    getScaledShieldAmount,
    getSkillResourceCost,
    getSkillCooldownDuration,
    addStatDuration,
    scaleBuffDuration,
    getPassiveRecoveryTickAmounts,
    getPlayerOnHitRecoveryAmounts,
    getLevelXp,
    getMonsterBaseHp,
    getMonsterHp,
    getMonsterDefense,
    getMonsterXp
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.combatFormulas = Object.assign({}, modules.combatFormulas || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
