(function initProjectStarfallEngineStats(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function createStatsObjectSignature(source) {
    if (!source || typeof source !== 'object') return '';
    const keys = Object.keys(source).filter((key) => Number(source[key] || 0) !== 0).sort();
    return keys.map((key) => `${normalizeId(key)}=${Number(source[key] || 0)}`).join(',');
  }

  function createStatsPotentialSignature(potential, options) {
    const settings = options || {};
    const normalizeLineCount = typeof settings.normalizePotentialLineCount === 'function'
      ? settings.normalizePotentialLineCount
      : (value) => Math.max(1, Math.floor(Number(value || 1) || 1));
    if (!potential || typeof potential !== 'object') return '';
    const lines = Array.isArray(potential.lines)
      ? potential.lines.map((line) => line ? `${normalizeId(line.stat)}:${Number(line.value || 0)}` : '').filter(Boolean).join(',')
      : '';
    return [
      normalizeId(potential.tier),
      normalizeLineCount(potential.lineCount || (Array.isArray(potential.lines) ? potential.lines.length : 1)),
      lines
    ].join('/');
  }

  function createStatsItemSignature(item, options) {
    const settings = options || {};
    const getObjectSignature = typeof settings.getStatsObjectSignature === 'function'
      ? settings.getStatsObjectSignature
      : createStatsObjectSignature;
    const getPotentialSignature = typeof settings.getStatsPotentialSignature === 'function'
      ? settings.getStatsPotentialSignature
      : (potential) => createStatsPotentialSignature(potential, settings);
    if (!item || typeof item !== 'object') return '';
    return [
      normalizeId(item.uid),
      normalizeId(item.id),
      normalizeId(item.slot),
      normalizeId(item.rarity),
      Number(item.level || 0),
      Number(item.upgrade || 0),
      getObjectSignature(item.stats),
      getObjectSignature(item.baseStats),
      getObjectSignature(item.statRoll),
      getPotentialSignature(item.potential)
    ].join('|');
  }

  function createStatsEquipmentSignature(equipment, options) {
    const settings = options || {};
    const getItemSignature = typeof settings.getStatsItemSignature === 'function'
      ? settings.getStatsItemSignature
      : (item) => createStatsItemSignature(item, settings);
    const source = equipment && typeof equipment === 'object' ? equipment : {};
    return Object.keys(source)
      .sort()
      .map((slot) => `${normalizeId(slot)}:${getItemSignature(source[slot])}`)
      .join(';');
  }

  function createStatsBuffSignature(player, nowSeconds, buffIds) {
    const activePlayer = player || {};
    const buffs = activePlayer.buffs && typeof activePlayer.buffs === 'object' ? activePlayer.buffs : {};
    const time = Number(nowSeconds || 0);
    const ids = Array.isArray(buffIds) ? buffIds : [];
    let buffMask = 0;
    for (let index = 0; index < ids.length; index += 1) {
      if (Number(buffs[ids[index]] || 0) > time) buffMask += 2 ** index;
    }
    return buffMask;
  }

  function createStatsFrameSignature(frameId, buffSignature) {
    const id = Math.max(0, Math.floor(Number(frameId || 0) || 0));
    if (!id) return '';
    const buffMask = typeof buffSignature === 'function' ? buffSignature() : buffSignature;
    return `${id}:${Number(buffMask || 0)}`;
  }

  function createStatsVitalSignature(state, options) {
    const settings = options || {};
    const source = state && typeof state === 'object' ? state : {};
    const player = source.player || {};
    const getObjectSignature = typeof settings.getStatsObjectSignature === 'function'
      ? settings.getStatsObjectSignature
      : createStatsObjectSignature;
    const getEquipmentSignature = typeof settings.getStatsEquipmentSignature === 'function'
      ? settings.getStatsEquipmentSignature
      : () => createStatsEquipmentSignature(source.equipment, settings);
    const buffSignature = typeof settings.getStatsBuffSignature === 'function'
      ? settings.getStatsBuffSignature()
      : settings.buffSignature;
    return [
      Math.max(0, Number(settings.statsRevision || 0)),
      Number(buffSignature || 0),
      player.classId || '',
      player.advancedClassId || '',
      Math.max(1, Number(player.level || 1) || 1),
      getObjectSignature(source.skills),
      getObjectSignature(source.statUpgrades && source.statUpgrades.allocations),
      getEquipmentSignature()
    ].join(':');
  }

  function createStatsLevelBreakdown(playerLevel) {
    const levelBonus = Math.max(0, Number(playerLevel || 1) - 1);
    return {
      hp: levelBonus * 10,
      mpMax: levelBonus * 4,
      resourceMax: Math.floor(levelBonus / 2),
      power: Math.round(levelBonus * 2.1),
      defense: Math.floor(levelBonus * 0.65)
    };
  }

  function createStatsBaseBreakdown(baseStats, baseCritChance, options) {
    const settings = options || {};
    const cloneStats = typeof settings.cloneStats === 'function'
      ? settings.cloneStats
      : (stats) => Object.assign({}, stats || {});
    const base = cloneStats(baseStats || {});
    base.crit = Number(base.crit || 0) + Number(baseCritChance || 0);
    return base;
  }

  function createStatsBuffBreakdown(options) {
    const settings = options || {};
    const hasBuff = typeof settings.hasBuff === 'function' ? settings.hasBuff : () => false;
    const scaleBuffStatAmount = typeof settings.scaleBuffStatAmount === 'function'
      ? settings.scaleBuffStatAmount
      : (value) => Number(value || 0);
    const buffScale = settings.buffScale || 0;
    const buffs = {};
    const addBuffStat = (key, value) => {
      const amount = scaleBuffStatAmount(value, buffScale);
      if (amount) buffs[key] = Number(buffs[key] || 0) + amount;
    };
    if (hasBuff('warCry')) addBuffStat('powerPercent', 18);
    if (hasBuff('ignitionAura')) {
      addBuffStat('powerPercent', 12);
      addBuffStat('burnDamage', 20);
    }
    if (hasBuff('runeCircle') || hasBuff('packCall')) addBuffStat('resourceGain', 8);
    if (hasBuff('swiftstepOil')) addBuffStat('speed', 46);
    if (hasBuff('eagleEye') || hasBuff('rallyingFlourish')) addBuffStat('crit', 15);
    if (hasBuff('eagleEye')) addBuffStat('critDamage', 20);
    if (hasBuff('stormfront')) addBuffStat('areaDamage', 12);
    if (hasBuff('tacticalField')) addBuffStat('trapDamage', 15);
    if (hasBuff('shieldWall')) addBuffStat('block', 10);
    return buffs;
  }

  function createStatsSourceBundle(sources, playerLevel) {
    const source = sources && typeof sources === 'object' ? sources : {};
    const base = source.base || {};
    const gear = source.gear || {};
    const cards = source.cards || {};
    const roster = source.roster || {};
    const party = source.party || {};
    const specialization = source.specialization || {};
    const statUpgrades = source.statUpgrades || {};
    const permanent = source.permanent || {};
    const passive = source.passive || {};
    return {
      base,
      gear,
      cards,
      roster,
      party,
      specialization,
      statUpgrades,
      permanent,
      passive,
      levelBonus: Math.max(0, playerLevel - 1),
      statSources: [gear, cards, roster, party, specialization, statUpgrades, permanent, passive]
    };
  }

  function createStatsPermanentBonuses(permanent, extraSources, options) {
    const settings = options || {};
    const target = permanent && typeof permanent === 'object' ? permanent : {};
    const mergeStats = typeof settings.addStats === 'function'
      ? settings.addStats
      : (destination, source) => {
        Object.entries(source || {}).forEach(([key, amount]) => {
          destination[key] = Number(destination[key] || 0) + Number(amount || 0);
        });
        return destination;
      };
    (Array.isArray(extraSources) ? extraSources : []).forEach((source) => {
      mergeStats(target, source);
    });
    return target;
  }

  function createStatsBreakdownBuffSources(equipment, sources) {
    const gearBreakdown = equipment && typeof equipment === 'object' ? equipment : {};
    const source = sources && typeof sources === 'object' ? sources : {};
    return [
      gearBreakdown.gear,
      gearBreakdown.attunement,
      gearBreakdown.set,
      source.cards || {},
      source.roster || {},
      source.party || {},
      source.specialization || {},
      source.statUpgrades || {},
      source.permanent || {},
      source.passive || {}
    ];
  }

  function createStatsBreakdownSnapshot(totals, sources, options) {
    const settings = options || {};
    const cloneStats = typeof settings.cloneStats === 'function'
      ? settings.cloneStats
      : (stats) => Object.assign({}, stats || {});
    const source = sources && typeof sources === 'object' ? sources : {};
    const equipment = source.equipment && typeof source.equipment === 'object' ? source.equipment : {};
    return {
      totals,
      sources: {
        base: source.base,
        level: source.level,
        gear: cloneStats(equipment.gear),
        attunement: cloneStats(equipment.attunement),
        set: cloneStats(equipment.set),
        cards: source.cards,
        roster: source.roster,
        specialization: source.specialization,
        statUpgrades: source.statUpgrades,
        permanent: source.permanent,
        passive: source.passive,
        party: source.party,
        buffs: source.buffs
      }
    };
  }

  function createStatsVitalTotals(sources, levelBonus, options) {
    const settings = options || {};
    const sumStatFromSources = typeof settings.sumStatFromSources === 'function'
      ? settings.sumStatFromSources
      : (key, statSources) => (statSources || []).reduce((total, source) => total + Number(source && source[key] || 0), 0);
    const source = sources && typeof sources === 'object' ? sources : {};
    const base = source.base || {};
    const gear = source.gear || {};
    const cards = source.cards || {};
    const roster = source.roster || {};
    const party = source.party || {};
    const specialization = source.specialization || {};
    const statUpgrades = source.statUpgrades || {};
    const permanent = source.permanent || {};
    const passive = source.passive || {};
    const statSources = Array.isArray(source.statSources)
      ? source.statSources
      : [gear, cards, roster, party, specialization, statUpgrades, permanent, passive];
    const level = Math.max(0, Number(levelBonus || 0) || 0);
    const maxHpFlat = base.hp + gear.hp + Number(cards.hp || 0) + Number(roster.hp || 0) + Number(party.hp || 0) + Number(specialization.hp || 0) + Number(statUpgrades.hp || 0) + Number(permanent.hp || 0) + Number(passive.hp || 0) + level * 10;
    const maxHpPercent = sumStatFromSources('maxHpPercent', statSources);
    const maxHp = Math.max(1, Math.round(maxHpFlat * (1 + maxHpPercent / 100)));
    const maxMpFlat = (base.mpMax || 100) + Math.max(0, gear.mpMax || 0) + Math.max(0, Number(cards.mpMax || 0)) + Math.max(0, Number(roster.mpMax || 0)) + Math.max(0, Number(party.mpMax || 0)) + Math.max(0, Number(specialization.mpMax || 0)) + Math.max(0, Number(statUpgrades.mpMax || 0)) + Math.max(0, Number(permanent.mpMax || 0)) + Math.max(0, Number(passive.mpMax || 0)) + level * 4;
    const maxMpPercent = sumStatFromSources('maxMpPercent', statSources);
    const maxMp = Math.max(1, Math.round(maxMpFlat * (1 + maxMpPercent / 100)));
    const secondaryResourceMax = base.resourceMax + gear.resourceMax + Number(cards.resourceMax || 0) + Number(roster.resourceMax || 0) + Number(party.resourceMax || 0) + Number(specialization.resourceMax || 0) + Number(statUpgrades.resourceMax || 0) + Number(permanent.resourceMax || 0) + Number(passive.resourceMax || 0) + Math.floor(level / 2);
    return {
      maxHp,
      maxMp,
      secondaryResourceMax,
      maxHpPercent,
      maxMpPercent
    };
  }

  const STATS_SOURCE_TOTAL_KEYS = Object.freeze([
    'hpRecoveryPercent',
    'mpRecoveryPercent',
    'skillEffectPercent',
    'buffEffectPercent',
    'bossDamagePercent',
    'eliteDamagePercent',
    'resourceCostReductionPercent',
    'shieldStrengthPercent',
    'markDuration',
    'runeDuration',
    'weakPointDuration',
    'cooldownRecoveryPercent',
    'buffDurationPercent',
    'damageReductionPercent',
    'potionEffectPercent',
    'executeDamagePercent',
    'mobilityCooldownPercent',
    'mobilityWindowPercent',
    'trapSpeed',
    'hpOnHit',
    'mpOnHit'
  ]);

  function createStatsSourceTotals(statSources, options) {
    const settings = options || {};
    const sumStatFromSources = typeof settings.sumStatFromSources === 'function'
      ? settings.sumStatFromSources
      : (key, sources) => (sources || []).reduce((total, source) => total + Number(source && source[key] || 0), 0);
    return STATS_SOURCE_TOTAL_KEYS.reduce((totals, key) => {
      totals[key] = sumStatFromSources(key, statSources);
      return totals;
    }, {});
  }

  function createStatsCoreCombatTotals(sources, levelBonus, options) {
    const settings = options || {};
    const source = sources && typeof sources === 'object' ? sources : {};
    const base = source.base || {};
    const gear = source.gear || {};
    const cards = source.cards || {};
    const roster = source.roster || {};
    const party = source.party || {};
    const specialization = source.specialization || {};
    const statUpgrades = source.statUpgrades || {};
    const permanent = source.permanent || {};
    const passive = source.passive || {};
    const statSources = Array.isArray(source.statSources)
      ? source.statSources
      : [gear, cards, roster, party, specialization, statUpgrades, permanent, passive];
    const getBuffEffectScaleFromSources = typeof settings.getBuffEffectScaleFromSources === 'function'
      ? settings.getBuffEffectScaleFromSources
      : (activeSources) => {
        const buffPercent = (activeSources || []).reduce((total, statSource) => total + Number(statSource && statSource.buffEffectPercent || 0), 0);
        return 1 + Math.max(0, buffPercent) / 100;
      };
    const hasBuff = typeof settings.hasBuff === 'function' ? settings.hasBuff : () => false;
    const scaleBuffStatAmount = typeof settings.scaleBuffStatAmount === 'function'
      ? settings.scaleBuffStatAmount
      : (value, scale) => Math.round(Number(value || 0) * Math.max(0, Number(scale || 1) || 1));
    const getDamageFloorPercent = typeof settings.getDamageFloorPercent === 'function'
      ? settings.getDamageFloorPercent
      : (bonus) => Number(bonus || 0);
    const level = Math.max(0, Number(levelBonus || 0) || 0);
    const buffScale = getBuffEffectScaleFromSources(statSources);
    const buffPower = hasBuff('warCry') ? 1 + scaleBuffStatAmount(18, buffScale) / 100 : 1;
    const firePower = hasBuff('ignitionAura') ? 1 + scaleBuffStatAmount(12, buffScale) / 100 : 1;
    const powerFlat = base.power + gear.power + Number(cards.power || 0) + Number(roster.power || 0) + Number(party.power || 0) + Number(specialization.power || 0) + Number(statUpgrades.power || 0) + Number(permanent.power || 0) + Number(passive.power || 0) + level * 2.1;
    const powerPercent = Number(gear.powerPercent || 0) + Number(cards.powerPercent || 0) + Number(roster.powerPercent || 0) + Number(party.powerPercent || 0) + Number(specialization.powerPercent || 0) + Number(statUpgrades.powerPercent || 0) + Number(permanent.powerPercent || 0) + Number(passive.powerPercent || 0);
    const attackDamagePercent = Number(gear.attackDamagePercent || 0) + Number(cards.attackDamagePercent || 0) + Number(roster.attackDamagePercent || 0) + Number(party.attackDamagePercent || 0) + Number(specialization.attackDamagePercent || 0) + Number(statUpgrades.attackDamagePercent || 0) + Number(permanent.attackDamagePercent || 0) + Number(passive.attackDamagePercent || 0);
    const power = Math.max(1, Math.round(powerFlat * (1 + powerPercent / 100) * buffPower * firePower));
    const damageFloorBonus = Number(gear.damageFloor || 0) + Number(cards.damageFloor || 0) + Number(roster.damageFloor || 0) + Number(party.damageFloor || 0) + Number(specialization.damageFloor || 0) + Number(statUpgrades.damageFloor || 0) + Number(permanent.damageFloor || 0) + Number(passive.damageFloor || 0);
    const damageFloor = getDamageFloorPercent(damageFloorBonus);
    const defenseFlat = base.defense + gear.defense + Number(cards.defense || 0) + Number(roster.defense || 0) + Number(party.defense || 0) + Number(specialization.defense || 0) + Number(statUpgrades.defense || 0) + Number(permanent.defense || 0) + Number(passive.defense || 0) + Math.floor(level * 0.65);
    const defensePercent = Number(gear.defensePercent || 0) + Number(cards.defensePercent || 0) + Number(roster.defensePercent || 0) + Number(party.defensePercent || 0) + Number(specialization.defensePercent || 0) + Number(statUpgrades.defensePercent || 0) + Number(permanent.defensePercent || 0) + Number(passive.defensePercent || 0);
    const resourceGain = gear.resourceGain + Number(cards.resourceGain || 0) + Number(roster.resourceGain || 0) + Number(party.resourceGain || 0) + Number(specialization.resourceGain || 0) + Number(statUpgrades.resourceGain || 0) + Number(permanent.resourceGain || 0) + Number(passive.resourceGain || 0) + (hasBuff('runeCircle') || hasBuff('packCall') ? scaleBuffStatAmount(8, buffScale) : 0);
    const resourceGainPercent = Number(gear.resourceGainPercent || 0) + Number(cards.resourceGainPercent || 0) + Number(roster.resourceGainPercent || 0) + Number(party.resourceGainPercent || 0) + Number(specialization.resourceGainPercent || 0) + Number(statUpgrades.resourceGainPercent || 0) + Number(permanent.resourceGainPercent || 0) + Number(passive.resourceGainPercent || 0);
    return {
      buffScale,
      power,
      powerPercent,
      attackDamagePercent,
      damageFloor,
      defense: Math.max(0, Math.round(defenseFlat * (1 + defensePercent / 100))),
      defensePercent,
      resourceGain,
      resourceGainPercent
    };
  }

  function createStatsSecondaryCombatTotals(sources, buffScale, options) {
    const settings = options || {};
    const source = sources && typeof sources === 'object' ? sources : {};
    const base = source.base || {};
    const gear = source.gear || {};
    const cards = source.cards || {};
    const roster = source.roster || {};
    const party = source.party || {};
    const specialization = source.specialization || {};
    const statUpgrades = source.statUpgrades || {};
    const permanent = source.permanent || {};
    const passive = source.passive || {};
    const hasBuff = typeof settings.hasBuff === 'function' ? settings.hasBuff : () => false;
    const scaleBuffStatAmount = typeof settings.scaleBuffStatAmount === 'function'
      ? settings.scaleBuffStatAmount
      : (value, scale) => Math.round(Number(value || 0) * Math.max(0, Number(scale || 1) || 1));
    const baseCritChance = Number(settings.baseCritChance || 0);
    const scale = Math.max(0, Number(buffScale || 1) || 1);
    return {
      avoid: Math.max(0, Math.round((base.speed + gear.speed + Number(cards.speed || 0) + Number(roster.speed || 0) + Number(party.speed || 0) + Number(specialization.speed || 0) + Number(statUpgrades.speed || 0) + Number(permanent.speed || 0) + Number(passive.speed || 0)) / 14 + (gear.crit + Number(cards.crit || 0) + Number(roster.crit || 0) + Number(party.crit || 0) + Number(specialization.crit || 0) + Number(statUpgrades.crit || 0) + Number(permanent.crit || 0) + Number(passive.crit || 0)) * 0.4 + Number(gear.avoid || 0) + Number(cards.avoid || 0) + Number(party.avoid || 0) + Number(specialization.avoid || 0) + Number(statUpgrades.avoid || 0) + Number(permanent.avoid || 0) + Number(passive.avoid || 0))),
      speed: Math.max(140, base.speed + gear.speed + Number(cards.speed || 0) + Number(roster.speed || 0) + Number(party.speed || 0) + Number(specialization.speed || 0) + Number(statUpgrades.speed || 0) + Number(permanent.speed || 0) + Number(passive.speed || 0) + (hasBuff('swiftstepOil') ? scaleBuffStatAmount(46, scale) : 0)),
      jump: base.jump,
      range: base.range + gear.range + Number(cards.range || 0) + Number(roster.range || 0) + Number(party.range || 0) + Number(specialization.range || 0) + Number(statUpgrades.range || 0) + Number(permanent.range || 0) + Number(passive.range || 0),
      crit: Math.max(0, baseCritChance + gear.crit + Number(cards.crit || 0) + Number(roster.crit || 0) + Number(party.crit || 0) + Number(specialization.crit || 0) + Number(statUpgrades.crit || 0) + Number(permanent.crit || 0) + Number(passive.crit || 0) + (hasBuff('eagleEye') || hasBuff('rallyingFlourish') ? scaleBuffStatAmount(15, scale) : 0)),
      critDamage: gear.critDamage + Number(cards.critDamage || 0) + Number(roster.critDamage || 0) + Number(party.critDamage || 0) + Number(specialization.critDamage || 0) + Number(statUpgrades.critDamage || 0) + Number(permanent.critDamage || 0) + Number(passive.critDamage || 0) + (hasBuff('eagleEye') ? scaleBuffStatAmount(20, scale) : 0),
      areaDamage: gear.areaDamage + Number(cards.areaDamage || 0) + Number(roster.areaDamage || 0) + Number(party.areaDamage || 0) + Number(specialization.areaDamage || 0) + Number(statUpgrades.areaDamage || 0) + Number(permanent.areaDamage || 0) + Number(passive.areaDamage || 0) + (hasBuff('stormfront') ? scaleBuffStatAmount(12, scale) : 0),
      armorBreak: gear.armorBreak + Number(cards.armorBreak || 0) + Number(roster.armorBreak || 0) + Number(party.armorBreak || 0) + Number(specialization.armorBreak || 0) + Number(statUpgrades.armorBreak || 0) + Number(permanent.armorBreak || 0) + Number(passive.armorBreak || 0),
      burnDamage: gear.burnDamage + Number(cards.burnDamage || 0) + Number(roster.burnDamage || 0) + Number(party.burnDamage || 0) + Number(specialization.burnDamage || 0) + Number(statUpgrades.burnDamage || 0) + Number(permanent.burnDamage || 0) + Number(passive.burnDamage || 0) + (hasBuff('ignitionAura') ? scaleBuffStatAmount(20, scale) : 0),
      trapDamage: gear.trapDamage + Number(cards.trapDamage || 0) + Number(roster.trapDamage || 0) + Number(party.trapDamage || 0) + Number(specialization.trapDamage || 0) + Number(statUpgrades.trapDamage || 0) + Number(permanent.trapDamage || 0) + Number(passive.trapDamage || 0) + (hasBuff('tacticalField') ? scaleBuffStatAmount(15, scale) : 0),
      block: gear.block + Number(cards.block || 0) + Number(roster.block || 0) + Number(party.block || 0) + Number(specialization.block || 0) + Number(statUpgrades.block || 0) + Number(permanent.block || 0) + Number(passive.block || 0) + (hasBuff('shieldWall') ? scaleBuffStatAmount(10, scale) : 0)
    };
  }

  function createStatsDamageRange(power, attackDamagePercent, damageFloor, options) {
    const settings = options || {};
    const makeDamageRange = typeof settings.makeDamageRange === 'function'
      ? settings.makeDamageRange
      : (adjustedPower, floorPercent) => ({ min: adjustedPower, max: adjustedPower, floorPercent });
    const adjustedPower = Math.max(1, Math.round(Number(power || 0) * (1 + Math.max(0, Number(attackDamagePercent || 0)) / 100)));
    return makeDamageRange(adjustedPower, damageFloor);
  }

  function createStatsRuntimeSnapshot(vitalTotals, sourceTotals, coreCombatTotals, secondaryCombatTotals, damageRange) {
    const vital = vitalTotals || {};
    const sources = sourceTotals || {};
    const core = coreCombatTotals || {};
    const secondary = secondaryCombatTotals || {};
    return {
      maxHp: vital.maxHp,
      maxMp: vital.maxMp,
      secondaryResourceMax: vital.secondaryResourceMax,
      resourceMax: vital.secondaryResourceMax,
      power: core.power,
      powerPercent: core.powerPercent,
      attackDamagePercent: core.attackDamagePercent,
      maxHpPercent: vital.maxHpPercent,
      maxMpPercent: vital.maxMpPercent,
      hpRecoveryPercent: sources.hpRecoveryPercent,
      mpRecoveryPercent: sources.mpRecoveryPercent,
      skillEffectPercent: sources.skillEffectPercent,
      buffEffectPercent: sources.buffEffectPercent,
      bossDamagePercent: sources.bossDamagePercent,
      eliteDamagePercent: sources.eliteDamagePercent,
      resourceCostReductionPercent: sources.resourceCostReductionPercent,
      shieldStrengthPercent: sources.shieldStrengthPercent,
      markDuration: sources.markDuration,
      runeDuration: sources.runeDuration,
      weakPointDuration: sources.weakPointDuration,
      cooldownRecoveryPercent: sources.cooldownRecoveryPercent,
      buffDurationPercent: sources.buffDurationPercent,
      damageReductionPercent: sources.damageReductionPercent,
      potionEffectPercent: sources.potionEffectPercent,
      executeDamagePercent: sources.executeDamagePercent,
      mobilityCooldownPercent: sources.mobilityCooldownPercent,
      mobilityWindowPercent: sources.mobilityWindowPercent,
      trapSpeed: sources.trapSpeed,
      hpOnHit: sources.hpOnHit,
      mpOnHit: sources.mpOnHit,
      defensePercent: core.defensePercent,
      resourceGainPercent: core.resourceGainPercent,
      damageFloor: core.damageFloor,
      damageRange,
      defense: core.defense,
      avoid: secondary.avoid,
      speed: secondary.speed,
      jump: secondary.jump,
      range: secondary.range,
      crit: secondary.crit,
      critDamage: secondary.critDamage,
      resourceGain: core.resourceGain,
      areaDamage: secondary.areaDamage,
      armorBreak: secondary.armorBreak,
      burnDamage: secondary.burnDamage,
      trapDamage: secondary.trapDamage,
      block: secondary.block
    };
  }

  function getCachedStatsValue(signature, cachedSignature, cachedValue) {
    return signature && cachedSignature === signature && cachedValue
      ? cachedValue
      : null;
  }

  function getCachedStatsFrameValue(frameId, frameSignature, statsRevision, cache) {
    const source = cache && typeof cache === 'object' ? cache : {};
    const revision = Math.max(0, Number(statsRevision || 0)) || 0;
    return frameSignature &&
      source.frameId === frameId &&
      source.frameRevision === revision &&
      source.frameSignature === frameSignature &&
      source.frameValue
      ? source.frameValue
      : null;
  }

  function createStatsFrameCacheState(frameId, frameSignature, statsRevision, frameValue) {
    if (!frameSignature) return null;
    return {
      frameId,
      frameRevision: Math.max(0, Number(statsRevision || 0)) || 0,
      frameSignature,
      frameValue
    };
  }

  function getCachedStatsVitalValue(frameId, frameSignature, statsRevision, vitalSignature, cachedVitalSignature, cachedVitalValue) {
    const cachedVitalStats = getCachedStatsValue(vitalSignature, cachedVitalSignature, cachedVitalValue);
    if (!cachedVitalStats) return null;
    const frameCache = createStatsFrameCacheState(frameId, frameSignature, statsRevision, cachedVitalStats);
    return {
      stats: cachedVitalStats,
      cacheState: frameCache
        ? {
          statsFrameId: frameCache.frameId,
          statsFrameRevision: frameCache.frameRevision,
          statsFrameSignature: frameCache.frameSignature,
          statsFrameValue: frameCache.frameValue
        }
        : null
    };
  }

  function createStatsComputedCacheState(frameId, frameSignature, statsRevision, vitalSignature, statsValue) {
    const state = {};
    const frameCache = createStatsFrameCacheState(frameId, frameSignature, statsRevision, statsValue);
    if (frameCache) {
      state.statsFrameId = frameCache.frameId;
      state.statsFrameRevision = frameCache.frameRevision;
      state.statsFrameSignature = frameCache.frameSignature;
      state.statsFrameValue = frameCache.frameValue;
    }
    if (vitalSignature) {
      state.statsVitalSignature = vitalSignature;
      state.statsVitalCache = statsValue;
    }
    return state;
  }

  function createStatsBreakdownCacheState(signature, breakdownValue) {
    if (!signature) return null;
    return {
      statBreakdownCacheSignature: signature,
      statBreakdownCache: breakdownValue
    };
  }

  function createEmptyStatsCacheState() {
    return {
      statsFrameId: 0,
      statsFrameRevision: 0,
      statsFrameSignature: '',
      statsFrameValue: null,
      statsVitalSignature: '',
      statsVitalCache: null,
      statBreakdownCacheSignature: '',
      statBreakdownCache: null
    };
  }

  function createStatsCacheInvalidationState(statsRevision) {
    return Object.assign({
      statsRevision: Math.max(0, Number(statsRevision || 0)) + 1
    }, createEmptyStatsCacheState());
  }

  function createStatsFrameResetState(statsRevision) {
    return {
      statsFrameId: 0,
      statsFrameRevision: Math.max(0, Number(statsRevision || 0)) || 0,
      statsFrameSignature: '',
      statsFrameValue: null
    };
  }

  const api = {
    createStatsObjectSignature,
    createStatsPotentialSignature,
    createStatsItemSignature,
    createStatsEquipmentSignature,
    createStatsBuffSignature,
    createStatsFrameSignature,
    createStatsVitalSignature,
    createStatsLevelBreakdown,
    createStatsBaseBreakdown,
    createStatsBuffBreakdown,
    createStatsSourceBundle,
    createStatsPermanentBonuses,
    createStatsBreakdownBuffSources,
    createStatsBreakdownSnapshot,
    createStatsVitalTotals,
    createStatsSourceTotals,
    createStatsCoreCombatTotals,
    createStatsSecondaryCombatTotals,
    createStatsDamageRange,
    createStatsRuntimeSnapshot,
    getCachedStatsValue,
    getCachedStatsFrameValue,
    getCachedStatsVitalValue,
    createStatsFrameCacheState,
    createStatsComputedCacheState,
    createStatsBreakdownCacheState,
    createEmptyStatsCacheState,
    createStatsCacheInvalidationState,
    createStatsFrameResetState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.stats = Object.assign({}, modules.stats || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
