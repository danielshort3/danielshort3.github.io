(function initProjectStarfallUiStatMetadata(global) {
  'use strict';

  const STAT_SCORE_WEIGHTS = Object.freeze({
    hp: 0.18,
    mpMax: 0.55,
    power: 3,
    powerPercent: 4.2,
    defense: 2.2,
    defensePercent: 2.6,
    avoid: 2,
    speed: 1.25,
    range: 0.08,
    crit: 4,
    critDamage: 0.8,
    attackDamagePercent: 4.8,
    resourceGain: 3.8,
    resourceGainPercent: 2.8,
    resourceMax: 0.55,
    maxHpPercent: 1.8,
    maxMpPercent: 1.4,
    hpRecoveryPercent: 0.65,
    mpRecoveryPercent: 0.7,
    skillEffectPercent: 4,
    buffEffectPercent: 3.4,
    bossDamagePercent: 3.8,
    eliteDamagePercent: 2.8,
    resourceCostReductionPercent: 2.5,
    shieldStrengthPercent: 1.8,
    markDuration: 0.9,
    cooldownRecoveryPercent: 3.2,
    buffDurationPercent: 1.6,
    damageReductionPercent: 3.4,
    potionEffectPercent: 0.8,
    executeDamagePercent: 3,
    mobilityCooldownPercent: 1.5,
    mobilityWindowPercent: 0.9,
    hpOnHit: 1.2,
    mpOnHit: 1.2,
    areaDamage: 3,
    armorBreak: 3.2,
    burnDamage: 2.6,
    trapDamage: 2.6,
    block: 3.2,
    damageFloor: 3.6,
    runeDuration: 0.8,
    weakPointDuration: 1,
    trapSpeed: 1.4
  });

  const STAT_DISPLAY_ORDER = Object.freeze(Object.keys(STAT_SCORE_WEIGHTS));

  function createStatDisplayOrderIndex(order) {
    const source = Array.isArray(order) ? order : STAT_DISPLAY_ORDER;
    return Object.freeze(source.reduce((map, key, index) => {
      map[key] = index;
      return map;
    }, {}));
  }

  const STAT_DISPLAY_ORDER_INDEX = createStatDisplayOrderIndex(STAT_DISPLAY_ORDER);

  const EQUIPMENT_BONUS_STAT_KEYS = Object.freeze([
    'resourceGain',
    'resourceGainPercent',
    'attackDamagePercent',
    'powerPercent',
    'bossDamagePercent',
    'eliteDamagePercent',
    'maxHpPercent',
    'maxMpPercent',
    'hpRecoveryPercent',
    'mpRecoveryPercent',
    'skillEffectPercent',
    'buffEffectPercent',
    'resourceCostReductionPercent',
    'shieldStrengthPercent',
    'cooldownRecoveryPercent',
    'buffDurationPercent',
    'damageReductionPercent',
    'potionEffectPercent',
    'executeDamagePercent',
    'mobilityCooldownPercent',
    'mobilityWindowPercent',
    'hpOnHit',
    'mpOnHit',
    'defensePercent',
    'critDamage',
    'areaDamage',
    'armorBreak',
    'burnDamage',
    'trapDamage',
    'damageFloor',
    'block',
    'speed',
    'avoid',
    'markDuration',
    'runeDuration',
    'weakPointDuration',
    'trapSpeed'
  ]);

  const STAT_BREAKDOWN_SOURCE_LABELS = Object.freeze({
    base: 'Base',
    level: 'Level',
    gear: 'Gear',
    attunement: 'Attunement',
    set: 'Set',
    roster: 'Roster',
    specialization: 'Specialization',
    statUpgrades: 'Stat Upgrades',
    permanent: 'Permanent',
    passive: 'Passive',
    party: 'Party',
    buffs: 'Buffs'
  });

  const STAT_ROLL_TIERS = Object.freeze({
    low: Object.freeze({ id: 'low', label: 'Low', color: '#b83232', fill: 'rgba(184,50,50,0.1)' }),
    under: Object.freeze({ id: 'under', label: 'Under', color: '#b96a20', fill: 'rgba(185,106,32,0.1)' }),
    base: Object.freeze({ id: 'base', label: 'Base', color: '#8a7a2b', fill: 'rgba(216,183,74,0.12)' }),
    strong: Object.freeze({ id: 'strong', label: 'Strong', color: '#177645', fill: 'rgba(23,118,69,0.1)' }),
    exceptional: Object.freeze({ id: 'exceptional', label: 'Exceptional', color: '#2f7dd6', fill: 'rgba(47,125,214,0.1)' })
  });
  const ITEM_STRENGTH_RARITY_ORDER = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic', 'Relic']);

  function getStatScoreWeight(statKey) {
    return STAT_SCORE_WEIGHTS[statKey] || 1;
  }

  function getStatDisplayOrderIndex(statKey, fallback) {
    return Object.prototype.hasOwnProperty.call(STAT_DISPLAY_ORDER_INDEX, statKey)
      ? STAT_DISPLAY_ORDER_INDEX[statKey]
      : Number(fallback == null ? 999 : fallback);
  }

  function normalizeStats(stats) {
    const result = {};
    Object.entries(stats || {}).forEach(([key, value]) => {
      result[key] = Number(value) || 0;
    });
    return result;
  }

  function getEffectiveStats(item, options) {
    const settings = options || {};
    const stats = {};
    if (!item || !item.stats) return stats;
    const getPotentialStats = typeof settings.getItemPotentialStats === 'function'
      ? settings.getItemPotentialStats
      : () => ({});
    const upgradeMultiplier = 1 + Number(item.upgrade || 0) * 0.08;
    Object.entries(item.stats).forEach(([key, value]) => {
      stats[key] = Math.round(Number(value || 0) * upgradeMultiplier);
    });
    if (settings.includePotential !== false) {
      Object.entries(getPotentialStats(item)).forEach(([key, value]) => {
        stats[key] = (stats[key] || 0) + Number(value || 0);
      });
    }
    return stats;
  }

  function getItemStrength(item, options) {
    const settings = options || {};
    if (!item) return 0;
    const getEffectiveStats = typeof settings.getEffectiveStats === 'function'
      ? settings.getEffectiveStats
      : (entry) => normalizeStats(entry && entry.stats);
    const statScoreWeights = settings.statScoreWeights || STAT_SCORE_WEIGHTS;
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : ITEM_STRENGTH_RARITY_ORDER;
    const stats = getEffectiveStats(item, settings);
    const total = Object.entries(stats).reduce((score, [key, value]) => {
      const amount = Number(value) || 0;
      const weight = statScoreWeights[key] || 1;
      return score + Math.max(0, amount) * weight;
    }, 0);
    const rarityBonus = rarityOrder.indexOf(item.rarity || 'Common') * 3;
    return Math.max(1, Math.round(total + rarityBonus + Number(item.upgrade || 0) * 4));
  }

  function getItemTemplateBaseStats(item, options) {
    const settings = options || {};
    if (!item) return {};
    const getEquipmentTemplate = typeof settings.getEquipmentTemplate === 'function'
      ? settings.getEquipmentTemplate
      : () => null;
    const savedBaseStats = item.baseStats && typeof item.baseStats === 'object' && Object.keys(item.baseStats).length ? item.baseStats : null;
    const template = getEquipmentTemplate(item.id);
    return normalizeStats(savedBaseStats || template && template.stats || item.stats);
  }

  function getItemStatRollTier(baseValue, rollDelta, expectedValue, options) {
    const tiers = options && options.statRollTiers || STAT_ROLL_TIERS;
    const base = Number(baseValue) || 0;
    const rolled = base + (Number(rollDelta) || 0);
    const expected = Number(expectedValue);
    const reference = Number.isFinite(expected) ? expected : base;
    const delta = rolled - reference;
    if (delta === 0) return tiers.base;
    if (reference === 0) {
      if (delta >= 5) return tiers.exceptional;
      if (delta > 0) return tiers.strong;
      if (delta <= -5) return tiers.low;
      return tiers.under;
    }
    const ratio = delta / Math.max(1, Math.abs(reference));
    if (ratio <= -0.25) return tiers.low;
    if (ratio <= -0.1) return tiers.under;
    if (ratio >= 0.25) return tiers.exceptional;
    if (ratio >= 0.1) return tiers.strong;
    return tiers.base;
  }

  function getItemStatRollExpectedStats(item) {
    const expectedStats = item && item.statRoll && item.statRoll.expectedStats;
    return expectedStats && typeof expectedStats === 'object' ? normalizeStats(expectedStats) : {};
  }

  function getItemStatRollBreakdown(item, options) {
    const settings = options || {};
    const getBaseStats = typeof settings.getItemTemplateBaseStats === 'function'
      ? settings.getItemTemplateBaseStats
      : (target) => normalizeStats(target && target.baseStats || target && target.stats);
    const getExpectedStats = typeof settings.getItemStatRollExpectedStats === 'function'
      ? settings.getItemStatRollExpectedStats
      : getItemStatRollExpectedStats;
    const getPotentialStats = typeof settings.getItemPotentialStats === 'function'
      ? settings.getItemPotentialStats
      : () => ({});
    const getEffectiveStatsForItem = typeof settings.getEffectiveStats === 'function'
      ? settings.getEffectiveStats
      : (target) => getEffectiveStats(target, settings);
    const getRollTier = typeof settings.getItemStatRollTier === 'function'
      ? settings.getItemStatRollTier
      : (baseValue, rollDelta, expectedValue) => getItemStatRollTier(baseValue, rollDelta, expectedValue, settings);
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (key) => String(key || '');
    const displayOrderIndex = settings.statDisplayOrderIndex || STAT_DISPLAY_ORDER_INDEX;
    const baseStats = getBaseStats(item);
    const rolledStats = normalizeStats(item && item.stats);
    const expectedStats = getExpectedStats(item);
    const potentialStats = normalizeStats(getPotentialStats(item));
    const effectiveStats = getEffectiveStatsForItem(item);
    return Array.from(new Set([
      ...Object.keys(baseStats),
      ...Object.keys(rolledStats),
      ...Object.keys(potentialStats),
      ...Object.keys(effectiveStats)
    ]))
      .filter((key) => baseStats[key] || rolledStats[key] || potentialStats[key] || effectiveStats[key])
      .sort((a, b) => {
        const orderA = Object.prototype.hasOwnProperty.call(displayOrderIndex, a) ? displayOrderIndex[a] : 999;
        const orderB = Object.prototype.hasOwnProperty.call(displayOrderIndex, b) ? displayOrderIndex[b] : 999;
        return orderA - orderB || a.localeCompare(b);
      })
      .map((key) => {
        const base = Number(baseStats[key] || 0);
        const rolled = Number(rolledStats[key] || 0);
        const roll = rolled - base;
        const expected = Object.prototype.hasOwnProperty.call(expectedStats, key) ? Number(expectedStats[key] || 0) : base;
        const upgrade = Math.round(rolled * (1 + Number(item && item.upgrade || 0) * 0.08)) - rolled;
        const potential = Number(potentialStats[key] || 0);
        return {
          key,
          label: formatStatName(key),
          base,
          roll,
          expected,
          upgrade,
          potential,
          value: Number(effectiveStats[key] || 0),
          tier: getRollTier(base, roll, expected)
        };
      });
  }

  function createStatUiHelpers(options) {
    const settings = options || {};
    const sharedOptions = Object.assign({}, settings, {
      statScoreWeights: settings.statScoreWeights || STAT_SCORE_WEIGHTS,
      rarityOrder: Array.isArray(settings.rarityOrder) ? settings.rarityOrder : ITEM_STRENGTH_RARITY_ORDER,
      statRollTiers: settings.statRollTiers || STAT_ROLL_TIERS,
      statDisplayOrderIndex: settings.statDisplayOrderIndex || STAT_DISPLAY_ORDER_INDEX
    });
    const helpers = {};
    const mergeOptions = (override) => Object.assign({}, sharedOptions, override || {});
    helpers.normalizeStats = normalizeStats;
    helpers.getEffectiveStats = (item, override) => getEffectiveStats(item, mergeOptions(override));
    helpers.getItemStrength = (item, override) => getItemStrength(item, Object.assign(mergeOptions(override), {
      getEffectiveStats: helpers.getEffectiveStats
    }));
    helpers.getItemTemplateBaseStats = (item, override) => getItemTemplateBaseStats(item, mergeOptions(override));
    helpers.getItemStatRollTier = (baseValue, rollDelta, expectedValue, override) => getItemStatRollTier(baseValue, rollDelta, expectedValue, mergeOptions(override));
    helpers.getItemStatRollExpectedStats = getItemStatRollExpectedStats;
    helpers.getItemStatRollBreakdown = (item, override) => getItemStatRollBreakdown(item, Object.assign(mergeOptions(override), {
      getItemTemplateBaseStats: helpers.getItemTemplateBaseStats,
      getItemStatRollExpectedStats: helpers.getItemStatRollExpectedStats,
      getEffectiveStats: helpers.getEffectiveStats,
      getItemStatRollTier: helpers.getItemStatRollTier
    }));
    helpers.getPotentialRollRank = getPotentialRollRank;
    helpers.getPotentialComparisonDeltaClass = getPotentialComparisonDeltaClass;
    helpers.getPotentialComparisonLineIndex = getPotentialComparisonLineIndex;
    helpers.getPotentialStatAbbreviation = (key, override) => getPotentialStatAbbreviation(key, mergeOptions(override));
    helpers.getAttunementStatDescription = (stat, override) => getAttunementStatDescription(stat, mergeOptions(override));
    helpers.getAttunementStatTooltip = (stat, override) => getAttunementStatTooltip(stat, mergeOptions(override));
    helpers.getStatUpgradeDomAction = getStatUpgradeDomAction;
    helpers.getStatUpgradeRegionAction = getStatUpgradeRegionAction;
    return Object.freeze(helpers);
  }

  function getPotentialRollRank(percentileLabel) {
    const raw = String(percentileLabel || '').replace('%', '').trim();
    if (!raw) {
      return {
        label: '?',
        className: 'is-rank-unknown',
        percentLabel: '--',
        fill: '#edf1f5',
        color: '#5f6f7a'
      };
    }
    const percent = Math.max(0, Math.min(100, Math.round(Number(raw) || 0)));
    if (percent >= 90) return { label: 'S', className: 'is-rank-s', percentLabel: `${percent}%`, fill: '#fdf1c9', color: '#805500' };
    if (percent >= 75) return { label: 'A', className: 'is-rank-a', percentLabel: `${percent}%`, fill: '#e5f7ed', color: '#177645' };
    if (percent >= 55) return { label: 'B', className: 'is-rank-b', percentLabel: `${percent}%`, fill: '#e9f3ff', color: '#2f7dd6' };
    if (percent >= 30) return { label: 'C', className: 'is-rank-c', percentLabel: `${percent}%`, fill: '#f1edf9', color: '#7950b4' };
    return { label: 'D', className: 'is-rank-d', percentLabel: `${percent}%`, fill: '#f8e8e4', color: '#a94432' };
  }

  function getPotentialComparisonDeltaClass(delta) {
    const value = Number(delta || 0);
    if (value > 0) return 'is-gain';
    if (value < 0) return 'is-loss';
    return 'is-same';
  }

  function getPotentialComparisonLineIndex(row) {
    if (!row) return 999;
    const index = Number(row.lineIndex);
    return Number.isFinite(index) ? index : 999;
  }

  function getPotentialStatAbbreviation(key, options) {
    const formatStatName = options && typeof options.formatStatName === 'function'
      ? options.formatStatName
      : (value) => String(value || '');
    const label = formatStatName(key);
    const letters = label.split(/\s+/)
      .filter(Boolean)
      .map((part) => part.charAt(0))
      .join('')
      .slice(0, 2)
      .toUpperCase();
    return letters || 'ST';
  }

  const ATTUNEMENT_STAT_DESCRIPTIONS = Object.freeze({
    powerPercent: 'Raises final Power from the item power stat.',
    attackDamagePercent: 'Increases final attack damage after Power is calculated.',
    bossDamagePercent: 'Increases damage dealt to boss enemies.',
    eliteDamagePercent: 'Increases damage dealt to elite enemies.',
    areaDamage: 'Increases area and splash damage from supported attacks.',
    burnDamage: 'Increases burn and fire damage over time.',
    armorBreak: 'Improves armor break scaling against enemy defense.',
    maxMpPercent: 'Increases maximum MP.',
    mpRecoveryPercent: 'Increases MP recovery effects.',
    resourceGainPercent: 'Increases class resource gained.',
    resourceMax: 'Increases maximum class resource capacity.',
    resourceCostReductionPercent: 'Reduces skill resource costs.',
    shieldStrengthPercent: 'Increases shield and barrier amounts.',
    runeDuration: 'Extends rune field duration.',
    block: 'Increases block chance.',
    crit: 'Increases Precision chance.',
    weakPointDuration: 'Extends weak-point vulnerability windows.',
    markDuration: 'Extends mark duration on enemies.',
    cooldownRecoveryPercent: 'Reduces skill cooldown time.',
    buffDurationPercent: 'Extends timed buff duration.',
    maxHpPercent: 'Increases maximum HP.',
    defensePercent: 'Increases defense from gear and stats.',
    hpRecoveryPercent: 'Increases HP recovery effects.',
    damageReductionPercent: 'Reduces incoming damage.',
    potionEffectPercent: 'Improves potion healing and resource effects.',
    critDamage: 'Increases Precision hit damage.',
    damageFloor: 'Raises low-end damage rolls.',
    trapDamage: 'Increases trap damage.',
    trapSpeed: 'Reduces trap setup and arm time.',
    executeDamagePercent: 'Increases damage against low-health enemies.',
    speed: 'Increases movement speed.',
    avoid: 'Increases avoid chance.',
    skillEffectPercent: 'Increases supported skill effect scaling.',
    mobilityCooldownPercent: 'Reduces mobility skill cooldown time.',
    mobilityWindowPercent: 'Extends mobility invulnerability and effect windows.',
    hpOnHit: 'Restores HP when hitting enemies.',
    mpOnHit: 'Restores MP when hitting enemies.',
    buffEffectPercent: 'Increases buff stat effects.'
  });

  function getAttunementStatDescription(stat, options) {
    const formatStatName = options && typeof options.formatStatName === 'function'
      ? options.formatStatName
      : (value) => String(value || '');
    const id = String(stat || '');
    return ATTUNEMENT_STAT_DESCRIPTIONS[id] || `Applies ${formatStatName(id)} as an attunement stat.`;
  }

  function getAttunementStatTooltip(stat, options) {
    const settings = options || {};
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (value) => String(value || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (key, value) => String(value);
    const parts = [`${formatStatName(stat)}: ${getAttunementStatDescription(stat, { formatStatName })}`];
    if (settings.value != null) parts.push(`Current roll ${formatStatValue(stat, settings.value)}.`);
    if (settings.rangeLabel) parts.push(`Tier range ${settings.rangeLabel}.`);
    if (settings.rankLabel) parts.push(`Roll rank ${settings.rankLabel}.`);
    parts.push('Duplicate lines stack.');
    return parts.join(' ');
  }

  function getStatUpgradeDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const statId = getAttribute('data-starfall-stat-upgrade');
    if (statId) {
      return {
        handled: true,
        type: 'upgradeStat',
        statId,
        amount: getAttribute('data-starfall-stat-upgrade-amount') || 1
      };
    }
    if (hasAttribute('data-starfall-stat-reset')) return { handled: true, type: 'resetStats' };
    return { handled: false, type: '' };
  }

  function getStatUpgradeRegionAction(region) {
    const source = region || {};
    if (source.type === 'stat-upgrade') {
      return {
        handled: true,
        type: 'upgradeStat',
        statId: source.statId,
        amount: source.amount
      };
    }
    if (source.type === 'stat-reset') return { handled: true, type: 'resetStats' };
    return { handled: false, type: '' };
  }

  const api = {
    STAT_SCORE_WEIGHTS,
    STAT_DISPLAY_ORDER,
    STAT_DISPLAY_ORDER_INDEX,
    EQUIPMENT_BONUS_STAT_KEYS,
    STAT_BREAKDOWN_SOURCE_LABELS,
    STAT_ROLL_TIERS,
    ATTUNEMENT_STAT_DESCRIPTIONS,
    ITEM_STRENGTH_RARITY_ORDER,
    createStatDisplayOrderIndex,
    getStatScoreWeight,
    getStatDisplayOrderIndex,
    normalizeStats,
    getEffectiveStats,
    getItemStrength,
    getItemTemplateBaseStats,
    getItemStatRollTier,
    getItemStatRollExpectedStats,
    getItemStatRollBreakdown,
    createStatUiHelpers,
    getPotentialRollRank,
    getPotentialComparisonDeltaClass,
    getPotentialComparisonLineIndex,
    getPotentialStatAbbreviation,
    getAttunementStatDescription,
    getAttunementStatTooltip,
    getStatUpgradeDomAction,
    getStatUpgradeRegionAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.statMetadata = Object.assign({}, modules.statMetadata || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
