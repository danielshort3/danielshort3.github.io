(function initProjectStarfallUiFormatting(global) {
  'use strict';

  const CoreFormat = (typeof require === 'function' ? require('../core/format.js') : null) || global.ProjectStarfallCore || {};
  const ABBREVIATED_NUMBER_SUFFIXES = CoreFormat.ABBREVIATED_NUMBER_SUFFIXES || Object.freeze(['', 'k', 'm', 'b', 't', 'q', 'qi', 'sx', 'sp', 'oc', 'no', 'dc']);
  const STAT_LABELS = Object.freeze({
    powerPercent: 'Power',
    attackDamagePercent: 'Attack Damage',
    maxHpPercent: 'Max HP',
    maxMpPercent: 'Max MP',
    hpRecoveryPercent: 'HP Recovery',
    mpRecoveryPercent: 'MP Recovery',
    defensePercent: 'Defense',
    resourceGainPercent: 'Resource Gain',
    skillEffectPercent: 'Skill Effect',
    buffEffectPercent: 'Buff Effect',
    bossDamagePercent: 'Boss Damage',
    eliteDamagePercent: 'Elite Damage',
    resourceCostReductionPercent: 'Skill Cost Reduction',
    shieldStrengthPercent: 'Shield Strength',
    markDuration: 'Mark Duration',
    cooldownRecoveryPercent: 'Cooldown Recovery',
    buffDurationPercent: 'Buff Duration',
    damageReductionPercent: 'Damage Reduction',
    potionEffectPercent: 'Potion Effect',
    executeDamagePercent: 'Execute Damage',
    mobilityCooldownPercent: 'Mobility Cooldown',
    mobilityWindowPercent: 'Mobility Window',
    hpOnHit: 'HP On Hit',
    mpOnHit: 'MP On Hit',
    crit: 'Precision',
    critDamage: 'Precision Damage'
  });

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatAdminDebugDetails(details) {
    return escapeHtml(JSON.stringify(details || {}))
      .replace(/Worldwright Console/g, 'Worldwright&nbsp;Console');
  }

  function formatStatName(key) {
    if (STAT_LABELS[key]) return STAT_LABELS[key];
    return String(key || '')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, (letter) => letter.toUpperCase());
  }

  function formatStatSourceName(key) {
    const id = String(key || '');
    if (/Percent$/.test(id) && id !== 'attackDamagePercent' && id !== 'resourceGainPercent') {
      return `${formatStatName(id)} Percent`;
    }
    return formatStatName(id);
  }

  function isPercentStat(key) {
    return /Percent$/.test(String(key || '')) ||
      key === 'crit' ||
      key === 'critDamage' ||
      key === 'block' ||
      key === 'areaDamage' ||
      key === 'burnDamage' ||
      key === 'trapDamage' ||
      key === 'damageFloor';
  }

  function trimFixedNumber(value) {
    return String(value || '0')
      .replace(/(\.\d*?)0+$/, '$1')
      .replace(/\.$/, '');
  }

  function formatAbbreviatedNumber(value, options) {
    const settings = options || {};
    const number = Number(value);
    if (!Number.isFinite(number)) return '0';
    const sign = number < 0 ? '-' : '';
    let amount = Math.abs(number);
    const smallDecimals = Math.max(0, Math.floor(Number(settings.smallDecimals || 0) || 0));
    if (amount < 1000) {
      const text = smallDecimals
        ? trimFixedNumber(amount.toFixed(smallDecimals))
        : String(Math.round(amount));
      return `${sign}${text}`;
    }
    let suffixIndex = 0;
    while (amount >= 1000 && suffixIndex < ABBREVIATED_NUMBER_SUFFIXES.length - 1) {
      amount /= 1000;
      suffixIndex += 1;
    }
    const decimals = Math.max(0, Math.floor(Number(settings.decimals == null ? 1 : settings.decimals) || 0));
    const factor = Math.pow(10, decimals);
    if (Math.round(amount * factor) / factor >= 1000 && suffixIndex < ABBREVIATED_NUMBER_SUFFIXES.length - 1) {
      amount /= 1000;
      suffixIndex += 1;
    }
    return `${sign}${amount.toFixed(decimals)}${ABBREVIATED_NUMBER_SUFFIXES[suffixIndex]}`;
  }

  function formatAbbreviatedInteger(value) {
    return formatAbbreviatedNumber(Math.round(Number(value) || 0), { decimals: 1 });
  }

  function formatStatNumber(value) {
    const number = Number(value) || 0;
    return formatAbbreviatedNumber(number, {
      decimals: 1,
      smallDecimals: Number.isInteger(number) ? 0 : 1
    });
  }

  function formatStatValue(key, value) {
    const number = Number(value) || 0;
    const suffix = isPercentStat(key) ? '%' : '';
    return `${number > 0 ? '+' : ''}${formatStatNumber(number)}${suffix}`;
  }

  function formatStatTotalValue(key, value) {
    const number = Number(value) || 0;
    const suffix = isPercentStat(key) ? '%' : '';
    return `${formatStatNumber(number)}${suffix}`;
  }

  function compactStats(stats, limit, options) {
    const settings = options || {};
    const formatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : formatStatName;
    const formatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : formatStatValue;
    const max = Number(limit) || 3;
    const entries = Object.entries(stats || {})
      .filter((entry) => Number(entry[1]) !== 0)
      .map(([key, value]) => `${formatName(key)} ${formatValue(key, value)}`);
    if (entries.length <= max) return entries.join(', ');
    return `${entries.slice(0, max).join(', ')} +${entries.length - max} more`;
  }

  function createFormattingUiHelpers(options) {
    const settings = options || {};
    const mergeOptions = (override) => Object.assign({}, settings, override || {});
    return Object.freeze({
      escapeHtml,
      formatAdminDebugDetails,
      formatStatName,
      formatStatSourceName,
      isPercentStat,
      trimFixedNumber,
      formatAbbreviatedNumber,
      formatAbbreviatedInteger,
      formatStatNumber,
      formatStatValue,
      formatStatTotalValue,
      compactStats: (stats, limit, override) => compactStats(stats, limit, mergeOptions(override))
    });
  }

  const api = {
    ABBREVIATED_NUMBER_SUFFIXES,
    STAT_LABELS,
    escapeHtml,
    formatAdminDebugDetails,
    formatStatName,
    formatStatSourceName,
    isPercentStat,
    trimFixedNumber,
    formatAbbreviatedNumber,
    formatAbbreviatedInteger,
    formatStatNumber,
    formatStatValue,
    formatStatTotalValue,
    compactStats,
    createFormattingUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.formatting = Object.assign({}, modules.formatting || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
