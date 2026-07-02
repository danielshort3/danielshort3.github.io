(function initProjectStarfallUiItemComparison(global) {
  'use strict';

  const INVENTORY_COMPARISON_CACHE_LIMIT = 256;

  function getFallbackStats(item) {
    return Object.assign({}, item && item.stats || {});
  }

  function getFallbackStrength(item) {
    return item ? 1 : 0;
  }

  function formatFallbackStatName(key) {
    return String(key || '')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }

  function formatFallbackStatValue(_key, value) {
    const number = Number(value) || 0;
    return `${number >= 0 ? '+' : ''}${number}`;
  }

  function formatFallbackSigned(value) {
    const number = Number(value) || 0;
    return `${number >= 0 ? '+' : ''}${number}`;
  }

  function compareItemToEquipped(snapshot, item, equippedOverride, options) {
    const settings = options || {};
    const getEffectiveStats = settings.getEffectiveStats || getFallbackStats;
    const getItemStrength = settings.getItemStrength || getFallbackStrength;
    const formatStatName = settings.formatStatName || formatFallbackStatName;
    const equipped = typeof equippedOverride === 'undefined' && item && item.slot ? snapshot.state.equipment[item.slot] : equippedOverride || null;
    const currentStats = getEffectiveStats(item);
    const equippedStats = getEffectiveStats(equipped);
    const statKeys = Array.from(new Set([...Object.keys(currentStats), ...Object.keys(equippedStats)]));
    return {
      equipped,
      strengthDelta: getItemStrength(item) - getItemStrength(equipped),
      stats: statKeys.map((key) => ({
        key,
        label: formatStatName(key),
        value: Number(currentStats[key] || 0),
        delta: Number(currentStats[key] || 0) - Number(equippedStats[key] || 0)
      })).filter((entry) => entry.value || entry.delta)
    };
  }

  function getComparisonTone(delta) {
    const number = Number(delta) || 0;
    if (number > 0) return 'is-better';
    if (number < 0) return 'is-worse';
    return 'is-even';
  }

  function getInventoryDeltaSummary(snapshot, item, limit, options) {
    const settings = options || {};
    const max = Number(limit) || 2;
    const formatStatValue = settings.formatStatValue || formatFallbackStatValue;
    const compare = compareItemToEquipped(snapshot, item, undefined, settings);
    const stats = compare.stats
      .filter((entry) => entry.delta !== 0)
      .slice(0, max)
      .map((entry) => `${entry.label} ${formatStatValue(entry.key, entry.delta)}`);
    return stats.length ? stats.join(' · ') : 'No stat change';
  }

  function getItemBaseVersion(item) {
    if (!item) return null;
    return Object.assign({}, item, { upgrade: 0, potential: null });
  }

  function compareBaseItemToEquipped(snapshot, item, options) {
    const equipped = item && item.slot ? snapshot.state.equipment[item.slot] : null;
    return compareItemToEquipped(snapshot, getItemBaseVersion(item), getItemBaseVersion(equipped), options);
  }

  function getInventoryArrowMeta(delta, options) {
    const formatSigned = options && options.formatSigned || formatFallbackSigned;
    const value = Number(delta) || 0;
    if (value > 0) return { tone: 'up', symbol: '▲', label: `Increase ${formatSigned(value)}` };
    if (value < 0) return { tone: 'down', symbol: '▼', label: `Decrease ${formatSigned(value)}` };
    return { tone: 'even', symbol: '→', label: 'No change' };
  }

  function getInventoryComparisonPair(snapshot, item, options) {
    const settings = options || {};
    const current = compareItemToEquipped(snapshot, item, undefined, settings);
    const base = compareBaseItemToEquipped(snapshot, item, settings);
    return {
      current,
      base,
      currentArrow: getInventoryArrowMeta(current.strengthDelta, settings),
      baseArrow: getInventoryArrowMeta(base.strengthDelta, settings)
    };
  }

  function createItemComparisonUiHelpers(options) {
    const settings = options || {};
    const mergeOptions = (override) => Object.assign({}, settings, override || {});
    return Object.freeze({
      compareItemToEquipped: (snapshot, item, equippedOverride, override) => compareItemToEquipped(snapshot, item, equippedOverride, mergeOptions(override)),
      getComparisonTone,
      getInventoryDeltaSummary: (snapshot, item, limit, override) => getInventoryDeltaSummary(snapshot, item, limit, mergeOptions(override)),
      getItemBaseVersion,
      compareBaseItemToEquipped: (snapshot, item, override) => compareBaseItemToEquipped(snapshot, item, mergeOptions(override)),
      getInventoryArrowMeta: (delta, override) => getInventoryArrowMeta(delta, mergeOptions(override)),
      getInventoryComparisonPair: (snapshot, item, override) => getInventoryComparisonPair(snapshot, item, mergeOptions(override))
    });
  }

  const api = {
    INVENTORY_COMPARISON_CACHE_LIMIT,
    compareItemToEquipped,
    getComparisonTone,
    getInventoryDeltaSummary,
    getItemBaseVersion,
    compareBaseItemToEquipped,
    getInventoryArrowMeta,
    getInventoryComparisonPair,
    createItemComparisonUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.itemComparison = Object.assign({}, modules.itemComparison || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
