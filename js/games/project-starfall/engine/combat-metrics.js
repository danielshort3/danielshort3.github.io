(function initProjectStarfallEngineCombatMetrics(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreTime = (typeof require === 'function' ? require('../core/time.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const nowSeconds = CoreTime.wallNowSeconds || function nowSecondsFallback() {
    return Date.now() / 1000;
  };

  const COMBAT_METRICS_ROLLING_WINDOW_SECONDS = 60;
  const COMBAT_METRIC_COUNTER_FIELDS = Object.freeze({
    kill: 'kills',
    bossKill: 'bossKills',
    eliteKill: 'eliteKills',
    drop: 'dropsGenerated',
    loot: 'dropsLooted',
    consumable: 'consumablesUsed',
    potion: 'potionsUsed',
    potionCost: 'potionCost',
    death: 'deaths',
    damageTaken: 'damageTaken'
  });

  function createMetricCounterMap(value) {
    const source = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    return Object.keys(source).reduce((result, key) => {
      const id = normalizeId(key);
      const amount = Math.max(0, Number(source[key] || 0));
      if (id && amount > 0) result[id] = amount;
      return result;
    }, {});
  }

  function getCombatMetricCounterField(kind) {
    return COMBAT_METRIC_COUNTER_FIELDS[normalizeId(kind)] || '';
  }

  function createCombatMetricsState(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      startedAt: Number(source.startedAt || 0),
      damageDealt: Math.max(0, Number(source.damageDealt || 0)),
      currencyGained: Math.max(0, Number(source.currencyGained || 0)),
      currencySpent: Math.max(0, Number(source.currencySpent || 0)),
      xpGained: Math.max(0, Number(source.xpGained || 0)),
      kills: Math.max(0, Number(source.kills || 0)),
      bossKills: Math.max(0, Number(source.bossKills || 0)),
      eliteKills: Math.max(0, Number(source.eliteKills || 0)),
      dropsGenerated: Math.max(0, Number(source.dropsGenerated || 0)),
      dropsLooted: Math.max(0, Number(source.dropsLooted || 0)),
      consumablesUsed: Math.max(0, Number(source.consumablesUsed || 0)),
      potionsUsed: Math.max(0, Number(source.potionsUsed || 0)),
      potionCost: Math.max(0, Number(source.potionCost || 0)),
      deaths: Math.max(0, Number(source.deaths || 0)),
      damageTaken: Math.max(0, Number(source.damageTaken || 0)),
      skillCasts: createMetricCounterMap(source.skillCasts),
      skillDamage: createMetricCounterMap(source.skillDamage),
      lootKinds: createMetricCounterMap(source.lootKinds),
      currencySinks: createMetricCounterMap(source.currencySinks),
      showPanel: !!source.showPanel
    };
  }

  function normalizeCombatMetricsState(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return createCombatMetricsState(value);
    value.startedAt = Number(value.startedAt || 0);
    value.damageDealt = Math.max(0, Number(value.damageDealt || 0));
    value.currencyGained = Math.max(0, Number(value.currencyGained || 0));
    value.currencySpent = Math.max(0, Number(value.currencySpent || 0));
    value.xpGained = Math.max(0, Number(value.xpGained || 0));
    value.kills = Math.max(0, Number(value.kills || 0));
    value.bossKills = Math.max(0, Number(value.bossKills || 0));
    value.eliteKills = Math.max(0, Number(value.eliteKills || 0));
    value.dropsGenerated = Math.max(0, Number(value.dropsGenerated || 0));
    value.dropsLooted = Math.max(0, Number(value.dropsLooted || 0));
    value.consumablesUsed = Math.max(0, Number(value.consumablesUsed || 0));
    value.potionsUsed = Math.max(0, Number(value.potionsUsed || 0));
    value.potionCost = Math.max(0, Number(value.potionCost || 0));
    value.deaths = Math.max(0, Number(value.deaths || 0));
    value.damageTaken = Math.max(0, Number(value.damageTaken || 0));
    value.skillCasts = createMetricCounterMap(value.skillCasts);
    value.skillDamage = createMetricCounterMap(value.skillDamage);
    value.lootKinds = createMetricCounterMap(value.lootKinds);
    value.currencySinks = createMetricCounterMap(value.currencySinks);
    value.showPanel = !!value.showPanel;
    return value;
  }

  function createCombatMetricBucket(second) {
    return {
      second,
      damage: 0,
      currency: 0,
      currencySpent: 0,
      xp: 0,
      kills: 0,
      bossKills: 0,
      eliteKills: 0,
      drops: 0,
      loot: 0,
      consumables: 0,
      potions: 0,
      potionCost: 0,
      deaths: 0,
      damageTaken: 0
    };
  }

  function getCombatMetricWindowSeconds(options) {
    const settings = options || {};
    return Math.max(1, Math.floor(Number(settings.windowSeconds || COMBAT_METRICS_ROLLING_WINDOW_SECONDS) || COMBAT_METRICS_ROLLING_WINDOW_SECONDS));
  }

  function getCombatMetricSecond(time) {
    return Math.floor(Number(time == null ? nowSeconds() : time) || 0);
  }

  function pruneCombatMetricBuckets(buckets, time, options) {
    const now = getCombatMetricSecond(time);
    const cutoff = now - getCombatMetricWindowSeconds(options) + 1;
    const target = Array.isArray(buckets) ? buckets : [];
    let writeIndex = 0;
    for (let index = 0; index < target.length; index += 1) {
      const bucket = target[index];
      if (!bucket || Number(bucket.second || 0) < cutoff) continue;
      target[writeIndex] = bucket;
      writeIndex += 1;
    }
    target.length = writeIndex;
    return target;
  }

  function findCombatMetricBucket(buckets, second) {
    if (buckets.length && Number(buckets[buckets.length - 1] && buckets[buckets.length - 1].second || 0) === second) {
      return buckets[buckets.length - 1];
    }
    for (let index = 0; index < buckets.length; index += 1) {
      const entry = buckets[index];
      if (Number(entry && entry.second || 0) === second) return entry;
    }
    return null;
  }

  function addCombatMetricBucketValue(bucket, kind, value) {
    if (kind === 'damage') bucket.damage += value;
    if (kind === 'currency') bucket.currency += value;
    if (kind === 'currencySpent') bucket.currencySpent += value;
    if (kind === 'xp') bucket.xp += value;
    if (kind === 'kill') bucket.kills = Math.max(0, Number(bucket.kills || 0)) + value;
    if (kind === 'bossKill') bucket.bossKills = Math.max(0, Number(bucket.bossKills || 0)) + value;
    if (kind === 'eliteKill') bucket.eliteKills = Math.max(0, Number(bucket.eliteKills || 0)) + value;
    if (kind === 'drop') bucket.drops = Math.max(0, Number(bucket.drops || 0)) + value;
    if (kind === 'loot') bucket.loot = Math.max(0, Number(bucket.loot || 0)) + value;
    if (kind === 'consumable') bucket.consumables = Math.max(0, Number(bucket.consumables || 0)) + value;
    if (kind === 'potion') bucket.potions = Math.max(0, Number(bucket.potions || 0)) + value;
    if (kind === 'potionCost') bucket.potionCost = Math.max(0, Number(bucket.potionCost || 0)) + value;
    if (kind === 'death') bucket.deaths = Math.max(0, Number(bucket.deaths || 0)) + value;
    if (kind === 'damageTaken') bucket.damageTaken = Math.max(0, Number(bucket.damageTaken || 0)) + value;
    return bucket;
  }

  function recordCombatMetricBucket(buckets, kind, amount, time, options) {
    const value = Math.max(0, Number(amount || 0));
    const target = Array.isArray(buckets) ? buckets : [];
    if (!value) return target;
    const second = getCombatMetricSecond(time);
    const pruned = pruneCombatMetricBuckets(target, second, options);
    let bucket = findCombatMetricBucket(pruned, second);
    if (!bucket) {
      bucket = createCombatMetricBucket(second);
      pruned.push(bucket);
    }
    addCombatMetricBucketValue(bucket, kind, value);
    return pruned;
  }

  const api = {
    COMBAT_METRICS_ROLLING_WINDOW_SECONDS,
    COMBAT_METRIC_COUNTER_FIELDS,
    createMetricCounterMap,
    getCombatMetricCounterField,
    createCombatMetricsState,
    normalizeCombatMetricsState,
    createCombatMetricBucket,
    pruneCombatMetricBuckets,
    recordCombatMetricBucket
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.combatMetrics = Object.assign({}, modules.combatMetrics || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
