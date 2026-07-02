(function initProjectStarfallEngineEquipment(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const rollNormalStatVariance = CoreMath.rollNormalStatVariance || function rollNormalStatVarianceFallback() {
    const u1 = Math.max(Number.MIN_VALUE, Math.random());
    const u2 = Math.random();
    return clamp(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2), -2.35, 2.35);
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const DEFAULT_RARITY_ORDER = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic', 'Relic']);
  const ITEM_UPGRADE_STAT_MULTIPLIER_STEP = 0.08;
  const ITEM_UPGRADE_STRENGTH_BONUS = 4;
  const RARITY_STRENGTH_BONUS = 3;
  const POTENTIAL_MAX_LINE_COUNT = 5;
  let equipmentCatalogCache = null;
  let genericEquipmentDropCatalogCache = null;
  const POTENTIAL_STAT_LABELS = Object.freeze({
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

  function getEquipmentCatalog(data) {
    const source = data || {};
    const shopItems = source.SHOP_ITEMS || [];
    const randomItems = source.RANDOM_EQUIPMENT_ITEMS || [];
    const bossItems = source.BOSS_EQUIPMENT_ITEMS || [];
    if (equipmentCatalogCache &&
      equipmentCatalogCache.shopItems === shopItems &&
      equipmentCatalogCache.randomItems === randomItems &&
      equipmentCatalogCache.bossItems === bossItems &&
      equipmentCatalogCache.shopLength === shopItems.length &&
      equipmentCatalogCache.randomLength === randomItems.length &&
      equipmentCatalogCache.bossLength === bossItems.length) {
      return equipmentCatalogCache.catalog;
    }
    const catalog = shopItems.concat(randomItems, bossItems);
    equipmentCatalogCache = {
      shopItems,
      randomItems,
      bossItems,
      shopLength: shopItems.length,
      randomLength: randomItems.length,
      bossLength: bossItems.length,
      catalog
    };
    return catalog;
  }

  function getGenericEquipmentDropCatalog(data) {
    const source = data || {};
    const shopItems = source.SHOP_ITEMS || [];
    const randomItems = source.RANDOM_EQUIPMENT_ITEMS || [];
    if (genericEquipmentDropCatalogCache &&
      genericEquipmentDropCatalogCache.shopItems === shopItems &&
      genericEquipmentDropCatalogCache.randomItems === randomItems &&
      genericEquipmentDropCatalogCache.shopLength === shopItems.length &&
      genericEquipmentDropCatalogCache.randomLength === randomItems.length) {
      return genericEquipmentDropCatalogCache.catalog;
    }
    const catalog = shopItems.concat(randomItems)
      .filter((item) => item && !item.setId && !item.bossId);
    genericEquipmentDropCatalogCache = {
      shopItems,
      randomItems,
      shopLength: shopItems.length,
      randomLength: randomItems.length,
      catalog
    };
    return catalog;
  }

  function getEquipmentDefinition(itemId, options) {
    const settings = options || {};
    const catalog = Array.isArray(settings.catalog) ? settings.catalog : getEquipmentCatalog(settings.data);
    for (let index = 0; index < catalog.length; index += 1) {
      if (catalog[index] && catalog[index].id === itemId) return catalog[index];
    }
    return null;
  }

  function getEquipmentSetDefinition(setId, options) {
    const source = options && options.data || {};
    const id = normalizeId(setId);
    return (source.EQUIPMENT_SETS || []).find((set) => set && set.id === id) || null;
  }

  function getBossEquipmentSource(bossId, options) {
    const source = options && options.data || {};
    const id = normalizeId(bossId);
    return (source.BOSS_EQUIPMENT_SOURCES || []).find((entry) => entry && entry.bossId === id) || null;
  }

  function createPotentialLookups(data) {
    const source = data || {};
    const tiers = Array.isArray(source.POTENTIAL_TIERS) ? source.POTENTIAL_TIERS : [];
    const pools = Array.isArray(source.POTENTIAL_LINE_POOLS) ? source.POTENTIAL_LINE_POOLS : [];
    return {
      tierFallback: tiers[0] || null,
      tiersById: Object.freeze(tiers.reduce((lookup, tier) => {
        const id = normalizeId(tier && tier.id);
        if (id) lookup[id] = tier;
        return lookup;
      }, {})),
      tierIndexById: Object.freeze(tiers.reduce((lookup, tier, index) => {
        const id = normalizeId(tier && tier.id);
        if (id) lookup[id] = index;
        return lookup;
      }, {})),
      linePools: pools,
      linePoolsByStat: Object.freeze(pools.reduce((lookup, pool) => {
        const stat = normalizeId(pool && pool.stat);
        if (stat && !lookup[stat]) lookup[stat] = pool;
        return lookup;
      }, {})),
      linePoolsByTierSlotCache: new Map()
    };
  }

  function getPotentialLookups(options) {
    const settings = options || {};
    return settings.lookups || createPotentialLookups(settings.data);
  }

  function getPotentialMaxLineCount(options) {
    const settings = options || {};
    return Math.max(1, Math.floor(Number(settings.maxLineCount || POTENTIAL_MAX_LINE_COUNT) || POTENTIAL_MAX_LINE_COUNT));
  }

  function getPotentialTierDefinition(tierId, options) {
    const lookups = getPotentialLookups(options);
    const id = normalizeId(tierId) || 'rare';
    return lookups.tiersById[id] || lookups.tierFallback;
  }

  function getPotentialTierRank(tierId, options) {
    const lookups = getPotentialLookups(options);
    const id = normalizeId(tierId);
    return Object.prototype.hasOwnProperty.call(lookups.tierIndexById, id) ? lookups.tierIndexById[id] : -1;
  }

  function formatPotentialStatLabel(stat) {
    const id = normalizeId(stat);
    if (POTENTIAL_STAT_LABELS[id]) return POTENTIAL_STAT_LABELS[id];
    return id.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function formatPotentialStatValue(stat, value) {
    const amount = Math.round(Number(value || 0) || 0);
    const suffix = /Percent|critDamage|crit|block|areaDamage|burnDamage|trapDamage|damageFloor|resourceGain/i.test(String(stat || '')) ? '%' : '';
    return `${amount > 0 ? '+' : ''}${amount}${suffix}`;
  }

  function formatPotentialLineLabel(line) {
    return `${formatPotentialStatLabel(line && line.stat)} ${formatPotentialStatValue(line && line.stat, line && line.value)}`;
  }

  function createEmptyEquipmentStats() {
    return {
      hp: 0,
      mpMax: 0,
      power: 0,
      defense: 0,
      avoid: 0,
      speed: 0,
      range: 0,
      crit: 0,
      critDamage: 0,
      resourceGain: 0,
      resourceMax: 0,
      powerPercent: 0,
      attackDamagePercent: 0,
      maxHpPercent: 0,
      maxMpPercent: 0,
      hpRecoveryPercent: 0,
      mpRecoveryPercent: 0,
      skillEffectPercent: 0,
      buffEffectPercent: 0,
      bossDamagePercent: 0,
      eliteDamagePercent: 0,
      resourceCostReductionPercent: 0,
      shieldStrengthPercent: 0,
      markDuration: 0,
      cooldownRecoveryPercent: 0,
      buffDurationPercent: 0,
      damageReductionPercent: 0,
      potionEffectPercent: 0,
      executeDamagePercent: 0,
      mobilityCooldownPercent: 0,
      mobilityWindowPercent: 0,
      hpOnHit: 0,
      mpOnHit: 0,
      defensePercent: 0,
      resourceGainPercent: 0,
      areaDamage: 0,
      armorBreak: 0,
      burnDamage: 0,
      trapDamage: 0,
      block: 0,
      damageFloor: 0,
      runeDuration: 0,
      weakPointDuration: 0,
      trapSpeed: 0
    };
  }

  function rollDroppedStatValue(value, meanMultiplier, options) {
    const settings = options || {};
    const rollVariance = typeof settings.rollNormalStatVariance === 'function'
      ? settings.rollNormalStatVariance
      : rollNormalStatVariance;
    const baseValue = Number(value) || 0;
    if (!baseValue) return 0;
    const variance = rollVariance() * 0.22;
    if (baseValue < 0) {
      const penaltyMultiplier = clamp(Number(meanMultiplier || 1) - variance, 0.35, 1.9);
      return Math.min(-1, Math.round(baseValue * penaltyMultiplier));
    }
    const multiplier = clamp(Number(meanMultiplier || 1) + variance, 0.55, 1.9);
    return Math.max(1, Math.round(baseValue * multiplier));
  }

  function getDroppedStatExpectedValue(value, meanMultiplier) {
    const baseValue = Number(value) || 0;
    if (!baseValue) return 0;
    const multiplier = clamp(Number(meanMultiplier || 1), 0.35, 1.9);
    if (baseValue < 0) return Math.min(-1, Math.round(baseValue * multiplier));
    return Math.max(1, Math.round(baseValue * multiplier));
  }

  function createStatRollMetadata(baseStats, meanMultiplier) {
    const expectedStats = {};
    Object.entries(baseStats || {}).forEach(([key, value]) => {
      expectedStats[key] = getDroppedStatExpectedValue(value, meanMultiplier);
    });
    return {
      meanMultiplier: Number((Number(meanMultiplier || 1) || 1).toFixed(4)),
      expectedStats
    };
  }

  function cloneStatRollStats(stats, options) {
    const settings = options || {};
    if (typeof settings.cloneStats === 'function') return settings.cloneStats(stats);
    return Object.entries(stats || {}).reduce((result, [key, value]) => {
      result[key] = Number(value) || 0;
      return result;
    }, {});
  }

  function getItemRarityRollMultiplier(item, levelBoost, options) {
    const settings = options || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : DEFAULT_RARITY_ORDER;
    const rarityIndex = Math.max(0, rarityOrder.indexOf(item && item.rarity || 'Common'));
    return 1 + rarityIndex * 0.12 + Math.max(0, Number(levelBoost || 0) || 0);
  }

  function inferLegacyItemStatRollLevelBoost(item) {
    const source = String(item && item.source || '').trim().toLowerCase();
    if (!source) return null;
    if (source === 'admin console') return 0;
    if (!source.endsWith(' drop')) return null;
    if (item && (item.setId || item.bossId)) return 0;
    if (source.includes('emberjaw') || source.includes('cracked mimic') || source.includes('crackedmimic')) return 0.25;
    return 0.08;
  }

  function itemStatsDifferFromBase(item, baseStats, options) {
    const stats = cloneStatRollStats(item && item.stats || {}, options);
    return Object.keys(baseStats || {}).some((key) => Math.round(Number(stats[key] || 0) || 0) !== Math.round(Number(baseStats[key] || 0) || 0));
  }

  function normalizeItemStatRollMetadata(item, options) {
    const settings = options || {};
    if (!item || !item.slot) return null;
    const getBaseStatSource = typeof settings.getItemBaseStatSource === 'function'
      ? settings.getItemBaseStatSource
      : () => ({});
    const createMetadata = typeof settings.createStatRollMetadata === 'function'
      ? settings.createStatRollMetadata
      : createStatRollMetadata;
    const baseStats = cloneStatRollStats(item.baseStats && Object.keys(item.baseStats).length ? item.baseStats : getBaseStatSource(item), settings);
    if (!Object.keys(baseStats).length) return null;
    const raw = item.statRoll && typeof item.statRoll === 'object' ? item.statRoll : null;
    const rawExpected = raw && raw.expectedStats && typeof raw.expectedStats === 'object' ? raw.expectedStats : null;
    const expectedStats = {};
    if (rawExpected) {
      Object.keys(baseStats).forEach((key) => {
        const value = Number(rawExpected[key]);
        if (Number.isFinite(value) && value) expectedStats[key] = Math.round(value);
      });
    }
    let meanMultiplier = Number(raw && raw.meanMultiplier || 0) || 0;
    if (!Object.keys(expectedStats).length) {
      if (!(meanMultiplier > 0)) {
        const inferredLevelBoost = inferLegacyItemStatRollLevelBoost(item);
        if (inferredLevelBoost == null || !itemStatsDifferFromBase(item, baseStats, settings)) return null;
        meanMultiplier = getItemRarityRollMultiplier(item, inferredLevelBoost, settings);
      }
      return createMetadata(baseStats, meanMultiplier);
    }
    const normalized = { expectedStats };
    if (meanMultiplier > 0) normalized.meanMultiplier = Number(meanMultiplier.toFixed(4));
    return normalized;
  }

  function getGearTraitDefinition(traitId, options) {
    const settings = options || {};
    const gearTraits = Array.isArray(settings.gearTraits) ? settings.gearTraits : [];
    const id = normalizeId(traitId);
    return gearTraits.find((trait) => trait && trait.id === id) || null;
  }

  function getRarityMinimumRank(rarity, options) {
    const settings = options || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : DEFAULT_RARITY_ORDER;
    const rank = rarityOrder.indexOf(rarity || 'Common');
    return rank < 0 ? 0 : rank;
  }

  function gearTraitMatchesItem(trait, item, options) {
    const settings = options || {};
    if (!trait || !item || !item.slot) return false;
    const slots = Array.isArray(trait.slots) ? trait.slots : [];
    if (slots.length && !slots.includes(item.slot)) return false;
    return getRarityMinimumRank(item.rarity, settings) >= getRarityMinimumRank(trait.rarity || 'Common', settings);
  }

  function getEligibleGearTraits(item, options) {
    const settings = options || {};
    const gearTraits = Array.isArray(settings.gearTraits) ? settings.gearTraits : [];
    return gearTraits.filter((trait) => gearTraitMatchesItem(trait, item, settings));
  }

  function getItemClassIds(item) {
    const ids = Array.isArray(item && item.classIds)
      ? item.classIds.map(normalizeId).filter(Boolean)
      : [];
    const single = normalizeId(item && item.classId || 'any');
    if (!ids.length && single) ids.push(single);
    return ids.length ? Array.from(new Set(ids)) : ['any'];
  }

  function itemMatchesPlayerClass(item, player) {
    const ids = getItemClassIds(item);
    if (ids.includes('any')) return true;
    const playerIds = new Set([
      normalizeId(player && player.classId),
      normalizeId(player && player.advancedClassId)
    ].filter(Boolean));
    return ids.some((id) => playerIds.has(id));
  }

  function getEquipmentDropClassWeight(item, player, options) {
    const settings = options || {};
    if (!item) return 1;
    const classWeights = settings.classWeights || {};
    if (item.classId === 'any' || Array.isArray(item.classIds) && item.classIds.includes('any')) {
      return Math.max(1, Number(classWeights.universal || 3));
    }
    const matchesPlayerClass = typeof settings.itemMatchesPlayerClass === 'function'
      ? settings.itemMatchesPlayerClass
      : itemMatchesPlayerClass;
    return matchesPlayerClass(item, player)
      ? Math.max(1, Number(classWeights.currentClass || 4))
      : Math.max(1, Number(classWeights.offClass || 1));
  }

  function playerOwnsEquipmentBase(state, base) {
    if (!state || !base || !base.id) return false;
    const inventory = Array.isArray(state.inventory) ? state.inventory : [];
    const equipped = state.equipment && typeof state.equipment === 'object' ? Object.values(state.equipment) : [];
    return inventory.concat(equipped).some((item) => item && item.id === base.id);
  }

  function getBossEquipmentDropWeight(base, player, state, options) {
    const settings = options || {};
    const pieceWeights = settings.pieceWeights || {};
    const ownedWeight = playerOwnsEquipmentBase(state, base)
      ? Math.max(1, Number(pieceWeights.duplicate || 1))
      : Math.max(1, Number(pieceWeights.missing || 6));
    return getEquipmentDropClassWeight(base, player, settings) * ownedWeight;
  }

  function normalizeEquipmentDropWeight(value, fallback) {
    return Math.max(1, Math.round(Number(value == null ? fallback : value) || fallback || 1));
  }

  function getMonsterEquipmentDropWeight(entry, state, options) {
    const settings = options || {};
    const getDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : () => null;
    const normalizeWeight = typeof settings.normalizeDropWeight === 'function'
      ? settings.normalizeDropWeight
      : normalizeEquipmentDropWeight;
    const base = getDefinition(entry && entry.itemId);
    if (!base) return normalizeWeight(entry && entry.weight, 1);
    return normalizeWeight(entry.weight, 1) * getEquipmentDropClassWeight(base, state && state.player, settings);
  }

  function getItemBaseStatSource(item, options) {
    const settings = options || {};
    if (!item) return {};
    const getDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : () => null;
    const base = getDefinition(item.id);
    return item.baseStats && Object.keys(item.baseStats).length ? item.baseStats : base && base.stats || item.stats || {};
  }

  function getAdminItemStatBounds(item, stat, options) {
    const settings = options || {};
    const id = normalizeId(stat);
    const getBaseStatSource = typeof settings.getItemBaseStatSource === 'function'
      ? settings.getItemBaseStatSource
      : (targetItem) => getItemBaseStatSource(targetItem, settings);
    const source = getBaseStatSource(item);
    if (!item || !id || !Object.prototype.hasOwnProperty.call(source, id)) return null;
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : DEFAULT_RARITY_ORDER;
    const baseValue = Number(source[id]) || 0;
    if (!baseValue) return { min: 0, max: 0, base: 0 };
    const rarityIndex = Math.max(0, rarityOrder.indexOf(item.rarity || 'Common'));
    const bonus = 1 + rarityIndex * 0.12;
    const varianceCap = 2.35 * 0.22;
    if (baseValue < 0) {
      const min = Math.min(-1, Math.round(baseValue * clamp(bonus + varianceCap, 0.35, 1.9)));
      const max = Math.min(-1, Math.round(baseValue * clamp(bonus - varianceCap, 0.35, 1.9)));
      return { min: Math.min(min, max), max: Math.max(min, max), base: baseValue };
    }
    const min = Math.max(1, Math.round(baseValue * clamp(bonus - varianceCap, 0.55, 1.9)));
    const max = Math.max(1, Math.round(baseValue * clamp(bonus + varianceCap, 0.55, 1.9)));
    return { min: Math.min(min, max), max: Math.max(min, max), base: baseValue };
  }

  function getAdminPotentialLineOptions(item, tierId, options) {
    const settings = options || {};
    const getTier = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : (id) => getPotentialTierDefinition(id, settings);
    const tier = getTier(tierId);
    if (!tier) return [];
    const lookups = getPotentialLookups(settings);
    const pools = Array.isArray(settings.potentialLinePools)
      ? settings.potentialLinePools
      : Array.isArray(lookups.linePools) ? lookups.linePools : [];
    const isEligible = typeof settings.isPotentialLinePoolEligible === 'function'
      ? settings.isPotentialLinePoolEligible
      : (pool, targetTierId, slot) => isPotentialLinePoolEligible(pool, targetTierId, slot);
    const getRange = typeof settings.getPotentialLineValueRange === 'function'
      ? settings.getPotentialLineValueRange
      : getPotentialLineValueRange;
    const formatLabel = typeof settings.formatPotentialStatLabel === 'function'
      ? settings.formatPotentialStatLabel
      : formatPotentialStatLabel;
    return pools
      .filter((pool) => isEligible(pool, tier.id, item && item.slot))
      .map((pool) => {
        const range = getRange(pool, tier.id);
        return {
          stat: pool.stat,
          min: range[0],
          max: range[1],
          label: formatLabel(pool.stat)
        };
      });
  }

  function getPotentialLinePoolForStat(stat, options) {
    const lookups = getPotentialLookups(options);
    const id = normalizeId(stat);
    return lookups.linePoolsByStat[id] || null;
  }

  function getPotentialLineValueRange(pool, tierId) {
    const values = pool && pool.values && pool.values[tierId] || null;
    if (!Array.isArray(values)) return [1, 1];
    const low = Math.ceil(Number(values[0]) || 0);
    const high = Math.floor(Number(values[1]) || low);
    return [Math.min(low, high), Math.max(low, high)];
  }

  function isPotentialLinePoolValidForSlot(pool, slot) {
    const slots = Array.isArray(pool && pool.slots) ? pool.slots : [];
    return !slots.length || !slot || slots.includes(normalizeId(slot));
  }

  function isPotentialLinePoolValidForTier(pool, tierId) {
    return !!(pool && pool.values && Array.isArray(pool.values[tierId]));
  }

  function isPotentialLinePoolEligible(pool, tierId, slot) {
    return isPotentialLinePoolValidForTier(pool, tierId) && isPotentialLinePoolValidForSlot(pool, slot);
  }

  function getPotentialLinePoolsForRoll(tierId, slot, options) {
    const lookups = getPotentialLookups(options);
    const tier = normalizeId(tierId);
    if (!tier) return [];
    const slotId = normalizeId(slot);
    const cacheKey = `${tier}:${slotId}`;
    const cache = lookups.linePoolsByTierSlotCache;
    if (cache && cache.has(cacheKey)) return cache.get(cacheKey);
    const pools = Array.isArray(lookups.linePools) ? lookups.linePools : [];
    const eligible = pools.filter((line) => isPotentialLinePoolEligible(line, tier, slotId));
    const source = Object.freeze(eligible.length ? eligible : pools.filter((line) => isPotentialLinePoolValidForTier(line, tier)));
    if (cache) cache.set(cacheKey, source);
    return source;
  }

  function normalizePotentialLine(line, tierId, item, options) {
    if (!line || typeof line !== 'object') return null;
    const stat = normalizeId(line.stat);
    const value = Math.round(Number(line.value || 0) || 0);
    const pool = getPotentialLinePoolForStat(stat, options);
    if (!stat || !value || !pool) return null;
    if (tierId && !isPotentialLinePoolEligible(pool, tierId, item && item.slot)) return null;
    return { stat, value };
  }

  function normalizePotentialLineCount(value, options) {
    return clamp(Math.floor(Number(value || 1) || 1), 1, getPotentialMaxLineCount(options));
  }

  function normalizePotentialLineArchive(archive, tierId, item, lineCount, options) {
    const source = archive && typeof archive === 'object' ? archive : {};
    const maxLineCount = getPotentialMaxLineCount(options);
    return Object.entries(source).reduce((normalized, [rawIndex, rawLine]) => {
      const index = Math.floor(Number(rawIndex));
      if (!Number.isFinite(index) || index < lineCount || index >= maxLineCount) return normalized;
      const line = normalizePotentialLine(rawLine, tierId, item, options);
      if (line) normalized[index] = line;
      return normalized;
    }, {});
  }

  function normalizeItemPotential(potential, item, options) {
    if (!potential || typeof potential !== 'object') return null;
    const tier = getPotentialTierDefinition(potential.tier, options);
    if (!tier) return null;
    const requestedLineCount = Object.prototype.hasOwnProperty.call(potential, 'lineCount')
      ? normalizePotentialLineCount(potential.lineCount, options)
      : normalizePotentialLineCount(Array.isArray(potential.lines) && potential.lines.length ? potential.lines.length : 1, options);
    const lines = (Array.isArray(potential.lines) ? potential.lines : [])
      .map((line) => normalizePotentialLine(line, tier.id, item, options))
      .filter(Boolean)
      .slice(0, requestedLineCount);
    if (!lines.length) return null;
    const lineCount = normalizePotentialLineCount(Math.min(requestedLineCount, lines.length), options);
    const lineArchive = normalizePotentialLineArchive(potential.lineArchive, tier.id, item, lineCount, options);
    const normalized = { tier: tier.id, lineCount, lines: lines.slice(0, lineCount) };
    if (Object.keys(lineArchive).length) normalized.lineArchive = lineArchive;
    return normalized;
  }

  function itemPotentialNeedsCleanup(potential, item, options) {
    if (!potential || typeof potential !== 'object') return false;
    return !normalizeItemPotential(potential, item, options);
  }

  function getItemPotentialStats(item, options) {
    const stats = {};
    const potential = normalizeItemPotential(item && item.potential, item, options);
    (potential && potential.lines || []).forEach((line) => {
      stats[line.stat] = (stats[line.stat] || 0) + Number(line.value || 0);
    });
    return stats;
  }

  function getEffectiveItemStats(item, options) {
    const settings = options || {};
    const stats = {};
    if (!item || !item.stats) return stats;
    const upgradeMultiplier = 1 + Number(item.upgrade || 0) * ITEM_UPGRADE_STAT_MULTIPLIER_STEP;
    Object.entries(item.stats).forEach(([key, value]) => {
      stats[key] = Math.round(Number(value || 0) * upgradeMultiplier);
    });
    if (settings.includePotential !== false && typeof settings.getItemPotentialStats === 'function') {
      const potentialStats = settings.getItemPotentialStats(item);
      if (typeof settings.addStats === 'function') {
        settings.addStats(stats, potentialStats);
      } else {
        Object.entries(potentialStats || {}).forEach(([key, value]) => {
          const amount = Math.round(Number(value || 0) || 0);
          if (amount) stats[key] = (stats[key] || 0) + amount;
        });
      }
    }
    return stats;
  }

  function getItemStrength(item, options) {
    const settings = options || {};
    if (!item) return 0;
    const getStats = typeof settings.getEffectiveItemStats === 'function'
      ? settings.getEffectiveItemStats
      : getEffectiveItemStats;
    const stats = getStats(item, settings);
    const statScoreWeights = settings.statScoreWeights || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : DEFAULT_RARITY_ORDER;
    const total = Object.entries(stats).reduce((score, [key, value]) => {
      const amount = Number(value) || 0;
      const weight = statScoreWeights[key] || 1;
      return score + Math.max(0, amount) * weight;
    }, 0);
    const rarityBonus = rarityOrder.indexOf(item.rarity || 'Common') * RARITY_STRENGTH_BONUS;
    return Math.max(1, Math.round(total + Math.max(0, rarityBonus) + Number(item.upgrade || 0) * ITEM_UPGRADE_STRENGTH_BONUS));
  }

  function getItemPowerDelta(state, item, options) {
    const settings = options || {};
    const getStrength = typeof settings.getItemStrength === 'function'
      ? settings.getItemStrength
      : getItemStrength;
    const equipped = item && item.slot && state && state.equipment ? state.equipment[item.slot] : null;
    const strengthOptions = Object.assign({}, settings, { includePotential: false });
    return getStrength(item, strengthOptions) - getStrength(equipped, strengthOptions);
  }

  function getItemSellValue(item, options) {
    const settings = options || {};
    const getDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : () => null;
    const getStrength = typeof settings.getItemStrength === 'function'
      ? settings.getItemStrength
      : getItemStrength;
    const rateValue = settings.sellValueRate == null ? 0.25 : Number(settings.sellValueRate);
    const sellValueRate = Math.max(0, Number.isFinite(rateValue) ? rateValue : 0.25);
    const base = getDefinition(item && item.id);
    const baseValue = Math.max(0, Number(item && item.baseCost || base && base.cost || 0)) || Math.max(1, getStrength(item) * 8);
    return Math.max(1, Math.round(baseValue * sellValueRate));
  }

  function getItemTypeIndex(item, options) {
    const settings = options || {};
    const equipmentSlots = Array.isArray(settings.equipmentSlots) ? settings.equipmentSlots : [];
    const slot = normalizeId(item && item.slot);
    const index = equipmentSlots.indexOf(slot);
    return index < 0 ? 999 : index;
  }

  function getRarityRank(rarity, options) {
    const settings = options || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : DEFAULT_RARITY_ORDER;
    return Math.max(0, rarityOrder.indexOf(rarity || 'Common'));
  }

  const api = {
    DEFAULT_RARITY_ORDER,
    ITEM_UPGRADE_STAT_MULTIPLIER_STEP,
    ITEM_UPGRADE_STRENGTH_BONUS,
    RARITY_STRENGTH_BONUS,
    POTENTIAL_MAX_LINE_COUNT,
    POTENTIAL_STAT_LABELS,
    getEquipmentCatalog,
    getGenericEquipmentDropCatalog,
    getEquipmentDefinition,
    getEquipmentSetDefinition,
    getBossEquipmentSource,
    createPotentialLookups,
    getPotentialTierDefinition,
    getPotentialTierRank,
    formatPotentialStatLabel,
    formatPotentialStatValue,
    formatPotentialLineLabel,
    createEmptyEquipmentStats,
    rollDroppedStatValue,
    getDroppedStatExpectedValue,
    createStatRollMetadata,
    getItemRarityRollMultiplier,
    inferLegacyItemStatRollLevelBoost,
    itemStatsDifferFromBase,
    normalizeItemStatRollMetadata,
    getGearTraitDefinition,
    getRarityMinimumRank,
    gearTraitMatchesItem,
    getEligibleGearTraits,
    getItemClassIds,
    itemMatchesPlayerClass,
    getEquipmentDropClassWeight,
    playerOwnsEquipmentBase,
    getBossEquipmentDropWeight,
    getMonsterEquipmentDropWeight,
    getItemBaseStatSource,
    getAdminItemStatBounds,
    getAdminPotentialLineOptions,
    getPotentialLinePoolForStat,
    getPotentialLineValueRange,
    isPotentialLinePoolValidForSlot,
    isPotentialLinePoolValidForTier,
    isPotentialLinePoolEligible,
    getPotentialLinePoolsForRoll,
    normalizePotentialLine,
    normalizePotentialLineCount,
    normalizePotentialLineArchive,
    normalizeItemPotential,
    itemPotentialNeedsCleanup,
    getItemPotentialStats,
    getEffectiveItemStats,
    getItemStrength,
    getItemPowerDelta,
    getItemSellValue,
    getItemTypeIndex,
    getRarityRank
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.equipment = Object.assign({}, modules.equipment || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
