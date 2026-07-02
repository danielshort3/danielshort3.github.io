(function initProjectStarfallEngineStatUpgrades(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const STAT_RESET_SCROLL_ID = 'stat_reset_scroll';

  function getStatUpgradeData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createPermanentStatsState(value) {
    const source = value && typeof value === 'object' ? value : {};
    return Object.entries(source).reduce((stats, [key, amount]) => {
      const id = normalizeId(key);
      if (id) stats[id] = Number(amount) || 0;
      return stats;
    }, {});
  }

  function getStatUpgradeDefinition(statId, options) {
    const data = getStatUpgradeData(options);
    const id = normalizeId(statId);
    return (data.STAT_UPGRADE_DEFINITIONS || []).find((definition) => definition && definition.id === id) || null;
  }

  function createStatUpgradeState(value, options) {
    const data = getStatUpgradeData(options);
    const source = value && typeof value === 'object' ? value : {};
    const sourceAllocations = source.allocations && typeof source.allocations === 'object' ? source.allocations : source;
    const definitions = new Set((data.STAT_UPGRADE_DEFINITIONS || []).map((definition) => definition && definition.id).filter(Boolean));
    const allocations = {};
    Object.entries(sourceAllocations || {}).forEach(([key, amount]) => {
      const id = normalizeId(key);
      if (!definitions.has(id)) return;
      const rank = Math.max(0, Math.floor(Number(amount) || 0));
      if (rank > 0) allocations[id] = rank;
    });
    return { allocations };
  }

  function getStatUpgradeSpentTotal(statUpgrades, options) {
    const state = createStatUpgradeState(statUpgrades, options);
    return Object.values(state.allocations || {}).reduce((sum, amount) => sum + Math.max(0, Math.floor(Number(amount) || 0)), 0);
  }

  function getStatUpgradeLevelBudget(player) {
    return Math.max(0, Math.floor(Number(player && player.level || 1) || 1) - 1);
  }

  function getStatUpgradeQuestBudget(progress, options) {
    const data = getStatUpgradeData(options);
    const completed = new Set((progress && progress.completedQuestIds || []).map(normalizeId));
    return (data.QUESTS || []).reduce((sum, quest) => {
      if (!quest || !completed.has(quest.id)) return sum;
      return sum + Math.max(0, Math.floor(Number(quest.rewards && quest.rewards.statUpgradePoints || 0) || 0));
    }, 0);
  }

  function createStatUpgradeBudget(player, progress, statUpgrades, options) {
    const level = getStatUpgradeLevelBudget(player);
    const quests = getStatUpgradeQuestBudget(progress, options);
    const total = level + quests;
    const spent = getStatUpgradeSpentTotal(statUpgrades, options);
    return {
      level,
      quests,
      total,
      spent,
      available: Math.max(0, total - spent)
    };
  }

  function createStatUpgradeBonuses(statUpgrades, options) {
    const state = createStatUpgradeState(statUpgrades, options);
    const bonuses = {};
    Object.entries(state.allocations || {}).forEach(([statId, rank]) => {
      const definition = getStatUpgradeDefinition(statId, options);
      const amount = Math.max(0, Math.floor(Number(rank) || 0));
      if (!definition || !amount) return;
      Object.entries(definition.statBonuses || {}).forEach(([key, value]) => {
        const id = normalizeId(key);
        if (!id) return;
        bonuses[id] = Number(bonuses[id] || 0) + Number(value || 0) * amount;
      });
    });
    return bonuses;
  }

  function getStatsObjectSignature(source) {
    if (!source || typeof source !== 'object') return '';
    const keys = Object.keys(source).filter((key) => Number(source[key] || 0) !== 0).sort();
    return keys.map((key) => `${normalizeId(key)}=${Number(source[key] || 0)}`).join(',');
  }

  function getStatUpgradeSnapshotCacheKey(player, progress, statUpgrades, consumables, options) {
    const data = getStatUpgradeData(options);
    const state = statUpgrades && typeof statUpgrades === 'object' ? statUpgrades : {};
    const allocations = state.allocations && typeof state.allocations === 'object' ? state.allocations : {};
    const progressState = progress && typeof progress === 'object' ? progress : {};
    let completedQuestIds = '';
    if (Array.isArray(progressState.completedQuestIds)) {
      progressState.completedQuestIds.forEach((rawId) => {
        const id = normalizeId(rawId);
        if (id) completedQuestIds += `${completedQuestIds ? ',' : ''}${id}`;
      });
    }
    return [
      Math.max(1, Number(player && player.level || 1) || 1),
      completedQuestIds,
      getStatsObjectSignature(allocations),
      Math.max(0, Math.floor(Number(consumables && consumables[STAT_RESET_SCROLL_ID] || 0) || 0)),
      (data.STAT_UPGRADE_DEFINITIONS || []).length,
      (data.QUESTS || []).length
    ].join('|');
  }

  function createStatUpgradeSnapshot(statUpgrades, budget, consumables, options) {
    const data = getStatUpgradeData(options);
    const allocations = Object.assign({}, statUpgrades && statUpgrades.allocations || {});
    return {
      allocations,
      budget,
      definitions: (data.STAT_UPGRADE_DEFINITIONS || []).map((definition) => Object.assign({}, definition, {
        rank: Math.max(0, Math.floor(Number(allocations[definition.id] || 0) || 0))
      })),
      resetScrollId: STAT_RESET_SCROLL_ID,
      resetScrollCount: Math.max(0, Math.floor(Number(consumables && consumables[STAT_RESET_SCROLL_ID] || 0) || 0))
    };
  }

  function createStatUpgradeSpendPlan(definition, budget, amount, currentRank) {
    if (!definition) return { ok: false, reason: 'invalid' };
    if (!budget || budget.available <= 0) return { ok: false, reason: 'noPoints' };
    const requested = String(amount) === 'max'
      ? budget.available
      : Math.max(0, Math.floor(Number(amount || 1) || 1));
    const applied = Math.min(requested, budget.available);
    if (applied <= 0) return { ok: false, reason: 'none' };
    return {
      ok: true,
      statId: definition.id,
      applied,
      nextRank: Math.max(0, Math.floor(Number(currentRank || 0) || 0)) + applied,
      toast: `${definition.name} +${applied}.`
    };
  }

  function createStatUpgradeResetPlan(statUpgrades, consumables, options) {
    const settings = options || {};
    const spent = getStatUpgradeSpentTotal(statUpgrades, settings);
    if (spent <= 0) return { ok: false, reason: 'none', spent };
    const consumeScroll = settings.consumeScroll !== false;
    const scrollCount = Math.max(0, Math.floor(Number(consumables && consumables[STAT_RESET_SCROLL_ID] || 0) || 0));
    if (consumeScroll && scrollCount <= 0) return { ok: false, reason: 'missingScroll', spent };
    return {
      ok: true,
      spent,
      consumeScroll,
      scrollCount,
      nextScrollCount: consumeScroll ? scrollCount - 1 : scrollCount,
      toast: `Reset ${spent} Stat Upgrade Point${spent === 1 ? '' : 's'}.`
    };
  }

  const api = {
    STAT_RESET_SCROLL_ID,
    createPermanentStatsState,
    getStatUpgradeDefinition,
    createStatUpgradeState,
    getStatUpgradeSpentTotal,
    getStatUpgradeLevelBudget,
    getStatUpgradeQuestBudget,
    createStatUpgradeBudget,
    createStatUpgradeBonuses,
    getStatUpgradeSnapshotCacheKey,
    createStatUpgradeSnapshot,
    createStatUpgradeSpendPlan,
    createStatUpgradeResetPlan
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.statUpgrades = Object.assign({}, modules.statUpgrades || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
