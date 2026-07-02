(function initProjectStarfallEngineTargetFarm(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getTargetFarmData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createTargetFarmState(value, options) {
    const data = getTargetFarmData(options);
    const source = value && typeof value === 'object' ? value : {};
    const enemyId = normalizeId(source.enemyId);
    const mapId = normalizeId(source.mapId);
    return {
      enemyId: getById(data.ENEMIES || [], enemyId) ? enemyId : '',
      mapId: getById(data.MAPS || [], mapId) ? mapId : '',
      streak: Math.max(0, Math.floor(Number(source.streak || 0) || 0)),
      lastDefeatAt: Number(source.lastDefeatAt || 0),
      bonusUntil: Number(source.bonusUntil || 0)
    };
  }

  function createTargetFarmSelectionPlan(enemy, farm, mapId, nowSeconds, options) {
    if (!enemy) return { ok: false, reason: 'missing' };
    const state = createTargetFarmState(farm, options);
    if (state.enemyId !== enemy.id) {
      state.enemyId = enemy.id;
      state.mapId = normalizeId(mapId);
      state.streak = 0;
    }
    state.bonusUntil = Number(nowSeconds || 0) + 900;
    return {
      ok: true,
      state,
      toast: `Target farming ${enemy.name}.`
    };
  }

  function createTargetFarmDefeatPlan(enemy, farm, options) {
    const settings = options || {};
    if (!enemy || !enemy.id) return { ok: false, reason: 'missing' };
    const state = createTargetFarmState(farm, settings);
    if (!state.enemyId) state.enemyId = normalizeId(settings.selectedEnemyId);
    const time = Number(settings.nowSeconds || 0);
    if (state.enemyId !== enemy.id) {
      if (state.streak > 0 && time - Number(state.lastDefeatAt || 0) > 60) state.streak = 0;
      return { ok: false, reason: 'differentEnemy', state };
    }
    state.mapId = normalizeId(settings.mapId);
    state.lastDefeatAt = time;
    state.bonusUntil = time + 900;
    state.streak = Math.min(50, Math.max(0, Number(state.streak || 0)) + 1 + Number(settings.researchBonus || 0));
    return { ok: true, state };
  }

  function createTargetFarmLootBonus(farm, enemy, options) {
    const data = getTargetFarmData(options);
    const settings = options || {};
    const state = farm && typeof farm === 'object' ? farm : null;
    if (!state || !enemy || state.enemyId !== enemy.id || Number(state.bonusUntil || 0) <= Number(settings.nowSeconds || 0)) return 0;
    const streakBonus = Math.min(0.22, Math.max(0, Number(state.streak || 0)) * 0.012);
    const mapBonus = Number(settings.mapBonus || 0);
    const affixBonus = (enemy.eliteAffixIds || []).reduce((sum, affixId) => {
      const affix = getById(data.ELITE_AFFIXES || [], affixId);
      return sum + Number(affix && affix.targetFarmBonus || 0);
    }, 0);
    return streakBonus + mapBonus + affixBonus;
  }

  function createTargetFarmSnapshot(farm, options) {
    const data = getTargetFarmData(options);
    const settings = options || {};
    const state = createTargetFarmState(farm, settings);
    const enemy = getById(data.ENEMIES || [], state.enemyId);
    const map = getById(data.MAPS || [], state.mapId);
    const table = (data.TARGET_FARM_TABLES || []).find((entry) => entry.enemyId === state.enemyId) || null;
    return {
      enemyId: state.enemyId,
      enemyName: enemy && enemy.name || '',
      mapId: state.mapId,
      mapName: map && map.name || '',
      streak: Math.max(0, Number(state.streak || 0)),
      active: !!(state.enemyId && Number(state.bonusUntil || 0) > Number(settings.nowSeconds || 0)),
      bonus: enemy ? createTargetFarmLootBonus(state, { id: enemy.id, eliteAffixIds: [] }, settings) : 0,
      bonusUntil: Number(state.bonusUntil || 0),
      table
    };
  }

  const api = {
    createTargetFarmState,
    createTargetFarmSelectionPlan,
    createTargetFarmDefeatPlan,
    createTargetFarmLootBonus,
    createTargetFarmSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.targetFarm = Object.assign({}, modules.targetFarm || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
