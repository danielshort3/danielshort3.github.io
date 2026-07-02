(function initProjectStarfallEngineRouteProgress(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getRouteProgressData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createRouteProgressState(value, options) {
    const data = getRouteProgressData(options);
    const source = value && typeof value === 'object' ? value : {};
    return (data.WORLD_ROUTES || []).reduce((routes, route) => {
      const routeSource = source[route.id] && typeof source[route.id] === 'object' ? source[route.id] : {};
      const sourceKills = routeSource.killsByMap && typeof routeSource.killsByMap === 'object' ? routeSource.killsByMap : {};
      const killsByMap = {};
      (route.fieldGoals || []).forEach((field) => {
        const mapId = normalizeId(field.mapId);
        if (!mapId) return;
        const raw = sourceKills[mapId] != null ? sourceKills[mapId] : routeSource[mapId];
        killsByMap[mapId] = Math.max(0, Math.floor(Number(raw) || 0));
      });
      routes[route.id] = { killsByMap };
      return routes;
    }, {});
  }

  function getWorldRoute(routeId, options) {
    const data = getRouteProgressData(options);
    return getById(data.WORLD_ROUTES || [], routeId);
  }

  function getRouteForFieldMap(mapId, options) {
    const data = getRouteProgressData(options);
    return (data.WORLD_ROUTES || []).find((route) => (route.fieldGoals || []).some((field) => field.mapId === mapId)) || null;
  }

  function getRouteForBossMap(mapId, options) {
    const data = getRouteProgressData(options);
    return (data.WORLD_ROUTES || []).find((route) => route.bossMapId === mapId) || null;
  }

  function getRouteForDungeon(dungeonId, options) {
    const data = getRouteProgressData(options);
    return (data.WORLD_ROUTES || []).find((route) => route.bossDungeonId === dungeonId) || null;
  }

  function getRouteForMap(mapId, options) {
    return getRouteForFieldMap(mapId, options) || getRouteForBossMap(mapId, options);
  }

  function createRouteFieldStatus(route, field, routeState, options) {
    const data = getRouteProgressData(options);
    const state = route && routeState && routeState[route.id] ? routeState[route.id] : { killsByMap: {} };
    const map = getById(data.MAPS || [], field && field.mapId);
    const goal = Math.max(1, Number(field && field.count || 1));
    const value = Math.min(goal, Math.max(0, Number(state.killsByMap && state.killsByMap[field.mapId]) || 0));
    return {
      routeId: route ? route.id : '',
      mapId: field.mapId,
      mapName: map ? map.name : field.mapId,
      value,
      goal,
      complete: value >= goal
    };
  }

  function createRouteSummary(route, routeState, options) {
    if (!route) return null;
    const fields = (route.fieldGoals || []).map((field) => createRouteFieldStatus(route, field, routeState, options));
    return {
      id: route.id,
      name: route.name,
      startMapId: route.startMapId,
      bossMapId: route.bossMapId,
      bossDungeonId: route.bossDungeonId,
      fields,
      complete: fields.every((field) => field.complete)
    };
  }

  function createRouteProgressSnapshot(routeState, options) {
    const data = getRouteProgressData(options);
    return {
      routes: (data.WORLD_ROUTES || []).map((route) => createRouteSummary(route, routeState, options)).filter(Boolean)
    };
  }

  const api = {
    createRouteProgressState,
    getWorldRoute,
    getRouteForFieldMap,
    getRouteForBossMap,
    getRouteForDungeon,
    getRouteForMap,
    createRouteFieldStatus,
    createRouteSummary,
    createRouteProgressSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.routeProgress = Object.assign({}, modules.routeProgress || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
