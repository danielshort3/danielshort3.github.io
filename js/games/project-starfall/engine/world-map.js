(function initProjectStarfallEngineWorldMap(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const EngineModules = global.ProjectStarfallEngineModules || {};
  const RouteProgress = (typeof require === 'function' ? require('./route-progress.js') : null) || EngineModules.routeProgress || {};
  const DungeonHelpers = (typeof require === 'function' ? require('./dungeons.js') : null) || EngineModules.dungeons || {};
  const PortalHelpers = (typeof require === 'function' ? require('./portals.js') : null) || EngineModules.portals || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const adjacentEdgesBySourceCache = new WeakMap();
  const dungeonDefinitionByMapIdCache = new WeakMap();

  function getWorldMapData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getMapDefinitionById(mapId, data) {
    return getById((data || {}).MAPS || [], normalizeId(mapId));
  }

  function getDungeonDefinitionById(dungeonId, data) {
    return getById((data || {}).DUNGEONS || [], normalizeId(dungeonId));
  }

  function getDungeonDefinitionByMapId(mapId, data) {
    const source = (data || {}).DUNGEONS || [];
    if (!source.length) return null;
    let cached = dungeonDefinitionByMapIdCache.get(source);
    if (!cached) {
      cached = new Map();
      for (let index = 0; index < source.length; index += 1) {
        const dungeon = source[index];
        const id = normalizeId(dungeon && dungeon.mapId);
        if (id) cached.set(id, dungeon);
      }
      dungeonDefinitionByMapIdCache.set(source, cached);
    }
    return cached.get(normalizeId(mapId)) || null;
  }

  function getRouteState(routeState, options) {
    if (routeState && typeof routeState === 'object') return routeState;
    return RouteProgress.createRouteProgressState
      ? RouteProgress.createRouteProgressState(routeState, options)
      : {};
  }

  function getDungeonState(dungeons, options) {
    if (dungeons && typeof dungeons === 'object') return dungeons;
    return DungeonHelpers.createDungeonState
      ? DungeonHelpers.createDungeonState(dungeons, options)
      : { completedDungeonIds: [] };
  }

  function getWorldRoute(routeId, options) {
    if (RouteProgress.getWorldRoute) return RouteProgress.getWorldRoute(routeId, options);
    const data = getWorldMapData(options);
    return getById(data.WORLD_ROUTES || [], normalizeId(routeId));
  }

  function getRouteForFieldMap(mapId, options) {
    if (RouteProgress.getRouteForFieldMap) return RouteProgress.getRouteForFieldMap(mapId, options);
    const data = getWorldMapData(options);
    return (data.WORLD_ROUTES || []).find((route) => (route.fieldGoals || []).some((field) => field.mapId === mapId)) || null;
  }

  function createRouteFieldStatus(route, field, routeState, options) {
    if (RouteProgress.createRouteFieldStatus) {
      return RouteProgress.createRouteFieldStatus(route, field, routeState, options);
    }
    const data = getWorldMapData(options);
    const state = route && routeState && routeState[route.id] ? routeState[route.id] : { killsByMap: {} };
    const map = getMapDefinitionById(field && field.mapId, data);
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

  function getWorldMapNodeDefinition(mapId, options) {
    const data = getWorldMapData(options);
    return (data.WORLD_MAP_NODES || []).find((node) => node && node.mapId === mapId) || null;
  }

  function getPortalDefinition(mapId, portalId, options) {
    const data = getWorldMapData(options);
    const map = getMapDefinitionById(mapId, data);
    if (!map || !portalId) return null;
    return (map.portals || []).find((portal) => portal && portal.id === portalId) || null;
  }

  function getWorldMapEdgePortal(edge, fromMapId, options) {
    if (!edge || !fromMapId || !edge.portalIds) return null;
    const portalId = fromMapId === edge.fromMapId ? edge.portalIds.from : fromMapId === edge.toMapId ? edge.portalIds.to : '';
    return portalId ? getPortalDefinition(fromMapId, portalId, options) : null;
  }

  function getWorldMapEdgeOtherMapId(edge, mapId) {
    if (!edge || !mapId) return '';
    if (edge.fromMapId === mapId) return edge.toMapId;
    if (edge.toMapId === mapId) return edge.fromMapId;
    return '';
  }

  function getWorldMapAdjacentEdges(mapId, options) {
    const data = getWorldMapData(options);
    const source = data.WORLD_MAP_EDGES || [];
    if (!source.length) return [];
    let cached = adjacentEdgesBySourceCache.get(source);
    if (!cached) {
      cached = new Map();
      for (let index = 0; index < source.length; index += 1) {
        const edge = source[index];
        if (!edge) continue;
        [edge.fromMapId, edge.toMapId].forEach((id) => {
          const key = normalizeId(id);
          if (!key) return;
          if (!cached.has(key)) cached.set(key, []);
          cached.get(key).push(edge);
        });
      }
      adjacentEdgesBySourceCache.set(source, cached);
    }
    return cached.get(normalizeId(mapId)) || [];
  }

  function createWorldMapEdgeLockReason(edge, fromMapId, options) {
    const settings = options || {};
    const data = getWorldMapData(settings);
    if (!edge) return 'Route is unavailable.';
    const sourceMapId = fromMapId || edge.fromMapId;
    const lockCache = settings.worldMapEdgeLockReasonCache;
    const cacheKey = lockCache ? `${edge.id || `${edge.fromMapId}>${edge.toMapId}`}|${sourceMapId}` : '';
    if (lockCache && lockCache.has(cacheKey)) return lockCache.get(cacheKey);
    const remember = (reason) => {
      if (lockCache) lockCache.set(cacheKey, reason);
      return reason;
    };
    const portal = getWorldMapEdgePortal(edge, sourceMapId, settings);
    if (portal && PortalHelpers.createPortalBlockReason) {
      return remember(PortalHelpers.createPortalBlockReason(portal, settings));
    }
    const player = settings.player || {};
    if (edge.requiredLevel && player.level < Number(edge.requiredLevel)) return remember(`Level ${edge.requiredLevel} required.`);
    if (edge.requiredDungeonId) {
      const dungeons = getDungeonState(settings.dungeons, settings);
      if (!(dungeons.completedDungeonIds || []).includes(edge.requiredDungeonId)) {
        const dungeon = getDungeonDefinitionById(edge.requiredDungeonId, data);
        return remember(`Clear ${dungeon ? dungeon.name : edge.requiredDungeonId} first.`);
      }
    }
    if (edge.requiredMapId && edge.routeId) {
      const route = getWorldRoute(edge.routeId, settings);
      const field = route && (route.fieldGoals || []).find((goal) => goal.mapId === edge.requiredMapId);
      const status = field ? createRouteFieldStatus(route, field, getRouteState(settings.routeState, settings), settings) : null;
      if (status && !status.complete) return remember(`${route.name}: clear ${status.mapName} (${status.value}/${status.goal}).`);
    }
    return remember('');
  }

  function createWorldMapStep(edge, fromMapId, lockedReason, options) {
    const nextMapId = getWorldMapEdgeOtherMapId(edge, fromMapId);
    const portal = getWorldMapEdgePortal(edge, fromMapId, options);
    return {
      edgeId: edge.id,
      fromMapId,
      toMapId: nextMapId,
      portalId: portal ? portal.id : '',
      portalLabel: portal ? portal.label : '',
      routeId: edge.routeId || '',
      type: edge.type || 'field',
      lockedReason
    };
  }

  function createWorldMapPathMap(fromMapId, options) {
    const settings = options || {};
    const data = getWorldMapData(settings);
    const startId = normalizeId(fromMapId);
    const pathMapCache = settings.worldMapPathMapCache;
    if (pathMapCache && pathMapCache.has(startId)) return pathMapCache.get(startId);
    const paths = new Map();
    if (!startId || !getMapDefinitionById(startId, data)) {
      if (pathMapCache) pathMapCache.set(startId, paths);
      return paths;
    }
    paths.set(startId, { fromMapId: startId, toMapId: startId, steps: [], lockedReason: '', blockedAt: null });
    const queue = [startId];
    while (queue.length) {
      const mapId = queue.shift();
      const currentPath = paths.get(mapId);
      const edges = getWorldMapAdjacentEdges(mapId, settings);
      for (let index = 0; index < edges.length; index += 1) {
        const edge = edges[index];
        const nextMapId = getWorldMapEdgeOtherMapId(edge, mapId);
        if (!nextMapId || paths.has(nextMapId)) continue;
        const lockedReason = createWorldMapEdgeLockReason(edge, mapId, settings);
        const step = createWorldMapStep(edge, mapId, lockedReason, settings);
        const steps = (currentPath && currentPath.steps || []).concat(step);
        const blockedAt = steps.find((item) => item.lockedReason) || null;
        paths.set(nextMapId, {
          fromMapId: startId,
          toMapId: nextMapId,
          steps,
          lockedReason: blockedAt ? blockedAt.lockedReason : '',
          blockedAt
        });
        queue.push(nextMapId);
      }
    }
    if (settings.worldMapPathCache) {
      paths.forEach((path, targetId) => {
        settings.worldMapPathCache.set(`${startId}|${targetId}`, path);
      });
    }
    if (pathMapCache) pathMapCache.set(startId, paths);
    return paths;
  }

  function createWorldMapPath(fromMapId, toMapId, options) {
    const settings = options || {};
    const data = getWorldMapData(settings);
    const startId = normalizeId(fromMapId);
    const targetId = normalizeId(toMapId);
    const pathCache = settings.worldMapPathCache;
    const cacheKey = pathCache ? `${startId}|${targetId}` : '';
    if (pathCache && pathCache.has(cacheKey)) return pathCache.get(cacheKey);
    const remember = (path) => {
      if (pathCache) pathCache.set(cacheKey, path);
      return path;
    };
    if (!startId || !targetId || !getMapDefinitionById(startId, data) || !getMapDefinitionById(targetId, data)) return remember(null);
    if (startId === targetId) return remember({ fromMapId: startId, toMapId: targetId, steps: [], lockedReason: '', blockedAt: null });
    const queue = [{ mapId: startId, steps: [] }];
    const visited = new Set([startId]);
    while (queue.length) {
      const current = queue.shift();
      const edges = getWorldMapAdjacentEdges(current.mapId, settings);
      for (let index = 0; index < edges.length; index += 1) {
        const edge = edges[index];
        const nextMapId = getWorldMapEdgeOtherMapId(edge, current.mapId);
        if (!nextMapId || visited.has(nextMapId)) continue;
        const lockedReason = createWorldMapEdgeLockReason(edge, current.mapId, settings);
        const step = createWorldMapStep(edge, current.mapId, lockedReason, settings);
        const steps = current.steps.concat(step);
        if (nextMapId === targetId) {
          const blockedAt = steps.find((item) => item.lockedReason) || null;
          return remember({
            fromMapId: startId,
            toMapId: targetId,
            steps,
            lockedReason: blockedAt ? blockedAt.lockedReason : '',
            blockedAt
          });
        }
        visited.add(nextMapId);
        queue.push({ mapId: nextMapId, steps });
      }
    }
    return remember(null);
  }

  function isWorldMapEdgeCompleted(edge, options) {
    const settings = options || {};
    if (!edge) return false;
    if (edge.type === 'dungeon' && edge.dungeonId) {
      const dungeons = getDungeonState(settings.dungeons, settings);
      return (dungeons.completedDungeonIds || []).includes(edge.dungeonId);
    }
    if (edge.requiredMapId && edge.routeId) {
      const route = getWorldRoute(edge.routeId, settings);
      const field = route && (route.fieldGoals || []).find((goal) => goal.mapId === edge.requiredMapId);
      const status = field ? createRouteFieldStatus(route, field, getRouteState(settings.routeState, settings), settings) : null;
      return !!(status && status.complete);
    }
    return false;
  }

  function createWorldMapNodeCompletion(map, options) {
    const settings = options || {};
    const data = getWorldMapData(settings);
    if (!map || map.safeZone) return false;
    if (map.isDungeon) {
      const dungeon = getDungeonDefinitionByMapId(map.id, data);
      const dungeons = getDungeonState(settings.dungeons, settings);
      return !!(dungeon && (dungeons.completedDungeonIds || []).includes(dungeon.id));
    }
    const route = getRouteForFieldMap(map.id, settings);
    const field = route && (route.fieldGoals || []).find((goal) => goal.mapId === map.id);
    const status = field ? createRouteFieldStatus(route, field, getRouteState(settings.routeState, settings), settings) : null;
    return !!(status && status.complete);
  }

  const api = {
    getWorldMapNodeDefinition,
    getPortalDefinition,
    getWorldMapEdgePortal,
    getWorldMapEdgeOtherMapId,
    createWorldMapEdgeLockReason,
    getWorldMapAdjacentEdges,
    createWorldMapPathMap,
    createWorldMapPath,
    isWorldMapEdgeCompleted,
    createWorldMapNodeCompletion
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.worldMap = Object.assign({}, modules.worldMap || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
