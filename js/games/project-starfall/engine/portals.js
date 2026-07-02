(function initProjectStarfallEnginePortals(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreGeometry = (typeof require === 'function' ? require('../core/geometry.js') : null) || global.ProjectStarfallCore || {};
  const EngineModules = global.ProjectStarfallEngineModules || {};
  const EngineViewport = (typeof require === 'function' ? require('./viewport.js') : null) || EngineModules.viewport || {};
  const RouteProgress = (typeof require === 'function' ? require('./route-progress.js') : null) || EngineModules.routeProgress || {};
  const DungeonHelpers = (typeof require === 'function' ? require('./dungeons.js') : null) || EngineModules.dungeons || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const isSlopePlatform = CoreGeometry.isSlopePlatform || function isSlopePlatformFallback(platform) {
    return !!(platform && platform.shape === 'slope' && Number.isFinite(Number(platform.y2)));
  };
  const getPlatformSurfaceY = CoreGeometry.getPlatformSurfaceY || function getPlatformSurfaceYFallback(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!isSlopePlatform(platform)) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  };

  const dungeonDefinitionByMapIdCache = new WeakMap();
  const DEFAULT_PORTAL_MIN_CENTER_SPACING = 124;

  function getPortalData(options) {
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
    const data = getPortalData(options);
    return getById(data.WORLD_ROUTES || [], normalizeId(routeId));
  }

  function getRouteForFieldMap(mapId, options) {
    if (RouteProgress.getRouteForFieldMap) return RouteProgress.getRouteForFieldMap(mapId, options);
    const data = getPortalData(options);
    return (data.WORLD_ROUTES || []).find((route) => (route.fieldGoals || []).some((field) => field.mapId === mapId)) || null;
  }

  function getRouteForBossMap(mapId, options) {
    if (RouteProgress.getRouteForBossMap) return RouteProgress.getRouteForBossMap(mapId, options);
    const data = getPortalData(options);
    return (data.WORLD_ROUTES || []).find((route) => route.bossMapId === mapId) || null;
  }

  function getRouteForDungeon(dungeonId, options) {
    if (RouteProgress.getRouteForDungeon) return RouteProgress.getRouteForDungeon(dungeonId, options);
    const data = getPortalData(options);
    return (data.WORLD_ROUTES || []).find((route) => route.bossDungeonId === dungeonId) || null;
  }

  function createRouteFieldStatus(route, field, routeState, options) {
    if (RouteProgress.createRouteFieldStatus) {
      return RouteProgress.createRouteFieldStatus(route, field, routeState, options);
    }
    const data = getPortalData(options);
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

  function createRouteSummary(route, routeState, options) {
    if (RouteProgress.createRouteSummary) return RouteProgress.createRouteSummary(route, routeState, options);
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

  function createDungeonStartBlockReason(dungeon, player) {
    if (DungeonHelpers.createDungeonStartBlockReason) {
      return DungeonHelpers.createDungeonStartBlockReason(dungeon, player);
    }
    const activePlayer = player || {};
    if (!dungeon) return 'Dungeon is unavailable.';
    if (!activePlayer.classId) return 'Choose a class first.';
    if (activePlayer.level < Number(dungeon.levelRequirement || 1)) return `Level ${dungeon.levelRequirement} required.`;
    if (dungeon.requiresAdvancedClass && !activePlayer.advancedClassId) return 'Choose an advanced class first.';
    return '';
  }

  function createRouteBossLockReason(route, routeState, options) {
    const summary = createRouteSummary(route, getRouteState(routeState, options), options);
    if (!summary || summary.complete) return '';
    const firstOpen = summary.fields.find((field) => !field.complete);
    return firstOpen ? `${summary.name}: clear ${firstOpen.mapName} (${firstOpen.value}/${firstOpen.goal}).` : `${summary.name} objectives are incomplete.`;
  }

  function createMapTravelBlockReason(mapId, options) {
    const settings = options || {};
    const data = getPortalData(settings);
    const map = getMapDefinitionById(mapId, data);
    if (!map) return 'Area is unavailable.';
    if (map.id === settings.currentMapId || map.safeZone) return '';
    const player = settings.player || {};
    if (!player.classId) return 'Choose a class first.';
    const routeState = getRouteState(settings.routeState, settings);
    if (map.isDungeon) {
      const dungeon = getDungeonDefinitionByMapId(map.id, data);
      const dungeonReason = createDungeonStartBlockReason(dungeon, player);
      if (dungeonReason) return dungeonReason;
      const route = dungeon ? getRouteForDungeon(dungeon.id, settings) : getRouteForBossMap(map.id, settings);
      return route ? createRouteBossLockReason(route, routeState, settings) : '';
    }
    const route = getRouteForFieldMap(map.id, settings);
    if (!route) return '';
    const fields = route.fieldGoals || [];
    const fieldIndex = fields.findIndex((field) => field.mapId === map.id);
    if (fieldIndex <= 0) return '';
    const blocker = fields.slice(0, fieldIndex)
      .map((field) => createRouteFieldStatus(route, field, routeState, settings))
      .find((field) => !field.complete);
    return blocker ? `${route.name}: clear ${blocker.mapName} (${blocker.value}/${blocker.goal}).` : '';
  }

  function createPortalBlockReason(portal, options) {
    const settings = options || {};
    const data = getPortalData(settings);
    const player = settings.player || {};
    if (!portal) return 'Portal is unavailable.';
    if (portal.returnPortal) return '';
    if (portal.requiredLevel && player.level < Number(portal.requiredLevel)) return `Level ${portal.requiredLevel} required.`;
    if (portal.requiredDungeonId) {
      const dungeons = getDungeonState(settings.dungeons, settings);
      if (!(dungeons.completedDungeonIds || []).includes(portal.requiredDungeonId)) {
        const dungeon = getDungeonDefinitionById(portal.requiredDungeonId, data);
        return `Clear ${dungeon ? dungeon.name : portal.requiredDungeonId} first.`;
      }
    }
    const routeState = getRouteState(settings.routeState, settings);
    if (portal.dungeonId) {
      const dungeon = getDungeonDefinitionById(portal.dungeonId, data);
      const dungeonReason = createDungeonStartBlockReason(dungeon, player);
      if (dungeonReason) return dungeonReason;
      const route = getRouteForDungeon(portal.dungeonId, settings);
      return route ? createRouteBossLockReason(route, routeState, settings) : '';
    }
    if (portal.requiredMapId && portal.routeId) {
      const route = getWorldRoute(portal.routeId, settings);
      const field = route && (route.fieldGoals || []).find((goal) => goal.mapId === portal.requiredMapId);
      const status = field ? createRouteFieldStatus(route, field, routeState, settings) : null;
      if (status && !status.complete) return `${route.name}: clear ${status.mapName} (${status.value}/${status.goal}).`;
    }
    if (portal.destinationMapId) return createMapTravelBlockReason(portal.destinationMapId, settings);
    return 'Portal destination is unavailable.';
  }

  function createPortalSummary(portal, options) {
    if (!portal) return null;
    const settings = options || {};
    const data = getPortalData(settings);
    const destinationMap = getMapDefinitionById(portal.destinationMapId, data);
    const dungeon = getDungeonDefinitionById(portal.dungeonId, data);
    const lockedReason = createPortalBlockReason(portal, settings);
    return Object.assign({}, portal, {
      destinationName: dungeon ? dungeon.name : destinationMap ? destinationMap.name : '',
      lockedReason,
      locked: !!lockedReason
    });
  }

  function alignPortal(rawPortal, index, mapId, platforms, options) {
    const settings = options || {};
    const portal = rawPortal || {};
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const platformIndex = clamp(Number(portal.platformIndex || 0), 0, Math.max(0, sourcePlatforms.length - 1));
    const platform = sourcePlatforms[platformIndex] || sourcePlatforms[0] || {
      x: 0,
      y: Number(settings.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640),
      w: 2200,
      h: 80,
      id: `${mapId}_platform_0`,
      index: 0
    };
    const defaultW = portal.shopDoor ? 94 : 58;
    const defaultH = portal.shopDoor ? 118 : 86;
    const w = Math.max(portal.shopDoor ? 72 : 46, Number(portal.w || defaultW) || defaultW);
    const h = Math.max(portal.shopDoor ? 96 : 64, Number(portal.h || defaultH) || defaultH);
    const x = clamp(Number(portal.x || platform.x + platform.w - 96), platform.x + 18, platform.x + platform.w - w - 18);
    const surfaceY = getPlatformSurfaceY(platform, x + w / 2);
    return Object.assign({}, portal, {
      id: portal.id || `${mapId}_portal_${index}`,
      label: portal.label || 'Portal',
      x,
      y: surfaceY - h,
      w,
      h,
      platformIndex,
      platformId: platform.id,
      mapId
    });
  }

  function distributePortalCluster(cluster, platform, options) {
    const settings = options || {};
    if (!cluster || cluster.length <= 1 || !platform) return;
    const minCenterSpacing = Math.max(0, Number(settings.minCenterSpacing || DEFAULT_PORTAL_MIN_CENTER_SPACING));
    const gap = Math.max(72, minCenterSpacing);
    const widest = cluster.reduce((width, portal) => Math.max(width, Number(portal.w || 0)), 58);
    const minCenter = platform.x + 18 + widest / 2;
    const maxCenter = platform.x + platform.w - 18 - widest / 2;
    const available = Math.max(0, maxCenter - minCenter);
    const spacing = cluster.length > 1 ? Math.min(gap, available / (cluster.length - 1) || gap) : 0;
    const span = spacing * (cluster.length - 1);
    const average = cluster.reduce((total, portal) => total + Number(portal.x || 0) + Number(portal.w || 0) / 2, 0) / cluster.length;
    const start = clamp(average - span / 2, minCenter, Math.max(minCenter, maxCenter - span));
    cluster.forEach((portal, clusterIndex) => {
      const center = start + spacing * clusterIndex;
      portal.x = clamp(center - Number(portal.w || widest) / 2, platform.x + 18, platform.x + platform.w - Number(portal.w || widest) - 18);
    });
  }

  function distributeOverlappingPortals(portals, platforms, options) {
    const settings = options || {};
    if (!Array.isArray(portals) || portals.length <= 1) return portals || [];
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const platformById = new Map(sourcePlatforms.map((platform) => [platform.id, platform]));
    const groups = new Map();
    portals.forEach((portal, index) => {
      if (!portal) return;
      portal.__portalOrder = index;
      const key = portal.platformId || `${portal.platformIndex || 0}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(portal);
    });
    groups.forEach((group, key) => {
      const platform = platformById.get(key) || sourcePlatforms[Number(group[0] && group[0].platformIndex || 0)];
      const sorted = group.slice().sort((a, b) =>
        (Number(a.x || 0) + Number(a.w || 0) / 2) - (Number(b.x || 0) + Number(b.w || 0) / 2) ||
        Number(a.__portalOrder || 0) - Number(b.__portalOrder || 0)
      );
      let cluster = [];
      sorted.forEach((portal) => {
        const center = Number(portal.x || 0) + Number(portal.w || 0) / 2;
        const previous = cluster.length ? cluster[cluster.length - 1] : null;
        const previousCenter = previous ? Number(previous.x || 0) + Number(previous.w || 0) / 2 : -Infinity;
        if (!cluster.length || center - previousCenter < Math.max(0, Number(settings.minCenterSpacing || DEFAULT_PORTAL_MIN_CENTER_SPACING))) {
          cluster.push(portal);
          return;
        }
        distributePortalCluster(cluster, platform, settings);
        cluster = [portal];
      });
      distributePortalCluster(cluster, platform, settings);
    });
    portals.forEach((portal) => {
      if (portal) delete portal.__portalOrder;
    });
    return portals;
  }

  const api = {
    DEFAULT_PORTAL_MIN_CENTER_SPACING,
    createRouteBossLockReason,
    createMapTravelBlockReason,
    createPortalBlockReason,
    createPortalSummary,
    alignPortal,
    distributeOverlappingPortals
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.portals = Object.assign({}, modules.portals || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
