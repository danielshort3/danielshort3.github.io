(function initProjectStarfallDataMapPublication(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataEnvironment = (typeof require === 'function' ? require('./environment.js') : null) || DataModules.environment || {};
  const DataWorld = (typeof require === 'function' ? require('./world.js') : null) || DataModules.world || {};
  const DataMapSizing = (typeof require === 'function' ? require('./map-sizing.js') : null) || DataModules.mapSizing || {};
  const DataMapPresentation = (typeof require === 'function' ? require('./map-presentation.js') : null) || DataModules.mapPresentation || {};
  const DataMapPortals = (typeof require === 'function' ? require('./map-portals.js') : null) || DataModules.mapPortals || {};

  const EMPTY_OBJECT = Object.freeze({});
  const EMPTY_ARRAY = Object.freeze([]);

  function attachAsset(record, asset) {
    return Object.freeze(Object.assign({}, record, { asset: asset || '' }));
  }

  function createDefaultMapPresentationData(settings) {
    if (!DataMapPresentation.createMapPresentationData) {
      return DataMapPresentation || EMPTY_OBJECT;
    }
    return DataMapPresentation.createMapPresentationData({
      getAuthoredMapWidth: settings.getAuthoredMapWidth || DataMapSizing.getAuthoredMapWidth
    });
  }

  function createMapPublicationData(options) {
    const settings = options || {};
    const mapPresentationData = settings.mapPresentationData || createDefaultMapPresentationData(settings);
    const mapLayoutRoles = settings.MAP_LAYOUT_ROLES || mapPresentationData.MAP_LAYOUT_ROLES || EMPTY_OBJECT;
    const mapLayoutRoleLabels = settings.MAP_LAYOUT_ROLE_LABELS || mapPresentationData.MAP_LAYOUT_ROLE_LABELS || EMPTY_OBJECT;
    const normalizeMapLayoutRole = settings.normalizeMapLayoutRole || mapPresentationData.normalizeMapLayoutRole || ((roleId, fallback) => roleId || fallback || 'trainingField');
    const getMapLayoutRoleFallback = settings.getMapLayoutRoleFallback || mapPresentationData.getMapLayoutRoleFallback || ((map) => map && map.safeZone ? 'town' : map && map.bossRoom ? 'bossArena' : map && map.isDungeon ? 'dungeon' : 'trainingField');
    const mapLayoutBlueprints = settings.MAP_LAYOUT_BLUEPRINTS || mapPresentationData.MAP_LAYOUT_BLUEPRINTS || EMPTY_OBJECT;
    const mapTownScenes = settings.MAP_TOWN_SCENES || mapPresentationData.MAP_TOWN_SCENES || EMPTY_OBJECT;
    const mapFieldCompositions = settings.MAP_FIELD_COMPOSITIONS || mapPresentationData.MAP_FIELD_COMPOSITIONS || EMPTY_OBJECT;
    const mapDesignIntents = settings.MAP_DESIGN_INTENTS || mapPresentationData.MAP_DESIGN_INTENTS || EMPTY_OBJECT;
    const mapPortalFiction = settings.MAP_PORTAL_FICTION || mapPresentationData.MAP_PORTAL_FICTION || EMPTY_OBJECT;
    const createDefaultTownScene = settings.createDefaultTownScene || mapPresentationData.createDefaultTownScene || (() => null);
    const createDefaultFieldComposition = settings.createDefaultFieldComposition || mapPresentationData.createDefaultFieldComposition || (() => null);
    const createDesignIntent = settings.createDesignIntent || mapPresentationData.createDesignIntent || ((config) => Object.freeze(Object.assign({}, config || EMPTY_OBJECT)));
    const getTownServicePlan = settings.getTownServicePlan || mapPresentationData.getTownServicePlan || (() => null);
    const getStationServiceIntent = settings.getStationServiceIntent || mapPresentationData.getStationServiceIntent || (() => EMPTY_OBJECT);
    const createSpawnSections = settings.createSpawnSections || mapPresentationData.createSpawnSections || (() => EMPTY_ARRAY);
    const attachSpawnSectionsToPoints = settings.attachSpawnSectionsToPoints || mapPresentationData.attachSpawnSectionsToPoints || ((map) => Object.freeze((map.spawnPoints || EMPTY_ARRAY).slice()));
    const worldAreas = settings.WORLD_AREAS || DataWorld.WORLD_AREAS || EMPTY_ARRAY;
    const worldMapNodes = settings.WORLD_MAP_NODES || DataWorld.WORLD_MAP_NODES || EMPTY_ARRAY;
    const mapAssets = settings.MAP_ASSETS || DataAssets.MAP_ASSETS || EMPTY_OBJECT;
    const stationAssets = settings.STATION_ASSETS || DataAssets.STATION_ASSETS || EMPTY_OBJECT;
    const mapEnvironmentProfiles = settings.MAP_ENVIRONMENT_PROFILES || DataEnvironment.MAP_ENVIRONMENT_PROFILES || EMPTY_OBJECT;
    const mapPortals = settings.MAP_PORTALS || DataMapPortals.MAP_PORTALS || EMPTY_OBJECT;

    function attachMapAssets(map) {
      const node = worldMapNodes.find((item) => item && item.mapId === map.id);
      const areaId = map.areaId || node && node.areaId || '';
      const area = areaId ? worldAreas.find((item) => item && item.id === areaId) : null;
      const blueprint = mapLayoutBlueprints[map.id] || EMPTY_OBJECT;
      const layoutRole = normalizeMapLayoutRole(blueprint.role || map.layoutRole || node && node.role, getMapLayoutRoleFallback(map));
      const townScene = map.safeZone ? map.townScene || mapTownScenes[map.id] || createDefaultTownScene(map) : null;
      const fieldComposition = !map.safeZone ? mapFieldCompositions[map.id] || createDefaultFieldComposition(map, blueprint) : null;
      const portalRoles = fieldComposition && fieldComposition.portalRoles || EMPTY_OBJECT;
      const designIntent = !map.safeZone && !map.shopInterior && !map.adminOnly
        ? mapDesignIntents[map.id] || createDesignIntent({
            intendedArchetype: map.isDungeon ? 'H arena-style map' : 'F loop map',
            intendedUseCase: map.isDungeon ? 'boss dungeon' : 'solo/duo',
            routeSummary: `Clear ${map.name || map.id} in a repeatable route and return as enemies repopulate.`,
            visualIdentityTag: map.name || map.id
          })
        : null;
      const spawnSections = createSpawnSections(map, fieldComposition, designIntent);
      const townServicePlan = map.safeZone && !map.shopInterior ? getTownServicePlan(map.id) : null;
      return Object.freeze(Object.assign({}, map, {
        areaId,
        areaName: area ? area.name : node && node.region || '',
        areaMechanic: area ? area.mechanic : '',
        layoutRole,
        layoutRoleLabel: mapLayoutRoleLabels[layoutRole] || 'Training Field',
        layoutMarker: mapLayoutRoles[layoutRole] && mapLayoutRoles[layoutRole].marker || '',
        routeStage: map.routeStage || blueprint.routeStage || '',
        mapRoadName: map.mapRoadName || blueprint.roadName || map.name || '',
        landmark: map.landmark || blueprint.landmark || '',
        portalPattern: map.portalPattern || blueprint.portalPattern || '',
        designIntent,
        spawnSections,
        townServicePlan,
        townScene,
        fieldComposition,
        asset: mapAssets[map.id] || '',
        environment: mapEnvironmentProfiles[map.id] || mapEnvironmentProfiles.greenrootMeadow,
        spawnPoints: attachSpawnSectionsToPoints(map, spawnSections),
        stations: (map.stations || []).map((station) => attachAsset(Object.assign({}, getStationServiceIntent(station.id), station), stationAssets[station.id])),
        questNpcs: (map.questNpcs || []).map((npc) => Object.freeze(Object.assign({}, npc, {
          questIds: Object.freeze((npc.questIds || []).slice())
        }))),
        portals: (map.portals || mapPortals[map.id] || []).map((portal) => {
          const portalFiction = mapPortalFiction[portal.id] || EMPTY_OBJECT;
          return Object.freeze(Object.assign({}, portal, {
            roleLabel: portal.roleLabel || portalRoles[portal.id] || portalFiction.roleLabel || '',
            portalStyle: portal.portalStyle || portalFiction.portalStyle || ''
          }));
        })
      }));
    }

    return Object.freeze({
      attachMapAssets
    });
  }

  const defaultMapPublicationData = createMapPublicationData();
  const api = Object.assign({
    attachAsset,
    createMapPublicationData
  }, defaultMapPublicationData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapPublication = Object.assign({}, modules.mapPublication || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
