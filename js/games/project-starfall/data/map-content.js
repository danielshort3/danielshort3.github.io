(function initProjectStarfallDataMapContent(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataEnvironment = (typeof require === 'function' ? require('./environment.js') : null) || DataModules.environment || {};
  const DataWorld = (typeof require === 'function' ? require('./world.js') : null) || DataModules.world || {};
  const DataShopVendors = (typeof require === 'function' ? require('./shop-vendors.js') : null) || DataModules.shopVendors || {};
  const DataMapSizing = (typeof require === 'function' ? require('./map-sizing.js') : null) || DataModules.mapSizing || {};
  const DataMapPresentation = (typeof require === 'function' ? require('./map-presentation.js') : null) || DataModules.mapPresentation || {};
  const DataMapPortals = (typeof require === 'function' ? require('./map-portals.js') : null) || DataModules.mapPortals || {};
  const DataMapPublication = (typeof require === 'function' ? require('./map-publication.js') : null) || DataModules.mapPublication || {};
  const DataMapCatalog = (typeof require === 'function' ? require('./map-catalog.js') : null) || DataModules.mapCatalog || {};

  function createMapContentData(options) {
    const settings = options || {};
    const MAP_ASSETS = settings.MAP_ASSETS || DataAssets.MAP_ASSETS;
    const STATION_ASSETS = settings.STATION_ASSETS || DataAssets.STATION_ASSETS;
    const MAP_ENVIRONMENT_PROFILES = settings.MAP_ENVIRONMENT_PROFILES || DataEnvironment.MAP_ENVIRONMENT_PROFILES;
    const WORLD_AREAS = settings.WORLD_AREAS || DataWorld.WORLD_AREAS;
    const WORLD_ROUTES = settings.WORLD_ROUTES || DataWorld.WORLD_ROUTES;
    const REGIONAL_TOWN_IDS = settings.REGIONAL_TOWN_IDS || DataWorld.REGIONAL_TOWN_IDS;
    const SHOP_INTERIOR_WORLD_WIDTH = settings.SHOP_INTERIOR_WORLD_WIDTH || DataWorld.SHOP_INTERIOR_WORLD_WIDTH;
    const WORLD_MAP_NODES = settings.WORLD_MAP_NODES || DataWorld.WORLD_MAP_NODES;
    const WORLD_MAP_EDGES = settings.WORLD_MAP_EDGES || DataWorld.WORLD_MAP_EDGES;
    const SHOP_VENDOR_TYPES = settings.SHOP_VENDOR_TYPES || DataShopVendors.SHOP_VENDOR_TYPES;
    const getTownShopVendorId = settings.getTownShopVendorId || DataShopVendors.getTownShopVendorId;
    const createTownShopDoorPortals = settings.createTownShopDoorPortals || DataShopVendors.createTownShopDoorPortals;
    const getAuthoredMapWidth = settings.getAuthoredMapWidth || DataMapSizing.getAuthoredMapWidth;
    const createMapPresentationData = settings.createMapPresentationData || DataMapPresentation.createMapPresentationData;
    const createMapPortalData = settings.createMapPortalData || DataMapPortals.createMapPortalData;
    const createMapPublicationData = settings.createMapPublicationData || DataMapPublication.createMapPublicationData;
    const createMapCatalogData = settings.createMapCatalogData || DataMapCatalog.createMapCatalogData;

    const mapPresentationData = createMapPresentationData({
      getAuthoredMapWidth
    });
    const mapPortalData = createMapPortalData({
      createTownShopDoorPortals
    });
    const MAP_PORTALS = mapPortalData.MAP_PORTALS;

    const mapPublicationData = createMapPublicationData({
      MAP_ASSETS,
      STATION_ASSETS,
      MAP_ENVIRONMENT_PROFILES,
      WORLD_AREAS,
      WORLD_MAP_NODES,
      MAP_PORTALS,
      mapPresentationData
    });
    const attachMapAssets = mapPublicationData.attachMapAssets;
    const mapCatalogData = createMapCatalogData({
      attachMapAssets
    });

    return Object.freeze({
      WORLD_AREAS,
      WORLD_ROUTES,
      REGIONAL_TOWN_IDS,
      SHOP_INTERIOR_WORLD_WIDTH,
      WORLD_MAP_NODES,
      WORLD_MAP_EDGES,
      SHOP_VENDOR_TYPES,
      getTownShopVendorId,
      createTownShopDoorPortals,
      mapPresentationData,
      MAP_LAYOUT_ROLES: mapPresentationData.MAP_LAYOUT_ROLES,
      MAP_LAYOUT_ROLE_LABELS: mapPresentationData.MAP_LAYOUT_ROLE_LABELS,
      MAP_LAYOUT_BLUEPRINTS: mapPresentationData.MAP_LAYOUT_BLUEPRINTS,
      MAP_TOWN_SCENES: mapPresentationData.MAP_TOWN_SCENES,
      MAP_FIELD_COMPOSITIONS: mapPresentationData.MAP_FIELD_COMPOSITIONS,
      MAP_DESIGN_INTENTS: mapPresentationData.MAP_DESIGN_INTENTS,
      MAP_MECHANIC_DEFINITIONS: mapPresentationData.MAP_MECHANIC_DEFINITIONS,
      TOWN_SERVICE_PLANS: mapPresentationData.TOWN_SERVICE_PLANS,
      MAP_PORTAL_FICTION: mapPresentationData.MAP_PORTAL_FICTION,
      MAP_PORTALS,
      attachMapAssets,
      MAPS: mapCatalogData.MAPS
    });
  }

  const defaultMapContentData = createMapContentData();
  const api = Object.assign({
    createMapContentData
  }, defaultMapContentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapContent = Object.assign({}, modules.mapContent || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
