(function initProjectStarfallDataMapTown(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataWorld = (typeof require === 'function' ? require('./world.js') : null) || DataModules.world || {};
  const DataShopVendors = (typeof require === 'function' ? require('./shop-vendors.js') : null) || DataModules.shopVendors || {};
  const DataMapLayouts = (typeof require === 'function' ? require('./map-layouts.js') : null) || DataModules.mapLayouts || {};
  const DataMapPresentation = (typeof require === 'function' ? require('./map-presentation.js') : null) || DataModules.mapPresentation || {};

  function freezeSceneEntries(entries) {
    return Object.freeze((entries || []).map((entry) => Object.freeze(Object.assign({}, entry))));
  }

  function defaultCreateTownScene(config) {
    const source = config || {};
    return Object.freeze({
      rearStructures: freezeSceneEntries(source.rearStructures),
      stationFacades: freezeSceneEntries(source.stationFacades),
      streetProps: freezeSceneEntries(source.streetProps),
      foregroundTrim: freezeSceneEntries(source.foregroundTrim)
    });
  }

  function defaultShopTypeTitle(typeId) {
    return String(typeId || '')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (letter) => letter.toUpperCase())
      .replace(/\s+/g, '');
  }

  function defaultGetTownShopInteriorMapId(townId, typeId) {
    return `${townId}${defaultShopTypeTitle(typeId)}Shop`;
  }

  function defaultGetTownStationPlacement(stationId) {
    const placements = {
      storage: { x: 360, platformIndex: 0 },
      shop: { x: 620, platformIndex: 0 },
      slots: { x: 1040, platformIndex: 2 },
      upgrade: { x: 1420, platformIndex: 2 },
      class: { x: 1290, platformIndex: 10 },
      plinko: { x: 2260, platformIndex: 5 }
    };
    return placements[stationId] || { x: 430, platformIndex: 0 };
  }

  function townServiceNpcId(mapId, serviceId) {
    const prefix = String(mapId || 'town')
      .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
      .replace(/[^a-zA-Z0-9_]+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '')
      .toLowerCase();
    return `${prefix}_${serviceId}_host`;
  }

  const REGIONAL_TOWN_IDS = DataWorld.REGIONAL_TOWN_IDS || Object.freeze([]);
  const SHOP_INTERIOR_WORLD_WIDTH = DataWorld.SHOP_INTERIOR_WORLD_WIDTH || 1280;
  const SHOP_VENDOR_TYPES = DataShopVendors.SHOP_VENDOR_TYPES || Object.freeze([]);
  const TOWN_SHOP_THEME_BY_TOWN = DataShopVendors.TOWN_SHOP_THEME_BY_TOWN || Object.freeze({});
  const shopTypeTitle = DataShopVendors.shopTypeTitle || defaultShopTypeTitle;
  const getTownShopInteriorMapId = DataShopVendors.getTownShopInteriorMapId || defaultGetTownShopInteriorMapId;
  const getTownShopVendorId = DataShopVendors.getTownShopVendorId || ((townId, typeId) => townServiceNpcId(townId, `${typeId}_shop`));
  const getTownStationPlacement = DataMapLayouts.getTownStationPlacement || defaultGetTownStationPlacement;
  const createTownScene = DataMapPresentation.createTownScene || defaultCreateTownScene;

  function createPlinkoHostNpc(mapId, options) {
    const settings = options || {};
    return {
      id: settings.id || townServiceNpcId(mapId, 'plinko'),
      name: settings.name || 'Plinko Host',
      xOverride: Number.isFinite(Number(settings.xOverride)) ? Number(settings.xOverride) : 2260,
      platformIndex: Number.isFinite(Number(settings.platformIndex)) ? Number(settings.platformIndex) : 5,
      questIds: [],
      servicePanelId: 'plinko',
      serviceStationId: 'plinko',
      serviceLabel: 'Starfall Plinko',
      serviceSummary: 'Spend coins or drop mob-earned balls into Starfall Plinko for weighted rewards and 100-ball jackpot pity.',
      color: settings.color || '#5a4fb3',
      accent: settings.accent || '#ffd166'
    };
  }

  function withTownServiceNpcs(mapId, questNpcs, options) {
    return [createPlinkoHostNpc(mapId, options && options.plinkoHost)].concat(questNpcs || []);
  }

  function assignTownStations(stations) {
    return (stations || []).map((station) => {
      const placement = getTownStationPlacement(station.id);
      return Object.assign({}, station, {
        x: Number.isFinite(Number(station.xOverride)) ? Number(station.xOverride) : placement.x,
        platformIndex: Number.isFinite(Number(station.platformIndex)) ? Number(station.platformIndex) : placement.platformIndex
      });
    });
  }

  function assignTownQuestNpcs(questNpcs) {
    return (questNpcs || []).map((npc, index) => Object.assign({}, npc, {
      x: Number.isFinite(Number(npc.xOverride)) ? Number(npc.xOverride) : index % 2 ? 2140 : 3000,
      platformIndex: Number.isFinite(Number(npc.platformIndex)) && Number(npc.platformIndex) > 0
        ? Number(npc.platformIndex)
        : index % 2 ? 5 : 6
    }));
  }

  function createShopInteriorScene(config) {
    const theme = config.theme || {};
    const type = config.type || {};
    const structureCell = type.id === 'special'
      ? theme.specialCell || 'lanternArch'
      : type.id === 'weapon'
        ? theme.weaponCell || 'rustcoilWorkshop'
        : type.id === 'armor'
          ? theme.armorCell || 'marketAwning'
          : theme.supplyCell || 'marketAwning';
    return createTownScene({
      rearStructures: [
        { cell: structureCell, x: 300, w: 560, h: 286, footOffset: 4, label: config.name || type.label || 'Shop' },
        { cell: 'marketAwning', x: 820, w: 280, h: 140, footOffset: 4, label: 'Counter' }
      ],
      stationFacades: [],
      streetProps: [
        { kind: type.id === 'supply' ? 'crate' : type.id === 'special' ? 'glow' : 'sign', x: 245, w: 44, h: 48, footOffset: 1 },
        { kind: type.id === 'weapon' ? 'crystal' : 'crate', x: 980, w: 44, h: 38, footOffset: 1 }
      ],
      foregroundTrim: []
    });
  }

  function createShopVendorNpc(townId, type) {
    const theme = TOWN_SHOP_THEME_BY_TOWN[townId] || {};
    const prefix = theme.prefix || 'Town';
    const vendorNames = {
      weapon: `${prefix} Smith`,
      armor: `${prefix} Armorer`,
      supply: `${prefix} Supplier`,
      special: theme.specialName || `${prefix} Specialist`
    };
    return {
      id: getTownShopVendorId(townId, type.id),
      name: vendorNames[type.id] || `${prefix} Vendor`,
      x: 680,
      platformIndex: 0,
      questIds: [],
      servicePanelId: 'shop',
      serviceStationId: 'shop',
      serviceLabel: vendorNames[type.id] || `${prefix} Vendor`,
      shopVendorId: getTownShopVendorId(townId, type.id),
      color: theme.vendorColor || '#5e7d9f',
      accent: theme.vendorAccent || '#ffd166'
    };
  }

  function createShopInteriorMap(townId, type) {
    const townTheme = TOWN_SHOP_THEME_BY_TOWN[townId] || {};
    const townName = townTheme.prefix || shopTypeTitle(townId);
    const vendor = createShopVendorNpc(townId, type);
    const name = `${townName} ${type.label}`;
    const width = SHOP_INTERIOR_WORLD_WIDTH;
    return {
      id: getTownShopInteriorMapId(townId, type.id),
      name,
      levelRange: [1, 99],
      safeZone: true,
      shopInterior: true,
      adminOnly: true,
      parentTownId: townId,
      shopVendorType: type.id,
      layoutRole: 'town',
      routeStage: 'Shop Interior',
      mapRoadName: name,
      portalPattern: 'shopDoorReturn',
      compactWorldWidth: width,
      palette: ['#fbfaf6', '#d8b74a', '#7bdff2'],
      purpose: `${name} interior with vendor-only buying and selling.`,
      enemies: [],
      platforms: [[0, 520, width, 80]],
      climbables: [],
      spawnPoints: [{ id: `${townId}_${type.id}_shop_spawn`, x: 180, platformIndex: 0, weight: 1 }],
      stations: [],
      questNpcs: [vendor],
      portals: [
        { id: `${townId}_${type.id}_shop_exit`, label: `${townName} Return`, destinationMapId: townId, returnPortal: true, x: 110, platformIndex: 0 }
      ],
      townScene: createShopInteriorScene({ name, theme: townTheme, type })
    };
  }

  function createShopInteriorMaps() {
    return REGIONAL_TOWN_IDS.flatMap((townId) => SHOP_VENDOR_TYPES.map((type) => createShopInteriorMap(townId, type)));
  }

  function makeTownHubMap(config) {
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange || [1, 99],
      safeZone: true,
      layoutRole: 'town',
      palette: config.palette,
      purpose: config.purpose,
      enemies: [],
      platforms: [[0, 520, 3000, 80]],
      climbables: [],
      spawnPoints: [],
      stations: [
        { id: 'storage', name: 'Storage Keeper', x: 430 },
        { id: 'shop', name: 'Shopkeeper', x: 440 },
        { id: 'slots', name: 'Slot Broker', x: 1060 },
        { id: 'upgrade', name: 'Upgrade Artisan', x: 1380 }
      ],
      questNpcs: withTownServiceNpcs(config.id, config.questNpcs || [], config)
    };
  }

  const api = Object.freeze({
    townServiceNpcId,
    createPlinkoHostNpc,
    withTownServiceNpcs,
    assignTownStations,
    assignTownQuestNpcs,
    createShopInteriorScene,
    createShopVendorNpc,
    createShopInteriorMap,
    createShopInteriorMaps,
    makeTownHubMap
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapTown = Object.assign({}, modules.mapTown || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
