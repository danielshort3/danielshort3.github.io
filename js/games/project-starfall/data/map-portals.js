(function initProjectStarfallDataMapPortals(global) {
  'use strict';

  function defaultCreateTownShopDoorPortals() {
    return [];
  }

  function createMapPortalData(options) {
    const settings = options || {};
    const createTownShopDoorPortals =
      typeof settings.createTownShopDoorPortals === 'function'
        ? settings.createTownShopDoorPortals
        : defaultCreateTownShopDoorPortals;

    const MAP_PORTALS = Object.freeze({
      starfallCrossing: Object.freeze([
        ...createTownShopDoorPortals('starfallCrossing'),
        Object.freeze({
          id: 'crossing_greenroot',
          label: 'Greenroot Gate',
          destinationMapId: 'greenrootMeadow',
          routeId: 'forest',
          x: 2040,
          platformIndex: 0,
        }),
      ]),
      greenrootMeadow: Object.freeze([
        Object.freeze({
          id: 'greenroot_crossing',
          label: 'Town Return',
          destinationMapId: 'starfallCrossing',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'greenroot_thornpath',
          label: 'Thornpath Pass',
          destinationMapId: 'thornpathThicket',
          routeId: 'forest',
          requiredMapId: 'greenrootMeadow',
          x: 4080,
          platformIndex: 0,
        }),
      ]),
      thornpathThicket: Object.freeze([
        Object.freeze({
          id: 'thornpath_greenroot',
          label: 'Greenroot Return',
          destinationMapId: 'greenrootMeadow',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'thornpath_bandit',
          label: 'Bandit Ridge',
          destinationMapId: 'banditRidgeCamp',
          routeId: 'forest',
          requiredMapId: 'thornpathThicket',
          x: 7330,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'thornpath_rustcoil_outpost',
          label: 'Rustcoil Outpost',
          destinationMapId: 'rustcoilOutpost',
          routeId: 'forest',
          requiredMapId: 'thornpathThicket',
          x: 7460,
          platformIndex: 0,
        }),
      ]),
      banditRidgeCamp: Object.freeze([
        Object.freeze({
          id: 'bandit_thornpath',
          label: 'Thornpath Return',
          destinationMapId: 'thornpathThicket',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'bandit_bramble',
          label: 'Bramble Depths',
          dungeonId: 'bramble_depths',
          routeId: 'forest',
          bossPortal: true,
          x: 5280,
          platformIndex: 0,
        }),
      ]),
      brambleDepths: Object.freeze([
        Object.freeze({
          id: 'bramble_bandit',
          label: 'Ridge Return',
          destinationMapId: 'banditRidgeCamp',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      rustcoilOutpost: Object.freeze([
        ...createTownShopDoorPortals('rustcoilOutpost'),
        Object.freeze({
          id: 'rustcoil_outpost_thornpath',
          label: 'Thornpath Return',
          destinationMapId: 'thornpathThicket',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'rustcoil_outpost_ruins',
          label: 'Rustcoil Ruins',
          destinationMapId: 'rustcoilRuins',
          routeId: 'ruins',
          x: 2660,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'rustcoil_outpost_quarry',
          label: 'Oreback Quarry',
          destinationMapId: 'orebackQuarry',
          routeId: 'ruins',
          requiredMapId: 'rustcoilRuins',
          x: 2840,
          platformIndex: 0,
        }),
      ]),
      rustcoilRuins: Object.freeze([
        Object.freeze({
          id: 'rustcoil_outpost_return',
          label: 'Outpost Return',
          destinationMapId: 'rustcoilOutpost',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
      ]),
      orebackQuarry: Object.freeze([
        Object.freeze({
          id: 'quarry_rustcoil_outpost',
          label: 'Outpost Return',
          destinationMapId: 'rustcoilOutpost',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'quarry_vault',
          label: 'Gearworks Vault',
          dungeonId: 'gearworks_vault',
          routeId: 'ruins',
          bossPortal: true,
          x: 4440,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'quarry_cinder_refuge',
          label: 'Cinder Refuge',
          destinationMapId: 'cinderRefuge',
          routeId: 'ruins',
          requiredMapId: 'orebackQuarry',
          x: 4680,
          platformIndex: 0,
        }),
      ]),
      gearworksVault: Object.freeze([
        Object.freeze({
          id: 'vault_quarry',
          label: 'Quarry Return',
          destinationMapId: 'orebackQuarry',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      cinderRefuge: Object.freeze([
        ...createTownShopDoorPortals('cinderRefuge'),
        Object.freeze({
          id: 'cinder_refuge_quarry',
          label: 'Oreback Return',
          destinationMapId: 'orebackQuarry',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'cinder_refuge_hollow',
          label: 'Cinder Hollow',
          destinationMapId: 'cinderHollow',
          routeId: 'cinder',
          x: 2660,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'cinder_refuge_ashglass',
          label: 'Ashglass Pass',
          destinationMapId: 'ashglassPass',
          routeId: 'frostfen',
          requiredLevel: 40,
          requiredDungeonId: 'emberjaw_lair',
          x: 2840,
          platformIndex: 0,
        }),
      ]),
      cinderHollow: Object.freeze([
        Object.freeze({
          id: 'cinder_refuge_return',
          label: 'Refuge Return',
          destinationMapId: 'cinderRefuge',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'cinder_emberjaw',
          label: 'Emberjaw Lair',
          dungeonId: 'emberjaw_lair',
          routeId: 'cinder',
          bossPortal: true,
          x: 7580,
          platformIndex: 0,
        }),
      ]),
      emberjawLair: Object.freeze([
        Object.freeze({
          id: 'lair_cinder',
          label: 'Cinder Return',
          destinationMapId: 'cinderHollow',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      ashglassPass: Object.freeze([
        Object.freeze({
          id: 'ashglass_cinder_refuge',
          label: 'Refuge Return',
          destinationMapId: 'cinderRefuge',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'ashglass_frostfen_camp',
          label: 'Frostfen Camp',
          destinationMapId: 'frostfenCamp',
          routeId: 'frostfen',
          requiredMapId: 'ashglassPass',
          x: 8060,
          platformIndex: 0,
        }),
      ]),
      frostfenCamp: Object.freeze([
        ...createTownShopDoorPortals('frostfenCamp'),
        Object.freeze({
          id: 'frostfen_camp_ashglass',
          label: 'Ashglass Return',
          destinationMapId: 'ashglassPass',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'frostfen_camp_outskirts',
          label: 'Frostfen Tundra',
          destinationMapId: 'frostfenOutskirts',
          routeId: 'frostfen',
          x: 2660,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'frostfen_camp_glacier',
          label: 'Glacier Spine',
          destinationMapId: 'glacierSpine',
          routeId: 'frostfen',
          requiredMapId: 'frostfenOutskirts',
          x: 2840,
          platformIndex: 0,
        }),
      ]),
      frostfenOutskirts: Object.freeze([
        Object.freeze({
          id: 'frostfen_camp_return',
          label: 'Camp Return',
          destinationMapId: 'frostfenCamp',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'frostfen_glacier',
          label: 'Glacier Spine',
          destinationMapId: 'glacierSpine',
          routeId: 'frostfen',
          requiredMapId: 'frostfenOutskirts',
          x: 8060,
          platformIndex: 0,
        }),
      ]),
      glacierSpine: Object.freeze([
        Object.freeze({
          id: 'glacier_frostfen_camp',
          label: 'Camp Return',
          destinationMapId: 'frostfenCamp',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'glacier_sanctum',
          label: 'Rimewarden Sanctum',
          dungeonId: 'rimewarden_sanctum',
          routeId: 'frostfen',
          bossPortal: true,
          x: 7920,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'glacier_stormbreak_haven',
          label: 'Stormbreak Haven',
          destinationMapId: 'stormbreakHaven',
          routeId: 'ascension',
          requiredDungeonId: 'rimewarden_sanctum',
          x: 8060,
          platformIndex: 0,
        }),
      ]),
      rimewardenSanctum: Object.freeze([
        Object.freeze({
          id: 'sanctum_glacier',
          label: 'Glacier Return',
          destinationMapId: 'glacierSpine',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      stormbreakHaven: Object.freeze([
        ...createTownShopDoorPortals('stormbreakHaven'),
        Object.freeze({
          id: 'stormbreak_haven_glacier',
          label: 'Glacier Return',
          destinationMapId: 'glacierSpine',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'stormbreak_haven_cliffs',
          label: 'Stormbreak Cliffs',
          destinationMapId: 'stormbreakCliffs',
          routeId: 'ascension',
          x: 2660,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'stormbreak_haven_observatory',
          label: 'Astral Observatory',
          destinationMapId: 'astralObservatory',
          routeId: 'ascension',
          requiredMapId: 'stormbreakCliffs',
          x: 2840,
          platformIndex: 0,
        }),
      ]),
      stormbreakCliffs: Object.freeze([
        Object.freeze({
          id: 'stormbreak_haven_return',
          label: 'Haven Return',
          destinationMapId: 'stormbreakHaven',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
      ]),
      astralObservatory: Object.freeze([
        ...createTownShopDoorPortals('astralObservatory'),
        Object.freeze({
          id: 'astral_observatory_stormbreak',
          label: 'Stormbreak Return',
          destinationMapId: 'stormbreakHaven',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'astral_observatory_archive',
          label: 'Astral Archive',
          destinationMapId: 'astralArchive',
          routeId: 'ascension',
          x: 2660,
          platformIndex: 0,
        }),
      ]),
      astralArchive: Object.freeze([
        Object.freeze({
          id: 'archive_observatory',
          label: 'Observatory Return',
          destinationMapId: 'astralObservatory',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'archive_eclipse',
          label: 'Eclipse Frontier',
          destinationMapId: 'eclipseFrontier',
          routeId: 'ascension',
          requiredMapId: 'astralArchive',
          x: 8060,
          platformIndex: 0,
        }),
      ]),
      eclipseFrontier: Object.freeze([
        Object.freeze({
          id: 'eclipse_archive',
          label: 'Archive Return',
          destinationMapId: 'astralArchive',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
        Object.freeze({
          id: 'eclipse_rift',
          label: 'Endless Rift',
          destinationMapId: 'endlessRift',
          routeId: 'ascension',
          requiredMapId: 'eclipseFrontier',
          x: 8060,
          platformIndex: 0,
        }),
      ]),
      endlessRift: Object.freeze([
        Object.freeze({
          id: 'rift_eclipse',
          label: 'Eclipse Return',
          destinationMapId: 'eclipseFrontier',
          returnPortal: true,
          x: 110,
          platformIndex: 0,
        }),
      ]),
      bramblekingCourt: Object.freeze([
        Object.freeze({
          id: 'court_return',
          label: 'Ridge Return',
          destinationMapId: 'banditRidgeCamp',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      titanFoundry: Object.freeze([
        Object.freeze({
          id: 'foundry_return',
          label: 'Quarry Return',
          destinationMapId: 'orebackQuarry',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      deepcoreCore: Object.freeze([
        Object.freeze({
          id: 'deepcore_return',
          label: 'Quarry Return',
          destinationMapId: 'orebackQuarry',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      emberjawFurnace: Object.freeze([
        Object.freeze({
          id: 'furnace_return',
          label: 'Cinder Return',
          destinationMapId: 'cinderHollow',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      rimewardenVault: Object.freeze([
        Object.freeze({
          id: 'vault_return',
          label: 'Glacier Return',
          destinationMapId: 'glacierSpine',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      stormbreakAerie: Object.freeze([
        Object.freeze({
          id: 'aerie_return',
          label: 'Stormbreak Return',
          destinationMapId: 'stormbreakCliffs',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      astralStacks: Object.freeze([
        Object.freeze({
          id: 'stacks_return',
          label: 'Archive Return',
          destinationMapId: 'astralArchive',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
      eclipseThrone: Object.freeze([
        Object.freeze({
          id: 'throne_return',
          label: 'Frontier Return',
          destinationMapId: 'eclipseFrontier',
          returnPortal: true,
          x: 100,
          platformIndex: 0,
        }),
      ]),
    });

    return Object.freeze({
      MAP_PORTALS,
    });
  }

  const defaultMapPortalData = createMapPortalData();
  const api = Object.assign(
    {
      createMapPortalData,
    },
    defaultMapPortalData,
  );

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapPortals = Object.assign({}, modules.mapPortals || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
