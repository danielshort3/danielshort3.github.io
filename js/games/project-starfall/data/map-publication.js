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

  const FEATURED_SPAWN_GROUP_PROFILES = Object.freeze({
    greenrootMeadow: Object.freeze([
      { sectionSuffix: 'starter_pond_loop', label: 'Pond Slimes', enemyWeights: [{ enemyId: 'dewSlime', weight: 6 }, { enemyId: 'slimelet', weight: 3 }, { enemyId: 'thornSprout', weight: 1 }], population: 6, respawnSeconds: 4, leash: 420 },
      { sectionSuffix: 'moss_lane_extension', label: 'Moss Shelf', enemyWeights: [{ enemyId: 'dewSlime', weight: 5 }, { enemyId: 'slimelet', weight: 3 }, { enemyId: 'thornSprout', weight: 2 }], population: 6, respawnSeconds: 4, leash: 460 },
      { sectionSuffix: 'canopy_practice', label: 'Canopy Pocket', enemyWeights: [{ enemyId: 'dewSlime', weight: 4 }, { enemyId: 'slimelet', weight: 3 }, { enemyId: 'thornSprout', weight: 2 }, { enemyId: 'mossback', weight: 1 }], population: 6, respawnSeconds: 5, leash: 400 },
      { sectionSuffix: 'thornpath_gate', label: 'Thornpath Gate', enemyWeights: [{ enemyId: 'dewSlime', weight: 3 }, { enemyId: 'slimelet', weight: 2 }, { enemyId: 'thornSprout', weight: 3 }, { enemyId: 'mossback', weight: 2 }], population: 6, respawnSeconds: 5, leash: 440 }
    ]),
    banditRidgeCamp: Object.freeze([
      { sectionSuffix: 'lower_cutter_lane', label: 'Lower Cutters', enemyWeights: [{ enemyId: 'banditCutter', weight: 8 }, { enemyId: 'briarStag', weight: 2 }], population: 8, respawnSeconds: 4, leash: 520 },
      { sectionSuffix: 'middle_thrower_camp', label: 'Thrower Camp', enemyWeights: [{ enemyId: 'banditThrower', weight: 7 }, { enemyId: 'banditCutter', weight: 3 }], population: 8, respawnSeconds: 5, leash: 480 },
      { sectionSuffix: 'high_rope_bridge', label: 'Rope Bridge', enemyWeights: [{ enemyId: 'banditThrower', weight: 6 }, { enemyId: 'vineSnapper', weight: 3 }, { enemyId: 'briarStag', weight: 1 }], population: 8, respawnSeconds: 5, leash: 420 },
      { sectionSuffix: 'campfire_regroup', label: 'Campfire Regroup', enemyWeights: [{ enemyId: 'banditCutter', weight: 5 }, { enemyId: 'banditThrower', weight: 3 }, { enemyId: 'briarStag', weight: 2 }], population: 6, respawnSeconds: 5, leash: 520 }
    ]),
    orebackQuarry: Object.freeze([
      { sectionSuffix: 'ore_cart_lane', label: 'Ore Cart Beetles', enemyWeights: [{ enemyId: 'orebackBeetle', weight: 8 }, { enemyId: 'scrapWarden', weight: 2 }], population: 8, respawnSeconds: 5, leash: 500 },
      { sectionSuffix: 'scaffold_sentries', label: 'Scaffold Sentries', enemyWeights: [{ enemyId: 'coilSentry', weight: 7 }, { enemyId: 'orebackBeetle', weight: 3 }], population: 7, respawnSeconds: 6, leash: 380 },
      { sectionSuffix: 'mushroom_pocket', label: 'Glowcap Pocket', enemyWeights: [{ enemyId: 'glowcapHealer', weight: 6 }, { enemyId: 'orebackBeetle', weight: 4 }], population: 7, respawnSeconds: 6, leash: 420 },
      { sectionSuffix: 'mine_event_pocket', label: 'Mimic Mine', enemyWeights: [{ enemyId: 'orebackBeetle', weight: 5 }, { enemyId: 'scrapWarden', weight: 3 }, { enemyId: 'crackedMimic', weight: 1 }], population: 4, respawnSeconds: 8, leash: 360 }
    ]),
    stormbreakCliffs: Object.freeze([
      { sectionSuffix: 'low_ram_lane', label: 'Thunder Ram Lane', enemyWeights: [{ enemyId: 'thunderRam', weight: 8 }, { enemyId: 'cloudcallAcolyte', weight: 2 }], population: 9, respawnSeconds: 5, leash: 500 },
      { sectionSuffix: 'mid_archer_bridge', label: 'Archer Bridge', enemyWeights: [{ enemyId: 'stormboundArcher', weight: 7 }, { enemyId: 'cloudcallAcolyte', weight: 3 }], population: 8, respawnSeconds: 5, leash: 440 },
      { sectionSuffix: 'high_harrier_airspace', label: 'Harrier Airspace', enemyWeights: [{ enemyId: 'galeHarrier', weight: 8 }, { enemyId: 'stormboundArcher', weight: 2 }], population: 9, respawnSeconds: 5, leash: 540, actorTraversal: { mode: 'air', allowLadders: false, allowRamps: true, stayInTerritory: true } },
      { sectionSuffix: 'lightning_rod_objective', label: 'Lightning Rod', enemyWeights: [{ enemyId: 'cloudcallAcolyte', weight: 6 }, { enemyId: 'thunderRam', weight: 3 }, { enemyId: 'crackedMimic', weight: 1 }], population: 6, respawnSeconds: 7, leash: 380 }
    ])
  });

  function normalizeSpawnEnemyWeights(source, fallbackEnemies) {
    const entries = Array.isArray(source) && source.length ? source : fallbackEnemies || EMPTY_ARRAY;
    const totals = {};
    const order = [];
    entries.forEach((entry) => {
      const enemyId = String(entry && typeof entry === 'object' ? entry.enemyId || entry.id : entry || '').trim();
      const weight = Math.max(0, Number(entry && typeof entry === 'object' ? entry.weight : 1) || 0);
      if (!enemyId || !weight) return;
      if (!totals[enemyId]) order.push(enemyId);
      totals[enemyId] = (totals[enemyId] || 0) + weight;
    });
    return Object.freeze(order.map((enemyId) => Object.freeze({ enemyId, weight: totals[enemyId] })));
  }

  function getPublishedPlatformId(map, platformIndex) {
    const platform = Array.isArray(map && map.platforms) ? map.platforms[platformIndex] : null;
    return String(platform && !Array.isArray(platform) && platform.id || `${map && map.id || 'map'}_platform_${platformIndex}`);
  }

  function normalizeActorTraversal(source) {
    const traversal = source && typeof source === 'object' ? source : EMPTY_OBJECT;
    return Object.freeze({
      mode: String(traversal.mode || 'ground'),
      allowLadders: !!traversal.allowLadders,
      allowRamps: traversal.allowRamps !== false,
      stayInTerritory: traversal.stayInTerritory !== false
    });
  }

  function createFallbackSpawnGroupProfiles(map, spawnSections) {
    const sections = Array.isArray(spawnSections) ? spawnSections : EMPTY_ARRAY;
    if (sections.length) {
      const population = Math.max(1, Math.floor(Number(map && map.waveMax || 0) / sections.length) || 1);
      let assigned = 0;
      return sections.map((section, index) => {
        const isLast = index === sections.length - 1;
        const targetPopulation = isLast
          ? Math.max(1, Number(map.waveMax || 0) - assigned || population)
          : population;
        assigned += targetPopulation;
        return {
          sectionId: section.id,
          label: section.label,
          population: targetPopulation
        };
      });
    }
    if (!map || map.safeZone || !(map.enemies || EMPTY_ARRAY).length) return EMPTY_ARRAY;
    return [{ id: `${map.id}_field`, label: map.name || 'Field', population: Math.max(1, Number(map.waveMax || 0) || (map.enemies || EMPTY_ARRAY).length) }];
  }

  function normalizeSpawnGroups(map, spawnSections, spawnPoints) {
    if (!map || map.safeZone) return EMPTY_ARRAY;
    const sections = Array.isArray(spawnSections) ? spawnSections : EMPTY_ARRAY;
    const points = Array.isArray(spawnPoints) ? spawnPoints : EMPTY_ARRAY;
    const authored = Array.isArray(map.spawnGroups) && map.spawnGroups.length
      ? map.spawnGroups
      : FEATURED_SPAWN_GROUP_PROFILES[map.id] || createFallbackSpawnGroupProfiles(map, sections);
    const seenIds = new Set();
    const normalized = authored.map((rawGroup, index) => {
      const source = rawGroup && typeof rawGroup === 'object' ? rawGroup : EMPTY_OBJECT;
      const section = sections.find((entry) => entry && (
        source.sectionId && entry.id === source.sectionId ||
        source.sectionSuffix && String(entry.id || '').endsWith(source.sectionSuffix)
      )) || null;
      const sectionId = String(source.sectionId || section && section.id || '');
      let id = String(source.id || sectionId || `${map.id}_spawn_group_${index + 1}`)
        .trim()
        .replace(/[^A-Za-z0-9_-]+/g, '_');
      if (!id) id = `${map.id}_spawn_group_${index + 1}`;
      if (seenIds.has(id)) id = `${id}_${index + 1}`;
      seenIds.add(id);
      const sectionPoints = points.filter((point) => point && (!sectionId || point.sectionId === sectionId));
      let platformIndices = (source.platformIndices || EMPTY_ARRAY)
        .map((value) => Math.floor(Number(value)))
        .filter((value) => Number.isInteger(value) && value >= 0 && value < (map.platforms || EMPTY_ARRAY).length);
      if (!platformIndices.length) {
        platformIndices = sectionPoints
          .map((point) => Math.floor(Number(point.platformIndex)))
          .filter((value) => Number.isInteger(value) && value >= 0);
      }
      if (!platformIndices.length && section) {
        const left = Number(section.x || 0);
        const right = left + Math.max(0, Number(section.w || 0));
        platformIndices = (map.platforms || EMPTY_ARRAY)
          .map((platform, platformIndex) => {
            const x = Array.isArray(platform) ? Number(platform[0] || 0) : Number(platform && platform.x || 0);
            const w = Array.isArray(platform) ? Number(platform[2] || 0) : Number(platform && platform.w || 0);
            return platformIndex > 0 && x + w >= left && x <= right ? platformIndex : -1;
          })
          .filter((value) => value >= 0);
        if (!platformIndices.length) {
          const sectionCenter = left + Math.max(0, Number(section.w || 0)) / 2;
          platformIndices = (map.platforms || EMPTY_ARRAY)
            .map((platform, platformIndex) => {
              const x = Array.isArray(platform) ? Number(platform[0] || 0) : Number(platform && platform.x || 0);
              const w = Array.isArray(platform) ? Number(platform[2] || 0) : Number(platform && platform.w || 0);
              return { platformIndex, distance: Math.abs(x + w / 2 - sectionCenter) };
            })
            .filter((entry) => entry.platformIndex > 0)
            .sort((a, b) => a.distance - b.distance)
            .slice(0, 2)
            .map((entry) => entry.platformIndex);
        }
      }
      const declaredPlatformIds = (source.platformIds || EMPTY_ARRAY).map(String).filter(Boolean);
      const platformIds = Array.from(new Set(declaredPlatformIds.concat(platformIndices.map((platformIndex) => getPublishedPlatformId(map, platformIndex)))));
      const enemyWeights = normalizeSpawnEnemyWeights(source.enemyWeights || source.enemies, map.enemies);
      if (!platformIds.length || !enemyWeights.length) return null;
      return Object.freeze({
        id,
        label: String(source.label || section && section.label || `Spawn Group ${index + 1}`),
        sectionId,
        platformIds: Object.freeze(platformIds),
        enemyWeights,
        population: Math.max(1, Math.floor(Number(source.population || 0)) || 1),
        respawnSeconds: Math.max(1, Math.min(60, Number(source.respawnSeconds || map.waveDelay || 5) || 5)),
        leash: Math.max(90, Math.min(2400, Number(source.leash || 480) || 480)),
        partyScaling: String(source.partyScaling || map.partyScaling || map.designIntent && map.designIntent.partyScaling || 'none'),
        maxPopulation: Math.max(1, Math.floor(Number(source.maxPopulation || 0)) || Math.ceil(Math.max(1, Number(source.population || 1)) * 1.5)),
        partyBonusPerMember: Math.max(0, Math.min(4, Number(source.partyBonusPerMember == null ? 1 : source.partyBonusPerMember) || 0)),
        actorTraversal: normalizeActorTraversal(source.actorTraversal)
      });
    }).filter(Boolean);
    return Object.freeze(normalized);
  }

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
    const defaultQuestNpcAsset = settings.DEFAULT_QUEST_NPC_ASSET || DataAssets.GENERIC_PLAYER_ASSET || '';
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
      const spawnPoints = attachSpawnSectionsToPoints(map, spawnSections);
      const spawnGroups = normalizeSpawnGroups(Object.assign({}, map, { designIntent }), spawnSections, spawnPoints);
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
        spawnPoints,
        spawnGroups,
        stations: (map.stations || []).map((station) => attachAsset(Object.assign({}, getStationServiceIntent(station.id), station), stationAssets[station.id])),
        questNpcs: (map.questNpcs || []).map((npc) => Object.freeze(Object.assign({
          asset: defaultQuestNpcAsset
        }, npc, {
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
      attachMapAssets,
      normalizeSpawnGroups
    });
  }

  const defaultMapPublicationData = createMapPublicationData();
  const api = Object.assign({
    attachAsset,
    FEATURED_SPAWN_GROUP_PROFILES,
    normalizeSpawnEnemyWeights,
    normalizeActorTraversal,
    normalizeSpawnGroups,
    createMapPublicationData
  }, defaultMapPublicationData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapPublication = Object.assign({}, modules.mapPublication || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
