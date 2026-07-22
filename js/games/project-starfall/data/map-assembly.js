(function initProjectStarfallDataMapAssembly(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataWorld = (typeof require === 'function' ? require('./world.js') : null) || DataModules.world || {};
  const DataMapGeometry = (typeof require === 'function' ? require('./map-geometry.js') : null) || DataModules.mapGeometry || {};
  const DataMapSizing = (typeof require === 'function' ? require('./map-sizing.js') : null) || DataModules.mapSizing || {};
  const DataMapLayouts = (typeof require === 'function' ? require('./map-layouts.js') : null) || DataModules.mapLayouts || {};
  const DataMapBuilders = (typeof require === 'function' ? require('./map-builders.js') : null) || DataModules.mapBuilders || {};
  const DataMapTown = (typeof require === 'function' ? require('./map-town.js') : null) || DataModules.mapTown || {};
  const DataMapPresentation = (typeof require === 'function' ? require('./map-presentation.js') : null) || DataModules.mapPresentation || {};

  const SHOP_INTERIOR_WORLD_WIDTH = DataWorld.SHOP_INTERIOR_WORLD_WIDTH || 1280;
  const makeRampConnections = DataMapGeometry.makeRampConnections;
  const assignStablePlatformIds = DataMapGeometry.assignStablePlatformIds || ((prefix, platforms) => Object.freeze((platforms || []).slice()));
  const getPlatformDefY = DataMapGeometry.getPlatformDefY;
  const getAuthoredMapWidth = DataMapSizing.getAuthoredMapWidth;
  const TRAINING_LANE_Y = DataMapLayouts.TRAINING_LANE_Y;
  const TOWN_WORLD_HEIGHT = DataMapLayouts.TOWN_WORLD_HEIGHT;
  const TOWN_LANE_Y = DataMapLayouts.TOWN_LANE_Y;
  const isVerticalFieldLayout = DataMapLayouts.isVerticalFieldLayout;
  const getFieldLayoutWorldHeight = DataMapLayouts.getFieldLayoutWorldHeight;
  const getFieldLaneY = DataMapLayouts.getFieldLaneY;
  const FIELD_LAYOUT_STYLES = DataMapLayouts.FIELD_LAYOUT_STYLES || Object.freeze({});
  const getFieldLayoutStyle = DataMapLayouts.getFieldLayoutStyle;
  const getDungeonArenaSkeleton = DataMapLayouts.getDungeonArenaSkeleton;
  const makePartyPlayPlatforms = DataMapBuilders.makePartyPlayPlatforms;
  const makePartyPlayClimbables = DataMapBuilders.makePartyPlayClimbables;
  const makePartyPlaySpawnPoints = DataMapBuilders.makePartyPlaySpawnPoints;
  const makeDungeonArenaPlatforms = DataMapBuilders.makeDungeonArenaPlatforms;
  const makeBanditRidgeCampClimbables = DataMapBuilders.makeBanditRidgeCampClimbables;
  const makeThornpathFractureCanopyClimbables = DataMapBuilders.makeThornpathFractureCanopyClimbables;
  const makeFrostfenMarshRunClimbables = DataMapBuilders.makeFrostfenMarshRunClimbables;
  const makeRustcoilOrreryCircuitClimbables = DataMapBuilders.makeRustcoilOrreryCircuitClimbables;
  const makeFieldPlatforms = DataMapBuilders.makeFieldPlatforms;
  const makeFieldTerrainVisuals = DataMapBuilders.makeFieldTerrainVisuals;
  const makeVerticalFieldClimbables = DataMapBuilders.makeVerticalFieldClimbables;
  const makeFieldClimbables = DataMapBuilders.makeFieldClimbables;
  const makeFieldSpawnPoints = DataMapBuilders.makeFieldSpawnPoints;
  const makeTownPlatforms = DataMapBuilders.makeTownPlatforms;
  const makeTownClimbables = DataMapBuilders.makeTownClimbables;
  const assignTownStations = DataMapTown.assignTownStations;
  const assignTownQuestNpcs = DataMapTown.assignTownQuestNpcs;
  const getMapLayoutRoleFallback = DataMapPresentation.getMapLayoutRoleFallback || ((map) => map && map.safeZone ? 'town' : map && map.bossRoom ? 'bossArena' : map && map.isDungeon ? 'dungeon' : 'trainingField');

  function attachStableSpawnPlatformIds(platforms, spawnPoints) {
    return Object.freeze((spawnPoints || []).map((point, index) => {
      const platform = platforms[Number(point && point.platformIndex)];
      return Object.freeze(Object.assign({}, point, {
        id: point && point.id || `spawn_${index + 1}`,
        platformId: String(point && point.platformId || platform && platform.id || '')
      }));
    }));
  }

  function createAuthoredGeometry(map) {
    const platforms = assignStablePlatformIds(map.id, map.platforms || []);
    const layoutStyle = map.layoutStyle || 'authored';
    return Object.assign({}, map, {
      geometryMode: 'authored',
      geometryGenerator: '',
      layoutStyle,
      layoutRole: map.layoutRole || getMapLayoutRoleFallback(map),
      worldHeight: Number(map.worldHeight || 0),
      authoredGroundY: platforms[0] ? getPlatformDefY(platforms[0]) : 520,
      platforms,
      terrainVisuals: Array.isArray(map.terrainVisuals) && map.terrainVisuals.length === platforms.length
        ? Object.freeze(map.terrainVisuals.slice())
        : makeFieldTerrainVisuals(platforms, layoutStyle),
      rampConnections: makeRampConnections(map.id, platforms),
      climbables: Object.freeze((map.climbables || []).map((climbable) => Object.freeze(Object.assign({}, climbable)))),
      spawnPoints: attachStableSpawnPlatformIds(platforms, map.spawnPoints || [])
    });
  }

  function applyPartyPlayGeometry(map) {
    if (!map) return map;
    if (map.geometryMode === 'authored' || map.adminOnly && map.geometryMode !== 'generated') {
      return createAuthoredGeometry(map);
    }
    if (map.shopInterior) {
      const width = Math.max(getAuthoredMapWidth(map), Number(map.compactWorldWidth || SHOP_INTERIOR_WORLD_WIDTH));
      const platforms = assignStablePlatformIds(map.id, [[0, 520, width, 80]]);
      return Object.assign({}, map, {
        geometryMode: 'generated',
        geometryGenerator: 'shopInterior',
        layoutStyle: 'shopInterior',
        layoutRole: 'town',
        worldHeight: 0,
        authoredGroundY: 520,
        platforms,
        terrainVisuals: [],
        climbables: [],
        spawnPoints: attachStableSpawnPlatformIds(platforms, [
          { id: `${map.id}_entrance`, x: 180, platformIndex: 0, weight: 1 }
        ]),
        stations: [],
        questNpcs: map.questNpcs || []
      });
    }
    if (map.safeZone) {
      const width = Math.max(getAuthoredMapWidth(map), 3600);
      const platforms = assignStablePlatformIds(map.id, makeTownPlatforms(width, map.id));
      return Object.assign({}, map, {
        geometryMode: 'generated',
        geometryGenerator: 'townVerticalHub',
        layoutStyle: 'townVerticalHub',
        layoutRole: 'town',
        worldHeight: TOWN_WORLD_HEIGHT,
        authoredGroundY: TOWN_LANE_Y.ground,
        platforms,
        terrainVisuals: makeFieldTerrainVisuals(platforms, 'townVerticalHub'),
        rampConnections: makeRampConnections(map.id, platforms),
        climbables: makeTownClimbables(map.id, platforms),
        spawnPoints: attachStableSpawnPlatformIds(platforms, [
          { id: `${map.id}_entry_plaza`, x: 220, platformIndex: 0, weight: 2 },
          { id: `${map.id}_market_walk`, x: 530, platformIndex: 1, weight: 1 },
          { id: `${map.id}_mid_walk`, x: 1120, platformIndex: 4, weight: 1 },
          { id: `${map.id}_high_walk`, x: 1750, platformIndex: 8, weight: 1 }
        ]),
        stations: assignTownStations(map.stations || []),
        questNpcs: assignTownQuestNpcs(map.questNpcs || [])
      });
    }
    const layoutStyle = getFieldLayoutStyle(map);
    const vertical = isVerticalFieldLayout(layoutStyle);
    const authoredWidth = vertical ? 3600 : getAuthoredMapWidth(map);
    const compactWidth = Math.max(0, Number(map.compactWorldWidth || 0));
    const requestedWidth = compactWidth || authoredWidth;
    const width = Math.max(requestedWidth, map.isDungeon ? 4600 : vertical ? compactWidth ? 4600 : 5200 : compactWidth ? 4000 : 8400);
    const dungeonSkeleton = map.isDungeon ? getDungeonArenaSkeleton(map.id) : null;
    const generatedPlatforms = map.isDungeon
      ? makeDungeonArenaPlatforms(width, map.id) || makePartyPlayPlatforms(width, { dungeon: true, variantKey: map.id })
      : makeFieldPlatforms(width, layoutStyle, map.id);
    const platforms = assignStablePlatformIds(map.id, generatedPlatforms);
    return Object.assign({}, map, {
      geometryMode: 'generated',
      geometryGenerator: map.geometryGenerator || (map.isDungeon ? 'dungeonArena' : 'fieldLayout'),
      layoutStyle: map.isDungeon ? 'dungeonArena' : layoutStyle,
      layoutRole: map.layoutRole || getMapLayoutRoleFallback(map),
      arenaSkeleton: dungeonSkeleton ? dungeonSkeleton.id : '',
      arenaMechanic: dungeonSkeleton ? dungeonSkeleton.mechanic : '',
      worldHeight: map.isDungeon ? 0 : getFieldLayoutWorldHeight(layoutStyle),
      authoredGroundY: map.isDungeon ? TRAINING_LANE_Y.ground : getFieldLaneY(layoutStyle).ground,
      platforms,
      terrainVisuals: makeFieldTerrainVisuals(platforms, map.isDungeon ? 'dungeonArena' : layoutStyle),
      rampConnections: makeRampConnections(map.id, platforms),
      climbables: map.isDungeon
        ? makeVerticalFieldClimbables(map.id, platforms, 'dungeonArena')
        : map.id === 'rustcoilRuins' && makeRustcoilOrreryCircuitClimbables
          ? makeRustcoilOrreryCircuitClimbables(platforms)
        : map.id === 'thornpathThicket' && makeThornpathFractureCanopyClimbables
          ? makeThornpathFractureCanopyClimbables(platforms)
          : map.id === 'frostfenOutskirts' && makeFrostfenMarshRunClimbables
            ? makeFrostfenMarshRunClimbables(platforms)
          : map.id === 'banditRidgeCamp' && makeBanditRidgeCampClimbables
            ? makeBanditRidgeCampClimbables(platforms)
            : makeFieldClimbables(map.id, platforms, layoutStyle),
      spawnPoints: attachStableSpawnPlatformIds(platforms, map.isDungeon ? makePartyPlaySpawnPoints(platforms) : makeFieldSpawnPoints(platforms))
    });
  }

  function makeExpandedTrainingMap(config) {
    const layoutStyle = config.layoutStyle || FIELD_LAYOUT_STYLES[config.id] || 'sharedLanes';
    const compactWorldWidth = Math.max(0, Math.ceil(Number(config.compactWorldWidth || 0) / 100) * 100);
    const width = compactWorldWidth || (isVerticalFieldLayout(layoutStyle) ? 5200 : 8400);
    const platforms = assignStablePlatformIds(config.id, makeFieldPlatforms(width, layoutStyle, config.id));
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange,
      safeZone: false,
      geometryMode: 'generated',
      geometryGenerator: config.geometryGenerator || 'fieldLayout',
      layoutStyle,
      layoutRole: config.layoutRole || (config.endlessScaling ? 'endlessField' : 'trainingField'),
      compactWorldWidth,
      scaleEnemies: true,
      endlessScaling: !!config.endlessScaling,
      movementProfile: config.movementProfile || '',
      areaMechanic: config.areaMechanic || '',
      waveMax: config.waveMax || 30,
      waveDelay: config.waveDelay || 5,
      worldHeight: getFieldLayoutWorldHeight(layoutStyle),
      authoredGroundY: getFieldLaneY(layoutStyle).ground,
      palette: config.palette,
      purpose: config.purpose,
      enemies: config.enemies,
      platforms,
      terrainVisuals: makeFieldTerrainVisuals(platforms, layoutStyle),
      rampConnections: makeRampConnections(config.id, platforms),
      climbables: makeFieldClimbables(config.id, platforms, layoutStyle),
      spawnPoints: attachStableSpawnPlatformIds(platforms, makeFieldSpawnPoints(platforms)),
      stations: [],
      questNpcs: config.questNpcs || []
    };
  }

  function makeBossRoomMap(config) {
    const width = Math.max(4600, Number(config.width || 4600));
    const platforms = assignStablePlatformIds(config.id, makePartyPlayPlatforms(width, { dungeon: true, variantKey: config.id }));
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange,
      safeZone: false,
      geometryMode: 'generated',
      geometryGenerator: config.geometryGenerator || 'bossArena',
      isDungeon: true,
      bossRoom: true,
      layoutRole: 'bossArena',
      dungeonId: `boss_${config.bossId}`,
      bossId: config.bossId,
      backgroundMode: config.backgroundMode || '',
      movementProfile: config.movementProfile || '',
      areaMechanic: config.areaMechanic || '',
      waveMax: config.waveMax || 8,
      waveDelay: config.waveDelay || 8,
      palette: config.palette,
      purpose: config.purpose,
      enemies: config.enemies,
      platforms,
      terrainVisuals: makeFieldTerrainVisuals(platforms, 'bossArena'),
      rampConnections: makeRampConnections(config.id, platforms),
      climbables: makePartyPlayClimbables(config.id, platforms, { dungeon: true }),
      spawnPoints: attachStableSpawnPlatformIds(platforms, makePartyPlaySpawnPoints(platforms)),
      stations: [],
      questNpcs: config.questNpcs || []
    };
  }

  const api = Object.freeze({
    applyPartyPlayGeometry,
    makeExpandedTrainingMap,
    makeBossRoomMap
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapAssembly = Object.assign({}, modules.mapAssembly || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
