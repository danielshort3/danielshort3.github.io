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

  function applyPartyPlayGeometry(map) {
    if (!map) return map;
    if (map.shopInterior) {
      const width = Math.max(getAuthoredMapWidth(map), Number(map.compactWorldWidth || SHOP_INTERIOR_WORLD_WIDTH));
      const platforms = [[0, 520, width, 80]];
      return Object.assign({}, map, {
        layoutStyle: 'shopInterior',
        layoutRole: 'town',
        worldHeight: 0,
        authoredGroundY: 520,
        platforms,
        terrainVisuals: [],
        climbables: [],
        spawnPoints: [
          { id: `${map.id}_entrance`, x: 180, platformIndex: 0, weight: 1 }
        ],
        stations: [],
        questNpcs: map.questNpcs || []
      });
    }
    if (map.safeZone) {
      const width = Math.max(getAuthoredMapWidth(map), 3600);
      const platforms = makeTownPlatforms(width, map.id);
      return Object.assign({}, map, {
        layoutStyle: 'townVerticalHub',
        layoutRole: 'town',
        worldHeight: TOWN_WORLD_HEIGHT,
        authoredGroundY: TOWN_LANE_Y.ground,
        platforms,
        terrainVisuals: makeFieldTerrainVisuals(platforms, 'townVerticalHub'),
        rampConnections: makeRampConnections(map.id, platforms),
        climbables: makeTownClimbables(map.id, platforms),
        spawnPoints: [
          { id: `${map.id}_entry_plaza`, x: 220, platformIndex: 0, weight: 2 },
          { id: `${map.id}_market_walk`, x: 530, platformIndex: 1, weight: 1 },
          { id: `${map.id}_mid_walk`, x: 1120, platformIndex: 4, weight: 1 },
          { id: `${map.id}_high_walk`, x: 1750, platformIndex: 8, weight: 1 }
        ],
        stations: assignTownStations(map.stations || []),
        questNpcs: assignTownQuestNpcs(map.questNpcs || [])
      });
    }
    const layoutStyle = getFieldLayoutStyle(map);
    const vertical = isVerticalFieldLayout(layoutStyle);
    const authoredWidth = vertical ? 3600 : getAuthoredMapWidth(map);
    const width = Math.max(authoredWidth, map.isDungeon ? 4600 : vertical ? 5200 : 8400);
    const dungeonSkeleton = map.isDungeon ? getDungeonArenaSkeleton(map.id) : null;
    const platforms = map.isDungeon
      ? makeDungeonArenaPlatforms(width, map.id) || makePartyPlayPlatforms(width, { dungeon: true, variantKey: map.id })
      : makeFieldPlatforms(width, layoutStyle, map.id);
    return Object.assign({}, map, {
      layoutStyle: map.isDungeon ? 'dungeonArena' : layoutStyle,
      layoutRole: map.layoutRole || getMapLayoutRoleFallback(map),
      arenaSkeleton: dungeonSkeleton ? dungeonSkeleton.id : '',
      arenaMechanic: dungeonSkeleton ? dungeonSkeleton.mechanic : '',
      worldHeight: map.isDungeon ? 0 : getFieldLayoutWorldHeight(layoutStyle),
      authoredGroundY: map.isDungeon ? TRAINING_LANE_Y.ground : getFieldLaneY(layoutStyle).ground,
      platforms,
      terrainVisuals: makeFieldTerrainVisuals(platforms, map.isDungeon ? 'dungeonArena' : layoutStyle),
      rampConnections: makeRampConnections(map.id, platforms),
      climbables: map.isDungeon ? makeVerticalFieldClimbables(map.id, platforms, 'dungeonArena') : makeFieldClimbables(map.id, platforms, layoutStyle),
      spawnPoints: map.isDungeon ? makePartyPlaySpawnPoints(platforms) : makeFieldSpawnPoints(platforms)
    });
  }

  function makeExpandedTrainingMap(config) {
    const layoutStyle = config.layoutStyle || FIELD_LAYOUT_STYLES[config.id] || 'sharedLanes';
    const width = isVerticalFieldLayout(layoutStyle) ? 5200 : 8400;
    const platforms = makeFieldPlatforms(width, layoutStyle, config.id);
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange,
      safeZone: false,
      layoutStyle,
      layoutRole: config.layoutRole || (config.endlessScaling ? 'endlessField' : 'trainingField'),
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
      spawnPoints: makeFieldSpawnPoints(platforms),
      stations: [],
      questNpcs: config.questNpcs || []
    };
  }

  function makeBossRoomMap(config) {
    const width = Math.max(4600, Number(config.width || 4600));
    const platforms = makePartyPlayPlatforms(width, { dungeon: true, variantKey: config.id });
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange,
      safeZone: false,
      isDungeon: true,
      bossRoom: true,
      layoutRole: 'bossArena',
      dungeonId: `boss_${config.bossId}`,
      bossId: config.bossId,
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
      spawnPoints: makePartyPlaySpawnPoints(platforms),
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
