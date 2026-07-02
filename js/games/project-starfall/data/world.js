(function initProjectStarfallDataWorld(global) {
  'use strict';

  const WORLD_AREAS = Object.freeze([
    Object.freeze({
      id: 'crossing',
      name: 'Starfall Crossing',
      themeId: 'town',
      routeId: '',
      levelRange: Object.freeze([1, 99]),
      color: '#f7d28a',
      accent: '#7ec8d8',
      x: 6,
      y: 38,
      w: 16,
      h: 30,
      summary: 'Town services, class setup, upgrades, quests, and route handoffs.',
      mechanic: 'Safe hub'
    }),
    Object.freeze({
      id: 'greenroot',
      name: 'Greenroot Wilds',
      themeId: 'grass',
      routeId: 'forest',
      levelRange: Object.freeze([1, 34]),
      color: '#77bf65',
      accent: '#e05b75',
      x: 20,
      y: 18,
      w: 32,
      h: 36,
      summary: 'Meadows, thorn paths, ridge camps, and the Brambleking route.',
      mechanic: 'Vines, ranged thorns, and mixed beast pressure'
    }),
    Object.freeze({
      id: 'rustcoil',
      name: 'Rustcoil Expanse',
      themeId: 'ruins',
      routeId: 'ruins',
      levelRange: Object.freeze([12, 48]),
      color: '#8c6b35',
      accent: '#29b3ad',
      x: 44,
      y: 48,
      w: 28,
      h: 30,
      summary: 'Construct ruins, quarry scaffolds, armor-break checks, and gearwork bosses.',
      mechanic: 'Armored enemies and ore material routes'
    }),
    Object.freeze({
      id: 'cinder',
      name: 'Cinder Basin',
      themeId: 'cinder',
      routeId: 'cinder',
      levelRange: Object.freeze([16, 55]),
      color: '#f06b37',
      accent: '#ffbe55',
      x: 64,
      y: 66,
      w: 22,
      h: 28,
      summary: 'Lava caves, ashglass exits, flying spirits, and Emberjaw progression.',
      mechanic: 'Flying fire pressure and dense vertical chains'
    }),
    Object.freeze({
      id: 'frostfen',
      name: 'Frostfen Tundra',
      themeId: 'frost',
      routeId: 'frostfen',
      levelRange: Object.freeze([45, 68]),
      color: '#d7f3ff',
      accent: '#5ca8e8',
      x: 73,
      y: 43,
      w: 24,
      h: 24,
      summary: 'Snowfields, glacier ledges, slippery movement, and the Rimewarden sanctum.',
      mechanic: 'Ice footing: slower acceleration, longer slide, frost enemy pressure'
    }),
    Object.freeze({
      id: 'stormbreak',
      name: 'Stormbreak Reach',
      themeId: 'storm',
      routeId: 'ascension',
      levelRange: Object.freeze([55, 70]),
      color: '#91dbe8',
      accent: '#ffe16a',
      x: 82,
      y: 28,
      w: 17,
      h: 20,
      summary: 'Wind-cut cliffs and long mobility lanes before the late-game arc.',
      mechanic: 'Ranged pressure and cliff traversal'
    }),
    Object.freeze({
      id: 'astral',
      name: 'Astral Dominion',
      themeId: 'astral',
      routeId: 'ascension',
      levelRange: Object.freeze([70, 100]),
      color: '#29365f',
      accent: '#c794ff',
      x: 76,
      y: 4,
      w: 24,
      h: 28,
      summary: 'Archives, eclipse frontier maps, and the endless rift training loop.',
      mechanic: 'Elite density and scaling endgame routes'
    })
  ]);

  const WORLD_ROUTES = Object.freeze([
    Object.freeze({
      id: 'forest',
      name: 'Forest Route',
      startMapId: 'greenrootMeadow',
      bossMapId: 'brambleDepths',
      bossDungeonId: 'bramble_depths',
      fieldGoals: Object.freeze([
        Object.freeze({ mapId: 'greenrootMeadow', count: 18 }),
        Object.freeze({ mapId: 'thornpathThicket', count: 24 }),
        Object.freeze({ mapId: 'banditRidgeCamp', count: 24 })
      ])
    }),
    Object.freeze({
      id: 'ruins',
      name: 'Ruins Route',
      startMapId: 'rustcoilRuins',
      bossMapId: 'gearworksVault',
      bossDungeonId: 'gearworks_vault',
      fieldGoals: Object.freeze([
        Object.freeze({ mapId: 'rustcoilRuins', count: 26 }),
        Object.freeze({ mapId: 'orebackQuarry', count: 26 })
      ])
    }),
    Object.freeze({
      id: 'cinder',
      name: 'Cinder Route',
      startMapId: 'cinderHollow',
      bossMapId: 'emberjawLair',
      bossDungeonId: 'emberjaw_lair',
      fieldGoals: Object.freeze([
        Object.freeze({ mapId: 'cinderHollow', count: 30 })
      ])
    }),
    Object.freeze({
      id: 'ascension',
      name: 'Ascension Route',
      startMapId: 'stormbreakCliffs',
      fieldGoals: Object.freeze([
        Object.freeze({ mapId: 'stormbreakCliffs', count: 36 }),
        Object.freeze({ mapId: 'astralArchive', count: 38 }),
        Object.freeze({ mapId: 'eclipseFrontier', count: 40 }),
        Object.freeze({ mapId: 'endlessRift', count: 44 })
      ])
    }),
    Object.freeze({
      id: 'frostfen',
      name: 'Frostfen Route',
      startMapId: 'ashglassPass',
      bossMapId: 'rimewardenSanctum',
      bossDungeonId: 'rimewarden_sanctum',
      fieldGoals: Object.freeze([
        Object.freeze({ mapId: 'ashglassPass', count: 34 }),
        Object.freeze({ mapId: 'frostfenOutskirts', count: 32 }),
        Object.freeze({ mapId: 'glacierSpine', count: 34 })
      ])
    })
  ]);

  const REGIONAL_TOWN_IDS = Object.freeze(['starfallCrossing', 'rustcoilOutpost', 'cinderRefuge', 'frostfenCamp', 'stormbreakHaven', 'astralObservatory']);

  const SHOP_INTERIOR_WORLD_WIDTH = 1280;

  const WORLD_MAP_NODES = Object.freeze([
    Object.freeze({ mapId: 'starfallCrossing', x: 13, y: 48, areaId: 'crossing', region: 'Starfall Crossing', type: 'town', labelSide: 'right' }),
    Object.freeze({ mapId: 'greenrootMeadow', x: 27, y: 50, areaId: 'greenroot', region: 'Greenroot Wilds', type: 'field', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'thornpathThicket', x: 35, y: 44, areaId: 'greenroot', region: 'Greenroot Wilds', type: 'field', labelSide: 'right' }),
    Object.freeze({ mapId: 'banditRidgeCamp', x: 38, y: 31, areaId: 'greenroot', region: 'Greenroot Wilds', type: 'field', labelSide: 'left' }),
    Object.freeze({ mapId: 'brambleDepths', x: 45, y: 24, areaId: 'greenroot', region: 'Greenroot Wilds', type: 'dungeon', labelSide: 'top' }),
    Object.freeze({ mapId: 'rustcoilOutpost', x: 46, y: 52, areaId: 'rustcoil', region: 'Rustcoil Expanse', type: 'town', labelSide: 'top' }),
    Object.freeze({ mapId: 'rustcoilRuins', x: 49, y: 56, areaId: 'rustcoil', region: 'Rustcoil Expanse', type: 'field', labelSide: 'right' }),
    Object.freeze({ mapId: 'orebackQuarry', x: 42, y: 68, areaId: 'rustcoil', region: 'Rustcoil Expanse', type: 'field', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'gearworksVault', x: 55, y: 45, areaId: 'rustcoil', region: 'Rustcoil Expanse', type: 'dungeon', labelSide: 'top' }),
    Object.freeze({ mapId: 'cinderRefuge', x: 60, y: 61, areaId: 'cinder', region: 'Cinder Basin', type: 'town', labelSide: 'top' }),
    Object.freeze({ mapId: 'cinderHollow', x: 63, y: 65, areaId: 'cinder', region: 'Cinder Basin', type: 'field', labelSide: 'right' }),
    Object.freeze({ mapId: 'emberjawLair', x: 66, y: 81, areaId: 'cinder', region: 'Cinder Basin', type: 'dungeon', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'ashglassPass', x: 69, y: 57, areaId: 'frostfen', region: 'Frostfen Tundra', type: 'field', labelSide: 'left' }),
    Object.freeze({ mapId: 'frostfenCamp', x: 72, y: 54, areaId: 'frostfen', region: 'Frostfen Tundra', type: 'town', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'frostfenOutskirts', x: 74, y: 50, areaId: 'frostfen', region: 'Frostfen Tundra', type: 'field', labelSide: 'right' }),
    Object.freeze({ mapId: 'glacierSpine', x: 78, y: 40, areaId: 'frostfen', region: 'Frostfen Tundra', type: 'field', labelSide: 'top' }),
    Object.freeze({ mapId: 'rimewardenSanctum', x: 79, y: 62, areaId: 'frostfen', region: 'Frostfen Tundra', type: 'dungeon', labelSide: 'right' }),
    Object.freeze({ mapId: 'stormbreakHaven', x: 85, y: 58, areaId: 'stormbreak', region: 'Stormbreak Reach', type: 'town', labelSide: 'left' }),
    Object.freeze({ mapId: 'stormbreakCliffs', x: 88, y: 62, areaId: 'stormbreak', region: 'Stormbreak Reach', type: 'field', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'astralObservatory', x: 84, y: 31, areaId: 'astral', region: 'Astral Dominion', type: 'town', labelSide: 'top' }),
    Object.freeze({ mapId: 'astralArchive', x: 86, y: 36, areaId: 'astral', region: 'Astral Dominion', type: 'field', labelSide: 'left' }),
    Object.freeze({ mapId: 'eclipseFrontier', x: 94, y: 27, areaId: 'astral', region: 'Astral Dominion', type: 'field', labelSide: 'right' }),
    Object.freeze({ mapId: 'endlessRift', x: 97, y: 10, areaId: 'astral', region: 'Astral Dominion', type: 'field', labelSide: 'left' }),
    Object.freeze({ mapId: 'bramblekingCourt', x: 43, y: 18, areaId: 'greenroot', region: 'Boss Echoes', type: 'dungeon', labelSide: 'top' }),
    Object.freeze({ mapId: 'titanFoundry', x: 58, y: 39, areaId: 'rustcoil', region: 'Boss Echoes', type: 'dungeon', labelSide: 'top' }),
    Object.freeze({ mapId: 'deepcoreCore', x: 47, y: 74, areaId: 'rustcoil', region: 'Boss Echoes', type: 'dungeon', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'emberjawFurnace', x: 67, y: 86, areaId: 'cinder', region: 'Boss Echoes', type: 'dungeon', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'rimewardenVault', x: 82, y: 58, areaId: 'frostfen', region: 'Boss Echoes', type: 'dungeon', labelSide: 'right' }),
    Object.freeze({ mapId: 'stormbreakAerie', x: 91, y: 58, areaId: 'stormbreak', region: 'Boss Echoes', type: 'dungeon', labelSide: 'bottom' }),
    Object.freeze({ mapId: 'astralStacks', x: 89, y: 31, areaId: 'astral', region: 'Boss Echoes', type: 'dungeon', labelSide: 'left' }),
    Object.freeze({ mapId: 'eclipseThrone', x: 98, y: 22, areaId: 'astral', region: 'Boss Echoes', type: 'dungeon', labelSide: 'right' })
  ]);

  const WORLD_MAP_EDGES = Object.freeze([
    Object.freeze({ id: 'crossing_greenroot', fromMapId: 'starfallCrossing', toMapId: 'greenrootMeadow', type: 'field', routeId: 'forest', portalIds: Object.freeze({ from: 'crossing_greenroot', to: 'greenroot_crossing' }) }),
    Object.freeze({ id: 'greenroot_thornpath', fromMapId: 'greenrootMeadow', toMapId: 'thornpathThicket', type: 'field', routeId: 'forest', requiredMapId: 'greenrootMeadow', portalIds: Object.freeze({ from: 'greenroot_thornpath', to: 'thornpath_greenroot' }) }),
    Object.freeze({ id: 'thornpath_bandit', fromMapId: 'thornpathThicket', toMapId: 'banditRidgeCamp', type: 'field', routeId: 'forest', requiredMapId: 'thornpathThicket', portalIds: Object.freeze({ from: 'thornpath_bandit', to: 'bandit_thornpath' }) }),
    Object.freeze({ id: 'bandit_bramble', fromMapId: 'banditRidgeCamp', toMapId: 'brambleDepths', type: 'dungeon', routeId: 'forest', dungeonId: 'bramble_depths', portalIds: Object.freeze({ from: 'bandit_bramble', to: 'bramble_bandit' }) }),
    Object.freeze({ id: 'thornpath_rustcoil_outpost', fromMapId: 'thornpathThicket', toMapId: 'rustcoilOutpost', type: 'field', routeId: 'forest', requiredMapId: 'thornpathThicket', portalIds: Object.freeze({ from: 'thornpath_rustcoil_outpost', to: 'rustcoil_outpost_thornpath' }) }),
    Object.freeze({ id: 'rustcoil_outpost_ruins', fromMapId: 'rustcoilOutpost', toMapId: 'rustcoilRuins', type: 'field', routeId: 'ruins', portalIds: Object.freeze({ from: 'rustcoil_outpost_ruins', to: 'rustcoil_outpost_return' }) }),
    Object.freeze({ id: 'rustcoil_outpost_quarry', fromMapId: 'rustcoilOutpost', toMapId: 'orebackQuarry', type: 'field', routeId: 'ruins', requiredMapId: 'rustcoilRuins', portalIds: Object.freeze({ from: 'rustcoil_outpost_quarry', to: 'quarry_rustcoil_outpost' }) }),
    Object.freeze({ id: 'quarry_vault', fromMapId: 'orebackQuarry', toMapId: 'gearworksVault', type: 'dungeon', routeId: 'ruins', dungeonId: 'gearworks_vault', portalIds: Object.freeze({ from: 'quarry_vault', to: 'vault_quarry' }) }),
    Object.freeze({ id: 'quarry_cinder_refuge', fromMapId: 'orebackQuarry', toMapId: 'cinderRefuge', type: 'field', routeId: 'ruins', requiredMapId: 'orebackQuarry', portalIds: Object.freeze({ from: 'quarry_cinder_refuge', to: 'cinder_refuge_quarry' }) }),
    Object.freeze({ id: 'cinder_refuge_hollow', fromMapId: 'cinderRefuge', toMapId: 'cinderHollow', type: 'field', routeId: 'cinder', portalIds: Object.freeze({ from: 'cinder_refuge_hollow', to: 'cinder_refuge_return' }) }),
    Object.freeze({ id: 'cinder_emberjaw', fromMapId: 'cinderHollow', toMapId: 'emberjawLair', type: 'dungeon', routeId: 'cinder', dungeonId: 'emberjaw_lair', portalIds: Object.freeze({ from: 'cinder_emberjaw', to: 'lair_cinder' }) }),
    Object.freeze({ id: 'cinder_refuge_ashglass', fromMapId: 'cinderRefuge', toMapId: 'ashglassPass', type: 'field', routeId: 'frostfen', requiredLevel: 40, requiredDungeonId: 'emberjaw_lair', portalIds: Object.freeze({ from: 'cinder_refuge_ashglass', to: 'ashglass_cinder_refuge' }) }),
    Object.freeze({ id: 'ashglass_frostfen_camp', fromMapId: 'ashglassPass', toMapId: 'frostfenCamp', type: 'field', routeId: 'frostfen', requiredMapId: 'ashglassPass', portalIds: Object.freeze({ from: 'ashglass_frostfen_camp', to: 'frostfen_camp_ashglass' }) }),
    Object.freeze({ id: 'frostfen_camp_outskirts', fromMapId: 'frostfenCamp', toMapId: 'frostfenOutskirts', type: 'field', routeId: 'frostfen', portalIds: Object.freeze({ from: 'frostfen_camp_outskirts', to: 'frostfen_camp_return' }) }),
    Object.freeze({ id: 'frostfen_camp_glacier', fromMapId: 'frostfenCamp', toMapId: 'glacierSpine', type: 'field', routeId: 'frostfen', requiredMapId: 'frostfenOutskirts', portalIds: Object.freeze({ from: 'frostfen_camp_glacier', to: 'glacier_frostfen_camp' }) }),
    Object.freeze({ id: 'glacier_sanctum', fromMapId: 'glacierSpine', toMapId: 'rimewardenSanctum', type: 'dungeon', routeId: 'frostfen', dungeonId: 'rimewarden_sanctum', portalIds: Object.freeze({ from: 'glacier_sanctum', to: 'sanctum_glacier' }) }),
    Object.freeze({ id: 'glacier_stormbreak_haven', fromMapId: 'glacierSpine', toMapId: 'stormbreakHaven', type: 'field', routeId: 'ascension', requiredDungeonId: 'rimewarden_sanctum', portalIds: Object.freeze({ from: 'glacier_stormbreak_haven', to: 'stormbreak_haven_glacier' }) }),
    Object.freeze({ id: 'stormbreak_haven_cliffs', fromMapId: 'stormbreakHaven', toMapId: 'stormbreakCliffs', type: 'field', routeId: 'ascension', portalIds: Object.freeze({ from: 'stormbreak_haven_cliffs', to: 'stormbreak_haven_return' }) }),
    Object.freeze({ id: 'stormbreak_haven_observatory', fromMapId: 'stormbreakHaven', toMapId: 'astralObservatory', type: 'field', routeId: 'ascension', requiredMapId: 'stormbreakCliffs', portalIds: Object.freeze({ from: 'stormbreak_haven_observatory', to: 'astral_observatory_stormbreak' }) }),
    Object.freeze({ id: 'astral_observatory_archive', fromMapId: 'astralObservatory', toMapId: 'astralArchive', type: 'field', routeId: 'ascension', portalIds: Object.freeze({ from: 'astral_observatory_archive', to: 'archive_observatory' }) }),
    Object.freeze({ id: 'archive_eclipse', fromMapId: 'astralArchive', toMapId: 'eclipseFrontier', type: 'field', routeId: 'ascension', requiredMapId: 'astralArchive', portalIds: Object.freeze({ from: 'archive_eclipse', to: 'eclipse_archive' }) }),
    Object.freeze({ id: 'eclipse_rift', fromMapId: 'eclipseFrontier', toMapId: 'endlessRift', type: 'field', routeId: 'ascension', requiredMapId: 'eclipseFrontier', portalIds: Object.freeze({ from: 'eclipse_rift', to: 'rift_eclipse' }) })
  ]);

  const api = {
    WORLD_AREAS,
    WORLD_ROUTES,
    REGIONAL_TOWN_IDS,
    SHOP_INTERIOR_WORLD_WIDTH,
    WORLD_MAP_NODES,
    WORLD_MAP_EDGES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.world = Object.assign({}, modules.world || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
