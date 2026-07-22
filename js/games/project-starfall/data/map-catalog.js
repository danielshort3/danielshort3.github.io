(function initProjectStarfallDataMapCatalog(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataMapTown = (typeof require === 'function' ? require('./map-town.js') : null) || DataModules.mapTown || {};
  const DataMapAssembly = (typeof require === 'function' ? require('./map-assembly.js') : null) || DataModules.mapAssembly || {};
  const DataMapPublication = (typeof require === 'function' ? require('./map-publication.js') : null) || DataModules.mapPublication || {};

  function createMapCatalogData(options) {
    const settings = options || {};
    const attachMapAssets = typeof settings.attachMapAssets === 'function'
      ? settings.attachMapAssets
      : DataMapPublication.attachMapAssets || ((map) => map);
    const withTownServiceNpcs = settings.withTownServiceNpcs || DataMapTown.withTownServiceNpcs;
    const createShopInteriorMaps = settings.createShopInteriorMaps || DataMapTown.createShopInteriorMaps;
    const makeTownHubMap = settings.makeTownHubMap || DataMapTown.makeTownHubMap;
    const applyPartyPlayGeometry = settings.applyPartyPlayGeometry || DataMapAssembly.applyPartyPlayGeometry;
    const makeExpandedTrainingMap = settings.makeExpandedTrainingMap || DataMapAssembly.makeExpandedTrainingMap;
    const makeBossRoomMap = settings.makeBossRoomMap || DataMapAssembly.makeBossRoomMap;

    const MAPS = Object.freeze([
      { id: 'starfallCrossing', name: 'Starfall Crossing', levelRange: [1, 99], safeZone: true, backgroundMode: 'panorama', palette: ['#101827', '#3d6575', '#d38b4c'], purpose: 'Fractured observatory frontier hub for expedition staging, repairs, shops, quest handoffs, storage, upgrades, and Starfall Plinko.', enemies: [], platforms: [[0, 520, 3800, 80], [420, 430, 260, 24], [980, 385, 300, 24], [1540, 430, 270, 24]], climbables: [], spawnPoints: [], stations: [{ id: 'storage', name: 'Storage Keeper', x: 420 }, { id: 'shop', name: 'Shopkeeper', x: 760 }, { id: 'slots', name: 'Slot Broker', x: 1090 }, { id: 'upgrade', name: 'Upgrade Artisan', x: 1460 }, { id: 'class', name: 'Class Supplier', x: 1840 }, { id: 'plinko', name: 'Starfall Plinko', x: 2260 }], questNpcs: withTownServiceNpcs('starfallCrossing', [{ id: 'crossing_class_master', name: 'Class Master', x: 2080, platformIndex: 0, questIds: ['trial_ready'], color: '#455e73', accent: '#d38b4c' }], { plinkoHost: { xOverride: 2260, platformIndex: 5 } }) },
      makeTownHubMap({
        id: 'rustcoilOutpost',
        name: 'Rustcoil Outpost',
        levelRange: [12, 34],
        palette: ['#8c6b35', '#7a8592', '#29b3ad'],
        purpose: 'Regional town for the Rustcoil Expanse with storage, slot services, and construct-field access.',
        shopName: 'Rustcoil Outfitter',
        questNpcs: [{ id: 'rustcoil_foreman', name: 'Rustcoil Foreman', x: 1660, platformIndex: 0, questIds: ['rustcoil_relay', 'gearworks_vault_report'], color: '#7a8592', accent: '#29b3ad' }]
      }),
      makeTownHubMap({
        id: 'cinderRefuge',
        name: 'Cinder Refuge',
        levelRange: [16, 45],
        palette: ['#28272d', '#f06b37', '#d8a531'],
        purpose: 'Volcanic safe hub between Oreback, Cinder Hollow, and Ashglass Pass.',
        shopName: 'Refuge Outfitter',
        questNpcs: [{ id: 'cinder_envoy', name: 'Cinder Envoy', x: 1660, platformIndex: 0, questIds: ['cinder_dispatch', 'ashglass_crossing'], color: '#9b4835', accent: '#ffcf70' }]
      }),
      makeTownHubMap({
        id: 'frostfenCamp',
        name: 'Frostfen Camp',
        levelRange: [40, 58],
        palette: ['#d8f4ff', '#6386a8', '#b7f2ff'],
        purpose: 'Frozen regional camp with shared storage and access to the Frostfen training routes.',
        shopName: 'Frostfen Outfitter',
        questNpcs: [{ id: 'frostfen_quartermaster', name: 'Frostfen Quartermaster', x: 1660, platformIndex: 0, questIds: ['frostfen_relay'], color: '#6386a8', accent: '#b7f2ff' }]
      }),
      makeTownHubMap({
        id: 'stormbreakHaven',
        name: 'Stormbreak Haven',
        levelRange: [52, 70],
        palette: ['#293b59', '#7bdff2', '#ffd166'],
        purpose: 'High-altitude town for Stormbreak routes with storage, slots, and upgrade access.',
        shopName: 'Haven Outfitter',
        questNpcs: [{ id: 'stormbreak_captain', name: 'Stormbreak Captain', x: 1660, platformIndex: 0, questIds: ['stormbreak_orders', 'astral_liaison'], color: '#4f6073', accent: '#ffe16a' }]
      }),
      makeTownHubMap({
        id: 'astralObservatory',
        name: 'Astral Observatory',
        levelRange: [60, 99],
        palette: ['#2b2856', '#8a6bcb', '#7bdff2'],
        purpose: 'Late-game town hub for Astral Dominion routes and shared account storage.',
        shopName: 'Astral Outfitter',
        questNpcs: [{ id: 'observatory_liaison', name: 'Observatory Liaison', x: 1660, platformIndex: 0, questIds: ['eclipse_frontier_message'], color: '#6b55a8', accent: '#7bdff2' }]
      }),
      {
        id: 'greenrootMeadow',
        name: 'Starfall Verge',
        levelRange: [1, 6],
        safeZone: false,
        geometryMode: 'generated',
        geometryGenerator: 'priorityFieldV2',
        compactWorldWidth: 4200,
        waveMax: 24,
        waveDelay: 5,
        palette: ['#18253b', '#4c6f88', '#d89b58'],
        purpose: 'Opening frontier route through a repair shelf, a starstone basin, and a fractured bridge.',
        enemies: ['faultSkitter', 'glassback', 'glassback', 'riftLantern', 'faultSkitter', 'glassback', 'glassback', 'riftLantern', 'faultSkitter', 'glassback', 'riftLantern', 'glassback'],
        platforms: [[0, 520, 7200, 80], [260, 452, 340, 22], [620, 388, 300, 22], [960, 318, 280, 22], [1360, 452, 410, 22], [1810, 382, 330, 22], [2180, 312, 290, 22], [2640, 452, 360, 22], [3040, 388, 320, 22], [3420, 322, 300, 22], [3840, 452, 420, 22], [4310, 382, 340, 22], [4700, 304, 300, 22], [5120, 452, 380, 22], [5560, 386, 340, 22], [5960, 316, 300, 22], [6320, 246, 260, 22], [6680, 452, 330, 22], [6900, 386, 260, 22]],
        climbables: [{ id: 'meadow_rope_1', x: 1088, y: 318, w: 26, h: 202 }, { id: 'meadow_rope_2', x: 2296, y: 312, w: 26, h: 208 }, { id: 'meadow_rope_3', x: 4814, y: 304, w: 26, h: 216 }, { id: 'meadow_rope_4', x: 6422, y: 246, w: 26, h: 274 }],
        spawnPoints: [{ x: 420, platformIndex: 1, weight: 3 }, { x: 790, platformIndex: 2, weight: 2 }, { x: 1110, platformIndex: 3, weight: 1 }, { x: 1570, platformIndex: 4, weight: 3 }, { x: 2310, platformIndex: 6, weight: 2 }, { x: 3190, platformIndex: 8, weight: 2 }, { x: 3560, platformIndex: 9, weight: 1 }, { x: 4480, platformIndex: 11, weight: 2 }, { x: 4860, platformIndex: 12, weight: 1 }, { x: 5740, platformIndex: 14, weight: 2 }, { x: 6430, platformIndex: 16, weight: 1 }, { x: 6840, platformIndex: 17, weight: 3 }],
        stations: [],
        questNpcs: [{ id: 'greenroot_guide', name: 'Verge Quartermaster', x: 320, platformIndex: 0, questIds: ['first_steps', 'greenroot_samples'], color: '#455e73', accent: '#d89b58' }]
      },
      {
        id: 'thornpathThicket',
        name: 'Thornpath Thicket',
        levelRange: [5, 16],
        safeZone: false,
        geometryGenerator: 'thornpathFractureCanopyV1',
        waveMax: 26,
        waveDelay: 5,
        palette: ['#233846', '#52746f', '#d5a85c'],
        purpose: 'A fractured starstone relay canopy with a drop-reset ascent and two readable route branches.',
        enemies: ['dewSlime', 'thornSprout', 'mossback', 'vineSnapper', 'briarStag', 'thornSprout', 'mossback', 'vineSnapper', 'thornSprout', 'briarStag', 'mossback', 'vineSnapper', 'dewSlime'],
        platforms: [[0, 520, 7600, 80], [260, 446, 390, 22], [700, 374, 310, 22], [1070, 300, 300, 22], [1460, 454, 420, 22], [1920, 386, 360, 22], [2350, 314, 310, 22], [2800, 452, 400, 22], [3240, 382, 340, 22], [3660, 302, 300, 22], [4080, 452, 440, 22], [4570, 380, 360, 22], [5000, 306, 320, 22], [5440, 452, 410, 22], [5900, 386, 340, 22], [6280, 318, 300, 22], [6600, 248, 290, 22], [6960, 380, 320, 22], [7300, 452, 260, 22]],
        climbables: [{ id: 'thicket_vine_1', x: 1180, y: 300, w: 28, h: 220 }, { id: 'thicket_vine_2', x: 2480, y: 314, w: 28, h: 206 }, { id: 'thicket_vine_3', x: 5120, y: 306, w: 28, h: 214 }, { id: 'thicket_vine_4', x: 6710, y: 248, w: 28, h: 272 }],
        spawnPoints: [{ x: 430, platformIndex: 1, weight: 2 }, { x: 820, platformIndex: 2, weight: 2 }, { x: 1200, platformIndex: 3, weight: 2 }, { x: 1680, platformIndex: 4, weight: 2 }, { x: 2110, platformIndex: 5, weight: 2 }, { x: 2510, platformIndex: 6, weight: 2 }, { x: 3030, platformIndex: 7, weight: 3 }, { x: 3840, platformIndex: 9, weight: 2 }, { x: 4310, platformIndex: 10, weight: 3 }, { x: 5170, platformIndex: 12, weight: 2 }, { x: 6070, platformIndex: 15, weight: 2 }, { x: 6740, platformIndex: 16, weight: 1 }, { x: 7410, platformIndex: 18, weight: 2 }],
        stations: [],
        questNpcs: [{ id: 'thornpath_scout', name: 'Thornpath Scout', x: 320, platformIndex: 0, questIds: ['field_scout', 'ridge_courier'], color: '#5d7f4e', accent: '#ffd166' }]
      },
      {
        id: 'brambleDepths',
        name: 'Bramble Depths',
        levelRange: [24, 34],
        safeZone: false,
        isDungeon: true,
        dungeonId: 'bramble_depths',
        waveMax: 8,
        waveDelay: 8,
        palette: ['#294d38', '#513325', '#e05b75'],
        purpose: 'MVP Alpha dungeon: layered vine platforms, thorn turrets, and the Brambleking boss.',
        enemies: ['thornSprout', 'vineSnapper', 'briarStag', 'thornSprout', 'brambleking', 'mossback', 'glowcapHealer', 'vineSnapper'],
        platforms: [[0, 520, 3400, 80], [280, 450, 420, 22], [760, 378, 350, 22], [1180, 306, 330, 22], [1580, 232, 300, 22], [1980, 448, 430, 22], [2480, 376, 360, 22], [2920, 304, 330, 22]],
        climbables: [{ id: 'bramble_vine_1', x: 900, y: 378, w: 28, h: 142 }, { id: 'bramble_vine_2', x: 1708, y: 232, w: 28, h: 288 }, { id: 'bramble_vine_3', x: 2630, y: 376, w: 28, h: 144 }],
        spawnPoints: [{ x: 500, platformIndex: 1, weight: 2 }, { x: 930, platformIndex: 2, weight: 2 }, { x: 1350, platformIndex: 3, weight: 1 }, { x: 1730, platformIndex: 4, weight: 1 }, { x: 2200, platformIndex: 5, weight: 2 }, { x: 2660, platformIndex: 6, weight: 2 }, { x: 3100, platformIndex: 7, weight: 1 }, { x: 1800, platformIndex: 0, weight: 2 }],
        stations: []
      },
      {
        id: 'rustcoilRuins',
        name: 'Rustcoil Ruins',
        levelRange: [12, 22],
        safeZone: false,
        compactWorldWidth: 5200,
        geometryGenerator: 'rustcoilOrreryCircuitV1',
        waveMax: 28,
        waveDelay: 5,
        palette: ['#8c6b35', '#7a8592', '#29b3ad'],
        purpose: 'A fractured celestial machine linking a survey yard, parallel switchworks, and the Warden Starcoil drop circuit.',
        enemies: ['rustRatchet', 'clockbug', 'coilSentry', 'rustRatchet', 'clockbug', 'coilSentry', 'rustRatchet', 'clockbug', 'scrapWarden', 'clockbug', 'rustRatchet', 'coilSentry', 'clockbug', 'scrapWarden'],
        platforms: [[0, 520, 7900, 80], [250, 446, 460, 22], [760, 374, 360, 22], [1180, 302, 320, 22], [1600, 232, 280, 22], [2040, 452, 420, 22], [2520, 382, 380, 22], [2980, 312, 340, 22], [3420, 452, 420, 22], [3900, 380, 360, 22], [4300, 304, 330, 22], [4720, 234, 280, 22], [5160, 452, 460, 22], [5680, 380, 360, 22], [6100, 308, 320, 22], [6500, 238, 300, 22], [6900, 452, 420, 22], [7360, 384, 330, 22], [7680, 314, 220, 22]],
        climbables: [{ id: 'ruins_ladder_1', x: 1300, y: 302, w: 30, h: 218 }, { id: 'ruins_ladder_2', x: 1722, y: 232, w: 30, h: 288 }, { id: 'ruins_ladder_3', x: 4435, y: 304, w: 30, h: 216 }, { id: 'ruins_ladder_4', x: 4830, y: 234, w: 30, h: 286 }, { id: 'ruins_ladder_5', x: 6628, y: 238, w: 30, h: 282 }],
        spawnPoints: [{ x: 500, platformIndex: 1, weight: 2 }, { x: 940, platformIndex: 2, weight: 2 }, { x: 1350, platformIndex: 3, weight: 2 }, { x: 1740, platformIndex: 4, weight: 1 }, { x: 2250, platformIndex: 5, weight: 3 }, { x: 2700, platformIndex: 6, weight: 2 }, { x: 3150, platformIndex: 7, weight: 2 }, { x: 3650, platformIndex: 8, weight: 3 }, { x: 4470, platformIndex: 10, weight: 2 }, { x: 4850, platformIndex: 11, weight: 1 }, { x: 5390, platformIndex: 12, weight: 3 }, { x: 6250, platformIndex: 14, weight: 2 }, { x: 6640, platformIndex: 15, weight: 1 }, { x: 7520, platformIndex: 17, weight: 2 }],
        stations: [],
        questNpcs: [{ id: 'ruins_surveyor', name: 'Rustcoil Surveyor', x: 520, platformIndex: 0, questIds: ['rustcoil_reclamation'], color: '#7a8592', accent: '#29b3ad' }]
      },
      {
        id: 'gearworksVault',
        name: 'Gearworks Vault',
        levelRange: [30, 48],
        safeZone: false,
        isDungeon: true,
        dungeonId: 'gearworks_vault',
        waveMax: 9,
        waveDelay: 8,
        palette: ['#665b48', '#7a8592', '#29b3ad'],
        purpose: 'MVP Alpha dungeon: construct gauntlets, armor-break checks, and two heavy boss targets.',
        enemies: ['rustRatchet', 'clockbug', 'clockworkTitan', 'coilSentry', 'orebackBeetle', 'quarryColossus', 'scrapWarden', 'clockbug', 'orebackBeetle'],
        platforms: [[0, 520, 3800, 80], [300, 448, 460, 22], [820, 374, 370, 22], [1260, 300, 340, 22], [1680, 226, 320, 22], [2140, 448, 470, 22], [2680, 374, 380, 22], [3120, 300, 350, 22], [3480, 226, 300, 22]],
        climbables: [{ id: 'vault_ladder_1', x: 960, y: 374, w: 30, h: 146 }, { id: 'vault_ladder_2', x: 1818, y: 226, w: 30, h: 294 }, { id: 'vault_ladder_3', x: 3260, y: 300, w: 30, h: 220 }],
        spawnPoints: [{ x: 540, platformIndex: 1, weight: 2 }, { x: 980, platformIndex: 2, weight: 2 }, { x: 1420, platformIndex: 3, weight: 1 }, { x: 1840, platformIndex: 4, weight: 1 }, { x: 2380, platformIndex: 5, weight: 2 }, { x: 2860, platformIndex: 6, weight: 2 }, { x: 3300, platformIndex: 7, weight: 1 }, { x: 3600, platformIndex: 8, weight: 1 }, { x: 1900, platformIndex: 0, weight: 2 }],
        stations: []
      },
      {
        id: 'cinderHollow',
        name: 'Cinder Hollow',
        levelRange: [16, 40],
        safeZone: false,
        waveMax: 24,
        waveDelay: 6,
        palette: ['#28272d', '#f06b37', '#9b4835'],
        purpose: 'Volcanic cave with ember vents, lava channels, flying spirits, and Emberjaw Golem.',
        enemies: ['emberWisp', 'ashCrawler', 'lavaTick', 'cinderSpitter', 'emberWisp', 'lavaTick', 'ashCrawler', 'emberWisp', 'cinderSpitter', 'lavaTick', 'ashCrawler', 'emberWisp'],
        platforms: [[0, 520, 7800, 80], [280, 454, 380, 22], [720, 386, 340, 22], [1110, 316, 320, 22], [1480, 246, 280, 22], [1940, 454, 420, 22], [2440, 384, 360, 22], [2860, 314, 330, 22], [3260, 244, 300, 22], [3700, 454, 420, 22], [4180, 386, 340, 22], [4580, 312, 330, 22], [5000, 242, 300, 22], [5440, 454, 400, 22], [5900, 384, 360, 22], [6320, 314, 330, 22], [6720, 244, 300, 22], [7100, 386, 330, 22], [7480, 454, 280, 22]],
        climbables: [{ id: 'cinder_chain_1', x: 1242, y: 316, w: 28, h: 204 }, { id: 'cinder_chain_2', x: 1600, y: 246, w: 28, h: 274 }, { id: 'cinder_chain_3', x: 3390, y: 244, w: 28, h: 276 }, { id: 'cinder_chain_4', x: 5128, y: 242, w: 28, h: 278 }, { id: 'cinder_chain_5', x: 6844, y: 244, w: 28, h: 276 }],
        spawnPoints: [{ x: 470, platformIndex: 1, weight: 2 }, { x: 890, platformIndex: 2, weight: 2 }, { x: 1280, platformIndex: 3, weight: 2 }, { x: 1620, platformIndex: 4, weight: 1 }, { x: 2150, platformIndex: 5, weight: 3 }, { x: 3040, platformIndex: 7, weight: 2 }, { x: 3420, platformIndex: 8, weight: 1 }, { x: 3910, platformIndex: 9, weight: 3 }, { x: 4750, platformIndex: 11, weight: 2 }, { x: 5150, platformIndex: 12, weight: 1 }, { x: 5620, platformIndex: 13, weight: 3 }, { x: 6480, platformIndex: 15, weight: 2 }, { x: 6860, platformIndex: 16, weight: 1 }, { x: 7590, platformIndex: 18, weight: 2 }],
        stations: [],
        questNpcs: [{ id: 'cinder_pathfinder', name: 'Cinder Pathfinder', x: 7460, platformIndex: 0, questIds: ['emberjaw_lair', 'cinder_samples', 'emberjaw_report'], color: '#9b4835', accent: '#ffcf70' }]
      },
      {
        id: 'emberjawLair',
        name: 'Emberjaw Lair',
        levelRange: [25, 40],
        safeZone: false,
        isDungeon: true,
        dungeonId: 'emberjaw_lair',
        waveMax: 8,
        waveDelay: 8,
        palette: ['#241f24', '#ff7842', '#d8a531'],
        purpose: 'Vertical-slice dungeon arena with ember platforms, construct adds, and the Emberjaw Golem boss.',
        enemies: ['emberWisp', 'lavaTick', 'orebackBeetle', 'cinderSpitter', 'emberjawGolem', 'emberWisp', 'ashCrawler', 'orebackBeetle'],
        platforms: [[0, 520, 3600, 80], [320, 442, 420, 22], [820, 366, 360, 22], [1280, 292, 330, 22], [1680, 214, 300, 22], [2100, 442, 440, 22], [2600, 360, 390, 22], [3100, 286, 340, 22]],
        climbables: [{ id: 'lair_chain_1', x: 960, y: 366, w: 30, h: 154 }, { id: 'lair_chain_2', x: 1814, y: 214, w: 30, h: 306 }, { id: 'lair_chain_3', x: 2760, y: 360, w: 30, h: 160 }],
        spawnPoints: [{ x: 520, platformIndex: 1, weight: 2 }, { x: 1000, platformIndex: 2, weight: 2 }, { x: 1440, platformIndex: 3, weight: 1 }, { x: 1840, platformIndex: 4, weight: 1 }, { x: 2320, platformIndex: 5, weight: 2 }, { x: 2820, platformIndex: 6, weight: 2 }, { x: 3240, platformIndex: 7, weight: 1 }, { x: 1900, platformIndex: 0, weight: 2 }],
        stations: []
      },
      {
        id: 'banditRidgeCamp',
        name: 'Bandit Ridge Camp',
        levelRange: [18, 30],
        safeZone: false,
        geometryMode: 'generated',
        geometryGenerator: 'banditRidgeAuthoredV2',
        compactWorldWidth: 5400,
        waveMax: 30,
        waveDelay: 5,
        palette: ['#6f5132', '#4f7b63', '#c3995b'],
        purpose: 'Authored four-beat raid through a cutter barricade, staggered thrower camp, high rope bridge, and safe campfire regroup.',
        enemies: ['banditCutter', 'banditCutter', 'banditThrower', 'briarStag', 'banditCutter', 'banditThrower', 'banditCutter', 'vineSnapper', 'banditThrower', 'banditCutter', 'banditCutter', 'banditThrower', 'briarStag', 'banditCutter', 'banditThrower'],
        platforms: [[0, 520, 7700, 80], [300, 452, 400, 22], [760, 382, 350, 22], [1180, 312, 320, 22], [1580, 242, 300, 22], [2040, 452, 420, 22], [2520, 382, 360, 22], [2940, 312, 330, 22], [3340, 242, 300, 22], [3780, 452, 430, 22], [4280, 382, 350, 22], [4700, 312, 320, 22], [5100, 242, 300, 22], [5540, 452, 420, 22], [6020, 382, 360, 22], [6440, 312, 330, 22], [6840, 242, 300, 22], [7200, 382, 320, 22], [7480, 452, 220, 22]],
        climbables: [{ id: 'ridge_rope_1', x: 1308, y: 312, w: 28, h: 208 }, { id: 'ridge_rope_2', x: 1708, y: 242, w: 28, h: 278 }, { id: 'ridge_rope_3', x: 3470, y: 242, w: 28, h: 278 }, { id: 'ridge_rope_4', x: 5230, y: 242, w: 28, h: 278 }, { id: 'ridge_rope_5', x: 6968, y: 242, w: 28, h: 278 }],
        spawnPoints: [{ x: 500, platformIndex: 1, weight: 2 }, { x: 930, platformIndex: 2, weight: 2 }, { x: 1340, platformIndex: 3, weight: 2 }, { x: 1720, platformIndex: 4, weight: 2 }, { x: 2260, platformIndex: 5, weight: 3 }, { x: 2710, platformIndex: 6, weight: 2 }, { x: 3100, platformIndex: 7, weight: 2 }, { x: 3500, platformIndex: 8, weight: 2 }, { x: 3990, platformIndex: 9, weight: 3 }, { x: 4850, platformIndex: 11, weight: 2 }, { x: 5250, platformIndex: 12, weight: 2 }, { x: 5720, platformIndex: 13, weight: 3 }, { x: 6600, platformIndex: 15, weight: 2 }, { x: 6980, platformIndex: 16, weight: 2 }, { x: 7580, platformIndex: 18, weight: 2 }],
        stations: [],
        questNpcs: [{ id: 'ridge_watch', name: 'Ridge Watch', x: 480, platformIndex: 0, questIds: ['ridge_cleanup', 'bramble_crown_report'], color: '#6f5132', accent: '#c3995b' }]
      },
      {
        id: 'banditAnimationLab',
        name: 'Bandit Animation Lab',
        adminOnly: true,
        geometryMode: 'authored',
        layoutStyle: 'authoredComparisonRoom',
        layoutRole: 'trainingField',
        levelRange: [18, 30],
        safeZone: false,
        waveMax: 5,
        waveDelay: 5,
        palette: ['#6f5132', '#4f7b63', '#c3995b'],
        purpose: 'Admin-only comparison room for Bandit Cutter animation generation methods.',
        enemies: ['banditCutter', 'banditCutterDirect', 'banditCutterReference', 'banditCutterHybrid', 'banditCutterPuppet'],
        platforms: [[0, 520, 3600, 80], [260, 432, 420, 22], [900, 432, 420, 22], [1540, 432, 420, 22], [2180, 432, 420, 22], [2820, 432, 420, 22]],
        climbables: [],
        spawnPoints: [
          { id: 'baseline_bandit', x: 470, platformIndex: 1, weight: 1 },
          { id: 'direct_bandit', x: 1110, platformIndex: 2, weight: 1 },
          { id: 'reference_bandit', x: 1750, platformIndex: 3, weight: 1 },
          { id: 'hybrid_bandit', x: 2390, platformIndex: 4, weight: 1 },
          { id: 'puppet_bandit', x: 3030, platformIndex: 5, weight: 1 }
        ],
        fixedEnemySpawns: [
          { id: 'baseline_bandit', enemyId: 'banditCutter', x: 470, platformIndex: 1 },
          { id: 'direct_bandit', enemyId: 'banditCutterDirect', x: 1110, platformIndex: 2 },
          { id: 'reference_bandit', enemyId: 'banditCutterReference', x: 1750, platformIndex: 3 },
          { id: 'hybrid_bandit', enemyId: 'banditCutterHybrid', x: 2390, platformIndex: 4 },
          { id: 'puppet_bandit', enemyId: 'banditCutterPuppet', x: 3030, platformIndex: 5 }
        ],
        stations: [],
        questNpcs: []
      },
      {
        id: 'orebackQuarry',
        name: 'Oreback Quarry',
        levelRange: [24, 35],
        safeZone: false,
        geometryMode: 'generated',
        geometryGenerator: 'priorityFieldV2',
        compactWorldWidth: 4800,
        waveMax: 26,
        waveDelay: 6,
        palette: ['#6b6960', '#b58a4a', '#62c5a2'],
        purpose: 'Mining platforms, ore veins, scaffolds, mushroom pockets, and heavy lanes.',
        enemies: ['orebackBeetle', 'glowcapHealer', 'orebackBeetle', 'scrapWarden', 'orebackBeetle', 'glowcapHealer', 'orebackBeetle', 'coilSentry', 'orebackBeetle', 'crackedMimic', 'orebackBeetle', 'glowcapHealer', 'scrapWarden'],
        platforms: [[0, 520, 8000, 80], [260, 452, 460, 22], [780, 382, 370, 22], [1220, 312, 340, 22], [1640, 242, 320, 22], [2100, 452, 470, 22], [2640, 382, 380, 22], [3080, 312, 350, 22], [3500, 242, 320, 22], [3960, 452, 450, 22], [4480, 382, 380, 22], [4920, 312, 350, 22], [5360, 242, 320, 22], [5820, 452, 470, 22], [6360, 382, 380, 22], [6800, 312, 350, 22], [7220, 242, 320, 22], [7600, 382, 330, 22], [7820, 452, 180, 22]],
        climbables: [{ id: 'quarry_lift_1', x: 1360, y: 312, w: 30, h: 208 }, { id: 'quarry_lift_2', x: 1780, y: 242, w: 30, h: 278 }, { id: 'quarry_lift_3', x: 3640, y: 242, w: 30, h: 278 }, { id: 'quarry_lift_4', x: 5500, y: 242, w: 30, h: 278 }, { id: 'quarry_lift_5', x: 7360, y: 242, w: 30, h: 278 }],
        spawnPoints: [{ x: 500, platformIndex: 1, weight: 2 }, { x: 960, platformIndex: 2, weight: 2 }, { x: 1400, platformIndex: 3, weight: 2 }, { x: 1800, platformIndex: 4, weight: 1 }, { x: 2350, platformIndex: 5, weight: 3 }, { x: 2830, platformIndex: 6, weight: 2 }, { x: 3260, platformIndex: 7, weight: 2 }, { x: 3660, platformIndex: 8, weight: 1 }, { x: 4180, platformIndex: 9, weight: 3 }, { x: 5100, platformIndex: 11, weight: 2 }, { x: 5520, platformIndex: 12, weight: 1 }, { x: 6070, platformIndex: 13, weight: 3 }, { x: 6980, platformIndex: 15, weight: 2 }, { x: 7380, platformIndex: 16, weight: 1 }, { x: 7900, platformIndex: 18, weight: 2 }],
        stations: [],
        questNpcs: [{ id: 'quarry_foreman', name: 'Quarry Foreman', x: 520, platformIndex: 0, questIds: ['quarry_contract'], color: '#6b6960', accent: '#62c5a2' }]
      },
      makeExpandedTrainingMap({
        id: 'ashglassPass',
        name: 'Ashglass Pass',
        levelRange: [40, 55],
        palette: ['#4d3d47', '#f06b37', '#d8a531'],
        purpose: 'High-level volcanic glass trails for post-dungeon training and elite material routes.',
        enemies: ['emberWisp', 'ashCrawler', 'lavaTick', 'cinderSpitter', 'emberWisp', 'orebackBeetle', 'lavaTick', 'crackedMimic', 'cinderSpitter', 'emberWisp', 'ashCrawler', 'lavaTick', 'orebackBeetle', 'emberWisp'],
        questNpcs: [{ id: 'ashglass_courier', name: 'Ashglass Courier', x: 520, platformIndex: 0, questIds: [], color: '#7a4d5b', accent: '#ffcf70' }]
      }),
      makeExpandedTrainingMap({
        id: 'frostfenOutskirts',
        name: 'Frostfen Outskirts',
        levelRange: [45, 58],
        compactWorldWidth: 5600,
        geometryGenerator: 'frostfenMarshRunV1',
        palette: ['#d7f3ff', '#5ca8e8', '#f7fbff'],
        purpose: 'Snowfield platforms and frozen marsh lanes that introduce slick footing and frost enemy packs.',
        enemies: ['shardling', 'frostlingScout', 'snowglareWisp', 'rimebackBrute', 'shardling', 'frostlingScout', 'icebloomOracle', 'snowglareWisp', 'rimebackBrute', 'shardling', 'frostlingScout', 'icebloomOracle'],
        movementProfile: 'ice',
        areaMechanic: 'Ice footing reduces ground acceleration and increases sliding.',
        waveMax: 31,
        waveDelay: 5,
        questNpcs: [{ id: 'frostfen_tracker', name: 'Frostfen Tracker', x: 520, platformIndex: 2, questIds: ['frostfen_field_notes'], color: '#6386a8', accent: '#b7f2ff' }]
      }),
      makeExpandedTrainingMap({
        id: 'glacierSpine',
        name: 'Glacier Spine',
        levelRange: [52, 68],
        palette: ['#9edcff', '#2f6fa6', '#eaf8ff'],
        purpose: 'Jagged glacier ridges with longer platform gaps, frost flyers, and armor-crack brute lanes.',
        enemies: ['rimebackBrute', 'snowglareWisp', 'glacierSentinel', 'frostlingScout', 'snowglareWisp', 'crackedMimic', 'rimebackBrute', 'shardling', 'glacierSentinel', 'icebloomOracle', 'rimebackBrute', 'snowglareWisp'],
        movementProfile: 'ice',
        areaMechanic: 'Ice footing rewards planned movement and mobility skills.',
        waveMax: 32,
        waveDelay: 5,
        questNpcs: [{ id: 'glacier_cartographer', name: 'Glacier Cartographer', x: 520, platformIndex: 0, questIds: ['glacier_cartography', 'rimewarden_sanctum_report'], color: '#2f6fa6', accent: '#eaf8ff' }]
      }),
      {
        id: 'rimewardenSanctum',
        name: 'Rimewarden Sanctum',
        levelRange: [58, 70],
        safeZone: false,
        isDungeon: true,
        dungeonId: 'rimewarden_sanctum',
        movementProfile: 'ice',
        areaMechanic: 'Frozen arena footing adds slide while the Rimewarden controls space.',
        waveMax: 9,
        waveDelay: 8,
        palette: ['#d7f3ff', '#2f6fa6', '#f7fbff'],
        purpose: 'Frostfen dungeon arena with slick platforms, frost brutes, flying spirits, and the Rimewarden boss.',
        enemies: ['frostlingScout', 'snowglareWisp', 'rimebackBrute', 'glacierSentinel', 'rimewarden', 'snowglareWisp', 'icebloomOracle', 'rimebackBrute', 'shardling'],
        platforms: [[0, 520, 3800, 80], [300, 448, 460, 22], [820, 374, 370, 22], [1260, 300, 340, 22], [1680, 226, 320, 22], [2140, 448, 470, 22], [2680, 374, 380, 22], [3120, 300, 350, 22], [3480, 226, 300, 22]],
        climbables: [{ id: 'sanctum_chain_1', x: 960, y: 374, w: 30, h: 146 }, { id: 'sanctum_chain_2', x: 1818, y: 226, w: 30, h: 294 }, { id: 'sanctum_chain_3', x: 3260, y: 300, w: 30, h: 220 }],
        spawnPoints: [{ x: 540, platformIndex: 1, weight: 2 }, { x: 980, platformIndex: 2, weight: 2 }, { x: 1420, platformIndex: 3, weight: 1 }, { x: 1840, platformIndex: 4, weight: 1 }, { x: 2380, platformIndex: 5, weight: 2 }, { x: 2860, platformIndex: 6, weight: 2 }, { x: 3300, platformIndex: 7, weight: 1 }, { x: 3600, platformIndex: 8, weight: 1 }, { x: 1900, platformIndex: 0, weight: 2 }],
        stations: []
      },
      makeExpandedTrainingMap({
        id: 'stormbreakCliffs',
        name: 'Stormbreak Cliffs',
        levelRange: [55, 70],
        palette: ['#4f6073', '#91dbe8', '#ffe16a'],
        purpose: 'Wind-cut cliff platforms with dense ranged pressure and long mobility lanes.',
        enemies: ['galeHarrier', 'stormboundArcher', 'thunderRam', 'galeHarrier', 'stormboundArcher', 'cloudcallAcolyte', 'thunderRam', 'crackedMimic', 'galeHarrier', 'stormboundArcher', 'thunderRam', 'cloudcallAcolyte', 'galeHarrier', 'stormboundArcher'],
        waveMax: 32,
        questNpcs: [{ id: 'stormbreak_scout', name: 'Stormbreak Scout', x: 520, platformIndex: 0, questIds: ['stormbreak_rods'], color: '#4f6073', accent: '#ffe16a' }]
      }),
      makeExpandedTrainingMap({
        id: 'astralArchive',
        name: 'Astral Archive',
        levelRange: [70, 85],
        palette: ['#29365f', '#28c7b7', '#c794ff'],
        purpose: 'Ancient archive stacks and rune walkways tuned for late-game mobbing practice.',
        enemies: ['indexScribe', 'lumenSentinel', 'voidMote', 'indexScribe', 'lumenSentinel', 'crackedMimic', 'voidMote', 'indexScribe', 'lumenSentinel', 'voidMote', 'indexScribe', 'lumenSentinel', 'voidMote', 'indexScribe'],
        waveMax: 34,
        questNpcs: [{ id: 'astral_scribe', name: 'Astral Scribe', x: 520, platformIndex: 0, questIds: ['astral_indexing'], color: '#29365f', accent: '#c794ff' }]
      }),
      makeExpandedTrainingMap({
        id: 'eclipseFrontier',
        name: 'Eclipse Frontier',
        levelRange: [85, 100],
        palette: ['#1f2330', '#ffbe55', '#7bdff2'],
        purpose: 'Outer frontier training grounds with elite-density routes before endless scaling.',
        enemies: ['eclipseDuelist', 'lumenSentinel', 'voidMote', 'indexScribe', 'eclipseDuelist', 'crackedMimic', 'voidMote', 'lumenSentinel', 'eclipseDuelist', 'indexScribe', 'voidMote', 'lumenSentinel', 'eclipseDuelist', 'voidMote'],
        waveMax: 34,
        waveDelay: 6,
        questNpcs: [{ id: 'eclipse_envoy', name: 'Eclipse Frontier Envoy', x: 520, platformIndex: 0, questIds: ['eclipse_frontier_patrol'], color: '#1f2330', accent: '#ffbe55' }]
      }),
      makeExpandedTrainingMap({
        id: 'endlessRift',
        name: 'Endless Rift',
        levelRange: [100, 100],
        palette: ['#191b2c', '#7bdff2', '#f06bff'],
        purpose: 'Scaling rift training that follows uncapped player levels after level 100.',
        enemies: ['riftAberration', 'voidMote', 'lumenSentinel', 'eclipseDuelist', 'indexScribe', 'voidMote', 'riftAberration', 'crackedMimic', 'eclipseDuelist', 'voidMote', 'lumenSentinel', 'indexScribe', 'riftAberration', 'voidMote'],
        waveMax: 36,
        waveDelay: 5,
        endlessScaling: true,
        questNpcs: [{ id: 'rift_watcher', name: 'Rift Watcher', x: 520, platformIndex: 0, questIds: ['rift_watch'], color: '#252845', accent: '#f06bff' }]
      }),
      makeBossRoomMap({
        id: 'bramblekingCourt',
        name: 'Brambleking Court',
        bossId: 'brambleking',
        levelRange: [32, 38],
        palette: ['#294d38', '#513325', '#e05b75'],
        purpose: 'Custom boss echo with root walls, thorn volleys, and a crowned vulnerability window.',
        enemies: ['brambleking', 'thornSprout', 'vineSnapper', 'glowcapHealer', 'briarStag'],
        questNpcs: [{ id: 'bramble_witness', name: 'Bramble Witness', x: 520, platformIndex: 0, questIds: ['brambleking_echo'], color: '#294d38', accent: '#e05b75' }]
      }),
      makeBossRoomMap({
        id: 'titanFoundry',
        name: 'Titan Foundry',
        bossId: 'clockworkTitan',
        levelRange: [50, 58],
        palette: ['#665b48', '#7a8592', '#29b3ad'],
        purpose: 'Custom boss echo with rotating gear lanes, overclock bursts, and exposed armor plates.',
        enemies: ['clockworkTitan', 'clockbug', 'coilSentry', 'rustRatchet', 'scrapWarden'],
        questNpcs: [{ id: 'titan_witness', name: 'Titan Witness', x: 520, platformIndex: 0, questIds: ['titan_foundry_echo'], color: '#665b48', accent: '#29b3ad' }]
      }),
      makeBossRoomMap({
        id: 'deepcoreCore',
        name: 'Deepcore Core',
        bossId: 'quarryColossus',
        levelRange: [60, 70],
        palette: ['#6b6960', '#b58a4a', '#62c5a2'],
        purpose: 'Custom boss echo with falling ore, quake anchors, and split-lane add pressure.',
        enemies: ['quarryColossus', 'orebackBeetle', 'glowcapHealer', 'scrapWarden', 'coilSentry'],
        questNpcs: [{ id: 'deepcore_witness', name: 'Deepcore Witness', x: 520, platformIndex: 0, questIds: ['deepcore_echo'], color: '#6b6960', accent: '#62c5a2' }]
      }),
      makeBossRoomMap({
        id: 'emberjawFurnace',
        name: 'Emberjaw Furnace',
        bossId: 'emberjawGolem',
        levelRange: [42, 50],
        palette: ['#241f24', '#ff7842', '#d8a531'],
        purpose: 'Custom boss echo with furnace cracks, lava charges, and overheat punish windows.',
        enemies: ['emberjawGolem', 'emberWisp', 'lavaTick', 'cinderSpitter', 'ashCrawler'],
        questNpcs: [{ id: 'emberjaw_witness', name: 'Emberjaw Witness', x: 520, platformIndex: 0, questIds: ['emberjaw_echo'], color: '#9b4835', accent: '#ffcf70' }]
      }),
      makeBossRoomMap({
        id: 'rimewardenVault',
        name: 'Rimewarden Vault',
        bossId: 'rimewarden',
        levelRange: [66, 74],
        movementProfile: 'ice',
        areaMechanic: 'Slick footing and whiteout lines reward deliberate vertical repositioning.',
        palette: ['#d7f3ff', '#2f6fa6', '#f7fbff'],
        purpose: 'Custom boss echo with ice walls, whiteout lanes, and frost-ring shockwaves.',
        enemies: ['rimewarden', 'frostlingScout', 'snowglareWisp', 'glacierSentinel', 'icebloomOracle'],
        questNpcs: [{ id: 'rimewarden_witness', name: 'Rimewarden Witness', x: 520, platformIndex: 0, questIds: ['rimewarden_echo'], color: '#2f6fa6', accent: '#eaf8ff' }]
      }),
      makeBossRoomMap({
        id: 'stormbreakAerie',
        name: 'Stormbreak Aerie',
        bossId: 'stormbreakRoc',
        levelRange: [76, 86],
        palette: ['#4f6073', '#91dbe8', '#ffe16a'],
        purpose: 'Custom boss echo with lightning rods, wind lanes, and high-speed divebombs.',
        enemies: ['stormbreakRoc', 'galeHarrier', 'stormboundArcher', 'thunderRam', 'cloudcallAcolyte'],
        questNpcs: [{ id: 'stormbreak_witness', name: 'Stormbreak Witness', x: 520, platformIndex: 0, questIds: ['stormbreak_echo'], color: '#4f6073', accent: '#ffe16a' }]
      }),
      makeBossRoomMap({
        id: 'astralStacks',
        name: 'Astral Stacks',
        bossId: 'astralArchivist',
        levelRange: [88, 98],
        palette: ['#29365f', '#28c7b7', '#c794ff'],
        purpose: 'Custom boss echo with rune pages, mirrored attacks, and action-memory seals.',
        enemies: ['astralArchivist', 'indexScribe', 'lumenSentinel', 'voidMote', 'cloudcallAcolyte'],
        questNpcs: [{ id: 'astral_witness', name: 'Astral Witness', x: 520, platformIndex: 0, questIds: ['astral_echo'], color: '#29365f', accent: '#c794ff' }]
      }),
      makeBossRoomMap({
        id: 'eclipseThrone',
        name: 'Eclipse Throne',
        bossId: 'eclipseSovereign',
        levelRange: [100, 112],
        backgroundMode: 'panorama',
        palette: ['#1f2330', '#ffbe55', '#7bdff2'],
        purpose: 'Custom boss echo with solar and lunar stance swaps, eclipse sigils, and totality burst windows.',
        enemies: ['eclipseSovereign', 'eclipseDuelist', 'voidMote', 'lumenSentinel', 'indexScribe'],
        questNpcs: [{ id: 'eclipse_witness', name: 'Eclipse Witness', x: 520, platformIndex: 0, questIds: ['eclipse_echo'], color: '#1f2330', accent: '#ffbe55' }]
      })
    ].concat(createShopInteriorMaps()).map(applyPartyPlayGeometry).map(attachMapAssets));

    return Object.freeze({
      MAPS
    });
  }

  const defaultMapCatalogData = createMapCatalogData();
  const api = Object.assign({
    createMapCatalogData
  }, defaultMapCatalogData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapCatalog = Object.assign({}, modules.mapCatalog || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
