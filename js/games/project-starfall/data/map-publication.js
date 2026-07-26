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
      { sectionSuffix: 'starter_pond_loop', label: 'Pond Slimelets', enemyWeights: [{ enemyId: 'slimelet', weight: 1 }], population: 4, maxPopulation: 4, respawnSeconds: 7, leash: 240, spawnBounds: { minX: 720, maxX: 1640 } },
      { sectionSuffix: 'moss_lane_extension', label: 'Dewdrop Shelf', enemyWeights: [{ enemyId: 'dewSlime', weight: 1 }], population: 4, maxPopulation: 4, respawnSeconds: 7, leash: 280, spawnBounds: { minX: 1700, maxX: 2680 } },
      { sectionSuffix: 'canopy_practice', label: 'Canopy Sprouts', enemyWeights: [{ enemyId: 'dewSlime', weight: 4 }, { enemyId: 'slimelet', weight: 3 }, { enemyId: 'thornSprout', weight: 3 }], population: 5, maxPopulation: 5, respawnSeconds: 8, leash: 300, spawnBounds: { minX: 2720, maxX: 3340 } },
      { sectionSuffix: 'thornpath_gate', label: 'Gate Guardians', enemyWeights: [{ enemyId: 'dewSlime', weight: 2 }, { enemyId: 'slimelet', weight: 1 }, { enemyId: 'thornSprout', weight: 3 }, { enemyId: 'mossback', weight: 4 }], population: 5, maxPopulation: 5, respawnSeconds: 9, leash: 340, spawnBounds: { minX: 3500, maxX: 3640 } }
    ]),
    thornpathThicket: Object.freeze([
      { sectionSuffix: 'scout_s_clearing', label: 'Scout’s Clearing', enemyWeights: [{ enemyId: 'dewSlime', weight: 4 }, { enemyId: 'mossback', weight: 3 }, { enemyId: 'thornSprout', weight: 3 }], population: 5, maxPopulation: 5, respawnSeconds: 7, leash: 280, spawnBounds: { minX: 760, maxX: 1180 } },
      { sectionSuffix: 'vine_tangle', label: 'Vine Tangle', enemyWeights: [{ enemyId: 'mossback', weight: 4 }, { enemyId: 'thornSprout', weight: 4 }, { enemyId: 'vineSnapper', weight: 2 }], population: 6, maxPopulation: 6, respawnSeconds: 8, leash: 340, spawnBounds: { minX: 1350, maxX: 2380 } },
      { sectionSuffix: 'thorn_canopy', label: 'Thorn Canopy', enemyWeights: [{ enemyId: 'thornSprout', weight: 4 }, { enemyId: 'vineSnapper', weight: 4 }, { enemyId: 'briarStag', weight: 2 }], population: 7, maxPopulation: 7, respawnSeconds: 8, leash: 380, spawnBounds: { minX: 2650, maxX: 3900 } },
      { sectionSuffix: 'deep_fork', label: 'Deep Fork', enemyWeights: [{ enemyId: 'mossback', weight: 3 }, { enemyId: 'vineSnapper', weight: 4 }, { enemyId: 'briarStag', weight: 3 }], population: 6, maxPopulation: 6, respawnSeconds: 9, leash: 360, spawnBounds: { minX: 4060, maxX: 4400 } }
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
    gearworksVault: Object.freeze([
      {
        sectionSuffix: 'intake_tank_lane',
        label: 'Intake Constructs',
        platformIds: ['gearworksVault_intake_lane', 'gearworksVault_intake_catwalk'],
        enemyWeights: [{ enemyId: 'rustRatchet', weight: 4 }, { enemyId: 'clockbug', weight: 3 }, { enemyId: 'scrapWarden', weight: 2 }, { enemyId: 'coilSentry', weight: 1 }],
        population: 4,
        maxPopulation: 4,
        respawnSeconds: 8,
        leash: 420,
        spawnBounds: { minX: 340, maxX: 1060 }
      },
      {
        sectionSuffix: 'titan_assembly',
        label: 'Titan Assembly',
        platformIds: ['gearworksVault_titan_floor', 'gearworksVault_sentry_catwalk'],
        enemyWeights: [{ enemyId: 'coilSentry', weight: 3 }, { enemyId: 'scrapWarden', weight: 2 }, { enemyId: 'clockworkTitan', weight: 1 }],
        population: 3,
        maxPopulation: 3,
        respawnSeconds: 10,
        leash: 520,
        spawnBounds: { minX: 1380, maxX: 2180 }
      },
      {
        sectionSuffix: 'assembly_core',
        label: 'Assembly Core',
        platformIds: ['gearworksVault_core_floor', 'gearworksVault_core_catwalk'],
        enemyWeights: [{ enemyId: 'orebackBeetle', weight: 3 }, { enemyId: 'scrapWarden', weight: 2 }, { enemyId: 'quarryColossus', weight: 1 }],
        population: 2,
        maxPopulation: 2,
        respawnSeconds: 12,
        leash: 620,
        spawnBounds: { minX: 3200, maxX: 4200 }
      }
    ]),
    cinderHollow: Object.freeze([
      {
        sectionSuffix: 'ash_floor',
        label: 'Ash Floor',
        enemyWeights: [{ enemyId: 'ashCrawler', weight: 5 }, { enemyId: 'lavaTick', weight: 4 }, { enemyId: 'cinderSpitter', weight: 1 }],
        population: 8,
        maxPopulation: 8,
        respawnSeconds: 6,
        leash: 360,
        spawnBounds: { minX: 620, maxX: 1400 }
      },
      {
        sectionSuffix: 'vent_shortcut',
        label: 'Vent Shortcut',
        enemyWeights: [{ enemyId: 'lavaTick', weight: 5 }, { enemyId: 'cinderSpitter', weight: 4 }, { enemyId: 'ashCrawler', weight: 1 }],
        population: 8,
        maxPopulation: 8,
        respawnSeconds: 6,
        leash: 340,
        spawnBounds: { minX: 1500, maxX: 2800 }
      },
      {
        sectionSuffix: 'wisp_turn',
        label: 'Wisp Turn',
        enemyWeights: [{ enemyId: 'emberWisp', weight: 6 }, { enemyId: 'cinderSpitter', weight: 3 }, { enemyId: 'lavaTick', weight: 1 }],
        population: 8,
        maxPopulation: 8,
        respawnSeconds: 7,
        leash: 380,
        spawnBounds: { minX: 2900, maxX: 4260 }
      }
    ]),
    frostfenOutskirts: Object.freeze([
      {
        sectionSuffix: 'marsh_flats',
        label: 'Marsh Flats',
        enemyWeights: [
          { enemyId: 'shardling', weight: 5 },
          { enemyId: 'frostlingScout', weight: 4 },
          { enemyId: 'rimebackBrute', weight: 1 }
        ],
        population: 10,
        maxPopulation: 10,
        respawnSeconds: 6,
        leash: 420,
        spawnBounds: { minX: 520, maxX: 1500 }
      },
      {
        sectionSuffix: 'ice_shelves',
        label: 'Ice Shelves',
        enemyWeights: [
          { enemyId: 'frostlingScout', weight: 3 },
          { enemyId: 'snowglareWisp', weight: 4 },
          { enemyId: 'rimebackBrute', weight: 3 },
          { enemyId: 'shardling', weight: 1 }
        ],
        population: 11,
        maxPopulation: 11,
        respawnSeconds: 7,
        leash: 500,
        spawnBounds: { minX: 2460, maxX: 5700 }
      },
      {
        sectionSuffix: 'oracle_grove',
        label: 'Oracle Grove',
        enemyWeights: [
          { enemyId: 'icebloomOracle', weight: 5 },
          { enemyId: 'snowglareWisp', weight: 3 },
          { enemyId: 'rimebackBrute', weight: 2 },
          { enemyId: 'frostlingScout', weight: 1 }
        ],
        population: 11,
        maxPopulation: 11,
        respawnSeconds: 8,
        leash: 460,
        spawnBounds: { minX: 6600, maxX: 7480 }
      }
    ]),
    rimewardenSanctum: Object.freeze([
      {
        sectionSuffix: 'brute_lane',
        label: 'Brute Gate Guard',
        platformIds: [
          'rimewarden_sanctum_solid_lane_01',
          'rimewarden_sanctum_solid_lane_02'
        ],
        enemyWeights: [
          { enemyId: 'rimebackBrute', weight: 5 },
          { enemyId: 'frostlingScout', weight: 3 },
          { enemyId: 'shardling', weight: 2 }
        ],
        population: 4,
        maxPopulation: 4,
        respawnSeconds: 9,
        leash: 360,
        spawnBounds: { minX: 760, maxX: 1490 }
      },
      {
        sectionSuffix: 'oracle_shelf',
        label: 'Whiteout Shelf Guard',
        platformIds: [
          'rimewarden_sanctum_solid_lane_03',
          'rimewarden_sanctum_solid_lane_04',
          'rimewarden_sanctum_solid_lane_06'
        ],
        enemyWeights: [
          { enemyId: 'icebloomOracle', weight: 5 },
          { enemyId: 'snowglareWisp', weight: 3 },
          { enemyId: 'glacierSentinel', weight: 2 }
        ],
        population: 4,
        maxPopulation: 4,
        respawnSeconds: 10,
        leash: 420,
        spawnBounds: { minX: 1600, maxX: 3000 }
      },
      {
        sectionSuffix: 'sentinel_shelf',
        label: 'Sentinel Seal Guard',
        platformIds: ['rimewarden_sanctum_solid_lane_05'],
        enemyWeights: [
          { enemyId: 'glacierSentinel', weight: 5 },
          { enemyId: 'snowglareWisp', weight: 3 },
          { enemyId: 'rimebackBrute', weight: 2 }
        ],
        population: 1,
        maxPopulation: 1,
        respawnSeconds: 12,
        leash: 360,
        spawnBounds: { minX: 3100, maxX: 3460 }
      }
    ]),
    stormbreakCliffs: Object.freeze([
      {
        sectionSuffix: 'low_ram_lane',
        label: 'Thunder Ram Lane',
        platformIds: [
          'stormbreak_cliffs_solid_lane_01',
          'stormbreak_cliffs_solid_lane_02',
          'stormbreak_cliffs_solid_lane_03'
        ],
        enemyWeights: [{ enemyId: 'thunderRam', weight: 8 }, { enemyId: 'cloudcallAcolyte', weight: 2 }],
        population: 12,
        maxPopulation: 15,
        respawnSeconds: 5,
        leash: 500,
        spawnBounds: { minX: 520, maxX: 1280 }
      },
      {
        sectionSuffix: 'mid_archer_bridge',
        label: 'Archer Bridge',
        platformIds: [
          'stormbreak_cliffs_solid_lane_04',
          'stormbreak_cliffs_solid_lane_05',
          'stormbreak_cliffs_solid_lane_06'
        ],
        enemyWeights: [{ enemyId: 'stormboundArcher', weight: 7 }, { enemyId: 'cloudcallAcolyte', weight: 3 }],
        population: 12,
        maxPopulation: 15,
        respawnSeconds: 5,
        leash: 440,
        spawnBounds: { minX: 1680, maxX: 2780 }
      },
      {
        sectionSuffix: 'high_harrier_airspace',
        label: 'Harrier Airspace',
        platformIds: [
          'stormbreak_cliffs_solid_lane_07',
          'stormbreak_cliffs_solid_lane_08',
          'stormbreak_cliffs_solid_lane_09',
          'stormbreak_cliffs_solid_lane_10'
        ],
        enemyWeights: [{ enemyId: 'galeHarrier', weight: 8 }, { enemyId: 'stormboundArcher', weight: 2 }],
        population: 12,
        maxPopulation: 15,
        respawnSeconds: 5,
        leash: 540,
        spawnBounds: { minX: 2880, maxX: 4190 },
        actorTraversal: { mode: 'air', allowLadders: false, allowRamps: true, stayInTerritory: true }
      }
    ]),
    eclipseFrontier: Object.freeze([
      {
        sectionSuffix: 'solar_outpost',
        label: 'Solar Sentinels',
        platformIds: [
          'eclipse_frontier_solid_lane_01',
          'eclipse_frontier_solid_lane_02',
          'eclipse_frontier_solid_lane_03'
        ],
        enemyWeights: [
          { enemyId: 'lumenSentinel', weight: 6 },
          { enemyId: 'indexScribe', weight: 2 },
          { enemyId: 'eclipseDuelist', weight: 2 }
        ],
        population: 8,
        maxPopulation: 10,
        respawnSeconds: 6,
        leash: 460,
        spawnBounds: { minX: 320, maxX: 1500 }
      },
      {
        sectionSuffix: 'lunar_outpost',
        label: 'Lunar Motes',
        platformIds: [
          'eclipse_frontier_solid_lane_04',
          'eclipse_frontier_solid_lane_05',
          'eclipse_frontier_solid_lane_06'
        ],
        enemyWeights: [{ enemyId: 'voidMote', weight: 1 }],
        population: 7,
        maxPopulation: 9,
        respawnSeconds: 6,
        leash: 460,
        spawnBounds: { minX: 1720, maxX: 2740 }
      },
      {
        sectionSuffix: 'eclipse_gate',
        label: 'Gate Duelists',
        platformIds: [
          'eclipse_frontier_solid_lane_07',
          'eclipse_frontier_solid_lane_08',
          'eclipse_frontier_solid_lane_09'
        ],
        enemyWeights: [{ enemyId: 'eclipseDuelist', weight: 1 }],
        population: 9,
        maxPopulation: 11,
        respawnSeconds: 7,
        leash: 500,
        spawnBounds: { minX: 2920, maxX: 4060 }
      },
      {
        sectionSuffix: 'elite_pocket',
        label: 'Totality Elite Pocket',
        platformIds: [
          'eclipse_frontier_solid_lane_10',
          'eclipse_frontier_solid_lane_11',
          'eclipse_frontier_solid_lane_12'
        ],
        enemyWeights: [
          { enemyId: 'eclipseDuelist', weight: 5 },
          { enemyId: 'crackedMimic', weight: 2 },
          { enemyId: 'voidMote', weight: 2 },
          { enemyId: 'lumenSentinel', weight: 1 }
        ],
        population: 10,
        maxPopulation: 12,
        respawnSeconds: 9,
        leash: 540,
        spawnBounds: { minX: 4290, maxX: 5290 }
      }
    ]),
    endlessRift: Object.freeze([
      {
        sectionSuffix: 'southwest_rift_quadrant',
        label: 'Southwest Rift',
        platformIds: ['endlessRift_sw_outer_low', 'endlessRift_sw_inner_low', 'endlessRift_sw_mid'],
        enemyWeights: [{ enemyId: 'voidMote', weight: 4 }, { enemyId: 'lumenSentinel', weight: 3 }, { enemyId: 'indexScribe', weight: 2 }],
        population: 9,
        maxPopulation: 9,
        respawnSeconds: 5,
        leash: 520,
        spawnBounds: { minX: 940, maxX: 2200 }
      },
      {
        sectionSuffix: 'northwest_rift_quadrant',
        label: 'Northwest Rift',
        platformIds: ['endlessRift_nw_outer_high', 'endlessRift_nw_inner_high', 'endlessRift_nw_peak'],
        enemyWeights: [{ enemyId: 'voidMote', weight: 4 }, { enemyId: 'lumenSentinel', weight: 3 }, { enemyId: 'riftAberration', weight: 2 }],
        population: 9,
        maxPopulation: 9,
        respawnSeconds: 5,
        leash: 480,
        spawnBounds: { minX: 940, maxX: 2200 }
      },
      {
        sectionSuffix: 'northeast_rift_quadrant',
        label: 'Northeast Rift',
        platformIds: ['endlessRift_ne_inner_high', 'endlessRift_ne_outer_high', 'endlessRift_ne_peak'],
        enemyWeights: [{ enemyId: 'eclipseDuelist', weight: 4 }, { enemyId: 'indexScribe', weight: 3 }, { enemyId: 'riftAberration', weight: 2 }],
        population: 9,
        maxPopulation: 9,
        respawnSeconds: 5,
        leash: 480,
        spawnBounds: { minX: 3000, maxX: 4240 }
      },
      {
        sectionSuffix: 'southeast_rift_quadrant',
        label: 'Southeast Rift',
        platformIds: ['endlessRift_se_inner_low', 'endlessRift_se_outer_low', 'endlessRift_se_mid'],
        enemyWeights: [{ enemyId: 'riftAberration', weight: 4 }, { enemyId: 'eclipseDuelist', weight: 3 }, { enemyId: 'voidMote', weight: 2 }, { enemyId: 'crackedMimic', weight: 1 }],
        population: 9,
        maxPopulation: 9,
        respawnSeconds: 5,
        leash: 520,
        spawnBounds: { minX: 3000, maxX: 4240 }
      }
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
      const rawSpawnBounds = source.spawnBounds && typeof source.spawnBounds === 'object' ? source.spawnBounds : EMPTY_OBJECT;
      const spawnMinX = Number(rawSpawnBounds.minX);
      const spawnMaxX = Number(rawSpawnBounds.maxX);
      const spawnBounds = Number.isFinite(spawnMinX) && Number.isFinite(spawnMaxX) && spawnMaxX >= spawnMinX
        ? Object.freeze({ minX: spawnMinX, maxX: spawnMaxX })
        : null;
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
        spawnBounds,
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
