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
  const FIELD_REGROUP_PLATFORMS = Object.freeze({
    greenrootMeadow: 'greenroot_meadow_solid_lane_05',
    thornpathThicket: 'thornpath_fork_beacon_perch',
    rustcoilRuins: 'rustcoil_warden_starcoil_service_bridge',
    cinderHollow: 'cinder_hollow_solid_lane_11',
    banditRidgeCamp: 'bandit_ridge_camp_regroup_plateau',
    orebackQuarry: 'oreback_quarry_solid_lane_10',
    ashglassPass: 'ashglass_pass_hop_01',
    frostfenOutskirts: 'frostfen_shelf_shelter_pocket',
    glacierSpine: 'glacier_spine_solid_lane_08',
    stormbreakCliffs: 'stormbreak_cliffs_solid_lane_10',
    astralArchive: 'astral_archive_island_02',
    eclipseFrontier: 'eclipse_frontier_island_02',
    endlessRift: 'endless_rift_hop_01'
  });

  function laneCircuit(prefix, main, optional) {
    const ids = (numbers) => Object.freeze(numbers.map((number) => `${prefix}_solid_lane_${String(number).padStart(2, '0')}`));
    return Object.freeze({ mainPlatformIds: ids(main), optionalPlatformIds: ids(optional) });
  }

  // Ordinary circuits use adjacent lower/middle hunting pockets and return via
  // existing ramps, lifts and ground lanes. Upper perches and the distant final
  // encounter form deliberate excursions; a full-map sweep remains useful for
  // exploration and validation, but is not the only repeatable hunting route.
  const FIELD_TRAINING_CIRCUITS = Object.freeze({
    greenrootMeadow: laneCircuit('greenroot_meadow', [1, 3, 2], [2, 4, 6]),
    thornpathThicket: Object.freeze({
      mainPlatformIds: Object.freeze(['thornpath_rootfall_lane', 'thornpath_rootfall_relay_shelf', 'thornpath_relay_mid_deck', 'thornpath_relay_lower_walk']),
      optionalPlatformIds: Object.freeze(['thornpath_relay_lower_walk', 'thornpath_relay_high_bough', 'thornpath_relay_shard_perch', 'thornpath_fork_observatory_branch', 'thornpath_fork_ridge_branch', 'thornpath_fork_lower_lane'])
    }),
    rustcoilRuins: Object.freeze({
      mainPlatformIds: Object.freeze(['rustcoil_yard_ratchet_lane', 'rustcoil_yard_service_gantry', 'rustcoil_switchworks_west_catwalk', 'rustcoil_switchworks_return_deck', 'rustcoil_switchworks_conveyor']),
      optionalPlatformIds: Object.freeze(['rustcoil_switchworks_conveyor', 'rustcoil_warden_return_belt', 'rustcoil_warden_gear_ring', 'rustcoil_warden_relay_dais', 'rustcoil_warden_starcoil_perch'])
    }),
    cinderHollow: laneCircuit('cinder_hollow', [1, 2, 5, 4], [4, 5, 6, 9, 8, 7]),
    banditRidgeCamp: Object.freeze({
      mainPlatformIds: Object.freeze(['bandit_ridge_camp_lower_barricade_lane', 'bandit_ridge_camp_lower_flank', 'bandit_ridge_camp_thrower_perch_west', 'bandit_ridge_camp_thrower_deck']),
      optionalPlatformIds: Object.freeze(['bandit_ridge_camp_thrower_deck', 'bandit_ridge_camp_thrower_perch_east', 'bandit_ridge_camp_rope_bridge', 'bandit_ridge_camp_bridge_approach', 'bandit_ridge_camp_bridge_return_lane'])
    }),
    orebackQuarry: laneCircuit('oreback_quarry', [1, 2, 5, 4], [4, 5, 6, 9, 8, 7]),
    ashglassPass: laneCircuit('ashglass_pass', [1, 2, 5, 4], [4, 5, 6, 10, 9, 8, 7]),
    frostfenOutskirts: Object.freeze({
      mainPlatformIds: Object.freeze(['frostfen_marsh_runway', 'frostfen_marsh_windbreak', 'frostfen_rimeglass_shelf', 'frostfen_shelf_lower_run']),
      optionalPlatformIds: Object.freeze(['frostfen_shelf_lower_run', 'frostfen_shelf_upper_drift', 'frostfen_oracle_grove_shelf', 'frostfen_oracle_bloom_perch', 'frostfen_oracle_exit_shelf', 'frostfen_oracle_recovery_run'])
    }),
    glacierSpine: laneCircuit('glacier_spine', [1, 2, 6, 5], [5, 6, 7, 11, 10, 9]),
    stormbreakCliffs: Object.freeze({
      mainPlatformIds: laneCircuit('stormbreak_cliffs', [1, 2, 5, 4], []).mainPlatformIds,
      optionalPlatformIds: Object.freeze(['stormbreak_cliffs_solid_lane_04', 'stormbreak_cliffs_solid_lane_05', 'stormbreak_cliffs_solid_lane_06', 'stormbreak_cliffs_hop_01', 'stormbreak_cliffs_solid_lane_09', 'stormbreak_cliffs_solid_lane_08', 'stormbreak_cliffs_solid_lane_07'])
    }),
    astralArchive: laneCircuit('astral_archive', [1, 2, 5, 4], [4, 5, 6, 9, 8, 7]),
    eclipseFrontier: laneCircuit('eclipse_frontier', [1, 2, 5, 4], [4, 5, 6, 9, 8, 7]),
    // Keep all four quadrant territories in the ordinary Rift rotation, so its
    // anti-camping and completed-cycle mechanics retain their intended meaning.
    endlessRift: laneCircuit('endless_rift', [1, 2, 5, 6, 9, 8, 11, 10], [8, 9, 12])
  });

  // Section identities stay stable; the roster expresses the actual encounter
  // rather than copying every biome enemy into every pocket.
  function encounter(sectionSuffix, label, roster, population, options) {
    return Object.assign({ sectionSuffix, label, population, respawnSeconds: 5, leash: 560,
      enemyWeights: roster.map(([enemyId, weight]) => ({ enemyId, weight })) }, options || {});
  }

  const FEATURED_SPAWN_GROUP_PROFILES = Object.freeze({
    greenrootMeadow: Object.freeze([
      { sectionSuffix: 'arrival_shelf', label: 'Arrival Shelf', enemyWeights: [{ enemyId: 'glassback', weight: 6 }, { enemyId: 'faultSkitter', weight: 3 }], population: 2, respawnSeconds: 6, leash: 380 },
      { sectionSuffix: 'glass_basin', label: 'Glass Basin', enemyWeights: [{ enemyId: 'glassback', weight: 4 }, { enemyId: 'faultSkitter', weight: 3 }, { enemyId: 'riftLantern', weight: 1 }], population: 4, respawnSeconds: 6, leash: 460 },
      { sectionSuffix: 'fractured_bridge', label: 'Fractured Bridge', enemyWeights: [{ enemyId: 'glassback', weight: 3 }, { enemyId: 'riftLantern', weight: 3 }, { enemyId: 'faultSkitter', weight: 2 }], population: 3, respawnSeconds: 7, leash: 520 },
      { sectionSuffix: 'beacon_approach', label: 'Beacon Approach', enemyWeights: [{ enemyId: 'glassback', weight: 3 }, { enemyId: 'riftLantern', weight: 4 }], population: 2, respawnSeconds: 7, leash: 420 }
    ]),
    thornpathThicket: Object.freeze([
      encounter('meadow_return', 'Rootfall Ground Packs', [['dewSlime', 5], ['vineSnapper', 3], ['mossback', 2]], 8),
      encounter('fracture_canopy', 'Canopy Ambush', [['thornSprout', 4], ['vineSnapper', 5], ['dewSlime', 1]], 10),
      encounter('observatory_fork', 'Briar Guard Branch', [['briarStag', 5], ['mossback', 3], ['vineSnapper', 2]], 8, { respawnSeconds: 6 })
    ]),
    rustcoilRuins: Object.freeze([
      encounter('surveyor_yard', 'Ratchet Patrol', [['rustRatchet', 7], ['clockbug', 3]], 10),
      encounter('coil_switchworks', 'Coil Crossfire', [['coilSentry', 4], ['rustRatchet', 4], ['clockbug', 2]], 10),
      encounter('warden_gearwell', 'Warden Gearwell', [['scrapWarden', 5], ['clockbug', 3], ['coilSentry', 2]], 8, { respawnSeconds: 6 })
    ]),
    cinderHollow: Object.freeze([
      encounter('ash_floor_loop', 'Ash Floor Ground Packs', [['ashCrawler', 6], ['lavaTick', 4]], 9),
      encounter('vent_shortcut', 'Lava Tick Vent Run', [['lavaTick', 7], ['cinderSpitter', 3]], 8),
      encounter('flyer_turns', 'Ember Crossfire', [['cinderSpitter', 5], ['lavaTick', 3], ['emberWisp', 2]], 7, { respawnSeconds: 6 })
    ]),
    banditRidgeCamp: Object.freeze([
      {
        sectionSuffix: 'lower_cutter_lane',
        label: 'Lower Cutters',
        platformIds: ['bandit_ridge_camp_lower_barricade_lane', 'bandit_ridge_camp_lower_flank'],
        enemyWeights: [{ enemyId: 'banditCutter', weight: 8 }, { enemyId: 'briarStag', weight: 2 }],
        population: 10,
        maxPopulation: 14,
        partyBonusPerMember: 2,
        respawnSeconds: 4,
        leash: 520
      },
      {
        sectionSuffix: 'middle_thrower_camp',
        label: 'Thrower Camp',
        platformIds: ['bandit_ridge_camp_thrower_deck', 'bandit_ridge_camp_thrower_perch_west', 'bandit_ridge_camp_thrower_perch_east'],
        enemyWeights: [{ enemyId: 'banditThrower', weight: 7 }, { enemyId: 'banditCutter', weight: 3 }],
        population: 10,
        maxPopulation: 14,
        partyBonusPerMember: 2,
        respawnSeconds: 5,
        leash: 480
      },
      {
        sectionSuffix: 'high_rope_bridge',
        label: 'Rope Bridge',
        platformIds: ['bandit_ridge_camp_bridge_approach', 'bandit_ridge_camp_rope_bridge', 'bandit_ridge_camp_bridge_return_lane'],
        enemyWeights: [{ enemyId: 'banditThrower', weight: 5 }, { enemyId: 'banditCutter', weight: 2 }, { enemyId: 'briarStag', weight: 3 }],
        population: 10,
        maxPopulation: 14,
        partyBonusPerMember: 2,
        respawnSeconds: 5,
        leash: 420
      }
    ]),
    orebackQuarry: Object.freeze([
      { sectionSuffix: 'ore_cart_lane', label: 'Ore Cart Beetles', enemyWeights: [{ enemyId: 'orebackBeetle', weight: 8 }, { enemyId: 'scrapWarden', weight: 2 }], population: 8, respawnSeconds: 5, leash: 500 },
      { sectionSuffix: 'scaffold_sentries', label: 'Scaffold Sentries', enemyWeights: [{ enemyId: 'coilSentry', weight: 7 }, { enemyId: 'orebackBeetle', weight: 3 }], population: 7, respawnSeconds: 6, leash: 380 },
      { sectionSuffix: 'mushroom_pocket', label: 'Glowcap Support Pocket', enemyWeights: [{ enemyId: 'glowcapHealer', weight: 2 }, { enemyId: 'orebackBeetle', weight: 8 }], enemyMaxAlive: { glowcapHealer: 1 }, population: 7, respawnSeconds: 6, leash: 420 },
      { sectionSuffix: 'mine_event_pocket', label: 'Mimic Mine', enemyWeights: [{ enemyId: 'orebackBeetle', weight: 5 }, { enemyId: 'scrapWarden', weight: 3 }, { enemyId: 'crackedMimic', weight: 1 }], population: 4, respawnSeconds: 8, leash: 360 }
    ]),
    ashglassPass: Object.freeze([
      encounter('ashglass_bridge', 'Basalt Bridge Patrol', [['ashCrawler', 5], ['lavaTick', 5]], 9),
      encounter('vent_side_pocket', 'Vent Crossfire', [['cinderSpitter', 7], ['lavaTick', 3]], 8),
      encounter('glass_shelf', 'Glass Shelf Airspace', [['emberWisp', 7], ['orebackBeetle', 3]], 8),
      encounter('elite_storm_pocket', 'Obsidian Elite Pocket', [['orebackBeetle', 6], ['cinderSpitter', 3], ['crackedMimic', 1]], 5, { respawnSeconds: 7, leash: 420 })
    ]),
    frostfenOutskirts: Object.freeze([
      encounter('frozen_marsh', 'Frozen Marsh Scouts', [['frostlingScout', 6], ['shardling', 4]], 11),
      encounter('rimeglass_shelf', 'Rimeglass Airspace', [['snowglareWisp', 5], ['rimebackBrute', 3], ['shardling', 2]], 11),
      encounter('oracle_grove', 'Oracle Grove Guards', [['icebloomOracle', 2], ['frostlingScout', 5], ['rimebackBrute', 3]], 9,
        { enemyMaxAlive: { icebloomOracle: 1 }, respawnSeconds: 6, leash: 440 })
    ]),
    glacierSpine: Object.freeze([
      encounter('entry', 'Lower Ridge Scouts', [['frostlingScout', 5], ['shardling', 3], ['rimebackBrute', 2]], 11,
        { partyScaling: 'section-count', partyBonusPerMember: 1, maxPopulation: 13 }),
      encounter('deep_route', 'Glacier Sentry Circuit', [['rimebackBrute', 2], ['snowglareWisp', 5], ['glacierSentinel', 2], ['frostlingScout', 1]], 12,
        { partyScaling: 'section-count', partyBonusPerMember: 2, maxPopulation: 16 }),
      encounter('exit', 'High Ridge Elite Guard', [['snowglareWisp', 4], ['rimebackBrute', 3], ['glacierSentinel', 2], ['icebloomOracle', 1], ['crackedMimic', 1]], 9,
        { partyScaling: 'section-count', partyBonusPerMember: 1, maxPopulation: 11, enemyMaxAlive: { icebloomOracle: 1 }, respawnSeconds: 6 })
    ]),
    stormbreakCliffs: Object.freeze([
      { sectionSuffix: 'low_ram_lane', label: 'Thunder Ram Lane', enemyWeights: [{ enemyId: 'thunderRam', weight: 8 }, { enemyId: 'cloudcallAcolyte', weight: 2 }], population: 9, respawnSeconds: 5, leash: 500 },
      { sectionSuffix: 'mid_archer_bridge', label: 'Archer Bridge', enemyWeights: [{ enemyId: 'stormboundArcher', weight: 7 }, { enemyId: 'cloudcallAcolyte', weight: 3 }], population: 8, respawnSeconds: 5, leash: 440 },
      { sectionSuffix: 'high_harrier_airspace', label: 'Harrier Airspace', enemyWeights: [{ enemyId: 'galeHarrier', weight: 8 }, { enemyId: 'stormboundArcher', weight: 2 }], population: 9, respawnSeconds: 5, leash: 540, actorTraversal: { mode: 'air', allowLadders: false, allowRamps: true, stayInTerritory: false } },
      { sectionSuffix: 'lightning_rod_objective', label: 'Lightning Rod', enemyWeights: [{ enemyId: 'cloudcallAcolyte', weight: 6 }, { enemyId: 'thunderRam', weight: 3 }, { enemyId: 'crackedMimic', weight: 1 }], population: 6, respawnSeconds: 7, leash: 380 }
    ]),
    astralArchive: Object.freeze([
      encounter('entry', 'Reading Room Guards', [['lumenSentinel', 6], ['indexScribe', 4]], 11),
      encounter('training_loop', 'Index Shelf Crossfire', [['indexScribe', 7], ['voidMote', 3]], 13),
      encounter('exit', 'Sealed Archive Pocket', [['voidMote', 5], ['lumenSentinel', 4], ['crackedMimic', 1]], 10, { respawnSeconds: 6 })
    ]),
    eclipseFrontier: Object.freeze([
      encounter('solar_outpost', 'Solar Sentinel Patrol', [['lumenSentinel', 6], ['eclipseDuelist', 4]], 9),
      encounter('lunar_outpost', 'Lunar Airspace', [['voidMote', 7], ['indexScribe', 3]], 8),
      encounter('eclipse_gate', 'Gate Crossfire', [['indexScribe', 5], ['eclipseDuelist', 5]], 9),
      encounter('elite_pocket', 'Eclipse Elite Guard', [['eclipseDuelist', 5], ['lumenSentinel', 4], ['crackedMimic', 1]], 8, { respawnSeconds: 6 })
    ]),
    endlessRift: Object.freeze([
      encounter('northwest_rift_quadrant', 'Western Sentinel Circuit', [['lumenSentinel', 6], ['indexScribe', 4]], 8),
      encounter('northeast_rift_quadrant', 'Upper Rift Airspace', [['voidMote', 7], ['eclipseDuelist', 3]], 8),
      encounter('southeast_rift_quadrant', 'Eastern Duelist Circuit', [['eclipseDuelist', 6], ['indexScribe', 4]], 8),
      encounter('southwest_rift_quadrant', 'Lower Rift Crossfire', [['indexScribe', 6], ['voidMote', 4]], 8),
      encounter('rift_core_regroup', 'Optional Rift Surge', [['riftAberration', 7], ['crackedMimic', 1], ['lumenSentinel', 2]], 4,
        { platformIds: ['endless_rift_solid_lane_12'], respawnSeconds: 7, leash: 380 })
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
      const platformIds = Array.from(new Set(declaredPlatformIds.length ? declaredPlatformIds : platformIndices.map((platformIndex) => getPublishedPlatformId(map, platformIndex))))
        .filter((platformId) => platformId !== FIELD_REGROUP_PLATFORMS[map.id]);
      const enemyWeights = normalizeSpawnEnemyWeights(source.enemyWeights || source.enemies, map.enemies);
      if (!platformIds.length || !enemyWeights.length) return null;
      return Object.freeze({
        id,
        label: String(source.label || section && section.label || `Spawn Group ${index + 1}`),
        sectionId,
        platformIds: Object.freeze(platformIds),
        enemyWeights,
        enemyMaxAlive: Object.freeze(Object.assign({}, ...enemyWeights
          .filter((entry) => ['glowcapHealer', 'icebloomOracle', 'cloudcallAcolyte'].includes(entry.enemyId))
          .map((entry) => ({ [entry.enemyId]: 1 })), source.enemyMaxAlive || {})),
        population: Math.max(1, Math.floor(Number(source.population || 0)) || 1),
        respawnSeconds: Math.max(1, Math.min(60, Number(source.respawnSeconds || map.waveDelay || 5) || 5)),
        leash: Math.max(90, Math.min(2400, Number(source.leash || 480) || 480)),
        partyScaling: String(source.partyScaling || map.partyScaling || map.designIntent && map.designIntent.partyScaling || 'none'),
        maxPopulation: Math.max(1, Math.floor(Number(source.maxPopulation || 0)) || Math.ceil(Math.max(1, Number(source.population || 1)) * 1.5)),
        partyBonusPerMember: Math.max(0, Math.min(4, Number(source.partyBonusPerMember == null ? 1 : source.partyBonusPerMember) || 0)),
        actorTraversal: normalizeActorTraversal(source.actorTraversal || {
          allowRamps: true,
          allowLadders: ['banditRidgeCamp', 'rustcoilRuins', 'astralArchive', 'eclipseFrontier', 'endlessRift'].includes(map.id),
          stayInTerritory: !!(map.isDungeon || map.bossRoom)
        })
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
      const spawnPoints = Object.freeze(attachSpawnSectionsToPoints(map, spawnSections).map((point) =>
        map.id === 'greenrootMeadow' && point.sectionId === 'greenrootMeadow_glass_basin'
          ? Object.freeze(Object.assign({}, point, { weight: 1 })) : point));
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
        asset: mapAssets[map.id] || map.asset || '',
        backgroundMode: 'panorama',
        environment: mapEnvironmentProfiles[map.id] || map.environment || mapEnvironmentProfiles.greenrootMeadow,
        spawnPoints,
        spawnGroups,
        trainingXpMultiplier: map.id === 'cinderHollow' ? 0.93 : map.trainingXpMultiplier,
        trainingRoute: FIELD_REGROUP_PLATFORMS[map.id] ? Object.freeze({
          regroupPlatformId: FIELD_REGROUP_PLATFORMS[map.id],
          mainRegroupPlatformId: getPublishedPlatformId(map, 0),
          mainRegroupX: 110,
          mainPlatformIds: FIELD_TRAINING_CIRCUITS[map.id].mainPlatformIds,
          optionalPlatformIds: FIELD_TRAINING_CIRCUITS[map.id].optionalPlatformIds,
          combatGroupIds: Object.freeze(spawnGroups.map((group) => group.id)),
          optionalGroupId: spawnGroups.length ? spawnGroups[spawnGroups.length - 1].id : ''
        }) : null,
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
