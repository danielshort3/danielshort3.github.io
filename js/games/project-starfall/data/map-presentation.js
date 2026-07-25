(function initProjectStarfallDataMapPresentation(global) {
  'use strict';

  function getDefaultPlatformRight(platform) {
    if (Array.isArray(platform)) return Number(platform[0] || 0) + Number(platform[2] || 0);
    return Number(platform && platform.x || 0) + Number(platform && platform.w || 0);
  }

  function defaultGetAuthoredMapWidth(map) {
    const source = map || {};
    const platformWidth = (source.platforms || []).reduce((width, platform) => Math.max(width, getDefaultPlatformRight(platform)), 0);
    const pointWidth = []
      .concat(source.spawnPoints || [])
      .concat(source.stations || [])
      .concat(source.questNpcs || [])
      .reduce((width, point) => Math.max(width, Number(point && point.x || 0) + 240), 0);
    if (source.shopInterior) {
      const compactWidth = Number(source.compactWorldWidth || 1280);
      return Math.max(compactWidth, platformWidth, pointWidth);
    }
    return Math.max(3600, platformWidth, pointWidth);
  }

  function createMapPresentationData(options) {
    const settings = options || {};
    const getAuthoredMapWidth = typeof settings.getAuthoredMapWidth === 'function'
      ? settings.getAuthoredMapWidth
      : defaultGetAuthoredMapWidth;

    const MAP_LAYOUT_ROLES = Object.freeze({
      town: Object.freeze({ id: 'town', label: 'Town Hub', marker: 'T' }),
      starterField: Object.freeze({ id: 'starterField', label: 'Starter Field', marker: '1' }),
      trainingField: Object.freeze({ id: 'trainingField', label: 'Training Field', marker: 'F' }),
      deepField: Object.freeze({ id: 'deepField', label: 'Deep Field', marker: 'D' }),
      dungeon: Object.freeze({ id: 'dungeon', label: 'Dungeon', marker: 'DG' }),
      bossArena: Object.freeze({ id: 'bossArena', label: 'Boss Echo', marker: 'B' }),
      endlessField: Object.freeze({ id: 'endlessField', label: 'Endless Field', marker: 'R' })
    });

    const MAP_LAYOUT_ROLE_LABELS = Object.freeze(Object.keys(MAP_LAYOUT_ROLES).reduce((labels, roleId) => {
      labels[roleId] = MAP_LAYOUT_ROLES[roleId].label;
      return labels;
    }, {}));

    function normalizeMapLayoutRole(roleId, fallback) {
      const id = String(roleId || '').trim();
      if (Object.prototype.hasOwnProperty.call(MAP_LAYOUT_ROLES, id)) return id;
      return Object.prototype.hasOwnProperty.call(MAP_LAYOUT_ROLES, fallback) ? fallback : 'trainingField';
    }

    function getMapLayoutRoleFallback(map) {
      if (!map) return 'trainingField';
      if (map.safeZone) return 'town';
      if (map.endlessScaling) return 'endlessField';
      if (map.bossRoom) return 'bossArena';
      if (map.isDungeon) return 'dungeon';
      return 'trainingField';
    }

      const MAP_LAYOUT_BLUEPRINTS = Object.freeze({
      starfallCrossing: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Crossing Plaza', landmark: 'central meteor plaza', portalPattern: 'serviceHub' }),
      rustcoilOutpost: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Outpost Main Street', landmark: 'gear tower', portalPattern: 'serviceHub' }),
      cinderRefuge: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Refuge Walk', landmark: 'furnace shelter', portalPattern: 'serviceHub' }),
      frostfenCamp: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Camp Line', landmark: 'ice signal post', portalPattern: 'serviceHub' }),
      stormbreakHaven: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Haven Span', landmark: 'storm mast', portalPattern: 'serviceHub' }),
      astralObservatory: Object.freeze({ role: 'town', routeStage: 'Town Hub', roadName: 'Observatory Ring', landmark: 'star lens', portalPattern: 'serviceHub' }),
      greenrootMeadow: Object.freeze({ role: 'starterField', routeStage: 'Outskirts', roadName: 'Greenroot Road I', landmark: 'pond bridges', portalPattern: 'leftReturnRightAdvance' }),
      thornpathThicket: Object.freeze({ role: 'trainingField', routeStage: 'Training Loop', roadName: 'Greenroot Road II', landmark: 'thorn canopy', portalPattern: 'leftReturnRightAdvance' }),
      banditRidgeCamp: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Greenroot Ridge', landmark: 'bandit lookout', portalPattern: 'leftReturnDungeon' }),
      banditAnimationLab: Object.freeze({ role: 'trainingField', routeStage: 'Admin Lab', roadName: 'Bandit Animation Lab', landmark: 'comparison stands', portalPattern: 'none' }),
      brambleDepths: Object.freeze({ role: 'dungeon', routeStage: 'Dungeon', roadName: 'Bramble Depths', landmark: 'root gate', portalPattern: 'returnPortal' }),
      rustcoilRuins: Object.freeze({ role: 'trainingField', routeStage: 'Training Loop', roadName: 'Rustcoil Road I', landmark: 'broken gearworks', portalPattern: 'leftReturn' }),
      orebackQuarry: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Rustcoil Road II', landmark: 'quarry lift', portalPattern: 'leftReturnDungeonAdvance' }),
      gearworksVault: Object.freeze({ role: 'dungeon', routeStage: 'Dungeon', roadName: 'Gearworks Vault', landmark: 'vault lock', portalPattern: 'returnPortal' }),
      cinderHollow: Object.freeze({ role: 'trainingField', routeStage: 'Training Loop', roadName: 'Cinder Road I', landmark: 'ember vents', portalPattern: 'leftReturnDungeon' }),
      emberjawLair: Object.freeze({ role: 'dungeon', routeStage: 'Dungeon', roadName: 'Emberjaw Lair', landmark: 'magma gate', portalPattern: 'returnPortal' }),
      ashglassPass: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Ashglass Pass', landmark: 'glass bridge', portalPattern: 'leftReturnRightAdvance' }),
      frostfenOutskirts: Object.freeze({ role: 'trainingField', routeStage: 'Training Loop', roadName: 'Frostfen Road I', landmark: 'frozen marsh', portalPattern: 'leftReturnRightAdvance' }),
      glacierSpine: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Frostfen Road II', landmark: 'glacier spine', portalPattern: 'leftReturnDungeonAdvance' }),
      rimewardenSanctum: Object.freeze({ role: 'dungeon', routeStage: 'Dungeon', roadName: 'Rimewarden Sanctum', landmark: 'frost vault', portalPattern: 'returnPortal' }),
      stormbreakCliffs: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Stormbreak Cliffs', landmark: 'wind rods', portalPattern: 'leftReturn' }),
      astralArchive: Object.freeze({ role: 'trainingField', routeStage: 'Training Loop', roadName: 'Astral Road I', landmark: 'living stacks', portalPattern: 'leftReturnRightAdvance' }),
      eclipseFrontier: Object.freeze({ role: 'deepField', routeStage: 'Deep Route', roadName: 'Astral Road II', landmark: 'eclipse gate', portalPattern: 'leftReturnRightAdvance' }),
      endlessRift: Object.freeze({ role: 'endlessField', routeStage: 'Endless Route', roadName: 'Endless Rift', landmark: 'rift lens', portalPattern: 'leftReturn' }),
      bramblekingCourt: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Brambleking Court', landmark: 'crowned root', portalPattern: 'returnPortal' }),
      titanFoundry: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Titan Foundry', landmark: 'titan forge', portalPattern: 'returnPortal' }),
      deepcoreCore: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Deepcore Core', landmark: 'ore core', portalPattern: 'returnPortal' }),
      emberjawFurnace: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Emberjaw Furnace', landmark: 'furnace maw', portalPattern: 'returnPortal' }),
      rimewardenVault: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Rimewarden Vault', landmark: 'ice seal', portalPattern: 'returnPortal' }),
      stormbreakAerie: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Stormbreak Aerie', landmark: 'aerie mast', portalPattern: 'returnPortal' }),
      astralStacks: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Astral Stacks', landmark: 'mirror shelves', portalPattern: 'returnPortal' }),
        eclipseThrone: Object.freeze({ role: 'bossArena', routeStage: 'Boss Echo', roadName: 'Eclipse Throne', landmark: 'totality dais', portalPattern: 'returnPortal' })
      });

    function freezeSceneEntries(entries) {
      return Object.freeze((entries || []).map((entry) => Object.freeze(Object.assign({}, entry))));
    }

    function freezeSceneObject(config) {
      const source = config || {};
      return Object.freeze({
        rearStructures: freezeSceneEntries(source.rearStructures),
        stationFacades: freezeSceneEntries(source.stationFacades),
        streetProps: freezeSceneEntries(source.streetProps),
        foregroundTrim: freezeSceneEntries(source.foregroundTrim)
      });
    }

    function freezeCompositionObject(config) {
      const source = config || {};
      return Object.freeze({
        routeSections: freezeSceneEntries(source.routeSections),
        portalRoles: Object.freeze(Object.assign({}, source.portalRoles || {})),
        landmarkBands: freezeSceneEntries(source.landmarkBands),
        spawnZoneLabels: freezeSceneEntries(source.spawnZoneLabels)
      });
    }

    function createTownScene(config) {
      return freezeSceneObject(config);
    }

    function createFieldComposition(config) {
      return freezeCompositionObject(config);
    }

    function createDefaultTownScene(map) {
      const id = map && map.id || 'town';
      const cell = id === 'rustcoilOutpost' ? 'rustcoilWorkshop'
        : id === 'cinderRefuge' ? 'cinderForge'
          : id === 'frostfenCamp' ? 'frostfenLodge'
            : id === 'stormbreakHaven' ? 'stormbreakGate'
              : id === 'astralObservatory' ? 'astralObservatory'
                : 'starfallGuildHall';
      return createTownScene({
        rearStructures: [
          { cell, x: 140, w: 560, h: 286, footOffset: 4, label: map && map.landmark || map && map.name || 'Town landmark' },
          { cell: 'marketAwning', x: 760, w: 360, h: 164, footOffset: 2, label: 'Market row' },
          { cell: 'lanternArch', x: 1860, w: 300, h: 202, footOffset: 2, label: 'Town arch' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -58, w: 220, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -62, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'lanternArch', dx: -48, w: 190, h: 150, footOffset: 3 }
        ],
        streetProps: [
          { kind: 'sign', x: 360, w: 42, h: 50, footOffset: 2 },
          { kind: 'crate', x: 1120, w: 46, h: 38, footOffset: 1 },
          { kind: 'glow', x: 1780, w: 42, h: 34, footOffset: 1 }
        ],
        foregroundTrim: [
          { kind: 'grass', startX: 150, endX: 2800, every: 420, w: 32, h: 18, footOffset: 0 }
        ]
      });
    }

    function createDefaultFieldComposition(map, blueprint) {
      const width = getAuthoredMapWidth(map || {});
      const safeWidth = Math.max(3600, width || 6200);
      return createFieldComposition({
        routeSections: [
          { label: 'Entry', x: 0, w: Math.round(safeWidth * 0.28), tier: 'return' },
          { label: blueprint && blueprint.routeStage || 'Route', x: Math.round(safeWidth * 0.28), w: Math.round(safeWidth * 0.44), tier: 'training' },
          { label: 'Exit', x: Math.round(safeWidth * 0.72), w: Math.round(safeWidth * 0.28), tier: 'advance' }
        ],
        portalRoles: {},
        landmarkBands: [
          { kind: 'tall', x: Math.round(safeWidth * 0.18), w: Math.round(safeWidth * 0.26), label: blueprint && blueprint.landmark || map && map.name || 'Route landmark' },
          { kind: 'sign', x: Math.round(safeWidth * 0.58), w: Math.round(safeWidth * 0.18), label: 'Route marker' }
        ],
        spawnZoneLabels: [
          { label: 'Lower lane', platformTier: 'low' },
          { label: 'Mid lane', platformTier: 'mid' },
          { label: 'High lane', platformTier: 'high' }
        ]
      });
    }

    function createArenaFieldComposition(config) {
      const settings = config || {};
      const labels = (settings.sections || []).map((section) =>
        typeof section === 'string' ? { label: section } : section
      ).filter((section) => section && section.label);
      const safeLabels = labels.length ? labels : [{ label: 'Entry' }, { label: 'Arena' }, { label: 'Exit' }];
      const width = Math.max(3600, Number(settings.width || 4600));
      const sectionWidth = width / safeLabels.length;
      const landmarkKinds = settings.landmarkKinds || ['crystal', 'tall', 'glow', 'rock'];
      return createFieldComposition({
        routeSections: safeLabels.map((section, index) => ({
          label: section.label,
          x: Math.round(section.x != null ? section.x : sectionWidth * index),
          w: Math.round(section.w != null ? section.w : sectionWidth),
          tier: section.tier || (index === 0 ? 'entry' : index === safeLabels.length - 1 ? 'boss' : 'mechanic')
        })),
        portalRoles: settings.portalRoles || {},
        landmarkBands: safeLabels.map((section, index) => ({
          kind: section.kind || landmarkKinds[index % landmarkKinds.length],
          x: Math.round(section.landmarkX != null ? section.landmarkX : sectionWidth * index + sectionWidth * 0.22),
          w: Math.round(section.landmarkW != null ? section.landmarkW : Math.max(360, sectionWidth * 0.48)),
          label: section.landmarkLabel || section.label
        })),
        spawnZoneLabels: safeLabels.map((section, index) => ({
          label: section.spawnLabel || section.label,
          platformTier: section.platformTier || ['low', 'mid', 'high', 'peak'][index % 4]
        }))
      });
    }

    const MAP_TOWN_SCENES = Object.freeze({
      starfallCrossing: createTownScene({
        rearStructures: [
          { cell: 'starfallGuildHall', x: 92, w: 660, h: 318, footOffset: 4, label: 'Adventurer Hall' },
          { cell: 'marketAwning', x: 760, w: 410, h: 176, footOffset: 4, label: 'Market Row' },
          { cell: 'lanternArch', x: 1460, w: 290, h: 206, footOffset: 2, label: 'Class Walk' },
          { cell: 'marketAwning', x: 2100, w: 460, h: 190, footOffset: 4, label: 'Greenroot Gate Market' },
          { cell: 'lanternArch', x: 3180, w: 320, h: 218, footOffset: 2, label: 'Greenroot Gate' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -70, w: 230, h: 118, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -74, w: 242, h: 124, footOffset: 4 },
          { stationId: 'slots', cell: 'marketAwning', dx: -70, w: 226, h: 116, footOffset: 4 },
          { stationId: 'upgrade', cell: 'lanternArch', dx: -52, w: 198, h: 154, footOffset: 3 },
          { stationId: 'class', cell: 'starfallGuildHall', dx: -118, w: 306, h: 180, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'sign', x: 280, w: 42, h: 52, footOffset: 2 },
          { kind: 'flower', x: 610, w: 34, h: 30, footOffset: 0 },
          { kind: 'crate', x: 1208, w: 48, h: 40, footOffset: 1 },
          { kind: 'glow', x: 1768, w: 42, h: 34, footOffset: 0 },
          { kind: 'sign', x: 3000, w: 44, h: 54, footOffset: 2 }
        ],
        foregroundTrim: [
          { kind: 'grass', startX: 160, endX: 3500, every: 360, w: 32, h: 18, footOffset: 0 },
          { kind: 'flower', startX: 420, endX: 3300, every: 620, w: 26, h: 26, footOffset: 0 }
        ]
      }),
      rustcoilOutpost: createTownScene({
        rearStructures: [
          { cell: 'rustcoilWorkshop', x: 120, w: 600, h: 292, footOffset: 4, label: 'Gear Tower' },
          { cell: 'marketAwning', x: 800, w: 390, h: 172, footOffset: 4, label: 'Scrap Market' },
          { cell: 'lanternArch', x: 2160, w: 300, h: 206, footOffset: 2, label: 'Ruins Gate' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -62, w: 224, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -64, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'rustcoilWorkshop', dx: -82, w: 240, h: 150, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'crate', x: 1220, w: 48, h: 40, footOffset: 1 },
          { kind: 'sign', x: 2500, w: 44, h: 54, footOffset: 2 },
          { kind: 'crystal', x: 1780, w: 40, h: 48, footOffset: 0 }
        ],
        foregroundTrim: [{ kind: 'rock', startX: 180, endX: 2820, every: 460, w: 36, h: 22, footOffset: 0 }]
      }),
      cinderRefuge: createTownScene({
        rearStructures: [
          { cell: 'cinderForge', x: 120, w: 600, h: 292, footOffset: 4, label: 'Furnace Shelter' },
          { cell: 'marketAwning', x: 820, w: 390, h: 172, footOffset: 4, label: 'Ash Market' },
          { cell: 'lanternArch', x: 2160, w: 300, h: 206, footOffset: 2, label: 'Hollow Gate' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -62, w: 224, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -64, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'cinderForge', dx: -84, w: 244, h: 154, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'glow', x: 1220, w: 42, h: 34, footOffset: 0 },
          { kind: 'crystal', x: 1760, w: 40, h: 48, footOffset: 0 },
          { kind: 'sign', x: 2510, w: 44, h: 54, footOffset: 2 }
        ],
        foregroundTrim: [{ kind: 'rock', startX: 180, endX: 2820, every: 420, w: 36, h: 22, footOffset: 0 }]
      }),
      frostfenCamp: createTownScene({
        rearStructures: [
          { cell: 'frostfenLodge', x: 120, w: 600, h: 292, footOffset: 4, label: 'Ice Signal Lodge' },
          { cell: 'marketAwning', x: 820, w: 390, h: 172, footOffset: 4, label: 'Supply Tents' },
          { cell: 'lanternArch', x: 2160, w: 300, h: 206, footOffset: 2, label: 'Tundra Gate' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -62, w: 224, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -64, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'frostfenLodge', dx: -86, w: 244, h: 154, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'crystal', x: 1220, w: 40, h: 48, footOffset: 0 },
          { kind: 'glow', x: 1760, w: 42, h: 34, footOffset: 0 },
          { kind: 'sign', x: 2510, w: 44, h: 54, footOffset: 2 }
        ],
        foregroundTrim: [{ kind: 'rock', startX: 180, endX: 2820, every: 460, w: 36, h: 22, footOffset: 0 }]
      }),
      stormbreakHaven: createTownScene({
        rearStructures: [
          { cell: 'stormbreakGate', x: 120, w: 600, h: 292, footOffset: 4, label: 'Storm Mast Gate' },
          { cell: 'marketAwning', x: 820, w: 390, h: 172, footOffset: 4, label: 'Sky Market' },
          { cell: 'lanternArch', x: 2160, w: 300, h: 206, footOffset: 2, label: 'Cliff Span' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -62, w: 224, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -64, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'stormbreakGate', dx: -88, w: 250, h: 158, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'glow', x: 1220, w: 42, h: 34, footOffset: 0 },
          { kind: 'crystal', x: 1760, w: 40, h: 48, footOffset: 0 },
          { kind: 'sign', x: 2510, w: 44, h: 54, footOffset: 2 }
        ],
        foregroundTrim: [{ kind: 'grass', startX: 180, endX: 2820, every: 460, w: 32, h: 18, footOffset: 0 }]
      }),
      astralObservatory: createTownScene({
        rearStructures: [
          { cell: 'astralObservatory', x: 100, w: 640, h: 306, footOffset: 4, label: 'Star Lens' },
          { cell: 'marketAwning', x: 830, w: 390, h: 172, footOffset: 4, label: 'Archive Market' },
          { cell: 'lanternArch', x: 2160, w: 300, h: 206, footOffset: 2, label: 'Archive Gate' }
        ],
        stationFacades: [
          { stationId: 'storage', cell: 'marketAwning', dx: -62, w: 224, h: 116, footOffset: 4 },
          { stationId: 'shop', cell: 'marketAwning', dx: -64, w: 232, h: 120, footOffset: 4 },
          { stationId: 'upgrade', cell: 'astralObservatory', dx: -92, w: 256, h: 164, footOffset: 4 }
        ],
        streetProps: [
          { kind: 'crystal', x: 1210, w: 40, h: 48, footOffset: 0 },
          { kind: 'glow', x: 1760, w: 42, h: 34, footOffset: 0 },
          { kind: 'sign', x: 2510, w: 44, h: 54, footOffset: 2 }
        ],
        foregroundTrim: [{ kind: 'glow', startX: 180, endX: 2820, every: 520, w: 34, h: 22, footOffset: 0 }]
      })
    });

    const MAP_FIELD_COMPOSITIONS = Object.freeze({
      greenrootMeadow: createFieldComposition({
        routeSections: [
          { label: 'Starter Pond Loop', x: 0, w: 1050, tier: 'starter' },
          { label: 'Moss Lane Extension', x: 1050, w: 1050, tier: 'training' },
          { label: 'Canopy Practice', x: 2100, w: 1050, tier: 'mobility' },
          { label: 'Thornpath Gate', x: 3150, w: 1050, tier: 'advance' }
        ],
        portalRoles: {
          greenroot_crossing: 'left town return',
          greenroot_thornpath: 'right route advance'
        },
        landmarkBands: [
          { kind: 'tree', x: 260, w: 680, label: 'Beginner Grove' },
          { kind: 'glow', x: 520, w: 360, label: 'Starter Pond' },
          { kind: 'vine', x: 2240, w: 640, label: 'Canopy Shortcut' },
          { kind: 'sign', x: 3540, w: 360, label: 'Thornpath Sign' }
        ],
        spawnZoneLabels: [
          { label: 'Starter pond', platformTier: 'low' },
          { label: 'Moss lane', platformTier: 'mid' },
          { label: 'Canopy practice', platformTier: 'high' },
          { label: 'Thornpath gate', platformTier: 'advance' }
        ]
      }),
      thornpathThicket: createFieldComposition({
        routeSections: [
          { label: 'Meadow Return', x: 0, w: 1700, tier: 'return' },
          { label: 'Thorn Canopy', x: 1700, w: 3900, tier: 'vertical' },
          { label: 'Deep Fork', x: 5600, w: 2100, tier: 'advance' }
        ],
        portalRoles: {
          thornpath_greenroot: 'left route return',
          thornpath_bandit: 'deep field advance',
          thornpath_rustcoil_outpost: 'regional town branch'
        },
        landmarkBands: [
          { kind: 'vine', x: 860, w: 1200, label: 'Low Vines' },
          { kind: 'tree', x: 2860, w: 1560, label: 'Thorn Canopy' },
          { kind: 'sign', x: 6660, w: 520, label: 'Fork Marker' }
        ],
        spawnZoneLabels: [
          { label: 'Vine snapper lane', platformTier: 'low' },
          { label: 'Thorn sprout shelf', platformTier: 'mid' },
          { label: 'Briar route', platformTier: 'high' }
        ]
      }),
      brambleDepths: createFieldComposition({
        routeSections: [
          { label: 'Ridge Return', x: 0, w: 1200, tier: 'return' },
          { label: 'Root Lanes', x: 1200, w: 2000, tier: 'dungeon' },
          { label: 'Court Gate', x: 3200, w: 1200, tier: 'boss' }
        ],
        portalRoles: { bramble_bandit: 'dungeon return' },
        landmarkBands: [
          { kind: 'vine', x: 620, w: 900, label: 'Root Gate' },
          { kind: 'crystal', x: 2420, w: 580, label: 'Bramble Seal' }
        ],
        spawnZoneLabels: [
          { label: 'Root floor', platformTier: 'low' },
          { label: 'Court shelf', platformTier: 'mid' },
          { label: 'Crown approach', platformTier: 'high' }
        ]
      }),
      gearworksVault: createArenaFieldComposition({
        sections: [
          { label: 'Tank Lane', tier: 'frontline', kind: 'crate', platformTier: 'low' },
          { label: 'Sentry Catwalk', tier: 'ranged', kind: 'tall', platformTier: 'mid' },
          { label: 'Gear Switch Shelf', tier: 'switch', kind: 'crystal', platformTier: 'high' }
        ],
        portalRoles: { vault_quarry: 'quarry return' }
      }),
      emberjawLair: createArenaFieldComposition({
        sections: [
          { label: 'West Vent', tier: 'vent', kind: 'glow', platformTier: 'low' },
          { label: 'Safe Pockets', tier: 'safe-room', kind: 'rock', platformTier: 'mid' },
          { label: 'Overheat Shelf', tier: 'boss', kind: 'crystal', platformTier: 'high' }
        ],
        portalRoles: { lair_cinder: 'cinder return' }
      }),
      rimewardenSanctum: createArenaFieldComposition({
        sections: [
          { label: 'Brute Lane', tier: 'frontline', kind: 'rock', platformTier: 'low' },
          { label: 'Oracle Shelf', tier: 'support', kind: 'glow', platformTier: 'mid' },
          { label: 'Sentinel Shelf', tier: 'ranged', kind: 'crystal', platformTier: 'high' }
        ],
        portalRoles: { sanctum_glacier: 'glacier return' }
      }),
      cinderHollow: createFieldComposition({
        routeSections: [
          { label: 'Ash Floor Loop', x: 0, w: 1650, tier: 'grounded' },
          { label: 'Vent Shortcut', x: 1650, w: 1650, tier: 'shortcut' },
          { label: 'Flyer Turns', x: 3300, w: 1900, tier: 'anti-air' }
        ],
        portalRoles: {
          cinder_refuge_return: 'left refuge return',
          cinder_emberjaw: 'emberjaw lair gate'
        },
        landmarkBands: [
          { kind: 'glow', x: 620, w: 860, label: 'Ash Crawler Floor' },
          { kind: 'crystal', x: 2240, w: 760, label: 'Vent Shortcut' },
          { kind: 'tall', x: 3940, w: 700, label: 'Wisp Turn' }
        ],
        spawnZoneLabels: [
          { label: 'Ash floor', platformTier: 'low' },
          { label: 'Vent shortcut', platformTier: 'mid' },
          { label: 'Flyer turn', platformTier: 'high' }
        ]
      }),
      banditRidgeCamp: createFieldComposition({
        routeSections: [
          { label: 'Lower Cutter Lane', x: 0, w: 1350, tier: 'frontline' },
          { label: 'Middle Thrower Camp', x: 1350, w: 1350, tier: 'ranged' },
          { label: 'High Rope Bridge', x: 2700, w: 1350, tier: 'anti-ranged' },
          { label: 'Campfire Regroup', x: 4050, w: 1350, tier: 'regroup' }
        ],
        portalRoles: {
          bandit_thornpath: 'ridge return',
          bandit_bramble: 'bramble dungeon gate'
        },
        landmarkBands: [
          { kind: 'crate', x: 520, w: 700, label: 'Cutter Barricade' },
          { kind: 'sign', x: 1760, w: 560, label: 'Thrower Camp' },
          { kind: 'vine', x: 3040, w: 660, label: 'Rope Bridge' },
          { kind: 'glow', x: 4500, w: 480, label: 'Campfire Regroup' }
        ],
        spawnZoneLabels: [
          { label: 'Cutter lane', platformTier: 'low' },
          { label: 'Thrower camp', platformTier: 'mid' },
          { label: 'Rope bridge', platformTier: 'high' },
          { label: 'Campfire regroup', platformTier: 'mid' }
        ]
      }),
      orebackQuarry: createFieldComposition({
        routeSections: [
          { label: 'Ore Cart Lane', x: 0, w: 1200, tier: 'frontline' },
          { label: 'Scaffold Sentries', x: 1200, w: 1200, tier: 'ranged' },
          { label: 'Mushroom Pocket', x: 2400, w: 1200, tier: 'support' },
          { label: 'Mine Event Pocket', x: 3600, w: 1200, tier: 'event' }
        ],
        portalRoles: {
          quarry_rustcoil_outpost: 'mine lift return',
          quarry_vault: 'gearworks vault gate',
          quarry_cinder_refuge: 'deep road advance'
        },
        landmarkBands: [
          { kind: 'crate', x: 360, w: 700, label: 'Ore Cart Lane' },
          { kind: 'tall', x: 1540, w: 620, label: 'Sentry Scaffold' },
          { kind: 'glow', x: 2720, w: 600, label: 'Glowcap Pocket' },
          { kind: 'crystal', x: 3960, w: 560, label: 'Mine Event Pocket' }
        ],
        spawnZoneLabels: [
          { label: 'Ore cart lane', platformTier: 'low' },
          { label: 'Scaffold sentries', platformTier: 'mid' },
          { label: 'Mushroom pocket', platformTier: 'high' },
          { label: 'Mine event pocket', platformTier: 'peak' }
        ]
      }),
      ashglassPass: createFieldComposition({
        routeSections: [
          { label: 'Ashglass Bridge', x: 0, w: 1800, tier: 'crossing' },
          { label: 'Vent Side Pocket', x: 1800, w: 1200, tier: 'side-pocket' },
          { label: 'Glass Shelf', x: 3000, w: 1200, tier: 'hazard' },
          { label: 'Elite Storm Pocket', x: 4200, w: 1000, tier: 'elite' }
        ],
        portalRoles: {
          ashglass_cinder_refuge: 'ash bridge return',
          ashglass_frostfen_camp: 'frostfen advance'
        },
        landmarkBands: [
          { kind: 'crystal', x: 520, w: 920, label: 'Ashglass Bridge' },
          { kind: 'glow', x: 2180, w: 580, label: 'Vent Side Pocket' },
          { kind: 'tall', x: 3420, w: 620, label: 'Glass Shelf' },
          { kind: 'rock', x: 4540, w: 440, label: 'Elite Storm Pocket' }
        ],
        spawnZoneLabels: [
          { label: 'Bridge walkers', platformTier: 'low' },
          { label: 'Vent flank', platformTier: 'mid' },
          { label: 'Glass shelf', platformTier: 'high' },
          { label: 'Elite pocket', platformTier: 'peak' }
        ]
      }),
      stormbreakCliffs: createFieldComposition({
        routeSections: [
          { label: 'Low Ram Lane', x: 0, w: 1300, tier: 'frontline' },
          { label: 'Mid Archer Bridge', x: 1300, w: 1500, tier: 'ranged' },
          { label: 'High Harrier Airspace', x: 2800, w: 1400, tier: 'anti-air' },
          { label: 'Lightning Rod Objective', x: 4200, w: 1000, tier: 'support' }
        ],
        portalRoles: {
          stormbreak_haven_return: 'cliff span return'
        },
        landmarkBands: [
          { kind: 'rock', x: 460, w: 720, label: 'Ram Lane' },
          { kind: 'sign', x: 1780, w: 740, label: 'Archer Bridge' },
          { kind: 'tall', x: 3180, w: 760, label: 'Harrier Airspace' },
          { kind: 'crystal', x: 4420, w: 460, label: 'Lightning Rod' }
        ],
        spawnZoneLabels: [
          { label: 'Ram lane', platformTier: 'low' },
          { label: 'Archer bridge', platformTier: 'mid' },
          { label: 'Harrier airspace', platformTier: 'high' },
          { label: 'Lightning rod', platformTier: 'peak' }
        ]
      }),
      eclipseFrontier: createFieldComposition({
        routeSections: [
          { label: 'Solar Outpost', x: 0, w: 1300, tier: 'outpost' },
          { label: 'Lunar Outpost', x: 1300, w: 1300, tier: 'outpost' },
          { label: 'Eclipse Gate', x: 2600, w: 1300, tier: 'sigil' },
          { label: 'Elite Pocket', x: 3900, w: 1300, tier: 'elite' }
        ],
        portalRoles: {
          eclipse_archive: 'archive return',
          eclipse_rift: 'rift advance'
        },
        landmarkBands: [
          { kind: 'glow', x: 420, w: 640, label: 'Solar Outpost' },
          { kind: 'crystal', x: 1680, w: 640, label: 'Lunar Outpost' },
          { kind: 'sign', x: 2920, w: 580, label: 'Eclipse Gate' },
          { kind: 'tall', x: 4200, w: 560, label: 'Elite Pocket' }
        ],
        spawnZoneLabels: [
          { label: 'Solar outpost', platformTier: 'low' },
          { label: 'Lunar outpost', platformTier: 'mid' },
          { label: 'Eclipse gate', platformTier: 'high' },
          { label: 'Elite pocket', platformTier: 'peak' }
        ]
      }),
      endlessRift: createFieldComposition({
        routeSections: [
          { label: 'Northwest Rift Quadrant', x: 0, w: 1300, tier: 'quadrant' },
          { label: 'Northeast Rift Quadrant', x: 1300, w: 1300, tier: 'quadrant' },
          { label: 'Southeast Rift Quadrant', x: 2600, w: 1300, tier: 'quadrant' },
          { label: 'Southwest Rift Quadrant', x: 3900, w: 900, tier: 'quadrant' },
          { label: 'Rift Core Regroup', x: 4800, w: 400, tier: 'surge' }
        ],
        portalRoles: {
          rift_eclipse: 'eclipse frontier return'
        },
        landmarkBands: [
          { kind: 'crystal', x: 420, w: 620, label: 'Northwest Rift' },
          { kind: 'glow', x: 1680, w: 620, label: 'Northeast Rift' },
          { kind: 'tall', x: 2940, w: 620, label: 'Southeast Rift' },
          { kind: 'flower', x: 4200, w: 520, label: 'Southwest Rift' },
          { kind: 'glow', x: 2500, w: 520, label: 'Rift Core' }
        ],
        spawnZoneLabels: [
          { label: 'Northwest quadrant', platformTier: 'high' },
          { label: 'Northeast quadrant', platformTier: 'mid' },
          { label: 'Southeast quadrant', platformTier: 'low' },
          { label: 'Southwest quadrant', platformTier: 'mid' },
          { label: 'Surge core', platformTier: 'peak' }
        ]
      }),
      bramblekingCourt: createArenaFieldComposition({
        sections: [
          { label: 'Root Lane', tier: 'frontline', kind: 'vine', platformTier: 'low' },
          { label: 'Thorn Pod Shelf', tier: 'add-control', kind: 'flower', platformTier: 'mid' },
          { label: 'Crown Platform', tier: 'boss', kind: 'crystal', platformTier: 'high' }
        ],
        portalRoles: { court_return: 'ridge return' }
      }),
      titanFoundry: createArenaFieldComposition({
        sections: [
          { label: 'Gear Floor', tier: 'frontline', kind: 'crate', platformTier: 'low' },
          { label: 'Armor Switches', tier: 'switch', kind: 'glow', platformTier: 'mid' },
          { label: 'Sentry Catwalk', tier: 'ranged', kind: 'tall', platformTier: 'high' }
        ],
        portalRoles: { foundry_return: 'quarry return' }
      }),
      deepcoreCore: createArenaFieldComposition({
        sections: [
          { label: 'Tank Chamber', tier: 'frontline', kind: 'rock', platformTier: 'low' },
          { label: 'Healer Lane', tier: 'support', kind: 'glow', platformTier: 'mid' },
          { label: 'Turret Lane', tier: 'ranged', kind: 'tall', platformTier: 'high' },
          { label: 'Ore Core', tier: 'boss', kind: 'crystal', platformTier: 'peak' }
        ],
        portalRoles: { deepcore_return: 'quarry return' }
      }),
      emberjawFurnace: createArenaFieldComposition({
        sections: [
          { label: 'Lava Cracks', tier: 'hazard', kind: 'glow', platformTier: 'low' },
          { label: 'Valve Shelf', tier: 'switch', kind: 'tall', platformTier: 'mid' },
          { label: 'Safe Pocket', tier: 'safe-room', kind: 'rock', platformTier: 'high' }
        ],
        portalRoles: { furnace_return: 'cinder return' }
      }),
      rimewardenVault: createArenaFieldComposition({
        sections: [
          { label: 'Brute Lane', tier: 'frontline', kind: 'rock', platformTier: 'low' },
          { label: 'Oracle Shelf', tier: 'support', kind: 'glow', platformTier: 'mid' },
          { label: 'Sentinel Shelf', tier: 'ranged', kind: 'crystal', platformTier: 'high' }
        ]
      }),
      stormbreakAerie: createArenaFieldComposition({
        sections: [
          { label: 'Ram Lane', tier: 'frontline', kind: 'rock', platformTier: 'low' },
          { label: 'Rod Perch', tier: 'support', kind: 'crystal', platformTier: 'mid' },
          { label: 'Harrier Airspace', tier: 'anti-air', kind: 'tall', platformTier: 'high' }
        ],
        portalRoles: { aerie_return: 'stormbreak return' }
      }),
      astralStacks: createArenaFieldComposition({
        sections: [
          { label: 'Left Stacks', tier: 'memory', kind: 'sign', platformTier: 'mid' },
          { label: 'Center Rune Shelf', tier: 'rune', kind: 'glow', platformTier: 'high' },
          { label: 'Right Stacks', tier: 'mirror', kind: 'crystal', platformTier: 'mid' }
        ]
      }),
      eclipseThrone: createArenaFieldComposition({
        sections: [
          { label: 'Solar Lane', tier: 'solar', kind: 'glow', platformTier: 'low' },
          { label: 'Eclipse Dais', tier: 'boss', kind: 'crystal', platformTier: 'mid' },
          { label: 'Lunar Lane', tier: 'lunar', kind: 'tall', platformTier: 'high' },
          { label: 'Mote Shelf', tier: 'add-control', kind: 'rock', platformTier: 'peak' }
        ]
      })
    });

    function normalizeDesignToken(value, fallback) {
      const token = String(value || fallback || 'section')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');
      return token || String(fallback || 'section');
    }

    function createDesignIntent(config) {
      return Object.freeze({
        intendedArchetype: config.intendedArchetype || 'F loop map',
        intendedUseCase: config.intendedUseCase || 'solo/duo',
        routeSummary: config.routeSummary || 'Clear the main route and return as spawns repopulate.',
        partyRoleTarget: config.partyRoleTarget || 'Solo rhythm first, light duo overlap.',
        farmingAbuseRisk: config.farmingAbuseRisk || 'medium',
        visualIdentityTag: config.visualIdentityTag || 'starlit frontier route',
        spawnSectionModel: config.spawnSectionModel || 'section-local loop pressure',
        partyScaling: config.partyScaling || 'none',
        priorityRedesign: !!config.priorityRedesign,
        implementationStatus: config.implementationStatus || 'intent-metadata'
      });
    }

    const MAP_DESIGN_INTENTS = Object.freeze([
      { id: 'greenrootMeadow', intendedArchetype: 'A -> F starter lane loop', intendedUseCase: 'solo', routeSummary: 'Clear bottom pond, hop to moss lane, drop back to start, then unlock wider meadow tiers.', partyRoleTarget: 'Keep early Greenroot mostly solo with optional canopy ranged practice.', farmingAbuseRisk: 'low', visualIdentityTag: 'beginner meadow and slime pond', spawnSectionModel: 'pond/moss/canopy starter sections', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'thornpathThicket', intendedArchetype: 'D vertical canopy with reset loop', intendedUseCase: 'solo/duo', routeSummary: 'Climb vine shelves, clear visible sprout perches, then use drops to reset.', partyRoleTarget: 'Duo split by low/mid/high canopy only after reset drops exist.', farmingAbuseRisk: 'medium', visualIdentityTag: 'thorn canopy fork', spawnSectionModel: 'low vines, mid sprout shelf, high briar route' },
      { id: 'brambleDepths', intendedArchetype: 'H/E root dungeon', intendedUseCase: 'boss dungeon', routeSummary: 'Push through root lanes and thorn shelves into the Brambleking gate.', partyRoleTarget: 'Small-party boss prep with healer pod and thorn pod priority.', farmingAbuseRisk: 'medium', visualIdentityTag: 'root court dungeon', spawnSectionModel: 'root floor, thorn shelf, crown approach', implementationStatus: 'arena-skeleton-v1' },
      { id: 'rustcoilRuins', intendedArchetype: 'C/F industrial terrace', intendedUseCase: 'solo/duo', routeSummary: 'Rotate lower construct lanes, mid sentry catwalks, and upper gear shelves.', partyRoleTarget: 'Duo can split lower armor lane and upper sentry lane.', farmingAbuseRisk: 'medium', visualIdentityTag: 'broken gear ruins', spawnSectionModel: 'lower ratchets, mid sentries, upper wardens' },
      { id: 'gearworksVault', intendedArchetype: 'H/E armor-check factory arena', intendedUseCase: 'small/full party dungeon', routeSummary: 'Hold lower tanks, control mid sentries, and hit upper gear switches during boss waves.', partyRoleTarget: 'Tank lower lane, ranged sentry duty, support near center switch.', farmingAbuseRisk: 'medium', visualIdentityTag: 'gear vault factory', spawnSectionModel: 'tank lane, sentry catwalk, switch shelf', implementationStatus: 'arena-skeleton-v1' },
      { id: 'cinderHollow', intendedArchetype: 'C/F hazard-lite volcanic route', intendedUseCase: 'solo/duo', routeSummary: 'Use grounded ash lanes and vent shortcuts while avoiding flyer-heavy turns.', partyRoleTarget: 'Duo split ground crawlers from vent/flyer pockets.', farmingAbuseRisk: 'medium-high', visualIdentityTag: 'ember vent cave', spawnSectionModel: 'ash floor, lava tick lane, flyer turns', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'emberjawLair', intendedArchetype: 'H/I furnace arena', intendedUseCase: 'boss dungeon', routeSummary: 'Control side vents, cross safe pockets, and answer Emberjaw overheat waves.', partyRoleTarget: 'Small party handles vent adds, safe pocket calls, and boss pressure.', farmingAbuseRisk: 'medium', visualIdentityTag: 'ember furnace lair', spawnSectionModel: 'side vents, mid safe pockets, overheat shelf', implementationStatus: 'arena-skeleton-v1' },
      { id: 'banditRidgeCamp', intendedArchetype: 'E split-lane party map', intendedUseCase: 'small party', routeSummary: 'Split lower cutter lane, middle thrower camp, and upper rope bridge, then regroup at campfire.', partyRoleTarget: 'Tank lower chokepoints, ranged high throwers, support campfire regroup.', farmingAbuseRisk: 'medium', visualIdentityTag: 'bandit rope-bridge camp', spawnSectionModel: 'lower/middle/high/regroup lane sections', partyScaling: 'section-count', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'orebackQuarry', intendedArchetype: 'E/J material party farm', intendedUseCase: 'small party high-density farming', routeSummary: 'Rotate ore carts, scaffold sentries, healer mushroom pockets, and timed mine events.', partyRoleTarget: 'Tank ore lane, ranged scaffold duty, burst classes clear healer pockets.', farmingAbuseRisk: 'high', visualIdentityTag: 'ore cart quarry', spawnSectionModel: 'ore lane, scaffold, mushroom pocket, event pocket', partyScaling: 'section-count', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'ashglassPass', intendedArchetype: 'G/F dangerous crossing', intendedUseCase: 'exploration dangerous progression', routeSummary: 'Cross the main ashglass bridge, dip into side pockets, and avoid elite glassstorm windows.', partyRoleTarget: 'Duo route with one main bridge player and one side-pocket control player.', farmingAbuseRisk: 'high', visualIdentityTag: 'ashglass bridge crossing', spawnSectionModel: 'bridge, vent pocket, glass shelf, elite side pocket', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'frostfenOutskirts', intendedArchetype: 'C/E ice solo-duo route', intendedUseCase: 'solo/duo', routeSummary: 'Clear marsh flats, slide across ice shelves, then reset through oracle grove drops.', partyRoleTarget: 'Duo split flats and oracle shelf while keeping support reachable.', farmingAbuseRisk: 'medium', visualIdentityTag: 'frozen marsh camp route', spawnSectionModel: 'marsh flats, ice shelf, oracle grove' },
      { id: 'glacierSpine', intendedArchetype: 'D/G glacier progression climb', intendedUseCase: 'dangerous progression small party', routeSummary: 'Climb glacier checkpoints, use lifts and one-way drops to rotate sentinel chokepoints.', partyRoleTarget: 'Small party split by height with central lift regroup.', farmingAbuseRisk: 'medium-high', visualIdentityTag: 'glacier lift spine', spawnSectionModel: 'lower climb, mid ridge, high ridge, flyer airspace' },
      { id: 'rimewardenSanctum', intendedArchetype: 'H/I frost vault', intendedUseCase: 'full party dungeon', routeSummary: 'Hold lower brutes, reach oracle shelves, and react to lane-locking ice walls.', partyRoleTarget: 'Tank lower lane, ranged oracle/sentinel control, support center safe room.', farmingAbuseRisk: 'medium', visualIdentityTag: 'rime vault sanctum', spawnSectionModel: 'brute lane, oracle shelf, sentinel shelf', implementationStatus: 'arena-skeleton-v1' },
      { id: 'stormbreakCliffs', intendedArchetype: 'D/E anti-air party field', intendedUseCase: 'small party', routeSummary: 'Clear low ram lane, mid archer bridge, high harrier airspace, and lightning rod objective.', partyRoleTarget: 'Frontliner low lane, ranged anti-air, support/control on lightning rod.', farmingAbuseRisk: 'high', visualIdentityTag: 'storm mast cliff climb', spawnSectionModel: 'ram lane, archer bridge, harrier airspace, rod objective', partyScaling: 'section-count', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'astralArchive', intendedArchetype: 'C/F room-loop archive', intendedUseCase: 'late solo/duo', routeSummary: 'Loop reading rooms through rune lifts and break line-of-sight shelves.', partyRoleTarget: 'Duo split adjacent rooms and regroup at archive console.', farmingAbuseRisk: 'high', visualIdentityTag: 'rune archive rooms', spawnSectionModel: 'reading room loops and index shelves' },
      { id: 'eclipseFrontier', intendedArchetype: 'G/J elite frontier', intendedUseCase: 'high-risk farming small party', routeSummary: 'Patrol three outposts, rotate eclipse sigils, and enter capped elite pockets.', partyRoleTarget: 'Small party splits outposts, then regroups for sigil/elite pulses.', farmingAbuseRisk: 'very high', visualIdentityTag: 'eclipse frontier outposts', spawnSectionModel: 'outpost A, outpost B, eclipse gate, elite pocket', partyScaling: 'section-count' },
      { id: 'endlessRift', intendedArchetype: 'F/J scaling circular loop', intendedUseCase: 'endgame high-density farming', routeSummary: 'Rotate four rift quadrants, respond to surge warnings, and regroup at the core.', partyRoleTarget: 'Party holds quadrants and regroups at rift core during surge events.', farmingAbuseRisk: 'very high', visualIdentityTag: 'unstable rift loop', spawnSectionModel: 'four quadrant loop with surge pocket', partyScaling: 'section-count', priorityRedesign: true, implementationStatus: 'geometry-spawn-v1' },
      { id: 'bramblekingCourt', intendedArchetype: 'H/E root boss room', intendedUseCase: 'boss', routeSummary: 'Move between root lanes, destroy thorn pods, and expose the Brambleking crown.', partyRoleTarget: 'Small party splits root floor and pod shelf.', farmingAbuseRisk: 'medium', visualIdentityTag: 'bramble crown court', spawnSectionModel: 'root lane, pod shelf, crown platform', implementationStatus: 'arena-skeleton-v1' },
      { id: 'titanFoundry', intendedArchetype: 'H/E factory boss room', intendedUseCase: 'boss full party', routeSummary: 'Operate armor switches while sentries pressure the upper catwalk.', partyRoleTarget: 'Tank boss floor, ranged sentries, support switch rotation.', farmingAbuseRisk: 'medium', visualIdentityTag: 'titan gear foundry', spawnSectionModel: 'gear floor, armor switches, sentry catwalk', implementationStatus: 'arena-skeleton-v1' },
      { id: 'deepcoreCore', intendedArchetype: 'H/E quarry core boss room', intendedUseCase: 'boss full party', routeSummary: 'Split four mining chambers around the central ore core.', partyRoleTarget: 'Full party assigns tank, healer-control, turret, and ore chamber jobs.', farmingAbuseRisk: 'medium-high', visualIdentityTag: 'deep quarry core', spawnSectionModel: 'tank chamber, healer lane, turret lane, ore core', implementationStatus: 'arena-skeleton-v1' },
      { id: 'emberjawFurnace', intendedArchetype: 'H/I furnace boss room', intendedUseCase: 'boss', routeSummary: 'Rotate lava cracks, vent valves, and safe pockets during overheat.', partyRoleTarget: 'Small party controls vent valves and side adds.', farmingAbuseRisk: 'medium', visualIdentityTag: 'emberjaw furnace cracks', spawnSectionModel: 'lava cracks, valve shelf, safe pocket', implementationStatus: 'arena-skeleton-v1' },
      { id: 'rimewardenVault', intendedArchetype: 'H/I frost vault boss room', intendedUseCase: 'boss full party', routeSummary: 'React to whiteout and ice-wall lane locks while clearing oracle shelves.', partyRoleTarget: 'Frontline lower brutes, ranged oracle shelf, support safe room.', farmingAbuseRisk: 'medium', visualIdentityTag: 'rimewarden ice vault', spawnSectionModel: 'brute lane, oracle shelf, sentinel shelf', implementationStatus: 'arena-skeleton-v1' },
      { id: 'stormbreakAerie', intendedArchetype: 'H/D vertical flying boss room', intendedUseCase: 'boss full party', routeSummary: 'Use wind lanes to reach lightning-rod perches while the Roc controls airspace.', partyRoleTarget: 'Ranged anti-air, melee rod/ram duty, support perch calls.', farmingAbuseRisk: 'medium-high', visualIdentityTag: 'storm aerie perches', spawnSectionModel: 'ram lane, rod perch, harrier airspace', implementationStatus: 'arena-skeleton-v1' },
      { id: 'astralStacks', intendedArchetype: 'H/G mirrored archive boss room', intendedUseCase: 'boss', routeSummary: 'Rotate mirrored shelves and rune lanes to satisfy action-memory pressure.', partyRoleTarget: 'Small party splits mirrored stacks and calls memory runes.', farmingAbuseRisk: 'medium-high', visualIdentityTag: 'mirrored astral stacks', spawnSectionModel: 'left stacks, right stacks, center rune shelf', implementationStatus: 'arena-skeleton-v1' },
      { id: 'eclipseThrone', intendedArchetype: 'H/I capstone zone-rotation room', intendedUseCase: 'boss full party', routeSummary: 'Rotate solar and lunar lanes around the central eclipse dais during totality.', partyRoleTarget: 'Full party splits solar/lunar duties and regroups for elite pulse.', farmingAbuseRisk: 'high', visualIdentityTag: 'solar lunar eclipse throne', spawnSectionModel: 'solar lane, lunar lane, eclipse dais, mote shelf', implementationStatus: 'arena-skeleton-v1' }
    ].reduce((intents, config) => {
      intents[config.id] = createDesignIntent(config);
      return intents;
    }, {}));

    function freezeMapMechanicSections(sections) {
      return Object.freeze((sections || []).map((section) => Object.freeze({
        id: section.id || '',
        label: section.label || '',
        role: section.role || '',
        weight: Number(section.weight || 1),
        rewardWeight: Number(section.rewardWeight || section.weight || 1)
      })));
    }

    function createMapMechanicDefinition(config) {
      return Object.freeze({
        id: config.id,
        mapId: config.mapId,
        type: config.type,
        label: config.label,
        summary: config.summary,
        sections: freezeMapMechanicSections(config.sections),
        activeSectionIds: Object.freeze((config.activeSectionIds || []).slice()),
        objectiveSectionId: config.objectiveSectionId || '',
        regroupSectionId: config.regroupSectionId || '',
        eventKillGoal: Math.max(1, Number(config.eventKillGoal || config.goal || 1)),
        requiredUniqueSections: Math.max(1, Number(config.requiredUniqueSections || 1)),
        repeatWarningThreshold: Math.max(2, Number(config.repeatWarningThreshold || 4)),
        penaltyPerStack: Math.max(0, Number(config.penaltyPerStack || 0.08)),
        minimumRewardScale: Math.max(0.1, Number(config.minimumRewardScale || 0.65)),
        reward: Object.freeze(config.reward || {}),
        rewardAbuseControl: config.rewardAbuseControl || '',
        partyRoleHook: config.partyRoleHook || ''
      });
    }

    const MAP_MECHANIC_DEFINITIONS = Object.freeze({
      orebackQuarry: createMapMechanicDefinition({
        id: 'oreback_material_rush',
        mapId: 'orebackQuarry',
        type: 'material-event',
        label: 'Ore Cart Material Rush',
        summary: 'Rotate through quarry lanes to charge the active mine pocket and claim capped ore rewards.',
        sections: [
          { id: 'orebackQuarry_ore_cart_lane', label: 'Ore Cart Lane', role: 'frontline', weight: 1.15 },
          { id: 'orebackQuarry_scaffold_sentries', label: 'Scaffold Sentries', role: 'ranged', weight: 1 },
          { id: 'orebackQuarry_mushroom_pocket', label: 'Mushroom Pocket', role: 'support', weight: 0.9 },
          { id: 'orebackQuarry_mine_event_pocket', label: 'Mine Event Pocket', role: 'event', weight: 1.35 }
        ],
        activeSectionIds: ['orebackQuarry_ore_cart_lane', 'orebackQuarry_scaffold_sentries', 'orebackQuarry_mushroom_pocket', 'orebackQuarry_mine_event_pocket'],
        objectiveSectionId: 'orebackQuarry_mine_event_pocket',
        regroupSectionId: 'orebackQuarry_mine_event_pocket',
        eventKillGoal: 5,
        requiredUniqueSections: 1,
        repeatWarningThreshold: 4,
        penaltyPerStack: 0.09,
        minimumRewardScale: 0.62,
        reward: { currency: 80, materials: { oreChunks: 8, upgradeDust: 3 } },
        rewardAbuseControl: 'Active section rotates after each material rush; repeated same-section kills reduce mechanic rewards.',
        partyRoleHook: 'Tank ore carts, ranged clears scaffold sentries, burst classes break mushroom pockets before the event pocket rotates.'
      }),
      stormbreakCliffs: createMapMechanicDefinition({
        id: 'stormbreak_lightning_rod',
        mapId: 'stormbreakCliffs',
        type: 'party-objective',
        label: 'Lightning Rod Charge',
        summary: 'Charge the cliff rod by splitting low, mid, and anti-air duties before regrouping at the objective perch.',
        sections: [
          { id: 'stormbreakCliffs_low_ram_lane', label: 'Low Ram Lane', role: 'frontline', weight: 1 },
          { id: 'stormbreakCliffs_mid_archer_bridge', label: 'Mid Archer Bridge', role: 'ranged', weight: 1.5 },
          { id: 'stormbreakCliffs_high_harrier_airspace', label: 'High Harrier Airspace', role: 'anti-air', weight: 2 },
          { id: 'stormbreakCliffs_lightning_rod_objective', label: 'Lightning Rod Objective', role: 'support', weight: 2.25 }
        ],
        activeSectionIds: ['stormbreakCliffs_low_ram_lane', 'stormbreakCliffs_mid_archer_bridge', 'stormbreakCliffs_high_harrier_airspace'],
        objectiveSectionId: 'stormbreakCliffs_lightning_rod_objective',
        regroupSectionId: 'stormbreakCliffs_lightning_rod_objective',
        eventKillGoal: 8,
        requiredUniqueSections: 3,
        repeatWarningThreshold: 4,
        penaltyPerStack: 0.07,
        minimumRewardScale: 0.68,
        reward: { currency: 120, materials: { stormFletching: 5, galeFeather: 3 } },
        rewardAbuseControl: 'Rod charge requires three distinct cliff jobs so one safe lane cannot farm the full objective.',
        partyRoleHook: 'Frontliner holds rams, ranged handles archers and harriers, support regroups at the lightning rod.'
      }),
      endlessRift: createMapMechanicDefinition({
        id: 'endless_rift_surge',
        mapId: 'endlessRift',
        type: 'surge-loop',
        label: 'Rift Surge Rotation',
        summary: 'Rotate all four Rift quadrants to trigger surge windows; repeated quadrant camping suppresses Rift score.',
        sections: [
          { id: 'endlessRift_northwest_rift_quadrant', label: 'Northwest Rift Quadrant', role: 'quadrant', weight: 1 },
          { id: 'endlessRift_northeast_rift_quadrant', label: 'Northeast Rift Quadrant', role: 'quadrant', weight: 1 },
          { id: 'endlessRift_southeast_rift_quadrant', label: 'Southeast Rift Quadrant', role: 'quadrant', weight: 1 },
          { id: 'endlessRift_southwest_rift_quadrant', label: 'Southwest Rift Quadrant', role: 'quadrant', weight: 1 },
          { id: 'endlessRift_rift_core_regroup', label: 'Rift Core Regroup', role: 'surge', weight: 1.5 }
        ],
        activeSectionIds: ['endlessRift_northwest_rift_quadrant', 'endlessRift_northeast_rift_quadrant', 'endlessRift_southeast_rift_quadrant', 'endlessRift_southwest_rift_quadrant'],
        objectiveSectionId: 'endlessRift_rift_core_regroup',
        regroupSectionId: 'endlessRift_rift_core_regroup',
        eventKillGoal: 12,
        requiredUniqueSections: 4,
        repeatWarningThreshold: 4,
        penaltyPerStack: 0.12,
        minimumRewardScale: 0.5,
        reward: { currency: 180, materials: { riftSplinter: 2, cubeFragment: 3 } },
        rewardAbuseControl: 'Surge rewards and Rift score scale down when players camp one quadrant instead of rotating all four.',
        partyRoleHook: 'Party members hold quadrants, then collapse to the Rift Core during surge windows.'
      })
    });

    const TOWN_SERVICE_PLANS = Object.freeze({
      starfallCrossing: Object.freeze({ phrase: 'meteor guild starter plaza', landmark: 'Meteor plaza / Adventurer Hall', socialPocket: 'Meteor fountain plaza', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade', 'plinko']), lowFrequencyServiceIds: Object.freeze(['class']), portalStyle: 'forest gate preview' }),
      rustcoilOutpost: Object.freeze({ phrase: 'gear tower outpost', landmark: 'Gear tower', socialPocket: 'scrap dispatcher yard', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade']), lowFrequencyServiceIds: Object.freeze([]), portalStyle: 'gear gate and mine lift' }),
      cinderRefuge: Object.freeze({ phrase: 'furnace shelter refuge', landmark: 'Furnace shelter', socialPocket: 'ash market hearth', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade']), lowFrequencyServiceIds: Object.freeze([]), portalStyle: 'cave vent and ashglass bridge' }),
      frostfenCamp: Object.freeze({ phrase: 'ice signal camp', landmark: 'Ice signal lodge', socialPocket: 'campfire supply circle', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade']), lowFrequencyServiceIds: Object.freeze([]), portalStyle: 'tundra gate and glacier lift' }),
      stormbreakHaven: Object.freeze({ phrase: 'storm mast haven', landmark: 'Storm mast', socialPocket: 'sheltered landing platform', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade']), lowFrequencyServiceIds: Object.freeze([]), portalStyle: 'cliff span and sky elevator' }),
      astralObservatory: Object.freeze({ phrase: 'star lens observatory', landmark: 'Star lens', socialPocket: 'star-lens plaza', highFrequencyServiceIds: Object.freeze(['storage', 'shop']), mediumFrequencyServiceIds: Object.freeze(['slots', 'upgrade']), lowFrequencyServiceIds: Object.freeze([]), portalStyle: 'archive gate and rune lift' })
    });

    const STATION_SERVICE_INTENTS = Object.freeze({
      storage: Object.freeze({ serviceTier: 'high', serviceRole: 'storage', serviceSummary: 'Quick loot deposit and shared storage.' }),
      shop: Object.freeze({ serviceTier: 'high', serviceRole: 'supply_repair_equipment', serviceSummary: 'Potion, repair, and equipment restock.' }),
      slots: Object.freeze({ serviceTier: 'medium', serviceRole: 'inventory_expansion', serviceSummary: 'Inventory-slot and account convenience spending.' }),
      upgrade: Object.freeze({ serviceTier: 'medium', serviceRole: 'upgrade_crafting', serviceSummary: 'Gear upgrade and crafting sink.' }),
      plinko: Object.freeze({ serviceTier: 'medium', serviceRole: 'minigame_reward_sink', serviceSummary: 'Mob-earned Plinko-ball reward loop.' }),
      class: Object.freeze({ serviceTier: 'low', serviceRole: 'class_flavor', serviceSummary: 'Class trials, class supplies, and onboarding flavor.' })
    });

    const MAP_PORTAL_FICTION = Object.freeze({
      crossing_greenroot: Object.freeze({ roleLabel: 'forest gate preview', portalStyle: 'forest gate' }),
      rustcoil_outpost_ruins: Object.freeze({ roleLabel: 'broken gear gate', portalStyle: 'gear gate' }),
      rustcoil_outpost_quarry: Object.freeze({ roleLabel: 'mine lift to quarry', portalStyle: 'mine lift' }),
      cinder_refuge_hollow: Object.freeze({ roleLabel: 'cave vent gate', portalStyle: 'cave vent' }),
      cinder_refuge_ashglass: Object.freeze({ roleLabel: 'locked ashglass bridge', portalStyle: 'ashglass bridge' }),
      frostfen_camp_outskirts: Object.freeze({ roleLabel: 'tundra gate', portalStyle: 'snowfield gate' }),
      frostfen_camp_glacier: Object.freeze({ roleLabel: 'glacier lift', portalStyle: 'glacier lift' }),
      stormbreak_haven_cliffs: Object.freeze({ roleLabel: 'storm cliff span', portalStyle: 'cliff span' }),
      stormbreak_haven_observatory: Object.freeze({ roleLabel: 'sky elevator to observatory', portalStyle: 'sky elevator' }),
      astral_observatory_archive: Object.freeze({ roleLabel: 'archive gate', portalStyle: 'archive gate' }),
      archive_eclipse: Object.freeze({ roleLabel: 'eclipse frontier gate', portalStyle: 'eclipse gate' }),
      eclipse_rift: Object.freeze({ roleLabel: 'unstable rift portal', portalStyle: 'rift lens' })
    });

    function getTownServicePlan(mapId) {
      return TOWN_SERVICE_PLANS[mapId] || null;
    }

    function getStationServiceIntent(stationId) {
      return STATION_SERVICE_INTENTS[stationId] || Object.freeze({ serviceTier: 'low', serviceRole: 'flavor', serviceSummary: 'Low-frequency town flavor service.' });
    }

    function createSpawnSections(map, fieldComposition, designIntent) {
      if (!map || map.safeZone) return Object.freeze([]);
      const worldWidth = Math.max(1, Number(map.platforms && map.platforms[0] && (map.platforms[0].w || map.platforms[0][2]) || map.worldWidth || 3600));
      const sourceSections = fieldComposition && Array.isArray(fieldComposition.routeSections) && fieldComposition.routeSections.length
        ? fieldComposition.routeSections
        : [
            { label: 'Entry', x: 0, w: worldWidth / 3, tier: 'entry' },
            { label: 'Route', x: worldWidth / 3, w: worldWidth / 3, tier: 'route' },
            { label: 'Exit', x: worldWidth * 2 / 3, w: worldWidth / 3, tier: 'exit' }
          ];
      return Object.freeze(sourceSections.map((section, index) => {
        const label = String(section.label || `Section ${index + 1}`);
        return Object.freeze({
          id: `${map.id}_${normalizeDesignToken(label, `section_${index + 1}`)}`,
          label,
          x: Math.round(Math.max(0, Number(section.x || 0))),
          w: Math.round(Math.max(1, Number(section.w || worldWidth / sourceSections.length))),
          tier: String(section.tier || ''),
          spawnModel: designIntent && designIntent.spawnSectionModel || 'section-local loop pressure'
        });
      }));
    }

    function getSpawnSectionForPoint(point, sections) {
      if (!point || !sections || !sections.length) return null;
      const x = Number(point.x || 0);
      return sections.find((section) => x >= section.x && x <= section.x + section.w) ||
        sections.slice().sort((a, b) =>
          Math.abs(x - (a.x + a.w / 2)) - Math.abs(x - (b.x + b.w / 2))
        )[0] ||
        null;
    }

    function attachSpawnSectionsToPoints(map, spawnSections) {
      return Object.freeze((map.spawnPoints || []).map((point, index) => {
        const section = getSpawnSectionForPoint(point, spawnSections);
        return Object.freeze(Object.assign({}, point, {
          id: point.id || `${map.id}_spawn_${index}`,
          sectionId: section ? section.id : '',
          sectionLabel: section ? section.label : ''
        }));
      }));
    }


    return Object.freeze({
      MAP_LAYOUT_ROLES,
      MAP_LAYOUT_ROLE_LABELS,
      normalizeMapLayoutRole,
      getMapLayoutRoleFallback,
      MAP_LAYOUT_BLUEPRINTS,
      MAP_TOWN_SCENES,
      MAP_FIELD_COMPOSITIONS,
      MAP_DESIGN_INTENTS,
      MAP_MECHANIC_DEFINITIONS,
      TOWN_SERVICE_PLANS,
      STATION_SERVICE_INTENTS,
      MAP_PORTAL_FICTION,
      createTownScene,
      createFieldComposition,
      createDefaultTownScene,
      createDefaultFieldComposition,
      createDesignIntent,
      getTownServicePlan,
      getStationServiceIntent,
      createSpawnSections,
      attachSpawnSectionsToPoints
    });
  }

  const defaultMapPresentationData = createMapPresentationData();
  const api = Object.assign({
    createMapPresentationData,
    defaultGetAuthoredMapWidth
  }, defaultMapPresentationData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapPresentation = Object.assign({}, modules.mapPresentation || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
