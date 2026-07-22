(function initProjectStarfallDataMapLayouts(global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  const TRAINING_LANE_Y = Object.freeze({
    ground: 520,
    low: 456,
    mid: 318,
    high: 180,
    lowConnector: 386,
    highConnector: 249
  });

  const TOWN_WORLD_HEIGHT = 900;
  const TOWN_LANE_Y = Object.freeze({
    ground: 780,
    low: 668,
    mid: 540,
    high: 414,
    roof: 302
  });

  function getTownStationPlacement(stationId) {
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

  const VERTICAL_FIELD_WORLD_HEIGHT = 1180;
  const TALL_FIELD_WORLD_HEIGHT = 1260;
  const VERTICAL_LANE_Y = Object.freeze({
    ground: 1040,
    low: 880,
    lowConnector: 790,
    mid: 700,
    highConnector: 610,
    high: 520,
    peak: 340,
    sky: 220
  });

  const VERTICAL_FIELD_LAYOUTS = Object.freeze([
    'verticalCanopy',
    'industrialStack',
    'lavaShaft',
    'quarryShaft',
    'glacierClimb',
    'stormClimb',
    'astralStack',
    'riftStack'
  ]);

  function isVerticalFieldLayout(layoutStyle) {
    return VERTICAL_FIELD_LAYOUTS.includes(layoutStyle);
  }

  function getFieldLayoutWorldHeight(layoutStyle) {
    if (!isVerticalFieldLayout(layoutStyle)) return 0;
    return layoutStyle === 'stormClimb' || layoutStyle === 'astralStack' || layoutStyle === 'riftStack'
      ? TALL_FIELD_WORLD_HEIGHT
      : VERTICAL_FIELD_WORLD_HEIGHT;
  }

  function getFieldLaneY(layoutStyle) {
    if (!isVerticalFieldLayout(layoutStyle)) return TRAINING_LANE_Y;
    const worldHeight = getFieldLayoutWorldHeight(layoutStyle);
    const ground = worldHeight - 140;
    return Object.freeze(Object.assign({}, VERTICAL_LANE_Y, {
      ground,
      low: ground - 160,
      lowConnector: ground - 250,
      mid: ground - 340,
      highConnector: ground - 430,
      high: ground - 520,
      peak: ground - 700,
      sky: ground - 820
    }));
  }

  function getPartyPlayZoneAnchors(width, options) {
    const worldWidth = Math.max(3600, Math.ceil(Number(width || 0) / 100) * 100);
    const settings = options || {};
    const zoneCount = settings.dungeon
      ? worldWidth >= 5600 ? 3 : 2
      : worldWidth >= 9000 ? 5 : 4;
    const first = 260;
    const last = Math.max(first, worldWidth - 2260);
    if (zoneCount <= 1) return [first];
    return Array.from({ length: zoneCount }, (_, index) => Math.round(first + (last - first) * index / (zoneCount - 1)));
  }

  function getFieldZoneAnchors(width, layoutStyle) {
    const worldWidth = Math.max(isVerticalFieldLayout(layoutStyle) ? 4600 : 6200, Math.ceil(Number(width || 0) / 100) * 100);
    const first = 260;
    const zoneCount = isVerticalFieldLayout(layoutStyle) ? worldWidth >= 6200 ? 4 : 3 : 4;
    const last = Math.max(first, worldWidth - (isVerticalFieldLayout(layoutStyle) ? 1500 : 2020));
    return Array.from({ length: zoneCount }, (_, index) => Math.round(first + (last - first) * index / Math.max(1, zoneCount - 1)));
  }

  function getFieldClimbableKind(layoutStyle) {
    if (layoutStyle === 'verticalCanopy') return 'vine';
    if (layoutStyle === 'lavaShaft') return 'chain';
    if (layoutStyle === 'industrialStack' || layoutStyle === 'quarryShaft') return 'lift';
    if (layoutStyle === 'glacierClimb') return 'frost_ladder';
    if (layoutStyle === 'stormClimb') return 'storm_stair';
    if (layoutStyle === 'astralStack' || layoutStyle === 'riftStack') return 'rune_stair';
    return 'rope';
  }

  const FIELD_LAYOUT_STYLES = Object.freeze({
    greenrootMeadow: 'sharedLanes',
    thornpathThicket: 'verticalCanopy',
    rustcoilRuins: 'industrialStack',
    cinderHollow: 'lavaShaft',
    banditRidgeCamp: 'switchbackTerraces',
    banditAnimationLab: 'sharedLanes',
    orebackQuarry: 'quarryShaft',
    ashglassPass: 'lavaShaft',
    frostfenOutskirts: 'switchbackTerraces',
    glacierSpine: 'glacierClimb',
    stormbreakCliffs: 'stormClimb',
    astralArchive: 'astralStack',
    eclipseFrontier: 'astralStack',
    endlessRift: 'riftStack'
  });

  function getFieldLayoutStyle(map) {
    return map && (map.layoutStyle || FIELD_LAYOUT_STYLES[map.id]) || 'sharedLanes';
  }

  const VERTICAL_FIELD_GEOMETRY = Object.freeze({
    verticalCanopy: Object.freeze({ low: 80, lowFlip: 260, mid: 360, midFlip: 80, high: 180, highFlip: 420, peak: 480, peakFlip: 220, sky: 560, skyFlip: 280, lowW: 780, midW: 760, highW: 700, peakW: 540, skyW: 360, groundRampW: 340, midRampW: 360, highRampW: 330, lift: 0 }),
    industrialStack: Object.freeze({ low: 140, lowFlip: 320, mid: 420, midFlip: 120, high: 250, highFlip: 500, peak: 560, peakFlip: 280, sky: 640, skyFlip: 360, lowW: 720, midW: 740, highW: 660, peakW: 600, skyW: 340, groundRampW: 320, midRampW: 330, highRampW: 320, lift: 0 }),
    lavaShaft: Object.freeze({ low: 180, lowFlip: 300, mid: 300, midFlip: 60, high: 90, highFlip: 520, peak: 430, peakFlip: 180, sky: 700, skyFlip: 300, lowW: 760, midW: 690, highW: 720, peakW: 500, skyW: 320, groundRampW: 330, midRampW: 360, highRampW: 340, lift: -6 }),
    quarryShaft: Object.freeze({ low: 40, lowFlip: 300, mid: 420, midFlip: 150, high: 180, highFlip: 470, peak: 360, peakFlip: 240, sky: 560, skyFlip: 420, lowW: 840, midW: 760, highW: 680, peakW: 560, skyW: 320, groundRampW: 360, midRampW: 320, highRampW: 320, lift: 0 }),
    glacierClimb: Object.freeze({ low: 120, lowFlip: 240, mid: 460, midFlip: 40, high: 260, highFlip: 420, peak: 520, peakFlip: 160, sky: 680, skyFlip: 260, lowW: 740, midW: 720, highW: 700, peakW: 560, skyW: 360, groundRampW: 320, midRampW: 340, highRampW: 320, lift: 12 }),
    stormClimb: Object.freeze({ low: 200, lowFlip: 360, mid: 500, midFlip: 170, high: 340, highFlip: 560, peak: 620, peakFlip: 300, sky: 760, skyFlip: 420, lowW: 700, midW: 700, highW: 660, peakW: 520, skyW: 340, groundRampW: 300, midRampW: 320, highRampW: 300, lift: 30 }),
    astralStack: Object.freeze({ low: 100, lowFlip: 340, mid: 500, midFlip: 140, high: 220, highFlip: 520, peak: 640, peakFlip: 300, sky: 760, skyFlip: 400, lowW: 720, midW: 700, highW: 680, peakW: 580, skyW: 360, groundRampW: 320, midRampW: 320, highRampW: 320, lift: 24 }),
    riftStack: Object.freeze({ low: 180, lowFlip: 220, mid: 540, midFlip: 80, high: 320, highFlip: 440, peak: 700, peakFlip: 200, sky: 820, skyFlip: 340, lowW: 680, midW: 680, highW: 660, peakW: 560, skyW: 380, groundRampW: 300, midRampW: 310, highRampW: 300, lift: 36 })
  });

  function getMapGeometrySeed(key) {
    const text = String(key || '');
    let seed = 0;
    for (let index = 0; index < text.length; index += 1) {
      seed = (seed * 31 + text.charCodeAt(index)) % 997;
    }
    return seed;
  }

  function getVerticalFieldGeometry(layoutStyle, variantKey) {
    const base = VERTICAL_FIELD_GEOMETRY[layoutStyle] || VERTICAL_FIELD_GEOMETRY.verticalCanopy;
    const seed = getMapGeometrySeed(variantKey || layoutStyle);
    const offset = (divisor, scale) => ((Math.floor(seed / divisor) % 5) - 2) * scale;
    return Object.freeze(Object.assign({}, base, {
      mapShift: offset(1, 44),
      peakShift: offset(5, 40),
      skyShift: offset(25, 42),
      lowW: clamp(base.lowW + offset(7, 26), 650, 900),
      midW: clamp(base.midW + offset(11, 24), 650, 860),
      highW: clamp(base.highW + offset(13, 22), 650, 820),
      peakW: clamp(base.peakW + offset(17, 24), 520, 660),
      skyW: clamp(base.skyW + offset(19, 18), 300, 430),
      groundRampW: clamp(base.groundRampW + offset(23, 10), 280, 390),
      midRampW: clamp(base.midRampW + offset(29, 10), 290, 390),
      highRampW: clamp(base.highRampW + offset(31, 10), 280, 370),
      lift: Number(base.lift || 0) + offset(37, 6)
    }));
  }

  const PRIORITY_FIELD_LAYOUT_IDS = Object.freeze([
    'greenrootMeadow',
    'thornpathThicket',
    'rustcoilRuins',
    'banditRidgeCamp',
    'orebackQuarry',
    'cinderHollow',
    'ashglassPass',
    'frostfenOutskirts',
    'stormbreakCliffs',
    'endlessRift'
  ]);

  const DUNGEON_ARENA_SKELETONS = Object.freeze({
    brambleDepths: Object.freeze({ id: 'root-lanes', mechanic: 'root lanes and thorn-pod shelves', left: 260, right: 2380, lowW: 980, midW: 820, highW: 700, midInset: 300, highInset: 620, rightMidInset: 260, rightHighInset: 540, lowShift: -8, midShift: -4, highShift: -12 }),
    gearworksVault: Object.freeze({ id: 'gear-switch-vault', mechanic: 'lower tanks, sentry catwalks, and gear switches', left: 320, right: 2460, lowW: 940, midW: 800, highW: 720, midInset: 260, highInset: 540, rightMidInset: 320, rightHighInset: 120, lowShift: 10, midShift: -12, highShift: 6 }),
    emberjawLair: Object.freeze({ id: 'furnace-vents', mechanic: 'side vents, safe pockets, and overheat shelves', left: 280, right: 2440, lowW: 900, midW: 760, highW: 680, midInset: 340, highInset: 660, rightMidInset: 220, rightHighInset: 520, lowShift: 18, midShift: 8, highShift: -16 }),
    rimewardenSanctum: Object.freeze({ id: 'ice-wall-vault', mechanic: 'brute lane, oracle shelf, and sentinel shelf', left: 340, right: 2400, lowW: 960, midW: 820, highW: 700, midInset: 280, highInset: 600, rightMidInset: 300, rightHighInset: 80, lowShift: -4, midShift: 16, highShift: 12 }),
    bramblekingCourt: Object.freeze({ id: 'crowned-root-court', mechanic: 'root floor, thorn pods, and crown platform', left: 300, right: 2500, lowW: 980, midW: 820, highW: 700, midInset: 320, highInset: 660, rightMidInset: 200, rightHighInset: 500, lowShift: -10, midShift: -8, highShift: -18 }),
    titanFoundry: Object.freeze({ id: 'armor-switch-foundry', mechanic: 'gear floor, armor switches, and upper sentries', left: 360, right: 2400, lowW: 940, midW: 820, highW: 720, midInset: 240, highInset: 560, rightMidInset: 360, rightHighInset: 160, lowShift: 14, midShift: -16, highShift: 2 }),
    deepcoreCore: Object.freeze({ id: 'four-chamber-core', mechanic: 'tank, healer, turret, and ore-core chambers', left: 260, right: 2380, lowW: 1000, midW: 820, highW: 700, midInset: 360, highInset: 680, rightMidInset: 260, rightHighInset: 520, lowShift: 8, midShift: 12, highShift: -6 }),
    emberjawFurnace: Object.freeze({ id: 'lava-valve-furnace', mechanic: 'lava cracks, vent valves, and safe pockets', left: 340, right: 2460, lowW: 900, midW: 760, highW: 680, midInset: 280, highInset: 620, rightMidInset: 260, rightHighInset: 500, lowShift: 20, midShift: 6, highShift: -20 }),
    rimewardenVault: Object.freeze({ id: 'whiteout-ice-vault', mechanic: 'whiteout lanes and locked ice shelves', left: 300, right: 2440, lowW: 960, midW: 800, highW: 720, midInset: 340, highInset: 620, rightMidInset: 220, rightHighInset: 120, lowShift: -6, midShift: 18, highShift: 10 }),
    stormbreakAerie: Object.freeze({ id: 'lightning-rod-aerie', mechanic: 'ram floor, rod perches, and harrier airspace', left: 260, right: 2340, lowW: 920, midW: 780, highW: 720, midInset: 360, highInset: 720, rightMidInset: 340, rightHighInset: 180, lowShift: -18, midShift: -22, highShift: -26 }),
    astralStacks: Object.freeze({ id: 'mirrored-archive-stacks', mechanic: 'mirrored shelves and center rune memory', left: 360, right: 2380, lowW: 900, midW: 820, highW: 760, midInset: 220, highInset: 540, rightMidInset: 420, rightHighInset: 220, lowShift: 8, midShift: -10, highShift: -8 }),
    eclipseThrone: Object.freeze({ id: 'solar-lunar-throne', mechanic: 'solar lane, lunar lane, eclipse dais, and mote shelf', left: 280, right: 2500, lowW: 960, midW: 800, highW: 700, midInset: 320, highInset: 660, rightMidInset: 180, rightHighInset: 460, lowShift: 4, midShift: -18, highShift: -4 })
  });

  function getDungeonArenaSkeleton(mapId) {
    return DUNGEON_ARENA_SKELETONS[String(mapId || '')] || null;
  }

  const api = Object.freeze({
    TRAINING_LANE_Y,
    TOWN_WORLD_HEIGHT,
    TOWN_LANE_Y,
    getTownStationPlacement,
    VERTICAL_FIELD_WORLD_HEIGHT,
    TALL_FIELD_WORLD_HEIGHT,
    VERTICAL_LANE_Y,
    VERTICAL_FIELD_LAYOUTS,
    isVerticalFieldLayout,
    getFieldLayoutWorldHeight,
    getFieldLaneY,
    getPartyPlayZoneAnchors,
    getFieldZoneAnchors,
    getFieldClimbableKind,
    FIELD_LAYOUT_STYLES,
    getFieldLayoutStyle,
    VERTICAL_FIELD_GEOMETRY,
    getMapGeometrySeed,
    getVerticalFieldGeometry,
    PRIORITY_FIELD_LAYOUT_IDS,
    DUNGEON_ARENA_SKELETONS,
    getDungeonArenaSkeleton
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapLayouts = Object.assign({}, modules.mapLayouts || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
