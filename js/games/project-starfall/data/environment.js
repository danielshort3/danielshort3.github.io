(function initProjectStarfallDataEnvironment(global) {
  'use strict';

  const DEFAULT_ASSET_ROOT = 'img/project-starfall';

  function createEnvironmentData(options) {
    const settings = options || {};
    const ASSET_ROOT = settings.ASSET_ROOT || DEFAULT_ASSET_ROOT;

    const ENVIRONMENT_THEME_IDS = Object.freeze([
      'starfall-crossing',
      'rustcoil-outpost',
      'cinder-refuge',
      'frostfen-camp',
      'stormbreak-haven',
      'astral-observatory',
      'greenroot-meadow',
      'thornpath-thicket',
      'bramble-depths',
      'rustcoil-ruins',
      'gearworks-vault',
      'cinder-hollow',
      'emberjaw-lair',
      'bandit-ridge-camp',
      'oreback-quarry',
      'ashglass-pass',
      'frostfen-outskirts',
      'glacier-spine',
      'rimewarden-sanctum',
      'stormbreak-cliffs',
      'astral-archive',
      'eclipse-frontier',
      'endless-rift',
      'brambleking-court',
      'titan-foundry',
      'deepcore-core',
      'emberjaw-furnace',
      'rimewarden-vault',
      'stormbreak-aerie',
      'astral-stacks',
      'eclipse-throne',
      'guardian-trial',
      'berserker-trial',
      'duelist-trial',
      'fire-mage-trial',
      'rune-mage-trial',
      'storm-mage-trial',
      'sniper-trial',
      'trapper-trial',
      'beast-archer-trial'
    ]);

    const ENVIRONMENT_ASSET_SOURCE_IDS = Object.freeze({
      'starfall-crossing': Object.freeze({
        terrain: 'astral-observatory',
        props: 'rustcoil-outpost',
        ramps: 'astral-observatory'
      })
    });

    function environmentAssetMap(folder) {
      return Object.freeze(ENVIRONMENT_THEME_IDS.reduce((assets, id) => {
        const isProps = folder === 'props';
        const isRamps = folder === 'ramps';
        const sourceId = ENVIRONMENT_ASSET_SOURCE_IDS[id] && ENVIRONMENT_ASSET_SOURCE_IDS[id][folder] || id;
        assets[id] = Object.freeze({
          path: `${ASSET_ROOT}/environment/${folder}/${sourceId}.png`,
          cellSize: isRamps ? 128 : 64,
          columns: isProps ? 6 : isRamps ? 4 : 8,
          schema: isRamps ? 'ramps-v1' : folder === 'terrain' ? 'modular-v2' : 'props-v1'
        });
        return assets;
      }, {}));
    }

    const ENVIRONMENT_ASSETS = Object.freeze({
      terrain: environmentAssetMap('terrain'),
      props: environmentAssetMap('props'),
      ramps: environmentAssetMap('ramps')
    });

    const ENVIRONMENT_STRUCTURE_ASSETS = Object.freeze({
      townLandmarks: Object.freeze({
        path: `${ASSET_ROOT}/environment/structures/town-landmarks.png`,
        cellSize: 256,
        columns: 4
      })
    });

    const ENVIRONMENT_TERRAIN_CELLS = Object.freeze({
      groundLeft: 0,
      groundMid: Object.freeze([1, 2]),
      groundRight: 3,
      platformLeft: 4,
      platformMid: Object.freeze([5, 6]),
      platformRight: 7,
      body: Object.freeze([8, 9, 10, 11]),
      bodyDeep: Object.freeze([12, 13, 14, 15]),
      underside: Object.freeze([16, 17, 18, 19]),
      left: 20,
      right: 21,
      cap: 22,
      detail: Object.freeze([23, 24, 25, 26]),
      undersideLong: Object.freeze([27, 28, 29, 30]),
      shadow: 31,
      top: Object.freeze([4, 5, 6, 7]),
      topAlt: Object.freeze([1, 2])
    });

    const ENVIRONMENT_PROP_CELLS = Object.freeze({
      grass: 0,
      bush: 1,
      tree: 2,
      rock: 3,
      flower: 4,
      small: 5,
      tall: 6,
      crate: 7,
      crystal: 8,
      vine: 9,
      sign: 10,
      glow: 11
    });

    const ENVIRONMENT_STRUCTURE_CELLS = Object.freeze({
      starfallGuildHall: 0,
      rustcoilWorkshop: 1,
      cinderForge: 2,
      frostfenLodge: 3,
      stormbreakGate: 4,
      astralObservatory: 5,
      marketAwning: 6,
      lanternArch: 7,
      fracturedObservatoryCore: 8,
      expeditionDepot: 9,
      lensWorkshop: 10,
      frontierGate: 11
    });

    const ENVIRONMENT_REAR_PROP_KINDS = Object.freeze(['tree', 'tall', 'vine', 'crystal', 'sign']);
    const ENVIRONMENT_FRONT_PROP_KINDS = Object.freeze(['grass', 'bush', 'flower', 'rock', 'small', 'crate', 'glow']);
    const ENVIRONMENT_UPPER_FRONT_PROP_KINDS = Object.freeze(['grass', 'flower', 'rock', 'glow']);

    const ENVIRONMENT_READABILITY_DEFAULTS = Object.freeze({
      maxFootOverlapPx: 0,
      combatClearancePx: 72,
      groundOnlyTallProps: true,
      upperPlatformPropScale: 0.6,
      rearDensityScale: 0.72,
      frontDensityScale: 0.26,
      maxUpperPropHeight: 20,
      maxFrontPropHeight: 32
    });

    const ENVIRONMENT_TERRAIN_STYLE_DEFAULTS = Object.freeze({
      topHeight: 20,
      groundTopHeight: 24,
      platformBodyDepth: 30,
      groundBodyDepth: 64,
      overhang: 8,
      groundOverhang: 0,
      edgeWidth: 36,
      undersideHeight: 14,
      undersideJitter: 8,
      detailDensity: 0.16,
      bodyAlpha: 0.94
    });

    const TERRAIN_STYLE_TOWN = Object.freeze({ topHeight: 18, groundTopHeight: 20, platformBodyDepth: 24, groundBodyDepth: 52, overhang: 6, undersideHeight: 10, undersideJitter: 3, detailDensity: 0.08, bodyAlpha: 0.9 });
    const TERRAIN_STYLE_FOREST = Object.freeze({ topHeight: 20, platformBodyDepth: 32, groundBodyDepth: 68, undersideJitter: 8, detailDensity: 0.18, bodyAlpha: 0.94 });
    const TERRAIN_STYLE_BANDIT = Object.freeze({ topHeight: 16, platformBodyDepth: 24, groundBodyDepth: 58, overhang: 6, undersideHeight: 10, undersideJitter: 2, detailDensity: 0.06, bodyAlpha: 0.88 });
    const TERRAIN_STYLE_RUST = Object.freeze({ topHeight: 18, platformBodyDepth: 28, groundBodyDepth: 62, overhang: 7, undersideHeight: 12, undersideJitter: 4, detailDensity: 0.1, bodyAlpha: 0.92 });
    const TERRAIN_STYLE_CINDER = Object.freeze({ topHeight: 18, platformBodyDepth: 28, groundBodyDepth: 60, overhang: 7, undersideHeight: 0, undersideJitter: 0, detailDensity: 0.1, bodyAlpha: 0.9 });
    const TERRAIN_STYLE_FROST = Object.freeze({ topHeight: 17, groundTopHeight: 22, platformBodyDepth: 24, groundBodyDepth: 58, overhang: 7, undersideHeight: 11, undersideJitter: 4, detailDensity: 0.1, bodyAlpha: 0.86 });
    const TERRAIN_STYLE_STORM = Object.freeze({ topHeight: 18, platformBodyDepth: 26, groundBodyDepth: 60, overhang: 7, undersideHeight: 12, undersideJitter: 5, detailDensity: 0.1, bodyAlpha: 0.9 });
    const TERRAIN_STYLE_ASTRAL = Object.freeze({ topHeight: 16, platformBodyDepth: 24, groundBodyDepth: 56, overhang: 6, undersideHeight: 10, undersideJitter: 3, detailDensity: 0.08, bodyAlpha: 0.84 });
    const ECLIPSE_OBSERVATORY_DECK_TREATMENT_ID = 'totality-observatory';

    function environmentProfile(config) {
      return Object.freeze(Object.assign({}, config, {
        visibility: Object.freeze(Object.assign({}, ENVIRONMENT_READABILITY_DEFAULTS, config.visibility || {})),
        terrainStyle: Object.freeze(Object.assign({}, ENVIRONMENT_TERRAIN_STYLE_DEFAULTS, config.terrainStyle || {}))
      }));
    }

    const MAP_ENVIRONMENT_PROFILES = Object.freeze({
      starfallCrossing: environmentProfile({
        terrain: 'starfall-crossing',
        props: 'starfall-crossing',
        ramps: 'starfall-crossing',
        tint: '#536777',
        density: 0.36,
        propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'sign', 'glow'],
        terrainStyle: Object.assign({}, TERRAIN_STYLE_ASTRAL, TERRAIN_STYLE_RUST, { bodyAlpha: 0.9 })
      }),
      rustcoilOutpost: environmentProfile({ terrain: 'rustcoil-outpost', props: 'rustcoil-outpost', density: 0.46, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_RUST) }),
      cinderRefuge: environmentProfile({ terrain: 'cinder-refuge', props: 'cinder-refuge', density: 0.42, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow', 'crate'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_CINDER) }),
      frostfenCamp: environmentProfile({ terrain: 'frostfen-camp', props: 'frostfen-camp', density: 0.46, propKinds: ['grass', 'rock', 'crystal', 'small', 'tall', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_FROST) }),
      stormbreakHaven: environmentProfile({ terrain: 'stormbreak-haven', props: 'stormbreak-haven', density: 0.44, propKinds: ['grass', 'bush', 'rock', 'crystal', 'small', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_STORM) }),
      astralObservatory: environmentProfile({ terrain: 'astral-observatory', props: 'astral-observatory', density: 0.42, propKinds: ['crystal', 'tall', 'sign', 'glow', 'rock', 'small'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_ASTRAL) }),
      greenrootMeadow: environmentProfile({
        terrain: 'greenroot-meadow',
        props: 'greenroot-meadow',
        ramps: 'greenroot-meadow',
        tint: '#66788b',
        density: 0.4,
        propKinds: ['rock', 'crystal', 'glow', 'small', 'sign'],
        terrainStyle: Object.assign({}, TERRAIN_STYLE_ASTRAL, { topHeight: 18, groundBodyDepth: 64, bodyAlpha: 0.92 })
      }),
      thornpathThicket: environmentProfile({
        terrain: 'thornpath-thicket',
        props: 'thornpath-thicket',
        ramps: 'thornpath-thicket',
        tint: '#526b68',
        density: 0.5,
        propKinds: ['tree', 'vine', 'rock', 'crystal', 'glow', 'sign'],
        terrainStyle: Object.assign({}, TERRAIN_STYLE_FOREST, TERRAIN_STYLE_ASTRAL, { groundBodyDepth: 64, bodyAlpha: 0.9 })
      }),
      brambleDepths: environmentProfile({ terrain: 'bramble-depths', props: 'bramble-depths', density: 0.58, propKinds: ['bush', 'tree', 'vine', 'flower', 'rock', 'crystal'], terrainStyle: TERRAIN_STYLE_FOREST }),
      rustcoilRuins: environmentProfile({ terrain: 'rustcoil-ruins', props: 'rustcoil-ruins', density: 0.52, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'sign'], terrainStyle: TERRAIN_STYLE_RUST }),
      gearworksVault: environmentProfile({ terrain: 'gearworks-vault', props: 'gearworks-vault', density: 0.5, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_RUST }),
      cinderHollow: environmentProfile({ terrain: 'cinder-hollow', props: 'cinder-hollow', density: 0.48, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      emberjawLair: environmentProfile({ terrain: 'emberjaw-lair', props: 'emberjaw-lair', density: 0.46, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      banditRidgeCamp: environmentProfile({ terrain: 'bandit-ridge-camp', props: 'bandit-ridge-camp', density: 0.56, propKinds: ['grass', 'bush', 'tree', 'crate', 'sign', 'small'], terrainStyle: TERRAIN_STYLE_BANDIT }),
      banditAnimationLab: environmentProfile({ terrain: 'bandit-ridge-camp', props: 'bandit-ridge-camp', density: 0.2, propKinds: ['crate', 'sign', 'small'], terrainStyle: TERRAIN_STYLE_BANDIT }),
      orebackQuarry: environmentProfile({ terrain: 'oreback-quarry', props: 'oreback-quarry', density: 0.54, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_RUST }),
      ashglassPass: environmentProfile({ terrain: 'ashglass-pass', props: 'ashglass-pass', density: 0.48, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      frostfenOutskirts: environmentProfile({ terrain: 'frostfen-outskirts', props: 'frostfen-outskirts', density: 0.54, propKinds: ['grass', 'rock', 'crystal', 'small', 'tall', 'glow'], terrainStyle: TERRAIN_STYLE_FROST }),
      glacierSpine: environmentProfile({ terrain: 'glacier-spine', props: 'glacier-spine', density: 0.5, propKinds: ['rock', 'crystal', 'tall', 'small', 'glow'], terrainStyle: TERRAIN_STYLE_FROST }),
      rimewardenSanctum: environmentProfile({ terrain: 'rimewarden-sanctum', props: 'rimewarden-sanctum', density: 0.44, propKinds: ['crystal', 'tall', 'rock', 'glow', 'small'], terrainStyle: TERRAIN_STYLE_FROST }),
      stormbreakCliffs: environmentProfile({ terrain: 'stormbreak-cliffs', props: 'stormbreak-cliffs', density: 0.52, propKinds: ['grass', 'bush', 'rock', 'crystal', 'flower', 'tall'], terrainStyle: TERRAIN_STYLE_STORM }),
      astralArchive: environmentProfile({ terrain: 'astral-archive', props: 'astral-archive', density: 0.48, propKinds: ['crystal', 'tall', 'sign', 'glow', 'rock'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
      eclipseFrontier: environmentProfile({ terrain: 'eclipse-frontier', props: 'eclipse-frontier', density: 0.46, propKinds: ['grass', 'crystal', 'tall', 'glow', 'rock'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
      endlessRift: environmentProfile({ terrain: 'endless-rift', props: 'endless-rift', density: 0.48, propKinds: ['crystal', 'tall', 'glow', 'rock', 'flower'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
      bramblekingCourt: environmentProfile({ terrain: 'brambleking-court', props: 'brambleking-court', density: 0.54, propKinds: ['bush', 'tree', 'vine', 'flower', 'rock', 'crystal'], terrainStyle: TERRAIN_STYLE_FOREST }),
      titanFoundry: environmentProfile({ terrain: 'titan-foundry', props: 'titan-foundry', density: 0.4, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_RUST }),
      deepcoreCore: environmentProfile({ terrain: 'deepcore-core', props: 'deepcore-core', density: 0.42, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_RUST }),
      emberjawFurnace: environmentProfile({ terrain: 'emberjaw-furnace', props: 'emberjaw-furnace', density: 0.44, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      rimewardenVault: environmentProfile({ terrain: 'rimewarden-vault', props: 'rimewarden-vault', density: 0.44, propKinds: ['crystal', 'tall', 'rock', 'glow', 'small'], terrainStyle: TERRAIN_STYLE_FROST }),
      stormbreakAerie: environmentProfile({ terrain: 'stormbreak-aerie', props: 'stormbreak-aerie', density: 0.48, propKinds: ['rock', 'crystal', 'tall', 'small', 'glow'], terrainStyle: TERRAIN_STYLE_STORM }),
      astralStacks: environmentProfile({ terrain: 'astral-stacks', props: 'astral-stacks', density: 0.4, propKinds: ['crystal', 'tall', 'sign', 'glow', 'rock'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
      eclipseThrone: environmentProfile({
        terrain: 'eclipse-throne',
        props: 'eclipse-throne',
        ramps: 'eclipse-throne',
        platformTreatment: ECLIPSE_OBSERVATORY_DECK_TREATMENT_ID,
        density: 0.24,
        propKinds: ['crystal', 'glow', 'rock', 'sign', 'small'],
        terrainStyle: Object.assign({}, TERRAIN_STYLE_ASTRAL, {
          topHeight: 14,
          groundTopHeight: 18,
          platformBodyDepth: 24,
          groundBodyDepth: 48,
          overhang: 5,
          undersideHeight: 8,
          detailDensity: 0.04,
          bodyAlpha: 0.68
        })
      }),
      guardian_trial: environmentProfile({ terrain: 'guardian-trial', props: 'guardian-trial', density: 0.42, propKinds: ['rock', 'small', 'tall', 'crate', 'sign', 'glow'], terrainStyle: TERRAIN_STYLE_RUST }),
      berserker_trial: environmentProfile({ terrain: 'berserker-trial', props: 'berserker-trial', density: 0.44, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      duelist_trial: environmentProfile({ terrain: 'duelist-trial', props: 'duelist-trial', density: 0.46, propKinds: ['grass', 'bush', 'crate', 'sign', 'small', 'rock'], terrainStyle: TERRAIN_STYLE_BANDIT }),
      fire_mage_trial: environmentProfile({ terrain: 'fire-mage-trial', props: 'fire-mage-trial', density: 0.38, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow'], terrainStyle: TERRAIN_STYLE_CINDER }),
      rune_mage_trial: environmentProfile({ terrain: 'rune-mage-trial', props: 'rune-mage-trial', density: 0.4, propKinds: ['crystal', 'tall', 'sign', 'glow', 'rock'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
      storm_mage_trial: environmentProfile({ terrain: 'storm-mage-trial', props: 'storm-mage-trial', density: 0.4, propKinds: ['rock', 'crystal', 'tall', 'small', 'glow'], terrainStyle: TERRAIN_STYLE_STORM }),
      sniper_trial: environmentProfile({ terrain: 'sniper-trial', props: 'sniper-trial', density: 0.42, propKinds: ['grass', 'bush', 'rock', 'crate', 'sign', 'small'], terrainStyle: TERRAIN_STYLE_BANDIT }),
      trapper_trial: environmentProfile({ terrain: 'trapper-trial', props: 'trapper-trial', density: 0.48, propKinds: ['grass', 'bush', 'tree', 'vine', 'rock', 'small'], terrainStyle: TERRAIN_STYLE_FOREST }),
      beast_archer_trial: environmentProfile({ terrain: 'beast-archer-trial', props: 'beast-archer-trial', density: 0.46, propKinds: ['grass', 'bush', 'tree', 'rock', 'vine', 'sign'], terrainStyle: TERRAIN_STYLE_FOREST })
    });

    return Object.freeze({
      ENVIRONMENT_THEME_IDS,
      ENVIRONMENT_ASSET_SOURCE_IDS,
      ENVIRONMENT_ASSETS,
      ENVIRONMENT_STRUCTURE_ASSETS,
      ENVIRONMENT_TERRAIN_CELLS,
      ENVIRONMENT_PROP_CELLS,
      ENVIRONMENT_STRUCTURE_CELLS,
      ENVIRONMENT_REAR_PROP_KINDS,
      ENVIRONMENT_FRONT_PROP_KINDS,
      ENVIRONMENT_UPPER_FRONT_PROP_KINDS,
      ECLIPSE_OBSERVATORY_DECK_TREATMENT_ID,
      ENVIRONMENT_READABILITY_DEFAULTS,
      ENVIRONMENT_TERRAIN_STYLE_DEFAULTS,
      MAP_ENVIRONMENT_PROFILES
    });
  }

  const defaultEnvironmentData = createEnvironmentData();

  const api = Object.assign({
    DEFAULT_ASSET_ROOT,
    createEnvironmentData
  }, defaultEnvironmentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.environment = Object.assign({}, modules.environment || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
