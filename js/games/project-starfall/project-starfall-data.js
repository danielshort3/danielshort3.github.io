(function initProjectStarfallData(global) {
  'use strict';

  const ROLE_TAGS = Object.freeze([
    'Mobbing',
    'Bossing',
    'Hybrid',
    'Control',
    'Support',
    'Party'
  ]);

  const SAVE_KEY = 'projectStarfallPrototypeSave.v1';
  const CHARACTER_ROSTER_KEY = 'projectStarfallCharacterRoster.v1';
  const CHARACTER_SLOT_COUNT = 8;
  const ASSET_ROOT = 'img/project-starfall';
  const BASE_SKILL_ICON_ROOT = `${ASSET_ROOT}/skills/base`;
  const ADVANCED_SKILL_ICON_ROOT = `${ASSET_ROOT}/skills/advanced`;
  const CARD_ICON_ROOT = `${ASSET_ROOT}/cards/icons`;
  const GENERIC_PLAYER_ASSET = `${ASSET_ROOT}/characters/generic-player.png`;
  const EQUIPMENT_VISUAL_ROOT = `${ASSET_ROOT}/equipment-layers`;
  const CHARACTER_SLOT_PEDESTAL_ASSET = `${ASSET_ROOT}/ui/character-slot-pedestal.png`;
  const LEVEL_CAP = null;
  const SPECIALIZATION_LEVEL = 60;
  const ROSTER_TRAIT_SLOTS = 2;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  const CLASS_FILE_IDS = Object.freeze({
    fighter: 'fighter',
    mage: 'mage',
    archer: 'archer',
    guardian: 'guardian',
    berserker: 'berserker',
    duelist: 'duelist',
    fireMage: 'fire-mage',
    runeMage: 'rune-mage',
    stormMage: 'storm-mage',
    sniper: 'sniper',
    trapper: 'trapper',
    beastArcher: 'beast-archer'
  });

  const CLASS_ASSETS = Object.freeze(Object.keys(CLASS_FILE_IDS).reduce((assets, classId) => {
    assets[classId] = `${ASSET_ROOT}/characters/${CLASS_FILE_IDS[classId]}.png`;
    return assets;
  }, {}));

  const CHARACTER_LOOKS = Object.freeze([
    Object.freeze({ id: 'sunlit', name: 'Sunlit', skin: '#d99a6c', hair: '#3b2725', shirt: '#526f86', shirtLight: '#8caabd', pants: '#33495b', pantsDark: '#24313f', boot: '#2a1f21' }),
    Object.freeze({ id: 'ember', name: 'Ember', skin: '#c98762', hair: '#5a2b22', shirt: '#8b3542', shirtLight: '#e0614f', pants: '#3b2d38', pantsDark: '#271f2c', boot: '#241a1f' }),
    Object.freeze({ id: 'skyline', name: 'Skyline', skin: '#e0aa7a', hair: '#26354d', shirt: '#376c9f', shirtLight: '#7bdff2', pants: '#27344c', pantsDark: '#1b2436', boot: '#18202e' }),
    Object.freeze({ id: 'greenroot', name: 'Greenroot', skin: '#b97858', hair: '#2f3926', shirt: '#4e7d52', shirtLight: '#8ec878', pants: '#2f463a', pantsDark: '#24362d', boot: '#1f261c' }),
    Object.freeze({ id: 'violet', name: 'Violet', skin: '#d6a16f', hair: '#3b2448', shirt: '#6251a1', shirtLight: '#c794ff', pants: '#2f2d55', pantsDark: '#20203d', boot: '#20182d' }),
    Object.freeze({ id: 'silver', name: 'Silver', skin: '#c99a78', hair: '#5e6570', shirt: '#586575', shirtLight: '#b7c3ca', pants: '#2c3540', pantsDark: '#202832', boot: '#1d2228' })
  ]);

  const ENEMY_ASSETS = Object.freeze({
    slimelet: `${ASSET_ROOT}/enemies/slimelet.png`,
    dewSlime: `${ASSET_ROOT}/enemies/dew-slime.png`,
    mossback: `${ASSET_ROOT}/enemies/mossback.png`,
    thornSprout: `${ASSET_ROOT}/enemies/thorn-sprout.png`,
    vineSnapper: `${ASSET_ROOT}/enemies/vine-snapper.png`,
    bristleBoar: `${ASSET_ROOT}/enemies/bristle-boar.png`,
    briarStag: `${ASSET_ROOT}/enemies/briar-stag.png`,
    dustImp: `${ASSET_ROOT}/enemies/dust-imp.png`,
    clockbug: `${ASSET_ROOT}/enemies/clockbug.png`,
    rustRatchet: `${ASSET_ROOT}/enemies/rust-ratchet.png`,
    coilSentry: `${ASSET_ROOT}/enemies/coil-sentry.png`,
    scrapWarden: `${ASSET_ROOT}/enemies/scrap-warden.png`,
    emberWisp: `${ASSET_ROOT}/enemies/ember-wisp.png`,
    ashCrawler: `${ASSET_ROOT}/enemies/ash-crawler.png`,
    lavaTick: `${ASSET_ROOT}/enemies/lava-tick.png`,
    cinderSpitter: `${ASSET_ROOT}/enemies/cinder-spitter.png`,
    banditCutter: `${ASSET_ROOT}/enemies/bandit-cutter.png`,
    banditCutterDirect: `${ASSET_ROOT}/enemies/bandit-cutter-direct.png`,
    banditCutterReference: `${ASSET_ROOT}/enemies/bandit-cutter-reference.png`,
    banditCutterHybrid: `${ASSET_ROOT}/enemies/bandit-cutter-hybrid.png`,
    banditCutterPuppet: `${ASSET_ROOT}/enemies/bandit-cutter-puppet.png`,
    banditThrower: `${ASSET_ROOT}/enemies/bandit-thrower.png`,
    orebackBeetle: `${ASSET_ROOT}/enemies/oreback-beetle.png`,
    glowcapHealer: `${ASSET_ROOT}/enemies/glowcap-healer.png`,
    crackedMimic: `${ASSET_ROOT}/enemies/cracked-mimic.png`,
    brambleking: `${ASSET_ROOT}/enemies/brambleking.png`,
    clockworkTitan: `${ASSET_ROOT}/enemies/clockwork-titan.png`,
    quarryColossus: `${ASSET_ROOT}/enemies/quarry-colossus.png`,
    emberjawGolem: `${ASSET_ROOT}/enemies/emberjaw-golem.png`,
    frostlingScout: `${ASSET_ROOT}/enemies/frostling-scout.png`,
    shardling: `${ASSET_ROOT}/enemies/shardling.png`,
    rimebackBrute: `${ASSET_ROOT}/enemies/rimeback-brute.png`,
    glacierSentinel: `${ASSET_ROOT}/enemies/glacier-sentinel.png`,
    snowglareWisp: `${ASSET_ROOT}/enemies/snowglare-wisp.png`,
    icebloomOracle: `${ASSET_ROOT}/enemies/icebloom-oracle.png`,
    galeHarrier: `${ASSET_ROOT}/enemies/gale-harrier.png`,
    stormboundArcher: `${ASSET_ROOT}/enemies/stormbound-archer.png`,
    thunderRam: `${ASSET_ROOT}/enemies/thunder-ram.png`,
    cloudcallAcolyte: `${ASSET_ROOT}/enemies/cloudcall-acolyte.png`,
    indexScribe: `${ASSET_ROOT}/enemies/index-scribe.png`,
    lumenSentinel: `${ASSET_ROOT}/enemies/lumen-sentinel.png`,
    voidMote: `${ASSET_ROOT}/enemies/void-mote.png`,
    eclipseDuelist: `${ASSET_ROOT}/enemies/eclipse-duelist.png`,
    riftAberration: `${ASSET_ROOT}/enemies/rift-aberration.png`,
    rimewarden: `${ASSET_ROOT}/enemies/rimewarden.png`,
    stormbreakRoc: `${ASSET_ROOT}/enemies/stormbreak-roc.png`,
    astralArchivist: `${ASSET_ROOT}/enemies/astral-archivist.png`,
    eclipseSovereign: `${ASSET_ROOT}/enemies/eclipse-sovereign.png`
  });

  const MAP_ASSETS = Object.freeze({
    starfallCrossing: `${ASSET_ROOT}/maps/starfall-crossing.webp`,
    rustcoilOutpost: `${ASSET_ROOT}/maps/rustcoil-outpost.webp`,
    cinderRefuge: `${ASSET_ROOT}/maps/cinder-refuge.webp`,
    frostfenCamp: `${ASSET_ROOT}/maps/frostfen-camp.webp`,
    stormbreakHaven: `${ASSET_ROOT}/maps/stormbreak-haven.webp`,
    astralObservatory: `${ASSET_ROOT}/maps/astral-observatory.webp`,
    greenrootMeadow: `${ASSET_ROOT}/maps/greenroot-meadow.webp`,
    thornpathThicket: `${ASSET_ROOT}/maps/thornpath-thicket.webp`,
    brambleDepths: `${ASSET_ROOT}/maps/bramble-depths.webp`,
    rustcoilRuins: `${ASSET_ROOT}/maps/rustcoil-ruins.webp`,
    gearworksVault: `${ASSET_ROOT}/maps/gearworks-vault.webp`,
    cinderHollow: `${ASSET_ROOT}/maps/cinder-hollow.webp`,
    emberjawLair: `${ASSET_ROOT}/maps/emberjaw-lair.webp`,
    banditRidgeCamp: `${ASSET_ROOT}/maps/bandit-ridge-camp.webp`,
    banditAnimationLab: `${ASSET_ROOT}/maps/bandit-ridge-camp.webp`,
    orebackQuarry: `${ASSET_ROOT}/maps/oreback-quarry.webp`,
    ashglassPass: `${ASSET_ROOT}/maps/ashglass-pass.webp`,
    frostfenOutskirts: `${ASSET_ROOT}/maps/frostfen-outskirts.webp`,
    glacierSpine: `${ASSET_ROOT}/maps/glacier-spine.webp`,
    rimewardenSanctum: `${ASSET_ROOT}/maps/rimewarden-sanctum.webp`,
    stormbreakCliffs: `${ASSET_ROOT}/maps/stormbreak-cliffs.webp`,
    astralArchive: `${ASSET_ROOT}/maps/astral-archive.webp`,
    eclipseFrontier: `${ASSET_ROOT}/maps/eclipse-frontier.webp`,
    endlessRift: `${ASSET_ROOT}/maps/endless-rift.webp`,
    bramblekingCourt: `${ASSET_ROOT}/maps/brambleking-court.webp`,
    titanFoundry: `${ASSET_ROOT}/maps/titan-foundry.webp`,
    deepcoreCore: `${ASSET_ROOT}/maps/deepcore-core.webp`,
    emberjawFurnace: `${ASSET_ROOT}/maps/emberjaw-furnace.webp`,
    rimewardenVault: `${ASSET_ROOT}/maps/rimewarden-vault.webp`,
    stormbreakAerie: `${ASSET_ROOT}/maps/stormbreak-aerie.webp`,
    astralStacks: `${ASSET_ROOT}/maps/astral-stacks.webp`,
    eclipseThrone: `${ASSET_ROOT}/maps/eclipse-throne.webp`
  });

  const UI_ASSETS = Object.freeze({
    splashScreen: `${ASSET_ROOT}/ui/splash-screen.png`,
    startScreen: `${ASSET_ROOT}/ui/start-screen.png`,
    characterSelectScreen: `${ASSET_ROOT}/ui/character-select-screen.png`
  });

  const MENU_ICON_ASSETS = Object.freeze({
    character: `${ASSET_ROOT}/ui/menu-icons/character.png`,
    equipment: `${ASSET_ROOT}/ui/menu-icons/equipment.png`,
    partyPanel: `${ASSET_ROOT}/ui/menu-icons/party-panel.png`,
    inventory: `${ASSET_ROOT}/ui/menu-icons/inventory.png`,
    skills: `${ASSET_ROOT}/ui/menu-icons/skills.png`,
    quests: `${ASSET_ROOT}/ui/menu-icons/quests.png`,
    worldmap: `${ASSET_ROOT}/ui/menu-icons/worldmap.png`,
    monsters: `${ASSET_ROOT}/ui/menu-icons/monsters.png`,
    shop: `${ASSET_ROOT}/ui/menu-icons/shop.png`,
    upgrade: `${ASSET_ROOT}/ui/menu-icons/upgrade.png`,
    cashShop: `${ASSET_ROOT}/ui/menu-icons/cash-shop.png`,
    beta: `${ASSET_ROOT}/ui/menu-icons/beta.png`,
    guide: `${ASSET_ROOT}/ui/menu-icons/guide.png`,
    log: `${ASSET_ROOT}/ui/menu-icons/log.png`,
    settings: `${ASSET_ROOT}/ui/menu-icons/settings.png`,
    keybinds: `${ASSET_ROOT}/ui/menu-icons/keybinds.png`,
    admin: `${ASSET_ROOT}/ui/menu-icons/admin.png`,
    logout: `${ASSET_ROOT}/ui/menu-icons/logout.png`
  });

  const CLASS_TRIAL_ASSETS = Object.freeze({
    guardian_trial: `${ASSET_ROOT}/maps/trials/guardian-trial.webp`,
    berserker_trial: `${ASSET_ROOT}/maps/trials/berserker-trial.webp`,
    duelist_trial: `${ASSET_ROOT}/maps/trials/duelist-trial.webp`,
    fire_mage_trial: `${ASSET_ROOT}/maps/trials/fire-mage-trial.webp`,
    rune_mage_trial: `${ASSET_ROOT}/maps/trials/rune-mage-trial.webp`,
    storm_mage_trial: `${ASSET_ROOT}/maps/trials/storm-mage-trial.webp`,
    sniper_trial: `${ASSET_ROOT}/maps/trials/sniper-trial.webp`,
    trapper_trial: `${ASSET_ROOT}/maps/trials/trapper-trial.webp`,
    beast_archer_trial: `${ASSET_ROOT}/maps/trials/beast-archer-trial.webp`
  });

  const WORLD_MAP_ATLAS = Object.freeze({
    asset: `${ASSET_ROOT}/world-map/starfall-atlas.webp`,
    width: 1920,
    height: 1080,
    style: 'painterly-atlas'
  });

  const STATION_ASSETS = Object.freeze({
    shop: `${ASSET_ROOT}/stations/shop.png`,
    storage: `${ASSET_ROOT}/stations/storage.png`,
    slots: `${ASSET_ROOT}/stations/slots.png`,
    upgrade: `${ASSET_ROOT}/stations/upgrade.png`,
    class: `${ASSET_ROOT}/stations/class.png`,
    plinko: `${ASSET_ROOT}/stations/slots.png`
  });

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

  function environmentAssetMap(folder) {
    return Object.freeze(ENVIRONMENT_THEME_IDS.reduce((assets, id) => {
      assets[id] = Object.freeze({
        path: `${ASSET_ROOT}/environment/${folder}/${id}.png`,
        cellSize: 64,
        columns: folder === 'props' ? 6 : 8,
        schema: folder === 'terrain' ? 'modular-v2' : 'props-v1'
      });
      return assets;
    }, {}));
  }

  const ENVIRONMENT_ASSETS = Object.freeze({
    terrain: environmentAssetMap('terrain'),
    props: environmentAssetMap('props')
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
    lanternArch: 7
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

  function environmentProfile(config) {
    return Object.freeze(Object.assign({}, config, {
      visibility: Object.freeze(Object.assign({}, ENVIRONMENT_READABILITY_DEFAULTS, config.visibility || {})),
      terrainStyle: Object.freeze(Object.assign({}, ENVIRONMENT_TERRAIN_STYLE_DEFAULTS, config.terrainStyle || {}))
    }));
  }

  const MAP_ENVIRONMENT_PROFILES = Object.freeze({
    starfallCrossing: environmentProfile({ terrain: 'starfall-crossing', props: 'starfall-crossing', density: 0.48, propKinds: ['grass', 'bush', 'flower', 'small', 'crate', 'sign'], terrainStyle: TERRAIN_STYLE_TOWN }),
    rustcoilOutpost: environmentProfile({ terrain: 'rustcoil-outpost', props: 'rustcoil-outpost', density: 0.46, propKinds: ['rock', 'small', 'tall', 'crate', 'crystal', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_RUST) }),
    cinderRefuge: environmentProfile({ terrain: 'cinder-refuge', props: 'cinder-refuge', density: 0.42, propKinds: ['rock', 'small', 'tall', 'crystal', 'glow', 'crate'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_CINDER) }),
    frostfenCamp: environmentProfile({ terrain: 'frostfen-camp', props: 'frostfen-camp', density: 0.46, propKinds: ['grass', 'rock', 'crystal', 'small', 'tall', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_FROST) }),
    stormbreakHaven: environmentProfile({ terrain: 'stormbreak-haven', props: 'stormbreak-haven', density: 0.44, propKinds: ['grass', 'bush', 'rock', 'crystal', 'small', 'sign'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_STORM) }),
    astralObservatory: environmentProfile({ terrain: 'astral-observatory', props: 'astral-observatory', density: 0.42, propKinds: ['crystal', 'tall', 'sign', 'glow', 'rock', 'small'], terrainStyle: Object.assign({}, TERRAIN_STYLE_TOWN, TERRAIN_STYLE_ASTRAL) }),
    greenrootMeadow: environmentProfile({ terrain: 'greenroot-meadow', props: 'greenroot-meadow', density: 0.62, propKinds: ['grass', 'bush', 'tree', 'flower', 'rock', 'vine'], terrainStyle: TERRAIN_STYLE_FOREST }),
    thornpathThicket: environmentProfile({ terrain: 'thornpath-thicket', props: 'thornpath-thicket', density: 0.64, propKinds: ['grass', 'bush', 'tree', 'flower', 'vine', 'rock'], terrainStyle: TERRAIN_STYLE_FOREST }),
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
    eclipseThrone: environmentProfile({ terrain: 'eclipse-throne', props: 'eclipse-throne', density: 0.36, propKinds: ['crystal', 'tall', 'glow', 'rock', 'sign'], terrainStyle: TERRAIN_STYLE_ASTRAL }),
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

  const ITEM_ICON_FRAME_SIZE = 64;
  const ITEM_ICON_SHEET_ROOT = `${ASSET_ROOT}/items/sheets`;

  function itemSheetAssets(sheetFile, columns, itemIds) {
    const rows = Math.ceil((itemIds || []).length / columns);
    const sheetWidth = columns * ITEM_ICON_FRAME_SIZE;
    const sheetHeight = rows * ITEM_ICON_FRAME_SIZE;
    return Object.freeze((itemIds || []).reduce((assets, id, index) => {
      const col = index % columns;
      const row = Math.floor(index / columns);
      assets[id] = `${ITEM_ICON_SHEET_ROOT}/${sheetFile}#frame=${col * ITEM_ICON_FRAME_SIZE},${row * ITEM_ICON_FRAME_SIZE},${ITEM_ICON_FRAME_SIZE},${ITEM_ICON_FRAME_SIZE},${sheetWidth},${sheetHeight}`;
      return assets;
    }, {}));
  }

  function materialAssetId(materialId) {
    return String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase();
  }

  function materialItem(materialId, name, icon, rarity, options) {
    const settings = options || {};
    return Object.freeze({
      id: materialId,
      materialId,
      assetId: settings.assetId || materialAssetId(materialId),
      name,
      icon,
      rarity: rarity || 'Common',
      starterQuantity: Math.max(0, Math.floor(Number(settings.starterQuantity || 0) || 0)),
      primaryDrop: settings.primaryDrop !== false,
      genericDrop: !!settings.genericDrop,
      dropLabels: Object.freeze((settings.dropLabels || [name]).slice())
    });
  }

  const MATERIAL_ITEMS = Object.freeze([
    materialItem('upgradeDust', 'Upgrade Dust', 'UD', 'Uncommon', { starterQuantity: 6, primaryDrop: false }),
    materialItem('upgradeCatalyst', 'Upgrade Catalyst', 'UC', 'Rare', { primaryDrop: false, dropLabels: ['Upgrade Catalyst', 'Rare catalyst'] }),
    materialItem('wardingScroll', 'Warding Scroll', 'WS', 'Rare', { primaryDrop: false }),
    materialItem('refinementCore', 'Refinement Core', 'RC', 'Epic', { primaryDrop: false }),
    materialItem('cubeFragment', 'Prism Shard', 'PS', 'Rare', { primaryDrop: false, dropLabels: ['Prism Shards', 'Prism Shard'] }),
    materialItem('gelDrop', 'Gel Drop', 'GD', 'Common', { genericDrop: true }),
    materialItem('oreChunks', 'Ore Chunks', 'OR', 'Common', { genericDrop: true }),
    materialItem('dewBead', 'Dew Bead', 'DEW', 'Common'),
    materialItem('mossHide', 'Moss Hide', 'MOS', 'Common'),
    materialItem('thornFiber', 'Thorn Fiber', 'THN', 'Common'),
    materialItem('vineFiber', 'Vine Fiber', 'VIN', 'Common'),
    materialItem('bristleHide', 'Bristle Hide', 'BRI', 'Common'),
    materialItem('briarAntler', 'Briar Antler', 'ANT', 'Uncommon'),
    materialItem('dustClaw', 'Dust Claw', 'CLW', 'Common'),
    materialItem('clockworkScrap', 'Clockwork Scrap', 'CLK', 'Common'),
    materialItem('chargedCoil', 'Charged Coil', 'COI', 'Uncommon'),
    materialItem('scrapPlate', 'Scrap Plate', 'SCP', 'Uncommon'),
    materialItem('emberDust', 'Ember Dust', 'EMB', 'Common'),
    materialItem('ashCarapace', 'Ash Carapace', 'ASH', 'Common'),
    materialItem('moltenFang', 'Molten Fang', 'FNG', 'Common'),
    materialItem('cinderGland', 'Cinder Gland', 'CIN', 'Uncommon'),
    materialItem('banditCloth', 'Bandit Cloth', 'BAN', 'Common'),
    materialItem('throwingKnifeScrap', 'Throwing Knife Scrap', 'TKS', 'Common'),
    materialItem('glowSpores', 'Glow Spores', 'GLW', 'Common'),
    materialItem('brambleCrown', 'Bramble Crown', 'BRC', 'Rare'),
    materialItem('titanCore', 'Titan Core', 'TIT', 'Rare'),
    materialItem('colossusOre', 'Colossus Ore', 'COL', 'Rare'),
    materialItem('emberjawBadge', 'Emberjaw Badge', 'EJB', 'Rare'),
    materialItem('rimeShard', 'Rime Shard', 'RIM', 'Common'),
    materialItem('frozenHide', 'Frozen Hide', 'FRZ', 'Common'),
    materialItem('glacierCore', 'Glacier Core', 'GLC', 'Uncommon'),
    materialItem('snowglareDust', 'Snowglare Dust', 'SNO', 'Common'),
    materialItem('icebloomPetal', 'Icebloom Petal', 'ICE', 'Common'),
    materialItem('galeFeather', 'Gale Feather', 'GAL', 'Common'),
    materialItem('stormFletching', 'Storm Fletching', 'FLT', 'Common'),
    materialItem('thunderHorn', 'Thunder Horn', 'THU', 'Uncommon'),
    materialItem('cloudSilk', 'Cloud Silk', 'CLD', 'Common'),
    materialItem('runicPage', 'Runic Page', 'RUN', 'Common'),
    materialItem('lumenPlate', 'Lumen Plate', 'LUM', 'Uncommon'),
    materialItem('voidDust', 'Void Dust', 'VOI', 'Common'),
    materialItem('eclipseSilk', 'Eclipse Silk', 'ECL', 'Uncommon'),
    materialItem('riftSplinter', 'Rift Splinter', 'RFT', 'Rare'),
    materialItem('rimewardenSigil', 'Rimewarden Sigil', 'RWS', 'Rare'),
    materialItem('stormbreakPlume', 'Stormbreak Plume', 'SBP', 'Rare'),
    materialItem('archivistIndex', 'Archivist Index', 'IDX', 'Rare'),
    materialItem('sovereignCorona', 'Sovereign Corona', 'COR', 'Rare')
  ]);

  const ITEM_ASSETS = Object.freeze(Object.assign({},
    itemSheetAssets('ai-items-consumables-materials-sheet.png', 5, [
      'coins',
      'town_return_scroll',
      'guard_tonic',
      'swiftstep_oil',
      'magnet_charm',
      'pet_whistle',
      'cube_fragment',
      'base_skill_manual',
      'advanced_skill_manual',
      'skill_reset_scroll',
      'admin_worldwright_console',
      'upgrade_dust',
      'upgrade_catalyst',
      'warding_scroll',
      'refinement_core',
      'gel_drop',
      'ore_chunks'
    ]),
    itemSheetAssets('ai-items-potion-tiers-sheet.png', 4, [
      'minor_health_potion',
      'standard_health_potion',
      'greater_health_potion',
      'superior_health_potion',
      'minor_resource_tonic',
      'standard_resource_tonic',
      'greater_resource_tonic',
      'superior_resource_tonic',
      'camp_ration',
      'field_ration',
      'expedition_ration',
      'hero_ration'
    ]),
    itemSheetAssets('ai-items-mob-materials-core-sheet.png', 5, [
      'dew_bead',
      'moss_hide',
      'thorn_fiber',
      'vine_fiber',
      'bristle_hide',
      'briar_antler',
      'dust_claw',
      'clockwork_scrap',
      'charged_coil',
      'scrap_plate',
      'ember_dust',
      'ash_carapace',
      'molten_fang',
      'cinder_gland',
      'bandit_cloth',
      'throwing_knife_scrap',
      'glow_spores',
      'bramble_crown',
      'titan_core',
      'colossus_ore'
    ]),
    itemSheetAssets('ai-items-mob-materials-late-sheet.png', 5, [
      'emberjaw_badge',
      'rime_shard',
      'frozen_hide',
      'glacier_core',
      'snowglare_dust',
      'icebloom_petal',
      'gale_feather',
      'storm_fletching',
      'thunder_horn',
      'cloud_silk',
      'runic_page',
      'lumen_plate',
      'void_dust',
      'eclipse_silk',
      'rift_splinter',
      'rimewarden_sigil',
      'stormbreak_plume',
      'archivist_index',
      'sovereign_corona'
    ]),
    itemSheetAssets('ai-items-coin-stacks-sheet.png', 4, [
      'coins_small',
      'coins_medium',
      'coins_large',
      'coins_huge'
    ]),
    itemSheetAssets('ai-items-shop-boss-forest-sheet.png', 5, [
      'training_sword',
      'training_wand',
      'training_bow',
      'copper_sword',
      'birch_wand',
      'simple_bow',
      'stitched_vest',
      'traveler_boots',
      'plain_ring',
      'iron_sword',
      'iron_axe',
      'apprentice_staff',
      'oak_longbow',
      'guardian_tower_shield',
      'berserker_war_grip',
      'ember_core',
      'rune_etched_focus',
      'deadeye_scope',
      'trap_kit',
      'thorncrown_greatsword',
      'thornroot_staff',
      'briarstring_longbow',
      'briar_crown',
      'barkplate_harness',
      'grasping_thorn_gloves'
    ]),
    itemSheetAssets('ai-items-world-drops-sheet.png', 5, [
      'adventurer_cutlass',
      'balanced_focus',
      'wanderer_charm',
      'fieldguard_helm',
      'trailwoven_gloves',
      'vanguard_blade',
      'bulwark_plate',
      'breaker_gauntlets',
      'sentinel_greaves',
      'starglass_staff',
      'runewoven_robes',
      'channeler_gloves',
      'aetherstep_boots',
      'ranger_recurve',
      'pathfinder_leathers',
      'deadeye_wraps',
      'windrunner_boots'
    ]),
    itemSheetAssets('ai-items-boss-core-storm-sheet.png', 5, [
      'rootstep_greaves',
      'emberjaw_cleaver',
      'magma_scepter',
      'cindercoil_bow',
      'ashen_jaw_helm',
      'furnaceplate',
      'lavaforged_gauntlets',
      'scorchtrail_boots',
      'gearcleaver',
      'chrono_staff',
      'ratchet_repeater',
      'titan_visor',
      'clockplate_harness',
      'gyro_gauntlets',
      'springstep_boots',
      'colossus_maul',
      'geode_scepter',
      'oreline_greatbow',
      'deepcore_helm',
      'bedrock_plate',
      'quarry_fists',
      'stonewake_boots',
      'stormtalon_saber',
      'cloudspine_rod',
      'skybreaker_bow'
    ]),
	    itemSheetAssets('ai-items-boss-astral-eclipse-sheet.png', 5, [
	      'rocfeather_mask',
	      'tempest_mantle',
	      'lightning_grip_gloves',
	      'gale_boots',
      'index_blade',
      'starbound_codex',
      'cometstring_bow',
      'archivist_crown',
      'astral_robes',
      'scribe_gloves',
      'orbit_boots',
      'eclipse_edge',
      'umbral_starstaff',
      'corona_longbow',
      'sovereign_crown',
      'eclipse_plate',
	      'penumbra_gloves',
	      'sunfall_boots'
	    ]),
    itemSheetAssets('ai-items-rate-coupons-sheet.png', 3, [
      'xp_coupon_1_2_1h',
      'xp_coupon_1_5_1h',
      'xp_coupon_2_0_1h',
      'drop_coupon_1_2_1h',
      'drop_coupon_1_5_1h',
      'drop_coupon_2_0_1h'
    ]),
			    {
			      stat_reset_scroll: `${ITEM_ICON_SHEET_ROOT}/ai-items-consumables-materials-sheet.png#frame=${4 * ITEM_ICON_FRAME_SIZE},${1 * ITEM_ICON_FRAME_SIZE},${ITEM_ICON_FRAME_SIZE},${ITEM_ICON_FRAME_SIZE},${5 * ITEM_ICON_FRAME_SIZE},${4 * ITEM_ICON_FRAME_SIZE}`
			    },
	    itemSheetAssets('ai-items-slot-prisms-plinko-sheet.png', 3, [
	      'equipment_slot_coupon',
	      'usable_slot_coupon',
	      'etc_slot_coupon',
	      'card_slot_coupon',
	      'potential_cube',
	      'preservation_cube',
	      'plinko_ball_basic',
	      'plinko_ball_polished',
	      'plinko_ball_meteor'
	    ])
		  ));

  const ITEM_RARITY_VISUALS = Object.freeze({
    Common: Object.freeze({ color: '#d8e5ec', glow: 7, alpha: 0.5, ring: 1.4 }),
    Uncommon: Object.freeze({ color: '#74d680', glow: 11, alpha: 0.7, ring: 1.8 }),
    Rare: Object.freeze({ color: '#68a9ff', glow: 15, alpha: 0.82, ring: 2.1 }),
    Epic: Object.freeze({ color: '#c794ff', glow: 19, alpha: 0.92, ring: 2.4, pulse: 0.16 }),
    Relic: Object.freeze({ color: '#ffbe55', glow: 23, alpha: 0.98, ring: 2.7, pulse: 0.2 })
  });

  const BASE_SKILL_ICONS = Object.freeze({
    fighter_heavy_strike: `${BASE_SKILL_ICON_ROOT}/fighter-heavy-strike.png`,
    fighter_dash_slash: `${BASE_SKILL_ICON_ROOT}/fighter-dash-slash.png`,
    fighter_guard: `${BASE_SKILL_ICON_ROOT}/fighter-guard.png`,
    fighter_ground_slam: `${BASE_SKILL_ICON_ROOT}/fighter-ground-slam.png`,
    fighter_power_break: `${BASE_SKILL_ICON_ROOT}/fighter-power-break.png`,
    fighter_momentum_burst: `${BASE_SKILL_ICON_ROOT}/fighter-momentum-burst.png`,
    fighter_damage_mastery: `${BASE_SKILL_ICON_ROOT}/fighter-damage-mastery.png`,
    mage_magic_bolt: `${BASE_SKILL_ICON_ROOT}/mage-magic-bolt.png`,
    mage_blink: `${BASE_SKILL_ICON_ROOT}/mage-blink.png`,
    mage_arcane_burst: `${BASE_SKILL_ICON_ROOT}/mage-arcane-burst.png`,
    mage_mana_shield: `${BASE_SKILL_ICON_ROOT}/mage-mana-shield.png`,
    mage_spell_mark: `${BASE_SKILL_ICON_ROOT}/mage-spell-mark.png`,
    mage_energy_release: `${BASE_SKILL_ICON_ROOT}/mage-energy-release.png`,
    mage_damage_mastery: `${BASE_SKILL_ICON_ROOT}/mage-damage-mastery.png`,
    archer_quick_shot: `${BASE_SKILL_ICON_ROOT}/archer-quick-shot.png`,
    archer_roll_shot: `${BASE_SKILL_ICON_ROOT}/archer-roll-shot.png`,
    archer_marked_shot: `${BASE_SKILL_ICON_ROOT}/archer-marked-shot.png`,
    archer_piercing_arrow: `${BASE_SKILL_ICON_ROOT}/archer-piercing-arrow.png`,
    archer_eagle_stance: `${BASE_SKILL_ICON_ROOT}/archer-eagle-stance.png`,
    archer_focused_volley: `${BASE_SKILL_ICON_ROOT}/archer-focused-volley.png`,
    archer_damage_mastery: `${BASE_SKILL_ICON_ROOT}/archer-damage-mastery.png`
  });

  const ADVANCED_SKILL_ICONS = Object.freeze({
    guardian_shield_bash: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-bash.png`,
    guardian_shield_dash: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-dash.png`,
    guardian_impact_guard: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-impact-guard.png`,
    guardian_oath_barrier: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-oath-barrier.png`,
    guardian_retaliation_wave: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-retaliation-wave.png`,
    guardian_hold_the_line: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-hold-the-line.png`,
    guardian_verdict: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-verdict.png`,
    guardian_shield_wall: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-wall.png`,
    guardian_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-damage-mastery.png`,
    berserker_blood_cleave: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-blood-cleave.png`,
    berserker_rage_surge: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-rage-surge.png`,
    berserker_reckless_leap: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-reckless-leap.png`,
    berserker_crimson_recovery: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-crimson-recovery.png`,
    berserker_pain_to_power: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-pain-to-power.png`,
    berserker_last_stand: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-last-stand.png`,
    berserker_war_cry: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-war-cry.png`,
    berserker_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-damage-mastery.png`,
    duelist_quick_cut: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-quick-cut.png`,
    duelist_flash_step: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-flash-step.png`,
    duelist_rallying_flourish: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-rallying-flourish.png`,
    duelist_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-damage-mastery.png`,
    fire_mage_fireball: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-fireball.png`,
    fire_mage_flame_trail: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-flame-trail.png`,
    fire_mage_burning_mark: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-burning-mark.png`,
    fire_mage_heat_vent: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-heat-vent.png`,
    fire_mage_wildfire: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-wildfire.png`,
    fire_mage_inferno_burst: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-inferno-burst.png`,
    fire_mage_ignition_aura: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-ignition-aura.png`,
    fire_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-damage-mastery.png`,
    rune_mage_rune_mark: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-mark.png`,
    rune_mage_rune_blink: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-blink.png`,
    rune_mage_ground_glyph: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-ground-glyph.png`,
    rune_mage_arcane_link: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-arcane-link.png`,
    rune_mage_rune_detonation: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-detonation.png`,
    rune_mage_mana_seal: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-mana-seal.png`,
    rune_mage_grand_inscription: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-grand-inscription.png`,
    rune_mage_rune_circle: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-circle.png`,
    rune_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-damage-mastery.png`,
    storm_mage_chain_bolt: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-chain-bolt.png`,
    storm_mage_static_shift: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-static-shift.png`,
    storm_mage_stormfront: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-stormfront.png`,
    storm_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-damage-mastery.png`,
    sniper_aimed_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-aimed-shot.png`,
    sniper_combat_roll: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-combat-roll.png`,
    sniper_weak_point_mark: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-weak-point-mark.png`,
    sniper_steady_breath: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-steady-breath.png`,
    sniper_pierce_armor: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-pierce-armor.png`,
    sniper_execution_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-execution-shot.png`,
    sniper_one_perfect_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-one-perfect-shot.png`,
    sniper_eagle_eye: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-eagle-eye.png`,
    sniper_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-damage-mastery.png`,
    trapper_snare_trap: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-snare-trap.png`,
    trapper_grapple_dash: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-grapple-dash.png`,
    trapper_spike_trap: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-spike-trap.png`,
    trapper_lure_shot: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-lure-shot.png`,
    trapper_tripwire: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-tripwire.png`,
    trapper_detonate: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-detonate.png`,
    trapper_kill_zone: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-kill-zone.png`,
    trapper_tactical_field: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-tactical-field.png`,
    trapper_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-damage-mastery.png`,
    beast_archer_companion_strike: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-companion-strike.png`,
    beast_archer_pounce_roll: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-pounce-roll.png`,
    beast_archer_pack_call: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-pack-call.png`,
    beast_archer_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-damage-mastery.png`
  });

  const CLASS_ICON_ASSETS = Object.freeze({
    fighter: BASE_SKILL_ICONS.fighter_heavy_strike,
    mage: BASE_SKILL_ICONS.mage_magic_bolt,
    archer: BASE_SKILL_ICONS.archer_quick_shot,
    guardian: ADVANCED_SKILL_ICONS.guardian_shield_bash,
    berserker: ADVANCED_SKILL_ICONS.berserker_blood_cleave,
    duelist: ADVANCED_SKILL_ICONS.duelist_quick_cut,
    fireMage: ADVANCED_SKILL_ICONS.fire_mage_fireball,
    runeMage: ADVANCED_SKILL_ICONS.rune_mage_rune_mark,
    stormMage: ADVANCED_SKILL_ICONS.storm_mage_chain_bolt,
    sniper: ADVANCED_SKILL_ICONS.sniper_aimed_shot,
    trapper: ADVANCED_SKILL_ICONS.trapper_snare_trap,
    beastArcher: ADVANCED_SKILL_ICONS.beast_archer_companion_strike
  });

  const ANIMATION_ROOT = `${ASSET_ROOT}/animations`;
  const COMBAT_FX_ANIMATION_ROOT = `${ANIMATION_ROOT}/combat-fx`;
  const ANIMATION_FRAME_SIZE = 160;
  const COMPACT_ENEMY_FRAME_SIZE = 128;
  const ENEMY_PROJECTILE_FRAME_SIZE = 64;

  const PLAYER_ANIMATION_ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat']);
  const ENEMY_ANIMATION_ROWS = Object.freeze(['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat']);
  const PET_ANIMATION_ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'loot', 'teleport']);
  const SKILL_FX_ANIMATION_ROWS = Object.freeze(['cast', 'projectile', 'impact', 'area']);
  const BASIC_ATTACK_FX_ANIMATION_ROWS = Object.freeze(['cast', 'projectile', 'impact', 'trail']);
  const ENEMY_COMBAT_FX_ANIMATION_ROWS = Object.freeze(['telegraph', 'melee', 'projectile', 'buff', 'impact']);

  const PLAYER_ANIMATION_CONFIG = Object.freeze({
    idle: { frames: 6, fps: 6, loop: true },
    run: { frames: 2, fps: 8, loop: true },
    jump: { frames: 2, fps: 10, loop: false },
    fall: { frames: 2, fps: 8, loop: false },
    climb: { frames: 2, fps: 8, loop: true },
    basic: { frames: 2, fps: 8, loop: false },
    skill: { frames: 2, fps: 8, loop: false },
    party: { frames: 6, fps: 12, loop: false },
    hit: { frames: 6, fps: 16, loop: false },
    defeat: { frames: 6, fps: 9, loop: false }
  });

  const ENEMY_ANIMATION_CONFIG = Object.freeze({
    idle: { frames: 6, fps: 6, loop: true },
    move: { frames: 6, fps: 10, loop: true },
    telegraph: { frames: 6, fps: 12, loop: false },
    attack: { frames: 6, fps: 16, loop: false },
    projectile: { frames: 6, fps: 14, loop: false },
    buff: { frames: 6, fps: 12, loop: false },
    hit: { frames: 6, fps: 16, loop: false },
    defeat: { frames: 6, fps: 9, loop: false }
  });

  const PET_ANIMATION_CONFIG = Object.freeze({
    idle: { frames: 6, fps: 6, loop: true },
    run: { frames: 6, fps: 12, loop: true },
    jump: { frames: 6, fps: 10, loop: false },
    fall: { frames: 6, fps: 9, loop: false },
    loot: { frames: 6, fps: 10, loop: true, loopDelay: 0.12 },
    teleport: { frames: 6, fps: 12, loop: false }
  });

  const FX_ANIMATION_CONFIG = Object.freeze({
    slash: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    cast: { frames: 6, fps: 14, loop: true, loopDelay: 0.18 },
    arrowRelease: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    partyBuff: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 },
    impact: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    defeatBurst: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 }
  });

  const COMBAT_FX_ANIMATION_CONFIG = Object.freeze({
    cast: { frames: 6, fps: 16, loop: true, loopDelay: 0.18 },
    projectile: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    impact: { frames: 6, fps: 20, loop: true, loopDelay: 0.18 },
    area: { frames: 6, fps: 14, loop: true, loopDelay: 0.18 },
    trail: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    telegraph: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 },
    melee: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
    buff: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 }
  });

  const PORTAL_ANIMATION_CONFIG = Object.freeze({
    idle: { frames: 6, fps: 7, loop: true }
  });

  function normalizeFrameHolds(holds, frames) {
    if (!Array.isArray(holds)) return null;
    const frameCount = Math.max(1, Number(frames) || holds.length || 1);
    const normalized = [];
    for (let index = 0; index < frameCount; index += 1) {
      normalized.push(Math.max(1, Math.round(Number(holds[index]) || 1)));
    }
    return Object.freeze(normalized);
  }

  function freezeAnimationStateConfig(config) {
    const state = Object.assign({}, config || {});
    if (Array.isArray(state.holds)) state.holds = normalizeFrameHolds(state.holds, state.frames);
    if (Object.prototype.hasOwnProperty.call(state, 'loopDelay')) state.loopDelay = Math.max(0, Number(state.loopDelay) || 0);
    return Object.freeze(state);
  }

  function makeAnimationStates(rows, config, overrides) {
    return Object.freeze(rows.reduce((states, state, row) => {
      states[state] = freezeAnimationStateConfig(Object.assign({ row }, config[state], overrides && overrides[state]));
      return states;
    }, {}));
  }

  function makeSheetAnimation(sheet, rows, config, overrides, options) {
    const settings = options || {};
    return Object.freeze({
      sheet,
      frameWidth: Number(settings.frameWidth || ANIMATION_FRAME_SIZE),
      frameHeight: Number(settings.frameHeight || settings.frameWidth || ANIMATION_FRAME_SIZE),
      states: makeAnimationStates(rows, config, overrides)
    });
  }

  function makePlayerAnimationAsset(fileId) {
    return makeSheetAnimation(`${ANIMATION_ROOT}/players/${fileId}-sheet.png`, PLAYER_ANIMATION_ROWS, PLAYER_ANIMATION_CONFIG);
  }

  function makeEquipmentVisualAnimation(fileId) {
    return makeSheetAnimation(`${EQUIPMENT_VISUAL_ROOT}/${fileId}-sheet.png`, PLAYER_ANIMATION_ROWS, PLAYER_ANIMATION_CONFIG);
  }

  function mergeAnimationOverrides(base, specific) {
    return Object.freeze(ENEMY_ANIMATION_ROWS.reduce((merged, stateId) => {
      const next = Object.assign({}, base && base[stateId], specific && specific[stateId]);
      if (Object.keys(next).length) merged[stateId] = freezeAnimationStateConfig(next);
      return merged;
    }, {}));
  }

  function makeEnemyAnimationAsset(fileId, enemyId) {
    return makeSheetAnimation(
      `${ANIMATION_ROOT}/enemies/${fileId}-sheet.png`,
      ENEMY_ANIMATION_ROWS,
      ENEMY_ANIMATION_CONFIG,
      mergeAnimationOverrides(ENEMY_ANIMATION_ROW_HOLDS, ENEMY_ANIMATION_TIMING_OVERRIDES[enemyId])
    );
  }

  function makeCompactEnemyAnimationAsset(fileId, enemyId) {
    const compactConfig = Object.freeze(ENEMY_ANIMATION_ROWS.reduce((config, stateId) => {
      const base = ENEMY_ANIMATION_CONFIG[stateId] || {};
      config[stateId] = Object.assign({}, base, {
        frames: stateId === 'hit' ? 1 : 3,
        fps: stateId === 'idle' ? 5 : stateId === 'move' ? 9 : stateId === 'defeat' ? 7 : base.fps || 10,
        holds: stateId === 'hit' ? [2] : stateId === 'idle' ? [4, 2, 4] : stateId === 'defeat' ? [1, 2, 5] : [1, 1, 2]
      });
      return config;
    }, {}));
    return makeSheetAnimation(
      `${ANIMATION_ROOT}/enemies/${fileId}-compact-sheet.png`,
      ENEMY_ANIMATION_ROWS,
      compactConfig,
      null,
      { frameWidth: COMPACT_ENEMY_FRAME_SIZE, frameHeight: COMPACT_ENEMY_FRAME_SIZE }
    );
  }

  function makeEnemyProjectileAnimationAsset(fileId) {
    return makeSheetAnimation(
      `${ANIMATION_ROOT}/enemy-projectiles/${fileId}-sheet.png`,
      ['projectile'],
      Object.freeze({ projectile: { frames: 3, fps: 12, loop: true, loopDelay: 0.18, holds: [1, 1, 1] } }),
      null,
      { frameWidth: ENEMY_PROJECTILE_FRAME_SIZE, frameHeight: ENEMY_PROJECTILE_FRAME_SIZE }
    );
  }

  function makePetAnimationAsset(fileId) {
    return makeSheetAnimation(`${ANIMATION_ROOT}/pets/${fileId}-sheet.png`, PET_ANIMATION_ROWS, PET_ANIMATION_CONFIG);
  }

  function makeFxAnimationAsset(fileId, stateId) {
    return makeSheetAnimation(`${ANIMATION_ROOT}/fx/${fileId}-sheet.png`, [stateId], Object.freeze({ [stateId]: FX_ANIMATION_CONFIG[stateId] }));
  }

  function makeCombatFxAnimationAsset(fileId, folder, rows) {
    const rowIds = Object.freeze((rows || SKILL_FX_ANIMATION_ROWS).slice());
    return makeSheetAnimation(
      `${COMBAT_FX_ANIMATION_ROOT}/${folder}/${fileId}-sheet.png`,
      rowIds,
      Object.freeze(rowIds.reduce((config, rowId) => {
        config[rowId] = COMBAT_FX_ANIMATION_CONFIG[rowId] || COMBAT_FX_ANIMATION_CONFIG.impact;
        return config;
      }, {}))
    );
  }

  function makePortalAnimationAsset(fileId) {
    return makeSheetAnimation(`${ANIMATION_ROOT}/portals/${fileId}-sheet.png`, ['idle'], PORTAL_ANIMATION_CONFIG);
  }

  const GENERIC_PLAYER_ANIMATION_ASSET = makePlayerAnimationAsset('generic-player');

  const PLAYER_ANIMATION_ASSETS = Object.freeze(Object.keys(CLASS_FILE_IDS).reduce((assets, classId) => {
    assets[classId] = makePlayerAnimationAsset(CLASS_FILE_IDS[classId]);
    return assets;
  }, {}));

	  function equipmentVisualFileId(id) {
	    return String(id || '').trim().replace(/_/g, '-');
	  }

	  function inferEquipmentVisualKind(id, slot) {
	    const text = String(id || '').toLowerCase();
	    if (slot === 'weapon') {
	      if (text.includes('bow') || text.includes('string') || text.includes('repeater') || text.includes('recurve')) return 'bow';
	      if (text.includes('axe') || text.includes('cleaver') || text.includes('maul')) return 'axe';
	      if (text.includes('staff') || text.includes('scepter') || text.includes('rod') || text.includes('codex') || text.includes('focus')) return 'staff';
	      return 'sword';
	    }
	    if (slot === 'amulet') return 'amulet';
	    if (slot === 'ring') return 'ring';
	    return slot || 'chest';
	  }

	  const EQUIPMENT_VISUAL_SLOT_META = Object.freeze({
	    weapon: Object.freeze({ layer: 'weapon', order: 40 }),
	    offhand: Object.freeze({ layer: 'offhand', order: 55 }),
	    head: Object.freeze({ layer: 'head', order: 35 }),
	    chest: Object.freeze({ layer: 'chest', order: 20 }),
	    gloves: Object.freeze({ layer: 'gloves', order: 45 }),
	    boots: Object.freeze({ layer: 'boots', order: 30 }),
	    ring: Object.freeze({ layer: 'accessory', order: 65 }),
	    amulet: Object.freeze({ layer: 'accessory', order: 47 })
	  });

	  function makeEquipmentVisualDefinition(config) {
	    const id = String(config && config.id || '').trim();
	    const slot = String(config && config.slot || '').trim();
	    const meta = EQUIPMENT_VISUAL_SLOT_META[slot] || EQUIPMENT_VISUAL_SLOT_META.chest;
	    const fileId = String(config && config.fileId || equipmentVisualFileId(id));
	    return Object.freeze({
	      id,
	      fileId,
	      slot,
	      kind: config && config.kind || inferEquipmentVisualKind(id, slot),
	      layer: config && config.layer || meta.layer,
	      order: config && config.order || meta.order,
	      assetId: config && config.assetId || '',
	      animation: makeEquipmentVisualAnimation(fileId)
	    });
	  }

	  const EXTRA_EQUIPMENT_VISUAL_CONFIGS = Object.freeze([
	    Object.freeze({ id: 'adventurer_cutlass', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'balanced_focus', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'wanderer_charm', slot: 'amulet', kind: 'amulet' }),
	    Object.freeze({ id: 'vanguard_blade', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'bulwark_plate', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'breaker_gauntlets', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'sentinel_greaves', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'starglass_staff', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'runewoven_robes', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'channeler_gloves', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'aetherstep_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'ranger_recurve', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'pathfinder_leathers', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'deadeye_wraps', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'windrunner_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'thorncrown_greatsword', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'thornroot_staff', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'briarstring_longbow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'briar_crown', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'barkplate_harness', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'grasping_thorn_gloves', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'rootstep_greaves', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'emberjaw_cleaver', slot: 'weapon', kind: 'axe' }),
	    Object.freeze({ id: 'magma_scepter', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'cindercoil_bow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'ashen_jaw_helm', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'furnaceplate', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'lavaforged_gauntlets', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'scorchtrail_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'gearcleaver', slot: 'weapon', kind: 'axe' }),
	    Object.freeze({ id: 'chrono_staff', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'ratchet_repeater', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'titan_visor', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'clockplate_harness', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'gyro_gauntlets', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'springstep_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'colossus_maul', slot: 'weapon', kind: 'axe' }),
	    Object.freeze({ id: 'geode_scepter', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'oreline_greatbow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'deepcore_helm', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'bedrock_plate', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'quarry_fists', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'stonewake_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'stormtalon_saber', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'cloudspine_rod', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'skybreaker_bow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'rocfeather_mask', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'tempest_mantle', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'lightning_grip_gloves', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'gale_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'index_blade', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'starbound_codex', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'cometstring_bow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'archivist_crown', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'astral_robes', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'scribe_gloves', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'orbit_boots', slot: 'boots', kind: 'boots' }),
	    Object.freeze({ id: 'eclipse_edge', slot: 'weapon', kind: 'sword' }),
	    Object.freeze({ id: 'umbral_starstaff', slot: 'weapon', kind: 'staff' }),
	    Object.freeze({ id: 'corona_longbow', slot: 'weapon', kind: 'bow' }),
	    Object.freeze({ id: 'sovereign_crown', slot: 'head', kind: 'head' }),
	    Object.freeze({ id: 'eclipse_plate', slot: 'chest', kind: 'chest' }),
	    Object.freeze({ id: 'penumbra_gloves', slot: 'gloves', kind: 'gloves' }),
	    Object.freeze({ id: 'sunfall_boots', slot: 'boots', kind: 'boots' })
	  ]);

	  const BASE_EQUIPMENT_VISUALS = Object.freeze({
	    training_sword: Object.freeze({ id: 'training_sword', fileId: 'training-sword', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('training-sword') }),
	    training_wand: Object.freeze({ id: 'training_wand', fileId: 'training-wand', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('training-wand') }),
	    training_bow: Object.freeze({ id: 'training_bow', fileId: 'training-bow', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('training-bow') }),
    copper_sword: Object.freeze({ id: 'copper_sword', fileId: 'copper-sword', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('copper-sword') }),
    birch_wand: Object.freeze({ id: 'birch_wand', fileId: 'birch-wand', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('birch-wand') }),
    simple_bow: Object.freeze({ id: 'simple_bow', fileId: 'simple-bow', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('simple-bow') }),
    stitched_vest: Object.freeze({ id: 'stitched_vest', fileId: 'stitched-vest', layer: 'chest', order: 20, animation: makeEquipmentVisualAnimation('stitched-vest') }),
    traveler_boots: Object.freeze({ id: 'traveler_boots', fileId: 'traveler-boots', layer: 'boots', order: 30, animation: makeEquipmentVisualAnimation('traveler-boots') }),
    fieldguard_helm: Object.freeze({ id: 'fieldguard_helm', fileId: 'fieldguard-helm', layer: 'head', order: 35, assetId: 'briar_crown', animation: makeEquipmentVisualAnimation('fieldguard-helm') }),
    trailwoven_gloves: Object.freeze({ id: 'trailwoven_gloves', fileId: 'trailwoven-gloves', layer: 'gloves', order: 45, assetId: 'grasping_thorn_gloves', animation: makeEquipmentVisualAnimation('trailwoven-gloves') }),
    plain_ring: Object.freeze({ id: 'plain_ring', fileId: 'plain-ring', layer: 'aura', order: 70, animation: makeEquipmentVisualAnimation('plain-ring') }),
    iron_sword: Object.freeze({ id: 'iron_sword', fileId: 'iron-sword', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('iron-sword') }),
    iron_axe: Object.freeze({ id: 'iron_axe', fileId: 'iron-axe', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('iron-axe') }),
    apprentice_staff: Object.freeze({ id: 'apprentice_staff', fileId: 'apprentice-staff', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('apprentice-staff') }),
    oak_longbow: Object.freeze({ id: 'oak_longbow', fileId: 'oak-longbow', layer: 'weapon', order: 40, animation: makeEquipmentVisualAnimation('oak-longbow') }),
    guardian_tower_shield: Object.freeze({ id: 'guardian_tower_shield', fileId: 'guardian-tower-shield', layer: 'offhand', order: 50, animation: makeEquipmentVisualAnimation('guardian-tower-shield') }),
    berserker_war_grip: Object.freeze({ id: 'berserker_war_grip', fileId: 'berserker-war-grip', layer: 'offhand', order: 55, animation: makeEquipmentVisualAnimation('berserker-war-grip') }),
    ember_core: Object.freeze({ id: 'ember_core', fileId: 'ember-core', layer: 'offhand', order: 55, animation: makeEquipmentVisualAnimation('ember-core') }),
    rune_etched_focus: Object.freeze({ id: 'rune_etched_focus', fileId: 'rune-etched-focus', layer: 'offhand', order: 55, animation: makeEquipmentVisualAnimation('rune-etched-focus') }),
	    deadeye_scope: Object.freeze({ id: 'deadeye_scope', fileId: 'deadeye-scope', layer: 'offhand', order: 55, animation: makeEquipmentVisualAnimation('deadeye-scope') }),
	    trap_kit: Object.freeze({ id: 'trap_kit', fileId: 'trap-kit', layer: 'offhand', order: 55, animation: makeEquipmentVisualAnimation('trap-kit') })
	  });

	  const EQUIPMENT_VISUALS = Object.freeze(Object.assign({},
	    BASE_EQUIPMENT_VISUALS,
	    EXTRA_EQUIPMENT_VISUAL_CONFIGS.reduce((visuals, config) => {
	      visuals[config.id] = makeEquipmentVisualDefinition(config);
	      return visuals;
	    }, {})
	  ));

  const FIGHTER_RIG_ANIMATION_STATES = Object.freeze({
    idle: Object.freeze({ frames: 4, fps: 5, loop: true, timeline: Object.freeze(['settle', 'breath', 'settle', 'breath']) }),
    run: Object.freeze({ frames: 4, fps: 9, loop: true, timeline: Object.freeze(['stepA', 'passA', 'stepB', 'passB']) }),
    jump: Object.freeze({ frames: 2, fps: 10, loop: false, timeline: Object.freeze(['launch', 'tuck']) }),
    fall: Object.freeze({ frames: 2, fps: 8, loop: false, timeline: Object.freeze(['hang', 'brace']) }),
    climb: Object.freeze({ frames: 4, fps: 8, loop: true, timeline: Object.freeze(['reachA', 'pullA', 'reachB', 'pullB']) }),
    basic: Object.freeze({ frames: 4, fps: 13, loop: false, timeline: Object.freeze(['windup', 'lunge', 'follow', 'recover']) }),
    skill: Object.freeze({ frames: 4, fps: 11, loop: false, timeline: Object.freeze(['charge', 'cleave', 'impact', 'recover']) }),
    party: Object.freeze({ frames: 6, fps: 12, loop: false, timeline: Object.freeze(['ready', 'raise', 'flare', 'flare', 'settle', 'settle']) }),
    hit: Object.freeze({ frames: 3, fps: 12, loop: false, timeline: Object.freeze(['recoil', 'recoil', 'settle']) }),
    defeat: Object.freeze({ frames: 4, fps: 8, loop: false, timeline: Object.freeze(['drop', 'down', 'down', 'down']) })
  });

	  function getEquipmentVisualTheme(id) {
	    const text = String(id || '').toLowerCase();
	    if (text.includes('thorn') || text.includes('briar') || text.includes('bark') || text.includes('root')) {
	      return Object.freeze({ dark: '#31452e', main: '#5f8b54', light: '#a9d46f', accent: '#f0d36a', leather: '#6b4a2f' });
	    }
	    if (text.includes('ember') || text.includes('magma') || text.includes('cinder') || text.includes('ashen') || text.includes('furnace') || text.includes('lava') || text.includes('scorch')) {
	      return Object.freeze({ dark: '#4d1f24', main: '#b94735', light: '#ff8a3d', accent: '#ffe16a', leather: '#6b3f2c' });
	    }
	    if (text.includes('gear') || text.includes('chrono') || text.includes('ratchet') || text.includes('titan') || text.includes('clock') || text.includes('gyro') || text.includes('spring')) {
	      return Object.freeze({ dark: '#374451', main: '#75828f', light: '#d6e1e8', accent: '#f3d86d', leather: '#5b4a35' });
	    }
	    if (text.includes('colossus') || text.includes('geode') || text.includes('ore') || text.includes('deepcore') || text.includes('bedrock') || text.includes('quarry') || text.includes('stone')) {
	      return Object.freeze({ dark: '#30343d', main: '#6d7480', light: '#b7c3ca', accent: '#7bdff2', leather: '#4a4039' });
	    }
	    if (text.includes('storm') || text.includes('cloud') || text.includes('sky') || text.includes('roc') || text.includes('tempest') || text.includes('lightning') || text.includes('gale')) {
	      return Object.freeze({ dark: '#243e62', main: '#4f8cff', light: '#9bdfff', accent: '#fff0a6', leather: '#33495b' });
	    }
	    if (text.includes('astral') || text.includes('star') || text.includes('comet') || text.includes('archive') || text.includes('scribe') || text.includes('orbit') || text.includes('index')) {
	      return Object.freeze({ dark: '#31275e', main: '#7f68d9', light: '#c6b8ff', accent: '#ffe16a', leather: '#4b3d74' });
	    }
	    if (text.includes('eclipse') || text.includes('umbral') || text.includes('corona') || text.includes('sovereign') || text.includes('penumbra') || text.includes('sunfall')) {
	      return Object.freeze({ dark: '#1f2333', main: '#4b4b78', light: '#d8c25f', accent: '#fff0a6', leather: '#2e2a38' });
	    }
	    if (text.includes('ranger') || text.includes('pathfinder') || text.includes('deadeye') || text.includes('windrunner')) {
	      return Object.freeze({ dark: '#2d4939', main: '#4f7b58', light: '#8ed174', accent: '#ffe16a', leather: '#5a3f2a' });
	    }
	    if (text.includes('starglass') || text.includes('runewoven') || text.includes('channeler') || text.includes('aether')) {
	      return Object.freeze({ dark: '#253963', main: '#526fbd', light: '#9bdfff', accent: '#b8fff2', leather: '#473b64' });
	    }
	    if (text.includes('vanguard') || text.includes('bulwark') || text.includes('breaker') || text.includes('sentinel')) {
	      return Object.freeze({ dark: '#3a4756', main: '#6f7f90', light: '#d8e5ec', accent: '#f0c36a', leather: '#5a4434' });
	    }
	    return Object.freeze({ dark: '#3d3441', main: '#8a6d4e', light: '#d6c18f', accent: '#ffe16a', leather: '#6f412b' });
	  }

	  function makeFighterRigEquipmentVisual(config) {
	    const kind = config && config.kind || inferEquipmentVisualKind(config && config.id, config && config.slot);
	    const theme = getEquipmentVisualTheme(config && config.id);
	    if (kind === 'sword') return Object.freeze({ kind, blade: theme.light, shine: '#ffffff', grip: theme.leather });
	    if (kind === 'axe') return Object.freeze({ kind, blade: theme.light, shine: '#ffffff', grip: theme.leather });
	    if (kind === 'staff' || kind === 'wand') return Object.freeze({ kind, rod: theme.leather, glow: theme.accent, gem: theme.light });
	    if (kind === 'bow') return Object.freeze({ kind, wood: theme.leather, string: '#fff5d0', arrow: theme.accent, long: true });
	    if (kind === 'chest') return Object.freeze({ kind, cloth: theme.main, trim: theme.light, stitch: theme.accent });
	    if (kind === 'boots') return Object.freeze({ kind, leather: theme.leather, sole: theme.dark, buckle: theme.accent });
	    if (kind === 'head') return Object.freeze({ kind, trim: theme.light, metal: theme.main, dark: theme.dark });
	    if (kind === 'gloves') return Object.freeze({ kind, dark: theme.leather, metal: theme.light, edge: theme.accent });
	    if (kind === 'amulet') return Object.freeze({ kind, metal: theme.light, glow: theme.accent, gem: theme.main });
	    if (kind === 'ring') return Object.freeze({ kind, metal: theme.light, glow: theme.accent });
	    return Object.freeze({ kind: 'chest', cloth: theme.main, trim: theme.light, stitch: theme.accent });
	  }

	  const BASE_FIGHTER_RIG_EQUIPMENT_VISUALS = Object.freeze({
	    training_sword: Object.freeze({ kind: 'sword', blade: '#aebec9', shine: '#eaf5ff', grip: '#8f5f39' }),
	    training_wand: Object.freeze({ kind: 'wand', rod: '#8b5f35', glow: '#8bd7ff', gem: '#c6f4ff' }),
	    training_bow: Object.freeze({ kind: 'bow', wood: '#9b6a35', string: '#f1e6ca', arrow: '#efe2a4' }),
    copper_sword: Object.freeze({ kind: 'sword', blade: '#c8753d', shine: '#ffc179', grip: '#6b3f2c' }),
    birch_wand: Object.freeze({ kind: 'wand', rod: '#d7bd7a', glow: '#9fffd1', gem: '#ecfff4' }),
    simple_bow: Object.freeze({ kind: 'bow', wood: '#8a5c2f', string: '#f5efd6', arrow: '#ffe16a' }),
    iron_sword: Object.freeze({ kind: 'sword', blade: '#dce6ee', shine: '#ffffff', grip: '#565f6c' }),
    iron_axe: Object.freeze({ kind: 'axe', blade: '#d6e1e8', shine: '#ffffff', grip: '#815334' }),
    apprentice_staff: Object.freeze({ kind: 'staff', rod: '#724c2f', glow: '#5fa8ff', gem: '#cbe8ff' }),
    oak_longbow: Object.freeze({ kind: 'bow', wood: '#704826', string: '#fff5d0', arrow: '#ffe16a', long: true }),
    stitched_vest: Object.freeze({ kind: 'chest', cloth: '#8c5a3a', trim: '#d39a5c', stitch: '#f3d5a0' }),
    party_plate: Object.freeze({ kind: 'chest', cloth: '#526f86', trim: '#b7c3ca', stitch: '#68a9ff' }),
    party_robes: Object.freeze({ kind: 'chest', cloth: '#5b62a8', trim: '#b8e6ff', stitch: '#7bdff2' }),
    party_leathers: Object.freeze({ kind: 'chest', cloth: '#4f7b58', trim: '#d5c66a', stitch: '#ffe16a' }),
    traveler_boots: Object.freeze({ kind: 'boots', leather: '#6f412b', sole: '#2b1d1b', buckle: '#d6a14a' }),
    fieldguard_helm: Object.freeze({ kind: 'head', trim: '#8da2af', metal: '#b7c3ca', dark: '#445766' }),
    trailwoven_gloves: Object.freeze({ kind: 'gloves', dark: '#6f412b', metal: '#b7c3ca', edge: '#d6a14a' }),
    plain_ring: Object.freeze({ kind: 'ring', metal: '#f7d879', glow: '#ffe16a' }),
    guardian_tower_shield: Object.freeze({ kind: 'shield', face: '#466d91', trim: '#d5ecff', metal: '#6fa8d9' }),
    berserker_war_grip: Object.freeze({ kind: 'grip', metal: '#a22d36', edge: '#ff6b5e', dark: '#4d1f24' }),
    ember_core: Object.freeze({ kind: 'core', core: '#ff6b35', glow: '#ffc15e', dark: '#8b2635' }),
    rune_etched_focus: Object.freeze({ kind: 'focus', core: '#28c7b7', glow: '#b8fff2', dark: '#146b72' }),
	    deadeye_scope: Object.freeze({ kind: 'scope', metal: '#4b5663', lens: '#ffe16a', trim: '#d8c25f' }),
	    trap_kit: Object.freeze({ kind: 'kit', leather: '#8a5a36', metal: '#b7c3ca', cord: '#3f2c24' })
	  });

	  const FIGHTER_RIG_EQUIPMENT_VISUALS = Object.freeze(Object.assign({},
	    BASE_FIGHTER_RIG_EQUIPMENT_VISUALS,
	    EXTRA_EQUIPMENT_VISUAL_CONFIGS.reduce((visuals, config) => {
	      visuals[config.id] = makeFighterRigEquipmentVisual(config);
	      return visuals;
	    }, {})
	  ));

  const GENERIC_PLAYER_RIG = Object.freeze({
      id: 'genericPlayer',
      renderer: 'blockyLayeredCanvas',
      width: 78,
      height: 92,
      scale: 0.74,
      groundY: 47,
      style: 'blocky-pixel-runtime-v1',
      palette: Object.freeze({
        outline: '#13222f',
        shadow: '#0e2636',
        skin: '#d99a6c',
        skinLight: '#f0bd88',
        hair: '#3b2725',
        shirt: '#526f86',
        shirtLight: '#8caabd',
        pants: '#33495b',
        pantsDark: '#24313f',
        belt: '#7d5132',
        boot: '#2a1f21',
        hand: '#e4aa78'
      }),
      drawOrder: Object.freeze([
        'backAura',
        'backArm',
        'backLeg',
        'frontLeg',
        'torso',
        'chest',
        'head',
        'offhand',
        'weapon',
        'frontArm',
        'frontHand',
        'frontAura'
      ]),
      anchors: Object.freeze({
        torso: Object.freeze({ x: 0, y: -18 }),
        head: Object.freeze({ x: 3, y: -51 }),
        backShoulder: Object.freeze({ x: -8, y: -28 }),
        frontShoulder: Object.freeze({ x: 10, y: -27 }),
        backHip: Object.freeze({ x: -5, y: 3 }),
        frontHip: Object.freeze({ x: 7, y: 3 }),
        weaponHand: Object.freeze({ x: 22, y: -10 }),
        offhand: Object.freeze({ x: -15, y: -8 }),
        backFoot: Object.freeze({ x: -7, y: 39 }),
        frontFoot: Object.freeze({ x: 11, y: 39 }),
        aura: Object.freeze({ x: 0, y: -18 })
      }),
      equipmentSlots: Object.freeze({
        weapon: 'weaponHand',
        offhand: 'offhand',
        head: 'head',
        chest: 'torso',
        gloves: 'weaponHand',
        boots: 'feet',
        ring: 'weaponHand',
        amulet: 'aura'
      }),
      attachments: Object.freeze({
        weapon: Object.freeze({ slot: 'weapon', anchor: 'weaponHand', layer: 'weapon' }),
        offhand: Object.freeze({ slot: 'offhand', anchor: 'offhand', layer: 'offhand' }),
        chest: Object.freeze({ slot: 'chest', anchor: 'torso', layer: 'torso' }),
        boots: Object.freeze({ slot: 'boots', anchor: 'feet', layer: 'feet' }),
        ring: Object.freeze({ slot: 'ring', anchor: 'weaponHand', layer: 'aura' }),
        amulet: Object.freeze({ slot: 'amulet', anchor: 'aura', layer: 'aura' }),
        head: Object.freeze({ slot: 'head', anchor: 'head', layer: 'head' }),
        gloves: Object.freeze({ slot: 'gloves', anchor: 'weaponHand', layer: 'hands' })
      }),
      animationStates: FIGHTER_RIG_ANIMATION_STATES,
      equipmentVisuals: FIGHTER_RIG_EQUIPMENT_VISUALS
    });

  const PLAYER_RIGS = Object.freeze(Object.keys(CLASS_FILE_IDS).reduce((rigs, classId) => {
    rigs[classId] = GENERIC_PLAYER_RIG;
    return rigs;
  }, {}));

  function freezeAnimationOverrideMap(map) {
    return Object.freeze(Object.keys(map || {}).reduce((frozen, stateId) => {
      frozen[stateId] = freezeAnimationStateConfig(map[stateId]);
      return frozen;
    }, {}));
  }

  function freezeEnemyTimingOverrides(map) {
    return Object.freeze(Object.keys(map || {}).reduce((frozen, enemyId) => {
      frozen[enemyId] = freezeAnimationOverrideMap(map[enemyId]);
      return frozen;
    }, {}));
  }

  const ENEMY_ANIMATION_ROW_HOLDS = freezeAnimationOverrideMap({
    idle: { holds: [5, 2, 3, 2, 3, 5] },
    move: { holds: [1, 1, 2, 1, 1, 2] },
    telegraph: { holds: [3, 2, 1, 1, 1, 2] },
    attack: { holds: [2, 1, 1, 1, 2, 3] },
    projectile: { holds: [2, 1, 1, 1, 2, 2] },
    buff: { holds: [3, 1, 2, 1, 2, 3] },
    hit: { holds: [1, 1, 2, 2, 2, 3] },
    defeat: { holds: [1, 1, 2, 2, 4, 6] }
  });

  const ENEMY_ANIMATION_TIMING_OVERRIDES = freezeEnemyTimingOverrides({
    briarStag: {
      idle: { fps: 5, holds: [6, 2, 3, 2, 3, 6] },
      move: { fps: 9, holds: [2, 1, 2, 1, 2, 2] },
      telegraph: { fps: 10, holds: [4, 2, 1, 1, 2, 3] },
      attack: { fps: 13, holds: [3, 1, 1, 2, 2, 4] },
      projectile: { fps: 11, holds: [2, 1, 1, 2, 2, 3] },
      buff: { fps: 9, holds: [4, 1, 2, 1, 3, 4] },
      hit: { fps: 12, holds: [1, 1, 2, 2, 3, 4] },
      defeat: { fps: 7, holds: [1, 1, 2, 3, 5, 8] }
    },
    vineSnapper: {
      idle: { fps: 5, holds: [5, 2, 4, 2, 4, 5] },
      attack: { fps: 14, holds: [3, 1, 1, 1, 2, 4] },
      defeat: { fps: 7, holds: [1, 1, 2, 3, 5, 8] }
    },
    banditCutter: {
      idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
      move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
      attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
    },
    banditCutterDirect: {
      idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
      move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
      attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
    },
    banditCutterReference: {
      idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
      move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
      attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
    },
    banditCutterHybrid: {
      idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
      move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
      attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
    },
    banditCutterPuppet: {
      idle: { fps: 6, holds: [5, 2, 3, 2, 3, 5] },
      move: { fps: 10, holds: [2, 1, 2, 1, 2, 2] },
      attack: { fps: 16, holds: [2, 1, 1, 2, 2, 3] }
    },
    banditThrower: {
      idle: { fps: 6, holds: [4, 2, 3, 2, 3, 4] },
      telegraph: { fps: 11, holds: [3, 2, 1, 1, 2, 3] },
      projectile: { fps: 15, holds: [2, 1, 1, 1, 2, 3] }
    },
    thunderRam: {
      idle: { fps: 5, holds: [6, 2, 3, 2, 3, 6] },
      move: { fps: 12, holds: [1, 1, 1, 1, 2, 2] },
      attack: { fps: 16, holds: [3, 1, 1, 1, 2, 3] }
    },
    glacierSentinel: {
      idle: { fps: 4, holds: [7, 2, 4, 2, 4, 7] },
      attack: { fps: 12, holds: [4, 2, 1, 1, 2, 5] },
      defeat: { fps: 6, holds: [1, 2, 3, 4, 6, 9] }
    }
  });

  const ENEMY_ANIMATION_FILE_IDS = Object.freeze({
    slimelet: 'slimelet',
    dewSlime: 'dew-slime',
    mossback: 'mossback',
    thornSprout: 'thorn-sprout',
    vineSnapper: 'vine-snapper',
    bristleBoar: 'bristle-boar',
    briarStag: 'briar-stag',
    dustImp: 'dust-imp',
    clockbug: 'clockbug',
    rustRatchet: 'rust-ratchet',
    coilSentry: 'coil-sentry',
    scrapWarden: 'scrap-warden',
    emberWisp: 'ember-wisp',
    ashCrawler: 'ash-crawler',
    lavaTick: 'lava-tick',
    cinderSpitter: 'cinder-spitter',
    banditCutter: 'bandit-cutter',
    banditCutterDirect: 'bandit-cutter-direct',
    banditCutterReference: 'bandit-cutter-reference',
    banditCutterHybrid: 'bandit-cutter-hybrid',
    banditCutterPuppet: 'bandit-cutter-puppet',
    banditThrower: 'bandit-thrower',
    orebackBeetle: 'oreback-beetle',
    glowcapHealer: 'glowcap-healer',
    crackedMimic: 'cracked-mimic',
    brambleking: 'brambleking',
    clockworkTitan: 'clockwork-titan',
    quarryColossus: 'quarry-colossus',
    emberjawGolem: 'emberjaw-golem',
    frostlingScout: 'frostling-scout',
    shardling: 'shardling',
    rimebackBrute: 'rimeback-brute',
    glacierSentinel: 'glacier-sentinel',
    snowglareWisp: 'snowglare-wisp',
    icebloomOracle: 'icebloom-oracle',
    galeHarrier: 'gale-harrier',
    stormboundArcher: 'stormbound-archer',
    thunderRam: 'thunder-ram',
    cloudcallAcolyte: 'cloudcall-acolyte',
    indexScribe: 'index-scribe',
    lumenSentinel: 'lumen-sentinel',
    voidMote: 'void-mote',
    eclipseDuelist: 'eclipse-duelist',
    riftAberration: 'rift-aberration',
    stormbreakRoc: 'stormbreak-roc',
    astralArchivist: 'astral-archivist',
    eclipseSovereign: 'eclipse-sovereign',
    rimewarden: 'rimewarden'
  });

  const COMPACT_ENEMY_ANIMATION_FILE_IDS = ENEMY_ANIMATION_FILE_IDS;

  const ENEMY_ANIMATION_ASSETS = Object.freeze(Object.keys(ENEMY_ANIMATION_FILE_IDS).reduce((assets, enemyId) => {
    assets[enemyId] = COMPACT_ENEMY_ANIMATION_FILE_IDS[enemyId]
      ? makeCompactEnemyAnimationAsset(COMPACT_ENEMY_ANIMATION_FILE_IDS[enemyId], enemyId)
      : makeEnemyAnimationAsset(ENEMY_ANIMATION_FILE_IDS[enemyId], enemyId);
    return assets;
  }, {}));

  const ENEMY_PROJECTILE_ANIMATION_ASSETS = Object.freeze({
    banditThrower: makeEnemyProjectileAnimationAsset('bandit-knife')
  });

  const ENEMY_ANIMATION_BEHAVIORS = Object.freeze({
    melee: Object.freeze({ id: 'melee', states: ENEMY_ANIMATION_ROWS }),
    ranged: Object.freeze({ id: 'ranged', states: ENEMY_ANIMATION_ROWS }),
    charger: Object.freeze({ id: 'charger', states: ENEMY_ANIMATION_ROWS }),
    flyer: Object.freeze({ id: 'flyer', states: ENEMY_ANIMATION_ROWS }),
    healer: Object.freeze({ id: 'healer', states: ENEMY_ANIMATION_ROWS }),
    elite: Object.freeze({ id: 'elite', states: ENEMY_ANIMATION_ROWS }),
    boss: Object.freeze({ id: 'boss', states: ENEMY_ANIMATION_ROWS })
  });

  const FX_ANIMATION_ASSETS = Object.freeze({
    slash: makeFxAnimationAsset('slash', 'slash'),
    cast: makeFxAnimationAsset('cast', 'cast'),
    arrowRelease: makeFxAnimationAsset('arrow-release', 'arrowRelease'),
    partyBuff: makeFxAnimationAsset('party-buff', 'partyBuff'),
    impact: makeFxAnimationAsset('impact', 'impact'),
    defeatBurst: makeFxAnimationAsset('defeat-burst', 'defeatBurst')
  });

  const PORTAL_ANIMATION_ASSETS = Object.freeze({
    standard: makePortalAnimationAsset('standard'),
    boss: makePortalAnimationAsset('boss'),
    locked: makePortalAnimationAsset('locked')
  });

  const PET_ANIMATION_ASSET = makePetAnimationAsset('starfall-fox');

  const BUFF_CAST_VISUALS = Object.freeze({
    fighter_guard: Object.freeze({ id: 'fighter_guard', style: 'barrier', color: '#68a9ff', accent: '#d5ecff' }),
    mage_mana_shield: Object.freeze({ id: 'mage_mana_shield', style: 'barrier', color: '#8bd7ff', accent: '#e7fbff' }),
    guardian_impact_guard: Object.freeze({ id: 'guardian_impact_guard', style: 'barrier', color: '#6fa8d9', accent: '#d5ecff' }),
    guardian_oath_barrier: Object.freeze({ id: 'guardian_oath_barrier', style: 'barrier', color: '#7fc4ff', accent: '#e3f6ff' }),
    guardian_hold_the_line: Object.freeze({ id: 'guardian_hold_the_line', style: 'barrier', color: '#466d91', accent: '#d5ecff' }),
    shieldWall: Object.freeze({ id: 'shieldWall', skillId: 'guardian_shield_wall', style: 'barrier', color: '#6fa8d9', accent: '#d5ecff' }),
    warCry: Object.freeze({ id: 'warCry', skillId: 'berserker_war_cry', style: 'cry', color: '#ef3d55', accent: '#ffbe55' }),
    rallyingFlourish: Object.freeze({ id: 'rallyingFlourish', skillId: 'duelist_rallying_flourish', style: 'flourish', color: '#5fd6c6', accent: '#ffe16a' }),
    ignitionAura: Object.freeze({ id: 'ignitionAura', skillId: 'fire_mage_ignition_aura', style: 'flame', color: '#ff7b3a', accent: '#ffd166' }),
    runeCircle: Object.freeze({ id: 'runeCircle', skillId: 'rune_mage_rune_circle', style: 'rune', color: '#28c7b7', accent: '#b8fff2' }),
    stormfront: Object.freeze({ id: 'stormfront', skillId: 'storm_mage_stormfront', style: 'storm', color: '#7aa7ff', accent: '#f0f7ff' }),
    eagleEye: Object.freeze({ id: 'eagleEye', skillId: 'sniper_eagle_eye', style: 'focus', color: '#d8c25f', accent: '#fff5b8' }),
    tacticalField: Object.freeze({ id: 'tacticalField', skillId: 'trapper_tactical_field', style: 'tactical', color: '#66d79a', accent: '#dbffe6' }),
    packCall: Object.freeze({ id: 'packCall', skillId: 'beast_archer_pack_call', style: 'pack', color: '#8ed174', accent: '#fff0a6' })
  });

  const CORE_ANIMATION_ASSETS = Object.freeze({
    players: PLAYER_ANIMATION_ASSETS,
    enemies: ENEMY_ANIMATION_ASSETS,
    enemyBehaviors: ENEMY_ANIMATION_BEHAVIORS,
    pet: PET_ANIMATION_ASSET,
    fx: FX_ANIMATION_ASSETS,
    portals: PORTAL_ANIMATION_ASSETS
  });

  function getEnemyAnimationBehavior(enemy) {
    if (!enemy) return 'melee';
    if (enemy.id === 'emberjawGolem' || enemy.behavior === 'boss') return 'boss';
    if (enemy.id === 'crackedMimic' || enemy.behavior === 'elite') return 'elite';
    if (enemy.behavior === 'flyer') return 'flyer';
    if (enemy.behavior === 'healer') return 'healer';
    if (enemy.behavior === 'charger') return 'charger';
    if (enemy.behavior === 'thrower' || enemy.behavior === 'turret') return 'ranged';
    return 'melee';
  }

  function attachAsset(record, asset) {
    return Object.freeze(Object.assign({}, record, { asset: asset || '' }));
  }

  function attachEnemyAssets(enemy) {
    const animationBehavior = getEnemyAnimationBehavior(enemy);
    return Object.freeze(Object.assign({}, enemy, {
      asset: ENEMY_ASSETS[enemy.id] || '',
      animation: ENEMY_ANIMATION_ASSETS[enemy.id] || null,
      animationBehavior
    }));
  }

  function attachMapAssets(map) {
    const node = (typeof WORLD_MAP_NODES !== 'undefined' ? WORLD_MAP_NODES : []).find((item) => item && item.mapId === map.id);
    const areaId = map.areaId || node && node.areaId || '';
    const area = areaId && typeof WORLD_AREAS !== 'undefined' ? WORLD_AREAS.find((item) => item && item.id === areaId) : null;
    const blueprint = MAP_LAYOUT_BLUEPRINTS[map.id] || Object.freeze({});
    const layoutRole = normalizeMapLayoutRole(blueprint.role || map.layoutRole || node && node.role, getMapLayoutRoleFallback(map));
    const townScene = map.safeZone ? MAP_TOWN_SCENES[map.id] || createDefaultTownScene(map) : null;
    const fieldComposition = !map.safeZone ? MAP_FIELD_COMPOSITIONS[map.id] || createDefaultFieldComposition(map, blueprint) : null;
    const portalRoles = fieldComposition && fieldComposition.portalRoles || {};
    return Object.freeze(Object.assign({}, map, {
      areaId,
      areaName: area ? area.name : node && node.region || '',
      areaMechanic: area ? area.mechanic : '',
      layoutRole,
      layoutRoleLabel: MAP_LAYOUT_ROLE_LABELS[layoutRole] || 'Training Field',
      layoutMarker: MAP_LAYOUT_ROLES[layoutRole] && MAP_LAYOUT_ROLES[layoutRole].marker || '',
      routeStage: map.routeStage || blueprint.routeStage || '',
      mapRoadName: map.mapRoadName || blueprint.roadName || map.name || '',
      landmark: map.landmark || blueprint.landmark || '',
      portalPattern: map.portalPattern || blueprint.portalPattern || '',
      townScene,
      fieldComposition,
      asset: MAP_ASSETS[map.id] || '',
      environment: MAP_ENVIRONMENT_PROFILES[map.id] || MAP_ENVIRONMENT_PROFILES.greenrootMeadow,
      stations: (map.stations || []).map((station) => attachAsset(station, STATION_ASSETS[station.id])),
      questNpcs: (map.questNpcs || []).map((npc) => Object.freeze(Object.assign({}, npc, {
        questIds: Object.freeze((npc.questIds || []).slice())
      }))),
      portals: (map.portals || MAP_PORTALS[map.id] || []).map((portal) => Object.freeze(Object.assign({}, portal, {
        roleLabel: portal.roleLabel || portalRoles[portal.id] || ''
      })))
    }));
  }

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
        { label: 'Town Gate', x: 0, w: 1900, tier: 'starter' },
        { label: 'Meadow Lanes', x: 1900, w: 3300, tier: 'training' },
        { label: 'Thornpath Exit', x: 5200, w: 2200, tier: 'advance' }
      ],
      portalRoles: {
        greenroot_crossing: 'left town return',
        greenroot_thornpath: 'right route advance'
      },
      landmarkBands: [
        { kind: 'tree', x: 460, w: 1220, label: 'Beginner Grove' },
        { kind: 'vine', x: 3600, w: 960, label: 'Canopy Shortcut' },
        { kind: 'sign', x: 6400, w: 420, label: 'Thornpath Sign' }
      ],
      spawnZoneLabels: [
        { label: 'Slime pond', platformTier: 'low' },
        { label: 'Moss lane', platformTier: 'mid' },
        { label: 'Canopy lane', platformTier: 'high' }
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
    })
  });

	  const CLASS_ROLE_PROFILES = Object.freeze({
    fighter: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Melee control',
      specialty: 'Close-range impact',
      summary: 'Reliable melee pressure with short-range control, guard windows, and finishers.'
    }),
    mage: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Area setup',
      specialty: 'Spell routing',
      summary: 'Ranged spell pressure with strong setup tools and the safest vertical targeting.'
    }),
    archer: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Mobile range',
      specialty: 'Marks and spacing',
      summary: 'Mobile ranged pressure that can jump attack and rewards choosing the right target.'
    }),
    guardian: Object.freeze({
      primary: 'Support / Control',
      secondary: 'Boss safety',
      specialty: 'Stored Impact',
      summary: 'Stores defensive pressure, breaks bosses open, and turns survival into counter-burst.'
    }),
    berserker: Object.freeze({
      primary: 'Bossing',
      secondary: 'Risk sustain',
      specialty: 'Low-HP burst',
      summary: 'Trades safety for strong single-target damage and sustain when fighting dangerous enemies.'
    }),
    duelist: Object.freeze({
      primary: 'Bossing',
      secondary: 'Tempo burst',
      specialty: 'Same-target combos',
      summary: 'Stacks Tempo on one target and rewards clean repeated hits over wide mob clearing.'
    }),
    fireMage: Object.freeze({
      primary: 'Mobbing',
      secondary: 'Burn ramp',
      specialty: 'Burn spread',
      summary: 'Spreads burns through clustered enemies and converts Heat into larger explosions.'
    }),
    runeMage: Object.freeze({
      primary: 'Support / Control',
      secondary: 'Setup burst',
      specialty: 'Linked runes',
      summary: 'Marks, links, slows, and detonates enemies after preparing rune setups.'
    }),
    stormMage: Object.freeze({
      primary: 'Mobbing',
      secondary: 'Chain pressure',
      specialty: 'Chain lightning',
      summary: 'Excellent clustered-enemy clearing through chained delayed lightning pulses.'
    }),
    sniper: Object.freeze({
      primary: 'Bossing',
      secondary: 'Weak-point burst',
      specialty: 'Single-target payoff',
      summary: 'Marks weak points and cashes them out with high-value shots against bosses.'
    }),
    trapper: Object.freeze({
      primary: 'Mobbing / Control',
      secondary: 'Area setup',
      specialty: 'Trap networks',
      summary: 'Prepares traps that slow, trigger, and detonate groups as enemies path into them.'
    }),
    beastArcher: Object.freeze({
      primary: 'Support / Hybrid',
      secondary: 'Sustain pressure',
      specialty: 'Companion marks',
      summary: 'Coordinates companion hits that mark targets and sustain the character over longer fights.'
    })
  });

  const BASE_CLASSES = Object.freeze({
    fighter: {
      id: 'fighter',
      name: 'Fighter',
      asset: CLASS_ASSETS.fighter,
      animation: PLAYER_ANIMATION_ASSETS.fighter,
      resourceName: 'Momentum',
      resourceColor: '#f25f4c',
      weaponType: 'melee',
      roleProfile: CLASS_ROLE_PROFILES.fighter,
      description: 'Close-range weapon class built around impact, guarded timing, and controlled aggression.',
      stats: {
        hp: 180,
        power: 18,
        defense: 8,
        speed: 220,
        jump: 520,
        range: 76,
        mpMax: 90,
        resourceMax: 100
      }
    },
    mage: {
      id: 'mage',
      name: 'Mage',
      asset: CLASS_ASSETS.mage,
      animation: PLAYER_ANIMATION_ASSETS.mage,
      resourceName: 'Energy',
      resourceColor: '#4f8cff',
      weaponType: 'projectile',
      roleProfile: CLASS_ROLE_PROFILES.mage,
      description: 'Ranged spellcaster with area damage, utility, and resource-aware burst windows.',
      stats: {
        hp: 135,
        power: 20,
        defense: 4,
        speed: 212,
        jump: 510,
        range: 380,
        mpMax: 130,
        resourceMax: 120
      }
    },
    archer: {
      id: 'archer',
      name: 'Archer',
      asset: CLASS_ASSETS.archer,
      animation: PLAYER_ANIMATION_ASSETS.archer,
      resourceName: 'Focus',
      resourceColor: '#3aa76d',
      weaponType: 'projectile',
      roleProfile: CLASS_ROLE_PROFILES.archer,
      description: 'Ranged physical class that rewards spacing, marking, mobility, and target selection.',
      stats: {
        hp: 150,
        power: 19,
        defense: 5,
        speed: 235,
        jump: 525,
        range: 430,
        mpMax: 100,
        resourceMax: 110
      }
    }
  });

  const ADVANCED_CLASSES = Object.freeze({
    guardian: {
      id: 'guardian',
      name: 'Guardian',
      asset: CLASS_ASSETS.guardian,
      animation: PLAYER_ANIMATION_ASSETS.guardian,
      baseClass: 'fighter',
      levelRequirement: 25,
      resourceName: 'Stored Impact',
      resourceColor: '#68a9ff',
      partySkillId: 'guardian_shield_wall',
      roleProfile: CLASS_ROLE_PROFILES.guardian,
      description: 'Defensive Fighter path that converts blocked damage into counter-pressure.'
    },
    berserker: {
      id: 'berserker',
      name: 'Berserker',
      asset: CLASS_ASSETS.berserker,
      animation: PLAYER_ANIMATION_ASSETS.berserker,
      baseClass: 'fighter',
      levelRequirement: 25,
      resourceName: 'Rage',
      resourceColor: '#ef3d55',
      partySkillId: 'berserker_war_cry',
      roleProfile: CLASS_ROLE_PROFILES.berserker,
      description: 'Risky Fighter path with low-health burst, lifesteal, and aggressive sustain.'
    },
    duelist: {
      id: 'duelist',
      name: 'Duelist',
      asset: CLASS_ASSETS.duelist,
      animation: PLAYER_ANIMATION_ASSETS.duelist,
      baseClass: 'fighter',
      levelRequirement: 25,
      resourceName: 'Tempo',
      resourceColor: '#f0c36a',
      partySkillId: 'duelist_rallying_flourish',
      roleProfile: CLASS_ROLE_PROFILES.duelist,
      description: 'Mobile Fighter path built around quick windows, precision counters, and party haste.'
    },
    fireMage: {
      id: 'fireMage',
      name: 'Fire Mage',
      asset: CLASS_ASSETS.fireMage,
      animation: PLAYER_ANIMATION_ASSETS.fireMage,
      baseClass: 'mage',
      levelRequirement: 25,
      resourceName: 'Heat',
      resourceColor: '#ff8a3d',
      partySkillId: 'fire_mage_ignition_aura',
      roleProfile: CLASS_ROLE_PROFILES.fireMage,
      description: 'Explosive Mage path that builds Heat and vents it into area pressure.'
    },
    runeMage: {
      id: 'runeMage',
      name: 'Rune Mage',
      asset: CLASS_ASSETS.runeMage,
      animation: PLAYER_ANIMATION_ASSETS.runeMage,
      baseClass: 'mage',
      levelRequirement: 25,
      resourceName: 'Runic Energy',
      resourceColor: '#28c7b7',
      partySkillId: 'rune_mage_rune_circle',
      roleProfile: CLASS_ROLE_PROFILES.runeMage,
      description: 'Setup Mage path that places, links, detonates, and empowers rune fields.'
    },
    stormMage: {
      id: 'stormMage',
      name: 'Storm Mage',
      asset: CLASS_ASSETS.stormMage,
      animation: PLAYER_ANIMATION_ASSETS.stormMage,
      baseClass: 'mage',
      levelRequirement: 25,
      resourceName: 'Charge',
      resourceColor: '#7bdff2',
      partySkillId: 'storm_mage_stormfront',
      roleProfile: CLASS_ROLE_PROFILES.stormMage,
      description: 'Fast Mage path that chains lightning through grouped targets and builds Charge through movement.'
    },
    sniper: {
      id: 'sniper',
      name: 'Sniper',
      asset: CLASS_ASSETS.sniper,
      animation: PLAYER_ANIMATION_ASSETS.sniper,
      baseClass: 'archer',
      levelRequirement: 25,
      resourceName: 'Aim',
      resourceColor: '#c9b35c',
      partySkillId: 'sniper_eagle_eye',
      roleProfile: CLASS_ROLE_PROFILES.sniper,
      description: 'Precision Archer path focused on weak points, precision strikes, and boss damage.'
    },
    trapper: {
      id: 'trapper',
      name: 'Trapper',
      asset: CLASS_ASSETS.trapper,
      animation: PLAYER_ANIMATION_ASSETS.trapper,
      baseClass: 'archer',
      levelRequirement: 25,
      resourceName: 'Preparation',
      resourceColor: '#b07a47',
      partySkillId: 'trapper_tactical_field',
      roleProfile: CLASS_ROLE_PROFILES.trapper,
      description: 'Tactical Archer path that wins through traps, slows, and enemy routing.'
    },
    beastArcher: {
      id: 'beastArcher',
      name: 'Beast Archer',
      asset: CLASS_ASSETS.beastArcher,
      animation: PLAYER_ANIMATION_ASSETS.beastArcher,
      baseClass: 'archer',
      levelRequirement: 25,
      resourceName: 'Bond',
      resourceColor: '#78b26a',
      partySkillId: 'beast_archer_pack_call',
      roleProfile: CLASS_ROLE_PROFILES.beastArcher,
      description: 'Companion Archer path focused on coordinated strikes, survivability, and party resource support.'
    }
  });

  const SKILL_PURPOSES = Object.freeze({
    trainer: Object.freeze({ label: 'Trainer', description: 'Reliable low-cooldown skill for normal leveling.' }),
    mobility: Object.freeze({ label: 'Mobility', description: 'Repositioning, gap crossing, or escape.' }),
    setup: Object.freeze({ label: 'Setup', description: 'Marks, cracks, links, or prepares a stronger follow-up.' }),
    control: Object.freeze({ label: 'Control', description: 'Slows, staggers, pulls, or otherwise shapes enemy movement.' }),
    mobbing: Object.freeze({ label: 'Mobbing', description: 'Clears clustered regular enemies.' }),
    bossing: Object.freeze({ label: 'Bossing', description: 'Focused damage or debuffs for durable targets.' }),
    defense: Object.freeze({ label: 'Defense', description: 'Prevents damage or turns pressure into resources.' }),
    sustain: Object.freeze({ label: 'Sustain', description: 'Restores HP, MP, or keeps resources stable.' }),
    resource: Object.freeze({ label: 'Resource', description: 'Builds, vents, or spends a class resource deliberately.' }),
    buff: Object.freeze({ label: 'Buff', description: 'Temporary self-enhancement for a combat window.' }),
    finisher: Object.freeze({ label: 'Finisher', description: 'Payoff skill with higher cost, setup, or cooldown.' }),
    passive: Object.freeze({ label: 'Passive', description: 'Permanent stat or mechanic upgrade.' }),
    party: Object.freeze({ label: 'Party', description: 'Solo self-buff now, party support hook later.' })
  });

  function normalizeSkillPurpose(value) {
    const purpose = String(value || '').trim();
    return Object.prototype.hasOwnProperty.call(SKILL_PURPOSES, purpose) ? purpose : '';
  }

  function inferSkillPurpose(id, type, roleTags, options) {
    const skillId = String(id || '').toLowerCase();
    const skillType = String(type || '').toLowerCase();
    const tags = roleTags || [];
    const config = options || {};
    const configured = normalizeSkillPurpose(config.purpose);
    if (configured) return configured;
    if (config.primaryTraining || skillType.includes('primary training') || skillType.includes('basic attack')) return 'trainer';
    if (skillType.includes('party') || config.partyEffect) return 'party';
    if (skillType.includes('passive') || config.passiveStats) return 'passive';
    if (config.movementEffect || skillId.includes('blink') || skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillType.includes('defense') || skillId.includes('_guard') || skillId.startsWith('guard_') || skillId.includes('shield') || skillId.includes('barrier')) return 'defense';
    if (skillType.includes('sustain') || skillId.includes('recovery')) return 'sustain';
    if (skillType.includes('resource') || skillId.includes('heat_vent')) return 'resource';
    if (skillType.includes('finisher') || skillType.includes('ultimate') || skillType.includes('burst') || skillId.includes('detonate')) return 'finisher';
    if ((!skillType.includes('debuff') && skillType.includes('buff')) || skillType.includes('stance') || skillId.includes('surge') || skillId.includes('breath')) return 'buff';
    if (skillType.includes('setup') || skillType.includes('combo') || skillType.includes('debuff') || skillId.includes('mark') || skillId.includes('break') || skillId.includes('armor')) return 'setup';
    if (skillType.includes('control') || skillType.includes('utility') || skillId.includes('seal') || skillId.includes('lure')) return 'control';
    if (tags.includes('Mobbing') || skillType.includes('area') || skillId.includes('trap') || skillId.includes('glyph') || skillId.includes('zone')) return 'mobbing';
    if (tags.includes('Bossing')) return 'bossing';
    return 'trainer';
  }

  function inferSkillIconKind(id, type, roleTags) {
    const skillId = String(id || '');
    const skillType = String(type || '').toLowerCase();
    const tags = roleTags || [];
    if (skillId.includes('blink')) return 'blink';
    if (skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillId.includes('guard') || skillId.includes('shield') || skillId.includes('barrier') || skillType.includes('defense')) return 'guard';
    if (skillId.includes('trap') || skillId.includes('field') || skillId.includes('glyph') || skillId.includes('circle')) return 'field';
    if (skillId.includes('mark') || skillId.includes('eye') || skillType.includes('setup')) return 'mark';
    if (skillId.includes('arrow') || skillId.includes('shot') || skillId.includes('volley')) return 'arrow';
    if (skillId.includes('bolt') || skillId.includes('fireball') || skillId.includes('rune')) return 'magic';
    if (skillId.includes('break') || skillId.includes('armor')) return 'break';
    if (skillType.includes('area') || tags.includes('Mobbing')) return 'area';
    if (skillType.includes('buff') || tags.includes('Party') || tags.includes('Support')) return 'buff';
    if (skillType.includes('finisher') || skillId.includes('burst')) return 'burst';
    return 'slash';
  }

  function inferSkillCategory(id, type, options) {
    const skillId = String(id || '').toLowerCase();
    const skillType = String(type || '').toLowerCase();
    const config = options || {};
    if (config.category) return config.category;
    if (skillType.includes('passive') || config.passiveStats) return 'passive';
    if (config.movementEffect || skillId.includes('blink') || skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillType.includes('debuff')) return 'attack';
    if (skillType.includes('buff') || skillType.includes('defense') || skillType.includes('party') || skillType.includes('stance')) return 'buff';
    return 'attack';
  }

  function inferSkillMaxRank(id, type, options) {
    const skillType = String(type || '').toLowerCase();
    const category = inferSkillCategory(id, type, options);
    if (category === 'mobility' || category === 'buff') return 5;
    if (category === 'passive') return 10;
    if (skillType.includes('setup') || skillType.includes('debuff') || skillType.includes('finisher') || skillType.includes('ultimate') || skillType.includes('resource') || skillType.includes('utility')) return 10;
    return 20;
  }

  function projectileTargeting(config) {
    return { mode: 'projectile', ...(config || {}) };
  }

  function skillVisual(kind, color, accent, options) {
    return Object.freeze(Object.assign({
      kind,
      color,
      accent,
      glow: 18,
      trail: 'spark',
      impact: 'spark'
    }, options || {}));
  }

  const SKILL_VISUALS = Object.freeze({
    arcaneBolt: skillVisual('orb', '#69c8ff', '#e7fbff', { trail: 'star', impact: 'arcaneRing' }),
    arcaneBurst: skillVisual('orbBurst', '#8f8cff', '#ffffff', { glow: 24, trail: 'comet', impact: 'burstRing' }),
    spellMark: skillVisual('markOrb', '#b794f4', '#fff2ff', { trail: 'runeDust', impact: 'markSigil' }),
    energyRelease: skillVisual('comet', '#7bdff2', '#fff7b8', { glow: 28, trail: 'comet', impact: 'burstRing' }),
    fireball: skillVisual('fireball', '#ff7b3a', '#ffd166', { glow: 28, trail: 'ember', impact: 'flameBloom' }),
    burningMark: skillVisual('fireMark', '#ff5a3d', '#ffd166', { trail: 'ember', impact: 'brand' }),
    wildfire: skillVisual('wildfire', '#ff8c3a', '#fff0a6', { glow: 30, trail: 'ember', impact: 'flameBloom' }),
    infernoBurst: skillVisual('inferno', '#ff3d2e', '#ffe16a', { glow: 34, trail: 'ember', impact: 'infernoBloom' }),
    heatVent: skillVisual('flameCone', '#ff7b3a', '#ffe16a', { trail: 'heat', impact: 'flameBloom' }),
    runeMark: skillVisual('runeBolt', '#28c7b7', '#b8fff2', { trail: 'runeDust', impact: 'runeSeal' }),
    runeLink: skillVisual('runeLink', '#37d6c7', '#f0fffb', { trail: 'runeDust', impact: 'runeLink' }),
    manaSeal: skillVisual('seal', '#4cc7ff', '#e7fbff', { trail: 'runeDust', impact: 'runeSeal' }),
    groundGlyph: skillVisual('glyph', '#28c7b7', '#b8fff2', { impact: 'groundGlyph' }),
    runeDetonation: skillVisual('runeBurst', '#28c7b7', '#ffffff', { glow: 30, impact: 'runeBurst' }),
    grandInscription: skillVisual('grandRune', '#38e6d0', '#ffffff', { glow: 34, impact: 'grandRune' }),
    chainBolt: skillVisual('lightning', '#d8f6ff', '#7aa7ff', { glow: 26, trail: 'lightning', impact: 'staticFork' }),
    quickShot: skillVisual('arrow', '#f2d273', '#ffffff', { trail: 'line', impact: 'arrowChip' }),
    markedShot: skillVisual('markedArrow', '#ffe16a', '#ffef99', { trail: 'line', impact: 'markReticle' }),
    piercingArrow: skillVisual('piercingArrow', '#d8c25f', '#ffffff', { trail: 'pierceLine', impact: 'arrowPierce' }),
    focusedVolley: skillVisual('volleyArrow', '#ffd166', '#ffffff', { trail: 'line', impact: 'volleySpark' }),
    aimedShot: skillVisual('sniperTracer', '#e9f7ff', '#ffd166', { glow: 16, trail: 'tracer', impact: 'reticleHit' }),
    weakPointMark: skillVisual('weakPointArrow', '#ffd166', '#ffffff', { trail: 'tracer', impact: 'weakPoint' }),
    pierceArmor: skillVisual('armorPierce', '#b7c3ca', '#ffe16a', { trail: 'pierceLine', impact: 'armorCrack' }),
    executionShot: skillVisual('executionTracer', '#ffef99', '#ffffff', { glow: 24, trail: 'tracer', impact: 'executionFlash' }),
    perfectShot: skillVisual('perfectShot', '#ffffff', '#ffd166', { glow: 30, trail: 'tracer', impact: 'perfectStar' }),
    lureShot: skillVisual('lureArrow', '#66d79a', '#dbffe6', { trail: 'line', impact: 'lurePulse' }),
    trapSnare: skillVisual('trapSnare', '#66d79a', '#dbffe6', { impact: 'trapCircle' }),
    trapSpike: skillVisual('trapSpike', '#b07a47', '#ffd166', { impact: 'spikeTrap' }),
    trapTripwire: skillVisual('tripwire', '#7bdff2', '#ffffff', { impact: 'tripwire' }),
    trapDetonate: skillVisual('trapDetonate', '#ffbe55', '#fff0a6', { impact: 'trapBurst' }),
    killZone: skillVisual('killZone', '#66d79a', '#ffffff', { impact: 'killZone' }),
    companionStrike: skillVisual('companionArrow', '#8ed174', '#fff0a6', { trail: 'line', impact: 'clawSpark' })
  });

  const SKILL_VISUAL_IDS = Object.freeze({
    mage_magic_bolt: 'arcaneBolt',
    mage_arcane_burst: 'arcaneBurst',
    mage_spell_mark: 'spellMark',
    mage_energy_release: 'energyRelease',
    fire_mage_fireball: 'fireball',
    fire_mage_burning_mark: 'burningMark',
    fire_mage_heat_vent: 'heatVent',
    fire_mage_wildfire: 'wildfire',
    fire_mage_inferno_burst: 'infernoBurst',
    rune_mage_rune_mark: 'runeMark',
    rune_mage_ground_glyph: 'groundGlyph',
    rune_mage_arcane_link: 'runeLink',
    rune_mage_rune_detonation: 'runeDetonation',
    rune_mage_mana_seal: 'manaSeal',
    rune_mage_grand_inscription: 'grandInscription',
    storm_mage_chain_bolt: 'chainBolt',
    archer_quick_shot: 'quickShot',
    archer_marked_shot: 'markedShot',
    archer_piercing_arrow: 'piercingArrow',
    archer_focused_volley: 'focusedVolley',
    sniper_aimed_shot: 'aimedShot',
    sniper_weak_point_mark: 'weakPointMark',
    sniper_pierce_armor: 'pierceArmor',
    sniper_execution_shot: 'executionShot',
    sniper_one_perfect_shot: 'perfectShot',
    trapper_snare_trap: 'trapSnare',
    trapper_spike_trap: 'trapSpike',
    trapper_lure_shot: 'lureShot',
    trapper_tripwire: 'trapTripwire',
    trapper_detonate: 'trapDetonate',
    trapper_kill_zone: 'killZone',
    beast_archer_companion_strike: 'companionStrike'
  });

  function skill(id, name, owner, batch, type, roleTags, prerequisites, description, options) {
    const config = options || {};
    const category = inferSkillCategory(id, type, config);
    return Object.freeze({
      id,
      name,
      owner,
      batch,
      type,
      category,
      roleTags,
      prerequisites: prerequisites || [],
      maxRank: config.maxRank || inferSkillMaxRank(id, type, config),
      defaultRank: Object.prototype.hasOwnProperty.call(config, 'defaultRank') ? Math.max(0, Math.floor(Number(config.defaultRank || 0) || 0)) : null,
      resourceCost: config.resourceCost || 0,
      cooldown: config.cooldown || 0,
      lineCount: Math.max(0, Math.floor(Number(config.lineCount || 0) || 0)),
      lineDamageScale: Number(config.lineDamageScale || 1) || 1,
      iconAsset: config.iconAsset || BASE_SKILL_ICONS[id] || ADVANCED_SKILL_ICONS[id] || '',
      iconKind: config.iconKind || inferSkillIconKind(id, type, roleTags),
      visualId: config.visualId || SKILL_VISUAL_IDS[id] || '',
      purpose: inferSkillPurpose(id, type, roleTags, config),
      primaryTraining: !!config.primaryTraining,
      passiveStats: config.passiveStats ? Object.freeze({ ...config.passiveStats }) : null,
      movementEffect: config.movementEffect ? Object.freeze({ ...config.movementEffect }) : null,
      targeting: config.targeting ? Object.freeze({ ...config.targeting }) : null,
      targetCaps: config.targetCaps ? Object.freeze({ ...config.targetCaps }) : null,
      partyEffect: config.partyEffect || '',
      futurePartyEffect: config.futurePartyEffect || '',
      description
    });
  }

  function toCombatFxFileId(value) {
    return String(value || '')
      .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
      .replace(/[_\s]+/g, '-')
      .replace(/[^a-zA-Z0-9-]+/g, '')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .toLowerCase();
  }

  const SKILLS = Object.freeze([
    skill('fighter_heavy_strike', 'Heavy Strike', 'fighter', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer strike for close-range Momentum. Repeated hits on one target add extra stagger.', { resourceCost: 6, cooldown: 0.34, lineCount: 1 }),
    skill('fighter_dash_slash', 'Dash Slash', 'fighter', 'Base Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'fighter_heavy_strike', rank: 3 }], 'Mobility slash for crossing gaps, dodging through lanes, and clipping enemies while moving.', { resourceCost: 16, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 250, distancePerRank: 5, duration: 0.2, damageRadius: 96, hitOffset: 124, invulnerable: 0.16 } }),
    skill('fighter_guard', 'Guard', 'fighter', 'Base Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'fighter_heavy_strike', rank: 3 }], 'Defensive window that reduces incoming damage and turns pressure into Momentum.', { resourceCost: 14, cooldown: 6 }),
    skill('fighter_ground_slam', 'Ground Slam', 'fighter', 'Base Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'fighter_heavy_strike', rank: 5 }], 'Mobbing slam that staggers clustered enemies and rewards hitting three or more targets.', { resourceCost: 28, cooldown: 5, lineCount: 2 }),
    skill('fighter_power_break', 'Power Break', 'fighter', 'Base Skill Batch', 'Debuff', ['Bossing'], [{ skillId: 'fighter_ground_slam', rank: 3, any: true }, { skillId: 'fighter_heavy_strike', rank: 5, any: true }], 'Setup strike that cracks a durable target so later attacks hit harder.', { resourceCost: 22, cooldown: 6, lineCount: 1 }),
    skill('fighter_momentum_burst', 'Momentum Burst', 'fighter', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'fighter_heavy_strike', rank: 5 }, { anyOf: ['fighter_dash_slash', 'fighter_ground_slam', 'fighter_power_break'], rank: 3 }], 'Momentum finisher that spends stored resource for a stronger shockwave payoff.', { resourceCost: 55, cooldown: 8, lineCount: 2 }),
    skill('fighter_damage_mastery', 'Fighter Damage Mastery', 'fighter', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Fighter damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

    skill('mage_magic_bolt', 'Magic Bolt', 'mage', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer projectile that safely builds Energy at range.', { resourceCost: 7, cooldown: 0.44, lineCount: 1, targeting: projectileTargeting({ projectileType: 'magic', range: 480, rangePerRank: 4, speed: 620 }) }),
    skill('mage_blink', 'Blink', 'mage', 'Base Skill Batch', 'Mobility', ['Support'], [{ skillId: 'mage_magic_bolt', rank: 3 }], 'Mobility teleport for changing platforms or escaping pressure without adding burst damage.', { resourceCost: 12, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 190, distancePerRank: 4, duration: 0, damageRadius: 68, hitOffset: 88, invulnerable: 0.24 } }),
    skill('mage_arcane_burst', 'Arcane Burst', 'mage', 'Base Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'mage_magic_bolt', rank: 5 }], 'Mobbing blast that clears nearby enemies more efficiently than Magic Bolt.', { resourceCost: 24, cooldown: 5, lineCount: 2, targeting: projectileTargeting({ projectileType: 'magic', range: 455, rangePerRank: 5, speed: 560, explodeRadius: 108, explodeRadiusPerRank: 3 }) }),
    skill('mage_mana_shield', 'Mana Shield', 'mage', 'Base Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'mage_magic_bolt', rank: 3 }], 'Defensive conversion that spends MP to absorb incoming damage.', { resourceCost: 22, cooldown: 8 }),
    skill('mage_spell_mark', 'Spell Mark', 'mage', 'Base Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'mage_magic_bolt', rank: 5 }], 'Setup mark that makes the next major spell payoff stronger.', { resourceCost: 18, cooldown: 5, lineCount: 1, targeting: projectileTargeting({ projectileType: 'magic', range: 500, rangePerRank: 4, speed: 640, applyMark: true }) }),
    skill('mage_energy_release', 'Energy Release', 'mage', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'mage_arcane_burst', rank: 5, any: true }, { skillId: 'mage_spell_mark', rank: 5, any: true }], 'Energy finisher that detonates around marked targets and spends stored resource for scale.', { resourceCost: 60, cooldown: 8, lineCount: 2, targeting: projectileTargeting({ projectileType: 'magic', range: 540, rangePerRank: 5, speed: 660, explodeRadius: 124, explodeRadiusPerRank: 3 }) }),
    skill('mage_damage_mastery', 'Mage Damage Mastery', 'mage', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

    skill('archer_quick_shot', 'Quick Shot', 'archer', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer shot that builds Focus quickly without needing setup.', { resourceCost: 6, cooldown: 0.38, lineCount: 2, targeting: projectileTargeting({ projectileType: 'arrow', range: 560, rangePerRank: 5, speed: 780, count: 2, spreadY: 6 }) }),
    skill('archer_roll_shot', 'Roll Shot', 'archer', 'Base Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'archer_quick_shot', rank: 3 }], 'Mobility roll that creates a firing lane instead of replacing damage skills.', { resourceCost: 14, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 170, distancePerRank: 3, duration: 0.24, damageRadius: 72, hitOffset: 92, invulnerable: 0.18 } }),
    skill('archer_marked_shot', 'Marked Shot', 'archer', 'Base Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'archer_quick_shot', rank: 3 }], 'Setup shot that marks a priority enemy for Focus spender payoff.', { resourceCost: 14, cooldown: 4, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 585, rangePerRank: 6, speed: 760, applyMark: true }) }),
    skill('archer_piercing_arrow', 'Piercing Arrow', 'archer', 'Base Skill Batch', 'Damage', ['Mobbing'], [{ skillId: 'archer_quick_shot', rank: 5 }], 'Mobbing arrow that travels through enemies lined up in a lane.', { resourceCost: 24, cooldown: 5, lineCount: 3, targeting: projectileTargeting({ projectileType: 'arrow', range: 640, rangePerRank: 7, speed: 790, pierce: 4 }) }),
    skill('archer_eagle_stance', 'Eagle Stance', 'archer', 'Base Skill Batch', 'Buff', ['Bossing'], [{ skillId: 'archer_marked_shot', rank: 3, any: true }, { skillId: 'archer_piercing_arrow', rank: 3, any: true }], 'Bossing buff window for longer range, better crits, and cleaner marked-target pressure.', { resourceCost: 18, cooldown: 10 }),
    skill('archer_focused_volley', 'Focused Volley', 'archer', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'archer_marked_shot', rank: 5, any: true }, { skillId: 'archer_piercing_arrow', rank: 5, any: true }], 'Focus finisher that concentrates multiple damage lines into a marked target.', { resourceCost: 55, cooldown: 8, lineCount: 5, targeting: projectileTargeting({ projectileType: 'arrow', range: 620, rangePerRank: 6, speed: 800, count: 5, spreadY: 10 }) }),
    skill('archer_damage_mastery', 'Archer Damage Mastery', 'archer', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Archer damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

    skill('guardian_shield_bash', 'Shield Bash', 'guardian', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Control'], [], 'Low-cooldown guarded strike for Guardian training. Staggers enemies and builds Stored Impact for larger counters.', { resourceCost: 4, cooldown: 0.48, lineCount: 1, lineDamageScale: 1.04, primaryTraining: true }),
    skill('guardian_damage_mastery', 'Guardian Damage Mastery', 'guardian', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Guardian damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('guardian_shield_dash', 'Shield Dash', 'guardian', 'Advanced Skill Batch', 'Mobility / control', ['Hybrid', 'Control'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'guardian_shield_bash', rank: 3 }], 'Dash forward behind your shield to claim space or cross short gaps.', { resourceCost: 22, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 300, distancePerRank: 6, duration: 0.24, damageRadius: 112, hitOffset: 118, invulnerable: 0.24 } }),
    skill('guardian_impact_guard', 'Impact Guard', 'guardian', 'Advanced Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'fighter_guard', rank: 5 }], 'Block incoming damage and convert part of it into Stored Impact.', { resourceCost: 18, cooldown: 7 }),
    skill('guardian_oath_barrier', 'Oath Barrier', 'guardian', 'Advanced Skill Batch', 'Defense', ['Support'], [{ skillId: 'guardian_impact_guard', rank: 5 }], 'Create a temporary shield for yourself, stronger with Stored Impact.', { resourceCost: 25, cooldown: 12 }),
    skill('guardian_retaliation_wave', 'Retaliation Wave', 'guardian', 'Advanced Skill Batch', 'Counterattack', ['Mobbing', 'Hybrid'], [{ skillId: 'guardian_impact_guard', rank: 5 }, { skillId: 'fighter_ground_slam', rank: 5 }], 'Release Stored Impact in a forward shockwave.', { resourceCost: 40, cooldown: 8, lineCount: 2 }),
    skill('guardian_hold_the_line', 'Hold the Line', 'guardian', 'Advanced Skill Batch', 'Passive', ['Support', 'Bossing'], [{ skillId: 'guardian_impact_guard', rank: 5 }], 'Permanently improve block, defense, and Momentum control while holding ground.', { cooldown: 0, passiveStats: { defense: 0.8, block: 0.8, resourceGain: 0.4 } }),
    skill('guardian_verdict', 'Guardian\'s Verdict', 'guardian', 'Advanced Skill Batch', 'Finisher', ['Bossing', 'Party'], [{ skillId: 'guardian_retaliation_wave', rank: 5 }, { skillId: 'fighter_guard', rank: 5 }], 'Spend all Stored Impact to create a defensive explosion and personal shield.', { resourceCost: 70, cooldown: 16, lineCount: 3 }),
    skill('guardian_shield_wall', 'Shield Wall', 'guardian', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'guardian_impact_guard', rank: 3 }], 'Self-buff prototype: damage reduction, shield, knockback resistance, and Momentum when hit.', { resourceCost: 45, cooldown: 60, partyEffect: 'Self: 20% damage reduction and temporary shield.', futurePartyEffect: 'Future party: nearby allies gain 10% damage reduction, shield, and reduced knockback.' }),

    skill('berserker_blood_cleave', 'Blood Cleave', 'berserker', 'Advanced Skill Batch', 'Primary training attack', ['Bossing', 'Hybrid'], [], 'Low-cooldown risk cleave for Berserker training. Damage rises as HP drops and is best against durable targets.', { resourceCost: 5, cooldown: 0.6, lineCount: 2, lineDamageScale: 0.72, primaryTraining: true }),
    skill('berserker_damage_mastery', 'Berserker Damage Mastery', 'berserker', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Berserker damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('berserker_rage_surge', 'Rage Surge', 'berserker', 'Advanced Skill Batch', 'Buff', ['Bossing'], [{ skillId: 'berserker_blood_cleave', rank: 3 }], 'Increase attack speed and damage while reducing defense.', { resourceCost: 20, cooldown: 12 }),
    skill('berserker_reckless_leap', 'Reckless Leap', 'berserker', 'Advanced Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'berserker_blood_cleave', rank: 3 }], 'Leap aggressively across gaps or into position.', { resourceCost: 24, cooldown: 0.5, movementEffect: { mode: 'leap', distance: 330, distancePerRank: 7, duration: 0.34, verticalVelocity: -420, damageRadius: 128, hitOffset: 140, invulnerable: 0.26 } }),
    skill('berserker_crimson_recovery', 'Crimson Recovery', 'berserker', 'Advanced Skill Batch', 'Sustain', ['Bossing', 'Support'], [{ skillId: 'berserker_rage_surge', rank: 5 }], 'Damage enemies and heal for a portion of damage dealt.', { resourceCost: 34, cooldown: 10, lineCount: 2 }),
    skill('berserker_pain_to_power', 'Pain to Power', 'berserker', 'Advanced Skill Batch', 'Passive', ['Bossing'], [{ skillId: 'berserker_blood_cleave', rank: 5 }], 'Permanently improve power and Rage generation as wounds become fuel.', { cooldown: 0, passiveStats: { power: 0.8, resourceGain: 0.9 } }),
    skill('berserker_last_stand', 'Last Stand', 'berserker', 'Advanced Skill Batch', 'Finisher', ['Bossing'], [{ skillId: 'berserker_pain_to_power', rank: 5 }, { skillId: 'fighter_momentum_burst', rank: 10 }], 'Lethal damage leaves you at 1 HP briefly and greatly increases damage.', { resourceCost: 70, cooldown: 18, lineCount: 3 }),
    skill('berserker_war_cry', 'War Cry', 'berserker', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'berserker_rage_surge', rank: 3 }], 'Self-buff prototype: attack power, Rage generation, and small lifesteal.', { resourceCost: 48, cooldown: 75, partyEffect: 'Self: 18% attack power, Rage generation, and capped lifesteal.', futurePartyEffect: 'Future party: nearby allies gain 8% attack power, minor healing on hit, and resource generation.' }),

    skill('duelist_quick_cut', 'Quick Cut', 'duelist', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Bossing'], [], 'Low-cooldown precise cut for Duelist training. Repeated hits on one target build Tempo.', { resourceCost: 4, cooldown: 0.46, lineCount: 3, lineDamageScale: 0.58, primaryTraining: true }),
    skill('duelist_damage_mastery', 'Duelist Damage Mastery', 'duelist', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Duelist damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('duelist_flash_step', 'Flash Step', 'duelist', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'duelist_quick_cut', rank: 3 }], 'Dash through a short opening while preserving Tempo for follow-up attacks.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 315, distancePerRank: 8, duration: 0.16, damageRadius: 74, hitOffset: 112, invulnerable: 0.24 } }),
    skill('duelist_rallying_flourish', 'Rallying Flourish', 'duelist', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'duelist_quick_cut', rank: 3 }], 'Self-buff prototype: haste, precision chance, and Tempo generation after chained attacks.', { resourceCost: 44, cooldown: 65, partyEffect: 'Self: movement speed, precision chance, and faster Tempo generation.', futurePartyEffect: 'Future party: nearby allies gain minor haste and precision chance during burst windows.' }),

    skill('fire_mage_fireball', 'Fireball', 'fireMage', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Mobbing'], [], 'Low-cooldown fire projectile for Fire Mage training. Explodes on impact and starts burn spread setups.', { resourceCost: 5, cooldown: 0.62, lineCount: 2, lineDamageScale: 0.76, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'fire', range: 500, rangePerRank: 5, speed: 560, explodeRadius: 92, explodeRadiusPerRank: 3, applyBurn: true }) }),
    skill('fire_mage_damage_mastery', 'Fire Mage Damage Mastery', 'fireMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Fire Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('fire_mage_flame_trail', 'Flame Trail', 'fireMage', 'Advanced Skill Batch', 'Movement effect', ['Mobbing', 'Support'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'fire_mage_fireball', rank: 3 }], 'Dash forward and leave a burning trail behind.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 285, distancePerRank: 7, duration: 0.22, damageRadius: 102, hitOffset: 104, invulnerable: 0.22, trail: 'flame' } }),
    skill('fire_mage_burning_mark', 'Burning Mark', 'fireMage', 'Advanced Skill Batch', 'Damage-over-time', ['Bossing', 'Hybrid'], [{ skillId: 'mage_spell_mark', rank: 5 }, { skillId: 'fire_mage_fireball', rank: 3 }], 'Mark an enemy with burn. Fire skills spread the burn.', { resourceCost: 18, cooldown: 6, lineCount: 1, targeting: projectileTargeting({ projectileType: 'fire', range: 510, rangePerRank: 4, speed: 610, applyMark: true, applyBurn: true }) }),
    skill('fire_mage_heat_vent', 'Heat Vent', 'fireMage', 'Advanced Skill Batch', 'Resource control', ['Hybrid'], [{ skillId: 'fire_mage_fireball', rank: 5 }], 'Release Heat in a cone of flame, preventing Overheat.', { resourceCost: 30, cooldown: 8, lineCount: 2 }),
    skill('fire_mage_wildfire', 'Wildfire', 'fireMage', 'Advanced Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'fire_mage_burning_mark', rank: 5 }, { skillId: 'mage_arcane_burst', rank: 5 }], 'Burn spreads between nearby enemies.', { resourceCost: 35, cooldown: 10, lineCount: 3, targeting: projectileTargeting({ projectileType: 'fire', range: 500, rangePerRank: 4, speed: 560, explodeRadius: 148, explodeRadiusPerRank: 4, applyBurn: true }) }),
    skill('fire_mage_inferno_burst', 'Inferno Burst', 'fireMage', 'Advanced Skill Batch', 'Finisher', ['Hybrid', 'Bossing'], [{ skillId: 'fire_mage_heat_vent', rank: 5 }, { skillId: 'mage_energy_release', rank: 10 }], 'Consume all Heat for a large explosion. At max Heat, harms the caster slightly.', { resourceCost: 75, cooldown: 16, lineCount: 3, targeting: projectileTargeting({ projectileType: 'fire', range: 470, rangePerRank: 4, speed: 530, explodeRadius: 168, explodeRadiusPerRank: 5, applyBurn: true }) }),
    skill('fire_mage_ignition_aura', 'Ignition Aura', 'fireMage', 'Advanced Skill Batch', 'Party skill', ['Mobbing', 'Party'], [{ skillId: 'fire_mage_fireball', rank: 3 }], 'Self-buff prototype: fire damage, burn power, Heat generation, and longer burns.', { resourceCost: 52, cooldown: 75, partyEffect: 'Self: 20% increased fire and burn damage, faster Heat, and stronger burn uptime.', futurePartyEffect: 'Future party: nearby allies deal 8% bonus damage against burning enemies and can apply minor burn.' }),

    skill('rune_mage_rune_mark', 'Rune Mark', 'runeMage', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Control'], [], 'Low-cooldown rune bolt for Rune Mage training. Marks and links targets for later detonations.', { resourceCost: 4, cooldown: 0.58, lineCount: 1, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'rune', range: 510, rangePerRank: 5, speed: 620, applyMark: true, applySlow: true }) }),
    skill('rune_mage_damage_mastery', 'Rune Mage Damage Mastery', 'runeMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Rune Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('rune_mage_rune_blink', 'Rune Blink', 'runeMage', 'Advanced Skill Batch', 'Mobility', ['Support', 'Hybrid'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'rune_mage_rune_mark', rank: 3 }], 'Blink through a short rune fold to reposition without breaking setup flow.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 285, distancePerRank: 7, duration: 0, damageRadius: 72, hitOffset: 88, invulnerable: 0.3 } }),
    skill('rune_mage_ground_glyph', 'Ground Glyph', 'runeMage', 'Advanced Skill Batch', 'Area setup', ['Mobbing', 'Control'], [{ skillId: 'rune_mage_rune_mark', rank: 3, any: true }, { skillId: 'mage_arcane_burst', rank: 5, any: true }], 'Place a wide rune field that damages enemies, slows and weakens enemies inside it, and grants modest recovery and casting haste while you stand within it.', { resourceCost: 32, cooldown: 9, lineCount: 2, targetCaps: { area: 6, field: 6 } }),
    skill('rune_mage_arcane_link', 'Arcane Link', 'runeMage', 'Advanced Skill Batch', 'Combo', ['Bossing', 'Hybrid'], [{ skillId: 'rune_mage_rune_mark', rank: 5 }], 'Link two runes so damage to one partially affects another.', { resourceCost: 24, cooldown: 7, lineCount: 2, targeting: projectileTargeting({ projectileType: 'rune', range: 520, rangePerRank: 5, speed: 620, count: 2, spreadY: 12, applyMark: true }) }),
    skill('rune_mage_rune_detonation', 'Rune Detonation', 'runeMage', 'Advanced Skill Batch', 'Burst', ['Hybrid'], [{ skillId: 'rune_mage_ground_glyph', rank: 5, any: true }, { skillId: 'rune_mage_arcane_link', rank: 5, any: true }], 'Detonate active runes for burst damage.', { resourceCost: 35, cooldown: 9, lineCount: 3, targetCaps: { runeDetonation: 6 } }),
    skill('rune_mage_mana_seal', 'Mana Seal', 'runeMage', 'Advanced Skill Batch', 'Control', ['Bossing', 'Support'], [{ skillId: 'rune_mage_rune_mark', rank: 5 }, { skillId: 'mage_mana_shield', rank: 5 }], 'Seal an enemy briefly, reducing movement or casting.', { resourceCost: 28, cooldown: 12, lineCount: 1, targeting: projectileTargeting({ projectileType: 'rune', range: 500, rangePerRank: 4, speed: 600, applySlow: true, applyMark: true }) }),
    skill('rune_mage_grand_inscription', 'Grand Inscription', 'runeMage', 'Advanced Skill Batch', 'Finisher', ['Mobbing', 'Support'], [{ skillId: 'rune_mage_rune_detonation', rank: 5 }, { skillId: 'mage_energy_release', rank: 10 }], 'Place a massive inscription field with stronger recovery, haste, enemy slow, and Rune Mark explosion bonuses while you stand within it.', { resourceCost: 88, cooldown: 24, lineCount: 3, targetCaps: { finisherArea: 8, field: 8 } }),
    skill('rune_mage_rune_circle', 'Rune Circle', 'runeMage', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'rune_mage_ground_glyph', rank: 3 }], 'Self-buff prototype with selectable Power, Guard, Focus, Cleanse, or Haste rune modes.', { resourceCost: 45, cooldown: 60, partyEffect: 'Self: selected rune grants skill damage, reduction, resource generation, cleanse, or haste.', futurePartyEffect: 'Future party: allies inside the circle receive reduced selected rune effects.' }),

    skill('storm_mage_chain_bolt', 'Chain Bolt', 'stormMage', 'Advanced Skill Batch', 'Primary training attack', ['Mobbing', 'Hybrid'], [], 'Low-cooldown lightning chain for Storm Mage training. Efficiently clears clustered enemies but has lower single-target boss value.', { resourceCost: 6, cooldown: 0.78, lineCount: 3, lineDamageScale: 0.58, primaryTraining: true, targeting: projectileTargeting({ mode: 'chain', projectileType: 'lightning', range: 380, rangePerRank: 4, speed: 720, chainRange: 220, chainRangePerRank: 6, chainTargets: 3, chainTargetsPerRanks: 2, maxChainTargets: 8, chainDamageFalloff: 0.92, applySlow: true }) }),
    skill('storm_mage_damage_mastery', 'Storm Mage Damage Mastery', 'stormMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Storm Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('storm_mage_static_shift', 'Static Shift', 'stormMage', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'storm_mage_chain_bolt', rank: 3 }], 'Blink through static current to reposition without breaking spell flow.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 300, distancePerRank: 8, duration: 0, damageRadius: 70, hitOffset: 88, invulnerable: 0.3 } }),
    skill('storm_mage_stormfront', 'Stormfront', 'stormMage', 'Advanced Skill Batch', 'Party skill', ['Mobbing', 'Party'], [{ skillId: 'storm_mage_chain_bolt', rank: 3 }], 'Self-buff prototype: faster Charge generation, bonus area damage, and lightning chain reach.', { resourceCost: 50, cooldown: 70, partyEffect: 'Self: Charge generation, area damage, and lightning reach.', futurePartyEffect: 'Future party: nearby allies gain minor skill haste and bonus damage to shocked enemies.' }),

    skill('sniper_aimed_shot', 'Aimed Shot', 'sniper', 'Advanced Skill Batch', 'Primary training attack', ['Bossing'], [], 'Low-cooldown precision shot for Sniper training. Strongest when held at range or aimed into weak points.', { resourceCost: 5, cooldown: 0.65, lineCount: 1, lineDamageScale: 0.92, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'arrow', range: 720, rangePerRank: 8, speed: 900 }) }),
    skill('sniper_damage_mastery', 'Sniper Damage Mastery', 'sniper', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Sniper damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('sniper_combat_roll', 'Combat Roll', 'sniper', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'sniper_aimed_shot', rank: 3 }], 'Roll backward into a clean firing lane while briefly avoiding damage.', { resourceCost: 16, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 245, distancePerRank: 5, duration: 0.24, damageRadius: 74, hitOffset: 90, invulnerable: 0.25 } }),
    skill('sniper_weak_point_mark', 'Weak Point Mark', 'sniper', 'Advanced Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'archer_marked_shot', rank: 5 }], 'Mark a target weak point. The next heavy shot deals bonus damage.', { resourceCost: 18, cooldown: 6, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 725, rangePerRank: 8, speed: 880, applyMark: true }) }),
    skill('sniper_steady_breath', 'Steady Breath', 'sniper', 'Advanced Skill Batch', 'Passive', ['Bossing'], [{ skillId: 'sniper_aimed_shot', rank: 5 }], 'Permanently improve range, precision damage, and steady Aim control.', { cooldown: 0, passiveStats: { range: 4, critDamage: 2, resourceGain: 0.4 } }),
    skill('sniper_pierce_armor', 'Pierce Armor', 'sniper', 'Advanced Skill Batch', 'Debuff', ['Bossing', 'Support'], [{ skillId: 'sniper_weak_point_mark', rank: 5 }, { skillId: 'archer_piercing_arrow', rank: 5 }], 'Shot that lowers enemy defense, stronger against marked enemies.', { resourceCost: 26, cooldown: 7, lineCount: 3, targeting: projectileTargeting({ projectileType: 'arrow', range: 730, rangePerRank: 8, speed: 860, pierce: 3, applyCrack: true }) }),
    skill('sniper_execution_shot', 'Execution Shot', 'sniper', 'Advanced Skill Batch', 'Finisher', ['Bossing'], [{ skillId: 'sniper_weak_point_mark', rank: 5 }, { skillId: 'sniper_aimed_shot', rank: 5 }], 'Massive damage against low-health or weak-point-marked enemies.', { resourceCost: 45, cooldown: 12, lineCount: 4, targeting: projectileTargeting({ projectileType: 'arrow', range: 720, rangePerRank: 8, speed: 890, pierce: 1 }) }),
    skill('sniper_one_perfect_shot', 'One Perfect Shot', 'sniper', 'Advanced Skill Batch', 'Ultimate-style', ['Bossing'], [{ skillId: 'sniper_execution_shot', rank: 5 }, { skillId: 'archer_focused_volley', rank: 10 }], 'Consume all Aim to fire a huge single-target attack.', { resourceCost: 80, cooldown: 18, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 790, rangePerRank: 10, speed: 940, width: 36, height: 12 }) }),
    skill('sniper_eagle_eye', 'Eagle Eye', 'sniper', 'Advanced Skill Batch', 'Party skill', ['Bossing', 'Party'], [{ skillId: 'sniper_weak_point_mark', rank: 3 }], 'Self-buff prototype: precision chance, precision damage, range, and Aim generation.', { resourceCost: 48, cooldown: 75, partyEffect: 'Self: 15% precision chance, 20% precision damage, increased range, and Aim against marked enemies.', futurePartyEffect: 'Future party: nearby allies gain 8% precision chance, 10% precision damage, and weak-point payoff.' }),

    skill('trapper_snare_trap', 'Snare Trap', 'trapper', 'Advanced Skill Batch', 'Primary training attack', ['Mobbing', 'Control'], [], 'Low-cooldown trap for Trapper training. Arms quickly, slows enemies, and sets up manual detonations.', { resourceCost: 4, cooldown: 0.52, lineCount: 1, lineDamageScale: 0.82, primaryTraining: true }),
    skill('trapper_damage_mastery', 'Trapper Damage Mastery', 'trapper', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Trapper damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('trapper_grapple_dash', 'Grapple Dash', 'trapper', 'Advanced Skill Batch', 'Mobility / utility', ['Hybrid', 'Control'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'trapper_snare_trap', rank: 3 }], 'Fire a short grapple line and dash into a better trap angle.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 285, distancePerRank: 7, duration: 0.25, damageRadius: 82, hitOffset: 110, invulnerable: 0.22 } }),
    skill('trapper_spike_trap', 'Spike Trap', 'trapper', 'Advanced Skill Batch', 'Damage', ['Mobbing'], [{ skillId: 'trapper_snare_trap', rank: 3 }], 'Place a trap that deals damage when stepped on.', { resourceCost: 20, cooldown: 6, lineCount: 3 }),
    skill('trapper_lure_shot', 'Lure Shot', 'trapper', 'Advanced Skill Batch', 'Utility', ['Control'], [{ skillId: 'archer_marked_shot', rank: 5 }, { skillId: 'trapper_snare_trap', rank: 3 }], 'Fire a shot that draws enemies toward a target area.', { resourceCost: 18, cooldown: 7, lineCount: 2, targeting: projectileTargeting({ projectileType: 'arrow', range: 590, rangePerRank: 6, speed: 760, applySlow: true, applyMark: true }) }),
    skill('trapper_tripwire', 'Tripwire', 'trapper', 'Advanced Skill Batch', 'Combo trap', ['Mobbing', 'Control'], [{ skillId: 'trapper_snare_trap', rank: 5 }, { skillId: 'trapper_spike_trap', rank: 3 }], 'Place a line trap that triggers when enemies cross it.', { resourceCost: 25, cooldown: 8, lineCount: 3 }),
    skill('trapper_detonate', 'Detonate', 'trapper', 'Advanced Skill Batch', 'Trigger', ['Hybrid'], [{ skillId: 'trapper_spike_trap', rank: 5, any: true }, { skillId: 'trapper_tripwire', rank: 5, any: true }], 'Manually trigger active traps.', { resourceCost: 24, cooldown: 7, lineCount: 3, targetCaps: { trapDetonate: 8 } }),
    skill('trapper_kill_zone', 'Kill Zone', 'trapper', 'Advanced Skill Batch', 'Finisher', ['Mobbing'], [{ skillId: 'trapper_detonate', rank: 5 }, { skillId: 'archer_focused_volley', rank: 10 }], 'Place a large trap field that chains trap activations.', { resourceCost: 70, cooldown: 18, lineCount: 5 }),
    skill('trapper_tactical_field', 'Tactical Field', 'trapper', 'Advanced Skill Batch', 'Party skill', ['Control', 'Party'], [{ skillId: 'trapper_snare_trap', rank: 3 }], 'Self-buff prototype: trap damage, arming speed, damage reduction inside field, and Focus on trap trigger.', { resourceCost: 46, cooldown: 70, partyEffect: 'Self: 15% trap damage, faster arming speed, field mitigation, and Focus from traps.', futurePartyEffect: 'Future party: allies gain damage reduction from enemies inside the field and bonus damage to controlled enemies.' }),

    skill('beast_archer_companion_strike', 'Companion Strike', 'beastArcher', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Support'], [], 'Low-cooldown coordinated shot for Beast Archer training. Marks targets for companion pressure and sustain.', { resourceCost: 5, cooldown: 0.58, lineCount: 3, lineDamageScale: 0.62, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'arrow', range: 575, rangePerRank: 6, speed: 780, count: 2, spreadY: 14, applyMark: true }) }),
    skill('beast_archer_damage_mastery', 'Beast Archer Damage Mastery', 'beastArcher', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Beast Archer damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
    skill('beast_archer_pounce_roll', 'Pounce Roll', 'beastArcher', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'beast_archer_companion_strike', rank: 3 }], 'Roll into a pounce lane, repositioning while your companion covers the retreat.', { resourceCost: 17, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 255, distancePerRank: 5, duration: 0.24, damageRadius: 72, hitOffset: 92, invulnerable: 0.25 } }),
    skill('beast_archer_pack_call', 'Pack Call', 'beastArcher', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'beast_archer_companion_strike', rank: 3 }], 'Self-buff prototype: Bond generation, max HP, and steady MP recovery during coordinated attacks.', { resourceCost: 45, cooldown: 70, partyEffect: 'Self: Bond generation, max HP, and MP recovery while attacking.', futurePartyEffect: 'Future party: nearby allies gain minor sustain and resource recovery.' })
  ]);

  function monsterDropEntry(type, id, weight, options) {
    const settings = options || {};
    const entry = {
      type,
      weight: Math.max(1, Math.round(Number(weight || 1)))
    };
    if (type === 'material') entry.materialId = id;
    else if (type === 'equipment') entry.itemId = id;
    else if (type === 'consumable') entry.consumableId = id;
    else if (type === 'card') entry.cardId = id;
    else entry.id = id;
    if (settings.minQuantity != null) entry.minQuantity = Math.max(1, Math.floor(Number(settings.minQuantity) || 1));
    if (settings.maxQuantity != null) entry.maxQuantity = Math.max(entry.minQuantity || 1, Math.floor(Number(settings.maxQuantity) || 1));
    if (settings.rarity) entry.rarity = settings.rarity;
    return Object.freeze(entry);
  }

  const materialDrop = (id, weight, options) => monsterDropEntry('material', id, weight, options);
  const equipmentDrop = (id, weight, options) => monsterDropEntry('equipment', id, weight, options);
  const cardDrop = (id, weight, options) => monsterDropEntry('card', id, weight, options);

  const MONSTER_EQUIPMENT_DROP_GROUPS = Object.freeze({
    training: Object.freeze(['adventurer_cutlass', 'fieldguard_helm', 'trailwoven_gloves']),
    traveler: Object.freeze(['traveler_boots', 'wanderer_charm']),
    focus: Object.freeze(['balanced_focus', 'starglass_staff', 'wanderer_charm', 'channeler_gloves']),
    guard: Object.freeze(['fieldguard_helm', 'bulwark_plate', 'sentinel_greaves']),
    sharp: Object.freeze(['ranger_recurve', 'deadeye_wraps', 'vanguard_blade']),
    steel: Object.freeze(['vanguard_blade', 'iron_sword', 'iron_axe']),
    support: Object.freeze(['wanderer_charm', 'runewoven_robes', 'channeler_gloves']),
    rustcoil: Object.freeze(['balanced_focus', 'breaker_gauntlets', 'sentinel_greaves', 'vanguard_blade']),
    cinder: Object.freeze(['ember_core', 'breaker_gauntlets', 'channeler_gloves', 'aetherstep_boots']),
    frost: Object.freeze(['aetherstep_boots', 'sentinel_greaves', 'windrunner_boots', 'runewoven_robes']),
    storm: Object.freeze(['windrunner_boots', 'deadeye_wraps', 'aetherstep_boots', 'ranger_recurve']),
    astral: Object.freeze(['starglass_staff', 'runewoven_robes', 'channeler_gloves', 'aetherstep_boots']),
    eclipse: Object.freeze(['breaker_gauntlets', 'sentinel_greaves', 'deadeye_wraps', 'windrunner_boots']),
    rare: Object.freeze(['breaker_gauntlets', 'sentinel_greaves', 'channeler_gloves', 'deadeye_wraps', 'windrunner_boots'])
  });

  function equipmentDrops(groupId, weight) {
    return (MONSTER_EQUIPMENT_DROP_GROUPS[groupId] || []).map((itemId) => equipmentDrop(itemId, weight));
  }

  const MONSTER_CARD_DROP_GROUPS = Object.freeze({
    forest: Object.freeze(['gel_spark', 'mossguard_oath', 'thorn_focus']),
    beast: Object.freeze(['bristle_charge', 'mossguard_oath', 'hunter_tempo']),
    construct: Object.freeze(['clockwork_patience', 'rustcoil_lens', 'titan_gearheart']),
    cinder: Object.freeze(['ember_glint', 'ashflare_core', 'stormbreak_plume']),
    bandit: Object.freeze(['hunter_tempo', 'storm_fletching', 'rift_splinter']),
    frost: Object.freeze(['frost_thread', 'mossguard_oath', 'cloudcall_vellum']),
    storm: Object.freeze(['storm_fletching', 'cloudcall_vellum', 'stormbreak_plume']),
    astral: Object.freeze(['astral_index', 'cloudcall_vellum', 'archivist_star']),
    eclipse: Object.freeze(['rift_splinter', 'eclipse_corona', 'archivist_star']),
    mimic: Object.freeze(['mimic_cache', 'rift_splinter', 'titan_gearheart']),
    bossForest: Object.freeze(['bramble_heart', 'mossguard_oath', 'thorn_focus']),
    bossConstruct: Object.freeze(['titan_gearheart', 'clockwork_patience', 'rustcoil_lens']),
    bossCinder: Object.freeze(['ashflare_core', 'ember_glint', 'stormbreak_plume']),
    bossFrost: Object.freeze(['frost_thread', 'bramble_heart', 'cloudcall_vellum']),
    bossStorm: Object.freeze(['stormbreak_plume', 'storm_fletching', 'cloudcall_vellum']),
    bossAstral: Object.freeze(['archivist_star', 'astral_index', 'cloudcall_vellum']),
    bossEclipse: Object.freeze(['eclipse_corona', 'rift_splinter', 'archivist_star'])
  });

  function cardDrops(groupId, weight) {
    return (MONSTER_CARD_DROP_GROUPS[groupId] || []).map((cardId) => cardDrop(cardId, weight));
  }

  function monsterDropPool(config) {
    const source = config || {};
    return Object.freeze({
      materials: Object.freeze((source.materials || []).slice()),
      equipment: Object.freeze((source.equipment || []).slice()),
      consumables: Object.freeze((source.consumables || []).slice()),
      cards: Object.freeze((source.cards || []).slice()),
      currencyWeight: Math.max(0, Math.round(Number(source.currencyWeight == null ? 14 : source.currencyWeight))),
      globalRareEligible: source.globalRareEligible !== false,
      basicConsumables: source.basicConsumables !== false
    });
  }

  const MONSTER_DROP_POOLS = Object.freeze({
    slimelet: monsterDropPool({ materials: [materialDrop('gelDrop', 48, { minQuantity: 1, maxQuantity: 2 })], equipment: equipmentDrops('training', 3), cards: cardDrops('forest', 3), currencyWeight: 10 }),
    dewSlime: monsterDropPool({ materials: [materialDrop('dewBead', 46), materialDrop('gelDrop', 18, { minQuantity: 1, maxQuantity: 2 })], equipment: equipmentDrops('training', 3), cards: cardDrops('forest', 3), currencyWeight: 10 }),
    mossback: monsterDropPool({ materials: [materialDrop('mossHide', 48)], equipment: equipmentDrops('guard', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    thornSprout: monsterDropPool({ materials: [materialDrop('thornFiber', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    vineSnapper: monsterDropPool({ materials: [materialDrop('vineFiber', 44), materialDrop('upgradeDust', 7)], equipment: equipmentDrops('focus', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    bristleBoar: monsterDropPool({ materials: [materialDrop('bristleHide', 46)], equipment: equipmentDrops('traveler', 4), cards: cardDrops('beast', 4), currencyWeight: 13 }),
    briarStag: monsterDropPool({ materials: [materialDrop('briarAntler', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('traveler', 4), cards: cardDrops('beast', 4), currencyWeight: 14 }),
    dustImp: monsterDropPool({ materials: [materialDrop('dustClaw', 44), materialDrop('upgradeDust', 9)], equipment: equipmentDrops('sharp', 2), cards: cardDrops('bandit', 3), currencyWeight: 12 }),
    clockbug: monsterDropPool({ materials: [materialDrop('clockworkScrap', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('construct', 3), currencyWeight: 13 }),
    rustRatchet: monsterDropPool({ materials: [materialDrop('clockworkScrap', 36), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('rustcoil', 3), cards: cardDrops('construct', 4), currencyWeight: 13 }),
    coilSentry: monsterDropPool({ materials: [materialDrop('chargedCoil', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('focus', 3), cards: cardDrops('construct', 4), currencyWeight: 14 }),
    scrapWarden: monsterDropPool({ materials: [materialDrop('scrapPlate', 42)], equipment: equipmentDrops('rustcoil', 4).concat(equipmentDrops('guard', 2)), cards: cardDrops('construct', 4), currencyWeight: 15 }),
    emberWisp: monsterDropPool({ materials: [materialDrop('emberDust', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('cinder', 3), currencyWeight: 13 }),
    ashCrawler: monsterDropPool({ materials: [materialDrop('ashCarapace', 42), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('cinder', 3), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    lavaTick: monsterDropPool({ materials: [materialDrop('moltenFang', 42), materialDrop('emberDust', 20)], equipment: equipmentDrops('cinder', 3), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    cinderSpitter: monsterDropPool({ materials: [materialDrop('cinderGland', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('focus', 2).concat(equipmentDrops('cinder', 3)), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    banditCutter: monsterDropPool({ materials: [materialDrop('banditCloth', 46)], equipment: equipmentDrops('steel', 4), cards: cardDrops('bandit', 4), currencyWeight: 16 }),
    banditCutterDirect: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterReference: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterHybrid: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterPuppet: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditThrower: monsterDropPool({ materials: [materialDrop('throwingKnifeScrap', 46)], equipment: equipmentDrops('sharp', 4), cards: cardDrops('bandit', 4), currencyWeight: 16 }),
    orebackBeetle: monsterDropPool({ materials: [materialDrop('oreChunks', 48, { minQuantity: 1, maxQuantity: 3 }), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('guard', 2), cards: cardDrops('construct', 3), currencyWeight: 12 }),
    glowcapHealer: monsterDropPool({ materials: [materialDrop('glowSpores', 46)], equipment: equipmentDrops('support', 4), cards: cardDrops('forest', 3).concat(cardDrops('frost', 2)), currencyWeight: 13 }),
    crackedMimic: monsterDropPool({ materials: [materialDrop('upgradeCatalyst', 10), materialDrop('wardingScroll', 4), materialDrop('refinementCore', 3)], equipment: equipmentDrops('rare', 8), cards: cardDrops('mimic', 7), currencyWeight: 48 }),
    brambleking: monsterDropPool({ materials: [materialDrop('brambleCrown', 46), materialDrop('upgradeCatalyst', 8)], cards: cardDrops('bossForest', 8), currencyWeight: 22 }),
    clockworkTitan: monsterDropPool({ materials: [materialDrop('titanCore', 46), materialDrop('upgradeCatalyst', 8)], cards: cardDrops('bossConstruct', 8), currencyWeight: 22 }),
    quarryColossus: monsterDropPool({ materials: [materialDrop('colossusOre', 46), materialDrop('upgradeCatalyst', 7), materialDrop('wardingScroll', 3)], cards: cardDrops('bossConstruct', 8), currencyWeight: 24 }),
    emberjawGolem: monsterDropPool({ materials: [materialDrop('emberjawBadge', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 3)], cards: cardDrops('bossCinder', 8), currencyWeight: 24 }),
    frostlingScout: monsterDropPool({ materials: [materialDrop('rimeShard', 46)], equipment: equipmentDrops('frost', 3), cards: cardDrops('frost', 3), currencyWeight: 14 }),
    shardling: monsterDropPool({ materials: [materialDrop('rimeShard', 42), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('frost', 3), cards: cardDrops('frost', 3), currencyWeight: 14 }),
    rimebackBrute: monsterDropPool({ materials: [materialDrop('frozenHide', 44), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('frost', 2), cards: cardDrops('frost', 4), currencyWeight: 15 }),
    glacierSentinel: monsterDropPool({ materials: [materialDrop('glacierCore', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('frost', 4), cards: cardDrops('frost', 4), currencyWeight: 15 }),
    snowglareWisp: monsterDropPool({ materials: [materialDrop('snowglareDust', 46)], equipment: equipmentDrops('focus', 4), cards: cardDrops('frost', 3).concat(cardDrops('astral', 2)), currencyWeight: 14 }),
    icebloomOracle: monsterDropPool({ materials: [materialDrop('icebloomPetal', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('support', 4), cards: cardDrops('frost', 3).concat(cardDrops('astral', 2)), currencyWeight: 14 }),
    galeHarrier: monsterDropPool({ materials: [materialDrop('galeFeather', 44)], equipment: equipmentDrops('storm', 4).concat(equipmentDrops('focus', 2)), cards: cardDrops('storm', 4), currencyWeight: 15 }),
    stormboundArcher: monsterDropPool({ materials: [materialDrop('stormFletching', 44)], equipment: equipmentDrops('storm', 4).concat(equipmentDrops('sharp', 2)), cards: cardDrops('storm', 4), currencyWeight: 15 }),
    thunderRam: monsterDropPool({ materials: [materialDrop('thunderHorn', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('traveler', 3).concat(equipmentDrops('storm', 3)), cards: cardDrops('storm', 4), currencyWeight: 16 }),
    cloudcallAcolyte: monsterDropPool({ materials: [materialDrop('cloudSilk', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('support', 4), cards: cardDrops('storm', 3).concat(cardDrops('astral', 3)), currencyWeight: 15 }),
    indexScribe: monsterDropPool({ materials: [materialDrop('runicPage', 44)], equipment: equipmentDrops('astral', 4).concat(equipmentDrops('focus', 2)), cards: cardDrops('astral', 4), currencyWeight: 16 }),
    lumenSentinel: monsterDropPool({ materials: [materialDrop('lumenPlate', 42)], equipment: equipmentDrops('astral', 4).concat(equipmentDrops('guard', 2)), cards: cardDrops('astral', 4), currencyWeight: 16 }),
    voidMote: monsterDropPool({ materials: [materialDrop('voidDust', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('focus', 3), cards: cardDrops('astral', 3).concat(cardDrops('eclipse', 2)), currencyWeight: 16 }),
    eclipseDuelist: monsterDropPool({ materials: [materialDrop('eclipseSilk', 42)], equipment: equipmentDrops('eclipse', 4).concat(equipmentDrops('sharp', 2)), cards: cardDrops('eclipse', 4), currencyWeight: 18 }),
    riftAberration: monsterDropPool({ materials: [materialDrop('riftSplinter', 40), materialDrop('upgradeCatalyst', 6), materialDrop('wardingScroll', 3), materialDrop('refinementCore', 3)], equipment: equipmentDrops('rare', 7), cards: cardDrops('mimic', 4).concat(cardDrops('eclipse', 5)), currencyWeight: 26 }),
    rimewarden: monsterDropPool({ materials: [materialDrop('rimewardenSigil', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 3)], cards: cardDrops('bossFrost', 8), currencyWeight: 24 }),
    stormbreakRoc: monsterDropPool({ materials: [materialDrop('stormbreakPlume', 46), materialDrop('upgradeCatalyst', 7), materialDrop('wardingScroll', 3)], cards: cardDrops('bossStorm', 8), currencyWeight: 24 }),
    astralArchivist: monsterDropPool({ materials: [materialDrop('archivistIndex', 46), materialDrop('upgradeCatalyst', 7), materialDrop('cubeFragment', 4)], cards: cardDrops('bossAstral', 8), currencyWeight: 26 }),
    eclipseSovereign: monsterDropPool({ materials: [materialDrop('sovereignCorona', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 4)], cards: cardDrops('bossEclipse', 8), currencyWeight: 28 })
  });

  function attachEnemyDropPool(enemy) {
    return Object.freeze(Object.assign({}, enemy, {
      dropPool: MONSTER_DROP_POOLS[enemy.id] || monsterDropPool({ globalRareEligible: false, currencyWeight: 0 })
    }));
  }

  const ENEMIES = Object.freeze([
    { id: 'slimelet', name: 'Slimelet', levelRange: [1, 6], role: 'Basic/swarm-light', family: 'Ooze', hpMult: 0.75, damageMult: 0.6, defenseMult: 0.6, expMult: 0.7, speed: 52, behavior: 'hopper', mechanic: 'Slow hop contact.', counter: 'Any basic attack, AoE.', drops: ['Gel Drop', 'Training gear'] },
    { id: 'dewSlime', name: 'Dew Slime', levelRange: [1, 8], role: 'Starter swarm', family: 'Ooze', hpMult: 0.68, damageMult: 0.55, defenseMult: 0.55, expMult: 0.72, speed: 58, behavior: 'hopper', mechanic: 'Quick wet hops and low-contact pressure.', counter: 'Basic attacks, short AoE, and knockback.', drops: ['Gel Drop', 'Dew Bead', 'Training gear'] },
    { id: 'mossback', name: 'Mossback', levelRange: [4, 10], role: 'Durable melee', family: 'Beast', hpMult: 1.25, damageMult: 0.85, defenseMult: 1.2, expMult: 1.1, speed: 42, behavior: 'bruiser', mechanic: 'Braces against knockback.', counter: 'Fire, armor break.', drops: ['Moss Hide', 'Guard Ring'] },
    { id: 'thornSprout', name: 'Thorn Sprout', levelRange: [5, 12], role: 'Stationary ranged', family: 'Plant', hpMult: 0.7, damageMult: 0.85, defenseMult: 0.7, expMult: 1, speed: 0, behavior: 'turret', mechanic: 'Fires dodgeable thorns.', counter: 'Fire, range, line attacks.', drops: ['Thorn Fiber', 'Focus Ring'] },
    { id: 'vineSnapper', name: 'Vine Snapper', levelRange: [8, 22], role: 'Ambush skirmisher', family: 'Plant', hpMult: 0.95, damageMult: 1, defenseMult: 0.85, expMult: 1.08, speed: 92, behavior: 'skirmisher', mechanic: 'Lunges from brush, then retreats to cover.', counter: 'Fire, traps, and quick ranged tags.', drops: ['Vine Fiber', 'Upgrade Dust', 'Focus accessory'] },
    { id: 'bristleBoar', name: 'Bristle Boar', levelRange: [8, 16], role: 'Charger', family: 'Beast', hpMult: 1.15, damageMult: 1.25, defenseMult: 1, expMult: 1.2, speed: 78, behavior: 'charger', mechanic: 'Telegraph charge.', counter: 'Jumping, traps, blocks.', drops: ['Bristle Hide', 'Traveler Boots'] },
    { id: 'briarStag', name: 'Briar Stag', levelRange: [12, 28], role: 'Heavy charger', family: 'Beast / Plant', hpMult: 1.3, damageMult: 1.22, defenseMult: 1.05, expMult: 1.24, speed: 84, behavior: 'charger', mechanic: 'Antler charge leaves a brief thorn hazard trail.', counter: 'Jump timing, slows, and burst after the charge.', drops: ['Briar Antler', 'Traveler Boots', 'Upgrade Catalyst'] },
    { id: 'dustImp', name: 'Dust Imp', levelRange: [10, 18], role: 'Fast melee', family: 'Imp', hpMult: 0.85, damageMult: 1, defenseMult: 0.75, expMult: 1.1, speed: 118, behavior: 'skirmisher', mechanic: 'Leap slash and retreat.', counter: 'AoE, traps, fast hits.', drops: ['Dust Claw', 'Upgrade Dust'] },
    { id: 'clockbug', name: 'Clockbug', levelRange: [12, 22], role: 'Armored tank', family: 'Construct', hpMult: 1.7, damageMult: 0.8, defenseMult: 1.7, expMult: 1.4, speed: 38, behavior: 'armored', mechanic: 'Armor crack state.', counter: 'Armor break, bossing skills.', drops: ['Clockwork Scrap', 'Focus Amulet'] },
    { id: 'rustRatchet', name: 'Rust Ratchet', levelRange: [12, 26], role: 'Construct skirmisher', family: 'Construct', hpMult: 0.9, damageMult: 1.04, defenseMult: 1, expMult: 1.12, speed: 104, behavior: 'skirmisher', mechanic: 'Skates along gear rails and snaps at close range.', counter: 'Area control, slows, and armor break.', drops: ['Clockwork Scrap', 'Upgrade Dust', 'Rustcoil gear'] },
    { id: 'coilSentry', name: 'Coil Sentry', levelRange: [14, 32], role: 'Stationary construct ranged', family: 'Construct', hpMult: 0.95, damageMult: 1.05, defenseMult: 1.35, expMult: 1.2, speed: 0, behavior: 'turret', mechanic: 'Charges visible electric bolts from fixed posts.', counter: 'Line-of-sight breaks, burst, and ranged attacks.', drops: ['Charged Coil', 'Focus Amulet', 'Upgrade Catalyst'] },
    { id: 'scrapWarden', name: 'Scrap Warden', levelRange: [24, 46], role: 'Armored blocker', family: 'Construct / Humanoid', hpMult: 1.55, damageMult: 1.08, defenseMult: 1.45, expMult: 1.36, speed: 56, behavior: 'blocker', mechanic: 'Raises a shield plate before short counter-swings.', counter: 'Back attacks, armor break, and stun windows.', drops: ['Scrap Plate', 'Guard Ring', 'Rustcoil gear'] },
    { id: 'emberWisp', name: 'Ember Wisp', levelRange: [16, 26], role: 'Flying ranged', family: 'Spirit', hpMult: 0.85, damageMult: 0.95, defenseMult: 0.7, expMult: 1.15, speed: 70, behavior: 'flyer', mechanic: 'Floating firebolt.', counter: 'Marks, lightning, ranged attacks.', drops: ['Ember Dust', 'Ember Ring'] },
    { id: 'ashCrawler', name: 'Ash Crawler', levelRange: [16, 40], role: 'Volcanic bruiser', family: 'Volcanic Beast', hpMult: 1.35, damageMult: 1.02, defenseMult: 1.18, expMult: 1.22, speed: 46, behavior: 'bruiser', mechanic: 'Shrugs off light hits while ember plates cool.', counter: 'Cold pressure, armor break, and sustained hits.', drops: ['Ash Carapace', 'Upgrade Dust', 'Cinder gear'] },
    { id: 'lavaTick', name: 'Lava Tick', levelRange: [22, 50], role: 'Fast burn skirmisher', family: 'Volcanic Beast', hpMult: 0.72, damageMult: 1.16, defenseMult: 0.78, expMult: 1.18, speed: 124, behavior: 'skirmisher', mechanic: 'Rapid bites with short burst movement between vents.', counter: 'AoE, traps, and burst before it retreats.', drops: ['Molten Fang', 'Ember Dust', 'Cinder gear'] },
    { id: 'cinderSpitter', name: 'Cinder Spitter', levelRange: [28, 55], role: 'Volcanic thrower', family: 'Volcanic Beast', hpMult: 0.88, damageMult: 1.12, defenseMult: 0.9, expMult: 1.26, speed: 62, behavior: 'thrower', mechanic: 'Lobs arcing cinder globs from mid-range.', counter: 'Gap closers, vertical movement, and stuns.', drops: ['Cinder Gland', 'Focus accessory', 'Upgrade Catalyst'] },
    { id: 'banditCutter', name: 'Bandit Cutter', levelRange: [18, 28], role: 'Melee blocker', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Frontal block.', counter: 'Back attacks, burst, stun.', drops: ['Bandit Cloth', 'Steel weapon'] },
    { id: 'banditCutterDirect', name: 'Direct Sheet Bandit', levelRange: [18, 28], role: 'Asset test - direct sheet', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Direct single-prompt sheet comparison.', counter: 'Back attacks, burst, stun.', drops: [] },
    { id: 'banditCutterReference', name: 'Reference Sheet Bandit', levelRange: [18, 28], role: 'Asset test - reference sheet', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Reference-guided sheet comparison.', counter: 'Back attacks, burst, stun.', drops: [] },
    { id: 'banditCutterHybrid', name: 'Hybrid Keyframe Bandit', levelRange: [18, 28], role: 'Asset test - hybrid keyframes', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Generated key poses with deterministic assembly.', counter: 'Back attacks, burst, stun.', drops: [] },
    { id: 'banditCutterPuppet', name: 'Puppet Composite Bandit', levelRange: [18, 28], role: 'Asset test - puppet composite', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Generated puppet poses with deterministic transforms.', counter: 'Back attacks, burst, stun.', drops: [] },
    { id: 'banditThrower', name: 'Bandit Thrower', levelRange: [20, 30], role: 'Ranged priority', family: 'Humanoid', hpMult: 0.8, damageMult: 0.9, defenseMult: 0.75, expMult: 1.05, speed: 74, behavior: 'thrower', mechanic: 'Retreats and throws arcs.', counter: 'Gap closers, long range.', drops: ['Throwing Knife Scrap', 'Sharp Ring'] },
    { id: 'orebackBeetle', name: 'Oreback Beetle', levelRange: [24, 35], role: 'Tank/material', family: 'Beast / Mineral', hpMult: 1.9, damageMult: 0.9, defenseMult: 1.8, expMult: 1.5, speed: 35, behavior: 'armored', mechanic: 'Shell crack.', counter: 'Armor break, weak points.', drops: ['Ore Chunks', 'Upgrade Catalyst'] },
    { id: 'glowcapHealer', name: 'Glowcap Healer', levelRange: [22, 34], role: 'Support', family: 'Plant', hpMult: 0.75, damageMult: 0.4, defenseMult: 0.7, expMult: 1.2, speed: 34, behavior: 'healer', mechanic: 'Heals nearby enemies.', counter: 'Fire, seal, target priority.', drops: ['Glow Spores', 'Support accessory'] },
    { id: 'crackedMimic', name: 'Cracked Mimic', levelRange: [15, 35], role: 'Rare elite', family: 'Construct / Treasure', hpMult: 4, damageMult: 1.5, defenseMult: 1.2, expMult: 4, speed: 96, behavior: 'elite', mechanic: 'Ambush and flee at low HP.', counter: 'Burst windows, stuns.', drops: ['Currency burst', 'Rare gear', 'Upgrade Catalyst'] },
    { id: 'brambleking', name: 'Brambleking', levelRange: [24, 34], role: 'Boss', family: 'Plant', hpMult: 6.8, damageMult: 1.45, defenseMult: 1.25, expMult: 5.8, speed: 24, behavior: 'boss', mechanic: 'Root waves, thorn volleys, and vine cages.', counter: 'Fire, mobility, and controlled burst windows.', drops: ['Bramble Crown', 'Boss gear', 'Upgrade Catalyst'] },
    { id: 'clockworkTitan', name: 'Clockwork Titan', levelRange: [30, 42], role: 'Boss', family: 'Construct', hpMult: 7.4, damageMult: 1.55, defenseMult: 1.9, expMult: 6.4, speed: 34, behavior: 'boss', mechanic: 'Gear slams, armor plates, and timed vulnerability phases.', counter: 'Armor break, bossing skills, and weak-point timing.', drops: ['Titan Core', 'Boss gear', 'Upgrade Catalyst'] },
    { id: 'quarryColossus', name: 'Quarry Colossus', levelRange: [34, 48], role: 'Boss', family: 'Mineral Construct', hpMult: 7.8, damageMult: 1.65, defenseMult: 2, expMult: 6.8, speed: 30, behavior: 'boss', mechanic: 'Ore armor, falling rocks, and healer add pressure.', counter: 'Sustained armor crack uptime and priority add control.', drops: ['Colossus Ore', 'Boss gear', 'Rare catalyst'] },
    { id: 'emberjawGolem', name: 'Emberjaw Golem', levelRange: [30, 40], role: 'Boss', family: 'Volcanic Construct', hpMult: 8, damageMult: 1.8, defenseMult: 1.6, expMult: 8, speed: 52, behavior: 'boss', mechanic: 'Ground slams, fire cracks, arena charge, ore minions, Overheat vulnerability.', counter: 'Dodging, control during Overheat, weak-point marks.', drops: ['Emberjaw Badge', 'Boss gear', 'Rare catalyst'] },
    { id: 'frostlingScout', name: 'Frostling Scout', levelRange: [45, 58], role: 'Fast melee', family: 'Frostkin', hpMult: 0.95, damageMult: 1.12, defenseMult: 0.9, expMult: 1.24, speed: 112, behavior: 'skirmisher', mechanic: 'Dashes across slick ground after short feints.', counter: 'Area control, slows, and quick burst windows.', drops: ['Rime Shard', 'Frost gear'] },
    { id: 'shardling', name: 'Shardling', levelRange: [45, 60], role: 'Frost swarm', family: 'Frost Construct', hpMult: 0.78, damageMult: 0.9, defenseMult: 0.95, expMult: 1.08, speed: 66, behavior: 'hopper', mechanic: 'Skips across ice in short crystalline hops.', counter: 'AoE and knockback before it surrounds you.', drops: ['Rime Shard', 'Upgrade Dust', 'Frost gear'] },
    { id: 'rimebackBrute', name: 'Rimeback Brute', levelRange: [48, 64], role: 'Armored tank', family: 'Frost Beast', hpMult: 2.05, damageMult: 1.05, defenseMult: 1.85, expMult: 1.55, speed: 42, behavior: 'armored', mechanic: 'Ice shell cracks under sustained armor break.', counter: 'Armor break and persistent damage.', drops: ['Frozen Hide', 'Upgrade Catalyst'] },
    { id: 'glacierSentinel', name: 'Glacier Sentinel', levelRange: [54, 70], role: 'Frozen turret', family: 'Frost Construct', hpMult: 1.35, damageMult: 1.12, defenseMult: 1.55, expMult: 1.42, speed: 0, behavior: 'turret', mechanic: 'Anchors itself and fires slow piercing ice lances.', counter: 'Line-of-sight breaks, armor break, and burst.', drops: ['Glacier Core', 'Frost gear', 'Upgrade Catalyst'] },
    { id: 'snowglareWisp', name: 'Snowglare Wisp', levelRange: [50, 66], role: 'Flying ranged', family: 'Frost Spirit', hpMult: 0.9, damageMult: 1.1, defenseMult: 0.75, expMult: 1.3, speed: 78, behavior: 'flyer', mechanic: 'Floats above ice lanes and fires chill bolts.', counter: 'Ranged attacks, marks, and mobility skills.', drops: ['Snowglare Dust', 'Focus accessory'] },
    { id: 'icebloomOracle', name: 'Icebloom Oracle', levelRange: [48, 66], role: 'Frost support', family: 'Plant / Frost Spirit', hpMult: 0.82, damageMult: 0.55, defenseMult: 0.82, expMult: 1.26, speed: 38, behavior: 'healer', mechanic: 'Channels frost blooms that heal nearby enemies.', counter: 'Target priority, fire, and interrupts.', drops: ['Icebloom Petal', 'Support accessory', 'Prism Shards'] },
    { id: 'galeHarrier', name: 'Gale Harrier', levelRange: [55, 72], role: 'Wind flyer', family: 'Storm Spirit', hpMult: 0.86, damageMult: 1.1, defenseMult: 0.78, expMult: 1.28, speed: 92, behavior: 'flyer', mechanic: 'Strafes through cliff lanes with gust bolts.', counter: 'Marks, lightning resistance, and ranged attacks.', drops: ['Gale Feather', 'Storm gear', 'Focus accessory'] },
    { id: 'stormboundArcher', name: 'Stormbound Archer', levelRange: [55, 72], role: 'Ranged pressure', family: 'Humanoid / Storm', hpMult: 0.92, damageMult: 1.14, defenseMult: 0.9, expMult: 1.3, speed: 76, behavior: 'thrower', mechanic: 'Kites between platforms while firing charged arrows.', counter: 'Gap closers, stuns, and vertical pressure.', drops: ['Storm Fletching', 'Sharp Ring', 'Storm gear'] },
    { id: 'thunderRam', name: 'Thunder Ram', levelRange: [58, 74], role: 'Storm charger', family: 'Beast / Storm', hpMult: 1.42, damageMult: 1.28, defenseMult: 1.15, expMult: 1.4, speed: 96, behavior: 'charger', mechanic: 'Crackling charge after a visible hoof spark.', counter: 'Jump timing, slows, and post-charge burst.', drops: ['Thunder Horn', 'Traveler Boots', 'Upgrade Catalyst'] },
    { id: 'cloudcallAcolyte', name: 'Cloudcall Acolyte', levelRange: [60, 74], role: 'Storm support', family: 'Humanoid / Storm', hpMult: 0.88, damageMult: 0.65, defenseMult: 0.9, expMult: 1.32, speed: 48, behavior: 'healer', mechanic: 'Calls cloud pulses that restore nearby allies.', counter: 'Target priority and interrupts.', drops: ['Cloud Silk', 'Support accessory', 'Prism Shards'] },
    { id: 'indexScribe', name: 'Index Scribe', levelRange: [70, 90], role: 'Astral ranged', family: 'Astral Humanoid', hpMult: 0.94, damageMult: 1.18, defenseMult: 0.95, expMult: 1.35, speed: 70, behavior: 'thrower', mechanic: 'Throws rune pages that arc over low cover.', counter: 'Gap closers, line movement, and burst.', drops: ['Runic Page', 'Focus accessory', 'Astral gear'] },
    { id: 'lumenSentinel', name: 'Lumen Sentinel', levelRange: [72, 100], role: 'Astral armored tank', family: 'Astral Construct', hpMult: 1.8, damageMult: 1.08, defenseMult: 1.85, expMult: 1.58, speed: 44, behavior: 'armored', mechanic: 'Luminous shell dims as armor breaks.', counter: 'Armor break, sustained damage, and weak-point marks.', drops: ['Lumen Plate', 'Guard Ring', 'Astral gear'] },
    { id: 'voidMote', name: 'Void Mote', levelRange: [75, 100], role: 'Astral flyer', family: 'Void Spirit', hpMult: 0.82, damageMult: 1.22, defenseMult: 0.78, expMult: 1.38, speed: 96, behavior: 'flyer', mechanic: 'Blinks in short arcs before firing void sparks.', counter: 'Ranged tracking, marks, and burst timing.', drops: ['Void Dust', 'Focus accessory', 'Prism Shards'] },
    { id: 'eclipseDuelist', name: 'Eclipse Duelist', levelRange: [85, 105], role: 'Late-game blocker', family: 'Astral Humanoid', hpMult: 1.16, damageMult: 1.28, defenseMult: 1.12, expMult: 1.46, speed: 92, behavior: 'blocker', mechanic: 'Parries from the front and lunges after blocking.', counter: 'Back attacks, stuns, and vertical repositioning.', drops: ['Eclipse Silk', 'Sharp Ring', 'Eclipse gear'] },
    { id: 'riftAberration', name: 'Rift Aberration', levelRange: [100, 100], role: 'Rift elite', family: 'Void Aberration', hpMult: 3.2, damageMult: 1.55, defenseMult: 1.25, expMult: 3.6, speed: 86, behavior: 'elite', mechanic: 'Warps through lanes and enrages at low HP.', counter: 'Burst windows, stuns, and target focus.', drops: ['Rift Splinter', 'Rare gear', 'Rare catalyst'] },
    { id: 'rimewarden', name: 'Rimewarden', levelRange: [58, 70], role: 'Boss', family: 'Frost Construct', hpMult: 8.2, damageMult: 1.72, defenseMult: 1.75, expMult: 7.2, speed: 38, behavior: 'boss', mechanic: 'Freezing shockwaves, armor phases, and slippery arena control.', counter: 'Positioning, armor crack uptime, and controlled burst.', drops: ['Rimewarden Sigil', 'Boss gear', 'Rare catalyst'] },
    { id: 'stormbreakRoc', name: 'Aurelion, Stormbreak Roc', levelRange: [72, 84], role: 'Boss', family: 'Storm Beast', hpMult: 8.8, damageMult: 1.84, defenseMult: 1.5, expMult: 7.8, speed: 74, behavior: 'boss', mechanic: 'Flying divebombs, lightning rods, and wind lane pressure.', counter: 'Ranged uptime, rod baiting, and vertical repositioning.', drops: ['Stormbreak Plume', 'Boss gear', 'Rare catalyst'] },
    { id: 'astralArchivist', name: 'The Astral Archivist', levelRange: [88, 98], role: 'Boss', family: 'Astral Humanoid', hpMult: 9.2, damageMult: 1.88, defenseMult: 1.62, expMult: 8.4, speed: 46, behavior: 'boss', mechanic: 'Rune pages, action-memory resistance, and mirrored delayed attacks.', counter: 'Skill variety, add control, and target focus.', drops: ['Archivist Index', 'Boss gear', 'Rare catalyst'] },
    { id: 'eclipseSovereign', name: 'Eclipse Sovereign', levelRange: [100, 112], role: 'Boss', family: 'Astral Royalty', hpMult: 10.2, damageMult: 2.02, defenseMult: 1.76, expMult: 9.2, speed: 54, behavior: 'boss', mechanic: 'Solar and lunar stance swaps, eclipse zones, and totality sigils.', counter: 'Zone swaps, coordinated burst, and phase awareness.', drops: ['Sovereign Corona', 'Boss gear', 'Rare catalyst'] }
  ].map(attachEnemyDropPool).map(attachEnemyAssets));

  const ACTIVE_COMBAT_SKILLS = Object.freeze(SKILLS.filter((skill) => skill && skill.category !== 'passive'));

  const SKILL_FX_ANIMATION_ASSETS = Object.freeze(ACTIVE_COMBAT_SKILLS.reduce((assets, skill) => {
    assets[skill.id] = makeCombatFxAnimationAsset(toCombatFxFileId(skill.id), 'skills', SKILL_FX_ANIMATION_ROWS);
    return assets;
  }, {}));

  const BASIC_ATTACK_FX_ANIMATION_ASSETS = Object.freeze(['fighter', 'mage', 'archer'].reduce((assets, classId) => {
    assets[classId] = makeCombatFxAnimationAsset(`basic-${toCombatFxFileId(classId)}`, 'basic', BASIC_ATTACK_FX_ANIMATION_ROWS);
    return assets;
  }, {}));

  const ENEMY_COMBAT_FX_ANIMATION_ASSETS = Object.freeze(ENEMIES.reduce((assets, enemy) => {
    assets[enemy.id] = makeCombatFxAnimationAsset(toCombatFxFileId(enemy.id), 'enemies', ENEMY_COMBAT_FX_ANIMATION_ROWS);
    return assets;
  }, {}));

  const ANIMATION_ASSETS = Object.freeze(Object.assign({}, CORE_ANIMATION_ASSETS, {
    skillFx: SKILL_FX_ANIMATION_ASSETS,
    basicAttackFx: BASIC_ATTACK_FX_ANIMATION_ASSETS,
    enemyCombatFx: ENEMY_COMBAT_FX_ANIMATION_ASSETS,
    enemyProjectiles: ENEMY_PROJECTILE_ANIMATION_ASSETS
  }));

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

  const MAP_PORTALS = Object.freeze({
    starfallCrossing: Object.freeze([
      Object.freeze({ id: 'crossing_greenroot', label: 'Greenroot Gate', destinationMapId: 'greenrootMeadow', routeId: 'forest', x: 2040, platformIndex: 0 })
    ]),
    greenrootMeadow: Object.freeze([
      Object.freeze({ id: 'greenroot_crossing', label: 'Town Return', destinationMapId: 'starfallCrossing', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'greenroot_thornpath', label: 'Thornpath Pass', destinationMapId: 'thornpathThicket', routeId: 'forest', requiredMapId: 'greenrootMeadow', x: 7070, platformIndex: 0 })
    ]),
    thornpathThicket: Object.freeze([
      Object.freeze({ id: 'thornpath_greenroot', label: 'Greenroot Return', destinationMapId: 'greenrootMeadow', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'thornpath_bandit', label: 'Bandit Ridge', destinationMapId: 'banditRidgeCamp', routeId: 'forest', requiredMapId: 'thornpathThicket', x: 7330, platformIndex: 0 }),
      Object.freeze({ id: 'thornpath_rustcoil_outpost', label: 'Rustcoil Outpost', destinationMapId: 'rustcoilOutpost', routeId: 'forest', requiredMapId: 'thornpathThicket', x: 7460, platformIndex: 0 })
    ]),
    banditRidgeCamp: Object.freeze([
      Object.freeze({ id: 'bandit_thornpath', label: 'Thornpath Return', destinationMapId: 'thornpathThicket', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'bandit_bramble', label: 'Bramble Depths', dungeonId: 'bramble_depths', routeId: 'forest', bossPortal: true, x: 7580, platformIndex: 0 })
    ]),
    brambleDepths: Object.freeze([
      Object.freeze({ id: 'bramble_bandit', label: 'Ridge Return', destinationMapId: 'banditRidgeCamp', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    rustcoilOutpost: Object.freeze([
      Object.freeze({ id: 'rustcoil_outpost_thornpath', label: 'Thornpath Return', destinationMapId: 'thornpathThicket', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'rustcoil_outpost_ruins', label: 'Rustcoil Ruins', destinationMapId: 'rustcoilRuins', routeId: 'ruins', x: 2660, platformIndex: 0 }),
      Object.freeze({ id: 'rustcoil_outpost_quarry', label: 'Oreback Quarry', destinationMapId: 'orebackQuarry', routeId: 'ruins', requiredMapId: 'rustcoilRuins', x: 2840, platformIndex: 0 })
    ]),
    rustcoilRuins: Object.freeze([
      Object.freeze({ id: 'rustcoil_outpost_return', label: 'Outpost Return', destinationMapId: 'rustcoilOutpost', returnPortal: true, x: 110, platformIndex: 0 })
    ]),
    orebackQuarry: Object.freeze([
      Object.freeze({ id: 'quarry_rustcoil_outpost', label: 'Outpost Return', destinationMapId: 'rustcoilOutpost', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'quarry_vault', label: 'Gearworks Vault', dungeonId: 'gearworks_vault', routeId: 'ruins', bossPortal: true, x: 7820, platformIndex: 0 }),
      Object.freeze({ id: 'quarry_cinder_refuge', label: 'Cinder Refuge', destinationMapId: 'cinderRefuge', routeId: 'ruins', requiredMapId: 'orebackQuarry', x: 7900, platformIndex: 0 })
    ]),
    gearworksVault: Object.freeze([
      Object.freeze({ id: 'vault_quarry', label: 'Quarry Return', destinationMapId: 'orebackQuarry', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    cinderRefuge: Object.freeze([
      Object.freeze({ id: 'cinder_refuge_quarry', label: 'Oreback Return', destinationMapId: 'orebackQuarry', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'cinder_refuge_hollow', label: 'Cinder Hollow', destinationMapId: 'cinderHollow', routeId: 'cinder', x: 2660, platformIndex: 0 }),
      Object.freeze({ id: 'cinder_refuge_ashglass', label: 'Ashglass Pass', destinationMapId: 'ashglassPass', routeId: 'frostfen', requiredLevel: 40, requiredDungeonId: 'emberjaw_lair', x: 2840, platformIndex: 0 })
    ]),
    cinderHollow: Object.freeze([
      Object.freeze({ id: 'cinder_refuge_return', label: 'Refuge Return', destinationMapId: 'cinderRefuge', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'cinder_emberjaw', label: 'Emberjaw Lair', dungeonId: 'emberjaw_lair', routeId: 'cinder', bossPortal: true, x: 7580, platformIndex: 0 })
    ]),
    emberjawLair: Object.freeze([
      Object.freeze({ id: 'lair_cinder', label: 'Cinder Return', destinationMapId: 'cinderHollow', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    ashglassPass: Object.freeze([
      Object.freeze({ id: 'ashglass_cinder_refuge', label: 'Refuge Return', destinationMapId: 'cinderRefuge', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'ashglass_frostfen_camp', label: 'Frostfen Camp', destinationMapId: 'frostfenCamp', routeId: 'frostfen', requiredMapId: 'ashglassPass', x: 8060, platformIndex: 0 })
    ]),
    frostfenCamp: Object.freeze([
      Object.freeze({ id: 'frostfen_camp_ashglass', label: 'Ashglass Return', destinationMapId: 'ashglassPass', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'frostfen_camp_outskirts', label: 'Frostfen Tundra', destinationMapId: 'frostfenOutskirts', routeId: 'frostfen', x: 2660, platformIndex: 0 }),
      Object.freeze({ id: 'frostfen_camp_glacier', label: 'Glacier Spine', destinationMapId: 'glacierSpine', routeId: 'frostfen', requiredMapId: 'frostfenOutskirts', x: 2840, platformIndex: 0 })
    ]),
    frostfenOutskirts: Object.freeze([
      Object.freeze({ id: 'frostfen_camp_return', label: 'Camp Return', destinationMapId: 'frostfenCamp', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'frostfen_glacier', label: 'Glacier Spine', destinationMapId: 'glacierSpine', routeId: 'frostfen', requiredMapId: 'frostfenOutskirts', x: 8060, platformIndex: 0 })
    ]),
    glacierSpine: Object.freeze([
      Object.freeze({ id: 'glacier_frostfen_camp', label: 'Camp Return', destinationMapId: 'frostfenCamp', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'glacier_sanctum', label: 'Rimewarden Sanctum', dungeonId: 'rimewarden_sanctum', routeId: 'frostfen', bossPortal: true, x: 7920, platformIndex: 0 }),
      Object.freeze({ id: 'glacier_stormbreak_haven', label: 'Stormbreak Haven', destinationMapId: 'stormbreakHaven', routeId: 'ascension', requiredDungeonId: 'rimewarden_sanctum', x: 8060, platformIndex: 0 })
    ]),
    rimewardenSanctum: Object.freeze([
      Object.freeze({ id: 'sanctum_glacier', label: 'Glacier Return', destinationMapId: 'glacierSpine', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    stormbreakHaven: Object.freeze([
      Object.freeze({ id: 'stormbreak_haven_glacier', label: 'Glacier Return', destinationMapId: 'glacierSpine', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'stormbreak_haven_cliffs', label: 'Stormbreak Cliffs', destinationMapId: 'stormbreakCliffs', routeId: 'ascension', x: 2660, platformIndex: 0 }),
      Object.freeze({ id: 'stormbreak_haven_observatory', label: 'Astral Observatory', destinationMapId: 'astralObservatory', routeId: 'ascension', requiredMapId: 'stormbreakCliffs', x: 2840, platformIndex: 0 })
    ]),
    stormbreakCliffs: Object.freeze([
      Object.freeze({ id: 'stormbreak_haven_return', label: 'Haven Return', destinationMapId: 'stormbreakHaven', returnPortal: true, x: 110, platformIndex: 0 })
    ]),
    astralObservatory: Object.freeze([
      Object.freeze({ id: 'astral_observatory_stormbreak', label: 'Stormbreak Return', destinationMapId: 'stormbreakHaven', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'astral_observatory_archive', label: 'Astral Archive', destinationMapId: 'astralArchive', routeId: 'ascension', x: 2660, platformIndex: 0 })
    ]),
    astralArchive: Object.freeze([
      Object.freeze({ id: 'archive_observatory', label: 'Observatory Return', destinationMapId: 'astralObservatory', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'archive_eclipse', label: 'Eclipse Frontier', destinationMapId: 'eclipseFrontier', routeId: 'ascension', requiredMapId: 'astralArchive', x: 8060, platformIndex: 0 })
    ]),
    eclipseFrontier: Object.freeze([
      Object.freeze({ id: 'eclipse_archive', label: 'Archive Return', destinationMapId: 'astralArchive', returnPortal: true, x: 110, platformIndex: 0 }),
      Object.freeze({ id: 'eclipse_rift', label: 'Endless Rift', destinationMapId: 'endlessRift', routeId: 'ascension', requiredMapId: 'eclipseFrontier', x: 8060, platformIndex: 0 })
    ]),
    endlessRift: Object.freeze([
      Object.freeze({ id: 'rift_eclipse', label: 'Eclipse Return', destinationMapId: 'eclipseFrontier', returnPortal: true, x: 110, platformIndex: 0 })
    ]),
    bramblekingCourt: Object.freeze([
      Object.freeze({ id: 'court_return', label: 'Ridge Return', destinationMapId: 'banditRidgeCamp', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    titanFoundry: Object.freeze([
      Object.freeze({ id: 'foundry_return', label: 'Quarry Return', destinationMapId: 'orebackQuarry', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    deepcoreCore: Object.freeze([
      Object.freeze({ id: 'deepcore_return', label: 'Quarry Return', destinationMapId: 'orebackQuarry', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    emberjawFurnace: Object.freeze([
      Object.freeze({ id: 'furnace_return', label: 'Cinder Return', destinationMapId: 'cinderHollow', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    rimewardenVault: Object.freeze([
      Object.freeze({ id: 'vault_return', label: 'Glacier Return', destinationMapId: 'glacierSpine', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    stormbreakAerie: Object.freeze([
      Object.freeze({ id: 'aerie_return', label: 'Stormbreak Return', destinationMapId: 'stormbreakCliffs', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    astralStacks: Object.freeze([
      Object.freeze({ id: 'stacks_return', label: 'Archive Return', destinationMapId: 'astralArchive', returnPortal: true, x: 100, platformIndex: 0 })
    ]),
    eclipseThrone: Object.freeze([
      Object.freeze({ id: 'throne_return', label: 'Frontier Return', destinationMapId: 'eclipseFrontier', returnPortal: true, x: 100, platformIndex: 0 })
    ])
  });

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

  function getAuthoredMapWidth(map) {
    const platformWidth = (map.platforms || []).reduce((width, platform) => Math.max(width, Number(platform[0] || 0) + Number(platform[2] || 0)), 0);
    const pointWidth = []
      .concat(map.spawnPoints || [])
      .concat(map.stations || [])
      .concat(map.questNpcs || [])
      .reduce((width, point) => Math.max(width, Number(point && point.x || 0) + 240), 0);
    return Math.max(3600, platformWidth, pointWidth);
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

  function makePartyPlayPlatforms(width, options) {
    const worldWidth = Math.max(3600, Math.ceil(Number(width || 0) / 100) * 100);
    const anchors = getPartyPlayZoneAnchors(worldWidth, options);
    const platforms = [[0, 520, worldWidth, 80]];
    const addPlatform = (x, y, w) => {
      const widthLimit = Math.min(w, worldWidth - x - 220);
      if (widthLimit >= 120) platforms.push([Math.round(x), y, Math.round(widthLimit), 22]);
    };
    anchors.forEach((anchor, index) => {
      const drift = index % 2 ? 80 : 0;
      addPlatform(anchor + drift, TRAINING_LANE_Y.low, 960);
      addPlatform(anchor + 1040 - drift * 0.25, TRAINING_LANE_Y.lowConnector, 220);
      addPlatform(anchor + 900 - drift * 0.35, TRAINING_LANE_Y.mid, 920);
      addPlatform(anchor + 600 + drift * 0.2, TRAINING_LANE_Y.highConnector, 220);
      addPlatform(anchor + 320 + drift * 0.15, TRAINING_LANE_Y.high, 880);
    });
    return platforms;
  }

  function makePartyPlayClimbables(prefix, width, options) {
    const worldWidth = Math.max(3600, Number(width || 0));
    const anchors = getPartyPlayZoneAnchors(worldWidth, options);
    const climbables = [];
    anchors.forEach((anchor, index) => {
      const drift = index % 2 ? 80 : 0;
      const groundX = anchor + 160 + drift;
      const lowerX = anchor + 900 + drift;
      const upperX = anchor + 980 + drift;
      if (groundX + 15 <= worldWidth - 240) {
        climbables.push({ id: `${prefix}_party_lift_${index + 1}_low`, x: groundX, y: TRAINING_LANE_Y.low, w: 30, h: TRAINING_LANE_Y.ground - TRAINING_LANE_Y.low });
      }
      if (lowerX + 15 <= worldWidth - 240) {
        climbables.push({ id: `${prefix}_party_lift_${index + 1}_mid`, x: lowerX, y: TRAINING_LANE_Y.mid, w: 30, h: TRAINING_LANE_Y.low - TRAINING_LANE_Y.mid });
      }
      if (upperX + 15 <= worldWidth - 240) {
        climbables.push({ id: `${prefix}_party_lift_${index + 1}_high`, x: upperX, y: TRAINING_LANE_Y.high, w: 30, h: TRAINING_LANE_Y.mid - TRAINING_LANE_Y.high });
      }
    });
    return climbables;
  }

  function makePartyPlaySpawnPoints(platforms) {
    return platforms
      .map((platform, index) => ({ platform, index }))
      .filter((entry) => entry.index > 0 && entry.platform[2] >= 640)
      .map((entry) => ({
        x: Math.round(entry.platform[0] + entry.platform[2] / 2),
        platformIndex: entry.index,
        weight: entry.platform[1] >= 430 ? 3 : entry.platform[1] >= 300 ? 2 : 1
      }));
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

  function getFieldZoneAnchors(width, layoutStyle) {
    const worldWidth = Math.max(isVerticalFieldLayout(layoutStyle) ? 4600 : 6200, Math.ceil(Number(width || 0) / 100) * 100);
    const first = 260;
    const zoneCount = isVerticalFieldLayout(layoutStyle) ? worldWidth >= 6200 ? 4 : 3 : 4;
    const last = Math.max(first, worldWidth - (isVerticalFieldLayout(layoutStyle) ? 1500 : 2020));
    return Array.from({ length: zoneCount }, (_, index) => Math.round(first + (last - first) * index / Math.max(1, zoneCount - 1)));
  }

  function makeFieldPlatforms(width, layoutStyle) {
    const vertical = isVerticalFieldLayout(layoutStyle);
    const worldWidth = Math.max(vertical ? 4600 : 6200, Math.ceil(Number(width || 0) / 100) * 100);
    const anchors = getFieldZoneAnchors(worldWidth, layoutStyle);
    const lanes = getFieldLaneY(layoutStyle);
    const platforms = [[0, lanes.ground, worldWidth, 80]];
    const addPlatform = (x, y, w) => {
      const safeX = Math.max(120, Math.round(x));
      const widthLimit = Math.min(Math.round(w), worldWidth - safeX - 180);
      if (widthLimit >= 120) platforms.push([safeX, y, widthLimit, 22]);
    };
    if (vertical) {
      anchors.forEach((anchor, index) => {
        const flip = index % 2 === 1;
        const lift = layoutStyle === 'stormClimb' || layoutStyle === 'astralStack' || layoutStyle === 'riftStack' ? 24 : 0;
        const lowX = anchor + (flip ? 260 : 80);
        const midX = anchor + (flip ? 80 : 360);
        const highX = anchor + (flip ? 420 : 180);
        const peakX = anchor + (flip ? 220 : 480);
        const skyX = anchor + (flip ? 280 : 560);
        addPlatform(lowX, lanes.low - lift, 920);
        addPlatform(midX, lanes.mid - lift, 840);
        addPlatform(highX, lanes.high - lift, 760);
        addPlatform(peakX, lanes.peak - lift, 620);
        addPlatform(skyX, lanes.sky - lift, 420);
      });
      return platforms;
    }
    anchors.forEach((anchor, index) => {
      if (layoutStyle === 'switchbackTerraces') {
        const drift = index % 2 ? 140 : 0;
        addPlatform(anchor + drift, lanes.low, 1320);
        addPlatform(anchor + 1340 - drift * 0.25, lanes.lowConnector, 240);
        addPlatform(anchor + 450 - drift * 0.6, lanes.mid, 1220);
        addPlatform(anchor + 180 + drift * 0.55, lanes.highConnector, 240);
        addPlatform(anchor + 40 + drift, lanes.high, 1120);
        return;
      }
      if (layoutStyle === 'verticalCanopy') {
        const drift = index % 2 ? 170 : 0;
        addPlatform(anchor + drift, lanes.low, 1040);
        addPlatform(anchor + 1180 - drift * 0.35, lanes.lowConnector, 220);
        addPlatform(anchor + 360 - drift * 0.25, lanes.mid, 980);
        addPlatform(anchor + 1040 + drift * 0.1, lanes.highConnector, 220);
        addPlatform(anchor + 120 + drift * 0.55, lanes.high, 940);
        return;
      }
      const laneOffset = index % 2 ? 90 : 0;
      addPlatform(anchor + laneOffset, lanes.low, 1560);
      addPlatform(anchor + 1510, lanes.lowConnector, 220);
      addPlatform(anchor + 260 - laneOffset * 0.4, lanes.mid, 1420);
      addPlatform(anchor + 390, lanes.highConnector, 220);
      addPlatform(anchor + 560 + laneOffset * 0.35, lanes.high, 1280);
    });
    return platforms;
  }

  function makeTerrainIslandSegments(platform, index, layoutStyle) {
    const width = Math.max(0, Number(platform && platform[2] || 0));
    const count = width >= 1400 ? 3 : width >= 900 ? 2 : 1;
    const baseWidth = count === 3 ? 300 : count === 2 ? 340 : Math.min(420, Math.max(260, width - 120));
    const styleDrift = layoutStyle === 'switchbackTerraces' ? 36 : layoutStyle === 'verticalCanopy' ? -28 : 0;
    return Object.freeze(Array.from({ length: count }, (_, segmentIndex) => {
      const drift = ((index + segmentIndex) % 2 ? 1 : -1) * (28 + segmentIndex * 8) + styleDrift;
      const rawCenter = width * (segmentIndex + 1) / (count + 1) + drift;
      const segmentWidth = Math.min(baseWidth + (segmentIndex % 2 ? 32 : 0), Math.max(180, width - 96));
      const x = clamp(Math.round(rawCenter - segmentWidth / 2), 36, Math.max(36, width - segmentWidth - 36));
      return Object.freeze({
        x,
        w: Math.round(segmentWidth),
        depth: 28 + (index + segmentIndex) % 3 * 4
      });
    }));
  }

  function makeFieldTerrainVisuals(platforms, layoutStyle) {
    return Object.freeze(platforms.map((platform, index) => {
      const width = Math.max(0, Number(platform && platform[2] || 0));
      if (index === 0) {
        return Object.freeze({ kind: 'ground', segments: Object.freeze([]) });
      }
      if (width <= 320) {
        return Object.freeze({ kind: 'connector', segments: Object.freeze([]) });
      }
      return Object.freeze({
        kind: 'solidLane',
        segments: Object.freeze([])
      });
    }));
  }

  function makeClimbableBetweenPlatforms(prefix, platforms, topIndex, bottomIndex, key, kind) {
    const top = platforms[topIndex];
    const bottom = platforms[bottomIndex];
    if (!top || !bottom || bottom[1] <= top[1]) return null;
    const overlapLeft = Math.max(top[0] + 54, bottom[0] + 54);
    const overlapRight = Math.min(top[0] + top[2] - 54, bottom[0] + bottom[2] - 54);
    const rawX = overlapLeft <= overlapRight
      ? (overlapLeft + overlapRight) / 2
      : top[0] + top[2] / 2;
    const width = kind === 'stair' ? 46 : 30;
    return {
      id: `${prefix}_${kind}_${key}`,
      x: Math.round(rawX - width / 2),
      y: top[1],
      w: width,
      h: Math.max(48, bottom[1] - top[1])
    };
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

  function makeVerticalFieldClimbables(prefix, platforms, layoutStyle) {
    const kind = getFieldClimbableKind(layoutStyle);
    return platforms
      .map((platform, topIndex) => ({ platform, topIndex }))
      .filter((entry) => entry.topIndex > 0)
      .map((entry, localIndex) => {
        const top = entry.platform;
        const bottomEntry = platforms
          .map((platform, bottomIndex) => ({ platform, bottomIndex }))
          .filter((candidate) => candidate.bottomIndex !== entry.topIndex && candidate.platform[1] > top[1])
          .sort((a, b) => {
            const aOverlap = Math.min(top[0] + top[2], a.platform[0] + a.platform[2]) - Math.max(top[0], a.platform[0]);
            const bOverlap = Math.min(top[0] + top[2], b.platform[0] + b.platform[2]) - Math.max(top[0], b.platform[0]);
            return Math.abs(a.platform[1] - top[1]) - Math.abs(b.platform[1] - top[1]) || bOverlap - aOverlap;
          })[0];
        return bottomEntry
          ? makeClimbableBetweenPlatforms(prefix, platforms, entry.topIndex, bottomEntry.bottomIndex, `${localIndex + 1}`, kind)
          : null;
      })
      .filter(Boolean);
  }

  function makeFieldClimbables(prefix, widthOrPlatforms, layoutStyle) {
    const platforms = Array.isArray(widthOrPlatforms) ? widthOrPlatforms : null;
    if (platforms && isVerticalFieldLayout(layoutStyle)) return makeVerticalFieldClimbables(prefix, platforms, layoutStyle);
    const width = platforms
      ? Math.max(6200, platforms.reduce((maxWidth, platform) => Math.max(maxWidth, platform[0] + platform[2]), 0))
      : widthOrPlatforms;
    const lanes = getFieldLaneY(layoutStyle);
    const anchors = getFieldZoneAnchors(width, layoutStyle);
    const climbables = [];
    anchors.forEach((anchor, index) => {
      if (layoutStyle === 'switchbackTerraces') {
        const drift = index % 2 ? 140 : 0;
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_low`, x: anchor + 240 + drift, y: lanes.low, w: 30, h: lanes.ground - lanes.low });
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_mid`, x: anchor + 760 - drift * 0.45, y: lanes.mid, w: 30, h: lanes.low - lanes.mid });
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_high`, x: anchor + 560 + drift * 0.35, y: lanes.high, w: 30, h: lanes.mid - lanes.high });
        return;
      }
      if (layoutStyle === 'verticalCanopy') {
        const drift = index % 2 ? 170 : 0;
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_low`, x: anchor + 180 + drift, y: lanes.low, w: 28, h: lanes.ground - lanes.low });
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_mid`, x: anchor + 720 - drift * 0.2, y: lanes.mid, w: 28, h: lanes.low - lanes.mid });
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_high`, x: anchor + 640 + drift * 0.15, y: lanes.high, w: 28, h: lanes.mid - lanes.high });
        return;
      }
      const laneOffset = index % 2 ? 90 : 0;
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_low`, x: anchor + 180 + laneOffset, y: lanes.low, w: 28, h: lanes.ground - lanes.low });
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_mid`, x: anchor + 1060, y: lanes.mid, w: 28, h: lanes.low - lanes.mid });
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_high`, x: anchor + 820 + laneOffset * 0.2, y: lanes.high, w: 28, h: lanes.mid - lanes.high });
    });
    return climbables;
  }

  function makeFieldSpawnPoints(platforms) {
    return platforms
      .map((platform, index) => ({ platform, index }))
      .filter((entry) => entry.index > 0 && entry.platform[2] >= 640)
      .reduce((points, entry) => {
        const platform = entry.platform;
        const weight = platform[1] >= 430 ? 3 : platform[1] >= 320 ? 2 : 1;
        if (platform[2] >= 1200) {
          points.push({ x: Math.round(platform[0] + platform[2] * 0.32), platformIndex: entry.index, weight });
          points.push({ x: Math.round(platform[0] + platform[2] * 0.68), platformIndex: entry.index, weight });
        } else {
          points.push({ x: Math.round(platform[0] + platform[2] / 2), platformIndex: entry.index, weight });
        }
        return points;
      }, []);
  }

  function makeTownPlatforms(width) {
    const worldWidth = Math.max(3600, Math.ceil(Number(width || 0) / 100) * 100);
    const platforms = [[0, TOWN_LANE_Y.ground, worldWidth, 80]];
    const add = (x, y, w) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW >= 160) platforms.push([safeX, y, safeW, 24]);
    };
    add(220, TOWN_LANE_Y.low, 740);
    add(1180, TOWN_LANE_Y.low, 720);
    add(2220, TOWN_LANE_Y.low, 780);
    add(680, TOWN_LANE_Y.mid, 860);
    add(1720, TOWN_LANE_Y.mid, 780);
    add(2860, TOWN_LANE_Y.mid, 520);
    add(360, TOWN_LANE_Y.high, 700);
    add(1440, TOWN_LANE_Y.high, 780);
    add(2440, TOWN_LANE_Y.high, 640);
    add(1040, TOWN_LANE_Y.roof, 640);
    return platforms;
  }

  function makeTownClimbables(prefix, platforms) {
    return [
      makeClimbableBetweenPlatforms(prefix, platforms, 1, 0, 'left_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 4, 1, 'left_roofwalk', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 7, 4, 'left_balcony', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 10, 8, 'guild_roof', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 2, 0, 'market_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 5, 2, 'market_roofwalk', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 8, 5, 'artisan_balcony', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 3, 0, 'gate_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 6, 3, 'gate_watch', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 9, 6, 'gate_roof', 'stair')
    ].filter(Boolean);
  }

  function getTownStationPlacement(stationId) {
    const placements = {
      storage: { x: 430, platformIndex: 0 },
      shop: { x: 440, platformIndex: 1 },
      slots: { x: 1020, platformIndex: 4 },
      upgrade: { x: 1750, platformIndex: 8 },
      class: { x: 1290, platformIndex: 10 }
    };
    return placements[stationId] || { x: 430, platformIndex: 0 };
  }

  function townServiceNpcId(mapId, serviceId) {
    const prefix = String(mapId || 'town')
      .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
      .replace(/[^a-zA-Z0-9_]+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '')
      .toLowerCase();
    return `${prefix}_${serviceId}_host`;
  }

  function createPlinkoHostNpc(mapId, options) {
    const settings = options || {};
    return {
      id: settings.id || townServiceNpcId(mapId, 'plinko'),
      name: settings.name || 'Plinko Host',
      xOverride: Number.isFinite(Number(settings.xOverride)) ? Number(settings.xOverride) : 2260,
      platformIndex: Number.isFinite(Number(settings.platformIndex)) ? Number(settings.platformIndex) : 5,
      questIds: [],
      servicePanelId: 'plinko',
      serviceStationId: 'plinko',
      serviceLabel: 'Starfall Plinko',
      serviceSummary: 'Spend coins or drop mob-earned balls into Starfall Plinko for weighted rewards and 100-ball jackpot pity.',
      color: settings.color || '#5a4fb3',
      accent: settings.accent || '#ffd166'
    };
  }

  function withTownServiceNpcs(mapId, questNpcs, options) {
    return [createPlinkoHostNpc(mapId, options && options.plinkoHost)].concat(questNpcs || []);
  }

  function assignTownStations(stations) {
    return (stations || []).map((station) => {
      const placement = getTownStationPlacement(station.id);
      return Object.assign({}, station, {
        x: Number.isFinite(Number(station.xOverride)) ? Number(station.xOverride) : placement.x,
        platformIndex: Number.isFinite(Number(station.platformIndex)) ? Number(station.platformIndex) : placement.platformIndex
      });
    });
  }

  function assignTownQuestNpcs(questNpcs) {
    return (questNpcs || []).map((npc, index) => Object.assign({}, npc, {
      x: Number.isFinite(Number(npc.xOverride)) ? Number(npc.xOverride) : index % 2 ? 2140 : 3000,
      platformIndex: Number.isFinite(Number(npc.platformIndex)) && Number(npc.platformIndex) > 0
        ? Number(npc.platformIndex)
        : index % 2 ? 5 : 6
    }));
  }

  function applyPartyPlayGeometry(map) {
    if (!map) return map;
    if (map.safeZone) {
      const width = Math.max(getAuthoredMapWidth(map), 3600);
      const platforms = makeTownPlatforms(width);
      return Object.assign({}, map, {
        layoutStyle: 'townVerticalHub',
        layoutRole: 'town',
        worldHeight: TOWN_WORLD_HEIGHT,
        authoredGroundY: TOWN_LANE_Y.ground,
        platforms,
        terrainVisuals: makeFieldTerrainVisuals(platforms, 'townVerticalHub'),
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
    const platforms = map.isDungeon
      ? makePartyPlayPlatforms(width, { dungeon: true })
      : makeFieldPlatforms(width, layoutStyle);
    return Object.assign({}, map, {
      layoutStyle: map.isDungeon ? 'dungeonArena' : layoutStyle,
      layoutRole: map.layoutRole || getMapLayoutRoleFallback(map),
      worldHeight: map.isDungeon ? 0 : getFieldLayoutWorldHeight(layoutStyle),
      authoredGroundY: map.isDungeon ? TRAINING_LANE_Y.ground : getFieldLaneY(layoutStyle).ground,
      platforms,
      terrainVisuals: map.isDungeon ? [] : makeFieldTerrainVisuals(platforms, layoutStyle),
      climbables: map.isDungeon ? makePartyPlayClimbables(map.id, width, { dungeon: true }) : makeFieldClimbables(map.id, platforms, layoutStyle),
      spawnPoints: map.isDungeon ? makePartyPlaySpawnPoints(platforms) : makeFieldSpawnPoints(platforms)
    });
  }

  function makeExpandedTrainingMap(config) {
    const layoutStyle = config.layoutStyle || FIELD_LAYOUT_STYLES[config.id] || 'sharedLanes';
    const width = isVerticalFieldLayout(layoutStyle) ? 5200 : 8400;
    const platforms = makeFieldPlatforms(width, layoutStyle);
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
      climbables: makeFieldClimbables(config.id, platforms, layoutStyle),
      spawnPoints: makeFieldSpawnPoints(platforms),
      stations: [],
      questNpcs: config.questNpcs || []
    };
  }

  function makeBossRoomMap(config) {
    const width = Math.max(4600, Number(config.width || 4600));
    const platforms = makePartyPlayPlatforms(width, { dungeon: true });
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
      climbables: makePartyPlayClimbables(config.id, width, { dungeon: true }),
      spawnPoints: makePartyPlaySpawnPoints(platforms),
      stations: [],
      questNpcs: config.questNpcs || []
    };
  }

  function makeTownHubMap(config) {
    return {
      id: config.id,
      name: config.name,
      levelRange: config.levelRange || [1, 99],
      safeZone: true,
      layoutRole: 'town',
      palette: config.palette,
      purpose: config.purpose,
      enemies: [],
      platforms: [[0, 520, 3000, 80]],
      climbables: [],
      spawnPoints: [],
      stations: [
        { id: 'storage', name: 'Storage Keeper', x: 430 },
        { id: 'shop', name: config.shopName || 'Regional Outfitter', x: 740 },
        { id: 'slots', name: 'Slot Broker', x: 1060 },
        { id: 'upgrade', name: 'Upgrade Artisan', x: 1380 }
      ],
      questNpcs: withTownServiceNpcs(config.id, config.questNpcs || [], config)
    };
  }

  const MAPS = Object.freeze([
    { id: 'starfallCrossing', name: 'Starfall Crossing', levelRange: [1, 99], safeZone: true, palette: ['#f7d28a', '#7ec8d8', '#f8f0dc'], purpose: 'Starter town hub, shops, quest handoffs, storage, upgrade artisan, and Starfall Plinko sink.', enemies: [], platforms: [[0, 520, 3800, 80], [420, 430, 260, 24], [980, 385, 300, 24], [1540, 430, 270, 24]], climbables: [], spawnPoints: [], stations: [{ id: 'storage', name: 'Storage Keeper', x: 420 }, { id: 'shop', name: 'Starter Outfitter', x: 760 }, { id: 'slots', name: 'Slot Broker', x: 1090 }, { id: 'upgrade', name: 'Upgrade Artisan', x: 1460 }, { id: 'class', name: 'Class Supplier', x: 1840 }, { id: 'plinko', name: 'Starfall Plinko', x: 2260 }], questNpcs: withTownServiceNpcs('starfallCrossing', [{ id: 'crossing_class_master', name: 'Class Master', x: 2080, platformIndex: 0, questIds: ['trial_ready'], color: '#5e7d9f', accent: '#ffd166' }], { plinkoHost: { xOverride: 2260, platformIndex: 5 } }) },
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
      name: 'Greenroot Meadow',
      levelRange: [1, 6],
      safeZone: false,
      waveMax: 24,
      waveDelay: 5,
      palette: ['#77bf65', '#91dbe8', '#f3d86d'],
      purpose: 'Beginner grassy platforms, ponds, and slime habitats with tree-canopy routes.',
      enemies: ['dewSlime', 'slimelet', 'dewSlime', 'slimelet', 'dewSlime', 'thornSprout', 'dewSlime', 'slimelet', 'thornSprout', 'dewSlime', 'mossback', 'thornSprout'],
      platforms: [[0, 520, 7200, 80], [260, 452, 340, 22], [620, 388, 300, 22], [960, 318, 280, 22], [1360, 452, 410, 22], [1810, 382, 330, 22], [2180, 312, 290, 22], [2640, 452, 360, 22], [3040, 388, 320, 22], [3420, 322, 300, 22], [3840, 452, 420, 22], [4310, 382, 340, 22], [4700, 304, 300, 22], [5120, 452, 380, 22], [5560, 386, 340, 22], [5960, 316, 300, 22], [6320, 246, 260, 22], [6680, 452, 330, 22], [6900, 386, 260, 22]],
      climbables: [{ id: 'meadow_rope_1', x: 1088, y: 318, w: 26, h: 202 }, { id: 'meadow_rope_2', x: 2296, y: 312, w: 26, h: 208 }, { id: 'meadow_rope_3', x: 4814, y: 304, w: 26, h: 216 }, { id: 'meadow_rope_4', x: 6422, y: 246, w: 26, h: 274 }],
      spawnPoints: [{ x: 420, platformIndex: 1, weight: 3 }, { x: 790, platformIndex: 2, weight: 2 }, { x: 1110, platformIndex: 3, weight: 1 }, { x: 1570, platformIndex: 4, weight: 3 }, { x: 2310, platformIndex: 6, weight: 2 }, { x: 3190, platformIndex: 8, weight: 2 }, { x: 3560, platformIndex: 9, weight: 1 }, { x: 4480, platformIndex: 11, weight: 2 }, { x: 4860, platformIndex: 12, weight: 1 }, { x: 5740, platformIndex: 14, weight: 2 }, { x: 6430, platformIndex: 16, weight: 1 }, { x: 6840, platformIndex: 17, weight: 3 }],
      stations: [],
      questNpcs: [{ id: 'greenroot_guide', name: 'Greenroot Guide', x: 320, platformIndex: 0, questIds: ['first_steps', 'greenroot_samples'], color: '#4f9d61', accent: '#ffe16a' }]
    },
    {
      id: 'thornpathThicket',
      name: 'Thornpath Thicket',
      levelRange: [5, 16],
      safeZone: false,
      waveMax: 26,
      waveDelay: 5,
      palette: ['#3f8f58', '#5b3d2d', '#c4475d'],
      purpose: 'Dense foliage, thorn bushes, footbridges, and ranged-enemy perches.',
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
      waveMax: 28,
      waveDelay: 5,
      palette: ['#8c6b35', '#7a8592', '#29b3ad'],
      purpose: 'Old construct machinery, brass gears, stone platforms, and catwalks.',
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
      waveMax: 30,
      waveDelay: 5,
      palette: ['#6f5132', '#4f7b63', '#c3995b'],
      purpose: 'Rope bridges, lookout platforms, crates, tents, and narrow bandit lanes.',
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
      palette: ['#d7f3ff', '#5ca8e8', '#f7fbff'],
      purpose: 'Snowfield platforms and frozen marsh lanes that introduce slick footing and frost enemy packs.',
      enemies: ['shardling', 'frostlingScout', 'snowglareWisp', 'rimebackBrute', 'shardling', 'frostlingScout', 'icebloomOracle', 'snowglareWisp', 'rimebackBrute', 'shardling', 'frostlingScout', 'icebloomOracle'],
      movementProfile: 'ice',
      areaMechanic: 'Ice footing reduces ground acceleration and increases sliding.',
      waveMax: 31,
      waveDelay: 5,
      questNpcs: [{ id: 'frostfen_tracker', name: 'Frostfen Tracker', x: 520, platformIndex: 0, questIds: ['frostfen_field_notes'], color: '#6386a8', accent: '#b7f2ff' }]
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
      palette: ['#1f2330', '#ffbe55', '#7bdff2'],
      purpose: 'Custom boss echo with solar and lunar stance swaps, eclipse sigils, and totality burst windows.',
      enemies: ['eclipseSovereign', 'eclipseDuelist', 'voidMote', 'lumenSentinel', 'indexScribe'],
      questNpcs: [{ id: 'eclipse_witness', name: 'Eclipse Witness', x: 520, platformIndex: 0, questIds: ['eclipse_echo'], color: '#1f2330', accent: '#ffbe55' }]
    })
  ].map(applyPartyPlayGeometry).map(attachMapAssets));

  const BOSS_ENCOUNTERS = Object.freeze([
    Object.freeze({
      id: 'brambleking',
      bossId: 'brambleking',
      name: 'Brambleking Court',
      mapId: 'bramblekingCourt',
      setId: 'thorncrown_regalia',
      color: '#e05b75',
      accent: '#8bd47a',
      roomAmbient: 'bramble',
      mechanic: 'Swap lanes when roots grow, destroy thorn pods when adds spawn, and burst during Crowned Root.',
      intro: 'The court roots itself across the arena. Move between lanes before the roots close.',
      clearText: 'The crown breaks and the roots pull back from the court.',
      summary: 'Root waves force lane swaps, thorn volleys punish stacking, and Crowned Root exposes a short damage window.',
      adds: Object.freeze(['thornSprout', 'vineSnapper', 'glowcapHealer']),
      phases: Object.freeze([
        Object.freeze({ id: 'rootCourt', name: 'Root Court', threshold: 1, description: 'Root lanes telegraph where the court will split.', actions: Object.freeze(['rootWave', 'thornVolley']) }),
        Object.freeze({ id: 'thornCanopy', name: 'Thorn Canopy', threshold: 0.7, description: 'Thorn pods call in sprouts while volleys punish stacked players.', actions: Object.freeze(['thornVolley', 'addWave', 'rootWave']) }),
        Object.freeze({ id: 'crownedRoot', name: 'Crowned Root', threshold: 0.38, description: 'The crown opens a short burst window after the vine cage.', actions: Object.freeze(['vineCage', 'rootWave', 'crownExpose']) })
      ])
    }),
    Object.freeze({
      id: 'clockworkTitan',
      bossId: 'clockworkTitan',
      name: 'Titan Foundry',
      mapId: 'titanFoundry',
      setId: 'titanwork_aegis',
      color: '#29b3ad',
      accent: '#d8b74a',
      roomAmbient: 'gear',
      mechanic: 'Use pressure-plate gaps between gear lanes, then punish exposed plates before overclock ends.',
      intro: 'The foundry wakes one gear at a time. Watch the floor before the Titan commits.',
      clearText: 'The Titan locks up and the foundry gears grind to a halt.',
      summary: 'Gear lanes sweep the arena while exposed plates create burst windows between overclock cycles.',
      adds: Object.freeze(['clockbug', 'coilSentry', 'rustRatchet']),
      phases: Object.freeze([
        Object.freeze({ id: 'gearStart', name: 'Gear Start', threshold: 1, description: 'Heavy gear slams mark safe pressure-plate gaps.', actions: Object.freeze(['gearSlam', 'plateExpose']) }),
        Object.freeze({ id: 'foundryShift', name: 'Foundry Shift', threshold: 0.7, description: 'Gear lanes sweep longer sections while foundry minions enter.', actions: Object.freeze(['gearLane', 'addWave', 'gearSlam']) }),
        Object.freeze({ id: 'overclocked', name: 'Overclocked', threshold: 0.38, description: 'Overclock pulses accelerate the pattern before plates open.', actions: Object.freeze(['overclock', 'gearLane', 'plateExpose']) })
      ])
    }),
    Object.freeze({
      id: 'quarryColossus',
      bossId: 'quarryColossus',
      name: 'Deepcore Core',
      mapId: 'deepcoreCore',
      setId: 'deepcore_colossus',
      color: '#69d1a6',
      accent: '#c3b48f',
      roomAmbient: 'core',
      mechanic: 'Spread away from quake anchors, dodge rockfall shadows, and collapse the cracked core pulse.',
      intro: 'The Deepcore awakens under the platforms. Quake anchors will split the party if ignored.',
      clearText: 'The core fractures cleanly and the quarry quiets.',
      summary: 'Orefalls and quake anchors split the party across the mining terraces before the core cracks open.',
      adds: Object.freeze(['orebackBeetle', 'scrapWarden', 'glowcapHealer']),
      phases: Object.freeze([
        Object.freeze({ id: 'stoneSkin', name: 'Stone Skin', threshold: 1, description: 'Rockfall shadows and quake anchors define the first safe paths.', actions: Object.freeze(['rockfall', 'quakeAnchor']) }),
        Object.freeze({ id: 'deepSeam', name: 'Deep Core', threshold: 0.7, description: 'Adds arrive from mining seams while anchors force movement.', actions: Object.freeze(['quakeAnchor', 'addWave', 'rockfall']) }),
        Object.freeze({ id: 'coreBreak', name: 'Core Break', threshold: 0.38, description: 'The exposed core pulses around the Colossus before the next cave-in.', actions: Object.freeze(['corePulse', 'rockfall', 'quakeAnchor']) })
      ])
    }),
    Object.freeze({
      id: 'emberjawGolem',
      bossId: 'emberjawGolem',
      name: 'Emberjaw Furnace',
      mapId: 'emberjawFurnace',
      setId: 'furnaceheart_arsenal',
      color: '#ff7842',
      accent: '#ffd166',
      roomAmbient: 'furnace',
      mechanic: 'Cross lava seams before they flare, avoid charge lanes, and burn the core during overheat.',
      intro: 'Emberjaw floods the furnace floor with seams. Keep moving before the lava breathes.',
      clearText: 'The furnace cools and Emberjaw collapses into slag.',
      summary: 'Furnace cracks mark burning lanes, lava charges reposition Emberjaw, and overheat briefly weakens the core.',
      adds: Object.freeze(['emberWisp', 'lavaTick', 'cinderSpitter']),
      phases: Object.freeze([
        Object.freeze({ id: 'heatedStone', name: 'Heated Stone', threshold: 1, description: 'Fire cracks mark lanes before Emberjaw charges.', actions: Object.freeze(['fireCrack', 'lavaCharge']) }),
        Object.freeze({ id: 'furnaceJaw', name: 'Furnace Jaw', threshold: 0.7, description: 'Cinder adds pressure the furnace while charge lanes shift.', actions: Object.freeze(['lavaCharge', 'addWave', 'fireCrack']) }),
        Object.freeze({ id: 'meltdownCore', name: 'Meltdown Core', threshold: 0.38, description: 'Overheat exposes the core before the floor erupts again.', actions: Object.freeze(['overheat', 'fireCrack', 'lavaCharge']) })
      ])
    }),
    Object.freeze({
      id: 'rimewarden',
      bossId: 'rimewarden',
      name: 'Rimewarden Vault',
      mapId: 'rimewardenVault',
      setId: '',
      color: '#79e7ff',
      accent: '#f7fbff',
      roomAmbient: 'rime',
      mechanic: 'Rotate around ice walls, clear whiteout lanes, and jump frost rings before they close.',
      intro: 'The vault seals behind you. Ice walls will narrow the room before the whiteout lands.',
      clearText: 'The vault thaw cracks and the Rimewarden fades.',
      summary: 'Ice walls close training lanes, whiteout blasts sweep one platform tier, and frost rings punish slow rotations.',
      adds: Object.freeze(['frostlingScout', 'snowglareWisp', 'icebloomOracle']),
      phases: Object.freeze([
        Object.freeze({ id: 'coldSeal', name: 'Cold Seal', threshold: 1, description: 'Frost rings expand while walls mark blocked space.', actions: Object.freeze(['iceShockwave', 'iceWall']) }),
        Object.freeze({ id: 'whiteVault', name: 'White Vault', threshold: 0.7, description: 'Whiteout sweeps one tier while frost adds close in.', actions: Object.freeze(['whiteout', 'addWave', 'iceShockwave']) }),
        Object.freeze({ id: 'absoluteZero', name: 'Absolute Zero', threshold: 0.38, description: 'Walls and whiteout overlap, forcing clean rotations.', actions: Object.freeze(['iceWall', 'whiteout', 'iceShockwave']) })
      ])
    }),
    Object.freeze({
      id: 'stormbreakRoc',
      bossId: 'stormbreakRoc',
      name: 'Stormbreak Aerie',
      mapId: 'stormbreakAerie',
      setId: 'stormcaller_tempest',
      color: '#ffe16a',
      accent: '#91dbe8',
      roomAmbient: 'storm',
      mechanic: 'Keep out of rod circles, cross wind lanes early, and move when the divebomb shadow appears.',
      intro: 'Aurelion circles above the aerie. Rods will ground the lightning before the sky falls.',
      clearText: 'The storm breaks open and Aurelion drops from the clouds.',
      summary: 'Aurelion drops lightning rods, pushes wind lanes across platforms, and divebombs isolated targets.',
      adds: Object.freeze(['galeHarrier', 'stormboundArcher', 'cloudcallAcolyte']),
      phases: Object.freeze([
        Object.freeze({ id: 'highWinds', name: 'High Winds', threshold: 1, description: 'Wind bolts and rod circles establish the storm pattern.', actions: Object.freeze(['windBolt', 'lightningRod']) }),
        Object.freeze({ id: 'stormPerch', name: 'Storm Perch', threshold: 0.7, description: 'Wind lanes push across platforms while stormbound adds arrive.', actions: Object.freeze(['windLane', 'addWave', 'lightningRod']) }),
        Object.freeze({ id: 'skyfall', name: 'Skyfall', threshold: 0.38, description: 'Divebomb shadows force immediate movement before the next lane.', actions: Object.freeze(['divebomb', 'windLane', 'lightningRod']) })
      ])
    }),
    Object.freeze({
      id: 'astralArchivist',
      bossId: 'astralArchivist',
      name: 'Astral Stacks',
      mapId: 'astralStacks',
      setId: 'astral_index',
      color: '#c794ff',
      accent: '#64d9c5',
      roomAmbient: 'astral',
      mechanic: 'Step around rune pages, break memory seals, and dodge mirrored echo lanes.',
      intro: 'The shelves reorder themselves. The Archivist records every repeated mistake.',
      clearText: 'The forbidden appendix snaps shut and the stacks realign.',
      summary: 'Rune pages travel between shelves while the Archivist seals repeated actions and mirrors delayed casts.',
      adds: Object.freeze(['indexScribe', 'lumenSentinel', 'voidMote']),
      phases: Object.freeze([
        Object.freeze({ id: 'openedIndex', name: 'Opened Index', threshold: 1, description: 'Rune pages travel between shelves and memory seals mark danger.', actions: Object.freeze(['runePages', 'memorySeal']) }),
        Object.freeze({ id: 'mirroredStacks', name: 'Mirrored Stacks', threshold: 0.7, description: 'Mirrored echo lanes replay delayed casts as adds enter.', actions: Object.freeze(['mirrorEcho', 'addWave', 'runePages']) }),
        Object.freeze({ id: 'forbiddenAppendix', name: 'Forbidden Appendix', threshold: 0.38, description: 'Seals and echoes overlap until the pages settle.', actions: Object.freeze(['memorySeal', 'mirrorEcho', 'runePages']) })
      ])
    }),
    Object.freeze({
      id: 'eclipseSovereign',
      bossId: 'eclipseSovereign',
      name: 'Eclipse Throne',
      mapId: 'eclipseThrone',
      setId: 'eclipse_paragon',
      color: '#ffbe55',
      accent: '#7bdff2',
      roomAmbient: 'eclipse',
      mechanic: 'Read solar and lunar safe zones, then split around totality sigils before they collapse.',
      intro: 'The throne enters eclipse. Solar and lunar stances will alternate safe lanes.',
      clearText: 'Totality fades and the throne releases its light.',
      summary: 'Solar and lunar stances alternate safe lanes until totality sigils demand coordinated repositioning.',
      adds: Object.freeze(['eclipseDuelist', 'voidMote', 'lumenSentinel']),
      phases: Object.freeze([
        Object.freeze({ id: 'solarCourt', name: 'Solar Court', threshold: 1, description: 'Solar flares expand from the throne while lunar marks choose targets.', actions: Object.freeze(['solarFlare', 'lunarMark']) }),
        Object.freeze({ id: 'lunarCourt', name: 'Lunar Court', threshold: 0.7, description: 'Lunar safe zones invert as eclipse duelists arrive.', actions: Object.freeze(['lunarMark', 'addWave', 'solarFlare']) }),
        Object.freeze({ id: 'totality', name: 'Totality', threshold: 0.38, description: 'Totality sigils split the arena before the next solar flare.', actions: Object.freeze(['eclipseSigils', 'solarFlare', 'lunarMark']) })
      ])
    })
  ]);

  const EQUIPMENT_SLOTS = Object.freeze(['weapon', 'offhand', 'head', 'chest', 'gloves', 'boots', 'ring', 'amulet']);

  const EQUIPMENT_SLOT_META = Object.freeze({
    weapon: Object.freeze({ label: 'Weapon', icon: 'WPN' }),
    offhand: Object.freeze({ label: 'Offhand', icon: 'OFF' }),
    head: Object.freeze({ label: 'Head', icon: 'HD' }),
    chest: Object.freeze({ label: 'Chest', icon: 'CH' }),
    gloves: Object.freeze({ label: 'Gloves', icon: 'GLV' }),
    boots: Object.freeze({ label: 'Boots', icon: 'BT' }),
    ring: Object.freeze({ label: 'Ring', icon: 'RG' }),
    amulet: Object.freeze({ label: 'Amulet', icon: 'AM' })
  });

  function getDefaultEquipmentVisualId(item) {
    const slot = String(item && item.slot || '').trim();
    if (slot === 'head') return 'fieldguard_helm';
    if (slot === 'gloves') return 'trailwoven_gloves';
    if (slot === 'amulet') return 'plain_ring';
    return '';
  }

	  function getEquipmentVisualDefinition(item) {
	    const candidates = [
	      item && item.id,
	      item && item.visualId,
	      getDefaultEquipmentVisualId(item)
	    ];
    for (const candidate of candidates) {
      const id = String(candidate || '').trim();
      if (id && EQUIPMENT_VISUALS[id]) return EQUIPMENT_VISUALS[id];
    }
    return null;
  }

  function attachEquipmentItemAssets(item) {
    const visual = getEquipmentVisualDefinition(item);
    const assetIds = [
      item.id,
      visual && visual.assetId,
      item.assetId,
      item.visualId
    ].filter(Boolean);
    const asset = assetIds.reduce((resolved, assetId) => resolved || ITEM_ASSETS[assetId] || '', '');
    return Object.freeze(Object.assign({}, item, {
      asset,
      visualId: visual ? visual.id : ''
    }));
  }

  const SHOP_ITEMS = Object.freeze([
    { id: 'training_sword', name: 'Training Sword', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'fighter', stats: { power: 8 }, source: 'Starter Outfitter' },
    { id: 'training_wand', name: 'Training Wand', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'mage', stats: { power: 8 }, source: 'Starter Outfitter' },
    { id: 'training_bow', name: 'Training Bow', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'archer', stats: { power: 8 }, source: 'Starter Outfitter' },
    { id: 'copper_sword', name: 'Copper Sword', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'fighter', stats: { power: 16, speed: 4 }, source: 'Starter Outfitter' },
    { id: 'birch_wand', name: 'Birch Wand', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'mage', stats: { power: 15, speed: 4 }, source: 'Starter Outfitter' },
    { id: 'simple_bow', name: 'Simple Bow', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'archer', stats: { power: 15, resourceGain: 2 }, source: 'Starter Outfitter' },
    { id: 'stitched_vest', name: 'Stitched Vest', slot: 'chest', rarity: 'Common', cost: 70, level: 5, classId: 'any', stats: { defense: 6, hp: 24 }, source: 'Starter Outfitter' },
    { id: 'traveler_boots', name: 'Traveler Boots', slot: 'boots', rarity: 'Uncommon', cost: 90, level: 5, classId: 'any', stats: { speed: 12, defense: 2 }, source: 'Starter Outfitter' },
    { id: 'plain_ring', name: 'Plain Ring', slot: 'ring', rarity: 'Common', cost: 65, level: 5, classId: 'any', stats: { hp: 18 }, source: 'Starter Outfitter' },
    { id: 'iron_sword', name: 'Iron Sword', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'fighter', stats: { power: 34, speed: 4 }, source: 'Weapon Smith' },
    { id: 'iron_axe', name: 'Iron Axe', slot: 'weapon', rarity: 'Uncommon', cost: 320, level: 15, classId: 'fighter', stats: { power: 39, armorBreak: 4, speed: -4 }, source: 'Weapon Smith' },
    { id: 'apprentice_staff', name: 'Apprentice Staff', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'mage', stats: { power: 32, areaDamage: 4 }, source: 'Weapon Smith' },
    { id: 'oak_longbow', name: 'Oak Longbow', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'archer', stats: { power: 31, critDamage: 5, range: 24 }, source: 'Weapon Smith' },
    { id: 'guardian_tower_shield', name: 'Guardian Tower Shield', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'guardian', stats: { defense: 40, hp: 180, block: 6 }, source: 'Class Supplier' },
    { id: 'berserker_war_grip', name: 'Berserker War Grip', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'berserker', stats: { power: 18, resourceGain: 6 }, source: 'Class Supplier' },
    { id: 'ember_core', name: 'Ember Core', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'fireMage', stats: { power: 16, burnDamage: 8, resourceMax: 5 }, source: 'Class Supplier' },
    { id: 'rune_etched_focus', name: 'Rune-Etched Focus', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'runeMage', stats: { power: 14, runeDuration: 8, resourceGain: 4 }, source: 'Class Supplier' },
    { id: 'deadeye_scope', name: 'Deadeye Scope', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'sniper', stats: { crit: 6, weakPointDuration: 8 }, source: 'Class Supplier' },
    { id: 'trap_kit', name: 'Trap Kit', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'trapper', stats: { trapSpeed: 10, trapDamage: 7 }, source: 'Class Supplier' }
  ].map(attachEquipmentItemAssets));

  const RANDOM_EQUIPMENT_ITEMS = Object.freeze([
    { id: 'adventurer_cutlass', name: 'Adventurer Cutlass', slot: 'weapon', rarity: 'Common', level: 8, classId: 'any', stats: { power: 18, speed: 3 }, source: 'World drop', visualId: 'copper_sword', assetId: 'copper_sword', dropOnly: true },
    { id: 'balanced_focus', name: 'Balanced Focus', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'any', stats: { power: 32, resourceGain: 3 }, source: 'World drop', visualId: 'birch_wand', assetId: 'birch_wand', dropOnly: true },
    { id: 'wanderer_charm', name: 'Wanderer Charm', slot: 'amulet', rarity: 'Uncommon', level: 12, classId: 'any', stats: { hp: 36, resourceGain: 2 }, source: 'World drop', assetId: 'plain_ring', dropOnly: true },
    { id: 'fieldguard_helm', name: 'Fieldguard Helm', slot: 'head', rarity: 'Common', level: 10, classId: 'any', stats: { hp: 34, defense: 5 }, source: 'World drop', assetId: 'stitched_vest', dropOnly: true },
    { id: 'trailwoven_gloves', name: 'Trailwoven Gloves', slot: 'gloves', rarity: 'Common', level: 14, classId: 'any', stats: { power: 6, speed: 5 }, source: 'World drop', assetId: 'traveler_boots', dropOnly: true },

    { id: 'vanguard_blade', name: 'Vanguard Blade', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'fighter', stats: { power: 38, hp: 42, armorBreak: 3 }, source: 'World drop', visualId: 'iron_sword', assetId: 'iron_sword', dropOnly: true },
    { id: 'bulwark_plate', name: 'Bulwark Plate', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'fighter', stats: { hp: 105, defense: 16, block: 3 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'stitched_vest', dropOnly: true },
    { id: 'breaker_gauntlets', name: 'Breaker Gauntlets', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'fighter', stats: { power: 18, armorBreak: 8, defense: 6 }, source: 'World drop', assetId: 'iron_axe', dropOnly: true },
    { id: 'sentinel_greaves', name: 'Sentinel Greaves', slot: 'boots', rarity: 'Rare', level: 34, classId: 'fighter', stats: { hp: 84, defense: 13, block: 4 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'traveler_boots', dropOnly: true },

    { id: 'starglass_staff', name: 'Starglass Staff', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'mage', stats: { power: 36, mpMax: 36, areaDamage: 4 }, source: 'World drop', visualId: 'apprentice_staff', assetId: 'apprentice_staff', dropOnly: true },
    { id: 'runewoven_robes', name: 'Runewoven Robes', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'mage', stats: { mpMax: 70, defense: 9, resourceGain: 4 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'stitched_vest', dropOnly: true },
    { id: 'channeler_gloves', name: 'Channeler Gloves', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'mage', stats: { power: 15, areaDamage: 7, resourceMax: 8 }, source: 'World drop', assetId: 'rune_etched_focus', dropOnly: true },
    { id: 'aetherstep_boots', name: 'Aetherstep Boots', slot: 'boots', rarity: 'Rare', level: 34, classId: 'mage', stats: { speed: 20, mpMax: 48, resourceGain: 5 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'traveler_boots', dropOnly: true },

    { id: 'ranger_recurve', name: 'Ranger Recurve', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'archer', stats: { power: 35, range: 34, crit: 3 }, source: 'World drop', visualId: 'oak_longbow', assetId: 'oak_longbow', dropOnly: true },
    { id: 'pathfinder_leathers', name: 'Pathfinder Leathers', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'archer', stats: { hp: 64, defense: 10, speed: 9 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'stitched_vest', dropOnly: true },
    { id: 'deadeye_wraps', name: 'Deadeye Wraps', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'archer', stats: { power: 14, critDamage: 10, range: 18 }, source: 'World drop', assetId: 'deadeye_scope', dropOnly: true },
    { id: 'windrunner_boots', name: 'Windrunner Boots', slot: 'boots', rarity: 'Rare', level: 34, classId: 'archer', stats: { speed: 28, avoid: 6, crit: 4 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'traveler_boots', dropOnly: true }
  ].map(attachEquipmentItemAssets));

  const EQUIPMENT_SETS = Object.freeze([
    Object.freeze({ id: 'thorncrown_regalia', name: 'Thorncrown Regalia', bossId: 'brambleking', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ hp: 120, defense: 10 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 10, resourceGain: 4 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ armorBreak: 8, areaDamage: 6 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 6, crit: 4 }) })
    ]) }),
    Object.freeze({ id: 'furnaceheart_arsenal', name: 'Furnaceheart Arsenal', bossId: 'emberjawGolem', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ burnDamage: 12 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 14, resourceGain: 5 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ areaDamage: 10, defense: 12 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 8, critDamage: 16 }) })
    ]) }),
    Object.freeze({ id: 'titanwork_aegis', name: 'Titanwork Aegis', bossId: 'clockworkTitan', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ defense: 20, block: 5 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ armorBreak: 12, resourceGain: 6 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ power: 16, crit: 4 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 8, hp: 250 }) })
    ]) }),
    Object.freeze({ id: 'deepcore_colossus', name: 'Deepcore Colossus', bossId: 'quarryColossus', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ defense: 28, hp: 220 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 22, armorBreak: 14 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ critDamage: 18, block: 5 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 10, areaDamage: 10 }) })
    ]) }),
    Object.freeze({ id: 'stormcaller_tempest', name: 'Stormcaller Tempest', bossId: 'stormbreakRoc', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ speed: 24, avoid: 6 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ crit: 8, range: 42 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ areaDamage: 14, resourceGain: 8 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 10, critDamage: 24 }) })
    ]) }),
    Object.freeze({ id: 'astral_index', name: 'Astral Index', bossId: 'astralArchivist', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ mpMax: 120, resourceMax: 18 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ power: 24, resourceGain: 10 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ range: 50, areaDamage: 16 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 12, crit: 8 }) })
    ]) }),
    Object.freeze({ id: 'eclipse_paragon', name: 'Eclipse Paragon', bossId: 'eclipseSovereign', pieceBonuses: Object.freeze([
      Object.freeze({ pieces: 2, stats: Object.freeze({ power: 26, defense: 24 }) }),
      Object.freeze({ pieces: 3, stats: Object.freeze({ crit: 10, resourceGain: 10 }) }),
      Object.freeze({ pieces: 4, stats: Object.freeze({ hp: 420, critDamage: 28 }) }),
      Object.freeze({ pieces: 5, stats: Object.freeze({ damageFloor: 14, areaDamage: 18 }) })
    ]) })
  ]);

  const BOSS_EQUIPMENT_SOURCES = Object.freeze([
    Object.freeze({ bossId: 'brambleking', name: 'Brambleking, Crowned Root', level: 35, rarity: 'Epic', setId: 'thorncrown_regalia', dropChance: 0.1 }),
    Object.freeze({ bossId: 'emberjawGolem', name: 'Emberjaw Prime', level: 45, rarity: 'Epic', setId: 'furnaceheart_arsenal', dropChance: 0.1 }),
    Object.freeze({ bossId: 'clockworkTitan', name: 'Clockwork Titan Mk II', level: 55, rarity: 'Epic', setId: 'titanwork_aegis', dropChance: 0.1 }),
    Object.freeze({ bossId: 'quarryColossus', name: 'Quarry Colossus, Deepcore Awakened', level: 65, rarity: 'Relic', setId: 'deepcore_colossus', dropChance: 0.06 }),
    Object.freeze({ bossId: 'stormbreakRoc', name: 'Aurelion, Stormbreak Roc', level: 78, rarity: 'Relic', setId: 'stormcaller_tempest', dropChance: 0.06 }),
    Object.freeze({ bossId: 'astralArchivist', name: 'The Astral Archivist', level: 92, rarity: 'Relic', setId: 'astral_index', dropChance: 0.06 }),
    Object.freeze({ bossId: 'eclipseSovereign', name: 'Eclipse Sovereign', level: 105, rarity: 'Relic', setId: 'eclipse_paragon', dropChance: 0.06 })
  ]);

  const DROP_ECONOMY = Object.freeze({
    normalDropChance: 0.12,
    eliteDropChance: 0.45,
    bossLootChance: 0.7,
    bossPity: Object.freeze({
      epicStart: 8,
      epicStep: 0.01,
      epicMax: 0.1,
      relicStart: 10,
      relicStep: 0.0075,
      relicMax: 0.09
    }),
	    globalRareChance: Object.freeze({
	      normal: 0.0025,
	      elite: 0.0075,
	      boss: 0.015,
	      specialElite: 0.02
	    }),
		    dropTableChances: Object.freeze({
		      coins: 0.5,
		      potions: 0.12,
		      equipment: 0.05,
		      bonusMaterials: 0.06,
		      cards: 0.035,
		      plinkoBalls: 0.04
		    }),
		    dropTableCaps: Object.freeze({
		      coins: 0.95,
		      potions: 0.6,
		      equipment: 0.4,
		      bonusMaterials: 0.4,
		      cards: 0.25,
		      plinkoBalls: 0.65,
		      rareValuables: 0.1
		    }),
	    classWeights: Object.freeze({
	      currentClass: 4,
	      universal: 3,
	      offClass: 1
	    }),
    bossPieceWeights: Object.freeze({
      missing: 6,
      duplicate: 1
    }),
    lootWeights: Object.freeze({
      equipment: 7,
      rareEquipment: 11,
      primaryEtc: 40,
      secondaryEtc: 9,
      upgradeDust: 6,
      upgradeDustBoosted: 9,
      currency: 42,
      currencyBurst: 58,
      healthPotion: 20,
      resourceTonic: 20,
      campRation: 12,
      townReturnScroll: 5,
      guardTonic: 8,
      swiftstepOil: 8,
	      magnetCharm: 6,
	      xpCoupon12: 2,
	      dropCoupon12: 2,
	      eliteXpCoupon15: 2,
	      eliteDropCoupon15: 2,
	      bossXpCoupon20: 1,
	      bossDropCoupon20: 1,
	      mimicXpCoupon20: 2,
	      mimicDropCoupon20: 2,
	      skillManual: 4,
	      eliteSkillManual: 7,
	      skillReset: 1,
      eliteSkillReset: 2,
      attunementPrism: 1,
      eliteAttunementPrism: 2,
      echoPrism: 1,
      eliteEchoPrism: 1,
      gelDrop: 40,
      oreChunks: 40,
      upgradeCatalyst: 5,
      wardingScroll: 1,
      eliteWardingScroll: 2,
      refinementCore: 1,
      eliteRefinementCore: 2,
	      equipmentSlotCoupon: 1,
	      eliteEquipmentSlotCoupon: 2,
	      mimicEquipmentSlotCoupon: 3,
	      usableSlotCoupon: 1,
	      eliteUsableSlotCoupon: 2,
	      mimicUsableSlotCoupon: 3,
	      etcSlotCoupon: 1,
	      eliteEtcSlotCoupon: 2,
	      mimicEtcSlotCoupon: 3,
	      cardSlotCoupon: 1,
	      eliteCardSlotCoupon: 2,
	      mimicCardSlotCoupon: 3
	    })
	  });

  function bossGearItem(config) {
    const source = BOSS_EQUIPMENT_SOURCES.find((entry) => entry.setId === config.setId) || {};
    return attachEquipmentItemAssets(Object.assign({
      rarity: source.rarity || 'Epic',
      level: source.level || 35,
      cost: 0,
      source: `${source.name || 'Boss'} drop`,
      bossId: source.bossId || config.bossId || '',
      dropOnly: true
    }, config));
  }

  const BOSS_EQUIPMENT_ITEMS = Object.freeze([
    bossGearItem({ id: 'thorncrown_greatsword', name: 'Thorncrown Greatsword', slot: 'weapon', classId: 'fighter', setId: 'thorncrown_regalia', visualId: 'iron_sword', stats: { power: 58, armorBreak: 7, hp: 80 } }),
    bossGearItem({ id: 'thornroot_staff', name: 'Thornroot Staff', slot: 'weapon', classId: 'mage', setId: 'thorncrown_regalia', visualId: 'apprentice_staff', stats: { power: 56, areaDamage: 8, resourceGain: 4 } }),
    bossGearItem({ id: 'briarstring_longbow', name: 'Briarstring Longbow', slot: 'weapon', classId: 'archer', setId: 'thorncrown_regalia', visualId: 'oak_longbow', stats: { power: 55, range: 38, crit: 5 } }),
    bossGearItem({ id: 'briar_crown', name: 'Briar Crown', slot: 'head', classId: 'any', setId: 'thorncrown_regalia', stats: { hp: 70, defense: 8, resourceGain: 3 } }),
    bossGearItem({ id: 'barkplate_harness', name: 'Barkplate Harness', slot: 'chest', classId: 'any', setId: 'thorncrown_regalia', visualId: 'stitched_vest', stats: { hp: 145, defense: 18 } }),
    bossGearItem({ id: 'grasping_thorn_gloves', name: 'Grasping Thorn Gloves', slot: 'gloves', classId: 'any', setId: 'thorncrown_regalia', stats: { power: 11, armorBreak: 6, crit: 3 } }),
    bossGearItem({ id: 'rootstep_greaves', name: 'Rootstep Greaves', slot: 'boots', classId: 'any', setId: 'thorncrown_regalia', visualId: 'traveler_boots', stats: { speed: 18, defense: 8, avoid: 3 } }),

    bossGearItem({ id: 'emberjaw_cleaver', name: 'Emberjaw Cleaver', slot: 'weapon', classId: 'fighter', setId: 'furnaceheart_arsenal', visualId: 'iron_axe', stats: { power: 72, burnDamage: 10, armorBreak: 8 } }),
    bossGearItem({ id: 'magma_scepter', name: 'Magma Scepter', slot: 'weapon', classId: 'mage', setId: 'furnaceheart_arsenal', visualId: 'apprentice_staff', stats: { power: 70, burnDamage: 16, areaDamage: 9 } }),
    bossGearItem({ id: 'cindercoil_bow', name: 'Cindercoil Bow', slot: 'weapon', classId: 'archer', setId: 'furnaceheart_arsenal', visualId: 'oak_longbow', stats: { power: 68, burnDamage: 8, critDamage: 12, range: 34 } }),
    bossGearItem({ id: 'ashen_jaw_helm', name: 'Ashen Jaw Helm', slot: 'head', classId: 'any', setId: 'furnaceheart_arsenal', stats: { power: 10, burnDamage: 8, defense: 8 } }),
    bossGearItem({ id: 'furnaceplate', name: 'Furnaceplate', slot: 'chest', classId: 'any', setId: 'furnaceheart_arsenal', visualId: 'stitched_vest', stats: { hp: 170, defense: 22, burnDamage: 6 } }),
    bossGearItem({ id: 'lavaforged_gauntlets', name: 'Lavaforged Gauntlets', slot: 'gloves', classId: 'any', setId: 'furnaceheart_arsenal', stats: { power: 16, critDamage: 10, burnDamage: 6 } }),
    bossGearItem({ id: 'scorchtrail_boots', name: 'Scorchtrail Boots', slot: 'boots', classId: 'any', setId: 'furnaceheart_arsenal', visualId: 'traveler_boots', stats: { speed: 22, areaDamage: 5, avoid: 4 } }),

    bossGearItem({ id: 'gearcleaver', name: 'Gearcleaver', slot: 'weapon', classId: 'fighter', setId: 'titanwork_aegis', visualId: 'iron_axe', stats: { power: 86, armorBreak: 15, block: 4 } }),
    bossGearItem({ id: 'chrono_staff', name: 'Chrono Staff', slot: 'weapon', classId: 'mage', setId: 'titanwork_aegis', visualId: 'apprentice_staff', stats: { power: 82, resourceGain: 8, armorBreak: 8, areaDamage: 8 } }),
    bossGearItem({ id: 'ratchet_repeater', name: 'Ratchet Repeater', slot: 'weapon', classId: 'archer', setId: 'titanwork_aegis', visualId: 'oak_longbow', stats: { power: 80, crit: 8, armorBreak: 8, range: 42 } }),
    bossGearItem({ id: 'titan_visor', name: 'Titan Visor', slot: 'head', classId: 'any', setId: 'titanwork_aegis', stats: { defense: 18, crit: 4, armorBreak: 5 } }),
    bossGearItem({ id: 'clockplate_harness', name: 'Clockplate Harness', slot: 'chest', classId: 'any', setId: 'titanwork_aegis', visualId: 'stitched_vest', stats: { hp: 240, defense: 34, block: 4 } }),
    bossGearItem({ id: 'gyro_gauntlets', name: 'Gyro Gauntlets', slot: 'gloves', classId: 'any', setId: 'titanwork_aegis', stats: { power: 20, armorBreak: 9, resourceGain: 4 } }),
    bossGearItem({ id: 'springstep_boots', name: 'Springstep Boots', slot: 'boots', classId: 'any', setId: 'titanwork_aegis', visualId: 'traveler_boots', stats: { speed: 26, defense: 12, avoid: 5 } }),

    bossGearItem({ id: 'colossus_maul', name: 'Colossus Maul', slot: 'weapon', classId: 'fighter', setId: 'deepcore_colossus', visualId: 'iron_axe', stats: { power: 106, armorBreak: 20, critDamage: 14, speed: -4 } }),
    bossGearItem({ id: 'geode_scepter', name: 'Geode Scepter', slot: 'weapon', classId: 'mage', setId: 'deepcore_colossus', visualId: 'apprentice_staff', stats: { power: 102, areaDamage: 15, armorBreak: 12, resourceMax: 12 } }),
    bossGearItem({ id: 'oreline_greatbow', name: 'Oreline Greatbow', slot: 'weapon', classId: 'archer', setId: 'deepcore_colossus', visualId: 'oak_longbow', stats: { power: 100, critDamage: 22, armorBreak: 12, range: 50 } }),
    bossGearItem({ id: 'deepcore_helm', name: 'Deepcore Helm', slot: 'head', classId: 'any', setId: 'deepcore_colossus', stats: { hp: 140, defense: 24, block: 4 } }),
    bossGearItem({ id: 'bedrock_plate', name: 'Bedrock Plate', slot: 'chest', classId: 'any', setId: 'deepcore_colossus', visualId: 'stitched_vest', stats: { hp: 330, defense: 46, armorBreak: 6 } }),
    bossGearItem({ id: 'quarry_fists', name: 'Quarry Fists', slot: 'gloves', classId: 'any', setId: 'deepcore_colossus', stats: { power: 28, armorBreak: 12, critDamage: 10 } }),
    bossGearItem({ id: 'stonewake_boots', name: 'Stonewake Boots', slot: 'boots', classId: 'any', setId: 'deepcore_colossus', visualId: 'traveler_boots', stats: { speed: 18, defense: 22, hp: 90 } }),

    bossGearItem({ id: 'stormtalon_saber', name: 'Stormtalon Saber', slot: 'weapon', classId: 'fighter', setId: 'stormcaller_tempest', visualId: 'iron_sword', stats: { power: 122, speed: 22, crit: 10, avoid: 5 } }),
    bossGearItem({ id: 'cloudspine_rod', name: 'Cloudspine Rod', slot: 'weapon', classId: 'mage', setId: 'stormcaller_tempest', visualId: 'apprentice_staff', stats: { power: 118, areaDamage: 18, speed: 18, resourceGain: 8 } }),
    bossGearItem({ id: 'skybreaker_bow', name: 'Skybreaker Bow', slot: 'weapon', classId: 'archer', setId: 'stormcaller_tempest', visualId: 'oak_longbow', stats: { power: 116, range: 72, crit: 12, critDamage: 18 } }),
    bossGearItem({ id: 'rocfeather_mask', name: 'Rocfeather Mask', slot: 'head', classId: 'any', setId: 'stormcaller_tempest', stats: { speed: 18, crit: 5, avoid: 6 } }),
    bossGearItem({ id: 'tempest_mantle', name: 'Tempest Mantle', slot: 'chest', classId: 'any', setId: 'stormcaller_tempest', visualId: 'stitched_vest', stats: { hp: 260, defense: 34, areaDamage: 8 } }),
    bossGearItem({ id: 'lightning_grip_gloves', name: 'Lightning-Grip Gloves', slot: 'gloves', classId: 'any', setId: 'stormcaller_tempest', stats: { power: 32, crit: 7, resourceGain: 6 } }),
    bossGearItem({ id: 'gale_boots', name: 'Gale Boots', slot: 'boots', classId: 'any', setId: 'stormcaller_tempest', visualId: 'traveler_boots', stats: { speed: 38, avoid: 9, range: 24 } }),

    bossGearItem({ id: 'index_blade', name: 'Index Blade', slot: 'weapon', classId: 'fighter', setId: 'astral_index', visualId: 'iron_sword', stats: { power: 138, resourceGain: 12, crit: 10, range: 18 } }),
    bossGearItem({ id: 'starbound_codex', name: 'Starbound Codex', slot: 'weapon', classId: 'mage', setId: 'astral_index', visualId: 'apprentice_staff', stats: { power: 136, mpMax: 120, resourceMax: 24, areaDamage: 20 } }),
    bossGearItem({ id: 'cometstring_bow', name: 'Cometstring Bow', slot: 'weapon', classId: 'archer', setId: 'astral_index', visualId: 'oak_longbow', stats: { power: 132, range: 88, crit: 13, resourceGain: 10 } }),
    bossGearItem({ id: 'archivist_crown', name: 'Archivist Crown', slot: 'head', classId: 'any', setId: 'astral_index', stats: { mpMax: 80, resourceMax: 14, crit: 6 } }),
    bossGearItem({ id: 'astral_robes', name: 'Astral Robes', slot: 'chest', classId: 'any', setId: 'astral_index', visualId: 'stitched_vest', stats: { hp: 280, mpMax: 100, defense: 36, areaDamage: 8 } }),
    bossGearItem({ id: 'scribe_gloves', name: 'Scribe Gloves', slot: 'gloves', classId: 'any', setId: 'astral_index', stats: { power: 36, resourceGain: 9, areaDamage: 8 } }),
    bossGearItem({ id: 'orbit_boots', name: 'Orbit Boots', slot: 'boots', classId: 'any', setId: 'astral_index', visualId: 'traveler_boots', stats: { speed: 30, range: 30, avoid: 7 } }),

    bossGearItem({ id: 'eclipse_edge', name: 'Eclipse Edge', slot: 'weapon', classId: 'fighter', setId: 'eclipse_paragon', visualId: 'iron_sword', stats: { power: 160, crit: 14, critDamage: 28, resourceGain: 10 } }),
    bossGearItem({ id: 'umbral_starstaff', name: 'Umbral Starstaff', slot: 'weapon', classId: 'mage', setId: 'eclipse_paragon', visualId: 'apprentice_staff', stats: { power: 156, areaDamage: 26, crit: 10, resourceMax: 26 } }),
    bossGearItem({ id: 'corona_longbow', name: 'Corona Longbow', slot: 'weapon', classId: 'archer', setId: 'eclipse_paragon', visualId: 'oak_longbow', stats: { power: 154, range: 100, crit: 16, critDamage: 32 } }),
    bossGearItem({ id: 'sovereign_crown', name: 'Sovereign Crown', slot: 'head', classId: 'any', setId: 'eclipse_paragon', stats: { power: 28, crit: 8, defense: 20 } }),
    bossGearItem({ id: 'eclipse_plate', name: 'Eclipse Plate', slot: 'chest', classId: 'any', setId: 'eclipse_paragon', visualId: 'stitched_vest', stats: { hp: 460, defense: 60, power: 18 } }),
    bossGearItem({ id: 'penumbra_gloves', name: 'Penumbra Gloves', slot: 'gloves', classId: 'any', setId: 'eclipse_paragon', stats: { power: 42, critDamage: 22, resourceGain: 8 } }),
    bossGearItem({ id: 'sunfall_boots', name: 'Sunfall Boots', slot: 'boots', classId: 'any', setId: 'eclipse_paragon', visualId: 'traveler_boots', stats: { speed: 34, avoid: 10, areaDamage: 10 } })
  ]);

	  function rateCouponItem(config) {
	    const multiplier = Number(config && config.multiplier || 1);
	    const type = String(config && config.type || '').trim();
	    const typeLabel = type === 'drop' ? 'Drop' : 'XP';
	    const tierLabel = multiplier >= 2 ? 'Radiant' : multiplier >= 1.5 ? 'Greater' : 'Lesser';
	    return Object.freeze({
	      id: `${type}_coupon_${multiplier.toFixed(1).replace('.', '_')}_1h`,
	      name: `${tierLabel} ${typeLabel} Coupon`,
	      icon: type === 'drop' ? 'DRP' : 'XP',
	      rarity: config && config.rarity || (multiplier >= 2 ? 'Epic' : multiplier >= 1.5 ? 'Rare' : 'Uncommon'),
	      effect: `Increase ${type === 'drop' ? 'monster drop chance' : 'XP gains'} by ${multiplier.toFixed(1)}x for 1 hour.`,
	      buffId: `${type}Coupon${Math.round(multiplier * 10)}`,
	      buffDuration: 3600,
	      rateBuffType: type,
	      rateMultiplier: multiplier
	    });
	  }

	  const RATE_COUPON_ITEMS = Object.freeze([
	    rateCouponItem({ type: 'xp', multiplier: 1.2, rarity: 'Uncommon' }),
	    rateCouponItem({ type: 'xp', multiplier: 1.5, rarity: 'Rare' }),
	    rateCouponItem({ type: 'xp', multiplier: 2, rarity: 'Epic' }),
	    rateCouponItem({ type: 'drop', multiplier: 1.2, rarity: 'Uncommon' }),
	    rateCouponItem({ type: 'drop', multiplier: 1.5, rarity: 'Rare' }),
	    rateCouponItem({ type: 'drop', multiplier: 2, rarity: 'Epic' })
	  ]);

  const PLINKO_BOUNCE_COUNT = 8;
  const PLINKO_ACTIVE_DROP_LIMIT = 0;
  const PLINKO_PITY_TARGET = 100;
  const PLINKO_SLOT_PROBABILITIES = Object.freeze([1, 8, 28, 56, 70, 56, 28, 8, 1]);
  const PLINKO_SLOT_DENOMINATOR = 256;
  const PLINKO_SLOT_PROBABILITY_TABLES = Object.freeze({
    5: Object.freeze([1, 4, 6, 4, 1]),
    7: Object.freeze([1, 6, 15, 20, 15, 6, 1]),
    9: PLINKO_SLOT_PROBABILITIES
  });

  function getPlinkoSlotProbabilities(slotCount) {
    const count = Math.max(1, Math.floor(Number(slotCount || 9) || 9));
    return PLINKO_SLOT_PROBABILITY_TABLES[count] || PLINKO_SLOT_PROBABILITIES;
  }

  function getPlinkoSlotDenominator(slotCount) {
    return getPlinkoSlotProbabilities(slotCount).reduce((sum, value) => sum + value, 0);
  }

  function plinkoSlot(config, slotCount) {
    const index = Math.max(0, Math.floor(Number(config.index || 0) || 0));
    const probabilities = getPlinkoSlotProbabilities(slotCount);
    const kind = config.kind || (config.teleport ? 'teleport' : 'reward');
    return Object.freeze(Object.assign({}, config, {
      index,
      kind,
      teleport: kind === 'teleport' || !!config.teleport,
      probability: probabilities[index] || 1,
      probabilityDenominator: getPlinkoSlotDenominator(slotCount),
      reward: Object.freeze(config.reward || {})
    }));
  }

  function plinkoBoard(boardId, slots, pityReward, options) {
    const settings = options || {};
    const normalizedSlots = slots || [];
    const slotCount = normalizedSlots.length;
    return Object.freeze({
      id: boardId,
      tier: settings.tier || boardId,
      stage: settings.stage || 'main',
      title: settings.title || `${boardId} Plinko`,
      slotCount,
      slots: Object.freeze(normalizedSlots.map((slot, index) => plinkoSlot(Object.assign({ boardId, index }, slot), slotCount))),
      pityReward: Object.freeze(pityReward || {})
    });
  }

  const PLINKO_BOARD_SLOTS = Object.freeze([
    Object.freeze({ id: 'jackpot_left', label: 'Left Jackpot', tone: '#f0648f', jackpot: true }),
    Object.freeze({ id: 'edge_left', label: 'Edge Prize', tone: '#ff9f5a' }),
    Object.freeze({ id: 'rare_left', label: 'Rare Prize', tone: '#b785ff' }),
    Object.freeze({ id: 'uncommon_left', label: 'Bonus Prize', tone: '#71d99b' }),
    Object.freeze({ id: 'center', label: 'Common Prize', tone: '#8bc7ff', common: true }),
    Object.freeze({ id: 'uncommon_right', label: 'Bonus Prize', tone: '#71d99b' }),
    Object.freeze({ id: 'rare_right', label: 'Rare Prize', tone: '#b785ff' }),
    Object.freeze({ id: 'edge_right', label: 'Edge Prize', tone: '#ff9f5a' }),
    Object.freeze({ id: 'jackpot_right', label: 'Right Jackpot', tone: '#f0648f', jackpot: true })
  ]);

  const PLINKO_BALLS = Object.freeze([
    Object.freeze({
      id: 'plinko_ball_basic',
      name: 'Starfall Ball',
      icon: 'PLK',
      rarity: 'Uncommon',
      effect: 'Drop it into Starfall Plinko for a modest reward roll.',
      plinkoBall: true,
      plinkoTier: 'basic',
      cost: 250,
      pity: 1,
      pityTarget: 100
    }),
    Object.freeze({
      id: 'plinko_ball_polished',
      name: 'Polished Starfall Ball',
      icon: 'PLK+',
      rarity: 'Rare',
      effect: 'Drop it into Starfall Plinko for stronger material, coupon, and gear odds.',
      plinkoBall: true,
      plinkoTier: 'polished',
      cost: 1200,
      pity: 1,
      pityTarget: 100
    }),
    Object.freeze({
      id: 'plinko_ball_meteor',
      name: 'Meteor Starfall Ball',
      icon: 'MET',
      rarity: 'Epic',
      effect: 'Drop it into Starfall Plinko for premium reward odds and 100-ball pity.',
      plinkoBall: true,
      plinkoTier: 'meteor',
      cost: 6000,
      pity: 1,
      pityTarget: 100
    })
  ]);

  const PLINKO_BOARDS = Object.freeze({
    basic: plinkoBoard('basic', [
      { id: 'basic_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'basic_bonus', rewardFamily: 'teleport' },
      { id: 'basic_prism_cache', label: 'Prism Cache', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'basic', reward: { materials: { cubeFragment: 10 }, consumables: { potential_cube: 1 } } },
      { id: 'basic_card', label: 'Starter Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'common', reward: { cards: { gel_spark: 1 } } },
      { id: 'basic_coupon', label: 'Rate Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'basic', reward: { consumables: { xp_coupon_1_2_1h: 1 } } },
      { id: 'basic_dust', label: 'Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 14 } } },
      { id: 'basic_catalyst', label: 'Catalyst', tone: '#71d99b', rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 18, upgradeCatalyst: 1 } } },
      { id: 'basic_cutlass', label: 'Gear', tone: '#b785ff', rewardFamily: 'gear', rewardTier: 'uncommon', reward: { items: [{ itemId: 'adventurer_cutlass', rarity: 'Uncommon' }] } },
      { id: 'basic_slot_coupon', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { usable_slot_coupon: 1 } } },
      { id: 'basic_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'basic_bonus', rewardFamily: 'teleport' }
    ], { currency: 900, materials: { upgradeDust: 30, upgradeCatalyst: 2 }, consumables: { drop_coupon_1_2_1h: 1 } }, { tier: 'basic', stage: 'main', title: 'Starfall Board' }),
    basic_bonus: plinkoBoard('basic_bonus', [
      { id: 'basic_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'basic_apex', rewardFamily: 'teleport' },
      { id: 'basic_bonus_coin_pack', label: 'Charm Cache', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'uncommon', reward: { items: [{ itemId: 'wanderer_charm', rarity: 'Uncommon' }], materials: { upgradeCatalyst: 2 } } },
      { id: 'basic_bonus_card', label: 'Wild Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'uncommon', reward: { cards: { vinebinder_loop: 1 } } },
      { id: 'basic_bonus_materials', label: 'Dust Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 36, cubeFragment: 6 } } },
      { id: 'basic_bonus_prisms', label: 'Prism Pair', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'basic', reward: { consumables: { potential_cube: 1, preservation_cube: 1 } } },
      { id: 'basic_bonus_slot_coupon', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { equipment_slot_coupon: 1 } } },
      { id: 'basic_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'basic_apex', rewardFamily: 'teleport' }
    ], { currency: 1600, materials: { upgradeDust: 44, upgradeCatalyst: 3 }, consumables: { xp_coupon_1_5_1h: 1 } }, { tier: 'basic', stage: 'bonus', title: 'Starfall Bonus Board' }),
    basic_apex: plinkoBoard('basic_apex', [
      { id: 'basic_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'rare', reward: { currency: 4200, items: [{ itemId: 'vanguard_blade', rarity: 'Rare', upgrade: 1 }], consumables: { drop_coupon_1_5_1h: 1 } } },
      { id: 'basic_apex_prism_cache', label: 'Prism Cache', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'basic', reward: { materials: { cubeFragment: 18 }, consumables: { potential_cube: 2 } } },
      { id: 'basic_apex_bundle', label: 'Apex Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 62, upgradeCatalyst: 3 } } },
      { id: 'basic_apex_slot_coupon', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { usable_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'basic_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'rare', reward: { currency: 4600, items: [{ itemId: 'runewoven_robes', rarity: 'Rare', upgrade: 1 }], consumables: { xp_coupon_1_5_1h: 1 } } }
    ], { currency: 2400, materials: { upgradeDust: 60, upgradeCatalyst: 4 }, consumables: { drop_coupon_1_5_1h: 1 } }, { tier: 'basic', stage: 'apex', title: 'Starfall Apex Board' }),
    polished: plinkoBoard('polished', [
      { id: 'polished_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'polished_bonus', rewardFamily: 'teleport' },
      { id: 'polished_rare_gear', label: 'Rare Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'rare', reward: { items: [{ itemId: 'vanguard_blade', rarity: 'Rare', upgrade: 1 }] } },
      { id: 'polished_card', label: 'Rare Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'rare', reward: { cards: { mimic_cache: 1 } } },
      { id: 'polished_coupon', label: 'Rate Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'rare', reward: { consumables: { xp_coupon_1_5_1h: 1 } } },
      { id: 'polished_dust', label: 'Dust Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 48, upgradeCatalyst: 1 } } },
      { id: 'polished_core', label: 'Core Bundle', tone: '#71d99b', rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeCatalyst: 4, refinementCore: 1 } } },
      { id: 'polished_prism', label: 'Prism', tone: '#b785ff', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 1, preservation_cube: 1 } } },
      { id: 'polished_storage', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'polished_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'polished_bonus', rewardFamily: 'teleport' }
    ], { currency: 3200, materials: { upgradeCatalyst: 6, cubeFragment: 14, refinementCore: 1 }, consumables: { xp_coupon_1_5_1h: 1 } }, { tier: 'polished', stage: 'main', title: 'Polished Board' }),
    polished_bonus: plinkoBoard('polished_bonus', [
      { id: 'polished_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'polished_apex', rewardFamily: 'teleport' },
      { id: 'polished_bonus_currency', label: 'Gear Trove', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'rare', reward: { items: [{ itemId: 'thornroot_staff', rarity: 'Rare', upgrade: 1 }], materials: { upgradeCatalyst: 6 } } },
      { id: 'polished_bonus_card', label: 'Rare Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'rare', reward: { cards: { astral_index: 1 } } },
      { id: 'polished_bonus_materials', label: 'Core Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 82, upgradeCatalyst: 5, cubeFragment: 16 } } },
      { id: 'polished_bonus_prism', label: 'Prism Trove', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 2, preservation_cube: 1 } } },
      { id: 'polished_bonus_slot_coupon', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 1, usable_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'polished_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'polished_apex', rewardFamily: 'teleport' }
    ], { currency: 5200, materials: { upgradeCatalyst: 9, cubeFragment: 24, refinementCore: 2 }, consumables: { drop_coupon_1_5_1h: 1 } }, { tier: 'polished', stage: 'bonus', title: 'Polished Bonus Board' }),
    polished_apex: plinkoBoard('polished_apex', [
      { id: 'polished_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'epic', reward: { currency: 11500, items: [{ itemId: 'thorncrown_greatsword', rarity: 'Epic', upgrade: 2 }], consumables: { drop_coupon_2_0_1h: 1 } } },
      { id: 'polished_apex_prism', label: 'Prism Trove', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 3, preservation_cube: 1 } } },
      { id: 'polished_apex_materials', label: 'Apex Core', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 120, upgradeCatalyst: 8, refinementCore: 2 } } },
      { id: 'polished_apex_slot_coupon', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 2, card_slot_coupon: 1 } } },
      { id: 'polished_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'epic', reward: { currency: 12500, items: [{ itemId: 'briarstring_longbow', rarity: 'Epic', upgrade: 2 }], consumables: { xp_coupon_2_0_1h: 1 } } }
    ], { currency: 7800, materials: { upgradeCatalyst: 12, cubeFragment: 30, refinementCore: 3 }, consumables: { xp_coupon_2_0_1h: 1 } }, { tier: 'polished', stage: 'apex', title: 'Polished Apex Board' }),
    meteor: plinkoBoard('meteor', [
      { id: 'meteor_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'meteor_bonus', rewardFamily: 'teleport' },
      { id: 'meteor_epic_gear', label: 'Epic Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'thorncrown_greatsword', rarity: 'Epic', upgrade: 2 }] } },
      { id: 'meteor_epic_card', label: 'Epic Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'epic', reward: { cards: { rift_splinter: 1 } } },
      { id: 'meteor_coupon', label: 'Power Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'epic', reward: { consumables: { xp_coupon_2_0_1h: 1, drop_coupon_1_5_1h: 1 } } },
      { id: 'meteor_dust', label: 'Meteor Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 72 } } },
      { id: 'meteor_gear_cache', label: 'Meteor Gear Cache', tone: '#71d99b', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'stormtalon_saber', rarity: 'Epic', upgrade: 1 }] } },
      { id: 'meteor_prism', label: 'Prism Trove', tone: '#b785ff', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 3, preservation_cube: 1 } } },
      { id: 'meteor_slot_pack', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 1, usable_slot_coupon: 1, etc_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'meteor_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'meteor_bonus', rewardFamily: 'teleport' }
    ], { currency: 12500, materials: { upgradeCatalyst: 18, cubeFragment: 36, refinementCore: 4 }, consumables: { drop_coupon_2_0_1h: 1 }, cards: { stormbreak_plume: 1 } }, { tier: 'meteor', stage: 'main', title: 'Meteor Board' }),
    meteor_bonus: plinkoBoard('meteor_bonus', [
      { id: 'meteor_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'meteor_apex', rewardFamily: 'teleport' },
      { id: 'meteor_bonus_currency', label: 'Meteor Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'stormtalon_saber', rarity: 'Epic', upgrade: 2 }], materials: { upgradeCatalyst: 18, refinementCore: 4 } } },
      { id: 'meteor_bonus_card', label: 'Epic Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'epic', reward: { cards: { stormbreak_plume: 1 } } },
      { id: 'meteor_bonus_materials', label: 'Core Trove', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 110, upgradeCatalyst: 8, cubeFragment: 20, refinementCore: 2 } } },
      { id: 'meteor_bonus_prism', label: 'Prism Vault', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 4, preservation_cube: 2 } } },
      { id: 'meteor_bonus_slot_pack', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 2, usable_slot_coupon: 1, etc_slot_coupon: 1, card_slot_coupon: 2 } } },
      { id: 'meteor_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'meteor_apex', rewardFamily: 'teleport' }
    ], { currency: 18000, materials: { upgradeCatalyst: 24, cubeFragment: 52, refinementCore: 6 }, consumables: { drop_coupon_2_0_1h: 1 } }, { tier: 'meteor', stage: 'bonus', title: 'Meteor Bonus Board' }),
    meteor_apex: plinkoBoard('meteor_apex', [
      { id: 'meteor_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'relic', reward: { currency: 42000, items: [{ itemId: 'eclipse_edge', rarity: 'Relic', upgrade: 4 }], consumables: { drop_coupon_2_0_1h: 2 } } },
      { id: 'meteor_apex_prism', label: 'Prism Vault', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 6, preservation_cube: 3 } } },
      { id: 'meteor_apex_materials', label: 'Apex Trove', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 170, upgradeCatalyst: 12, cubeFragment: 32, refinementCore: 4 } } },
      { id: 'meteor_apex_slot_pack', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 2, usable_slot_coupon: 2, etc_slot_coupon: 2, card_slot_coupon: 2 } } },
      { id: 'meteor_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'relic', reward: { currency: 46000, items: [{ itemId: 'corona_longbow', rarity: 'Relic', upgrade: 4 }], consumables: { xp_coupon_2_0_1h: 2 } } }
    ], { currency: 30000, materials: { upgradeCatalyst: 32, cubeFragment: 80, refinementCore: 9 }, consumables: { drop_coupon_2_0_1h: 1 }, cards: { eclipse_corona: 1 } }, { tier: 'meteor', stage: 'apex', title: 'Meteor Apex Board' })
  });

  const PLINKO_REWARD_TABLES = PLINKO_BOARDS;

	  const CONSUMABLE_ITEMS = Object.freeze([
	    Object.freeze({ id: 'minor_health_potion', name: 'Minor Health Potion', icon: 'HP', rarity: 'Common', effect: 'Restore 60 HP.', hpFlat: 60 }),
	    Object.freeze({ id: 'minor_resource_tonic', name: 'Minor MP Tonic', icon: 'MP', rarity: 'Common', effect: 'Restore 45 MP.', resourceFlat: 45 }),
	    Object.freeze({ id: 'camp_ration', name: 'Camp Ration', icon: 'RAT', rarity: 'Common', effect: 'Restore 40 HP and 25 MP.', hpFlat: 40, resourceFlat: 25 }),
	    Object.freeze({ id: 'standard_health_potion', name: 'Standard Health Potion', icon: 'HP+', rarity: 'Uncommon', effect: 'Restore 150 HP.', hpFlat: 150 }),
	    Object.freeze({ id: 'standard_resource_tonic', name: 'Standard MP Tonic', icon: 'MP+', rarity: 'Uncommon', effect: 'Restore 90 MP.', resourceFlat: 90 }),
	    Object.freeze({ id: 'field_ration', name: 'Field Ration', icon: 'RAT+', rarity: 'Uncommon', effect: 'Restore 100 HP and 55 MP.', hpFlat: 100, resourceFlat: 55 }),
	    Object.freeze({ id: 'greater_health_potion', name: 'Greater Health Potion', icon: 'GHP', rarity: 'Rare', effect: 'Restore 320 HP.', hpFlat: 320 }),
	    Object.freeze({ id: 'greater_resource_tonic', name: 'Greater MP Tonic', icon: 'GMP', rarity: 'Rare', effect: 'Restore 180 MP.', resourceFlat: 180 }),
	    Object.freeze({ id: 'expedition_ration', name: 'Expedition Ration', icon: 'EXR', rarity: 'Rare', effect: 'Restore 220 HP and 110 MP.', hpFlat: 220, resourceFlat: 110 }),
	    Object.freeze({ id: 'superior_health_potion', name: 'Superior Health Potion', icon: 'SHP', rarity: 'Epic', effect: 'Restore 560 HP.', hpFlat: 560 }),
	    Object.freeze({ id: 'superior_resource_tonic', name: 'Superior MP Tonic', icon: 'SMP', rarity: 'Epic', effect: 'Restore 320 MP.', resourceFlat: 320 }),
	    Object.freeze({ id: 'hero_ration', name: 'Hero Ration', icon: 'HER', rarity: 'Epic', effect: 'Restore 380 HP and 190 MP.', hpFlat: 380, resourceFlat: 190 }),
	    Object.freeze({ id: 'town_return_scroll', name: 'Town Return Scroll', icon: 'TWN', effect: 'Return to the nearest regional town.', returnMapId: 'starfallCrossing', dynamicTownReturn: true }),
	    Object.freeze({ id: 'guard_tonic', name: 'Guard Tonic', icon: 'GRD', effect: 'Take reduced damage for 12 seconds.', buffId: 'guardTonic', buffDuration: 12 }),
	    Object.freeze({ id: 'swiftstep_oil', name: 'Swiftstep Oil', icon: 'SPD', effect: 'Move faster for 12 seconds.', buffId: 'swiftstepOil', buffDuration: 12 }),
	    Object.freeze({ id: 'magnet_charm', name: 'Magnet Charm', icon: 'MAG', effect: 'Increase loot pickup reach for 30 seconds.', buffId: 'magnetCharm', buffDuration: 30 }),
	    ...RATE_COUPON_ITEMS,
	    ...PLINKO_BALLS,
	    Object.freeze({ id: 'pet_whistle', name: 'Pet Whistle', icon: 'PET', effect: 'Permanently unlock Pet Assist automation.', petUnlock: true }),
	    Object.freeze({ id: 'equipment_slot_coupon', name: 'Equipment Slot Coupon', icon: 'EQP', effect: 'Expands the Equipment inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'equipment' }),
	    Object.freeze({ id: 'usable_slot_coupon', name: 'Usable Slot Coupon', icon: 'USE', effect: 'Expands the Usable inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'usable' }),
	    Object.freeze({ id: 'etc_slot_coupon', name: 'Etc Slot Coupon', icon: 'ETC', effect: 'Expands the Etc inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'etc' }),
	    Object.freeze({ id: 'card_slot_coupon', name: 'Card Slot Coupon', icon: 'CRD', effect: 'Expands the Cards inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'cards' }),
	    Object.freeze({ id: 'potential_cube', name: 'Attunement Prism', icon: 'PRS', rarity: 'Rare', effect: 'Retunes tiered attunement bonus lines on selected gear.', potentialCube: true }),
	    Object.freeze({ id: 'preservation_cube', name: 'Echo Prism', icon: 'ECHO', rarity: 'Epic', effect: 'Retunes attunement once, then lets you keep the current attunement or apply the new one.', preservationCube: true }),
	    Object.freeze({ id: 'admin_worldwright_console', name: 'Worldwright Console', icon: 'ADM', rarity: 'Relic', effect: 'Admin-only limit testing console for spawning mobs, granting items, and editing gear.', adminOnly: true, opensAdminConsole: true }),
    Object.freeze({ id: 'base_skill_manual', name: 'Base SP Manual', icon: 'BSP', effect: 'Grants 1 Base SP until the base job can max every skill except one.', skillPointPool: 'baseSkillPoints', skillPointAmount: 1 }),
    Object.freeze({ id: 'advanced_skill_manual', name: 'Advanced SP Manual', icon: 'ASP', effect: 'Grants 1 Advanced SP until the advanced job can max every skill except one.', skillPointPool: 'advancedSkillPoints', skillPointAmount: 1 }),
    Object.freeze({ id: 'skill_reset_scroll', name: 'SP Reset Scroll', icon: 'RST', effect: 'Reset all skill ranks and refund spent SP to the matching job pools.', resetSkillPoints: true }),
    Object.freeze({ id: 'stat_reset_scroll', name: 'Stat Reset Scroll', icon: 'AP', effect: 'Reset all Stat Upgrade Point allocations.', resetStatUpgrades: true })
  ]);

  const STAT_UPGRADE_DEFINITIONS = Object.freeze([
    Object.freeze({ id: 'might', name: 'Might', summary: 'Raises direct attack power.', statBonuses: Object.freeze({ power: 1 }) }),
    Object.freeze({ id: 'vitality', name: 'Vitality', summary: 'Raises maximum HP.', statBonuses: Object.freeze({ hp: 12 }) }),
    Object.freeze({ id: 'guard', name: 'Guard', summary: 'Improves mitigation and blocking.', statBonuses: Object.freeze({ defense: 0.5, block: 0.03 }) }),
    Object.freeze({ id: 'focus', name: 'Focus', summary: 'Raises MP and class-resource gain.', statBonuses: Object.freeze({ mpMax: 8, resourceGain: 0.05 }) }),
    Object.freeze({ id: 'agility', name: 'Agility', summary: 'Improves movement and avoidance.', statBonuses: Object.freeze({ speed: 0.35, avoid: 0.05 }) }),
    Object.freeze({ id: 'precision', name: 'Precision', summary: 'Improves crit chance and crit damage.', statBonuses: Object.freeze({ crit: 0.05, critDamage: 0.12 }) })
  ]);

  function getQuestStatUpgradePoints(config) {
    const reward = config && config.rewards || {};
    if (reward.statUpgradePoints != null) return Math.max(0, Math.floor(Number(reward.statUpgradePoints) || 0));
    return config && config.chainId === 'boss_echoes' ? 2 : 1;
  }

  function freezeQuestReward(reward) {
    const source = reward || {};
    const frozen = Object.assign({}, source);
    ['materials', 'consumables', 'items', 'timedBuffs', 'permanentStats'].forEach((key) => {
      if (frozen[key] && typeof frozen[key] === 'object') frozen[key] = Object.freeze(frozen[key]);
    });
    return Object.freeze(frozen);
  }

  function quest(config) {
    const rewards = Object.assign({}, config.rewards || {}, {
      statUpgradePoints: getQuestStatUpgradePoints(config)
    });
    return Object.freeze(Object.assign({}, config, {
      objectives: Object.freeze((config.objectives || []).map((objective) => Object.freeze(objective))),
      rewards: freezeQuestReward(rewards)
    }));
  }

  const QUESTS = Object.freeze([
    Object.freeze({
      id: 'first_steps',
      title: 'First Steps in Greenroot',
      summary: 'Scout the starter meadow, learn the loot loop, and return with early materials.',
      objectives: Object.freeze([
        Object.freeze({ id: 'travel_greenroot', type: 'travel', mapId: 'greenrootMeadow', count: 1, label: 'Travel to Greenroot Meadow' }),
        Object.freeze({ id: 'defeat_slimelets', type: 'defeat', enemyId: 'slimelet', count: 3, label: 'Defeat 3 Slimelets' }),
        Object.freeze({ id: 'loot_drop', type: 'loot', count: 1, label: 'Loot 1 dropped item' })
      ]),
      rewards: Object.freeze({ xp: 90, currency: 60, materials: Object.freeze({ upgradeDust: 2 }), statUpgradePoints: 1 }),
      nextQuestId: 'field_scout'
    }),
    Object.freeze({
      id: 'field_scout',
      title: 'Thornpath Field Scout',
      summary: 'Push into the thicket and prove you can handle mixed melee and ranged packs.',
      objectives: Object.freeze([
        Object.freeze({ id: 'travel_thornpath', type: 'travel', mapId: 'thornpathThicket', count: 1, label: 'Travel to Thornpath Thicket' }),
        Object.freeze({ id: 'defeat_mossbacks', type: 'defeat', enemyId: 'mossback', count: 2, label: 'Defeat 2 Mossbacks' }),
        Object.freeze({ id: 'defeat_thorns', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
      ]),
      rewards: Object.freeze({ xp: 180, currency: 90, materials: Object.freeze({ upgradeDust: 3, gelDrop: 1 }), statUpgradePoints: 1 }),
      nextQuestId: 'trial_ready'
    }),
    Object.freeze({
      id: 'trial_ready',
      title: 'Ready for Advancement',
      summary: 'Reach the trial tier and complete any branch trial before choosing an advanced class.',
      objectives: Object.freeze([
        Object.freeze({ id: 'reach_20', type: 'level', level: 20, label: 'Reach Level 20' }),
        Object.freeze({ id: 'complete_trial', type: 'trialComplete', count: 1, label: 'Complete any class trial' })
      ]),
      rewards: Object.freeze({ xp: 260, currency: 140, materials: Object.freeze({ upgradeDust: 4, upgradeCatalyst: 1 }), statUpgradePoints: 1 }),
      nextQuestId: 'emberjaw_lair'
    }),
    Object.freeze({
      id: 'emberjaw_lair',
      title: 'Emberjaw Vertical Slice',
      summary: 'Enter the first dungeon, test advanced-class power, and defeat the Emberjaw Golem.',
      objectives: Object.freeze([
        Object.freeze({ id: 'reach_25', type: 'level', level: 25, label: 'Reach Level 25' }),
        Object.freeze({ id: 'clear_emberjaw', type: 'dungeonComplete', dungeonId: 'emberjaw_lair', count: 1, label: 'Clear Emberjaw Lair' })
      ]),
      rewards: Object.freeze({ xp: 520, currency: 260, materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 3 }), statUpgradePoints: 1 }),
      nextQuestId: ''
    }),
    quest({
      id: 'greenroot_samples',
      chainId: 'greenroot_relief',
      title: 'Greenroot Field Samples',
      summary: 'Collect ooze samples and clear the meadow so the guide can stock supplies for new scouts.',
      requiredQuestIds: ['first_steps'],
      objectives: [
        { id: 'defeat_dew_slimes', type: 'defeat', enemyId: 'dewSlime', mapId: 'greenrootMeadow', count: 4, label: 'Defeat 4 Dew Slimes' },
        { id: 'collect_gel_drops', type: 'loot', materialId: 'gelDrop', count: 2, label: 'Collect 2 Gel Drops' }
      ],
      rewards: { xp: 140, currency: 80, materials: { upgradeDust: 3, gelDrop: 1 }, consumables: { minor_health_potion: 2 } }
    }),
    quest({
      id: 'ridge_courier',
      chainId: 'thornpath_ridge',
      title: 'Courier to the Ridge',
      summary: 'Carry Thornpath Scout orders to the Ridge Watch before the bandits settle in.',
      requiredQuestIds: ['field_scout'],
      objectives: [
        { id: 'thin_vines', type: 'defeat', enemyId: 'vineSnapper', mapId: 'thornpathThicket', count: 3, label: 'Defeat 3 Vine Snappers' },
        { id: 'talk_ridge_watch', type: 'talk', npcId: 'ridge_watch', mapId: 'banditRidgeCamp', count: 1, label: 'Report to Ridge Watch' }
      ],
      rewards: { xp: 260, currency: 130, materials: { upgradeDust: 4 }, consumables: { camp_ration: 1 } }
    }),
    quest({
      id: 'ridge_cleanup',
      chainId: 'thornpath_ridge',
      title: 'Ridge Cleanup',
      summary: 'Break the bandit camp foothold and recover upgrade supplies from the ridge.',
      requiredQuestIds: ['ridge_courier'],
      requiredLevel: 18,
      objectives: [
        { id: 'defeat_cutters', type: 'defeat', enemyId: 'banditCutter', mapId: 'banditRidgeCamp', count: 6, label: 'Defeat 6 Bandit Cutters' },
        { id: 'defeat_throwers', type: 'defeat', enemyId: 'banditThrower', mapId: 'banditRidgeCamp', count: 4, label: 'Defeat 4 Bandit Throwers' },
        { id: 'recover_upgrade_dust', type: 'loot', materialId: 'upgradeDust', count: 2, label: 'Recover 2 Upgrade Dust' }
      ],
      rewards: { xp: 520, currency: 240, materials: { upgradeDust: 7 }, consumables: { guard_tonic: 1 } }
    }),
    quest({
      id: 'bramble_crown_report',
      chainId: 'thornpath_ridge',
      title: 'Bramble Crown Report',
      summary: 'Push through Bramble Depths and bring proof that the old root crown can be contained.',
      requiredQuestIds: ['ridge_cleanup'],
      requiredLevel: 25,
      objectives: [
        { id: 'clear_bramble_depths', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 1, label: 'Clear Bramble Depths' }
      ],
      rewards: { xp: 820, currency: 420, materials: { upgradeDust: 10, upgradeCatalyst: 2 }, consumables: { base_skill_manual: 1 } }
    }),
    quest({
      id: 'rustcoil_relay',
      chainId: 'rustcoil_front',
      title: 'Rustcoil Relay',
      summary: 'Meet the Rustcoil Surveyor and open a safer path into the construct ruins.',
      requiredQuestIds: ['field_scout'],
      requiredLevel: 12,
      objectives: [
        { id: 'talk_surveyor', type: 'talk', npcId: 'ruins_surveyor', mapId: 'rustcoilRuins', count: 1, label: 'Speak with the Rustcoil Surveyor' }
      ],
      rewards: { xp: 300, currency: 150, materials: { upgradeDust: 5 }, consumables: { minor_resource_tonic: 2 } }
    }),
    quest({
      id: 'rustcoil_reclamation',
      chainId: 'rustcoil_front',
      title: 'Rustcoil Reclamation',
      summary: 'Disable the first construct patrols and recover upgrade dust from the ruined machinery.',
      requiredQuestIds: ['rustcoil_relay'],
      objectives: [
        { id: 'defeat_ratchets', type: 'defeat', enemyId: 'rustRatchet', mapId: 'rustcoilRuins', count: 5, label: 'Defeat 5 Rust Ratchets' },
        { id: 'defeat_clockbugs', type: 'defeat', enemyId: 'clockbug', mapId: 'rustcoilRuins', count: 4, label: 'Defeat 4 Clockbugs' },
        { id: 'collect_rust_dust', type: 'loot', materialId: 'upgradeDust', count: 3, label: 'Collect 3 Upgrade Dust' }
      ],
      rewards: { xp: 620, currency: 300, materials: { upgradeDust: 10, upgradeCatalyst: 1 }, consumables: { guard_tonic: 1 } }
    }),
    quest({
      id: 'quarry_contract',
      chainId: 'rustcoil_front',
      title: 'Oreback Quarry Contract',
      summary: 'Help the quarry crew reclaim ore lanes and ship raw ore back to Rustcoil Outpost.',
      requiredQuestIds: ['rustcoil_reclamation'],
      requiredLevel: 24,
      objectives: [
        { id: 'defeat_orebacks', type: 'defeat', enemyId: 'orebackBeetle', mapId: 'orebackQuarry', count: 6, label: 'Defeat 6 Oreback Beetles' },
        { id: 'collect_ore_chunks', type: 'loot', materialId: 'oreChunks', count: 4, label: 'Collect 4 Ore Chunks' }
      ],
      rewards: { xp: 880, currency: 430, materials: { oreChunks: 8, upgradeCatalyst: 2 }, consumables: { camp_ration: 2 } }
    }),
    quest({
      id: 'gearworks_vault_report',
      chainId: 'rustcoil_front',
      title: 'Gearworks Vault Report',
      summary: 'Clear the Gearworks Vault and brief the Cinder Envoy on what the constructs were guarding.',
      requiredQuestIds: ['quarry_contract'],
      requiredLevel: 35,
      objectives: [
        { id: 'clear_gearworks', type: 'dungeonComplete', dungeonId: 'gearworks_vault', count: 1, label: 'Clear Gearworks Vault' },
        { id: 'talk_cinder_envoy', type: 'talk', npcId: 'cinder_envoy', mapId: 'cinderRefuge', count: 1, label: 'Report to the Cinder Envoy' }
      ],
      rewards: { xp: 1220, currency: 650, materials: { upgradeDust: 14, upgradeCatalyst: 3 }, consumables: { potential_cube: 1 } }
    }),
    quest({
      id: 'cinder_dispatch',
      chainId: 'cinder_front',
      title: 'Cinder Dispatch',
      summary: 'Carry the refuge orders into Cinder Hollow and find the pathfinder watching the furnace roads.',
      requiredQuestIds: ['trial_ready'],
      requiredLevel: 25,
      objectives: [
        { id: 'talk_pathfinder', type: 'talk', npcId: 'cinder_pathfinder', mapId: 'cinderHollow', count: 1, label: 'Speak with the Cinder Pathfinder' }
      ],
      rewards: { xp: 520, currency: 280, materials: { upgradeDust: 6, upgradeCatalyst: 1 }, consumables: { minor_health_potion: 2, minor_resource_tonic: 2 } }
    }),
    quest({
      id: 'cinder_samples',
      chainId: 'cinder_front',
      title: 'Cinder Samples',
      summary: 'Gather volatile cinder samples while cutting down the fast volcanic packs.',
      requiredQuestIds: ['cinder_dispatch'],
      objectives: [
        { id: 'defeat_lava_ticks', type: 'defeat', enemyId: 'lavaTick', mapId: 'cinderHollow', count: 6, label: 'Defeat 6 Lava Ticks' },
        { id: 'defeat_spitters', type: 'defeat', enemyId: 'cinderSpitter', mapId: 'cinderHollow', count: 4, label: 'Defeat 4 Cinder Spitters' },
        { id: 'collect_catalyst', type: 'loot', materialId: 'upgradeCatalyst', count: 1, label: 'Collect 1 Upgrade Catalyst' }
      ],
      rewards: { xp: 820, currency: 420, materials: { upgradeDust: 10, upgradeCatalyst: 2 }, consumables: { swiftstep_oil: 1 } }
    }),
    quest({
      id: 'emberjaw_report',
      chainId: 'cinder_front',
      title: 'Emberjaw Report',
      summary: 'Confirm Emberjaw Lair is contained and return the report to the refuge.',
      requiredQuestIds: ['cinder_samples'],
      requiredLevel: 25,
      objectives: [
        { id: 'clear_emberjaw_lair', type: 'dungeonComplete', dungeonId: 'emberjaw_lair', count: 1, label: 'Clear Emberjaw Lair' },
        { id: 'report_cinder_envoy', type: 'talk', npcId: 'cinder_envoy', mapId: 'cinderRefuge', count: 1, label: 'Report to the Cinder Envoy' }
      ],
      rewards: { xp: 980, currency: 520, materials: { upgradeDust: 12, upgradeCatalyst: 3 }, consumables: { advanced_skill_manual: 1 } }
    }),
    quest({
      id: 'ashglass_crossing',
      chainId: 'cinder_front',
      title: 'Ashglass Crossing',
      summary: 'Open the ashglass route by meeting the courier and clearing the elite glass trail.',
      requiredQuestIds: ['emberjaw_report'],
      requiredLevel: 40,
      objectives: [
        { id: 'talk_ashglass_courier', type: 'talk', npcId: 'ashglass_courier', mapId: 'ashglassPass', count: 1, label: 'Meet the Ashglass Courier' },
        { id: 'clear_ashglass_wisps', type: 'defeat', enemyId: 'emberWisp', mapId: 'ashglassPass', count: 6, label: 'Defeat 6 Ember Wisps in Ashglass Pass' },
        { id: 'collect_ash_catalysts', type: 'loot', materialId: 'upgradeCatalyst', count: 2, label: 'Collect 2 Upgrade Catalysts' }
      ],
      rewards: { xp: 1400, currency: 760, materials: { upgradeDust: 16, upgradeCatalyst: 4 }, consumables: { guard_tonic: 2 } }
    }),
    quest({
      id: 'frostfen_relay',
      chainId: 'frostfen_front',
      title: 'Frostfen Relay',
      summary: 'Carry the Ashglass route report to Frostfen and meet the tracker at the frozen outskirts.',
      requiredQuestIds: ['ashglass_crossing'],
      requiredLevel: 45,
      objectives: [
        { id: 'talk_frostfen_tracker', type: 'talk', npcId: 'frostfen_tracker', mapId: 'frostfenOutskirts', count: 1, label: 'Speak with the Frostfen Tracker' }
      ],
      rewards: { xp: 1100, currency: 620, materials: { upgradeDust: 12, cubeFragment: 1 }, consumables: { camp_ration: 2 } }
    }),
    quest({
      id: 'frostfen_field_notes',
      chainId: 'frostfen_front',
      title: 'Frostfen Field Notes',
      summary: 'Document frost packs and recover prism shards for the quartermaster.',
      requiredQuestIds: ['frostfen_relay'],
      objectives: [
        { id: 'defeat_shardlings', type: 'defeat', enemyId: 'shardling', mapId: 'frostfenOutskirts', count: 7, label: 'Defeat 7 Shardlings' },
        { id: 'defeat_scouts', type: 'defeat', enemyId: 'frostlingScout', mapId: 'frostfenOutskirts', count: 5, label: 'Defeat 5 Frostling Scouts' },
        { id: 'collect_prism_shard', type: 'loot', materialId: 'cubeFragment', count: 1, label: 'Recover 1 Prism Shard' }
      ],
      rewards: { xp: 1580, currency: 860, materials: { upgradeDust: 18, cubeFragment: 2 }, consumables: { potential_cube: 1 } }
    }),
    quest({
      id: 'glacier_cartography',
      chainId: 'frostfen_front',
      title: 'Glacier Cartography',
      summary: 'Map the glacier ridge, break sentinel positions, and return the chart to Frostfen Camp.',
      requiredQuestIds: ['frostfen_field_notes'],
      requiredLevel: 52,
      objectives: [
        { id: 'defeat_sentinels', type: 'defeat', enemyId: 'glacierSentinel', mapId: 'glacierSpine', count: 5, label: 'Defeat 5 Glacier Sentinels' },
        { id: 'defeat_brutes', type: 'defeat', enemyId: 'rimebackBrute', mapId: 'glacierSpine', count: 5, label: 'Defeat 5 Rimeback Brutes' },
        { id: 'return_quartermaster', type: 'talk', npcId: 'frostfen_quartermaster', mapId: 'frostfenCamp', count: 1, label: 'Return the chart to Frostfen Camp' }
      ],
      rewards: { xp: 1950, currency: 1040, materials: { upgradeCatalyst: 4, cubeFragment: 2 }, consumables: { preservation_cube: 1 } }
    }),
    quest({
      id: 'rimewarden_sanctum_report',
      chainId: 'frostfen_front',
      title: 'Rimewarden Sanctum Report',
      summary: 'Clear the sanctum and prepare the highland route for Stormbreak support.',
      requiredQuestIds: ['glacier_cartography'],
      requiredLevel: 58,
      objectives: [
        { id: 'clear_rimewarden', type: 'dungeonComplete', dungeonId: 'rimewarden_sanctum', count: 1, label: 'Clear Rimewarden Sanctum' }
      ],
      rewards: { xp: 2400, currency: 1300, materials: { upgradeCatalyst: 5, cubeFragment: 3, refinementCore: 1 }, consumables: { advanced_skill_manual: 1 } }
    }),
    quest({
      id: 'stormbreak_orders',
      chainId: 'stormbreak_front',
      title: 'Stormbreak Orders',
      summary: 'Meet the cliff scout and prepare lightning rods for the upper routes.',
      requiredQuestIds: ['rimewarden_sanctum_report'],
      requiredLevel: 60,
      objectives: [
        { id: 'talk_stormbreak_scout', type: 'talk', npcId: 'stormbreak_scout', mapId: 'stormbreakCliffs', count: 1, label: 'Speak with the Stormbreak Scout' }
      ],
      rewards: { xp: 1800, currency: 980, materials: { cubeFragment: 2 }, consumables: { swiftstep_oil: 2 } }
    }),
    quest({
      id: 'stormbreak_rods',
      chainId: 'stormbreak_front',
      title: 'Stormbreak Rods',
      summary: 'Clear storm packs and recover prism shards to tune the lightning rods.',
      requiredQuestIds: ['stormbreak_orders'],
      objectives: [
        { id: 'defeat_harriers', type: 'defeat', enemyId: 'galeHarrier', mapId: 'stormbreakCliffs', count: 7, label: 'Defeat 7 Gale Harriers' },
        { id: 'defeat_archers', type: 'defeat', enemyId: 'stormboundArcher', mapId: 'stormbreakCliffs', count: 6, label: 'Defeat 6 Stormbound Archers' },
        { id: 'collect_rod_shards', type: 'loot', materialId: 'cubeFragment', count: 2, label: 'Collect 2 Prism Shards' }
      ],
      rewards: { xp: 2600, currency: 1450, materials: { upgradeCatalyst: 5, cubeFragment: 4 }, consumables: { potential_cube: 2 } }
    }),
    quest({
      id: 'astral_liaison',
      chainId: 'astral_front',
      title: 'Astral Liaison',
      summary: 'Carry Stormbreak findings to the observatory and coordinate with the astral scribe.',
      requiredQuestIds: ['stormbreak_rods'],
      requiredLevel: 70,
      objectives: [
        { id: 'talk_observatory_liaison', type: 'talk', npcId: 'observatory_liaison', mapId: 'astralObservatory', count: 1, label: 'Report to the Observatory Liaison' },
        { id: 'talk_astral_scribe', type: 'talk', npcId: 'astral_scribe', mapId: 'astralArchive', count: 1, label: 'Meet the Astral Scribe' }
      ],
      rewards: { xp: 2200, currency: 1200, materials: { cubeFragment: 3 }, consumables: { camp_ration: 3 } }
    }),
    quest({
      id: 'astral_indexing',
      chainId: 'astral_front',
      title: 'Astral Indexing',
      summary: 'Rebuild damaged archive indices by defeating living entries and recovering prism shards.',
      requiredQuestIds: ['astral_liaison'],
      objectives: [
        { id: 'defeat_scribes', type: 'defeat', enemyId: 'indexScribe', mapId: 'astralArchive', count: 8, label: 'Defeat 8 Index Scribes' },
        { id: 'defeat_sentinels', type: 'defeat', enemyId: 'lumenSentinel', mapId: 'astralArchive', count: 6, label: 'Defeat 6 Lumen Sentinels' },
        { id: 'collect_archive_shards', type: 'loot', materialId: 'cubeFragment', count: 3, label: 'Collect 3 Prism Shards' }
      ],
      rewards: { xp: 3400, currency: 1900, materials: { cubeFragment: 6, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
    }),
    quest({
      id: 'eclipse_frontier_message',
      chainId: 'eclipse_front',
      title: 'Message to Eclipse Frontier',
      summary: 'Carry the observatory warning to the frontier envoy before the rift pressure builds.',
      requiredQuestIds: ['astral_indexing'],
      requiredLevel: 85,
      objectives: [
        { id: 'talk_eclipse_envoy', type: 'talk', npcId: 'eclipse_envoy', mapId: 'eclipseFrontier', count: 1, label: 'Deliver the message to the Eclipse Envoy' }
      ],
      rewards: { xp: 2600, currency: 1500, materials: { cubeFragment: 4 }, consumables: { advanced_skill_manual: 1 } }
    }),
    quest({
      id: 'eclipse_frontier_patrol',
      chainId: 'eclipse_front',
      title: 'Eclipse Frontier Patrol',
      summary: 'Hold the outer frontier by thinning elite duelists and recovering enough prism shards to reinforce the line.',
      requiredQuestIds: ['eclipse_frontier_message'],
      objectives: [
        { id: 'defeat_duelists', type: 'defeat', enemyId: 'eclipseDuelist', mapId: 'eclipseFrontier', count: 9, label: 'Defeat 9 Eclipse Duelists' },
        { id: 'defeat_void_motes', type: 'defeat', enemyId: 'voidMote', mapId: 'eclipseFrontier', count: 7, label: 'Defeat 7 Void Motes' },
        { id: 'collect_frontier_shards', type: 'loot', materialId: 'cubeFragment', count: 4, label: 'Collect 4 Prism Shards' }
      ],
      rewards: { xp: 4300, currency: 2400, materials: { cubeFragment: 8, refinementCore: 2 }, consumables: { potential_cube: 2, preservation_cube: 1 } }
    }),
    quest({
      id: 'rift_watch',
      chainId: 'eclipse_front',
      title: 'Rift Watch',
      summary: 'Test the endless rift front and recover refined cores from the most unstable enemies.',
      requiredQuestIds: ['eclipse_frontier_patrol'],
      requiredLevel: 100,
      objectives: [
        { id: 'defeat_aberrations', type: 'defeat', enemyId: 'riftAberration', mapId: 'endlessRift', count: 8, label: 'Defeat 8 Rift Aberrations' },
        { id: 'collect_rift_cores', type: 'loot', materialId: 'refinementCore', count: 1, label: 'Recover 1 Refinement Core' }
      ],
      rewards: { xp: 5600, currency: 3200, materials: { cubeFragment: 10, refinementCore: 3 }, consumables: { preservation_cube: 2 } }
    }),
    quest({
      id: 'brambleking_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Brambleking Court',
      summary: 'Enter the Brambleking echo room and defeat the crowned root before it spreads again.',
      requiredQuestIds: ['bramble_crown_report'],
      requiredLevel: 32,
      objectives: [
        { id: 'reach_bramble_court', type: 'travel', mapId: 'bramblekingCourt', count: 1, label: 'Enter Brambleking Court' },
        { id: 'defeat_brambleking_echo', type: 'defeatBoss', bossId: 'brambleking', mapId: 'bramblekingCourt', count: 1, label: 'Defeat Brambleking' }
      ],
      rewards: { xp: 1800, currency: 760, materials: { upgradeCatalyst: 3, cubeFragment: 2 }, consumables: { potential_cube: 1 } }
    }),
    quest({
      id: 'titan_foundry_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Titan Foundry',
      summary: 'Challenge the Clockwork Titan in its foundry echo and recover the exposed core readings.',
      requiredQuestIds: ['gearworks_vault_report'],
      requiredLevel: 50,
      objectives: [
        { id: 'reach_titan_foundry', type: 'travel', mapId: 'titanFoundry', count: 1, label: 'Enter Titan Foundry' },
        { id: 'defeat_titan_echo', type: 'defeatBoss', bossId: 'clockworkTitan', mapId: 'titanFoundry', count: 1, label: 'Defeat Clockwork Titan' }
      ],
      rewards: { xp: 2600, currency: 1200, materials: { upgradeCatalyst: 4, cubeFragment: 3 }, consumables: { potential_cube: 1 } }
    }),
    quest({
      id: 'deepcore_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Deepcore Core',
      summary: 'Fight the Quarry Colossus inside the deepcore echo and break its ore armor.',
      requiredQuestIds: ['gearworks_vault_report'],
      requiredLevel: 60,
      objectives: [
        { id: 'reach_deepcore', type: 'travel', mapId: 'deepcoreCore', count: 1, label: 'Enter Deepcore Core' },
        { id: 'defeat_colossus_echo', type: 'defeatBoss', bossId: 'quarryColossus', mapId: 'deepcoreCore', count: 1, label: 'Defeat Quarry Colossus' }
      ],
      rewards: { xp: 3200, currency: 1550, materials: { upgradeCatalyst: 5, cubeFragment: 3, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
    }),
    quest({
      id: 'emberjaw_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Emberjaw Furnace',
      summary: 'Face Emberjaw in the furnace echo and use its overheat windows to end the fight cleanly.',
      requiredQuestIds: ['emberjaw_report'],
      requiredLevel: 42,
      objectives: [
        { id: 'reach_emberjaw_furnace', type: 'travel', mapId: 'emberjawFurnace', count: 1, label: 'Enter Emberjaw Furnace' },
        { id: 'defeat_emberjaw_echo', type: 'defeatBoss', bossId: 'emberjawGolem', mapId: 'emberjawFurnace', count: 1, label: 'Defeat Emberjaw Golem' }
      ],
      rewards: { xp: 2300, currency: 1080, materials: { upgradeCatalyst: 4, cubeFragment: 3 }, consumables: { potential_cube: 1 } }
    }),
    quest({
      id: 'rimewarden_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Rimewarden Vault',
      summary: 'Enter the Rimewarden echo and hold position through whiteout lanes and frost rings.',
      requiredQuestIds: ['rimewarden_sanctum_report'],
      requiredLevel: 66,
      objectives: [
        { id: 'reach_rimewarden_vault', type: 'travel', mapId: 'rimewardenVault', count: 1, label: 'Enter Rimewarden Vault' },
        { id: 'defeat_rimewarden_echo', type: 'defeatBoss', bossId: 'rimewarden', mapId: 'rimewardenVault', count: 1, label: 'Defeat Rimewarden' }
      ],
      rewards: { xp: 3800, currency: 1900, materials: { cubeFragment: 5, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
    }),
    quest({
      id: 'stormbreak_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Stormbreak Aerie',
      summary: 'Fight Aurelion in the aerie echo and manage lightning rods through divebomb phases.',
      requiredQuestIds: ['stormbreak_rods'],
      requiredLevel: 76,
      objectives: [
        { id: 'reach_stormbreak_aerie', type: 'travel', mapId: 'stormbreakAerie', count: 1, label: 'Enter Stormbreak Aerie' },
        { id: 'defeat_roc_echo', type: 'defeatBoss', bossId: 'stormbreakRoc', mapId: 'stormbreakAerie', count: 1, label: 'Defeat Aurelion' }
      ],
      rewards: { xp: 4500, currency: 2300, materials: { cubeFragment: 6, refinementCore: 1 }, consumables: { potential_cube: 2 } }
    }),
    quest({
      id: 'astral_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Astral Stacks',
      summary: 'Challenge the Astral Archivist echo and keep skill variety high through mirrored pages.',
      requiredQuestIds: ['astral_indexing'],
      requiredLevel: 88,
      objectives: [
        { id: 'reach_astral_stacks', type: 'travel', mapId: 'astralStacks', count: 1, label: 'Enter Astral Stacks' },
        { id: 'defeat_archivist_echo', type: 'defeatBoss', bossId: 'astralArchivist', mapId: 'astralStacks', count: 1, label: 'Defeat the Astral Archivist' }
      ],
      rewards: { xp: 5400, currency: 2800, materials: { cubeFragment: 8, refinementCore: 2 }, consumables: { preservation_cube: 1 } }
    }),
    quest({
      id: 'eclipse_echo',
      chainId: 'boss_echoes',
      title: 'Echo: Eclipse Throne',
      summary: 'Enter the sovereign echo and survive solar and lunar stance swaps through totality.',
      requiredQuestIds: ['eclipse_frontier_patrol'],
      requiredLevel: 100,
      objectives: [
        { id: 'reach_eclipse_throne', type: 'travel', mapId: 'eclipseThrone', count: 1, label: 'Enter Eclipse Throne' },
        { id: 'defeat_sovereign_echo', type: 'defeatBoss', bossId: 'eclipseSovereign', mapId: 'eclipseThrone', count: 1, label: 'Defeat Eclipse Sovereign' }
      ],
      rewards: { xp: 6800, currency: 3600, materials: { cubeFragment: 10, refinementCore: 3 }, consumables: { preservation_cube: 2, advanced_skill_manual: 1 } }
    })
  ]);

  const DUNGEONS = Object.freeze([
    Object.freeze({
      id: 'bramble_depths',
      name: 'Bramble Depths',
      summary: 'An early MVP Alpha dungeon built around control, fire payoff, and vertical add pressure.',
      mapId: 'brambleDepths',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'brambleking',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 360, currency: 180, materials: Object.freeze({ upgradeDust: 5, gelDrop: 2 }) })
    }),
    Object.freeze({
      id: 'emberjaw_lair',
      name: 'Emberjaw Lair',
      summary: 'A compact party-style dungeon that culminates in the Emberjaw Golem boss fight.',
      mapId: 'emberjawLair',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'emberjawGolem',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 420, currency: 220, materials: Object.freeze({ upgradeDust: 6, upgradeCatalyst: 1 }) })
    }),
    Object.freeze({
      id: 'gearworks_vault',
      name: 'Gearworks Vault',
      summary: 'A late prototype dungeon that checks armor break, add control, and boss uptime.',
      mapId: 'gearworksVault',
      levelRequirement: 35,
      recommendedPartySize: 4,
      bossId: 'quarryColossus',
      bossIds: Object.freeze(['clockworkTitan', 'quarryColossus']),
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 620, currency: 320, materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 3 }) })
    }),
    Object.freeze({
      id: 'rimewarden_sanctum',
      name: 'Rimewarden Sanctum',
      summary: 'A Frostfen dungeon built around slick footing, frost flyers, and Rimewarden arena control.',
      mapId: 'rimewardenSanctum',
      levelRequirement: 58,
      recommendedPartySize: 4,
      bossId: 'rimewarden',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 760, currency: 380, materials: Object.freeze({ upgradeDust: 9, upgradeCatalyst: 3, refinementCore: 1 }) })
    })
  ]);

  const ROSTER_TRAITS = Object.freeze([
    Object.freeze({ id: 'guardian_bulwark', name: 'Guardian Bulwark', sourceAdvancedId: 'guardian', summary: 'A roster bonus from proving the Guardian path.', statBonuses: Object.freeze({ hp: 60, defense: 4 }) }),
    Object.freeze({ id: 'berserker_fervor', name: 'Berserker Fervor', sourceAdvancedId: 'berserker', summary: 'A roster bonus from proving the Berserker path.', statBonuses: Object.freeze({ power: 3, resourceGain: 2 }) }),
    Object.freeze({ id: 'duelist_tempo', name: 'Duelist Tempo', sourceAdvancedId: 'duelist', summary: 'A roster bonus from proving the Duelist path.', statBonuses: Object.freeze({ speed: 6, crit: 2 }) }),
    Object.freeze({ id: 'fire_mage_kindling', name: 'Kindling Memory', sourceAdvancedId: 'fireMage', summary: 'A roster bonus from proving the Fire Mage path.', statBonuses: Object.freeze({ power: 2, burnDamage: 5 }) }),
    Object.freeze({ id: 'rune_mage_pattern', name: 'Pattern Memory', sourceAdvancedId: 'runeMage', summary: 'A roster bonus from proving the Rune Mage path.', statBonuses: Object.freeze({ mpMax: 35, resourceGain: 3 }) }),
    Object.freeze({ id: 'storm_mage_charge', name: 'Storm Charge', sourceAdvancedId: 'stormMage', summary: 'A roster bonus from proving the Storm Mage path.', statBonuses: Object.freeze({ mpMax: 30, areaDamage: 3 }) }),
    Object.freeze({ id: 'sniper_focus', name: 'Sniper Focus', sourceAdvancedId: 'sniper', summary: 'A roster bonus from proving the Sniper path.', statBonuses: Object.freeze({ crit: 3, range: 18 }) }),
    Object.freeze({ id: 'trapper_routes', name: 'Trapper Routes', sourceAdvancedId: 'trapper', summary: 'A roster bonus from proving the Trapper path.', statBonuses: Object.freeze({ speed: 8, trapDamage: 4 }) }),
    Object.freeze({ id: 'beast_archer_bond', name: 'Pack Bond', sourceAdvancedId: 'beastArcher', summary: 'A roster bonus from proving the Beast Archer path.', statBonuses: Object.freeze({ hp: 35, resourceGain: 3 }) }),
    Object.freeze({ id: 'dungeon_veteran', name: 'Dungeon Veteran', sourceDungeonId: 'emberjaw_lair', summary: 'Unlocked by clearing Emberjaw Lair.', statBonuses: Object.freeze({ hp: 40, power: 2 }) }),
    Object.freeze({ id: 'vaultbreaker', name: 'Vaultbreaker', sourceDungeonId: 'gearworks_vault', summary: 'Unlocked by clearing Gearworks Vault.', statBonuses: Object.freeze({ defense: 3, armorBreak: 4 }) })
  ]);

  const CLASS_TRIALS = Object.freeze([
    Object.freeze({
      id: 'guardian_trial',
      advancedId: 'guardian',
      baseClass: 'fighter',
      title: 'Guardian Trial: Hold the Line',
      summary: 'Break sturdy beasts in Thornpath to prove you can turn pressure into protection.',
      levelRequirement: 20,
      mapId: 'thornpathThicket',
      objectives: Object.freeze([
        Object.freeze({ id: 'mossbacks', type: 'defeat', enemyId: 'mossback', count: 3, label: 'Defeat 3 Mossbacks' }),
        Object.freeze({ id: 'boars', type: 'defeat', enemyId: 'bristleBoar', count: 2, label: 'Defeat 2 Bristle Boars' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'berserker_trial',
      advancedId: 'berserker',
      baseClass: 'fighter',
      title: 'Berserker Trial: Blood Rush',
      summary: 'Hunt fast Dust Imps and prove you can stay aggressive under pressure.',
      levelRequirement: 20,
      mapId: 'rustcoilRuins',
      objectives: Object.freeze([
        Object.freeze({ id: 'dust_imps', type: 'defeat', enemyId: 'dustImp', count: 5, label: 'Defeat 5 Dust Imps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 1, label: 'Defeat 1 Clockbug' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'duelist_trial',
      advancedId: 'duelist',
      baseClass: 'fighter',
      title: 'Duelist Trial: Clean Openings',
      summary: 'Challenge Bandit Ridge and prove you can create precise openings under pressure.',
      levelRequirement: 20,
      mapId: 'banditRidgeCamp',
      objectives: Object.freeze([
        Object.freeze({ id: 'cutters', type: 'defeat', enemyId: 'banditCutter', count: 3, label: 'Defeat 3 Bandit Cutters' }),
        Object.freeze({ id: 'throwers', type: 'defeat', enemyId: 'banditThrower', count: 2, label: 'Defeat 2 Bandit Throwers' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'fire_mage_trial',
      advancedId: 'fireMage',
      baseClass: 'mage',
      title: 'Fire Mage Trial: Ember Control',
      summary: 'Challenge Cinder Hollow spirits and prove you can manage explosive area pressure.',
      levelRequirement: 20,
      mapId: 'cinderHollow',
      objectives: Object.freeze([
        Object.freeze({ id: 'wisps', type: 'defeat', enemyId: 'emberWisp', count: 5, label: 'Defeat 5 Ember Wisps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 1, label: 'Defeat 1 Clockbug' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'rune_mage_trial',
      advancedId: 'runeMage',
      baseClass: 'mage',
      title: 'Rune Mage Trial: Pattern Study',
      summary: 'Study construct movement in Rustcoil and prove you can handle setup combat.',
      levelRequirement: 20,
      mapId: 'rustcoilRuins',
      objectives: Object.freeze([
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 4, label: 'Defeat 4 Clockbugs' }),
        Object.freeze({ id: 'thorn_sprouts', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'storm_mage_trial',
      advancedId: 'stormMage',
      baseClass: 'mage',
      title: 'Storm Mage Trial: Conductive Lines',
      summary: 'Fight constructs and ember spirits while learning to chain damage through crowded lanes.',
      levelRequirement: 20,
      mapId: 'cinderHollow',
      objectives: Object.freeze([
        Object.freeze({ id: 'wisps', type: 'defeat', enemyId: 'emberWisp', count: 4, label: 'Defeat 4 Ember Wisps' }),
        Object.freeze({ id: 'clockbugs', type: 'defeat', enemyId: 'clockbug', count: 2, label: 'Defeat 2 Clockbugs' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'sniper_trial',
      advancedId: 'sniper',
      baseClass: 'archer',
      title: 'Sniper Trial: Priority Targets',
      summary: 'Pick off Bandit Throwers from ridge platforms to prove precision target control.',
      levelRequirement: 20,
      mapId: 'banditRidgeCamp',
      objectives: Object.freeze([
        Object.freeze({ id: 'throwers', type: 'defeat', enemyId: 'banditThrower', count: 4, label: 'Defeat 4 Bandit Throwers' }),
        Object.freeze({ id: 'cutters', type: 'defeat', enemyId: 'banditCutter', count: 2, label: 'Defeat 2 Bandit Cutters' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'trapper_trial',
      advancedId: 'trapper',
      baseClass: 'archer',
      title: 'Trapper Trial: Route Control',
      summary: 'Stop charging beasts in Thornpath and prove you can control enemy lanes.',
      levelRequirement: 20,
      mapId: 'thornpathThicket',
      objectives: Object.freeze([
        Object.freeze({ id: 'boars', type: 'defeat', enemyId: 'bristleBoar', count: 4, label: 'Defeat 4 Bristle Boars' }),
        Object.freeze({ id: 'thorn_sprouts', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    }),
    Object.freeze({
      id: 'beast_archer_trial',
      advancedId: 'beastArcher',
      baseClass: 'archer',
      title: 'Beast Archer Trial: Pack Routes',
      summary: 'Clear Bramble lanes and quarry supports while proving you can coordinate companion pressure.',
      levelRequirement: 20,
      mapId: 'orebackQuarry',
      objectives: Object.freeze([
        Object.freeze({ id: 'beetles', type: 'defeat', enemyId: 'orebackBeetle', count: 3, label: 'Defeat 3 Oreback Beetles' }),
        Object.freeze({ id: 'healers', type: 'defeat', enemyId: 'glowcapHealer', count: 2, label: 'Defeat 2 Glowcap Healers' })
      ]),
      rewards: Object.freeze({ currency: 120, materials: Object.freeze({ upgradeDust: 4 }) })
    })
  ]);

  const SPECIALIZATIONS = Object.freeze([
    Object.freeze({ id: 'guardian_aegis_captain', advancedId: 'guardian', name: 'Aegis Captain', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Leans into shields, boss control, and party mitigation.', statBonuses: Object.freeze({ hp: 120, defense: 8, block: 4 }) }),
    Object.freeze({ id: 'berserker_crimson_reaver', advancedId: 'berserker', name: 'Crimson Reaver', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Turns missing health and risk windows into sustained damage.', statBonuses: Object.freeze({ power: 8, resourceGain: 4 }) }),
    Object.freeze({ id: 'duelist_blade_dancer', advancedId: 'duelist', name: 'Blade Dancer', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Improves haste, precision consistency, and short repositioning windows.', statBonuses: Object.freeze({ speed: 16, crit: 5, critDamage: 10 }) }),
    Object.freeze({ id: 'fire_mage_ash_caller', advancedId: 'fireMage', name: 'Ash Caller', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Focuses Heat into larger burn fields and boss burst windows.', statBonuses: Object.freeze({ power: 7, burnDamage: 12, areaDamage: 4 }) }),
    Object.freeze({ id: 'rune_mage_seal_architect', advancedId: 'runeMage', name: 'Seal Architect', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Strengthens rune duration, MP capacity, and field control.', statBonuses: Object.freeze({ mpMax: 70, runeDuration: 10, resourceGain: 4 }) }),
    Object.freeze({ id: 'storm_mage_tempest_weaver', advancedId: 'stormMage', name: 'Tempest Weaver', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Adds reliable area damage and MP depth to lightning chains.', statBonuses: Object.freeze({ mpMax: 55, areaDamage: 8, speed: 8 }) }),
    Object.freeze({ id: 'sniper_deadeye_commander', advancedId: 'sniper', name: 'Deadeye Commander', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Rewards long-range weak point uptime with heavy precision scaling.', statBonuses: Object.freeze({ crit: 7, critDamage: 14, range: 34 }) }),
    Object.freeze({ id: 'trapper_field_engineer', advancedId: 'trapper', name: 'Field Engineer', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Improves trap setup speed, trap payoff, and controlled kill zones.', statBonuses: Object.freeze({ trapSpeed: 12, trapDamage: 10, defense: 3 }) }),
    Object.freeze({ id: 'beast_archer_pack_warden', advancedId: 'beastArcher', name: 'Pack Warden', levelRequirement: SPECIALIZATION_LEVEL, summary: 'Builds companion sustain, Bond generation, and flexible party support.', statBonuses: Object.freeze({ hp: 85, resourceGain: 5, avoid: 3 }) })
  ]);

  const MARKET_LISTINGS = Object.freeze([
    Object.freeze({ id: 'dust_cache', name: 'Upgrade Dust Cache', summary: 'A repeatable material bundle for early upgrade attempts.', cost: 180, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 8 }) }) }),
    Object.freeze({ id: 'catalyst_cache', name: 'Catalyst Cache', summary: 'Chance boosters for risky upgrade attempts.', cost: 320, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 2 }) }) }),
	    Object.freeze({ id: 'warding_scroll_offer', name: 'Warding Scroll', summary: 'Protects gear from one selected destroy-risk upgrade attempt.', cost: 420, reward: Object.freeze({ materials: Object.freeze({ wardingScroll: 1 }) }) }),
	    Object.freeze({ id: 'refinement_core_offer', name: 'Refinement Core', summary: 'Enhances one selected successful upgrade attempt.', cost: 560, reward: Object.freeze({ materials: Object.freeze({ refinementCore: 1 }) }) }),
	    Object.freeze({ id: 'field_supply_crate', name: 'Field Supply Crate', summary: 'Consumables for longer map sessions and dungeon attempts.', cost: 220, reward: Object.freeze({ consumables: Object.freeze({ camp_ration: 2, minor_resource_tonic: 2 }) }) }),
	    Object.freeze({ id: 'lesser_xp_coupon_offer', name: 'Lesser XP Coupon', summary: 'A one-hour 1.2x XP boost for focused leveling sessions.', cost: 480, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_1_2_1h: 1 }) }) }),
	    Object.freeze({ id: 'greater_xp_coupon_offer', name: 'Greater XP Coupon', summary: 'A one-hour 1.5x XP boost for dungeon and boss routing.', cost: 1250, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_1_5_1h: 1 }) }) }),
	    Object.freeze({ id: 'radiant_xp_coupon_offer', name: 'Radiant XP Coupon', summary: 'A one-hour 2.0x XP boost for high-effort progression pushes.', cost: 3000, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_2_0_1h: 1 }) }) }),
	    Object.freeze({ id: 'lesser_drop_coupon_offer', name: 'Lesser Drop Coupon', summary: 'A one-hour 1.2x monster drop chance boost.', cost: 520, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_1_2_1h: 1 }) }) }),
	    Object.freeze({ id: 'greater_drop_coupon_offer', name: 'Greater Drop Coupon', summary: 'A one-hour 1.5x monster drop chance boost.', cost: 1350, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_1_5_1h: 1 }) }) }),
	    Object.freeze({ id: 'radiant_drop_coupon_offer', name: 'Radiant Drop Coupon', summary: 'A one-hour 2.0x monster drop chance boost.', cost: 3200, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_2_0_1h: 1 }) }) }),
	    Object.freeze({ id: 'equipment_slot_coupon_offer', name: 'Equipment Coupon Offer', summary: 'A rare account-service purchase that expands gear storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ equipment_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'usable_slot_coupon_offer', name: 'Usable Coupon Offer', summary: 'A rare account-service purchase that expands usable storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ usable_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'etc_slot_coupon_offer', name: 'Etc Coupon Offer', summary: 'A rare account-service purchase that expands material storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ etc_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'card_slot_coupon_offer', name: 'Card Coupon Offer', summary: 'A rare account-service purchase that expands card storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ card_slot_coupon: 1 }) }) })
	  ]);

  const COSMETICS = Object.freeze([
    Object.freeze({ id: 'crossing_cape', name: 'Crossing Cape', slot: 'aura', icon: 'CPE', cost: 160, summary: 'A clean starter cape cosmetic for town and field play.' }),
    Object.freeze({ id: 'ember_trim', name: 'Ember Trim', slot: 'aura', icon: 'EMB', cost: 240, summary: 'A warm orange combat accent inspired by Emberjaw Lair.' }),
    Object.freeze({ id: 'vault_spark', name: 'Vault Spark', slot: 'aura', icon: 'VLT', cost: 280, summary: 'A blue-white gearwork spark cosmetic from construct regions.' }),
    Object.freeze({ id: 'ember_impact_splat', name: 'Ember Impact', slot: 'damageSplat', icon: 'EIM', cost: 0, cashShopOnly: true, summary: 'Hot orange damage numbers with ember flares and slash sparks.', damageSplatStyle: Object.freeze({ id: 'ember_impact', variant: 'ember', color: '#ffe16a', criticalColor: '#fff27a', stroke: 'rgba(94, 26, 26, 0.94)', burstColor: '#ff5d5d', accentColor: '#ff8a3d', ringColor: '#fff27a', slashColor: '#ff3d2e', shardColor: '#ffbe55', secondaryShardColor: '#ff5d5d' }) }),
    Object.freeze({ id: 'vault_arc_splat', name: 'Vault Arc', slot: 'damageSplat', icon: 'VAR', cost: 0, cashShopOnly: true, summary: 'Gearwork blue damage numbers with electric arcs and clean rings.', damageSplatStyle: Object.freeze({ id: 'vault_arc', variant: 'vault', color: '#d9f8ff', criticalColor: '#f4fdff', stroke: 'rgba(10, 48, 82, 0.94)', burstColor: '#7bdff2', accentColor: '#ffffff', ringColor: '#9cf6ff', slashColor: '#68a9ff', shardColor: '#7bdff2', secondaryShardColor: '#ffffff' }) }),
    Object.freeze({ id: 'frost_shatter_splat', name: 'Frost Shatter', slot: 'damageSplat', icon: 'FRS', cost: 0, cashShopOnly: true, summary: 'Crisp cyan damage numbers with icy shards and a sharp pop.', damageSplatStyle: Object.freeze({ id: 'frost_shatter', variant: 'frost', color: '#e9fbff', criticalColor: '#ffffff', stroke: 'rgba(22, 61, 92, 0.94)', burstColor: '#a8f1ff', accentColor: '#f7fbff', ringColor: '#b8e6ff', slashColor: '#7bdff2', shardColor: '#e9fbff', secondaryShardColor: '#7bdff2' }) }),
    Object.freeze({ id: 'astral_prism_splat', name: 'Astral Prism', slot: 'damageSplat', icon: 'APR', cost: 0, cashShopOnly: true, summary: 'Violet-gold damage numbers with prism flashes and orbiting shards.', damageSplatStyle: Object.freeze({ id: 'astral_prism', variant: 'astral', color: '#fff0a6', criticalColor: '#fff7d6', stroke: 'rgba(50, 31, 90, 0.94)', burstColor: '#c794ff', accentColor: '#ffe16a', ringColor: '#c794ff', slashColor: '#8f8cff', shardColor: '#ffe16a', secondaryShardColor: '#c794ff' }) }),
    Object.freeze({ id: 'stormcall_strike_splat', name: 'Stormcall Strike', slot: 'damageSplat', icon: 'STM', cost: 0, cashShopOnly: true, summary: 'Yellow-blue damage numbers with fast lightning slashes.', damageSplatStyle: Object.freeze({ id: 'stormcall_strike', variant: 'storm', color: '#fff7b8', criticalColor: '#ffffff', stroke: 'rgba(30, 45, 91, 0.94)', burstColor: '#68a9ff', accentColor: '#ffe16a', ringColor: '#7bdff2', slashColor: '#ffe16a', shardColor: '#68a9ff', secondaryShardColor: '#fff7b8' }) }),
    Object.freeze({ id: 'verdant_bloom_splat', name: 'Verdant Bloom', slot: 'damageSplat', icon: 'VBL', cost: 0, cashShopOnly: true, summary: 'Green-gold damage numbers with petal arcs and a soft bloom.', damageSplatStyle: Object.freeze({ id: 'verdant_bloom', variant: 'verdant', color: '#e6ffd1', criticalColor: '#f7ffe6', stroke: 'rgba(30, 73, 43, 0.94)', burstColor: '#8ec878', accentColor: '#fff0a6', ringColor: '#68d58d', slashColor: '#8ec878', shardColor: '#e6ffd1', secondaryShardColor: '#68d58d' }) }),
    Object.freeze({ id: 'founder_spark', name: 'Founder Spark', slot: 'aura', icon: 'FND', cost: 0, summary: 'Season reward cosmetic for clearing the Beta foundation goals.', seasonReward: true })
  ]);

  const CASH_SHOP_CATEGORIES = Object.freeze([
    Object.freeze({ id: 'featured', label: 'Featured' }),
    Object.freeze({ id: 'cosmetics', label: 'Cosmetics' }),
    Object.freeze({ id: 'effects', label: 'Effects' }),
    Object.freeze({ id: 'buffs', label: 'Buffs' }),
    Object.freeze({ id: 'owned', label: 'Owned' })
  ]);

  const CASH_SHOP_ITEMS = Object.freeze([
    Object.freeze({ id: 'crossing_cape_cash', name: 'Crossing Cape', category: 'cosmetics', featured: true, kind: 'cosmetic', cosmeticId: 'crossing_cape', icon: 'CPE', price: 80, summary: 'A clean starter cape cosmetic for town and field play.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'ember_trim_cash', name: 'Ember Trim', category: 'effects', featured: true, kind: 'cosmetic', cosmeticId: 'ember_trim', icon: 'EMB', price: 120, summary: 'A warm orange combat accent inspired by Emberjaw Lair.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'vault_spark_cash', name: 'Vault Spark', category: 'effects', kind: 'cosmetic', cosmeticId: 'vault_spark', icon: 'VLT', price: 160, summary: 'A blue-white gearwork spark cosmetic from construct regions.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'ember_impact_splat_cash', name: 'Ember Impact', category: 'effects', kind: 'cosmetic', cosmeticId: 'ember_impact_splat', icon: 'EIM', price: 95, summary: 'Hot orange damage numbers with ember flares and slash sparks.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'vault_arc_splat_cash', name: 'Vault Arc', category: 'effects', kind: 'cosmetic', cosmeticId: 'vault_arc_splat', icon: 'VAR', price: 110, summary: 'Gearwork blue damage numbers with electric arcs and clean rings.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'frost_shatter_splat_cash', name: 'Frost Shatter', category: 'effects', kind: 'cosmetic', cosmeticId: 'frost_shatter_splat', icon: 'FRS', price: 125, summary: 'Crisp cyan damage numbers with icy shards and a sharp pop.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'astral_prism_splat_cash', name: 'Astral Prism', category: 'effects', kind: 'cosmetic', cosmeticId: 'astral_prism_splat', icon: 'APR', price: 145, summary: 'Violet-gold damage numbers with prism flashes and orbiting shards.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'stormcall_strike_splat_cash', name: 'Stormcall Strike', category: 'effects', kind: 'cosmetic', cosmeticId: 'stormcall_strike_splat', icon: 'STM', price: 160, summary: 'Yellow-blue damage numbers with fast lightning slashes.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'verdant_bloom_splat_cash', name: 'Verdant Bloom', category: 'effects', kind: 'cosmetic', cosmeticId: 'verdant_bloom_splat', icon: 'VBL', price: 105, summary: 'Green-gold damage numbers with petal arcs and a soft bloom.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'guard_tonic_pack', name: 'Guard Tonic Pack', category: 'buffs', featured: true, kind: 'buffBundle', icon: 'GRD', price: 45, weeklyLimit: 3, summary: 'Three normal Guard Tonics. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ guard_tonic: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'swiftstep_oil_pack', name: 'Swiftstep Oil Pack', category: 'buffs', kind: 'buffBundle', icon: 'SFT', price: 45, weeklyLimit: 3, summary: 'Three normal Swiftstep Oils. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ swiftstep_oil: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'magnet_charm_pack', name: 'Magnet Charm Pack', category: 'buffs', kind: 'buffBundle', icon: 'MAG', price: 60, weeklyLimit: 3, summary: 'Two normal Magnet Charms. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ magnet_charm: 2 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'field_ration_crate', name: 'Field Ration Crate', category: 'buffs', kind: 'buffBundle', icon: 'RAT', price: 35, weeklyLimit: 3, summary: 'Three Camp Rations for longer field sessions.', reward: Object.freeze({ consumables: Object.freeze({ camp_ration: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) })
  ]);

  const SEASONS = Object.freeze([
    Object.freeze({
      id: 'beta_foundations',
      name: 'Beta Foundations',
      active: true,
      summary: 'Prototype season goals aimed at proving dungeons, loot, and expanded class paths.',
      objectives: Object.freeze([
        Object.freeze({ id: 'field_bosses', type: 'defeatBoss', count: 2, label: 'Defeat 2 bosses' }),
        Object.freeze({ id: 'dungeon_clears', type: 'dungeonComplete', count: 2, label: 'Clear 2 dungeons' }),
        Object.freeze({ id: 'advanced_path', type: 'advancedClass', count: 1, label: 'Choose an advanced class' })
      ]),
      rewards: Object.freeze({ currency: 300, starTokens: 180, materials: Object.freeze({ upgradeDust: 10, upgradeCatalyst: 1 }), cosmeticId: 'founder_spark' })
    })
  ]);

  function partyAiLoadout(equipment, skills) {
    return Object.freeze({
      equipment: Object.freeze(Object.assign({}, equipment || {})),
      skills: Object.freeze((skills || []).map((entry) => Object.freeze(typeof entry === 'string' ? { skillId: entry } : Object.assign({}, entry))))
    });
  }

  const PARTY_AI_LOADOUTS = Object.freeze({
    fighter: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots' }, [
      { skillId: 'fighter_power_break', minCooldown: 6.2, priority: 3 },
      { skillId: 'fighter_ground_slam', minCooldown: 5.4, priority: 2 },
      { skillId: 'fighter_heavy_strike', minCooldown: 1.75, priority: 1 }
    ]),
    guardian: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots', offhand: 'guardian_tower_shield' }, [
      { skillId: 'guardian_oath_barrier', minCooldown: 12, priority: 4, support: true },
      { skillId: 'guardian_retaliation_wave', minCooldown: 8, priority: 3 },
      { skillId: 'guardian_shield_bash', minCooldown: 1.85, priority: 1 }
    ]),
    berserker: partyAiLoadout({ weapon: 'iron_axe', chest: 'party_plate', boots: 'traveler_boots', offhand: 'berserker_war_grip' }, [
      { skillId: 'berserker_war_cry', minCooldown: 18, priority: 4, support: true },
      { skillId: 'berserker_crimson_recovery', minCooldown: 10, priority: 3 },
      { skillId: 'berserker_blood_cleave', minCooldown: 1.9, priority: 1 }
    ]),
    duelist: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots' }, [
      { skillId: 'duelist_rallying_flourish', minCooldown: 16, priority: 3, support: true },
      { skillId: 'duelist_quick_cut', minCooldown: 1.65, priority: 1 }
    ]),
    mage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots' }, [
      { skillId: 'mage_spell_mark', minCooldown: 5.4, priority: 3 },
      { skillId: 'mage_arcane_burst', minCooldown: 5.6, priority: 2 },
      { skillId: 'mage_magic_bolt', minCooldown: 1.8, priority: 1 }
    ]),
    fireMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'ember_core' }, [
      { skillId: 'fire_mage_wildfire', minCooldown: 10, priority: 4 },
      { skillId: 'fire_mage_burning_mark', minCooldown: 6, priority: 3 },
      { skillId: 'fire_mage_fireball', minCooldown: 1.95, priority: 1 }
    ]),
    runeMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'rune_etched_focus' }, [
      { skillId: 'rune_mage_ground_glyph', minCooldown: 9, priority: 4 },
      { skillId: 'rune_mage_arcane_link', minCooldown: 7, priority: 3 },
      { skillId: 'rune_mage_rune_mark', minCooldown: 1.85, priority: 1 }
    ]),
    stormMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'rune_etched_focus' }, [
      { skillId: 'storm_mage_stormfront', minCooldown: 18, priority: 4, support: true },
      { skillId: 'storm_mage_chain_bolt', minCooldown: 2.05, priority: 1 }
    ]),
    archer: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots' }, [
      { skillId: 'archer_marked_shot', minCooldown: 4.5, priority: 3 },
      { skillId: 'archer_piercing_arrow', minCooldown: 5.4, priority: 2 },
      { skillId: 'archer_quick_shot', minCooldown: 1.7, priority: 1 }
    ]),
    sniper: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'deadeye_scope' }, [
      { skillId: 'sniper_pierce_armor', minCooldown: 7, priority: 4 },
      { skillId: 'sniper_weak_point_mark', minCooldown: 6, priority: 3 },
      { skillId: 'sniper_aimed_shot', minCooldown: 2.05, priority: 1 }
    ]),
    trapper: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'trap_kit' }, [
      { skillId: 'trapper_spike_trap', minCooldown: 6, priority: 4 },
      { skillId: 'trapper_lure_shot', minCooldown: 7, priority: 3 },
      { skillId: 'trapper_snare_trap', minCooldown: 1.9, priority: 1 }
    ]),
    beastArcher: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'trap_kit' }, [
      { skillId: 'beast_archer_pack_call', minCooldown: 18, priority: 3, support: true },
      { skillId: 'beast_archer_companion_strike', minCooldown: 1.9, priority: 1 }
    ])
  });

  const PROTOTYPE_PARTY_MEMBERS = Object.freeze([
    Object.freeze({
      id: 'aegis_mira',
      name: 'Mira',
      classId: 'guardian',
      role: 'Defense',
      summary: 'A simulated Guardian ally who adds mitigation and occasional shields.',
      statBonuses: Object.freeze({ hp: 42, defense: 3, block: 2 }),
      assist: Object.freeze({ type: 'shield', cooldown: 7.5, shieldPercent: 0.08, color: '#68a9ff' })
    }),
    Object.freeze({
      id: 'cinder_jo',
      name: 'Jo',
      classId: 'fireMage',
      role: 'Mobbing',
      summary: 'A simulated Fire Mage ally who throws small area bursts into packs.',
      statBonuses: Object.freeze({ power: 2, areaDamage: 3, burnDamage: 4 }),
      assist: Object.freeze({ type: 'damage', cooldown: 4.8, powerScale: 0.34, radius: 92, color: '#ff8a3d' })
    }),
    Object.freeze({
      id: 'deadeye_len',
      name: 'Len',
      classId: 'sniper',
      role: 'Bossing',
      summary: 'A simulated Sniper ally who helps mark priority targets.',
      statBonuses: Object.freeze({ crit: 2, critDamage: 5, range: 18 }),
      assist: Object.freeze({ type: 'mark', cooldown: 5.2, powerScale: 0.3, radius: 40, color: '#ffe16a' })
    }),
    Object.freeze({
      id: 'field_tamsin',
      name: 'Tamsin',
      classId: 'trapper',
      role: 'Control',
      summary: 'A simulated Trapper ally who slows clustered enemies during long fights.',
      statBonuses: Object.freeze({ trapDamage: 3, defense: 2, speed: 4 }),
      assist: Object.freeze({ type: 'control', cooldown: 5.8, powerScale: 0.22, radius: 118, color: '#7bdff2' })
    })
  ]);

  const ONBOARDING_STEPS = Object.freeze([
    Object.freeze({ id: 'choose_class', event: 'classSelected', title: 'Choose a class', summary: 'Pick Fighter, Mage, or Archer to enter Starfall Crossing.' }),
    Object.freeze({ id: 'open_worldmap', event: 'openPanel', panelId: 'worldmap', title: 'Check the world map', summary: 'Open the world map to see the current route and nearby areas.' }),
    Object.freeze({ id: 'travel_greenroot', event: 'travel', mapId: 'greenrootMeadow', title: 'Travel to Greenroot', summary: 'Use the Greenroot Gate or world map to enter the first field.' }),
    Object.freeze({ id: 'defeat_enemy', event: 'defeat', title: 'Defeat an enemy', summary: 'Use basic attacks or a bound skill to defeat a field enemy.' }),
    Object.freeze({ id: 'loot_drop', event: 'loot', title: 'Pick up loot', summary: 'Hold the loot key near a dropped item to collect coins, gear, or materials.' }),
    Object.freeze({ id: 'open_inventory', event: 'openPanel', panelId: 'inventory', title: 'Review the item grid', summary: 'Open Inventory to compare level requirements, base strength, and item tier glow.' }),
    Object.freeze({ id: 'equip_item', event: 'equip', title: 'Equip an item', summary: 'Open inventory and equip a stronger item into the matching slot.' }),
    Object.freeze({ id: 'open_skills', event: 'openPanel', panelId: 'skills', title: 'Open your skill batch', summary: 'Open Skills to see trainable nodes, role tags, and next-rank changes.' }),
    Object.freeze({ id: 'rank_skill', event: 'rankSkill', title: 'Rank a skill', summary: 'Spend a skill point in the skill window to improve your class kit.' }),
    Object.freeze({ id: 'open_upgrade', event: 'openPanel', panelId: 'upgrade', title: 'Inspect upgrade risk', summary: 'Open the Upgrade Station to preview before/after power and failure outcomes.' }),
    Object.freeze({ id: 'upgrade_item', event: 'upgrade', title: 'Attempt an upgrade', summary: 'Use Upgrade Dust at the artisan to see the risk table and try an item upgrade.' }),
    Object.freeze({ id: 'open_party', event: 'openPanel', panelId: 'partyPanel', title: 'Open the party panel', summary: 'Open Party to see simulated ally roles, assist timing, and current self-buff behavior.' }),
    Object.freeze({ id: 'find_party', event: 'partyFind', title: 'Find prototype allies', summary: 'Open Party and fill the local simulated party slots for passive assists.' }),
    Object.freeze({ id: 'open_trials', event: 'openPanel', panelId: 'quests', title: 'Check trials', summary: 'Open Quests and switch to Trials when you are ready to preview advanced branches.' }),
    Object.freeze({ id: 'start_trial', event: 'startTrial', title: 'Start a class trial', summary: 'At Level 20, start a branch trial from the Quest window.' }),
    Object.freeze({ id: 'choose_advanced', event: 'advancedClass', title: 'Choose an advanced class', summary: 'Complete a trial and choose your permanent Level 25 advanced branch.' }),
    Object.freeze({ id: 'clear_dungeon', event: 'dungeonComplete', title: 'Clear a dungeon', summary: 'Enter a boss dungeon and defeat its boss to complete the vertical-slice loop.' })
  ]);

  const AUDIO_CUES = Object.freeze({
    uiConfirm: Object.freeze({ label: 'UI Confirm', type: 'tone', frequency: 660, duration: 0.08, gain: 0.045 }),
    attack: Object.freeze({ label: 'Basic Attack', type: 'noise', frequency: 180, duration: 0.07, gain: 0.05 }),
    skill: Object.freeze({ label: 'Skill Cast', type: 'sweep', frequency: 360, endFrequency: 780, duration: 0.12, gain: 0.06 }),
    buff: Object.freeze({ label: 'Buff Cast', type: 'chime', frequency: 520, endFrequency: 1040, duration: 0.18, gain: 0.055 }),
    loot: Object.freeze({ label: 'Loot Pickup', type: 'chime', frequency: 740, endFrequency: 1180, duration: 0.11, gain: 0.045 }),
    level: Object.freeze({ label: 'Level Up', type: 'chime', frequency: 660, endFrequency: 1320, duration: 0.26, gain: 0.07 }),
    upgradeSuccess: Object.freeze({ label: 'Upgrade Success', type: 'chime', frequency: 580, endFrequency: 980, duration: 0.22, gain: 0.065 }),
    upgradeFail: Object.freeze({ label: 'Upgrade Fail', type: 'sweep', frequency: 260, endFrequency: 120, duration: 0.18, gain: 0.045 }),
    damage: Object.freeze({ label: 'Player Hit', type: 'noise', frequency: 120, duration: 0.09, gain: 0.05 }),
    defeat: Object.freeze({ label: 'Enemy Defeat', type: 'sweep', frequency: 420, endFrequency: 190, duration: 0.13, gain: 0.05 }),
    travel: Object.freeze({ label: 'Map Travel', type: 'chime', frequency: 440, endFrequency: 760, duration: 0.2, gain: 0.05 }),
    partyAssist: Object.freeze({ label: 'Party Assist', type: 'chime', frequency: 510, endFrequency: 820, duration: 0.14, gain: 0.04 })
  });

  const ACCOMPLISHMENTS = Object.freeze([
    Object.freeze({
      id: 'first_cache',
      title: 'First Field Cache',
      summary: 'Loot your first handful of field drops.',
      category: 'Collection',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'loot_any', type: 'loot', count: 5, label: 'Loot 5 ground drops' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ minor_health_potion: 2, minor_resource_tonic: 2 }),
        materials: Object.freeze({ upgradeDust: 2 })
      })
    }),
    Object.freeze({
      id: 'slime_sweeper',
      title: 'Slime Sweeper',
      summary: 'Clear enough Slimelets to make Greenroot safer.',
      category: 'Combat',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'slimelets', type: 'defeat', enemyId: 'slimelet', count: 20, label: 'Defeat 20 Slimelets' })
      ]),
      rewards: Object.freeze({
        items: Object.freeze([Object.freeze({ itemId: 'plain_ring', rarity: 'Uncommon', upgrade: 1, source: 'Slime Sweeper accomplishment' })]),
        timedBuffs: Object.freeze([Object.freeze({ buffId: 'guardTonic', duration: 30 })])
      })
    }),
    Object.freeze({
      id: 'greenroot_pathfinder',
      title: 'Greenroot Pathfinder',
      summary: 'Push through Greenroot and Thornpath route objectives.',
      category: 'Exploration',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'greenroot_kills', type: 'defeat', mapId: 'greenrootMeadow', count: 18, label: 'Defeat 18 enemies in Greenroot Meadow' }),
        Object.freeze({ id: 'thornpath_kills', type: 'defeat', mapId: 'thornpathThicket', count: 18, label: 'Defeat 18 enemies in Thornpath Thicket' })
      ]),
      rewards: Object.freeze({
        items: Object.freeze([Object.freeze({ itemId: 'traveler_boots', rarity: 'Rare', upgrade: 2, source: 'Greenroot Pathfinder accomplishment' })]),
        materials: Object.freeze({ upgradeDust: 6, gelDrop: 3 })
      })
    }),
    Object.freeze({
      id: 'upgrade_apprentice',
      title: 'Upgrade Apprentice',
      summary: 'Make several upgrade attempts and learn the gear risk loop.',
      category: 'Crafting',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'upgrade_attempts', type: 'upgrade', count: 5, label: 'Attempt 5 item upgrades' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 1 }),
        materials: Object.freeze({ upgradeCatalyst: 1 })
      })
    }),
    Object.freeze({
      id: 'advanced_path',
      title: 'Advanced Path',
      summary: 'Choose any advanced class after completing its trial.',
      category: 'Progression',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'advanced_class', type: 'advancedClass', count: 1, label: 'Choose an advanced class' })
      ]),
      rewards: Object.freeze({
        currency: 300,
        consumables: Object.freeze({ equipment_slot_coupon: 1 }),
        permanentStats: Object.freeze({ mpMax: 20, resourceGain: 1 })
      })
    }),
    Object.freeze({
      id: 'bramblebreaker',
      title: 'Bramblebreaker',
      summary: 'Clear the Bramble Depths route boss dungeon.',
      category: 'Boss',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'bramble_depths', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 1, label: 'Clear Bramble Depths' })
      ]),
      rewards: Object.freeze({
        currency: 420,
        materials: Object.freeze({ upgradeDust: 10, upgradeCatalyst: 2 }),
        permanentStats: Object.freeze({ hp: 40, defense: 2 })
      })
    }),
    Object.freeze({
      id: 'bossbreaker',
      title: 'Bossbreaker',
      summary: 'Defeat several area bosses across the route chain.',
      category: 'Boss',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'bosses', type: 'defeatBoss', count: 3, label: 'Defeat 3 bosses' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ upgradeDust: 16, upgradeCatalyst: 4 }),
        timedBuffs: Object.freeze([Object.freeze({ buffId: 'swiftstepOil', duration: 45 })]),
        permanentStats: Object.freeze({ power: 3, crit: 2 })
      })
    }),
    Object.freeze({
      id: 'level_60_vanguard',
      title: 'Level 60 Vanguard',
      summary: 'Reach the level 60 specialization milestone.',
      category: 'Progression',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_cap', type: 'level', level: 60, label: 'Reach Level 60' })
      ]),
      rewards: Object.freeze({
        currency: 800,
        cosmeticId: 'vault_spark',
        permanentStats: Object.freeze({ hp: 80, mpMax: 35, power: 2, defense: 2 })
      })
    })
  ]);

  const STARTER_ITEMS = Object.freeze({
    fighter: ['training_sword'],
    mage: ['training_wand'],
    archer: ['training_bow']
  });

  const STARTER_CONSUMABLES = Object.freeze({
    minor_health_potion: 3,
    minor_resource_tonic: 2,
    pet_whistle: 1,
    stat_reset_scroll: 1,
    admin_worldwright_console: 1
  });

	  const UPGRADE_OUTCOMES = Object.freeze([
	    { id: 'success', label: 'Success', weightByRange: [90, 64, 39, 22], effect: 'Increase upgrade level by +1.' },
	    { id: 'fail', label: 'Failure', weightByRange: [10, 36, 53, 62], effect: 'Decrease upgrade level by -1. Upgrade Dust is consumed.' },
	    { id: 'destroy', label: 'Destroy', weightByRange: [0, 0, 8, 16], effect: 'Destroy the item. Upgrade Dust is consumed.' }
	  ]);

	  const UPGRADE_DUST_COST_BY_RANGE = Object.freeze([1, 3, 6, 10]);

	  const UPGRADE_AIDES = Object.freeze([
	    Object.freeze({ id: 'upgradeCatalyst', materialId: 'upgradeCatalyst', type: 'chance', name: 'Upgrade Catalyst', icon: 'UC', rarity: 'Rare', successBonus: 10, summary: 'Adds +10% success chance to the next upgrade attempt.' }),
	    Object.freeze({ id: 'wardingScroll', materialId: 'wardingScroll', type: 'protection', name: 'Warding Scroll', icon: 'WS', rarity: 'Rare', protectsDestroy: true, summary: 'Turns a destroy result into a failure so the item survives.' }),
	    Object.freeze({ id: 'refinementCore', materialId: 'refinementCore', type: 'enhancement', name: 'Refinement Core', icon: 'RC', rarity: 'Epic', successUpgradeBonus: 1, summary: 'Successful upgrades gain +2 instead of +1.' })
	  ]);

	  const POTENTIAL_TIERS = Object.freeze([
	    Object.freeze({ id: 'rare', name: 'Rare Attunement', lineCount: 1, nextTier: 'epic', tierUpChance: 0.12 }),
	    Object.freeze({ id: 'epic', name: 'Epic Attunement', lineCount: 2, nextTier: 'relic', tierUpChance: 0.04 }),
	    Object.freeze({ id: 'relic', name: 'Relic Attunement', lineCount: 3, nextTier: 'mythic', tierUpChance: 0.015 }),
	    Object.freeze({ id: 'mythic', name: 'Mythic Attunement', lineCount: 3, nextTier: 'ascendant', tierUpChance: 0.005 }),
	    Object.freeze({ id: 'ascendant', name: 'Ascendant Attunement', lineCount: 4, nextTier: 'celestial', tierUpChance: 0.0015 }),
	    Object.freeze({ id: 'celestial', name: 'Celestial Attunement', lineCount: 4, nextTier: '', tierUpChance: 0 })
	  ]);

	  const POTENTIAL_LINE_POOLS = Object.freeze([
	    Object.freeze({ stat: 'powerPercent', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 12], mythic: [13, 16], ascendant: [17, 22], celestial: [23, 32] }) }),
	    Object.freeze({ stat: 'attackDamagePercent', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [1, 3], epic: [3, 6], relic: [7, 11], mythic: [12, 15], ascendant: [16, 21], celestial: [22, 30] }) }),
	    Object.freeze({ stat: 'critDamage', slots: Object.freeze(['weapon']), values: Object.freeze({ rare: [4, 7], epic: [8, 14], relic: [16, 26], mythic: [28, 34], ascendant: [36, 44], celestial: [48, 60] }) }),
	    Object.freeze({ stat: 'maxHpPercent', values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 14], mythic: [15, 18], ascendant: [19, 25], celestial: [26, 36] }) }),
	    Object.freeze({ stat: 'defensePercent', values: Object.freeze({ rare: [1, 3], epic: [4, 7], relic: [8, 14], mythic: [15, 18], ascendant: [19, 25], celestial: [26, 36] }) }),
	    Object.freeze({ stat: 'resourceGainPercent', values: Object.freeze({ rare: [2, 5], epic: [6, 10], relic: [12, 20], mythic: [22, 27], ascendant: [29, 36], celestial: [38, 50] }) })
	  ]);

  const MUTATIONS = Object.freeze([
    Object.freeze({ id: 'echoing', name: 'Echoing', effect: 'Skills have a chance to repeat at reduced power.' }),
    Object.freeze({ id: 'splintering', name: 'Splintering', effect: 'Attacks can split into smaller secondary hits.' }),
    Object.freeze({ id: 'guarded', name: 'Guarded', effect: 'Defensive skills grant minor resource.' }),
    Object.freeze({ id: 'burning', name: 'Burning', effect: 'Attacks can apply burn.' }),
    Object.freeze({ id: 'focused', name: 'Focused', effect: 'Marks, runes, or weak points last longer.' }),
    Object.freeze({ id: 'volatile', name: 'Volatile', effect: 'Higher damage but increased resource cost.' })
  ]);

  const MAP_MODIFIERS = Object.freeze([
    Object.freeze({ id: 'overgrown', name: 'Overgrown Lanes', summary: 'Dense terrain slows monsters but improves material yield.', mapTypes: Object.freeze(['field', 'dungeon']), enemySpeedScale: 0.94, lootBonus: 0.08, xpBonus: 0.03 }),
    Object.freeze({ id: 'glass_cannon', name: 'Glass Cannon Packs', summary: 'Enemies hit harder but have lower armor and grant more XP.', mapTypes: Object.freeze(['field', 'rift']), enemyDamageScale: 1.1, enemyDefenseScale: 0.88, xpBonus: 0.08 }),
    Object.freeze({ id: 'treasure_wind', name: 'Treasure Wind', summary: 'More elites and better drop rolls appear while field pressure rises.', mapTypes: Object.freeze(['field', 'rift']), eliteChanceBonus: 0.08, lootBonus: 0.12, enemyHpScale: 1.04 }),
    Object.freeze({ id: 'unstable_floor', name: 'Unstable Floor', summary: 'Dungeon enemies move faster and bosses build break gauge faster.', mapTypes: Object.freeze(['dungeon', 'rift']), enemySpeedScale: 1.08, bossBreakBonus: 0.12, currencyBonus: 0.08 }),
    Object.freeze({ id: 'thick_hide', name: 'Thick Hide', summary: 'Enemies are tougher, but break and armor tools pay out extra.', mapTypes: Object.freeze(['field', 'dungeon', 'rift']), enemyHpScale: 1.08, enemyDefenseScale: 1.08, breakRewardBonus: 0.15 }),
    Object.freeze({ id: 'lucent_cache', name: 'Lucent Cache', summary: 'Target-farm drops and Monster Guide research advance faster.', mapTypes: Object.freeze(['field', 'dungeon', 'rift']), targetFarmBonus: 0.14, researchBonus: 1 })
  ]);

  const ELITE_AFFIXES = Object.freeze([
    Object.freeze({ id: 'bulwark', name: 'Bulwark', summary: 'Higher HP and defense, vulnerable to break effects.', hpScale: 1.22, defenseScale: 1.16, breakTakenScale: 1.18 }),
    Object.freeze({ id: 'swift', name: 'Swift', summary: 'Moves and attacks faster, drops extra currency.', speedScale: 1.18, attackCooldownScale: 0.9, currencyBonus: 0.18 }),
    Object.freeze({ id: 'volatile', name: 'Volatile', summary: 'Deals more damage and grants extra XP.', damageScale: 1.14, xpBonus: 0.18 }),
    Object.freeze({ id: 'mender', name: 'Mender', summary: 'Periodically stabilizes nearby packs and drops more consumables.', hpScale: 1.12, lootBonus: 0.08 }),
    Object.freeze({ id: 'marked', name: 'Marked', summary: 'Starts with a weak point and improves target-farm progress.', weakPointDuration: 5, targetFarmBonus: 0.12 }),
    Object.freeze({ id: 'riftbound', name: 'Riftbound', summary: 'Rift elites scale harder and improve ladder score.', hpScale: 1.18, damageScale: 1.08, riftScoreBonus: 0.2 })
  ]);

  const BOSS_BREAK_PROFILES = Object.freeze([
    Object.freeze({ bossId: 'brambleking', name: 'Thorn Heart', maxGauge: 140, duration: 4.2, damageTakenScale: 1.18, reward: Object.freeze({ materials: Object.freeze({ gelDrop: 1 }) }) }),
    Object.freeze({ bossId: 'clockworkTitan', name: 'Gear Exposure', maxGauge: 165, duration: 4.4, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ oreChunks: 1 }) }) }),
    Object.freeze({ bossId: 'quarryColossus', name: 'Ore Shell Crack', maxGauge: 180, duration: 4.6, damageTakenScale: 1.22, reward: Object.freeze({ materials: Object.freeze({ oreChunks: 2 }) }) }),
    Object.freeze({ bossId: 'emberjawGolem', name: 'Overheat Break', maxGauge: 155, duration: 4.8, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 1 }) }) }),
    Object.freeze({ bossId: 'rimewarden', name: 'Frostplate Break', maxGauge: 175, duration: 4.5, damageTakenScale: 1.19, reward: Object.freeze({ materials: Object.freeze({ refinementCore: 1 }) }) }),
    Object.freeze({ bossId: 'stormbreakRoc', name: 'Wing Stagger', maxGauge: 170, duration: 4.1, damageTakenScale: 1.18, reward: Object.freeze({ currency: 120 }) }),
    Object.freeze({ bossId: 'astralArchivist', name: 'Page Lockout', maxGauge: 185, duration: 4.3, damageTakenScale: 1.2, reward: Object.freeze({ materials: Object.freeze({ cubeFragment: 2 }) }) }),
    Object.freeze({ bossId: 'eclipseSovereign', name: 'Totality Fracture', maxGauge: 210, duration: 4.7, damageTakenScale: 1.24, reward: Object.freeze({ materials: Object.freeze({ cubeFragment: 3 }) }) })
  ]);

  const SKILL_MODIFIERS = Object.freeze([
    Object.freeze({ id: 'heavy_strike_guardbreaker', skillId: 'fighter_heavy_strike', name: 'Guardbreaker', summary: 'Heavy Strike builds more break gauge on bosses.', unlockLevel: 8, breakScale: 1.35, damageScale: 1.04 }),
    Object.freeze({ id: 'ground_slam_aftershock', skillId: 'fighter_ground_slam', name: 'Aftershock', summary: 'Ground Slam gains area damage against clustered targets.', unlockLevel: 14, damageScale: 1.06, extraLines: 1 }),
    Object.freeze({ id: 'magic_bolt_arc', skillId: 'mage_magic_bolt', name: 'Arc Primer', summary: 'Magic Bolt marks targets briefly for follow-up skills.', unlockLevel: 8, markDuration: 2.5, damageScale: 1.03 }),
    Object.freeze({ id: 'chain_bolt_conductor', skillId: 'storm_mage_chain_bolt', name: 'Conductor', summary: 'Chain Bolt adds one extra line and slightly cheaper casts.', unlockLevel: 25, extraLines: 1, resourceCostScale: 0.92 }),
    Object.freeze({ id: 'fireball_ember_core', skillId: 'fire_mage_fireball', name: 'Ember Core', summary: 'Fireball burns longer and hits broken bosses harder.', unlockLevel: 25, burnDuration: 2, brokenDamageScale: 1.12 }),
    Object.freeze({ id: 'rune_mark_anchor', skillId: 'rune_mage_rune_mark', name: 'Anchor Rune', summary: 'Rune Mark lasts longer and improves link detonation.', unlockLevel: 25, markDuration: 2, runeDuration: 2 }),
    Object.freeze({ id: 'quick_shot_fletching', skillId: 'archer_quick_shot', name: 'Fine Fletching', summary: 'Quick Shot gets smoother single-target uptime.', unlockLevel: 8, cooldownScale: 0.94, damageScale: 1.02 }),
    Object.freeze({ id: 'piercing_arrow_broadhead', skillId: 'archer_piercing_arrow', name: 'Broadhead', summary: 'Piercing Arrow builds break and weak-point pressure.', unlockLevel: 14, breakScale: 1.22, weakPointDuration: 2 }),
    Object.freeze({ id: 'sniper_aimed_shot_deadeye', skillId: 'sniper_aimed_shot', name: 'Deadeye Window', summary: 'Aimed Shot deals more damage to marked or broken targets.', unlockLevel: 25, markedDamageScale: 1.12, brokenDamageScale: 1.12 }),
    Object.freeze({ id: 'trapper_snare_relay', skillId: 'trapper_snare_trap', name: 'Snare Relay', summary: 'Snare Trap arms faster and slows elites longer.', unlockLevel: 25, cooldownScale: 0.94, slowDuration: 2 }),
    Object.freeze({ id: 'guardian_bash_shatter', skillId: 'guardian_shield_bash', name: 'Shatter Bash', summary: 'Shield Bash has a stronger boss break role.', unlockLevel: 25, breakScale: 1.45 }),
    Object.freeze({ id: 'beast_companion_hunt', skillId: 'beast_archer_companion_strike', name: 'Hunt Signal', summary: 'Companion Strike advances target-farm streaks faster.', unlockLevel: 25, targetFarmBonus: 0.12, damageScale: 1.04 })
  ]);

  const GEAR_TRAITS = Object.freeze([
    Object.freeze({ id: 'bossbreaker', name: 'Bossbreaker', summary: 'Boosts boss and break-window damage.', slots: Object.freeze(['weapon', 'offhand']), rarity: 'Rare', statBonuses: Object.freeze({ armorBreak: 4, attackDamagePercent: 2 }) }),
    Object.freeze({ id: 'rift_hunter', name: 'Rift Hunter', summary: 'Improves Endless Rift score and elite rewards.', slots: Object.freeze(['weapon', 'gloves', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ power: 3, crit: 1 }) }),
    Object.freeze({ id: 'field_sweeper', name: 'Field Sweeper', summary: 'Improves mobbing and target-farm drops.', slots: Object.freeze(['weapon', 'chest', 'ring']), rarity: 'Uncommon', statBonuses: Object.freeze({ areaDamage: 3, resourceGain: 1 }) }),
    Object.freeze({ id: 'warded', name: 'Warded', summary: 'Adds survival stats for dungeon objectives.', slots: Object.freeze(['chest', 'head', 'boots']), rarity: 'Uncommon', statBonuses: Object.freeze({ hp: 35, defense: 2 }) }),
    Object.freeze({ id: 'quickstep', name: 'Quickstep', summary: 'Improves movement uptime and party spread commands.', slots: Object.freeze(['boots']), rarity: 'Uncommon', statBonuses: Object.freeze({ speed: 8, avoid: 1 }) }),
    Object.freeze({ id: 'focus_lens', name: 'Focus Lens', summary: 'Improves marks, weak points, and precision skills.', slots: Object.freeze(['offhand', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ crit: 2, critDamage: 5, weakPointDuration: 1 }) }),
    Object.freeze({ id: 'resource_loop', name: 'Resource Loop', summary: 'Raises MP and secondary-resource uptime.', slots: Object.freeze(['offhand', 'chest', 'ring']), rarity: 'Rare', statBonuses: Object.freeze({ mpMax: 28, resourceGainPercent: 3 }) }),
    Object.freeze({ id: 'party_standard', name: 'Party Standard', summary: 'Improves AI ally buffs and party objective safety.', slots: Object.freeze(['head', 'chest']), rarity: 'Epic', statBonuses: Object.freeze({ hp: 40, power: 2, defense: 2 }) })
  ]);

  const CARD_DEFINITIONS = Object.freeze([
    Object.freeze({ id: 'gel_spark', name: 'Gel Spark', icon: 'GS', rarity: 'Common', tags: Object.freeze(['Ooze', 'Starter']), summary: 'Small HP and resource-flow card for early grinding.', baseStats: Object.freeze({ hp: 24, resourceGain: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'mossguard_oath', name: 'Mossguard Oath', icon: 'MO', rarity: 'Common', tags: Object.freeze(['Guard', 'Forest']), summary: 'Steady defense for safer platform routes.', baseStats: Object.freeze({ hp: 18, defense: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'thorn_focus', name: 'Thorn Focus', icon: 'TF', rarity: 'Common', tags: Object.freeze(['Focus', 'Forest']), summary: 'Low-rarity power with a light mobbing bonus.', baseStats: Object.freeze({ power: 2, areaDamage: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'bristle_charge', name: 'Bristle Charge', icon: 'BC', rarity: 'Common', tags: Object.freeze(['Beast', 'Mobility']), summary: 'Movement and power for aggressive farming lanes.', baseStats: Object.freeze({ speed: 8, power: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'clockwork_patience', name: 'Clockwork Patience', icon: 'CP', rarity: 'Common', tags: Object.freeze(['Construct', 'Guard']), summary: 'A defensive card for learning boss timings.', baseStats: Object.freeze({ defense: 2, block: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'ember_glint', name: 'Ember Glint', icon: 'EG', rarity: 'Common', tags: Object.freeze(['Cinder', 'Burn']), summary: 'Entry burn support with a flat power bump.', baseStats: Object.freeze({ burnDamage: 2, power: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'hunter_tempo', name: 'Hunter Tempo', icon: 'HT', rarity: 'Common', tags: Object.freeze(['Bandit', 'Crit']), summary: 'Crit and movement for active combat routes.', baseStats: Object.freeze({ crit: 1, speed: 6 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'frost_thread', name: 'Frost Thread', icon: 'FT', rarity: 'Common', tags: Object.freeze(['Frost', 'Guard']), summary: 'Early percent HP with enough defense to matter.', baseStats: Object.freeze({ maxHpPercent: 1, defense: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'vinebinder_loop', name: 'Vinebinder Loop', icon: 'VL', rarity: 'Uncommon', tags: Object.freeze(['Forest', 'Mobbing']), summary: 'Area pressure with smoother resource returns.', baseStats: Object.freeze({ areaDamage: 3, resourceGain: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'rustcoil_lens', name: 'Rustcoil Lens', icon: 'RL', rarity: 'Uncommon', tags: Object.freeze(['Construct', 'Break']), summary: 'Break-focused card for armored monsters.', baseStats: Object.freeze({ armorBreak: 3, defensePercent: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'ashflare_core', name: 'Ashflare Core', icon: 'AC', rarity: 'Uncommon', tags: Object.freeze(['Cinder', 'Burn']), summary: 'Burn damage with a light percent power bonus.', baseStats: Object.freeze({ burnDamage: 5, powerPercent: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'storm_fletching', name: 'Storm Fletching', icon: 'SF', rarity: 'Uncommon', tags: Object.freeze(['Storm', 'Ranged']), summary: 'Range and crit for safer ranged grinding.', baseStats: Object.freeze({ range: 12, crit: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'cloudcall_vellum', name: 'Cloudcall Vellum', icon: 'CV', rarity: 'Uncommon', tags: Object.freeze(['Support', 'Resource']), summary: 'Resource uptime for skill-heavy rotations.', baseStats: Object.freeze({ mpMax: 20, resourceGainPercent: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'astral_index', name: 'Astral Index', icon: 'AI', rarity: 'Rare', tags: Object.freeze(['Astral', 'Crit']), summary: 'Precision card for builds that scale crit damage.', baseStats: Object.freeze({ critDamage: 8, power: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'bramble_heart', name: 'Bramble Heart', icon: 'BH', rarity: 'Rare', tags: Object.freeze(['Boss', 'Guard']), summary: 'Boss-farm survival card from deep forest routes.', baseStats: Object.freeze({ hp: 60, maxHpPercent: 2, defense: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'titan_gearheart', name: 'Titan Gearheart', icon: 'TG', rarity: 'Rare', tags: Object.freeze(['Boss', 'Break']), summary: 'Break and block package for construct bosses.', baseStats: Object.freeze({ armorBreak: 5, defensePercent: 2, block: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'mimic_cache', name: 'Mimic Cache', icon: 'MC', rarity: 'Rare', tags: Object.freeze(['Treasure', 'Hybrid']), summary: 'Flexible rare card with crit and resource gain.', baseStats: Object.freeze({ crit: 3, resourceGainPercent: 3 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'rift_splinter', name: 'Rift Splinter', icon: 'RS', rarity: 'Epic', tags: Object.freeze(['Rift', 'Damage']), summary: 'High-end damage card for rift and boss pushes.', baseStats: Object.freeze({ attackDamagePercent: 3, critDamage: 10 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'stormbreak_plume', name: 'Stormbreak Plume', icon: 'SP', rarity: 'Epic', tags: Object.freeze(['Storm', 'Mobility']), summary: 'Fast ranged card for evasive farming routes.', baseStats: Object.freeze({ speed: 16, crit: 4, range: 18 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'archivist_star', name: 'Archivist Star', icon: 'AS', rarity: 'Relic', tags: Object.freeze(['Astral', 'Relic']), summary: 'Relic resource and power card from late-route bosses.', baseStats: Object.freeze({ mpMax: 45, resourceGainPercent: 5, powerPercent: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'eclipse_corona', name: 'Eclipse Corona', icon: 'EC', rarity: 'Relic', tags: Object.freeze(['Eclipse', 'Relic']), summary: 'Relic damage card for the strongest single-target loadouts.', baseStats: Object.freeze({ attackDamagePercent: 4, powerPercent: 2, critDamage: 12 }), rankScale: 0.35, maxRank: 5 })
  ]);

  const CARD_ASSETS = Object.freeze(CARD_DEFINITIONS.reduce((assets, card) => {
    assets[card.id] = `${CARD_ICON_ROOT}/${card.id}.png`;
    return assets;
  }, {}));

  const CLASS_MASTERY_TRACKS = Object.freeze(Object.keys(CLASS_FILE_IDS).map((classId) => Object.freeze({
    classId,
    name: `${classId.replace(/([A-Z])/g, ' $1').replace(/^./, (letter) => letter.toUpperCase())} Mastery`,
    milestones: Object.freeze([
      Object.freeze({ level: 1, xp: 0, statBonuses: Object.freeze({}) }),
      Object.freeze({ level: 2, xp: 2500, statBonuses: Object.freeze({ power: 1 }) }),
      Object.freeze({ level: 3, xp: 8500, statBonuses: Object.freeze({ hp: 30, resourceGain: 1 }) }),
      Object.freeze({ level: 4, xp: 18000, statBonuses: Object.freeze({ attackDamagePercent: 1, defense: 1 }) }),
      Object.freeze({ level: 5, xp: 36000, statBonuses: Object.freeze({ power: 2, crit: 1 }) })
    ])
  })));

  const DUNGEON_OBJECTIVES = Object.freeze([
    Object.freeze({ id: 'break_boss', name: 'Break the Boss', summary: 'Trigger at least one boss break window before the clear.', type: 'bossBreak', goal: 1, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 2 }) }) }),
    Object.freeze({ id: 'elite_control', name: 'Elite Control', summary: 'Defeat an elite or affixed enemy during the run.', type: 'defeatElite', goal: 1, reward: Object.freeze({ currency: 80 }) }),
    Object.freeze({ id: 'party_survival', name: 'Keep the Party Up', summary: 'Clear with no visible AI ally defeated during the run.', type: 'partySurvival', goal: 1, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 1 }) }) }),
    Object.freeze({ id: 'clear_adds', name: 'Clean Sweep', summary: 'Defeat enough dungeon adds before the boss falls.', type: 'defeatDungeonEnemy', goal: 8, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 3 }) }) }),
    Object.freeze({ id: 'swift_clear', name: 'Swift Clear', summary: 'Clear the dungeon inside the bonus timer.', type: 'timedClear', goal: 420, reward: Object.freeze({ currency: 150 }) })
  ]);

  const ROSTER_SYNERGIES = Object.freeze([
    Object.freeze({ id: 'frontline_pair', name: 'Frontline Pair', summary: 'Guardian plus Berserker roster traits improve durable offense.', requiredTraitIds: Object.freeze(['guardian_bulwark', 'berserker_fervor']), statBonuses: Object.freeze({ hp: 35, power: 2 }) }),
    Object.freeze({ id: 'arcane_weather', name: 'Arcane Weather', summary: 'Fire, Rune, and Storm memories improve spell area damage.', requiredTraitIds: Object.freeze(['fire_mage_kindling', 'rune_mage_pattern', 'storm_mage_charge']), statBonuses: Object.freeze({ areaDamage: 4, mpMax: 25 }) }),
    Object.freeze({ id: 'hunter_net', name: 'Hunter Net', summary: 'Sniper, Trapper, and Beast Archer memories improve target farming.', requiredTraitIds: Object.freeze(['sniper_focus', 'trapper_routes', 'beast_archer_bond']), statBonuses: Object.freeze({ crit: 2, range: 12 }) }),
    Object.freeze({ id: 'dungeon_vanguard', name: 'Dungeon Vanguard', summary: 'Dungeon Veteran plus Vaultbreaker improves boss dungeons.', requiredTraitIds: Object.freeze(['dungeon_veteran', 'vaultbreaker']), statBonuses: Object.freeze({ defense: 2, armorBreak: 3 }) })
  ]);

  const PARTY_COMMANDS = Object.freeze([
    Object.freeze({ id: 'balanced', name: 'Balanced', summary: 'Allies split targets naturally and use normal assists.', powerScale: 0, defenseScale: 0 }),
    Object.freeze({ id: 'focus_boss', name: 'Focus Boss', summary: 'Allies prefer bosses, marks, and break windows.', preferBoss: true, powerScale: 0.04 }),
    Object.freeze({ id: 'spread_clear', name: 'Spread Clear', summary: 'Allies split across mobs more aggressively for field farming.', splitTargets: true, areaScale: 0.04 }),
    Object.freeze({ id: 'guard_player', name: 'Guard Player', summary: 'Allies favor defensive assists and party-survival objectives.', defenseScale: 0.05, shieldScale: 0.08 }),
    Object.freeze({ id: 'burst_window', name: 'Burst Window', summary: 'Allies save pressure for marked and broken targets.', preferMarked: true, powerScale: 0.06, cooldownScale: 1.08 })
  ]);

  const TARGET_FARM_TABLES = Object.freeze([
    Object.freeze({ enemyId: 'crackedMimic', name: 'Mimic Cache', summary: 'Higher chance for coupons, Echo Prisms, and rare gear from Mimics.' }),
    Object.freeze({ enemyId: 'emberjawGolem', name: 'Emberjaw Set Hunt', summary: 'Higher boss-set pressure while farming Emberjaw.' }),
    Object.freeze({ enemyId: 'riftAberration', name: 'Rift Splinter Hunt', summary: 'Improves rift material and elite reward streaks.' }),
    Object.freeze({ enemyId: 'rimewarden', name: 'Rimewarden Sigil Hunt', summary: 'Improves frost boss material and Relic chase odds.' }),
    Object.freeze({ enemyId: 'stormbreakRoc', name: 'Stormbreak Plume Hunt', summary: 'Improves airborne boss material and currency drops.' })
  ]);

  const ADVANCED_FEATURE_GUIDE = Object.freeze([
    Object.freeze({ id: 'status_icons', title: 'Status Icons', summary: 'Buff icons show effects active right now with warm duration bars; cooldown icons show unavailable skills with dark overlays and ready timers.', detail: 'Use this convention for future combat procedures: buffs answer what is affecting the player, cooldowns answer what can be pressed next. Rune Mage aura buffs and glyph cooldowns are intentionally separate.', panelId: 'guide' }),
    Object.freeze({ id: 'map_modifiers', title: 'Map Modifiers', summary: 'Fields, dungeons, and rifts can roll local rules that alter enemy pressure and rewards.', panelId: 'worldmap' }),
    Object.freeze({ id: 'boss_break', title: 'Boss Break Windows', summary: 'Bosses build a break gauge from control, armor-break, and focused attacks. Broken bosses take extra damage.', panelId: 'monsters' }),
    Object.freeze({ id: 'skill_modifiers', title: 'Skill Modifiers', summary: 'Skills unlock automatic modifiers as you level and invest in the class kit.', panelId: 'skills' }),
    Object.freeze({ id: 'class_mastery', title: 'Class Mastery', summary: 'Defeating enemies and using skills earns class mastery XP for small permanent class bonuses.', panelId: 'character' }),
    Object.freeze({ id: 'gear_traits', title: 'Gear Traits', summary: 'Rare and better drops can roll traits that change stat priorities and reward loops.', panelId: 'equipment' }),
    Object.freeze({ id: 'card_decks', title: 'Card Decks', summary: 'Cards stack by same card and tier, live in the Cards inventory tab, and slot into a six-card deck from Equipment.', detail: 'Equipment > Cards shows the active deck, each equipped card, and the full Active Deck Bonuses total. Equip one stack per card definition at a time; duplicate tiers of the same card do not stack in the deck. Right-click a stack in Inventory > Cards and choose Combine to turn three unlocked copies of the same card and tier into one card of the next tier. Upgrade All Cards combines every eligible stack as far as your current copies allow, including locked cards.', panelId: 'equipment' }),
    Object.freeze({ id: 'monster_research', title: 'Monster Research', summary: 'Monster Guide kills reveal stats, drops, and mastery bonuses while feeding target farming.', panelId: 'monsters' }),
    Object.freeze({ id: 'target_farming', title: 'Target Farming', summary: 'Repeatedly defeating the selected guide monster improves its drop pressure for a short streak.', panelId: 'monsters' }),
    Object.freeze({ id: 'roster_synergy', title: 'Roster Synergy', summary: 'Specific unlocked roster-trait combinations add extra account-style stat bonuses.', panelId: 'beta' }),
    Object.freeze({ id: 'dungeon_objectives', title: 'Dungeon Objectives', summary: 'Dungeons track bonus objectives such as boss breaks, add clears, and party survival.', panelId: 'quests' }),
    Object.freeze({ id: 'elite_affixes', title: 'Elite Affixes', summary: 'Rare elites can spawn with combat affixes that change pressure and reward priority.', panelId: 'monsters' }),
    Object.freeze({ id: 'party_commands', title: 'Party Commands', summary: 'Party commands steer visible AI allies toward boss focus, spread clears, guarding, or burst windows.', panelId: 'partyPanel' }),
    Object.freeze({ id: 'rift_ladder', title: 'Endless Rift Ladder', summary: 'Endless Rift tracks tier, mutations, and score as a repeatable performance ladder.', panelId: 'worldmap' }),
    Object.freeze({ id: 'starfall_plinko', title: 'Starfall Plinko', summary: 'Mob-earned balls and optional coin-bought balls feed a town Plinko board for coins, materials, coupons, prisms, gear, and 100-ball jackpot pity.', detail: 'Find the Plinko Host service NPC in any regional town. Monster drops are the efficient path; buying balls spends coins for a negative-expected-value sink with clear 100-ball jackpot pity progress.', panelId: 'plinko' })
  ]);

  const ASSET_BACKUP_ROOT = `${ASSET_ROOT}/backups/procedural`;
  const AI_DERIVED_MAP_BACKUP_KEYS = Object.freeze([
    'frostfenOutskirts',
    'glacierSpine',
    'rimewardenSanctum',
    'bramblekingCourt',
    'titanFoundry',
    'deepcoreCore',
    'emberjawFurnace',
    'rimewardenVault',
    'stormbreakAerie',
    'astralStacks',
    'eclipseThrone'
  ]);

  function assetSourcePath(assetPath) {
    return String(assetPath || '').split('#')[0].trim();
  }

  function proceduralBackupPath(assetPath) {
    const sourcePath = assetSourcePath(assetPath);
    if (!sourcePath || sourcePath.indexOf(`${ASSET_ROOT}/`) !== 0) return '';
    return `${ASSET_BACKUP_ROOT}/${sourcePath.slice(`${ASSET_ROOT}/`.length)}`;
  }

  function addAssetBackupPath(paths, assetPath) {
    const sourcePath = assetSourcePath(assetPath);
    const backupPath = proceduralBackupPath(sourcePath);
    if (sourcePath && backupPath) paths[sourcePath] = backupPath;
  }

  function collectAnimationBackupPaths(paths, value) {
    if (!value || typeof value !== 'object') return;
    Object.keys(value).forEach((key) => {
      const child = value[key];
      if ((key === 'sheet' || key === 'path') && typeof child === 'string') {
        addAssetBackupPath(paths, child);
        return;
      }
      collectAnimationBackupPaths(paths, child);
    });
  }

  const ASSET_BACKUP_PATHS = Object.freeze((() => {
    const paths = {};
    addAssetBackupPath(paths, GENERIC_PLAYER_ASSET);
    collectAnimationBackupPaths(paths, GENERIC_PLAYER_ANIMATION_ASSET);
    Object.values(MENU_ICON_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
    Object.values(CLASS_TRIAL_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
    AI_DERIVED_MAP_BACKUP_KEYS.forEach((key) => addAssetBackupPath(paths, MAP_ASSETS[key]));
    Object.values(ITEM_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
    Object.values(CARD_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
    collectAnimationBackupPaths(paths, ENVIRONMENT_STRUCTURE_ASSETS);
    collectAnimationBackupPaths(paths, FX_ANIMATION_ASSETS);
    collectAnimationBackupPaths(paths, BASIC_ATTACK_FX_ANIMATION_ASSETS);
    collectAnimationBackupPaths(paths, ENEMY_COMBAT_FX_ANIMATION_ASSETS);
    collectAnimationBackupPaths(paths, ENEMY_PROJECTILE_ANIMATION_ASSETS);
    collectAnimationBackupPaths(paths, SKILL_FX_ANIMATION_ASSETS);
    collectAnimationBackupPaths(paths, PORTAL_ANIMATION_ASSETS);
    return paths;
  })());

  const DATA = Object.freeze({
    SAVE_KEY,
    CHARACTER_ROSTER_KEY,
    CHARACTER_SLOT_COUNT,
    ASSET_ROOT,
    ASSET_BACKUP_ROOT,
    ASSET_BACKUP_PATHS,
    LEVEL_CAP,
    SPECIALIZATION_LEVEL,
    ROSTER_TRAIT_SLOTS,
    BASE_SKILL_ICON_ROOT,
    ADVANCED_SKILL_ICON_ROOT,
    CARD_ICON_ROOT,
    GENERIC_PLAYER_ASSET,
    CHARACTER_SLOT_PEDESTAL_ASSET,
    CLASS_FILE_IDS,
    CLASS_ASSETS,
    CHARACTER_LOOKS,
    ENEMY_ASSETS,
    MAP_ASSETS,
    UI_ASSETS,
    MENU_ICON_ASSETS,
    CLASS_TRIAL_ASSETS,
    WORLD_MAP_ATLAS,
    MAP_LAYOUT_ROLES,
    MAP_LAYOUT_ROLE_LABELS,
    MAP_LAYOUT_BLUEPRINTS,
    STATION_ASSETS,
    ENVIRONMENT_ASSETS,
    ENVIRONMENT_STRUCTURE_ASSETS,
    ENVIRONMENT_READABILITY_DEFAULTS,
    ENVIRONMENT_TERRAIN_STYLE_DEFAULTS,
    ENVIRONMENT_TERRAIN_CELLS,
    ENVIRONMENT_PROP_CELLS,
    ENVIRONMENT_STRUCTURE_CELLS,
    ENVIRONMENT_REAR_PROP_KINDS,
    ENVIRONMENT_FRONT_PROP_KINDS,
    ENVIRONMENT_UPPER_FRONT_PROP_KINDS,
    MAP_ENVIRONMENT_PROFILES,
    MAP_TOWN_SCENES,
    MAP_FIELD_COMPOSITIONS,
    ITEM_ASSETS,
    CARD_ASSETS,
    ITEM_RARITY_VISUALS,
    BASE_SKILL_ICONS,
    ADVANCED_SKILL_ICONS,
    CLASS_ICON_ASSETS,
    ANIMATION_ROOT,
    COMBAT_FX_ANIMATION_ROOT,
    EQUIPMENT_VISUAL_ROOT,
    ANIMATION_ASSETS,
    PLAYER_ANIMATION_ROWS,
    ENEMY_ANIMATION_ROWS,
    PET_ANIMATION_ROWS,
    SKILL_FX_ANIMATION_ROWS,
    BASIC_ATTACK_FX_ANIMATION_ROWS,
    ENEMY_COMBAT_FX_ANIMATION_ROWS,
    GENERIC_PLAYER_ANIMATION_ASSET,
    PLAYER_ANIMATION_ASSETS,
    EQUIPMENT_VISUALS,
    PLAYER_RIGS,
    ENEMY_ANIMATION_ROW_HOLDS,
    ENEMY_ANIMATION_TIMING_OVERRIDES,
    ENEMY_ANIMATION_FILE_IDS,
    ENEMY_ANIMATION_ASSETS,
    ENEMY_ANIMATION_BEHAVIORS,
    PET_ANIMATION_ASSET,
    ENEMY_PROJECTILE_ANIMATION_ASSETS,
    FX_ANIMATION_ASSETS,
    SKILL_FX_ANIMATION_ASSETS,
    BASIC_ATTACK_FX_ANIMATION_ASSETS,
    ENEMY_COMBAT_FX_ANIMATION_ASSETS,
    PORTAL_ANIMATION_ASSETS,
    BUFF_CAST_VISUALS,
    SKILL_PURPOSES,
    SKILL_VISUALS,
    SKILL_VISUAL_IDS,
    ROLE_TAGS,
    CLASS_ROLE_PROFILES,
    BASE_CLASSES,
    ADVANCED_CLASSES,
    SKILLS,
    ENEMIES,
    WORLD_AREAS,
    WORLD_ROUTES,
    WORLD_MAP_NODES,
    WORLD_MAP_EDGES,
    MAPS,
    BOSS_ENCOUNTERS,
    EQUIPMENT_SLOTS,
    EQUIPMENT_SLOT_META,
    SHOP_ITEMS,
    RANDOM_EQUIPMENT_ITEMS,
    BOSS_EQUIPMENT_SOURCES,
    DROP_ECONOMY,
    PLINKO_BOUNCE_COUNT,
    PLINKO_ACTIVE_DROP_LIMIT,
    PLINKO_PITY_TARGET,
    PLINKO_SLOT_PROBABILITIES,
    PLINKO_SLOT_DENOMINATOR,
    PLINKO_SLOT_PROBABILITY_TABLES,
    PLINKO_BALLS,
    PLINKO_BOARD_SLOTS,
    PLINKO_BOARDS,
    PLINKO_REWARD_TABLES,
    EQUIPMENT_SETS,
    BOSS_EQUIPMENT_ITEMS,
    MATERIAL_ITEMS,
    CONSUMABLE_ITEMS,
    STAT_UPGRADE_DEFINITIONS,
    QUESTS,
    CLASS_TRIALS,
    DUNGEONS,
    ROSTER_TRAITS,
    SPECIALIZATIONS,
    MARKET_LISTINGS,
    COSMETICS,
    CASH_SHOP_CATEGORIES,
    CASH_SHOP_ITEMS,
    SEASONS,
    PARTY_AI_LOADOUTS,
    PROTOTYPE_PARTY_MEMBERS,
    ONBOARDING_STEPS,
    AUDIO_CUES,
    ACCOMPLISHMENTS,
    STARTER_ITEMS,
	    STARTER_CONSUMABLES,
	    UPGRADE_OUTCOMES,
	    UPGRADE_DUST_COST_BY_RANGE,
	    UPGRADE_AIDES,
	    POTENTIAL_TIERS,
	    POTENTIAL_LINE_POOLS,
	    MUTATIONS,
	    MAP_MODIFIERS,
	    ELITE_AFFIXES,
	    BOSS_BREAK_PROFILES,
	    SKILL_MODIFIERS,
	    GEAR_TRAITS,
	    CARD_DEFINITIONS,
	    CLASS_MASTERY_TRACKS,
	    DUNGEON_OBJECTIVES,
	    ROSTER_SYNERGIES,
	    PARTY_COMMANDS,
	    TARGET_FARM_TABLES,
	    ADVANCED_FEATURE_GUIDE
	  });

  global.ProjectStarfallData = DATA;

  if (typeof module === 'object' && module.exports) {
    module.exports = DATA;
  }
})(typeof window !== 'undefined' ? window : globalThis);
