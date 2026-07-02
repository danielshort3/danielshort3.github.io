(function initProjectStarfallDataAssets(global) {
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
    daily: `${ASSET_ROOT}/ui/menu-icons/beta.png`,
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

  const api = {
    ROLE_TAGS,
    SAVE_KEY,
    CHARACTER_ROSTER_KEY,
    CHARACTER_SLOT_COUNT,
    ASSET_ROOT,
    BASE_SKILL_ICON_ROOT,
    ADVANCED_SKILL_ICON_ROOT,
    CARD_ICON_ROOT,
    GENERIC_PLAYER_ASSET,
    EQUIPMENT_VISUAL_ROOT,
    CHARACTER_SLOT_PEDESTAL_ASSET,
    LEVEL_CAP,
    SPECIALIZATION_LEVEL,
    ROSTER_TRAIT_SLOTS,
    CLASS_FILE_IDS,
    CLASS_ASSETS,
    CHARACTER_LOOKS,
    ENEMY_ASSETS,
    MAP_ASSETS,
    UI_ASSETS,
    MENU_ICON_ASSETS,
    CLASS_TRIAL_ASSETS,
    WORLD_MAP_ATLAS,
    STATION_ASSETS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.assets = Object.assign({}, modules.assets || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
