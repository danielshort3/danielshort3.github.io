(function initProjectStarfallDataIndex(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataEnvironmentContent = (typeof require === 'function' ? require('./environment-content.js') : null) || DataModules.environmentContent || {};
  const DataItems = (typeof require === 'function' ? require('./items.js') : null) || DataModules.items || {};
  const DataConsumables = (typeof require === 'function' ? require('./consumables.js') : null) || DataModules.consumables || {};
  const DataPlinko = (typeof require === 'function' ? require('./plinko.js') : null) || DataModules.plinko || {};
  const DataEquipmentMeta = (typeof require === 'function' ? require('./equipment-meta.js') : null) || DataModules.equipmentMeta || {};
  const DataCards = (typeof require === 'function' ? require('./cards.js') : null) || DataModules.cards || {};
  const DataMonsterDrops = (typeof require === 'function' ? require('./monster-drops.js') : null) || DataModules.monsterDrops || {};
  const DataProgressionContent = (typeof require === 'function' ? require('./progression-content.js') : null) || DataModules.progressionContent || {};
  const DataGuides = (typeof require === 'function' ? require('./guides.js') : null) || DataModules.guides || {};
  const DataActorCombat = (typeof require === 'function' ? require('./actor-combat.js') : null) || DataModules.actorCombat || {};
  const DataAssetBackupContent = (typeof require === 'function' ? require('./asset-backup-content.js') : null) || DataModules.assetBackupContent || {};
  const DataMapRules = (typeof require === 'function' ? require('./map-rules.js') : null) || DataModules.mapRules || {};
  const DataCombatModifiers = (typeof require === 'function' ? require('./combat-modifiers.js') : null) || DataModules.combatModifiers || {};
  const DataUpgrades = (typeof require === 'function' ? require('./upgrades.js') : null) || DataModules.upgrades || {};
  const DataAccomplishments = (typeof require === 'function' ? require('./accomplishments.js') : null) || DataModules.accomplishments || {};
  const DataDungeons = (typeof require === 'function' ? require('./dungeons.js') : null) || DataModules.dungeons || {};
  const DataBossEncounters = (typeof require === 'function' ? require('./boss-encounters.js') : null) || DataModules.bossEncounters || {};
  const DataBossMechanics = (typeof require === 'function' ? require('./boss-mechanics.js') : null) || DataModules.bossMechanics || {};
  const DataEquipmentSets = (typeof require === 'function' ? require('./equipment-sets.js') : null) || DataModules.equipmentSets || {};
  const DataEquipmentEconomy = (typeof require === 'function' ? require('./equipment-economy.js') : null) || DataModules.equipmentEconomy || {};
  const DataCommerce = (typeof require === 'function' ? require('./commerce.js') : null) || DataModules.commerce || {};
  const DataMapContent = (typeof require === 'function' ? require('./map-content.js') : null) || DataModules.mapContent || {};
  const DataStatUpgrades = (typeof require === 'function' ? require('./stat-upgrades.js') : null) || DataModules.statUpgrades || {};
  const DataRewardContent = (typeof require === 'function' ? require('./reward-content.js') : null) || DataModules.rewardContent || {};
  const DataParty = (typeof require === 'function' ? require('./party.js') : null) || DataModules.party || {};
  const DataOnboarding = (typeof require === 'function' ? require('./onboarding.js') : null) || DataModules.onboarding || {};
  const DataSpecializations = (typeof require === 'function' ? require('./specializations.js') : null) || DataModules.specializations || {};
  const DataClassSkillDesign = (typeof require === 'function' ? require('./class-skill-design.js') : null) || DataModules.classSkillDesign || {};
  const ROLE_TAGS = DataAssets.ROLE_TAGS;
  const SAVE_KEY = DataAssets.SAVE_KEY;
  const CHARACTER_ROSTER_KEY = DataAssets.CHARACTER_ROSTER_KEY;
  const CHARACTER_SLOT_COUNT = DataAssets.CHARACTER_SLOT_COUNT;
  const ASSET_ROOT = DataAssets.ASSET_ROOT;
  const BASE_SKILL_ICON_ROOT = DataAssets.BASE_SKILL_ICON_ROOT;
  const ADVANCED_SKILL_ICON_ROOT = DataAssets.ADVANCED_SKILL_ICON_ROOT;
  const CARD_ICON_ROOT = DataAssets.CARD_ICON_ROOT;
  const GENERIC_PLAYER_ASSET = DataAssets.GENERIC_PLAYER_ASSET;
  const EQUIPMENT_ATLAS_ROOT = DataAssets.EQUIPMENT_ATLAS_ROOT;
  const CHARACTER_SLOT_PEDESTAL_ASSET = DataAssets.CHARACTER_SLOT_PEDESTAL_ASSET;
  const LEVEL_CAP = DataAssets.LEVEL_CAP;
  const SPECIALIZATION_LEVEL = DataAssets.SPECIALIZATION_LEVEL;
  const ROSTER_TRAIT_SLOTS = DataAssets.ROSTER_TRAIT_SLOTS;
  const CLASS_FILE_IDS = DataAssets.CLASS_FILE_IDS;
  const CLASS_ASSETS = DataAssets.CLASS_ASSETS;
  const CHARACTER_LOOKS = DataAssets.CHARACTER_LOOKS;
  const ENEMY_ASSETS = DataAssets.ENEMY_ASSETS;
  const MAP_ASSETS = DataAssets.MAP_ASSETS;
  const UI_ASSETS = DataAssets.UI_ASSETS;
  const MENU_ICON_ASSETS = DataAssets.MENU_ICON_ASSETS;
  const CLASS_TRIAL_ASSETS = DataAssets.CLASS_TRIAL_ASSETS;
  const WORLD_MAP_ATLAS = DataAssets.WORLD_MAP_ATLAS;
  const STATION_ASSETS = DataAssets.STATION_ASSETS;

  const environmentContentData = DataEnvironmentContent.createEnvironmentContentData({
    assetData: DataAssets
  });
  const ENVIRONMENT_ASSETS = environmentContentData.ENVIRONMENT_ASSETS;
  const ENVIRONMENT_STRUCTURE_ASSETS = environmentContentData.ENVIRONMENT_STRUCTURE_ASSETS;
  const ENVIRONMENT_TERRAIN_CELLS = environmentContentData.ENVIRONMENT_TERRAIN_CELLS;
  const ENVIRONMENT_PROP_CELLS = environmentContentData.ENVIRONMENT_PROP_CELLS;
  const ENVIRONMENT_STRUCTURE_CELLS = environmentContentData.ENVIRONMENT_STRUCTURE_CELLS;
  const ENVIRONMENT_REAR_PROP_KINDS = environmentContentData.ENVIRONMENT_REAR_PROP_KINDS;
  const ENVIRONMENT_FRONT_PROP_KINDS = environmentContentData.ENVIRONMENT_FRONT_PROP_KINDS;
  const ENVIRONMENT_UPPER_FRONT_PROP_KINDS = environmentContentData.ENVIRONMENT_UPPER_FRONT_PROP_KINDS;
  const ENVIRONMENT_READABILITY_DEFAULTS = environmentContentData.ENVIRONMENT_READABILITY_DEFAULTS;
  const ENVIRONMENT_TERRAIN_STYLE_DEFAULTS = environmentContentData.ENVIRONMENT_TERRAIN_STYLE_DEFAULTS;
  const MAP_ENVIRONMENT_PROFILES = environmentContentData.MAP_ENVIRONMENT_PROFILES;

  const ITEM_ASSETS = DataItems.ITEM_ASSETS;
  const ITEM_SHEET_BACKUP_ASSETS = DataItems.ITEM_SHEET_BACKUP_ASSETS;
  const ITEM_RARITY_VISUALS = DataItems.ITEM_RARITY_VISUALS;
  const MATERIAL_ITEMS = DataItems.MATERIAL_ITEMS;

  const actorCombatData = DataActorCombat.createActorCombatData({
    ASSET_ROOT,
    EQUIPMENT_ATLAS_ROOT,
    CLASS_FILE_IDS,
    CLASS_ASSETS,
    ENEMY_ASSETS
  });
  const BASE_SKILL_ICONS = actorCombatData.BASE_SKILL_ICONS;
  const ADVANCED_SKILL_ICONS = actorCombatData.ADVANCED_SKILL_ICONS;
  const CLASS_ICON_ASSETS = actorCombatData.CLASS_ICON_ASSETS;
  const ANIMATION_ROOT = actorCombatData.ANIMATION_ROOT;
  const COMBAT_FX_ANIMATION_ROOT = actorCombatData.COMBAT_FX_ANIMATION_ROOT;
  const PLAYER_ANIMATION_ROWS = actorCombatData.PLAYER_ANIMATION_ROWS;
  const ENEMY_ANIMATION_ROWS = actorCombatData.ENEMY_ANIMATION_ROWS;
  const PET_ANIMATION_ROWS = actorCombatData.PET_ANIMATION_ROWS;
  const SKILL_FX_ANIMATION_ROWS = actorCombatData.SKILL_FX_ANIMATION_ROWS;
  const BASIC_ATTACK_FX_ANIMATION_ROWS = actorCombatData.BASIC_ATTACK_FX_ANIMATION_ROWS;
  const ENEMY_COMBAT_FX_ANIMATION_ROWS = actorCombatData.ENEMY_COMBAT_FX_ANIMATION_ROWS;
  const GENERIC_PLAYER_ANIMATION_ASSET = actorCombatData.GENERIC_PLAYER_ANIMATION_ASSET;
  const PLAYER_ANIMATION_ASSETS = actorCombatData.PLAYER_ANIMATION_ASSETS;
  const EQUIPMENT_VISUALS = actorCombatData.EQUIPMENT_VISUALS;
  const PLAYER_RIGS = actorCombatData.PLAYER_RIGS;
  const ENEMY_ANIMATION_ROW_HOLDS = actorCombatData.ENEMY_ANIMATION_ROW_HOLDS;
  const ENEMY_ANIMATION_TIMING_OVERRIDES = actorCombatData.ENEMY_ANIMATION_TIMING_OVERRIDES;
  const ENEMY_ANIMATION_FILE_IDS = actorCombatData.ENEMY_ANIMATION_FILE_IDS;
  const ENEMY_ANIMATION_ASSETS = actorCombatData.ENEMY_ANIMATION_ASSETS;
  const ENEMY_PROJECTILE_ANIMATION_ASSETS = actorCombatData.ENEMY_PROJECTILE_ANIMATION_ASSETS;
  const ENEMY_ANIMATION_BEHAVIORS = actorCombatData.ENEMY_ANIMATION_BEHAVIORS;
  const FX_ANIMATION_ASSETS = actorCombatData.FX_ANIMATION_ASSETS;
  const PORTAL_ANIMATION_ASSETS = actorCombatData.PORTAL_ANIMATION_ASSETS;
  const PET_ANIMATION_ASSET = actorCombatData.PET_ANIMATION_ASSET;
  const BUFF_CAST_VISUALS = actorCombatData.BUFF_CAST_VISUALS;
  const CLASS_ROLE_PROFILES = actorCombatData.CLASS_ROLE_PROFILES;
  const BASE_CLASSES = actorCombatData.BASE_CLASSES;
  const ADVANCED_CLASSES = actorCombatData.ADVANCED_CLASSES;
  const SKILL_PURPOSES = actorCombatData.SKILL_PURPOSES;
  const SKILL_VISUALS = actorCombatData.SKILL_VISUALS;
  const SKILL_VISUAL_IDS = actorCombatData.SKILL_VISUAL_IDS;
  const SKILLS = actorCombatData.SKILLS;
  const CLASS_SKILL_GUIDE_CONTRACT = DataClassSkillDesign.CLASS_SKILL_GUIDE_CONTRACT;
  const CLASS_RUNTIME_REQUIRED_FIELDS = DataClassSkillDesign.CLASS_RUNTIME_REQUIRED_FIELDS;
  const ADVANCED_CLASS_REQUIRED_FIELDS = DataClassSkillDesign.ADVANCED_CLASS_REQUIRED_FIELDS;
  const SKILL_RUNTIME_REQUIRED_FIELDS = DataClassSkillDesign.SKILL_RUNTIME_REQUIRED_FIELDS;
  const SKILL_DESIGN_OPTIONAL_FIELDS = DataClassSkillDesign.SKILL_DESIGN_OPTIONAL_FIELDS;
  const CLASS_RESOURCE_DEFINITIONS = DataClassSkillDesign.CLASS_RESOURCE_DEFINITIONS;
  const STATUS_EFFECT_DEFINITIONS = DataClassSkillDesign.STATUS_EFFECT_DEFINITIONS;
  const CLASS_SKILL_TOOLTIP_FORMAT = DataClassSkillDesign.CLASS_SKILL_TOOLTIP_FORMAT;
  const CLASS_SKILL_BALANCE_TUNING_FIELDS = DataClassSkillDesign.CLASS_SKILL_BALANCE_TUNING_FIELDS;
  const CLASS_SKILL_LOADOUT_SLOTS = DataClassSkillDesign.CLASS_SKILL_LOADOUT_SLOTS;
  const CLASS_SKILL_ENCOUNTER_TEST_CASES = DataClassSkillDesign.CLASS_SKILL_ENCOUNTER_TEST_CASES;
  const CLASS_SKILL_DEBUG_SCENARIOS = DataClassSkillDesign.CLASS_SKILL_DEBUG_SCENARIOS;
  const MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS = actorCombatData.MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS;
  const ENEMIES = actorCombatData.ENEMIES;
  const SKILL_FX_ANIMATION_ASSETS = actorCombatData.SKILL_FX_ANIMATION_ASSETS;
  const BASIC_ATTACK_FX_ANIMATION_ASSETS = actorCombatData.BASIC_ATTACK_FX_ANIMATION_ASSETS;
  const ENEMY_COMBAT_FX_ANIMATION_ASSETS = actorCombatData.ENEMY_COMBAT_FX_ANIMATION_ASSETS;
  const ANIMATION_ASSETS = actorCombatData.ANIMATION_ASSETS;

  const mapContentData = DataMapContent.createMapContentData({
    MAP_ASSETS,
    STATION_ASSETS,
    MAP_ENVIRONMENT_PROFILES
  });
  const WORLD_AREAS = mapContentData.WORLD_AREAS;
  const WORLD_ROUTES = mapContentData.WORLD_ROUTES;
  const REGIONAL_TOWN_IDS = mapContentData.REGIONAL_TOWN_IDS;
  const SHOP_INTERIOR_WORLD_WIDTH = mapContentData.SHOP_INTERIOR_WORLD_WIDTH;
  const WORLD_MAP_NODES = mapContentData.WORLD_MAP_NODES;
  const WORLD_MAP_EDGES = mapContentData.WORLD_MAP_EDGES;
  const SHOP_VENDOR_TYPES = mapContentData.SHOP_VENDOR_TYPES;
  const getTownShopVendorId = mapContentData.getTownShopVendorId;
  const MAP_LAYOUT_ROLES = mapContentData.MAP_LAYOUT_ROLES;
  const MAP_LAYOUT_ROLE_LABELS = mapContentData.MAP_LAYOUT_ROLE_LABELS;
  const MAP_LAYOUT_BLUEPRINTS = mapContentData.MAP_LAYOUT_BLUEPRINTS;
  const MAP_TOWN_SCENES = mapContentData.MAP_TOWN_SCENES;
  const MAP_FIELD_COMPOSITIONS = mapContentData.MAP_FIELD_COMPOSITIONS;
  const MAP_DESIGN_INTENTS = mapContentData.MAP_DESIGN_INTENTS;
  const MAP_MECHANIC_DEFINITIONS = mapContentData.MAP_MECHANIC_DEFINITIONS;
  const TOWN_SERVICE_PLANS = mapContentData.TOWN_SERVICE_PLANS;
  const MAP_PORTAL_FICTION = mapContentData.MAP_PORTAL_FICTION;

  const STAR_CARD_MATERIAL_IDS = DataMonsterDrops.STAR_CARD_MATERIAL_IDS;
  const STAR_CARD_DROP_TABLES = DataMonsterDrops.STAR_CARD_DROP_TABLES;

  const MAPS = mapContentData.MAPS;

  const BOSS_ENCOUNTERS = DataBossEncounters.BOSS_ENCOUNTERS;

  const BOSS_SPATIAL_MECHANICS = DataBossMechanics.BOSS_SPATIAL_MECHANICS;

  const EQUIPMENT_SLOTS = DataEquipmentMeta.EQUIPMENT_SLOTS;
  const EQUIPMENT_SLOT_META = DataEquipmentMeta.EQUIPMENT_SLOT_META;

  const equipmentEconomyData = DataEquipmentEconomy.createEquipmentEconomyData({
    ITEM_ASSETS,
    EQUIPMENT_VISUALS
  });
  const SHOP_ITEMS = equipmentEconomyData.SHOP_ITEMS;
  const RANDOM_EQUIPMENT_ITEMS = equipmentEconomyData.RANDOM_EQUIPMENT_ITEMS;

  const EQUIPMENT_SETS = DataEquipmentSets.EQUIPMENT_SETS;

  const BOSS_EQUIPMENT_SOURCES = equipmentEconomyData.BOSS_EQUIPMENT_SOURCES;
  const DROP_ECONOMY = equipmentEconomyData.DROP_ECONOMY;
  const BOSS_EQUIPMENT_ITEMS = equipmentEconomyData.BOSS_EQUIPMENT_ITEMS;

  const RATE_COUPON_ITEMS = DataConsumables.RATE_COUPON_ITEMS;

  const PLINKO_BOUNCE_COUNT = DataPlinko.PLINKO_BOUNCE_COUNT;
  const PLINKO_ACTIVE_DROP_LIMIT = DataPlinko.PLINKO_ACTIVE_DROP_LIMIT;
  const PLINKO_PITY_TARGET = DataPlinko.PLINKO_PITY_TARGET;
  const PLINKO_SLOT_PROBABILITIES = DataPlinko.PLINKO_SLOT_PROBABILITIES;
  const PLINKO_SLOT_DENOMINATOR = DataPlinko.PLINKO_SLOT_DENOMINATOR;
  const PLINKO_SLOT_PROBABILITY_TABLES = DataPlinko.PLINKO_SLOT_PROBABILITY_TABLES;
  const PLINKO_BOARD_SLOTS = DataPlinko.PLINKO_BOARD_SLOTS;

  const PLINKO_BALLS = DataConsumables.PLINKO_BALLS;

  const PLINKO_BOARDS = DataPlinko.PLINKO_BOARDS;
  const PLINKO_REWARD_TABLES = DataPlinko.PLINKO_REWARD_TABLES;

  const CONSUMABLE_ITEMS = DataConsumables.CONSUMABLE_ITEMS;

  const STAT_UPGRADE_DEFINITIONS = DataStatUpgrades.STAT_UPGRADE_DEFINITIONS;

  const rewardContentData = DataRewardContent.createRewardContentData({
    getTownShopVendorId
  });
  const QUESTS = rewardContentData.QUESTS;

  const DUNGEONS = DataDungeons.DUNGEONS;

  const ROSTER_TRAITS = DataSpecializations.ROSTER_TRAITS;
  const CLASS_TRIALS = DataSpecializations.CLASS_TRIALS;
  const SPECIALIZATIONS = DataSpecializations.SPECIALIZATIONS;

  const MARKET_LISTINGS = DataCommerce.MARKET_LISTINGS;

  const SHOP_VENDOR_CATALOGS = rewardContentData.SHOP_VENDOR_CATALOGS;

  const COSMETICS = DataCommerce.COSMETICS;
  const CASH_SHOP_CATEGORIES = DataCommerce.CASH_SHOP_CATEGORIES;
  const CASH_SHOP_ITEMS = DataCommerce.CASH_SHOP_ITEMS;
  const SEASONS = DataCommerce.SEASONS;

  const DAILY_LOGIN_REWARDS = rewardContentData.DAILY_LOGIN_REWARDS;
  const DAILY_LOGIN_MILESTONES = rewardContentData.DAILY_LOGIN_MILESTONES;

  const PARTY_AI_LOADOUTS = DataParty.PARTY_AI_LOADOUTS;
  const PROTOTYPE_PARTY_MEMBERS = DataParty.PROTOTYPE_PARTY_MEMBERS;

  const ONBOARDING_STEPS = DataOnboarding.ONBOARDING_STEPS;
  const AUDIO_CUES = DataOnboarding.AUDIO_CUES;

  const ACCOMPLISHMENTS = DataAccomplishments.ACCOMPLISHMENTS;

  const STARTER_ITEMS = DataOnboarding.STARTER_ITEMS;
  const STARTER_CONSUMABLES = DataOnboarding.STARTER_CONSUMABLES;

  const UPGRADE_OUTCOMES = DataUpgrades.UPGRADE_OUTCOMES;
  const UPGRADE_DUST_COST_BY_RANGE = DataUpgrades.UPGRADE_DUST_COST_BY_RANGE;
  const UPGRADE_AIDES = DataUpgrades.UPGRADE_AIDES;
  const POTENTIAL_TIERS = DataUpgrades.POTENTIAL_TIERS;
  const POTENTIAL_LINE_POOLS = DataUpgrades.POTENTIAL_LINE_POOLS;

  const MUTATIONS = DataMapRules.MUTATIONS;
  const MAP_MODIFIERS = DataMapRules.MAP_MODIFIERS;
  const ELITE_AFFIXES = DataMapRules.ELITE_AFFIXES;

  const BOSS_BREAK_PROFILES = DataCombatModifiers.BOSS_BREAK_PROFILES;
  const SKILL_MODIFIERS = DataCombatModifiers.SKILL_MODIFIERS;
  const GEAR_TRAITS = DataCombatModifiers.GEAR_TRAITS;

  const CARD_DEFINITIONS = DataCards.CARD_DEFINITIONS;
  const CARD_ASSETS = DataCards.CARD_ASSETS;

  const progressionContentData = DataProgressionContent.createProgressionContentData({
    assetData: DataAssets
  });
  const CLASS_MASTERY_TRACKS = progressionContentData.CLASS_MASTERY_TRACKS;
  const DUNGEON_OBJECTIVES = progressionContentData.DUNGEON_OBJECTIVES;
  const ROSTER_SYNERGIES = progressionContentData.ROSTER_SYNERGIES;
  const PARTY_COMMANDS = progressionContentData.PARTY_COMMANDS;

  const TARGET_FARM_TABLES = DataGuides.TARGET_FARM_TABLES;
  const ADVANCED_FEATURE_GUIDE = DataGuides.ADVANCED_FEATURE_GUIDE;

  const assetBackupContentData = DataAssetBackupContent.createAssetBackupContentData({
    assetData: DataAssets,
    itemData: DataItems,
    cardData: DataCards,
    environmentData: environmentContentData,
    actorCombatData
  });
  const ASSET_BACKUP_ROOT = assetBackupContentData.ASSET_BACKUP_ROOT;
  const ASSET_BACKUP_PATHS = assetBackupContentData.ASSET_BACKUP_PATHS;

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
    MAP_DESIGN_INTENTS,
    MAP_MECHANIC_DEFINITIONS,
    BOSS_SPATIAL_MECHANICS,
    TOWN_SERVICE_PLANS,
    MAP_PORTAL_FICTION,
    ITEM_ASSETS,
    CARD_ASSETS,
    ITEM_RARITY_VISUALS,
    BASE_SKILL_ICONS,
    ADVANCED_SKILL_ICONS,
    CLASS_ICON_ASSETS,
    ANIMATION_ROOT,
    COMBAT_FX_ANIMATION_ROOT,
    EQUIPMENT_ATLAS_ROOT,
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
    CLASS_SKILL_GUIDE_CONTRACT,
    CLASS_RUNTIME_REQUIRED_FIELDS,
    ADVANCED_CLASS_REQUIRED_FIELDS,
    SKILL_RUNTIME_REQUIRED_FIELDS,
    SKILL_DESIGN_OPTIONAL_FIELDS,
    CLASS_RESOURCE_DEFINITIONS,
    STATUS_EFFECT_DEFINITIONS,
    CLASS_SKILL_TOOLTIP_FORMAT,
    CLASS_SKILL_BALANCE_TUNING_FIELDS,
    CLASS_SKILL_LOADOUT_SLOTS,
    CLASS_SKILL_ENCOUNTER_TEST_CASES,
    CLASS_SKILL_DEBUG_SCENARIOS,
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
    REGIONAL_TOWN_IDS,
    SHOP_INTERIOR_WORLD_WIDTH,
    SHOP_VENDOR_TYPES,
    SHOP_VENDOR_CATALOGS,
    MAPS,
    BOSS_ENCOUNTERS,
    EQUIPMENT_SLOTS,
    EQUIPMENT_SLOT_META,
    SHOP_ITEMS,
    RANDOM_EQUIPMENT_ITEMS,
    BOSS_EQUIPMENT_SOURCES,
    MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS,
    DROP_ECONOMY,
    STAR_CARD_MATERIAL_IDS,
    STAR_CARD_DROP_TABLES,
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
    DAILY_LOGIN_REWARDS,
    DAILY_LOGIN_MILESTONES,
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

  const modules = global.ProjectStarfallDataModules || {};
  modules.index = Object.assign({}, modules.index || {}, {
    DATA
  });
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = DATA;
  }
})(typeof window !== 'undefined' ? window : globalThis);
