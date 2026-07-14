(function initProjectStarfallDataActorCombat(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataEquipmentVisuals = (typeof require === 'function' ? require('./equipment-visuals.js') : null) || DataModules.equipmentVisuals || {};
  const DataAnimations = (typeof require === 'function' ? require('./animations.js') : null) || DataModules.animations || {};
  const DataClasses = (typeof require === 'function' ? require('./classes.js') : null) || DataModules.classes || {};
  const DataSkills = (typeof require === 'function' ? require('./skills.js') : null) || DataModules.skills || {};
  const DataMonsterDrops = (typeof require === 'function' ? require('./monster-drops.js') : null) || DataModules.monsterDrops || {};
  const DataGuides = (typeof require === 'function' ? require('./guides.js') : null) || DataModules.guides || {};
  const DataEnemies = (typeof require === 'function' ? require('./enemies.js') : null) || DataModules.enemies || {};
  const DataCombatFx = (typeof require === 'function' ? require('./combat-fx.js') : null) || DataModules.combatFx || {};

  function createActorCombatData(options) {
    const settings = options || {};
    const ASSET_ROOT = settings.ASSET_ROOT || DataAssets.ASSET_ROOT;
    const EQUIPMENT_ATLAS_ROOT = settings.EQUIPMENT_ATLAS_ROOT || DataAssets.EQUIPMENT_ATLAS_ROOT;
    const CLASS_FILE_IDS = settings.CLASS_FILE_IDS || DataAssets.CLASS_FILE_IDS;
    const CLASS_ASSETS = settings.CLASS_ASSETS || DataAssets.CLASS_ASSETS;
    const ENEMY_ASSETS = settings.ENEMY_ASSETS || DataAssets.ENEMY_ASSETS;
    const BASE_SKILL_ICONS = settings.BASE_SKILL_ICONS || DataSkills.BASE_SKILL_ICONS;
    const ADVANCED_SKILL_ICONS = settings.ADVANCED_SKILL_ICONS || DataSkills.ADVANCED_SKILL_ICONS;
    const CLASS_ICON_ASSETS = settings.CLASS_ICON_ASSETS || DataSkills.CLASS_ICON_ASSETS;
    const MONSTER_DROP_POOLS = settings.MONSTER_DROP_POOLS || DataMonsterDrops.MONSTER_DROP_POOLS;
    const monsterDropPool = settings.monsterDropPool || DataMonsterDrops.monsterDropPool;
    const MONSTER_GUIDE_FUTURE_ENEMY_IDS = settings.MONSTER_GUIDE_FUTURE_ENEMY_IDS || DataGuides.MONSTER_GUIDE_FUTURE_ENEMY_IDS;
    const MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS = settings.MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS || DataGuides.MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS;

    const animationData = DataAnimations.createAnimationData({
      ASSET_ROOT,
      EQUIPMENT_ATLAS_ROOT,
      CLASS_FILE_IDS,
      createEquipmentVisualData: settings.createEquipmentVisualData || DataEquipmentVisuals.createEquipmentVisualData
    });

    const classData = DataClasses.createClassData
      ? DataClasses.createClassData({
        classAssets: CLASS_ASSETS,
        playerAnimationAssets: animationData.PLAYER_ANIMATION_ASSETS
      })
      : {};

    const skillData = DataSkills.createSkillData
      ? DataSkills.createSkillData({
        baseSkillIcons: BASE_SKILL_ICONS,
        advancedSkillIcons: ADVANCED_SKILL_ICONS
      })
      : {};

    const enemyData = DataEnemies.createEnemyData({
      ENEMY_ASSETS,
      ENEMY_ANIMATION_ASSETS: animationData.ENEMY_ANIMATION_ASSETS,
      MONSTER_DROP_POOLS,
      createFallbackDropPool: () => monsterDropPool({ globalRareEligible: false, currencyWeight: 0 }),
      MONSTER_GUIDE_FUTURE_ENEMY_IDS,
      MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS,
      getEnemyAnimationBehavior: animationData.getEnemyAnimationBehavior
    });

    const combatFxData = DataCombatFx.createCombatFxData({
      SKILLS: skillData.SKILLS,
      ENEMIES: enemyData.ENEMIES,
      CORE_ANIMATION_ASSETS: animationData.CORE_ANIMATION_ASSETS,
      ENEMY_PROJECTILE_ANIMATION_ASSETS: animationData.ENEMY_PROJECTILE_ANIMATION_ASSETS,
      SKILL_FX_ANIMATION_ROWS: animationData.SKILL_FX_ANIMATION_ROWS,
      BASIC_ATTACK_FX_ANIMATION_ROWS: animationData.BASIC_ATTACK_FX_ANIMATION_ROWS,
      ENEMY_COMBAT_FX_ANIMATION_ROWS: animationData.ENEMY_COMBAT_FX_ANIMATION_ROWS,
      makeCombatFxAnimationAsset: animationData.makeCombatFxAnimationAsset
    });

    return Object.freeze({
      BASE_SKILL_ICONS,
      ADVANCED_SKILL_ICONS,
      CLASS_ICON_ASSETS,
      ANIMATION_ROOT: animationData.ANIMATION_ROOT,
      COMBAT_FX_ANIMATION_ROOT: animationData.COMBAT_FX_ANIMATION_ROOT,
      EQUIPMENT_ATLAS_ROOT: animationData.EQUIPMENT_ATLAS_ROOT,
      PLAYER_ANIMATION_ROWS: animationData.PLAYER_ANIMATION_ROWS,
      ENEMY_ANIMATION_ROWS: animationData.ENEMY_ANIMATION_ROWS,
      PET_ANIMATION_ROWS: animationData.PET_ANIMATION_ROWS,
      SKILL_FX_ANIMATION_ROWS: animationData.SKILL_FX_ANIMATION_ROWS,
      BASIC_ATTACK_FX_ANIMATION_ROWS: animationData.BASIC_ATTACK_FX_ANIMATION_ROWS,
      ENEMY_COMBAT_FX_ANIMATION_ROWS: animationData.ENEMY_COMBAT_FX_ANIMATION_ROWS,
      GENERIC_PLAYER_ANIMATION_ASSET: animationData.GENERIC_PLAYER_ANIMATION_ASSET,
      PLAYER_ANIMATION_ASSETS: animationData.PLAYER_ANIMATION_ASSETS,
      EQUIPMENT_VISUALS: animationData.EQUIPMENT_VISUALS,
      PLAYER_RIGS: animationData.PLAYER_RIGS,
      ENEMY_ANIMATION_ROW_HOLDS: animationData.ENEMY_ANIMATION_ROW_HOLDS,
      ENEMY_ANIMATION_TIMING_OVERRIDES: animationData.ENEMY_ANIMATION_TIMING_OVERRIDES,
      ENEMY_ANIMATION_FILE_IDS: animationData.ENEMY_ANIMATION_FILE_IDS,
      ENEMY_ANIMATION_ASSETS: animationData.ENEMY_ANIMATION_ASSETS,
      ENEMY_PROJECTILE_ANIMATION_ASSETS: animationData.ENEMY_PROJECTILE_ANIMATION_ASSETS,
      ENEMY_ANIMATION_BEHAVIORS: animationData.ENEMY_ANIMATION_BEHAVIORS,
      FX_ANIMATION_ASSETS: animationData.FX_ANIMATION_ASSETS,
      PORTAL_ANIMATION_ASSETS: animationData.PORTAL_ANIMATION_ASSETS,
      PET_ANIMATION_ASSET: animationData.PET_ANIMATION_ASSET,
      BUFF_CAST_VISUALS: animationData.BUFF_CAST_VISUALS,
      CORE_ANIMATION_ASSETS: animationData.CORE_ANIMATION_ASSETS,
      makeCombatFxAnimationAsset: animationData.makeCombatFxAnimationAsset,
      getEnemyAnimationBehavior: animationData.getEnemyAnimationBehavior,
      CLASS_ROLE_PROFILES: classData.CLASS_ROLE_PROFILES,
      BASE_CLASSES: classData.BASE_CLASSES,
      ADVANCED_CLASSES: classData.ADVANCED_CLASSES,
      SKILL_PURPOSES: skillData.SKILL_PURPOSES,
      SKILL_VISUALS: skillData.SKILL_VISUALS,
      SKILL_VISUAL_IDS: skillData.SKILL_VISUAL_IDS,
      SKILLS: skillData.SKILLS,
      MONSTER_GUIDE_FUTURE_ENEMY_IDS,
      MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS,
      ENEMIES: enemyData.ENEMIES,
      SKILL_FX_ANIMATION_ASSETS: combatFxData.SKILL_FX_ANIMATION_ASSETS,
      BASIC_ATTACK_FX_ANIMATION_ASSETS: combatFxData.BASIC_ATTACK_FX_ANIMATION_ASSETS,
      ENEMY_COMBAT_FX_ANIMATION_ASSETS: combatFxData.ENEMY_COMBAT_FX_ANIMATION_ASSETS,
      ANIMATION_ASSETS: combatFxData.ANIMATION_ASSETS
    });
  }

  const defaultActorCombatData = createActorCombatData();
  const api = Object.assign({
    createActorCombatData
  }, defaultActorCombatData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.actorCombat = Object.assign({}, modules.actorCombat || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
