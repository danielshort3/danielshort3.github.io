(function initProjectStarfallDataAssetBackupContent(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataItems = (typeof require === 'function' ? require('./items.js') : null) || DataModules.items || {};
  const DataCards = (typeof require === 'function' ? require('./cards.js') : null) || DataModules.cards || {};
  const DataEnvironment = (typeof require === 'function' ? require('./environment.js') : null) || DataModules.environment || {};
  const DataActorCombat = (typeof require === 'function' ? require('./actor-combat.js') : null) || DataModules.actorCombat || {};
  const DataAssetBackups = (typeof require === 'function' ? require('./asset-backups.js') : null) || DataModules.assetBackups || {};

  function createAssetBackupContentData(options) {
    const settings = options || {};
    const assetData = settings.assetData || DataAssets;
    const itemData = settings.itemData || DataItems;
    const cardData = settings.cardData || DataCards;
    const environmentData = settings.environmentData || DataEnvironment;
    const actorCombatData = settings.actorCombatData || DataActorCombat;
    const createAssetBackupData = settings.createAssetBackupData || DataAssetBackups.createAssetBackupData;

    return createAssetBackupData({
      ASSET_ROOT: settings.ASSET_ROOT || assetData.ASSET_ROOT,
      GENERIC_PLAYER_ASSET: settings.GENERIC_PLAYER_ASSET || assetData.GENERIC_PLAYER_ASSET,
      GENERIC_PLAYER_ANIMATION_ASSET: settings.GENERIC_PLAYER_ANIMATION_ASSET || actorCombatData.GENERIC_PLAYER_ANIMATION_ASSET,
      MENU_ICON_ASSETS: settings.MENU_ICON_ASSETS || assetData.MENU_ICON_ASSETS,
      CLASS_TRIAL_ASSETS: settings.CLASS_TRIAL_ASSETS || assetData.CLASS_TRIAL_ASSETS,
      MAP_ASSETS: settings.MAP_ASSETS || assetData.MAP_ASSETS,
      ITEM_ASSETS: settings.ITEM_ASSETS || itemData.ITEM_ASSETS,
      ITEM_SHEET_BACKUP_ASSETS: settings.ITEM_SHEET_BACKUP_ASSETS || itemData.ITEM_SHEET_BACKUP_ASSETS,
      CARD_ASSETS: settings.CARD_ASSETS || cardData.CARD_ASSETS,
      ENVIRONMENT_STRUCTURE_ASSETS: settings.ENVIRONMENT_STRUCTURE_ASSETS || environmentData.ENVIRONMENT_STRUCTURE_ASSETS,
      FX_ANIMATION_ASSETS: settings.FX_ANIMATION_ASSETS || actorCombatData.FX_ANIMATION_ASSETS,
      BASIC_ATTACK_FX_ANIMATION_ASSETS: settings.BASIC_ATTACK_FX_ANIMATION_ASSETS || actorCombatData.BASIC_ATTACK_FX_ANIMATION_ASSETS,
      ENEMY_COMBAT_FX_ANIMATION_ASSETS: settings.ENEMY_COMBAT_FX_ANIMATION_ASSETS || actorCombatData.ENEMY_COMBAT_FX_ANIMATION_ASSETS,
      ENEMY_PROJECTILE_ANIMATION_ASSETS: settings.ENEMY_PROJECTILE_ANIMATION_ASSETS || actorCombatData.ENEMY_PROJECTILE_ANIMATION_ASSETS,
      SKILL_FX_ANIMATION_ASSETS: settings.SKILL_FX_ANIMATION_ASSETS || actorCombatData.SKILL_FX_ANIMATION_ASSETS,
      PORTAL_ANIMATION_ASSETS: settings.PORTAL_ANIMATION_ASSETS || actorCombatData.PORTAL_ANIMATION_ASSETS
    });
  }

  const defaultAssetBackupContentData = createAssetBackupContentData();
  const api = Object.assign({
    createAssetBackupContentData
  }, defaultAssetBackupContentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.assetBackupContent = Object.assign({}, modules.assetBackupContent || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
