(function initProjectStarfallDataAssetBackups(global) {
  'use strict';

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

  function createAssetBackupData(options) {
    const settings = options || {};
    const ASSET_ROOT = settings.ASSET_ROOT || 'img/project-starfall';
    const ASSET_BACKUP_ROOT = `${ASSET_ROOT}/backups/procedural`;

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
      addAssetBackupPath(paths, settings.GENERIC_PLAYER_ASSET);
      collectAnimationBackupPaths(paths, settings.GENERIC_PLAYER_ANIMATION_ASSET);
      Object.values(settings.MENU_ICON_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
      Object.values(settings.CLASS_TRIAL_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
      AI_DERIVED_MAP_BACKUP_KEYS.forEach((key) => addAssetBackupPath(paths, settings.MAP_ASSETS && settings.MAP_ASSETS[key]));
      Object.values(settings.ITEM_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
      (settings.ITEM_SHEET_BACKUP_ASSETS || []).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
      Object.values(settings.CARD_ASSETS || {}).forEach((assetPath) => addAssetBackupPath(paths, assetPath));
      collectAnimationBackupPaths(paths, settings.ENVIRONMENT_STRUCTURE_ASSETS);
      collectAnimationBackupPaths(paths, settings.FX_ANIMATION_ASSETS);
      collectAnimationBackupPaths(paths, settings.BASIC_ATTACK_FX_ANIMATION_ASSETS);
      collectAnimationBackupPaths(paths, settings.ENEMY_COMBAT_FX_ANIMATION_ASSETS);
      collectAnimationBackupPaths(paths, settings.ENEMY_PROJECTILE_ANIMATION_ASSETS);
      collectAnimationBackupPaths(paths, settings.SKILL_FX_ANIMATION_ASSETS);
      collectAnimationBackupPaths(paths, settings.PORTAL_ANIMATION_ASSETS);
      return paths;
    })());

    return Object.freeze({
      ASSET_BACKUP_ROOT,
      ASSET_BACKUP_PATHS
    });
  }

  const api = Object.freeze({
    AI_DERIVED_MAP_BACKUP_KEYS,
    assetSourcePath,
    createAssetBackupData
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.assetBackups = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
