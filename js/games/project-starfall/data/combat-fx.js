(function initProjectStarfallDataCombatFx(global) {
  'use strict';

  function toCombatFxFileId(value) {
    return String(value || '')
      .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
      .replace(/[_\s]+/g, '-')
      .replace(/[^a-zA-Z0-9-]+/g, '')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .toLowerCase();
  }

  function createCombatFxData(options) {
    const settings = options || {};
    const SKILLS = Array.isArray(settings.SKILLS) ? settings.SKILLS : [];
    const ENEMIES = Array.isArray(settings.ENEMIES) ? settings.ENEMIES : [];
    const CORE_ANIMATION_ASSETS = settings.CORE_ANIMATION_ASSETS || {};
    const ENEMY_PROJECTILE_ANIMATION_ASSETS = settings.ENEMY_PROJECTILE_ANIMATION_ASSETS || {};
    const ENEMY_COMBAT_FX_FILE_IDS = settings.ENEMY_COMBAT_FX_FILE_IDS || {};
    const SKILL_FX_ANIMATION_ROWS = settings.SKILL_FX_ANIMATION_ROWS || Object.freeze(['cast', 'projectile', 'impact', 'area']);
    const BASIC_ATTACK_FX_ANIMATION_ROWS = settings.BASIC_ATTACK_FX_ANIMATION_ROWS || Object.freeze(['cast', 'projectile', 'impact', 'trail']);
    const ENEMY_COMBAT_FX_ANIMATION_ROWS = settings.ENEMY_COMBAT_FX_ANIMATION_ROWS || Object.freeze(['telegraph', 'melee', 'projectile', 'buff', 'impact']);
    const makeCombatFxAnimationAsset = typeof settings.makeCombatFxAnimationAsset === 'function'
      ? settings.makeCombatFxAnimationAsset
      : () => null;

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
      assets[enemy.id] = makeCombatFxAnimationAsset(
        ENEMY_COMBAT_FX_FILE_IDS[enemy.id] || toCombatFxFileId(enemy.id),
        'enemies',
        ENEMY_COMBAT_FX_ANIMATION_ROWS
      );
      return assets;
    }, {}));

    const ANIMATION_ASSETS = Object.freeze(Object.assign({}, CORE_ANIMATION_ASSETS, {
      skillFx: SKILL_FX_ANIMATION_ASSETS,
      basicAttackFx: BASIC_ATTACK_FX_ANIMATION_ASSETS,
      enemyCombatFx: ENEMY_COMBAT_FX_ANIMATION_ASSETS,
      enemyProjectiles: ENEMY_PROJECTILE_ANIMATION_ASSETS
    }));

    return Object.freeze({
      ACTIVE_COMBAT_SKILLS,
      SKILL_FX_ANIMATION_ASSETS,
      BASIC_ATTACK_FX_ANIMATION_ASSETS,
      ENEMY_COMBAT_FX_ANIMATION_ASSETS,
      ANIMATION_ASSETS,
      toCombatFxFileId
    });
  }

  const api = Object.freeze({
    createCombatFxData,
    toCombatFxFileId
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.combatFx = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
