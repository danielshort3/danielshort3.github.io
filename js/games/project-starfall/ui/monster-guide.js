(function initProjectStarfallUiMonsterGuide(global) {
  'use strict';

  function getMonsterGuideRegionAction(region, options) {
    const source = region || {};
    const settings = options || {};
    const keepsSearchFocus = source.type === 'monster-guide-search' || source.type === 'monster-guide-search-clear';
    const blurSearch = !!settings.searchFocused && !keepsSearchFocus;
    if (source.type === 'monster-guide-search') {
      return { handled: true, phase: 'search', type: 'focusSearch', blurSearch: false };
    }
    if (source.type === 'monster-guide-search-clear') {
      return { handled: true, phase: 'search', type: 'clearSearch', blurSearch: false };
    }
    if (source.type === 'monster-guide-filter') {
      return {
        handled: true,
        phase: 'search',
        type: 'setFilter',
        filterId: source.filterId || 'all',
        blurSearch
      };
    }
    if (source.type === 'monster-guide-select') {
      return { handled: true, phase: 'selection', type: 'selectEnemy', enemyId: source.enemyId, blurSearch };
    }
    return { handled: false, phase: '', type: '', blurSearch };
  }

  function createMonsterGuideUiHelpers() {
    return Object.freeze({
      getMonsterGuideRegionAction
    });
  }

  const api = {
    createMonsterGuideUiHelpers,
    getMonsterGuideRegionAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.monsterGuide = Object.assign({}, modules.monsterGuide || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
