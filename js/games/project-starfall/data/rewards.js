(function initProjectStarfallDataRewards(global) {
  'use strict';

  function freezeQuestReward(reward) {
    const source = reward || {};
    const frozen = Object.assign({}, source);
    ['materials', 'consumables', 'items', 'timedBuffs', 'permanentStats'].forEach((key) => {
      if (frozen[key] && typeof frozen[key] === 'object') frozen[key] = Object.freeze(frozen[key]);
    });
    return Object.freeze(frozen);
  }

  const api = Object.freeze({
    freezeQuestReward
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.rewards = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
