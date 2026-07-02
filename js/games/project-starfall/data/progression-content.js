(function initProjectStarfallDataProgressionContent(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataProgression = (typeof require === 'function' ? require('./progression.js') : null) || DataModules.progression || {};

  function createProgressionContentData(options) {
    const settings = options || {};
    const assetData = settings.assetData || DataAssets;
    const classFileIds = settings.classFileIds || assetData.CLASS_FILE_IDS;
    const createProgressionData = settings.createProgressionData || DataProgression.createProgressionData;

    return createProgressionData({
      classFileIds
    });
  }

  const defaultProgressionContentData = createProgressionContentData();
  const api = Object.assign({
    createProgressionContentData
  }, defaultProgressionContentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.progressionContent = Object.assign({}, modules.progressionContent || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
