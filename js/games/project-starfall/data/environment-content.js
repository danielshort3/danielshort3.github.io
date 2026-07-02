(function initProjectStarfallDataEnvironmentContent(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const DataEnvironment = (typeof require === 'function' ? require('./environment.js') : null) || DataModules.environment || {};

  function createEnvironmentContentData(options) {
    const settings = options || {};
    const assetData = settings.assetData || DataAssets;
    const ASSET_ROOT = settings.ASSET_ROOT || assetData.ASSET_ROOT;
    const createEnvironmentData = settings.createEnvironmentData || DataEnvironment.createEnvironmentData;

    return createEnvironmentData({
      ASSET_ROOT
    });
  }

  const defaultEnvironmentContentData = createEnvironmentContentData();
  const api = Object.assign({
    createEnvironmentContentData
  }, defaultEnvironmentContentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.environmentContent = Object.assign({}, modules.environmentContent || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
