(function initProjectStarfallData(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataIndex = (typeof require === 'function' ? require('./data/index.js') : null) || DataModules.index || {};
  const DATA = DataIndex.DATA || DataIndex;

  global.ProjectStarfallData = DATA;

  if (typeof module === 'object' && module.exports) {
    module.exports = DATA;
  }
})(typeof window !== 'undefined' ? window : globalThis);
