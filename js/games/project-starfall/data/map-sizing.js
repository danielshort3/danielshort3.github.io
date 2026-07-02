(function initProjectStarfallDataMapSizing(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataWorld = (typeof require === 'function' ? require('./world.js') : null) || DataModules.world || {};
  const DataMapGeometry = (typeof require === 'function' ? require('./map-geometry.js') : null) || DataModules.mapGeometry || {};

  function defaultGetPlatformDefRight(platform) {
    if (Array.isArray(platform)) return Number(platform[0] || 0) + Number(platform[2] || 0);
    return Number(platform && platform.x || 0) + Number(platform && platform.w || 0);
  }

  const SHOP_INTERIOR_WORLD_WIDTH = DataWorld.SHOP_INTERIOR_WORLD_WIDTH || 1280;
  const getPlatformDefRight = DataMapGeometry.getPlatformDefRight || defaultGetPlatformDefRight;

  function getAuthoredMapWidth(map) {
    const source = map || {};
    const platformWidth = (source.platforms || []).reduce((width, platform) => Math.max(width, getPlatformDefRight(platform)), 0);
    const pointWidth = []
      .concat(source.spawnPoints || [])
      .concat(source.stations || [])
      .concat(source.questNpcs || [])
      .reduce((width, point) => Math.max(width, Number(point && point.x || 0) + 240), 0);
    if (source.shopInterior) {
      const compactWidth = Number(source.compactWorldWidth || SHOP_INTERIOR_WORLD_WIDTH);
      return Math.max(compactWidth, platformWidth, pointWidth);
    }
    return Math.max(3600, platformWidth, pointWidth);
  }

  const api = Object.freeze({
    getAuthoredMapWidth
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapSizing = Object.assign({}, modules.mapSizing || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
