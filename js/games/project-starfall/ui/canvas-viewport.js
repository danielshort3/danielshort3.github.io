(function initProjectStarfallUiCanvasViewport(global) {
  'use strict';

  const CANVAS_PLAYFIELD_HEIGHT = 640;
  const CANVAS_SOLID_PLATFORM_HEIGHT = 48;
  const CANVAS_STATUS_HUD_HEIGHT = 84;
  const CANVAS_STATUS_HUD_Y = CANVAS_PLAYFIELD_HEIGHT + CANVAS_SOLID_PLATFORM_HEIGHT;
  const CANVAS_VIEW_WIDTH = 1280;
  const CANVAS_VIEW_HEIGHT = CANVAS_STATUS_HUD_Y + CANVAS_STATUS_HUD_HEIGHT;

  const api = {
    CANVAS_PLAYFIELD_HEIGHT,
    CANVAS_SOLID_PLATFORM_HEIGHT,
    CANVAS_STATUS_HUD_HEIGHT,
    CANVAS_STATUS_HUD_Y,
    CANVAS_VIEW_WIDTH,
    CANVAS_VIEW_HEIGHT
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.canvasViewport = Object.assign({}, modules.canvasViewport || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
