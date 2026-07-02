(function initProjectStarfallCoreGeometry(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('./math.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  function platformContainsX(platform, x, padding) {
    const pad = Number(padding) || 0;
    return !!platform && x >= platform.x - pad && x <= platform.x + platform.w + pad;
  }

  function isSlopePlatform(platform) {
    return !!(platform && platform.shape === 'slope' && Number.isFinite(Number(platform.y2)));
  }

  function getPlatformSurfaceY(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!isSlopePlatform(platform)) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  }

  function getPlatformTopY(platform) {
    if (!platform) return 0;
    return isSlopePlatform(platform) ? Math.min(Number(platform.y || 0), Number(platform.y2 || platform.y || 0)) : Number(platform.y || 0);
  }

  function getPlatformBottomY(platform) {
    if (!platform) return 0;
    const surfaceBottom = isSlopePlatform(platform) ? Math.max(Number(platform.y || 0), Number(platform.y2 || platform.y || 0)) : Number(platform.y || 0);
    return surfaceBottom + Math.max(1, Number(platform.h || 0));
  }

  function isRectInBounds(rect, bounds, padding) {
    if (!rect || !bounds) return true;
    const pad = Number(padding || 0);
    return rect.x + rect.w >= Number(bounds.left || 0) - pad &&
      rect.x <= Number(bounds.right || 0) + pad &&
      rect.y + rect.h >= Number(bounds.top || 0) - pad &&
      rect.y <= Number(bounds.bottom || 0) + pad;
  }

  function isPointInBounds(point, bounds, padding) {
    if (!point || !bounds) return true;
    const pad = Number(padding || 0);
    return Number(point.x || 0) >= Number(bounds.left || 0) - pad &&
      Number(point.x || 0) <= Number(bounds.right || 0) + pad &&
      Number(point.y || 0) >= Number(bounds.top || 0) - pad &&
      Number(point.y || 0) <= Number(bounds.bottom || 0) + pad;
  }

  function pointInRect(point, rect) {
    return point && rect && point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h;
  }

  function translateCanvasRegion(region, dx, dy) {
    const translated = Object.assign({}, region);
    ['x', 'trackX'].forEach((key) => {
      if (Number.isFinite(Number(translated[key]))) translated[key] = Number(translated[key]) + dx;
    });
    ['y', 'trackY'].forEach((key) => {
      if (Number.isFinite(Number(translated[key]))) translated[key] = Number(translated[key]) + dy;
    });
    return translated;
  }

  const api = {
    platformContainsX,
    isSlopePlatform,
    getPlatformSurfaceY,
    getPlatformTopY,
    getPlatformBottomY,
    isRectInBounds,
    isPointInBounds,
    pointInRect,
    translateCanvasRegion
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
