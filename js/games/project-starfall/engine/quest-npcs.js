(function initProjectStarfallEngineQuestNpcs(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreGeometry = (typeof require === 'function' ? require('../core/geometry.js') : null) || global.ProjectStarfallCore || {};
  const EngineModules = global.ProjectStarfallEngineModules || {};
  const EngineViewport = (typeof require === 'function' ? require('./viewport.js') : null) || EngineModules.viewport || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const isSlopePlatform = CoreGeometry.isSlopePlatform || function isSlopePlatformFallback(platform) {
    return !!(platform && platform.shape === 'slope' && Number.isFinite(Number(platform.y2)));
  };
  const getPlatformSurfaceY = CoreGeometry.getPlatformSurfaceY || function getPlatformSurfaceYFallback(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!isSlopePlatform(platform)) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  };

  function alignQuestNpc(rawNpc, index, mapId, platforms, options) {
    const settings = options || {};
    const npc = rawNpc || {};
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const platformIndex = clamp(Number(npc.platformIndex || 0), 0, Math.max(0, sourcePlatforms.length - 1));
    const platform = sourcePlatforms[platformIndex] || sourcePlatforms[0] || {
      x: 0,
      y: Number(settings.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640),
      w: 2200,
      h: 80,
      id: `${mapId}_platform_0`,
      index: 0
    };
    const w = Math.max(32, Number(npc.w || 38) || 38);
    const h = Math.max(56, Number(npc.h || 68) || 68);
    const x = clamp(Number(npc.x || platform.x + 140), platform.x + 18, platform.x + platform.w - w - 18);
    const surfaceY = getPlatformSurfaceY(platform, x + w / 2);
    return Object.assign({}, npc, {
      id: npc.id || `${mapId}_quest_npc_${index}`,
      name: npc.name || 'Quest NPC',
      questIds: Array.isArray(npc.questIds) ? npc.questIds.map(normalizeId).filter(Boolean) : [],
      x,
      y: surfaceY - h,
      w,
      h,
      platformIndex,
      platformId: platform.id,
      mapId
    });
  }

  function createMapHuntNpcDefinition(map) {
    const palette = Array.isArray(map && map.palette) ? map.palette : [];
    return {
      id: `${map && map.id || 'map'}_hunt_warden`,
      name: `${map && map.name || 'Map'} Warden`,
      x: 320,
      platformIndex: 0,
      questIds: [],
      color: palette[0] || '#5e7d9f',
      accent: palette[2] || palette[1] || '#ffd166'
    };
  }

  const api = {
    alignQuestNpc,
    createMapHuntNpcDefinition
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.questNpcs = Object.assign({}, modules.questNpcs || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
