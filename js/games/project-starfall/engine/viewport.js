(function initProjectStarfallEngineViewport(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreSettings = (typeof require === 'function' ? require('../core/settings.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeHudScaleSetting = CoreSettings.normalizeHudScale || function normalizeHudScaleSettingFallback(value, fallback) {
    const scale = Number(value);
    if (!Number.isFinite(scale)) return Number(fallback);
    return Math.max(0.9, Math.min(1.2, Math.round(scale * 10) / 10));
  };
  const normalizeDamageNumberDensitySetting = CoreSettings.normalizeDamageNumberDensity || function normalizeDamageNumberDensitySettingFallback(value, fallback) {
    const id = String(value || '').trim();
    return ['normal', 'reduced', 'minimal'].includes(id) ? id : fallback;
  };
  const normalizeFrameRateLimitSetting = CoreSettings.normalizeFrameRateLimit || function normalizeFrameRateLimitSettingFallback(value, options, fallback) {
    const source = Array.isArray(options) ? options : [];
    const rate = Math.round(Number(value));
    return source.includes(rate) ? rate : fallback;
  };

  const PLAYFIELD_WIDTH = 1280;
  const PLAYFIELD_HEIGHT = 640;
  const SOLID_PLATFORM_HEIGHT = 48;
  const CANVAS_STATUS_HUD_HEIGHT = 84;
  const HUD_TOP = PLAYFIELD_HEIGHT + SOLID_PLATFORM_HEIGHT;
  const WORLD_HEIGHT = HUD_TOP;
  const LOGICAL_VIEW_WIDTH = 1280;
  const LOGICAL_VIEW_HEIGHT = 806;
  const VIEW_WIDTH = LOGICAL_VIEW_WIDTH;
  const VIEW_HEIGHT = LOGICAL_VIEW_HEIGHT;
  const VIEWPORT_WIDTH_MIN = 1024;
  const VIEWPORT_WIDTH_MAX = 1920;
  const VIEWPORT_HEIGHT_MIN = 704;
  const VIEWPORT_HEIGHT_MAX = 1080;
  const DEFAULT_WORLD_ZOOM = 1.32;
  const VIEWPORT_PRESETS = Object.freeze([
    Object.freeze({ id: 'compact', label: 'Compact', width: 1024, height: 704 }),
    Object.freeze({ id: 'standard', label: 'Standard', width: 1280, height: 806 }),
    Object.freeze({ id: 'large', label: 'Large', width: 1600, height: 930 }),
    Object.freeze({ id: 'ultra', label: 'Ultra', width: 1920, height: 1080 })
  ]);
  const FRAME_RATE_LIMIT_OPTIONS = Object.freeze([60, 120, 240, 0]);
  const DEFAULT_FRAME_RATE_LIMIT = 60;
  const DEFAULT_USER_SETTINGS = Object.freeze({
    video: Object.freeze({ viewportPreset: 'standard', width: 1280, height: 806, hudScale: 1, frameRateLimit: DEFAULT_FRAME_RATE_LIMIT }),
    audio: Object.freeze({ sfxEnabled: false, sfxVolume: 0.42, musicEnabled: false, musicVolume: 0.25 }),
    accessibility: Object.freeze({ reducedEffects: false, damageNumbers: 'normal' })
  });

  function getViewportPreset(id) {
    const presetId = normalizeId(id) || 'standard';
    return VIEWPORT_PRESETS.find((preset) => preset.id === presetId) || null;
  }

  function normalizeHudScale(value) {
    return normalizeHudScaleSetting(value, DEFAULT_USER_SETTINGS.video.hudScale);
  }

  function normalizeFrameRateLimit(value) {
    return normalizeFrameRateLimitSetting(value, FRAME_RATE_LIMIT_OPTIONS, DEFAULT_USER_SETTINGS.video.frameRateLimit);
  }

  function normalizeDamageNumberDensity(value) {
    return normalizeDamageNumberDensitySetting(value, DEFAULT_USER_SETTINGS.accessibility.damageNumbers);
  }

  function createUserSettings(value) {
    const source = value && typeof value === 'object' ? value : {};
    const sourceVideo = source.video && typeof source.video === 'object' ? source.video : {};
    const sourceAudio = source.audio && typeof source.audio === 'object' ? source.audio : {};
    const sourceAccessibility = source.accessibility && typeof source.accessibility === 'object' ? source.accessibility : {};
    const preset = getViewportPreset(sourceVideo.viewportPreset);
    const custom = !preset && normalizeId(sourceVideo.viewportPreset) === 'custom';
    const base = preset || DEFAULT_USER_SETTINGS.video;
    const width = clamp(Math.round(Number(sourceVideo.width || base.width) || base.width), VIEWPORT_WIDTH_MIN, VIEWPORT_WIDTH_MAX);
    const height = clamp(Math.round(Number(sourceVideo.height || base.height) || base.height), VIEWPORT_HEIGHT_MIN, VIEWPORT_HEIGHT_MAX);
    return {
      video: {
        viewportPreset: custom ? 'custom' : preset ? preset.id : DEFAULT_USER_SETTINGS.video.viewportPreset,
        width,
        height,
        hudScale: normalizeHudScale(sourceVideo.hudScale),
        frameRateLimit: normalizeFrameRateLimit(sourceVideo.frameRateLimit)
      },
      audio: {
        sfxEnabled: !!sourceAudio.sfxEnabled,
        sfxVolume: clamp(Number(sourceAudio.sfxVolume == null ? DEFAULT_USER_SETTINGS.audio.sfxVolume : sourceAudio.sfxVolume), 0, 1),
        musicEnabled: !!sourceAudio.musicEnabled,
        musicVolume: clamp(Number(sourceAudio.musicVolume == null ? DEFAULT_USER_SETTINGS.audio.musicVolume : sourceAudio.musicVolume), 0, 1)
      },
      accessibility: {
        reducedEffects: !!sourceAccessibility.reducedEffects,
        damageNumbers: normalizeDamageNumberDensity(sourceAccessibility.damageNumbers)
      }
    };
  }

  function createViewportMetrics(settings) {
    const normalized = createUserSettings(settings);
    const displayWidth = normalized.video.width;
    const displayHeight = normalized.video.height;
    const width = LOGICAL_VIEW_WIDTH;
    const height = LOGICAL_VIEW_HEIGHT;
    const solidPlatformHeight = SOLID_PLATFORM_HEIGHT;
    const statusHudHeight = CANVAS_STATUS_HUD_HEIGHT;
    const playfieldHeight = Math.max(520, height - solidPlatformHeight - statusHudHeight);
    return {
      width,
      height,
      logicalWidth: width,
      logicalHeight: height,
      displayWidth,
      displayHeight,
      displayScale: Math.min(displayWidth / width, displayHeight / height),
      hudScale: normalized.video.hudScale,
      playfieldHeight,
      solidPlatformHeight,
      statusHudHeight,
      hudTop: playfieldHeight + solidPlatformHeight,
      worldHeight: playfieldHeight + solidPlatformHeight
    };
  }

  function getWorldViewWidth(width, zoom) {
    const screenWidth = Math.max(1, Number(width || VIEW_WIDTH));
    return screenWidth / Math.max(1, Number(zoom || DEFAULT_WORLD_ZOOM));
  }

  function getWorldViewHeight(height, zoom) {
    const screenHeight = Math.max(1, Number(height || PLAYFIELD_HEIGHT));
    return screenHeight / Math.max(1, Number(zoom || DEFAULT_WORLD_ZOOM));
  }

  function getTargetFrameIntervalMs(limit) {
    const normalized = normalizeFrameRateLimit(limit);
    return normalized > 0 ? 1000 / normalized : 0;
  }

  const api = {
    PLAYFIELD_WIDTH,
    PLAYFIELD_HEIGHT,
    SOLID_PLATFORM_HEIGHT,
    CANVAS_STATUS_HUD_HEIGHT,
    HUD_TOP,
    WORLD_HEIGHT,
    LOGICAL_VIEW_WIDTH,
    LOGICAL_VIEW_HEIGHT,
    VIEW_WIDTH,
    VIEW_HEIGHT,
    VIEWPORT_WIDTH_MIN,
    VIEWPORT_WIDTH_MAX,
    VIEWPORT_HEIGHT_MIN,
    VIEWPORT_HEIGHT_MAX,
    DEFAULT_WORLD_ZOOM,
    VIEWPORT_PRESETS,
    FRAME_RATE_LIMIT_OPTIONS,
    DEFAULT_FRAME_RATE_LIMIT,
    DEFAULT_USER_SETTINGS,
    getViewportPreset,
    normalizeHudScale,
    normalizeFrameRateLimit,
    normalizeDamageNumberDensity,
    createUserSettings,
    createViewportMetrics,
    getWorldViewWidth,
    getWorldViewHeight,
    getTargetFrameIntervalMs
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.viewport = Object.assign({}, modules.viewport || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
