(function initProjectStarfallUiSettings(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreSettings = (typeof require === 'function' ? require('../core/settings.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
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

  const SETTINGS_STORAGE_KEY = 'projectStarfallPrototypeSettings.v1';
  const VIEWPORT_PRESETS = Object.freeze([
    { id: 'compact', label: 'Compact', width: 1024, height: 704 },
    { id: 'standard', label: 'Standard', width: 1280, height: 806 },
    { id: 'large', label: 'Large', width: 1600, height: 930 },
    { id: 'ultra', label: 'Ultra', width: 1920, height: 1080 }
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

  function normalizeDamageNumberDensity(value) {
    return normalizeDamageNumberDensitySetting(value, DEFAULT_USER_SETTINGS.accessibility.damageNumbers);
  }

  function normalizeFrameRateLimit(value) {
    return normalizeFrameRateLimitSetting(value, FRAME_RATE_LIMIT_OPTIONS, DEFAULT_USER_SETTINGS.video.frameRateLimit);
  }

  function formatFrameRateLimitLabel(value) {
    const limit = normalizeFrameRateLimit(value);
    return limit ? `${limit} FPS` : 'Display';
  }

  function normalizeUserSettings(value) {
    const source = value && typeof value === 'object' ? value : {};
    const video = source.video && typeof source.video === 'object' ? source.video : {};
    const audio = source.audio && typeof source.audio === 'object' ? source.audio : {};
    const accessibility = source.accessibility && typeof source.accessibility === 'object' ? source.accessibility : {};
    const preset = getViewportPreset(video.viewportPreset);
    const custom = !preset && normalizeId(video.viewportPreset) === 'custom';
    const base = preset || DEFAULT_USER_SETTINGS.video;
    return {
      video: {
        viewportPreset: custom ? 'custom' : preset ? preset.id : DEFAULT_USER_SETTINGS.video.viewportPreset,
        width: clamp(Math.round(Number(video.width || base.width) || base.width), 1024, 1920),
        height: clamp(Math.round(Number(video.height || base.height) || base.height), 704, 1080),
        hudScale: normalizeHudScale(video.hudScale),
        frameRateLimit: normalizeFrameRateLimit(video.frameRateLimit)
      },
      audio: {
        sfxEnabled: !!audio.sfxEnabled,
        sfxVolume: clamp(Number(audio.sfxVolume == null ? DEFAULT_USER_SETTINGS.audio.sfxVolume : audio.sfxVolume), 0, 1),
        musicEnabled: !!audio.musicEnabled,
        musicVolume: clamp(Number(audio.musicVolume == null ? DEFAULT_USER_SETTINGS.audio.musicVolume : audio.musicVolume), 0, 1)
      },
      accessibility: {
        reducedEffects: !!accessibility.reducedEffects,
        damageNumbers: normalizeDamageNumberDensity(accessibility.damageNumbers)
      }
    };
  }

  function getSettingsPanelMetadata(settings, options) {
    const normalized = normalizeUserSettings(settings);
    const video = normalized.video;
    const audio = normalized.audio;
    const accessibility = normalized.accessibility;
    const panelOptions = options || {};
    const hudScales = Array.isArray(panelOptions.hudScales) ? panelOptions.hudScales : [0.9, 1, 1.1, 1.2];
    const damageModes = Array.isArray(panelOptions.damageModes) ? panelOptions.damageModes : ['normal', 'reduced', 'minimal'];
    const frameRateOptions = Array.isArray(panelOptions.frameRateOptions) ? panelOptions.frameRateOptions : FRAME_RATE_LIMIT_OPTIONS;
    const getDamageLabel = typeof panelOptions.formatStatName === 'function'
      ? panelOptions.formatStatName
      : function formatDamageLabelFallback(value) {
          return normalizeId(value).replace(/(^|-)([a-z])/g, (match) => match.toUpperCase()).replace(/-/g, ' ');
        };
    const getFrameRateLabel = typeof panelOptions.formatFrameRateLimitLabel === 'function'
      ? panelOptions.formatFrameRateLimitLabel
      : formatFrameRateLimitLabel;
    return {
      settings: normalized,
      viewportPresets: VIEWPORT_PRESETS.map((preset) => ({
        id: preset.id,
        label: preset.label,
        selected: video.viewportPreset === preset.id,
        ariaPressed: video.viewportPreset === preset.id ? 'true' : 'false'
      })),
      widthInput: {
        value: Number(video.width || 1280),
        min: 1024,
        max: 1920,
        step: 1
      },
      heightInput: {
        value: Number(video.height || 806),
        min: 704,
        max: 1080,
        step: 1
      },
      audio: {
        sfxEnabled: !!audio.sfxEnabled,
        sfxVolumePercent: Math.round(audio.sfxVolume * 100),
        sfxVolumeValue: Number(audio.sfxVolume || 0),
        musicEnabled: !!audio.musicEnabled,
        musicVolumePercent: Math.round(audio.musicVolume * 100),
        musicVolumeValue: Number(audio.musicVolume || 0)
      },
      hudScaleButtons: hudScales.map((scale) => ({
        value: scale,
        label: `${Math.round(scale * 100)}%`,
        selected: Number(video.hudScale) === scale,
        ariaPressed: Number(video.hudScale) === scale ? 'true' : 'false'
      })),
      damageButtons: damageModes.map((mode) => ({
        value: mode,
        label: getDamageLabel(mode),
        selected: accessibility.damageNumbers === mode,
        ariaPressed: accessibility.damageNumbers === mode ? 'true' : 'false'
      })),
      frameRateButtons: frameRateOptions.map((limit) => ({
        value: limit,
        label: getFrameRateLabel(limit),
        selected: Number(video.frameRateLimit) === limit,
        ariaPressed: Number(video.frameRateLimit) === limit ? 'true' : 'false'
      })),
      reducedEffects: !!accessibility.reducedEffects
    };
  }

  function getSettingsPresetControlsMetadata(video, x, y, w, options) {
    const source = video || {};
    const settings = options || {};
    const presets = Array.isArray(settings.presets) ? settings.presets : VIEWPORT_PRESETS;
    const h = 112;
    const gap = 6;
    const buttonW = Math.max(64, Math.floor((w - 24 - gap * (presets.length - 1)) / presets.length));
    return {
      h,
      frame: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: '#fbfaf6',
        stroke: 'rgba(16,32,51,0.14)'
      },
      titleText: {
        value: 'Window Size',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      sizeText: {
        value: `${Number(source.width || 1280)} x ${Number(source.height || 806)}`,
        x: x + w - 12,
        y: y + 10,
        color: source.viewportPreset === 'custom' ? '#8a6b00' : '#2f7dd6',
        font: '900 11px system-ui',
        align: 'right',
        maxWidth: 130,
        lineHeight: 12,
        maxLines: 1
      },
      presetButtons: presets.map((preset, index) => {
        const buttonX = x + 12 + index * (buttonW + gap);
        const selected = source.viewportPreset === preset.id;
        return {
          label: preset.label,
          x: buttonX,
          y: y + 34,
          w: buttonW,
          h: 28,
          region: { type: 'setting-preset', presetId: preset.id },
          disabled: false,
          selected,
          highlight: selected
            ? {
                x: buttonX + 2,
                y: y + 36,
                w: buttonW - 4,
                h: 24,
                radius: 6,
                fill: 'rgba(47,125,214,0.14)',
                stroke: 'rgba(47,125,214,0.72)'
              }
            : null
        };
      }),
      sizeButtons: [
        {
          label: '- Width',
          x: x + 12,
          y: y + 74,
          w: 70,
          h: 26,
          region: { type: 'setting-size', settingId: 'width', delta: -128 },
          disabled: source.width <= 1024
        },
        {
          label: '+ Width',
          x: x + 88,
          y: y + 74,
          w: 70,
          h: 26,
          region: { type: 'setting-size', settingId: 'width', delta: 128 },
          disabled: source.width >= 1920
        },
        {
          label: '- Height',
          x: x + 168,
          y: y + 74,
          w: 76,
          h: 26,
          region: { type: 'setting-size', settingId: 'height', delta: -74 },
          disabled: source.height <= 704
        },
        {
          label: '+ Height',
          x: x + 250,
          y: y + 74,
          w: 76,
          h: 26,
          region: { type: 'setting-size', settingId: 'height', delta: 74 },
          disabled: source.height >= 1080
        }
      ],
      nextY: y + h
    };
  }

  function getSettingsAudioControlsMetadata(audio, x, y, w) {
    const source = audio || {};
    const h = 94;
    return {
      h,
      frame: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: '#fbfaf6',
        stroke: 'rgba(16,32,51,0.14)'
      },
      titleText: {
        value: 'Audio',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      buttons: [
        {
          label: source.sfxEnabled ? 'SFX On' : 'SFX Off',
          x: x + 12,
          y: y + 34,
          w: 82,
          h: 26,
          region: { type: 'setting-toggle', settingId: 'sfxEnabled' },
          disabled: false
        },
        {
          label: '- SFX',
          x: x + 102,
          y: y + 34,
          w: 58,
          h: 26,
          region: { type: 'setting-volume', settingId: 'sfxVolume', delta: -0.1 },
          disabled: source.sfxVolume <= 0
        },
        {
          label: `${Math.round(source.sfxVolume * 100)}%`,
          x: x + 166,
          y: y + 34,
          w: 58,
          h: 26,
          region: null,
          disabled: true
        },
        {
          label: '+ SFX',
          x: x + 230,
          y: y + 34,
          w: 58,
          h: 26,
          region: { type: 'setting-volume', settingId: 'sfxVolume', delta: 0.1 },
          disabled: source.sfxVolume >= 1
        },
        {
          label: source.musicEnabled ? 'Music On' : 'Music Off',
          x: x + 12,
          y: y + 64,
          w: 82,
          h: 26,
          region: { type: 'setting-toggle', settingId: 'musicEnabled' },
          disabled: false
        },
        {
          label: '- Music',
          x: x + 102,
          y: y + 64,
          w: 58,
          h: 26,
          region: { type: 'setting-volume', settingId: 'musicVolume', delta: -0.1 },
          disabled: source.musicVolume <= 0
        },
        {
          label: `${Math.round(source.musicVolume * 100)}%`,
          x: x + 166,
          y: y + 64,
          w: 58,
          h: 26,
          region: null,
          disabled: true
        },
        {
          label: '+ Music',
          x: x + 230,
          y: y + 64,
          w: 58,
          h: 26,
          region: { type: 'setting-volume', settingId: 'musicVolume', delta: 0.1 },
          disabled: source.musicVolume >= 1
        }
      ],
      nextY: y + h
    };
  }

  function getSettingsDisplayControlsMetadata(video, accessibility, x, y, w, options) {
    const videoSource = video || {};
    const accessibilitySource = accessibility || {};
    const settings = options || {};
    const scaleOptions = Array.isArray(settings.scaleOptions) ? settings.scaleOptions : [0.9, 1, 1.1, 1.2];
    const damageModes = Array.isArray(settings.damageModes)
      ? settings.damageModes
      : [
          { id: 'normal', label: 'Normal' },
          { id: 'reduced', label: 'Reduced' },
          { id: 'minimal', label: 'Minimal' }
        ];
    const frameRateOptions = Array.isArray(settings.frameRateOptions) ? settings.frameRateOptions : FRAME_RATE_LIMIT_OPTIONS;
    const getFrameRateLabel = typeof settings.formatFrameRateLimitLabel === 'function'
      ? settings.formatFrameRateLimitLabel
      : formatFrameRateLimitLabel;
    const h = 126;
    const createHighlight = (buttonX, buttonY, width, height) => ({
      x: buttonX + 2,
      y: buttonY + 2,
      w: width - 4,
      h: height - 4,
      radius: 6,
      fill: 'rgba(47,125,214,0.14)',
      stroke: 'rgba(47,125,214,0.72)'
    });
    const hudScaleButtons = scaleOptions.map((scale, index) => {
      const buttonX = x + 12 + index * 60;
      const buttonY = y + 34;
      const selected = Math.abs(Number(videoSource.hudScale || 1) - scale) < 0.01;
      return {
        label: `${Math.round(scale * 100)}%`,
        x: buttonX,
        y: buttonY,
        w: 54,
        h: 26,
        region: { type: 'setting-hud-scale', value: scale },
        disabled: false,
        selected,
        highlight: selected ? createHighlight(buttonX, buttonY, 54, 26) : null
      };
    });
    const damageModeButtons = damageModes.map((mode, index) => {
      const buttonX = x + 12 + index * 78;
      const buttonY = y + 70;
      const selected = accessibilitySource.damageNumbers === mode.id;
      return {
        label: mode.label,
        x: buttonX,
        y: buttonY,
        w: 72,
        h: 26,
        region: { type: 'setting-damage-density', value: mode.id },
        disabled: false,
        selected,
        highlight: selected ? createHighlight(buttonX, buttonY, 72, 26) : null
      };
    });
    const frameRateButtons = frameRateOptions.map((limit, index) => {
      const buttonX = x + 12 + index * 60;
      const buttonY = y + 100;
      const selected = Number(videoSource.frameRateLimit) === limit;
      return {
        label: getFrameRateLabel(limit),
        x: buttonX,
        y: buttonY,
        w: 54,
        h: 26,
        region: { type: 'setting-frame-rate', value: limit },
        disabled: false,
        selected,
        highlight: selected ? createHighlight(buttonX, buttonY, 54, 26) : null
      };
    });
    return {
      h,
      frame: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: '#fbfaf6',
        stroke: 'rgba(16,32,51,0.14)'
      },
      titleText: {
        value: 'Display',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      hudScaleButtons,
      damageModeButtons,
      frameRateButtons,
      reducedEffectsButton: {
        label: accessibilitySource.reducedEffects ? 'Reduced FX On' : 'Reduced FX Off',
        x: x + Math.min(262, Math.max(12, w - 138)),
        y: y + 100,
        w: 126,
        h: 26,
        region: { type: 'setting-toggle', settingId: 'reducedEffects' },
        disabled: false
      },
      nextY: y + h
    };
  }

  function getSettingsDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const presetId = getAttribute('data-starfall-setting-preset');
    if (presetId) return { handled: true, type: 'preset', presetId };
    const hudScale = getAttribute('data-starfall-setting-hud');
    if (hudScale) return { handled: true, type: 'hudScale', value: Number(hudScale) };
    const damageDensity = getAttribute('data-starfall-setting-damage');
    if (damageDensity) return { handled: true, type: 'damageDensity', value: damageDensity };
    if (hasAttribute('data-starfall-setting-frame-rate')) {
      return { handled: true, type: 'frameRate', value: getAttribute('data-starfall-setting-frame-rate') };
    }
    if (hasAttribute('data-starfall-reset-settings')) return { handled: true, type: 'reset' };
    return { handled: false, type: '' };
  }

  function getSettingsRegionAction(region) {
    const source = region || {};
    if (source.type === 'setting-preset') return { handled: true, type: 'preset', presetId: source.presetId };
    if (source.type === 'setting-size') {
      return { handled: true, type: 'size', settingId: source.settingId, delta: source.delta };
    }
    if (source.type === 'setting-toggle') return { handled: true, type: 'toggle', settingId: source.settingId };
    if (source.type === 'setting-volume') {
      return { handled: true, type: 'volume', settingId: source.settingId, delta: source.delta };
    }
    if (source.type === 'setting-hud-scale') return { handled: true, type: 'hudScale', value: source.value };
    if (source.type === 'setting-damage-density') return { handled: true, type: 'damageDensity', value: source.value };
    if (source.type === 'setting-frame-rate') return { handled: true, type: 'frameRate', value: source.value };
    if (source.type === 'reset-settings') return { handled: true, type: 'reset' };
    return { handled: false, type: '' };
  }

  function getSettingsInputDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const settingId = getAttribute('data-starfall-setting');
    if (!settingId) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'applySettingInput',
      settingId,
      raw: source && source.type === 'checkbox' ? source.checked : source && source.value
    };
  }

  function createSettingsUiHelpers() {
    return Object.freeze({
      getViewportPreset,
      normalizeHudScale,
      normalizeDamageNumberDensity,
      normalizeFrameRateLimit,
      formatFrameRateLimitLabel,
      normalizeUserSettings,
      getSettingsPanelMetadata,
      getSettingsPresetControlsMetadata,
      getSettingsAudioControlsMetadata,
      getSettingsDisplayControlsMetadata,
      getSettingsDomAction,
      getSettingsRegionAction,
      getSettingsInputDomAction
    });
  }

  const api = {
    SETTINGS_STORAGE_KEY,
    VIEWPORT_PRESETS,
    FRAME_RATE_LIMIT_OPTIONS,
    DEFAULT_FRAME_RATE_LIMIT,
    DEFAULT_USER_SETTINGS,
    createSettingsUiHelpers,
    getViewportPreset,
    normalizeHudScale,
    normalizeDamageNumberDensity,
    normalizeFrameRateLimit,
    formatFrameRateLimitLabel,
    normalizeUserSettings,
    getSettingsPanelMetadata,
    getSettingsPresetControlsMetadata,
    getSettingsAudioControlsMetadata,
    getSettingsDisplayControlsMetadata,
    getSettingsDomAction,
    getSettingsRegionAction,
    getSettingsInputDomAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.settings = Object.assign({}, modules.settings || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
