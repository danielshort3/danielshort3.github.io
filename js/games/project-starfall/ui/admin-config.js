(function initProjectStarfallUiAdminConfig(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreSettings = (typeof require === 'function' ? require('../core/settings.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeAdminRateSetting = CoreSettings.normalizeAdminRate || function normalizeAdminRateSettingFallback(value, fallback, min, max) {
    const low = Number.isFinite(Number(min)) ? Number(min) : 1;
    const high = Number.isFinite(Number(max)) ? Number(max) : 1000;
    const defaultValue = Number.isFinite(Number(fallback)) ? Number(fallback) : low;
    return Math.max(low, Math.min(high, Math.round(Number(value) || defaultValue)));
  };
  const normalizePerformanceDebugModeSetting = CoreSettings.normalizePerformanceDebugMode || function normalizePerformanceDebugModeSettingFallback(value, modes, fallback) {
    const source = Array.isArray(modes) ? modes : [];
    const defaultValue = String(fallback || 'off');
    const mode = String(value || defaultValue);
    return source.includes(mode) ? mode : defaultValue;
  };

  const ADMIN_RATE_MIN = 1;
  const ADMIN_RATE_MAX = 1000;
  const PERFORMANCE_DEBUG_MODES = Object.freeze(['off', 'fps', 'breakdown']);
  const PERFORMANCE_DEBUG_MODE_LABELS = Object.freeze({
    off: 'Off',
    fps: 'FPS',
    breakdown: 'Breakdown'
  });
  const PERFORMANCE_DEBUG_OVERLAY_CACHE_MS = 250;
  const ADMIN_CONSOLE_TABS = Object.freeze([
    { id: 'commands', label: 'Commands' },
    { id: 'mobs', label: 'Mobs' },
    { id: 'bosses', label: 'Bosses' },
    { id: 'inventory', label: 'Inventory' },
    { id: 'gear', label: 'Gear Editor' },
    { id: 'attunement', label: 'Attunement' }
  ]);
  const ADMIN_CONSOLE_ITEM_ID = 'admin_worldwright_console';
  const DOM_ADMIN_COMMAND_INPUT_ATTRIBUTES = Object.freeze([
    'data-starfall-admin-command-input'
  ]);
  const DOM_ADMIN_COMMAND_INPUT_SELECTOR = DOM_ADMIN_COMMAND_INPUT_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_ADMIN_RATE_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-admin-rate'
  ]);
  const DOM_ADMIN_RATE_TARGET_SELECTOR = DOM_ADMIN_RATE_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function normalizeAdminRate(value, options) {
    const settings = options || {};
    const min = Number.isFinite(Number(settings.min)) ? Number(settings.min) : ADMIN_RATE_MIN;
    const max = Number.isFinite(Number(settings.max)) ? Number(settings.max) : ADMIN_RATE_MAX;
    const fallback = Number.isFinite(Number(settings.fallback)) ? Number(settings.fallback) : min;
    return normalizeAdminRateSetting(value, fallback, min, max);
  }

  function adminRateToPercent(value, options) {
    const settings = options || {};
    const max = Number.isFinite(Number(settings.max)) ? Number(settings.max) : ADMIN_RATE_MAX;
    const normalized = normalizeAdminRate(value, settings);
    return clamp(Math.log10(normalized) / Math.log10(max), 0, 1);
  }

  function adminPercentToRate(value, options) {
    const settings = options || {};
    const max = Number.isFinite(Number(settings.max)) ? Number(settings.max) : ADMIN_RATE_MAX;
    const percent = clamp(Number(value) || 0, 0, 1);
    return normalizeAdminRate(Math.pow(max, percent), settings);
  }

  function normalizePerformanceDebugMode(value, options) {
    const settings = options || {};
    const modes = Array.isArray(settings.modes) ? settings.modes : PERFORMANCE_DEBUG_MODES;
    return normalizePerformanceDebugModeSetting(value, modes, settings.fallback || 'off');
  }

  function getPerformanceDebugModeLabel(mode, options) {
    const labels = options && options.labels || PERFORMANCE_DEBUG_MODE_LABELS;
    return labels[normalizePerformanceDebugMode(mode, options)] || labels.off;
  }

  function getPerformanceDebugOverlayCacheState(cache, w, h, now, options) {
    const source = cache || {};
    const settings = options || {};
    const maxAgeMs = Number(settings.maxAgeMs || PERFORMANCE_DEBUG_OVERLAY_CACHE_MS);
    const stale = !source.canvas || source.w !== w || source.h !== h || Number(now || 0) - Number(source.updatedAt || 0) > maxAgeMs;
    return {
      stale,
      reusableCanvas: source.canvas && source.w === w && source.h === h ? source.canvas : null
    };
  }

  function getPerformanceDebugOverlayMetadata(debug, width, height, bottomY, options) {
    if (!debug) return null;
    const settings = options || {};
    const mode = normalizePerformanceDebugMode(debug.mode, settings);
    if (mode === 'off') return null;
    if (mode === 'fps') {
      const frameMs = Number(debug.frameMs || debug.averageFrameMs || 0);
      const slow = frameMs >= Number(debug.slowFrameThresholdMs || 33.34);
      const w = 116;
      const h = 30;
      const x = Math.round(width / 2 - w / 2);
      const y = 14;
      const formatFrameMs = typeof settings.formatPerformanceMs === 'function'
        ? settings.formatPerformanceMs
        : (value) => `${Number(value || 0).toFixed(1)}ms`;
      return {
        mode,
        fps: Math.round(Number(debug.currentFps || debug.averageFps || 0)),
        frameMs,
        slow,
        frame: {
          x,
          y,
          w,
          h,
          radius: 8,
          fill: slow ? 'rgba(128,45,39,0.88)' : 'rgba(9,31,59,0.86)',
          stroke: 'rgba(255,255,255,0.28)'
        },
        fpsText: {
          value: `${Math.round(Number(debug.currentFps || debug.averageFps || 0))} FPS`,
          x: x + 10,
          y: y + 7,
          color: '#ffffff',
          font: '950 12px system-ui',
          maxWidth: 58,
          lineHeight: 12,
          maxLines: 1
        },
        frameText: {
          value: formatFrameMs(frameMs),
          x: x + w - 10,
          y: y + 8,
          color: slow ? '#ffd166' : '#9be7ff',
          font: '900 10px system-ui',
          align: 'right',
          maxWidth: 50,
          lineHeight: 11,
          maxLines: 1
        }
      };
    }
    const w = 408;
    const h = 264;
    const x = clamp(Math.round(width / 2 - w / 2), 10, Math.max(10, width - w - 10));
    const y = clamp(12, 8, Math.max(8, Number(bottomY || height) - h - 10));
    return {
      mode,
      box: { x, y, w, h }
    };
  }

  function getPerformanceDebugModeControlsMetadata(mode, x, y, w, options) {
    const settings = options || {};
    const modes = Array.isArray(settings.modes) ? settings.modes : PERFORMANCE_DEBUG_MODES;
    const h = 88;
    const buttonY = y + 36;
    const gap = 7;
    const copyW = 142;
    const modeW = Math.max(56, Math.floor((w - 24 - copyW - gap * modes.length) / modes.length));
    const getLabel = typeof settings.getPerformanceDebugModeLabel === 'function'
      ? settings.getPerformanceDebugModeLabel
      : (candidate) => getPerformanceDebugModeLabel(candidate, settings);
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
        value: 'Performance Debug',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      statusText: {
        value: `F3 cycles: ${getLabel(mode)}`,
        x: x + w - 12,
        y: y + 10,
        color: mode === 'off' ? '#5f6f7a' : '#2f7dd6',
        font: '900 11px system-ui',
        align: 'right'
      },
      modeButtons: modes.map((candidate, index) => {
        const buttonX = x + 12 + index * (modeW + gap);
        const selected = candidate === mode;
        return {
          label: getLabel(candidate),
          x: buttonX,
          y: buttonY,
          w: modeW,
          h: 28,
          region: { type: 'performance-debug-mode', mode: candidate },
          disabled: false,
          selected,
          highlight: selected
            ? {
                x: buttonX + 2,
                y: buttonY + 2,
                w: modeW - 4,
                h: 24,
                radius: 6,
                fill: 'rgba(47,125,214,0.14)',
                stroke: 'rgba(47,125,214,0.72)'
              }
            : null
        };
      }),
      copyButton: {
        label: 'Copy Debug Report',
        x: x + w - copyW - 12,
        y: buttonY,
        w: copyW,
        h: 28,
        region: { type: 'copy-performance-debug' },
        disabled: false
      },
      nextY: y + h
    };
  }

  function getPerformanceBenchmarkControlsMetadata(benchmark, x, y, w, options) {
    const settings = options || {};
    const state = benchmark || { active: false, phase: 'idle', progress: 0 };
    const result = state.result || null;
    const h = 104;
    const progress = clamp(Number(state.progress || 0), 0, 1);
    const formatFps = typeof settings.formatPerformanceFps === 'function'
      ? settings.formatPerformanceFps
      : (value) => `${Math.round(Number(value) || 0)} fps`;
    const formatMs = typeof settings.formatPerformanceMs === 'function'
      ? settings.formatPerformanceMs
      : (value) => `${(Number(value) || 0).toFixed(2)}ms`;
    const phaseLabel = state.active
      ? state.phase === 'sample' ? 'Sampling' : 'Warming up'
      : result ? 'Complete' : 'Ready';
    const resultLabel = result
      ? `${formatFps(result.averageFps)} avg | p95 ${formatMs(result.p95FrameMs)} | slow ${Number(result.slowFrameCount || 0)}`
      : state.active
        ? `${state.mapName || 'Benchmark Arena'} | ${state.characterName || 'Benchmark Echo'} | ${Math.round(progress * 100)}%`
        : 'Dedicated benchmark arena';
    const bottleneck = result && result.suspectedBottleneck
      ? `${result.suspectedBottleneck.group}:${result.suspectedBottleneck.name} ${formatMs(result.suspectedBottleneck.averageMs)}`
      : '';
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
        value: 'In-Game Benchmark',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      statusText: {
        value: phaseLabel,
        x: x + w - 12,
        y: y + 10,
        color: state.active || result ? '#2f7dd6' : '#5f6f7a',
        font: '900 11px system-ui',
        align: 'right'
      },
      resultText: {
        value: resultLabel,
        x: x + 12,
        y: y + 31,
        color: '#5f6f7a',
        font: '800 10px system-ui',
        maxWidth: w - 24,
        lineHeight: 11,
        maxLines: 1
      },
      bottleneckText: bottleneck
        ? {
            value: `Top: ${bottleneck}`,
            x: x + 12,
            y: y + 47,
            color: '#5f6f7a',
            font: '800 10px system-ui',
            maxWidth: w - 24,
            lineHeight: 11,
            maxLines: 1
          }
        : null,
      progressBar: !bottleneck && state.active
        ? {
            track: {
              x: x + 12,
              y: y + 52,
              w: w - 24,
              h: 8,
              radius: 6,
              fill: 'rgba(16,32,51,0.14)',
              stroke: ''
            },
            fill: {
              x: x + 12,
              y: y + 52,
              w: Math.max(6, (w - 24) * progress),
              h: 8,
              radius: 6,
              fill: '#2f7dd6',
              stroke: ''
            }
          }
        : null,
      startButton: {
        label: state.active ? 'Cancel' : 'Run Benchmark',
        x: x + 12,
        y: y + 70,
        w: 124,
        h: 28,
        region: { type: state.active ? 'performance-benchmark-cancel' : 'performance-benchmark-start' },
        disabled: false
      },
      copyButton: {
        label: 'Copy Result',
        x: x + 144,
        y: y + 70,
        w: 104,
        h: 28,
        region: { type: 'copy-performance-benchmark' },
        disabled: !result
      },
      nextY: y + h
    };
  }

  function getPerformanceDebugDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const debugMode = getAttribute('data-starfall-performance-debug-mode');
    if (debugMode) return { handled: true, type: 'setMode', mode: debugMode };
    if (hasAttribute('data-starfall-copy-performance-debug')) return { handled: true, type: 'copyDebugReport' };
    if (hasAttribute('data-starfall-copy-admin-debug')) return { handled: true, type: 'copyAdminDebugLog' };
    if (hasAttribute('data-starfall-clear-admin-debug')) return { handled: true, type: 'clearAdminDebugLog' };
    const benchmarkAction = getAttribute('data-starfall-performance-benchmark');
    if (benchmarkAction) return { handled: true, type: 'benchmark', action: benchmarkAction };
    if (hasAttribute('data-starfall-copy-performance-benchmark')) return { handled: true, type: 'copyBenchmarkReport' };
    if (hasAttribute('data-starfall-toggle-combat-metrics')) return { handled: true, type: 'toggleCombatMetrics' };
    return { handled: false, type: '' };
  }

  function getPerformanceDebugRegionAction(region) {
    const source = region || {};
    if (source.type === 'toggle-combat-metrics') return { handled: true, type: 'toggleCombatMetrics' };
    if (source.type === 'reset-admin-settings') return { handled: true, type: 'resetAdminSettings' };
    if (source.type === 'performance-debug-mode') return { handled: true, type: 'setMode', mode: source.mode };
    if (source.type === 'copy-performance-debug') return { handled: true, type: 'copyDebugReport' };
    if (source.type === 'performance-benchmark-start') return { handled: true, type: 'benchmark', action: 'start' };
    if (source.type === 'performance-benchmark-cancel') return { handled: true, type: 'benchmark', action: 'cancel' };
    if (source.type === 'copy-performance-benchmark') return { handled: true, type: 'copyBenchmarkReport' };
    return { handled: false, type: '' };
  }

  function getAdminResetDomAction(target) {
    const source = target || null;
    if (source && typeof source.hasAttribute === 'function' && source.hasAttribute('data-starfall-reset-admin')) {
      return { handled: true, type: 'resetAdminSettings' };
    }
    return { handled: false, type: '' };
  }

  function getAdminGameplayDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const spawnMode = getAttribute('data-starfall-admin-spawn');
    if (spawnMode) return { handled: true, type: 'spawn', mode: spawnMode };
    if (hasAttribute('data-starfall-admin-kill-enemies')) return { handled: true, type: 'killEnemies' };
    if (hasAttribute('data-starfall-admin-boss-teleport')) return { handled: true, type: 'bossTeleport' };
    if (hasAttribute('data-starfall-admin-grant')) return { handled: true, type: 'grant' };
    if (hasAttribute('data-starfall-admin-gear-min')) return { handled: true, type: 'gearStat', mode: 'min' };
    if (hasAttribute('data-starfall-admin-gear-max')) return { handled: true, type: 'gearStat', mode: 'max' };
    if (hasAttribute('data-starfall-admin-gear-apply')) return { handled: true, type: 'gearStat', mode: 'value' };
    if (hasAttribute('data-starfall-admin-attunement-apply')) return { handled: true, type: 'attunementApply' };
    return { handled: false, type: '' };
  }

  function getAdminRateSliderMetadata(label, rateId, value, x, y, w, options) {
    const settings = options || {};
    const h = 72;
    const trackX = x + 16;
    const trackY = y + 40;
    const trackW = Math.max(120, w - 32);
    const percent = adminRateToPercent(value, settings);
    const knobX = trackX + trackW * percent;
    const tickSource = Array.isArray(settings.ticks)
      ? settings.ticks
      : [[ADMIN_RATE_MIN, '1x'], [10, '10x'], [100, '100x'], [ADMIN_RATE_MAX, '1000x']];
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
      labelText: {
        value: label,
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      valueText: {
        value: `${value}x`,
        x: x + w - 12,
        y: y + 10,
        color: '#2f7dd6',
        font: '900 12px system-ui',
        align: 'right'
      },
      track: {
        x: trackX,
        y: trackY,
        w: trackW,
        h: 8,
        radius: 6,
        fill: 'rgba(16,32,51,0.14)',
        stroke: ''
      },
      fill: {
        x: trackX,
        y: trackY,
        w: Math.max(8, knobX - trackX),
        h: 8,
        radius: 6,
        fill: '#2f7dd6',
        stroke: ''
      },
      knob: {
        x: knobX - 7,
        y: trackY - 6,
        w: 14,
        h: 20,
        radius: 7,
        fill: '#ffffff',
        stroke: 'rgba(47,125,214,0.78)'
      },
      ticks: tickSource.map(([tickValue, tickLabel]) => {
        const tickX = trackX + trackW * adminRateToPercent(tickValue, settings);
        return {
          value: tickValue,
          label: tickLabel,
          marker: {
            x: tickX - 1,
            y: trackY + 13,
            w: 2,
            h: 6,
            fillStyle: 'rgba(16,32,51,0.32)'
          },
          text: {
            value: tickLabel,
            x: tickX,
            y: trackY + 22,
            color: '#5f6f7a',
            font: '9px system-ui',
            align: 'center'
          }
        };
      }),
      region: {
        type: 'admin-rate-slider',
        rateId,
        trackX,
        trackW,
        x: trackX - 8,
        y: trackY - 14,
        w: trackW + 16,
        h: 42
      },
      nextY: y + h
    };
  }

  function getAdminRateSliderStartAction(region) {
    const source = region || {};
    if (source.type !== 'admin-rate-slider') {
      return { handled: false, type: '', drag: null, shouldPreventDefault: false };
    }
    return {
      handled: true,
      type: 'startAdminRateSliderDrag',
      drag: { rateId: source.rateId, trackX: source.trackX, trackW: source.trackW },
      shouldPreventDefault: true
    };
  }

  function getAdminRateSliderMoveAction(drag) {
    const source = drag || null;
    if (!source) {
      return { handled: false, type: '', drag: null, updateOptions: null, shouldPreventDefault: false };
    }
    return {
      handled: true,
      type: 'previewAdminRateFromPoint',
      drag: source,
      updateOptions: { coalesce: true },
      shouldPreventDefault: true
    };
  }

  function getAdminRateSliderReleaseAction(drag) {
    const source = drag || null;
    if (!source) return { handled: false, type: '', drag: null, shouldClearDrag: false };
    return {
      handled: true,
      type: 'commitAdminRateFromPoint',
      drag: source,
      shouldClearDrag: true
    };
  }

  function getAdminRateSliderCancelAction(drag) {
    const source = drag || null;
    if (!source) return { handled: false, type: '', rateId: '', shouldClearDrag: false };
    return {
      handled: true,
      type: 'commitAdminRatePreview',
      rateId: source.rateId,
      shouldClearDrag: true
    };
  }

  function getCycledAdminConsoleValue(control, values, state, delta) {
    const items = (values || []).filter(Boolean);
    if (!items.length) return { ok: false, value: null };
    const source = state || {};
    const current = source[control];
    const index = Math.max(0, items.findIndex((value) => value === current));
    return {
      ok: true,
      value: items[(index + Number(delta || 1) + items.length) % items.length]
    };
  }

  function getAdjustedAdminConsoleNumberValue(control, delta, min, max, state) {
    const source = state || {};
    const value = Number(source[control] || 0) || 0;
    return clamp(value + Number(delta || 0), min == null ? -999999 : min, max == null ? 999999 : max);
  }

  function normalizeAdminNumberBound(value, fallback) {
    if (value === '' || value == null) return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function getAdminNumberPromptValue(prompt) {
    const source = prompt || {};
    const raw = String(source.text == null ? '' : source.text).trim();
    const value = Number(raw);
    if (!Number.isFinite(value)) return null;
    return clamp(Math.round(value), normalizeAdminNumberBound(source.min, -999999), normalizeAdminNumberBound(source.max, 999999));
  }

  function createAdminNumberPrompt(control, min, max, label, state, options) {
    const settings = options || {};
    const id = String(control || '');
    if (!id) return null;
    const source = state || {};
    const value = source[id] == null ? '' : String(source[id]);
    const lower = normalizeAdminNumberBound(min, -999999);
    const upper = normalizeAdminNumberBound(max, 999999);
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (valueId) => valueId;
    return {
      control: id,
      label: String(label || formatStatName(id) || 'Value'),
      min: lower,
      max: upper,
      text: value === '' ? String(Math.max(0, lower)) : String(clamp(Math.round(Number(value) || 0), lower, upper))
    };
  }

  function sanitizeAdminNumberPromptText(text, options) {
    const settings = options || {};
    const cleaned = String(text == null ? '' : text).replace(/[^\d-]/g, '').slice(0, 8);
    const min = Object.prototype.hasOwnProperty.call(settings, 'min')
      ? normalizeAdminNumberBound(settings.min, -999999)
      : normalizeAdminNumberBound(settings.prompt && settings.prompt.min, -999999);
    return min < 0 ? cleaned.replace(/(?!^)-/g, '') : cleaned.replace(/-/g, '');
  }

  function getAdjustedAdminNumberPromptText(prompt, delta) {
    const source = prompt || {};
    const current = getAdminNumberPromptValue(source);
    const min = normalizeAdminNumberBound(source.min, -999999);
    const max = normalizeAdminNumberBound(source.max, 999999);
    const base = current == null ? clamp(0, min, max) : current;
    return String(clamp(base + Number(delta || 0), min, max));
  }

  function getAdminNumberPromptBoundText(prompt, bound) {
    const source = prompt || {};
    const value = bound === 'max'
      ? normalizeAdminNumberBound(source.max, 999999)
      : normalizeAdminNumberBound(source.min, -999999);
    return String(value);
  }

  function getAdminNumberPromptKeyAction(event, isDown, prompt) {
    const source = prompt || null;
    if (!isDown || !source) return { handled: false, preventDefault: false, type: '' };
    const code = event && event.code || '';
    if (code === 'Enter' || code === 'NumpadEnter') {
      return { handled: true, preventDefault: true, type: 'confirm' };
    }
    if (code === 'Escape') {
      return { handled: true, preventDefault: true, type: 'cancel' };
    }
    if (code === 'Backspace') {
      return {
        handled: true,
        preventDefault: true,
        type: 'setText',
        text: String(source.text || '').slice(0, -1)
      };
    }
    if (code === 'Delete') {
      return { handled: true, preventDefault: true, type: 'setText', text: '' };
    }
    if (code === 'ArrowLeft' || code === 'ArrowDown') {
      return { handled: true, preventDefault: true, type: 'adjust', delta: -1 };
    }
    if (code === 'ArrowRight' || code === 'ArrowUp') {
      return { handled: true, preventDefault: true, type: 'adjust', delta: 1 };
    }
    if (code === 'Home') {
      return { handled: true, preventDefault: true, type: 'bound', bound: 'min' };
    }
    if (code === 'End') {
      return { handled: true, preventDefault: true, type: 'bound', bound: 'max' };
    }
    if (/^Digit\d$/.test(code) || /^Numpad\d$/.test(code)) {
      const digit = code.replace('Digit', '').replace('Numpad', '');
      return {
        handled: true,
        preventDefault: true,
        type: 'setText',
        text: `${source.text || ''}${digit}`
      };
    }
    if (code === 'Minus' || code === 'NumpadSubtract') {
      const min = normalizeAdminNumberBound(source.min, -999999);
      if (min < 0) {
        const text = String(source.text || '');
        return {
          handled: true,
          preventDefault: true,
          type: 'setText',
          text: text.startsWith('-') ? text.slice(1) : `-${text}`
        };
      }
      return { handled: true, preventDefault: true, type: 'noop' };
    }
    return { handled: true, preventDefault: true, type: 'noop' };
  }

  function getAdminCommandInputKeyAction(event, isDown, state) {
    const source = state || {};
    if (!isDown) return { handled: true, preventDefault: false, type: 'noop' };
    const code = event && event.code || '';
    if (code === 'Enter' || code === 'NumpadEnter') {
      return { handled: true, preventDefault: true, type: 'run' };
    }
    if (code === 'ArrowUp' || code === 'ArrowDown') {
      const history = Array.isArray(source.commandHistory) ? source.commandHistory : [];
      if (!history.length) {
        return { handled: true, preventDefault: true, type: 'noop' };
      }
      const delta = code === 'ArrowUp' ? 1 : -1;
      const index = clamp(Math.floor(Number(source.commandHistoryIndex || 0) || 0) + delta, -1, history.length - 1);
      const selected = index >= 0 ? history[index] : null;
      return {
        handled: true,
        preventDefault: true,
        type: 'history',
        commandHistoryIndex: index,
        commandInput: selected && selected.command || ''
      };
    }
    if (code === 'Escape') {
      return { handled: true, preventDefault: true, type: 'blur' };
    }
    return { handled: false, preventDefault: false, type: '' };
  }

  function getAdminCommandInputTarget(target, selector) {
    const source = target || null;
    const commandInputSelector = selector || DOM_ADMIN_COMMAND_INPUT_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(commandInputSelector)
      : null;
  }

  function getAdminRateTarget(target, selector) {
    const source = target || null;
    const rateSelector = selector || DOM_ADMIN_RATE_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(rateSelector)
      : null;
  }

  function getAdminNumberPromptRegionAction(region) {
    const source = region || {};
    if (source.type === 'admin-number-decrease') return { handled: true, type: 'adjust', delta: -1 };
    if (source.type === 'admin-number-increase') return { handled: true, type: 'adjust', delta: 1 };
    if (source.type === 'admin-number-delta') return { handled: true, type: 'adjust', delta: source.delta };
    if (source.type === 'admin-number-min') return { handled: true, type: 'bound', bound: 'min' };
    if (source.type === 'admin-number-max') return { handled: true, type: 'bound', bound: 'max' };
    if (source.type === 'admin-number-clear') return { handled: true, type: 'setText', text: '' };
    if (source.type === 'admin-number-cancel') return { handled: true, type: 'cancel' };
    if (source.type === 'admin-number-apply') return { handled: true, type: 'confirm' };
    return { handled: false, type: '' };
  }

  function getAdminNumberPromptDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const control = getAttribute('data-starfall-admin-number-open');
    if (control) {
      return {
        handled: true,
        type: 'open',
        control,
        min: getAttribute('data-starfall-admin-number-min'),
        max: getAttribute('data-starfall-admin-number-max'),
        label: getAttribute('data-starfall-admin-number-label')
      };
    }
    const delta = getAttribute('data-starfall-admin-number-delta');
    if (delta) return { handled: true, type: 'adjust', delta: Number(delta) };
    if (hasAttribute('data-starfall-admin-number-min')) return { handled: true, type: 'bound', bound: 'min' };
    if (hasAttribute('data-starfall-admin-number-max')) return { handled: true, type: 'bound', bound: 'max' };
    if (hasAttribute('data-starfall-admin-number-clear')) return { handled: true, type: 'setText', text: '' };
    if (hasAttribute('data-starfall-admin-number-cancel')) return { handled: true, type: 'cancel' };
    if (hasAttribute('data-starfall-admin-number-apply')) return { handled: true, type: 'confirm' };
    return { handled: false, type: '' };
  }

  function getAdminConsoleDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (hasAttribute('data-starfall-admin-console-restore')) return { handled: true, type: 'restore' };
    if (hasAttribute('data-starfall-admin-console-open')) return { handled: true, type: 'open' };
    if (hasAttribute('data-starfall-admin-console-close')) return { handled: true, type: 'close' };
    const tabId = getAttribute('data-starfall-admin-console-tab');
    if (tabId) return { handled: true, type: 'setTab', tabId };
    if (hasAttribute('data-starfall-admin-command-run')) return { handled: true, type: 'runCommand' };
    const command = getAttribute('data-starfall-admin-command-sample');
    if (command) return { handled: true, type: 'setCommandInput', command };
    if (hasAttribute('data-starfall-admin-command-clear')) return { handled: true, type: 'clearCommand' };
    const pickerControl = getAttribute('data-starfall-admin-picker-toggle');
    if (pickerControl) return { handled: true, type: 'togglePicker', control: pickerControl };
    const pickerValue = getAttribute('data-starfall-admin-picker-option');
    if (pickerValue) {
      return {
        handled: true,
        type: 'selectPickerOption',
        control: getAttribute('data-starfall-admin-picker-control'),
        value: pickerValue
      };
    }
    return { handled: false, type: '' };
  }

  function getAdminConsoleInputDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const value = source && Object.prototype.hasOwnProperty.call(source, 'value') ? source.value : undefined;
    const control = getAttribute('data-starfall-admin-console-control');
    if (control) {
      return {
        handled: true,
        type: 'setControl',
        control,
        value,
        skipRender: source && source.tagName !== 'SELECT'
      };
    }
    if (hasAttribute('data-starfall-admin-command-input')) {
      return { handled: true, type: 'setCommandInput', value, skipRender: true };
    }
    return { handled: false, type: '' };
  }

  function getAdminRateInputDomAction(target, eventType) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const rateId = getAttribute('data-starfall-admin-rate');
    if (!rateId) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'previewRate',
      rateId,
      value: source && source.value,
      commit: eventType === 'change'
    };
  }

  function getAdminRateCommitDomAction(target) {
    const source = getAdminRateTarget(target);
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (!source) return { handled: false, type: '', target: null };
    return {
      handled: true,
      type: 'commitRatePreview',
      rateId: getAttribute('data-starfall-admin-rate'),
      value: source && source.value,
      target: source
    };
  }

  function getAdminConsoleRegionAction(region) {
    const source = region || {};
    if (source.type === 'admin-console-open') return { handled: true, type: 'toggleOpen' };
    if (source.type === 'admin-console-restore') return { handled: true, type: 'restore' };
    if (source.type === 'admin-console-close') return { handled: true, type: 'close' };
    if (source.type === 'admin-console-tab') return { handled: true, type: 'setTab', tabId: source.tabId };
    if (source.type === 'admin-console-command-run') return { handled: true, type: 'runCommand' };
    if (source.type === 'admin-console-command-sample') return { handled: true, type: 'setCommandInput', command: source.command || '' };
    if (source.type === 'admin-console-command-clear') return { handled: true, type: 'clearCommand' };
    if (source.type === 'admin-console-spawn') return { handled: true, type: 'spawn', mode: source.mode };
    if (source.type === 'admin-console-kill-enemies') return { handled: true, type: 'killEnemies' };
    if (source.type === 'admin-console-boss-teleport') return { handled: true, type: 'bossTeleport' };
    if (source.type === 'admin-console-grant') return { handled: true, type: 'grant' };
    if (source.type === 'admin-console-gear-min') return { handled: true, type: 'gearStat', mode: 'min' };
    if (source.type === 'admin-console-gear-max') return { handled: true, type: 'gearStat', mode: 'max' };
    if (source.type === 'admin-console-gear-apply') return { handled: true, type: 'gearStat', mode: 'value' };
    if (source.type === 'admin-console-attunement-apply') return { handled: true, type: 'attunementApply' };
    if (source.type === 'admin-console-picker') return { handled: true, type: 'togglePicker', control: source.control };
    if (source.type === 'admin-console-picker-option') {
      return { handled: true, type: 'selectPickerOption', control: source.control, value: source.value };
    }
    if (source.type === 'admin-console-cycle') {
      return {
        handled: true,
        type: 'cycle',
        control: source.control,
        values: source.values,
        delta: source.delta
      };
    }
    if (source.type === 'admin-console-number') {
      return {
        handled: true,
        type: 'adjustNumber',
        control: source.control,
        delta: source.delta,
        min: source.min,
        max: source.max
      };
    }
    if (source.type === 'admin-console-number-input') {
      return {
        handled: true,
        type: 'promptNumber',
        control: source.control,
        min: source.min,
        max: source.max,
        label: source.label
      };
    }
    return { handled: false, type: '' };
  }

  function createAdminConfigUiHelpers() {
    return Object.freeze({
      normalizeAdminRate,
      adminRateToPercent,
      adminPercentToRate,
      normalizePerformanceDebugMode,
      getPerformanceDebugModeLabel,
      getPerformanceDebugOverlayCacheState,
      getPerformanceDebugOverlayMetadata,
      getPerformanceDebugModeControlsMetadata,
      getPerformanceBenchmarkControlsMetadata,
      getPerformanceDebugDomAction,
      getPerformanceDebugRegionAction,
      getAdminResetDomAction,
      getAdminGameplayDomAction,
      getAdminRateSliderMetadata,
      getAdminRateSliderStartAction,
      getAdminRateSliderMoveAction,
      getAdminRateSliderReleaseAction,
      getAdminRateSliderCancelAction,
      getCycledAdminConsoleValue,
      getAdjustedAdminConsoleNumberValue,
      normalizeAdminNumberBound,
      getAdminNumberPromptValue,
      createAdminNumberPrompt,
      sanitizeAdminNumberPromptText,
      getAdjustedAdminNumberPromptText,
      getAdminNumberPromptBoundText,
      getAdminNumberPromptKeyAction,
      getAdminCommandInputKeyAction,
      getAdminCommandInputTarget,
      getAdminRateTarget,
      getAdminNumberPromptRegionAction,
      getAdminNumberPromptDomAction,
      getAdminConsoleDomAction,
      getAdminConsoleInputDomAction,
      getAdminRateInputDomAction,
      getAdminRateCommitDomAction,
      getAdminConsoleRegionAction
    });
  }

  function createAdminConfigDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_ADMIN_COMMAND_INPUT_ATTRIBUTES,
      DOM_ADMIN_COMMAND_INPUT_SELECTOR,
      DOM_ADMIN_RATE_TARGET_ATTRIBUTES,
      DOM_ADMIN_RATE_TARGET_SELECTOR
    });
  }

  const api = {
    ADMIN_RATE_MIN,
    ADMIN_RATE_MAX,
    PERFORMANCE_DEBUG_MODES,
    PERFORMANCE_DEBUG_MODE_LABELS,
    PERFORMANCE_DEBUG_OVERLAY_CACHE_MS,
    ADMIN_CONSOLE_TABS,
    ADMIN_CONSOLE_ITEM_ID,
    DOM_ADMIN_COMMAND_INPUT_ATTRIBUTES,
    DOM_ADMIN_COMMAND_INPUT_SELECTOR,
    DOM_ADMIN_RATE_TARGET_ATTRIBUTES,
    DOM_ADMIN_RATE_TARGET_SELECTOR,
    normalizeAdminRate,
    adminRateToPercent,
    adminPercentToRate,
    normalizePerformanceDebugMode,
    getPerformanceDebugModeLabel,
    getPerformanceDebugOverlayCacheState,
    getPerformanceDebugOverlayMetadata,
    getPerformanceDebugModeControlsMetadata,
    getPerformanceBenchmarkControlsMetadata,
    getPerformanceDebugDomAction,
    getPerformanceDebugRegionAction,
    getAdminResetDomAction,
    getAdminGameplayDomAction,
    getAdminRateSliderMetadata,
    getAdminRateSliderStartAction,
    getAdminRateSliderMoveAction,
    getAdminRateSliderReleaseAction,
    getAdminRateSliderCancelAction,
    getCycledAdminConsoleValue,
    getAdjustedAdminConsoleNumberValue,
    normalizeAdminNumberBound,
    getAdminNumberPromptValue,
    createAdminNumberPrompt,
    sanitizeAdminNumberPromptText,
    getAdjustedAdminNumberPromptText,
    getAdminNumberPromptBoundText,
    getAdminNumberPromptKeyAction,
    getAdminCommandInputKeyAction,
    getAdminCommandInputTarget,
    getAdminRateTarget,
    getAdminNumberPromptRegionAction,
    getAdminNumberPromptDomAction,
    getAdminConsoleDomAction,
    getAdminConsoleInputDomAction,
    getAdminRateInputDomAction,
    getAdminRateCommitDomAction,
    getAdminConsoleRegionAction,
    createAdminConfigUiHelpers,
    createAdminConfigDomSelectorUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.adminConfig = Object.assign({}, modules.adminConfig || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
