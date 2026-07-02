(function initProjectStarfallCoreSettings(global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeHudScale(value, fallback) {
    const scale = Number(value);
    if (!Number.isFinite(scale)) return Number(fallback);
    return clamp(Math.round(scale * 10) / 10, 0.9, 1.2);
  }

  function normalizeDamageNumberDensity(value, fallback) {
    const id = String(value || '').trim();
    return ['normal', 'reduced', 'minimal'].includes(id) ? id : fallback;
  }

  function normalizeFrameRateLimit(value, options, fallback) {
    const source = Array.isArray(options) ? options : [];
    const rate = Math.round(Number(value));
    return source.includes(rate) ? rate : fallback;
  }

  function normalizeAdminRate(value, fallback, min, max) {
    const low = Number.isFinite(Number(min)) ? Number(min) : 1;
    const high = Number.isFinite(Number(max)) ? Number(max) : 1000;
    const defaultValue = Number.isFinite(Number(fallback)) ? Number(fallback) : low;
    return clamp(Math.round(Number(value) || defaultValue), low, high);
  }

  function normalizePerformanceDebugMode(value, modes, fallback) {
    const source = Array.isArray(modes) ? modes : [];
    const defaultValue = String(fallback || 'off');
    const mode = String(value || defaultValue);
    return source.includes(mode) ? mode : defaultValue;
  }

  const api = {
    normalizeHudScale,
    normalizeDamageNumberDensity,
    normalizeFrameRateLimit,
    normalizeAdminRate,
    normalizePerformanceDebugMode
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
