(function initProjectStarfallEngineDebugLog(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const ADMIN_DEBUG_LOG_LIMIT = 300;

  function clonePlain(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function normalizeDebugLogLimit(limit, maxLimit) {
    const cap = Math.max(1, Math.floor(Number(maxLimit || ADMIN_DEBUG_LOG_LIMIT) || ADMIN_DEBUG_LOG_LIMIT));
    const value = Math.floor(Number(limit || 80) || 80);
    return Math.max(1, Math.min(cap, value));
  }

  function cloneDebugDetails(details) {
    try {
      return clonePlain(details || {});
    } catch (err) {
      return { note: 'Details could not be cloned.' };
    }
  }

  function createDebugLogEntry(options) {
    const settings = options || {};
    const player = settings.player || {};
    return {
      seq: Math.max(0, Math.floor(Number(settings.seq || 0) || 0)),
      timestamp: Number(settings.timestamp || Date.now()),
      time: String(settings.time || new Date().toISOString()),
      category: normalizeId(settings.category) || 'debug',
      action: normalizeId(settings.action) || 'event',
      mapId: String(settings.mapId || ''),
      channelId: String(settings.channelId || ''),
      player: {
        x: Math.round(Number(player.x || 0)),
        y: Math.round(Number(player.y || 0)),
        grounded: !!player.grounded,
        platformId: normalizeId(player.groundedPlatformId || player.platformId)
      },
      details: cloneDebugDetails(settings.details)
    };
  }

  function appendDebugLogEntry(log, entry, limit) {
    const target = Array.isArray(log) ? log : [];
    if (entry) target.unshift(entry);
    const max = Math.max(1, Math.floor(Number(limit || ADMIN_DEBUG_LOG_LIMIT) || ADMIN_DEBUG_LOG_LIMIT));
    if (target.length > max) target.length = max;
    return target;
  }

  function isDebugLogSnapshotCacheCurrent(cache, log, max, seq) {
    if (!cache) return false;
    const source = Array.isArray(log) ? log : [];
    const first = source[0] || null;
    const lastIncluded = source.length > 0 ? source[Math.min(max, source.length) - 1] : null;
    return cache.log === source &&
      cache.max === max &&
      cache.seq === seq &&
      cache.length === source.length &&
      cache.first === first &&
      cache.lastIncluded === lastIncluded;
  }

  function createDebugLogSnapshot(log, options) {
    const settings = options || {};
    const source = Array.isArray(log) ? log : [];
    const max = normalizeDebugLogLimit(settings.limit, settings.maxLimit);
    return {
      entries: source.slice(0, max).map((entry) => clonePlain(entry)),
      total: source.length,
      limit: Math.max(1, Math.floor(Number(settings.maxLimit || ADMIN_DEBUG_LOG_LIMIT) || ADMIN_DEBUG_LOG_LIMIT))
    };
  }

  function createDebugLogSummarySnapshot(log, maxLimit) {
    return {
      entries: [],
      total: (Array.isArray(log) ? log : []).length,
      limit: Math.max(1, Math.floor(Number(maxLimit || ADMIN_DEBUG_LOG_LIMIT) || ADMIN_DEBUG_LOG_LIMIT))
    };
  }

  function createDebugLogReport(log) {
    const entries = (Array.isArray(log) ? log : []).slice().reverse();
    if (!entries.length) return 'Project Starfall debug log is empty.';
    return entries.map((entry) => {
      const details = entry.details && Object.keys(entry.details).length ? ` ${JSON.stringify(entry.details)}` : '';
      return `[${entry.time}] #${entry.seq} ${entry.category}:${entry.action} map=${entry.mapId || '-'} ch=${entry.channelId || '-'} p=${entry.player.x},${entry.player.y}${details}`;
    }).join('\n');
  }

  const api = {
    ADMIN_DEBUG_LOG_LIMIT,
    normalizeDebugLogLimit,
    cloneDebugDetails,
    createDebugLogEntry,
    appendDebugLogEntry,
    isDebugLogSnapshotCacheCurrent,
    createDebugLogSnapshot,
    createDebugLogSummarySnapshot,
    createDebugLogReport
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.debugLog = Object.assign({}, modules.debugLog || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
