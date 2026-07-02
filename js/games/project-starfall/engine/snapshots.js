(function initProjectStarfallEngineSnapshots(global) {
  'use strict';

  const CoreTime = (typeof require === 'function' ? require('../core/time.js') : null) || global.ProjectStarfallCore || {};
  const performanceNowMs = CoreTime.nowMs || function performanceNowMsFallback() {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  };

  const OVERLAY_SNAPSHOT_CACHE_MS = 750;
  const OVERLAY_SNAPSHOT_DYNAMIC_FAST_MS = 120;
  const OVERLAY_SNAPSHOT_DYNAMIC_STATUS_MS = 90;
  const OVERLAY_SNAPSHOT_DYNAMIC_WORLD_MS = 500;
  const OVERLAY_SNAPSHOT_CACHE_HIT_DYNAMIC_MS = 48;
  const OVERLAY_SNAPSHOT_DOMAINS = Object.freeze([
    'hud',
    'session',
    'inventory',
    'equipment',
    'cards',
    'skills',
    'quests',
    'world',
    'guide',
    'monsterGuide',
    'party',
    'pet',
    'daily',
    'shop',
    'settings',
    'debug'
  ]);

  function createOverlaySnapshotDomainRevisions(source) {
    return OVERLAY_SNAPSHOT_DOMAINS.reduce((revisions, domain) => {
      revisions[domain] = Math.max(0, Math.floor(Number(source && source[domain]) || 0));
      return revisions;
    }, {});
  }

  function normalizeOverlaySnapshotDomains(domains) {
    if (domains === 'all' || domains == null) return OVERLAY_SNAPSHOT_DOMAINS.slice();
    const source = Array.isArray(domains) ? domains : [domains];
    const normalized = source
      .map((domain) => String(domain || '').trim())
      .filter((domain) => OVERLAY_SNAPSHOT_DOMAINS.includes(domain));
    if (!normalized.length || normalized.includes('all')) return OVERLAY_SNAPSHOT_DOMAINS.slice();
    return Array.from(new Set(normalized));
  }

  function getOverlaySnapshotRefreshState(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return {};
    if (!snapshot.__dynamicRefreshAt || typeof snapshot.__dynamicRefreshAt !== 'object') {
      Object.defineProperty(snapshot, '__dynamicRefreshAt', {
        value: {},
        configurable: true,
        enumerable: false,
        writable: true
      });
    }
    return snapshot.__dynamicRefreshAt;
  }

  function markOverlaySnapshotFieldFresh(snapshot, field, nowMs) {
    const state = getOverlaySnapshotRefreshState(snapshot);
    state[field] = Math.max(0, Number(nowMs) || performanceNowMs());
  }

  function isOverlaySnapshotFieldStale(snapshot, field, nowMs, ttlMs) {
    const state = getOverlaySnapshotRefreshState(snapshot);
    const previous = Number(state[field] || 0);
    return !previous || Math.max(0, Number(nowMs) || 0) - previous >= Math.max(0, Number(ttlMs) || 0);
  }

  function refreshOverlaySnapshotCachedField(snapshot, field, nowMs, ttlMs, producer) {
    if (!snapshot || !field || typeof producer !== 'function') return null;
    if (!Object.prototype.hasOwnProperty.call(snapshot, field) ||
      isOverlaySnapshotFieldStale(snapshot, field, nowMs, ttlMs)) {
      snapshot[field] = producer();
      markOverlaySnapshotFieldFresh(snapshot, field, nowMs);
    }
    return snapshot[field];
  }

  function shouldRefreshOverlaySnapshotCacheHitDynamicFields(snapshot, nowMs, currentRevision, cacheHitDynamicMs) {
    if (!snapshot) return false;
    if (Number(snapshot.cacheRevision || 0) !== Number(currentRevision || 0)) return true;
    const state = getOverlaySnapshotRefreshState(snapshot);
    const previous = Number(state.cacheHitDynamicRefresh || 0);
    const threshold = Math.max(0, Number(cacheHitDynamicMs == null ? OVERLAY_SNAPSHOT_CACHE_HIT_DYNAMIC_MS : cacheHitDynamicMs) || 0);
    return !previous || Math.max(0, Number(nowMs || 0) - previous) >= threshold;
  }

  const api = {
    OVERLAY_SNAPSHOT_CACHE_MS,
    OVERLAY_SNAPSHOT_DYNAMIC_FAST_MS,
    OVERLAY_SNAPSHOT_DYNAMIC_STATUS_MS,
    OVERLAY_SNAPSHOT_DYNAMIC_WORLD_MS,
    OVERLAY_SNAPSHOT_CACHE_HIT_DYNAMIC_MS,
    OVERLAY_SNAPSHOT_DOMAINS,
    createOverlaySnapshotDomainRevisions,
    normalizeOverlaySnapshotDomains,
    getOverlaySnapshotRefreshState,
    markOverlaySnapshotFieldFresh,
    isOverlaySnapshotFieldStale,
    refreshOverlaySnapshotCachedField,
    shouldRefreshOverlaySnapshotCacheHitDynamicFields
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.snapshots = Object.assign({}, modules.snapshots || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
