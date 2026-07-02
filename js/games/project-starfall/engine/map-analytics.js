(function initProjectStarfallEngineMapAnalytics(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  function getMapAnalyticsData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createMapAnalyticsEntryState(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      visits: Math.max(0, Math.floor(Number(source.visits || 0) || 0)),
      repeatVisits: Math.max(0, Math.floor(Number(source.repeatVisits || 0) || 0)),
      completedVisits: Math.max(0, Math.floor(Number(source.completedVisits || 0) || 0)),
      earlyExits: Math.max(0, Math.floor(Number(source.earlyExits || 0) || 0)),
      kills: Math.max(0, Math.floor(Number(source.kills || 0) || 0)),
      totalSeconds: Math.max(0, Number(source.totalSeconds || 0) || 0),
      lastVisitSeconds: Math.max(0, Number(source.lastVisitSeconds || 0) || 0),
      lastVisitKills: Math.max(0, Math.floor(Number(source.lastVisitKills || 0) || 0)),
      lastEnteredAt: Math.max(0, Number(source.lastEnteredAt || 0) || 0),
      lastExitedAt: Math.max(0, Number(source.lastExitedAt || 0) || 0)
    };
  }

  function createMapAnalyticsState(value, options) {
    const data = getMapAnalyticsData(options);
    const source = value && typeof value === 'object' ? value : {};
    const sourceByMapId = source.byMapId && typeof source.byMapId === 'object'
      ? source.byMapId
      : source;
    const byMapId = {};
    Object.entries(sourceByMapId || {}).forEach(([mapId, entry]) => {
      const id = normalizeId(mapId);
      if (id && getById(data.MAPS || [], id)) byMapId[id] = createMapAnalyticsEntryState(entry);
    });
    const visit = source.currentVisit && typeof source.currentVisit === 'object' ? source.currentVisit : null;
    const visitMapId = normalizeId(visit && visit.mapId);
    return {
      byMapId,
      currentVisit: visitMapId && getById(data.MAPS || [], visitMapId)
        ? {
            mapId: visitMapId,
            previousMapId: normalizeId(visit.previousMapId),
            enteredAt: Math.max(0, Number(visit.enteredAt || 0) || 0),
            kills: Math.max(0, Math.floor(Number(visit.kills || 0) || 0))
          }
        : null
    };
  }

  function shouldTrackMapAnalytics(map) {
    return !!(map && map.id && !map.adminOnly && !map.isTrialInstance);
  }

  function createMapAnalyticsVisitStart(entry, currentVisit, map, options) {
    const settings = options || {};
    if (!shouldTrackMapAnalytics(map)) return { ok: false };
    const nowMs = Number(settings.nowMs || 0);
    const nextEntry = createMapAnalyticsEntryState(entry);
    const existing = currentVisit && typeof currentVisit === 'object' ? currentVisit : null;
    if (existing && existing.mapId === map.id && settings.preserveCurrent) {
      return {
        ok: true,
        entry: nextEntry,
        currentVisit: {
          mapId: map.id,
          previousMapId: normalizeId(existing.previousMapId),
          enteredAt: nowMs,
          kills: Math.max(0, Math.floor(Number(existing.kills || 0) || 0))
        }
      };
    }
    if (nextEntry.visits > 0) nextEntry.repeatVisits += 1;
    nextEntry.visits += 1;
    nextEntry.lastEnteredAt = nowMs;
    return {
      ok: true,
      entry: nextEntry,
      currentVisit: {
        mapId: map.id,
        previousMapId: normalizeId(settings.previousMapId),
        enteredAt: nowMs,
        kills: 0
      }
    };
  }

  function createMapAnalyticsVisitFinish(entry, currentVisit, map, nextMapId, options) {
    const settings = options || {};
    const visit = currentVisit && typeof currentVisit === 'object' ? currentVisit : null;
    if (!visit || !visit.mapId) return { ok: false };
    const nextEntry = createMapAnalyticsEntryState(entry);
    const nowMs = Number(settings.nowMs || 0);
    const elapsed = Math.max(0, (nowMs - Number(visit.enteredAt || nowMs)) / 1000);
    const kills = Math.max(0, Math.floor(Number(visit.kills || 0) || 0));
    nextEntry.completedVisits += 1;
    nextEntry.totalSeconds += elapsed;
    nextEntry.lastVisitSeconds = elapsed;
    nextEntry.lastVisitKills = kills;
    nextEntry.lastExitedAt = nowMs;
    const combatMap = !!(map && !map.safeZone && !map.shopInterior);
    if (combatMap && normalizeId(nextMapId) !== (map && map.id) && elapsed < 45 && kills < 5) nextEntry.earlyExits += 1;
    return {
      ok: true,
      entry: nextEntry,
      currentVisit: null
    };
  }

  function createMapAnalyticsDefeatUpdate(entry, currentVisit, map) {
    if (!shouldTrackMapAnalytics(map)) return { ok: false };
    const visit = currentVisit && currentVisit.mapId === map.id ? currentVisit : null;
    if (!visit) return { ok: false };
    const nextEntry = createMapAnalyticsEntryState(entry);
    nextEntry.kills += 1;
    return {
      ok: true,
      entry: nextEntry,
      currentVisit: {
        mapId: map.id,
        previousMapId: normalizeId(visit.previousMapId),
        enteredAt: Math.max(0, Number(visit.enteredAt || 0) || 0),
        kills: Math.max(0, Number(visit.kills || 0)) + 1
      }
    };
  }

  function createMapAnalyticsSnapshot(map, entry, currentVisit, options) {
    const settings = options || {};
    if (!shouldTrackMapAnalytics(map)) return { active: false, mapId: normalizeId(settings.mapId || settings.currentMapId) };
    const state = createMapAnalyticsEntryState(entry);
    const current = currentVisit && currentVisit.mapId === map.id ? currentVisit : null;
    const nowMs = Number(settings.nowMs || 0);
    const currentVisitSeconds = current ? Math.max(0, (nowMs - Number(current.enteredAt || nowMs)) / 1000) : 0;
    const visits = Math.max(0, Number(state.visits || 0));
    const completedVisits = Math.max(0, Number(state.completedVisits || 0));
    const totalSeconds = Number(state.totalSeconds || 0) + currentVisitSeconds;
    const kills = Math.max(0, Number(state.kills || 0));
    return {
      active: true,
      mapId: map.id,
      mapName: map.name || map.id,
      currentVisitActive: !!current,
      currentVisitSeconds: Math.round(currentVisitSeconds),
      currentVisitKills: Math.max(0, Number(current && current.kills || 0)),
      visits,
      repeatVisits: Math.max(0, Number(state.repeatVisits || 0)),
      repeatVisitationRate: visits > 0 ? clamp(Number(state.repeatVisits || 0) / visits, 0, 1) : 0,
      completedVisits,
      earlyExits: Math.max(0, Number(state.earlyExits || 0)),
      abandonmentRate: completedVisits > 0 ? clamp(Number(state.earlyExits || 0) / completedVisits, 0, 1) : 0,
      kills,
      killsPerVisit: visits > 0 ? kills / visits : 0,
      totalSeconds,
      averageVisitSeconds: visits > 0 ? totalSeconds / visits : 0,
      lastVisitSeconds: Number(state.lastVisitSeconds || 0),
      lastVisitKills: Math.max(0, Number(state.lastVisitKills || 0)),
      lastEnteredAt: Math.max(0, Number(state.lastEnteredAt || 0)),
      lastExitedAt: Math.max(0, Number(state.lastExitedAt || 0))
    };
  }

  const api = {
    createMapAnalyticsEntryState,
    createMapAnalyticsState,
    shouldTrackMapAnalytics,
    createMapAnalyticsVisitStart,
    createMapAnalyticsVisitFinish,
    createMapAnalyticsDefeatUpdate,
    createMapAnalyticsSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.mapAnalytics = Object.assign({}, modules.mapAnalytics || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
