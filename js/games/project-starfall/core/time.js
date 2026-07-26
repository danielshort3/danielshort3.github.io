(function initProjectStarfallCoreTime(global) {
  'use strict';

  function nowMs() {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function nowSeconds() {
    return nowMs() / 1000;
  }

  function wallNowMs() {
    return Date.now();
  }

  function wallNowSeconds() {
    return wallNowMs() / 1000;
  }

  const DAILY_LOGIN_DAY_MS = 24 * 60 * 60 * 1000;

  function padDailyDatePart(value) {
    return String(Math.max(0, Math.floor(Number(value || 0) || 0))).padStart(2, '0');
  }

  function getDailyLoginDateKey(nowMs) {
    const date = new Date(Number.isFinite(Number(nowMs)) ? Number(nowMs) : Date.now());
    return `${date.getFullYear()}-${padDailyDatePart(date.getMonth() + 1)}-${padDailyDatePart(date.getDate())}`;
  }

  function parseDailyLoginDateKey(dateKey) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(dateKey || ''));
    if (!match) return 0;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!year || month < 1 || month > 12 || day < 1 || day > 31) return 0;
    return new Date(year, month - 1, day).getTime();
  }

  function shiftDailyLoginDateKey(dateKey, deltaDays) {
    const timestamp = parseDailyLoginDateKey(dateKey);
    if (!timestamp) return '';
    return getDailyLoginDateKey(timestamp + Math.floor(Number(deltaDays || 0) || 0) * DAILY_LOGIN_DAY_MS);
  }

  function getDailyLoginDayDistance(fromDateKey, toDateKey) {
    const fromMs = parseDailyLoginDateKey(fromDateKey);
    const toMs = parseDailyLoginDateKey(toDateKey);
    if (!fromMs || !toMs) return 0;
    return Math.round((toMs - fromMs) / DAILY_LOGIN_DAY_MS);
  }

  const WEEKLY_ROUTE_DAY_MS = 24 * 60 * 60 * 1000;
  const WEEKLY_ROUTE_WEEK_MS = 7 * WEEKLY_ROUTE_DAY_MS;

  function getWeeklyRouteWeekStartMs(nowMs) {
    const timestamp = Number.isFinite(Number(nowMs)) ? Number(nowMs) : Date.now();
    const date = new Date(timestamp);
    const utcDayStart = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
    const daysSinceMonday = (date.getUTCDay() + 6) % 7;
    return utcDayStart - daysSinceMonday * WEEKLY_ROUTE_DAY_MS;
  }

  function getWeeklyRouteWeekKey(nowMs) {
    return new Date(getWeeklyRouteWeekStartMs(nowMs)).toISOString().slice(0, 10);
  }

  function getWeeklyRouteResetAt(nowMs) {
    return getWeeklyRouteWeekStartMs(nowMs) + WEEKLY_ROUTE_WEEK_MS;
  }

  const api = {
    nowMs,
    nowSeconds,
    wallNowMs,
    wallNowSeconds,
    DAILY_LOGIN_DAY_MS,
    padDailyDatePart,
    getDailyLoginDateKey,
    parseDailyLoginDateKey,
    shiftDailyLoginDateKey,
    getDailyLoginDayDistance,
    WEEKLY_ROUTE_DAY_MS,
    WEEKLY_ROUTE_WEEK_MS,
    getWeeklyRouteWeekStartMs,
    getWeeklyRouteWeekKey,
    getWeeklyRouteResetAt
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
