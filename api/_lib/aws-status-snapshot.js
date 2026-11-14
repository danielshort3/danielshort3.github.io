'use strict';

const HISTORY_DAYS = 90;
const HISTORY_HOURS = 72;
const HOUR_MS = 60 * 60_000;
const REGION = 'us-east-2';
const MAX_CHECKS = 3 * 70 + 1;
const STATES = new Set(['operational', 'degraded', 'outage', 'unknown']);
const SOURCES = new Set(['production', 'browser', 'local']);
const CONNECTIONS = new Map([
  ['cognito', 'Cognito sign-in'],
  ['tracker', 'Tracker database'],
  ['short-links', 'Short Links'],
  ['contact', 'Contact email'],
  ['transcribe', 'File Transcriber'],
  ['chatbot', 'Website chatbot'],
  ['markdown', 'SageMaker Markdown']
]);
const DAY_KEY = /^\d{4}-\d{2}-\d{2}$/;
const HOUR_KEY = /^\d{4}-\d{2}-\d{2}T\d{2}:00:00\.000Z$/;
const HOUR_STATES = /^[-uodx]{72}$/;
const DAY_STATES = /^[-uodx]{90}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/;
const isTimestamp = (value) => typeof value === 'string' && ISO_TIMESTAMP.test(value) && Number.isFinite(Date.parse(value));
const issueText = (value) => typeof value === 'string' ? value.trim().slice(0, 240) : '';

function canonicalTime(value, format) {
  if (typeof value !== 'string' || !format.test(value)) return NaN;
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return NaN;
  return new Date(time).toISOString().startsWith(value) ? time : NaN;
}

function sanitizePeriod(period) {
  if (!STATES.has(period?.state) || !isTimestamp(period.from) ||
    !isTimestamp(period.to) || Date.parse(period.from) > Date.parse(period.to)) return null;

  const clean = {
    state: period.state,
    from: period.from,
    to: period.to,
    issue: issueText(period.issue)
  };
  if (isTimestamp(period.observedAt)) clean.observedAt = period.observedAt;
  return clean;
}

function sanitizeDay(day) {
  const period = DAY_KEY.test(day?.date) && sanitizePeriod(day);
  return period ? { date: day.date, ...period } : null;
}

function sanitizeHour(hour) {
  const start = canonicalTime(hour?.start, HOUR_KEY);
  if (!Number.isFinite(start)) return null;
  const period = sanitizePeriod(hour);
  const end = start + HOUR_MS;
  if (!period || Date.parse(period.from) < start || Date.parse(period.to) > end) return null;

  const observedAt = Date.parse(period.observedAt);
  if (observedAt < start || observedAt >= end) delete period.observedAt;
  return { start: hour.start, ...period };
}

function sanitizeCatalogHistory(history, catalog, now) {
  const start = canonicalTime(history?.hourStart, HOUR_KEY);
  if (!Number.isFinite(start) || !Number.isFinite(canonicalTime(history.dayStart, DAY_KEY)) ||
    !Array.isArray(history.observedAt) || history.observedAt.length !== HISTORY_HOURS ||
    !history.results || typeof history.results !== 'object' || Array.isArray(history.results)) return undefined;

  const observedAt = history.observedAt.map((value, index) => {
    const checked = Date.parse(value);
    const hour = start + index * HOUR_MS;
    return isTimestamp(value) && checked >= hour && checked < hour + HOUR_MS && checked <= now ? value : null;
  });
  const results = {};
  for (const { id } of catalog) {
    if (!Object.hasOwn(history.results, id)) continue;
    const item = history.results[id];
    if (typeof item?.hours !== 'string' || !HOUR_STATES.test(item.hours) ||
      typeof item.days !== 'string' || !DAY_STATES.test(item.days)) continue;
    results[id] = {
      hours: [...item.hours].map((state, index) => observedAt[index] ? state : '-').join(''),
      days: item.days
    };
  }
  return { hourStart: history.hourStart, dayStart: history.dayStart, observedAt, results };
}

function sanitizeHistory(periods, limit, key, sanitize) {
  const byKey = new Map();
  if (Array.isArray(periods) && periods.length <= limit) {
    for (const period of periods) {
      const valid = sanitize(period);
      if (valid && !byKey.has(valid[key])) byKey.set(valid[key], valid);
    }
  }
  return [...byKey.values()].sort((left, right) => left[key].localeCompare(right[key]));
}

function sanitizeConnections(connections) {
  const byId = new Map(connections.map((connection) => [connection?.id, connection]));
  return [...CONNECTIONS].map(([id, label]) => {
    const stored = byId.get(id);
    const output = { id, label, days: sanitizeHistory(stored?.days, HISTORY_DAYS, 'date', sanitizeDay) };
    if (Array.isArray(stored?.hours) && stored.hours.length <= HISTORY_HOURS) {
      output.hours = sanitizeHistory(stored.hours, HISTORY_HOURS, 'start', sanitizeHour);
    }
    return output;
  });
}

function rankCheck(check, stale) {
  if (stale) return 0;
  if (check.state === 'unknown') return 1;
  if (check.source !== 'production') return 3;
  if (check.id === 'project:website' && check.method === 'homepage-head') {
    return check.state === 'operational' ? 2 : 4;
  }
  return 5;
}

function projectCatalog(catalog, checks, now) {
  const byId = new Map();
  for (const check of checks) {
    if (!check || !SOURCES.has(check.source) || !STATES.has(check.state) || !isTimestamp(check.lastChecked)) continue;
    const checked = Date.parse(check.lastChecked);
    if (checked > now + 60_000) continue;
    const allowedAge = check.source === 'production' ? 26 * 60 * 60_000 : 7 * 24 * 60 * 60_000;
    const expiry = isTimestamp(check.expiresAt)
      ? Math.min(Date.parse(check.expiresAt), checked + allowedAge)
      : checked + allowedAge;
    const stale = expiry <= now;
    const rank = rankCheck(check, stale);
    const previous = byId.get(check.id);
    if (!previous || rank > previous.rank || (rank === previous.rank && checked > previous.checked)) {
      byId.set(check.id, { check, checked, rank, stale });
    }
  }

  return catalog.map((item) => {
    const result = byId.get(item.id);
    if (!result) return { ...item, state: 'unknown', lastChecked: null, issue: 'Awaiting a check.' };
    const { check, stale } = result;
    return {
      ...item,
      state: stale ? 'unknown' : check.state,
      lastChecked: check.lastChecked,
      source: check.source,
      issue: stale ? 'The last result has expired.' : issueText(check.issue)
    };
  });
}

function sanitizeSnapshot(snapshot, catalog, now = Date.now()) {
  if (snapshot?.ok !== true || snapshot.region !== REGION || snapshot.days !== HISTORY_DAYS ||
    (snapshot.checkedAt !== null && !isTimestamp(snapshot.checkedAt)) ||
    !Array.isArray(snapshot.connections) || snapshot.connections.length > CONNECTIONS.size ||
    !Array.isArray(snapshot.checks) || snapshot.checks.length > MAX_CHECKS) {
    throw new Error('Invalid AWS history snapshot.');
  }
  const catalogHistory = sanitizeCatalogHistory(snapshot.catalogHistory, catalog, now);
  return {
    ok: true,
    region: REGION,
    days: HISTORY_DAYS,
    checkedAt: snapshot.checkedAt,
    connections: sanitizeConnections(snapshot.connections),
    catalog: projectCatalog(catalog, snapshot.checks, now),
    ...(catalogHistory ? { catalogHistory } : {})
  };
}

module.exports = { sanitizeSnapshot };
