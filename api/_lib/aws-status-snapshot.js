'use strict';

const HISTORY_DAYS = 90;
const HISTORY_HOURS = 72;
const HOUR_MS = 60 * 60_000;
const HOURLY_GRACE_MS = 20 * 60_000;
const CYCLE_MS = 6 * HOUR_MS;
const REGION = 'us-east-2';
const MAX_CHECKS = 3 * 70 + 1;
const STATES = new Set(['operational', 'degraded', 'outage', 'unknown']);
const CANARY_STATES = {
  passed: 'operational', failed: 'degraded', mismatch: 'degraded', timeout: 'degraded',
  unavailable: 'unknown', running: 'unknown'
};
const SOURCES = new Set(['production', 'browser', 'local']);
const CONNECTIONS = new Map([
  ['cognito', 'Cognito sign-in'],
  ['tracker', 'Tracker database'],
  ['short-links', 'Short Links'],
  ['contact', 'Contact email'],
  ['transcribe', 'File Transcriber (AWS)'],
  ['chatbot', 'Website chatbot'],
  ['markdown', 'SageMaker Markdown']
]);
const HOME_GPU_ID = 'home-gpu';
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

function jobCounts(value) {
  const { completed, failed, inProgress } = value || {};
  if (![completed, failed, inProgress].every((count) => Number.isSafeInteger(count) && count >= 0)) return null;
  return { completed, failed, inProgress };
}

function sanitizeTranscription(period) {
  const counts = jobCounts(period.transcription);
  if (!counts) return null;
  const { failed, completed } = counts;
  const state = failed ? 'degraded' : completed ? 'operational' : 'unknown';
  return period.state === state ? counts : null;
}

function sanitizePeriod(period, includeTranscription) {
  if (!STATES.has(period?.state) || !isTimestamp(period.from) ||
    !isTimestamp(period.to) || Date.parse(period.from) > Date.parse(period.to)) return null;

  const clean = {
    state: period.state,
    from: period.from,
    to: period.to,
    issue: issueText(period.issue)
  };
  if (isTimestamp(period.observedAt)) clean.observedAt = period.observedAt;
  if (includeTranscription && Object.hasOwn(period, 'canary')) {
    if (typeof period.canary !== 'string' || !Object.hasOwn(CANARY_STATES, period.canary) ||
      period.state !== CANARY_STATES[period.canary]) {
      return { ...clean, state: 'unknown', issue: 'The transcription test result could not be verified.' };
    }
    clean.canary = period.canary;
    const activity = jobCounts(period.previousSixHourActivity);
    if (activity) clean.previousSixHourActivity = activity;
  } else if (includeTranscription) {
    const transcription = sanitizeTranscription(period);
    if (transcription) clean.transcription = transcription;
  }
  return clean;
}

function sanitizeDay(day, includeTranscription) {
  const period = DAY_KEY.test(day?.date) && sanitizePeriod(day, includeTranscription);
  return period ? { date: day.date, ...period } : null;
}

function sanitizeHour(hour, includeTranscription, cadenceStart, now) {
  const start = canonicalTime(hour?.start, HOUR_KEY);
  if (!Number.isFinite(start)) return null;
  const period = sanitizePeriod(hour, includeTranscription);
  const cycle = start >= cadenceStart;
  if (cycle && start % CYCLE_MS !== 0) return null;
  const end = start + (cycle ? CYCLE_MS : HOUR_MS);
  if (!period || Date.parse(period.from) < start || Date.parse(period.to) > end) return null;

  const observedAt = Date.parse(period.observedAt);
  if (cycle && (Date.parse(period.from) > now || observedAt > now)) return null;
  if (observedAt < start || observedAt >= end || observedAt > now) delete period.observedAt;
  return { start: hour.start, ...period, ...(hour.complete === true ? { complete: true } : {}) };
}

function sanitizeCatalogDetails(codes, states) {
  if (typeof codes !== 'string' || codes.length !== states.length) return '-'.repeat(states.length);
  return [...states].map((state, index) => {
    const reason = codes[index];
    return (state === 'd' && /^[rtme]$/.test(reason)) || (state === 'u' && reason === 'n') ? reason : '-';
  }).join('');
}

function sanitizeCatalogHistory(history, catalog, now, cadenceStart) {
  const start = canonicalTime(history?.hourStart, HOUR_KEY);
  if (!Number.isFinite(start) || !Number.isFinite(canonicalTime(history.dayStart, DAY_KEY)) ||
    !Array.isArray(history.observedAt) || history.observedAt.length !== HISTORY_HOURS ||
    !history.results || typeof history.results !== 'object' || Array.isArray(history.results)) return undefined;

  const observedAt = history.observedAt.map((value, index) => {
    const checked = Date.parse(value);
    const hour = start + index * HOUR_MS;
    const cycle = hour >= cadenceStart;
    if (cycle && hour % CYCLE_MS !== 0) return null;
    return isTimestamp(value) && checked >= hour && checked < hour + (cycle ? CYCLE_MS : HOUR_MS) && checked <= now ? value : null;
  });
  const results = {};
  for (const { id } of catalog) {
    if (!Object.hasOwn(history.results, id)) continue;
    const item = history.results[id];
    if (typeof item?.hours !== 'string' || !HOUR_STATES.test(item.hours) ||
      typeof item.days !== 'string' || !DAY_STATES.test(item.days)) continue;
    const hours = [...item.hours].map((state, index) => observedAt[index] ? state : '-').join('');
    const days = item.days;
    const detailHours = sanitizeCatalogDetails(item.details?.hours, hours);
    const detailDays = sanitizeCatalogDetails(item.details?.days, days);
    const details = /[rtmen]/.test(detailHours + detailDays) ? { hours: detailHours, days: detailDays } : null;
    results[id] = { hours, days, ...(details ? { details } : {}) };
  }
  return { hourStart: history.hourStart, dayStart: history.dayStart, observedAt, results };
}

function sanitizeHistory(periods, limit, key, sanitize, includeTranscription) {
  const byKey = new Map();
  if (Array.isArray(periods) && periods.length <= limit) {
    for (const period of periods) {
      const valid = sanitize(period, includeTranscription);
      if (valid && !byKey.has(valid[key])) byKey.set(valid[key], valid);
    }
  }
  return [...byKey.values()].sort((left, right) => left[key].localeCompare(right[key]));
}

function sanitizeConnection(stored, id, label, cadenceStart, now) {
  const includeTranscription = id === 'transcribe';
  const days = sanitizeHistory(stored?.days, HISTORY_DAYS, 'date', sanitizeDay, includeTranscription);
  const output = { id, label, days };
  if (Array.isArray(stored?.hours) && stored.hours.length <= HISTORY_HOURS) {
    output.hours = sanitizeHistory(stored.hours, HISTORY_HOURS, 'start',
      (hour) => sanitizeHour(hour, includeTranscription, cadenceStart, now));
  }
  return output;
}

function latestPublishedHour(connections, history, catalog, now, cadenceStart) {
  const currentHour = Math.floor(now / HOUR_MS) * HOUR_MS;
  const cycle = currentHour >= cadenceStart;
  const lastHour = cycle ? Math.floor(now / CYCLE_MS) * CYCLE_MS : currentHour - HOUR_MS;
  const start = new Date(lastHour).toISOString();
  if (now >= lastHour + HOUR_MS + HOURLY_GRACE_MS) return start;

  const index = (lastHour - Date.parse(history?.hourStart)) / HOUR_MS;
  const catalogFinished = history?.observedAt[index] && catalog.every(({ id }) => {
    const state = history.results[id]?.hours[index];
    return state && state !== '-';
  });
  const connectionsFinished = connections.every(({ hours }) =>
    hours?.some((hour) => hour.start === start && hour.complete));
  const previous = lastHour - (cycle && lastHour > cadenceStart ? CYCLE_MS : HOUR_MS);
  return catalogFinished && connectionsFinished ? start : new Date(previous).toISOString();
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
  const firstCycle = canonicalTime(snapshot.cadenceStart, HOUR_KEY);
  const cadenceStart = snapshot.cadenceHours === 6 && firstCycle % CYCLE_MS === 0 ? firstCycle : Infinity;
  const catalogHistory = sanitizeCatalogHistory(snapshot.catalogHistory, catalog, now, cadenceStart);
  const byId = new Map(snapshot.connections.map((connection) => [connection?.id, connection]));
  const connections = [...CONNECTIONS].map(([id, label]) => sanitizeConnection(byId.get(id), id, label, cadenceStart, now));
  const homeGpu = snapshot.homeGpu?.id === HOME_GPU_ID
    ? sanitizeConnection(snapshot.homeGpu, HOME_GPU_ID, 'Home GPU', cadenceStart, now) : null;
  const monitored = homeGpu ? [...connections, homeGpu] : connections;
  return {
    ok: true,
    region: REGION,
    days: HISTORY_DAYS,
    checkedAt: snapshot.checkedAt,
    ...(Number.isFinite(cadenceStart) ? { cadenceHours: 6, cadenceStart: snapshot.cadenceStart } : {}),
    hourlyReadyThrough: latestPublishedHour(monitored, catalogHistory, catalog, now, cadenceStart),
    connections,
    ...(homeGpu ? { homeGpu } : {}),
    catalog: projectCatalog(catalog, snapshot.checks, now),
    ...(catalogHistory ? { catalogHistory } : {})
  };
}

module.exports = { sanitizeSnapshot };
