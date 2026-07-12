/*
  Admin API for click history: /api/short-links/clicks/<slug>
  - Requires SHORTLINKS_ADMIN_TOKEN.
  - Requires SHORTLINKS_DDB_CLICKS_TABLE to be configured.
*/
'use strict';

const {
  getLinkWithLegacyFallback,
  listClickHistory
} = require('../../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('../../_lib/short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/short-links\/clicks\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : '';
  } catch {
    return '';
  }
}

function parseLimit(req){
  const raw = req.query && req.query.limit ? String(req.query.limit) : '';
  const value = Number(raw);
  if (!Number.isFinite(value)) return 100;
  return Math.max(1, Math.min(500, Math.floor(value)));
}

function toNonNegativeInteger(value){
  if (value === null || typeof value === 'undefined' || value === '') return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return Math.max(0, Math.floor(parsed));
}

function isHistoricalBaseline(item){
  if (!item || typeof item !== 'object') return false;
  if (item.clickId === '__historical_baseline__') return true;
  if (item.entityType === 'clickBaseline') return true;
  return typeof item.eventType === 'string' && /baseline/i.test(item.eventType);
}

function buildHistorySummary(item, link, detailedEventsReturned){
  const baselineAvailable = Boolean(item);
  const legacyCount = item ? toNonNegativeInteger(item.count) : null;
  const historicalClicks = item
    ? (toNonNegativeInteger(item.historicalClicks) ?? legacyCount ?? 0)
    : null;
  const recordedEventCount = item ? toNonNegativeInteger(item.recordedEventCount) : null;
  const linkAggregate = link ? toNonNegativeInteger(link.clicks) : null;
  const aggregateClicks = item
    ? (toNonNegativeInteger(item.aggregateClicks) ?? (
        historicalClicks !== null && recordedEventCount !== null
          ? historicalClicks + recordedEventCount
          : linkAggregate
      ))
    : linkAggregate;

  return {
    baselineAvailable,
    historicalClicks,
    recordedEventCount,
    aggregateClicks,
    reconciledAt: item && typeof item.reconciledAt === 'string' ? item.reconciledAt : '',
    detailedEventsReturned,
    retentionMode: 'indefinite-target',
    retentionVerified: false
  };
}

module.exports = async (req, res) => {
  const adminToken = getAdminToken();
  if (!adminToken) {
    sendJson(res, 503, { ok: false, error: 'SHORTLINKS_ADMIN_TOKEN is not configured' });
    return;
  }
  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const slug = normalizeSlug(getSlugFromRequest(req));
  if (!slug) {
    sendJson(res, 400, { ok: false, error: 'Invalid slug' });
    return;
  }

  const limit = parseLimit(req);
  let resolvedSlug = slug;
  let link = null;

  try {
    link = await getLinkWithLegacyFallback(slug);
    if (link && typeof link.slug === 'string') resolvedSlug = link.slug;
  } catch (err) {
    if (err && err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
  }

  let items = [];
  try {
    items = await listClickHistory({ slug: resolvedSlug, limit });
  } catch (err) {
    if (err && err.code === 'DDB_CLICKS_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    if (err && err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
    return;
  }

  const baseline = items.find(isHistoricalBaseline) || null;
  const clicks = items.filter(item => !isHistoricalBaseline(item)).slice(0, limit).map(item => ({
    clickId: typeof item.clickId === 'string' ? item.clickId : '',
    clickedAt: typeof item.clickedAt === 'string' ? item.clickedAt : '',
    destination: typeof item.destination === 'string' ? item.destination : '',
    statusCode: Number.isFinite(Number(item.statusCode)) ? Number(item.statusCode) : 0,
    host: typeof item.host === 'string' ? item.host : '',
    path: typeof item.path === 'string' ? item.path : '',
    referer: typeof item.referer === 'string' ? item.referer : '',
    refererHost: typeof item.refererHost === 'string' ? item.refererHost : '',
    userAgent: typeof item.userAgent === 'string' ? item.userAgent : '',
    country: typeof item.country === 'string' ? item.country : '',
    region: typeof item.region === 'string' ? item.region : '',
    city: typeof item.city === 'string' ? item.city : '',
    timezone: typeof item.timezone === 'string' ? item.timezone : '',
    latitude: typeof item.latitude === 'string' ? item.latitude : '',
    longitude: typeof item.longitude === 'string' ? item.longitude : ''
  }));

  sendJson(res, 200, {
    ok: true,
    slug: resolvedSlug,
    limit,
    clicks,
    summary: buildHistorySummary(baseline, link, clicks.length)
  });
};
