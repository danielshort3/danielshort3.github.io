/*
  Admin API for click history, delegated through the stable Short Links catch-all.
*/
'use strict';

const { getLinkWithLegacyFallback, listClicks } = require('./short-links-store');
const {
  authorizeShortLinksAdmin,
  sendJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('./short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && (req.query.slug || req.query['...slug']);
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
  const headerValue = req.headers && req.headers['x-shortlinks-limit'];
  const raw = headerValue
    ? String(headerValue)
    : (req.query && req.query.limit ? String(req.query.limit) : '');
  const value = Number(raw);
  if (!Number.isFinite(value)) return 100;
  return Math.max(1, Math.min(500, Math.floor(value)));
}

module.exports = async (req, res) => {
  const admin = await authorizeShortLinksAdmin(req);
  if (!admin.authorized) {
    sendJson(res, admin.statusCode, { ok: false, error: admin.error });
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

  try {
    const link = await getLinkWithLegacyFallback(slug);
    if (link && typeof link.slug === 'string') resolvedSlug = link.slug;
  } catch (err) {
    if (err && err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
  }

  let items = [];
  try {
    items = await listClicks({ slug: resolvedSlug, limit });
  } catch (err) {
    if (err && (err.code === 'DDB_CLICKS_ENV_MISSING' || err.code === 'DDB_ENV_MISSING')) {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
    return;
  }

  const clicks = items.map(item => ({
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

  sendJson(res, 200, { ok: true, slug: resolvedSlug, limit, clicks });
};
