/*
  Admin API for managing short links.
  Requires SHORTLINKS_ADMIN_TOKEN and DynamoDB env vars.
*/
'use strict';

const { serializeLink, normalizeTags, normalizeQrDesign, buildAnalyticsReport } = require('../_lib/short-links-management');

const {
  isSlugConflictError,
  listLinks,
  listLinksPage,
  listAnalyticsEvents,
  upsertLink
} = require('../_lib/short-links-store');
const {
  DEFAULT_RANDOM_LENGTH,
  MAX_RANDOM_LENGTH,
  MIN_RANDOM_LENGTH,
  generateRandomSlug,
  authorizeAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeSlugLower,
  normalizeRandomLength,
  normalizeDestination
} = require('../_lib/short-links');

const RANDOM_SLUG_RETRY_LIMIT = 40;


function buildMetadata(body){
  return {
    label: body?.label,
    tags: typeof body?.tags === 'undefined' ? undefined : normalizeTags(body.tags),
    qrDesign: typeof body?.qrDesign === 'undefined' ? undefined : normalizeQrDesign(body.qrDesign),
    templateId: body?.templateId,
    templateTitle: body?.templateTitle,
    batchId: body?.batchId,
    batchTitle: body?.batchTitle,
    contextType: body?.contextType,
    contextEntryId: body?.contextEntryId,
    contextCompany: body?.contextCompany,
    contextTitle: body?.contextTitle
  };
}

function buildLowerSlugMap(items){
  const map = new Map();
  (Array.isArray(items) ? items : []).forEach((item) => {
    const lower = normalizeSlugLower(item && item.slug);
    if (!lower || map.has(lower)) return;
    map.set(lower, item);
  });
  return map;
}

function getSlugConflict(lowerSlugMap, slug){
  const target = normalizeSlugLower(slug);
  if (!target || !lowerSlugMap || typeof lowerSlugMap.get !== 'function') return null;
  return lowerSlugMap.get(target) || null;
}

async function resolveRequestedSlug(body, lowerSlugMap){
  const slugMode = typeof body?.slugMode === 'string' ? body.slugMode.trim().toLowerCase() :
    (body?.intent === 'create' && !body?.slug ? 'random' : '');
  const randomLength = normalizeRandomLength(body?.randomLength, DEFAULT_RANDOM_LENGTH);
  const manualSlug = normalizeSlug(body?.slug);

  if (slugMode === 'random') {
    for (let i = 0; i < RANDOM_SLUG_RETRY_LIMIT; i += 1) {
      const candidate = generateRandomSlug(randomLength);
      const conflict = getSlugConflict(lowerSlugMap, candidate);
      if (!conflict) {
        if (lowerSlugMap && typeof lowerSlugMap.set === 'function') {
          lowerSlugMap.set(normalizeSlugLower(candidate), { slug: candidate });
        }
        return { slug: candidate, randomLength, generated: true };
      }
    }
    const err = new Error('Unable to generate a unique short code right now');
    err.statusCode = 409;
    throw err;
  }

  if (!manualSlug) {
    const err = new Error('Invalid slug (use letters/numbers/-/_ and / for nesting)');
    err.statusCode = 400;
    throw err;
  }

  const conflict = getSlugConflict(lowerSlugMap, manualSlug);
  if (conflict && (body?.intent === 'create' || String(conflict.slug || '') !== manualSlug)) {
    const err = new Error(`Slug conflicts with existing link "${conflict.slug}"`);
    err.statusCode = 409;
    throw err;
  }

  return { slug: manualSlug, randomLength, generated: false };
}

function getRequestUrl(req){
  try {
    return new URL(req.url, 'https://www.danielshort.me');
  } catch {
    return new URL('https://www.danielshort.me/');
  }
}

function getQueryValue(params, key){
  if (!params || !params.has(key)) return '';
  return String(params.get(key) || '').trim();
}

function isLinkExpired(link){
  const expiresAt = Number.isFinite(Number(link && link.expiresAt)) ? Number(link.expiresAt) : 0;
  return !!expiresAt && expiresAt * 1000 <= Date.now();
}

function matchesStatus(link, status){
  const normalized = String(status || '').trim().toLowerCase();
  if (!normalized || normalized === 'all') return true;
  if (normalized === 'disabled') return !!(link && link.disabled);
  if (normalized === 'expired') return isLinkExpired(link);
  if (normalized === 'temporary') return !(link && link.permanent);
  if (normalized === 'permanent') return !!(link && link.permanent);
  if (normalized === 'active') return !!link && !link.disabled && !isLinkExpired(link);
  return true;
}

function matchesQuery(link, query){
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) return true;
  const haystack = [
    link && link.slug,
    link && link.destination,
    link && link.label,
    link && link.templateTitle,
    link && link.batchTitle,
    link && link.contextCompany,
    link && link.contextTitle
  ].join(' ').toLowerCase();
  return haystack.includes(needle);
}

function getComparableValue(link, key){
  switch (key) {
    case 'clicks':
      return Number.isFinite(Number(link && link.clicks)) ? Number(link.clicks) : 0;
    case 'updated':
      return Date.parse(link && link.updatedAt) || 0;
    case 'created':
      return Date.parse(link && link.createdAt) || 0;
    case 'expires':
      return Number.isFinite(Number(link && link.expiresAt)) ? Number(link.expiresAt) : 0;
    case 'destination':
      return String(link && link.destination || '').toLowerCase();
    case 'slug':
    default:
      return String(link && link.slug || '').toLowerCase();
  }
}

function sortLinks(links, rawSort){
  const requested = String(rawSort || 'slug').trim().toLowerCase();
  const descending = requested.startsWith('-');
  const key = descending ? requested.slice(1) : requested;
  const allowed = new Set(['slug', 'destination', 'clicks', 'updated', 'created', 'expires']);
  const sortKey = allowed.has(key) ? key : 'slug';

  return links.slice().sort((a, b) => {
    const av = getComparableValue(a, sortKey);
    const bv = getComparableValue(b, sortKey);
    let result = 0;
    if (typeof av === 'number' && typeof bv === 'number') result = av - bv;
    else result = String(av).localeCompare(String(bv), undefined, { sensitivity: 'base' });
    if (!result) result = String(a.slug || '').localeCompare(String(b.slug || ''));
    return descending ? -result : result;
  });
}

function normalizeLimit(value){
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(1, Math.min(500, Math.floor(numeric)));
}

function normalizeCursor(value){
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.floor(numeric));
}

function hasListQuery(params){
  return ['q', 'status', 'sort', 'limit', 'cursor'].some(key => params && params.has(key));
}

function applyListQuery(links, params){
  const q = getQueryValue(params, 'q');
  const status = getQueryValue(params, 'status');
  const sort = getQueryValue(params, 'sort') || 'slug';
  const limit = normalizeLimit(getQueryValue(params, 'limit'));
  const cursor = normalizeCursor(getQueryValue(params, 'cursor'));

  const filtered = sortLinks(
    (Array.isArray(links) ? links : []).filter(link => matchesQuery(link, q) && matchesStatus(link, status)),
    sort
  );

  if (!limit) {
    return {
      links: filtered,
      pagination: {
        total: filtered.length,
        limit: 0,
        cursor: 0,
        nextCursor: ''
      }
    };
  }

  const page = filtered.slice(cursor, cursor + limit);
  const nextCursor = cursor + limit < filtered.length ? String(cursor + limit) : '';
  return {
    links: page,
    pagination: {
      total: filtered.length,
      limit,
      cursor,
      nextCursor
    }
  };
}

async function handleAnalytics(req, res, params){
  const days = getQueryValue(params, 'days') || '30';
  if (!['7', '30', '90', 'all'].includes(days)) {
    sendJson(res, 400, { ok: false, error: 'Choose 7, 30, 90, or all days' });
    return;
  }
  const requestedSlug = getQueryValue(params, 'slug');
  if (requestedSlug && !normalizeSlug(requestedSlug)) {
    sendJson(res, 400, { ok: false, error: 'Invalid link ending' });
    return;
  }
  try {
    let links = await listLinks();
    if (requestedSlug) links = links.filter(link => normalizeSlugLower(link.slug) === normalizeSlugLower(requestedSlug));
    if (requestedSlug && !links.length) {
      sendJson(res, 404, { ok: false, error: 'Link not found' });
      return;
    }
    const slug = requestedSlug ? links[0].slug : '';
    const history = await listAnalyticsEvents({ slug });
    sendJson(res, 200, { ok: true, analytics: buildAnalyticsReport({ links, ...history, slug, days }) });
  } catch (err) {
    sendJson(res, err.code === 'DDB_ENV_MISSING' ? 503 : 502, { ok: false, error: 'Analytics history is unavailable' });
  }
}

async function handler(req, res){
  if (!await authorizeAdminRequest(req, res)) return;

  if (req.method === 'GET') {
    const params = getRequestUrl(req).searchParams;
    if (getQueryValue(params, 'view') === 'analytics') {
      await handleAnalytics(req, res, params);
      return;
    }
    if (getQueryValue(params, 'pageMode').toLowerCase() === 'storage') {
      const unsupportedParam = ['q', 'status', 'sort'].find(key => getQueryValue(params, key));
      if (unsupportedParam) {
        sendJson(res, 400, {
          ok: false,
          error: `Storage pagination does not support ${unsupportedParam}; omit pageMode to use filtered sorting.`
        });
        return;
      }
      const limit = normalizeLimit(getQueryValue(params, 'limit')) || 100;
      try {
        const page = await listLinksPage({
          limit,
          cursor: getQueryValue(params, 'cursor')
        });
        const links = page.items
          .map(item => serializeLink(item, normalizeSlug(item?.slug), typeof item?.updatedAt === 'string' ? item.updatedAt : ''))
          .filter(link => link.slug && link.destination);
        sendJson(res, 200, {
          ok: true,
          basePath: 'go',
          links,
          pagination: {
            mode: 'storage',
            total: null,
            limit,
            cursor: getQueryValue(params, 'cursor'),
            nextCursor: page.nextCursor
          }
        });
        return;
      } catch (err) {
        if (err.code === 'DDB_ENV_MISSING') {
          sendJson(res, 503, { ok: false, error: err.message });
          return;
        }
        if (err.code === 'INVALID_CURSOR') {
          sendJson(res, 400, { ok: false, error: err.message });
          return;
        }
        sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
        return;
      }
    }

    let items = [];
    try {
      items = await listLinks();
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    const links = items
      .map(item => serializeLink(item, normalizeSlug(item?.slug), typeof item?.updatedAt === 'string' ? item.updatedAt : ''))
      .filter(link => link.slug && link.destination)
      .sort((a, b) => a.slug.localeCompare(b.slug, undefined, { sensitivity: 'base' }) || a.slug.localeCompare(b.slug));

    if (hasListQuery(params)) {
      const result = applyListQuery(links, params);
      sendJson(res, 200, {
        ok: true,
        basePath: 'go',
        links: result.links,
        pagination: result.pagination
      });
      return;
    }

    sendJson(res, 200, { ok: true, basePath: 'go', links });
    return;
  }

  if (req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
      if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('Invalid body');
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    let requestedSlug;
    let lowerSlugMap;
    try {
      const existingLinks = await listLinks();
      lowerSlugMap = buildLowerSlugMap(existingLinks);
      requestedSlug = await resolveRequestedSlug(body || {}, lowerSlugMap);
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, err.statusCode || 400, { ok: false, error: err.message || 'Invalid slug' });
      return;
    }

    let slug = requestedSlug.slug;
    let metadata;
    try {
      metadata = buildMetadata(body);
    } catch (err) {
      sendJson(res, 400, { ok: false, error: err.message });
      return;
    }
    const destination = normalizeDestination(body.destination, { absolutizeInternalPath: true });
    const permanent = !!body.permanent;
    const hasExpiresAt = !!(body && Object.prototype.hasOwnProperty.call(body, 'expiresAt'));
    let expiresAt = 0;

    if (!destination) {
      sendJson(res, 400, { ok: false, error: 'Invalid destination (must start with / or http(s)://)' });
      return;
    }

    if (hasExpiresAt) {
      const numericExpiresAt = Number(body.expiresAt);
      if (!Number.isFinite(numericExpiresAt) || numericExpiresAt < 0) {
        sendJson(res, 400, { ok: false, error: 'Invalid expiresAt (expected a Unix timestamp in seconds)' });
        return;
      }
      expiresAt = Math.floor(numericExpiresAt);
      const nowSeconds = Math.floor(Date.now() / 1000);
      if (expiresAt && expiresAt <= nowSeconds) {
        sendJson(res, 400, { ok: false, error: 'Invalid expiresAt (must be in the future)' });
        return;
      }
      if (permanent && expiresAt) {
        sendJson(res, 400, { ok: false, error: 'Permanent links cannot have expiresAt' });
        return;
      }
    }

    const now = new Date().toISOString();

    let record = null;
    const writeAttempts = requestedSlug.generated ? RANDOM_SLUG_RETRY_LIMIT : 1;
    for (let attempt = 0; attempt < writeAttempts && !record; attempt += 1) {
      try {
        record = await upsertLink({
          slug,
          destination,
          permanent,
          expiresAt: permanent ? 0 : (hasExpiresAt ? expiresAt : undefined),
          updatedAt: now,
          metadata,
          createOnly: body.intent === 'create' || requestedSlug.generated
        });
      } catch (err) {
        if (err.code === 'DDB_ENV_MISSING') {
          sendJson(res, 503, { ok: false, error: err.message });
          return;
        }
        if (isSlugConflictError(err)) {
          if (requestedSlug.generated && attempt + 1 < writeAttempts) {
            try {
              requestedSlug = await resolveRequestedSlug(body || {}, lowerSlugMap);
              slug = requestedSlug.slug;
              continue;
            } catch (retryErr) {
              sendJson(res, retryErr.statusCode || 409, {
                ok: false,
                error: retryErr.message || 'Unable to generate a unique short code right now'
              });
              return;
            }
          }
          sendJson(res, 409, { ok: false, error: 'Slug conflicts with an existing link' });
          return;
        }
        sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
        return;
      }
    }

    if (!record) {
      sendJson(res, 409, { ok: false, error: 'Unable to generate a unique short code right now' });
      return;
    }

    sendJson(res, 200, {
      ok: true,
      link: serializeLink(record, slug, now),
      generated: requestedSlug.generated,
      randomLength: requestedSlug.generated ? requestedSlug.randomLength : 0,
      randomLengthRange: { min: MIN_RANDOM_LENGTH, max: MAX_RANDOM_LENGTH }
    });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
}

module.exports = handler;
module.exports._internal = {
  applyListQuery,
  buildLowerSlugMap,
  resolveRequestedSlug,
  serializeLink
};
