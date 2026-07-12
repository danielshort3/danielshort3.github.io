'use strict';

const crypto = require('crypto');
const { sendJson, readJson, getBearerToken, clampLimit } = require('../_lib/tools-api');
const { runReport, totalTimeoutMs } = require('../_lib/ga4-data-api');

const GA4_RESPONSE_CACHE = new Map();
const GA4_RATE_WINDOWS = new Map();
let ga4ResponseCacheBytes = 0;
const DEFAULT_MAX_DATE_RANGE_DAYS = 366;
const DEFAULT_CACHE_TTL_SECONDS = 60;
const DEFAULT_CACHE_MAX_ENTRIES = 24;
const DEFAULT_CACHE_MAX_BYTES = 2 * 1024 * 1024;
const DEFAULT_CACHE_TOTAL_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_RESPONSE_BYTES = 3 * 1024 * 1024;
const HARD_MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const DEFAULT_RATE_LIMIT_PER_MINUTE = 30;

function positiveInteger(value, fallback, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.min(parsed, max);
}

function timingSafeTokenEqual(provided, expected) {
  const providedBuffer = Buffer.from(String(provided || ''), 'utf8');
  const expectedBuffer = Buffer.from(String(expected || ''), 'utf8');
  if (!providedBuffer.length || providedBuffer.length !== expectedBuffer.length) return false;
  try {
    return crypto.timingSafeEqual(providedBuffer, expectedBuffer);
  } catch {
    return false;
  }
}

function getClientKey(req) {
  const forwarded = String(req?.headers?.['x-forwarded-for'] || '').split(',')[0].trim();
  const direct = String(req?.socket?.remoteAddress || '').trim();
  return crypto.createHash('sha256').update(forwarded || direct || 'unknown').digest('hex').slice(0, 24);
}

function consumeRateLimit(req) {
  const now = Date.now();
  const windowMs = 60_000;
  const limit = positiveInteger(process.env.GA4_RATE_LIMIT_PER_MINUTE, DEFAULT_RATE_LIMIT_PER_MINUTE, 300);
  const key = getClientKey(req);
  const current = GA4_RATE_WINDOWS.get(key);
  const entry = !current || current.resetAt <= now ? { count: 0, resetAt: now + windowMs } : current;
  entry.count += 1;
  GA4_RATE_WINDOWS.set(key, entry);

  if (GA4_RATE_WINDOWS.size > 500) {
    for (const [storedKey, stored] of GA4_RATE_WINDOWS) {
      if (stored.resetAt <= now) GA4_RATE_WINDOWS.delete(storedKey);
    }
  }
  while (GA4_RATE_WINDOWS.size > 5_000) {
    const oldestKey = GA4_RATE_WINDOWS.keys().next().value;
    if (!oldestKey) break;
    GA4_RATE_WINDOWS.delete(oldestKey);
  }

  return {
    allowed: entry.count <= limit,
    limit,
    remaining: Math.max(0, limit - entry.count),
    retryAfterSeconds: Math.max(1, Math.ceil((entry.resetAt - now) / 1000))
  };
}

function getAllowedPropertyIds() {
  const configured = String(process.env.GA4_ALLOWED_PROPERTY_IDS || '')
    .split(/[\s,]+/)
    .map(normalizePropertyId)
    .filter(Boolean);
  const defaultProperty = normalizePropertyId(process.env.GA4_PROPERTY_ID);
  if (defaultProperty) configured.push(defaultProperty);
  return new Set(configured);
}

function parseIsoDate(value) {
  const normalized = normalizeIsoDate(value);
  if (!normalized) return null;
  const [year, month, day] = normalized.split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) return null;
  return date;
}

function validateDateRange(startDate, endDate) {
  const start = parseIsoDate(startDate);
  const end = parseIsoDate(endDate);
  if (!start || !end) return { ok: false, error: 'Invalid startDate/endDate (expected real YYYY-MM-DD dates).' };
  if (start > end) return { ok: false, error: 'startDate must be on or before endDate.' };
  const today = new Date();
  const todayUtc = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());
  if (end.getTime() > todayUtc) return { ok: false, error: 'endDate cannot be in the future.' };
  const spanDays = Math.floor((end.getTime() - start.getTime()) / 86_400_000) + 1;
  const maxDays = positiveInteger(process.env.GA4_MAX_DATE_RANGE_DAYS, DEFAULT_MAX_DATE_RANGE_DAYS, 730);
  if (spanDays > maxDays) return { ok: false, error: `Date range cannot exceed ${maxDays} days.` };
  return { ok: true, spanDays };
}

function buildResponseCacheKey(body, context) {
  const cacheInput = JSON.stringify({ body: body || {}, ...context });
  return crypto.createHash('sha256').update(cacheInput).digest('hex');
}

function getCachedResponse(cacheKey) {
  const cached = GA4_RESPONSE_CACHE.get(cacheKey);
  if (!cached) return null;
  if (cached.expiresAt <= Date.now()) {
    GA4_RESPONSE_CACHE.delete(cacheKey);
    ga4ResponseCacheBytes = Math.max(0, ga4ResponseCacheBytes - (Number(cached.bytes) || 0));
    return null;
  }
  return cached.payload;
}

function setCachedResponse(cacheKey, payload) {
  const serialized = JSON.stringify(payload);
  const bytes = Buffer.byteLength(serialized, 'utf8');
  const maxBytes = positiveInteger(process.env.GA4_CACHE_MAX_BYTES, DEFAULT_CACHE_MAX_BYTES, 5 * 1024 * 1024);
  const totalBytes = positiveInteger(process.env.GA4_CACHE_TOTAL_BYTES, DEFAULT_CACHE_TOTAL_BYTES, 20 * 1024 * 1024);
  if (bytes > maxBytes || bytes > totalBytes) return;
  const ttlSeconds = positiveInteger(process.env.GA4_CACHE_TTL_SECONDS, DEFAULT_CACHE_TTL_SECONDS, 300);
  const maxEntries = positiveInteger(process.env.GA4_CACHE_MAX_ENTRIES, DEFAULT_CACHE_MAX_ENTRIES, 100);
  const existing = GA4_RESPONSE_CACHE.get(cacheKey);
  if (existing) ga4ResponseCacheBytes = Math.max(0, ga4ResponseCacheBytes - (Number(existing.bytes) || 0));
  GA4_RESPONSE_CACHE.delete(cacheKey);
  GA4_RESPONSE_CACHE.set(cacheKey, { payload, bytes, expiresAt: Date.now() + ttlSeconds * 1000 });
  ga4ResponseCacheBytes += bytes;
  while (GA4_RESPONSE_CACHE.size > maxEntries || ga4ResponseCacheBytes > totalBytes) {
    const oldestKey = GA4_RESPONSE_CACHE.keys().next().value;
    if (!oldestKey) break;
    const oldest = GA4_RESPONSE_CACHE.get(oldestKey);
    GA4_RESPONSE_CACHE.delete(oldestKey);
    ga4ResponseCacheBytes = Math.max(0, ga4ResponseCacheBytes - (Number(oldest?.bytes) || 0));
  }
}

function payloadBytes(payload) {
  return Buffer.byteLength(JSON.stringify(payload), 'utf8');
}

function fitResponsePayload(payload) {
  const maxBytes = positiveInteger(
    process.env.GA4_MAX_RESPONSE_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    HARD_MAX_RESPONSE_BYTES
  );
  if (payloadBytes(payload) <= maxBytes) return payload;
  if (!Array.isArray(payload?.rows) || !payload.rows.length) return null;

  let low = 0;
  let high = payload.rows.length;
  let fitted = null;
  while (low <= high) {
    const count = Math.floor((low + high) / 2);
    const candidate = {
      ...payload,
      rows: payload.rows.slice(0, count),
      returnedRows: count,
      truncated: true,
      responseTruncated: true
    };
    if (payloadBytes(candidate) <= maxBytes) {
      fitted = candidate;
      low = count + 1;
    } else {
      high = count - 1;
    }
  }
  return fitted;
}

function sendReportSuccess(res, cacheKey, payload) {
  const boundedPayload = fitResponsePayload(payload);
  if (!boundedPayload) {
    sendJson(res, 502, { ok: false, error: 'The GA4 response exceeded the configured response limit.' });
    return;
  }
  setCachedResponse(cacheKey, boundedPayload);
  res.setHeader('X-GA4-Cache', 'MISS');
  sendJson(res, 200, boundedPayload);
}

const UTM_FIELDS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'utm_id'];
const UTM_DIMENSIONS = {
  utm_source: 'sessionSource',
  utm_medium: 'sessionMedium',
  utm_campaign: 'sessionCampaignName',
  utm_content: 'sessionManualAdContent',
  utm_term: 'sessionManualTerm',
  utm_id: 'sessionCampaignId'
};

const INSIGHTS_BREAKDOWNS = new Set([
  'country',
  'region',
  'city',
  'language',
  'deviceCategory',
  'browser',
  'operatingSystem',
  'platform',
  'userAgeBracket',
  'userGender'
]);

function normalizeAdminToken(value) {
  return String(value || '').trim();
}

function normalizePropertyId(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (!/^\d{1,20}$/.test(raw)) return '';
  return raw;
}

function normalizeIsoDate(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) return '';
  return raw;
}

function normalizeUtmValue(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.length > 200) return raw.slice(0, 200);
  return raw;
}

function normalizeGa4FieldName(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.length > 80) return '';
  if (!/^[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?$/.test(raw)) return '';
  return raw;
}

function normalizeDimensionFilters(value, maxItems = 15) {
  if (!value) return [];
  const list = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
  const cleaned = [];
  const seen = new Set();
  const invalid = [];

  for (const entry of list) {
    if (!entry || typeof entry !== 'object') continue;

    const rawFieldName = String(entry.fieldName || '').trim();
    const fieldName = normalizeGa4FieldName(rawFieldName);
    const rawValue = normalizeUtmValue(entry.value);
    if (rawFieldName && !fieldName) {
      invalid.push(rawFieldName);
      continue;
    }
    if (!fieldName || !rawValue) continue;

    const match = normalizeMatchMode(entry.match || entry.matchType);
    const key = `${fieldName}|${match}|${rawValue}`;
    if (seen.has(key)) continue;
    seen.add(key);
    cleaned.push({ fieldName, match, value: rawValue });
  }

  if (invalid.length) {
    const err = new Error(`Invalid dimension filter field name(s): ${invalid.slice(0, 3).join(', ')}`);
    err.code = 'GA4_INVALID_FILTER_FIELDS';
    throw err;
  }

  if (cleaned.length > maxItems) {
    const err = new Error(`Too many dimension filters (max ${maxItems}).`);
    err.code = 'GA4_TOO_MANY_FILTERS';
    throw err;
  }

  return cleaned;
}

function coerceFieldList(value) {
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    return value.split(/[,\n]/).map((part) => part.trim()).filter(Boolean);
  }
  return [];
}

function normalizeFieldList(value, label, maxItems) {
  const list = [];
  const invalid = [];
  coerceFieldList(value).forEach((item) => {
    const normalized = normalizeGa4FieldName(item);
    if (!normalized) {
      const raw = String(item || '').trim();
      if (raw) invalid.push(raw);
      return;
    }
    if (!list.includes(normalized)) list.push(normalized);
  });

  if (invalid.length) {
    const err = new Error(`Invalid ${label} name(s): ${invalid.slice(0, 3).join(', ')}`);
    err.code = 'GA4_INVALID_FIELDS';
    throw err;
  }
  if (maxItems && list.length > maxItems) {
    const err = new Error(`Too many ${label}s selected (max ${maxItems}).`);
    err.code = 'GA4_TOO_MANY_FIELDS';
    throw err;
  }
  return list;
}

function normalizeKind(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return 'utm-urls';
  if (raw === 'utm' || raw === 'utmurls' || raw === 'utm_urls' || raw === 'utm-urls') return 'utm-urls';
  if (raw === 'insights') return 'insights';
  if (raw === 'explore') return 'explore';
  if (raw === 'ping' || raw === 'check') return 'ping';
  return '';
}

function escapeRegex(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function buildQueryParamRegex(paramName, rawValue, mode = 'exact') {
  const name = String(paramName || '').trim();
  if (!name) return '';
  const raw = String(rawValue || '').trim();
  if (!raw) return '';

  const encoded = encodeURIComponent(raw);
  const escaped = escapeRegex(encoded).replace(/%20/g, '(?:%20|\\+)');
  const m = String(mode || '').trim().toLowerCase();
  if (m === 'contains') {
    return `[?&]${escapeRegex(name)}=[^&#]*${escaped}[^&#]*(?:&|#|$)`;
  }
  return `[?&]${escapeRegex(name)}=${escaped}(?:&|#|$)`;
}

function buildPageLocationFilter(filters) {
  const f = filters && typeof filters === 'object' ? filters : {};
  const expressions = [];

  const addContains = (value) => {
    const v = String(value || '').trim();
    if (!v) return;
    expressions.push({
      filter: {
        fieldName: 'pageLocation',
        stringFilter: {
          matchType: 'CONTAINS',
          value: v,
          caseSensitive: false
        }
      }
    });
  };

  const addRegex = (pattern) => {
    const v = String(pattern || '').trim();
    if (!v) return;
    expressions.push({
      filter: {
        fieldName: 'pageLocation',
        stringFilter: {
          matchType: 'PARTIAL_REGEXP',
          value: v,
          caseSensitive: false
        }
      }
    });
  };

  const catchAll = f.catchAll !== false;
  if (catchAll) addContains('utm_');

  const source = normalizeUtmValue(f.utm_source);
  const medium = normalizeUtmValue(f.utm_medium);
  const campaign = normalizeUtmValue(f.utm_campaign);
  const term = normalizeUtmValue(f.utm_term);
  const content = normalizeUtmValue(f.utm_content);
  const id = normalizeUtmValue(f.utm_id);

  const matchMode = String(f.match || 'exact').trim().toLowerCase() === 'contains' ? 'contains' : 'exact';

  if (source) addRegex(buildQueryParamRegex('utm_source', source, matchMode));
  if (medium) addRegex(buildQueryParamRegex('utm_medium', medium, matchMode));
  if (campaign) addRegex(buildQueryParamRegex('utm_campaign', campaign, matchMode));
  if (term) addRegex(buildQueryParamRegex('utm_term', term, matchMode));
  if (content) addRegex(buildQueryParamRegex('utm_content', content, matchMode));
  if (id) addRegex(buildQueryParamRegex('utm_id', id, matchMode));

  if (!expressions.length) return null;
  if (expressions.length === 1) return expressions[0];
  return { andGroup: { expressions } };
}

function normalizeMatchMode(value) {
  const raw = String(value || '').trim().toLowerCase();
  return raw === 'contains' ? 'contains' : 'exact';
}

function normalizeInsightsBreakdown(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  return INSIGHTS_BREAKDOWNS.has(raw) ? raw : '';
}

function normalizeUtmGroupFields(value) {
  if (!Array.isArray(value)) return [];
  const cleaned = value
    .map((field) => String(field || '').trim())
    .filter((field) => UTM_FIELDS.includes(field));
  return Array.from(new Set(cleaned));
}

function buildTrafficSourceFilter(filters) {
  const f = filters && typeof filters === 'object' ? filters : {};
  const expressions = [];

  const matchMode = normalizeMatchMode(f.match);
  const ga4MatchType = matchMode === 'contains' ? 'CONTAINS' : 'EXACT';

  const add = (fieldName, value) => {
    const v = normalizeUtmValue(value);
    if (!v) return;
    expressions.push({
      filter: {
        fieldName,
        stringFilter: {
          matchType: ga4MatchType,
          value: v,
          caseSensitive: false
        }
      }
    });
  };

  add('sessionSource', f.utm_source);
  add('sessionMedium', f.utm_medium);
  add('sessionCampaignName', f.utm_campaign);
  add('sessionManualAdContent', f.utm_content);
  add('sessionManualTerm', f.utm_term);
  add('sessionCampaignId', f.utm_id);

  if (!expressions.length) return null;
  if (expressions.length === 1) return expressions[0];
  return { andGroup: { expressions } };
}

function buildDimensionFiltersExpression(filters) {
  const list = Array.isArray(filters) ? filters : [];
  const expressions = [];

  list.forEach((entry) => {
    if (!entry || typeof entry !== 'object') return;
    const fieldName = normalizeGa4FieldName(entry.fieldName);
    const rawValue = normalizeUtmValue(entry.value);
    if (!fieldName || !rawValue) return;

    const matchMode = normalizeMatchMode(entry.match || entry.matchType);
    const ga4MatchType = matchMode === 'contains' ? 'CONTAINS' : 'EXACT';
    expressions.push({
      filter: {
        fieldName,
        stringFilter: {
          matchType: ga4MatchType,
          value: rawValue,
          caseSensitive: false
        }
      }
    });
  });

  if (!expressions.length) return null;
  if (expressions.length === 1) return expressions[0];
  return { andGroup: { expressions } };
}

function mergeDimensionFilters(...filters) {
  const expressions = [];
  filters.forEach((filter) => {
    if (filter && typeof filter === 'object') expressions.push(filter);
  });
  if (!expressions.length) return null;
  if (expressions.length === 1) return expressions[0];
  return { andGroup: { expressions } };
}

function loadServiceAccount() {
  const b64 = String(process.env.GA4_SERVICE_ACCOUNT_JSON_B64 || '').trim();
  if (b64) {
    const decoded = Buffer.from(b64, 'base64').toString('utf8');
    return JSON.parse(decoded);
  }

  const raw = String(process.env.GA4_SERVICE_ACCOUNT_JSON || '').trim();
  if (raw) return JSON.parse(raw);

  const err = new Error(
    'Missing GA4 service account credentials. Set GA4_SERVICE_ACCOUNT_JSON_B64 (preferred) or GA4_SERVICE_ACCOUNT_JSON.'
  );
  err.code = 'GA4_CREDS_MISSING';
  throw err;
}

function coerceMetricValue(value) {
  const raw = String(value || '').trim();
  if (!raw) return 0;
  const n = Number(raw);
  if (!Number.isFinite(n)) return 0;
  return n;
}

function flattenRows(rawRows, dimensionNames, metricNames) {
  const rows = Array.isArray(rawRows) ? rawRows : [];
  const dims = Array.isArray(dimensionNames) ? dimensionNames : [];
  const metrics = Array.isArray(metricNames) ? metricNames : [];

  return rows
    .map((row) => {
      const out = {};
      dims.forEach((name, idx) => {
        out[name] = String(row?.dimensionValues?.[idx]?.value || '').trim();
      });
      metrics.forEach((name, idx) => {
        out[name] = coerceMetricValue(row?.metricValues?.[idx]?.value);
      });
      return out;
    })
    .filter((row) => row && typeof row === 'object');
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.statusCode = 405;
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const expectedToken = normalizeAdminToken(process.env.GA4_ADMIN_TOKEN);
  if (!expectedToken) {
    sendJson(res, 503, { ok: false, error: 'GA4_ADMIN_TOKEN is not configured.' });
    return;
  }

  const providedToken = normalizeAdminToken(getBearerToken(req));
  if (!timingSafeTokenEqual(providedToken, expectedToken)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  const rate = consumeRateLimit(req);
  res.setHeader('X-RateLimit-Limit', String(rate.limit));
  res.setHeader('X-RateLimit-Remaining', String(rate.remaining));
  if (!rate.allowed) {
    res.setHeader('Retry-After', String(rate.retryAfterSeconds));
    sendJson(res, 429, { ok: false, error: 'Too many GA4 requests. Retry shortly.' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch (err) {
    sendJson(res, err?.statusCode === 413 ? 413 : 400, { ok: false, error: err?.message || 'Invalid JSON body.' });
    return;
  }

  const kind = normalizeKind(body?.kind);
  if (!kind) {
    sendJson(res, 400, { ok: false, error: 'Invalid report kind.' });
    return;
  }

  const propertyId = normalizePropertyId(body?.propertyId || process.env.GA4_PROPERTY_ID);
  if (!propertyId) {
    sendJson(res, 400, { ok: false, error: 'Missing GA4 propertyId.' });
    return;
  }

  const allowedPropertyIds = getAllowedPropertyIds();
  if (!allowedPropertyIds.size) {
    sendJson(res, 503, { ok: false, error: 'Configure GA4_PROPERTY_ID or GA4_ALLOWED_PROPERTY_IDS.' });
    return;
  }
  if (!allowedPropertyIds.has(propertyId)) {
    sendJson(res, 403, { ok: false, error: 'GA4 property is not allowlisted.' });
    return;
  }

  const startDateInput = normalizeIsoDate(body?.startDate);
  const endDateInput = normalizeIsoDate(body?.endDate);

  const getDateRange = () => {
    if (startDateInput && endDateInput) return { startDate: startDateInput, endDate: endDateInput };
    if (kind !== 'ping') return null;

    const now = new Date();
    const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const start = new Date(end.getTime() - 6 * 24 * 60 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    const iso = (d) => `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
    return { startDate: iso(start), endDate: iso(end) };
  };

  const dateRange = getDateRange();
  if (!dateRange) {
    sendJson(res, 400, { ok: false, error: 'Missing startDate/endDate (expected YYYY-MM-DD).' });
    return;
  }

  const { startDate, endDate } = dateRange;
  const dateValidation = validateDateRange(startDate, endDate);
  if (!dateValidation.ok) {
    sendJson(res, 400, { ok: false, error: dateValidation.error });
    return;
  }

  const cacheKey = buildResponseCacheKey(body, { kind, propertyId, startDate, endDate });
  const cachedPayload = getCachedResponse(cacheKey);
  if (cachedPayload) {
    res.setHeader('X-GA4-Cache', 'HIT');
    sendJson(res, 200, cachedPayload);
    return;
  }

  let serviceAccount;
  try {
    serviceAccount = loadServiceAccount();
  } catch (err) {
    sendJson(res, 503, { ok: false, error: err.message });
    return;
  }
  const upstreamDeadlineAt = Date.now() + totalTimeoutMs();

  if (kind === 'utm-urls') {
    const utmMaxRows = positiveInteger(process.env.GA4_MAX_UTM_ROWS, 10_000, 25_000);
    const pageSize = clampLimit(body?.pageSize, Math.min(5000, utmMaxRows), Math.min(10_000, utmMaxRows));
    const maxRows = clampLimit(body?.maxRows, utmMaxRows, utmMaxRows);
    const firstPageSize = Math.min(pageSize, maxRows);

    const dimensionFilter = buildPageLocationFilter(body?.filters);
    const dimensionNames = ['pageLocation'];
    const metricNames = ['sessions', 'totalUsers', 'eventCount'];
    const reportBase = {
      dateRanges: [{ startDate, endDate }],
      dimensions: dimensionNames.map((name) => ({ name })),
      metrics: metricNames.map((name) => ({ name })),
      limit: firstPageSize,
      offset: 0
    };
    if (dimensionFilter) reportBase.dimensionFilter = dimensionFilter;

    let first;
    try {
      first = await runReport(serviceAccount, propertyId, reportBase, { deadlineAt: upstreamDeadlineAt });
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const rowCount = Number(first?.rowCount) || 0;
    const rawRows = Array.isArray(first?.rows) ? first.rows.slice() : [];
    let quota = first && typeof first.propertyQuota === 'object' ? first.propertyQuota : null;

    let offset = rawRows.length;
    while (rowCount && offset < rowCount && rawRows.length < maxRows) {
      let page;
      const requestedRows = Math.min(pageSize, maxRows - rawRows.length);
      try {
        page = await runReport(
          serviceAccount,
          propertyId,
          { ...reportBase, limit: requestedRows, offset },
          { deadlineAt: upstreamDeadlineAt }
        );
      } catch (err) {
        sendJson(res, 502, { ok: false, error: err.message });
        return;
      }

      const pageRows = Array.isArray(page?.rows) ? page.rows : [];
      if (!pageRows.length) break;
      rawRows.push(...pageRows);
      offset += pageRows.length;
      if (pageRows.length < requestedRows) break;
      if (page && typeof page.propertyQuota === 'object') quota = page.propertyQuota;
    }

    const rows = flattenRows(rawRows.slice(0, maxRows), dimensionNames, metricNames)
      .filter((row) => String(row.pageLocation || '').trim());

    sendReportSuccess(res, cacheKey, {
      ok: true,
      kind,
      propertyId,
      startDate,
      endDate,
      dimensionNames,
      metricNames,
      rowCount,
      returnedRows: rows.length,
      truncated: rowCount ? rows.length < rowCount : rawRows.length >= maxRows,
      quota,
      rows
    });
    return;
  }

  if (kind === 'explore') {
    let dimensions;
    let metrics;
    let dimensionFilters;
    try {
      dimensions = normalizeFieldList(body?.dimensions, 'dimension', 9);
      metrics = normalizeFieldList(body?.metrics, 'metric', 10);
      dimensionFilters = normalizeDimensionFilters(body?.dimensionFilters);
    } catch (err) {
      sendJson(res, 400, { ok: false, error: err.message || 'Invalid explore configuration.' });
      return;
    }

    if (!metrics.length) {
      sendJson(res, 400, { ok: false, error: 'Select at least 1 metric.' });
      return;
    }

    const exploreMaxRows = positiveInteger(process.env.GA4_MAX_EXPLORE_ROWS, 5_000, 10_000);
    const maxRows = clampLimit(body?.maxRows, 200, exploreMaxRows);
    const orderDir = String(body?.orderDir || 'desc').trim().toLowerCase() === 'asc' ? 'asc' : 'desc';
    const orderBy = normalizeGa4FieldName(body?.orderBy);

    const trafficSourceFilter = buildTrafficSourceFilter(body?.filters);
    const customDimensionFilter = buildDimensionFiltersExpression(dimensionFilters);
    const dimensionFilter = mergeDimensionFilters(trafficSourceFilter, customDimensionFilter);

    const reportBody = {
      dateRanges: [{ startDate, endDate }],
      metrics: metrics.map((name) => ({ name })),
      limit: maxRows
    };
    if (dimensions.length) reportBody.dimensions = dimensions.map((name) => ({ name }));
    if (dimensionFilter) reportBody.dimensionFilter = dimensionFilter;

    if (orderBy) {
      if (metrics.includes(orderBy)) {
        reportBody.orderBys = [{ metric: { metricName: orderBy }, desc: orderDir !== 'asc' }];
      } else if (dimensions.includes(orderBy)) {
        reportBody.orderBys = [{ dimension: { dimensionName: orderBy }, desc: orderDir !== 'asc' }];
      }
    }

    if (!reportBody.orderBys) {
      reportBody.orderBys = [{ metric: { metricName: metrics[0] }, desc: orderDir !== 'asc' }];
    }

    let report;
    try {
      report = await runReport(serviceAccount, propertyId, reportBody, { deadlineAt: upstreamDeadlineAt });
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const rowCount = Number(report?.rowCount) || 0;
    const rows = flattenRows(report?.rows, dimensions, metrics);
    const quota = report && typeof report.propertyQuota === 'object' ? report.propertyQuota : null;

    sendReportSuccess(res, cacheKey, {
      ok: true,
      kind,
      propertyId,
      startDate,
      endDate,
      dimensionNames: dimensions,
      metricNames: metrics,
      rowCount,
      returnedRows: rows.length,
      truncated: rowCount ? rows.length < rowCount : false,
      quota,
      rows
    });
    return;
  }

  if (kind === 'insights' || kind === 'ping') {
    const includeUtm = kind === 'ping' ? false : body?.includeUtm !== false;
    const breakdown = kind === 'ping' ? 'country' : normalizeInsightsBreakdown(body?.breakdown);
    const requestedFields = normalizeUtmGroupFields(body?.groupFields);
    const defaultFields = ['utm_source', 'utm_medium', 'utm_campaign'];
    const utmFields = requestedFields.length ? requestedFields : defaultFields;

    const dimensions = [];
    if (includeUtm) {
      utmFields.forEach((field) => {
        const dim = UTM_DIMENSIONS[field];
        if (dim && !dimensions.includes(dim)) dimensions.push(dim);
      });
    }
    if (breakdown && !dimensions.includes(breakdown)) dimensions.push(breakdown);

    if (!dimensions.length) {
      sendJson(res, 400, { ok: false, error: 'Select a breakdown and/or include UTM dimensions.' });
      return;
    }

    const metricNames = [
      'sessions',
      'totalUsers',
      'newUsers',
      'engagedSessions',
      'engagementRate',
      'bounceRate',
      'eventCount'
    ];

    const insightsMaxRows = positiveInteger(process.env.GA4_MAX_INSIGHTS_ROWS, 5_000, 10_000);
    const maxRows = kind === 'ping' ? 1 : clampLimit(body?.maxRows, 200, insightsMaxRows);
    const orderByFieldRaw = String(body?.orderBy || 'sessions').trim();
    const allowedOrderBys = new Set(['sessions', 'totalUsers', 'newUsers', 'engagedSessions', 'eventCount']);
    const orderByField = allowedOrderBys.has(orderByFieldRaw) ? orderByFieldRaw : 'sessions';
    const orderDir = String(body?.orderDir || 'desc').trim().toLowerCase() === 'asc' ? 'asc' : 'desc';

    const dimensionFilter = kind === 'ping' ? null : buildTrafficSourceFilter(body?.filters);

    const reportBody = {
      dateRanges: [{ startDate, endDate }],
      dimensions: dimensions.map((name) => ({ name })),
      metrics: metricNames.map((name) => ({ name })),
      limit: maxRows,
      orderBys: [
        {
          metric: { metricName: orderByField },
          desc: orderDir !== 'asc'
        }
      ]
    };
    if (dimensionFilter) reportBody.dimensionFilter = dimensionFilter;

    let report;
    try {
      report = await runReport(serviceAccount, propertyId, reportBody, { deadlineAt: upstreamDeadlineAt });
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const rowCount = Number(report?.rowCount) || 0;
    const rows = flattenRows(report?.rows, dimensions, metricNames);
    const quota = report && typeof report.propertyQuota === 'object' ? report.propertyQuota : null;

    sendReportSuccess(res, cacheKey, {
      ok: true,
      kind,
      propertyId,
      startDate,
      endDate,
      dimensionNames: dimensions,
      metricNames,
      rowCount,
      returnedRows: rows.length,
      truncated: rowCount ? rows.length < rowCount : false,
      quota,
      rows
    });
    return;
  }

  sendJson(res, 400, { ok: false, error: 'Unsupported report kind.' });
};

module.exports._internal = {
  buildResponseCacheKey,
  consumeRateLimit,
  fitResponsePayload,
  getAllowedPropertyIds,
  getCachedResponse,
  parseIsoDate,
  setCachedResponse,
  timingSafeTokenEqual,
  validateDateRange
};
