'use strict';

const { sendJson, readJson, getBearerToken, clampLimit } = require('../_lib/tools-api');
const { runReport } = require('../_lib/ga4-data-api');

function normalizeAdminToken(value) {
  return String(value || '').trim();
}

function normalizePropertyId(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (!/^\d{6,20}$/.test(raw)) return '';
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
  if (!providedToken || providedToken !== expectedToken) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body.' });
    return;
  }

  const propertyId = normalizePropertyId(body?.propertyId || process.env.GA4_PROPERTY_ID);
  if (!propertyId) {
    sendJson(res, 400, { ok: false, error: 'Missing GA4 propertyId.' });
    return;
  }

  const startDate = normalizeIsoDate(body?.startDate);
  const endDate = normalizeIsoDate(body?.endDate);
  if (!startDate || !endDate) {
    sendJson(res, 400, { ok: false, error: 'Missing startDate/endDate (expected YYYY-MM-DD).' });
    return;
  }

  const pageSize = clampLimit(body?.pageSize, 5000, 10000);
  const maxRows = clampLimit(body?.maxRows, 25000, 50000);

  let serviceAccount;
  try {
    serviceAccount = loadServiceAccount();
  } catch (err) {
    sendJson(res, 503, { ok: false, error: err.message });
    return;
  }

  const dimensionFilter = buildPageLocationFilter(body?.filters);
  const reportBase = {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: 'pageLocation' }],
    metrics: [{ name: 'sessions' }, { name: 'totalUsers' }, { name: 'eventCount' }],
    limit: pageSize,
    offset: 0
  };
  if (dimensionFilter) reportBase.dimensionFilter = dimensionFilter;

  let first;
  try {
    first = await runReport(serviceAccount, propertyId, reportBase);
  } catch (err) {
    sendJson(res, 502, { ok: false, error: err.message });
    return;
  }

  const rowCount = Number(first?.rowCount) || 0;
  const rawRows = Array.isArray(first?.rows) ? first.rows.slice() : [];

  let offset = rawRows.length;
  while (rowCount && offset < rowCount && rawRows.length < maxRows) {
    let page;
    try {
      page = await runReport(serviceAccount, propertyId, { ...reportBase, offset });
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const pageRows = Array.isArray(page?.rows) ? page.rows : [];
    if (!pageRows.length) break;
    rawRows.push(...pageRows);
    offset += pageRows.length;
    if (pageRows.length < pageSize) break;
  }

  const rows = rawRows
    .slice(0, maxRows)
    .map((row) => {
      const pageLocation = String(row?.dimensionValues?.[0]?.value || '').trim();
      if (!pageLocation) return null;
      return {
        pageLocation,
        sessions: coerceMetricValue(row?.metricValues?.[0]?.value),
        totalUsers: coerceMetricValue(row?.metricValues?.[1]?.value),
        eventCount: coerceMetricValue(row?.metricValues?.[2]?.value)
      };
    })
    .filter(Boolean);

  sendJson(res, 200, {
    ok: true,
    propertyId,
    startDate,
    endDate,
    rowCount,
    returnedRows: rows.length,
    truncated: rowCount ? rows.length < rowCount : rawRows.length >= maxRows,
    rows
  });
};
