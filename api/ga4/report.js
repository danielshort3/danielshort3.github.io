'use strict';

const { sendJson, readJson, getBearerToken, clampLimit } = require('../_lib/tools-api');
const { runReport } = require('../_lib/ga4-data-api');

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

function normalizeGa4FieldName(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.length > 80) return '';
  if (!/^[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?$/.test(raw)) return '';
  return raw;
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

  let serviceAccount;
  try {
    serviceAccount = loadServiceAccount();
  } catch (err) {
    sendJson(res, 503, { ok: false, error: err.message });
    return;
  }

  const startDateInput = normalizeIsoDate(body?.startDate);
  const endDateInput = normalizeIsoDate(body?.endDate);

  const getDateRange = () => {
    if (startDateInput && endDateInput) return { startDate: startDateInput, endDate: endDateInput };
    if (kind !== 'ping') return null;

    const now = new Date();
    const end = now;
    const start = new Date(now.getTime() - 6 * 24 * 60 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    const iso = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
    return { startDate: iso(start), endDate: iso(end) };
  };

  const dateRange = getDateRange();
  if (!dateRange) {
    sendJson(res, 400, { ok: false, error: 'Missing startDate/endDate (expected YYYY-MM-DD).' });
    return;
  }

  const { startDate, endDate } = dateRange;

  if (kind === 'utm-urls') {
    const pageSize = clampLimit(body?.pageSize, 5000, 10000);
    const maxRows = clampLimit(body?.maxRows, 25000, 50000);

    const dimensionFilter = buildPageLocationFilter(body?.filters);
    const dimensionNames = ['pageLocation'];
    const metricNames = ['sessions', 'totalUsers', 'eventCount'];
    const reportBase = {
      dateRanges: [{ startDate, endDate }],
      dimensions: dimensionNames.map((name) => ({ name })),
      metrics: metricNames.map((name) => ({ name })),
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
    let quota = first && typeof first.propertyQuota === 'object' ? first.propertyQuota : null;

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
      if (page && typeof page.propertyQuota === 'object') quota = page.propertyQuota;
    }

    const rows = flattenRows(rawRows.slice(0, maxRows), dimensionNames, metricNames)
      .filter((row) => String(row.pageLocation || '').trim());

    sendJson(res, 200, {
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
    try {
      dimensions = normalizeFieldList(body?.dimensions, 'dimension', 9);
      metrics = normalizeFieldList(body?.metrics, 'metric', 10);
    } catch (err) {
      sendJson(res, 400, { ok: false, error: err.message || 'Invalid explore configuration.' });
      return;
    }

    if (!metrics.length) {
      sendJson(res, 400, { ok: false, error: 'Select at least 1 metric.' });
      return;
    }

    const maxRows = clampLimit(body?.maxRows, 200, 10000);
    const orderDir = String(body?.orderDir || 'desc').trim().toLowerCase() === 'asc' ? 'asc' : 'desc';
    const orderBy = normalizeGa4FieldName(body?.orderBy);

    const dimensionFilter = buildTrafficSourceFilter(body?.filters);

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
      report = await runReport(serviceAccount, propertyId, reportBody);
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const rowCount = Number(report?.rowCount) || 0;
    const rows = flattenRows(report?.rows, dimensions, metrics);
    const quota = report && typeof report.propertyQuota === 'object' ? report.propertyQuota : null;

    sendJson(res, 200, {
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

    const maxRows = kind === 'ping' ? 1 : clampLimit(body?.maxRows, 200, 10000);
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
      report = await runReport(serviceAccount, propertyId, reportBody);
    } catch (err) {
      sendJson(res, 502, { ok: false, error: err.message });
      return;
    }

    const rowCount = Number(report?.rowCount) || 0;
    const rows = flattenRows(report?.rows, dimensions, metricNames);
    const quota = report && typeof report.propertyQuota === 'object' ? report.propertyQuota : null;

    sendJson(res, 200, {
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
