(() => {
  'use strict';

  const TOOL_ID = 'ga4-utm-performance';

  const TOKEN_STORAGE_KEY = 'ga4UtmPerformance:adminToken';
  const PROFILE_STORAGE_KEY = 'ga4UtmPerformance:profiles';
  const ACTIVE_PROFILE_STORAGE_KEY = 'ga4UtmPerformance:activeProfile';

  const MAX_GROUP_RENDER_ROWS = 5000;
  const MAX_RAW_RENDER_ROWS = 5000;
  const MAX_DRILLDOWN_RENDER_ROWS = 500;
  const MAX_INSIGHTS_RENDER_ROWS = 2000;

  const DEFAULT_GROUP_FIELDS = ['utm_source', 'utm_medium', 'utm_campaign'];
  const UTM_FIELDS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'utm_id'];
  const URL_METRIC_FIELDS = ['sessions', 'totalUsers', 'eventCount'];
  const RATE_METRICS = new Set(['engagementRate', 'bounceRate']);

  const DIMENSION_LABELS = {
    sessionSource: 'utm_source',
    sessionMedium: 'utm_medium',
    sessionCampaignName: 'utm_campaign',
    sessionManualAdContent: 'utm_content',
    sessionManualTerm: 'utm_term',
    sessionCampaignId: 'utm_id',
    country: 'Country',
    region: 'Region',
    city: 'City',
    language: 'Language',
    deviceCategory: 'Device category',
    browser: 'Browser',
    operatingSystem: 'Operating system',
    platform: 'Platform',
    userAgeBracket: 'Age bracket',
    userGender: 'Gender',
    pageLocation: 'Page URL',
    landingPagePlusQueryString: 'Landing page',
    eventName: 'Event name'
  };

  const METRIC_LABELS = {
    sessions: 'Sessions',
    totalUsers: 'Users',
    activeUsers: 'Active users',
    newUsers: 'New users',
    engagedSessions: 'Engaged sessions',
    engagementRate: 'Engagement rate',
    bounceRate: 'Bounce rate',
    eventCount: 'Events',
    screenPageViews: 'Page views',
    conversions: 'Conversions',
    totalRevenue: 'Revenue',
    userEngagementDuration: 'Engagement duration',
    averageSessionDuration: 'Avg session duration'
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const main = document.getElementById('main');
  if (!main) return;

  const accessForm = $('[data-ga4="auth"]', main);
  const tokenInput = $('[data-ga4="token"]', main);
  const forgetTokenBtn = $('[data-ga4="forget-token"]', main);
  const checkAccessBtn = $('[data-ga4="check-access"]', main);
  const profileSelect = $('[data-ga4="profile-select"]', main);
  const profileLabelInput = $('[data-ga4="profile-label"]', main);
  const propertyIdInput = $('[data-ga4="property-id"]', main);
  const saveProfileBtn = $('[data-ga4="save-profile"]', main);
  const deleteProfileBtn = $('[data-ga4="delete-profile"]', main);
  const accessMetaEl = $('[data-ga4="access-meta"]', main);
  const accessStatusEl = $('[data-ga4="access-status"]', main);
  const quotaEl = $('[data-ga4="quota"]', main);

  const scopeForm = $('[data-ga4="scope-form"]', main);
  const startEl = $('[data-ga4="start"]', main);
  const endEl = $('[data-ga4="end"]', main);

  const filtersForm = $('[data-ga4="filters-form"]', main);
  const matchTypeEl = $('[data-ga4="utm-match"]', main);
  const utmSourceEl = $('[data-ga4="utm-source"]', main);
  const utmMediumEl = $('[data-ga4="utm-medium"]', main);
  const utmCampaignEl = $('[data-ga4="utm-campaign"]', main);
  const utmContentEl = $('[data-ga4="utm-content"]', main);
  const utmTermEl = $('[data-ga4="utm-term"]', main);
  const utmIdEl = $('[data-ga4="utm-id"]', main);
  const catchAllEl = $('[data-ga4="catch-all"]', main);

  const utmForm = $('[data-ga4="utm-form"]', main);
  const runBtn = $('[data-ga4="run"]', main);
  const downloadGroupedBtn = $('[data-ga4="download-grouped"]', main);
  const downloadRawBtn = $('[data-ga4="download-raw"]', main);
  const viewModeEl = $('[data-ga4="view-mode"]', main);
  const sortFieldEl = $('[data-ga4="sort-field"]', main);
  const sortDirEl = $('[data-ga4="sort-dir"]', main);
  const localFilterEl = $('[data-ga4="local-filter"]', main);
  const utmStatusEl = $('[data-ga4="utm-status"]', main);
  const utmSummaryEl = $('[data-ga4="utm-summary"]', main);
  const utmOutputEl = $('[data-ga4="utm-output"]', main);
  const utmDrilldownEl = $('[data-ga4="utm-drilldown"]', main);

  const exploreForm = $('[data-ga4="explore-form"]', main);
  const explorePresetEl = $('[data-ga4="explore-preset"]', main);
  const exploreDimensionsWrap = $('[data-ga4="explore-dimensions"]', main);
  const exploreMetricsWrap = $('[data-ga4="explore-metrics"]', main);
  const exploreCustomDimensionsEl = $('[data-ga4="explore-custom-dimensions"]', main);
  const exploreCustomMetricsEl = $('[data-ga4="explore-custom-metrics"]', main);
  const exploreMaxRowsEl = $('[data-ga4="explore-max-rows"]', main);
  const exploreOrderByEl = $('[data-ga4="explore-order-by"]', main);
  const exploreDirEl = $('[data-ga4="explore-dir"]', main);
  const exploreFilterEl = $('[data-ga4="explore-filter"]', main);
  const exploreRunBtn = $('[data-ga4="explore-run"]', main);
  const exploreDownloadBtn = $('[data-ga4="explore-download"]', main);
  const exploreStatusEl = $('[data-ga4="explore-status"]', main);
  const exploreSummaryEl = $('[data-ga4="explore-summary"]', main);
  const exploreOutputEl = $('[data-ga4="explore-output"]', main);

  if (!accessForm || !tokenInput || !forgetTokenBtn || !checkAccessBtn) return;
  if (!profileSelect || !profileLabelInput || !propertyIdInput || !saveProfileBtn || !deleteProfileBtn) return;
  if (!startEl || !endEl || !utmForm || !runBtn || !downloadGroupedBtn || !downloadRawBtn) return;
  if (!viewModeEl || !sortFieldEl || !sortDirEl || !localFilterEl) return;
  if (!utmStatusEl || !utmSummaryEl || !utmOutputEl || !utmDrilldownEl) return;
  if (!exploreForm || !explorePresetEl || !exploreDimensionsWrap || !exploreMetricsWrap) return;
  if (!exploreCustomDimensionsEl || !exploreCustomMetricsEl || !exploreMaxRowsEl || !exploreOrderByEl) return;
  if (!exploreDirEl || !exploreFilterEl || !exploreRunBtn || !exploreDownloadBtn) return;
  if (!exploreStatusEl || !exploreSummaryEl || !exploreOutputEl) return;
  if (!matchTypeEl || !catchAllEl) return;
  if (!utmSourceEl || !utmMediumEl || !utmCampaignEl || !utmContentEl || !utmTermEl || !utmIdEl) return;

  const storage = (() => {
    try {
      const key = '__ga4_test__';
      window.localStorage.setItem(key, '1');
      window.localStorage.removeItem(key);
      return window.localStorage;
    } catch {
      return null;
    }
  })();

  let memoryToken = '';
  let lastUtm = null;
  let lastExplore = null;
  let lastQuota = null;
  let drilldownGroup = null;
  let utmBusy = false;
  let exploreBusy = false;

  const getSavedToken = () => {
    if (storage) return storage.getItem(TOKEN_STORAGE_KEY) || '';
    return memoryToken;
  };

  const saveToken = (token) => {
    const value = String(token || '').trim();
    if (storage) {
      if (!value) storage.removeItem(TOKEN_STORAGE_KEY);
      else storage.setItem(TOKEN_STORAGE_KEY, value);
      return;
    }
    memoryToken = value;
  };

  const readStorageJson = (key, fallback) => {
    if (!storage) return fallback;
    const raw = storage.getItem(key);
    if (!raw) return fallback;
    try {
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  };

  const writeStorageJson = (key, value) => {
    if (!storage) return;
    try {
      storage.setItem(key, JSON.stringify(value));
    } catch {}
  };

  const deleteStorageKey = (key) => {
    if (!storage) return;
    try {
      storage.removeItem(key);
    } catch {}
  };

  const setStatus = (el, text, tone = '') => {
    if (!el) return;
    el.textContent = String(text || '');
    el.dataset.tone = tone;
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const normalizePropertyId = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return '';
    const digits = raw.replace(/[^\d]/g, '');
    if (!digits) return '';
    if (digits.length > 20) return digits.slice(0, 20);
    return digits;
  };

  const getPropertyId = () => normalizePropertyId(propertyIdInput.value);

  const formatNumber = (value) => Number(value || 0).toLocaleString('en-US');

  const formatPercent = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return '0%';
    return `${(n * 100).toFixed(1)}%`;
  };

  const escapeCsv = (value) => {
    const raw = String(value ?? '');
    if (!/[\",\n\r]/.test(raw)) return raw;
    return `"${raw.replace(/\"/g, '""')}"`;
  };

  const toCsv = (rows, headers) => {
    const head = headers.map(escapeCsv).join(',');
    const body = rows.map((row) => headers.map((h) => escapeCsv(row[h])).join(',')).join('\n');
    return `${head}\n${body}\n`;
  };

  const downloadCsv = (rows, headers, filename) => {
    const csv = toCsv(rows, headers);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const updateAccessMeta = () => {
    if (!accessMetaEl) return;
    accessMetaEl.textContent = getSavedToken() ? 'Token stored' : 'Token required';
  };

  const coerceInt = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return Math.floor(n);
  };

  const normalizeQuotaBucket = (bucket) => {
    if (!bucket || typeof bucket !== 'object') return null;
    const consumed = coerceInt(bucket.consumed);
    const remaining = coerceInt(bucket.remaining);
    if (consumed === null && remaining === null) return null;
    return { consumed, remaining };
  };

  const formatQuotaBucket = (label, bucket) => {
    if (!bucket) return '';
    const consumed = bucket.consumed;
    const remaining = bucket.remaining;
    if (consumed !== null && remaining !== null) {
      const total = consumed + remaining;
      return `${label}: ${formatNumber(consumed)}/${formatNumber(total)} (${formatNumber(remaining)} left)`;
    }
    if (remaining !== null) return `${label}: ${formatNumber(remaining)} left`;
    if (consumed !== null) return `${label}: ${formatNumber(consumed)} used`;
    return '';
  };

  const renderQuota = (quota) => {
    lastQuota = quota && typeof quota === 'object' ? quota : null;
    if (!quotaEl) return;
    quotaEl.replaceChildren();
    if (!lastQuota) return;

    const hour = normalizeQuotaBucket(lastQuota.tokensPerHour);
    const day = normalizeQuotaBucket(lastQuota.tokensPerDay);
    const parts = [
      formatQuotaBucket('Tokens/hr', hour),
      formatQuotaBucket('Tokens/day', day)
    ].filter(Boolean);
    if (!parts.length) return;

    parts.forEach((text) => {
      const pill = document.createElement('span');
      pill.className = 'tool-pill';
      pill.textContent = text;
      quotaEl.appendChild(pill);
    });
  };

  const loadProfiles = () => {
    const raw = readStorageJson(PROFILE_STORAGE_KEY, []);
    if (!Array.isArray(raw)) return [];
    return raw
      .map((entry) => {
        const label = String(entry?.label || '').trim();
        const propertyId = normalizePropertyId(entry?.propertyId);
        if (!label || !propertyId) return null;
        return { label, propertyId };
      })
      .filter(Boolean);
  };

  const saveProfiles = (profiles) => {
    const cleaned = (Array.isArray(profiles) ? profiles : [])
      .map((entry) => {
        const label = String(entry?.label || '').trim();
        const propertyId = normalizePropertyId(entry?.propertyId);
        if (!label || !propertyId) return null;
        return { label, propertyId };
      })
      .filter(Boolean)
      .sort((a, b) => a.label.localeCompare(b.label));
    writeStorageJson(PROFILE_STORAGE_KEY, cleaned);
    return cleaned;
  };

  const getActiveProfile = () => {
    const value = storage ? String(storage.getItem(ACTIVE_PROFILE_STORAGE_KEY) || '') : '';
    return normalizePropertyId(value);
  };

  const setActiveProfile = (propertyId) => {
    const value = normalizePropertyId(propertyId);
    if (!storage) return;
    if (!value) deleteStorageKey(ACTIVE_PROFILE_STORAGE_KEY);
    else storage.setItem(ACTIVE_PROFILE_STORAGE_KEY, value);
  };

  const renderProfileSelect = (profiles) => {
    const list = Array.isArray(profiles) ? profiles : loadProfiles();
    const active = getActiveProfile();

    const currentValue = normalizePropertyId(profileSelect.value);
    const keepCurrent = currentValue && list.some((p) => p.propertyId === currentValue);

    profileSelect.replaceChildren();
    const customOpt = document.createElement('option');
    customOpt.value = '';
    customOpt.textContent = 'Custom';
    profileSelect.appendChild(customOpt);

    list.forEach((profile) => {
      const opt = document.createElement('option');
      opt.value = profile.propertyId;
      opt.textContent = `${profile.label} (${profile.propertyId})`;
      profileSelect.appendChild(opt);
    });

    if (keepCurrent) {
      profileSelect.value = currentValue;
      return;
    }

    profileSelect.value = active && list.some((p) => p.propertyId === active) ? active : '';
  };

  const syncProfileInputs = () => {
    const profiles = loadProfiles();
    const selected = normalizePropertyId(profileSelect.value);
    const match = profiles.find((p) => p.propertyId === selected);
    if (match) {
      profileLabelInput.value = match.label;
      propertyIdInput.value = match.propertyId;
      setActiveProfile(match.propertyId);
      return;
    }

    if (selected) {
      propertyIdInput.value = selected;
      setActiveProfile(selected);
      return;
    }

    setActiveProfile('');
  };

  const isoDate = (d) => {
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };

  const initDefaultDates = () => {
    try {
      if (startEl.value && endEl.value) return;
      const end = new Date();
      const start = new Date(end.getTime() - 29 * 24 * 60 * 60 * 1000);
      if (!endEl.value) endEl.value = isoDate(end);
      if (!startEl.value) startEl.value = isoDate(start);
    } catch {}
  };

  const safeUrl = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return null;
    try {
      return new URL(raw);
    } catch {
      try {
        return new URL(raw, window.location.origin);
      } catch {
        return null;
      }
    }
  };

  const parseUtmFromUrl = (pageLocation) => {
    const url = safeUrl(pageLocation);
    if (!url) {
      return {
        utm_source: '',
        utm_medium: '',
        utm_campaign: '',
        utm_content: '',
        utm_term: '',
        utm_id: ''
      };
    }
    const p = url.searchParams;
    return {
      utm_source: p.get('utm_source') || '',
      utm_medium: p.get('utm_medium') || '',
      utm_campaign: p.get('utm_campaign') || '',
      utm_content: p.get('utm_content') || '',
      utm_term: p.get('utm_term') || '',
      utm_id: p.get('utm_id') || ''
    };
  };

  const getGroupFields = () => {
    const inputs = $$('input[name="ga4-group"]', main);
    const fields = inputs
      .filter((el) => el && el.checked)
      .map((el) => String(el.value || '').trim())
      .filter((field) => UTM_FIELDS.includes(field));
    return fields.length ? fields : DEFAULT_GROUP_FIELDS.slice();
  };

  const normalizeViewMode = (value) => String(value || '').trim().toLowerCase() === 'raw' ? 'raw' : 'grouped';

  const normalizeSortDir = (value) => String(value || '').trim().toLowerCase() === 'asc' ? 'asc' : 'desc';

  const numberOrZero = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  const sortState = {
    grouped: { field: 'sessions', dir: 'desc' },
    raw: { field: 'sessions', dir: 'desc' },
    drilldown: { field: 'sessions', dir: 'desc' }
  };

  const getLocalFilterQuery = () => String(localFilterEl.value || '').trim().toLowerCase();

  const getSortOptions = (mode) => {
    const baseStrings = mode === 'raw'
      ? [{ value: 'pathname', label: 'Path', type: 'string' }, { value: 'pageLocation', label: 'URL', type: 'string' }]
      : [];
    const baseNumbers = mode === 'grouped'
      ? [
        { value: 'sessions', label: 'Sessions', type: 'number' },
        { value: 'totalUsers', label: 'Users', type: 'number' },
        { value: 'eventCount', label: 'Events', type: 'number' },
        { value: 'rows', label: 'URLs', type: 'number' }
      ]
      : [
        { value: 'sessions', label: 'Sessions', type: 'number' },
        { value: 'totalUsers', label: 'Users', type: 'number' },
        { value: 'eventCount', label: 'Events', type: 'number' }
      ];
    return [
      ...baseNumbers,
      ...baseStrings,
      ...UTM_FIELDS.map((field) => ({ value: field, label: field, type: 'string' }))
    ];
  };

  const getFieldType = (mode, field) => {
    const options = getSortOptions(mode);
    const match = options.find((opt) => opt.value === field);
    return match ? match.type : 'string';
  };

  const syncSortControls = () => {
    const mode = normalizeViewMode(viewModeEl.value);
    const options = getSortOptions(mode);
    const allowed = new Set(options.map((opt) => opt.value));
    const current = sortState[mode] || sortState.grouped;
    if (!allowed.has(current.field)) {
      current.field = 'sessions';
      current.dir = getFieldType(mode, current.field) === 'number' ? 'desc' : 'asc';
    }

    sortFieldEl.replaceChildren();
    options.forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.label;
      sortFieldEl.appendChild(option);
    });
    sortFieldEl.value = current.field;
    sortDirEl.value = current.dir;
  };

  const sortList = (items, mode) => {
    const state = sortState[mode] || sortState.grouped;
    const field = String(state.field || '').trim() || 'sessions';
    const dir = normalizeSortDir(state.dir);
    const type = getFieldType(mode, field);

    const copy = Array.isArray(items) ? items.slice() : [];
    copy.sort((a, b) => {
      if (type === 'number') {
        const av = numberOrZero(a?.[field]);
        const bv = numberOrZero(b?.[field]);
        return dir === 'asc' ? (av - bv) : (bv - av);
      }

      const as = String(a?.[field] || '').toLowerCase();
      const bs = String(b?.[field] || '').toLowerCase();
      const aEmpty = !as;
      const bEmpty = !bs;
      if (aEmpty && !bEmpty) return 1;
      if (!aEmpty && bEmpty) return -1;
      if (aEmpty && bEmpty) return 0;
      const cmp = as.localeCompare(bs);
      return dir === 'asc' ? cmp : -cmp;
    });
    return copy;
  };

  const formatCellText = (value) => {
    const raw = String(value ?? '').trim();
    return raw ? raw : '(not set)';
  };

  const buildParsedUrlRows = (rows) => {
    const list = Array.isArray(rows) ? rows : [];
    return list
      .map((row) => {
        const pageLocation = String(row?.pageLocation || '').trim();
        if (!pageLocation) return null;

        const url = safeUrl(pageLocation);
        const pathname = url ? String(url.pathname || '') : '';
        const utm = parseUtmFromUrl(pageLocation);

        const out = {
          pageLocation,
          pathname,
          sessions: numberOrZero(row?.sessions),
          totalUsers: numberOrZero(row?.totalUsers),
          eventCount: numberOrZero(row?.eventCount)
        };
        UTM_FIELDS.forEach((field) => { out[field] = String(utm[field] || ''); });
        return out;
      })
      .filter(Boolean);
  };

  const aggregateGroups = (parsedRows, groupFields) => {
    const fields = Array.isArray(groupFields) ? groupFields : [];
    const map = new Map();

    (Array.isArray(parsedRows) ? parsedRows : []).forEach((row) => {
      const key = fields.map((field) => String(row?.[field] || '')).join('\u0000');
      let entry = map.get(key);
      if (!entry) {
        entry = { key, sessions: 0, totalUsers: 0, eventCount: 0, rows: 0 };
        fields.forEach((field) => { entry[field] = String(row?.[field] || ''); });
        map.set(key, entry);
      }
      entry.sessions += numberOrZero(row?.sessions);
      entry.totalUsers += numberOrZero(row?.totalUsers);
      entry.eventCount += numberOrZero(row?.eventCount);
      entry.rows += 1;
    });

    return Array.from(map.values());
  };

  const sumUrlMetrics = (parsedRows) => {
    const totals = { sessions: 0, totalUsers: 0, eventCount: 0, rows: 0 };
    (Array.isArray(parsedRows) ? parsedRows : []).forEach((row) => {
      totals.sessions += numberOrZero(row?.sessions);
      totals.totalUsers += numberOrZero(row?.totalUsers);
      totals.eventCount += numberOrZero(row?.eventCount);
      totals.rows += 1;
    });
    return totals;
  };

  const applyLocalFilterGrouped = (groups, groupFields, query) => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return Array.isArray(groups) ? groups.slice() : [];
    const fields = Array.isArray(groupFields) && groupFields.length ? groupFields : UTM_FIELDS;
    return (Array.isArray(groups) ? groups : []).filter((group) => {
      if (!group || typeof group !== 'object') return false;
      if (String(group.sessions || '').includes(q)) return true;
      return fields.some((field) => String(group[field] || '').toLowerCase().includes(q));
    });
  };

  const applyLocalFilterRaw = (rows, query) => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return Array.isArray(rows) ? rows.slice() : [];
    return (Array.isArray(rows) ? rows : []).filter((row) => {
      if (!row || typeof row !== 'object') return false;
      if (String(row.pageLocation || '').toLowerCase().includes(q)) return true;
      if (String(row.pathname || '').toLowerCase().includes(q)) return true;
      return UTM_FIELDS.some((field) => String(row[field] || '').toLowerCase().includes(q));
    });
  };

  const renderUrlSummary = (data) => {
    utmSummaryEl.replaceChildren();
    if (!data || typeof data !== 'object') return;

    const totals = data.totals || sumUrlMetrics(data.parsedRows);
    const pills = document.createElement('div');
    pills.className = 'tools-actions';

    const makePill = (label, value) => {
      const span = document.createElement('span');
      span.className = 'tool-pill';
      span.textContent = `${label}: ${value}`;
      return span;
    };

    pills.appendChild(makePill('URLs', formatNumber(totals.rows || 0)));
    pills.appendChild(makePill('Sessions', formatNumber(totals.sessions || 0)));
    pills.appendChild(makePill('Users', formatNumber(totals.totalUsers || 0)));
    pills.appendChild(makePill('Events', formatNumber(totals.eventCount || 0)));
    if (Array.isArray(data.groups)) {
      pills.appendChild(makePill('Groups', formatNumber(data.groups.length || 0)));
    }

    utmSummaryEl.appendChild(pills);
  };

  const makeStyledTable = (headers, rows, cellBuilder) => {
    const wrap = document.createElement('div');
    wrap.className = 'shortlinks-table-wrap';

    const table = document.createElement('table');
    table.className = 'shortlinks-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((header) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = header;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      cellBuilder(tr, row);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    wrap.appendChild(table);
    return wrap;
  };

  const renderGroupedView = (data) => {
    utmOutputEl.replaceChildren();

    const groupFields = Array.isArray(data?.groupFields) ? data.groupFields : getGroupFields();
    const groups = Array.isArray(data?.groups) ? data.groups : [];
    const query = getLocalFilterQuery();

    const filtered = applyLocalFilterGrouped(groups, groupFields, query);
    const sorted = sortList(filtered, 'grouped');

    if (!sorted.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching groups.';
      utmOutputEl.appendChild(p);
      return;
    }

    const meta = document.createElement('p');
    meta.className = 'contact-form-note';
    if (sorted.length > MAX_GROUP_RENDER_ROWS) {
      meta.textContent = `Showing first ${formatNumber(MAX_GROUP_RENDER_ROWS)} of ${formatNumber(sorted.length)} groups.`;
      utmOutputEl.appendChild(meta);
    } else if (query) {
      meta.textContent = `Showing ${formatNumber(sorted.length)} group(s) after client-side filter.`;
      utmOutputEl.appendChild(meta);
    }

    const headers = [...groupFields, ...URL_METRIC_FIELDS, 'rows', 'details'];
    const wrap = makeStyledTable(headers, sorted.slice(0, MAX_GROUP_RENDER_ROWS), (tr, group) => {
      groupFields.forEach((field) => {
        const td = document.createElement('td');
        td.textContent = formatCellText(group?.[field]);
        tr.appendChild(td);
      });

      URL_METRIC_FIELDS.forEach((field) => {
        const td = document.createElement('td');
        td.textContent = formatNumber(numberOrZero(group?.[field]));
        tr.appendChild(td);
      });

      const tdRows = document.createElement('td');
      tdRows.textContent = formatNumber(numberOrZero(group?.rows));
      tr.appendChild(tdRows);

      const tdDetails = document.createElement('td');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tool-pill tool-pill-button';
      btn.textContent = 'Details';
      btn.addEventListener('click', () => {
        drilldownGroup = group;
        renderDrilldown(lastUtm);
        markSessionDirty();
      });
      tdDetails.appendChild(btn);
      tr.appendChild(tdDetails);
    });

    utmOutputEl.appendChild(wrap);
  };

  const renderRawView = (data) => {
    utmOutputEl.replaceChildren();

    const query = getLocalFilterQuery();
    const rows = Array.isArray(data?.parsedRows) ? data.parsedRows : [];
    const filtered = applyLocalFilterRaw(rows, query);
    const sorted = sortList(filtered, 'raw');

    if (!sorted.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching URLs.';
      utmOutputEl.appendChild(p);
      return;
    }

    const meta = document.createElement('p');
    meta.className = 'contact-form-note';
    if (sorted.length > MAX_RAW_RENDER_ROWS) {
      meta.textContent = `Showing first ${formatNumber(MAX_RAW_RENDER_ROWS)} of ${formatNumber(sorted.length)} URLs.`;
      utmOutputEl.appendChild(meta);
    } else if (query) {
      meta.textContent = `Showing ${formatNumber(sorted.length)} URL(s) after client-side filter.`;
      utmOutputEl.appendChild(meta);
    }

    const headers = ['pathname', ...UTM_FIELDS, ...URL_METRIC_FIELDS, 'pageLocation'];
    const wrap = makeStyledTable(headers, sorted.slice(0, MAX_RAW_RENDER_ROWS), (tr, row) => {
      headers.forEach((header) => {
        const td = document.createElement('td');
        if (URL_METRIC_FIELDS.includes(header)) td.textContent = formatNumber(numberOrZero(row?.[header]));
        else if (header === 'pathname') td.textContent = row?.pathname || '';
        else if (UTM_FIELDS.includes(header)) td.textContent = formatCellText(row?.[header]);
        else td.textContent = String(row?.pageLocation || '');
        tr.appendChild(td);
      });
    });
    utmOutputEl.appendChild(wrap);
  };

  const renderDrilldown = (data) => {
    utmDrilldownEl.replaceChildren();
    if (!data || !Array.isArray(data.parsedRows) || !drilldownGroup) return;
    if (normalizeViewMode(viewModeEl.value) !== 'grouped') return;

    const groupFields = Array.isArray(data.groupFields) ? data.groupFields : getGroupFields();
    const matches = data.parsedRows.filter((row) => groupFields.every((field) => String(row?.[field] || '') === String(drilldownGroup?.[field] || '')));

    const header = document.createElement('div');
    header.className = 'tools-actions';

    const label = groupFields
      .map((field) => `${field}=${formatCellText(drilldownGroup?.[field])}`)
      .join(' · ');
    const title = document.createElement('span');
    title.className = 'tool-pill';
    title.textContent = label || 'Drilldown';
    header.appendChild(title);

    const totals = sumUrlMetrics(matches);
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `URLs: ${formatNumber(totals.rows)}` }));
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `Sessions: ${formatNumber(totals.sessions)}` }));
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `Users: ${formatNumber(totals.totalUsers)}` }));

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn-ghost';
    closeBtn.textContent = 'Close drilldown';
    closeBtn.addEventListener('click', () => {
      drilldownGroup = null;
      renderDrilldown(lastUtm);
      markSessionDirty();
    });
    header.appendChild(closeBtn);

    utmDrilldownEl.appendChild(header);

    if (!matches.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching URLs for this group.';
      utmDrilldownEl.appendChild(p);
      return;
    }

    const toolbar = document.createElement('div');
    toolbar.className = 'tools-actions';

    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.className = 'btn-secondary';
    exportBtn.textContent = 'Download drilldown CSV';
    exportBtn.addEventListener('click', () => {
      const headers = ['pathname', ...UTM_FIELDS, ...URL_METRIC_FIELDS, 'pageLocation'];
      const rows = matches.map((row) => {
        const out = {};
        headers.forEach((h) => { out[h] = row?.[h] ?? ''; });
        return out;
      });
      downloadCsv(rows, headers, `ga4-utm-drilldown_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`);
    });
    toolbar.appendChild(exportBtn);

    utmDrilldownEl.appendChild(toolbar);

    const headers = ['pathname', ...URL_METRIC_FIELDS, 'pageLocation'];
    const sortedMatches = matches.slice().sort((a, b) => numberOrZero(b.sessions) - numberOrZero(a.sessions));
    const wrap = makeStyledTable(headers, sortedMatches.slice(0, MAX_DRILLDOWN_RENDER_ROWS), (tr, row) => {
      headers.forEach((header) => {
        const td = document.createElement('td');
        if (URL_METRIC_FIELDS.includes(header)) td.textContent = formatNumber(numberOrZero(row?.[header]));
        else td.textContent = String(row?.[header] || '');
        tr.appendChild(td);
      });
    });
    utmDrilldownEl.appendChild(wrap);

    if (sortedMatches.length > MAX_DRILLDOWN_RENDER_ROWS) {
      const note = document.createElement('p');
      note.className = 'contact-form-note';
      note.textContent = `Showing first ${formatNumber(MAX_DRILLDOWN_RENDER_ROWS)} of ${formatNumber(sortedMatches.length)} URLs. Export for full list.`;
      utmDrilldownEl.appendChild(note);
    }
  };

  const renderUtmAll = () => {
    syncSortControls();
    renderUrlSummary(lastUtm);

    downloadGroupedBtn.disabled = utmBusy || !(lastUtm && Array.isArray(lastUtm.groups) && lastUtm.groups.length);
    downloadRawBtn.disabled = utmBusy || !(lastUtm && Array.isArray(lastUtm.parsedRows) && lastUtm.parsedRows.length);

    if (!lastUtm) {
      utmOutputEl.replaceChildren();
      utmDrilldownEl.replaceChildren();
      return;
    }

    if (normalizeViewMode(viewModeEl.value) === 'raw') {
      drilldownGroup = null;
      utmDrilldownEl.replaceChildren();
      renderRawView(lastUtm);
      return;
    }

    renderGroupedView(lastUtm);
    renderDrilldown(lastUtm);
  };

  const setUtmBusy = (busy) => {
    utmBusy = !!busy;
    runBtn.disabled = utmBusy;
    downloadGroupedBtn.disabled = utmBusy || !(lastUtm && Array.isArray(lastUtm.groups) && lastUtm.groups.length);
    downloadRawBtn.disabled = utmBusy || !(lastUtm && Array.isArray(lastUtm.parsedRows) && lastUtm.parsedRows.length);
  };

  const setExploreBusy = (busy) => {
    exploreBusy = !!busy;
    exploreRunBtn.disabled = exploreBusy;
    exploreDownloadBtn.disabled = exploreBusy || !(lastExplore && Array.isArray(lastExplore.rows) && lastExplore.rows.length);
  };

  const fetchGa4Report = async (payload) => {
    const token = getSavedToken();
    if (!token) throw new Error('Admin token required.');

    const res = await fetch('/api/ga4/report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(payload || {})
    });

    let data = null;
    try {
      data = await res.json();
    } catch {}

    if (!res.ok || !data || data.ok !== true) {
      const msg = String(data?.error || `Request failed (${res.status})`);
      throw new Error(msg);
    }

    return data;
  };

  const buildUtmFiltersPayload = () => ({
    match: String(matchTypeEl.value || 'exact').trim(),
    catchAll: !!catchAllEl.checked,
    utm_source: String(utmSourceEl.value || '').trim(),
    utm_medium: String(utmMediumEl.value || '').trim(),
    utm_campaign: String(utmCampaignEl.value || '').trim(),
    utm_content: String(utmContentEl.value || '').trim(),
    utm_term: String(utmTermEl.value || '').trim(),
    utm_id: String(utmIdEl.value || '').trim()
  });

  const runUtmReport = async () => {
    const propertyId = getPropertyId();
    if (!propertyId) {
      setStatus(utmStatusEl, 'GA4 property ID required.', 'error');
      return;
    }
    if (!startEl.value || !endEl.value) {
      setStatus(utmStatusEl, 'Start/end dates required.', 'error');
      return;
    }

    setUtmBusy(true);
    setStatus(utmStatusEl, 'Fetching GA4 data…');
    try {
      const data = await fetchGa4Report({
        kind: 'utm-urls',
        propertyId,
        startDate: String(startEl.value || '').trim(),
        endDate: String(endEl.value || '').trim(),
        filters: buildUtmFiltersPayload()
      });
      renderQuota(data.quota);

      const rows = Array.isArray(data.rows) ? data.rows : [];
      const parsedRows = buildParsedUrlRows(rows);
      const groupFields = getGroupFields();
      const groups = aggregateGroups(parsedRows, groupFields);

      lastUtm = {
        kind: 'utm-urls',
        propertyId: String(data.propertyId || propertyId).trim(),
        startDate: String(data.startDate || startEl.value || '').trim(),
        endDate: String(data.endDate || endEl.value || '').trim(),
        truncated: !!data.truncated,
        rowCount: numberOrZero(data.rowCount),
        returnedRows: numberOrZero(data.returnedRows),
        filters: buildUtmFiltersPayload(),
        groupFields,
        groups,
        parsedRows,
        totals: sumUrlMetrics(parsedRows)
      };

      drilldownGroup = null;
      renderUtmAll();

      const urlsLabel = lastUtm.returnedRows || parsedRows.length;
      const groupsLabel = groups.length;
      const truncatedNote = lastUtm.truncated ? ' (truncated)' : '';
      setStatus(utmStatusEl, `Done. ${formatNumber(urlsLabel)} URL(s), ${formatNumber(groupsLabel)} group(s)${truncatedNote}.`, 'success');
      markSessionDirty();
    } catch (err) {
      lastUtm = null;
      drilldownGroup = null;
      utmSummaryEl.replaceChildren();
      utmOutputEl.replaceChildren();
      utmDrilldownEl.replaceChildren();
      setStatus(utmStatusEl, err.message || 'Failed to fetch report.', 'error');
      markSessionDirty();
    } finally {
      setUtmBusy(false);
    }
  };

  const normalizeGa4FieldName = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return '';
    if (raw.length > 80) return '';
    if (!/^[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?$/.test(raw)) return '';
    return raw;
  };

  const splitListInput = (value) => String(value || '').split(/[,\n]/).map((part) => part.trim()).filter(Boolean);

  const readCustomList = (raw, kindLabel, { throwOnInvalid = true } = {}) => {
    const parts = splitListInput(raw);
    const invalid = [];
    const valid = [];
    parts.forEach((part) => {
      const normalized = normalizeGa4FieldName(part);
      if (!normalized) invalid.push(part);
      else valid.push(normalized);
    });
    if (invalid.length && throwOnInvalid) {
      const sample = invalid.slice(0, 3).join(', ');
      throw new Error(`Invalid ${kindLabel} name(s): ${sample}`);
    }
    return valid;
  };

  const readCheckedValues = (wrap) => {
    return $$('input[type="checkbox"]', wrap)
      .filter((el) => el && el.checked)
      .map((el) => normalizeGa4FieldName(el.value))
      .filter(Boolean);
  };

  const uniq = (list) => Array.from(new Set((Array.isArray(list) ? list : []).filter(Boolean)));

  const getExploreDimensions = ({ enforceLimit = true } = {}) => {
    const base = readCheckedValues(exploreDimensionsWrap);
    const extra = readCustomList(exploreCustomDimensionsEl.value, 'dimension', { throwOnInvalid: enforceLimit });
    const merged = uniq([...base, ...extra]);
    if (enforceLimit && merged.length > 9) throw new Error('Too many dimensions selected (max 9).');
    return merged;
  };

  const getExploreMetrics = ({ enforceLimit = true } = {}) => {
    const base = readCheckedValues(exploreMetricsWrap);
    const extra = readCustomList(exploreCustomMetricsEl.value, 'metric', { throwOnInvalid: enforceLimit });
    const merged = uniq([...base, ...extra]);
    if (enforceLimit && !merged.length) throw new Error('Select at least 1 metric.');
    if (enforceLimit && merged.length > 10) throw new Error('Too many metrics selected (max 10).');
    return merged;
  };

  const setExploreCheckboxes = (wrap, values) => {
    const set = new Set(Array.isArray(values) ? values : []);
    $$('input[type="checkbox"]', wrap).forEach((el) => {
      const name = normalizeGa4FieldName(el?.value);
      el.checked = !!name && set.has(name);
    });
  };

  const PRESETS = {
    custom: null,
    utm: {
      dimensions: ['sessionSource', 'sessionMedium', 'sessionCampaignName'],
      metrics: ['sessions', 'totalUsers', 'newUsers', 'engagedSessions', 'engagementRate', 'bounceRate', 'eventCount']
    },
    'utm-country': {
      dimensions: ['sessionSource', 'sessionMedium', 'sessionCampaignName', 'country'],
      metrics: ['sessions', 'totalUsers', 'newUsers', 'engagedSessions', 'engagementRate', 'bounceRate', 'eventCount']
    },
    'utm-device': {
      dimensions: ['sessionSource', 'sessionMedium', 'sessionCampaignName', 'deviceCategory'],
      metrics: ['sessions', 'totalUsers', 'newUsers', 'engagedSessions', 'engagementRate', 'bounceRate', 'eventCount']
    },
    'utm-demographics': {
      dimensions: ['sessionSource', 'sessionMedium', 'sessionCampaignName', 'userAgeBracket', 'userGender'],
      metrics: ['sessions', 'totalUsers', 'newUsers', 'engagedSessions', 'engagementRate', 'bounceRate', 'eventCount']
    }
  };

  const applyExplorePreset = (presetId) => {
    const key = String(presetId || '').trim();
    const preset = PRESETS[key];
    if (!preset) return;
    setExploreCheckboxes(exploreDimensionsWrap, preset.dimensions);
    setExploreCheckboxes(exploreMetricsWrap, preset.metrics);
    exploreCustomDimensionsEl.value = '';
    exploreCustomMetricsEl.value = '';
  };

  const syncExploreOrderByOptions = () => {
    const metrics = getExploreMetrics({ enforceLimit: false });
    const dimensions = getExploreDimensions({ enforceLimit: false });
    const options = [
      ...metrics.map((name) => ({ type: 'metric', value: name })),
      ...dimensions.map((name) => ({ type: 'dimension', value: name }))
    ];

    const current = String(exploreOrderByEl.value || '').trim();
    const allowed = new Set(options.map((opt) => opt.value));

    exploreOrderByEl.replaceChildren();
    options.forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.type === 'metric' ? getMetricLabel(opt.value) : getDimensionLabel(opt.value);
      exploreOrderByEl.appendChild(option);
    });

    if (current && allowed.has(current)) {
      exploreOrderByEl.value = current;
      return;
    }

    exploreOrderByEl.value = metrics[0] || dimensions[0] || '';
  };

  const getExploreFilterQuery = () => String(exploreFilterEl.value || '').trim().toLowerCase();

  const applyExploreFilter = (rows, dimensionNames, metricNames, query) => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return Array.isArray(rows) ? rows.slice() : [];

    const dims = Array.isArray(dimensionNames) ? dimensionNames : [];
    const metrics = Array.isArray(metricNames) ? metricNames : [];
    return (Array.isArray(rows) ? rows : []).filter((row) => {
      if (!row || typeof row !== 'object') return false;
      for (const dim of dims) {
        if (String(row[dim] || '').toLowerCase().includes(q)) return true;
      }
      for (const metric of metrics) {
        if (String(row[metric] ?? '').toLowerCase().includes(q)) return true;
      }
      return false;
    });
  };

  const formatMetricValue = (metricName, value) => {
    if (RATE_METRICS.has(metricName)) return formatPercent(value);
    return formatNumber(numberOrZero(value));
  };

  const getDimensionLabel = (name) => DIMENSION_LABELS[String(name || '').trim()] || String(name || '').trim();
  const getMetricLabel = (name) => METRIC_LABELS[String(name || '').trim()] || String(name || '').trim();

  const renderExploreSummary = (data, filteredRows) => {
    exploreSummaryEl.replaceChildren();
    if (!data || typeof data !== 'object') return;

    const rows = Array.isArray(filteredRows) ? filteredRows : (Array.isArray(data.rows) ? data.rows : []);
    const metricNames = Array.isArray(data.metricNames) ? data.metricNames : [];

    const totals = {};
    metricNames.forEach((metric) => {
      if (RATE_METRICS.has(metric)) return;
      totals[metric] = 0;
    });

    const hasSessions = metricNames.includes('sessions');
    let sessionsForRate = 0;
    let engagementRateWeighted = 0;
    let bounceRateWeighted = 0;

    rows.forEach((row) => {
      const sessions = numberOrZero(row.sessions);
      if (hasSessions) {
        sessionsForRate += sessions;
        engagementRateWeighted += sessions * numberOrZero(row.engagementRate);
        bounceRateWeighted += sessions * numberOrZero(row.bounceRate);
      }

      Object.keys(totals).forEach((metric) => {
        totals[metric] += numberOrZero(row[metric]);
      });
    });

    const pills = document.createElement('div');
    pills.className = 'tools-actions';

    const makePill = (label, value) => {
      const span = document.createElement('span');
      span.className = 'tool-pill';
      span.textContent = `${label}: ${value}`;
      return span;
    };

    pills.appendChild(makePill('Rows', formatNumber(rows.length)));

    Object.keys(totals).forEach((metric) => {
      const label = getMetricLabel(metric);
      pills.appendChild(makePill(label, formatMetricValue(metric, totals[metric])));
    });

    if (hasSessions && sessionsForRate > 0 && metricNames.includes('engagementRate')) {
      pills.appendChild(makePill(getMetricLabel('engagementRate'), formatPercent(engagementRateWeighted / sessionsForRate)));
    }
    if (hasSessions && sessionsForRate > 0 && metricNames.includes('bounceRate')) {
      pills.appendChild(makePill(getMetricLabel('bounceRate'), formatPercent(bounceRateWeighted / sessionsForRate)));
    }

    exploreSummaryEl.appendChild(pills);
  };

  const renderExploreTable = (data) => {
    exploreOutputEl.replaceChildren();
    if (!data || typeof data !== 'object') return;

    const dimensionNames = Array.isArray(data.dimensionNames) ? data.dimensionNames : [];
    const metricNames = Array.isArray(data.metricNames) ? data.metricNames : [];
    const rawRows = Array.isArray(data.rows) ? data.rows : [];
    const query = getExploreFilterQuery();

    const filtered = applyExploreFilter(rawRows, dimensionNames, metricNames, query);
    const rowsToRender = filtered.slice(0, MAX_INSIGHTS_RENDER_ROWS);

    renderExploreSummary(data, filtered);

    if (!filtered.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching rows.';
      exploreOutputEl.appendChild(p);
      return;
    }

    if (filtered.length > MAX_INSIGHTS_RENDER_ROWS) {
      const note = document.createElement('p');
      note.className = 'contact-form-note';
      note.textContent = `Showing first ${formatNumber(MAX_INSIGHTS_RENDER_ROWS)} of ${formatNumber(filtered.length)} rows. Export for full list.`;
      exploreOutputEl.appendChild(note);
    } else if (query) {
      const note = document.createElement('p');
      note.className = 'contact-form-note';
      note.textContent = `Showing ${formatNumber(filtered.length)} row(s) after client-side filter.`;
      exploreOutputEl.appendChild(note);
    }

    const displayHeaders = [...dimensionNames.map(getDimensionLabel), ...metricNames.map(getMetricLabel)];
    const wrap = makeStyledTable(displayHeaders, rowsToRender, (tr, row) => {
      dimensionNames.forEach((dim) => {
        const td = document.createElement('td');
        td.textContent = formatCellText(row?.[dim]);
        tr.appendChild(td);
      });
      metricNames.forEach((metric) => {
        const td = document.createElement('td');
        td.textContent = formatMetricValue(metric, row?.[metric]);
        tr.appendChild(td);
      });
    });
    exploreOutputEl.appendChild(wrap);
  };

  const runExploreReport = async () => {
    const propertyId = getPropertyId();
    if (!propertyId) {
      setStatus(exploreStatusEl, 'GA4 property ID required.', 'error');
      return;
    }
    if (!startEl.value || !endEl.value) {
      setStatus(exploreStatusEl, 'Start/end dates required.', 'error');
      return;
    }

    let dimensions;
    let metrics;
    try {
      dimensions = getExploreDimensions();
      metrics = getExploreMetrics();
    } catch (err) {
      setStatus(exploreStatusEl, err.message || 'Invalid explore configuration.', 'error');
      return;
    }

    const maxRows = Math.max(10, Math.min(10000, Math.floor(numberOrZero(exploreMaxRowsEl.value) || 200)));
    const orderBy = String(exploreOrderByEl.value || '').trim() || metrics[0] || '';
    const orderDir = normalizeSortDir(exploreDirEl.value);

    if (!metrics.length) {
      setStatus(exploreStatusEl, 'Select at least 1 metric.', 'error');
      return;
    }

    setExploreBusy(true);
    setStatus(exploreStatusEl, 'Fetching GA4 explore report…');
    try {
      const data = await fetchGa4Report({
        kind: 'explore',
        propertyId,
        startDate: String(startEl.value || '').trim(),
        endDate: String(endEl.value || '').trim(),
        dimensions,
        metrics,
        maxRows,
        orderBy,
        orderDir,
        filters: buildUtmFiltersPayload()
      });
      renderQuota(data.quota);

      lastExplore = {
        kind: 'explore',
        propertyId: String(data.propertyId || propertyId).trim(),
        startDate: String(data.startDate || startEl.value || '').trim(),
        endDate: String(data.endDate || endEl.value || '').trim(),
        dimensions,
        metrics,
        maxRows,
        orderBy,
        orderDir,
        filters: buildUtmFiltersPayload(),
        dimensionNames: Array.isArray(data.dimensionNames) ? data.dimensionNames : dimensions,
        metricNames: Array.isArray(data.metricNames) ? data.metricNames : metrics,
        rowCount: numberOrZero(data.rowCount),
        returnedRows: numberOrZero(data.returnedRows),
        truncated: !!data.truncated,
        rows: Array.isArray(data.rows) ? data.rows : []
      };

      renderExploreTable(lastExplore);
      exploreDownloadBtn.disabled = exploreBusy || !(lastExplore.rows && lastExplore.rows.length);

      const trunc = lastExplore.truncated ? ' (truncated)' : '';
      setStatus(exploreStatusEl, `Done. ${formatNumber(lastExplore.returnedRows || lastExplore.rows.length)} row(s)${trunc}.`, 'success');
      markSessionDirty();
    } catch (err) {
      lastExplore = null;
      exploreSummaryEl.replaceChildren();
      exploreOutputEl.replaceChildren();
      exploreDownloadBtn.disabled = true;
      setStatus(exploreStatusEl, err.message || 'Failed to fetch explore report.', 'error');
      markSessionDirty();
    } finally {
      setExploreBusy(false);
    }
  };

  const checkAccess = async () => {
    const propertyId = getPropertyId();
    if (!propertyId) {
      setStatus(accessStatusEl, 'GA4 property ID required.', 'error');
      return;
    }

    setStatus(accessStatusEl, 'Checking access…');
    try {
      const data = await fetchGa4Report({
        kind: 'ping',
        propertyId,
        startDate: String(startEl.value || '').trim(),
        endDate: String(endEl.value || '').trim()
      });
      renderQuota(data.quota);
      const rows = Array.isArray(data.rows) ? data.rows : [];
      setStatus(accessStatusEl, rows.length ? 'Access OK. GA4 report returned data.' : 'Access OK. No rows returned for the selected range.', 'success');
    } catch (err) {
      setStatus(accessStatusEl, err.message || 'Access check failed.', 'error');
    }
  };

  accessForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const token = String(tokenInput.value || '').trim();
    if (!token) {
      updateAccessMeta();
      setStatus(accessStatusEl, getSavedToken() ? 'Token already stored.' : 'Paste your admin token to unlock this tool.', getSavedToken() ? 'success' : 'error');
      return;
    }
    saveToken(token);
    updateAccessMeta();
    tokenInput.value = '';
    setStatus(accessStatusEl, 'Token saved.', 'success');
    markSessionDirty();
  });

  forgetTokenBtn.addEventListener('click', () => {
    saveToken('');
    updateAccessMeta();
    tokenInput.value = '';
    setStatus(accessStatusEl, 'Token forgotten on this device.', 'success');
    renderQuota(null);
    markSessionDirty();
  });

  profileSelect.addEventListener('change', () => {
    syncProfileInputs();
    markSessionDirty();
  });

  saveProfileBtn.addEventListener('click', () => {
    const label = String(profileLabelInput.value || '').trim();
    const propertyId = getPropertyId();
    if (!label || !propertyId) {
      setStatus(accessStatusEl, 'Profile label + GA4 property ID are required.', 'error');
      return;
    }

    const profiles = loadProfiles();
    const existing = profiles.find((p) => p.propertyId === propertyId);
    if (existing) existing.label = label;
    else profiles.push({ label, propertyId });

    const saved = saveProfiles(profiles);
    renderProfileSelect(saved);
    profileSelect.value = propertyId;
    setActiveProfile(propertyId);
    setStatus(accessStatusEl, 'Profile saved.', 'success');
    markSessionDirty();
  });

  deleteProfileBtn.addEventListener('click', () => {
    const selected = normalizePropertyId(profileSelect.value);
    if (!selected) {
      setStatus(accessStatusEl, 'Select a saved profile to delete.', 'error');
      return;
    }

    const profiles = loadProfiles().filter((p) => p.propertyId !== selected);
    saveProfiles(profiles);
    renderProfileSelect(profiles);
    profileSelect.value = '';
    setActiveProfile('');
    setStatus(accessStatusEl, 'Profile deleted.', 'success');
    markSessionDirty();
  });

  checkAccessBtn.addEventListener('click', () => {
    checkAccess();
  });

  scopeForm?.addEventListener('submit', (event) => {
    event.preventDefault();
  });

  filtersForm?.addEventListener('submit', (event) => {
    event.preventDefault();
  });

  utmForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runUtmReport();
  });

  viewModeEl.addEventListener('change', () => {
    renderUtmAll();
    markSessionDirty();
  });

  sortFieldEl.addEventListener('change', () => {
    const mode = normalizeViewMode(viewModeEl.value);
    const field = String(sortFieldEl.value || '').trim() || 'sessions';
    sortState[mode].field = field;
    if (!sortState[mode].dir) {
      sortState[mode].dir = getFieldType(mode, field) === 'number' ? 'desc' : 'asc';
    }
    renderUtmAll();
    markSessionDirty();
  });

  sortDirEl.addEventListener('change', () => {
    const mode = normalizeViewMode(viewModeEl.value);
    sortState[mode].dir = normalizeSortDir(sortDirEl.value);
    renderUtmAll();
    markSessionDirty();
  });

  let filterDebounceTimer = 0;
  localFilterEl.addEventListener('input', () => {
    try { window.clearTimeout(filterDebounceTimer); } catch {}
    filterDebounceTimer = window.setTimeout(() => {
      renderUtmAll();
      markSessionDirty();
    }, 120);
  });

  $$('input[name="ga4-group"]', main).forEach((el) => {
    el.addEventListener('change', () => {
      if (!lastUtm || !Array.isArray(lastUtm.parsedRows) || !lastUtm.parsedRows.length) {
        markSessionDirty();
        return;
      }
      const groupFields = getGroupFields();
      lastUtm.groupFields = groupFields;
      lastUtm.groups = aggregateGroups(lastUtm.parsedRows, groupFields);
      drilldownGroup = null;
      renderUtmAll();
      markSessionDirty();
    });
  });

  matchTypeEl.addEventListener('change', markSessionDirty);
  catchAllEl.addEventListener('change', markSessionDirty);
  utmSourceEl.addEventListener('input', markSessionDirty);
  utmMediumEl.addEventListener('input', markSessionDirty);
  utmCampaignEl.addEventListener('input', markSessionDirty);
  utmContentEl.addEventListener('input', markSessionDirty);
  utmTermEl.addEventListener('input', markSessionDirty);
  utmIdEl.addEventListener('input', markSessionDirty);

  downloadGroupedBtn.addEventListener('click', () => {
    if (!lastUtm || !Array.isArray(lastUtm.groups) || !lastUtm.groups.length) return;

    const groupFields = Array.isArray(lastUtm.groupFields) ? lastUtm.groupFields : getGroupFields();
    const query = normalizeViewMode(viewModeEl.value) === 'grouped' ? getLocalFilterQuery() : '';
    const filtered = applyLocalFilterGrouped(lastUtm.groups, groupFields, query);
    const sorted = sortList(filtered, 'grouped');

    const headers = [...groupFields, ...URL_METRIC_FIELDS, 'rows'];
    const rows = sorted.map((row) => {
      const out = {};
      groupFields.forEach((field) => { out[field] = row?.[field] || ''; });
      URL_METRIC_FIELDS.forEach((field) => { out[field] = numberOrZero(row?.[field]); });
      out.rows = numberOrZero(row?.rows);
      return out;
    });

    downloadCsv(rows, headers, `ga4-utm-grouped_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`);
  });

  downloadRawBtn.addEventListener('click', () => {
    if (!lastUtm || !Array.isArray(lastUtm.parsedRows) || !lastUtm.parsedRows.length) return;

    const query = normalizeViewMode(viewModeEl.value) === 'raw' ? getLocalFilterQuery() : '';
    const filtered = applyLocalFilterRaw(lastUtm.parsedRows, query);
    const sorted = sortList(filtered, 'raw');

    const headers = ['pathname', ...UTM_FIELDS, ...URL_METRIC_FIELDS, 'pageLocation'];
    const rows = sorted.map((row) => {
      const out = {};
      headers.forEach((header) => {
        if (URL_METRIC_FIELDS.includes(header)) out[header] = numberOrZero(row?.[header]);
        else out[header] = row?.[header] ?? '';
      });
      return out;
    });

    downloadCsv(rows, headers, `ga4-utm-raw_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`);
  });

  exploreForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runExploreReport();
  });

  explorePresetEl.addEventListener('change', () => {
    if (String(explorePresetEl.value || '') !== 'custom') {
      applyExplorePreset(explorePresetEl.value);
    }
    syncExploreOrderByOptions();
    markSessionDirty();
  });

  const onExploreConfigChange = () => {
    syncExploreOrderByOptions();
    markSessionDirty();
  };

  $$('input[type="checkbox"]', exploreDimensionsWrap).forEach((el) => {
    el.addEventListener('change', onExploreConfigChange);
  });
  $$('input[type="checkbox"]', exploreMetricsWrap).forEach((el) => {
    el.addEventListener('change', onExploreConfigChange);
  });

  let exploreFieldsTimer = 0;
  const onExploreFieldInput = () => {
    try { window.clearTimeout(exploreFieldsTimer); } catch {}
    exploreFieldsTimer = window.setTimeout(() => {
      syncExploreOrderByOptions();
      markSessionDirty();
    }, 140);
  };
  exploreCustomDimensionsEl.addEventListener('input', onExploreFieldInput);
  exploreCustomMetricsEl.addEventListener('input', onExploreFieldInput);

  exploreMaxRowsEl.addEventListener('input', markSessionDirty);
  exploreOrderByEl.addEventListener('change', markSessionDirty);
  exploreDirEl.addEventListener('change', markSessionDirty);

  let exploreFilterTimer = 0;
  exploreFilterEl.addEventListener('input', () => {
    try { window.clearTimeout(exploreFilterTimer); } catch {}
    exploreFilterTimer = window.setTimeout(() => {
      renderExploreTable(lastExplore);
      markSessionDirty();
    }, 120);
  });

  exploreDownloadBtn.addEventListener('click', () => {
    if (!lastExplore || !Array.isArray(lastExplore.rows) || !lastExplore.rows.length) return;

    const dimensionNames = Array.isArray(lastExplore.dimensionNames) ? lastExplore.dimensionNames : [];
    const metricNames = Array.isArray(lastExplore.metricNames) ? lastExplore.metricNames : [];
    const query = getExploreFilterQuery();
    const filtered = applyExploreFilter(lastExplore.rows, dimensionNames, metricNames, query);

    const headers = [...dimensionNames, ...metricNames];
    const rows = filtered.map((row) => {
      const out = {};
      dimensionNames.forEach((dim) => { out[dim] = row?.[dim] ?? ''; });
      metricNames.forEach((metric) => { out[metric] = row?.[metric] ?? 0; });
      return out;
    });

    downloadCsv(rows, headers, `ga4-explore_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`);
  });

  const buildAccessSummary = () => {
    const bits = [];
    if (getPropertyId()) bits.push(`Property ${getPropertyId()}`);
    if (startEl.value && endEl.value) bits.push(`${startEl.value} → ${endEl.value}`);
    return bits.join(' · ');
  };

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const utmSummary = String(utmStatusEl.textContent || '').trim();
    const exploreSummary = String(exploreStatusEl.textContent || '').trim();
    const summary = [utmSummary, exploreSummary].filter(Boolean).join(' | ').trim();
    payload.outputSummary = summary;

    const groupFields = getGroupFields();
    payload.inputs = {
      'Property ID': getPropertyId(),
      'Date range': startEl.value && endEl.value ? `${startEl.value} → ${endEl.value}` : '',
      'UTM match': String(matchTypeEl.value || 'exact'),
      'Catch all': catchAllEl.checked ? 'Yes' : 'No',
      'utm_source': String(utmSourceEl.value || '').trim(),
      'utm_medium': String(utmMediumEl.value || '').trim(),
      'utm_campaign': String(utmCampaignEl.value || '').trim(),
      'utm_content': String(utmContentEl.value || '').trim(),
      'utm_term': String(utmTermEl.value || '').trim(),
      'utm_id': String(utmIdEl.value || '').trim(),
      'Group by': groupFields.join(', '),
      'UTM view': normalizeViewMode(viewModeEl.value),
      'UTM sort': `${String(sortFieldEl.value || '').trim()} (${String(sortDirEl.value || '').trim()})`,
      'UTM filter': String(localFilterEl.value || '').trim(),
      'Explore preset': String(explorePresetEl.value || '').trim(),
      'Explore dimensions': getExploreDimensions({ enforceLimit: false }).join(', '),
      'Explore metrics': getExploreMetrics({ enforceLimit: false }).join(', '),
      'Explore custom dimensions': String(exploreCustomDimensionsEl.value || '').trim(),
      'Explore custom metrics': String(exploreCustomMetricsEl.value || '').trim(),
      'Explore max rows': String(exploreMaxRowsEl.value || '').trim(),
      'Explore order by': String(exploreOrderByEl.value || '').trim(),
      'Explore dir': String(exploreDirEl.value || '').trim(),
      'Explore filter': String(exploreFilterEl.value || '').trim()
    };

    payload.output = {
      kind: 'json',
      summary,
      access: {
        note: buildAccessSummary()
      },
      utm: lastUtm ? {
        summary: utmSummary,
        propertyId: lastUtm.propertyId || '',
        startDate: lastUtm.startDate || '',
        endDate: lastUtm.endDate || '',
        totals: lastUtm.totals || null,
        groupFields: lastUtm.groupFields || [],
        groups: Array.isArray(lastUtm.groups) ? lastUtm.groups.slice(0, 200) : [],
        rawRows: Array.isArray(lastUtm.parsedRows) ? lastUtm.parsedRows.slice(0, 120) : []
      } : null,
      explore: lastExplore ? {
        summary: exploreSummary,
        propertyId: lastExplore.propertyId || '',
        startDate: lastExplore.startDate || '',
        endDate: lastExplore.endDate || '',
        dimensionNames: Array.isArray(lastExplore.dimensionNames) ? lastExplore.dimensionNames : [],
        metricNames: Array.isArray(lastExplore.metricNames) ? lastExplore.metricNames : [],
        rows: Array.isArray(lastExplore.rows) ? lastExplore.rows.slice(0, 250) : []
      } : null
    };
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const snapshot = detail?.snapshot;
    const output = snapshot?.output;
    if (!output || typeof output !== 'object') return;
    if (String(output.kind || '') !== 'json') return;

    const utm = output?.utm;
    if (utm && typeof utm === 'object') {
      const parsedRows = Array.isArray(utm.rawRows) ? utm.rawRows : [];
      const groupFields = Array.isArray(utm.groupFields) ? utm.groupFields : getGroupFields();
      const groups = Array.isArray(utm.groups) ? utm.groups : aggregateGroups(parsedRows, groupFields);
      const totals = utm.totals || (parsedRows.length ? sumUrlMetrics(parsedRows) : null);
      lastUtm = {
        kind: 'utm-urls',
        propertyId: String(utm.propertyId || '').trim(),
        startDate: String(utm.startDate || '').trim(),
        endDate: String(utm.endDate || '').trim(),
        groupFields,
        groups,
        parsedRows,
        totals
      };
    } else {
      lastUtm = null;
    }

    const explore = output?.explore;
    if (explore && typeof explore === 'object') {
      lastExplore = {
        kind: 'explore',
        propertyId: String(explore.propertyId || '').trim(),
        startDate: String(explore.startDate || '').trim(),
        endDate: String(explore.endDate || '').trim(),
        dimensionNames: Array.isArray(explore.dimensionNames) ? explore.dimensionNames : [],
        metricNames: Array.isArray(explore.metricNames) ? explore.metricNames : [],
        rows: Array.isArray(explore.rows) ? explore.rows : []
      };
    } else {
      lastExplore = null;
    }

    drilldownGroup = null;
    renderUtmAll();
    syncExploreOrderByOptions();
    renderExploreTable(lastExplore);

    if (utm && typeof utm.summary === 'string') setStatus(utmStatusEl, utm.summary);
    if (explore && typeof explore.summary === 'string') setStatus(exploreStatusEl, explore.summary);
  });

  initDefaultDates();
  renderProfileSelect(loadProfiles());
  syncProfileInputs();
  updateAccessMeta();
  renderUtmAll();
  syncExploreOrderByOptions();
  renderExploreTable(lastExplore);
})();
