(() => {
  'use strict';

  const TOOL_ID = 'ga4-utm-performance';
  const TOKEN_STORAGE_KEY = 'ga4UtmPerformance:adminToken';
  const MAX_GROUP_RENDER_ROWS = 5000;
  const MAX_RAW_RENDER_ROWS = 5000;
  const MAX_DRILLDOWN_RENDER_ROWS = 500;

  const UTM_FIELDS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'utm_id'];
  const METRIC_FIELDS = ['sessions', 'totalUsers', 'eventCount'];

  const $ = (sel) => document.querySelector(sel);

  const authForm = $('#ga4-auth-form');
  const tokenInput = $('#ga4-admin-token');
  const forgetBtn = $('#ga4-forget-token');
  const accessMetaEl = $('#ga4-access-meta');

  const reportForm = $('#ga4-report-form');
  const propertyEl = $('#ga4-property-id');
  const startEl = $('#ga4-start');
  const endEl = $('#ga4-end');
  const catchAllEl = $('#ga4-catch-all');
  const matchTypeEl = $('#ga4-utm-match');
  const utmSourceEl = $('#ga4-utm-source');
  const utmMediumEl = $('#ga4-utm-medium');
  const utmCampaignEl = $('#ga4-utm-campaign');
  const utmContentEl = $('#ga4-utm-content');
  const utmTermEl = $('#ga4-utm-term');
  const utmIdEl = $('#ga4-utm-id');

  const runBtn = $('#ga4-run');
  const downloadBtn = $('#ga4-download-csv');
  const downloadRawBtn = $('#ga4-download-raw-csv');
  const viewModeEl = $('#ga4-view-mode');
  const sortFieldEl = $('#ga4-sort-field');
  const sortDirEl = $('#ga4-sort-dir');
  const localFilterEl = $('#ga4-local-filter');
  const statusEl = $('#ga4-status');
  const summaryEl = $('#ga4-summary');
  const outputEl = $('#ga4-output');
  const drilldownEl = $('#ga4-drilldown');

  if (!authForm || !tokenInput || !forgetBtn || !reportForm || !startEl || !endEl || !statusEl || !outputEl) return;
  if (!downloadBtn || !downloadRawBtn || !viewModeEl || !sortFieldEl || !sortDirEl || !localFilterEl || !summaryEl || !drilldownEl) return;
  if (!matchTypeEl) return;

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
  let lastData = null;
  let drilldownGroup = null;
  let isBusy = false;

  const normalizeViewMode = (value) => {
    const raw = String(value || '').trim().toLowerCase();
    if (raw === 'raw') return 'raw';
    return 'grouped';
  };

  const sortState = {
    grouped: { field: 'sessions', dir: 'desc' },
    raw: { field: 'sessions', dir: 'desc' },
    drilldown: { field: 'sessions', dir: 'desc' }
  };

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

  const setStatus = (text, tone = '') => {
    statusEl.textContent = String(text || '');
    statusEl.dataset.tone = tone;
  };

  const updateAccessMeta = () => {
    if (!accessMetaEl) return;
    accessMetaEl.textContent = getSavedToken() ? 'Token stored' : 'Token required';
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
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

  const parseUtm = (pageLocation) => {
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
    const inputs = Array.from(document.querySelectorAll('input[name="ga4-group"]'));
    const fields = inputs.filter((el) => el && el.checked).map((el) => String(el.value || '').trim()).filter(Boolean);
    return fields.length ? fields : ['utm_source', 'utm_medium', 'utm_campaign'];
  };

  const numberOrZero = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  const getActiveViewMode = () => normalizeViewMode(viewModeEl?.value);

  const getLocalFilterQuery = () => String(localFilterEl?.value || '').trim().toLowerCase();

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
      ...UTM_FIELDS.map((f) => ({ value: f, label: f, type: 'string' }))
    ];
  };

  const getFieldType = (mode, field) => {
    const options = getSortOptions(mode);
    const match = options.find((opt) => opt.value === field);
    return match ? match.type : 'string';
  };

  const normalizeSortDir = (value) => {
    const raw = String(value || '').trim().toLowerCase();
    if (raw === 'asc') return 'asc';
    return 'desc';
  };

  const syncSortControls = () => {
    const mode = getActiveViewMode();
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

  const buildParsedRows = (rows) => {
    const list = Array.isArray(rows) ? rows : [];
    return list
      .map((row) => {
        const pageLocation = String(row?.pageLocation || '').trim();
        if (!pageLocation) return null;

        const url = safeUrl(pageLocation);
        const pathname = url ? String(url.pathname || '') : '';
        const utm = parseUtm(pageLocation);

        const out = {
          pageLocation,
          pathname,
          sessions: numberOrZero(row?.sessions),
          totalUsers: numberOrZero(row?.totalUsers),
          eventCount: numberOrZero(row?.eventCount)
        };
        UTM_FIELDS.forEach((f) => { out[f] = String(utm[f] || ''); });
        return out;
      })
      .filter(Boolean);
  };

  const aggregateGroups = (parsedRows, groupFields) => {
    const fields = Array.isArray(groupFields) ? groupFields : [];
    const map = new Map();

    (Array.isArray(parsedRows) ? parsedRows : []).forEach((row) => {
      const key = fields.map((f) => String(row?.[f] || '')).join('\u0000');
      let entry = map.get(key);
      if (!entry) {
        entry = { key, sessions: 0, totalUsers: 0, eventCount: 0, rows: 0 };
        fields.forEach((f) => { entry[f] = String(row?.[f] || ''); });
        map.set(key, entry);
      }
      entry.sessions += numberOrZero(row?.sessions);
      entry.totalUsers += numberOrZero(row?.totalUsers);
      entry.eventCount += numberOrZero(row?.eventCount);
      entry.rows += 1;
    });

    return Array.from(map.values());
  };

  const sumMetrics = (parsedRows) => {
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
      return fields.some((f) => String(group[f] || '').toLowerCase().includes(q));
    });
  };

  const applyLocalFilterRaw = (rows, query) => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return Array.isArray(rows) ? rows.slice() : [];
    return (Array.isArray(rows) ? rows : []).filter((row) => {
      if (!row || typeof row !== 'object') return false;
      if (String(row.pageLocation || '').toLowerCase().includes(q)) return true;
      if (String(row.pathname || '').toLowerCase().includes(q)) return true;
      return UTM_FIELDS.some((f) => String(row[f] || '').toLowerCase().includes(q));
    });
  };

  const renderSummary = (data) => {
    summaryEl.replaceChildren();
    if (!data || typeof data !== 'object') return;

    const totals = data.totals || sumMetrics(data.parsedRows);
    const pills = document.createElement('div');
    pills.className = 'tools-actions';

    const makePill = (label, value) => {
      const span = document.createElement('span');
      span.className = 'tool-pill';
      span.textContent = `${label}: ${value}`;
      return span;
    };

    pills.appendChild(makePill('URLs', Number(totals.rows || 0).toLocaleString('en-US')));
    pills.appendChild(makePill('Sessions', Number(totals.sessions || 0).toLocaleString('en-US')));
    pills.appendChild(makePill('Users', Number(totals.totalUsers || 0).toLocaleString('en-US')));
    pills.appendChild(makePill('Events', Number(totals.eventCount || 0).toLocaleString('en-US')));
    if (Array.isArray(data.groups)) {
      pills.appendChild(makePill('Groups', Number(data.groups.length || 0).toLocaleString('en-US')));
    }

    summaryEl.appendChild(pills);
  };

  const renderGroupedView = (data) => {
    outputEl.replaceChildren();

    const groupFields = Array.isArray(data?.groupFields) ? data.groupFields : getGroupFields();
    const groups = Array.isArray(data?.groups) ? data.groups : [];
    const query = getLocalFilterQuery();

    const filtered = applyLocalFilterGrouped(groups, groupFields, query);
    const sorted = sortList(filtered, 'grouped');

    if (!sorted.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching groups.';
      outputEl.appendChild(p);
      return;
    }

    const meta = document.createElement('p');
    meta.className = 'tools-helper';
    if (sorted.length > MAX_GROUP_RENDER_ROWS) {
      meta.textContent = `Showing first ${MAX_GROUP_RENDER_ROWS.toLocaleString('en-US')} of ${sorted.length.toLocaleString('en-US')} groups.`;
      outputEl.appendChild(meta);
    } else if (query) {
      meta.textContent = `Showing ${sorted.length.toLocaleString('en-US')} group(s) after client-side filter.`;
      outputEl.appendChild(meta);
    }

    const headers = [...groupFields, ...METRIC_FIELDS, 'rows', 'details'];

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((h) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = h;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    sorted.slice(0, MAX_GROUP_RENDER_ROWS).forEach((group) => {
      const tr = document.createElement('tr');

      groupFields.forEach((field) => {
        const td = document.createElement('td');
        td.textContent = formatCellText(group?.[field]);
        tr.appendChild(td);
      });

      METRIC_FIELDS.forEach((field) => {
        const td = document.createElement('td');
        td.textContent = numberOrZero(group?.[field]).toLocaleString('en-US');
        tr.appendChild(td);
      });

      const tdRows = document.createElement('td');
      tdRows.textContent = numberOrZero(group?.rows).toLocaleString('en-US');
      tr.appendChild(tdRows);

      const tdDetails = document.createElement('td');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tool-pill tool-pill-button';
      btn.textContent = 'Details';
      btn.addEventListener('click', () => {
        drilldownGroup = group;
        renderDrilldown(lastData);
        markSessionDirty();
      });
      tdDetails.appendChild(btn);
      tr.appendChild(tdDetails);

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    outputEl.appendChild(table);
  };

  const renderRawView = (data) => {
    outputEl.replaceChildren();

    const query = getLocalFilterQuery();
    const rows = Array.isArray(data?.parsedRows) ? data.parsedRows : [];
    const filtered = applyLocalFilterRaw(rows, query);
    const sorted = sortList(filtered, 'raw');

    if (!sorted.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching URLs.';
      outputEl.appendChild(p);
      return;
    }

    const meta = document.createElement('p');
    meta.className = 'tools-helper';
    if (sorted.length > MAX_RAW_RENDER_ROWS) {
      meta.textContent = `Showing first ${MAX_RAW_RENDER_ROWS.toLocaleString('en-US')} of ${sorted.length.toLocaleString('en-US')} URLs.`;
      outputEl.appendChild(meta);
    } else if (query) {
      meta.textContent = `Showing ${sorted.length.toLocaleString('en-US')} URL(s) after client-side filter.`;
      outputEl.appendChild(meta);
    }

    const headers = ['pathname', ...UTM_FIELDS, ...METRIC_FIELDS, 'pageLocation'];

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((h) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = h;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    sorted.slice(0, MAX_RAW_RENDER_ROWS).forEach((row) => {
      const tr = document.createElement('tr');
      headers.forEach((h) => {
        const td = document.createElement('td');
        if (METRIC_FIELDS.includes(h)) td.textContent = numberOrZero(row?.[h]).toLocaleString('en-US');
        else if (h === 'pathname') td.textContent = row?.pathname || '';
        else if (UTM_FIELDS.includes(h)) td.textContent = formatCellText(row?.[h]);
        else td.textContent = String(row?.pageLocation || '');
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    outputEl.appendChild(table);
  };

  const renderDrilldown = (data) => {
    drilldownEl.replaceChildren();
    if (!data || !Array.isArray(data.parsedRows) || !drilldownGroup) return;
    if (getActiveViewMode() !== 'grouped') return;

    const groupFields = Array.isArray(data.groupFields) ? data.groupFields : getGroupFields();
    const matches = data.parsedRows.filter((row) => {
      return groupFields.every((f) => String(row?.[f] || '') === String(drilldownGroup?.[f] || ''));
    });

    const header = document.createElement('div');
    header.className = 'tools-actions';

    const label = groupFields
      .map((f) => `${f}=${formatCellText(drilldownGroup?.[f])}`)
      .join(' · ');
    const title = document.createElement('span');
    title.className = 'tool-pill';
    title.textContent = label || 'Drilldown';
    header.appendChild(title);

    const totals = sumMetrics(matches);
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `URLs: ${totals.rows.toLocaleString('en-US')}` }));
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `Sessions: ${totals.sessions.toLocaleString('en-US')}` }));
    header.appendChild(Object.assign(document.createElement('span'), { className: 'tool-pill', textContent: `Users: ${totals.totalUsers.toLocaleString('en-US')}` }));

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn-ghost';
    closeBtn.textContent = 'Close drilldown';
    closeBtn.addEventListener('click', () => {
      drilldownGroup = null;
      renderDrilldown(lastData);
      markSessionDirty();
    });
    header.appendChild(closeBtn);

    drilldownEl.appendChild(header);

    if (!matches.length) {
      const p = document.createElement('p');
      p.textContent = 'No matching URLs for this group.';
      drilldownEl.appendChild(p);
      return;
    }

    const sortWrap = document.createElement('div');
    sortWrap.className = 'tools-actions';

    const sortLabel = document.createElement('label');
    sortLabel.textContent = 'Sort drilldown';
    sortLabel.htmlFor = 'ga4-drilldown-sort-field';
    sortWrap.appendChild(sortLabel);

    const sortSelect = document.createElement('select');
    sortSelect.id = 'ga4-drilldown-sort-field';
    const drilldownOptions = [
      { value: 'sessions', label: 'Sessions', type: 'number' },
      { value: 'totalUsers', label: 'Users', type: 'number' },
      { value: 'eventCount', label: 'Events', type: 'number' },
      { value: 'pathname', label: 'Path', type: 'string' },
      { value: 'pageLocation', label: 'URL', type: 'string' }
    ];
    drilldownOptions.forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.label;
      sortSelect.appendChild(option);
    });
    sortSelect.value = sortState.drilldown.field || 'sessions';
    sortWrap.appendChild(sortSelect);

    const dirSelect = document.createElement('select');
    const dirDesc = document.createElement('option');
    dirDesc.value = 'desc';
    dirDesc.textContent = 'Desc';
    const dirAsc = document.createElement('option');
    dirAsc.value = 'asc';
    dirAsc.textContent = 'Asc';
    dirSelect.appendChild(dirDesc);
    dirSelect.appendChild(dirAsc);
    dirSelect.value = sortState.drilldown.dir || 'desc';
    sortWrap.appendChild(dirSelect);

    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.className = 'btn-secondary';
    exportBtn.textContent = 'Download drilldown CSV';
    exportBtn.addEventListener('click', () => {
      const headers = ['pathname', ...UTM_FIELDS, ...METRIC_FIELDS, 'pageLocation'];
      const rows = matches.map((row) => {
        const out = {};
        headers.forEach((h) => { out[h] = row?.[h] ?? ''; });
        return out;
      });
      const csv = toCsv(rows, headers);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ga4-utm-drilldown_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    });
    sortWrap.appendChild(exportBtn);

    drilldownEl.appendChild(sortWrap);

    const doRenderTable = () => {
      const field = String(sortSelect.value || '').trim() || 'sessions';
      sortState.drilldown.field = field;
      sortState.drilldown.dir = normalizeSortDir(dirSelect.value);

      const type = drilldownOptions.find((opt) => opt.value === field)?.type || 'string';
      const sorted = matches.slice().sort((a, b) => {
        if (type === 'number') {
          const av = numberOrZero(a?.[field]);
          const bv = numberOrZero(b?.[field]);
          return sortState.drilldown.dir === 'asc' ? (av - bv) : (bv - av);
        }
        const as = String(a?.[field] || '').toLowerCase();
        const bs = String(b?.[field] || '').toLowerCase();
        const cmp = as.localeCompare(bs);
        return sortState.drilldown.dir === 'asc' ? cmp : -cmp;
      });

      const existing = drilldownEl.querySelector('table[data-ga4-drilldown="true"]');
      if (existing) existing.remove();

      const table = document.createElement('table');
      table.dataset.ga4Drilldown = 'true';
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      ['sessions', 'totalUsers', 'eventCount', 'pathname', 'pageLocation'].forEach((h) => {
        const th = document.createElement('th');
        th.scope = 'col';
        th.textContent = h;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      sorted.slice(0, MAX_DRILLDOWN_RENDER_ROWS).forEach((row) => {
        const tr = document.createElement('tr');
        ['sessions', 'totalUsers', 'eventCount', 'pathname', 'pageLocation'].forEach((h) => {
          const td = document.createElement('td');
          if (METRIC_FIELDS.includes(h)) td.textContent = numberOrZero(row?.[h]).toLocaleString('en-US');
          else td.textContent = String(row?.[h] || '');
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      drilldownEl.appendChild(table);
    };

    sortSelect.addEventListener('change', doRenderTable);
    dirSelect.addEventListener('change', doRenderTable);
    doRenderTable();
  };

  const renderAll = () => {
    const mode = getActiveViewMode();
    syncSortControls();
    renderSummary(lastData);
    downloadBtn.disabled = isBusy || !(lastData && Array.isArray(lastData.groups) && lastData.groups.length);
    downloadRawBtn.disabled = isBusy || !(lastData && Array.isArray(lastData.parsedRows) && lastData.parsedRows.length);
    if (!lastData) {
      outputEl.replaceChildren();
      drilldownEl.replaceChildren();
      return;
    }
    if (mode === 'raw') {
      drilldownGroup = null;
      drilldownEl.replaceChildren();
      renderRawView(lastData);
    } else {
      renderGroupedView(lastData);
      renderDrilldown(lastData);
    }
  };

  const setBusy = (busy) => {
    isBusy = !!busy;
    if (runBtn) runBtn.disabled = busy;
    if (downloadBtn) downloadBtn.disabled = busy || !(lastData && Array.isArray(lastData.groups) && lastData.groups.length);
    if (downloadRawBtn) downloadRawBtn.disabled = busy || !(lastData && Array.isArray(lastData.parsedRows) && lastData.parsedRows.length);
    reportForm.querySelectorAll('input, button').forEach((el) => {
      if (!el) return;
      if (el === downloadBtn || el === downloadRawBtn || el === runBtn || el === forgetBtn) return;
      if (busy) el.setAttribute('aria-busy', 'true');
      else el.removeAttribute('aria-busy');
    });
  };

  const fetchReport = async (payload) => {
    const res = await fetch('/api/ga4/report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getSavedToken()}`
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

  const buildPayload = () => ({
    propertyId: String(propertyEl?.value || '').trim(),
    startDate: String(startEl.value || '').trim(),
    endDate: String(endEl.value || '').trim(),
    filters: {
      match: String(matchTypeEl?.value || 'exact').trim(),
      catchAll: catchAllEl ? !!catchAllEl.checked : true,
      utm_source: String(utmSourceEl?.value || '').trim(),
      utm_medium: String(utmMediumEl?.value || '').trim(),
      utm_campaign: String(utmCampaignEl?.value || '').trim(),
      utm_content: String(utmContentEl?.value || '').trim(),
      utm_term: String(utmTermEl?.value || '').trim(),
      utm_id: String(utmIdEl?.value || '').trim()
    }
  });

  const runReport = async () => {
    if (!getSavedToken()) {
      setStatus('Admin token required.', 'error');
      return;
    }

    const groupFields = getGroupFields();
    const payload = buildPayload();
    if (!payload.startDate || !payload.endDate) {
      setStatus('Start/end dates required.', 'error');
      return;
    }

    setBusy(true);
    setStatus('Fetching GA4 data…');
    try {
      const data = await fetchReport(payload);
      const rows = Array.isArray(data.rows) ? data.rows : [];
      const parsedRows = buildParsedRows(rows);
      const groups = aggregateGroups(parsedRows, groupFields);
      lastData = {
        propertyId: String(data.propertyId || payload.propertyId || '').trim(),
        startDate: String(data.startDate || payload.startDate || '').trim(),
        endDate: String(data.endDate || payload.endDate || '').trim(),
        truncated: !!data.truncated,
        rowCount: numberOrZero(data.rowCount),
        returnedRows: numberOrZero(data.returnedRows),
        filters: payload.filters || {},
        groupFields,
        groups,
        parsedRows,
        totals: sumMetrics(parsedRows)
      };
      drilldownGroup = null;
      renderAll();

      const urlsLabel = lastData.returnedRows || parsedRows.length;
      const groupsLabel = groups.length;
      const truncatedNote = lastData.truncated ? ' (truncated)' : '';
      setStatus(`Done. ${urlsLabel.toLocaleString('en-US')} URL(s), ${groupsLabel.toLocaleString('en-US')} group(s)${truncatedNote}.`, 'success');
      markSessionDirty();
    } catch (err) {
      lastData = null;
      drilldownGroup = null;
      if (downloadBtn) downloadBtn.disabled = true;
      if (downloadRawBtn) downloadRawBtn.disabled = true;
      summaryEl.replaceChildren();
      outputEl.replaceChildren();
      drilldownEl.replaceChildren();
      setStatus(err.message || 'Failed to fetch report.', 'error');
      markSessionDirty();
    } finally {
      setBusy(false);
    }
  };

  authForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const token = String(tokenInput.value || '').trim();
    if (!token) {
      updateAccessMeta();
      setStatus(getSavedToken() ? 'Token already stored.' : 'Paste your admin token to unlock this tool.', getSavedToken() ? 'success' : 'error');
      return;
    }
    saveToken(token);
    updateAccessMeta();
    tokenInput.value = '';
    setStatus('Token saved.', 'success');
    markSessionDirty();
  });

  forgetBtn.addEventListener('click', () => {
    saveToken('');
    updateAccessMeta();
    setStatus('Token forgotten on this device.', 'success');
    tokenInput.value = '';
    lastData = null;
    drilldownGroup = null;
    summaryEl.replaceChildren();
    outputEl.replaceChildren();
    drilldownEl.replaceChildren();
    if (downloadBtn) downloadBtn.disabled = true;
    if (downloadRawBtn) downloadRawBtn.disabled = true;
    markSessionDirty();
  });

  reportForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runReport();
  });

  const rebuildGroupsFromLastData = () => {
    if (!lastData || !Array.isArray(lastData.parsedRows) || !lastData.parsedRows.length) return;
    const groupFields = getGroupFields();
    lastData.groupFields = groupFields;
    lastData.groups = aggregateGroups(lastData.parsedRows, groupFields);
    drilldownGroup = null;
    renderAll();
  };

  viewModeEl.addEventListener('change', () => {
    renderAll();
    markSessionDirty();
  });

  sortFieldEl.addEventListener('change', () => {
    const mode = getActiveViewMode();
    const field = String(sortFieldEl.value || '').trim() || 'sessions';
    sortState[mode].field = field;
    if (!sortState[mode].dir) {
      sortState[mode].dir = getFieldType(mode, field) === 'number' ? 'desc' : 'asc';
    }
    renderAll();
    markSessionDirty();
  });

  sortDirEl.addEventListener('change', () => {
    const mode = getActiveViewMode();
    sortState[mode].dir = normalizeSortDir(sortDirEl.value);
    renderAll();
    markSessionDirty();
  });

  let filterDebounceTimer = 0;
  localFilterEl.addEventListener('input', () => {
    try { window.clearTimeout(filterDebounceTimer); } catch {}
    filterDebounceTimer = window.setTimeout(() => {
      renderAll();
      markSessionDirty();
    }, 120);
  });

  document.querySelectorAll('input[name=\"ga4-group\"]').forEach((el) => {
    el.addEventListener('change', () => {
      rebuildGroupsFromLastData();
      markSessionDirty();
    });
  });

  matchTypeEl.addEventListener('change', markSessionDirty);
  catchAllEl?.addEventListener('change', markSessionDirty);
  utmSourceEl?.addEventListener('input', markSessionDirty);
  utmMediumEl?.addEventListener('input', markSessionDirty);
  utmCampaignEl?.addEventListener('input', markSessionDirty);
  utmContentEl?.addEventListener('input', markSessionDirty);
  utmTermEl?.addEventListener('input', markSessionDirty);
  utmIdEl?.addEventListener('input', markSessionDirty);

  downloadBtn.addEventListener('click', () => {
    if (!lastData || !Array.isArray(lastData.groups) || !lastData.groups.length) return;

    const mode = getActiveViewMode();
    const groupFields = Array.isArray(lastData.groupFields) ? lastData.groupFields : getGroupFields();
    const query = mode === 'grouped' ? getLocalFilterQuery() : '';

    const filtered = applyLocalFilterGrouped(lastData.groups, groupFields, query);
    const sorted = sortList(filtered, 'grouped');

    const headers = [...groupFields, ...METRIC_FIELDS, 'rows'];
    const rows = sorted.map((row) => {
      const out = {};
      groupFields.forEach((f) => { out[f] = row?.[f] || ''; });
      METRIC_FIELDS.forEach((f) => { out[f] = numberOrZero(row?.[f]); });
      out.rows = numberOrZero(row?.rows);
      return out;
    });

    const csv = toCsv(rows, headers);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ga4-utm-grouped_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });

  downloadRawBtn.addEventListener('click', () => {
    if (!lastData || !Array.isArray(lastData.parsedRows) || !lastData.parsedRows.length) return;

    const mode = getActiveViewMode();
    const query = mode === 'raw' ? getLocalFilterQuery() : '';
    const filtered = applyLocalFilterRaw(lastData.parsedRows, query);
    const sorted = sortList(filtered, 'raw');

    const headers = ['pathname', ...UTM_FIELDS, ...METRIC_FIELDS, 'pageLocation'];
    const rows = sorted.map((row) => {
      const out = {};
      headers.forEach((h) => {
        if (METRIC_FIELDS.includes(h)) out[h] = numberOrZero(row?.[h]);
        else out[h] = row?.[h] ?? '';
      });
      return out;
    });

    const csv = toCsv(rows, headers);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ga4-utm-raw_${startEl.value || 'start'}_${endEl.value || 'end'}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const summary = String(statusEl.textContent || '').trim();
    payload.outputSummary = summary;

    const groupFields = lastData?.groupFields || getGroupFields();
    payload.inputs = {
      'Property ID': String(propertyEl?.value || '').trim(),
      'Start date': String(startEl.value || '').trim(),
      'End date': String(endEl.value || '').trim(),
      'UTM match': String(matchTypeEl?.value || 'exact'),
      'Catch all': catchAllEl && catchAllEl.checked ? 'Yes' : 'No',
      'utm_source': String(utmSourceEl?.value || '').trim(),
      'utm_medium': String(utmMediumEl?.value || '').trim(),
      'utm_campaign': String(utmCampaignEl?.value || '').trim(),
      'utm_content': String(utmContentEl?.value || '').trim(),
      'utm_term': String(utmTermEl?.value || '').trim(),
      'utm_id': String(utmIdEl?.value || '').trim(),
      'Group by': groupFields.join(', '),
      'View': getActiveViewMode(),
      'Sort': `${String(sortFieldEl?.value || '').trim()} (${String(sortDirEl?.value || '').trim()})`,
      'Filter': String(localFilterEl?.value || '').trim()
    };

    if (lastData) {
      const groups = Array.isArray(lastData.groups) ? lastData.groups : [];
      const parsedRows = Array.isArray(lastData.parsedRows) ? lastData.parsedRows : [];
      payload.output = {
        kind: 'json',
        summary,
        propertyId: lastData.propertyId || '',
        startDate: lastData.startDate || '',
        endDate: lastData.endDate || '',
        totals: lastData.totals || null,
        groupFields,
        groups: groups.slice(0, 200),
        rawRows: parsedRows.slice(0, 120)
      };
    }
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const snapshot = detail?.snapshot;
    const output = snapshot?.output;
    if (!output || typeof output !== 'object') return;
    if (String(output.kind || '') !== 'json') return;

    const groupFields = Array.isArray(output.groupFields) ? output.groupFields : getGroupFields();
    const groups = Array.isArray(output.groups) ? output.groups : [];
    const parsedRows = Array.isArray(output.rawRows) ? output.rawRows : [];
    const totals = output.totals || (parsedRows.length ? sumMetrics(parsedRows) : null);
    lastData = {
      propertyId: String(output.propertyId || '').trim(),
      startDate: String(output.startDate || '').trim(),
      endDate: String(output.endDate || '').trim(),
      groupFields,
      groups,
      parsedRows,
      totals
    };
    drilldownGroup = null;
    renderAll();
    const summary = String(output.summary || '').trim();
    if (summary) setStatus(summary);
  });

  initDefaultDates();
  updateAccessMeta();
  syncSortControls();
  renderAll();
})();
