(() => {
  'use strict';
  const main = document.getElementById('main');
  if (!main) return;

  const configSource = document.body || main;
  const config = {
    apiBase: (configSource.dataset.apiBase || '').trim(),
    cognitoDomain: (configSource.dataset.cognitoDomain || '').trim(),
    cognitoClientId: (configSource.dataset.cognitoClientId || '').trim(),
    cognitoRedirect: (configSource.dataset.cognitoRedirect || '').trim(),
    cognitoScopes: (configSource.dataset.cognitoScopes || 'openid email profile').trim()
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const els = {
    signIn: $('[data-jobtrack="sign-in"]'),
    signOut: $('[data-jobtrack="sign-out"]'),
    authStatus: $('[data-jobtrack="auth-status"]'),
    apiStatus: $('[data-jobtrack="api-status"]'),
    cognitoStatus: $('[data-jobtrack="cognito-status"]'),
    entryForm: $('[data-jobtrack="entry-form"]'),
    entryFormStatus: $('[data-jobtrack="entry-form-status"]'),
    entryType: $('[data-jobtrack="entry-type"]'),
    entryTypeInputs: $$('[data-jobtrack="entry-type"] input[name="entryType"]'),
    companyInput: $('#jobtrack-company'),
    titleInput: $('#jobtrack-title'),
    jobUrlInput: $('#jobtrack-job-url'),
    jobUrlHelp: $('[data-jobtrack="job-url-help"]'),
    locationInput: $('#jobtrack-location'),
    sourceInput: $('#jobtrack-source'),
    appliedDateInput: $('#jobtrack-date'),
    postingDateInput: $('#jobtrack-posting-date'),
    postingUnknownInput: $('#jobtrack-posting-unknown'),
    captureDateInput: $('#jobtrack-capture-date'),
    captureLabel: $('[data-jobtrack="capture-label"]'),
    captureHelp: $('[data-jobtrack="capture-help"]'),
    statusInput: $('#jobtrack-status'),
    notesInput: $('#jobtrack-notes'),
    entrySubmit: $('[data-jobtrack="entry-submit"]'),
    entryApplicationFields: $$('[data-jobtrack-entry="application"]'),
    entryProspectFields: $$('[data-jobtrack-entry="prospect"]'),
    resumeInput: $('#jobtrack-resume'),
    coverInput: $('#jobtrack-cover'),
    importFile: $('#jobtrack-import-file'),
    importAttachments: $('#jobtrack-import-attachments'),
    importSubmit: $('[data-jobtrack="import-submit"]'),
    importTemplate: $('[data-jobtrack="import-template"]'),
    importStatus: $('[data-jobtrack="import-status"]'),
    entryList: $('[data-jobtrack="entry-list"]'),
    entryListStatus: $('[data-jobtrack="entry-list-status"]'),
    entriesRefresh: $('[data-jobtrack="refresh-entries"]'),
    entryFilter: $('[data-jobtrack="entry-filter"]'),
    entryFilterQuery: $('[data-jobtrack="entry-filter-query"]'),
    entryFilterType: $('[data-jobtrack="entry-filter-type"]'),
    entryFilterStatus: $('[data-jobtrack="entry-filter-status"]'),
    entryFilterLocation: $('[data-jobtrack="entry-filter-location"]'),
    entryFilterStart: $('[data-jobtrack="entry-filter-start"]'),
    entryFilterEnd: $('[data-jobtrack="entry-filter-end"]'),
    entryFilterReset: $('[data-jobtrack="entry-filter-reset"]'),
    entrySortButtons: $$('[data-jobtrack-sort]'),
    exportForm: $('[data-jobtrack="export-form"]'),
    exportStart: $('[data-jobtrack="export-start"]'),
    exportEnd: $('[data-jobtrack="export-end"]'),
    exportSubmit: $('[data-jobtrack="export-submit"]'),
    exportStatus: $('[data-jobtrack="export-status"]'),
    dashboard: $('[data-jobtrack="dashboard"]'),
    dashboardStatus: $('[data-jobtrack="dashboard-status"]'),
    filterStart: $('[data-jobtrack="filter-start"]'),
    filterEnd: $('[data-jobtrack="filter-end"]'),
    filterReset: $('[data-jobtrack="filter-reset"]'),
    filterRefresh: $('[data-jobtrack="filter-refresh"]'),
    kpiTotal: $('[data-jobtrack="kpi-total"]'),
    kpiInterviews: $('[data-jobtrack="kpi-interviews"]'),
    kpiOffers: $('[data-jobtrack="kpi-offers"]'),
    kpiRejections: $('[data-jobtrack="kpi-rejections"]'),
    kpiFoundToApplied: $('[data-jobtrack="kpi-found-to-applied"]'),
    kpiFoundCount: $('[data-jobtrack="kpi-found-count"]'),
    kpiPostedToApplied: $('[data-jobtrack="kpi-posted-to-applied"]'),
    kpiPostedCount: $('[data-jobtrack="kpi-posted-count"]'),
    kpiResponseTime: $('[data-jobtrack="kpi-response-time"]'),
    kpiResponseTimeCount: $('[data-jobtrack="kpi-response-time-count"]'),
    kpiResponseRate: $('[data-jobtrack="kpi-response-rate"]'),
    kpiResponseCount: $('[data-jobtrack="kpi-response-count"]'),
    kpiTopSource: $('[data-jobtrack="kpi-top-source"]'),
    kpiTopSourceCount: $('[data-jobtrack="kpi-top-source-count"]'),
    kpiBestWeekday: $('[data-jobtrack="kpi-best-weekday"]'),
    kpiBestWeekdayCount: $('[data-jobtrack="kpi-best-weekday-count"]'),
    lineRange: $('[data-jobtrack="line-range"]'),
    lineTotal: $('[data-jobtrack="line-total"]'),
    statusTotal: $('[data-jobtrack="status-total"]'),
    calendarRange: $('[data-jobtrack="calendar-range"]'),
    lineOverlay: $('[data-jobtrack="line-overlay"]'),
    statusOverlay: $('[data-jobtrack="status-overlay"]'),
    calendarGrid: $('[data-jobtrack="calendar-grid"]'),
    mapContainer: $('[data-jobtrack="map"]'),
    mapPlaceholder: $('[data-jobtrack="map-placeholder"]'),
    mapTotal: $('[data-jobtrack="map-total"]')
  };

  const tabs = {
    buttons: $$('[data-jobtrack-tab]'),
    panels: $$('[data-jobtrack-panel]')
  };

  const STORAGE_KEY = 'jobTrackerAuth';
  const STATE_KEY = 'jobTrackerAuthState';
  const VERIFIER_KEY = 'jobTrackerCodeVerifier';
  const CSV_TEMPLATE = 'company,title,jobUrl,location,source,postingDate,appliedDate,status,notes,resumeFile,coverLetterFile\nAcme Corp,Data Analyst,https://acme.com/jobs/123,Remote,LinkedIn,2025-01-10,2025-01-15,Applied,Reached out to recruiter,Acme-Resume.pdf,Acme-Cover.pdf';
  const APPLICATION_STATUSES = ['Applied', 'Screening', 'Interview', 'Offer', 'Rejected', 'Withdrawn'];
  const PROSPECT_STATUSES = ['Active', 'Interested', 'Inactive'];
  const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const REMOTE_HINTS = ['remote', 'work from home', 'wfh', 'virtual'];
  const ONSITE_HINTS = ['on-site', 'onsite', 'on site', 'in office', 'in-office', 'hybrid'];
  const US_STATES = [
    { code: 'AL', name: 'Alabama' },
    { code: 'AK', name: 'Alaska' },
    { code: 'AZ', name: 'Arizona' },
    { code: 'AR', name: 'Arkansas' },
    { code: 'CA', name: 'California' },
    { code: 'CO', name: 'Colorado' },
    { code: 'CT', name: 'Connecticut' },
    { code: 'DE', name: 'Delaware' },
    { code: 'FL', name: 'Florida' },
    { code: 'GA', name: 'Georgia' },
    { code: 'HI', name: 'Hawaii' },
    { code: 'ID', name: 'Idaho' },
    { code: 'IL', name: 'Illinois' },
    { code: 'IN', name: 'Indiana' },
    { code: 'IA', name: 'Iowa' },
    { code: 'KS', name: 'Kansas' },
    { code: 'KY', name: 'Kentucky' },
    { code: 'LA', name: 'Louisiana' },
    { code: 'ME', name: 'Maine' },
    { code: 'MD', name: 'Maryland' },
    { code: 'MA', name: 'Massachusetts' },
    { code: 'MI', name: 'Michigan' },
    { code: 'MN', name: 'Minnesota' },
    { code: 'MS', name: 'Mississippi' },
    { code: 'MO', name: 'Missouri' },
    { code: 'MT', name: 'Montana' },
    { code: 'NE', name: 'Nebraska' },
    { code: 'NV', name: 'Nevada' },
    { code: 'NH', name: 'New Hampshire' },
    { code: 'NJ', name: 'New Jersey' },
    { code: 'NM', name: 'New Mexico' },
    { code: 'NY', name: 'New York' },
    { code: 'NC', name: 'North Carolina' },
    { code: 'ND', name: 'North Dakota' },
    { code: 'OH', name: 'Ohio' },
    { code: 'OK', name: 'Oklahoma' },
    { code: 'OR', name: 'Oregon' },
    { code: 'PA', name: 'Pennsylvania' },
    { code: 'RI', name: 'Rhode Island' },
    { code: 'SC', name: 'South Carolina' },
    { code: 'SD', name: 'South Dakota' },
    { code: 'TN', name: 'Tennessee' },
    { code: 'TX', name: 'Texas' },
    { code: 'UT', name: 'Utah' },
    { code: 'VT', name: 'Vermont' },
    { code: 'VA', name: 'Virginia' },
    { code: 'WA', name: 'Washington' },
    { code: 'WV', name: 'West Virginia' },
    { code: 'WI', name: 'Wisconsin' },
    { code: 'WY', name: 'Wyoming' },
    { code: 'DC', name: 'District Of Columbia' }
  ];
  const STATE_CODE_SET = new Set(US_STATES.map(state => state.code.toLowerCase()));
  const STATE_NAME_LOOKUP = new Map(US_STATES.map(state => [state.code, state.name]));
  const STATE_NAME_BY_LOWER = new Map(US_STATES.map(state => [state.name.toLowerCase(), state.code]));

  const state = {
    auth: null,
    lineChart: null,
    statusChart: null,
    range: null,
    editingEntryId: null,
    editingEntry: null,
    entryType: 'application',
    entries: [],
    entryItems: new Map(),
    entrySort: { key: 'date', direction: 'desc' },
    entryFilters: {
      query: '',
      type: 'all',
      status: 'all',
      location: 'all',
      start: '',
      end: ''
    },
    mapLoaded: false,
    mapSvg: null,
    isResettingEntry: false
  };

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const runWithConcurrency = async (items, limit, task) => {
    const results = [];
    let index = 0;
    const runWorker = async () => {
      while (index < items.length) {
        const current = items[index];
        index += 1;
        try {
          const value = await task(current);
          results.push({ ok: true, value });
        } catch (err) {
          results.push({ ok: false, error: err });
        }
      }
    };
    const workerCount = Math.max(1, Math.min(limit, items.length));
    const workers = Array.from({ length: workerCount }, () => runWorker());
    await Promise.all(workers);
    return results;
  };

  const formatDateInput = (date) => date.toISOString().slice(0, 10);
  const parseDateInput = (value) => {
    if (!value) return null;
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };
  const parseIsoDate = (value) => {
    if (!value) return null;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };
  const formatDateLabel = (value) => {
    const parsed = parseDateInput(value) || parseIsoDate(value);
    if (!parsed) return '—';
    return parsed.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };
  const formatDays = (value) => {
    if (!Number.isFinite(value)) return '--';
    return `${value.toFixed(1)} days`;
  };
  const formatPercent = (value) => {
    if (!Number.isFinite(value)) return '--';
    return `${Math.round(value)}%`;
  };
  const isRemoteLocation = (location = '') => {
    const text = (location || '').toString().toLowerCase();
    if (!text) return false;
    const hasRemote = REMOTE_HINTS.some(hint => text.includes(hint));
    const hasOnsite = ONSITE_HINTS.some(hint => text.includes(hint));
    return hasRemote && !hasOnsite;
  };
  const extractStateCode = (location = '') => {
    const text = (location || '').toString().toLowerCase();
    if (!text) return null;
    for (const [nameLower, code] of STATE_NAME_BY_LOWER.entries()) {
      if (text.includes(nameLower)) return code;
    }
    const upper = text.toUpperCase();
    for (const state of US_STATES) {
      const pattern = new RegExp(`\\b${state.code}\\b`);
      if (pattern.test(upper)) return state.code;
    }
    return null;
  };

  const syncUnknownDate = (dateInput, unknownInput) => {
    if (!dateInput || !unknownInput) return;
    const isUnknown = Boolean(unknownInput.checked);
    dateInput.disabled = isUnknown;
    if (isUnknown) dateInput.value = '';
  };

  const initUnknownDateToggle = (dateInput, unknownInput, defaultUnknown = true) => {
    if (!dateInput || !unknownInput) return;
    if (defaultUnknown && !dateInput.value) {
      unknownInput.checked = true;
    }
    const sync = () => syncUnknownDate(dateInput, unknownInput);
    unknownInput.addEventListener('change', sync);
    sync();
  };

  const setUnknownDateValue = (dateInput, unknownInput, value, defaultUnknown = true) => {
    if (dateInput) dateInput.value = value || '';
    if (unknownInput) {
      unknownInput.checked = value ? false : defaultUnknown;
    }
    syncUnknownDate(dateInput, unknownInput);
  };

  const toTitle = (value) => value
    .toLowerCase()
    .split(' ')
    .filter(Boolean)
    .map(word => word[0].toUpperCase() + word.slice(1))
    .join(' ');

  const stripBom = (value) => value.replace(/^\uFEFF/, '');

  const normalizeHeader = (value) => stripBom(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');

  const parseCsv = (text) => {
    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        row.push(field);
        field = '';
      } else if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && text[i + 1] === '\n') i += 1;
        row.push(field);
        if (row.some(cell => cell.trim() !== '')) rows.push(row);
        row = [];
        field = '';
      } else {
        field += char;
      }
    }
    if (field.length || row.length) {
      row.push(field);
      if (row.some(cell => cell.trim() !== '')) rows.push(row);
    }
    return rows;
  };

  const CSV_HEADERS = {
    company: ['company', 'companyname', 'employer', 'organization'],
    title: ['title', 'role', 'position', 'jobtitle'],
    appliedDate: ['applieddate', 'dateapplied', 'applicationdate', 'applied', 'date'],
    postingDate: ['postingdate', 'posteddate', 'jobpostingdate', 'dateposted'],
    status: ['status', 'stage'],
    notes: ['notes', 'note', 'details'],
    jobUrl: ['joburl', 'url', 'link', 'joblink', 'applicationurl'],
    location: ['location', 'city', 'region', 'locale'],
    source: ['source', 'referral', 'channel', 'board'],
    resumeFile: ['resume', 'resumefile', 'resumefilename', 'resumeattachment', 'resumeattachmentname'],
    coverLetterFile: ['coverletter', 'coverletterfile', 'coverletterfilename', 'cover', 'coverfile']
  };

  const buildHeaderMap = (headers = []) => {
    const normalized = headers.map(header => normalizeHeader(header));
    const map = {};
    Object.entries(CSV_HEADERS).forEach(([key, aliases]) => {
      const idx = normalized.findIndex(value => aliases.includes(value));
      if (idx >= 0) map[key] = idx;
    });
    return map;
  };

  const parseCsvDate = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return '';
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return trimmed;
    const mdy = trimmed.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{2,4})$/);
    if (mdy) {
      const year = mdy[3].length === 2 ? `20${mdy[3]}` : mdy[3];
      const month = mdy[1].padStart(2, '0');
      const day = mdy[2].padStart(2, '0');
      const iso = `${year}-${month}-${day}`;
      return parseDateInput(iso) ? iso : '';
    }
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? '' : formatDateInput(parsed);
  };

  const normalizeUrl = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return '';
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    return `https://${trimmed}`;
  };

  const readFileText = (file) => {
    if (file && typeof file.text === 'function') return file.text();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result || '');
      reader.onerror = () => reject(reader.error || new Error('Unable to read file.'));
      reader.readAsText(file);
    });
  };

  const parseJwt = (token) => {
    try {
      const payload = token.split('.')[1];
      if (!payload) return null;
      const normalized = payload.replace(/-/g, '+').replace(/_/g, '/');
      const decoded = atob(normalized.padEnd(normalized.length + (4 - normalized.length % 4) % 4, '='));
      return JSON.parse(decoded);
    } catch {
      return null;
    }
  };

  const loadAuth = () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.idToken) return null;
      return parsed;
    } catch {
      return null;
    }
  };

  const saveAuth = (auth) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(auth));
    } catch {}
  };

  const clearAuth = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
    state.auth = null;
  };

  const authIsValid = (auth) => {
    if (!auth || !auth.idToken) return false;
    if (auth.expiresAt && Date.now() > auth.expiresAt - 60 * 1000) return false;
    return true;
  };

  const getCssColor = (value) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return value;
    ctx.fillStyle = value;
    return ctx.fillStyle;
  };

  const toRgba = (value, alpha) => {
    const normalized = getCssColor(value);
    if (normalized.startsWith('rgba(')) {
      return normalized.replace(/rgba\\(([^)]+),\\s*[^)]+\\)/, `rgba($1, ${alpha})`);
    }
    if (normalized.startsWith('rgb(')) {
      return normalized.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
    }
    return value;
  };

  const readCssVar = (name, fallback) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (value && value.trim()) ? value.trim() : fallback;
  };

  const joinUrl = (base, path) => {
    if (!base) return path;
    if (!path) return base;
    const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${cleanBase}${cleanPath}`;
  };

  const setStatus = (el, message, tone = '') => {
    if (!el) return;
    el.textContent = message;
    if (tone) {
      el.dataset.tone = tone;
    } else {
      delete el.dataset.tone;
    }
  };

  const setOverlay = (el, message) => {
    if (!el) return;
    if (!message) {
      el.dataset.state = 'hidden';
      el.textContent = '';
      return;
    }
    el.dataset.state = '';
    el.textContent = message;
  };

  const confirmAction = (message) => {
    if (typeof window === 'undefined' || typeof window.confirm !== 'function') return true;
    return window.confirm(message);
  };

  const updateConfigStatus = () => {
    if (els.apiStatus) {
      els.apiStatus.textContent = config.apiBase ? 'Configured' : 'Not configured';
    }
    if (els.cognitoStatus) {
      els.cognitoStatus.textContent = (config.cognitoDomain && config.cognitoClientId && config.cognitoRedirect)
        ? 'Configured'
        : 'Not configured';
    }
  };

  const storeEntries = (items = []) => {
    state.entries = items;
    state.entryItems = new Map();
    items.forEach((item) => {
      if (item && item.applicationId) state.entryItems.set(item.applicationId, item);
    });
  };

  const getEntryType = (item) => (item && item.recordType === 'prospect' ? 'prospect' : 'application');

  const updateEntrySubmitLabel = () => {
    if (!els.entrySubmit) return;
    const label = state.entryType === 'prospect' ? 'prospect' : 'application';
    els.entrySubmit.textContent = state.editingEntryId ? `Update ${label}` : `Save ${label}`;
  };

  const toggleEntryGroup = (nodes = [], isVisible) => {
    nodes.forEach((node) => {
      node.hidden = !isVisible;
      const inputs = node.querySelectorAll('input,select,textarea,button');
      inputs.forEach((input) => {
        input.disabled = !isVisible;
      });
    });
  };

  const updateStatusOptions = (type, preserveSelection = true) => {
    if (!els.statusInput) return;
    const options = type === 'prospect' ? PROSPECT_STATUSES : APPLICATION_STATUSES;
    const current = preserveSelection ? els.statusInput.value : '';
    els.statusInput.innerHTML = '';
    options.forEach((option) => {
      const opt = document.createElement('option');
      opt.value = option;
      opt.textContent = option;
      els.statusInput.appendChild(opt);
    });
    if (current && options.includes(current)) {
      els.statusInput.value = current;
    } else {
      els.statusInput.value = options[0];
    }
  };

  const setEntryType = (type, { preserveStatus = true } = {}) => {
    const nextType = type === 'prospect' ? 'prospect' : 'application';
    state.entryType = nextType;
    if (els.entryTypeInputs.length) {
      els.entryTypeInputs.forEach((input) => {
        input.checked = input.value === nextType;
      });
    }
    toggleEntryGroup(els.entryApplicationFields, nextType === 'application');
    toggleEntryGroup(els.entryProspectFields, nextType === 'prospect');
    if (els.jobUrlInput) els.jobUrlInput.required = nextType === 'prospect';
    if (els.appliedDateInput) els.appliedDateInput.required = nextType === 'application';
    if (els.captureDateInput) els.captureDateInput.required = nextType === 'prospect';
    if (els.jobUrlHelp) {
      els.jobUrlHelp.textContent = nextType === 'prospect' ? 'Required for prospects.' : 'Optional for applications.';
    }
    if (els.captureLabel) {
      els.captureLabel.textContent = nextType === 'prospect' ? 'Found date' : 'Found date (optional)';
    }
    if (els.captureHelp) {
      els.captureHelp.textContent = nextType === 'prospect'
        ? 'Defaults to today for prospects.'
        : 'When you first saved or discovered the role.';
    }
    updateStatusOptions(nextType, preserveStatus);
    if (nextType === 'prospect') clearAttachmentInputs();
    updateEntrySubmitLabel();
  };

  const setEntryEditMode = (item) => {
    if (!item) return;
    const entryType = getEntryType(item);
    state.editingEntryId = item.applicationId || null;
    state.editingEntry = item;
    setEntryType(entryType, { preserveStatus: false });
    if (els.companyInput) els.companyInput.value = item.company || '';
    if (els.titleInput) els.titleInput.value = item.title || '';
    if (els.jobUrlInput) els.jobUrlInput.value = item.jobUrl || '';
    if (els.locationInput) els.locationInput.value = item.location || '';
    if (els.sourceInput) els.sourceInput.value = item.source || '';
    if (els.appliedDateInput) els.appliedDateInput.value = item.appliedDate || '';
    if (els.captureDateInput) {
      els.captureDateInput.value = item.captureDate || '';
    }
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, item.postingDate || '');
    if (els.statusInput) els.statusInput.value = item.status || (entryType === 'prospect' ? 'Active' : 'Applied');
    if (els.notesInput) els.notesInput.value = item.notes || '';
    updateEntrySubmitLabel();
    if (els.entryFormStatus) {
      const label = [item.title, item.company].filter(Boolean).join(' · ') || 'entry';
      setStatus(els.entryFormStatus, `Editing ${label}. Save to update or clear to cancel.`, 'info');
    }
    activateTab('entry', true);
  };

  const clearEntryEditMode = (message = 'Ready to save entries.', tone = '') => {
    state.editingEntryId = null;
    state.editingEntry = null;
    updateEntrySubmitLabel();
    if (els.entryFormStatus && message) setStatus(els.entryFormStatus, message, tone);
  };

  const activateTab = (name, shouldFocus = false) => {
    if (!tabs.buttons.length || !tabs.panels.length) return;
    const authed = authIsValid(state.auth);
    const safeName = authed || name === 'account' ? name : 'account';
    tabs.buttons.forEach((button) => {
      const selected = button.dataset.jobtrackTab === safeName;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    tabs.panels.forEach((panel) => {
      panel.hidden = panel.dataset.jobtrackPanel !== safeName;
    });
    if (shouldFocus) {
      const activeButton = tabs.buttons.find(button => button.dataset.jobtrackTab === safeName);
      if (activeButton) activeButton.focus();
    }
    if (safeName === 'dashboard') {
      window.requestAnimationFrame(() => {
        if (state.lineChart) state.lineChart.resize();
        if (state.statusChart) state.statusChart.resize();
      });
    }
  };

  const initTabs = () => {
    if (!tabs.buttons.length || !tabs.panels.length) return;
    const defaultTab = tabs.buttons.find(button => button.getAttribute('aria-selected') === 'true')?.dataset.jobtrackTab
      || tabs.buttons[0].dataset.jobtrackTab;
    const hash = window.location && window.location.hash
      ? window.location.hash.replace('#', '')
      : '';
    const initial = tabs.buttons.some(button => button.dataset.jobtrackTab === hash) ? hash : defaultTab;
    activateTab(initial);
    tabs.buttons.forEach((button, index) => {
      button.addEventListener('click', () => activateTab(button.dataset.jobtrackTab, true));
      button.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          const next = (index + 1) % tabs.buttons.length;
          activateTab(tabs.buttons[next].dataset.jobtrackTab, true);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          const prev = (index - 1 + tabs.buttons.length) % tabs.buttons.length;
          activateTab(tabs.buttons[prev].dataset.jobtrackTab, true);
        } else if (event.key === 'Home') {
          event.preventDefault();
          activateTab(tabs.buttons[0].dataset.jobtrackTab, true);
        } else if (event.key === 'End') {
          event.preventDefault();
          activateTab(tabs.buttons[tabs.buttons.length - 1].dataset.jobtrackTab, true);
        }
      });
    });
  };

  const randomBase64Url = (size = 32) => {
    const buffer = new Uint8Array(size);
    crypto.getRandomValues(buffer);
    const binary = String.fromCharCode(...buffer);
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  };

  const sha256 = async (plain) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(plain);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(digest);
  };

  const buildAuthorizeUrl = async () => {
    const verifier = randomBase64Url(48);
    const challengeBytes = await sha256(verifier);
    const challenge = btoa(String.fromCharCode(...challengeBytes))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    const authState = randomBase64Url(16);
    sessionStorage.setItem(STATE_KEY, authState);
    sessionStorage.setItem(VERIFIER_KEY, verifier);

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      scope: config.cognitoScopes,
      code_challenge_method: 'S256',
      code_challenge: challenge,
      state: authState
    });
    return `https://${config.cognitoDomain}/oauth2/authorize?${params.toString()}`;
  };

  const exchangeCodeForTokens = async (code) => {
    const verifier = sessionStorage.getItem(VERIFIER_KEY) || '';
    sessionStorage.removeItem(VERIFIER_KEY);
    sessionStorage.removeItem(STATE_KEY);
    if (!verifier) throw new Error('Missing PKCE verifier.');

    const params = new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      code,
      code_verifier: verifier
    });
    const res = await fetch(`https://${config.cognitoDomain}/oauth2/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params.toString()
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Unable to exchange auth code.');
    }
    const data = await res.json();
    if (!data.id_token) throw new Error('Missing id_token from auth response.');
    const claims = parseJwt(data.id_token) || {};
    const expiresAt = claims.exp ? claims.exp * 1000 : Date.now() + (data.expires_in || 3600) * 1000;
    const auth = {
      idToken: data.id_token,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt,
      claims
    };
    saveAuth(auth);
    state.auth = auth;
    return auth;
  };

  const handleAuthRedirect = async () => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const returnedState = params.get('state') || '';
    const storedState = sessionStorage.getItem(STATE_KEY) || '';
    if (!code) return false;
    if (storedState && returnedState && storedState !== returnedState) {
      throw new Error('Auth state mismatch.');
    }
    setStatus(els.authStatus, 'Finalizing sign-in...', 'info');
    await exchangeCodeForTokens(code);
    params.delete('code');
    params.delete('state');
    const next = params.toString();
    const nextUrl = next ? `${window.location.pathname}?${next}` : window.location.pathname;
    window.history.replaceState({}, document.title, nextUrl);
    return true;
  };

  const updateAuthUI = () => {
    const authed = authIsValid(state.auth);
    if (els.signIn) {
      els.signIn.disabled = authed;
      els.signIn.setAttribute('aria-disabled', authed ? 'true' : 'false');
    }
    if (els.signOut) {
      els.signOut.disabled = !authed;
      els.signOut.setAttribute('aria-disabled', !authed ? 'true' : 'false');
    }
    if (els.authStatus) {
      if (authed) {
        const claims = state.auth?.claims || parseJwt(state.auth?.idToken || '') || {};
        const label = claims.email || claims['cognito:username'] || claims.username || 'Signed in';
        els.authStatus.textContent = `Signed in as ${label}.`;
      } else {
        els.authStatus.textContent = 'Not signed in.';
      }
    }
    tabs.buttons.forEach((button) => {
      const isAccount = button.dataset.jobtrackTab === 'account';
      const disabled = !authed && !isAccount;
      button.disabled = disabled;
      button.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    });
    if (!authed) activateTab('account');
    if (els.importStatus) {
      setStatus(els.importStatus, authed ? 'Ready to import applications.' : 'Sign in to import applications.', authed ? '' : 'info');
    }
    if (els.entryFormStatus && !state.editingEntryId) {
      const entryLabel = state.entryType === 'prospect' ? 'prospects' : 'applications';
      setStatus(els.entryFormStatus, authed ? `Ready to save ${entryLabel}.` : 'Sign in to save entries.', authed ? '' : 'info');
    }
    if (els.entryListStatus) {
      setStatus(els.entryListStatus, authed ? 'Use filters and sorting to review your entries.' : 'Sign in to load your entries.', authed ? '' : 'info');
    }
    if (els.exportStatus) {
      setStatus(els.exportStatus, authed ? 'Choose a date range to export applications.' : 'Sign in to export applications.', authed ? '' : 'info');
    }
  };

  const getAuthHeader = () => {
    if (!state.auth || !state.auth.idToken) return null;
    return `Bearer ${state.auth.idToken}`;
  };

  const requestJson = async (path, { method = 'GET', body } = {}) => {
    if (!config.apiBase) throw new Error('API base URL is not configured.');
    const authHeader = getAuthHeader();
    if (!authHeader) throw new Error('Sign in to use the tracker.');
    const res = await fetch(joinUrl(config.apiBase, path), {
      method,
      headers: {
        'Content-Type': 'application/json',
        Authorization: authHeader
      },
      body: body ? JSON.stringify(body) : undefined
    });
    const text = await res.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (!res.ok) {
      throw new Error(data?.error || data?.message || text || `${res.status} ${res.statusText}`);
    }
    return data || {};
  };

  const getAttachmentLabel = (attachment = {}) => {
    const kind = (attachment.kind || '').toString().toLowerCase();
    if (kind === 'cover' || kind === 'cover-letter' || kind === 'coverletter') return 'Cover letter';
    return 'Resume';
  };

  const collectAttachments = () => {
    const attachments = [];
    const resumeFile = els.resumeInput?.files?.[0];
    const coverFile = els.coverInput?.files?.[0];
    if (resumeFile) attachments.push({ file: resumeFile, kind: 'resume' });
    if (coverFile) attachments.push({ file: coverFile, kind: 'cover-letter' });
    return attachments;
  };

  const clearAttachmentInputs = () => {
    if (els.resumeInput) els.resumeInput.value = '';
    if (els.coverInput) els.coverInput.value = '';
  };

  const resetEntryDateFields = (type = state.entryType) => {
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, '');
    if (type === 'prospect' && els.captureDateInput) {
      els.captureDateInput.value = els.captureDateInput.value || formatDateInput(new Date());
    }
  };

  const uploadAttachment = async (applicationId, attachment) => {
    const file = attachment.file;
    const contentType = file.type || 'application/octet-stream';
    const presign = await requestJson('/api/attachments/presign', {
      method: 'POST',
      body: {
        applicationId,
        filename: file.name || 'attachment',
        contentType
      }
    });
    const res = await fetch(presign.uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': contentType },
      body: file
    });
    if (!res.ok) {
      throw new Error('Unable to upload attachment.');
    }
    return {
      key: presign.key,
      filename: file.name || 'attachment',
      contentType,
      kind: attachment.kind || '',
      uploadedAt: new Date().toISOString()
    };
  };

  const uploadAttachments = async (applicationId, attachments = []) => {
    const uploaded = [];
    for (const attachment of attachments) {
      const item = await uploadAttachment(applicationId, attachment);
      uploaded.push(item);
    }
    return uploaded;
  };

  const requestAttachmentDownload = async (key) => requestJson('/api/attachments/download', {
    method: 'POST',
    body: { key }
  });

  const defaultRange = () => {
    const end = new Date();
    const start = new Date();
    start.setUTCDate(end.getUTCDate() - 89);
    return { start, end };
  };

  const readRange = () => {
    const start = parseDateInput(els.filterStart?.value) || defaultRange().start;
    const end = parseDateInput(els.filterEnd?.value) || defaultRange().end;
    if (start > end) return { start: end, end: start };
    return { start, end };
  };

  const formatRangeLabel = (start, end) => {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return `${start.toLocaleDateString('en-US', options)} - ${end.toLocaleDateString('en-US', options)}`;
  };

  const updateRangeInputs = (range) => {
    if (els.filterStart) els.filterStart.value = formatDateInput(range.start);
    if (els.filterEnd) els.filterEnd.value = formatDateInput(range.end);
  };

  const buildQuery = (range) => {
    const params = new URLSearchParams({
      start: formatDateInput(range.start),
      end: formatDateInput(range.end)
    });
    return params.toString();
  };

  const updateKpis = (summary) => {
    if (els.kpiTotal) els.kpiTotal.textContent = summary.totalApplications ?? 0;
    if (els.kpiInterviews) els.kpiInterviews.textContent = summary.interviews ?? 0;
    if (els.kpiOffers) els.kpiOffers.textContent = summary.offers ?? 0;
    if (els.kpiRejections) els.kpiRejections.textContent = summary.rejections ?? 0;
  };

  const updateLineChart = (series, rangeLabel) => {
    if (els.lineRange) els.lineRange.textContent = rangeLabel;
    if (els.lineTotal) {
      const total = series.reduce((acc, item) => acc + (item.count || 0), 0);
      els.lineTotal.textContent = `${total} apps`;
    }
    const ctx = document.getElementById('jobtrack-line-chart');
    if (!ctx || !window.Chart) return;
    const labels = series.map(item => item.date);
    const data = series.map(item => item.count);
    const accent = readCssVar('--jobtrack-accent', '#2396AD');
    const grid = toRgba('#ffffff', 0.08);
    const text = readCssVar('--text-muted', '#BFC8D3');
    const dataset = {
      label: 'Applications',
      data,
      borderColor: accent,
      backgroundColor: toRgba(accent, 0.2),
      tension: 0.35,
      fill: true,
      pointRadius: 2,
      pointHoverRadius: 4,
      borderWidth: 2
    };

    if (state.lineChart) {
      state.lineChart.data.labels = labels;
      state.lineChart.data.datasets = [dataset];
      state.lineChart.update();
      return;
    }
    state.lineChart = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [dataset]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { color: text, maxTicksLimit: 6 },
            grid: { color: grid }
          },
          y: {
            ticks: { color: text, precision: 0 },
            grid: { color: grid },
            beginAtZero: true
          }
        }
      }
    });
  };

  const updateStatusChart = (statuses) => {
    if (els.statusTotal) {
      const total = statuses.reduce((acc, item) => acc + (item.count || 0), 0);
      els.statusTotal.textContent = `${total} statuses`;
    }
    const ctx = document.getElementById('jobtrack-status-chart');
    if (!ctx || !window.Chart) return;
    const labels = statuses.map(item => item.status);
    const data = statuses.map(item => item.count);
    const accent = readCssVar('--jobtrack-accent', '#2396AD');
    const grid = toRgba('#ffffff', 0.08);
    const text = readCssVar('--text-muted', '#BFC8D3');
    const dataset = {
      label: 'Applications',
      data,
      backgroundColor: toRgba(accent, 0.35),
      borderColor: accent,
      borderWidth: 1.5
    };
    if (state.statusChart) {
      state.statusChart.data.labels = labels;
      state.statusChart.data.datasets = [dataset];
      state.statusChart.update();
      return;
    }
    state.statusChart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [dataset]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: text, precision: 0 },
            grid: { color: grid },
            beginAtZero: true
          },
          y: {
            ticks: { color: text },
            grid: { color: 'transparent' }
          }
        }
      }
    });
  };

  const buildCalendar = (days, rangeLabel, range) => {
    if (!els.calendarGrid) return;
    if (els.calendarRange) els.calendarRange.textContent = rangeLabel;
    els.calendarGrid.innerHTML = '';
    const counts = new Map(days.map(item => [item.date, item.count]));
    const max = Math.max(0, ...days.map(item => item.count || 0));
    const scale = (count) => {
      if (!count) return 0;
      if (max <= 1) return 1;
      return Math.min(4, Math.ceil((count / max) * 4));
    };

    const start = new Date(range.start);
    const end = new Date(range.end);
    const startDay = start.getUTCDay();
    const startOffset = (startDay + 6) % 7;
    const calendarStart = new Date(start);
    calendarStart.setUTCDate(start.getUTCDate() - startOffset);
    const endDay = end.getUTCDay();
    const endOffset = (7 - ((endDay + 6) % 7) - 1);
    const calendarEnd = new Date(end);
    calendarEnd.setUTCDate(end.getUTCDate() + endOffset);

    let cursor = new Date(calendarStart);
    while (cursor <= calendarEnd) {
      const iso = formatDateInput(cursor);
      const count = counts.get(iso) || 0;
      const intensity = scale(count);
      const cell = document.createElement('div');
      cell.className = 'jobtrack-calendar-day';
      cell.dataset.intensity = String(intensity);
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-label', `${iso}: ${count} applications`);
      cell.title = `${iso}: ${count} applications`;
      els.calendarGrid.appendChild(cell);
      cursor = new Date(cursor.getTime() + 86400000);
    }
  };

  const getFirstResponseDate = (entry) => {
    const history = Array.isArray(entry?.statusHistory) ? entry.statusHistory : [];
    if (!history.length) {
      const status = (entry?.status || '').toString().trim().toLowerCase();
      if (status && status !== 'applied') {
        return parseIsoDate(entry?.updatedAt) || parseIsoDate(entry?.createdAt);
      }
      return null;
    }
    const sorted = history
      .map(item => ({
        status: (item?.status || '').toString().trim().toLowerCase(),
        date: parseIsoDate(item?.date) || parseDateInput(item?.date)
      }))
      .filter(item => item.date)
      .sort((a, b) => a.date - b.date);
    for (const item of sorted) {
      if (item.status && item.status !== 'applied') return item.date;
    }
    return null;
  };

  const buildInsights = (items = []) => {
    let foundSum = 0;
    let foundCount = 0;
    let postedSum = 0;
    let postedCount = 0;
    let responseSum = 0;
    let responseCount = 0;
    const sourceCounts = new Map();
    const weekdayCounts = new Map(WEEKDAYS.map(day => [day, 0]));

    items.forEach((item) => {
      const applied = parseDateInput(item?.appliedDate);
      if (!applied) return;
      const appliedTime = applied.getTime();

      const captureDate = parseDateInput(item?.captureDate);
      if (captureDate) {
        const delta = (appliedTime - captureDate.getTime()) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          foundSum += delta;
          foundCount += 1;
        }
      }

      const postingDate = parseDateInput(item?.postingDate);
      if (postingDate) {
        const delta = (appliedTime - postingDate.getTime()) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          postedSum += delta;
          postedCount += 1;
        }
      }

      const responseDate = getFirstResponseDate(item);
      if (responseDate) {
        const delta = (responseDate.getTime() - appliedTime) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          responseSum += delta;
          responseCount += 1;
        }
      }

      const source = (item?.source || '').toString().trim();
      if (source) {
        const key = source.toLowerCase();
        const current = sourceCounts.get(key) || { label: source, count: 0 };
        current.count += 1;
        sourceCounts.set(key, current);
      }

      const weekday = WEEKDAYS[applied.getUTCDay()];
      if (weekday) {
        weekdayCounts.set(weekday, (weekdayCounts.get(weekday) || 0) + 1);
      }
    });

    let topSource = null;
    let topSourceCount = 0;
    sourceCounts.forEach((value) => {
      if (value.count > topSourceCount) {
        topSourceCount = value.count;
        topSource = value.label;
      }
    });

    let bestWeekday = null;
    let bestWeekdayCount = 0;
    weekdayCounts.forEach((count, day) => {
      if (count > bestWeekdayCount) {
        bestWeekdayCount = count;
        bestWeekday = day;
      }
    });

    const total = items.length;
    return {
      avgFoundToApplied: foundCount ? foundSum / foundCount : null,
      foundCount,
      avgPostedToApplied: postedCount ? postedSum / postedCount : null,
      postedCount,
      avgResponseTime: responseCount ? responseSum / responseCount : null,
      responseCount,
      responseRate: total ? (responseCount / total) * 100 : null,
      total,
      topSource: topSource || null,
      topSourceCount,
      bestWeekday: bestWeekday || null,
      bestWeekdayCount
    };
  };

  const updateInsights = (insights = {}) => {
    if (els.kpiFoundToApplied) {
      els.kpiFoundToApplied.textContent = formatDays(insights.avgFoundToApplied);
    }
    if (els.kpiFoundCount) {
      els.kpiFoundCount.textContent = insights.foundCount
        ? `${insights.foundCount} apps`
        : 'No data yet';
    }
    if (els.kpiPostedToApplied) {
      els.kpiPostedToApplied.textContent = formatDays(insights.avgPostedToApplied);
    }
    if (els.kpiPostedCount) {
      els.kpiPostedCount.textContent = insights.postedCount
        ? `${insights.postedCount} apps`
        : 'No data yet';
    }
    if (els.kpiResponseTime) {
      els.kpiResponseTime.textContent = formatDays(insights.avgResponseTime);
    }
    if (els.kpiResponseTimeCount) {
      els.kpiResponseTimeCount.textContent = insights.responseCount
        ? `${insights.responseCount} responses`
        : 'No responses yet';
    }
    if (els.kpiResponseRate) {
      els.kpiResponseRate.textContent = formatPercent(insights.responseRate);
    }
    if (els.kpiResponseCount) {
      els.kpiResponseCount.textContent = insights.total
        ? `${insights.responseCount} of ${insights.total} apps`
        : 'No data yet';
    }
    if (els.kpiTopSource) {
      els.kpiTopSource.textContent = insights.topSource || '--';
    }
    if (els.kpiTopSourceCount) {
      els.kpiTopSourceCount.textContent = insights.topSourceCount
        ? `${insights.topSourceCount} apps`
        : 'No sources yet';
    }
    if (els.kpiBestWeekday) {
      els.kpiBestWeekday.textContent = insights.bestWeekday || '--';
    }
    if (els.kpiBestWeekdayCount) {
      els.kpiBestWeekdayCount.textContent = insights.bestWeekdayCount
        ? `${insights.bestWeekdayCount} apps`
        : 'No data yet';
    }
  };

  const getMapStateCode = (node) => {
    const className = (node.getAttribute('class') || '').toString();
    const classes = className.split(/\s+/).filter(Boolean);
    const match = classes.find(value => STATE_CODE_SET.has(value.toLowerCase()));
    return match ? match.toUpperCase() : null;
  };

  const loadMap = async () => {
    if (!els.mapContainer || state.mapLoaded) return state.mapSvg;
    const src = (els.mapContainer.dataset.jobtrackMapSrc || '').trim();
    if (!src) {
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Map source missing.';
      return null;
    }
    try {
      const res = await fetch(src);
      if (!res.ok) throw new Error('Unable to load map.');
      const text = await res.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(text, 'image/svg+xml');
      const svg = doc.querySelector('svg');
      if (!svg) throw new Error('Map SVG not found.');
      svg.classList.add('jobtrack-map-svg');
      svg.removeAttribute('width');
      svg.removeAttribute('height');
      const style = svg.querySelector('style');
      if (style) style.remove();
      els.mapContainer.innerHTML = '';
      els.mapContainer.appendChild(svg);
      state.mapLoaded = true;
      state.mapSvg = svg;
      return svg;
    } catch (err) {
      console.error('Map load failed', err);
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Unable to load map.';
      return null;
    }
  };

  const updateMap = async (applications = []) => {
    if (!els.mapContainer) return;
    const svg = await loadMap();
    if (!svg) return;

    const counts = new Map();
    let totalApps = 0;
    applications.forEach((item) => {
      const location = (item?.location || '').toString();
      if (!location || isRemoteLocation(location)) return;
      const code = extractStateCode(location);
      if (!code) return;
      counts.set(code, (counts.get(code) || 0) + 1);
      totalApps += 1;
    });

    const max = Math.max(0, ...Array.from(counts.values()));
    const scale = (count) => {
      if (!count) return 0;
      if (max <= 1) return 1;
      return Math.min(4, Math.ceil((count / max) * 4));
    };

    const nodes = svg.querySelectorAll('.state path, .state circle');
    nodes.forEach((node) => {
      const code = getMapStateCode(node);
      if (!code) return;
      const count = counts.get(code) || 0;
      const intensity = scale(count);
      if (intensity) {
        node.dataset.intensity = String(intensity);
      } else {
        node.removeAttribute('data-intensity');
      }
      const name = STATE_NAME_LOOKUP.get(code) || code;
      const label = `${name}: ${count} application${count === 1 ? '' : 's'}`;
      node.setAttribute('aria-label', label);
      node.setAttribute('title', label);
    });

    if (els.mapTotal) {
      const stateCount = counts.size;
      if (stateCount) {
        els.mapTotal.textContent = `${stateCount} state${stateCount === 1 ? '' : 's'} · ${totalApps} application${totalApps === 1 ? '' : 's'}`;
      } else {
        els.mapTotal.textContent = 'No states yet';
      }
    }
  };

  const refreshDashboard = async () => {
    if (!els.dashboard || !els.dashboardStatus) return;
    if (!config.apiBase) {
      setStatus(els.dashboardStatus, 'Set the API base URL to load dashboards.', 'error');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Set the API base URL to load the map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Map unavailable';
      updateInsights({});
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.dashboardStatus, 'Sign in to load your dashboards.', 'info');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Sign in to load the map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Sign in to view map';
      updateInsights({});
      return;
    }
    const range = readRange();
    state.range = range;
    updateRangeInputs(range);
    const rangeLabel = formatRangeLabel(range.start, range.end);
    setStatus(els.dashboardStatus, 'Loading dashboards...', 'info');
    if (els.dashboard) els.dashboard.setAttribute('aria-busy', 'true');
    setOverlay(els.lineOverlay, 'Loading chart...');
    setOverlay(els.statusOverlay, 'Loading chart...');
    try {
      const query = buildQuery(range);
      const [summary, timeline, statuses, calendar, applications] = await Promise.all([
        requestJson(`/api/analytics/summary?${query}`),
        requestJson(`/api/analytics/applications-over-time?${query}`),
        requestJson(`/api/analytics/status-breakdown?${query}`),
        requestJson(`/api/analytics/calendar?${query}`),
        requestJson(`/api/applications?${query}`)
      ]);
      updateKpis(summary);
      const series = timeline.series || [];
      const statusSeries = statuses.statuses || [];
      updateLineChart(series, rangeLabel);
      updateStatusChart(statusSeries);
      buildCalendar(calendar.days || [], rangeLabel, range);
      const appItems = Array.isArray(applications.items) ? applications.items : [];
      updateInsights(buildInsights(appItems));
      await updateMap(appItems);
      setOverlay(els.lineOverlay, series.length ? '' : 'No activity yet.');
      setOverlay(els.statusOverlay, statusSeries.length ? '' : 'No statuses yet.');
      setStatus(els.dashboardStatus, `Loaded ${summary.totalApplications || 0} applications.`, 'success');
    } catch (err) {
      console.error('Dashboard load failed', err);
      setOverlay(els.lineOverlay, 'Unable to load chart.');
      setOverlay(els.statusOverlay, 'Unable to load chart.');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Unable to load map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Map unavailable';
      updateInsights({});
      setStatus(els.dashboardStatus, err?.message || 'Unable to load dashboards.', 'error');
    } finally {
      if (els.dashboard) els.dashboard.setAttribute('aria-busy', 'false');
    }
  };

  const normalizeEntry = (item, entryType) => ({
    ...item,
    recordType: entryType,
    entryType
  });

  const getEntryDateValue = (entry) => (entry.entryType === 'prospect' ? entry.captureDate : entry.appliedDate);

  const getEntryDate = (entry) => {
    const raw = getEntryDateValue(entry);
    return parseDateInput(raw);
  };

  const updateEntryStatusFilter = (items = []) => {
    if (!els.entryFilterStatus) return;
    const current = els.entryFilterStatus.value || 'all';
    const statuses = new Set();
    items.forEach((item) => {
      const entryType = getEntryType(item);
      const raw = (item.status || (entryType === 'prospect' ? 'Active' : 'Applied')).toString();
      statuses.add(toTitle(raw));
    });
    const sorted = Array.from(statuses).sort((a, b) => a.localeCompare(b));
    els.entryFilterStatus.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = 'all';
    allOpt.textContent = 'All statuses';
    els.entryFilterStatus.appendChild(allOpt);
    sorted.forEach((status) => {
      const opt = document.createElement('option');
      opt.value = status.toLowerCase();
      opt.textContent = status;
      els.entryFilterStatus.appendChild(opt);
    });
    if ([...els.entryFilterStatus.options].some(opt => opt.value === current)) {
      els.entryFilterStatus.value = current;
    }
  };

  const matchesQuery = (entry, terms = []) => {
    if (!terms.length) return true;
    const haystack = [
      entry.company,
      entry.title,
      entry.location,
      entry.source,
      entry.notes,
      entry.status
    ].join(' ').toLowerCase();
    return terms.every(term => haystack.includes(term));
  };

  const filterEntries = (items = []) => {
    const query = (els.entryFilterQuery?.value || '').trim().toLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);
    const type = (els.entryFilterType?.value || 'all').trim();
    const status = (els.entryFilterStatus?.value || 'all').trim();
    const locationType = (els.entryFilterLocation?.value || 'all').trim();
    const start = parseDateInput(els.entryFilterStart?.value || '');
    const end = parseDateInput(els.entryFilterEnd?.value || '');

    return items.filter((entry) => {
      const entryType = entry.entryType || getEntryType(entry);
      if (type !== 'all' && entryType !== type) return false;
      const entryStatus = toTitle((entry.status || (entryType === 'prospect' ? 'Active' : 'Applied')).toString());
      if (status !== 'all' && entryStatus.toLowerCase() !== status) return false;
      if (locationType !== 'all') {
        const hasLocation = Boolean((entry.location || '').trim());
        const remote = isRemoteLocation(entry.location || '');
        if (locationType === 'remote' && (!hasLocation || !remote)) return false;
        if (locationType === 'onsite' && (!hasLocation || remote)) return false;
      }
      if (!matchesQuery(entry, terms)) return false;
      if (start || end) {
        const dateValue = getEntryDate(entry);
        if (!dateValue) return false;
        if (start && dateValue < start) return false;
        if (end && dateValue > end) return false;
      }
      return true;
    });
  };

  const sortEntries = (items = []) => {
    const { key, direction } = state.entrySort;
    const multiplier = direction === 'asc' ? 1 : -1;
    const sorted = [...items];
    sorted.sort((a, b) => {
      let aVal = '';
      let bVal = '';
      if (key === 'date') {
        const aDate = getEntryDate(a);
        const bDate = getEntryDate(b);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return (aTime - bTime) * multiplier;
      }
      if (key === 'postingDate') {
        const aDate = parseDateInput(a.postingDate);
        const bDate = parseDateInput(b.postingDate);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return (aTime - bTime) * multiplier;
      }
      if (key === 'type') {
        aVal = a.entryType || getEntryType(a);
        bVal = b.entryType || getEntryType(b);
      } else if (key === 'company') {
        aVal = a.company || '';
        bVal = b.company || '';
      } else if (key === 'title') {
        aVal = a.title || '';
        bVal = b.title || '';
      } else if (key === 'status') {
        aVal = a.status || '';
        bVal = b.status || '';
      } else if (key === 'location') {
        aVal = a.location || '';
        bVal = b.location || '';
      } else if (key === 'source') {
        aVal = a.source || '';
        bVal = b.source || '';
      }
      return aVal.toString().localeCompare(bVal.toString(), 'en', { sensitivity: 'base' }) * multiplier;
    });
    return sorted;
  };

  const renderEntryList = (items = [], emptyLabel = 'No entries yet.') => {
    if (!els.entryList) return;
    els.entryList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'jobtrack-table-empty';
      empty.textContent = emptyLabel;
      els.entryList.appendChild(empty);
      return;
    }
    items.forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'jobtrack-table-row';
      row.setAttribute('role', 'row');
      const entryType = entry.entryType || getEntryType(entry);

      const typeCell = document.createElement('div');
      typeCell.className = 'jobtrack-table-cell';
      const pill = document.createElement('span');
      pill.className = 'jobtrack-pill';
      if (entryType === 'prospect') pill.dataset.tone = 'prospect';
      pill.textContent = entryType === 'prospect' ? 'Prospect' : 'Application';
      typeCell.appendChild(pill);
      row.appendChild(typeCell);

      const companyCell = document.createElement('div');
      companyCell.className = 'jobtrack-table-cell';
      companyCell.textContent = entry.company || '—';
      row.appendChild(companyCell);

      const titleCell = document.createElement('div');
      titleCell.className = 'jobtrack-table-cell';
      titleCell.textContent = entry.title || '—';
      row.appendChild(titleCell);

      const statusCell = document.createElement('div');
      statusCell.className = 'jobtrack-table-cell';
      statusCell.textContent = toTitle(entry.status || (entryType === 'prospect' ? 'Active' : 'Applied'));
      row.appendChild(statusCell);

      const locationCell = document.createElement('div');
      locationCell.className = 'jobtrack-table-cell';
      locationCell.textContent = entry.location || '—';
      row.appendChild(locationCell);

      const postingCell = document.createElement('div');
      postingCell.className = 'jobtrack-table-cell';
      postingCell.textContent = entry.postingDate ? formatDateLabel(entry.postingDate) : '—';
      row.appendChild(postingCell);

      const dateCell = document.createElement('div');
      dateCell.className = 'jobtrack-table-cell';
      const dateLabel = getEntryDateValue(entry);
      dateCell.textContent = dateLabel
        ? `${entryType === 'prospect' ? 'Found' : 'Applied'} ${formatDateLabel(dateLabel)}`
        : '—';
      row.appendChild(dateCell);

      const sourceCell = document.createElement('div');
      sourceCell.className = 'jobtrack-table-cell';
      sourceCell.textContent = entry.source || '—';
      row.appendChild(sourceCell);

      const actionsCell = document.createElement('div');
      actionsCell.className = 'jobtrack-table-cell jobtrack-table-actions';
      if (entry.applicationId) {
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'btn-ghost';
        editBtn.dataset.jobtrackEntry = 'edit';
        editBtn.dataset.id = entry.applicationId;
        editBtn.textContent = 'Edit';
        actionsCell.appendChild(editBtn);

        if (entryType === 'prospect') {
          const applyBtn = document.createElement('button');
          applyBtn.type = 'button';
          applyBtn.className = 'btn-ghost';
          applyBtn.dataset.jobtrackEntry = 'apply';
          applyBtn.dataset.id = entry.applicationId;
          applyBtn.textContent = 'Apply';
          actionsCell.appendChild(applyBtn);
        }

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'btn-ghost';
        deleteBtn.dataset.jobtrackEntry = 'delete';
        deleteBtn.dataset.id = entry.applicationId;
        deleteBtn.textContent = 'Delete';
        actionsCell.appendChild(deleteBtn);
      }
      row.appendChild(actionsCell);

      els.entryList.appendChild(row);
    });
  };

  const updateSortIndicators = () => {
    if (!els.entrySortButtons.length) return;
    els.entrySortButtons.forEach((button) => {
      const key = button.dataset.jobtrackSort;
      if (key === state.entrySort.key) {
        button.setAttribute('aria-sort', state.entrySort.direction === 'asc' ? 'ascending' : 'descending');
      } else {
        button.setAttribute('aria-sort', 'none');
      }
    });
  };

  const applyEntryFilters = () => {
    const filtered = filterEntries(state.entries);
    const sorted = sortEntries(filtered);
    const emptyLabel = state.entries.length ? 'No entries match your filters yet.' : 'No entries yet.';
    renderEntryList(sorted, emptyLabel);
    updateSortIndicators();
  };

  const refreshEntries = async () => {
    if (!els.entryList) return;
    if (!config.apiBase) {
      storeEntries([]);
      renderEntryList([], 'Set the API base URL to load entries.');
      return;
    }
    if (!authIsValid(state.auth)) {
      storeEntries([]);
      renderEntryList([], 'Sign in to load your entries.');
      return;
    }
    try {
      setStatus(els.entryListStatus, 'Loading entries...', 'info');
      const [apps, prospects] = await Promise.all([
        requestJson('/api/applications'),
        requestJson('/api/prospects')
      ]);
      const appItems = (apps.items || []).map(item => normalizeEntry(item, 'application'));
      const prospectItems = (prospects.items || []).map(item => normalizeEntry(item, 'prospect'));
      const items = [...appItems, ...prospectItems];
      storeEntries(items);
      updateEntryStatusFilter(items);
      applyEntryFilters();
      setStatus(els.entryListStatus, `Loaded ${items.length} entries.`, 'success');
    } catch (err) {
      console.error('Entry load failed', err);
      storeEntries([]);
      renderEntryList([], 'Unable to load entries.');
      setStatus(els.entryListStatus, err?.message || 'Unable to load entries.', 'error');
    }
  };

  const initEntryList = () => {
    if (els.entriesRefresh) {
      els.entriesRefresh.addEventListener('click', () => refreshEntries());
    }
    if (els.entryFilterQuery) {
      els.entryFilterQuery.addEventListener('input', () => applyEntryFilters());
    }
    if (els.entryFilterType) {
      els.entryFilterType.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterStatus) {
      els.entryFilterStatus.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterLocation) {
      els.entryFilterLocation.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterStart) {
      els.entryFilterStart.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterEnd) {
      els.entryFilterEnd.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterReset) {
      els.entryFilterReset.addEventListener('click', () => {
        if (els.entryFilterQuery) els.entryFilterQuery.value = '';
        if (els.entryFilterType) els.entryFilterType.value = 'all';
        if (els.entryFilterStatus) els.entryFilterStatus.value = 'all';
        if (els.entryFilterLocation) els.entryFilterLocation.value = 'all';
        if (els.entryFilterStart) els.entryFilterStart.value = '';
        if (els.entryFilterEnd) els.entryFilterEnd.value = '';
        applyEntryFilters();
      });
    }
    if (els.entrySortButtons.length) {
      els.entrySortButtons.forEach((button) => {
        button.addEventListener('click', () => {
          const key = button.dataset.jobtrackSort;
          if (!key) return;
          if (state.entrySort.key === key) {
            state.entrySort.direction = state.entrySort.direction === 'asc' ? 'desc' : 'asc';
          } else {
            state.entrySort.key = key;
            state.entrySort.direction = key === 'date' ? 'desc' : 'asc';
          }
          applyEntryFilters();
        });
      });
    }
    if (els.entryList) {
      els.entryList.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-jobtrack-entry]');
        if (!button) return;
        const entryId = button.dataset.id;
        if (!entryId) return;
        const action = button.dataset.jobtrackEntry;
        if (action === 'edit') {
          const item = state.entryItems.get(entryId);
          if (item) setEntryEditMode(item);
          return;
        }
        if (action === 'apply') {
          applyProspect(entryId);
          return;
        }
        if (action === 'delete') {
          deleteEntry(entryId);
        }
      });
    }
  };

  const initExport = () => {
    if (!els.exportForm) return;
    const range = defaultRange();
    if (els.exportStart && !els.exportStart.value) els.exportStart.value = formatDateInput(range.start);
    if (els.exportEnd && !els.exportEnd.value) els.exportEnd.value = formatDateInput(range.end);
    if (els.exportSubmit) {
      els.exportSubmit.addEventListener('click', async () => {
        if (!authIsValid(state.auth)) {
          setStatus(els.exportStatus, 'Sign in to export applications.', 'error');
          return;
        }
        if (!config.apiBase) {
          setStatus(els.exportStatus, 'Set the API base URL to export applications.', 'error');
          return;
        }
        let startValue = (els.exportStart?.value || '').trim();
        let endValue = (els.exportEnd?.value || '').trim();
        if (!startValue && !endValue) {
          const fallback = defaultRange();
          startValue = formatDateInput(fallback.start);
          endValue = formatDateInput(fallback.end);
          if (els.exportStart) els.exportStart.value = startValue;
          if (els.exportEnd) els.exportEnd.value = endValue;
        }
        if (!startValue || !endValue) {
          setStatus(els.exportStatus, 'Choose a start and end date for the export.', 'error');
          return;
        }
        const startDate = parseDateInput(startValue);
        const endDate = parseDateInput(endValue);
        if (!startDate || !endDate) {
          setStatus(els.exportStatus, 'Export dates must be valid.', 'error');
          return;
        }
        let start = startValue;
        let end = endValue;
        if (startDate > endDate) {
          start = endValue;
          end = startValue;
          if (els.exportStart) els.exportStart.value = start;
          if (els.exportEnd) els.exportEnd.value = end;
        }
        try {
          setStatus(els.exportStatus, 'Preparing export...', 'info');
          const result = await requestJson('/api/exports', {
            method: 'POST',
            body: { start, end }
          });
          const url = result?.downloadUrl;
          if (url) {
            const link = document.createElement('a');
            link.href = url;
            link.download = '';
            document.body.appendChild(link);
            link.click();
            link.remove();
            setStatus(els.exportStatus, 'Export ready. Download should begin shortly.', 'success');
          } else {
            setStatus(els.exportStatus, 'Export ready, but no download link was returned.', 'error');
          }
        } catch (err) {
          console.error('Export failed', err);
          setStatus(els.exportStatus, err?.message || 'Unable to export applications.', 'error');
        }
      });
    }
  };

  const deleteEntry = async (entryId) => {
    if (!entryId) return;
    if (!config.apiBase) {
      setStatus(els.entryListStatus, 'Set the API base URL to delete entries.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to delete entries.', 'error');
      return;
    }
    const item = state.entryItems.get(entryId);
    const label = [item?.title, item?.company].filter(Boolean).join(' · ') || 'this entry';
    if (!confirmAction(`Delete ${label}? This cannot be undone.`)) return;
    try {
      setStatus(els.entryListStatus, 'Deleting entry...', 'info');
      await requestJson(`/api/applications/${encodeURIComponent(entryId)}`, { method: 'DELETE' });
      if (state.editingEntryId === entryId) {
        clearEntryEditMode('Ready to save entries.', 'info');
        if (els.entryForm) {
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          resetEntryDateFields();
          clearAttachmentInputs();
        }
      }
      setStatus(els.entryListStatus, 'Entry deleted.', 'success');
      await Promise.all([refreshEntries(), refreshDashboard()]);
    } catch (err) {
      console.error('Entry delete failed', err);
      setStatus(els.entryListStatus, err?.message || 'Unable to delete entry.', 'error');
    }
  };

  const submitProspect = async (payload) => {
    if (!els.entryFormStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryFormStatus, 'Sign in to save prospects.', 'error');
      return false;
    }
    const editingId = state.editingEntryId;
    try {
      setStatus(els.entryFormStatus, editingId ? 'Updating prospect...' : 'Saving prospect...', 'info');
      if (editingId) {
        await requestJson(`/api/prospects/${encodeURIComponent(editingId)}`, { method: 'PATCH', body: payload });
      } else {
        await requestJson('/api/prospects', { method: 'POST', body: payload });
      }
      clearEntryEditMode(editingId ? 'Prospect updated.' : 'Prospect saved.', 'success');
      await sleep(200);
      await refreshEntries();
      return true;
    } catch (err) {
      console.error('Prospect save failed', err);
      setStatus(els.entryFormStatus, err?.message || 'Unable to save prospect.', 'error');
      return false;
    }
  };

  const buildApplicationPayloadFromProspect = (item, appliedDate) => {
    const payload = {
      company: (item?.company || '').toString().trim(),
      title: (item?.title || '').toString().trim(),
      appliedDate,
      status: 'Applied',
      notes: (item?.notes || '').toString().trim()
    };
    if (item?.jobUrl) payload.jobUrl = item.jobUrl;
    if (item?.location) payload.location = item.location;
    if (item?.source) payload.source = item.source;
    if (item?.postingDate) payload.postingDate = item.postingDate;
    if (item?.captureDate) payload.captureDate = item.captureDate;
    return payload;
  };

  const convertProspectToApplication = async (payload, prospectId, statusEl = null) => {
    const statusTarget = statusEl || els.entryListStatus || els.entryFormStatus;
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to save applications.', 'error');
      return false;
    }
    try {
      setStatus(statusTarget, 'Moving prospect to applications...', 'info');
      await requestJson('/api/applications', { method: 'POST', body: payload });
      let deleteError = null;
      if (prospectId) {
        try {
          await requestJson(`/api/applications/${encodeURIComponent(prospectId)}`, { method: 'DELETE' });
        } catch (err) {
          deleteError = err;
        }
      }
      if (deleteError) {
        setStatus(statusTarget, 'Application saved, but the prospect could not be removed.', 'error');
      } else {
        setStatus(statusTarget, prospectId ? 'Prospect moved to applications.' : 'Application saved.', 'success');
      }
      clearEntryEditMode();
      await sleep(200);
      await Promise.all([refreshEntries(), refreshDashboard()]);
      return true;
    } catch (err) {
      console.error('Prospect conversion failed', err);
      setStatus(statusTarget, err?.message || 'Unable to move prospect to applications.', 'error');
      return false;
    }
  };

  const applyProspect = async (prospectId) => {
    if (!prospectId) return;
    const item = state.entryItems.get(prospectId);
    if (!item) return;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to move prospects.', 'error');
      return;
    }
    const defaultDate = formatDateInput(new Date());
    const response = typeof window !== 'undefined' && typeof window.prompt === 'function'
      ? window.prompt('Applied date (YYYY-MM-DD):', defaultDate)
      : defaultDate;
    if (response === null) return;
    const appliedDate = response.toString().trim();
    if (!appliedDate || !parseDateInput(appliedDate)) {
      setStatus(els.entryListStatus, 'Applied date must be valid (YYYY-MM-DD).', 'error');
      return;
    }
    const payload = buildApplicationPayloadFromProspect(item, appliedDate);
    await convertProspectToApplication(payload, prospectId, els.entryListStatus);
  };

  const submitApplication = async (payload, attachments = []) => {
    if (!els.entryFormStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryFormStatus, 'Sign in to save applications.', 'error');
      return false;
    }
    const editingId = state.editingEntryId;
    const editingItem = state.editingEntry;
    try {
      setStatus(els.entryFormStatus, editingId ? 'Updating application...' : 'Saving application...', 'info');
      let applicationId = editingId;
      let currentAttachments = Array.isArray(editingItem?.attachments) ? editingItem.attachments : [];
      if (editingId) {
        await requestJson(`/api/applications/${encodeURIComponent(editingId)}`, { method: 'PATCH', body: payload });
      } else {
        const created = await requestJson('/api/applications', { method: 'POST', body: payload });
        applicationId = created?.applicationId;
        currentAttachments = Array.isArray(created?.attachments) ? created.attachments : [];
      }
      let attachmentError = null;
      if (attachments.length && applicationId) {
        try {
          const label = attachments.length === 1 ? 'attachment' : 'attachments';
          setStatus(els.entryFormStatus, `Uploading ${attachments.length} ${label}...`, 'info');
          const uploaded = await uploadAttachments(applicationId, attachments);
          const merged = [...currentAttachments, ...uploaded].slice(-12);
          await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, {
            method: 'PATCH',
            body: { attachments: merged }
          });
        } catch (err) {
          attachmentError = err;
        }
      }
      if (attachmentError) {
        console.error('Attachment upload failed', attachmentError);
        clearEntryEditMode(
          editingId ? 'Updated application, but attachments failed to upload.' : 'Saved application, but attachments failed to upload.',
          'error'
        );
      } else {
        clearEntryEditMode(
          editingId ? 'Application updated. Refreshing dashboards...' : 'Application saved. Updating dashboards...',
          'success'
        );
      }
      await sleep(200);
      await Promise.all([refreshDashboard(), refreshEntries()]);
      return true;
    } catch (err) {
      console.error('Application save failed', err);
      setStatus(els.entryFormStatus, err?.message || 'Unable to save application.', 'error');
      return false;
    }
  };

  const initEntryForm = () => {
    if (!els.entryForm) return;
    initUnknownDateToggle(els.postingDateInput, els.postingUnknownInput, true);
    setEntryType(state.entryType, { preserveStatus: false });
    resetEntryDateFields(state.entryType);
    if (els.entryTypeInputs.length) {
      els.entryTypeInputs.forEach((input) => {
        input.addEventListener('change', () => {
          const nextType = input.value;
          if (state.editingEntry && getEntryType(state.editingEntry) !== nextType) {
            clearEntryEditMode('Entry type changed. Start a new entry below.', 'info');
          }
          setEntryType(nextType);
          resetEntryDateFields(nextType);
        });
      });
    }
    els.entryForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(els.entryForm);
      const entryType = (formData.get('entryType') || state.entryType).toString().trim();
      const company = (formData.get('company') || '').toString().trim();
      const title = (formData.get('title') || '').toString().trim();
      const jobUrl = normalizeUrl(formData.get('jobUrl'));
      const location = (formData.get('location') || '').toString().trim();
      const source = (formData.get('source') || '').toString().trim();
      const appliedDate = (formData.get('appliedDate') || '').toString().trim();
      const postingDate = (formData.get('postingDate') || '').toString().trim();
      const postingUnknown = Boolean(formData.get('postingDateUnknown'));
      const captureDate = (formData.get('captureDate') || '').toString().trim();
      const status = (formData.get('status') || (entryType === 'prospect' ? 'Active' : 'Applied')).toString().trim();
      const notes = (formData.get('notes') || '').toString().trim();

      if (!company || !title) {
        setStatus(els.entryFormStatus, 'Company and role title are required.', 'error');
        return;
      }
      if (!postingUnknown && !postingDate) {
        setStatus(els.entryFormStatus, 'Add a posting date or mark it as unknown.', 'error');
        return;
      }
      if (postingDate && !parseDateInput(postingDate)) {
        setStatus(els.entryFormStatus, 'Posting date must be valid.', 'error');
        return;
      }

      if (entryType === 'application') {
        if (!appliedDate) {
          setStatus(els.entryFormStatus, 'Applied date is required for applications.', 'error');
          return;
        }
        if (!parseDateInput(appliedDate)) {
          setStatus(els.entryFormStatus, 'Applied date must be valid.', 'error');
          return;
        }
        if (captureDate && !parseDateInput(captureDate)) {
          setStatus(els.entryFormStatus, 'Found date must be valid.', 'error');
          return;
        }
        const attachments = collectAttachments();
        const editing = state.editingEntry;
        const payload = { company, title, appliedDate, notes };
        if (jobUrl || editing?.jobUrl) payload.jobUrl = jobUrl;
        if (location || editing?.location) payload.location = location;
        if (source || editing?.source) payload.source = source;
        if (postingUnknown) {
          payload.postingDate = null;
        } else if (postingDate) {
          payload.postingDate = postingDate;
        }
        if (captureDate) {
          payload.captureDate = captureDate;
        } else if (editing?.captureDate) {
          payload.captureDate = null;
        }
        const existingStatus = editing?.status ? editing.status.toString().trim().toLowerCase() : '';
        if (!editing || !existingStatus || status.toLowerCase() !== existingStatus) {
          payload.status = status;
        }
        const ok = await submitApplication(payload, attachments);
        if (ok) {
          const nextType = entryType;
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          clearAttachmentInputs();
          setEntryType(nextType);
          resetEntryDateFields(nextType);
        }
        return;
      }

      if (!jobUrl) {
        setStatus(els.entryFormStatus, 'Job URL is required for prospects.', 'error');
        return;
      }
      if (!captureDate) {
        setStatus(els.entryFormStatus, 'Found date is required for prospects.', 'error');
        return;
      }
      if (!parseDateInput(captureDate)) {
        setStatus(els.entryFormStatus, 'Found date must be valid.', 'error');
        return;
      }
      const payload = { company, title, jobUrl, location, source, status, notes, captureDate };
      if (postingUnknown) {
        payload.postingDate = null;
      } else if (postingDate) {
        payload.postingDate = postingDate;
      }
      const ok = await submitProspect(payload);
      if (ok) {
        const nextType = entryType;
        state.isResettingEntry = true;
        els.entryForm.reset();
        state.isResettingEntry = false;
        setEntryType(nextType);
        resetEntryDateFields(nextType);
      }
    });
    els.entryForm.addEventListener('reset', () => {
      clearAttachmentInputs();
      const nextType = state.entryType;
      resetEntryDateFields(nextType);
      if (state.isResettingEntry) return;
      setEntryType(nextType);
      clearEntryEditMode();
    });
  };

  const parseImportPayloads = (text) => {
    const rows = parseCsv(text || '');
    if (!rows.length) return { entries: [], skipped: 0, missing: ['company', 'title', 'appliedDate'] };
    const headers = rows.shift().map(header => header.trim());
    const map = buildHeaderMap(headers);
    const missing = ['company', 'title', 'appliedDate'].filter(key => map[key] === undefined);
    if (missing.length) return { entries: [], skipped: rows.length, missing };

    const entries = [];
    let skipped = 0;
    rows.forEach((row) => {
      const company = (row[map.company] || '').toString().trim();
      const title = (row[map.title] || '').toString().trim();
      const appliedDate = parseCsvDate(row[map.appliedDate]);
      const postingDate = map.postingDate !== undefined ? parseCsvDate(row[map.postingDate]) : '';
      const status = map.status !== undefined ? (row[map.status] || '').toString().trim() : '';
      const notes = map.notes !== undefined ? (row[map.notes] || '').toString().trim() : '';
      const jobUrl = map.jobUrl !== undefined ? normalizeUrl(row[map.jobUrl]) : '';
      const location = map.location !== undefined ? (row[map.location] || '').toString().trim() : '';
      const source = map.source !== undefined ? (row[map.source] || '').toString().trim() : '';
      const resumeFile = map.resumeFile !== undefined ? (row[map.resumeFile] || '').toString().trim() : '';
      const coverLetterFile = map.coverLetterFile !== undefined ? (row[map.coverLetterFile] || '').toString().trim() : '';
      if (!company || !title || !appliedDate) {
        skipped += 1;
        return;
      }
      const payload = {
        company,
        title,
        appliedDate,
        status: status || 'Applied',
        notes
      };
      if (postingDate) payload.postingDate = postingDate;
      if (jobUrl) payload.jobUrl = jobUrl;
      if (location) payload.location = location;
      if (source) payload.source = source;
      entries.push({
        payload,
        resumeFile,
        coverLetterFile
      });
    });
    return { entries, skipped, missing: [] };
  };

  const buildImportAttachmentMap = (files = []) => {
    const map = new Map();
    files.forEach((file) => {
      if (file && file.name) {
        map.set(file.name.toLowerCase(), file);
      }
    });
    return map;
  };

  const resolveImportAttachment = (name, lookup, missing) => {
    const trimmed = (name || '').toString().trim();
    if (!trimmed) return null;
    const file = lookup.get(trimmed.toLowerCase());
    if (!file && missing) missing.add(trimmed);
    return file || null;
  };

  const initImport = () => {
    if (els.importTemplate) {
      els.importTemplate.addEventListener('click', () => {
        const blob = new Blob([CSV_TEMPLATE], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'job-applications-template.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      });
    }
    if (els.importSubmit) {
      els.importSubmit.addEventListener('click', async () => {
        if (!authIsValid(state.auth)) {
          setStatus(els.importStatus, 'Sign in to import applications.', 'error');
          return;
        }
        if (!config.apiBase) {
          setStatus(els.importStatus, 'Set the API base URL to import applications.', 'error');
          return;
        }
        const file = els.importFile?.files?.[0];
        if (!file) {
          setStatus(els.importStatus, 'Choose a CSV file to import.', 'error');
          return;
        }
        try {
          setStatus(els.importStatus, 'Reading CSV...', 'info');
          const text = await readFileText(file);
          const { entries, skipped, missing } = parseImportPayloads(text);
          if (missing.length) {
            setStatus(els.importStatus, `Missing columns: ${missing.join(', ')}.`, 'error');
            return;
          }
          if (!entries.length) {
            setStatus(els.importStatus, 'No valid rows found in the CSV.', 'error');
            return;
          }
          const attachmentLookup = buildImportAttachmentMap(Array.from(els.importAttachments?.files || []));
          const missingAttachments = new Set();
          const attachmentRequests = entries.reduce((total, entry) => total
            + (entry.resumeFile ? 1 : 0)
            + (entry.coverLetterFile ? 1 : 0), 0);
          const importLabel = attachmentRequests
            ? `Importing ${entries.length} applications and ${attachmentRequests} attachments...`
            : `Importing ${entries.length} applications...`;
          setStatus(els.importStatus, importLabel, 'info');
          const results = await runWithConcurrency(entries, 3, async (entry) => {
            const created = await requestJson('/api/applications', {
              method: 'POST',
              body: entry.payload
            });
            const applicationId = created?.applicationId;
            const attachments = [];
            const resumeFile = resolveImportAttachment(entry.resumeFile, attachmentLookup, missingAttachments);
            if (resumeFile) attachments.push({ file: resumeFile, kind: 'resume' });
            const coverFile = resolveImportAttachment(entry.coverLetterFile, attachmentLookup, missingAttachments);
            if (coverFile) attachments.push({ file: coverFile, kind: 'cover-letter' });
            let attachmentsUploaded = 0;
            let attachmentsFailed = 0;
            if (attachments.length && applicationId) {
              try {
                const uploaded = await uploadAttachments(applicationId, attachments);
                await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, {
                  method: 'PATCH',
                  body: { attachments: uploaded }
                });
                attachmentsUploaded = uploaded.length;
              } catch {
                attachmentsFailed = attachments.length;
              }
            } else if (attachments.length && !applicationId) {
              attachmentsFailed = attachments.length;
            }
            return { attachmentsUploaded, attachmentsFailed };
          });
          const success = results.filter(result => result.ok).length;
          const failed = results.length - success;
          const attachmentTotals = results.reduce((acc, result) => {
            if (result.ok) {
              acc.uploaded += result.value?.attachmentsUploaded || 0;
              acc.failed += result.value?.attachmentsFailed || 0;
            }
            return acc;
          }, { uploaded: 0, failed: 0 });
          const parts = [`Imported ${success} of ${entries.length} applications.`];
          if (skipped) parts.push(`Skipped ${skipped} rows.`);
          if (failed) parts.push(`${failed} failed.`);
          if (missingAttachments.size) {
            const missingList = Array.from(missingAttachments);
            const preview = missingList.slice(0, 3);
            const suffix = missingList.length > preview.length ? ', ...' : '';
            parts.push(`Missing ${missingList.length} attachment file${missingList.length === 1 ? '' : 's'}: ${preview.join(', ')}${suffix}.`);
          }
          if (attachmentTotals.uploaded) {
            parts.push(`Uploaded ${attachmentTotals.uploaded} attachment${attachmentTotals.uploaded === 1 ? '' : 's'}.`);
          }
          if (attachmentTotals.failed) {
            parts.push(`Attachment uploads failed for ${attachmentTotals.failed} file${attachmentTotals.failed === 1 ? '' : 's'}.`);
          }
          const hasIssues = failed || attachmentTotals.failed || missingAttachments.size;
          setStatus(els.importStatus, parts.join(' '), hasIssues ? 'error' : 'success');
          if (success) {
            await Promise.all([refreshDashboard(), refreshEntries()]);
          }
          if (els.importFile) els.importFile.value = '';
          if (els.importAttachments) els.importAttachments.value = '';
        } catch (err) {
          console.error('CSV import failed', err);
          setStatus(els.importStatus, err?.message || 'Unable to import applications.', 'error');
        }
      });
    }
  };

  const initFilters = () => {
    const range = defaultRange();
    updateRangeInputs(range);
    if (els.filterReset) {
      els.filterReset.addEventListener('click', () => {
        const next = defaultRange();
        updateRangeInputs(next);
        refreshDashboard();
      });
    }
    if (els.filterRefresh) {
      els.filterRefresh.addEventListener('click', () => refreshDashboard());
    }
  };

  const initAuth = async () => {
    updateConfigStatus();
    const stored = loadAuth();
    if (authIsValid(stored)) {
      state.auth = stored;
    }
    try {
      await handleAuthRedirect();
    } catch (err) {
      console.error('Auth redirect failed', err);
      setStatus(els.authStatus, err?.message || 'Sign-in failed.', 'error');
    }
    updateAuthUI();
    if (els.signIn) {
      els.signIn.addEventListener('click', async () => {
        if (!config.cognitoDomain || !config.cognitoClientId || !config.cognitoRedirect) {
          setStatus(els.authStatus, 'Cognito settings are missing.', 'error');
          return;
        }
        try {
          const url = await buildAuthorizeUrl();
          window.location.assign(url);
        } catch (err) {
          console.error('Sign-in failed', err);
          setStatus(els.authStatus, err?.message || 'Unable to start sign-in.', 'error');
        }
      });
    }
    if (els.signOut) {
      els.signOut.addEventListener('click', () => {
        clearAuth();
        clearEntryEditMode('Sign in to save entries.', 'info');
        if (els.entryForm) {
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          setEntryType('application');
          resetEntryDateFields('application');
          clearAttachmentInputs();
        }
        updateAuthUI();
        refreshDashboard();
        refreshEntries();
      });
    }
  };

  const init = async () => {
    initTabs();
    initFilters();
    initEntryForm();
    initImport();
    initEntryList();
    initExport();
    await initAuth();
    updateAuthUI();
    refreshDashboard();
    refreshEntries();
  };

  init();
})();
