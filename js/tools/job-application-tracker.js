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
    form: $('[data-jobtrack="application-form"]'),
    formStatus: $('[data-jobtrack="form-status"]'),
    companyInput: $('#jobtrack-company'),
    titleInput: $('#jobtrack-title'),
    jobUrlInput: $('#jobtrack-job-url'),
    locationInput: $('#jobtrack-location'),
    sourceInput: $('#jobtrack-source'),
    dateInput: $('#jobtrack-date'),
    postingDateInput: $('#jobtrack-posting-date'),
    postingUnknownInput: $('#jobtrack-posting-unknown'),
    statusInput: $('#jobtrack-status'),
    notesInput: $('#jobtrack-notes'),
    applicationSubmit: $('[data-jobtrack="application-submit"]'),
    recentList: $('[data-jobtrack="recent-list"]'),
    recentRefresh: $('[data-jobtrack="refresh-recent"]'),
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
    lineRange: $('[data-jobtrack="line-range"]'),
    lineTotal: $('[data-jobtrack="line-total"]'),
    statusTotal: $('[data-jobtrack="status-total"]'),
    calendarRange: $('[data-jobtrack="calendar-range"]'),
    lineOverlay: $('[data-jobtrack="line-overlay"]'),
    statusOverlay: $('[data-jobtrack="status-overlay"]'),
    calendarGrid: $('[data-jobtrack="calendar-grid"]'),
    resumeInput: $('#jobtrack-resume'),
    coverInput: $('#jobtrack-cover'),
    importFile: $('#jobtrack-import-file'),
    importAttachments: $('#jobtrack-import-attachments'),
    importSubmit: $('[data-jobtrack="import-submit"]'),
    importTemplate: $('[data-jobtrack="import-template"]'),
    importStatus: $('[data-jobtrack="import-status"]'),
    recentStatus: $('[data-jobtrack="recent-status"]'),
    prospectForm: $('[data-jobtrack="prospect-form"]'),
    prospectStatus: $('[data-jobtrack="prospect-status"]'),
    prospectList: $('[data-jobtrack="prospect-list"]'),
    prospectRefresh: $('[data-jobtrack="refresh-prospects"]'),
    prospectListStatus: $('[data-jobtrack="prospect-list-status"]'),
    prospectCompanyInput: $('#jobtrack-prospect-company'),
    prospectTitleInput: $('#jobtrack-prospect-title'),
    prospectUrlInput: $('#jobtrack-prospect-url'),
    prospectLocationInput: $('#jobtrack-prospect-location'),
    prospectSourceInput: $('#jobtrack-prospect-source'),
    prospectPostingDateInput: $('#jobtrack-prospect-posting-date'),
    prospectPostingUnknownInput: $('#jobtrack-prospect-posting-unknown'),
    prospectAppliedDateInput: $('#jobtrack-prospect-applied-date'),
    prospectCaptureDateInput: $('#jobtrack-prospect-capture-date'),
    prospectStatusInput: $('#jobtrack-prospect-status'),
    prospectNotesInput: $('#jobtrack-prospect-notes'),
    prospectSubmit: $('[data-jobtrack="prospect-submit"]')
  };

  const tabs = {
    buttons: $$('[data-jobtrack-tab]'),
    panels: $$('[data-jobtrack-panel]')
  };

  const STORAGE_KEY = 'jobTrackerAuth';
  const STATE_KEY = 'jobTrackerAuthState';
  const VERIFIER_KEY = 'jobTrackerCodeVerifier';
  const CSV_TEMPLATE = 'company,title,jobUrl,location,source,postingDate,appliedDate,status,notes,resumeFile,coverLetterFile\nAcme Corp,Data Analyst,https://acme.com/jobs/123,Remote,LinkedIn,2025-01-10,2025-01-15,Applied,Reached out to recruiter,Acme-Resume.pdf,Acme-Cover.pdf';

  const state = {
    auth: null,
    lineChart: null,
    statusChart: null,
    range: null,
    editingApplicationId: null,
    editingApplication: null,
    editingProspectId: null,
    editingProspect: null,
    recentItems: new Map(),
    prospectItems: new Map(),
    isResettingApplication: false,
    isResettingProspect: false
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

  const storeRecentItems = (items = []) => {
    state.recentItems = new Map();
    items.forEach((item) => {
      if (item && item.applicationId) state.recentItems.set(item.applicationId, item);
    });
  };

  const storeProspectItems = (items = []) => {
    state.prospectItems = new Map();
    items.forEach((item) => {
      if (item && item.applicationId) state.prospectItems.set(item.applicationId, item);
    });
  };

  const setApplicationEditMode = (item) => {
    if (!item) return;
    state.editingApplicationId = item.applicationId || null;
    state.editingApplication = item;
    if (els.companyInput) els.companyInput.value = item.company || '';
    if (els.titleInput) els.titleInput.value = item.title || '';
    if (els.jobUrlInput) els.jobUrlInput.value = item.jobUrl || '';
    if (els.locationInput) els.locationInput.value = item.location || '';
    if (els.sourceInput) els.sourceInput.value = item.source || '';
    if (els.dateInput) els.dateInput.value = item.appliedDate || '';
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, item.postingDate || '');
    if (els.statusInput) els.statusInput.value = item.status || 'Applied';
    if (els.notesInput) els.notesInput.value = item.notes || '';
    clearAttachmentInputs();
    if (els.applicationSubmit) els.applicationSubmit.textContent = 'Update application';
    if (els.formStatus) {
      const label = [item.title, item.company].filter(Boolean).join(' · ') || 'application';
      setStatus(els.formStatus, `Editing ${label}. Save to update or clear to cancel.`, 'info');
    }
    activateTab('applications', true);
  };

  const clearApplicationEditMode = (message = 'Ready to log a new application.', tone = '') => {
    state.editingApplicationId = null;
    state.editingApplication = null;
    if (els.applicationSubmit) els.applicationSubmit.textContent = 'Save application';
    if (els.formStatus && message) setStatus(els.formStatus, message, tone);
  };

  const setProspectEditMode = (item) => {
    if (!item) return;
    state.editingProspectId = item.applicationId || null;
    state.editingProspect = item;
    if (els.prospectCompanyInput) els.prospectCompanyInput.value = item.company || '';
    if (els.prospectTitleInput) els.prospectTitleInput.value = item.title || '';
    if (els.prospectUrlInput) els.prospectUrlInput.value = item.jobUrl || '';
    if (els.prospectLocationInput) els.prospectLocationInput.value = item.location || '';
    if (els.prospectSourceInput) els.prospectSourceInput.value = item.source || '';
    setUnknownDateValue(els.prospectPostingDateInput, els.prospectPostingUnknownInput, item.postingDate || '');
    if (els.prospectAppliedDateInput) els.prospectAppliedDateInput.value = item.appliedDate || '';
    if (els.prospectCaptureDateInput) {
      els.prospectCaptureDateInput.value = item.captureDate || formatDateInput(new Date());
    }
    if (els.prospectStatusInput) els.prospectStatusInput.value = item.status || 'Active';
    if (els.prospectNotesInput) els.prospectNotesInput.value = item.notes || '';
    if (els.prospectSubmit) els.prospectSubmit.textContent = 'Update prospect';
    if (els.prospectStatus) {
      const label = [item.title, item.company].filter(Boolean).join(' · ') || 'prospect';
      setStatus(els.prospectStatus, `Editing ${label}. Save to update or clear to cancel.`, 'info');
    }
    activateTab('prospects', true);
  };

  const clearProspectEditMode = (message = 'Ready to save prospects.', tone = '') => {
    state.editingProspectId = null;
    state.editingProspect = null;
    if (els.prospectSubmit) els.prospectSubmit.textContent = 'Save prospect';
    if (els.prospectStatus && message) setStatus(els.prospectStatus, message, tone);
  };

  const activateTab = (name, shouldFocus = false) => {
    if (!tabs.buttons.length || !tabs.panels.length) return;
    tabs.buttons.forEach((button) => {
      const selected = button.dataset.jobtrackTab === name;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    tabs.panels.forEach((panel) => {
      panel.hidden = panel.dataset.jobtrackPanel !== name;
    });
    if (shouldFocus) {
      const activeButton = tabs.buttons.find(button => button.dataset.jobtrackTab === name);
      if (activeButton) activeButton.focus();
    }
    if (name === 'dashboard') {
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
    if (els.signIn) els.signIn.hidden = authed;
    if (els.signOut) els.signOut.hidden = !authed;
    if (els.authStatus) {
      if (authed) {
        const claims = state.auth?.claims || parseJwt(state.auth?.idToken || '') || {};
        const label = claims.email || claims['cognito:username'] || claims.username || 'Signed in';
        els.authStatus.textContent = `Signed in as ${label}.`;
      } else {
        els.authStatus.textContent = 'Not signed in.';
      }
    }
    if (els.importStatus) {
      setStatus(els.importStatus, authed ? 'Ready to import applications.' : 'Sign in to import applications.', authed ? '' : 'info');
    }
    if (els.formStatus && !state.editingApplicationId) {
      setStatus(els.formStatus, authed ? 'Ready to log a new application.' : 'Sign in to save new applications.', authed ? '' : 'info');
    }
    if (els.recentStatus) {
      setStatus(els.recentStatus, authed ? 'Select an attachment to download.' : 'Sign in to download attachments.', authed ? '' : 'info');
    }
    if (els.prospectStatus && !state.editingProspectId) {
      setStatus(els.prospectStatus, authed ? 'Ready to save prospects.' : 'Sign in to save prospects.', authed ? '' : 'info');
    }
    if (els.prospectListStatus) {
      setStatus(els.prospectListStatus, authed ? 'Select a prospect to update.' : 'Sign in to manage prospects.', authed ? '' : 'info');
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

  const resetApplicationDateFields = () => {
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, '');
  };

  const resetProspectDateFields = () => {
    const today = formatDateInput(new Date());
    if (els.prospectCaptureDateInput) {
      els.prospectCaptureDateInput.value = els.prospectCaptureDateInput.value || today;
    }
    if (els.prospectAppliedDateInput) {
      els.prospectAppliedDateInput.value = '';
    }
    const postingValue = els.prospectPostingDateInput?.value || '';
    setUnknownDateValue(els.prospectPostingDateInput, els.prospectPostingUnknownInput, postingValue);
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

  const refreshDashboard = async () => {
    if (!els.dashboard || !els.dashboardStatus) return;
    if (!config.apiBase) {
      setStatus(els.dashboardStatus, 'Set the API base URL to load dashboards.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.dashboardStatus, 'Sign in to load your dashboards.', 'info');
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
      const [summary, timeline, statuses, calendar] = await Promise.all([
        requestJson(`/api/analytics/summary?${query}`),
        requestJson(`/api/analytics/applications-over-time?${query}`),
        requestJson(`/api/analytics/status-breakdown?${query}`),
        requestJson(`/api/analytics/calendar?${query}`)
      ]);
      updateKpis(summary);
      const series = timeline.series || [];
      const statusSeries = statuses.statuses || [];
      updateLineChart(series, rangeLabel);
      updateStatusChart(statusSeries);
      buildCalendar(calendar.days || [], rangeLabel, range);
      setOverlay(els.lineOverlay, series.length ? '' : 'No activity yet.');
      setOverlay(els.statusOverlay, statusSeries.length ? '' : 'No statuses yet.');
      setStatus(els.dashboardStatus, `Loaded ${summary.totalApplications || 0} applications.`, 'success');
    } catch (err) {
      console.error('Dashboard load failed', err);
      setOverlay(els.lineOverlay, 'Unable to load chart.');
      setOverlay(els.statusOverlay, 'Unable to load chart.');
      setStatus(els.dashboardStatus, err?.message || 'Unable to load dashboards.', 'error');
    } finally {
      if (els.dashboard) els.dashboard.setAttribute('aria-busy', 'false');
    }
  };

  const renderRecentList = (items = []) => {
    if (!els.recentList) return;
    els.recentList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('li');
      empty.className = 'jobtrack-recent-empty';
      empty.textContent = 'No applications found in this range yet.';
      els.recentList.appendChild(empty);
      return;
    }
    items.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'jobtrack-recent-item';
      const title = document.createElement('div');
      title.className = 'jobtrack-recent-title';
      title.textContent = `${item.title || 'Role'} · ${item.company || 'Company'}`;
      const meta = document.createElement('div');
      meta.className = 'jobtrack-recent-meta';
      const date = item.appliedDate ? new Date(`${item.appliedDate}T00:00:00Z`) : null;
      const dateLabel = date ? date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'No date';
      meta.textContent = `${dateLabel} · ${toTitle(item.status || 'Applied')}`;
      li.appendChild(title);
      li.appendChild(meta);
      const attachments = Array.isArray(item.attachments)
        ? item.attachments.filter(attachment => attachment && attachment.key)
        : [];
      if (attachments.length) {
        const wrap = document.createElement('div');
        wrap.className = 'jobtrack-recent-attachments';
        attachments.forEach((attachment) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'btn-ghost jobtrack-attachment-btn';
          button.dataset.jobtrackAttachment = 'download';
          button.dataset.key = attachment.key;
          if (attachment.filename) button.dataset.filename = attachment.filename;
          const label = getAttachmentLabel(attachment);
          const name = attachment.filename ? `${label}: ${attachment.filename}` : `Download ${label}`;
          button.textContent = name;
          button.title = name;
          wrap.appendChild(button);
        });
        li.appendChild(wrap);
      }
      if (item.applicationId) {
        const actions = document.createElement('div');
        actions.className = 'jobtrack-recent-actions';
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'btn-ghost jobtrack-recent-action';
        editBtn.dataset.jobtrackApplication = 'edit';
        editBtn.dataset.id = item.applicationId;
        editBtn.textContent = 'Edit';
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'btn-ghost jobtrack-recent-action';
        deleteBtn.dataset.jobtrackApplication = 'delete';
        deleteBtn.dataset.id = item.applicationId;
        deleteBtn.textContent = 'Delete';
        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);
        li.appendChild(actions);
      }
      els.recentList.appendChild(li);
    });
  };

  const refreshRecent = async () => {
    if (!els.recentList) return;
    if (!config.apiBase) {
      storeRecentItems([]);
      renderRecentList([]);
      return;
    }
    if (!authIsValid(state.auth)) {
      storeRecentItems([]);
      renderRecentList([]);
      return;
    }
    try {
      const data = await requestJson('/api/applications?limit=5');
      const items = data.items || [];
      storeRecentItems(items);
      renderRecentList(items);
    } catch (err) {
      console.error('Recent applications failed', err);
      storeRecentItems([]);
      renderRecentList([]);
    }
  };

  const deleteApplication = async (applicationId) => {
    if (!applicationId) return;
    if (!config.apiBase) {
      setStatus(els.recentStatus, 'Set the API base URL to delete applications.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.recentStatus, 'Sign in to delete applications.', 'error');
      return;
    }
    const item = state.recentItems.get(applicationId);
    const label = [item?.title, item?.company].filter(Boolean).join(' · ') || 'this application';
    if (!confirmAction(`Delete ${label}? This cannot be undone.`)) return;
    try {
      setStatus(els.recentStatus, 'Deleting application...', 'info');
      await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, { method: 'DELETE' });
      if (state.editingApplicationId === applicationId) {
        clearApplicationEditMode('Ready to log a new application.');
        if (els.form) {
          state.isResettingApplication = true;
          els.form.reset();
          state.isResettingApplication = false;
          clearAttachmentInputs();
        }
      }
      setStatus(els.recentStatus, 'Application deleted.', 'success');
      await Promise.all([refreshDashboard(), refreshRecent()]);
    } catch (err) {
      console.error('Application delete failed', err);
      setStatus(els.recentStatus, err?.message || 'Unable to delete application.', 'error');
    }
  };

  const renderProspects = (items = [], emptyLabel = 'No prospects yet.') => {
    if (!els.prospectList) return;
    els.prospectList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('li');
      empty.className = 'jobtrack-prospect-empty';
      empty.textContent = emptyLabel;
      els.prospectList.appendChild(empty);
      return;
    }
    items.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'jobtrack-prospect-item';
      const title = document.createElement('div');
      title.className = 'jobtrack-prospect-title';
      title.textContent = `${item.title || 'Role'} · ${item.company || 'Company'}`;
      li.appendChild(title);

      const metaBits = [];
      const statusLabel = toTitle(item.status || 'Active');
      if (statusLabel) metaBits.push(statusLabel);
      if (item.location) metaBits.push(item.location);
      if (item.source) metaBits.push(item.source);
      if (metaBits.length) {
        const meta = document.createElement('div');
        meta.className = 'jobtrack-prospect-meta';
        meta.textContent = metaBits.join(' · ');
        li.appendChild(meta);
      }

      if (item.jobUrl) {
        const link = document.createElement('a');
        link.className = 'jobtrack-prospect-link';
        link.href = item.jobUrl;
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = 'Open job posting';
        li.appendChild(link);
      }

      if (item.notes) {
        const notes = document.createElement('p');
        notes.className = 'jobtrack-prospect-notes';
        notes.textContent = item.notes;
        li.appendChild(notes);
      }

      if (item.applicationId) {
        const actions = document.createElement('div');
        actions.className = 'jobtrack-prospect-actions';
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'btn-ghost jobtrack-prospect-action';
        editBtn.dataset.jobtrackProspect = 'edit';
        editBtn.dataset.id = item.applicationId;
        editBtn.textContent = 'Edit';
        const applyBtn = document.createElement('button');
        applyBtn.type = 'button';
        applyBtn.className = 'btn-ghost jobtrack-prospect-action';
        applyBtn.dataset.jobtrackProspect = 'apply';
        applyBtn.dataset.id = item.applicationId;
        applyBtn.textContent = 'Mark applied';
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'btn-ghost jobtrack-prospect-action';
        toggle.dataset.jobtrackProspect = 'toggle';
        toggle.dataset.id = item.applicationId;
        const isInactive = (item.status || '').toString().toLowerCase() === 'inactive';
        toggle.dataset.nextStatus = isInactive ? 'Active' : 'Inactive';
        toggle.textContent = isInactive ? 'Mark active' : 'Mark inactive';
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'btn-ghost jobtrack-prospect-action';
        deleteBtn.dataset.jobtrackProspect = 'delete';
        deleteBtn.dataset.id = item.applicationId;
        deleteBtn.textContent = 'Delete';
        actions.appendChild(editBtn);
        actions.appendChild(applyBtn);
        actions.appendChild(toggle);
        actions.appendChild(deleteBtn);
        li.appendChild(actions);
      }

      els.prospectList.appendChild(li);
    });
  };

  const refreshProspects = async () => {
    if (!els.prospectList) return;
    if (!config.apiBase) {
      storeProspectItems([]);
      renderProspects([], 'Set the API base URL to load prospects.');
      return;
    }
    if (!authIsValid(state.auth)) {
      storeProspectItems([]);
      renderProspects([], 'Sign in to load prospects.');
      return;
    }
    try {
      const data = await requestJson('/api/prospects?limit=8');
      const items = data.items || [];
      storeProspectItems(items);
      renderProspects(items);
    } catch (err) {
      console.error('Prospect load failed', err);
      storeProspectItems([]);
      renderProspects([], 'Unable to load prospects.');
    }
  };

  const deleteProspect = async (prospectId) => {
    if (!prospectId) return;
    if (!config.apiBase) {
      setStatus(els.prospectListStatus, 'Set the API base URL to delete prospects.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.prospectListStatus, 'Sign in to delete prospects.', 'error');
      return;
    }
    const item = state.prospectItems.get(prospectId);
    const label = [item?.title, item?.company].filter(Boolean).join(' · ') || 'this prospect';
    if (!confirmAction(`Delete ${label}? This cannot be undone.`)) return;
    try {
      setStatus(els.prospectListStatus, 'Deleting prospect...', 'info');
      await requestJson(`/api/applications/${encodeURIComponent(prospectId)}`, { method: 'DELETE' });
      if (state.editingProspectId === prospectId) {
        clearProspectEditMode();
        if (els.prospectForm) {
          state.isResettingProspect = true;
          els.prospectForm.reset();
          state.isResettingProspect = false;
        }
      }
      setStatus(els.prospectListStatus, 'Prospect deleted.', 'success');
      await refreshProspects();
    } catch (err) {
      console.error('Prospect delete failed', err);
      setStatus(els.prospectListStatus, err?.message || 'Unable to delete prospect.', 'error');
    }
  };

  const submitProspect = async (payload) => {
    if (!els.prospectStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.prospectStatus, 'Sign in to save prospects.', 'error');
      return false;
    }
    const editingId = state.editingProspectId;
    try {
      setStatus(els.prospectStatus, editingId ? 'Updating prospect...' : 'Saving prospect...', 'info');
      if (editingId) {
        await requestJson(`/api/prospects/${encodeURIComponent(editingId)}`, { method: 'PATCH', body: payload });
      } else {
        await requestJson('/api/prospects', { method: 'POST', body: payload });
      }
      clearProspectEditMode(editingId ? 'Prospect updated.' : 'Prospect saved.', 'success');
      await sleep(200);
      await refreshProspects();
      return true;
    } catch (err) {
      console.error('Prospect save failed', err);
      setStatus(els.prospectStatus, err?.message || 'Unable to save prospect.', 'error');
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
    return payload;
  };

  const convertProspectToApplication = async (payload, prospectId) => {
    if (!els.prospectStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.prospectStatus, 'Sign in to save applications.', 'error');
      return false;
    }
    try {
      setStatus(els.prospectStatus, 'Moving prospect to applications...', 'info');
      await requestJson('/api/applications', { method: 'POST', body: payload });
      let deleteError = null;
      if (prospectId) {
        try {
          await requestJson(`/api/applications/${encodeURIComponent(prospectId)}`, { method: 'DELETE' });
        } catch (err) {
          deleteError = err;
        }
      }
      clearProspectEditMode(
        deleteError
          ? 'Application saved, but the prospect could not be removed.'
          : (prospectId ? 'Prospect moved to applications.' : 'Application saved.'),
        deleteError ? 'error' : 'success'
      );
      await sleep(200);
      await Promise.all([refreshProspects(), refreshRecent(), refreshDashboard()]);
      return true;
    } catch (err) {
      console.error('Prospect conversion failed', err);
      setStatus(els.prospectStatus, err?.message || 'Unable to move prospect to applications.', 'error');
      return false;
    }
  };

  const updateProspectStatus = async (prospectId, nextStatus) => {
    if (!prospectId) return;
    try {
      setStatus(els.prospectListStatus, 'Updating prospect...', 'info');
      await requestJson(`/api/prospects/${encodeURIComponent(prospectId)}`, {
        method: 'PATCH',
        body: { status: nextStatus }
      });
      setStatus(els.prospectListStatus, 'Prospect updated.', 'success');
      await refreshProspects();
    } catch (err) {
      console.error('Prospect update failed', err);
      setStatus(els.prospectListStatus, err?.message || 'Unable to update prospect.', 'error');
    }
  };

  const applyProspect = async (prospectId) => {
    if (!prospectId) return;
    const item = state.prospectItems.get(prospectId);
    if (!item) return;
    if (!authIsValid(state.auth)) {
      setStatus(els.prospectListStatus, 'Sign in to move prospects.', 'error');
      return;
    }
    const defaultDate = formatDateInput(new Date());
    if (typeof window === 'undefined' || typeof window.prompt !== 'function') {
      setProspectEditMode(item);
      if (els.prospectAppliedDateInput) {
        els.prospectAppliedDateInput.value = defaultDate;
        els.prospectAppliedDateInput.focus();
      }
      setStatus(els.prospectStatus, 'Enter an applied date to move this prospect.', 'info');
      return;
    }
    const response = window.prompt('Applied date (YYYY-MM-DD):', defaultDate);
    if (response === null) return;
    const appliedDate = response.toString().trim();
    if (!appliedDate || !parseDateInput(appliedDate)) {
      setStatus(els.prospectListStatus, 'Applied date must be valid (YYYY-MM-DD).', 'error');
      return;
    }
    const payload = buildApplicationPayloadFromProspect(item, appliedDate);
    await convertProspectToApplication(payload, prospectId);
  };

  const initProspects = () => {
    if (els.prospectForm) {
      initUnknownDateToggle(els.prospectPostingDateInput, els.prospectPostingUnknownInput, true);
      resetProspectDateFields();
      els.prospectForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(els.prospectForm);
        const company = (formData.get('company') || '').toString().trim();
        const title = (formData.get('title') || '').toString().trim();
        const jobUrl = normalizeUrl(formData.get('jobUrl'));
        const location = (formData.get('location') || '').toString().trim();
        const source = (formData.get('source') || '').toString().trim();
        const postingDate = (formData.get('postingDate') || '').toString().trim();
        const postingUnknown = Boolean(formData.get('postingDateUnknown'));
        const appliedDate = (formData.get('appliedDate') || '').toString().trim();
        const captureDate = (formData.get('captureDate') || '').toString().trim();
        const status = (formData.get('status') || 'Active').toString().trim();
        const notes = (formData.get('notes') || '').toString().trim();
        if (!company || !title || !jobUrl) {
          setStatus(els.prospectStatus, 'Company, role title, and job URL are required.', 'error');
          return;
        }
        if (!postingUnknown && !postingDate) {
          setStatus(els.prospectStatus, 'Add a posting date or mark it as unknown.', 'error');
          return;
        }
        if (postingDate && !parseDateInput(postingDate)) {
          setStatus(els.prospectStatus, 'Posting date must be valid.', 'error');
          return;
        }
        if (appliedDate && !parseDateInput(appliedDate)) {
          setStatus(els.prospectStatus, 'Applied date must be valid.', 'error');
          return;
        }
        if (!captureDate) {
          setStatus(els.prospectStatus, 'Capture date is required.', 'error');
          return;
        }
        if (!parseDateInput(captureDate)) {
          setStatus(els.prospectStatus, 'Capture date must be valid.', 'error');
          return;
        }
        if (appliedDate) {
          const applicationPayload = {
            company,
            title,
            appliedDate,
            status: 'Applied',
            notes
          };
          if (jobUrl) applicationPayload.jobUrl = jobUrl;
          if (location) applicationPayload.location = location;
          if (source) applicationPayload.source = source;
          if (postingUnknown) {
            applicationPayload.postingDate = null;
          } else if (postingDate) {
            applicationPayload.postingDate = postingDate;
          }
          const ok = await convertProspectToApplication(applicationPayload, state.editingProspectId);
          if (ok) {
            state.isResettingProspect = true;
            els.prospectForm.reset();
            state.isResettingProspect = false;
            resetProspectDateFields();
          }
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
          state.isResettingProspect = true;
          els.prospectForm.reset();
          state.isResettingProspect = false;
          resetProspectDateFields();
        }
      });
      els.prospectForm.addEventListener('reset', () => {
        if (state.isResettingProspect) return;
        resetProspectDateFields();
        clearProspectEditMode();
      });
    }

    if (els.prospectRefresh) {
      els.prospectRefresh.addEventListener('click', () => refreshProspects());
    }

    if (els.prospectList) {
      els.prospectList.addEventListener('click', (event) => {
        const target = event.target && event.target.closest
          ? event.target.closest('[data-jobtrack-prospect]')
          : null;
        if (!target) return;
        event.preventDefault();
        const action = target.dataset.jobtrackProspect;
        const prospectId = (target.dataset.id || '').trim();
        if (action === 'toggle') {
          const nextStatus = (target.dataset.nextStatus || 'Inactive').trim();
          updateProspectStatus(prospectId, nextStatus);
        } else if (action === 'apply') {
          applyProspect(prospectId);
        } else if (action === 'edit') {
          const item = state.prospectItems.get(prospectId);
          if (item) setProspectEditMode(item);
        } else if (action === 'delete') {
          deleteProspect(prospectId);
        }
      });
    }
  };

  const initAttachmentDownloads = () => {
    if (!els.recentList) return;
    els.recentList.addEventListener('click', async (event) => {
      const target = event.target && event.target.closest
        ? event.target.closest('[data-jobtrack-attachment="download"]')
        : null;
      if (!target) return;
      event.preventDefault();
      const key = (target.dataset.key || '').trim();
      if (!key) return;
      try {
        setStatus(els.recentStatus, 'Preparing download...', 'info');
        const data = await requestAttachmentDownload(key);
        if (data?.downloadUrl) {
          window.open(data.downloadUrl, '_blank', 'noopener');
          setStatus(els.recentStatus, 'Download ready.', 'success');
        } else {
          throw new Error('Download unavailable.');
        }
      } catch (err) {
        console.error('Attachment download failed', err);
        setStatus(els.recentStatus, err?.message || 'Unable to download attachment.', 'error');
      }
    });
  };

  const initApplicationActions = () => {
    if (!els.recentList) return;
    els.recentList.addEventListener('click', (event) => {
      const target = event.target && event.target.closest
        ? event.target.closest('[data-jobtrack-application]')
        : null;
      if (!target) return;
      event.preventDefault();
      const action = target.dataset.jobtrackApplication;
      const applicationId = (target.dataset.id || '').trim();
      if (!applicationId) return;
      if (action === 'edit') {
        const item = state.recentItems.get(applicationId);
        if (item) setApplicationEditMode(item);
      } else if (action === 'delete') {
        deleteApplication(applicationId);
      }
    });
  };

  const submitApplication = async (payload, attachments = []) => {
    if (!els.formStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.formStatus, 'Sign in to save new applications.', 'error');
      return false;
    }
    const editingId = state.editingApplicationId;
    const editingItem = state.editingApplication;
    try {
      setStatus(els.formStatus, editingId ? 'Updating application...' : 'Saving application...', 'info');
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
          setStatus(els.formStatus, `Uploading ${attachments.length} ${label}...`, 'info');
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
        clearApplicationEditMode(
          editingId ? 'Updated application, but attachments failed to upload.' : 'Saved application, but attachments failed to upload.',
          'error'
        );
      } else {
        clearApplicationEditMode(
          editingId ? 'Application updated. Refreshing dashboards...' : 'Application saved. Updating dashboards...',
          'success'
        );
      }
      await sleep(200);
      await Promise.all([refreshDashboard(), refreshRecent()]);
      return true;
    } catch (err) {
      console.error('Application save failed', err);
      setStatus(els.formStatus, err?.message || 'Unable to save application.', 'error');
      return false;
    }
  };

  const initForm = () => {
    if (!els.form) return;
    initUnknownDateToggle(els.postingDateInput, els.postingUnknownInput, true);
    els.form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(els.form);
      const company = (formData.get('company') || '').toString().trim();
      const title = (formData.get('title') || '').toString().trim();
      const jobUrl = normalizeUrl(formData.get('jobUrl'));
      const location = (formData.get('location') || '').toString().trim();
      const source = (formData.get('source') || '').toString().trim();
      const appliedDate = (formData.get('appliedDate') || '').toString().trim();
      const postingDate = (formData.get('postingDate') || '').toString().trim();
      const postingUnknown = Boolean(formData.get('postingDateUnknown'));
      const status = (formData.get('status') || 'Applied').toString().trim();
      const notes = (formData.get('notes') || '').toString().trim();
      if (!company || !title || !appliedDate) {
        setStatus(els.formStatus, 'Company, role title, and applied date are required.', 'error');
        return;
      }
      if (!postingUnknown && !postingDate) {
        setStatus(els.formStatus, 'Add a posting date or mark it as unknown.', 'error');
        return;
      }
      if (postingDate && !parseDateInput(postingDate)) {
        setStatus(els.formStatus, 'Posting date must be valid.', 'error');
        return;
      }
      const attachments = collectAttachments();
      const editing = state.editingApplication;
      const payload = { company, title, appliedDate, notes };
      if (jobUrl || editing?.jobUrl) payload.jobUrl = jobUrl;
      if (location || editing?.location) payload.location = location;
      if (source || editing?.source) payload.source = source;
      if (postingUnknown) {
        payload.postingDate = null;
      } else if (postingDate) {
        payload.postingDate = postingDate;
      }
      const existingStatus = editing?.status ? editing.status.toString().trim().toLowerCase() : '';
      if (!editing || !existingStatus || status.toLowerCase() !== existingStatus) {
        payload.status = status;
      }
      const ok = await submitApplication(payload, attachments);
      if (ok) {
        state.isResettingApplication = true;
        els.form.reset();
        state.isResettingApplication = false;
        clearAttachmentInputs();
        resetApplicationDateFields();
      }
    });
    els.form.addEventListener('reset', () => {
      clearAttachmentInputs();
      resetApplicationDateFields();
      if (state.isResettingApplication) return;
      clearApplicationEditMode();
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
            await Promise.all([refreshDashboard(), refreshRecent()]);
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
        clearApplicationEditMode('Sign in to save new applications.', 'info');
        clearProspectEditMode('Sign in to save prospects.', 'info');
        updateAuthUI();
        refreshDashboard();
        refreshRecent();
        refreshProspects();
      });
    }
  };

  const init = async () => {
    initTabs();
    initFilters();
    initForm();
    initImport();
    initAttachmentDownloads();
    initApplicationActions();
    initProspects();
    if (els.recentRefresh) {
      els.recentRefresh.addEventListener('click', () => refreshRecent());
    }
    await initAuth();
    updateAuthUI();
    refreshDashboard();
    refreshRecent();
    refreshProspects();
  };

  init();
})();
