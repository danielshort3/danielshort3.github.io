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
    calendarGrid: $('[data-jobtrack="calendar-grid"]')
  };

  const STORAGE_KEY = 'jobTrackerAuth';
  const STATE_KEY = 'jobTrackerAuthState';
  const VERIFIER_KEY = 'jobTrackerCodeVerifier';

  const state = {
    auth: null,
    lineChart: null,
    statusChart: null,
    range: null
  };

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const formatDateInput = (date) => date.toISOString().slice(0, 10);
  const parseDateInput = (value) => {
    if (!value) return null;
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };

  const toTitle = (value) => value
    .toLowerCase()
    .split(' ')
    .filter(Boolean)
    .map(word => word[0].toUpperCase() + word.slice(1))
    .join(' ');

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
      els.recentList.appendChild(li);
    });
  };

  const refreshRecent = async () => {
    if (!els.recentList) return;
    if (!config.apiBase) {
      renderRecentList([]);
      return;
    }
    if (!authIsValid(state.auth)) {
      renderRecentList([]);
      return;
    }
    try {
      const data = await requestJson('/api/applications?limit=5');
      renderRecentList(data.items || []);
    } catch (err) {
      console.error('Recent applications failed', err);
      renderRecentList([]);
    }
  };

  const submitApplication = async (payload) => {
    if (!els.formStatus) return;
    if (!authIsValid(state.auth)) {
      setStatus(els.formStatus, 'Sign in to save new applications.', 'error');
      return;
    }
    try {
      setStatus(els.formStatus, 'Saving application...', 'info');
      await requestJson('/api/applications', { method: 'POST', body: payload });
      setStatus(els.formStatus, 'Saved. Updating dashboards...', 'success');
      await sleep(200);
      await Promise.all([refreshDashboard(), refreshRecent()]);
    } catch (err) {
      console.error('Application save failed', err);
      setStatus(els.formStatus, err?.message || 'Unable to save application.', 'error');
    }
  };

  const initForm = () => {
    if (!els.form) return;
    els.form.addEventListener('submit', (event) => {
      event.preventDefault();
      const formData = new FormData(els.form);
      const company = (formData.get('company') || '').toString().trim();
      const title = (formData.get('title') || '').toString().trim();
      const appliedDate = (formData.get('appliedDate') || '').toString().trim();
      const status = (formData.get('status') || 'Applied').toString().trim();
      const notes = (formData.get('notes') || '').toString().trim();
      if (!company || !title || !appliedDate) {
        setStatus(els.formStatus, 'Company, role title, and applied date are required.', 'error');
        return;
      }
      submitApplication({ company, title, appliedDate, status, notes });
      els.form.reset();
    });
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
        updateAuthUI();
        refreshDashboard();
        refreshRecent();
      });
    }
  };

  const init = async () => {
    initFilters();
    initForm();
    if (els.recentRefresh) {
      els.recentRefresh.addEventListener('click', () => refreshRecent());
    }
    await initAuth();
    updateAuthUI();
    refreshDashboard();
    refreshRecent();
  };

  init();
})();
