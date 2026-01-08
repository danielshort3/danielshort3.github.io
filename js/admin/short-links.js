/* Short links admin dashboard (token-based). */
(() => {
  'use strict';

  const STORAGE_KEY = 'shortlinks_admin_token';
  const DEFAULT_BASE_PATH = 'go';
  const DEFAULT_PUBLIC_ORIGIN = 'https://danielshort.me';

  const authForm = document.querySelector('[data-shortlinks="auth"]');
  const editorForm = document.querySelector('[data-shortlinks="editor"]');
  const listEl = document.querySelector('[data-shortlinks="list"]');
  if (!authForm || !editorForm || !listEl) return;

  const accessDetails = document.querySelector('[data-shortlinks="access-details"]');
  const accessMetaEl = document.querySelector('[data-shortlinks="access-meta"]');
  const filterInput = document.querySelector('[data-shortlinks="filter"]');
  const countEl = document.querySelector('[data-shortlinks="count"]');
  const listStatusEl = document.querySelector('[data-shortlinks="list-status"]');

  const tokenInput = authForm.querySelector('[data-shortlinks="token"]');
  const refreshButton = authForm.querySelector('[data-shortlinks="refresh"]');
  const healthButton = authForm.querySelector('[data-shortlinks="health"]');
  const forgetButton = authForm.querySelector('[data-shortlinks="forget-token"]');
  const statusEl = authForm.querySelector('[data-shortlinks="status"]');
  const healthStatusEl = authForm.querySelector('[data-shortlinks="health-status"]');

  const slugInput = editorForm.querySelector('[data-shortlinks="slug"]');
  const destinationInput = editorForm.querySelector('[data-shortlinks="destination"]');
  const permanentInput = editorForm.querySelector('[data-shortlinks="permanent"]');
  const clearButton = editorForm.querySelector('[data-shortlinks="clear"]');
  const editorStatusEl = editorForm.querySelector('[data-shortlinks="editor-status"]');

  const destinationPickerOpen = editorForm.querySelector('[data-shortlinks="destination-picker-open"]');
  const destinationModal = document.querySelector('[data-shortlinks="destination-modal"]');
  const destinationModalClose = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-modal-close"]')
    : null;
  const destinationSearch = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-search"]')
    : null;
  const destinationResults = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-results"]')
    : null;

  const clicksModal = document.querySelector('[data-shortlinks="clicks-modal"]');
  const clicksModalClose = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-modal-close"]')
    : null;
  const clicksSlugEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-slug"]')
    : null;
  const clicksStatusEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-status"]')
    : null;
  const clicksListEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-list"]')
    : null;
  const clicksMetaEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-meta"]')
    : null;
  const clicksRefreshButton = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-refresh"]')
    : null;

  const DESTINATIONS_MANIFEST_PATH = 'dist/shortlinks-destinations.json';
  const CLICK_HISTORY_LIMIT = 250;

  const FALLBACK_DESTINATIONS = [
    { path: '/', label: 'Home', group: 'Pages' },
    { path: '/portfolio', label: 'Portfolio', group: 'Portfolio' },
    { path: '/resume', label: 'Resume', group: 'Pages' },
    { path: '/contact', label: 'Contact', group: 'Pages' },
    { path: '/tools', label: 'Tools', group: 'Tools' }
  ];

  let destinationsManifest = null;
  let destinationModalPrevFocus = null;
  let clicksModalPrevFocus = null;
  let activeClicksSlug = '';

  let basePath = DEFAULT_BASE_PATH;
  let allLinks = [];

  function setStatus(el, msg, tone){
    if (!el) return;
    el.textContent = msg || '';
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }

  function getStorage(preferLocal){
    const candidate = preferLocal ? window.localStorage : window.sessionStorage;
    try {
      if (!candidate) return null;
      const key = '__shortlinks_test__';
      candidate.setItem(key, '1');
      candidate.removeItem(key);
      return candidate;
    } catch {
      return null;
    }
  }

  const storage = getStorage(true) || getStorage(false);
  let memoryToken = '';

  function getSavedToken(){
    if (storage) return storage.getItem(STORAGE_KEY) || '';
    return memoryToken;
  }

  function saveToken(token){
    const value = String(token || '').trim();
    if (storage) {
      if (!value) storage.removeItem(STORAGE_KEY);
      else storage.setItem(STORAGE_KEY, value);
      return;
    }
    memoryToken = value;
  }

  function updateAccessMeta(){
    if (!accessMetaEl) return;
    accessMetaEl.textContent = getSavedToken() ? 'Token stored' : 'Token required';
  }

  function setCount(shown, total){
    if (!countEl) return;
    if (!total) {
      countEl.textContent = '';
      return;
    }
    const label = total === 1 ? 'link' : 'links';
    if (shown === total) {
      countEl.textContent = `${total} ${label}`;
      return;
    }
    countEl.textContent = `Showing ${shown} of ${total} ${label}`;
  }

  function getFilterQuery(){
    if (!filterInput) return '';
    return String(filterInput.value || '').trim().toLowerCase();
  }

  function getFilteredLinks(){
    const query = getFilterQuery();
    if (!query) return allLinks.slice();
    return allLinks.filter(link => {
      const slug = String(link.slug || '').toLowerCase();
      const destination = String(link.destination || '').toLowerCase();
      return slug.includes(query) || destination.includes(query);
    });
  }

  function isDevHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'localhost' || host === '127.0.0.1';
  }

  function isProdHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'danielshort.me' || host === 'www.danielshort.me';
  }

  function getCanonicalSiteOrigin(){
    const origin = destinationsManifest && destinationsManifest.origin ? String(destinationsManifest.origin) : '';
    if (isDevHost(window.location.hostname) || window.location.hostname.endsWith('.vercel.app')) {
      return window.location.origin;
    }
    if (isProdHost(window.location.hostname)) {
      return origin || 'https://danielshort.me';
    }
    return origin || window.location.origin;
  }

  function joinOriginAndPath(origin, pathname){
    const base = String(origin || '').replace(/\/+$/g, '');
    const path = String(pathname || '');
    if (!base) return path;
    if (!path) return base;
    if (path.startsWith('/')) return `${base}${path}`;
    return `${base}/${path}`;
  }

  function normalizeDestinationForSave(value){
    const raw = String(value || '').trim();
    if (!raw) return '';
    if (raw.startsWith('/')) return joinOriginAndPath(getCanonicalSiteOrigin(), raw);
    return raw;
  }

  function formatAbsoluteUrl(input){
    const raw = typeof input === 'string' ? input.trim() : '';
    if (!raw) return '';
    try {
      return new URL(raw, window.location.origin).toString();
    } catch {
      return raw;
    }
  }

  function syncModalOpenState(){
    if (!document || !document.body) return;
    if (!document.querySelector('.modal.active')) {
      document.body.classList.remove('modal-open');
    }
  }

  function closeDestinationPicker(){
    if (!destinationModal) return;
    destinationModal.classList.remove('active');
    destinationModal.setAttribute('aria-hidden', 'true');
    syncModalOpenState();
    if (destinationSearch) destinationSearch.value = '';
    if (destinationResults) destinationResults.replaceChildren();
    if (destinationModalPrevFocus && document.contains(destinationModalPrevFocus)) {
      destinationModalPrevFocus.focus();
    }
    destinationModalPrevFocus = null;
  }

  async function loadDestinationsManifest(){
    if (destinationsManifest) return destinationsManifest;
    const fallback = { origin: 'https://danielshort.me', pages: FALLBACK_DESTINATIONS };
    try {
      const resp = await fetch(DESTINATIONS_MANIFEST_PATH, { method: 'GET', cache: 'no-store' });
      if (!resp.ok) throw new Error(`Manifest request failed (${resp.status})`);
      const data = await resp.json().catch(() => null);
      if (!data || typeof data !== 'object' || !Array.isArray(data.pages)) throw new Error('Invalid manifest');
      destinationsManifest = data;
      return destinationsManifest;
    } catch {
      destinationsManifest = fallback;
      return destinationsManifest;
    }
  }

  function getDestinationQuery(){
    if (!destinationSearch) return '';
    return String(destinationSearch.value || '').trim().toLowerCase();
  }

  function getFilteredDestinations(){
    const manifest = destinationsManifest;
    if (!manifest || !Array.isArray(manifest.pages)) return [];
    const query = getDestinationQuery();
    const pages = manifest.pages.filter(item => item && typeof item.path === 'string' && typeof item.label === 'string');
    if (!query) return pages;
    return pages.filter(item => {
      const hay = `${item.label} ${item.path}`.toLowerCase();
      return hay.includes(query);
    });
  }

  function buildSuggestedSlugFromPath(pathname){
    const clean = String(pathname || '').replace(/^\/+|\/+$/g, '');
    if (!clean) return '';
    const last = clean.split('/').filter(Boolean).slice(-1)[0] || '';
    return last.toLowerCase();
  }

  function renderDestinations(){
    if (!destinationResults) return;
    destinationResults.replaceChildren();

    const pages = getFilteredDestinations();
    const query = getDestinationQuery();
    if (!pages.length) {
      const empty = document.createElement('p');
      empty.className = 'shortlinks-picker-empty';
      empty.textContent = query ? `No matches for "${query}".` : 'No destinations found.';
      destinationResults.appendChild(empty);
      return;
    }

    const groupOrder = ['Pages', 'Tools', 'Portfolio', 'Demos'];
    const grouped = new Map();
    pages.forEach(item => {
      const group = item.group && groupOrder.includes(item.group) ? item.group : 'Pages';
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(item);
    });

    groupOrder.forEach(group => {
      const items = grouped.get(group);
      if (!items || !items.length) return;

      const section = document.createElement('section');
      section.className = 'shortlinks-picker-group';

      const title = document.createElement('h4');
      title.className = 'shortlinks-picker-group-title';
      title.textContent = group;
      section.appendChild(title);

      const list = document.createElement('div');
      list.className = 'shortlinks-picker-group-list';

      items.forEach(item => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'shortlinks-picker-item';

        const label = document.createElement('span');
        label.className = 'shortlinks-picker-item-label';
        label.textContent = item.label || item.path;

        const pathCode = document.createElement('code');
        pathCode.className = 'shortlinks-picker-item-path';
        pathCode.textContent = item.path;

        button.appendChild(label);
        button.appendChild(pathCode);

        button.addEventListener('click', () => {
          const origin = getCanonicalSiteOrigin();
          const absolute = joinOriginAndPath(origin, item.path);
          destinationInput.value = absolute;

          if (!String(slugInput.value || '').trim()) {
            slugInput.value = buildSuggestedSlugFromPath(item.path);
          }

          setStatus(editorStatusEl, `Selected ${item.path}`, 'success');
          closeDestinationPicker();
          destinationInput.focus();
        });

        list.appendChild(button);
      });

      section.appendChild(list);
      destinationResults.appendChild(section);
    });
  }

  async function openDestinationPicker(){
    if (!destinationModal) return;
    destinationModalPrevFocus = document.activeElement;
    destinationModal.classList.add('active');
    destinationModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');

    await loadDestinationsManifest();
    renderDestinations();

    if (destinationSearch) {
      destinationSearch.focus({ preventScroll: true });
    }
  }

  function closeClicksModal(){
    if (!clicksModal) return;
    clicksModal.classList.remove('active');
    clicksModal.setAttribute('aria-hidden', 'true');
    syncModalOpenState();
    if (clicksSlugEl) clicksSlugEl.textContent = '';
    if (clicksListEl) clicksListEl.replaceChildren();
    if (clicksMetaEl) clicksMetaEl.textContent = '';
    setStatus(clicksStatusEl, '');
    activeClicksSlug = '';
    if (clicksModalPrevFocus && document.contains(clicksModalPrevFocus)) {
      clicksModalPrevFocus.focus();
    }
    clicksModalPrevFocus = null;
  }

  function formatTimestamp(value){
    const raw = typeof value === 'string' ? value : '';
    if (!raw) return '';
    const dt = new Date(raw);
    if (!Number.isFinite(dt.getTime())) return raw;
    return dt.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  function renderClickHistory(items){
    if (!clicksListEl) return;
    clicksListEl.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'shortlinks-clicks-empty';
      empty.textContent = 'No click events found (history starts after click logging was enabled).';
      clicksListEl.appendChild(empty);
      return;
    }

    items.forEach(item => {
      const card = document.createElement('article');
      card.className = 'shortlinks-click';

      const top = document.createElement('div');
      top.className = 'shortlinks-click-top';

      const when = document.createElement('time');
      const clickedAt = typeof item.clickedAt === 'string' ? item.clickedAt : '';
      when.className = 'shortlinks-click-when';
      if (clickedAt) when.dateTime = clickedAt;
      when.textContent = formatTimestamp(clickedAt) || clickedAt || 'Unknown time';

      const pills = document.createElement('div');
      pills.className = 'shortlinks-click-pills';

      const statusCode = Number.isFinite(Number(item.statusCode)) ? Number(item.statusCode) : 0;
      if (statusCode) {
        const statusPill = document.createElement('span');
        statusPill.className = 'tool-pill';
        statusPill.textContent = String(statusCode);
        pills.appendChild(statusPill);
      }

      const host = typeof item.host === 'string' ? item.host : '';
      if (host) {
        const hostPill = document.createElement('span');
        hostPill.className = 'tool-pill shortlinks-click-pill-muted';
        hostPill.textContent = host;
        pills.appendChild(hostPill);
      }

      const geoParts = [item.city, item.region, item.country]
        .filter(part => typeof part === 'string' && part.trim())
        .map(part => String(part).trim());
      if (geoParts.length) {
        const geoPill = document.createElement('span');
        geoPill.className = 'tool-pill shortlinks-click-pill-muted';
        geoPill.textContent = geoParts.join(', ');
        pills.appendChild(geoPill);
      }

      top.appendChild(when);
      top.appendChild(pills);

      const rows = document.createElement('div');
      rows.className = 'shortlinks-click-rows';

      function addRow(label, value, url){
        if (!value) return;
        const row = document.createElement('div');
        row.className = 'shortlinks-click-row';

        const labelEl = document.createElement('span');
        labelEl.className = 'shortlinks-click-label';
        labelEl.textContent = label;

        let valueEl;
        if (url) {
          const anchor = document.createElement('a');
          anchor.className = 'shortlinks-click-value';
          anchor.href = url;
          anchor.target = '_blank';
          anchor.rel = 'noopener noreferrer';
          anchor.textContent = value;
          valueEl = anchor;
        } else {
          const text = document.createElement('span');
          text.className = 'shortlinks-click-value';
          text.textContent = value;
          valueEl = text;
        }

        row.appendChild(labelEl);
        row.appendChild(valueEl);
        rows.appendChild(row);
      }

      const destination = typeof item.destination === 'string' ? item.destination : '';
      addRow('To', destination, destination);

      const referer = typeof item.referer === 'string' ? item.referer : '';
      addRow('From', referer, referer);

      const userAgent = typeof item.userAgent === 'string' ? item.userAgent : '';
      addRow('UA', userAgent);

      card.appendChild(top);
      card.appendChild(rows);

      clicksListEl.appendChild(card);
    });
  }

  async function refreshClickHistory(slug){
    if (!slug) return;
    if (!clicksModal) return;
    setStatus(clicksStatusEl, 'Loading click history…');
    if (clicksMetaEl) clicksMetaEl.textContent = '';

    try {
      const data = await api(`/api/short-links/clicks/${encodeURIComponent(slug)}?limit=${CLICK_HISTORY_LIMIT}`, { method: 'GET' });
      const events = Array.isArray(data.clicks) ? data.clicks : [];
      renderClickHistory(events);
      const countLabel = events.length === 1 ? 'event' : 'events';
      if (clicksMetaEl) clicksMetaEl.textContent = `Showing ${events.length} ${countLabel}.`;
      setStatus(clicksStatusEl, events.length ? '' : 'No events yet.', events.length ? 'success' : 'success');
    } catch (err) {
      if (clicksListEl) clicksListEl.replaceChildren();
      setStatus(clicksStatusEl, err.message, 'error');
    }
  }

  async function openClicksModal(slug){
    if (!clicksModal) return;
    const cleanSlug = normalizeSlugInput(slug);
    if (!cleanSlug) return;
    clicksModalPrevFocus = document.activeElement;
    activeClicksSlug = cleanSlug;
    if (clicksSlugEl) clicksSlugEl.textContent = buildPublicPath(cleanSlug) || cleanSlug;

    clicksModal.classList.add('active');
    clicksModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');

    await refreshClickHistory(cleanSlug);
  }

  function flashButtonText(button, text){
    if (!button) return;
    const original = button.textContent;
    button.textContent = text;
    window.setTimeout(() => {
      if (button.textContent === text) button.textContent = original;
    }, 1200);
  }

  async function testRedirect(slug, button){
    const clean = normalizeSlugInput(slug);
    if (!clean) return;

    const target = `/api/go/${encodeURIComponent(clean)}`;
    const label = buildPublicPath(clean) || clean;
    const start = Date.now();

    const originalText = button && typeof button.textContent === 'string' ? button.textContent : '';
    const flashText = (text) => {
      if (!button) return;
      button.textContent = text;
      window.setTimeout(() => {
        if (!button) return;
        if (button.textContent === text) button.textContent = originalText;
      }, 1200);
    };

    if (button) {
      button.disabled = true;
      button.textContent = 'Testing…';
    }
    setStatus(listStatusEl, `Testing ${label}…`);

    try {
      const resp = await fetch(target, { method: 'HEAD', redirect: 'manual', cache: 'no-store' });
      const ms = Date.now() - start;
      const location = resp.headers.get('location') || '';
      const code = resp.status;

      if (code >= 300 && code < 400 && location) {
        const displayLoc = location.length > 200 ? `${location.slice(0, 197)}…` : location;
        setStatus(listStatusEl, `OK: ${code} → ${displayLoc} (${ms}ms)`, 'success');
        flashText('OK');
        return;
      }

      const detail = code ? `(${code})` : '';
      setStatus(listStatusEl, `Test failed ${detail}.`, 'error');
      flashText('Failed');
    } catch (err) {
      setStatus(listStatusEl, err && err.message ? err.message : 'Test failed.', 'error');
      flashText('Failed');
    } finally {
      if (button) {
        button.disabled = false;
        if (button.textContent === 'Testing…' && originalText) button.textContent = originalText;
      }
    }
  }

  async function api(path, options = {}){
    const token = getSavedToken();
    const headers = Object.assign({}, options.headers || {});
    if (token) headers.Authorization = `Bearer ${token}`;

    const hasBody = typeof options.body !== 'undefined';
    if (hasBody && !headers['Content-Type']) headers['Content-Type'] = 'application/json';

    const resp = await fetch(path, Object.assign({}, options, { headers }));
    const isJson = (resp.headers.get('content-type') || '').includes('application/json');
    const data = isJson ? await resp.json().catch(() => null) : null;
    if (!resp.ok || !data || data.ok !== true) {
      const errMsg = (data && data.error) ? data.error : `Request failed (${resp.status})`;
      const err = new Error(errMsg);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  async function apiInspect(path){
    const token = getSavedToken();
    const headers = {};
    if (token) headers.Authorization = `Bearer ${token}`;

    const resp = await fetch(path, { method: 'GET', headers });
    const isJson = (resp.headers.get('content-type') || '').includes('application/json');
    const data = isJson ? await resp.json().catch(() => null) : null;
    return { status: resp.status, data };
  }

  function formatHealthPayload(payload){
    if (!payload || typeof payload !== 'object') return '';
    const debugBits = [];
    const aws = payload.aws || {};
    if (aws.accessKeyIdSource) debugBits.push(`Creds: ${aws.accessKeyIdSource}.`);
    if (aws.secretFingerprint) debugBits.push(`Secret fp: ${aws.secretFingerprint}.`);
    if (aws.sessionTokenConfigured) {
      debugBits.push(`Session token: ${aws.sessionTokenUsed ? 'used' : 'ignored'}.`);
    }
    const debug = debugBits.length ? ` ${debugBits.join(' ')}` : '';

	    if (payload.ok === true) {
	      const keyId = payload.aws && payload.aws.accessKeyId ? payload.aws.accessKeyId : '';
	      const table = payload.table && payload.table.name ? payload.table.name : '';
	      const region = payload.aws && payload.aws.region ? payload.aws.region : '';
	      const status = payload.table && payload.table.status ? payload.table.status : '';
	      const billing = payload.table && payload.table.billingMode ? payload.table.billingMode : '';
	      const clicks = payload.clicks || {};
	      const bits = [
	        `Backend OK${table ? `: ${table}` : ''}${status ? ` (${status})` : ''}.`,
	        region ? `Region: ${region}.` : '',
	        billing ? `Billing: ${billing}.` : '',
	        keyId ? `Access key: ${keyId}.` : ''
	      ].filter(Boolean);
	      if (clicks && clicks.configured) {
	        const clicksName = clicks.table && clicks.table.name ? clicks.table.name : '';
	        const clicksStatus = clicks.table && clicks.table.status ? clicks.table.status : '';
	        bits.push(`Click log${clicksName ? `: ${clicksName}` : ''}${clicksStatus ? ` (${clicksStatus})` : ''}.`);
	        if (clicks.error && (clicks.error.name || clicks.error.message)) {
	          const errName = clicks.error.name ? String(clicks.error.name) : '';
	          const errMsg = clicks.error.message ? String(clicks.error.message) : '';
	          const compact = [errName, errMsg].filter(Boolean).join(': ');
	          const clipped = compact.length > 140 ? `${compact.slice(0, 137)}…` : compact;
	          if (clipped) bits.push(`Click log error: ${clipped}.`);
	        }
	      } else {
	        bits.push('Click log: not configured.');
	      }
	      return bits.join(' ') + debug;
	    }

    const details = payload.details || {};
    const name = details.name ? String(details.name) : '';
    const message = details.message ? String(details.message) : '';
    const base = payload.error ? String(payload.error) : 'Backend check failed';
    const extra = [name, message].filter(Boolean).join(': ');
    return (extra ? `${base} (${extra})` : base) + debug;
  }

  function healthHints(payload){
    if (!payload || typeof payload !== 'object') return '';
    const details = payload.details || {};
    const message = details.message ? String(details.message) : '';
    const name = details.name ? String(details.name) : '';

    if (payload.aws && payload.aws.secretTrimmed) {
      return 'Your AWS secret appears to have leading/trailing whitespace. Re-save it in Vercel (or redeploy after trimming).';
    }
    if (payload.aws && payload.aws.sessionTokenIgnored) {
      return 'AWS_SESSION_TOKEN is set, but your access key looks like a long-term key (AKIA). Remove AWS_SESSION_TOKEN in Vercel, or set SHORTLINKS_AWS_ACCESS_KEY_ID/SHORTLINKS_AWS_SECRET_ACCESS_KEY (preferred) and leave the session token unset.';
    }
    if (name === 'UnrecognizedClientException' || /security token.*invalid/i.test(message)) {
      return 'AWS rejected the key pair. Re-copy the access key + secret from the same CSV row (no quotes/whitespace) and redeploy. If you have duplicate AWS_* vars in Vercel, set SHORTLINKS_AWS_ACCESS_KEY_ID/SHORTLINKS_AWS_SECRET_ACCESS_KEY instead.';
    }
    if (name === 'AccessDeniedException') {
      return 'AWS credentials are valid but lack DynamoDB permissions for this table.';
    }
    if (name === 'ResourceNotFoundException') {
      return 'Table not found. Double-check AWS_REGION and SHORTLINKS_DDB_TABLE.';
    }
    return '';
  }

  function clearList(){
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
  }

  function normalizeOrigin(origin){
    const raw = typeof origin === 'string' ? origin.trim() : '';
    if (!raw) return '';
    try {
      const url = new URL(raw);
      return `${url.protocol}//${url.host}`;
    } catch {
      return raw.replace(/\/+$/g, '');
    }
  }

  function isShortDomainHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'dshort.me' || host === 'www.dshort.me';
  }

  function getPublicOrigin(){
    if (isDevHost(window.location.hostname) || isShortDomainHost(window.location.hostname)) {
      return normalizeOrigin(window.location.origin) || window.location.origin;
    }
    return normalizeOrigin(DEFAULT_PUBLIC_ORIGIN) || window.location.origin;
  }

  function getPublicBasePath(){
    if (isShortDomainHost(window.location.hostname)) return '';
    return basePath;
  }

  function buildPublicPath(slug){
    const clean = String(slug || '').replace(/^\/+|\/+$/g, '');
    if (!clean) return '';
    const prefix = String(getPublicBasePath() || '').replace(/^\/+|\/+$/g, '');
    return prefix ? `/${prefix}/${clean}` : `/${clean}`;
  }

  function buildShortUrl(slug){
    const origin = getPublicOrigin();
    const path = buildPublicPath(slug);
    return path ? `${origin}${path}` : origin;
  }

  function normalizeSlugInput(value){
    return String(value || '').trim().replace(/^\/+|\/+$/g, '').toLowerCase();
  }

  function renderLinks(links){
    clearList();
    if (!Array.isArray(links) || links.length === 0) {
      const empty = document.createElement('p');
      const query = getFilterQuery();
      empty.className = 'shortlinks-empty';
      empty.textContent = query ? `No matches for "${query}".` : 'No short links yet.';
      listEl.appendChild(empty);
      return;
    }

    links.forEach(link => {
      const shortUrl = buildShortUrl(link.slug);
      const destinationUrl = formatAbsoluteUrl(link.destination);

      const card = document.createElement('article');
      card.className = 'shortlinks-item';
      if (link.disabled) card.classList.add('shortlinks-item-disabled');

      const head = document.createElement('div');
      head.className = 'shortlinks-item-head';

      const titleWrap = document.createElement('div');
      titleWrap.className = 'shortlinks-item-title';

      const slugCode = document.createElement('code');
      slugCode.className = 'shortlinks-slug';
      slugCode.textContent = buildPublicPath(link.slug);

      const meta = document.createElement('div');
      meta.className = 'shortlinks-item-meta';

      const statusPill = document.createElement('span');
      statusPill.className = 'tool-pill';
      statusPill.textContent = link.permanent ? '301' : '302';

      const clicksPill = document.createElement('button');
      clicksPill.type = 'button';
      clicksPill.className = 'tool-pill shortlinks-pill-button';
      clicksPill.textContent = `${Number(link.clicks) || 0} clicks`;
      clicksPill.addEventListener('click', () => {
        openClicksModal(link.slug);
      });

      meta.appendChild(statusPill);
      if (link.disabled) {
        const disabledPill = document.createElement('span');
        disabledPill.className = 'tool-pill shortlinks-pill-disabled';
        disabledPill.textContent = 'Disabled';
        meta.appendChild(disabledPill);
      }
      meta.appendChild(clicksPill);
      titleWrap.appendChild(slugCode);
      titleWrap.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'shortlinks-actions';

      const copyButton = document.createElement('button');
      copyButton.type = 'button';
      copyButton.className = 'btn-ghost';
      copyButton.textContent = 'Copy short URL';
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(shortUrl);
          flashButtonText(copyButton, 'Copied');
          setStatus(listStatusEl, `Copied: ${shortUrl}`, 'success');
        } catch {
          flashButtonText(copyButton, 'Copy failed');
          setStatus(listStatusEl, 'Copy failed (clipboard permission blocked).', 'error');
        }
      });

      const openShort = document.createElement('a');
      openShort.className = 'btn-secondary';
      openShort.href = shortUrl;
      openShort.target = '_blank';
      openShort.rel = 'noopener noreferrer';
      openShort.textContent = 'Open';

      const testButton = document.createElement('button');
      testButton.type = 'button';
      testButton.className = 'btn-secondary';
      testButton.textContent = 'Test';
      testButton.addEventListener('click', () => {
        testRedirect(link.slug, testButton);
      });

      const editButton = document.createElement('button');
      editButton.type = 'button';
      editButton.className = 'btn-secondary';
      editButton.textContent = 'Edit';
      editButton.addEventListener('click', () => {
        slugInput.value = link.slug;
        destinationInput.value = link.destination;
        permanentInput.checked = !!link.permanent;
        slugInput.focus();
        setStatus(editorStatusEl, `Editing ${link.slug}${link.disabled ? ' (disabled)' : ''}`, 'success');
      });

      const toggleButton = document.createElement('button');
      toggleButton.type = 'button';
      toggleButton.className = 'btn-secondary';
      toggleButton.textContent = link.disabled ? 'Enable' : 'Disable';
      toggleButton.addEventListener('click', async () => {
        const nextDisabled = !link.disabled;
        if (nextDisabled) {
          const ok = window.confirm(`Disable ${buildPublicPath(link.slug)}?`);
          if (!ok) return;
        }
        try {
          await api(`/api/short-links/${encodeURIComponent(link.slug)}`, {
            method: 'PATCH',
            body: JSON.stringify({ disabled: nextDisabled })
          });
          setStatus(listStatusEl, `${nextDisabled ? 'Disabled' : 'Enabled'} ${link.slug}`, 'success');
          await refreshLinks();
        } catch (err) {
          setStatus(listStatusEl, err.message, 'error');
        }
      });

      const deleteButton = document.createElement('button');
      deleteButton.type = 'button';
      deleteButton.className = 'btn-secondary shortlinks-danger';
      deleteButton.textContent = 'Delete';
      deleteButton.addEventListener('click', async () => {
        const ok = window.confirm(`Delete ${buildPublicPath(link.slug)}?`);
        if (!ok) return;
        try {
          await api(`/api/short-links/${encodeURIComponent(link.slug)}`, { method: 'DELETE' });
          setStatus(listStatusEl, `Deleted ${link.slug}`, 'success');
          await refreshLinks();
        } catch (err) {
          setStatus(listStatusEl, err.message, 'error');
        }
      });

      actions.appendChild(copyButton);
      actions.appendChild(openShort);
      actions.appendChild(testButton);
      actions.appendChild(editButton);
      actions.appendChild(toggleButton);
      actions.appendChild(deleteButton);

      head.appendChild(titleWrap);
      head.appendChild(actions);

      const linksWrap = document.createElement('div');
      linksWrap.className = 'shortlinks-item-links';

      const shortRow = document.createElement('div');
      shortRow.className = 'shortlinks-link-row';
      const shortLabel = document.createElement('span');
      shortLabel.className = 'shortlinks-link-label';
      shortLabel.textContent = 'Short';
      const shortAnchor = document.createElement('a');
      shortAnchor.className = 'shortlinks-link-value';
      shortAnchor.href = shortUrl;
      shortAnchor.target = '_blank';
      shortAnchor.rel = 'noopener noreferrer';
      shortAnchor.textContent = shortUrl;
      shortRow.appendChild(shortLabel);
      shortRow.appendChild(shortAnchor);

      const destRow = document.createElement('div');
      destRow.className = 'shortlinks-link-row';
      const destLabel = document.createElement('span');
      destLabel.className = 'shortlinks-link-label';
      destLabel.textContent = 'To';
      const destAnchor = document.createElement('a');
      destAnchor.className = 'shortlinks-link-value';
      destAnchor.href = destinationUrl;
      destAnchor.target = '_blank';
      destAnchor.rel = 'noopener noreferrer';
      destAnchor.textContent = destinationUrl;
      destRow.appendChild(destLabel);
      destRow.appendChild(destAnchor);

      linksWrap.appendChild(shortRow);
      linksWrap.appendChild(destRow);

      card.appendChild(head);
      card.appendChild(linksWrap);

      listEl.appendChild(card);
    });
  }

  function applyFilterAndRender(){
    const filtered = getFilteredLinks();
    renderLinks(filtered);
    setCount(filtered.length, allLinks.length);
  }

  async function refreshLinks(){
    setStatus(listStatusEl, 'Loading…');
    setStatus(healthStatusEl, '');
    try {
      const data = await api('/api/short-links', { method: 'GET' });
      basePath = typeof data.basePath === 'string' && data.basePath.trim() ? data.basePath.trim() : DEFAULT_BASE_PATH;
      allLinks = Array.isArray(data.links) ? data.links : [];
      applyFilterAndRender();
      setStatus(listStatusEl, `Loaded ${allLinks.length} link(s).`, 'success');
    } catch (err) {
      allLinks = [];
      clearList();
      setCount(0, 0);
      setStatus(listStatusEl, err.message, 'error');
    }
  }

  async function refreshHealth(){
    setStatus(healthStatusEl, 'Checking backend…');
    try {
      const result = await apiInspect('/api/short-links/health');
      const payload = result.data || {};
      const msg = formatHealthPayload(payload) || `Backend check failed (${result.status})`;
      if (payload.ok === true) {
        setStatus(healthStatusEl, msg, 'success');
        return;
      }
      const hint = healthHints(payload);
      setStatus(healthStatusEl, hint ? `${msg} ${hint}` : msg, 'error');
    } catch (err) {
      setStatus(healthStatusEl, err.message || 'Backend check failed.', 'error');
    }
  }

  function clearEditor(){
    slugInput.value = '';
    destinationInput.value = '';
    permanentInput.checked = false;
    setStatus(editorStatusEl, '');
  }

  authForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const token = tokenInput.value.trim();
    if (!token) {
      if (getSavedToken()) {
        setStatus(statusEl, 'Token already stored. Paste a token to replace, or click "Forget token".', 'success');
      } else {
        setStatus(statusEl, 'Paste your admin token to unlock this dashboard.', 'error');
      }
      return;
    }
    saveToken(token);
    updateAccessMeta();
    tokenInput.value = '';
    setStatus(statusEl, 'Token saved. Loading links…');
    if (accessDetails) accessDetails.open = false;
    await refreshLinks();
  });

  refreshButton.addEventListener('click', () => {
    if (!getSavedToken()) {
      setStatus(statusEl, 'Admin token required.', 'error');
      setStatus(listStatusEl, 'Admin token required.', 'error');
      return;
    }
    refreshLinks();
  });

  if (healthButton) {
    healthButton.addEventListener('click', () => {
      if (!getSavedToken()) {
        setStatus(healthStatusEl, 'Admin token required.', 'error');
        return;
      }
      refreshHealth();
    });
  }

  if (forgetButton) {
    forgetButton.addEventListener('click', () => {
      saveToken('');
      updateAccessMeta();
      tokenInput.value = '';
      clearList();
      setStatus(statusEl, 'Token forgotten on this device.', 'success');
      setStatus(healthStatusEl, '');
      setStatus(listStatusEl, '');
      setCount(0, 0);
      if (accessDetails) accessDetails.open = true;
    });
  }

  clearButton.addEventListener('click', () => {
    clearEditor();
  });

  if (destinationPickerOpen && destinationModal) {
    destinationPickerOpen.addEventListener('click', () => {
      openDestinationPicker();
    });
  }

  if (destinationModalClose) {
    destinationModalClose.addEventListener('click', () => {
      closeDestinationPicker();
    });
  }

  if (destinationModal) {
    destinationModal.addEventListener('click', (event) => {
      if (event.target === destinationModal) closeDestinationPicker();
    });
  }

  if (destinationSearch) {
    destinationSearch.addEventListener('input', () => {
      renderDestinations();
    });
  }

  if (clicksModalClose) {
    clicksModalClose.addEventListener('click', () => {
      closeClicksModal();
    });
  }

  if (clicksRefreshButton) {
    clicksRefreshButton.addEventListener('click', () => {
      if (!activeClicksSlug) {
        setStatus(clicksStatusEl, 'No link selected.', 'error');
        return;
      }
      refreshClickHistory(activeClicksSlug);
    });
  }

  if (clicksModal) {
    clicksModal.addEventListener('click', (event) => {
      if (event.target === clicksModal) closeClicksModal();
    });
  }

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (destinationModal && destinationModal.classList.contains('active')) {
      closeDestinationPicker();
      return;
    }
    if (clicksModal && clicksModal.classList.contains('active')) {
      closeClicksModal();
    }
  });

  editorForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const slug = normalizeSlugInput(slugInput.value);
    const destination = normalizeDestinationForSave(destinationInput.value);
    const permanent = !!permanentInput.checked;

    if (!slug) {
      setStatus(editorStatusEl, 'Slug is required.', 'error');
      return;
    }
    if (!destination) {
      setStatus(editorStatusEl, 'Destination is required.', 'error');
      return;
    }

    setStatus(editorStatusEl, 'Saving…');
    try {
      await api('/api/short-links', {
        method: 'POST',
        body: JSON.stringify({ slug, destination, permanent })
      });
      setStatus(editorStatusEl, `Saved ${buildPublicPath(slug)}`, 'success');
      await refreshLinks();
    } catch (err) {
      setStatus(editorStatusEl, err.message, 'error');
    }
  });

  updateAccessMeta();
  if (getSavedToken()) {
    setStatus(statusEl, 'Token loaded from this browser. Loading links…', 'success');
    refreshLinks();
  }

  if (filterInput) {
    filterInput.addEventListener('input', () => {
      applyFilterAndRender();
    });
  }
})();
