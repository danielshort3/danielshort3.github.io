/* Short links admin dashboard (token-based). */
(() => {
  'use strict';

  const STORAGE_KEY = 'shortlinks_admin_token';
  const DEFAULT_BASE_PATH = 'go';
  const DEFAULT_PUBLIC_ORIGIN = 'https://dshort.me';

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

  function formatAbsoluteUrl(input){
    const raw = typeof input === 'string' ? input.trim() : '';
    if (!raw) return '';
    try {
      return new URL(raw, window.location.origin).toString();
    } catch {
      return raw;
    }
  }

  function flashButtonText(button, text){
    if (!button) return;
    const original = button.textContent;
    button.textContent = text;
    window.setTimeout(() => {
      if (button.textContent === text) button.textContent = original;
    }, 1200);
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
      const bits = [
        `Backend OK${table ? `: ${table}` : ''}${status ? ` (${status})` : ''}.`,
        region ? `Region: ${region}.` : '',
        billing ? `Billing: ${billing}.` : '',
        keyId ? `Access key: ${keyId}.` : ''
      ].filter(Boolean);
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

  function isDevHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'localhost' || host === '127.0.0.1';
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
    if (isDevHost(window.location.hostname)) return basePath;
    return '';
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

      const clicksPill = document.createElement('span');
      clicksPill.className = 'tool-pill';
      clicksPill.textContent = `${Number(link.clicks) || 0} clicks`;

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

  editorForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const slug = normalizeSlugInput(slugInput.value);
    const destination = String(destinationInput.value || '').trim();
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
