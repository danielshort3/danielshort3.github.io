/* Short links admin dashboard (token-based). */
(() => {
  'use strict';

  const STORAGE_KEY = 'shortlinks_admin_token';
  const DEFAULT_BASE_PATH = 'go';

  const authForm = document.querySelector('[data-shortlinks="auth"]');
  const editorForm = document.querySelector('[data-shortlinks="editor"]');
  const listEl = document.querySelector('[data-shortlinks="list"]');
  if (!authForm || !editorForm || !listEl) return;

  const tokenInput = authForm.querySelector('[data-shortlinks="token"]');
  const refreshButton = authForm.querySelector('[data-shortlinks="refresh"]');
  const statusEl = authForm.querySelector('[data-shortlinks="status"]');

  const slugInput = editorForm.querySelector('[data-shortlinks="slug"]');
  const destinationInput = editorForm.querySelector('[data-shortlinks="destination"]');
  const permanentInput = editorForm.querySelector('[data-shortlinks="permanent"]');
  const clearButton = editorForm.querySelector('[data-shortlinks="clear"]');
  const editorStatusEl = editorForm.querySelector('[data-shortlinks="editor-status"]');

  let basePath = DEFAULT_BASE_PATH;

  function setStatus(el, msg, tone){
    if (!el) return;
    el.textContent = msg || '';
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }

  function getSavedToken(){
    try {
      return sessionStorage.getItem(STORAGE_KEY) || '';
    } catch {
      return '';
    }
  }

  function saveToken(token){
    try {
      if (!token) sessionStorage.removeItem(STORAGE_KEY);
      else sessionStorage.setItem(STORAGE_KEY, token);
    } catch {}
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

  function clearList(){
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
  }

  function buildShortUrl(slug){
    const clean = String(slug || '').replace(/^\/+|\/+$/g, '');
    return `${window.location.origin}/${basePath}/${clean}`;
  }

  function normalizeSlugInput(value){
    return String(value || '').trim().replace(/^\/+|\/+$/g, '').toLowerCase();
  }

  function renderLinks(links){
    clearList();
    if (!Array.isArray(links) || links.length === 0) {
      const empty = document.createElement('p');
      empty.textContent = 'No short links yet.';
      listEl.appendChild(empty);
      return;
    }

    links.forEach(link => {
      const card = document.createElement('article');
      card.className = 'tool-card';

      const head = document.createElement('div');
      head.className = 'tool-top';

      const main = document.createElement('div');

      const title = document.createElement('h3');
      title.textContent = `/${basePath}/${link.slug}`;

      const meta = document.createElement('div');
      meta.className = 'tool-meta';

      const statusPill = document.createElement('span');
      statusPill.className = 'tool-pill';
      statusPill.textContent = link.permanent ? '301' : '302';

      const clicksPill = document.createElement('span');
      clicksPill.className = 'tool-pill';
      clicksPill.textContent = `${Number(link.clicks) || 0} clicks`;

      meta.appendChild(statusPill);
      meta.appendChild(clicksPill);
      main.appendChild(title);
      main.appendChild(meta);
      head.appendChild(main);
      card.appendChild(head);

      const destLine = document.createElement('p');
      const destAnchor = document.createElement('a');
      destAnchor.href = link.destination;
      destAnchor.target = '_blank';
      destAnchor.rel = 'noopener noreferrer';
      destAnchor.textContent = link.destination;
      destLine.appendChild(destAnchor);
      card.appendChild(destLine);

      const actions = document.createElement('div');
      actions.className = 'contact-form-alt';

      const openShort = document.createElement('a');
      openShort.className = 'btn-ghost';
      openShort.href = `/${basePath}/${link.slug}`;
      openShort.target = '_blank';
      openShort.rel = 'noopener noreferrer';
      openShort.textContent = 'Open';

      const copyButton = document.createElement('button');
      copyButton.type = 'button';
      copyButton.className = 'btn-ghost';
      copyButton.textContent = 'Copy';
      copyButton.addEventListener('click', async () => {
        const shortUrl = buildShortUrl(link.slug);
        try {
          await navigator.clipboard.writeText(shortUrl);
          setStatus(statusEl, `Copied: ${shortUrl}`, 'success');
        } catch {
          setStatus(statusEl, 'Copy failed (clipboard permission blocked).', 'error');
        }
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
        setStatus(editorStatusEl, `Editing ${link.slug}`, 'success');
      });

      const deleteButton = document.createElement('button');
      deleteButton.type = 'button';
      deleteButton.className = 'btn-secondary';
      deleteButton.textContent = 'Delete';
      deleteButton.addEventListener('click', async () => {
        const ok = window.confirm(`Delete /${basePath}/${link.slug}?`);
        if (!ok) return;
        try {
          await api(`/api/short-links/${encodeURIComponent(link.slug)}`, { method: 'DELETE' });
          setStatus(statusEl, `Deleted ${link.slug}`, 'success');
          await refreshLinks();
        } catch (err) {
          setStatus(statusEl, err.message, 'error');
        }
      });

      actions.appendChild(openShort);
      actions.appendChild(copyButton);
      actions.appendChild(editButton);
      actions.appendChild(deleteButton);
      card.appendChild(actions);

      listEl.appendChild(card);
    });
  }

  async function refreshLinks(){
    setStatus(statusEl, 'Loading…');
    try {
      const data = await api('/api/short-links', { method: 'GET' });
      basePath = typeof data.basePath === 'string' && data.basePath.trim() ? data.basePath.trim() : DEFAULT_BASE_PATH;
      renderLinks(data.links || []);
      setStatus(statusEl, `Loaded ${Array.isArray(data.links) ? data.links.length : 0} link(s).`, 'success');
    } catch (err) {
      clearList();
      setStatus(statusEl, err.message, 'error');
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
      saveToken('');
      setStatus(statusEl, 'Token cleared.', 'success');
      clearList();
      return;
    }
    saveToken(token);
    setStatus(statusEl, 'Token saved. Loading links…');
    await refreshLinks();
  });

  refreshButton.addEventListener('click', () => {
    refreshLinks();
  });

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
      setStatus(editorStatusEl, `Saved /${basePath}/${slug}`, 'success');
      await refreshLinks();
    } catch (err) {
      setStatus(editorStatusEl, err.message, 'error');
    }
  });

  if (getSavedToken()) {
    setStatus(statusEl, 'Token found in sessionStorage. Click "Refresh list" to load.', 'success');
  }
})();

