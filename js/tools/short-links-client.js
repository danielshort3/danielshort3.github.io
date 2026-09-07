/* Shared access and saved-link URLs for Links & QR codes. */
(() => {
  'use strict';

  const TOKEN_KEY = 'shortlinks_admin_token';
  let memoryToken = '';
  const readToken = (name) => {
    try { return String(window[name].getItem(TOKEN_KEY) || '').trim(); } catch { return ''; }
  };
  const getToken = () => readToken('sessionStorage') || readToken('localStorage') || memoryToken;
  const setToken = (token, remember = false) => {
    memoryToken = String(token || '').trim();
    ['sessionStorage', 'localStorage'].forEach((name) => {
      try { window[name].removeItem(TOKEN_KEY); } catch {}
    });
    if (memoryToken) {
      try { window[remember ? 'localStorage' : 'sessionStorage'].setItem(TOKEN_KEY, memoryToken); } catch {}
    }
    window.dispatchEvent(new CustomEvent('shortlinks:access-changed'));
  };

  const request = async (path, options = {}) => {
    const target = new URL(path, window.location.origin);
    if (target.origin !== window.location.origin || !target.pathname.startsWith('/api/short-links')) {
      throw new Error('Link management requests must use this workspace.');
    }
    const headers = new Headers(options.headers || {});
    const token = getToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    else {
      if (!window.ToolsAuth && document.readyState === 'loading') {
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
      }
      const auth = await window.ToolsAuth?.ensureFreshAuth?.();
      if (auth?.idToken && !auth.sessionOnly) headers.set('Authorization', `Bearer ${auth.idToken}`);
    }
    let body = options.body;
    if (body && typeof body === 'object') body = JSON.stringify(body);
    if (body) headers.set('Content-Type', 'application/json');
    const response = await fetch(target.pathname + target.search, {
      ...options, body, headers, credentials: 'same-origin', cache: 'no-store'
    });
    const data = await response.json().catch(() => null);
    if (!response.ok || !data || data.ok === false) {
      const error = new Error(response.status === 401 || response.status === 403
        ? 'Sign in with an authorized account or connect workspace access to manage links.'
        : data?.error || `Link request failed (${response.status}).`);
      error.status = response.status;
      error.code = data?.code || '';
      error.data = data;
      throw error;
    }
    return data;
  };

  const publicUrl = (slug, { qr = false } = {}) => {
    const ending = String(slug || '').trim().replace(/^\/+|\/+$/g, '').split('/').map(encodeURIComponent).join('/');
    return `https://dshort.me/${ending}${qr ? '?__qr=1' : ''}`;
  };
  const qrEditorUrl = (slug) => `/tools/qr-code-generator?link=${encodeURIComponent(String(slug || ''))}`;
  window.ShortLinksClient = Object.freeze({
    getToken, setToken, clearToken: () => setToken(''), request, publicUrl, qrEditorUrl,
    list: () => request('/api/short-links'),
    get: (slug) => request(`/api/short-links/${encodeURIComponent(slug)}`),
    create: (payload) => request('/api/short-links', { method: 'POST', body: { ...payload, intent: 'create' } }),
    update: (slug, payload) => request(`/api/short-links/${encodeURIComponent(slug)}`, { method: 'PATCH', body: payload })
  });
})();
