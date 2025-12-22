(function (root) {
  'use strict';

  const DEFAULT_QUERY_KEYS = ['endpoint', 'fn', 'api'];

  const safeGet = (key) => {
    if (!key) return null;
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const safeSet = (key, value) => {
    if (!key) return;
    try {
      localStorage.setItem(key, value);
    } catch {
      // Storage may be blocked; ignore.
    }
  };

  const normalizeBase = (url) => {
    if (!url) return '';
    const raw = String(url).trim();
    if (!raw) return '';
    return raw.endsWith('/') ? raw : `${raw}/`;
  };

  const joinUrl = (base, path = '') => {
    const left = String(base || '').trim();
    const right = String(path || '').trim();
    if (!left) return right;
    if (!right) return left;
    if (left.endsWith('/') && right.startsWith('/')) return left + right.slice(1);
    if (!left.endsWith('/') && !right.startsWith('/')) return `${left}/${right}`;
    return left + right;
  };

  const unique = (list) => Array.from(new Set(list.filter(Boolean)));

  const readQuery = (keys = DEFAULT_QUERY_KEYS) => {
    try {
      const params = new URLSearchParams(window.location.search);
      for (const key of keys) {
        const val = params.get(key);
        if (val) return val;
      }
    } catch {
      // Ignore query errors.
    }
    return '';
  };

  const listCandidates = ({
    defaultUrl = '',
    storageKey = '',
    legacyKeys = [],
    queryKeys = DEFAULT_QUERY_KEYS
  } = {}) => {
    const out = [];
    const fromQuery = readQuery(queryKeys);
    if (fromQuery) {
      out.push(normalizeBase(fromQuery));
      if (storageKey) safeSet(storageKey, fromQuery);
    }

    const storageKeys = Array.isArray(legacyKeys)
      ? legacyKeys.slice()
      : (legacyKeys ? [legacyKeys] : []);
    if (storageKey) storageKeys.unshift(storageKey);
    for (const key of storageKeys) {
      const stored = safeGet(key);
      if (stored && stored !== fromQuery) out.push(normalizeBase(stored));
    }

    if (defaultUrl) out.push(normalizeBase(defaultUrl));
    return unique(out);
  };

  const resolveEndpoint = (options) => {
    const list = listCandidates(options);
    return list[0] || '';
  };

  const rememberEndpoint = (base, storageKey) => {
    if (storageKey && base) safeSet(storageKey, base);
  };

  const requestJson = async (url, options = {}) => {
    const res = await fetch(url, options);
    const text = await res.text();
    let data = null;
    let parsed = false;
    if (text) {
      try {
        data = JSON.parse(text);
        parsed = true;
      } catch {
        parsed = false;
      }
    }
    if (!res.ok) {
      const message = data?.error || data?.message || text || `${res.status} ${res.statusText}`;
      const err = new Error(message);
      err.status = res.status;
      err.data = data;
      err.url = url;
      throw err;
    }
    if (text && !parsed) {
      const err = new Error('Invalid JSON response');
      err.status = res.status;
      err.data = null;
      err.url = url;
      err.raw = text;
      throw err;
    }
    return data;
  };

  const getJson = (url, options = {}) => {
    return requestJson(url, { ...options, method: 'GET' });
  };

  const postJson = (url, payload, options = {}) => {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    return requestJson(url, {
      ...options,
      method: 'POST',
      headers,
      body: JSON.stringify(payload ?? {})
    });
  };

  const postWithFallback = async (base, paths, payload, options = {}) => {
    const attempts = Array.isArray(paths) ? paths : [paths];
    let lastErr = null;
    for (const path of attempts) {
      const url = joinUrl(base, path || '');
      try {
        return await postJson(url, payload, options);
      } catch (err) {
        lastErr = err;
        if (err && err.status === 404) continue;
        break;
      }
    }
    throw lastErr || new Error('Request failed');
  };

  root.DemoAws = {
    DEFAULT_QUERY_KEYS,
    normalizeBase,
    joinUrl,
    listCandidates,
    resolveEndpoint,
    rememberEndpoint,
    requestJson,
    getJson,
    postJson,
    postWithFallback
  };
})(typeof window !== 'undefined' ? window : globalThis);
