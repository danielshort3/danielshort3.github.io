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

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const RETRYABLE_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);

  const isRetryableError = (err) => {
    if (!err) return false;
    if (typeof err.status === 'number') return RETRYABLE_STATUSES.has(err.status);
    if (err.name === 'AbortError') return true;
    const message = String(err.message || '').toLowerCase();
    return (
      message.includes('failed to fetch') ||
      message.includes('networkerror') ||
      message.includes('load failed') ||
      message.includes('timed out') ||
      message.includes('timeout')
    );
  };

  const retryRequest = async (operation, options = {}) => {
    const retries = Number.isFinite(options.retries) ? Math.max(0, options.retries) : 2;
    const baseDelayMs = Number.isFinite(options.baseDelayMs) ? Math.max(0, options.baseDelayMs) : 600;
    const factor = Number.isFinite(options.factor) && options.factor > 1 ? options.factor : 2;
    const maxDelayMs = Number.isFinite(options.maxDelayMs)
      ? Math.max(0, options.maxDelayMs)
      : 2500;
    const shouldRetry = typeof options.shouldRetry === 'function'
      ? options.shouldRetry
      : isRetryableError;

    let attempt = 0;
    let delayMs = baseDelayMs;
    let lastErr = null;

    while (attempt <= retries) {
      try {
        return await operation(attempt);
      } catch (err) {
        lastErr = err;
        if (attempt >= retries || !shouldRetry(err, attempt)) {
          throw err;
        }
        if (delayMs > 0) {
          await sleep(delayMs);
        }
        delayMs = Math.min(Math.max(delayMs * factor, 1), maxDelayMs);
        attempt += 1;
      }
    }

    throw lastErr || new Error('Request failed');
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
    isRetryableError,
    retryRequest,
    requestJson,
    getJson,
    postJson,
    postWithFallback
  };
})(typeof window !== 'undefined' ? window : globalThis);
