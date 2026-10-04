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
    queryKeys = DEFAULT_QUERY_KEYS,
    allowOverrides = true
  } = {}) => {
    const out = [];
    const fromQuery = allowOverrides ? readQuery(queryKeys) : '';
    if (fromQuery) {
      out.push(normalizeBase(fromQuery));
      if (storageKey) safeSet(storageKey, fromQuery);
    }

    const storageKeys = allowOverrides && Array.isArray(legacyKeys)
      ? legacyKeys.slice()
      : (allowOverrides && legacyKeys ? [legacyKeys] : []);
    if (allowOverrides && storageKey) storageKeys.unshift(storageKey);
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

  const RETRYABLE_STATUSES = new Set([408, 425, 500, 502, 503, 504]);

  const endpointConfigurationErrors = new Map();
  const isConfigurationError = (err) => err?.status === 503
    && err?.code === 'DEMO_PROXY_CONFIGURATION_UNAVAILABLE';

  const runEndpointRequest = async (base, operation, refreshConfiguration = false) => {
    const key = normalizeBase(base);
    if (refreshConfiguration) endpointConfigurationErrors.delete(key);
    if (endpointConfigurationErrors.has(key)) throw endpointConfigurationErrors.get(key);
    try {
      return await operation();
    } catch (err) {
      if (isConfigurationError(err)) endpointConfigurationErrors.set(key, err);
      throw err;
    }
  };

  const isRetryableError = (err) => {
    if (['DEMO_REQUEST_CANCELLED', 'DEMO_REQUEST_TIMEOUT', 'DEMO_FUNCTION_ERROR', 'DEMO_TIMEOUT'].includes(err?.code)) return false;
    if (isConfigurationError(err)) return false;
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

  const boundedRequest = async (operation, options = {}) => {
    const timeoutMs = Number.isFinite(options.timeoutMs) && options.timeoutMs > 0 ? options.timeoutMs : 0;
    if (!timeoutMs && !options.signal) return operation(undefined);
    const controller = new AbortController();
    let timer = null;
    let expired = false;
    let rejectAbort;
    const abort = () => {
      controller.abort();
      const err = new Error(expired ? 'The request timed out. Please try again.' : 'The request was cancelled.');
      err.name = expired ? 'TimeoutError' : 'AbortError';
      err.code = expired ? 'DEMO_REQUEST_TIMEOUT' : 'DEMO_REQUEST_CANCELLED';
      rejectAbort(err);
    };
    const interrupted = new Promise((resolve, reject) => { rejectAbort = reject; });
    if (options.signal?.aborted) abort();
    else options.signal?.addEventListener('abort', abort, { once: true });
    if (timeoutMs) timer = setTimeout(() => { expired = true; abort(); }, timeoutMs);
    try {
      // The race also bounds response-body reads and implementations that ignore abort.
      return await Promise.race([interrupted, Promise.resolve().then(() => {
        if (controller.signal.aborted) throw new Error('Request aborted before sending.');
        return operation(controller.signal);
      })]);
    } finally {
      if (timer !== null) clearTimeout(timer);
      options.signal?.removeEventListener('abort', abort);
    }
  };

  const retryRequest = (operation, options = {}) => boundedRequest(async (signal) => {
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
      if (signal?.aborted) return;
      try {
        return await operation(attempt, { signal });
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
  }, options);

  const readJsonResponse = async (url, options = {}) => {
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
      const message = data?.error || data?.message || (typeof data?.detail === 'string' ? data.detail : null) || text || `${res.status} ${res.statusText}`;
      const err = new Error(message);
      err.status = res.status;
      err.code = typeof data?.code === 'string' ? data.code : '';
      const retryAfter = Number.parseInt(String(res.headers.get('Retry-After') || ''), 10);
      if (Number.isFinite(retryAfter) && retryAfter > 0) err.retryAfter = retryAfter;
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

  const requestJson = (url, options = {}) => boundedRequest((signal) => {
    const { timeoutMs, ...requestOptions } = options;
    return readJsonResponse(url, { ...requestOptions, ...(signal ? { signal } : {}) });
  }, options);

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

  const healthJson = (base, options = {}) => {
    // A new health check allows manual reconnect after configuration is corrected.
    return runEndpointRequest(base, () => getJson(joinUrl(normalizeBase(base), 'health'), options), true);
  };

  const warmupJson = (base, payload = {}, options = {}) => {
    return runEndpointRequest(base, () => postJson(joinUrl(normalizeBase(base), 'warmup'), payload, options));
  };

  const postWithFallback = async (base, paths, payload, options = {}) => {
    const key = normalizeBase(base);
    if (endpointConfigurationErrors.has(key)) {
      const error = endpointConfigurationErrors.get(key);
      // Skip this initialization fallback, while allowing a later user retry.
      endpointConfigurationErrors.delete(key);
      throw error;
    }
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
    healthJson,
    warmupJson,
    postWithFallback
  };
})(typeof window !== 'undefined' ? window : globalThis);
