'use strict';

const UPSTREAM_BASE = 'https://fhp2is6v8h.execute-api.us-east-2.amazonaws.com/prod';
const LOCAL_PREFIX = '/api/job-tracker';
const MAX_BODY_BYTES = 1024 * 1024;
const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 30000;
const LIST_QUERY = ['start', 'end', 'limit', 'cursor'];
const ROUTES = Object.freeze({
  '/api/applications': { methods: ['GET', 'POST'], query: LIST_QUERY },
  '/api/applications/capture': { methods: ['POST'] },
  '/api/prospects': { methods: ['GET', 'POST'], query: LIST_QUERY },
  '/api/views': { methods: ['GET', 'POST'] },
  '/api/exports': { methods: ['POST'] },
  '/api/attachments/presign': { methods: ['POST'] },
  '/api/attachments/download': { methods: ['POST'] },
  '/api/attachments/zip': { methods: ['POST'] }
});
const ANALYTICS = new Set(['dashboard', 'summary', 'applications-over-time', 'status-breakdown', 'calendar', 'funnel', 'time-in-stage', 'followups']);

function sendJson(res, status, value) {
  if (res.destroyed || res.writableEnded) return;
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(JSON.stringify(value));
}

function getRoute(pathname) {
  if (Object.hasOwn(ROUTES, pathname)) return ROUTES[pathname];
  const analytics = /^\/api\/analytics\/([a-z-]+)$/.exec(pathname);
  if (analytics && ANALYTICS.has(analytics[1])) {
    return { methods: ['GET'], query: analytics[1] === 'followups' ? ['start', 'end', 'includeOverdue'] : ['start', 'end'] };
  }
  const record = /^\/api\/(applications|prospects|views)\/([^/]+)$/.exec(pathname);
  if (!record) return null;
  let id;
  try { id = decodeURIComponent(record[2]); } catch { return null; }
  if (!/^[A-Za-z0-9_:#-]{1,256}$/.test(id)) return null;
  return { methods: record[1] === 'views' ? ['DELETE'] : ['PATCH', 'DELETE'] };
}

function validQuery(url, route, method) {
  const allowed = method === 'GET' ? route.query || [] : [];
  const seen = new Set();
  for (const [key, value] of url.searchParams) {
    if (!allowed.includes(key) || seen.has(key)) return false;
    seen.add(key);
    if (key === 'cursor') {
      if (!/^[A-Za-z0-9_-]{1,2048}$/.test(value)) return false;
    } else if (key === 'limit') {
      if (!/^[1-9]\d{0,5}$/.test(value)) return false;
    } else if (key === 'includeOverdue') {
      if (!['true', 'false'].includes(value)) return false;
    } else if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  }
  return true;
}

function readBody(req, signal) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    const cleanup = () => {
      req.removeListener('data', onData);
      req.removeListener('end', onEnd);
      req.removeListener('error', onError);
      signal.removeEventListener('abort', onAbort);
    };
    const fail = (status) => {
      cleanup();
      req.resume();
      reject(Object.assign(new Error('Request body rejected'), { status }));
    };
    const onData = (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) { fail(413); return; }
      chunks.push(Buffer.from(chunk));
    };
    const onEnd = () => { cleanup(); resolve(Buffer.concat(chunks)); };
    const onError = () => fail(400);
    const onAbort = () => fail(504);
    req.on('data', onData);
    req.once('end', onEnd);
    req.once('error', onError);
    signal.addEventListener('abort', onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
}

async function handleLocalJobTrackerRequest(req, res, { fetchImpl = fetch, timeoutMs = REQUEST_TIMEOUT_MS } = {}) {
  let url;
  let origin;
  try {
    const host = String(req.headers.host || '');
    const local = new URL(`http://${host}`);
    if (!['localhost', '127.0.0.1', '[::1]'].includes(local.hostname) || local.username || local.password ||
        local.pathname !== '/' || local.search || local.hash || host !== local.host) throw new Error('Invalid host');
    origin = local.origin;
    const rawUrl = String(req.url || '');
    if (!rawUrl.startsWith(`${LOCAL_PREFIX}/`) || rawUrl.length > 8192 || rawUrl.includes('\\') || rawUrl.includes('#')) throw new Error('Invalid URL');
    url = new URL(rawUrl, origin);
    if (url.origin !== origin || url.pathname !== rawUrl.split('?')[0]) throw new Error('Invalid URL');
  } catch {
    sendJson(res, 400, { error: 'Invalid local tracker address.' });
    return;
  }
  const site = String(req.headers['sec-fetch-site'] || '').toLowerCase();
  if ((site && !['same-origin', 'none'].includes(site)) || (req.headers.origin && req.headers.origin !== origin)) {
    sendJson(res, 403, { error: 'Same-origin tracker request required.' });
    return;
  }
  const path = url.pathname.slice(LOCAL_PREFIX.length);
  const route = getRoute(path);
  if (!route) { sendJson(res, 404, { error: 'Unknown local tracker route.' }); return; }
  if (!route.methods.includes(req.method)) {
    res.setHeader('Allow', route.methods.join(', '));
    sendJson(res, 405, { error: 'Method not allowed.' });
    return;
  }
  if (!validQuery(url, route, req.method)) { sendJson(res, 400, { error: 'Invalid tracker query.' }); return; }
  const authorization = String(req.headers.authorization || '');
  // API Gateway remains the JWT verifier. Local cookies cannot authorize this separate service.
  if (!/^Bearer [A-Za-z0-9._~+\/-]+=*$/i.test(authorization) || authorization.length > 16384) {
    sendJson(res, 401, { error: 'Sign in again to connect to Job Application Tracker.' });
    return;
  }
  const hasBody = req.method === 'POST' || req.method === 'PATCH';
  const contentType = String(req.headers['content-type'] || '');
  if (hasBody && (contentType.split(';')[0].trim().toLowerCase() !== 'application/json' || req.headers['content-encoding'])) {
    sendJson(res, 415, { error: 'Content-Type must be uncompressed application/json.' });
    return;
  }
  if (Number(req.headers['content-length']) > MAX_BODY_BYTES ||
      (!hasBody && (Number(req.headers['content-length']) > 0 || req.headers['transfer-encoding']))) {
    res.setHeader('Connection', 'close');
    sendJson(res, hasBody ? 413 : 400, { error: 'Invalid tracker request size.' });
    return;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Math.min(Math.max(timeoutMs, 1), REQUEST_TIMEOUT_MS));
  const cancel = () => controller.abort();
  res.once('close', cancel);
  req.once('aborted', cancel);
  try {
    let body;
    if (hasBody) {
      body = await readBody(req, controller.signal);
      let parsed;
      try { parsed = JSON.parse(body.toString('utf8')); } catch {
        sendJson(res, 400, { error: 'Invalid JSON request.' }); return;
      }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        sendJson(res, 400, { error: 'Tracker requests require a JSON object.' }); return;
      }
    }
    const upstream = await fetchImpl(`${UPSTREAM_BASE}${path}${url.search}`, {
      method: req.method,
      headers: { Authorization: authorization, Accept: 'application/json', ...(contentType ? { 'Content-Type': contentType } : {}) },
      ...(body ? { body } : {}),
      redirect: 'error', signal: controller.signal
    });
    if (upstream.status >= 300 && upstream.status < 400) throw new Error('Unexpected redirect');
    if (Number(upstream.headers.get('content-length')) > MAX_RESPONSE_BYTES) throw new Error('Response too large');
    const chunks = [];
    let size = 0;
    if (upstream.body) {
      for await (const chunk of upstream.body) {
        size += chunk.length;
        if (size > MAX_RESPONSE_BYTES) throw new Error('Response too large');
        chunks.push(Buffer.from(chunk));
      }
    }
    const output = Buffer.concat(chunks);
    if (upstream.status !== 204) {
      if (!/^application\/json(?:\s*;|$)/i.test(upstream.headers.get('content-type') || '')) throw new Error('Unexpected content');
      JSON.parse(output.toString('utf8'));
    }
    if (res.destroyed || res.writableEnded) return;
    const retryAfter = upstream.headers.get('retry-after') || '';
    if (/^\d{1,6}$/.test(retryAfter)) res.setHeader('Retry-After', retryAfter);
    if (upstream.status >= 500) {
      sendJson(res, upstream.status, { error: 'Job tracker service is temporarily unavailable.' });
      return;
    }
    res.statusCode = upstream.status;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.end(output);
  } catch (error) {
    const status = error.status || (controller.signal.aborted ? 504 : 502);
    controller.abort();
    res.setHeader('Connection', 'close');
    sendJson(res, status, { error: status === 413 ? 'Tracker request is too large.' : status === 504 ? 'Job tracker connection timed out.' : 'Local job tracker connection failed.' });
  } finally {
    clearTimeout(timeout);
    res.removeListener('close', cancel);
    req.removeListener('aborted', cancel);
  }
}

module.exports = { handleLocalJobTrackerRequest, MAX_BODY_BYTES, MAX_RESPONSE_BYTES };
