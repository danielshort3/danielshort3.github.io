/*
  Same-origin proxy for the Smart Sentence Retriever Lambda.

  The Lambda URL currently does not emit browser-readable CORS headers for
  localhost, so browser requests go through this route instead.
*/
'use strict';

const DEFAULT_SENTENCE_ENDPOINT = 'https://7aa3jt3tzkinheprc5ds52bihq0itutx.lambda-url.us-east-2.on.aws/';
const SITE_ORIGIN = 'https://www.danielshort.me';
const MAX_BODY_BYTES = 24_000;
const UPSTREAM_TIMEOUT_MS = 25_000;

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function getUpstreamBase() {
  return pickEnv(['SENTENCE_DEMO_ENDPOINT', 'SENTENCE_DEMO_LAMBDA_URL'])
    || DEFAULT_SENTENCE_ENDPOINT;
}

function isProductionRuntime() {
  return String(process.env.VERCEL_ENV || '').trim() === 'production'
    || String(process.env.NODE_ENV || '').trim() === 'production';
}

function allowedOrigins() {
  const configured = String(process.env.SENTENCE_DEMO_ALLOWED_ORIGINS || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  const vercelUrl = process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : '';
  return new Set([
    SITE_ORIGIN,
    'https://danielshort.me',
    vercelUrl,
    ...configured
  ].filter(Boolean));
}

function isAllowedOrigin(origin) {
  if (!origin) return true;
  if (allowedOrigins().has(origin)) return true;

  try {
    const url = new URL(origin);
    if (!isProductionRuntime() && ['localhost', '127.0.0.1', '::1'].includes(url.hostname)) {
      return true;
    }
    return !isProductionRuntime() && url.hostname.endsWith('.vercel.app');
  } catch {
    return false;
  }
}

function normalizeRoute(value) {
  const route = String(value || '').replace(/^\/+|\/+$/g, '');
  if (route === 'health') return 'health';
  if (route === 'rank') return 'rank';
  return '';
}

function getRouteFromRequest(req) {
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return normalizeRoute(querySlug.join('/'));
  if (typeof querySlug === 'string') return normalizeRoute(querySlug);

  try {
    const url = new URL(req.url, 'https://example.com');
    const match = url.pathname.match(/\/api\/sentence-demo\/(.+)$/);
    return normalizeRoute(match ? decodeURIComponent(match[1]) : '');
  } catch {
    return '';
  }
}

function setCorsHeaders(req, res) {
  const origin = String(req.headers.origin || '').trim();
  if (!isAllowedOrigin(origin)) return false;

  if (origin) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  } else {
    res.setHeader('Access-Control-Allow-Origin', SITE_ORIGIN);
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Max-Age', '600');
  return true;
}

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(JSON.stringify(payload));
}

async function readBody(req) {
  if (req.body && typeof req.body === 'object') return JSON.stringify(req.body);
  if (typeof req.body === 'string') return req.body;

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > MAX_BODY_BYTES) {
      const err = new Error('Request body too large');
      err.status = 413;
      throw err;
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function buildUpstreamUrl(route) {
  const base = getUpstreamBase();
  const normalizedBase = String(base || '').endsWith('/') ? base : `${base}/`;
  return new URL(route, normalizedBase).toString();
}

async function fetchUpstream(route, req) {
  const method = String(req.method || 'GET').toUpperCase();
  const headers = { Accept: 'application/json' };
  const init = { method, headers };

  if (method === 'POST') {
    init.body = await readBody(req);
    headers['Content-Type'] = 'application/json';
  }

  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    init.signal = AbortSignal.timeout(UPSTREAM_TIMEOUT_MS);
  }

  return fetch(buildUpstreamUrl(route), init);
}

module.exports = async (req, res) => {
  if (!setCorsHeaders(req, res)) {
    sendJson(res, 403, { ok: false, error: 'Origin not allowed' });
    return;
  }

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  const route = getRouteFromRequest(req);
  if (!route) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const method = String(req.method || 'GET').toUpperCase();
  if (route === 'health' && method !== 'GET' && method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD, OPTIONS');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }
  if (route === 'rank' && method !== 'POST') {
    res.setHeader('Allow', 'POST, OPTIONS');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  let upstream;
  try {
    upstream = await fetchUpstream(route, req);
  } catch (err) {
    const isTimeout = err && (err.name === 'TimeoutError' || err.name === 'AbortError');
    sendJson(res, isTimeout ? 504 : 502, {
      ok: false,
      error: isTimeout ? 'Sentence service timed out' : 'Sentence service unavailable'
    });
    return;
  }

  const text = await upstream.text().catch(() => '');
  const contentType = upstream.headers.get('content-type') || 'application/json; charset=utf-8';
  res.statusCode = upstream.status;
  res.setHeader('Content-Type', contentType);
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  if (method === 'HEAD') {
    res.end();
    return;
  }
  res.end(text);
};
