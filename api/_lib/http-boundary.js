'use strict';

const DEFAULT_SITE_ORIGINS = Object.freeze([
  'https://www.danielshort.me',
  'https://danielshort.me'
]);

function headerValue(req, name) {
  const headers = req && req.headers && typeof req.headers === 'object' ? req.headers : {};
  const lowerName = String(name || '').toLowerCase();
  const direct = headers[lowerName];
  if (Array.isArray(direct)) return String(direct[0] || '').trim();
  if (typeof direct !== 'undefined') return String(direct || '').trim();

  const match = Object.keys(headers).find((key) => String(key).toLowerCase() === lowerName);
  if (!match) return '';
  const value = headers[match];
  return Array.isArray(value) ? String(value[0] || '').trim() : String(value || '').trim();
}

function normalizeOrigin(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    const parsed = new URL(raw);
    if (!['http:', 'https:'].includes(parsed.protocol)) return '';
    return parsed.origin;
  } catch {
    return '';
  }
}

function configuredOrigins(env, key) {
  return String((env && env[key]) || '')
    .split(',')
    .map(normalizeOrigin)
    .filter(Boolean);
}

function requestHostOrigin(req) {
  const host = headerValue(req, 'x-forwarded-host') || headerValue(req, 'host');
  if (!host || /[\s/\\]/.test(host)) return '';
  const proto = (headerValue(req, 'x-forwarded-proto') || 'https').split(',')[0].trim().toLowerCase();
  if (!['http', 'https'].includes(proto)) return '';
  return normalizeOrigin(`${proto}://${host}`);
}

function allowedOrigins(req, options = {}) {
  const env = options.env || process.env;
  const origins = new Set(options.includeDefaultSiteOrigins === false ? [] : DEFAULT_SITE_ORIGINS);
  const requestOrigin = requestHostOrigin(req);
  const vercelUrl = String(env.VERCEL_URL || '').trim();

  if (requestOrigin) origins.add(requestOrigin);
  if (vercelUrl) {
    const previewOrigin = normalizeOrigin(`https://${vercelUrl}`);
    if (previewOrigin) origins.add(previewOrigin);
  }
  for (const origin of configuredOrigins(env, options.allowedOriginsEnv || 'ALLOWED_ORIGINS')) {
    origins.add(origin);
  }
  return origins;
}

function isSameOriginRequest(req, options = {}) {
  const fetchSite = headerValue(req, 'sec-fetch-site').toLowerCase();
  if (fetchSite === 'cross-site') return false;

  const rawOrigin = headerValue(req, 'origin');
  if (!rawOrigin) return true;
  const origin = normalizeOrigin(rawOrigin);
  if (!origin) return false;
  return allowedOrigins(req, options).has(origin);
}

function applyBoundaryHeaders(req, res, options = {}) {
  const methods = Array.isArray(options.methods) && options.methods.length
    ? options.methods.map((method) => String(method).toUpperCase())
    : ['GET', 'POST', 'OPTIONS'];
  const rawOrigin = headerValue(req, 'origin');
  const origin = normalizeOrigin(rawOrigin);

  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'same-origin');
  res.setHeader('Access-Control-Allow-Methods', methods.join(', '));
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Max-Age', '600');

  if (origin && allowedOrigins(req, options).has(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  }
}

function httpError(statusCode, code, message) {
  const err = new Error(message);
  err.statusCode = statusCode;
  err.code = code;
  return err;
}

async function readRawBody(req, maxBytes) {
  const limit = Number.isFinite(maxBytes) ? Math.max(0, maxBytes) : 64_000;
  const contentLength = Number(headerValue(req, 'content-length'));
  if (Number.isFinite(contentLength) && contentLength > limit) {
    throw httpError(413, 'REQUEST_BODY_TOO_LARGE', 'Request body too large');
  }

  if (Buffer.isBuffer(req.body)) {
    if (req.body.length > limit) throw httpError(413, 'REQUEST_BODY_TOO_LARGE', 'Request body too large');
    return req.body.toString('utf8');
  }
  if (typeof req.body === 'string') {
    if (Buffer.byteLength(req.body, 'utf8') > limit) {
      throw httpError(413, 'REQUEST_BODY_TOO_LARGE', 'Request body too large');
    }
    return req.body;
  }
  if (req.body && typeof req.body === 'object') {
    let serialized;
    try {
      serialized = JSON.stringify(req.body);
    } catch {
      throw httpError(400, 'REQUEST_JSON_INVALID', 'Invalid JSON body');
    }
    if (Buffer.byteLength(serialized, 'utf8') > limit) {
      throw httpError(413, 'REQUEST_BODY_TOO_LARGE', 'Request body too large');
    }
    return serialized;
  }

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > limit) throw httpError(413, 'REQUEST_BODY_TOO_LARGE', 'Request body too large');
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function readJsonBody(req, maxBytes, options = {}) {
  const raw = await readRawBody(req, maxBytes);
  if (!raw.trim()) return options.allowEmpty ? { raw: '', value: {} } : (() => {
    throw httpError(400, 'REQUEST_BODY_REQUIRED', 'JSON body required');
  })();

  const contentType = headerValue(req, 'content-type').toLowerCase();
  if (options.requireJsonContentType !== false && contentType && !contentType.includes('application/json')) {
    throw httpError(415, 'REQUEST_CONTENT_TYPE_INVALID', 'Content-Type must be application/json');
  }

  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    throw httpError(400, 'REQUEST_JSON_INVALID', 'Invalid JSON body');
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw httpError(400, 'REQUEST_JSON_OBJECT_REQUIRED', 'JSON body must be an object');
  }
  return { raw, value };
}

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(JSON.stringify(payload));
}

function clientIp(req) {
  const forwarded = headerValue(req, 'x-forwarded-for').split(',')[0].trim();
  const real = headerValue(req, 'x-real-ip');
  const socket = req && req.socket && req.socket.remoteAddress ? String(req.socket.remoteAddress) : '';
  return forwarded || real || socket || 'unknown';
}

module.exports = {
  DEFAULT_SITE_ORIGINS,
  allowedOrigins,
  applyBoundaryHeaders,
  clientIp,
  headerValue,
  httpError,
  isSameOriginRequest,
  normalizeOrigin,
  readJsonBody,
  readRawBody,
  requestHostOrigin,
  sendJson
};
