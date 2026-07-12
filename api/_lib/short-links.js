/*
  Short links helpers (validation, keys, auth).
*/
'use strict';

const crypto = require('crypto');

const MAX_JSON_BODY_BYTES = 256 * 1024;

const SHORTLINKS_PREFIX = 'shortlinks';
const SLUG_SET_KEY = `${SHORTLINKS_PREFIX}:slugs`;
const SET_KEY_PREFIX = '__set__/';
const BATCH_KEY_PREFIX = '__batch__/';
const CANONICAL_SITE_ORIGIN = 'https://www.danielshort.me';
const DEFAULT_RANDOM_LENGTH = 6;
const MIN_RANDOM_LENGTH = 4;
const MAX_RANDOM_LENGTH = 12;
const RANDOM_SLUG_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

function linkKey(slug){
  return `${SHORTLINKS_PREFIX}:link:${slug}`;
}

function clicksKey(slug){
  return `${SHORTLINKS_PREFIX}:clicks:${slug}`;
}

function buildSetRecordKey(setId){
  return `${SET_KEY_PREFIX}${setId}`;
}

function buildBatchRecordKey(batchId){
  return `${BATCH_KEY_PREFIX}${batchId}`;
}

function isInternalRecordSlug(slug){
  const raw = String(slug || '').trim();
  if (!raw) return false;
  return raw.startsWith(SET_KEY_PREFIX) || raw.startsWith(BATCH_KEY_PREFIX);
}

function getAdminToken(){
  const token = process.env.SHORTLINKS_ADMIN_TOKEN;
  return typeof token === 'string' ? token.trim() : '';
}

function isAdminRequest(req){
  const configured = getAdminToken();
  if (!configured) return false;

  const auth = (req.headers && req.headers.authorization) ? String(req.headers.authorization) : '';
  const headerToken = (req.headers && (req.headers['x-admin-token'] || req.headers['x-shortlinks-token']))
    ? String(req.headers['x-admin-token'] || req.headers['x-shortlinks-token'])
    : '';

  let provided = headerToken.trim();
  if (!provided && auth.toLowerCase().startsWith('bearer ')) {
    provided = auth.slice(7).trim();
  }
  if (!provided) return false;

  const providedBuffer = Buffer.from(provided, 'utf8');
  const configuredBuffer = Buffer.from(configured, 'utf8');
  if (providedBuffer.length !== configuredBuffer.length) {
    try {
      crypto.timingSafeEqual(configuredBuffer, configuredBuffer);
    } catch {}
    return false;
  }

  try {
    return crypto.timingSafeEqual(providedBuffer, configuredBuffer);
  } catch {
    return false;
  }
}

function sendJson(res, status, body){
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
}

function createBodyTooLargeError(maxBytes){
  const err = new Error(`JSON request body exceeds ${maxBytes} bytes.`);
  err.code = 'BODY_TOO_LARGE';
  err.statusCode = 413;
  return err;
}

function assertBodyWithinLimit(body, maxBytes){
  const raw = typeof body === 'string' ? body : JSON.stringify(body);
  if (Buffer.byteLength(raw, 'utf8') > maxBytes) throw createBodyTooLargeError(maxBytes);
}

function readJson(req, options = {}){
  const maxBytes = Math.max(1, Number(options.maxBytes) || MAX_JSON_BODY_BYTES);
  return new Promise((resolve, reject) => {
    const declaredBytes = Number(req?.headers?.['content-length']);
    if (Number.isFinite(declaredBytes) && declaredBytes > maxBytes) {
      reject(createBodyTooLargeError(maxBytes));
      return;
    }
    if (req.body && typeof req.body === 'object') {
      try {
        assertBodyWithinLimit(req.body, maxBytes);
        return resolve(req.body);
      } catch (err) {
        return reject(err);
      }
    }
    if (typeof req.body === 'string') {
      try {
        assertBodyWithinLimit(req.body, maxBytes);
        return resolve(JSON.parse(req.body));
      } catch (err) {
        return reject(err);
      }
    }
    let raw = '';
    let bytes = 0;
    let tooLarge = false;
    req.on('data', chunk => {
      if (tooLarge) return;
      bytes += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(String(chunk), 'utf8');
      if (bytes > maxBytes) {
        tooLarge = true;
        raw = '';
        reject(createBodyTooLargeError(maxBytes));
        return;
      }
      raw += chunk;
    });
    req.on('end', () => {
      if (tooLarge) return;
      if (!raw) return resolve({});
      try { resolve(JSON.parse(raw)); } catch (err) { reject(err); }
    });
    req.on('error', reject);
  });
}

function normalizeSlug(slug){
  const raw = String(slug || '').trim();
  const trimmed = raw.replace(/^\/+|\/+$/g, '');
  if (!trimmed) return '';
  if (trimmed.length > 128) return '';
  if (trimmed.includes('..')) return '';
  const ok = /^[A-Za-z0-9][A-Za-z0-9-_]*(?:\/[A-Za-z0-9][A-Za-z0-9-_]*)*$/.test(trimmed);
  return ok ? trimmed : '';
}

function normalizeSlugLower(slug){
  const normalized = normalizeSlug(slug);
  return normalized ? normalized.toLowerCase() : '';
}

function normalizeSetId(setId){
  const raw = String(setId || '').trim().toLowerCase();
  if (!raw) return '';
  if (raw.length > 64) return '';
  return /^[a-z0-9][a-z0-9-_]*$/.test(raw) ? raw : '';
}

function createSetId(){
  return crypto.randomBytes(8).toString('hex');
}

function createBatchId(){
  return crypto.randomBytes(10).toString('hex');
}

function normalizeRandomLength(value, fallback = DEFAULT_RANDOM_LENGTH){
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(MIN_RANDOM_LENGTH, Math.min(MAX_RANDOM_LENGTH, Math.floor(numeric)));
}

function generateRandomSlug(length){
  const safeLength = normalizeRandomLength(length);
  let slug = '';
  for (let i = 0; i < safeLength; i += 1) {
    const index = crypto.randomInt(0, RANDOM_SLUG_CHARSET.length);
    slug += RANDOM_SLUG_CHARSET[index];
  }
  return slug;
}

function normalizeDestination(destination, options = {}){
  const raw = String(destination || '').trim();
  if (!raw) return '';
  if (raw.length > 2048) return '';
  if (raw.startsWith('/')) {
    if (options && options.absolutizeInternalPath) {
      try {
        return new URL(raw, CANONICAL_SITE_ORIGIN).toString();
      } catch {
        return '';
      }
    }
    return raw;
  }
  if (/^https?:\/\//i.test(raw)) return raw;
  return '';
}

function getRequestBaseUrl(req){
  const protoHeader = req.headers && (req.headers['x-forwarded-proto'] || req.headers['x-forwarded-protocol']);
  const proto = protoHeader ? String(protoHeader).split(',')[0].trim() : 'https';
  const host = req.headers && req.headers.host ? String(req.headers.host) : 'localhost';
  return `${proto}://${host}`;
}

module.exports = {
  MAX_JSON_BODY_BYTES,
  SLUG_SET_KEY,
  SET_KEY_PREFIX,
  BATCH_KEY_PREFIX,
  DEFAULT_RANDOM_LENGTH,
  MIN_RANDOM_LENGTH,
  MAX_RANDOM_LENGTH,
  linkKey,
  clicksKey,
  buildSetRecordKey,
  buildBatchRecordKey,
  isInternalRecordSlug,
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeSlugLower,
  normalizeSetId,
  createSetId,
  createBatchId,
  normalizeRandomLength,
  generateRandomSlug,
  normalizeDestination,
  getRequestBaseUrl
};
