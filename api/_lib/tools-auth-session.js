/*
  Tools account authentication bridge.

  Cognito ID tokens may be exchanged for a short-lived, encrypted, HttpOnly
  same-origin session cookie. Existing Bearer-token clients remain supported
  while TOOLS_AUTH_BEARER_FALLBACK is enabled (the default during migration).

  Required for cookie sessions:
  - TOOLS_SESSION_SECRETS: comma-separated 32-byte base64url or 64-character
    hex keys. The first key encrypts new cookies; all keys can decrypt during
    rotation. TOOLS_SESSION_SECRET is accepted as a single-key alias.
*/
'use strict';

const crypto = require('crypto');
const { getBearerToken } = require('./tools-api');
const { verifyCognitoIdToken } = require('./cognito-jwt');

const COOKIE_NAME = '__Host-tools_session';
const COOKIE_VERSION = 'v1';
const COOKIE_AAD = Buffer.from(`${COOKIE_NAME}:${COOKIE_VERSION}`, 'utf8');
const DEFAULT_SESSION_TTL_SECONDS = 8 * 60 * 60;
const MIN_SESSION_TTL_SECONDS = 15 * 60;
const MAX_SESSION_TTL_SECONDS = 24 * 60 * 60;
const MAX_COOKIE_VALUE_BYTES = 3800;
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

function createAuthError(code, message, statusCode = 401){
  const err = new Error(message);
  err.code = code;
  err.statusCode = statusCode;
  return err;
}

function base64UrlEncode(value){
  return Buffer.from(value)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function base64UrlDecode(value){
  const normalized = String(value || '').replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized.padEnd(normalized.length + (4 - normalized.length % 4) % 4, '=');
  return Buffer.from(padded, 'base64');
}

function parseSessionKey(value){
  const raw = String(value || '').trim();
  if (!raw) return null;
  if (/^[a-f0-9]{64}$/i.test(raw)) return Buffer.from(raw, 'hex');
  if (!/^[A-Za-z0-9_-]+$/.test(raw)) return null;
  const decoded = base64UrlDecode(raw);
  return decoded.length === 32 ? decoded : null;
}

function getSessionKeys(){
  const configured = String(process.env.TOOLS_SESSION_SECRETS || process.env.TOOLS_SESSION_SECRET || '').trim();
  const entries = configured.split(',').map((entry) => entry.trim()).filter(Boolean);
  if (!entries.length) {
    throw createAuthError(
      'TOOLS_SESSION_SECRET_MISSING',
      'Cookie sessions are not configured. Set TOOLS_SESSION_SECRETS to at least one 32-byte key.',
      503
    );
  }
  const keys = entries.map(parseSessionKey);
  if (keys.some((key) => !key)) {
    throw createAuthError(
      'TOOLS_SESSION_SECRET_INVALID',
      'TOOLS_SESSION_SECRETS contains an invalid key. Every key must decode to exactly 32 bytes.',
      503
    );
  }
  return keys;
}

function getSessionTtlSeconds(){
  const configured = Number(process.env.TOOLS_SESSION_TTL_SECONDS);
  if (!Number.isFinite(configured)) return DEFAULT_SESSION_TTL_SECONDS;
  return Math.max(MIN_SESSION_TTL_SECONDS, Math.min(MAX_SESSION_TTL_SECONDS, Math.floor(configured)));
}

function isBearerFallbackEnabled(){
  const raw = String(process.env.TOOLS_AUTH_BEARER_FALLBACK || '').trim().toLowerCase();
  if (!raw) return true;
  return !['0', 'false', 'no', 'off'].includes(raw);
}

function normalizeClaim(value, maxLength){
  return String(value || '').trim().slice(0, maxLength);
}

function normalizeGroups(value){
  const entries = Array.isArray(value)
    ? value
    : String(value || '').split(/[\s,]+/g);
  return entries
    .map((entry) => normalizeClaim(entry, 80).toLowerCase())
    .filter(Boolean)
    .slice(0, 20);
}

function createSessionPayload(claims, nowSeconds = Math.floor(Date.now() / 1000)){
  const sub = normalizeClaim(claims?.sub, 160);
  if (!sub) throw createAuthError('AUTH_UNAUTHORIZED', 'Verified token is missing a subject.', 401);
  const ttlSeconds = getSessionTtlSeconds();
  const authenticationTime = Number(claims?.auth_time);
  const authenticationExpiresAt = Number.isFinite(authenticationTime) && authenticationTime <= nowSeconds + 60
    ? authenticationTime + ttlSeconds
    : nowSeconds + ttlSeconds;
  const expiresAt = Math.min(nowSeconds + ttlSeconds, authenticationExpiresAt);
  if (expiresAt <= nowSeconds) {
    throw createAuthError('AUTH_REAUTH_REQUIRED', 'A fresh sign-in is required.', 401);
  }
  return {
    v: 1,
    sub,
    email: normalizeClaim(claims?.email, 320),
    name: normalizeClaim(claims?.name || claims?.['cognito:username'], 240),
    groups: normalizeGroups(claims?.['cognito:groups'] || claims?.groups),
    iat: nowSeconds,
    exp: expiresAt
  };
}

function encryptSessionPayload(payload, key = getSessionKeys()[0]){
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  cipher.setAAD(COOKIE_AAD);
  const encrypted = Buffer.concat([
    cipher.update(JSON.stringify(payload), 'utf8'),
    cipher.final()
  ]);
  const tag = cipher.getAuthTag();
  const value = [
    COOKIE_VERSION,
    base64UrlEncode(iv),
    base64UrlEncode(encrypted),
    base64UrlEncode(tag)
  ].join('.');
  if (Buffer.byteLength(value, 'utf8') > MAX_COOKIE_VALUE_BYTES) {
    throw createAuthError('TOOLS_SESSION_TOO_LARGE', 'Session cookie exceeds the safe size limit.', 500);
  }
  return value;
}

function decryptWithKey(parts, key){
  const iv = base64UrlDecode(parts[1]);
  const encrypted = base64UrlDecode(parts[2]);
  const tag = base64UrlDecode(parts[3]);
  if (iv.length !== 12 || tag.length !== 16 || !encrypted.length) {
    throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie.', 401);
  }
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAAD(COOKIE_AAD);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(encrypted), decipher.final()]);
}

function validateSessionPayload(payload, nowSeconds = Math.floor(Date.now() / 1000)){
  if (!payload || payload.v !== 1 || !normalizeClaim(payload.sub, 160)) {
    throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie.', 401);
  }
  const issuedAt = Number(payload.iat);
  const expiresAt = Number(payload.exp);
  if (!Number.isFinite(issuedAt) || !Number.isFinite(expiresAt) || issuedAt > nowSeconds + 60) {
    throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie.', 401);
  }
  if (expiresAt <= nowSeconds) {
    throw createAuthError('TOOLS_SESSION_EXPIRED', 'Session expired.', 401);
  }
  if (expiresAt - issuedAt > MAX_SESSION_TTL_SECONDS + 60) {
    throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie lifetime.', 401);
  }
  return {
    v: 1,
    sub: normalizeClaim(payload.sub, 160),
    email: normalizeClaim(payload.email, 320),
    name: normalizeClaim(payload.name, 240),
    groups: normalizeGroups(payload.groups),
    iat: issuedAt,
    exp: expiresAt
  };
}

function decryptSessionPayload(value, nowSeconds = Math.floor(Date.now() / 1000)){
  const raw = String(value || '').trim();
  const parts = raw.split('.');
  if (parts.length !== 4 || parts[0] !== COOKIE_VERSION || raw.length > MAX_COOKIE_VALUE_BYTES) {
    throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie.', 401);
  }
  const keys = getSessionKeys();
  for (const key of keys) {
    try {
      const decoded = decryptWithKey(parts, key);
      return validateSessionPayload(JSON.parse(decoded.toString('utf8')), nowSeconds);
    } catch (err) {
      if (err?.code === 'TOOLS_SESSION_EXPIRED') throw err;
    }
  }
  throw createAuthError('TOOLS_SESSION_INVALID', 'Invalid session cookie.', 401);
}

function parseCookies(req){
  const header = String(req?.headers?.cookie || '');
  const cookies = {};
  header.split(';').forEach((part) => {
    const index = part.indexOf('=');
    if (index < 1) return;
    const name = part.slice(0, index).trim();
    if (!name || Object.prototype.hasOwnProperty.call(cookies, name)) return;
    cookies[name] = part.slice(index + 1).trim();
  });
  return cookies;
}

function getSessionCookie(req){
  return String(parseCookies(req)[COOKIE_NAME] || '').trim();
}

function serializeSessionCookie(value, expiresAtSeconds){
  const maxAge = Math.max(0, Math.floor(Number(expiresAtSeconds) - Date.now() / 1000));
  return `${COOKIE_NAME}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}; Expires=${new Date(Number(expiresAtSeconds) * 1000).toUTCString()}; Priority=High`;
}

function serializeClearedSessionCookie(){
  return `${COOKIE_NAME}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Priority=High`;
}

function requestOrigin(req){
  const headers = req?.headers || {};
  const forwardedProto = String(headers['x-forwarded-proto'] || '').split(',')[0].trim().toLowerCase();
  const protocol = forwardedProto || (req?.socket?.encrypted ? 'https' : 'http');
  const host = String(headers.host || headers['x-forwarded-host'] || '').split(',')[0].trim().toLowerCase();
  return host ? `${protocol}://${host}` : '';
}

function assertSameOriginRequest(req){
  const method = String(req?.method || 'GET').toUpperCase();
  if (!UNSAFE_METHODS.has(method)) return;
  const suppliedOrigin = String(req?.headers?.origin || '').trim().toLowerCase().replace(/\/$/, '');
  const expectedOrigin = requestOrigin(req);
  if (!suppliedOrigin || !expectedOrigin || suppliedOrigin !== expectedOrigin) {
    throw createAuthError('AUTH_ORIGIN_MISMATCH', 'Same-origin request required.', 403);
  }
}

function sessionPayloadToClaims(payload){
  return {
    sub: payload.sub,
    email: payload.email,
    name: payload.name,
    'cognito:groups': payload.groups,
    iat: payload.iat,
    exp: payload.exp
  };
}

async function authenticateToolsRequest(req, options = {}){
  const cookie = getSessionCookie(req);
  let cookieError = null;
  if (cookie) {
    try {
      const payload = decryptSessionPayload(cookie);
      if (options.enforceSameOrigin !== false) assertSameOriginRequest(req);
      return { source: 'cookie', claims: sessionPayloadToClaims(payload), expiresAt: payload.exp };
    } catch (err) {
      cookieError = err;
    }
  }

  if (cookieError?.code === 'AUTH_ORIGIN_MISMATCH') throw cookieError;

  const allowBearer = typeof options.allowBearer === 'boolean'
    ? options.allowBearer
    : isBearerFallbackEnabled();
  const token = allowBearer ? getBearerToken(req) : '';
  if (token) {
    const verifyToken = typeof options.verifyToken === 'function'
      ? options.verifyToken
      : verifyCognitoIdToken;
    const claims = await verifyToken(token);
    const sub = normalizeClaim(claims?.sub, 160);
    if (!sub) throw createAuthError('AUTH_UNAUTHORIZED', 'Unauthorized.', 401);
    return { source: 'bearer', claims, expiresAt: Number(claims?.exp) || 0 };
  }

  if (['TOOLS_SESSION_SECRET_MISSING', 'TOOLS_SESSION_SECRET_INVALID'].includes(cookieError?.code)) throw cookieError;
  throw createAuthError('AUTH_UNAUTHORIZED', 'Unauthorized.', 401);
}

function createSessionFromClaims(claims){
  const payload = createSessionPayload(claims);
  const value = encryptSessionPayload(payload);
  return {
    value,
    payload,
    cookie: serializeSessionCookie(value, payload.exp)
  };
}

module.exports = {
  COOKIE_NAME,
  DEFAULT_SESSION_TTL_SECONDS,
  MIN_SESSION_TTL_SECONDS,
  MAX_SESSION_TTL_SECONDS,
  getSessionKeys,
  getSessionTtlSeconds,
  isBearerFallbackEnabled,
  createSessionPayload,
  encryptSessionPayload,
  decryptSessionPayload,
  getSessionCookie,
  serializeSessionCookie,
  serializeClearedSessionCookie,
  assertSameOriginRequest,
  sessionPayloadToClaims,
  authenticateToolsRequest,
  createSessionFromClaims
};
