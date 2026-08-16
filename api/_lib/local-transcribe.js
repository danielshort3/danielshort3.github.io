/*
  Admin-only bridge configuration and signed tickets for the local transcription
  worker. The browser uploads media directly to the worker; the website API
  never receives the file or exposes the shared HMAC secret.

  Required to enable the bridge:
  - LOCAL_TRANSCRIBE_SHARED_SECRET (at least 32 UTF-8 bytes)
  - LOCAL_TRANSCRIBE_WORKER_ORIGIN (defaults to http://127.0.0.1:8765)

  Production should use a stable HTTPS tunnel. Loopback HTTP may only be used in
  production when LOCAL_TRANSCRIBE_ALLOW_LOOPBACK=true is set explicitly.
*/
'use strict';

const crypto = require('crypto');
const { readJson, sendJson, getBearerToken } = require('./tools-api');
const { verifyCognitoIdToken } = require('./cognito-jwt');
const {
  authenticateToolsRequest,
  createSessionFromClaims,
  isEmailVerifiedClaim
} = require('./tools-auth-session');

const SUPPORTED_FORMATS = Object.freeze(['amr', 'flac', 'm4a', 'mp3', 'mp4', 'ogg', 'wav', 'webm']);
const DEFAULT_ADMIN_GROUPS = Object.freeze(['admin', 'admins']);
const DEFAULT_ADMIN_EMAILS = Object.freeze(['daniel@danielshort.me', 'danielshort3@gmail.com']);
const DEFAULT_WORKER_ORIGIN = 'http://127.0.0.1:8765';
const DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_FILES_PER_RUN = 10;
const DEFAULT_MAX_FILE_BYTES = 500 * 1024 * 1024;
const DEFAULT_MAX_DURATION_SECONDS = 8 * 60 * 60;
const MIN_DURATION_SECONDS = 15;
const DEFAULT_TICKET_TTL_SECONDS = 6 * 60 * 60;
const MIN_TICKET_TTL_SECONDS = 15 * 60;
const MAX_TICKET_TTL_SECONDS = 12 * 60 * 60;
const MAX_FILENAME_CHARS = 120;

function hasOwn(object, key){
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function positiveSafeInteger(value, fallback){
  const number = Number(value);
  return Number.isSafeInteger(number) && number > 0 ? number : fallback;
}

function boundedInteger(value, fallback, min, max){
  const number = Number(value);
  if (!Number.isSafeInteger(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}

function envFlag(value){
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

function parseAllowlist(env, key, defaults){
  if (!hasOwn(env, key)) return new Set(defaults);
  return new Set(String(env[key] || '')
    .split(/[\s,;]+/g)
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean)
    .slice(0, 100));
}

function isLoopbackHostname(value){
  const hostname = String(value || '').trim().toLowerCase().replace(/^\[|\]$/g, '');
  return hostname === 'localhost' || hostname === '::1' || /^127(?:\.\d{1,3}){3}$/.test(hostname);
}

function normalizeHttpOrigin(value){
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw);
    if (!['http:', 'https:'].includes(url.protocol)) return '';
    if (url.username || url.password || url.search || url.hash) return '';
    if (url.pathname !== '/' && url.pathname !== '') return '';
    return url.origin === 'null' ? '' : url.origin;
  } catch {
    return '';
  }
}

function normalizeWorkerOrigin(value){
  const origin = normalizeHttpOrigin(value);
  if (!origin) return '';
  const url = new URL(origin);
  if (url.protocol === 'http:' && !isLoopbackHostname(url.hostname)) return '';
  return origin;
}

function isProductionRuntime(env){
  return String(env?.VERCEL_ENV || '').trim().toLowerCase() === 'production' ||
    String(env?.NODE_ENV || '').trim().toLowerCase() === 'production';
}

function getLocalTranscribeConfig(env = process.env){
  const rawSecret = typeof env.LOCAL_TRANSCRIBE_SHARED_SECRET === 'string'
    ? env.LOCAL_TRANSCRIBE_SHARED_SECRET
    : '';
  const sharedSecretBytes = Buffer.byteLength(rawSecret, 'utf8');
  const configuredOrigin = hasOwn(env, 'LOCAL_TRANSCRIBE_WORKER_ORIGIN')
    ? env.LOCAL_TRANSCRIBE_WORKER_ORIGIN
    : DEFAULT_WORKER_ORIGIN;
  const workerOrigin = normalizeWorkerOrigin(configuredOrigin);
  const production = isProductionRuntime(env);
  const loopbackOrigin = Boolean(workerOrigin && isLoopbackHostname(new URL(workerOrigin).hostname));
  const allowProductionLoopback = envFlag(env.LOCAL_TRANSCRIBE_ALLOW_LOOPBACK);
  const disabledReasons = [];

  if (sharedSecretBytes < 32) {
    disabledReasons.push('LOCAL_TRANSCRIBE_SHARED_SECRET must contain at least 32 UTF-8 bytes.');
  }
  if (!workerOrigin) {
    disabledReasons.push('LOCAL_TRANSCRIBE_WORKER_ORIGIN must be an HTTPS origin or a loopback HTTP origin with no path.');
  } else if (production && loopbackOrigin && !allowProductionLoopback) {
    disabledReasons.push(
      'Loopback workers are disabled in production. Configure a stable HTTPS worker origin or explicitly set LOCAL_TRANSCRIBE_ALLOW_LOOPBACK=true.'
    );
  }

  const maxFileBytes = Math.min(
    positiveSafeInteger(env.LOCAL_TRANSCRIBE_MAX_FILE_BYTES, DEFAULT_MAX_FILE_BYTES),
    DEFAULT_MAX_FILE_BYTES
  );
  const ticketTtlSeconds = boundedInteger(
    env.LOCAL_TRANSCRIBE_TICKET_TTL_SECONDS,
    DEFAULT_TICKET_TTL_SECONDS,
    MIN_TICKET_TTL_SECONDS,
    MAX_TICKET_TTL_SECONDS
  );
  const configured = disabledReasons.length === 0;

  return {
    enabled: configured,
    configured,
    disabledReason: disabledReasons.join(' '),
    workerOrigin,
    sharedSecret: rawSecret,
    // This is a fixed browser/worker protocol value. Changing it on only the
    // website would make Content-Range validation disagree with the worker.
    chunkBytes: DEFAULT_CHUNK_BYTES,
    maxFilesPerRun: positiveSafeInteger(env.LOCAL_TRANSCRIBE_MAX_FILES_PER_RUN, DEFAULT_MAX_FILES_PER_RUN),
    maxFileBytes,
    maxServiceDurationSeconds: Math.min(
      positiveSafeInteger(env.LOCAL_TRANSCRIBE_MAX_DURATION_SECONDS, DEFAULT_MAX_DURATION_SECONDS),
      DEFAULT_MAX_DURATION_SECONDS
    ),
    minDurationSeconds: MIN_DURATION_SECONDS,
    ticketTtlSeconds,
    adminGroups: parseAllowlist(env, 'LOCAL_TRANSCRIBE_ADMIN_GROUPS', DEFAULT_ADMIN_GROUPS),
    adminEmails: parseAllowlist(env, 'LOCAL_TRANSCRIBE_ADMIN_EMAILS', DEFAULT_ADMIN_EMAILS)
  };
}

function publicLocalConfig(config){
  return {
    ok: true,
    enabled: Boolean(config.enabled),
    configured: Boolean(config.configured),
    disabledReason: String(config.disabledReason || ''),
    service: 'Local GPU (RTX 5090)',
    workerOrigin: String(config.workerOrigin || ''),
    chunkBytes: config.chunkBytes,
    maxFilesPerRun: config.maxFilesPerRun,
    maxFileBytes: config.maxFileBytes,
    maxServiceDurationSeconds: config.maxServiceDurationSeconds,
    minDurationSeconds: config.minDurationSeconds,
    supportedFormats: [...SUPPORTED_FORMATS],
    historyStored: false
  };
}

function normalizeGroups(value){
  const groups = Array.isArray(value) ? value : String(value || '').split(/[\s,;]+/g);
  return groups
    .map((group) => String(group || '').trim().toLowerCase())
    .filter(Boolean)
    .slice(0, 100);
}

function isAdminClaims(claims, config){
  const groups = normalizeGroups(claims?.['cognito:groups'] || claims?.groups);
  if (groups.some((group) => config.adminGroups.has(group))) return true;
  const email = String(claims?.email || '').trim().toLowerCase();
  return Boolean(
    email &&
    isEmailVerifiedClaim(claims?.email_verified) &&
    config.adminEmails.has(email)
  );
}

function safeFilename(value){
  const raw = String(value || '').trim().replace(/\\/g, '/').split('/').pop() || '';
  const cleaned = raw
    .replace(/[\u0000-\u001f\u007f]+/g, '')
    .replace(/[^a-zA-Z0-9._ -]+/g, '_')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned || cleaned === '.' || cleaned === '..') return '';
  if (cleaned.length <= MAX_FILENAME_CHARS) return cleaned;
  const extensionMatch = cleaned.match(/(\.[a-zA-Z0-9]{1,10})$/);
  const extension = extensionMatch ? extensionMatch[1] : '';
  const stemLength = Math.max(1, MAX_FILENAME_CHARS - extension.length);
  return `${cleaned.slice(0, stemLength).trim()}${extension}`;
}

function formatFromFilename(filename){
  const match = String(filename || '').toLowerCase().match(/\.([a-z0-9]+)$/);
  return match ? match[1] : '';
}

function normalizeContentType(value){
  const contentType = String(value || '').split(';', 1)[0].trim().toLowerCase();
  if (!contentType) return 'application/octet-stream';
  if (contentType.length > 120 || !/^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/.test(contentType)) {
    return '';
  }
  return contentType;
}

function createValidationError(message, statusCode = 400){
  const err = new Error(message);
  err.code = 'LOCAL_TRANSCRIBE_VALIDATION';
  err.statusCode = statusCode;
  return err;
}

function validateTicketInput(body, config){
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw createValidationError('A JSON object is required.');
  }
  const filename = safeFilename(body.filename || body.name);
  if (!filename) throw createValidationError('A valid filename is required.');

  const filenameFormat = formatFromFilename(filename);
  const requestedFormat = String(body.format || filenameFormat).trim().toLowerCase().replace(/^\./, '');
  if (!SUPPORTED_FORMATS.includes(requestedFormat)) {
    throw createValidationError(`Unsupported file format: ${requestedFormat || 'unknown'}.`);
  }
  if (filenameFormat !== requestedFormat) {
    throw createValidationError('File format must match the filename extension.');
  }

  const contentType = normalizeContentType(body.contentType || body.content_type || body.type);
  if (!contentType) throw createValidationError('Invalid media content type.');

  const bytes = Number(body.bytes ?? body.size);
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw createValidationError('File size is required.');
  }
  if (bytes > config.maxFileBytes) {
    throw createValidationError(`File exceeds ${config.maxFileBytes} bytes.`, 413);
  }

  const durationSeconds = Number(body.durationSeconds ?? body.duration_seconds ?? body.duration);
  if (!Number.isFinite(durationSeconds) || durationSeconds < config.minDurationSeconds) {
    throw createValidationError(`File must be at least ${config.minDurationSeconds} seconds long.`);
  }
  if (durationSeconds > config.maxServiceDurationSeconds) {
    throw createValidationError(`File exceeds the ${config.maxServiceDurationSeconds}-second local processing limit.`);
  }

  return {
    filename,
    format: requestedFormat,
    contentType,
    bytes,
    durationSeconds: Number(durationSeconds.toFixed(3))
  };
}

function requestServerOrigin(req){
  const headers = req?.headers || {};
  const forwardedProto = String(headers['x-forwarded-proto'] || '').split(',')[0].trim().toLowerCase();
  const protocol = forwardedProto || (req?.socket?.encrypted ? 'https' : 'http');
  if (!['http', 'https'].includes(protocol)) return '';
  const host = String(headers.host || headers['x-forwarded-host'] || '').split(',')[0].trim();
  if (!host) return '';
  return normalizeHttpOrigin(`${protocol}://${host}`);
}

function ticketRequestOrigin(req){
  const rawSuppliedOrigin = String(req?.headers?.origin || '').trim();
  const suppliedOrigin = normalizeHttpOrigin(rawSuppliedOrigin);
  const serverOrigin = requestServerOrigin(req);
  if (rawSuppliedOrigin && !suppliedOrigin) {
    throw createValidationError('A valid same-origin request Origin is required.', 403);
  }
  if (suppliedOrigin && serverOrigin && suppliedOrigin !== serverOrigin) {
    throw createValidationError('Same-origin request required.', 403);
  }
  const origin = suppliedOrigin || serverOrigin;
  if (!origin) throw createValidationError('A valid request origin is required.');
  return origin;
}

function signTicket(payload, secret){
  const body = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url');
  const signature = crypto.createHmac('sha256', Buffer.from(secret, 'utf8'))
    .update(body, 'utf8')
    .digest('base64url');
  return `${body}.${signature}`;
}

function createTicket({ claims, input, origin, config, nowSeconds, randomBytes }){
  const sub = String(claims?.sub || '').trim();
  if (!sub) throw createValidationError('Verified account is missing a subject.', 401);
  const iat = Math.floor(Number(nowSeconds));
  if (!Number.isFinite(iat) || iat <= 0) throw new Error('Unable to issue a local transcription ticket.');
  const exp = iat + config.ticketTtlSeconds;
  const jobId = randomBytes(16).toString('hex');
  const payload = {
    v: 1,
    type: 'local_transcribe',
    aud: 'local-transcribe-worker',
    sub,
    jobId,
    filename: input.filename,
    format: input.format,
    contentType: input.contentType,
    bytes: input.bytes,
    durationSeconds: input.durationSeconds,
    origin,
    iat,
    exp
  };
  return { jobId, exp, payload, ticket: signTicket(payload, config.sharedSecret) };
}

async function upgradeLegacyEmailAdminSession(req, res, config, authentication, dependencies){
  if (authentication?.source !== 'cookie') return null;
  const cookieClaims = authentication.claims;
  const cookieSub = String(cookieClaims?.sub || '').trim();
  const cookieEmail = String(cookieClaims?.email || '').trim().toLowerCase();
  if (
    !cookieSub ||
    !cookieEmail ||
    isEmailVerifiedClaim(cookieClaims?.email_verified) ||
    !config.adminEmails.has(cookieEmail)
  ) {
    return null;
  }

  const token = getBearerToken(req);
  if (!token) return null;

  try {
    const verifiedClaims = await dependencies.verifyToken(token);
    const verifiedSub = String(verifiedClaims?.sub || '').trim();
    const verifiedEmail = String(verifiedClaims?.email || '').trim().toLowerCase();
    if (
      verifiedSub !== cookieSub ||
      verifiedEmail !== cookieEmail ||
      !isEmailVerifiedClaim(verifiedClaims?.email_verified) ||
      !isAdminClaims(verifiedClaims, config)
    ) {
      return null;
    }
    const session = dependencies.createSession(verifiedClaims);
    if (!session?.cookie) return null;
    res.setHeader('Set-Cookie', session.cookie);
    return verifiedClaims;
  } catch {
    return null;
  }
}

async function requireAdmin(req, res, config, dependencies){
  try {
    const authentication = await dependencies.authenticateRequest(req, { allowBearer: true });
    let claims = authentication?.claims;
    if (!String(claims?.sub || '').trim()) {
      sendJson(res, 401, { ok: false, error: 'Sign in before using local transcription.' });
      return null;
    }
    if (!isAdminClaims(claims, config)) {
      claims = await upgradeLegacyEmailAdminSession(req, res, config, authentication, dependencies);
      if (!claims) {
        sendJson(res, 403, { ok: false, error: 'Administrator access is required for local transcription.' });
        return null;
      }
    }
    return claims;
  } catch (err) {
    if (['COGNITO_ENV_MISSING', 'TOOLS_SESSION_SECRET_MISSING', 'TOOLS_SESSION_SECRET_INVALID'].includes(err?.code)) {
      sendJson(res, 503, { ok: false, error: err.message });
      return null;
    }
    if (err?.code === 'AUTH_ORIGIN_MISMATCH') {
      sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
      return null;
    }
    sendJson(res, 401, { ok: false, error: 'Sign in before using local transcription.' });
    return null;
  }
}

function createHandler(dependencies = {}){
  const authenticateRequest = dependencies.authenticateRequest || authenticateToolsRequest;
  const verifyToken = dependencies.verifyToken || verifyCognitoIdToken;
  const createSession = dependencies.createSession || createSessionFromClaims;
  const readRequestJson = dependencies.readRequestJson || readJson;
  const nowSeconds = dependencies.nowSeconds || (() => Math.floor(Date.now() / 1000));
  const randomBytes = dependencies.randomBytes || crypto.randomBytes;
  const env = dependencies.env || process.env;

  return async function handleLocalTranscribe(req, res, action){
    const normalizedAction = String(action || '').trim().toLowerCase();
    const config = getLocalTranscribeConfig(env);

    if (normalizedAction === 'config') {
      if (req.method !== 'GET') {
        res.setHeader('Allow', 'GET');
        sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
        return;
      }
      const claims = await requireAdmin(req, res, config, { authenticateRequest, verifyToken, createSession });
      if (!claims) return;
      sendJson(res, 200, publicLocalConfig(config));
      return;
    }

    if (normalizedAction === 'ticket') {
      if (req.method !== 'POST') {
        res.setHeader('Allow', 'POST');
        sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
        return;
      }
      const claims = await requireAdmin(req, res, config, { authenticateRequest, verifyToken, createSession });
      if (!claims) return;
      if (!config.configured) {
        sendJson(res, 503, {
          ok: false,
          error: config.disabledReason || 'Local transcription is not configured.'
        });
        return;
      }

      let body;
      try {
        body = await readRequestJson(req);
      } catch (err) {
        sendJson(res, err?.statusCode === 413 ? 413 : 400, {
          ok: false,
          error: err?.statusCode === 413 ? err.message : 'Invalid JSON payload.'
        });
        return;
      }

      try {
        const input = validateTicketInput(body, config);
        const origin = ticketRequestOrigin(req);
        const issued = createTicket({
          claims,
          input,
          origin,
          config,
          nowSeconds: nowSeconds(),
          randomBytes
        });
        sendJson(res, 200, {
          ok: true,
          enabled: true,
          configured: true,
          workerOrigin: config.workerOrigin,
          chunkBytes: config.chunkBytes,
          ticket: issued.ticket,
          expiresAt: issued.exp * 1000,
          job: {
            id: issued.jobId,
            filename: input.filename,
            format: input.format,
            contentType: input.contentType,
            bytes: input.bytes,
            durationSeconds: input.durationSeconds
          }
        });
      } catch (err) {
        sendJson(res, Number(err?.statusCode) || 500, {
          ok: false,
          error: err?.code === 'LOCAL_TRANSCRIBE_VALIDATION'
            ? err.message
            : 'Unable to issue a local transcription ticket.'
        });
      }
      return;
    }

    sendJson(res, 404, { ok: false, error: 'Not Found' });
  };
}

const handleLocalTranscribe = createHandler();

handleLocalTranscribe.createHandler = createHandler;
handleLocalTranscribe._internal = {
  DEFAULT_ADMIN_EMAILS,
  DEFAULT_ADMIN_GROUPS,
  DEFAULT_CHUNK_BYTES,
  DEFAULT_TICKET_TTL_SECONDS,
  DEFAULT_WORKER_ORIGIN,
  MAX_TICKET_TTL_SECONDS,
  MIN_TICKET_TTL_SECONDS,
  SUPPORTED_FORMATS,
  createTicket,
  getLocalTranscribeConfig,
  isAdminClaims,
  isEmailVerifiedClaim,
  isLoopbackHostname,
  normalizeContentType,
  normalizeWorkerOrigin,
  publicLocalConfig,
  requestServerOrigin,
  safeFilename,
  signTicket,
  ticketRequestOrigin,
  upgradeLegacyEmailAdminSession,
  validateTicketInput
};

module.exports = handleLocalTranscribe;
