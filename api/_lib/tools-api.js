/*
  Tools account API helpers (JSON, auth header parsing).
*/
'use strict';

const MAX_JSON_BODY_BYTES = 512 * 1024;
const KNOWN_TOOL_IDS = new Set([
  'background-remover',
  'ga4-utm-performance',
  'image-optimizer',
  'job-application-tracker',
  'nbsp-cleaner',
  'oxford-comma-checker',
  'point-of-view-checker',
  'qr-code-generator',
  'screen-recorder',
  'short-links',
  'text-compare',
  'transcribe',
  'utm-batch-builder',
  'whisper-transcribe-monitor',
  'word-frequency'
]);

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

function getBearerToken(req){
  const auth = (req.headers && req.headers.authorization) ? String(req.headers.authorization) : '';
  const headerToken = (req.headers && (req.headers['x-tools-token'] || req.headers['x-user-token']))
    ? String(req.headers['x-tools-token'] || req.headers['x-user-token'])
    : '';

  let provided = headerToken.trim();
  if (!provided && auth.toLowerCase().startsWith('bearer ')) {
    provided = auth.slice(7).trim();
  }
  return provided;
}

function normalizeToolId(value){
  const toolId = String(value || '').trim();
  if (!toolId) return '';
  if (toolId.length > 80) return '';
  const ok = /^[a-z0-9][a-z0-9-]*$/.test(toolId);
  return ok ? toolId : '';
}

function normalizeKnownToolId(value){
  const toolId = normalizeToolId(value);
  return toolId && KNOWN_TOOL_IDS.has(toolId) ? toolId : '';
}

function normalizeSessionId(value){
  const sessionId = String(value || '').trim();
  if (!sessionId) return '';
  if (sessionId.length > 128) return '';
  const ok = /^[A-Za-z0-9_-]+$/.test(sessionId);
  return ok ? sessionId : '';
}

function clampLimit(value, fallback, max){
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return fallback;
  return Math.min(Math.floor(numeric), max);
}

module.exports = {
  MAX_JSON_BODY_BYTES,
  KNOWN_TOOL_IDS,
  sendJson,
  readJson,
  getBearerToken,
  normalizeToolId,
  normalizeKnownToolId,
  normalizeSessionId,
  clampLimit
};
