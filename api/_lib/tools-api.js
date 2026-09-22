/*
  Tools account API helpers (JSON, auth header parsing).
*/
'use strict';

const MAX_JSON_BODY_BYTES = 512 * 1024;
const KNOWN_TOOL_IDS = new Set([
  'background-remover',
  'campaign-creative-tracker',
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

function tooLarge(maxBytes){
  const err = new Error(`JSON request body exceeds ${maxBytes} bytes.`);
  err.code = 'BODY_TOO_LARGE';
  err.statusCode = 413;
  return err;
}

async function readJson(req, options = {}){
  const configured = Number(options.maxBytes);
  const maxBytes = Number.isFinite(configured) && configured > 0
    ? Math.max(1, Math.floor(configured)) : MAX_JSON_BODY_BYTES;
  const declared = Number(req?.headers?.['content-length']);
  if (Number.isFinite(declared) && declared > maxBytes) throw tooLarge(maxBytes);

  if (req.body !== undefined) {
    if (Buffer.isBuffer(req.body) || typeof req.body === 'string') {
      if (Buffer.byteLength(req.body, 'utf8') > maxBytes) throw tooLarge(maxBytes);
      const raw = req.body.toString();
      return raw.trim() ? JSON.parse(raw) : {};
    }
    const raw = JSON.stringify(req.body);
    if (typeof raw !== 'string') throw new TypeError('Invalid JSON body');
    if (Buffer.byteLength(raw, 'utf8') > maxBytes) throw tooLarge(maxBytes);
    return req.body;
  }

  return new Promise((resolve, reject) => {
    let chunks = [];
    let size = 0;
    let settled = false;
    const fail = (err) => {
      if (settled) return;
      settled = true;
      chunks = [];
      reject(err);
    };
    req.on('data', chunk => {
      if (settled) return;
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk), 'utf8');
      size += bytes.length;
      if (size > maxBytes) { fail(tooLarge(maxBytes)); return; }
      chunks.push(bytes);
    });
    req.on('end', () => {
      if (settled) return;
      const raw = Buffer.concat(chunks).toString('utf8');
      chunks = [];
      try {
        const body = raw.trim() ? JSON.parse(raw) : {};
        settled = true;
        resolve(body);
      } catch (err) { fail(err); }
    });
    req.on('error', fail);
    req.on('aborted', () => fail(new Error('Request aborted')));
  });
}

function sendJson(res, status, body){
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
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
