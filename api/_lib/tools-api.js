/*
  Tools account API helpers (JSON, auth header parsing).
*/
'use strict';

const { readJson } = require('./json-body');

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
