/*
  Tools account API helpers (JSON, auth header parsing).
*/
'use strict';

function sendJson(res, status, body){
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
}

function readJson(req){
  return new Promise((resolve, reject) => {
    if (req.body && typeof req.body === 'object') return resolve(req.body);
    if (typeof req.body === 'string') {
      try { return resolve(JSON.parse(req.body)); } catch (err) { return reject(err); }
    }
    let raw = '';
    req.on('data', chunk => { raw += chunk; });
    req.on('end', () => {
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
  sendJson,
  readJson,
  getBearerToken,
  normalizeToolId,
  normalizeSessionId,
  clampLimit
};

