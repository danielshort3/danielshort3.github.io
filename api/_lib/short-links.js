/*
  Short links helpers (validation, keys, auth).
*/
'use strict';

const SHORTLINKS_PREFIX = 'shortlinks';
const SLUG_SET_KEY = `${SHORTLINKS_PREFIX}:slugs`;

function linkKey(slug){
  return `${SHORTLINKS_PREFIX}:link:${slug}`;
}

function clicksKey(slug){
  return `${SHORTLINKS_PREFIX}:clicks:${slug}`;
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
  return provided.length > 0 && provided === configured;
}

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

function normalizeSlug(slug){
  const raw = String(slug || '').trim();
  const trimmed = raw.replace(/^\/+|\/+$/g, '');
  const normalized = trimmed.toLowerCase();
  if (!normalized) return '';
  if (normalized.length > 128) return '';
  if (normalized.includes('..')) return '';
  const ok = /^[a-z0-9][a-z0-9-_]*(?:\/[a-z0-9][a-z0-9-_]*)*$/.test(normalized);
  return ok ? normalized : '';
}

function normalizeDestination(destination){
  const raw = String(destination || '').trim();
  if (!raw) return '';
  if (raw.length > 2048) return '';
  if (raw.startsWith('/')) return raw;
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
  SLUG_SET_KEY,
  linkKey,
  clicksKey,
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeDestination,
  getRequestBaseUrl
};

