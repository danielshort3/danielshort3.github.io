/*
  Admin API for testing a short link: /api/short-links/test/<slug>
  - Requires SHORTLINKS_ADMIN_TOKEN.
  - Does not increment clicks or record click history.
*/
'use strict';

const dns = require('dns').promises;
const net = require('net');
const { getLinkWithLegacyFallback } = require('../../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('../../_lib/short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/short-links\/test\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : '';
  } catch {
    return '';
  }
}

function resolveDestination(destination, base){
  const raw = typeof destination === 'string' ? destination.trim() : '';
  if (!raw) return '';
  try {
    return new URL(raw, base).toString();
  } catch {
    return raw;
  }
}

function getErrorMessage(err){
  if (!err) return '';
  if (typeof err === 'string') return err;
  if (err.name === 'AbortError') return 'Request timed out';
  if (err.message) return String(err.message);
  return '';
}

function isPrivateIpv4(address){
  const parts = String(address || '').split('.').map(part => Number(part));
  if (parts.length !== 4 || parts.some(part => !Number.isInteger(part) || part < 0 || part > 255)) return true;
  const [a, b] = parts;
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a >= 224) return true;
  return false;
}

function isPrivateIpv6(address){
  const normalized = String(address || '').toLowerCase();
  if (!normalized) return true;
  if (normalized === '::1' || normalized === '::') return true;
  if (normalized.startsWith('fc') || normalized.startsWith('fd')) return true;
  if (normalized.startsWith('fe8') || normalized.startsWith('fe9') || normalized.startsWith('fea') || normalized.startsWith('feb')) return true;
  if (normalized.startsWith('::ffff:')) {
    const mapped = normalized.slice('::ffff:'.length);
    if (net.isIP(mapped) === 4) return isPrivateIpv4(mapped);
  }
  return false;
}

function isBlockedAddress(address){
  const family = net.isIP(address);
  if (family === 4) return isPrivateIpv4(address);
  if (family === 6) return isPrivateIpv6(address);
  return true;
}

function isLocalHostname(hostname){
  const host = String(hostname || '').trim().toLowerCase().replace(/\.$/, '');
  return !host || host === 'localhost' || host.endsWith('.localhost') || host === 'local' || host.endsWith('.local');
}

async function assertSafeDestinationUrl(rawUrl){
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    const err = new Error('Invalid destination URL');
    err.code = 'INVALID_DESTINATION_URL';
    throw err;
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    const err = new Error('Only http and https destinations can be tested');
    err.code = 'UNSAFE_PROTOCOL';
    throw err;
  }

  if (parsed.username || parsed.password) {
    const err = new Error('Destination URLs with embedded credentials cannot be tested');
    err.code = 'UNSAFE_CREDENTIALS';
    throw err;
  }

  const hostname = String(parsed.hostname || '').replace(/^\[|\]$/g, '');

  if (isLocalHostname(hostname)) {
    const err = new Error('Local destinations cannot be tested');
    err.code = 'LOCAL_DESTINATION';
    throw err;
  }

  if (net.isIP(hostname)) {
    if (isBlockedAddress(hostname)) {
      const err = new Error('Private or local network destinations cannot be tested');
      err.code = 'PRIVATE_DESTINATION';
      throw err;
    }
    return parsed.toString();
  }

  let records;
  try {
    records = await dns.lookup(hostname, { all: true, verbatim: true });
  } catch {
    const err = new Error('Destination host could not be resolved');
    err.code = 'DNS_LOOKUP_FAILED';
    throw err;
  }

  if (!Array.isArray(records) || records.length === 0) {
    const err = new Error('Destination host could not be resolved');
    err.code = 'DNS_LOOKUP_EMPTY';
    throw err;
  }

  if (records.some(record => isBlockedAddress(record && record.address))) {
    const err = new Error('Destination resolves to a private or local network address');
    err.code = 'PRIVATE_DESTINATION';
    throw err;
  }

  return parsed.toString();
}

function getManualRedirectUrl(resp, baseUrl){
  const status = Number(resp && resp.status);
  if (status < 300 || status >= 400 || !resp || !resp.headers || typeof resp.headers.get !== 'function') return '';
  const location = resp.headers.get('location');
  if (!location) return '';
  try {
    return new URL(location, baseUrl).toString();
  } catch {
    return '';
  }
}

async function fetchWithTimeout(url, options, timeoutMs){
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, Object.assign({}, options, { signal: controller.signal }));
  } finally {
    clearTimeout(timeout);
  }
}

async function checkDestination(url){
  const timeoutMs = 5000;
  const headers = {
    'user-agent': 'Mozilla/5.0 (compatible; DanielShortShortlinksTest/1.0; +https://www.danielshort.me)',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
  };

  let safeUrl;
  try {
    safeUrl = await assertSafeDestinationUrl(url);
  } catch (err) {
    return {
      ok: false,
      method: 'VALIDATE',
      status: 0,
      url: '',
      redirected: false,
      ms: 0,
      error: getErrorMessage(err) || 'Unsafe destination'
    };
  }

  const startedAt = Date.now();
  try {
    const resp = await fetchWithTimeout(safeUrl, {
      method: 'HEAD',
      redirect: 'manual',
      cache: 'no-store',
      headers
    }, timeoutMs);

    if (resp.status === 405 || resp.status === 501) {
      const err = new Error('HEAD_NOT_SUPPORTED');
      err.code = 'HEAD_NOT_SUPPORTED';
      throw err;
    }

    const redirectUrl = getManualRedirectUrl(resp, safeUrl);
    if (redirectUrl) {
      try {
        await assertSafeDestinationUrl(redirectUrl);
      } catch (err) {
        return {
          ok: false,
          method: 'HEAD',
          status: resp.status,
          url: redirectUrl,
          redirected: true,
          ms: Date.now() - startedAt,
          error: getErrorMessage(err) || 'Redirect target is unsafe'
        };
      }
    }

    return {
      ok: resp.status >= 200 && resp.status < 400,
      method: 'HEAD',
      status: resp.status,
      url: redirectUrl || resp.url || safeUrl,
      redirected: !!redirectUrl,
      ms: Date.now() - startedAt
    };
  } catch (err) {
    if (err && err.code !== 'HEAD_NOT_SUPPORTED' && err.message !== 'HEAD_NOT_SUPPORTED') {
      return {
        ok: false,
        method: 'HEAD',
        status: 0,
        url: '',
        redirected: false,
        ms: Date.now() - startedAt,
        error: getErrorMessage(err) || 'Request failed'
      };
    }
  }

  const startedGet = Date.now();
  try {
    const resp = await fetchWithTimeout(safeUrl, {
      method: 'GET',
      redirect: 'manual',
      cache: 'no-store',
      headers: Object.assign({}, headers, { range: 'bytes=0-0' })
    }, timeoutMs);

    try {
      if (resp.body && typeof resp.body.cancel === 'function') {
        await resp.body.cancel();
      }
    } catch {}

    const redirectUrl = getManualRedirectUrl(resp, safeUrl);
    if (redirectUrl) {
      try {
        await assertSafeDestinationUrl(redirectUrl);
      } catch (err) {
        return {
          ok: false,
          method: 'GET',
          status: resp.status,
          url: redirectUrl,
          redirected: true,
          ms: Date.now() - startedGet,
          error: getErrorMessage(err) || 'Redirect target is unsafe'
        };
      }
    }

    return {
      ok: resp.status >= 200 && resp.status < 400,
      method: 'GET',
      status: resp.status,
      url: redirectUrl || resp.url || safeUrl,
      redirected: !!redirectUrl,
      ms: Date.now() - startedGet
    };
  } catch (err) {
    return {
      ok: false,
      method: 'GET',
      status: 0,
      url: '',
      redirected: false,
      ms: Date.now() - startedGet,
      error: getErrorMessage(err) || 'Request failed'
    };
  }
}

async function handler(req, res){
  const adminToken = getAdminToken();
  if (!adminToken) {
    sendJson(res, 503, { ok: false, error: 'SHORTLINKS_ADMIN_TOKEN is not configured' });
    return;
  }
  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const slug = normalizeSlug(getSlugFromRequest(req));
  if (!slug) {
    sendJson(res, 400, { ok: false, error: 'Invalid slug' });
    return;
  }

  let link;
  try {
    link = await getLinkWithLegacyFallback(slug);
  } catch (err) {
    if (err && err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
    return;
  }

  if (!link || link.disabled) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
  if (expiresAt && Math.floor(Date.now() / 1000) >= expiresAt) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const base = getRequestBaseUrl(req);
  const destination = resolveDestination(link.destination, base);
  if (!destination) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const statusCode = link.permanent ? 301 : 302;
  const check = await checkDestination(destination);

  sendJson(res, 200, {
    ok: true,
    slug,
    redirect: {
      statusCode,
      destination
    },
    check
  });
}

module.exports = handler;
module.exports._internal = {
  assertSafeDestinationUrl,
  isBlockedAddress,
  isLocalHostname,
  isPrivateIpv4,
  isPrivateIpv6,
  getManualRedirectUrl
};
