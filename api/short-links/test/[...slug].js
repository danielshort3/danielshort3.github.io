/*
  Admin API for testing a short link: /api/short-links/test/<slug>
  - Requires SHORTLINKS_ADMIN_TOKEN.
  - Does not increment clicks or record click history.
*/
'use strict';

const { getLink } = require('../../_lib/short-links-store');
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
    'user-agent': 'Mozilla/5.0 (compatible; DanielShortShortlinksTest/1.0; +https://danielshort.me)',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
  };

  const startedAt = Date.now();
  try {
    const resp = await fetchWithTimeout(url, {
      method: 'HEAD',
      redirect: 'follow',
      cache: 'no-store',
      headers
    }, timeoutMs);

    if (resp.status === 405 || resp.status === 501) {
      const err = new Error('HEAD_NOT_SUPPORTED');
      err.code = 'HEAD_NOT_SUPPORTED';
      throw err;
    }

    return {
      ok: resp.status >= 200 && resp.status < 400,
      method: 'HEAD',
      status: resp.status,
      url: resp.url || '',
      redirected: !!resp.redirected,
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
    const resp = await fetchWithTimeout(url, {
      method: 'GET',
      redirect: 'follow',
      cache: 'no-store',
      headers: Object.assign({}, headers, { range: 'bytes=0-0' })
    }, timeoutMs);

    try {
      if (resp.body && typeof resp.body.cancel === 'function') {
        await resp.body.cancel();
      }
    } catch {}

    return {
      ok: resp.status >= 200 && resp.status < 400,
      method: 'GET',
      status: resp.status,
      url: resp.url || '',
      redirected: !!resp.redirected,
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

module.exports = async (req, res) => {
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
    link = await getLink(slug);
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
};

