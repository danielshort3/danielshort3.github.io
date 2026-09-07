/*
  Short-link detail, health, click history, sets, and tests share one Vercel Function.
*/
'use strict';

const { serializeLink, buildLinkPatch } = require('../_lib/short-links-management');

const { deleteLink, getLinkWithLegacyFallback, updateLink } = require('../_lib/short-links-store');
const {
  authorizeAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('../_lib/short-links');

function decodeRequestValue(value){
  return decodeURIComponent(String(value || ''));
}

function getSlugFromRequest(req){
  // Derive routing from the URL before consulting Vercel's path parameters so a
  // user-supplied ?slug= or ?setId= cannot change the selected endpoint.
  const url = new URL(req.url || '', getRequestBaseUrl(req));
  const match = url.pathname.match(/\/api\/short-links\/(.+)$/);
  if (match) return decodeRequestValue(match[1]).replace(/\/+$/, '');
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return decodeRequestValue(querySlug.join('/'));
  if (typeof querySlug === 'string') return decodeRequestValue(querySlug);
  return '';
}


module.exports = async (req, res) => {
  let requestSlug;
  try {
    requestSlug = getSlugFromRequest(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid short-link route.' });
    return;
  }
  const parts = requestSlug.split('/').filter(Boolean);
  req.query = { ...req.query, slug: parts };

  if (requestSlug === 'health') {
    delete req.query.slug;
    return require('../_lib/short-links-endpoints/health')(req, res);
  }
  if (requestSlug === 'sets' || requestSlug.startsWith('sets/')) {
    delete req.query.slug;
    req.query.setId = parts.length > 1 ? parts.slice(1) : ['__collection__'];
    return require('../_lib/short-links-endpoints/sets')(req, res);
  }
  if (requestSlug.startsWith('clicks/')) {
    req.query.slug = parts.slice(1);
    return require('../_lib/short-links-endpoints/clicks')(req, res);
  }
  if (requestSlug.startsWith('test/')) {
    const testSlug = requestSlug.slice('test/'.length);
    req.query.slug = parts.slice(1);
    const testHandler = require('../_lib/short-links-test');
    await testHandler(req, res, { slug: testSlug });
    return;
  }

  if (!await authorizeAdminRequest(req, res)) return;

  const slug = normalizeSlug(requestSlug);
  if (!slug) {
    sendJson(res, 400, { ok: false, error: 'Invalid slug' });
    return;
  }

  if (req.method === 'GET') {
    let link;
    try {
      link = await getLinkWithLegacyFallback(slug);
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    if (!link) {
      sendJson(res, 404, { ok: false, error: 'Not Found' });
      return;
    }

    sendJson(res, 200, { ok: true, link: serializeLink(link, slug, '') });
    return;
  }

  if (req.method === 'PATCH') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const now = new Date().toISOString();

    let updated;
    try {
      const current = await getLinkWithLegacyFallback(slug);
      if (!current) {
        sendJson(res, 404, { ok: false, error: 'Not Found' });
        return;
      }
      const patch = buildLinkPatch(body, current);
      updated = await updateLink({
        slug: current.slug, patch, updatedAt: now, expectedUpdatedAt: current.updatedAt
      });
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      if (err.name === 'ConditionalCheckFailedException') {
        sendJson(res, 409, { ok: false, error: 'This link changed. Reload it and try again.' });
        return;
      }
      if (err.statusCode === 400) {
        sendJson(res, 400, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, {
      ok: true,
      link: serializeLink(updated, slug, now)
    });
    return;
  }

  if (req.method === 'DELETE') {
    try {
      await deleteLink(slug);
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, { ok: true });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, PATCH, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
