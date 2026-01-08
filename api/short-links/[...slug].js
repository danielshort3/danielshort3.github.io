/*
  Admin API for a single short link: /api/short-links/<slug>
*/
'use strict';

const { deleteLink, getLink, setLinkDisabled } = require('../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('../_lib/short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/short-links\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : '';
  } catch {
    return '';
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

  const slug = normalizeSlug(getSlugFromRequest(req));
  if (!slug) {
    sendJson(res, 400, { ok: false, error: 'Invalid slug' });
    return;
  }

  if (req.method === 'GET') {
    let link;
    try {
      link = await getLink(slug);
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

    sendJson(res, 200, {
      ok: true,
      link: {
        slug,
        destination: typeof link.destination === 'string' ? link.destination : '',
        permanent: !!link.permanent,
        expiresAt: Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0,
        disabled: !!link.disabled,
        createdAt: typeof link.createdAt === 'string' ? link.createdAt : '',
        updatedAt: typeof link.updatedAt === 'string' ? link.updatedAt : '',
        clicks: Number.isFinite(Number(link.clicks)) ? Number(link.clicks) : 0
      }
    });
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

    if (!body || typeof body.disabled !== 'boolean') {
      sendJson(res, 400, { ok: false, error: 'Invalid payload (expected { disabled: true|false })' });
      return;
    }

    const now = new Date().toISOString();

    let updated;
    try {
      updated = await setLinkDisabled({ slug, disabled: body.disabled, updatedAt: now });
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      if (err.name === 'ConditionalCheckFailedException') {
        sendJson(res, 404, { ok: false, error: 'Not Found' });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, {
      ok: true,
      link: {
        slug,
        destination: typeof updated.destination === 'string' ? updated.destination : '',
        permanent: !!updated.permanent,
        expiresAt: Number.isFinite(Number(updated.expiresAt)) ? Number(updated.expiresAt) : 0,
        disabled: !!updated.disabled,
        createdAt: typeof updated.createdAt === 'string' ? updated.createdAt : '',
        updatedAt: typeof updated.updatedAt === 'string' ? updated.updatedAt : now,
        clicks: Number.isFinite(Number(updated.clicks)) ? Number(updated.clicks) : 0
      }
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
