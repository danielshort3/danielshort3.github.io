/*
  Admin API for a single short link: /api/short-links/<slug>
*/
'use strict';

const { deleteLink, getLink } = require('../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  normalizeSlug
} = require('../_lib/short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  return '';
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
      sendJson(res, err.code === 'DDB_ENV_MISSING' ? 503 : 502, { ok: false, error: 'DynamoDB backend unavailable' });
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
        createdAt: typeof link.createdAt === 'string' ? link.createdAt : '',
        updatedAt: typeof link.updatedAt === 'string' ? link.updatedAt : '',
        clicks: Number.isFinite(Number(link.clicks)) ? Number(link.clicks) : 0
      }
    });
    return;
  }

  if (req.method === 'DELETE') {
    try {
      await deleteLink(slug);
    } catch (err) {
      sendJson(res, err.code === 'DDB_ENV_MISSING' ? 503 : 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, { ok: true });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
