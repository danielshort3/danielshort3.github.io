/*
  Admin API for a single short link: /api/short-links/<slug>
*/
'use strict';

const { kvGet, kvDel, kvSrem } = require('../_lib/kv');
const {
  SLUG_SET_KEY,
  linkKey,
  clicksKey,
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

async function getClicksForSlug(slug){
  const raw = await kvGet(clicksKey(slug));
  const parsedClicks = raw == null ? 0 : parseInt(String(raw), 10);
  return Number.isFinite(parsedClicks) ? parsedClicks : 0;
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
    let raw;
    try {
      raw = await kvGet(linkKey(slug));
    } catch (err) {
      sendJson(res, err.code === 'KV_ENV_MISSING' ? 503 : 502, { ok: false, error: 'KV backend unavailable' });
      return;
    }

    if (!raw) {
      sendJson(res, 404, { ok: false, error: 'Not Found' });
      return;
    }

    let link;
    try {
      link = JSON.parse(raw);
    } catch {
      sendJson(res, 500, { ok: false, error: 'Invalid short link record' });
      return;
    }

    const clicks = await getClicksForSlug(slug).catch(() => 0);
    sendJson(res, 200, {
      ok: true,
      link: {
        slug,
        destination: link.destination || '',
        permanent: !!link.permanent,
        createdAt: link.createdAt || '',
        updatedAt: link.updatedAt || '',
        clicks
      }
    });
    return;
  }

  if (req.method === 'DELETE') {
    try {
      await kvDel(linkKey(slug), clicksKey(slug));
      await kvSrem(SLUG_SET_KEY, slug);
    } catch (err) {
      sendJson(res, err.code === 'KV_ENV_MISSING' ? 503 : 502, { ok: false, error: 'KV backend unavailable' });
      return;
    }

    sendJson(res, 200, { ok: true });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
