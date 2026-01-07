/*
  Admin API for managing short links.
  Requires SHORTLINKS_ADMIN_TOKEN and DynamoDB env vars.
*/
'use strict';

const { listLinks, upsertLink } = require('../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeDestination
} = require('../_lib/short-links');

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

  if (req.method === 'GET') {
    let items = [];
    try {
      items = await listLinks();
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    const links = items
      .map(item => ({
        slug: normalizeSlug(item.slug),
        destination: typeof item.destination === 'string' ? item.destination : '',
        permanent: !!item.permanent,
        disabled: !!item.disabled,
        createdAt: typeof item.createdAt === 'string' ? item.createdAt : '',
        updatedAt: typeof item.updatedAt === 'string' ? item.updatedAt : '',
        clicks: Number.isFinite(Number(item.clicks)) ? Number(item.clicks) : 0
      }))
      .filter(link => link.slug && link.destination)
      .sort((a, b) => a.slug.localeCompare(b.slug));

    sendJson(res, 200, { ok: true, basePath: 'go', links });
    return;
  }

  if (req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const slug = normalizeSlug(body.slug);
    const destination = normalizeDestination(body.destination);
    const permanent = !!body.permanent;

    if (!slug) {
      sendJson(res, 400, { ok: false, error: 'Invalid slug (use letters/numbers/-/_ and / for nesting)' });
      return;
    }
    if (!destination) {
      sendJson(res, 400, { ok: false, error: 'Invalid destination (must start with / or http(s)://)' });
      return;
    }

    const now = new Date().toISOString();

    let record;
    try {
      record = await upsertLink({ slug, destination, permanent, updatedAt: now });
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, {
      ok: true,
      link: {
        slug,
        destination,
        permanent,
        disabled: record ? !!record.disabled : false,
        createdAt: record && record.createdAt ? record.createdAt : now,
        updatedAt: now,
        clicks: record && Number.isFinite(Number(record.clicks)) ? Number(record.clicks) : 0
      }
    });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
