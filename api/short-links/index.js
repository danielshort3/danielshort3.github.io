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
        expiresAt: Number.isFinite(Number(item.expiresAt)) ? Number(item.expiresAt) : 0,
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
    const hasExpiresAt = !!(body && Object.prototype.hasOwnProperty.call(body, 'expiresAt'));
    let expiresAt = 0;

    if (!slug) {
      sendJson(res, 400, { ok: false, error: 'Invalid slug (use letters/numbers/-/_ and / for nesting)' });
      return;
    }
    if (!destination) {
      sendJson(res, 400, { ok: false, error: 'Invalid destination (must start with / or http(s)://)' });
      return;
    }

    if (hasExpiresAt) {
      const numericExpiresAt = Number(body.expiresAt);
      if (!Number.isFinite(numericExpiresAt) || numericExpiresAt < 0) {
        sendJson(res, 400, { ok: false, error: 'Invalid expiresAt (expected a Unix timestamp in seconds)' });
        return;
      }
      expiresAt = Math.floor(numericExpiresAt);
      const nowSeconds = Math.floor(Date.now() / 1000);
      if (expiresAt && expiresAt <= nowSeconds) {
        sendJson(res, 400, { ok: false, error: 'Invalid expiresAt (must be in the future)' });
        return;
      }
      if (permanent && expiresAt) {
        sendJson(res, 400, { ok: false, error: 'Permanent links cannot have expiresAt' });
        return;
      }
    }

    const now = new Date().toISOString();

    let record;
    try {
      record = await upsertLink({
        slug,
        destination,
        permanent,
        expiresAt: hasExpiresAt ? expiresAt : undefined,
        updatedAt: now
      });
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
        expiresAt: record && Number.isFinite(Number(record.expiresAt)) ? Number(record.expiresAt) : 0,
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
