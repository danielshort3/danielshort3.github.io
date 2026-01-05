/*
  Admin API for managing short links.
  Requires SHORTLINKS_ADMIN_TOKEN and KV env vars.
*/
'use strict';

const { kvGet, kvSet, kvSadd, kvSmembers } = require('../_lib/kv');
const {
  SLUG_SET_KEY,
  linkKey,
  clicksKey,
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeDestination
} = require('../_lib/short-links');

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

  if (req.method === 'GET') {
    let slugs = [];
    try {
      slugs = await kvSmembers(SLUG_SET_KEY);
    } catch (err) {
      sendJson(res, err.code === 'KV_ENV_MISSING' ? 503 : 502, { ok: false, error: 'KV backend unavailable' });
      return;
    }

    const uniqueSlugs = Array.from(new Set(slugs.map(slugValue => normalizeSlug(slugValue)).filter(Boolean)));
    uniqueSlugs.sort((a, b) => a.localeCompare(b));

    const links = [];
    for (const slug of uniqueSlugs) {
      try {
        const raw = await kvGet(linkKey(slug));
        if (!raw) continue;
        const link = JSON.parse(raw);
        const clicks = await getClicksForSlug(slug);
        links.push({
          slug,
          destination: link.destination || '',
          permanent: !!link.permanent,
          createdAt: link.createdAt || '',
          updatedAt: link.updatedAt || '',
          clicks
        });
      } catch {
        continue;
      }
    }

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
    let createdAt = now;
    try {
      const existingRaw = await kvGet(linkKey(slug));
      if (existingRaw) {
        const existing = JSON.parse(existingRaw);
        if (existing && typeof existing.createdAt === 'string' && existing.createdAt.trim()) {
          createdAt = existing.createdAt.trim();
        }
      }
    } catch {}

    const record = { slug, destination, permanent, createdAt, updatedAt: now };

    try {
      await kvSet(linkKey(slug), JSON.stringify(record));
      await kvSadd(SLUG_SET_KEY, slug);
    } catch (err) {
      sendJson(res, err.code === 'KV_ENV_MISSING' ? 503 : 502, { ok: false, error: 'KV backend unavailable' });
      return;
    }

    const clicks = await getClicksForSlug(slug).catch(() => 0);
    sendJson(res, 200, { ok: true, link: { ...record, clicks } });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
