/*
  Admin API for a single short link: /api/short-links/<slug>
*/
'use strict';

const { deleteLink, getLinkWithLegacyFallback, setLinkDisabled } = require('../_lib/short-links-store');
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

function serializeLink(record, fallbackSlug, fallbackUpdatedAt){
  return {
    slug: typeof record?.slug === 'string' ? record.slug : fallbackSlug,
    destination: typeof record?.destination === 'string' ? record.destination : '',
    permanent: !!record?.permanent,
    expiresAt: Number.isFinite(Number(record?.expiresAt)) ? Number(record.expiresAt) : 0,
    disabled: !!record?.disabled,
    createdAt: typeof record?.createdAt === 'string' ? record.createdAt : '',
    updatedAt: typeof record?.updatedAt === 'string' ? record.updatedAt : fallbackUpdatedAt,
    clicks: Number.isFinite(Number(record?.clicks)) ? Number(record.clicks) : 0,
    label: typeof record?.label === 'string' ? record.label : '',
    templateId: typeof record?.templateId === 'string' ? record.templateId : '',
    templateTitle: typeof record?.templateTitle === 'string' ? record.templateTitle : '',
    batchId: typeof record?.batchId === 'string' ? record.batchId : '',
    batchTitle: typeof record?.batchTitle === 'string' ? record.batchTitle : '',
    contextType: typeof record?.contextType === 'string' ? record.contextType : '',
    contextEntryId: typeof record?.contextEntryId === 'string' ? record.contextEntryId : '',
    contextCompany: typeof record?.contextCompany === 'string' ? record.contextCompany : '',
    contextTitle: typeof record?.contextTitle === 'string' ? record.contextTitle : ''
  };
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
