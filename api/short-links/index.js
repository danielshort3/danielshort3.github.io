/*
  Admin API for managing short links.
  Requires SHORTLINKS_ADMIN_TOKEN and DynamoDB env vars.
*/
'use strict';

const { findLinkByLowerSlug, listLinks, upsertLink } = require('../_lib/short-links-store');
const {
  DEFAULT_RANDOM_LENGTH,
  MAX_RANDOM_LENGTH,
  MIN_RANDOM_LENGTH,
  generateRandomSlug,
  getAdminToken,
  isAdminRequest,
  sendJson,
  readJson,
  normalizeSlug,
  normalizeRandomLength,
  normalizeDestination
} = require('../_lib/short-links');

const RANDOM_SLUG_RETRY_LIMIT = 40;

function serializeLink(record, fallbackSlug, fallbackUpdatedAt){
  return {
    slug: typeof record?.slug === 'string' ? record.slug : fallbackSlug,
    destination: typeof record?.destination === 'string' ? record.destination : '',
    permanent: !!record?.permanent,
    expiresAt: Number.isFinite(Number(record?.expiresAt)) ? Number(record.expiresAt) : 0,
    disabled: !!record?.disabled,
    createdAt: typeof record?.createdAt === 'string' ? record.createdAt : fallbackUpdatedAt,
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

function buildMetadata(body){
  return {
    label: body?.label,
    templateId: body?.templateId,
    templateTitle: body?.templateTitle,
    batchId: body?.batchId,
    batchTitle: body?.batchTitle,
    contextType: body?.contextType,
    contextEntryId: body?.contextEntryId,
    contextCompany: body?.contextCompany,
    contextTitle: body?.contextTitle
  };
}

async function resolveRequestedSlug(body){
  const slugMode = typeof body?.slugMode === 'string' ? body.slugMode.trim().toLowerCase() : '';
  const randomLength = normalizeRandomLength(body?.randomLength, DEFAULT_RANDOM_LENGTH);
  const manualSlug = normalizeSlug(body?.slug);

  if (slugMode === 'random') {
    for (let i = 0; i < RANDOM_SLUG_RETRY_LIMIT; i += 1) {
      const candidate = generateRandomSlug(randomLength);
      const conflict = await findLinkByLowerSlug(candidate);
      if (!conflict) {
        return { slug: candidate, randomLength, generated: true };
      }
    }
    const err = new Error('Unable to generate a unique short code right now');
    err.statusCode = 409;
    throw err;
  }

  if (!manualSlug) {
    const err = new Error('Invalid slug (use letters/numbers/-/_ and / for nesting)');
    err.statusCode = 400;
    throw err;
  }

  const conflict = await findLinkByLowerSlug(manualSlug);
  if (conflict && String(conflict.slug || '') !== manualSlug) {
    const err = new Error(`Slug conflicts with existing link "${conflict.slug}"`);
    err.statusCode = 409;
    throw err;
  }

  return { slug: manualSlug, randomLength, generated: false };
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
      .map(item => serializeLink(item, normalizeSlug(item?.slug), typeof item?.updatedAt === 'string' ? item.updatedAt : ''))
      .filter(link => link.slug && link.destination)
      .sort((a, b) => a.slug.localeCompare(b.slug, undefined, { sensitivity: 'base' }) || a.slug.localeCompare(b.slug));

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

    let requestedSlug;
    try {
      requestedSlug = await resolveRequestedSlug(body || {});
    } catch (err) {
      sendJson(res, err.statusCode || 400, { ok: false, error: err.message || 'Invalid slug' });
      return;
    }

    const slug = requestedSlug.slug;
    const destination = normalizeDestination(body.destination, { absolutizeInternalPath: true });
    const permanent = !!body.permanent;
    const hasExpiresAt = !!(body && Object.prototype.hasOwnProperty.call(body, 'expiresAt'));
    let expiresAt = 0;

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
        updatedAt: now,
        metadata: buildMetadata(body)
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
      link: serializeLink(record, slug, now),
      generated: requestedSlug.generated,
      randomLength: requestedSlug.generated ? requestedSlug.randomLength : 0,
      randomLengthRange: { min: MIN_RANDOM_LENGTH, max: MAX_RANDOM_LENGTH }
    });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
