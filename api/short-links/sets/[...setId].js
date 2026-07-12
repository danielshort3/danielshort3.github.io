/*
  Admin API for a short-link set template or template generation.
*/
'use strict';

const {
  deleteSetTemplate,
  getSetTemplate,
  isSlugConflictError,
  listLinks,
  listSetTemplates,
  listSetTemplatesPage,
  saveGeneratedBatch,
  saveSetTemplate,
  upsertLink
} = require('../../_lib/short-links-store');
const {
  buildBatchRecordKey,
  createBatchId,
  generateRandomSlug,
  getAdminToken,
  getRequestBaseUrl,
  isAdminRequest,
  normalizeDestination,
  normalizeRandomLength,
  normalizeSetId,
  normalizeSlugLower,
  readJson,
  sendJson
} = require('../../_lib/short-links');
const {
  buildBatchTitle,
  buildSetTemplateRecord,
  normalizeGenerationContext,
  resolveBatchTiming,
  serializeSetTemplate
} = require('../../_lib/short-links-sets');

const RANDOM_SLUG_RETRY_LIMIT = 40;
const COLLECTION_SENTINEL = '__collection__';

function getRouteParts(req){
  const direct = req.query && req.query.setId;
  if (Array.isArray(direct)) return direct.map(part => String(part || '').trim()).filter(Boolean);
  if (typeof direct === 'string' && direct.trim()) return [direct.trim()];
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/short-links\/sets\/(.+)$/);
    if (!match) return [];
    return match[1]
      .split('/')
      .map(part => decodeURIComponent(part || '').trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

function buildShortUrl(slug, req){
  const base = getRequestBaseUrl(req);
  try {
    const url = new URL(base);
    const host = String(url.hostname || '').toLowerCase();
    if (host === 'localhost' || host === '127.0.0.1' || host.endsWith('.vercel.app')) {
      return `${url.origin}/go/${slug}`;
    }
  } catch {}
  return `https://dshort.me/${slug}`;
}

function isCollectionRoute(parts){
  return Array.isArray(parts)
    && parts.length === 1
    && String(parts[0] || '').trim().toLowerCase() === COLLECTION_SENTINEL;
}

function getSearchParams(req){
  try {
    return new URL(req.url, getRequestBaseUrl(req)).searchParams;
  } catch {
    return new URLSearchParams();
  }
}

function normalizePageLimit(value){
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 50;
  return Math.max(1, Math.min(200, Math.floor(numeric)));
}

function serializeGeneratedLink(record, req){
  const slug = typeof record?.slug === 'string' ? record.slug : '';
  return {
    label: typeof record?.label === 'string' ? record.label : '',
    slug,
    shortUrl: slug ? buildShortUrl(slug, req) : '',
    destination: typeof record?.destination === 'string' ? record.destination : ''
  };
}

function createUniqueSlug(length, usedLowerSlugs){
  const safeLength = normalizeRandomLength(length);
  for (let i = 0; i < RANDOM_SLUG_RETRY_LIMIT; i += 1) {
    const candidate = generateRandomSlug(safeLength);
    const lower = normalizeSlugLower(candidate);
    if (!lower || usedLowerSlugs.has(lower)) continue;
    usedLowerSlugs.add(lower);
    return candidate;
  }
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

  const parts = getRouteParts(req);
  if (isCollectionRoute(parts)) {
    if (req.method === 'GET') {
      try {
        const params = getSearchParams(req);
        if (String(params.get('pageMode') || '').trim().toLowerCase() === 'storage') {
          const limit = normalizePageLimit(params.get('limit'));
          const page = await listSetTemplatesPage({
            limit,
            cursor: String(params.get('cursor') || '').trim()
          });
          sendJson(res, 200, {
            ok: true,
            sets: page.items.map(serializeSetTemplate),
            pagination: {
              mode: 'storage',
              total: null,
              limit,
              cursor: String(params.get('cursor') || '').trim(),
              nextCursor: page.nextCursor
            }
          });
          return;
        }
        const items = await listSetTemplates();
        sendJson(res, 200, {
          ok: true,
          sets: items.map(serializeSetTemplate)
        });
        return;
      } catch (err) {
        if (err.code === 'DDB_ENV_MISSING') {
          sendJson(res, 503, { ok: false, error: err.message });
          return;
        }
        if (err.code === 'INVALID_CURSOR') {
          sendJson(res, 400, { ok: false, error: err.message });
          return;
        }
        sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
        return;
      }
    }

    if (req.method === 'POST') {
      let body;
      try {
        body = await readJson(req);
      } catch {
        sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
        return;
      }

      const record = buildSetTemplateRecord(body, null);
      if (!record.title) {
        sendJson(res, 400, { ok: false, error: 'Template title is required' });
        return;
      }
      if (!Array.isArray(record.entries) || record.entries.length === 0) {
        sendJson(res, 400, { ok: false, error: 'Add at least one valid URL row before saving the set' });
        return;
      }

      try {
        await saveSetTemplate(record);
      } catch (err) {
        if (err.code === 'DDB_ENV_MISSING') {
          sendJson(res, 503, { ok: false, error: err.message });
          return;
        }
        sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
        return;
      }

      sendJson(res, 200, { ok: true, set: serializeSetTemplate(record) });
      return;
    }

    res.statusCode = 405;
    res.setHeader('Allow', 'GET, POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const isGenerateRoute = parts.length >= 2 && parts[parts.length - 1].toLowerCase() === 'generate';
  const setId = normalizeSetId(isGenerateRoute ? parts.slice(0, -1).join('-') : parts.join('-'));
  if (!setId) {
    sendJson(res, 400, { ok: false, error: 'Invalid set ID' });
    return;
  }

  let existing;
  try {
    existing = await getSetTemplate(setId);
  } catch (err) {
    if (err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
    return;
  }

  if (!existing) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  if (!isGenerateRoute && req.method === 'GET') {
    sendJson(res, 200, { ok: true, set: serializeSetTemplate(existing) });
    return;
  }

  if (!isGenerateRoute && req.method === 'PATCH') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const next = buildSetTemplateRecord(body, existing);
    if (!next.title) {
      sendJson(res, 400, { ok: false, error: 'Template title is required' });
      return;
    }
    if (!Array.isArray(next.entries) || next.entries.length === 0) {
      sendJson(res, 400, { ok: false, error: 'Add at least one valid URL row before saving the set' });
      return;
    }

    try {
      await saveSetTemplate(next);
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, { ok: true, set: serializeSetTemplate(next) });
    return;
  }

  if (!isGenerateRoute && req.method === 'DELETE') {
    try {
      await deleteSetTemplate(setId);
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

  if (isGenerateRoute && req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const template = serializeSetTemplate(existing);
    const enabledEntries = (Array.isArray(template.entries) ? template.entries : []).filter(entry => entry && entry.enabled !== false);
    if (!enabledEntries.length) {
      sendJson(res, 400, { ok: false, error: 'This template does not have any enabled URLs to generate' });
      return;
    }

    const timing = resolveBatchTiming(body, template);
    if (!timing.ok) {
      sendJson(res, 400, { ok: false, error: timing.error || 'Invalid expiration settings' });
      return;
    }

    let existingLinks;
    try {
      existingLinks = await listLinks();
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    const usedLowerSlugs = new Set(
      (Array.isArray(existingLinks) ? existingLinks : [])
        .map(item => normalizeSlugLower(item && item.slug))
        .filter(Boolean)
    );

    const randomLength = normalizeRandomLength(body && body.randomLength, template.defaultRandomLength);
    const context = normalizeGenerationContext(body && body.context);
    const batchId = createBatchId();
    const batchTitle = buildBatchTitle(body && body.batchTitle, template.title, context);
    const createdAt = new Date().toISOString();
    const savedLinks = [];

    for (const entry of enabledEntries) {
      const destination = normalizeDestination(entry.destination, { absolutizeInternalPath: true });
      if (!destination) {
        sendJson(res, 400, { ok: false, error: `Invalid destination in template row "${entry.label}"` });
        return;
      }

      let record = null;
      for (let attempt = 0; attempt < RANDOM_SLUG_RETRY_LIMIT && !record; attempt += 1) {
        const slug = createUniqueSlug(randomLength, usedLowerSlugs);
        if (!slug) break;
        try {
          record = await upsertLink({
            slug,
            destination,
            permanent: timing.permanent,
            expiresAt: timing.permanent ? 0 : timing.expiresAt,
            updatedAt: createdAt,
            metadata: {
              label: entry.label,
              templateId: template.setId,
              templateTitle: template.title,
              batchId,
              batchTitle,
              contextType: context.type,
              contextEntryId: context.entryId,
              contextCompany: context.company,
              contextTitle: context.title
            }
          });
        } catch (err) {
          if (isSlugConflictError(err)) continue;
          if (err.code === 'DDB_ENV_MISSING') {
            sendJson(res, 503, { ok: false, error: err.message });
            return;
          }
          sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
          return;
        }
      }

      if (!record) {
        sendJson(res, 409, { ok: false, error: 'Unable to generate a unique short code right now' });
        return;
      }
      savedLinks.push(record);
    }

    try {
      await saveGeneratedBatch({
        slug: buildBatchRecordKey(batchId),
        entityType: 'generatedBatch',
        batchId,
        templateId: template.setId,
        templateTitle: template.title,
        batchTitle,
        contextType: context.type,
        contextEntryId: context.entryId,
        contextCompany: context.company,
        contextTitle: context.title,
        randomLength,
        permanent: timing.permanent,
        expiresAt: timing.permanent ? 0 : timing.expiresAt,
        expirationMode: timing.expirationMode,
        durationValue: timing.durationValue,
        durationUnit: timing.durationUnit,
        links: savedLinks.map(link => ({
          label: typeof link?.label === 'string' ? link.label : '',
          slug: typeof link?.slug === 'string' ? link.slug : '',
          destination: typeof link?.destination === 'string' ? link.destination : ''
        })),
        createdAt,
        updatedAt: createdAt
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
      batch: {
        batchId,
        batchTitle,
        templateId: template.setId,
        templateTitle: template.title,
        randomLength,
        permanent: timing.permanent,
        expiresAt: timing.permanent ? 0 : timing.expiresAt,
        context
      },
      links: savedLinks.map(link => serializeGeneratedLink(link, req))
    });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', isGenerateRoute ? 'POST' : 'GET, PATCH, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
