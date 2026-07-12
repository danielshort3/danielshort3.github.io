/*
  Authenticated tools activity API.

  - GET /api/tools/activity?tool=<toolId>&limit=100&cursor=<opaque>
  - GET /api/tools/activity?limit=100&cursor=<opaque> (global)
  - POST /api/tools/activity { toolId, type, summary?, data? }
*/
'use strict';

const {
  sendJson,
  readJson,
  normalizeKnownToolId,
  normalizeToolId,
  clampLimit
} = require('../tools-api');
const { authenticateToolsRequest } = require('../tools-auth-session');
const { logActivity, listActivity } = require('../tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

function logStorageError(err){
  const cancellationReasons = Array.isArray(err?.CancellationReasons)
    ? err.CancellationReasons.map(reason => String(reason?.Code || '')).filter(Boolean).slice(0, 8)
    : [];
  console.error('[tools-activity] storage error', {
    name: String(err?.name || ''),
    code: String(err?.code || ''),
    statusCode: Number(err?.$metadata?.httpStatusCode) || 0,
    cancellationReasons
  });
}

function sendStorageError(res, err){
  if (err?.code === 'ACTIVITY_TOO_LARGE') {
    sendJson(res, 413, { ok: false, error: err.message });
    return;
  }
  if (err?.code === 'INVALID_CURSOR') {
    sendJson(res, 400, { ok: false, error: err.message });
    return;
  }
  if (err?.code === 'KV_ENV_MISSING' || err?.code === 'DDB_ENV_MISSING') {
    sendJson(res, 503, { ok: false, error: err.message });
    return;
  }
  logStorageError(err);
  sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
}

module.exports = async (req, res) => {
  let claims;
  try {
    ({ claims } = await authenticateToolsRequest(req));
  } catch (err) {
    if (['COGNITO_ENV_MISSING', 'TOOLS_SESSION_SECRET_MISSING', 'TOOLS_SESSION_SECRET_INVALID'].includes(err.code)) {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    if (err.code === 'AUTH_ORIGIN_MISMATCH') {
      sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
      return;
    }
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  const sub = String(claims?.sub || '').trim();
  if (!sub) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method === 'GET') {
    const rawToolId = pickQuery(req.query?.tool || req.query?.toolId);
    const toolId = normalizeToolId(rawToolId);
    const limit = clampLimit(pickQuery(req.query?.limit), 100, 200);
    const cursor = String(pickQuery(req.query?.cursor) || '').trim();
    if (rawToolId && !toolId) {
      sendJson(res, 400, { ok: false, error: 'Unknown toolId' });
      return;
    }
    try {
      const events = await listActivity({ sub, toolId: toolId || undefined, limit, cursor });
      sendJson(res, 200, { ok: true, toolId: toolId || '', events, nextCursor: events.nextCursor || '' });
      return;
    } catch (err) {
      sendStorageError(res, err);
      return;
    }
  }

  if (req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
    } catch (err) {
      if (err?.code === 'BODY_TOO_LARGE') {
        sendJson(res, 413, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const toolId = normalizeKnownToolId(body?.toolId || body?.tool);
    const type = String(body?.type || '').trim();
    const summary = body?.summary;
    const data = body?.data;

    if (!toolId) {
      sendJson(res, 400, { ok: false, error: 'Invalid toolId' });
      return;
    }
    if (!type || type.length > 80) {
      sendJson(res, 400, { ok: false, error: 'Invalid type' });
      return;
    }

    try {
      const event = await logActivity({ sub, toolId, type, summary, data });
      sendJson(res, 200, { ok: true, event });
      return;
    } catch (err) {
      sendStorageError(res, err);
      return;
    }
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
