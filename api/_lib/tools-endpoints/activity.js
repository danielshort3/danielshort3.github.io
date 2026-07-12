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
  getBearerToken,
  normalizeKnownToolId,
  normalizeToolId,
  clampLimit
} = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');
const { logActivity, listActivity } = require('../tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

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
  sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
}

module.exports = async (req, res) => {
  const token = getBearerToken(req);
  if (!token) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  let claims;
  try {
    claims = await verifyCognitoIdToken(token);
  } catch (err) {
    if (err.code === 'COGNITO_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
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
