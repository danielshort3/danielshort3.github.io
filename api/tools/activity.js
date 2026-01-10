/*
  Tools activity API (KV-backed).

  - GET /api/tools/activity?tool=<toolId>&limit=100
  - GET /api/tools/activity?limit=100 (global)
  - POST /api/tools/activity { toolId, type, summary?, data? }
*/
'use strict';

const {
  sendJson,
  readJson,
  getBearerToken,
  normalizeToolId,
  clampLimit
} = require('../_lib/tools-api');
const { verifyCognitoIdToken } = require('../_lib/cognito-jwt');
const { logActivity, listActivity } = require('../_lib/tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

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
    const toolId = normalizeToolId(pickQuery(req.query?.tool || req.query?.toolId));
    const limit = clampLimit(pickQuery(req.query?.limit), 100, 200);
    try {
      const events = await listActivity({ sub, toolId: toolId || undefined, limit });
      sendJson(res, 200, { ok: true, toolId: toolId || '', events });
      return;
    } catch (err) {
      if (err.code === 'KV_ENV_MISSING' || err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
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

    const toolId = normalizeToolId(body?.toolId || body?.tool);
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
      if (err.code === 'KV_ENV_MISSING' || err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
      return;
    }
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
