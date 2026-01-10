/*
  Tools sessions API (KV-backed).

  - GET /api/tools/state?tool=<toolId>&limit=25
  - GET /api/tools/state?tool=<toolId>&session=<sessionId>
  - POST /api/tools/state { toolId, sessionId?, snapshot, outputSummary? }
  - DELETE /api/tools/state?tool=<toolId>&session=<sessionId>
*/
'use strict';

const {
  sendJson,
  readJson,
  getBearerToken,
  normalizeToolId,
  normalizeSessionId,
  clampLimit
} = require('../_lib/tools-api');
const { verifyCognitoIdToken } = require('../_lib/cognito-jwt');
const {
  MAX_SNAPSHOT_BYTES,
  saveSession,
  listSessions,
  getSession,
  deleteSession,
  listRecentSessions
} = require('../_lib/tools-store');

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
    const sessionId = normalizeSessionId(pickQuery(req.query?.session || req.query?.sessionId));
    const limit = clampLimit(pickQuery(req.query?.limit), 25, 50);

    try {
      if (toolId && sessionId) {
        const record = await getSession({ sub, toolId, sessionId });
        if (!record) {
          sendJson(res, 404, { ok: false, error: 'Session not found' });
          return;
        }
        sendJson(res, 200, { ok: true, session: record });
        return;
      }

      if (toolId) {
        const sessions = await listSessions({ sub, toolId, limit });
        sendJson(res, 200, { ok: true, toolId, sessions });
        return;
      }

      const sessions = await listRecentSessions(sub, limit);
      sendJson(res, 200, { ok: true, sessions });
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
    const sessionId = normalizeSessionId(body?.sessionId || body?.session);
    const snapshot = body?.snapshot;
    const outputSummary = body?.outputSummary;

    if (!toolId) {
      sendJson(res, 400, { ok: false, error: 'Invalid toolId' });
      return;
    }
    if (!snapshot || typeof snapshot !== 'object') {
      sendJson(res, 400, { ok: false, error: 'Invalid snapshot (expected object)' });
      return;
    }

    try {
      const record = await saveSession({ sub, toolId, sessionId: sessionId || undefined, snapshot, outputSummary });
      sendJson(res, 200, {
        ok: true,
        session: {
          toolId: record.toolId,
          sessionId: record.sessionId,
          createdAt: record.createdAt,
          updatedAt: record.updatedAt,
          outputSummary: record.outputSummary || ''
        },
        limits: {
          maxSnapshotBytes: MAX_SNAPSHOT_BYTES
        }
      });
      return;
    } catch (err) {
      if (err.code === 'SNAPSHOT_TOO_LARGE') {
        sendJson(res, 413, { ok: false, error: err.message });
        return;
      }
      if (err.code === 'KV_ENV_MISSING' || err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
      return;
    }
  }

  if (req.method === 'DELETE') {
    const toolId = normalizeToolId(pickQuery(req.query?.tool || req.query?.toolId));
    const sessionId = normalizeSessionId(pickQuery(req.query?.session || req.query?.sessionId));
    if (!toolId || !sessionId) {
      sendJson(res, 400, { ok: false, error: 'tool and session are required' });
      return;
    }

    try {
      const existing = await getSession({ sub, toolId, sessionId });
      if (!existing) {
        sendJson(res, 404, { ok: false, error: 'Session not found' });
        return;
      }
      await deleteSession({ sub, toolId, sessionId });
      sendJson(res, 200, { ok: true });
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
  res.setHeader('Allow', 'GET, POST, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
