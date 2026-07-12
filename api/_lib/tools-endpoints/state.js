/*
  Authenticated tools sessions API.

  - GET /api/tools/state?tool=<toolId>&limit=25&cursor=<opaque>
  - GET /api/tools/state?tool=<toolId>&session=<sessionId>
  - POST /api/tools/state { toolId, sessionId?, snapshot, outputSummary?, expectedVersion? }
  - PATCH /api/tools/state { toolId, sessionId, title?, note?, tags?, pinned?, expectedVersion? }
  - DELETE /api/tools/state?tool=<toolId>&session=<sessionId>&version=<expectedVersion>
  - DELETE /api/tools/state?all=true with X-Tools-Confirm: DELETE-ALL
*/
'use strict';

const {
  sendJson,
  readJson,
  getBearerToken,
  normalizeKnownToolId,
  normalizeToolId,
  normalizeSessionId,
  clampLimit
} = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');
const {
  MAX_SNAPSHOT_BYTES,
  saveSession,
  listSessions,
  getSession,
  updateSessionMeta,
  deleteSession,
  listRecentSessions,
  deleteAllUserData
} = require('../tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

function parseExpectedVersion(value){
  if (typeof value === 'undefined' || value === null || value === '') return { ok: true, value: undefined };
  const version = Number(value);
  if (!Number.isInteger(version) || version < 0) return { ok: false, value: undefined };
  return { ok: true, value: version };
}

function isTruthyQuery(value){
  return ['1', 'true', 'yes'].includes(String(pickQuery(value) || '').trim().toLowerCase());
}

function sendStorageError(res, err){
  if (err?.code === 'BODY_TOO_LARGE') {
    sendJson(res, 413, { ok: false, error: err.message });
    return;
  }
  if (err?.code === 'INVALID_CURSOR') {
    sendJson(res, 400, { ok: false, error: err.message });
    return;
  }
  if (['VERSION_CONFLICT', 'WRITE_CONFLICT'].includes(err?.code)) {
    sendJson(res, 409, { ok: false, error: err.message, code: err.code });
    return;
  }
  if (err?.code === 'SESSION_EXPIRED') {
    sendJson(res, 410, { ok: false, error: err.message, code: err.code });
    return;
  }
  if (err?.code === 'SESSION_QUOTA_EXCEEDED') {
    sendJson(res, 409, { ok: false, error: err.message, code: err.code });
    return;
  }
  if (err?.code === 'DELETE_TOO_LARGE') {
    sendJson(res, 413, { ok: false, error: err.message, code: err.code });
    return;
  }
  if (err?.code === 'DELETE_IN_PROGRESS') {
    sendJson(res, 409, { ok: false, error: err.message, code: err.code });
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
    const sessionId = normalizeSessionId(pickQuery(req.query?.session || req.query?.sessionId));
    const limit = clampLimit(pickQuery(req.query?.limit), 25, 50);
    const cursor = String(pickQuery(req.query?.cursor) || '').trim();

    if (rawToolId && !toolId) {
      sendJson(res, 400, { ok: false, error: 'Unknown toolId' });
      return;
    }

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
        const sessions = await listSessions({ sub, toolId, limit, cursor });
        sendJson(res, 200, { ok: true, toolId, sessions, nextCursor: sessions.nextCursor || '' });
        return;
      }

      const sessions = await listRecentSessions(sub, limit, cursor);
      sendJson(res, 200, { ok: true, sessions, nextCursor: sessions.nextCursor || '' });
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
    const sessionId = normalizeSessionId(body?.sessionId || body?.session);
    const snapshot = body?.snapshot;
    const outputSummary = body?.outputSummary;
    const expected = parseExpectedVersion(body?.expectedVersion);

    if (!toolId) {
      sendJson(res, 400, { ok: false, error: 'Invalid toolId' });
      return;
    }
    if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
      sendJson(res, 400, { ok: false, error: 'Invalid snapshot (expected object)' });
      return;
    }
    if (!expected.ok) {
      sendJson(res, 400, { ok: false, error: 'Invalid expectedVersion' });
      return;
    }
    if (typeof expected.value === 'undefined') {
      sendJson(res, 428, { ok: false, error: 'expectedVersion is required (use 0 when creating a session).' });
      return;
    }

    try {
      const record = await saveSession({
        sub,
        toolId,
        sessionId: sessionId || undefined,
        snapshot,
        outputSummary,
        expectedVersion: expected.value
      });
      sendJson(res, 200, {
        ok: true,
        session: {
          toolId: record.toolId,
          sessionId: record.sessionId,
          createdAt: record.createdAt,
          updatedAt: record.updatedAt,
          outputSummary: record.outputSummary || '',
          version: record.version,
          expiresAt: record.expiresAt || 0
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
      sendStorageError(res, err);
      return;
    }
  }

  if (req.method === 'PATCH') {
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
    const sessionId = normalizeSessionId(body?.sessionId || body?.session);
    const expected = parseExpectedVersion(body?.expectedVersion);

    if (!toolId) {
      sendJson(res, 400, { ok: false, error: 'Invalid toolId' });
      return;
    }
    if (!sessionId) {
      sendJson(res, 400, { ok: false, error: 'Invalid sessionId' });
      return;
    }
    if (!expected.ok) {
      sendJson(res, 400, { ok: false, error: 'Invalid expectedVersion' });
      return;
    }
    if (typeof expected.value === 'undefined') {
      sendJson(res, 428, { ok: false, error: 'expectedVersion is required.' });
      return;
    }

    const hasAnyUpdate = ['title', 'note', 'tags', 'pinned'].some((key) => Object.prototype.hasOwnProperty.call(body || {}, key));
    if (!hasAnyUpdate) {
      sendJson(res, 400, { ok: false, error: 'No session updates provided' });
      return;
    }

    try {
      const record = await updateSessionMeta({
        sub,
        toolId,
        sessionId,
        title: body?.title,
        note: body?.note,
        tags: body?.tags,
        pinned: body?.pinned,
        expectedVersion: expected.value
      });

      if (!record) {
        sendJson(res, 404, { ok: false, error: 'Session not found' });
        return;
      }

      sendJson(res, 200, { ok: true, session: record });
      return;
    } catch (err) {
      sendStorageError(res, err);
      return;
    }
  }

  if (req.method === 'DELETE') {
    if (isTruthyQuery(req.query?.all)) {
      const confirmation = String(req.headers?.['x-tools-confirm'] || '').trim();
      if (confirmation !== 'DELETE-ALL') {
        sendJson(res, 400, { ok: false, error: 'Delete-all requires the X-Tools-Confirm: DELETE-ALL header.' });
        return;
      }
      if (typeof deleteAllUserData !== 'function') {
        sendJson(res, 503, { ok: false, error: 'Delete-all requires the DynamoDB storage backend.' });
        return;
      }
      try {
        const result = await deleteAllUserData({ sub });
        sendJson(res, 200, { ok: true, deleted: result });
      } catch (err) {
        sendStorageError(res, err);
      }
      return;
    }

    const rawToolId = pickQuery(req.query?.tool || req.query?.toolId);
    const toolId = normalizeToolId(rawToolId);
    const sessionId = normalizeSessionId(pickQuery(req.query?.session || req.query?.sessionId));
    const expected = parseExpectedVersion(pickQuery(req.query?.version || req.query?.expectedVersion));
    if (!toolId || !sessionId) {
      sendJson(res, 400, { ok: false, error: 'tool and session are required' });
      return;
    }
    if (!expected.ok) {
      sendJson(res, 400, { ok: false, error: 'Invalid expectedVersion' });
      return;
    }
    if (typeof expected.value === 'undefined') {
      sendJson(res, 428, { ok: false, error: 'expectedVersion is required.' });
      return;
    }

    try {
      const deleted = await deleteSession({ sub, toolId, sessionId, expectedVersion: expected.value });
      if (!deleted) {
        sendJson(res, 404, { ok: false, error: 'Session not found' });
        return;
      }
      sendJson(res, 200, { ok: true });
      return;
    } catch (err) {
      sendStorageError(res, err);
      return;
    }
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST, PATCH, DELETE');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
