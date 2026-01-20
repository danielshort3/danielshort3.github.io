/*
  Aggregate dashboard data for the signed-in user (tools used, sessions, activity).
*/
'use strict';

const { sendJson, getBearerToken, clampLimit } = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');
const { listUserTools, getToolMeta, listRecentSessions, listActivity } = require('../tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

module.exports = async (req, res) => {
  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

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

  const sessionsLimit = clampLimit(pickQuery(req.query?.sessionsLimit), 20, 50);
  const activityLimit = clampLimit(pickQuery(req.query?.activityLimit), 50, 200);

  try {
    const toolIds = await listUserTools(sub);
    const tools = [];
    for (const toolId of toolIds) {
      const meta = await getToolMeta(sub, toolId);
      tools.push({ toolId, meta: meta || { toolId } });
    }
    const recentSessions = await listRecentSessions(sub, sessionsLimit);
    const recentActivity = await listActivity({ sub, limit: activityLimit });

    sendJson(res, 200, {
      ok: true,
      user: {
        sub,
        email: String(claims?.email || '').trim(),
        name: String(claims?.name || claims?.['cognito:username'] || '').trim()
      },
      tools,
      recentSessions,
      recentActivity
    });
  } catch (err) {
    if (err.code === 'KV_ENV_MISSING' || err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
  }
};

