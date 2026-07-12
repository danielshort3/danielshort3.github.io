/*
  Aggregate dashboard data for the signed-in user (tools used, sessions, activity).
*/
'use strict';

const { sendJson, clampLimit } = require('../tools-api');
const { authenticateToolsRequest } = require('../tools-auth-session');
const { listUserTools, getToolMeta, listRecentSessions, listActivity } = require('../tools-store');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;

module.exports = async (req, res) => {
  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

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
      sessionsNextCursor: recentSessions.nextCursor || '',
      recentActivity,
      activityNextCursor: recentActivity.nextCursor || ''
    });
  } catch (err) {
    if (err.code === 'KV_ENV_MISSING' || err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'Storage backend unavailable' });
  }
};
