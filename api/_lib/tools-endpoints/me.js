/*
  Return the current signed-in user (validated via Cognito ID token).
*/
'use strict';

const { sendJson } = require('../tools-api');
const { authenticateToolsRequest } = require('../tools-auth-session');

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

  sendJson(res, 200, {
    ok: true,
    user: {
      sub,
      email: String(claims?.email || '').trim(),
      name: String(claims?.name || claims?.['cognito:username'] || '').trim()
    }
  });
};
