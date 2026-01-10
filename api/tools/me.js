/*
  Return the current signed-in user (validated via Cognito ID token).
*/
'use strict';

const { sendJson, getBearerToken } = require('../_lib/tools-api');
const { verifyCognitoIdToken } = require('../_lib/cognito-jwt');

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

  sendJson(res, 200, {
    ok: true,
    user: {
      sub,
      email: String(claims?.email || '').trim(),
      name: String(claims?.name || claims?.['cognito:username'] || '').trim()
    }
  });
};

