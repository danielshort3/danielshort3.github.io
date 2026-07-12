/*
  Same-origin tools session bridge.

  - POST /api/tools/auth/exchange  (Cognito Bearer token -> HttpOnly cookie)
  - GET  /api/tools/auth/session   (read current cookie/bearer session)
  - POST /api/tools/auth/logout    (clear the cookie)
*/
'use strict';

const { sendJson, getBearerToken } = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');
const {
  assertSameOriginRequest,
  authenticateToolsRequest,
  createSessionFromClaims,
  serializeClearedSessionCookie
} = require('../tools-auth-session');

function userFromClaims(claims){
  return {
    sub: String(claims?.sub || '').trim(),
    email: String(claims?.email || '').trim(),
    name: String(claims?.name || claims?.['cognito:username'] || '').trim(),
    groups: Array.isArray(claims?.['cognito:groups'])
      ? claims['cognito:groups'].map((group) => String(group || '').trim()).filter(Boolean)
      : []
  };
}

function sendAuthError(res, err){
  if (['COGNITO_ENV_MISSING', 'TOOLS_SESSION_SECRET_MISSING', 'TOOLS_SESSION_SECRET_INVALID'].includes(err?.code)) {
    sendJson(res, 503, { ok: false, error: err.message });
    return;
  }
  if (err?.code === 'AUTH_ORIGIN_MISMATCH') {
    sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
    return;
  }
  sendJson(res, 401, { ok: false, error: 'Unauthorized' });
}

async function exchange(req, res, dependencies = {}){
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  try {
    assertSameOriginRequest(req);
    const token = getBearerToken(req);
    if (!token) {
      sendJson(res, 401, { ok: false, error: 'Unauthorized' });
      return;
    }
    const verifyToken = typeof dependencies.verifyToken === 'function'
      ? dependencies.verifyToken
      : verifyCognitoIdToken;
    const claims = await verifyToken(token);
    const session = createSessionFromClaims(claims);
    res.setHeader('Set-Cookie', session.cookie);
    sendJson(res, 200, {
      ok: true,
      source: 'cookie',
      expiresAt: session.payload.exp,
      user: userFromClaims(claims)
    });
  } catch (err) {
    sendAuthError(res, err);
  }
}

async function session(req, res, dependencies = {}){
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  try {
    const authenticateRequest = typeof dependencies.authenticateRequest === 'function'
      ? dependencies.authenticateRequest
      : authenticateToolsRequest;
    const auth = await authenticateRequest(req);
    sendJson(res, 200, {
      ok: true,
      source: auth.source,
      expiresAt: auth.expiresAt,
      user: userFromClaims(auth.claims)
    });
  } catch (err) {
    sendAuthError(res, err);
  }
}

function logout(req, res){
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  try {
    assertSameOriginRequest(req);
    res.setHeader('Set-Cookie', serializeClearedSessionCookie());
    sendJson(res, 200, { ok: true });
  } catch (err) {
    sendAuthError(res, err);
  }
}

function createHandler(dependencies = {}){
  return async (req, res, segments = []) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Pragma', 'no-cache');
    const action = String(segments[0] || '').trim().toLowerCase();
    if (action === 'exchange') return exchange(req, res, dependencies);
    if (action === 'session') return session(req, res, dependencies);
    if (action === 'logout') return logout(req, res);
    sendJson(res, 404, { ok: false, error: 'Not Found' });
  };
}

module.exports = createHandler();

module.exports.userFromClaims = userFromClaims;
module.exports.createHandler = createHandler;
