'use strict';

const crypto = require('crypto');
const { getAwsAuthMode } = require('./aws-credentials');
const { verifyCognitoIdToken } = require('./cognito-jwt');

const ADMIN_GROUP = 'admins';
const LEGACY_TOKEN_ENVS = new Set([
  'SHORTLINKS_ADMIN_TOKEN',
  'CHATBOT_ADMIN_TOKEN'
]);

function headerValue(req, name) {
  const headers = req && req.headers && typeof req.headers === 'object' ? req.headers : {};
  const lower = String(name || '').toLowerCase();
  const direct = headers[lower];
  if (Array.isArray(direct)) return String(direct[0] || '').trim();
  if (typeof direct !== 'undefined') return String(direct || '').trim();
  const key = Object.keys(headers).find((candidate) => String(candidate).toLowerCase() === lower);
  if (!key) return '';
  const value = headers[key];
  return Array.isArray(value) ? String(value[0] || '').trim() : String(value || '').trim();
}

function getBearerToken(req) {
  const authorization = headerValue(req, 'authorization');
  return authorization.toLowerCase().startsWith('bearer ')
    ? authorization.slice(7).trim()
    : '';
}

function getLegacyTokenCandidate(req, legacyTokenEnv) {
  const serviceHeader = legacyTokenEnv === 'CHATBOT_ADMIN_TOKEN'
    ? headerValue(req, 'x-chatbot-admin-token')
    : headerValue(req, 'x-shortlinks-token');
  return headerValue(req, 'x-admin-token')
    || serviceHeader
    || getBearerToken(req);
}

function timingSafeMatch(provided, configured) {
  const left = Buffer.from(String(provided || ''), 'utf8');
  const right = Buffer.from(String(configured || ''), 'utf8');
  if (!left.length || !right.length || left.length !== right.length) {
    if (right.length) {
      try { crypto.timingSafeEqual(right, right); } catch {}
    }
    return false;
  }
  try {
    return crypto.timingSafeEqual(left, right);
  } catch {
    return false;
  }
}

function verifiedPrincipal(claims) {
  if (!claims || typeof claims !== 'object') return null;
  if (String(claims.token_use || '') !== 'id') return null;
  const sub = String(claims.sub || '').trim();
  if (!sub) return null;
  const groups = claims['cognito:groups'];
  const normalizedGroups = Array.isArray(groups)
    ? groups.map((group) => String(group || '').trim()).filter(Boolean)
    : [];
  return {
    sub,
    groups: normalizedGroups,
    isAdmin: normalizedGroups.includes(ADMIN_GROUP)
  };
}

function authFailure(statusCode, code, error, mode) {
  return { authorized: false, statusCode, code, error, mode };
}

function isConfigurationError(err) {
  return ['COGNITO_ENV_MISSING', 'JWKS_FETCH_FAILED', 'AWS_AUTH_MODE_REQUIRED', 'AWS_AUTH_MODE_INVALID']
    .includes(String(err && err.code || ''));
}

async function authorizeAdminRequest(req, options = {}) {
  const env = options.env || process.env;
  const verifier = options.verifyToken || verifyCognitoIdToken;
  const legacyTokenEnv = String(options.legacyTokenEnv || '').trim();
  if (!LEGACY_TOKEN_ENVS.has(legacyTokenEnv)) {
    throw new Error('authorizeAdminRequest requires an approved legacyTokenEnv');
  }

  let mode;
  try {
    mode = getAwsAuthMode(env);
  } catch (err) {
    return authFailure(503, err.code || 'ADMIN_AUTH_CONFIG', 'Admin authentication unavailable', 'invalid');
  }

  const bearer = getBearerToken(req);
  let verificationError = null;
  if (bearer) {
    try {
      const claims = await verifier(bearer);
      const principal = verifiedPrincipal(claims);
      if (!principal) {
        return authFailure(401, 'ADMIN_TOKEN_INVALID', 'Unauthorized', mode);
      }
      if (!principal.isAdmin) {
        return authFailure(403, 'ADMIN_GROUP_REQUIRED', 'Forbidden', mode);
      }
      return {
        authorized: true,
        mode,
        authType: 'cognito',
        group: ADMIN_GROUP,
        principal: { sub: principal.sub }
      };
    } catch (err) {
      verificationError = err;
      if (mode !== 'legacy') {
        if (isConfigurationError(err)) {
          return authFailure(503, err.code || 'ADMIN_AUTH_CONFIG', 'Admin authentication unavailable', mode);
        }
        return authFailure(401, 'ADMIN_TOKEN_INVALID', 'Unauthorized', mode);
      }
    }
  }

  if (mode === 'legacy') {
    const configured = String(env[legacyTokenEnv] || '').trim();
    const provided = getLegacyTokenCandidate(req, legacyTokenEnv);
    if (configured && timingSafeMatch(provided, configured)) {
      return {
        authorized: true,
        mode,
        authType: 'legacy-token',
        group: '',
        principal: null
      };
    }
    if (verificationError && isConfigurationError(verificationError) && !configured) {
      return authFailure(503, verificationError.code || 'ADMIN_AUTH_CONFIG', 'Admin authentication unavailable', mode);
    }
  }

  return authFailure(401, 'ADMIN_TOKEN_REQUIRED', 'Unauthorized', mode);
}

module.exports = {
  ADMIN_GROUP,
  authorizeAdminRequest,
  getBearerToken,
  getLegacyTokenCandidate,
  timingSafeMatch,
  verifiedPrincipal,
  _private: {
    LEGACY_TOKEN_ENVS,
    authFailure,
    headerValue,
    isConfigurationError
  }
};
