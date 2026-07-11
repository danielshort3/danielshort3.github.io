'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { authorizeAdminRequest, verifiedPrincipal } = require('../api/_lib/admin-auth');
const { verifyCognitoIdToken } = require('../api/_lib/cognito-jwt');

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function req(headers = {}) {
  return { headers };
}

function bearer(token) {
  return req({ authorization: `Bearer ${token}` });
}

function base64UrlJson(value) {
  return Buffer.from(JSON.stringify(value)).toString('base64url');
}

function signJwt(payload, privateKey, kid) {
  const header = base64UrlJson({ alg: 'RS256', typ: 'JWT', kid });
  const body = base64UrlJson(payload);
  const signingInput = `${header}.${body}`;
  const signature = crypto.sign('RSA-SHA256', Buffer.from(signingInput), privateKey).toString('base64url');
  return `${signingInput}.${signature}`;
}

function response() {
  return {
    statusCode: 200,
    headers: {},
    body: '',
    setHeader(name, value) { this.headers[String(name).toLowerCase()] = value; },
    end(value) { this.body = String(value || ''); }
  };
}

async function main() {
  const original = {
    AWS_AUTH_MODE: process.env.AWS_AUTH_MODE,
    TOOLS_COGNITO_ISSUER: process.env.TOOLS_COGNITO_ISSUER,
    TOOLS_COGNITO_CLIENT_ID: process.env.TOOLS_COGNITO_CLIENT_ID,
    SHORTLINKS_ADMIN_TOKEN: process.env.SHORTLINKS_ADMIN_TOKEN,
    CHATBOT_ADMIN_TOKEN: process.env.CHATBOT_ADMIN_TOKEN
  };
  const originalFetch = global.fetch;

  const issuer = 'https://cognito-idp.us-east-2.amazonaws.com/us-east-2_AdminAuthTest';
  const clientId = 'admin-auth-client';
  const kid = 'admin-auth-test-key';
  const now = Math.floor(Date.now() / 1000);
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  const jwk = publicKey.export({ format: 'jwk' });
  Object.assign(jwk, { kid, alg: 'RS256', use: 'sig' });

  const claims = (groups) => ({
    iss: issuer,
    aud: clientId,
    sub: 'user-123',
    token_use: 'id',
    exp: now + 3600,
    iat: now,
    ...(typeof groups === 'undefined' ? {} : { 'cognito:groups': groups })
  });

  try {
    process.env.AWS_AUTH_MODE = 'oidc';
    process.env.TOOLS_COGNITO_ISSUER = issuer;
    process.env.TOOLS_COGNITO_CLIENT_ID = clientId;
    process.env.SHORTLINKS_ADMIN_TOKEN = 'legacy-shortlinks-secret';
    process.env.CHATBOT_ADMIN_TOKEN = 'legacy-chatbot-secret';
    global.fetch = async (url) => {
      check(url === `${issuer}/.well-known/jwks.json`, 'JWT verifier should fetch only the configured issuer JWKS');
      return { ok: true, status: 200, json: async () => ({ keys: [jwk] }) };
    };

    const adminToken = signJwt(claims(['admins']), privateKey, kid);
    const verified = await verifyCognitoIdToken(adminToken);
    check(verified.sub === 'user-123', 'valid signed Cognito ID token should verify');
    check(verified['cognito:groups'].includes('admins'), 'verified token should retain Cognito groups');

    const [header, payload, signature] = adminToken.split('.');
    const forgedPayload = base64UrlJson({ ...claims(['admins']), sub: 'forged-user' });
    await assert.rejects(
      verifyCognitoIdToken(`${header}.${forgedPayload}.${signature}`),
      (err) => err && err.code === 'JWT_SIG',
      'modified claims with the original signature must be rejected'
    );
    checks += 1;

    const missingUse = { ...claims(['admins']) };
    delete missingUse.token_use;
    await assert.rejects(
      verifyCognitoIdToken(signJwt(missingUse, privateKey, kid)),
      (err) => err && err.code === 'JWT_USE',
      'tokens without token_use=id must be rejected'
    );
    checks += 1;

    const oidcEnv = { AWS_AUTH_MODE: 'oidc', SHORTLINKS_ADMIN_TOKEN: 'legacy-secret' };
    const verifier = async (token) => {
      if (token === 'valid-admin') return claims(['admins']);
      if (token === 'wrong-group') return claims(['editors']);
      if (token === 'no-group') return claims(undefined);
      if (token === 'fallback-group') return { ...claims(undefined), groups: ['admins'] };
      if (token === 'malformed-verified') return { 'cognito:groups': ['admins'] };
      const err = new Error('bad signature');
      err.code = 'JWT_SIG';
      throw err;
    };

    const valid = await authorizeAdminRequest(bearer('valid-admin'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(valid.authorized && valid.authType === 'cognito', 'verified admins-group token should authorize');
    check(valid.principal.sub === 'user-123' && !valid.principal.email, 'admin principal should use subject, not email');

    const wrongGroup = await authorizeAdminRequest(bearer('wrong-group'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!wrongGroup.authorized && wrongGroup.statusCode === 403, 'verified wrong-group token should be forbidden');

    const noGroup = await authorizeAdminRequest(bearer('no-group'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!noGroup.authorized && noGroup.statusCode === 403, 'verified token without groups should be forbidden');

    const fallbackGroup = await authorizeAdminRequest(bearer('fallback-group'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!fallbackGroup.authorized && fallbackGroup.statusCode === 403, 'generic groups claim must not replace cognito:groups');

    const malformed = await authorizeAdminRequest(bearer('malformed-verified'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!malformed.authorized && malformed.statusCode === 401, 'claims missing verified ID-token invariants should be unauthorized');

    const missing = await authorizeAdminRequest(req(), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!missing.authorized && missing.statusCode === 401, 'missing bearer token should be unauthorized');

    const forged = await authorizeAdminRequest(bearer('forged.jwt.value'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!forged.authorized && forged.statusCode === 401, 'unverified forged claims should be unauthorized');

    const staticBearerInOidc = await authorizeAdminRequest(bearer('legacy-secret'), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!staticBearerInOidc.authorized, 'static bearer token must be ignored in OIDC mode');

    const staticHeaderInOidc = await authorizeAdminRequest(req({ 'x-admin-token': 'legacy-secret' }), {
      env: oidcEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!staticHeaderInOidc.authorized, 'static admin header must be ignored in OIDC mode');

    const legacyEnv = { AWS_AUTH_MODE: 'legacy', SHORTLINKS_ADMIN_TOKEN: 'legacy-secret' };
    const legacy = await authorizeAdminRequest(req({ 'x-shortlinks-token': 'legacy-secret' }), {
      env: legacyEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(legacy.authorized && legacy.authType === 'legacy-token', 'static token should work only in explicit legacy mode');

    const wrongServiceHeader = await authorizeAdminRequest(req({ 'x-chatbot-admin-token': 'legacy-secret' }), {
      env: legacyEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!wrongServiceHeader.authorized, 'legacy service-specific headers must not cross authorization boundaries');

    const legacyWrong = await authorizeAdminRequest(bearer('wrong-secret'), {
      env: legacyEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!legacyWrong.authorized, 'wrong legacy token should be rejected');

    const cognitoDuringLegacy = await authorizeAdminRequest(bearer('valid-admin'), {
      env: legacyEnv,
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(cognitoDuringLegacy.authorized && cognitoDuringLegacy.authType === 'cognito', 'Cognito admin should remain valid during rollback mode');

    const defaultMode = await authorizeAdminRequest(req({ 'x-admin-token': 'legacy-secret' }), {
      env: { SHORTLINKS_ADMIN_TOKEN: 'legacy-secret' },
      verifyToken: verifier,
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!defaultMode.authorized, 'static token must not work in implicit local/default mode');

    const configFailure = await authorizeAdminRequest(bearer('anything'), {
      env: oidcEnv,
      verifyToken: async () => {
        const err = new Error('missing Cognito env');
        err.code = 'COGNITO_ENV_MISSING';
        throw err;
      },
      legacyTokenEnv: 'SHORTLINKS_ADMIN_TOKEN'
    });
    check(!configFailure.authorized && configFailure.statusCode === 503, 'OIDC Cognito configuration failures should fail closed');

    check(verifiedPrincipal(claims(['admins'])).isAdmin, 'exact admins Cognito group should be recognized');
    check(!verifiedPrincipal(claims(['admin'])).isAdmin, 'singular admin group should not be recognized');
    check(!verifiedPrincipal({ ...claims(undefined), 'cognito:groups': 'admins' }).isAdmin, 'string group claim should not be trusted');

    const frontendConfig = fs.readFileSync(path.join(__dirname, '..', 'js', 'accounts', 'tools-config.js'), 'utf8');
    const frontendAuth = fs.readFileSync(path.join(__dirname, '..', 'js', 'accounts', 'tools-auth.js'), 'utf8');
    check(!frontendConfig.includes('adminEmails') && !frontendAuth.includes('getConfiguredAdminEmails'), 'frontend admin authorization must not use email allowlists');
    check(frontendAuth.includes("claims['cognito:groups']") && !frontendAuth.includes('claims.groups'), 'frontend admin UI should derive only from Cognito group claims');

    const protectedFiles = [
      'api/short-links/index.js',
      'api/short-links/[...slug].js',
      'api/_lib/short-links-clicks-handler.js',
      'api/short-links/sets/[...setId].js',
      'api/short-links/test/[...slug].js',
      'api/short-links/health.js'
    ];
    for (const file of protectedFiles) {
      const source = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
      check(source.includes('authorizeShortLinksAdmin'), `${file} should enforce the Cognito admin boundary`);
      check(!source.includes('isAdminRequest'), `${file} should not use the old static-token helper`);
    }
    const chatbotApi = fs.readFileSync(path.join(__dirname, '..', 'api', '_lib', 'chatbot-logs-api.js'), 'utf8');
    check(chatbotApi.includes('authorizeAdminRequest') && chatbotApi.includes("legacyTokenEnv: 'CHATBOT_ADMIN_TOKEN'"), 'chatbot log endpoint should enforce shared admin authorization');

    process.env.AWS_AUTH_MODE = 'oidc';
    const shortLinksApi = require('../api/short-links/index.js');
    const wrongGroupRes = response();
    await shortLinksApi({ method: 'GET', headers: { authorization: `Bearer ${signJwt(claims(['editors']), privateKey, kid)}` } }, wrongGroupRes);
    check(wrongGroupRes.statusCode === 403, 'Short Links endpoint should reject a verified non-admin group before storage access');

    const chatbotLogsApi = require('../api/_lib/chatbot-logs-api');
    const forgedRes = response();
    await chatbotLogsApi({ method: 'GET', headers: { authorization: `Bearer ${header}.${forgedPayload}.${signature}` }, query: {} }, forgedRes);
    check(forgedRes.statusCode === 401, 'chatbot logs endpoint should reject a forged bearer token');
  } finally {
    global.fetch = originalFetch;
    for (const [key, value] of Object.entries(original)) {
      if (typeof value === 'undefined') delete process.env[key];
      else process.env[key] = value;
    }
  }

  process.stdout.write(`Admin authorization tests passed (${checks} checks).\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
