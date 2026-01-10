/*
  Verify Cognito JWTs (RS256) without external dependencies.

  Required env vars:
  - TOOLS_COGNITO_ISSUER (e.g. https://cognito-idp.us-east-2.amazonaws.com/us-east-2_XXXXXXX)
  - TOOLS_COGNITO_CLIENT_ID (Cognito App Client ID; compared to aud on ID tokens)
*/
'use strict';

const crypto = require('crypto');

const JWKS_CACHE_TTL_MS = 10 * 60 * 1000;
const jwksCache = new Map();

function getCognitoEnv(){
  const issuer = typeof process.env.TOOLS_COGNITO_ISSUER === 'string'
    ? process.env.TOOLS_COGNITO_ISSUER.trim().replace(/\/+$/, '')
    : '';
  const clientId = typeof process.env.TOOLS_COGNITO_CLIENT_ID === 'string'
    ? process.env.TOOLS_COGNITO_CLIENT_ID.trim()
    : '';

  if (!issuer || !clientId) {
    const err = new Error(
      'Missing Cognito env vars. Set TOOLS_COGNITO_ISSUER and TOOLS_COGNITO_CLIENT_ID.'
    );
    err.code = 'COGNITO_ENV_MISSING';
    throw err;
  }

  return { issuer, clientId };
}

function base64UrlDecode(value){
  const normalized = String(value || '').replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized.padEnd(normalized.length + (4 - normalized.length % 4) % 4, '=');
  return Buffer.from(padded, 'base64');
}

function parseJwt(token){
  const raw = String(token || '').trim();
  const parts = raw.split('.');
  if (parts.length !== 3) return null;
  const [headerB64, payloadB64, sigB64] = parts;
  let header;
  let payload;
  try {
    header = JSON.parse(base64UrlDecode(headerB64).toString('utf8'));
    payload = JSON.parse(base64UrlDecode(payloadB64).toString('utf8'));
  } catch {
    return null;
  }
  const signature = base64UrlDecode(sigB64);
  const signingInput = `${headerB64}.${payloadB64}`;
  return { header, payload, signature, signingInput };
}

async function fetchJwks(issuer){
  const cached = jwksCache.get(issuer);
  if (cached && cached.expiresAt > Date.now()) return cached.keysByKid;

  const url = `${issuer}/.well-known/jwks.json`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = new Error(`Unable to fetch JWKS (${res.status})`);
    err.code = 'JWKS_FETCH_FAILED';
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  const keys = Array.isArray(data?.keys) ? data.keys : [];
  const keysByKid = new Map();
  keys.forEach((key) => {
    if (key && key.kid) keysByKid.set(String(key.kid), key);
  });
  jwksCache.set(issuer, { keysByKid, expiresAt: Date.now() + JWKS_CACHE_TTL_MS });
  return keysByKid;
}

function verifyJwtSignature(jwt, jwk){
  const keyObject = crypto.createPublicKey({ key: jwk, format: 'jwk' });
  return crypto.verify(
    'RSA-SHA256',
    Buffer.from(jwt.signingInput, 'utf8'),
    keyObject,
    jwt.signature
  );
}

async function verifyCognitoIdToken(token){
  const env = getCognitoEnv();
  const jwt = parseJwt(token);
  if (!jwt) {
    const err = new Error('Invalid token format');
    err.code = 'JWT_INVALID';
    throw err;
  }
  if (jwt.header?.alg !== 'RS256') {
    const err = new Error('Unsupported token algorithm');
    err.code = 'JWT_ALG';
    throw err;
  }
  const kid = String(jwt.header?.kid || '').trim();
  if (!kid) {
    const err = new Error('Missing token kid');
    err.code = 'JWT_KID';
    throw err;
  }

  const nowSeconds = Math.floor(Date.now() / 1000);
  const iss = String(jwt.payload?.iss || '').trim().replace(/\/+$/, '');
  if (!iss || iss !== env.issuer) {
    const err = new Error('Invalid token issuer');
    err.code = 'JWT_ISS';
    throw err;
  }
  const aud = String(jwt.payload?.aud || '').trim();
  if (!aud || aud !== env.clientId) {
    const err = new Error('Invalid token audience');
    err.code = 'JWT_AUD';
    throw err;
  }
  const tokenUse = String(jwt.payload?.token_use || '').trim();
  if (tokenUse && tokenUse !== 'id') {
    const err = new Error('Invalid token_use (expected id)');
    err.code = 'JWT_USE';
    throw err;
  }
  const exp = Number(jwt.payload?.exp);
  if (!Number.isFinite(exp) || exp <= nowSeconds) {
    const err = new Error('Token expired');
    err.code = 'JWT_EXP';
    throw err;
  }

  const jwks = await fetchJwks(env.issuer);
  const jwk = jwks.get(kid);
  if (!jwk) {
    const err = new Error('Unknown token kid');
    err.code = 'JWT_KID_UNKNOWN';
    throw err;
  }
  const ok = verifyJwtSignature(jwt, jwk);
  if (!ok) {
    const err = new Error('Invalid token signature');
    err.code = 'JWT_SIG';
    throw err;
  }

  return jwt.payload;
}

module.exports = {
  getCognitoEnv,
  verifyCognitoIdToken
};

