'use strict';

const crypto = require('crypto');

function base64UrlEncode(value) {
  const buf = Buffer.isBuffer(value) ? value : Buffer.from(String(value || ''), 'utf8');
  return buf.toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function signJwtRs256(payload, privateKeyPem, header = {}) {
  const jwtHeader = { alg: 'RS256', typ: 'JWT', ...header };
  const encodedHeader = base64UrlEncode(JSON.stringify(jwtHeader));
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const signingInput = `${encodedHeader}.${encodedPayload}`;
  const signer = crypto.createSign('RSA-SHA256');
  signer.update(signingInput);
  signer.end();
  const signature = signer.sign(privateKeyPem);
  return `${signingInput}.${base64UrlEncode(signature)}`;
}

function buildServiceAccountAssertion(serviceAccount, scope) {
  const now = Math.floor(Date.now() / 1000);
  const iss = String(serviceAccount.client_email || '').trim();
  const aud = String(serviceAccount.token_uri || '').trim();
  const privateKey = String(serviceAccount.private_key || '').trim();
  if (!iss || !aud || !privateKey) {
    throw new Error('Service account credentials are missing required fields.');
  }
  return signJwtRs256(
    {
      iss,
      scope: String(scope || '').trim(),
      aud,
      iat: now,
      exp: now + 60 * 60
    },
    privateKey
  );
}

async function fetchAccessToken(serviceAccount, scope) {
  const tokenUri = String(serviceAccount.token_uri || '').trim();
  if (!tokenUri) throw new Error('Service account token_uri is missing.');

  const assertion = buildServiceAccountAssertion(serviceAccount, scope);
  const body = new URLSearchParams();
  body.set('grant_type', 'urn:ietf:params:oauth:grant-type:jwt-bearer');
  body.set('assertion', assertion);

  const res = await fetch(tokenUri, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString()
  });

  let payload = null;
  try {
    payload = await res.json();
  } catch {}

  if (!res.ok) {
    const message = payload?.error_description || payload?.error || 'Unable to obtain a Google access token.';
    const err = new Error(message);
    err.status = res.status;
    err.details = payload;
    throw err;
  }

  const accessToken = String(payload?.access_token || '').trim();
  const expiresIn = Number(payload?.expires_in) || 3600;
  if (!accessToken) throw new Error('Google token response missing access_token.');
  return { accessToken, expiresIn };
}

let cachedToken = '';
let cachedTokenExpiresAt = 0;

async function getAccessToken(serviceAccount, scope) {
  if (cachedToken && Date.now() < cachedTokenExpiresAt - 30_000) return cachedToken;
  const { accessToken, expiresIn } = await fetchAccessToken(serviceAccount, scope);
  cachedToken = accessToken;
  cachedTokenExpiresAt = Date.now() + Math.max(60, expiresIn) * 1000;
  return cachedToken;
}

async function runReport(serviceAccount, propertyId, reportBody) {
  const token = await getAccessToken(serviceAccount, 'https://www.googleapis.com/auth/analytics.readonly');
  const url = `https://analyticsdata.googleapis.com/v1beta/properties/${encodeURIComponent(propertyId)}:runReport`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(reportBody || {})
  });

  let payload = null;
  try {
    payload = await res.json();
  } catch {}

  if (!res.ok) {
    const message = payload?.error?.message || 'GA4 Data API request failed.';
    const err = new Error(message);
    err.status = res.status;
    err.details = payload?.error || payload;
    throw err;
  }

  return payload || {};
}

module.exports = {
  getAccessToken,
  runReport
};

