'use strict';

const crypto = require('crypto');

const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_TOTAL_TIMEOUT_MS = 25_000;
const DEFAULT_MAX_ATTEMPTS = 3;
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

function positiveInteger(value, fallback, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.min(parsed, max);
}

function requestTimeoutMs() {
  return positiveInteger(process.env.GA4_REQUEST_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, 30_000);
}

function retryAttempts() {
  return positiveInteger(process.env.GA4_RETRY_ATTEMPTS, DEFAULT_MAX_ATTEMPTS, 4);
}

function totalTimeoutMs() {
  return positiveInteger(process.env.GA4_TOTAL_TIMEOUT_MS, DEFAULT_TOTAL_TIMEOUT_MS, 55_000);
}

function retryDelayMs(response, attempt) {
  const retryAfter = Number(response?.headers?.get?.('retry-after'));
  if (Number.isFinite(retryAfter) && retryAfter >= 0) return Math.min(2_000, retryAfter * 1000);
  return Math.min(1_500, 200 * (2 ** Math.max(0, attempt - 1)));
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createTimeoutError() {
  const err = new Error('GA4 request timed out.');
  err.status = 504;
  return err;
}

async function fetchJsonWithRetry(url, options = {}, controls = {}) {
  const attempts = retryAttempts();
  const deadlineAt = Number(controls.deadlineAt) || (Date.now() + totalTimeoutMs());
  let lastError = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const remainingMs = deadlineAt - Date.now();
    if (remainingMs <= 0) throw createTimeoutError();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), Math.min(requestTimeoutMs(), remainingMs));
    let response;
    let payload = null;
    try {
      response = await fetch(url, { ...options, signal: controller.signal });
      try {
        payload = await response.json();
      } catch (err) {
        if (err?.name === 'AbortError') throw err;
      }
    } catch (err) {
      lastError = err;
      if (attempt >= attempts || Date.now() >= deadlineAt) {
        if (err?.name === 'AbortError') throw createTimeoutError();
        throw err;
      }
      await wait(Math.min(retryDelayMs(null, attempt), Math.max(0, deadlineAt - Date.now())));
      continue;
    } finally {
      clearTimeout(timeout);
    }

    if (response.ok || !RETRYABLE_STATUSES.has(response.status) || attempt >= attempts) {
      return { response, payload };
    }

    await wait(Math.min(retryDelayMs(response, attempt), Math.max(0, deadlineAt - Date.now())));
  }

  throw lastError || new Error('GA4 request failed.');
}

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

async function fetchAccessToken(serviceAccount, scope, deadlineAt) {
  const tokenUri = String(serviceAccount.token_uri || '').trim();
  if (!tokenUri) throw new Error('Service account token_uri is missing.');

  const assertion = buildServiceAccountAssertion(serviceAccount, scope);
  const body = new URLSearchParams();
  body.set('grant_type', 'urn:ietf:params:oauth:grant-type:jwt-bearer');
  body.set('assertion', assertion);

  const { response: res, payload } = await fetchJsonWithRetry(tokenUri, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString()
  }, { deadlineAt });

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

function clearCachedAccessToken() {
  cachedToken = '';
  cachedTokenExpiresAt = 0;
}

async function getAccessToken(serviceAccount, scope, deadlineAt) {
  if (cachedToken && Date.now() < cachedTokenExpiresAt - 30_000) return cachedToken;
  const { accessToken, expiresIn } = await fetchAccessToken(serviceAccount, scope, deadlineAt);
  cachedToken = accessToken;
  cachedTokenExpiresAt = Date.now() + Math.max(60, expiresIn) * 1000;
  return cachedToken;
}

async function runReport(serviceAccount, propertyId, reportBody, controls = {}) {
  const url = `https://analyticsdata.googleapis.com/v1beta/properties/${encodeURIComponent(propertyId)}:runReport`;
  const deadlineAt = Number(controls.deadlineAt) || (Date.now() + totalTimeoutMs());

  for (let authAttempt = 0; authAttempt < 2; authAttempt += 1) {
    const token = await getAccessToken(serviceAccount, 'https://www.googleapis.com/auth/analytics.readonly', deadlineAt);
    const { response: res, payload } = await fetchJsonWithRetry(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(reportBody || {})
    }, { deadlineAt });

    if (res.ok) return payload || {};
    if (res.status === 401 && authAttempt === 0) {
      clearCachedAccessToken();
      continue;
    }

    const message = payload?.error?.message || 'GA4 Data API request failed.';
    const err = new Error(message);
    err.status = res.status;
    err.details = payload?.error || payload;
    throw err;
  }

  throw new Error('GA4 authentication failed.');
}

module.exports = {
  clearCachedAccessToken,
  fetchJsonWithRetry,
  getAccessToken,
  runReport,
  totalTimeoutMs
};
