'use strict';

const crypto = require('crypto');

const MAX_BODY_BYTES = 32 * 1024;
const MAX_SIGNATURE_AGE_SECONDS = 300;

function normalizeOrigin(value) {
  try {
    const url = new URL(String(value || ''));
    if (url.username || url.password || url.pathname !== '/' || url.search || url.hash) return '';
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) return '';
    return url.origin;
  } catch {
    return '';
  }
}

function getProxySecret(env = process.env) {
  const secret = String(env.CONTACT_PROXY_SECRET || '');
  if (Buffer.byteLength(secret, 'utf8') < 32) {
    const error = new Error('Contact request protection is not configured.');
    error.code = 'CONTACT_PROTECTION_MISSING';
    throw error;
  }
  return secret;
}

function signatureValue({ timestamp, nonce, actor, origin, body }, secret) {
  const hash = crypto.createHash('sha256').update(body, 'utf8').digest('hex');
  return crypto.createHmac('sha256', secret)
    .update(`v1\n${timestamp}\n${nonce}\n${actor}\n${origin}\n${hash}`)
    .digest('hex');
}

function createSignedHeaders({ body, clientIp, origin, now = Date.now(), nonce = crypto.randomBytes(16).toString('hex') }, secret) {
  const timestamp = String(Math.floor(now / 1000));
  const actor = crypto.createHmac('sha256', secret).update(String(clientIp || 'unknown')).digest('hex');
  return {
    'Content-Type': 'application/json',
    Origin: origin,
    'X-Contact-Timestamp': timestamp,
    'X-Contact-Nonce': nonce,
    'X-Contact-Actor': actor,
    'X-Contact-Signature': signatureValue({ timestamp, nonce, actor, origin, body }, secret)
  };
}

function verifySignedRequest(event, secret, now = Date.now()) {
  const headers = Object.fromEntries(Object.entries(event.headers || {}).map(([key, value]) => [key.toLowerCase(), String(value || '')]));
  const timestamp = headers['x-contact-timestamp'] || '';
  const nonce = headers['x-contact-nonce'] || '';
  const actor = headers['x-contact-actor'] || '';
  const signature = headers['x-contact-signature'] || '';
  const origin = normalizeOrigin(headers.origin);
  const body = event.isBase64Encoded ? Buffer.from(event.body || '', 'base64').toString('utf8') : String(event.body || '');
  if (!/^\d{10}$/.test(timestamp) || !/^[a-f0-9]{32}$/.test(nonce) || !/^[a-f0-9]{64}$/.test(actor) || !/^[a-f0-9]{64}$/.test(signature) || !origin) return null;
  if (Math.abs(Math.floor(now / 1000) - Number(timestamp)) > MAX_SIGNATURE_AGE_SECONDS || Buffer.byteLength(body, 'utf8') > MAX_BODY_BYTES) return null;
  const expected = signatureValue({ timestamp, nonce, actor, origin, body }, secret);
  if (!crypto.timingSafeEqual(Buffer.from(signature, 'hex'), Buffer.from(expected, 'hex'))) return null;
  return { actor, nonce, origin, body };
}

function validatePayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
  if (!['name', 'email', 'message'].every((key) => typeof payload[key] === 'string')) return null;
  if (typeof payload.company !== 'undefined' && typeof payload.company !== 'string') return null;
  const sanitize = (value) => value.replace(/[\r\n\t]+/g, ' ').trim();
  const name = sanitize(payload.name);
  const email = sanitize(payload.email);
  const message = payload.message.trim();
  const company = sanitize(payload.company || '');
  if (!name || name.length > 200 || email.length > 320 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || !message || message.length > 4000) return null;
  return { name, email, message, company };
}

module.exports = { MAX_BODY_BYTES, MAX_SIGNATURE_AGE_SECONDS, createSignedHeaders, getProxySecret, normalizeOrigin, validatePayload, verifySignedRequest };
