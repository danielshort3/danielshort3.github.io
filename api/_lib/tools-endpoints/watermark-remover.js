'use strict';

const { createHmac, randomBytes } = require('node:crypto');
const { isToolsAdminClaims } = require('../short-links');
const { sendJson } = require('../tools-api');
const { assertSameOriginRequest, authenticateToolsRequest } = require('../tools-auth-session');

const FUNNEL_ORIGIN = 'https://daniel-pc.tail9686c5.ts.net:8443';
const LAUNCH_PATH = '/home-tools/watermark/session';
const TICKET_TTL_SECONDS = 60;

function getWorkerOrigin(env) {
  let target;
  try {
    target = new URL(env.HOME_GPU_WATERMARK_ORIGIN);
  } catch {
    return '';
  }
  if (target.username || target.password || target.pathname !== '/' || target.search || target.hash) return '';
  if (target.origin === FUNNEL_ORIGIN) return target.origin;

  const deployed = Boolean(env.VERCEL || env.VERCEL_ENV || env.NODE_ENV === 'production');
  const loopback = ['127.0.0.1', '[::1]', 'localhost'].includes(target.hostname);
  return !deployed && loopback && target.protocol === 'http:' ? target.origin : '';
}

async function handleWatermarkRemover(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return sendJson(res, 405, { ok: false, error: 'Method not allowed.' });
  }

  let claims;
  let siteOrigin;
  try {
    assertSameOriginRequest(req);
    siteOrigin = new URL(req.headers.origin).origin;
    ({ claims } = await authenticateToolsRequest(req, { allowBearer: true }));
  } catch (error) {
    if (error.code === 'AUTH_ORIGIN_MISMATCH') {
      return sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
    }
    return sendJson(res, 401, { ok: false, error: 'Sign in to open Watermark Remover.' });
  }
  if (!claims?.sub || !isToolsAdminClaims(claims)) {
    return sendJson(res, 403, { ok: false, error: 'Administrator access required.' });
  }

  const workerOrigin = getWorkerOrigin(process.env);
  const secret = process.env.HOME_GPU_WATERMARK_SECRET || '';
  if (!workerOrigin || Buffer.byteLength(secret, 'utf8') < 32) {
    return sendJson(res, 503, { ok: false, error: 'Watermark Remover is unavailable.' });
  }

  const now = Math.floor(Date.now() / 1000);
  const payload = Buffer.from(JSON.stringify({
    v: 1,
    aud: 'home-gpu-watermark',
    sub: claims.sub,
    origin: siteOrigin,
    iat: now,
    exp: now + TICKET_TTL_SECONDS,
    jti: randomBytes(16).toString('base64url')
  }), 'utf8').toString('base64url');
  const signature = createHmac('sha256', secret).update(payload, 'ascii').digest('base64url');
  return sendJson(res, 200, { ok: true, launchUrl: `${workerOrigin}${LAUNCH_PATH}`, ticket: `${payload}.${signature}` });
}

module.exports = handleWatermarkRemover;
module.exports.getWorkerOrigin = getWorkerOrigin;
