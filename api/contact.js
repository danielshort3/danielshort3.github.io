'use strict';

const DEFAULT_UPSTREAM = 'https://muee4eg6ze.execute-api.us-east-2.amazonaws.com/prod/contact';
const UPSTREAM_DEADLINE_MS = 20000;
const DELIVERY_UNKNOWN = 'CONTACT_DELIVERY_UNKNOWN';
const UNKNOWN_MESSAGE = 'We couldn’t confirm delivery. Your message may have been sent. Your draft is still here.';

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

async function readJson(req) {
  if (req.body && typeof req.body === 'object') {
    return req.body;
  }
  if (typeof req.body === 'string' && req.body.trim()) {
    return JSON.parse(req.body);
  }

  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  }

  const raw = Buffer.concat(chunks).toString('utf8').trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.statusCode = 405;
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const upstream = String(process.env.CONTACT_FORM_UPSTREAM || DEFAULT_UPSTREAM).trim();
  if (!upstream) {
    sendJson(res, 503, { ok: false, error: 'CONTACT_FORM_UPSTREAM is not configured' });
    return;
  }

  let payload;
  try {
    payload = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
    return;
  }

  const controller = new AbortController();
  let deadline;
  let timedOut = false;
  try {
    // Race the complete response, including its body: receiving headers is not
    // confirmation that SES accepted the message.
    const request = (async () => {
      const upstreamRes = await fetch(upstream, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload || {}),
        signal: controller.signal
      });
      const raw = await upstreamRes.text();
      let data = null;
      try { data = JSON.parse(raw); } catch (_) {}
      return { upstreamRes, data };
    })();
    const timeout = new Promise((_, reject) => {
      deadline = setTimeout(() => {
        timedOut = true;
        reject(new Error(DELIVERY_UNKNOWN));
        controller.abort();
      }, UPSTREAM_DEADLINE_MS);
    });
    const { upstreamRes, data } = await Promise.race([request, timeout]);

    if (!upstreamRes.ok) {
      const rejected = upstreamRes.status >= 400 && upstreamRes.status < 500;
      sendJson(res, rejected ? upstreamRes.status : upstreamRes.status === 504 ? 504 : 502, {
        ok: false,
        code: rejected ? 'CONTACT_REJECTED' : DELIVERY_UNKNOWN,
        error: rejected ? 'Your message was not accepted. Check the form or email me directly.' : UNKNOWN_MESSAGE
      });
      return;
    }
    if (!data || data.ok !== true || data.error) {
      sendJson(res, 502, { ok: false, code: DELIVERY_UNKNOWN, error: UNKNOWN_MESSAGE });
      return;
    }
    sendJson(res, 200, { ok: true });
  } catch {
    sendJson(res, timedOut ? 504 : 502, { ok: false, code: DELIVERY_UNKNOWN, error: UNKNOWN_MESSAGE });
  } finally {
    clearTimeout(deadline);
  }
};
