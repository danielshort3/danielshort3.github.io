'use strict';

const DEFAULT_UPSTREAM = 'https://muee4eg6ze.execute-api.us-east-2.amazonaws.com/prod/contact';

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
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

  try {
    const upstreamRes = await fetch(upstream, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload || {})
    });

    const raw = await upstreamRes.text();
    let data = {};
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = { ok: upstreamRes.ok, message: raw };
      }
    }

    if (!upstreamRes.ok) {
      sendJson(res, upstreamRes.status, {
        ok: false,
        error: data && data.error ? data.error : 'Unable to send message.'
      });
      return;
    }

    sendJson(res, 200, data && typeof data === 'object' ? data : { ok: true });
  } catch {
    sendJson(res, 502, { ok: false, error: 'Contact service unavailable' });
  }
};
