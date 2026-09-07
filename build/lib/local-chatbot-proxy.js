'use strict';

const UPSTREAM_BASE = 'https://k8bys9gicf.execute-api.us-east-2.amazonaws.com/prod';
const ROUTES = {
  '/api/chatbot-demo/bedrock/status': { method: 'GET', path: '/bedrock/status' },
  '/api/chatbot-demo/qwen/status': { method: 'GET', path: '/status' },
  '/api/chatbot-demo/qwen/warmup': { method: 'POST', path: '/warmup' },
  '/api/chatbot-demo/qwen/submit': { method: 'POST', path: '/submit' },
  '/api/chatbot-demo/qwen/result': { method: 'GET', path: '/result', result: true }
};
const MAX_BODY_BYTES = 8 * 1024;
const MAX_RESPONSE_BYTES = 1024 * 1024;

function sendJson(res, status, value) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(JSON.stringify(value));
}

async function handleLocalChatbotRequest(req, res, { fetchImpl = fetch } = {}) {
  const origin = `http://${req.headers.host}`;
  if (req.headers['sec-fetch-site'] === 'cross-site' ||
    (req.headers.origin && req.headers.origin !== origin)) {
    sendJson(res, 403, { error: 'Origin not allowed.' });
    return;
  }
  const url = new URL(req.url, origin);
  const route = Object.hasOwn(ROUTES, url.pathname) ? ROUTES[url.pathname] : null;
  if (!route) {
    sendJson(res, 404, { error: 'Unknown local chatbot route.' });
    return;
  }
  if (req.method !== route.method) {
    res.setHeader('Allow', route.method);
    sendJson(res, 405, { error: 'Method not allowed.' });
    return;
  }
  const query = [...url.searchParams];
  const [key, value] = query[0] || [];
  const validQuery = route.result
    ? query.length === 1 && ['jobId', 'outputUri'].includes(key) && value.trim() &&
      value.length <= (key === 'jobId' ? 128 : 2048)
    : query.length === 0;
  if (!validQuery) {
    sendJson(res, 400, { error: 'Invalid chatbot query.' });
    return;
  }

  let body;
  if (route.method === 'POST') {
    if (String(req.headers['content-type'] || '').split(';')[0].trim().toLowerCase() !== 'application/json') {
      sendJson(res, 415, { error: 'Content-Type must be application/json.' });
      return;
    }
    let size = 0;
    const chunks = [];
    try {
      if (Number(req.headers['content-length']) > MAX_BODY_BYTES) {
        sendJson(res, 413, { error: 'Chatbot request is too large.' });
        return;
      }
      for await (const chunk of req) {
        const buffer = Buffer.from(chunk);
        size += buffer.length;
        if (size > MAX_BODY_BYTES) {
          sendJson(res, 413, { error: 'Chatbot request is too large.' });
          return;
        }
        chunks.push(buffer);
      }
      const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid body');
      body = JSON.stringify(data);
    } catch {
      sendJson(res, 400, { error: 'Invalid JSON request.' });
      return;
    }
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 35000);
  const cancel = () => controller.abort();
  res.once('close', cancel);
  try {
    const upstream = await fetchImpl(`${UPSTREAM_BASE}${route.path}${url.search}`, {
      method: route.method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      ...(body ? { body } : {}),
      redirect: 'error',
      signal: controller.signal
    });
    const chunks = [];
    let size = 0;
    for await (const chunk of upstream.body) {
      size += chunk.length;
      if (size > MAX_RESPONSE_BYTES) {
        controller.abort();
        throw new Error('Response too large');
      }
      chunks.push(Buffer.from(chunk));
    }
    const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (upstream.status === 429 && /^\d+$/.test(upstream.headers.get('retry-after') || '')) {
      res.setHeader('Retry-After', upstream.headers.get('retry-after'));
    }
    sendJson(res, upstream.status, upstream.status >= 500 ? { error: 'Chatbot service is unavailable.' } : data);
  } catch {
    if (!res.destroyed && !res.writableEnded) {
      sendJson(res, 502, { error: 'Local chatbot connection failed.' });
    }
  } finally {
    clearTimeout(timeout);
    res.removeListener('close', cancel);
  }
}

module.exports = { handleLocalChatbotRequest };
