'use strict';

const {
  getAdminToken,
  getChatbotLog,
  isAdminRequest,
  listChatbotLogs,
  summarizeLog
} = require('../_lib/chatbot-logs');

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

function pickQuery(value) {
  return Array.isArray(value) ? value[0] : value;
}

function queryFromRequest(req) {
  if (req.query && typeof req.query === 'object') return req.query;
  try {
    const url = new URL(req.url, 'https://example.com');
    return Object.fromEntries(url.searchParams.entries());
  } catch {
    return {};
  }
}

module.exports = async (req, res) => {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  if (!getAdminToken()) {
    sendJson(res, 503, { ok: false, error: 'CHATBOT_ADMIN_TOKEN is not configured' });
    return;
  }

  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  const query = queryFromRequest(req);
  const logId = String(pickQuery(query.id) || '').trim();

  try {
    if (logId) {
      const log = await getChatbotLog(logId);
      if (!log) {
        sendJson(res, 404, { ok: false, error: 'Not Found' });
        return;
      }
      sendJson(res, 200, { ok: true, log, summary: summarizeLog(log) });
      return;
    }

    const limit = Number(pickQuery(query.limit)) || 50;
    const logs = await listChatbotLogs({ limit });
    sendJson(res, 200, { ok: true, logs });
  } catch (err) {
    if (err && err.code === 'CHATBOT_LOG_STORE_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'Chatbot log store unavailable' });
  }
};
