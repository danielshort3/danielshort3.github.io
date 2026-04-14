/*
  Admin API for short-link set templates.
*/
'use strict';

const { listSetTemplates, saveSetTemplate } = require('../../_lib/short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  readJson,
  sendJson
} = require('../../_lib/short-links');
const {
  buildSetTemplateRecord,
  serializeSetTemplate
} = require('../../_lib/short-links-sets');

module.exports = async (req, res) => {
  const adminToken = getAdminToken();
  if (!adminToken) {
    sendJson(res, 503, { ok: false, error: 'SHORTLINKS_ADMIN_TOKEN is not configured' });
    return;
  }
  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method === 'GET') {
    try {
      const items = await listSetTemplates();
      sendJson(res, 200, {
        ok: true,
        sets: items.map(serializeSetTemplate)
      });
      return;
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }
  }

  if (req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    const record = buildSetTemplateRecord(body, null);
    if (!record.title) {
      sendJson(res, 400, { ok: false, error: 'Template title is required' });
      return;
    }
    if (!Array.isArray(record.entries) || record.entries.length === 0) {
      sendJson(res, 400, { ok: false, error: 'Add at least one valid URL row before saving the set' });
      return;
    }

    try {
      await saveSetTemplate(record);
    } catch (err) {
      if (err.code === 'DDB_ENV_MISSING') {
        sendJson(res, 503, { ok: false, error: err.message });
        return;
      }
      sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
      return;
    }

    sendJson(res, 200, { ok: true, set: serializeSetTemplate(record) });
    return;
  }

  res.statusCode = 405;
  res.setHeader('Allow', 'GET, POST');
  sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
};
