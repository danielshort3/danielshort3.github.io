/*
  Tools account endpoints router: /api/tools/<endpoint>

  Consolidated into a single Serverless Function to avoid hitting Vercel Hobby plan limits.
*/
'use strict';

const { sendJson } = require('../_lib/tools-api');
const handleMe = require('../_lib/tools-endpoints/me');
const handleDashboard = require('../_lib/tools-endpoints/dashboard');
const handleState = require('../_lib/tools-endpoints/state');
const handleActivity = require('../_lib/tools-endpoints/activity');

function getEndpointFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug[0] || '';
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, 'https://example.com');
    const match = url.pathname.match(/\/api\/tools\/(.+)$/);
    if (!match) return '';
    const raw = decodeURIComponent(match[1]);
    return raw.split('/')[0] || '';
  } catch {
    return '';
  }
}

module.exports = async (req, res) => {
  const endpoint = getEndpointFromRequest(req);

  if (endpoint === 'me') return handleMe(req, res);
  if (endpoint === 'dashboard') return handleDashboard(req, res);
  if (endpoint === 'state') return handleState(req, res);
  if (endpoint === 'activity') return handleActivity(req, res);

  sendJson(res, 404, { ok: false, error: 'Not Found' });
};

