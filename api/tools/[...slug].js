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
const handleTranscribe = require('../_lib/tools-endpoints/transcribe');
const handleAuth = require('../_lib/tools-endpoints/auth');
const handleAwsStatus = require('../_lib/tools-endpoints/aws-status');

function getEndpointSegmentsFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.filter(Boolean).map(value => String(value));
  if (typeof querySlug === 'string') return querySlug.split('/').filter(Boolean);
  try {
    const url = new URL(req.url, 'https://example.com');
    const match = url.pathname.match(/\/api\/tools\/(.+)$/);
    if (!match) return [];
    const raw = decodeURIComponent(match[1]);
    return raw.split('/').filter(Boolean);
  } catch {
    return [];
  }
}

module.exports = async (req, res) => {
  const segments = getEndpointSegmentsFromRequest(req);
  const endpoint = segments[0] || '';
  const rest = segments.slice(1);

  if (endpoint === 'me') return handleMe(req, res);
  if (endpoint === 'dashboard') return handleDashboard(req, res);
  if (endpoint === 'state') return handleState(req, res);
  if (endpoint === 'activity') return handleActivity(req, res);
  if (endpoint === 'transcribe') return handleTranscribe(req, res, rest);
  if (endpoint === 'auth') return handleAuth(req, res, rest);
  if (endpoint === 'aws-status' && rest.length === 0) return handleAwsStatus(req, res);

  sendJson(res, 404, { ok: false, error: 'Not Found' });
};
