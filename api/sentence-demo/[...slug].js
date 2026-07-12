'use strict';

const { handleDemoRequest } = require('../_lib/demo-proxy');

function splitSegments(value) {
  const values = Array.isArray(value) ? value : [value];
  const segments = [];
  for (const item of values) {
    if (typeof item !== 'string') continue;
    let decoded;
    try {
      decoded = decodeURIComponent(item);
    } catch {
      return [];
    }
    segments.push(...decoded.split('/').filter(Boolean));
  }
  return segments;
}

function getLegacySegments(req) {
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug) || typeof querySlug === 'string') {
    return ['smart-sentence', ...splitSegments(querySlug)];
  }
  try {
    const url = new URL(req.url, 'https://example.com');
    const match = url.pathname.match(/\/api\/sentence-demo\/(.+)$/);
    return ['smart-sentence', ...(match ? splitSegments(match[1]) : [])];
  } catch {
    return ['smart-sentence'];
  }
}

module.exports = async (req, res) => handleDemoRequest(req, res, getLegacySegments(req));
