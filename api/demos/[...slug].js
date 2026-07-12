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

function getSegments(req) {
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug) || typeof querySlug === 'string') return splitSegments(querySlug);
  try {
    const url = new URL(req.url, 'https://example.com');
    const match = url.pathname.match(/\/api\/demos\/(.+)$/);
    return match ? splitSegments(match[1]) : [];
  } catch {
    return [];
  }
}

module.exports = async (req, res) => handleDemoRequest(req, res, getSegments(req));
