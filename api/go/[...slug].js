/*
  Public redirect endpoint: /go/<slug>
  Backed by Upstash/Vercel KV.
*/
'use strict';

const { kvGet, kvIncr } = require('../_lib/kv');
const { clicksKey, linkKey, normalizeSlug, getRequestBaseUrl } = require('../_lib/short-links');

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/go\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : '';
  } catch {
    return '';
  }
}

module.exports = async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET, HEAD');
    res.end('Method Not Allowed');
    return;
  }

  const slug = normalizeSlug(getSlugFromRequest(req));
  if (!slug) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  let linkRaw;
  try {
    linkRaw = await kvGet(linkKey(slug));
  } catch (err) {
    res.statusCode = err.code === 'KV_ENV_MISSING' ? 503 : 502;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Short links backend unavailable');
    return;
  }

  if (!linkRaw) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  let link;
  try {
    link = JSON.parse(linkRaw);
  } catch {
    res.statusCode = 500;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Invalid short link record');
    return;
  }

  const destination = typeof link.destination === 'string' ? link.destination.trim() : '';
  if (!destination) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  const base = getRequestBaseUrl(req);
  let finalUrl;
  try {
    const reqUrl = new URL(req.url, base);
    const destUrl = new URL(destination, base);
    reqUrl.searchParams.forEach((value, key) => destUrl.searchParams.append(key, value));
    finalUrl = destUrl.toString();
  } catch {
    finalUrl = destination;
  }

  try {
    await kvIncr(clicksKey(slug));
  } catch {}

  res.statusCode = link.permanent ? 301 : 302;
  res.setHeader('Location', finalUrl);
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex');
  res.end();
};
