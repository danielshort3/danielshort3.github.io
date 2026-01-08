/*
  Public redirect endpoint: /go/<slug>
  Backed by AWS DynamoDB.
*/
'use strict';

const crypto = require('crypto');
const { getLink, incrementClicks, recordClick } = require('../_lib/short-links-store');
const { normalizeSlug, getRequestBaseUrl } = require('../_lib/short-links');

function isShortDomainHost(req){
  const hostHeader = req.headers && req.headers.host ? String(req.headers.host) : '';
  const host = hostHeader.split(':')[0].trim().toLowerCase();
  return host === 'dshort.me' || host === 'www.dshort.me';
}

function getSlugFromRequest(req){
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug.join('/');
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/(?:api\/)?go\/(.+)$/);
    if (match) return decodeURIComponent(match[1]);
    if (isShortDomainHost(req)) return url.pathname.replace(/^\/+|\/+$/g, '');
    return '';
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

  let link;
  try {
    link = await getLink(slug);
  } catch (err) {
    res.statusCode = err.code === 'DDB_ENV_MISSING' ? 503 : 502;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Short links backend unavailable');
    return;
  }

  if (!link) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  if (link.disabled) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Robots-Tag', 'noindex');
    res.end('Not Found');
    return;
  }

  const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
  if (expiresAt && Math.floor(Date.now() / 1000) >= expiresAt) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Robots-Tag', 'noindex');
    res.end('Not Found');
    return;
  }

  const destination = typeof link.destination === 'string' ? String(link.destination).trim() : '';
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
    reqUrl.searchParams.forEach((value, key) => {
      if (key === 'slug' || key === '...slug' || key === 'first' || key === 'rest' || key.startsWith('...')) return;
      destUrl.searchParams.append(key, value);
    });
    finalUrl = destUrl.toString();
  } catch {
    finalUrl = destination;
  }

  if (req.method === 'GET') {
    const now = new Date();
    const clickId = `${now.toISOString()}#${crypto.randomBytes(4).toString('hex')}`;
    const hostHeader = req.headers && req.headers.host ? String(req.headers.host) : '';
    const referer = (req.headers && (req.headers.referer || req.headers.referrer))
      ? String(req.headers.referer || req.headers.referrer)
      : '';
    const userAgent = (req.headers && req.headers['user-agent']) ? String(req.headers['user-agent']) : '';
    const country = (req.headers && req.headers['x-vercel-ip-country']) ? String(req.headers['x-vercel-ip-country']) : '';
    const region = (req.headers && req.headers['x-vercel-ip-country-region']) ? String(req.headers['x-vercel-ip-country-region']) : '';
    const city = (req.headers && req.headers['x-vercel-ip-city']) ? String(req.headers['x-vercel-ip-city']) : '';

    const clickEvent = {
      slug,
      clickId,
      clickedAt: now.toISOString(),
      destination: finalUrl,
      statusCode: link.permanent ? 301 : 302,
      host: hostHeader.split(':')[0].trim(),
      referer,
      userAgent,
      country,
      region,
      city
    };

    try {
      const results = await Promise.allSettled([
        incrementClicks(slug),
        recordClick(clickEvent)
      ]);
      const clickResult = results[1];
      if (clickResult && clickResult.status === 'rejected') {
        const reason = clickResult.reason || {};
        console.warn('Shortlinks click log failed', {
          slug,
          name: reason.name || '',
          message: reason.message || ''
        });
      }
    } catch {}
  }

  res.statusCode = link.permanent ? 301 : 302;
  res.setHeader('Location', finalUrl);
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex');
  res.end();
};
