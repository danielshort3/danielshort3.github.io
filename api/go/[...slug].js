/*
  Public redirect endpoint: /go/<slug>
  Backed by AWS DynamoDB.
*/
'use strict';

const crypto = require('crypto');
const { getLinkWithLegacyFallback, incrementClicks, recordClick } = require('../_lib/short-links-store');
const { normalizeSlug, getRequestBaseUrl } = require('../_lib/short-links');

function isShortDomainHost(req){
  const hostHeader = req.headers && req.headers.host ? String(req.headers.host) : '';
  const host = hostHeader.split(':')[0].trim().toLowerCase();
  return host === 'dshort.me' || host === 'www.dshort.me';
}

function getRequestHost(req){
  const hostHeader = req.headers && req.headers.host ? String(req.headers.host) : '';
  return hostHeader.split(':')[0].trim();
}

function getHeader(req, name){
  if (!req || !req.headers) return '';
  const value = req.headers[name];
  if (Array.isArray(value)) return value[0] ? String(value[0]) : '';
  return value ? String(value) : '';
}

function decodeHeaderValue(value){
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function getRequestPath(req, base){
  try {
    const url = new URL(req.url, base);
    return `${url.pathname}${url.search || ''}`;
  } catch {
    return typeof req.url === 'string' ? req.url : '';
  }
}

function getUrlHost(value){
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw).hostname;
  } catch {
    return '';
  }
}

function buildUnavailableRedirect(req, slug){
  const host = getRequestHost(req) || 'dshort.me';
  const params = new URLSearchParams();
  params.set('from', host);
  if (slug) params.set('path', slug);
  return `https://www.danielshort.me/dshort?${params.toString()}`;
}

function sendUnavailable(res, req, slug){
  if (req.method !== 'GET') {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Robots-Tag', 'noindex');
    res.end('Not Found');
    return;
  }

  res.statusCode = 302;
  res.setHeader('Location', buildUnavailableRedirect(req, slug));
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex');
  res.end();
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
    link = await getLinkWithLegacyFallback(slug);
  } catch (err) {
    res.statusCode = err.code === 'DDB_ENV_MISSING' ? 503 : 502;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Short links backend unavailable');
    return;
  }

  if (!link) {
    if (isShortDomainHost(req)) {
      sendUnavailable(res, req, slug);
      return;
    }

    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  if (link.disabled) {
    if (isShortDomainHost(req)) {
      sendUnavailable(res, req, slug);
      return;
    }

    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Robots-Tag', 'noindex');
    res.end('Not Found');
    return;
  }

  const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
  if (expiresAt && Math.floor(Date.now() / 1000) >= expiresAt) {
    if (isShortDomainHost(req)) {
      sendUnavailable(res, req, slug);
      return;
    }

    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Robots-Tag', 'noindex');
    res.end('Not Found');
    return;
  }

  const destination = typeof link.destination === 'string' ? String(link.destination).trim() : '';
  if (!destination) {
    if (isShortDomainHost(req)) {
      sendUnavailable(res, req, slug);
      return;
    }

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
    const hostHeader = getHeader(req, 'host');
    const referer = getHeader(req, 'referer') || getHeader(req, 'referrer');
    const userAgent = getHeader(req, 'user-agent');
    const country = decodeHeaderValue(getHeader(req, 'x-vercel-ip-country'));
    const region = decodeHeaderValue(getHeader(req, 'x-vercel-ip-country-region'));
    const city = decodeHeaderValue(getHeader(req, 'x-vercel-ip-city'));
    const timezone = decodeHeaderValue(getHeader(req, 'x-vercel-ip-timezone'));
    const latitude = decodeHeaderValue(getHeader(req, 'x-vercel-ip-latitude'));
    const longitude = decodeHeaderValue(getHeader(req, 'x-vercel-ip-longitude'));

    const clickEvent = {
      slug,
      clickId,
      clickedAt: now.toISOString(),
      destination: finalUrl,
      statusCode: link.permanent ? 301 : 302,
      host: hostHeader.split(':')[0].trim(),
      path: getRequestPath(req, base),
      referer,
      refererHost: getUrlHost(referer),
      userAgent,
      country,
      region,
      city,
      timezone,
      latitude,
      longitude
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
