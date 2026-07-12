/*
  Shared implementation for the admin short-link destination test endpoint.
  Outbound requests resolve, validate, and then pin each connection to one of
  the validated addresses so a later DNS answer cannot redirect the socket.
*/
'use strict';

const dns = require('dns').promises;
const http = require('http');
const https = require('https');
const net = require('net');
const { getLinkWithLegacyFallback } = require('./short-links-store');
const {
  getAdminToken,
  isAdminRequest,
  sendJson,
  normalizeSlug,
  getRequestBaseUrl
} = require('./short-links');

const REQUEST_TIMEOUT_MS = 5000;
const MAX_REDIRECTS = 5;

function decodeRequestValue(value){
  const raw = String(value || '');
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function getSlugFromRequest(req, slugOverride){
  if (typeof slugOverride === 'string') return slugOverride;
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return decodeRequestValue(querySlug.join('/'));
  if (typeof querySlug === 'string') return decodeRequestValue(querySlug);
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/short-links\/test\/(.+)$/);
    return match ? decodeRequestValue(match[1]) : '';
  } catch {
    return '';
  }
}

function resolveDestination(destination, base){
  const raw = typeof destination === 'string' ? destination.trim() : '';
  if (!raw) return '';
  try {
    return new URL(raw, base).toString();
  } catch {
    return raw;
  }
}

function getErrorMessage(err){
  if (!err) return '';
  if (typeof err === 'string') return err;
  if (err.code === 'REQUEST_TIMEOUT') return 'Request timed out';
  if (err.message) return String(err.message);
  return '';
}

function parseIpv4(address){
  const parts = String(address || '').split('.').map(part => Number(part));
  if (parts.length !== 4 || parts.some(part => !Number.isInteger(part) || part < 0 || part > 255)) return null;
  return parts;
}

function isPrivateIpv4(address){
  const parts = parseIpv4(address);
  if (!parts) return true;
  const [a, b, c] = parts;
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 192 && b === 0 && c === 0) return true;
  if (a === 192 && b === 0 && c === 2) return true;
  if (a === 198 && (b === 18 || b === 19)) return true;
  if (a === 198 && b === 51 && c === 100) return true;
  if (a === 203 && b === 0 && c === 113) return true;
  if (a >= 224) return true;
  return false;
}

function expandIpv6(address){
  let normalized = String(address || '')
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, '')
    .replace(/%[^%]+$/, '');
  if (net.isIP(normalized) !== 6) return null;

  if (normalized.includes('.')) {
    const separator = normalized.lastIndexOf(':');
    const ipv4 = parseIpv4(normalized.slice(separator + 1));
    if (!ipv4) return null;
    const high = ((ipv4[0] << 8) | ipv4[1]).toString(16);
    const low = ((ipv4[2] << 8) | ipv4[3]).toString(16);
    normalized = `${normalized.slice(0, separator)}:${high}:${low}`;
  }

  const halves = normalized.split('::');
  if (halves.length > 2) return null;
  const left = halves[0] ? halves[0].split(':') : [];
  const right = halves.length === 2 && halves[1] ? halves[1].split(':') : [];
  const parseHextets = values => values.map(value => Number.parseInt(value, 16));
  const leftValues = parseHextets(left);
  const rightValues = parseHextets(right);
  if ([...leftValues, ...rightValues].some(value => !Number.isInteger(value) || value < 0 || value > 0xffff)) return null;

  if (halves.length === 1) {
    return leftValues.length === 8 ? leftValues : null;
  }

  const missing = 8 - leftValues.length - rightValues.length;
  if (missing < 1) return null;
  return [...leftValues, ...Array(missing).fill(0), ...rightValues];
}

function ipv4FromHextets(high, low){
  return [high >> 8, high & 0xff, low >> 8, low & 0xff].join('.');
}

function mappedIpv4FromIpv6(address){
  const hextets = expandIpv6(address);
  if (!hextets) return '';
  const mappedPrefix = hextets.slice(0, 5).every(value => value === 0) && hextets[5] === 0xffff;
  return mappedPrefix ? ipv4FromHextets(hextets[6], hextets[7]) : '';
}

function isPrivateIpv6(address){
  const hextets = expandIpv6(address);
  if (!hextets) return true;

  const mapped = mappedIpv4FromIpv6(address);
  if (mapped) return isPrivateIpv4(mapped);

  // IPv4-compatible IPv6 addresses are deprecated and can conceal IPv4 targets.
  if (hextets.slice(0, 6).every(value => value === 0)) return true;

  const first = hextets[0];
  if ((first & 0xfe00) === 0xfc00) return true;
  if ((first & 0xffc0) === 0xfe80) return true;
  if ((first & 0xffc0) === 0xfec0) return true;
  if ((first & 0xff00) === 0xff00) return true;
  return false;
}

function isBlockedAddress(address){
  const family = net.isIP(String(address || '').replace(/^\[|\]$/g, ''));
  if (family === 4) return isPrivateIpv4(address);
  if (family === 6) return isPrivateIpv6(address);
  return true;
}

function isLocalHostname(hostname){
  const host = String(hostname || '').trim().toLowerCase().replace(/\.$/, '');
  return !host || host === 'localhost' || host.endsWith('.localhost') || host === 'local' || host.endsWith('.local');
}

function createDestinationError(message, code){
  const err = new Error(message);
  err.code = code;
  return err;
}

async function resolveSafeDestinationUrl(rawUrl, lookup = dns.lookup){
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw createDestinationError('Invalid destination URL', 'INVALID_DESTINATION_URL');
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw createDestinationError('Only http and https destinations can be tested', 'UNSAFE_PROTOCOL');
  }

  if (parsed.username || parsed.password) {
    throw createDestinationError('Destination URLs with embedded credentials cannot be tested', 'UNSAFE_CREDENTIALS');
  }

  const hostname = String(parsed.hostname || '').replace(/^\[|\]$/g, '');
  if (isLocalHostname(hostname)) {
    throw createDestinationError('Local destinations cannot be tested', 'LOCAL_DESTINATION');
  }

  const literalFamily = net.isIP(hostname);
  if (literalFamily) {
    if (isBlockedAddress(hostname)) {
      throw createDestinationError('Private or local network destinations cannot be tested', 'PRIVATE_DESTINATION');
    }
    return {
      url: parsed.toString(),
      parsed,
      hostname,
      addresses: [{ address: hostname, family: literalFamily }]
    };
  }

  let records;
  try {
    records = await lookup(hostname, { all: true, verbatim: true });
  } catch {
    throw createDestinationError('Destination host could not be resolved', 'DNS_LOOKUP_FAILED');
  }

  if (!Array.isArray(records) || records.length === 0) {
    throw createDestinationError('Destination host could not be resolved', 'DNS_LOOKUP_EMPTY');
  }

  const addresses = [];
  const seen = new Set();
  for (const record of records) {
    const address = String(record && record.address || '').replace(/^\[|\]$/g, '');
    const family = net.isIP(address);
    if (!family || isBlockedAddress(address)) {
      throw createDestinationError('Destination resolves to a private or local network address', 'PRIVATE_DESTINATION');
    }
    const key = `${family}:${address.toLowerCase()}`;
    if (!seen.has(key)) {
      seen.add(key);
      addresses.push({ address, family });
    }
  }

  return {
    url: parsed.toString(),
    parsed,
    hostname,
    addresses
  };
}

async function assertSafeDestinationUrl(rawUrl, lookup){
  const resolved = await resolveSafeDestinationUrl(rawUrl, lookup);
  return resolved.url;
}

function getResponseHeader(resp, name){
  if (!resp || !resp.headers) return '';
  if (typeof resp.headers.get === 'function') return resp.headers.get(name) || '';
  const value = resp.headers[String(name || '').toLowerCase()];
  return Array.isArray(value) ? String(value[0] || '') : String(value || '');
}

function getManualRedirectUrl(resp, baseUrl){
  const status = Number(resp && (resp.status || resp.statusCode));
  if (status < 300 || status >= 400) return '';
  const location = getResponseHeader(resp, 'location');
  if (!location) return '';
  try {
    return new URL(location, baseUrl).toString();
  } catch {
    return '';
  }
}

function createPinnedLookup(record){
  const pinned = {
    address: String(record && record.address || ''),
    family: Number(record && record.family) || net.isIP(record && record.address)
  };
  return (_hostname, options, callback) => {
    let lookupOptions = options;
    let done = callback;
    if (typeof lookupOptions === 'function') {
      done = lookupOptions;
      lookupOptions = {};
    }
    if (lookupOptions && lookupOptions.all) {
      done(null, [pinned]);
      return;
    }
    done(null, pinned.address, pinned.family);
  };
}

function createTimeoutError(){
  return createDestinationError('Request timed out', 'REQUEST_TIMEOUT');
}

function requestPinnedOnce(resolved, record, options){
  const parsed = resolved.parsed;
  const hostname = resolved.hostname;
  const transport = parsed.protocol === 'https:' ? https : http;
  const headers = Object.assign({}, options.headers, { host: parsed.host });
  const requestOptions = {
    protocol: parsed.protocol,
    hostname,
    port: parsed.port || undefined,
    path: `${parsed.pathname}${parsed.search}`,
    method: options.method,
    headers,
    agent: false,
    family: record.family,
    autoSelectFamily: false,
    lookup: createPinnedLookup(record)
  };

  if (parsed.protocol === 'https:' && net.isIP(hostname) === 0) {
    requestOptions.servername = hostname;
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let timer;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };

    const request = transport.request(requestOptions, response => {
      const result = {
        status: Number(response.statusCode) || 0,
        headers: response.headers || {},
        url: resolved.url,
        address: record.address
      };
      finish(resolve, result);
      response.destroy();
    });

    request.once('error', err => finish(reject, err));
    timer = setTimeout(() => request.destroy(createTimeoutError()), Math.max(1, options.timeoutMs));
    request.end();
  });
}

async function requestPinned(resolved, options, deadline){
  let lastError;
  for (let index = 0; index < resolved.addresses.length; index += 1) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) throw createTimeoutError();
    const remainingAddresses = resolved.addresses.length - index;
    const attemptTimeoutMs = Math.min(remainingMs, Math.max(250, Math.floor(remainingMs / remainingAddresses)));
    try {
      return await requestPinnedOnce(resolved, resolved.addresses[index], {
        method: options.method,
        headers: options.headers,
        timeoutMs: attemptTimeoutMs
      });
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error('Request failed');
}

function annotateRedirectError(err, status, url, redirected){
  const error = err instanceof Error ? err : new Error(String(err || 'Request failed'));
  error.status = Number(status) || 0;
  error.url = url || '';
  error.redirected = !!redirected;
  return error;
}

async function requestWithSafeRedirects(initialResolved, options){
  const deadline = Date.now() + options.timeoutMs;
  let resolved = initialResolved;
  let redirected = false;
  let redirectCount = 0;
  let previousStatus = 0;

  while (true) {
    let response;
    try {
      response = await requestPinned(resolved, options, deadline);
    } catch (err) {
      throw annotateRedirectError(err, previousStatus, resolved.url, redirected);
    }

    const redirectUrl = getManualRedirectUrl(response, resolved.url);
    if (!redirectUrl) {
      return Object.assign({}, response, { redirected, redirectCount });
    }

    if (redirectCount >= MAX_REDIRECTS) {
      throw annotateRedirectError(
        createDestinationError(`Too many redirects (maximum ${MAX_REDIRECTS})`, 'TOO_MANY_REDIRECTS'),
        response.status,
        redirectUrl,
        true
      );
    }

    previousStatus = response.status;
    redirected = true;
    redirectCount += 1;
    try {
      resolved = await resolveSafeDestinationUrl(redirectUrl);
    } catch (err) {
      throw annotateRedirectError(err, response.status, redirectUrl, true);
    }
  }
}

function failureResult(method, startedAt, err){
  return {
    ok: false,
    method,
    status: Number(err && err.status) || 0,
    url: err && err.url ? String(err.url) : '',
    redirected: !!(err && err.redirected),
    ms: Date.now() - startedAt,
    error: getErrorMessage(err) || 'Request failed'
  };
}

async function checkDestination(url){
  const headers = {
    'user-agent': 'Mozilla/5.0 (compatible; DanielShortShortlinksTest/1.0; +https://www.danielshort.me)',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
  };

  let resolved;
  try {
    resolved = await resolveSafeDestinationUrl(url);
  } catch (err) {
    return {
      ok: false,
      method: 'VALIDATE',
      status: 0,
      url: '',
      redirected: false,
      ms: 0,
      error: getErrorMessage(err) || 'Unsafe destination'
    };
  }

  const startedHead = Date.now();
  let headResult;
  try {
    headResult = await requestWithSafeRedirects(resolved, {
      method: 'HEAD',
      headers,
      timeoutMs: REQUEST_TIMEOUT_MS
    });
  } catch (err) {
    return failureResult('HEAD', startedHead, err);
  }

  if (headResult.status !== 405 && headResult.status !== 501) {
    return {
      ok: headResult.status >= 200 && headResult.status < 400,
      method: 'HEAD',
      status: headResult.status,
      url: headResult.url,
      redirected: headResult.redirected,
      ms: Date.now() - startedHead
    };
  }

  const startedGet = Date.now();
  try {
    const getResult = await requestWithSafeRedirects(resolved, {
      method: 'GET',
      headers: Object.assign({}, headers, { range: 'bytes=0-0' }),
      timeoutMs: REQUEST_TIMEOUT_MS
    });
    return {
      ok: getResult.status >= 200 && getResult.status < 400,
      method: 'GET',
      status: getResult.status,
      url: getResult.url,
      redirected: getResult.redirected,
      ms: Date.now() - startedGet
    };
  } catch (err) {
    return failureResult('GET', startedGet, err);
  }
}

async function handler(req, res, options = {}){
  const adminToken = getAdminToken();
  if (!adminToken) {
    sendJson(res, 503, { ok: false, error: 'SHORTLINKS_ADMIN_TOKEN is not configured' });
    return;
  }
  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const slug = normalizeSlug(getSlugFromRequest(req, options.slug));
  if (!slug) {
    sendJson(res, 400, { ok: false, error: 'Invalid slug' });
    return;
  }

  let link;
  try {
    link = await getLinkWithLegacyFallback(slug);
  } catch (err) {
    if (err && err.code === 'DDB_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return;
    }
    sendJson(res, 502, { ok: false, error: 'DynamoDB backend unavailable' });
    return;
  }

  if (!link || link.disabled) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
  if (expiresAt && Math.floor(Date.now() / 1000) >= expiresAt) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const base = getRequestBaseUrl(req);
  const destination = resolveDestination(link.destination, base);
  if (!destination) {
    sendJson(res, 404, { ok: false, error: 'Not Found' });
    return;
  }

  const statusCode = link.permanent ? 301 : 302;
  const check = await checkDestination(destination);

  sendJson(res, 200, {
    ok: true,
    slug,
    redirect: {
      statusCode,
      destination
    },
    check
  });
}

module.exports = handler;
module.exports._internal = {
  assertSafeDestinationUrl,
  checkDestination,
  createPinnedLookup,
  expandIpv6,
  getManualRedirectUrl,
  isBlockedAddress,
  isLocalHostname,
  isPrivateIpv4,
  isPrivateIpv6,
  mappedIpv4FromIpv6,
  requestWithSafeRedirects,
  resolveSafeDestinationUrl
};
