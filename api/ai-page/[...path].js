/*
  Serve deterministic AI digests generated at build time.
  Public debug route: /ai/<path> -> /api/ai-page/<path>?debug=1
  AI-agent route rewrites keep the browser-visible canonical URL.
*/
'use strict';

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..', '..');
const manifestCandidates = [
  path.join(root, 'dist', 'ai-digest-manifest.json'),
  path.join(root, 'public', 'dist', 'ai-digest-manifest.json'),
  path.join(process.cwd(), 'dist', 'ai-digest-manifest.json'),
  path.join(process.cwd(), 'public', 'dist', 'ai-digest-manifest.json')
];

let cachedManifest = null;

function loadManifest() {
  if (cachedManifest) return cachedManifest;
  for (const filePath of manifestCandidates) {
    try {
      if (!fs.existsSync(filePath)) continue;
      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      if (parsed && parsed.routes && typeof parsed.routes === 'object') {
        cachedManifest = { manifest: parsed, baseDir: path.dirname(path.dirname(filePath)) };
        return cachedManifest;
      }
    } catch {}
  }
  cachedManifest = { manifest: null, baseDir: process.cwd() };
  return cachedManifest;
}

function getRequestBaseUrl(req) {
  const proto = req && req.headers && (req.headers['x-forwarded-proto'] || req.headers['x-forwarded-protocol'])
    ? String(req.headers['x-forwarded-proto'] || req.headers['x-forwarded-protocol']).split(',')[0].trim()
    : 'https';
  const host = req && req.headers && req.headers.host ? String(req.headers.host) : 'www.danielshort.me';
  return `${proto || 'https'}://${host || 'www.danielshort.me'}`;
}

function getRouteFromRequest(req) {
  const queryPath = req.query && (req.query.path || req.query.slug);
  if (Array.isArray(queryPath)) return queryPath.join('/');
  if (typeof queryPath === 'string') return queryPath;

  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    const match = url.pathname.match(/\/api\/ai-page\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : '';
  } catch {
    return '';
  }
}

function normalizeRoutePath(value) {
  const raw = String(value || '').trim().replace(/\\/g, '/');
  if (!raw || raw.includes('\0')) return '';
  const withoutQuery = raw.split('#')[0].split('?')[0];
  const parts = withoutQuery
    .replace(/^\/+|\/+$/g, '')
    .split('/')
    .filter(Boolean);
  if (!parts.length) return '';
  if (parts.some((part) => part === '..' || part === '.')) return '';
  let route = `/${parts.join('/')}`;
  if (route !== '/' && route.endsWith('.html')) route = route.slice(0, -5);
  return route;
}

function isDebugRequest(req) {
  try {
    const url = new URL(req.url, getRequestBaseUrl(req));
    return url.searchParams.get('debug') === '1';
  } catch {
    return false;
  }
}

function sendText(res, statusCode, body) {
  const text = String(body || '');
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex');
  res.end(text);
}

function resolveDigestFile(baseDir, outputPath) {
  const rel = String(outputPath || '').replace(/\\/g, '/');
  if (!rel || rel.includes('..') || path.isAbsolute(rel)) return '';
  const filePath = path.join(baseDir, rel);
  const relative = path.relative(baseDir, filePath);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return '';
  return filePath;
}

module.exports = async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET, HEAD');
    res.end('Method Not Allowed');
    return;
  }

  const routePath = normalizeRoutePath(getRouteFromRequest(req));
  if (!routePath) {
    sendText(res, 404, 'AI digest not found');
    return;
  }

  const { manifest, baseDir } = loadManifest();
  const route = manifest && manifest.routes ? manifest.routes[routePath] : null;
  if (!route || !route.outputPath) {
    sendText(res, 404, 'AI digest not found');
    return;
  }

  const filePath = resolveDigestFile(baseDir, route.outputPath);
  if (!filePath || !fs.existsSync(filePath)) {
    sendText(res, 404, 'AI digest not found');
    return;
  }

  let html;
  try {
    html = fs.readFileSync(filePath, 'utf8');
  } catch {
    sendText(res, 500, 'AI digest unavailable');
    return;
  }

  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.setHeader('Cache-Control', 'public, max-age=300, stale-while-revalidate=86400');
  res.setHeader('Vary', 'User-Agent');
  res.setHeader('X-AI-Digest', '1');
  if (route.canonicalUrl) res.setHeader('Link', `<${route.canonicalUrl}>; rel="canonical"`);
  if (isDebugRequest(req)) res.setHeader('X-Robots-Tag', 'noindex');
  if (req.method === 'HEAD') {
    res.setHeader('Content-Length', String(Buffer.byteLength(html)));
    res.end();
    return;
  }
  res.end(html);
};
