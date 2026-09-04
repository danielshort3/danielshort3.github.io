/* Service worker: offline shell + cache-first static assets.
   Never intercepts /api/ or third-party (analytics, GTM) traffic. */
'use strict';

const VERSION = 'ds-v2';
const SHELL = [
  '/',
  '/index.html',
  '/portfolio',
  '/tools',
  '/games'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) =>
      Promise.allSettled(SHELL.map((url) => cache.add(url)))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

const SKIP = /^https?:\/\/(www\.)?(googletagmanager|google-analytics|fonts\.(googleapis|gstatic))\./i;
const SKIP_HOSTS = [
  'job-tracker-auth-886623862678.auth.us-east-2.amazoncognito.com',
  'cognito-identity.us-east-2.amazonaws.com',
  'dynamodb.us-east-2.amazonaws.com'
];

const HARD_DOCUMENT_PATHS = new Set([
  '/tools/background-remover',
  '/tools/transcribe',
  '/tools/job-application-tracker'
]);

function normalizePathname(value) {
  let path = String(value || '/');
  path = path.replace(/\/index\.html$/i, '/').replace(/\.html$/i, '').replace(/\/+$/, '');
  return path || '/';
}

function shouldHandle(request) {
  try {
    const url = new URL(request.url);
    if (url.origin !== location.origin) return false;
    const path = url.pathname;
    if (path.startsWith('/api/')) return false;
    if (path.startsWith('/admin')) return false;
    if (HARD_DOCUMENT_PATHS.has(normalizePathname(path))) return false;
    if (SKIP.test(request.url)) return false;
    if (SKIP_HOSTS.includes(url.host)) return false;
    return request.method === 'GET';
  } catch (e) {
    return false;
  }
}

function canStore(response) {
  if (!response || !response.ok || response.type === 'opaque') return false;
  const cacheControl = String(response.headers.get('cache-control') || '').toLowerCase();
  return !/(?:^|,)\s*(?:no-store|private)(?:\s|,|$)/.test(cacheControl);
}

async function cacheSuccessfulResponse(request, response) {
  if (!canStore(response)) return response;
  const cache = await caches.open(VERSION);
  await cache.put(request, response.clone());
  return response;
}

async function networkFirstDocument(request, options = {}) {
  try {
    const response = await fetch(request);
    await cacheSuccessfulResponse(request, response).catch(() => {});
    return response;
  } catch (error) {
    const exact = await caches.match(request);
    if (exact) return exact;
    if (options.allowRootFallback) {
      const url = new URL(request.url);
      if (normalizePathname(url.pathname) === '/') {
        const root = await caches.match('/index.html');
        if (root) return root;
      }
    }
    throw error;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (!shouldHandle(request)) return;

  // Router HTML requests use an exact network-first response. Never substitute
  // the homepage for another route; the router validates each route manifest.
  if (request.headers.get('X-Site-Route') === '1') {
    event.respondWith(networkFirstDocument(request));
    return;
  }

  // Full document navigations are also network-first. Only the root URL may
  // use the cached root shell when offline.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstDocument(request, { allowRootFallback: true }));
    return;
  }

  // Static assets (css/js/img/wasm/fonts): cache-first with background refresh.
  const asset = /\.(css|js|mjs|png|jpe?g|svg|webp|ico|woff2?|wasm|json|webmanifest)$/i.test(new URL(request.url).pathname);
  if (asset) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const network = fetch(request)
          .then((response) => {
            if (response.ok) {
              const copy = response.clone();
              caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
            }
            return response;
          })
          .catch(() => cached);
        return cached || network;
      })
    );
    return;
  }

  // Anything else same-origin GET: network with cache fallback.
  event.respondWith(
    fetch(request).catch(() => caches.match(request))
  );
});
