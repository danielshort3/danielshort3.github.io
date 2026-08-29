/* Service worker: offline shell + cache-first static assets.
   Never intercepts /api/ or third-party (analytics, GTM) traffic. */
'use strict';

const VERSION = 'ds-v1';
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

function shouldHandle(request) {
  try {
    const url = new URL(request.url);
    if (url.origin !== location.origin) return false;
    const path = url.pathname;
    if (path.startsWith('/api/')) return false;
    if (path.startsWith('/admin')) return false;
    if (SKIP.test(request.url)) return false;
    if (SKIP_HOSTS.includes(url.host)) return false;
    return request.method === 'GET';
  } catch (e) {
    return false;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (!shouldHandle(request)) return;

  // HTML documents: network-first, fall back to cache (offline shell).
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
          return response;
        })
        .catch(() =>
          caches.match(request).then((cached) =>
            cached || caches.match('/index.html')
          )
        )
    );
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
