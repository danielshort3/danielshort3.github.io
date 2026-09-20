/* Public offline cache. Account/API traffic and private responses never enter it. */
'use strict';

const VERSION = 'ds-site-v3';
const DAY = 24 * 60 * 60 * 1000;
const MIB = 1024 * 1024;
const POLICIES = {
  documents: { entries: 32, bytes: 8 * MIB, entryBytes: MIB, age: 7 * DAY },
  core: { entries: 192, bytes: 16 * MIB, entryBytes: 8 * MIB, age: 30 * DAY },
  media: { entries: 96, bytes: 48 * MIB, entryBytes: 8 * MIB, age: 14 * DAY }
};
const SHELL = ['/', '/index.html', '/portfolio', '/tools', '/games'];
const HARD_DOCUMENT_PATHS = new Set([
  '/tools/background-remover', '/tools/transcribe', '/tools/job-application-tracker',
  '/tools/dashboard', '/tools-dashboard', '/pages/tools-dashboard'
]);
let maintenance = Promise.resolve();
let lastUse = 0;

function serialize(operation) {
  const result = maintenance.then(operation);
  maintenance = result.catch(() => {});
  return result;
}

function normalizePathname(value) {
  return String(value || '/').replace(/\/index\.html$/i, '/').replace(/\.html$/i, '').replace(/\/+$/, '') || '/';
}

function shouldHandle(request) {
  const url = new URL(request.url);
  const path = normalizePathname(url.pathname);
  return request.method === 'GET' && url.origin === location.origin
    && !request.headers.has('Authorization') && !request.headers.has('Range')
    // OAuth and shared-state URLs can contain credentials or private input in
    // their cache keys, even when the returned HTML is a public static shell.
    && ![...url.searchParams.keys()].some(key => /^(?:code|state|token|access_token|id_token|refresh_token)$/i.test(key))
    && path !== '/api' && !path.startsWith('/api/') && !path.startsWith('/admin')
    && !HARD_DOCUMENT_PATHS.has(path)
    && !HARD_DOCUMENT_PATHS.has(path.replace(/^\/pages\//, '/tools/'));
}

function hasPrivatePolicy(response) {
  return /(?:^|,)\s*(?:no-store|private)(?:\s*(?:=|,|$))/i.test(response.headers.get('Cache-Control') || '')
    || (response.headers.get('Vary') || '').split(',').some(value => value.trim() === '*');
}

function canStore(response) {
  return Boolean(response && response.status === 200 && !response.redirected
    && response.type !== 'opaque' && response.type !== 'opaqueredirect' && !hasPrivatePolicy(response));
}

function bucketFor(request) {
  if (request.mode === 'navigate' || request.headers.get('X-Site-Route') === '1') return 'documents';
  const path = new URL(request.url).pathname;
  if (/\.(png|jpe?g|svg|webp|avif|gif|bmp|ico|wasm)$/i.test(path)) return 'media';
  if (/\.(css|js|mjs|woff2?|json|webmanifest)$/i.test(path)) return 'core';
  return null;
}

function isVersioned(request) {
  const url = new URL(request.url);
  return /\.[a-f0-9]{8,64}\.(?:css|js|mjs)$/i.test(url.pathname)
    || /^[a-f0-9]{8,64}$/i.test(url.searchParams.get('v') || '');
}

function metadataKey(bucket) {
  return `${location.origin}/__site-cache-metadata__/${bucket}`;
}

async function loadBucket(bucket) {
  const cache = await caches.open(`${VERSION}-${bucket}`);
  const metadata = await caches.open(`${VERSION}-metadata`);
  const record = await metadata.match(metadataKey(bucket));
  let index = [];
  try { index = record ? await record.json() : []; } catch {}
  if (!Array.isArray(index)) index = [];
  index = index.filter(entry => entry && typeof entry.url === 'string'
    && entry.url.startsWith(`${location.origin}/`) && Number.isFinite(entry.bytes) && entry.bytes >= 0
    && Number.isFinite(entry.storedAt) && Number.isFinite(entry.usedAt));
  // Recover interrupted writes without trusting responses that have no age/size record.
  const keys = await cache.keys();
  const present = new Set(keys.map(key => key.url));
  const recorded = new Set(index.map(entry => entry.url));
  await Promise.all(keys.filter(key => !recorded.has(key.url)).map(key => cache.delete(key, { ignoreVary: true })));
  return { cache, metadata, index: index.filter(entry => present.has(entry.url)), bucket };
}

async function writeIndex(state) {
  await state.metadata.put(metadataKey(state.bucket), new Response(JSON.stringify(state.index), {
    headers: { 'Content-Type': 'application/json' }
  }));
}

async function removeEntry(state, entry) {
  // Metadata budgets one body per URL. Remove all Vary variants together.
  await state.cache.delete(entry.url, { ignoreVary: true });
  state.index = state.index.filter(item => item.url !== entry.url);
}

async function prune(state, reserve = { entries: 0, bytes: 0 }) {
  const policy = POLICIES[state.bucket];
  const now = Date.now();
  for (const entry of [...state.index]) {
    if (now - entry.storedAt >= policy.age || entry.storedAt > now + DAY || entry.bytes > policy.entryBytes) {
      await removeEntry(state, entry);
    }
  }
  state.index.sort((a, b) => a.usedAt - b.usedAt);
  let size = state.index.reduce((sum, entry) => sum + entry.bytes, 0);
  while (state.index.length && (state.index.length + reserve.entries > policy.entries || size + reserve.bytes > policy.bytes)) {
    const entry = state.index[0];
    size -= entry.bytes;
    await removeEntry(state, entry);
  }
}

async function removeCached(bucket, request) {
  return serialize(async () => {
    const state = await loadBucket(bucket);
    const entry = state.index.find(item => item.url === request.url);
    if (entry) await removeEntry(state, entry);
    await writeIndex(state);
  }).catch(() => {});
}

async function readCached(bucket, request) {
  return serialize(async () => {
    const state = await loadBucket(bucket);
    await prune(state);
    const entry = state.index.find(item => item.url === request.url);
    const response = entry ? await state.cache.match(request) : undefined;
    if (response) entry.usedAt = lastUse = Math.max(Date.now(), lastUse + 1);
    await writeIndex(state);
    return response;
  }).catch(() => undefined);
}

async function responseSize(response, limit) {
  const declared = Number(response.headers.get('Content-Length'));
  if (Number.isFinite(declared) && declared > limit) {
    response.body?.cancel().catch(() => {});
    return null;
  }
  if (!response.body) return 0;
  const reader = response.body.getReader();
  let bytes = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) return bytes;
      bytes += next.value.byteLength;
      if (bytes > limit) {
        // Do not await cancellation of a tee: the visitor still owns the other branch.
        reader.cancel().catch(() => {});
        return null;
      }
    }
  } finally { reader.releaseLock(); }
}

async function freeQuota() {
  for (const bucket of Object.keys(POLICIES)) {
    const state = await loadBucket(bucket);
    await prune(state);
    state.index.sort((a, b) => a.usedAt - b.usedAt);
    const count = Math.ceil(state.index.length / 4);
    for (const entry of state.index.slice(0, count)) await removeEntry(state, entry);
    await writeIndex(state);
  }
}

async function storeResponse(bucket, request, response) {
  if (!canStore(response)) {
    if (response && hasPrivatePolicy(response)) await removeCached(bucket, request);
    return false;
  }
  // Clone before yielding: the page can consume its network response immediately.
  const saved = response.clone();
  const measured = response.clone();
  let bytes;
  try { bytes = await responseSize(measured, POLICIES[bucket].entryBytes); } catch {
    saved.body?.cancel().catch(() => {});
    return false;
  }
  if (bytes === null) {
    saved.body?.cancel().catch(() => {});
    await removeCached(bucket, request);
    return false;
  }
  return serialize(async () => {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const state = await loadBucket(bucket);
      const previous = state.index.find(entry => entry.url === request.url);
      if (previous) await removeEntry(state, previous);
      await prune(state, { entries: 1, bytes });
      try {
        await state.cache.put(request, saved.clone());
        const now = Date.now();
        state.index.push({ url: request.url, bytes, storedAt: now, usedAt: lastUse = Math.max(now, lastUse + 1) });
        await writeIndex(state);
        return true;
      } catch (error) {
        // A failed metadata write must not leave an unbounded orphan body.
        await state.cache.delete(request, { ignoreVary: true }).catch(() => {});
        if (error.name !== 'QuotaExceededError' || attempt) return false;
        await freeQuota();
      }
    }
    return false;
  }).catch(() => false).finally(() => { saved.body?.cancel().catch(() => {}); });
}

function isOwnedCache(name) {
  return /^ds-v[12]$/.test(name) || /^ds-site-v\d+-(?:documents|core|media|metadata)$/.test(name);
}

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const root = new Request(`${location.origin}/`);
    if (!await storeResponse('documents', root, await fetch(root))) throw new Error('Offline shell unavailable');
    await Promise.allSettled(SHELL.slice(1).map(async path => {
      const request = new Request(new URL(path, location.origin));
      await storeResponse('documents', request, await fetch(request));
    }));
  })());
  // Updates wait for existing pages to close; never interrupt an active game or recording.
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const current = new Set([...Object.keys(POLICIES), 'metadata'].map(bucket => `${VERSION}-${bucket}`));
    await Promise.all((await caches.keys()).filter(name => isOwnedCache(name) && !current.has(name)).map(name => caches.delete(name)));
    await serialize(async () => {
      for (const bucket of Object.keys(POLICIES)) {
        const state = await loadBucket(bucket);
        await prune(state);
        await writeIndex(state);
      }
    }).catch(() => {});
    await self.clients.claim();
  })());
});

function networkResponse(request, bucket) {
  const network = fetch(request);
  const stored = network.then(response => storeResponse(bucket, request, response)).catch(() => false);
  return { network, stored };
}

self.addEventListener('fetch', event => {
  const request = event.request;
  if (!shouldHandle(request)) return;
  const bucket = bucketFor(request);
  if (!bucket) return;

  if (bucket === 'documents') {
    const { network, stored } = networkResponse(request, bucket);
    event.waitUntil(stored);
    event.respondWith(network.catch(async error => {
      const exact = await readCached(bucket, request);
      if (exact) return exact;
      if (request.mode === 'navigate' && normalizePathname(new URL(request.url).pathname) === '/') {
        const root = await readCached(bucket, new Request(`${location.origin}/index.html`));
        if (root) return root;
      }
      throw error;
    }));
    return;
  }

  const cached = readCached(bucket, request);
  const delivery = cached.then(async hit => {
    if (hit && isVersioned(request)) return { response: hit, stored: Promise.resolve() };
    const { network, stored } = networkResponse(request, bucket);
    return { response: hit || await network, stored };
  });
  // Register immediately, even when the cache lookup resolves asynchronously.
  event.waitUntil(delivery.then(result => result.stored).catch(() => {}));
  event.respondWith(delivery.then(result => result.response));
});
