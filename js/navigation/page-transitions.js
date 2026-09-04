/* ===================================================================
   File: page-transitions.js
   Purpose: Persistent-shell navigation for personal site routes.
=================================================================== */
(() => {
  'use strict';

  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  const CONTENT_SELECTOR = '[data-site-route-content], [data-personal-detail-content]';
  const MANIFEST_SELECTOR = 'script#site-route-manifest[data-site-route-manifest]';
  const SHELL_SELECTOR = '[data-site-persistent-shell], [data-personal-accordion-shell], [data-home-accordion]';
  const PROGRESS_DELAY_MS = 350;
  const ROUTE_CACHE_LIMIT = 12;
  const ROUTE_EVENT = 'site:route-change';
  const CONTENT_EVENT = 'site:content-updated';
  const NAVIGATION_EVENT = 'site:navigation-start';
  const REQUEST_HEADER = 'X-Site-Route';
  const ROUTE_VIEWS = new Set(['overview', 'library', 'detail']);
  const ROUTE_CATEGORIES = new Set(['about', 'projects', 'tools', 'games', 'contact']);
  const PROFESSIONAL_AUDIENCES = new Set(['analytics', 'data-science', 'tourism']);
  const HARD_BOUNDARY_PATHS = new Set([
    '/tools/background-remover',
    '/tools/transcribe',
    '/tools/job-application-tracker'
  ]);
  const PERSISTENT_BODY_CLASSES = new Set([
    'consent-blocked',
    'has-mobile-site-dock',
    'has-mobile-site-masthead',
    'is-mobile-site-dock-hidden'
  ]);
  const ROUTE_DATA_PREFIXES = [
    'data-site-route-',
    'data-personal-',
    'data-tools-',
    'data-game-'
  ];
  const ROUTE_RUNTIME_BODY_ATTRIBUTES = new Set([
    'data-contact-audience',
    'data-performance-tier',
    'data-project-demo-autosize',
    'data-qrtool-ui-mode',
    'data-shortlinks-mode'
  ]);
  const PERSISTENT_SCRIPT_PATTERNS = [
    /\/js\/common\/no-js\.js$/i,
    /\/(?:dist\/)?site-(?:shell|consent|tools-account)(?:[.-]|$)/i,
    /\/js\/common\/(?:no-js|common|audience-config|site-realm|modal-accessibility|certifications-modal)(?:[.-]|$)/i,
    /\/js\/navigation\/navigation(?:[.-]|$)/i,
    /\/js\/animations\/animations(?:[.-]|$)/i,
    /\/js\/(?:analytics|privacy)\//i
  ];

  const routeCache = new Map();
  const pendingFetches = new Map();
  const loadedScripts = new Set();
  let activeNavigation = null;
  let navigationSequence = 0;
  let historySequence = 0;
  let committedHistoryIndex = 0;
  let committedHistoryState = null;
  let committedUrl = window.location.href;
  let restoringVetoedPop = false;

  Array.from(document.scripts || []).forEach((script) => {
    if (script.src) loadedScripts.add(normalizeAssetUrl(script.src));
  });

  function resolveUrl(value, base = document.baseURI || window.location.href) {
    try {
      return new URL(String(value || ''), base);
    } catch (_) {
      return null;
    }
  }

  function normalizePathname(value) {
    let pathname = String(value || '/');
    pathname = pathname.replace(/\/index\.html$/i, '/');
    pathname = pathname.replace(/\.html$/i, '');
    pathname = pathname.replace(/\/+$/, '');
    return pathname || '/';
  }

  function normalizeRouteUrl(url) {
    if (!url) return '';
    return `${url.origin}${normalizePathname(url.pathname)}${url.search}`;
  }

  function normalizeAssetUrl(value, base) {
    const url = resolveUrl(value, base);
    if (!url) return '';
    url.hash = '';
    return url.href;
  }

  function getDocumentAssetBase(scope, routeUrl) {
    const originBase = routeUrl ? `${routeUrl.origin}/` : document.baseURI || window.location.href;
    const declared = scope?.querySelector?.('base[href]')?.getAttribute('href');
    return resolveUrl(declared || '/', originBase)?.href || originBase;
  }

  function makeAbortError() {
    try {
      return new DOMException('Navigation was superseded.', 'AbortError');
    } catch (_) {
      const error = new Error('Navigation was superseded.');
      error.name = 'AbortError';
      return error;
    }
  }

  function isAbortError(error) {
    return Boolean(error && error.name === 'AbortError');
  }

  function throwIfAborted(signal) {
    if (signal?.aborted) throw signal.reason || makeAbortError();
  }

  function dispatch(name, detail = {}, options = {}) {
    let event;
    try {
      event = new CustomEvent(name, {
        bubbles: Boolean(options.bubbles),
        cancelable: Boolean(options.cancelable),
        detail
      });
    } catch (_) {
      event = document.createEvent('CustomEvent');
      event.initCustomEvent(name, Boolean(options.bubbles), Boolean(options.cancelable), detail);
    }
    document.dispatchEvent(event);
    return event;
  }

  function prefersReducedMotion() {
    try {
      return Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
    } catch (_) {
      return false;
    }
  }

  function isHardBoundary(url) {
    return Boolean(url && HARD_BOUNDARY_PATHS.has(normalizePathname(url.pathname)));
  }

  function isProfessionalAudienceUrl(url) {
    if (!url) return false;
    const path = normalizePathname(url.pathname);
    if (/^\/(?:professional|analytics|data-science|tourism)(?:\/|$)/i.test(path)) return true;
    const audience = String(url.searchParams.get('audience') || '').trim().toLowerCase();
    const mode = String(url.searchParams.get('mode') || '').trim().toLowerCase();
    return PROFESSIONAL_AUDIENCES.has(audience) ||
      ['professional', 'work', 'career', 'analytics'].includes(mode);
  }

  function getPersonalRouteIntent(url) {
    if (!url || isProfessionalAudienceUrl(url) || isHardBoundary(url)) return null;
    const path = normalizePathname(url.pathname);
    if (/^\/(?:api|admin|documents|demos)(?:\/|$)/i.test(path)) return null;

    if (path === '/') {
      let category = String(url.hash || '').replace(/^#/, '') || 'about';
      try {
        category = decodeURIComponent(category);
      } catch (_) {
        category = 'about';
      }
      if (!ROUTE_CATEGORIES.has(category)) return null;
      const library = url.searchParams.get('view') === 'library' &&
        ['projects', 'tools', 'games'].includes(category);
      return { category, view: library ? 'library' : 'overview' };
    }

    if (/^\/[a-z0-9-]+-demo$/i.test(path)) {
      return { category: 'projects', view: 'detail' };
    }
    if (path === '/portfolio' || path.startsWith('/portfolio/')) {
      return { category: 'projects', view: path === '/portfolio' ? 'library' : 'detail' };
    }
    if (path === '/tools' || path.startsWith('/tools/')) {
      return { category: 'tools', view: path === '/tools' ? 'library' : 'detail' };
    }
    if (path === '/games' || path.startsWith('/games/')) {
      return { category: 'games', view: path === '/games' ? 'library' : 'detail' };
    }
    if (path === '/contact') return { category: 'contact', view: 'detail' };
    if (path === '/job-application-copilot' || path === '/job-application-copilot/privacy') {
      return { category: 'tools', view: 'detail' };
    }
    if (['/privacy', '/sitemap', '/sitemap-pretty'].includes(path)) {
      return { category: 'about', view: 'detail' };
    }
    if (path === '/search') return { category: 'tools', view: 'detail' };
    if (path === '/solutions') return { category: 'projects', view: 'detail' };
    return null;
  }

  function getHomepageHistoryIntent(state) {
    if (!state || typeof state !== 'object') return null;
    if (String(state.siteRoute?.id || '').trim() !== 'home') return null;
    const category = String(state.homePanel || '').trim();
    const view = String(state.homeView || '').trim();
    if (!ROUTE_CATEGORIES.has(category) || !['overview', 'library'].includes(view)) return null;
    if (view === 'library' && !['projects', 'tools', 'games'].includes(category)) return null;
    return { category, view };
  }

  function isDocumentLikeUrl(url) {
    if (!url || !/^https?:$/i.test(url.protocol) || url.origin !== window.location.origin) return false;
    const segment = url.pathname.split('/').pop() || '';
    const dot = segment.lastIndexOf('.');
    const extension = dot > 0 ? segment.slice(dot + 1).toLowerCase() : '';
    return !extension || extension === 'html' || extension === 'htm';
  }

  function hasBlockingInteractionLayer() {
    return Boolean(
      document.fullscreenElement ||
      document.pointerLockElement ||
      document.body?.classList.contains('modal-open') ||
      document.body?.classList.contains('media-viewer-open') ||
      document.querySelector('dialog[open], .modal.active, [data-tools-account-modal][aria-hidden="false"]')
    );
  }

  function readRouteManifest(scope, url, options = {}) {
    const node = scope?.querySelector?.(MANIFEST_SELECTOR);
    if (!node) return null;
    let value;
    try {
      value = JSON.parse(node.textContent || '{}');
    } catch (error) {
      if (options.strict) throw new Error('The route manifest is not valid JSON.', { cause: error });
      return null;
    }
    if (!value || typeof value !== 'object') return null;
    const base = getDocumentAssetBase(scope, url);
    const normalizeList = (items) => {
      if (!Array.isArray(items)) return [];
      const unique = new Set();
      items.forEach((item) => {
        const normalized = normalizeAssetUrl(item, base);
        if (normalized) unique.add(normalized);
      });
      return Array.from(unique);
    };
    const manifest = {
      version: Number(value.version),
      id: String(value.id || '').trim(),
      path: normalizePathname(value.path || url?.pathname || '/'),
      category: String(value.category || '').trim(),
      view: String(value.view || '').trim(),
      navigation: String(value.navigation || '').trim().toLowerCase(),
      styles: normalizeList(value.styles),
      scripts: normalizeList(value.scripts),
      module: String(value.module || '').trim()
    };
    if (!options.strict) return manifest;
    if (manifest.version !== 1 || !manifest.id) throw new Error('The route manifest has an unsupported version or no id.');
    if (manifest.navigation !== 'soft') throw new Error('The destination is not a soft-navigation route.');
    if (!ROUTE_CATEGORIES.has(manifest.category) || !ROUTE_VIEWS.has(manifest.view)) {
      throw new Error('The route manifest has an invalid category or view.');
    }
    if (!url || manifest.path !== normalizePathname(url.pathname)) {
      throw new Error('The route manifest does not match the requested path.');
    }
    if (isHardBoundary(url) || isProfessionalAudienceUrl(url)) {
      throw new Error('The destination requires document navigation.');
    }
    return manifest;
  }

  function isCurrentRouteSoft() {
    const currentUrl = resolveUrl(window.location.href);
    const manifest = readRouteManifest(document, currentUrl, { strict: false });
    if (manifest) return manifest.navigation === 'soft' && !isHardBoundary(currentUrl);
    return Boolean(
      document.body?.dataset.audience === 'personal' &&
      document.querySelector(SHELL_SELECTOR) &&
      document.querySelector(CONTENT_SELECTOR)
    );
  }

  function getEligibleLinkUrl(link) {
    if (!link || link.closest('[data-contact-modal-link]')) return null;
    if (link.hasAttribute('download')) return null;
    if (link.dataset.navigation === 'hard' || link.closest('[data-navigation="hard"]')) return null;
    if (link.dataset.pageTransition === 'false') return null;
    const target = String(link.getAttribute('target') || '').trim().toLowerCase();
    if (target && target !== '_self') return null;
    const href = String(link.getAttribute('href') || '').trim();
    if (!href || href.startsWith('#') || /^(?:mailto:|tel:|javascript:)/i.test(href)) return null;
    const url = resolveUrl(href);
    if (!isDocumentLikeUrl(url) || !getPersonalRouteIntent(url) || !isCurrentRouteSoft()) return null;
    const current = resolveUrl(window.location.href);
    if (current && normalizeRouteUrl(current) === normalizeRouteUrl(url)) return null;
    return url;
  }

  function getRouteOutlet(scope) {
    return scope?.querySelector?.(CONTENT_SELECTOR) || null;
  }

  function putRouteCache(key, value) {
    routeCache.delete(key);
    routeCache.set(key, value);
    while (routeCache.size > ROUTE_CACHE_LIMIT) {
      routeCache.delete(routeCache.keys().next().value);
    }
  }

  function getRouteCache(key) {
    if (!routeCache.has(key)) return null;
    const value = routeCache.get(key);
    routeCache.delete(key);
    routeCache.set(key, value);
    return value;
  }

  function raceWithAbort(promise, signal) {
    if (!signal) return promise;
    throwIfAborted(signal);
    return new Promise((resolve, reject) => {
      const abort = () => reject(signal.reason || makeAbortError());
      signal.addEventListener('abort', abort, { once: true });
      promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
    });
  }

  async function fetchRouteDocument(url, options = {}) {
    const key = normalizeRouteUrl(url);
    const cached = getRouteCache(key);
    if (cached) {
      try {
        return parseRouteResponse(cached, url);
      } catch (error) {
        routeCache.delete(key);
        throw error;
      }
    }

    let pending = pendingFetches.get(key);
    if (!pending) {
      const controller = new AbortController();
      const abortFromNavigation = () => controller.abort(options.signal?.reason || makeAbortError());
      options.signal?.addEventListener('abort', abortFromNavigation, { once: true });
      const promise = fetch(url.href, {
        cache: options.prefetch ? 'force-cache' : 'default',
        credentials: 'same-origin',
        headers: { Accept: 'text/html', [REQUEST_HEADER]: '1' },
        redirect: 'follow',
        signal: controller.signal
      }).then(async (response) => {
        if (!response.ok) throw new Error(`Route request failed with ${response.status}.`);
        const responseUrl = resolveUrl(response.url || url.href);
        if (!responseUrl || responseUrl.origin !== window.location.origin) {
          throw new Error('Route request left the current origin.');
        }
        const type = String(response.headers?.get?.('content-type') || '').toLowerCase();
        if (type && !type.includes('text/html')) throw new Error('Route response was not HTML.');
        const html = await response.text();
        const payload = { html, responseUrl: responseUrl.href };
        putRouteCache(key, payload);
        return payload;
      }).finally(() => {
        options.signal?.removeEventListener('abort', abortFromNavigation);
        if (pendingFetches.get(key)?.promise === promise) pendingFetches.delete(key);
      });
      pending = { controller, promise };
      pendingFetches.set(key, pending);
    }
    let payload;
    try {
      payload = await raceWithAbort(pending.promise, options.signal);
    } catch (error) {
      const retryAbortedPending = options.retryAbortedPending !== false &&
        isAbortError(error) && !options.signal?.aborted;
      if (!retryAbortedPending) throw error;
      if (pendingFetches.get(key) === pending) pendingFetches.delete(key);
      return fetchRouteDocument(url, { ...options, retryAbortedPending: false });
    }
    try {
      return parseRouteResponse(payload, url);
    } catch (error) {
      routeCache.delete(key);
      throw error;
    }
  }

  function parseRouteResponse(payload, requestedUrl) {
    if (!payload || typeof payload.html !== 'string') throw new Error('Route response was empty.');
    if (typeof DOMParser !== 'function') throw new Error('This browser cannot parse route documents.');
    const parsed = new DOMParser().parseFromString(payload.html, 'text/html');
    if (!parsed?.documentElement || parsed.querySelector('parsererror')) {
      throw new Error('Route response could not be parsed.');
    }
    const manifest = readRouteManifest(parsed, requestedUrl, { strict: true });
    const outlet = getRouteOutlet(parsed);
    const shell = parsed.querySelector(SHELL_SELECTOR);
    if (!manifest || !outlet || !shell) throw new Error('Route response is missing its persistent-shell contract.');
    return { document: parsed, manifest, outlet, url: requestedUrl };
  }

  function prefetchRoute(url) {
    if (!url || !isCurrentRouteSoft()) return Promise.resolve(null);
    return fetchRouteDocument(url, { prefetch: true }).catch(() => null);
  }

  function isPersistentScript(src) {
    const url = resolveUrl(src);
    if (!url) return true;
    return PERSISTENT_SCRIPT_PATTERNS.some((pattern) => pattern.test(url.pathname));
  }

  function routeScriptsFor(manifest) {
    return manifest.scripts.filter((src) => !isPersistentScript(src));
  }

  function loadScript(src, targetDocument, targetUrl, signal) {
    const normalized = normalizeAssetUrl(src);
    if (!normalized) return Promise.resolve();
    const alreadyPresent = Array.from(document.scripts || []).some((script) =>
      normalizeAssetUrl(script.getAttribute('src') || script.src) === normalized
    );
    if (alreadyPresent) loadedScripts.add(normalized);
    if (loadedScripts.has(normalized)) return Promise.resolve();
    throwIfAborted(signal);
    return new Promise((resolve, reject) => {
      const source = Array.from(targetDocument.scripts || []).find((node) =>
        normalizeAssetUrl(node.getAttribute('src') || node.src, getDocumentAssetBase(targetDocument, targetUrl)) === normalized
      );
      const script = document.createElement('script');
      if (source) {
        Array.from(source.attributes || []).forEach((attribute) => {
          if (!['src', 'defer', 'async'].includes(attribute.name.toLowerCase())) {
            script.setAttribute(attribute.name, attribute.value);
          }
        });
      }
      script.src = normalized;
      script.async = false;
      script.dataset.siteRouteScript = 'true';
      const cleanup = () => signal?.removeEventListener('abort', abort);
      const abort = () => {
        script.remove();
        cleanup();
        reject(signal.reason || makeAbortError());
      };
      script.addEventListener('load', () => {
        loadedScripts.add(normalized);
        cleanup();
        resolve();
      }, { once: true });
      script.addEventListener('error', () => {
        script.remove();
        cleanup();
        reject(new Error(`Unable to load route script ${normalized}.`));
      }, { once: true });
      signal?.addEventListener('abort', abort, { once: true });
      document.head.appendChild(script);
    });
  }

  async function loadScriptsInOrder(scripts, targetDocument, targetUrl, signal) {
    for (const src of scripts) await loadScript(src, targetDocument, targetUrl, signal);
  }

  async function preloadScriptBytes(scripts, signal) {
    await Promise.all(scripts.map(async (src) => {
      throwIfAborted(signal);
      const response = await fetch(src, {
        cache: 'force-cache',
        credentials: 'same-origin',
        signal
      });
      if (!response.ok) throw new Error(`Unable to preload route script ${src}.`);
      await response.arrayBuffer();
    }));
  }

  function registeredRouteKey(manifest) {
    const runtime = window.SiteRoutes;
    if (!runtime?.get) return '';
    if (manifest.module && runtime.get(manifest.module)) return manifest.module;
    if (runtime.get(manifest.id)) return manifest.id;
    return '';
  }

  async function prepareRouteScripts(route, signal) {
    const runtime = window.SiteRoutes;
    if (!runtime?.mount || !runtime?.get) throw new Error('The route lifecycle runtime is unavailable.');
    const persistentScripts = route.manifest.scripts.filter((src) => isPersistentScript(src));
    const scripts = routeScriptsFor(route.manifest);
    await loadScriptsInOrder(persistentScripts, route.document, route.url, signal);
    let routeKey = registeredRouteKey(route.manifest);

    if (!routeKey) {
      if (typeof runtime.ensureLegacyRoute !== 'function') {
        throw new Error(`Route ${route.manifest.id} has no lifecycle module.`);
      }
      runtime.ensureLegacyRoute(route.manifest.id, { scripts });
      routeKey = registeredRouteKey(route.manifest);
      if (!routeKey) throw new Error(`Route ${route.manifest.id} could not install a lifecycle adapter.`);
    }
    await preloadScriptBytes(scripts, signal);
    return registeredRouteKey(route.manifest) || routeKey;
  }

  function findStylesheet(href) {
    return Array.from(document.querySelectorAll('link[rel~="stylesheet"][href]')).find((link) =>
      normalizeAssetUrl(link.href || link.getAttribute('href')) === href
    ) || null;
  }

  function loadStylesheet(href, targetDocument, signal) {
    if (findStylesheet(href)) return Promise.resolve({ link: null, added: false });
    throwIfAborted(signal);
    return new Promise((resolve, reject) => {
      const targetLink = Array.from(targetDocument.querySelectorAll('link[rel~="stylesheet"][href]')).find((link) =>
        normalizeAssetUrl(link.getAttribute('href') || link.href, getDocumentAssetBase(targetDocument)) === href
      );
      const link = document.createElement('link');
      if (targetLink) {
        Array.from(targetLink.attributes || []).forEach((attribute) => {
          if (!['href', 'rel'].includes(attribute.name.toLowerCase())) {
            link.setAttribute(attribute.name, attribute.value);
          }
        });
      }
      link.rel = 'stylesheet';
      link.href = href;
      link.dataset.siteRouteStyle = 'true';
      const intendedMedia = targetLink?.getAttribute('media');
      link.dataset.siteRouteMedia = intendedMedia || '';
      link.media = 'not all';
      const cleanup = () => signal?.removeEventListener('abort', abort);
      const abort = () => {
        link.remove();
        cleanup();
        reject(signal.reason || makeAbortError());
      };
      link.addEventListener('load', () => {
        cleanup();
        resolve({ link, added: true });
      }, { once: true });
      link.addEventListener('error', () => {
        link.remove();
        cleanup();
        reject(new Error(`Unable to load route stylesheet ${href}.`));
      }, { once: true });
      signal?.addEventListener('abort', abort, { once: true });
      document.head.appendChild(link);
    });
  }

  async function prepareRouteStyles(route, signal) {
    const loaded = await Promise.all(route.manifest.styles.map((href) =>
      loadStylesheet(href, route.document, signal)
    ));
    return loaded.filter((entry) => entry.added).map((entry) => entry.link);
  }

  function activatePreparedStyles(links) {
    links.forEach((link) => {
      if (!link) return;
      const media = link.dataset.siteRouteMedia || '';
      delete link.dataset.siteRouteMedia;
      if (media) link.media = media;
      else link.removeAttribute('media');
    });
  }

  function retireOldStyles(previousManifest, nextManifest) {
    if (!previousManifest) return;
    const keep = new Set(nextManifest.styles);
    previousManifest.styles.forEach((href) => {
      if (!keep.has(href)) findStylesheet(href)?.remove();
    });
  }

  function syncBody(targetBody) {
    const body = document.body;
    if (!body || !targetBody) return;
    const persistentClasses = Array.from(body.classList).filter((name) => PERSISTENT_BODY_CLASSES.has(name));
    body.className = Array.from(new Set([...Array.from(targetBody.classList), ...persistentClasses])).join(' ');

    Array.from(body.attributes).forEach((attribute) => {
      const name = attribute.name.toLowerCase();
      if (name === 'class' || name === 'data-consent-banner') return;
      const routeOwned = name === 'data-page' || name === 'data-audience' || ROUTE_RUNTIME_BODY_ATTRIBUTES.has(name) ||
        ROUTE_DATA_PREFIXES.some((prefix) => name.startsWith(prefix));
      if (routeOwned && !targetBody.hasAttribute(name)) body.removeAttribute(name);
    });
    Array.from(targetBody.attributes).forEach((attribute) => {
      const name = attribute.name.toLowerCase();
      if (name !== 'class' && name !== 'data-consent-banner') body.setAttribute(attribute.name, attribute.value);
    });
  }

  function syncSingleHeadElement(targetDocument, selector) {
    const current = document.head.querySelector(selector);
    const target = targetDocument.head.querySelector(selector);
    if (!target) {
      current?.remove();
      return;
    }
    const clone = document.importNode(target, true);
    if (current) current.replaceWith(clone);
    else document.head.appendChild(clone);
  }

  function syncHead(targetDocument, manifestDocument = targetDocument) {
    document.title = targetDocument.title;
    if (targetDocument.documentElement.lang) document.documentElement.lang = targetDocument.documentElement.lang;
    [
      'link[rel="canonical"]',
      'meta[name="description"]',
      'meta[name="robots"]',
      'meta[name="theme-color"]'
    ].forEach((selector) => syncSingleHeadElement(targetDocument, selector));

    const multiSelector = 'meta[property^="og:"], meta[name^="twitter:"], script[type="application/ld+json"]:not([data-site-route-manifest])';
    document.head.querySelectorAll(multiSelector).forEach((node) => node.remove());
    targetDocument.head.querySelectorAll(multiSelector).forEach((node) => {
      document.head.appendChild(document.importNode(node, true));
    });

    const currentManifest = document.querySelector(MANIFEST_SELECTOR);
    const targetManifest = manifestDocument.querySelector(MANIFEST_SELECTOR);
    if (currentManifest && targetManifest) currentManifest.textContent = targetManifest.textContent;
    else if (targetManifest) document.head.appendChild(document.importNode(targetManifest, true));
  }

  function syncSkipLink(targetDocument) {
    const current = document.querySelector('[data-site-shell-skip-link], .skip-link');
    const target = targetDocument.querySelector('[data-site-shell-skip-link], .skip-link');
    if (!current || !target) return;
    ['href', 'aria-label'].forEach((name) => {
      if (target.hasAttribute(name)) current.setAttribute(name, target.getAttribute(name));
      else current.removeAttribute(name);
    });
    current.textContent = target.textContent;
  }

  function replaceRouteContent(route, metadataDocument = route.document) {
    const currentOutlet = getRouteOutlet(document);
    if (!currentOutlet) throw new Error('The current route content outlet is unavailable.');
    const progress = document.querySelector('[data-site-route-progress]');
    if (progress && currentOutlet.contains(progress)) currentOutlet.before(progress);
    syncBody(route.document.body);
    syncHead(metadataDocument, route.document);
    syncSkipLink(route.document);
    const nextOutlet = document.importNode(route.outlet, true);
    nextOutlet.classList.add('site-route-content--preparing');
    currentOutlet.replaceWith(nextOutlet);
    const nextPanel = nextOutlet.matches?.('[data-site-route-panel], .personal-accordion__panel')
      ? nextOutlet
      : nextOutlet.querySelector?.('[data-site-route-panel], .personal-accordion__panel, .home-accordion__item.is-active .home-accordion__panel');
    if (progress && nextPanel) nextPanel.prepend(progress);
    return nextOutlet;
  }

  function getScrollOwner(outlet = getRouteOutlet(document)) {
    if (!outlet) return null;
    const descendants = Array.from(outlet.querySelectorAll?.(
      '[data-site-route-scroll], [data-personal-detail-content], [data-home-timeline-scroller], .home-accordion__item.is-active .home-accordion__scroller'
    ) || []);
    const candidates = [outlet, ...descendants];
    for (const node of candidates) {
      try {
        const style = window.getComputedStyle?.(node);
        const scrollsVertically = /(?:auto|scroll)/.test(String(style?.overflowY || '')) && node.scrollHeight > node.clientHeight;
        const scrollsHorizontally = /(?:auto|scroll)/.test(String(style?.overflowX || '')) && node.scrollWidth > node.clientWidth;
        if (scrollsVertically || scrollsHorizontally) return node;
      } catch (_) {}
    }
    return null;
  }

  function captureScroll() {
    const owner = getScrollOwner();
    return {
      panelLeft: Number(owner?.scrollLeft || 0),
      panelTop: Number(owner?.scrollTop || 0),
      windowX: Number(window.scrollX || window.pageXOffset || 0),
      windowY: Number(window.scrollY || window.pageYOffset || 0)
    };
  }

  function restoreScroll(value, url) {
    const scroll = value && typeof value === 'object' ? value : {};
    window.requestAnimationFrame(() => {
      const hash = String(url?.hash || '').replace(/^#/, '');
      if (hash) {
        let id = hash;
        try { id = decodeURIComponent(hash); } catch (_) {}
        const target = document.getElementById(id);
        if (target) {
          target.scrollIntoView({ block: 'start' });
          return;
        }
      }
      const owner = getScrollOwner();
      if (owner) {
        owner.scrollTop = Number(scroll.panelTop || 0);
        owner.scrollLeft = Number(scroll.panelLeft || 0);
      }
      window.scrollTo(Number(scroll.windowX || 0), Number(scroll.windowY || 0));
    });
  }

  function getReturnFocusId(trigger) {
    if (!trigger) return '';
    if (trigger.id) return trigger.id;
    return trigger.closest?.('[data-site-route-focus-key]')?.getAttribute('data-site-route-focus-key') || '';
  }

  function saveCurrentHistory(returnFocus = '') {
    if (typeof window.history?.replaceState !== 'function') return;
    const previous = window.history.state && typeof window.history.state === 'object' ? window.history.state : {};
    const manifest = readRouteManifest(document, resolveUrl(window.location.href), { strict: false });
    try {
      window.history.replaceState({
        ...previous,
        siteRoute: {
          ...(previous.siteRoute || {}),
          category: manifest?.category || previous.siteRoute?.category || '',
          id: manifest?.id || previous.siteRoute?.id || '',
          index: Number(previous.siteRoute?.index ?? historySequence),
          returnFocus: returnFocus || previous.siteRoute?.returnFocus || '',
          scroll: captureScroll(),
          url: window.location.href,
          view: manifest?.view || previous.siteRoute?.view || ''
        }
      }, '', window.location.href);
      if (window.location.href === committedUrl) committedHistoryState = window.history.state;
    } catch (_) {}
  }

  function pushRouteHistory(url, manifest, targetState = null) {
    historySequence += 1;
    const captured = targetState && typeof targetState === 'object' ? targetState : {};
    const capturedState = { ...captured };
    delete capturedState.siteRouteProvisional;
    if (manifest.id !== 'home') {
      delete capturedState.homePanel;
      delete capturedState.homeView;
      delete capturedState.personalCategory;
      delete capturedState.personalView;
    }
    window.history.pushState({
      ...capturedState,
      siteRoute: {
        ...(captured.siteRoute || {}),
        category: manifest.category,
        id: manifest.id,
        index: historySequence,
        scroll: { panelLeft: 0, panelTop: 0, windowX: 0, windowY: 0 },
        url: url.href,
        view: manifest.view
      }
    }, '', url.href);
    committedHistoryIndex = historySequence;
    committedHistoryState = window.history.state;
    committedUrl = url.href;
  }

  function beginProvisionalHistory(url) {
    const previousUrl = window.location.href;
    const previousState = window.history.state;
    window.history.replaceState({
      ...(previousState && typeof previousState === 'object' ? previousState : {}),
      siteRouteProvisional: true
    }, '', url.href);
    return {
      active: true,
      previousState,
      previousUrl
    };
  }

  function restoreProvisionalHistory(provisional) {
    if (!provisional?.active) return;
    provisional.active = false;
    window.history.replaceState(provisional.previousState, '', provisional.previousUrl);
  }

  function acceptPoppedHistory(url, state) {
    committedHistoryIndex = Number(state?.siteRoute?.index ?? committedHistoryIndex);
    committedHistoryState = state;
    committedUrl = url.href;
  }

  function restoreVetoedPop(targetState) {
    const targetIndex = Number(targetState?.siteRoute?.index);
    const delta = committedHistoryIndex - targetIndex;
    if (Number.isFinite(targetIndex) && delta && typeof window.history.go === 'function') {
      restoringVetoedPop = true;
      window.history.go(delta);
      return;
    }
    if (committedUrl) window.location.assign(committedUrl);
  }

  function cssEscape(value) {
    if (window.CSS?.escape) return window.CSS.escape(value);
    return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
  }

  function focusRouteHeading(navigationType, historyState) {
    if (navigationType === 'pop') {
      const returnFocus = String(historyState?.siteRoute?.returnFocus || '');
      if (returnFocus) {
        const target = document.getElementById(returnFocus) ||
          document.querySelector(`[data-site-route-focus-key="${cssEscape(returnFocus)}"]`);
        if (target) {
          target.focus({ preventScroll: true });
          return;
        }
      }
    }
    const heading = getRouteOutlet(document)?.querySelector('h1, [role="heading"][aria-level="1"]');
    if (!heading) return;
    if (!heading.matches('a[href], button, input, select, textarea, [tabindex]')) {
      heading.setAttribute('tabindex', '-1');
      heading.addEventListener('blur', () => heading.removeAttribute('tabindex'), { once: true });
    }
    heading.focus({ preventScroll: true });
  }

  function announce(message) {
    let status = document.querySelector('[data-site-route-announcer], [data-site-route-status]');
    if (!status) {
      status = document.createElement('div');
      status.className = 'site-route-status';
      status.dataset.siteRouteAnnouncer = 'true';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      document.body.appendChild(status);
    }
    status.textContent = '';
    window.requestAnimationFrame(() => { status.textContent = message; });
  }

  function getProgressElement() {
    const panel = document.querySelector('[data-site-route-panel], .personal-accordion__panel, .home-accordion__item.is-active .home-accordion__panel');
    let progress = document.querySelector('[data-site-route-progress]');
    if (progress) {
      if (panel && progress.parentElement !== panel) panel.prepend(progress);
      return progress;
    }
    if (!panel) return null;
    progress = document.createElement('div');
    progress.className = 'site-route-progress';
    progress.dataset.siteRouteProgress = 'true';
    progress.setAttribute('aria-hidden', 'true');
    progress.hidden = true;
    panel.prepend(progress);
    return progress;
  }

  function beginNavigationUi(trigger) {
    const outlet = getRouteOutlet(document);
    const progress = getProgressElement();
    const previousBusy = trigger?.getAttribute?.('aria-busy');
    outlet?.setAttribute('aria-busy', 'true');
    trigger?.setAttribute?.('aria-busy', 'true');
    const timer = window.setTimeout(() => {
      if (progress) progress.hidden = false;
      document.documentElement.classList.add('site-route-is-loading');
    }, PROGRESS_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      if (progress) progress.hidden = true;
      document.documentElement.classList.remove('site-route-is-loading');
      getRouteOutlet(document)?.removeAttribute('aria-busy');
      if (trigger) {
        if (previousBusy == null) trigger.removeAttribute('aria-busy');
        else trigger.setAttribute('aria-busy', previousBusy);
      }
    };
  }

  function clearRouteError() {
    document.querySelector('[data-site-route-error]')?.remove();
  }

  function showOfflineError(url) {
    clearRouteError();
    const outlet = getRouteOutlet(document);
    if (!outlet) return;
    const error = document.createElement('div');
    error.className = 'site-route-error';
    error.dataset.siteRouteError = 'true';
    error.setAttribute('role', 'status');
    const message = document.createElement('span');
    message.textContent = 'This page is not available offline yet.';
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'site-route-error__retry';
    retry.textContent = 'Retry';
    retry.addEventListener('click', () => navigate(url, { trigger: retry }));
    error.append(message, retry);
    outlet.parentElement?.insertBefore(error, outlet);
    retry.focus();
  }

  function animateOutgoing(outlet, direction) {
    if (!outlet || prefersReducedMotion() || typeof outlet.animate !== 'function') return Promise.resolve();
    const motion = window.matchMedia?.('(max-width: 959px), (max-height: 619px)')?.matches ? 4 : 6;
    const distance = direction === 'back' ? motion : -motion;
    const animation = outlet.animate([
      { opacity: 1, transform: 'translate3d(0, 0, 0)' },
      { opacity: 0, transform: `translate3d(${distance}px, 0, 0)` }
    ], {
      duration: 72,
      easing: 'cubic-bezier(.4, 0, 1, 1)',
      fill: 'both'
    });
    return animation.finished.catch(() => {}).finally(() => {
      outlet.classList.add('site-route-content--leaving');
      animation.cancel();
    });
  }

  function animateIncoming(outlet, direction) {
    if (!outlet) return;
    outlet.classList.remove('site-route-content--preparing');
    if (prefersReducedMotion()) return;
    document.documentElement.dataset.siteRouteDirection = direction;
    outlet.classList.remove('site-route-content--entering');
    void outlet.offsetWidth;
    outlet.classList.add('site-route-content--entering');
    const cleanup = () => {
      outlet.classList.remove('site-route-content--entering');
      delete document.documentElement.dataset.siteRouteDirection;
    };
    outlet.addEventListener('animationend', cleanup, { once: true });
    window.setTimeout(cleanup, 240);
  }

  function navigationDirection(previousManifest, nextManifest, navigationType) {
    if (navigationType === 'pop') return 'back';
    const depth = { overview: 0, library: 1, detail: 2 };
    if (!previousManifest || previousManifest.category !== nextManifest.category) return 'cross';
    if (depth[nextManifest.view] < depth[previousManifest.view]) return 'back';
    return 'forward';
  }

  function sendVirtualPageview(url) {
    let consent = null;
    try { consent = window.consentAPI?.get?.(); } catch (_) {}
    if (!consent?.analytics) return;
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({
      event: 'virtual_page_view',
      page_location: url.href,
      page_path: `${url.pathname}${url.search}${url.hash}`,
      page_title: document.title
    });
  }

  function hardNavigate(url) {
    window.location.assign(url.href);
  }

  async function navigate(urlLike, options = {}) {
    const url = urlLike instanceof URL ? urlLike : resolveUrl(urlLike);
    if (!url) return false;
    if (!isDocumentLikeUrl(url) || !getPersonalRouteIntent(url) || isHardBoundary(url) || !isCurrentRouteSoft()) {
      hardNavigate(url);
      return false;
    }

    const targetKey = `${normalizeRouteUrl(url)}${url.hash}`;
    if (activeNavigation?.key === targetKey) return activeNavigation.promise;
    if (activeNavigation?.committed) {
      return activeNavigation.promise.then(() => navigate(url, options));
    }
    if (activeNavigation) activeNavigation.controller.abort(makeAbortError());

    const controller = new AbortController();
    const sequence = ++navigationSequence;
    const finishUi = beginNavigationUi(options.trigger);
    const previousManifest = readRouteManifest(document, resolveUrl(window.location.href), { strict: false });
    const returnFocus = getReturnFocusId(options.trigger);
    const navigationType = options.navigationType || 'push';
    const homepageHistoryIntent = navigationType === 'pop'
      ? getHomepageHistoryIntent(options.historyState)
      : null;
    const documentUrl = homepageHistoryIntent ? resolveUrl('/', url.href) : url;
    clearRouteError();
    dispatch(NAVIGATION_EVENT, { href: url.href, navigationType });

    const operation = (async () => {
      let addedStyles = [];
      let committed = false;
      let provisionalHistory = null;
      try {
        const routePromise = fetchRouteDocument(documentUrl, { signal: controller.signal });
        const metadataPromise = homepageHistoryIntent && normalizeRouteUrl(documentUrl) !== normalizeRouteUrl(url)
          ? fetchRouteDocument(url, { signal: controller.signal })
          : routePromise;
        const [route, metadataRoute] = await Promise.all([routePromise, metadataPromise]);
        throwIfAborted(controller.signal);
        const prepared = await Promise.all([
          prepareRouteStyles(route, controller.signal),
          prepareRouteScripts(route, controller.signal)
        ]);
        addedStyles = prepared[0];
        throwIfAborted(controller.signal);

        const runtime = window.SiteRoutes;
        const canLeave = await runtime.beforeLeave({ navigationType, reason: 'navigate', url });
        if (canLeave === false) {
          addedStyles.forEach((link) => link?.remove());
          if (navigationType === 'pop') restoreVetoedPop(options.historyState);
          return false;
        }

        if (navigationType === 'push') saveCurrentHistory(returnFocus);
        const direction = navigationDirection(previousManifest, route.manifest, navigationType);
        await animateOutgoing(getRouteOutlet(document), direction);
        throwIfAborted(controller.signal);
        await runtime.unmount({ navigationType, reason: 'navigate', url });
        throwIfAborted(controller.signal);

        activatePreparedStyles(addedStyles);
        const outlet = replaceRouteContent(route, metadataRoute.document);
        committed = true;
        if (activeNavigation?.sequence === sequence) activeNavigation.committed = true;
        retireOldStyles(previousManifest, route.manifest);
        const routeKey = registeredRouteKey(route.manifest);
        if (!routeKey) throw new Error(`Route ${route.manifest.id} lost its lifecycle registration.`);
        if (navigationType === 'push') provisionalHistory = beginProvisionalHistory(url);
        await runtime.mount(routeKey, {
          manifest: route.manifest,
          navigationType,
          root: outlet,
          signal: controller.signal,
          url
        });
        throwIfAborted(controller.signal);

        if (navigationType === 'push') {
          const targetState = window.history.state;
          restoreProvisionalHistory(provisionalHistory);
          pushRouteHistory(url, route.manifest, targetState);
        } else {
          acceptPoppedHistory(url, options.historyState);
        }
        const historyState = navigationType === 'pop' ? options.historyState : window.history.state;
        restoreScroll(navigationType === 'pop' ? historyState?.siteRoute?.scroll : null, url);
        window.requestAnimationFrame(() => focusRouteHeading(navigationType, historyState));
        animateIncoming(outlet, direction);
        announce(`${document.title} loaded.`);
        const detail = {
          category: route.manifest.category,
          from: previousManifest?.id || '',
          id: route.manifest.id,
          navigationType,
          to: route.manifest.id,
          url: url.href,
          view: route.manifest.view
        };
        dispatch(CONTENT_EVENT, detail);
        dispatch(ROUTE_EVENT, detail);
        sendVirtualPageview(url);
        return true;
      } catch (error) {
        restoreProvisionalHistory(provisionalHistory);
        getRouteOutlet(document)?.classList.remove('site-route-content--leaving');
        if (!committed) addedStyles.forEach((link) => link?.remove());
        if (!committed && (isAbortError(error) || controller.signal.aborted || sequence !== navigationSequence)) return false;
        dispatch('site:route-navigation-error', { error, href: url.href });
        if (committed || navigator.onLine !== false) hardNavigate(url);
        else {
          if (navigationType === 'pop') restoreVetoedPop(options.historyState);
          showOfflineError(url);
          announce('The requested page is unavailable offline.');
        }
        return false;
      } finally {
        finishUi();
        if (activeNavigation?.sequence === sequence) activeNavigation = null;
      }
    })();

    activeNavigation = { committed: false, controller, key: targetKey, promise: operation, sequence };
    return operation;
  }

  function schedulePrefetch(url) {
    if (!url) return;
    const run = () => prefetchRoute(url);
    if (typeof window.requestIdleCallback === 'function') window.requestIdleCallback(run, { timeout: 900 });
    else window.setTimeout(run, 0);
  }

  function initHistory() {
    if ('scrollRestoration' in window.history) window.history.scrollRestoration = 'manual';
    const state = window.history.state && typeof window.history.state === 'object' ? window.history.state : {};
    historySequence = Number(state.siteRoute?.index || 0);
    committedHistoryIndex = historySequence;
    committedHistoryState = state;
    committedUrl = window.location.href;
    if (!state.siteRoute) saveCurrentHistory();
  }

  function initNavigation() {
    initHistory();

    document.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.__contactHandled) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof event.button === 'number' && event.button !== 0) return;
      const link = event.target?.closest?.('a[href]');
      if (!link) return;
      const url = getEligibleLinkUrl(link);
      if (!url || hasBlockingInteractionLayer()) return;
      if (document.querySelector('.home-accordion.is-view-changing')) {
        event.preventDefault();
        return;
      }
      event.preventDefault();
      navigate(url, { trigger: link });
    });

    document.addEventListener('submit', (event) => {
      if (event.defaultPrevented) return;
      const form = event.target?.closest?.('form');
      if (!form || String(form.method || 'get').toLowerCase() !== 'get') return;
      if (form.dataset.navigation === 'hard' || form.closest('[data-navigation="hard"]')) return;
      const action = resolveUrl(form.getAttribute('action') || window.location.href);
      if (!action) return;
      action.search = new URLSearchParams(new FormData(form, event.submitter)).toString();
      if (!isDocumentLikeUrl(action) || !getPersonalRouteIntent(action) || !isCurrentRouteSoft()) return;
      event.preventDefault();
      navigate(action, { trigger: event.submitter || form });
    });

    const prefetchFromEvent = (event) => {
      const link = event.target?.closest?.('a[href]');
      const url = getEligibleLinkUrl(link);
      if (url) schedulePrefetch(url);
    };
    document.addEventListener('pointerover', prefetchFromEvent, { passive: true });
    document.addEventListener('focusin', prefetchFromEvent);
    document.addEventListener('pointerdown', (event) => {
      const link = event.target?.closest?.('a[href]');
      const url = getEligibleLinkUrl(link);
      if (url) prefetchRoute(url);
    }, { passive: true });

    window.addEventListener('popstate', (event) => {
      if (restoringVetoedPop) {
        restoringVetoedPop = false;
        committedHistoryIndex = Number(event.state?.siteRoute?.index ?? committedHistoryIndex);
        committedHistoryState = event.state || committedHistoryState;
        committedUrl = window.location.href;
        return;
      }
      const url = resolveUrl(window.location.href);
      if (!url || !getPersonalRouteIntent(url) || isHardBoundary(url) || !isCurrentRouteSoft()) return;
      const currentManifest = readRouteManifest(document, url, { strict: false });
      if (currentManifest?.id === 'home' && getHomepageHistoryIntent(event.state)) {
        acceptPoppedHistory(url, event.state);
        return;
      }
      navigate(url, { historyState: event.state, navigationType: 'pop' });
    });

    window.addEventListener('pagehide', () => {
      saveCurrentHistory();
      activeNavigation?.controller.abort(makeAbortError());
    });

    window.addEventListener('pageshow', (event) => {
      if (!event.persisted) return;
      activeNavigation = null;
      document.documentElement.classList.remove('site-route-is-loading');
      const progress = document.querySelector('[data-site-route-progress]');
      if (progress) progress.hidden = true;
      restoreScroll(window.history.state?.siteRoute?.scroll, resolveUrl(window.location.href));
    });
  }

  window.SiteNavigation = Object.freeze({
    version: 1,
    hardBoundaryPaths: Object.freeze(Array.from(HARD_BOUNDARY_PATHS)),
    navigate,
    prefetch: (value) => {
      const url = resolveUrl(value);
      return url && getPersonalRouteIntent(url) ? prefetchRoute(url) : Promise.resolve(null);
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNavigation, { once: true });
  } else {
    initNavigation();
  }
})();
