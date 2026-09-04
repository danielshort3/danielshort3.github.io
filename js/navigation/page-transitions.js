/* ===================================================================
   File: page-transitions.js
   Purpose: Smooth same-origin page transitions and header prefetching
=================================================================== */
(() => {
  'use strict';

  const NAVIGATION_EVENT = 'site:navigation-start';
  const STORAGE_KEY = 'sitePageTransition';
  const ARRIVAL_FOCUS_KEY = 'sitePageTransitionFocus';
  const TRANSITION_TTL_MS = 30000;
  const ARRIVAL_FOCUS_TTL_MS = 30000;
  const FALLBACK_EXIT_MS = 176;
  const FALLBACK_ENTRY_MS = 244;
  const COMPACT_FALLBACK_EXIT_MS = 140;
  const COMPACT_FALLBACK_ENTRY_MS = 200;
  const NEUTRAL_EXIT_MS = 180;
  const NEUTRAL_ENTRY_MS = 220;
  const NATIVE_TRANSITION_SAFETY_MS = 800;
  const TRANSITION_VIEWS = new Set(['overview', 'library', 'detail']);
  const TRANSITION_DIRECTIONS = new Set(['forward', 'back', 'cross', 'replace']);
  const TRANSITION_MODES = new Set(['personal', 'neutral']);
  const TRANSITION_TRANSPORTS = new Set(['fallback', 'native']);
  const VIEW_DEPTH = Object.freeze({ overview: 0, library: 1, detail: 2 });
  let navigationLocked = false;
  let navigationDepartureObserved = false;
  let navigationCommitStartedAt = 0;
  let navigationTarget = '';
  let incomingTransitionActive = false;
  const prefetchedTargets = new Set();
  const PERSONAL_ROUTE_CONFIG = Object.freeze([
    Object.freeze({ path: '/portfolio', category: 'projects', rootView: 'library' }),
    Object.freeze({ path: '/tools', category: 'tools', rootView: 'library' }),
    Object.freeze({ path: '/games', category: 'games', rootView: 'library' }),
    Object.freeze({ path: '/contact', category: 'contact', rootView: 'detail' }),
    Object.freeze({ path: '/privacy', category: 'about', rootView: 'detail', exact: true }),
    Object.freeze({ path: '/sitemap', category: 'about', rootView: 'detail', exact: true }),
    Object.freeze({ path: '/sitemap-pretty', category: 'about', rootView: 'detail', exact: true }),
    Object.freeze({ path: '/search', category: 'tools', rootView: 'detail', exact: true }),
    Object.freeze({ path: '/solutions', category: 'projects', rootView: 'detail', exact: true })
  ]);
  const HOME_CATEGORY_IDS = new Set(['about', 'projects', 'tools', 'games', 'contact']);
  const PROFESSIONAL_AUDIENCES = new Set(['analytics', 'data-science', 'tourism']);

  const prefersReducedMotion = () => {
    try {
      return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch {
      return false;
    }
  };

  const resolveUrl = (href) => {
    try {
      return new URL(href, document.baseURI || window.location.href);
    } catch {
      return null;
    }
  };

  const normalizePathname = (pathname) => {
    let next = String(pathname || '/');
    next = next.replace(/\/index\.html$/i, '/');
    next = next.replace(/\.html$/i, '');
    next = next.replace(/\/+$/, '');
    if (!next) next = '/';
    return next;
  };

  const normalizeTarget = (url) => {
    if (!url) return '';
    return `${url.origin}${normalizePathname(url.pathname)}${url.search}`;
  };

  const normalizeArrivalTarget = (url) => {
    if (!url) return '';
    return `${normalizeTarget(url)}${url.hash || ''}`;
  };

  const isProfessionalAudienceUrl = (url) => {
    if (!url) return false;
    const pathname = normalizePathname(url.pathname);
    if (/^\/(?:analytics|data-science|tourism)(?:\/|$)/.test(pathname)) return true;
    const audience = String(url.searchParams?.get('audience') || '').trim().toLowerCase();
    const mode = String(url.searchParams?.get('mode') || '').trim().toLowerCase();
    return PROFESSIONAL_AUDIENCES.has(audience) || ['professional', 'work', 'career', 'analytics'].includes(mode);
  };

  const getPersonalRouteIntent = (url) => {
    if (!url) return null;
    if (isProfessionalAudienceUrl(url)) return null;
    const pathname = normalizePathname(url.pathname);
    if (pathname === '/') {
      let category = String(url.hash || '').replace(/^#/, '');
      try {
        category = decodeURIComponent(category);
      } catch {
        category = '';
      }
      if (!category) category = 'about';
      if (!HOME_CATEGORY_IDS.has(category)) return null;
      const requestedLibrary = url.searchParams?.get('view') === 'library';
      const view = requestedLibrary && ['projects', 'tools', 'games'].includes(category)
        ? 'library'
        : 'overview';
      return { category, view };
    }

    if (/^\/[a-z0-9-]+-demo$/i.test(pathname)) {
      return { category: 'projects', view: 'detail' };
    }

    const route = PERSONAL_ROUTE_CONFIG.find((entry) => (
      pathname === entry.path || (!entry.exact && pathname.startsWith(`${entry.path}/`))
    ));
    if (!route) return null;
    return {
      category: route.category,
      view: pathname === route.path ? route.rootView : 'detail'
    };
  };

  const getPathExtension = (url) => {
    if (!url || !url.pathname) return '';
    const segment = url.pathname.split('/').pop() || '';
    const dotIndex = segment.lastIndexOf('.');
    if (dotIndex <= 0) return '';
    return segment.slice(dotIndex + 1).toLowerCase();
  };

  const isDocumentLikeUrl = (url) => {
    if (!url || !/^https?:$/i.test(url.protocol)) return false;
    if (url.origin !== window.location.origin) return false;
    const extension = getPathExtension(url);
    return !extension || extension === 'html' || extension === 'htm';
  };

  const isSameDocumentNavigation = (url) => {
    const currentUrl = resolveUrl(window.location.href);
    return normalizeTarget(url) === normalizeTarget(currentUrl);
  };

  const supportsCrossDocumentViewTransitions = () => {
    try {
      return typeof CSS !== 'undefined' &&
        CSS.supports('view-transition-name: site-shell') &&
        'onpageswap' in window &&
        'onpagereveal' in window;
    } catch {
      return false;
    }
  };

  const isPersonalSurfaceNavigation = (link) => Boolean(
    link?.dataset.personalTransition ||
    link?.closest('[data-home-accordion], [data-personal-accordion-shell]')
  );

  const usesCompactTransition = () => {
    try {
      return Boolean(window.matchMedia && window.matchMedia('(max-width: 959px), (max-height: 619px)').matches);
    } catch {
      return false;
    }
  };

  const hasBlockingInteractionLayer = () => Boolean(
    document.fullscreenElement ||
    document.pointerLockElement ||
    document.body?.classList?.contains('modal-open') ||
    document.body?.classList?.contains('media-viewer-open') ||
    document.querySelector?.('dialog[open], .modal.active, [data-tools-account-modal][aria-hidden="false"]')
  );

  const hasActiveSameDocumentTransition = () => Boolean(
    document.querySelector?.('.home-accordion.is-view-changing')
  );

  const getEligibleNavigationUrl = (link) => {
    if (!link || link.dataset.pageTransition === 'false') return null;
    if (link.hasAttribute('download')) return null;
    if (link.closest('[data-contact-modal-link]')) return null;

    const target = String(link.getAttribute('target') || '').trim().toLowerCase();
    if (target && target !== '_self') return null;

    const href = String(link.getAttribute('href') || '').trim();
    if (!href || href.startsWith('#')) return null;
    if (/^(mailto:|tel:|javascript:)/i.test(href)) return null;

    const url = resolveUrl(href);
    if (!isDocumentLikeUrl(url)) return null;
    if (isSameDocumentNavigation(url)) return null;

    return url;
  };

  const getStorage = () => {
    try {
      return window.sessionStorage;
    } catch {
      return null;
    }
  };

  const storeArrivalFocus = (url, intentOverride = null) => {
    const storage = getStorage();
    const intent = intentOverride || getPersonalRouteIntent(url);
    if (!storage || !url || !intent) return;
    if (!HOME_CATEGORY_IDS.has(intent.category) || !TRANSITION_VIEWS.has(intent.view)) return;
    try {
      storage.setItem(ARRIVAL_FOCUS_KEY, JSON.stringify({
        target: normalizeArrivalTarget(url),
        category: intent.category,
        view: intent.view,
        ts: Date.now()
      }));
    } catch {}
  };

  const consumeArrivalFocus = () => {
    const storage = getStorage();
    if (!storage) return null;

    let raw = null;
    try {
      raw = storage.getItem(ARRIVAL_FOCUS_KEY);
      storage.removeItem(ARRIVAL_FOCUS_KEY);
    } catch {
      return null;
    }
    if (!raw) return null;

    try {
      const payload = JSON.parse(raw);
      if (!payload || typeof payload !== 'object') return null;
      if (typeof payload.target !== 'string' || typeof payload.category !== 'string') return null;
      if (!['overview', 'library', 'detail'].includes(payload.view)) return null;
      if (!Number.isFinite(payload.ts) || (Date.now() - payload.ts) > ARRIVAL_FOCUS_TTL_MS) return null;
      return payload;
    } catch {
      return null;
    }
  };

  const currentPersonalState = () => {
    const routeIntent = getPersonalRouteIntent(resolveUrl(window.location.href));
    const homeRoot = document.querySelector('[data-home-accordion]');
    const personalShell = document.querySelector('[data-personal-accordion-shell]');
    if (!homeRoot && !personalShell) return null;
    const bodyCategory = String(document.body?.dataset.personalCategory || '').trim();
    const bodyView = String(document.body?.dataset.personalAccordionView || '').trim();
    const category = bodyCategory || routeIntent?.category || String(homeRoot?.dataset.activePanel || '').trim();
    const view = bodyView || routeIntent?.view || String(homeRoot?.dataset.homeView || '').trim();
    if (!HOME_CATEGORY_IDS.has(category) || !['overview', 'library', 'detail'].includes(view)) return null;
    return { category, view };
  };

  const getTransitionDescriptor = (link, url) => {
    const from = currentPersonalState();
    let to = getPersonalRouteIntent(url);
    const personalHint = String(link?.dataset?.personalTransition || '').trim().toLowerCase();
    if (!to && isPersonalSurfaceNavigation(link) && from) {
      to = {
        category: from.category,
        view: personalHint === 'detail' ? 'detail' : from.view
      };
    }

    const mode = from && to ? 'personal' : 'neutral';
    const category = to?.category || from?.category || 'neutral';
    let direction = 'replace';
    if (personalHint === 'collapse') {
      direction = 'back';
    } else if (from && to) {
      if (from.category !== to.category) direction = 'cross';
      else if (VIEW_DEPTH[to.view] > VIEW_DEPTH[from.view]) direction = 'forward';
      else if (VIEW_DEPTH[to.view] < VIEW_DEPTH[from.view]) direction = 'back';
    } else if (to) {
      direction = 'forward';
    }

    return Object.freeze({
      mode,
      category: HOME_CATEGORY_IDS.has(category) ? category : 'neutral',
      fromView: from?.view || 'detail',
      toView: to?.view || 'detail',
      direction,
      targetIntent: to
    });
  };

  const syncPersonalHistoryState = () => {
    const semantic = currentPersonalState();
    if (!semantic || typeof window.history?.replaceState !== 'function') return;
    const currentState = window.history.state && typeof window.history.state === 'object'
      ? window.history.state
      : {};
    if (currentState.personalCategory === semantic.category && currentState.personalView === semantic.view) return;
    try {
      window.history.replaceState({
        ...currentState,
        personalCategory: semantic.category,
        personalView: semantic.view
      }, '', window.location.href);
    } catch {}
  };

  const getArrivalFocusTarget = (intent) => {
    if (!intent) return null;
    const category = HOME_CATEGORY_IDS.has(intent.category) ? intent.category : '';
    if (intent.view === 'overview' && category) {
      const homeRoot = document.querySelector('[data-home-accordion]');
      return homeRoot?.querySelector(`[data-home-library-open="${category}"]`) ||
        homeRoot?.querySelector(`[data-home-accordion-trigger="${category}"]`) ||
        null;
    }
    if (intent.view === 'library') {
      return document.querySelector(
        `[data-home-library-view="${category}"]:not([hidden]) [data-home-library-heading], ` +
        'body[data-personal-accordion-view="library"] [data-personal-detail-content] h1, ' +
        'body[data-personal-accordion-view="library"] main h1'
      );
    }
    return document.querySelector(
      'body[data-personal-accordion-view="detail"] [data-personal-detail-content] h1, ' +
      'body[data-personal-accordion-view="detail"] main h1'
    );
  };

  const focusArrivalTarget = (intent) => {
    const target = getArrivalFocusTarget(intent);
    if (!target || target.closest?.('[hidden], [inert], [aria-hidden="true"]')) return false;
    const alreadyFocusable = typeof target.matches === 'function' &&
      target.matches('a[href], button, input, select, textarea, [tabindex]');
    const hadTabIndex = target.hasAttribute?.('tabindex');
    if (!alreadyFocusable && !hadTabIndex) {
      target.setAttribute('tabindex', '-1');
      target.addEventListener?.('blur', () => target.removeAttribute('tabindex'), { once: true });
    }
    try {
      target.focus({ preventScroll: true });
    } catch {
      target.focus?.();
    }
    return document.activeElement === target || typeof document.activeElement === 'undefined';
  };

  const scheduleArrivalFocus = (intent, attempt = 0) => {
    if (!intent || attempt > 120) return;
    window.requestAnimationFrame(() => {
      if (!focusArrivalTarget(intent)) scheduleArrivalFocus(intent, attempt + 1);
    });
  };

  const consumeArrivalFocusForCurrentPage = () => {
    const pending = consumeArrivalFocus();
    const currentUrl = resolveUrl(window.location.href);
    if (!pending || !currentUrl || pending.target !== normalizeArrivalTarget(currentUrl)) return false;
    return pending;
  };

  const focusCurrentPersonalState = () => {
    const state = window.history?.state;
    const historyIntent = state && typeof state === 'object'
      ? { category: state.personalCategory, view: state.personalView }
      : null;
    scheduleArrivalFocus(historyIntent || currentPersonalState());
  };

  const isHistoryTraversal = () => {
    try {
      const entries = window.performance?.getEntriesByType?.('navigation') || [];
      return entries[0]?.type === 'back_forward';
    } catch {
      return false;
    }
  };

  const storePendingNavigation = (url, descriptor, transport = 'fallback') => {
    const storage = getStorage();
    if (!storage || !url || !descriptor) return;
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({
        target: normalizeTarget(url),
        mode: TRANSITION_MODES.has(descriptor.mode) ? descriptor.mode : 'neutral',
        category: HOME_CATEGORY_IDS.has(descriptor.category) ? descriptor.category : 'neutral',
        fromView: TRANSITION_VIEWS.has(descriptor.fromView) ? descriptor.fromView : 'detail',
        toView: TRANSITION_VIEWS.has(descriptor.toView) ? descriptor.toView : 'detail',
        direction: TRANSITION_DIRECTIONS.has(descriptor.direction) ? descriptor.direction : 'replace',
        transport: TRANSITION_TRANSPORTS.has(transport) ? transport : 'fallback',
        ts: Date.now()
      }));
    } catch {}
  };

  const consumePendingNavigation = () => {
    const storage = getStorage();
    if (!storage) return null;

    let raw = null;
    try {
      raw = storage.getItem(STORAGE_KEY);
      storage.removeItem(STORAGE_KEY);
    } catch {
      return null;
    }

    if (!raw) return null;

    try {
      const payload = JSON.parse(raw);
      if (!payload || typeof payload !== 'object') return null;
      if (typeof payload.target !== 'string') return null;
      if (!Number.isFinite(payload.ts)) return null;
      if ((Date.now() - payload.ts) > TRANSITION_TTL_MS) return null;
      const legacyMode = payload.mode === 'continuous' ? 'personal' : payload.mode === 'fade' ? 'neutral' : payload.mode;
      return {
        target: payload.target,
        mode: TRANSITION_MODES.has(legacyMode) ? legacyMode : 'neutral',
        category: HOME_CATEGORY_IDS.has(payload.category) ? payload.category : 'neutral',
        fromView: TRANSITION_VIEWS.has(payload.fromView) ? payload.fromView : 'detail',
        toView: TRANSITION_VIEWS.has(payload.toView) ? payload.toView : 'detail',
        direction: TRANSITION_DIRECTIONS.has(payload.direction) ? payload.direction : 'replace',
        transport: TRANSITION_TRANSPORTS.has(payload.transport) ? payload.transport : 'fallback',
        ts: payload.ts
      };
    } catch {
      return null;
    }
  };

  const transitionNodes = () => [document.documentElement, document.body].filter(Boolean);

  const setTransitionPresentation = (descriptor) => {
    const mode = TRANSITION_MODES.has(descriptor?.mode) ? descriptor.mode : 'neutral';
    const category = HOME_CATEGORY_IDS.has(descriptor?.category) ? descriptor.category : 'neutral';
    const direction = TRANSITION_DIRECTIONS.has(descriptor?.direction) ? descriptor.direction : 'replace';
    const transport = TRANSITION_TRANSPORTS.has(descriptor?.transport) ? descriptor.transport : 'fallback';
    transitionNodes().forEach((node) => {
      node.dataset.siteTransitionMode = mode;
      node.dataset.siteTransitionCategory = category;
      node.dataset.siteTransitionDirection = direction;
      node.dataset.siteTransitionTransport = transport;
    });
  };

  const clearTransitionPresentation = () => {
    transitionNodes().forEach((node) => {
      delete node.dataset.siteTransitionMode;
      delete node.dataset.siteTransitionCategory;
      delete node.dataset.siteTransitionDirection;
      delete node.dataset.siteTransitionTransport;
    });
  };

  const clearTransitionClasses = (options = {}) => {
    const preservePreload = options.preservePreload === true;
    document.documentElement.classList.remove(
      'site-is-navigating',
      'site-page-transition-out',
      'site-page-transition-in',
      'site-page-transition-continuous-out',
      'site-page-transition-continuous-in',
      'site-page-transition-native-preload'
    );
    if (!preservePreload) {
      document.documentElement.classList.remove(
        'site-page-transition-preload',
        'site-page-transition-continuous-preload'
      );
    }
    document.body?.classList.remove(
      'site-is-navigating',
      'site-page-transition-out',
      'site-page-transition-in',
      'site-page-transition-continuous-out',
      'site-page-transition-continuous-in',
      'site-page-transition-native-preload'
    );
    if (!preservePreload) {
      document.body?.classList.remove(
        'site-page-transition-preload',
        'site-page-transition-continuous-preload'
      );
    }
  };

  const markNavigating = () => {
    document.documentElement.classList.add('site-is-navigating');
    document.body?.classList.add('site-is-navigating');
  };

  const dispatchNavigationEvent = (url) => {
    try {
      document.dispatchEvent(new CustomEvent(NAVIGATION_EVENT, {
        detail: {
          href: url ? url.href : ''
        }
      }));
    } catch {
      const event = document.createEvent('CustomEvent');
      event.initCustomEvent(NAVIGATION_EVENT, false, false, {
        href: url ? url.href : ''
      });
      document.dispatchEvent(event);
    }
  };

  const startFallbackExit = () => {
    document.documentElement.classList.add('site-page-transition-out');
    document.body?.classList.add('site-page-transition-out');
  };

  const scheduleFallbackEntry = (pending, onComplete) => {
    document.documentElement.classList.remove('site-page-transition-preload');
    document.body?.classList.remove('site-page-transition-preload');
    document.documentElement.classList.add('site-page-transition-in');
    document.body?.classList.add('site-page-transition-in');
    const cleanupDelay = pending.mode === 'personal'
      ? (usesCompactTransition() ? COMPACT_FALLBACK_ENTRY_MS : FALLBACK_ENTRY_MS)
      : NEUTRAL_ENTRY_MS;
    window.setTimeout(() => {
      document.documentElement.classList.remove('site-page-transition-in');
      document.body?.classList.remove('site-page-transition-in');
      clearTransitionPresentation();
      if (typeof onComplete === 'function') onComplete();
    }, cleanupDelay);
  };

  const scheduleNativeEntryCompletion = (onComplete) => {
    let settled = false;
    let safetyTimer = 0;
    const finish = () => {
      if (settled) return;
      settled = true;
      if (safetyTimer) window.clearTimeout(safetyTimer);
      document.documentElement.classList.remove('site-page-transition-native-preload');
      document.body?.classList.remove('site-page-transition-native-preload');
      clearTransitionPresentation();
      if (typeof onComplete === 'function') onComplete();
    };
    const handleViewTransition = (viewTransition) => {
      const finished = viewTransition?.finished;
      if (finished && typeof finished.then === 'function') {
        Promise.resolve(finished).catch(() => {}).finally(finish);
        return;
      }
      window.requestAnimationFrame(() => window.requestAnimationFrame(finish));
    };

    safetyTimer = window.setTimeout(finish, NATIVE_TRANSITION_SAFETY_MS);
    const revealState = window.__sitePageRevealState;
    if (revealState?.seen) {
      handleViewTransition(revealState.viewTransition);
    } else if (Array.isArray(revealState?.callbacks)) {
      revealState.callbacks.push(handleViewTransition);
    } else {
      window.addEventListener('pagereveal', (event) => {
        handleViewTransition(event?.viewTransition || null);
      }, { once: true });
    }
  };

  const hydrateIncomingTransition = (onComplete) => {
    clearTransitionClasses({ preservePreload: true });
    const pending = consumePendingNavigation();
    if (!pending) {
      clearTransitionClasses();
      clearTransitionPresentation();
      return false;
    }

    const currentUrl = resolveUrl(window.location.href);
    if (!currentUrl || pending.target !== normalizeTarget(currentUrl)) {
      clearTransitionClasses();
      clearTransitionPresentation();
      return false;
    }
    if (prefersReducedMotion()) {
      clearTransitionClasses();
      clearTransitionPresentation();
      return false;
    }

    setTransitionPresentation(pending);
    if (pending.transport === 'native') {
      document.documentElement.classList.add('site-page-transition-native-preload');
      document.body?.classList.add('site-page-transition-native-preload');
      scheduleNativeEntryCompletion(onComplete);
      return true;
    }
    document.documentElement.classList.add('site-page-transition-preload');
    document.body?.classList.add('site-page-transition-preload');
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => scheduleFallbackEntry(pending, onComplete));
    });
    return true;
  };

  const handleNavigation = (url, descriptor) => {
    if (!url || navigationLocked) return;

    navigationLocked = true;
    navigationDepartureObserved = false;
    navigationCommitStartedAt = 0;
    navigationTarget = normalizeTarget(url);
    dispatchNavigationEvent(url);
    const assignTarget = () => {
      navigationCommitStartedAt = Date.now();
      try {
        window.location.assign(url.href);
      } catch {
        navigationLocked = false;
        navigationCommitStartedAt = 0;
        navigationTarget = '';
        clearTransitionClasses();
        clearTransitionPresentation();
      }
    };
    if (prefersReducedMotion()) {
      assignTarget();
      return;
    }

    const fallbackDescriptor = { ...descriptor, transport: 'fallback' };
    storePendingNavigation(url, fallbackDescriptor, 'fallback');
    setTransitionPresentation(fallbackDescriptor);
    markNavigating();
    startFallbackExit();
    const delay = descriptor.mode === 'personal'
      ? (usesCompactTransition() ? COMPACT_FALLBACK_EXIT_MS : FALLBACK_EXIT_MS)
      : NEUTRAL_EXIT_MS;
    window.setTimeout(assignTarget, delay);
  };

  const prefetchTarget = (url) => {
    if (!url) return;
    const key = normalizeTarget(url);
    if (!key || prefetchedTargets.has(key)) return;
    prefetchedTargets.add(key);

    const tag = document.createElement('link');
    tag.rel = 'prefetch';
    tag.as = 'document';
    tag.href = url.href;
    tag.dataset.prefetch = 'page-transition';
    document.head?.appendChild(tag);
  };

  const schedulePrefetch = (url) => {
    if (!url) return;
    if (typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(() => {
        prefetchTarget(url);
      }, { timeout: 1200 });
      return;
    }
    window.setTimeout(() => {
      prefetchTarget(url);
    }, 0);
  };

  const initNavigationPrefetch = () => {
    const queueLinkPrefetch = (event) => {
      const link = event.target?.closest?.('a[href]');
      if (!link) return;
      const url = getEligibleNavigationUrl(link);
      if (url) schedulePrefetch(url);
    };

    document.addEventListener('pointerover', queueLinkPrefetch, { passive: true });
    document.addEventListener('focusin', queueLinkPrefetch);
    document.addEventListener('pointerdown', (event) => {
      const link = event.target?.closest?.('a[href]');
      if (!link || !isPersonalSurfaceNavigation(link)) return;
      const url = getEligibleNavigationUrl(link);
      if (url) prefetchTarget(url);
    }, { passive: true });
  };

  const initClickInterception = () => {
    document.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.__contactHandled) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof event.button === 'number' && event.button !== 0) return;

      const link = event.target?.closest?.('a[href]');
      if (!link) return;

      const url = getEligibleNavigationUrl(link);
      if (!url) return;
      if (hasBlockingInteractionLayer()) return;
      if (hasActiveSameDocumentTransition()) {
        event.preventDefault();
        return;
      }

      const descriptor = getTransitionDescriptor(link, url);

      if (navigationLocked) {
        event.preventDefault();
        return;
      }

      if (prefersReducedMotion()) {
        event.preventDefault();
        if (descriptor.targetIntent) storeArrivalFocus(url, descriptor.targetIntent);
        handleNavigation(url, descriptor);
        return;
      }

      if (supportsCrossDocumentViewTransitions()) {
        navigationLocked = true;
        navigationDepartureObserved = false;
        navigationCommitStartedAt = Date.now();
        navigationTarget = normalizeTarget(url);
        const nativeDescriptor = { ...descriptor, transport: 'native' };
        storePendingNavigation(url, nativeDescriptor, 'native');
        if (descriptor.targetIntent) storeArrivalFocus(url, descriptor.targetIntent);
        setTransitionPresentation(nativeDescriptor);
        dispatchNavigationEvent(url);
        return;
      }

      event.preventDefault();
      if (descriptor.targetIntent) storeArrivalFocus(url, descriptor.targetIntent);
      handleNavigation(url, descriptor);
    });
  };

  const init = () => {
    syncPersonalHistoryState();
    const arrivalIntent = consumeArrivalFocusForCurrentPage();
    const incomingTransition = hydrateIncomingTransition(() => {
      incomingTransitionActive = false;
      navigationLocked = false;
      if (arrivalIntent) scheduleArrivalFocus(arrivalIntent);
    });
    if (incomingTransition) {
      incomingTransitionActive = true;
      navigationLocked = true;
    }
    if (!incomingTransition && arrivalIntent) scheduleArrivalFocus(arrivalIntent);
    if (!incomingTransition && !arrivalIntent && isHistoryTraversal()) focusCurrentPersonalState();
    initClickInterception();
    initNavigationPrefetch();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  window.addEventListener('pagehide', () => {
    navigationDepartureObserved = true;
  });

  window.addEventListener('focus', () => {
    if (!navigationLocked || !navigationCommitStartedAt || navigationDepartureObserved) return;
    window.setTimeout(() => {
      const currentUrl = resolveUrl(window.location.href);
      if (!navigationLocked || navigationDepartureObserved || !currentUrl) return;
      if (navigationTarget && normalizeTarget(currentUrl) === navigationTarget) return;
      navigationLocked = false;
      navigationCommitStartedAt = 0;
      navigationTarget = '';
      clearTransitionClasses();
      clearTransitionPresentation();
    }, 0);
  });

  window.addEventListener('pageshow', (event) => {
    if (!incomingTransitionActive) navigationLocked = false;
    navigationDepartureObserved = false;
    navigationCommitStartedAt = 0;
    navigationTarget = '';
    syncPersonalHistoryState();
    if (event.persisted) {
      incomingTransitionActive = false;
      navigationLocked = false;
      clearTransitionClasses();
      clearTransitionPresentation();
      focusCurrentPersonalState();
    }
  });
})();
