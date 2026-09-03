/* ===================================================================
   File: page-transitions.js
   Purpose: Smooth same-origin page transitions and header prefetching
=================================================================== */
(() => {
  'use strict';

  const NAVIGATION_EVENT = 'site:navigation-start';
  const STORAGE_KEY = 'sitePageTransition';
  const TRANSITION_TTL_MS = 4000;
  const FALLBACK_EXIT_MS = 220;
  const FALLBACK_EXIT_REDUCED_MS = 80;
  const FALLBACK_ENTRY_MS = 260;
  const FALLBACK_ENTRY_REDUCED_MS = 120;
  const CONTINUOUS_EXIT_MS = 90;
  let navigationLocked = false;
  const prefetchedTargets = new Set();

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

  const storePendingNavigation = (url, mode = 'fade') => {
    const storage = getStorage();
    if (!storage || !url) return;
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({
        target: normalizeTarget(url),
        mode,
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
      return payload;
    } catch {
      return null;
    }
  };

  const clearTransitionClasses = () => {
    document.documentElement.classList.remove(
      'site-is-navigating',
      'site-page-transition-out',
      'site-page-transition-in',
      'site-page-transition-continuous-out',
      'site-page-transition-continuous-in'
    );
    document.body?.classList.remove(
      'site-is-navigating',
      'site-page-transition-out',
      'site-page-transition-in',
      'site-page-transition-continuous-out',
      'site-page-transition-continuous-in'
    );
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

  const startContinuousExit = () => {
    document.documentElement.classList.add('site-page-transition-continuous-out');
    document.body?.classList.add('site-page-transition-continuous-out');
  };

  const scheduleFallbackEntry = () => {
    document.documentElement.classList.remove('site-page-transition-preload');
    document.documentElement.classList.add('site-page-transition-in');
    document.body?.classList.add('site-page-transition-in');
    const cleanupDelay = prefersReducedMotion() ? FALLBACK_ENTRY_REDUCED_MS : FALLBACK_ENTRY_MS;
    window.setTimeout(() => {
      document.documentElement.classList.remove('site-page-transition-in');
      document.body?.classList.remove('site-page-transition-in');
    }, cleanupDelay);
  };

  const scheduleContinuousEntry = () => {
    document.documentElement.classList.remove('site-page-transition-continuous-preload');
    document.documentElement.classList.add('site-page-transition-continuous-in');
    document.body?.classList.add('site-page-transition-continuous-in');
    window.setTimeout(() => {
      document.documentElement.classList.remove('site-page-transition-continuous-in');
      document.body?.classList.remove('site-page-transition-continuous-in');
    }, FALLBACK_ENTRY_MS);
  };

  const hydrateIncomingTransition = () => {
    clearTransitionClasses();
    const pending = consumePendingNavigation();
    if (!pending) {
      document.documentElement.classList.remove('site-page-transition-preload');
      document.documentElement.classList.remove('site-page-transition-continuous-preload');
      return;
    }

    const currentUrl = resolveUrl(window.location.href);
    if (!currentUrl || pending.target !== normalizeTarget(currentUrl)) {
      document.documentElement.classList.remove('site-page-transition-preload');
      document.documentElement.classList.remove('site-page-transition-continuous-preload');
      return;
    }
    window.requestAnimationFrame(() => {
      if (pending.mode === 'continuous') scheduleContinuousEntry();
      else scheduleFallbackEntry();
    });
  };

  const handleNavigation = (url, options = {}) => {
    if (!url || navigationLocked) return;

    const continuous = options.continuous === true;
    navigationLocked = true;
    dispatchNavigationEvent(url);
    if (continuous && prefersReducedMotion()) {
      window.location.assign(url.href);
      return;
    }

    storePendingNavigation(url, continuous ? 'continuous' : 'fade');
    markNavigating();
    if (continuous) {
      startContinuousExit();
      window.setTimeout(() => {
        window.location.assign(url.href);
      }, CONTINUOUS_EXIT_MS);
      return;
    }

    startFallbackExit();
    const delay = prefersReducedMotion() ? FALLBACK_EXIT_REDUCED_MS : FALLBACK_EXIT_MS;
    window.setTimeout(() => {
      window.location.assign(url.href);
    }, delay);
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
  };

  const initClickInterception = () => {
    document.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.__contactHandled) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof event.button === 'number' && event.button !== 0) return;

      const link = event.target.closest('a[href]');
      if (!link) return;

      const url = getEligibleNavigationUrl(link);
      if (!url) return;

      const personalSurfaceNavigation = isPersonalSurfaceNavigation(link);

      if (personalSurfaceNavigation && supportsCrossDocumentViewTransitions()) {
        dispatchNavigationEvent(url);
        return;
      }

      if (navigationLocked) {
        event.preventDefault();
        return;
      }

      event.preventDefault();
      handleNavigation(url, { continuous: personalSurfaceNavigation });
    });
  };

  const init = () => {
    hydrateIncomingTransition();
    initClickInterception();
    initNavigationPrefetch();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  window.addEventListener('pageshow', (event) => {
    clearTransitionClasses();
    document.documentElement.classList.remove('site-page-transition-preload');
    document.documentElement.classList.remove('site-page-transition-continuous-preload');
    if (event.persisted) {
      navigationLocked = false;
    }
  });
})();
