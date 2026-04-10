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

  const storePendingNavigation = (url) => {
    const storage = getStorage();
    if (!storage || !url) return;
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({
        target: normalizeTarget(url),
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
    document.documentElement.classList.remove('site-is-navigating', 'site-page-transition-out', 'site-page-transition-in');
    document.body?.classList.remove('site-is-navigating', 'site-page-transition-out', 'site-page-transition-in');
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

  const hydrateIncomingTransition = () => {
    clearTransitionClasses();
    const pending = consumePendingNavigation();
    if (!pending) {
      document.documentElement.classList.remove('site-page-transition-preload');
      return;
    }

    const currentUrl = resolveUrl(window.location.href);
    if (!currentUrl || pending.target !== normalizeTarget(currentUrl)) {
      document.documentElement.classList.remove('site-page-transition-preload');
      return;
    }
    window.requestAnimationFrame(() => {
      scheduleFallbackEntry();
    });
  };

  const handleNavigation = (url) => {
    if (!url || navigationLocked) return;

    navigationLocked = true;
    storePendingNavigation(url);
    markNavigating();
    dispatchNavigationEvent(url);
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

  const initHeaderPrefetch = () => {
    const host = document.getElementById('combined-header-nav');
    if (!host) return;

    host.querySelectorAll('a[href]').forEach((link) => {
      if (link.dataset.prefetchBound === 'yes') return;
      const url = getEligibleNavigationUrl(link);
      if (!url) return;

      link.dataset.prefetchBound = 'yes';
      const queuePrefetch = () => {
        schedulePrefetch(url);
      };
      link.addEventListener('pointerenter', queuePrefetch, { once: true });
      link.addEventListener('focus', queuePrefetch, { once: true });
    });
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

      if (navigationLocked) {
        event.preventDefault();
        return;
      }

      event.preventDefault();
      handleNavigation(url);
    });
  };

  const init = () => {
    hydrateIncomingTransition();
    initClickInterception();
    initHeaderPrefetch();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  window.addEventListener('pageshow', (event) => {
    clearTransitionClasses();
    document.documentElement.classList.remove('site-page-transition-preload');
    if (event.persisted) {
      navigationLocked = false;
    }
  });
})();
