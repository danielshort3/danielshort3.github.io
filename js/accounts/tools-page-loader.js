(() => {
  'use strict';

  const LOADER_ID = 'tools-account-loader';
  const FALLBACK_SRC = 'dist/site-tools-account.js';
  const currentScript = document.currentScript;
  const managedSrc = String(currentScript?.dataset?.toolsAccountSrc || '').trim() || FALLBACK_SRC;
  const existingController = window.__toolsAccountLoaderController;

  if (existingController?.version) {
    existingController.sync(managedSrc);
    return;
  }

  let accountBundleSrc = managedSrc;
  let loadPromise = null;
  let activeBinding = null;

  const loadAccountBundle = () => {
    if (window.__toolsAccountBundleLoaded) {
      window.__toolsAccountUiController?.sync?.();
      return Promise.resolve();
    }
    if (loadPromise) return loadPromise;

    const existing = document.getElementById(LOADER_ID);
    loadPromise = new Promise((resolve, reject) => {
      const handleLoad = () => {
        window.__toolsAccountBundleLoaded = true;
        window.__toolsAccountUiController?.sync?.();
        resolve();
      };
      const handleError = () => {
        existing?.remove();
        reject(new Error(`Failed to load tools account bundle: ${accountBundleSrc}`));
      };

      if (existing) {
        existing.addEventListener('load', handleLoad, { once: true });
        existing.addEventListener('error', handleError, { once: true });
        return;
      }

      const tag = document.createElement('script');
      tag.id = LOADER_ID;
      tag.src = accountBundleSrc;
      tag.defer = true;
      tag.addEventListener('load', handleLoad, { once: true });
      tag.addEventListener('error', () => {
        tag.remove();
        reject(new Error(`Failed to load tools account bundle: ${accountBundleSrc}`));
      }, { once: true });
      document.head.appendChild(tag);
    }).catch((err) => {
      loadPromise = null;
      throw err;
    });

    return loadPromise;
  };

  const releaseBinding = (binding = activeBinding) => {
    if (!binding || binding.released) return;
    binding.released = true;
    if (binding.idleTimer) window.clearTimeout(binding.idleTimer);
    binding.observer?.disconnect();
    binding.dock.removeEventListener('pointerenter', binding.triggerLoad);
    binding.dock.removeEventListener('focusin', binding.triggerLoad);
    binding.dock.removeEventListener('click', binding.triggerLoad);
    binding.dock.removeEventListener('keydown', binding.triggerLoad);
    delete binding.dock.dataset.toolsAccountLoaderReady;
    if (activeBinding === binding) activeBinding = null;
  };

  const bindCurrentDock = () => {
    const dock = document.querySelector('[data-tools-account="dock"]');
    if (activeBinding?.dock === dock && !activeBinding.released) {
      if (window.__toolsAccountBundleLoaded) window.__toolsAccountUiController?.sync?.();
      return;
    }
    releaseBinding();
    if (!dock || dock.dataset.toolsAccountLoaderReady === 'true') return;

    const binding = {
      dock,
      idleTimer: 0,
      observer: null,
      released: false,
      triggerLoad: null
    };
    activeBinding = binding;
    dock.dataset.toolsAccountLoaderReady = 'true';

    binding.triggerLoad = () => {
      if (binding.released) return;
      if (binding.idleTimer) {
        window.clearTimeout(binding.idleTimer);
        binding.idleTimer = 0;
      }
      binding.observer?.disconnect();
      binding.observer = null;
      dock.removeEventListener('pointerenter', binding.triggerLoad);
      dock.removeEventListener('focusin', binding.triggerLoad);
      dock.removeEventListener('click', binding.triggerLoad);
      dock.removeEventListener('keydown', binding.triggerLoad);
      loadAccountBundle().catch((err) => {
        try {
          console.warn('[tools-page-loader]', err);
        } catch {}
      });
    };

    dock.addEventListener('pointerenter', binding.triggerLoad, { once: true });
    dock.addEventListener('focusin', binding.triggerLoad, { once: true });
    dock.addEventListener('click', binding.triggerLoad, { once: true });
    dock.addEventListener('keydown', binding.triggerLoad, { once: true });

    const routeId = String(
      window.SiteRoutes?.current?.()?.id ||
      document.body?.dataset?.siteRouteId ||
      ''
    ).trim();
    window.SiteRoutes?.addCleanup?.(() => releaseBinding(binding), routeId);

    if (window.__toolsAccountBundleLoaded ||
        document.body?.dataset?.page === 'tools' ||
        document.body?.dataset?.toolsLayout === 'directory') {
      binding.triggerLoad();
      return;
    }

    if ('IntersectionObserver' in window) {
      binding.observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        binding.idleTimer = window.setTimeout(binding.triggerLoad, 2200);
      }, { threshold: 0.75 });
      binding.observer.observe(dock);
    } else {
      binding.idleTimer = window.setTimeout(binding.triggerLoad, 2200);
    }
  };

  const controller = Object.freeze({
    version: 2,
    sync: (nextSrc = '') => {
      if (String(nextSrc || '').trim()) accountBundleSrc = String(nextSrc).trim();
      bindCurrentDock();
    }
  });
  window.__toolsAccountLoaderController = controller;

  document.addEventListener('site:route-mounted', bindCurrentDock);
  document.addEventListener('site:route-unmounted', () => releaseBinding());

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindCurrentDock, { once: true });
  } else {
    bindCurrentDock();
  }
})();
