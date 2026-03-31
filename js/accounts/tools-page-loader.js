(() => {
  'use strict';

  const LOADER_ID = 'tools-account-loader';
  const FALLBACK_SRC = 'dist/site-tools-account.js';
  const currentScript = document.currentScript;
  const managedSrc = String(currentScript?.dataset?.toolsAccountSrc || '').trim() || FALLBACK_SRC;
  const dock = document.querySelector('[data-tools-account="dock"]');

  if (!dock || window.__toolsAccountLoaderReady) return;
  window.__toolsAccountLoaderReady = true;

  let loadPromise = null;
  let idleTimer = null;
  let observer = null;

  const cleanup = () => {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    if (observer) {
      observer.disconnect();
      observer = null;
    }
    dock.removeEventListener('pointerenter', triggerLoad);
    dock.removeEventListener('focusin', triggerLoad);
    dock.removeEventListener('click', triggerLoad);
    dock.removeEventListener('keydown', triggerLoad);
  };

  function loadAccountBundle() {
    if (window.__toolsAccountBundleLoaded) return Promise.resolve();
    if (loadPromise) return loadPromise;

    const existing = document.getElementById(LOADER_ID);
    if (existing) {
      loadPromise = Promise.resolve();
      return loadPromise;
    }

    loadPromise = new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.id = LOADER_ID;
      tag.src = managedSrc;
      tag.defer = true;
      tag.onload = () => {
        window.__toolsAccountBundleLoaded = true;
        resolve();
      };
      tag.onerror = () => reject(new Error(`Failed to load tools account bundle: ${managedSrc}`));
      document.head.appendChild(tag);
    }).catch((err) => {
      loadPromise = null;
      throw err;
    });

    return loadPromise;
  }

  function triggerLoad() {
    cleanup();
    loadAccountBundle().catch((err) => {
      try {
        console.warn('[tools-page-loader]', err);
      } catch {}
    });
  }

  dock.addEventListener('pointerenter', triggerLoad, { once: true });
  dock.addEventListener('focusin', triggerLoad, { once: true });
  dock.addEventListener('click', triggerLoad, { once: true });
  dock.addEventListener('keydown', triggerLoad, { once: true });

  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      idleTimer = window.setTimeout(triggerLoad, 2200);
    }, { threshold: 0.75 });
    observer.observe(dock);
  } else {
    idleTimer = window.setTimeout(triggerLoad, 2200);
  }
})();
