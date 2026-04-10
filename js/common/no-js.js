(() => {
  'use strict';
  try {
    const root = document.documentElement;
    if (!root) return;
    if (root.classList) {
      root.classList.remove('no-js');
      const STORAGE_KEY = 'sitePageTransition';
      const TRANSITION_TTL_MS = 4000;
      const normalizePathname = (pathname) => {
        let next = String(pathname || '/');
        next = next.replace(/\/index\.html$/i, '/');
        next = next.replace(/\.html$/i, '');
        next = next.replace(/\/+$/, '');
        if (!next) next = '/';
        return next;
      };
      const normalizeTarget = (urlLike) => {
        try {
          const url = new URL(String(urlLike || window.location.href), window.location.href);
          return `${url.origin}${normalizePathname(url.pathname)}${url.search}`;
        } catch {
          return '';
        }
      };

      try {
        const raw = window.sessionStorage.getItem(STORAGE_KEY);
        if (raw) {
          const payload = JSON.parse(raw);
          const isFresh = payload && Number.isFinite(payload.ts) && (Date.now() - payload.ts) <= TRANSITION_TTL_MS;
          const matchesCurrent = payload && typeof payload.target === 'string' && payload.target === normalizeTarget(window.location.href);
          if (isFresh && matchesCurrent) {
            root.classList.add('site-page-transition-preload');
          }
        }
      } catch {}
      return;
    }
    root.className = (root.className || '').replace(/\bno-js\b/g, '').trim();
  } catch (_) {}
})();
