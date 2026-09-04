(() => {
  'use strict';
  try {
    const revealState = window.__sitePageRevealState || {
      callbacks: [],
      seen: false,
      viewTransition: null
    };
    window.__sitePageRevealState = revealState;
    window.addEventListener?.('pagereveal', (event) => {
      revealState.seen = true;
      revealState.viewTransition = event?.viewTransition || null;
      const callbacks = Array.isArray(revealState.callbacks)
        ? revealState.callbacks.splice(0)
        : [];
      callbacks.forEach((callback) => {
        try {
          callback(revealState.viewTransition);
        } catch {}
      });
    }, { once: true });

    if (window.location.hostname === 'danielshort3.github.io') {
      let canonicalPath = String(window.location.pathname || '/');
      canonicalPath = canonicalPath.replace(/^\/pages\//i, '/').replace(/\/index\.html$/i, '/');
      canonicalPath = canonicalPath.replace(/\.html$/i, '') || '/';
      window.location.replace(`https://www.danielshort.me${canonicalPath}${window.location.search || ''}${window.location.hash || ''}`);
      return;
    }

    const root = document.documentElement;
    if (!root) return;
    if (root.classList) {
      root.classList.remove('no-js');
      try {
        const query = new URLSearchParams(window.location.search || '');
        const audience = String(query.get('audience') || '').trim().toLowerCase();
        const mode = String(query.get('mode') || '').trim().toLowerCase();
        const professionalAudience = ['analytics', 'data-science', 'tourism'].includes(audience);
        const legacyProfessionalMode = ['professional', 'work', 'career', 'analytics'].includes(mode);
        const path = String(window.location.pathname || '/').replace(/\.html$/i, '').replace(/\/+$/, '') || '/';
        const sharedAudiencePage = path === '/portfolio'
          || path.startsWith('/portfolio/')
          || path === '/contact';
        if (sharedAudiencePage && (professionalAudience || legacyProfessionalMode)) {
          root.classList.add('site-realm-query-pending');
        }
      } catch {}
      const STORAGE_KEY = 'sitePageTransition';
      const TRANSITION_TTL_MS = 30000;
      const TRANSITION_MODES = ['personal', 'neutral'];
      const TRANSITION_CATEGORIES = ['about', 'projects', 'tools', 'games', 'contact', 'neutral'];
      const TRANSITION_DIRECTIONS = ['forward', 'back', 'cross', 'replace'];
      const TRANSITION_TRANSPORTS = ['fallback', 'native'];
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
          const reducedMotion = Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
          const legacyMode = payload?.mode === 'continuous'
            ? 'personal'
            : payload?.mode === 'fade' ? 'neutral' : payload?.mode;
          const mode = TRANSITION_MODES.includes(legacyMode) ? legacyMode : 'neutral';
          const category = TRANSITION_CATEGORIES.includes(payload?.category) ? payload.category : 'neutral';
          const direction = TRANSITION_DIRECTIONS.includes(payload?.direction) ? payload.direction : 'replace';
          const transport = TRANSITION_TRANSPORTS.includes(payload?.transport) ? payload.transport : 'fallback';
          if (isFresh && matchesCurrent && !reducedMotion) {
            root.dataset.siteTransitionMode = mode;
            root.dataset.siteTransitionCategory = category;
            root.dataset.siteTransitionDirection = direction;
            root.dataset.siteTransitionTransport = transport;
            root.classList.add(transport === 'native'
              ? 'site-page-transition-native-preload'
              : 'site-page-transition-preload');
          }
        }
      } catch {}
      return;
    }
    root.className = (root.className || '').replace(/\bno-js\b/g, '').trim();
  } catch (_) {}
})();
