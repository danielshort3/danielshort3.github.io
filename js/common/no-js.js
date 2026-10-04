(() => {
  'use strict';
  try {
    if (window.location.hostname === 'danielshort3.github.io') {
      let canonicalPath = String(window.location.pathname || '/');
      let canonicalSearch = String(window.location.search || '');
      // Authored clean routes (notably tools and isolated demos) do not always
      // share the legacy file's directory. The head publishes the real route.
      try {
        const canonicalHref = document.querySelector('link[rel="canonical"]')?.getAttribute('href');
        const canonical = canonicalHref && new URL(canonicalHref, 'https://www.danielshort.me');
        // A custom 404 can be served for any requested clean route. Its own
        // canonical describes the error document, not the requested location.
        const errorDocument = canonical && /^\/404(?:\.html)?\/?$/i.test(canonical.pathname);
        if (canonical?.origin === 'https://www.danielshort.me' && !errorDocument) {
          canonicalPath = canonical.pathname;
          const incoming = new URLSearchParams(canonicalSearch);
          canonical.searchParams.forEach((value, key) => {
            if (!incoming.has(key)) {
              canonicalSearch += `${canonicalSearch ? '&' : '?'}${new URLSearchParams([[key, value]])}`;
            }
          });
        }
      } catch (_) {}
      // Preserve /pages when no canonical is available so Vercel's explicit
      // aliases can resolve it, rather than inventing a root-level tool URL.
      canonicalPath = canonicalPath.replace(/\/index\.html$/i, '/');
      canonicalPath = canonicalPath.replace(/\.html$/i, '') || '/';
      window.location.replace(`https://www.danielshort.me${canonicalPath}${canonicalSearch}${window.location.hash || ''}`);
      return;
    }

    const root = document.documentElement;
    if (!root) return;
    // Reserve the first-visit mobile banner's space before the first paint.
    // The deferred consent bundle replaces this space with its in-flow banner.
    const path = String(window.location.pathname || '').replace(/\.html$/i, '');
    if (!/^(?:\/pages\/|\/tools\/|\/)job-application-tracker$/i.test(path)) {
      let needsConsent = true;
      try {
        const query = new URLSearchParams(window.location.search || '');
        needsConsent = query.get('show_consent') === '1'
          || query.get('reset_consent') === '1'
          || !window.localStorage.getItem('pcz_consent_v1');
        if (window.self !== window.top && window.top.location.origin === window.location.origin) needsConsent = false;
      } catch (_) {}
      if (needsConsent) root.setAttribute('data-consent-reserve', 'true');
    }
    if (root.classList) {
      root.classList.remove('no-js');
      root.classList.add('js');
      [
        'site-is-navigating',
        'site-page-transition-preload',
        'site-page-transition-native-preload',
        'site-page-transition-out',
        'site-page-transition-in',
        'site-page-transition-continuous-preload',
        'site-page-transition-continuous-out',
        'site-page-transition-continuous-in'
      ].forEach((name) => root.classList.remove(name));
      [
        'siteTransitionMode',
        'siteTransitionCategory',
        'siteTransitionDirection',
        'siteTransitionTransport'
      ].forEach((name) => { delete root.dataset[name]; });
      try { window.sessionStorage.removeItem('sitePageTransition'); } catch (_) {}

      try {
        const query = new URLSearchParams(window.location.search || '');
        const audience = String(query.get('audience') || '').trim().toLowerCase();
        const mode = String(query.get('mode') || '').trim().toLowerCase();
        const professionalAudience = ['analytics', 'data-science', 'tourism'].includes(audience);
        const legacyProfessionalMode = ['professional', 'work', 'career', 'analytics'].includes(mode);
        const path = String(window.location.pathname || '/').replace(/\.html$/i, '').replace(/\/+$/, '') || '/';
        const sharedAudiencePage = path === '/portfolio' || path.startsWith('/portfolio/') || path === '/contact';
        if (sharedAudiencePage && (professionalAudience || legacyProfessionalMode)) {
          root.classList.add('site-realm-query-pending');
        }
      } catch (_) {}
      return;
    }
    root.className = (root.className || '').replace(/\bno-js\b/g, '').trim();
  } catch (_) {}
})();
