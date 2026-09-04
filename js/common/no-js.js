(() => {
  'use strict';
  try {
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
