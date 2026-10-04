(() => {
  'use strict';

  const PERSONAL = 'personal';
  const LEGACY_AUDIENCES = new Set(['analytics', 'data-science', 'datascience', 'data_science', 'tourism', 'tourism-analytics']);
  const LEGACY_MODES = new Set(['professional', 'work', 'career', 'analytics']);

  function clearLegacyRouteContext() {
    const url = new URL(window.location.href);
    const path = url.pathname.replace(/\.html$/i, '') || '/';
    if (!/^(?:\/|\/portfolio(?:\/.*)?|\/contact|\/search)$/.test(path)) return;
    const audience = String(url.searchParams.get('audience') || '').toLowerCase();
    const mode = String(url.searchParams.get('mode') || '').toLowerCase();
    if (LEGACY_AUDIENCES.has(audience)) url.searchParams.delete('audience');
    if (LEGACY_MODES.has(mode)) url.searchParams.delete('mode');
    const next = `${url.pathname}${url.search}${url.hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== current) window.history.replaceState(window.history.state, '', next);
  }

  function sync() {
    clearLegacyRouteContext();
    const root = document.documentElement;
    root.classList.remove('site-realm-professional', 'site-realm-query-pending', 'site-realm-professional-home');
    root.classList.add('site-realm-personal');
    if (document.body) {
      document.body.dataset.siteRealm = PERSONAL;
      document.body.dataset.audience = PERSONAL;
      delete document.body.dataset.siteRealmHome;
      document.body.classList.remove('professional-home-page', 'professional-contact-page');
    }
    document.querySelectorAll('input[data-search-audience]').forEach((input) => input.remove());
    document.querySelectorAll('[data-entry-home-link="true"]').forEach((link) => link.setAttribute('href', '/'));
    document.head?.querySelector('meta[data-site-realm-robots="professional"]')?.remove();
    window.SITE_REALM = PERSONAL;
    window.SITE_AUDIENCE = PERSONAL;
    window.getSiteRealm = () => PERSONAL;
    window.getSiteAudience = () => PERSONAL;
    window.isProfessionalRealm = () => false;
    return PERSONAL;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', sync, { once: true });
  else sync();
  document.addEventListener('site:route-before-mount', sync);
  document.addEventListener('site:route-mounted', sync);
  window.SiteRealm = Object.freeze({ sync });
})();
