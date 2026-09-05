(() => {
  'use strict';

  const labels = { about: 'About', projects: 'Projects', tools: 'Tools', games: 'Games', resume: 'Resume', contact: 'Contact' };
  const colors = { about: '#091f3b', projects: '#155dfc', tools: '#087f8c', games: '#c94b0a', resume: '#087f8c', contact: '#334155' };
  const cleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const cleanPath = (value) => String(value || '/').replace(/\.html$/i, '').replace(/\/+$/, '') || '/';

  function readPage() {
    let manifest = {};
    try { manifest = JSON.parse(document.querySelector('[data-site-route-manifest]')?.textContent || '{}'); } catch (_) {}
    const data = document.body.dataset;
    const state = window.SiteFrame?.current?.();
    const shell = window.SiteFrame?.root?.() || document.querySelector('[data-personal-accordion-shell], [data-home-accordion]');
    const audience = data.audience || 'personal';
    const config = window.getSiteAudienceConfig?.(audience) || {};
    const url = new URL(window.location.href);
    // Inline libraries retain the home document and its canonical manifest path.
    const path = cleanPath(manifest.id === 'home' ? url.pathname : (manifest.path || url.pathname));
    const category = state?.category || data.siteRouteCategory || data.personalCategory || manifest.category || 'about';
    const view = state?.view || data.siteRouteView || data.personalAccordionView || manifest.view;
    const routeBody = document.querySelector('[data-site-route-body], [data-personal-detail-content]') || document;
    const heading = cleanText(routeBody.querySelector('h1')?.textContent);
    const title = cleanText(document.title).replace(/\s*[|–—-]\s*Daniel Short\s*$/i, '').trim();
    const back = document.querySelector('[data-site-route-toolbar] .personal-accordion__back');
    let parent = null;
    if (data.page === 'project-demo' && back) {
      const parentUrl = new URL(back.getAttribute('href') || '/', url);
      const label = cleanText(back.getAttribute('aria-label')).replace(/^Back to\s+/i, '');
      if (parentUrl.origin === url.origin && /^\/portfolio\/[^/]+$/i.test(cleanPath(parentUrl.pathname)) && label) {
        parent = { label, href: `${parentUrl.pathname}${parentUrl.search}${parentUrl.hash}` };
      }
    }
    return {
      path, category, view, audience, config, parent,
      home: manifest.id === 'home' || data.page === 'home',
      demo: data.page === 'project-demo',
      title: heading || title || cleanText(data.page).replace(/[-_]/g, ' ') || 'Page',
      hard: (data.siteRouteNavigation || manifest.navigation) === 'hard',
      accent: shell ? getComputedStyle(shell).getPropertyValue('--panel-color').trim() : ''
    };
  }

  function buildTrail(page) {
    const homePath = page.config.homePath || '/';
    if (page.view === 'overview' || (page.home && page.view !== 'library') ||
      (page.path === cleanPath(homePath) && page.view !== 'library')) return [];
    const trail = [{ label: 'Home', href: homePath }];
    if (page.path === '/search') return [...trail, { label: 'Search' }];
    if (page.path === '/contact') return [...trail, { label: 'Contact' }];
    if (page.category === 'resume' || /^\/resume(?:-|$)/.test(page.path)) {
      if (/-pdf$/.test(page.path)) {
        trail.push({ label: 'Resume', href: page.config.resumePath || page.path.replace(/-pdf$/, '') });
        trail.push({ label: 'PDF Preview' });
      } else trail.push({ label: 'Resume' });
      return trail;
    }
    const libraryHref = { projects: page.config.portfolioPath || '/portfolio', tools: '/tools', games: '/games' }[page.category];
    if (libraryHref) {
      const label = labels[page.category];
      const libraryPath = cleanPath(new URL(libraryHref, window.location.href).pathname);
      if (page.view === 'library' || page.path === libraryPath || (page.category === 'projects' && page.path === '/projects')) {
        return [...trail, { label }];
      }
      trail.push({ label, href: libraryHref });
    }
    if (page.demo && page.parent) trail.push(page.parent, { label: 'Demo' });
    else trail.push({ label: page.title });
    return trail;
  }

  function init() {
    const nav = document.querySelector('[data-header-breadcrumbs]');
    const list = nav?.querySelector('[data-header-breadcrumb-list]');
    if (!list || nav.dataset.headerBreadcrumbsBound === 'true') return;
    nav.dataset.headerBreadcrumbsBound = 'true';
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let animation = null;
    let previous = '';

    function update(animate = true) {
      const page = readPage();
      const trail = buildTrail(page);
      const signature = JSON.stringify([trail, page.hard, page.accent, page.category]);
      if (signature === previous) return;
      previous = signature;
      animation?.cancel();
      animation = null;
      nav.style.setProperty('--breadcrumb-accent', page.accent || colors[page.category] || colors.about);
      const fragment = document.createDocumentFragment();
      trail.forEach((crumb, index) => {
        const item = document.createElement('li');
        item.className = 'header-breadcrumbs__item';
        if (index) {
          const separator = document.createElement('span');
          separator.className = 'header-breadcrumbs__separator';
          separator.setAttribute('aria-hidden', 'true');
          separator.textContent = '›';
          item.append(separator);
        }
        const current = index === trail.length - 1;
        const label = document.createElement(current ? 'span' : 'a');
        label.className = `header-breadcrumbs__${current ? 'current' : 'link'}`;
        label.textContent = crumb.label;
        if (current) label.setAttribute('aria-current', 'page');
        else {
          label.href = crumb.href;
          if (page.hard) label.dataset.navigation = 'hard';
        }
        item.append(label);
        fragment.append(item);
      });
      // Replace only after the next trail is complete; never blank the masthead between routes.
      list.replaceChildren(fragment);
      nav.hidden = !trail.length;
      const duration = window.SiteMotion?.duration(nav, '--motion-fast', 160) ?? 160;
      if (animate && trail.length && !reducedMotion.matches && duration > 0 && list.animate && nav.getClientRects().length) {
        animation = list.animate([{ transform: 'translateY(3px)' }, { transform: 'translateY(0)' }], {
          duration,
          easing: getComputedStyle(nav).getPropertyValue('--easing-standard').trim() || 'ease-out'
        });
      }
    }

    update(false);
    document.addEventListener('site:route-change', () => update());
    // Local home changes commit without entering the document router.
    document.addEventListener('home:category-change', () => update());
    reducedMotion.addEventListener?.('change', () => {
      if (reducedMotion.matches) { animation?.cancel(); animation = null; }
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
