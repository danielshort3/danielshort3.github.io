/* ===================================================================
   File: navigation.js
   Purpose: Enhances the shared brand/search masthead and measures its layout
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const NAVIGATION_EVENT = 'site:navigation-start';
  const NAV_HEIGHT_FALLBACK = 60;
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[char]);
  let cachedNavHeight = null;
  let navHeightRaf = null;
  let navResizeObserver = null;

  const measureNavHeight = () => {
    const visibleHeights = $$('.mobile-site-masthead, #combined-header-nav .nav')
      .map((header) => header.getBoundingClientRect().height)
      .filter((height) => Number.isFinite(height) && height > 0);
    return visibleHeights.length ? Math.max(...visibleHeights) : NAV_HEIGHT_FALLBACK;
  };

  const setCssNavHeight = (value) => {
    document.documentElement.style.setProperty('--nav-height', `${value}px`);
    window.__navHeight = value;
  };

  const emitNavHeightChange = (value) => {
    try {
      document.dispatchEvent(new CustomEvent('navheightchange', { detail: value }));
    } catch {
      const evt = document.createEvent('CustomEvent');
      evt.initCustomEvent('navheightchange', false, false, value);
      document.dispatchEvent(evt);
    }
  };

  window.getNavOffset = () => {
    if (typeof window.__navHeight === 'number' && window.__navHeight > 0) {
      return window.__navHeight;
    }
    const measured = measureNavHeight();
    setCssNavHeight(measured);
    cachedNavHeight = measured;
    return measured;
  };

  document.addEventListener('DOMContentLoaded', () => {
    initNav();
    setNavHeight();
    setupNavHeightObservers();
    window.addEventListener('load', setNavHeight);
    window.addEventListener('resize', setNavHeight);
    window.addEventListener('orientationchange', setNavHeight);
  });
  function setNavHeight(){
    const next = measureNavHeight();
    if (!Number.isFinite(next) || next <= 0) return;
    if (cachedNavHeight !== null && Math.abs(next - cachedNavHeight) < 0.5) return;
    cachedNavHeight = next;
    setCssNavHeight(next);
    emitNavHeightChange(next);
  }
  function scheduleNavHeightUpdate(){
    if (navHeightRaf !== null) return;
    const requestFrame = window.requestAnimationFrame || ((fn) => window.setTimeout(fn, 16));
    navHeightRaf = requestFrame(() => {
      navHeightRaf = null;
      setNavHeight();
    });
  }
  function setupNavHeightObservers(){
    const headers = $$('.mobile-site-masthead, #combined-header-nav .nav');
    if (!headers.length) return;

    if (navResizeObserver) {
      try { navResizeObserver.disconnect(); } catch {}
      navResizeObserver = null;
    }

    if (typeof ResizeObserver === 'function') {
      navResizeObserver = new ResizeObserver(() => {
        scheduleNavHeightUpdate();
      });
      headers.forEach((header) => navResizeObserver.observe(header));
    }

    if (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === 'function') {
      document.fonts.ready
        .then(() => {
          scheduleNavHeightUpdate();
        })
        .catch(() => {});
    }
  }

  const MOBILE_MASTHEAD_SEARCH_ICON = `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5"></circle>
      <path d="m16.2 16.2 4.3 4.3"></path>
    </svg>
  `;

  function setupMobileSiteMasthead(config) {
    if (!document.body || document.querySelector('[data-mobile-site-masthead]')) return;

    const { entryHome, currentPathVariants, activeAudience } = config;
    const isHome = (currentPathVariants || []).includes('/') || document.body?.dataset?.page === 'home';
    const masthead = document.createElement('header');
    masthead.className = `mobile-site-masthead${isHome ? ' mobile-site-masthead--home' : ''}`;
    masthead.dataset.mobileSiteMasthead = '';
    masthead.innerHTML = `
      <div class="mobile-site-masthead__inner">
        <a class="mobile-site-masthead__brand" data-entry-home-link="true" href="${escapeHtml(entryHome || '/')}" aria-label="Daniel Short home">
          <img src="img/brand/00-ds-logo-master-full-color.svg" srcset="img/brand/00-ds-logo-master-full-color.svg 1x" sizes="40px" alt="Daniel Short DS logo" class="mobile-site-masthead__logo" decoding="async" loading="eager" width="381" height="392">
          <span class="mobile-site-masthead__name">
            <span class="mobile-site-masthead__title">Daniel Short</span>
          </span>
        </a>
        <div class="mobile-site-masthead__explore" data-mobile-explore>
          <button class="mobile-site-masthead__explore-button" type="button" aria-expanded="false" aria-controls="mobile-explore-links">
            <span>Explore</span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>
          </button>
          <nav id="mobile-explore-links" class="mobile-site-masthead__explore-links" aria-label="Explore sections" hidden>
            ${['about', 'projects', 'tools', 'games', 'contact'].map((category) => `
              <a href="/#${category}" data-mobile-explore-category="${category}"><span class="mobile-site-masthead__section-dot" aria-hidden="true"></span><span>${category.charAt(0).toUpperCase() + category.slice(1)}</span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"></path></svg></a>
            `).join('')}
          </nav>
        </div>
        <form class="mobile-site-masthead__search" action="/search" method="get" role="search" data-mobile-masthead-search="collapsed">
          <label class="visually-hidden" for="mobile-masthead-search-q">Search site</label>
          <input id="mobile-masthead-search-q" class="mobile-site-masthead__search-input" type="search" name="q" placeholder="Search" autocomplete="off">
          <button class="mobile-site-masthead__search-button" type="submit" aria-controls="mobile-masthead-search-q" aria-expanded="false" aria-label="Open search">
            ${MOBILE_MASTHEAD_SEARCH_ICON}
          </button>
        </form>
      </div>
    `;

    let mastheadRaf = null;
    const syncMastheadSurface = () => {
      mastheadRaf = null;
      masthead.classList.toggle('is-scrolled', window.scrollY > 8);
    };
    const queueMastheadSurface = () => {
      if (mastheadRaf !== null) return;
      mastheadRaf = window.requestAnimationFrame(syncMastheadSurface);
    };

    window.addEventListener('scroll', queueMastheadSurface, { passive: true });
    syncMastheadSurface();

    const searchForm = masthead.querySelector('.mobile-site-masthead__search');
    const searchInput = masthead.querySelector('.mobile-site-masthead__search-input');
    const searchButton = masthead.querySelector('.mobile-site-masthead__search-button');
    const brand = masthead.querySelector('.mobile-site-masthead__brand');
    const explore = masthead.querySelector('[data-mobile-explore]');
    const exploreButton = explore.querySelector('button');
    const exploreLinks = explore.querySelector('nav');
    const mobileQuery = window.matchMedia('(max-width: 768px)');
    const setExploreExpanded = (expanded, options = {}) => {
      const nextExpanded = Boolean(expanded && mobileQuery.matches && !explore.hidden);
      exploreButton.setAttribute('aria-expanded', String(nextExpanded));
      exploreLinks.hidden = !nextExpanded;
      if (options.restoreFocus) exploreButton.focus({ preventScroll: true });
    };
    const setSearchExpanded = (expanded, options = {}) => {
      if (!searchForm || !searchInput || !searchButton) return;
      const nextExpanded = Boolean(expanded);
      if (nextExpanded) setExploreExpanded(false);
      masthead.classList.toggle('is-search-expanded', nextExpanded);
      brand.inert = nextExpanded;
      explore.inert = nextExpanded;
      searchForm.classList.toggle('is-expanded', nextExpanded);
      searchForm.dataset.mobileMastheadSearch = nextExpanded ? 'expanded' : 'collapsed';
      searchButton.setAttribute('aria-expanded', String(nextExpanded));
      searchButton.setAttribute('aria-label', nextExpanded ? 'Search site' : 'Open search');
      searchInput.tabIndex = nextExpanded ? 0 : -1;
      searchInput.setAttribute('aria-hidden', nextExpanded ? 'false' : 'true');
      if (options.focusInput && nextExpanded) {
        requestAnimationFrame(() => {
          if (searchForm.classList.contains('is-expanded')) searchInput.focus();
        });
      }
    };

    exploreButton.addEventListener('click', () => {
      setSearchExpanded(false);
      setExploreExpanded(exploreLinks.hidden);
    });
    exploreLinks.addEventListener('click', (event) => {
      const link = event.target.closest('[data-mobile-explore-category]');
      if (!link || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.button > 0) return;
      setExploreExpanded(false, { restoreFocus: true });
      const request = new CustomEvent('home:category-select', { cancelable: true, detail: { category: link.dataset.mobileExploreCategory } });
      if (window.SiteFrame?.root()?.dispatchEvent(request) === false) event.preventDefault();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || exploreLinks.hidden) return;
      event.preventDefault();
      setExploreExpanded(false, { restoreFocus: true });
    });
    document.addEventListener('pointerdown', (event) => {
      if (!exploreLinks.hidden && !explore.contains(event.target)) setExploreExpanded(false);
    }, true);
    explore.addEventListener('focusout', (event) => {
      if (event.relatedTarget && !explore.contains(event.relatedTarget)) setExploreExpanded(false);
    });
    mobileQuery.addEventListener('change', () => {
      const hadFocus = masthead.contains(document.activeElement);
      setExploreExpanded(false);
      setSearchExpanded(false);
      if (hadFocus && !mobileQuery.matches) {
        document.querySelector('#combined-header-nav .brand')?.focus({ preventScroll: true });
      }
    });
    masthead.__closeExplore = () => setExploreExpanded(false);
    masthead.__syncExplore = (context) => {
      explore.hidden = context.activeAudience.key !== 'personal';
      if (explore.hidden) setExploreExpanded(false);
      const state = window.SiteFrame?.current();
      const category = state?.view === 'closed' ? '' : state?.category || document.body.dataset.siteRouteCategory;
      exploreLinks.querySelectorAll('[data-mobile-explore-category]').forEach((link) => {
        if (link.dataset.mobileExploreCategory === category) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
      });
    };
    document.addEventListener('home:category-change', () => masthead.__syncExplore(getNavigationContext()));

    if (searchForm && searchInput && searchButton) {
      searchForm.__closeSearch = () => setSearchExpanded(false);
      setSearchExpanded(false);
      searchForm.addEventListener('submit', (event) => {
        if (!searchForm.classList.contains('is-expanded')) {
          event.preventDefault();
          setSearchExpanded(true, { focusInput: true });
          return;
        }
        if (!searchInput.value.trim()) {
          event.preventDefault();
          searchInput.focus();
        }
      });
      searchForm.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape' || !searchForm.classList.contains('is-expanded')) return;
        event.preventDefault();
        searchInput.value = '';
        setSearchExpanded(false);
        searchButton.focus();
      });
      document.addEventListener('pointerdown', (event) => {
        if (!searchForm.classList.contains('is-expanded') || searchForm.contains(event.target)) return;
        setSearchExpanded(false);
      }, true);
    }

    syncSearchAudience(searchForm, activeAudience);
    const desktopHeader = document.querySelector('#combined-header-nav');
    desktopHeader.parentNode.insertBefore(masthead, desktopHeader);
    document.body.classList.add('has-mobile-site-masthead');
  }

  function getNavigationContext() {
    const audienceApi = window.SITE_AUDIENCE_CONFIG || null;
    const normalizeAudience = audienceApi && typeof audienceApi.normalizeAudience === 'function'
      ? audienceApi.normalizeAudience
      : (() => 'personal');
    const getAudience = audienceApi && typeof audienceApi.getAudience === 'function'
      ? audienceApi.getAudience
      : (() => ({
          key: 'personal',
          homePath: '/',
          portfolioPath: '/portfolio',
          portfolioAllPath: '/portfolio'
        }));
    const detectAudienceFromPath = audienceApi && typeof audienceApi.detectAudienceFromPath === 'function'
      ? audienceApi.detectAudienceFromPath
      : (() => null);

    const AUDIENCE_KEY = 'siteAudience';
    const ENTRY_HOME_KEY = 'entryHome';
    const readSession = (key) => {
      try {
        return window.sessionStorage.getItem(key);
      } catch {
        return null;
      }
    };
    const writeSession = (key, value) => {
      try {
        window.sessionStorage.setItem(key, value);
      } catch {}
    };

    const normalizePath = (value) => {
      if (!value) return '/';
      let next = value;
      try {
        next = new URL(next, location.href).pathname;
      } catch {
        if (!next.startsWith('/')) next = `/${next}`;
      }
      next = next.replace(/\/index\.html$/i, '/');
      next = next.replace(/\/+$/, '');
      if (!next) next = '/';
      return next;
    };

    const currentPath = normalizePath(location.pathname);
    const altCurrentPath = currentPath.startsWith('/pages/')
      ? normalizePath(currentPath.replace(/^\/pages/, '') || '/')
      : currentPath;
    const currentPathVariants = [...new Set(
      [currentPath, altCurrentPath].flatMap((path) => {
        const variants = [path];
        if (path.endsWith('.html')) {
          variants.push(normalizePath(path.replace(/\.html$/i, '') || '/'));
        }
        return variants;
      })
    )];
    const queryAudience = (() => {
      try {
        const params = new URLSearchParams(window.location.search || '');
        return params.get('audience');
      } catch {
        return null;
      }
    })();
    const bodyAudience = document.body && document.body.dataset
      ? document.body.dataset.audience
      : '';
    const siteRealm = document.body && document.body.dataset
      ? String(document.body.dataset.siteRealm || '').trim().toLowerCase()
      : '';
    const pathAudience = currentPathVariants
      .map((path) => detectAudienceFromPath(path))
      .find(Boolean);
    const explicitAudienceCandidates = [queryAudience, pathAudience, bodyAudience].filter(Boolean);
    const explicitProfessionalAudience = explicitAudienceCandidates
      .find((audience) => normalizeAudience(audience) !== 'personal');
    const explicitAudience = explicitProfessionalAudience || explicitAudienceCandidates[0] || '';
    const realmAudience = siteRealm === 'professional'
      ? 'analytics'
      : (siteRealm === 'personal' ? 'personal' : '');
    const storedAudience = readSession(AUDIENCE_KEY);
    const activeAudience = getAudience(explicitAudience || realmAudience || storedAudience);
    const activeAudienceKey = normalizeAudience(activeAudience && activeAudience.key);
    const isRootHome = currentPathVariants.includes('/');
    const entryHome = isRootHome ? '/' : String(activeAudience.homePath || '/');

    writeSession(AUDIENCE_KEY, activeAudienceKey);
    writeSession(ENTRY_HOME_KEY, entryHome);

    return { activeAudience, currentPathVariants, entryHome };
  }

  function syncSearchAudience(form, audience) {
    if (!form || !audience) return;
    let input = form.querySelector('[data-search-audience]');
    if (!audience.key || audience.key === 'personal') {
      if (input) input.remove();
      return;
    }
    if (!input) {
      input = document.createElement('input');
      input.type = 'hidden';
      input.name = 'audience';
      input.dataset.searchAudience = '';
      form.appendChild(input);
    }
    input.value = audience.key;
  }

  function syncHeaderContext() {
    const config = getNavigationContext();
    const host = $('#combined-header-nav');
    const masthead = $('[data-mobile-site-masthead]');
    [host, masthead].filter(Boolean).forEach((header) => {
      $$('[data-entry-home-link="true"]', header).forEach((link) => link.setAttribute('href', config.entryHome));
      syncSearchAudience(header.querySelector('form[role="search"]'), config.activeAudience);
    });
    if (host) host.dataset.siteRealmNav = config.activeAudience.key;
    if (masthead) {
      masthead.classList.toggle('mobile-site-masthead--home', config.currentPathVariants.includes('/') || document.body?.dataset?.page === 'home');
      masthead.__syncExplore?.(config);
    }
    scheduleNavHeightUpdate();
    return config;
  }

  function initNav() {
    const host = $('#combined-header-nav');
    if (!host || !host.querySelector('.nav') || host.dataset.mastheadEnhanced === 'true') return;
    host.dataset.mastheadEnhanced = 'true';
    setupMobileSiteMasthead(getNavigationContext());
    setupHeaderSearch(host);
    syncHeaderContext();
    document.addEventListener(NAVIGATION_EVENT, () => {
      closeHeaderSearch(host);
      const mobileSearch = document.querySelector('.mobile-site-masthead__search');
      if (mobileSearch && typeof mobileSearch.__closeSearch === 'function') mobileSearch.__closeSearch();
      document.querySelector('[data-mobile-site-masthead]')?.__closeExplore?.();
    });
    document.addEventListener('site:route-change', syncHeaderContext);
  }

  function setupHeaderSearch(host) {
    const form = host && host.querySelector('.nav-search');
    if (!form || form.__navSearchReady) return;
    const input = form.querySelector('.nav-search-input');
    const button = form.querySelector('.nav-search-button');
    if (!input || !button) return;
    form.__navSearchReady = true;
    const desktopMatcher = window.matchMedia('(min-width: 769px)');

    const setExpanded = (expanded, options = {}) => {
      const { focusInput = false, restoreButtonFocus = false } = options;
      const enhanced = Boolean(desktopMatcher.matches);
      const nextExpanded = enhanced && Boolean(expanded);
      form.classList.toggle('nav-search-is-enhanced', enhanced);
      form.classList.toggle('is-expanded', nextExpanded);
      form.dataset.navSearch = enhanced ? (nextExpanded ? 'expanded' : 'collapsed') : 'full';
      button.setAttribute('aria-expanded', String(nextExpanded));
      button.setAttribute('aria-label', enhanced && !nextExpanded ? 'Open search' : 'Search site');
      input.tabIndex = enhanced && !nextExpanded ? -1 : 0;
      input.setAttribute('aria-hidden', enhanced && !nextExpanded ? 'true' : 'false');
      if (focusInput && nextExpanded) {
        requestAnimationFrame(() => {
          if (form.classList.contains('is-expanded')) input.focus();
        });
      } else if (restoreButtonFocus && enhanced) {
        button.focus();
      }
    };

    form.__closeSearch = () => setExpanded(false);
    setExpanded(false);

    form.addEventListener('submit', (event) => {
      if (!desktopMatcher.matches) return;
      if (!form.classList.contains('is-expanded')) {
        event.preventDefault();
        setExpanded(true, { focusInput: true });
        return;
      }
      if (!input.value.trim()) {
        event.preventDefault();
        input.focus();
      }
    });

    input.addEventListener('focus', () => {
      if (desktopMatcher.matches && !form.classList.contains('is-expanded')) {
        setExpanded(true);
      }
    });

    form.addEventListener('keydown', (event) => {
      if (!desktopMatcher.matches || event.key !== 'Escape') return;
      if (!form.classList.contains('is-expanded')) return;
      event.preventDefault();
      setExpanded(false, { restoreButtonFocus: true });
    });

    document.addEventListener('pointerdown', (event) => {
      if (!desktopMatcher.matches || !form.classList.contains('is-expanded')) return;
      if (form.contains(event.target)) return;
      setExpanded(false);
    }, true);

    const syncMode = () => setExpanded(false);
    if (typeof desktopMatcher.addEventListener === 'function') {
      desktopMatcher.addEventListener('change', syncMode);
    } else if (typeof desktopMatcher.addListener === 'function') {
      desktopMatcher.addListener(syncMode);
    }
  }

  function closeHeaderSearch(host) {
    const form = host && host.querySelector('.nav-search');
    if (form && typeof form.__closeSearch === 'function') {
      form.__closeSearch();
    }
  }

})();
