/* ===================================================================
   File: navigation.js
   Purpose: Enhances header navigation and handles nav layout
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const NAVIGATION_EVENT = 'site:navigation-start';
  const NAV_HEIGHT_FALLBACK = 72;
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
    const nav = document.querySelector('.nav');
    if (!nav) return NAV_HEIGHT_FALLBACK;
    const rect = nav.getBoundingClientRect();
    const height = rect.height || nav.offsetHeight || NAV_HEIGHT_FALLBACK;
    return Math.max(height, NAV_HEIGHT_FALLBACK);
  };

  const setCssNavHeight = (value) => {
    document.documentElement.style.setProperty('--nav-height', `${value}px`);
    window.__navHeight = value;
  };

  const clampDropdownToViewport = (dropdown) => {
    if (!dropdown || typeof dropdown.getBoundingClientRect !== 'function') return;
    dropdown.style.removeProperty('--dropdown-shift');
    const rect = dropdown.getBoundingClientRect();
    const viewportWidth = document.documentElement?.clientWidth || window.innerWidth || 0;
    if (!viewportWidth || !rect?.width) return;
    const padding = 12;
    const overflowLeft = Math.max(0, padding - rect.left);
    const overflowRight = Math.max(0, rect.right - (viewportWidth - padding));
    let shift = 0;
    if (overflowLeft > 0) {
      shift = overflowLeft;
    } else if (overflowRight > 0) {
      shift = -overflowRight;
    }
    if (shift !== 0) {
      dropdown.style.setProperty('--dropdown-shift', `${shift}px`);
    } else {
      dropdown.style.removeProperty('--dropdown-shift');
    }
  };

  const clampDropdownsToViewport = () => {
    document.querySelectorAll('.nav-dropdown').forEach(clampDropdownToViewport);
  };

  const updateNavDropdownOffset = () => {
    const nav = document.querySelector('.nav');
    if (!nav) return;
    const row = nav.querySelector('.nav-row');
    if (!row) return;
    const navRect = nav.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();
    if (!navRect?.bottom || !rowRect?.bottom) return;
    const gap = Math.max(0, navRect.bottom - rowRect.bottom);
    nav.style.setProperty('--nav-bottom-gap', `${gap}px`);
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
    updateNavDropdownOffset();
    clampDropdownsToViewport();
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
    const nav = document.querySelector('.nav');
    if (!nav) return;

    if (navResizeObserver) {
      try { navResizeObserver.disconnect(); } catch {}
      navResizeObserver = null;
    }

    if (typeof ResizeObserver === 'function') {
      navResizeObserver = new ResizeObserver(() => {
        scheduleNavHeightUpdate();
      });
      navResizeObserver.observe(nav);
    }

    if (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === 'function') {
      document.fonts.ready
        .then(() => {
          scheduleNavHeightUpdate();
        })
        .catch(() => {});
    }
  }

  function setupNavPreviewVideos(root){
    if (!root) return;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const finePointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
    if (reduce || !finePointer) return;
    root.querySelectorAll('.nav-project-card').forEach((card) => {
      if (card._previewVideoBound) return;
      const thumb = card.querySelector('.nav-project-thumb');
      if (!thumb) return;

      const previewPoster = String(thumb.dataset.previewPoster || '').trim();
      const previewWebm = String(thumb.dataset.previewWebm || '').trim();
      const previewMp4 = String(thumb.dataset.previewMp4 || '').trim();
      const previewSources = [];
      if (previewWebm) previewSources.push({ src: previewWebm, type: 'video/webm' });
      if (previewMp4) previewSources.push({ src: previewMp4, type: 'video/mp4' });

      let vid = thumb.querySelector('video.nav-project-thumb-media') || null;
      if (!vid && !previewSources.length) return;

      const ensureVideoElement = () => {
        if (!vid || !vid.isConnected) {
          vid = document.createElement('video');
          vid.className = 'nav-project-thumb-media';
          vid.muted = true;
          vid.playsInline = true;
          vid.loop = true;
          vid.preload = 'none';
          vid.setAttribute('muted', '');
          vid.setAttribute('playsinline', '');
          if (previewPoster) {
            vid.setAttribute('poster', previewPoster);
          }
          previewSources.forEach((entry) => {
            const source = document.createElement('source');
            source.dataset.src = entry.src;
            source.type = entry.type;
            vid.appendChild(source);
          });
          thumb.appendChild(vid);
          return vid;
        }

        if (previewPoster && !vid.getAttribute('poster')) {
          vid.setAttribute('poster', previewPoster);
        }
        if (!vid.querySelector('source[data-src]') && previewSources.length) {
          previewSources.forEach((entry) => {
            const source = document.createElement('source');
            source.dataset.src = entry.src;
            source.type = entry.type;
            vid.appendChild(source);
          });
        }
        return vid;
      };

      const loadSources = () => {
        const media = ensureVideoElement();
        if (media.dataset.loaded === 'true') return media;
        const sources = [...media.querySelectorAll('source[data-src]')];
        sources.forEach((source) => {
          if (!source.src && source.dataset.src) {
            source.src = source.dataset.src;
          }
        });
        media.dataset.loaded = 'true';
        try { media.load(); } catch {}
        return media;
      };
      const playVideo = () => {
        const media = loadSources();
        card.classList.add('is-video-active');
        try { media.play && media.play().catch(() => {}); } catch {}
      };
      const pauseVideo = () => {
        try { vid?.pause && vid.pause(); } catch {}
        card.classList.remove('is-video-active');
      };
      card._previewVideoBound = true;
      card.addEventListener('pointerenter', playVideo);
      card.addEventListener('focusin', playVideo);
      card.addEventListener('pointerleave', pauseVideo);
      card.addEventListener('focusout', pauseVideo);
    });
  }

  const MOBILE_DOCK_ICONS = {
    projects: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3.5 7.5a2 2 0 0 1 2-2h4.2l1.8 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"></path>
        <path d="M7.5 13.5h9"></path>
      </svg>
    `,
    tools: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7"></path>
        <rect x="3" y="7" width="18" height="13" rx="2.25"></rect>
        <path d="M3 12.5h18"></path>
        <path d="M10 12.5v2h4v-2"></path>
      </svg>
    `,
    games: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7.2 9h9.6a4.7 4.7 0 0 1 4.5 5.9l-.4 1.7a2.5 2.5 0 0 1-4.2 1.2l-1.5-1.5H8.8l-1.5 1.5a2.5 2.5 0 0 1-4.2-1.2l-.4-1.7A4.7 4.7 0 0 1 7.2 9z"></path>
        <path d="M8 12v3M6.5 13.5h3M15.7 13h.1M18.2 14.6h.1"></path>
      </svg>
    `,
    resume: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6 3.5h8l4 4v13H6z"></path>
        <path d="M14 3.5v4h4"></path>
        <path d="M8.6 12h6.8"></path>
        <path d="M8.6 15.5h4.9"></path>
      </svg>
    `,
    contact: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4.5 6.5h15a1.7 1.7 0 0 1 1.7 1.7v8.6a1.7 1.7 0 0 1-1.7 1.7h-15a1.7 1.7 0 0 1-1.7-1.7V8.2a1.7 1.7 0 0 1 1.7-1.7z"></path>
        <path d="m4.2 8.4 7.8 5.2 7.8-5.2"></path>
      </svg>
    `
  };

  const MOBILE_MASTHEAD_SEARCH_ICON = `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5"></circle>
      <path d="m16.2 16.2 4.3 4.3"></path>
    </svg>
  `;

  const MOBILE_CONTACT_LINKS = [
    {
      label: 'Message',
      href: '/contact#contact-modal',
      icon: `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 4.5h16a2 2 0 0 1 2 2v8.8a2 2 0 0 1-2 2h-5.1L9 22v-4.7H4a2 2 0 0 1-2-2V6.5a2 2 0 0 1 2-2z"></path>
          <path d="M7 9.2h10"></path>
          <path d="M7 13h6.4"></path>
        </svg>
      `,
      contact: true
    },
    {
      label: 'Email',
      href: 'mailto:daniel@danielshort.me',
      icon: `
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3.5" y="6" width="17" height="12" rx="2"></rect>
          <path d="m4 8 8 5.3L20 8"></path>
        </svg>
      `
    },
    {
      label: 'LinkedIn',
      href: 'https://www.linkedin.com/in/danielshort3/',
      target: '_blank',
      rel: 'noopener noreferrer',
      icon: `
        <svg class="mobile-site-dock__brand-icon" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="5" cy="5" r="2"></circle>
          <path d="M3.5 9.5h3v11h-3z"></path>
          <path d="M10 9.5h2.9v1.8h.1c.6-1.1 1.8-2.1 3.8-2.1 2.9 0 4.7 1.9 4.7 5.7v5.6h-3.1v-5c0-1.7-.6-2.8-2.2-2.8-1.8 0-3.1 1.1-3.1 3.2v4.6H10z"></path>
        </svg>
      `
    },
    {
      label: 'GitHub',
      href: 'https://github.com/danielshort3',
      target: '_blank',
      rel: 'noopener noreferrer',
      icon: `
        <svg class="mobile-site-dock__brand-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2.8a9.2 9.2 0 0 0-2.9 17.9c.46.08.62-.2.62-.44v-1.7c-2.52.54-3.05-1.08-3.05-1.08-.42-1.06-1.02-1.34-1.02-1.34-.84-.58.06-.56.06-.56.93.06 1.42.96 1.42.96.82 1.4 2.16 1 2.7.76.08-.6.32-1 .58-1.24-2.02-.24-4.14-1.02-4.14-4.48 0-1 .36-1.8.94-2.44-.1-.24-.4-1.22.1-2.54 0 0 .78-.24 2.54.94a8.8 8.8 0 0 1 4.62 0c1.76-1.18 2.54-.94 2.54-.94.5 1.32.2 2.3.1 2.54.58.64.94 1.44.94 2.44 0 3.48-2.12 4.24-4.14 4.48.34.3.64.88.64 1.76v2.44c0 .24.16.52.64.44A9.2 9.2 0 0 0 12 2.8z"></path>
        </svg>
      `
    }
  ];

  function setupMobileSiteMasthead(config) {
    if (!document.body || document.querySelector('[data-mobile-site-masthead]')) return;

    const { entryHome, currentPathVariants } = config;
    const isHome = (currentPathVariants || []).includes('/') || document.body?.dataset?.page === 'home';
    const masthead = document.createElement('header');
    masthead.className = `mobile-site-masthead${isHome ? ' mobile-site-masthead--home' : ''}`;
    masthead.dataset.mobileSiteMasthead = '';
    masthead.innerHTML = `
      <div class="mobile-site-masthead__inner">
        <a class="mobile-site-masthead__brand" href="${escapeHtml(entryHome || '/')}" aria-label="Daniel Short home">
          <img src="img/brand/00-ds-logo-master-full-color.svg" srcset="img/brand/00-ds-logo-master-full-color.svg 1x" sizes="40px" alt="Daniel Short DS logo" class="mobile-site-masthead__logo" decoding="async" loading="eager" width="381" height="392">
          <span class="mobile-site-masthead__name">
            <span class="mobile-site-masthead__title">Daniel Short</span>
          </span>
        </a>
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
    const setSearchExpanded = (expanded, options = {}) => {
      if (!searchForm || !searchInput || !searchButton) return;
      const nextExpanded = Boolean(expanded);
      searchForm.classList.toggle('is-expanded', nextExpanded);
      searchForm.dataset.mobileMastheadSearch = nextExpanded ? 'expanded' : 'collapsed';
      searchButton.setAttribute('aria-expanded', String(nextExpanded));
      searchButton.setAttribute('aria-label', nextExpanded ? 'Search site' : 'Open search');
      searchInput.tabIndex = nextExpanded ? 0 : -1;
      searchInput.setAttribute('aria-hidden', nextExpanded ? 'false' : 'true');
      if (options.focusInput && nextExpanded) {
        requestAnimationFrame(() => searchInput.focus());
      }
    };

    if (searchForm && searchInput && searchButton) {
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

    document.body.appendChild(masthead);
    document.body.classList.add('has-mobile-site-masthead');
  }

  function setupMobileSiteDock(config){
    if (!document.body || document.querySelector('[data-mobile-site-dock]')) return;

    const {
      activeAudience,
      currentPathVariants,
      entryHome,
      normalizePath
    } = config;
    const homeHref = entryHome || '/';
    const portfolioHref = activeAudience?.portfolioPath || '/portfolio';
    const resumeHref = activeAudience?.resumePath || '/resume';
    const resumePreviewHref = activeAudience?.resumePreviewPath || '';
    const isProfessionalAudience = activeAudience?.key && activeAudience.key !== 'personal';
    const pathSet = new Set(currentPathVariants || []);
    const isCurrentPath = (...paths) => paths.filter(Boolean).some((path) => pathSet.has(normalizePath(path)));
    const isCurrentSection = (path) => {
      const sectionPath = normalizePath(path);
      return [...pathSet].some((entry) => entry === sectionPath || entry.startsWith(`${sectionPath}/`));
    };
    const isHome = isCurrentPath('/', '/index.html', activeAudience?.homePath);
    const homeItem = {
      id: 'home',
      label: 'Home',
      href: homeHref,
      home: true,
      accent: '#155dfc',
      active: isHome
    };
    const personalItems = [
      {
        id: 'projects',
        label: 'Projects',
        href: portfolioHref,
        icon: MOBILE_DOCK_ICONS.projects,
        accent: '#155dfc',
        active: isCurrentSection('/portfolio')
      },
      {
        id: 'tools',
        label: 'Tools',
        href: '/tools',
        icon: MOBILE_DOCK_ICONS.tools,
        accent: '#0891b2',
        active: isCurrentSection('/tools')
      },
      homeItem,
      {
        id: 'games',
        label: 'Games',
        href: '/games',
        icon: MOBILE_DOCK_ICONS.games,
        accent: '#f97316',
        active: isCurrentSection('/games')
      },
      {
        id: 'contact',
        label: 'Contact',
        href: '#mobile-contact-options',
        icon: MOBILE_DOCK_ICONS.contact,
        accent: '#64748b',
        active: isCurrentSection('/contact'),
        contactOptions: true
      }
    ];
    const professionalItems = [
      {
        id: 'projects',
        label: 'Portfolio',
        href: portfolioHref,
        icon: MOBILE_DOCK_ICONS.projects,
        accent: '#155dfc',
        active: isCurrentSection('/portfolio')
      },
      {
        id: 'resume',
        label: 'Resume',
        href: resumeHref,
        icon: MOBILE_DOCK_ICONS.resume,
        accent: '#334155',
        active: isCurrentPath(resumeHref, resumePreviewHref)
      },
      homeItem,
      {
        id: 'contact',
        label: 'Contact',
        href: '#mobile-contact-options',
        icon: MOBILE_DOCK_ICONS.contact,
        accent: '#64748b',
        active: isCurrentSection('/contact'),
        contactOptions: true
      }
    ];
    const items = isProfessionalAudience ? professionalItems : personalItems;
    const contactMenuId = 'mobile-site-contact-menu';
    const contactMenuHtml = `
      <div class="mobile-site-dock__contact-menu" id="${contactMenuId}" role="menu" aria-label="Contact options" aria-hidden="true" data-mobile-contact-menu>
        <div class="mobile-site-dock__contact-actions">
          ${MOBILE_CONTACT_LINKS.map((link) => {
            const target = link.target ? ` target="${escapeHtml(link.target)}"` : '';
            const rel = link.rel ? ` rel="${escapeHtml(link.rel)}"` : '';
            const contactAttr = link.contact ? ' data-contact-modal-link="true"' : '';
            const directClass = link.contact ? ' mobile-site-dock__contact-link--direct' : '';
            return `
              <a class="mobile-site-dock__contact-link${directClass}" href="${escapeHtml(link.href)}"${target}${rel}${contactAttr} role="menuitem" data-mobile-contact-action tabindex="-1">
                <span class="mobile-site-dock__contact-icon" aria-hidden="true">${link.icon}</span>
                <span class="mobile-site-dock__contact-text">${escapeHtml(link.label)}</span>
              </a>
            `;
          }).join('')}
        </div>
      </div>
    `;

    const html = items.map((item) => {
      const activeClass = item.active ? ' is-active' : '';
      const ariaCurrent = item.active ? ' aria-current="page"' : '';
      const contactAttr = item.contact ? ' data-contact-modal-link="true"' : '';
      const dataAttr = ` data-mobile-dock-item="${item.id}"`;
      const accentStyle = ` style="--dock-accent: ${item.accent}"`;
      if (item.home) {
        return `
          <a class="mobile-site-dock__home${activeClass}" href="${item.href}"${ariaCurrent}${dataAttr}${accentStyle} aria-label="Daniel Short home">
            <img src="img/brand/00-ds-logo-master-full-color.svg" alt="" width="76" height="76" decoding="async">
            <span class="visually-hidden">${item.label}</span>
          </a>
        `;
      }
      if (item.contactOptions) {
        return `
          <div class="mobile-site-dock__contact" data-mobile-dock-item="${item.id}"${accentStyle}>
            ${contactMenuHtml}
            <button class="mobile-site-dock__item mobile-site-dock__contact-toggle${activeClass}" type="button"${ariaCurrent} aria-expanded="false" aria-haspopup="menu" aria-controls="${contactMenuId}" data-mobile-contact-options>
              <span class="mobile-site-dock__icon" aria-hidden="true">${item.icon}</span>
              <span class="mobile-site-dock__label">${item.label}</span>
            </button>
          </div>
        `;
      }
      return `
        <a class="mobile-site-dock__item${activeClass}" href="${item.href}"${ariaCurrent}${contactAttr}${dataAttr}${accentStyle}>
          <span class="mobile-site-dock__icon" aria-hidden="true">${item.icon}</span>
          <span class="mobile-site-dock__label">${item.label}</span>
        </a>
      `;
    }).join('');

    const dock = document.createElement('nav');
    dock.className = 'mobile-site-dock';
    dock.dataset.mobileSiteDock = '';
    dock.dataset.mobileDockLayout = isProfessionalAudience ? 'professional' : 'personal';
    dock.setAttribute('aria-label', 'Mobile primary');
    dock.innerHTML = html;
    const contactToggle = dock.querySelector('[data-mobile-contact-options]');
    const contactMenu = dock.querySelector('[data-mobile-contact-menu]');
    const contactActions = [...dock.querySelectorAll('[data-mobile-contact-action]')];
    const setContactExpanded = (expanded) => {
      if (!contactToggle || !contactMenu) return;
      const nextExpanded = Boolean(expanded);
      dock.classList.toggle('is-contact-open', nextExpanded);
      contactToggle.setAttribute('aria-expanded', String(nextExpanded));
      contactToggle.setAttribute('aria-label', nextExpanded ? 'Close contact options' : 'Open contact options');
      contactMenu.setAttribute('aria-hidden', String(!nextExpanded));
      contactActions.forEach((action) => {
        action.tabIndex = nextExpanded ? 0 : -1;
      });
      if (!nextExpanded && contactMenu.contains(document.activeElement)) {
        contactToggle.focus({ preventScroll: true });
      }
    };
    setContactExpanded(false);

    dock.addEventListener('click', (event) => {
      const contactOptions = event.target.closest('[data-mobile-contact-options]');
      if (contactOptions && dock.contains(contactOptions)) {
        event.preventDefault();
        setContactExpanded(!dock.classList.contains('is-contact-open'));
        return;
      }
      const contactAction = event.target.closest('[data-mobile-contact-action]');
      if (contactAction && dock.contains(contactAction)) {
        setContactExpanded(false);
        document.dispatchEvent(new CustomEvent(NAVIGATION_EVENT));
        return;
      }
      const link = event.target.closest('a');
      if (!link || !dock.contains(link)) return;
      setContactExpanded(false);
      document.dispatchEvent(new CustomEvent(NAVIGATION_EVENT));
    });
    document.addEventListener('click', (event) => {
      if (!dock.classList.contains('is-contact-open')) return;
      if (dock.contains(event.target)) return;
      setContactExpanded(false);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !dock.classList.contains('is-contact-open')) return;
      event.preventDefault();
      setContactExpanded(false);
    });
    document.body.appendChild(dock);
    document.body.classList.add('has-mobile-site-dock');
  }

  function initNav(){
    const host = $('#combined-header-nav');
    if (!host) return;

    const nav = host.querySelector('.nav');
    if (!nav) return;

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

    const animate = !readSession('navEntryPlayed');
    writeSession('navEntryPlayed', 'yes');
    if (animate) nav.classList.add('animate-entry');

    setupNavPreviewVideos(host);

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
    const explicitAudienceCandidates = [bodyAudience, queryAudience, pathAudience].filter(Boolean);
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

    $$('[data-entry-home-link="true"]', host).forEach((link) => {
      link.setAttribute('href', entryHome);
    });
    $$('[data-audience-home-link="true"]', host).forEach((link) => {
      link.setAttribute('href', activeAudience.homePath || '/');
    });
    $$('[data-portfolio-home-link="true"]', host).forEach((link) => {
      link.setAttribute('href', activeAudience.portfolioPath || '/portfolio');
    });
    $$('[data-portfolio-default-link="true"]', host).forEach((link) => {
      link.setAttribute('href', activeAudience.portfolioAllPath || '/portfolio');
    });
    $$('[data-audience-link]', host).forEach((link) => {
      const audience = getAudience(link.dataset.audienceLink);
      link.setAttribute('href', audience.homePath || '/');
      const isActive = normalizeAudience(audience.key) === activeAudienceKey;
      link.classList.toggle('is-current', isActive);
      if (isActive) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });

    $$('.nav-link', host).forEach((link) => {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) return;
      let targetPath = normalizePath(href);
      const targetNoHtml = targetPath.endsWith('.html')
        ? normalizePath(targetPath.replace(/\.html$/i, '') || '/')
        : targetPath;
      const matchesExact = [currentPath, altCurrentPath].some(p => p === targetPath || p === targetNoHtml);
      const matchesSection = (() => {
        if (targetNoHtml === '/portfolio') {
          return [currentPath, altCurrentPath].some(p => p === '/portfolio' || p === '/portfolio.html' || p.startsWith('/portfolio/'));
        }
        if (targetNoHtml === '/tools') {
          return [currentPath, altCurrentPath].some(p => p === '/tools' || p === '/tools.html' || p.startsWith('/tools/'));
        }
        if (targetNoHtml === '/games') {
          return [currentPath, altCurrentPath].some(p => p === '/games' || p === '/games.html' || p.startsWith('/games/'));
        }
        return false;
      })();
      const matches = matchesExact || matchesSection;
      if (matches) {
        link.classList.add('is-current');
        link.setAttribute('aria-current','page');
      } else {
        link.classList.remove('is-current');
        link.removeAttribute('aria-current');
      }
    });

    setupMobileSiteDock({
      activeAudience,
      currentPathVariants,
      entryHome,
      normalizePath
    });
    setupMobileSiteMasthead({
      activeAudience,
      currentPathVariants,
      entryHome
    });

    const burger = host.querySelector('#nav-toggle');
    const menu   = host.querySelector('#primary-menu');
    let closeMenu = () => {};
    host.querySelectorAll('.nav-item').forEach(setupDropdown);
    setupHeaderSearch(host);

    const hoverMatcher = window.matchMedia('(hover: hover) and (pointer: fine)');
    if (hoverMatcher.matches) {
      host.querySelectorAll('.nav-item').forEach((item) => {
        item.addEventListener('pointerenter', () => closeActiveDropdowns(item, { forceBlur: true }));
      });
    }

    if(burger && menu){
      let prevFocus = null;
      let outsideCloseAttached = false;
      const syncBodyMenuState = (isOpen) => {
        document.body.classList.toggle('menu-open', Boolean(isOpen));
      };
      const trapKeydown = (e) => {
        if (e.key === 'Escape') {
          closeMenu();
          return;
        }
        if (e.key !== 'Tab') return;
        const focusables = menu.querySelectorAll('a,button,[tabindex]:not([tabindex=\"-1\"])');
        if (!focusables.length) return;
        const first = focusables[0];
        const last  = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      };

      closeMenu = ({ restoreFocus = true } = {}) => {
        if (!menu.classList.contains('open')) return;
        menu.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
        syncBodyMenuState(false);
        document.removeEventListener('keydown', trapKeydown);
        if (outsideCloseAttached){
          document.removeEventListener('pointerdown', handleOutsidePointer, true);
          outsideCloseAttached = false;
        }
        if (restoreFocus && prevFocus) {
          prevFocus.focus();
        }
        prevFocus = null;
      };

      const handleOutsidePointer = (event) => {
        if (!menu.classList.contains('open')) return;
        const target = event.target;
        if (menu.contains(target) || burger.contains(target)) return;
        closeMenu();
      };

      const openMenu = () => {
        if (menu.classList.contains('open')) return;
        const headerBar = burger.closest('.nav') || host;
        const headerBottom = headerBar.getBoundingClientRect().bottom;
        menu.style.top = `${headerBottom}px`;
        menu.classList.add('open');
        burger.setAttribute('aria-expanded', 'true');
        syncBodyMenuState(true);
        prevFocus = document.activeElement;
        document.addEventListener('keydown', trapKeydown);
        if (!outsideCloseAttached){
          document.addEventListener('pointerdown', handleOutsidePointer, true);
          outsideCloseAttached = true;
        }
        // Focus first nav link for keyboard users
        const firstLink = menu.querySelector('.nav-link');
        firstLink && firstLink.focus();
      };

      burger.addEventListener('click', () => {
        if (menu.classList.contains('open')) {
          closeMenu();
        } else {
          openMenu();
        }
      });
    }

    document.addEventListener(NAVIGATION_EVENT, () => {
      closeActiveDropdowns(null, { forceBlur: true });
      closeHeaderSearch(host);
      closeMenu({ restoreFocus: false });
    });
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
        requestAnimationFrame(() => input.focus());
      } else if (restoreButtonFocus && enhanced && document.activeElement === input) {
        button.focus();
      }
    };

    form.__closeSearch = () => setExpanded(false);
    setExpanded(false);

    form.addEventListener('submit', (event) => {
      if (!desktopMatcher.matches) return;
      if (!form.classList.contains('is-expanded')) {
        event.preventDefault();
        closeActiveDropdowns(null, { forceBlur: true });
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

  const closeActiveDropdowns = (excludeItem, options = {}) => {
    const { forceBlur = false } = options;
    const activeEl = document.activeElement;
    document.querySelectorAll('.nav-item.dropdown-open').forEach((openItem) => {
      if (openItem === excludeItem) return;
      if (typeof openItem.__closeDropdown === 'function') {
        openItem.__closeDropdown();
      } else {
        openItem.classList.remove('dropdown-open');
      }
      if (forceBlur && activeEl && openItem.contains(activeEl) && typeof activeEl.blur === 'function') {
        activeEl.blur();
      }
    });
  };
  function setupDropdown(item){
    if(!item) return;
    const dropdown = item.querySelector('.nav-dropdown');
    const trigger = item.querySelector('.nav-link-has-menu');
    if(!dropdown || !trigger) return;
    trigger.setAttribute('aria-expanded', 'false');
    let closeTimer = null;
    const close = () => {
      clearTimeout(closeTimer);
      item.classList.remove('dropdown-open');
      trigger.setAttribute('aria-expanded', 'false');
    };
    item.__closeDropdown = close;
    const open = () => {
      clearTimeout(closeTimer);
      closeActiveDropdowns(item);
      item.classList.add('dropdown-open');
      trigger.setAttribute('aria-expanded', 'true');
    };
    const scheduleClose = () => {
      clearTimeout(closeTimer);
      closeTimer = setTimeout(() => {
        close();
      }, 320);
    };
    item.addEventListener('focusin', open);
    item.addEventListener('focusout', (event) => {
      const next = event.relatedTarget;
      if(!next || !item.contains(next)){
        scheduleClose();
      }
    });
    const prefersHover = window.matchMedia('(hover: hover) and (pointer: fine)');
    if(prefersHover.matches){
      item.addEventListener('mouseenter', open);
      item.addEventListener('mouseleave', scheduleClose);
      dropdown.addEventListener('mouseenter', open);
      dropdown.addEventListener('mouseleave', scheduleClose);
    }
    const onMediaChange = (event) => {
      if(!event.matches){
        item.classList.remove('dropdown-open');
      }
    };
    if(typeof prefersHover.addEventListener === 'function'){
      prefersHover.addEventListener('change', onMediaChange);
    } else if(typeof prefersHover.addListener === 'function'){
      prefersHover.addListener(onMediaChange);
    }
    updateNavDropdownOffset();
  }
})();
