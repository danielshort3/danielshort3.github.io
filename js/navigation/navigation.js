/* ===================================================================
   File: navigation.js
   Purpose: Enhances header navigation and handles nav layout
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const NAV_HEIGHT_FALLBACK = 72;
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
      const vid = card.querySelector('video.nav-project-thumb-media');
      if (!vid) return;
      const sources = [...vid.querySelectorAll('source[data-src]')];
      const loadSources = () => {
        if (vid.dataset.loaded === 'true') return;
        sources.forEach((source) => {
          if (!source.src && source.dataset.src) {
            source.src = source.dataset.src;
          }
        });
        vid.dataset.loaded = 'true';
        try { vid.load(); } catch {}
      };
      const playVideo = () => {
        loadSources();
        card.classList.add('is-video-active');
        try { vid.play && vid.play().catch(() => {}); } catch {}
      };
      const pauseVideo = () => {
        try { vid.pause && vid.pause(); } catch {}
        card.classList.remove('is-video-active');
      };
      card._previewVideoBound = true;
      card.addEventListener('pointerenter', playVideo);
      card.addEventListener('focusin', playVideo);
      card.addEventListener('pointerleave', pauseVideo);
      card.addEventListener('focusout', pauseVideo);
    });
  }

  function initNav(){
    const host = $('#combined-header-nav');
    if (!host) return;

    const nav = host.querySelector('.nav');
    if (!nav) return;

    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed', 'yes');
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

    const burger = host.querySelector('#nav-toggle');
    const menu   = host.querySelector('#primary-menu');
    setupDropdown(host.querySelector('.nav-item-portfolio'));
    setupDropdown(host.querySelector('.nav-item-resume'));
    setupDropdown(host.querySelector('.nav-item-contact'));

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

      const closeMenu = () => {
        if (!menu.classList.contains('open')) return;
        menu.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
        syncBodyMenuState(false);
        document.removeEventListener('keydown', trapKeydown);
        if (outsideCloseAttached){
          document.removeEventListener('pointerdown', handleOutsidePointer, true);
          outsideCloseAttached = false;
        }
        if (prevFocus) { prevFocus.focus(); prevFocus = null; }
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
