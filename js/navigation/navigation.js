/* ===================================================================
   File: navigation.js
   Purpose: Injects header navigation and footer components
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const PAGE_TRANSITION_DURATION = 240;
  const PAGE_TRANSITION_WAIT = PAGE_TRANSITION_DURATION + 40;
  const HEAD_SYNC_SELECTORS = [
    'meta[name="description"]',
    'link[rel="canonical"]',
    'meta[property^="og:"]',
    'meta[name^="twitter:"]',
    'script[type="application/ld+json"]'
  ];
  let burgerButton = null;
  let navMenu = null;
  let releaseMenuTrap = null;
  const loadedScripts = new Set();

  const resolveURL = (raw) => {
    if (!raw) return '';
    try { return new URL(raw, `${location.origin}/`).href; }
    catch { return raw; }
  };

  const navKeyFromHref = (raw) => {
    if (!raw) return 'index.html';
    try {
      const url = new URL(raw, `${location.origin}/`);
      const path = url.pathname.replace(/^\/+/, '');
      return path || 'index.html';
    } catch {
      const cleaned = String(raw).replace(/^\/+/, '');
      return cleaned || 'index.html';
    }
  };

  let currentViewKey = navKeyFromHref(location.pathname);

  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const markLoadedScripts = () => {
    document.querySelectorAll('script[src]').forEach(script => {
      const src = script.getAttribute('src');
      if (!src) return;
      loadedScripts.add(resolveURL(src));
    });
  };

  const markSyncableHeadElements = () => {
    HEAD_SYNC_SELECTORS.forEach(selector => {
      document.head.querySelectorAll(selector).forEach(node => {
        node.dataset.softSync = 'true';
      });
    });
  };

  const syncHeadElements = (doc) => {
    if (!doc || !doc.head) return;
    HEAD_SYNC_SELECTORS.forEach(selector => {
      document.head
        .querySelectorAll(`${selector}[data-soft-sync]`)
        .forEach(node => node.remove());
      doc.head.querySelectorAll(selector).forEach(node => {
        const clone = node.cloneNode(true);
        clone.dataset.softSync = 'true';
        document.head.appendChild(clone);
      });
    });
  };

  const extractScriptDescriptors = (doc) => {
    if (!doc) return [];
    return [...doc.querySelectorAll('script[src]')].map(script => ({
      src: script.getAttribute('src'),
      attrs: [...script.attributes].map(attr => ({ name: attr.name, value: attr.value }))
    }));
  };

  const loadScriptsSequential = async (scriptDescriptors) => {
    for (const descriptor of scriptDescriptors) {
      if (!descriptor.src) continue;
      const resolved = resolveURL(descriptor.src);
      if (loadedScripts.has(resolved)) continue;
      await new Promise((resolve, reject) => {
        const el = document.createElement('script');
        descriptor.attrs.forEach(attr => el.setAttribute(attr.name, attr.value));
        el.addEventListener('load', () => { loadedScripts.add(resolved); resolve(); });
        el.addEventListener('error', reject);
        document.head.appendChild(el);
      });
    }
  };

  const setActiveNavLink = (key) => {
    $$('.nav-link').forEach(link => {
      const match = navKeyFromHref(link.getAttribute('href'));
      const active = match === key;
      link.classList.toggle('btn-primary', active);
      link.classList.toggle('btn-secondary', !active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    currentViewKey = key;
  };

  const closeMenu = () => {
    if (typeof releaseMenuTrap === 'function') {
      releaseMenuTrap();
      return;
    }
    if (navMenu) navMenu.classList.remove('open');
    if (burgerButton) burgerButton.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('menu-open');
  };

  const syncBodyState = (sourceBody) => {
    if (!sourceBody) return;
    const hadTransition = document.body.classList.contains('page-transition-enabled');
    document.body.className = sourceBody.className || '';
    if (hadTransition) document.body.classList.add('page-transition-enabled');
    if (sourceBody.dataset && sourceBody.dataset.page) {
      document.body.dataset.page = sourceBody.dataset.page;
    } else {
      delete document.body.dataset.page;
    }
  };

  const focusMain = (mainEl) => {
    if (!mainEl) return;
    const previous = mainEl.getAttribute('tabindex');
    mainEl.setAttribute('tabindex', '-1');
    try {
      mainEl.focus({ preventScroll: true });
    } catch {}
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    if (previous !== null) mainEl.setAttribute('tabindex', previous);
    else mainEl.removeAttribute('tabindex');
  };
  document.addEventListener('DOMContentLoaded', () => {
    injectNav();
    initPageTransitions();
    injectFooter();
    setNavHeight();
    window.addEventListener('load', setNavHeight);
    window.addEventListener('resize', setNavHeight);
    window.addEventListener('orientationchange', setNavHeight);
  });
  function setNavHeight(){
    const nav = document.querySelector('.nav');
    if(!nav) return;
    const h = nav.getBoundingClientRect().height;
    document.documentElement.style.setProperty('--nav-height', `${h}px`);
  }
  function injectNav(){
    const host = $('#combined-header-nav');
    if(!host) return;
    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed','yes');
    host.innerHTML=`
      <nav class="nav ${animate?'animate-entry':''}" aria-label="Primary">
        <div class="wrapper">
          <a href="index.html" class="brand" aria-label="Home">
            <img src="img/ui/logo-64.png" srcset="img/ui/logo-64.png 1x, img/ui/logo-192.png 3x" sizes="64px" alt="DS logo" class="brand-logo" decoding="async" loading="eager" width="64" height="64">
            <span class="brand-name">
              <span class="brand-line name">Daniel Short</span>
              <span class="brand-line divider">│</span>
              <span class="brand-line tagline">Data Science & Analytics</span>
            </span>
          </a>
          <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false" aria-controls="primary-menu">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </button>
          <div id="primary-menu" class="nav-row" data-collapsible role="navigation">
            <a href="index.html" class="btn-secondary nav-link">Home</a>
            <a href="portfolio.html" class="btn-secondary nav-link">Portfolio</a>
            <a href="contributions.html" class="btn-secondary nav-link">Contributions</a>
            <a href="contact.html" class="btn-secondary nav-link">Contact</a>
            <a href="resume.html" class="btn-secondary nav-link">Resume</a>
          </div>
        </div>
      </nav>`;
    setActiveNavLink(navKeyFromHref(location.pathname));
    burgerButton = host.querySelector('#nav-toggle');
    navMenu = host.querySelector('#primary-menu');
    const burger = burgerButton;
    const menu   = navMenu;

    // Simple focus trap within the mobile drawer
    let prevFocus = null;
    const trapKeydown = (e) => {
      if (e.key === 'Escape') {
        // close drawer
        burger.click();
        return;
      }
      if (e.key !== 'Tab') return;
      const focusables = menu.querySelectorAll('a,button,[tabindex]:not([tabindex="-1"])');
      if (!focusables.length) return;
      const first = focusables[0];
      const last  = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };

    if(burger && menu){
      burger.addEventListener('click', () => {
        const headerBar = burger.closest('.nav') || host;
        const headerBottom = headerBar.getBoundingClientRect().bottom;
        menu.style.top = `${headerBottom}px`;
        const open = menu.classList.toggle('open');
        burger.setAttribute('aria-expanded', open);
        document.body.classList.toggle('menu-open', open);

        if (open) {
          prevFocus = document.activeElement;
          document.addEventListener('keydown', trapKeydown);
          // Focus first nav link for keyboard users
          const firstLink = menu.querySelector('.nav-link');
          firstLink && firstLink.focus();
        } else {
          document.removeEventListener('keydown', trapKeydown);
          if (prevFocus) { prevFocus.focus(); prevFocus = null; }
        }
      });
      releaseMenuTrap = () => {
        menu.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
        document.body.classList.remove('menu-open');
        document.removeEventListener('keydown', trapKeydown);
        if (prevFocus) { prevFocus.focus(); prevFocus = null; }
        releaseMenuTrap = null;
      };
    }
  }
  function initPageTransitions(){
    const body = document.body;
    if (!body) return;

    body.classList.add('page-transition-enabled');

    const prefersReduced = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const supportsSoft = 'fetch' in window && typeof DOMParser !== 'undefined' && window.history && typeof window.history.pushState === 'function';

    markSyncableHeadElements();
    markLoadedScripts();

    const playEntry = () => {
      if (prefersReduced) return;
      body.classList.add('page-transition-enter');
      requestAnimationFrame(() => {
        body.classList.remove('page-transition-enter');
      });
    };

    playEntry();

    const navHost = $('#combined-header-nav');
    const links = navHost ? navHost.querySelectorAll('a.nav-link, a.brand') : [];
    let isNavigating = false;

    window.addEventListener('pageshow', (event) => {
      isNavigating = false;
      body.classList.remove('page-transition-exit');
      body.removeAttribute('aria-busy');
      if (!prefersReduced && event.persisted) playEntry();
    });

    const navigate = async (href, { replace = false } = {}) => {
      if (isNavigating) return;
      const url = new URL(href, location.href);
      const targetKey = navKeyFromHref(url.pathname);
      if (currentViewKey === targetKey && url.hash === location.hash && url.search === location.search) {
        setActiveNavLink(targetKey);
        return;
      }
      isNavigating = true;
      closeMenu();

      if (!prefersReduced) body.classList.add('page-transition-exit');
      body.setAttribute('aria-busy', 'true');

      if (!supportsSoft) {
        window.setTimeout(() => { window.location.href = url.href; }, prefersReduced ? 0 : PAGE_TRANSITION_WAIT);
        return;
      }

      const success = await performSoftNavigation(url, { replace });
      if (!success) {
        window.location.href = url.href;
      }
    };

    const performSoftNavigation = async (url, { replace = false } = {}) => {
      try {
        const response = await fetch(url.href, {
          credentials: 'same-origin',
          headers: { 'X-Requested-With': 'fetch' }
        });
        if (!response.ok) return false;
        const html = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const nextMain = doc.querySelector('#main');
        const currentMain = $('#main');
        if (!nextMain || !currentMain) return false;

        const scriptsToLoad = extractScriptDescriptors(doc);

        if (!prefersReduced) await wait(PAGE_TRANSITION_WAIT);

        const importedMain = document.importNode(nextMain, true);
        currentMain.replaceWith(importedMain);

        const nextFooter = doc.querySelector('footer');
        const currentFooter = document.querySelector('footer');
        if (nextFooter && currentFooter) {
          currentFooter.replaceWith(document.importNode(nextFooter, true));
          injectFooter();
        }

        syncBodyState(doc.body);
        syncHeadElements(doc);
        document.title = doc.title || document.title;

        await loadScriptsSequential(scriptsToLoad);

        const newUrl = url.pathname + url.search + url.hash;
        if (replace && history.replaceState) history.replaceState({}, doc.title, newUrl);
        else if (!replace && history.pushState) history.pushState({}, doc.title, newUrl);

        setActiveNavLink(navKeyFromHref(url.pathname));
        focusMain(importedMain);
        if (typeof window.runPageEntrypoints === 'function') window.runPageEntrypoints();

        body.classList.remove('page-transition-exit');
        if (!prefersReduced) playEntry();
        body.removeAttribute('aria-busy');
        isNavigating = false;
        setTimeout(() => { setNavHeight(); }, 0);
        return true;
      } catch (error) {
        console.error('Soft navigation failed', error);
        body.classList.remove('page-transition-exit');
        body.removeAttribute('aria-busy');
        isNavigating = false;
        return false;
      }
    };

    if (links.length) {
      links.forEach(link => {
        if (link.dataset.transitionBound === 'yes') return;
        link.dataset.transitionBound = 'yes';
        link.addEventListener('click', (event) => {
          if (isNavigating || event.defaultPrevented) return;
          if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return;
          if (link.target && link.target !== '_self') return;
          const href = link.getAttribute('href');
          if (!href || href.startsWith('#')) return;
          event.preventDefault();
          navigate(link.href);
        });
      });
    }

    if (supportsSoft) {
      window.addEventListener('popstate', () => {
        navigate(location.href, { replace: true });
      });
    }
  }
  function injectFooter(){
    const f = $('footer');
    if(!f) return;
    f.classList.add('footer');
    const year = new Date().getFullYear();
    f.innerHTML=`
      <div class="social">
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="5" width="18" height="14" rx="2"></rect>
            <path d="M3 7l9 6 9-6"></path>
          </svg>
        </a>
        <a class="btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
          <svg class="brand-fill" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="4" cy="4" r="2"></circle>
            <rect x="2" y="9" width="4" height="12" rx="1"></rect>
            <path d="M10 9h3.8v2.1h.1C14.8 9.7 16.1 9 17.9 9c3 0 5.1 1.9 5.1 5.9V21h-4v-5.9c0-1.7-.7-2.9-2.6-2.9s-2.7 1.4-2.7 3V21H10z"></path>
          </svg>
        </a>
        <a class="btn-icon" href="https://github.com/danielshort3" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
          <span class="icon icon-github" aria-hidden="true"></span>
        </a>
      </div>
      <p>© ${year} Daniel Short. All rights reserved. <a href="privacy.html">Privacy Policy</a></p>`;
  
    // dev-only reset button removed per request
  }
})();
