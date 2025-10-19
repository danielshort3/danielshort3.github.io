/* ===================================================================
   File: navigation.js
   Purpose: Injects header navigation and footer components
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const THEME_KEY = 'ds-theme';
  const storage = (() => {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        return window.localStorage;
      }
    } catch {}
    return null;
  })();

  const getPreferredTheme = () => {
    const stored = storage && storage.getItem(THEME_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }
    return 'dark';
  };

  const applyTheme = (theme) => {
    if (typeof document === 'undefined') return;
    const root = document.documentElement || document.body;
    if (!root) return;
    const next = theme === 'light' ? 'light' : 'dark';
    if (typeof root.setAttribute === 'function') {
      root.setAttribute('data-theme', next);
    }
    if (root.classList && typeof root.classList.toggle === 'function') {
      root.classList.toggle('is-light-theme', next === 'light');
    }
    storage && storage.setItem(THEME_KEY, next);
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(next === 'light'));
      toggle.querySelector('.theme-label').textContent = next === 'light' ? 'Dark mode' : 'Light mode';
    }
  };

  applyTheme(getPreferredTheme());

  if (typeof window !== 'undefined' && window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (evt) => {
      const stored = storage && storage.getItem(THEME_KEY);
      if (!stored) applyTheme(evt.matches ? 'light' : 'dark');
    });
  }
  document.addEventListener('DOMContentLoaded', () => {
    injectNav();
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
      <nav class="nav ${animate?'animate-entry':''}" aria-label="Primary" role="navigation">
        <div class="wrapper">
          <a href="index.html" class="brand" aria-label="Daniel Short home">
            <picture>
              <source srcset="img/ui/logo-64.webp 1x, img/ui/logo-192.webp 3x" type="image/webp" sizes="64px">
              <img src="img/ui/logo-64.png" srcset="img/ui/logo-64.png 1x, img/ui/logo-192.png 3x" sizes="64px" alt="Daniel Short monogram" class="brand-logo" decoding="async" loading="eager" width="64" height="64">
            </picture>
            <span class="brand-name">
              <span class="brand-line name">Daniel Short</span>
              <span class="brand-line divider">│</span>
              <span class="brand-line tagline">Data Science & Analytics</span>
            </span>
          </a>
          <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false" aria-controls="primary-menu">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </button>
          <div id="primary-menu" class="nav-row" data-collapsible role="navigation" aria-label="Primary links">
            <a href="index.html" class="btn-secondary nav-link">Home</a>
            <a href="portfolio.html" class="btn-secondary nav-link">Portfolio</a>
            <a href="contributions.html" class="btn-secondary nav-link">Contributions</a>
            <a href="contact.html" class="btn-secondary nav-link">Contact</a>
            <a href="resume.html" class="btn-secondary nav-link">Resume</a>
            <a href="documents/Resume.pdf" class="btn-primary nav-link nav-link-cta" download data-analytics="download_resume">Download Resume</a>
            <button id="theme-toggle" class="btn-icon theme-toggle" type="button" aria-pressed="false">
              <span class="theme-icon moon" aria-hidden="true">
                <svg viewBox="0 0 24 24" role="img" focusable="false"><path d="M21 12.79A9 9 0 0 1 11.21 3a7 7 0 1 0 9.79 9.79Z"/></svg>
              </span>
              <span class="theme-icon sun" aria-hidden="true">
                <svg viewBox="0 0 24 24" role="img" focusable="false"><path d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Zm0 4v-2m0-16V2m10 10h-2M4 12H2m18.36 7.36-1.42-1.42M7.05 7.05 5.63 5.63m12.73 0-1.42 1.42M7.05 16.95 5.63 18.36"/></svg>
              </span>
              <span class="theme-label sr-only">Toggle theme</span>
            </button>
          </div>
        </div>
      </nav>`;
    const cur = location.pathname.split('/').pop() || 'index.html';
    $$('.nav-link').forEach(l=>{
      if(l.getAttribute('href')===cur){
        l.classList.replace('btn-secondary','btn-primary');
        l.setAttribute('aria-current','page');
      }
    });
    const burger = host.querySelector('#nav-toggle');
    const menu   = host.querySelector('#primary-menu');

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
    }

    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const nextTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        applyTheme(nextTheme);
      });
      applyTheme(getPreferredTheme());
    }
  }
  function injectFooter(){
    const f = $('footer');
    if(!f) return;
    f.classList.add('footer');
    const year = new Date().getFullYear();
    f.innerHTML=`
      <div class="social" role="navigation" aria-label="Social links">
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email Daniel Short">
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
