/* ===================================================================
   File: navigation.js
   Purpose: Injects header navigation and footer components
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
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
      <nav class="nav ${animate?'animate-entry':''}" aria-label="Primary">
        <div class="wrapper">
          <a href="index.html" class="brand" aria-label="Home">
            <img src="img/ui/logo.png" alt="DS logo" class="brand-logo" decoding="async">
            <span class="brand-name">
              <span class="brand-line name">Daniel Short</span>
              <span class="brand-line divider">│</span>
              <span class="brand-line tagline">Data & Insights</span>
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
  }
  function injectFooter(){
    const f = $('footer');
    if(!f) return;
    f.classList.add('footer');
    const year = new Date().getFullYear();
    f.innerHTML=`
      <div class="social">
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="3" y="5" width="18" height="14" rx="2"/>
            <path d="M3 7l9 6 9-6"/>
          </svg>
        </a>
        <a class="btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M4 4h4v4H4z"/>
            <path d="M4 10h4v10H4z"/>
            <path d="M10 10h4v2a3 3 0 016 0v8h-4v-7a1 1 0 10-2 0v7h-4z"/>
          </svg>
        </a>
        <a class="btn-icon" href="https://github.com/danielshort3" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M8 17l-5-5 5-5"/>
            <path d="M16 7l5 5-5 5"/>
            <path d="M13 5l-2 14"/>
          </svg>
        </a>
      </div>
      <p>© ${year} Daniel Short. All rights reserved. <a href="privacy.html">Privacy&nbsp;Policy</a></p>`;
  }
})();
