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
      <nav class="nav ${animate?'animate-entry':''}">
        <div class="wrapper">
          <a href="index.html" class="brand">
            <img src="img/ui/logo.png" alt="DS logo" class="brand-logo">
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
            <a href="documents/Resume.pdf" class="btn-secondary nav-link" target="_blank" rel="noopener" download>Resume</a>
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
    if(burger && menu){
      burger.addEventListener('click', () => {
        const headerBar = burger.closest('.nav') || host;
        const headerBottom = headerBar.getBoundingClientRect().bottom;
        menu.style.top = `${headerBottom}px`;
        const open = menu.classList.toggle('open');
        burger.setAttribute('aria-expanded', open);
        document.body.classList.toggle('menu-open', open);
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
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email"><i class="fas fa-envelope" aria-hidden="true"></i></a>
        <a class="btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" rel="noopener" aria-label="LinkedIn"><i class="fab fa-linkedin-in" aria-hidden="true"></i></a>
        <a class="btn-icon" href="https://github.com/danielshort3" target="_blank" rel="noopener" aria-label="GitHub"><i class="fab fa-github" aria-hidden="true"></i></a>
      </div>
      <p>© ${year} Daniel Short. All rights reserved.</p>`;
  }
})();
