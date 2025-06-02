/* ===================================================================
   File: navigation.js
   Purpose: Inject navigation and footer elements
=================================================================== */
(() => {
  'use strict';
  const $ = (sel, ctx=document) => ctx.querySelector(sel);
  const $$= (sel, ctx=document) => [...ctx.querySelectorAll(sel)];

  function injectNav(){
    const host = $('#combined-header-nav');
    if(!host) return;
    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed','yes');
    host.innerHTML=`
      <nav class="nav ${animate?'animate-entry':''}">
        <div class="wrapper">
          <a href="index.html" class="brand">
            <img src="img/icons/logo.png" alt="DS logo" class="brand-logo">
            <span class="brand-name">
              <span class="brand-line name">Daniel Short</span>
              <span class="brand-line divider">│</span>
              <span class="brand-line tagline">Data & Insights</span>
            </span>
          </a>
          <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </button>
          <nav class="nav-row" data-collapsible>
            <a href="index.html"           class="btn-secondary nav-link">Home</a>
            <a href="portfolio.html"       class="btn-secondary nav-link">Portfolio</a>
            <a href="contributions.html"   class="btn-secondary nav-link">Contributions</a>
            <a href="contact.html"         class="btn-secondary nav-link">Contact</a>
            <a href="documents/Resume.pdf" class="btn-secondary nav-link" target="_blank" download>Resume</a>
          </nav>
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
    const menu   = host.querySelector('.nav-row');
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
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email"><i class="fas fa-envelope"></i></a>
        <a class="btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
        <a class="btn-icon" href="https://github.com/danielshort3" target="_blank" aria-label="GitHub"><i class="fab fa-github"></i></a>
      </div>
      <p>© ${year} Daniel Short. All rights reserved.</p>`;
  }

  document.addEventListener('DOMContentLoaded', () => {
    injectNav();
    injectFooter();
  });
})();
