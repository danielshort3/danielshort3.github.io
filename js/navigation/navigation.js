/* ===================================================================
   File: navigation.js
   Purpose: Injects header navigation and footer components
=================================================================== */
(() => {
  'use strict';
  const $  = (s, c=document) => c.querySelector(s);
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const NAV_HEIGHT_FALLBACK = 72;
  let cachedNavHeight = null;

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
    injectNav();
    injectFooter();
    setNavHeight();
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
  function injectNav(){
    const host = $('#combined-header-nav');
    if(!host) return;
    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed','yes');
    const portfolioHighlights = [
      { id: 'shapeClassifier',  title: 'Shape Classifier Demo',      subtitle: 'Handwritten shape recognition' },
      { id: 'chatbotLora',      title: 'Chatbot (LoRA + RAG)',       subtitle: 'Fine-tuned assistant with RAG' },
      { id: 'sheetMusicUpscale',title: 'Sheet Music Restoration',    subtitle: 'UNet watermark removal + VDSR' },
      { id: 'digitGenerator',   title: 'Synthetic Digit Generator',  subtitle: 'Variational autoencoder (VAE)' },
      { id: 'nonogram',         title: 'Nonogram Solver',            subtitle: '94% accuracy reinforcement learning' }
    ];
    const portfolioMenu = `
      <div class="nav-dropdown-header" aria-hidden="true">Top 5 Projects</div>
      <div class="nav-dropdown-list" role="list">
        ${portfolioHighlights.map(
          p => `<a href="portfolio.html?project=${p.id}" class="nav-dropdown-link" role="listitem">
                  <span class="nav-dropdown-title">${p.title}</span>
                  <span class="nav-dropdown-subtitle">${p.subtitle}</span>
                </a>`
        ).join('')}
      </div>
      <div class="nav-dropdown-footer">
        <a href="portfolio.html?view=all#filters" class="nav-dropdown-link nav-dropdown-all" role="button">
          <span class="nav-dropdown-title">View all projects</span>
          <span class="nav-dropdown-subtitle">Browse the complete portfolio</span>
        </a>
      </div>
    `;
    const contributionSections = [
      { id: 'public-contributions', title: 'Public Reports', subtitle: 'Economic outlooks & city budgets' },
      { id: 'council-briefings',   title: 'Council Briefings', subtitle: 'Bi-weekly council intelligence' },
      { id: 'enewsletters',        title: 'Stakeholder eNews', subtitle: 'Industry pacing & KPI updates' }
    ];
    const contributionsMenu = `
      <div class="nav-dropdown-header" aria-hidden="true">Browse categories</div>
      <div class="nav-dropdown-list" role="list">
        ${contributionSections.map(
          c => `<a href="contributions.html#${c.id}" class="nav-dropdown-link" role="listitem">
                  <span class="nav-dropdown-title">${c.title}</span>
                  <span class="nav-dropdown-subtitle">${c.subtitle}</span>
                </a>`
        ).join('')}
      </div>
    `;
    const resumeMenu = `
      <div class="nav-dropdown-header" aria-hidden="true">Download</div>
      <div class="nav-dropdown-list" role="list">
        <a href="documents/Resume.pdf" class="nav-dropdown-link" role="listitem" download>
          <span class="nav-dropdown-title">Resume (PDF)</span>
          <span class="nav-dropdown-subtitle">Latest copy, ready to share</span>
        </a>
      </div>
    `;
    const contactOptions = [
      { title: 'Email', subtitle: 'danielshort3@gmail.com', href: 'mailto:danielshort3@gmail.com' },
      { title: 'LinkedIn', subtitle: 'linkedin.com/in/danielshort3', href: 'https://www.linkedin.com/in/danielshort3/', external: true },
      { title: 'GitHub', subtitle: 'github.com/danielshort3', href: 'https://github.com/danielshort3', external: true }
    ];
    const contactMenu = `
      <div class="nav-dropdown-header" aria-hidden="true">Get in touch</div>
      <div class="nav-dropdown-list" role="list">
        ${contactOptions.map((option) => {
          const attrs = option.external ? ' target="_blank" rel="noopener noreferrer"' : '';
          return `<a href="${option.href}" class="nav-dropdown-link" role="listitem"${attrs}>
                    <span class="nav-dropdown-title">${option.title}</span>
                    <span class="nav-dropdown-subtitle">${option.subtitle}</span>
                  </a>`;
        }).join('')}
      </div>
    `;
    host.innerHTML=`
      <nav class="nav ${animate?'animate-entry':''}" aria-label="Primary">
        <div class="wrapper nav-wrapper">
          <a href="index.html" class="brand" aria-label="Home">
            <img src="img/ui/logo-64.png" srcset="img/ui/logo-64.png 1x, img/ui/logo-192.png 3x" sizes="64px" alt="DS logo" class="brand-logo" decoding="async" loading="eager" width="64" height="64">
            <span class="brand-name">
              <span class="brand-title">Daniel Short</span>
              <span class="brand-divider" aria-hidden="true"></span>
              <span class="brand-tagline">
                <span class="brand-tagline-chunk">Data Science</span>
                <span class="brand-tagline-chunk">&amp; Analytics</span>
              </span>
            </span>
          </a>
          <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false" aria-controls="primary-menu">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </button>
          <div id="primary-menu" class="nav-row" data-collapsible role="navigation">
            <a href="index.html" class="nav-link">Home</a>
            <div class="nav-item nav-item-portfolio">
              <a href="portfolio.html" class="nav-link nav-link-has-menu" aria-haspopup="true">
                Portfolio
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" aria-label="Highlighted projects">
                ${portfolioMenu}
              </div>
            </div>
            <div class="nav-item nav-item-contributions">
              <a href="contributions.html" class="nav-link nav-link-has-menu" aria-haspopup="true">
                Contributions
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" aria-label="Contributions categories">
                ${contributionsMenu}
              </div>
            </div>
            <div class="nav-item nav-item-resume">
              <a href="resume.html" class="nav-link nav-link-has-menu" aria-haspopup="true">
                Resume
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" aria-label="Resume download">
                ${resumeMenu}
              </div>
            </div>
            <div class="nav-item nav-item-contact">
              <a href="contact.html" class="nav-link nav-link-cta nav-link-has-menu" aria-haspopup="true">
                Contact
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown nav-dropdown-contact" aria-label="Contact options">
                ${contactMenu}
              </div>
            </div>
          </div>
        </div>
      </nav>`;
    const cur = location.pathname.split('/').pop() || 'index.html';
    $$('.nav-link').forEach(l=>{
      if(l.getAttribute('href')===cur){
        l.classList.add('is-current');
        l.setAttribute('aria-current','page');
      }
    });
    const burger = host.querySelector('#nav-toggle');
    const menu   = host.querySelector('#primary-menu');
    setupDropdown(host.querySelector('.nav-item-portfolio'));
    setupDropdown(host.querySelector('.nav-item-contributions'));
    setupDropdown(host.querySelector('.nav-item-resume'));
    setupDropdown(host.querySelector('.nav-item-contact'));

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
  const closeActiveDropdowns = (excludeItem) => {
    document.querySelectorAll('.nav-item.dropdown-open').forEach((openItem) => {
      if (openItem === excludeItem) return;
      if (typeof openItem.__closeDropdown === 'function') {
        openItem.__closeDropdown();
      } else {
        openItem.classList.remove('dropdown-open');
      }
    });
  };
  function setupDropdown(item){
    if(!item) return;
    const dropdown = item.querySelector('.nav-dropdown');
    if(!dropdown) return;
    let closeTimer = null;
    const close = () => {
      clearTimeout(closeTimer);
      item.classList.remove('dropdown-open');
    };
    item.__closeDropdown = close;
    const open = () => {
      clearTimeout(closeTimer);
      closeActiveDropdowns(item);
      item.classList.add('dropdown-open');
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
