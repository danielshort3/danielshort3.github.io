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
    updateNavDropdownOffset();
  }
  function injectNav(){
    const host = $('#combined-header-nav');
    if(!host) return;
    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed','yes');
    const iconSprite = {
      grid: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
      spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.4 4.2L18 9l-4.1 1.8L12 15l-1.9-4.2L6 9l4.6-1.8z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><circle cx="18.5" cy="15.5" r="1.4" fill="currentColor"/><circle cx="7" cy="17" r="1.2" fill="currentColor"/></svg>',
      doc: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 3.5h7l4 4v13h-11z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M13.5 3.5v4h4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M8.5 12.5h7M8.5 16h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
      chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 18l-3 3V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M7 9h10M7 13h6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
      link: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 14.5l-2 2a3.5 3.5 0 0 1-5-5l4-4a3.5 3.5 0 0 1 5 0l1 1" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M14.5 9.5l1-1a3.5 3.5 0 0 1 5 5l-4 4a3.5 3.5 0 0 1-5 0l-1-1" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
      mail: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="4.5" width="17" height="15" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3.5 7l8 5.2L20.5 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
    };
    const renderActionTile = (action) => {
      const attrParts = [];
      if(action.external){
        attrParts.push('target="_blank"', 'rel="noopener noreferrer"');
      }
      if(action.modalLink){
        attrParts.push('data-contact-modal-link="true"');
      }
      if(action.download){
        attrParts.push('download');
      }
      const attrs = attrParts.length ? ` ${attrParts.join(' ')}` : '';
      const badge = action.badge ? `<span class="nav-dropdown-badge" aria-hidden="true">${action.badge}</span>` : '';
      const icon = iconSprite[action.icon] || iconSprite.grid;
      return `<a href="${action.href}" class="nav-dropdown-card" role="listitem"${attrs}>
                <span class="nav-dropdown-icon" aria-hidden="true">${icon}</span>
                <span class="nav-dropdown-card-copy">
                  <span class="nav-dropdown-title">${action.title}${badge}</span>
                  <span class="nav-dropdown-subtitle">${action.subtitle}</span>
                </span>
              </a>`;
    };
    const renderActionColumn = (label, actions) => `
      <div class="nav-dropdown-column nav-dropdown-column-actions">
        <div class="nav-dropdown-header">${label}</div>
        <div class="nav-dropdown-actions" role="list">
          ${actions.map(renderActionTile).join('')}
        </div>
      </div>
    `;
    const portfolioHighlights = [
      { id: 'shapeClassifier',  title: 'Shape Classifier Demo',      subtitle: 'Handwritten shape recognition' },
      { id: 'chatbotLora',      title: 'Chatbot (LoRA + RAG)',       subtitle: 'Fine-tuned assistant with RAG' },
      { id: 'sheetMusicUpscale',title: 'Sheet Music Restoration',    subtitle: 'UNet watermark removal + VDSR' },
      { id: 'digitGenerator',   title: 'Synthetic Digit Generator',  subtitle: 'Variational autoencoder (VAE)' },
      { id: 'nonogram',         title: 'Nonogram Solver',            subtitle: '94% accuracy reinforcement learning' }
    ];
    const portfolioMenu = `
      <div class="nav-dropdown-inner">
        <div class="nav-dropdown-column nav-dropdown-column-list">
          <div class="nav-dropdown-header" aria-hidden="true">Top 5 Projects</div>
          <div class="nav-dropdown-list" role="list">
            ${portfolioHighlights.map(
              p => `<a href="portfolio.html?project=${p.id}" class="nav-dropdown-link nav-dropdown-featured" role="listitem">
                      <span class="nav-dropdown-title">${p.title}</span>
                      <span class="nav-dropdown-subtitle">${p.subtitle}</span>
                    </a>`
            ).join('')}
          </div>
        </div>
        ${renderActionColumn('Visual shortcuts', [
          { title:'Browse all projects', subtitle:'See every build with filters', href:'portfolio.html?view=all#filters', icon:'grid' },
          { title:'Run the chatbot demo', subtitle:'Try the LoRA + RAG assistant live', href:'demos/chatbot-demo.html', icon:'spark' }
        ])}
      </div>
    `;
    const contributionSections = [
      { id: 'public-contributions', title: 'Public Reports', subtitle: 'Economic outlooks & city budgets' },
      { id: 'council-briefings',   title: 'Council Briefings', subtitle: 'Bi-weekly council intelligence' },
      { id: 'enewsletters',        title: 'Stakeholder eNews', subtitle: 'Industry pacing & KPI updates' }
    ];
    const contributionsMenu = `
      <div class="nav-dropdown-inner">
        <div class="nav-dropdown-column nav-dropdown-column-list">
          <div class="nav-dropdown-header" aria-hidden="true">Browse categories</div>
          <div class="nav-dropdown-list" role="list">
            ${contributionSections.map(
              c => `<a href="contributions.html#${c.id}" class="nav-dropdown-link" role="listitem">
                    <span class="nav-dropdown-title">${c.title}</span>
                    <span class="nav-dropdown-subtitle">${c.subtitle}</span>
                  </a>`
            ).join('')}
          </div>
        </div>
        ${renderActionColumn('Visual shortcuts', [
          { title:'Contributions overview', subtitle:'Jump into reports & briefings', href:'contributions.html', icon:'grid' },
          { title:'Request a briefing', subtitle:'Book a walkthrough or follow-up', href:'contact.html#contact-modal', icon:'chat', modalLink:true }
        ])}
      </div>
    `;
    const resumeMenu = `
      <div class="nav-dropdown-inner">
        <div class="nav-dropdown-column nav-dropdown-column-list">
          <div class="nav-dropdown-header" aria-hidden="true">Resume shortcuts</div>
          <div class="nav-dropdown-list" role="list">
            <a href="resume.html" class="nav-dropdown-link" role="listitem">
              <span class="nav-dropdown-title">View Resume</span>
              <span class="nav-dropdown-subtitle">Open the full resume page</span>
            </a>
            <a href="documents/Resume.pdf" class="nav-dropdown-link" role="listitem" download>
              <span class="nav-dropdown-title">Download Resume</span>
              <span class="nav-dropdown-subtitle">Save the latest PDF copy</span>
            </a>
          </div>
        </div>
        ${renderActionColumn('Visual shortcuts', [
          { title:'LinkedIn profile', subtitle:'See endorsements and background', href:'https://www.linkedin.com/in/danielshort3/', icon:'link', external:true },
          { title:'Shareable PDF', subtitle:'Download the latest resume instantly', href:'documents/Resume.pdf', icon:'doc', download:true }
        ])}
      </div>
    `;
    const contactOptions = [
      { title: 'Message through website', subtitle: 'Send a message via website', href: 'contact.html#contact-modal', recommended: true, modalLink: true },
      { title: 'Email', subtitle: 'daniel@danielshort.me', href: 'mailto:daniel@danielshort.me' },
      { title: 'LinkedIn', subtitle: 'linkedin.com/in/danielshort3', href: 'https://www.linkedin.com/in/danielshort3/', external: true },
      { title: 'GitHub', subtitle: 'github.com/danielshort3', href: 'https://github.com/danielshort3', external: true }
    ];
    const contactMenu = `
      <div class="nav-dropdown-inner">
        <div class="nav-dropdown-column nav-dropdown-column-list">
          <div class="nav-dropdown-header" aria-hidden="true">Get in touch</div>
          <div class="nav-dropdown-list" role="list">
            ${contactOptions.map((option) => {
              const attrParts = [];
              if(option.external){
                attrParts.push('target="_blank"', 'rel="noopener noreferrer"');
              }
              if(option.modalLink){
                attrParts.push('data-contact-modal-link="true"');
              }
              const attrs = attrParts.length ? ` ${attrParts.join(' ')}` : '';
              const badge = option.recommended ? '<span class="nav-dropdown-badge" aria-hidden="true">Recommended</span>' : '';
              return `<a href="${option.href}" class="nav-dropdown-link${option.recommended ? ' nav-dropdown-link-recommended' : ''}" role="listitem"${attrs}>
                        <span class="nav-dropdown-title">${option.title}${badge}</span>
                        <span class="nav-dropdown-subtitle">${option.subtitle}</span>
                      </a>`;
            }).join('')}
          </div>
        </div>
        ${renderActionColumn('Visual shortcuts', [
          { title:'Start a project', subtitle:'Share what you need and timelines', href:'contact.html#contact-modal', icon:'spark', modalLink:true, badge:'Recommended' },
          { title:'Email directly', subtitle:'daniel@danielshort.me', href:'mailto:daniel@danielshort.me', icon:'mail' }
        ])}
      </div>
    `;
    const dropdownIds = {
      portfolio: 'nav-dropdown-portfolio',
      contributions: 'nav-dropdown-contributions',
      resume: 'nav-dropdown-resume',
      contact: 'nav-dropdown-contact'
    };
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
              <a href="portfolio.html" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="${dropdownIds.portfolio}">
                Portfolio
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" id="${dropdownIds.portfolio}" aria-label="Highlighted projects">
                ${portfolioMenu}
              </div>
            </div>
            <div class="nav-item nav-item-contributions">
              <a href="contributions.html" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="${dropdownIds.contributions}">
                Contributions
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" id="${dropdownIds.contributions}" aria-label="Contributions categories">
                ${contributionsMenu}
              </div>
            </div>
            <div class="nav-item nav-item-resume">
              <a href="resume.html" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="${dropdownIds.resume}">
                Resume
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown" id="${dropdownIds.resume}" aria-label="Resume download">
                ${resumeMenu}
              </div>
            </div>
            <div class="nav-item nav-item-contact">
              <a href="contact.html" class="nav-link nav-link-cta nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="${dropdownIds.contact}">
                Contact
                <span class="nav-link-caret" aria-hidden="true"></span>
              </a>
              <div class="nav-dropdown nav-dropdown-contact" id="${dropdownIds.contact}" aria-label="Contact options">
                ${contactMenu}
              </div>
            </div>
          </div>
        </div>
      </nav>`;
    const normalizePath = (path) => {
      if (!path) return '/';
      try {
        path = new URL(path, location.href).pathname;
      } catch {
        if (!path.startsWith('/')) path = `/${path}`;
      }
      path = path.replace(/\/index\.html$/i, '/');
      path = path.replace(/\/+$/, '');
      if (!path) path = '/';
      return path;
    };
    const currentPath = normalizePath(location.pathname);
    const altCurrentPath = currentPath.startsWith('/pages/')
      ? normalizePath(currentPath.replace(/^\/pages/, '') || '/')
      : currentPath;
    $$('.nav-link').forEach((link) => {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) return;
      let targetPath = normalizePath(href);
      const targetNoHtml = targetPath.endsWith('.html')
        ? normalizePath(targetPath.replace(/\.html$/i, '') || '/')
        : targetPath;
      const matches = [currentPath, altCurrentPath].some(p => p === targetPath || p === targetNoHtml);
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
    setupDropdown(host.querySelector('.nav-item-contributions'));
    setupDropdown(host.querySelector('.nav-item-resume'));
    setupDropdown(host.querySelector('.nav-item-contact'));

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
        const focusables = menu.querySelectorAll('a,button,[tabindex]:not([tabindex="-1"])');
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
  function injectFooter(){
    const f = $('footer');
    if(!f) return;
    f.classList.add('footer');
    const year = new Date().getFullYear();
    f.innerHTML=`
      <div class="social">
        <a class="btn-icon btn-icon-featured" href="contact.html#contact-modal" data-contact-modal-link="true" aria-label="Send a message">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>
            <path d="M7 9h10"></path>
            <path d="M7 13h6"></path>
          </svg>
        </a>
        <a class="btn-icon" href="mailto:daniel@danielshort.me" aria-label="Email">
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
