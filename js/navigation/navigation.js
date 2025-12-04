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
    clampDropdownsToViewport();
  }
  function injectNav(){
    const host = $('#combined-header-nav');
    if(!host) return;
    const animate = !sessionStorage.getItem('navEntryPlayed');
    sessionStorage.setItem('navEntryPlayed','yes');
    const projectAsset = (id, ext = 'webp') => `/img/projects/${id}.${ext}`;
    const renderProjectCard = (project, index) => {
      const thumb = projectAsset(project.id, 'webp');
      return `<a href="portfolio.html?project=${project.id}" class="nav-project-card" data-project-id="${project.id}" role="listitem">
                <span class="nav-project-thumb" style="background-image:url('${thumb}');"></span>
                <span class="nav-project-meta">
                  <span class="nav-project-badge">#${index}</span>
                  <span class="nav-dropdown-title">${project.title}</span>
                  <span class="nav-dropdown-subtitle">${project.subtitle}</span>
                </span>
              </a>`;
    };
    const renderHighlightCard = (item) => `
      <a href="${item.href}" class="nav-highlight-card" role="listitem"${item.external ? ' target="_blank" rel="noopener noreferrer"' : ''}>
        <span class="nav-highlight-label">${item.label}</span>
        <span class="nav-dropdown-title">${item.title}</span>
        <span class="nav-dropdown-subtitle">${item.subtitle}</span>
      </a>
    `;
    const portfolioHighlights = [
      { id: 'shapeClassifier',  title: 'Shape Classifier Demo',      subtitle: 'Handwritten shape recognition' },
      { id: 'chatbotLora',      title: 'Chatbot (LoRA + RAG)',       subtitle: 'Fine-tuned assistant with RAG' },
      { id: 'sheetMusicUpscale',title: 'Sheet Music Restoration',    subtitle: 'UNet watermark removal + VDSR' },
      { id: 'digitGenerator',   title: 'Synthetic Digit Generator',  subtitle: 'Variational autoencoder (VAE)' },
      { id: 'nonogram',         title: 'Nonogram Solver',            subtitle: '94% accuracy reinforcement learning' }
    ];
    const portfolioMenu = `
      <div class="nav-dropdown-inner nav-dropdown-inner-portfolio">
        <div class="nav-dropdown-column nav-dropdown-column-list nav-portfolio-stack">
          <div class="nav-dropdown-header" aria-hidden="true">Top 5 Projects</div>
          <div class="nav-project-grid nav-project-stack" role="list">
            ${portfolioHighlights.map((p, i) => renderProjectCard(p, i + 1)).join('')}
          </div>
          <div class="nav-dropdown-footer nav-dropdown-footer-inline">
            <a href="portfolio.html?view=all#filters" class="nav-dropdown-link nav-dropdown-all" role="button">
              <span class="nav-dropdown-title">View all projects</span>
              <span class="nav-dropdown-subtitle">Browse the complete portfolio</span>
            </a>
          </div>
        </div>
      </div>
    `;
    const contributionSections = [
      { id: 'public-contributions', title: 'Public Reports', subtitle: 'Economic outlooks & city budgets' },
      { id: 'council-briefings',   title: 'Council Briefings', subtitle: 'Bi-weekly council intelligence' },
      { id: 'enewsletters',        title: 'Stakeholder eNews', subtitle: 'Industry pacing & KPI updates' }
    ];
    const latestStakeholder = {
      label:'Latest Stakeholder',
      title:'Stakeholder eNewsletter · November 2025',
      subtitle:'Fresh pacing & KPI insights',
      href:'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=1370d609e9',
      external:true
    };
    const latestCouncil = {
      label:'Latest Council Briefing',
      title:'Council Briefing · Nov 24, 2025',
      subtitle:'Newest council intel drop',
      href:'https://ccbrief.my.canva.site/city-council-briefing-nov-24-2025',
      external:true
    };
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
        <div class="nav-dropdown-column nav-dropdown-column-actions nav-dropdown-column-highlights">
          <div class="nav-dropdown-header">Latest releases</div>
          <div class="nav-dropdown-actions" role="list">
            ${renderHighlightCard(latestStakeholder)}
            ${renderHighlightCard(latestCouncil)}
          </div>
        </div>
      </div>
    `;
    const resumeMenu = `
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
    `;
    const contactOptions = [
      { title: 'Message through website', subtitle: 'Send a message via website', href: 'contact.html#contact-modal', recommended: true, modalLink: true },
      { title: 'Email', subtitle: 'daniel@danielshort.me', href: 'mailto:daniel@danielshort.me' },
      { title: 'LinkedIn', subtitle: 'linkedin.com/in/danielshort3', href: 'https://www.linkedin.com/in/danielshort3/', external: true },
      { title: 'GitHub', subtitle: 'github.com/danielshort3', href: 'https://github.com/danielshort3', external: true }
    ];
    const contactMenu = `
      <div class="nav-dropdown-inner nav-dropdown-inner-contact">
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
        <div class="nav-dropdown-column nav-dropdown-column-actions nav-dropdown-mini-form-wrap">
          <div class="nav-dropdown-header">Quick message</div>
          <form id="nav-contact-mini-form" class="nav-mini-form" novalidate>
            <label for="nav-mini-name">Name</label>
            <input id="nav-mini-name" name="name" type="text" autocomplete="name" required placeholder="Jane Doe">
            <label for="nav-mini-email">Email</label>
            <input id="nav-mini-email" name="email" type="email" autocomplete="email" required placeholder="you@example.com">
            <label for="nav-mini-message">How can I help?</label>
            <textarea id="nav-mini-message" name="message" rows="3" required placeholder="Project, role, or opportunity details"></textarea>
            <p class="nav-mini-status" id="nav-mini-status" role="status" aria-live="polite" hidden></p>
            <button type="submit" class="btn-primary nav-mini-submit">Start the conversation</button>
          </form>
        </div>
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
    setupMiniContactForm();
    setupDropdown(host.querySelector('.nav-item-portfolio'));
    setupDropdown(host.querySelector('.nav-item-contributions'));
    setupDropdown(host.querySelector('.nav-item-resume'));
    setupDropdown(host.querySelector('.nav-item-contact'));
    setupPortfolioPreview(host.querySelector('.nav-item-portfolio'));
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
  function setupPortfolioPreview(item){
    if(!item) return;
    const dropdown = item.querySelector('.nav-dropdown');
    let preview = item.querySelector('.nav-project-preview');
    const cards = item.querySelectorAll('.nav-project-card');
    if(!dropdown || !cards.length) return;
    if(!preview){
      preview = document.createElement('div');
      preview.className = 'nav-project-preview';
      preview.setAttribute('aria-hidden', 'true');
      item.appendChild(preview);
    } else {
      item.appendChild(preview);
    }
    let videoEl = preview.querySelector('video');
    if(!videoEl){
      videoEl = document.createElement('video');
      videoEl.muted = true;
      videoEl.loop = true;
      videoEl.playsInline = true;
      videoEl.autoplay = true;
      videoEl.setAttribute('aria-hidden','true');
      videoEl.className = 'nav-project-preview-video';
      preview.appendChild(videoEl);
    }
    const setAspectRatio = (w, h) => {
      if (!w || !h) return;
      preview.style.aspectRatio = `${w}/${h}`;
    };
    const positionPreview = (card) => {
      const dropdownRect = dropdown.getBoundingClientRect();
      const cardRect = card?.getBoundingClientRect() || cards[0]?.getBoundingClientRect();
      const gutter = 16;
      const vw = document.documentElement?.clientWidth || window.innerWidth || 0;
      const previewWidth = preview.offsetWidth || 0;
      let left = dropdownRect.right + gutter;
      if (vw) {
        const maxLeft = Math.max(gutter, vw - previewWidth - gutter);
        left = Math.min(left, maxLeft);
      }
      preview.style.left = `${left}px`;
      if (cardRect) {
        preview.style.top = `${cardRect.top}px`;
      } else {
        preview.style.top = `${dropdownRect.top}px`;
      }
    };
    const setVideoSource = (id) => {
      if(!videoEl) return;
      const src = `/img/projects/${id}.webm`;
      if(videoEl.dataset.src === src) return;
      videoEl.pause();
      videoEl.removeAttribute('src');
      videoEl.load();
      videoEl.dataset.src = src;
      videoEl.src = src;
      videoEl.poster = `/img/projects/${id}.webp`;
      videoEl.onloadedmetadata = () => setAspectRatio(videoEl.videoWidth, videoEl.videoHeight);
      videoEl.play().catch(() => {/* ignore autoplay blocks */});
    };
    const preloadImage = (id) => {
      const img = new Image();
      img.onload = () => setAspectRatio(img.naturalWidth, img.naturalHeight);
      img.src = `/img/projects/${id}.webp`;
    };
    const showPreview = (card) => {
      if(!card) return;
      const id = card.getAttribute('data-project-id');
      if(!id) return;
      preview.style.setProperty('--preview-image', `url('/img/projects/${id}.webp')`);
      setVideoSource(id);
      preloadImage(id);
      positionPreview(card);
      preview.classList.add('nav-project-preview-visible');
    };
    cards.forEach((card, index) => {
      const activate = () => showPreview(card);
      const handleLeave = (event) => {
        const next = event.relatedTarget;
        if (next && next.closest && next.closest('.nav-project-card')) return;
        hide();
      };
      card.addEventListener('mouseenter', activate);
      card.addEventListener('focus', activate);
      card.addEventListener('pointerleave', handleLeave);
    });
    const hide = () => {
      preview.classList.remove('nav-project-preview-visible');
      if (videoEl) {
        videoEl.pause();
      }
    };
    dropdown.addEventListener('pointerleave', hide);
    item.addEventListener('mouseleave', hide);
    item.addEventListener('focusout', (event) => {
      const next = event.relatedTarget;
      if(!next || !item.contains(next)){
        hide();
      }
    });
    const originalClose = item.__closeDropdown;
    item.__closeDropdown = () => {
      originalClose && originalClose();
      hide();
    };
  }
  function setupMiniContactForm(){
    const form = document.getElementById('nav-contact-mini-form');
    if(!form) return;
    const nameInput = form.querySelector('#nav-mini-name');
    const emailInput = form.querySelector('#nav-mini-email');
    const messageInput = form.querySelector('#nav-mini-message');
    const statusEl = document.getElementById('nav-mini-status');
    const setStatus = (message = '', tone = 'info') => {
      if (!statusEl) return;
      statusEl.textContent = message;
      if (message) {
        statusEl.dataset.tone = tone;
        statusEl.hidden = false;
      } else {
        statusEl.hidden = true;
        delete statusEl.dataset.tone;
      }
    };
    const emailValid = (value = '') => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim());
    const persistDraft = (payload) => {
      try {
        sessionStorage.setItem('navContactPrefill', JSON.stringify(payload));
      } catch { /* noop */ }
    };
    const hydrateContactPage = (payload) => {
      const contactForm = document.getElementById('contact-form');
      if (!contactForm || !payload) return false;
      const nameField = contactForm.querySelector('#contact-name');
      const emailField = contactForm.querySelector('#contact-email');
      const messageField = contactForm.querySelector('#contact-message');
      if (nameField && payload.name) nameField.value = payload.name;
      if (emailField && payload.email) emailField.value = payload.email;
      if (messageField && payload.message) messageField.value = payload.message;
      return true;
    };
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const payload = {
        name: (nameInput?.value || '').trim(),
        email: (emailInput?.value || '').trim(),
        message: (messageInput?.value || '').trim()
      };
      if (!payload.name || !payload.email || !payload.message) {
        setStatus('Please add your name, email, and a short note.', 'error');
        return;
      }
      if (!emailValid(payload.email)) {
        setStatus('That email looks off. Try again?', 'error');
        return;
      }
      setStatus('Opening the contact form…', 'info');
      persistDraft(payload);
      if (location.pathname.includes('/contact.html')) {
        const hydrated = hydrateContactPage(payload);
        if (hydrated) {
          try { sessionStorage.removeItem('navContactPrefill'); } catch { /* noop */ }
        }
        location.hash = '#contact-modal';
      } else {
        window.location.href = 'contact.html#contact-modal';
      }
    });
    // Hydrate if we're already on contact.html and a draft exists
    try {
      const cached = sessionStorage.getItem('navContactPrefill');
      if (cached) {
        const payload = JSON.parse(cached);
        const hydrated = hydrateContactPage(payload);
        if (hydrated) sessionStorage.removeItem('navContactPrefill');
      }
    } catch { /* ignore */ }
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
