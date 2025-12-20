/* ===================================================================
   File: common.js
   Purpose: Site-wide helpers and home page popups
=================================================================== */
(() => {
  'use strict';
  const $  = (s,c=document)=>c.querySelector(s);
  const $$ = (s,c=document)=>[...c.querySelectorAll(s)];
  const on = (n,e,f,o)=>n&&n.addEventListener(e,f,o);
  const run = fn=>typeof fn==='function'&&fn();
  const isPage = (...names)=>names.includes(document.body.dataset.page);
  const CONTACT_CONTEXT_KEY = 'contactOrigin';

  const storeContactOrigin = () => {
    try {
      const title = (document.title || '').trim();
      const url = (window.location && window.location.href) ? window.location.href.trim() : '';
      sessionStorage.setItem(CONTACT_CONTEXT_KEY, JSON.stringify({ title, url, ts: Date.now() }));
    } catch {}
  };

  const trackContactOrigin = () => {
    if (!document || !document.addEventListener) return;
    document.addEventListener('click', (event) => {
      const trigger = event.target.closest('[data-contact-modal-link], #contact-form-toggle');
      if (!trigger) return;
      storeContactOrigin();
    });
    document.addEventListener('submit', (event) => {
      const target = event.target;
      if (target && target.id === 'nav-contact-mini-form') {
        storeContactOrigin();
      }
    });
  };

  let jumpPanelScrollToken = 0;
  let activeJumpPanelScrollToken = 0;
  let jumpPanelScrollTimer = null;
  const beginJumpPanelAutoScroll = (timeoutMs) => {
    jumpPanelScrollToken += 1;
    const token = jumpPanelScrollToken;
    activeJumpPanelScrollToken = token;
    if (jumpPanelScrollTimer) {
      clearTimeout(jumpPanelScrollTimer);
      jumpPanelScrollTimer = null;
    }
    if (typeof timeoutMs === 'number' && timeoutMs > 0) {
      jumpPanelScrollTimer = setTimeout(() => {
        if (activeJumpPanelScrollToken === token) activeJumpPanelScrollToken = 0;
        jumpPanelScrollTimer = null;
      }, timeoutMs);
    }
    return token;
  };
  const endJumpPanelAutoScroll = (token) => {
    if (activeJumpPanelScrollToken === token) activeJumpPanelScrollToken = 0;
    if (jumpPanelScrollTimer) {
      clearTimeout(jumpPanelScrollTimer);
      jumpPanelScrollTimer = null;
    }
  };
  const isJumpPanelAutoScrolling = () => activeJumpPanelScrollToken !== 0;

  const loadedScripts = new Map();
  let portfolioBundle = null;
  let modalsPromise = null;
  let modalsHydrated = false;

  const resetScrollLocks = () => {
    const body = document.body;
    if (!body) return;
    const menu = document.getElementById('primary-menu');
    if (!menu || !menu.classList.contains('open')) {
      body.classList.remove('menu-open');
    }
    if (!document.querySelector('.modal.active')) {
      body.classList.remove('modal-open');
    }
    if (!document.querySelector('.media-viewer.active')) {
      body.classList.remove('media-viewer-open');
    }
  };

  window.addEventListener('pageshow', resetScrollLocks);
  document.addEventListener('DOMContentLoaded', () => {
    resetScrollLocks();
    if (isPage('portfolio')) {
      ensurePortfolioScripts().then(() => {
        run(window.buildPortfolioCarousel);
        run(window.buildPortfolio);
        run(window.initSeeMore);
      }).catch(err => console.warn('Failed to initialize portfolio page', err));
    }
    if (isPage('home')) {
      initSkillPopups();
      initSmoothScrollLinks();
      initJumpPanelSpy();
    }
  });
  trackContactOrigin();

  function loadScriptOnce(src){
    if (loadedScripts.has(src)) return loadedScripts.get(src);
    const promise = new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.src = src;
      tag.async = false;
      tag.onload = () => resolve();
      tag.onerror = () => reject(new Error(`Failed to load script: ${src}`));
      document.head.appendChild(tag);
    });
    loadedScripts.set(src, promise);
    return promise;
  }

  function ensurePortfolioScripts(){
    if (portfolioBundle) return portfolioBundle;
    const chain = ['js/portfolio/projects-data.js','js/portfolio/modal-helpers.js','js/portfolio/portfolio.js']
      .reduce((p, src) => p.then(() => loadScriptOnce(src)), Promise.resolve());
    portfolioBundle = chain.catch(err => {
      portfolioBundle = null;
      throw err;
    });
    return portfolioBundle;
  }

  function hydrateProjectModals(){
    if (modalsHydrated) return;
    if (!Array.isArray(window.PROJECTS)) return;
    const host = $('#modals') || (() => {
      const d = document.createElement('div');
      d.id = 'modals';
      document.body.appendChild(d);
      return d;
    })();
    window.PROJECTS.forEach(p => {
      if ($('#' + p.id + '-modal')) return;
      const modal = document.createElement('div');
      modal.className = 'modal';
      modal.id = `${p.id}-modal`;
      modal.innerHTML = window.generateProjectModal(p);
      host.appendChild(modal);
    });
    modalsHydrated = true;
  }

  function preparePortfolioModals(){
    if (modalsPromise) return modalsPromise;
    modalsPromise = ensurePortfolioScripts()
      .then(() => {
        if (typeof window.generateProjectModal !== 'function' || typeof window.openModal !== 'function') {
          throw new Error('Portfolio modal helpers missing');
        }
        hydrateProjectModals();
      })
      .catch(err => {
        console.warn('Failed to prepare project modals', err);
        modalsHydrated = false;
        modalsPromise = null;
        throw err;
      });
    return modalsPromise;
  }

  function openProjectModal(id){
    if (!id) return;
    const p = preparePortfolioModals();
    if (!p || typeof p.then !== 'function') return;
    return p.then(() => {
      if (typeof window.openModal === 'function') window.openModal(id);
    });
  }

  function initSkillPopups(){
    if (!isPage('home')) return;
    const buttons = $$('.skill-link, [data-project-modal="true"]');
    if (!buttons.length) return;

    const safePreload = () => {
      const promise = preparePortfolioModals();
      if (promise && typeof promise.then === 'function') promise.catch(() => {});
    };

    buttons.forEach(btn => {
      if (btn.dataset.modalBound === 'yes') return;
      btn.dataset.modalBound = 'yes';
      btn.setAttribute('aria-haspopup', 'dialog');
      on(btn, 'pointerenter', safePreload, { once: true });
      on(btn, 'focus', safePreload, { once: true });
      on(btn, 'touchstart', safePreload, { once: true, passive: true });
      on(btn, 'click', (evt) => {
        if (!btn.dataset.project) return;
        if (evt && (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey)) return;
        if (evt && typeof evt.button === 'number' && evt.button !== 0) return;
        const tag = (btn.tagName || '').toLowerCase();
        if (tag === 'a') evt.preventDefault();
        const promise = openProjectModal(btn.dataset.project);
        if (promise && typeof promise.catch === 'function') promise.catch(() => {});
      });
      on(btn, 'keydown', (evt) => {
        const tag = (btn.tagName || '').toLowerCase();
        if (tag === 'a') return;
        if (evt.key === 'Enter' || evt.key === ' ') {
          evt.preventDefault();
          if (!btn.dataset.project) return;
          const promise = openProjectModal(btn.dataset.project);
          if (promise && typeof promise.catch === 'function') promise.catch(() => {});
        }
      });
    });

    const projectFromLocation = () => {
      try {
        const search = (location.search || '').replace(/^\?/, '');
        if (search) {
          const parts = search.split('&');
          for (const kv of parts) {
            const [k, v] = kv.split('=');
            if (decodeURIComponent(k) === 'project' && v) return decodeURIComponent(v);
          }
        }
        if (location.hash && location.hash.length > 1) {
          const hashId = decodeURIComponent(location.hash.slice(1));
          if (!document.getElementById(hashId)) return hashId;
        }
      } catch {
        if (location.hash && location.hash.length > 1) {
          const hashId = location.hash.slice(1);
          if (!document.getElementById(hashId)) return hashId;
        }
      }
      return null;
    };

    const deepLinkId = projectFromLocation();
    if (deepLinkId) {
      const promise = openProjectModal(deepLinkId);
      if (promise && typeof promise.catch === 'function') promise.catch(() => {});
    }
  }

  function initSmoothScrollLinks(){
    if (!isPage('home')) return;
    const links = $$('a[data-smooth-scroll="true"]');
    if (!links.length) return;

    const prefersReducedMotion = () => {
      try {
        return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      } catch {
        return false;
      }
    };

    const smoothScrollToTarget = (target, options = {}) => {
      if (!target) return;
      const startTop = window.scrollY || window.pageYOffset || 0;
      const navOffset = typeof window.getNavOffset === 'function' ? window.getNavOffset() : 0;
      let marginTop = 0;
      try {
        marginTop = parseFloat(window.getComputedStyle(target).scrollMarginTop || '0') || 0;
      } catch {}
      const targetTop = target.getBoundingClientRect().top + startTop - navOffset - marginTop;
      const maxScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      const clampedTop = Math.min(Math.max(0, targetTop), maxScroll);
      const distance = clampedTop - startTop;
      if (Math.abs(distance) < 1) return;

      const jumpPanelToken = options.jumpPanel ? beginJumpPanelAutoScroll() : 0;
      let jumpPanelTailTimer = null;
      let rafId = null;
      let startTime = null;
      let cleaned = false;
      const duration = Math.min(1200, Math.max(260, Math.abs(distance) * 0.6));
      const ease = (t) => (
        t < 0.5
          ? 2 * t * t
          : 1 - Math.pow(-2 * t + 2, 2) / 2
      );

      const releaseJumpPanelAutoScroll = (delayMs = 0) => {
        if (!jumpPanelToken) return;
        if (jumpPanelTailTimer) {
          clearTimeout(jumpPanelTailTimer);
          jumpPanelTailTimer = null;
        }
        if (delayMs > 0) {
          jumpPanelTailTimer = setTimeout(() => {
            endJumpPanelAutoScroll(jumpPanelToken);
          }, delayMs);
          return;
        }
        endJumpPanelAutoScroll(jumpPanelToken);
      };

      const cleanup = ({ cancelled = false } = {}) => {
        if (cleaned) return;
        cleaned = true;
        if (rafId) cancelAnimationFrame(rafId);
        window.removeEventListener('wheel', cancel);
        window.removeEventListener('touchstart', cancel);
        window.removeEventListener('keydown', onKeydown);
        if (jumpPanelToken) {
          if (cancelled) {
            releaseJumpPanelAutoScroll();
          } else {
            releaseJumpPanelAutoScroll(180);
          }
        }
      };

      const cancel = () => {
        cleanup({ cancelled: true });
      };

      const onKeydown = (event) => {
        const key = event?.key;
        if (!key) return;
        if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(key)) {
          cancel();
        }
      };

      const step = (now) => {
        if (cleaned) return;
        if (startTime === null) startTime = now;
        const progress = Math.min(1, (now - startTime) / duration);
        const nextTop = startTop + distance * ease(progress);
        window.scrollTo(0, nextTop);
        if (progress < 1) {
          rafId = requestAnimationFrame(step);
        } else {
          cleanup();
        }
      };

      window.addEventListener('wheel', cancel, { passive: true });
      window.addEventListener('touchstart', cancel, { passive: true });
      window.addEventListener('keydown', onKeydown);
      rafId = requestAnimationFrame(step);
    };

    links.forEach((link) => {
      if (link.dataset.smoothBound === 'yes') return;
      link.dataset.smoothBound = 'yes';
      on(link, 'click', (evt) => {
        if (evt && (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey)) return;
        if (evt && typeof evt.button === 'number' && evt.button !== 0) return;

        const href = link.getAttribute('href') || '';
        if (!href.startsWith('#') || href.length < 2) return;
        const targetId = decodeURIComponent(href.slice(1));
        const target = document.getElementById(targetId);
        if (!target) return;
        const isJumpPanelLink = Boolean(link.closest('.jump-panel'));
        if (prefersReducedMotion()) {
          if (isJumpPanelLink) beginJumpPanelAutoScroll(400);
          return;
        }

        evt.preventDefault();
        smoothScrollToTarget(target, { jumpPanel: isJumpPanelLink });
        try {
          history.pushState(null, '', href);
        } catch {}
      });
    });
  }

  function initJumpPanelSpy(){
    if (!isPage('home')) return;
    const panel = document.querySelector('.jump-panel');
    if (!panel) return;
    const links = $$('.jump-panel-link', panel);
    const hideBtn = panel.querySelector('[data-jump-hide]');
    const showBtn = document.querySelector('[data-jump-show]');
    if (!links.length) return;
    const focusables = [...links];
    if (hideBtn) focusables.push(hideBtn);
    const items = links.map((link) => {
      const href = link.getAttribute('href') || '';
      if (!href.startsWith('#') || href.length < 2) return null;
      let id = href.slice(1);
      try {
        id = decodeURIComponent(id);
      } catch {}
      const target = document.getElementById(id);
      if (!target) return null;
      return { id, link, target };
    }).filter(Boolean);
    if (!items.length) return;

    let activeId = null;
    let ticking = false;
    let manualOverrideId = null;

    const getNavOffset = () => {
      if (typeof window.getNavOffset === 'function') {
        return window.getNavOffset();
      }
      return 72;
    };

    const setActive = (id) => {
      items.forEach((item) => {
        const isActive = item.id === id;
        item.link.classList.toggle('is-active', isActive);
        if (isActive) {
          item.link.setAttribute('aria-current', 'location');
        } else {
          item.link.removeAttribute('aria-current');
        }
      });
    };

    const setManualActive = (item, event) => {
      if (!item) return;
      if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)) return;
      if (event && typeof event.button === 'number' && event.button !== 0) return;
      manualOverrideId = item.id;
      activeId = item.id;
      setActive(item.id);
    };

    const blurAfterPointer = (link) => {
      requestAnimationFrame(() => {
        if (document.activeElement === link) link.blur();
      });
    };

    const bindPointerBlur = (link) => {
      if (link.dataset.jumpBlur === 'yes') return;
      link.dataset.jumpBlur = 'yes';
      if ('PointerEvent' in window) {
        link.addEventListener('pointerup', (event) => {
          const type = event?.pointerType;
          if (type && !['mouse', 'touch', 'pen'].includes(type)) return;
          blurAfterPointer(link);
        });
      } else {
        link.addEventListener('mouseup', (event) => {
          if (event && event.button && event.button !== 0) return;
          blurAfterPointer(link);
        });
        link.addEventListener('touchend', () => blurAfterPointer(link), { passive: true });
      }
    };

    const shouldCondenseOnScroll = () => {
      try {
        return Boolean(window.matchMedia && window.matchMedia('(hover: none) and (pointer: coarse), (max-width: 768px)').matches);
      } catch {
        return true;
      }
    };

    const setPanelCondensed = (condensed) => {
      if (!shouldCondenseOnScroll()) return;
      panel.classList.toggle('is-condensed', condensed);
    };

    const clearPanelFocusOnScroll = () => {
      if (!shouldCondenseOnScroll()) return;
      const active = document.activeElement;
      if (active && panel.contains(active)) active.blur();
    };

    const setFocusable = (el, hidden) => {
      if (!el) return;
      if (hidden) {
        if (!Object.prototype.hasOwnProperty.call(el.dataset, 'jumpTabindex')) {
          const prev = el.getAttribute('tabindex');
          el.dataset.jumpTabindex = prev === null ? '' : prev;
        }
        el.setAttribute('tabindex', '-1');
        return;
      }
      if (!Object.prototype.hasOwnProperty.call(el.dataset, 'jumpTabindex')) {
        return;
      }
      const prev = el.dataset.jumpTabindex;
      delete el.dataset.jumpTabindex;
      if (prev === '') {
        el.removeAttribute('tabindex');
      } else {
        el.setAttribute('tabindex', prev);
      }
    };

    const setPanelHidden = (hidden, { focus = true } = {}) => {
      panel.classList.toggle('is-hidden', hidden);
      panel.setAttribute('aria-hidden', hidden ? 'true' : 'false');
      focusables.forEach((el) => setFocusable(el, hidden));
      if (showBtn) {
        showBtn.setAttribute('aria-hidden', hidden ? 'false' : 'true');
        showBtn.setAttribute('aria-expanded', hidden ? 'false' : 'true');
        showBtn.setAttribute('tabindex', hidden ? '0' : '-1');
      }
      if (hideBtn) {
        hideBtn.setAttribute('aria-expanded', hidden ? 'false' : 'true');
      }
      if (!focus) return;
      if (hidden) {
        showBtn?.focus();
        return;
      }
      links[0]?.focus();
    };

    items.forEach((item) => {
      bindPointerBlur(item.link);
      if (item.link.dataset.jumpManual === 'yes') return;
      item.link.dataset.jumpManual = 'yes';
      item.link.addEventListener('click', (event) => setManualActive(item, event));
    });

    const clearPanelFocus = (event) => {
      if (event?.pointerType && event.pointerType !== 'mouse') return;
      const active = document.activeElement;
      if (active && panel.contains(active)) active.blur();
    };
    panel.addEventListener('pointerleave', clearPanelFocus);
    panel.addEventListener('mouseleave', clearPanelFocus);
    panel.addEventListener('pointerdown', () => setPanelCondensed(false));
    panel.addEventListener('focusin', () => setPanelCondensed(false));

    if (hideBtn && hideBtn.dataset.jumpHideBound !== 'yes') {
      hideBtn.dataset.jumpHideBound = 'yes';
      hideBtn.addEventListener('click', () => setPanelHidden(true));
    }
    if (showBtn && showBtn.dataset.jumpShowBound !== 'yes') {
      showBtn.dataset.jumpShowBound = 'yes';
      showBtn.addEventListener('click', () => {
        setPanelHidden(false);
        setPanelCondensed(false);
      });
    }
    if (hideBtn || showBtn) {
      setPanelHidden(false, { focus: false });
    }

    const update = () => {
      ticking = false;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      if (!viewportHeight) {
        setActive(null);
        return;
      }
      if (manualOverrideId) return;
      const navOffset = getNavOffset();
      const topLimit = Math.min(Math.max(0, navOffset), viewportHeight);
      const bottomLimit = viewportHeight;
      const focusSpan = Math.max(0, bottomLimit - topLimit);

      let best = null;
      let bestRatio = 0;
      let bestDistance = Infinity;

      items.forEach((item) => {
        const rect = item.target.getBoundingClientRect();
        const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
        if (visible <= 0) return;
        const denom = Math.min(rect.height || 0, focusSpan) || 1;
        const ratio = visible / denom;
        const distance = Math.abs(rect.top - topLimit);
        if (ratio > bestRatio || (Math.abs(ratio - bestRatio) < 0.001 && distance < bestDistance)) {
          bestRatio = ratio;
          bestDistance = distance;
          best = item;
        }
      });

      let nextId = best ? best.id : null;

      const doc = document.documentElement;
      const scrollTop = window.scrollY || window.pageYOffset || 0;
      const atBottom = scrollTop + viewportHeight >= (doc.scrollHeight - 2);
      const lastItem = items[items.length - 1];
      if (lastItem && atBottom) {
        const rect = lastItem.target.getBoundingClientRect();
        const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
        if (visible > 0) nextId = lastItem.id;
      }

      if (nextId === activeId) return;
      activeId = nextId;
      setActive(nextId);
    };

    const requestUpdate = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };

    requestUpdate();
    const handleScroll = () => {
      const autoScrolling = isJumpPanelAutoScrolling();
      if (!autoScrolling) {
        clearPanelFocusOnScroll();
        setPanelCondensed(true);
      }
      if (manualOverrideId) {
        manualOverrideId = null;
      }
      requestUpdate();
    };
    const handleResize = () => {
      if (!shouldCondenseOnScroll()) {
        panel.classList.remove('is-condensed');
      }
      requestUpdate();
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    document.addEventListener('navheightchange', requestUpdate);
  }

  function initSpeedDial(){
    if (!document || !document.body || typeof document.createElement !== 'function') return;
    if (document.querySelector('[data-speed-dial]')) return;

    const dial = document.createElement('div');
    if (!dial || typeof dial.setAttribute !== 'function') return;
    const menuId = 'speed-dial-menu';
    dial.className = 'speed-dial';
    dial.setAttribute('data-speed-dial', 'true');
    dial.innerHTML = `
      <div class="speed-dial__tray" data-speed-dial-tray>
        <div class="speed-dial__actions" id="${menuId}" role="menu" aria-label="Contact options" aria-hidden="true" data-speed-dial-menu>
          <div class="speed-dial__item">
            <span class="speed-dial__label" aria-hidden="true">Direct Message</span>
            <a class="speed-dial__action btn-icon speed-dial__action--direct" href="contact.html#contact-modal" data-contact-modal-link="true" aria-label="Send a direct message" role="menuitem" data-speed-dial-action>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>
                <path d="M7 9h10"></path>
                <path d="M7 13h6"></path>
              </svg>
            </a>
          </div>
          <div class="speed-dial__item">
            <span class="speed-dial__label" aria-hidden="true">Send Email</span>
            <a class="speed-dial__action btn-icon" href="mailto:daniel@danielshort.me" aria-label="Send Email" role="menuitem" data-speed-dial-action>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="3" y="5" width="18" height="14" rx="2"></rect>
                <path d="M3 7l9 6 9-6"></path>
              </svg>
            </a>
          </div>
          <div class="speed-dial__item">
            <span class="speed-dial__label" aria-hidden="true">View LinkedIn</span>
            <a class="speed-dial__action btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" rel="noopener noreferrer" aria-label="View LinkedIn" role="menuitem" data-speed-dial-action>
              <svg class="brand-fill" viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="4" cy="4" r="2"></circle>
                <rect x="2" y="9" width="4" height="12" rx="1"></rect>
                <path d="M10 9h3.8v2.1h.1C14.8 9.7 16.1 9 17.9 9c3 0 5.1 1.9 5.1 5.9V21h-4v-5.9c0-1.7-.7-2.9-2.6-2.9s-2.7 1.4-2.7 3V21H10z"></path>
              </svg>
            </a>
          </div>
          <div class="speed-dial__item">
            <span class="speed-dial__label" aria-hidden="true">View GitHub</span>
            <a class="speed-dial__action btn-icon" href="https://github.com/danielshort3" target="_blank" rel="noopener noreferrer" aria-label="View GitHub" role="menuitem" data-speed-dial-action>
              <span class="icon icon-github" aria-hidden="true"></span>
            </a>
          </div>
        </div>
      </div>
      <button class="speed-dial__toggle btn-icon btn-icon-featured" type="button" aria-expanded="false" aria-haspopup="menu" aria-controls="${menuId}" aria-label="Open contact options" data-speed-dial-toggle>
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>
          <path d="M12 8v6"></path>
          <path d="M9 11h6"></path>
        </svg>
      </button>
    `;
    document.body.appendChild(dial);

    const toggle = dial.querySelector('[data-speed-dial-toggle]');
    const menu = dial.querySelector('[data-speed-dial-menu]');
    const actions = [...dial.querySelectorAll('[data-speed-dial-action]')];
    if (!toggle || !menu || !actions.length) return;

    let isLocked = false;
    let suppressHover = false;

    const setExpanded = (expanded) => {
      dial.classList.toggle('is-open', expanded);
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      toggle.setAttribute('aria-label', expanded ? 'Close contact options' : 'Open contact options');
      menu.setAttribute('aria-hidden', expanded ? 'false' : 'true');
      actions.forEach(action => {
        action.tabIndex = expanded ? 0 : -1;
      });
      if (!expanded && menu.contains(document.activeElement)) {
        toggle.focus({ preventScroll: true });
      }
    };

    const closeMenu = () => {
      isLocked = false;
      suppressHover = false;
      setExpanded(false);
    };

    setExpanded(false);

    toggle.addEventListener('click', (event) => {
      event.preventDefault();
      if (isLocked) {
        isLocked = false;
        try {
          if (dial.matches(':hover')) suppressHover = true;
        } catch {}
        setExpanded(false);
        return;
      }
      isLocked = true;
      suppressHover = false;
      setExpanded(true);
    });

    actions.forEach(action => {
      action.addEventListener('click', closeMenu);
    });

    let canHover = false;
    try {
      canHover = Boolean(window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches);
    } catch {}
    if (canHover) {
      dial.addEventListener('pointerenter', () => {
        if (isLocked || suppressHover) return;
        setExpanded(true);
      });
      dial.addEventListener('pointerleave', () => {
        if (isLocked) return;
        suppressHover = false;
        setExpanded(false);
      });
    }

    document.addEventListener('click', (event) => {
      if (!dial.classList.contains('is-open')) return;
      if (dial.contains(event.target)) return;
      closeMenu();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!dial.classList.contains('is-open')) return;
      event.preventDefault();
      closeMenu();
      toggle.focus({ preventScroll: true });
    });
  }

  initSpeedDial();

  // ---- Global modal close handlers (X button and backdrop) ----
  document.addEventListener('click', (e) => {
    // 1) Close when X is clicked
    const closeBtn = e.target.closest('.modal-close');
    if (closeBtn) {
      e.preventDefault();
      const modal = closeBtn.closest('.modal');
      if (modal) {
        const id = modal.id?.replace(/-modal$/, '') || modal.id || 'modal';
        window.closeModal && window.closeModal(id);
      }
      return;
    }
    // 2) Close when clicking the backdrop (outside modal-content)
    const backdrop = e.target.closest('.modal');
    const insideContent = e.target.closest('.modal-content');
    if (backdrop && !insideContent) {
      const id = backdrop.id?.replace(/-modal$/, '') || backdrop.id || 'modal';
      window.closeModal && window.closeModal(id);
    }
  });
})();
