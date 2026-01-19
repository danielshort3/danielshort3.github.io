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
  const CONTACT_MODAL_ID = 'contact-modal';
  const CONTACT_MODAL_SCRIPT = 'js/forms/contact.js';
  const CONTACT_MODAL_MARKUP = `
    <div id="contact-modal" class="modal">
      <div class="modal-content" role="dialog" aria-modal="true" tabindex="0" aria-labelledby="contact-modal-title">
        <button class="modal-close" aria-label="Close dialog">&times;</button>
        <div class="modal-title-strip">
          <h3 class="modal-title" id="contact-modal-title">Send a Message</h3>
        </div>
        <div class="modal-body">
          <form id="contact-form" class="contact-form" method="post" action="https://muee4eg6ze.execute-api.us-east-2.amazonaws.com/prod/contact" data-endpoint="https://muee4eg6ze.execute-api.us-east-2.amazonaws.com/prod/contact" novalidate>
            <div class="form-field">
              <label for="contact-name">Name <span class="field-required" id="contact-name-required" hidden>- Required</span></label>
              <input id="contact-name" name="name" type="text" autocomplete="name" required maxlength="200" placeholder="Jane Doe" aria-describedby="contact-name-required">
            </div>
            <div class="form-field">
              <label for="contact-email">Email <span class="field-required" id="contact-email-required" hidden>- Required</span></label>
              <input id="contact-email" name="email" type="email" autocomplete="email" required placeholder="you@example.com" aria-describedby="contact-email-required">
            </div>
            <div class="form-field">
              <label for="contact-message">How can I help? <span class="field-required" id="contact-message-required" hidden>- Required</span></label>
              <textarea id="contact-message" name="message" rows="5" maxlength="4000" required placeholder="Share a few details about your project, role, or opportunity." aria-describedby="contact-message-required"></textarea>
            </div>
            <div class="form-field honeypot" aria-hidden="true">
              <label for="contact-company">Company</label>
              <input id="contact-company" name="company" type="text" tabindex="-1" autocomplete="off">
            </div>
            <p id="contact-status" class="contact-form-status" role="status" aria-live="polite" tabindex="-1"></p>
            <div id="contact-alt" class="contact-form-alt" hidden>
              <a href="mailto:daniel@danielshort.me" class="btn-ghost">Email me directly</a>
            </div>
            <div class="form-actions">
              <button type="submit" class="btn-primary">
                <span class="btn-spinner" aria-hidden="true"></span>
                <span class="btn-label">Send Message</span>
              </button>
              <button type="button" class="btn-secondary" data-contact-close>Close</button>
              <button type="button" class="btn-ghost" data-contact-reset>Clear form</button>
            </div>
          </form>
          <div class="contact-form-success" id="contact-success" hidden tabindex="-1" role="status" aria-live="polite">
            <span class="success-icon" aria-hidden="true"></span>
            <h4>Message sent</h4>
            <p>Thanks for reaching out. I received your note and will reply shortly. If it&rsquo;s urgent, feel free to send a direct email as well.</p>
            <div class="form-actions">
              <button type="button" class="btn-primary" data-contact-new>Start another message</button>
              <a href="mailto:daniel@danielshort.me" class="btn-secondary">Email me directly</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

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
    initSmoothScrollLinks();
    if (isPage('portfolio')) {
      ensurePortfolioScripts().then(() => {
        run(window.buildPortfolioCarousel);
        run(window.buildPortfolio);
        run(window.initSeeMore);
      }).catch(err => console.warn('Failed to initialize portfolio page', err));
    }
    if (isPage('home')) {
      initSkillPopups();
      initJumpPanelSpy();
    }
    if (isPage('project')) {
      initProjectDemoTabs();
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

  const ensureContactModal = () => {
    if (!document || !document.body || typeof document.createElement !== 'function') return null;
    const existing = document.getElementById(CONTACT_MODAL_ID);
    if (existing) return existing;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = CONTACT_MODAL_MARKUP.trim();
    const modal = wrapper.firstElementChild;
    if (!modal) return null;
    document.body.appendChild(modal);
    return modal;
  };

  const ensureContactScript = () => {
    const promise = loadScriptOnce(CONTACT_MODAL_SCRIPT);
    if (promise && typeof promise.catch === 'function') {
      promise.catch(err => console.warn('Failed to load contact form script', err));
    }
    return promise;
  };

  const applyContactPrefill = (payload) => {
    if (!payload) return;
    const nameField = document.getElementById('contact-name');
    const emailField = document.getElementById('contact-email');
    const messageField = document.getElementById('contact-message');
    if (nameField && payload.name) nameField.value = payload.name;
    if (emailField && payload.email) emailField.value = payload.email;
    if (messageField && payload.message) messageField.value = payload.message;
  };

  const requestContactModal = (payload) => {
    storeContactOrigin();
    const ensured = document.getElementById(CONTACT_MODAL_ID) || ensureContactModal();
    if (!ensured) return;
    const open = () => {
      if (typeof window.openContactModal === 'function') {
        window.openContactModal();
        applyContactPrefill(payload);
        return;
      }
      applyContactPrefill(payload);
      try {
        if (location.hash !== `#${CONTACT_MODAL_ID}`) {
          location.hash = `#${CONTACT_MODAL_ID}`;
        }
      } catch {}
    };
    if (window.__contactModalReady) {
      open();
      return;
    }
    const scriptPromise = ensureContactScript();
    if (scriptPromise && typeof scriptPromise.then === 'function') {
      scriptPromise.then(open);
    } else {
      open();
    }
  };

  window.requestContactModal = requestContactModal;

  document.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-contact-modal-link]');
    if (!trigger) return;
    event.preventDefault();
    event.__contactHandled = true;
    requestContactModal();
  });

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
      if (!p || p.published === false) return;
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
	    const buttons = $$('[data-project-modal="true"]');
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

      const openAncestorDetails = (node) => {
        let current = node;
        while (current && current.closest) {
          const details = current.closest('details');
          if (!details) break;
          if (!details.open) details.open = true;
          current = details.parentElement;
        }
      };
      openAncestorDetails(target);

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

  function initProjectDemoTabs() {
    const shells = $$('[data-demo-tabs="true"]');
    if (!shells.length) return;

    shells.forEach((shell) => {
      const tabs = $$('[role="tab"]', shell);
      const panels = $$('[role="tabpanel"]', shell);
      if (tabs.length < 2 || panels.length < 2) return;

      const panelById = new Map(
        panels
          .map((panel) => [panel.id, panel])
          .filter(([id]) => Boolean(id))
      );

      const getPanelForTab = (tab) => {
        if (!tab) return null;
        const panelId = tab.getAttribute('aria-controls');
        if (!panelId) return null;
        return panelById.get(panelId) || document.getElementById(panelId);
      };

      const loadPanelIframes = (panel) => {
        if (!panel) return;
        $$('iframe[data-src]', panel).forEach((iframe) => {
          if (!iframe || iframe.getAttribute('src')) return;
          const dataSrc = iframe.getAttribute('data-src');
          if (!dataSrc) return;
          iframe.setAttribute('src', dataSrc);
          iframe.removeAttribute('data-src');
        });
      };

      const setActiveTab = (nextTab, { focus = false } = {}) => {
        if (!nextTab) return;
        const nextPanel = getPanelForTab(nextTab);
        if (!nextPanel) return;

        tabs.forEach((tab) => {
          const active = tab === nextTab;
          tab.classList.toggle('is-active', active);
          tab.setAttribute('aria-selected', String(active));
          if (active) {
            tab.removeAttribute('tabindex');
          } else {
            tab.setAttribute('tabindex', '-1');
          }
        });

        panels.forEach((panel) => {
          const active = panel === nextPanel;
          panel.classList.toggle('is-active', active);
          if (active) {
            panel.removeAttribute('hidden');
          } else {
            panel.setAttribute('hidden', '');
          }
        });

        loadPanelIframes(nextPanel);
        if (focus) nextTab.focus();
      };

      tabs.forEach((tab) => {
        on(tab, 'click', () => setActiveTab(tab));
        on(tab, 'keydown', (event) => {
          const key = event?.key;
          if (!key) return;
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return;
          event.preventDefault();
          const currentIndex = tabs.indexOf(tab);
          if (currentIndex === -1) return;
          const lastIndex = tabs.length - 1;
          const nextIndex = (() => {
            if (key === 'Home') return 0;
            if (key === 'End') return lastIndex;
            if (key === 'ArrowLeft') return currentIndex === 0 ? lastIndex : currentIndex - 1;
            return currentIndex === lastIndex ? 0 : currentIndex + 1;
          })();
          const nextTab = tabs[nextIndex];
          setActiveTab(nextTab, { focus: true });
        });
      });

      on(shell, 'click', (event) => {
        const trigger = event.target.closest('[data-demo-tabs-open="demo"]');
        if (!trigger) return;
        const demoTab = tabs.find((tab) => tab.textContent.trim().toLowerCase() === 'demo') || tabs[1];
        setActiveTab(demoTab, { focus: true });
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
            <a class="speed-dial__action btn-icon speed-dial__action--direct" href="#contact-modal" data-contact-modal-link="true" aria-label="Send a direct message" role="menuitem" data-speed-dial-action>
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

  function initCookieSettingsButton(){
    if (!document || !document.body || typeof document.createElement !== 'function') return;
    try {
      if (window.self !== window.top) return;
    } catch {
      return;
    }
    if (document.getElementById('privacy-settings-link')) return;
    if (document.querySelector('[data-cookie-settings]')) return;

    const host = document.createElement('div');
    if (!host || typeof host.setAttribute !== 'function') return;
    host.className = 'cookie-settings';
    host.setAttribute('data-cookie-settings', 'true');
    host.innerHTML = `
      <div class="cookie-settings__item">
        <span class="cookie-settings__label" aria-hidden="true">Cookie settings</span>
        <button id="privacy-settings-link" type="button" class="cookie-settings__toggle btn-icon btn-icon-featured" aria-label="Cookie settings" aria-haspopup="dialog">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M21 13a4 4 0 0 1-4-4a4 4 0 0 1-4-4A9 9 0 1 0 21 13z"></path>
            <circle cx="10" cy="10" r="1" fill="currentColor" stroke="none"></circle>
            <circle cx="13" cy="13" r="1" fill="currentColor" stroke="none"></circle>
            <circle cx="9" cy="15.5" r="1" fill="currentColor" stroke="none"></circle>
          </svg>
        </button>
      </div>
    `;
    document.body.appendChild(host);
  }

  initCookieSettingsButton();
  initSpeedDial();

  // ---- Global modal close handlers (X button and backdrop) ----
  document.addEventListener('click', (e) => {
    // 1) Close when X is clicked
    const closeBtn = e.target.closest('.modal-close');
    if (closeBtn) {
      const modal = closeBtn.closest('.modal');
      if (modal && modal.id === CONTACT_MODAL_ID) return;
      e.preventDefault();
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
      if (backdrop.id === CONTACT_MODAL_ID) return;
      const id = backdrop.id?.replace(/-modal$/, '') || backdrop.id || 'modal';
      window.closeModal && window.closeModal(id);
    }
  });

  const setProjectEmbedIframeHeight = (ifr, height) => {
    if (!ifr) return;
    if (!Number.isFinite(height) || height <= 0) return;
    const next = `${Math.floor(height)}px`;
    if (ifr.style.height === next) return;
    ifr.style.height = next;
  };

  const resizeProjectEmbedIframe = (ifr) => {
    if (!ifr) return;
    try {
      const doc = ifr.contentDocument || ifr.contentWindow?.document;
      if (!doc) return;
      const body = doc.body;
      const docEl = doc.documentElement;
      const height = Math.max(
        body ? body.scrollHeight : 0,
        body ? body.offsetHeight : 0,
        docEl ? docEl.clientHeight : 0,
        docEl ? docEl.scrollHeight : 0,
        docEl ? docEl.offsetHeight : 0
      );
      setProjectEmbedIframeHeight(ifr, height);
    } catch {}
  };

  const observeProjectEmbedIframe = (ifr) => {
    if (!ifr) return;
    if (typeof ResizeObserver !== 'function') return;

    try {
      if (ifr._projectEmbedResizeObserver) {
        ifr._projectEmbedResizeObserver.disconnect();
      }
    } catch {}
    ifr._projectEmbedResizeObserver = null;

    let doc = null;
    try {
      doc = ifr.contentDocument || ifr.contentWindow?.document;
    } catch {}
    if (!doc) return;

    const body = doc.body;
    const docEl = doc.documentElement;
    if (!body && !docEl) return;

    const scheduleResize = () => {
      if (ifr._projectEmbedResizeScheduled) return;
      ifr._projectEmbedResizeScheduled = true;
      requestAnimationFrame(() => {
        ifr._projectEmbedResizeScheduled = false;
        resizeProjectEmbedIframe(ifr);
      });
    };

    const ro = new ResizeObserver(scheduleResize);
    try { if (docEl) ro.observe(docEl); } catch {}
    try { if (body) ro.observe(body); } catch {}
    ifr._projectEmbedResizeObserver = ro;
    scheduleResize();
  };

  const bindProjectEmbedResize = () => {
    document.querySelectorAll('.project-embed-frame').forEach((ifr) => {
      if (ifr._resizeBound) return;
      ifr._resizeBound = true;
      ifr.addEventListener('load', () => {
        resizeProjectEmbedIframe(ifr);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 50);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 350);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 1000);
        observeProjectEmbedIframe(ifr);
      });
      resizeProjectEmbedIframe(ifr);
      observeProjectEmbedIframe(ifr);
    });
  };

  document.addEventListener('DOMContentLoaded', bindProjectEmbedResize);
  window.addEventListener('load', bindProjectEmbedResize);
  window.addEventListener('message', (event) => {
    const data = event && event.data || {};
    const type = typeof data?.type === 'string' ? data.type : '';
    if (!/(pizza-demo-resize|retail-loss-sales-demo-resize|minesweeper-demo-resize)/.test(type)) return;
    const ifrs = document.querySelectorAll('.project-embed-frame');
    for (const ifr of ifrs) {
      if (ifr.contentWindow === event.source) {
        const h = typeof data.height === 'number' && isFinite(data.height)
          ? Math.max(0, Math.floor(data.height))
          : null;
        if (h) {
          ifr.style.height = `${h}px`;
        } else {
          resizeProjectEmbedIframe(ifr);
        }
        break;
      }
    }
  });
})();
