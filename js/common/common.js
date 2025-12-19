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

  const loadedScripts = new Map();
  let portfolioBundle = null;
  let modalsPromise = null;
  let modalsHydrated = false;

  document.addEventListener('DOMContentLoaded', () => {
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
          return decodeURIComponent(location.hash.slice(1));
        }
      } catch {
        if (location.hash && location.hash.length > 1) return location.hash.slice(1);
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

    links.forEach((link) => {
      if (link.dataset.smoothBound === 'yes') return;
      link.dataset.smoothBound = 'yes';
      on(link, 'click', (evt) => {
        if (prefersReducedMotion()) return;
        if (evt && (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey)) return;
        if (evt && typeof evt.button === 'number' && evt.button !== 0) return;

        const href = link.getAttribute('href') || '';
        if (!href.startsWith('#') || href.length < 2) return;
        const targetId = decodeURIComponent(href.slice(1));
        const target = document.getElementById(targetId);
        if (!target) return;

        evt.preventDefault();
        try {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch {
          target.scrollIntoView();
        }
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
    if (!links.length) return;
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

    items.forEach((item) => bindPointerBlur(item.link));

    const clearPanelFocus = (event) => {
      if (event?.pointerType && event.pointerType !== 'mouse') return;
      const active = document.activeElement;
      if (active && panel.contains(active)) active.blur();
    };
    panel.addEventListener('pointerleave', clearPanelFocus);
    panel.addEventListener('mouseleave', clearPanelFocus);

    const update = () => {
      ticking = false;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      if (!viewportHeight) {
        setActive(null);
        return;
      }
      const navOffset = getNavOffset();
      const topLimit = Math.min(Math.max(0, navOffset), viewportHeight);
      const bottomLimit = viewportHeight;
      const focusSpan = Math.max(0, bottomLimit - topLimit);

      let best = null;
      let bestVisible = 0;
      let bestRect = null;

      items.forEach((item) => {
        const rect = item.target.getBoundingClientRect();
        const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
        if (visible > bestVisible) {
          bestVisible = visible;
          best = item;
          bestRect = rect;
        }
      });

      let nextId = null;
      if (best && bestRect && bestVisible > 0) {
        const minVisible = Math.min(focusSpan * 0.18, bestRect.height * 0.35);
        if (bestVisible >= minVisible) {
          nextId = best.id;
        }
      }

      const doc = document.documentElement;
      const scrollTop = window.scrollY || window.pageYOffset || 0;
      const atBottom = scrollTop + viewportHeight >= (doc.scrollHeight - 2);
      if (!nextId && atBottom) {
        const lastItem = items[items.length - 1];
        if (lastItem) {
          const rect = lastItem.target.getBoundingClientRect();
          const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
          if (visible > 0) nextId = lastItem.id;
        }
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
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate);
    window.addEventListener('orientationchange', requestUpdate);
    document.addEventListener('navheightchange', requestUpdate);
  }

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
