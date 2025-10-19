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
  const currentScript = document.currentScript || document.querySelector('script[src*="common"]');
  const scriptSrc = currentScript ? currentScript.getAttribute('src') || '' : '';
  const assetVariant = /\/dist\/js\//.test(scriptSrc) ? 'dist/js/' : 'js/';
  const resolveAsset = (src) => {
    if (!src) return src;
    if (/^(?:https?:)?\/\//.test(src)) return src;
    if (src.startsWith('/')) return src;
    if (assetVariant === 'dist/js/') return src.replace(/^js\//, 'dist/js/');
    return src;
  };

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
    if (isPage('home')) initSkillPopups();
  });

  function loadScriptOnce(src){
    const finalSrc = resolveAsset(src);
    if (loadedScripts.has(src)) return loadedScripts.get(src);
    const promise = new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.src = finalSrc;
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
    const sources = ['js/portfolio/projects-data.js','js/portfolio/modal-helpers.js','js/portfolio/portfolio.js'];
    const chain = sources
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
    const buttons = $$('.skill-link');
    if (!buttons.length) return;

    const safePreload = () => {
      const promise = preparePortfolioModals();
      if (promise && typeof promise.then === 'function') promise.catch(() => {});
    };

    buttons.forEach(btn => {
      if (btn.dataset.modalBound === 'yes') return;
      btn.dataset.modalBound = 'yes';
      on(btn, 'pointerenter', safePreload, { once: true });
      on(btn, 'focus', safePreload, { once: true });
      on(btn, 'touchstart', safePreload, { once: true, passive: true });
      on(btn, 'click', () => {
        if (!btn.dataset.project) return;
        const promise = openProjectModal(btn.dataset.project);
        if (promise && typeof promise.catch === 'function') promise.catch(() => {});
      });
      on(btn, 'keydown', (evt) => {
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
