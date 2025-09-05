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

  document.addEventListener('DOMContentLoaded', () => {
    if (isPage('home')) run(window.buildFeaturedCarousel);
    if (isPage('portfolio')) {
      run(window.buildPortfolioCarousel);
      run(window.buildPortfolio);
      run(initSeeMore);
    }
    if (isPage('home')) run(initSkillPopups);
  });

  function initSkillPopups(){
    if (!document.body.dataset.page?.includes('home')) return;
    const modalsRoot = $('#modals') || (()=>{ const d=document.createElement('div'); d.id='modals'; document.body.appendChild(d); return d;})();
    window.PROJECTS.forEach(p=>{
      if ($('#'+p.id+'-modal')) return;
      const m=document.createElement('div');
      m.className='modal';
      m.id=`${p.id}-modal`;
      m.innerHTML=window.generateProjectModal(p);
      modalsRoot.appendChild(m);
    });
    $$('.skill-link').forEach(btn => {
      // Make focusable
      btn.setAttribute('tabindex','0');
      // Activate on Enter/Space
      btn.addEventListener('keydown', ev => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          openModal(btn.dataset.project);
        }
      });
      // Click still works
      on(btn,'click', e => {
        e.preventDefault();
        openModal(btn.dataset.project);
      });
    });
    try {
      const qs = (location.search || '').replace(/^\?/, '');
      const pairs = qs ? qs.split('&') : [];
      let id = null;
      for (const kv of pairs) {
        const [k,v] = kv.split('=');
        if (decodeURIComponent(k) === 'project' && v) { id = decodeURIComponent(v); break; }
      }
      if (!id && location.hash && location.hash.length > 1) id = decodeURIComponent(location.hash.slice(1));
      if (id) openModal(id);
    } catch {
      if (location.hash && location.hash.length > 1) openModal(location.hash.slice(1));
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    if (window.buildPortfolio) run(initSkillPopups);
  });

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
