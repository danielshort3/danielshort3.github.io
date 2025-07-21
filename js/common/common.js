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
    $$('.skill-link').forEach(btn=>{
      btn.setAttribute('tabindex','0');
      on(btn,'click',e=>{e.preventDefault();openModal(btn.dataset.project);});
      on(btn,'keydown',e=>{
        if(e.key==='Enter'||e.key===' '){
          e.preventDefault();
          openModal(btn.dataset.project);
        }
      });
    });
    if(location.hash) openModal(location.hash.slice(1));
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    if (window.buildPortfolio) run(initSkillPopups);
  });
})();
