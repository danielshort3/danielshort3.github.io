/* ===================================================================
   File: common.js
   Purpose: Core helpers and skill popups
=================================================================== */
/* common.js • site-wide UX helpers (v2 • lean core)
   ▸ Everything portfolio-specific now lives in portfolio.js
   ─────────────────────────────────────────────────────────── */
(() => {
  "use strict";

  /* shorthand helpers */
  const   $ = (sel,  ctx = document) => ctx.querySelector(sel);
  const  $$ = (sel,  ctx = document) => [...ctx.querySelectorAll(sel)];
  const on  = (node, evt, fn, opt)  => node && node.addEventListener(evt, fn, opt);
  const run = fn  => typeof fn === "function" && fn();

  /* utility – match <body data-page=""> against strings */
  const isPage = (...names) => names.includes(document.body.dataset.page);

  /* ───────────────────────── DOM READY BOOTSTRAP ─────────────────────── */
  document.addEventListener("DOMContentLoaded", () => {
    if (isPage("home")) run(window.buildFeaturedCarousel);  // optional helper

    
    if (isPage("portfolio")) {
      run(window.buildPortfolioCarousel);
      run(window.buildPortfolio);
      run(initSeeMore);
    }
    if (isPage("home"))      run(initSkillPopups);      // ← new line

  });

    /* ╭────────── Home-page skill pop-ups (no duplication) ─────────╮ */
    function initSkillPopups(){
      if (!document.body.dataset.page?.includes("home")) return;

      const modalsRoot = $("#modals") || (()=> {
        const d = document.createElement("div");
        d.id="modals"; document.body.appendChild(d); return d;
      })();

      /* build modals once using the shared portfolio helper */
      window.PROJECTS.forEach(p=>{
        if ($("#"+p.id+"-modal")) return;          // already exists
        const m = document.createElement("div");
        m.className="modal"; m.id=`${p.id}-modal`;
        m.innerHTML = window.generateProjectModal(p);
        modalsRoot.appendChild(m);
      });

      /* click tile → open */
      $$(".skill-link").forEach(btn=>{
        on(btn,"click",e=>{
          e.preventDefault();
          openModal(btn.dataset.project);
        });
      });

      if (location.hash) openModal(location.hash.slice(1));
    }

    /* call it during DOMContentLoaded */
    document.addEventListener("DOMContentLoaded", ()=> {
      if (window.buildPortfolio) run(initSkillPopups);
    });

})();
