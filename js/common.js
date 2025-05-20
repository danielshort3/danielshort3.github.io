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
    injectNav();
    injectFooter();
    if (isPage("home")) run(window.buildFeaturedCarousel);  // optional helper

    initReveal();
    setScrollbarVar();
    initChevronHint();
    initCertTicker();

    if (isPage("portfolio")) {
      run(window.buildPortfolioCarousel);
      run(window.buildPortfolio);
      run(window.initSeeMore);
    }
    if (isPage("home"))      run(initSkillPopups);      // ← new line

  });

  /* ╭───────────────────── REVEAL-ON-SCROLL ────────────────────╮ */
  function initReveal(){
    const io = new IntersectionObserver(
      (ents, o) => ents.forEach(e=>{
        if (e.isIntersecting){ e.target.classList.add("active"); o.unobserve(e.target); }
      }),
      { threshold:.15 }
    );
    $$(".reveal:not(.no-reveal)").forEach(el => io.observe(el));
  }

  /* ╭────────── compensate for native scrollbar width ──────────╮ */
  const setScrollbarVar = () => {
    const sb = window.innerWidth - document.documentElement.clientWidth;
    document.documentElement.style.setProperty("--scrollbar", `${sb}px`);
  };

  /* ╭──────────────────── SCROLL HINT CHEVRON ──────────────────╮ */
  function initChevronHint(){
    const chev = $(".chevron-hint"), hero = $(".hero");
    if (!chev || !hero) return;

    const io = new IntersectionObserver(ents => {
      chev.classList.toggle("hide", !ents[0].isIntersecting);
    });
    io.observe(hero);
  }

  /* ╭──────────────────── CERTIFICATION TICKER ─────────────────╮ */
    function initCertTicker () {
      const track = document.querySelector(".cert-track");
      if (!track) return;

      /* constants + state -------------------------------------------------- */
      const GAP   = 160,          // space between tiles
            BASE  =  90,          // auto-scroll px/s
            DRAG  =  15;          // <- bigger: ignore finger jitter

      let v = BASE, target = BASE;    // velocity, velocity target
      let bandW, stripW = 0;

      let down = false, moved = false;
      let sx   = 0,     lx    = 0;
      let paused = false;            // <- NEW: true only while dragging
      let cancelClk = false;

      const originals = [...track.children];
      const tiles     = [...originals];

      const setPos = (el,x) => {
        el.dataset.x = x;
        el.style.transform = `translateX(${x}px)`;
      };

    /* ----- lay out + clone --------------------------------------------- */
    const fill = () => {
      /* wipe old clones */
      tiles.slice(originals.length).forEach(el => el.remove());
      tiles.length = originals.length;
      stripW = 0;

      /* lay out originals */
      originals.forEach((t, i) => {
        const w = t.offsetWidth + GAP;
        Object.assign(t.dataset, { w, orig: i });
        setPos(t, stripW);
        stripW += w;
      });

      const baseW = stripW;                          // ① width of originals
      bandW = track.getBoundingClientRect().width;   // ② viewport width

      /* clone whole rows until first clone is ≥ 1 viewport away --------- */
      while (stripW < baseW + bandW) {               // ③ new stop-condition
        originals.forEach(src => {
          const clone = src.cloneNode(true);
          track.appendChild(clone);                  // must be in DOM first
          const w = clone.offsetWidth + GAP;
          Object.assign(clone.dataset, { w, orig: +src.dataset.orig });
          setPos(clone, stripW);
          stripW += w;
          tiles.push(clone);
        });
      }
    };

      /* wait for images, then build strip; rebuild on resize */
      window.addEventListener("load", fill);
      window.addEventListener("resize", fill);

        /* hover pause – desktop only ------------------------------------ */
        const isTouchDevice = matchMedia("(hover: none) and (pointer: coarse)").matches;
        //   ↳ modern, reliable test for phones & tablets

        if (!isTouchDevice){                 // skip on mobile
          track.addEventListener("mouseenter", () => (target = 0));
          track.addEventListener("mouseleave", () => (target = BASE));
}

      /* drag (touch / mouse) ------------------------------------- */
      const move = dx =>
        tiles.forEach(t=>{
          let x = +t.dataset.x + dx,
              w = +t.dataset.w;
          x = dx<0
              ? (x + w <= 0 ? x + stripW : x)
              : (x >= bandW ? x - stripW : x);
          setPos(t,x);
        });

      const endDrag = () => {
        down = moved = false;
        paused = false;                     // resume auto-scroll
        track.classList.remove("dragging");
        ["pointermove","pointerup","pointercancel"]
          .forEach(e=>window.removeEventListener(e,endDrag));
      };

      track.addEventListener("pointerdown", e=>{
        if (e.button) return;
        down = true;  moved = false;  paused = false;
        sx = lx = e.clientX;

        const onMove = e=>{
          if (!down) return;
          const dxT = e.clientX - sx;
          if (!moved && Math.abs(dxT) >= DRAG){
            moved = true;
            paused = true;                  // pause only while dragging
            track.classList.add("dragging");
          }
          if (moved){
            move(e.clientX - lx);
            lx = e.clientX;
            e.preventDefault();
          }
        };

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup",   endDrag, {once:true});
        window.addEventListener("pointercancel", endDrag, {once:true});
      });

      /* suppress bogus clicks that started as drags */
      track.addEventListener("click", e=>{
        if (cancelClk){ e.preventDefault(); e.stopImmediatePropagation(); }
        cancelClk = false;
      }, true);

      /* auto-scroll ---------------------------------------------- */
        let last = performance.now();

        /* NEW ▸ if the tab was hidden, restart the timer */
        const resetClock = () => { last = performance.now(); };
        document.addEventListener("visibilitychange", resetClock);
        window.addEventListener("focus", resetClock);        // Safari/iOS fallback

        (function loop(now){
          /* NEW ▸ clamp dt so one frame can’t jump the strip off-screen */
          const dt = Math.min( (now - last) / 1000, 0.25 );  // max 0.25 s
          last = now;

          v += (target - v) * Math.min(1, dt * 4);

          if (!paused){
            tiles.forEach(t=>{
              let x = +t.dataset.x - v*dt,
                  w = +t.dataset.w;
              if (x < -w)     x += stripW;   // single modulo is fine now
              setPos(t, x);
            });
          }
          requestAnimationFrame(loop);
        })(last);
    }

  /* ╭──────────────────────── NAV & FOOTER ─────────────────────╮ */
  function injectNav(){
    const host=$("#combined-header-nav");
    if(!host) return;

    const animate = !sessionStorage.getItem("navEntryPlayed");
    sessionStorage.setItem("navEntryPlayed","yes");

    host.innerHTML=`
      <nav class="nav ${animate?"animate-entry":""}">
        <div class="wrapper">
          <a href="index.html" class="brand">
            <img src="images/logo.png" alt="DS logo" class="brand-logo">

            <!-- split the title into three spans we can style independently -->
            <span class="brand-name">
              <span class="brand-line name">Daniel Short</span>
              <span class="brand-line divider">│</span>
              <span class="brand-line tagline">Data & Insights</span>
            </span>
          </a>

          <!-- new ↓ button lives beside the logo -->
          <button id="nav-toggle"
                  class="burger"
                  aria-label="Toggle navigation"
                  aria-expanded="false">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </button>

          <!-- unchanged links – give wrapper a data-attr so we can slide it -->
          <nav class="nav-row" data-collapsible>
            <a href="index.html"           class="btn-secondary nav-link">Home</a>
            <a href="portfolio.html"       class="btn-secondary nav-link">Portfolio</a>
            <a href="contributions.html"   class="btn-secondary nav-link">Contributions</a>
            <a href="contact.html"         class="btn-secondary nav-link">Contact</a>
            <a href="documents/Resume.pdf" class="btn-secondary nav-link" target="_blank" download>Resume</a>
          </nav>
        </div>
      </nav>`;

    /* highlight current page */
    const cur = location.pathname.split("/").pop() || "index.html";
    $$(".nav-link").forEach(l=>{
      if (l.getAttribute("href")===cur){
        l.classList.replace("btn-secondary","btn-primary");
        l.setAttribute("aria-current","page");
      }
    });

      /* hamburger behaviour */
    const burger = host.querySelector("#nav-toggle");
    const menu   = host.querySelector(".nav-row");

    if (burger && menu){
      burger.addEventListener("click", () => {

        /* ①  Find the **visible** fixed header bar the burger sits in  */
        const headerBar = burger.closest(".nav") || host;

        /* ②  Measure its distance from the top of the viewport         */
        const headerBottom = headerBar.getBoundingClientRect().bottom;

        /* ③  Position the drawer right below it                        */
        menu.style.top = `${headerBottom}px`;

        /* ④  DEBUG so you can verify the number                        */
        console.log("[NAV] header bottom =", headerBottom, "→ drawer top =", menu.style.top);

        /* ⑤  Toggle drawer open / closed                               */
        const open = menu.classList.toggle("open");
        burger.setAttribute("aria-expanded", open);
        document.body.classList.toggle("menu-open", open);
      });
    }
  }

  function injectFooter(){
    const f=$("footer");
    if(!f) return;
    f.classList.add("footer");
    f.innerHTML=`
      <div class="social">
        <a class="btn-icon" href="mailto:danielshort3@gmail.com" aria-label="Email"><i class="fas fa-envelope"></i></a>
        <a class="btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
        <a class="btn-icon" href="https://github.com/danielshort3" target="_blank" aria-label="GitHub"><i class="fab fa-github"></i></a>
      </div>
      <p>© 2025 Daniel Short. All rights reserved.</p>`;
  }


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
