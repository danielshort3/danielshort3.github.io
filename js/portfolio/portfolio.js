/* portfolio.js - Build portfolio UI components. Project data now lives in projects-data.js */

// ---- Modal focus management (shared) ---------------------------------
let __modalPrevFocus = null;
function trapFocus(modalEl){
  const focusables = modalEl.querySelectorAll('a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])');
  if (!focusables.length) return;
  const first = focusables[0], last = focusables[focusables.length - 1];
  modalEl.addEventListener('keydown', modalEl._trap = (e) => {
    if (e.key !== 'Tab') return;
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
}
function untrapFocus(modalEl){
  if (modalEl._trap) modalEl.removeEventListener('keydown', modalEl._trap);
}

// Ensure a global close helper exists
if (typeof window.closeModal !== 'function') {
  window.closeModal = function(id){
    const modal = document.getElementById(`${id}-modal`) || document.getElementById(id);
    if (!modal) return;
    modal.classList.remove('open');
    untrapFocus(modal);
    if (__modalPrevFocus) { __modalPrevFocus.focus(); __modalPrevFocus = null; }
    window.trackModalClose && window.trackModalClose(id);
  };
}

// Ensure a global open helper exists
if (typeof window.openModal !== 'function') {
  window.openModal = function(id){
    const modal = document.getElementById(`${id}-modal`) || document.getElementById(id);
    if (!modal) return;
    __modalPrevFocus = document.activeElement;
    modal.classList.add('open');
    const content = modal.querySelector('.modal-content') || modal;
    content.focus({preventScroll:true});
    trapFocus(content);
  };
}

// Close on ESC for any open modal
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const open = document.querySelector('.modal.open');
    if (open) {
      const id = open.id?.replace('-modal','') || 'modal';
      window.closeModal(id);
    }
  }
});

window.generateProjectModal = function (p) {
  const isTableau = p.embed?.type === "tableau";
  const isIframe  = p.embed?.type === "iframe";

  /* helper – which Tableau layout should load right now? */
  const tableauDevice = () =>
    window.matchMedia("(max-width:768px)").matches ? "phone" : "desktop";

  /* build the right-hand visual (image, iframe, or Tableau) */
  const visual = (() => {
    if (isIframe) {
      return `
        <div class="modal-embed">
          <iframe src="${p.embed.url}" loading="lazy"></iframe>
        </div>`;
    }

    if (!isTableau) {
      return `
        <div class="modal-image">
          <img src="${p.image}" alt="${p.title}">
        </div>`;
    }

    /* use the clean, param-free URL you stored as p.embed.base */
    const base = p.embed.base || p.embed.url;   // fall back if needed
    const src  = `${base}?${[
      ":embed=y",
      ":showVizHome=no",
      `:device=${tableauDevice()}`
    ].join("&")}`;

    return `
      <div class="modal-embed tableau-fit">
        <iframe
          src="${src}"
          loading="lazy"
          allowfullscreen
          data-base="${base}"></iframe>
      </div>`;
  })();

  /* full modal template ------------------------------------------------ */
  return `
    <div class="modal-content" role="dialog" aria-modal="true" tabindex="0">
      <button class="modal-close" aria-label="Close dialog">&times;</button>
      <div class="modal-title-strip"><h3 class="modal-title">${p.title}</h3></div>

      <div class="modal-body ${isTableau ? "stacked" : ""}">
        <div class="modal-header-details">
          <div class="modal-half">
            <p class="header-label">Tools</p>
            <div class="tool-badges">
              ${p.tools.map(t => `<span class="badge">${t}</span>`).join("")}
            </div>
          </div>
          <div class="modal-divider" aria-hidden="true"></div>
          <div class="modal-half">
            <p class="header-label">Downloads / Links</p>
            <div class="icon-row">
              ${p.resources.map(r => `
                <a href="${r.url}" target="_blank" rel="noopener" title="${r.label}">
                  <img src="${r.icon}" alt="${r.label}" class="icon">
                </a>`).join("")}
            </div>
          </div>
        </div>

        <div class="modal-text">
          <p class="modal-subtitle">${p.subtitle}</p>
          <h4>Problem</h4><p>${p.problem}</p>
          <h4>Action</h4><ul>${p.actions.map(a => `<li>${a}</li>`).join("")}</ul>
          <h4>Result</h4><ul>${p.results.map(r => `<li>${r}</li>`).join("")}</ul>
        </div>

        ${visual}
      </div>
    </div>`;
};

/* ────────────────────────────────────────────────────────────
   Portfolio Carousel (top of page) – no wrap-around version
   ------------------------------------------------------------------ */
function buildPortfolioCarousel() {
  const container = document.getElementById("portfolio-carousel");
  if (!container || !window.PROJECTS) return;

  const track = container.querySelector(".carousel-track");
  const dots  = container.querySelector(".carousel-dots");

  // 1‒5 featured projects -------------------------------------------------
  let projects = [];
  if (Array.isArray(window.FEATURED_IDS) && window.FEATURED_IDS.length) {
    window.FEATURED_IDS.forEach(id => {
      const p = window.PROJECTS.find(pr => pr.id === id);
      if (p) projects.push(p);
    });
  } else {
    projects = window.PROJECTS.slice(0, 5);
  }

  track.innerHTML = "";
  dots.innerHTML  = "";

  projects.forEach((p, i) => {
    /* slide */
    const card = document.createElement("div");
    card.className = "project-card carousel-card";
    card.innerHTML = `
      <div class="overlay"></div>
      <div class="project-title">${p.title}</div>
      <div class="project-subtitle">${p.subtitle}</div>
      <img src="${p.image}" alt="${p.title}">
    `;
    card.addEventListener("click", () => { if (!moved) openModal(p.id); });
    track.appendChild(card);

    /* nav dot */
    const dot = document.createElement("button");
    dot.className = "carousel-dot";
    dot.type  = "button";
    dot.setAttribute("aria-label", `Show ${p.title}`);
    dot.addEventListener("click", () => { goTo(i); });
    dots.appendChild(dot);
  });

  // -----------------------------------------------------------------------
  let current    = 0;
  let pause      = false;
  const AUTO_MS  = 3000;
  let autoTimer  = null;

  const update = () => {
    const cardW  = track.children[0].offsetWidth + 24;            // 24 = gap
    const offset = (container.offsetWidth - cardW) / 2;
    track.style.transform = `translateX(${ -current * cardW + offset }px)`;

    [...track.children].forEach((c, i) => c.classList.toggle("active", i === current));
    [...dots.children].forEach((d, i) => d.classList.toggle("active", i === current));
  };

  /* ---- navigation helpers (NO WRAP) ----------------------------------- */
  const restartAuto = () => {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(() => {
      if (!pause) next(true);
      restartAuto();
    }, AUTO_MS);
  };

  const goTo = (i, auto = false) => {
    if (i < 0 || i >= projects.length) return; // ignore out-of-range requests
    current = i;
    update();
    if (!auto) restartAuto();
  };

  const next     = (auto = false) => goTo((current + 1) % projects.length, auto);
  const previous = (auto = false) => goTo((current - 1 + projects.length) % projects.length, auto);

  // -----------------------------------------------------------------------
  container.addEventListener("mouseenter",  () => pause = true);
  container.addEventListener("mouseleave",  () => pause = false);

  /* drag / swipe --------------------------------------------------------- */
  let dragStart = 0, dragging = false, moved = false;
  const getX = e => (e.touches ? e.touches[0].clientX : e.clientX);

  const onDown = e => {
    dragging = true;
    moved    = false;
    dragStart = getX(e);
    container.classList.add("dragging");
  };

  const onMove = e => {
    if (!dragging) return;
    const diff = getX(e) - dragStart;
    if (Math.abs(diff) > 40) {
      dragging = false;
      moved    = true;
      if (diff < 0) next(); else previous();
    }
  };

  const onUp = () => { dragging = false; container.classList.remove("dragging"); };

  container.addEventListener("mousedown",  onDown);
  container.addEventListener("touchstart", onDown, { passive: true });
  container.addEventListener("mousemove",  onMove);
  container.addEventListener("touchmove",  onMove, { passive: true });
  container.addEventListener("mouseup",    onUp);
  container.addEventListener("mouseleave", onUp);
  container.addEventListener("touchend",   onUp);

  /* autoplay ------------------------------------------------------------- */
  restartAuto();

  window.addEventListener("resize", update);

  update(); // initial positioning
}


function initSeeMore(){
  const btn = document.getElementById("see-more");
  const filters  = document.getElementById("filters");
  const grid     = document.getElementById("projects");
  const gap      = document.getElementById("carousel-gap");
  const menu     = document.getElementById("filter-menu");
  const mobile   = window.matchMedia("(max-width: 768px)");
  const gapPad   = gap ? parseFloat(getComputedStyle(gap).paddingTop) || 32 : 0;
  const carousel = document.getElementById("portfolio-carousel-section");
  if(!btn || !filters || !grid) return;

  const selectAll = () => {
    if (!menu) return;
    const allBtn = menu.querySelector('[data-filter="all"]');
    if (!allBtn) return;
    [...menu.children].forEach(b => {
      b.classList.replace("btn-primary", "btn-secondary");
      b.setAttribute("aria-selected", "false");
    });
    allBtn.classList.replace("btn-secondary", "btn-primary");
    allBtn.setAttribute("aria-selected", "true");
    [...grid.children].forEach(c => c.classList.remove("hide"));
  };
  btn.addEventListener("click", () => {
    const expanded = btn.dataset.expanded === "true";
    btn.dataset.expanded = expanded ? "false" : "true";
    btn.textContent = expanded ? "See More" : "See Less";

    if (mobile.matches) {
      filters.classList.toggle("hide", expanded);
      grid.classList.toggle("hide", expanded);
      if (gap) gap.classList.toggle("hide", expanded);
      return;
    }

    if (expanded) {
      // collapse grid, filters, and gap smoothly
      const gStart   = grid.offsetHeight;
      const fStart   = filters.offsetHeight;

      grid.style.height = `${gStart}px`;
      filters.style.height = `${fStart}px`;
      if (gap) {
        gap.style.paddingTop = `${gapPad}px`;
        gap.style.paddingBottom = `${gapPad}px`;
      }
      filters.classList.add("grid-fade");
      grid.classList.add("grid-fade");
      if (gap) gap.classList.add("grid-fade");

      requestAnimationFrame(() => {
        grid.style.height = "0px";
        filters.style.height = "0px";
        if (gap) {
          gap.style.paddingTop = "0px";
          gap.style.paddingBottom = "0px";
        }
        grid.style.paddingTop = "0px";
        grid.style.paddingBottom = "0px";
        filters.style.paddingTop = "0px";
        filters.style.paddingBottom = "0px";
      });

      setTimeout(() => {
        grid.classList.add("hide");
        filters.classList.add("hide");
        if (gap) gap.classList.add("hide");
        grid.classList.remove("grid-fade");
        filters.classList.remove("grid-fade");
        if (gap) gap.classList.remove("grid-fade");
        grid.style.height = "";
        filters.style.height = "";
        if (gap) {
          gap.style.paddingTop = "";
          gap.style.paddingBottom = "";
        }
        grid.style.paddingTop = "";
        grid.style.paddingBottom = "";
        filters.style.paddingTop = "";
        filters.style.paddingBottom = "";
      carousel?.scrollIntoView({ behavior: "smooth" });
      }, 450); // height transition duration
    } else {
      // expand grid, filters, and gap smoothly
      selectAll();
      filters.classList.remove("hide");
      grid.classList.remove("hide");
      if (gap) gap.classList.remove("hide");
      // ensure reveal animations don't keep them hidden
      filters.classList.add("active");
      grid.classList.add("active");
      if (gap) gap.classList.add("active");

      /* Safari sometimes returns 0 if measured immediately */
      void grid.offsetHeight;
      void filters.offsetHeight;

      const gTarget = grid.scrollHeight;
      const fTarget = filters.scrollHeight;
      const gapTarget = gap ? gapPad : 0;

      grid.style.height = "0px";
      filters.style.height = "0px";
      if (gap) {
        gap.style.paddingTop = "0px";
        gap.style.paddingBottom = "0px";
      }
      grid.style.paddingTop = "0px";
      grid.style.paddingBottom = "0px";
      filters.style.paddingTop = "0px";
      filters.style.paddingBottom = "0px";
      filters.classList.add("grid-fade");
      grid.classList.add("grid-fade");
      if (gap) gap.classList.add("grid-fade");

      requestAnimationFrame(() => {
        grid.style.height = `${gTarget}px`;
        filters.style.height = `${fTarget}px`;
        if (gap) {
          gap.style.paddingTop = `${gapPad}px`;
          gap.style.paddingBottom = `${gapPad}px`;
        }
        grid.style.paddingTop = "";
        grid.style.paddingBottom = "";
        filters.style.paddingTop = "";
        filters.style.paddingBottom = "";
        // cascade project cards as they reappear
        [...grid.children].forEach((card, i) => {
          if (card.classList.contains("hide")) return;
          card.classList.remove("ripple-in");
          void card.offsetWidth;
          card.style.animationDelay = `${i * 80}ms`;
          card.classList.add("ripple-in");
        });
        grid.classList.remove("grid-fade");
        filters.classList.remove("grid-fade");
        if (gap) gap.classList.remove("grid-fade");
      });

      setTimeout(() => {
        grid.style.height = "";
        filters.style.height = "";
        if (gap) {
          gap.style.paddingTop = `${gapPad}px`;
          gap.style.paddingBottom = `${gapPad}px`;
        }

        // ensure the newly revealed filters are visible on mobile
        const nav = parseFloat(
          getComputedStyle(document.documentElement).getPropertyValue(
            "--nav-height"
          )
        ) || 0;
        const target = gap || filters;
        const y = target.getBoundingClientRect().top + window.scrollY - nav;
        window.scrollTo({ top: y, behavior: "smooth" });
      }, 450);
    }
  });
}

/* ────────────────────────────────────────────────────────────
   DOM-builder  (loads all projects immediately)
   ------------------------------------------------------------------
   • Builds cards inside  #projects
   • Builds modals inside #modals
   • Populates #filter-menu counts & click-to-filter behaviour
   ------------------------------------------------------------------ */
function buildPortfolio() {
  const grid   = document.getElementById("projects");
  const modals = document.getElementById("modals");
  const menu   = document.getElementById("filter-menu");
  if (!grid || !modals || !menu || !window.PROJECTS) return;

  /* helper – create & return element */
  const el = (tag, cls = "", html = "") => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html) n.innerHTML = html;
    return n;
  };

(() => {
  const mq = window.matchMedia("(max-width:768px)");
  const updateIframes = () => {
    document.querySelectorAll(".modal-embed iframe[data-base]")
      .forEach(f => {
        const base = f.dataset.base;
        f.src = `${base}?${[
          ":embed=y",
          ":showVizHome=no",
          `:device=${mq.matches ? "phone" : "desktop"}`
        ].join("&")}`;
      });
  };
  mq.addEventListener("change", updateIframes);
})();


  /* ➊ Build cards & modals ----------------------------------------- */
  window.PROJECTS.forEach((p, i) => {
    /* card */
    const card = el("div", "project-card", `
      <div class="overlay"></div>
      <div class="project-title">${p.title}</div>
      <div class="project-subtitle">${p.subtitle}</div>
      <img src="${p.image}" alt="${p.title}" loading="lazy">`);
    card.dataset.index = i;
    card.dataset.tags  = p.tools.join(",");
    card.addEventListener("click", () => openModal(p.id));
    grid.appendChild(card);

    /* modal */
    const modal = el("div","modal");
    modal.id = `${p.id}-modal`;
    modal.innerHTML = window.generateProjectModal(p);
    modals.appendChild(modal);
  });

  /* ➋ Animate cards right away (no IntersectionObserver) ----------- */
  [...grid.children].forEach((c, i) => {
    c.style.animationDelay = `${i * 80}ms`;
    c.classList.add("ripple-in");
  });

  /* ── auto-scroll to first card once it fades in (mobile only) ── */
  const isMobileInitial = window.matchMedia("(max-width: 768px)").matches;
  if (isMobileInitial) {
    const first = grid.firstElementChild;
    if (first) {
      const offset =
        parseFloat(
          getComputedStyle(document.documentElement).getPropertyValue(
            "--nav-height"
          )
        ) || 0;
      const y = first.getBoundingClientRect().top + window.scrollY - offset;
      setTimeout(() => {
        window.scrollTo({ top: y, behavior: "smooth" });
      }, 600); // ripple-in animation ≈550ms
    }
  }

  /* ➌ Build filter-button counts ----------------------------------- */
  const counts = { all: window.PROJECTS.length };
  window.PROJECTS.forEach(p => p.tools.forEach(t => counts[t] = (counts[t] || 0) + 1));
  [...menu.children].forEach(btn => {
    const tag = btn.dataset.filter;
    btn.innerHTML = `${btn.textContent.trim()} ${(counts[tag] || 0)}/${counts.all}`;
  });

  /* ➍ Filter behaviour (fade-out → update → fade-in) --------------- */
  menu.addEventListener("click", e => {
    if (!e.target.dataset.filter) return;

    /* button UI */
    [...menu.children].forEach(b => {
      b.classList.replace("btn-primary", "btn-secondary");
      b.setAttribute("aria-selected", "false");
    });
    e.target.classList.replace("btn-secondary", "btn-primary");
    e.target.setAttribute("aria-selected", "true");

    const tag   = e.target.dataset.filter;
    const start = grid.offsetHeight;
    grid.style.height = `${start}px`;
    grid.classList.add("grid-fade");

    setTimeout(() => {
      [...grid.children].forEach(card => {
        card.classList.toggle("hide", tag !== "all" && !card.dataset.tags.includes(tag));
      });

      const visible = [...grid.children].filter(c => !c.classList.contains("hide"));
      grid.style.height = `${grid.scrollHeight}px`;

      visible.forEach((card, i) => {
        card.classList.remove("ripple-in");
        void card.offsetWidth;           // restart animation
        card.style.animationDelay = `${i * 80}ms`;
        card.classList.add("ripple-in");
      });

      grid.classList.remove("grid-fade");

      setTimeout(() => {
        grid.style.height = "";

        /* ─── ensure first visible card is flush on mobile ─── */
        const isMobileFilter = window.matchMedia("(max-width: 768px)").matches;
        if (isMobileFilter) {
          const first = visible[0];
          if (first) {
            const offset =
              parseFloat(
                getComputedStyle(document.documentElement).getPropertyValue(
                  "--nav-height"
                )
              ) || 0;
            const y = first.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top: y, behavior: "smooth" });
          }
        }
      }, 450); // height transition duration
    }, 350);   // grid fade duration
  });
}
