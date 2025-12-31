/* portfolio.js - Build portfolio UI components. Project data now lives in projects-data.js */

const getSrStatus = typeof window.getSrStatusNode === 'function'
  ? window.getSrStatusNode
  : (function () {
      let el = null;
      return function () {
        if (el) return el;
        el = document.createElement('div');
        el.id = 'sr-status';
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');
        el.setAttribute('aria-atomic', 'true');
        el.style.position = 'absolute';
        el.style.left = '-9999px';
        el.style.width = '1px';
        el.style.height = '1px';
        el.style.overflow = 'hidden';
        document.body.appendChild(el);
        return el;
      };
    })();

const srStatus = () => getSrStatus();
const getImageSizeAttr = (p = {}) => {
  const width = Number(p.imageWidth);
  const height = Number(p.imageHeight);
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
    return ` width="${width}" height="${height}"`;
  }
  return '';
};

const projectMedia = window.projectMedia || ((p = {}) => {
  if (!p.image) return '';
  const sizeAttr = getImageSizeAttr(p);
  return `<img src="${p.image}" alt="${p.title || ''}" loading="lazy" decoding="async" draggable="false"${sizeAttr}>`;
});

const setupPreviewVideo = (card) => {
  if (!card || card._previewVideoBound) return;
  const vid = card.querySelector && card.querySelector('video.gif-video');
  if (!vid) return;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
  const sources = [...vid.querySelectorAll('source[data-src]')];
  const loadSources = () => {
    if (vid.dataset.loaded === 'true') return;
    sources.forEach((source) => {
      if (!source.src && source.dataset.src) {
        source.src = source.dataset.src;
      }
    });
    vid.dataset.loaded = 'true';
    try { vid.load(); } catch {}
  };
  const playVideo = () => {
    loadSources();
    card.classList.add('is-video-active');
    try { vid.play && vid.play().catch(() => {}); } catch {}
  };
  const pauseVideo = () => {
    try { vid.pause && vid.pause(); } catch {}
    card.classList.remove('is-video-active');
  };
  card._previewVideoStop = pauseVideo;
  if (reduce || !finePointer) {
    pauseVideo();
    return;
  }
  card._previewVideoBound = true;
  card.addEventListener('pointerenter', playVideo);
  card.addEventListener('focusin', playVideo);
  card.addEventListener('pointerleave', pauseVideo);
  card.addEventListener('focusout', pauseVideo);
};

const isPublishedProject = (project) => project && project.published !== false;

const hasModalHelpers = typeof window.openModal === 'function' && typeof window.generateProjectModal === 'function';
if (!hasModalHelpers) {
  console.warn('modal-helpers.js was not loaded before portfolio.js; modal interactions will be limited.');
}
const openModal = hasModalHelpers ? window.openModal.bind(window) : () => {};

/* ────────────────────────────────────────────────────────────
   Portfolio Carousel (top of page) – no wrap-around version
   ------------------------------------------------------------------ */
function buildPortfolioCarousel() {
  const container = document.getElementById("portfolio-carousel");
  if (!container || !window.PROJECTS) return;
  const allProjects = (Array.isArray(window.PROJECTS) ? window.PROJECTS : []).filter(isPublishedProject);
  if (!allProjects.length) return;

  const track = container.querySelector(".carousel-track");
  const dots  = container.querySelector(".carousel-dots");
  if (!track || !dots) return;
  const isPortfolioPage = document.body && document.body.dataset.page === 'portfolio';
  const usesModals = isPortfolioPage;
  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isTypingTarget = (node) => {
    if (!node) return false;
    if (node.isContentEditable) return true;
    const tag = (node.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select';
  };

  // Make carousel focusable and describe semantics for AT users
  container.setAttribute('tabindex', '0');
  container.setAttribute('role', 'region');
  container.setAttribute('aria-roledescription', 'carousel');
  container.setAttribute('aria-label', 'Featured projects');

  // 1‒5 featured projects -------------------------------------------------
  let projects = [];
  if (Array.isArray(window.FEATURED_IDS) && window.FEATURED_IDS.length) {
    window.FEATURED_IDS.forEach(id => {
      const p = allProjects.find(pr => pr.id === id);
      if (p) projects.push(p);
    });
  } else {
    projects = allProjects.slice(0, 5);
  }

  const ld = {
    "@context":"https://schema.org",
    "@type":"ItemList",
    "itemListElement": projects.map((p, i) => ({
      "@type":"ListItem",
      "position": i+1,
      "item": {
        "@type":"CreativeWork",
        "name": p.title,
        "description": p.subtitle,
        "url": `https://danielshort.me/portfolio/${p.id}`,
        "image": `https://danielshort.me/${p.image}`
      }
    }))
  };
  const itemListId = "portfolio-carousel-itemlist";
  const s = document.getElementById(itemListId) || document.createElement("script");
  s.id = itemListId;
  s.type = "application/ld+json";
  s.textContent = JSON.stringify(ld);
  if (!s.parentNode) document.head.appendChild(s);

  // Per-project structured data for better discoverability
  if (isPortfolioPage) {
    try {
      const graph = allProjects.map(p => ({
        "@type": "CreativeWork",
        "name": p.title,
        "description": p.subtitle,
        "url": `https://danielshort.me/portfolio/${p.id}`,
        "image": `https://danielshort.me/${p.image}`
      }));
      const graphId = "portfolio-carousel-graph";
      const s2 = document.getElementById(graphId) || document.createElement('script');
      s2.id = graphId;
      s2.type = 'application/ld+json';
      s2.textContent = JSON.stringify({ "@context": "https://schema.org", "@graph": graph });
      if (!s2.parentNode) document.head.appendChild(s2);
    } catch {}
  }

  track.innerHTML = "";
  dots.innerHTML  = "";

  projects.forEach((p, i) => {
    /* slide */
    const sizeAttr = getImageSizeAttr(p);
    const card = usesModals ? document.createElement("button") : document.createElement("a");
    if (usesModals) {
      card.type = "button";
    } else {
      card.href = `portfolio/${p.id}`;
    }
    card.className = "project-card carousel-card";
    card.setAttribute("aria-label", usesModals ? `View details of ${p.title}` : `Read case study: ${p.title}`);
    const media = (() => {
      const hasVideo = !!(p.videoWebm || p.videoMp4);
      const hasImage = !!p.image;
      const img = (() => {
        if (!hasImage) return '';
        const src = p.image || '';
        const lower = src.toLowerCase();
        const webp = lower.endsWith('.png') ? src.replace(/\.png$/i, '.webp')
                   : lower.endsWith('.jpg') ? src.replace(/\.jpg$/i, '.webp')
                   : lower.endsWith('.jpeg') ? src.replace(/\.jpeg$/i, '.webp')
                   : null;
        if (webp) {
          return `<picture>
            <source srcset="${webp}" type="image/webp">
            <img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false"${sizeAttr} fetchpriority="${i===0 ? 'high' : 'auto'}">
          </picture>`;
        }
        return `<img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false"${sizeAttr} fetchpriority="${i===0 ? 'high' : 'auto'}">`;
      })();
      if (!hasVideo) return img;
      const mp4  = p.videoMp4  ? `<source data-src="${p.videoMp4}" type="video/mp4">`   : '';
      const webm = p.videoWebm ? `<source data-src="${p.videoWebm}" type="video/webm">` : '';
      const poster = p.image ? ` poster="${p.image}"` : '';
      const videoClass = (p.videoOnly || !hasImage) ? 'gif-video gif-video-only' : 'gif-video';
      const video = `
        <video class="${videoClass}" muted playsinline loop preload="none"${poster} draggable="false">
          ${mp4}
          ${webm}
        </video>`;
      if (p.videoOnly || !hasImage) {
        return video;
      }
      return `
        ${video}
        ${img}`;
    })();
    card.innerHTML = `
      <div class="overlay"></div>
      <div class="project-text">
        <div class="project-title">${p.title}</div>
        <div class="project-subtitle">${p.subtitle}</div>
      </div>
      ${media}
    `;
    if (usesModals) {
      card.addEventListener("click", () => { if (!moved) openModal(p.id); });
      card.addEventListener("keydown", ev => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          if (!moved) openModal(p.id);
        }
      });
    } else {
      card.addEventListener("click", (ev) => {
        if (moved) ev.preventDefault();
      });
    }
    setupPreviewVideo(card);
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
    const firstCard = track.children[0];
    if (!firstCard) return;
    const cs = getComputedStyle(track);
    const gap = parseFloat(cs.columnGap || cs.gap || '0') || 0;   // actual flex gap
    const cardWidth = firstCard.offsetWidth;                      // layout width of a card
    const step = cardWidth + gap;                                 // distance between card left-edges
    const offset = (container.offsetWidth - cardWidth) / 2;       // center a single card (no gap)
    track.style.transform = `translateX(${ -current * step + offset }px)`;

    [...track.children].forEach((c, i) => c.classList.toggle("active", i === current));
    [...dots.children].forEach((d, i) => {
      d.classList.toggle("active", i === current);
      d.setAttribute('role', 'tab');
      d.setAttribute('aria-selected', String(i === current));
      d.tabIndex = i === current ? 0 : -1;
    });
    // Ensure non-active slides don't keep preview videos running
    try {
      [...track.children].forEach((card, i) => {
        if (i !== current && card._previewVideoStop) {
          card._previewVideoStop();
        }
      });
    } catch {}
    try {
      const title = projects[current]?.title || '';
      srStatus().textContent = `Slide ${current+1} of ${projects.length}: ${title}`;
    } catch {}
  };

  /* ---- navigation helpers (NO WRAP) ----------------------------------- */
  const restartAuto = () => {
    if (prefersReduced) return; // respect reduced motion on mobile
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

  // Keyboard navigation for desktop users
  container.addEventListener('keydown', (e) => {
    switch (e.key) {
      case 'ArrowRight': next(); e.preventDefault(); break;
      case 'ArrowLeft':  previous(); e.preventDefault(); break;
      case 'Home':       goTo(0); e.preventDefault(); break;
      case 'End':        goTo(projects.length - 1); e.preventDefault(); break;
    }
  });
  if (isPortfolioPage && !container.dataset.globalKeysBound) {
    container.dataset.globalKeysBound = 'yes';
    document.addEventListener('keydown', (e) => {
      if (e.defaultPrevented) return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.key === 'ArrowRight') {
        next();
        e.preventDefault();
      } else if (e.key === 'ArrowLeft') {
        previous();
        e.preventDefault();
      }
    });
  }

  /* drag / swipe --------------------------------------------------------- */
  let dragStart = 0, dragging = false, moved = false;
  const getX = e => (e.touches ? e.touches[0].clientX : e.clientX);

  const onDown = e => {
    dragging = true;
    moved    = false;
    dragStart = getX(e);
    container.classList.add("dragging");
    if (e.type === 'mousedown') {
      e.preventDefault(); // prevent native image dragging/select
    }
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
  // Prevent native drag on images/videos inside carousel
  container.addEventListener('dragstart', (ev) => ev.preventDefault());

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
  const params   = new URLSearchParams(window.location.search);
  if(!btn || !filters || !grid) return;

  const selectAll = () => {
    if (!menu) return;
    const allBtn = menu.querySelector('[data-filter="all"]');
    if (!allBtn) return;
    [...menu.children].forEach(b => {
      b.classList.replace("btn-primary", "btn-secondary");
      b.setAttribute("aria-pressed", "false");
    });
    allBtn.classList.replace("btn-secondary", "btn-primary");
    allBtn.setAttribute("aria-pressed", "true");
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

  const desiredView = params.get("view");
  if (desiredView === "all" && btn.dataset.expanded !== "true") {
    const hash = location.hash || '';
    btn.click();
    params.delete("view");
    const nextSearch = params.toString();
    const newUrl = `${location.pathname}${nextSearch ? `?${nextSearch}` : ""}${hash}`;
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", newUrl);
    }

    const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const behavior = prefersReduced ? 'auto' : 'smooth';
    const scrollToTarget = () => {
      let targetId = (hash || '').replace(/^#/, '');
      if (targetId) {
        try { targetId = decodeURIComponent(targetId); } catch {}
      }
      const explicitTarget = targetId ? document.getElementById(targetId) : null;
      const target = explicitTarget || filters || grid || gap;
      if (!target) return;
      const nav = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue('--nav-height')
      ) || 0;
      const y = target.getBoundingClientRect().top + window.scrollY - nav;
      window.scrollTo({ top: y, behavior });
    };

    // Align with the show/hide transitions (desktop animates; mobile is immediate)
    setTimeout(scrollToTarget, mobile.matches ? 80 : 520);
  }
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
  const projects = (Array.isArray(window.PROJECTS) ? window.PROJECTS : []).filter(isPublishedProject);
  if (!projects.length) return;

  grid.innerHTML = "";
  modals.innerHTML = "";
  const TOTAL_PROJECTS = projects.length;
  const filterGroups = new Map();
  const groupState = {};
  const groupButtons = [...menu.querySelectorAll('button[data-filter-group]')];
  if (!groupButtons.length) return;
  groupButtons.forEach((btn) => {
    const group = btn.dataset.filterGroup || 'tools';
    if (!filterGroups.has(group)) filterGroups.set(group, []);
    filterGroups.get(group).push(btn);
    btn.dataset.baseLabel = btn.textContent.trim();
  });
  filterGroups.forEach((buttons, group) => {
    const defaultBtn = buttons.find(btn => (btn.dataset.filter || '').toLowerCase() === 'all') || buttons[0];
    groupState[group] = defaultBtn?.dataset.filter || 'all';
  });
  const valueAccessors = {
    concept: (project) => Array.isArray(project.concepts) ? project.concepts : [],
    tools: (project) => Array.isArray(project.tools) ? project.tools : []
  };
  const filterGroupKeys = [...filterGroups.keys()];
  const getProjectValues = (project, group) => {
    const accessor = valueAccessors[group];
    if (!accessor) return [];
    const values = accessor(project);
    return Array.isArray(values) ? values.filter(Boolean) : [];
  };
  const matchesState = (project, overrides = {}) => {
    const state = { ...groupState, ...overrides };
    return filterGroupKeys.every((group) => {
      const selected = state[group] || 'all';
      if (selected === 'all') return true;
      return getProjectValues(project, group).includes(selected);
    });
  };
  const computeGroupCounts = (group) => {
    const counts = { all: 0 };
    projects.forEach((project) => {
      if (!matchesState(project, { [group]: 'all' })) return;
      counts.all++;
      getProjectValues(project, group).forEach((value) => {
        counts[value] = (counts[value] || 0) + 1;
      });
    });
    return counts;
  };

  // Data order now reflects desired grid order (no runtime reordering)

  // Optional prefilters from query string (e.g., ?filterTools=Python&filterConcept=Machine%20Learning)
  const params = new URLSearchParams(window.location.search);
  const applyPrefilter = (paramName, group) => {
    const value = params.get(paramName);
    if (!value || !filterGroups.has(group)) return false;
    const buttons = filterGroups.get(group);
    const target = buttons.find(
      (btn) => (btn.dataset.filter || '').toLowerCase() === value.toLowerCase()
    );
    if (!target) return false;
    groupState[group] = target.dataset.filter || 'all';
    return true;
  };
  applyPrefilter('filterTools', 'tools');
  applyPrefilter('filterConcept', 'concept');

  /* helper – create & return element */
  const el = (tag, cls = "", html = "") => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html) n.innerHTML = html;
    return n;
  };

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const mobileMq = window.matchMedia
    ? window.matchMedia("(max-width: 768px)")
    : { matches: false, addEventListener() {}, addListener() {} };

  const setupCardPreview = (card) => {
    setupPreviewVideo(card);
  };

  (() => {
    const updateIframes = () => {
      document.querySelectorAll(".modal-embed iframe[data-base]")
        .forEach(f => {
          const base = f.dataset.base;
          f.src = `${base}?${[
            ":embed=y",
            ":showVizHome=no",
            `:device=${mobileMq.matches ? "phone" : "desktop"}`
          ].join("&")}`;
        });
    };
    mobileMq.addEventListener("change", updateIframes);
  })();


  /* ➊ Build cards & modals ----------------------------------------- */
  projects.forEach((p, i) => {
    /* card */
    const mediaMarkup = projectMedia(p);
    const card = el("button", "project-card", `
      <div class="overlay"></div>
      <div class="project-text">
        <div class="project-title">${p.title}</div>
        <div class="project-subtitle">${p.subtitle}</div>
      </div>
      ${mediaMarkup}`);
    card.type = "button";
    card.setAttribute("aria-label", `View details of ${p.title}`);
    card.dataset.index = i;
    card.dataset.tools = (Array.isArray(p.tools) ? p.tools : []).join('|');
    card.dataset.concepts = (Array.isArray(p.concepts) ? p.concepts : []).join('|');
    card.addEventListener("click", () => openModal(p.id));
    setupCardPreview(card);
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

  /* ➌ Build filter-button counts ----------------------------------- */
  const refreshFilterLabels = () => {
    filterGroups.forEach((buttons, group) => {
      const counts = computeGroupCounts(group);
      const total = counts.all || TOTAL_PROJECTS;
      let current = groupState[group] || 'all';
      if (current !== 'all' && !counts[current]) {
        current = 'all';
        groupState[group] = current;
      }
      buttons.forEach(btn => {
        const value = btn.dataset.filter || 'all';
        const baseLabel = btn.dataset.baseLabel || btn.textContent.trim();
        const count = value === 'all' ? total : (counts[value] || 0);
        const isActive = value === current;
        btn.innerHTML = `${baseLabel} ${count}/${total}`;
        btn.classList.toggle('btn-primary', isActive);
        btn.classList.toggle('btn-secondary', !isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
        const disable = value !== 'all' && count === 0;
        btn.disabled = disable;
        btn.setAttribute('aria-disabled', disable ? 'true' : 'false');
        btn.classList.toggle('filter-chip-disabled', disable);
      });
    });
  };
  refreshFilterLabels();

  /* ➍ Filter behaviour (fade-out → update → fade-in) --------------- */
  const GRID_FADE_MS   = reduceMotion ? 0 : 350; // match #projects opacity transition
  const GRID_RESIZE_MS = reduceMotion ? 0 : 450; // match #projects height transition
  const GRID_HIDDEN_CLASS = "grid-hidden";
  let fadeTimer;
  let revealTimer;

  const FILTER_GROUP_LABELS = {
    concept: 'Focus',
    tools: 'Tools'
  };
  const matchesSelections = (card) => {
    const index = Number(card.dataset.index);
    const project = projects[index];
    if (!project) return true;
    return matchesState(project);
  };
  const runFilter = () => {
    clearTimeout(fadeTimer);
    clearTimeout(revealTimer);

    const startHeight = grid.offsetHeight;
    grid.style.height = `${startHeight}px`;
    grid.classList.remove(GRID_HIDDEN_CLASS);
    if (!reduceMotion) grid.classList.add("grid-fade");

    const applyFilter = () => {
      if (!reduceMotion) grid.classList.add(GRID_HIDDEN_CLASS);
      const cards = [...grid.children];
      cards.forEach(card => {
        const shouldShow = matchesSelections(card);
        card.classList.toggle("hide", !shouldShow);
      });

      const visible = cards.filter(c => !c.classList.contains("hide"));
      grid.style.height = `${grid.scrollHeight}px`;

      const reveal = () => {
        visible.forEach((card, i) => {
          card.classList.remove("ripple-in");
          void card.offsetWidth;
          card.style.animationDelay = `${i * 80}ms`;
          card.classList.add("ripple-in");
        });

        grid.classList.remove(GRID_HIDDEN_CLASS);
        grid.classList.remove("grid-fade");
        grid.style.height = "";

        const isMobileFilter = window.matchMedia("(max-width: 768px)").matches;
        if (isMobileFilter && !reduceMotion) {
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

        try {
          const visibleCount = visible.length;
          const summary = filterGroupKeys
            .map(group => `${FILTER_GROUP_LABELS[group] || group}: ${groupState[group] || 'all'}`)
            .join('; ');
          srStatus().textContent = `Showing ${visibleCount} projects. ${summary}`;
        } catch {}
      };

      if (GRID_RESIZE_MS) {
        revealTimer = setTimeout(reveal, GRID_RESIZE_MS);
      } else {
        reveal();
      }
    };

    if (GRID_FADE_MS) {
      fadeTimer = setTimeout(applyFilter, GRID_FADE_MS);
    } else {
      applyFilter();
    }
  };

  menu.addEventListener("click", e => {
    const resetTarget = e.target.closest('button[data-filter-reset]');
    if (resetTarget) {
      const group = resetTarget.dataset.filterReset || 'tools';
      if (filterGroups.has(group)) {
        groupState[group] = 'all';
        refreshFilterLabels();
        runFilter();
      }
      e.preventDefault();
      return;
    }
    const targetBtn = e.target.closest('button[data-filter-group]');
    if (!targetBtn) return;
    e.preventDefault();
    const group = targetBtn.dataset.filterGroup || 'tools';
    if (!filterGroups.has(group)) return;
    groupState[group] = targetBtn.dataset.filter || 'all';
    refreshFilterLabels();
    runFilter();
  });

  // Apply any preselected filters immediately on load
  runFilter();

  /* ➎ Open modal based on URL (hash, clean path, or query) --------- */
  const getProjectIdFromURL = () => {
    // 1) query param: ?project=id (preferred)
    try {
      const qs = (location.search || '').replace(/^\?/, '');
      if (qs) {
        const pairs = qs.split('&');
        for (const kv of pairs) {
          const [k, v] = kv.split('=');
          if (decodeURIComponent(k) === 'project' && v) return decodeURIComponent(v);
        }
      }
    } catch {}
    // 2) hash fragment: #id (legacy)
    if (location.hash && location.hash.length > 1) return decodeURIComponent(location.hash.slice(1));
    // 3) clean path: /portfolio/<id> (back-compat: normalize to ?project=)
    try {
      const m = location.pathname.match(/\/portfolio\/(?:index\.html\/)?([A-Za-z0-9_-]+)\/?$/);
      if (m && m[1]) return decodeURIComponent(m[1]);
    } catch {}
    return null;
  };

  const openFromURL = () => {
    let id = getProjectIdFromURL();
    // If path was a clean slug, normalize URL to ?project=
    try {
      const pathSlug = location.pathname.match(/\/portfolio\/(?:index\.html\/)?([A-Za-z0-9_-]+)\/?$/);
      const base = portfolioBasePath();
      if (!id && pathSlug && pathSlug[1] && base && history && history.replaceState) {
        id = decodeURIComponent(pathSlug[1]);
        history.replaceState(null, '', `${base}?project=${encodeURIComponent(id)}`);
      }
    } catch {}
    const modal = id && document.getElementById(`${id}-modal`);
    if (modal) openModal(id);
    else if (!id) {
      // If URL lacks a project and a modal is open, close it
      const open = document.querySelector('.modal.active');
      if (open) closeModal(open.id.replace(/-modal$/, ''));
    }
  };

  // Initial open + respond to both hash and history navigation
  openFromURL();
  window.addEventListener('hashchange', openFromURL);
  window.addEventListener('popstate', openFromURL);
}

// Test/helper: expose URL parsing so tests can verify hash support
if (typeof window.__portfolio_getIdFromURL !== 'function') {
  window.__portfolio_getIdFromURL = function(){
    try {
      const qs = (location.search || '').replace(/^\?/, '');
      if (qs) {
        const pairs = qs.split('&');
        for (const kv of pairs) {
          const [k, v] = kv.split('=');
          if (decodeURIComponent(k) === 'project' && v) return decodeURIComponent(v);
        }
      }
    } catch {}
    if (location.hash && location.hash.length > 1) return decodeURIComponent(location.hash.slice(1));
    try {
      const m = location.pathname.match(/\/portfolio\/(?:index\.html\/)?([A-Za-z0-9_-]+)\/?$/);
      if (m && m[1]) return decodeURIComponent(m[1]);
    } catch {}
    return null;
  };
}
