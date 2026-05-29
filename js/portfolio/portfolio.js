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

const buildResponsiveSrcset = (base, ext, width) => {
  const fullW = Number(width);
  if (!Number.isFinite(fullW) || fullW <= 0) return `${base}.${ext}`;
  const parts = [];
  if (fullW > 640) parts.push(`${base}-640.${ext} 640w`);
  if (fullW > 960) parts.push(`${base}-960.${ext} 960w`);
  parts.push(`${base}.${ext} ${fullW}w`);
  return parts.join(', ');
};

const buildResponsivePicture = (src, alt, options = {}) => {
  if (!src) return '';
  const match = String(src).match(/\.(png|jpe?g)$/i);
  if (!match) {
    const sizeAttr = options.sizeAttr || '';
    const fetch = options.fetchpriority ? ` fetchpriority="${options.fetchpriority}"` : '';
    const sizes = options.sizes ? ` sizes="${options.sizes}"` : '';
    const loading = options.loading ? ` loading="${options.loading}"` : '';
    const decoding = options.decoding ? ` decoding="${options.decoding}"` : '';
    const draggable = options.draggable != null ? ` draggable="${options.draggable ? 'true' : 'false'}"` : '';
    return `<img src="${src}" alt="${alt || ''}"${loading}${decoding}${draggable}${sizeAttr}${sizes}${fetch}>`;
  }

  const base = src.replace(/\.(png|jpe?g)$/i, '');
  const width = Number(options.width);
  const height = Number(options.height);
  const sizeAttr = options.sizeAttr || (Number.isFinite(width) && Number.isFinite(height) ? ` width="${width}" height="${height}"` : '');
  const fetch = options.fetchpriority ? ` fetchpriority="${options.fetchpriority}"` : '';
  const sizes = options.sizes ? ` sizes="${options.sizes}"` : '';
  const loading = options.loading ? ` loading="${options.loading}"` : '';
  const decoding = options.decoding ? ` decoding="${options.decoding}"` : '';
  const draggable = options.draggable != null ? ` draggable="${options.draggable ? 'true' : 'false'}"` : '';

  const avifSrcset = buildResponsiveSrcset(base, 'avif', width);
  const webpSrcset = buildResponsiveSrcset(base, 'webp', width);
  return `<picture>
    <source srcset="${avifSrcset}" type="image/avif">
    <source srcset="${webpSrcset}" type="image/webp">
    <img src="${src}" alt="${alt || ''}"${loading}${decoding}${draggable}${sizeAttr}${sizes}${fetch}>
  </picture>`;
};

const projectMedia = window.projectMedia || ((p = {}) => {
  if (!p.image) return '';
  const sizeAttr = getImageSizeAttr(p);
  return buildResponsivePicture(p.image, p.title || '', {
    width: p.imageWidth,
    height: p.imageHeight,
    sizeAttr,
    loading: 'lazy',
    decoding: 'async',
    draggable: false,
    sizes: '(max-width: 640px) 92vw, 340px'
  });
});

const setupPreviewVideo = (card, options = {}) => {
  if (!card || card._previewVideoBound) return;
  const vid = card.querySelector && card.querySelector('video.gif-video');
  if (!vid) return;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
  const allowUserPreview = options.allowUserPreview !== false;
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
    if (reduce) return;
    loadSources();
    card.classList.add('is-video-active');
    try { vid.play && vid.play().catch(() => {}); } catch {}
  };
  const pauseVideo = () => {
    try { vid.pause && vid.pause(); } catch {}
    card.classList.remove('is-video-active');
  };
  card._previewVideoPlay = playVideo;
  card._previewVideoStop = pauseVideo;
  card._previewVideoBound = true;
  if (!allowUserPreview || reduce || !finePointer) {
    pauseVideo();
    return;
  }
  card.addEventListener('pointerenter', playVideo);
  card.addEventListener('focusin', playVideo);
  card.addEventListener('pointerleave', pauseVideo);
  card.addEventListener('focusout', pauseVideo);
};

const isPublishedProject = (project) => project && project.published !== false;

const getAudienceApi = () => window.SITE_AUDIENCE_CONFIG || null;
const normalizeAudience = (value) => {
  const api = getAudienceApi();
  if (api && typeof api.normalizeAudience === 'function') {
    return api.normalizeAudience(value);
  }
  return 'personal';
};
const getAudienceConfig = (value) => {
  const api = getAudienceApi();
  if (api && typeof api.getAudience === 'function') {
    return api.getAudience(value);
  }
  return {
    key: 'personal',
    featuredProjectIds: Array.isArray(window.FEATURED_IDS) ? window.FEATURED_IDS : [],
    portfolioTitle: 'Project Library',
    portfolioDescription: 'Projects, tools, experiments, and applied data work.'
  };
};
const getPortfolioAudienceKey = () => {
  try {
    const params = new URLSearchParams(window.location.search || '');
    const audience = params.get('audience');
    if (audience) return normalizeAudience(audience);
  } catch {}
  return null;
};
const getFeaturedProjectIds = (audienceKey) => {
  const audienceApi = getAudienceApi();
  const fallbackAudience = audienceApi && audienceApi.defaultAudience
    ? audienceApi.defaultAudience
    : 'personal';
  const resolvedAudience = audienceKey ? normalizeAudience(audienceKey) : fallbackAudience;
  const config = getAudienceConfig(resolvedAudience);
  if (config && Array.isArray(config.featuredProjectIds) && config.featuredProjectIds.length) {
    return config.featuredProjectIds;
  }
  return Array.isArray(window.FEATURED_IDS) ? window.FEATURED_IDS : [];
};
const applyPortfolioAudienceContent = (audienceKey) => {
  const audienceApi = getAudienceApi();
  const defaultAudience = audienceApi && audienceApi.defaultAudience ? audienceApi.defaultAudience : 'personal';
  const config = getAudienceConfig(audienceKey || defaultAudience);
  const title = document.getElementById('portfolio-hero-title');
  const tagline = document.getElementById('portfolio-hero-tagline');
  const eyebrow = document.getElementById('portfolio-hero-eyebrow');
  const topHeading = document.getElementById('top-projects-title');
  const allHeading = document.getElementById('all-projects-title');
  const allCopy = document.getElementById('portfolio-library-copy');
  if (!config) {
    if (eyebrow) eyebrow.textContent = 'Portfolio';
    if (title) title.textContent = 'Project Portfolio';
    if (tagline) {
      tagline.textContent = 'Start with featured projects, then browse the full project library.';
    }
    if (topHeading) topHeading.textContent = 'Top 5 Projects';
    if (allHeading) allHeading.textContent = 'Other Projects';
    if (allCopy) {
      allCopy.textContent = 'Additional case studies, tools, and experiments beyond the featured projects.';
    }
    return;
  }
  if (eyebrow) eyebrow.textContent = audienceKey ? (config.label || config.shortLabel || 'Portfolio') : 'Portfolio';
  if (title) title.textContent = config.portfolioTitle || 'Project Portfolio';
  if (tagline) tagline.textContent = config.portfolioDescription || '';
  if (topHeading) topHeading.textContent = 'Featured Projects';
  if (allHeading) allHeading.textContent = 'Other Projects';
  if (allCopy) {
    allCopy.textContent = 'Additional case studies, tools, and experiments beyond the featured projects.';
  }
};

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
  const audienceKey = getPortfolioAudienceKey();
  const audienceConfig = audienceKey ? getAudienceConfig(audienceKey) : null;

  const track = container.querySelector(".carousel-track");
  const dots  = container.querySelector(".carousel-dots");
  if (!track || !dots) return;
  const isPortfolioPage = document.body && document.body.dataset.page === 'portfolio';
  const usesModals = false;
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
  const featuredIds = audienceConfig && Array.isArray(audienceConfig.featuredProjectIds) && audienceConfig.featuredProjectIds.length
    ? audienceConfig.featuredProjectIds
    : getFeaturedProjectIds(audienceKey);
  if (featuredIds.length) {
    featuredIds.forEach(id => {
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
        "url": `https://www.danielshort.me/portfolio/${p.id}`,
        "image": `https://www.danielshort.me/${p.image}`
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
        "url": `https://www.danielshort.me/portfolio/${p.id}`,
        "image": `https://www.danielshort.me/${p.image}`
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
  dots.setAttribute('aria-label', 'Select a featured project');

  projects.forEach((p, i) => {
    /* slide */
    const sizeAttr = getImageSizeAttr(p);
    const card = usesModals ? document.createElement("button") : document.createElement("a");
    if (usesModals) {
      card.type = "button";
    } else {
      card.href = `portfolio/${encodeURIComponent(p.id)}`;
    }
    card.className = "project-card carousel-card";
    card.id = `portfolio-carousel-slide-${i}`;
    card.setAttribute("aria-label", usesModals ? `View details of ${p.title}` : `Read case study: ${p.title}`);
    const media = (() => {
      const hasVideo = !!(p.videoWebm || p.videoMp4);
      const hasImage = !!p.image;
      const img = (() => {
        if (!hasImage) return '';
        const src = p.image || '';
        return buildResponsivePicture(src, p.title || '', {
          width: p.imageWidth,
          height: p.imageHeight,
          sizeAttr,
          loading: 'lazy',
          decoding: 'async',
          draggable: false,
          sizes: '(max-width: 768px) 80vw, 340px',
          fetchpriority: i === 0 ? 'high' : 'auto'
        });
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
    setupPreviewVideo(card, { allowUserPreview: false });
    track.appendChild(card);

    /* nav dot */
    const dot = document.createElement("button");
    dot.className = "carousel-dot";
    dot.type  = "button";
    dot.setAttribute("aria-label", `Show ${p.title}`);
    dot.setAttribute("aria-controls", card.id);
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

    [...track.children].forEach((card, i) => {
      const isActive = i === current;
      card.classList.toggle("active", isActive);
      card.classList.toggle("is-animated-preview", isActive);
      if (isActive) {
        card.setAttribute('aria-current', 'true');
      } else {
        card.removeAttribute('aria-current');
      }
      if (card._previewVideoStop) {
        if (prefersReduced || !isActive) {
          card._previewVideoStop();
        } else if (card._previewVideoPlay) {
          card._previewVideoPlay();
        }
      }
    });
    [...dots.children].forEach((d, i) => {
      d.classList.toggle("active", i === current);
      d.setAttribute('role', 'tab');
      d.setAttribute('aria-selected', String(i === current));
      d.tabIndex = i === current ? 0 : -1;
    });
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


/* ────────────────────────────────────────────────────────────
   DOM-builder  (loads all projects immediately)
   ------------------------------------------------------------------
   • Builds cards inside  #projects
   • Builds modals inside #modals
   • Excludes the current audience's featured top five from the grid
   ------------------------------------------------------------------ */
function buildPortfolio() {
  const grid = document.getElementById("projects");
  const modals = document.getElementById("modals");
  if (!grid || !modals || !window.PROJECTS) return;

  const audienceKey = getPortfolioAudienceKey();
  applyPortfolioAudienceContent(audienceKey);

  const allProjects = (Array.isArray(window.PROJECTS) ? window.PROJECTS : []).filter(isPublishedProject);
  if (!allProjects.length) return;

  const featuredIds = new Set(getFeaturedProjectIds(audienceKey).slice(0, 5));
  const libraryProjects = allProjects.filter((project) => !featuredIds.has(project.id));

  grid.innerHTML = "";
  modals.innerHTML = "";

  const el = (tag, cls = "", html = "") => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html) n.innerHTML = html;
    return n;
  };

  const mobileMq = window.matchMedia
    ? window.matchMedia("(max-width: 768px)")
    : { matches: false, addEventListener() {}, addListener() {} };

  (() => {
    const updateIframes = () => {
      document.querySelectorAll(".modal-embed iframe[data-base]")
        .forEach((f) => {
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

  libraryProjects.forEach((project, index) => {
    const mediaMarkup = projectMedia(project);
    const card = el("a", "project-card", `
      <div class="overlay"></div>
      <div class="project-text">
        <div class="project-title">${project.title}</div>
        <div class="project-subtitle">${project.subtitle}</div>
      </div>
      ${mediaMarkup}`);
    card.href = `portfolio/${encodeURIComponent(project.id)}`;
    card.setAttribute("aria-label", `Read case study: ${project.title}`);
    card.dataset.index = index;
    setupPreviewVideo(card);
    grid.appendChild(card);
  });

  allProjects.forEach((project) => {
    const modal = el("div", "modal");
    modal.id = `${project.id}-modal`;
    modal.innerHTML = window.generateProjectModal(project);
    modals.appendChild(modal);
  });

  [...grid.children].forEach((card, index) => {
    card.style.animationDelay = `${index * 80}ms`;
    card.classList.add("ripple-in");
  });

  try {
    srStatus().textContent = `Showing ${libraryProjects.length} projects in the library.`;
  } catch {}

  /* ➊ Open modal based on URL (hash, clean path, or query) --------- */
  const getProjectIdFromQuery = () => {
    try {
      const params = new URLSearchParams(window.location.search || '');
      const id = params.get('project');
      return id ? String(id).trim() : null;
    } catch {
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
    }
    return null;
  };

  const getProjectIdFromURL = () => {
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
    // Canonicalize legacy deep links like /portfolio?project=<id> to the full page /portfolio/<id>.
    const queryId = getProjectIdFromQuery();
    if (queryId) {
      try {
        const base = portfolioBasePath();
        const prefix = base ? base.replace(/\/(?:pages\/)?portfolio(?:\.html)?$/, '') : '';
        const canonical = `${prefix}/portfolio/${encodeURIComponent(queryId)}`.replace(/\/{2,}/g, '/');
        location.replace(canonical);
      } catch {
        location.replace(`/portfolio/${encodeURIComponent(queryId)}`);
      }
      return;
    }

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
