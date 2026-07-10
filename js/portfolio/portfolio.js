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
const escapeHtml = (value = '') => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
const projectToolsMarkup = (project = {}, limit = 3) => {
  const tools = Array.isArray(project.tools) ? project.tools.filter(Boolean).slice(0, limit) : [];
  if (!tools.length) return '';
  return `<div class="project-card-tags">${tools.map((tool) => `<span>${escapeHtml(tool)}</span>`).join('')}</div>`;
};
const projectSignalLabel = (project = {}, index = 0, prefix = 'Project') => {
  const tools = Array.isArray(project.tools) ? project.tools.filter(Boolean) : [];
  const signal = tools[0] || String(project.subtitle || '').split(/\s+/).filter(Boolean)[0] || 'Build';
  return `${prefix} ${String(index + 1).padStart(2, '0')} / ${signal}`;
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

const FALLBACK_AUDIENCES = {
  analytics: {
    key: 'analytics',
    label: 'Data Analytics',
    shortLabel: 'Analytics',
    portfolioPath: '/portfolio?audience=analytics',
    resumePath: '/resume',
    featuredProjectIds: ['retailStore', 'targetEmptyPackage', 'pizzaDashboard', 'deliveryTip', 'ufoDashboard'],
    portfolioTitle: 'Analytics Portfolio',
    portfolioDescription: 'Featured analytics projects: reporting automation, SQL workflows, dashboarding, forecasting, and technical depth.'
  },
  personal: {
    key: 'personal',
    label: 'Personal Site',
    shortLabel: 'Personal',
    portfolioPath: '/portfolio',
    resumePath: '',
    featuredProjectIds: ['retailStore', 'chatbotLora', 'digitGenerator', 'smartSentence', 'website'],
    portfolioTitle: 'Project Library',
    portfolioDescription: 'Machine learning, analytics, software tools, and browser experiments by Daniel Short.'
  },
  'data-science': {
    key: 'data-science',
    label: 'Data Science',
    shortLabel: 'Data Science',
    portfolioPath: '/portfolio?audience=data-science',
    resumePath: '/resume-data-science',
    featuredProjectIds: ['smartSentence', 'chatbotLora', 'shapeClassifier', 'digitGenerator', 'handwritingRating'],
    portfolioTitle: 'Data Science Portfolio',
    portfolioDescription: 'Machine learning, NLP, modeling, evaluation, and production-minded experimentation.'
  },
  tourism: {
    key: 'tourism',
    label: 'Tourism Analytics',
    shortLabel: 'Tourism',
    portfolioPath: '/portfolio?audience=tourism',
    resumePath: '/resume-tourism',
    featuredProjectIds: ['chatbotLora', 'pizzaDashboard', 'covidAnalysis', 'retailStore', 'smartSentence'],
    portfolioTitle: 'Tourism Analytics Portfolio',
    portfolioDescription: 'Destination reporting, visitor demand analysis, stakeholder communication, and public-sector decision support.'
  }
};
const FALLBACK_AUDIENCE_ORDER = ['personal', 'analytics', 'data-science', 'tourism'];
const getAudienceApi = () => window.SITE_AUDIENCE_CONFIG || null;
const normalizeAudience = (value) => {
  const api = getAudienceApi();
  if (api && typeof api.normalizeAudience === 'function') {
    return api.normalizeAudience(value);
  }
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return 'personal';
  if (raw === 'datascience' || raw === 'data_science') return 'data-science';
  if (raw === 'tourism-analytics') return 'tourism';
  return FALLBACK_AUDIENCES[raw] ? raw : 'personal';
};
const getAudienceConfig = (value) => {
  const api = getAudienceApi();
  if (api && typeof api.getAudience === 'function') {
    return api.getAudience(value);
  }
  return FALLBACK_AUDIENCES[normalizeAudience(value)] || FALLBACK_AUDIENCES.personal;
};
const getPortfolioAudienceKey = () => {
  try {
    const params = new URLSearchParams(window.location.search || '');
    const audience = params.get('audience');
    if (audience) return normalizeAudience(audience);
    const mode = String(params.get('mode') || '').trim().toLowerCase();
    if (['professional', 'work', 'career', 'analytics'].includes(mode)) return normalizeAudience('analytics');
  } catch {}
  if (typeof window.isProfessionalRealm === 'function' && window.isProfessionalRealm()) {
    return normalizeAudience('analytics');
  }
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
    if (document.body) {
      document.body.dataset.audience = 'personal';
    }
    if (eyebrow) eyebrow.textContent = 'Projects';
    if (title) title.textContent = 'Project Library';
    if (tagline) {
      tagline.textContent = 'Machine learning, analytics, data systems, and browser experiments organized by the problem each project is trying to make clearer.';
    }
    if (topHeading) topHeading.textContent = 'Featured systems';
    if (allHeading) allHeading.textContent = 'Project library';
    if (allCopy) {
      allCopy.textContent = 'Additional case studies, tools, and experiments across machine learning, analytics, visual interfaces, and software systems.';
    }
    return;
  }
  const personalMode = !audienceKey || normalizeAudience(audienceKey) === 'personal';
  if (document.body) {
    document.body.dataset.audience = config.key || (personalMode ? 'personal' : normalizeAudience(audienceKey));
  }
  if (eyebrow) eyebrow.textContent = personalMode ? 'Projects' : (config.label || config.shortLabel || 'Portfolio');
  if (title) title.textContent = config.portfolioTitle || 'Project Portfolio';
  if (tagline) tagline.textContent = config.portfolioDescription || '';
  if (topHeading) topHeading.textContent = personalMode ? 'Featured systems' : 'Featured Projects';
  if (allHeading) allHeading.textContent = 'Project library';
  if (allCopy) {
    allCopy.textContent = personalMode
      ? 'Additional case studies, tools, and experiments across machine learning, analytics, visual interfaces, and software systems.'
      : 'Additional case studies, tools, and experiments beyond the featured projects.';
  }
};

const hasModalHelpers = typeof window.openModal === 'function' && typeof window.generateProjectModal === 'function';
if (!hasModalHelpers) {
  console.warn('modal-helpers.js was not loaded before portfolio.js; modal interactions will be limited.');
}
const openModal = hasModalHelpers ? window.openModal.bind(window) : () => {};

function setupPortfolioMobileFilterSheet(options = {}) {
  const {
    enabled,
    root,
    filterHost,
    filterGroups = [],
    state,
    allItems = [],
    sortSelect,
    searchInput,
    itemSingular = 'project',
    itemPlural = 'projects',
    slugify = (value = '') => String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
    requestRender = () => {}
  } = options;
  const noop = {
    render() {},
    syncSortControls() {}
  };
  if (!enabled || !root || !filterHost || !state || !filterGroups.length) return noop;

  const filterPanel = filterHost.closest('.portfolio-workbench__filters');
  if (!filterPanel) return noop;

  root.dataset.mobileFilters = 'true';
  const sheetTitleId = `${root.id || 'portfolio'}-mobile-filter-title`;
  const sheetSummaryId = `${root.id || 'portfolio'}-mobile-filter-summary`;
  if (!filterPanel.id) filterPanel.id = `${root.id || 'portfolio'}-filter-sheet`;
  filterPanel.setAttribute('aria-labelledby', sheetTitleId);

  const handle = document.createElement('div');
  handle.className = 'portfolio-filter-sheet-handle';
  handle.setAttribute('aria-hidden', 'true');
  filterPanel.prepend(handle);

  let toggleButton = null;
  const panelHead = filterPanel.querySelector('.portfolio-filter-head');
  if (panelHead) {
    const heading = panelHead.querySelector('h2');
    if (heading) heading.id = sheetTitleId;
    toggleButton = document.createElement('button');
    toggleButton.type = 'button';
    toggleButton.className = 'portfolio-filter-sheet-close';
    toggleButton.textContent = 'Done';
    toggleButton.dataset.portfolioFilterSheetToggle = 'true';
    toggleButton.setAttribute('aria-controls', filterPanel.id);
    toggleButton.setAttribute('aria-expanded', 'false');
    panelHead.append(toggleButton);
  }

  const resultsToolbar = root.querySelector('.portfolio-results-toolbar');
  const triggerButton = document.createElement('button');
  triggerButton.type = 'button';
  triggerButton.className = 'portfolio-mobile-filter-trigger';
  triggerButton.dataset.portfolioFilterSheetOpen = 'true';
  triggerButton.setAttribute('aria-controls', filterPanel.id);
  triggerButton.setAttribute('aria-expanded', 'false');
  triggerButton.innerHTML = `
    <span class="portfolio-mobile-filter-trigger__label">Filters</span>
    <span class="portfolio-mobile-filter-trigger__status" data-portfolio-mobile-filter-status>All</span>
    <span class="portfolio-mobile-filter-trigger__badge" data-portfolio-mobile-filter-badge hidden></span>
  `;
  if (resultsToolbar) {
    const sortControl = resultsToolbar.querySelector('.portfolio-sort-control');
    resultsToolbar.insertBefore(triggerButton, sortControl || null);
  }

  const backdrop = document.createElement('button');
  backdrop.type = 'button';
  backdrop.className = 'portfolio-filter-backdrop';
  backdrop.dataset.portfolioFilterSheetClose = 'true';
  backdrop.setAttribute('aria-label', `Close ${itemSingular} filters`);
  root.append(backdrop);

  const summary = document.createElement('section');
  summary.className = 'portfolio-mobile-filter-summary';
  summary.setAttribute('aria-labelledby', sheetSummaryId);
  summary.innerHTML = `
    <div class="portfolio-mobile-filter-summary__head">
      <h3 id="${escapeHtml(sheetSummaryId)}">Selected</h3>
      <button type="button" data-portfolio-mobile-clear>Clear all</button>
    </div>
    <div class="portfolio-mobile-filter-summary__chips" data-portfolio-mobile-active-chips></div>
  `;
  filterHost.before(summary);

  const quickFilters = document.createElement('section');
  quickFilters.className = 'portfolio-mobile-quick-filters';
  quickFilters.innerHTML = `
    <h3>Quick filters</h3>
    <div class="portfolio-mobile-quick-filters__grid" data-portfolio-mobile-quick-filters></div>
  `;
  filterHost.before(quickFilters);

  const allFiltersLabel = document.createElement('h3');
  allFiltersLabel.className = 'portfolio-mobile-all-filters-label';
  allFiltersLabel.textContent = 'All filters';
  filterHost.before(allFiltersLabel);

  const footer = document.createElement('div');
  footer.className = 'portfolio-mobile-filter-actions';
  footer.innerHTML = '<button type="button" data-portfolio-filter-sheet-close data-portfolio-filter-sheet-count>Show projects</button>';
  filterPanel.append(footer);

  const mobileSort = null;
  const activeChipHost = summary.querySelector('[data-portfolio-mobile-active-chips]');
  const quickHost = quickFilters.querySelector('[data-portfolio-mobile-quick-filters]');
  const clearMobileButton = summary.querySelector('[data-portfolio-mobile-clear]');
  const statusNode = triggerButton.querySelector('[data-portfolio-mobile-filter-status]');
  const badgeNode = triggerButton.querySelector('[data-portfolio-mobile-filter-badge]');
  const showCountButton = footer.querySelector('[data-portfolio-filter-sheet-count]');
  const openButton = triggerButton;

  if (sortSelect && mobileSort) {
    mobileSort.innerHTML = sortSelect.innerHTML;
    mobileSort.value = sortSelect.value;
  }

  const optionCounts = () => {
    const counts = new Map();
    filterGroups.forEach((group) => {
      group.options.forEach((option) => {
        counts.set(`${group.id}:${option.value}`, allItems.filter(option.match).length);
      });
    });
    return counts;
  };

  const activeEntries = () => {
    const entries = [];
    filterGroups.forEach((group) => {
      const selected = state.filters[group.id];
      if (!selected || !selected.size) return;
      group.options.forEach((option) => {
        if (selected.has(option.value)) {
          entries.push({
            groupId: group.id,
            groupTitle: group.title,
            value: option.value,
            label: option.label
          });
        }
      });
    });
    return entries;
  };

  const quickEntries = (() => {
    const preferred = [
      { group: 'focus', label: 'Analytics' },
      { group: 'focus', label: 'Dashboards' },
      { group: 'focus', label: 'Machine Learning' },
      { group: 'stack', label: 'Python' },
      { group: 'stack', label: 'SQL' },
      { group: 'stack', label: 'AWS' }
    ];
    const entries = [];
    const seen = new Set();
    const addEntry = (group, option) => {
      if (!group || !option) return;
      const key = `${group.id}:${option.value}`;
      if (seen.has(key)) return;
      seen.add(key);
      entries.push({
        groupId: group.id,
        groupTitle: group.title,
        value: option.value,
        label: option.label
      });
    };
    preferred.forEach((item) => {
      const group = filterGroups.find((candidate) => candidate.id === item.group);
      const option = group && group.options.find((candidate) => slugify(candidate.label) === slugify(item.label));
      addEntry(group, option);
    });
    filterGroups.forEach((group) => {
      group.options.forEach((option) => {
        if (entries.length < 6) addEntry(group, option);
      });
    });
    return entries.slice(0, 6);
  })();

  const expandableNodes = [summary, quickFilters, allFiltersLabel, filterHost, footer];
  let desktopMode = false;
  const setExpandableHidden = (hidden) => {
    expandableNodes.forEach((node) => {
      if (node) node.hidden = hidden;
    });
  };
  const syncToggleLabel = (activeCount = activeEntries().length) => {
    const open = root.classList.contains('is-filter-sheet-open');
    if (toggleButton) {
      toggleButton.textContent = 'Done';
      toggleButton.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggleButton.setAttribute('aria-label', `Close ${itemSingular} filters`);
    }
    triggerButton.setAttribute('aria-expanded', open ? 'true' : 'false');
    triggerButton.setAttribute('aria-label', activeCount
      ? `Open ${itemSingular} filters, ${activeCount} active`
      : `Open ${itemSingular} filters`);
  };
  const setSheetOpen = (open) => {
    const wasOpen = root.classList.contains('is-filter-sheet-open');
    const nextOpen = Boolean(open) && !desktopMode;
    root.classList.toggle('is-filter-sheet-open', nextOpen);
    filterPanel.dataset.expanded = nextOpen ? 'true' : 'false';
    if (document.body) document.body.classList.toggle('portfolio-filter-sheet-open', nextOpen);
    if (!desktopMode) {
      setExpandableHidden(!nextOpen);
      filterPanel.setAttribute('aria-hidden', nextOpen ? 'false' : 'true');
      filterPanel.toggleAttribute('inert', !nextOpen);
    }
    syncToggleLabel();
    if (nextOpen) {
      window.requestAnimationFrame(() => {
        const focusTarget = toggleButton || filterPanel.querySelector('.portfolio-filter-option input:not(:disabled), button:not(:disabled), input:not(:disabled), select:not(:disabled)');
        if (focusTarget) focusTarget.focus({ preventScroll: true });
      });
    } else if (wasOpen && !desktopMode) {
      window.requestAnimationFrame(() => triggerButton.focus({ preventScroll: true }));
    }
  };

  if (openButton) {
    openButton.addEventListener('click', (event) => {
      event.stopPropagation();
      setSheetOpen(!root.classList.contains('is-filter-sheet-open'));
    });
  }

  const clearFilters = () => {
    Object.values(state.filters).forEach((set) => set.clear());
    state.search = '';
    if (searchInput) searchInput.value = '';
    requestRender();
  };

  const toggleFilter = (groupId, value) => {
    const selected = state.filters[groupId];
    if (!selected) return;
    if (selected.has(value)) selected.delete(value);
    else selected.add(value);
    requestRender();
  };

  root.addEventListener('click', (event) => {
    const openTrigger = event.target.closest('[data-portfolio-filter-sheet-open]');
    if (openTrigger) {
      setSheetOpen(true);
      return;
    }
    if (event.target.closest('[data-portfolio-filter-sheet-toggle]')) {
      setSheetOpen(!root.classList.contains('is-filter-sheet-open'));
      return;
    }
    if (event.target.closest('[data-portfolio-filter-sheet-close]')) {
      setSheetOpen(false);
      return;
    }
    const quickTrigger = event.target.closest('[data-portfolio-mobile-quick-filter]');
    if (quickTrigger) {
      toggleFilter(quickTrigger.dataset.mobileFilterGroup, quickTrigger.dataset.mobileFilterValue);
      return;
    }
    const removeTrigger = event.target.closest('[data-portfolio-mobile-remove-filter]');
    if (removeTrigger) {
      const selected = state.filters[removeTrigger.dataset.mobileFilterGroup];
      if (selected) {
        selected.delete(removeTrigger.dataset.mobileFilterValue);
        requestRender();
      }
    }
  });

  if (clearMobileButton) {
    clearMobileButton.addEventListener('click', clearFilters);
  }

  if (mobileSort) {
    mobileSort.addEventListener('change', () => {
      state.sort = mobileSort.value;
      if (sortSelect) sortSelect.value = mobileSort.value;
      requestRender();
    });
  }

  document.addEventListener('keydown', (event) => {
    if (!root.classList.contains('is-filter-sheet-open')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      setSheetOpen(false);
      if (openButton) openButton.focus({ preventScroll: true });
      return;
    }
    if (event.key === 'Tab') {
      const focusable = Array.from(filterPanel.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), [href], [tabindex]:not([tabindex="-1"])'))
        .filter((node) => !node.hidden && node.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });

  if (window.matchMedia) {
    const desktopQuery = window.matchMedia('(min-width: 821px)');
    const syncDialogMode = () => {
      desktopMode = desktopQuery.matches;
      if (desktopQuery.matches) {
        setSheetOpen(false);
        filterPanel.removeAttribute('role');
        filterPanel.removeAttribute('aria-modal');
        filterPanel.removeAttribute('aria-hidden');
        filterPanel.removeAttribute('inert');
        setExpandableHidden(false);
      } else {
        filterPanel.setAttribute('role', 'dialog');
        filterPanel.setAttribute('aria-modal', 'true');
        filterPanel.setAttribute('aria-hidden', root.classList.contains('is-filter-sheet-open') ? 'false' : 'true');
        filterPanel.toggleAttribute('inert', !root.classList.contains('is-filter-sheet-open'));
        setExpandableHidden(!root.classList.contains('is-filter-sheet-open'));
      }
    };
    if (typeof desktopQuery.addEventListener === 'function') {
      desktopQuery.addEventListener('change', syncDialogMode);
    } else if (typeof desktopQuery.addListener === 'function') {
      desktopQuery.addListener(syncDialogMode);
    }
    syncDialogMode();
  } else {
    setExpandableHidden(true);
  }

  const syncSortControls = () => {
    if (mobileSort && mobileSort.value !== state.sort) mobileSort.value = state.sort;
    if (sortSelect && sortSelect.value !== state.sort) sortSelect.value = state.sort;
  };

  const render = (projects = []) => {
    syncSortControls();
    const counts = optionCounts();
    const entries = activeEntries();
    const activeCount = entries.length;
    syncToggleLabel(activeCount);
    if (statusNode) statusNode.textContent = activeCount ? `${activeCount} active` : 'All';
    if (badgeNode) {
      badgeNode.hidden = !activeCount;
      badgeNode.textContent = String(activeCount);
    }
    if (showCountButton) {
      showCountButton.textContent = `Show ${projects.length} ${projects.length === 1 ? itemSingular : itemPlural}`;
    }
    if (clearMobileButton) clearMobileButton.disabled = activeCount === 0 && !state.search;
    if (activeChipHost) {
      activeChipHost.innerHTML = entries.length ? entries.map((entry) => `
        <button type="button" class="portfolio-mobile-filter-chip" data-portfolio-mobile-remove-filter data-mobile-filter-group="${escapeHtml(entry.groupId)}" data-mobile-filter-value="${escapeHtml(entry.value)}">
          <span>${escapeHtml(entry.label)}</span>
          <span aria-hidden="true">x</span>
        </button>
      `).join('') : '<span class="portfolio-mobile-filter-summary__empty">None selected</span>';
    }
    if (quickHost) {
      quickHost.innerHTML = quickEntries.map((entry) => {
        const active = state.filters[entry.groupId] && state.filters[entry.groupId].has(entry.value);
        return `
          <button type="button" class="portfolio-mobile-quick-filter${active ? ' is-active' : ''}" data-portfolio-mobile-quick-filter data-mobile-filter-group="${escapeHtml(entry.groupId)}" data-mobile-filter-value="${escapeHtml(entry.value)}" aria-pressed="${active ? 'true' : 'false'}">
            <span>${escapeHtml(entry.label)}</span>
            <span>${counts.get(`${entry.groupId}:${entry.value}`) || 0}</span>
          </button>
        `;
      }).join('');
    }
  };

  return {
    render,
    syncSortControls
  };
}

/* ────────────────────────────────────────────────────────────
   Portfolio Carousel (top of page) – no wrap-around version
   ------------------------------------------------------------------ */
function buildPortfolioCarousel() {
  if (document.querySelector('[data-portfolio-workbench]')) return;
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
        <div class="project-card-kicker">${escapeHtml(projectSignalLabel(p, i, 'Featured'))}</div>
        <div class="project-title">${escapeHtml(p.title)}</div>
        <div class="project-subtitle">${escapeHtml(p.subtitle)}</div>
        ${projectToolsMarkup(p)}
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
function buildPortfolioWorkbenchLegacy() {
  const root = document.querySelector('[data-portfolio-workbench]');
  if (!root || !window.PROJECTS) return false;

  const filterHost = root.querySelector('[data-portfolio-filters]');
  const resultHost = root.querySelector('[data-portfolio-results]');
  const inspector = root.querySelector('[data-portfolio-inspector]');
  const countNode = root.querySelector('[data-portfolio-results-count]');
  const clearButton = root.querySelector('[data-portfolio-clear-filters]');
  const searchInput = root.querySelector('[data-portfolio-search]');
  const sortSelect = root.querySelector('[data-portfolio-sort]');
  const emptyState = root.querySelector('[data-portfolio-empty]');
  if (!filterHost || !resultHost || !inspector || !countNode) return false;

  const allProjects = (Array.isArray(window.PROJECTS) ? window.PROJECTS : []).filter(isPublishedProject);
  if (!allProjects.length) return true;

  const normalizeText = (value = '') => String(value || '').toLowerCase();
  const slugify = (value = '') => normalizeText(value).replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const truncate = (value = '', max = 150) => {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (text.length <= max) return text;
    return `${text.slice(0, Math.max(0, max - 3)).trim()}...`;
  };
  const listText = (values = []) => Array.isArray(values) ? values.filter(Boolean).join(' ') : '';
  const unique = (values = []) => Array.from(new Set(values.filter(Boolean)));
  const prioritySorter = (priority = [], countForLabel = () => 0) => {
    const rank = new Map(priority.map((label, index) => [slugify(label), index]));
    return (a, b) => {
      const aRank = rank.has(slugify(a)) ? rank.get(slugify(a)) : Infinity;
      const bRank = rank.has(slugify(b)) ? rank.get(slugify(b)) : Infinity;
      if (aRank !== bRank) return aRank - bRank;
      const countDelta = countForLabel(b) - countForLabel(a);
      if (countDelta) return countDelta;
      return String(a).localeCompare(String(b));
    };
  };
  const hasTerm = (project, term) => {
    const haystack = normalizeText([
      project.title,
      project.subtitle,
      project.notes,
      project.problem,
      listText(project.tools),
      listText(project.concepts),
      listText(project.audiences),
      listText(project.results),
      listText(project.actions)
    ].join(' '));
    return haystack.includes(normalizeText(term));
  };
  const getProjectHref = (project = {}) => `portfolio/${encodeURIComponent(project.id)}`;
  const getProjectFormats = (project = {}) => {
    const formats = ['Case Study'];
    const resourceText = Array.isArray(project.resources)
      ? project.resources.map((resource) => resource && resource.label).join(' ')
      : '';
    const text = normalizeText(`${project.title} ${project.subtitle} ${resourceText} ${listText(project.tools)}`);
    if (project.embed || text.includes('live demo') || text.includes('interactive')) formats.push('Interactive Demo');
    if (text.includes('dashboard') || text.includes('tableau') || text.includes('bi')) formats.push('Dashboard');
    if (text.includes('notebook') || text.includes('pdf') || text.includes('excel')) formats.push('Notebook / Report');
    if (text.includes('demo') || text.includes('retriever') || text.includes('solver') || text.includes('generator')) formats.push('Tool / Library');
    return unique(formats);
  };
  const getProjectFocuses = (project = {}) => {
    const focuses = Array.isArray(project.concepts) ? [...project.concepts] : [];
    if (getProjectFormats(project).includes('Dashboard')) focuses.push('Dashboards');
    if (hasTerm(project, 'nlp') || hasTerm(project, 'chatbot') || hasTerm(project, 'semantic')) focuses.push('NLP');
    if (hasTerm(project, 'etl') || hasTerm(project, 'sql') || hasTerm(project, 'pipeline')) focuses.push('Data Engineering');
    return unique(focuses);
  };
  const getPrimaryFormat = (project = {}) => {
    const formats = getProjectFormats(project);
    if (formats.includes('Dashboard')) return 'Dashboard';
    if (formats.includes('Interactive Demo')) return 'Interactive Demo';
    if (formats.includes('Tool / Library')) return 'Tool / Library';
    return 'Case Study';
  };
  const getSummary = (project = {}) => truncate(
    (Array.isArray(project.results) && project.results[0]) ||
    project.problem ||
    project.notes ||
    project.subtitle ||
    '',
    140
  );
  const focusOptionLabels = unique(allProjects.flatMap((project) => getProjectFocuses(project)))
    .sort(prioritySorter(
      ['Analytics', 'Machine Learning', 'Automation', 'Dashboards', 'Visualization', 'Data Engineering', 'NLP', 'Product'],
      (label) => allProjects.filter((project) => getProjectFocuses(project).includes(label)).length
    ));
  const stackOptionLabels = unique(allProjects.flatMap((project) => Array.isArray(project.tools) ? project.tools : []))
    .sort(prioritySorter(
      ['Python', 'AWS', 'Docker', 'SQL', 'Excel', 'Tableau', 'PyTorch', 'JavaScript'],
      (label) => allProjects.filter((project) => (
        Array.isArray(project.tools) ? project.tools : []
      ).some((tool) => slugify(tool) === slugify(label))).length
    ));
  const chipMarkup = (values = [], limit = 4, accentFirst = true) => {
    const chips = values.filter(Boolean).slice(0, limit);
    if (!chips.length) return '';
    return `<div class="portfolio-chip-row">${chips.map((value, index) => {
      const accent = accentFirst && index === 0 ? ' portfolio-chip--accent' : '';
      return `<span class="portfolio-chip${accent}">${escapeHtml(value)}</span>`;
    }).join('')}</div>`;
  };
  const calendarIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 2v4"></path><path d="M16 2v4"></path><rect x="3" y="5" width="18" height="16" rx="2"></rect><path d="M3 10h18"></path></svg>';
  const formatIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V5"></path><path d="M4 19h16"></path><path d="M8 15l3-3 3 2 5-7"></path></svg>';
  const listIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"></path><path d="M4 12h16"></path><path d="M4 17h16"></path></svg>';

  const filterGroups = [
    {
      id: 'focus',
      title: 'Focus',
      options: focusOptionLabels.map((label) => ({
        value: slugify(label),
        label,
        match: (project) => getProjectFocuses(project).some((focus) => slugify(focus) === slugify(label))
      }))
    },
    {
      id: 'stack',
      title: 'Stack',
      options: stackOptionLabels.map((label) => ({
        value: slugify(label),
        label,
        match: (project) => (Array.isArray(project.tools) ? project.tools : []).some((tool) => slugify(tool) === slugify(label))
      }))
    },
    {
      id: 'format',
      title: 'Format',
      options: [
        { value: 'case-study', label: 'Case Study', match: (project) => getProjectFormats(project).includes('Case Study') },
        { value: 'interactive-demo', label: 'Interactive Demo', match: (project) => getProjectFormats(project).includes('Interactive Demo') },
        { value: 'dashboard', label: 'Dashboard', match: (project) => getProjectFormats(project).includes('Dashboard') },
        { value: 'notebook-report', label: 'Notebook / Report', match: (project) => getProjectFormats(project).includes('Notebook / Report') },
        { value: 'tool-library', label: 'Tool / Library', match: (project) => getProjectFormats(project).includes('Tool / Library') }
      ]
    }
  ];

  const initialProjectId = (() => {
    try {
      const params = new URLSearchParams(window.location.search || '');
      const queryId = params.get('project');
      if (queryId) return queryId;
    } catch {}
    if (location.hash && location.hash.length > 1) return decodeURIComponent(location.hash.slice(1));
    return null;
  })();
  const state = {
    filters: Object.fromEntries(filterGroups.map((group) => [group.id, new Set()])),
    collapsedFilters: Object.fromEntries(filterGroups.map((group) => [group.id, false])),
    search: '',
    sort: sortSelect ? sortSelect.value : 'default',
    selectedId: allProjects.some((project) => project.id === initialProjectId) ? initialProjectId : null
  };
  let pendingSelectionScrollTop = null;

  const renderFilters = () => {
    const counts = new Map();
    filterGroups.forEach((group) => {
      group.options.forEach((option) => {
        counts.set(`${group.id}:${option.value}`, allProjects.filter(option.match).length);
      });
    });
    filterHost.innerHTML = filterGroups.map((group) => {
      const collapsed = !!state.collapsedFilters[group.id];
      const optionsId = `portfolio-filter-options-${escapeHtml(group.id)}`;
      return `
      <fieldset class="portfolio-filter-group${collapsed ? ' is-collapsed' : ''}">
        <legend class="portfolio-filter-group__legend">
          <button type="button" class="portfolio-filter-group__toggle" aria-expanded="${collapsed ? 'false' : 'true'}" aria-controls="${optionsId}" data-portfolio-filter-toggle="${escapeHtml(group.id)}">
            <span class="portfolio-filter-group__title">${escapeHtml(group.title)}</span>
            <span class="portfolio-filter-group__chevron" aria-hidden="true"></span>
          </button>
        </legend>
        <div class="portfolio-filter-options" id="${optionsId}" aria-hidden="${collapsed ? 'true' : 'false'}">
          <div class="portfolio-filter-options__inner">
            ${group.options.map((option) => `
              <label class="portfolio-filter-option">
                <input type="checkbox" name="${escapeHtml(group.id)}" value="${escapeHtml(option.value)}"${state.filters[group.id].has(option.value) ? ' checked' : ''}${collapsed ? ' disabled' : ''}>
                <span>${escapeHtml(option.label)}</span>
                <span class="portfolio-filter-option__count">${counts.get(`${group.id}:${option.value}`) || 0}</span>
              </label>
            `).join('')}
          </div>
        </div>
      </fieldset>
    `;
    }).join('');
  };

  const setFilterGroupCollapsed = (groupId, collapsed) => {
    state.collapsedFilters[groupId] = collapsed;
    const button = Array.from(filterHost.querySelectorAll('[data-portfolio-filter-toggle]'))
      .find((toggle) => toggle.dataset.portfolioFilterToggle === groupId);
    if (!button) return;
    const group = button.closest('.portfolio-filter-group');
    const options = document.getElementById(button.getAttribute('aria-controls'));
    if (!group || !options) return;
    group.classList.toggle('is-collapsed', collapsed);
    button.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    options.setAttribute('aria-hidden', collapsed ? 'true' : 'false');
    options.querySelectorAll('input').forEach((input) => {
      input.disabled = collapsed;
    });
  };

  const projectMatchesFilters = (project) => {
    if (state.search) {
      const terms = state.search.split(/\s+/).map((term) => term.trim()).filter(Boolean);
      if (terms.some((term) => !hasTerm(project, term))) return false;
    }
    return filterGroups.every((group) => {
      const active = state.filters[group.id];
      if (!active || !active.size) return true;
      return group.options.some((option) => active.has(option.value) && option.match(project));
    });
  };

  const sortProjects = (projects) => {
    const copy = [...projects];
    if (state.sort === 'title') {
      return copy.sort((a, b) => String(a.title || '').localeCompare(String(b.title || '')));
    }
    return copy;
  };

  const filteredProjects = () => sortProjects(allProjects.filter(projectMatchesFilters));

  const renderResults = (projects) => {
    resultHost.innerHTML = projects.map((project, index) => {
      const media = project.image ? buildResponsivePicture(project.image, `Preview of ${project.title}`, {
        width: project.imageWidth,
        height: project.imageHeight,
        loading: 'lazy',
        decoding: 'async',
        draggable: false,
        sizes: '(max-width: 820px) 92vw, 280px'
      }) : '';
      const focuses = getProjectFocuses(project);
      const visibleChips = unique([getPrimaryFormat(project), ...focuses, ...(Array.isArray(project.tools) ? project.tools : [])]);
      return `
        <button type="button" class="portfolio-result-card${project.id === state.selectedId ? ' is-selected' : ''}" role="listitem" data-project-id="${escapeHtml(project.id)}" aria-pressed="${project.id === state.selectedId ? 'true' : 'false'}">
          <span class="portfolio-result-card__media" aria-hidden="true">${media}</span>
          <span class="portfolio-result-card__body">
            <span class="portfolio-result-card__title">${escapeHtml(project.title)}</span>
            <span class="portfolio-result-card__summary">${escapeHtml(getSummary(project))}</span>
            ${chipMarkup(visibleChips, 4)}
            <span class="portfolio-result-meta">
              <span>${calendarIcon}${escapeHtml(projectSignalLabel(project, index, 'Project'))}</span>
              <span>${formatIcon}${escapeHtml(getPrimaryFormat(project))}</span>
            </span>
          </span>
        </button>
      `;
    }).join('');
  };

  const renderInspector = (project) => {
    if (!project) {
      inspector.innerHTML = '<div class="portfolio-inspector__loading">Choose a project to see details.</div>';
      return;
    }
    const tools = Array.isArray(project.tools) ? project.tools : [];
    const focuses = getProjectFocuses(project);
    const resultHighlights = Array.isArray(project.results) ? project.results.filter(Boolean) : [];
    const actionItems = Array.isArray(project.actions) ? project.actions.filter(Boolean) : [];
    const highlights = resultHighlights.length ? resultHighlights : actionItems;
    const approachItems = resultHighlights.length ? actionItems : [];
    const summary = project.problem || project.notes || project.subtitle || getSummary(project);
    inspector.innerHTML = `
      <div class="portfolio-inspector__head">
        <div>
          <h2 class="portfolio-inspector__title">${escapeHtml(project.title)}</h2>
          <div class="portfolio-inspector__rule" aria-hidden="true"></div>
        </div>
        <button type="button" class="portfolio-inspector__close" data-portfolio-inspector-close aria-label="Close project details">Close</button>
      </div>
      <div class="portfolio-inspector__type">${escapeHtml(getPrimaryFormat(project))}</div>
      <section class="portfolio-inspector__section">
        <h3 class="portfolio-inspector__section-title">Summary</h3>
        <p class="portfolio-inspector__copy">${escapeHtml(summary)}</p>
      </section>
      ${highlights.length ? `
        <section class="portfolio-inspector__section">
          <h3 class="portfolio-inspector__section-title">Highlights</h3>
          <ul class="portfolio-inspector__list">
            ${highlights.map((item) => `<li>${listIcon}<span>${escapeHtml(item)}</span></li>`).join('')}
          </ul>
        </section>
      ` : ''}
      ${approachItems.length ? `
        <section class="portfolio-inspector__section">
          <h3 class="portfolio-inspector__section-title">Approach</h3>
          <ul class="portfolio-inspector__list">
            ${approachItems.map((item) => `<li>${listIcon}<span>${escapeHtml(item)}</span></li>`).join('')}
          </ul>
        </section>
      ` : ''}
      <section class="portfolio-inspector__section">
        <h3 class="portfolio-inspector__section-title">Tools & Stack</h3>
        ${chipMarkup(unique([...tools, ...focuses]), 12, false)}
      </section>
      <a class="portfolio-inspector__cta" href="${escapeHtml(getProjectHref(project))}">View case study <span aria-hidden="true">-&gt;</span></a>
    `;
  };

  const renderSelection = () => {
    const selectedProject = allProjects.find((project) => project.id === state.selectedId);
    root.classList.toggle('has-selected-project', Boolean(selectedProject));
    resultHost.querySelectorAll('[data-project-id]').forEach((card) => {
      const selected = card.dataset.projectId === state.selectedId;
      card.classList.toggle('is-selected', selected);
      card.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    renderInspector(selectedProject);
  };

  const clearSelection = () => {
    if (!state.selectedId) return;
    state.selectedId = null;
    pendingSelectionScrollTop = null;
    renderSelection();
  };

  const isMobileSelectionCard = () => root.dataset.mobileFilters === 'true' && window.matchMedia('(max-width: 820px)').matches;

  const restoreResultScroll = (scrollTop) => {
    resultHost.scrollTop = scrollTop;
    window.requestAnimationFrame(() => {
      resultHost.scrollTop = scrollTop;
    });
  };

  const updateClearButton = () => {
    if (!clearButton) return;
    const hasFilters = filterGroups.some((group) => state.filters[group.id] && state.filters[group.id].size) || !!state.search;
    clearButton.disabled = !hasFilters;
  };

  const render = () => {
    if (sortSelect && sortSelect.value !== state.sort) state.sort = sortSelect.value;
    renderFilters();
    const projects = filteredProjects();
    if (!projects.some((project) => project.id === state.selectedId)) {
      state.selectedId = null;
    }
    renderResults(projects);
    renderSelection();
    countNode.textContent = `${projects.length} project${projects.length === 1 ? '' : 's'}`;
    if (emptyState) emptyState.hidden = projects.length > 0;
    updateClearButton();
    try {
      srStatus().textContent = `Showing ${projects.length} project${projects.length === 1 ? '' : 's'}.`;
    } catch {}
  };

  filterHost.addEventListener('click', (event) => {
    const button = event.target.closest('[data-portfolio-filter-toggle]');
    if (!button) return;
    const groupId = button.dataset.portfolioFilterToggle;
    if (!Object.prototype.hasOwnProperty.call(state.collapsedFilters, groupId)) return;
    setFilterGroupCollapsed(groupId, !state.collapsedFilters[groupId]);
  });

  filterHost.addEventListener('change', (event) => {
    const input = event.target;
    if (!input || input.type !== 'checkbox') return;
    const group = state.filters[input.name];
    if (!group) return;
    if (input.checked) group.add(input.value);
    else group.delete(input.value);
    render();
  });

  inspector.addEventListener('click', (event) => {
    const closeButton = event.target.closest('[data-portfolio-inspector-close]');
    if (!closeButton) return;
    event.preventDefault();
    clearSelection();
  });

  document.addEventListener('pointerdown', (event) => {
    if (!state.selectedId || !isMobileSelectionCard()) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (inspector.contains(target) || target.closest('[data-project-id]')) return;
    clearSelection();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !state.selectedId) return;
    if (root.classList.contains('is-filter-sheet-open')) return;
    clearSelection();
  });

  resultHost.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('[data-project-id]')) return;
    pendingSelectionScrollTop = resultHost.scrollTop;
  });

  resultHost.addEventListener('mousedown', (event) => {
    if (!event.target.closest('[data-project-id]')) return;
    event.preventDefault();
  });

  resultHost.addEventListener('click', (event) => {
    const card = event.target.closest('[data-project-id]');
    if (!card) return;
    if (card.dataset.projectId === state.selectedId) {
      pendingSelectionScrollTop = null;
      return;
    }
    const scrollTop = pendingSelectionScrollTop === null ? resultHost.scrollTop : pendingSelectionScrollTop;
    pendingSelectionScrollTop = null;
    state.selectedId = card.dataset.projectId;
    renderSelection();
    restoreResultScroll(scrollTop);
  });

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      state.search = searchInput.value.trim();
      render();
    });
  }

  if (sortSelect) {
    sortSelect.addEventListener('change', () => {
      state.sort = sortSelect.value;
      render();
    });
  }

  if (clearButton) {
    clearButton.addEventListener('click', () => {
      Object.values(state.filters).forEach((set) => set.clear());
      state.search = '';
      if (searchInput) searchInput.value = '';
      render();
    });
  }

  render();
  return true;
}

function buildPortfolioWorkbench() {
  const root = document.querySelector('[data-portfolio-workbench]');
  const directoryConfig = window.DIRECTORY_WORKBENCH && Array.isArray(window.DIRECTORY_WORKBENCH.items)
    ? window.DIRECTORY_WORKBENCH
    : null;
  if (!root || (!window.PROJECTS && !directoryConfig)) return false;

  const filterHost = root.querySelector('[data-portfolio-filters]');
  const resultHost = root.querySelector('[data-portfolio-results]');
  const inspector = root.querySelector('[data-portfolio-inspector]');
  const countNode = root.querySelector('[data-portfolio-results-count]');
  const scopeToggle = root.querySelector('[data-portfolio-scope-toggle]');
  const clearButton = root.querySelector('[data-portfolio-clear-filters]');
  const searchInput = root.querySelector('[data-portfolio-search]');
  const sortSelect = root.querySelector('[data-portfolio-sort]');
  const emptyState = root.querySelector('[data-portfolio-empty]');
  if (!filterHost || !resultHost || !inspector || !countNode) return false;

  const isDirectoryWorkbench = Boolean(directoryConfig);
  const itemSingular = directoryConfig && directoryConfig.itemSingular ? directoryConfig.itemSingular : 'project';
  const itemPlural = directoryConfig && directoryConfig.itemPlural ? directoryConfig.itemPlural : 'projects';
  const queryParam = directoryConfig && directoryConfig.queryParam ? directoryConfig.queryParam : 'project';
  const ctaLabel = directoryConfig && directoryConfig.ctaLabel ? directoryConfig.ctaLabel : 'View case study';
  const signalPrefix = directoryConfig && directoryConfig.itemSignalPrefix ? directoryConfig.itemSignalPrefix : 'Project';
  const summaryTitle = directoryConfig && directoryConfig.summaryTitle ? directoryConfig.summaryTitle : 'Problem';
  const highlightsTitle = directoryConfig && directoryConfig.highlightsTitle ? directoryConfig.highlightsTitle : 'Outcome';
  const approachTitle = directoryConfig && directoryConfig.approachTitle ? directoryConfig.approachTitle : 'How it works';
  const stackTitle = directoryConfig && directoryConfig.stackTitle ? directoryConfig.stackTitle : 'Stack';
  const emptySelectionText = directoryConfig && directoryConfig.emptySelectionText
    ? directoryConfig.emptySelectionText
    : 'Choose a project to see details.';

  const sourceProjects = isDirectoryWorkbench
    ? directoryConfig.items.filter((item) => item && item.id)
    : (Array.isArray(window.PROJECTS) ? window.PROJECTS : []).filter(isPublishedProject);
  if (!sourceProjects.length) return true;

  const directoryKind = isDirectoryWorkbench ? String(directoryConfig.kind || '').trim().toLowerCase() : '';
  const getDirectoryAuthContext = () => {
    const authApi = window.ToolsAuth || {};
    const auth = typeof authApi.getAuth === 'function' ? authApi.getAuth() : null;
    const authed = typeof authApi.authIsValid === 'function' ? authApi.authIsValid(auth) : false;
    const admin = authed && typeof authApi.isAdmin === 'function' ? authApi.isAdmin(auth) : false;
    return { authed: !!authed, admin: !!admin };
  };
  const isDirectoryItemVisible = (item = {}) => {
    if (!isDirectoryWorkbench || directoryKind !== 'tools') return true;
    const rule = String(item.visibility || 'public').trim().toLowerCase();
    if (!rule || rule === 'public') return true;
    const context = getDirectoryAuthContext();
    if (rule === 'authed' || rule === 'authenticated' || rule === 'logged-in') return context.authed;
    if (rule === 'admin' || rule === 'admins') return context.admin;
    return true;
  };
  const audienceKey = isDirectoryWorkbench ? null : getPortfolioAudienceKey();
  const activeAudience = isDirectoryWorkbench ? null : getAudienceConfig(audienceKey || 'personal');
  const activeAudienceKey = activeAudience ? normalizeAudience(activeAudience.key || audienceKey || 'personal') : 'personal';
  const featuredProjectIds = isDirectoryWorkbench ? [] : getFeaturedProjectIds(activeAudienceKey);
  const featuredProjectIdSet = new Set(featuredProjectIds);
  const featuredProjectRank = new Map(featuredProjectIds.map((id, index) => [id, index]));
  const toList = (values = []) => {
    if (Array.isArray(values)) return values.filter(Boolean);
    return values ? [values] : [];
  };
  const isAudienceScopedView = !isDirectoryWorkbench && activeAudienceKey !== 'personal';
  const projectMatchesAudienceScope = (project = {}) => {
    if (!isAudienceScopedView) return true;
    if (featuredProjectIdSet.has(project.id)) return true;
    return toList(project.audiences).some((audience) => normalizeAudience(audience) === activeAudienceKey);
  };
  const getVisibleProjects = (applyAudienceScope = true) => {
    const visible = sourceProjects.filter(isDirectoryItemVisible);
    return applyAudienceScope && isAudienceScopedView
      ? visible.filter(projectMatchesAudienceScope)
      : visible;
  };
  let allProjects = getVisibleProjects(true);

  const portfolioProofPoints = {
    analytics: [
      { value: '99%', label: 'faster reporting' },
      { value: '200+', label: 'hours saved annually' },
      { value: '24%', label: 'inventory loss reduction' },
      { value: '57.6%', label: 'reporting lift' }
    ],
    'data-science': [
      { value: '95%', label: 'workflow time cut' },
      { value: '10x', label: 'serial tracking coverage' },
      { value: '98%', label: 'anomaly precision' },
      { value: '+14.13%', label: 'pageviews per user' }
    ],
    tourism: [
      { value: '99%', label: 'faster reporting' },
      { value: '+23.3%', label: 'listing pageview growth' },
      { value: '+9.4%', label: 'organic sessions' },
      { value: '10x', label: 'AI referral growth' }
    ]
  };

  const hydratePortfolioBrand = () => {
    if (isDirectoryWorkbench) return;
    root.dataset.portfolioAudience = activeAudienceKey;
    if (document.body) document.body.dataset.audience = activeAudienceKey;

    const titleNode = root.querySelector('[data-portfolio-brand-title]');
    const descriptionNode = root.querySelector('[data-portfolio-brand-description]');
    const resumeLink = root.querySelector('[data-portfolio-resume-link]');
    const proofHost = root.querySelector('[data-portfolio-proof]');

    if (titleNode) titleNode.textContent = activeAudience.portfolioTitle || 'Project Library';
    if (descriptionNode) {
      descriptionNode.textContent = activeAudience.portfolioDescription || 'Projects, tools, experiments, and applied data work.';
    }
    if (resumeLink) {
      const resumePath = activeAudience.resumePath || '';
      resumeLink.hidden = !resumePath;
      if (resumePath) resumeLink.href = resumePath.replace(/^\//, '');
    }
    root.querySelectorAll('[data-portfolio-audience-link]').forEach((link) => {
      const linkKey = normalizeAudience(link.dataset.portfolioAudienceLink || 'personal');
      const active = linkKey === activeAudienceKey;
      link.classList.toggle('is-active', active);
      link.setAttribute('aria-current', active ? 'page' : 'false');
    });
    if (proofHost) {
      const points = portfolioProofPoints[activeAudienceKey] || [
        { value: `${allProjects.length}`, label: 'project builds' },
        { value: `${featuredProjectIds.length || 5}`, label: 'featured systems' },
        { value: 'SQL', label: 'Python and BI' },
        { value: 'Live', label: 'demos and case studies' }
      ];
      proofHost.innerHTML = points.map((point) => `
        <li>
          <strong>${escapeHtml(point.value)}</strong>
          <span>${escapeHtml(point.label)}</span>
        </li>
      `).join('');
    }
  };
  hydratePortfolioBrand();

  const normalizeText = (value = '') => String(value || '').toLowerCase();
  const slugify = (value = '') => normalizeText(value).replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const truncate = (value = '', max = 150) => {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (text.length <= max) return text;
    return `${text.slice(0, Math.max(0, max - 3)).trim()}...`;
  };
  const listText = (values = []) => toList(values).join(' ');
  const unique = (values = []) => Array.from(new Set(values.filter(Boolean)));
  const prioritySorter = (priority = [], countForLabel = () => 0) => {
    const rank = new Map(priority.map((label, index) => [slugify(label), index]));
    return (a, b) => {
      const aRank = rank.has(slugify(a)) ? rank.get(slugify(a)) : Infinity;
      const bRank = rank.has(slugify(b)) ? rank.get(slugify(b)) : Infinity;
      if (aRank !== bRank) return aRank - bRank;
      const countDelta = countForLabel(b) - countForLabel(a);
      if (countDelta) return countDelta;
      return String(a).localeCompare(String(b));
    };
  };
  const hasTerm = (project, term) => {
    const haystack = normalizeText([
      project.title,
      project.subtitle,
      project.summary,
      project.notes,
      project.problem,
      project.category,
      project.type,
      listText(project.tags),
      listText(project.formats),
      listText(project.tools),
      listText(project.concepts),
      listText(project.audiences),
      listText(project.results),
      listText(project.actions)
    ].join(' '));
    return haystack.includes(normalizeText(term));
  };
  const fieldValues = (project = {}, field = '') => toList(project[field]);
  const optionMatchesField = (option, project) => {
    if (!option || !option.field) return false;
    return fieldValues(project, option.field).some((value) => {
      const normalized = slugify(value);
      return normalized === slugify(option.value) || normalized === slugify(option.label);
    });
  };
  const getProjectHref = (project = {}) => project.href || `portfolio/${encodeURIComponent(project.id)}`;
  const getProjectFormats = (project = {}) => {
    if (Array.isArray(project.formats) && project.formats.length) return unique(project.formats);
    if (project.type) return [project.type];
    const formats = ['Case Study'];
    const resourceText = Array.isArray(project.resources)
      ? project.resources.map((resource) => resource && resource.label).join(' ')
      : '';
    const text = normalizeText(`${project.title} ${project.subtitle} ${resourceText} ${listText(project.tools)}`);
    if (project.embed || text.includes('live demo') || text.includes('interactive')) formats.push('Interactive Demo');
    if (text.includes('dashboard') || text.includes('tableau') || text.includes('bi')) formats.push('Dashboard');
    if (text.includes('notebook') || text.includes('pdf') || text.includes('excel')) formats.push('Notebook / Report');
    if (text.includes('demo') || text.includes('retriever') || text.includes('solver') || text.includes('generator')) formats.push('Tool / Library');
    return unique(formats);
  };
  const getProjectFocuses = (project = {}) => {
    const focuses = Array.isArray(project.concepts) ? [...project.concepts] : [];
    if (project.category) focuses.push(project.category);
    if (getProjectFormats(project).includes('Dashboard')) focuses.push('Dashboards');
    if (hasTerm(project, 'nlp') || hasTerm(project, 'chatbot') || hasTerm(project, 'semantic')) focuses.push('NLP');
    if (hasTerm(project, 'etl') || hasTerm(project, 'sql') || hasTerm(project, 'pipeline')) focuses.push('Data Engineering');
    return unique(focuses);
  };
  const getPrimaryFormat = (project = {}) => {
    if (project.type) return project.type;
    const formats = getProjectFormats(project);
    if (formats.includes('Dashboard')) return 'Dashboard';
    if (formats.includes('Interactive Demo')) return 'Interactive Demo';
    if (formats.includes('Tool / Library')) return 'Tool / Library';
    return formats[0] || 'Case Study';
  };
  const getSummary = (project = {}) => truncate(
    project.summary ||
    (Array.isArray(project.results) && project.results[0]) ||
    project.problem ||
    project.notes ||
    project.subtitle ||
    '',
    140
  );
  const filterOptionProjects = isAudienceScopedView ? getVisibleProjects(false) : allProjects;
  const focusOptionLabels = unique(filterOptionProjects.flatMap((project) => getProjectFocuses(project)))
    .sort(prioritySorter(
      ['Analytics', 'Machine Learning', 'Automation', 'Dashboards', 'Visualization', 'Data Engineering', 'NLP', 'Product'],
      (label) => filterOptionProjects.filter((project) => getProjectFocuses(project).includes(label)).length
    ));
  const stackOptionLabels = unique(filterOptionProjects.flatMap((project) => toList(project.tools)))
    .sort(prioritySorter(
      ['Python', 'AWS', 'Docker', 'SQL', 'Excel', 'Tableau', 'PyTorch', 'JavaScript'],
      (label) => filterOptionProjects.filter((project) => toList(project.tools).some((tool) => slugify(tool) === slugify(label))).length
    ));
  const chipMarkup = (values = [], limit = 4, accentFirst = true) => {
    const chips = values.filter(Boolean).slice(0, limit);
    if (!chips.length) return '';
    return `<div class="portfolio-chip-row">${chips.map((value, index) => {
      const accent = accentFirst && index === 0 ? ' portfolio-chip--accent' : '';
      return `<span class="portfolio-chip${accent}">${escapeHtml(value)}</span>`;
    }).join('')}</div>`;
  };
  const calendarIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 2v4"></path><path d="M16 2v4"></path><rect x="3" y="5" width="18" height="16" rx="2"></rect><path d="M3 10h18"></path></svg>';
  const formatIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V5"></path><path d="M4 19h16"></path><path d="M8 15l3-3 3 2 5-7"></path></svg>';
  const listIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"></path><path d="M4 12h16"></path><path d="M4 17h16"></path></svg>';

  const configuredFilterGroups = isDirectoryWorkbench
    ? toList(directoryConfig.filterGroups).map((group) => ({
      id: group.id,
      title: group.title,
      options: toList(group.options).map((option) => ({
        value: option.value || slugify(option.label),
        label: option.label,
        field: option.field,
        match: (project) => optionMatchesField(option, project)
      })).filter((option) => option.value && option.label)
    })).filter((group) => group.id && group.title && group.options.length)
    : [];
  const filterGroups = configuredFilterGroups.length ? configuredFilterGroups : [
    {
      id: 'focus',
      title: 'Focus',
      options: focusOptionLabels.map((label) => ({
        value: slugify(label),
        label,
        match: (project) => getProjectFocuses(project).some((focus) => slugify(focus) === slugify(label))
      }))
    },
    {
      id: 'stack',
      title: 'Stack',
      options: stackOptionLabels.map((label) => ({
        value: slugify(label),
        label,
        match: (project) => toList(project.tools).some((tool) => slugify(tool) === slugify(label))
      }))
    },
    {
      id: 'format',
      title: 'Format',
      options: [
        { value: 'case-study', label: 'Case Study', match: (project) => getProjectFormats(project).includes('Case Study') },
        { value: 'interactive-demo', label: 'Interactive Demo', match: (project) => getProjectFormats(project).includes('Interactive Demo') },
        { value: 'dashboard', label: 'Dashboard', match: (project) => getProjectFormats(project).includes('Dashboard') },
        { value: 'notebook-report', label: 'Notebook / Report', match: (project) => getProjectFormats(project).includes('Notebook / Report') },
        { value: 'tool-library', label: 'Tool / Library', match: (project) => getProjectFormats(project).includes('Tool / Library') }
      ]
    }
  ];

  const initialProjectId = (() => {
    try {
      const params = new URLSearchParams(window.location.search || '');
      const queryId = params.get(queryParam);
      if (queryId) return queryId;
    } catch {}
    if (location.hash && location.hash.length > 1) return decodeURIComponent(location.hash.slice(1));
    return null;
  })();
  const initialProjectInScopedPool = allProjects.some((project) => project.id === initialProjectId);
  const initialProjectInFullPool = getVisibleProjects(false).some((project) => project.id === initialProjectId);
  const state = {
    filters: Object.fromEntries(filterGroups.map((group) => [group.id, new Set()])),
    collapsedFilters: Object.fromEntries(filterGroups.map((group) => [group.id, false])),
    search: '',
    sort: sortSelect ? sortSelect.value : 'default',
    showAllAudienceProjects: isAudienceScopedView && Boolean(initialProjectId) && !initialProjectInScopedPool && initialProjectInFullPool,
    selectedId: initialProjectInScopedPool || initialProjectInFullPool ? initialProjectId : null
  };
  let pendingSelectionScrollTop = null;
  const mobileSelectionOverlayEnabled = !isDirectoryWorkbench && root.id === 'portfolio-workbench';
  const mobileFilterSheetEnabled = mobileSelectionOverlayEnabled || (isDirectoryWorkbench && directoryKind === 'games');
  if (mobileSelectionOverlayEnabled) root.dataset.mobileSelection = 'overlay';
  const mobileFilters = setupPortfolioMobileFilterSheet({
    enabled: mobileFilterSheetEnabled,
    root,
    filterHost,
    filterGroups,
    state,
    allItems: filterOptionProjects,
    sortSelect,
    searchInput,
    itemSingular,
    itemPlural,
    slugify,
    requestRender: () => render()
  });

  const renderFilters = () => {
    const counts = new Map();
    filterGroups.forEach((group) => {
      group.options.forEach((option) => {
        counts.set(`${group.id}:${option.value}`, allProjects.filter(option.match).length);
      });
    });
    filterHost.innerHTML = filterGroups.map((group) => {
      const collapsed = !!state.collapsedFilters[group.id];
      const optionsId = `portfolio-filter-options-${escapeHtml(group.id)}`;
      return `
      <fieldset class="portfolio-filter-group${collapsed ? ' is-collapsed' : ''}">
        <legend class="portfolio-filter-group__legend">
          <button type="button" class="portfolio-filter-group__toggle" aria-expanded="${collapsed ? 'false' : 'true'}" aria-controls="${optionsId}" data-portfolio-filter-toggle="${escapeHtml(group.id)}">
            <span class="portfolio-filter-group__title">${escapeHtml(group.title)}</span>
            <span class="portfolio-filter-group__chevron" aria-hidden="true"></span>
          </button>
        </legend>
        <div class="portfolio-filter-options" id="${optionsId}" aria-hidden="${collapsed ? 'true' : 'false'}">
          <div class="portfolio-filter-options__inner">
            ${group.options.map((option) => `
              <label class="portfolio-filter-option">
                <input type="checkbox" name="${escapeHtml(group.id)}" value="${escapeHtml(option.value)}"${state.filters[group.id].has(option.value) ? ' checked' : ''}${collapsed ? ' disabled' : ''}>
                <span>${escapeHtml(option.label)}</span>
                <span class="portfolio-filter-option__count">${counts.get(`${group.id}:${option.value}`) || 0}</span>
              </label>
            `).join('')}
          </div>
        </div>
      </fieldset>
    `;
    }).join('');
  };

  const setFilterGroupCollapsed = (groupId, collapsed) => {
    state.collapsedFilters[groupId] = collapsed;
    const button = Array.from(filterHost.querySelectorAll('[data-portfolio-filter-toggle]'))
      .find((toggle) => toggle.dataset.portfolioFilterToggle === groupId);
    if (!button) return;
    const group = button.closest('.portfolio-filter-group');
    const options = document.getElementById(button.getAttribute('aria-controls'));
    if (!group || !options) return;
    group.classList.toggle('is-collapsed', collapsed);
    button.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    options.setAttribute('aria-hidden', collapsed ? 'true' : 'false');
    options.querySelectorAll('input').forEach((input) => {
      input.disabled = collapsed;
    });
  };

  const projectMatchesFilters = (project) => {
    if (state.search) {
      const terms = state.search.split(/\s+/).map((term) => term.trim()).filter(Boolean);
      if (terms.some((term) => !hasTerm(project, term))) return false;
    }
    return filterGroups.every((group) => {
      const active = state.filters[group.id];
      if (!active || !active.size) return true;
      return group.options.some((option) => active.has(option.value) && option.match(project));
    });
  };

  const sortProjects = (projects) => {
    const copy = [...projects];
    if (state.sort === 'title') {
      return copy.sort((a, b) => String(a.title || '').localeCompare(String(b.title || '')));
    }
    if (featuredProjectRank.size) {
      return copy.sort((a, b) => {
        const aRank = featuredProjectRank.has(a.id) ? featuredProjectRank.get(a.id) : Infinity;
        const bRank = featuredProjectRank.has(b.id) ? featuredProjectRank.get(b.id) : Infinity;
        if (aRank !== bRank) return aRank - bRank;
        return Number(a.order || 0) - Number(b.order || 0);
      });
    }
    return copy;
  };

  const filteredProjects = () => sortProjects(allProjects.filter(projectMatchesFilters));

  const renderWorkbenchMedia = (project = {}) => {
    if (project.image) {
      return buildResponsivePicture(project.image, `Preview of ${project.title}`, {
        width: project.imageWidth,
        height: project.imageHeight,
        loading: 'lazy',
        decoding: 'async',
        draggable: false,
        sizes: '(max-width: 820px) 92vw, 280px'
      });
    }
    if (project.iconImage) {
      return `<span class="portfolio-result-card__icon"><img src="${escapeHtml(project.iconImage)}" alt="" loading="lazy" decoding="async"></span>`;
    }
    if (project.iconHtml) {
      return `<span class="portfolio-result-card__icon">${project.iconHtml}</span>`;
    }
    const initial = String(project.title || itemSingular || '?').trim().charAt(0) || '?';
    return `<span class="portfolio-result-card__icon"><span class="portfolio-result-card__initial">${escapeHtml(initial)}</span></span>`;
  };

  const renderResults = (projects) => {
    resultHost.innerHTML = projects.map((project, index) => {
      const focuses = getProjectFocuses(project);
      const visibleChips = unique([getPrimaryFormat(project), ...focuses, ...toList(project.tools), ...toList(project.tags)]);
      const visibilityAttr = isDirectoryWorkbench && directoryKind === 'tools'
        ? ` data-tools-visibility="${escapeHtml(project.visibility || 'public')}"`
        : '';
      return `
        <button type="button" class="portfolio-result-card${project.id === state.selectedId ? ' is-selected' : ''}" role="listitem" data-project-id="${escapeHtml(project.id)}"${visibilityAttr} aria-pressed="${project.id === state.selectedId ? 'true' : 'false'}">
          <span class="portfolio-result-card__media${project.image ? '' : ' portfolio-result-card__media--icon'}" aria-hidden="true">${renderWorkbenchMedia(project)}</span>
          <span class="portfolio-result-card__body">
            <span class="portfolio-result-card__title">${escapeHtml(project.title)}</span>
            <span class="portfolio-result-card__summary">${escapeHtml(getSummary(project))}</span>
            ${chipMarkup(visibleChips, 4)}
            <span class="portfolio-result-meta">
              <span>${calendarIcon}${escapeHtml(projectSignalLabel(project, index, signalPrefix))}</span>
              <span>${formatIcon}${escapeHtml(getPrimaryFormat(project))}</span>
            </span>
          </span>
        </button>
      `;
    }).join('');
  };

  const renderInspector = (project) => {
    if (!project) {
      inspector.innerHTML = `<div class="portfolio-inspector__loading">${escapeHtml(emptySelectionText)}</div>`;
      return;
    }
    const tools = toList(project.tools);
    const tags = toList(project.tags);
    const focuses = getProjectFocuses(project);
    const summary = project.problem || project.notes || project.summary || project.subtitle || getSummary(project);
    const resultHighlights = toList(project.results).filter((item) => String(item || '').trim() !== String(summary || '').trim());
    const actionItems = toList(project.actions);
    const highlights = resultHighlights.length ? resultHighlights : actionItems;
    const approachItems = resultHighlights.length ? actionItems : [];
    inspector.innerHTML = `
      <div class="portfolio-inspector__head">
        <div>
          <h2 class="portfolio-inspector__title">${escapeHtml(project.title)}</h2>
          <div class="portfolio-inspector__rule" aria-hidden="true"></div>
        </div>
        <button type="button" class="portfolio-inspector__close" data-portfolio-inspector-close aria-label="Close project details">Close</button>
      </div>
      <div class="portfolio-inspector__type">${escapeHtml(getPrimaryFormat(project))}</div>
      <section class="portfolio-inspector__section">
        <h3 class="portfolio-inspector__section-title">${escapeHtml(summaryTitle)}</h3>
        <p class="portfolio-inspector__copy">${escapeHtml(summary)}</p>
      </section>
      ${highlights.length ? `
        <section class="portfolio-inspector__section">
          <h3 class="portfolio-inspector__section-title">${escapeHtml(highlightsTitle)}</h3>
          <ul class="portfolio-inspector__list">
            ${highlights.map((item) => `<li>${listIcon}<span>${escapeHtml(item)}</span></li>`).join('')}
          </ul>
        </section>
      ` : ''}
      ${approachItems.length ? `
        <section class="portfolio-inspector__section">
          <h3 class="portfolio-inspector__section-title">${escapeHtml(approachTitle)}</h3>
          <ul class="portfolio-inspector__list">
            ${approachItems.map((item) => `<li>${listIcon}<span>${escapeHtml(item)}</span></li>`).join('')}
          </ul>
        </section>
      ` : ''}
      <section class="portfolio-inspector__section">
        <h3 class="portfolio-inspector__section-title">${escapeHtml(stackTitle)}</h3>
        ${chipMarkup(unique([...tools, ...tags, ...focuses]), 12, false)}
      </section>
      <a class="portfolio-inspector__cta" href="${escapeHtml(getProjectHref(project))}">${escapeHtml(ctaLabel)} <span aria-hidden="true">-&gt;</span></a>
    `;
  };

  const renderSelection = () => {
    const selectedProject = allProjects.find((project) => project.id === state.selectedId);
    root.classList.toggle('has-selected-project', Boolean(selectedProject));
    resultHost.querySelectorAll('[data-project-id]').forEach((card) => {
      const selected = card.dataset.projectId === state.selectedId;
      card.classList.toggle('is-selected', selected);
      card.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    renderInspector(selectedProject);
  };

  const clearSelection = () => {
    if (!state.selectedId) return;
    state.selectedId = null;
    pendingSelectionScrollTop = null;
    renderSelection();
  };

  const isMobileSelectionCard = () => root.dataset.mobileSelection === 'overlay' && window.matchMedia('(max-width: 820px)').matches;

  const restoreResultScroll = (scrollTop) => {
    resultHost.scrollTop = scrollTop;
    window.requestAnimationFrame(() => {
      resultHost.scrollTop = scrollTop;
      setTimeout(() => {
        resultHost.scrollTop = scrollTop;
      }, 0);
    });
  };

  const updateClearButton = () => {
    if (!clearButton) return;
    const hasFilters = filterGroups.some((group) => state.filters[group.id] && state.filters[group.id].size) || !!state.search;
    clearButton.disabled = !hasFilters;
  };

  const audienceScopeLabel = () => String(
    (activeAudience && (activeAudience.shortLabel || activeAudience.label)) || activeAudienceKey || itemPlural
  ).replace(/\s+/g, ' ').trim().toLowerCase();

  const formatCountText = (projects) => {
    if (!isAudienceScopedView) {
      return `${projects.length} ${projects.length === 1 ? itemSingular : itemPlural}`;
    }
    const alignedTotal = getVisibleProjects(true).length;
    const label = audienceScopeLabel();
    if (state.showAllAudienceProjects) {
      return `${projects.length} total ${projects.length === 1 ? itemSingular : itemPlural} (${alignedTotal} ${label}-aligned)`;
    }
    return `${projects.length} ${label}-aligned ${projects.length === 1 ? itemSingular : itemPlural}`;
  };

  const updateScopeToggle = () => {
    if (!scopeToggle) return;
    scopeToggle.hidden = !isAudienceScopedView;
    if (!isAudienceScopedView) return;
    scopeToggle.textContent = state.showAllAudienceProjects
      ? `Show ${audienceScopeLabel()} only`
      : 'Show all projects';
    scopeToggle.setAttribute('aria-pressed', state.showAllAudienceProjects ? 'true' : 'false');
  };

  const render = () => {
    allProjects = getVisibleProjects(!state.showAllAudienceProjects);
    if (sortSelect && sortSelect.value !== state.sort) state.sort = sortSelect.value;
    renderFilters();
    if (!state.selectedId && initialProjectId && allProjects.some((project) => project.id === initialProjectId)) {
      state.selectedId = initialProjectId;
    }
    const projects = filteredProjects();
    if (!projects.some((project) => project.id === state.selectedId)) {
      const shouldAutoSelect = (isDirectoryWorkbench || !isMobileSelectionCard()) && projects[0];
      state.selectedId = shouldAutoSelect ? projects[0].id : null;
    }
    renderResults(projects);
    renderSelection();
    countNode.textContent = formatCountText(projects);
    if (emptyState) emptyState.hidden = projects.length > 0;
    updateClearButton();
    updateScopeToggle();
    mobileFilters.render(projects);
    try {
      srStatus().textContent = `Showing ${projects.length} ${projects.length === 1 ? itemSingular : itemPlural}.`;
    } catch {}
  };

  filterHost.addEventListener('click', (event) => {
    const button = event.target.closest('[data-portfolio-filter-toggle]');
    if (!button) return;
    const groupId = button.dataset.portfolioFilterToggle;
    if (!Object.prototype.hasOwnProperty.call(state.collapsedFilters, groupId)) return;
    setFilterGroupCollapsed(groupId, !state.collapsedFilters[groupId]);
  });

  filterHost.addEventListener('change', (event) => {
    const input = event.target;
    if (!input || input.type !== 'checkbox') return;
    const group = state.filters[input.name];
    if (!group) return;
    if (input.checked) group.add(input.value);
    else group.delete(input.value);
    render();
  });

  inspector.addEventListener('click', (event) => {
    const closeButton = event.target.closest('[data-portfolio-inspector-close]');
    if (!closeButton) return;
    event.preventDefault();
    clearSelection();
  });

  document.addEventListener('pointerdown', (event) => {
    if (!state.selectedId || !isMobileSelectionCard()) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (root.classList.contains('is-filter-sheet-open') || target.closest('[data-portfolio-filter-sheet-open], [data-portfolio-filter-sheet-toggle]')) return;
    if (inspector.contains(target) || target.closest('[data-project-id]')) return;
    clearSelection();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !state.selectedId) return;
    if (root.classList.contains('is-filter-sheet-open')) return;
    clearSelection();
  });

  resultHost.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('[data-project-id]')) return;
    pendingSelectionScrollTop = resultHost.scrollTop;
  });

  resultHost.addEventListener('mousedown', (event) => {
    if (!event.target.closest('[data-project-id]')) return;
    event.preventDefault();
  });

  resultHost.addEventListener('click', (event) => {
    const card = event.target.closest('[data-project-id]');
    if (!card) return;
    if (card.dataset.projectId === state.selectedId) {
      pendingSelectionScrollTop = null;
      return;
    }
    const scrollTop = pendingSelectionScrollTop === null ? resultHost.scrollTop : pendingSelectionScrollTop;
    pendingSelectionScrollTop = null;
    state.selectedId = card.dataset.projectId;
    renderSelection();
    restoreResultScroll(scrollTop);
  });

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      state.search = searchInput.value.trim();
      render();
    });
  }

  if (sortSelect) {
    sortSelect.addEventListener('change', () => {
      state.sort = sortSelect.value;
      render();
    });
  }

  if (clearButton) {
    clearButton.addEventListener('click', () => {
      Object.values(state.filters).forEach((set) => set.clear());
      state.search = '';
      if (searchInput) searchInput.value = '';
      render();
    });
  }

  if (scopeToggle) {
    scopeToggle.addEventListener('click', () => {
      state.showAllAudienceProjects = !state.showAllAudienceProjects;
      render();
    });
  }

  if (isDirectoryWorkbench && directoryKind === 'tools') {
    document.addEventListener('tools:auth-changed', render);
    document.addEventListener('tools:visibility-updated', render);
  }

  render();
  return true;
}

function buildPortfolio() {
  if (buildPortfolioWorkbench()) return;

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
        <div class="project-card-kicker">${escapeHtml(projectSignalLabel(project, index, 'Project'))}</div>
        <div class="project-title">${escapeHtml(project.title)}</div>
        <div class="project-subtitle">${escapeHtml(project.subtitle)}</div>
        ${projectToolsMarkup(project)}
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
