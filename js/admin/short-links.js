/* Short links admin dashboard (token-based). */
(() => {
  'use strict';

  const STORAGE_KEY = 'shortlinks_admin_token';
  const MODE_STORAGE_KEY = 'shortlinks_active_mode';
  const VIEWS_STORAGE_KEY = 'shortlinks_saved_views';
  const DEFAULT_BASE_PATH = 'go';
  const DEFAULT_PUBLIC_ORIGIN = 'https://dshort.me';
  const TABLE_LAYOUT_MEDIA_QUERY = '(min-width: 900px)';

  const authForm = document.querySelector('[data-shortlinks="auth"]');
  const editorForm = document.querySelector('[data-shortlinks="editor"]');
  const listEl = document.querySelector('[data-shortlinks="list"]');
  if (!authForm || !editorForm || !listEl) return;

  const projectsListEl = document.querySelector('[data-shortlinks="projects-list"]');
  const projectsStatusEl = document.querySelector('[data-shortlinks="projects-status"]');
  const projectsMetaEl = document.querySelector('[data-shortlinks="projects-meta"]');
  const projectsRefreshButton = document.querySelector('[data-shortlinks="projects-refresh"]');
  const projectsEnsureButton = document.querySelector('[data-shortlinks="projects-ensure"]');

  const accessCard = document.querySelector('[data-shortlinks="access-card"]');
  const adminAccessSummaryEl = document.querySelector('[data-shortlinks="admin-access-summary"]');
  const adminProjectSummaryEl = document.querySelector('[data-shortlinks="admin-project-summary"]');
  const adminExportSummaryEl = document.querySelector('[data-shortlinks="admin-export-summary"]');
  const accessMetaEl = document.querySelector('[data-shortlinks="access-meta"]');
  const filterInput = document.querySelector('[data-shortlinks="filter"]');
  const statusFilterSelect = document.querySelector('[data-shortlinks="status-filter"]');
  const sortSelect = document.querySelector('[data-shortlinks="sort"]');
  const densitySelect = document.querySelector('[data-shortlinks="density"]');
  const newLinkFromListButton = document.querySelector('[data-shortlinks="new-link-from-list"]');
  const healthStripEl = document.querySelector('[data-shortlinks="health-strip"]');
  const savedViewSelect = document.querySelector('[data-shortlinks="saved-view"]');
  const saveViewButton = document.querySelector('[data-shortlinks="save-view"]');
  const deleteViewButton = document.querySelector('[data-shortlinks="delete-view"]');
  const selectVisibleInput = document.querySelector('[data-shortlinks="select-visible"]');
  const selectionCountEl = document.querySelector('[data-shortlinks="selection-count"]');
  const testSelectedButton = document.querySelector('[data-shortlinks="test-selected"]');
  const exportViewButton = document.querySelector('[data-shortlinks="export-view"]');
  const clearSelectionButton = document.querySelector('[data-shortlinks="clear-selection"]');
  const detailPanelEl = document.querySelector('[data-shortlinks="detail-panel"]');
  const exportModeSelect = document.querySelector('[data-shortlinks="export-mode"]');
  const exportClickLimitInput = document.querySelector('[data-shortlinks="export-click-limit"]');
  const exportButton = document.querySelector('[data-shortlinks="export"]');
  const countEl = document.querySelector('[data-shortlinks="count"]');
  const listStatusEl = document.querySelector('[data-shortlinks="list-status"]');
  const summaryEl = document.querySelector('[data-shortlinks="summary"]');
  const modeTabEls = Array.from(document.querySelectorAll('[data-shortlinks="mode-tab"]'));
  const modePanelEls = Array.from(document.querySelectorAll('[data-shortlinks-mode-panel]'));
  const modeSummaryEl = document.querySelector('[data-shortlinks="mode-summary"]');

  const tokenInput = authForm.querySelector('[data-shortlinks="token"]');
  const rememberTokenInput = authForm.querySelector('[data-shortlinks="remember-token"]');
  const refreshButton = authForm.querySelector('[data-shortlinks="refresh"]');
  const healthButton = authForm.querySelector('[data-shortlinks="health"]');
  const forgetButton = authForm.querySelector('[data-shortlinks="forget-token"]');
  const statusEl = authForm.querySelector('[data-shortlinks="status"]');
  const healthStatusEl = authForm.querySelector('[data-shortlinks="health-status"]');

  const slugInput = editorForm.querySelector('[data-shortlinks="slug"]');
  const slugModeSelect = editorForm.querySelector('[data-shortlinks="slug-mode"]');
  const slugFieldEl = editorForm.querySelector('[data-shortlinks="slug-field"]');
  const randomLengthFieldEl = editorForm.querySelector('[data-shortlinks="random-length-field"]');
  const randomLengthInput = editorForm.querySelector('[data-shortlinks="random-length"]');
  const destinationInput = editorForm.querySelector('[data-shortlinks="destination"]');
  const audienceFieldEl = editorForm.querySelector('[data-shortlinks="audience-field"]');
  const audienceSelect = editorForm.querySelector('[data-shortlinks="audience"]');
  const expirationModeSelect = editorForm.querySelector('[data-shortlinks="expiration-mode"]');
  const expirationDurationFields = editorForm.querySelector('[data-shortlinks="expiration-duration-fields"]');
  const expirationDurationValueInput = editorForm.querySelector('[data-shortlinks="expiration-duration-value"]');
  const expirationDurationUnitSelect = editorForm.querySelector('[data-shortlinks="expiration-duration-unit"]');
  const createLinkButton = editorForm.querySelector('[data-shortlinks="create-link"]');
  const getPermanentButton = editorForm.querySelector('[data-shortlinks="get-permanent"]');
  const getTemporaryButton = editorForm.querySelector('[data-shortlinks="get-temporary"]');
  const clearButton = editorForm.querySelector('[data-shortlinks="clear"]');
  const editorStatusEl = editorForm.querySelector('[data-shortlinks="editor-status"]');
  const editorMetaEl = document.querySelector('[data-shortlinks="editor-meta"]');

  const setsFilterInput = document.querySelector('[data-shortlinks="sets-filter"]');
  const setsRefreshButton = document.querySelector('[data-shortlinks="sets-refresh"]');
  const setsNewButton = document.querySelector('[data-shortlinks="set-new"]');
  const setsStatusEl = document.querySelector('[data-shortlinks="sets-status"]');
  const setsListEl = document.querySelector('[data-shortlinks="sets-list"]');
  const setEditorForm = document.querySelector('[data-shortlinks="set-editor"]');
  const setTitleInput = document.querySelector('[data-shortlinks="set-title"]');
  const setDefaultRandomLengthInput = document.querySelector('[data-shortlinks="set-default-random-length"]');
  const setDefaultExpirationModeSelect = document.querySelector('[data-shortlinks="set-default-expiration-mode"]');
  const setDefaultDurationFields = document.querySelector('[data-shortlinks="set-default-duration-fields"]');
  const setDefaultDurationValueInput = document.querySelector('[data-shortlinks="set-default-duration-value"]');
  const setDefaultDurationUnitSelect = document.querySelector('[data-shortlinks="set-default-duration-unit"]');
  const setRowsEl = document.querySelector('[data-shortlinks="set-rows"]');
  const setAddRowButton = document.querySelector('[data-shortlinks="set-add-row"]');
  const setDeleteButton = document.querySelector('[data-shortlinks="set-delete"]');
  const setEditorStatusEl = document.querySelector('[data-shortlinks="set-editor-status"]');
  const setGenerateForm = document.querySelector('[data-shortlinks="set-generate"]');
  const batchTitleInput = document.querySelector('[data-shortlinks="batch-title"]');
  const batchRandomLengthInput = document.querySelector('[data-shortlinks="batch-random-length"]');
  const batchExpirationModeSelect = document.querySelector('[data-shortlinks="batch-expiration-mode"]');
  const batchDurationFields = document.querySelector('[data-shortlinks="batch-duration-fields"]');
  const batchDurationValueInput = document.querySelector('[data-shortlinks="batch-duration-value"]');
  const batchDurationUnitSelect = document.querySelector('[data-shortlinks="batch-duration-unit"]');
  const batchStatusEl = document.querySelector('[data-shortlinks="batch-status"]');
  const batchResultsEl = document.querySelector('[data-shortlinks="batch-results"]');

  const destinationPickerOpen = editorForm.querySelector('[data-shortlinks="destination-picker-open"]');
  const destinationModal = document.querySelector('[data-shortlinks="destination-modal"]');
  const destinationModalClose = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-modal-close"]')
    : null;
  const destinationSearch = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-search"]')
    : null;
  const destinationResults = destinationModal
    ? destinationModal.querySelector('[data-shortlinks="destination-results"]')
    : null;

  const clicksModal = document.querySelector('[data-shortlinks="clicks-modal"]');
  const clicksModalClose = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-modal-close"]')
    : null;
  const clicksSlugEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-slug"]')
    : null;
  const clicksStatusEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-status"]')
    : null;
  const clicksListEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-list"]')
    : null;
  const clicksMetaEl = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-meta"]')
    : null;
  const clicksRefreshButton = clicksModal
    ? clicksModal.querySelector('[data-shortlinks="clicks-refresh"]')
    : null;

  const temporaryModal = document.querySelector('[data-shortlinks="temporary-modal"]');
  const temporaryModalClose = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-modal-close"]')
    : null;
  const temporaryForm = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-form"]')
    : null;
  const temporaryValueInput = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-value"]')
    : null;
  const temporaryUnitSelect = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-unit"]')
    : null;
  const temporaryCancel = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-cancel"]')
    : null;
  const temporaryStatusEl = temporaryModal
    ? temporaryModal.querySelector('[data-shortlinks="temporary-status"]')
    : null;

  const DESTINATIONS_MANIFEST_PATH = 'dist/shortlinks-destinations.json';
  const CLICK_HISTORY_LIMIT = 250;
  const TOOL_ID = 'short-links';
  const MAX_SAVED_LINK_LINES = 120;
  const PROJECT_SLUG_PREFIX = 'p';
  const DEFAULT_RANDOM_LENGTH = 6;
  const MIN_RANDOM_LENGTH = 4;
  const MAX_RANDOM_LENGTH = 12;
  const DEFAULT_SET_DURATION_VALUE = 7;
  const DEFAULT_SET_DURATION_UNIT = 'days';
  const EXPORT_MODE_REDIRECTS_ONLY = 'redirects-only';
  const EXPORT_MODE_WITH_CLICKS = 'with-clicks';
  const EXPORT_DEFAULT_CLICK_LIMIT = 100;
  const EXPORT_MAX_CLICK_LIMIT = 500;
  const FALLBACK_AUDIENCES = {
    analytics: {
      key: 'analytics',
      label: 'Data Analytics',
      shortLabel: 'Analytics',
      homePath: '/analytics',
      portfolioPath: '/portfolio?audience=analytics',
      resumePath: '/resume',
      resumePreviewPath: '/resume-pdf',
      resumeDownloadPath: '/documents/Resume.pdf'
    },
    'data-science': {
      key: 'data-science',
      label: 'Data Science',
      shortLabel: 'Data Science',
      homePath: '/data-science',
      portfolioPath: '/portfolio?audience=data-science',
      resumePath: '/resume-data-science',
      resumePreviewPath: '/resume-data-science-pdf',
      resumeDownloadPath: '/documents/Resume-Data-Science.pdf'
    },
    tourism: {
      key: 'tourism',
      label: 'Tourism Analytics',
      shortLabel: 'Tourism',
      homePath: '/tourism',
      portfolioPath: '/portfolio?audience=tourism',
      resumePath: '/resume-tourism',
      resumePreviewPath: '/resume-tourism-pdf',
      resumeDownloadPath: '/documents/Resume-Tourism.pdf'
    }
  };
  const FALLBACK_AUDIENCE_ORDER = ['analytics', 'data-science', 'tourism'];
  const FALLBACK_AUDIENCE_DEFAULT = 'analytics';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const FALLBACK_DESTINATIONS = [
    { path: '/', label: 'Home', group: 'Pages' },
    { path: '/portfolio', label: 'Portfolio', group: 'Portfolio' },
    { path: '/resume', label: 'Resume', group: 'Pages' },
    { path: '/contact', label: 'Contact', group: 'Pages' },
    { path: '/tools', label: 'Tools', group: 'Tools' }
  ];

  let destinationsManifest = null;
  let destinationModalPrevFocus = null;
  let clicksModalPrevFocus = null;
  let temporaryModalPrevFocus = null;
  let activeClicksSlug = '';
  let pendingTemporaryPayload = null;
  let projectCatalog = [];
  let ensuringProjectLinks = false;
  let accessCardAttentionTimer = 0;

  let basePath = DEFAULT_BASE_PATH;
  let allLinks = [];
  let visibleLinksCount = 0;
  let visibleLinkSlugs = [];
  let selectedSlugs = new Set();
  let linkHealth = new Map();
  let memorySavedViews = [];
  let activeDetailSlug = '';
  let projectHealth = { total: 0, missing: 0, mismatched: 0 };
  let allSets = [];
  let activeSetId = '';
  let setRowCounter = 0;

  function setStatus(el, msg, tone){
    if (!el) return;
    el.textContent = msg || '';
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }

  function setAdminBadge(el, message, tone){
    if (!el) return;
    el.textContent = String(message || '');
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }

  function setEditorMeta(message){
    if (!editorMetaEl) return;
    editorMetaEl.textContent = String(message || 'New short link');
  }

  function getModePanel(mode){
    const target = String(mode || '').trim().toLowerCase();
    return modePanelEls.find((panel) => String(panel?.dataset?.shortlinksModePanel || '').trim().toLowerCase() === target) || null;
  }

  function setActiveMode(mode, options = {}){
    const fallbackMode = modePanelEls[0]
      ? String(modePanelEls[0].dataset.shortlinksModePanel || 'single')
      : 'single';
    const nextMode = getModePanel(mode)
      ? String(mode || '').trim().toLowerCase()
      : fallbackMode;

    modeTabEls.forEach((tab) => {
      const isActive = String(tab?.dataset?.shortlinksMode || '').trim().toLowerCase() === nextMode;
      tab.classList.toggle('is-active', isActive);
      tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
      tab.tabIndex = isActive ? 0 : -1;
    });

    modePanelEls.forEach((panel) => {
      const isActive = String(panel?.dataset?.shortlinksModePanel || '').trim().toLowerCase() === nextMode;
      panel.classList.toggle('is-active', isActive);
      panel.hidden = !isActive;
    });

    const activeTab = modeTabEls.find((tab) => String(tab?.dataset?.shortlinksMode || '').trim().toLowerCase() === nextMode);
    if (modeSummaryEl) {
      modeSummaryEl.textContent = activeTab?.textContent?.trim() || 'Single link';
    }
    document.body.dataset.shortlinksMode = nextMode;
    if (!options.skipPersist) saveActiveMode(nextMode);

    if (options.focusTab) {
      if (activeTab && typeof activeTab.focus === 'function') activeTab.focus();
    }
  }

  function revealAccessCard(options = {}){
    if (!accessCard) {
      if (options.focusInput && tokenInput) tokenInput.focus();
      return;
    }

    accessCard.classList.add('is-attention');
    window.clearTimeout(accessCardAttentionTimer);
    accessCardAttentionTimer = window.setTimeout(() => {
      accessCard.classList.remove('is-attention');
    }, 1800);

    if (typeof accessCard.scrollIntoView === 'function') {
      accessCard.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
    if (options.focusInput && tokenInput) tokenInput.focus();
  }

  function getStorage(preferLocal){
    const candidate = preferLocal ? window.localStorage : window.sessionStorage;
    try {
      if (!candidate) return null;
      const key = '__shortlinks_test__';
      candidate.setItem(key, '1');
      candidate.removeItem(key);
      return candidate;
    } catch {
      return null;
    }
  }

  const localTokenStorage = getStorage(true);
  const sessionTokenStorage = getStorage(false);
  let memoryToken = '';
  const tableLayoutQuery = typeof window.matchMedia === 'function'
    ? window.matchMedia(TABLE_LAYOUT_MEDIA_QUERY)
    : null;

  function prefersTableLayout(){
    if (tableLayoutQuery) return !!tableLayoutQuery.matches;
    return typeof window.innerWidth === 'number' ? window.innerWidth >= 900 : true;
  }

  function getSavedToken(){
    const sessionToken = sessionTokenStorage ? sessionTokenStorage.getItem(STORAGE_KEY) || '' : '';
    if (sessionToken) return sessionToken;
    const localToken = localTokenStorage ? localTokenStorage.getItem(STORAGE_KEY) || '' : '';
    if (localToken) return localToken;
    return memoryToken;
  }

  function saveToken(token, remember){
    const value = String(token || '').trim();
    if (!value) {
      if (sessionTokenStorage) sessionTokenStorage.removeItem(STORAGE_KEY);
      if (localTokenStorage) localTokenStorage.removeItem(STORAGE_KEY);
      memoryToken = '';
      return;
    }

    if (remember && localTokenStorage) {
      localTokenStorage.setItem(STORAGE_KEY, value);
      if (sessionTokenStorage) sessionTokenStorage.removeItem(STORAGE_KEY);
      memoryToken = '';
      return;
    }

    if (sessionTokenStorage) {
      sessionTokenStorage.setItem(STORAGE_KEY, value);
      if (localTokenStorage) localTokenStorage.removeItem(STORAGE_KEY);
      memoryToken = '';
      return;
    }

    if (localTokenStorage) {
      localTokenStorage.setItem(STORAGE_KEY, value);
      memoryToken = '';
      return;
    }

    memoryToken = value;
  }

  function isTokenRemembered(){
    return !!(localTokenStorage && localTokenStorage.getItem(STORAGE_KEY));
  }

  function getSavedMode(){
    const mode = sessionTokenStorage ? String(sessionTokenStorage.getItem(MODE_STORAGE_KEY) || '').trim().toLowerCase() : '';
    return getModePanel(mode) ? mode : '';
  }

  function saveActiveMode(mode){
    if (!sessionTokenStorage) return;
    const clean = String(mode || '').trim().toLowerCase();
    if (!getModePanel(clean)) return;
    sessionTokenStorage.setItem(MODE_STORAGE_KEY, clean);
  }

  function getInitialMode(){
    return getSavedMode() || (getSavedToken() ? 'links' : 'single');
  }

  function getSavedViews(){
    const fallback = Array.isArray(memorySavedViews) ? memorySavedViews : [];
    if (!localTokenStorage) return fallback;
    try {
      const parsed = JSON.parse(localTokenStorage.getItem(VIEWS_STORAGE_KEY) || '[]');
      return Array.isArray(parsed)
        ? parsed
          .filter(view => view && typeof view.name === 'string')
          .map(view => ({
            name: String(view.name || '').trim(),
            query: String(view.query || ''),
            status: String(view.status || 'all'),
            sort: String(view.sort || 'slug'),
            density: String(view.density || 'comfortable')
          }))
          .filter(view => view.name)
        : [];
    } catch {
      return fallback;
    }
  }

  function saveSavedViews(views){
    const clean = (Array.isArray(views) ? views : [])
      .filter(view => view && String(view.name || '').trim())
      .map(view => ({
        name: String(view.name || '').trim().slice(0, 64),
        query: String(view.query || '').trim().slice(0, 180),
        status: String(view.status || 'all').trim().toLowerCase(),
        sort: String(view.sort || 'slug').trim().toLowerCase(),
        density: String(view.density || 'comfortable').trim().toLowerCase() === 'compact' ? 'compact' : 'comfortable'
      }))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
    memorySavedViews = clean;
    if (localTokenStorage) {
      try {
        localTokenStorage.setItem(VIEWS_STORAGE_KEY, JSON.stringify(clean));
      } catch {}
    }
    return clean;
  }

  function getCurrentViewConfig(name){
    return {
      name: String(name || '').trim(),
      query: filterInput ? String(filterInput.value || '').trim() : '',
      status: getStatusFilter(),
      sort: getSortMode(),
      density: getDensityMode()
    };
  }

  function renderSavedViewOptions(selectedName){
    if (!savedViewSelect) return;
    const selected = String(selectedName || savedViewSelect.value || '').trim();
    const views = getSavedViews();
    savedViewSelect.replaceChildren();
    const current = document.createElement('option');
    current.value = '';
    current.textContent = 'Current filters';
    savedViewSelect.appendChild(current);
    views.forEach((view) => {
      const option = document.createElement('option');
      option.value = view.name;
      option.textContent = view.name;
      savedViewSelect.appendChild(option);
    });
    savedViewSelect.value = views.some(view => view.name === selected) ? selected : '';
    if (deleteViewButton) deleteViewButton.disabled = !savedViewSelect.value;
  }

  function applySavedView(name){
    const view = getSavedViews().find(item => item.name === name);
    if (!view) return;
    if (filterInput) filterInput.value = view.query || '';
    if (statusFilterSelect) statusFilterSelect.value = view.status || 'all';
    if (sortSelect) sortSelect.value = view.sort || 'slug';
    if (densitySelect) densitySelect.value = view.density === 'compact' ? 'compact' : 'comfortable';
    applyFilterAndRender();
    markSessionDirty();
  }

  function saveCurrentView(){
    if (!savedViewSelect) return;
    const defaultName = savedViewSelect.value || getFilterQuery() || getStatusFilter() || 'Link view';
    const raw = window.prompt('Save current filter view as:', defaultName);
    const name = String(raw || '').trim();
    if (!name) return;
    const views = getSavedViews().filter(view => view.name.toLowerCase() !== name.toLowerCase());
    views.push(getCurrentViewConfig(name));
    saveSavedViews(views);
    renderSavedViewOptions(name);
    setStatus(listStatusEl, `Saved view "${name}".`, 'success');
  }

  function deleteCurrentView(){
    if (!savedViewSelect || !savedViewSelect.value) return;
    const name = savedViewSelect.value;
    const views = getSavedViews().filter(view => view.name !== name);
    saveSavedViews(views);
    renderSavedViewOptions('');
    setStatus(listStatusEl, `Deleted view "${name}".`, 'success');
  }

  function updateAccessMeta(){
    if (accessMetaEl) {
      accessMetaEl.textContent = getSavedToken()
        ? (isTokenRemembered() ? 'Token remembered' : 'Token stored for session')
        : 'Token required';
    }
    if (rememberTokenInput) rememberTokenInput.checked = isTokenRemembered();
    updateAdminSummary();
  }

  function setCount(shown, total){
    if (!countEl) return;
    if (!total) {
      countEl.textContent = '0 links';
      return;
    }
    const label = total === 1 ? 'link' : 'links';
    if (shown === total) {
      countEl.textContent = `${total} ${label}`;
      return;
    }
    countEl.textContent = `Showing ${shown} of ${total} ${label}`;
  }

  function getFilterQuery(){
    if (!filterInput) return '';
    return String(filterInput.value || '').trim().toLowerCase();
  }

  function getStatusFilter(){
    return statusFilterSelect ? String(statusFilterSelect.value || 'all').trim().toLowerCase() : 'all';
  }

  function getSortMode(){
    return sortSelect ? String(sortSelect.value || 'slug').trim().toLowerCase() : 'slug';
  }

  function getDensityMode(){
    return densitySelect && String(densitySelect.value || '').trim().toLowerCase() === 'compact'
      ? 'compact'
      : 'comfortable';
  }

  function isLinkExpired(link){
    const expiresAt = Number.isFinite(Number(link && link.expiresAt)) ? Number(link.expiresAt) : 0;
    return !!expiresAt && expiresAt * 1000 <= Date.now();
  }

  function getLinkExpiresMs(link){
    const expiresAt = Number.isFinite(Number(link && link.expiresAt)) ? Number(link.expiresAt) : 0;
    return expiresAt ? expiresAt * 1000 - Date.now() : 0;
  }

  function isLinkExpiringSoon(link){
    const ms = getLinkExpiresMs(link);
    return ms > 0 && ms <= 7 * 24 * 60 * 60 * 1000;
  }

  function getLinkStatus(link){
    if (!link) return 'unknown';
    if (link.disabled) return 'disabled';
    if (isLinkExpired(link)) return 'expired';
    return 'active';
  }

  function getLinkBySlug(slug){
    const key = normalizeSlugInput(slug);
    return allLinks.find(link => normalizeSlugInput(link && link.slug) === key) || null;
  }

  function getStoredHealth(slug){
    const key = normalizeSlugInput(slug);
    return key ? linkHealth.get(key) || null : null;
  }

  function getLinkHealth(link){
    const status = getLinkStatus(link);
    if (status === 'disabled') {
      return { key: 'disabled', label: 'Disabled', tone: 'muted', note: 'Redirect disabled' };
    }
    if (status === 'expired') {
      return { key: 'expired', label: 'Expired', tone: 'warning', note: 'No longer resolves' };
    }
    if (!String(link && link.destination || '').trim()) {
      return { key: 'broken', label: 'Missing destination', tone: 'error', note: 'Destination is empty' };
    }

    const stored = getStoredHealth(link && link.slug);
    if (stored) return stored;

    if (isLinkExpiringSoon(link)) {
      return { key: 'expiring-soon', label: 'Expires soon', tone: 'warning', note: `Expires in ${formatCountdown(getLinkExpiresMs(link))}` };
    }

    return { key: 'unchecked', label: 'Unchecked', tone: '', note: 'Not tested this session' };
  }

  function getLinkHealthClass(health){
    const tone = String(health && health.tone || '').trim().toLowerCase();
    if (tone === 'success') return 'shortlinks-health-success';
    if (tone === 'warning') return 'shortlinks-health-warning';
    if (tone === 'error') return 'shortlinks-health-error';
    if (tone === 'muted') return 'shortlinks-health-muted';
    return 'shortlinks-health-neutral';
  }

  function makeHealthPill(link){
    const health = getLinkHealth(link);
    const pill = document.createElement('span');
    pill.className = `tool-pill shortlinks-health-pill ${getLinkHealthClass(health)}`;
    pill.textContent = health.label;
    if (health.note) pill.title = health.note;
    return pill;
  }

  function matchesStatusFilter(link){
    const filter = getStatusFilter();
    if (!filter || filter === 'all') return true;
    if (filter === 'active') return getLinkStatus(link) === 'active';
    if (filter === 'disabled') return getLinkStatus(link) === 'disabled';
    if (filter === 'expired') return getLinkStatus(link) === 'expired';
    if (filter === 'temporary') return !link?.permanent;
    if (filter === 'permanent') return !!link?.permanent;
    if (filter === 'expiring-soon') return getLinkHealth(link).key === 'expiring-soon';
    if (filter === 'warning') return ['warning', 'error'].includes(getLinkHealth(link).tone);
    if (filter === 'healthy') return getLinkHealth(link).key === 'healthy';
    return true;
  }

  function getLinkSortValue(link, key){
    switch (key) {
      case 'clicks':
        return Number.isFinite(Number(link?.clicks)) ? Number(link.clicks) : 0;
      case 'updated':
        return Date.parse(link?.updatedAt) || 0;
      case 'created':
        return Date.parse(link?.createdAt) || 0;
      case 'expires':
        return Number.isFinite(Number(link?.expiresAt)) ? Number(link.expiresAt) : Number.MAX_SAFE_INTEGER;
      case 'destination':
        return String(link?.destination || '').toLowerCase();
      case 'slug':
      default:
        return normalizeSlugKey(link?.slug);
    }
  }

  function sortVisibleLinks(links){
    const mode = getSortMode();
    const descending = mode.startsWith('-');
    const key = descending ? mode.slice(1) : mode;
    return (Array.isArray(links) ? links.slice() : []).sort((a, b) => {
      const av = getLinkSortValue(a, key);
      const bv = getLinkSortValue(b, key);
      let result = 0;
      if (typeof av === 'number' && typeof bv === 'number') result = av - bv;
      else result = String(av).localeCompare(String(bv), undefined, { sensitivity: 'base' });
      if (!result) result = normalizeSlugInput(a?.slug).localeCompare(normalizeSlugInput(b?.slug));
      return descending ? -result : result;
    });
  }

  function getFilteredLinks(){
    const query = getFilterQuery();
    const filtered = allLinks.filter(link => {
      if (!matchesStatusFilter(link)) return false;
      if (!query) return true;
      const haystack = [
        link?.slug,
        link?.destination,
        link?.label,
        link?.templateTitle,
        link?.batchTitle,
        link?.contextCompany,
        link?.contextTitle,
        getLinkHealth(link).label,
        getLinkHealth(link).note
      ].join(' ').toLowerCase();
      return haystack.includes(query);
    });
    return sortVisibleLinks(filtered);
  }

  function formatCount(value){
    const n = Number(value);
    if (!Number.isFinite(n)) return '0';
    return Math.max(0, Math.floor(n)).toLocaleString('en-US');
  }

  function toCsvCell(value){
    const raw = String(value ?? '');
    const escaped = raw.replace(/"/g, '""');
    if (/[",\n\r]/.test(escaped)) return `"${escaped}"`;
    return escaped;
  }

  function getExportTimestamp(){
    try {
      return new Date().toISOString().replace(/[:.]/g, '-');
    } catch {
      return String(Date.now());
    }
  }

  function downloadTextFile({ filename, text, mime }){
    let blob;
    try {
      blob = new Blob([String(text ?? '')], { type: String(mime || 'text/plain;charset=utf-8') });
    } catch {
      return false;
    }

    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = String(filename || 'download.txt');
    link.rel = 'noopener';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    return true;
  }

  function normalizeExportClickLimit(value){
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return EXPORT_DEFAULT_CLICK_LIMIT;
    return Math.max(1, Math.min(EXPORT_MAX_CLICK_LIMIT, Math.floor(numeric)));
  }

  function setExportBusy(isBusy){
    const busy = !!isBusy;
    [exportModeSelect, exportClickLimitInput, exportButton].forEach((control) => {
      if (!control) return;
      control.disabled = busy;
    });
  }

  function syncExportControls(){
    const mode = exportModeSelect ? String(exportModeSelect.value || '').trim() : EXPORT_MODE_REDIRECTS_ONLY;
    const clicksEnabled = mode === EXPORT_MODE_WITH_CLICKS;
    if (exportClickLimitInput) {
      exportClickLimitInput.disabled = !clicksEnabled;
    }
    updateAdminSummary();
  }

  function getSortedLinksForExport(){
    return allLinks
      .slice()
      .sort((a, b) => normalizeSlugKey(a?.slug).localeCompare(normalizeSlugKey(b?.slug)) || normalizeSlugInput(a?.slug).localeCompare(normalizeSlugInput(b?.slug)));
  }

  function serializeLinkForExport(link){
    const slug = normalizeSlugInput(link?.slug);
    const destination = formatAbsoluteUrl(link?.destination || '');
    const permanent = !!link?.permanent;
    const statusCode = permanent ? 301 : 302;
    const expiresAt = Number.isFinite(Number(link?.expiresAt)) ? Number(link.expiresAt) : 0;
    return {
      slug,
      shortPath: slug ? buildPublicPath(slug) : '',
      shortUrl: slug ? buildShortUrl(slug) : '',
      destination,
      statusCode,
      permanent,
      disabled: !!link?.disabled,
      expiresAt,
      clicks: Number.isFinite(Number(link?.clicks)) ? Number(link.clicks) : 0,
      createdAt: typeof link?.createdAt === 'string' ? link.createdAt : '',
      updatedAt: typeof link?.updatedAt === 'string' ? link.updatedAt : '',
      label: typeof link?.label === 'string' ? link.label : '',
      templateId: typeof link?.templateId === 'string' ? link.templateId : '',
      templateTitle: typeof link?.templateTitle === 'string' ? link.templateTitle : '',
      batchId: typeof link?.batchId === 'string' ? link.batchId : '',
      batchTitle: typeof link?.batchTitle === 'string' ? link.batchTitle : '',
      contextType: typeof link?.contextType === 'string' ? link.contextType : '',
      contextEntryId: typeof link?.contextEntryId === 'string' ? link.contextEntryId : '',
      contextCompany: typeof link?.contextCompany === 'string' ? link.contextCompany : '',
      contextTitle: typeof link?.contextTitle === 'string' ? link.contextTitle : ''
    };
  }

  function buildRedirectsCsv(links){
    const headers = [
      'slug',
      'short_path',
      'short_url',
      'destination',
      'status_code',
      'permanent',
      'disabled',
      'expires_at_unix',
      'clicks',
      'created_at',
      'updated_at',
      'label',
      'template_id',
      'template_title',
      'batch_id',
      'batch_title',
      'context_type',
      'context_entry_id',
      'context_company',
      'context_title'
    ];

    const rows = links.map((link) => {
      const item = serializeLinkForExport(link);
      return [
        item.slug,
        item.shortPath,
        item.shortUrl,
        item.destination,
        item.statusCode,
        item.permanent ? 'true' : 'false',
        item.disabled ? 'true' : 'false',
        item.expiresAt || '',
        item.clicks,
        item.createdAt,
        item.updatedAt,
        item.label,
        item.templateId,
        item.templateTitle,
        item.batchId,
        item.batchTitle,
        item.contextType,
        item.contextEntryId,
        item.contextCompany,
        item.contextTitle
      ].map(toCsvCell).join(',');
    });

    return `${headers.map(toCsvCell).join(',')}\n${rows.join('\n')}`;
  }

  async function fetchClickHistoryForExport(slug, limit){
    if (!slug) return [];
    const safeLimit = normalizeExportClickLimit(limit);
    const data = await api(`/api/short-links/clicks/${encodeURIComponent(slug)}?limit=${safeLimit}`, { method: 'GET' });
    return Array.isArray(data?.clicks) ? data.clicks : [];
  }

  async function exportRedirectsOnly(links){
    const csv = buildRedirectsCsv(links);
    const filename = `short-links-redirects-${getExportTimestamp()}.csv`;
    const ok = downloadTextFile({ filename, text: csv, mime: 'text/csv;charset=utf-8' });
    if (!ok) {
      setStatus(listStatusEl, 'Unable to export file on this browser.', 'error');
      return;
    }
    setStatus(listStatusEl, `Exported ${links.length} redirect${links.length === 1 ? '' : 's'} to ${filename}.`, 'success');
  }

  async function exportLinksWithClickHistory(links){
    const clickLimit = normalizeExportClickLimit(exportClickLimitInput?.value);
    if (exportClickLimitInput) exportClickLimitInput.value = String(clickLimit);

    const exportLinks = [];
    const errors = [];
    let totalClicks = 0;

    for (let index = 0; index < links.length; index += 1) {
      const raw = links[index];
      const item = serializeLinkForExport(raw);
      const progress = `${index + 1}/${links.length}`;
      const label = item.shortPath || item.slug || 'link';
      setStatus(listStatusEl, `Exporting click history ${progress} (${label})…`);
      try {
        const clicks = await fetchClickHistoryForExport(item.slug, clickLimit);
        item.clickEvents = clicks;
        totalClicks += clicks.length;
      } catch (err) {
        item.clickEvents = [];
        item.clicksError = err?.message || 'Unable to fetch click history.';
        errors.push(`${item.slug || label}: ${item.clicksError}`);
      }
      exportLinks.push(item);
    }

    const payload = {
      generatedAt: new Date().toISOString(),
      exportMode: EXPORT_MODE_WITH_CLICKS,
      clickHistoryLimitPerLink: clickLimit,
      totals: {
        links: exportLinks.length,
        clickEvents: totalClicks,
        clickFetchErrors: errors.length
      },
      links: exportLinks
    };

    const filename = `short-links-with-clicks-${getExportTimestamp()}.json`;
    const ok = downloadTextFile({
      filename,
      text: JSON.stringify(payload, null, 2),
      mime: 'application/json;charset=utf-8'
    });
    if (!ok) {
      setStatus(listStatusEl, 'Unable to export file on this browser.', 'error');
      return;
    }

    if (errors.length) {
      setStatus(
        listStatusEl,
        `Exported ${exportLinks.length} links with ${totalClicks} click events. ${errors.length} link(s) had click history errors.`,
        'warning'
      );
      return;
    }

    setStatus(listStatusEl, `Exported ${exportLinks.length} links with ${totalClicks} click events to ${filename}.`, 'success');
  }

  async function handleExport(){
    if (!requireToken(listStatusEl)) return;
    const links = getSortedLinksForExport();
    if (!links.length) {
      setStatus(listStatusEl, 'No links to export.', 'warning');
      return;
    }

    const mode = exportModeSelect ? String(exportModeSelect.value || '').trim() : EXPORT_MODE_REDIRECTS_ONLY;
    setExportBusy(true);
    try {
      if (mode === EXPORT_MODE_WITH_CLICKS) {
        await exportLinksWithClickHistory(links);
      } else {
        await exportRedirectsOnly(links);
      }
    } finally {
      setExportBusy(false);
      syncExportControls();
    }
  }

  function makeSnapshotCard({ label, value, note, tone }){
    const card = document.createElement('article');
    card.className = 'shortlinks-snapshot-card';
    const toneClass = String(tone || '').trim();
    if (toneClass) card.classList.add(`is-${toneClass}`);

    const cardLabel = document.createElement('p');
    cardLabel.className = 'shortlinks-snapshot-label';
    cardLabel.textContent = String(label || '');
    card.appendChild(cardLabel);

    const cardValue = document.createElement('p');
    cardValue.className = 'shortlinks-snapshot-value';
    cardValue.textContent = String(value || '');
    card.appendChild(cardValue);

    const cardNote = document.createElement('p');
    cardNote.className = 'shortlinks-snapshot-note';
    cardNote.textContent = String(note || '');
    card.appendChild(cardNote);

    return card;
  }

  function getHealthCounts(links){
    return (Array.isArray(links) ? links : []).reduce((counts, link) => {
      const health = getLinkHealth(link);
      counts.total += 1;
      if (!['healthy', 'warning', 'broken'].includes(health.key)) {
        counts[health.key] = (counts[health.key] || 0) + 1;
      }
      if (health.tone === 'success') counts.healthy += 1;
      if (health.tone === 'warning') counts.warning += 1;
      if (health.tone === 'error') counts.broken += 1;
      if (isLinkExpiringSoon(link)) counts.expiringSoon += 1;
      return counts;
    }, {
      total: 0,
      healthy: 0,
      warning: 0,
      broken: 0,
      expiringSoon: 0,
      unchecked: 0,
      disabled: 0,
      expired: 0
    });
  }

  function makeHealthMetric({ label, value, tone, filter }){
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `shortlinks-health-metric ${tone ? `is-${tone}` : ''}`;
    button.innerHTML = `<span>${label}</span><strong>${formatCount(value)}</strong>`;
    button.addEventListener('click', () => {
      if (statusFilterSelect && filter) statusFilterSelect.value = filter;
      applyFilterAndRender();
    });
    return button;
  }

  function renderHealthStrip(){
    if (!healthStripEl) return;
    healthStripEl.replaceChildren();
    const counts = getHealthCounts(allLinks);
    [
      { label: 'Healthy', value: counts.healthy, tone: 'success', filter: 'healthy' },
      { label: 'Warnings', value: counts.warning + counts.broken, tone: counts.warning || counts.broken ? 'warning' : '', filter: 'warning' },
      { label: 'Expiring', value: counts.expiringSoon, tone: counts.expiringSoon ? 'warning' : '', filter: 'expiring-soon' },
      { label: 'Unchecked', value: counts.unchecked, tone: counts.unchecked ? 'neutral' : '', filter: 'all' },
      { label: 'Disabled', value: counts.disabled, tone: counts.disabled ? 'muted' : '', filter: 'disabled' }
    ].forEach(metric => {
      healthStripEl.appendChild(makeHealthMetric(metric));
    });
  }

  function pruneSelectedSlugs(){
    const valid = new Set(allLinks.map(link => normalizeSlugInput(link && link.slug)).filter(Boolean));
    selectedSlugs = new Set(Array.from(selectedSlugs).filter(slug => valid.has(slug)));
  }

  function getSelectedLinks(){
    const selected = new Set(Array.from(selectedSlugs).map(normalizeSlugInput));
    return allLinks.filter(link => selected.has(normalizeSlugInput(link && link.slug)));
  }

  function updateSelectionControls(){
    pruneSelectedSlugs();
    const selectedCount = selectedSlugs.size;
    if (selectionCountEl) {
      selectionCountEl.textContent = `${formatCount(selectedCount)} selected`;
    }
    if (selectVisibleInput) {
      const visibleKeys = visibleLinkSlugs.map(normalizeSlugInput).filter(Boolean);
      const visibleSelected = visibleKeys.filter(slug => selectedSlugs.has(slug)).length;
      selectVisibleInput.checked = visibleKeys.length > 0 && visibleSelected === visibleKeys.length;
      selectVisibleInput.indeterminate = visibleSelected > 0 && visibleSelected < visibleKeys.length;
      selectVisibleInput.disabled = visibleKeys.length === 0;
    }
    [testSelectedButton, clearSelectionButton].forEach((button) => {
      if (button) button.disabled = selectedCount === 0;
    });
    if (exportViewButton) exportViewButton.disabled = visibleLinkSlugs.length === 0;
  }

  function makeSelectionControl(link){
    const slug = normalizeSlugInput(link && link.slug);
    const label = buildPublicPath(slug);
    const wrap = document.createElement('label');
    wrap.className = 'shortlinks-select-link';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = selectedSlugs.has(slug);
    input.setAttribute('aria-label', `Select ${label}`);
    input.addEventListener('change', () => {
      if (input.checked) selectedSlugs.add(slug);
      else selectedSlugs.delete(slug);
      updateSelectionControls();
    });
    const text = document.createElement('span');
    text.className = 'visually-hidden';
    text.textContent = `Select ${label}`;
    wrap.appendChild(input);
    wrap.appendChild(text);
    return wrap;
  }

  function renderDetailPanel(link){
    if (!detailPanelEl) return;
    detailPanelEl.replaceChildren();
    if (!link) {
      detailPanelEl.hidden = true;
      activeDetailSlug = '';
      return;
    }

    activeDetailSlug = normalizeSlugInput(link.slug);
    detailPanelEl.hidden = false;
    const shortUrl = buildShortUrl(link.slug);
    const destinationUrl = formatAbsoluteUrl(link.destination);
    const health = getLinkHealth(link);
    const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
    const stored = getStoredHealth(link.slug);

    const head = document.createElement('div');
    head.className = 'shortlinks-detail-head';
    const copy = document.createElement('div');
    copy.className = 'shortlinks-card-copy';
    const kicker = document.createElement('p');
    kicker.className = 'shortlinks-kicker';
    kicker.textContent = 'Link detail';
    const title = document.createElement('h3');
    title.className = 'shortlinks-output-title';
    title.textContent = buildPublicPath(link.slug);
    copy.appendChild(kicker);
    copy.appendChild(title);
    head.appendChild(copy);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'btn-ghost shortlinks-detail-close';
    close.textContent = 'Close';
    close.addEventListener('click', () => renderDetailPanel(null));
    head.appendChild(close);
    detailPanelEl.appendChild(head);

    const status = document.createElement('div');
    status.className = 'shortlinks-detail-status';
    status.appendChild(makeHealthPill(link));
    const statusNote = document.createElement('span');
    statusNote.textContent = stored?.checkedAt
      ? `${health.note || health.label} · checked ${formatTimestamp(stored.checkedAt)}`
      : (health.note || 'Not tested this session');
    status.appendChild(statusNote);
    detailPanelEl.appendChild(status);

    const grid = document.createElement('dl');
    grid.className = 'shortlinks-detail-grid';
    [
      ['Short URL', shortUrl],
      ['Destination', destinationUrl],
      ['Redirect', link.permanent ? '301 permanent' : '302 temporary'],
      ['Clicks', `${formatCount(link.clicks)} total`],
      ['Created', formatTimestamp(link.createdAt) || 'Unknown'],
      ['Updated', formatTimestamp(link.updatedAt) || 'Unknown'],
      ['Expires', expiresAt ? new Date(expiresAt * 1000).toLocaleString() : 'Never'],
      ['Campaign', link.batchTitle || link.templateTitle || link.label || 'None']
    ].forEach(([label, value]) => {
      const wrap = document.createElement('div');
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = value;
      wrap.appendChild(dt);
      wrap.appendChild(dd);
      grid.appendChild(wrap);
    });
    detailPanelEl.appendChild(grid);

    const actions = document.createElement('div');
    actions.className = 'shortlinks-detail-actions shortlinks-action-row shortlinks-action-row-compact';
    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.className = 'btn-secondary';
    copyButton.textContent = 'Copy short URL';
    copyButton.addEventListener('click', async () => {
      await copyTextToClipboard({
        text: shortUrl,
        button: copyButton,
        statusTarget: listStatusEl,
        successMessage: `Copied: ${shortUrl}`
      });
    });
    const editButton = document.createElement('button');
    editButton.type = 'button';
    editButton.className = 'btn-secondary';
    editButton.textContent = 'Edit';
    editButton.addEventListener('click', () => {
      openEditorForLink({
        slug: link.slug,
        destination: link.destination,
        disabled: !!link.disabled,
        expiresAt
      });
    });
    const testButton = document.createElement('button');
    testButton.type = 'button';
    testButton.className = 'btn-secondary';
    testButton.textContent = 'Test';
    testButton.addEventListener('click', async () => {
      await testRedirect(link.slug, testButton);
    });
    actions.appendChild(copyButton);
    actions.appendChild(editButton);
    actions.appendChild(testButton);
    detailPanelEl.appendChild(actions);
  }

  function renderDashboardSummary(){
    if (!summaryEl) return;
    summaryEl.replaceChildren();

    const links = Array.isArray(allLinks) ? allLinks : [];
    const totalLinks = links.length;
    const permanentLinks = links.reduce((count, link) => count + (link && link.permanent ? 1 : 0), 0);
    const temporaryLinks = Math.max(0, totalLinks - permanentLinks);
    const activeLinks = links.reduce((count, link) => {
      if (!link || link.disabled || isLinkExpired(link)) return count;
      return count + 1;
    }, 0);
    const healthCounts = getHealthCounts(links);
    const expiringSoon = healthCounts.expiringSoon;
    const warnings = healthCounts.warning + healthCounts.broken;

    const totalProjects = Number.isFinite(Number(projectHealth.total)) ? Number(projectHealth.total) : 0;
    const missingProjects = Number.isFinite(Number(projectHealth.missing)) ? Number(projectHealth.missing) : 0;
    const mismatchedProjects = Number.isFinite(Number(projectHealth.mismatched)) ? Number(projectHealth.mismatched) : 0;
    const syncedProjects = Math.max(0, totalProjects - missingProjects - mismatchedProjects);

    const query = getFilterQuery();

    summaryEl.appendChild(makeSnapshotCard({
      label: 'Total links',
      value: formatCount(totalLinks),
      note: query ? `${formatCount(visibleLinksCount)} visible in search` : 'All saved slugs'
    }));

    summaryEl.appendChild(makeSnapshotCard({
      label: 'Active links',
      value: formatCount(activeLinks),
      note: expiringSoon ? `${formatCount(expiringSoon)} expiring within 7 days` : 'Live and available now',
      tone: expiringSoon ? 'warning' : ''
    }));

    summaryEl.appendChild(makeSnapshotCard({
      label: 'Temporary links',
      value: formatCount(temporaryLinks),
      note: `${formatCount(permanentLinks)} permanent redirects`,
      tone: temporaryLinks ? 'info' : ''
    }));

    summaryEl.appendChild(makeSnapshotCard({
      label: 'Health checks',
      value: warnings ? formatCount(warnings) : formatCount(healthCounts.healthy),
      note: warnings ? `${formatCount(warnings)} warning${warnings === 1 ? '' : 's'} need review` : `${formatCount(healthCounts.unchecked)} unchecked this session`,
      tone: warnings ? 'warning' : (healthCounts.healthy ? 'success' : '')
    }));

    if (totalProjects > 0) {
      const issues = missingProjects + mismatchedProjects;
      const noteBits = [];
      if (missingProjects) noteBits.push(`${formatCount(missingProjects)} missing`);
      if (mismatchedProjects) noteBits.push(`${formatCount(mismatchedProjects)} mismatched`);
      summaryEl.appendChild(makeSnapshotCard({
        label: 'Project sync',
        value: `${formatCount(syncedProjects)} / ${formatCount(totalProjects)}`,
        note: noteBits.length ? noteBits.join(' · ') : 'All mapped correctly',
        tone: issues ? 'warning' : 'success'
      }));
    } else {
      summaryEl.appendChild(makeSnapshotCard({
        label: 'Project sync',
        value: 'Loading',
        note: 'Waiting for portfolio destinations'
      }));
    }

    updateAdminSummary();
  }

  function updateAdminSummary(){
    const hasToken = !!getSavedToken();
    setAdminBadge(
      adminAccessSummaryEl,
      hasToken ? 'Token saved' : 'Token required',
      hasToken ? 'success' : 'warning'
    );

    const totalProjects = Number.isFinite(Number(projectHealth.total)) ? Number(projectHealth.total) : 0;
    const missingProjects = Number.isFinite(Number(projectHealth.missing)) ? Number(projectHealth.missing) : 0;
    const mismatchedProjects = Number.isFinite(Number(projectHealth.mismatched)) ? Number(projectHealth.mismatched) : 0;
    const projectIssues = missingProjects + mismatchedProjects;

    if (!totalProjects) {
      setAdminBadge(adminProjectSummaryEl, 'Project sync loading', '');
    } else {
      setAdminBadge(
        adminProjectSummaryEl,
        projectIssues
          ? `${formatCount(projectIssues)} sync issue${projectIssues === 1 ? '' : 's'}`
          : 'Project sync healthy',
        projectIssues ? 'warning' : 'success'
      );
    }

    const mode = exportModeSelect ? String(exportModeSelect.value || '').trim() : EXPORT_MODE_REDIRECTS_ONLY;
    if (mode === EXPORT_MODE_WITH_CLICKS) {
      const clickLimit = normalizeExportClickLimit(exportClickLimitInput?.value);
      setAdminBadge(adminExportSummaryEl, `JSON export · ${formatCount(clickLimit)} clicks`, 'info');
      return;
    }

    setAdminBadge(adminExportSummaryEl, 'CSV export ready', '');
  }

  function getAudienceApi(){
    return window.SITE_AUDIENCE_CONFIG || null;
  }

  function getAudienceOrder(){
    const order = Array.isArray(window.SITE_AUDIENCE_ORDER) && window.SITE_AUDIENCE_ORDER.length
      ? window.SITE_AUDIENCE_ORDER
      : FALLBACK_AUDIENCE_ORDER;
    return order
      .map((value) => String(value || '').trim().toLowerCase())
      .filter((value) => value && (FALLBACK_AUDIENCES[value] || (window.SITE_AUDIENCES && window.SITE_AUDIENCES[value])));
  }

  function normalizeAudienceKey(value){
    const api = getAudienceApi();
    if (api && typeof api.normalizeAudience === 'function') {
      return api.normalizeAudience(value);
    }
    const raw = String(value || '').trim().toLowerCase();
    if (!raw) return FALLBACK_AUDIENCE_DEFAULT;
    if (raw === 'datascience' || raw === 'data_science') return 'data-science';
    if (raw === 'tourism-analytics') return 'tourism';
    return FALLBACK_AUDIENCES[raw] ? raw : FALLBACK_AUDIENCE_DEFAULT;
  }

  function getAudienceConfig(value){
    const key = normalizeAudienceKey(value);
    const api = getAudienceApi();
    if (api && typeof api.getAudience === 'function') {
      return api.getAudience(key);
    }
    return FALLBACK_AUDIENCES[key] || FALLBACK_AUDIENCES[FALLBACK_AUDIENCE_DEFAULT];
  }

  function getSelectedAudienceKey(){
    return normalizeAudienceKey(audienceSelect ? audienceSelect.value : FALLBACK_AUDIENCE_DEFAULT);
  }

  function normalizeSitePath(pathname){
    const raw = String(pathname || '').trim();
    if (!raw) return '/';
    const normalized = raw.replace(/\/+$/g, '');
    return normalized || '/';
  }

  function isManagedSiteHost(hostname){
    const host = String(hostname || '').trim().toLowerCase();
    if (!host) return false;
    return isProdHost(host) || isPreviewHost(host) || isDevHost(host);
  }

  function parseInternalSiteDestination(value){
    const raw = String(value || '').trim();
    if (!raw) return null;

    const base = getCanonicalSiteOrigin() || window.location.origin;
    try {
      const url = raw.startsWith('/')
        ? new URL(raw, base)
        : new URL(raw);
      if (!raw.startsWith('/') && !isManagedSiteHost(url.hostname)) return null;
      return {
        raw,
        url,
        pathname: normalizeSitePath(url.pathname),
        searchParams: new URLSearchParams(url.search || ''),
        hash: url.hash || ''
      };
    } catch {
      return null;
    }
  }

  function buildAbsoluteSiteUrl(pathname, searchParams, hash){
    const base = getCanonicalSiteOrigin() || window.location.origin;
    const url = new URL(String(pathname || '/'), base);
    const params = searchParams instanceof URLSearchParams
      ? new URLSearchParams(searchParams)
      : new URLSearchParams();
    url.search = params.toString();
    url.hash = hash || '';
    return url.toString();
  }

  function getAudienceConfigPaths(field){
    return getAudienceOrder()
      .map((key) => {
        const config = getAudienceConfig(key);
        return normalizeSitePath(config && config[field]);
      })
      .filter(Boolean);
  }

  function isAudienceHomePath(pathname){
    return getAudienceConfigPaths('homePath').includes(normalizeSitePath(pathname));
  }

  function isAudienceResumePath(pathname){
    return getAudienceConfigPaths('resumePath').includes(normalizeSitePath(pathname));
  }

  function isAudienceResumePreviewPath(pathname){
    return getAudienceConfigPaths('resumePreviewPath').includes(normalizeSitePath(pathname));
  }

  function isAudienceResumeDownloadPath(pathname){
    return getAudienceConfigPaths('resumeDownloadPath').includes(normalizeSitePath(pathname));
  }

  function isPortfolioPath(pathname){
    const path = normalizeSitePath(pathname);
    return path === '/portfolio' || path.startsWith('/portfolio/');
  }

  function isAudienceAwarePath(pathname){
    const path = normalizeSitePath(pathname);
    return path === '/'
      || isAudienceHomePath(path)
      || path === '/resume'
      || isAudienceResumePath(path)
      || path === '/resume-pdf'
      || isAudienceResumePreviewPath(path)
      || path === '/documents/Resume.pdf'
      || isAudienceResumeDownloadPath(path)
      || isPortfolioPath(path);
  }

  function shouldShowAudienceFieldForValue(value){
    const info = parseInternalSiteDestination(value);
    if (!info) return false;
    return isAudienceAwarePath(info.pathname);
  }

  function isPortfolioProjectPath(pathname){
    const path = normalizeSitePath(pathname);
    return path.startsWith('/portfolio/') && path !== '/portfolio';
  }

  function buildDisplayDestination(value, audienceKey){
    const info = parseInternalSiteDestination(value);
    if (!info) return normalizeDestinationForSave(value);

    const audience = getAudienceConfig(audienceKey);
    const path = normalizeSitePath(info.pathname);
    const params = new URLSearchParams(info.searchParams);

    if (path === '/' || isAudienceHomePath(path)) {
      return buildAbsoluteSiteUrl(audience.homePath);
    }
    if (path === '/resume' || isAudienceResumePath(path)) {
      return buildAbsoluteSiteUrl(audience.resumePath);
    }
    if (path === '/resume-pdf' || isAudienceResumePreviewPath(path)) {
      return buildAbsoluteSiteUrl(audience.resumePreviewPath);
    }
    if (path === '/documents/Resume.pdf' || isAudienceResumeDownloadPath(path)) {
      return buildAbsoluteSiteUrl(audience.resumeDownloadPath);
    }
    if (isPortfolioPath(path)) {
      params.delete('audience');
      params.set('audience', audience.key);
      return buildAbsoluteSiteUrl(path, params, info.hash);
    }

    return buildAbsoluteSiteUrl(path, params, info.hash);
  }

  function buildStoredDestination(value, audienceKey){
    const info = parseInternalSiteDestination(value);
    if (!info) return normalizeDestinationForSave(value);

    const audience = getAudienceConfig(audienceKey);
    const path = normalizeSitePath(info.pathname);
    const params = new URLSearchParams(info.searchParams);

    if (path === '/' || isAudienceHomePath(path)) {
      return buildAbsoluteSiteUrl(audience.homePath);
    }
    if (path === '/resume' || isAudienceResumePath(path)) {
      return buildAbsoluteSiteUrl(audience.resumePath);
    }
    if (path === '/resume-pdf' || isAudienceResumePreviewPath(path)) {
      return buildAbsoluteSiteUrl(audience.resumePreviewPath);
    }
    if (path === '/documents/Resume.pdf' || isAudienceResumeDownloadPath(path)) {
      return buildAbsoluteSiteUrl(audience.resumeDownloadPath);
    }
    if (isPortfolioPath(path)) {
      params.delete('audience');
      return buildAbsoluteSiteUrl(path, params, info.hash);
    }

    return buildAbsoluteSiteUrl(path, params, info.hash);
  }

  function buildDisplayPath(value, audienceKey){
    const resolved = buildDisplayDestination(value, audienceKey);
    const info = parseInternalSiteDestination(resolved);
    if (!info) return String(value || '').trim();
    const search = info.url.search || '';
    const hash = info.url.hash || '';
    return `${info.pathname}${search}${hash}`;
  }

  function buildShareShortUrl(slug, destination, audienceKey){
    const baseShortUrl = buildShortUrl(slug);
    if (!baseShortUrl) return '';

    const info = parseInternalSiteDestination(destination);
    if (!info || !isPortfolioPath(info.pathname)) return baseShortUrl;

    try {
      const url = new URL(baseShortUrl);
      url.searchParams.set('audience', normalizeAudienceKey(audienceKey));
      return url.toString();
    } catch {
      return baseShortUrl;
    }
  }

  function buildBaseSlugFromPath(pathname){
    const clean = normalizeSitePath(pathname).replace(/^\/+|\/+$/g, '');
    if (!clean) return 'home';
    const last = clean.split('/').filter(Boolean).slice(-1)[0] || '';
    return last.toLowerCase();
  }

  function buildSuggestedSlugFromPath(pathname, audienceKey){
    const path = normalizeSitePath(pathname);
    const audience = getAudienceConfig(audienceKey);

    if (path === '/' || isAudienceHomePath(path)) {
      return normalizeSlugInput(audience.key);
    }
    if (path === '/resume' || isAudienceResumePath(path)) {
      return normalizeSlugInput(`resume/${audience.key}`);
    }
    if (path === '/resume-pdf' || isAudienceResumePreviewPath(path)) {
      return normalizeSlugInput(`resume/${audience.key}/pdf`);
    }
    if (path === '/documents/Resume.pdf' || isAudienceResumeDownloadPath(path)) {
      return normalizeSlugInput(`resume/${audience.key}/download`);
    }
    if (path === '/portfolio') {
      return 'portfolio';
    }
    return buildBaseSlugFromPath(path);
  }

  function setSuggestedSlug(value){
    if (!slugInput) return;
    const suggestion = normalizeSlugInput(value);
    const current = normalizeSlugKey(slugInput.value);
    const prior = normalizeSlugKey(slugInput.dataset.autoSuggested || '');

    if (!current || current === prior) {
      slugInput.value = suggestion;
    }

    if (suggestion) slugInput.dataset.autoSuggested = suggestion;
    else delete slugInput.dataset.autoSuggested;
  }

  function syncAudienceFieldVisibility(){
    const showAudienceField = shouldShowAudienceFieldForValue(destinationInput?.value);
    if (audienceFieldEl) audienceFieldEl.hidden = !showAudienceField;
    if (audienceSelect) audienceSelect.disabled = !showAudienceField;
    return showAudienceField;
  }

  function syncEditorAudienceState(options = {}){
    if (!destinationInput) return;
    const raw = String(destinationInput.value || '').trim();
    const showAudienceField = syncAudienceFieldVisibility();
    if (!raw) return;

    const audienceKey = getSelectedAudienceKey();
    const nextDestination = buildDisplayDestination(raw, audienceKey);
    if (nextDestination) {
      destinationInput.value = nextDestination;
      const info = parseInternalSiteDestination(nextDestination);
      if (info) {
        setSuggestedSlug(buildSuggestedSlugFromPath(info.pathname, audienceKey));
      }
    }

    if (options.announce && showAudienceField) {
      const audience = getAudienceConfig(audienceKey);
      setStatus(editorStatusEl, `Using ${audience.shortLabel || audience.label || audience.key} destinations where supported.`, 'success');
    }
  }

  function isDevHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'localhost' || host === '127.0.0.1';
  }

  function isPreviewHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return isDevHost(host) || host.endsWith('.vercel.app');
  }

  function isProdHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'danielshort.me' || host === 'www.danielshort.me';
  }

  function getCanonicalSiteOrigin(){
    const origin = destinationsManifest && destinationsManifest.origin ? String(destinationsManifest.origin) : '';
    if (isPreviewHost(window.location.hostname)) {
      return window.location.origin;
    }
    if (isProdHost(window.location.hostname)) {
      return origin || 'https://www.danielshort.me';
    }
    return origin || window.location.origin;
  }

  function joinOriginAndPath(origin, pathname){
    const base = String(origin || '').replace(/\/+$/g, '');
    const path = String(pathname || '');
    if (!base) return path;
    if (!path) return base;
    if (path.startsWith('/')) return `${base}${path}`;
    return `${base}/${path}`;
  }

  function normalizeDestinationForSave(value){
    const raw = String(value || '').trim();
    if (!raw) return '';
    if (raw.startsWith('/')) return joinOriginAndPath(getCanonicalSiteOrigin(), raw);
    return raw;
  }

  function formatAbsoluteUrl(input){
    const raw = typeof input === 'string' ? input.trim() : '';
    if (!raw) return '';
    try {
      return new URL(raw, window.location.origin).toString();
    } catch {
      return raw;
    }
  }

  function syncModalOpenState(){
    if (!document || !document.body) return;
    if (!document.querySelector('.modal.active')) {
      document.body.classList.remove('modal-open');
    }
  }

  function closeDestinationPicker(){
    if (!destinationModal) return;
    destinationModal.classList.remove('active');
    destinationModal.setAttribute('aria-hidden', 'true');
    syncModalOpenState();
    if (destinationSearch) destinationSearch.value = '';
    if (destinationResults) destinationResults.replaceChildren();
    if (destinationModalPrevFocus && document.contains(destinationModalPrevFocus)) {
      destinationModalPrevFocus.focus();
    }
    destinationModalPrevFocus = null;
  }

  async function loadDestinationsManifest(){
    if (destinationsManifest) return destinationsManifest;
    const fallback = { origin: 'https://www.danielshort.me', pages: FALLBACK_DESTINATIONS };
    try {
      const resp = await fetch(DESTINATIONS_MANIFEST_PATH, { method: 'GET', cache: 'no-store' });
      if (!resp.ok) throw new Error(`Manifest request failed (${resp.status})`);
      const data = await resp.json().catch(() => null);
      if (!data || typeof data !== 'object' || !Array.isArray(data.pages)) throw new Error('Invalid manifest');
      destinationsManifest = data;
      return destinationsManifest;
    } catch {
      destinationsManifest = fallback;
      return destinationsManifest;
    }
  }

  function getDestinationQuery(){
    if (!destinationSearch) return '';
    return String(destinationSearch.value || '').trim().toLowerCase();
  }

  function getFilteredDestinations(){
    const manifest = destinationsManifest;
    if (!manifest || !Array.isArray(manifest.pages)) return [];
    const query = getDestinationQuery();
    const audienceKey = getSelectedAudienceKey();
    const pages = manifest.pages.filter(item => item && typeof item.path === 'string' && typeof item.label === 'string');
    if (!query) return pages;
    return pages.filter(item => {
      const displayPath = buildDisplayPath(item.path, audienceKey);
      const hay = `${item.label} ${item.path} ${displayPath}`.toLowerCase();
      return hay.includes(query);
    });
  }

  function setProjectsMeta(text){
    if (!projectsMetaEl) return;
    projectsMetaEl.textContent = text || '';
  }

  function setProjectsBusy(isBusy){
    const busy = !!isBusy;
    const controls = [projectsRefreshButton, projectsEnsureButton];
    controls.forEach(control => {
      if (!control) return;
      control.disabled = busy;
    });
  }

  function buildProjectSlugFromPath(pathname){
    const suffix = buildBaseSlugFromPath(pathname);
    if (!suffix) return '';
    const prefix = normalizeSlugInput(PROJECT_SLUG_PREFIX);
    return prefix ? `${prefix}/${suffix}` : suffix;
  }

  function getPortfolioProjectsFromManifest(){
    const manifest = destinationsManifest;
    const pages = manifest && Array.isArray(manifest.pages) ? manifest.pages : [];
    return pages
      .filter(item => item && item.group === 'Portfolio' && typeof item.path === 'string')
      .filter(item => item.path.startsWith('/portfolio/') && item.path !== '/portfolio')
      .map(item => ({
        path: item.path,
        label: typeof item.label === 'string' ? item.label : item.path
      }));
  }

  function rebuildProjectCatalog(){
    const origin = getCanonicalSiteOrigin();
    const projects = getPortfolioProjectsFromManifest();
    projectCatalog = projects
      .map(project => {
        const path = String(project.path || '');
        const id = path.split('/').filter(Boolean).slice(-1)[0] || '';
        const slug = buildProjectSlugFromPath(path);
        const destination = joinOriginAndPath(origin, path);
        const label = String(project.label || '').trim() || id || path;
        return { id, path, label, slug, destination };
      })
      .filter(project => project.slug && project.destination)
      .sort((a, b) => a.label.localeCompare(b.label));
  }

  function upsertLinkInMemory(link){
    if (!link || typeof link.slug !== 'string') return;
    const slug = normalizeSlugInput(link.slug);
    if (!slug) return;

    const normalized = {
      slug,
      destination: typeof link.destination === 'string' ? link.destination : '',
      permanent: !!link.permanent,
      expiresAt: Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0,
      disabled: !!link.disabled,
      createdAt: typeof link.createdAt === 'string' ? link.createdAt : '',
      updatedAt: typeof link.updatedAt === 'string' ? link.updatedAt : '',
      clicks: Number.isFinite(Number(link.clicks)) ? Number(link.clicks) : 0,
      label: typeof link.label === 'string' ? link.label : '',
      templateId: typeof link.templateId === 'string' ? link.templateId : '',
      templateTitle: typeof link.templateTitle === 'string' ? link.templateTitle : '',
      batchId: typeof link.batchId === 'string' ? link.batchId : '',
      batchTitle: typeof link.batchTitle === 'string' ? link.batchTitle : '',
      contextType: typeof link.contextType === 'string' ? link.contextType : '',
      contextEntryId: typeof link.contextEntryId === 'string' ? link.contextEntryId : '',
      contextCompany: typeof link.contextCompany === 'string' ? link.contextCompany : '',
      contextTitle: typeof link.contextTitle === 'string' ? link.contextTitle : ''
    };

    const idx = allLinks.findIndex(item => normalizeSlugKey(item.slug) === normalizeSlugKey(slug));
    if (idx >= 0) allLinks[idx] = Object.assign({}, allLinks[idx], normalized);
    else allLinks.push(normalized);

    allLinks.sort((a, b) => normalizeSlugKey(a?.slug).localeCompare(normalizeSlugKey(b?.slug)) || String(a.slug || '').localeCompare(String(b.slug || '')));
  }

  function normalizeDestinationForCompare(value){
    const raw = typeof value === 'string' ? value.trim() : '';
    if (!raw) return '';
    return formatAbsoluteUrl(raw);
  }

  function renderProjectLinks(){
    if (!projectsListEl) return;
    projectsListEl.replaceChildren();

    if (!Array.isArray(projectCatalog) || projectCatalog.length === 0) {
      projectHealth = { total: 0, missing: 0, mismatched: 0 };
      setProjectsMeta('');
      const empty = document.createElement('p');
      empty.className = 'shortlinks-empty shortlinks-empty-state';
      empty.textContent = destinationsManifest ? 'No portfolio projects found.' : 'Loading projects…';
      projectsListEl.appendChild(empty);
      renderDashboardSummary();
      return;
    }

    const linkMap = new Map();
    allLinks.forEach(link => {
      const slug = normalizeSlugKey(link && link.slug);
      if (!slug) return;
      linkMap.set(slug, link);
    });

    const total = projectCatalog.length;
    let missing = 0;
    let mismatched = 0;

    projectCatalog.forEach(project => {
      const expectedSlug = normalizeSlugKey(project.slug);
      const expectedDestination = normalizeDestinationForCompare(project.destination);
      const link = expectedSlug ? linkMap.get(expectedSlug) : null;
      const hasLink = !!(link && typeof link.destination === 'string');
      const destMatches = hasLink
        ? normalizeDestinationForCompare(link.destination) === expectedDestination
        : false;

      if (!hasLink) missing += 1;
      else if (!destMatches) mismatched += 1;

      const shortUrl = expectedSlug ? buildShortUrl(expectedSlug) : '';
      const destinationUrl = formatAbsoluteUrl(project.destination);

      const card = document.createElement('article');
      card.className = 'shortlinks-item shortlinks-project-item';
      if (!hasLink) card.classList.add('shortlinks-item-missing');
      if (hasLink && link.disabled) card.classList.add('shortlinks-item-disabled');
      if (hasLink && !destMatches) card.classList.add('shortlinks-item-mismatch');

      const head = document.createElement('div');
      head.className = 'shortlinks-item-head';

      const titleWrap = document.createElement('div');
      titleWrap.className = 'shortlinks-item-title';

      const projectName = document.createElement('p');
      projectName.className = 'shortlinks-project-name';
      projectName.textContent = project.label || project.id || 'Project';

      const slugCode = document.createElement('code');
      slugCode.className = 'shortlinks-slug';
      slugCode.textContent = expectedSlug ? buildPublicPath(expectedSlug) : '';

      const meta = document.createElement('div');
      meta.className = 'shortlinks-item-meta';

      if (!hasLink) {
        const missingPill = document.createElement('span');
        missingPill.className = 'tool-pill shortlinks-pill-missing';
        missingPill.textContent = 'Missing';
        meta.appendChild(missingPill);
      } else {
        const statusPill = document.createElement('span');
        statusPill.className = 'tool-pill';
        statusPill.textContent = link.permanent ? '301' : '302';
        meta.appendChild(statusPill);
      }

      if (hasLink && !destMatches) {
        const mismatchPill = document.createElement('span');
        mismatchPill.className = 'tool-pill shortlinks-pill-mismatch';
        mismatchPill.textContent = 'Destination differs';
        mismatchPill.title = project.destination;
        meta.appendChild(mismatchPill);
      }

      const clicksPill = document.createElement('button');
      clicksPill.type = 'button';
      clicksPill.className = 'tool-pill shortlinks-pill-button';
      clicksPill.textContent = `${hasLink ? (Number(link.clicks) || 0) : 0} clicks`;
      clicksPill.disabled = !hasLink;
      clicksPill.addEventListener('click', () => {
        if (!hasLink) return;
        openClicksModal(expectedSlug);
      });
      meta.appendChild(clicksPill);

      titleWrap.appendChild(projectName);
      titleWrap.appendChild(slugCode);
      titleWrap.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'shortlinks-actions';

      const openShort = document.createElement('a');
      openShort.className = 'btn-secondary';
      openShort.href = shortUrl || destinationUrl;
      openShort.target = '_blank';
      openShort.rel = 'noopener noreferrer';
      openShort.textContent = hasLink ? 'Open' : 'Destination';
      if (!openShort.href) openShort.setAttribute('aria-disabled', 'true');
      actions.appendChild(openShort);

      if (!hasLink) {
        const createButton = document.createElement('button');
        createButton.type = 'button';
        createButton.className = 'btn-primary';
        createButton.textContent = 'Create';
        createButton.addEventListener('click', async () => {
          await ensureProjectLinks({ only: [project], includeMismatched: true, silent: false });
        });
        actions.appendChild(createButton);
      } else if (!destMatches) {
        const fixButton = document.createElement('button');
        fixButton.type = 'button';
        fixButton.className = 'btn-primary';
        fixButton.textContent = 'Fix destination';
        fixButton.addEventListener('click', async () => {
          await ensureProjectLinks({ only: [project], includeMismatched: true, silent: false });
        });
        actions.appendChild(fixButton);
      }

      const projectMenu = buildActionMenu([
        {
          label: 'Edit link',
          onSelect: async () => {
            openEditorForLink({ slug: expectedSlug, destination: destinationUrl, disabled: !!(link && link.disabled), expiresAt: Number(link && link.expiresAt) || 0 });
          }
        },
        hasLink ? {
          label: 'Copy short URL',
          onSelect: async () => {
            await copyTextToClipboard({
              text: shortUrl,
              statusTarget: projectsStatusEl,
              successMessage: `Copied: ${shortUrl}`
            });
          }
        } : null,
        hasLink ? {
          label: 'Test redirect',
          onSelect: async () => {
            const menuButton = actions.querySelector('.shortlinks-menu-trigger');
            await testRedirect(expectedSlug, menuButton, projectsStatusEl);
          }
        } : null
      ], {
        label: 'More',
        ariaLabel: `${project.label || project.id || 'Project'} actions`
      });
      if (projectMenu) actions.appendChild(projectMenu);

      head.appendChild(titleWrap);
      head.appendChild(actions);
      card.appendChild(head);

      const linksWrap = document.createElement('div');
      linksWrap.className = 'shortlinks-item-links';

      const shortRow = document.createElement('div');
      shortRow.className = 'shortlinks-link-row';
      const shortLabel = document.createElement('span');
      shortLabel.className = 'shortlinks-link-label';
      shortLabel.textContent = 'Short';
      const shortAnchor = document.createElement('a');
      shortAnchor.className = 'shortlinks-link-value';
      shortAnchor.href = shortUrl;
      shortAnchor.target = '_blank';
      shortAnchor.rel = 'noopener noreferrer';
      shortAnchor.textContent = shortUrl;
      shortRow.appendChild(shortLabel);
      shortRow.appendChild(shortAnchor);

      const destRow = document.createElement('div');
      destRow.className = 'shortlinks-link-row';
      const destLabel = document.createElement('span');
      destLabel.className = 'shortlinks-link-label';
      destLabel.textContent = 'To';
      const destAnchor = document.createElement('a');
      destAnchor.className = 'shortlinks-link-value';
      destAnchor.href = destinationUrl;
      destAnchor.target = '_blank';
      destAnchor.rel = 'noopener noreferrer';
      destAnchor.textContent = destinationUrl;
      destRow.appendChild(destLabel);
      destRow.appendChild(destAnchor);

      linksWrap.appendChild(shortRow);
      linksWrap.appendChild(destRow);

      card.appendChild(linksWrap);
      projectsListEl.appendChild(card);
    });

    const bits = [`${total} project${total === 1 ? '' : 's'}`];
    if (missing) bits.push(`${missing} missing`);
    if (mismatched) bits.push(`${mismatched} mismatch${mismatched === 1 ? '' : 'es'}`);
    projectHealth = { total, missing, mismatched };
    setProjectsMeta(bits.join(' • '));
    renderDashboardSummary();
  }

  async function refreshProjectsSection({ ensureMissing } = {}){
    if (!projectsListEl) return;
    setStatus(projectsStatusEl, '');
    await loadDestinationsManifest();
    rebuildProjectCatalog();
    renderProjectLinks();
    if (ensureMissing) {
      void ensureProjectLinks({ silent: true });
    }
  }

  async function ensureProjectLinks(options = {}){
    if (!projectsListEl) return;
    if (ensuringProjectLinks) return;
    const silent = options && options.silent === true;
    const only = options && Array.isArray(options.only) ? options.only : null;
    const includeMismatched = options && options.includeMismatched === true;

    if (!requireToken(projectsStatusEl)) return;
    ensuringProjectLinks = true;
    setProjectsBusy(true);

    try {
      if (!destinationsManifest) await loadDestinationsManifest();
      if (!Array.isArray(projectCatalog) || projectCatalog.length === 0) rebuildProjectCatalog();

      const targets = only && only.length ? only : projectCatalog.slice();
      if (!targets.length) {
        if (!silent) setStatus(projectsStatusEl, 'No portfolio projects found.', 'warning');
        return;
      }

      const linkMap = new Map();
      allLinks.forEach(link => {
        const slug = normalizeSlugKey(link && link.slug);
        if (!slug) return;
        linkMap.set(slug, link);
      });

      const missing = targets.filter(project => {
        const slug = normalizeSlugKey(project.slug);
        return slug && !linkMap.has(slug);
      });
      const mismatched = includeMismatched
        ? targets.filter((project) => {
          const slug = normalizeSlugKey(project.slug);
          if (!slug) return false;
          const existing = linkMap.get(slug);
          if (!existing || typeof existing.destination !== 'string') return false;
          const expectedDestination = normalizeDestinationForCompare(project.destination);
          const currentDestination = normalizeDestinationForCompare(existing.destination);
          return expectedDestination && currentDestination !== expectedDestination;
        })
        : [];

      if (!missing.length && !mismatched.length) {
        if (!silent) setStatus(projectsStatusEl, 'All selected project links are already in sync.', 'success');
        return;
      }

      if (!silent) {
        const totalCreates = missing.length;
        const totalFixes = mismatched.length;
        const bits = [];
        if (totalCreates) bits.push(`${totalCreates} missing`);
        if (totalFixes) bits.push(`${totalFixes} mismatched`);
        setStatus(projectsStatusEl, `Syncing project links (${bits.join(', ')})…`);
      }

      let createdCount = 0;
      let fixedCount = 0;
      const errors = [];

      for (const project of missing) {
        const slug = normalizeSlugInput(project.slug);
        const destination = normalizeDestinationForSave(project.destination);
        if (!slug || !destination) continue;
        try {
          const data = await api('/api/short-links', {
            method: 'POST',
            body: JSON.stringify({ slug, destination, permanent: true, expiresAt: 0 })
          });
          if (data && data.link) {
            upsertLinkInMemory(data.link);
            createdCount += 1;
          }
        } catch (err) {
          errors.push(`${slug}: ${err.message}`);
        }
      }

      for (const project of mismatched) {
        const slug = normalizeSlugInput(project.slug);
        const destination = normalizeDestinationForSave(project.destination);
        if (!slug || !destination) continue;
        try {
          const data = await api('/api/short-links', {
            method: 'POST',
            body: JSON.stringify({ slug, destination, permanent: true, expiresAt: 0 })
          });
          if (data && data.link) {
            upsertLinkInMemory(data.link);
            fixedCount += 1;
          }
        } catch (err) {
          errors.push(`${slug}: ${err.message}`);
        }
      }

      if (createdCount || fixedCount) {
        applyFilterAndRender();
        renderProjectLinks();
        markSessionDirty();
      }

      if (!silent) {
        const totalCreates = missing.length;
        const totalFixes = mismatched.length;
        const expectedTotal = totalCreates + totalFixes;
        const appliedTotal = createdCount + fixedCount;
        const detail = [];
        if (totalCreates) detail.push(`${createdCount}/${totalCreates} created`);
        if (totalFixes) detail.push(`${fixedCount}/${totalFixes} fixed`);
        const msg = appliedTotal === expectedTotal
          ? `Project links synced (${detail.join(', ')}).`
          : `Project links synced (${detail.join(', ')}). Updated ${appliedTotal} of ${expectedTotal}.`;
        setStatus(projectsStatusEl, errors.length ? `${msg} ${errors[0]}` : msg, errors.length ? 'warning' : 'success');
      }
    } finally {
      ensuringProjectLinks = false;
      setProjectsBusy(false);
    }
  }

  function renderDestinations(){
    if (!destinationResults) return;
    destinationResults.replaceChildren();

    const pages = getFilteredDestinations();
    const query = getDestinationQuery();
    const audienceKey = getSelectedAudienceKey();
    if (!pages.length) {
      const empty = document.createElement('p');
      empty.className = 'shortlinks-picker-empty';
      empty.textContent = query ? `No matches for "${query}".` : 'No destinations found.';
      destinationResults.appendChild(empty);
      return;
    }

    const groupOrder = ['Pages', 'Tools', 'Portfolio', 'Demos'];
    const grouped = new Map();
    pages.forEach(item => {
      const group = item.group && groupOrder.includes(item.group) ? item.group : 'Pages';
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(item);
    });

    groupOrder.forEach(group => {
      const items = grouped.get(group);
      if (!items || !items.length) return;

      const section = document.createElement('section');
      section.className = 'shortlinks-picker-group';

      const title = document.createElement('h4');
      title.className = 'shortlinks-picker-group-title';
      title.textContent = group;
      section.appendChild(title);

      const list = document.createElement('div');
      list.className = 'shortlinks-picker-group-list';

      items.forEach(item => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'shortlinks-picker-item';
        const displayPath = buildDisplayPath(item.path, audienceKey);

        const label = document.createElement('span');
        label.className = 'shortlinks-picker-item-label';
        label.textContent = item.label || item.path;

        const pathCode = document.createElement('code');
        pathCode.className = 'shortlinks-picker-item-path';
        pathCode.textContent = displayPath;

        button.appendChild(label);
        button.appendChild(pathCode);

        button.addEventListener('click', () => {
          const absolute = buildDisplayDestination(item.path, audienceKey);
          destinationInput.value = absolute;
          syncEditorAudienceState({ announce: false });
          setStatus(editorStatusEl, `Selected ${displayPath}`, 'success');
          closeDestinationPicker();
          destinationInput.focus();
        });

        list.appendChild(button);
      });

      section.appendChild(list);
      destinationResults.appendChild(section);
    });
  }

  async function openDestinationPicker(){
    if (!destinationModal) return;
    destinationModalPrevFocus = document.activeElement;
    destinationModal.classList.add('active');
    destinationModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');

    await loadDestinationsManifest();
    renderDestinations();

    if (destinationSearch) {
      destinationSearch.focus({ preventScroll: true });
    }
  }

  function closeClicksModal(){
    if (!clicksModal) return;
    clicksModal.classList.remove('active');
    clicksModal.setAttribute('aria-hidden', 'true');
    syncModalOpenState();
    if (clicksSlugEl) clicksSlugEl.textContent = '';
    if (clicksListEl) clicksListEl.replaceChildren();
    if (clicksMetaEl) clicksMetaEl.textContent = '';
    setStatus(clicksStatusEl, '');
    activeClicksSlug = '';
    if (clicksModalPrevFocus && document.contains(clicksModalPrevFocus)) {
      clicksModalPrevFocus.focus();
    }
    clicksModalPrevFocus = null;
  }

  function closeTemporaryModal(){
    if (!temporaryModal) return;
    temporaryModal.classList.remove('active');
    temporaryModal.setAttribute('aria-hidden', 'true');
    syncModalOpenState();
    pendingTemporaryPayload = null;
    setStatus(temporaryStatusEl, '');
    if (temporaryModalPrevFocus && document.contains(temporaryModalPrevFocus)) {
      temporaryModalPrevFocus.focus();
    }
    temporaryModalPrevFocus = null;
  }

  function openTemporaryModal(payload){
    if (!temporaryModal) return;
    pendingTemporaryPayload = payload || null;
    temporaryModalPrevFocus = document.activeElement;
    temporaryModal.classList.add('active');
    temporaryModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    setStatus(temporaryStatusEl, '');
    if (temporaryValueInput) {
      temporaryValueInput.focus({ preventScroll: true });
      try { temporaryValueInput.select(); } catch {}
    }
  }

  function formatTimestamp(value){
    const raw = typeof value === 'string' ? value : '';
    if (!raw) return '';
    const dt = new Date(raw);
    if (!Number.isFinite(dt.getTime())) return raw;
    return dt.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  function renderClickHistory(items){
    if (!clicksListEl) return;
    clicksListEl.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'shortlinks-clicks-empty';
      empty.textContent = 'No click events found (history starts after click logging was enabled).';
      clicksListEl.appendChild(empty);
      return;
    }

    items.forEach(item => {
      const card = document.createElement('article');
      card.className = 'shortlinks-click';

      const top = document.createElement('div');
      top.className = 'shortlinks-click-top';

      const when = document.createElement('time');
      const clickedAt = typeof item.clickedAt === 'string' ? item.clickedAt : '';
      when.className = 'shortlinks-click-when';
      if (clickedAt) when.dateTime = clickedAt;
      when.textContent = formatTimestamp(clickedAt) || clickedAt || 'Unknown time';

      const pills = document.createElement('div');
      pills.className = 'shortlinks-click-pills';

      const statusCode = Number.isFinite(Number(item.statusCode)) ? Number(item.statusCode) : 0;
      if (statusCode) {
        const statusPill = document.createElement('span');
        statusPill.className = 'tool-pill';
        statusPill.textContent = String(statusCode);
        pills.appendChild(statusPill);
      }

      const host = typeof item.host === 'string' ? item.host : '';
      if (host) {
        const hostPill = document.createElement('span');
        hostPill.className = 'tool-pill shortlinks-click-pill-muted';
        hostPill.textContent = host;
        pills.appendChild(hostPill);
      }

      const geoParts = [item.city, item.region, item.country]
        .filter(part => typeof part === 'string' && part.trim())
        .map(part => String(part).trim());
      if (geoParts.length) {
        const geoPill = document.createElement('span');
        geoPill.className = 'tool-pill shortlinks-click-pill-muted';
        geoPill.textContent = geoParts.join(', ');
        pills.appendChild(geoPill);
      }

      top.appendChild(when);
      top.appendChild(pills);

      const rows = document.createElement('div');
      rows.className = 'shortlinks-click-rows';

      function addRow(label, value, url){
        if (!value) return;
        const row = document.createElement('div');
        row.className = 'shortlinks-click-row';

        const labelEl = document.createElement('span');
        labelEl.className = 'shortlinks-click-label';
        labelEl.textContent = label;

        let valueEl;
        if (url) {
          const anchor = document.createElement('a');
          anchor.className = 'shortlinks-click-value';
          anchor.href = url;
          anchor.target = '_blank';
          anchor.rel = 'noopener noreferrer';
          anchor.textContent = value;
          valueEl = anchor;
        } else {
          const text = document.createElement('span');
          text.className = 'shortlinks-click-value';
          text.textContent = value;
          valueEl = text;
        }

        row.appendChild(labelEl);
        row.appendChild(valueEl);
        rows.appendChild(row);
      }

      const destination = typeof item.destination === 'string' ? item.destination : '';
      addRow('To', destination, destination);

      const referer = typeof item.referer === 'string' ? item.referer : '';
      addRow('From', referer, referer);

      const userAgent = typeof item.userAgent === 'string' ? item.userAgent : '';
      addRow('UA', userAgent);

      card.appendChild(top);
      card.appendChild(rows);

      clicksListEl.appendChild(card);
    });
  }

  async function refreshClickHistory(slug){
    if (!slug) return;
    if (!clicksModal) return;
    setStatus(clicksStatusEl, 'Loading click history…');
    if (clicksMetaEl) clicksMetaEl.textContent = '';

    try {
      const data = await api(`/api/short-links/clicks/${encodeURIComponent(slug)}?limit=${CLICK_HISTORY_LIMIT}`, { method: 'GET' });
      const events = Array.isArray(data.clicks) ? data.clicks : [];
      renderClickHistory(events);
      const countLabel = events.length === 1 ? 'event' : 'events';
      if (clicksMetaEl) clicksMetaEl.textContent = `Showing ${events.length} ${countLabel}.`;
      setStatus(clicksStatusEl, events.length ? '' : 'No events yet.', events.length ? 'success' : 'success');
    } catch (err) {
      if (clicksListEl) clicksListEl.replaceChildren();
      setStatus(clicksStatusEl, err.message, 'error');
    }
  }

  async function openClicksModal(slug){
    if (!clicksModal) return;
    const cleanSlug = normalizeSlugInput(slug);
    if (!cleanSlug) return;
    clicksModalPrevFocus = document.activeElement;
    activeClicksSlug = cleanSlug;
    if (clicksSlugEl) clicksSlugEl.textContent = buildPublicPath(cleanSlug) || cleanSlug;

    clicksModal.classList.add('active');
    clicksModal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');

    await refreshClickHistory(cleanSlug);
  }

  function makeIcon(name){
    const paths = {
      copy: [
        '<rect x="9" y="9" width="10" height="10" rx="2"></rect>',
        '<path d="M5 15V5a2 2 0 0 1 2-2h10"></path>'
      ],
      test: [
        '<path d="M4 12h4l2-6 4 12 2-6h4"></path>'
      ],
      open: [
        '<path d="M14 3h7v7"></path>',
        '<path d="M10 14 21 3"></path>',
        '<path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"></path>'
      ]
    };
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = (paths[name] || paths.open).join('');
    return svg;
  }

  function makeIconButton({ icon, label, href, onClick }){
    const control = href ? document.createElement('a') : document.createElement('button');
    control.className = 'shortlinks-icon-button';
    control.setAttribute('aria-label', label);
    control.title = label;
    if (href) {
      control.href = href;
      control.target = '_blank';
      control.rel = 'noopener noreferrer';
    } else {
      control.type = 'button';
      if (typeof onClick === 'function') control.addEventListener('click', onClick);
    }
    control.appendChild(makeIcon(icon));
    const text = document.createElement('span');
    text.className = 'visually-hidden';
    text.textContent = label;
    control.appendChild(text);
    return control;
  }

  function flashButtonText(button, text){
    if (!button) return;
    if (button.classList && button.classList.contains('shortlinks-icon-button')) {
      const originalLabel = button.getAttribute('aria-label') || '';
      const originalTitle = button.title || '';
      button.setAttribute('aria-label', text);
      button.title = text;
      button.classList.add('is-flashing');
      window.setTimeout(() => {
        button.setAttribute('aria-label', originalLabel);
        button.title = originalTitle;
        button.classList.remove('is-flashing');
      }, 1200);
      return;
    }
    const original = button.textContent;
    button.textContent = text;
    window.setTimeout(() => {
      if (button.textContent === text) button.textContent = original;
    }, 1200);
  }

  function renderTestResult(statusEl, detail, tone, openTarget){
    if (!statusEl) return;
    statusEl.replaceChildren();
    if (tone) statusEl.dataset.tone = tone;
    else delete statusEl.dataset.tone;

    const wrap = document.createElement('div');
    wrap.className = 'shortlinks-test-result';
    const copy = document.createElement('span');
    copy.className = 'shortlinks-test-result-copy';
    copy.textContent = detail;
    wrap.appendChild(copy);

    if (openTarget) {
      const open = document.createElement('a');
      open.className = 'btn-secondary shortlinks-test-open';
      open.href = openTarget;
      open.target = '_blank';
      open.rel = 'noopener noreferrer';
      open.textContent = 'Open destination';
      wrap.appendChild(open);
    }

    statusEl.appendChild(wrap);
  }

  function recordLinkHealthFromResult(slug, data, elapsedMs){
    const key = normalizeSlugInput(slug);
    if (!key) return null;
    const check = data && typeof data.check === 'object' ? data.check : null;
    const checkOk = check && check.ok === true;
    const checkStatus = check && Number.isFinite(Number(check.status)) ? Number(check.status) : 0;
    const checkMethod = check && typeof check.method === 'string' ? check.method : '';
    const checkMs = check && Number.isFinite(Number(check.ms)) ? Number(check.ms) : 0;
    const checkUrl = check && typeof check.url === 'string' ? check.url : '';
    const checkError = check && typeof check.error === 'string' ? check.error : '';
    const tone = check && check.ok === false
      ? (checkStatus ? 'warning' : 'error')
      : 'success';
    const record = {
      key: tone === 'success' ? 'healthy' : (tone === 'error' ? 'broken' : 'warning'),
      label: tone === 'success' ? 'Healthy' : (tone === 'error' ? 'Broken' : 'Destination warning'),
      tone,
      note: tone === 'success'
        ? `${checkMethod || 'Check'} ${checkStatus || 200} in ${checkMs || elapsedMs}ms`
        : (checkError || (checkStatus ? `Destination returned ${checkStatus}` : 'Destination check failed')),
      status: checkStatus,
      method: checkMethod,
      finalUrl: checkUrl,
      checkedAt: new Date().toISOString()
    };
    linkHealth.set(key, record);
    return record;
  }

  function recordLinkHealthError(slug, err){
    const key = normalizeSlugInput(slug);
    if (!key) return null;
    const record = {
      key: 'broken',
      label: 'Broken',
      tone: 'error',
      note: err && err.message ? err.message : 'Redirect test failed',
      checkedAt: new Date().toISOString()
    };
    linkHealth.set(key, record);
    return record;
  }

  async function testRedirect(slug, button, statusTarget){
    const clean = normalizeSlugInput(slug);
    if (!clean) return;

    const label = buildPublicPath(clean) || clean;
    const statusEl = statusTarget || listStatusEl;
    const start = Date.now();

    const flashText = (text) => {
      if (!button) return;
      flashButtonText(button, text);
    };

    if (button) {
      button.disabled = true;
      if (button.classList && button.classList.contains('shortlinks-icon-button')) {
        button.classList.add('is-flashing');
      } else {
        button.dataset.originalText = button.textContent || '';
        button.textContent = 'Testing...';
      }
    }
    setStatus(statusEl, `Testing ${label}…`);

    try {
      const data = await api(`/api/short-links/test/${encodeURIComponent(clean)}`, { method: 'GET' });
      const ms = Date.now() - start;
      recordLinkHealthFromResult(clean, data, ms);

      const redirect = data && typeof data.redirect === 'object' ? data.redirect : null;
      const destination = redirect && typeof redirect.destination === 'string' ? redirect.destination : '';
      const code = redirect && Number.isFinite(Number(redirect.statusCode)) ? Number(redirect.statusCode) : 0;

      const check = data && typeof data.check === 'object' ? data.check : null;
      const checkOk = check && check.ok === true;
      const checkStatus = check && Number.isFinite(Number(check.status)) ? Number(check.status) : 0;
      const checkMethod = check && typeof check.method === 'string' ? check.method : '';
      const checkMs = check && Number.isFinite(Number(check.ms)) ? Number(check.ms) : 0;
      const checkUrl = check && typeof check.url === 'string' ? check.url : '';
      const checkError = check && typeof check.error === 'string' ? check.error : '';

      const displayDest = destination.length > 200 ? `${destination.slice(0, 197)}…` : destination;
      const redirectLine = code && destination ? `${code} → ${displayDest}` : (destination || 'Redirect configured');
      let detail = `Redirect OK: ${redirectLine} (${ms}ms)`;

      if (checkMethod) {
        if (checkOk) {
          const finalLabel = checkUrl ? (checkUrl.length > 200 ? `${checkUrl.slice(0, 197)}…` : checkUrl) : '';
          const finalBit = finalLabel ? ` → ${finalLabel}` : '';
          detail += ` • Destination ${checkMethod} ${checkStatus}${finalBit} (${checkMs || ms}ms)`;
        } else {
          const errorLabel = checkError || (checkStatus ? `Status ${checkStatus}` : 'Unreachable');
          detail += ` • Destination check failed (${errorLabel})`;
        }
      }

      const tone = check && check.ok === false ? 'warning' : 'success';
      const openTarget = checkUrl || destination;
      renderTestResult(statusEl, detail, tone, openTarget);
      applyFilterAndRender();
      flashText(check && check.ok === false ? 'Warn' : 'OK');
    } catch (err) {
      recordLinkHealthError(clean, err);
      applyFilterAndRender();
      setStatus(statusEl, err && err.message ? err.message : 'Test failed.', 'error');
      flashText('Failed');
    } finally {
      if (button) {
        button.disabled = false;
        if (button.classList && button.classList.contains('shortlinks-icon-button')) {
          button.classList.remove('is-flashing');
        } else if (button.textContent === 'Testing...' && button.dataset.originalText) {
          button.textContent = button.dataset.originalText;
          delete button.dataset.originalText;
        }
      }
    }
  }

  async function testSelectedLinks(){
    const links = getSelectedLinks();
    if (!links.length) {
      setStatus(listStatusEl, 'Select at least one visible link to test.', 'warning');
      return;
    }
    const total = links.length;
    let healthy = 0;
    let warnings = 0;
    let skipped = 0;
    setStatus(listStatusEl, `Testing ${total} selected link${total === 1 ? '' : 's'}...`);
    for (let index = 0; index < links.length; index += 1) {
      const link = links[index];
      const slug = normalizeSlugInput(link && link.slug);
      if (!slug) continue;
      if (link.disabled || isLinkExpired(link)) {
        skipped += 1;
        continue;
      }
      try {
        const started = Date.now();
        const data = await api(`/api/short-links/test/${encodeURIComponent(slug)}`, { method: 'GET' });
        const record = recordLinkHealthFromResult(slug, data, Date.now() - started);
        if (record && record.tone === 'success') healthy += 1;
        else warnings += 1;
      } catch (err) {
        recordLinkHealthError(slug, err);
        warnings += 1;
      }
      setStatus(listStatusEl, `Tested ${index + 1}/${total} selected link${total === 1 ? '' : 's'}...`);
    }
    applyFilterAndRender();
    const skippedBit = skipped ? `, ${skipped} skipped` : '';
    setStatus(
      listStatusEl,
      warnings
        ? `Tested ${total} link${total === 1 ? '' : 's'}: ${healthy} healthy, ${warnings} warning${warnings === 1 ? '' : 's'}${skippedBit}.`
        : `Tested ${total} link${total === 1 ? '' : 's'}: ${healthy} healthy${skippedBit}.`,
      warnings ? 'warning' : 'success'
    );
  }

  function exportCurrentView(){
    const links = getFilteredLinks();
    if (!links.length) {
      setStatus(listStatusEl, 'Nothing in the current view to export.', 'warning');
      return;
    }
    exportRedirectsOnly(links);
  }

  async function api(path, options = {}){
    const token = getSavedToken();
    const headers = Object.assign({}, options.headers || {});
    if (token) headers.Authorization = `Bearer ${token}`;

    const hasBody = typeof options.body !== 'undefined';
    if (hasBody && !headers['Content-Type']) headers['Content-Type'] = 'application/json';

    const resp = await fetch(path, Object.assign({}, options, { headers }));
    const isJson = (resp.headers.get('content-type') || '').includes('application/json');
    const data = isJson ? await resp.json().catch(() => null) : null;
    if (!resp.ok || !data || data.ok !== true) {
      const errMsg = (data && data.error) ? data.error : `Request failed (${resp.status})`;
      const err = new Error(errMsg);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  async function apiInspect(path){
    const token = getSavedToken();
    const headers = {};
    if (token) headers.Authorization = `Bearer ${token}`;

    const resp = await fetch(path, { method: 'GET', headers });
    const isJson = (resp.headers.get('content-type') || '').includes('application/json');
    const data = isJson ? await resp.json().catch(() => null) : null;
    return { status: resp.status, data };
  }

  function formatHealthPayload(payload){
    if (!payload || typeof payload !== 'object') return '';
    const debugBits = [];
    const aws = payload.aws || {};
    if (aws.accessKeyIdSource) debugBits.push(`Creds: ${aws.accessKeyIdSource}.`);
    if (aws.secretFingerprint) debugBits.push(`Secret fp: ${aws.secretFingerprint}.`);
    if (aws.sessionTokenConfigured) {
      debugBits.push(`Session token: ${aws.sessionTokenUsed ? 'used' : 'ignored'}.`);
    }
    const debug = debugBits.length ? ` ${debugBits.join(' ')}` : '';

	    if (payload.ok === true) {
	      const keyId = payload.aws && payload.aws.accessKeyId ? payload.aws.accessKeyId : '';
	      const table = payload.table && payload.table.name ? payload.table.name : '';
	      const region = payload.aws && payload.aws.region ? payload.aws.region : '';
	      const status = payload.table && payload.table.status ? payload.table.status : '';
	      const billing = payload.table && payload.table.billingMode ? payload.table.billingMode : '';
	      const clicks = payload.clicks || {};
	      const bits = [
	        `Backend OK${table ? `: ${table}` : ''}${status ? ` (${status})` : ''}.`,
	        region ? `Region: ${region}.` : '',
	        billing ? `Billing: ${billing}.` : '',
	        keyId ? `Access key: ${keyId}.` : ''
	      ].filter(Boolean);
	      if (clicks && clicks.configured) {
	        const clicksName = clicks.table && clicks.table.name ? clicks.table.name : '';
	        const clicksStatus = clicks.table && clicks.table.status ? clicks.table.status : '';
	        bits.push(`Click log${clicksName ? `: ${clicksName}` : ''}${clicksStatus ? ` (${clicksStatus})` : ''}.`);
	        if (clicks.error && (clicks.error.name || clicks.error.message)) {
	          const errName = clicks.error.name ? String(clicks.error.name) : '';
	          const errMsg = clicks.error.message ? String(clicks.error.message) : '';
	          const compact = [errName, errMsg].filter(Boolean).join(': ');
	          const clipped = compact.length > 140 ? `${compact.slice(0, 137)}…` : compact;
	          if (clipped) bits.push(`Click log error: ${clipped}.`);
	        }
	      } else {
	        bits.push('Click log: not configured.');
	      }
	      return bits.join(' ') + debug;
	    }

    const details = payload.details || {};
    const name = details.name ? String(details.name) : '';
    const message = details.message ? String(details.message) : '';
    const base = payload.error ? String(payload.error) : 'Backend check failed';
    const extra = [name, message].filter(Boolean).join(': ');
    return (extra ? `${base} (${extra})` : base) + debug;
  }

  function healthHints(payload){
    if (!payload || typeof payload !== 'object') return '';
    const details = payload.details || {};
    const message = details.message ? String(details.message) : '';
    const name = details.name ? String(details.name) : '';

    if (payload.aws && payload.aws.secretTrimmed) {
      return 'Your AWS secret appears to have leading/trailing whitespace. Re-save it in Vercel (or redeploy after trimming).';
    }
    if (payload.aws && payload.aws.sessionTokenIgnored) {
      return 'AWS_SESSION_TOKEN is set, but your access key looks like a long-term key (AKIA). Remove AWS_SESSION_TOKEN in Vercel, or set SHORTLINKS_AWS_ACCESS_KEY_ID/SHORTLINKS_AWS_SECRET_ACCESS_KEY (preferred) and leave the session token unset.';
    }
    if (name === 'UnrecognizedClientException' || /security token.*invalid/i.test(message)) {
      return 'AWS rejected the key pair. Re-copy the access key + secret from the same CSV row (no quotes/whitespace) and redeploy. If you have duplicate AWS_* vars in Vercel, set SHORTLINKS_AWS_ACCESS_KEY_ID/SHORTLINKS_AWS_SECRET_ACCESS_KEY instead.';
    }
    if (name === 'AccessDeniedException') {
      return 'AWS credentials are valid but lack DynamoDB permissions for this table.';
    }
    if (name === 'ResourceNotFoundException') {
      return 'Table not found. Double-check AWS_REGION and SHORTLINKS_DDB_TABLE.';
    }
    return '';
  }

  function clearList(){
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
  }

  function normalizeOrigin(origin){
    const raw = typeof origin === 'string' ? origin.trim() : '';
    if (!raw) return '';
    try {
      const url = new URL(raw);
      return `${url.protocol}//${url.host}`;
    } catch {
      return raw.replace(/\/+$/g, '');
    }
  }

  function isShortDomainHost(hostname){
    const host = String(hostname || '').toLowerCase();
    return host === 'dshort.me' || host === 'www.dshort.me';
  }

  function getPublicOrigin(){
    if (isPreviewHost(window.location.hostname) || isShortDomainHost(window.location.hostname)) {
      return normalizeOrigin(window.location.origin) || window.location.origin;
    }
    return normalizeOrigin(DEFAULT_PUBLIC_ORIGIN) || window.location.origin;
  }

  function getPublicBasePath(){
    if (isPreviewHost(window.location.hostname)) return basePath;
    return '';
  }

  function buildPublicPath(slug){
    const clean = String(slug || '').replace(/^\/+|\/+$/g, '');
    if (!clean) return '';
    const prefix = String(getPublicBasePath() || '').replace(/^\/+|\/+$/g, '');
    return prefix ? `/${prefix}/${clean}` : `/${clean}`;
  }

  function buildShortUrl(slug){
    const origin = getPublicOrigin();
    const path = buildPublicPath(slug);
    return path ? `${origin}${path}` : origin;
  }

  function normalizeSlugInput(value){
    return String(value || '').trim().replace(/^\/+|\/+$/g, '');
  }

  function normalizeSlugKey(value){
    return normalizeSlugInput(value).toLowerCase();
  }

  function clampRandomLength(value, fallback = DEFAULT_RANDOM_LENGTH){
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(MIN_RANDOM_LENGTH, Math.min(MAX_RANDOM_LENGTH, Math.floor(numeric)));
  }

  function normalizeDurationValue(value, fallback = DEFAULT_SET_DURATION_VALUE){
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(1, Math.min(365, Math.floor(numeric)));
  }

  function normalizeDurationUnit(value){
    const raw = String(value || '').trim().toLowerCase();
    if (raw === 'hours' || raw === 'weeks') return raw;
    return DEFAULT_SET_DURATION_UNIT;
  }

  function normalizeExpirationMode(value){
    return String(value || '').trim().toLowerCase() === 'temporary' ? 'temporary' : 'permanent';
  }

  function getCreateExpirationMode(){
    return normalizeExpirationMode(expirationModeSelect?.value);
  }

  function syncCreateTimingVisibility(){
    const mode = getCreateExpirationMode();
    if (expirationDurationFields) expirationDurationFields.hidden = mode !== 'temporary';
    if (expirationDurationValueInput) expirationDurationValueInput.disabled = mode !== 'temporary';
    if (expirationDurationUnitSelect) expirationDurationUnitSelect.disabled = mode !== 'temporary';
  }

  function getCreateExpirationConfig(){
    const mode = getCreateExpirationMode();
    if (mode !== 'temporary') {
      return {
        permanent: true,
        expiresAt: 0
      };
    }

    const rawValue = Number(expirationDurationValueInput?.value);
    const durationValue = Number.isFinite(rawValue) ? Math.floor(rawValue) : NaN;
    if (!Number.isFinite(durationValue) || durationValue <= 0) {
      setStatus(editorStatusEl, 'Enter a duration greater than 0.', 'error');
      return null;
    }

    const durationUnit = String(expirationDurationUnitSelect?.value || '').trim().toLowerCase();
    const secondsPerUnit = unitToSeconds(durationUnit);
    if (!secondsPerUnit) {
      setStatus(editorStatusEl, 'Select a valid duration unit.', 'error');
      return null;
    }

    const totalSeconds = durationValue * secondsPerUnit;
    const maxSeconds = 60 * 60 * 24 * 366;
    if (!Number.isFinite(totalSeconds) || totalSeconds > maxSeconds) {
      setStatus(editorStatusEl, 'Duration too long (max 1 year).', 'error');
      return null;
    }

    return {
      permanent: false,
      expiresAt: Math.floor(Date.now() / 1000) + totalSeconds
    };
  }

  function formatCountdown(ms){
    const seconds = Math.max(0, Math.floor(Number(ms) / 1000));
    if (!Number.isFinite(seconds)) return '';
    if (seconds < 60) return 'under 1m';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 48) return `${hours}h`;
    const days = Math.floor(hours / 24);
    if (days < 14) return `${days}d`;
    const weeks = Math.floor(days / 7);
    return `${weeks}w`;
  }

  async function copyTextToClipboard({ text, button, statusTarget, successMessage }){
    try {
      await navigator.clipboard.writeText(String(text || ''));
      flashButtonText(button, 'Copied');
      setStatus(statusTarget, successMessage || `Copied: ${text}`, 'success');
      return true;
    } catch {
      flashButtonText(button, 'Copy failed');
      setStatus(statusTarget, 'Copy failed (clipboard permission blocked).', 'error');
      return false;
    }
  }

  function openEditorForLink({ slug, destination, disabled, expiresAt }){
    if (slugModeSelect) slugModeSelect.value = 'custom';
    slugInput.value = slug || '';
    destinationInput.value = destination || '';
    syncSlugModeState();
    syncEditorAudienceState({ announce: false });
    setActiveMode('single');
    slugInput.focus();
    const expiresLabel = expiresAt ? ` (expires ${new Date(expiresAt * 1000).toLocaleString()})` : '';
    setEditorMeta(slug ? `Editing ${buildPublicPath(slug)}` : 'New short link');
    setStatus(
      editorStatusEl,
      slug ? `Editing ${buildPublicPath(slug)}${disabled ? ' (disabled)' : ''}${expiresLabel}` : 'Ready to create a new link.',
      'success'
    );
  }

  function startNewLinkFromList(){
    clearEditor();
    setActiveMode('single');
    setEditorMeta('New short link');
    setStatus(editorStatusEl, 'Ready to create a new short link.', 'success');
    if (destinationInput && typeof destinationInput.focus === 'function') {
      destinationInput.focus({ preventScroll: false });
    }
  }

  async function setLinkDisabled(link, nextDisabled, statusTarget){
    if (!link || !normalizeSlugInput(link.slug)) return;
    if (nextDisabled) {
      const ok = window.confirm(`Disable ${buildPublicPath(link.slug)}?`);
      if (!ok) return;
    }
    try {
      await api(`/api/short-links/${encodeURIComponent(link.slug)}`, {
        method: 'PATCH',
        body: JSON.stringify({ disabled: nextDisabled })
      });
      setStatus(statusTarget || listStatusEl, `${nextDisabled ? 'Disabled' : 'Enabled'} ${link.slug}`, 'success');
      await refreshLinks();
    } catch (err) {
      setStatus(statusTarget || listStatusEl, err.message, 'error');
    }
  }

  async function deleteLinkEntry(link, statusTarget){
    if (!link || !normalizeSlugInput(link.slug)) return;
    const ok = window.confirm(`Delete ${buildPublicPath(link.slug)}?`);
    if (!ok) return;
    try {
      await api(`/api/short-links/${encodeURIComponent(link.slug)}`, { method: 'DELETE' });
      setStatus(statusTarget || listStatusEl, `Deleted ${link.slug}`, 'success');
      await refreshLinks();
    } catch (err) {
      setStatus(statusTarget || listStatusEl, err.message, 'error');
    }
  }

  function buildActionMenu(items, options = {}){
    const entries = Array.isArray(items) ? items.filter(Boolean) : [];
    if (!entries.length) return null;

    const details = document.createElement('details');
    details.className = 'shortlinks-menu';

    const summary = document.createElement('summary');
    summary.className = 'shortlinks-menu-trigger';
    summary.textContent = options.label || 'More';
    if (options.ariaLabel) summary.setAttribute('aria-label', options.ariaLabel);
    details.appendChild(summary);

    const popover = document.createElement('div');
    popover.className = 'shortlinks-menu-popover';

    entries.forEach((item) => {
      let control;
      if (item.href) {
        control = document.createElement('a');
        control.href = item.href;
        control.target = item.target || '_blank';
        control.rel = item.rel || 'noopener noreferrer';
        control.addEventListener('click', () => {
          details.open = false;
        });
      } else {
        control = document.createElement('button');
        control.type = 'button';
        control.disabled = !!item.disabled;
        control.addEventListener('click', async () => {
          details.open = false;
          if (typeof item.onSelect === 'function') await item.onSelect();
        });
      }
      control.className = `shortlinks-menu-item${item.danger ? ' shortlinks-menu-item-danger' : ''}`;
      control.textContent = item.label;
      if (item.title) control.title = item.title;
      popover.appendChild(control);
    });

    details.addEventListener('toggle', () => {
      if (!details.open) return;
      document.querySelectorAll('.shortlinks-menu[open]').forEach((menu) => {
        if (menu !== details) menu.open = false;
      });
    });

    details.appendChild(popover);
    return details;
  }

  function renderLinksTable(links){
    clearList();
    listEl.classList.add('shortlinks-list-table');

    if (!Array.isArray(links) || links.length === 0) {
      const empty = document.createElement('p');
      const query = getFilterQuery();
      empty.className = 'shortlinks-empty shortlinks-empty-state';
      empty.textContent = query ? `No matches for "${query}".` : 'No short links yet.';
      listEl.appendChild(empty);
      return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'shortlinks-table-wrap shortlinks-table-wrap-main';

    const table = document.createElement('table');
    table.className = 'shortlinks-table shortlinks-table-main';

    const caption = document.createElement('caption');
    caption.className = 'shortlinks-table-caption';
    caption.textContent = 'Short links list';
    table.appendChild(caption);

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    [
      { key: 'select', label: '' },
      { key: 'slug', label: 'Slug' },
      { key: 'destination', label: 'Destination' },
      { key: 'clicks', label: 'Clicks' },
      { key: 'actions', label: 'Actions' }
    ].forEach(col => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = col.label;
      if (col.key === 'select') th.className = 'shortlinks-table-cell-select';
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');

    links.forEach(link => {
      const shortUrl = buildShortUrl(link.slug);
      const destinationUrl = formatAbsoluteUrl(link.destination);
      const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
      const expiresMs = expiresAt ? expiresAt * 1000 - Date.now() : 0;
      const isExpired = expiresAt && expiresMs <= 0;

      const row = document.createElement('tr');
      row.className = 'shortlinks-row';
      if (link.disabled) row.classList.add('shortlinks-row-disabled');
      if (isExpired) row.classList.add('shortlinks-row-expired');

      const selectCell = document.createElement('td');
      selectCell.className = 'shortlinks-table-cell-select';
      selectCell.appendChild(makeSelectionControl(link));
      row.appendChild(selectCell);

      const slugCell = document.createElement('td');
      const slugAnchor = document.createElement('a');
      slugAnchor.className = 'shortlinks-table-slug';
      slugAnchor.href = shortUrl;
      slugAnchor.target = '_blank';
      slugAnchor.rel = 'noopener noreferrer';
      slugAnchor.title = shortUrl;

      const slugCode = document.createElement('code');
      slugCode.className = 'shortlinks-table-slug-code';
      slugCode.textContent = buildPublicPath(link.slug);

      slugAnchor.appendChild(slugCode);
      slugCell.appendChild(slugAnchor);

      const meta = document.createElement('div');
      meta.className = 'shortlinks-table-meta';

      const statusPill = document.createElement('span');
      statusPill.className = 'tool-pill';
      statusPill.textContent = link.permanent ? '301' : '302';
      meta.appendChild(statusPill);
      meta.appendChild(makeHealthPill(link));

      if (expiresAt) {
        const expiresPill = document.createElement('span');
        expiresPill.className = `tool-pill ${isExpired ? 'shortlinks-pill-expired' : 'shortlinks-pill-expiry'}`;
        expiresPill.textContent = isExpired ? 'Expired' : `Expires in ${formatCountdown(expiresMs)}`;
        expiresPill.title = `Expires ${new Date(expiresAt * 1000).toLocaleString()}`;
        meta.appendChild(expiresPill);
      }

      if (link.disabled) {
        const disabledPill = document.createElement('span');
        disabledPill.className = 'tool-pill shortlinks-pill-disabled';
        disabledPill.textContent = 'Disabled';
        meta.appendChild(disabledPill);
      }

      if (link.batchTitle || link.templateTitle) {
        const generatedPill = document.createElement('span');
        generatedPill.className = 'tool-pill shortlinks-click-pill-muted';
        generatedPill.textContent = link.batchTitle || link.templateTitle;
        generatedPill.title = link.contextCompany
          ? `${link.contextCompany}${link.contextTitle ? ` · ${link.contextTitle}` : ''}`
          : (link.templateTitle || link.batchTitle || '');
        meta.appendChild(generatedPill);
      }

      slugCell.appendChild(meta);
      row.appendChild(slugCell);

      const destCell = document.createElement('td');
      const destAnchor = document.createElement('a');
      destAnchor.className = 'shortlinks-table-destination';
      destAnchor.href = destinationUrl;
      destAnchor.target = '_blank';
      destAnchor.rel = 'noopener noreferrer';
      destAnchor.title = destinationUrl;
      destAnchor.textContent = destinationUrl;
      destCell.appendChild(destAnchor);
      row.appendChild(destCell);

      const clicksCell = document.createElement('td');
      clicksCell.className = 'shortlinks-table-cell-clicks';
      const clicksPill = document.createElement('button');
      clicksPill.type = 'button';
      clicksPill.className = 'tool-pill shortlinks-pill-button';
      clicksPill.textContent = `${Number(link.clicks) || 0} clicks`;
      clicksPill.addEventListener('click', () => {
        openClicksModal(link.slug);
      });
      clicksCell.appendChild(clicksPill);
      row.appendChild(clicksCell);

      const actionsCell = document.createElement('td');
      actionsCell.className = 'shortlinks-table-cell-actions';

      const actions = document.createElement('div');
      actions.className = 'shortlinks-table-actions shortlinks-inline-actions';

      const copyButton = makeIconButton({
        icon: 'copy',
        label: `Copy ${buildPublicPath(link.slug)}`,
        onClick: async () => {
          await copyTextToClipboard({
            text: shortUrl,
            button: copyButton,
            statusTarget: listStatusEl,
            successMessage: `Copied: ${shortUrl}`
          });
        }
      });

      const openButton = makeIconButton({
        icon: 'open',
        label: `Open ${buildPublicPath(link.slug)}`,
        href: shortUrl
      });

      const testButton = makeIconButton({
        icon: 'test',
        label: `Test ${buildPublicPath(link.slug)}`,
        onClick: () => {
          testRedirect(link.slug, testButton);
        }
      });

      actions.appendChild(copyButton);
      actions.appendChild(openButton);
      actions.appendChild(testButton);
      const rowMenu = buildActionMenu([
        {
          label: 'View details',
          onSelect: async () => {
            renderDetailPanel(link);
          }
        },
        {
          label: 'Edit link',
          onSelect: async () => {
            openEditorForLink({
              slug: link.slug,
              destination: link.destination,
              disabled: !!link.disabled,
              expiresAt: Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0
            });
          }
        },
        {
          label: 'Open short link',
          href: shortUrl
        },
        {
          label: 'Open destination',
          href: destinationUrl
        },
        {
          label: 'Test redirect',
          onSelect: async () => {
            await testRedirect(link.slug, testButton);
          }
        },
        {
          label: link.disabled ? 'Enable link' : 'Disable link',
          onSelect: async () => {
            await setLinkDisabled(link, !link.disabled, listStatusEl);
          }
        },
        {
          label: 'Delete link',
          danger: true,
          onSelect: async () => {
            await deleteLinkEntry(link, listStatusEl);
          }
        }
      ], {
        label: 'More',
        ariaLabel: `${buildPublicPath(link.slug)} actions`
      });
      if (rowMenu) actions.appendChild(rowMenu);

      actionsCell.appendChild(actions);
      row.appendChild(actionsCell);

      tbody.appendChild(row);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    wrap.appendChild(table);
    listEl.appendChild(wrap);
  }

  function renderLinks(links){
    clearList();
    listEl.classList.remove('shortlinks-list-table');
    if (!Array.isArray(links) || links.length === 0) {
      const empty = document.createElement('p');
      const query = getFilterQuery();
      empty.className = 'shortlinks-empty shortlinks-empty-state';
      empty.textContent = query ? `No matches for "${query}".` : 'No short links yet.';
      listEl.appendChild(empty);
      return;
    }

    links.forEach(link => {
      const shortUrl = buildShortUrl(link.slug);
      const destinationUrl = formatAbsoluteUrl(link.destination);
      const expiresAt = Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0;
      const expiresMs = expiresAt ? expiresAt * 1000 - Date.now() : 0;
      const isExpired = expiresAt && expiresMs <= 0;

      const card = document.createElement('article');
      card.className = 'shortlinks-item';
      if (link.disabled) card.classList.add('shortlinks-item-disabled');
      if (isExpired) card.classList.add('shortlinks-item-expired');

      const head = document.createElement('div');
      head.className = 'shortlinks-item-head';

      const titleWrap = document.createElement('div');
      titleWrap.className = 'shortlinks-item-title';

      titleWrap.appendChild(makeSelectionControl(link));

      const slugCode = document.createElement('code');
      slugCode.className = 'shortlinks-slug';
      slugCode.textContent = buildPublicPath(link.slug);

      const meta = document.createElement('div');
      meta.className = 'shortlinks-item-meta';

      const statusPill = document.createElement('span');
      statusPill.className = 'tool-pill';
      statusPill.textContent = link.permanent ? '301' : '302';
      meta.appendChild(statusPill);
      meta.appendChild(makeHealthPill(link));

      if (expiresAt) {
        const expiresPill = document.createElement('span');
        expiresPill.className = `tool-pill ${isExpired ? 'shortlinks-pill-expired' : 'shortlinks-pill-expiry'}`;
        expiresPill.textContent = isExpired ? 'Expired' : `Expires in ${formatCountdown(expiresMs)}`;
        expiresPill.title = `Expires ${new Date(expiresAt * 1000).toLocaleString()}`;
        meta.appendChild(expiresPill);
      }

      const clicksPill = document.createElement('button');
      clicksPill.type = 'button';
      clicksPill.className = 'tool-pill shortlinks-pill-button';
      clicksPill.textContent = `${Number(link.clicks) || 0} clicks`;
      clicksPill.addEventListener('click', () => {
        openClicksModal(link.slug);
      });

      if (link.disabled) {
        const disabledPill = document.createElement('span');
        disabledPill.className = 'tool-pill shortlinks-pill-disabled';
        disabledPill.textContent = 'Disabled';
        meta.appendChild(disabledPill);
      }
      if (link.batchTitle || link.templateTitle) {
        const generatedPill = document.createElement('span');
        generatedPill.className = 'tool-pill shortlinks-click-pill-muted';
        generatedPill.textContent = link.batchTitle || link.templateTitle;
        generatedPill.title = link.contextCompany
          ? `${link.contextCompany}${link.contextTitle ? ` · ${link.contextTitle}` : ''}`
          : (link.templateTitle || link.batchTitle || '');
        meta.appendChild(generatedPill);
      }
      meta.appendChild(clicksPill);
      titleWrap.appendChild(slugCode);
      titleWrap.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'shortlinks-actions shortlinks-inline-actions';

      const copyButton = makeIconButton({
        icon: 'copy',
        label: `Copy ${buildPublicPath(link.slug)}`,
        onClick: async () => {
          await copyTextToClipboard({
            text: shortUrl,
            button: copyButton,
            statusTarget: listStatusEl,
            successMessage: `Copied: ${shortUrl}`
          });
        }
      });

      const openButton = makeIconButton({
        icon: 'open',
        label: `Open ${buildPublicPath(link.slug)}`,
        href: shortUrl
      });

      const testButton = makeIconButton({
        icon: 'test',
        label: `Test ${buildPublicPath(link.slug)}`,
        onClick: () => {
          testRedirect(link.slug, testButton);
        }
      });

      actions.appendChild(copyButton);
      actions.appendChild(openButton);
      actions.appendChild(testButton);
      const cardMenu = buildActionMenu([
        {
          label: 'View details',
          onSelect: async () => {
            renderDetailPanel(link);
          }
        },
        {
          label: 'Edit link',
          onSelect: async () => {
            openEditorForLink({
              slug: link.slug,
              destination: link.destination,
              disabled: !!link.disabled,
              expiresAt: Number.isFinite(Number(link.expiresAt)) ? Number(link.expiresAt) : 0
            });
          }
        },
        {
          label: 'Open short link',
          href: shortUrl
        },
        {
          label: 'Open destination',
          href: destinationUrl
        },
        {
          label: 'Test redirect',
          onSelect: async () => {
            await testRedirect(link.slug, testButton);
          }
        },
        {
          label: link.disabled ? 'Enable link' : 'Disable link',
          onSelect: async () => {
            await setLinkDisabled(link, !link.disabled, listStatusEl);
          }
        },
        {
          label: 'Delete link',
          danger: true,
          onSelect: async () => {
            await deleteLinkEntry(link, listStatusEl);
          }
        }
      ], {
        label: 'More',
        ariaLabel: `${buildPublicPath(link.slug)} actions`
      });
      if (cardMenu) actions.appendChild(cardMenu);

      head.appendChild(titleWrap);
      head.appendChild(actions);

      const linksWrap = document.createElement('div');
      linksWrap.className = 'shortlinks-item-links';

      const shortRow = document.createElement('div');
      shortRow.className = 'shortlinks-link-row';
      const shortLabel = document.createElement('span');
      shortLabel.className = 'shortlinks-link-label';
      shortLabel.textContent = 'Short';
      const shortAnchor = document.createElement('a');
      shortAnchor.className = 'shortlinks-link-value';
      shortAnchor.href = shortUrl;
      shortAnchor.target = '_blank';
      shortAnchor.rel = 'noopener noreferrer';
      shortAnchor.textContent = shortUrl;
      shortRow.appendChild(shortLabel);
      shortRow.appendChild(shortAnchor);

      const destRow = document.createElement('div');
      destRow.className = 'shortlinks-link-row';
      const destLabel = document.createElement('span');
      destLabel.className = 'shortlinks-link-label';
      destLabel.textContent = 'To';
      const destAnchor = document.createElement('a');
      destAnchor.className = 'shortlinks-link-value';
      destAnchor.href = destinationUrl;
      destAnchor.target = '_blank';
      destAnchor.rel = 'noopener noreferrer';
      destAnchor.textContent = destinationUrl;
      destRow.appendChild(destLabel);
      destRow.appendChild(destAnchor);

      linksWrap.appendChild(shortRow);
      linksWrap.appendChild(destRow);

      card.appendChild(head);
      card.appendChild(linksWrap);

      listEl.appendChild(card);
    });
  }

  function applyFilterAndRender(){
    const filtered = getFilteredLinks();
    visibleLinkSlugs = filtered.map(link => normalizeSlugInput(link && link.slug)).filter(Boolean);
    listEl.dataset.density = getDensityMode();
    if (prefersTableLayout()) {
      renderLinksTable(filtered);
    } else {
      renderLinks(filtered);
    }
    visibleLinksCount = filtered.length;
    setCount(filtered.length, allLinks.length);
    renderHealthStrip();
    updateSelectionControls();
    if (activeDetailSlug) renderDetailPanel(getLinkBySlug(activeDetailSlug));
    renderDashboardSummary();
  }

  async function refreshLinks(){
    setStatus(listStatusEl, 'Loading…');
    setStatus(healthStatusEl, '');
    try {
      const data = await api('/api/short-links', { method: 'GET' });
      basePath = typeof data.basePath === 'string' && data.basePath.trim() ? data.basePath.trim() : DEFAULT_BASE_PATH;
      allLinks = Array.isArray(data.links) ? data.links : [];
      pruneSelectedSlugs();
      applyFilterAndRender();
      void refreshProjectsSection({ ensureMissing: true });
      setStatus(listStatusEl, `Loaded ${allLinks.length} link(s).`, 'success');
      markSessionDirty();
    } catch (err) {
      allLinks = [];
      visibleLinkSlugs = [];
      selectedSlugs = new Set();
      clearList();
      visibleLinksCount = 0;
      setCount(0, 0);
      setStatus(listStatusEl, err.message, 'error');
      renderDashboardSummary();
      markSessionDirty();
    }
  }

  async function refreshHealth(){
    setStatus(healthStatusEl, 'Checking backend…');
    try {
      const result = await apiInspect('/api/short-links/health');
      const payload = result.data || {};
      const msg = formatHealthPayload(payload) || `Backend check failed (${result.status})`;
      if (payload.ok === true) {
        setStatus(healthStatusEl, msg, 'success');
        return;
      }
      const hint = healthHints(payload);
      setStatus(healthStatusEl, hint ? `${msg} ${hint}` : msg, 'error');
    } catch (err) {
      setStatus(healthStatusEl, err.message || 'Backend check failed.', 'error');
    }
  }

  function getSelectedSlugMode(){
    return slugModeSelect && String(slugModeSelect.value || '').trim().toLowerCase() === 'random'
      ? 'random'
      : 'custom';
  }

  function syncSlugModeState(){
    const mode = getSelectedSlugMode();
    if (slugFieldEl) slugFieldEl.hidden = mode !== 'custom';
    if (randomLengthFieldEl) randomLengthFieldEl.hidden = mode !== 'random';
    if (slugInput) slugInput.disabled = mode !== 'custom';
    if (randomLengthInput) {
      randomLengthInput.disabled = mode !== 'random';
      randomLengthInput.value = String(clampRandomLength(randomLengthInput.value));
    }

    if (!String(editorStatusEl?.dataset?.tone || '').trim()) {
      setEditorMeta(mode === 'random' ? 'Random short code' : 'New short link');
    }
  }

  function getEditorPayload(){
    const audienceKey = getSelectedAudienceKey();
    const slugMode = getSelectedSlugMode();
    const slug = normalizeSlugInput(slugInput.value);
    const randomLength = clampRandomLength(randomLengthInput?.value, DEFAULT_RANDOM_LENGTH);
    const destination = buildStoredDestination(destinationInput.value, audienceKey);

    if (slugMode === 'custom' && !slug) {
      setStatus(editorStatusEl, 'Slug is required.', 'error');
      return null;
    }
    if (!destination) {
      setStatus(editorStatusEl, 'Destination is required.', 'error');
      return null;
    }

    return { slug, slugMode, randomLength, destination, audienceKey };
  }

  function setEditorBusy(isBusy){
    const busy = !!isBusy;
    const controls = [
      createLinkButton,
      getPermanentButton,
      getTemporaryButton,
      clearButton,
      destinationPickerOpen,
      audienceSelect,
      slugModeSelect,
      slugInput,
      randomLengthInput,
      expirationModeSelect,
      expirationDurationValueInput,
      expirationDurationUnitSelect
    ];
    controls.forEach(control => {
      if (!control) return;
      control.disabled = busy;
    });
  }

  async function createOrUpdateLink({ slug, slugMode, randomLength, destination, permanent, expiresAt, statusEl, audienceKey }){
    const targetStatus = statusEl || editorStatusEl;
    if (!getSavedToken()) {
      setStatus(targetStatus, 'Admin token required.', 'error');
      return null;
    }

    setEditorBusy(true);
    const creatingRandom = slugMode === 'random';
    setStatus(targetStatus, permanent ? 'Creating permanent link…' : 'Creating temporary link…');
    try {
      const body = {
        destination,
        permanent: !!permanent,
        slugMode: creatingRandom ? 'random' : 'custom'
      };
      if (creatingRandom) body.randomLength = clampRandomLength(randomLength, DEFAULT_RANDOM_LENGTH);
      else body.slug = slug;
      if (typeof expiresAt !== 'undefined') body.expiresAt = expiresAt;
      const data = await api('/api/short-links', {
        method: 'POST',
        body: JSON.stringify(body)
      });
      const savedLink = data && data.link ? data.link : { slug, destination };
      if (savedLink && savedLink.slug) upsertLinkInMemory(savedLink);

      const resolvedSlug = normalizeSlugInput(savedLink?.slug || slug);
      const resolvedDestination = typeof savedLink?.destination === 'string' && savedLink.destination.trim()
        ? savedLink.destination
        : destination;
      const shortUrl = buildShareShortUrl(resolvedSlug, resolvedDestination, audienceKey);
      let copied = false;
      try {
        await navigator.clipboard.writeText(shortUrl);
        copied = true;
      } catch {}

      const label = permanent ? 'Permanent link' : 'Temporary link';
      setEditorMeta(`Saved ${buildPublicPath(resolvedSlug)}`);
      setStatus(editorStatusEl, `${label}: ${shortUrl}${copied ? ' (copied)' : ''}`, 'success');
      markSessionDirty();
      await refreshLinks();
      return shortUrl;
    } catch (err) {
      setStatus(targetStatus, err.message, 'error');
      markSessionDirty();
      return null;
    } finally {
      setEditorBusy(false);
    }
  }

  function clearEditor(){
    slugInput.value = '';
    destinationInput.value = '';
    if (randomLengthInput) randomLengthInput.value = String(DEFAULT_RANDOM_LENGTH);
    if (expirationModeSelect) expirationModeSelect.value = 'permanent';
    if (expirationDurationValueInput) expirationDurationValueInput.value = String(DEFAULT_SET_DURATION_VALUE);
    if (expirationDurationUnitSelect) expirationDurationUnitSelect.value = DEFAULT_SET_DURATION_UNIT;
    if (slugModeSelect) slugModeSelect.value = 'custom';
    if (slugInput) delete slugInput.dataset.autoSuggested;
    syncSlugModeState();
    syncCreateTimingVisibility();
    syncAudienceFieldVisibility();
    setEditorMeta('New short link');
    setStatus(editorStatusEl, '');
    markSessionDirty();
  }

  function getSetById(setId){
    const target = String(setId || '').trim();
    if (!target) return null;
    return allSets.find((item) => String(item?.setId || '').trim() === target) || null;
  }

  function sortSetsInMemory(){
    allSets.sort((a, b) => String(a?.title || '').localeCompare(String(b?.title || '')));
  }

  function getFilteredSets(){
    const query = String(setsFilterInput?.value || '').trim().toLowerCase();
    if (!query) return allSets.slice();
    return allSets.filter((item) => String(item?.title || '').toLowerCase().includes(query));
  }

  function syncSetDefaultTimingVisibility(){
    const mode = normalizeExpirationMode(setDefaultExpirationModeSelect?.value);
    if (setDefaultDurationFields) setDefaultDurationFields.hidden = mode !== 'temporary';
  }

  function syncBatchTimingVisibility(){
    const mode = normalizeExpirationMode(batchExpirationModeSelect?.value);
    if (batchDurationFields) batchDurationFields.hidden = mode !== 'temporary';
  }

  function createSetRow(entry = {}){
    if (!setRowsEl) return null;
    setRowCounter += 1;
    const rowId = String(entry.rowId || `entry-${setRowCounter}`);
    const labelId = `shortlinks-set-row-label-${setRowCounter}`;
    const destinationId = `shortlinks-set-row-destination-${setRowCounter}`;
    const enabledId = `shortlinks-set-row-enabled-${setRowCounter}`;

    const row = document.createElement('article');
    row.className = 'shortlinks-set-row';
    row.dataset.setRowId = rowId;

    const fields = document.createElement('div');
    fields.className = 'shortlinks-inline-grid shortlinks-set-row-grid';

    const labelField = document.createElement('div');
    labelField.className = 'form-field';
    const labelLabel = document.createElement('label');
    labelLabel.setAttribute('for', labelId);
    labelLabel.textContent = 'Label';
    const labelInput = document.createElement('input');
    labelInput.id = labelId;
    labelInput.type = 'text';
    labelInput.autocomplete = 'off';
    labelInput.spellcheck = false;
    labelInput.placeholder = 'Analytics resume';
    labelInput.value = String(entry.label || '');
    labelInput.dataset.setRowField = 'label';
    labelField.appendChild(labelLabel);
    labelField.appendChild(labelInput);

    const destinationField = document.createElement('div');
    destinationField.className = 'form-field';
    const destinationLabel = document.createElement('label');
    destinationLabel.setAttribute('for', destinationId);
    destinationLabel.textContent = 'Destination';
    const destinationEntry = document.createElement('input');
    destinationEntry.id = destinationId;
    destinationEntry.type = 'text';
    destinationEntry.autocomplete = 'off';
    destinationEntry.spellcheck = false;
    destinationEntry.placeholder = 'https://example.com or /analytics';
    destinationEntry.value = String(entry.destination || '');
    destinationEntry.dataset.setRowField = 'destination';
    destinationField.appendChild(destinationLabel);
    destinationField.appendChild(destinationEntry);

    fields.appendChild(labelField);
    fields.appendChild(destinationField);
    row.appendChild(fields);

    const actions = document.createElement('div');
    actions.className = 'shortlinks-action-row shortlinks-set-row-actions';

    const enabledWrap = document.createElement('label');
    enabledWrap.className = 'shortlinks-set-row-toggle';
    const enabledInput = document.createElement('input');
    enabledInput.id = enabledId;
    enabledInput.type = 'checkbox';
    enabledInput.checked = entry.enabled !== false;
    enabledInput.dataset.setRowField = 'enabled';
    enabledWrap.appendChild(enabledInput);
    enabledWrap.appendChild(document.createTextNode(' Include in generated batches'));
    actions.appendChild(enabledWrap);

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn-ghost';
    removeButton.textContent = 'Remove row';
    removeButton.addEventListener('click', () => {
      row.remove();
      if (!setRowsEl.querySelector('[data-set-row-id]')) {
        const blank = createSetRow();
        if (blank) setRowsEl.appendChild(blank);
      }
    });
    actions.appendChild(removeButton);

    row.appendChild(actions);
    return row;
  }

  function renderSetRows(entries){
    if (!setRowsEl) return;
    setRowsEl.replaceChildren();
    const items = Array.isArray(entries) && entries.length ? entries : [{}];
    items.forEach((entry) => {
      const row = createSetRow(entry);
      if (row) setRowsEl.appendChild(row);
    });
  }

  function collectSetEntries(){
    if (!setRowsEl) return [];
    return [...setRowsEl.querySelectorAll('[data-set-row-id]')]
      .map((row, index) => ({
        rowId: String(row.dataset.setRowId || `entry-${index + 1}`),
        label: String(row.querySelector('[data-set-row-field="label"]')?.value || '').trim(),
        destination: String(row.querySelector('[data-set-row-field="destination"]')?.value || '').trim(),
        enabled: !!row.querySelector('[data-set-row-field="enabled"]')?.checked
      }))
      .filter((entry) => entry.label && entry.destination);
  }

  function resetSetEditor(options = {}){
    activeSetId = '';
    if (setTitleInput) setTitleInput.value = '';
    if (setDefaultRandomLengthInput) setDefaultRandomLengthInput.value = String(DEFAULT_RANDOM_LENGTH);
    if (setDefaultExpirationModeSelect) setDefaultExpirationModeSelect.value = 'permanent';
    if (setDefaultDurationValueInput) setDefaultDurationValueInput.value = String(DEFAULT_SET_DURATION_VALUE);
    if (setDefaultDurationUnitSelect) setDefaultDurationUnitSelect.value = DEFAULT_SET_DURATION_UNIT;
    if (batchTitleInput) batchTitleInput.value = '';
    if (batchRandomLengthInput) batchRandomLengthInput.value = String(DEFAULT_RANDOM_LENGTH);
    if (batchExpirationModeSelect) batchExpirationModeSelect.value = 'permanent';
    if (batchDurationValueInput) batchDurationValueInput.value = String(DEFAULT_SET_DURATION_VALUE);
    if (batchDurationUnitSelect) batchDurationUnitSelect.value = DEFAULT_SET_DURATION_UNIT;
    if (setDeleteButton) setDeleteButton.disabled = true;
    renderSetRows([{}]);
    syncSetDefaultTimingVisibility();
    syncBatchTimingVisibility();
    renderSetLibrary();
    clearBatchResults();
    setStatus(batchStatusEl, '');
    if (!options.keepStatus) setStatus(setEditorStatusEl, '');
  }

  function populateSetEditor(setRecord){
    const item = setRecord && typeof setRecord === 'object' ? setRecord : null;
    if (!item) {
      resetSetEditor();
      return;
    }
    activeSetId = String(item.setId || '').trim();
    if (setTitleInput) setTitleInput.value = String(item.title || '');
    if (setDefaultRandomLengthInput) setDefaultRandomLengthInput.value = String(clampRandomLength(item.defaultRandomLength, DEFAULT_RANDOM_LENGTH));
    if (setDefaultExpirationModeSelect) setDefaultExpirationModeSelect.value = normalizeExpirationMode(item.defaultExpirationMode);
    if (setDefaultDurationValueInput) setDefaultDurationValueInput.value = String(normalizeDurationValue(item.defaultDurationValue, DEFAULT_SET_DURATION_VALUE));
    if (setDefaultDurationUnitSelect) setDefaultDurationUnitSelect.value = normalizeDurationUnit(item.defaultDurationUnit);
    if (batchTitleInput) batchTitleInput.value = String(item.title || '');
    if (batchRandomLengthInput) batchRandomLengthInput.value = String(clampRandomLength(item.defaultRandomLength, DEFAULT_RANDOM_LENGTH));
    if (batchExpirationModeSelect) batchExpirationModeSelect.value = normalizeExpirationMode(item.defaultExpirationMode);
    if (batchDurationValueInput) batchDurationValueInput.value = String(normalizeDurationValue(item.defaultDurationValue, DEFAULT_SET_DURATION_VALUE));
    if (batchDurationUnitSelect) batchDurationUnitSelect.value = normalizeDurationUnit(item.defaultDurationUnit);
    if (setDeleteButton) setDeleteButton.disabled = false;
    renderSetRows(item.entries);
    syncSetDefaultTimingVisibility();
    syncBatchTimingVisibility();
    renderSetLibrary();
    clearBatchResults();
    setStatus(batchStatusEl, '');
    setStatus(setEditorStatusEl, `Editing template "${item.title}".`, 'success');
  }

  function renderSetLibrary(){
    if (!setsListEl) return;
    setsListEl.replaceChildren();

    const sets = getFilteredSets();
    const query = String(setsFilterInput?.value || '').trim();
    if (!sets.length) {
      const empty = document.createElement('p');
      empty.className = 'shortlinks-empty shortlinks-empty-state';
      empty.textContent = query ? `No templates match "${query}".` : 'No templates saved yet.';
      setsListEl.appendChild(empty);
      return;
    }

    sets.forEach((item) => {
      const card = document.createElement('article');
      card.className = 'shortlinks-item shortlinks-set-item';
      if (String(item.setId || '') === activeSetId) card.classList.add('shortlinks-set-item-active');

      const head = document.createElement('div');
      head.className = 'shortlinks-item-head';

      const titleWrap = document.createElement('div');
      titleWrap.className = 'shortlinks-item-title';
      const title = document.createElement('p');
      title.className = 'shortlinks-project-name';
      title.textContent = item.title || 'Untitled template';
      titleWrap.appendChild(title);

      const meta = document.createElement('div');
      meta.className = 'shortlinks-item-meta';
      const linksPill = document.createElement('span');
      linksPill.className = 'tool-pill';
      linksPill.textContent = `${Array.isArray(item.entries) ? item.entries.length : 0} URLs`;
      meta.appendChild(linksPill);

      const lengthPill = document.createElement('span');
      lengthPill.className = 'tool-pill shortlinks-click-pill-muted';
      lengthPill.textContent = `${clampRandomLength(item.defaultRandomLength, DEFAULT_RANDOM_LENGTH)} chars`;
      meta.appendChild(lengthPill);

      const expiryPill = document.createElement('span');
      expiryPill.className = 'tool-pill shortlinks-click-pill-muted';
      expiryPill.textContent = normalizeExpirationMode(item.defaultExpirationMode) === 'temporary'
        ? `Default ${normalizeDurationValue(item.defaultDurationValue, DEFAULT_SET_DURATION_VALUE)} ${normalizeDurationUnit(item.defaultDurationUnit)}`
        : 'Permanent';
      meta.appendChild(expiryPill);

      titleWrap.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'shortlinks-actions shortlinks-inline-actions';
      const loadButton = document.createElement('button');
      loadButton.type = 'button';
      loadButton.className = 'btn-secondary';
      loadButton.textContent = String(item.setId || '') === activeSetId ? 'Selected' : 'Use';
      loadButton.disabled = String(item.setId || '') === activeSetId;
      loadButton.addEventListener('click', () => {
        populateSetEditor(item);
      });
      actions.appendChild(loadButton);

      head.appendChild(titleWrap);
      head.appendChild(actions);
      card.appendChild(head);

      const note = document.createElement('p');
      note.className = 'shortlinks-panel-lead';
      note.textContent = `Updated ${formatTimestamp(item.updatedAt) || 'recently'}`;
      card.appendChild(note);

      setsListEl.appendChild(card);
    });
  }

  function clearBatchResults(){
    if (!batchResultsEl) return;
    batchResultsEl.replaceChildren();
  }

  function buildBatchCopyText(links){
    return (Array.isArray(links) ? links : [])
      .map((link) => {
        const label = String(link?.label || '').trim();
        const url = String(link?.shortUrl || '').trim();
        return label && url ? `${label}: ${url}` : url;
      })
      .filter(Boolean)
      .join('\n');
  }

  function renderBatchResults(payload){
    if (!batchResultsEl) return;
    batchResultsEl.replaceChildren();

    const links = Array.isArray(payload?.links) ? payload.links : [];
    if (!links.length) return;

    const head = document.createElement('div');
    head.className = 'shortlinks-card-head shortlinks-card-head-tight';

    const copy = document.createElement('div');
    copy.className = 'shortlinks-card-copy';
    const kicker = document.createElement('p');
    kicker.className = 'shortlinks-kicker';
    kicker.textContent = 'Generated set';
    const title = document.createElement('h3');
    title.className = 'shortlinks-output-title';
    title.textContent = payload?.batch?.batchTitle || 'Generated links';
    const subtitle = document.createElement('p');
    subtitle.className = 'shortlinks-output-meta';
    subtitle.textContent = payload?.batch?.permanent
      ? 'Permanent links'
      : `Expires ${payload?.batch?.expiresAt ? new Date(Number(payload.batch.expiresAt) * 1000).toLocaleString() : ''}`;
    copy.appendChild(kicker);
    copy.appendChild(title);
    copy.appendChild(subtitle);

    const actions = document.createElement('div');
    actions.className = 'shortlinks-action-row';
    const copyAllButton = document.createElement('button');
    copyAllButton.type = 'button';
    copyAllButton.className = 'btn-secondary';
    copyAllButton.textContent = 'Copy all';
    copyAllButton.addEventListener('click', async () => {
      await copyTextToClipboard({
        text: buildBatchCopyText(links),
        button: copyAllButton,
        statusTarget: batchStatusEl,
        successMessage: 'Copied all generated short links.'
      });
    });
    actions.appendChild(copyAllButton);

    head.appendChild(copy);
    head.appendChild(actions);
    batchResultsEl.appendChild(head);

    const wrap = document.createElement('div');
    wrap.className = 'shortlinks-table-wrap';
    const table = document.createElement('table');
    table.className = 'shortlinks-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Label', 'Short URL', 'Destination'].forEach((label) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    links.forEach((link) => {
      const row = document.createElement('tr');

      const labelCell = document.createElement('td');
      labelCell.textContent = String(link?.label || '');

      const shortCell = document.createElement('td');
      const shortAnchor = document.createElement('a');
      shortAnchor.className = 'shortlinks-table-destination';
      shortAnchor.href = String(link?.shortUrl || '');
      shortAnchor.target = '_blank';
      shortAnchor.rel = 'noopener noreferrer';
      shortAnchor.textContent = String(link?.shortUrl || '');
      shortCell.appendChild(shortAnchor);

      const destinationCell = document.createElement('td');
      const destinationAnchor = document.createElement('a');
      destinationAnchor.className = 'shortlinks-table-destination';
      destinationAnchor.href = String(link?.destination || '');
      destinationAnchor.target = '_blank';
      destinationAnchor.rel = 'noopener noreferrer';
      destinationAnchor.textContent = String(link?.destination || '');
      destinationCell.appendChild(destinationAnchor);

      row.appendChild(labelCell);
      row.appendChild(shortCell);
      row.appendChild(destinationCell);
      tbody.appendChild(row);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    batchResultsEl.appendChild(wrap);
  }

  async function refreshSets(options = {}){
    if (!setsListEl) return;
    const preserveSelection = options.preserveSelection !== false;
    const preferredSetId = String(options.preferredSetId || '').trim();
    if (!getSavedToken()) {
      allSets = [];
      renderSetLibrary();
      if (!options.silent) setStatus(setsStatusEl, 'Admin token required.', 'error');
      resetSetEditor({ keepStatus: true });
      return;
    }

    setStatus(setsStatusEl, 'Loading templates…');
    try {
      const data = await api('/api/short-links/sets', { method: 'GET' });
      allSets = Array.isArray(data?.sets) ? data.sets.slice() : [];
      sortSetsInMemory();

      const nextId = preferredSetId
        || (preserveSelection && getSetById(activeSetId) ? activeSetId : '')
        || (allSets[0] && allSets[0].setId ? String(allSets[0].setId) : '');

      renderSetLibrary();
      if (nextId) {
        const next = getSetById(nextId);
        if (next) populateSetEditor(next);
      } else {
        resetSetEditor({ keepStatus: true });
      }
      setStatus(setsStatusEl, `Loaded ${allSets.length} template(s).`, 'success');
    } catch (err) {
      allSets = [];
      renderSetLibrary();
      resetSetEditor({ keepStatus: true });
      setStatus(setsStatusEl, err.message || 'Unable to load templates.', 'error');
    }
  }

  function buildSetEditorPayload(){
    const title = String(setTitleInput?.value || '').trim();
    const defaultRandomLength = clampRandomLength(setDefaultRandomLengthInput?.value, DEFAULT_RANDOM_LENGTH);
    const defaultExpirationMode = normalizeExpirationMode(setDefaultExpirationModeSelect?.value);
    const payload = {
      title,
      defaultRandomLength,
      defaultExpirationMode,
      defaultDurationValue: normalizeDurationValue(setDefaultDurationValueInput?.value, DEFAULT_SET_DURATION_VALUE),
      defaultDurationUnit: normalizeDurationUnit(setDefaultDurationUnitSelect?.value),
      entries: collectSetEntries()
    };
    return payload;
  }

  async function saveSetFromEditor(){
    if (!requireToken(setEditorStatusEl)) return;
    if (!setEditorForm) return;
    const payload = buildSetEditorPayload();
    if (!payload.title) {
      setStatus(setEditorStatusEl, 'Template title is required.', 'error');
      return;
    }
    if (!Array.isArray(payload.entries) || payload.entries.length === 0) {
      setStatus(setEditorStatusEl, 'Add at least one complete URL row before saving.', 'error');
      return;
    }

    setStatus(setEditorStatusEl, activeSetId ? 'Saving template…' : 'Creating template…');
    try {
      const endpoint = activeSetId
        ? `/api/short-links/sets/${encodeURIComponent(activeSetId)}`
        : '/api/short-links/sets';
      const method = activeSetId ? 'PATCH' : 'POST';
      const data = await api(endpoint, {
        method,
        body: JSON.stringify(payload)
      });
      const savedSet = data?.set || null;
      if (savedSet?.setId) {
        await refreshSets({ preserveSelection: false, preferredSetId: savedSet.setId, silent: true });
      } else {
        await refreshSets({ preserveSelection: true, silent: true });
      }
      setStatus(setEditorStatusEl, `Saved template "${savedSet?.title || payload.title}".`, 'success');
      markSessionDirty();
    } catch (err) {
      setStatus(setEditorStatusEl, err.message || 'Unable to save template.', 'error');
    }
  }

  async function deleteActiveSet(){
    if (!requireToken(setEditorStatusEl)) return;
    if (!activeSetId) {
      setStatus(setEditorStatusEl, 'Select a template first.', 'error');
      return;
    }
    const current = getSetById(activeSetId);
    const ok = window.confirm(`Delete template "${current?.title || activeSetId}"?`);
    if (!ok) return;
    try {
      await api(`/api/short-links/sets/${encodeURIComponent(activeSetId)}`, { method: 'DELETE' });
      const deletedId = activeSetId;
      resetSetEditor({ keepStatus: true });
      await refreshSets({ preserveSelection: false, silent: true });
      setStatus(setEditorStatusEl, `Deleted template "${current?.title || deletedId}".`, 'success');
      markSessionDirty();
    } catch (err) {
      setStatus(setEditorStatusEl, err.message || 'Unable to delete template.', 'error');
    }
  }

  async function generateBatchFromEditor(){
    if (!requireToken(batchStatusEl)) return;
    if (!activeSetId) {
      setStatus(batchStatusEl, 'Select a template before generating links.', 'error');
      return;
    }

    const payload = {
      batchTitle: String(batchTitleInput?.value || '').trim(),
      randomLength: clampRandomLength(batchRandomLengthInput?.value, DEFAULT_RANDOM_LENGTH),
      expirationMode: normalizeExpirationMode(batchExpirationModeSelect?.value),
      durationValue: normalizeDurationValue(batchDurationValueInput?.value, DEFAULT_SET_DURATION_VALUE),
      durationUnit: normalizeDurationUnit(batchDurationUnitSelect?.value)
    };

    setStatus(batchStatusEl, 'Generating short links...');
    try {
      const data = await api(`/api/short-links/sets/${encodeURIComponent(activeSetId)}/generate`, {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      renderBatchResults(data);
      setStatus(batchStatusEl, `Generated ${Array.isArray(data?.links) ? data.links.length : 0} short link(s).`, 'success');
      await refreshLinks();
      markSessionDirty();
    } catch (err) {
      clearBatchResults();
      setStatus(batchStatusEl, err.message || 'Unable to generate short links.', 'error');
    }
  }

  authForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const token = tokenInput.value.trim();
    if (!token) {
      if (getSavedToken()) {
        setStatus(statusEl, 'Token already stored. Paste a token to replace, or click "Forget token".', 'success');
      } else {
        setStatus(statusEl, 'Paste your admin token to unlock this dashboard.', 'error');
      }
      return;
    }
    const remember = !!(rememberTokenInput && rememberTokenInput.checked);
    saveToken(token, remember);
    updateAccessMeta();
    tokenInput.value = '';
    setStatus(statusEl, remember ? 'Token remembered. Loading links...' : 'Token saved for this session. Loading links...');
    if (accessCard) accessCard.classList.remove('is-attention');
    setActiveMode('links');
    await refreshLinks();
    await refreshSets({ preserveSelection: true, silent: true });
  });

  refreshButton.addEventListener('click', async () => {
    if (!getSavedToken()) {
      setStatus(statusEl, 'Admin token required.', 'error');
      setStatus(listStatusEl, 'Admin token required.', 'error');
      revealAccessCard({ focusInput: true });
      return;
    }
    await refreshLinks();
    await refreshSets({ preserveSelection: true, silent: true });
  });

  if (projectsRefreshButton) {
    projectsRefreshButton.addEventListener('click', () => {
      if (!requireToken(projectsStatusEl)) return;
      refreshLinks();
    });
  }

  if (projectsEnsureButton) {
    projectsEnsureButton.addEventListener('click', () => {
      void ensureProjectLinks({ includeMismatched: true, silent: false });
    });
  }

  if (healthButton) {
    healthButton.addEventListener('click', () => {
      if (!getSavedToken()) {
        setStatus(healthStatusEl, 'Admin token required.', 'error');
        revealAccessCard({ focusInput: true });
        return;
      }
      refreshHealth();
    });
  }

  if (forgetButton) {
    forgetButton.addEventListener('click', () => {
      saveToken('');
      updateAccessMeta();
      tokenInput.value = '';
      allLinks = [];
      clearList();
      visibleLinksCount = 0;
      setStatus(statusEl, 'Token forgotten on this device.', 'success');
      setStatus(healthStatusEl, '');
      setStatus(listStatusEl, '');
      setStatus(setsStatusEl, '');
      setCount(0, 0);
      revealAccessCard({ focusInput: true });
      renderProjectLinks();
      allSets = [];
      renderSetLibrary();
      resetSetEditor({ keepStatus: true });
    });
  }

  clearButton.addEventListener('click', () => {
    clearEditor();
  });

  if (destinationPickerOpen && destinationModal) {
    destinationPickerOpen.addEventListener('click', () => {
      openDestinationPicker();
    });
  }

  if (destinationModalClose) {
    destinationModalClose.addEventListener('click', () => {
      closeDestinationPicker();
    });
  }

  if (destinationModal) {
    destinationModal.addEventListener('click', (event) => {
      if (event.target === destinationModal) closeDestinationPicker();
    });
  }

  if (destinationSearch) {
    destinationSearch.addEventListener('input', () => {
      renderDestinations();
    });
  }

  if (clicksModalClose) {
    clicksModalClose.addEventListener('click', () => {
      closeClicksModal();
    });
  }

  if (clicksRefreshButton) {
    clicksRefreshButton.addEventListener('click', () => {
      if (!activeClicksSlug) {
        setStatus(clicksStatusEl, 'No link selected.', 'error');
        return;
      }
      refreshClickHistory(activeClicksSlug);
    });
  }

  if (clicksModal) {
    clicksModal.addEventListener('click', (event) => {
      if (event.target === clicksModal) closeClicksModal();
    });
  }

  if (temporaryModalClose) {
    temporaryModalClose.addEventListener('click', () => {
      closeTemporaryModal();
    });
  }

  if (temporaryCancel) {
    temporaryCancel.addEventListener('click', () => {
      closeTemporaryModal();
    });
  }

  if (temporaryModal) {
    temporaryModal.addEventListener('click', (event) => {
      if (event.target === temporaryModal) closeTemporaryModal();
    });
  }

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    document.querySelectorAll('.shortlinks-menu[open]').forEach((menu) => {
      menu.open = false;
    });
    if (temporaryModal && temporaryModal.classList.contains('active')) {
      closeTemporaryModal();
      return;
    }
    if (destinationModal && destinationModal.classList.contains('active')) {
      closeDestinationPicker();
      return;
    }
    if (clicksModal && clicksModal.classList.contains('active')) {
      closeClicksModal();
    }
  });

  document.addEventListener('click', (event) => {
    document.querySelectorAll('.shortlinks-menu[open]').forEach((menu) => {
      if (!menu.contains(event.target)) menu.open = false;
    });
  });

  function requireToken(target){
    if (getSavedToken()) return true;
    setStatus(target || editorStatusEl, 'Admin token required.', 'error');
    revealAccessCard({ focusInput: true });
    return false;
  }

  async function handleCreateLink(){
    if (!requireToken(editorStatusEl)) return;
    const payload = getEditorPayload();
    if (!payload) return;
    const expiration = getCreateExpirationConfig();
    if (!expiration) return;
    await createOrUpdateLink(Object.assign({}, payload, expiration));
  }

  async function handleGetPermanent(){
    if (!requireToken(editorStatusEl)) return;
    const payload = getEditorPayload();
    if (!payload) return;
    await createOrUpdateLink(Object.assign({}, payload, { permanent: true, expiresAt: 0 }));
  }

  function handleGetTemporary(){
    if (!requireToken(editorStatusEl)) return;
    const payload = getEditorPayload();
    if (!payload) return;
    openTemporaryModal(payload);
  }

  function unitToSeconds(unit){
    switch (String(unit || '').toLowerCase()) {
      case 'minutes':
        return 60;
      case 'hours':
        return 60 * 60;
      case 'days':
        return 60 * 60 * 24;
      case 'weeks':
        return 60 * 60 * 24 * 7;
      default:
        return 0;
    }
  }

  function setTemporaryBusy(isBusy){
    const busy = !!isBusy;
    const controls = [
      temporaryValueInput,
      temporaryUnitSelect,
      temporaryCancel,
      temporaryModalClose
    ];
    if (temporaryForm) {
      const submitButton = temporaryForm.querySelector('button[type="submit"]');
      if (submitButton) controls.push(submitButton);
    }
    controls.forEach(control => {
      if (!control) return;
      control.disabled = busy;
    });
  }

  async function submitTemporary(event){
    event.preventDefault();
    if (!requireToken(temporaryStatusEl)) return;
    if (!pendingTemporaryPayload) {
      setStatus(temporaryStatusEl, 'Missing link details. Close this dialog and try again.', 'error');
      return;
    }

    const rawValue = temporaryValueInput ? Number(temporaryValueInput.value) : NaN;
    const value = Number.isFinite(rawValue) ? Math.floor(rawValue) : NaN;
    if (!Number.isFinite(value) || value <= 0) {
      setStatus(temporaryStatusEl, 'Enter a duration greater than 0.', 'error');
      return;
    }

    const unit = temporaryUnitSelect ? String(temporaryUnitSelect.value) : '';
    const unitSeconds = unitToSeconds(unit);
    if (!unitSeconds) {
      setStatus(temporaryStatusEl, 'Select a valid duration unit.', 'error');
      return;
    }

    const seconds = value * unitSeconds;
    const maxSeconds = 60 * 60 * 24 * 366;
    if (!Number.isFinite(seconds) || seconds > maxSeconds) {
      setStatus(temporaryStatusEl, 'Duration too long (max 1 year).', 'error');
      return;
    }

    const expiresAt = Math.floor(Date.now() / 1000) + seconds;
    setTemporaryBusy(true);
    try {
      const shortUrl = await createOrUpdateLink(Object.assign({}, pendingTemporaryPayload, {
        permanent: false,
        expiresAt,
        statusEl: temporaryStatusEl
      }));
      if (shortUrl) closeTemporaryModal();
    } finally {
      setTemporaryBusy(false);
    }
  }

  if (getPermanentButton) {
    getPermanentButton.addEventListener('click', () => {
      void handleGetPermanent();
    });
  }

  if (getTemporaryButton) {
    getTemporaryButton.addEventListener('click', () => {
      handleGetTemporary();
    });
  }

  editorForm.addEventListener('submit', (event) => {
    event.preventDefault();
    void handleCreateLink();
  });

  if (temporaryForm) {
    temporaryForm.addEventListener('submit', submitTemporary);
  }

  updateAccessMeta();
  if (audienceSelect) {
    audienceSelect.value = getSelectedAudienceKey();
  }
  syncSlugModeState();
  syncCreateTimingVisibility();
  syncAudienceFieldVisibility();
  setActiveMode(getInitialMode(), { skipPersist: true });
  setEditorMeta('New short link');
  syncExportControls();
  renderSavedViewOptions('');
  renderHealthStrip();
  updateSelectionControls();
  void refreshProjectsSection();
  resetSetEditor({ keepStatus: true });
  if (getSavedToken()) {
    setStatus(statusEl, isTokenRemembered() ? 'Remembered token loaded. Loading links...' : 'Session token loaded. Loading links...', 'success');
    refreshLinks();
    refreshSets({ preserveSelection: true, silent: true });
  }

  if (modeTabEls.length) {
    const orderedTabs = modeTabEls.slice();
    const focusModeTabByOffset = (currentTab, offset) => {
      const currentIndex = orderedTabs.indexOf(currentTab);
      if (currentIndex === -1) return;
      const nextIndex = (currentIndex + offset + orderedTabs.length) % orderedTabs.length;
      const nextTab = orderedTabs[nextIndex];
      if (!nextTab) return;
      setActiveMode(nextTab.dataset.shortlinksMode, { focusTab: true });
    };

    orderedTabs.forEach((tab, index) => {
      tab.addEventListener('click', () => {
        setActiveMode(tab.dataset.shortlinksMode);
      });
      tab.addEventListener('keydown', (event) => {
        switch (event.key) {
          case 'ArrowLeft':
          case 'ArrowUp':
            event.preventDefault();
            focusModeTabByOffset(tab, -1);
            break;
          case 'ArrowRight':
          case 'ArrowDown':
            event.preventDefault();
            focusModeTabByOffset(tab, 1);
            break;
          case 'Home':
            event.preventDefault();
            setActiveMode(orderedTabs[0]?.dataset.shortlinksMode, { focusTab: true });
            break;
          case 'End':
            event.preventDefault();
            setActiveMode(orderedTabs[orderedTabs.length - 1]?.dataset.shortlinksMode, { focusTab: true });
            break;
          case 'Enter':
          case ' ':
            event.preventDefault();
            setActiveMode(tab.dataset.shortlinksMode, { focusTab: true });
            break;
          default:
            break;
        }
      });
    });
  }

  if (filterInput) {
    filterInput.addEventListener('input', () => {
      if (savedViewSelect) savedViewSelect.value = '';
      if (deleteViewButton) deleteViewButton.disabled = true;
      applyFilterAndRender();
    });
  }

  [statusFilterSelect, sortSelect, densitySelect].forEach((control) => {
    if (!control) return;
    control.addEventListener('change', () => {
      if (savedViewSelect) savedViewSelect.value = '';
      if (deleteViewButton) deleteViewButton.disabled = true;
      applyFilterAndRender();
      markSessionDirty();
    });
  });

  if (savedViewSelect) {
    savedViewSelect.addEventListener('change', () => {
      if (savedViewSelect.value) applySavedView(savedViewSelect.value);
      if (deleteViewButton) deleteViewButton.disabled = !savedViewSelect.value;
    });
  }

  if (saveViewButton) {
    saveViewButton.addEventListener('click', () => {
      saveCurrentView();
    });
  }

  if (deleteViewButton) {
    deleteViewButton.addEventListener('click', () => {
      deleteCurrentView();
    });
  }

  if (selectVisibleInput) {
    selectVisibleInput.addEventListener('change', () => {
      const shouldSelect = selectVisibleInput.checked;
      visibleLinkSlugs.forEach((slug) => {
        if (shouldSelect) selectedSlugs.add(slug);
        else selectedSlugs.delete(slug);
      });
      applyFilterAndRender();
    });
  }

  if (clearSelectionButton) {
    clearSelectionButton.addEventListener('click', () => {
      selectedSlugs = new Set();
      applyFilterAndRender();
    });
  }

  if (testSelectedButton) {
    testSelectedButton.addEventListener('click', () => {
      void testSelectedLinks();
    });
  }

  if (exportViewButton) {
    exportViewButton.addEventListener('click', () => {
      exportCurrentView();
    });
  }

  if (newLinkFromListButton) {
    newLinkFromListButton.addEventListener('click', () => {
      startNewLinkFromList();
    });
  }

  if (audienceSelect) {
    audienceSelect.addEventListener('change', () => {
      syncEditorAudienceState({ announce: !!String(destinationInput?.value || '').trim() });
      renderDestinations();
      markSessionDirty();
    });
  }

  if (slugModeSelect) {
    slugModeSelect.addEventListener('change', () => {
      syncSlugModeState();
      markSessionDirty();
    });
  }

  if (expirationModeSelect) {
    expirationModeSelect.addEventListener('change', () => {
      syncCreateTimingVisibility();
      markSessionDirty();
    });
  }

  if (destinationInput) {
    destinationInput.addEventListener('input', () => {
      syncAudienceFieldVisibility();
      markSessionDirty();
    });
    destinationInput.addEventListener('change', () => {
      syncEditorAudienceState();
      markSessionDirty();
    });
  }

  if (tableLayoutQuery) {
    const handleTableLayoutChange = () => {
      applyFilterAndRender();
    };
    if (typeof tableLayoutQuery.addEventListener === 'function') {
      tableLayoutQuery.addEventListener('change', handleTableLayoutChange);
    } else if (typeof tableLayoutQuery.addListener === 'function') {
      tableLayoutQuery.addListener(handleTableLayoutChange);
    }
  }

  if (exportModeSelect) {
    exportModeSelect.addEventListener('change', () => {
      syncExportControls();
    });
  }

  if (exportClickLimitInput) {
    exportClickLimitInput.addEventListener('input', () => {
      updateAdminSummary();
    });
  }

  if (exportButton) {
    exportButton.addEventListener('click', () => {
      void handleExport();
    });
  }

  if (setsFilterInput) {
    setsFilterInput.addEventListener('input', () => {
      renderSetLibrary();
    });
  }

  if (setsRefreshButton) {
    setsRefreshButton.addEventListener('click', () => {
      void refreshSets({ preserveSelection: true });
    });
  }

  if (setsNewButton) {
    setsNewButton.addEventListener('click', () => {
      resetSetEditor();
      setStatus(setEditorStatusEl, 'Starting a new template.', 'success');
      markSessionDirty();
    });
  }

  if (setDefaultExpirationModeSelect) {
    setDefaultExpirationModeSelect.addEventListener('change', () => {
      syncSetDefaultTimingVisibility();
      markSessionDirty();
    });
  }

  if (batchExpirationModeSelect) {
    batchExpirationModeSelect.addEventListener('change', () => {
      syncBatchTimingVisibility();
      markSessionDirty();
    });
  }

  if (setAddRowButton) {
    setAddRowButton.addEventListener('click', () => {
      const row = createSetRow();
      if (row && setRowsEl) setRowsEl.appendChild(row);
      markSessionDirty();
    });
  }

  if (setRowsEl) {
    setRowsEl.addEventListener('input', () => {
      markSessionDirty();
    });
    setRowsEl.addEventListener('change', () => {
      markSessionDirty();
    });
  }

  if (setEditorForm) {
    setEditorForm.addEventListener('input', () => {
      markSessionDirty();
    });
    setEditorForm.addEventListener('change', () => {
      markSessionDirty();
    });
    setEditorForm.addEventListener('submit', (event) => {
      event.preventDefault();
      void saveSetFromEditor();
    });
  }

  if (setDeleteButton) {
    setDeleteButton.addEventListener('click', () => {
      void deleteActiveSet();
    });
  }

  if (setGenerateForm) {
    setGenerateForm.addEventListener('input', () => {
      markSessionDirty();
    });
    setGenerateForm.addEventListener('change', () => {
      markSessionDirty();
    });
    setGenerateForm.addEventListener('submit', (event) => {
      event.preventDefault();
      void generateBatchFromEditor();
    });
  }

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (!detail || detail.toolId !== TOOL_ID) return;

    const payload = detail.payload;
    if (!payload || typeof payload !== 'object') return;

    const audienceKey = getSelectedAudienceKey();
    const audience = getAudienceConfig(audienceKey);
    const slugMode = getSelectedSlugMode();
    const slug = normalizeSlugInput(slugInput?.value);
    const displayDestination = buildDisplayDestination(destinationInput?.value, audienceKey);
    const destination = buildStoredDestination(destinationInput?.value, audienceKey);
    const showAudienceField = shouldShowAudienceFieldForValue(destinationInput?.value);
    const expirationMode = getCreateExpirationMode();
    const expirationSummary = expirationMode === 'temporary'
      ? `Temporary (${normalizeDurationValue(expirationDurationValueInput?.value, DEFAULT_SET_DURATION_VALUE)} ${String(expirationDurationUnitSelect?.value || DEFAULT_SET_DURATION_UNIT)})`
      : 'Permanent';
    const shortUrl = slugMode === 'custom' && slug ? buildShareShortUrl(slug, destination, audienceKey) : '';
    const activeSet = getSetById(activeSetId);

    const outputSummary = (() => {
      if (slugMode === 'custom' && slug) return `Editing ${buildPublicPath(slug)}`;
      if (activeSet) return `Editing template "${activeSet.title}"`;
      if (allLinks.length) return `${allLinks.length} saved link${allLinks.length === 1 ? '' : 's'}`;
      return 'Short links';
    })();

    payload.inputs = {
      ...(showAudienceField ? { 'Audience target': audience.shortLabel || audience.label || audience.key } : {}),
      'Short code type': slugMode === 'random' ? `Random (${clampRandomLength(randomLengthInput?.value)} chars)` : 'Custom',
      Expiration: expirationSummary,
      ...(slugMode === 'custom' && slug ? { Slug: slug } : {}),
      ...(displayDestination ? { Destination: displayDestination } : {})
    };
    if (activeSet) payload.inputs['Active template'] = activeSet.title || activeSet.setId;

    payload.outputSummary = outputSummary;

    const lines = [];
    if (shortUrl) lines.push(`Short URL: ${shortUrl}`);
    if (displayDestination) lines.push(`Destination: ${displayDestination}`);
    if (destination && destination !== displayDestination) lines.push(`Stored destination: ${destination}`);
    if (activeSet) {
      lines.push('');
      lines.push(`Active template: ${activeSet.title}`);
      lines.push(`Template URLs: ${Array.isArray(activeSet.entries) ? activeSet.entries.length : 0}`);
    }

    if (allLinks.length) {
      if (lines.length) lines.push('');
      lines.push(`Saved links (${allLinks.length}):`);
      allLinks.slice(0, MAX_SAVED_LINK_LINES).forEach((link) => {
        const linkSlug = normalizeSlugInput(link?.slug);
        if (!linkSlug) return;
        const dest = String(link?.destination || '').trim();
        lines.push(`${buildPublicPath(linkSlug)} → ${dest}`);
      });
      if (allLinks.length > MAX_SAVED_LINK_LINES) {
        lines.push(`…and ${allLinks.length - MAX_SAVED_LINK_LINES} more`);
      }
    }

    if (!lines.length) return;

    payload.output = {
      kind: 'text',
      summary: outputSummary,
      text: lines.join('\n')
    };
  });
})();
