(() => {
  'use strict';

  const API_BASE = '/api/cms';
  const PATTERN_KEY = 'local-cms-section-patterns-v1';
  const AUTOSAVE_KEY = 'local-cms-autosave-v2';
  const ROUTE_SENSITIVE_COLLECTIONS = new Set(['audiences', 'pages', 'resumes', 'tools']);
  const PAGE_PATHS = [
    { path: '', label: 'Document page' },
    { path: 'page', label: 'Landing page' },
    { path: 'digitalPage', label: 'Digital resume' },
    { path: 'pdfPage', label: 'PDF preview' }
  ];
  const PREVIEW_VIEWPORTS = {
    desktop: { width: 1280, height: 900 },
    tablet: { width: 820, height: 1100 },
    mobile: { width: 390, height: 844 }
  };
  const TEXT_SELECTOR = 'h1,h2,h3,h4,h5,p,li,a,button,small,figcaption';

  const state = {
    collections: [],
    content: {},
    widgets: [],
    mediaAssets: [],
    health: null,
    snapshots: [],
    library: { templates: [], sections: [], drafts: [] },
    patterns: [],
    activeView: 'dashboard',
    collection: 'pages',
    id: 'contact',
    pagePath: '',
    workingDocument: {},
    visual: null,
    activeSectionId: '',
    inspectorMode: 'section',
    busy: false,
    assistantBusy: false,
    pendingOllamaEdit: null,
    ollamaModels: [],
    exportUrl: '',
    cleanSnapshot: '',
    siteCleanSnapshots: {},
    previewDevice: 'desktop',
    previewAudience: '',
    previewTimer: 0,
    autosaveTimer: 0,
    previewRequestId: 0,
    pendingInspectorFocus: null,
    draggedSectionId: ''
  };

  const $ = (selector) => document.querySelector(selector);

  const elements = {
    status: $('[data-cms="status"]'),
    user: $('[data-cms="user"]'),
    accessPanel: $('[data-cms="access-panel"]'),
    accessMessage: $('[data-cms="access-message"]'),
    workspace: $('[data-cms="workspace"]'),
    collection: $('[data-cms="collection"]'),
    document: $('[data-cms="document"]'),
    pageTarget: $('[data-cms="page-target"]'),
    documentId: $('[data-cms="document-id"]'),
    newDocument: $('[data-cms="new-document"]'),
    refresh: $('[data-cms="refresh"]'),
    save: $('[data-cms="save"]'),
    export: $('[data-cms="export"]'),
    saveDraft: $('[data-cms="save-draft"]'),
    saveSection: $('[data-cms="save-section"]'),
    dashboardNew: $('[data-cms="dashboard-new"]'),
    dashboardSearch: $('[data-cms="dashboard-search"]'),
    dashboardPages: $('[data-cms="dashboard-pages"]'),
    dashboardTemplates: $('[data-cms="dashboard-templates"]'),
    dashboardDrafts: $('[data-cms="dashboard-drafts"]'),
    dashboardAutosave: $('[data-cms="dashboard-autosave"]'),
    dashboardHealth: $('[data-cms="dashboard-health"]'),
    dashboardSnapshots: $('[data-cms="dashboard-snapshots"]'),
    healthRefresh: $('[data-cms="health-refresh"]'),
    libraryRefresh: $('[data-cms="library-refresh"]'),
    libraryTemplates: $('[data-cms="library-templates"]'),
    librarySections: $('[data-cms="library-sections"]'),
    libraryDrafts: $('[data-cms="library-drafts"]'),
    globalHeader: $('[data-cms="global-header"]'),
    globalFooter: $('[data-cms="global-footer"]'),
    documentList: $('[data-cms="document-list"]'),
    widgetList: $('[data-cms="widget-list"]'),
    patternList: $('[data-cms="pattern-list"]'),
    sectionList: $('[data-cms="section-list"]'),
    inspector: $('[data-cms="inspector"]'),
    editorTitle: $('[data-cms="editor-title"]'),
    documentMeta: $('[data-cms="document-meta"]'),
    pageMessage: $('[data-cms="page-message"]'),
    preview: $('[data-cms="preview"]'),
    previewStage: $('[data-cms="preview-stage"]'),
    previewViewport: $('[data-cms="preview-viewport"]'),
    previewState: $('[data-cms="preview-state"]'),
    previewRefresh: $('[data-cms="preview-refresh"]'),
    previewOpen: $('[data-cms="preview-open"]'),
    previewAudience: $('[data-cms="preview-audience"]'),
    previewDeviceButtons: Array.from(document.querySelectorAll('[data-cms-preview-device]')),
    editor: $('[data-cms="editor"]'),
    applyJson: $('[data-cms="apply-json"]'),
    advancedPanel: $('[data-cms="advanced-panel"]'),
    viewButtons: Array.from(document.querySelectorAll('[data-cms-view-target]')),
    views: Array.from(document.querySelectorAll('[data-cms-view]')),
    modeButtons: Array.from(document.querySelectorAll('[data-cms-mode]')),
    ollamaModel: $('[data-cms="ollama-model"]'),
    ollamaRefresh: $('[data-cms="ollama-refresh"]'),
    ollamaModelStatus: $('[data-cms="ollama-model-status"]'),
    ollamaPrompt: $('[data-cms="ollama-prompt"]'),
    ollamaSend: $('[data-cms="ollama-send"]'),
    ollamaLog: $('[data-cms="ollama-log"]'),
    aiActionButtons: Array.from(document.querySelectorAll('[data-cms-ai-action]'))
  };

  const clone = (value) => JSON.parse(JSON.stringify(value || {}));
  const escapeHtml = (value) => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const newId = (prefix = 'section') => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;

  const formatDate = (value) => {
    const numeric = Number(value) || 0;
    if (!numeric) return '';
    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short'
      }).format(new Date(numeric));
    } catch {
      return new Date(numeric).toISOString();
    }
  };

  const setStatus = (message, stateName = '') => {
    elements.status.textContent = message || '';
    if (stateName) elements.status.dataset.state = stateName;
    else delete elements.status.dataset.state;
  };

  const setBusy = (busy) => {
    state.busy = !!busy;
    [
      elements.collection,
      elements.document,
      elements.pageTarget,
      elements.previewAudience,
      elements.documentId,
      elements.newDocument,
      elements.refresh,
      elements.save,
      elements.saveDraft,
      elements.saveSection,
      elements.healthRefresh,
      elements.export,
      elements.editor,
      elements.applyJson,
      elements.ollamaModel,
      elements.ollamaRefresh,
      elements.ollamaPrompt,
      elements.ollamaSend
    ].forEach((element) => {
      if (element) element.disabled = state.busy;
    });
    elements.aiActionButtons.forEach((button) => {
      button.disabled = state.busy || state.assistantBusy || !!state.pendingOllamaEdit;
    });
    if (elements.ollamaSend) {
      elements.ollamaSend.disabled = state.busy || state.assistantBusy || !!state.pendingOllamaEdit;
    }
  };

  const apiFetch = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers
    });
    const text = await res.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { ok: false, error: text };
      }
    }
    if (!res.ok || data.ok === false) {
      const err = new Error(data.error || `Request failed (${res.status})`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  };

  const getCollectionDocs = (collection) => Array.isArray(state.content[collection])
    ? state.content[collection]
    : [];

  const getCurrentRecord = () => getCollectionDocs(state.collection)
    .find((record) => record.id === state.id) || null;

  const getSiteRecord = (id) => getCollectionDocs('site')
    .find((record) => record.id === id) || null;

  const getSiteDocument = (id) => {
    const record = getSiteRecord(id);
    return record && record.document && typeof record.document === 'object' && !Array.isArray(record.document)
      ? record.document
      : {};
  };

  const getSiteSnapshot = (id) => JSON.stringify(getSiteDocument(id) || {});

  const markSiteClean = () => {
    ['settings', 'navigation', 'footer'].forEach((id) => {
      state.siteCleanSnapshots[id] = getSiteSnapshot(id);
    });
  };

  const hasSiteChanges = () => {
    return ['settings', 'navigation', 'footer'].some((id) => {
      return state.siteCleanSnapshots[id] && state.siteCleanSnapshots[id] !== getSiteSnapshot(id);
    });
  };

  const deepMerge = (target, source) => {
    if (!source || typeof source !== 'object' || Array.isArray(source)) return target;
    Object.entries(source).forEach(([key, value]) => {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        if (!target[key] || typeof target[key] !== 'object' || Array.isArray(target[key])) target[key] = {};
        deepMerge(target[key], value);
        return;
      }
      target[key] = value;
    });
    return target;
  };

  const getValueAtPath = (object, path) => {
    if (!path) return object;
    return String(path).split('.').reduce((acc, key) => acc && acc[key], object);
  };

  const setValueAtPath = (object, path, value) => {
    if (!path) return value;
    const parts = String(path).split('.');
    let cursor = object;
    parts.slice(0, -1).forEach((part) => {
      if (!cursor[part] || typeof cursor[part] !== 'object') cursor[part] = {};
      cursor = cursor[part];
    });
    cursor[parts[parts.length - 1]] = value;
    return object;
  };

  const isPageLike = (value) => {
    return !!value
      && typeof value === 'object'
      && !Array.isArray(value)
      && (typeof value.template === 'string'
        || typeof value.bodyHtml === 'string'
        || Array.isArray(value.sections)
        || typeof value.outputPath === 'string');
  };

  const getPageCandidates = (documentBody = state.workingDocument) => {
    return PAGE_PATHS
      .map((item) => ({ ...item, page: getValueAtPath(documentBody, item.path) }))
      .filter((item) => isPageLike(item.page));
  };

  const getActivePage = () => {
    const page = getValueAtPath(state.workingDocument, state.pagePath);
    return isPageLike(page) ? page : null;
  };

  const setActivePage = (page) => {
    state.workingDocument = setValueAtPath(state.workingDocument, state.pagePath, page);
  };

  const normalizeSlug = (value, fallback = 'item') => {
    const slug = String(value || '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return slug || fallback;
  };

  const splitList = (value) => String(value || '')
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);

  const toNumberOrEmpty = (value) => {
    const raw = String(value ?? '').trim();
    if (!raw) return '';
    const numeric = Number(raw);
    return Number.isFinite(numeric) ? numeric : raw;
  };

  const getToolCategories = () => {
    const toolsPageRecord = getCollectionDocs('pages').find((record) => record.id === 'tools');
    const categories = toolsPageRecord && toolsPageRecord.document && Array.isArray(toolsPageRecord.document.categories)
      ? toolsPageRecord.document.categories
      : [];
    return categories.map((category) => ({
      id: String(category && category.id ? category.id : '').trim(),
      title: String(category && category.title ? category.title : '').trim()
    })).filter((category) => category.id);
  };

  const getLibraryRecords = (type) => {
    const key = type.endsWith('s') ? type : `${type}s`;
    return Array.isArray(state.library[key]) ? state.library[key] : [];
  };

  const getLibraryItems = (type) => getLibraryRecords(type)
    .map((record) => record.item || record)
    .filter(Boolean);

  const getSectionHtml = (section) => String(section && (section.html || (section.props && section.props.html)) || '').trim();

  const normalizeVisualSection = (section, index = 0, fallbackLabel = 'Section') => {
    const next = clone(section || {});
    const html = getSectionHtml(next)
      || '<section class="surface-band reveal"><div class="wrapper"><h2 class="section-title">New section</h2><p>Add content here.</p></div></section>';
    const element = sectionElementFromHtml(html);
    const label = next.label || (element ? getSectionLabel(element, `${fallbackLabel} ${index + 1}`) : `${fallbackLabel} ${index + 1}`);
    return {
      ...next,
      id: next.id || newId('section'),
      type: next.type || 'legacy-html',
      label,
      enabled: next.enabled !== false,
      html
    };
  };

  const getPageRecordItems = () => {
    const items = [];
    ['pages', 'audiences', 'resumes'].forEach((collection) => {
      getCollectionDocs(collection).forEach((record) => {
        getPageCandidates(record.document).forEach((candidate) => {
          const page = candidate.page || {};
          items.push({
            collection,
            id: record.id,
            pagePath: candidate.path,
            label: page.title || `${collection}/${record.id}`,
            description: page.description || record.relPath || '',
            relPath: record.relPath || '',
            page
          });
        });
      });
    });
    return items;
  };

  const getManagedRecordItems = () => {
    const items = getPageRecordItems().map((item) => ({
      ...item,
      kind: item.pagePath ? 'Page variant' : 'Page',
      status: 'Managed page'
    }));
    getCollectionDocs('projects').forEach((record) => {
      const documentBody = record.document || {};
      items.push({
        collection: 'projects',
        id: record.id,
        pagePath: '',
        label: documentBody.title || record.id,
        description: documentBody.subtitle || record.relPath || '',
        relPath: record.relPath || '',
        updatedAt: record.updatedAt,
        kind: 'Project',
        status: 'Generated page'
      });
    });
    getCollectionDocs('tools').forEach((record) => {
      const documentBody = record.document || {};
      items.push({
        collection: 'tools',
        id: record.id,
        pagePath: '',
        label: documentBody.title || record.id,
        description: documentBody.summary || record.relPath || '',
        relPath: record.relPath || '',
        updatedAt: record.updatedAt,
        kind: 'Tool',
        status: documentBody.hidden ? 'Hidden' : (documentBody.visibility || 'Public')
      });
    });
    return items;
  };

  const switchView = (view) => {
    const next = ['dashboard', 'builder', 'library', 'globals', 'assistant'].includes(view) ? view : 'dashboard';
    state.activeView = next;
    elements.views.forEach((panel) => {
      panel.hidden = panel.dataset.cmsView !== next;
    });
    elements.viewButtons.forEach((button) => {
      button.setAttribute('aria-current', button.dataset.cmsViewTarget === next ? 'true' : 'false');
    });
    if (next === 'dashboard') renderDashboard();
    if (next === 'library') renderLibraryView();
    if (next === 'globals') renderGlobalsView();
  };

  const serializeAttrs = (element) => {
    const attrs = {};
    Array.from(element.attributes || []).forEach((attr) => {
      attrs[attr.name] = attr.value;
    });
    if (!attrs.id) attrs.id = 'main';
    return attrs;
  };

  const attrsToString = (attrs) => {
    return Object.entries(attrs || {})
      .filter(([, value]) => value !== false && value != null && value !== '')
      .map(([key, value]) => `${key}="${String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;')}"`)
      .join(' ');
  };

  const parseHtmlDocument = (html) => {
    const parser = new DOMParser();
    return parser.parseFromString(String(html || ''), 'text/html');
  };

  const classifySection = (element) => {
    const className = String(element.getAttribute('class') || '');
    if (className.includes('hero')) return 'Hero';
    if (className.includes('cta')) return 'Call to Action';
    if (className.includes('cert')) return 'Certifications';
    if (className.includes('work')) return 'Work Experience';
    if (className.includes('project')) return 'Projects';
    if (className.includes('modal')) return 'Modal';
    if (element.tagName === 'NAV') return 'Navigation';
    return element.tagName ? element.tagName.toLowerCase().replace(/^\w/, (char) => char.toUpperCase()) : 'Section';
  };

  const getSectionLabel = (element, fallback) => {
    const explicit = element.getAttribute('aria-label') || element.getAttribute('id');
    if (explicit) return explicit;
    const heading = element.querySelector('h1,h2,h3,h4');
    if (heading && heading.textContent.trim()) return heading.textContent.trim().slice(0, 80);
    return fallback;
  };

  const loadVisualFromPage = () => {
    const page = getActivePage();
    state.visual = null;
    state.activeSectionId = '';

    if (!page) {
      elements.pageMessage.textContent = 'This document does not contain a managed page. Use Advanced JSON for this collection.';
      return;
    }

    if (page.template === 'tools-directory' && !page.bodyHtml) {
      elements.pageMessage.textContent = 'This page is generated from tool records and category settings. Use the structured fields in the inspector, then Advanced JSON only for deeper category changes.';
      return;
    }

    if (page.template === 'visual-page' && Array.isArray(page.sections)) {
      const sections = page.sections.length
        ? page.sections.map((section, index) => normalizeVisualSection(section, index))
        : [{
          id: newId('section'),
          type: 'legacy-html',
          label: 'Empty section',
          enabled: true,
          html: '<section class="surface-band reveal"><div class="wrapper"><h2 class="section-title">New section</h2><p>Add content here.</p></div></section>'
        }];
      state.visual = {
        mainAttributes: page.mainAttributes || { id: 'main' },
        sections
      };
      state.activeSectionId = sections[0] ? sections[0].id : '';
      elements.pageMessage.textContent = '';
      return;
    }

    const html = String(page.bodyHtml || '');
    const doc = parseHtmlDocument(html || '<main id="main"></main>');
    const main = doc.querySelector('main');
    const source = main || doc.body;
    const children = Array.from(source.children);
    const sections = children.length
      ? children.map((element, index) => ({
        id: element.getAttribute('data-cms-section-id') || element.id || newId('section'),
        type: element.getAttribute('data-cms-section-type') || classifySection(element),
        label: getSectionLabel(element, `Section ${index + 1}`),
        enabled: true,
        html: element.outerHTML
      }))
      : [{
        id: newId('section'),
        type: 'legacy-html',
        label: 'Empty section',
        enabled: true,
        html: '<section class="surface-band reveal"><div class="wrapper"><h2 class="section-title">New section</h2><p>Add content here.</p></div></section>'
      }];

    state.visual = {
      mainAttributes: main ? serializeAttrs(main) : { id: 'main' },
      sections
    };
    state.activeSectionId = sections[0] ? sections[0].id : '';
    elements.pageMessage.textContent = '';
  };

  const annotateSectionHtml = (section) => {
    const element = sectionElementFromHtml(section && section.html);
    if (!element) return section && section.html ? section.html : '';
    element.setAttribute('data-cms-section-id', section.id || '');
    element.setAttribute('data-cms-section-type', section.type || 'Section');
    return element.outerHTML;
  };

  const assembleBodyHtml = (options = {}) => {
    if (!state.visual) return '';
    const attrs = attrsToString(state.visual.mainAttributes || { id: 'main' });
    const sections = state.visual.sections
      .filter((section) => section.enabled !== false)
      .map((section) => options.preview ? annotateSectionHtml(section) : section.html)
      .join('\n\n');
    return `<main${attrs ? ` ${attrs}` : ''}>\n${sections}\n</main>`;
  };

  const syncPageFromVisual = () => {
    const page = getActivePage();
    if (!page || !state.visual) return;
    if (page.template === 'visual-page') {
      page.mainAttributes = state.visual.mainAttributes || { id: 'main' };
      page.sections = state.visual.sections.map((section, index) => ({
        id: section.id || newId('section'),
        type: 'legacy-html',
        label: section.label || `Section ${index + 1}`,
        enabled: section.enabled !== false,
        variant: section.variant || 'default',
        librarySource: section.librarySource || undefined,
        locks: section.locks || undefined,
        props: {
          html: section.html || ''
        }
      }));
      delete page.bodyHtml;
    } else {
      page.bodyHtml = assembleBodyHtml();
      if (!page.template) page.template = 'raw-body';
    }
    setActivePage(page);
  };

  const getEditorSnapshot = () => {
    syncPageFromVisual();
    return JSON.stringify({
      collection: state.collection,
      id: elements.documentId.value || state.id,
      pagePath: state.pagePath,
      body: state.workingDocument
    });
  };

  const hasDocumentChanges = () => !!state.cleanSnapshot && getEditorSnapshot() !== state.cleanSnapshot;

  const hasUnsavedChanges = () => hasDocumentChanges() || hasSiteChanges();

  const getAutosave = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(AUTOSAVE_KEY) || 'null');
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  };

  const renderDashboardIfActive = () => {
    if (state.activeView === 'dashboard') renderDashboard();
  };

  const writeAutosave = () => {
    if (!hasUnsavedChanges()) return;
    syncPageFromVisual();
    try {
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify({
        savedAt: new Date().toISOString(),
        collection: state.collection,
        id: elements.documentId.value || state.id,
        pagePath: state.pagePath,
        document: clone(state.workingDocument)
      }));
      renderDashboardIfActive();
    } catch {}
  };

  const scheduleAutosave = () => {
    window.clearTimeout(state.autosaveTimer);
    if (!hasUnsavedChanges()) return;
    state.autosaveTimer = window.setTimeout(writeAutosave, 900);
  };

  const clearAutosave = () => {
    window.clearTimeout(state.autosaveTimer);
    try {
      localStorage.removeItem(AUTOSAVE_KEY);
    } catch {}
    renderDashboardIfActive();
  };

  const updateDirtyState = () => {
    if (hasUnsavedChanges()) {
      elements.workspace.dataset.dirty = 'true';
      scheduleAutosave();
    } else {
      delete elements.workspace.dataset.dirty;
    }
  };

  const refreshAdvancedEditor = () => {
    syncPageFromVisual();
    elements.editor.value = `${JSON.stringify(state.workingDocument || {}, null, 2)}\n`;
  };

  const markClean = () => {
    refreshAdvancedEditor();
    state.cleanSnapshot = getEditorSnapshot();
    updateDirtyState();
  };

  const confirmDiscardChanges = () => {
    if (!hasUnsavedChanges()) return true;
    return window.confirm('Discard unsaved CMS edits?');
  };

  const getNewDocumentNote = (collection) => {
    if (!ROUTE_SENSITIVE_COLLECTIONS.has(collection)) return '';
    return 'New documents in this collection may also need route, catalog, or generated-page wiring before they appear on the site.';
  };

  const renderCollectionSelect = () => {
    elements.collection.innerHTML = '';
    state.collections.forEach((collection) => {
      const option = document.createElement('option');
      option.value = collection.name;
      option.textContent = collection.label || collection.name;
      elements.collection.appendChild(option);
    });
    elements.collection.value = state.collection;
  };

  const renderDocumentSelect = () => {
    const docs = getCollectionDocs(state.collection);
    elements.document.innerHTML = '';
    docs.forEach((record) => {
      const option = document.createElement('option');
      option.value = record.id;
      option.textContent = record.id;
      elements.document.appendChild(option);
    });
    elements.document.value = state.id;
    elements.document.disabled = state.busy || !docs.length;
  };

  const renderPageTargetSelect = () => {
    const candidates = getPageCandidates();
    elements.pageTarget.innerHTML = '';
    if (!candidates.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No managed page';
      elements.pageTarget.appendChild(option);
      elements.pageTarget.disabled = true;
      return;
    }
    candidates.forEach((candidate) => {
      const option = document.createElement('option');
      option.value = candidate.path;
      option.textContent = candidate.label;
      elements.pageTarget.appendChild(option);
    });
    if (!candidates.some((candidate) => candidate.path === state.pagePath)) {
      state.pagePath = candidates[0].path;
    }
    elements.pageTarget.value = state.pagePath;
    elements.pageTarget.disabled = state.busy || candidates.length < 2;
  };

  const renderPreviewAudienceSelect = () => {
    if (!elements.previewAudience) return;
    const audiences = getCollectionDocs('audiences')
      .map((record) => record.document || {})
      .filter((audience) => audience.key);
    elements.previewAudience.innerHTML = '<option value="">Default audience</option>';
    audiences.forEach((audience) => {
      const option = document.createElement('option');
      option.value = audience.key;
      option.textContent = audience.label || audience.brandNavPrimary || audience.key;
      elements.previewAudience.appendChild(option);
    });
    if (state.previewAudience && !audiences.some((audience) => audience.key === state.previewAudience)) {
      state.previewAudience = '';
    }
    elements.previewAudience.value = state.previewAudience;
  };

  const setInspectorMode = (mode) => {
    const normalized = ['section', 'add', 'library', 'metadata', 'header', 'footer'].includes(mode) ? mode : 'section';
    state.inspectorMode = normalized;
    elements.modeButtons.forEach((button) => {
      button.setAttribute('aria-selected', button.dataset.cmsMode === normalized ? 'true' : 'false');
    });
  };

  const renderDocumentList = () => {
    const docs = getCollectionDocs(state.collection);
    elements.documentList.innerHTML = '';
    if (!docs.length) {
      const empty = document.createElement('p');
      empty.className = 'cms-empty';
      empty.textContent = 'No documents.';
      elements.documentList.appendChild(empty);
      return;
    }
    docs.forEach((record) => {
      const button = document.createElement('button');
      button.className = 'cms-doc-button';
      button.type = 'button';
      button.dataset.id = record.id;
      button.setAttribute('aria-current', record.id === state.id ? 'true' : 'false');
      button.innerHTML = '<strong></strong><span></span>';
      button.querySelector('strong').textContent = record.id;
      button.querySelector('span').textContent = record.relPath || (record.updatedAt ? `Updated ${formatDate(record.updatedAt)}` : 'Local file');
      elements.documentList.appendChild(button);
    });
  };

  const renderWidgetList = () => {
    elements.widgetList.innerHTML = '';
    const groups = state.widgets.reduce((acc, widget) => {
      const key = widget.category || 'Widgets';
      if (!acc[key]) acc[key] = [];
      acc[key].push(widget);
      return acc;
    }, {});

    Object.entries(groups).forEach(([category, widgets]) => {
      const heading = document.createElement('p');
      heading.className = 'cms-empty';
      heading.textContent = category;
      elements.widgetList.appendChild(heading);
      widgets.forEach((widget) => {
        const button = document.createElement('button');
        button.className = 'cms-widget-button';
        button.type = 'button';
        button.dataset.widgetType = widget.type;
        button.innerHTML = '<strong></strong><span></span>';
        button.querySelector('strong').textContent = widget.label || widget.type;
        button.querySelector('span').textContent = widget.description || 'Add section';
        elements.widgetList.appendChild(button);
      });
    });
  };

  const loadPatterns = () => {
    state.patterns = getLibraryItems('sections');
  };

  const savePatterns = () => {
    loadPatterns();
  };

  const renderPatternList = () => {
    elements.patternList.innerHTML = '';
    if (!state.patterns.length) return;
    const heading = document.createElement('p');
    heading.className = 'cms-empty';
    heading.textContent = 'Saved patterns';
    elements.patternList.appendChild(heading);
    state.patterns.forEach((pattern) => {
      const button = document.createElement('button');
      button.className = 'cms-pattern-button';
      button.type = 'button';
      button.dataset.patternId = pattern.id;
      button.innerHTML = '<strong></strong><span></span>';
      button.querySelector('strong').textContent = pattern.name || pattern.title || 'Pattern';
      button.querySelector('span').textContent = [
        pattern.section && pattern.section.type ? pattern.section.type : 'Saved section',
        sectionLockSummary(pattern)
      ].filter(Boolean).join(' / ');
      elements.patternList.appendChild(button);
    });
  };

  const addEmptyMessage = (container, message) => {
    container.innerHTML = '';
    const empty = document.createElement('p');
    empty.className = 'cms-empty';
    empty.textContent = message;
    container.appendChild(empty);
  };

  const appendListCard = (container, item) => {
    const button = document.createElement('button');
    button.className = 'cms-list-card';
    button.type = 'button';
    Object.entries(item.dataset || {}).forEach(([key, value]) => {
      button.dataset[key] = value;
    });
    if (item.current) button.setAttribute('aria-current', 'true');
    button.innerHTML = '<strong></strong><span></span>';
    button.querySelector('strong').textContent = item.title || '';
    button.querySelector('span').textContent = item.meta || '';
    container.appendChild(button);
  };

  const appendPageManagerCard = (container, item) => {
    const card = document.createElement('article');
    card.className = 'cms-list-card cms-record-card';
    if (item.collection === state.collection && item.id === state.id && item.pagePath === state.pagePath) {
      card.setAttribute('aria-current', 'true');
    }
    card.innerHTML = [
      '<div class="cms-record-title"><strong></strong><span class="cms-record-badge"></span></div>',
      '<span></span>',
      '<div class="cms-record-actions">',
      '<button class="cms-button cms-button-secondary" type="button" data-dashboard-action="open-page">Edit</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-dashboard-action="preview-page">Preview</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-dashboard-action="duplicate-record">Duplicate</button>',
      '</div>'
    ].join('');
    Object.entries({
      collection: item.collection,
      id: item.id,
      pagePath: item.pagePath || ''
    }).forEach(([key, value]) => {
      card.querySelectorAll('[data-dashboard-action]').forEach((button) => {
        button.dataset[key] = value;
      });
    });
    card.querySelector('strong').textContent = item.label || item.id;
    card.querySelector('.cms-record-badge').textContent = item.kind || item.collection;
    card.querySelector('span:not(.cms-record-badge)').textContent = [
      item.status,
      item.collection,
      item.id,
      item.updatedAt ? `Updated ${formatDate(item.updatedAt)}` : item.relPath
    ].filter(Boolean).join(' / ');
    container.appendChild(card);
  };

  const renderListCards = (container, items, emptyMessage) => {
    container.innerHTML = '';
    if (!items.length) {
      addEmptyMessage(container, emptyMessage);
      return;
    }
    items.forEach((item) => appendListCard(container, item));
  };

  const appendHealthIssueCard = (container, issue) => {
    const button = document.createElement('button');
    button.className = 'cms-list-card cms-health-card';
    button.type = 'button';
    button.dataset.dashboardAction = issue.collection && issue.id ? 'open-page' : 'noop';
    button.dataset.collection = issue.collection || '';
    button.dataset.id = issue.id || '';
    button.dataset.pagePath = '';
    button.dataset.severity = issue.severity || 'info';
    button.innerHTML = '<strong></strong><span></span>';
    button.querySelector('strong').textContent = issue.title || 'CMS issue';
    button.querySelector('span').textContent = [
      (issue.severity || 'info').toUpperCase(),
      issue.collection && issue.id ? `${issue.collection}/${issue.id}` : '',
      issue.fieldPath,
      issue.detail
    ].filter(Boolean).join(' / ');
    container.appendChild(button);
  };

  const renderDashboardHealth = () => {
    if (!elements.dashboardHealth) return;
    elements.dashboardHealth.innerHTML = '';
    const health = state.health;
    if (!health) {
      addEmptyMessage(elements.dashboardHealth, 'Content health checks have not loaded yet.');
      return;
    }
    const summary = document.createElement('article');
    summary.className = 'cms-health-summary';
    summary.dataset.ready = health.deployReady ? 'true' : 'false';
    summary.innerHTML = [
      '<strong></strong>',
      '<span></span>',
      '<div class="cms-health-stats">',
      `<b>${Number(health.counts && health.counts.errors) || 0}</b><span>Errors</span>`,
      `<b>${Number(health.counts && health.counts.warnings) || 0}</b><span>Warnings</span>`,
      `<b>${Number(health.counts && health.counts.unusedMedia) || 0}</b><span>Unused media</span>`,
      '</div>'
    ].join('');
    summary.querySelector('strong').textContent = health.deployReady ? 'Ready for manual deploy checks' : 'Needs attention before deploy';
    summary.querySelector('span').textContent = health.summary || '';
    elements.dashboardHealth.appendChild(summary);

    (Array.isArray(health.checklist) ? health.checklist : []).forEach((item) => {
      const row = document.createElement('div');
      row.className = 'cms-check-row';
      row.dataset.complete = item.complete ? 'true' : 'false';
      row.textContent = `${item.complete ? 'OK' : 'Review'} - ${item.label}`;
      elements.dashboardHealth.appendChild(row);
    });

    const issues = Array.isArray(health.issues) ? health.issues.slice(0, 12) : [];
    if (!issues.length) {
      const clean = document.createElement('p');
      clean.className = 'cms-empty';
      clean.textContent = 'No CMS health issues found.';
      elements.dashboardHealth.appendChild(clean);
      return;
    }
    issues.forEach((issue) => appendHealthIssueCard(elements.dashboardHealth, issue));
  };

  const renderDashboardSnapshots = () => {
    if (!elements.dashboardSnapshots) return;
    renderListCards(elements.dashboardSnapshots, state.snapshots.slice(0, 12).map((snapshot) => ({
      title: `${snapshot.collection}/${snapshot.documentId}`,
      meta: [snapshot.reason || 'Snapshot', snapshot.createdAt ? `Saved ${formatDate(Date.parse(snapshot.createdAt))}` : '', snapshot.filePath || snapshot.relPath].filter(Boolean).join(' / '),
      dataset: {
        dashboardAction: 'load-snapshot',
        snapshotId: snapshot.snapshotId
      }
    })), 'No local save history yet. Snapshots appear after saving an existing CMS document.');
  };

  const renderDashboard = () => {
    const search = String(elements.dashboardSearch.value || '').trim().toLowerCase();
    const pages = getManagedRecordItems()
      .filter((item) => {
        const haystack = [item.label, item.description, item.collection, item.id, item.relPath].join(' ').toLowerCase();
        return !search || haystack.includes(search);
      })
      .sort((a, b) => `${a.collection}/${a.label}`.localeCompare(`${b.collection}/${b.label}`));

    elements.dashboardPages.innerHTML = '';
    if (!pages.length) addEmptyMessage(elements.dashboardPages, 'No managed records found.');
    else pages.forEach((item) => appendPageManagerCard(elements.dashboardPages, item));

    renderListCards(elements.dashboardTemplates, getLibraryItems('templates').map((template) => ({
      title: template.name || template.id,
      meta: template.description || template.folder || 'Template',
      dataset: {
        dashboardAction: 'create-template',
        templateId: template.id
      }
    })), 'No templates saved.');

    renderListCards(elements.dashboardDrafts, getLibraryItems('drafts').map((draft) => ({
      title: draft.name || draft.id,
      meta: [draft.source && draft.source.collection, draft.source && draft.source.id, draft.updatedAt ? `Saved ${draft.updatedAt}` : 'Draft'].filter(Boolean).join(' / '),
      dataset: {
        dashboardAction: 'load-draft',
        draftId: draft.id
      }
    })), 'No drafts saved.');

    const autosave = getAutosave();
    renderListCards(elements.dashboardAutosave, autosave ? [
      {
        title: 'Recover autosave',
        meta: [autosave.collection, autosave.id, autosave.savedAt ? `Saved ${autosave.savedAt}` : 'Unsaved work'].filter(Boolean).join(' / '),
        dataset: { dashboardAction: 'recover-autosave' }
      },
      {
        title: 'Clear autosave',
        meta: 'Remove local recovery data.',
        dataset: { dashboardAction: 'clear-autosave' }
      }
    ] : [], 'No local autosave recovery available.');

    renderDashboardHealth();
    renderDashboardSnapshots();
  };

  const renderLibraryView = () => {
    renderListCards(elements.libraryTemplates, getLibraryItems('templates').map((template) => ({
      title: template.name || template.id,
      meta: template.description || template.folder || 'Template',
      dataset: {
        libraryAction: 'create-template',
        templateId: template.id
      }
    })), 'No templates saved.');

    renderListCards(elements.librarySections, getLibraryItems('sections').map((item) => ({
      title: item.name || item.id,
      meta: [item.folder, item.description || (item.section && item.section.type), sectionLockSummary(item)].filter(Boolean).join(' / '),
      dataset: {
        libraryAction: 'insert-section',
        sectionId: item.id
      }
    })), 'No saved sections.');

    renderListCards(elements.libraryDrafts, getLibraryItems('drafts').map((draft) => ({
      title: draft.name || draft.id,
      meta: [draft.source && draft.source.collection, draft.source && draft.source.id, draft.updatedAt || draft.createdAt].filter(Boolean).join(' / '),
      dataset: {
        libraryAction: 'load-draft',
        draftId: draft.id
      }
    })), 'No drafts saved.');
  };

  const renderGlobalsView = () => {
    renderHeaderInspector(elements.globalHeader, { includeBack: false });
    renderFooterInspector(elements.globalFooter, { includeBack: false });
  };

  const getActiveSection = () => {
    return state.visual && state.visual.sections.find((section) => section.id === state.activeSectionId);
  };

  const getSectionLocks = (section) => ({
    editable: !(section && section.locks && section.locks.editable === false),
    lockPosition: Boolean(section && section.locks && section.locks.lockPosition),
    lockRemoval: Boolean(section && section.locks && section.locks.lockRemoval),
    lockText: Boolean(section && section.locks && section.locks.lockText),
    lockMedia: Boolean(section && section.locks && section.locks.lockMedia)
  });

  const sectionLockSummary = (section) => {
    const locks = getSectionLocks(section);
    return [
      !locks.editable ? 'read-only' : '',
      locks.lockPosition ? 'position locked' : '',
      locks.lockRemoval ? 'removal locked' : '',
      locks.lockText ? 'text locked' : '',
      locks.lockMedia ? 'media locked' : ''
    ].filter(Boolean).join(', ');
  };

  const appendStructuredActions = (target, message) => {
    const tools = document.createElement('div');
    tools.className = 'cms-inspector-tools';
    tools.innerHTML = [
      '<button class="cms-button cms-button-secondary" type="button" data-page-action="preview">Refresh Preview</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-structured-action="advanced">Advanced JSON</button>'
    ].join('');
    target.appendChild(tools);
    if (message) target.appendChild(createHint(message));
  };

  const renderProjectInspector = () => {
    elements.inspector.innerHTML = '';
    appendStructuredActions(elements.inspector, 'Project pages use a generated layout. Edit the common project fields here, then use Advanced JSON only for deeper case-study structure.');
    const project = state.workingDocument || {};
    const form = document.createElement('div');
    form.className = 'cms-field-list';
    form.appendChild(createFieldControl({
      label: 'Project title',
      value: project.title || '',
      dataset: { documentField: 'title' }
    }));
    form.appendChild(createFieldControl({
      label: 'Subtitle',
      value: project.subtitle || '',
      dataset: { documentField: 'subtitle' }
    }));
    form.appendChild(createFieldControl({
      label: 'Navigation subtitle',
      value: project.navSubtitle || '',
      dataset: { documentField: 'navSubtitle' }
    }));
    form.appendChild(createMediaField({
      label: 'Main image',
      value: project.image || '',
      placeholder: 'img/projects/example.png',
      dataset: { documentField: 'image' }
    }));
    form.appendChild(createFieldControl({
      label: 'Image alt text',
      value: project.imageAlt || '',
      dataset: { documentField: 'imageAlt' }
    }));
    form.appendChild(createMediaField({
      label: 'Thumbnail image',
      value: project.thumbImage || '',
      placeholder: 'img/projects/example.webp',
      dataset: { documentField: 'thumbImage' }
    }));
    form.appendChild(createFieldControl({
      label: 'Video WebM',
      value: project.videoWebm || '',
      placeholder: 'img/projects/example.webm',
      dataset: { documentField: 'videoWebm' }
    }));
    form.appendChild(createFieldControl({
      label: 'Video MP4',
      value: project.videoMp4 || '',
      placeholder: 'img/projects/example.mp4',
      dataset: { documentField: 'videoMp4' }
    }));
    form.appendChild(createFieldControl({
      label: 'Tools',
      value: Array.isArray(project.tools) ? project.tools.join(', ') : '',
      placeholder: 'Python, Tableau, SQL',
      dataset: { documentArrayField: 'tools' }
    }));
    form.appendChild(createFieldControl({
      label: 'Concepts',
      value: Array.isArray(project.concepts) ? project.concepts.join(', ') : '',
      placeholder: 'Visualization, Forecasting',
      dataset: { documentArrayField: 'concepts' }
    }));
    form.appendChild(createFieldControl({
      label: 'Audiences',
      value: Array.isArray(project.audiences) ? project.audiences.join(', ') : '',
      placeholder: 'analytics, tourism',
      dataset: { documentArrayField: 'audiences' }
    }));
    form.appendChild(createFieldControl({
      label: 'Problem',
      value: project.problem || '',
      multiline: true,
      dataset: { documentField: 'problem' }
    }));
    form.appendChild(createFieldControl({
      label: 'Actions',
      value: Array.isArray(project.actions) ? project.actions.join('\n') : '',
      multiline: true,
      dataset: { documentArrayField: 'actions' }
    }));
    form.appendChild(createFieldControl({
      label: 'Results',
      value: Array.isArray(project.results) ? project.results.join('\n') : '',
      multiline: true,
      dataset: { documentArrayField: 'results' }
    }));
    form.appendChild(createFieldControl({
      label: 'Role / contribution',
      value: Array.isArray(project.role) ? project.role.join('\n') : (project.role || ''),
      multiline: true,
      dataset: { documentArrayField: 'role' }
    }));
    form.appendChild(createFieldControl({
      label: 'Notes / confidentiality',
      value: project.notes || '',
      multiline: true,
      dataset: { documentField: 'notes' }
    }));
    form.appendChild(createSelectControl({
      label: 'Embed type',
      value: project.embed && project.embed.type ? project.embed.type : '',
      options: [
        { value: '', label: 'No embedded demo' },
        { value: 'iframe', label: 'Iframe demo' },
        { value: 'tableau', label: 'Tableau dashboard' }
      ],
      dataset: { documentField: 'embed.type' }
    }));
    form.appendChild(createFieldControl({
      label: 'Embed URL',
      value: project.embed && (project.embed.url || project.embed.base) ? (project.embed.url || project.embed.base) : '',
      placeholder: 'demo.html or Tableau share URL',
      dataset: { documentField: project.embed && project.embed.type === 'tableau' ? 'embed.base' : 'embed.url' }
    }));
    form.appendChild(createFieldControl({
      label: 'Demo instructions lead',
      value: project.demoInstructions && project.demoInstructions.lead ? project.demoInstructions.lead : '',
      multiline: true,
      dataset: { documentField: 'demoInstructions.lead' }
    }));
    form.appendChild(createFieldControl({
      label: 'Demo instruction bullets',
      value: project.demoInstructions && Array.isArray(project.demoInstructions.bullets) ? project.demoInstructions.bullets.join('\n') : '',
      multiline: true,
      dataset: { documentArrayField: 'demoInstructions.bullets' }
    }));
    form.appendChild(createFieldControl({
      label: 'Image width',
      value: project.imageWidth || '',
      inputType: 'number',
      dataset: { documentNumberField: 'imageWidth' }
    }));
    form.appendChild(createFieldControl({
      label: 'Image height',
      value: project.imageHeight || '',
      inputType: 'number',
      dataset: { documentNumberField: 'imageHeight' }
    }));
    form.appendChild(createCheckboxControl({
      label: 'Published',
      checked: project.published !== false,
      dataset: { projectPublishedField: 'published' }
    }));

    const resourcesDetails = document.createElement('details');
    resourcesDetails.className = 'cms-advanced cms-structured-details';
    resourcesDetails.open = true;
    resourcesDetails.innerHTML = '<summary>Resources</summary>';
    const resourcesForm = document.createElement('div');
    resourcesForm.className = 'cms-field-list';
    const resources = Array.isArray(project.resources) ? project.resources : [];
    Array.from({ length: Math.max(resources.length + 1, 3) }).forEach((_, index) => {
      const resource = resources[index] || {};
      resourcesForm.appendChild(createFieldControl({
        label: `Resource ${index + 1} label`,
        value: resource.label || '',
        dataset: { projectResourceField: `${index}.label` }
      }));
      resourcesForm.appendChild(createFieldControl({
        label: `Resource ${index + 1} URL`,
        value: resource.url || '',
        dataset: { projectResourceField: `${index}.url` }
      }));
      resourcesForm.appendChild(createMediaField({
        label: `Resource ${index + 1} icon`,
        value: resource.icon || '',
        placeholder: 'img/icons/github-icon.png',
        dataset: { projectResourceField: `${index}.icon` }
      }));
      resourcesForm.appendChild(createFieldControl({
        label: `Resource ${index + 1} type`,
        value: resource.type || '',
        placeholder: 'data, code, report, demo',
        dataset: { projectResourceField: `${index}.type` }
      }));
    });
    resourcesDetails.appendChild(resourcesForm);
    form.appendChild(resourcesDetails);

    const caseDetails = document.createElement('details');
    caseDetails.className = 'cms-advanced cms-structured-details';
    caseDetails.open = true;
    caseDetails.innerHTML = '<summary>Case Study Sections</summary>';
    const caseForm = document.createElement('div');
    caseForm.className = 'cms-field-list';
    const caseStudy = Array.isArray(project.caseStudy) ? project.caseStudy : [];
    Array.from({ length: Math.max(caseStudy.length + 1, 3) }).forEach((_, index) => {
      const section = caseStudy[index] || {};
      caseForm.appendChild(createFieldControl({
        label: `Case section ${index + 1} title`,
        value: section.title || '',
        dataset: { projectCaseField: `${index}.title` }
      }));
      caseForm.appendChild(createFieldControl({
        label: `Case section ${index + 1} lead`,
        value: section.lead || '',
        multiline: true,
        dataset: { projectCaseField: `${index}.lead` }
      }));
      caseForm.appendChild(createFieldControl({
        label: `Case section ${index + 1} bullets`,
        value: Array.isArray(section.bullets) ? section.bullets.join('\n') : '',
        multiline: true,
        dataset: { projectCaseBullets: String(index) }
      }));
    });
    caseDetails.appendChild(caseForm);
    form.appendChild(caseDetails);

    form.appendChild(createFieldControl({
      label: 'Order',
      value: project.order || '',
      inputType: 'number',
      dataset: { documentNumberField: 'order' }
    }));
    elements.inspector.appendChild(form);
  };

  const renderToolInspector = () => {
    elements.inspector.innerHTML = '';
    appendStructuredActions(elements.inspector, 'Tool records feed the generated tools directory. This editor covers the card, routing metadata, visibility, and ordering.');
    const tool = state.workingDocument || {};
    const categories = getToolCategories();
    const form = document.createElement('div');
    form.className = 'cms-field-list';
    form.appendChild(createFieldControl({
      label: 'Tool title',
      value: tool.title || '',
      dataset: { documentField: 'title' }
    }));
    form.appendChild(createFieldControl({
      label: 'Href',
      value: tool.href || '',
      placeholder: `tools/${tool.slug || state.id || 'tool-slug'}`,
      dataset: { documentField: 'href' }
    }));
    form.appendChild(createSelectControl({
      label: 'Category',
      value: tool.categoryId || '',
      options: [{ value: '', label: 'Select category' }].concat(categories.map((category) => ({
        value: category.id,
        label: category.title || category.id
      }))),
      dataset: { documentField: 'categoryId' }
    }));
    form.appendChild(createFieldControl({
      label: 'Summary',
      value: tool.summary || '',
      multiline: true,
      dataset: { documentField: 'summary' }
    }));
    form.appendChild(createFieldControl({
      label: 'Pills',
      value: Array.isArray(tool.pills) ? tool.pills.map((pill) => [pill.label, pill.variant].filter(Boolean).join(':')).join(', ') : '',
      placeholder: 'Local:local, Text, Stopwords',
      dataset: { toolPills: 'true' }
    }));
    form.appendChild(createSelectControl({
      label: 'Visibility',
      value: tool.visibility || 'public',
      options: [
        { value: 'public', label: 'Public' },
        { value: 'authed', label: 'Signed-in users' },
        { value: 'admin', label: 'Admin only' }
      ],
      dataset: { documentField: 'visibility' }
    }));
    form.appendChild(createCheckboxControl({
      label: 'Hide from directory',
      checked: !!tool.hidden,
      dataset: { documentBooleanField: 'hidden' }
    }));
    form.appendChild(createCheckboxControl({
      label: 'Noindex tool page',
      checked: !!tool.noindex,
      dataset: { documentBooleanField: 'noindex' }
    }));
    form.appendChild(createFieldControl({
      label: 'Order',
      value: tool.order || '',
      inputType: 'number',
      dataset: { documentNumberField: 'order' }
    }));
    const iconDetails = document.createElement('details');
    iconDetails.className = 'cms-advanced';
    iconDetails.innerHTML = '<summary>Advanced icon HTML</summary><textarea data-document-field="iconHtml"></textarea>';
    iconDetails.querySelector('textarea').value = tool.iconHtml || '';
    form.appendChild(iconDetails);
    elements.inspector.appendChild(form);
  };

  const renderToolsDirectoryInspector = () => {
    elements.inspector.innerHTML = '';
    const page = getActivePage();
    appendStructuredActions(elements.inspector, 'The tools directory is generated from this page configuration plus individual tool records.');
    const form = document.createElement('div');
    form.className = 'cms-field-list';
    [
      { label: 'Hero eyebrow', path: 'heroEyebrow' },
      { label: 'Hero title', path: 'heroTitle' },
      { label: 'Hero lead', path: 'heroLead', multiline: true },
      { label: 'Directory kicker', path: 'directoryKicker' },
      { label: 'Directory title', path: 'directoryTitle' },
      { label: 'Directory description', path: 'directoryDescription', multiline: true },
      { label: 'Search placeholder', path: 'filter.placeholder' },
      { label: 'Clear button label', path: 'filter.clearLabel' }
    ].forEach((field) => {
      form.appendChild(createFieldControl({
        label: field.label,
        value: getValueAtPath(page, field.path) || '',
        multiline: !!field.multiline,
        dataset: { pageField: field.path }
      }));
    });
    elements.inspector.appendChild(form);
  };

  const renderSectionList = () => {
    elements.sectionList.innerHTML = '';
    if (!state.visual) {
      const empty = document.createElement('p');
      empty.className = 'cms-empty';
      empty.textContent = 'No visual page selected.';
      elements.sectionList.appendChild(empty);
      return;
    }
    state.visual.sections.forEach((section, index) => {
      const card = document.createElement('button');
      card.className = 'cms-section-card';
      card.type = 'button';
      card.draggable = true;
      card.dataset.sectionId = section.id;
      card.setAttribute('aria-current', section.id === state.activeSectionId ? 'true' : 'false');
      card.setAttribute('aria-disabled', section.enabled === false ? 'true' : 'false');
      card.innerHTML = [
        '<span class="cms-drag-handle" aria-hidden="true">::</span>',
        '<span><strong></strong><span></span></span>',
        '<span class="cms-section-actions">',
        '<span class="cms-icon-button" role="button" tabindex="0" data-section-list-action="up" title="Move up" aria-label="Move section up">Up</span>',
        '<span class="cms-icon-button" role="button" tabindex="0" data-section-list-action="down" title="Move down" aria-label="Move section down">Dn</span>',
        '<span class="cms-icon-button" role="button" tabindex="0" data-section-list-action="duplicate" title="Duplicate" aria-label="Duplicate section">Cp</span>',
        '<span class="cms-icon-button" role="button" tabindex="0" data-section-list-action="toggle" title="Hide or show" aria-label="Hide or show section">On</span>',
        '</span>'
      ].join('');
      card.querySelector('strong').textContent = section.label || `Section ${index + 1}`;
      card.querySelector('span span').textContent = [
        section.type || 'Section',
        section.librarySource && section.librarySource.id ? `library: ${section.librarySource.id}` : '',
        sectionLockSummary(section)
      ].filter(Boolean).join(' / ');
      elements.sectionList.appendChild(card);
    });
  };

  const sectionElementFromHtml = (html) => {
    const template = document.createElement('template');
    template.innerHTML = String(html || '').trim();
    return template.content.firstElementChild;
  };

  const updateSectionHtmlFromElement = (section, element) => {
    if (!section || !element) return;
    section.html = element.outerHTML;
    const label = getSectionLabel(element, section.label || 'Section');
    section.label = label || section.label;
  };

  const collectTextFields = (element) => {
    return Array.from(element.querySelectorAll(TEXT_SELECTOR))
      .filter((node) => node.textContent.trim())
      .slice(0, 36)
      .map((node, index) => ({
        type: 'text',
        index,
        label: `${node.tagName.toLowerCase()} ${index + 1}`,
        value: node.textContent.trim()
      }));
  };

  const collectLinkFields = (element) => {
    return Array.from(element.querySelectorAll('a[href]'))
      .slice(0, 24)
      .map((node, index) => ({
        type: 'link',
        index,
        label: `Link ${index + 1}`,
        text: node.textContent.trim(),
        href: node.getAttribute('href') || ''
      }));
  };

  const collectImageFields = (element) => {
    return Array.from(element.querySelectorAll('img'))
      .slice(0, 18)
      .map((node, index) => ({
        type: 'image',
        index,
        label: `Image ${index + 1}`,
        src: node.getAttribute('src') || '',
        alt: node.getAttribute('alt') || ''
      }));
  };

  const createFieldControl = ({
    label,
    value = '',
    multiline = false,
    inputType = 'text',
    placeholder = '',
    dataset = {}
  }) => {
    const field = document.createElement('label');
    const span = document.createElement('span');
    const input = document.createElement(multiline ? 'textarea' : 'input');
    span.textContent = label;
    if (!multiline) input.type = inputType;
    input.value = value == null ? '' : String(value);
    if (placeholder) input.placeholder = placeholder;
    Object.entries(dataset).forEach(([key, entryValue]) => {
      input.dataset[key] = entryValue;
    });
    field.append(span, input);
    return field;
  };

  const createSelectControl = ({ label, value = '', options = [], dataset = {} }) => {
    const field = document.createElement('label');
    const span = document.createElement('span');
    const select = document.createElement('select');
    span.textContent = label;
    options.forEach((option) => {
      const node = document.createElement('option');
      node.value = option.value;
      node.textContent = option.label;
      select.appendChild(node);
    });
    Object.entries(dataset).forEach(([key, entryValue]) => {
      select.dataset[key] = entryValue;
    });
    select.value = value == null ? '' : String(value);
    field.append(span, select);
    return field;
  };

  const createCheckboxControl = ({ label, checked = false, dataset = {} }) => {
    const field = document.createElement('label');
    field.className = 'cms-checkbox-field';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!checked;
    Object.entries(dataset).forEach(([key, entryValue]) => {
      input.dataset[key] = entryValue;
    });
    const span = document.createElement('span');
    span.textContent = label;
    field.append(input, span);
    return field;
  };

  const createMediaField = ({ label, value = '', placeholder = '', dataset = {} }) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'cms-media-field';
    const row = document.createElement('div');
    row.className = 'cms-inline-field';
    const field = createFieldControl({
      label,
      value,
      placeholder,
      dataset
    });
    const button = document.createElement('button');
    button.className = 'cms-button cms-button-secondary';
    button.type = 'button';
    button.dataset.mediaPicker = 'true';
    button.textContent = 'Pick';
    row.append(field, button);
    wrapper.appendChild(row);
    return wrapper;
  };

  const renderMediaPicker = (field) => {
    if (!field) return;
    const existing = field.querySelector('.cms-media-picker');
    if (existing) {
      existing.remove();
      return;
    }
    const picker = document.createElement('div');
    picker.className = 'cms-media-picker';
    if (!state.mediaAssets.length) {
      const empty = document.createElement('p');
      empty.className = 'cms-empty';
      empty.textContent = 'No local image assets found under img/.';
      picker.appendChild(empty);
      field.appendChild(picker);
      return;
    }
    state.mediaAssets.slice(0, 120).forEach((asset) => {
      const button = document.createElement('button');
      button.className = 'cms-media-option';
      button.type = 'button';
      button.dataset.mediaPath = asset.path;
      const img = document.createElement('img');
      img.src = `/${asset.path}`;
      img.alt = '';
      img.loading = 'lazy';
      const label = document.createElement('span');
      label.textContent = [
        asset.path,
        asset.width && asset.height ? `${asset.width}x${asset.height}` : '',
        asset.usageCount ? `used ${asset.usageCount}x` : 'unused',
        asset.missingAlt ? 'missing alt' : ''
      ].filter(Boolean).join(' / ');
      button.dataset.mediaStatus = asset.status || '';
      button.append(img, label);
      picker.appendChild(button);
    });
    field.appendChild(picker);
  };

  const createHint = (text) => {
    const hint = document.createElement('p');
    hint.className = 'cms-field-hint';
    hint.textContent = text;
    return hint;
  };

  const parseStyleValue = (value) => {
    return String(value || '')
      .split(';')
      .map((entry) => entry.trim())
      .filter(Boolean)
      .reduce((acc, entry) => {
        const index = entry.indexOf(':');
        if (index <= 0) return acc;
        const key = entry.slice(0, index).trim().toLowerCase();
        const styleValue = entry.slice(index + 1).trim();
        if (key) acc[key] = styleValue;
        return acc;
      }, {});
  };

  const serializeStyleValue = (styles) => {
    return Object.entries(styles || {})
      .filter(([, value]) => String(value || '').trim())
      .map(([key, value]) => `${key}: ${value}`)
      .join('; ');
  };

  const setElementStyleProperty = (element, property, value) => {
    const styles = parseStyleValue(element.getAttribute('style') || '');
    const raw = String(value || '').trim();
    if (raw) styles[property] = raw;
    else delete styles[property];
    const next = serializeStyleValue(styles);
    if (next) element.setAttribute('style', next);
    else element.removeAttribute('style');
  };

  const getBackgroundImagePath = (element) => {
    const styles = parseStyleValue(element && element.getAttribute('style'));
    const match = String(styles['background-image'] || '').match(/url\((["']?)(.*?)\1\)/i);
    return match ? match[2] : '';
  };

  const getSectionSpacing = (element) => {
    const styles = parseStyleValue(element && element.getAttribute('style'));
    const top = String(styles['padding-top'] || '').trim();
    const bottom = String(styles['padding-bottom'] || '').trim();
    if (top === '2rem' && bottom === '2rem') return 'compact';
    if (top === '5rem' && bottom === '5rem') return 'spacious';
    return 'default';
  };

  const applySectionSpacing = (element, value) => {
    const spacing = String(value || 'default');
    if (spacing === 'compact') {
      setElementStyleProperty(element, 'padding-top', '2rem');
      setElementStyleProperty(element, 'padding-bottom', '2rem');
      return;
    }
    if (spacing === 'spacious') {
      setElementStyleProperty(element, 'padding-top', '5rem');
      setElementStyleProperty(element, 'padding-bottom', '5rem');
      return;
    }
    setElementStyleProperty(element, 'padding-top', '');
    setElementStyleProperty(element, 'padding-bottom', '');
  };

  const renderSectionInspector = () => {
    elements.inspector.innerHTML = '';
    const section = getActiveSection();
    if (!section) {
      const page = getActivePage();
      if (page && page.template === 'tools-directory') {
        renderToolsDirectoryInspector();
        return;
      }
      if (state.collection === 'projects') {
        renderProjectInspector();
        return;
      }
      if (state.collection === 'tools') {
        renderToolInspector();
        return;
      }
      const empty = document.createElement('p');
      empty.className = 'cms-inspector-empty';
      empty.textContent = 'Select a section to edit text, links, images, settings, or save it as a reusable pattern.';
      elements.inspector.appendChild(empty);
      return;
    }

    const tools = document.createElement('div');
    tools.className = 'cms-inspector-tools';
    tools.innerHTML = [
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="up">Move Up</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="down">Move Down</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="duplicate">Duplicate</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="toggle">Enable/Disable</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="pattern">Save Pattern</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-section-action="delete">Delete</button>'
    ].join('');
    elements.inspector.appendChild(tools);
    const lockText = sectionLockSummary(section);
    if (section.librarySource || lockText) {
      elements.inspector.appendChild(createHint([
        section.librarySource && section.librarySource.id ? `Library pattern: ${section.librarySource.id}.` : '',
        lockText ? `Locks: ${lockText}.` : ''
      ].filter(Boolean).join(' ')));
    }

    const form = document.createElement('div');
    form.className = 'cms-field-list';

    const labelField = document.createElement('label');
    labelField.innerHTML = '<span>Section label</span><input type="text" data-section-label>';
    labelField.querySelector('input').value = section.label || '';
    form.appendChild(labelField);

    const element = sectionElementFromHtml(section.html);
    if (!element) {
      const empty = document.createElement('p');
      empty.className = 'cms-inspector-empty';
      empty.textContent = 'This section markup could not be parsed. Use Advanced HTML below.';
      form.appendChild(empty);
    } else {
      const styles = parseStyleValue(element.getAttribute('style') || '');
      const settingsDetails = document.createElement('details');
      settingsDetails.className = 'cms-advanced';
      settingsDetails.open = true;
      settingsDetails.innerHTML = '<summary>Section settings</summary>';
      const settingsForm = document.createElement('div');
      settingsForm.className = 'cms-field-list';
      settingsForm.appendChild(createFieldControl({
        label: 'Element id',
        value: element.getAttribute('id') || '',
        placeholder: 'section-id',
        dataset: { sectionSetting: 'id' }
      }));
      settingsForm.appendChild(createFieldControl({
        label: 'CSS classes',
        value: element.getAttribute('class') || '',
        placeholder: 'surface-band reveal',
        dataset: { sectionSetting: 'class' }
      }));
      settingsForm.appendChild(createFieldControl({
        label: 'ARIA label',
        value: element.getAttribute('aria-label') || '',
        placeholder: 'Optional section label',
        dataset: { sectionSetting: 'aria-label' }
      }));
      settingsForm.appendChild(createSelectControl({
        label: 'Vertical spacing',
        value: getSectionSpacing(element),
        options: [
          { value: 'default', label: 'Default' },
          { value: 'compact', label: 'Compact' },
          { value: 'spacious', label: 'Spacious' }
        ],
        dataset: { sectionSetting: 'spacing' }
      }));
      settingsForm.appendChild(createFieldControl({
        label: 'Background color',
        value: styles['background-color'] || '',
        placeholder: '#f4f7fb',
        inputType: 'text',
        dataset: { sectionSetting: 'background-color' }
      }));
      settingsForm.appendChild(createMediaField({
        label: 'Background image',
        value: getBackgroundImagePath(element),
        placeholder: 'img/hero/head.jpg',
        dataset: { sectionSetting: 'background-image' }
      }));
      settingsForm.appendChild(createFieldControl({
        label: 'Minimum height',
        value: styles['min-height'] || '',
        placeholder: '60vh',
        dataset: { sectionSetting: 'min-height' }
      }));
      settingsDetails.appendChild(settingsForm);
      form.appendChild(settingsDetails);

      collectTextFields(element).forEach((field) => {
        const label = document.createElement('label');
        label.innerHTML = '<span></span><textarea data-field-type="text"></textarea>';
        label.querySelector('span').textContent = field.label;
        const input = label.querySelector('textarea');
        input.dataset.fieldIndex = String(field.index);
        input.value = field.value;
        form.appendChild(label);
      });

      collectLinkFields(element).forEach((field) => {
        const label = document.createElement('label');
        label.innerHTML = '<span></span><input type="text" data-field-type="link">';
        label.querySelector('span').textContent = `${field.label}${field.text ? `: ${field.text.slice(0, 32)}` : ''}`;
        const input = label.querySelector('input');
        input.dataset.fieldIndex = String(field.index);
        input.value = field.href;
        form.appendChild(label);
      });

      collectImageFields(element).forEach((field) => {
        form.appendChild(createMediaField({
          label: `${field.label} source`,
          value: field.src,
          dataset: {
            fieldType: 'image-src',
            fieldIndex: String(field.index)
          }
        }));

        const alt = document.createElement('label');
        alt.innerHTML = '<span></span><input type="text" data-field-type="image-alt">';
        alt.querySelector('span').textContent = `${field.label} alt text`;
        alt.querySelector('input').dataset.fieldIndex = String(field.index);
        alt.querySelector('input').value = field.alt;
        form.appendChild(alt);
      });
    }

    const htmlDetails = document.createElement('details');
    htmlDetails.className = 'cms-advanced';
    htmlDetails.innerHTML = '<summary>Advanced section HTML</summary><textarea data-section-html></textarea>';
    htmlDetails.querySelector('textarea').value = section.html || '';
    form.appendChild(htmlDetails);

    elements.inspector.appendChild(form);
  };

  const renderMetadataInspector = () => {
    elements.inspector.innerHTML = '';
    const page = getActivePage();
    if (!page) {
      const empty = document.createElement('p');
      empty.className = 'cms-inspector-empty';
      empty.textContent = 'Select a page-like document before editing page metadata.';
      elements.inspector.appendChild(empty);
      return;
    }

    const tools = document.createElement('div');
    tools.className = 'cms-inspector-tools';
    tools.innerHTML = '<button class="cms-button cms-button-secondary" type="button" data-page-action="preview">Refresh Preview</button><button class="cms-button cms-button-secondary" type="button" data-page-action="section">Back to Sections</button>';
    elements.inspector.appendChild(tools);

    const form = document.createElement('div');
    form.className = 'cms-field-list';
    form.appendChild(createHint('These fields render into the page title, canonical URL, search snippet, Open Graph tags, Twitter tags, robots meta, and theme color.'));
    [
      { label: 'Browser title', path: 'title' },
      { label: 'Canonical path', path: 'canonicalPath', placeholder: '/contact' },
      { label: 'Meta description', path: 'description', multiline: true },
      { label: 'Robots', path: 'robots', placeholder: 'index, follow' },
      { label: 'Open Graph title', path: 'ogTitle' },
      { label: 'Open Graph description', path: 'ogDescription', multiline: true },
      { label: 'Open Graph image', path: 'ogImage.src', placeholder: 'img/social-card.png' },
      { label: 'Open Graph image alt', path: 'ogImage.alt' },
      { label: 'Open Graph type', path: 'ogType', placeholder: 'website' },
      { label: 'Site name override', path: 'siteName' },
      { label: 'Twitter title', path: 'twitterTitle' },
      { label: 'Twitter description', path: 'twitterDescription', multiline: true },
      { label: 'Twitter site', path: 'twitterSite', placeholder: '@danielshort3' },
      { label: 'Theme color', path: 'themeColor', placeholder: '#0D1117' },
      { label: 'Google Analytics ID', path: 'analytics.googleAnalyticsId', placeholder: 'G-XXXXXXXXXX' },
      { label: 'Google Tag Manager ID', path: 'analytics.googleTagManagerId', placeholder: 'GTM-XXXXXXX' },
      { label: 'Facebook Pixel ID', path: 'analytics.facebookPixelId' },
      { label: 'LinkedIn Partner ID', path: 'analytics.linkedinPartnerId' }
    ].forEach((field) => {
      const options = {
        label: field.label,
        value: getValueAtPath(page, field.path) || '',
        multiline: !!field.multiline,
        placeholder: field.placeholder || '',
        dataset: { pageField: field.path }
      };
      form.appendChild(field.path === 'ogImage.src'
        ? createMediaField(options)
        : createFieldControl(options));
    });
    elements.inspector.appendChild(form);
  };

  const renderGlobalJsonEditor = (docId, title) => {
    const details = document.createElement('details');
    details.className = 'cms-advanced';
    details.innerHTML = '<summary></summary><textarea></textarea><button class="cms-button cms-button-secondary" type="button"></button>';
    details.querySelector('summary').textContent = `Advanced ${title} JSON`;
    const textarea = details.querySelector('textarea');
    textarea.dataset.globalJson = docId;
    textarea.value = `${JSON.stringify(getSiteDocument(docId), null, 2)}\n`;
    const button = details.querySelector('button');
    button.dataset.globalAction = 'apply-json';
    button.dataset.globalDoc = docId;
    button.textContent = `Apply ${title} JSON`;
    return details;
  };

  const renderHeaderInspector = (target = elements.inspector, options = {}) => {
    target.innerHTML = '';
    const navigation = getSiteDocument('navigation');
    const tools = document.createElement('div');
    tools.className = 'cms-inspector-tools';
    const backButton = options.includeBack === false
      ? ''
      : '<button class="cms-button cms-button-secondary" type="button" data-page-action="section">Back to Sections</button>';
    tools.innerHTML = `<button class="cms-button cms-button-secondary" type="button" data-global-action="save" data-global-doc="navigation">Save Header</button>${backButton}`;
    target.appendChild(tools);

    const form = document.createElement('div');
    form.className = 'cms-field-list';
    form.appendChild(createHint('Header changes come from content/site/navigation.json and apply to every generated page after saving and rebuilding.'));
    [
      { label: 'Brand home path', path: 'brand.homePath' },
      { label: 'Default tagline', path: 'brand.defaultTagline' },
      { label: 'Logo source', path: 'brand.logoSrc' },
      { label: 'Logo alt text', path: 'brand.logoAlt' },
      { label: 'Portfolio label', path: 'portfolio.label' },
      { label: 'Portfolio href', path: 'portfolio.href' },
      { label: 'Portfolio menu header', path: 'portfolio.header' },
      { label: 'Featured project IDs', path: 'portfolio.featuredProjectIds', placeholder: 'retailStore, smartSentence' },
      { label: 'Resume label', path: 'resume.label' },
      { label: 'Resume href', path: 'resume.href' },
      { label: 'Resume menu header', path: 'resume.header' },
      { label: 'Contact label', path: 'contact.label' },
      { label: 'Contact href', path: 'contact.href' },
      { label: 'Contact menu header', path: 'contact.header' },
      { label: 'Search action', path: 'search.action' },
      { label: 'Search placeholder', path: 'search.placeholder' }
    ].forEach((field) => {
      const value = getValueAtPath(navigation, field.path);
      form.appendChild(createFieldControl({
        label: field.label,
        value: Array.isArray(value) ? value.join(', ') : (value || ''),
        placeholder: field.placeholder || '',
        dataset: { globalDoc: 'navigation', globalField: field.path }
      }));
    });
    form.appendChild(renderGlobalJsonEditor('navigation', 'header'));
    target.appendChild(form);
  };

  const renderFooterInspector = (target = elements.inspector, options = {}) => {
    target.innerHTML = '';
    const footer = getSiteDocument('footer');
    const tools = document.createElement('div');
    tools.className = 'cms-inspector-tools';
    const backButton = options.includeBack === false
      ? ''
      : '<button class="cms-button cms-button-secondary" type="button" data-page-action="section">Back to Sections</button>';
    tools.innerHTML = `<button class="cms-button cms-button-secondary" type="button" data-global-action="save" data-global-doc="footer">Save Footer</button>${backButton}`;
    target.appendChild(tools);

    const form = document.createElement('div');
    form.className = 'cms-field-list';
    form.appendChild(createHint('Footer changes come from content/site/footer.json and apply to every generated page after saving and rebuilding.'));
    [
      { label: 'Copyright name', path: 'copyrightName' },
      { label: 'Cookie settings label', path: 'cookieSettingsLabel' },
      { label: 'Speed dial menu label', path: 'speedDial.menuLabel' },
      { label: 'Speed dial toggle label', path: 'speedDial.toggleLabel' }
    ].forEach((field) => {
      form.appendChild(createFieldControl({
        label: field.label,
        value: getValueAtPath(footer, field.path) || '',
        dataset: { globalDoc: 'footer', globalField: field.path }
      }));
    });

    (Array.isArray(footer.columns) ? footer.columns : []).forEach((column, columnIndex) => {
      form.appendChild(createFieldControl({
        label: `Footer column ${columnIndex + 1} title`,
        value: column.title || '',
        dataset: { globalDoc: 'footer', globalField: `columns.${columnIndex}.title` }
      }));
      (Array.isArray(column.links) ? column.links : []).slice(0, 8).forEach((link, linkIndex) => {
        form.appendChild(createFieldControl({
          label: `Column ${columnIndex + 1} link ${linkIndex + 1} label`,
          value: link.label || '',
          dataset: { globalDoc: 'footer', globalField: `columns.${columnIndex}.links.${linkIndex}.label` }
        }));
        form.appendChild(createFieldControl({
          label: `Column ${columnIndex + 1} link ${linkIndex + 1} href`,
          value: link.href || '',
          dataset: { globalDoc: 'footer', globalField: `columns.${columnIndex}.links.${linkIndex}.href` }
        }));
      });
    });
    form.appendChild(renderGlobalJsonEditor('footer', 'footer'));
    target.appendChild(form);
  };

  const renderAddInspector = () => {
    elements.inspector.innerHTML = '';
    const form = document.createElement('div');
    form.className = 'cms-field-list';
    state.widgets.forEach((widget) => {
      const button = document.createElement('button');
      button.className = 'cms-widget-button';
      button.type = 'button';
      button.dataset.inspectorWidgetType = widget.type;
      button.innerHTML = '<strong></strong><span></span>';
      button.querySelector('strong').textContent = widget.label || widget.type;
      button.querySelector('span').textContent = widget.description || 'Add section';
      form.appendChild(button);
    });
    if (!state.widgets.length) {
      const empty = document.createElement('p');
      empty.className = 'cms-inspector-empty';
      empty.textContent = 'No section widgets are available.';
      form.appendChild(empty);
    }
    elements.inspector.appendChild(form);
  };

  const renderLibraryInspector = () => {
    elements.inspector.innerHTML = '';
    const form = document.createElement('div');
    form.className = 'cms-field-list';
    getLibraryItems('sections').forEach((item) => {
      const button = document.createElement('button');
      button.className = 'cms-pattern-button';
      button.type = 'button';
      button.dataset.inspectorPatternId = item.id;
      button.innerHTML = '<strong></strong><span></span>';
      button.querySelector('strong').textContent = item.name || item.id;
      button.querySelector('span').textContent = [
        item.description || (item.section && item.section.type) || 'Saved section',
        sectionLockSummary(item)
      ].filter(Boolean).join(' / ');
      form.appendChild(button);
    });
    if (!getLibraryItems('sections').length) {
      const empty = document.createElement('p');
      empty.className = 'cms-inspector-empty';
      empty.textContent = 'No saved sections are available.';
      form.appendChild(empty);
    }
    elements.inspector.appendChild(form);
  };

  const renderInspector = () => {
    if (state.inspectorMode === 'add') {
      renderAddInspector();
      return;
    }
    if (state.inspectorMode === 'library') {
      renderLibraryInspector();
      return;
    }
    if (state.inspectorMode === 'metadata') {
      renderMetadataInspector();
      return;
    }
    if (state.inspectorMode === 'header') {
      renderHeaderInspector();
      return;
    }
    if (state.inspectorMode === 'footer') {
      renderFooterInspector();
      return;
    }
    renderSectionInspector();
  };

  const renderAll = () => {
    renderCollectionSelect();
    renderDocumentSelect();
    renderPageTargetSelect();
    renderPreviewAudienceSelect();
    renderDocumentList();
    renderWidgetList();
    renderPatternList();
    renderSectionList();
    setInspectorMode(state.inspectorMode);
    renderInspector();
    if (state.activeView === 'dashboard') renderDashboard();
    if (state.activeView === 'library') renderLibraryView();
    if (state.activeView === 'globals') renderGlobalsView();
  };

  const getPreviewDevice = (value) => Object.prototype.hasOwnProperty.call(PREVIEW_VIEWPORTS, value)
    ? value
    : 'desktop';

  const updateMainPreviewViewport = () => {
    const device = getPreviewDevice(state.previewDevice);
    const viewport = PREVIEW_VIEWPORTS[device];
    if (!elements.previewViewport || !elements.previewStage || !elements.preview) return;
    state.previewDevice = device;
    elements.previewStage.dataset.previewDevice = device;
    elements.previewDeviceButtons.forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.cmsPreviewDevice === device ? 'true' : 'false');
    });
    const availableWidth = Math.max(
      1,
      elements.previewStage.clientWidth - 24 || elements.previewStage.getBoundingClientRect().width - 24 || viewport.width
    );
    const scale = Math.min(availableWidth / viewport.width, 1);
    const offsetX = Math.max(0, (availableWidth - (viewport.width * scale)) / 2);
    elements.previewViewport.style.height = `${Math.ceil(viewport.height * scale)}px`;
    elements.previewStage.style.setProperty('--cms-preview-scale', scale.toFixed(4));
    elements.previewStage.style.setProperty('--cms-preview-offset-x', `${Math.floor(offsetX)}px`);
    elements.preview.style.width = `${viewport.width}px`;
    elements.preview.style.height = `${viewport.height}px`;
  };

  const setPreviewDevice = (deviceName) => {
    state.previewDevice = getPreviewDevice(deviceName);
    updateMainPreviewViewport();
  };

  const schedulePreview = () => {
    window.clearTimeout(state.previewTimer);
    state.previewTimer = window.setTimeout(renderPreview, 350);
  };

  const previewStatusHtml = (title, message) => {
    return [
      '<!DOCTYPE html>',
      '<html lang="en">',
      '<head>',
      '<meta charset="utf-8">',
      '<style>',
      'html,body{height:100%;margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172331;background:#fff;}',
      'body{display:grid;place-items:center;}',
      '.preview-state{display:grid;gap:.7rem;justify-items:center;text-align:center;}',
      '.spinner{width:2rem;height:2rem;border:3px solid #d5dfeb;border-top-color:#146c5c;border-radius:50%;animation:spin .8s linear infinite;}',
      '.title{font-weight:760;}',
      '.message{max-width:32rem;margin:0;color:#5b6b7f;font-size:.92rem;}',
      '@keyframes spin{to{transform:rotate(360deg)}}',
      '</style>',
      '</head>',
      '<body>',
      '<div class="preview-state">',
      '<div class="spinner" aria-hidden="true"></div>',
      `<div class="title">${escapeHtml(title)}</div>`,
      `<p class="message">${escapeHtml(message)}</p>`,
      '</div>',
      '</body>',
      '</html>'
    ].join('');
  };

  const normalizePreviewHtml = (html) => {
    const baseTag = `<base href="${escapeHtml(`${window.location.origin}/`)}">`;
    const raw = String(html || '');
    if (/<base\s+[^>]*href=/i.test(raw)) {
      return raw.replace(/<base\s+[^>]*href=(["'])[^"']*\1[^>]*>/i, baseTag);
    }
    if (/<head(\s[^>]*)?>/i.test(raw)) {
      return raw.replace(/<head(\s[^>]*)?>/i, (match) => `${match}\n  ${baseTag}`);
    }
    return raw;
  };

  const getPreviewSiteOverrides = () => {
    const settings = clone(getSiteDocument('settings'));
    if (state.previewAudience) settings.defaultAudience = state.previewAudience;
    return {
      settings,
      navigation: clone(getSiteDocument('navigation')),
      footer: clone(getSiteDocument('footer'))
    };
  };

  const getPreviewPage = (page) => {
    const previewPage = clone(page);
    if (state.previewAudience) previewPage.audienceKey = state.previewAudience;
    if (state.visual) {
      previewPage.template = 'raw-body';
      previewPage.bodyHtml = assembleBodyHtml({ preview: true });
    }
    return previewPage;
  };

  const renderPreview = async () => {
    const page = getActivePage();
    const canPreviewProject = !page && state.collection === 'projects' && state.workingDocument && state.workingDocument.id;
    const canPreviewTool = !page && state.collection === 'tools' && state.workingDocument && state.workingDocument.slug;
    if (!page && !canPreviewProject && !canPreviewTool) {
      elements.preview.srcdoc = previewStatusHtml('No managed page selected', 'Choose a page-like document to use the live preview.');
      elements.previewState.textContent = 'No page';
      updateMainPreviewViewport();
      return;
    }
    if (page) syncPageFromVisual();
    const requestId = state.previewRequestId + 1;
    state.previewRequestId = requestId;
    elements.preview.srcdoc = previewStatusHtml('Rendering preview...', 'The CMS is rebuilding this page preview with your current local edits.');
    elements.previewState.textContent = 'Rendering...';
    updateMainPreviewViewport();
    try {
      const endpoint = canPreviewProject ? '/preview-project' : (canPreviewTool ? '/preview-tool' : '/preview');
      const data = await apiFetch(endpoint, {
        method: 'POST',
        body: JSON.stringify(canPreviewProject
          ? {
            project: clone(state.workingDocument),
            site: getPreviewSiteOverrides()
          }
          : canPreviewTool
          ? {
            tool: clone(state.workingDocument),
            site: getPreviewSiteOverrides()
          }
          : {
            page: getPreviewPage(page),
            site: getPreviewSiteOverrides()
          })
      });
      if (requestId !== state.previewRequestId) return;
      elements.preview.srcdoc = normalizePreviewHtml(data.html || '');
      elements.previewState.textContent = 'Live';
      updateMainPreviewViewport();
    } catch (err) {
      if (requestId !== state.previewRequestId) return;
      elements.previewState.textContent = 'Error';
      elements.preview.srcdoc = previewStatusHtml('Preview failed', err.message || 'Unable to render preview.');
      updateMainPreviewViewport();
      setStatus(err.message || 'Unable to render preview.', 'error');
    }
  };

  const openPreviewWindow = () => {
    const html = elements.preview ? elements.preview.srcdoc : '';
    if (!html) return;
    const previewWindow = window.open('', '_blank');
    if (!previewWindow) {
      setStatus('The browser blocked the preview window.', 'error');
      return;
    }
    previewWindow.document.open();
    previewWindow.document.write(html);
    previewWindow.document.close();
  };

  const focusInspectorField = (field) => {
    if (!field) return;
    window.requestAnimationFrame(() => {
      const selector = `[data-field-type="${field.type}"][data-field-index="${field.index}"]`;
      const input = elements.inspector.querySelector(selector);
      if (!input) return;
      input.scrollIntoView({ block: 'center', behavior: 'smooth' });
      input.focus({ preventScroll: true });
    });
  };

  const selectSectionFromPreview = (sectionId, field) => {
    if (!sectionId) return;
    state.activeSectionId = sectionId;
    setInspectorMode('section');
    renderSectionList();
    renderInspector();
    focusInspectorField(field);
  };

  const getPreviewFieldForTarget = (target, sectionRoot) => {
    if (!target || !sectionRoot) return null;
    const image = target.closest('img');
    if (image && sectionRoot.contains(image)) {
      const index = Array.from(sectionRoot.querySelectorAll('img')).indexOf(image);
      return index >= 0 ? { type: 'image-src', index } : null;
    }
    const link = target.closest('a[href]');
    if (link && sectionRoot.contains(link)) {
      const index = Array.from(sectionRoot.querySelectorAll('a[href]')).indexOf(link);
      return index >= 0 ? { type: 'link', index } : null;
    }
    const textNode = target.closest(TEXT_SELECTOR);
    if (textNode && sectionRoot.contains(textNode) && textNode.textContent.trim()) {
      const nodes = Array.from(sectionRoot.querySelectorAll(TEXT_SELECTOR))
        .filter((node) => node.textContent.trim())
        .slice(0, 36);
      const index = nodes.indexOf(textNode);
      return index >= 0 ? { type: 'text', index } : null;
    }
    return null;
  };

  const bindPreviewInspector = () => {
    let doc;
    try {
      doc = elements.preview.contentDocument;
    } catch {
      return;
    }
    if (!doc) return;
    const style = doc.createElement('style');
    style.textContent = [
      '[data-cms-section-id]{outline:2px dashed rgba(20,108,92,.45);outline-offset:3px;cursor:pointer;}',
      '[data-cms-section-id]:hover{outline-color:rgba(20,108,92,.85);}',
      '[data-cms-section-id][data-cms-section-active="true"]{outline:3px solid rgba(47,125,225,.85);}',
      '[contenteditable][data-cms-inline-editing="true"]{outline:3px solid rgba(47,125,225,.9)!important;outline-offset:3px;background:rgba(47,125,225,.08);cursor:text;}',
      '.cms-inline-toolbar{position:fixed;z-index:2147483647;display:flex;gap:4px;padding:5px;background:#172331;border:1px solid rgba(255,255,255,.16);border-radius:6px;box-shadow:0 12px 32px rgba(0,0,0,.22);}',
      '.cms-inline-toolbar button{min-width:2rem;min-height:1.9rem;padding:0 .45rem;color:#fff;background:transparent;border:1px solid rgba(255,255,255,.24);border-radius:4px;font:700 13px/1 system-ui,sans-serif;}',
      '.cms-inline-toolbar button:hover{background:rgba(255,255,255,.14);}'
    ].join('');
    doc.head.appendChild(style);
    doc.querySelectorAll('[data-cms-section-id]').forEach((node) => {
      if (node.getAttribute('data-cms-section-id') === state.activeSectionId) {
        node.setAttribute('data-cms-section-active', 'true');
      }
    });
    const saveInlineText = (editable, sectionRoot, index, toolbar) => {
      const id = sectionRoot.getAttribute('data-cms-section-id');
      const section = state.visual && state.visual.sections.find((item) => item.id === id);
      const element = sectionElementFromHtml(section && section.html);
      const target = element
        ? Array.from(element.querySelectorAll(TEXT_SELECTOR)).filter((node) => node.textContent.trim()).slice(0, 36)[index]
        : null;
      if (section && target) {
        target.innerHTML = editable.innerHTML;
        updateSectionHtmlFromElement(section, element);
        state.activeSectionId = section.id;
        renderSectionList();
        renderInspector();
        schedulePreview();
        updateDirtyState();
      }
      editable.removeAttribute('contenteditable');
      editable.removeAttribute('data-cms-inline-editing');
      if (toolbar) toolbar.remove();
    };

    const startInlineTextEdit = (textNode, sectionRoot, index) => {
      const rect = textNode.getBoundingClientRect();
      const toolbar = doc.createElement('div');
      toolbar.className = 'cms-inline-toolbar';
      toolbar.innerHTML = [
        '<button type="button" data-inline-command="bold" aria-label="Bold">B</button>',
        '<button type="button" data-inline-command="italic" aria-label="Italic">I</button>',
        '<button type="button" data-inline-command="createLink" aria-label="Link">Link</button>',
        '<button type="button" data-inline-command="removeFormat" aria-label="Clear formatting">Clear</button>'
      ].join('');
      toolbar.style.left = `${Math.max(8, rect.left)}px`;
      toolbar.style.top = `${Math.max(8, rect.top - 44)}px`;
      doc.body.appendChild(toolbar);

      textNode.setAttribute('contenteditable', 'true');
      textNode.setAttribute('data-cms-inline-editing', 'true');
      textNode.focus();

      toolbar.addEventListener('mousedown', (event) => event.preventDefault());
      toolbar.addEventListener('click', (event) => {
        const button = event.target.closest('[data-inline-command]');
        if (!button) return;
        const command = button.dataset.inlineCommand;
        if (command === 'createLink') {
          const href = window.prompt('Link URL', '');
          if (href) doc.execCommand(command, false, href);
          return;
        }
        doc.execCommand(command, false, null);
      });

      let committed = false;
      const commit = () => {
        if (committed) return;
        committed = true;
        saveInlineText(textNode, sectionRoot, index, toolbar);
      };
      textNode.addEventListener('blur', () => window.setTimeout(commit, 100), { once: true });
      textNode.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          textNode.blur();
        }
        if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
          event.preventDefault();
          commit();
        }
      });
    };

    doc.addEventListener('dblclick', (event) => {
      const target = event.target;
      const sectionRoot = target && target.closest ? target.closest('[data-cms-section-id]') : null;
      const textNode = target && target.closest ? target.closest(TEXT_SELECTOR) : null;
      if (!sectionRoot || !textNode || !sectionRoot.contains(textNode) || !textNode.textContent.trim()) return;
      event.preventDefault();
      event.stopPropagation();
      const nodes = Array.from(sectionRoot.querySelectorAll(TEXT_SELECTOR))
        .filter((node) => node.textContent.trim())
        .slice(0, 36);
      const index = nodes.indexOf(textNode);
      if (index < 0) return;
      selectSectionFromPreview(sectionRoot.getAttribute('data-cms-section-id'), { type: 'text', index });
      startInlineTextEdit(textNode, sectionRoot, index);
    }, true);

    doc.addEventListener('click', (event) => {
      const target = event.target;
      if (target && target.closest && target.closest('[contenteditable="true"],.cms-inline-toolbar')) return;
      const sectionRoot = target && target.closest ? target.closest('[data-cms-section-id]') : null;
      if (sectionRoot) {
        event.preventDefault();
        event.stopPropagation();
        const id = sectionRoot.getAttribute('data-cms-section-id');
        selectSectionFromPreview(id, getPreviewFieldForTarget(target, sectionRoot));
        return;
      }
      if (target && target.closest && target.closest('#combined-header-nav')) {
        event.preventDefault();
        event.stopPropagation();
        setInspectorMode('header');
        renderInspector();
        return;
      }
      if (target && target.closest && target.closest('.footer-classic,.speed-dial,.cookie-settings')) {
        event.preventDefault();
        event.stopPropagation();
        setInspectorMode('footer');
        renderInspector();
      }
    }, true);
  };

  const loadCurrentRecord = () => {
    const record = getCurrentRecord();
    elements.documentId.value = state.id || '';
    elements.editorTitle.textContent = state.id ? `${state.collection}/${state.id}` : 'Page Builder';
    elements.documentMeta.textContent = record
      ? [record.relPath, record.updatedAt ? `Updated ${formatDate(record.updatedAt)}` : ''].filter(Boolean).join(' · ')
      : 'New document';
    state.workingDocument = clone(record ? record.document : {});

    const candidates = getPageCandidates();
    state.pagePath = candidates.some((candidate) => candidate.path === state.pagePath)
      ? state.pagePath
      : (candidates[0] ? candidates[0].path : '');
    loadVisualFromPage();
    renderAll();
    markClean();
    schedulePreview();
  };

  const refreshSelectedDocument = (savedRecord) => {
    const docs = getCollectionDocs(savedRecord.collection);
    const index = docs.findIndex((record) => record.id === savedRecord.id);
    if (index >= 0) docs[index] = savedRecord;
    else docs.push(savedRecord);
    docs.sort((a, b) => a.id.localeCompare(b.id));
    state.collection = savedRecord.collection;
    state.id = savedRecord.id;
    state.workingDocument = clone(savedRecord.document);
  };

  const saveSiteDocument = async (docId) => {
    const record = getSiteRecord(docId);
    if (!record) throw new Error(`Missing site/${docId} document`);
    const data = await apiFetch('/content', {
      method: 'PUT',
      body: JSON.stringify({
        collection: 'site',
        id: docId,
        document: record.document,
        expectedRevisionId: String(record.revisionId || '')
      })
    });
    const docs = getCollectionDocs(data.document.collection);
    const index = docs.findIndex((item) => item.id === data.document.id);
    if (index >= 0) docs[index] = data.document;
    else docs.push(data.document);
    docs.sort((a, b) => a.id.localeCompare(b.id));
    state.siteCleanSnapshots[docId] = getSiteSnapshot(docId);
    if (state.collection === 'site' && state.id === docId) {
      state.workingDocument = clone(data.document.document);
      elements.editor.value = `${JSON.stringify(state.workingDocument || {}, null, 2)}\n`;
    }
    return data.document;
  };

  const getChangedSiteDocumentIds = () => {
    return ['settings', 'navigation', 'footer'].filter((id) => {
      return state.siteCleanSnapshots[id] && state.siteCleanSnapshots[id] !== getSiteSnapshot(id);
    });
  };

  const saveChangedSiteDocuments = async () => {
    const changed = getChangedSiteDocumentIds();
    const saved = [];
    for (const docId of changed) {
      await saveSiteDocument(docId);
      saved.push(docId);
    }
    return saved;
  };

  const saveDocument = async () => {
    const collection = String(elements.collection.value || state.collection).trim();
    const id = String(elements.documentId.value || '').trim();
    if (!id) {
      setStatus('Document ID is required.', 'error');
      return;
    }

    syncPageFromVisual();
    if (collection === 'projects' && !elements.advancedPanel.open) cleanupProjectStructuredArrays();
    let documentBody = clone(state.workingDocument);
    try {
      if (elements.advancedPanel.open) {
        documentBody = JSON.parse(elements.editor.value || '{}');
      }
    } catch (err) {
      setStatus(`Invalid JSON: ${err.message}`, 'error');
      return;
    }

    if (!documentBody || typeof documentBody !== 'object' || Array.isArray(documentBody)) {
      setStatus('Document JSON must be an object.', 'error');
      return;
    }

    if (collection === 'pages' && documentBody.id) documentBody.id = id;
    if (collection === 'projects' && documentBody.id) documentBody.id = id;
    if ((collection === 'audiences' || collection === 'resumes') && documentBody.key) documentBody.key = id;
    if (collection === 'tools' && documentBody.slug) documentBody.slug = id;

    const targetRecord = getCollectionDocs(collection).find((record) => record.id === id) || null;
    const expectedRevisionId = targetRecord ? String(targetRecord.revisionId || '') : '';

    setBusy(true);
    setStatus('Saving local content file...');
    try {
      const data = await apiFetch('/content', {
        method: 'PUT',
        body: JSON.stringify({ collection, id, document: documentBody, expectedRevisionId })
      });
      refreshSelectedDocument(data.document);
      const savedSiteDocs = await saveChangedSiteDocuments();
      await Promise.allSettled([loadHealthData(), loadSnapshotData()]);
      loadCurrentRecord();
      clearAutosave();
      const note = getNewDocumentNote(collection);
      const siteNote = savedSiteDocs.length ? ` Saved shared ${savedSiteDocs.map((docId) => `site/${docId}`).join(', ')}.` : '';
      setStatus(`Saved to content/.${siteNote} Run npm run build && npm test, then review git diff before deploying.${note ? ` ${note}` : ''}`, 'success');
    } catch (err) {
      const message = err.status === 409
        ? `${err.message} Use Refresh to load the latest file before saving.`
        : (err.message || 'Unable to save local content file.');
      setStatus(message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const createNewDocument = () => {
    const collection = String(elements.collection.value || state.collection).trim();
    if (collection === 'pages') {
      startNewPage('basic-page');
      return;
    }
    if (!confirmDiscardChanges()) return;
    state.collection = collection;
    state.id = '';
    const page = createPageTemplate('basic-page');
    state.workingDocument = collection === 'pages'
      ? page
      : {};
    state.pagePath = collection === 'pages' ? '' : '';
    elements.documentId.value = '';
    loadVisualFromPage();
    renderAll();
    markClean();
    refreshAdvancedEditor();
    const note = getNewDocumentNote(collection);
    setStatus(note || 'New local document ready.', note ? '' : 'success');
  };

  const exportContent = async () => {
    setBusy(true);
    setStatus('Preparing export...');
    try {
      const data = await apiFetch('/content');
      if (state.exportUrl) URL.revokeObjectURL(state.exportUrl);
      const blob = new Blob([`${JSON.stringify({
        generatedAt: new Date().toISOString(),
        source: 'local-cms',
        content: data.content || {}
      }, null, 2)}\n`], { type: 'application/json' });
      state.exportUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = state.exportUrl;
      link.download = `cms-content-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus('Export prepared.', 'success');
    } catch (err) {
      setStatus(err.message || 'Unable to export content.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const createSectionFromWidget = (type) => {
    const widget = state.widgets.find((item) => item.type === type) || state.widgets[0];
    if (!widget || !widget.defaultSection) return null;
    const section = clone(widget.defaultSection);
    section.id = newId(type || 'section');
    section.label = widget.label || section.label || type;
    return normalizeVisualSection(section);
  };

  const prepareTemplatePage = (page, templateId) => {
    const next = clone(page || {});
    const slug = normalizeSlug(elements.documentId.value || next.id || templateId || 'new-page', 'new-page');
    next.id = slug;
    next.template = 'visual-page';
    next.outputPath = `pages/${slug}.html`;
    next.canonicalPath = `/${slug}`;
    next.title = next.title || `${slug.replace(/[-_]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())} | Daniel Short`;
    next.description = next.description || 'New managed page.';
    next.bodyAttributes = {
      ...(next.bodyAttributes || {}),
      'data-page': slug,
      class: next.bodyAttributes && next.bodyAttributes.class ? next.bodyAttributes.class : 'site-page'
    };
    next.mainAttributes = next.mainAttributes || { id: 'main' };
    next.sections = Array.isArray(next.sections)
      ? next.sections.map((section, index) => normalizeVisualSection(section, index))
      : [];
    delete next.bodyHtml;
    return next;
  };

  const createPageTemplate = (templateName) => {
    const template = getLibraryItems('templates').find((item) => item.id === templateName);
    if (template && template.page) return prepareTemplatePage(template.page, template.id);

    const hero = createSectionFromWidget('hero');
    const rich = createSectionFromWidget('rich-text');
    const cta = createSectionFromWidget('cta');
    const sections = templateName === 'landing-page'
      ? [hero, createSectionFromWidget('image-gallery'), rich, cta].filter(Boolean)
      : [hero, rich, cta].filter(Boolean);
    const slug = normalizeSlug(elements.documentId.value || 'new-page', 'new-page');
    return prepareTemplatePage({
      id: elements.documentId.value.trim() || 'new-page',
      template: 'visual-page',
      outputPath: `pages/${slug}.html`,
      title: 'New Page | Daniel Short',
      canonicalPath: `/${slug}`,
      description: 'New managed page.',
      bodyAttributes: {
        'data-page': slug,
        class: 'site-page'
      },
      stylesheets: ['dist/styles.css'],
      headScripts: [{ src: 'dist/site-shell.js', defer: true }, { src: 'js/common/no-js.js' }],
      bottomScripts: [{ src: 'dist/site-consent.js', defer: true }],
      mainAttributes: { id: 'main' },
      sections
    }, templateName);
  };

  const addSection = (section) => {
    if (!state.visual) {
      setStatus('Select a page-like document before adding sections.', 'error');
      return;
    }
    const next = normalizeVisualSection(section, state.visual.sections.length);
    next.id = newId(next.type || 'section');
    state.visual.sections.push(next);
    state.activeSectionId = next.id;
    renderSectionList();
    renderInspector();
    schedulePreview();
    updateDirtyState();
  };

  const moveSection = (sectionId, direction) => {
    const sections = state.visual ? state.visual.sections : [];
    const index = sections.findIndex((section) => section.id === sectionId);
    if (index >= 0 && getSectionLocks(sections[index]).lockPosition) {
      setStatus('This library section has position locking enabled.', 'error');
      return;
    }
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= sections.length) return;
    const [section] = sections.splice(index, 1);
    sections.splice(nextIndex, 0, section);
    renderSectionList();
    schedulePreview();
    updateDirtyState();
  };

  const duplicateSection = (sectionId) => {
    const sections = state.visual ? state.visual.sections : [];
    const index = sections.findIndex((section) => section.id === sectionId);
    if (index < 0) return;
    const copy = clone(sections[index]);
    copy.id = newId(copy.type || 'section');
    copy.label = `${copy.label || 'Section'} copy`;
    sections.splice(index + 1, 0, copy);
    state.activeSectionId = copy.id;
    renderSectionList();
    renderInspector();
    schedulePreview();
    updateDirtyState();
  };

  const deleteSection = (sectionId) => {
    const sections = state.visual ? state.visual.sections : [];
    const index = sections.findIndex((section) => section.id === sectionId);
    if (index < 0) return;
    if (getSectionLocks(sections[index]).lockRemoval) {
      setStatus('This library section has removal locking enabled.', 'error');
      return;
    }
    sections.splice(index, 1);
    state.activeSectionId = sections[Math.max(0, index - 1)] ? sections[Math.max(0, index - 1)].id : '';
    renderSectionList();
    renderInspector();
    schedulePreview();
    updateDirtyState();
  };

  const saveSectionPattern = async (sectionId) => {
    const section = getActiveSection();
    if (!section || section.id !== sectionId) return;
    const title = window.prompt('Pattern name', section.label || 'Saved section');
    if (!title) return;
    const id = normalizeSlug(title, newId('section'));
    setBusy(true);
    setStatus('Saving reusable section...');
    try {
      await apiFetch('/library', {
        method: 'PUT',
        body: JSON.stringify({
          type: 'section',
          item: {
            id,
            name: title,
            description: section.type || 'Saved section',
            folder: 'Reusable Sections',
            tags: [section.type || 'section'],
            locks: section.locks || {},
            section: normalizeVisualSection(section)
          }
        })
      });
      await loadLibraryData();
      renderPatternList();
      renderLibraryView();
      setStatus('Reusable section saved to content/cms-library/sections/.', 'success');
    } catch (err) {
      setStatus(err.message || 'Unable to save reusable section.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleSectionAction = (action) => {
    const section = getActiveSection();
    if (!section) return;
    if (action === 'up') moveSection(section.id, -1);
    if (action === 'down') moveSection(section.id, 1);
    if (action === 'duplicate') duplicateSection(section.id);
    if (action === 'delete') deleteSection(section.id);
    if (action === 'toggle') {
      if (getSectionLocks(section).lockRemoval) {
        setStatus('This library section has removal locking enabled.', 'error');
        return;
      }
      section.enabled = section.enabled === false;
      renderSectionList();
      schedulePreview();
      updateDirtyState();
    }
    if (action === 'pattern') saveSectionPattern(section.id);
  };

  const handlePageAction = (action) => {
    if (action === 'preview') renderPreview();
    if (action === 'section') {
      setInspectorMode('section');
      renderInspector();
    }
  };

  const handleGlobalAction = async (button) => {
    const action = button.dataset.globalAction;
    const docId = button.dataset.globalDoc;
    if (!docId) return;
    if (action === 'apply-json') {
      applyGlobalJsonEditor(docId);
      return;
    }
    if (action !== 'save') return;
    setBusy(true);
    setStatus(`Saving site/${docId}.json...`);
    try {
      await saveSiteDocument(docId);
      renderInspector();
      if (state.activeView === 'globals') renderGlobalsView();
      updateDirtyState();
      schedulePreview();
      setStatus(`Saved content/site/${docId}.json. It will apply to every generated page after rebuild.`, 'success');
    } catch (err) {
      const message = err.status === 409
        ? `${err.message} Use Refresh to load the latest file before saving.`
        : (err.message || `Unable to save site/${docId}.json.`);
      setStatus(message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const updateSectionField = (target) => {
    const section = getActiveSection();
    if (!section) return;
    const locks = getSectionLocks(section);
    if (!locks.editable) {
      setStatus('This library section is read-only.', 'error');
      return;
    }

    if (target.matches('[data-section-label]')) {
      section.label = target.value;
      renderSectionList();
      updateDirtyState();
      return;
    }

    if (target.matches('[data-section-html]')) {
      section.html = target.value;
      renderSectionList();
      schedulePreview();
      updateDirtyState();
      return;
    }

    const element = sectionElementFromHtml(section.html);
    if (!element) return;

    if (target.matches('[data-section-setting]')) {
      const setting = target.dataset.sectionSetting;
      if (setting === 'id') {
        const value = String(target.value || '').trim();
        if (value) element.setAttribute('id', value);
        else element.removeAttribute('id');
      }
      if (setting === 'class') {
        const value = String(target.value || '').trim();
        if (value) element.setAttribute('class', value);
        else element.removeAttribute('class');
      }
      if (setting === 'aria-label') {
        const value = String(target.value || '').trim();
        if (value) element.setAttribute('aria-label', value);
        else element.removeAttribute('aria-label');
      }
      if (setting === 'spacing') applySectionSpacing(element, target.value);
      if (setting === 'background-color') setElementStyleProperty(element, 'background-color', target.value);
      if (setting === 'background-image') {
        const value = String(target.value || '').trim();
        setElementStyleProperty(element, 'background-image', value ? `url("${value.replace(/"/g, '\\"')}")` : '');
      }
      if (setting === 'min-height') setElementStyleProperty(element, 'min-height', target.value);
      updateSectionHtmlFromElement(section, element);
      renderSectionList();
      schedulePreview();
      updateDirtyState();
      return;
    }

    const type = target.dataset.fieldType;
    const index = Number(target.dataset.fieldIndex);
    if (type === 'text') {
      if (locks.lockText) {
        setStatus('This library section has text locking enabled.', 'error');
        renderInspector();
        return;
      }
      const nodes = collectTextFields(element);
      const field = nodes[index];
      const editable = Array.from(element.querySelectorAll(TEXT_SELECTOR)).filter((node) => node.textContent.trim()).slice(0, 36)[index];
      if (field && editable) editable.textContent = target.value;
    }
    if (type === 'link') {
      const link = Array.from(element.querySelectorAll('a[href]')).slice(0, 24)[index];
      if (link) link.setAttribute('href', target.value);
    }
    if (type === 'image-src') {
      if (locks.lockMedia) {
        setStatus('This library section has media locking enabled.', 'error');
        renderInspector();
        return;
      }
      const image = Array.from(element.querySelectorAll('img')).slice(0, 18)[index];
      if (image) image.setAttribute('src', target.value);
    }
    if (type === 'image-alt') {
      if (locks.lockMedia) {
        setStatus('This library section has media locking enabled.', 'error');
        renderInspector();
        return;
      }
      const image = Array.from(element.querySelectorAll('img')).slice(0, 18)[index];
      if (image) image.setAttribute('alt', target.value);
    }

    updateSectionHtmlFromElement(section, element);
    renderSectionList();
    schedulePreview();
    updateDirtyState();
  };

  const updatePageMetadataField = (target) => {
    if (!target.matches('[data-page-field]')) return false;
    const page = getActivePage();
    if (!page) return true;
    setValueAtPath(page, target.dataset.pageField, target.value);
    setActivePage(page);
    schedulePreview();
    updateDirtyState();
    return true;
  };

  const parseToolPills = (value) => {
    return splitList(value).map((item) => {
      const [label, variant] = item.split(':').map((part) => part.trim()).filter(Boolean);
      const pill = { label: label || item };
      if (variant) pill.variant = variant;
      return pill;
    }).filter((pill) => pill.label);
  };

  const setIndexedObjectField = (fieldName, indexedPath, value) => {
    const [indexRaw, key] = String(indexedPath || '').split('.');
    const index = Number(indexRaw);
    if (!Number.isInteger(index) || index < 0 || !key) return;
    if (!Array.isArray(state.workingDocument[fieldName])) state.workingDocument[fieldName] = [];
    if (!state.workingDocument[fieldName][index] || typeof state.workingDocument[fieldName][index] !== 'object') {
      state.workingDocument[fieldName][index] = {};
    }
    state.workingDocument[fieldName][index][key] = value;
  };

  const cleanupProjectStructuredArrays = () => {
    if (Array.isArray(state.workingDocument.resources)) {
      state.workingDocument.resources = state.workingDocument.resources.filter((resource) => {
        return resource && typeof resource === 'object' && ['label', 'url', 'icon', 'type'].some((key) => String(resource[key] || '').trim());
      });
    }
    if (Array.isArray(state.workingDocument.caseStudy)) {
      state.workingDocument.caseStudy = state.workingDocument.caseStudy.filter((section) => {
        return section && typeof section === 'object' && (
          String(section.title || '').trim()
          || String(section.lead || '').trim()
          || (Array.isArray(section.bullets) && section.bullets.length)
        );
      });
    }
  };

  const updateStructuredDocumentField = (target) => {
    if (!target || !target.matches) return false;
    if (target.matches('[data-document-field]')) {
      setValueAtPath(state.workingDocument, target.dataset.documentField, target.value);
    } else if (target.matches('[data-document-array-field]')) {
      setValueAtPath(state.workingDocument, target.dataset.documentArrayField, splitList(target.value));
    } else if (target.matches('[data-document-number-field]')) {
      const value = toNumberOrEmpty(target.value);
      if (value === '') setValueAtPath(state.workingDocument, target.dataset.documentNumberField, undefined);
      else setValueAtPath(state.workingDocument, target.dataset.documentNumberField, value);
    } else if (target.matches('[data-document-boolean-field]')) {
      setValueAtPath(state.workingDocument, target.dataset.documentBooleanField, !!target.checked);
    } else if (target.matches('[data-project-published-field]')) {
      if (target.checked) delete state.workingDocument.published;
      else state.workingDocument.published = false;
    } else if (target.matches('[data-project-resource-field]')) {
      setIndexedObjectField('resources', target.dataset.projectResourceField, target.value);
    } else if (target.matches('[data-project-case-field]')) {
      setIndexedObjectField('caseStudy', target.dataset.projectCaseField, target.value);
    } else if (target.matches('[data-project-case-bullets]')) {
      const index = Number(target.dataset.projectCaseBullets);
      if (Number.isInteger(index) && index >= 0) {
        if (!Array.isArray(state.workingDocument.caseStudy)) state.workingDocument.caseStudy = [];
        if (!state.workingDocument.caseStudy[index] || typeof state.workingDocument.caseStudy[index] !== 'object') {
          state.workingDocument.caseStudy[index] = {};
        }
        state.workingDocument.caseStudy[index].bullets = splitList(target.value);
      }
    } else if (target.matches('[data-tool-pills]')) {
      state.workingDocument.pills = parseToolPills(target.value);
    } else {
      return false;
    }
    if (state.collection === 'projects' && state.workingDocument.id !== state.id) state.workingDocument.id = state.id;
    if (state.collection === 'tools' && state.workingDocument.slug !== state.id) state.workingDocument.slug = state.id;
    refreshAdvancedEditor();
    schedulePreview();
    updateDirtyState();
    return true;
  };

  const normalizeGlobalFieldValue = (path, value) => {
    if (path === 'portfolio.featuredProjectIds') {
      return String(value || '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
    }
    return value;
  };

  const updateGlobalField = (target) => {
    if (!target.matches('[data-global-field]')) return false;
    const docId = target.dataset.globalDoc;
    const path = target.dataset.globalField;
    const doc = getSiteDocument(docId);
    if (!docId || !path || !doc) return true;
    setValueAtPath(doc, path, normalizeGlobalFieldValue(path, target.value));
    if (state.collection === 'site' && state.id === docId) {
      state.workingDocument = clone(doc);
      elements.editor.value = `${JSON.stringify(state.workingDocument || {}, null, 2)}\n`;
    }
    schedulePreview();
    updateDirtyState();
    return true;
  };

  const applyGlobalJsonEditor = (docId) => {
    const textarea = document.querySelector(`[data-global-json="${docId}"]`);
    if (!textarea) return;
    try {
      const next = JSON.parse(textarea.value || '{}');
      if (!next || typeof next !== 'object' || Array.isArray(next)) {
        throw new Error('Global JSON must be an object');
      }
      const record = getSiteRecord(docId);
      if (record) record.document = next;
      if (state.collection === 'site' && state.id === docId) {
        state.workingDocument = clone(next);
        elements.editor.value = `${JSON.stringify(state.workingDocument || {}, null, 2)}\n`;
      }
      renderInspector();
      if (state.activeView === 'globals') renderGlobalsView();
      schedulePreview();
      updateDirtyState();
      setStatus(`${docId} JSON applied locally. Save to write content/site/${docId}.json.`, 'success');
    } catch (err) {
      setStatus(`Invalid ${docId} JSON: ${err.message}`, 'error');
    }
  };

  const applyJsonEditor = () => {
    try {
      const next = JSON.parse(elements.editor.value || '{}');
      state.workingDocument = next;
      const candidates = getPageCandidates();
      state.pagePath = candidates[0] ? candidates[0].path : '';
      loadVisualFromPage();
      renderAll();
      schedulePreview();
      updateDirtyState();
      setStatus('Advanced JSON applied to the visual editor.', 'success');
    } catch (err) {
      setStatus(`Invalid JSON: ${err.message}`, 'error');
    }
  };

  const appendAssistantMessage = (role, message) => {
    if (!elements.ollamaLog) return;
    const item = document.createElement('p');
    item.className = 'cms-assistant-message';
    item.dataset.role = role;
    item.textContent = message;
    elements.ollamaLog.appendChild(item);
    item.scrollIntoView({ block: 'nearest' });
  };

  const setOllamaModelStatus = (message, stateName = '') => {
    if (!elements.ollamaModelStatus) return;
    elements.ollamaModelStatus.textContent = message || '';
    if (stateName) elements.ollamaModelStatus.dataset.state = stateName;
    else delete elements.ollamaModelStatus.dataset.state;
  };

  const renderOllamaModels = (models, preferredModel = '') => {
    const selected = preferredModel || localStorage.getItem('local-cms-ollama-model') || '';
    elements.ollamaModel.innerHTML = '';
    if (!models.length) {
      const option = document.createElement('option');
      option.value = selected || '';
      option.textContent = selected ? `${selected} (saved)` : 'No installed models found';
      elements.ollamaModel.appendChild(option);
      elements.ollamaModel.value = option.value;
      return;
    }
    models.forEach((model) => {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = model.name;
      elements.ollamaModel.appendChild(option);
    });
    if (selected && !models.some((model) => model.name === selected)) {
      const option = document.createElement('option');
      option.value = selected;
      option.textContent = `${selected} (saved)`;
      elements.ollamaModel.insertBefore(option, elements.ollamaModel.firstChild);
    }
    elements.ollamaModel.value = selected || models[0].name;
    localStorage.setItem('local-cms-ollama-model', elements.ollamaModel.value);
  };

  const formatBytes = (bytes) => {
    const value = Number(bytes) || 0;
    if (!value) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = value;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }
    return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
  };

  const loadOllamaModels = async () => {
    const preferred = localStorage.getItem('local-cms-ollama-model') || '';
    setOllamaModelStatus('Loading installed Ollama models...');
    if (elements.ollamaRefresh) elements.ollamaRefresh.disabled = true;
    try {
      const data = await apiFetch('/ollama-models');
      state.ollamaModels = Array.isArray(data.models) ? data.models : [];
      renderOllamaModels(state.ollamaModels, preferred);
      const selected = state.ollamaModels.find((model) => model.name === elements.ollamaModel.value);
      const size = selected ? formatBytes(selected.size) : '';
      setOllamaModelStatus(
        state.ollamaModels.length
          ? `Loaded ${state.ollamaModels.length} installed model${state.ollamaModels.length === 1 ? '' : 's'}${size ? `. Selected model size: ${size}.` : '.'}`
          : 'Ollama is running, but no installed models were returned.',
        state.ollamaModels.length ? 'success' : ''
      );
    } catch (err) {
      state.ollamaModels = [];
      renderOllamaModels([], preferred);
      setOllamaModelStatus(err.message || 'Unable to load installed Ollama models.', 'error');
    } finally {
      if (elements.ollamaRefresh) elements.ollamaRefresh.disabled = state.busy;
    }
  };

  const updateAssistantControls = () => {
    if (elements.ollamaSend) {
      elements.ollamaSend.disabled = state.busy || state.assistantBusy || !!state.pendingOllamaEdit;
    }
    elements.aiActionButtons.forEach((button) => {
      button.disabled = state.busy || state.assistantBusy || !!state.pendingOllamaEdit;
    });
  };

  const createOllamaReviewSnapshot = () => {
    syncPageFromVisual();
    return {
      collection: state.collection,
      id: state.id,
      pagePath: state.pagePath,
      activeSectionId: state.activeSectionId,
      workingDocument: clone(state.workingDocument),
      visual: state.visual ? clone(state.visual) : null,
      page: clone(getActivePage() || {}),
      visualSections: clone(state.visual && state.visual.sections ? state.visual.sections : []),
      activeSection: clone(getActiveSection() || {}),
      site: {
        settings: clone(getSiteDocument('settings')),
        navigation: clone(getSiteDocument('navigation')),
        footer: clone(getSiteDocument('footer'))
      }
    };
  };

  const getSnapshotActivePage = (snapshot) => getValueAtPath(snapshot.workingDocument, snapshot.pagePath);

  const setSnapshotActivePage = (snapshot, page) => {
    snapshot.workingDocument = setValueAtPath(snapshot.workingDocument, snapshot.pagePath, page);
  };

  const syncSnapshotPageFromVisual = (snapshot) => {
    const page = getSnapshotActivePage(snapshot);
    if (!page || !snapshot.visual) return;
    if (page.template === 'visual-page') {
      page.mainAttributes = snapshot.visual.mainAttributes || { id: 'main' };
      page.sections = snapshot.visual.sections.map((section, index) => ({
        id: section.id || newId('section'),
        type: 'legacy-html',
        label: section.label || `Section ${index + 1}`,
        enabled: section.enabled !== false,
        variant: section.variant || 'default',
        librarySource: section.librarySource || undefined,
        locks: section.locks || undefined,
        props: {
          html: section.html || ''
        }
      }));
      delete page.bodyHtml;
    } else {
      const attrs = attrsToString(snapshot.visual.mainAttributes || { id: 'main' });
      const sections = snapshot.visual.sections
        .filter((section) => section.enabled !== false)
        .map((section) => section.html)
        .join('\n\n');
      page.bodyHtml = `<main${attrs ? ` ${attrs}` : ''}>\n${sections}\n</main>`;
      if (!page.template) page.template = 'raw-body';
    }
    setSnapshotActivePage(snapshot, page);
  };

  const assembleSnapshotBodyHtml = (snapshot, options = {}) => {
    if (!snapshot.visual) return '';
    const attrs = attrsToString(snapshot.visual.mainAttributes || { id: 'main' });
    const sections = snapshot.visual.sections
      .filter((section) => section.enabled !== false)
      .map((section) => options.preview ? annotateSectionHtml(section) : section.html)
      .join('\n\n');
    return `<main${attrs ? ` ${attrs}` : ''}>\n${sections}\n</main>`;
  };

  const getSnapshotPreviewPage = (snapshot) => {
    syncSnapshotPageFromVisual(snapshot);
    const page = clone(getSnapshotActivePage(snapshot) || {});
    if (snapshot.visual) {
      page.template = 'raw-body';
      page.bodyHtml = assembleSnapshotBodyHtml(snapshot, { preview: true });
    }
    return page;
  };

  const getSnapshotSiteOverrides = (snapshot) => ({
    settings: clone(snapshot.site && snapshot.site.settings),
    navigation: clone(snapshot.site && snapshot.site.navigation),
    footer: clone(snapshot.site && snapshot.site.footer)
  });

  const renderSnapshotPreviewHtml = async (snapshot, targetSelector) => {
    const page = getSnapshotActivePage(snapshot);
    if (!page) return previewStatusHtml('No visual preview', 'This Ollama edit did not include a page preview target.');
    const data = await apiFetch('/preview', {
      method: 'POST',
      body: JSON.stringify({
        page: getSnapshotPreviewPage(snapshot),
        site: getSnapshotSiteOverrides(snapshot)
      })
    });
    return decorateReviewPreviewHtml(normalizePreviewHtml(data.html || ''), targetSelector);
  };

  const decorateReviewPreviewHtml = (html, targetSelector) => {
    if (!targetSelector) return html;
    const marker = [
      '<style>',
      '.cms-review-target-highlight{outline:4px solid rgba(47,125,225,.92)!important;outline-offset:5px!important;box-shadow:0 0 0 8px rgba(47,125,225,.16)!important;}',
      '</style>',
      '<script>',
      '(function(){',
      `var selector=${JSON.stringify(targetSelector)};`,
      'function focusTarget(){',
      'var target=document.querySelector(selector);',
      'if(!target)return;',
      'target.classList.add("cms-review-target-highlight");',
      'target.scrollIntoView({block:"center",inline:"nearest"});',
      '}',
      'if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",focusTarget,{once:true});else setTimeout(focusTarget,60);',
      '}());',
      '</script>'
    ].join('');
    return String(html || '').includes('</body>')
      ? String(html).replace('</body>', `${marker}</body>`)
      : `${html}${marker}`;
  };

  const selectorValue = (value) => String(value || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');

  const getSectionReviewTarget = (section) => ({
    type: 'section',
    label: `Section: ${section && section.id ? section.id : 'selected'}${section && section.label ? ` / ${section.label}` : ''}`,
    selector: section && section.id ? `[data-cms-section-id="${selectorValue(section.id)}"]` : 'main',
    visual: true
  });

  const getDefaultReviewTarget = () => ({
    type: 'metadata',
    label: 'Metadata only',
    selector: '',
    visual: false
  });

  const applyOllamaEditsToSnapshot = (snapshot, edits) => {
    if (!edits || typeof edits !== 'object') return { changed: false, target: getDefaultReviewTarget() };
    let changed = false;
    let target = getDefaultReviewTarget();
    const page = getSnapshotActivePage(snapshot);

    if (page && edits.metadata && typeof edits.metadata === 'object' && !Array.isArray(edits.metadata)) {
      Object.entries(edits.metadata).forEach(([key, value]) => {
        page[key] = value;
      });
      setSnapshotActivePage(snapshot, page);
      changed = true;
      target = getDefaultReviewTarget();
    }

    if (page && edits.page && typeof edits.page === 'object' && !Array.isArray(edits.page)) {
      deepMerge(page, edits.page);
      setSnapshotActivePage(snapshot, page);
      changed = true;
      target = {
        type: 'page',
        label: 'Page body',
        selector: 'main',
        visual: true
      };
    }

    if (snapshot.visual && Array.isArray(edits.sections)) {
      snapshot.visual.sections = edits.sections.map((section, index) => normalizeVisualSection(section, index));
      snapshot.activeSectionId = snapshot.visual.sections[0] ? snapshot.visual.sections[0].id : '';
      changed = true;
      target = getSectionReviewTarget(snapshot.visual.sections[0]);
    }

    const sectionPatch = edits.section && typeof edits.section === 'object' && !Array.isArray(edits.section)
      ? edits.section
      : null;
    if (snapshot.visual && sectionPatch) {
      const sectionId = sectionPatch.id || snapshot.activeSectionId;
      const section = snapshot.visual.sections.find((item) => item.id === sectionId) || snapshot.visual.sections[0];
      if (section) {
        ['label', 'type', 'html', 'enabled'].forEach((key) => {
          if (Object.prototype.hasOwnProperty.call(sectionPatch, key)) section[key] = sectionPatch[key];
        });
        Object.assign(section, normalizeVisualSection(section));
        snapshot.activeSectionId = section.id;
        changed = true;
        target = getSectionReviewTarget(section);
      }
    }

    if (edits.navigation && typeof edits.navigation === 'object' && !Array.isArray(edits.navigation)) {
      deepMerge(snapshot.site.navigation, edits.navigation);
      changed = true;
      target = {
        type: 'navigation',
        label: 'Global header',
        selector: '#combined-header-nav',
        visual: true
      };
    }

    if (edits.footer && typeof edits.footer === 'object' && !Array.isArray(edits.footer)) {
      deepMerge(snapshot.site.footer, edits.footer);
      changed = true;
      target = {
        type: 'footer',
        label: 'Global footer',
        selector: '.footer-classic,.speed-dial,.cookie-settings',
        visual: true
      };
    }

    syncSnapshotPageFromVisual(snapshot);
    snapshot.page = clone(getSnapshotActivePage(snapshot) || {});
    snapshot.visualSections = clone(snapshot.visual && snapshot.visual.sections ? snapshot.visual.sections : []);
    snapshot.activeSection = clone(snapshot.visual && snapshot.visual.sections
      ? snapshot.visual.sections.find((section) => section.id === snapshot.activeSectionId) || {}
      : {});

    return { changed, target };
  };

  const getChangedOllamaReview = (before, after) => {
    const beforeReview = {
      page: before.page,
      visualSections: before.visualSections,
      activeSection: before.activeSection,
      navigation: before.site && before.site.navigation,
      footer: before.site && before.site.footer
    };
    const afterReview = {
      page: after.page,
      visualSections: after.visualSections,
      activeSection: after.activeSection,
      navigation: after.site && after.site.navigation,
      footer: after.site && after.site.footer
    };
    return Object.keys(afterReview).reduce((acc, key) => {
      if (JSON.stringify(beforeReview[key]) !== JSON.stringify(afterReview[key])) {
        acc.before[key] = beforeReview[key];
        acc.after[key] = afterReview[key];
      }
      return acc;
    }, { before: {}, after: {} });
  };

  const REVIEW_PREVIEW_VIEWPORTS = {
    desktop: { width: 1280, height: 900 },
    mobile: { width: 390, height: 844 }
  };

  const getReviewPreviewDevice = (value) => Object.prototype.hasOwnProperty.call(REVIEW_PREVIEW_VIEWPORTS, value)
    ? value
    : 'desktop';

  const setAssistantReviewPreviewDevice = (card, deviceName) => {
    if (!card) return;
    const device = getReviewPreviewDevice(deviceName);
    const viewport = REVIEW_PREVIEW_VIEWPORTS[device];
    card.dataset.reviewPreviewDevice = device;
    card.querySelectorAll('[data-ollama-preview-device]').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.ollamaPreviewDevice === device ? 'true' : 'false');
    });
    card.querySelectorAll('.cms-assistant-review-viewport').forEach((container) => {
      const frame = container.querySelector('.cms-assistant-review-frame');
      const availableWidth = Math.max(1, container.clientWidth || container.getBoundingClientRect().width || viewport.width);
      const scale = Math.min(availableWidth / viewport.width, 1);
      const offsetX = Math.max(0, (availableWidth - (viewport.width * scale)) / 2);
      container.style.height = `${Math.ceil(viewport.height * scale)}px`;
      container.style.setProperty('--cms-review-scale', scale.toFixed(4));
      container.style.setProperty('--cms-review-offset-x', `${Math.floor(offsetX)}px`);
      if (frame) {
        frame.style.width = `${viewport.width}px`;
        frame.style.height = `${viewport.height}px`;
      }
    });
  };

  const updateAssistantReviewPreviews = () => {
    if (!elements.ollamaLog) return;
    elements.ollamaLog.querySelectorAll('.cms-assistant-review[data-review-preview-device]').forEach((card) => {
      setAssistantReviewPreviewDevice(card, card.dataset.reviewPreviewDevice);
    });
  };

  const appendAssistantReview = ({ pendingId, reply, target, beforeHtml, afterHtml, details }) => {
    if (!elements.ollamaLog) return;
    const item = document.createElement('article');
    item.className = 'cms-assistant-message cms-assistant-review';
    item.dataset.role = 'assistant';
    item.dataset.ollamaReviewId = pendingId;
    item.dataset.reviewPreviewDevice = 'desktop';

    const heading = document.createElement('h3');
    heading.textContent = 'Review Ollama edit';
    const message = document.createElement('p');
    message.textContent = reply || 'Ollama returned editable changes.';
    const targetLabel = document.createElement('p');
    targetLabel.className = 'cms-assistant-review-target';
    targetLabel.textContent = target && target.label ? target.label : 'Review target';

    const body = document.createElement('div');
    body.className = 'cms-assistant-review-grid';

    if (target && target.visual) {
      const toolbar = document.createElement('div');
      toolbar.className = 'cms-assistant-review-toolbar';
      toolbar.setAttribute('role', 'group');
      toolbar.setAttribute('aria-label', 'Review preview viewport');
      toolbar.innerHTML = [
        '<button class="cms-segment-button" type="button" data-ollama-preview-device="desktop" aria-pressed="true">Desktop</button>',
        '<button class="cms-segment-button" type="button" data-ollama-preview-device="mobile" aria-pressed="false">Mobile</button>'
      ].join('');
      [
        ['Before', beforeHtml],
        ['After', afterHtml]
      ].forEach(([label, html]) => {
        const pane = document.createElement('div');
        pane.className = 'cms-assistant-review-pane';
        const strong = document.createElement('strong');
        strong.textContent = label;
        const iframe = document.createElement('iframe');
        iframe.className = 'cms-assistant-review-frame';
        iframe.title = `${label} Ollama edit preview`;
        iframe.srcdoc = html || previewStatusHtml('Preview unavailable', 'Unable to render this review preview.');
        const viewport = document.createElement('div');
        viewport.className = 'cms-assistant-review-viewport';
        viewport.appendChild(iframe);
        pane.append(strong, viewport);
        body.appendChild(pane);
      });
      item.append(heading, message, targetLabel, toolbar);
    } else {
      const note = document.createElement('p');
      note.className = 'cms-assistant-review-note';
      note.textContent = 'No visible body change was detected. Review the JSON details, then accept or discard the metadata change.';
      body.appendChild(note);
      item.append(heading, message, targetLabel);
    }

    const actions = document.createElement('div');
    actions.className = 'cms-assistant-review-actions';
    actions.innerHTML = [
      '<button class="cms-button" type="button" data-ollama-review-action="accept">Accept Changes</button>',
      '<button class="cms-button cms-button-secondary" type="button" data-ollama-review-action="discard">Discard</button>'
    ].join('');

    const detailsNode = document.createElement('details');
    detailsNode.className = 'cms-assistant-review-details';
    detailsNode.innerHTML = '<summary>Show JSON Details</summary>';
    [
      ['Before', details && details.before],
      ['After', details && details.after]
    ].forEach(([label, value]) => {
      const block = document.createElement('div');
      block.className = 'cms-assistant-review-json';
      const strong = document.createElement('strong');
      strong.textContent = label;
      const pre = document.createElement('pre');
      pre.textContent = `${JSON.stringify(value || {}, null, 2)}\n`;
      block.append(strong, pre);
      detailsNode.appendChild(block);
    });

    item.append(body, actions, detailsNode);
    elements.ollamaLog.appendChild(item);
    requestAnimationFrame(() => setAssistantReviewPreviewDevice(item, 'desktop'));
    item.scrollIntoView({ block: 'nearest' });
  };

  const setSiteDocumentFromSnapshot = (docId, documentBody) => {
    const record = getSiteRecord(docId);
    if (record) record.document = clone(documentBody);
    if (state.collection === 'site' && state.id === docId) {
      state.workingDocument = clone(documentBody);
      refreshAdvancedEditor();
    }
  };

  const acceptPendingOllamaEdit = (reviewId) => {
    if (!state.pendingOllamaEdit || state.pendingOllamaEdit.id !== reviewId) return;
    const snapshot = state.pendingOllamaEdit.after;
    state.workingDocument = clone(snapshot.workingDocument);
    state.visual = snapshot.visual ? clone(snapshot.visual) : null;
    state.activeSectionId = snapshot.activeSectionId || '';
    setSiteDocumentFromSnapshot('settings', snapshot.site.settings);
    setSiteDocumentFromSnapshot('navigation', snapshot.site.navigation);
    setSiteDocumentFromSnapshot('footer', snapshot.site.footer);
    const card = elements.ollamaLog.querySelector(`[data-ollama-review-id="${reviewId}"]`);
    if (card) {
      card.dataset.reviewState = 'accepted';
      card.querySelectorAll('[data-ollama-review-action]').forEach((button) => {
        button.disabled = true;
      });
    }
    state.pendingOllamaEdit = null;
    renderAll();
    refreshAdvancedEditor();
    schedulePreview();
    updateDirtyState();
    updateAssistantControls();
    setStatus('Ollama edit accepted locally. Save to file when ready.', 'success');
  };

  const discardPendingOllamaEdit = (reviewId) => {
    if (!state.pendingOllamaEdit || state.pendingOllamaEdit.id !== reviewId) return;
    const card = elements.ollamaLog.querySelector(`[data-ollama-review-id="${reviewId}"]`);
    if (card) card.remove();
    state.pendingOllamaEdit = null;
    updateAssistantControls();
    setStatus('Ollama edit discarded.', 'success');
  };

  const getOllamaContext = () => {
    syncPageFromVisual();
    return {
      collection: state.collection,
      id: state.id,
      pagePath: state.pagePath,
      inspectorMode: state.inspectorMode,
      activeSectionId: state.activeSectionId,
      page: clone(getActivePage() || {}),
      visual: clone(state.visual || {}),
      activeSection: clone(getActiveSection() || {}),
      site: getPreviewSiteOverrides()
    };
  };

  const applyOllamaEdits = (edits) => {
    if (!edits || typeof edits !== 'object') return false;
    let changed = false;
    const page = getActivePage();

    if (page && edits.metadata && typeof edits.metadata === 'object' && !Array.isArray(edits.metadata)) {
      Object.entries(edits.metadata).forEach(([key, value]) => {
        page[key] = value;
      });
      setActivePage(page);
      changed = true;
    }

    if (page && edits.page && typeof edits.page === 'object' && !Array.isArray(edits.page)) {
      deepMerge(page, edits.page);
      setActivePage(page);
      changed = true;
    }

    if (state.visual && Array.isArray(edits.sections)) {
      state.visual.sections = edits.sections.map((section, index) => normalizeVisualSection(section, index));
      state.activeSectionId = state.visual.sections[0] ? state.visual.sections[0].id : '';
      changed = true;
    }

    const sectionPatch = edits.section && typeof edits.section === 'object' && !Array.isArray(edits.section)
      ? edits.section
      : null;
    if (state.visual && sectionPatch) {
      const sectionId = sectionPatch.id || state.activeSectionId;
      const section = state.visual.sections.find((item) => item.id === sectionId) || getActiveSection();
      if (section) {
        ['label', 'type', 'html', 'enabled'].forEach((key) => {
          if (Object.prototype.hasOwnProperty.call(sectionPatch, key)) section[key] = sectionPatch[key];
        });
        state.activeSectionId = section.id;
        changed = true;
      }
    }

    if (edits.navigation && typeof edits.navigation === 'object' && !Array.isArray(edits.navigation)) {
      deepMerge(getSiteDocument('navigation'), edits.navigation);
      changed = true;
    }

    if (edits.footer && typeof edits.footer === 'object' && !Array.isArray(edits.footer)) {
      deepMerge(getSiteDocument('footer'), edits.footer);
      changed = true;
    }

    if (changed) {
      renderAll();
      schedulePreview();
      updateDirtyState();
    }
    return changed;
  };

  const requestOllamaEdit = async () => {
    const prompt = String(elements.ollamaPrompt.value || '').trim();
    if (!prompt || state.assistantBusy) return;
    if (state.pendingOllamaEdit) {
      setStatus('Accept or discard the pending Ollama edit before requesting another one.', 'error');
      return;
    }
    const model = String(elements.ollamaModel.value || '').trim();
    if (!model) {
      setStatus('Select an installed Ollama model before applying an AI edit.', 'error');
      return;
    }
    const beforeSnapshot = createOllamaReviewSnapshot();
    state.assistantBusy = true;
    elements.ollamaSend.disabled = true;
    appendAssistantMessage('user', prompt);
    setStatus('Sending edit request to local Ollama...');
    try {
      const data = await apiFetch('/ollama', {
        method: 'POST',
        body: JSON.stringify({
          prompt,
          model,
          context: getOllamaContext()
        })
      });
      const reply = data.reply || 'Ollama returned an edit response.';
      const afterSnapshot = clone(beforeSnapshot);
      if (beforeSnapshot.visual) afterSnapshot.visual = clone(beforeSnapshot.visual);
      const result = applyOllamaEditsToSnapshot(afterSnapshot, data.edits || {});
      const changed = result.changed && JSON.stringify(beforeSnapshot) !== JSON.stringify(afterSnapshot);
      if (changed) {
        const review = getChangedOllamaReview(beforeSnapshot, afterSnapshot);
        const reviewId = newId('ollama-review');
        const beforeHtml = result.target.visual
          ? await renderSnapshotPreviewHtml(clone(beforeSnapshot), result.target.selector)
          : '';
        const afterHtml = result.target.visual
          ? await renderSnapshotPreviewHtml(clone(afterSnapshot), result.target.selector)
          : '';
        state.pendingOllamaEdit = {
          id: reviewId,
          reply,
          target: result.target,
          before: beforeSnapshot,
          after: afterSnapshot,
          details: review
        };
        appendAssistantReview({
          pendingId: reviewId,
          reply,
          target: result.target,
          beforeHtml,
          afterHtml,
          details: review
        });
        updateAssistantControls();
      } else {
        appendAssistantMessage('assistant', reply);
      }
      elements.ollamaPrompt.value = '';
      setStatus(changed ? 'Ollama edit ready for review. Accept or discard it before requesting another edit.' : 'Ollama replied without editable changes.', changed ? 'success' : '');
    } catch (err) {
      appendAssistantMessage('error', err.message || 'Unable to reach local Ollama.');
      setStatus(err.message || 'Unable to reach local Ollama.', 'error');
    } finally {
      state.assistantBusy = false;
      updateAssistantControls();
    }
  };

  const setOllamaShortcutPrompt = (action) => {
    const section = getActiveSection();
    const page = getActivePage();
    const prompts = {
      headline: section
        ? `Improve the headline and lead copy in the selected section. Keep the same intent, make it specific to my analytics portfolio, and return only a section edit for section id "${section.id}".`
        : 'Improve the main headline and lead for the current page. Keep it specific to my analytics, BI, tourism intelligence, and data science background.',
      metadata: `Improve the metadata for ${state.collection}/${state.id}. Return metadata fields only: title, description, canonicalPath, ogTitle, ogDescription, robots, and themeColor.`,
      project: state.collection === 'projects'
        ? 'Rewrite this project record for a hiring manager. Tighten the subtitle, problem, actions, results, role, notes, resources labels, demo instructions, and case study sections while preserving factual claims.'
        : 'Tighten this page for a hiring manager. Make the copy more specific, concise, and outcome-focused without inventing new facts.',
      structured: section
        ? `Convert the selected section into cleaner structured HTML while preserving links, images, and meaning. Return a section edit for section id "${section.id}".`
        : 'Convert the most important raw HTML on this page into cleaner structured page content while preserving links, images, and meaning.'
    };
    elements.ollamaPrompt.value = prompts[action] || prompts.headline;
    if (page && state.previewAudience) {
      elements.ollamaPrompt.value += ` Optimize for the "${state.previewAudience}" audience.`;
    }
    switchView('assistant');
    elements.ollamaPrompt.focus();
  };

  const setEmptyLibrary = () => {
    state.library = { templates: [], sections: [], drafts: [] };
    loadPatterns();
    renderPatternList();
    if (state.activeView === 'dashboard') renderDashboard();
    if (state.activeView === 'library') renderLibraryView();
  };

  const loadLibraryData = async (options = {}) => {
    try {
      const data = await apiFetch('/library');
      state.library = data.library || { templates: [], sections: [], drafts: [] };
      loadPatterns();
      renderPatternList();
      if (state.activeView === 'dashboard') renderDashboard();
      if (state.activeView === 'library') renderLibraryView();
      return { ok: true };
    } catch (err) {
      setEmptyLibrary();
      if (options.optional) {
        return {
          ok: false,
          error: err,
          message: err && err.status === 404
            ? 'The CMS library endpoint is not available from this server. Stop the current local CMS server and run start-local-cms-wsl.bat again.'
            : (err.message || 'The CMS library could not be loaded.')
        };
      }
      throw err;
    }
  };

  const loadHealthData = async () => {
    const data = await apiFetch('/health');
    state.health = data.health || null;
    if (state.activeView === 'dashboard') renderDashboard();
    return state.health;
  };

  const loadSnapshotData = async () => {
    const data = await apiFetch('/snapshots?limit=24');
    state.snapshots = Array.isArray(data.snapshots) ? data.snapshots : [];
    if (state.activeView === 'dashboard') renderDashboard();
    return state.snapshots;
  };

  const openPageFromDashboard = (button) => {
    if (!confirmDiscardChanges()) return;
    state.collection = button.dataset.collection || state.collection;
    state.id = button.dataset.id || state.id;
    state.pagePath = button.dataset.pagePath || '';
    loadCurrentRecord();
    switchView('builder');
  };

  const previewPageFromDashboard = (button) => {
    openPageFromDashboard(button);
    renderPreview();
  };

  const duplicateRecordFromDashboard = (button) => {
    const collection = button.dataset.collection || '';
    const id = button.dataset.id || '';
    const record = getCollectionDocs(collection).find((item) => item.id === id);
    if (!record || !confirmDiscardChanges()) return;
    const requested = window.prompt('New document id', `${id}-copy`);
    if (!requested) return;
    const nextId = normalizeSlug(requested, `${id}-copy`);
    const nextDocument = clone(record.document || {});
    if (collection === 'pages') {
      nextDocument.id = nextId;
      nextDocument.outputPath = `pages/${nextId}.html`;
      nextDocument.canonicalPath = `/${nextId}`;
      nextDocument.title = nextDocument.title ? `${nextDocument.title.replace(/\s*\|\s*Daniel Short$/i, '')} Copy | Daniel Short` : 'New Page | Daniel Short';
      if (nextDocument.bodyAttributes) nextDocument.bodyAttributes['data-page'] = nextId;
    }
    if (collection === 'audiences' || collection === 'resumes') nextDocument.key = nextId;
    if (collection === 'projects') {
      nextDocument.id = nextId;
      nextDocument.title = nextDocument.title ? `${nextDocument.title} Copy` : 'Project Copy';
    }
    if (collection === 'tools') {
      nextDocument.slug = nextId;
      nextDocument.href = `tools/${nextId}`;
      nextDocument.title = nextDocument.title ? `${nextDocument.title} Copy` : 'Tool Copy';
    }
    state.collection = collection;
    state.id = nextId;
    state.pagePath = button.dataset.pagePath || '';
    state.workingDocument = nextDocument;
    elements.documentId.value = nextId;
    loadVisualFromPage();
    renderAll();
    refreshAdvancedEditor();
    state.cleanSnapshot = `duplicate:${collection}/${id}:${Date.now()}`;
    updateDirtyState();
    schedulePreview();
    switchView('builder');
    setStatus('Duplicated locally. Review the preview, then save to file.', 'success');
  };

  const applyTemplateToCurrent = (templateId) => {
    if (!confirmDiscardChanges()) return;
    const page = createPageTemplate(templateId);
    state.collection = elements.collection.value || state.collection || 'pages';
    if (state.collection === 'pages') {
      state.workingDocument = page;
      state.pagePath = '';
    } else if (state.pagePath) {
      setActivePage(page);
    } else {
      state.workingDocument = { ...state.workingDocument, page };
      state.pagePath = 'page';
    }
    state.id = page.id || state.id;
    elements.documentId.value = state.id;
    loadVisualFromPage();
    renderAll();
    refreshAdvancedEditor();
    schedulePreview();
    updateDirtyState();
    switchView('builder');
  };

  const startNewPage = (templateId = 'basic-page') => {
    if (!confirmDiscardChanges()) return;
    const requested = window.prompt('Page slug', 'new-page');
    if (!requested) return;
    state.collection = 'pages';
    state.id = normalizeSlug(requested, 'new-page');
    elements.documentId.value = state.id;
    state.workingDocument = createPageTemplate(templateId);
    state.pagePath = '';
    loadVisualFromPage();
    renderAll();
    refreshAdvancedEditor();
    state.cleanSnapshot = `new-page:${Date.now()}`;
    updateDirtyState();
    schedulePreview();
    switchView('builder');
  };

  const saveDraft = async () => {
    syncPageFromVisual();
    const title = window.prompt('Draft name', `${state.collection}/${elements.documentId.value || state.id}`);
    if (!title) return;
    const id = normalizeSlug(title, `draft-${Date.now().toString(36)}`);
    setBusy(true);
    setStatus('Saving local CMS draft...');
    try {
      await apiFetch('/library', {
        method: 'PUT',
        body: JSON.stringify({
          type: 'draft',
          item: {
            id,
            name: title,
            description: `Draft for ${state.collection}/${elements.documentId.value || state.id}`,
            folder: 'Drafts',
            tags: [state.collection],
            source: {
              collection: state.collection,
              id: elements.documentId.value || state.id,
              pagePath: state.pagePath
            },
            pagePath: state.pagePath,
            document: clone(state.workingDocument)
          }
        })
      });
      await loadLibraryData();
      setStatus('Draft saved to content/cms-library/drafts/.', 'success');
    } catch (err) {
      setStatus(err.message || 'Unable to save draft.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const loadDraft = (draftId) => {
    const draft = getLibraryItems('drafts').find((item) => item.id === draftId);
    if (!draft || !draft.document || !confirmDiscardChanges()) return;
    const source = draft.source && typeof draft.source === 'object' ? draft.source : {};
    state.collection = source.collection || state.collection || 'pages';
    state.id = source.id || draft.id || state.id;
    state.pagePath = draft.pagePath || source.pagePath || '';
    state.workingDocument = clone(draft.document);
    elements.documentId.value = state.id;
    loadVisualFromPage();
    renderAll();
    refreshAdvancedEditor();
    state.cleanSnapshot = `draft:${draft.id}:${Date.now()}`;
    updateDirtyState();
    schedulePreview();
    switchView('builder');
    setStatus('Draft loaded locally. Save to file when ready.', 'success');
  };

  const loadSnapshot = async (snapshotId) => {
    if (!snapshotId || !confirmDiscardChanges()) return;
    setBusy(true);
    setStatus('Loading local save snapshot...');
    try {
      const data = await apiFetch('/snapshots', {
        method: 'POST',
        body: JSON.stringify({ snapshotId })
      });
      const snapshot = data.snapshot || {};
      if (!snapshot.document) throw new Error('Snapshot did not include document content.');
      state.collection = snapshot.collection || state.collection;
      state.id = snapshot.documentId || state.id;
      state.pagePath = '';
      state.workingDocument = clone(snapshot.document);
      elements.documentId.value = state.id;
      loadVisualFromPage();
      renderAll();
      refreshAdvancedEditor();
      state.cleanSnapshot = `snapshot:${snapshot.snapshotId}:${Date.now()}`;
      updateDirtyState();
      schedulePreview();
      switchView('builder');
      const changedLines = data.diff && Number(data.diff.changedLines) ? ` Diff versus current file: ${data.diff.changedLines} changed line${data.diff.changedLines === 1 ? '' : 's'}.` : '';
      setStatus(`Snapshot loaded into the editor as unsaved local content.${changedLines} Save to file only after reviewing the preview.`, 'success');
    } catch (err) {
      setStatus(err.message || 'Unable to load local snapshot.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const recoverAutosave = () => {
    const autosave = getAutosave();
    if (!autosave || !autosave.document || !confirmDiscardChanges()) return;
    state.collection = autosave.collection || state.collection || 'pages';
    state.id = autosave.id || state.id;
    state.pagePath = autosave.pagePath || '';
    state.workingDocument = clone(autosave.document);
    elements.documentId.value = state.id;
    loadVisualFromPage();
    renderAll();
    refreshAdvancedEditor();
    state.cleanSnapshot = `autosave:${Date.now()}`;
    updateDirtyState();
    schedulePreview();
    switchView('builder');
    setStatus('Autosave recovered. Review the preview, then save to file.', 'success');
  };

  const insertLibrarySection = (sectionId) => {
    const item = getLibraryItems('sections').find((entry) => entry.id === sectionId);
    if (item && item.section) {
      const section = clone(item.section);
      section.librarySource = {
        id: item.id,
        name: item.name || item.id,
        synced: false,
        insertedAt: new Date().toISOString()
      };
      if (item.locks) section.locks = clone(item.locks);
      addSection(section);
    }
    switchView('builder');
  };

  const handleDashboardAction = (button) => {
    const action = button.dataset.dashboardAction;
    if (action === 'open-page') openPageFromDashboard(button);
    if (action === 'preview-page') previewPageFromDashboard(button);
    if (action === 'duplicate-record') duplicateRecordFromDashboard(button);
    if (action === 'create-template') startNewPage(button.dataset.templateId || 'basic-page');
    if (action === 'load-draft') loadDraft(button.dataset.draftId);
    if (action === 'recover-autosave') recoverAutosave();
    if (action === 'clear-autosave') clearAutosave();
    if (action === 'load-snapshot') loadSnapshot(button.dataset.snapshotId);
  };

  const handleLibraryAction = (button) => {
    const action = button.dataset.libraryAction;
    if (action === 'create-template') startNewPage(button.dataset.templateId || 'basic-page');
    if (action === 'insert-section') insertLibrarySection(button.dataset.sectionId);
    if (action === 'load-draft') loadDraft(button.dataset.draftId);
  };

  const loadContent = async () => {
    setBusy(true);
    setStatus('Loading local content files...');
    try {
      const [contentData, widgetData, mediaData, healthData, snapshotData] = await Promise.all([
        apiFetch('/content'),
        apiFetch('/widgets'),
        apiFetch('/media'),
        apiFetch('/health'),
        apiFetch('/snapshots?limit=24')
      ]);
      state.collections = Array.isArray(contentData.collections) ? contentData.collections : [];
      state.content = contentData.content || {};
      state.widgets = Array.isArray(widgetData.widgets) ? widgetData.widgets : [];
      state.mediaAssets = Array.isArray(mediaData.assets) ? mediaData.assets : [];
      state.health = healthData.health || null;
      state.snapshots = Array.isArray(snapshotData.snapshots) ? snapshotData.snapshots : [];
      markSiteClean();
      const libraryStatus = await loadLibraryData({ optional: true });
      if (!state.collections.some((collection) => collection.name === state.collection)) {
        state.collection = state.collections[0] ? state.collections[0].name : 'pages';
      }
      const docs = getCollectionDocs(state.collection);
      if (docs.length && !docs.some((record) => record.id === state.id)) {
        state.id = docs[0].id;
      }
      loadCurrentRecord();
      elements.workspace.hidden = false;
      elements.accessPanel.hidden = true;
      setStatus(libraryStatus.ok ? 'Local visual CMS loaded.' : libraryStatus.message, libraryStatus.ok ? 'success' : 'error');
    } catch (err) {
      elements.workspace.hidden = true;
      elements.accessPanel.hidden = false;
      elements.accessMessage.textContent = err.message || 'Unable to load local CMS content.';
      setStatus(err.message || 'Unable to load local CMS content.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const bindEvents = () => {
    elements.viewButtons.forEach((button) => {
      button.addEventListener('click', () => switchView(button.dataset.cmsViewTarget));
    });

    elements.dashboardNew.addEventListener('click', () => startNewPage('basic-page'));
    elements.dashboardSearch.addEventListener('input', renderDashboard);
    [
      elements.dashboardPages,
      elements.dashboardTemplates,
      elements.dashboardDrafts,
      elements.dashboardAutosave,
      elements.dashboardHealth,
      elements.dashboardSnapshots
    ].forEach((container) => {
      if (!container) return;
      container.addEventListener('click', (event) => {
        const button = event.target.closest('[data-dashboard-action]');
        if (button) handleDashboardAction(button);
      });
    });

    if (elements.healthRefresh) {
      elements.healthRefresh.addEventListener('click', async () => {
        setBusy(true);
        setStatus('Refreshing CMS health checks...');
        try {
          await Promise.all([loadHealthData(), loadSnapshotData()]);
          setStatus('CMS health checks refreshed.', 'success');
        } catch (err) {
          setStatus(err.message || 'Unable to refresh CMS health checks.', 'error');
        } finally {
          setBusy(false);
        }
      });
    }

    elements.libraryRefresh.addEventListener('click', async () => {
      setBusy(true);
      setStatus('Refreshing CMS library...');
      try {
        await loadLibraryData();
        setStatus('CMS library refreshed.', 'success');
      } catch (err) {
        setStatus(err.message || 'Unable to refresh CMS library.', 'error');
      } finally {
        setBusy(false);
      }
    });

    [
      elements.libraryTemplates,
      elements.librarySections,
      elements.libraryDrafts
    ].forEach((container) => {
      container.addEventListener('click', (event) => {
        const button = event.target.closest('[data-library-action]');
        if (button) handleLibraryAction(button);
      });
    });

    elements.saveDraft.addEventListener('click', saveDraft);
    elements.saveSection.addEventListener('click', () => {
      const section = getActiveSection();
      if (section) saveSectionPattern(section.id);
    });

    [elements.globalHeader, elements.globalFooter].forEach((container) => {
      container.addEventListener('click', (event) => {
        const globalButton = event.target.closest('[data-global-action]');
        if (globalButton) handleGlobalAction(globalButton);
      });
      container.addEventListener('input', (event) => {
        updateGlobalField(event.target);
      });
    });

    elements.collection.addEventListener('change', () => {
      if (!confirmDiscardChanges()) {
        elements.collection.value = state.collection;
        return;
      }
      state.collection = elements.collection.value;
      const docs = getCollectionDocs(state.collection);
      state.id = docs[0] ? docs[0].id : '';
      state.pagePath = '';
      loadCurrentRecord();
    });

    elements.document.addEventListener('change', () => {
      if (!confirmDiscardChanges()) {
        elements.document.value = state.id;
        return;
      }
      state.id = elements.document.value;
      state.pagePath = '';
      loadCurrentRecord();
    });

    elements.pageTarget.addEventListener('change', () => {
      if (!confirmDiscardChanges()) {
        elements.pageTarget.value = state.pagePath;
        return;
      }
      state.pagePath = elements.pageTarget.value;
      loadVisualFromPage();
      renderAll();
      markClean();
      schedulePreview();
    });

    elements.documentId.addEventListener('input', () => {
      state.id = elements.documentId.value.trim();
      elements.editorTitle.textContent = state.id ? `${state.collection}/${state.id}` : 'Page Builder';
      updateDirtyState();
    });

    elements.documentList.addEventListener('click', (event) => {
      const button = event.target.closest('[data-id]');
      if (!button || !confirmDiscardChanges()) return;
      state.id = button.dataset.id;
      state.pagePath = '';
      loadCurrentRecord();
    });

    elements.widgetList.addEventListener('click', (event) => {
      const button = event.target.closest('[data-widget-type]');
      if (!button) return;
      const section = createSectionFromWidget(button.dataset.widgetType);
      if (section) addSection(section);
    });

    elements.patternList.addEventListener('click', (event) => {
      const button = event.target.closest('[data-pattern-id]');
      if (!button) return;
      const pattern = state.patterns.find((item) => item.id === button.dataset.patternId);
      if (pattern && pattern.section) addSection(pattern.section);
    });

    document.querySelectorAll('[data-cms-template]').forEach((button) => {
      button.addEventListener('click', () => {
        applyTemplateToCurrent(button.dataset.cmsTemplate);
      });
    });

    elements.modeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        setInspectorMode(button.dataset.cmsMode);
        renderInspector();
      });
    });

    elements.sectionList.addEventListener('click', (event) => {
      const actionButton = event.target.closest('[data-section-list-action]');
      if (actionButton) {
        event.preventDefault();
        event.stopPropagation();
        const card = actionButton.closest('[data-section-id]');
        if (!card) return;
        state.activeSectionId = card.dataset.sectionId;
        handleSectionAction(actionButton.dataset.sectionListAction);
        return;
      }
      const card = event.target.closest('[data-section-id]');
      if (!card) return;
      state.activeSectionId = card.dataset.sectionId;
      setInspectorMode('section');
      renderSectionList();
      renderInspector();
    });

    elements.sectionList.addEventListener('keydown', (event) => {
      const actionButton = event.target.closest('[data-section-list-action]');
      if (!actionButton || (event.key !== 'Enter' && event.key !== ' ')) return;
      event.preventDefault();
      actionButton.click();
    });

    elements.sectionList.addEventListener('dragstart', (event) => {
      const card = event.target.closest('[data-section-id]');
      if (!card) return;
      state.draggedSectionId = card.dataset.sectionId;
      event.dataTransfer.effectAllowed = 'move';
    });

    elements.sectionList.addEventListener('dragover', (event) => {
      if (!state.draggedSectionId) return;
      event.preventDefault();
    });

    elements.sectionList.addEventListener('drop', (event) => {
      const card = event.target.closest('[data-section-id]');
      if (!card || !state.draggedSectionId || !state.visual) return;
      event.preventDefault();
      const sections = state.visual.sections;
      const from = sections.findIndex((section) => section.id === state.draggedSectionId);
      const to = sections.findIndex((section) => section.id === card.dataset.sectionId);
      if (from < 0 || to < 0 || from === to) return;
      const [section] = sections.splice(from, 1);
      sections.splice(to, 0, section);
      state.draggedSectionId = '';
      renderSectionList();
      schedulePreview();
      updateDirtyState();
    });

    elements.inspector.addEventListener('click', (event) => {
      const mediaPickerButton = event.target.closest('[data-media-picker]');
      if (mediaPickerButton) {
        const field = mediaPickerButton.closest('.cms-media-field');
        renderMediaPicker(field);
        return;
      }
      const mediaOption = event.target.closest('[data-media-path]');
      if (mediaOption) {
        const field = mediaOption.closest('.cms-media-field');
        const input = field && field.querySelector('input,textarea');
        if (input) {
          input.value = mediaOption.dataset.mediaPath || '';
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        const picker = field && field.querySelector('.cms-media-picker');
        if (picker) picker.remove();
        return;
      }
      const structuredButton = event.target.closest('[data-structured-action]');
      if (structuredButton && structuredButton.dataset.structuredAction === 'advanced') {
        elements.advancedPanel.open = true;
        elements.editor.scrollIntoView({ block: 'center', behavior: 'smooth' });
        elements.editor.focus({ preventScroll: true });
        return;
      }
      const widgetButton = event.target.closest('[data-inspector-widget-type]');
      if (widgetButton) {
        const section = createSectionFromWidget(widgetButton.dataset.inspectorWidgetType);
        if (section) addSection(section);
        return;
      }
      const patternButton = event.target.closest('[data-inspector-pattern-id]');
      if (patternButton) {
        insertLibrarySection(patternButton.dataset.inspectorPatternId);
        return;
      }
      const button = event.target.closest('[data-section-action]');
      if (button) handleSectionAction(button.dataset.sectionAction);
      const pageButton = event.target.closest('[data-page-action]');
      if (pageButton) handlePageAction(pageButton.dataset.pageAction);
      const globalButton = event.target.closest('[data-global-action]');
      if (globalButton) handleGlobalAction(globalButton);
    });

    elements.inspector.addEventListener('input', (event) => {
      if (updateStructuredDocumentField(event.target)) return;
      if (updatePageMetadataField(event.target)) return;
      if (updateGlobalField(event.target)) return;
      updateSectionField(event.target);
    });

    elements.inspector.addEventListener('change', (event) => {
      if (updateStructuredDocumentField(event.target)) return;
      if (updatePageMetadataField(event.target)) return;
      if (updateGlobalField(event.target)) return;
      updateSectionField(event.target);
    });

    elements.preview.addEventListener('load', bindPreviewInspector);
    elements.ollamaRefresh.addEventListener('click', loadOllamaModels);
    elements.ollamaModel.addEventListener('change', () => {
      localStorage.setItem('local-cms-ollama-model', elements.ollamaModel.value.trim());
      const selected = state.ollamaModels.find((model) => model.name === elements.ollamaModel.value);
      const size = selected ? formatBytes(selected.size) : '';
      setOllamaModelStatus(selected && size ? `Selected model size: ${size}.` : '');
    });
    elements.ollamaLog.addEventListener('click', (event) => {
      const deviceButton = event.target.closest('[data-ollama-preview-device]');
      if (deviceButton) {
        const card = deviceButton.closest('[data-ollama-review-id]');
        setAssistantReviewPreviewDevice(card, deviceButton.dataset.ollamaPreviewDevice);
        return;
      }
      const button = event.target.closest('[data-ollama-review-action]');
      if (!button) return;
      const card = button.closest('[data-ollama-review-id]');
      const reviewId = card ? card.dataset.ollamaReviewId : '';
      if (button.dataset.ollamaReviewAction === 'accept') acceptPendingOllamaEdit(reviewId);
      if (button.dataset.ollamaReviewAction === 'discard') discardPendingOllamaEdit(reviewId);
    });
    elements.ollamaSend.addEventListener('click', requestOllamaEdit);
    elements.aiActionButtons.forEach((button) => {
      button.addEventListener('click', () => setOllamaShortcutPrompt(button.dataset.cmsAiAction));
    });
    elements.ollamaPrompt.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        requestOllamaEdit();
      }
    });
    elements.previewDeviceButtons.forEach((button) => {
      button.addEventListener('click', () => setPreviewDevice(button.dataset.cmsPreviewDevice));
    });
    if (elements.previewAudience) {
      elements.previewAudience.addEventListener('change', () => {
        state.previewAudience = elements.previewAudience.value || '';
        schedulePreview();
      });
    }
    elements.previewOpen.addEventListener('click', openPreviewWindow);
    window.addEventListener('resize', () => {
      updateAssistantReviewPreviews();
      updateMainPreviewViewport();
    });
    elements.previewRefresh.addEventListener('click', renderPreview);
    elements.applyJson.addEventListener('click', applyJsonEditor);
    elements.newDocument.addEventListener('click', createNewDocument);
    elements.refresh.addEventListener('click', () => {
      if (confirmDiscardChanges()) loadContent();
    });
    elements.save.addEventListener('click', saveDocument);
    elements.export.addEventListener('click', exportContent);
    window.addEventListener('beforeunload', (event) => {
      if (!hasUnsavedChanges()) return;
      event.preventDefault();
      event.returnValue = '';
    });
  };

  const init = async () => {
    elements.user.textContent = 'Local visual editor';
    renderOllamaModels([], localStorage.getItem('local-cms-ollama-model') || '');
    setPreviewDevice('desktop');
    bindEvents();
    await loadContent();
    loadOllamaModels();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
